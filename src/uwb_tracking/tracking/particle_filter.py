from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ..config import ParticleFilterConfig
from ..data import UWBData


@dataclass
class ParticleFilterDiagnostics:
    runtime_ms_per_update: float
    resample_count: int
    effective_sample_size: np.ndarray


def _student_t_constant(nu: float) -> float:
    return (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * math.log(nu * math.pi)
    )


def _student_t_logpdf(
    residual: np.ndarray,
    scale: np.ndarray | float,
    nu: float,
    constant: float | None = None,
) -> np.ndarray:
    s = np.maximum(np.asarray(scale, dtype=residual.dtype), 1e-4)
    z2 = (residual / s) ** 2
    c = _student_t_constant(nu) if constant is None else constant
    return c - np.log(s) - 0.5 * (nu + 1.0) * np.log1p(z2 / nu)


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = weights.size
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    return np.searchsorted(cumulative, positions, side="left")


def _anchor_distances(
    particle_xy: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    """Compute each particle-to-anchor distance exactly once per update."""
    delta = particle_xy[:, None, :] - anchors[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def run_particle_filter(
    tof_total_hat_ns: np.ndarray,
    predicted_scale_ns: np.ndarray,
    data: UWBData,
    time_s: np.ndarray,
    cfg: ParticleFilterConfig,
    seed: int,
    adaptive: bool,
    global_scale_ns: float,
    c_m_per_ns: float = 0.299792458,
    predicted_outlier_probability: np.ndarray | None = None,
) -> tuple[np.ndarray, ParticleFilterDiagnostics]:
    rng = np.random.default_rng(seed)
    dtype = np.float32 if getattr(cfg, "numeric_dtype", "float64") == "float32" else np.float64
    measurements = np.asarray(tof_total_hat_ns, dtype=dtype)
    scales = np.asarray(predicted_scale_ns, dtype=dtype)
    quality = None
    if predicted_outlier_probability is not None:
        quality = np.asarray(predicted_outlier_probability, dtype=dtype)
        if quality.shape != measurements.shape:
            raise ValueError("predicted_outlier_probability must match measurement shape")
    anchors = np.asarray(data.anchors, dtype=dtype)
    n_time, _ = measurements.shape
    n_particles = int(cfg.num_particles)
    bounds = np.asarray(cfg.bounds_xy, dtype=dtype)

    particles = np.zeros((n_particles, 4), dtype=dtype)
    particles[:, 0] = rng.uniform(bounds[0, 0], bounds[0, 1], n_particles)
    particles[:, 1] = rng.uniform(bounds[1, 0], bounds[1, 1], n_particles)
    weights = np.full(n_particles, 1.0 / n_particles, dtype=dtype)
    estimates = np.zeros((n_time, 2), dtype=dtype)
    ess_history = np.zeros(n_time, dtype=dtype)
    resample_count = 0

    good_constant = _student_t_constant(float(cfg.student_nu))
    broad_constant = _student_t_constant(3.0)
    log_good_prior = math.log(max(1.0 - float(cfg.outlier_prior), 1e-6))
    log_outlier_prior = math.log(max(float(cfg.outlier_prior), 1e-6))
    link_pairs = np.asarray(data.link_pairs, dtype=np.int64)

    started = time.perf_counter()
    for t in range(n_time):
        dt = 0.20 if t == 0 else max(float(time_s[t] - time_s[t - 1]), 1e-4)
        particles[:, 0] += dt * particles[:, 2] + cfg.position_noise_m * rng.normal(size=n_particles)
        particles[:, 1] += dt * particles[:, 3] + cfg.position_noise_m * rng.normal(size=n_particles)
        particles[:, 2] += cfg.velocity_noise_mps * rng.normal(size=n_particles)
        particles[:, 3] += cfg.velocity_noise_mps * rng.normal(size=n_particles)
        particles[:, 0] = np.clip(particles[:, 0], bounds[0, 0], bounds[0, 1])
        particles[:, 1] = np.clip(particles[:, 1], bounds[1, 0], bounds[1, 1])

        # Exact same bistatic path-length math as before, but each of the four
        # anchor distances is computed once and then reused by all six links.
        anchor_dist = _anchor_distances(particles[:, :2], anchors)

        # Stream one link at a time. This preserves the exact likelihood math
        # while avoiding two [particles x links] temporary matrices
        # (expected_all and residual_all), which matters for constrained targets.
        log_w = np.log(np.maximum(weights, np.finfo(dtype).tiny))
        for link, (anchor_i, anchor_j) in enumerate(link_pairs):
            expected = (anchor_dist[:, anchor_i] + anchor_dist[:, anchor_j]) / c_m_per_ns
            residual = measurements[t, link] - expected
            scale = scales[t, link] if adaptive else global_scale_ns
            good = _student_t_logpdf(
                residual, scale, float(cfg.student_nu), constant=good_constant
            )
            if adaptive:
                broad = _student_t_logpdf(
                    residual, cfg.broad_scale_ns, 3.0, constant=broad_constant
                )
                if quality is None:
                    eps = float(cfg.outlier_prior)
                else:
                    # Learned quality augments, rather than replaces, the small
                    # baseline contamination probability. Cap at 0.50 so the
                    # informative Student-t component always remains present.
                    eps = float(
                        np.clip(
                            cfg.outlier_prior
                            + (0.50 - cfg.outlier_prior) * quality[t, link],
                            cfg.outlier_prior,
                            0.50,
                        )
                    )
                log_like = np.logaddexp(
                    math.log(max(1.0 - eps, 1e-6)) + good,
                    math.log(max(eps, 1e-6)) + broad,
                )
            else:
                log_like = good
            log_w += log_like

        log_w -= _logsumexp(log_w)
        weights = np.exp(log_w).astype(dtype, copy=False)
        estimates[t] = np.sum(particles[:, :2] * weights[:, None], axis=0)
        ess = 1.0 / float(np.sum(weights * weights))
        ess_history[t] = ess
        if ess < cfg.resample_fraction * n_particles:
            idx = _systematic_resample(weights, rng)
            particles = particles[idx]
            weights.fill(1.0 / n_particles)
            resample_count += 1

    runtime = 1000.0 * (time.perf_counter() - started) / max(n_time, 1)
    return estimates.astype(np.float64, copy=False), ParticleFilterDiagnostics(
        runtime, resample_count, ess_history.astype(np.float64, copy=False)
    )


def run_repository_particle_filter(
    tof_total_hat_ns: np.ndarray,
    data: UWBData,
    time_s: np.ndarray,
    seed: int,
    error_scale_ns: float,
    error_location_ns: float = 0.0,
    error_nu: float = 4.0,
    num_particles: int = 200,
    velocity_noise_mps: float = 5.0,
    c_m_per_ns: float = 0.299792458,
) -> tuple[np.ndarray, ParticleFilterDiagnostics]:
    """Faithful vectorized port of ParticleFilter4Nodes.m.

    The official code initializes particles in x∈[-5,3], y∈[-5,1], applies
    position-only Gaussian diffusion `dt * 5 * randn`, evaluates one global
    t-location-scale likelihood for all six links, and multinomial-resamples at
    every update. The final sample repeats the preceding estimate because the
    MATLAB loop emits T-1 positions for T timestamps.
    """
    rng = np.random.default_rng(seed)
    measurements = np.asarray(tof_total_hat_ns, dtype=np.float64)
    anchors = np.asarray(data.anchors, dtype=np.float64)
    link_pairs = np.asarray(data.link_pairs, dtype=np.int64)
    n_time, _ = measurements.shape
    if n_time < 2:
        raise ValueError("Repository PF requires at least two timestamps")
    particles = np.empty((num_particles, 2), dtype=np.float64)
    particles[:, 0] = rng.uniform(-5.0, 3.0, num_particles)
    particles[:, 1] = rng.uniform(-5.0, 1.0, num_particles)
    estimates = np.zeros((n_time, 2), dtype=np.float64)
    ess_history = np.zeros(n_time, dtype=np.float64)
    student_constant = _student_t_constant(float(error_nu))
    started = time.perf_counter()

    for t in range(n_time - 1):
        dt = max(float(time_s[t + 1] - time_s[t]), 1e-4)
        updated = particles + dt * velocity_noise_mps * rng.normal(size=particles.shape)
        anchor_dist = _anchor_distances(updated, anchors)
        log_w = np.zeros(num_particles, dtype=np.float64)
        for link, (anchor_i, anchor_j) in enumerate(link_pairs):
            expected = (anchor_dist[:, anchor_i] + anchor_dist[:, anchor_j]) / c_m_per_ns
            residual = measurements[t, link] - expected - error_location_ns
            log_w += _student_t_logpdf(
                residual, error_scale_ns, error_nu, constant=student_constant
            )
        log_w -= _logsumexp(log_w)
        weights = np.exp(log_w)
        ess_history[t] = 1.0 / np.sum(weights**2)
        indices = rng.choice(num_particles, size=num_particles, replace=True, p=weights)
        particles = updated[indices]
        estimates[t] = particles.mean(axis=0)

    estimates[-1] = estimates[-2]
    ess_history[-1] = ess_history[-2]
    runtime = 1000.0 * (time.perf_counter() - started) / max(n_time - 1, 1)
    return estimates, ParticleFilterDiagnostics(runtime, n_time - 1, ess_history)
