from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp

from ..config import ParticleFilterConfig
from ..data import UWBData


@dataclass
class ParticleFilterDiagnostics:
    runtime_ms_per_update: float
    resample_count: int
    effective_sample_size: np.ndarray


def _student_t_logpdf(residual: np.ndarray, scale: np.ndarray | float, nu: float) -> np.ndarray:
    s = np.maximum(np.asarray(scale, dtype=np.float64), 1e-4)
    z2 = (residual / s) ** 2
    return (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - np.log(s)
        - 0.5 * (nu + 1.0) * np.log1p(z2 / nu)
    )


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = weights.size
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    return np.searchsorted(cumulative, positions, side="left")


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
) -> tuple[np.ndarray, ParticleFilterDiagnostics]:
    rng = np.random.default_rng(seed)
    measurements = np.asarray(tof_total_hat_ns, dtype=np.float64)
    scales = np.asarray(predicted_scale_ns, dtype=np.float64)
    n_time, n_links = measurements.shape
    n_particles = cfg.num_particles
    bounds = np.asarray(cfg.bounds_xy, dtype=np.float64)

    particles = np.zeros((n_particles, 4), dtype=np.float64)
    particles[:, 0] = rng.uniform(bounds[0, 0], bounds[0, 1], n_particles)
    particles[:, 1] = rng.uniform(bounds[1, 0], bounds[1, 1], n_particles)
    weights = np.full(n_particles, 1.0 / n_particles)
    estimates = np.zeros((n_time, 2), dtype=np.float64)
    ess_history = np.zeros(n_time, dtype=np.float64)
    resample_count = 0

    started = time.perf_counter()
    for t in range(n_time):
        dt = 0.20 if t == 0 else max(float(time_s[t] - time_s[t - 1]), 1e-4)
        particles[:, 0] += dt * particles[:, 2] + cfg.position_noise_m * rng.normal(size=n_particles)
        particles[:, 1] += dt * particles[:, 3] + cfg.position_noise_m * rng.normal(size=n_particles)
        particles[:, 2] += cfg.velocity_noise_mps * rng.normal(size=n_particles)
        particles[:, 3] += cfg.velocity_noise_mps * rng.normal(size=n_particles)
        particles[:, 0] = np.clip(particles[:, 0], bounds[0, 0], bounds[0, 1])
        particles[:, 1] = np.clip(particles[:, 1], bounds[1, 0], bounds[1, 1])

        log_w = np.log(np.maximum(weights, 1e-300))
        for link, (i, j) in enumerate(data.link_pairs):
            expected = (
                np.linalg.norm(particles[:, :2] - data.anchors[i], axis=1)
                + np.linalg.norm(particles[:, :2] - data.anchors[j], axis=1)
            ) / c_m_per_ns
            residual = measurements[t, link] - expected
            scale = scales[t, link] if adaptive else global_scale_ns
            good = _student_t_logpdf(residual, scale, cfg.student_nu)
            if adaptive:
                broad = _student_t_logpdf(residual, cfg.broad_scale_ns, 3.0)
                log_like = np.logaddexp(
                    np.log(max(1.0 - cfg.outlier_prior, 1e-6)) + good,
                    np.log(max(cfg.outlier_prior, 1e-6)) + broad,
                )
            else:
                log_like = good
            log_w += log_like

        log_w -= logsumexp(log_w)
        weights = np.exp(log_w)
        estimates[t] = np.sum(particles[:, :2] * weights[:, None], axis=0)
        ess = 1.0 / np.sum(weights**2)
        ess_history[t] = ess
        if ess < cfg.resample_fraction * n_particles:
            idx = _systematic_resample(weights, rng)
            particles = particles[idx]
            weights.fill(1.0 / n_particles)
            resample_count += 1

    runtime = 1000.0 * (time.perf_counter() - started) / max(n_time, 1)
    return estimates, ParticleFilterDiagnostics(runtime, resample_count, ess_history)


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
    n_time, n_links = measurements.shape
    if n_time < 2:
        raise ValueError("Repository PF requires at least two timestamps")
    particles = np.empty((num_particles, 2), dtype=np.float64)
    particles[:, 0] = rng.uniform(-5.0, 3.0, num_particles)
    particles[:, 1] = rng.uniform(-5.0, 1.0, num_particles)
    estimates = np.zeros((n_time, 2), dtype=np.float64)
    ess_history = np.zeros(n_time, dtype=np.float64)
    started = time.perf_counter()

    for t in range(n_time - 1):
        dt = max(float(time_s[t + 1] - time_s[t]), 1e-4)
        updated = particles + dt * velocity_noise_mps * rng.normal(size=particles.shape)
        log_w = np.zeros(num_particles, dtype=np.float64)
        for link, (i, j) in enumerate(data.link_pairs):
            expected = (
                np.linalg.norm(updated - data.anchors[i], axis=1)
                + np.linalg.norm(updated - data.anchors[j], axis=1)
            ) / c_m_per_ns
            residual = measurements[t, link] - expected - error_location_ns
            log_w += _student_t_logpdf(residual, error_scale_ns, error_nu)
        log_w -= logsumexp(log_w)
        weights = np.exp(log_w)
        ess_history[t] = 1.0 / np.sum(weights**2)
        indices = rng.choice(num_particles, size=num_particles, replace=True, p=weights)
        particles = updated[indices]
        estimates[t] = particles.mean(axis=0)

    estimates[-1] = estimates[-2]
    ess_history[-1] = ess_history[-2]
    runtime = 1000.0 * (time.perf_counter() - started) / max(n_time - 1, 1)
    return estimates, ParticleFilterDiagnostics(runtime, n_time - 1, ess_history)
