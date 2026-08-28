from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from ..config import ParticleFilterConfig
from ..data import PreparedInputs, UWBData
from ..metrics import tof_metrics, tracking_metrics, uncertainty_metrics
from ..tracking.particle_filter import run_particle_filter
from .exporter import ESP32ExportNet, RawINT8Bundle, raw_int8_inference
from .model import ESP32StudentNet
from .training import student_t_nll


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def decode_raw_outputs(
    raw: np.ndarray,
    delay_max_ns: float,
    min_scale_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the three exported logits into physical deployment outputs."""

    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("raw must have shape [samples, 3]")
    mean_ns = _sigmoid(raw[:, 0]) * float(delay_max_ns)
    scale_ns = (
        np.logaddexp(0.0, raw[:, 1]) + float(min_scale_fraction)
    ) * float(delay_max_ns)
    outlier = _sigmoid(raw[:, 2])
    return (
        mean_ns.astype(np.float32),
        np.maximum(scale_ns, 0.05).astype(np.float32),
        outlier.astype(np.float32),
    )


@torch.inference_mode()
def predict_fp32_raw(
    model: ESP32StudentNet,
    x: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Run the exact BN-folded graph that is later quantized/exported."""

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 6:
        raise ValueError("x must have shape [samples, 6, input_length]")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    folded = ESP32ExportNet(model.cpu().eval()).eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        chunks.append(folded(torch.from_numpy(x[start : start + batch_size])).numpy())
    if not chunks:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def predict_int8_raw(
    bundle: RawINT8Bundle,
    x: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Run the integer-reference path in bounded batches to limit host RAM."""

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 6:
        raise ValueError("x must have shape [samples, 6, input_length]")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    chunks: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        chunks.append(raw_int8_inference(bundle, x[start : start + batch_size]))
    if not chunks:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def deployment_point_metrics(
    raw: np.ndarray,
    prepared: PreparedInputs,
    delay_max_ns: float,
    min_scale_fraction: float,
    student_nu: float = 4.0,
) -> dict[str, float]:
    """ToF/uncertainty metrics directly on an exported model's raw outputs."""

    mean_ns, scale_ns, outlier = decode_raw_outputs(raw, delay_max_ns, min_scale_fraction)
    if mean_ns.size != prepared.target_delay_ns.size:
        raise ValueError("prediction count does not match prepared targets")
    result = tof_metrics(mean_ns, prepared.target_delay_ns)

    mean_fraction = np.clip(mean_ns / float(delay_max_ns), 0.0, 1.0).astype(np.float32)
    scale_fraction = np.maximum(scale_ns / float(delay_max_ns), 1e-8).astype(np.float32)
    target_fraction = prepared.target_fraction.astype(np.float32, copy=False)
    with torch.inference_mode():
        nll = student_t_nll(
            torch.from_numpy(target_fraction),
            torch.from_numpy(mean_fraction),
            torch.from_numpy(scale_fraction),
            float(student_nu),
        ).mean()
    corr = prepared.corruption.astype(np.float64, copy=False)
    prob = np.clip(outlier.astype(np.float64), 1e-7, 1.0 - 1e-7)
    bce = -np.mean(corr * np.log(prob) + (1.0 - corr) * np.log(1.0 - prob))
    result.update(
        {
            "nll": float(nll),
            "outlier_bce": float(bce),
            **uncertainty_metrics(
                mean_ns,
                prepared.target_delay_ns,
                scale_ns,
                prepared.corruption,
                outlier_probability=outlier,
            ),
        }
    )
    return result


def _reshape_time_link(values: np.ndarray, prepared: PreparedInputs, n_time: int, n_links: int) -> np.ndarray:
    values = np.asarray(values)
    if values.size != prepared.sample_time.size:
        raise ValueError("values must have one entry per prepared sample")
    out = np.empty((n_time, n_links), dtype=values.dtype)
    out[prepared.sample_time, prepared.sample_link] = values
    return out


def deployment_tracking_metrics(
    raw: np.ndarray,
    prepared: PreparedInputs,
    data: UWBData,
    true_xy: np.ndarray,
    time_s: np.ndarray,
    pf_cfg: ParticleFilterConfig,
    seed: int,
    delay_max_ns: float,
    min_scale_fraction: float,
    student_nu: float = 4.0,
) -> dict[str, float | int | str]:
    """Evaluate ToF + complete uncertainty-aware PF tracking for one scenario."""

    n_time = int(np.asarray(time_s).size)
    n_links = data.num_links
    mean_ns, scale_ns, outlier = decode_raw_outputs(raw, delay_max_ns, min_scale_fraction)
    mean_2d = _reshape_time_link(mean_ns, prepared, n_time, n_links)
    scale_2d = _reshape_time_link(scale_ns, prepared, n_time, n_links)
    outlier_2d = _reshape_time_link(outlier, prepared, n_time, n_links)
    total_tof = mean_2d + data.tof_los_ns[None, :]

    estimate, diag = run_particle_filter(
        total_tof,
        scale_2d,
        data,
        np.asarray(time_s, dtype=np.float64),
        pf_cfg,
        seed=seed,
        adaptive=True,
        global_scale_ns=float(np.median(scale_2d)),
        predicted_outlier_probability=outlier_2d,
    )
    result: dict[str, float | int | str] = {}
    result.update(deployment_point_metrics(raw, prepared, delay_max_ns, min_scale_fraction, student_nu))
    result.update(tracking_metrics(estimate, true_xy))
    result.update(
        {
            "pf_runtime_ms_per_update": float(diag.runtime_ms_per_update),
            "pf_resample_count": int(diag.resample_count),
            "pf_mean_ess": float(np.mean(diag.effective_sample_size)),
        }
    )
    return result


def particle_filter_config_from_mapping(raw: dict | None) -> ParticleFilterConfig:
    cfg = ParticleFilterConfig()
    for key, value in (raw or {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    if cfg.num_particles < 2:
        raise ValueError("deployment PF must use at least two particles")
    return cfg


def numeric_metrics(metrics: dict[str, object], keys: Iterable[str]) -> dict[str, float]:
    """Small helper used when writing compact quality-guard records."""

    return {key: float(metrics[key]) for key in keys if key in metrics}
