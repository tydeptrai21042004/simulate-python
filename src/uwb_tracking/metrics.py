from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def tracking_metrics(estimated_xy: np.ndarray, true_xy: np.ndarray) -> dict[str, float]:
    error_m = np.linalg.norm(np.asarray(estimated_xy) - np.asarray(true_xy), axis=1)
    return {
        "tracking_rmse_cm": float(100.0 * np.sqrt(np.mean(error_m**2))),
        "tracking_mae_cm": float(100.0 * np.mean(error_m)),
        "tracking_median_cm": float(100.0 * np.median(error_m)),
        "tracking_p90_cm": float(100.0 * np.percentile(error_m, 90)),
    }


def tof_metrics(pred_delay_ns: np.ndarray, true_delay_ns: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred_delay_ns) - np.asarray(true_delay_ns)
    return {
        "tof_mae_ns": float(np.mean(np.abs(err))),
        "tof_rmse_ns": float(np.sqrt(np.mean(err**2))),
        "tof_median_ae_ns": float(np.median(np.abs(err))),
        "tof_p90_ae_ns": float(np.percentile(np.abs(err), 90)),
    }


def uncertainty_metrics(
    pred_delay_ns: np.ndarray,
    true_delay_ns: np.ndarray,
    scale_ns: np.ndarray,
    corruption_mask: np.ndarray,
    tolerance_ns: float = 1.0,
    bins: int = 10,
    outlier_probability: np.ndarray | None = None,
) -> dict[str, float]:
    err = np.abs(np.asarray(pred_delay_ns) - np.asarray(true_delay_ns))
    scale = np.maximum(np.asarray(scale_ns), 1e-4)
    confidence = 1.0 / (1.0 + scale)
    correctness = (err <= tolerance_ns).astype(float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (confidence >= left) & (confidence < right if right < 1 else confidence <= right)
        if np.any(in_bin):
            ece += np.mean(in_bin) * abs(float(np.mean(confidence[in_bin])) - float(np.mean(correctness[in_bin])))
    brier = float(np.mean((confidence - correctness) ** 2))
    result = {"confidence_ece": float(ece), "confidence_brier": brier}
    mask = np.asarray(corruption_mask).astype(int)
    mask_flat = mask.reshape(-1)
    if np.unique(mask_flat).size == 2:
        unreliability = np.asarray(scale, dtype=float).reshape(-1)
        if unreliability.size != mask_flat.size:
            raise ValueError("scale_ns must match corruption_mask size")
        result["corruption_auroc"] = float(roc_auc_score(mask_flat, unreliability))
        result["corruption_auprc"] = float(average_precision_score(mask_flat, unreliability))
    else:
        result["corruption_auroc"] = float("nan")
        result["corruption_auprc"] = float("nan")
    if outlier_probability is not None:
        probability = np.clip(np.asarray(outlier_probability, dtype=float).reshape(-1), 0.0, 1.0)
        if probability.size != mask_flat.size:
            raise ValueError("outlier_probability must match corruption_mask size")
        result["outlier_probability_brier"] = float(np.mean((probability - mask_flat) ** 2))
        if np.unique(mask_flat).size == 2:
            result["outlier_probability_auroc"] = float(roc_auc_score(mask_flat, probability))
            result["outlier_probability_auprc"] = float(average_precision_score(mask_flat, probability))
        else:
            result["outlier_probability_auroc"] = float("nan")
            result["outlier_probability_auprc"] = float("nan")
    return result
