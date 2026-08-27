from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import UWBData


def _minmax_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    lo = np.min(x, axis=-1, keepdims=True)
    hi = np.max(x, axis=-1, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, eps)


def _resample(x: np.ndarray, old_grid: np.ndarray, new_grid: np.ndarray) -> np.ndarray:
    if x.shape[-1] == new_grid.size and np.allclose(old_grid, new_grid):
        return np.asarray(x, dtype=np.float32)
    flat = np.asarray(x).reshape(-1, x.shape[-1])
    out = np.empty((flat.shape[0], new_grid.size), dtype=np.float32)
    for row in range(flat.shape[0]):
        out[row] = np.interp(new_grid, old_grid, flat[row]).astype(np.float32)
    return out.reshape(*x.shape[:-1], new_grid.size)


@dataclass
class StreamingPreprocessor:
    """Frame-at-a-time preprocessing for low-RAM deployment.

    Static backgrounds are normalized once. `prepare_frame` then allocates only
    the current six-link feature tensor instead of preprocessing an experiment.
    """

    delay_grid_ns: np.ndarray
    input_length: int
    cir_bg_n: np.ndarray
    var_bg_n: np.ndarray
    target_grid_ns: np.ndarray

    @classmethod
    def from_data(cls, data: UWBData, input_length: int | None = None) -> "StreamingPreprocessor":
        length = int(data.delay_grid_ns.size if input_length is None else input_length)
        if length < 1:
            raise ValueError("input_length must be >= 1")
        target_grid = np.linspace(
            float(data.delay_grid_ns[0]), float(data.delay_grid_ns[-1]), length
        )
        cir_bg = _resample(data.cir_background, data.delay_grid_ns, target_grid)
        var_bg = _resample(data.var_background, data.delay_grid_ns, target_grid)
        return cls(
            delay_grid_ns=np.asarray(data.delay_grid_ns, dtype=np.float64),
            input_length=length,
            cir_bg_n=_minmax_rows(np.abs(cir_bg)).astype(np.float32),
            var_bg_n=_minmax_rows(np.maximum(var_bg, 0.0)).astype(np.float32),
            target_grid_ns=target_grid,
        )

    def prepare_frame(self, cir_dynamic: np.ndarray, var_dynamic: np.ndarray) -> np.ndarray:
        """Return `[links, 6, input_length]` float32 features."""
        cir_dynamic = np.asarray(cir_dynamic)
        var_dynamic = np.asarray(var_dynamic)
        expected = (self.cir_bg_n.shape[0], self.delay_grid_ns.size)
        if cir_dynamic.shape != expected or var_dynamic.shape != expected:
            raise ValueError(f"frame arrays must both have shape {expected}")
        if not np.all(np.isfinite(cir_dynamic)) or not np.all(np.isfinite(var_dynamic)):
            raise ValueError("frame arrays must contain only finite values")
        cir = _resample(cir_dynamic, self.delay_grid_ns, self.target_grid_ns)
        var = _resample(var_dynamic, self.delay_grid_ns, self.target_grid_ns)
        cir_n = _minmax_rows(np.abs(cir))
        var_n = _minmax_rows(np.maximum(var, 0.0))
        cir_diff = _minmax_rows(np.abs(cir_n - self.cir_bg_n))
        var_diff = _minmax_rows(np.abs(var_n - self.var_bg_n))
        return np.stack(
            [cir_n, self.cir_bg_n, cir_diff, var_n, self.var_bg_n, var_diff], axis=1
        ).astype(np.float32, copy=False)


def infer_frame(
    model: torch.nn.Module,
    features: np.ndarray,
    delay_max_ns: float,
    scale_multiplier: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run one batched six-link frame without DataLoader/model transfers."""
    dev = torch.device(device)
    with torch.inference_mode():
        out = model(torch.from_numpy(features).to(dev))
        mean = out["mean_fraction"].cpu().numpy() * float(delay_max_ns)
        scale = (
            out["scale_fraction"].cpu().numpy()
            * float(delay_max_ns)
            * float(scale_multiplier)
        )
    return mean.astype(np.float32), np.maximum(scale, 0.05).astype(np.float32)
