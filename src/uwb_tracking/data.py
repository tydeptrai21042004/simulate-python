from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.io import loadmat, savemat


@dataclass
class UWBData:
    anchors: np.ndarray
    link_pairs: np.ndarray
    time_s: np.ndarray
    delay_grid_ns: np.ndarray
    trajectory_xy: np.ndarray
    tof_total_ns: np.ndarray
    tof_los_ns: np.ndarray
    cir_background: np.ndarray
    var_background: np.ndarray
    cir_dynamic: np.ndarray
    var_dynamic: np.ndarray
    description: str = ""

    @property
    def num_time(self) -> int:
        return int(self.time_s.shape[0])

    @property
    def num_links(self) -> int:
        return int(self.link_pairs.shape[0])

    @property
    def excess_tof_ns(self) -> np.ndarray:
        return np.maximum(self.tof_total_ns - self.tof_los_ns[None, :], 0.0)

    def validate(self, c_m_per_ns: float = 0.299792458) -> None:
        t, l, b = self.cir_dynamic.shape
        assert self.var_dynamic.shape == (t, l, b)
        assert self.cir_background.shape == (l, b)
        assert self.var_background.shape == (l, b)
        assert self.tof_total_ns.shape == (t, l)
        assert self.trajectory_xy.shape == (t, 2)
        assert self.anchors.ndim == 2 and self.anchors.shape[1] == 2
        assert self.link_pairs.min() >= 0
        expected = geometry_tof(self.trajectory_xy, self.anchors, self.link_pairs, c_m_per_ns)
        max_error = float(np.max(np.abs(expected - self.tof_total_ns)))
        if max_error > 1e-5:
            raise ValueError(f"Geometry/ToF mismatch: max error={max_error:.6g} ns")


@dataclass
class ObservationSet:
    time_indices: np.ndarray
    cir_dynamic: np.ndarray
    var_dynamic: np.ndarray
    corruption_mask: np.ndarray
    true_tof_ns: np.ndarray
    true_xy: np.ndarray
    time_s: np.ndarray


def load_uwb_mat(path: str | Path) -> UWBData:
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    pairs = np.asarray(raw["link_pairs"], dtype=np.int64)
    if pairs.min() >= 1:
        pairs = pairs - 1
    data = UWBData(
        anchors=np.asarray(raw["anchors"], dtype=np.float64),
        link_pairs=pairs,
        time_s=np.asarray(raw["time_s"], dtype=np.float64).reshape(-1),
        delay_grid_ns=np.asarray(raw["delay_grid_ns"], dtype=np.float64).reshape(-1),
        trajectory_xy=np.asarray(raw["trajectory_xy"], dtype=np.float64),
        tof_total_ns=np.asarray(raw["tof_total_ns"], dtype=np.float64),
        tof_los_ns=np.asarray(raw["tof_los_ns"], dtype=np.float64).reshape(-1),
        cir_background=np.asarray(raw["cir_background"], dtype=np.float32),
        var_background=np.asarray(raw["var_background"], dtype=np.float32),
        cir_dynamic=np.asarray(raw["cir_dynamic"], dtype=np.float32),
        var_dynamic=np.asarray(raw["var_dynamic"], dtype=np.float32),
        description=str(raw.get("description", "")),
    )
    data.validate()
    return data


def geometry_tof(
    points_xy: np.ndarray,
    anchors: np.ndarray,
    link_pairs: np.ndarray,
    c_m_per_ns: float = 0.299792458,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    result = np.empty((points.shape[0], link_pairs.shape[0]), dtype=np.float64)
    for link, (i, j) in enumerate(link_pairs):
        result[:, link] = (
            np.linalg.norm(points - anchors[i], axis=1)
            + np.linalg.norm(points - anchors[j], axis=1)
        ) / c_m_per_ns
    return result


def get_case_split(num_samples: int, case_id: int) -> tuple[np.ndarray, np.ndarray]:
    n1 = num_samples // 3
    n2 = (2 * num_samples) // 3
    if case_id == 1:
        return np.arange(0, n2), np.arange(n2, num_samples)
    if case_id == 2:
        return np.r_[0:n1, n2:num_samples], np.arange(n1, n2)
    if case_id == 3:
        return np.arange(n1, num_samples), np.arange(0, n1)
    raise ValueError("case_id must be 1, 2, or 3")


def subset_observations(data: UWBData, indices: np.ndarray) -> ObservationSet:
    idx = np.asarray(indices, dtype=np.int64)
    return ObservationSet(
        time_indices=idx,
        cir_dynamic=data.cir_dynamic[idx].copy(),
        var_dynamic=data.var_dynamic[idx].copy(),
        corruption_mask=np.zeros((idx.size, data.num_links), dtype=bool),
        true_tof_ns=data.tof_total_ns[idx].copy(),
        true_xy=data.trajectory_xy[idx].copy(),
        time_s=data.time_s[idx].copy(),
    )


def _minmax_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    lo = np.min(x, axis=-1, keepdims=True)
    hi = np.max(x, axis=-1, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, eps)


def _resample_last_axis(x: np.ndarray, old_grid: np.ndarray, length: int) -> np.ndarray:
    if int(length) < 1:
        raise ValueError("length must be >= 1")
    if x.shape[-1] != np.asarray(old_grid).size:
        raise ValueError("old_grid length must match the last axis of x")
    new_grid = np.linspace(float(old_grid[0]), float(old_grid[-1]), int(length))
    flat = x.reshape(-1, x.shape[-1])
    out = np.empty((flat.shape[0], length), dtype=np.float32)
    for row in range(flat.shape[0]):
        out[row] = np.interp(new_grid, old_grid, flat[row]).astype(np.float32)
    return out.reshape(*x.shape[:-1], length)


@dataclass
class PreparedInputs:
    paper_cir: np.ndarray
    paper_var: np.ndarray
    fusion: np.ndarray
    target_fraction: np.ndarray
    target_index: np.ndarray
    target_delay_ns: np.ndarray
    sample_time: np.ndarray
    sample_link: np.ndarray
    corruption: np.ndarray


def prepare_inputs(
    data: UWBData,
    obs: ObservationSet,
    input_length: int = 500,
) -> PreparedInputs:
    if int(input_length) < 1:
        raise ValueError("input_length must be >= 1")
    n, l, _ = obs.cir_dynamic.shape
    if n < 1:
        raise ValueError("observation set must contain at least one timestamp")
    if l != data.num_links:
        raise ValueError("observation link count does not match dataset geometry")
    cir_dyn = _resample_last_axis(obs.cir_dynamic, data.delay_grid_ns, input_length)
    var_dyn = _resample_last_axis(obs.var_dynamic, data.delay_grid_ns, input_length)
    cir_bg = _resample_last_axis(data.cir_background, data.delay_grid_ns, input_length)
    var_bg = _resample_last_axis(data.var_background, data.delay_grid_ns, input_length)
    cir_bg_t = np.broadcast_to(cir_bg[None, :, :], cir_dyn.shape)
    var_bg_t = np.broadcast_to(var_bg[None, :, :], var_dyn.shape)

    cir_dyn_n = _minmax_rows(np.abs(cir_dyn))
    cir_bg_n = _minmax_rows(np.abs(cir_bg_t))
    var_dyn_n = _minmax_rows(np.maximum(var_dyn, 0.0))
    var_bg_n = _minmax_rows(np.maximum(var_bg_t, 0.0))
    cir_diff_n = _minmax_rows(np.abs(cir_dyn_n - cir_bg_n))
    var_diff_n = _minmax_rows(np.abs(var_dyn_n - var_bg_n))

    paper_cir = np.stack([cir_dyn_n, cir_bg_n], axis=-1)[:, :, None, :, :]
    paper_var = np.stack([var_dyn_n, var_bg_n], axis=-1)[:, :, None, :, :]
    fusion = np.stack(
        [cir_dyn_n, cir_bg_n, cir_diff_n, var_dyn_n, var_bg_n, var_diff_n],
        axis=2,
    )

    target_delay = np.maximum(obs.true_tof_ns - data.tof_los_ns[None, :], 0.0)
    delay_start = float(data.delay_grid_ns[0])
    delay_end = float(data.delay_grid_ns[-1])
    delay_max = max(delay_end, 1e-8)
    target_fraction = np.clip(target_delay / delay_max, 0.0, 1.0)
    resampled_grid = np.linspace(delay_start, delay_end, input_length)
    if input_length > 1:
        step = float(resampled_grid[1] - resampled_grid[0])
        target_index = np.rint((target_delay - delay_start) / step).astype(np.int64) + 1
    else:
        target_index = np.ones_like(target_delay, dtype=np.int64)
    target_index = np.clip(target_index, 1, input_length).astype(np.float32)

    sample_time = np.repeat(np.arange(n), l)
    sample_link = np.tile(np.arange(l), n)
    return PreparedInputs(
        paper_cir=paper_cir.reshape(n * l, 1, input_length, 2).astype(np.float32),
        paper_var=paper_var.reshape(n * l, 1, input_length, 2).astype(np.float32),
        fusion=fusion.reshape(n * l, 6, input_length).astype(np.float32),
        target_fraction=target_fraction.reshape(-1).astype(np.float32),
        target_index=target_index.reshape(-1).astype(np.float32),
        target_delay_ns=target_delay.reshape(-1).astype(np.float32),
        sample_time=sample_time,
        sample_link=sample_link,
        corruption=obs.corruption_mask.reshape(-1).astype(np.float32),
    )


def export_standard_mat(data: UWBData, path: str | Path) -> None:
    savemat(
        path,
        {
            "anchors": data.anchors,
            "link_pairs": data.link_pairs + 1,
            "time_s": data.time_s,
            "delay_grid_ns": data.delay_grid_ns,
            "trajectory_xy": data.trajectory_xy,
            "tof_total_ns": data.tof_total_ns,
            "tof_los_ns": data.tof_los_ns,
            "cir_background": data.cir_background,
            "var_background": data.var_background,
            "cir_dynamic": data.cir_dynamic,
            "var_dynamic": data.var_dynamic,
            "description": data.description,
        },
        do_compression=True,
    )
