#!/usr/bin/env python3
"""Convert the official MATLAB repository files into this project's standard MAT format.

Expected inputs from CLongLi/UWB-Radar-Pedestrian-Tracking:
  - Bg_CIR_VAR.mat
  - Dyn_CIR_VAR.mat
  - AnchorPos.mat

The official dynamic file is not bundled in the GitHub repository. Obtain it from
its author-provided link, then run this script locally.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

LINKS = ("01", "02", "04", "12", "14", "24")
PAIRS = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)


def merged(*paths: Path) -> dict:
    out: dict = {}
    for path in paths:
        out.update({k: v for k, v in loadmat(path, squeeze_me=True).items() if not k.startswith("__")})
    return out


def find_key(raw: dict, candidates: list[str]) -> str:
    lower = {k.lower(): k for k in raw}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise KeyError(f"Missing variable. Tried {candidates}. Available examples: {list(raw)[:30]}")


def get(raw: dict, candidates: list[str]) -> np.ndarray:
    return np.asarray(raw[find_key(raw, candidates)])


def as_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x if x.shape[0] >= x.shape[1] or x.shape[1] > 100 else x.T


def interp_rows(time_old: np.ndarray, values: np.ndarray, time_new: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[0] != time_old.size and values.shape[1] == time_old.size:
        values = values.T
    result = np.empty((time_new.size, values.shape[1]), dtype=np.float32)
    for col in range(values.shape[1]):
        result[:, col] = np.interp(time_new, time_old, values[:, col])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", required=True, type=Path)
    parser.add_argument("--dynamic", required=True, type=Path)
    parser.add_argument("--anchors", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/uwb_original_standard.mat"), type=Path)
    args = parser.parse_args()

    raw = merged(args.background, args.dynamic, args.anchors)
    anchors = get(raw, ["AnchorPos", "anchors"]).astype(float)
    if anchors.shape == (2, 4):
        anchors = anchors.T
    delay_grid = get(raw, ["re_SampTime", "delay_grid_ns"]).reshape(-1).astype(float)

    times, cirs, variances, mus, total_tofs = [], [], [], [], []
    cir_bg, var_bg, los = [], [], []
    for link in LINKS:
        time = get(raw, [f"Dyn_re_tUWB{link}"]).reshape(-1).astype(float)
        cir = get(raw, [f"Dyn_re_CIR{link}"]).astype(float)
        var = get(raw, [f"Dyn_re_VAR{link}", f"Dyn_re_CIRVar{link}", f"Dyn_re_Var{link}"]).astype(float)
        mu = get(raw, [f"Dyn_re_MU{link}"]).astype(float)
        tof = get(raw, [f"Dyn_real_ToF{link}"]).reshape(-1).astype(float)
        times.append(time)
        cirs.append(cir)
        variances.append(var)
        mus.append(mu)
        total_tofs.append(tof)
        cir_bg.append(get(raw, [f"Bg_re_CIR{link}"]).reshape(-1))
        var_bg.append(get(raw, [f"Bg_re_VAR{link}", f"Bg_re_CIRVar{link}", f"Bg_re_Var{link}"]).reshape(-1))
        los.append(float(get(raw, [f"ToF_TRx{link}"]).reshape(-1)[0]))

    start = max(t.min() for t in times)
    end = min(t.max() for t in times)
    dt = float(np.median(np.concatenate([np.diff(t) for t in times])))
    common_time = np.arange(start, end + 0.5 * dt, dt)
    t_count, l_count, b_count = common_time.size, len(LINKS), delay_grid.size
    cir_dynamic = np.empty((t_count, l_count, b_count), dtype=np.float32)
    var_dynamic = np.empty_like(cir_dynamic)
    tof_total = np.empty((t_count, l_count), dtype=np.float64)
    xy_per_link = []
    for l in range(l_count):
        cir_dynamic[:, l] = interp_rows(times[l], cirs[l], common_time)
        var_dynamic[:, l] = interp_rows(times[l], variances[l], common_time)
        tof_total[:, l] = np.interp(common_time, times[l], total_tofs[l])
        mu = mus[l]
        if mu.shape[0] != times[l].size and mu.shape[1] == times[l].size:
            mu = mu.T
        xy_per_link.append(interp_rows(times[l], mu[:, :2], common_time))
    trajectory_xy = np.mean(np.stack(xy_per_link), axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        args.output,
        {
            "anchors": anchors,
            "link_pairs": PAIRS + 1,
            "time_s": common_time,
            "delay_grid_ns": delay_grid,
            "trajectory_xy": trajectory_xy,
            "tof_total_ns": tof_total,
            "tof_los_ns": np.asarray(los),
            "cir_background": np.stack(cir_bg).astype(np.float32),
            "var_background": np.stack(var_bg).astype(np.float32),
            "cir_dynamic": cir_dynamic,
            "var_dynamic": var_dynamic,
            "description": "Converted from the official UWB Radar Pedestrian Tracking MATLAB dataset.",
        },
        do_compression=True,
    )
    print(f"Wrote {args.output} with T={t_count}, L={l_count}, B={b_count}")


if __name__ == "__main__":
    main()
