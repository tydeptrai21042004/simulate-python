#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwb_tracking.data import load_uwb_mat, prepare_inputs, subset_observations
from uwb_tracking.simulation import augment_training_observations
from uwb_tracking.official_data import ensure_official_standard_dataset
from uwb_tracking.esp32.exporter import export_checkpoint
from uwb_tracking.esp32.preprocess_export import export_preprocess_and_geometry
from uwb_tracking.esp32.training import load_student_checkpoint, is_structured_lth_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="Export a trained structured-LTH ESP32 student.")
    p.add_argument("--checkpoint", default="results/esp32s3/checkpoints/best_student.pt")
    p.add_argument("--data", default="data/uwb_demo_input.mat")
    p.add_argument("--output", default="results/esp32s3/export")
    p.add_argument("--target", default="esp32s3", choices=["esp32", "esp32s3", "esp32p4", "c"])
    p.add_argument("--calibration-samples", type=int, default=256)
    p.add_argument("--auto-data", action="store_true", help="Fetch official data if --data is missing.")
    p.add_argument(
        "--source-dir",
        default="data/original_uwb/UWB-Radar-Pedestrian-Tracking",
        help="Cache directory for the official GitHub/Google-Drive source files.",
    )
    p.add_argument(
        "--allow-non-lth",
        action="store_true",
        help="Debug-only override. By default direct export rejects random/non-LTH checkpoints.",
    )
    p.add_argument("--onnx", action="store_true")
    p.add_argument("--espdl", action="store_true")
    args = p.parse_args()

    checkpoint = ROOT / args.checkpoint
    _, meta = load_student_checkpoint(checkpoint)
    if not is_structured_lth_checkpoint(meta) and not args.allow_non_lth:
        role = (meta.get("extra", {}) or {}).get("role", "unknown")
        raise RuntimeError(
            f"Refusing to export non-LTH checkpoint (role={role!r}). "
            "Use best_student.pt from train_esp32_pipeline.py. "
            "Pass --allow-non-lth only for an explicit debugging/control export."
        )

    data_path = ROOT / args.data
    if not data_path.exists() and args.auto_data:
        data_path = ensure_official_standard_dataset(
            data_path,
            ROOT / args.source_dir,
        )
    data = load_uwb_mat(data_path)

    rng = np.random.default_rng(20260827)
    n_time = min(data.num_time, max(24, args.calibration_samples // max(data.num_links, 1)))
    time_idx = np.sort(rng.choice(data.num_time, size=n_time, replace=False))
    clean_obs = subset_observations(data, time_idx)
    aug_obs = augment_training_observations(data, time_idx, seed=20260827, probability=0.35)
    clean = prepare_inputs(data, clean_obs, int(meta["input_length"]))
    aug = prepare_inputs(data, aug_obs, int(meta["input_length"]))
    pool = np.concatenate([clean.fusion, aug.fusion], axis=0)
    count = min(args.calibration_samples, len(pool))
    calibration = pool[rng.choice(len(pool), size=count, replace=False)]

    report = export_checkpoint(
        checkpoint,
        calibration,
        ROOT / args.output,
        target=args.target,
        export_onnx_file=args.onnx or args.espdl,
        export_espdl_file=args.espdl,
    )
    report["checkpoint_policy"] = {
        "lth_required": not args.allow_non_lth,
        "checkpoint_is_lth": is_structured_lth_checkpoint(meta),
        "role": (meta.get("extra", {}) or {}).get("role", "unknown"),
    }
    report["runtime_constants"] = export_preprocess_and_geometry(
        data, int(meta["input_length"]), ROOT / args.output
    )
    print(report)


if __name__ == "__main__":
    main()
