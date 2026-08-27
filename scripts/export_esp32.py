#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwb_tracking.data import load_uwb_mat, prepare_inputs, subset_observations
from uwb_tracking.esp32.exporter import export_checkpoint
from uwb_tracking.esp32.preprocess_export import export_preprocess_and_geometry
from uwb_tracking.esp32.training import load_student_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="Export a trained ESP32 student to raw INT8 / ONNX / ESP-DL.")
    p.add_argument("--checkpoint", default="results/esp32s3/checkpoints/best_student.pt")
    p.add_argument("--data", default="data/uwb_demo_input.mat")
    p.add_argument("--output", default="results/esp32s3/export")
    p.add_argument("--target", default="esp32s3", choices=["esp32", "esp32s3", "esp32p4", "c"])
    p.add_argument("--calibration-samples", type=int, default=256)
    p.add_argument("--onnx", action="store_true")
    p.add_argument("--espdl", action="store_true")
    args = p.parse_args()

    _, meta = load_student_checkpoint(ROOT / args.checkpoint)
    data = load_uwb_mat(ROOT / args.data)
    obs = subset_observations(data, np.arange(min(data.num_time, 120)))
    prepared = prepare_inputs(data, obs, int(meta["input_length"]))
    calibration = prepared.fusion[: min(args.calibration_samples, len(prepared.fusion))]
    report = export_checkpoint(
        ROOT / args.checkpoint,
        calibration,
        ROOT / args.output,
        target=args.target,
        export_onnx_file=args.onnx or args.espdl,
        export_espdl_file=args.espdl,
    )
    report["runtime_constants"] = export_preprocess_and_geometry(
        data, int(meta["input_length"]), ROOT / args.output
    )
    print(report)


if __name__ == "__main__":
    main()
