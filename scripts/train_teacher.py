#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwb_tracking.config import load_config
from uwb_tracking.data import get_case_split, load_uwb_mat
from uwb_tracking.training import train_models_for_case


def main() -> None:
    p = argparse.ArgumentParser(description="Train the rich offline U-Fuse teacher trio.")
    p.add_argument("--config", default="configs/full.yaml")
    p.add_argument("--case", type=int, default=1)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output", default="results/teacher/case1_seed11")
    args = p.parse_args()

    cfg = load_config(ROOT / args.config)
    data = load_uwb_mat(ROOT / cfg.data_path)
    train_idx, _ = get_case_split(data.num_time, args.case)
    models = train_models_for_case(data, train_idx, cfg, args.seed, ROOT / args.output)
    print({name: {"params": m.parameter_count, "val_mae_ns": m.validation_mae_ns} for name, m in models.items()})


if __name__ == "__main__":
    main()
