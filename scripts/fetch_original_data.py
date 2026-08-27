#!/usr/bin/env python3
"""Download the official UWB repository + Dyn_CIR_VAR.mat and convert them."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwb_tracking.official_data import ensure_official_standard_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch CLongLi/UWB-Radar-Pedestrian-Tracking and build the standard dataset."
    )
    parser.add_argument("--output", default="data/uwb_original_standard.mat")
    parser.add_argument("--source-dir", default="data/original_uwb/UWB-Radar-Pedestrian-Tracking")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    args = parser.parse_args()

    output = ensure_official_standard_dataset(
        ROOT / args.output,
        ROOT / args.source_dir,
        force_download=args.force_download,
        force_convert=args.force_convert,
    )
    print(f"Official standardized dataset ready: {output}")


if __name__ == "__main__":
    main()
