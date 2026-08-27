#!/usr/bin/env python3
"""Convert the official MATLAB repository files into this project's MAT schema."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwb_tracking.official_data import convert_official_matlab_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", required=True, type=Path)
    parser.add_argument("--dynamic", required=True, type=Path)
    parser.add_argument("--anchors", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/uwb_original_standard.mat"), type=Path)
    args = parser.parse_args()

    output = convert_official_matlab_data(
        args.background,
        args.dynamic,
        args.anchors,
        args.output,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
