#!/usr/bin/env bash
set -euo pipefail
python scripts/run_esp32_study.py \
  --config configs/esp32s3_official.yaml \
  --cases 1 2 3 \
  --seeds 11 22 33
