#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python scripts/export_esp32.py --checkpoint results/esp32s3/checkpoints/best_student.pt --target esp32s3 "$@"
