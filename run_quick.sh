#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e .
uwb-track quick-run --config configs/quick.yaml
