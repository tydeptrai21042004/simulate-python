#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e .
uwb-track full --config configs/full.yaml
