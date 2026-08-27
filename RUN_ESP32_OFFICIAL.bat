@echo off
set PYTHONPATH=src
python scripts\train_esp32_pipeline.py --config configs\esp32s3_official.yaml --auto-data %*
