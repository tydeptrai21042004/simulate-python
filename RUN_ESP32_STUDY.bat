@echo off
python scripts\run_esp32_study.py --config configs\esp32s3_official.yaml --cases 1 2 3 --seeds 11 22 33
if errorlevel 1 exit /b %errorlevel%
