@echo off
python -m pip install -e .
if errorlevel 1 exit /b 1
uwb-track full --config configs/full.yaml
