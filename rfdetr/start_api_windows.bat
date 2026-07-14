@echo off
cd /d "%~dp0"
echo Installing API requirements...
python -m pip install -r requirements_api.txt
echo Starting RF-DETR API...
python run_api.py
pause
