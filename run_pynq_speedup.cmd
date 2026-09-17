@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

if not "%~1"=="" set "PYNQ_HOST=%~1"
if not defined PYNQ_HOST set "PYNQ_HOST=192.168.2.99"
if not defined SD_DECISION_BACKEND set "SD_DECISION_BACKEND=auto"

echo PYNQ-Z2: %PYNQ_HOST%:9000
echo Decision backend: %SD_DECISION_BACKEND% ^(auto prefers PYNQ and falls back to PC^)
set "SD_PYTHON=D:\lenovo\download\conda\envs\sd_accel\python.exe"
if exist "%SD_PYTHON%" (
    "%SD_PYTHON%" combined_speed_test.py --with-baseline
) else (
    call conda run --no-capture-output -n sd_accel python combined_speed_test.py --with-baseline
)
if errorlevel 1 exit /b %errorlevel%
