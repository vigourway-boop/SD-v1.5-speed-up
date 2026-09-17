@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
set "SD_PYTHON=D:\lenovo\download\conda\envs\sd_accel\python.exe"
if exist "%SD_PYTHON%" (
    "%SD_PYTHON%" combined_speed_test.py --with-baseline --controller-profile profiles/temporal_fast_100.json %*
) else (
    call conda run --no-capture-output -n sd_accel python combined_speed_test.py --with-baseline --controller-profile profiles/temporal_fast_100.json %*
)
exit /b %errorlevel%
