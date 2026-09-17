@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if not defined SD_PYTHON set "SD_PYTHON=D:\lenovo\download\conda\envs\sd_accel\python.exe"
if exist "%SD_PYTHON%" (
    "%SD_PYTHON%" generate_standard.py %*
) else (
    call conda run --no-capture-output -n sd_accel python generate_standard.py %*
)
exit /b %errorlevel%
