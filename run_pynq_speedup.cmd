@echo off
setlocal
cd /d "%~dp0"

if not "%~1"=="" set "PYNQ_HOST=%~1"
if not defined PYNQ_HOST set "PYNQ_HOST=192.168.2.99"

echo PYNQ-Z2: %PYNQ_HOST%:9000
set "SD_PYTHON=D:\lenovo\download\conda\envs\sd_accel\python.exe"
if exist "%SD_PYTHON%" (
    "%SD_PYTHON%" combined_speed_test.py --with-baseline
) else (
    call conda run --no-capture-output -n sd_accel python combined_speed_test.py --with-baseline
)
if errorlevel 1 exit /b %errorlevel%
