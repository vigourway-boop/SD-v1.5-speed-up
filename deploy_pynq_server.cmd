@echo off
setlocal
cd /d "%~dp0"

set "PYNQ_HOST=192.168.2.99"
if not "%~1"=="" set "PYNQ_HOST=%~1"
set "PYNQ_USER=xilinx"
if not "%~2"=="" set "PYNQ_USER=%~2"
set "REMOTE_DIR=/home/%PYNQ_USER%/pynq_cosine"

echo Deploying PYNQ service to %PYNQ_USER%@%PYNQ_HOST%:%REMOTE_DIR%
ssh -o StrictHostKeyChecking=accept-new %PYNQ_USER%@%PYNQ_HOST% "mkdir -p %REMOTE_DIR%"
if errorlevel 1 exit /b %errorlevel%
scp -o StrictHostKeyChecking=accept-new "pynq_cosine_overlay\artifacts\cosine_overlay.bit" "pynq_cosine_overlay\artifacts\cosine_overlay.hwh" "pynq_cosine_overlay\artifacts\pynq_cosine_server.py" "pynq_cosine_overlay\artifacts\start_pynq_cosine.sh" "pynq_cosine_overlay\artifacts\install_pynq_service.sh" %PYNQ_USER%@%PYNQ_HOST%:%REMOTE_DIR%/
if errorlevel 1 exit /b %errorlevel%
if defined PYNQ_SUDO_PASSWORD (
    echo(%PYNQ_SUDO_PASSWORD%| ssh -o StrictHostKeyChecking=accept-new %PYNQ_USER%@%PYNQ_HOST% "sudo -S -v && cd %REMOTE_DIR% && sh install_pynq_service.sh"
) else (
    ssh -o StrictHostKeyChecking=accept-new -t %PYNQ_USER%@%PYNQ_HOST% "cd %REMOTE_DIR% && sh install_pynq_service.sh"
)
if errorlevel 1 exit /b %errorlevel%

echo PYNQ service installed and started.
