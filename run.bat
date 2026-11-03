@echo off
setlocal
title ResNet18 Object Recognition Launcher
cd /d "%~dp0"

echo ============================================
echo   ResNet18 Object Recognition App
echo ============================================
echo.

REM ---- 1. Locate a usable Python interpreter (bundled venv first, then system) ----
set "PY="
if exist "%~dp0venv\Scripts\python.exe" set "PY=%~dp0venv\Scripts\python.exe"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    where python3 >nul 2>&1 && set "PY=python3"
)
if not defined PY (
    echo [ERROR] Python 3.9-3.11 not found.
    echo         Install it from https://www.python.org/downloads/
    echo         and tick "Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM ---- 2. Install dependencies when they are missing ----
%PY% -c "import torch, torchvision, flask" >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Installing dependencies, this may take a few minutes...
    %PY% -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo.
        echo [ERROR] Dependency installation failed. Check your network and try again.
        pause
        exit /b 1
    )
)

REM ---- 3. Generate model files on first run ----
if not exist "%~dp0resnet18_feature.pt" (
    echo [SETUP] Model files not found, preparing them now ^(needs internet^)...
    %PY% "%~dp0prepare_model.py"
    if errorlevel 1 (
        echo.
        echo [ERROR] Model preparation failed. Check your network and try again.
        pause
        exit /b 1
    )
)

REM ---- 4. Start the server (or reuse the running one) ----
netstat -ano | findstr ":8081" >nul 2>&1
if %errorlevel%==0 (
    echo Server is already running on port 8081. Opening browser...
    timeout /t 1 /nobreak >nul
    start "" http://localhost:8081
    exit /b 0
)

echo Starting Flask server...
start "ResNet18 Server" cmd /k ""%PY%" "%~dp0feature_app.py""

echo Waiting for the server to warm up...
timeout /t 8 /nobreak >nul

start "" http://localhost:8081

echo.
echo Done. The app is running in the "ResNet18 Server" window.
echo Close that window to stop the server.
timeout /t 3 >nul
