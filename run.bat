@echo off
title ResNet18 Classifier Launcher
echo ============================================
echo   Starting ResNet18 Object Recognition App
echo ============================================
echo.

REM Check if the server is already running on port 8081
netstat -ano | findstr ":8081" >nul 2>&1
if %errorlevel%==0 (
    echo Server already running on port 8081.
    echo Opening browser...
    timeout /t 1 /nobreak >nul
    start "" http://localhost:8081
    exit /b 0
)

REM Start the Flask server in a new window
echo Starting Flask server...
start "ResNet18 Server" cmd /k "cd /d C:\resnet18-classifier && C:\resnet18-classifier\venv\Scripts\python.exe feature_app.py"

REM Wait for the server to warm up
echo Waiting for server to start...
timeout /t 6 /nobreak >nul

REM Open the browser
echo Opening browser at http://localhost:8081
start "" http://localhost:8081

echo.
echo Done! The app is running in the "ResNet18 Server" window.
echo Close that window to stop the server.
timeout /t 3 /nobreak >nul
