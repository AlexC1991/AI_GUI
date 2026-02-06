@echo off
title Iron Tunnel Host v10
color 0A
cls

echo ════════════════════════════════════════════════════════
echo        IRON TUNNEL HOST v10.0 - Secure AI Gateway
echo ════════════════════════════════════════════════════════
echo.
echo [SYSTEM] Checking Python...

python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERROR] Python not found!
    echo.
    echo Install Python 3.10+ from python.org
    echo Make sure to check "Add Python to PATH"
    echo.
    pause
    exit
)

echo [SYSTEM] Starting host...
echo.

python iron_host.py

if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERROR] Host crashed. See error above.
    echo.
    pause
)
