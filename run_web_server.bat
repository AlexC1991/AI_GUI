@echo off
cd /d "%~dp0"

echo.
echo  VoxAI Web Server - Remote Access
echo  =================================
echo.

:: ============================================
:: CONFIGURATION - EDIT THESE
:: ============================================

:: Port to run on (make sure to port forward this on your router)
set "SERVER_PORT=7860"

:: Password for access (CHANGE THIS!)
set "SERVER_PASSWORD=voxai2024"

:: Bind to all interfaces (0.0.0.0) to allow remote access
set "SERVER_HOST=0.0.0.0"

:: ============================================
:: DISPLAY INFO
:: ============================================

:: Get local IP
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    for /f "tokens=1" %%b in ("%%a") do (
        set "LOCAL_IP=%%b"
        goto :found_ip
    )
)
:found_ip

echo  Local URL:    http://localhost:%SERVER_PORT%
echo  Network URL:  http://%LOCAL_IP%:%SERVER_PORT%
echo  Port:         %SERVER_PORT%
echo  Password:     %SERVER_PASSWORD%
echo.
echo  To access from outside your network:
echo  1. Port forward %SERVER_PORT% on your router
echo  2. Use your public IP (google "what is my ip")
echo.
echo  Or use Tailscale/Cloudflare Tunnel for secure access
echo.
echo  =================================
echo.

:: ============================================
:: ACTIVATE VENV AND RUN
:: ============================================

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found!
    echo Please create it first: python -m venv venv
    pause
    exit /b 1
)

:: Install Flask if needed
pip show flask >nul 2>&1
if errorlevel 1 (
    echo Installing Flask...
    pip install flask flask-cors --quiet
)

:: Run the server
python web_server.py --host %SERVER_HOST% --port %SERVER_PORT% --password "%SERVER_PASSWORD%"

pause
