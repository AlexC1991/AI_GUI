@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
:: AI_GUI Setup Script
:: Target: AMD GPU with ZLUDA (RX 6000 series)
:: ============================================================================

title AI_GUI Setup

echo.
echo  ==============================================================
echo              AI_GUI - Auto Setup
echo                A:\Github\AI_GUI
echo  ==============================================================
echo.

:: ----------------------------------------------------------------------------
:: Configuration
:: ----------------------------------------------------------------------------
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

set "VENV_DIR=%PROJECT_DIR%\venv"
set "ZLUDA_DIR=C:\Users\batty\AppData\Local\zluda\zluda"

:: ----------------------------------------------------------------------------
:: Check Python
:: ----------------------------------------------------------------------------
echo [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         Please install Python 3.10+ from python.org
    pause
    exit /b 1
)
python --version

:: ----------------------------------------------------------------------------
:: Create Virtual Env
:: ----------------------------------------------------------------------------
echo.
echo [2/6] Creating Virtual Environment...
if not exist "%VENV_DIR%" (
    python -m venv "%VENV_DIR%"
    echo        Created new venv.
) else (
    echo        Using existing venv.
)

call "%VENV_DIR%\Scripts\activate.bat"

:: ----------------------------------------------------------------------------
:: Install Dependencies
:: ----------------------------------------------------------------------------
echo.
echo [3/6] Installing Dependencies...
pip install --upgrade pip --quiet

if exist "%PROJECT_DIR%\requirements.txt" (
    pip install -r "%PROJECT_DIR%\requirements.txt"
) else (
    echo        No requirements.txt found, installing core deps...
    pip install PySide6 markdown pygments psutil requests pillow
)

:: ----------------------------------------------------------------------------
:: GPU Setup - ZLUDA requires cu118, NOT cu121!
:: ----------------------------------------------------------------------------
echo.
echo [4/6] Installing PyTorch for ZLUDA (cu118)...

:: Check current torch version
pip show torch 2>nul | findstr "Version" >nul
if not errorlevel 1 (
    for /f "tokens=2" %%v in ('pip show torch ^| findstr "Version"') do set "TORCH_VER=%%v"
    echo        Current torch: !TORCH_VER!
    
    :: Check if it's cu118
    echo !TORCH_VER! | findstr "cu118" >nul
    if errorlevel 1 (
        echo        Wrong CUDA version detected, reinstalling...
        pip uninstall torch torchvision torchaudio -y
        pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ) else (
        echo        Correct cu118 version already installed.
    )
) else (
    echo        Installing PyTorch 2.6.0+cu118...
    pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

:: Install diffusers and transformers
echo.
echo        Installing Diffusers...
pip install diffusers transformers accelerate safetensors

:: ----------------------------------------------------------------------------
:: ZLUDA Patching
:: ----------------------------------------------------------------------------
echo.
echo [5/6] Patching PyTorch with ZLUDA DLLs...

set "TORCH_LIB=%VENV_DIR%\Lib\site-packages\torch\lib"

if not exist "%ZLUDA_DIR%\cublas.dll" (
    echo [WARNING] ZLUDA not found at %ZLUDA_DIR%
    echo           Please install ZLUDA first.
    goto :skip_patch
)

if not exist "%TORCH_LIB%" (
    echo [WARNING] Torch lib not found at %TORCH_LIB%
    goto :skip_patch
)

:: Create backup
set "BACKUP_DIR=%TORCH_LIB%\cuda_backup"
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

:: Patch cublas
if exist "%TORCH_LIB%\cublas64_11.dll" (
    if not exist "%BACKUP_DIR%\cublas64_11.dll" (
        copy "%TORCH_LIB%\cublas64_11.dll" "%BACKUP_DIR%\" >nul
    )
    copy /y "%ZLUDA_DIR%\cublas.dll" "%TORCH_LIB%\cublas64_11.dll" >nul
    echo        Patched cublas64_11.dll
)

:: Patch cusparse
if exist "%TORCH_LIB%\cusparse64_11.dll" (
    if not exist "%BACKUP_DIR%\cusparse64_11.dll" (
        copy "%TORCH_LIB%\cusparse64_11.dll" "%BACKUP_DIR%\" >nul
    )
    copy /y "%ZLUDA_DIR%\cusparse.dll" "%TORCH_LIB%\cusparse64_11.dll" >nul
    echo        Patched cusparse64_11.dll
)

:: Patch nvrtc
if exist "%TORCH_LIB%\nvrtc64_112_0.dll" (
    if not exist "%BACKUP_DIR%\nvrtc64_112_0.dll" (
        copy "%TORCH_LIB%\nvrtc64_112_0.dll" "%BACKUP_DIR%\" >nul
    )
    copy /y "%ZLUDA_DIR%\nvrtc.dll" "%TORCH_LIB%\nvrtc64_112_0.dll" >nul
    echo        Patched nvrtc64_112_0.dll
)

echo        ZLUDA patch complete!

:skip_patch

:: ----------------------------------------------------------------------------
:: Create Directories
:: ----------------------------------------------------------------------------
echo.
echo [6/6] Creating directories...
if not exist "%PROJECT_DIR%\models\checkpoints" mkdir "%PROJECT_DIR%\models\checkpoints"
if not exist "%PROJECT_DIR%\models\loras" mkdir "%PROJECT_DIR%\models\loras"
if not exist "%PROJECT_DIR%\outputs\images" mkdir "%PROJECT_DIR%\outputs\images"
if not exist "%PROJECT_DIR%\cache\zluda" mkdir "%PROJECT_DIR%\cache\zluda"

:: Initialize config if missing
if not exist "%PROJECT_DIR%\config.json" (
    echo {} > "%PROJECT_DIR%\config.json"
)

:: ----------------------------------------------------------------------------
:: Finish
:: ----------------------------------------------------------------------------
echo.
echo  ==============================================================
echo                     Setup Complete!
echo  ==============================================================
echo.
echo  PyTorch: cu118 (ZLUDA compatible)
echo  ZLUDA:   Patched into torch\lib
echo.
echo  Run 'run_gui.bat' to start the application.
echo.
pause
