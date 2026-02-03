@echo off
cd /d "%~dp0"

:: ============================================================================
:: AI_GUI Launcher
:: Sets up ZLUDA/HIP environment for AMD GPU acceleration
:: ============================================================================

echo.
echo  AI_GUI - Starting with ZLUDA Support
echo  =====================================
echo.

:: ----------------------------------------------------------------------------
:: ZLUDA / HIP Environment (MUST be set before Python starts)
:: ----------------------------------------------------------------------------

:: HIP SDK path (REQUIRED for ZLUDA)
set "HIP_PATH=C:\Program Files\AMD\ROCm\6.2\"
set "PATH=%HIP_PATH%bin;%PATH%"

:: ZLUDA path
set "ZLUDA_PATH=C:\Users\batty\AppData\Local\zluda\zluda"
set "PATH=%ZLUDA_PATH%;%PATH%"

:: ZLUDA kernel cache (project local - speeds up subsequent runs)
set "ZLUDA_CACHE_DIR=%~dp0cache\zluda"
if not exist "%ZLUDA_CACHE_DIR%" mkdir "%ZLUDA_CACHE_DIR%"

:: Relocate AMD/ROCm caches to E: drive
set "AMD_SHADER_DISK_CACHE_PATH=E:\RocM_Cache"
set "MIOPEN_USER_DB_PATH=E:\RocM_Cache\miopen"
set "ROC_CACHE_DIR=E:\RocM_Cache\roc"
if not exist "E:\RocM_Cache\miopen" mkdir "E:\RocM_Cache\miopen" 2>nul
if not exist "E:\RocM_Cache\roc" mkdir "E:\RocM_Cache\roc" 2>nul

:: ZLUDA workarounds
set "DISABLE_ADDMM_CUDA_LT=1"
set "HIP_VISIBLE_DEVICES=0"

:: ----------------------------------------------------------------------------
:: Activate Virtual Environment
:: ----------------------------------------------------------------------------
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found!
    echo         Run setup.bat first.
    pause
    exit /b 1
)

:: ----------------------------------------------------------------------------
:: Create directories if missing
:: ----------------------------------------------------------------------------
if not exist "models\checkpoints" mkdir "models\checkpoints"
if not exist "models\loras" mkdir "models\loras"
if not exist "outputs\images" mkdir "outputs\images"

:: ----------------------------------------------------------------------------
:: Launch Application
:: ----------------------------------------------------------------------------
echo Launching AI_GUI...
python main.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application crashed. See error above.
    pause
)
