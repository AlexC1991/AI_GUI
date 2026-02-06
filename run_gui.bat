@echo off
cd /d "%~dp0"

echo.
echo  AI_GUI - Launching with HYBRID ENGINE (ZLUDA + VULKAN)
echo  ======================================================
echo.

:: ----------------------------------------------------------------------------
:: 0. CHECK BOOTSTRAP EXISTS
:: ----------------------------------------------------------------------------
if not exist "bootstrap.py" (
    echo [ERROR] bootstrap.py not found!
    echo This file is required to redirect temp/cache away from C: drive.
    echo Please ensure bootstrap.py is in the project root.
    pause
    exit /b 1
)

:: ----------------------------------------------------------------------------
:: 1. ZLUDA / HIP Environment (For Image Gen)
:: ----------------------------------------------------------------------------
set "HIP_PATH=C:\Program Files\AMD\ROCm\6.2\"
set "PATH=%HIP_PATH%bin;%PATH%"

set "ZLUDA_PATH=C:\Users\batty\AppData\Local\zluda\zluda"
set "PATH=%ZLUDA_PATH%;%PATH%"

set "ZLUDA_CACHE_DIR=%~dp0cache\zluda"
if not exist "%ZLUDA_CACHE_DIR%" mkdir "%ZLUDA_CACHE_DIR%"

:: AMD shader cache - keep on E: drive (has space)
set "AMD_SHADER_DISK_CACHE_PATH=E:\RocM_Cache"
set "MIOPEN_USER_DB_PATH=E:\RocM_Cache\miopen"
set "ROC_CACHE_DIR=E:\RocM_Cache\roc"
if not exist "E:\RocM_Cache\miopen" mkdir "E:\RocM_Cache\miopen" 2>nul
if not exist "E:\RocM_Cache\roc" mkdir "E:\RocM_Cache\roc" 2>nul

set "DISABLE_ADDMM_CUDA_LT=1"
set "HIP_VISIBLE_DEVICES=0"

:: ----------------------------------------------------------------------------
:: 2. VOX-AI VULKAN BRIDGE (The Fix for Chat Speed)
:: ----------------------------------------------------------------------------
set "PROJECT_ROOT=%~dp0"
set "VOX_API=%PROJECT_ROOT%VoxAI_Chat_API"

:: Add the API folder to PATH so the GUI can find ggml-vulkan.dll
set "PATH=%VOX_API%;%PATH%"

:: Force encoding to fix the crash
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=utf-8

:: ----------------------------------------------------------------------------
:: 3. PRE-SET TEMP DIRECTORIES (Belt and suspenders with bootstrap.py)
:: ----------------------------------------------------------------------------
:: These get overridden by bootstrap.py but we set them here too just in case
:: something loads before Python starts

set "AI_TEMP_DRIVE=A:"
if not exist "%AI_TEMP_DRIVE%\AI_Temp" mkdir "%AI_TEMP_DRIVE%\AI_Temp" 2>nul
if not exist "%AI_TEMP_DRIVE%\AI_Cache" mkdir "%AI_TEMP_DRIVE%\AI_Cache" 2>nul

:: Only set these if A: drive exists and has the folders
if exist "%AI_TEMP_DRIVE%\AI_Temp" (
    set "TEMP=%AI_TEMP_DRIVE%\AI_Temp\general"
    set "TMP=%AI_TEMP_DRIVE%\AI_Temp\general"
    set "TMPDIR=%AI_TEMP_DRIVE%\AI_Temp\general"
    set "HF_HOME=%AI_TEMP_DRIVE%\AI_Cache\huggingface"
    set "TORCH_HOME=%AI_TEMP_DRIVE%\AI_Cache\torch"
    set "TRANSFORMERS_CACHE=%AI_TEMP_DRIVE%\AI_Cache\huggingface\transformers"
    set "DIFFUSERS_CACHE=%AI_TEMP_DRIVE%\AI_Cache\huggingface\diffusers"
    if not exist "%AI_TEMP_DRIVE%\AI_Temp\general" mkdir "%AI_TEMP_DRIVE%\AI_Temp\general" 2>nul
    if not exist "%AI_TEMP_DRIVE%\AI_Cache\huggingface" mkdir "%AI_TEMP_DRIVE%\AI_Cache\huggingface" 2>nul
    if not exist "%AI_TEMP_DRIVE%\AI_Cache\torch" mkdir "%AI_TEMP_DRIVE%\AI_Cache\torch" 2>nul
    echo [OK] Temp/Cache directories set to A: drive
) else (
    echo [WARN] A: drive not available, bootstrap.py will handle temp directories
)

:: PyTorch memory management
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

:: Disable HuggingFace telemetry
set "HF_HUB_DISABLE_TELEMETRY=1"

:: Allow duplicate OpenMP runtimes (Fix for OMP Error #15)
set "KMP_DUPLICATE_LIB_OK=TRUE"

:: ----------------------------------------------------------------------------
:: 4. Launch
:: ----------------------------------------------------------------------------
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found!
    pause
    exit /b 1
)

if not exist "models\checkpoints" mkdir "models\checkpoints"
if not exist "models\loras" mkdir "models\loras"
if not exist "outputs\images" mkdir "outputs\images"

echo.
echo Launching AI_GUI...
echo (bootstrap.py will configure temp directories on startup)
echo.

python main.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%
    pause
)
