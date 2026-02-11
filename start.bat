@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title VoxAI Orchestrator - Launcher
color 0B

:: ============================================================================
::                        VoxAI Orchestrator Launcher
:: ============================================================================

echo.
echo.
echo        ##     ##  #######  ##     ##    ###    ####
echo        ##     ## ##     ##  ##   ##    ## ##    ##
echo        ##     ## ##     ##   ## ##    ##   ##   ##
echo        ##     ## ##     ##    ###    ##     ##  ##
echo         ##   ##  ##     ##   ## ##   #########  ##
echo          ## ##   ##     ##  ##   ##  ##     ##  ##
echo           ###     #######  ##     ## ##     ## ####
echo.
echo     .===========================================.
echo     :  VoxAI Orchestrator - Desktop Launcher    :
echo     :  Engine: ZLUDA + VULKAN Hybrid            :
echo     '==========================================='
echo.

:: Timestamp
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set "TODAY=%%a/%%b/%%c"
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set "NOW=%%a:%%b"
echo     [%TODAY% %NOW%]  Pre-flight sequence initiated...
echo.
echo     _______________________________________________
echo     ^|                                             ^|
echo     ^|          SYSTEM DIAGNOSTICS                 ^|
echo     ^|_____________________________________________^|
echo.

:: ============================================================================
::  PHASE 1 - CORE FILE CHECKS
:: ============================================================================
echo      [1/10] CORE FILES
echo      ~~~~~~~~~~~~~~~~~~

set "ERRORS=0"

if exist "bootstrap.py" (
    echo        + bootstrap.py ................ OK
) else (
    echo        x bootstrap.py ................ MISSING
    set /a ERRORS+=1
)

if exist "main.py" (
    echo        + main.py ..................... OK
) else (
    echo        x main.py ..................... MISSING
    set /a ERRORS+=1
)

if exist "main_window.py" (
    echo        + main_window.py .............. OK
) else (
    echo        x main_window.py .............. MISSING
    set /a ERRORS+=1
)

if exist "config.json" (
    echo        + config.json ................. OK
) else (
    echo        ~ config.json ................. WARN (copy config.example.json)
)

echo.

:: ============================================================================
::  PHASE 2 - VIRTUAL ENVIRONMENT
:: ============================================================================
echo      [2/10] VIRTUAL ENVIRONMENT
echo      ~~~~~~~~~~~~~~~~~~~~~~~~~~~

if exist "venv\Scripts\activate.bat" (
    echo        + venv found .................. OK
    call venv\Scripts\activate.bat
    echo        + venv activated .............. OK
) else (
    echo        x venv not found .............. FAIL
    echo          Run: python -m venv venv
    set /a ERRORS+=1
    goto :launch_failed
)

for /f "tokens=*" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
echo        i %PYVER%

echo.

:: ============================================================================
::  PHASE 3 - DEPENDENCY INSTALLATION
:: ============================================================================
echo      [3/10] DEPENDENCIES
echo      ~~~~~~~~~~~~~~~~~~~~~
echo        i Installing requirements...

pip install -r requirements.txt --quiet --disable-pip-version-check 2>nul
if !ERRORLEVEL! EQU 0 (
    echo        + requirements.txt ............ OK
) else (
    echo        ~ requirements.txt ............ WARN (some may have failed)
)

:: Sub-project requirements
if exist "engine\requirements.txt" (
    pip install -r engine\requirements.txt --quiet --disable-pip-version-check 2>nul
    echo        + engine deps ................. OK
)

if exist "gateway\requirements.txt" (
    pip install -r gateway\requirements.txt --quiet --disable-pip-version-check 2>nul
    echo        + gateway deps ................ OK
)

:: Verify critical packages
python -c "import PySide6" >nul 2>&1
if !ERRORLEVEL! EQU 0 ( echo        + PySide6 ..................... OK ) else ( echo        x PySide6 ..................... MISSING & set /a ERRORS+=1 )

python -c "import msgpack" >nul 2>&1
if !ERRORLEVEL! EQU 0 ( echo        + msgpack ..................... OK ) else ( echo        ~ msgpack ..................... WARN )

python -c "import chromadb" >nul 2>&1
if !ERRORLEVEL! EQU 0 ( echo        + chromadb .................... OK ) else ( echo        ~ chromadb .................... WARN )

python -c "import psutil" >nul 2>&1
if !ERRORLEVEL! EQU 0 ( echo        + psutil ...................... OK ) else ( echo        ~ psutil ...................... WARN )

echo.

:: ============================================================================
::  PHASE 4 - DLL / RUNTIME CHECKS
:: ============================================================================
echo      [4/10] RUNTIME LIBRARIES
echo      ~~~~~~~~~~~~~~~~~~~~~~~~~

if exist "llama.dll" (
    echo        + llama.dll ................... OK
) else (
    echo        ~ llama.dll ................... WARN
)

if exist "ggml.dll" (
    echo        + ggml.dll .................... OK
) else (
    echo        ~ ggml.dll .................... WARN
)

if exist "ggml-vulkan.dll" (
    echo        + ggml-vulkan.dll ............. OK  [Vulkan]
) else (
    echo        ~ ggml-vulkan.dll ............. WARN
)

if exist "ggml-cpu-haswell.dll" (
    echo        + ggml-cpu-haswell.dll ........ OK  [CPU]
) else (
    echo        ~ ggml-cpu-haswell.dll ........ WARN
)

echo.

:: ============================================================================
::  PHASE 5 - AI MODELS
:: ============================================================================
echo      [5/10] AI MODELS
echo      ~~~~~~~~~~~~~~~~~~

if exist "engine" (
    echo        + engine ...................... OK
) else (
    echo        ~ engine ...................... WARN (no AI chat)
)

set "MODEL_COUNT=0"
if exist "models\llm" (
    for %%f in ("models\llm\*.gguf") do set /a MODEL_COUNT+=1
)
if !MODEL_COUNT! GTR 0 (
    echo        + GGUF models ................. !MODEL_COUNT! found
) else (
    echo        ~ GGUF models ................. NONE
)

if not exist "models\checkpoints" mkdir "models\checkpoints"
echo        + models\checkpoints ........... OK

if not exist "models\loras" mkdir "models\loras"
echo        + models\loras ................. OK

echo.

:: ============================================================================
::  PHASE 6 - ZLUDA / HIP (AMD GPU for Image Generation)
:: ============================================================================
echo      [6/10] ZLUDA / HIP  (AMD GPU)
echo      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

set "HIP_PATH=C:\Program Files\AMD\ROCm\6.2\"
if exist "%HIP_PATH%bin" (
    set "PATH=%HIP_PATH%bin;%PATH%"
    echo        + ROCm 6.2 HIP ............... OK
) else (
    echo        ~ ROCm HIP ................... NOT FOUND
)

set "ZLUDA_PATH=C:\Users\batty\AppData\Local\zluda\zluda"
if exist "%ZLUDA_PATH%" (
    set "PATH=%ZLUDA_PATH%;%PATH%"
    echo        + ZLUDA bridge ................ OK
) else (
    echo        ~ ZLUDA bridge ................ NOT FOUND
)

set "ZLUDA_CACHE_DIR=%~dp0cache\zluda"
if not exist "%ZLUDA_CACHE_DIR%" mkdir "%ZLUDA_CACHE_DIR%" 2>nul
echo        + ZLUDA cache ................. OK

set "AMD_SHADER_DISK_CACHE_PATH=E:\RocM_Cache"
set "MIOPEN_USER_DB_PATH=E:\RocM_Cache\miopen"
set "ROC_CACHE_DIR=E:\RocM_Cache\roc"
if exist "E:\" (
    if not exist "E:\RocM_Cache\miopen" mkdir "E:\RocM_Cache\miopen" 2>nul
    if not exist "E:\RocM_Cache\roc" mkdir "E:\RocM_Cache\roc" 2>nul
    echo        + AMD shader cache ............ E:\RocM_Cache
) else (
    echo        ~ E: drive .................... NOT AVAILABLE
)

set "DISABLE_ADDMM_CUDA_LT=1"
set "HIP_VISIBLE_DEVICES=0"

echo.

:: ============================================================================
::  PHASE 7 - VULKAN BRIDGE (LLM Chat Speed)
:: ============================================================================
echo      [7/10] VULKAN BRIDGE  (LLM)
echo      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

set "PROJECT_ROOT=%~dp0"
set "VOX_API=%PROJECT_ROOT%engine"
set "PATH=%VOX_API%;%PATH%"
echo        + engine on PATH .............. OK

set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=utf-8
echo        + Python UTF-8 encoding ........ OK

echo.

:: ============================================================================
::  PHASE 8 - TEMP / CACHE DIRECTORIES (A: Drive)
:: ============================================================================
echo      [8/10] TEMP / CACHE DIRS
echo      ~~~~~~~~~~~~~~~~~~~~~~~~~

set "AI_TEMP_DRIVE=A:"
if exist "%AI_TEMP_DRIVE%\" (
    if not exist "%AI_TEMP_DRIVE%\AI_Temp" mkdir "%AI_TEMP_DRIVE%\AI_Temp" 2>nul
    if not exist "%AI_TEMP_DRIVE%\AI_Cache" mkdir "%AI_TEMP_DRIVE%\AI_Cache" 2>nul

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

    echo        + TEMP dir .................... A:\AI_Temp
    echo        + CACHE dir ................... A:\AI_Cache
) else (
    echo        ~ A: drive .................... NOT AVAILABLE
    echo          bootstrap.py will handle redirect at runtime
)

set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo        + PyTorch expandable segs ...... ON

set "HF_HUB_DISABLE_TELEMETRY=1"
echo        + HuggingFace telemetry ........ OFF

set "KMP_DUPLICATE_LIB_OK=TRUE"
echo        + OpenMP dup lib workaround .... ON

echo.

:: ============================================================================
::  PHASE 9 - IRONGATE CHECK
:: ============================================================================
echo      [9/10] IRONGATE INTEGRATION
echo      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if exist "gateway" (
    echo        + gateway ..................... OK
    if exist "gateway\iron_host.py" (
        echo        + iron_host.py ................ OK  [Web Gateway]
    ) else (
        echo        ~ iron_host.py ................ MISSING
    )
    if exist "gateway\iron_desktop.py" (
        echo        + iron_desktop.py ............. OK  [Desktop Svc]
    ) else (
        echo        ~ iron_desktop.py ............. MISSING
    )
) else (
    echo        ~ gateway ..................... NOT FOUND
)

echo.

:: ============================================================================
::  PHASE 10 - OUTPUT / DATA DIRECTORIES
:: ============================================================================
echo      [10/10] OUTPUT / DATA DIRS
echo      ~~~~~~~~~~~~~~~~~~~~~~~~~~~

if not exist "outputs" mkdir "outputs"
if not exist "outputs\images" mkdir "outputs\images"
echo        + outputs\images ............... OK

if not exist "engine\data" mkdir "engine\data" 2>nul
if not exist "engine\data\conversations" mkdir "engine\data\conversations" 2>nul
echo        + Elastic Memory data dir ...... OK

if not exist "engine\data\vectordb" mkdir "engine\data\vectordb" 2>nul
echo        + Vector DB dir ................ OK

if not exist "models\llm" mkdir "models\llm" 2>nul
echo        + models\llm ................... OK

echo.

:: ============================================================================
::  PRE-FLIGHT SUMMARY
:: ============================================================================
if !ERRORS! GTR 0 (
    goto :launch_failed
)

echo     _______________________________________________
echo     ^|                                             ^|
echo     ^|        ALL SYSTEMS GO                       ^|
echo     ^|_____________________________________________^|
echo.
echo.
echo          //                          \\
echo         //  LAUNCHING VOX-AI ENGINE   \\
echo        //______________________________\\
echo        \\                              //
echo         \\    %PYVER%       //
echo          \\    Models: !MODEL_COUNT! GGUF loaded    //
echo           \\____________________________//
echo.
echo.

python main.py

:: ============================================================================
::  EXIT HANDLING
:: ============================================================================
if errorlevel 1 (
    echo.
    echo     _______________________________________________
    echo     ^|                                             ^|
    echo     ^|        APPLICATION CRASHED                  ^|
    echo     ^|        Exit Code: %errorlevel%                          ^|
    echo     ^|_____________________________________________^|
    echo.
    echo      Troubleshooting:
    echo        1. Check terminal output above for tracebacks
    echo        2. pip install -r requirements.txt
    echo        3. Verify config.json has valid API key
    echo        4. Check models\llm\ has .gguf files
    echo.
    pause
) else (
    echo.
    echo      VoxAI Orchestrator closed normally.
    echo.
    timeout /t 3 >nul
)
goto :eof

:: ============================================================================
::  ERROR EXIT
:: ============================================================================
:launch_failed
echo.
echo     _______________________________________________
echo     ^|                                             ^|
echo     ^|        LAUNCH ABORTED                       ^|
echo     ^|        !ERRORS! critical error(s) found              ^|
echo     ^|_____________________________________________^|
echo.
echo      Fix the errors listed above, then try again.
echo.
pause
exit /b 1
