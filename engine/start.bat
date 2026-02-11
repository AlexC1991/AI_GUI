@echo off
title VoxAI Local Server Boot

:: ============================================================================
:: ZLUDA / HIP Environment Setup (Copied from AI_GUI)
:: ============================================================================

:: HIP SDK path
set "HIP_PATH=C:\Program Files\AMD\ROCm\6.2\"
set "PATH=%HIP_PATH%bin;%PATH%"

:: ZLUDA path
set "ZLUDA_PATH=C:\Users\batty\AppData\Local\zluda\zluda"
set "PATH=%ZLUDA_PATH%;%PATH%"

:: Relocate AMD/ROCm caches to E: drive
set "AMD_SHADER_DISK_CACHE_PATH=E:\RocM_Cache"
set "MIOPEN_USER_DB_PATH=E:\RocM_Cache\miopen"
set "ROC_CACHE_DIR=E:\RocM_Cache\roc"
if not exist "E:\RocM_Cache\miopen" mkdir "E:\RocM_Cache\miopen" 2>nul
if not exist "E:\RocM_Cache\roc" mkdir "E:\RocM_Cache\roc" 2>nul

:: ZLUDA workarounds
set "DISABLE_ADDMM_CUDA_LT=1"
set "HIP_VISIBLE_DEVICES=0"
set "KMP_DUPLICATE_LIB_OK=TRUE"

echo Starting VoxAI Initialization with ZLUDA...
python vox_core_chat.py
pause