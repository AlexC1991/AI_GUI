@echo off
TITLE VoxAI GitHub Uploader
COLOR 0A

echo ========================================================
echo       VoxAI Orchestrator - Auto Git Uploader
echo ========================================================
echo.

:: 1. Check if Git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in your PATH.
    echo Please install Git from https://git-scm.com/
    pause
    exit /b
)

:: 2. Initialize Repostory if needed
if not exist ".git" (
    echo [INFO] Initializing new Git repository...
    git init
    git branch -M main
    echo [INFO] Adding Remote Origin...
    git remote add origin https://github.com/AlexC1991/AI_GUI.git
) else (
    echo [INFO] Git repository detected.
)

:: 3. Status Check
echo.
echo [STATUS] Checking for changes...
git status -s

echo.
echo ========================================================
set /p commit_msg="Enter commit message (Press Enter for 'Update UI'): "

if "%commit_msg%"=="" set commit_msg=Update UI

:: 4. Add, Commit, Push
echo.
echo [1/3] Adding files...
git add .

echo [2/3] Committing changes...
git commit -m "%commit_msg%"

echo [3/3] Pushing to GitHub (origin main)...
git push -u origin main

echo.
echo ========================================================
if %errorlevel% equ 0 (
    echo [SUCCESS] Upload Complete!
) else (
    echo [ERROR] Something went wrong. Check your internet or credentials.
)
echo ========================================================
pause