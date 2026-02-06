@echo off
cd /d "%~dp0"
echo =================================================
echo  VOX-AI BRAIN SURGEON (RX 6600 Edition)
echo =================================================
echo.
echo 1. Activating Virtual Environment...
call venv\Scripts\activate

echo.
echo 2. Removing wrong/broken libraries...
pip uninstall -y llama-cpp-python
pip uninstall -y llama-cpp-python-cuda

echo.
echo 3. Cleaning Pip Cache (Critical)...
pip cache purge

echo.
echo 4. Installing VULKAN Engine (Native AMD)...
:: This is the specific pre-built wheel for Vulkan
pip install llama-cpp-python --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan

echo.
echo =================================================
echo  REPAIR COMPLETE.
echo =================================================
pause