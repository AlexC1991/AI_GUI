"""
AI_GUI (VoxAI Orchestrator) - Main Entry Point

CRITICAL: The first thing we do is import bootstrap to set up temp directories.
This MUST happen before ANY other imports to prevent C: drive usage.
"""

# =========================================================
# STEP 0: BOOTSTRAP - MUST BE THE ABSOLUTE FIRST IMPORT
# =========================================================
# This sets up temp/cache directories and lies to Windows about C: drive
# having only 250MB free. This MUST run before torch/diffusers/etc load.

import bootstrap  # noqa: F401 - KEEP THIS FIRST!

# =========================================================
# STEP 1: BACKEND SETUP - MUST BE BEFORE LLAMA IMPORTS
# =========================================================
# This mirrors exactly what vox_core_chat.py does:
# 1. Set LLAMA_CPP_LIB to point to our custom llama.dll
# 2. Add DLL directories to PATH
# 3. Pre-load ggml.dll and call ggml_backend_load_all()

import sys
import os
import ctypes
from pathlib import Path

# Get app directory
APP_DIR = Path(__file__).parent.absolute()
VOX_API_DIR = APP_DIR / "VoxAI_Chat_API"

def _setup_vox_backend():
    """
    Set up the VOX-AI backend exactly like vox_core_chat.py does.
    
    This MUST run before any import that could touch llama_cpp.
    """
    print("[VoxAI] Setting up custom backend...")
    
    # Check if VoxAI_Chat_API exists
    if not VOX_API_DIR.exists():
        print(f"[VoxAI] WARNING: {VOX_API_DIR} not found")
        return False
    
    vox_str = str(VOX_API_DIR)
    llama_dll = VOX_API_DIR / "llama.dll"
    ggml_dll = VOX_API_DIR / "ggml.dll"
    
    if not llama_dll.exists():
        print(f"[VoxAI] WARNING: llama.dll not found in {vox_str}")
        return False
    
    # =========================================================
    # 1. Set LLAMA_CPP_LIB - This is the KEY
    # =========================================================
    os.environ["LLAMA_CPP_LIB"] = str(llama_dll)
    print(f"[VoxAI] LLAMA_CPP_LIB = {llama_dll}")
    
    # =========================================================
    # 2. Set GGML backend search path
    # =========================================================
    os.environ["GGML_BACKEND_SEARCH_PATH"] = vox_str
    
    # =========================================================
    # 3. Add to PATH (for dependent DLLs)
    # =========================================================
    os.environ["PATH"] = vox_str + os.pathsep + os.environ.get("PATH", "")
    
    # Also add parent dir (for ZLUDA if it's there)
    parent_dir = str(APP_DIR)
    os.environ["PATH"] = parent_dir + os.pathsep + os.environ["PATH"]
    
    # =========================================================
    # 4. Add DLL directories (Windows 10+ requirement)
    # =========================================================
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(vox_str)
            os.add_dll_directory(parent_dir)
            print(f"[VoxAI] Added DLL directories")
        except Exception as e:
            print(f"[VoxAI] DLL directory warning: {e}")
    
    # =========================================================
    # 5. Pre-load ggml.dll and initialize backends
    # =========================================================
    if ggml_dll.exists():
        try:
            ggml = ctypes.CDLL(str(ggml_dll))
            print(f"[VoxAI] Loaded ggml.dll")
            
            if hasattr(ggml, 'ggml_backend_load_all'):
                ggml.ggml_backend_load_all()
                print("[VoxAI] ✓ Backends loaded (ggml_backend_load_all)")
        except Exception as e:
            print(f"[VoxAI] Backend loading error: {e}")
            return False
    else:
        print(f"[VoxAI] WARNING: ggml.dll not found")
        return False
    
    # Mark as initialized
    os.environ["_VOX_BACKEND_INITIALIZED"] = "1"
    print("[VoxAI] ✓ Backend setup complete")
    return True

# RUN IMMEDIATELY
_vox_ok = _setup_vox_backend()

# =========================================================
# NOW it's safe to import everything else
# =========================================================

import subprocess
import importlib
import atexit

# Set process priority (like standalone does)
try:
    import psutil
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)
except:
    pass

# -------------------------
# Environment Configuration
# -------------------------

# Fix ZLUDA/MIOpen issues for AMD GPUs
os.environ["MIOPEN_FIND_MODE"] = "NORMAL"
os.environ["MIOPEN_USER_DB_PATH"] = str(APP_DIR / "models" / "miopen_cache")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"

# Disable Flash Attention (incompatible with ZLUDA)
os.environ["DIFFUSERS_FLASH_ATTN"] = "0"
os.environ["XFORMERS_DISABLED"] = "1"

# Note: Temp directories are now handled by bootstrap.py
# The following are kept for backward compatibility but bootstrap takes priority

# MIOpen cache (this can stay on E: as configured in bat file)
MIOPEN_CACHE = APP_DIR / "models" / "miopen_cache"
MIOPEN_CACHE.mkdir(parents=True, exist_ok=True)

# -------------------------
# Dependency Configuration
# -------------------------

REQUIRED_PACKAGES = {
    "PySide6": "PySide6>=6.6",
    "markdown": "markdown",
    "pygments": "pygments",
    "psutil": "psutil",
    "requests": "requests",
    "peft": "peft",
    "diffusers": "diffusers>=0.31.0",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
    "huggingface_hub": "huggingface_hub",
    "sentencepiece": "sentencepiece",
    "protobuf": "protobuf",
    "gguf": "gguf",
    # DO NOT auto-install llama-cpp-python - we use custom DLLs
}


def install_package(package: str):
    print(f"[VoxAI] Installing: {package}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        print(f"[VoxAI] Failed to install {package}: {e}")
        sys.exit(1)


def ensure_dependencies():
    print("[VoxAI] Checking dependencies...")
    missing = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append((module_name, package_name))
    
    if missing:
        print(f"[VoxAI] Installing {len(missing)} missing packages...")
        for module_name, package_name in missing:
            install_package(package_name)
    
    print("[VoxAI] Dependencies OK\n")


# -------------------------
# Cleanup Functions
# -------------------------

def run_startup_cleanup():
    print("[VoxAI] Running startup cleanup...")
    try:
        # Use bootstrap's clear_temp if available
        import bootstrap
        cleared = bootstrap.clear_temp()
        if cleared > 0:
            print(f"[VoxAI] Cleaned {cleared / 1024**2:.1f} MB from temp")
    except:
        pass
    
    try:
        from backend.cleanup import get_cleanup_manager
        manager = get_cleanup_manager(APP_DIR)
        freed = manager.cleanup_all(include_hf_cache=False, include_miopen=False)
        if freed > 0:
            print(f"[VoxAI] Cleaned {freed / 1024**3:.2f} GB")
        else:
            print("[VoxAI] No additional temp files to clean")
    except ImportError:
        fallback_cleanup()
    except Exception as e:
        print(f"[VoxAI] Cleanup error: {e}")


def fallback_cleanup():
    import shutil
    import tempfile
    total = 0
    
    # Use bootstrap temp dir
    try:
        import bootstrap
        temp_dir = bootstrap.get_temp_dir()
    except:
        temp_dir = APP_DIR / "temp_workspace"
    
    if temp_dir.exists():
        for item in temp_dir.iterdir():
            try:
                if item.is_file():
                    total += item.stat().st_size
                    item.unlink()
                elif item.is_dir():
                    total += sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                    shutil.rmtree(item, ignore_errors=True)
            except:
                pass
    
    # Clean system temp (though with bootstrap it should be redirected)
    system_temp = Path(tempfile.gettempdir())
    for pattern in ["flux_t5_offload", "t5_offload", "offload_folder"]:
        target = system_temp / pattern
        if target.exists():
            try:
                total += sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
                shutil.rmtree(target, ignore_errors=True)
            except:
                pass
    
    if total > 0:
        print(f"[VoxAI] Cleaned {total / 1024**2:.1f} MB")


def run_shutdown_cleanup():
    print("\n[VoxAI] Shutdown cleanup...")
    try:
        from backend.cleanup import cleanup_on_shutdown
        cleanup_on_shutdown()
    except:
        fallback_cleanup()
    print("[VoxAI] Goodbye!")


def register_cleanup_handlers():
    atexit.register(run_shutdown_cleanup)


# -------------------------
# PyTorch Setup
# -------------------------

def setup_torch():
    try:
        import torch
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(True)
        
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[VoxAI] GPU: {gpu} ({vram:.1f} GB)")
            if "ZLUDA" in gpu:
                print("[VoxAI] ZLUDA detected - Flash Attention disabled")
        else:
            print("[VoxAI] No GPU detected")
    except ImportError:
        pass
    except Exception as e:
        print(f"[VoxAI] PyTorch warning: {e}")


# -------------------------
# Debug Setup
# -------------------------

def setup_debug():
    try:
        from backend.debug import enable_debug, DebugLevel
        if "--debug" in sys.argv or os.environ.get("AI_GUI_DEBUG"):
            enable_debug(DebugLevel.VERBOSE)
        elif "--trace" in sys.argv:
            enable_debug(DebugLevel.TRACE)
        else:
            enable_debug(DebugLevel.INFO)
    except ImportError:
        pass


# -------------------------
# Main Entry Point
# -------------------------

def main():
    print("=" * 50)
    print("  VoxAI Orchestrator")
    print("=" * 50)
    print()
    
    if _vox_ok:
        print("[VoxAI] ✓ Custom backend ready")
    else:
        print("[VoxAI] ⚠ Backend not loaded - will use CPU")
    print()
    
    ensure_dependencies()
    setup_torch()
    setup_debug()
    run_startup_cleanup()
    register_cleanup_handlers()
    
    print("[VoxAI] Launching...\n")
    
    from PySide6.QtWidgets import QApplication
    from main_window import MainWindow
    
    app = QApplication(sys.argv)
    app.setApplicationName("VoxAI Orchestrator")
    app.setOrganizationName("AI_GUI")
    app.setApplicationVersion("0.1.0")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
