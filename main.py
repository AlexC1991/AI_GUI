"""
AI_GUI (VoxAI Orchestrator) - Main Entry Point

This is the main entry point for the application.
Handles:
- Environment setup (ZLUDA, MIOpen, temp directories)
- Dependency checking
- Cleanup of temp files on startup/shutdown
- Application launch
"""

import sys
import os
import subprocess
import importlib
import atexit
from pathlib import Path

# -------------------------
# Application Directory
# -------------------------

APP_DIR = Path(__file__).parent.absolute()

# -------------------------
# Environment Configuration
# -------------------------

# Fix ZLUDA/MIOpen issues for AMD GPUs
os.environ["MIOPEN_FIND_MODE"] = "NORMAL"
os.environ["MIOPEN_USER_DB_PATH"] = str(APP_DIR / "models" / "miopen_cache")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"

# Disable Flash Attention (incompatible with ZLUDA)
os.environ["DIFFUSERS_FLASH_ATTN"] = "0"
os.environ["XFORMERS_DISABLED"] = "1"

# Set temp directories to app folder (easier to clean)
TEMP_DIR = APP_DIR / "temp_workspace"
TEMP_DIR.mkdir(exist_ok=True)

os.environ["TMPDIR"] = str(TEMP_DIR)
os.environ["TEMP"] = str(TEMP_DIR)
os.environ["TMP"] = str(TEMP_DIR)

# HuggingFace cache in app folder
HF_CACHE = APP_DIR / "models" / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE)

# Create MIOpen cache directory
MIOPEN_CACHE = APP_DIR / "models" / "miopen_cache"
MIOPEN_CACHE.mkdir(parents=True, exist_ok=True)

# -------------------------
# Dependency Configuration
# -------------------------

REQUIRED_PACKAGES = {
    # Core UI
    "PySide6": "PySide6>=6.6",
    "markdown": "markdown",
    "pygments": "pygments",
    "psutil": "psutil",
    
    # LLM Providers
    "requests": "requests",
    
    # Image Generation
    "peft": "peft",
    "diffusers": "diffusers>=0.31.0",  # Need 0.31+ for GGUF support
    "transformers": "transformers",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
    "huggingface_hub": "huggingface_hub",
    
    # Flux support
    "sentencepiece": "sentencepiece",
    "protobuf": "protobuf",
    
    # GGUF support
    "gguf": "gguf",
}

# -------------------------
# Dependency Utilities
# -------------------------

def install_package(package: str):
    """Install a package using pip."""
    print(f"[VoxAI] Installing dependency: {package}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        print(f"[VoxAI] Failed to install {package}. Error: {e}")
        sys.exit(1)


def ensure_dependencies():
    """Check and install missing dependencies."""
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
            print(f"[VoxAI] Missing module: {module_name}")
            install_package(package_name)
    
    print("[VoxAI] All dependencies satisfied.\n")


# -------------------------
# Cleanup Functions
# -------------------------

def run_startup_cleanup():
    """Clean up temp files from previous sessions."""
    print("[VoxAI] Running startup cleanup...")
    
    try:
        from backend.cleanup import cleanup_on_startup, get_cleanup_manager
        
        manager = get_cleanup_manager(APP_DIR)
        freed = manager.cleanup_all(include_hf_cache=False, include_miopen=False)
        
        if freed > 0:
            print(f"[VoxAI] Cleaned {freed / 1024**3:.2f} GB of temp files")
        else:
            print("[VoxAI] No temp files to clean")
            
    except ImportError:
        # Fallback if cleanup module not available
        print("[VoxAI] Cleanup module not found, using fallback...")
        fallback_cleanup()
    except Exception as e:
        print(f"[VoxAI] Cleanup error (non-fatal): {e}")


def fallback_cleanup():
    """Fallback cleanup if the cleanup module isn't available."""
    import shutil
    import tempfile
    
    total_freed = 0
    
    # Clean temp_workspace
    if TEMP_DIR.exists():
        for item in TEMP_DIR.iterdir():
            try:
                if item.is_file():
                    size = item.stat().st_size
                    item.unlink()
                    total_freed += size
                elif item.is_dir():
                    size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                    shutil.rmtree(item, ignore_errors=True)
                    total_freed += size
            except Exception:
                pass
    
    # Clean T5 offload from system temp
    system_temp = Path(tempfile.gettempdir())
    for pattern in ["flux_t5_offload", "t5_offload", "offload_folder"]:
        target = system_temp / pattern
        if target.exists():
            try:
                size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
                shutil.rmtree(target, ignore_errors=True)
                total_freed += size
                print(f"[VoxAI] Cleaned T5 offload: {pattern}")
            except Exception:
                pass
    
    if total_freed > 0:
        print(f"[VoxAI] Fallback cleanup freed: {total_freed / 1024**3:.2f} GB")


def run_shutdown_cleanup():
    """Clean up temp files when application exits."""
    print("\n[VoxAI] Running shutdown cleanup...")
    
    try:
        from backend.cleanup import cleanup_on_shutdown
        freed = cleanup_on_shutdown()
        if freed > 0:
            print(f"[VoxAI] Cleaned {freed / 1024**3:.2f} GB of temp files")
    except ImportError:
        fallback_cleanup()
    except Exception as e:
        print(f"[VoxAI] Shutdown cleanup error: {e}")
    
    print("[VoxAI] Goodbye!")


def register_cleanup_handlers():
    """Register cleanup to run at exit."""
    atexit.register(run_shutdown_cleanup)


# -------------------------
# PyTorch/CUDA Setup
# -------------------------

def setup_torch():
    """Configure PyTorch for optimal performance with ZLUDA."""
    try:
        import torch
        
        # Disable Flash Attention backends (incompatible with ZLUDA)
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(True)
        
        # Log GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[VoxAI] GPU: {gpu_name} ({vram:.1f} GB VRAM)")
            
            if "ZLUDA" in gpu_name:
                print("[VoxAI] ZLUDA detected - Flash Attention disabled")
        else:
            print("[VoxAI] No GPU detected, using CPU")
            
    except ImportError:
        print("[VoxAI] PyTorch not yet installed")
    except Exception as e:
        print(f"[VoxAI] PyTorch setup warning: {e}")


# -------------------------
# Debug Setup
# -------------------------

def setup_debug():
    """Initialize the debug system."""
    try:
        from backend.debug import enable_debug, DebugLevel
        
        # Check for debug flag
        if "--debug" in sys.argv or os.environ.get("AI_GUI_DEBUG"):
            enable_debug(DebugLevel.VERBOSE)
            print("[VoxAI] Debug mode: VERBOSE")
        elif "--trace" in sys.argv:
            enable_debug(DebugLevel.TRACE)
            print("[VoxAI] Debug mode: TRACE")
        else:
            enable_debug(DebugLevel.INFO)
            
    except ImportError:
        pass  # Debug module not available


# -------------------------
# Application Entry Point
# -------------------------

def main():
    """Main entry point for the application."""
    print("=" * 50)
    print("  VoxAI Orchestrator - Starting...")
    print("=" * 50)
    print()
    
    # 1. Ensure dependencies are installed
    ensure_dependencies()
    
    # 2. Setup PyTorch/CUDA
    setup_torch()
    
    # 3. Setup debug system
    setup_debug()
    
    # 4. Run startup cleanup (remove leftover temp files)
    run_startup_cleanup()
    
    # 5. Register shutdown cleanup
    register_cleanup_handlers()
    
    # 6. Launch the application
    print("[VoxAI] Launching application...")
    print()
    
    from PySide6.QtWidgets import QApplication
    from main_window import MainWindow
    
    app = QApplication(sys.argv)
    
    # Set application info
    app.setApplicationName("VoxAI Orchestrator")
    app.setOrganizationName("AI_GUI")
    app.setApplicationVersion("0.1.0")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run the application
    exit_code = app.exec()
    
    # Exit
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
