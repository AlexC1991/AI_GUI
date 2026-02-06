"""
VOXAI BOOTSTRAP - IMPORT THIS FIRST!

This module MUST be imported at the very top of main.py before ANY other imports.
It configures temp directories and lies to Windows about C: drive space.

Usage in main.py:
    # Line 1 - before ANYTHING else
    import bootstrap  # noqa: F401
    
    # Now safe to import everything else
    import sys
    from PySide6.QtWidgets import QApplication
    ...
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from collections import namedtuple

# ============================================
# CONFIGURATION
# ============================================

# Preferred drive for temp/cache
PREFERRED_TEMP_DRIVE = "A:"

# Minimum free space required (MB)
MIN_FREE_SPACE_MB = 500

# C: drive fake size - report this to make Windows think C: is full
# 250MB is the magic number that finally convinces Windows to stop using it
C_DRIVE_FAKE_FREE_MB = 250

# Subdirectory names
TEMP_SUBDIR = "AI_Temp"
CACHE_SUBDIR = "AI_Cache"

# ============================================
# DRIVE DETECTION
# ============================================

def _real_disk_usage(path):
    """Get REAL disk usage (before we patch shutil)."""
    try:
        return shutil._original_disk_usage(path)
    except AttributeError:
        return shutil.disk_usage(path)

def _get_drive_free_mb(drive: str) -> float:
    """Get actual free space on a drive in MB."""
    try:
        total, used, free = _real_disk_usage(drive)
        return free / (1024 ** 2)
    except:
        return 0

def _find_best_drive() -> str:
    """Find best drive for temp files, avoiding C:."""
    # Try preferred drive first
    if PREFERRED_TEMP_DRIVE:
        free = _get_drive_free_mb(PREFERRED_TEMP_DRIVE)
        if free >= MIN_FREE_SPACE_MB:
            return PREFERRED_TEMP_DRIVE
        print(f"[Bootstrap] Warning: {PREFERRED_TEMP_DRIVE} has only {free:.0f}MB free")
    
    # Find drive with most space (excluding C:)
    best_drive = None
    best_space = 0
    
    for letter in "DEFGHIJKLMNOPQRSTUVWXYZAB":
        drive = f"{letter}:"
        if drive.upper() == "C:":
            continue
        
        free = _get_drive_free_mb(drive)
        if free > best_space:
            best_space = free
            best_drive = drive
    
    if best_drive and best_space >= MIN_FREE_SPACE_MB:
        return best_drive
    
    # Last resort: app directory
    app_dir = Path(__file__).parent
    print(f"[Bootstrap] No suitable drive, using: {app_dir}")
    return str(app_dir)

# ============================================
# SETUP DIRECTORIES
# ============================================

_BASE_DRIVE = _find_best_drive()

if len(_BASE_DRIVE) == 2 and _BASE_DRIVE[1] == ":":
    TEMP_BASE = Path(f"{_BASE_DRIVE}/{TEMP_SUBDIR}")
    CACHE_BASE = Path(f"{_BASE_DRIVE}/{CACHE_SUBDIR}")
else:
    TEMP_BASE = Path(_BASE_DRIVE) / TEMP_SUBDIR
    CACHE_BASE = Path(_BASE_DRIVE) / CACHE_SUBDIR

# Create all directories
DIRS = {
    "temp": TEMP_BASE / "general",
    "temp_imagegen": TEMP_BASE / "image_gen",
    "cache_hf": CACHE_BASE / "huggingface",
    "cache_torch": CACHE_BASE / "torch",
}

for name, path in DIRS.items():
    path.mkdir(parents=True, exist_ok=True)

# ============================================
# ENVIRONMENT VARIABLES
# ============================================

# Windows temp
os.environ["TEMP"] = str(DIRS["temp"])
os.environ["TMP"] = str(DIRS["temp"])
os.environ["TMPDIR"] = str(DIRS["temp"])
os.environ["TEMPDIR"] = str(DIRS["temp"])

# HuggingFace - ALL the cache vars
os.environ["HF_HOME"] = str(DIRS["cache_hf"])
os.environ["HF_DATASETS_CACHE"] = str(DIRS["cache_hf"] / "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(DIRS["cache_hf"] / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(DIRS["cache_hf"] / "transformers")
os.environ["DIFFUSERS_CACHE"] = str(DIRS["cache_hf"] / "diffusers")

# Torch
os.environ["TORCH_HOME"] = str(DIRS["cache_torch"])
os.environ["TORCH_EXTENSIONS_DIR"] = str(DIRS["cache_torch"] / "extensions")

# XDG (some libs check these on Windows)
os.environ["XDG_CACHE_HOME"] = str(CACHE_BASE)
os.environ["XDG_DATA_HOME"] = str(CACHE_BASE)

# Disable telemetry
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["WANDB_DISABLED"] = "true"

# PyTorch CUDA memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================
# MONKEY-PATCH TEMPFILE
# ============================================

tempfile.tempdir = str(DIRS["temp"])

_original_gettempdir = tempfile.gettempdir
def _patched_gettempdir():
    return str(DIRS["temp"])
tempfile.gettempdir = _patched_gettempdir

if hasattr(tempfile, '_get_default_tempdir'):
    tempfile._get_default_tempdir = lambda: str(DIRS["temp"])

# ============================================
# MONKEY-PATCH SHUTIL.DISK_USAGE
# Lie about C: drive to make it look full
# ============================================

shutil._original_disk_usage = shutil.disk_usage

def _patched_disk_usage(path):
    """Return fake low space for C: drive."""
    result = shutil._original_disk_usage(path)
    
    path_str = str(path).upper()
    is_c_drive = (
        path_str.startswith("C:") or 
        path_str.startswith("C\\") or 
        path_str.startswith("C/") or
        path_str == "C" or
        "\\USERS\\" in path_str.upper() or
        "/USERS/" in path_str.upper() or
        "\\APPDATA\\" in path_str.upper() or
        "\\WINDOWS\\" in path_str.upper()
    )
    
    if is_c_drive:
        fake_free = C_DRIVE_FAKE_FREE_MB * 1024 * 1024
        DiskUsage = namedtuple('usage', ['total', 'used', 'free'])
        return DiskUsage(result.total, result.total - fake_free, fake_free)
    
    return result

shutil.disk_usage = _patched_disk_usage

# ============================================
# MONKEY-PATCH OS.PATH FUNCTIONS
# ============================================

_original_expanduser = os.path.expanduser

def _patched_expanduser(path):
    """Redirect ~ away from C: drive."""
    result = _original_expanduser(path)
    
    # If it resolved to C:\Users\..., redirect to our cache
    if result.lower().startswith("c:\\users\\") or result.lower().startswith("c:/users/"):
        # Extract the part after the username
        parts = result.split(os.sep)
        if len(parts) > 3:
            # Rebuild path under our cache
            subpath = os.sep.join(parts[3:])
            result = str(CACHE_BASE / "user_home" / subpath)
            os.makedirs(os.path.dirname(result), exist_ok=True)
    
    return result

os.path.expanduser = _patched_expanduser

# ============================================
# REPORT
# ============================================

_free_space = _get_drive_free_mb(_BASE_DRIVE)
print(f"[Bootstrap] ═══════════════════════════════════════════")
print(f"[Bootstrap] Temp/Cache Drive: {_BASE_DRIVE} ({_free_space:.0f}MB free)")
print(f"[Bootstrap] Temp Directory:   {TEMP_BASE}")
print(f"[Bootstrap] Cache Directory:  {CACHE_BASE}")
print(f"[Bootstrap] C: drive reports: {C_DRIVE_FAKE_FREE_MB}MB (fake - discourages use)")
print(f"[Bootstrap] ═══════════════════════════════════════════")

# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_temp_dir() -> Path:
    """Get the temp directory."""
    return DIRS["temp"]

def get_cache_dir() -> Path:
    """Get the cache directory."""
    return CACHE_BASE

def clear_temp():
    """Clear temp files."""
    cleared = 0
    for temp_dir in [DIRS["temp"], DIRS["temp_imagegen"]]:
        if temp_dir.exists():
            for item in temp_dir.iterdir():
                try:
                    if item.is_file():
                        cleared += item.stat().st_size
                        item.unlink()
                    elif item.is_dir():
                        for f in item.rglob('*'):
                            if f.is_file():
                                cleared += f.stat().st_size
                        shutil.rmtree(item)
                except:
                    pass
    
    if cleared > 0:
        print(f"[Bootstrap] Cleared {cleared / 1024 / 1024:.1f}MB temp files")
    return cleared

def verify():
    """Print verification of all paths."""
    print("\n[Bootstrap] PATH VERIFICATION:")
    print(f"  tempfile.gettempdir() = {tempfile.gettempdir()}")
    print(f"  os.environ['TEMP']    = {os.environ.get('TEMP')}")
    print(f"  os.environ['HF_HOME'] = {os.environ.get('HF_HOME')}")
    print(f"  shutil.disk_usage('C:').free = {shutil.disk_usage('C:').free / 1024 / 1024:.0f}MB (should be ~{C_DRIVE_FAKE_FREE_MB}MB)")
    print()
