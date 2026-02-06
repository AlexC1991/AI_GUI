"""
VOX-AI Backend Patcher

This script REPLACES the bundled llama.cpp DLLs inside llama-cpp-python
with YOUR custom optimized DLLs from VoxAI_Chat_API.

Run this ONCE after installing llama-cpp-python, and again if you update
the pip package.

Usage:
    python patch_llama_cpp.py
"""

import os
import sys
import shutil
from pathlib import Path

def find_llama_cpp_package():
    """Find the llama_cpp package in site-packages"""
    try:
        import llama_cpp
        return Path(llama_cpp.__file__).parent
    except ImportError:
        # Search manually
        for path in sys.path:
            candidate = Path(path) / "llama_cpp"
            if candidate.exists() and (candidate / "__init__.py").exists():
                return candidate
    return None


def main():
    print("=" * 60)
    print("  VOX-AI Backend Patcher")
    print("  Replaces llama-cpp-python DLLs with your optimized ones")
    print("=" * 60)
    print()
    
    # Find source DLLs (VoxAI_Chat_API)
    script_dir = Path(__file__).parent.absolute()
    vox_api_dir = script_dir / "VoxAI_Chat_API"
    
    if not vox_api_dir.exists():
        print(f"[ERROR] VoxAI_Chat_API not found at: {vox_api_dir}")
        print("        Make sure this script is in your AI_GUI root folder.")
        return 1
    
    print(f"[INFO] Source: {vox_api_dir}")
    
    # Find target (llama_cpp package)
    llama_pkg = find_llama_cpp_package()
    
    if not llama_pkg:
        print("[ERROR] llama-cpp-python not installed!")
        print("        Install it first: pip install llama-cpp-python")
        return 1
    
    print(f"[INFO] Target: {llama_pkg}")
    print()
    
    # Determine target lib folder
    # Newer versions use lib/ subfolder, older ones put DLLs directly in package root
    lib_folder = llama_pkg / "lib"
    if not lib_folder.exists():
        lib_folder = llama_pkg  # Fallback to package root
    
    print(f"[INFO] DLL folder: {lib_folder}")
    print()
    
    # Files to copy
    dll_files = [
        "llama.dll",
        "ggml.dll",
        "ggml-base.dll",
        "ggml-cpu.dll",
        "ggml-vulkan.dll",
        "ggml-cuda.dll",  # May not exist, that's OK
    ]
    
    # Also copy any other ggml-*.dll files
    for f in vox_api_dir.glob("ggml-*.dll"):
        if f.name not in dll_files:
            dll_files.append(f.name)
    
    print("[INFO] Files to patch:")
    for f in dll_files:
        src = vox_api_dir / f
        if src.exists():
            print(f"       ✓ {f}")
        else:
            print(f"       - {f} (not in source, skipping)")
    print()
    
    # Create backup
    backup_dir = lib_folder / "backup_original"
    if not backup_dir.exists():
        backup_dir.mkdir()
        print(f"[INFO] Created backup folder: {backup_dir}")
    
    # Copy files
    copied = 0
    for filename in dll_files:
        src = vox_api_dir / filename
        dst = lib_folder / filename
        
        if not src.exists():
            continue
        
        # Backup original if exists and not already backed up
        if dst.exists():
            backup_path = backup_dir / filename
            if not backup_path.exists():
                print(f"[BACKUP] {filename}")
                shutil.copy2(dst, backup_path)
        
        # Copy new file
        print(f"[COPY] {filename}")
        shutil.copy2(src, dst)
        copied += 1
    
    print()
    print("=" * 60)
    print(f"  ✓ Patched {copied} files")
    print()
    print("  Your llama-cpp-python now uses your custom optimized DLLs!")
    print()
    print("  To restore originals, copy files from:")
    print(f"    {backup_dir}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
