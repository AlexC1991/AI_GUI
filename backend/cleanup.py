"""
AI_GUI Cleanup Utility
Removes temporary files from image generation, HuggingFace cache, and disk offload.

This module handles cleanup of:
- T5 GGUF disk offload files (can be 4-10GB+)
- HuggingFace hub cache (downloaded models)
- PyTorch temp files
- MIOpen cache (AMD GPU)
- Diffusers cache
- Temp workspace files

Usage:
    from backend.cleanup import cleanup_all, cleanup_on_startup, cleanup_on_shutdown
    
    # At startup
    cleanup_on_startup()
    
    # At shutdown
    cleanup_on_shutdown()
    
    # Manual full cleanup
    cleanup_all(include_hf_cache=True)
"""

import os
import sys
import shutil
import tempfile
import atexit
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, timedelta

# Try to import debug
try:
    from backend.debug import debug
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False
    class DebugStub:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    debug = DebugStub()


class CleanupManager:
    """Manages cleanup of temporary files created during image generation."""
    
    # Known temp directories that can accumulate large files
    TEMP_PATTERNS = [
        # T5 GGUF disk offload (the big one - 4-10GB)
        "flux_t5_offload",
        "t5_offload",
        "offload_folder",
        "accelerate_offload",
        
        # Diffusers temp
        "diffusers_*",
        "transformers_*",
        
        # PyTorch
        "torch_*",
        "pytorch_*",
        
        # General AI temp
        "ai_gui_*",
        "voxai_*",
    ]
    
    # File extensions to clean in temp
    TEMP_EXTENSIONS = [
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".gguf",
        ".tmp",
        ".temp",
        ".cache",
    ]
    
    def __init__(self, app_dir: Optional[Path] = None):
        """
        Initialize cleanup manager.
        
        Args:
            app_dir: Application root directory. If None, uses current working directory.
        """
        self.app_dir = Path(app_dir) if app_dir else Path.cwd()
        
        # Define cleanup locations
        self.temp_workspace = self.app_dir / "temp_workspace"
        self.hf_cache = self.app_dir / "models" / "hf_cache"
        self.miopen_cache = self.app_dir / "models" / "miopen_cache"
        
        # System temp directory
        self.system_temp = Path(tempfile.gettempdir())
        
        # Track what we've cleaned
        self.last_cleanup: Optional[datetime] = None
        self.total_cleaned_bytes: int = 0
    
    def get_size_str(self, size_bytes: int) -> str:
        """Convert bytes to human-readable string."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 ** 2:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 ** 3:
            return f"{size_bytes / 1024**2:.1f} MB"
        else:
            return f"{size_bytes / 1024**3:.2f} GB"
    
    def get_dir_size(self, path: Path) -> int:
        """Get total size of a directory."""
        total = 0
        try:
            if path.is_file():
                return path.stat().st_size
            for item in path.rglob("*"):
                if item.is_file():
                    try:
                        total += item.stat().st_size
                    except (OSError, PermissionError):
                        pass
        except (OSError, PermissionError):
            pass
        return total
    
    def safe_remove_file(self, path: Path) -> Tuple[bool, int]:
        """
        Safely remove a file.
        
        Returns:
            Tuple of (success, bytes_freed)
        """
        try:
            if path.exists() and path.is_file():
                size = path.stat().st_size
                path.unlink()
                debug.io(f"Removed file: {path} ({self.get_size_str(size)})")
                return True, size
        except (OSError, PermissionError) as e:
            debug.warn(f"Could not remove {path}: {e}")
        return False, 0
    
    def safe_remove_dir(self, path: Path) -> Tuple[bool, int]:
        """
        Safely remove a directory and all contents.
        
        Returns:
            Tuple of (success, bytes_freed)
        """
        try:
            if path.exists() and path.is_dir():
                size = self.get_dir_size(path)
                shutil.rmtree(path, ignore_errors=True)
                debug.io(f"Removed directory: {path} ({self.get_size_str(size)})")
                return True, size
        except (OSError, PermissionError) as e:
            debug.warn(f"Could not remove {path}: {e}")
        return False, 0
    
    def cleanup_temp_workspace(self) -> int:
        """
        Clean the app's temp_workspace directory.
        
        Returns:
            Bytes freed
        """
        debug.separator("CLEANUP: TEMP WORKSPACE")
        total_freed = 0
        
        if self.temp_workspace.exists():
            size = self.get_dir_size(self.temp_workspace)
            debug.info(f"Temp workspace size: {self.get_size_str(size)}")
            
            # Remove all contents but keep the directory
            for item in self.temp_workspace.iterdir():
                if item.is_file():
                    _, freed = self.safe_remove_file(item)
                    total_freed += freed
                elif item.is_dir():
                    _, freed = self.safe_remove_dir(item)
                    total_freed += freed
            
            debug.info(f"Freed from temp_workspace: {self.get_size_str(total_freed)}")
        else:
            debug.info("Temp workspace does not exist")
            self.temp_workspace.mkdir(parents=True, exist_ok=True)
        
        return total_freed
    
    def cleanup_system_temp(self) -> int:
        """
        Clean AI-related files from system temp directory.
        
        Returns:
            Bytes freed
        """
        debug.separator("CLEANUP: SYSTEM TEMP")
        total_freed = 0
        
        debug.info(f"Scanning system temp: {self.system_temp}")
        
        # Clean known temp directories
        for pattern in self.TEMP_PATTERNS:
            if "*" in pattern:
                # Glob pattern
                base_pattern = pattern.replace("*", "")
                for item in self.system_temp.iterdir():
                    if item.name.startswith(base_pattern) or item.name.endswith(base_pattern.strip("*")):
                        if item.is_dir():
                            _, freed = self.safe_remove_dir(item)
                            total_freed += freed
                        else:
                            _, freed = self.safe_remove_file(item)
                            total_freed += freed
            else:
                # Exact match
                target = self.system_temp / pattern
                if target.exists():
                    if target.is_dir():
                        _, freed = self.safe_remove_dir(target)
                        total_freed += freed
                    else:
                        _, freed = self.safe_remove_file(target)
                        total_freed += freed
        
        # Clean large temp files with known extensions
        for ext in self.TEMP_EXTENSIONS:
            for item in self.system_temp.glob(f"*{ext}"):
                if item.is_file():
                    size = item.stat().st_size
                    # Only clean files > 100MB
                    if size > 100 * 1024 * 1024:
                        _, freed = self.safe_remove_file(item)
                        total_freed += freed
        
        debug.info(f"Freed from system temp: {self.get_size_str(total_freed)}")
        return total_freed
    
    def cleanup_t5_offload(self) -> int:
        """
        Specifically clean T5 GGUF disk offload files.
        This is often the biggest culprit (4-10GB).
        
        Returns:
            Bytes freed
        """
        debug.separator("CLEANUP: T5 OFFLOAD")
        total_freed = 0
        
        # Check common offload locations
        offload_locations = [
            self.system_temp / "flux_t5_offload",
            self.system_temp / "t5_offload",
            self.system_temp / "offload_folder",
            self.temp_workspace / "offload_t5",
            self.app_dir / "offload_folder",
            Path.home() / ".cache" / "huggingface" / "accelerate" / "offload",
        ]
        
        for location in offload_locations:
            if location.exists():
                size = self.get_dir_size(location)
                debug.info(f"Found T5 offload: {location} ({self.get_size_str(size)})")
                _, freed = self.safe_remove_dir(location)
                total_freed += freed
        
        debug.info(f"Freed from T5 offload: {self.get_size_str(total_freed)}")
        return total_freed
    
    def cleanup_hf_cache(self, max_age_days: int = 7) -> int:
        """
        Clean old HuggingFace cache files.
        
        Args:
            max_age_days: Only remove files older than this many days
            
        Returns:
            Bytes freed
        """
        debug.separator("CLEANUP: HUGGINGFACE CACHE")
        total_freed = 0
        
        # HF cache locations
        hf_locations = [
            self.hf_cache,
            Path.home() / ".cache" / "huggingface",
            Path(os.environ.get("HF_HOME", "")) if os.environ.get("HF_HOME") else None,
        ]
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        for location in hf_locations:
            if location and location.exists():
                size = self.get_dir_size(location)
                debug.info(f"HF cache: {location} ({self.get_size_str(size)})")
                
                # Only clean old blob files, not the entire cache
                hub_dir = location / "hub"
                if hub_dir.exists():
                    for blob_dir in hub_dir.glob("models--*/blobs"):
                        for blob_file in blob_dir.iterdir():
                            if blob_file.is_file():
                                try:
                                    mtime = datetime.fromtimestamp(blob_file.stat().st_mtime)
                                    if mtime < cutoff_time:
                                        _, freed = self.safe_remove_file(blob_file)
                                        total_freed += freed
                                except (OSError, PermissionError):
                                    pass
        
        debug.info(f"Freed from HF cache: {self.get_size_str(total_freed)}")
        return total_freed
    
    def cleanup_miopen_cache(self) -> int:
        """
        Clean MIOpen cache (AMD GPU kernel cache).
        This can grow large but is usually rebuilt quickly.
        
        Returns:
            Bytes freed
        """
        debug.separator("CLEANUP: MIOPEN CACHE")
        total_freed = 0
        
        if self.miopen_cache.exists():
            size = self.get_dir_size(self.miopen_cache)
            debug.info(f"MIOpen cache: {self.miopen_cache} ({self.get_size_str(size)})")
            
            # Only clean if > 1GB (it's useful to keep some cached kernels)
            if size > 1024 ** 3:
                _, freed = self.safe_remove_dir(self.miopen_cache)
                total_freed += freed
                self.miopen_cache.mkdir(parents=True, exist_ok=True)
        
        debug.info(f"Freed from MIOpen cache: {self.get_size_str(total_freed)}")
        return total_freed
    
    def cleanup_torch_cache(self) -> int:
        """
        Clean PyTorch compilation cache.
        
        Returns:
            Bytes freed
        """
        debug.separator("CLEANUP: PYTORCH CACHE")
        total_freed = 0
        
        torch_cache_locations = [
            Path.home() / ".cache" / "torch",
            Path.home() / ".cache" / "torch_extensions",
            self.system_temp / "torch_extensions",
        ]
        
        for location in torch_cache_locations:
            if location.exists():
                size = self.get_dir_size(location)
                # Only clean if > 500MB
                if size > 500 * 1024 * 1024:
                    debug.info(f"PyTorch cache: {location} ({self.get_size_str(size)})")
                    _, freed = self.safe_remove_dir(location)
                    total_freed += freed
        
        debug.info(f"Freed from PyTorch cache: {self.get_size_str(total_freed)}")
        return total_freed
    
    def cleanup_all(self, include_hf_cache: bool = False, include_miopen: bool = False) -> int:
        """
        Run all cleanup operations.
        
        Args:
            include_hf_cache: Also clean HuggingFace cache (will need re-download)
            include_miopen: Also clean MIOpen cache (will need recompile)
            
        Returns:
            Total bytes freed
        """
        debug.separator("FULL CLEANUP STARTING")
        print("[Cleanup] Starting full cleanup...")
        
        total_freed = 0
        
        # Always clean these
        total_freed += self.cleanup_temp_workspace()
        total_freed += self.cleanup_t5_offload()
        total_freed += self.cleanup_system_temp()
        total_freed += self.cleanup_torch_cache()
        
        # Optional cleanups
        if include_hf_cache:
            total_freed += self.cleanup_hf_cache()
        
        if include_miopen:
            total_freed += self.cleanup_miopen_cache()
        
        self.last_cleanup = datetime.now()
        self.total_cleaned_bytes += total_freed
        
        debug.separator("CLEANUP COMPLETE")
        debug.info(f"Total freed: {self.get_size_str(total_freed)}")
        debug.info(f"Session total cleaned: {self.get_size_str(self.total_cleaned_bytes)}")
        print(f"[Cleanup] Freed {self.get_size_str(total_freed)}")
        
        return total_freed
    
    def get_cleanup_status(self) -> dict:
        """Get information about cleanable space."""
        status = {
            "temp_workspace": 0,
            "system_temp": 0,
            "t5_offload": 0,
            "hf_cache": 0,
            "miopen_cache": 0,
            "torch_cache": 0,
            "total": 0,
        }
        
        # Temp workspace
        if self.temp_workspace.exists():
            status["temp_workspace"] = self.get_dir_size(self.temp_workspace)
        
        # T5 offload
        for loc in [self.system_temp / "flux_t5_offload", self.system_temp / "t5_offload"]:
            if loc.exists():
                status["t5_offload"] += self.get_dir_size(loc)
        
        # HF cache
        hf_dir = Path.home() / ".cache" / "huggingface"
        if hf_dir.exists():
            status["hf_cache"] = self.get_dir_size(hf_dir)
        
        # MIOpen
        if self.miopen_cache.exists():
            status["miopen_cache"] = self.get_dir_size(self.miopen_cache)
        
        # Total
        status["total"] = sum(v for k, v in status.items() if k != "total")
        
        return status


# ==================== GLOBAL INSTANCE ====================

_cleanup_manager: Optional[CleanupManager] = None


def get_cleanup_manager(app_dir: Optional[Path] = None) -> CleanupManager:
    """Get or create the global cleanup manager."""
    global _cleanup_manager
    if _cleanup_manager is None:
        _cleanup_manager = CleanupManager(app_dir)
    return _cleanup_manager


def cleanup_on_startup(app_dir: Optional[Path] = None) -> int:
    """
    Run cleanup at application startup.
    Cleans temp files from previous sessions.
    
    Returns:
        Bytes freed
    """
    print("[Cleanup] Running startup cleanup...")
    manager = get_cleanup_manager(app_dir)
    return manager.cleanup_all(include_hf_cache=False, include_miopen=False)


def cleanup_on_shutdown() -> int:
    """
    Run cleanup at application shutdown.
    Cleans temp files from current session.
    
    Returns:
        Bytes freed
    """
    print("[Cleanup] Running shutdown cleanup...")
    manager = get_cleanup_manager()
    return manager.cleanup_all(include_hf_cache=False, include_miopen=False)


def cleanup_all(include_hf_cache: bool = False) -> int:
    """
    Run full cleanup.
    
    Args:
        include_hf_cache: Also clean HuggingFace cache
        
    Returns:
        Bytes freed
    """
    manager = get_cleanup_manager()
    return manager.cleanup_all(include_hf_cache=include_hf_cache)


def register_shutdown_cleanup():
    """Register cleanup to run at program exit."""
    atexit.register(cleanup_on_shutdown)


def get_cleanable_space() -> dict:
    """Get information about space that can be cleaned."""
    manager = get_cleanup_manager()
    return manager.get_cleanup_status()


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI_GUI Cleanup Utility")
    parser.add_argument("--all", action="store_true", help="Full cleanup including HF cache")
    parser.add_argument("--status", action="store_true", help="Show cleanable space")
    parser.add_argument("--app-dir", type=str, help="Application directory", default=".")
    
    args = parser.parse_args()
    
    manager = CleanupManager(Path(args.app_dir))
    
    if args.status:
        status = manager.get_cleanup_status()
        print("\n=== Cleanable Space ===")
        for key, value in status.items():
            print(f"  {key}: {manager.get_size_str(value)}")
    else:
        freed = manager.cleanup_all(include_hf_cache=args.all)
        print(f"\nTotal freed: {manager.get_size_str(freed)}")
