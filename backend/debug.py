"""
AI_GUI Debug Utility
Comprehensive logging and debugging for all image generation processes.

Usage:
    from backend.debug import debug, DebugLevel, enable_debug, disable_debug
    
    debug.info("Loading model...")
    debug.gpu("VRAM usage: 4.2 GB")
    debug.step(1, 20, "Denoising...")
    debug.error("Something went wrong", exc_info=True)
"""
from __future__ import annotations

import os
import sys
import time
import functools
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Callable, Dict, List
from enum import Enum, auto
from contextlib import contextmanager
import threading

# Try to import torch for GPU debugging
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DebugLevel(Enum):
    """Debug verbosity levels."""
    OFF = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    VERBOSE = 4
    TRACE = 5  # Everything including tensor shapes, memory allocations


class DebugCategory(Enum):
    """Categories for filtering debug output."""
    GENERAL = auto()
    MODEL = auto()      # Model loading/unloading
    GPU = auto()        # VRAM, CUDA operations
    CPU = auto()        # RAM, CPU operations
    PIPELINE = auto()   # Pipeline assembly
    GENERATION = auto() # Inference steps
    TENSOR = auto()     # Tensor operations, shapes
    IO = auto()         # File I/O operations
    NETWORK = auto()    # HuggingFace downloads
    ATTENTION = auto()  # Attention mechanisms
    SCHEDULER = auto()  # Scheduler/sampler operations


# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


class DebugTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str, debugger: 'Debugger'):
        self.name = name
        self.debugger = debugger
        self.start_time = None
        self.start_vram = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_vram = torch.cuda.memory_allocated(0) / 1024**3
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        
        vram_delta = ""
        if self.start_vram is not None:
            torch.cuda.synchronize()
            end_vram = torch.cuda.memory_allocated(0) / 1024**3
            delta = end_vram - self.start_vram
            if abs(delta) > 0.01:  # Only show if > 10MB change
                sign = "+" if delta > 0 else ""
                vram_delta = f" | VRAM: {sign}{delta:.2f} GB"
        
        self.debugger._log(
            DebugLevel.INFO,
            DebugCategory.GENERAL,
            f"⏱ {self.name}: {elapsed:.2f}s{vram_delta}"
        )


class Debugger:
    """
    Central debug manager for AI_GUI.
    Singleton pattern - use get_debugger() or the global 'debug' instance.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.level = DebugLevel.INFO
        self.enabled_categories: set = set(DebugCategory)  # All enabled by default
        self.use_colors = True
        self.log_file: Optional[Path] = None
        self._file_handle = None
        self._step_times: List[float] = []
        self._last_step_time: Optional[float] = None
        
        # Detect if we're in a terminal that supports colors
        self.use_colors = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        
        # Windows color support
        if sys.platform == 'win32':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                self.use_colors = True
            except Exception:
                self.use_colors = False
    
    def configure(self,
                  level: DebugLevel = DebugLevel.INFO,
                  categories: Optional[set] = None,
                  log_file: Optional[str] = None,
                  use_colors: bool = True):
        """Configure the debugger."""
        self.level = level
        if categories is not None:
            self.enabled_categories = categories
        self.use_colors = use_colors
        
        if log_file:
            self.log_file = Path(log_file)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self.log_file, 'a', encoding='utf-8')
            self._log(DebugLevel.INFO, DebugCategory.GENERAL, 
                     f"=== Debug session started: {datetime.now().isoformat()} ===")
    
    def _colorize(self, text: str, color: str) -> str:
        """Apply color if enabled."""
        if self.use_colors:
            return f"{color}{text}{Colors.RESET}"
        return text
    
    def _format_message(self, level: DebugLevel, category: DebugCategory, message: str) -> str:
        """Format a debug message with timestamp and category."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Level indicators with colors
        level_indicators = {
            DebugLevel.ERROR: (self._colorize("ERROR", Colors.RED + Colors.BOLD), "❌"),
            DebugLevel.WARN: (self._colorize("WARN ", Colors.YELLOW), "⚠️"),
            DebugLevel.INFO: (self._colorize("INFO ", Colors.GREEN), "ℹ️"),
            DebugLevel.VERBOSE: (self._colorize("VERB ", Colors.CYAN), "📝"),
            DebugLevel.TRACE: (self._colorize("TRACE", Colors.GRAY), "🔍"),
        }
        
        # Category colors
        category_colors = {
            DebugCategory.GENERAL: Colors.WHITE,
            DebugCategory.MODEL: Colors.MAGENTA,
            DebugCategory.GPU: Colors.YELLOW,
            DebugCategory.CPU: Colors.BLUE,
            DebugCategory.PIPELINE: Colors.CYAN,
            DebugCategory.GENERATION: Colors.GREEN,
            DebugCategory.TENSOR: Colors.GRAY,
            DebugCategory.IO: Colors.BLUE,
            DebugCategory.NETWORK: Colors.CYAN,
            DebugCategory.ATTENTION: Colors.MAGENTA,
            DebugCategory.SCHEDULER: Colors.YELLOW,
        }
        
        level_str, _ = level_indicators.get(level, ("?????", "❓"))
        cat_color = category_colors.get(category, Colors.WHITE)
        cat_str = self._colorize(f"[{category.name:10}]", cat_color)
        time_str = self._colorize(timestamp, Colors.DIM)
        
        return f"{time_str} {level_str} {cat_str} {message}"
    
    def _log(self, level: DebugLevel, category: DebugCategory, message: str, exc_info: bool = False):
        """Internal logging method."""
        if level.value > self.level.value:
            return
        if category not in self.enabled_categories:
            return
        
        formatted = self._format_message(level, category, message)
        print(formatted)
        
        if exc_info:
            tb = traceback.format_exc()
            print(self._colorize(tb, Colors.RED))
        
        # Also write to file if configured
        if self._file_handle:
            # Strip ANSI codes for file
            import re
            clean = re.sub(r'\033\[[0-9;]*m', '', formatted)
            self._file_handle.write(clean + '\n')
            if exc_info:
                self._file_handle.write(traceback.format_exc() + '\n')
            self._file_handle.flush()
    
    # ==================== PUBLIC API ====================
    
    def error(self, message: str, exc_info: bool = False):
        """Log an error message."""
        self._log(DebugLevel.ERROR, DebugCategory.GENERAL, message, exc_info)
    
    def warn(self, message: str):
        """Log a warning message."""
        self._log(DebugLevel.WARN, DebugCategory.GENERAL, message)
    
    def info(self, message: str):
        """Log an info message."""
        self._log(DebugLevel.INFO, DebugCategory.GENERAL, message)
    
    def verbose(self, message: str):
        """Log a verbose message."""
        self._log(DebugLevel.VERBOSE, DebugCategory.GENERAL, message)
    
    def trace(self, message: str):
        """Log a trace message (most verbose)."""
        self._log(DebugLevel.TRACE, DebugCategory.GENERAL, message)
    
    # ==================== CATEGORY-SPECIFIC ====================
    
    def model(self, message: str, level: DebugLevel = DebugLevel.INFO):
        """Log model loading/unloading operations."""
        self._log(level, DebugCategory.MODEL, message)
    
    def gpu(self, message: str = None, level: DebugLevel = DebugLevel.INFO):
        """Log GPU/VRAM information. If no message, logs current VRAM stats."""
        if message is None and TORCH_AVAILABLE and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - allocated
            message = f"VRAM: {allocated:.2f}/{total:.2f} GB used | {free:.2f} GB free | {reserved:.2f} GB reserved"
        self._log(level, DebugCategory.GPU, message or "GPU info unavailable")
    
    def cpu(self, message: str = None, level: DebugLevel = DebugLevel.INFO):
        """Log CPU/RAM information."""
        if message is None:
            try:
                import psutil
                ram = psutil.virtual_memory()
                message = f"RAM: {ram.used/1024**3:.1f}/{ram.total/1024**3:.1f} GB ({ram.percent}%)"
            except ImportError:
                message = "RAM info unavailable (psutil not installed)"
        self._log(level, DebugCategory.CPU, message)
    
    def pipeline(self, message: str, level: DebugLevel = DebugLevel.INFO):
        """Log pipeline assembly operations."""
        self._log(level, DebugCategory.PIPELINE, message)
    
    def generation(self, message: str, level: DebugLevel = DebugLevel.INFO):
        """Log generation/inference operations."""
        self._log(level, DebugCategory.GENERATION, message)
    
    def tensor(self, name: str, t: Any, level: DebugLevel = DebugLevel.VERBOSE):
        """Log tensor information (shape, dtype, device)."""
        if TORCH_AVAILABLE and isinstance(t, torch.Tensor):
            message = f"{name}: shape={list(t.shape)}, dtype={t.dtype}, device={t.device}"
            if t.numel() < 10:
                message += f", values={t.tolist()}"
        else:
            message = f"{name}: {type(t).__name__}"
        self._log(level, DebugCategory.TENSOR, message)
    
    def io(self, message: str, level: DebugLevel = DebugLevel.INFO):
        """Log file I/O operations."""
        self._log(level, DebugCategory.IO, message)
    
    def network(self, message: str, level: DebugLevel = DebugLevel.INFO):
        """Log network operations (downloads, etc.)."""
        self._log(level, DebugCategory.NETWORK, message)
    
    def attention(self, message: str, level: DebugLevel = DebugLevel.VERBOSE):
        """Log attention mechanism details."""
        self._log(level, DebugCategory.ATTENTION, message)
    
    def scheduler(self, message: str, level: DebugLevel = DebugLevel.VERBOSE):
        """Log scheduler/sampler operations."""
        self._log(level, DebugCategory.SCHEDULER, message)
    
    # ==================== SPECIAL METHODS ====================
    
    def step(self, current: int, total: int, message: str = ""):
        """Log a generation step with timing."""
        now = time.perf_counter()
        
        step_time = ""
        if self._last_step_time is not None:
            elapsed = now - self._last_step_time
            self._step_times.append(elapsed)
            avg_time = sum(self._step_times) / len(self._step_times)
            remaining = (total - current) * avg_time
            step_time = f" | {elapsed:.2f}s/step | ETA: {remaining:.1f}s"
        
        self._last_step_time = now
        
        # Progress bar
        pct = current / total
        bar_width = 20
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        msg = f"Step {current}/{total} [{bar}] {pct*100:.0f}%{step_time}"
        if message:
            msg += f" | {message}"
        
        self._log(DebugLevel.INFO, DebugCategory.GENERATION, msg)
    
    def reset_step_timer(self):
        """Reset step timing for a new generation."""
        self._step_times = []
        self._last_step_time = None
    
    def timer(self, name: str) -> DebugTimer:
        """Create a context manager for timing an operation."""
        return DebugTimer(name, self)
    
    def separator(self, title: str = ""):
        """Print a visual separator."""
        width = 60
        if title:
            padding = (width - len(title) - 2) // 2
            line = "=" * padding + f" {title} " + "=" * padding
        else:
            line = "=" * width
        self._log(DebugLevel.INFO, DebugCategory.GENERAL, self._colorize(line, Colors.BOLD))
    
    def system_info(self):
        """Log comprehensive system information."""
        self.separator("SYSTEM INFO")
        
        # Python
        self.info(f"Python: {sys.version}")
        
        # PyTorch
        if TORCH_AVAILABLE:
            self.info(f"PyTorch: {torch.__version__}")
            if torch.cuda.is_available():
                self.info(f"CUDA: {torch.version.cuda}")
                self.info(f"GPU: {torch.cuda.get_device_name(0)}")
                props = torch.cuda.get_device_properties(0)
                self.info(f"GPU Memory: {props.total_memory / 1024**3:.1f} GB")
                self.info(f"GPU Compute: SM{props.major}{props.minor}")
                
                # Check for ZLUDA
                if "ZLUDA" in torch.cuda.get_device_name(0):
                    self.warn("ZLUDA detected - Flash Attention disabled")
        else:
            self.warn("PyTorch not available")
        
        # RAM
        self.cpu()
        
        # Diffusers
        try:
            import diffusers
            self.info(f"Diffusers: {diffusers.__version__}")
        except ImportError:
            self.warn("Diffusers not available")
        
        # Transformers
        try:
            import transformers
            self.info(f"Transformers: {transformers.__version__}")
        except ImportError:
            self.warn("Transformers not available")
        
        self.separator()
    
    def generation_config(self, config: Any):
        """Log generation configuration."""
        self.separator("GENERATION CONFIG")
        if hasattr(config, '__dict__'):
            for key, value in config.__dict__.items():
                if not key.startswith('_'):
                    self.info(f"  {key}: {value}")
        elif isinstance(config, dict):
            for key, value in config.items():
                self.info(f"  {key}: {value}")
        self.separator()
    
    def close(self):
        """Close the debug file handle."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None


# ==================== GLOBAL INSTANCE ====================

debug = Debugger()


def get_debugger() -> Debugger:
    """Get the global debugger instance."""
    return debug


def enable_debug(level: DebugLevel = DebugLevel.VERBOSE, log_file: Optional[str] = None):
    """Enable debugging with specified level."""
    debug.configure(level=level, log_file=log_file)
    debug.system_info()


def disable_debug():
    """Disable debugging."""
    debug.configure(level=DebugLevel.OFF)


# ==================== DECORATORS ====================

def log_function(level: DebugLevel = DebugLevel.VERBOSE):
    """Decorator to log function calls."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__
            debug._log(level, DebugCategory.GENERAL, f"→ {func_name}()")
            try:
                with debug.timer(func_name):
                    result = func(*args, **kwargs)
                debug._log(level, DebugCategory.GENERAL, f"← {func_name}() completed")
                return result
            except Exception as e:
                debug.error(f"✗ {func_name}() failed: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


def log_gpu_memory(func: Callable) -> Callable:
    """Decorator to log GPU memory before and after a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            debug.gpu(f"Before {func.__name__}:")
        result = func(*args, **kwargs)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            debug.gpu(f"After {func.__name__}:")
        return result
    return wrapper


# ==================== QUICK TEST ====================

if __name__ == "__main__":
    # Quick test of the debug system
    enable_debug(DebugLevel.TRACE)
    
    debug.info("This is an info message")
    debug.warn("This is a warning")
    debug.error("This is an error")
    debug.verbose("This is verbose")
    debug.trace("This is trace level")
    
    debug.model("Loading model: test.safetensors")
    debug.gpu()
    debug.cpu()
    
    debug.separator("TEST SECTION")
    
    with debug.timer("Test operation"):
        time.sleep(0.5)
    
    debug.step(1, 10, "First step")
    debug.step(2, 10, "Second step")
    
    @log_function()
    def example_function():
        time.sleep(0.1)
        return "done"
    
    example_function()
    
    debug.close()
