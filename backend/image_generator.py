"""
Image Generator - Manager/Orchestrator

This script ONLY handles:
- Memory Management (VRAM tracking, cleanup)
- GPU Control (device selection, dtype)
- CPU Control (offloading decisions)
- Disk Management (temp files, cache paths)
- Output (saving images, metadata)
- Model Detection → Routes to correct pipeline

The actual model loading and inference is handled by:
- pipelines/sd15_pipeline.py
- pipelines/sdxl_pipeline.py
- pipelines/flux_pipeline.py
"""

import os
import sys
import gc
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List, Union
from datetime import datetime
from enum import Enum


# ============================================
# TEMP/CACHE DIRECTORY CONFIGURATION
# ============================================
# 
# IMPORTANT: Windows ignores environment variables for temp directories!
# Python's tempfile module and many libraries hardcode C:\Users\...\AppData\Local\Temp
# 
# This section MUST run before importing torch/diffusers/transformers.
# We monkey-patch Python's tempfile module to force our preferred location.
#
# Why A: drive? On systems with limited C: drive space (SSDs), we need to
# redirect all temp files and model caches to a drive with more space.
# Change TEMP_DRIVE below to match your system.
# ============================================

def _setup_temp_directories():
    """
    Configure temp/cache directories BEFORE importing heavy libraries.
    
    This function:
    1. Detects available drives and space
    2. Sets environment variables (for libraries that respect them)
    3. Monkey-patches tempfile module (for libraries that don't)
    4. Creates necessary directories
    
    Returns the base temp directory path.
    """
    
    # --- CONFIGURATION ---
    # Preferred drive for temp/cache (change this for your system)
    # Set to None to auto-detect, or specify like "A:", "D:", etc.
    PREFERRED_TEMP_DRIVE = "A:"
    
    # Minimum free space required (MB) - will skip drives with less
    # Set low (250MB) because Windows is stubborn and ignores drives 
    # it thinks are "too full" even when we tell it not to use C:
    MIN_FREE_SPACE_MB = 500
    
    # C: drive "pretend" size - tell the system C: only has this much space
    # Forces libraries to look elsewhere. Set to 250 or lower to be safe.
    C_DRIVE_FAKE_FREE_MB = 250
    
    # Subdirectory names
    TEMP_SUBDIR = "AI_Temp"
    CACHE_SUBDIR = "AI_Cache"
    
    # --- AUTO-DETECTION ---
    def get_drive_free_space(drive: str, fake_c_drive: bool = True) -> float:
        """
        Get free space on a drive in MB.
        
        If fake_c_drive is True and drive is C:, returns the fake size
        to discourage libraries from using it.
        """
        try:
            # For C: drive, lie about available space to force other drives
            if fake_c_drive and drive.upper().startswith("C"):
                return C_DRIVE_FAKE_FREE_MB
            
            import shutil
            total, used, free = shutil.disk_usage(drive)
            return free / (1024 ** 2)  # Convert to MB
        except:
            return 0
    
    def find_best_drive() -> str:
        """Find the best drive for temp files."""
        # If preferred drive is set and has space, use it
        if PREFERRED_TEMP_DRIVE:
            free = get_drive_free_space(PREFERRED_TEMP_DRIVE, fake_c_drive=False)
            if free >= MIN_FREE_SPACE_MB:
                return PREFERRED_TEMP_DRIVE
            else:
                print(f"[TempConfig] Warning: {PREFERRED_TEMP_DRIVE} has only {free:.0f}MB free")
        
        # Otherwise, find drive with most space (excluding C:)
        best_drive = None
        best_space = 0
        
        for letter in "DEFGHIJKLMNOPQRSTUVWXYZAB":  # A and B last (often floppy)
            drive = f"{letter}:"
            if drive.upper() == "C:":
                continue  # Skip C: drive entirely
            
            free = get_drive_free_space(drive, fake_c_drive=False)
            if free > best_space:
                best_space = free
                best_drive = drive
        
        if best_drive and best_space >= MIN_FREE_SPACE_MB:
            return best_drive
        
        # Fallback to app directory if no suitable drive found
        app_dir = Path(__file__).parent.parent
        print(f"[TempConfig] No suitable drive found, using app directory: {app_dir}")
        return str(app_dir)
    
    # --- SETUP ---
    base_drive = find_best_drive()
    
    # Handle both "X:" style and path style
    if len(base_drive) == 2 and base_drive[1] == ":":
        temp_base = Path(f"{base_drive}/{TEMP_SUBDIR}")
        cache_base = Path(f"{base_drive}/{CACHE_SUBDIR}")
    else:
        temp_base = Path(base_drive) / TEMP_SUBDIR
        cache_base = Path(base_drive) / CACHE_SUBDIR
    
    # Create subdirectories
    dirs = {
        "temp_general": temp_base / "general",
        "temp_imagegen": temp_base / "image_gen", 
        "cache_hf": cache_base / "huggingface",
        "cache_torch": cache_base / "torch",
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
    
    # --- ENVIRONMENT VARIABLES ---
    # Set these for libraries that actually read them
    
    # Windows temp
    os.environ["TEMP"] = str(dirs["temp_general"])
    os.environ["TMP"] = str(dirs["temp_general"])
    os.environ["TMPDIR"] = str(dirs["temp_general"])
    
    # HuggingFace
    os.environ["HF_HOME"] = str(dirs["cache_hf"])
    os.environ["HF_DATASETS_CACHE"] = str(dirs["cache_hf"] / "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(dirs["cache_hf"] / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(dirs["cache_hf"] / "transformers")
    os.environ["DIFFUSERS_CACHE"] = str(dirs["cache_hf"] / "diffusers")
    
    # Torch
    os.environ["TORCH_HOME"] = str(dirs["cache_torch"])
    os.environ["TORCH_EXTENSIONS_DIR"] = str(dirs["cache_torch"] / "extensions")
    
    # XDG (some libs check these even on Windows)
    os.environ["XDG_CACHE_HOME"] = str(cache_base)
    
    # Disable telemetry/symlinks that cause issues
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    # PyTorch CUDA memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # --- MONKEY-PATCH TEMPFILE ---
    # This is the critical part - Python's tempfile ignores env vars!
    
    tempfile.tempdir = str(dirs["temp_general"])
    
    _original_gettempdir = tempfile.gettempdir
    def _forced_gettempdir():
        return str(dirs["temp_general"])
    tempfile.gettempdir = _forced_gettempdir
    
    # Also patch _get_default_tempdir if it exists
    if hasattr(tempfile, '_get_default_tempdir'):
        tempfile._get_default_tempdir = lambda: str(dirs["temp_general"])
    
    # --- MONKEY-PATCH SHUTIL.DISK_USAGE ---
    # Some libraries call shutil.disk_usage directly to check space
    # We lie about C: drive to make it look nearly full
    
    import shutil
    _original_disk_usage = shutil.disk_usage
    
    def _patched_disk_usage(path):
        """Return fake low space for C: drive to discourage its use."""
        result = _original_disk_usage(path)
        
        # Check if this is C: drive
        path_str = str(path).upper()
        if path_str.startswith("C:") or path_str.startswith("C\\") or path_str == "C":
            # Return a namedtuple-like with fake low free space
            # total and used stay real, but free becomes tiny
            fake_free = C_DRIVE_FAKE_FREE_MB * 1024 * 1024  # Convert MB to bytes
            
            # Create a new namedtuple with fake free space
            from collections import namedtuple
            DiskUsage = namedtuple('usage', ['total', 'used', 'free'])
            return DiskUsage(result.total, result.total - fake_free, fake_free)
        
        return result
    
    shutil.disk_usage = _patched_disk_usage
    
    # --- REPORT ---
    free_space = get_drive_free_space(base_drive, fake_c_drive=False)
    print(f"[TempConfig] Using {base_drive} for temp/cache ({free_space:.0f}MB free)")
    print(f"[TempConfig] Temp: {temp_base}")
    print(f"[TempConfig] Cache: {cache_base}")
    print(f"[TempConfig] C: drive reported as {C_DRIVE_FAKE_FREE_MB}MB to discourage use")
    
    return {
        "base_drive": base_drive,
        "temp_base": temp_base,
        "cache_base": cache_base,
        "dirs": dirs
    }


# Run temp setup IMMEDIATELY (before any other imports)
_temp_config = _setup_temp_directories()


# ============================================
# NOW SAFE TO IMPORT HEAVY LIBRARIES
# ============================================

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("[ImageGen] PyTorch not available")

try:
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
except ImportError:
    print("[ImageGen] PIL not available")

try:
    from backend.debug import debug, enable_debug, DebugLevel
    DEBUG_AVAILABLE = True
    enable_debug(DebugLevel.VERBOSE)
except ImportError:
    DEBUG_AVAILABLE = False
    class DebugStub:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    debug = DebugStub()


# ============================================
# MODEL FAMILY DETECTION
# ============================================

class ModelFamily(Enum):
    """Supported model families."""
    SD15 = "sd15"
    SD20 = "sd20"
    SDXL = "sdxl"
    PONY = "pony"
    FLUX = "flux"


def detect_model_family(model_name: str) -> ModelFamily:
    """Detect model family from filename."""
    name_lower = model_name.lower()
    
    if 'flux' in name_lower:
        return ModelFamily.FLUX
    
    if any(x in name_lower for x in ['pony', 'illustrious', 'animagine']):
        return ModelFamily.PONY
    
    if any(x in name_lower for x in ['sdxl', 'xl_', '_xl']):
        return ModelFamily.SDXL
    
    if any(x in name_lower for x in ['sd2', 'v2-', '768-v', '2.0', '2.1']):
        return ModelFamily.SD20
    
    return ModelFamily.SD15


# ============================================
# GENERATION CONFIG
# ============================================

@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1
    sampler: str = "euler"
    
    vae_name: Optional[str] = None
    loras: List[Dict[str, Any]] = field(default_factory=list)
    text_encoders: List[str] = field(default_factory=list)


# ============================================
# IMAGE GENERATOR MANAGER
# ============================================

class ImageGenerator:
    """
    Manager/Orchestrator for image generation.
    """
    
    def __init__(self,
                 checkpoint_dir: str = "models/checkpoints",
                 lora_dir: str = "models/loras",
                 vae_dir: str = "models/vae",
                 text_encoder_dir: str = "models/text_encoders",
                 output_dir: str = "outputs/images"):
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.lora_dir = Path(lora_dir)
        self.vae_dir = Path(vae_dir)
        self.text_encoder_dir = Path(text_encoder_dir)
        self.output_dir = Path(output_dir)
        
        for d in [self.checkpoint_dir, self.lora_dir, self.vae_dir, 
                  self.text_encoder_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Hardware info
        self.device = "cpu"
        self.dtype = torch.float32 if TORCH_AVAILABLE else None
        self.vram_gb = 0
        self.gpu_name = "CPU"
        
        # Pipeline state
        self.pipeline = None
        self.current_family: Optional[ModelFamily] = None
        self.current_model: Optional[str] = None
        
        # Memory management
        self._gen_count = 0
        self._deep_cleanup_interval = 3
        
        self._detect_hardware()
    
    def _detect_hardware(self):
        """Detect GPU and available VRAM."""
        debug.separator("HARDWARE DETECTION")
        
        if not TORCH_AVAILABLE:
            debug.warn("PyTorch not available, using CPU")
            print("[Manager] PyTorch not available, using CPU")
            return
        
        if torch.cuda.is_available():
            self.device = "cuda"
            self.gpu_name = torch.cuda.get_device_name(0)
            self.vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            debug.info(f"GPU: {self.gpu_name}")
            debug.info(f"VRAM: {self.vram_gb:.1f} GB")
            
            if "ZLUDA" in self.gpu_name:
                debug.warn("ZLUDA detected - Flash Attention will be disabled")
            
            self.dtype = torch.float16
            debug.info(f"Default dtype: {self.dtype}")
            print(f"[Manager] GPU: {self.gpu_name} ({self.vram_gb:.1f} GB VRAM)")
        else:
            debug.warn("CUDA not available, using CPU")
            print("[Manager] CUDA not available, using CPU")
        
        debug.separator()
    
    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"used": 0, "total": 0, "free": 0}
        
        used = torch.cuda.memory_allocated(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            "used": used,
            "total": total,
            "free": total - used
        }
    
    # ============================================
    # MEMORY MANAGEMENT
    # ============================================
    
    def clear_vram(self, deep: bool = False):
        """Clear VRAM between generations."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        vram_before = self.get_vram_usage()
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if deep:
            debug.info("Performing deep VRAM cleanup...")
            torch.cuda.reset_peak_memory_stats()
            
            for _ in range(3):
                gc.collect()
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
        
        vram_after = self.get_vram_usage()
        freed = vram_before['used'] - vram_after['used']
        
        if freed > 0.01:
            debug.info(f"VRAM cleared: {freed:.2f} GB freed")
            print(f"[Manager] VRAM cleared: {freed:.2f} GB freed")
    
    def clear_disk_temp(self):
        """Clear temporary files from temp directory."""
        import shutil
        
        temp_dir = _temp_config["dirs"]["temp_imagegen"]
        temp_cleared = 0
        
        if temp_dir.exists():
            for item in temp_dir.iterdir():
                try:
                    if item.is_file():
                        size = item.stat().st_size
                        item.unlink()
                        temp_cleared += size
                    elif item.is_dir():
                        size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                        shutil.rmtree(item)
                        temp_cleared += size
                except Exception as e:
                    debug.warn(f"Could not delete {item}: {e}")
        
        if temp_cleared > 0:
            debug.info(f"Disk temp cleared: {temp_cleared / 1024**2:.1f} MB")
            print(f"[Manager] Disk temp cleared: {temp_cleared / 1024**2:.1f} MB")
    
    def pre_generation_cleanup(self):
        """Called before each generation."""
        self._gen_count += 1
        do_deep = (self._gen_count % self._deep_cleanup_interval == 0)
        
        debug.info(f"Pre-generation cleanup (gen #{self._gen_count}, deep={do_deep})")
        self.clear_vram(deep=do_deep)
        
        if do_deep:
            self.clear_disk_temp()
        
        # Force deep cleanup if low on VRAM
        vram = self.get_vram_usage()
        if vram['free'] < 1.0:
            debug.warn(f"Low VRAM: {vram['free']:.2f} GB free, forcing deep cleanup")
            self.clear_vram(deep=True)
            self.clear_disk_temp()
    
    def post_generation_cleanup(self):
        """Called after each generation."""
        debug.info("Post-generation cleanup")
        gc.collect()
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # ============================================
    # MODEL SCANNING
    # ============================================
    
    def scan_checkpoints(self) -> List[Dict[str, Any]]:
        """Get checkpoints with metadata."""
        files = []
        if self.checkpoint_dir.exists():
            for ext in ["*.safetensors", "*.ckpt", "*.gguf"]:
                for f in self.checkpoint_dir.glob(ext):
                    family = detect_model_family(f.name)
                    files.append({
                        "name": f.name,
                        "family": family.value,
                        "is_gguf": f.suffix.lower() == ".gguf",
                        "size_gb": f.stat().st_size / 1024**3,
                    })
        return sorted(files, key=lambda x: x["name"])
    
    def scan_loras(self) -> List[str]:
        files = []
        if self.lora_dir.exists():
            for ext in ["*.safetensors", "*.ckpt"]:
                files.extend([f.name for f in self.lora_dir.glob(ext)])
        return sorted(files)
    
    def scan_vaes(self) -> List[str]:
        files = []
        if self.vae_dir.exists():
            for ext in ["*.safetensors", "*.sft", "*.ckpt", "*.pt"]:
                files.extend([f.name for f in self.vae_dir.glob(ext)])
        return sorted(files)
    
    def scan_text_encoders(self) -> List[str]:
        files = []
        if self.text_encoder_dir.exists():
            for ext in ["*.safetensors", "*.gguf", "*.bin"]:
                files.extend([f.name for f in self.text_encoder_dir.glob(ext)])
        return sorted(files)
    
    # ============================================
    # PIPELINE MANAGEMENT
    # ============================================
    
    def _get_pipeline_for_family(self, family: ModelFamily):
        """Get the appropriate pipeline class for a model family."""
        from backend.pipelines import SD15Pipeline, SDXLPipeline, FluxPipeline
        
        offload = self.vram_gb < 12
        
        if family == ModelFamily.FLUX:
            return FluxPipeline(
                device=self.device,
                dtype=torch.bfloat16,
                offload_to_cpu=offload
            )
        elif family in [ModelFamily.SDXL, ModelFamily.PONY]:
            return SDXLPipeline(
                device=self.device,
                dtype=self.dtype,
                offload_to_cpu=offload,
                is_pony=(family == ModelFamily.PONY)
            )
        else:
            return SD15Pipeline(
                device=self.device,
                dtype=self.dtype,
                offload_to_cpu=offload
            )
    
    def load_model(self,
                   model_id: str,
                   vae_name: Optional[str] = None,
                   text_encoder_names: Optional[List[str]] = None,
                   sampler: str = "euler",
                   progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """Load a model by routing to the appropriate pipeline."""
        family = detect_model_family(model_id)
        
        debug.separator("MODEL LOADING")
        debug.model(f"Model ID: {model_id}")
        debug.model(f"Family: {family.value}")
        debug.model(f"VAE: {vae_name or 'Auto'}")
        debug.model(f"Text Encoders: {text_encoder_names or 'Default'}")
        debug.model(f"Sampler: {sampler}")
        debug.gpu("Current VRAM state")
        
        print(f"[Manager] Model: {model_id}")
        print(f"[Manager] Family: {family.value}")
        print(f"[Manager] VRAM: {self.vram_gb:.1f} GB")
        
        # Clear VRAM before loading new model
        self.clear_vram(deep=True)
        
        if self.pipeline and self.current_family != family:
            debug.model("Switching pipeline family, unloading current...")
            if progress_callback:
                progress_callback("Unloading previous model...")
            self.unload()
        
        checkpoint_path = self.checkpoint_dir / model_id
        if not checkpoint_path.exists():
            debug.error(f"Checkpoint not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        debug.io(f"Checkpoint path: {checkpoint_path}")
        debug.io(f"Checkpoint size: {checkpoint_path.stat().st_size / 1024**3:.2f} GB")
        
        vae_path = None
        if vae_name and "Auto" not in vae_name:
            vae_path = self.vae_dir / vae_name
            if not vae_path.exists():
                debug.warn(f"VAE not found: {vae_path}")
                vae_path = None
            else:
                debug.io(f"VAE path: {vae_path}")
        
        text_encoder_paths = []
        if text_encoder_names:
            for name in text_encoder_names:
                path = self.text_encoder_dir / name
                if path.exists():
                    text_encoder_paths.append(path)
                    debug.io(f"Text encoder: {path}")
                else:
                    debug.warn(f"Text encoder not found: {path}")
        
        if self.pipeline is None or self.current_family != family:
            debug.pipeline(f"Creating new {family.value} pipeline...")
            if progress_callback:
                progress_callback(f"Initializing {family.value.upper()} pipeline...")
            self.pipeline = self._get_pipeline_for_family(family)
            self.current_family = family
        
        debug.model("Calling pipeline.load_model()...")
        self.pipeline.load_model(
            checkpoint_path=checkpoint_path,
            vae_path=vae_path,
            text_encoder_paths=text_encoder_paths if text_encoder_paths else None,
            progress_callback=progress_callback
        )
        
        self.current_model = model_id
        
        debug.model("Model loading complete!")
        debug.gpu("Final VRAM state")
        
        if progress_callback:
            progress_callback(f"Model loaded: {model_id}")
    
    def unload(self) -> None:
        """Unload current model and free memory."""
        if self.pipeline:
            self.pipeline.unload()
            self.pipeline = None
        
        self.current_family = None
        self.current_model = None
        
        self.clear_vram(deep=True)
        self.clear_disk_temp()
        
        print("[Manager] Model unloaded, memory cleared")
    
    # ============================================
    # GENERATION
    # ============================================
    
    def generate(self,
                 config: GenerationConfig,
                 progress_callback: Optional[Callable[[int, int], None]] = None) -> Image.Image:
        """Generate an image using the loaded pipeline."""
        if not self.pipeline or not self.pipeline.is_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        print(f"[Manager] Generating: {config.width}x{config.height}, "
              f"{config.steps} steps, CFG {config.cfg_scale}")
        
        self.pre_generation_cleanup()
        
        if config.loras:
            self.pipeline.apply_loras(config.loras, self.lora_dir)
        
        try:
            image = self.pipeline.generate(config, progress_callback)
            return image
        finally:
            self.post_generation_cleanup()
    
    def generate_and_save(self,
                          config: GenerationConfig,
                          progress_callback: Optional[Callable[[int, int], None]] = None) -> str:
        """Generate an image and save it with metadata."""
        image = self.generate(config, progress_callback)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed = config.seed if config.seed != -1 else "random"
        filename = f"{timestamp}_{seed}.png"
        output_path = self.output_dir / filename
        
        metadata = PngInfo()
        metadata.add_text("prompt", config.prompt)
        metadata.add_text("negative_prompt", config.negative_prompt)
        metadata.add_text("width", str(config.width))
        metadata.add_text("height", str(config.height))
        metadata.add_text("steps", str(config.steps))
        metadata.add_text("cfg_scale", str(config.cfg_scale))
        metadata.add_text("seed", str(seed))
        metadata.add_text("sampler", config.sampler)
        metadata.add_text("model", self.current_model or "unknown")
        metadata.add_text("model_family", self.current_family.value if self.current_family else "unknown")
        
        image.save(str(output_path), pnginfo=metadata)
        print(f"[Manager] Saved: {output_path}")
        
        return str(output_path)


# ============================================
# SINGLETON INSTANCE
# ============================================

_generator_instance = None


def get_generator(checkpoint_dir: str = "models/checkpoints",
                  lora_dir: str = "models/loras",
                  vae_dir: str = "models/vae",
                  text_encoder_dir: str = "models/text_encoders",
                  output_dir: str = "outputs/images") -> ImageGenerator:
    """Get singleton ImageGenerator instance."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = ImageGenerator(
            checkpoint_dir, lora_dir, vae_dir, text_encoder_dir, output_dir
        )
    return _generator_instance


def reset_generator():
    """Force reset the singleton."""
    global _generator_instance
    if _generator_instance:
        _generator_instance.unload()
    _generator_instance = None


def get_temp_config() -> dict:
    """Get the current temp/cache configuration."""
    return _temp_config
