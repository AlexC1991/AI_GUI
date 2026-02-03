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

# Redirect temp/cache BEFORE importing torch
_app_dir = Path(__file__).parent.parent
_temp_dir = _app_dir / "temp_workspace"
_cache_dir = _app_dir / "models" / "hf_cache"
_temp_dir.mkdir(exist_ok=True)
_cache_dir.mkdir(parents=True, exist_ok=True)

os.environ["TMPDIR"] = str(_temp_dir)
os.environ["TEMP"] = str(_temp_dir)
os.environ["TMP"] = str(_temp_dir)
os.environ["HF_HOME"] = str(_cache_dir)
os.environ["TRANSFORMERS_CACHE"] = str(_cache_dir)
os.environ["HF_DATASETS_CACHE"] = str(_cache_dir)

# Check for torch
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("[ImageGen] PyTorch not available")

# Check for PIL
try:
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
except ImportError:
    print("[ImageGen] PIL not available")

# Import debug system
try:
    from backend.debug import debug, enable_debug, DebugLevel
    DEBUG_AVAILABLE = True
    # Enable debug by default at INFO level
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
    
    # Flux detection (must be first - includes GGUF)
    if 'flux' in name_lower:
        return ModelFamily.FLUX
    
    # SDXL detection - includes Pony which are SDXL-based
    if any(x in name_lower for x in ['pony', 'illustrious', 'animagine']):
        return ModelFamily.PONY
    
    if any(x in name_lower for x in ['sdxl', 'xl_', '_xl']):
        return ModelFamily.SDXL
    
    # SD 2.x detection
    if any(x in name_lower for x in ['sd2', 'v2-', '768-v', '2.0', '2.1']):
        return ModelFamily.SD20
    
    # Default to SD 1.5
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
    
    # Optional customization
    vae_name: Optional[str] = None
    loras: List[Dict[str, Any]] = field(default_factory=list)
    text_encoders: List[str] = field(default_factory=list)


# ============================================
# IMAGE GENERATOR MANAGER
# ============================================

class ImageGenerator:
    """
    Manager/Orchestrator for image generation.
    
    Responsibilities:
    - Detect hardware capabilities
    - Route to appropriate pipeline based on model family
    - Manage memory (VRAM/RAM)
    - Handle output (saving, metadata)
    
    Does NOT directly load models or run inference.
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
        
        # Ensure directories exist
        for d in [self.checkpoint_dir, self.lora_dir, self.vae_dir, 
                  self.text_encoder_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Hardware info
        self.device = "cpu"
        self.dtype = torch.float32 if TORCH_AVAILABLE else None
        self.vram_gb = 0
        self.gpu_name = "CPU"
        
        # Active pipeline
        self.pipeline = None
        self.current_family: Optional[ModelFamily] = None
        self.current_model: Optional[str] = None
        
        # Detect hardware
        self._detect_hardware()
    
    # ============================================
    # HARDWARE DETECTION
    # ============================================
    
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
            
            # Check for ZLUDA
            if "ZLUDA" in self.gpu_name:
                debug.warn("ZLUDA detected - Flash Attention will be disabled")
            
            # Determine safe dtype
            if self.vram_gb >= 12:
                self.dtype = torch.float16
            else:
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
        """Get list of available LoRA files."""
        files = []
        if self.lora_dir.exists():
            for ext in ["*.safetensors", "*.ckpt"]:
                files.extend([f.name for f in self.lora_dir.glob(ext)])
        return sorted(files)
    
    def scan_vaes(self) -> List[str]:
        """Get list of available VAE files."""
        files = []
        if self.vae_dir.exists():
            for ext in ["*.safetensors", "*.sft", "*.ckpt", "*.pt"]:
                files.extend([f.name for f in self.vae_dir.glob(ext)])
        return sorted(files)
    
    def scan_text_encoders(self) -> List[str]:
        """Get list of available text encoder files."""
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
        # Import here to avoid circular imports
        from backend.pipelines import SD15Pipeline, SDXLPipeline, FluxPipeline
        
        # Determine if we need CPU offload
        offload = self.vram_gb < 12
        
        if family == ModelFamily.FLUX:
            return FluxPipeline(
                device=self.device,
                dtype=torch.bfloat16,  # Flux prefers bfloat16
                offload_to_cpu=offload
            )
        elif family in [ModelFamily.SDXL, ModelFamily.PONY]:
            return SDXLPipeline(
                device=self.device,
                dtype=self.dtype,
                offload_to_cpu=offload,
                is_pony=(family == ModelFamily.PONY)
            )
        else:  # SD15, SD20
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
        """
        Load a model by routing to the appropriate pipeline.
        """
        # Detect model family
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
        
        # Check if we need to switch pipelines
        if self.pipeline and self.current_family != family:
            debug.model("Switching pipeline family, unloading current...")
            if progress_callback:
                progress_callback("Unloading previous model...")
            self.unload()
        
        # Get checkpoint path
        checkpoint_path = self.checkpoint_dir / model_id
        if not checkpoint_path.exists():
            debug.error(f"Checkpoint not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        debug.io(f"Checkpoint path: {checkpoint_path}")
        debug.io(f"Checkpoint size: {checkpoint_path.stat().st_size / 1024**3:.2f} GB")
        
        # Get VAE path
        vae_path = None
        if vae_name and "Auto" not in vae_name:
            vae_path = self.vae_dir / vae_name
            if not vae_path.exists():
                debug.warn(f"VAE not found: {vae_path}")
                vae_path = None
            else:
                debug.io(f"VAE path: {vae_path}")
        
        # Get text encoder paths
        text_encoder_paths = []
        if text_encoder_names:
            for name in text_encoder_names:
                path = self.text_encoder_dir / name
                if path.exists():
                    text_encoder_paths.append(path)
                    debug.io(f"Text encoder: {path}")
                else:
                    debug.warn(f"Text encoder not found: {path}")
        
        # Create pipeline if needed
        if self.pipeline is None or self.current_family != family:
            debug.pipeline(f"Creating new {family.value} pipeline...")
            if progress_callback:
                progress_callback(f"Initializing {family.value.upper()} pipeline...")
            self.pipeline = self._get_pipeline_for_family(family)
            self.current_family = family
        
        # Load model
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
        
        # Force garbage collection
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("[Manager] Model unloaded, memory cleared")
    
    # ============================================
    # GENERATION
    # ============================================
    
    def generate(self,
                 config: GenerationConfig,
                 progress_callback: Optional[Callable[[int, int], None]] = None) -> Image.Image:
        """
        Generate an image using the loaded pipeline.
        
        Args:
            config: Generation configuration
            progress_callback: Called with (current_step, total_steps)
            
        Returns:
            Generated PIL Image
        """
        if not self.pipeline or not self.pipeline.is_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        print(f"[Manager] Generating: {config.width}x{config.height}, "
              f"{config.steps} steps, CFG {config.cfg_scale}")
        
        # Apply LoRAs if any
        if config.loras:
            self.pipeline.apply_loras(config.loras, self.lora_dir)
        
        # Generate
        image = self.pipeline.generate(config, progress_callback)
        
        return image
    
    def generate_and_save(self,
                          config: GenerationConfig,
                          progress_callback: Optional[Callable[[int, int], None]] = None) -> str:
        """
        Generate an image and save it with metadata.
        
        Returns:
            Path to saved image
        """
        image = self.generate(config, progress_callback)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed = config.seed if config.seed != -1 else "random"
        filename = f"{timestamp}_{seed}.png"
        output_path = self.output_dir / filename
        
        # Create metadata
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
        
        # Save
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
