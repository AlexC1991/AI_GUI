"""
Base Pipeline - Abstract base class for all image generation pipelines.
Each model family (SD1.5, SDXL, Flux) implements this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List, TYPE_CHECKING
from pathlib import Path
import torch

if TYPE_CHECKING:
    from PIL import Image

# Import debug system
try:
    from backend.debug import debug, DebugLevel, DebugCategory
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False
    class DebugStub:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    debug = DebugStub()


@dataclass
class GenerationConfig:
    """Configuration for image generation - shared across all pipelines."""
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
    loras: Optional[List[Dict[str, Any]]] = None
    text_encoders: Optional[List[str]] = None


@dataclass 
class PipelineInfo:
    """Information about a pipeline's capabilities and requirements."""
    name: str
    family: str
    min_vram_gb: float
    native_resolution: int
    default_cfg: float
    default_steps: int
    supports_negative_prompt: bool = True
    supports_cfg: bool = True
    description: str = ""


class BasePipeline(ABC):
    """
    Abstract base class for image generation pipelines.
    
    Each pipeline is responsible for:
    - Loading its specific model architecture
    - Managing its text encoders
    - Managing its VAE
    - Running inference
    - Returning generated images
    
    The manager (ImageGenerator) handles:
    - Memory management
    - Device selection
    - Output saving
    - Pipeline selection/routing
    """
    
    def __init__(self, 
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 offload_to_cpu: bool = False):
        self.device = device
        self.dtype = dtype
        self.offload_to_cpu = offload_to_cpu
        self.pipe = None
        self.is_loaded = False
        self.current_model: Optional[str] = None
    
    @property
    @abstractmethod
    def info(self) -> PipelineInfo:
        """Return information about this pipeline's capabilities."""
        pass
    
    @abstractmethod
    def load_model(self,
                   checkpoint_path: Path,
                   vae_path: Optional[Path] = None,
                   text_encoder_paths: Optional[List[Path]] = None,
                   progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """Load the model and all required components."""
        pass
    
    @abstractmethod
    def generate(self,
                 config: GenerationConfig,
                 progress_callback: Optional[Callable[[int, int], None]] = None) -> "Image.Image":
        """Generate an image."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free memory."""
        pass
    
    def apply_loras(self, loras: List[Dict[str, Any]], lora_dir: Path) -> None:
        """Apply LoRA weights to the loaded model."""
        if not self.pipe or not loras:
            return
        
        debug.separator("LORA APPLICATION")
        debug.model(f"Applying {len(loras)} LoRA(s)")
        
        for lora in loras:
            lora_path = lora_dir / lora["name"]
            if lora_path.exists():
                try:
                    debug.model(f"Loading LoRA: {lora['name']}")
                    debug.io(f"LoRA path: {lora_path}")
                    debug.io(f"LoRA size: {lora_path.stat().st_size / 1024**2:.1f} MB")
                    
                    with debug.timer(f"LoRA {lora['name']}"):
                        self.pipe.load_lora_weights(str(lora_path))
                        self.pipe.fuse_lora(lora_scale=lora.get("strength", 1.0))
                    
                    debug.model(f"Applied LoRA: {lora['name']} @ {lora.get('strength', 1.0)}")
                    print(f"[{self.info.name}] Applied LoRA: {lora['name']} @ {lora.get('strength', 1.0)}")
                except Exception as e:
                    debug.error(f"Failed to load LoRA {lora['name']}: {e}", exc_info=True)
                    print(f"[{self.info.name}] Failed to load LoRA {lora['name']}: {e}")
            else:
                debug.warn(f"LoRA not found: {lora_path}")
        
        debug.gpu("After LoRA application")
        debug.separator()
    
    def get_scheduler(self, sampler_name: str, config: Any):
        """Get the appropriate scheduler for the sampler name."""
        from diffusers import (
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
            DDIMScheduler,
            UniPCMultistepScheduler,
            HeunDiscreteScheduler,
            LMSDiscreteScheduler,
        )
        
        sampler_lower = sampler_name.lower().replace(" ", "_")
        debug.scheduler(f"Resolving sampler: {sampler_name} -> {sampler_lower}")
        
        if "euler_a" in sampler_lower or "ancestral" in sampler_lower:
            debug.scheduler("Using EulerAncestralDiscreteScheduler")
            return EulerAncestralDiscreteScheduler.from_config(config)
        elif "euler" in sampler_lower:
            debug.scheduler("Using EulerDiscreteScheduler")
            return EulerDiscreteScheduler.from_config(config)
        elif "dpm" in sampler_lower and "sde" in sampler_lower:
            sched = DPMSolverMultistepScheduler.from_config(config)
            sched.config.algorithm_type = "sde-dpmsolver++"
            if "karras" in sampler_lower:
                sched.config.use_karras_sigmas = True
                debug.scheduler("Using DPMSolverMultistepScheduler (SDE, Karras)")
            else:
                debug.scheduler("Using DPMSolverMultistepScheduler (SDE)")
            return sched
        elif "dpm" in sampler_lower:
            sched = DPMSolverMultistepScheduler.from_config(config)
            if "karras" in sampler_lower:
                sched.config.use_karras_sigmas = True
                debug.scheduler("Using DPMSolverMultistepScheduler (Karras)")
            else:
                debug.scheduler("Using DPMSolverMultistepScheduler")
            return sched
        elif "ddim" in sampler_lower:
            debug.scheduler("Using DDIMScheduler")
            return DDIMScheduler.from_config(config)
        elif "unipc" in sampler_lower:
            debug.scheduler("Using UniPCMultistepScheduler")
            return UniPCMultistepScheduler.from_config(config)
        elif "heun" in sampler_lower:
            debug.scheduler("Using HeunDiscreteScheduler")
            return HeunDiscreteScheduler.from_config(config)
        elif "lms" in sampler_lower:
            debug.scheduler("Using LMSDiscreteScheduler")
            return LMSDiscreteScheduler.from_config(config)
        else:
            debug.scheduler("Using default: DPMSolverMultistepScheduler (Karras)")
            sched = DPMSolverMultistepScheduler.from_config(config)
            sched.config.use_karras_sigmas = True
            return sched
    
    def _clear_memory(self):
        """Clear GPU memory."""
        import gc
        debug.gpu("Clearing memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        debug.gpu("Memory cleared")
