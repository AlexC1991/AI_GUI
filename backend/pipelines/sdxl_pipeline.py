"""
SDXL Pipeline - Stable Diffusion XL and compatible models.
Handles: SDXL, SDXL Turbo, Pony, Illustrious, Animagine
"""
from __future__ import annotations

from typing import Optional, Callable, List, Dict, Any
from pathlib import Path
import torch

from .base_pipeline import BasePipeline, PipelineInfo, GenerationConfig

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


class SDXLPipeline(BasePipeline):
    """
    Pipeline for SDXL-based models.
    
    Characteristics:
    - 1024x1024 native resolution
    - Dual text encoders (CLIP ViT-L + OpenCLIP ViT-bigG)
    - UNet with cross-attention
    - Standard VAE (4 latent channels)
    - Supports negative prompts
    - CFG scale typically 5-10
    """
    
    def __init__(self, 
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 offload_to_cpu: bool = False,
                 is_pony: bool = False):
        super().__init__(device, dtype, offload_to_cpu)
        self.is_pony = is_pony
        debug.model(f"SDXL Pipeline initialized (Pony mode: {is_pony})")
    
    @property
    def info(self) -> PipelineInfo:
        if self.is_pony:
            return PipelineInfo(
                name="SDXLPipeline (Pony)",
                family="pony",
                min_vram_gb=6.0,
                native_resolution=1024,
                default_cfg=6.0,
                default_steps=25,
                supports_negative_prompt=True,
                supports_cfg=True,
                description="Pony/Illustrious | 1024px | CFG 5-8 | quality tags"
            )
        return PipelineInfo(
            name="SDXLPipeline",
            family="sdxl",
            min_vram_gb=6.0,
            native_resolution=1024,
            default_cfg=7.0,
            default_steps=25,
            supports_negative_prompt=True,
            supports_cfg=True,
            description="SDXL | 1024px | CFG 5-10 | natural language"
        )
    
    def load_model(self,
                   checkpoint_path: Path,
                   vae_path: Optional[Path] = None,
                   text_encoder_paths: Optional[List[Path]] = None,
                   progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """Load SDXL model from checkpoint."""
        from diffusers import StableDiffusionXLPipeline, AutoencoderKL
        
        model_type = "Pony/Illustrious" if self.is_pony else "SDXL"
        
        debug.separator(f"{model_type.upper()} MODEL LOADING")
        debug.model(f"Checkpoint: {checkpoint_path.name}")
        debug.model(f"Size: {checkpoint_path.stat().st_size / 1024**3:.2f} GB")
        debug.model(f"Model Type: {model_type}")
        debug.model(f"Dtype: {self.dtype}")
        debug.model(f"Device: {self.device}")
        debug.model(f"CPU Offload: {self.offload_to_cpu}")
        debug.gpu("Initial VRAM state")
        
        if progress_callback:
            progress_callback(f"Loading {model_type}: {checkpoint_path.name}...")
        
        print(f"[SDXL] Loading checkpoint: {checkpoint_path}")
        
        # Load main pipeline
        debug.pipeline("Loading StableDiffusionXLPipeline from single file...")
        with debug.timer("Pipeline loading"):
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                str(checkpoint_path),
                torch_dtype=self.dtype,
                use_safetensors=checkpoint_path.suffix == ".safetensors",
                local_files_only=True,
            )
        debug.gpu("After pipeline load")
        
        # Log pipeline components
        debug.pipeline(f"UNet: {type(self.pipe.unet).__name__}")
        debug.pipeline(f"VAE: {type(self.pipe.vae).__name__}")
        debug.pipeline(f"Text Encoder 1: {type(self.pipe.text_encoder).__name__}")
        debug.pipeline(f"Text Encoder 2: {type(self.pipe.text_encoder_2).__name__}")
        debug.pipeline(f"Scheduler: {type(self.pipe.scheduler).__name__}")
        
        # Custom VAE
        if vae_path and vae_path.exists():
            if progress_callback:
                progress_callback(f"Loading custom VAE: {vae_path.name}...")
            debug.separator("CUSTOM VAE")
            debug.model(f"VAE path: {vae_path}")
            print(f"[SDXL] Loading custom VAE: {vae_path}")
            try:
                vae_name_lower = vae_path.name.lower()
                if "flux" in vae_name_lower or vae_path.suffix == ".sft":
                    debug.warn(f"Skipping Flux VAE - incompatible with SDXL")
                    print(f"[SDXL] Skipping Flux VAE - incompatible with SDXL")
                else:
                    with debug.timer("VAE loading"):
                        vae = AutoencoderKL.from_single_file(
                            str(vae_path),
                            torch_dtype=self.dtype
                        )
                        self.pipe.vae = vae
                    debug.model("Custom VAE loaded successfully")
                    debug.gpu("After VAE load")
            except Exception as e:
                debug.error(f"Custom VAE failed: {e}", exc_info=True)
                print(f"[SDXL] Custom VAE failed: {e}")
        
        # Move to device or enable offload
        debug.separator("DEVICE PLACEMENT")
        if self.offload_to_cpu:
            if progress_callback:
                progress_callback("Enabling CPU offload...")
            debug.model("Enabling CPU offload...")
            with debug.timer("CPU offload setup"):
                self.pipe.enable_model_cpu_offload()
            debug.model("CPU offload enabled")
        else:
            debug.model(f"Moving pipeline to {self.device}...")
            with debug.timer("Device transfer"):
                self.pipe.to(self.device)
            debug.model(f"Pipeline on {self.device}")
        
        debug.gpu("Final VRAM state")
        
        self.is_loaded = True
        self.current_model = checkpoint_path.name
        debug.model(f"{model_type} model loaded successfully!")
        debug.separator()
        print(f"[SDXL] Model loaded successfully")
    
    def generate(self,
                 config: GenerationConfig,
                 progress_callback: Optional[Callable[[int, int], None]] = None):
        """Generate image with SDXL."""
        from PIL import Image
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        model_type = "Pony" if self.is_pony else "SDXL"
        
        debug.separator(f"{model_type.upper()} GENERATION")
        debug.generation_config(config)
        debug.reset_step_timer()
        debug.gpu("Before generation")
        
        # Set scheduler
        debug.scheduler(f"Setting scheduler for sampler: {config.sampler}")
        self.pipe.scheduler = self.get_scheduler(config.sampler, self.pipe.scheduler.config)
        debug.scheduler(f"Using: {type(self.pipe.scheduler).__name__}")
        
        # Prepare generator for seed
        seed = config.seed
        if seed == -1:
            import random
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        debug.generation(f"Seed: {seed}")
        
        debug.generation(f"Generating: {config.width}x{config.height}, {config.steps} steps, CFG {config.cfg_scale}")
        print(f"[SDXL] Generating: {config.width}x{config.height}, {config.steps} steps, CFG {config.cfg_scale}, seed {seed}")
        
        # Progress callback wrapper
        def step_callback(pipe, step, timestep, callback_kwargs):
            debug.step(step + 1, config.steps, f"timestep={timestep:.1f}")
            if progress_callback:
                progress_callback(step + 1, config.steps)
            return callback_kwargs
        
        # Generate
        debug.generation("Starting inference...")
        with debug.timer(f"{model_type} inference"):
            try:
                result = self.pipe(
                    prompt=config.prompt,
                    negative_prompt=config.negative_prompt,
                    width=config.width,
                    height=config.height,
                    num_inference_steps=config.steps,
                    guidance_scale=config.cfg_scale,
                    generator=generator,
                    callback_on_step_end=step_callback if progress_callback else None,
                )
            except Exception as e:
                debug.error(f"Generation failed: {e}", exc_info=True)
                raise
        
        debug.gpu("After generation")
        debug.generation("Generation complete!")
        debug.separator()
        
        return result.images[0]
    
    def unload(self) -> None:
        """Unload model and free memory."""
        model_type = "Pony" if self.is_pony else "SDXL"
        
        debug.separator(f"{model_type.upper()} UNLOAD")
        debug.gpu("Before unload")
        
        if self.pipe:
            debug.model("Deleting pipeline...")
            del self.pipe
            self.pipe = None
        
        self.is_loaded = False
        self.current_model = None
        self._clear_memory()
        
        debug.gpu("After unload")
        debug.model(f"{model_type} model unloaded")
        debug.separator()
        print("[SDXL] Model unloaded")
