"""
Flux Pipeline - Black Forest Labs Flux models.
Handles: Flux.1 Schnell, Flux.1 Dev, Flux fine-tunes
Supports: GGUF quantized (8GB VRAM) and safetensors (12GB+ VRAM)
"""
from __future__ import annotations

import os

# CRITICAL: Disable Flash Attention for AMD/ZLUDA compatibility
# Must be set BEFORE importing torch or diffusers
os.environ["DIFFUSERS_FLASH_ATTN"] = "0"
os.environ["XFORMERS_DISABLED"] = "1"
# Force PyTorch to use standard SDPA backend, not flash
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

from typing import Optional, Callable, List, Dict, Any
from pathlib import Path
import torch

# Disable flash attention in PyTorch if available
if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(False)
if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
    torch.backends.cuda.enable_mem_efficient_sdp(False)
if hasattr(torch.backends.cuda, 'enable_math_sdp'):
    torch.backends.cuda.enable_math_sdp(True)  # Use the basic math implementation

from .base_pipeline import BasePipeline, PipelineInfo, GenerationConfig

# Import debug system
try:
    from backend.debug import debug, DebugLevel, DebugCategory
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False
    # Fallback debug stub
    class DebugStub:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    debug = DebugStub()


class FluxPipeline(BasePipeline):
    """
    Pipeline for Flux models.
    
    Flux is COMPLETELY DIFFERENT from SD:
    - Uses DiT (Diffusion Transformer) instead of UNet
    - Dual text encoders: CLIP + T5-XXL
    - 16-channel VAE (vs 4-channel in SD)
    - No negative prompts
    - CFG scale 1-4 (much lower than SD)
    - Flow matching scheduler (not DDPM)
    
    Supports two loading modes:
    - GGUF: Quantized models for 8GB VRAM (bfloat16 compute)
    - Safetensors: Full precision for 12GB+ VRAM (float16)
    """
    
    def __init__(self,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16,
                 offload_to_cpu: bool = True):
        super().__init__(device, dtype, offload_to_cpu)
        self.is_gguf = False
        self.text_encoder = None
        self.text_encoder_2 = None
        self.vae = None
        self.transformer = None
    
    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            name="FluxPipeline",
            family="flux",
            min_vram_gb=8.0 if self.is_gguf else 12.0,
            native_resolution=1024,
            default_cfg=3.5,
            default_steps=4 if "schnell" in (self.current_model or "").lower() else 20,
            supports_negative_prompt=False,
            supports_cfg=True,
            description="FLUX | 1024px | CFG 1-4 | natural prompts"
        )
    
    def load_model(self,
                   checkpoint_path: Path,
                   vae_path: Optional[Path] = None,
                   text_encoder_paths: Optional[List[Path]] = None,
                   progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """Load Flux model with all components."""
        from diffusers import FluxPipeline as DiffusersFluxPipeline
        from diffusers import AutoencoderKL
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
        
        debug.separator("FLUX MODEL LOADING")
        debug.gpu("Initial VRAM state")
        
        # Determine if GGUF mode
        self.is_gguf = checkpoint_path.suffix.lower() == ".gguf"
        
        # Set dtype based on model type
        self.dtype = torch.bfloat16 if self.is_gguf else torch.float16
        
        debug.model(f"Checkpoint: {checkpoint_path.name}")
        debug.model(f"Mode: {'GGUF (bfloat16)' if self.is_gguf else 'Safetensors (float16)'}")
        debug.model(f"Dtype: {self.dtype}")
        print(f"[Flux] Loading mode: {'GGUF (bfloat16)' if self.is_gguf else 'Safetensors (float16)'}")
        
        # Determine config repo based on model name
        config_repo = "black-forest-labs/FLUX.1-schnell"
        if "dev" in checkpoint_path.name.lower():
            config_repo = "black-forest-labs/FLUX.1-dev"
        debug.model(f"Config repo: {config_repo}")
        
        # ===== 1. LOAD TEXT ENCODERS =====
        clip_path = None
        t5_path = None
        
        if text_encoder_paths:
            for path in text_encoder_paths:
                if path and path.exists():
                    name_lower = path.name.lower()
                    if "t5" in name_lower:
                        t5_path = path
                        debug.model(f"T5 path: {path}")
                    elif "clip" in name_lower:
                        clip_path = path
                        debug.model(f"CLIP path: {path}")
        
        # Load CLIP
        if progress_callback:
            progress_callback("Loading CLIP text encoder...")
        
        debug.separator("CLIP TEXT ENCODER")
        with debug.timer("CLIP loading"):
            if clip_path and clip_path.exists():
                print(f"[Flux] Loading custom CLIP: {clip_path}")
                debug.model(f"Loading custom CLIP: {clip_path}")
                self.text_encoder = self._load_clip(clip_path, config_repo)
            else:
                print("[Flux] Loading default CLIP from HuggingFace...")
                debug.network("Downloading CLIP from HuggingFace")
                self.text_encoder = CLIPTextModel.from_pretrained(
                    config_repo,
                    subfolder="text_encoder",
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                )
        debug.gpu("After CLIP load")
        
        # Load T5
        if progress_callback:
            progress_callback("Loading T5 text encoder...")
        
        debug.separator("T5 TEXT ENCODER")
        with debug.timer("T5 loading"):
            if t5_path and t5_path.exists():
                print(f"[Flux] Loading custom T5: {t5_path}")
                debug.model(f"Loading custom T5: {t5_path}")
                debug.model(f"T5 format: {t5_path.suffix}")
                self.text_encoder_2 = self._load_t5(t5_path, config_repo, progress_callback)
            else:
                print("[Flux] Loading default T5 from HuggingFace...")
                debug.network("Downloading T5 from HuggingFace")
                self.text_encoder_2 = T5EncoderModel.from_pretrained(
                    config_repo,
                    subfolder="text_encoder_2",
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                )
        debug.gpu("After T5 load")
        
        # ===== 2. LOAD TRANSFORMER =====
        if progress_callback:
            progress_callback("Loading Flux transformer...")
        
        debug.separator("FLUX TRANSFORMER")
        with debug.timer("Transformer loading"):
            self.transformer = self._load_transformer(checkpoint_path, config_repo, progress_callback)
        debug.gpu("After transformer load")
        
        # ===== 3. LOAD VAE =====
        if progress_callback:
            progress_callback("Loading Flux VAE...")
        
        debug.separator("FLUX VAE")
        with debug.timer("VAE loading"):
            self.vae = self._load_vae(vae_path, config_repo)
        debug.gpu("After VAE load")
        
        # ===== 4. LOAD TOKENIZERS =====
        if progress_callback:
            progress_callback("Loading tokenizers...")
        
        debug.model("Loading tokenizers...")
        tokenizer = CLIPTokenizer.from_pretrained(config_repo, subfolder="tokenizer")
        tokenizer_2 = T5TokenizerFast.from_pretrained(config_repo, subfolder="tokenizer_2")
        
        # ===== 5. LOAD SCHEDULER =====
        debug.scheduler("Loading FlowMatchEulerDiscreteScheduler")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config_repo,
            subfolder="scheduler"
        )
        
        # ===== 6. ASSEMBLE PIPELINE =====
        if progress_callback:
            progress_callback("Assembling Flux pipeline...")
        
        debug.separator("PIPELINE ASSEMBLY")
        debug.pipeline("Creating FluxPipeline instance")
        
        self.pipe = DiffusersFluxPipeline(
            transformer=self.transformer,
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
        )
        
        # ===== CRITICAL: Disable Flash Attention for AMD/ZLUDA =====
        # Flash Attention kernels are built for NVIDIA SM80+ (Ampere)
        # ZLUDA emulates SM37 which is incompatible
        if progress_callback:
            progress_callback("Configuring attention for AMD GPU...")
        
        debug.separator("ATTENTION CONFIGURATION")
        
        try:
            # Force use of standard attention instead of Flash Attention
            from diffusers.models.attention_processor import AttnProcessor2_0, AttnProcessor
            
            # Check if we're on ZLUDA/AMD by looking for the emulated architecture
            is_zluda = "ZLUDA" in torch.cuda.get_device_name(0) if torch.cuda.is_available() else False
            debug.attention(f"ZLUDA detected: {is_zluda}")
            debug.attention(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
            
            if is_zluda:
                print("[Flux] ZLUDA detected - disabling Flash Attention")
                debug.attention("Disabling Flash Attention for ZLUDA compatibility")
                
                # Use standard scaled dot product attention (PyTorch native)
                # This avoids the CUTLASS Flash Attention kernels
                debug.attention("Setting transformer attention processor to AttnProcessor2_0")
                self.pipe.transformer.set_attn_processor(AttnProcessor2_0())
                
                # Also disable for VAE if it has attention
                if hasattr(self.pipe.vae, 'set_attn_processor'):
                    debug.attention("Setting VAE attention processor to AttnProcessor2_0")
                    self.pipe.vae.set_attn_processor(AttnProcessor2_0())
                    
                print("[Flux] Using AttnProcessor2_0 (PyTorch SDPA)")
                debug.attention("Using AttnProcessor2_0 (PyTorch SDPA)")
                
                # Log PyTorch SDPA backend status
                if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
                    debug.attention(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
                if hasattr(torch.backends.cuda, 'mem_efficient_sdp_enabled'):
                    debug.attention(f"Mem efficient SDP enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
                if hasattr(torch.backends.cuda, 'math_sdp_enabled'):
                    debug.attention(f"Math SDP enabled: {torch.backends.cuda.math_sdp_enabled()}")
                    
        except Exception as e:
            debug.error(f"Could not set attention processor: {e}", exc_info=True)
            print(f"[Flux] Warning: Could not set attention processor: {e}")
            # Try alternative: completely disable memory efficient attention
            try:
                # Disable xformers if enabled
                if hasattr(self.pipe, 'disable_xformers_memory_efficient_attention'):
                    self.pipe.disable_xformers_memory_efficient_attention()
                    debug.attention("Disabled xformers memory efficient attention")
                    print("[Flux] Disabled xformers memory efficient attention")
            except Exception as e2:
                debug.error(f"Could not disable xformers: {e2}")
        
        debug.gpu("After attention configuration")
        
        # ===== 7. MEMORY OPTIMIZATION =====
        if self.offload_to_cpu:
            if progress_callback:
                progress_callback("Enabling CPU offload...")
            
            # CRITICAL: T5 is managed by Accelerate disk offload
            # We must protect it from Diffusers' CPU offload
            temp_te2 = self.pipe.text_encoder_2
            self.pipe.text_encoder_2 = None
            
            self.pipe.enable_model_cpu_offload()
            
            self.pipe.text_encoder_2 = temp_te2
            print("[Flux] Protected T5 from CPU offload (disk-managed)")
        else:
            self.pipe.to(self.device)
        
        self.is_loaded = True
        self.current_model = checkpoint_path.name
        print(f"[Flux] Model loaded successfully")
    
    def _load_clip(self, clip_path: Path, config_repo: str):
        """Load CLIP text encoder from local file."""
        from transformers import CLIPTextModel, AutoConfig
        from safetensors.torch import load_file as load_safetensors
        
        try:
            config = AutoConfig.from_pretrained(config_repo, subfolder="text_encoder")
            state_dict = load_safetensors(str(clip_path))
            model = CLIPTextModel(config)
            model.load_state_dict(state_dict, strict=False)
            return model.to(self.dtype)
        except Exception as e:
            print(f"[Flux] Custom CLIP load failed: {e}, using HF default")
            from transformers import CLIPTextModel
            return CLIPTextModel.from_pretrained(
                config_repo,
                subfolder="text_encoder",
                torch_dtype=self.dtype,
            )
    
    def _load_t5(self, t5_path: Path, config_repo: str, 
                 progress_callback: Optional[Callable[[str], None]] = None):
        """Load T5 encoder - handles both GGUF and safetensors."""
        from transformers import T5EncoderModel, AutoConfig
        import tempfile
        
        if t5_path.suffix.lower() == ".gguf":
            print("[Flux] T5 GGUF detected - using disk offload...")
            
            temp_dir = Path(tempfile.gettempdir()) / "flux_t5_offload"
            temp_dir.mkdir(exist_ok=True)
            
            t5_config = AutoConfig.from_pretrained(config_repo, subfolder="text_encoder_2")
            
            return T5EncoderModel.from_pretrained(
                str(t5_path.parent),
                gguf_file=t5_path.name,
                config=t5_config,
                subfolder="",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                max_memory={0: 0, "cpu": "4GB"},
                offload_folder=str(temp_dir),
                offload_state_dict=True,
            )
        else:
            from safetensors.torch import load_file as load_safetensors
            config = AutoConfig.from_pretrained(config_repo, subfolder="text_encoder_2")
            state_dict = load_safetensors(str(t5_path))
            model = T5EncoderModel(config)
            model.load_state_dict(state_dict, strict=False)
            return model.to(self.dtype)
    
    def _load_transformer(self, checkpoint_path: Path, config_repo: str,
                          progress_callback: Optional[Callable[[str], None]] = None):
        """Load Flux transformer - handles both GGUF and safetensors."""
        from diffusers import FluxTransformer2DModel
        
        if self.is_gguf:
            print(f"[Flux] Loading GGUF transformer: {checkpoint_path}")
            try:
                from diffusers import GGUFQuantizationConfig
                
                return FluxTransformer2DModel.from_single_file(
                    str(checkpoint_path),
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                    torch_dtype=torch.bfloat16,
                )
            except ImportError:
                raise RuntimeError(
                    "GGUF support requires diffusers >= 0.31.0\n"
                    "Run: pip install --upgrade diffusers"
                )
        else:
            print(f"[Flux] Loading safetensors transformer: {checkpoint_path}")
            return FluxTransformer2DModel.from_single_file(
                str(checkpoint_path),
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
    
    def _load_vae(self, vae_path: Optional[Path], config_repo: str):
        """Load Flux VAE - must be Flux-compatible (16 channels)."""
        from diffusers import AutoencoderKL
        
        if vae_path and vae_path.exists():
            vae_name_lower = vae_path.name.lower()
            is_flux_vae = (
                "flux" in vae_name_lower or
                vae_name_lower in ["ae.safetensors", "ae.sft"] or
                vae_path.suffix.lower() == ".sft"
            )
            
            if is_flux_vae:
                print(f"[Flux] Loading custom VAE: {vae_path}")
                try:
                    return AutoencoderKL.from_single_file(
                        str(vae_path),
                        torch_dtype=self.dtype,
                    )
                except Exception as e:
                    print(f"[Flux] Custom VAE failed: {e}")
            else:
                print(f"[Flux] Skipping non-Flux VAE: {vae_path.name}")
        
        print("[Flux] Loading default VAE from HuggingFace...")
        return AutoencoderKL.from_pretrained(
            config_repo,
            subfolder="vae",
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
    
    def generate(self,
                 config: GenerationConfig,
                 progress_callback: Optional[Callable[[int, int], None]] = None):
        """Generate image with Flux."""
        from PIL import Image
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        debug.separator("FLUX GENERATION")
        debug.generation_config(config)
        debug.reset_step_timer()
        
        # Prepare generator for seed
        seed = config.seed
        if seed == -1:
            import random
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        debug.generation(f"Seed: {seed}")
        
        # Flux-specific parameters
        steps = config.steps
        if "schnell" in (self.current_model or "").lower() and steps > 8:
            debug.generation(f"Schnell model detected, reducing steps from {steps} to 4")
            print(f"[Flux] Schnell model detected, using 4 steps instead of {steps}")
            steps = 4
        
        cfg = min(config.cfg_scale, 4.0)
        if config.cfg_scale > 4.0:
            debug.generation(f"CFG clamped from {config.cfg_scale} to {cfg}")
            print(f"[Flux] CFG clamped from {config.cfg_scale} to {cfg} (Flux optimal range)")
        
        debug.generation(f"Final params: {config.width}x{config.height}, {steps} steps, CFG {cfg}")
        print(f"[Flux] Generating: {config.width}x{config.height}, {steps} steps, CFG {cfg}, seed {seed}")
        
        if config.negative_prompt:
            debug.warn("Negative prompts ignored (not supported by Flux)")
            print("[Flux] Warning: Negative prompts ignored (not supported by Flux)")
        
        debug.gpu("Before generation")
        
        # Generate with timing
        debug.generation("Starting inference...")
        with debug.timer("Flux inference"):
            try:
                result = self.pipe(
                    prompt=config.prompt,
                    width=config.width,
                    height=config.height,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=generator,
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
        if self.text_encoder:
            del self.text_encoder
            self.text_encoder = None
        if self.text_encoder_2:
            del self.text_encoder_2
            self.text_encoder_2 = None
        if self.vae:
            del self.vae
            self.vae = None
        if self.transformer:
            del self.transformer
            self.transformer = None
        if self.pipe:
            del self.pipe
            self.pipe = None
        
        self.is_loaded = False
        self.current_model = None
        self._clear_memory()
        print("[Flux] Model unloaded")
    
    def apply_loras(self, loras: List[Dict[str, Any]], lora_dir: Path) -> None:
        """Apply LoRAs to Flux model."""
        super().apply_loras(loras, lora_dir)
