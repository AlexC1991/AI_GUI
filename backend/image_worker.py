"""
AI_GUI Backend - Image Generation Worker
QThread wrapper for non-blocking image generation

Works with the new modular pipeline architecture:
- backend/image_generator.py (Manager)
- backend/pipelines/sd15_pipeline.py
- backend/pipelines/sdxl_pipeline.py
- backend/pipelines/flux_pipeline.py
"""
from PySide6.QtCore import QThread, Signal
from pathlib import Path


class ImageWorker(QThread):
    """Background worker for image generation."""
    
    progress = Signal(str)
    step_progress = Signal(int, int)
    finished = Signal(str)
    error = Signal(str)
    auth_required = Signal(str)  # Emits model name when HF auth needed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._prompt = ""
        self._negative_prompt = ""
        self._model = None
        self._sampler = "euler"
        self._width = 512
        self._height = 512
        self._steps = 20
        self._cfg_scale = 7.0
        self._seed = -1
        self._loras = []  # List of {"name": str, "strength": float}
        self._vae = None
        self._text_encoders = []
        self._checkpoint_dir = "models/checkpoints"
        self._lora_dir = "models/loras"
        self._vae_dir = "models/vae"
        self._text_encoder_dir = "models/text_encoders"
        self._output_dir = "outputs/images"
    
    def setup(self,
              prompt: str,
              negative_prompt: str = None,
              model: str = None,
              width: int = 512,
              height: int = 512,
              steps: int = 25,
              cfg_scale: float = 7.0,
              seed: int = -1,
              sampler: str = "euler",
              lora: str = None,
              lora_weight: float = 1.0,
              vae: str = None,
              text_encoders: list = None,
              checkpoint_dir: str = "models/checkpoints",
              lora_dir: str = "models/loras",
              vae_dir: str = "models/vae",
              text_encoder_dir: str = "models/text_encoders",
              output_dir: str = "outputs/images"):
        """Configure the generation job."""
        
        self._prompt = prompt
        self._negative_prompt = negative_prompt or ""
        self._model = model
        self._width = width
        self._height = height
        self._steps = steps
        self._cfg_scale = cfg_scale
        self._seed = seed if seed is not None else -1
        self._sampler = sampler
        self._vae = vae
        self._text_encoders = text_encoders or []
        
        # Build LoRA list (supports single or multiple)
        self._loras = []
        if lora and lora != "None":
            self._loras.append({
                "name": lora,
                "strength": lora_weight
            })
        
        # Directories
        self._checkpoint_dir = checkpoint_dir
        self._lora_dir = lora_dir
        self._vae_dir = vae_dir
        self._text_encoder_dir = text_encoder_dir
        self._output_dir = output_dir
    
    def add_lora(self, lora_name: str, strength: float = 1.0):
        """Add additional LoRA to the generation."""
        if lora_name and lora_name != "None":
            self._loras.append({
                "name": lora_name,
                "strength": strength
            })
    
    def run(self):
        """Execute generation in background thread."""
        try:
            # Import from new location
            from backend.image_generator import get_generator, GenerationConfig
            
            self.progress.emit("Initializing...")
            
            # Get the manager instance
            generator = get_generator(
                checkpoint_dir=self._checkpoint_dir,
                lora_dir=self._lora_dir,
                vae_dir=self._vae_dir,
                text_encoder_dir=self._text_encoder_dir,
                output_dir=self._output_dir
            )
            
            # Load model (manager routes to correct pipeline)
            self.progress.emit(f"Loading: {self._model or 'default'}...")
            generator.load_model(
                model_id=self._model,
                vae_name=self._vae,
                text_encoder_names=self._text_encoders,
                sampler=self._sampler,
                progress_callback=lambda msg: self.progress.emit(msg)
            )
            
            # Create generation config
            config = GenerationConfig(
                prompt=self._prompt,
                negative_prompt=self._negative_prompt,
                width=self._width,
                height=self._height,
                steps=self._steps,
                cfg_scale=self._cfg_scale,
                seed=self._seed,
                sampler=self._sampler,
                vae_name=self._vae,
                loras=self._loras,
                text_encoders=self._text_encoders,
            )
            
            self.progress.emit("Generating...")
            
            # Generate and save (manager handles everything)
            output_path = generator.generate_and_save(
                config=config,
                progress_callback=lambda step, total: self._emit_step(step, total)
            )
            
            self.progress.emit("Done!")
            self.finished.emit(str(output_path))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            error_str = str(e)
            
            # Check for HuggingFace auth errors
            if "401" in error_str and ("gated" in error_str.lower() or "unauthorized" in error_str.lower()):
                self.auth_required.emit(self._model or "this model")
            elif "403" in error_str and "gated" in error_str.lower():
                self.error.emit(
                    f"⚠️ License required: You need to accept the model license.\n"
                    f"Visit the model's HuggingFace page and click 'Agree and access repository'."
                )
            elif "12GB" in error_str or "VRAM" in error_str:
                # VRAM error from Flux safetensors check
                self.error.emit(
                    f"⚠️ Insufficient VRAM: {error_str}\n"
                    f"Use a GGUF quantized model for 8GB cards."
                )
            else:
                self.error.emit(error_str)
    
    def _emit_step(self, step: int, total: int):
        """Emit step progress signals."""
        self.step_progress.emit(step, total)
        self.progress.emit(f"Step {step}/{total}")
    
    def abort(self):
        """Request thread interruption."""
        self.requestInterruption()


class ModelLoaderWorker(QThread):
    """Background worker for model pre-loading."""
    
    progress = Signal(str)
    finished = Signal(bool, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None
        self._sampler = None
        self._vae = None
        self._text_encoders = []
        self._checkpoint_dir = "models/checkpoints"
        self._lora_dir = "models/loras"
        self._vae_dir = "models/vae"
        self._text_encoder_dir = "models/text_encoders"
        self._output_dir = "outputs/images"
    
    def setup(self, 
              model: str, 
              sampler: str = None,
              vae: str = None,
              text_encoders: list = None,
              checkpoint_dir: str = "models/checkpoints",
              lora_dir: str = "models/loras",
              vae_dir: str = "models/vae",
              text_encoder_dir: str = "models/text_encoders",
              output_dir: str = "outputs/images"):
        """Configure the preload job."""
        self._model = model
        self._sampler = sampler
        self._vae = vae
        self._text_encoders = text_encoders or []
        self._checkpoint_dir = checkpoint_dir
        self._lora_dir = lora_dir
        self._vae_dir = vae_dir
        self._text_encoder_dir = text_encoder_dir
        self._output_dir = output_dir
    
    def run(self):
        """Execute model loading in background."""
        try:
            from backend.image_generator import get_generator
            
            self.progress.emit(f"Pre-loading {self._model}...")
            
            generator = get_generator(
                checkpoint_dir=self._checkpoint_dir,
                lora_dir=self._lora_dir,
                vae_dir=self._vae_dir,
                text_encoder_dir=self._text_encoder_dir,
                output_dir=self._output_dir
            )
            
            generator.load_model(
                model_id=self._model,
                vae_name=self._vae,
                text_encoder_names=self._text_encoders,
                sampler=self._sampler,
                progress_callback=lambda msg: self.progress.emit(msg)
            )
            
            self.finished.emit(True, f"Ready: {self._model}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e))


class ModelUnloaderWorker(QThread):
    """Background worker to unload model and free VRAM."""
    
    progress = Signal(str)
    finished = Signal()
    
    def run(self):
        """Execute model unload in background."""
        from backend.image_generator import reset_generator
        
        self.progress.emit("Unloading model...")
        reset_generator()
        self.progress.emit("Model unloaded")
        self.finished.emit()


class VRAMMonitorWorker(QThread):
    """Background worker to monitor VRAM usage."""
    
    vram_updated = Signal(float, float)  # (used_gb, total_gb)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self._interval = 1.0  # Update every second
    
    def run(self):
        """Continuously monitor VRAM."""
        import time
        try:
            import torch
            if not torch.cuda.is_available():
                return
                
            while self._running:
                used = torch.cuda.memory_allocated(0) / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.vram_updated.emit(used, total)
                time.sleep(self._interval)
        except Exception:
            pass
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
