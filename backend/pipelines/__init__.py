"""
AI_GUI Pipelines Package

Each pipeline handles a specific model architecture:
- SD15Pipeline: Stable Diffusion 1.5 and compatible
- SDXLPipeline: SDXL, Pony, Illustrious
- FluxPipeline: Flux.1 Schnell/Dev (GGUF and safetensors)
"""

from .base_pipeline import BasePipeline, PipelineInfo, GenerationConfig
from .sd15_pipeline import SD15Pipeline
from .sdxl_pipeline import SDXLPipeline
from .flux_pipeline import FluxPipeline

__all__ = [
    "BasePipeline",
    "PipelineInfo", 
    "GenerationConfig",
    "SD15Pipeline",
    "SDXLPipeline",
    "FluxPipeline",
]
