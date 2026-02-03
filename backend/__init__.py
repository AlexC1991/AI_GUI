"""
AI_GUI Backend Module
Provides image generation with ZLUDA/AMD GPU support
"""

from .image_generator import ImageGenerator, GenerationConfig
from .image_worker import ImageWorker, ModelLoaderWorker

__all__ = [
    'ImageGenerator',
    'GenerationConfig', 
    'ImageWorker',
    'ModelLoaderWorker'
]
