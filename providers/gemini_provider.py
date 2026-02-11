"""Gemini API Provider — Chat, Image, and Video Generation"""
import os
import time
from typing import Generator, Optional

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from .base_provider import BaseProvider, Message, ProviderStatus


class GeminiProvider(BaseProvider):
    """Google Gemini API provider for cloud-based LLM."""
    
    def __init__(self, model: str = None, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._model = None
        self._configured = False
        self.model_name = model
        
        if GENAI_AVAILABLE and self.api_key:
            self._configure()
            # Auto-select first available model if none specified
            if not self.model_name:
                models = self.list_available_models()
                if models:
                    self.model_name = models[0]
                    self._model = genai.GenerativeModel(self.model_name)
    
    def _configure(self):
        """Configure the Gemini API."""
        try:
            genai.configure(api_key=self.api_key)
            if self.model_name:
                self._model = genai.GenerativeModel(self.model_name)
            self._configured = True
        except Exception as e:
            print(f"Gemini configuration error: {e}")
            self._configured = False
    
    # Curated LLM chat models — text-only, no image gen, max 5
    CHAT_MODELS = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
    ]

    # Gemini image generation models (for image gen tab)
    IMAGE_MODELS = [
        "gemini-2.5-flash-image",          # Nano Banana — fast image gen
        "gemini-3-pro-image-preview",       # Pro — higher quality image gen
    ]

    # Veo video generation models
    VIDEO_MODELS = [
        "veo-3.1-generate-preview",         # High quality + audio
        "veo-3.1-fast-generate-preview",    # Fast variant
    ]

    @classmethod
    def list_available_models(cls, api_key: str = None) -> list[str]:
        """Return curated list of Gemini LLM chat models (max 5).

        Only returns text chat models — no image generation, no nano, no experimental.
        """
        return cls.CHAT_MODELS.copy()

    @classmethod
    def list_image_models(cls, api_key: str = None) -> list[str]:
        """Return Gemini models that support image generation."""
        return cls.IMAGE_MODELS.copy()

    @classmethod
    def list_video_models(cls, api_key: str = None) -> list[str]:
        """Return Veo models that support video generation."""
        return cls.VIDEO_MODELS.copy()

    @classmethod
    def clear_model_cache(cls):
        """No-op — models are now a curated static list."""
        pass
    
    def _build_history(self, history: list[Message] = None) -> list[dict]:
        """Convert Message objects to Gemini format."""
        if not history:
            return []
        
        gemini_history = []
        for msg in history:
            role = "user" if msg.role == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [msg.content]
            })
        return gemini_history
    
    def send_message(self, prompt: str, history: list[Message] = None,
                     system_prompt: str = None) -> str:
        """Send a message to Gemini and get a response."""
        if not self._configured:
            raise RuntimeError("Gemini provider not configured. Set GEMINI_API_KEY.")
        
        if not self.model_name:
            raise RuntimeError("No model selected. Please select a model in Settings.")
        
        try:
            # Create chat with history
            chat_history = self._build_history(history)
            
            # Use system instruction if provided
            if system_prompt:
                model = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=system_prompt
                )
            else:
                model = self._model
            
            chat = model.start_chat(history=chat_history)
            response = chat.send_message(prompt)
            
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            if "API_KEY" in error_msg.upper() or "401" in error_msg:
                raise RuntimeError("Invalid Gemini API key. Please check your key in Settings.")
            elif "QUOTA" in error_msg.upper() or "429" in error_msg:
                raise RuntimeError("Gemini API quota exceeded. Please try again later.")
            else:
                raise RuntimeError(f"Gemini error: {error_msg}")
    
    def stream_message(self, prompt: str, history: list[Message] = None,
                       system_prompt: str = None) -> Generator[str, None, None]:
        """Stream a response from Gemini token by token."""
        if not self._configured:
            raise RuntimeError("Gemini provider not configured. Set GEMINI_API_KEY.")
        
        if not self.model_name:
            raise RuntimeError("No model selected. Please select a model in Settings.")
        
        try:
            chat_history = self._build_history(history)
            
            if system_prompt:
                model = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=system_prompt
                )
            else:
                model = self._model
            
            chat = model.start_chat(history=chat_history)
            response = chat.send_message(prompt, stream=True)
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            yield f"[Error: {str(e)}]"
    
    def get_status(self) -> ProviderStatus:
        """Check Gemini availability."""
        if not GENAI_AVAILABLE:
            return ProviderStatus(
                available=False,
                message="google-generativeai package not installed"
            )
        
        if not self.api_key:
            return ProviderStatus(
                available=False,
                message="GEMINI_API_KEY not set"
            )
        
        if not self._configured:
            return ProviderStatus(
                available=False,
                message="Failed to configure Gemini API"
            )
        
        if not self.model_name:
            return ProviderStatus(
                available=False,
                message="No model selected"
            )
        
        return ProviderStatus(
            available=True,
            message="Ready",
            model=self.model_name
        )

    # ------------------------------------------------------------------
    # Veo Video Generation
    # ------------------------------------------------------------------

    def create_and_wait_video(self, prompt: str, model: str = "veo-3.1-generate-preview",
                              aspect_ratio: str = "16:9", resolution: str = "720p",
                              duration: int = 8, save_path: str = "output.mp4",
                              progress_callback=None,
                              poll_interval: int = 10) -> str:
        """Full Veo pipeline: submit → poll → download. Returns save_path.
        Uses google.genai SDK (not the legacy google.generativeai).
        """
        from google import genai as genai_new
        from google.genai import types

        client = genai_new.Client(api_key=self.api_key)

        if progress_callback:
            progress_callback(f"Submitting to {model}...")

        operation = client.models.generate_videos(
            model=model,
            prompt=prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                duration_seconds=str(duration),
            ),
        )

        if progress_callback:
            progress_callback("Job queued, rendering video...")

        elapsed = 0
        while not operation.done:
            elapsed += poll_interval
            if progress_callback:
                progress_callback(f"Rendering... {elapsed}s elapsed")
            time.sleep(poll_interval)
            operation = client.operations.get(operation)

        # Download the video
        if not operation.response or not operation.response.generated_videos:
            raise RuntimeError("Veo returned no video data.")

        if progress_callback:
            progress_callback("Downloading video...")

        generated_video = operation.response.generated_videos[0]
        client.files.download(file=generated_video.video)
        generated_video.video.save(save_path)

        return save_path
