"""OpenAI API Provider — Chat, Image, and Video Generation"""
import os
import time
from typing import Generator, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base_provider import BaseProvider, Message, ProviderStatus


class OpenAIProvider(BaseProvider):
    """OpenAI API provider for cloud-based LLM and image generation."""

    # Top 5 curated LLM chat models — best of OpenAI
    CHAT_MODELS = [
        "gpt-4.1",       # Fast, great for coding + long context
        "gpt-4.1-mini",  # Affordable everyday model
        "o3",             # Reasoning powerhouse
        "o4-mini",        # Fast reasoning, budget-friendly
        "gpt-4o",         # Proven all-rounder
    ]

    # Reasoning models that need special API params
    REASONING_MODELS = {"o3", "o4-mini"}

    # Image generation models
    IMAGE_MODELS = [
        "gpt-image-1",
        "dall-e-3",
    ]

    # Video generation models (Sora)
    VIDEO_MODELS = [
        "sora-2",
        "sora-2-pro",
    ]

    def __init__(self, model: str = None, api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None
        self._configured = False
        self.model_name = model

        if OPENAI_AVAILABLE and self.api_key:
            self._configure()
            # Auto-select first model if none specified
            if not self.model_name:
                self.model_name = self.CHAT_MODELS[0]

    def _configure(self):
        """Configure the OpenAI client."""
        try:
            self._client = OpenAI(api_key=self.api_key)
            self._configured = True
        except Exception as e:
            print(f"OpenAI configuration error: {e}")
            self._configured = False

    @classmethod
    def list_available_models(cls, api_key: str = None) -> list[str]:
        """Return curated list of OpenAI LLM chat models."""
        return cls.CHAT_MODELS.copy()

    @classmethod
    def list_image_models(cls, api_key: str = None) -> list[str]:
        """Return OpenAI models that support image generation."""
        return cls.IMAGE_MODELS.copy()

    @classmethod
    def list_video_models(cls, api_key: str = None) -> list[str]:
        """Return OpenAI models that support video generation."""
        return cls.VIDEO_MODELS.copy()

    @classmethod
    def clear_model_cache(cls):
        """No-op — models are a curated static list."""
        pass

    def _is_reasoning_model(self) -> bool:
        """Check if the current model is an o-series reasoning model."""
        return self.model_name in self.REASONING_MODELS

    def _build_messages(self, prompt: str, history: list[Message] = None,
                        system_prompt: str = None) -> list[dict]:
        """Convert Message objects to OpenAI chat format."""
        messages = []

        # System prompt: o-series uses "developer" role, GPT uses "system"
        if system_prompt:
            role = "developer" if self._is_reasoning_model() else "system"
            messages.append({"role": role, "content": system_prompt})

        # History
        if history:
            for msg in history:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # Current prompt
        messages.append({"role": "user", "content": prompt})
        return messages

    def send_message(self, prompt: str, history: list[Message] = None,
                     system_prompt: str = None) -> str:
        """Send a message to OpenAI and get a response."""
        if not self._configured:
            raise RuntimeError("OpenAI provider not configured. Set API key.")

        if not self.model_name:
            raise RuntimeError("No model selected. Please select a model in Settings.")

        try:
            messages = self._build_messages(prompt, history, system_prompt)

            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )

            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "invalid_api_key" in error_msg.lower():
                raise RuntimeError("Invalid OpenAI API key. Please check your key in Settings.")
            elif "429" in error_msg or "rate_limit" in error_msg.lower():
                raise RuntimeError("OpenAI rate limit exceeded. Please try again later.")
            elif "insufficient_quota" in error_msg.lower():
                raise RuntimeError("OpenAI quota exceeded. Check your billing at platform.openai.com.")
            else:
                raise RuntimeError(f"OpenAI error: {error_msg}")

    def stream_message(self, prompt: str, history: list[Message] = None,
                       system_prompt: str = None) -> Generator[str, None, None]:
        """Stream a response from OpenAI token by token."""
        if not self._configured:
            raise RuntimeError("OpenAI provider not configured. Set API key.")

        if not self.model_name:
            raise RuntimeError("No model selected. Please select a model in Settings.")

        try:
            messages = self._build_messages(prompt, history, system_prompt)

            stream = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"[Error: {str(e)}]"

    def get_status(self) -> ProviderStatus:
        """Check OpenAI availability."""
        if not OPENAI_AVAILABLE:
            return ProviderStatus(
                available=False,
                message="openai package not installed"
            )

        if not self.api_key:
            return ProviderStatus(
                available=False,
                message="OPENAI_API_KEY not set"
            )

        if not self._configured:
            return ProviderStatus(
                available=False,
                message="Failed to configure OpenAI API"
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
    # Sora Video Generation
    # ------------------------------------------------------------------

    def create_video(self, prompt: str, model: str = "sora-2",
                     duration: int = 4, resolution: str = "1280x720") -> str:
        """Submit a video generation job. Returns the video job ID."""
        if not self._configured:
            raise RuntimeError("OpenAI provider not configured.")

        # Sora API uses 'seconds' (not duration) and 'size' (not resolution)
        # Valid seconds: 4, 8, 12  |  Valid sizes: 1280x720, 720x1280, 1792x1024, 1024x1792
        video = self._client.videos.create(
            model=model,
            prompt=prompt,
            seconds=duration,
            size=resolution,
        )
        return video.id

    def poll_video(self, video_id: str) -> dict:
        """Check status of a video generation job.
        Returns {"status": "queued"|"in_progress"|"completed"|"failed", ...}
        """
        if not self._configured:
            raise RuntimeError("OpenAI provider not configured.")

        video = self._client.videos.retrieve(video_id)
        return {
            "status": video.status,
            "id": video.id,
        }

    def download_video(self, video_id: str, save_path: str):
        """Download a completed video to save_path."""
        if not self._configured:
            raise RuntimeError("OpenAI provider not configured.")

        response = self._client.videos.content(video_id)
        with open(save_path, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

    def create_and_wait_video(self, prompt: str, model: str = "sora-2",
                              duration: int = 4, resolution: str = "1280x720",
                              save_path: str = "output.mp4",
                              progress_callback=None,
                              poll_interval: int = 10) -> str:
        """Full pipeline: create → poll → download. Returns save_path.
        progress_callback(msg: str) is called with status updates.
        """
        if not self._configured:
            raise RuntimeError("OpenAI provider not configured.")

        if progress_callback:
            progress_callback(f"Submitting to {model}...")

        video_id = self.create_video(prompt, model, duration, resolution)

        if progress_callback:
            progress_callback(f"Job queued ({video_id[:8]}...)")

        elapsed = 0
        while True:
            status = self.poll_video(video_id)

            if status["status"] == "completed":
                if progress_callback:
                    progress_callback("Downloading video...")
                self.download_video(video_id, save_path)
                return save_path

            elif status["status"] == "failed":
                raise RuntimeError(f"Sora video generation failed (job {video_id})")

            else:
                elapsed += poll_interval
                if progress_callback:
                    progress_callback(f"Rendering... {elapsed}s elapsed")
                time.sleep(poll_interval)
