"""xAI (Grok) Provider"""
import os
from typing import Generator
from .base_provider import BaseProvider, Message, ProviderStatus

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class XAIProvider(BaseProvider):
    """xAI (Grok) API provider."""
    
    BASE_URL = "https://api.x.ai/v1"
    
    def __init__(self, model: str = None, api_key: str = None):
        self.api_key = api_key or os.environ.get("XAI_API_KEY", "")
        self.model_name = model
        self._client = None
        self._configured = False

        if OPENAI_AVAILABLE and self.api_key:
            self._configure()

    def _configure(self):
        try:
            self._client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)
            self._configured = True
        except Exception as e:
            print(f"xAI configuration error: {e}")
            self._configured = False

    @staticmethod
    def list_available_models(api_key: str) -> list[str]:
        """Fetch models from xAI."""
        if not OPENAI_AVAILABLE: return []
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            models = client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            print(f"Error fetching xAI models: {e}")
            return []

    def send_message(self, prompt: str, history: list[Message] = None, system_prompt: str = None) -> str:
        if not self._configured: raise RuntimeError("xAI not configured")
        
        messages = []
        if system_prompt: messages.append({"role": "system", "content": system_prompt})
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return resp.choices[0].message.content

    def stream_message(self, prompt: str, history: list[Message] = None, system_prompt: str = None) -> Generator[str, None, None]:
        if not self._configured: raise RuntimeError("xAI not configured")
        
        messages = []
        if system_prompt: messages.append({"role": "system", "content": system_prompt})
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": prompt})

        stream = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_status(self) -> ProviderStatus:
        return ProviderStatus(available=self._configured, message="Ready" if self._configured else "Not Configured", model=self.model_name)
