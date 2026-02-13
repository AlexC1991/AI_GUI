"""Z.ai (ZhipuAI) Provider"""
import os
from typing import Generator
from .base_provider import BaseProvider, Message, ProviderStatus

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class ZaiProvider(BaseProvider):
    """Z.ai (ZhipuAI) API provider."""
    
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
    
    def __init__(self, model: str = None, api_key: str = None):
        self.api_key = api_key or os.environ.get("ZHIPUAI_API_KEY", "")
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
            print(f"Z.ai configuration error: {e}")
            self._configured = False

    # Model IDs that are NOT usable for chat completions (embeddings, image, video, code-interp, etc.)
    _EXCLUDED_PATTERNS = ["embedding", "cogview", "cogvideo", "charglm", "codegeex", "alltools"]

    @staticmethod
    def list_available_models(api_key: str) -> list[str]:
        """Fetch chat-compatible models from ZhipuAI."""
        if not OPENAI_AVAILABLE: return []
        try:
            client = OpenAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4/")
            models = client.models.list()
            result = []
            for m in models.data:
                mid = m.id.lower()
                if any(pat in mid for pat in ZaiProvider._EXCLUDED_PATTERNS):
                    continue
                result.append(m.id)
            return result
        except Exception as e:
            print(f"Error fetching Z.ai models: {e}")
            return []

    def send_message(self, prompt: str, history: list[Message] = None, system_prompt: str = None) -> str:
        if not self._configured: raise RuntimeError("Z.ai not configured")
        
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
        if not self._configured: raise RuntimeError("Z.ai not configured")
        
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
