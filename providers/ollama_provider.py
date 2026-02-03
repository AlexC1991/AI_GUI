"""Ollama Local LLM Provider"""
import os
import json
from typing import Generator, Optional, List

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .base_provider import BaseProvider, Message, ProviderStatus


class OllamaProvider(BaseProvider):
    """Ollama provider for local LLM inference."""
    
    DEFAULT_HOST = "http://localhost:11434"
    
    def __init__(self, model: str = None, host: str = None):
        self.host = host or os.environ.get("OLLAMA_HOST", self.DEFAULT_HOST)
        self.model_name = model
        self._available = None
    
    def _api_url(self, endpoint: str) -> str:
        """Build full API URL."""
        return f"{self.host}{endpoint}"
    
    def is_running(self) -> bool:
        """Check if Ollama server is running."""
        if not REQUESTS_AVAILABLE:
            return False
        try:
            response = requests.get(self._api_url("/"), timeout=2)
            return response.status_code == 200
        except:
            return False
    
    @classmethod
    def list_local_models(cls, host: str = None) -> List[str]:
        """Get list of locally available models."""
        if not REQUESTS_AVAILABLE:
            return []
        
        host = host or os.environ.get("OLLAMA_HOST", cls.DEFAULT_HOST)
        try:
            response = requests.get(f"{host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return sorted(models)
        except:
            pass
        return []
    
    def _build_messages(self, prompt: str, history: List[Message] = None, 
                        system_prompt: str = None) -> List[dict]:
        """Build messages array for Ollama chat API."""
        messages = []
        
        # System prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # History
        if history:
            for msg in history:
                role = "user" if msg.role == "user" else "assistant"
                messages.append({"role": role, "content": msg.content})
        
        # Current prompt
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def send_message(self, prompt: str, history: List[Message] = None,
                     system_prompt: str = None) -> str:
        """Send message to Ollama and get response."""
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library not available")
        
        if not self.model_name:
            raise RuntimeError("No model selected. Please select a model in Settings.")
        
        if not self.is_running():
            raise RuntimeError("Ollama is not running. Start it with 'ollama serve'")
        
        messages = self._build_messages(prompt, history, system_prompt)
        
        try:
            response = requests.post(
                self._api_url("/api/chat"),
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("message", {}).get("content", "")
            else:
                raise RuntimeError(f"Ollama error: {response.text}")
                
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama request timed out")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {str(e)}")
    
    def stream_message(self, prompt: str, history: List[Message] = None,
                       system_prompt: str = None) -> Generator[str, None, None]:
        """Stream response from Ollama."""
        if not REQUESTS_AVAILABLE:
            yield "[Error: requests library not available]"
            return
        
        if not self.model_name:
            yield "[Error: No model selected]"
            return
        
        if not self.is_running():
            yield "[Error: Ollama is not running. Start with 'ollama serve']"
            return
        
        messages = self._build_messages(prompt, history, system_prompt)
        
        try:
            response = requests.post(
                self._api_url("/api/chat"),
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": True
                },
                stream=True,
                timeout=120
            )
            
            if response.status_code != 200:
                yield f"[Error: {response.text}]"
                return
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            yield f"[Error: {str(e)}]"
    
    def get_status(self) -> ProviderStatus:
        """Check Ollama availability."""
        if not REQUESTS_AVAILABLE:
            return ProviderStatus(
                available=False,
                message="requests library not installed"
            )
        
        if not self.is_running():
            return ProviderStatus(
                available=False,
                message="Ollama not running"
            )
        
        models = self.list_local_models(self.host)
        if not models:
            return ProviderStatus(
                available=False,
                message="No models installed (run 'ollama pull llama3')"
            )
        
        if not self.model_name:
            return ProviderStatus(
                available=False,
                message="No model selected"
            )
        
        if self.model_name not in models:
            return ProviderStatus(
                available=False,
                message=f"Model '{self.model_name}' not found locally"
            )
        
        return ProviderStatus(
            available=True,
            message="Ready",
            model=self.model_name
        )
