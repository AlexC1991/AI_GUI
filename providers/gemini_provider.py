"""Gemini API Provider"""
import os
from typing import Generator, Optional

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from .base_provider import BaseProvider, Message, ProviderStatus


class GeminiProvider(BaseProvider):
    """Google Gemini API provider for cloud-based LLM."""
    
    # Cache for available models
    _available_models = None
    
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
    
    # Pro tier models (not always returned by list_models API)
    PRO_TIER_MODELS = [
        # Auto-updating aliases (recommended)
        "gemini-pro-latest",
        "gemini-flash-latest",
        # Gemini 3 Series (Frontier)
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        # Gemini 2.5 Series (Stable)
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]
    
    @classmethod
    def list_available_models(cls, api_key: str = None) -> list[str]:
        """Fetch list of available models from Gemini API.
        
        Returns:
            List of model names that support generateContent
        """
        if not GENAI_AVAILABLE:
            return cls.PRO_TIER_MODELS.copy()
        
        # Use cached result if available
        if cls._available_models is not None:
            return cls._available_models
        
        try:
            key = api_key or os.environ.get("GEMINI_API_KEY", "")
            if key:
                genai.configure(api_key=key)
            
            # Start with Pro tier models
            models = set(cls.PRO_TIER_MODELS)
            
            # Add models from API
            for model in genai.list_models():
                # Only include models that support generateContent
                if "generateContent" in model.supported_generation_methods:
                    # Extract the model name (remove 'models/' prefix)
                    name = model.name
                    if name.startswith("models/"):
                        name = name[7:]
                    
                    # Filter to only useful models (flash and pro)
                    skip_patterns = [
                        "aqa", "embedding", "vision", "legacy", 
                        "exp", "thinking", "learnlm", "imagen", "tts"
                    ]
                    if any(pattern in name.lower() for pattern in skip_patterns):
                        continue
                    
                    # Only include flash and pro models
                    if "flash" in name.lower() or "pro" in name.lower():
                        models.add(name)
            
            # Convert to list and sort
            models = list(models)
            
            def sort_key(m):
                # Priority: latest aliases first, then by version
                if "latest" in m:
                    return (0, 0 if "flash" in m else 1, m)
                elif "-3-" in m:
                    return (1, 0 if "flash" in m else 1, m)
                elif "2.5" in m:
                    return (2, 0 if "flash" in m else 1, m)
                elif "2.0" in m:
                    return (3, 0 if "flash" in m else 1, m)
                elif "1.5" in m:
                    return (4, 0 if "flash" in m else 1, m)
                else:
                    return (5, 0 if "flash" in m else 1, m)
            
            models.sort(key=sort_key)
            cls._available_models = models
            return models
            
        except Exception as e:
            print(f"Error listing models: {e}")
            return cls.PRO_TIER_MODELS.copy()
    
    @classmethod
    def clear_model_cache(cls):
        """Clear the cached model list."""
        cls._available_models = None
    
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
