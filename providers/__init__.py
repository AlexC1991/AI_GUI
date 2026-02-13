# VoxAI Providers Module
from .base_provider import BaseProvider, Message
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider
from .openrouter_provider import OpenRouterProvider
from .deepseek_provider import DeepSeekProvider
from .kimi_provider import KimiProvider
from .mistral_provider import MistralProvider
from .xai_provider import XAIProvider
from .zai_provider import ZaiProvider

__all__ = [
    "BaseProvider", "Message",
    "GeminiProvider", "OpenAIProvider", "OllamaProvider",
    "OpenRouterProvider", "DeepSeekProvider", "KimiProvider",
    "MistralProvider", "XAIProvider", "ZaiProvider",
]

# Registry: config key → (provider class, display name)
# Used by chat_worker and main_window to resolve providers dynamically
PROVIDER_REGISTRY = {
    "gemini":     (GeminiProvider,     "Gemini"),
    "openai":     (OpenAIProvider,     "OpenAI"),
    "openrouter": (OpenRouterProvider, "OpenRouter"),
    "deepseek":   (DeepSeekProvider,   "DeepSeek"),
    "kimi":       (KimiProvider,       "Kimi"),
    "mistral":    (MistralProvider,    "Mistral"),
    "xai":        (XAIProvider,        "xAI"),
    "zai":        (ZaiProvider,        "Z.ai"),
}
