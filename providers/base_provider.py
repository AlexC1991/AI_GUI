"""Base Provider Abstract Class"""
from abc import ABC, abstractmethod
from typing import Generator, Optional
from dataclasses import dataclass


@dataclass
class Message:
    """Represents a chat message."""
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ProviderStatus:
    """Provider availability status."""
    available: bool
    message: str
    model: Optional[str] = None


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    @abstractmethod
    def send_message(self, prompt: str, history: list[Message] = None, 
                     system_prompt: str = None) -> str:
        """Send a message and get a response.
        
        Args:
            prompt: The user's message
            history: Previous conversation messages
            system_prompt: Optional system prompt for character/behavior
            
        Returns:
            The assistant's response text
        """
        pass
    
    @abstractmethod
    def stream_message(self, prompt: str, history: list[Message] = None,
                       system_prompt: str = None) -> Generator[str, None, None]:
        """Stream a response token by token.
        
        Args:
            prompt: The user's message
            history: Previous conversation messages
            system_prompt: Optional system prompt
            
        Yields:
            Response text chunks as they arrive
        """
        pass
    
    @abstractmethod
    def get_status(self) -> ProviderStatus:
        """Check if the provider is available and configured.
        
        Returns:
            ProviderStatus with availability info
        """
        pass
