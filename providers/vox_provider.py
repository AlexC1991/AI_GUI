"""
VoxAI Provider Implementation

IMPORTANT: This uses lazy loading for VoxAPI to ensure the backend
setup in main.py has time to run before llama_cpp is imported.

Features:
- Lazy loading to avoid import-time issues
- Fallback status reporting
- Proper error handling with recovery
"""

import sys
import os
import time
from typing import Generator
from providers.base_provider import BaseProvider, ProviderStatus, Message

# Path to VoxAI API (one level up from providers/, then into VoxAI_Chat_API/)
VOX_API_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "VoxAI_Chat_API")
)

# Add to Python path if not already there
if VOX_API_PATH not in sys.path:
    sys.path.insert(0, VOX_API_PATH)

# Lazy loading placeholders
VoxAPI = None
VOX_AVAILABLE = None


def _lazy_load_vox_api():
    """
    Lazy load VoxAPI - called when VoxProvider is instantiated,
    NOT when this module is imported.
    """
    global VoxAPI, VOX_AVAILABLE
    
    if VOX_AVAILABLE is not None:
        return VOX_AVAILABLE
    
    try:
        from vox_api import VoxAPI as _VoxAPI
        VoxAPI = _VoxAPI
        VOX_AVAILABLE = True
        print("[VoxProvider] ✓ VoxAPI loaded")
        return True
    except ImportError as e:
        print(f"[VoxProvider] ✗ VoxAPI import failed: {e}")
        VOX_AVAILABLE = False
        return False


class VoxProvider(BaseProvider):
    """Provider for the custom VoxAI Optimized Engine with GPU fallback support."""
    
    def __init__(self, model_name=None):
        self.engine = None
        self.init_error = None
        self.status = ProviderStatus(
            available=False, 
            message="Initializing...", 
            model=None
        )
        
        # Lazy load VoxAPI
        if not _lazy_load_vox_api():
            self.status = ProviderStatus(
                available=False, 
                message="VoxAI API not found. Ensure 'VoxAI_Chat_API' folder exists.",
                model=None
            )
            return

        try:
            print("[VoxProvider] Initializing engine...")
            
            # Resolve Model Name to Path
            model_path = None
            if model_name and not model_name.startswith("Loading") and not model_name.startswith("("):
                model_path = self._resolve_model(model_name)
                print(f"[VoxProvider] Model: {model_name} -> {model_path}")

            # Initialize engine with retry/fallback support
            self.engine = VoxAPI(
                model_path=model_path, 
                verbose=True,
                max_retries=2,      # Try GPU twice before fallback
                retry_delay=1.5     # Wait 1.5s between retries
            )
            
            # Get stats and build status message
            stats = self.engine.get_stats()
            
            if self.engine.using_fallback:
                status_msg = "Online (CPU Fallback - GPU retry on next load)"
            else:
                status_msg = f"Online ({stats['mode']})"
            
            self.status = ProviderStatus(
                available=True,
                message=status_msg,
                model=self.engine.model_name
            )
            
            print(f"[VoxProvider] ✓ Ready: {stats['mode']} | Layers: {stats['gpu_layers']}")
            
        except Exception as e:
            print(f"[VoxProvider] ✗ Init failed: {e}")
            import traceback
            traceback.print_exc()
            self.init_error = str(e)
            self.status = ProviderStatus(
                available=False,
                message=f"Initialization Error: {e}",
                model=None
            )

    def _resolve_model(self, name: str) -> str:
        """Find the actual .gguf file for a given name (short or long)."""
        models_dir = os.path.join(VOX_API_PATH, "models")
        if not os.path.exists(models_dir):
            return None
        
        # 1. Exact Match (full filename)
        potential_path = os.path.join(models_dir, name)
        if os.path.exists(potential_path):
            return potential_path
        
        # 2. Short Name Match
        for f in os.listdir(models_dir):
            if not f.endswith(".gguf"):
                continue
            
            clean = f.replace(".gguf", "").replace("-", " ").replace("_", " ")
            parts = clean.split()
            short = clean
            if len(parts) >= 2: 
                short = f"{parts[0]} {parts[1]}"
            
            if short.lower() == name.lower():
                return os.path.join(models_dir, f)
        
        # 3. Substring Fallback
        for f in os.listdir(models_dir):
            if name.lower() in f.lower():
                return os.path.join(models_dir, f)
                
        return None

    def send_message(self, prompt: str, history: list[Message] = None, 
                     system_prompt: str = None) -> str:
        """Send a message and get a complete response."""
        if self.init_error:
            raise RuntimeError(f"Engine failed: {self.init_error}")
        if not self.engine:
            raise RuntimeError("VoxAI Engine is not loaded.")
            
        self._sync_history(history, system_prompt)
        return self.engine.chat(prompt, stream=False)

    def stream_message(self, prompt: str, history: list[Message] = None, 
                       system_prompt: str = None) -> Generator[str, None, None]:
        """Stream a message response token by token."""
        if self.init_error:
            raise RuntimeError(f"Engine failed: {self.init_error}")
        if not self.engine:
            raise RuntimeError("VoxAI Engine is not loaded.")
            
        self._sync_history(history, system_prompt)
        
        # Track performance
        start_time = time.perf_counter()
        token_count = 0
        
        for chunk in self.engine.chat(prompt, stream=True):
            token_count += 1
            yield chunk
            
        # Log performance stats
        duration = time.perf_counter() - start_time
        if duration > 0:
            speed = token_count / duration
            mode_info = "(CPU fallback)" if self.engine.using_fallback else ""
            print(f"[VoxAI] {speed:.2f} t/s ({token_count} tokens, {duration:.2f}s) {mode_info}")

    def get_status(self) -> ProviderStatus:
        """Get the current provider status."""
        return self.status

    def mark_priority(self, message_index: int = -2):
        """Mark a message as priority ('remember this')."""
        if self.engine:
            self.engine.mark_priority(message_index)

    def set_pending_priority(self):
        """Flag the next turn's user message as priority."""
        if self.engine:
            self.engine.set_pending_priority()

    def set_session(self, session_id: str):
        """Switch to a different conversation session."""
        if self.engine:
            self.engine.set_session(session_id)

    def _sync_history(self, history: list[Message], system_prompt: str):
        """Sync conversation history to the engine.

        When Elastic Memory is active, the ContextManager handles history
        assembly via RAG retrieval, so we only need a minimal sync here.
        """
        # If elastic memory is active, skip full history rebuild —
        # ContextManager.prepare_context() handles it in engine.chat()
        if hasattr(self.engine, '_elastic_enabled') and self.engine._elastic_enabled:
            # Just ensure history is clean for the next chat() call
            self.engine.history = []
            return

        # Classic mode: full history rebuild
        self.engine.clear_history()

        sys_prompt = system_prompt or "You are a helpful assistant."
        self.engine.history.append({"role": "system", "content": sys_prompt})

        if history:
            for msg in history:
                self.engine.history.append({
                    "role": msg.role,
                    "content": msg.content
                })
