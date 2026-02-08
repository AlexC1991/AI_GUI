from PySide6.QtCore import QThread, Signal
import traceback
import time
from providers.gemini_provider import GeminiProvider
from providers.vox_provider import VoxProvider
from providers.base_provider import Message

class ChatWorker(QThread):
    chunk_received = Signal(str)
    chat_finished = Signal()
    speed_update = Signal(float, int, float)  # speed_tps, token_count, duration_secs
    error = Signal(str)

    # CLASS-LEVEL (SINGLETON) provider storage
    _shared_vox_provider = None
    _shared_model_name = None

    def __init__(self):
        super().__init__()
        self.provider_type = None
        self.model_name = None
        self.api_key = None
        self.prompt = None
        self.history = []
        self.system_prompt = None
        self._provider = None
        self._abort = False  # Flag for clean interruption
    
    def stop(self):
        """Request clean stop of generation."""
        self._abort = True

    def setup(self, provider_type, model_name, api_key, prompt, history, system_prompt=None, session_id=None):
        self._abort = False  # Reset abort flag
        self.provider_type = provider_type
        self.model_name = model_name
        self.api_key = api_key
        self.prompt = prompt
        self.history = history
        self.session_id = session_id
        
        # Enforce Code Protocol
        protocol = (
            " When providing code or scripts, follow these rules strictly:"
            " 1. When writing a NEW script or file, put the COMPLETE code in ONE single code block"
            " 2. State the filename on the line before the code block (e.g. '**app.py**')"
            " 3. NEVER split one script across multiple code blocks - put it ALL in one block"
            " 4. When EXPLAINING existing code, do NOT use a filename header - just use code blocks without naming them"
            " 5. Each unique file must have a different filename"
        )

        # Append protocol to system prompt
        if system_prompt:
            self.system_prompt = system_prompt + protocol
        else:
            self.system_prompt = "You are a helpful assistant." + protocol

    def run(self):
        print(f"[ChatWorker] Starting... Provider: {self.provider_type}, Model: {self.model_name}")
        try:
            # Initialize Provider - use SINGLETON for VoxAI to prevent VRAM exhaustion
            if self.provider_type == "Gemini":
                self._provider = GeminiProvider(model=self.model_name, api_key=self.api_key)
            elif self.provider_type == "VoxAI" or self.provider_type == "Ollama":
                # Reuse existing VoxProvider if same model
                if (ChatWorker._shared_vox_provider is not None and 
                    ChatWorker._shared_model_name == self.model_name):
                    print(f"[ChatWorker] Reusing existing VoxProvider for {self.model_name}")
                    self._provider = ChatWorker._shared_vox_provider
                else:
                    # Different model or first load - create new provider
                    print(f"[ChatWorker] Creating new VoxProvider for {self.model_name}")
                    ChatWorker._shared_vox_provider = VoxProvider(model_name=self.model_name)
                    ChatWorker._shared_model_name = self.model_name
                    self._provider = ChatWorker._shared_vox_provider
                
                # Pass "remember this" priority flag if set
                if getattr(self, '_pending_priority', False):
                    self._provider.set_pending_priority()
                
                # Elastic Memory: Set Session ID
                if self.session_id and hasattr(self._provider, 'set_session'):
                    self._provider.set_session(self.session_id)
            else:
                self.error.emit(f"Unknown provider: {self.provider_type}")
                return

            # Convert history dicts to Message objects if needed
            msg_history = []
            for h in self.history:
                if isinstance(h, dict):
                    msg_history.append(Message(role=h['role'], content=h['content']))
                else:
                    msg_history.append(h)
            
            token_count = 0
            start_time = time.perf_counter()

            # Stream generation
            for chunk in self._provider.stream_message(
                prompt=self.prompt,
                history=msg_history,
                system_prompt=self.system_prompt
            ):
                # Check for abort request (from Search Trigger)
                if self._abort:
                    print("[ChatWorker] Generation aborted by agent")
                    break

                token_count += 1
                self.chunk_received.emit(chunk)

            duration = time.perf_counter() - start_time
            speed = token_count / duration if duration > 0 else 0
            print(f"[ChatWorker] Finished. Speed: {speed:.2f} t/s")
            
            self.speed_update.emit(speed, token_count, duration)
            self.chat_finished.emit()

        except Exception as e:
            self.error.emit(str(e))