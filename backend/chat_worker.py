from PySide6.QtCore import QThread, Signal
import traceback
import time
from providers.gemini_provider import GeminiProvider
from providers.vox_provider import VoxProvider
from providers.cloud_streamer import CloudStreamer
from providers.base_provider import Message
from providers import PROVIDER_REGISTRY

class ChatWorker(QThread):
    chunk_received = Signal(str)
    chat_finished = Signal()
    speed_update = Signal(float, int, float)  # speed_tps, token_count, duration_secs
    error = Signal(str)

    # CLASS-LEVEL (SINGLETON) provider storage
    _shared_vox_provider = None
    _shared_model_name = None
    _shared_cloud_streamer = None

    def set_execution_mode(self, mode, model_data):
        """Set execution mode from model selection.
        mode: 'local', 'cloud', or 'provider'
        model_data: dict with model info from the selector panel
        """
        if mode == "cloud":
            from utils.config_manager import ConfigManager
            config = ConfigManager.load_config()
            cloud_cfg = config.get("cloud", {})
            self.cloud_enabled = True
            self.cloud_api_key = cloud_cfg.get("runpod_api_key", "")
            self.cloud_gpu_tier = "Auto"
            self.cloud_pod_id = cloud_cfg.get("pod_id", "")
            self.cloud_port = 8000
            self.cloud_model_id = model_data.get("cloud_id", "") if isinstance(model_data, dict) else ""
            print(f"[ChatWorker] Execution mode: CLOUD | Pod: {self.cloud_pod_id} | Model: {self.cloud_model_id}")
        else:
            self.cloud_enabled = False
            self.cloud_model_id = None
            print(f"[ChatWorker] Execution mode: {mode.upper()}")

    def set_cloud_config(self, enabled, api_key, gpu_tier="Auto", pod_id=None, port=8000):
        """Legacy method — kept for backward compatibility."""
        self.cloud_enabled = enabled
        self.cloud_api_key = api_key
        self.cloud_gpu_tier = gpu_tier
        self.cloud_pod_id = pod_id
        self.cloud_port = port

    def __init__(self):
        super().__init__()
        self.provider_type = None
        self.model_name = None
        self.api_key = None
        self.prompt = None
        self.history = []
        self.system_prompt = None
        self._provider = None
        self._abort = False

        # Cloud Configuration Persistence
        self.cloud_enabled = False
        self.cloud_api_key = None
        self.cloud_gpu_tier = "Auto"
        self.cloud_pod_id = None
        self.cloud_port = 8000
        self.cloud_model_id = None

    def stop(self):
        """Request clean stop of generation."""
        self._abort = True

    def setup(self, provider_type, model_name, api_key, prompt, history, system_prompt=None, session_id=None):
        self._abort = False
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

        if system_prompt:
            self.system_prompt = system_prompt + protocol
        else:
            self.system_prompt = "You are a helpful assistant." + protocol

    def run(self):
        print(f"[ChatWorker] Starting... Provider: {self.provider_type}, Model: {self.model_name}")
        try:
            if self.provider_type == "Gemini":
                self._provider = GeminiProvider(model=self.model_name, api_key=self.api_key)

            elif self.provider_type == "OpenAI":
                from providers.openai_provider import OpenAIProvider
                self._provider = OpenAIProvider(model=self.model_name, api_key=self.api_key)

            elif self.provider_type == "VoxAI" or self.provider_type == "Ollama":
                # ===== CLOUD MODE: Use CloudStreamer (NO local engine) =====
                if self.cloud_enabled:
                    if ChatWorker._shared_cloud_streamer is None:
                        ChatWorker._shared_cloud_streamer = CloudStreamer()
                    streamer = ChatWorker._shared_cloud_streamer
                    streamer.configure(
                        api_key=self.cloud_api_key,
                        pod_id=self.cloud_pod_id,
                        port=self.cloud_port,
                        gpu_tier=self.cloud_gpu_tier,
                        model_id=self.cloud_model_id
                    )
                    self._provider = streamer
                    print(f"[ChatWorker] Cloud mode — using CloudStreamer (no local engine)")

                # ===== LOCAL MODE: Use VoxProvider with GGUF engine =====
                else:
                    if (ChatWorker._shared_vox_provider is not None and
                        ChatWorker._shared_model_name == self.model_name):
                        print(f"[ChatWorker] Reusing existing VoxProvider for {self.model_name}")
                        self._provider = ChatWorker._shared_vox_provider
                    else:
                        # Release old provider's VRAM first
                        if ChatWorker._shared_vox_provider is not None:
                            old = ChatWorker._shared_vox_provider
                            if hasattr(old, 'engine') and old.engine:
                                print(f"[ChatWorker] Releasing old engine VRAM...")
                                try:
                                    del old.engine
                                    old.engine = None
                                    import gc
                                    gc.collect()
                                except: pass
                        print(f"[ChatWorker] Creating new VoxProvider for {self.model_name}")
                        ChatWorker._shared_vox_provider = VoxProvider(model_name=self.model_name)
                        ChatWorker._shared_model_name = self.model_name
                        self._provider = ChatWorker._shared_vox_provider

                    # Ensure cloud is disabled for local
                    if hasattr(self._provider, 'set_cloud_mode'):
                        self._provider.set_cloud_mode(False)

                # Pass "remember this" priority flag if set
                if getattr(self, '_pending_priority', False) and hasattr(self._provider, 'set_pending_priority'):
                    self._provider.set_pending_priority()

                # Elastic Memory: Set Session ID
                if self.session_id and hasattr(self._provider, 'set_session'):
                    self._provider.set_session(self.session_id)
            else:
                # --- Dynamic provider resolution via registry ---
                resolved = False
                for cfg_key, (cls, display_name) in PROVIDER_REGISTRY.items():
                    if self.provider_type in (display_name, cfg_key, cls.__name__):
                        self._provider = cls(model=self.model_name, api_key=self.api_key)
                        print(f"[ChatWorker] Using {display_name} provider for {self.model_name}")
                        resolved = True
                        break
                if not resolved:
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
            # Add provider/model context to error messages for clarity
            err_str = str(e)
            provider = self.provider_type or "Unknown"
            model = self.model_name or "Unknown"
            if "not a valid model" in err_str.lower() or "model_not_found" in err_str.lower() or "does not exist" in err_str.lower():
                self.error.emit(f"[{provider}] Model '{model}' is not available for chat. Re-fetch models in Settings to update the list.")
            else:
                self.error.emit(f"[{provider}] {err_str}")
