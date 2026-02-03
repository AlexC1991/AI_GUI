from PySide6.QtCore import QThread, Signal
import traceback
import socket
import subprocess
import shutil
import time
import os
import sys
from utils.ollama_helper import get_ollama_path
from utils.config_manager import ConfigManager
from providers.gemini_provider import GeminiProvider
from providers.ollama_provider import OllamaProvider
from providers.base_provider import Message

class ChatWorker(QThread):
    chunk_received = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.provider_type = None
        self.model_name = None
        self.api_key = None
        self.prompt = None
        self.history = []
        self.system_prompt = None
        self._provider = None

    def setup(self, provider_type, model_name, api_key, prompt, history, system_prompt=None):
        self.provider_type = provider_type
        self.model_name = model_name
        self.api_key = api_key
        self.prompt = prompt
        self.history = history
        self.system_prompt = system_prompt

    def run(self):
        print(f"[DEBUG] ChatWorker started for provider: {self.provider_type}, model: {self.model_name}")
        try:
            # Initialize Provider
            if self.provider_type == "Gemini":
                print("[DEBUG] Initializing Gemini Provider")
                self._provider = GeminiProvider(model=self.model_name, api_key=self.api_key)
            elif self.provider_type == "Ollama":
                print("[DEBUG] Initializing Ollama Provider")
                # AUTO-START OLLAMA LOGIC
                if not self._is_ollama_running():
                    print("[DEBUG] Ollama not running. Attempting auto-start...")
                    self.chunk_received.emit("[System: Starting Ollama Server...]\n")
                    if not self._start_ollama():
                        print("[DEBUG] Failed to start Ollama")
                        self.error.emit("Failed to start Ollama. Is it installed?")
                        return
                    print("[DEBUG] Ollama started successfully")
                    self.chunk_received.emit("[System: Ollama Started. Generating...]\n")
                
                self._provider = OllamaProvider(model=self.model_name)
            else:
                self.error.emit(f"Unknown provider: {self.provider_type}")
                return

            print("[DEBUG] Provider initialized. Preparing history...")
            # Convert history dicts to Message objects if needed
            msg_history = []
            for h in self.history:
                if isinstance(h, dict):
                    msg_history.append(Message(role=h['role'], content=h['content']))
                else:
                    msg_history.append(h)
            
            print(f"[DEBUG] History prepared ({len(msg_history)} messages). Starting stream...")
            for chunk in self._provider.stream_message(
                prompt=self.prompt,
                history=msg_history,
                system_prompt=self.system_prompt
            ):
                self.chunk_received.emit(chunk)
            
            self.finished.emit()

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))

    def _is_ollama_running(self):
        try:
            with socket.create_connection(("localhost", 11434), timeout=1):
                return True
        except (socket.timeout, ConnectionRefusedError):
            return False

    def _start_ollama(self):
        ollama_exe = get_ollama_path()
        if not ollama_exe:
            return False
            
        try:
            # Prepare Environment
            config = ConfigManager.load_config()
            llm_cfg = config.get("llm", {})
            custom_dir = llm_cfg.get("local_model_dir", "")
            
            env = os.environ.copy()
            if custom_dir:
                # Ensure absolute path
                abs_path = os.path.abspath(custom_dir)
                if not os.path.exists(abs_path):
                    os.makedirs(abs_path, exist_ok=True)
                env["OLLAMA_MODELS"] = abs_path

            # Start detached process
            subprocess.Popen(
                [ollama_exe, "serve"], 
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
                env=env
            )
            
            # Wait for it to come up (max 10s)
            for _ in range(10):
                if self._is_ollama_running():
                    return True
                time.sleep(1)
                
            return False
        except Exception:
            return False
