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
from providers.gemini_provider import GeminiProvider
from providers.vox_provider import VoxProvider
from providers.base_provider import Message

class ChatWorker(QThread):
    chunk_received = Signal(str)
    chat_finished = Signal()
    speed_update = Signal(float, int, float)  # speed_tps, token_count, duration_secs
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
        print(f"[DEBUG] ChatWorker started for provider: {self.provider_type}, model: {self.model_name}")
        try:
            # Initialize Provider
            if self.provider_type == "Gemini":
                print("[DEBUG] Initializing Gemini Provider")
                self._provider = GeminiProvider(model=self.model_name, api_key=self.api_key)
            elif self.provider_type == "VoxAI" or self.provider_type == "Ollama": # Backwards compat
                print(f"[DEBUG] Initializing VoxAI Provider with model: {self.model_name}")
                self._provider = VoxProvider(model_name=self.model_name)
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
            token_count = 0
            start_time = time.perf_counter()

            for chunk in self._provider.stream_message(
                prompt=self.prompt,
                history=msg_history,
                system_prompt=self.system_prompt
            ):
                token_count += 1
                self.chunk_received.emit(chunk)

            duration = time.perf_counter() - start_time
            speed = token_count / duration if duration > 0 else 0
            print(f"[ChatWorker] {speed:.2f} t/s ({token_count} tokens, {duration:.2f}s)")
            self.speed_update.emit(speed, token_count, duration)
            self.chat_finished.emit()

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))


