"""
CloudStreamer — Dedicated RunPod vLLM HTTP Streamer

Handles cloud GPU chat streaming WITHOUT touching local GGUF engines.
Uses RunPodDriver for pod management and raw HTTP for chat completions.
"""

import sys
import os
import json
import requests
from typing import Generator
from providers.base_provider import BaseProvider, ProviderStatus, Message

# Path to engine for RunPodDriver imports
_VOX_API_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "engine")
)
if _VOX_API_PATH not in sys.path:
    sys.path.insert(0, _VOX_API_PATH)


class CloudStreamer(BaseProvider):
    """Provider for RunPod cloud GPU streaming. No local model loading."""

    def __init__(self):
        self.cloud_driver = None
        self.cloud_model_id = None
        self.cloud_key = None
        self.cloud_pod_id = None
        self.cloud_port = 8000
        self.status = ProviderStatus(
            available=False,
            message="Cloud not configured",
            model=None
        )

    def configure(self, api_key: str, pod_id: str = None, port: int = 8000,
                  gpu_tier: str = "Auto", model_id: str = None):
        """Configure cloud connection. Call before streaming."""
        self.cloud_key = api_key
        self.cloud_pod_id = pod_id
        self.cloud_port = port
        self.cloud_model_id = model_id

        if not api_key:
            self.status = ProviderStatus(False, "No RunPod API key", None)
            return

        try:
            from runpod_interface import RunPodDriver
            from config import POD_ID as CFG_POD, API_KEY as CFG_KEY

            key = api_key or CFG_KEY
            pod = pod_id or CFG_POD

            if not self.cloud_driver:
                self.cloud_driver = RunPodDriver(key, pod, port)
                print(f"[CloudStreamer] Driver initialized (Pod: {pod}, Port: {port})")
            else:
                # Re-authenticate CLI if key changed
                if self.cloud_driver.api_key != key:
                    self.cloud_driver._configure_cli(key)
                self.cloud_driver.api_key = key
                self.cloud_driver.pod_id = pod
                self.cloud_driver.port = port

            if hasattr(self.cloud_driver, 'set_gpu_tier'):
                self.cloud_driver.set_gpu_tier(gpu_tier)

            self.status = ProviderStatus(
                available=True,
                message=f"Cloud ready: {model_id or 'No model set'}",
                model=model_id
            )

        except ImportError as e:
            print(f"[CloudStreamer] Import failed: {e}")
            self.status = ProviderStatus(False, f"Cloud import error: {e}", None)

    def send_message(self, prompt: str, history: list[Message] = None,
                     system_prompt: str = None) -> str:
        """Non-streaming cloud message (collects full response)."""
        result = ""
        for chunk in self.stream_message(prompt, history, system_prompt):
            result += chunk
        return result

    def stream_message(self, prompt: str, history: list[Message] = None,
                       system_prompt: str = None) -> Generator[str, None, None]:
        """Stream response from RunPod vLLM endpoint."""
        if not self.cloud_model_id:
            yield "Error: No Cloud Model ID set. Select a cloud model from the Model Selector."
            return

        if not self.cloud_driver:
            yield "Error: Cloud driver not initialized. Check RunPod API key in Settings."
            return

        # Auto-boot pod if not active
        if not self.cloud_driver.new_pod_id:
            # Emit boot progress as special [BOOT:...] tokens that ChatAgent can intercept
            import time as _time
            boot_start = _time.perf_counter()

            # Display name from config
            from config import MODEL_MAP
            display_name = MODEL_MAP.get(self.cloud_model_id, self.cloud_model_id.split("/")[-1])
            yield f"[BOOT_START:{display_name}]"

            # Start boot in a thread so we can emit progress ticks
            import threading
            boot_result = {"success": False, "done": False}

            def _do_boot():
                boot_result["success"] = self.cloud_driver.switch_model(self.cloud_model_id)
                boot_result["done"] = True

            boot_thread = threading.Thread(target=_do_boot, daemon=True)
            boot_thread.start()

            # Emit progress ticks every 3s while booting
            while not boot_result["done"]:
                _time.sleep(3)
                elapsed = int(_time.perf_counter() - boot_start)
                yield f"[BOOT_TICK:{elapsed}]"

            if boot_result["success"]:
                total = int(_time.perf_counter() - boot_start)
                yield f"[BOOT_DONE:{total}]"
            else:
                yield "[BOOT_FAIL]"
                return

        # Build messages
        messages = [{"role": "system", "content": system_prompt or "You are a helpful assistant."}]
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": prompt})

        # Stream from vLLM endpoint
        url = f"https://{self.cloud_driver.new_pod_id}-{self.cloud_port}.proxy.runpod.net/v1/chat/completions"
        payload = {
            "model": self.cloud_model_id,
            "messages": messages,
            "max_tokens": 2048,
            "stream": True
        }
        headers = {"Authorization": f"Bearer {self.cloud_key}"}

        try:
            response = requests.post(url, json=payload, headers=headers, stream=True, timeout=120)
            for chunk in response.iter_lines():
                if chunk:
                    decoded = chunk.decode('utf-8')
                    if decoded.startswith("data: "):
                        decoded = decoded[6:]
                    decoded = decoded.strip()
                    if decoded == "[DONE]":
                        break
                    try:
                        j = json.loads(decoded)
                        if 'choices' in j and len(j['choices']) > 0:
                            delta = j['choices'][0].get('delta', {})
                            if 'content' in delta:
                                yield delta['content']
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            yield f"\n[Cloud Error] {e}"

    def get_status(self) -> ProviderStatus:
        return self.status

    def get_balance(self):
        """Get RunPod account balance."""
        if self.cloud_driver:
            return self.cloud_driver.get_balance()
        return None

    def get_cost(self):
        """Get current pod hourly cost."""
        if self.cloud_driver:
            return self.cloud_driver.pod_cost
        return None

    def terminate_pod(self):
        """Kill switch — terminate active pod."""
        if self.cloud_driver:
            self.cloud_driver.terminate_pod()
