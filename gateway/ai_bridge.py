"""
AI Bridge - Connects IronGate to the local VoxAI Chat Engine + Cloud GPU

Two execution paths:
  - LOCAL: VoxAPI (in-process llama_cpp) — for .gguf models
  - CLOUD: CloudStreamer (RunPod vLLM HTTP) — for HuggingFace models

Lazy-loads singletons on first use.
"""

import sys
import os
import io
import re
import time

# --- Null Writer for suppressing llama_cpp / VoxAPI init output ---
class _NullWriter(io.TextIOBase):
    """Silently discards all output. Avoids charmap codec errors on Windows."""
    def write(self, s): return len(s)
    def flush(self): pass

# --- Path Setup ---
_BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
_AI_GUI_ROOT = os.path.dirname(_BRIDGE_DIR)
_VOX_API_PATH = os.path.join(_AI_GUI_ROOT, "engine")

if _VOX_API_PATH not in sys.path:
    sys.path.insert(0, _VOX_API_PATH)
if _AI_GUI_ROOT not in sys.path:
    sys.path.insert(0, _AI_GUI_ROOT)

# --- Lazy Singletons ---
_vox_engine = None
_vox_available = None
_current_model = None
_cloud_streamer = None


def _get_default_model():
    """Find the first .gguf model in models/llm/"""
    models_dir = os.path.join(_AI_GUI_ROOT, "models", "llm")
    if not os.path.exists(models_dir):
        return None
    for f in sorted(os.listdir(models_dir)):
        if f.endswith(".gguf"):
            return os.path.join(models_dir, f)
    return None


def _get_engine():
    """Lazy-load VoxAPI singleton. Returns the engine or None."""
    global _vox_engine, _vox_available, _current_model

    if _vox_available is False:
        return None

    if _vox_engine is not None:
        return _vox_engine

    try:
        from vox_api import VoxAPI

        model_path = _get_default_model()
        if not model_path:
            _vox_available = False
            return None

        _old_out, _old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _NullWriter(), _NullWriter()
        try:
            _vox_engine = VoxAPI(
                model_path=model_path,
                verbose=False,
                max_retries=2,
                retry_delay=1.5
            )
        finally:
            sys.stdout, sys.stderr = _old_out, _old_err

        _current_model = os.path.basename(model_path)
        _vox_available = True
        return _vox_engine

    except ImportError:
        _vox_available = False
        return None
    except Exception:
        _vox_available = False
        return None


def _get_cloud_streamer():
    """Lazy-load CloudStreamer singleton."""
    global _cloud_streamer
    if _cloud_streamer is not None:
        return _cloud_streamer
    try:
        from providers.cloud_streamer import CloudStreamer
        _cloud_streamer = CloudStreamer()
        return _cloud_streamer
    except ImportError as e:
        print(f"[AI Bridge] CloudStreamer import failed: {e}")
        return None


def get_cloud_models():
    """Return cloud model list from config."""
    try:
        from utils.config_manager import ConfigManager
        config = ConfigManager.load_config()
        cloud = config.get("cloud", {})
        models = cloud.get("models", {})
        return [{"id": hf_id, "display": display} for hf_id, display in models.items()]
    except Exception:
        return []


def set_model(model_filename):
    """Switch the active LLM model by filename (e.g. 'Llama-3.2-3B-Instruct-Q4_K_M.gguf')."""
    global _vox_engine, _vox_available, _current_model

    models_dir = os.path.join(_AI_GUI_ROOT, "models", "llm")
    model_path = os.path.join(models_dir, model_filename)

    if not os.path.exists(model_path):
        return False

    if _current_model == model_filename and _vox_engine is not None:
        return True

    try:
        from vox_api import VoxAPI

        if _vox_engine is not None:
            try:
                _vox_engine.shutdown()
            except Exception:
                pass

        _old_out, _old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _NullWriter(), _NullWriter()
        try:
            _vox_engine = VoxAPI(
                model_path=model_path,
                verbose=False,
                max_retries=2,
                retry_delay=1.5
            )
        finally:
            sys.stdout, sys.stderr = _old_out, _old_err

        _current_model = model_filename
        _vox_available = True
        return True

    except Exception:
        _vox_available = False
        _vox_engine = None
        return False


def ask_the_brain(user_message, max_tokens=500, temperature=0.7, session_id=None):
    """Send message to VoxAI and return complete response (non-streaming)."""
    engine = _get_engine()
    if engine is None:
        return "[System] VoxAI Engine not available. No .gguf models found in models/llm/"

    try:
        if session_id and hasattr(engine, 'set_session'):
            engine.set_session(session_id)

        response = engine.chat(user_message, stream=False)
        return response.strip() if response else "[System] Empty response from AI."
    except Exception as e:
        return f"[System] AI Error: {e}"


def _parse_thinking(chunk, state):
    """Parse <think>...</think> tokens from a stream chunk.

    Returns list of events: [{"type": "think_start"}, {"type": "think_chunk", "text": "..."},
                              {"type": "think_end"}, {"type": "chunk", "text": "..."}]
    """
    events = []
    text = state.get("buffer", "") + chunk
    state["buffer"] = ""

    while text:
        if state.get("in_think"):
            # Looking for </think>
            end_idx = text.find("</think>")
            if end_idx != -1:
                # Emit think content before the tag
                think_text = text[:end_idx]
                if think_text:
                    events.append({"type": "think_chunk", "text": think_text})
                events.append({"type": "think_end"})
                state["in_think"] = False
                text = text[end_idx + 8:]  # len("</think>") = 8
            else:
                # Might be partial tag at end
                if "<" in text and text.endswith(("<", "</", "</t", "</th", "</thi", "</thin", "</think")):
                    state["buffer"] = text
                    break
                events.append({"type": "think_chunk", "text": text})
                text = ""
        else:
            # Looking for <think>
            start_idx = text.find("<think>")
            if start_idx != -1:
                # Emit normal content before the tag
                before = text[:start_idx]
                if before.strip():
                    events.append({"type": "chunk", "text": before})
                events.append({"type": "think_start"})
                state["in_think"] = True
                text = text[start_idx + 7:]  # len("<think>") = 7
            else:
                # Might be partial tag at end
                if "<" in text and text.endswith(("<", "<t", "<th", "<thi", "<thin", "<think")):
                    state["buffer"] = text
                    break
                if text.strip():
                    events.append({"type": "chunk", "text": text})
                text = ""

    return events


# --- Special token stripping (same list as desktop ChatAgent) ---
_STOP_TOKENS = [
    '<|im_end|>', '|im_end|>', '<|im_start|>', '<|endoftext|>',
    '<|eot_id|>', '<|end_of_text|>', '<|start_header_id|>', '<|end_header_id|>',
    '<|end▁of▁sentence|>', '<|begin▁of▁sentence|>',
    '<|end|>', '<|user|>', '<|assistant|>', '<|system|>', '<|channel|>',
    '<|start|>', '<|message|>', '<|call|>', '<|return|>',
    '<|constrain|>', '<|endofprompt|>', '<|startoftext|>', '<|padding|>',
]
_SPECIAL_RE = re.compile(r'<\|[^|]*\|>')


def _strip_tokens(text):
    """Remove special tokens from text."""
    for token in _STOP_TOKENS:
        text = text.replace(token, '')
    text = _SPECIAL_RE.sub('', text)
    return text


def stream_the_brain(user_message, max_tokens=500, temperature=0.7, session_id=None,
                     history=None, system_prompt=None):
    """Stream response from VoxAI, yielding chunks with performance stats and thinking support.

    Yields dicts:
        {"type": "chunk", "text": "..."}        — visible content
        {"type": "think_start"}                  — thinking section begins
        {"type": "think_chunk", "text": "..."}   — thinking content
        {"type": "think_end"}                    — thinking section ends
        {"type": "done", "stats": {...}}         — generation complete
        {"type": "error", "text": "..."}         — error
    """
    engine = _get_engine()
    if engine is None:
        yield {"type": "error", "text": "[System] VoxAI Engine not available. No .gguf models found."}
        return

    try:
        if session_id and hasattr(engine, 'set_session'):
            engine.set_session(session_id)

        # Inject history if provided
        if history:
            engine.clear_history()
            engine.history.append({"role": "system", "content": system_prompt or "You are a helpful assistant."})
            for msg in history:
                engine.history.append({"role": msg["role"], "content": msg["content"]})

        token_count = 0
        start_time = time.perf_counter()
        think_state = {"in_think": False, "buffer": ""}

        for chunk in engine.chat(user_message, stream=True):
            if chunk:
                token_count += 1
                cleaned = _strip_tokens(chunk)
                if not cleaned:
                    continue

                # Parse thinking sections
                events = _parse_thinking(cleaned, think_state)
                for event in events:
                    yield event

        # Flush any remaining buffer
        if think_state["buffer"]:
            remaining = think_state["buffer"]
            if think_state["in_think"]:
                yield {"type": "think_chunk", "text": remaining}
                yield {"type": "think_end"}
            elif remaining.strip():
                yield {"type": "chunk", "text": remaining}

        duration = time.perf_counter() - start_time
        speed = token_count / duration if duration > 0 else 0

        yield {
            "type": "done",
            "stats": {
                "tokens": token_count,
                "duration": round(duration, 2),
                "speed": round(speed, 2)
            }
        }

    except Exception as e:
        yield {"type": "error", "text": f"[System] AI Error: {e}"}


def stream_cloud(user_message, model_id, history=None, system_prompt=None):
    """Stream response from RunPod cloud GPU via CloudStreamer.

    Yields same event format as stream_the_brain for UI consistency.
    """
    streamer = _get_cloud_streamer()
    if streamer is None:
        yield {"type": "error", "text": "[System] Cloud streaming not available."}
        return

    try:
        from utils.config_manager import ConfigManager
        config = ConfigManager.load_config()
        cloud_cfg = config.get("cloud", {})

        streamer.configure(
            api_key=cloud_cfg.get("runpod_api_key", ""),
            pod_id=cloud_cfg.get("pod_id", ""),
            port=8000,
            gpu_tier="Auto",
            model_id=model_id
        )

        # Convert history to Message objects
        from providers.base_provider import Message
        msg_history = None
        if history:
            msg_history = [Message(role=m["role"], content=m["content"]) for m in history]

        token_count = 0
        start_time = time.perf_counter()
        think_state = {"in_think": False, "buffer": ""}

        for chunk in streamer.stream_message(
            prompt=user_message,
            history=msg_history,
            system_prompt=system_prompt or "You are a helpful assistant."
        ):
            if not chunk:
                continue

            # Handle boot progress tokens
            boot_start = re.match(r'^\[BOOT_START:(.+?)\]$', chunk.strip())
            if boot_start:
                # Determine estimated boot time based on model size
                # 70B+ models ~512s, smaller models ~235s
                model_lower = (model_id or "").lower()
                is_large = any(kw in model_lower for kw in ["70b", "72b", "120b", "grok", "miqu"])
                eta = 512 if is_large else 235
                yield {"type": "boot_start", "text": boot_start.group(1), "eta": eta}
                continue

            boot_tick = re.match(r'^\[BOOT_TICK:(\d+)\]$', chunk.strip())
            if boot_tick:
                yield {"type": "boot_tick", "elapsed": int(boot_tick.group(1))}
                continue

            boot_done = re.match(r'^\[BOOT_DONE:(\d+)\]$', chunk.strip())
            if boot_done:
                yield {"type": "boot_done", "elapsed": int(boot_done.group(1))}
                continue

            if chunk.strip() == "[BOOT_FAIL]":
                yield {"type": "boot_fail"}
                return

            token_count += 1
            cleaned = _strip_tokens(chunk)
            if not cleaned:
                continue

            # Parse thinking sections (cloud models can think too)
            events = _parse_thinking(cleaned, think_state)
            for event in events:
                yield event

        # Flush remaining
        if think_state["buffer"]:
            remaining = think_state["buffer"]
            if think_state["in_think"]:
                yield {"type": "think_chunk", "text": remaining}
                yield {"type": "think_end"}
            elif remaining.strip():
                yield {"type": "chunk", "text": remaining}

        duration = time.perf_counter() - start_time
        speed = token_count / duration if duration > 0 else 0

        yield {
            "type": "done",
            "stats": {
                "tokens": token_count,
                "duration": round(duration, 2),
                "speed": round(speed, 2)
            }
        }

    except Exception as e:
        yield {"type": "error", "text": f"[Cloud Error] {e}"}


def terminate_cloud_pod():
    """Terminate any active RunPod cloud pod. Safe to call even if none active."""
    global _cloud_streamer
    if _cloud_streamer is not None:
        try:
            _cloud_streamer.terminate_pod()
            print("[AI Bridge] Cloud pod terminated.")
        except Exception as e:
            print(f"[AI Bridge] Pod termination error: {e}")


if __name__ == "__main__":
    print(f"Testing AI Bridge (VoxAPI Direct)")
    print(f"VoxAPI Path: {_VOX_API_PATH}")
    print(f"Default Model: {_get_default_model()}")
    print(f"Cloud Models: {get_cloud_models()}")
    print(f"Response: {ask_the_brain('Hello')}")
