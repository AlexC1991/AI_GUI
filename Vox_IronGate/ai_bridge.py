"""
AI Bridge - Connects IronGate to the local VoxAI Chat Engine

Uses VoxAPI directly (in-process via llama_cpp) instead of HTTP.
Lazy-loads a singleton VoxAPI instance on first use.
"""

import sys
import os
import io
import time

# --- Null Writer for suppressing llama_cpp / VoxAPI init output ---
class _NullWriter(io.TextIOBase):
    """Silently discards all output. Avoids charmap codec errors on Windows."""
    def write(self, s): return len(s)
    def flush(self): pass

# --- Path Setup ---
# IronGate lives in AI_GUI/Vox_IronGate/
# VoxAI_Chat_API lives in AI_GUI/VoxAI_Chat_API/
_BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
_AI_GUI_ROOT = os.path.dirname(_BRIDGE_DIR)
_VOX_API_PATH = os.path.join(_AI_GUI_ROOT, "VoxAI_Chat_API")

if _VOX_API_PATH not in sys.path:
    sys.path.insert(0, _VOX_API_PATH)

# --- Lazy Singleton ---
_vox_engine = None
_vox_available = None
_current_model = None


def _get_default_model():
    """Find the first .gguf model in VoxAI_Chat_API/models/"""
    models_dir = os.path.join(_VOX_API_PATH, "models")
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

        # Suppress [HANDSHAKE], llama_context warnings, etc. during init
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


def set_model(model_filename):
    """Switch the active LLM model by filename (e.g. 'Llama-3.2-3B-Instruct-Q4_K_M.gguf')."""
    global _vox_engine, _vox_available, _current_model

    models_dir = os.path.join(_VOX_API_PATH, "models")
    model_path = os.path.join(models_dir, model_filename)

    if not os.path.exists(model_path):
        return False

    # If same model already loaded, skip
    if _current_model == model_filename and _vox_engine is not None:
        return True

    try:
        from vox_api import VoxAPI

        # Shutdown old engine if exists
        if _vox_engine is not None:
            try:
                _vox_engine.shutdown()
            except Exception:
                pass

        # Suppress [HANDSHAKE], llama_context warnings, etc. during init
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
        return "[System] VoxAI Engine not available. No .gguf models found in VoxAI_Chat_API/models/"

    try:
        # Elastic Memory Session Support
        if session_id and hasattr(engine, 'set_session'):
            engine.set_session(session_id)
            
        response = engine.chat(user_message, stream=False)
        return response.strip() if response else "[System] Empty response from AI."
    except Exception as e:
        return f"[System] AI Error: {e}"


def stream_the_brain(user_message, max_tokens=500, temperature=0.7, session_id=None):
    """Stream response from VoxAI, yielding chunks with performance stats.

    Yields dicts: {"type": "chunk", "text": "..."} for content,
                  {"type": "done", "stats": {"tokens": N, "duration": F, "speed": F}}
    """
    engine = _get_engine()
    if engine is None:
        yield {"type": "error", "text": "[System] VoxAI Engine not available. No .gguf models found."}
        return

    try:
        # Elastic Memory Session Support
        if session_id and hasattr(engine, 'set_session'):
            engine.set_session(session_id)

        token_count = 0
        start_time = time.perf_counter()

        for chunk in engine.chat(user_message, stream=True):
            if chunk:
                token_count += 1
                yield {"type": "chunk", "text": chunk}

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


if __name__ == "__main__":
    print(f"Testing AI Bridge (VoxAPI Direct)")
    print(f"VoxAPI Path: {_VOX_API_PATH}")
    print(f"Default Model: {_get_default_model()}")
    print(f"Response: {ask_the_brain('Hello')}")
