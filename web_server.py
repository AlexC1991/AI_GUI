"""
VoxAI Web Server - Remote Access Interface

This creates a web-accessible version of VoxAI that you can access from anywhere.
Features:
- Password protected access
- Chat with LLM (streaming)
- Image generation with model selection
- Model hot-swapping via keyboard shortcuts
- Mobile-friendly responsive UI

Usage:
    python web_server.py --port 7860 --password yourpassword

Then access from anywhere: http://your-ip:7860
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import asyncio
import hashlib
import secrets
import argparse
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps

# Add parent directory to path for imports
APP_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(APP_DIR))
from backend.image_generator import GenerationConfig

# Add VoxAI_Chat_API to path
VOX_API_DIR = APP_DIR / "VoxAI_Chat_API"
if VOX_API_DIR.exists():
    sys.path.insert(0, str(VOX_API_DIR))

# =========================================================
# BACKEND SETUP (CRITICAL FOR LOCAL LLAMA.CPP)
# =========================================================
def _setup_vox_backend():
    """Set up custom backend environment variables and DLLs."""
    import ctypes
    print("[VoxAI Server] Setting up custom backend...")
    
    if not VOX_API_DIR.exists():
        return False

    vox_str = str(VOX_API_DIR)
    llama_dll = VOX_API_DIR / "llama.dll"
    ggml_dll = VOX_API_DIR / "ggml.dll"

    # 1. Set LLAMA_CPP_LIB to point to our custom DLL
    if llama_dll.exists():
        os.environ["LLAMA_CPP_LIB"] = str(llama_dll)
        print(f"[VoxAI Server] LLAMA_CPP_LIB = {llama_dll}")

    # 2. Set Backend Search Path
    os.environ["GGML_BACKEND_SEARCH_PATH"] = vox_str

    # 3. Add to PATH
    os.environ["PATH"] = vox_str + os.pathsep + os.environ.get("PATH", "")

    # 4. Windows DLL Directories
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(vox_str)
        except Exception as e:
            print(f"[VoxAI Server] DLL dir warning: {e}")

    # 5. Pre-load ggml.dll and Backends
    if ggml_dll.exists():
        try:
            ggml = ctypes.CDLL(str(ggml_dll))
            if hasattr(ggml, 'ggml_backend_load_all'):
                ggml.ggml_backend_load_all()
                print("[VoxAI Server] Backends loaded (ggml_backend_load_all)")
        except Exception as e:
            print(f"[VoxAI Server] Backend load error: {e}")
            return False
            
    return True

# EXECUTE SETUP IMMEDIATELY
_setup_vox_backend()

# Flask imports
try:
    from flask import Flask, request, jsonify, Response, render_template_string, session, redirect, url_for, send_file
    from flask_cors import CORS
except ImportError:
    print("[WebServer] Installing Flask...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "flask-cors", "--quiet"])
    from flask import Flask, request, jsonify, Response, render_template_string, session, redirect, url_for, send_file
    from flask_cors import CORS

# ============================================
# CONFIGURATION
# ============================================

class ServerConfig:
    HOST = "0.0.0.0"
    PORT = 7860
    PASSWORD = None  # Set via command line or config
    SECRET_KEY = secrets.token_hex(32)
    SESSION_TIMEOUT = 24 * 60 * 60  # 24 hours
    
    # Paths
    MODELS_DIR = APP_DIR / "VoxAI_Chat_API" / "models"
    OUTPUT_DIR = APP_DIR / "outputs" / "images"
    CHECKPOINTS_DIR = APP_DIR / "models" / "checkpoints"
    
    # Image generation presets
    IMAGE_PRESETS = {
        "0": {
            "name": "SDXL (Balanced)",
            "checkpoint": "sd_xl_base_1.0.safetensors",
            "width": 1024, "height": 1024,
            "steps": 25, "cfg": 7.0,
            "sampler": "euler_ancestral"
        },
        "1": {
            "name": "SDXL (Quality)", 
            "checkpoint": "sd_xl_base_1.0.safetensors",
            "width": 1024, "height": 1024,
            "steps": 40, "cfg": 7.5,
            "sampler": "dpmpp_2m"
        },
        "2": {
            "name": "Flux (Fast)",
            "checkpoint": "flux1-schnell.safetensors",
            "width": 1024, "height": 1024,
            "steps": 4, "cfg": 1.0,
            "sampler": "euler"
        },
        "3": {
            "name": "Flux (Dev)",
            "checkpoint": "flux1-dev.safetensors",
            "width": 1024, "height": 1024,
            "steps": 25, "cfg": 3.5,
            "sampler": "euler"
        }
    }

config = ServerConfig()

# ============================================
# FLASK APP
# ============================================

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
CORS(app)

# Global state
class AppState:
    vox_api = None
    image_generator = None
    current_llm_model = None
    current_image_preset = "0"
    is_generating = False
    chat_history = []

state = AppState()

# ============================================
# AUTHENTICATION
# ============================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if config.PASSWORD:
            if 'authenticated' not in session or not session['authenticated']:
                if request.is_json:
                    return jsonify({"error": "Not authenticated"}), 401
                return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password') or request.json.get('password')
        if password and hash_password(password) == hash_password(config.PASSWORD):
            session['authenticated'] = True
            session['login_time'] = datetime.now().isoformat()
            if request.is_json:
                return jsonify({"success": True})
            return redirect(url_for('index'))
        else:
            if request.is_json:
                return jsonify({"error": "Invalid password"}), 401
            return render_template_string(LOGIN_HTML, error="Invalid password")
    
    return render_template_string(LOGIN_HTML, error=None)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/')
@require_auth
def index():
    return render_template_string(MAIN_HTML)

@app.route('/api/status')
@require_auth
def api_status():
    """Get current server status."""
    return jsonify({
        "status": "online",
        "current_llm": state.current_llm_model,
        "current_image_preset": state.current_image_preset,
        "is_generating": state.is_generating,
        "available_llm_models": get_available_llm_models(),
        "image_presets": config.IMAGE_PRESETS
    })

@app.route('/api/models/llm')
@require_auth
def api_llm_models():
    """List available LLM models."""
    models = get_available_llm_models()
    return jsonify({"models": models, "current": state.current_llm_model})

@app.route('/api/models/llm/load', methods=['POST'])
@require_auth
def api_load_llm():
    """Load a specific LLM model."""
    data = request.json
    model_name = data.get('model')
    
    if not model_name:
        return jsonify({"error": "No model specified"}), 400
    
    try:
        success = load_llm_model(model_name)
        if success:
            return jsonify({"success": True, "model": model_name})
        else:
            return jsonify({"error": "Failed to load model"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
@require_auth
def api_chat():
    """Send a chat message and get streaming response."""
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    # Add to history
    state.chat_history.append({"role": "user", "content": message})
    
    def generate():
        try:
            response_text = ""
            for chunk in chat_stream(message):
                response_text += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # Add assistant response to history
            state.chat_history.append({"role": "assistant", "content": response_text})
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/chat/clear', methods=['POST'])
@require_auth
def api_clear_chat():
    """Clear chat history."""
    state.chat_history = []
    if state.vox_api:
        try:
            state.vox_api.reset_context()
        except:
            pass
    return jsonify({"success": True})

@app.route('/api/image/presets')
@require_auth
def api_image_presets():
    """Get available image generation presets."""
    return jsonify({
        "presets": config.IMAGE_PRESETS,
        "current": state.current_image_preset
    })

@app.route('/api/image/preset', methods=['POST'])
@require_auth
def api_set_image_preset():
    """Set image generation preset."""
    data = request.json
    preset_id = str(data.get('preset', '0'))
    
    if preset_id in config.IMAGE_PRESETS:
        state.current_image_preset = preset_id
        return jsonify({"success": True, "preset": config.IMAGE_PRESETS[preset_id]})
    else:
        return jsonify({"error": "Invalid preset"}), 400

@app.route('/api/image/generate', methods=['POST'])
@require_auth
def api_generate_image():
    """Generate an image from prompt."""
    if state.is_generating:
        return jsonify({"error": "Generation already in progress"}), 429
    
    data = request.json
    prompt = data.get('prompt', '')
    negative_prompt = data.get('negative_prompt', '')
    preset_id = data.get('preset', state.current_image_preset)
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    preset = config.IMAGE_PRESETS.get(str(preset_id), config.IMAGE_PRESETS["0"])
    
    def generate():
        state.is_generating = True
        try:
            yield f"data: {json.dumps({'status': 'starting', 'preset': preset['name']})}\n\n"
            
            result = generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                checkpoint=preset.get('checkpoint'),
                **{k: v for k, v in preset.items() if k not in ['name', 'checkpoint']}
            )
            
            if result and 'path' in result:
                yield f"data: {json.dumps({'status': 'complete', 'image': result['path'], 'filename': result['filename']})}\n\n"
            elif result and 'error' in result:
                yield f"data: {json.dumps({'status': 'error', 'error': result['error']})}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'error', 'error': 'Unknown generation failure'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
        finally:
            state.is_generating = False
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/image/output/<filename>')
@require_auth
def api_get_image(filename):
    """Serve generated images."""
    image_path = config.OUTPUT_DIR / filename
    if image_path.exists():
        return send_file(image_path, mimetype='image/png')
    return jsonify({"error": "Image not found"}), 404

@app.route('/api/history')
@require_auth
def api_history():
    """Get chat history."""
    return jsonify({"history": state.chat_history})

# ============================================
# BACKEND INTEGRATION
# ============================================

def get_available_llm_models():
    """Get list of available GGUF models."""
    models = []
    
    if config.MODELS_DIR.exists():
        for f in config.MODELS_DIR.glob("*.gguf"):
            models.append({
                "name": f.stem,
                "path": str(f),
                "size_gb": f.stat().st_size / (1024**3)
            })
    
    return sorted(models, key=lambda x: x['name'])

def load_llm_model(model_name: str) -> bool:
    """Load an LLM model."""
    global state
    
    try:
        # Find model file
        model_path = None
        for f in config.MODELS_DIR.glob("*.gguf"):
            if f.stem == model_name or model_name in f.name:
                model_path = f
                break
        
        if not model_path:
            print(f"[WebServer] Model not found: {model_name}")
            return False
        
        # Unload existing model first
        if state.vox_api is not None:
            try:
                state.vox_api.shutdown()
            except:
                pass
            state.vox_api = None
        
        # Import and initialize VoxAPI with the new model
        try:
            # Try direct import first (if VoxAI_Chat_API is in path)
            from vox_api import VoxAPI
        except ImportError:
            try:
                # Try as submodule
                from VoxAI_Chat_API.vox_api import VoxAPI
            except ImportError:
                print("[WebServer] VoxAPI not available")
                print(f"[WebServer] Looked in: {VOX_API_DIR}")
                return False
        
        # Create new VoxAPI instance with the model
        state.vox_api = VoxAPI(
            model_path=str(model_path),
            verbose=True
        )
        
        state.current_llm_model = model_name
        print(f"[WebServer] Loaded model: {model_name}")
        return True
        
    except Exception as e:
        print(f"[WebServer] Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def chat_stream(message: str):
    """Stream chat response from LLM."""
    if state.vox_api is None:
        # Try to initialize with first available model
        try:
            # Try direct import first
            try:
                from vox_api import VoxAPI
            except ImportError:
                from VoxAI_Chat_API.vox_api import VoxAPI
            
            models = get_available_llm_models()
            if models:
                model_path = models[0]['path']
                state.vox_api = VoxAPI(
                    model_path=model_path,
                    verbose=True
                )
                state.current_llm_model = models[0]['name']
                print(f"[WebServer] Auto-loaded model: {state.current_llm_model}")
            else:
                yield "Error: No models found in models directory."
                return
        except ImportError as e:
            yield f"Error: VoxAPI not available. Make sure VoxAI_Chat_API is set up. ({e})"
            return
        except Exception as e:
            yield f"Error: Could not initialize LLM backend. ({e})"
            return
    
    if state.vox_api is None:
        yield "Error: Could not initialize LLM backend."
        return
    
    try:
        for token in state.vox_api.chat(message, stream=True):
            yield token
    except Exception as e:
        yield f"Error: {str(e)}"

def generate_image(prompt: str, negative_prompt: str = "", **kwargs):
    """Generate an image using the image generator."""
    try:
        print("[WebServer] Request received: generate_image")
        # Lazy load image generator
        if state.image_generator is None:
            print("[WebServer] Lazy loading ImageGenerator...")
            try:
                from backend.image_generator import ImageGenerator
                print("[WebServer] Imported ImageGenerator class.")
                state.image_generator = ImageGenerator()
                print("[WebServer] Instantiated ImageGenerator.")
            except ImportError as e:
                print(f"[WebServer] ImageGenerator import failed: {e}")
                import traceback
                traceback.print_exc()
                return {"error": f"Failed to import ImageGenerator: {e}"}
            except Exception as e:
                print(f"[WebServer] ImageGenerator init failed: {e}")
                import traceback
                traceback.print_exc()
                return {"error": f"Failed to initialize ImageGenerator: {e}"}

        # 1. Parse Args to GenerationConfig

        # 1. Parse Args to GenerationConfig
        # Map web args to config
        width = int(kwargs.get('width', 1024))
        height = int(kwargs.get('height', 1024))
        steps = int(kwargs.get('steps', 20))
        cfg_scale = float(kwargs.get('cfg', 7.0))
        sampler = kwargs.get('sampler', "euler")
        
        gen_config = GenerationConfig(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width, 
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler=sampler
        )
        
        # 2. Check/Load Model
        checkpoint = kwargs.get('checkpoint')
        if checkpoint:
             if state.image_generator.current_model != checkpoint:
                 print(f"[WebServer] Switching model to {checkpoint}")
                 # We need to find the full path or just pass name if manager handles it
                 # ImageGenerator.load_model takes model_id (filename)
                 try:
                     state.image_generator.load_model(checkpoint)
                 except Exception as e:
                     print(f"[WebServer] Model load failed: {e}")
                     import traceback
                     traceback.print_exc()
                     return {"error": f"Failed to load model {checkpoint}: {e}"}
        elif not state.image_generator.current_model:
             # Try to load a default if nothing loaded
             print("[WebServer] No model loaded, trying default...")
             # This might fail if user has no models, but better than crash
        
        # 3. Generate
        # Use generate_and_save to get a file path
        output_path_str = state.image_generator.generate_and_save(gen_config)
        
        if output_path_str:
            filename = os.path.basename(output_path_str)
            return {
                "path": f"/api/image/output/{filename}",
                "filename": filename,
                "full_path": output_path_str
            }
        
        
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"[WebServer] Image generation error: {e}")
        print(trace)
        return {"error": str(e)}
    
    return {"error": "Unknown error (silent failure)"}

# ============================================
# HTML TEMPLATES
# ============================================

LOGIN_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoxAI - Login</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
        }
        .login-box {
            background: rgba(30, 30, 46, 0.95);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            width: 100%;
            max-width: 400px;
            border: 1px solid rgba(0, 128, 128, 0.3);
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            color: #00b4b4;
            font-size: 28px;
        }
        .logo {
            text-align: center;
            font-size: 48px;
            margin-bottom: 10px;
        }
        input[type="password"] {
            width: 100%;
            padding: 14px 16px;
            background: #252536;
            border: 1px solid #444;
            border-radius: 8px;
            color: #fff;
            font-size: 16px;
            margin-bottom: 20px;
        }
        input:focus {
            outline: none;
            border-color: #008080;
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #008080, #006666);
            border: none;
            border-radius: 8px;
            color: #fff;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 128, 128, 0.4);
        }
        .error {
            background: rgba(255, 82, 82, 0.2);
            border: 1px solid #ff5252;
            color: #ff8a8a;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="login-box">
        <div class="logo">🤖</div>
        <h1>VoxAI Remote</h1>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        <form method="POST">
            <input type="password" name="password" placeholder="Enter password..." autofocus required>
            <button type="submit">Access VoxAI</button>
        </form>
    </div>
</body>
</html>
'''

MAIN_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoxAI Remote</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #1e1e2e;
            --bg-tertiary: #252536;
            --accent: #008080;
            --accent-light: #00b4b4;
            --text-primary: #e0e0e0;
            --text-secondary: #888;
            --border: #333;
            --user-msg: #006666;
            --assistant-msg: #2d2d44;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        /* Header */
        .header {
            background: var(--bg-secondary);
            padding: 12px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .header h1 {
            font-size: 20px;
            color: var(--accent-light);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .header-actions {
            display: flex;
            gap: 10px;
        }
        
        .btn {
            padding: 8px 16px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        
        .btn:hover {
            background: var(--accent);
            border-color: var(--accent);
        }
        
        .btn-primary {
            background: var(--accent);
            border-color: var(--accent);
        }
        
        /* Mode Tabs */
        .mode-tabs {
            display: flex;
            background: var(--bg-secondary);
            padding: 0 20px;
            border-bottom: 1px solid var(--border);
        }
        
        .mode-tab {
            padding: 12px 24px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            color: var(--text-secondary);
            transition: all 0.2s;
        }
        
        .mode-tab:hover {
            color: var(--text-primary);
        }
        
        .mode-tab.active {
            color: var(--accent-light);
            border-bottom-color: var(--accent-light);
        }
        
        /* Main Content */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 900px;
            margin: 0 auto;
            width: 100%;
            padding: 20px;
        }
        
        /* Chat Container */
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding-bottom: 20px;
        }
        
        .message {
            margin-bottom: 16px;
            display: flex;
            flex-direction: column;
        }
        
        .message.user {
            align-items: flex-end;
        }
        
        .message-content {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 12px;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        
        .message.user .message-content {
            background: var(--user-msg);
            border-bottom-right-radius: 4px;
        }
        
        .message.assistant .message-content {
            background: var(--assistant-msg);
            border-bottom-left-radius: 4px;
        }
        
        .message-label {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }
        
        /* Input Area */
        .input-area {
            background: var(--bg-secondary);
            padding: 16px;
            border-radius: 12px;
            border: 1px solid var(--border);
        }
        
        .input-row {
            display: flex;
            gap: 10px;
        }
        
        #message-input {
            flex: 1;
            padding: 16px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 16px;
            resize: vertical;
            min-height: 80px;
            max-height: 300px;
            line-height: 1.5;
            font-family: inherit;
        }
        
        #message-input:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        #send-btn {
            padding: 12px 24px;
            background: var(--accent);
            border: none;
            border-radius: 8px;
            color: #fff;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        #send-btn:hover {
            background: var(--accent-light);
        }
        
        #send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        /* Model Selector */
        .model-selector {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }
        
        .model-selector label {
            color: var(--text-secondary);
            font-size: 13px;
        }
        
        .model-selector select {
            flex: 1;
            padding: 8px 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 14px;
        }
        
        /* Image Generation */
        .image-section {
            display: none;
        }
        
        .image-section.active {
            display: block;
        }
        
        .preset-cards {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }
        
        .preset-card {
            padding: 16px;
            background: var(--bg-tertiary);
            border: 2px solid var(--border);
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .preset-card:hover {
            border-color: var(--accent);
        }
        
        .preset-card.selected {
            border-color: var(--accent-light);
            background: rgba(0, 128, 128, 0.1);
        }
        
        .preset-card h3 {
            margin-bottom: 8px;
            color: var(--accent-light);
        }
        
        .preset-card p {
            font-size: 12px;
            color: var(--text-secondary);
        }
        
        .image-preview {
            margin-top: 20px;
            text-align: center;
        }
        
        .image-preview img {
            max-width: 100%;
            max-height: 512px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        
        .status-text {
            color: var(--accent-light);
            margin: 10px 0;
        }
        
        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        
        .modal-overlay.active {
            display: flex;
        }
        
        .modal {
            background: var(--bg-secondary);
            padding: 24px;
            border-radius: 12px;
            max-width: 500px;
            width: 90%;
            border: 1px solid var(--border);
        }
        
        .modal h2 {
            margin-bottom: 16px;
            color: var(--accent-light);
        }
        
        .model-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .model-item {
            padding: 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            margin-bottom: 8px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .model-item:hover {
            border-color: var(--accent);
        }
        
        .model-item.current {
            border-color: var(--accent-light);
            background: rgba(0, 128, 128, 0.1);
        }
        
        .model-size {
            font-size: 12px;
            color: var(--text-secondary);
        }
        
        /* Welcome */
        .welcome {
            text-align: center;
            padding: 60px 20px;
        }
        
        .welcome h2 {
            font-size: 28px;
            margin-bottom: 16px;
            color: var(--accent-light);
        }
        
        .welcome p {
            color: var(--text-secondary);
            margin-bottom: 24px;
        }
        
        /* Keyboard hint */
        .keyboard-hint {
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: var(--bg-secondary);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            color: var(--text-secondary);
            border: 1px solid var(--border);
        }
        
        kbd {
            background: var(--bg-tertiary);
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid var(--border);
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .main-content {
                padding: 10px;
            }
            
            .message-content {
                max-width: 90%;
            }
            
            .preset-cards {
                grid-template-columns: 1fr 1fr;
            }
            
            .keyboard-hint {
                display: none;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 VoxAI Remote</h1>
        <div class="header-actions">
            <span id="model-badge" class="btn" style="cursor: default;">No model loaded</span>
            <button class="btn" onclick="showModelSelector()">Switch Model</button>
            <button class="btn" onclick="clearChat()">Clear</button>
            <a href="/logout" class="btn">Logout</a>
        </div>
    </div>
    
    <div class="mode-tabs">
        <div class="mode-tab active" data-mode="chat" onclick="switchMode('chat')">💬 Chat</div>
        <div class="mode-tab" data-mode="image" onclick="switchMode('image')">🎨 Image Generation</div>
    </div>
    
    <div class="main-content">
        <!-- Chat Section -->
        <div id="chat-section" class="chat-section">
            <div class="chat-container" id="chat-container">
                <div class="welcome">
                    <h2>Hey, what can I do for you today?</h2>
                    <p>Start chatting or switch to Image Generation mode</p>
                </div>
            </div>
            
            <div class="input-area">
                <div class="input-row">
                    <textarea id="message-input" placeholder="Type your message..." rows="1"></textarea>
                    <button id="send-btn" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
        
        <!-- Image Section -->
        <div id="image-section" class="image-section">
            <h2 style="margin-bottom: 16px;">Choose a preset:</h2>
            <div class="preset-cards" id="preset-cards"></div>
            
            <div class="input-area" style="border: 1px solid var(--accent); box-shadow: 0 0 15px rgba(0, 128, 128, 0.1);">
                <div class="input-row" style="flex-direction: column;">
                    <label style="color: var(--accent-light); font-size: 14px; margin-bottom: 8px; font-weight: bold;">PROMPT</label>
                    <textarea id="image-prompt" placeholder="Describe your masterpiece..." rows="4" style="margin-bottom: 12px;"></textarea>
                    <button id="generate-btn" onclick="generateImage()" style="width: 100%; padding: 16px; font-size: 18px;">✨ IMPACT GENERATE</button>
                </div>
            </div>
            
            <div class="image-preview" id="image-preview">
                <p class="status-text" id="image-status"></p>
                <img id="generated-image" style="display: none;">
            </div>
        </div>
    </div>
    
    <!-- Model Selector Modal -->
    <div class="modal-overlay" id="model-modal">
        <div class="modal">
            <h2>Select LLM Model</h2>
            <div class="model-list" id="model-list"></div>
            <button class="btn" style="margin-top: 16px; width: 100%;" onclick="hideModelSelector()">Cancel</button>
        </div>
    </div>
    
    <div class="keyboard-hint">
        <kbd>Ctrl</kbd> + <kbd>M</kbd> to switch models
    </div>
    
    <script>
        let currentMode = 'chat';
        let currentModel = null;
        let selectedPreset = '0';
        let isGenerating = false;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            loadStatus();
            loadPresets();
            
            // Auto-resize textarea
            const inputs = document.querySelectorAll('textarea');
            inputs.forEach(input => {
                input.addEventListener('input', () => {
                    input.style.height = 'auto';
                    input.style.height = Math.min(input.scrollHeight, 150) + 'px';
                });
            });
            
            // Enter to send
            document.getElementById('message-input').addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            
            document.getElementById('image-prompt').addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    generateImage();
                }
            });
        });
        
        // Keyboard shortcut
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'm') {
                e.preventDefault();
                showModelSelector();
            }
        });
        
        async function loadStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                
                if (data.current_llm) {
                    currentModel = data.current_llm;
                    document.getElementById('model-badge').textContent = currentModel;
                }
            } catch (e) {
                console.error('Failed to load status:', e);
            }
        }
        
        function switchMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.mode-tab').forEach(tab => {
                tab.classList.toggle('active', tab.dataset.mode === mode);
            });
            
            document.getElementById('chat-section').style.display = mode === 'chat' ? 'flex' : 'none';
            document.getElementById('chat-section').style.flexDirection = 'column';
            document.getElementById('chat-section').style.flex = '1';
            document.getElementById('image-section').classList.toggle('active', mode === 'image');
        }
        
        async function loadPresets() {
            try {
                const res = await fetch('/api/image/presets');
                const data = await res.json();
                
                const container = document.getElementById('preset-cards');
                container.innerHTML = '';
                
                for (const [id, preset] of Object.entries(data.presets)) {
                    const card = document.createElement('div');
                    card.className = 'preset-card' + (id === selectedPreset ? ' selected' : '');
                    card.onclick = () => selectPreset(id);
                    card.innerHTML = `
                        <h3>[${id}] ${preset.name}</h3>
                        <p>${preset.width}x${preset.height} • ${preset.steps} steps • CFG ${preset.cfg}</p>
                    `;
                    container.appendChild(card);
                }
            } catch (e) {
                console.error('Failed to load presets:', e);
            }
        }
        
        function selectPreset(id) {
            selectedPreset = id;
            document.querySelectorAll('.preset-card').forEach(card => {
                card.classList.remove('selected');
            });
            event.target.closest('.preset-card').classList.add('selected');
        }
        
        async function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();
            
            if (!message || isGenerating) return;
            
            // Clear welcome message
            const welcome = document.querySelector('.welcome');
            if (welcome) welcome.remove();
            
            // Add user message
            addMessage('user', message);
            input.value = '';
            input.style.height = 'auto';
            
            // Disable button
            document.getElementById('send-btn').disabled = true;
            isGenerating = true;
            
            // Add assistant placeholder
            const assistantDiv = addMessage('assistant', '');
            const contentDiv = assistantDiv.querySelector('.message-content');
            
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message})
                });
                
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                
                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    
                    const text = decoder.decode(value);
                    const lines = text.split('\\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.chunk) {
                                    contentDiv.textContent += data.chunk;
                                    scrollToBottom();
                                }
                            } catch (e) {}
                        }
                    }
                }
            } catch (e) {
                contentDiv.textContent = 'Error: ' + e.message;
            }
            
            document.getElementById('send-btn').disabled = false;
            isGenerating = false;
        }
        
        function addMessage(role, content) {
            const container = document.getElementById('chat-container');
            const div = document.createElement('div');
            div.className = 'message ' + role;
            div.innerHTML = `
                <span class="message-label">${role === 'user' ? 'You' : 'VoxAI'}</span>
                <div class="message-content">${content}</div>
            `;
            container.appendChild(div);
            scrollToBottom();
            return div;
        }
        
        function scrollToBottom() {
            const container = document.getElementById('chat-container');
            container.scrollTop = container.scrollHeight;
        }
        
        async function clearChat() {
            if (!confirm('Clear chat history?')) return;
            
            try {
                await fetch('/api/chat/clear', {method: 'POST'});
                document.getElementById('chat-container').innerHTML = `
                    <div class="welcome">
                        <h2>Hey, what can I do for you today?</h2>
                        <p>Start chatting or switch to Image Generation mode</p>
                    </div>
                `;
            } catch (e) {
                console.error('Failed to clear chat:', e);
            }
        }
        
        async function generateImage() {
            const prompt = document.getElementById('image-prompt').value.trim();
            if (!prompt || isGenerating) return;
            
            isGenerating = true;
            document.getElementById('generate-btn').disabled = true;
            document.getElementById('image-status').textContent = 'Starting generation...';
            document.getElementById('generated-image').style.display = 'none';
            
            try {
                const res = await fetch('/api/image/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt, preset: selectedPreset})
                });
                
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                
                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    
                    const text = decoder.decode(value);
                    const lines = text.split('\\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                
                                if (data.status === 'starting') {
                                    document.getElementById('image-status').textContent = 
                                        `Generating with ${data.preset}...`;
                                } else if (data.status === 'complete') {
                                    document.getElementById('image-status').textContent = 'Complete!';
                                    const img = document.getElementById('generated-image');
                                    img.src = data.image + '?t=' + Date.now();
                                    img.style.display = 'block';
                                } else if (data.error) {
                                    document.getElementById('image-status').textContent = 'Error: ' + data.error;
                                }
                            } catch (e) {}
                        }
                    }
                }
            } catch (e) {
                document.getElementById('image-status').textContent = 'Error: ' + e.message;
            }
            
            isGenerating = false;
            document.getElementById('generate-btn').disabled = false;
        }
        
        async function showModelSelector() {
            const modal = document.getElementById('model-modal');
            const list = document.getElementById('model-list');
            
            list.innerHTML = '<p style="color: var(--text-secondary);">Loading models...</p>';
            modal.classList.add('active');
            
            try {
                const res = await fetch('/api/models/llm');
                const data = await res.json();
                
                if (data.models.length === 0) {
                    list.innerHTML = '<p style="color: var(--text-secondary);">No models found</p>';
                    return;
                }
                
                list.innerHTML = '';
                data.models.forEach(model => {
                    const item = document.createElement('div');
                    item.className = 'model-item' + (model.name === currentModel ? ' current' : '');
                    item.onclick = () => loadModel(model.name);
                    item.innerHTML = `
                        <span>${model.name}</span>
                        <span class="model-size">${model.size_gb.toFixed(1)} GB</span>
                    `;
                    list.appendChild(item);
                });
            } catch (e) {
                list.innerHTML = '<p style="color: #ff5252;">Failed to load models</p>';
            }
        }
        
        function hideModelSelector() {
            document.getElementById('model-modal').classList.remove('active');
        }
        
        async function loadModel(name) {
            hideModelSelector();
            document.getElementById('model-badge').textContent = 'Loading...';
            
            try {
                const res = await fetch('/api/models/llm/load', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model: name})
                });
                
                const data = await res.json();
                
                if (data.success) {
                    currentModel = name;
                    document.getElementById('model-badge').textContent = name;
                } else {
                    alert('Failed to load model: ' + (data.error || 'Unknown error'));
                    document.getElementById('model-badge').textContent = currentModel || 'No model';
                }
            } catch (e) {
                alert('Error: ' + e.message);
                document.getElementById('model-badge').textContent = currentModel || 'No model';
            }
        }
        
        // Close modal on click outside
        document.getElementById('model-modal').addEventListener('click', (e) => {
            if (e.target.classList.contains('modal-overlay')) {
                hideModelSelector();
            }
        });
    </script>
</body>
</html>
'''

# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description='VoxAI Web Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=7860, help='Port to listen on')
    parser.add_argument('--password', required=True, help='Access password')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    config.HOST = args.host
    config.PORT = args.port
    config.PASSWORD = args.password
    
    print("=" * 50)
    print("  VoxAI Web Server")
    print("=" * 50)
    print(f"\n  URL: http://{args.host}:{args.port}")
    print(f"  Password: {'*' * len(args.password)}")
    print(f"\n  To access remotely, use your public IP or set up port forwarding")
    print("=" * 50 + "\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()
