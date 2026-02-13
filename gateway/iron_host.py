#!/usr/bin/env python3
"""
Iron Tunnel Host v10.3
Secure AI Gateway - Main Application
"""
import os
import sys
import time
import uvicorn
import socket
import logging
from datetime import datetime
from pathlib import Path
from typing import List
from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from colorama import init, Fore, Style

# Import Core Modules
from lib.config import (
    load_config, save_config, load_host_identity, log_event,
    VERSION, TEMPLATES_DIR, STATIC_DIR, EXPORTS_DIR, host_config, host_identity
)
from lib.database import load_database, save_database, client_db
from lib.security import is_banned, record_failed_attempt, get_session, create_session, check_rate_limit, get_decoy_url
from lib.client_gen import generate_client_package
from lib.admin import start_admin_console

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ai_bridge
try:
    import pyngrok.ngrok as ngrok
except ImportError:
    print("Error: pyngrok not installed. Run 'pip install pyngrok'")
    sys.exit(1)

# Initialize
init(autoreset=True)
app = FastAPI(docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Mount Static
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Shared Paths (from parent AI_GUI config.json) ---
AI_GUI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # A:\Github\AI_GUI
AI_GUI_CONFIG = os.path.join(AI_GUI_ROOT, "config.json")

def get_shared_paths():
    """Read the parent AI_GUI config.json for model/output directories."""
    import json
    defaults = {
        "checkpoint_dir": os.path.join(AI_GUI_ROOT, "models", "checkpoints"),
        "lora_dir": os.path.join(AI_GUI_ROOT, "models", "loras"),
        "vae_dir": os.path.join(AI_GUI_ROOT, "models", "vae"),
        "text_encoder_dir": os.path.join(AI_GUI_ROOT, "models", "text_encoders"),
        "output_dir": os.path.join(AI_GUI_ROOT, "outputs", "images"),
    }

    if os.path.exists(AI_GUI_CONFIG):
        try:
            with open(AI_GUI_CONFIG, "r") as f:
                cfg = json.load(f)
            img_cfg = cfg.get("image", {})

            # Resolve relative paths against AI_GUI root
            for key in ["checkpoint_dir", "lora_dir", "vae_dir", "text_encoder_dir", "output_dir"]:
                val = img_cfg.get(key)
                if val:
                    if not os.path.isabs(val):
                        defaults[key] = os.path.join(AI_GUI_ROOT, val)
                    else:
                        defaults[key] = val
        except Exception as e:
            print(f"{Fore.YELLOW}[WARN] Could not read AI_GUI config: {e}")

    return defaults

SHARED_PATHS = get_shared_paths()

# =============================================================================
# MIDDLEWARE & UTILS
# =============================================================================

def get_client_ip(request: Request):
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]
    return request.client.host

from lib.security import is_banned, record_failed_attempt, get_session, create_session, check_rate_limit, get_decoy_url, register_ip, is_ip_authenticated

# ...

async def verify_access(request: Request):
    ip = get_client_ip(request)

    # PUBLIC MODE: If enabled, grant access to everyone (no install code needed)
    if host_config.get("public_mode", False):
        # Still create a session for consistency
        token = request.cookies.get("iron_session")
        session = get_session(token) if token else None
        if not session:
            token = create_session(f"PUBLIC_{ip}", ip)
            session = get_session(token)
            session["_new_token"] = token
        return session

    # 0. Check Magic Link (Host Access)
    magic_key = request.query_params.get("key")
    if magic_key and magic_key == host_identity.get("secret_key"):
        # Auto-trust this IP as HOST
        register_ip(ip, "HOST")
        print(f"{Fore.GREEN}[SECURITY] Host Authenticated via Magic Link: {ip}")
        # Create session immediately
        token = create_session("HOST", ip)
        session = get_session(token)
        session["_new_token"] = token
        return session

    # 1. Check Ban
    if is_banned(ip):
        # Allow localhost even if "banned" (safety net)
        if ip not in ["127.0.0.1", "::1"]:
            log_event("BLOCK", details={"ip": ip, "reason": "Banned"})
            print(f"{Fore.RED}[SECURITY] Blocked banned IP: {ip} -> Redirecting to Decoy")
            return RedirectResponse(get_decoy_url())

    # 2. Check Session Cookie
    token = request.cookies.get("iron_session")
    session = get_session(token) if token else None
    
    # 3. Check IP Authentication (Client Exe)
    if not session:
        ip_auth = is_ip_authenticated(ip)
        if ip_auth:
            # Create a session for them so they have a cookie next time
            token = create_session(ip_auth["client_id"], ip)
            session = get_session(token)
            # We can't set cookie here easily without response object, 
            # so we'll just return session and let route handle it? 
            # Or just rely on IP auth for now.
            session["_new_token"] = token # Hint to set cookie
    
    if not session:
        # Allow localhost without session (for desktop app SearchService, etc.)
        if ip in ["127.0.0.1", "::1"]:
            return {"client_id": "LOCAL", "ip": ip, "_local": True}

        # Log the attempt
        print(f"{Fore.YELLOW}[ACCESS] Unauthorized access attempt from: {ip} -> Redirecting to Decoy")
        record_failed_attempt(ip) # Only record if truly unauthorized
        return RedirectResponse(get_decoy_url())

    return session

# ... (Routes) ...

@app.post("/api/activate")
async def activate_client(request: Request):
    """Handle client activation"""
    try:
        data = await request.json()
        install_code = data.get("install_code")
        client_id = data.get("client_id")
        
        # Verify code against DB
        # We need to scan clients for this code
        target_client = None
        for cid, c in client_db["clients"].items():
            if c.get("install_code") == install_code and not c.get("install_code_used"):
                target_client = cid
                break
        
        if target_client:
            # Activate!
            client_db["clients"][target_client]["install_code_used"] = True
            client_db["clients"][target_client]["activated_at"] = time.time()
            client_db["clients"][target_client]["status"] = "ACTIVE"
            client_db["clients"][target_client]["hwid"] = client_id # Bind to their random ID
            save_database()
            
            print(f"{Fore.GREEN}[ACTIVATION] Client {target_client} activated successfully!")
            return JSONResponse({"status": "OK", "message": "Activation successful"})
        else:
            print(f"{Fore.RED}[ACTIVATION] Failed attempt with code: {install_code}")
            return JSONResponse({"status": "ERROR", "message": "Invalid code"}, status_code=400)
            
    except Exception as e:
        return JSONResponse({"status": "ERROR", "message": str(e)}, status_code=500)

@app.post("/api/beacon")
async def beacon(request: Request):
    """Handle client heartbeat"""
    ip = get_client_ip(request)
    # in real setup, client sends signed payload. For now, trust IP if active?
    # Client sends pairing code? 
    # Let's just trust for now if they have a valid client ID in payload?
    # Or simplified: if they hit this endpoint, we register their IP if they send a valid Install Code?
    # No, install code is one-time.
    # The client sends "client_id" and "signature".
    
    try:
        data = await request.json()
        cid = data.get("client_id")
        # sig = data.get("signature") 
        
        if cid and cid in client_db["clients"]:
             if client_db["clients"][cid]["status"] == "ACTIVE":
                 register_ip(ip, cid)
                 return JSONResponse({"status": "APPROVED"})
        
        return JSONResponse({"status": "DENIED"})
    except:
        return JSONResponse({"status": "ERROR"})


# =============================================================================
# ROUTES
# =============================================================================

# =============================================================================
# BACKEND SETUP
# =============================================================================
# =============================================================================
# BACKEND SETUP
# =============================================================================
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Add parent directory (AI_GUI) to path to find 'backend' package
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Suppress backend logging from flooding IronGate CMD
# The image gen backend uses both print() and Python logging for debug output
logging.getLogger("backend").setLevel(logging.CRITICAL)
logging.getLogger("diffusers").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("torch").setLevel(logging.CRITICAL)

try:
    from backend.image_generator import get_generator, GenerationConfig, ModelFamily
    IMAGE_GEN_AVAILABLE = True
except ImportError:
    IMAGE_GEN_AVAILABLE = False

# --- Suppress backend stdout in IronGate CMD ---
# The image gen backend uses raw print() for debug output ([Manager], [SD15], etc.)
# We use a NullWriter class instead of os.devnull to avoid charmap encoding errors on Windows.
import contextlib
import io

class _NullWriter(io.TextIOBase):
    """A writer that silently discards everything. Avoids charmap codec errors."""
    def write(self, s): return len(s)
    def flush(self): pass

@contextlib.contextmanager
def _quiet():
    """Temporarily suppress stdout/stderr from backend print() calls."""
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = _NullWriter()
        sys.stderr = _NullWriter()
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err

# =============================================================================
# ROUTES
# =============================================================================

@app.get("/")
async def root(request: Request):
    """Entry point"""
    session = await verify_access(request)
    
    if isinstance(session, RedirectResponse):
        return session
    
    if session:
        return RedirectResponse("/app")
    
    # If not authenticated, show login or decoy? 
    # For now, simplistic login page
    return templates.TemplateResponse("login.html", {"request": request, "site_name": host_config.get("site_name", "VoxAI")}) 

@app.get("/app")
async def app_page(request: Request):
    """Main Application Interface"""
    session = await verify_access(request)
    
    if isinstance(session, RedirectResponse):
        return session
    
    if not session:
        return RedirectResponse("/")

    # Layout Engine
    layout = request.cookies.get("vox_layout", "aurora")
    valid_layouts = ["aurora"]
    if layout not in valid_layouts: layout = "aurora"
    
    # Serve specific layout template
    template_path = f"layouts/{layout}.html"

    response = templates.TemplateResponse(template_path, {
        "request": request,
        "site_name": host_config.get("site_name", "VoxAI"),
        "client_name": session.get("client_id", "Debug User") if session else "Debug User",
        "client_id": session.get("client_id", "DEBUG") if session else "DEBUG",
        "image_gen_enabled": IMAGE_GEN_AVAILABLE,
        "current_layout": layout
    })

    # Set session cookie if created via IP auth (no cookie yet)
    new_token = session.get("_new_token") if session else None
    if new_token:
        response.set_cookie(key="iron_session", value=new_token, max_age=86400)

    return response

@app.get("/status")
async def client_status(request: Request):
    """Check if a client has been approved (used by waiting page polling)"""
    client_id = request.query_params.get("client_id", "")
    if client_id and client_id in client_db["clients"]:
        client = client_db["clients"][client_id]
        if client.get("status") == "ACTIVE":
            ip = get_client_ip(request)
            register_ip(ip, client_id)
            token = create_session(client_id, ip)
            return JSONResponse({
                "status": "APPROVED",
                "redirect": f"/app"
            })
    return JSONResponse({"status": "WAITING"})

@app.post("/api/set_layout")
async def set_layout(request: Request):
    """Switch UI Layout"""
    session = await verify_access(request)
    if isinstance(session, RedirectResponse) or not session:
        return JSONResponse({"error": "Unauthorized"}, 403)
        
    data = await request.json()
    layout = data.get("layout", "standard")
    
    response = JSONResponse({"status": "success", "layout": layout})
    response.set_cookie(key="vox_layout", value=layout, max_age=31536000) # 1 year
    return response

# --- Model Name Helpers ---

def shorten_checkpoint_name(filename):
    """Shorten image checkpoint filenames to readable display names.
    e.g. 'copaxTimeless_xivSDXL.safetensors' -> 'Copax Timeless SDXL'
         'flux1-schnell-Q4_K_S.gguf' -> 'Flux1 Schnell'
         'fluxFusionV24StepsGGUFNF4_V2Fp8.safetensors' -> 'Flux Fusion V2'
    """
    import re
    name = filename
    # Remove extension
    for ext in ['.safetensors', '.ckpt', '.gguf']:
        name = name.replace(ext, '')

    # Remove quantization suffixes (Q4_K_M, Q4_K_S, IQ4_XS, Fp8, FP16, NF4, etc.)
    name = re.sub(r'[-_.]?(?:Q\d+_K_[A-Z]+|IQ\d+_[A-Z]+|[NF]F?\d+|Fp\d+|FP\d+)', '', name, flags=re.IGNORECASE)
    # Remove common suffixes like VAE, GGUF mentions
    name = re.sub(r'[-_.]?(?:GGUF|VAE|safetensors)', '', name, flags=re.IGNORECASE)

    # Split camelCase and underscores into words
    # Insert space before uppercase that follows lowercase: "copaxTimeless" -> "copax Timeless"
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    # Replace separators with space
    name = name.replace('-', ' ').replace('_', ' ').replace('.', ' ')
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()

    # Clean up remaining noise words
    parts = name.split()
    # Remove parts that are just version noise like "v02", "V2", "xiv", "based On"
    noise = {'based', 'on', 'based on'}
    cleaned = [p for p in parts if p.lower() not in noise]

    # Capitalize first letter of each word
    cleaned = [p[0].upper() + p[1:] if p else p for p in cleaned]

    # Limit to 3 meaningful words max
    if len(cleaned) > 3:
        cleaned = cleaned[:3]

    return ' '.join(cleaned) if cleaned else filename

def shorten_llm_name(filename):
    """Shorten LLM .gguf filenames to readable display names.
    e.g. 'Llama-3.2-3B-Instruct-Q4_K_M.gguf' -> 'Llama 3.2 3B'
         'Qwen2.5-0.5B-Instruct-Q4_K_M.gguf' -> 'Qwen2.5 0.5B'
         'dolphin-2.8-mistral-7b-v02-Q4_K_M.gguf' -> 'Dolphin 2.8 Mistral 7B'
    """
    import re
    name = filename.replace('.gguf', '')

    # Remove quantization suffixes
    name = re.sub(r'[-_]?(?:Q\d+_K_[A-Z]+|IQ\d+_[A-Z]+)', '', name, flags=re.IGNORECASE)

    # Remove "Instruct", "it", "chat" suffixes
    name = re.sub(r'[-_.]?(?:Instruct|instruct|chat)', '', name)
    # Remove version noise like "-v02"
    name = re.sub(r'[-_]v\d+', '', name)

    # Split on hyphens and underscores BUT preserve dots within version numbers (3.2, 2.5)
    # First, protect version-style dots: digit.digit
    name = re.sub(r'(\d)\.(\d)', r'\1_DOT_\2', name)
    # Now split on separators
    name = name.replace('-', ' ').replace('_', ' ')
    # Restore dots
    name = name.replace(' DOT ', '.')
    name = re.sub(r'\s+', ' ', name).strip()

    parts = name.split()
    # Remove stray dots that aren't part of versions
    cleaned = [p for p in parts if p != '.']

    # Capitalize and format size tokens (7b -> 7B)
    result = []
    for p in cleaned:
        if not p:
            continue
        if re.match(r'^\d+[bB]$', p):
            result.append(p.upper())
        elif p[0].islower():
            result.append(p[0].upper() + p[1:])
        else:
            result.append(p)

    # Limit to 4 words
    if len(result) > 4:
        result = result[:4]

    return ' '.join(result) if result else filename

# --- LLM Models API ---

@app.get("/api/llm-models")
async def list_llm_models(request: Request):
    """List available LLM chat models — local .gguf, cloud (RunPod), and provider (Gemini)"""
    session = await verify_access(request)
    if isinstance(session, RedirectResponse) or not session:
        return JSONResponse({"error": "Unauthorized"}, 403)

    # --- LOCAL models (.gguf) ---
    local_models = []
    llm_models_dir = os.path.join(AI_GUI_ROOT, "models", "llm")
    if os.path.exists(llm_models_dir):
        for f in sorted(os.listdir(llm_models_dir)):
            if f.endswith(".gguf"):
                filepath = os.path.join(llm_models_dir, f)
                try:
                    size_gb = os.path.getsize(filepath) / (1024 ** 3)
                except:
                    size_gb = 0
                local_models.append({
                    "name": f,
                    "display": shorten_llm_name(f),
                    "size_gb": round(size_gb, 1),
                    "source": "local"
                })

    # --- CLOUD models (RunPod) ---
    cloud_models = []
    try:
        cloud_list = ai_bridge.get_cloud_models()
        for m in cloud_list:
            cloud_models.append({
                "name": m["id"],
                "display": m["display"],
                "size_gb": 0,
                "source": "cloud"
            })
    except Exception:
        pass

    # --- PROVIDER models (all registered: Gemini, OpenAI, OpenRouter, DeepSeek, etc.) ---
    provider_models = []
    try:
        from utils.config_manager import ConfigManager
        from providers import PROVIDER_REGISTRY

        config = ConfigManager.load_config()
        llm_cfg = config.get("llm", {})
        providers_cfg = llm_cfg.get("providers", {})

        for cfg_key, (cls, display_name) in PROVIDER_REGISTRY.items():
            api_key = ConfigManager.get_provider_key(cfg_key)
            if not api_key:
                continue

            try:
                # Use selected models from config if available, else fetch from API
                selected = providers_cfg.get(cfg_key, {}).get("models", [])
                if selected:
                    model_ids = selected
                else:
                    model_ids = cls.list_available_models(api_key=api_key)

                for name in model_ids:
                    # Clean up display name
                    display = name.replace("models/", "")
                    if "/" in display:
                        display = display.split("/", 1)[1]  # OpenRouter: strip company prefix

                    provider_models.append({
                        "name": name,
                        "display": display,
                        "size_gb": 0,
                        "source": "provider",
                        "provider": display_name
                    })
            except Exception as e:
                print(f"[IronGate] Could not load {display_name} models: {e}")

    except Exception as e:
        print(f"[IronGate] Provider model loading error: {e}")

    return JSONResponse({
        "models": local_models,
        "cloud_models": cloud_models,
        "provider_models": provider_models
    })

# --- Cloud Pod Management ---

@app.post("/api/cloud/terminate")
async def terminate_cloud_pod(request: Request):
    """Terminate the active RunPod cloud pod."""
    session = await verify_access(request)
    if isinstance(session, RedirectResponse) or not session:
        return JSONResponse({"error": "Unauthorized"}, 403)
    try:
        ai_bridge.terminate_cloud_pod()
        return JSONResponse({"status": "success", "message": "Cloud pod terminated"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

# --- Image Generation API ---

@app.get("/api/models")
async def list_models(request: Request):
    """List available image generation models (local checkpoints + Gemini image models)"""
    session = await verify_access(request)
    if isinstance(session, RedirectResponse) or not session:
        return JSONResponse({"error": "Unauthorized"}, 403)

    models = []
    current = None
    current_display = None

    # --- Local checkpoints (SD/SDXL/Flux) ---
    if IMAGE_GEN_AVAILABLE:
        with _quiet():
            gen = get_generator(
                checkpoint_dir=SHARED_PATHS["checkpoint_dir"],
                lora_dir=SHARED_PATHS["lora_dir"],
                vae_dir=SHARED_PATHS["vae_dir"],
                text_encoder_dir=SHARED_PATHS["text_encoder_dir"],
                output_dir=SHARED_PATHS["output_dir"]
            )
            raw_models = gen.scan_checkpoints()

        for m in raw_models:
            m["display"] = shorten_checkpoint_name(m["name"])
            m["source"] = "local"
            models.append(m)

        current = gen.current_model
        current_display = shorten_checkpoint_name(current) if current else None

    # --- Cloud image generation models (Gemini + OpenAI) ---
    try:
        from utils.config_manager import ConfigManager

        gemini_key = ConfigManager.get_provider_key("gemini")
        openai_key = ConfigManager.get_provider_key("openai")

        # Gemini image models
        if gemini_key:
            try:
                from providers.gemini_provider import GeminiProvider
                _img_display = {
                    "gemini-2.5-flash-image": "Gemini Flash Image 🍌",
                    "gemini-3-pro-image-preview": "Gemini Pro Image ⚡",
                }
                for name in GeminiProvider.list_image_models(gemini_key):
                    display = _img_display.get(name, name)
                    models.append({
                        "name": name,
                        "display": display,
                        "family": "gemini",
                        "size_gb": 0,
                        "source": "provider"
                    })
            except Exception as e:
                print(f"[IronGate] Could not load Gemini image models: {e}")

        # OpenAI image models (DALL-E)
        if openai_key:
            try:
                from providers.openai_provider import OpenAIProvider
                _dalle_display = {
                    "gpt-image-1": "GPT Image 1  [Cloud]",
                    "dall-e-3": "DALL·E 3  [Cloud]",
                }
                for name in OpenAIProvider.IMAGE_MODELS:
                    display = _dalle_display.get(name, name)
                    models.append({
                        "name": name,
                        "display": display,
                        "family": "openai",
                        "size_gb": 0,
                        "source": "provider"
                    })
            except Exception as e:
                print(f"[IronGate] Could not load OpenAI image models: {e}")

        # --- Video generation models ---
        # Veo (Gemini key)
        if gemini_key:
            try:
                from providers.gemini_provider import GeminiProvider
                _veo_display = {
                    "veo-3.1-generate-preview": "Veo 3.1  [Video]",
                    "veo-3.1-fast-generate-preview": "Veo 3.1 Fast  [Video]",
                }
                for name in GeminiProvider.list_video_models():
                    models.append({
                        "name": name,
                        "display": _veo_display.get(name, name),
                        "family": "video",
                        "size_gb": 0,
                        "source": "provider"
                    })
            except Exception as e:
                print(f"[IronGate] Could not load Veo video models: {e}")

        # Sora (OpenAI key)
        if openai_key:
            try:
                from providers.openai_provider import OpenAIProvider
                _sora_display = {
                    "sora-2": "Sora 2  [Video]",
                    "sora-2-pro": "Sora 2 Pro  [Video]",
                }
                for name in OpenAIProvider.list_video_models():
                    models.append({
                        "name": name,
                        "display": _sora_display.get(name, name),
                        "family": "video",
                        "size_gb": 0,
                        "source": "provider"
                    })
            except Exception as e:
                print(f"[IronGate] Could not load Sora video models: {e}")

    except Exception as e:
        print(f"[IronGate] Provider image model loading error: {e}")

    return JSONResponse({"models": models, "current": current, "current_display": current_display})

@app.post("/api/image/generate")
async def generate_image_api(request: Request):
    """Generate Image Endpoint — routes to local backend, Gemini, or DALL-E."""
    session = await verify_access(request)
    if isinstance(session, RedirectResponse) or not session:
        return JSONResponse({"error": "Unauthorized"}, 403)

    try:
        data = await request.json()
        prompt = data.get("prompt")
        if not prompt:
            return JSONResponse({"error": "No prompt"}, 400)

        model = data.get("model", "")

        # --- Gemini Image Generation (uses google.genai SDK) ---
        if model.startswith("gemini-"):
            try:
                from utils.config_manager import ConfigManager
                api_key = ConfigManager.get_provider_key("gemini")
                if not api_key:
                    return JSONResponse({"error": "No Gemini API key configured"}, 400)

                from google import genai
                from google.genai import types

                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"]
                    )
                )

                # Extract image bytes from response parts
                img_data = None
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data is not None:
                            img_data = part.inline_data.data
                            break

                if not img_data:
                    return JSONResponse({"error": "Gemini returned no image data"}, 500)

                # Save to output dir
                output_dir = SHARED_PATHS["output_dir"]
                os.makedirs(output_dir, exist_ok=True)
                filename = f"gemini_{int(time.time())}.png"
                save_path = os.path.join(output_dir, filename)
                with open(save_path, "wb") as f:
                    f.write(img_data)

                return JSONResponse({
                    "status": "success",
                    "url": f"/api/image/output/{filename}",
                    "filename": filename
                })

            except Exception as e:
                return JSONResponse({"error": f"Gemini image gen error: {e}"}, 500)

        # --- OpenAI Image Generation (DALL-E 3 / GPT Image 1) ---
        if model.startswith("dall-e") or model.startswith("gpt-image"):
            try:
                from utils.config_manager import ConfigManager
                api_key = ConfigManager.get_provider_key("openai")
                if not api_key:
                    return JSONResponse({"error": "No OpenAI API key configured"}, 400)

                from openai import OpenAI
                import base64
                client = OpenAI(api_key=api_key)

                is_gpt_image = model.startswith("gpt-image")
                response = client.images.generate(
                    model=model,
                    prompt=prompt,
                    size="1024x1024",
                    quality="auto" if is_gpt_image else "standard",
                    n=1,
                )

                # GPT Image returns base64, DALL-E returns URL
                img_bytes = None
                if response.data[0].b64_json:
                    img_bytes = base64.b64decode(response.data[0].b64_json)
                elif response.data[0].url:
                    import requests as _requests
                    img_response = _requests.get(response.data[0].url, timeout=60)
                    img_response.raise_for_status()
                    img_bytes = img_response.content

                if not img_bytes:
                    return JSONResponse({"error": "OpenAI returned no image data"}, 500)

                # Save to output dir
                output_dir = SHARED_PATHS["output_dir"]
                os.makedirs(output_dir, exist_ok=True)
                tag = "gptimg" if is_gpt_image else "dalle"
                filename = f"{tag}_{int(time.time())}.png"
                save_path = os.path.join(output_dir, filename)
                with open(save_path, "wb") as f:
                    f.write(img_bytes)

                return JSONResponse({
                    "status": "success",
                    "url": f"/api/image/output/{filename}",
                    "filename": filename
                })

            except Exception as e:
                return JSONResponse({"error": f"OpenAI image gen error: {e}"}, 500)

        # --- Local Image Generation (SD/SDXL/Flux) ---
        if not IMAGE_GEN_AVAILABLE:
            return JSONResponse({"error": "Image backend unavailable"}, 503)

        with _quiet():
            gen = get_generator(
                checkpoint_dir=SHARED_PATHS["checkpoint_dir"],
                lora_dir=SHARED_PATHS["lora_dir"],
                vae_dir=SHARED_PATHS["vae_dir"],
                text_encoder_dir=SHARED_PATHS["text_encoder_dir"],
                output_dir=SHARED_PATHS["output_dir"]
            )

            # Load model if needed
            if model and model != gen.current_model:
                gen.load_model(model)

            config = GenerationConfig(
                prompt=prompt,
                negative_prompt=data.get("negative_prompt", ""),
                width=int(data.get("width", 1024)),
                height=int(data.get("height", 1024)),
                steps=int(data.get("steps", 20)),
                cfg_scale=float(data.get("cfg", 7.0)),
                sampler=data.get("sampler", "euler")
            )

            saved_path = gen.generate_and_save(config)

        return JSONResponse({
            "status": "success",
            "url": f"/api/image/output/{os.path.basename(saved_path)}",
            "filename": os.path.basename(saved_path)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

@app.get("/api/image/output/{filename}")
async def get_image_file(filename: str, request: Request):
    """Serve generated image/video from shared output directory"""
    session = await verify_access(request)
    if isinstance(session, RedirectResponse) or not session:
        return RedirectResponse("/")

    path = os.path.join(SHARED_PATHS["output_dir"], filename)
    if os.path.exists(path):
        # Serve videos with correct MIME type
        if filename.endswith(".mp4"):
            return FileResponse(path, media_type="video/mp4")
        return FileResponse(path)
    return JSONResponse({"error": "File not found"}, 404)


@app.post("/api/video/generate")
async def generate_video_api(request: Request):
    """Generate Video Endpoint — routes to Sora (OpenAI) or Veo (Gemini)."""
    session = await verify_access(request)
    if isinstance(session, RedirectResponse) or not session:
        return JSONResponse({"error": "Unauthorized"}, 403)

    try:
        data = await request.json()
        prompt = data.get("prompt")
        if not prompt:
            return JSONResponse({"error": "No prompt"}, 400)

        model = data.get("model", "")
        duration = int(data.get("duration", 8))
        aspect = data.get("aspect", "16:9")

        from utils.config_manager import ConfigManager

        output_dir = SHARED_PATHS["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        # --- Sora (OpenAI) ---
        if model.startswith("sora-"):
            api_key = ConfigManager.get_provider_key("openai")
            if not api_key:
                return JSONResponse({"error": "No OpenAI API key configured"}, 400)

            from providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(api_key=api_key)

            sora_res = "720x1280" if aspect == "9:16" else "1280x720"
            filename = f"sora_{int(time.time())}.mp4"
            save_path = os.path.join(output_dir, filename)

            provider.create_and_wait_video(
                prompt=prompt, model=model,
                duration=duration, resolution=sora_res,
                save_path=save_path,
            )

            return JSONResponse({
                "status": "success",
                "url": f"/api/image/output/{filename}",
                "filename": filename,
                "type": "video"
            })

        # --- Veo (Gemini) ---
        elif model.startswith("veo-"):
            api_key = ConfigManager.get_provider_key("gemini")
            if not api_key:
                return JSONResponse({"error": "No Gemini API key configured"}, 400)

            from providers.gemini_provider import GeminiProvider
            provider = GeminiProvider(api_key=api_key)

            filename = f"veo_{int(time.time())}.mp4"
            save_path = os.path.join(output_dir, filename)

            provider.create_and_wait_video(
                prompt=prompt, model=model,
                aspect_ratio=aspect, resolution="720p",
                duration=duration, save_path=save_path,
            )

            return JSONResponse({
                "status": "success",
                "url": f"/api/image/output/{filename}",
                "filename": filename,
                "type": "video"
            })

        else:
            return JSONResponse({"error": f"Unknown video model: {model}"}, 400)

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# --- File Upload API ---

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/upload")
async def upload_files(request: Request, files: List[UploadFile] = File(...)):
    """Handle file uploads from the web UI. Returns file info for AI context."""
    session = await verify_access(request)
    if isinstance(session, RedirectResponse) or not session:
        return JSONResponse({"error": "Unauthorized"}, 403)

    results = []
    TEXT_EXTENSIONS = {'.txt', '.py', '.js', '.json', '.csv', '.md', '.html', '.css',
                       '.xml', '.yaml', '.yml', '.log', '.ts', '.jsx', '.tsx', '.sh',
                       '.bat', '.cfg', '.ini', '.toml', '.sql', '.java', '.cpp', '.c',
                       '.h', '.rs', '.go', '.rb', '.php'}
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'}

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        content = await file.read()

        # Save to uploads dir
        safe_name = f"{int(time.time())}_{file.filename}"
        save_path = os.path.join(UPLOAD_DIR, safe_name)
        with open(save_path, 'wb') as f:
            f.write(content)

        if ext in TEXT_EXTENSIONS:
            # Read text content for AI context
            try:
                text_content = content.decode('utf-8', errors='replace')
                # Limit to ~4000 chars to avoid overwhelming the LLM
                if len(text_content) > 4000:
                    text_content = text_content[:4000] + '\n... [truncated]'
                results.append({
                    'filename': file.filename,
                    'type': 'text',
                    'content': text_content,
                    'size': len(content)
                })
            except Exception:
                results.append({
                    'filename': file.filename,
                    'type': 'binary',
                    'content': '',
                    'size': len(content)
                })
        elif ext in IMAGE_EXTENSIONS:
            results.append({
                'filename': file.filename,
                'type': 'image',
                'description': f'Image file ({ext.upper()}, {len(content)} bytes)',
                'url': f'/api/upload/{safe_name}',
                'size': len(content)
            })
        else:
            results.append({
                'filename': file.filename,
                'type': 'binary',
                'content': '',
                'size': len(content)
            })

    return JSONResponse({'files': results})

@app.get("/api/upload/{filename}")
async def get_upload(filename: str, request: Request):
    """Serve uploaded files"""
    session = await verify_access(request)
    if isinstance(session, RedirectResponse) or not session:
        return RedirectResponse("/")

    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "File not found"}, 404)

# --- Web Search API ---

# Search permissions: what domains are allowed/blocked
SEARCH_CONFIG = {
    "enabled": True,
    "max_results": 5,
    "allowed_domains": [],        # Empty = all allowed
    "blocked_domains": [          # Safety: block known harmful sites
        "*.onion", "darkweb.*",
    ],
    "safe_search": "moderate"     # "off", "moderate", "strict"
}

@app.post("/api/search")
async def web_search_endpoint(request: Request):
    """Secure web search through IronGate. Returns search results for AI context."""
    session = await verify_access(request)
    if isinstance(session, RedirectResponse) or not session:
        return JSONResponse({"error": "Unauthorized"}, 403)

    if not SEARCH_CONFIG["enabled"]:
        return JSONResponse({"error": "Web search is disabled"}, 403)

    try:
        data = await request.json()
        query = data.get("query", "").strip()
        max_results = min(int(data.get("max_results", 5)), 10)

        if not query:
            return JSONResponse({"error": "No search query"}, 400)

        # Perform search using DuckDuckGo (no API key needed, privacy-focused)
        try:
            # Try new 'ddgs' package first, fall back to old 'duckduckgo_search'
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            ddgs_client = DDGS()
            raw = ddgs_client.text(query, max_results=max_results, safesearch=SEARCH_CONFIG["safe_search"])

            results = []
            for r in raw:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })

            print(f"{Fore.CYAN}[SEARCH] '{query}' -> {len(results)} results")

            return JSONResponse({
                "status": "success",
                "query": query,
                "results": results,
                "count": len(results)
            })

        except ImportError:
            return JSONResponse({"error": "Search module not installed. Run: pip install ddgs"}, 503)
        except Exception as e:
            return JSONResponse({"error": f"Search failed: {str(e)}"}, 500)

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

@app.post("/api/search/fetch")
async def fetch_page_endpoint(request: Request):
    """Fetch a URL's text content through IronGate (for AI context enrichment)."""
    session = await verify_access(request)
    if isinstance(session, RedirectResponse) or not session:
        return JSONResponse({"error": "Unauthorized"}, 403)

    try:
        data = await request.json()
        url = data.get("url", "").strip()
        if not url:
            return JSONResponse({"error": "No URL"}, 400)

        # Basic safety check
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return JSONResponse({"error": "Only HTTP/HTTPS URLs allowed"}, 400)

        import requests as req_lib
        resp = req_lib.get(url, timeout=10, headers={
            'User-Agent': 'VoxAI-IronGate/1.0 (Research Assistant)'
        })
        resp.raise_for_status()

        # Extract text content (strip HTML)
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
                self._skip = False
            def handle_starttag(self, tag, attrs):
                if tag in ('script', 'style', 'nav', 'footer', 'header'):
                    self._skip = True
            def handle_endtag(self, tag):
                if tag in ('script', 'style', 'nav', 'footer', 'header'):
                    self._skip = False
            def handle_data(self, data):
                if not self._skip:
                    text = data.strip()
                    if text:
                        self.text.append(text)

        extractor = TextExtractor()
        extractor.feed(resp.text)
        page_text = '\n'.join(extractor.text)

        # Limit to ~4000 chars for AI context
        if len(page_text) > 4000:
            page_text = page_text[:4000] + '\n... [truncated]'

        return JSONResponse({
            "status": "success",
            "url": url,
            "title": parsed.netloc,
            "content": page_text
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

# --- Chat API ---

@app.post("/api/login")
async def login(request: Request):
    """Handle login via install code for web-based access"""
    ip = get_client_ip(request)

    try:
        data = await request.json()
        install_code = data.get("install_code", "").strip()

        if not install_code:
            return JSONResponse({"status": "ERROR", "message": "No code provided"}, status_code=400)

        # Check if this is the host secret key (admin shortcut)
        if install_code == host_identity.get("secret_key"):
            register_ip(ip, "HOST")
            token = create_session("HOST", ip)
            response = JSONResponse({"status": "OK", "message": "Host authenticated"})
            response.set_cookie(key="iron_session", value=token, max_age=86400)
            return response

        # Check against client install codes
        target_client = None
        for cid, c in client_db["clients"].items():
            if c.get("install_code") == install_code:
                if c.get("status") == "ACTIVE":
                    target_client = cid
                    break
                elif not c.get("install_code_used"):
                    # Activate on first web login
                    c["install_code_used"] = True
                    c["activated_at"] = time.time()
                    c["status"] = "ACTIVE"
                    save_database()
                    target_client = cid
                    break

        if target_client:
            register_ip(ip, target_client)
            token = create_session(target_client, ip)
            response = JSONResponse({"status": "OK", "message": "Access granted"})
            response.set_cookie(key="iron_session", value=token, max_age=86400)
            print(f"{Fore.GREEN}[LOGIN] Web login success: {target_client} from {ip}")
            return response
        else:
            record_failed_attempt(ip)
            print(f"{Fore.RED}[LOGIN] Failed web login from {ip}")
            return JSONResponse({"status": "ERROR", "message": "Invalid access code"}, status_code=401)

    except Exception as e:
        return JSONResponse({"status": "ERROR", "message": str(e)}, status_code=500)

@app.post("/chat")
async def chat_endpoint(request: Request):
    """AI Chat Endpoint (non-streaming fallback)"""
    session = await verify_access(request)
    if isinstance(session, RedirectResponse):
        return session

    try:
        data = await request.json()
        message = data.get("message", "")
        model = data.get("model")

        if not message:
            return JSONResponse({"reply": "Error: Empty message"})

        # Switch model if requested
        if model:
            ai_bridge.set_model(model)

        reply = ai_bridge.ask_the_brain(message)
        return JSONResponse({"reply": reply})
    except Exception as e:
        return JSONResponse({"reply": f"Error: {str(e)}"})

@app.post("/chat/stream")
async def chat_stream_endpoint(request: Request):
    """AI Chat Endpoint with Server-Sent Events.

    Supports three model sources: local (.gguf), cloud (RunPod), provider (Gemini/OpenAI).
    Streams thinking sections, boot progress, and content chunks.
    """
    session = await verify_access(request)
    if isinstance(session, RedirectResponse):
        return session

    try:
        data = await request.json()
        message = data.get("message", "")
        model = data.get("model")
        source = data.get("source", "local")  # "local" | "cloud" | "provider"
        provider_name = data.get("provider", "")  # "Gemini" | "OpenAI"
        history = data.get("history", [])
        system_prompt = data.get("system_prompt")

        if not message:
            return JSONResponse({"error": "Empty message"}, 400)

        import json as _json

        def event_generator():
            if source == "cloud":
                # Cloud GPU streaming via CloudStreamer
                try:
                    for event in ai_bridge.stream_cloud(
                        message, model_id=model,
                        history=history, system_prompt=system_prompt
                    ):
                        yield f"data: {_json.dumps(event)}\n\n"
                except Exception as e:
                    print(f"[IronGate] Cloud stream error: {e}")
                    yield f"data: {_json.dumps({'type': 'error', 'text': f'[Cloud Error] {e}'})}\n\n"

            elif source == "provider":
                # Provider streaming — uses provider's own API
                try:
                    from utils.config_manager import ConfigManager
                    from providers.base_provider import Message
                    from providers import PROVIDER_REGISTRY
                    config = ConfigManager.load_config()
                    llm_cfg = config.get("llm", {})
                    providers_cfg = llm_cfg.get("providers", {})

                    prov = provider_name or "Gemini"

                    # Resolve provider class and API key via registry
                    llm_provider = None
                    for cfg_key, (cls, display_name) in PROVIDER_REGISTRY.items():
                        if prov in (display_name, cfg_key, cls.__name__):
                            api_key = providers_cfg.get(cfg_key, {}).get("api_key", "")
                            llm_provider = cls(model=model, api_key=api_key)
                            break

                    if not llm_provider:
                        yield f"data: {_json.dumps({'type': 'error', 'text': f'Unknown provider: {prov}'})}\n\n"
                        return

                    msg_history = [Message(role=m["role"], content=m["content"]) for m in history] if history else None

                    import time as _time
                    token_count = 0
                    start_time = _time.perf_counter()

                    for chunk in llm_provider.stream_message(
                        prompt=message, history=msg_history,
                        system_prompt=system_prompt or "You are a helpful assistant."
                    ):
                        if chunk:
                            token_count += 1
                            yield f"data: {_json.dumps({'type': 'chunk', 'text': chunk})}\n\n"

                    duration = _time.perf_counter() - start_time
                    speed = token_count / duration if duration > 0 else 0
                    yield f"data: {_json.dumps({'type': 'done', 'stats': {'tokens': token_count, 'duration': round(duration, 2), 'speed': round(speed, 2)}})}\n\n"

                except Exception as e:
                    print(f"[IronGate] Provider stream error: {e}")
                    yield f"data: {_json.dumps({'type': 'error', 'text': f'[Provider Error] {e}'})}\n\n"

            else:
                # Local .gguf model via VoxAPI
                try:
                    if model:
                        ai_bridge.set_model(model)
                    for event in ai_bridge.stream_the_brain(
                        message, history=history, system_prompt=system_prompt
                    ):
                        yield f"data: {_json.dumps(event)}\n\n"
                except Exception as e:
                    print(f"[IronGate] Local stream error: {e}")
                    yield f"data: {_json.dumps({'type': 'error', 'text': f'[Local Error] {e}'})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

# =============================================================================
# STARTUP / CLEANUP
# =============================================================================

def _cleanup_ngrok():
    """Kill any existing ngrok tunnels and processes."""
    try:
        ngrok.kill()
    except Exception:
        pass
    # Also kill any orphaned ngrok processes on Windows
    if os.name == 'nt':
        try:
            import subprocess
            subprocess.run(
                ['powershell.exe', '-Command',
                 'Get-Process ngrok -ErrorAction SilentlyContinue | Stop-Process -Force'],
                capture_output=True, timeout=5
            )
        except Exception:
            pass

def _free_port(port):
    """Kill any process using the specified port (Windows)."""
    if os.name == 'nt':
        try:
            import subprocess
            # Use PowerShell to find and kill process on port
            ps_cmd = (
                f"$conn = Get-NetTCPConnection -LocalPort {port} -State Listen "
                f"-ErrorAction SilentlyContinue; "
                f"if ($conn) {{ $conn | ForEach-Object {{ "
                f"if ($_.OwningProcess -ne {os.getpid()}) {{ "
                f"Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue "
                f"}} }} }}"
            )
            result = subprocess.run(
                ['powershell.exe', '-Command', ps_cmd],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                # Check if port is actually free now
                check = subprocess.run(
                    ['powershell.exe', '-Command',
                     f'Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue'],
                    capture_output=True, text=True, timeout=5
                )
                if not check.stdout.strip():
                    print(f"{Fore.GREEN}[CLEANUP] Port {port} freed")
                else:
                    print(f"{Fore.YELLOW}[CLEANUP] Port {port} still in use")
        except Exception as e:
            print(f"{Fore.YELLOW}[CLEANUP] Could not free port {port}: {e}")

def _graceful_shutdown():
    """Clean shutdown: close tunnels, terminate cloud pods, free resources."""
    print(f"\n{Fore.YELLOW}Shutting down IronGate...")
    try:
        ai_bridge.terminate_cloud_pod()
    except Exception:
        pass
    try:
        _cleanup_ngrok()
    except Exception:
        pass
    print("IronGate Shutdown")

def force_shutdown(signum, frame):
    _graceful_shutdown()
    os._exit(0)

def main():
    import signal
    signal.signal(signal.SIGINT, force_shutdown)
    signal.signal(signal.SIGTERM, force_shutdown)

    os.system("cls" if os.name == "nt" else "clear")
    print(f"{Fore.GREEN}{Style.BRIGHT}=== IRON TUNNEL HOST v{VERSION} ===")

    # Load Configs
    load_config()
    load_database()
    load_host_identity()

    # --- Pre-flight cleanup: kill stale ngrok and free port ---
    print(f"{Fore.WHITE}Cleaning up stale processes...", end=" ", flush=True)
    _cleanup_ngrok()
    time.sleep(1)
    _free_port(8000)
    time.sleep(0.5)
    print(f"{Fore.GREEN}Done")

    # Check Ngrok
    token = host_config.get("ngrok_token")
    if not token:
        print(f"{Fore.YELLOW}Ngrok Token invalid/missing!")
        token = input("Enter Ngrok Token: ").strip()
        host_config["ngrok_token"] = token
        save_config()

    ngrok.set_auth_token(token)

    # Start Tunnel (custom domain or random)
    custom_domain = host_config.get("ngrok_domain", "")
    print(f"{Fore.WHITE}Opening tunnel...", end=" ", flush=True)
    url = None
    for attempt in range(3):
        try:
            if custom_domain:
                tunnel = ngrok.connect(8000, domain=custom_domain)
            else:
                tunnel = ngrok.connect(8000)
            url = tunnel.public_url
            # Ensure https
            if url.startswith("http://"):
                url = "https://" + url[7:]
            print(f"{Fore.GREEN}OK! -> {url}")
            break
        except Exception as e:
            if attempt < 2:
                print(f"{Fore.YELLOW}Attempt {attempt+1} failed, retrying...")
                _cleanup_ngrok()
                time.sleep(2)
                ngrok.set_auth_token(token)
            else:
                print(f"{Fore.RED}Failed after 3 attempts: {e}")

    if url:
        print(f"{Fore.MAGENTA}Magic Host Link: {Fore.WHITE}{url}/?key={host_identity['secret_key']}")
        host_config["tunnel_url"] = url
        save_config()

        # Show public mode status
        if host_config.get("public_mode", False):
            print(f"{Fore.GREEN}{Style.BRIGHT}PUBLIC MODE: ON")
            print(f"{Fore.WHITE}  Public URL: {Fore.CYAN}{url}")
            print(f"{Fore.WHITE}  Anyone with this link can access the web UI")
        else:
            print(f"{Fore.YELLOW}Public Mode: OFF  (Use 'public' command to enable)")
    else:
        print(f"{Fore.YELLOW}[WARN] Running without tunnel (local only: http://localhost:8000)")
        url = "http://localhost:8000"
        host_config["tunnel_url"] = url

    # Start Admin Console
    db_callback = lambda cid, data: (client_db["clients"].update({cid: data}), save_database()) # Helper to update DB
    gen_callback = generate_client_package
    start_admin_console(url, db_callback, gen_callback)

    # Start Server
    print(f"{Fore.WHITE}Server starting on port 8000...")
    print("IronGate Started")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    except Exception as e:
         print(f"Server Error: {e}")
    finally:
        _graceful_shutdown()
        os._exit(0)

if __name__ == "__main__":
    main()