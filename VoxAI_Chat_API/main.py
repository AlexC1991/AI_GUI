import sys
import subprocess
import os
import urllib.request
import json
import zipfile
import importlib.util

# ==========================================
# 1. CONFIGURATION (GAME STYLE)
# ==========================================
MODELS_DIR = "./models"
# ENGINE IS NOW ROOT (The same place as this script)
ENGINE_DIR = "." 
TARGET_SCRIPT = "vox_core_chat.py"

# GitHub API endpoints
LATEST_ENGINE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
WRAPPER_COMMIT_API = "https://api.github.com/repos/abetlen/llama-cpp-python/commits/main"

DEFAULT_MODEL_URL = "https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/resolve/main/qwen1_5-0_5b-chat-q4_k_m.gguf?download=true"
DEFAULT_MODEL_NAME = "qwen_0.5b_chat.gguf"
GIT_WRAPPER_URL = "git+https://github.com/abetlen/llama-cpp-python.git"
VERSION_FILE = ".wrapper_version"

def print_header(text):
    print(f"\n[VOX BOOT] --- {text} ---")

def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rDownloading... {percent}%")
    sys.stdout.flush()

# ==========================================
# 2. SYSTEM CLEANER
# ==========================================
def purge_system_dependency():
    print_header("Sanitizing Python Environment")
    loader = importlib.util.find_spec('llama_cpp')
    if not loader: return

    package_dir = os.path.dirname(loader.origin)
    system_dll = os.path.join(package_dir, "llama.dll")
    
    if os.path.exists(system_dll):
        try:
            os.remove(system_dll)
            print(f" [CLEAN] Removed conflicting system dependency.")
        except PermissionError:
            pass # Currently in use, that's fine
    else:
        print(" [OK] Environment is clean.")

# ==========================================
# 3. ROOT ENGINE VERIFICATION
# ==========================================
def verify_root_engine():
    print_header("Verifying Engine (Game-Style)")
    
    # We check for the two VIPs in the root folder
    local_dll = os.path.abspath("llama.dll")
    vulkan_dll = os.path.abspath("ggml-vulkan.dll")
    
    if os.path.exists(local_dll) and os.path.exists(vulkan_dll):
        print(" [OK] Engine files found in root directory.")
        return

    print(" [ERROR] Engine files missing from root!")
    print(" Please run 'repair_engine.py' then 'move_engine.py' if files are missing.")
    # We don't auto-download here to avoid messing up your Game-Style setup
    sys.exit(1)

# ==========================================
# 4. WRAPPER CHECK
# ==========================================
def check_wrapper_updates():
    print_header("Checking Python Wrapper")
    if os.path.exists(VERSION_FILE):
        print(" [OK] Wrapper is version-locked.")
        return

    print(" [INIT] First-time setup. Syncing...")
    try:
        req = urllib.request.Request(WRAPPER_COMMIT_API, headers={'User-Agent': 'VoxAI/1.0'})
        with urllib.request.urlopen(req) as response:
            latest_commit = json.loads(response.read().decode())['sha']
            
        process = subprocess.Popen(
            [sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", GIT_WRAPPER_URL],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in process.stdout:
            sys.stdout.write(f" [DATA] {line}")
            sys.stdout.flush()
        process.wait()
        
        if process.returncode == 0:
            with open(VERSION_FILE, "w") as f:
                f.write(latest_commit)
            print("\n [SUCCESS] Wrapper installed.")
    except Exception as e:
        print(f"\n [WARN] Update failed ({e}). Proceeding offline.")

# ==========================================
# 5. ENVIRONMENT & LAUNCH
# ==========================================
def check_environment():
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    gguf_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")]
    print(f"\n [READY] {len(gguf_files)} models available.")

def launch_chat():
    print_header("Handing over to Core AI Engine")
    subprocess.run([sys.executable, TARGET_SCRIPT])

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    check_wrapper_updates()
    verify_root_engine() # Checks root folder "."
    purge_system_dependency()
    check_environment()
    launch_chat()