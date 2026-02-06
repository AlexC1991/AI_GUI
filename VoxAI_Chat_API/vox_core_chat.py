import sys
import os
import psutil
import ctypes
import time

# ==========================================
# 0. SYSTEM PREP & ZLUDA LOCATOR
# ==========================================
try:
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)
except: pass

# Get current and parent directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# ==========================================
# 1. SETUP ENVIRONMENT *BEFORE* IMPORTING LLAMA
# ==========================================
# Add paths to environment
os.environ["PATH"] += os.pathsep + current_dir + os.pathsep + parent_dir

if hasattr(os, 'add_dll_directory'): 
    try:
        os.add_dll_directory(current_dir)
        os.add_dll_directory(parent_dir)
    except: pass

# Locate Custom Backend (llama.dll)
dll_name = "llama.dll"
backend_dir = None
possible_dirs = [current_dir, parent_dir]

for d in possible_dirs:
    if os.path.exists(os.path.join(d, dll_name)):
        backend_dir = d
        print(f"[VOX CORE] Found Engine at: {d}")
        break

if backend_dir:
    # CRITICAL: Set this BEFORE importing llama_cpp
    os.environ["GGML_BACKEND_SEARCH_PATH"] = backend_dir
    os.environ["LLAMA_CPP_LIB"] = os.path.join(backend_dir, dll_name)
    print(f"[VOX CORE] Set LLAMA_CPP_LIB = {os.environ['LLAMA_CPP_LIB']}")
    
    # Pre-load ggml if possible
    try:
        ggml_path = os.path.join(backend_dir, "ggml.dll")
        if os.path.exists(ggml_path):
             ggml = ctypes.CDLL(ggml_path)
             if hasattr(ggml, 'ggml_backend_load_all'):
                 ggml.ggml_backend_load_all()
                 print("[VOX CORE] Pre-loaded ggml backends.")
    except Exception as e:
        print(f"[VOX CORE] Pre-load warning: {e}")

else:
    print("[VOX CORE] WARNING: Could not find custom llama.dll. Using default.")

# ==========================================
# 2. NOW IT IS SAFE TO IMPORT LLAMA
# ==========================================
try:
    import llama_cpp
    from llama_cpp import Llama
    
    # Try to re-trigger backend load via module if available
    try:
        if hasattr(llama_cpp, "ggml_backend_load_all"):
            llama_cpp.ggml_backend_load_all()
        elif hasattr(llama_cpp, "llama_cpp") and hasattr(llama_cpp.llama_cpp, "ggml_backend_load_all"):
            llama_cpp.llama_cpp.ggml_backend_load_all()
    except: pass
    
except ImportError:
    sys.exit("[CRITICAL] Could not import llama_cpp. Is virtualenv active?")

# ==========================================
# 3. CALL THE HANDSHAKE
# ==========================================
print("[VOX CORE] Initializing...")
print("[VOX CORE] Requesting Hardware Handshake...")

try:
    import machine_engine_handshake
    detected_mode, phys_cores, cfg = machine_engine_handshake.get_hardware_config()
    print(f"[VOX CORE] Handshake: {detected_mode}")
except ImportError:
    sys.exit("\n[CRITICAL] Missing 'machine_engine_handshake.py'. Cannot detect hardware.")

# Apply Runtime Config
os.environ["GGML_VK_FORCE_BUSY_WAIT"] = cfg["busy_wait"]
os.environ["GGML_NUMA"] = "0"

# ==========================================
# 4. MODEL SELECTOR
# ==========================================
MODELS_DIR = "./models"
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")]

print("\n============================================")
print(f" VOX-AI UNIVERSAL ENGINE | STATUS: ONLINE")
print("============================================")
print(f" [VOX CORE] Mode:   {detected_mode}")
print(f" [VOX CORE] Config: {cfg['n_gpu_layers']} Layers | {cfg['n_threads']} Threads")
print("============================================")

for idx, model_name in enumerate(model_files):
    print(f" [{idx + 1}] {model_name}")

if not model_files:
    print("[ERROR] No models found in ./models")
    input("Press Enter to exit...")
    sys.exit(1)

while True:
    try:
        choice = int(input("\nSelect a model: "))
        if 1 <= choice <= len(model_files):
            selected_model = model_files[choice - 1]
            break
    except ValueError: pass

# ==========================================
# 5. INITIALIZATION
# ==========================================
print(f"\n[VOX CORE] Booting Engine...")

try:
    llm = Llama(
        model_path=os.path.join(MODELS_DIR, selected_model),
        n_ctx=4096, 
        n_gpu_layers=cfg['n_gpu_layers'],
        n_threads=cfg['n_threads'],
        n_threads_batch=cfg['n_threads_batch'],
        n_batch=cfg['n_batch'],
        flash_attn=cfg['flash_attn'],
        use_mlock=cfg['use_mlock'],
        cache_type_k=cfg['cache_type_k'],
        cache_type_v=cfg['cache_type_v'],
        verbose=True
    )

    print("[VOX CORE] Engine Loaded. Performing Warmup...")
    llm.create_chat_completion(messages=[{"role": "user", "content": "ready"}], max_tokens=1)
    print("[VOX CORE] Warmup Complete. Listening for input.")
    
except Exception as e:
    print(f"\n[CRITICAL ERROR] Engine Failed: {e}")
    print("[TIP] Ensure ZLUDA files (nvcuda.dll etc) are in the parent AI_GUI folder.")
    sys.exit(1)

# ==========================================
# 6. CHAT LOOP
# ==========================================
print("\n" + "="*40)
print(" VOX-AI READY")
print("="*40)
history = [{"role": "system", "content": "You are a helpful assistant."}]

while True:
    user_in = input("USER: ")
    if user_in.lower() in ["exit", "quit"]: break
    history.append({"role": "user", "content": user_in})
    print("VOX: ", end="", flush=True)
    
    token_count = 0
    start_time = time.time()
    full_resp = ""
    
    try:
        stream = llm.create_chat_completion(
            messages=history, stream=True, max_tokens=2000,
            temperature=0.7,
            repeat_penalty=1.1,
            top_k=40
        )
        
        first_token = True
        for chunk in stream:
            if "content" in chunk["choices"][0]["delta"]:
                if first_token: first_token = False
                
                tok = chunk["choices"][0]["delta"]["content"]
                print(tok, end="", flush=True)
                full_resp += tok
                token_count += 1
                
        total_time = time.time() - start_time
        tps = token_count / total_time if total_time > 0 else 0
        
        print(f"\n\n[STATS] {token_count} tokens in {total_time:.2f}s | Speed: {tps:.2f} t/s")
        print("-" * 40)
        
        history.append({"role": "assistant", "content": full_resp})
        
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        break