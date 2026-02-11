import sys
import os
import time
import ctypes
import psutil
import gc
import statistics
from llama_cpp import Llama

# ==========================================
# 0. SYSTEM PREP (The Standard Boot)
# ==========================================
try:
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)
except: pass

root_path = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + root_path
if hasattr(os, 'add_dll_directory'): os.add_dll_directory(root_path)

# Custom DLL Loading
os.environ["GGML_BACKEND_SEARCH_PATH"] = root_path
os.environ["LLAMA_CPP_LIB"] = os.path.join(root_path, "llama.dll")
os.environ["GGML_NUMA"] = "0"

try:
    ggml = ctypes.CDLL(os.path.join(root_path, "ggml.dll"))
    if hasattr(ggml, 'ggml_backend_load_all'):
        ggml.ggml_backend_load_all()
except: pass

# ==========================================
# 1. SETUP
# ==========================================
MODELS_DIR = os.path.join(root_path, "models")
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")]
if not model_files: sys.exit("[ERROR] No models found.")

print("\nAVAILABLE MODELS:")
for i, f in enumerate(model_files):
    print(f" [{i+1}] {f}")

while True:
    try:
        idx = int(input("\nSelect model (1-N): "))
        if 1 <= idx <= len(model_files):
            model_path = os.path.join(MODELS_DIR, model_files[idx-1])
            break
    except ValueError: pass

# FIXED PROMPT & PARAMS FOR CONSISTENCY
PROMPT = "Explain the theory of relativity in simple terms."
SEED = 42
TEMP = 0.0 # Deterministic output

configs = [
    {
        "name": "HYBRID VULKAN",
        "gpu_layers": 12,
        "busy_wait": "1"
    },
    {
        "name": "PURE CPU",
        "gpu_layers": 0,
        "busy_wait": "0"
    }
]

print("\n" + "="*60)
print(" VOX-AI: SCIENTIFIC BENCHMARK")
print("="*60)
print(f"Cycles: 3 Runs per Config (Averaged)")
print(f"Load:   Fixed Seed {SEED} | Temp {TEMP} (Identical Output)")

for cfg in configs:
    print(f"\n\n>>> TESTING CONFIG: {cfg['name']}")
    
    # Apply Environment
    os.environ["GGML_VK_FORCE_BUSY_WAIT"] = cfg['busy_wait']
    
    try:
        # Initialize
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=cfg['gpu_layers'],
            n_threads=8,
            n_threads_batch=4,
            n_batch=512,
            cache_type_k="f16",
            cache_type_v="f16",
            verbose=False,
            seed=SEED # Lock the seed
        )
        
        # Warmup
        print("    [Warmup] Firing engine...", end="", flush=True)
        llm.create_chat_completion(messages=[{"role":"user","content":"hi"}], max_tokens=1)
        print(" Done.")
        
        run_speeds = []
        
        # THE GAUNTLET (3 RUNS)
        for i in range(3):
            print(f"    [Run {i+1}/3] Generating...", end="", flush=True)
            
            start_time = time.time()
            
            # Non-streaming allows us to get the EXACT output object
            output = llm.create_chat_completion(
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=150,
                temperature=TEMP,
                seed=SEED
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Calculate EXACT tokens used (Prompt + Completion)
            # We care about generation speed, so we count completion tokens
            usage = output['usage']
            gen_tokens = usage['completion_tokens']
            
            tps = gen_tokens / duration
            run_speeds.append(tps)
            print(f" {tps:.2f} t/s ({gen_tokens} tok)")
            
        # Stats
        avg_speed = statistics.mean(run_speeds)
        
        print(f"    -----------------------------")
        print(f"    AVERAGE SPEED: {avg_speed:.2f} t/s")
        
        del llm
        gc.collect()
        
    except Exception as e:
        print(f"\n    CRASH: {e}")

print("\n" + "="*60)
print(" SCIENTIFIC RESULT")
print("="*60)
input("Press Enter to exit...")