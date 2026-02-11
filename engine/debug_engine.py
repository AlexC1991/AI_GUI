import sys
import os
import time
import psutil
# NOTE: We are NOT importing ctypes or setting LLAMA_CPP_LIB. 
# We are letting the library work exactly as designed.
from llama_cpp import Llama

# ==========================================
# 0. SYSTEM PREP
# ==========================================
try:
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)
except: pass

root_path = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. MODEL DETECTION
# ==========================================
MODELS_DIR = os.path.join(root_path, "models")
if not os.path.exists(MODELS_DIR):
    sys.exit(f"\n[ERROR] Models folder not found at {MODELS_DIR}")

model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")]
if not model_files:
    sys.exit("\n[ERROR] No .gguf files found in ./models")

print("\nAVAILABLE MODELS:")
for i, f in enumerate(model_files):
    print(f" [{i+1}] {f}")

while True:
    try:
        idx = int(input("\nSelect model for benchmark (1-N): "))
        if 1 <= idx <= len(model_files):
            model_path = os.path.join(MODELS_DIR, model_files[idx-1])
            break
    except ValueError: pass

PROMPT = "Explain how a computer processor works in 100 words."

# ==========================================
# 2. THE CLEAN SHOOTOUT
# ==========================================
# We are testing if the "Default" behavior (which Ollama likely mimics) is better
configs = [
    {
        "name": "Standard CPU (Default Load)",
        "gpu_layers": 0, 
        "threads": 8,
        "batch": 512
    },
    {
        "name": "Heavy CPU (High Batch)",
        "gpu_layers": 0,
        "threads": 8,
        "batch": 1024 # Ollama uses larger batches
    },
    {
        "name": "Hybrid (12 Layers)",
        "gpu_layers": 12, # Try offloading a little bit
        "threads": 8,
        "batch": 512
    }
]

print("\n" + "="*50)
print(" VOX-AI: SANITY CHECK (NO DLL OVERRIDES)")
print("="*50)

for cfg in configs:
    print(f"\n--- TESTING: {cfg['name']} ---")
    
    try:
        # PURE STANDARD LOAD
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=cfg['gpu_layers'],
            n_threads=cfg['threads'],
            n_batch=cfg['batch'],
            verbose=False
        )
        
        # Warmup
        llm.create_chat_completion(messages=[{"role":"user","content":"."}], max_tokens=1)
        
        # Sprint
        start = time.time()
        token_count = 0
        stream = llm.create_chat_completion(
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=200,
            stream=True
        )
        
        print("    >> Generating...", end="", flush=True)
        for chunk in stream:
            if "content" in chunk["choices"][0]["delta"]:
                token_count += 1
                if token_count % 20 == 0: print(".", end="", flush=True)
                
        duration = time.time() - start
        tps = token_count / duration
        print(f"\n    >> SPEED: {tps:.2f} t/s")
        
        del llm
        
    except Exception as e:
        print(f"\n    >> FAILED: {e}")

print("\n" + "="*50)
print(" TEST COMPLETE")
print("="*50)
input("Press Enter to exit...")