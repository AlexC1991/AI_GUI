import os
import shutil
import sys

# Ensure library is found
try:
    from gguf import GGUFReader, GGUFWriter
except ImportError:
    print("Installing GGUF library...")
    os.system(f"{sys.executable} -m pip install gguf")
    from gguf import GGUFReader, GGUFWriter

from llama_cpp import Llama

# --- SETTINGS ---
MODELS_DIR = "./models"
BACKEND_DLL = "./llama.dll" # Path to your custom optimized DLL
TARGET_ARCH = "llama"
INCOMPATIBLE_ARCHS = ["mistral3", "ministral"]

def patch_model_if_needed(model_path):
    """Checks and patches GGUF architecture tags without permanent changes to originals."""
    try:
        reader = GGUFReader(model_path)
        current_arch = None
        
        # Identify current architecture
        for field in reader.fields.values():
            if field.name == "general.architecture":
                current_arch = str(field.parts[field.data[0]])
                break

        if current_arch in INCOMPATIBLE_ARCHS:
            patched_name = f"VOX_READY_{os.path.basename(model_path)}"
            patched_path = os.path.join(os.path.dirname(model_path), patched_name)
            
            if os.path.exists(patched_path):
                print(f"[PRE-FLIGHT] {patched_name} already exists.")
                return patched_path

            print(f"[PRE-FLIGHT] Detected '{current_arch}'. Patching to '{TARGET_ARCH}'...")
            
            # Create a patched copy
            shutil.copy2(model_path, patched_path)
            writer = GGUFWriter(patched_path, arch=TARGET_ARCH)
            
            # Clone all metadata except architecture
            for field in reader.fields.values():
                if field.name != "general.architecture":
                    writer.add_key_value(field.name, field.parts, field.types[0])
            
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            print(f"[SUCCESS] Patched model created: {patched_name}")
            return patched_path
            
    except Exception as e:
        print(f"[ERROR] Could not read metadata: {e}")
    return model_path

def test_load(model_path):
    """Verifies the model can actually initialize on your custom engine."""
    print(f"[TEST] Attempting dry-run load of {os.path.basename(model_path)}...")
    try:
        # Load with minimal context and 0 layers just to check architecture compatibility
        test_llm = Llama(model_path=model_path, n_ctx=16, n_gpu_layers=0, verbose=False)
        print(f"[OK] Model is compatible with your custom engine.")
        del test_llm
        return True
    except Exception as e:
        print(f"[FAIL] Engine rejected model: {e}")
        return False

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("=== VOX-AI PRE-FLIGHT COMPATIBILITY TOOL ===\n")
    
    if not os.path.exists(MODELS_DIR):
        print(f"Models directory not found at: {MODELS_DIR}")
        sys.exit(1)

    # List models
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")]
    
    if not files:
        print("No .gguf models found in ./models")
        sys.exit(1)

    for idx, f in enumerate(files):
        print(f"[{idx}] {f}")
    
    try:
        choice_input = input("\nSelect model to verify/patch: ")
        if not choice_input.strip():
            sys.exit(0)
            
        choice = int(choice_input)
        if choice < 0 or choice >= len(files):
             raise ValueError
    except ValueError:
        print("Invalid selection")
        sys.exit(1)

    target_file = os.path.join(MODELS_DIR, files[choice])
    
    # 1. Patch if it's a Mistral 3 model
    ready_path = patch_model_if_needed(target_file)
    
    # 2. Test if it runs on your DLL
    if test_load(ready_path):
        print("\n[READY] You can now select this model in vox_core_chat.py")
    else:
        print("\n[WARNING] This model still fails. Your DLL might need a specific kernel update.")
