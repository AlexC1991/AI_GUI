import sys
import os
import time
import requests
import json
from config import API_KEY, POD_ID, MODEL_MAP

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

def launch_chat():
    print(f"{CYAN}Starting VoxAI Initialization...{RESET}\n")

    # [SECTION] Environment Selection
    print(f"{CYAN}")
    print(r"""
██╗   ██╗ ██████╗ ██╗  ██╗       █████╗ ██╗
██║   ██║██╔═══██╗╚██╗██╔╝      ██╔══██╗██║
██║   ██║██║   ██║ ╚███╔╝ █████╗███████║██║
╚██╗ ██╔╝██║   ██║ ██╔██╗ ╚════╝██╔══██║██║
 ╚████╔╝ ╚██████╔╝██╔╝ ██╗      ██║  ██║██║
  ╚═══╝   ╚═════╝ ╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝
    """)
    print(f"     VOX-AI UNIVERSAL ENGINE | {YELLOW}BOOT{CYAN}")
    print(f"============================================{RESET}")
    print(" [1] LOCAL (GPU/CPU) | [2] CLOUD (RunPod)")

    choice = input("Select Environment: ").strip()
    use_cloud = choice == "2"

    # [SECTION] Model Selection
    # MODEL_MAP format: {hf_id: display_name} — keys ARE the cloud IDs
    print("\n--- Available Brains ---")
    model_entries = list(MODEL_MAP.items())  # [(hf_id, display_name), ...]
    for i, (hf_id, display_name) in enumerate(model_entries):
        print(f" [{i+1}] {display_name}  ({hf_id})")

    try:
        m_choice = int(input("\nSelect Model: ")) - 1
        selected_hf_id, selected_display = model_entries[m_choice]
    except:
        print(f"{RED}[ERROR] Invalid selection.{RESET}")
        sys.exit(1)

    # [SECTION] Engine Logic
    llm = None
    cloud_driver = None

    if use_cloud:
        try:
            from runpod_interface import RunPodDriver
            cloud_driver = RunPodDriver(API_KEY, POD_ID)

            # The HF ID is already the key from MODEL_MAP — no phone book needed
            print(f"\n[SYSTEM] Target Cloud Model: {selected_hf_id}")

            # Try to switch/boot the cloud pod
            if cloud_driver.switch_model(selected_hf_id):
                print(f"\n[SYSTEM] ☁️  {GREEN}Cloud Link Established.{RESET}")
            else:
                print(f"\n[FALLBACK] ⚠️ {YELLOW}Cloud failed. Engaging Local Hardware...{RESET}")
                use_cloud = False
        except Exception as e:
            print(f"[ERROR] Cloud init failed: {e}")
            use_cloud = False

    # [SECTION] Local Fallback
    if not use_cloud:
        # For local mode, the HF ID won't resolve to a local file
        # We need a local model path — check if there's a .gguf in models/
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")

        # Try to find a local .gguf that matches the display name
        local_file = None
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith(".gguf"):
                    if selected_display.lower().replace(" ", "") in f.lower().replace("-", "").replace("_", ""):
                        local_file = f
                        break
            # If no match, let user pick from available local files
            if not local_file:
                gguf_files = [f for f in os.listdir(models_dir) if f.endswith(".gguf")]
                if gguf_files:
                    print(f"\n{YELLOW}[LOCAL] No matching GGUF for '{selected_display}'. Available local models:{RESET}")
                    for i, f in enumerate(gguf_files):
                        print(f" [{i+1}] {f}")
                    try:
                        l_choice = int(input("Select local model: ")) - 1
                        local_file = gguf_files[l_choice]
                    except:
                        print(f"{RED}[ERROR] Invalid selection.{RESET}")
                        sys.exit(1)
                else:
                    print(f"{RED}[ERROR] No .gguf files found in {models_dir}{RESET}")
                    sys.exit(1)

        model_path = os.path.join(models_dir, local_file).replace("\\", "/")
        print(f"\n[LOCAL] 🛡️ {CYAN}Loading GGUF: {model_path}...{RESET}")

        if not os.path.exists(model_path):
            print(f"{RED}[ERROR] ❌ File not found: {model_path}{RESET}")
            sys.exit(1)

        try:
            # [HANDSHAKE] Import and Run Hardware Check
            import machine_engine_handshake
            import ctypes
            mode, phys_cores, cfg = machine_engine_handshake.get_hardware_config()

            root_path = os.path.abspath(".")
            os.environ["GGML_VK_FORCE_BUSY_WAIT"] = cfg["busy_wait"]
            os.environ["GGML_BACKEND_SEARCH_PATH"] = root_path

            try:
                ggml = ctypes.CDLL(os.path.join(root_path, "ggml.dll"))
                if hasattr(ggml, 'ggml_backend_load_all'):
                    ggml.ggml_backend_load_all()
                print(f"[LOCAL] 🟢 Backend drivers loaded manually.")
            except Exception as e:
                print(f"[LOCAL] ⚠️ Backend load warning: {e}")

            print(f"[LOCAL] 🛠️ Initializing Llama Engine ({mode})...")
            from llama_cpp import Llama
            llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                verbose=True,
                n_gpu_layers=cfg['n_gpu_layers'],
                n_threads=cfg['n_threads'],
                n_threads_batch=cfg['n_threads_batch'],
                n_batch=cfg['n_batch'],
                flash_attn=cfg['flash_attn'],
                use_mlock=cfg['use_mlock'],
                cache_type_k=cfg['cache_type_k'],
                cache_type_v=cfg['cache_type_v']
            )
            print(f"[LOCAL] {GREEN}✅ Engine Online.{RESET}")
        except Exception as e:
            print(f"{RED}[ERROR] Failed to load local model: {e}{RESET}")
            sys.exit(1)

    # [SECTION] The Chat Loop
    print("\n" + "="*40)
    print(f"VoxAI Online. Model: {selected_display}")
    print("Commands: 'exit', 'swap'")
    print("="*40 + "\n")

    messages = []

    while True:
        try:
            user_input = input(f"{CYAN}You:{RESET} ").strip()
            if user_input.lower() == "exit":
                if use_cloud and cloud_driver:
                    print(f"{YELLOW}[SYSTEM] Shutting down Cloud Resources...{RESET}")
                    cloud_driver.terminate_pod()
                else:
                    print(f"{YELLOW}[LOCAL] Shutting down Engine...{RESET}")
                break

            if user_input.lower() == "swap":
                print(f"{YELLOW}[SYSTEM] Triggering Model Swap...{RESET}")

                print("\n--- Available Brains ---")
                for i, (hf_id, display_name) in enumerate(model_entries):
                    print(f" [{i+1}] {display_name}  ({hf_id})")

                try:
                    selection = input("\nSelect Model (0 to cancel): ").strip()
                    if selection == "0": continue

                    m_choice = int(selection) - 1
                    if 0 <= m_choice < len(model_entries):
                        new_hf_id, new_display = model_entries[m_choice]

                        # --- CLOUD SWAP ---
                        if use_cloud and cloud_driver:
                            if cloud_driver.switch_model(new_hf_id):
                                selected_hf_id = new_hf_id
                                selected_display = new_display
                                print(f"\n[SYSTEM] ☁️  {GREEN}Swap Complete. Now running: {selected_display}{RESET}")
                                print("="*40 + "\n")
                            else:
                                print(f"{RED}[ERROR] Swap failed. Staying on current model.{RESET}")

                        # --- LOCAL SWAP ---
                        else:
                            # For local, need to find matching .gguf
                            new_local = None
                            for f in os.listdir(models_dir):
                                if f.endswith(".gguf") and new_display.lower().replace(" ", "") in f.lower().replace("-", "").replace("_", ""):
                                    new_local = f
                                    break

                            if not new_local:
                                print(f"{RED}[ERROR] No local GGUF found for {new_display}{RESET}")
                                continue

                            new_path = os.path.join(models_dir, new_local)
                            print(f"\n[LOCAL] 🔄 Unloading current model...")
                            if 'llm' in locals():
                                del llm
                                import gc
                                gc.collect()

                            print(f"[LOCAL] 🛡️ Loading New GGUF: {new_display}...")
                            import machine_engine_handshake
                            _, _, cfg = machine_engine_handshake.get_hardware_config()

                            print(f"[LOCAL] 🛠️ Initializing Llama Engine...")
                            from llama_cpp import Llama
                            llm = Llama(
                                model_path=new_path,
                                n_ctx=4096,
                                verbose=True,
                                n_gpu_layers=cfg['n_gpu_layers'],
                                n_threads=cfg['n_threads'],
                                n_threads_batch=cfg['n_threads_batch'],
                                n_batch=cfg['n_batch'],
                                flash_attn=cfg['flash_attn'],
                                use_mlock=cfg['use_mlock'],
                                cache_type_k=cfg['cache_type_k'],
                                cache_type_v=cfg['cache_type_v']
                            )
                            selected_hf_id = new_hf_id
                            selected_display = new_display
                            print(f"[LOCAL] {GREEN}✅ Swap Complete. Engine Online.{RESET}")
                            print("="*40 + "\n")

                    else:
                        print(f"{RED}[ERROR] Invalid selection.{RESET}")

                except Exception as e:
                    print(f"{RED}[ERROR] Selection error: {e}{RESET}")

                continue

            messages.append({"role": "user", "content": user_input})

            print(f"{GREEN}VoxAI:{RESET} ", end="", flush=True)

            if use_cloud and cloud_driver and cloud_driver.new_pod_id:
                # CLOUD GENERATION — use cloud_driver.port, not hardcoded 8000
                url = f"https://{cloud_driver.new_pod_id}-{cloud_driver.port}.proxy.runpod.net/v1/chat/completions"
                payload = {
                    "model": selected_hf_id,
                    "messages": messages,
                    "max_tokens": 2048,
                    "stream": True
                }
                headers = {"Authorization": f"Bearer {API_KEY}"}

                try:
                    response = requests.post(url, json=payload, headers=headers, stream=True)
                    full_response = ""

                    t0 = time.time()
                    token_count = 0

                    for chunk in response.iter_lines():
                        if chunk:
                            try:
                                decoded = chunk.decode('utf-8')
                                if decoded.startswith("data: "):
                                    decoded = decoded[6:]

                                if decoded.strip() == "[DONE]":
                                    continue

                                j = json.loads(decoded)
                                if 'choices' in j and len(j['choices']) > 0:
                                    delta = j['choices'][0].get('delta', {})
                                    token = delta.get('content', '')
                                    if token:
                                        print(token, end="", flush=True)
                                        full_response += token
                                        token_count += 1
                            except: pass

                    dt = time.time() - t0
                    if token_count > 0 and dt > 0:
                        speed = token_count / dt
                        balance = cloud_driver.get_balance() or 0.0
                        cost = cloud_driver.pod_cost or 0.0
                        print(f"\n{YELLOW}({speed:.2f} t/s) | Balance: ${float(balance):.2f} | Cost: ${float(cost):.3f}/hr{RESET}")
                    else:
                         print()

                    messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    print(f"\n{RED}[Cloud Error] {e}{RESET}")

            elif llm:
                # LOCAL GENERATION
                try:
                    stream = llm.create_chat_completion(
                        messages=messages,
                        max_tokens=2048,
                        stream=True
                    )
                    full_response = ""

                    t0 = time.time()
                    token_count = 0

                    for chunk in stream:
                        if 'content' in chunk['choices'][0]['delta']:
                            token = chunk['choices'][0]['delta']['content']
                            print(token, end="", flush=True)
                            full_response += token
                            token_count += 1

                    dt = time.time() - t0
                    if token_count > 0 and dt > 0:
                         print(f"\n{YELLOW}({token_count / dt:.2f} t/s){RESET}")
                    else:
                         print()

                    messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    print(f"\n{RED}[Local Error] {e}{RESET}")

        except KeyboardInterrupt:
            print("\n[SYSTEM] Interrupted. Exiting...")
            if use_cloud and cloud_driver:
                print(f"{YELLOW}[SYSTEM] Shutting down Cloud Resources...{RESET}")
                try:
                    cloud_driver.terminate_pod()
                except Exception as e:
                    print(f"{RED}[ERROR] Failed to terminate pod on exit: {e}{RESET}")
            break

if __name__ == "__main__":
    launch_chat()
