import os
import psutil
import subprocess
import sys

def get_hardware_config():
    """
    VOX-AI Hardware Handshake
    - Detects CPU Topology
    - Detects GPU (NVIDIA vs AMD)
    - Applies RX 6600 Specific Tuning (Vulkan Safe-Mode)
    """
    print("\n[HANDSHAKE] --- PROTOCOL STARTED ---")
    
    # =========================================================
    # 1. CPU TOPOLOGY PROBE
    # =========================================================
    print("[HANDSHAKE] Probing CPU Topology...")
    try:
        # Get physical cores to avoid hyper-threading overhead issues
        physical_cores = psutil.cpu_count(logical=False) or 4
        logical_cores = psutil.cpu_count(logical=True) or 8
        print(f"[HANDSHAKE] Detected {physical_cores} Physical Cores / {logical_cores} Logical Threads.")
        
        # Thread Logic: Reserve 1 core for OS/GPU driver overhead
        optimal_threads = max(1, physical_cores - 1)
        optimal_batch_threads = max(2, physical_cores // 2)
        
    except Exception as e:
        print(f"[HANDSHAKE] CPU Probe Warning: {e}")
        physical_cores = 4
        optimal_threads = 3
        optimal_batch_threads = 2

    # =========================================================
    # 2. GPU DETECTION (DEEP SCAN)
    # =========================================================
    gpu_type = "INTEGRATED"
    gpu_name = "Unknown"
    
    try:
        # Check for NVIDIA (Fastest check)
        if os.system("nvidia-smi >nul 2>&1") == 0:
            gpu_type = "DISCRETE_NVIDIA"
        else:
            # Check for AMD via PowerShell to avoid WMIC deprecation
            cmd = 'powershell "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"'
            try:
                output = subprocess.check_output(cmd, shell=True).decode().upper().strip()
                # Clean up output to handle multi-line returns (if multiple GPUs exist)
                lines = output.split('\n')
                for line in lines:
                    if "RX " in line or "RADEON" in line:
                        gpu_name = line.strip()
                        gpu_type = "AMD_RADEON"
                        print(f"[HANDSHAKE] GPU Found: {gpu_name}")
                        break
            except:
                pass

        # =========================================================
        # 3. SPECIFIC HARDWARE OVERRIDES
        # =========================================================
        
        # RX 6600 / 6600 XT Detection
        # Strategy: Use Vulkan for Chat (Stable/Fast) and let OS handle ZLUDA for Images
        if "6600" in gpu_name or "7600" in gpu_name:
             print(f"[HANDSHAKE] 8GB VRAM Limit Detected ({gpu_name}).")
             print("[HANDSHAKE] applying VULKAN_SAFE_MODE (35 Layers) to prevent crashes.")
             gpu_type = "AMD_8GB_SAFE"

    except Exception as e:
        print(f"[HANDSHAKE] GPU Probe Failed: {e}")

    # =========================================================
    # 4. CONFIGURATION MATRIX
    # =========================================================
    mode = "CPU (Fallback)"
    layers = 0
    use_flash_attn = False
    
    if gpu_type == "DISCRETE_NVIDIA":
        mode = "NVIDIA (CUDA Unleashed)"
        layers = -1 # All layers
        use_flash_attn = True
        
    elif gpu_type == "AMD_8GB_SAFE":
        # THE FIX: Run 35 Layers on Vulkan.
        # This leaves ~1.5GB VRAM free, preventing crashes when Image Gen starts.
        mode = "AMD (Vulkan - 35 Layer Limit)"
        layers = 35 
        use_flash_attn = False 
        
    elif gpu_type == "AMD_RADEON":
        # Generic AMD (Higher VRAM cards like 6800/7900)
        mode = "AMD (Vulkan - Full)"
        layers = -1
        use_flash_attn = False
        
    else:
        # Integrated or Unknown
        mode = "CPU/iGPU Optimized"
        layers = 0 
    
    print(f"[HANDSHAKE] Detected GPU Class: {gpu_type}")
    
    # Final Config Dictionary
    config = {
        "n_gpu_layers": layers,
        "n_threads": optimal_threads, 
        "n_threads_batch": optimal_batch_threads, 
        "n_batch": 512,
        "flash_attn": use_flash_attn,
        "use_mlock": True,     # Lock memory to prevent swapping
        "busy_wait": "1",      # Vulkan optimization
        "cache_type_k": "f16", # Save VRAM
        "cache_type_v": "f16"  # Save VRAM
    }

    print(f"[HANDSHAKE] Final Mode Decision: {mode}")
    print("[HANDSHAKE] --- PROTOCOL COMPLETE ---\n")
    return mode, physical_cores, config

if __name__ == "__main__":
    get_hardware_config()