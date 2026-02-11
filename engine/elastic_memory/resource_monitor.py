import psutil
import logging
from typing import Dict

class ResourceMonitor:
    """
    Monitors System VRAM/RAM to calculate safe context limits.
    """
    def __init__(self, model_params_b: float, gpu_name: str, vram_total_mb: int, vram_free_mb: int):
        self.model_params_b = model_params_b
        self.gpu_name = gpu_name
        self.vram_total_mb = vram_total_mb
        self.vram_free_mb = vram_free_mb
        self.ram_total_mb = 0
        self.ram_free_mb = 0
        self._update_system_ram()

    def _update_system_ram(self):
        try:
            mem = psutil.virtual_memory()
            self.ram_total_mb = mem.total // (1024 * 1024)
            self.ram_free_mb = mem.available // (1024 * 1024)
        except:
             self.ram_free_mb = 4096

    def compute_safe_n_ctx(self, model_file_size_bytes: int, kv_bytes_per_token: int) -> int:
        """
        Calculate the maximum safe n_ctx based on available memory.
        """
        # 1. Calculate Model Overhead
        # Rule of Thumb: Model takes file_size + 10-20% runtime overhead
        model_mem_mb = (model_file_size_bytes / (1024 * 1024)) * 1.2
        
        # 2. Determine Memory Ceiling
        # If GPU available, use VRAM. If CPU fallback, use System RAM.
        if self.vram_total_mb > 0:
            # VRAM Mode
            # Ceiling depends on model size (larger models need more clearance)
            ceiling_pct = 0.50
            if self.model_params_b >= 12: ceiling_pct = 0.85
            elif self.model_params_b >= 7: ceiling_pct = 0.70
            
            safe_mem_mb = self.vram_total_mb * ceiling_pct
            available_for_kv = safe_mem_mb - model_mem_mb
            
            # If model doesn't fit in safe VRAM, it might be partially offloaded
            # In that case, we rely on GGUF's mmap, so VRAM isn't the only limit.
            # But for "fast" context, we want it in VRAM.
            if available_for_kv < 0:
                available_for_kv = 512 # Minimum safeguard
        else:
            # RAM Mode (CPU)
            # Leave 4GB for OS + App
            safe_mem_mb = self.ram_free_mb - 4096
            available_for_kv = safe_mem_mb - model_mem_mb
            if available_for_kv < 0: available_for_kv = 1024

        # 3. Calculate Token Budget
        kv_mb_per_token = kv_bytes_per_token / (1024 * 1024)
        if kv_mb_per_token <= 0: kv_mb_per_token = 0.0005 # Fallback (typical 8B f16)
        
        max_tokens = int(available_for_kv / kv_mb_per_token)
        
        # 4. Clamp results
        # Min: 2048 (usable)
        # Max: 32768 (reasonable usage)
        final_n_ctx = max(2048, min(max_tokens, 32768))
        
        return final_n_ctx

    def get_vram_ceiling_pct(self):
        if self.model_params_b >= 12: return 0.85
        if self.model_params_b >= 7: return 0.70
        return 0.50
