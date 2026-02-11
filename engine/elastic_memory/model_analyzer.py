import os
import struct
from typing import Dict, Any

class ModelAnalyzer:
    """
    Analyzes GGUF files to extract architecture metadata.
    """
    
    @staticmethod
    def analyze(model_path: str) -> Dict[str, Any]:
        """
        Reads GGUF header to get n_layers, d_model, and architecture.
        """
        info = {
            "n_layers": 32, # Defaults
            "d_model": 4096,
            "param_count_b": 7.0,
            "file_size_bytes": 0,
            "arch": "llama"
        }
        
        if not os.path.exists(model_path):
            return info
            
        info["file_size_bytes"] = os.path.getsize(model_path)
        
        # Rough param estimate from file size (assuming Q4 ~5.5 bits/param)
        # file_bytes * 8 bits / 5.5 = params
        est_params = (info["file_size_bytes"] * 8) / 5.5
        info["param_count_b"] = round(est_params / 1e9, 1)

        try:
            from gguf import GGUFReader
            reader = GGUFReader(model_path)
            
            # Find architecture
            arch = "llama"
            for field in reader.fields.values():
                if field.name == "general.architecture":
                    arch = str(field.parts[field.data[0]])  # e.g., 'llama', 'mistral'
                    info["arch"] = arch
                    break
            
            # Find specific keys
            # Ex: llama.block_count, llama.embedding_length
            cnt_key = f"{arch}.block_count"
            dim_key = f"{arch}.embedding_length"
            
            for field in reader.fields.values():
                if field.name == cnt_key:
                    info["n_layers"] = int(field.parts[field.data[0]]) or int(field.data[0]) 
                elif field.name == dim_key:
                    info["d_model"] = int(field.parts[field.data[0]]) or int(field.data[0])

        except Exception as e:
            print(f"[ModelAnalyzer] Warning: Could not read GGUF metadata: {e}")
            
        return info

    @staticmethod
    def estimate_kv_bytes_per_token(n_layers: int, d_model: int, cache_bit=16) -> int:
        """
        Estimate RAM per token for KV cache.
        Formula: 2 (K+V) * n_layers * d_model * bits/8
        """
        bytes_per_val = cache_bit / 8
        return int(2 * n_layers * d_model * bytes_per_val)
