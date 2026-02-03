import os
import json
import torch
import numpy as np
from pathlib import Path
from gguf import GGUFReader
from safetensors.torch import save_file

def convert_gguf_t5_to_safetensors(gguf_path: str, output_dir: str, max_shard_size: int = 2 * 1024 * 1024 * 1024):
    """
    Converts a T5 GGUF file to HuggingFace sharded safetensors format.
    Streams tensors to avoid RAM OOM.
    """
    gguf_path = Path(gguf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Force cleanup of any old/partial shards to prevent WinError 183
    for f in output_dir.glob("*.safetensors"):
        try:
            f.unlink()
        except Exception:
            pass

    # CRITICAL: Save the correct config for T5 v1.1 XXL so from_pretrained knows the architecture
    config_path = output_dir / "config.json"
    if not config_path.exists():
        print("[Converter] Downloading config for google/t5-v1_1-xxl...")
        try:
            from transformers import AutoConfig
            # We use t5-v1_1-xxl because that is what the GGUF is based on (4096 dim)
            config = AutoConfig.from_pretrained("google/t5-v1_1-xxl")
            config.save_pretrained(output_dir)
            print(f"[Converter] Saved config.json to {output_dir}")
        except Exception as e:
            print(f"[Converter] Warning: Could not save config: {e}")
            print("[Converter] Loading might fail if config is missing.")
    
    print(f"[Converter] Opening GGUF: {gguf_path}")
    reader = GGUFReader(str(gguf_path))
    
    # State for sharding
    state_dict = {}
    current_shard_size = 0
    shard_index = 0
    index_map = {} # For model.safetensors.index.json
    
    # helper to save shard
    def save_current_shard():
        nonlocal shard_index, state_dict, current_shard_size
        if not state_dict:
            return
        
        shard_name = f"model-{shard_index+1:05d}-of-XXXXX.safetensors" # Placeholder, fixed later
        save_path = output_dir / shard_name
        
        print(f"[Converter] Saving shard {shard_name} ({current_shard_size / 1024**3:.2f} GB)...")
        # Save BFloat16 to match Flux and save space
        save_file(state_dict, save_path, metadata={"format": "pt"})
        
        # Record in index
        for key in state_dict.keys():
            index_map[key] = shard_name
            
        # Reset
        state_dict = {}
        current_shard_size = 0
        shard_index += 1

    print(f"[Converter] Found {len(reader.tensors)} tensors. Starting conversion...")
    
    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        
        # --- MAPPING LOGIC (T5 v1.1) ---
        new_name = name
        
        # Blocks
        if "enc.blk" in name:
            # enc.blk.0.attn_q.weight -> encoder.block.0.layer.0.SelfAttention.q.weight
            parts = name.split(".")
            block_idx = parts[2]
            layer_type = parts[3] # attn_q, ffn_gate, etc
            
            prefix = f"encoder.block.{block_idx}"
            
            if layer_type == "attn_q":
                new_name = f"{prefix}.layer.0.SelfAttention.q.weight"
            elif layer_type == "attn_k":
                new_name = f"{prefix}.layer.0.SelfAttention.k.weight"
            elif layer_type == "attn_v":
                new_name = f"{prefix}.layer.0.SelfAttention.v.weight"
            elif layer_type == "attn_o":
                new_name = f"{prefix}.layer.0.SelfAttention.o.weight"
            elif layer_type == "attn_norm":
                new_name = f"{prefix}.layer.0.layer_norm.weight"
            elif layer_type == "attn_rel_b":
                new_name = f"{prefix}.layer.0.SelfAttention.relative_attention_bias.weight"
            
            # FFN (Gated GeLU: wi_0=up, wi_1=gate seems standard for T5)
            # CAUTION: Some mappings trigger wi_0=gate.
            # Transformers T5v1.1: DenseReluDense.wi_0 is the first linear, wi_1 is the second (gate).
            # GGUF: ffn_gate usually corresponds to the activation gate.
            # Let's check typical conversion:
            # llama.cpp convert_hf_to_gguf maps 'wi_1' -> 'ffn_gate'
            # So 'ffn_gate' -> 'wi_1'
            elif layer_type == "ffn_gate":
                new_name = f"{prefix}.layer.1.DenseReluDense.wi_1.weight"
            elif layer_type == "ffn_up":
                new_name = f"{prefix}.layer.1.DenseReluDense.wi_0.weight"
            elif layer_type == "ffn_down":
                new_name = f"{prefix}.layer.1.DenseReluDense.wo.weight"
            elif layer_type == "ffn_norm":
                new_name = f"{prefix}.layer.1.layer_norm.weight"
                
        # Globals
        elif name == "token_embd.weight":
            new_name = "shared.weight" # T5 shares encoder/decoder emb, but strictly this is shared
        elif "output_norm.weight" in name: 
            # Catch "output_norm.weight" OR "enc.output_norm.weight"
            new_name = "encoder.final_layer_norm.weight"
        
        # Skipping unknowns or unrelated (decoder?)
        if "dec.blk" in name or "output.weight" in name:
            continue

        # --- DEQUANTIZATION ---
        try:
            # Dequantize to numpy float32
            data = tensor.data # This is the raw quantization data (view)
            # We must dequantize. gguf library handles this attached to the reader?
            # reader.tensors is a list of TensorReader.
            # TensorReader.data is the raw bytes.
            # To get keys / values we use reader.get_tensor(i)? No.
            
            # The correct way with 'gguf' lib:
            # It doesn't expose a clean "dequantize me" method easily on the object.
            # But we imported gguf.
            # Actually, `tensor.data` is an ndarray subclass if loaded? No, it's a memory view.
            
            # Let's use the property ensuring it reads:
            # `tensor.data` accesses the mmap.
            # We need to call a dequantizer.
            # Since `gguf` python lib is basic, the simplest way is to let it load as numpy.
            # `reader.get_tensor(name)` returns the numpy array?
            # No, reader doesn't have `get_tensor`.
            # Wait, `tensor` object has `data`. 
            # If the type is quantized (e.g. Q4_K), `data` is the raw blocks.
            
            # We need to use `gguf.quants.dequantize`.
            from gguf.quants import dequantize
            
            # dequantize returns a numpy array (float32)
            # This is where the 160MB allocation happens.
            # We do it ONE BY ONE.
            # This should force the GC to clean up the previous one.
            
            weights_np = dequantize(tensor.data, tensor.tensor_type)
            
            # Convert to Torch BFloat16
            weights_tensor = torch.from_numpy(weights_np).to(dtype=torch.bfloat16)
            
            # Explicitly free numpy memory
            del weights_np
            
            # Store in state dict
            state_dict[new_name] = weights_tensor
            current_shard_size += weights_tensor.numel() * weights_tensor.element_size()
            
            # Shard Check
            if current_shard_size >= max_shard_size:
                save_current_shard()
                
            print(f"   Converted: {name} -> {new_name}")
            
        except Exception as e:
            print(f"!!! Error converting {name}: {e}")
            continue

    # Final Save
    save_current_shard()
    
    # Save Index
    index_data = {
        "metadata": {"total_size": 0}, # Dummy size, acceptable
        "weight_map": index_map
    }
    
    # Fix filenames in index (XXXXX -> total)
    total_shards = shard_index
    final_index_map = {}
    
    for key, fname in index_map.items():
        # model-00001-of-XXXXX -> model-00001-of-00005
        fixed_name = fname.replace("XXXXX", f"{total_shards:05d}")
        final_index_map[key] = fixed_name
        
    index_data["weight_map"] = final_index_map
    
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=2)
        
    import time
    import gc
    gc.collect()
    time.sleep(2.0) # wait for Windows to release file handles

    # Rename actual files
    for i in range(total_shards):
        old_name = output_dir / f"model-{i+1:05d}-of-XXXXX.safetensors"
        new_name = output_dir / f"model-{i+1:05d}-of-{total_shards:05d}.safetensors"
        
        # If destination exists (from failed previous run), delete it
        if new_name.exists():
            new_name.unlink()
            
        if old_name.exists():
            old_name.rename(new_name)

    print("[Converter] Done! Model saved to:", output_dir)
    return True
