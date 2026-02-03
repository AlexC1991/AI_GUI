from gguf import GGUFReader
import sys
from pathlib import Path

# Path to the model
model_path = Path("A:/Github/AI_GUI/models/text_encoders/t5-v1_1-xxl-encoder-Q4_K_M.gguf")

if not model_path.exists():
    print(f"Error: {model_path} not found")
    sys.exit(1)

print(f"Reading keys from {model_path}...")
reader = GGUFReader(str(model_path))

count = 0
for tensor in reader.tensors:
    if count < 20: # Just print first 20 to see pattern
        print(f"Key: {tensor.name} | Shape: {tensor.shape}")
    count += 1

print(f"Total keys: {count}")
