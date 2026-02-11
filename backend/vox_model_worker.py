from PySide6.QtCore import QThread, Signal
import os
import sys

# Resolve path to LLM models
def get_vox_models_dir():
    # Go up from backend/ -> root -> models/llm
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root_dir, "models", "llm")

class VoxModelWorker(QThread):
    progress = Signal(str)      
    finished = Signal()
    error = Signal(str)
    models_ready = Signal(list) 
    
    def __init__(self, command, model=None):
        super().__init__()
        self.command = command # "list" 
        self.model = model
        
    def run(self):
        try:
            if self.command == "list":
                self._run_list()
            else:
                self.error.emit(f"Command '{self.command}' not supported by VoxAI Engine yet.")
        except Exception as e:
            self.error.emit(str(e))

    def _format_model_name(self, filename):
        """Clean up .gguf filename to readable short display name.
        Examples:
            Llama-3.2-3B-Instruct-Q4_K_M.gguf -> Llama 3.2 3B
            Qwen2.5-0.5B-Instruct-Q4_K_M.gguf -> Qwen2.5 0.5B
            dolphin-2.8-mistral-7b-v02-Q4_K_M.gguf -> Dolphin 2.8 Mistral 7B
            gemma-2-9b-it-IQ4_XS.gguf -> Gemma 2 9B
        """
        import re
        name = filename.replace(".gguf", "")

        # Remove quantization suffixes (Q4_K_M, IQ4_XS, etc.)
        name = re.sub(r'[-_]?(?:Q\d+_K_[A-Z]+|IQ\d+_[A-Z]+)', '', name, flags=re.IGNORECASE)

        # Remove common noise: Instruct, it, chat, v02
        name = re.sub(r'[-_.]?(?:Instruct|instruct|chat)', '', name)
        name = re.sub(r'[-_]v\d+', '', name)

        # Preserve version dots (3.2, 2.5) but split on hyphens/underscores
        name = re.sub(r'(\d)\.(\d)', r'\1_DOT_\2', name)
        name = name.replace('-', ' ').replace('_', ' ')
        name = name.replace(' DOT ', '.')
        name = re.sub(r'\s+', ' ', name).strip()

        parts = name.split()
        result = []
        for p in parts:
            if not p or p == '.':
                continue
            # Uppercase size tokens (7b -> 7B, 3b -> 3B)
            if re.match(r'^\d+[bB]$', p):
                result.append(p.upper())
            # Capitalize first letter
            elif p[0].islower():
                result.append(p[0].upper() + p[1:])
            else:
                result.append(p)

        # Limit to 4 words
        if len(result) > 4:
            result = result[:4]

        return ' '.join(result) if result else filename

    def _run_list(self):
        print(f"[VoxModelWorker] Scanning for models...")
        models_dir = get_vox_models_dir()
        print(f"[VoxModelWorker] Directory: {models_dir}")
        
        if not os.path.exists(models_dir):
            print("[VoxModelWorker] Models directory not found.")
            self.error.emit("VoxAI models directory missing.")
            return

        try:
            files = [f for f in os.listdir(models_dir) if f.endswith(".gguf")]
            print(f"[VoxModelWorker] Found: {files}")

            structured_list = []
            for f in files:
                filepath = os.path.join(models_dir, f)
                size_text = ""
                try:
                    size_bytes = os.path.getsize(filepath)
                    if size_bytes >= 1024**3:
                        size_text = f"{size_bytes / (1024**3):.1f} GB"
                    elif size_bytes >= 1024**2:
                        size_text = f"{size_bytes / (1024**2):.0f} MB"
                except Exception:
                    pass
                structured_list.append({
                    "display": self._format_model_name(f),
                    "filename": f,
                    "size": size_text,
                })
            
            if not files:
                self.models_ready.emit([])
            else:
                self.models_ready.emit(structured_list)
        except Exception as e:
            print(f"[VoxModelWorker] Error listing files: {e}")
            self.error.emit(str(e))
