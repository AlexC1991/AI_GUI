from PySide6.QtCore import QThread, Signal
import subprocess
import sys
from utils.ollama_helper import get_ollama_path

class OllamaWorker(QThread):
    progress = Signal(str)      # "Downloading: 45%"
    finished = Signal()
    error = Signal(str)
    models_ready = Signal(list) # Returns list of model names
    
    def __init__(self, command, model=None):
        super().__init__()
        self.command = command # "pull" or "list"
        self.model = model
        
    def run(self):
        try:
            if self.command == "pull":
                self._run_pull()
            elif self.command == "list":
                self._run_list()
        except Exception as e:
            self.error.emit(str(e))

    def _run_list(self):
        print(f"[DEBUG] Starting list command")
        ollama_exe = get_ollama_path()
        print(f"[DEBUG] Ollama Path for list: {ollama_exe}")
        
        if not ollama_exe:
            print("[DEBUG] Error: Ollama not found")
            self.error.emit("Ollama not found")
            return
            
        try:
            cmd = [ollama_exe, "list"]
            print(f"[DEBUG] Executing list command: {cmd}")
            # Run with timeout to avoid stuck threads
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            print(f"[DEBUG] List finished. Code: {result.returncode}")
            if result.returncode != 0:
                 print(f"[DEBUG] List error: {result.stderr}")
            
            lines = result.stdout.strip().split('\n')
            models = []
            # Skip header (NAME ID SIZE MODIFIED)
            if len(lines) > 0 and "NAME" in lines[0]:
                lines = lines[1:]
            
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if parts: models.append(parts[0])
                    
            print(f"[DEBUG] Models found: {models}")
            self.models_ready.emit(models)
            
        except subprocess.TimeoutExpired:
            print("[DEBUG] List timed out!")
            self.error.emit("Ollama Unresponsive (Timeout)")
        except Exception as e:
            print(f"[DEBUG] List Exception: {e}")
            self.error.emit(str(e))

    def _run_pull(self):
        print(f"[DEBUG] Starting pull for model: {self.model}")
        
        # Check if ollama exists
        ollama_exe = get_ollama_path()
        print(f"[DEBUG] Ollama Path resolved: {ollama_exe}")
        
        if not ollama_exe:
            print("[DEBUG] Error: Ollama not found")
            self.error.emit("Ollama is not installed or not in PATH.")
            return

        cmd = [ollama_exe, "pull", self.model]
        print(f"[DEBUG] Executing command: {cmd}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            print("[DEBUG] Process started. Reading stdout...")
            
            for line in process.stdout:
                line = line.strip()
                if line:
                    print(f"[DEBUG] Ollama Output: {line}")
                    self.progress.emit(line)
                    
            process.wait()
            print(f"[DEBUG] Process finished with code: {process.returncode}")
            
            if process.returncode == 0:
                self.finished.emit()
            else:
                self.error.emit(f"Failed with code {process.returncode}")
        except Exception as e:
            print(f"[DEBUG] Exception in Popen: {e}")
            self.error.emit(str(e))
