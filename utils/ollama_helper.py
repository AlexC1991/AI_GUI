import shutil
import os
from pathlib import Path

def get_ollama_path():
    """Attempts to find the Ollama executable in PATH or default Windows locations."""
    
    # 1. Check Global PATH
    if shutil.which("ollama"):
        return "ollama"
    
    # 2. Check Local App Data (Standard Install)
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        default_path = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
        if default_path.exists():
            return str(default_path)
            
    # 3. Validation fallback (just in case env var missed)
    user_profile = os.environ.get("USERPROFILE", "")
    if user_profile:
        chk = Path(user_profile) / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
        if chk.exists():
            return str(chk)

    return None
