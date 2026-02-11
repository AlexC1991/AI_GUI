import importlib
import sys
import pkg_resources
import subprocess

# Define the list of required packages and their corresponding import names
# Format: (Package Name in pip, Import Name in python)
# If import name is None, we only check pip installation
REQUIRED_PACKAGES = [
    # --- Core GUI ---
    ("PySide6", "PySide6"),
    ("pillow", "PIL"),
    ("requests", "requests"),
    ("python-dotenv", "dotenv"),
    ("markdown", "markdown"),
    ("pygments", "pygments"),

    # --- AI Providers ---
    ("google-generativeai", "google.generativeai"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("gguf", "gguf"),
    ("msgpack", "msgpack"),
    ("chromadb", "chromadb"),

    # --- Local LLM ---
    ("llama-cpp-python", "llama_cpp"),
    ("psutil", "psutil"),

    # --- IronGate Web Gateway ---
    ("ddgs", "ddgs"), # detected as 'ddgs' package exposing 'ddgs' module? or 'duckduckgo_search'? pip show ddgs says name is ddgs.
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pyngrok", "pyngrok"),
    ("colorama", "colorama"),
    ("filelock", "filelock"),
    ("jinja2", "jinja2"),
    ("python-multipart", "multipart"), # Import is often 'multipart' but package is python-multipart
    ("pyinstaller", "PyInstaller"),

    # --- Utilities ---
    ("paramiko", "paramiko"),
    ("scp", "scp"),
]

def check_dependencies():
    print("="*60)
    print("      VOXAI DEPENDENCY CHECKER")
    print("="*60)
    print(f"Python: {sys.version.split()[0]}")
    print("-"*60)
    
    missing = []
    installed = []
    
    # Get list of installed packages from pip
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    for pkg_name, import_name in REQUIRED_PACKAGES:
        status = "MISSING"
        details = ""
        
        # 1. Check PIP installation
        is_installed = pkg_name.lower() in installed_packages
        version = installed_packages.get(pkg_name.lower(), "N/A")
        
        # 2. Check Import (if specified)
        is_importable = False
        if import_name:
            try:
                importlib.import_module(import_name)
                is_importable = True
            except ImportError:
                pass
        
        # Determine Status
        if is_installed:
            if import_name and not is_importable:
                status = "BROKEN" # Installed but can't import
                details = f"(Pip: {version}, Import Failed)"
                missing.append(pkg_name)
            else:
                status = "OK"
                details = f"({version})"
                installed.append(pkg_name)
        else:
            # Maybe it's installed but under a different pip name?
            # Try import as fallback
            if import_name and is_importable:
                 status = "OK (Hidden)"
                 details = "(Imported, but pip name mismatch)"
                 installed.append(pkg_name)
            else:
                 status = "MISSING"
                 missing.append(pkg_name)

        # Print Result
        symbol = "[OK]" if "OK" in status else "[MISSING]"
        if status == "BROKEN": symbol = "[BROKEN]"
        
        print(f"{symbol:<10} {pkg_name:<20} {status:<10} {details}")

    print("-" * 60)
    if missing:
        print(f"[!] Found {len(missing)} missing or broken packages.")
        print("Run the following command to fix:")
        print(f"pip install {' '.join(missing)}")
        sys.exit(1)
    else:
        print("[OK] All dependencies are satisfied.")
        sys.exit(0)

if __name__ == "__main__":
    check_dependencies()
