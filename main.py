import sys
import subprocess
import importlib

# -------------------------
# Dependency Configuration
# -------------------------

REQUIRED_PACKAGES = {
    "PySide6": "PySide6>=6.6",
    "markdown": "markdown",
    "pygments": "pygments",
    "psutil": "psutil"  # <--- NEW DEPENDENCY FOR SYSTEM INFO
}

# -------------------------
# Dependency Utilities
# -------------------------

def install_package(package: str):
    print(f"[VoxAI] Installing dependency: {package}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package]
        )
    except subprocess.CalledProcessError as e:
        print(f"[VoxAI] Failed to install {package}. Error: {e}")
        sys.exit(1)

def ensure_dependencies():
    print("[VoxAI] Checking dependencies...")
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            print(f"[VoxAI] Missing module: {module_name}")
            install_package(package_name)

    print("[VoxAI] All dependencies satisfied.\n")

# -------------------------
# Application Entry Point
# -------------------------

def main():
    ensure_dependencies()

    from PySide6.QtWidgets import QApplication
    from main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()