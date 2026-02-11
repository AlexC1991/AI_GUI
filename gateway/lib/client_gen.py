"""
Client Generation Module
"""
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from colorama import Fore
from .config import EXPORTS_DIR, VERSION, log_event, host_config, host_identity

# Embedded Client Template (Robust & Standalone)
# NOTE: Double braces {{}} are escaped so .format() only replaces the 5 placeholders.
CLIENT_TEMPLATE = '''import os
import sys
import json
import time
import webbrowser
import urllib.request
import urllib.error

# Configuration
CLIENT_ID = "{client_id}"
CLIENT_NAME = "{client_name}"
CLIENT_SIGNATURE = "{signature}"
HOST_ID = "{host_id}"
TUNNEL_URL = "{tunnel_url}"

def log(msg):
    print(f"[IronClient] {{msg}}")

def api_post(endpoint, data):
    url = f"{{TUNNEL_URL.rstrip('/')}}{{endpoint}}"
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={{'Content-Type': 'application/json'}}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        log(f"API Error {{e.code}}: {{e.reason}}")
        return None
    except Exception as e:
        log(f"Connection Error: {{e}}")
        return None

def main():
    os.system("cls" if os.name == "nt" else "clear")
    print("=== VoxAI Iron Gate Client ===")
    print(f"Client: {{CLIENT_NAME}} ({{CLIENT_ID}})")
    print(f"Server: {{TUNNEL_URL}}")
    print("\\nConnecting...")

    while True:
        # 1. Heartbeat / Check Status
        resp = api_post("/api/beacon", {{"client_id": CLIENT_ID, "signature": CLIENT_SIGNATURE}})

        if resp and resp.get("status") == "APPROVED":
            print("\\n[SUCCESS] Connection Approved!")
            print("Opening VoxAI in your browser...")
            webbrowser.open(f"{{TUNNEL_URL}}/app")
            # Keep alive loop
            while True:
                time.sleep(60)
                api_post("/api/beacon", {{"client_id": CLIENT_ID}})

        elif resp and resp.get("status") == "DENIED":
            print("\\n[WARN] Device not trusted by Host.")
            code = input("Enter Install Code to Activate: ").strip()

            act_resp = api_post("/api/activate", {{
                "client_id": CLIENT_ID,
                "install_code": code,
                "signature": CLIENT_SIGNATURE
            }})

            if act_resp and act_resp.get("status") == "OK":
                print("[SUCCESS] Activation Successful!")
                continue # Retry beacon
            else:
                print(f"[ERROR] Activation Failed: {{act_resp.get('message') if act_resp else 'Unknown'}}")

        else:
            print("[ERROR] Could not connect to Host. Retrying in 5s...")
            time.sleep(5)

if __name__ == "__main__":
    main()
'''

def generate_id(prefix="ID"):
    import uuid
    return f"{prefix}-{str(uuid.uuid4())[:8].upper()}"

def short_id(full_id):
    return full_id.split("-")[1]

def generate_signature():
    import secrets
    return secrets.token_hex(16)

def generate_code(parts=3, length=4):
    import secrets, string
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "-".join("".join(secrets.choice(alphabet) for _ in range(length)) for _ in range(parts))

def generate_client_package(name, tunnel_url, db_callback):
    """Generate a new client package with exe and zip."""

    client_id = generate_id("CLT")
    signature = generate_signature()
    install_code = generate_code(4, 3)

    # Update DB via callback
    db_callback(client_id, {
        "name": name,
        "signature": signature,
        "install_code": install_code,
        "install_code_used": False,
        "bound_host": host_identity.get("host_id"),
        "created_at": datetime.now().isoformat(),
        "activated_at": None,
        "status": "PENDING_ACTIVATION"
    })

    # Prepare package path
    safe_name = "".join(c if c.isalnum() or c in "_ " else "_" for c in name).replace(" ", "_")
    short = short_id(client_id)
    package_folder_name = f"VoxAI_Client_{safe_name}_{short}"
    package_path = os.path.join(EXPORTS_DIR, package_folder_name)

    if os.path.exists(package_path):
        shutil.rmtree(package_path)
    os.makedirs(package_path, exist_ok=True)

    # Write Client Script
    script_content = CLIENT_TEMPLATE.format(
        client_id=client_id,
        client_name=name,
        signature=signature,
        host_id=host_identity.get("host_id"),
        tunnel_url=tunnel_url
    )

    script_path = os.path.join(package_path, "client.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    # Write activation code to a readme inside the package
    readme_path = os.path.join(package_path, "ACTIVATION_CODE.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"=== VoxAI Iron Gate - Client Activation ===\n\n")
        f.write(f"Client Name : {name}\n")
        f.write(f"Client ID   : {client_id}\n")
        f.write(f"Server      : {tunnel_url}\n\n")
        f.write(f"ACTIVATION CODE: {install_code}\n\n")
        f.write(f"Instructions:\n")
        f.write(f"1. Run VoxAI_Client.exe (or start.bat if no exe)\n")
        f.write(f"2. When prompted, enter the activation code above\n")
        f.write(f"3. Once activated, VoxAI will open in your browser\n")

    # Write Start Batch (fallback if no exe)
    bat_path = os.path.join(package_path, "start.bat")
    with open(bat_path, "w") as f:
        f.write("@echo off\ntitle VoxAI Client\npython client.py\npause")

    # --- Build EXE with PyInstaller ---
    exe_built = False
    exe_name = "VoxAI_Client.exe"
    try:
        print(f"{Fore.WHITE}  Building executable...", end=" ", flush=True)
        result = subprocess.run([
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--console",
            "--name", "VoxAI_Client",
            "--distpath", package_path,
            "--workpath", os.path.join(package_path, "_build"),
            "--specpath", os.path.join(package_path, "_build"),
            "--clean",
            "--noconfirm",
            script_path
        ], capture_output=True, text=True, timeout=120, cwd=package_path)

        if result.returncode == 0 and os.path.exists(os.path.join(package_path, exe_name)):
            exe_built = True
            print(f"{Fore.GREEN}OK")
        else:
            print(f"{Fore.YELLOW}Failed (will include .bat + .py instead)")
            if result.stderr:
                err_lines = result.stderr.strip().split('\n')
                for line in err_lines[-3:]:
                    print(f"  {Fore.RED}{line}")
    except FileNotFoundError:
        print(f"{Fore.YELLOW}PyInstaller not found (pip install pyinstaller)")
    except subprocess.TimeoutExpired:
        print(f"{Fore.YELLOW}Build timed out")
    except Exception as e:
        print(f"{Fore.YELLOW}Build error: {e}")

    # Clean up build artifacts
    build_dir = os.path.join(package_path, "_build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir, ignore_errors=True)
    # Remove .py source and .bat if exe was built successfully
    if exe_built:
        try:
            os.remove(script_path)
            os.remove(bat_path)
        except:
            pass

    # --- Create ZIP ---
    zip_name = f"{package_folder_name}.zip"
    zip_path = os.path.join(EXPORTS_DIR, zip_name)
    try:
        print(f"{Fore.WHITE}  Creating zip...", end=" ", flush=True)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(package_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join(package_folder_name, os.path.relpath(file_path, package_path))
                    zf.write(file_path, arcname)
        print(f"{Fore.GREEN}OK -> {zip_name}")
    except Exception as e:
        print(f"{Fore.YELLOW}Zip failed: {e}")
        zip_path = None

    # [GUI INTEGRATION] Write install code for desktop app to read
    try:
        with open("vox_install_code.txt", "w") as f:
            f.write(f"Install Code: {install_code}")
    except:
        pass

    # Return result
    return True, {
        "client_id": client_id,
        "name": name,
        "install_code": install_code,
        "exe_path": os.path.join(package_path, exe_name) if exe_built else bat_path,
        "exe_name": exe_name if exe_built else "start.bat",
        "zip_path": zip_path,
        "zip_name": zip_name if zip_path else None,
        "package_path": package_path
    }
