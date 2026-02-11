"""
Configuration and Constants for IronGate
"""

import os
import json
import logging
from colorama import Fore, Style

# Version
VERSION = "10.3"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(BASE_DIR, "host_config.json")
DATABASE_FILE = os.path.join(BASE_DIR, "clients.json")
IDENTITY_FILE = os.path.join(BASE_DIR, "host_identity.key")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Constants
LOCKDOWN_MODE = False
MAX_LOGIN_ATTEMPTS = 5
BAN_DURATION = 300  # 5 minutes
SESSION_TIMEOUT = 3600 * 24  # 24 hours
RATE_LIMIT_Window = 60
RATE_LIMIT_Max = 30

# Globals (will be loaded)
host_config = {}
host_identity = {}

def load_config():
    global host_config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                host_config.clear()
                host_config.update(data)
                # DEBUG: print(f"[DEBUG] Loaded config keys: {list(host_config.keys())}")
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Failed to load config: {e}")
            # Don't clear if load fails? Or clear to empty?
            # Safer to keep existing defaults if we had them, but here we start empty.
            pass
    else:
        # print(f"{Fore.YELLOW}[WARN] Config file not found at: {CONFIG_FILE}")
        pass
        
    return host_config

def save_config():
    with open(CONFIG_FILE, "w") as f:
        json.dump(host_config, f, indent=4)

def load_host_identity():
    global host_identity
    if os.path.exists(IDENTITY_FILE):
        try:
            with open(IDENTITY_FILE, "r") as f:
                data = json.load(f)
                host_identity.clear()
                host_identity.update(data)
            return True
        except:
            pass
    
    # Generate new if needed
    import uuid
    import secrets
    
    new_identity = {
        "host_id": str(uuid.uuid4()),
        "secret_key": secrets.token_hex(32),
        "created_at": str(os.path.getctime(CONFIG_FILE) if os.path.exists(CONFIG_FILE) else "")
    }
    
    host_identity.clear()
    host_identity.update(new_identity)
    
    with open(IDENTITY_FILE, "w") as f:
        json.dump(host_identity, f, indent=4)
    return False

# Logging
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "iron_host.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def log_event(event_type, client_id="SYSTEM", details=None):
    msg = f"[{event_type}] Client: {client_id} | {json.dumps(details or {})}"
    logging.info(msg)
    print(f"{Fore.CYAN}[LOG] {event_type}: {client_id}")

