import json
from pathlib import Path

CONFIG_FILE = Path("config.json")

DEFAULT_CONFIG = {
    "llm": {
        "provider": "Google Gemini",
        "api_key": "",
        "local_model_dir": "models/llm",
        "cache_dir": "cache"
    },
    "image": {
        "checkpoint_dir": "models/checkpoints",
        "lora_dir": "models/loras",
        "output_dir": "outputs/images",
        "refiner_model": "None"
    },
    "remote": {
        "gpu_ip": "",
        "gpu_port": "8188",
        "gpu_auth": "",
        "server_ip": "0.0.0.0",
        "server_port": "7860",
        "server_active": False,
        "storage_provider": "Cloudflare R2",
        "storage_url": "",
        "storage_key": ""
    }
}

class ConfigManager:
    @staticmethod
    def load_config():
        if not CONFIG_FILE.exists():
            ConfigManager.save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG
        
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Merge with default to ensure all keys exist
                return ConfigManager._merge_defaults(config, DEFAULT_CONFIG)
        except Exception:
            return DEFAULT_CONFIG

    @staticmethod
    def save_config(config):
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    @staticmethod
    def _merge_defaults(user, default):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            elif isinstance(v, dict) and isinstance(user[k], dict):
                ConfigManager._merge_defaults(user[k], v)
        return user
