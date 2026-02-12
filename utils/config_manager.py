import json
import os
from pathlib import Path

# Resolve config.json relative to AI_GUI root (parent of utils/), not cwd
# This ensures IronGate (running from gateway/) still finds the right file
_AI_GUI_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_FILE = _AI_GUI_ROOT / "config.json"

DEFAULT_CONFIG = {
    "llm": {
        "provider": "Google Gemini",
        "providers": {
            "openai": { "api_key": "", "models": [] },
            "gemini": { "api_key": "", "models": [] },
            "anthropic": { "api_key": "", "models": [] },
            "deepseek": { "api_key": "", "models": [] },
            "kimi": { "api_key": "", "models": [] },
            "zai": { "api_key": "", "models": [] },
            "xai": { "api_key": "", "models": [] },
            "mistral": { "api_key": "", "models": [] },
            "openrouter": { "api_key": "", "models": [] }
        },
        "local_model_dir": "models/llm",
        "cache_dir": "cache"
    },
    "cloud": {
        "runpod_api_key": "",
        "hf_token": "",
        "pod_id": "",
        "models": {
            "Qwen/Qwen2.5-72B-Instruct-AWQ": "Qwen 2.5 72B (AWQ)",
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "DeepSeek R1 8B",
            "mistralai/Mistral-Small-3.2-24B-Instruct-2506": "Mistral Small 24B",
            "meta-llama/Llama-3.3-70B-Instruct": "Llama 3.3 70B",
            "Qwen/Qwen3-14B": "Qwen 3 14B"
        },
        "gpu_tiers": {
            "tier_standard": [
                "NVIDIA A40",
                "NVIDIA RTX A6000",
                "NVIDIA RTX 6000 Ada",
                "NVIDIA A100 80GB PCIe"
            ],
            "tier_ultra": [
                "NVIDIA A100 80GB PCIe",
                "NVIDIA A100-SXM4-80GB",
                "NVIDIA H100 80GB HBM3"
            ]
        },
        "keyword_tiers": [
            [["70b", "72b", "miqu", "grok", "120b"], "tier_ultra"],
            [["*"], "tier_standard"]
        ]
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
                config = ConfigManager._merge_defaults(config, DEFAULT_CONFIG)
                # Migrate old cloud config keys from llm section
                config = ConfigManager._migrate_legacy_cloud_config(config)
                return config
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

    @staticmethod
    def _migrate_legacy_cloud_config(config):
        """Migrate old llm.runpod_key/runpod_id/runpod_port to cloud section."""
        llm = config.get("llm", {})
        cloud = config.get("cloud", {})
        migrated = False

        if "runpod_key" in llm and llm["runpod_key"]:
            cloud["runpod_api_key"] = llm.pop("runpod_key")
            migrated = True
        if "runpod_id" in llm:
            cloud["pod_id"] = llm.pop("runpod_id")
            migrated = True
        if "runpod_port" in llm:
            llm.pop("runpod_port")  # Port is no longer user-configurable
            migrated = True

        if migrated:
            config["cloud"] = cloud
            config["llm"] = llm
            ConfigManager.save_config(config)

        return ConfigManager._migrate_legacy_keys(config)

    @staticmethod
    def _migrate_legacy_keys(config):
        """Migrate old top-level keys to provider specific config."""
        llm = config.get("llm", {})
        providers = llm.get("providers", {})
        migrated = False

        # OpenAI
        if "openai_api_key" in llm and llm["openai_api_key"]:
            if "openai" not in providers: providers["openai"] = {"api_key": "", "models": []}
            providers["openai"]["api_key"] = llm.pop("openai_api_key")
            migrated = True

        # Gemini
        if "api_key" in llm and llm["api_key"]:
            if "gemini" not in providers: providers["gemini"] = {"api_key": "", "models": []}
            providers["gemini"]["api_key"] = llm.pop("api_key")
            migrated = True

        # Anthropic
        if "anthropic_api_key" in llm and llm["anthropic_api_key"]:
            if "anthropic" not in providers: providers["anthropic"] = {"api_key": "", "models": []}
            providers["anthropic"]["api_key"] = llm.pop("anthropic_api_key")
            migrated = True

        if migrated:
            llm["providers"] = providers
            config["llm"] = llm
            ConfigManager.save_config(config)

        return config
