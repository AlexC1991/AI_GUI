# config.py — Bridge to AI_GUI's ConfigManager
#
# RunPod backend reads GPU_TIERS, MODEL_SPECIFIC_TIERS, KEYWORD_TIERS
# from here. This file loads them from config.json via ConfigManager.

import os
import sys

# Ensure AI_GUI root is in path so we can import ConfigManager
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    from utils.config_manager import ConfigManager
    _config = ConfigManager.load_config()
    _cloud = _config.get("cloud", {})
except Exception:
    _cloud = {}

# --- Exported Config Values ---
API_KEY = _cloud.get("runpod_api_key", "")
HF_TOKEN = _cloud.get("hf_token", "")
POD_ID = _cloud.get("pod_id", None) or None  # None if empty string

# Cloud model map: {hf_id: display_name}
MODEL_MAP = _cloud.get("models", {})

# GPU Tiers
GPU_TIERS = _cloud.get("gpu_tiers", {
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
})

# Model-specific tier overrides (exact HF ID match)
MODEL_SPECIFIC_TIERS = {
    "cecibas/Midnight-Miqu-70B-v1.5-4bit": "tier_ultra",
    "Qwen/Qwen2.5-72B-Instruct-AWQ": "tier_ultra",
    "meta-llama/Llama-3.3-70B-Instruct": "tier_ultra",
}

# Keyword-based tier fallback
_raw_kw = _cloud.get("keyword_tiers", [
    [["70b", "72b", "miqu", "grok", "120b"], "tier_ultra"],
    [["*"], "tier_standard"]
])
# Convert from JSON list format [[keywords, tier], ...] to tuple format
KEYWORD_TIERS = [(kw, tier) for kw, tier in _raw_kw]
