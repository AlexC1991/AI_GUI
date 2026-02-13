# VoxAI Orchestrator

<div align="center">

```
 ##     ##  #######  ##     ##    ###    ####
 ##     ## ##     ##  ##   ##    ## ##    ##
 ##     ## ##     ##   ## ##    ##   ##   ##
 ##     ## ##     ##    ###    ##     ##  ##
  ##   ##  ##     ##   ## ##   #########  ##
   ## ##   ##     ##  ##   ##  ##     ##  ##
    ###     #######  ##     ## ##     ## ####
```

**A unified local AI workspace with agentic capabilities**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-green.svg)](https://doc.qt.io/qtforpython/)
[![Status: Beta](https://img.shields.io/badge/Status-Beta-orange.svg)]()

*Local LLM Chat &bull; Cloud Providers &bull; Image & Video Generation &bull; Agentic Web Search &bull; Remote Access*

> **Beta Software:** This project is under active development. Expect bugs, breaking changes, and missing features. Contributions and bug reports welcome!

</div>

---

## What Makes VoxAI Different?

VoxAI isn't just another AI GUI — it's an **agentic framework** that gives local models superpowers:

- **Metacognition** — Your AI understands its own knowledge limitations
- **Autonomous Search** — AI decides when to search the web, no manual triggers needed
- **10 LLM Providers** — Local GGUF, Ollama, Gemini, OpenAI, OpenRouter, DeepSeek, Kimi, Mistral, xAI, Z.ai
- **Image & Video Gen** — Local Diffusers (SD/SDXL/Flux) + cloud (Gemini Image, DALL-E 3, Veo, Sora)
- **IronGate Gateway** — Full web UI with 10 themes, ngrok tunneling, and client package generation
- **Budget-Friendly** — Runs on 8GB VRAM AMD/NVIDIA GPUs

> *"I gave Qwen3-8B internet access and it achieved metacognition"* — Creator

---

## Features

### Chat Interface
| Feature | Description |
|---------|-------------|
| **Multi-Provider** | 10 cloud providers + local GGUF + Ollama — switch models mid-conversation |
| **Provider Tiles** | Visual tile grid for browsing providers, drill-down model lists, star favorites |
| **Streaming** | Real-time token-by-token output with speed stats (tok/s) |
| **Code Detection** | Auto-extracts code blocks to a persistent syntax-highlighted side panel |
| **Copy & Download** | One-click copy/download buttons on every code block and file card |
| **File Uploads** | Attach files for AI analysis |
| **Thinking Models** | Collapsible `<think>` reasoning sections for Qwen3, DeepSeek-R1, etc. |
| **Cross-Provider History** | Full conversation follows you when switching between models |

### Agentic Search (Metacognition)
| Feature | Description |
|---------|-------------|
| **Autonomous** | AI decides when it needs current information |
| **Self-Aware** | Understands its knowledge cutoff and acts on it |
| **Failsafe** | Catches refusal phrases ("I don't have access to...") and auto-searches |
| **Natural** | Just ask normally — the AI handles the rest |

> **Note:** Agentic Search requires the Desktop Service running (starts automatically from Settings or `python gateway/iron_desktop.py`)

### Image & Video Generation
| Feature | Description |
|---------|-------------|
| **Local Backend** | Native Diffusers pipeline — no ComfyUI dependency |
| **Image Models** | SD 1.5, SDXL, Pony, Illustrious, Flux (GGUF quantized for 8GB VRAM) |
| **Cloud Image** | Gemini Flash Image, Gemini Pro Image, GPT Image 1, DALL-E 3 |
| **Video Gen** | Veo 3.1 (Gemini) and Sora 2 / Sora 2 Pro (OpenAI) |
| **LoRA & VAE** | Full LoRA adapter and custom VAE support |
| **AMD Optimized** | ZLUDA acceleration for RX 6000/7000 series |
| **Prompt Enhancer** | AI-powered prompt rewriting for better image quality |

### Cloud LLM Providers
| Provider | Config Key | Models |
|----------|-----------|--------|
| **Gemini** | `gemini` | gemini-2.0-flash, gemini-2.5-pro, etc. |
| **OpenAI** | `openai` | gpt-4o, o3, o4-mini, etc. |
| **OpenRouter** | `openrouter` | 300+ models from all major providers |
| **DeepSeek** | `deepseek` | deepseek-chat, deepseek-reasoner |
| **Kimi** | `kimi` | kimi-k2.5, moonshot-v1-128k |
| **Mistral** | `mistral` | mistral-large, codestral, etc. |
| **xAI** | `xai` | grok-4, grok-3-mini, etc. |
| **Z.ai** | `zai` | glm-5, glm-4.7, etc. |

All providers use the OpenAI-compatible API pattern and are dynamically resolved via the `PROVIDER_REGISTRY`.

### VoxAI Chat Engine (Local LLM)
| Feature | Description |
|---------|-------------|
| **Local LLM** | llama.cpp with Vulkan acceleration |
| **Hot-Swap** | Switch models without restart |
| **Auto-Templates** | Correct chat format per architecture (ChatML, Llama, Alpaca, etc.) |
| **Elastic Memory** | Dynamic context windows, RAG retrieval, msgpack persistence |
| **Zero Dependencies** | No Ollama or external servers required |

### IronGate Web Gateway
| Feature | Description |
|---------|-------------|
| **Remote Access** | ngrok tunnel with custom domain support |
| **Aurora Web UI** | Glass-morphism themed interface with 10 color themes |
| **All Providers** | Every configured LLM provider available in the web UI |
| **Image & Video** | Full image/video generation from the browser |
| **Client Generation** | Standalone `.exe` packages with activation codes |
| **Security** | IP banning, rate limiting, session management |
| **Themes** | Arctic, Cyberpunk, Ember, Gemini, Jade, Midnight, Sunset, Synthwave, Violet, Zen |

### Coming Soon
| Feature | Status |
|---------|--------|
| **Code / IDE Mode** | Coming Soon |
| **Audio Generation** | Coming Soon |

---

## Quick Start

### Prerequisites
- Python 3.10+
- AMD GPU with ZLUDA (or NVIDIA with CUDA)
- 8GB+ VRAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/AlexC1991/AI_GUI.git
cd AI_GUI

# Launch (handles everything automatically)
start.bat
```

`start.bat` automatically handles:
- Virtual environment creation and activation
- Dependency installation from `requirements.txt`
- DLL verification (llama.cpp Vulkan binaries)
- ZLUDA/HIP setup for AMD GPUs
- Temp/cache directory routing (A: drive, E: for shader cache)
- Directory structure creation

### Manual Setup (if not using start.bat)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Configure
copy config.example.json config.json
# Edit config.json with your API keys

# Add a GGUF model to models/llm/

# Launch
python main.py
```

### DLL Setup (Local LLM)
For Vulkan-accelerated local inference, place these in the project root:
- `llama.dll` / `ggml.dll` / `ggml-vulkan.dll` / `ggml-cpu-haswell.dll`

Build from [llama.cpp](https://github.com/ggerganov/llama.cpp) with Vulkan, or use pre-built releases.

---

## Model Setup

### Chat Models (GGUF) — For Local AI Chat

Place `.gguf` model files in the `models/llm/` folder:

```
AI_GUI/
└── models/
    └── llm/
        ├── Qwen3-8B-Q5_K_M.gguf        ← Recommended for agentic search
        ├── Llama-3.2-3B-Instruct.gguf   ← Lightweight alternative
        └── (any .gguf model works)
```

**Recommended Models:**
| Model | Size | VRAM | Best For |
|-------|------|------|----------|
| Qwen3-8B-Q5_K_M | 5.5GB | 8GB | Agentic search, reasoning |
| Qwen3-Coder-30B-A3B | 18GB | 12GB+ | Code generation |
| Llama-3.2-3B-Instruct | 2GB | 4GB | Fast responses, low VRAM |
| DeepSeek-R1-Distill-8B | 5GB | 8GB | Deep reasoning |

> **Tip:** Q4_K_M or Q5_K_M quantizations offer the best quality/size balance.

### Cloud Providers — API Key Setup

1. Open Settings in the app (gear icon)
2. Scroll to the LLM Providers section
3. Paste your API key for each provider
4. Click **Fetch Models** to populate the available model list
5. Select models and click **Add** (up to 5 per provider, 50 for OpenRouter)
6. Click **Apply LLM Settings**

Your selected models appear as **provider tiles** in the model selector panel.

---

### Image Models — For Local Image Generation

```
AI_GUI/
└── models/
    ├── checkpoints/                      ← Diffusion model files
    │   ├── sd_xl_base_1.0.safetensors
    │   ├── ponyDiffusionV6XL.safetensors
    │   └── flux1-schnell-Q4_K_S.gguf    ← Flux (8GB VRAM!)
    │
    ├── loras/                            ← LoRA style files
    │   └── your-lora-file.safetensors
    │
    ├── vae/                              ← VAE files (optional)
    │   └── sdxl_vae.safetensors
    │
    └── text_encoders/                    ← For Flux models
        ├── clip_l.safetensors
        └── t5-v1_1-xxl-encoder-Q4_K_M.gguf
```

**Supported Formats:** `.safetensors`, `.ckpt`, `.gguf`

Cloud image/video models (Gemini Image, DALL-E, Veo, Sora) require no local files — just API keys.

---

### Flux on 8GB VRAM

> **Work in Progress:** Flux GGUF support is experimental.

1. Download [flux1-schnell-Q4_K_S.gguf](https://huggingface.co/city96/FLUX.1-schnell-gguf) to `models/checkpoints/`
2. Download [t5-v1_1-xxl-encoder-Q4_K_M.gguf](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf) to `models/text_encoders/`
3. Download [clip_l.safetensors](https://huggingface.co/comfyanonymous/flux_text_encoders) to `models/text_encoders/`
4. Select the Flux model from the Image Generation tab

---

## IronGate Remote Access

Access your local AI from anywhere through a secure web tunnel.

### Setup
1. Get an [ngrok](https://ngrok.com) auth token
2. Create `gateway/host_config.json`:
   ```json
   {
       "ngrok_token": "YOUR_TOKEN",
       "ngrok_domain": "your-domain.ngrok.app"
   }
   ```
3. Start: `python gateway/iron_host.py`
4. Share the Magic Host Link!

### Web UI Features
- All configured LLM providers available (Gemini, OpenAI, DeepSeek, etc.)
- Image generation (local + cloud)
- Video generation (Veo, Sora)
- 10 color themes
- Chat history with localStorage persistence
- Model stats (VRAM, speed, parameter count)

### Generate Client Packages
```
Admin> gen "Friend Name"
```
Creates standalone `.exe` + activation code in `exports/`.

---

## Supported Hardware

| GPU | VRAM | Local LLM | SD 1.5 | SDXL | Flux GGUF | Flux Full |
|-----|------|:---------:|:------:|:----:|:---------:|:---------:|
| RX 6600 | 8GB | ✅ | ✅ | ✅ | ✅ | ❌ |
| RX 6700 XT | 12GB | ✅ | ✅ | ✅ | ✅ | ✅ |
| RX 7900 XTX | 24GB | ✅ | ✅ | ✅ | ✅ | ✅ |
| RTX 3060 | 12GB | ✅ | ✅ | ✅ | ✅ | ✅ |
| RTX 4090 | 24GB | ✅ | ✅ | ✅ | ✅ | ✅ |

Cloud providers (Gemini, OpenAI, etc.) work on **any hardware** — no GPU required.

---

## Project Structure

```
AI_GUI/
├── main.py                  # Entry point
├── main_window.py           # Main orchestrator & UI wiring
├── bootstrap.py             # Temp/cache directory setup
├── start.bat                # Windows launcher (handles all setup)
├── config.example.json      # Configuration template
├── requirements.txt         # Python dependencies
│
├── backend/                 # Workers & orchestration
│   ├── chat_agent.py        # Agentic controller (search, thinking, failsafe)
│   ├── chat_worker.py       # LLM streaming thread
│   ├── image_worker.py      # Image/video generation thread
│   ├── image_generator.py   # Diffusers pipeline (SD/SDXL/Flux)
│   ├── search_service.py    # Web search (DuckDuckGo)
│   └── pipelines/           # SD1.5, SDXL, Flux pipeline modules
│
├── providers/               # LLM & media providers
│   ├── __init__.py          # PROVIDER_REGISTRY (8 cloud providers)
│   ├── base_provider.py     # Abstract base class
│   ├── vox_provider.py      # Local GGUF engine wrapper
│   ├── gemini_provider.py   # Gemini (chat + image + video)
│   ├── openai_provider.py   # OpenAI (chat + DALL-E + Sora)
│   ├── openrouter_provider.py
│   ├── deepseek_provider.py
│   ├── kimi_provider.py     # Moonshot/Kimi
│   ├── mistral_provider.py
│   ├── xai_provider.py      # Grok
│   ├── zai_provider.py      # ZhipuAI/GLM
│   ├── ollama_provider.py
│   └── cloud_streamer.py    # RunPod GPU streaming
│
├── engine/                  # VoxAI local LLM engine
│   ├── vox_api.py           # llama.cpp wrapper with Vulkan
│   ├── chat_templates.py    # ChatML, Llama, Alpaca, etc.
│   ├── vox_core_chat.py     # Core chat logic
│   └── elastic_memory/      # Dynamic context, RAG, persistence
│
├── widgets/                 # PySide6 UI components
│   ├── model_selector_panel.py  # Tile grid, drill-down, favorites
│   ├── chat_display.py      # Message rendering
│   ├── message_bubble.py    # Message UI with thinking sections
│   ├── code_panel.py        # Persistent code extraction panel
│   ├── file_card.py         # File cards with copy/download
│   ├── input_bar.py         # Input with file upload
│   ├── image_gen_view.py    # Image/video generation UI
│   ├── settings_view.py     # Settings with provider management
│   ├── sidebar.py           # Navigation sidebar
│   └── sidebar_panels.py    # Sidebar sub-panels
│
├── gateway/                 # IronGate Web Gateway
│   ├── iron_host.py         # FastAPI server
│   ├── ai_bridge.py         # AI orchestration bridge
│   ├── templates/           # HTML (Aurora theme)
│   ├── static/css/themes/   # 10 color themes
│   ├── static/js/           # Web UI JavaScript
│   └── lib/                 # Security, admin, client gen
│
├── utils/                   # Utilities
│   ├── config_manager.py    # Config with migration & favorites
│   └── file_handler.py      # File processing
│
├── models/                  # Model storage
│   ├── llm/                 # GGUF chat models
│   ├── checkpoints/         # Diffusion checkpoints
│   ├── loras/               # LoRA adapters
│   ├── vae/                 # VAE models
│   └── text_encoders/       # CLIP & T5 encoders
│
└── data/                    # Conversation history & sessions
```

---

## Configuration

`config.json` manages all settings (copy from `config.example.json`):

```json
{
  "llm": {
    "providers": {
      "gemini":    { "api_key": "", "models": [] },
      "openai":    { "api_key": "", "models": [] },
      "openrouter": { "api_key": "", "models": [] },
      "deepseek":  { "api_key": "", "models": [] },
      "kimi":      { "api_key": "", "models": [] },
      "mistral":   { "api_key": "", "models": [] },
      "xai":       { "api_key": "", "models": [] },
      "zai":       { "api_key": "", "models": [] }
    },
    "local_model_dir": "models/llm",
    "cache_dir": "cache"
  },
  "cloud": {
    "runpod_api_key": "",
    "pod_id": ""
  },
  "image": {
    "checkpoint_dir": "models/checkpoints",
    "lora_dir": "models/loras",
    "output_dir": "outputs/images"
  },
  "favorites": []
}
```

API keys can also be set through the Settings UI — no manual JSON editing required.

---

## Dependencies

| Category | Packages |
|----------|----------|
| **GUI** | PySide6, Pillow, pygments, markdown |
| **AI** | diffusers, transformers, torch, llama-cpp-python |
| **Providers** | google-generativeai, google-genai, openai |
| **Search** | ddgs (DuckDuckGo) |
| **Memory** | msgpack, chromadb |
| **Gateway** | fastapi, uvicorn, pyngrok, jinja2 |
| **Utils** | psutil, colorama, paramiko |

---

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for the local AI community**

*Giving small models big capabilities*

</div>
