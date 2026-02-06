# VoxAI Orchestrator

```
 ##     ##  #######  ##     ##    ###    ####
 ##     ## ##     ##  ##   ##    ## ##    ##
 ##     ## ##     ##   ## ##    ##   ##   ##
 ##     ## ##     ##    ###    ##     ##  ##
  ##   ##  ##     ##   ## ##   #########  ##
   ## ##   ##     ##  ##   ##  ##     ##  ##
    ###     #######  ##     ## ##     ## ####
```

A modern, dark-themed desktop application that unifies local AI workflows into a single interface. Built with **Python** and **PySide6**, featuring native image generation, local LLM chat, web-based remote access, and AMD GPU acceleration through ZLUDA.

---

## Features

### Chat Interface
- **Multi-Provider Support** - VoxAI (local LLM via llama.cpp), Ollama, and Gemini (cloud)
- **Streaming Responses** - Real-time token-by-token output with speed stats
- **Code Detection** - Extracts code blocks into a dedicated code panel with syntax highlighting
- **File Uploads** - Attach files to chat context for AI analysis
- **Web Search** - DuckDuckGo integration injects live search results into AI context

### Image Generation
- **Native Diffusers Backend** - No ComfyUI dependency
- **Multi-Model Support** - SD 1.5, SD 2.x, SDXL, Pony, Illustrious, Flux (including GGUF quantized for 8GB VRAM)
- **Customization** - Custom VAE, LoRA stacking with strength controls, custom text encoders (CLIP, T5 GGUF)
- **AMD GPU Optimized** - ZLUDA integration for RX 6000/7000 series

### VoxAI Chat Engine
- **Local LLM inference** via llama.cpp with Vulkan acceleration
- **Multi-model support** - Hot-swap between any GGUF model
- **Chat templates** - Automatic prompt formatting per model architecture
- **Zero dependencies** - No Ollama or external servers required

### IronGate Web Gateway
- **Remote Access** - Expose your local AI through an ngrok tunnel with custom domain
- **Client System** - Generate standalone client executables with activation codes
- **Security** - IP banning, rate limiting, session management, admin console
- **Desktop Service** - Lightweight local API (port 8001) for search and file uploads
- **Web UI** - Full-featured browser interface with chat, search, and file upload

### Gallery
- Browse generated images with metadata (prompt, settings, seed)
- Organize outputs by date

### Settings
- API key management
- Model path configuration
- Hardware monitoring
- IronGate service controls (Desktop Service + Web Gateway)

---

## Project Structure

```
AI_GUI/
|-- main.py                    # Application entry point
|-- main_window.py             # Main window orchestrator
|-- bootstrap.py               # Temp/cache directory redirect
|-- run_gui.bat                # Launch script with pre-flight checks
|
|-- backend/
|   |-- chat_worker.py         # Chat thread (streaming + providers)
|   |-- image_worker.py        # Image generation thread
|   |-- image_generator.py     # Diffusers pipeline manager
|   |-- search_service.py      # DuckDuckGo web search client
|   |-- ollama_worker.py       # Ollama provider thread
|   |-- vox_model_worker.py    # VoxAI model operations
|   |-- gguf_converter.py      # GGUF model utilities
|   |-- cleanup.py             # Resource cleanup
|   '-- debug.py               # Debug utilities
|
|-- widgets/
|   |-- chat_view.py           # Chat tab layout
|   |-- chat_display.py        # Message rendering area
|   |-- message_bubble.py      # Individual message bubbles
|   |-- input_bar.py           # Chat input with attachments
|   |-- code_panel.py          # Code viewer/editor panel
|   |-- file_card.py           # File attachment cards
|   |-- image_gen_view.py      # Image generation tab
|   |-- settings_view.py       # Settings tab + service controls
|   |-- sidebar.py             # Navigation sidebar
|   |-- sidebar_panels.py      # Sidebar panel contents
|   |-- model_manager_dialog.py # Model download/management
|   '-- hf_auth_dialog.py      # HuggingFace auth dialog
|
|-- providers/
|   |-- base_provider.py       # Abstract provider interface
|   |-- vox_provider.py        # VoxAI (local llama.cpp)
|   |-- ollama_provider.py     # Ollama integration
|   '-- gemini_provider.py     # Google Gemini API
|
|-- utils/
|   |-- config_manager.py      # Configuration management
|   |-- file_handler.py        # File operations
|   '-- ollama_helper.py       # Ollama utilities
|
|-- VoxAI_Chat_API/
|   |-- vox_api.py             # Core LLM API (llama.cpp wrapper)
|   |-- vox_core_chat.py       # Chat completion engine
|   |-- chat_templates.py      # Model-specific prompt templates
|   |-- machine_engine_handshake.py # Hardware detection
|   |-- model_preflight.py     # Model validation
|   '-- models/                # GGUF model files (not tracked)
|
|-- Vox_IronGate/
|   |-- iron_host.py           # Web gateway (FastAPI + ngrok)
|   |-- iron_desktop.py        # Desktop search/upload service
|   |-- ai_bridge.py           # Connects IronGate to VoxAI engine
|   |-- lib/
|   |   |-- admin.py           # Admin console commands
|   |   |-- client_gen.py      # Client package generator
|   |   |-- config.py          # IronGate configuration
|   |   |-- database.py        # Client database
|   |   '-- security.py        # IP banning, rate limiting
|   |-- templates/             # HTML templates (login, app, waiting)
|   '-- static/                # CSS + JavaScript for web UI
|
|-- models/
|   |-- checkpoints/           # Diffusion model files (not tracked)
|   '-- loras/                 # LoRA files (not tracked)
|
'-- outputs/
    '-- images/                # Generated images
```

---

## Installation

### Prerequisites
- Python 3.10+
- AMD GPU with ZLUDA (or NVIDIA with CUDA)
- 8GB+ VRAM recommended

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AlexC1991/AI_GUI.git
   cd AI_GUI
   ```

2. **Set up the environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure:**
   ```bash
   copy config.example.json config.json
   # Edit config.json with your API keys
   ```

4. **Add GGUF models** (for local AI chat):
   - Place `.gguf` model files in `VoxAI_Chat_API/models/`
   - Any GGUF model works (Llama, Mistral, Qwen, Phi, etc.)

5. **Launch:**
   ```bash
   run_gui.bat
   ```
   The launcher runs 9 pre-flight checks (DLLs, models, GPU, cache dirs) then auto-starts.

### DLL Setup (for local LLM)
The following DLLs are required in the project root for Vulkan-accelerated LLM inference:
- `llama.dll`
- `ggml.dll`
- `ggml-vulkan.dll`
- `ggml-cpu-haswell.dll`

Build from [llama.cpp](https://github.com/ggerganov/llama.cpp) with Vulkan backend, or use pre-built releases.

---

## Model Setup

### Chat Models (GGUF)
Place any GGUF model in `VoxAI_Chat_API/models/`:
```
VoxAI_Chat_API/models/
  Llama-3.2-3B-Instruct-Q4_K_M.gguf
  Qwen3-8B-Q5_K_M.gguf
  Mistral-Nemo-12B-Instruct.Q4_K_M.gguf
```

### Image Generation Models
```
models/
  checkpoints/    # .safetensors, .ckpt, .gguf diffusion models
  loras/          # LoRA files
  vae/            # VAE files
  text_encoders/  # CLIP and T5 encoder files
```

### Flux on 8GB VRAM
Use GGUF quantized models:
1. `flux1-schnell-Q4_K_S.gguf` from [city96/FLUX.1-schnell-gguf](https://huggingface.co/city96/FLUX.1-schnell-gguf)
2. `t5-v1_1-xxl-encoder-Q4_K_M.gguf` for text encoder
3. `clip_l.safetensors` for CLIP encoder

---

## IronGate Remote Access

IronGate lets you access your local AI from anywhere through a web browser.

### Setup
1. Get an [ngrok](https://ngrok.com) auth token (free or paid for custom domain)
2. Create `Vox_IronGate/host_config.json`:
   ```json
   {
       "ngrok_token": "YOUR_TOKEN",
       "ngrok_domain": "your-domain.ngrok.app"
   }
   ```
3. Start the gateway: `python Vox_IronGate/iron_host.py`
4. Share the Magic Host Link with clients

### Client Generation
From the admin console:
```
Admin> gen "Friend Name"
```
Generates a standalone `.exe` + activation code + `.zip` package in `exports/`.

---

## Supported Hardware

| GPU | VRAM | SD 1.5 | SDXL | Flux GGUF | Flux Full |
|-----|------|--------|------|-----------|-----------|
| RX 6600 | 8GB | Yes | Yes | Yes | No |
| RX 6700 XT | 12GB | Yes | Yes | Yes | Yes |
| RX 7900 XTX | 24GB | Yes | Yes | Yes | Yes |
| RTX 3060 | 12GB | Yes | Yes | Yes | Yes |
| RTX 4090 | 24GB | Yes | Yes | Yes | Yes |

---

## Dependencies

### Core
- **PySide6** - Qt6 GUI framework
- **diffusers** - Image generation pipelines
- **transformers** - Text encoders (CLIP, T5)
- **torch** - PyTorch with CUDA/ZLUDA
- **accelerate** - Memory optimization
- **llama-cpp-python** - Local LLM inference

### Providers
- **google-generativeai** - Gemini API
- **ollama** - Local LLM integration
- **ddgs** - DuckDuckGo web search

### Web Gateway
- **fastapi** - API framework
- **uvicorn** - ASGI server
- **pyngrok** - ngrok tunnel management
- **pyinstaller** - Client executable building

### Utilities
- **psutil** - Hardware monitoring
- **Pillow** - Image processing
- **safetensors** - Model loading
- **gguf** - GGUF model support
- **colorama** - Terminal colors

---

## License

MIT License
