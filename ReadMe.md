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

*Local LLM Chat • Image Generation • Agentic Web Search • Remote Access*

</div>

---

## ✨ What Makes VoxAI Different?

VoxAI isn't just another AI GUI—it's an **agentic framework** that gives local models superpowers:

- 🧠 **Metacognition** — Your AI understands its own knowledge limitations
- 🔍 **Autonomous Search** — AI decides when to search the web, no manual triggers
- 🖼️ **Image Generation** — Native Diffusers with SDXL, Flux, and 8GB VRAM support
- 🌐 **Remote Access** — Share your local AI through secure web tunnels
- 💰 **Budget-Friendly** — Runs on < $500 AUD hardware

> *"I gave Qwen3-8B internet access and it achieved metacognition"* — Creator

---

## 🚀 Features

### Chat Interface
| Feature | Description |
|---------|-------------|
| **Multi-Provider** | VoxAI (local), Ollama, Gemini (cloud) |
| **Streaming** | Real-time token-by-token with speed stats |
| **Code Detection** | Auto-extracts code to syntax-highlighted panel |
| **File Uploads** | Attach files for AI analysis |
| **Thinking Models** | Full support for `<think>` reasoning blocks |

### Agentic Search (Metacognition)
| Feature | Description |
|---------|-------------|
| **Autonomous** | AI decides when it needs current information |
| **Self-Aware** | Understands its knowledge cutoff |
| **Failsafe** | Catches refusals and auto-searches |
| **Natural** | Just ask normally—the AI handles the rest |

### Image Generation
| Feature | Description |
|---------|-------------|
| **Native Backend** | No ComfyUI dependency |
| **Models** | SD 1.5, SDXL, Pony, Illustrious, Flux |
| **8GB VRAM** | GGUF quantized Flux support |
| **AMD Optimized** | ZLUDA for RX 6000/7000 series |

### VoxAI Chat Engine
| Feature | Description |
|---------|-------------|
| **Local LLM** | llama.cpp with Vulkan acceleration |
| **Hot-Swap** | Switch models without restart |
| **Auto-Templates** | Correct prompting per architecture |
| **Zero Dependencies** | No Ollama or external servers |

### IronGate Web Gateway
| Feature | Description |
|---------|-------------|
| **Remote Access** | ngrok tunnel with custom domain |
| **Client Generation** | Standalone `.exe` with activation codes |
| **Security** | IP banning, rate limiting, sessions |
| **Web UI** | Full chat interface in browser |

---

## 📦 Quick Start

### Prerequisites
- Python 3.10+
- AMD GPU with ZLUDA (or NVIDIA with CUDA)
- 8GB+ VRAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/AlexC1991/AI_GUI.git
cd AI_GUI

# Set up environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Configure
copy config.example.json config.json
# Edit config.json with your API keys (optional)

# Add a GGUF model
# Place any .gguf file in VoxAI_Chat_API/models/

# Launch
run_gui.bat
```

### DLL Setup (Local LLM)
For Vulkan-accelerated inference, place these in the project root:
- `llama.dll` / `ggml.dll` / `ggml-vulkan.dll` / `ggml-cpu-haswell.dll`

Build from [llama.cpp](https://github.com/ggerganov/llama.cpp) with Vulkan, or use pre-built releases.

---

## 🎯 Model Setup

### Chat Models (GGUF)
```
VoxAI_Chat_API/models/
  ├── Qwen3-8B-Q5_K_M.gguf
  ├── Llama-3.2-3B-Instruct-Q4_K_M.gguf
  └── Any GGUF model works!
```

### Image Models
```
models/
  ├── checkpoints/    # Diffusion models
  ├── loras/          # LoRA files
  ├── vae/            # VAE files
  └── text_encoders/  # CLIP/T5 encoders
```

### Flux on 8GB VRAM
1. `flux1-schnell-Q4_K_S.gguf` from [city96/FLUX.1-schnell-gguf](https://huggingface.co/city96/FLUX.1-schnell-gguf)
2. `t5-v1_1-xxl-encoder-Q4_K_M.gguf` for text encoder
3. `clip_l.safetensors` for CLIP

---

## 🌐 IronGate Remote Access

Access your local AI from anywhere through a secure web tunnel.

### Setup
1. Get an [ngrok](https://ngrok.com) auth token
2. Create `Vox_IronGate/host_config.json`:
   ```json
   {
       "ngrok_token": "YOUR_TOKEN",
       "ngrok_domain": "your-domain.ngrok.app"
   }
   ```
3. Start: `python Vox_IronGate/iron_host.py`
4. Share the Magic Host Link!

### Generate Client Packages
```
Admin> gen "Friend Name"
```
Creates standalone `.exe` + activation code in `exports/`.

---

## 💻 Supported Hardware

| GPU | VRAM | SD 1.5 | SDXL | Flux GGUF | Flux Full |
|-----|------|:------:|:----:|:---------:|:---------:|
| RX 6600 | 8GB | ✅ | ✅ | ✅ | ❌ |
| RX 6700 XT | 12GB | ✅ | ✅ | ✅ | ✅ |
| RX 7900 XTX | 24GB | ✅ | ✅ | ✅ | ✅ |
| RTX 3060 | 12GB | ✅ | ✅ | ✅ | ✅ |
| RTX 4090 | 24GB | ✅ | ✅ | ✅ | ✅ |

---

## 📁 Project Structure

```
AI_GUI/
├── main.py                 # Entry point
├── main_window.py          # Main orchestrator
├── backend/                # Workers & services
│   ├── chat_worker.py      # Chat streaming
│   ├── image_worker.py     # Image generation
│   └── search_service.py   # Web search
├── widgets/                # UI components
├── providers/              # LLM providers
├── VoxAI_Chat_API/         # Local LLM engine
├── Vox_IronGate/           # Web gateway
└── models/                 # Model files
```

---

## 🛠️ Dependencies

| Category | Packages |
|----------|----------|
| **GUI** | PySide6 |
| **AI** | diffusers, transformers, torch, llama-cpp-python |
| **Providers** | google-generativeai, ollama, ddgs |
| **Web** | fastapi, uvicorn, pyngrok |
| **Utils** | psutil, Pillow, safetensors, gguf |

---

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for the local AI community**

*Giving small models big capabilities*

</div>
