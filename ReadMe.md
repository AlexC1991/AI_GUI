# AI_GUI (VoxAI Orchestrator)

**AI_GUI** is a modern, dark-themed desktop application that unifies local AI workflows into a single interface. Built with **Python** and **PySide6**, it provides native image generation via Diffusers with AMD GPU support through ZLUDA.

## 🚀 Key Features

### 💬 Chat Interface
- **Multi-Provider Support:** Ollama (local) and Gemini (cloud)
- **Streaming Responses:** Real-time token streaming
- **Markdown Rendering:** Code syntax highlighting
- **Model Selection:** Dynamic model discovery from Ollama

### 🎨 Image Generation
- **Native Diffusers Backend:** No ComfyUI dependency required
- **Multi-Model Support:**
  - Stable Diffusion 1.5
  - Stable Diffusion 2.x
  - SDXL / Pony / Illustrious
  - **Flux** (including GGUF quantized for 8GB VRAM)
- **Customization:**
  - Custom VAE loading
  - LoRA stacking with strength controls
  - Custom text encoders (CLIP, T5 GGUF)
- **AMD GPU Optimized:** ZLUDA integration for RX 6000/7000 series

### 🖼️ Gallery
- Browse generated images
- View generation metadata (prompt, settings, seed)
- Organize outputs by date

### ⚙️ Settings
- API key management
- Model path configuration
- Hardware monitoring

## 🛠️ Installation

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

2. **Run the setup script:**
   ```bash
   # Windows
   setup.bat
   
   # Or manually:
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **For AMD GPUs, patch ZLUDA:**
   ```bash
   patch_zluda.bat
   ```

4. **Launch the application:**
   ```bash
   # Windows with ZLUDA
   run_gui.bat
   
   # Or directly:
   python main.py
   ```

## 📁 Model Setup

Place your models in the following directories:

```
AI_GUI/
├── models/
│   ├── checkpoints/     # Main model files (.safetensors, .ckpt, .gguf)
│   ├── loras/           # LoRA files
│   ├── vae/             # VAE files (.safetensors, .sft)
│   └── text_encoders/   # CLIP and T5 encoders (.safetensors, .gguf)
```

### Flux Setup (8GB VRAM)
For Flux on 8GB VRAM, use GGUF quantized models:

1. Download `flux1-schnell-Q4_K_S.gguf` from [city96/FLUX.1-schnell-gguf](https://huggingface.co/city96/FLUX.1-schnell-gguf)
2. Download `t5-v1_1-xxl-encoder-Q4_K_M.gguf` for text encoder
3. Download `clip_l.safetensors` for CLIP encoder
4. Place in respective `models/` subdirectories

## 📦 Dependencies

### Core
- **PySide6:** Qt6 GUI Framework
- **diffusers:** Image generation pipeline
- **transformers:** Text encoders (CLIP, T5)
- **torch:** PyTorch with CUDA/ZLUDA support
- **accelerate:** Memory optimization

### Providers
- **google-generativeai:** Gemini API
- **ollama:** Local LLM integration

### Utilities
- **psutil:** Hardware monitoring
- **Pillow:** Image processing
- **safetensors:** Model loading
- **gguf:** GGUF quantized model support

## 🎮 Supported Hardware

| GPU | VRAM | SD 1.5 | SDXL | Flux GGUF | Flux Full |
|-----|------|--------|------|-----------|-----------|
| RX 6600 | 8GB | ✅ | ✅ | ✅ | ❌ |
| RX 6700 XT | 12GB | ✅ | ✅ | ✅ | ✅ |
| RX 7900 XTX | 24GB | ✅ | ✅ | ✅ | ✅ |
| RTX 3060 | 12GB | ✅ | ✅ | ✅ | ✅ |
| RTX 4090 | 24GB | ✅ | ✅ | ✅ | ✅ |

## 🤝 Contributing

Contributions welcome! The project is actively developed with focus on:
- AMD GPU compatibility
- Low VRAM optimizations
- New model architecture support

## 📜 License

MIT License
