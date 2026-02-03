# Project Overview: AI_GUI (VoxAI Orchestrator)

## 🎯 The Vision

AI_GUI is the "Operating System" for local AI. Instead of juggling multiple applications (ComfyUI for images, Ollama for chat, separate code editors), AI_GUI brings them into a single, native, high-performance desktop application optimized for AMD GPUs via ZLUDA.

## 🏗 Architecture

The project uses a **Modular Frontend** with a **Native Backend** approach.

### Frontend
- **Framework:** PySide6 (Qt6) desktop application
- **Layout:** Sidebar navigation with switchable views (Chat, Image Gen, Gallery, Settings)
- **Theme:** Dark mode with modern styling

### Backend (Implemented)
- **Chat:** 
  - Ollama integration (local LLMs)
  - Gemini API support (cloud)
  - Real-time streaming responses
  
- **Image Generation:**
  - Native Diffusers backend (no ComfyUI dependency)
  - Multi-model support: SD 1.5, SD 2.x, SDXL, Pony, Flux
  - GGUF quantized model support for low VRAM
  - ZLUDA optimization for AMD GPUs
  - Custom VAE, LoRA, and Text Encoder support
  - CPU offload for 8GB VRAM systems

- **Hardware:**
  - AMD GPU support via ZLUDA
  - Real-time VRAM/RAM monitoring
  - Automatic model family detection
  - Memory-optimized loading strategies

## 🎮 Supported Models

| Model Type | VRAM Required | Status |
|------------|---------------|--------|
| SD 1.5 | ~4GB | ✅ Working |
| SD 2.x | ~5GB | ✅ Working |
| SDXL | ~6GB | ✅ Working |
| Pony (SDXL-based) | ~6GB | ✅ Working |
| Flux (GGUF Q4) | ~8GB | ✅ Working |
| Flux (safetensors) | 12GB+ | ✅ Working |

## 🛣 Roadmap

### Phase 1: UI & UX (✅ Completed)
- [x] Dark Mode Theme
- [x] Chat Interface with Code Highlighting
- [x] Image Gen Interface with LoRA Stacking
- [x] Global Settings Menu
- [x] Navigation & View Switching
- [x] Gallery with metadata display

### Phase 2: Backend Integration (✅ Completed)
- [x] Ollama LLM integration
- [x] Gemini API integration
- [x] Native Diffusers image generation
- [x] Multi-model family support (SD1.5, SDXL, Pony, Flux)
- [x] ZLUDA/AMD GPU optimization
- [x] Custom VAE/LoRA/Text Encoder loading
- [x] GGUF quantized model support

### Phase 3: Advanced Features (🚧 In Progress)
- [x] GGUF Flux transformer support (low VRAM)
- [x] Automatic dtype handling (bfloat16/float16)
- [ ] ControlNet support
- [ ] Inpainting/Outpainting
- [ ] Batch generation
- [ ] Prompt queue system

### Phase 4: Future Enhancements
- [ ] Remote Server Mode (network control)
- [ ] Voice Mode (STT/TTS)
- [ ] Video generation support
- [ ] Plugin system for custom nodes
