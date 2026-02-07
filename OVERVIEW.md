# VoxAI Orchestrator - Technical Overview

## 🎯 High Level Architecture

VoxAI Orchestrator is designed as a modular, local-first AI operating system. It moves away from the fragmented ecosystem of separate tools (Ollama for chat, ComfyUI for images) into a single, unified PyQt6 application.

The core philosophy is **Native Implementation**:
- **No API wrappers around other local servers** (where possible).
- **Direct integration** with `llama.cpp` for text via `llama-cpp-python`.
- **Direct integration** with `Diffusers` for images.
- **Direct integration** with `DuckDuckGo` for search.

This reduces overhead, simplifies dependency management, and allows for deeper integration like shared memory management and unified hardware monitoring.

## 🏗 Core Components

### 1. The Orchestrator (`main_window.py`)
The central nervous system of the application. It handles:
- **Thread Management:** Spawning and monitoring `QThread` workers for Chat, Image Gen, and Search.
- **State Management:** Tracking conversation history, model states, and hardware utilization.
- **Signal Routing:** Connecting UI events to backend workers and vice-versa.

### 2. VoxAI Chat Engine (`VoxAI_Chat_API/`)
A custom local inference engine built on top of `llama-cpp-python`.
- **Hardware Handshake:** Automatically detects NVIDIA (CUDA), AMD (ZLUDA/Vulkan), or CPU capabilities on startup.
- **Model Hot-Swapping:** Unloads/reloads models dynamically to manage VRAM.
- **Thinking Blocks:** Parses `<think>` tags from reasoning models and formats them into collapsible UI elements.
- **Context Management:** Handles sliding windows and context limits automatically.

### 3. Agentic Search & Metacognition
Instead of traditional RAG (Retrieval Augmented Generation), VoxAI uses an **Agentic** approach.
- **Metacognition:** The system prompt instructs the AI to recognize its own knowledge cutoffs.
- **Autonomous Trigger:** The AI decides *when* to search based on the user prompt. No explicit `[Search]` command is required from the user.
- **Failsafe:** If the AI refuses a task due to lack of real-time info, the system intercepts the refusal and triggers a search automatically.
- **Desktop Service:** Runs a lightweight local API (`iron_desktop.py`) to handle sanitized web searches.

### 4. Image Generation Backend (`backend/image_generator.py`)
A custom implementation of the HuggingFace `Diffusers` library.
- **Pipeline Strategy:** Uses a Singleton pattern to keep models loaded only when needed.
- **VRAM Optimization:** Aggressive offloading and `bfloat16`/`float16` management.
- **Model Support:**
    - **SD 1.5 / SD 2.1:** Standard pipelines.
    - **SDXL / Pony:** `StableDiffusionXLPipeline` with specialized schedulers.
    - **Flux.1:** Supports `FluxPipeline` and **GGUF Quantized Flux** models (allowing Flux on 8GB VRAM).
- **AMD ZLUDA:** Special initialization paths to enable CUDA-like performance on AMD ROCm via ZLUDA.

### 5. IronGate Remote Access (`Vox_IronGate/`)
A secure gateway for remote access.
- **Tunneling:** Uses `ngrok` to tunnel localhost to a secure public URL.
- **Security:**
    - IP Banning & Rate Limiting.
    - **Client Identity:** Generated `.exe` clients are cryptographically bound to the host.
    - **Session Approval:** Manual admin approval required for new browser sessions.

## 📂 Key Directory Structure

```
AI_GUI/
├── main.py                 # Application Entry Point
├── main_window.py          # Core Orchestration Logic
├── database.py             # SQLite Storage for Chats/Settings
├── config.json             # User Preferences
├── backend/                # Heavy Lifting Workers
│   ├── chat_worker.py      # LLM Inference Thread
│   ├── image_worker.py     # Diffusion Generation Thread
│   ├── search_service.py   # Web Search Interface
│   └── image_generator.py  # Diffusers Pipeline Manager
├── VoxAI_Chat_API/         # Local LLM Engine
│   ├── vox_api.py          # Llama.cpp Wrapper
│   ├── machine_engine...   # Hardware Detection Logic
│   └── models/             # GGUF Model Storage
├── Vox_IronGate/           # Remote Access System
│   ├── iron_host.py        # Tunnel Host
│   ├── iron_desktop.py     # Local Search API
│   └── security.py         # Auth & Encryption
├── widgets/                # UI Components
│   ├── chat_view.py        # Chat Tab
│   ├── image_gen_view.py   # Image Gen Tab
│   ├── code_panel.py       # Syntax Highlighting
│   └── ...
└── outputs/                # Generated Content
```

## 🛠️ Development Practices

- **Qt Main Thread Safety:** All heavy computation (Checking models, generation, networking) MUST happen in `QThread` workers. Never block `main_window.py`.
- **Config persistence:** All user settings are saved immediately to `config.json`.
- **Error Handling:** Workers emit `error` signals which are caught by the UI to display toast notifications, preventing crashes.
