# VoxAI Orchestrator
## Current Implementation State

This document describes the **current, implemented state** of the VoxAI Orchestrator project.
It is intended for developers and AI-assisted contributors (e.g. Opus).

---

## 1. Project Status

**Phase:** UI Skeleton / Environment Bootstrap  
**Backend:** Not implemented  
**AI Providers:** Not implemented

The project currently focuses on:
- Desktop UI layout
- Dependency bootstrapping
- Development environment stability

---

## 2. What Exists Right Now

### UI
- Framework: **PySide6 (Qt)**
- Native desktop window
- Fixed sidebar layout
- Central chat display area
- Bottom input bar with Send button

No business logic is connected yet.

---

### Dependency Handling
- Dependencies are declared in `main.py`
- On startup:
    - The app checks for required Python modules
    - Missing modules are installed automatically using `pip`
- This is a **temporary bootstrap system**, not the final JIT installer

---

### Entry Point
- Application entry point: `main.py`
- Uses the currently active Python interpreter
- Designed to work in:
    - PyCharm Debug
    - Windows CMD
    - Linux terminal

---

## 3. What Does NOT Exist Yet

The following systems are **explicitly not implemented**:

- Local LLM execution (Ollama / llama.cpp)
- Image generation (Diffusers / ONNX / ROCm)
- Cloud providers (Gemini API)
- Remote node support (FastAPI)
- Persistent memory
- Task orchestration
- Character presets (DJ BATTYVOX)

These will be added in later phases.

---

## 4. Development Rules (Important)

- Do NOT redesign the UI layout
- Do NOT block the Qt main thread
- All future inference must run in background threads (`QThread`)
- Treat the current UI as the layout contract

---

## 5. Immediate Next Steps

Planned next implementation steps:

1. Convert chat display to message widgets
2. Wire Send button to message append logic
3. Add Image Generation panel using same layout shell
4. Replace bootstrap dependency logic with a full JIT installer

---

## 6. Notes for AI-Assisted Contributors

- Assume this project is intentionally incomplete
- Do not infer missing features
- Do not install heavy AI dependencies unless instructed
- Follow existing structure exactly

This document represents the current truth of the codebase.
