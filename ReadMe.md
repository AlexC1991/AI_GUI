# VoxAI Orchestrator

**VoxAI Orchestrator** is a modern, dark-themed GUI designed to unify various local AI workflows into a single, cohesive interface. Built with **Python** and **PySide6**, it serves as a central hub for LLM Chat, Image Generation, and System Monitoring.

![VoxAI Interface](https://via.placeholder.com/800x450.png?text=VoxAI+Interface+Preview) 
*(Replace with a real screenshot link after uploading)*

## 🚀 Key Features

* **Unified Chat Interface:**
    * Markdown support with syntax highlighting for code blocks.
    * "Discord-style" message bubbles.
    * Sidebar controls for Model selection, Memory settings, and Token limits.
* **Image Generation Studio:**
    * **Split-View Design:** Prompts on the left, visual output on the right.
    * **Stackable LoRA Rack:** A layer-based system to stack multiple LoRAs with individual strength sliders.
    * **Real-time System Stats:** Live monitoring of VRAM, GPU Load, and RAM usage.
* **Global Settings:**
    * Centralized configuration for API keys (OpenAI, Gemini, Anthropic).
    * Path management for Local LLMs, Checkpoints, and LoRAs.
    * Remote Access configuration for hosting the UI as a server.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AlexC1991/AI_GUI.git](https://github.com/AlexC1991/AI_GUI.git)
    cd AI_GUI
    ```

2.  **Create a Virtual Environment (Optional but recommended):**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Run the Application:**
    The application handles its own dependencies on the first run.
    ```bash
    python main.py
    ```

## 📦 Dependencies

* **PySide6:** Core GUI Framework.
* **psutil:** Real-time hardware monitoring (CPU/RAM).
* **markdown:** For rendering chat text formatting.
* **pygments:** For IDE-style code syntax highlighting.

## 🤝 Contributing

This is currently a UI Mockup / Prototype phase. The backend logic (connecting to ComfyUI, Ollama, etc.) is in active development. Pull requests are welcome!

## 📜 License

MIT License.