\# Project Overview: VoxAI Orchestrator



\## 🎯 The Vision

VoxAI Orchestrator aims to be the "Operating System" for local AI. Instead of opening 5 different browser tabs (one for ChatGPT, one for Automatic1111, one for a Code Editor), VoxAI brings them into a single, native, high-performance desktop application.



\## 🏗 Architecture



The project is designed with a \*\*Modular Frontend\*\* and a \*\*Pluggable Backend\*\*.



\* \*\*Frontend (Current State):\*\* A PySide6 desktop application that handles all user interaction, layout, and visualization.

\* \*\*Backend (Planned):\*\*

&nbsp;   \* \*\*Chat:\*\* Will connect via API to local endpoints (Ollama, LM Studio) or Cloud Providers (OpenAI/Anthropic).

&nbsp;   \* \*\*Images:\*\* Will act as a frontend for \*\*ComfyUI\*\*. The "Node Graph" will run headless in the background, while VoxAI provides the user-friendly "Rack" interface.

&nbsp;   \* \*\*Audio:\*\* Planned integration with RVC (Voice Conversion) and Suno/Bark models.



\## 🛣 Roadmap



\### Phase 1: UI \& UX (✅ Completed)

\* \[x] Dark Mode Theme.

\* \[x] Chat Interface with Code Highlighting.

\* \[x] Image Gen Interface with LoRA Stacking.

\* \[x] Global Settings Menu.

\* \[x] Navigation \& View Switching.



\### Phase 2: Functionality Wiring (🚧 Next Steps)

\* \[ ] \*\*Settings Persistence:\*\* Save paths and API keys to `config.json` or `.env`.

\* \[ ] \*\*Local LLM Hookup:\*\* Connect Chat window to an Ollama instance.

\* \[ ] \*\*Image Gen Hookup:\*\* Send API requests to a local ComfyUI instance to generate real images.

\* \[ ] \*\*File Handling:\*\* Allow drag-and-drop uploads for RAG (Retrieval Augmented Generation).



\### Phase 3: Advanced Features

\* \[ ] \*\*Remote Server Mode:\*\* Allow the GUI to run on a powerful desktop while being controlled by a laptop over the network.

\* \[ ] \*\*Voice Mode:\*\* Real-time speech-to-text and text-to-speech conversation.

