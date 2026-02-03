from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget
)
from PySide6.QtCore import QTimer
from widgets.sidebar import Sidebar
from widgets.chat_view import ChatView
from widgets.input_bar import InputBar
from widgets.code_panel import CodePanel
from widgets.settings_view import SettingsView
from widgets.image_gen_view import ImageGenView
from utils.config_manager import ConfigManager
from backend.chat_worker import ChatWorker
from utils.file_handler import FileHandler

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("VoxAI Orchestrator")
        self.resize(1400, 800)

        central = QWidget()
        self.setCentralWidget(central)

        # --- MAIN LAYOUT (3 Columns: Sidebar | Stack | CodePanel) ---
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. LEFT SIDEBAR
        self.sidebar = Sidebar()
        main_layout.addWidget(self.sidebar)

        # 2. CENTRAL AREA (STACKED WIDGET)
        self.stack = QStackedWidget()

        # --- PAGE 0: CHAT INTERFACE ---
        self.chat_container_widget = QWidget()
        chat_layout = QVBoxLayout(self.chat_container_widget)
        chat_layout.setContentsMargins(0, 0, 0, 0)

        self.chat_view = ChatView()

        input_container = QHBoxLayout()
        input_container.setContentsMargins(50, 0, 20, 20)
        self.input_bar = InputBar()
        input_container.addWidget(self.input_bar)

        chat_layout.addWidget(self.chat_view, 1)
        chat_layout.addLayout(input_container)

        self.stack.addWidget(self.chat_container_widget)  # Index 0

        # --- PAGE 1: SETTINGS INTERFACE ---
        self.settings_view = SettingsView()
        self.stack.addWidget(self.settings_view)  # Index 1

        # --- PAGE 2: IMAGE GEN INTERFACE ---
        self.image_gen_view = ImageGenView()
        self.stack.addWidget(self.image_gen_view)  # Index 2

        # Add Stack to Main Layout
        main_layout.addWidget(self.stack, 1)

        # 3. RIGHT CODE PANEL
        self.code_panel = CodePanel()
        main_layout.addWidget(self.code_panel)

        # --- CONNECTIONS ---
        self.file_handler = FileHandler(self)

        # Connect Input Bar
        self.input_bar.send_button.clicked.connect(self.handle_send)
        if hasattr(self.input_bar.input, 'return_pressed'):
            self.input_bar.input.return_pressed.connect(self.handle_send)

        # Initial Button Connections (For Default Chat Mode)
        if self.sidebar.upload_btn:
            self.sidebar.upload_btn.clicked.connect(self.handle_upload)
        if self.sidebar.clear_btn:
            self.sidebar.clear_btn.clicked.connect(self.handle_clear_chat)

        # CRITICAL: View Switching
        self.sidebar.mode_changed.connect(self.handle_mode_switch)
        
        # Initial Model Check
        self._init_ollama_watcher()

    def _init_ollama_watcher(self):
        """Start looking for models immediately."""
        print("[DEBUG] Initializing Ollama Watcher...")
        from backend.ollama_worker import OllamaWorker
        self.ollama_list_worker = OllamaWorker("list")
        self.ollama_list_worker.models_ready.connect(self.handle_ollama_list)
        self.ollama_list_worker.error.connect(self.handle_ollama_error)
        self.ollama_list_worker.start()

    def handle_mode_switch(self, mode):
        """Switches the central view based on sidebar selection."""
        if mode == "settings":
            self.stack.setCurrentIndex(1)
            self.code_panel.slide_out()
        elif mode == "chat":
            self.stack.setCurrentIndex(0)
        elif mode == "image":
            self.stack.setCurrentIndex(2)
            self.code_panel.slide_out()
        else:
            self.stack.setCurrentIndex(0)

        # --- RECONNECT DYNAMIC SIDEBAR BUTTONS ---
        try:
            self.sidebar.upload_btn.clicked.disconnect()
        except:
            pass
        try:
            self.sidebar.clear_btn.clicked.disconnect()
        except:
            pass

        if mode == "chat":
            if self.sidebar.upload_btn:
                self.sidebar.upload_btn.clicked.connect(self.handle_upload)
            if self.sidebar.clear_btn:
                self.sidebar.clear_btn.clicked.connect(self.handle_clear_chat)

        if mode == "image":
            panel = self.sidebar.current_panel
            if hasattr(panel, 'gen_btn'):
                try:
                    panel.gen_btn.clicked.disconnect()
                except Exception:
                    pass  # Ignore disconnect errors
                panel.gen_btn.clicked.connect(self.start_image_generation)
                 
            if hasattr(panel, 'abort_btn'):
                try:
                    panel.abort_btn.clicked.disconnect()
                except Exception:
                    pass
                panel.abort_btn.clicked.connect(self.abort_generation)

            if hasattr(panel, 'refresh_btn'):
                try:
                    panel.refresh_btn.clicked.disconnect()
                except Exception:
                    pass
                panel.refresh_btn.clicked.connect(self.image_gen_view.refresh_assets)

    def _setup_backend(self):
        """Initialize backend workers."""
        print("[DEBUG] Setting up backend workers...")
        from backend.image_worker import ImageWorker
        self.image_worker = ImageWorker()
        
        # Connect image signals
        self.image_worker.progress.connect(self._on_gen_progress_msg)
        self.image_worker.step_progress.connect(self._on_gen_step)
        self.image_worker.finished.connect(self._on_gen_finished)
        self.image_worker.error.connect(self._on_gen_error)
        self.image_worker.auth_required.connect(self._on_auth_required)
        
        self.chat_worker = ChatWorker()
        self.chat_worker.chunk_received.connect(self._on_chat_chunk)
        self.chat_worker.finished.connect(self._on_chat_finished)
        self.chat_worker.error.connect(self._on_chat_error)

    def handle_ollama_list(self, models):
        """Update sidebar with models."""
        print(f"[DEBUG] Main Window Received Models: {models}")
        panel = self.sidebar.current_panel
        if panel and hasattr(panel, 'set_local_models'):
            panel.set_local_models(models)
        else:
            print("[DEBUG] Sidebar Chat Panel not active or not found.")

    def handle_ollama_error(self, err):
        """Update sidebar to show error state."""
        print(f"[DEBUG] Main Window Received Ollama Error: {err}")
        panel = self.sidebar.current_panel
        if panel and hasattr(panel, 'set_local_models'):
            panel.set_local_models(["(Connection Error)"])

    def start_image_generation(self):
        """Gather params and start generation."""
        # Check backend
        if not hasattr(self, 'image_worker'):
            self._setup_backend()
            
        view = self.image_gen_view
        panel = self.sidebar.current_panel
        
        # UI State - Disable Sidebar Button
        if panel and hasattr(panel, 'gen_btn'):
            panel.gen_btn.setEnabled(False)
            if hasattr(panel, 'abort_btn'):
                panel.abort_btn.setEnabled(True)
        
        view.progress_bar.setValue(0)
        view.set_status("Starting generation...", "#4ade80")
        self.sidebar.update_status("processing")
        
        # Gather Params using helper methods
        prompt = view.get_positive_prompt()
        neg_prompt = view.get_negative_prompt()
        steps = view.get_steps()
        cfg = view.get_cfg_scale()
        seed = view.get_seed()
        model = view.get_selected_checkpoint()
        sampler = view.get_sampler()
        
        # Get resolution tuple directly (fixes the parsing error)
        width, height = view.get_resolution()
        
        # Get LoRAs
        loras = view.get_active_loras()
        lora_name = loras[0][0] if loras else None
        lora_weight = loras[0][1] if loras else 1.0

        # Get VAE & Text Encoders
        vae_name = view.get_active_vae()
        text_encoders = view.get_active_text_encoders()
        
        # Gather Paths from Config
        config = ConfigManager.load_config()
        img_cfg = config.get("image", {})
        ckpt_dir = img_cfg.get("checkpoint_dir", "models/checkpoints")
        lora_dir = img_cfg.get("lora_dir", "models/loras")
        vae_dir = img_cfg.get("vae_dir", "models/vae")
        text_enc_dir = img_cfg.get("text_encoder_dir", "models/text_encoders")
        out_dir = img_cfg.get("output_dir", "outputs/images")

        print(f"[DEBUG] Generation: {width}x{height}, {steps} steps, CFG {cfg}, Sampler: {sampler}")
        print(f"[DEBUG] VAE: {vae_name}, TextEncoders: {text_encoders}")

        # Start Worker
        self.image_worker.setup(
            prompt=prompt,
            negative_prompt=neg_prompt,
            model=model,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg,
            seed=seed,
            sampler=sampler,
            lora=lora_name,
            lora_weight=lora_weight,
            vae=vae_name,
            text_encoders=text_encoders,
            checkpoint_dir=ckpt_dir,
            lora_dir=lora_dir,
            vae_dir=vae_dir,
            text_encoder_dir=text_enc_dir,
            output_dir=out_dir
        )

        self.image_worker.start()
        
    def abort_generation(self):
        """Abort current generation."""
        if hasattr(self, 'image_worker'):
            self.image_worker.abort()
            self._reset_gen_ui()
            self.sidebar.update_status("aborted")
            self.image_gen_view.set_status("Generation aborted", "#f59e0b")

    def _on_gen_progress_msg(self, msg):
        self.image_gen_view.set_status(msg, "#4ade80")
        self.sidebar.update_status(f"Generating...")
    
    def _on_gen_step(self, step, total):
        self.image_gen_view.set_progress(step, total)
        self.image_gen_view.set_status(f"Step {step}/{total}", "#4ade80")
    
    def _on_gen_finished(self, path):
        self._reset_gen_ui()
        self.sidebar.update_status("idle")
        self.image_gen_view.set_status(f"✅ Saved: {path}", "#4ade80")
        
        # Display Image using helper method
        self.image_gen_view.show_image(path, 0)

    def _on_gen_error(self, err):
        self._reset_gen_ui()
        self.sidebar.update_status("error")
        self.image_gen_view.set_status(f"❌ {err}", "#ef4444")
        print(f"Generation Error: {err}")
    
    def _on_auth_required(self, model_name):
        """Handle HuggingFace authentication request."""
        self._reset_gen_ui()
        self.sidebar.update_status("auth required")
        self.image_gen_view.set_status("🔐 Authentication required", "#f59e0b")
        
        # Show auth dialog
        from widgets.hf_auth_dialog import request_hf_auth
        if request_hf_auth(self, model_name):
            # User authenticated, retry generation
            self.image_gen_view.set_status("Token saved - retrying...", "#4ade80")
            QTimer.singleShot(500, self.start_generation)
        else:
            self.image_gen_view.set_status("Authentication cancelled", "#888")

    def _reset_gen_ui(self):
        """Re-enable generation UI."""
        if self.sidebar.current_panel and hasattr(self.sidebar.current_panel, 'gen_btn'):
            self.sidebar.current_panel.gen_btn.setEnabled(True)
            if hasattr(self.sidebar.current_panel, 'abort_btn'):
                self.sidebar.current_panel.abort_btn.setEnabled(False)

    def handle_send(self):
        user_text = self.input_bar.input.toPlainText().strip()
        if not user_text:
            return
        
        # Add User Message
        self.chat_view.chat_display.add_message(user_text, "user")
        self.input_bar.input.clear()
        
        # Check backend
        if not hasattr(self, 'chat_worker'):
            self._setup_backend()
            
        # Get Settings from Sidebar/Config
        if not self.sidebar.current_panel or not hasattr(self.sidebar.current_panel, 'mode_combo'):
            mode = "Provider"
            model = "Gemini Pro"
        else:
            panel = self.sidebar.current_panel
            mode = panel.mode_combo.currentText()
            model = panel.model_combo.currentText()

        # Config for keys
        cfg = ConfigManager.load_config()
        llm_cfg = cfg.get("llm", {})
        
        provider_type = "Gemini" if "Gemini" in model or "Provider" in mode else "Ollama"
        if "Llama" in model or "Mistral" in model or "Gemma" in model or "qwen" in model.lower():
            provider_type = "Ollama"
             
        api_key = llm_cfg.get("api_key", "")
        
        # Set Status to "Thinking"
        self.sidebar.update_status("thinking")
        
        # Add the Thinking Bubble
        self.thinking_bubble = self.chat_view.chat_display.show_thinking()
        self.chat_view.chat_display.scroll_to_bottom()
        
        # Map UI Name to API ID
        model_map = {
            "Gemini Pro": "gemini-1.5-pro",
            "Llama 3": "llama3",
            "Ministral 8B": "ministral",
            "Mistral": "mistral",
            "Mistral:latest": "mistral:latest",
            "Gemma": "gemma",
            "GPT-4o": "gpt-4o", 
            "Claude 3.5": "claude-3-5-sonnet",
            "qwen2.5-coder:latest": "qwen2.5-coder:latest",
            "llama3.2:latest": "llama3.2:latest",
        }
        api_model = model_map.get(model, model)
        
        # Start Worker
        self.chat_worker.setup(
            provider_type=provider_type,
            model_name=api_model,
            api_key=api_key,
            prompt=user_text,
            history=self.chat_view.chat_display.get_history()[:-1], 
            system_prompt=None
        )
        self.chat_worker.start()

    def _on_chat_chunk(self, chunk):
        # Remove thinking bubble on first chunk
        if hasattr(self, 'thinking_bubble') and self.thinking_bubble:
            self.chat_view.chat_display.remove_bubble(self.thinking_bubble)
            self.thinking_bubble = None
            self.chat_view.chat_display.start_streaming_message()
            
        self.chat_view.chat_display.update_streaming_message(chunk)

    def _on_chat_finished(self):
        self.chat_view.chat_display.end_streaming_message()
        self.sidebar.update_status("idle")
        
        # --- CODE EXTRACTION LOGIC ---
        import re
        history = self.chat_view.chat_display.get_history()
        if not history:
            return
        
        last_msg = history[-1]
        if last_msg['role'] != 'assistant':
            return
        
        content = last_msg['content']
        extracted_count = 0
        
        def replacement_handler(match):
            nonlocal extracted_count
            lang = match.group(1)
            code = match.group(2)
            
            if not code or not code.strip():
                return match.group(0)

            lang = lang.strip() if lang else "text"
            lang = lang.lower()
            
            # Generate Filename
            ext = lang
            if lang == "python":
                ext = "py"
            elif lang == "javascript":
                ext = "js"
            elif lang == "c++":
                ext = "cpp"
            elif lang == "c#":
                ext = "cs"
            elif lang == "markdown":
                ext = "md"
            
            filename = f"script_{extracted_count}.{ext}"
            
            # Add to Code Panel
            self.code_panel.add_file(filename, code.strip())
            self.code_panel.slide_in()
            extracted_count += 1
            
            return f"\n> *[Code extracted to Utility Panel: {filename}]*\n"

        # Regex to find ```language ... ``` blocks
        new_content = re.sub(r"```(\w+)?\n(.*?)```", replacement_handler, content, flags=re.DOTALL)
        
        # Update UI if changes were made
        if extracted_count > 0:
            self.chat_view.chat_display.rewrite_last_message(new_content)
            self.sidebar.update_status(f"Extracted {extracted_count} snippets")

    def _on_chat_error(self, err):
        self.chat_view.chat_display.update_streaming_message(f"\n[Error: {err}]")
        self.chat_view.chat_display.end_streaming_message()
        self.sidebar.update_status("error")

    def handle_clear_chat(self):
        self.chat_view.chat_display.clear_chat()
        self.sidebar.update_status("idle")
        self.code_panel.slide_out()

    def handle_upload(self):
        file_path = self.file_handler.open_file_dialog()
        if file_path:
            file_data = self.file_handler.process_file(file_path)
            
            content = ""
            if file_data['extension'] in ['.py', '.txt', '.json', '.js', '.md', '.csv']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    self.code_panel.add_file(file_data['name'], content)
                except Exception as e:
                    print(f"Error reading file: {e}")

            # Inject into Chat History with nice UI but full context
            if content:
                context_msg = f"I am uploading a file named '{file_data['name']}'.\n\nCONTENT:\n```\n{content}\n```\n\nPlease analyze this file."
                display_msg = f"📂 Uploaded: {file_data['name']} ({file_data['size']})"
                self.chat_view.chat_display.add_message(context_msg, "user", display_text=display_msg)
            else:
                self.chat_view.chat_display.add_message(f"📂 Uploaded: {file_data['name']}", "user")

            # Simulate AI acknowledgement
            QTimer.singleShot(1000, lambda: self.finalize_ai_response(
                f"I've loaded **{file_data['name']}** into the context."
            ))

    def finalize_ai_response(self, text):
        """Adds a simple AI message to the chat (used for system events)."""
        self.chat_view.chat_display.add_message(text, "ai")

    def closeEvent(self, event):
        """Cleanup workers on exit to prevent crash."""
        print("[System] Shutting down...")
        
        # Stop Chat Worker
        if hasattr(self, 'chat_worker') and self.chat_worker.isRunning():
            print("Stopping Chat Worker...")
            self.chat_worker.quit()
            self.chat_worker.wait(1000)
            
        # Stop Image Worker
        if hasattr(self, 'image_worker') and self.image_worker.isRunning():
            print("Stopping Image Worker...")
            self.image_worker.quit()
            self.image_worker.wait(1000)
            
        # Stop Ollama List Worker
        if hasattr(self, 'ollama_list_worker') and self.ollama_list_worker.isRunning():
            print("Stopping Ollama Worker...")
            self.ollama_list_worker.quit()
            self.ollama_list_worker.wait(1000)
            
        super().closeEvent(event)
