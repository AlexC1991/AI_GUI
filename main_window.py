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
from backend.search_service import SearchService
from widgets.settings_view import DesktopServiceThread
from utils.file_handler import FileHandler
import os
import re as _re

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("VoxAI Orchestrator")
        self.resize(1400, 800)

        # Thinking model state
        self._in_thinking = False
        self._thinking_buffer = ""
        self._thinking_section = None

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
        self.search_service = SearchService()

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
        
        # Session Management (Elastic Memory)
        import time
        self.current_session_id = f"session_{int(time.time())}"
        print(f"[MainWindow] Persistent Session ID: {self.current_session_id}")

        # Auto-start Iron Desktop Service (local search/upload API on port 8001)
        self._start_desktop_service()

    @staticmethod
    def _kill_port(port):
        """Kill any process listening on the given port (Windows)."""
        import subprocess, sys as _sys
        if _sys.platform != "win32":
            return
        try:
            # Get PID(s) on this port
            r = subprocess.run(
                ['powershell.exe', '-NoProfile', '-Command',
                 f'(Get-NetTCPConnection -LocalPort {port} -State Listen '
                 f'-ErrorAction SilentlyContinue).OwningProcess'],
                capture_output=True, text=True, timeout=5
            )
            pids = [p.strip() for p in r.stdout.strip().split('\n') if p.strip().isdigit()]
            if not pids:
                return  # Port is free
            my_pid = str(os.getpid())
            for pid in pids:
                if pid == my_pid:
                    continue
                print(f"[System] Killing stale process PID {pid} on port {port}...")
                # Try taskkill (works across sessions)
                subprocess.run(
                    ['taskkill', '/F', '/PID', pid],
                    capture_output=True, text=True, timeout=5
                )
            import time as _time
            _time.sleep(0.5)
        except Exception as e:
            print(f"[System] Port {port} cleanup error: {e}")

    def _start_desktop_service(self):
        """Auto-start Iron Desktop Service (port 8001) for search & file APIs."""
        # Kill any stale process on port 8001 first (from previous crashed session etc.)
        self._kill_port(8001)

        print("[System] Starting Iron Desktop Service (port 8001)...")
        self._desktop_service = DesktopServiceThread()
        self._desktop_service.started.connect(
            lambda: print("[System] Iron Desktop Service started (localhost:8001)")
        )
        self._desktop_service.error.connect(
            lambda e: print(f"[System] Desktop Service error: {e}")
        )
        self._desktop_service.start()

    def _init_ollama_watcher(self):
        """Start looking for models immediately."""
        print("[DEBUG] Initializing VoxAI Model Watcher...")
        from backend.vox_model_worker import VoxModelWorker
        self.ollama_list_worker = VoxModelWorker("list")
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
        self.chat_worker.chat_finished.connect(self._on_chat_finished)
        self.chat_worker.speed_update.connect(self._on_speed_update)
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

    def _detect_search_intent(self, text):
        """Detect if user wants a web search. Returns search query or None.

        Only triggers for messages that clearly ask for a web search.
        Must have both a search keyword AND a meaningful query (5+ chars).
        """
        lower = text.lower().strip()

        # Too short to be a real search request
        if len(lower) < 10:
            return None

        # --- Patterns: "search for X", "look up X", "google X" ---
        patterns = [
            # "search for best laptop" / "search best laptop under 900"
            _re.compile(r'^search\s+(?:for\s+)?(.+)', _re.IGNORECASE),
            # "look up next public holiday"
            _re.compile(r'^look\s+up\s+(.+)', _re.IGNORECASE),
            # "google best laptop deals"
            _re.compile(r'^google\s+(.+)', _re.IGNORECASE),
            # "find info on / find information about X"
            _re.compile(r'^find\s+(?:info|information)\s+(?:on|about)\s+(.+)', _re.IGNORECASE),
        ]
        for pat in patterns:
            m = pat.match(text)
            if m:
                query = m.group(1).strip().rstrip('?').strip()
                if len(query) >= 5:
                    return query

        # --- "Can you search online for X" / "Could you look up X" ---
        # These need special handling: extract everything AFTER "for" or the search keyword
        conversational = [
            # "can you search online for when the next holiday is" -> captures "when the next holiday is"
            _re.compile(r'(?:can|could|would|please)?\s*(?:you\s+)?search\s+(?:online|the\s+web|the\s+internet)\s+(?:for\s+|and\s+(?:tell|find|show|let)\s+\w+\s+)(.+)', _re.IGNORECASE),
            # "can you search for X"
            _re.compile(r'(?:can|could|would|please)?\s*(?:you\s+)?search\s+for\s+(.+)', _re.IGNORECASE),
            # "can you look up X"
            _re.compile(r'(?:can|could|would|please)?\s*(?:you\s+)?look\s+up\s+(.+)', _re.IGNORECASE),
        ]
        for pat in conversational:
            m = _re.search(pat, text)
            if m:
                query = m.group(1).strip().rstrip('?').strip()
                if len(query) >= 5:
                    return query

        # --- Keyword-triggered: message contains "search online" / "search the web" ---
        # Use the WHOLE message as the search query (stripped of the trigger phrase)
        if 'search online' in lower or 'search the web' in lower or 'search the internet' in lower:
            # Strip the trigger phrase and use the rest as query
            query = _re.sub(r'(?:can|could|would|please)?\s*(?:you\s+)?search\s+(?:online|the\s+web|the\s+internet)\s*(?:for|and)?\s*', '', text, flags=_re.IGNORECASE).strip().rstrip('?').strip()
            if len(query) >= 5:
                return query
            # If stripping leaves nothing useful, use the full text
            return text.rstrip('?').strip()

        return None

    def _ai_doesnt_know(self, response_text):
        """Check if the AI's response indicates it doesn't know the answer
        and would benefit from a web search."""
        lower = response_text.lower()

        # Phrases that indicate the AI can't answer / doesn't have the info
        cant_help = [
            "i don't have access to real-time",
            "i don't have real-time",
            "i can't provide specific information",
            "i can't provide current",
            "i can't provide up-to-date",
            "i don't have current information",
            "i don't have up-to-date",
            "i don't have the latest",
            "i cannot provide real-time",
            "i cannot browse the internet",
            "i can't browse the internet",
            "i can't search the web",
            "i cannot search the web",
            "i don't have access to the internet",
            "i'm unable to provide",
            "my knowledge is limited to",
            "my training data",
            "my knowledge cutoff",
            "my information may be outdated",
            "i'm not able to access",
            "i do not have the ability to",
            "as an ai, i don't have",
            "as an ai, i can't",
            "as an artificial intelligence",
            "i lack the ability to",
            "i cannot access external",
            "i don't have access to external",
            "beyond my knowledge",
            "outside of my training",
            "i recommend checking",
            "i suggest visiting",
            "please check online",
            "you may want to check",
            "you could try searching",
            "for the most accurate information",
            "for the latest information",
            "for up-to-date information",
            "i'd recommend visiting",
        ]

        for phrase in cant_help:
            if phrase in lower:
                return True

        return False

    def handle_send(self):
        user_text = self.input_bar.input.toPlainText().strip()
        if not user_text:
            return

        # Check for /remember command — flag for priority storage
        self._remember_this = False
        if user_text.lower().startswith("/remember "):
            self._remember_this = True
            user_text = user_text[10:].strip()  # Strip the /remember prefix
        elif "remember this:" in user_text.lower():
            self._remember_this = True

        # Add User Message
        self.chat_view.chat_display.add_message(user_text, "user")
        self.input_bar.input.clear()

        # Clear speed stats from previous response
        self.input_bar.clear_speed()

        # Check for explicit web search intent ("search for X", "look up X")
        search_query = self._detect_search_intent(user_text)
        prompt_text = user_text

        if search_query:
            try:
                print(f"[Search] Searching for: {search_query}")
                self.sidebar.update_status("searching")
                
                # Show temporary thinking bubble
                search_bubble = self.chat_view.chat_display.show_thinking()
                
                results = self.search_service.search(search_query, max_results=5)
                
                # Remove bubble
                self.chat_view.chat_display.remove_bubble(search_bubble)
                
                if results:
                    # Inject results into AI prompt SILENTLY (No chat output)
                    import datetime
                    current_date_str = datetime.datetime.now().strftime("%A, %B %d, %Y")
                    
                    search_context = self.search_service.format_for_ai(results, search_query)
                    
                    # Force date context + results
                    prompt_text = (
                        f"[Current Date: {current_date_str}]\n"
                        f"User Question: {user_text}\n\n"
                        f"{search_context}"
                    )
                    print(f"[Search] Injected {len(results)} results into prompt.")
                else:
                    print("[Search] No results found.")
            except Exception as e:
                print(f"[Search] Error: {e}")
                self.sidebar.update_status("error")

        # Store original user text for potential auto-search retry
        self._last_user_text = user_text
        self._search_already_done = search_query is not None

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
            # If VoxAI Local, use the hidden filename (userData)
            if "VoxAI" in mode:
                model = panel.model_combo.currentData() or panel.model_combo.currentText()
            else:
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

        # Reset streaming state
        self._in_thinking = False
        self._thinking_buffer = ""
        self._thinking_section = None
        self._chat_streaming_started = False
        self._refusal_check_buffer = "" # Buffer for catching "I don't know" early
        self._chat_finished_guard = False

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

        # FORCE TOOL USE: Append instruction to the prompt content (User Message)
        # This helps reasoning models that might ignore the system prompt.
        prompt_text += "\n\n(Remember: You have internet access. If you need to search, use [SEARCH: query])"

        # Flag priority ("remember this") if detected
        if hasattr(self, '_remember_this') and self._remember_this:
            self.chat_worker._pending_priority = True
        else:
            self.chat_worker._pending_priority = False

        # Construct System Prompt with Date (More Authoritative)
        import datetime
        current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
        system_instructions = (
            f"SYSTEM: The Current Date is strictly {current_date}.\n"
            "You are VoxAI, a helpful and intelligent assistant.\n"
            "Always use the Current Date as your temporal anchor for 'today', 'yesterday', or 'tomorrow'.\n"
            "Do not hallucinate past dates as current.\n"
            "You have internet access to search for current information when needed."
        )

        # Start Worker
        self.chat_worker.setup(
            provider_type=provider_type,
            model_name=api_model,
            api_key=api_key,
            prompt=prompt_text,
            history=self.chat_view.chat_display.get_history()[:-1],
            system_prompt=system_instructions,
            session_id=self.current_session_id
        )
        self.chat_worker.start()

    def _is_thinking_uncertain(self, text):
        """Check if the thinking block indicates uncertainty or a need to search."""
        triggers = [
            "i don't know",
            "i'm not sure",
            "i am not sure",
            "uncertain",
            "no information",
            "lack of information",
            "search for",
            "check online",
            "google it",
            "verify",
            "speculate",
            "guess",
            "unsure",
            "i need to check",
            "let me check",
            "maybe i can find",
        ]
        lower_text = text.lower()
        for t in triggers:
            if t in lower_text:
                return True
        return False

    def _on_chat_chunk(self, chunk):
        # Remove thinking bubble (dots) on first chunk
        if hasattr(self, 'thinking_bubble') and self.thinking_bubble:
            self.chat_view.chat_display.remove_bubble(self.thinking_bubble)
            self.thinking_bubble = None

        # --- Command Detection ([SEARCH: query]) ---
        # Only trigger on INTENTIONAL search commands, not when the AI is discussing the tool.
        check_text = self._refusal_check_buffer + chunk
        # Keep buffer small but enough for failsafe detection
        if len(check_text) > 200:
             check_text = check_text[-200:]
        self._refusal_check_buffer = check_text

        # NOTE: [SEARCH:] command detection DISABLED
        # The AI kept mentioning [SEARCH:] in its thinking when discussing the tool,
        # which triggered false positives even with negation detection.
        # Now relying ONLY on the failsafe (catches explicit refusals in thinking blocks).

        # --- Thinking model detection (<think>...</think>) ---
        text = self._thinking_buffer + chunk
        self._thinking_buffer = ""

        while text:
            if self._in_thinking:
                # Look for </think> closing tag
                end_idx = text.find("</think>")
                if end_idx != -1:
                    # Send remaining thinking text to section
                    thinking_part = text[:end_idx]
                    if thinking_part and self._thinking_section:
                        self._thinking_section.append_text(thinking_part)

                    # Finalize thinking section
                    self.chat_view.chat_display.end_thinking_section(self._thinking_section)
                    self._in_thinking = False
                    self._thinking_section = None

                    text = text[end_idx + len("</think>"):]
                    continue
                else:
                    # Check for partial tag at end
                    for partial in ["</think", "</thin", "</thi", "</th", "</t", "</", "<"]:
                        if text.endswith(partial):
                            self._thinking_buffer = partial
                            text = text[:-len(partial)]
                            break

                    if text and self._thinking_section:
                        self._thinking_section.append_text(text)
                        
                        # NOTE: [SEARCH:] detection in thinking blocks DISABLED
                        # AI mentions [SEARCH:] when discussing whether to use the tool,
                        # causing false positives. Relying only on failsafe below.

                        # --- THINKING MODEL FAILSAFE ---
                        # If the AI says it CAN'T access real-time data in its thinking,
                        # we intercept and trigger search BEFORE the visible response.
                        # This only affects thinking models (non-thinking models don't use <think>).
                        # IMPORTANT: Only use EXPLICIT refusal phrases, not casual mentions!
                        lower_thinking = text.lower()
                        realtime_failsafe_triggers = [
                            "i don't have real-time access",
                            "i can't access real-time data",
                            "i cannot access real-time data",
                            "i don't have access to real-time",
                            "unable to access current information",
                            "cannot provide real-time information"
                        ]
                        
                        # Only trigger if NOT already searched and we have a user query
                        if not getattr(self, '_search_already_done', False):
                            for failsafe in realtime_failsafe_triggers:
                                if failsafe in lower_thinking:
                                    user_query = getattr(self, '_last_user_text', '')
                                    if user_query:
                                        print(f"[AutoSearch] Failsafe triggered in thoughts: '{failsafe}'")
                                        print(f"[AutoSearch] Searching for user query: '{user_query}'")
                                        
                                        self._chat_finished_guard = True 
                                        if hasattr(self, 'chat_worker') and self.chat_worker.isRunning():
                                            self.chat_worker.terminate()
                                            self.chat_worker.wait()
                                            
                                        self._in_thinking = False
                                        self._thinking_section = None
                                        self._perform_auto_search(user_query)
                                        return

                    return
            else:
                # Look for <think> opening tag
                start_idx = text.find("<think>")
                if start_idx != -1:
                    # Send text before <think> to chat bubble
                    before = text[:start_idx]
                    if before.strip():
                        if not hasattr(self, '_chat_streaming_started') or not self._chat_streaming_started:
                            self.chat_view.chat_display.start_streaming_message()
                            self._chat_streaming_started = True
                        self.chat_view.chat_display.update_streaming_message(before)

                    # Start thinking section
                    self._in_thinking = True
                    self._thinking_section = self.chat_view.chat_display.start_thinking_section()
                    self.sidebar.update_status("reasoning")

                    # NOTE: Seed thought injection removed - it was self-triggering intent detection.
                    # The AI knows about [SEARCH: query] from the system prompt.

                    text = text[start_idx + len("<think>"):]
                    continue
                else:
                    # Check for partial tag at end
                    for partial in ["<think", "<thin", "<thi", "<th", "<t", "<"]:
                        if text.endswith(partial):
                            self._thinking_buffer = partial
                            text = text[:-len(partial)]
                            break

                    # Normal text -> chat bubble
                    if text:
                        if not hasattr(self, '_chat_streaming_started') or not self._chat_streaming_started:
                            self.chat_view.chat_display.start_streaming_message()
                            self._chat_streaming_started = True
                        self.chat_view.chat_display.update_streaming_message(text)
                    return

    def _perform_auto_search(self, user_query):
        """Execute auto-search logic: stop current stream, search, and restart chat."""
        print(f"[AutoSearch] Initiating search for: {user_query}")
        
        # COMPREHENSIVE CLEANUP: Remove ALL leftover UI elements
        # 1. Remove thinking section if exists
        if self._thinking_section:
            try:
                self.chat_view.chat_display.remove_bubble(self._thinking_section)
            except Exception as e:
                print(f"[AutoSearch] Failed to remove thinking section: {e}")
            self._thinking_section = None
            self._in_thinking = False
        
        # 2. Remove partial streaming message if exists
        if getattr(self, '_chat_streaming_started', False):
            try:
                # End streaming gracefully (clears internal state)
                self.chat_view.chat_display.end_streaming_message()
                # Remove the last message (which was partial)
                self.chat_view.chat_display.remove_last_message()
            except Exception as e:
                print(f"[AutoSearch] Failed to remove streaming message: {e}")
            self._chat_streaming_started = False
        
        # 3. Remove thinking bubble (dots) if exists
        if getattr(self, 'thinking_bubble', None):
            try:
                self.chat_view.chat_display.remove_bubble(self.thinking_bubble)
            except Exception as e:
                print(f"[AutoSearch] Failed to remove thinking bubble: {e}")
            self.thinking_bubble = None
        
        # Stop current worker if running
        if hasattr(self, 'chat_worker') and self.chat_worker.isRunning():
            self._chat_finished_guard = True # Prevent normal finish logic
            self.chat_worker.terminate()
            self.chat_worker.wait()
        
        try:
            # Show search status subtly
            self.sidebar.update_status("searching")
            
            # Show temporary thinking bubble for search
            self.thinking_bubble = self.chat_view.chat_display.show_thinking()
            self.chat_view.chat_display.scroll_to_bottom()

            results = self.search_service.search(user_query, max_results=5)
            if results:
                # Remove bubble before restarting chat
                if self.thinking_bubble:
                    self.chat_view.chat_display.remove_bubble(self.thinking_bubble)
                    self.thinking_bubble = None

                # Re-ask with search context
                import datetime
                current_date_str = datetime.datetime.now().strftime("%A, %B %d, %Y")
                
                search_context = self.search_service.format_for_ai(results, user_query)
                
                # Force date context + search results
                prompt_with_search = (
                    f"[Current Date: {current_date_str}]\n"
                    f"User Question: {user_query}\n\n"
                    f"{search_context}"
                )

                # Flag retry so we don't loop
                self._search_retry_active = True
                self._search_already_done = True

                # Reset streaming state
                self._in_thinking = False
                self._thinking_buffer = ""
                self._thinking_section = None
                self._chat_streaming_started = False
                self._chat_finished_guard = False
                self._refusal_check_buffer = ""

                # Show thinking bubble for new response
                self.thinking_bubble = self.chat_view.chat_display.show_thinking()
                self.chat_view.chat_display.scroll_to_bottom()
                self.sidebar.update_status("thinking")

                # Re-construct system prompt
                import datetime
                current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
                system_instructions = (
                    f"SYSTEM: The Current Date is strictly {current_date}.\n"
                    "You are VoxAI, a helpful and intelligent assistant.\n"
                    "Always use the Current Date as your temporal anchor for 'today', 'yesterday', or 'tomorrow'.\n"
                    "Do not hallucinate past dates as current."
                )

                # Re-send to AI with search results
                self.chat_worker.setup(
                    provider_type=self.chat_worker.provider_type,
                    model_name=self.chat_worker.model_name,
                    api_key=self.chat_worker.api_key,
                    prompt=prompt_with_search,
                    history=self.chat_view.chat_display.get_history()[:-1],
                    system_prompt=system_instructions,
                    session_id=self.current_session_id
                )
                self.chat_worker.start()
        except Exception as e:
            print(f"[AutoSearch] Error: {e}")
            self.sidebar.update_status("error")

    def _on_chat_finished(self):
        import re

        # Guard against double execution
        if getattr(self, '_chat_finished_guard', False):
            print("[CodeExtract] Guard: _on_chat_finished already ran, skipping")
            return
        self._chat_finished_guard = True

        # Finalize any remaining thinking section
        if self._in_thinking and self._thinking_section:
            # Flush remaining buffer
            if self._thinking_buffer:
                self._thinking_section.append_text(self._thinking_buffer)
                self._thinking_buffer = ""
            self.chat_view.chat_display.end_thinking_section(self._thinking_section)
            self._in_thinking = False
            self._thinking_section = None

    def _perform_auto_search(self, user_query):
        """Execute auto-search logic: stop current stream, search, and restart chat."""
        
        # Stop current worker if running
        if hasattr(self, 'chat_worker') and self.chat_worker.isRunning():
            self._chat_finished_guard = True # Prevent normal finish logic
            self.chat_worker.terminate()
            self.chat_worker.wait()
        
        try:
            # Show search status subtly
            print(f"[AutoSearch] Searching for: {user_query}")
            self.sidebar.update_status("searching")
            
            # Show temporary thinking bubble if not already there
            if not getattr(self, 'thinking_bubble', None):
               self.thinking_bubble = self.chat_view.chat_display.show_thinking()
               self.chat_view.chat_display.scroll_to_bottom()

            results = self.search_service.search(user_query, max_results=5)
            if results:
                # Remove bubble before restarting chat
                if self.thinking_bubble:
                    self.chat_view.chat_display.remove_bubble(self.thinking_bubble)
                    self.thinking_bubble = None

                # Re-ask with search context
                import datetime
                current_date_str = datetime.datetime.now().strftime("%A, %B %d, %Y")
                
                search_context = self.search_service.format_for_ai(results, user_query)
                
                # Force date context + search results
                prompt_with_search = (
                    f"[Current Date: {current_date_str}]\n"
                    f"User Question: {user_query}\n\n"
                    f"{search_context}"
                )

                # Flag retry so we don't loop
                self._search_retry_active = True
                self._search_already_done = True

                # Reset streaming state
                self._in_thinking = False
                self._thinking_buffer = ""
                self._thinking_section = None
                self._chat_streaming_started = False
                self._chat_finished_guard = False
                self._refusal_check_buffer = ""

                # Show thinking bubble for new response
                self.thinking_bubble = self.chat_view.chat_display.show_thinking()
                self.chat_view.chat_display.scroll_to_bottom()
                self.sidebar.update_status("thinking")

                # SYSTEM PROMPT [Round 3 Style]
                import datetime
                current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
                system_instructions = (
                    f"SYSTEM: The Current Date is strictly {current_date}.\n"
                    "You are VoxAI, a helpful and intelligent assistant.\n"
                    "Always use the Current Date as your temporal anchor for 'today', 'yesterday', or 'tomorrow'."
                )

                # Re-send to AI with search results
                self.chat_worker.setup(
                    provider_type=self.chat_worker.provider_type,
                    model_name=self.chat_worker.model_name,
                    api_key=self.chat_worker.api_key,
                    prompt=prompt_with_search,
                    history=self.chat_view.chat_display.get_history()[:-1],
                    system_prompt=system_instructions,
                    session_id=self.current_session_id
                )
                self.chat_worker.start()
        except Exception as e:
            print(f"[AutoSearch] Error: {e}")
            self.sidebar.update_status("error")

        # Only end streaming if we started a chat bubble
        if getattr(self, '_chat_streaming_started', False):
            self.chat_view.chat_display.end_streaming_message()

        # --- AUTO-SEARCH RETRY: If the AI said "I don't know", search and re-ask ---
        # --- AUTO-SEARCH RETRY: If the AI said "I don't know", search and re-ask ---
        if not getattr(self, '_search_already_done', False) and not getattr(self, '_search_retry_active', False):
            history = self.chat_view.chat_display.get_history()
            if history and history[-1]['role'] == 'assistant':
                ai_response = history[-1]['content']
                user_query = getattr(self, '_last_user_text', '')
                
                if user_query and self._ai_doesnt_know(ai_response):
                    print(f"[AutoSearch] AI doesn't know — searching for: {user_query}")
                    
                    # Remove the "I don't know" response
                    self.chat_view.chat_display.remove_last_message()
                    
                    # Call shared search method
                    self._perform_auto_search(user_query)
                    return # Exit — the retry will call _on_chat_finished again when done

        # Clear retry flag
        self._search_retry_active = False
        self.sidebar.update_status("idle")

        # --- POST-MESSAGE CODE EXTRACTION ---
        history = self.chat_view.chat_display.get_history()
        if not history:
            print("[CodeExtract] No history found")
            return

        last_msg = history[-1]
        if last_msg['role'] != 'assistant':
            print(f"[CodeExtract] Last msg role is '{last_msg['role']}', not assistant")
            return

        content = last_msg['content']
        print(f"[CodeExtract] Content length: {len(content)}, first 200 chars: {repr(content[:200])}")

        # Strip <think>...</think> from content for display purposes
        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        # Find all code blocks
        pattern = r"```\s*(\w+)?[ \t]*\r?\n(.*?)```"
        blocks = list(re.finditer(pattern, clean_content, flags=re.DOTALL))
        print(f"[CodeExtract] Found {len(blocks)} code fence matches in content")

        # --- FILENAME-DRIVEN EXTRACTION ---
        FILE_EXTENSIONS = r'py|js|ts|html|css|cpp|h|json|txt|md|bat|sh|sql|go|rs|java|rb|php'
        FILENAME_PATTERN = re.compile(
            r'(?:\*{2}|#{1,3}\s+|[Ff]ile:\s*)?'
            r'([a-zA-Z0-9_\-]+\.(?:' + FILE_EXTENSIONS + r'))'
            r'\*{0,2}\s*:?\s*$'
        )

        # Pass 1: Scan blocks, detect filenames, group by filename
        named_files = {}
        named_match_indices = set()

        for idx, match in enumerate(blocks):
            lang = (match.group(1) or "text").strip().lower()
            code = match.group(2).strip()
            code_lines = [l for l in code.split('\n') if l.strip()]

            if len(code_lines) < 2:
                continue

            preceding = clean_content[max(0, match.start() - 300):match.start()]
            name_match = FILENAME_PATTERN.search(preceding)

            if name_match:
                filename = name_match.group(1)
                print(f"[CodeExtract]   Block {idx}: detected filename '{filename}' ({len(code_lines)} lines)")

                if filename not in named_files:
                    named_files[filename] = {'lang': lang, 'codes': [], 'match_indices': []}

                named_files[filename]['codes'].append(code)
                named_files[filename]['match_indices'].append(idx)
                named_match_indices.add(idx)
            else:
                print(f"[CodeExtract]   Block {idx}: no filename (explanation snippet, {len(code_lines)} lines)")

        print(f"[CodeExtract] {len(named_files)} named files found: {list(named_files.keys())}")

        if not named_files:
            return

        # Pass 2: Merge code blocks per filename
        real_blocks = []
        for filename, info in named_files.items():
            merged_code = '\n\n'.join(info['codes'])
            real_blocks.append((info['lang'], merged_code, filename))
            print(f"[CodeExtract]   Merged '{filename}': {len(info['codes'])} blocks -> {len(merged_code)} chars")

        # --- Strip named code blocks from the chat bubble display ---
        display_text = clean_content

        for idx in sorted(named_match_indices, reverse=True):
            match = blocks[idx]
            display_text = display_text[:match.start()] + display_text[match.end():]

        for filename in named_files.keys():
            fn_escaped = re.escape(filename)
            display_text = re.sub(
                r'(?:^|\n)\s*(?:#{1,3}\s+|\*{2}|[Ff]ile:\s*)?' + fn_escaped + r'\*{0,2}:?\s*(?:\n|$)',
                '\n', display_text
            )

        display_text = re.sub(r'\n{3,}', '\n\n', display_text).strip()

        if display_text:
            self.chat_view.chat_display.rewrite_last_message(display_text, history_text=content)
        else:
            self.chat_view.chat_display.rewrite_last_message("Here's the code:", history_text=content)

        # --- Create file cards ---
        from widgets.file_card import FileCardRow, LANG_EXTENSIONS
        self.code_panel.clear_files()

        card_row = FileCardRow()
        for lang, code, filename in real_blocks:
            self.code_panel.add_file(filename, code, lang, auto_open=False)
            card = card_row.add_card(filename, code, lang)
            card.clicked.connect(self._on_file_card_clicked)

        self.chat_view.chat_display.insert_widget_after_last_message(card_row)

    def _on_file_card_clicked(self, filename, content, language):
        """Open the CodePanel with the clicked file selected."""
        self.code_panel.file_selector.setCurrentText(filename)
        if self.code_panel.width() == 0:
            self.code_panel.slide_in()

    def _on_speed_update(self, speed, tokens, duration):
        """Display tokens/sec stats below the input bar."""
        self.input_bar.show_speed(speed, tokens, duration)

    def _on_chat_error(self, err):
        self.chat_view.chat_display.update_streaming_message(f"\n[Error: {err}]")
        self.chat_view.chat_display.end_streaming_message()
        self.sidebar.update_status("error")

    def handle_clear_chat(self):
        self.chat_view.chat_display.clear_chat()
        self.sidebar.update_status("idle")
        self.code_panel.slide_out()
        self.input_bar.clear_speed()

    def handle_upload(self):
        file_path = self.file_handler.open_file_dialog()
        if file_path:
            file_data = self.file_handler.process_file(file_path)

            content = ""
            TEXT_EXTS = ['.py', '.txt', '.json', '.js', '.md', '.csv', '.html', '.css',
                         '.xml', '.yaml', '.yml', '.log', '.ts', '.jsx', '.tsx', '.sh',
                         '.bat', '.cfg', '.ini', '.toml', '.sql', '.java', '.cpp', '.c',
                         '.h', '.rs', '.go', '.rb', '.php']

            if file_data['extension'] in TEXT_EXTS:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    # Limit very large files
                    if len(content) > 6000:
                        content = content[:6000] + '\n... [truncated]'
                    self.code_panel.add_file(file_data['name'], content)
                except Exception as e:
                    print(f"Error reading file: {e}")

            # Build context message for AI and display message for user
            if content:
                context_msg = (f"I am uploading a file named '{file_data['name']}' "
                               f"({file_data['size']}).\n\nFILE CONTENT:\n```\n{content}\n```\n\n"
                               f"Please analyze this file and tell me what it does.")
                display_msg = f"📂 Uploaded: {file_data['name']} ({file_data['size']})"
                self.chat_view.chat_display.add_message(context_msg, "user", display_text=display_msg)
            else:
                context_msg = (f"I am uploading a file named '{file_data['name']}' "
                               f"({file_data['size']}, {file_data['extension']} file). "
                               f"This is a binary file so I cannot show the content directly. "
                               f"Please acknowledge you understand a file was uploaded.")
                display_msg = f"📂 Uploaded: {file_data['name']} ({file_data['size']})"
                self.chat_view.chat_display.add_message(context_msg, "user", display_text=display_msg)

            # Actually send to AI instead of fake response
            self._send_after_upload()

    def _send_after_upload(self):
        """Trigger AI response after a file upload, using the file context in history."""
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
            if "VoxAI" in mode:
                model = panel.model_combo.currentData() or panel.model_combo.currentText()
            else:
                model = panel.model_combo.currentText()

        cfg = ConfigManager.load_config()
        llm_cfg = cfg.get("llm", {})

        provider_type = "Gemini" if "Gemini" in model or "Provider" in mode else "Ollama"
        if "Llama" in model or "Mistral" in model or "Gemma" in model or "qwen" in model.lower():
            provider_type = "Ollama"


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

        # The last message in history IS the file content (user message)
        # Send it as the prompt, with history being everything before it
        history = self.chat_view.chat_display.get_history()
        if history:
            prompt = history[-1]['content']
            prev_history = history[:-1]
        else:
            return

        api_key = llm_cfg.get("api_key", "")
        
        # System Prompt for file context
        import datetime
        current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
        system_instructions = (
            f"SYSTEM: The Current Date is strictly {current_date}.\n"
            "You are VoxAI, a helpful and intelligent assistant.\n"
            "Always use the Current Date as your temporal anchor.\n"
            "TOOLS AVAILABLE:\n"
            "- Internet Search: If you need real-time information, output `[SEARCH: your query]` on a new line."
        )

        self.chat_worker.setup(
            provider_type=provider_type,
            model_name=api_model,
            api_key=api_key,
            prompt=prompt,
            history=prev_history,
            system_prompt=system_instructions
        )
        self.chat_worker.start()

    def finalize_ai_response(self, text):
        """Adds a simple AI message to the chat (used for system events)."""
        self.chat_view.chat_display.add_message(text, "ai")

    def closeEvent(self, event):
        """Cleanup workers on exit to prevent crash."""
        print("[System] Shutting down...")

        # Stop Desktop Service (thread + force-kill the port)
        if hasattr(self, '_desktop_service') and self._desktop_service.isRunning():
            print("Stopping Desktop Service...")
            self._desktop_service.stop()
            self._desktop_service.wait(3000)
        # Force-kill port 8001 in case the process outlives the thread
        self._kill_port(8001)

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

    def _perform_auto_search(self, user_query):
        """Execute auto-search logic: stop current stream, search, and restart chat."""
        
        # Stop current worker if running
        if hasattr(self, 'chat_worker') and self.chat_worker.isRunning():
            self._chat_finished_guard = True # Prevent normal finish logic
            self.chat_worker.terminate()
            self.chat_worker.wait()
        
        try:
            # Show search status subtly
            print(f"[AutoSearch] Searching for: {user_query}")
            self.sidebar.update_status("searching")
            
            # Show temporary thinking bubble if not already there
            if not getattr(self, 'thinking_bubble', None):
               self.thinking_bubble = self.chat_view.chat_display.show_thinking()
               self.chat_view.chat_display.scroll_to_bottom()

            results = self.search_service.search(user_query, max_results=5)
            if results:
                # Remove bubble before restarting chat
                if self.thinking_bubble:
                    self.chat_view.chat_display.remove_bubble(self.thinking_bubble)
                    self.thinking_bubble = None

                # Re-ask with search context
                import datetime
                current_date_str = datetime.datetime.now().strftime("%A, %B %d, %Y")
                
                search_context = self.search_service.format_for_ai(results, user_query)
                
                # Force date context + search results
                prompt_with_search = (
                    f"User Question: {user_query}\n\n"
                    f"{search_context}"
                )

                # Flag retry so we don't loop
                self._search_retry_active = True
                self._search_already_done = True

                # Reset streaming state
                self._in_thinking = False
                self._thinking_buffer = ""
                self._thinking_section = None
                self._chat_streaming_started = False
                self._chat_finished_guard = False
                self._refusal_check_buffer = ""

                # Show thinking bubble for new response
                self.thinking_bubble = self.chat_view.chat_display.show_thinking()
                self.chat_view.chat_display.scroll_to_bottom()
                self.sidebar.update_status("thinking")

                # SYSTEM PROMPT [Round 3 Style]
                import datetime
                current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
                
                # Manual System Prompt construction since we are bypassing main flow
                # To be safe, we just pass the critical date info and identity
                system_instructions = (
                    f"SYSTEM: The Current Date is strictly {current_date}.\n"
                    "You are VoxAI, a helpful and intelligent assistant.\n"
                    "Always use the Current Date as your temporal anchor for 'today', 'yesterday', or 'tomorrow'."
                )

                # Re-send to AI with search results
                self.chat_worker.setup(
                    provider_type=self.chat_worker.provider_type,
                    model_name=self.chat_worker.model_name,
                    api_key=self.chat_worker.api_key,
                    prompt=prompt_with_search,
                    history=self.chat_view.chat_display.get_history()[:-1],
                    system_prompt=system_instructions,
                    session_id=self.current_session_id
                )
                self.chat_worker.start()
        except Exception as e:
            print(f"[AutoSearch] Error: {e}")
            self.sidebar.update_status("error")
