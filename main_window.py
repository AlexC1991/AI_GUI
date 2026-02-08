import os
import re
import time
import datetime
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget
)
from PySide6.QtCore import QTimer, QRect

from widgets.sidebar import Sidebar
from widgets.chat_view import ChatView
from widgets.input_bar import InputBar
from widgets.code_panel import CodePanel
from widgets.settings_view import SettingsView, DesktopServiceThread
from widgets.image_gen_view import ImageGenView
from widgets.model_selector_panel import ModelSelectorPanel
from widgets.file_card import FileCardRow
from utils.config_manager import ConfigManager
from utils.file_handler import FileHandler
from backend.chat_worker import ChatWorker
from backend.search_service import SearchService
from backend.chat_agent import ChatAgent
from backend.image_worker import ImageWorker
from backend.vox_model_worker import VoxModelWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("VoxAI Orchestrator")
        self.resize(1400, 800)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sidebar = Sidebar()
        main_layout.addWidget(self.sidebar)

        self.stack = QStackedWidget()

        # Chat
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
        self.stack.addWidget(self.chat_container_widget)

        # Settings
        self.settings_view = SettingsView()
        self.stack.addWidget(self.settings_view)

        # Image Gen
        self.image_gen_view = ImageGenView()
        self.stack.addWidget(self.image_gen_view)

        main_layout.addWidget(self.stack, 1)

        self.code_panel = CodePanel()
        main_layout.addWidget(self.code_panel)

        self.file_handler = FileHandler(self)
        self.search_service = SearchService()
        self._chat_finished_guard = False
        self._setup_backend()
        
        self.input_bar.send_button.clicked.connect(self.handle_send)
        if hasattr(self.input_bar.input, 'return_pressed'):
            self.input_bar.input.return_pressed.connect(self.handle_send)
        self.input_bar.attach_btn.clicked.connect(self.handle_upload)
        self.input_bar.clear_requested.connect(self.handle_clear_chat)
        self.sidebar.mode_changed.connect(self.handle_mode_switch)

        if self.sidebar.change_model_btn:
            self.sidebar.change_model_btn.clicked.connect(self._show_model_panel)

        self.model_panel = ModelSelectorPanel(self)
        self.model_panel.model_selected.connect(self._on_model_selected)
        self.model_panel.hide()

        self.current_session_id = f"session_{int(time.time())}"
        print(f"[MainWindow] Persistent Session ID: {self.current_session_id}")
        self._init_ollama_watcher()
        self._start_desktop_service()
        # Verify network status after service has time to start
        QTimer.singleShot(2000, self._verify_desktop_service)

    def _setup_backend(self):
        print("[DEBUG] Setting up backend workers...")
        from backend.image_worker import ImageWorker
        self.image_worker = ImageWorker()
        self.image_worker.progress.connect(self._on_gen_progress_msg)
        self.image_worker.step_progress.connect(self._on_gen_step)
        self.image_worker.finished.connect(self._on_gen_finished)
        self.image_worker.error.connect(self._on_gen_error)
        self.image_worker.auth_required.connect(self._on_auth_required)

        self.chat_worker = ChatWorker()
        self.chat_worker.speed_update.connect(self._on_speed_update)
        
        self.agent = ChatAgent(self)
        self.agent.status_changed.connect(self.sidebar.update_status)

    def handle_send(self):
        user_text = self.input_bar.input.toPlainText().strip()
        if not user_text: return

        self._remember_this = False
        if user_text.lower().startswith("/remember "):
            self._remember_this = True
            user_text = user_text[10:].strip()
        elif "remember this:" in user_text.lower():
            self._remember_this = True
        
        if hasattr(self, 'chat_worker'):
            self.chat_worker._pending_priority = self._remember_this

        self.chat_view.chat_display.add_message(user_text, "user")
        self.input_bar.input.clear()
        self.input_bar.clear_speed()

        force_reasoning = False
        if self.sidebar.current_panel and hasattr(self.sidebar.current_panel, 'reasoning_mode_chk'):
            force_reasoning = self.sidebar.current_panel.reasoning_mode_chk.isChecked()

        self.agent.start_new_turn(user_text, force_reasoning)

    def _on_chat_finished(self):
        """Called by Agent. Fixes File Card extraction."""
        import re
        if getattr(self, '_chat_finished_guard', False): return
        self._chat_finished_guard = True

        self.input_bar.set_generating(False)
        self.sidebar.update_status("idle")

        # FIX: Ensure streaming buffer is committed
        if hasattr(self, '_chat_streaming_started') and self._chat_streaming_started:
            self.chat_view.chat_display.end_streaming_message()
            self._chat_streaming_started = False

        if self.agent.current_thinking_widget:
             self.chat_view.chat_display.end_thinking_section(self.agent.current_thinking_widget)
             self.agent.current_thinking_widget = None

        # Parse History
        history = self.chat_view.chat_display.get_history()
        if not history: return
        last_msg = history[-1]
        if last_msg['role'] != 'assistant': return

        content = last_msg['content']
        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        pattern = r"```\s*(\w+)?[ \t]*\r?\n(.*?)```"
        blocks = list(re.finditer(pattern, clean_content, flags=re.DOTALL))

        # Helper Regex
        FILE_EXTENSIONS = r'py|js|ts|html|css|cpp|h|json|txt|md|bat|sh|sql|go|rs|java|rb|php'
        FILENAME_PATTERN = re.compile(
            r'(?:\*{2}|#{1,3}\s+|[Ff]ile:\s*)?'
            r'([a-zA-Z0-9_\-]+\.(?:' + FILE_EXTENSIONS + r'))'
            r'\*{0,2}\s*:?\s*$'
        )

        named_files = {}
        named_match_indices = set()

        for idx, match in enumerate(blocks):
            lang = (match.group(1) or "text").strip().lower()
            code = match.group(2).strip()
            if len(code.split('\n')) < 2: continue

            preceding = clean_content[max(0, match.start() - 300):match.start()]
            name_match = FILENAME_PATTERN.search(preceding)

            # --- FIX: Support Unnamed Snippets ---
            if name_match:
                filename = name_match.group(1)
            else:
                ext = self.code_panel._get_extension(lang) if hasattr(self.code_panel, '_get_extension') else '.txt'
                filename = f"snippet_{idx + 1}{ext}"

            if filename not in named_files:
                named_files[filename] = {'lang': lang, 'codes': [], 'match_indices': []}
            named_files[filename]['codes'].append(code)
            named_files[filename]['match_indices'].append(idx)
            named_match_indices.add(idx)

        if not named_files: return

        real_blocks = []
        for filename, info in named_files.items():
            merged_code = '\n\n'.join(info['codes'])
            real_blocks.append((info['lang'], merged_code, filename))

        # Rewrite chat bubble (Strip code out)
        display_text = clean_content
        for idx in sorted(named_match_indices, reverse=True):
            match = blocks[idx]
            display_text = display_text[:match.start()] + display_text[match.end():]
        
        # Strip filenames IF they were explicit
        for filename in named_files.keys():
            if "snippet_" not in filename:
                fn_escaped = re.escape(filename)
                display_text = re.sub(
                    r'(?:^|\n)\s*(?:#{1,3}\s+|\*{2}|[Ff]ile:\s*)?' + fn_escaped + r'\*{0,2}:?\s*(?:\n|$)',
                    '\n', display_text
                )
        display_text = re.sub(r'\n{3,}', '\n\n', display_text).strip()

        if display_text:
            self.chat_view.chat_display.rewrite_last_message(display_text, history_text=content)
        else:
            self.chat_view.chat_display.rewrite_last_message("Here are the generated files:", history_text=content)

        # Generate Cards
        from widgets.file_card import FileCardRow
        self.code_panel.clear_files()
        card_row = FileCardRow()
        for lang, code, filename in real_blocks:
            self.code_panel.add_file(filename, code, lang, auto_open=False)
            card = card_row.add_card(filename, code, lang)
            card.clicked.connect(self._on_file_card_clicked)
        self.chat_view.chat_display.insert_widget_after_last_message(card_row)

    def _on_file_card_clicked(self, filename, content, language):
        self.code_panel.file_selector.setCurrentText(filename)
        if self.code_panel.width() == 0: self.code_panel.slide_in()

    def handle_mode_switch(self, mode):
        idx = 0
        if mode == "settings": idx = 1
        elif mode == "image": idx = 2
        self.stack.setCurrentIndex(idx)
        if mode != "chat": self.code_panel.slide_out()
        
        try: self.sidebar.upload_btn.clicked.disconnect()
        except: pass
        try: self.sidebar.clear_btn.clicked.disconnect()
        except: pass
        
        if mode == "chat":
            if self.sidebar.upload_btn: self.sidebar.upload_btn.clicked.connect(self.handle_upload)
            if self.sidebar.clear_btn: self.sidebar.clear_btn.clicked.connect(self.handle_clear_chat)
            QTimer.singleShot(500, self._verify_desktop_service)

        if mode == "image":
            panel = self.sidebar.current_panel
            if hasattr(panel, 'gen_btn'):
                try: panel.gen_btn.clicked.disconnect()
                except: pass
                panel.gen_btn.clicked.connect(self.start_image_generation)
            if hasattr(panel, 'abort_btn'):
                try: panel.abort_btn.clicked.disconnect()
                except: pass
                panel.abort_btn.clicked.connect(self.abort_generation)
            if hasattr(panel, 'refresh_btn'):
                try: panel.refresh_btn.clicked.disconnect()
                except: pass
                panel.refresh_btn.clicked.connect(self.image_gen_view.refresh_assets)

    @staticmethod
    def _kill_port(port):
        import subprocess, sys as _sys
        if _sys.platform != "win32": return
        try:
            r = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
                 f'(Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue).OwningProcess'],
                capture_output=True, text=True, timeout=5)
            pids = [p.strip() for p in r.stdout.strip().split('\n') if p.strip().isdigit()]
            if not pids: return
            for pid in pids:
                if pid == str(os.getpid()): continue
                subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True, text=True, timeout=5)
        except: pass

    def _start_desktop_service(self):
        self._kill_port(8001)
        print("[System] Starting Iron Desktop Service (port 8001)...")
        self._desktop_service = DesktopServiceThread()
        self._desktop_service.started.connect(lambda: print("[System] Desktop Service ready"))
        self._desktop_service.start()
        
    def _verify_desktop_service(self):
        try:
            if self.search_service.is_available():
                if hasattr(self.sidebar, 'current_panel') and hasattr(self.sidebar.current_panel, 'set_network_status'):
                    self.sidebar.current_panel.set_network_status("online")
            else:
                if hasattr(self.sidebar, 'current_panel') and hasattr(self.sidebar.current_panel, 'set_network_status'):
                    self.sidebar.current_panel.set_network_status("offline")
        except: pass

    def _init_ollama_watcher(self):
        from backend.vox_model_worker import VoxModelWorker
        self.ollama_list_worker = VoxModelWorker("list")
        self.ollama_list_worker.models_ready.connect(self.handle_ollama_list)
        self.ollama_list_worker.error.connect(self.handle_ollama_error)
        self.ollama_list_worker.start()

    def handle_ollama_list(self, models):
        self._available_models = models if models else []
        panel = self.sidebar.current_panel
        if panel and hasattr(panel, 'set_local_models'): panel.set_local_models(models)
        if hasattr(self, 'model_panel'): self.model_panel.set_models(models)

    def handle_ollama_error(self, err):
        print(f"[DEBUG] Ollama Error: {err}")
        panel = self.sidebar.current_panel
        if panel and hasattr(panel, 'set_local_models'): panel.set_local_models(["(Connection Error)"])

    def _show_model_panel(self):
        panel_rect = QRect(self.sidebar.width(), 0, self.model_panel.panel_width, self.height())
        if hasattr(self, '_available_models'): self.model_panel.set_models(self._available_models)
        self.model_panel.toggle_panel(panel_rect)

    def _on_model_selected(self, model_data):
        print(f"[DEBUG] Model Selected: {model_data}")
        panel = self.sidebar.current_panel
        if panel and hasattr(panel, 'set_selected_model'):
            panel.set_selected_model(model_data)

    def handle_clear_chat(self):
        self.chat_view.chat_display.clear_chat()
        self.sidebar.update_status("idle")
        self.code_panel.slide_out()
        self.input_bar.clear_speed()
        
    def handle_upload(self):
        file_path = self.file_handler.open_file_dialog()
        if not file_path: return
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
                if len(content) > 8000: content = content[:8000] + '\n... [truncated]'
                self.code_panel.add_file(file_data['name'], content)
            except: pass
        display_msg = f"📂 Uploaded: {file_data['name']} ({file_data['size']})"
        if content:
            context_msg = (f"I am uploading a file named '{file_data['name']}' "
                           f"({file_data['size']}).\n\nFILE CONTENT:\n```\n{content}\n```\n\n"
                           f"Please analyze this file.")
            self.chat_view.chat_display.add_message(context_msg, "user", display_text=display_msg)
            self.agent.start_new_turn(context_msg, force_reasoning=False)
        else:
            context_msg = (f"I am uploading a binary file named '{file_data['name']}' "
                           f"({file_data['size']}).")
            self.chat_view.chat_display.add_message(context_msg, "user", display_text=display_msg)
            self.agent.start_new_turn(context_msg, force_reasoning=False)

    def _on_speed_update(self, speed, tokens, duration):
        self.input_bar.show_speed(speed, tokens, duration)
        
    # Image Gen methods (identical logic)
    def start_image_generation(self):
        if not hasattr(self, 'image_worker'): self._setup_backend()
        view = self.image_gen_view
        panel = self.sidebar.current_panel
        if panel and hasattr(panel, 'gen_btn'): panel.gen_btn.setEnabled(False)
        if panel and hasattr(panel, 'abort_btn'): panel.abort_btn.setEnabled(True)
        view.progress_bar.setValue(0)
        view.set_status("Starting generation...", "#4ade80")
        self.sidebar.update_status("processing")
        config = ConfigManager.load_config()
        img_cfg = config.get("image", {})
        width, height = view.get_resolution()
        loras = view.get_active_loras()
        self.image_worker.setup(
            prompt=view.get_positive_prompt(),
            negative_prompt=view.get_negative_prompt(),
            model=view.get_selected_checkpoint(),
            width=width, height=height, steps=view.get_steps(),
            cfg_scale=view.get_cfg_scale(), seed=view.get_seed(),
            sampler=view.get_sampler(),
            lora=loras[0][0] if loras else None,
            lora_weight=loras[0][1] if loras else 1.0,
            vae=view.get_active_vae(),
            text_encoders=view.get_active_text_encoders(),
            checkpoint_dir=img_cfg.get("checkpoint_dir", "models/checkpoints"),
            lora_dir=img_cfg.get("lora_dir", "models/loras"),
            vae_dir=img_cfg.get("vae_dir", "models/vae"),
            text_encoder_dir=img_cfg.get("text_encoder_dir", "models/text_encoders"),
            output_dir=img_cfg.get("output_dir", "outputs/images")
        )
        self.image_worker.start()

    def abort_generation(self):
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
        self.image_gen_view.show_image(path, 0)

    def _on_gen_error(self, err):
        self._reset_gen_ui()
        self.sidebar.update_status("error")
        self.image_gen_view.set_status(f"❌ {err}", "#ef4444")

    def _on_auth_required(self, model_name):
        self._reset_gen_ui()
        self.sidebar.update_status("auth required")
        self.image_gen_view.set_status("🔐 Authentication required", "#f59e0b")
        from widgets.hf_auth_dialog import request_hf_auth
        if request_hf_auth(self, model_name):
            self.image_gen_view.set_status("Token saved - retrying...", "#4ade80")
            QTimer.singleShot(500, self.start_image_generation)
        else:
            self.image_gen_view.set_status("Authentication cancelled", "#888")

    def _reset_gen_ui(self):
        if self.sidebar.current_panel:
            if hasattr(self.sidebar.current_panel, 'gen_btn'):
                self.sidebar.current_panel.gen_btn.setEnabled(True)
            if hasattr(self.sidebar.current_panel, 'abort_btn'):
                self.sidebar.current_panel.abort_btn.setEnabled(False)

    def closeEvent(self, event):
        print("[System] Shutting down...")
        if hasattr(self, '_desktop_service') and self._desktop_service.isRunning():
            self._desktop_service.stop()
            self._desktop_service.wait(3000)
        self._kill_port(8001)
        if hasattr(self, 'chat_worker') and self.chat_worker.isRunning():
            self.chat_worker.quit()
            self.chat_worker.wait(1000)
        if hasattr(self, 'image_worker') and self.image_worker.isRunning():
            self.image_worker.quit()
            self.image_worker.wait(1000)
        if hasattr(self, 'ollama_list_worker') and self.ollama_list_worker.isRunning():
            self.ollama_list_worker.quit()
            self.ollama_list_worker.wait(1000)
        super().closeEvent(event)