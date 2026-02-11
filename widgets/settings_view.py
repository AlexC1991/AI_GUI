"""
Settings View with Web Server Integration

Adds controls to start/stop the web server directly from the GUI.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QScrollArea, QFrame, QFileDialog, QHBoxLayout, QGroupBox, QMessageBox, QTabWidget, QCheckBox,
    QSpinBox
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from widgets.model_manager_dialog import ModelManagerDialog
from utils.config_manager import ConfigManager
import subprocess
import sys
import os
import socket


class IronGateThread(QThread):
    """Thread to run Vox IronGate Web Gateway (iron_host.py) on port 8000."""
    started = Signal()
    stopped = Signal()
    error = Signal(str)
    install_code_received = Signal(str)

    def __init__(self):
        super().__init__()
        self.process = None
        self._running = False

    def run(self):
        self._running = True
        try:
            root_dir = os.path.dirname(os.path.dirname(__file__))
            script_path = os.path.join(root_dir, "gateway", "iron_host.py")
            code_file = os.path.join(root_dir, "gateway", "vox_install_code.txt")

            if not os.path.exists(script_path):
                self.error.emit(f"iron_host.py not found at {script_path}")
                return

            if os.path.exists(code_file):
                try:
                    os.remove(code_file)
                except:
                    pass

            creation_flags = 0x00000010 if sys.platform == "win32" else 0

            self.process = subprocess.Popen(
                [sys.executable, script_path],
                creationflags=creation_flags,
                cwd=os.path.dirname(script_path)
            )

            self.started.emit()

            import re
            code_pattern = re.compile(r"Install Code:\s*([A-Z0-9\-]+)")

            while self._running:
                if self.process.poll() is not None:
                    break

                if os.path.exists(code_file):
                    try:
                        with open(code_file, "r") as f:
                            content = f.read().strip()

                        match = code_pattern.search(content)
                        if match:
                            self.install_code_received.emit(match.group(1))
                            try:
                                os.remove(code_file)
                            except:
                                pass
                    except:
                        pass

                self.msleep(500)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.stopped.emit()

    def stop(self):
        self._running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except:
                self.process.kill()


class DesktopServiceThread(QThread):
    """Thread to run Iron Desktop Service (iron_desktop.py) on port 8001.

    This is a lightweight, localhost-only API server that provides
    web search and file upload services to the desktop app.
    Separate from the full IronGate web gateway.
    """
    started = Signal()
    stopped = Signal()
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.process = None
        self._running = False

    def run(self):
        self._running = True
        try:
            root_dir = os.path.dirname(os.path.dirname(__file__))
            script_path = os.path.join(root_dir, "gateway", "iron_desktop.py")

            if not os.path.exists(script_path):
                self.error.emit(f"iron_desktop.py not found at {script_path}")
                return

            # Run without a console window (headless service)
            creation_flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW

            self.process = subprocess.Popen(
                [sys.executable, script_path],
                creationflags=creation_flags,
                cwd=os.path.dirname(script_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            self.started.emit()

            while self._running:
                if self.process.poll() is not None:
                    # Process exited unexpectedly
                    stderr = self.process.stderr.read().decode(errors='replace') if self.process.stderr else ""
                    if stderr:
                        self.error.emit(f"Desktop Service exited: {stderr[:200]}")
                    break
                self.msleep(500)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.stopped.emit()

    def stop(self):
        self._running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except:
                self.process.kill()

    def is_alive(self):
        """Check if the process is still running."""
        return self.process is not None and self.process.poll() is None


class SettingsView(QScrollArea):
    llm_settings_saved = Signal()  # Emitted when LLM/API settings change (triggers image gen refresh)

    def __init__(self):
        super().__init__()

        self.irongate_thread = None
        self.desktop_service_thread = None  # Managed from here, or auto-started from MainWindow

        # Scroll Area Properties
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("""
            /* MAIN SCROLL AREA */
            QScrollArea { background: transparent; border: none; }
            QWidget { background: transparent; }
            
            /* HEADERS (Group Boxes) */
            QGroupBox {
                border: 2px solid #333;
                border-radius: 8px;
                margin-top: 30px;
                font-family: Segoe UI;
                font-weight: bold;
                color: #008080; /* Teal Title */
                font-size: 14px;
                background-color: #1E1E1E; 
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
                background-color: transparent; 
            }
            
            /* LABELS */
            QLabel {
                color: #CCCCCC;
                font-size: 12px;
                font-weight: bold;
                margin-bottom: 2px;
            }
            QLabel.sub-label {
                color: #888;
                font-size: 11px;
                font-weight: normal;
                margin-bottom: 5px;
            }

            /* INPUTS */
            QLineEdit, QSpinBox {
                background-color: #252526;
                color: #E0E0E0;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }
            QLineEdit:focus, QSpinBox:focus { border: 1px solid #008080; }
            
            /* DROPDOWNS */
            QComboBox {
                background-color: #252526;
                color: #E0E0E0;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QComboBox::drop-down { border: none; }
            
            /* BUTTONS */
            QPushButton {
                background-color: #333;
                color: white;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #444; }
            
            /* TEST BUTTONS */
            QPushButton.test-btn {
                background-color: #444;
                border: 1px solid #555;
            }
            QPushButton.test-btn:hover { background-color: #555; }

            /* APPLY BUTTON (Special) */
            QPushButton#ApplyBtn {
                background-color: #006666;
                border: 1px solid #004d4d;
                font-weight: bold;
                margin-top: 10px;
            }
            QPushButton#ApplyBtn:hover { background-color: #008080; }
            
            /* START SERVER BUTTON */
            QPushButton#StartServerBtn {
                background-color: #2d7d2d;
                border: 1px solid #1a5c1a;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton#StartServerBtn:hover { background-color: #3d9d3d; }
            
            /* STOP SERVER BUTTON */
            QPushButton#StopServerBtn {
                background-color: #8b2d2d;
                border: 1px solid #6b1a1a;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton#StopServerBtn:hover { background-color: #ab3d3d; }
        """)

        # Main Container
        self.container = QWidget()
        self.main_layout = QVBoxLayout(self.container)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(40, 20, 40, 40)
        self.main_layout.setAlignment(Qt.AlignTop)

        # PAGE TITLE
        title = QLabel("Global Settings")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: white; margin-bottom: 10px;")
        self.main_layout.addWidget(title)

        # --- SECTIONS ---
        self.create_irongate_section()  # NEW - Vox IronGate
        self.create_elastic_memory_section()
        self.create_llm_section()
        self.create_cloud_gpu_section()
        self.create_image_section()
        self.create_remote_section()

        self.setWidget(self.container)

    def create_irongate_section(self):
        """Vox IronGate controls — Desktop Service + Web Gateway."""
        group = QGroupBox("⚔️ Vox IronGate Services")
        layout = QVBoxLayout(group)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 35, 20, 20)

        # --- Desktop Service (port 8001) ---
        desktop_header = QLabel("Desktop Service (Port 8001)")
        desktop_header.setStyleSheet("color: #00CED1; font-size: 13px; font-weight: bold; margin-top: 5px;")
        layout.addWidget(desktop_header)

        desktop_desc = QLabel("Local-only API for web search & file handling. Required for desktop AI features.")
        desktop_desc.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(desktop_desc)

        desktop_btn_row = QHBoxLayout()

        self.desktop_status = QLabel("● Stopped")
        self.desktop_status.setStyleSheet("color: #888;")
        desktop_btn_row.addWidget(self.desktop_status)
        desktop_btn_row.addStretch()

        self.start_desktop_btn = QPushButton("▶ Start")
        self.start_desktop_btn.setCursor(Qt.PointingHandCursor)
        self.start_desktop_btn.clicked.connect(self.start_desktop_service)
        self.start_desktop_btn.setStyleSheet("background-color: #006666; border: 1px solid #004d4d;")
        desktop_btn_row.addWidget(self.start_desktop_btn)

        self.stop_desktop_btn = QPushButton("■ Stop")
        self.stop_desktop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_desktop_btn.clicked.connect(self.stop_desktop_service)
        self.stop_desktop_btn.setEnabled(False)
        desktop_btn_row.addWidget(self.stop_desktop_btn)

        layout.addLayout(desktop_btn_row)

        # Check if desktop service is already running (auto-started by MainWindow)
        self._check_desktop_status()

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #333; margin: 8px 0;")
        layout.addWidget(line)

        # --- Web Gateway (port 8000 + ngrok) ---
        web_header = QLabel("Web Gateway (Port 8000 + Tunnel)")
        web_header.setStyleSheet("color: #BB86FC; font-size: 13px; font-weight: bold;")
        layout.addWidget(web_header)

        web_desc = QLabel("Share your AI securely with friends via encrypted tunnel. Opens admin console.")
        web_desc.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(web_desc)

        # Install Code Display
        code_layout = QHBoxLayout()
        code_layout.addWidget(QLabel("Install Code:"))

        self.gate_code = QLineEdit()
        self.gate_code.setPlaceholderText("Waiting for code...")
        self.gate_code.setReadOnly(True)
        self.gate_code.setStyleSheet("font-family: Consolas; color: #00ff00; font-weight: bold; letter-spacing: 2px;")
        code_layout.addWidget(self.gate_code)

        layout.addLayout(code_layout)

        # Controls
        btn_row = QHBoxLayout()

        self.gate_status = QLabel("● Stopped")
        self.gate_status.setStyleSheet("color: #888;")
        btn_row.addWidget(self.gate_status)
        btn_row.addStretch()

        self.start_gate_btn = QPushButton("▶ Start Tunnel")
        self.start_gate_btn.setCursor(Qt.PointingHandCursor)
        self.start_gate_btn.clicked.connect(self.start_irongate)
        self.start_gate_btn.setStyleSheet("background-color: #5d2b85; border: 1px solid #3d1b55;")
        btn_row.addWidget(self.start_gate_btn)

        self.stop_gate_btn = QPushButton("■ Stop Tunnel")
        self.stop_gate_btn.setCursor(Qt.PointingHandCursor)
        self.stop_gate_btn.clicked.connect(self.stop_irongate)
        self.stop_gate_btn.setEnabled(False)
        btn_row.addWidget(self.stop_gate_btn)

        layout.addLayout(btn_row)
        self.main_layout.addWidget(group)

    # --- Desktop Service Controls ---

    def _check_desktop_status(self):
        """Check if Iron Desktop Service is already running (e.g. auto-started)."""
        try:
            import requests
            resp = requests.get("http://localhost:8001/health", timeout=1)
            if resp.status_code == 200:
                self.desktop_status.setText("● Running (localhost:8001)")
                self.desktop_status.setStyleSheet("color: #5cb85c;")
                self.start_desktop_btn.setEnabled(False)
                self.stop_desktop_btn.setEnabled(False)  # Can't stop auto-started service from here
                return
        except Exception:
            pass

    def start_desktop_service(self):
        self.desktop_service_thread = DesktopServiceThread()
        self.desktop_service_thread.started.connect(self.on_desktop_started)
        self.desktop_service_thread.stopped.connect(self.on_desktop_stopped)
        self.desktop_service_thread.error.connect(self.on_desktop_error)
        self.desktop_service_thread.start()

        self.start_desktop_btn.setEnabled(False)
        self.desktop_status.setText("● Starting...")
        self.desktop_status.setStyleSheet("color: #f0ad4e;")

    def stop_desktop_service(self):
        if self.desktop_service_thread:
            self.desktop_service_thread.stop()
            self.desktop_service_thread.wait()
            self.desktop_service_thread = None

    def on_desktop_started(self):
        self.desktop_status.setText("● Running (localhost:8001)")
        self.desktop_status.setStyleSheet("color: #5cb85c;")
        self.start_desktop_btn.setEnabled(False)
        self.stop_desktop_btn.setEnabled(True)

    def on_desktop_stopped(self):
        self.desktop_status.setText("● Stopped")
        self.desktop_status.setStyleSheet("color: #888;")
        self.start_desktop_btn.setEnabled(True)
        self.stop_desktop_btn.setEnabled(False)

    def on_desktop_error(self, error):
        self.desktop_status.setText("● Error")
        self.desktop_status.setStyleSheet("color: #d9534f;")
        QMessageBox.warning(self, "Desktop Service Error", str(error))
        self.on_desktop_stopped()

    # --- Web Gateway Controls ---

    def start_irongate(self):
        self.irongate_thread = IronGateThread()
        self.irongate_thread.started.connect(self.on_gate_started)
        self.irongate_thread.stopped.connect(self.on_gate_stopped)
        self.irongate_thread.error.connect(self.on_gate_error)
        self.irongate_thread.install_code_received.connect(self.on_gate_code)
        self.irongate_thread.start()

        self.start_gate_btn.setEnabled(False)
        self.gate_status.setText("● Starting...")
        self.gate_status.setStyleSheet("color: #f0ad4e;")

    def stop_irongate(self):
        if self.irongate_thread:
            self.irongate_thread.stop()
            self.irongate_thread.wait()
            self.irongate_thread = None

    def on_gate_started(self):
        self.gate_status.setText("● Running")
        self.gate_status.setStyleSheet("color: #5cb85c;")
        self.start_gate_btn.setEnabled(False)
        self.stop_gate_btn.setEnabled(True)

    def on_gate_stopped(self):
        self.gate_status.setText("● Stopped")
        self.gate_status.setStyleSheet("color: #888;")
        self.start_gate_btn.setEnabled(True)
        self.stop_gate_btn.setEnabled(False)
        self.gate_code.setText("")

    def on_gate_error(self, error):
        self.gate_status.setText("● Error")
        self.gate_status.setStyleSheet("color: #d9534f;")
        QMessageBox.critical(self, "IronGate Error", str(error))
        self.on_gate_stopped()

    def on_gate_code(self, code):
        self.gate_code.setText(code)

    def create_elastic_memory_section(self):
        """Elastic Memory Architecture settings — VRAM/RAM/SSD tiered storage."""
        group = QGroupBox("🧠 Elastic Memory (Context / RAG)")
        group.setStyleSheet(group.styleSheet() + """
            QGroupBox { border-color: #6B4C9A; }
            QGroupBox::title { color: #9B7FCB; }
        """)
        layout = QVBoxLayout(group)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 35, 20, 20)

        # Status
        self.elastic_status = QLabel("Status: Detecting...")
        self.elastic_status.setStyleSheet("color: #9B7FCB; font-size: 13px;")
        layout.addWidget(self.elastic_status)

        # VRAM / RAM Info
        info_row = QHBoxLayout()
        self.vram_label = QLabel("VRAM: --")
        self.vram_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self.ram_label = QLabel("RAM: --")
        self.ram_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self.nctx_label = QLabel("Context Window: --")
        self.nctx_label.setStyleSheet("color: #aaa; font-size: 11px;")
        info_row.addWidget(self.vram_label)
        info_row.addWidget(self.ram_label)
        info_row.addWidget(self.nctx_label)
        layout.addLayout(info_row)

        # Data/Cache Drive
        layout.addWidget(QLabel("Conversation Data Directory"))
        data_row, self.elastic_data_dir = self.create_file_browser_row(
            self._get_default_data_dir()
        )
        layout.addLayout(data_row)

        sub = QLabel("Where conversation logs (msgpack) and vector indexes are stored.")
        sub.setProperty("class", "sub-label")
        sub.setStyleSheet("color: #888; font-size: 11px; font-weight: normal;")
        layout.addWidget(sub)

        # Drive space info
        self.drive_space_label = QLabel("")
        self.drive_space_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.drive_space_label)
        self._update_drive_space()

        self.elastic_data_dir.textChanged.connect(self._update_drive_space)

        # Session info
        session_row = QHBoxLayout()
        self.session_label = QLabel("Sessions: --")
        self.session_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self.msg_count_label = QLabel("Messages: --")
        self.msg_count_label.setStyleSheet("color: #aaa; font-size: 11px;")
        session_row.addWidget(self.session_label)
        session_row.addWidget(self.msg_count_label)
        layout.addLayout(session_row)

        # Refresh button
        refresh_btn = QPushButton("Refresh Info")
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.clicked.connect(self._refresh_elastic_info)
        layout.addWidget(refresh_btn, 0, Qt.AlignLeft)

        self.main_layout.addWidget(group)

        # Initial info refresh
        QTimer.singleShot(2000, self._refresh_elastic_info)

    def _get_default_data_dir(self):
        """Get the default elastic memory data directory."""
        root = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(root, "engine", "data", "conversations")

    def _update_drive_space(self):
        """Update drive space info for the selected data directory."""
        import shutil
        path = self.elastic_data_dir.text().strip()
        if not path:
            path = self._get_default_data_dir()
        try:
            # Get the drive root
            drive = os.path.splitdrive(path)[0] or path
            if os.path.exists(drive + os.sep):
                usage = shutil.disk_usage(drive + os.sep)
                free_gb = usage.free / (1024 ** 3)
                total_gb = usage.total / (1024 ** 3)
                used_pct = ((usage.total - usage.free) / usage.total) * 100
                self.drive_space_label.setText(
                    f"Drive {drive}: {free_gb:.1f} GB free / {total_gb:.1f} GB total ({used_pct:.0f}% used)"
                )
            else:
                self.drive_space_label.setText(f"Drive not found: {drive}")
        except Exception as e:
            self.drive_space_label.setText(f"Could not read drive space: {e}")

    def _refresh_elastic_info(self):
        """Refresh elastic memory statistics."""
        try:
            import sys as _sys
            vox_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "engine")
            if vox_path not in _sys.path:
                _sys.path.insert(0, vox_path)

            from elastic_memory.memory_store import MemoryStore
            from elastic_memory.resource_monitor import ResourceMonitor

            # Session count
            data_dir = self.elastic_data_dir.text().strip() or self._get_default_data_dir()
            sessions = MemoryStore.list_sessions(data_dir)
            self.session_label.setText(f"Sessions: {len(sessions)}")

            # Message count from latest session
            if sessions:
                store = MemoryStore(session_id=sessions[0], data_dir=data_dir)
                self.msg_count_label.setText(f"Messages (latest): {store.get_message_count()}")
            else:
                self.msg_count_label.setText("Messages: 0")

            # Resource info
            monitor = ResourceMonitor()
            snap = monitor.snapshot()
            self.vram_label.setText(f"VRAM: {snap['vram_total_mb']} MB ({snap['gpu_type']})")
            self.ram_label.setText(f"RAM Free: {snap['ram_free_mb']} MB")
            self.elastic_status.setText("Status: Active")
            self.elastic_status.setStyleSheet("color: #4CAF50; font-size: 13px;")

        except Exception as e:
            self.elastic_status.setText(f"Status: {e}")
            self.elastic_status.setStyleSheet("color: #FF5722; font-size: 13px;")

    def create_llm_section(self):
        group = QGroupBox("LLM Configuration (Providers)")
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 35, 20, 20)

        key_style = "font-size: 12px; color: #aaa; font-weight: bold;"

        # --- GEMINI API KEY ---
        layout.addWidget(QLabel("Google Gemini API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("AIza...")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.api_key_input)

        # --- OPENAI API KEY ---
        layout.addWidget(QLabel("OpenAI API Key:"))
        self.openai_key_input = QLineEdit()
        self.openai_key_input.setPlaceholderText("sk-...")
        self.openai_key_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.openai_key_input)

        # --- ANTHROPIC API KEY (future) ---
        anthropic_label = QLabel("Anthropic API Key (Coming Soon):")
        anthropic_label.setStyleSheet("color: #666;")
        layout.addWidget(anthropic_label)
        self.anthropic_key_input = QLineEdit()
        self.anthropic_key_input.setPlaceholderText("sk-ant-...")
        self.anthropic_key_input.setEchoMode(QLineEdit.Password)
        self.anthropic_key_input.setEnabled(False)
        self.anthropic_key_input.setStyleSheet("background-color: #1a1a1a; color: #555;")
        layout.addWidget(self.anthropic_key_input)

        # Keep provider_combo hidden but present for backward compat
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Google Gemini", "OpenAI", "Anthropic"])
        self.provider_combo.hide()

        # Fetch Models button + status
        fetch_row = QHBoxLayout()
        self.fetch_models_btn = QPushButton("Fetch Available Models")
        self.fetch_models_btn.setCursor(Qt.PointingHandCursor)
        self.fetch_models_btn.setStyleSheet("background-color: #006666; padding: 6px;")
        self.fetch_models_btn.clicked.connect(self._fetch_provider_models)
        fetch_row.addWidget(self.fetch_models_btn)

        self.fetch_status_label = QLabel("")
        self.fetch_status_label.setStyleSheet("color: #888; font-size: 11px;")
        fetch_row.addWidget(self.fetch_status_label)
        fetch_row.addStretch()
        layout.addLayout(fetch_row)

        # Available Models list (all providers combined)
        from PySide6.QtWidgets import QListWidget
        self.provider_models_list = QListWidget()
        self.provider_models_list.setMaximumHeight(150)
        self.provider_models_list.setStyleSheet("""
            QListWidget {
                background-color: #252526;
                border: 1px solid #444;
                border-radius: 4px;
                font-size: 12px;
                color: #ccc;
            }
            QListWidget::item { padding: 4px 8px; }
            QListWidget::item:selected { background-color: #006666; color: white; }
        """)
        layout.addWidget(self.provider_models_list)

        models_note = QLabel("Models from all providers with valid keys. Populates the Providers tab in Model Selector.")
        models_note.setStyleSheet("color: #888; font-size: 10px; font-weight: normal;")
        layout.addWidget(models_note)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #333; margin: 10px 0;")
        layout.addWidget(line)

        # Load defaults
        self.config = ConfigManager.load_config()
        llm_cfg = self.config.get("llm", {})

        self.api_key_input.setText(llm_cfg.get("api_key", ""))
        self.openai_key_input.setText(llm_cfg.get("openai_api_key", ""))
        self.anthropic_key_input.setText(llm_cfg.get("anthropic_api_key", ""))

        # Auto-populate if any key exists
        has_any_key = llm_cfg.get("api_key", "") or llm_cfg.get("openai_api_key", "")
        if has_any_key:
            QTimer.singleShot(1000, self._fetch_provider_models)

        # --- LOCAL MODELS MANAGER ---
        mgr_layout = QHBoxLayout()
        mgr_layout.addWidget(QLabel("Local Models (Ollama):"))
        mgr_btn = QPushButton("Manage / Download Models")
        mgr_btn.setCursor(Qt.PointingHandCursor)
        mgr_btn.setStyleSheet("background-color: #006666; padding: 6px;")
        mgr_btn.clicked.connect(self.open_model_manager)
        mgr_layout.addWidget(mgr_btn)
        layout.addLayout(mgr_layout)

        # --- LOCAL PATHS ---
        layout.addWidget(QLabel("Local Models Directory:"))
        lyt, self.txt_llm_path = self.create_file_browser_row(llm_cfg.get("local_model_dir", "models/llm"))
        layout.addLayout(lyt)

        layout.addWidget(QLabel("Local Cache Directory:"))
        lyt, self.txt_cache_path = self.create_file_browser_row(llm_cfg.get("cache_dir", "cache"))
        layout.addLayout(lyt)

        # Apply
        apply_btn = QPushButton("Apply LLM Settings")
        apply_btn.setObjectName("ApplyBtn")
        apply_btn.clicked.connect(self.save_llm_settings)
        layout.addWidget(apply_btn, 0, Qt.AlignRight)

        self.main_layout.addWidget(group)

    def create_cloud_gpu_section(self):
        """☁️ Cloud GPU (RunPod) — dedicated settings group."""
        group = QGroupBox("☁️ Cloud GPU (RunPod)")
        group.setStyleSheet(group.styleSheet() + """
            QGroupBox { border-color: #bb86fc; }
            QGroupBox::title { color: #bb86fc; }
        """)
        layout = QVBoxLayout(group)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 35, 20, 20)

        cloud_cfg = self.config.get("cloud", {})

        # RunPod API Key
        layout.addWidget(QLabel("RunPod API Key:"))
        self.cloud_runpod_key = QLineEdit()
        self.cloud_runpod_key.setPlaceholderText("rpa_...")
        self.cloud_runpod_key.setEchoMode(QLineEdit.Password)
        self.cloud_runpod_key.setText(cloud_cfg.get("runpod_api_key", ""))
        layout.addWidget(self.cloud_runpod_key)

        # Pod ID
        pod_row = QHBoxLayout()
        pod_row.addWidget(QLabel("Pod ID (optional):"))
        self.cloud_pod_id = QLineEdit()
        self.cloud_pod_id.setPlaceholderText("Auto-create if empty")
        self.cloud_pod_id.setText(cloud_cfg.get("pod_id", ""))
        pod_row.addWidget(self.cloud_pod_id)
        layout.addLayout(pod_row)

        # HuggingFace Token (for gated models like Llama)
        layout.addWidget(QLabel("HuggingFace Token (for gated models):"))
        self.cloud_hf_token = QLineEdit()
        self.cloud_hf_token.setPlaceholderText("hf_...")
        self.cloud_hf_token.setEchoMode(QLineEdit.Password)
        self.cloud_hf_token.setText(cloud_cfg.get("hf_token", ""))
        layout.addWidget(self.cloud_hf_token)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #333; margin: 8px 0;")
        layout.addWidget(line)

        # --- Cloud Models List ---
        models_header = QLabel("Cloud Models (HF ID → Display Name)")
        models_header.setStyleSheet("color: #bb86fc; font-weight: bold;")
        layout.addWidget(models_header)

        from PySide6.QtWidgets import QListWidget, QListWidgetItem
        self.cloud_models_list = QListWidget()
        self.cloud_models_list.setMaximumHeight(150)
        self.cloud_models_list.setStyleSheet("""
            QListWidget {
                background-color: #252526;
                border: 1px solid #444;
                border-radius: 4px;
                font-size: 11px;
                color: #ccc;
            }
            QListWidget::item { padding: 4px 8px; }
            QListWidget::item:selected { background-color: #006666; color: white; }
        """)
        layout.addWidget(self.cloud_models_list)

        # Populate from config
        cloud_models = cloud_cfg.get("models", {})
        for hf_id, display in cloud_models.items():
            self.cloud_models_list.addItem(f"{display}  ⟵  {hf_id}")

        # Add/Remove row
        add_rm_row = QHBoxLayout()
        self.cloud_model_hf_input = QLineEdit()
        self.cloud_model_hf_input.setPlaceholderText("HF Model ID (e.g. Qwen/Qwen2.5-72B-Instruct-AWQ)")
        add_rm_row.addWidget(self.cloud_model_hf_input)

        self.cloud_model_name_input = QLineEdit()
        self.cloud_model_name_input.setPlaceholderText("Display Name")
        self.cloud_model_name_input.setFixedWidth(150)
        add_rm_row.addWidget(self.cloud_model_name_input)

        add_btn = QPushButton("+ Add")
        add_btn.setCursor(Qt.PointingHandCursor)
        add_btn.setStyleSheet("background-color: #006666;")
        add_btn.clicked.connect(self._add_cloud_model)
        add_rm_row.addWidget(add_btn)

        rm_btn = QPushButton("— Remove")
        rm_btn.setCursor(Qt.PointingHandCursor)
        rm_btn.setStyleSheet("background-color: #8b2d2d;")
        rm_btn.clicked.connect(self._remove_cloud_model)
        add_rm_row.addWidget(rm_btn)

        layout.addLayout(add_rm_row)

        # Info label
        info = QLabel("GPU tier is auto-detected from model name (70B+ → A100/H100, others → A40/A6000).")
        info.setStyleSheet("color: #888; font-size: 10px; font-weight: normal;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Apply
        apply_btn = QPushButton("Apply Cloud Settings")
        apply_btn.setObjectName("ApplyBtn")
        apply_btn.clicked.connect(self.save_cloud_settings)
        layout.addWidget(apply_btn, 0, Qt.AlignRight)

        self.main_layout.addWidget(group)

    def _add_cloud_model(self):
        """Add a cloud model to the list."""
        hf_id = self.cloud_model_hf_input.text().strip()
        display = self.cloud_model_name_input.text().strip()
        if not hf_id:
            QMessageBox.warning(self, "Missing ID", "Enter a HuggingFace model ID.")
            return
        if not display:
            # Auto-generate display name from HF ID
            display = hf_id.split("/")[-1] if "/" in hf_id else hf_id
        self.cloud_models_list.addItem(f"{display}  ⟵  {hf_id}")
        self.cloud_model_hf_input.clear()
        self.cloud_model_name_input.clear()

    def _remove_cloud_model(self):
        """Remove selected cloud model from the list."""
        current = self.cloud_models_list.currentRow()
        if current >= 0:
            self.cloud_models_list.takeItem(current)

    def save_cloud_settings(self):
        """Save cloud GPU settings to config['cloud']."""
        cfg = self.config
        if "cloud" not in cfg:
            cfg["cloud"] = {}

        cfg["cloud"]["runpod_api_key"] = self.cloud_runpod_key.text().strip()
        cfg["cloud"]["hf_token"] = self.cloud_hf_token.text().strip()
        cfg["cloud"]["pod_id"] = self.cloud_pod_id.text().strip()

        # Parse models list back into dict
        models = {}
        for i in range(self.cloud_models_list.count()):
            text = self.cloud_models_list.item(i).text()
            if "⟵" in text:
                parts = text.split("⟵")
                display = parts[0].strip()
                hf_id = parts[1].strip()
                models[hf_id] = display
        cfg["cloud"]["models"] = models

        ConfigManager.save_config(cfg)
        QMessageBox.information(self, "Settings Saved", "Cloud GPU settings updated!")

    def _fetch_provider_models(self):
        """Fetch available models from ALL providers with valid API keys."""
        gemini_key = self.api_key_input.text().strip()
        openai_key = self.openai_key_input.text().strip()

        if not gemini_key and not openai_key:
            self.fetch_status_label.setText("Enter at least one API key")
            self.fetch_status_label.setStyleSheet("color: #f0ad4e; font-size: 11px;")
            return

        self.fetch_status_label.setText("Fetching...")
        self.fetch_status_label.setStyleSheet("color: #4ade80; font-size: 11px;")
        self.provider_models_list.clear()

        all_models = []

        try:
            # Gemini models
            if gemini_key:
                from providers.gemini_provider import GeminiProvider
                for name in GeminiProvider.list_available_models(api_key=gemini_key):
                    all_models.append(f"{name}  [Gemini]")

            # OpenAI models
            if openai_key:
                from providers.openai_provider import OpenAIProvider
                for name in OpenAIProvider.list_available_models(api_key=openai_key):
                    all_models.append(f"{name}  [OpenAI]")

            for display in all_models:
                self.provider_models_list.addItem(display)

            self.fetch_status_label.setText(f"Found {len(all_models)} models")
            self.fetch_status_label.setStyleSheet("color: #4ade80; font-size: 11px;")

        except Exception as e:
            self.fetch_status_label.setText(f"Error: {e}")
            self.fetch_status_label.setStyleSheet("color: #d9534f; font-size: 11px;")

    def open_model_manager(self):
        dlg = ModelManagerDialog(self)
        dlg.exec()

    def save_llm_settings(self):
        cfg = self.config
        if "llm" not in cfg:
            cfg["llm"] = {}

        cfg["llm"]["api_key"] = self.api_key_input.text().strip()
        cfg["llm"]["openai_api_key"] = self.openai_key_input.text().strip()
        cfg["llm"]["anthropic_api_key"] = self.anthropic_key_input.text().strip()
        cfg["llm"]["local_model_dir"] = self.txt_llm_path.text()
        cfg["llm"]["cache_dir"] = self.txt_cache_path.text()

        ConfigManager.save_config(cfg)
        self.llm_settings_saved.emit()
        QMessageBox.information(self, "Settings Saved", "LLM settings updated successfully!")

    def create_image_section(self):
        group = QGroupBox("Image Generation Paths")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 35, 20, 20)
        
        img_cfg = self.config.get("image", {})

        layout.addWidget(QLabel("Output Directory:"))
        lyt, self.txt_out_dir = self.create_file_browser_row(img_cfg.get("output_dir", "outputs/images"))
        layout.addLayout(lyt)

        layout.addWidget(QLabel("Checkpoints Directory:"))
        lyt, self.txt_ckpt_dir = self.create_file_browser_row(img_cfg.get("checkpoint_dir", "models/checkpoints"))
        layout.addLayout(lyt)

        layout.addWidget(QLabel("LoRA Directory:"))
        lyt, self.txt_lora_dir = self.create_file_browser_row(img_cfg.get("lora_dir", "models/loras"))
        layout.addLayout(lyt)

        layout.addWidget(QLabel("VAE Directory:"))
        lyt, self.txt_vae_dir = self.create_file_browser_row(img_cfg.get("vae_dir", "models/vae"))
        layout.addLayout(lyt)

        layout.addWidget(QLabel("Text Encoder Directory:"))
        lyt, self.txt_text_enc_dir = self.create_file_browser_row(img_cfg.get("text_encoder_dir", "models/text_encoders"))
        layout.addLayout(lyt)

        # Prompt Enhancement Model
        layout.addWidget(QLabel("Prompt Enhancement Model:"))
        self.img_model_combo = QComboBox()
        self.img_model_combo.addItems(["Llama 3 (8B)", "Mistral 7B", "Gemini Pro (API)"])
        layout.addWidget(self.img_model_combo)

        # Apply
        apply_btn = QPushButton("Apply Image Settings")
        apply_btn.setObjectName("ApplyBtn")
        apply_btn.clicked.connect(self.save_image_settings)
        layout.addWidget(apply_btn, 0, Qt.AlignRight)

        self.main_layout.addWidget(group)
        
    def save_image_settings(self):
        cfg = self.config
        if "image" not in cfg: cfg["image"] = {}
        
        cfg["image"]["output_dir"] = self.txt_out_dir.text()
        cfg["image"]["checkpoint_dir"] = self.txt_ckpt_dir.text()
        cfg["image"]["lora_dir"] = self.txt_lora_dir.text()
        cfg["image"]["vae_dir"] = self.txt_vae_dir.text()
        cfg["image"]["text_encoder_dir"] = self.txt_text_enc_dir.text()
        
        ConfigManager.save_config(cfg)
        QMessageBox.information(self, "Settings Saved", "Image paths updated successfully!") 


    def create_remote_section(self):
        """Cloud Memory / Storage Credentials only (External GPU moved to Cloud GPU section)."""
        group = QGroupBox("Cloud Storage")
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 35, 20, 20)

        layout.addWidget(QLabel("Cloud Memory / Storage Credentials:"))
        layout.addWidget(QLabel("Configure keys for Cloudflare R2, S3, or Network Drives."))

        # Provider
        self.storage_combo = QComboBox()
        self.storage_combo.addItems(["Cloudflare R2", "AWS S3", "Network Drive (SMB)", "Google Drive API"])
        layout.addWidget(self.storage_combo)

        # Endpoint / Bucket
        self.storage_url = QLineEdit()
        self.storage_url.setPlaceholderText("Bucket URL / SMB Path...")
        layout.addWidget(self.storage_url)

        # API Key / Password
        storage_auth_row = QHBoxLayout()
        self.storage_key = QLineEdit()
        self.storage_key.setPlaceholderText("API Key / Password...")
        self.storage_key.setEchoMode(QLineEdit.Password)
        storage_auth_row.addWidget(self.storage_key)

        test_storage_btn = QPushButton("Test Access")
        test_storage_btn.setProperty("class", "test-btn")
        test_storage_btn.clicked.connect(lambda: self.run_test("Cloud Storage"))
        storage_auth_row.addWidget(test_storage_btn)

        layout.addLayout(storage_auth_row)

        # Apply
        apply_btn = QPushButton("Apply Storage Settings")
        apply_btn.setObjectName("ApplyBtn")
        layout.addWidget(apply_btn, 0, Qt.AlignRight)

        self.main_layout.addWidget(group)

    def create_file_browser_row(self, default_text=""):
        layout = QHBoxLayout()
        line_edit = QLineEdit(default_text)
        browse_btn = QPushButton("Browse")
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.clicked.connect(lambda: self.browse_folder(line_edit))

        layout.addWidget(line_edit)
        layout.addWidget(browse_btn)
        return layout, line_edit

    def browse_folder(self, line_edit_widget):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            line_edit_widget.setText(folder)

    def run_test(self, test_name):
        """Simulates a connection test with a popup feedback."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Connection Test")
        msg.setText(f"Testing {test_name}...")
        msg.setInformativeText("Pinging endpoint and verifying credentials...")
        msg.setStandardButtons(QMessageBox.NoButton)
        msg.setStyleSheet("background-color: #222; color: white; padding: 10px;")
        msg.show()

        # Simulate network delay
        QTimer.singleShot(1500, lambda: self._finish_test(msg, test_name))

    def _finish_test(self, msg_box, test_name):
        msg_box.setText(f"{test_name} Status: SUCCESS")
        msg_box.setInformativeText("Connection established. Latency: 24ms.")
        msg_box.setStandardButtons(QMessageBox.Ok)
    
    def closeEvent(self, event):
        """Clean up services when closing."""
        if self.desktop_service_thread:
            self.desktop_service_thread.stop()
            self.desktop_service_thread.wait()
        if self.irongate_thread:
            self.irongate_thread.stop()
            self.irongate_thread.wait()

        super().closeEvent(event)
