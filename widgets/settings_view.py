from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QScrollArea, QFrame, QFileDialog, QCheckBox, QHBoxLayout, QGroupBox, QMessageBox
)
from PySide6.QtCore import Qt, QTimer

class SettingsView(QScrollArea):
    def __init__(self):
        super().__init__()

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
            QLineEdit {
                background-color: #252526;
                color: #E0E0E0;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }
            QLineEdit:focus { border: 1px solid #008080; }
            
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
        self.create_llm_section()
        self.create_image_section()
        self.create_remote_section()

        self.setWidget(self.container)

    def create_llm_section(self):
        group = QGroupBox("LLM Configuration (Local & Provider)")
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 35, 20, 20)

        # --- CLOUD PROVIDER ---
        layout.addWidget(QLabel("Provider API Key (Auto-saves to .env):"))

        row1 = QHBoxLayout()
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Google Gemini", "OpenAI", "Anthropic", "Mistral API"])
        self.provider_combo.setFixedWidth(150)

        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("sk-...")
        self.api_key_input.setEchoMode(QLineEdit.Password)

        row1.addWidget(self.provider_combo)
        row1.addWidget(self.api_key_input)
        layout.addLayout(row1)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #333; margin: 10px 0;")
        layout.addWidget(line)

        # --- LOCAL PATHS ---
        layout.addWidget(QLabel("Local Models Directory:"))
        self.llm_path = self.create_file_browser_row()
        layout.addLayout(self.llm_path)

        layout.addWidget(QLabel("Local Cache Directory:"))
        self.cache_path = self.create_file_browser_row()
        layout.addLayout(self.cache_path)

        # Apply
        apply_btn = QPushButton("Apply LLM Settings")
        apply_btn.setObjectName("ApplyBtn")
        layout.addWidget(apply_btn, 0, Qt.AlignRight)

        self.main_layout.addWidget(group)

    def create_image_section(self):
        group = QGroupBox("Image Generation Paths")
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 35, 20, 20)

        layout.addWidget(QLabel("Output Directory:"))
        self.img_out_path = self.create_file_browser_row()
        layout.addLayout(self.img_out_path)

        layout.addWidget(QLabel("Checkpoints Directory:"))
        self.ckpt_path = self.create_file_browser_row()
        layout.addLayout(self.ckpt_path)

        layout.addWidget(QLabel("LoRA Directory:"))
        self.lora_path = self.create_file_browser_row()
        layout.addLayout(self.lora_path)

        layout.addWidget(QLabel("Prompt Refiner Model (LLM):"))
        self.img_model_combo = QComboBox()
        self.img_model_combo.addItems(["Llama 3 (8B)", "Mistral 7B", "Gemini Pro (API)"])
        layout.addWidget(self.img_model_combo)

        # Apply
        apply_btn = QPushButton("Apply Image Settings")
        apply_btn.setObjectName("ApplyBtn")
        layout.addWidget(apply_btn, 0, Qt.AlignRight)

        self.main_layout.addWidget(group)

    def create_remote_section(self):
        group = QGroupBox("Remote Resources & Cloud Storage")
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 35, 20, 20)

        # =======================================================
        # 1. EXTERNAL GPU WORKER (CONNECT TO...)
        # =======================================================
        layout.addWidget(QLabel("External GPU Worker (Client Mode):"))
        layout.addWidget(QLabel("Connect to an external RunPod or Desktop GPU.", styleSheet="color:#888; font-size:11px;"))

        gpu_row = QHBoxLayout()

        # Endpoint
        endpoint_layout = QVBoxLayout()
        endpoint_layout.addWidget(QLabel("Endpoint URL / IP:", styleSheet="font-weight:normal; font-size:11px; color:#AAA"))
        self.gpu_ip_input = QLineEdit()
        self.gpu_ip_input.setPlaceholderText("wss://runpod-id... or 192.168.1.X")
        endpoint_layout.addWidget(self.gpu_ip_input)

        # Port
        port_layout = QVBoxLayout()
        port_layout.addWidget(QLabel("Port:", styleSheet="font-weight:normal; font-size:11px; color:#AAA"))
        self.gpu_port_input = QLineEdit()
        self.gpu_port_input.setPlaceholderText("8188")
        self.gpu_port_input.setFixedWidth(80)
        port_layout.addWidget(self.gpu_port_input)

        gpu_row.addLayout(endpoint_layout)
        gpu_row.addLayout(port_layout)
        layout.addLayout(gpu_row)

        # Auth & Test
        auth_row = QHBoxLayout()
        self.gpu_auth_input = QLineEdit()
        self.gpu_auth_input.setPlaceholderText("Worker Auth Key (Optional)")
        self.gpu_auth_input.setEchoMode(QLineEdit.Password)
        auth_row.addWidget(self.gpu_auth_input)

        test_gpu_btn = QPushButton("Test Connection")
        test_gpu_btn.setProperty("class", "test-btn")
        test_gpu_btn.clicked.connect(lambda: self.run_test("GPU Worker"))
        auth_row.addWidget(test_gpu_btn)

        layout.addLayout(auth_row)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #333; margin: 15px 0;")
        layout.addWidget(line)

        # =======================================================
        # 2. REMOTE ACCESS (HOST SERVER)
        # =======================================================
        layout.addWidget(QLabel("Remote Access (Server Mode):"))
        layout.addWidget(QLabel("Allow other users to connect to THIS machine.", styleSheet="color:#888; font-size:11px;"))

        server_row = QHBoxLayout()

        # IP Bind
        ip_bind_layout = QVBoxLayout()
        ip_bind_layout.addWidget(QLabel("Bind IP:", styleSheet="font-weight:normal; font-size:11px; color:#AAA"))
        self.server_ip = QLineEdit("0.0.0.0")
        self.server_ip.setPlaceholderText("0.0.0.0 (All Interfaces)")
        ip_bind_layout.addWidget(self.server_ip)

        # Port
        server_port_layout = QVBoxLayout()
        server_port_layout.addWidget(QLabel("Port:", styleSheet="font-weight:normal; font-size:11px; color:#AAA"))
        self.server_port = QLineEdit("7860")
        self.server_port.setFixedWidth(80)
        server_port_layout.addWidget(self.server_port)

        server_row.addLayout(ip_bind_layout)
        server_row.addLayout(server_port_layout)
        layout.addLayout(server_row)

        # Active Toggle & Test
        server_action_row = QHBoxLayout()
        self.server_active = QCheckBox("Enable Remote Access")
        self.server_active.setStyleSheet("""
            QCheckBox { color: #E0E0E0; font-size: 13px; font-weight: bold; }
            QCheckBox::indicator { width: 18px; height: 18px; border: 1px solid #555; background: #222; border-radius: 4px; }
            QCheckBox::indicator:checked { background: #008080; border: 1px solid #008080; }
        """)
        server_action_row.addWidget(self.server_active)

        test_server_btn = QPushButton("Test Server Status")
        test_server_btn.setProperty("class", "test-btn")
        test_server_btn.clicked.connect(lambda: self.run_test("Remote Server"))
        server_action_row.addWidget(test_server_btn)

        layout.addLayout(server_action_row)

        # Separator
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #333; margin: 15px 0;")
        layout.addWidget(line2)

        # =======================================================
        # 3. CLOUD MEMORY CREDENTIALS
        # =======================================================
        layout.addWidget(QLabel("Cloud Memory / Storage Credentials:"))
        layout.addWidget(QLabel("Configure keys for Cloudflare R2, S3, or Network Drives.", styleSheet="color:#888; font-size:11px;"))

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
        apply_btn = QPushButton("Apply Remote Settings")
        apply_btn.setObjectName("ApplyBtn")
        layout.addWidget(apply_btn, 0, Qt.AlignRight)

        self.main_layout.addWidget(group)

    def create_file_browser_row(self):
        layout = QHBoxLayout()
        line_edit = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.clicked.connect(lambda: self.browse_folder(line_edit))

        layout.addWidget(line_edit)
        layout.addWidget(browse_btn)
        return layout

    def browse_folder(self, line_edit_widget):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            line_edit_widget.setText(folder)

    def run_test(self, test_name):
        """Simulates a connection test with a popup feedback."""
        # Visual feedback on the button itself could be added here
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