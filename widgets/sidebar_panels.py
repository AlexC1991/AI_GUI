from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame,
    QComboBox, QSpinBox, QAbstractSpinBox, QProgressBar
)
from PySide6.QtCore import Qt
import platform
import psutil
import subprocess
import sys

# ============================================
# 1. CHAT OPTIONS PANEL
# ============================================
class ChatOptionsPanel(QWidget):
    def __init__(self, parent_window=None):
        super().__init__()

        self.setStyleSheet("""
            QWidget { color: #CCCCCC; font-family: Segoe UI, sans-serif; font-size: 12px; background: transparent; }
            QLabel { font-weight: bold; margin-top: 5px; border: none; }
            QPushButton, QComboBox, QSpinBox {
                background-color: #333; border: 1px solid #444; border-radius: 4px; color: white; padding: 5px; min-height: 15px;
            }
            #ModeBox, #MemoryBox, #CloudBox, #ModelBox { padding-left: 8px; }
            QComboBox::drop-down { border: none; }
            QSpinBox::up-button, QSpinBox::down-button { width: 0px; border: none; }
            QPushButton { background-color: #2E2E2E; }
            QPushButton:hover { background-color: #3E3E3E; }
            #ClearBtn { background-color: #4a0000; border: 1px solid #600000; }
            #ClearBtn:hover { background-color: #6a0000; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Chat Section:", styleSheet="font-size: 14px; color: #FFF;"), alignment=Qt.AlignHCenter)

        self.upload_btn = QPushButton("Upload")
        layout.addWidget(self.upload_btn)

        layout.addWidget(QLabel("Mode:"), alignment=Qt.AlignHCenter)
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("ModeBox")
        self.mode_combo.addItems(["Local", "Provider"])
        self.mode_combo.currentTextChanged.connect(self._update_models)
        layout.addWidget(self.mode_combo)

        layout.addWidget(QLabel("Token Limit:"), alignment=Qt.AlignHCenter)
        self.token_spin = QSpinBox()
        self.token_spin.setObjectName("TokenBox")
        self.token_spin.setRange(100, 32000)
        self.token_spin.setValue(500)
        self.token_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.token_spin.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.token_spin)

        layout.addWidget(QLabel("Memory:"), alignment=Qt.AlignHCenter)
        self.memory_combo = QComboBox()
        self.memory_combo.setObjectName("MemoryBox")
        self.memory_combo.addItems(["None", "Low", "Medium", "High"])
        self.memory_combo.setCurrentText("Low")
        layout.addWidget(self.memory_combo)

        layout.addWidget(QLabel("Cloud Memory:"), alignment=Qt.AlignHCenter)
        self.cloud_combo = QComboBox()
        self.cloud_combo.setObjectName("CloudBox")
        self.cloud_combo.addItems(["On", "Off"])
        layout.addWidget(self.cloud_combo)

        layout.addWidget(QLabel("Model:"), alignment=Qt.AlignHCenter)
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("ModelBox")
        self.model_combo.addItems(["Llama 3", "Mistral", "Gemma"])
        layout.addWidget(self.model_combo)

        layout.addSpacing(15)
        self.clear_btn = QPushButton("Clear Chat")
        self.clear_btn.setObjectName("ClearBtn")
        self.clear_btn.setFixedHeight(30)
        layout.addWidget(self.clear_btn)

        layout.addSpacing(10)
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #00FF00; font-weight: bold;")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def _update_models(self, mode):
        self.model_combo.clear()
        if mode == "Local":
            self.model_combo.addItems(["Llama 3", "Mistral", "Gemma"])
        else:
            self.model_combo.addItems(["GPT-4o", "Claude 3.5", "Gemini Pro"])

    def update_status(self, state):
        color = "#00FF00" if state.lower() == "idle" else "#00FFFF"
        self.status_label.setText(f"Status: {state.title()}")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

# ============================================
# 2. IMAGE GEN PANEL
# ============================================
class ImageGenPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignTop)

        # Header
        layout.addWidget(QLabel("System Info:", styleSheet="font-weight:bold; color:white; font-size:14px; margin-top:10px;"))

        # --- HARDWARE STATS (Compact) ---
        specs = self.get_system_specs()

        self.add_stat(layout, f"GPU: {specs['gpu']}", "Load: 0%", 0)
        self.add_stat(layout, "VRAM (Est):", "4.2 / 24 GB", 18)
        self.add_stat(layout, f"CPU: {specs['cpu']}", "Load: 12%", 12)
        self.add_stat(layout, "System RAM:", specs['ram_text'], specs['ram_percent'])

        # --- REFINER MODEL ---
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #444; margin: 10px 0;")
        layout.addWidget(line)

        layout.addWidget(QLabel("Refiner Model:", styleSheet="color:#CCC; font-weight:bold;"))
        self.refiner_combo = QComboBox()
        self.refiner_combo.addItems(["Local: SDXL Refiner", "Provider: Gemini Pro Vision", "None"])
        self.refiner_combo.setStyleSheet("""
            QComboBox { background: #333; border: 1px solid #444; color: white; padding: 5px; }
            QComboBox::drop-down { border: none; }
        """)
        layout.addWidget(self.refiner_combo)

        # Spacer
        layout.addStretch()

        # --- ACTION BUTTONS (Above Settings) ---
        self.gen_btn = QPushButton("GENERATE")
        self.gen_btn.setFixedHeight(45)
        self.gen_btn.setCursor(Qt.PointingHandCursor)
        self.gen_btn.setStyleSheet("""
            QPushButton { 
                background-color: #006666; color: white; font-weight: bold; 
                border: 1px solid #004d4d; border-radius: 4px; font-size: 14px;
            }
            QPushButton:hover { background-color: #008080; }
        """)
        layout.addWidget(self.gen_btn)

        self.abort_btn = QPushButton("ABORT")
        self.abort_btn.setFixedHeight(30)
        self.abort_btn.setCursor(Qt.PointingHandCursor)
        self.abort_btn.setStyleSheet("""
            QPushButton { 
                background-color: #442222; color: #FFCCCC; 
                border: 1px solid #663333; border-radius: 4px;
            }
            QPushButton:hover { background-color: #663333; }
        """)
        layout.addWidget(self.abort_btn)

        layout.addSpacing(10)

    def add_stat(self, layout, title, text, percent):
        container = QWidget()
        l = QVBoxLayout(container)
        l.setContentsMargins(0,0,0,0)
        l.setSpacing(2)

        # --- FIXED: QHBoxLayout is now imported ---
        row = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setStyleSheet("color:#CCC; font-size:10px; font-weight:bold;")
        val = QLabel(text)
        val.setStyleSheet("color:#AAA; font-size:10px;")
        val.setAlignment(Qt.AlignRight)
        row.addWidget(lbl, 1) # Stretch title
        row.addWidget(val)
        l.addLayout(row)

        bar = QProgressBar()
        bar.setFixedHeight(4)
        bar.setTextVisible(False)
        bar.setRange(0, 100)
        bar.setValue(int(percent))
        bar.setStyleSheet("QProgressBar { background: #222; border: none; border-radius: 2px; } QProgressBar::chunk { background: #00AA00; border-radius: 2px; }")
        l.addWidget(bar)

        layout.addWidget(container)

    def get_system_specs(self):
        specs = { "cpu": "Unknown", "gpu": "Unknown", "ram_text": "Unknown", "ram_percent": 0 }

        if sys.platform == "win32":
            try:
                cmd = "wmic cpu get name"
                output = subprocess.check_output(cmd, shell=True).decode().strip()
                lines = [line.strip() for line in output.split('\n') if line.strip()]
                if len(lines) > 1: specs['cpu'] = lines[1]
            except: specs['cpu'] = "CPU"

            try:
                cmd = "wmic path win32_VideoController get name"
                output = subprocess.check_output(cmd, shell=True).decode().strip()
                lines = [line.strip() for line in output.split('\n') if line.strip()]
                if len(lines) > 1: specs['gpu'] = lines[1]
            except: specs['gpu'] = "GPU"

        try:
            mem = psutil.virtual_memory()
            specs['ram_text'] = f"{round(mem.used/1024**3, 1)}/{round(mem.total/1024**3, 1)} GB"
            specs['ram_percent'] = mem.percent
        except: pass

        return specs

# ============================================
# 3. SYSTEM INFO PANEL
# ============================================
class SystemInfoPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 0, 15, 0)
        layout.setSpacing(10)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #444; margin-bottom: 5px;")
        layout.addWidget(line)

        title = QLabel("System Info:")
        title.setStyleSheet("font-weight: bold; color: #FFF; font-size: 14px; background: transparent;")
        layout.addWidget(title)

        specs = self.get_system_specs()

        self.add_stat_row(layout, "CPU:", specs['cpu'])
        self.add_stat_row(layout, "GPU:", specs['gpu'])

        self.add_stat_row(layout, "RAM:", specs['ram_text'])
        self.ram_bar = self.create_usage_bar(specs['ram_percent'])
        layout.addWidget(self.ram_bar)

        layout.addSpacing(5)
        self.add_stat_row(layout, "VRAM (Est):", "4.2 / 24 GB")
        self.vram_bar = self.create_usage_bar(18)
        layout.addWidget(self.vram_bar)

        layout.addSpacing(5)
        self.add_stat_row(layout, "OS:", specs['os'])
        self.add_stat_row(layout, "Version:", "v1.0.2")

        layout.addStretch()

    def add_stat_row(self, layout, label_text, value_text):
        container = QWidget()
        row = QVBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)

        lbl = QLabel(label_text)
        lbl.setStyleSheet("color: #AAA; font-size: 11px; font-weight: normal; background: transparent;")
        val = QLabel(value_text)
        val.setWordWrap(True)
        val.setStyleSheet("color: #FFFFFF; font-size: 12px; font-weight: bold; background: transparent;")

        row.addWidget(lbl)
        row.addWidget(val)
        layout.addWidget(container)

    def create_usage_bar(self, value):
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(value))
        bar.setFixedHeight(6)
        bar.setTextVisible(False)
        bar.setStyleSheet("QProgressBar { border: 1px solid #444; border-radius: 3px; background: #222; } QProgressBar::chunk { background: #00FF00; }")
        return bar

    def get_system_specs(self):
        specs = { "cpu": "Unknown", "gpu": "Unknown", "ram_text": "Unknown", "ram_percent": 0, "os": f"{platform.system()} {platform.release()}" }
        if sys.platform == "win32":
            try:
                cmd = "wmic cpu get name"
                output = subprocess.check_output(cmd, shell=True).decode().strip()
                lines = [line.strip() for line in output.split('\n') if line.strip()]
                if len(lines) > 1: specs['cpu'] = lines[1]
            except: specs['cpu'] = platform.processor()
        else: specs['cpu'] = platform.processor()

        try:
            mem = psutil.virtual_memory()
            specs['ram_text'] = f"{round(mem.used/1024**3, 1)}/{round(mem.total/1024**3, 1)} GB"
            specs['ram_percent'] = mem.percent
        except: pass

        if sys.platform == "win32":
            try:
                cmd = "wmic path win32_VideoController get name"
                output = subprocess.check_output(cmd, shell=True).decode().strip()
                lines = [line.strip() for line in output.split('\n') if line.strip()]
                if len(lines) > 1: specs['gpu'] = lines[1]
            except: pass
        return specs

# ============================================
# 4. PLACEHOLDER PANEL
# ============================================
class PlaceholderPanel(QWidget):
    def __init__(self, mode_name):
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel(f"{mode_name}\nSettings Coming Soon")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #666; font-style: italic; background: transparent;")
        layout.addWidget(label)