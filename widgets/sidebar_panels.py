"""
AI_GUI - Sidebar Panels
Fixed for Windows 11 (uses PowerShell instead of deprecated wmic)
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame,
    QComboBox, QSpinBox, QAbstractSpinBox, QProgressBar, QCheckBox
)
from PySide6.QtCore import Qt
import platform
import subprocess
import sys

# Try to import psutil, but don't crash if missing
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def get_system_specs_powershell():
    """
    Get system specs using PowerShell (Windows 11 compatible).
    Falls back to platform module if PowerShell fails.
    """
    specs = {
        "cpu": "Unknown CPU",
        "gpu": "Unknown GPU", 
        "ram_text": "Unknown",
        "ram_percent": 0,
        "os": f"{platform.system()} {platform.release()}"
    }
    
    if sys.platform != "win32":
        specs["cpu"] = platform.processor() or "Unknown CPU"
        return specs
    
    # CPU - PowerShell
    try:
        cmd = 'powershell -Command "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"'
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            specs["cpu"] = result.stdout.strip().split('\n')[0]
    except Exception:
        specs["cpu"] = platform.processor() or "Unknown CPU"
    
    # GPU - PowerShell
    try:
        cmd = 'powershell -Command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"'
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            # Get first GPU (might have multiple)
            gpus = [g.strip() for g in result.stdout.strip().split('\n') if g.strip()]
            if gpus:
                # Prefer discrete GPU over integrated
                for gpu in gpus:
                    if any(x in gpu.lower() for x in ['nvidia', 'radeon', 'geforce', 'rtx', 'gtx', 'rx ']):
                        specs["gpu"] = gpu
                        break
                else:
                    specs["gpu"] = gpus[0]
    except Exception:
        pass
    
    # RAM - psutil
    if PSUTIL_AVAILABLE:
        try:
            mem = psutil.virtual_memory()
            specs["ram_text"] = f"{round(mem.used/1024**3, 1)}/{round(mem.total/1024**3, 1)} GB"
            specs["ram_percent"] = mem.percent
        except Exception:
            pass
    
    return specs


# Cache specs so we don't call PowerShell repeatedly
_cached_specs = None

def get_cached_specs():
    global _cached_specs
    if _cached_specs is None:
        _cached_specs = get_system_specs_powershell()
    return _cached_specs


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
                background-color: #333; border: 1px solid #444; border-radius: 4px; color: white; padding: 5px; min-height: 25px;
            }
            #ModeBox, #MemoryBox, #CloudBox { padding-left: 8px; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2E2E2E; color: white; selection-background-color: #006666; min-width: 150px;
            }
            QSpinBox::up-button, QSpinBox::down-button { width: 0px; border: none; }
            QPushButton { background-color: #2E2E2E; }
            QPushButton:hover { background-color: #3E3E3E; }
            #ClearBtn { background-color: #4a0000; border: 1px solid #600000; }
            #ClearBtn:hover { background-color: #6a0000; }
            #ChangeModelBtn { background-color: #006666; border: 1px solid #004d4d; }
            #ChangeModelBtn:hover { background-color: #008080; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Chat Section:", styleSheet="font-size: 14px; color: #FFF;"), alignment=Qt.AlignHCenter)

        layout.addWidget(QLabel("Mode:"), alignment=Qt.AlignHCenter)
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("ModeBox")
        self.mode_combo.addItems(["VoxAI (Local)", "Provider"])
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

        # --- Model Section (Replaces Dropdown) ---
        layout.addSpacing(10)
        layout.addWidget(QLabel("Current Model:"), alignment=Qt.AlignHCenter)
        
        self.current_model_label = QLabel("(None)")
        self.current_model_label.setAlignment(Qt.AlignCenter)
        self.current_model_label.setStyleSheet("color: #00FFFF; font-weight: bold; font-size: 13px;")
        self.current_model_label.setWordWrap(True)
        layout.addWidget(self.current_model_label)
        
        self.change_model_btn = QPushButton("Change Model")
        self.change_model_btn.setObjectName("ChangeModelBtn")
        self.change_model_btn.setFixedHeight(30)
        layout.addWidget(self.change_model_btn)

        layout.addSpacing(10)
        self.reasoning_mode_chk = QCheckBox("Force Reasoning Mode")
        self.reasoning_mode_chk.setToolTip("Injects system prompt to force <think> tags")
        self.reasoning_mode_chk.setStyleSheet("color: #E0E0E0;")
        layout.addWidget(self.reasoning_mode_chk)

        layout.addSpacing(15)
        
        # --- Network Status Indicator ---
        net_layout = QHBoxLayout()
        net_layout.setContentsMargins(0, 0, 0, 0)
        net_layout.setSpacing(6)
        
        self.network_icon = QLabel("🌐")
        self.network_icon.setStyleSheet("font-size: 16px;")
        net_layout.addWidget(self.network_icon)
        
        self.network_label = QLabel("Checking...")
        self.network_label.setStyleSheet("color: #888; font-size: 11px;")
        net_layout.addWidget(self.network_label)
        net_layout.addStretch()
        
        layout.addLayout(net_layout)

        layout.addSpacing(10)
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #00FF00; font-weight: bold;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.local_models_cache = []  # Cache for local models
        self.selected_model = None    # Currently selected model data
    
    # --- Network Status Methods ---
    
    def set_network_status(self, status: str):
        """Update network status indicator. 
        status: 'online', 'offline', 'searching', 'checking'
        """
        if status == "online":
            self.network_icon.setText("🟢")
            self.network_label.setText("Online")
            self.network_label.setStyleSheet("color: #5cb85c; font-size: 11px;")
        elif status == "offline":
            self.network_icon.setText("🔴")
            self.network_label.setText("Offline")
            self.network_label.setStyleSheet("color: #d9534f; font-size: 11px;")
        elif status == "searching":
            self.network_icon.setText("🔍")
            self.network_label.setText("Searching...")
            self.network_label.setStyleSheet("color: #f0ad4e; font-size: 11px;")
        else:  # checking
            self.network_icon.setText("🌐")
            self.network_label.setText("Checking...")
            self.network_label.setStyleSheet("color: #888; font-size: 11px;")

    def set_local_models(self, models):
        """Called by MainWindow to update the list with real installed models"""
        self.local_models_cache = models if models else []
        
        # Auto-select first model if none selected
        if self.local_models_cache and self.selected_model is None:
            first = self.local_models_cache[0]
            if isinstance(first, dict):
                self.set_selected_model(first)
    
    def set_selected_model(self, model_data):
        """Set the currently selected model and update the label."""
        self.selected_model = model_data
        if isinstance(model_data, dict):
            # Extract short name (max 2 words from display name)
            display = model_data.get("display", "Unknown")
            parts = display.split()[:2]  # Take first 2 words
            short_name = " ".join(parts)
            self.current_model_label.setText(short_name)
        else:
            self.current_model_label.setText(str(model_data) if model_data else "(None)")
    
    def _update_models(self, mode):
        """Handle mode change - reset model selection for Provider mode"""
        if "VoxAI" not in mode:
            # Provider mode - show placeholder
            self.current_model_label.setText("API Model")
            self.selected_model = {"display": "API Model", "filename": "provider"}
        else:
            # Local mode - restore cached selection or first model
            if self.selected_model and self.selected_model.get("filename") != "provider":
                self.set_selected_model(self.selected_model)
            elif self.local_models_cache:
                self.set_selected_model(self.local_models_cache[0])

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

        # Get system specs (cached)
        specs = get_cached_specs()

        self.add_stat(layout, f"GPU:", specs['gpu'], 0)
        self.add_stat(layout, "VRAM (Est):", "6.0 / 8.0 GB", 75)
        self.add_stat(layout, f"CPU:", specs['cpu'][:30], 12)
        self.add_stat(layout, "System RAM:", specs['ram_text'], specs['ram_percent'])

        # Spacer
        layout.addStretch()

        # --- ACTION BUTTONS ---
        self.refresh_btn = QPushButton("Refresh Assets 🔄")
        self.refresh_btn.setFixedHeight(30)
        self.refresh_btn.setCursor(Qt.PointingHandCursor)
        self.refresh_btn.setStyleSheet("""
            QPushButton { 
                background-color: #333333; color: #CCC; 
                border: 1px solid #555; border-radius: 4px;
            }
            QPushButton:hover { background-color: #444444; color: white; }
        """)
        layout.addWidget(self.refresh_btn)

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
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(2)

        row = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setStyleSheet("color:#CCC; font-size:10px; font-weight:bold;")
        val = QLabel(text)
        val.setStyleSheet("color:#AAA; font-size:10px;")
        val.setAlignment(Qt.AlignRight)
        row.addWidget(lbl, 1)
        row.addWidget(val)
        l.addLayout(row)

        bar = QProgressBar()
        bar.setFixedHeight(4)
        bar.setTextVisible(False)
        bar.setRange(0, 100)
        bar.setValue(int(percent))
        bar.setStyleSheet("""
            QProgressBar { background: #222; border: none; border-radius: 2px; } 
            QProgressBar::chunk { background: #00AA00; border-radius: 2px; }
        """)
        l.addWidget(bar)

        layout.addWidget(container)


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

        specs = get_cached_specs()

        self.add_stat_row(layout, "CPU:", specs['cpu'])
        self.add_stat_row(layout, "GPU:", specs['gpu'])

        self.add_stat_row(layout, "RAM:", specs['ram_text'])
        self.ram_bar = self.create_usage_bar(specs['ram_percent'])
        layout.addWidget(self.ram_bar)

        layout.addSpacing(5)
        self.add_stat_row(layout, "VRAM (Est):", "6.0 / 8.0 GB")
        self.vram_bar = self.create_usage_bar(75)
        layout.addWidget(self.vram_bar)

        layout.addSpacing(5)
        self.add_stat_row(layout, "OS:", specs['os'])
        self.add_stat_row(layout, "Version:", "v1.0.0")

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
        bar.setStyleSheet("""
            QProgressBar { border: 1px solid #444; border-radius: 3px; background: #222; } 
            QProgressBar::chunk { background: #00FF00; }
        """)
        return bar


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
