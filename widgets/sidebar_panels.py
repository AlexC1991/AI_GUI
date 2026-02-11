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
            QPushButton {
                background-color: #2E2E2E; border: 1px solid #444; border-radius: 4px; color: white; padding: 5px; min-height: 25px;
            }
            QPushButton:hover { background-color: #3E3E3E; }
            #ChangeModelBtn { background-color: #006666; border: 1px solid #004d4d; }
            #ChangeModelBtn:hover { background-color: #008080; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Chat Section:", styleSheet="font-size: 14px; color: #FFF;"), alignment=Qt.AlignHCenter)

        # --- Current Model Display ---
        layout.addSpacing(5)
        self.current_model_label = QLabel("(None)")
        self.current_model_label.setAlignment(Qt.AlignCenter)
        self.current_model_label.setStyleSheet("color: #00FFFF; font-weight: bold; font-size: 13px;")
        self.current_model_label.setWordWrap(True)
        layout.addWidget(self.current_model_label)

        self.change_model_btn = QPushButton("Change Model")
        self.change_model_btn.setObjectName("ChangeModelBtn")
        self.change_model_btn.setFixedHeight(30)
        self.change_model_btn.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.change_model_btn)

        # --- Force Reasoning ---
        layout.addSpacing(8)
        self.reasoning_mode_chk = QCheckBox("Force Reasoning Mode")
        self.reasoning_mode_chk.setToolTip("Injects system prompt to force <think> tags")
        self.reasoning_mode_chk.setStyleSheet("color: #E0E0E0;")
        layout.addWidget(self.reasoning_mode_chk)

        # --- Session Info Frame ---
        layout.addSpacing(12)
        self.session_frame = QFrame()
        self.session_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 30, 30, 0.6);
                border: 1px solid #333;
                border-radius: 6px;
            }
        """)
        session_layout = QVBoxLayout(self.session_frame)
        session_layout.setContentsMargins(10, 8, 10, 8)
        session_layout.setSpacing(4)

        session_header = QLabel("Session Info:")
        session_header.setStyleSheet("color: #888; font-size: 11px; font-weight: bold; margin-top: 0px; border: none;")
        session_layout.addWidget(session_header)

        # Stat rows (label: value pairs)
        stat_style = "color: #AAAAAA; font-size: 11px; font-weight: normal; margin-top: 0px; border: none;"
        val_style = "color: #FFFFFF; font-size: 11px; font-weight: bold; margin-top: 0px; border: none;"

        # Mode
        self.stat_mode = self._add_stat_row(session_layout, "Mode:", "Local", stat_style, val_style)
        # Engine / Provider
        self.stat_engine = self._add_stat_row(session_layout, "Engine:", "VoxAI (llama.cpp)", stat_style, val_style)
        # Model Size / Quant
        self.stat_quant = self._add_stat_row(session_layout, "Quant:", "—", stat_style, val_style)
        # Context
        self.stat_context = self._add_stat_row(session_layout, "Context:", "4096 tokens", stat_style, val_style)
        # GPU / Tier
        self.stat_gpu = self._add_stat_row(session_layout, "GPU:", "Local (Vulkan)", stat_style, val_style)
        # Cost / Credits
        self.stat_cost = self._add_stat_row(session_layout, "Cost:", "Free", stat_style, val_style)

        # Separator inside session frame
        mini_line = QFrame()
        mini_line.setFrameShape(QFrame.HLine)
        mini_line.setStyleSheet("background-color: #444; margin: 2px 0; border: none;")
        session_layout.addWidget(mini_line)

        # Network status inside session info
        net_layout = QHBoxLayout()
        net_layout.setContentsMargins(0, 2, 0, 0)
        net_layout.setSpacing(6)

        self.network_icon = QLabel("🌐")
        self.network_icon.setStyleSheet("font-size: 14px; margin-top: 0px; border: none;")
        net_layout.addWidget(self.network_icon)

        self.network_label = QLabel("Checking...")
        self.network_label.setStyleSheet("color: #888; font-size: 11px; font-weight: normal; margin-top: 0px; border: none;")
        net_layout.addWidget(self.network_label)
        net_layout.addStretch()

        session_layout.addLayout(net_layout)
        layout.addWidget(self.session_frame)

        # --- Status ---
        layout.addSpacing(8)
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #00FF00; font-weight: bold;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        # --- State ---
        self.execution_mode = "local"     # "local", "cloud", "provider"
        self.local_models_cache = []
        self.selected_model = None

    # --- Helpers ---

    def _add_stat_row(self, parent_layout, label_text, value_text, label_style, value_style):
        """Add a label: value stat row. Returns the value QLabel for later updates."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        lbl = QLabel(label_text)
        lbl.setStyleSheet(label_style)
        lbl.setFixedWidth(55)
        val = QLabel(value_text)
        val.setStyleSheet(value_style)
        val.setWordWrap(True)
        row.addWidget(lbl)
        row.addWidget(val, 1)
        parent_layout.addLayout(row)
        return val

    # --- Execution Mode ---

    def set_execution_mode(self, mode, model_data):
        """Set execution mode and update model display + all session stats.
        mode: 'local', 'cloud', 'provider'
        model_data: dict with mode-specific keys
        """
        self.execution_mode = mode
        self.selected_model = model_data

        # Update model label with mode suffix
        display = ""
        if isinstance(model_data, dict):
            display = model_data.get("display", "Unknown")

        mode_suffix = {"local": "Local", "cloud": "Cloud GPU", "provider": "API"}.get(mode, "")
        self.current_model_label.setText(f"{display}\n({mode_suffix})")

        # Update all stat rows based on mode
        if mode == "local":
            filename = model_data.get("filename", "") if isinstance(model_data, dict) else ""
            size = model_data.get("size", "") if isinstance(model_data, dict) else ""
            # Detect quant from filename
            quant = self._detect_quant(filename)
            self.stat_mode.setText("Local")
            self.stat_mode.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: bold; border: none;")
            self.stat_engine.setText("VoxAI (llama.cpp)")
            self.stat_quant.setText(quant if quant else (size if size else "—"))
            self.stat_context.setText("4096 tokens")
            self.stat_gpu.setText("Local (Vulkan)")
            self.stat_cost.setText("Free")
            self.stat_cost.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: bold; border: none;")

        elif mode == "cloud":
            cloud_id = model_data.get("cloud_id", "") if isinstance(model_data, dict) else ""
            quant = self._detect_quant(cloud_id)
            self.stat_mode.setText("Cloud GPU")
            self.stat_mode.setStyleSheet("color: #bb86fc; font-size: 11px; font-weight: bold; border: none;")
            self.stat_engine.setText("vLLM (RunPod)")
            self.stat_quant.setText(quant if quant else "AWQ/FP16")
            self.stat_context.setText("8192 tokens")
            # Auto-detect GPU tier from model name
            gpu_tier = "A100/H100" if any(k in cloud_id.lower() for k in ["70b", "72b", "120b"]) else "A40/A6000"
            self.stat_gpu.setText(f"RunPod ({gpu_tier})")
            self.stat_cost.setText("Checking...")
            self.stat_cost.setStyleSheet("color: #f0ad4e; font-size: 11px; font-weight: bold; border: none;")

        elif mode == "provider":
            provider_name = model_data.get("provider", "Gemini") if isinstance(model_data, dict) else "Gemini"
            self.stat_mode.setText("API Provider")
            self.stat_mode.setStyleSheet("color: #5bc0de; font-size: 11px; font-weight: bold; border: none;")
            self.stat_engine.setText(f"{provider_name} API")
            self.stat_quant.setText("Native")
            self.stat_context.setText("1M tokens")
            self.stat_gpu.setText("Google Cloud")
            self.stat_cost.setText("$0.00 / session")
            self.stat_cost.setStyleSheet("color: #5bc0de; font-size: 11px; font-weight: bold; border: none;")

    def _detect_quant(self, name):
        """Detect quantization level from filename/model ID."""
        if not name:
            return ""
        name_lower = name.lower()
        for q in ["q2_k", "q3_k_m", "q3_k_s", "q4_0", "q4_k_m", "q4_k_s", "q5_0", "q5_k_m", "q5_k_s",
                   "q6_k", "q8_0", "fp16", "f16", "awq", "gptq", "gguf", "4bit", "8bit"]:
            if q in name_lower:
                return q.upper().replace("_", " ")
        return ""

    def update_session_info(self, info_dict):
        """Dynamically update session info stats. Accepts mode-specific data.
        Local:    {"context_limit": 4096, "speed_tps": 12.5}
        Cloud:    {"credits": 12.47, "cost_per_hr": 0.65, "gpu_type": "A100"}
        Provider: {"api_cost_session": 0.18, "context_used": 4521}
        """
        if "context_limit" in info_dict:
            self.stat_context.setText(f"{info_dict['context_limit']} tokens")
        if "speed_tps" in info_dict:
            self.stat_cost.setText(f"{info_dict['speed_tps']:.1f} t/s")
            self.stat_cost.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: bold; border: none;")
        if "credits" in info_dict:
            cost = info_dict.get("cost_per_hr", 0)
            self.stat_cost.setText(f"${info_dict['credits']:.2f} ({cost:.2f}/hr)")
            self.stat_cost.setStyleSheet("color: #f0ad4e; font-size: 11px; font-weight: bold; border: none;")
        if "gpu_type" in info_dict:
            self.stat_gpu.setText(f"RunPod ({info_dict['gpu_type']})")
        if "api_cost_session" in info_dict:
            self.stat_cost.setText(f"${info_dict['api_cost_session']:.4f}")
            self.stat_cost.setStyleSheet("color: #5bc0de; font-size: 11px; font-weight: bold; border: none;")
        if "context_used" in info_dict:
            self.stat_context.setText(f"{info_dict['context_used']} tokens used")

    # --- Network Status ---

    def set_network_status(self, status: str):
        """Update network status indicator.
        status: 'online', 'offline', 'searching', 'checking'
        """
        if status == "online":
            self.network_icon.setText("🟢")
            self.network_label.setText("Online")
            self.network_label.setStyleSheet("color: #5cb85c; font-size: 11px; font-weight: normal; margin-top: 0px; border: none;")
        elif status == "offline":
            self.network_icon.setText("🔴")
            self.network_label.setText("Offline")
            self.network_label.setStyleSheet("color: #d9534f; font-size: 11px; font-weight: normal; margin-top: 0px; border: none;")
        elif status == "searching":
            self.network_icon.setText("🔍")
            self.network_label.setText("Searching...")
            self.network_label.setStyleSheet("color: #f0ad4e; font-size: 11px; font-weight: normal; margin-top: 0px; border: none;")
        else:  # checking
            self.network_icon.setText("🌐")
            self.network_label.setText("Checking...")
            self.network_label.setStyleSheet("color: #888; font-size: 11px; font-weight: normal; margin-top: 0px; border: none;")

    # --- Model Cache (for local models list) ---

    def set_local_models(self, models):
        """Called by MainWindow to update the list with real installed models"""
        self.local_models_cache = models if models else []

        # Auto-select first model if none selected
        if self.local_models_cache and self.selected_model is None:
            first = self.local_models_cache[0]
            if isinstance(first, dict):
                self.set_execution_mode("local", first)

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
        layout.setContentsMargins(10, 0, 10, 0) # Fix margines
        label = QLabel(f"{mode_name}\nSettings Coming Soon")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #666; font-style: italic; background: transparent;")
        layout.addWidget(label)
