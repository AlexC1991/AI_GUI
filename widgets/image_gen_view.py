from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QComboBox, QPushButton, QFrame, QScrollArea, QSlider,
    QSpinBox, QGridLayout, QGroupBox, QDoubleSpinBox, QTabWidget,
    QProgressBar, QSplitter, QCheckBox, QSizePolicy, QStackedWidget
)
from PySide6.QtCore import Qt, Signal, QUrl, QTimer
from PySide6.QtGui import QPixmap

try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PySide6.QtMultimediaWidgets import QVideoWidget
    MULTIMEDIA_AVAILABLE = True
except ImportError:
    MULTIMEDIA_AVAILABLE = False
from pathlib import Path
from utils.config_manager import ConfigManager
from backend.chat_worker import ChatWorker


SAMPLERS = [
    ("dpm++_2m_karras", "DPM++ 2M Karras"),
    ("dpm++_2m", "DPM++ 2M"),
    ("dpm++_sde_karras", "DPM++ SDE Karras"),
    ("euler", "Euler"),
    ("euler_a", "Euler Ancestral"),
    ("ddim", "DDIM"),
    ("unipc", "UniPC"),
    ("heun", "Heun"),
]

MODEL_PROFILES = {
    "sd15": {"name": "SD 1.5", "native_res": 512, "default_steps": 25, "default_cfg": 7.0, "supports_negative": True, "prompt_hint": "Natural language or tags"},
    "sd20": {"name": "SD 2.x", "native_res": 768, "default_steps": 28, "default_cfg": 7.5, "supports_negative": True, "prompt_hint": "Natural language"},
    "sdxl": {"name": "SDXL", "native_res": 1024, "default_steps": 30, "default_cfg": 5.5, "supports_negative": True, "prompt_hint": "Detailed natural language"},
    "flux": {"name": "Flux", "native_res": 1024, "default_steps": 20, "default_cfg": 3.5, "supports_negative": False, "prompt_hint": "Natural language (no negative)"},
    "pony": {"name": "Pony/Anime", "native_res": 768, "default_steps": 25, "default_cfg": 6.0, "supports_negative": True, "prompt_hint": "Booru tags"},
    "gemini": {"name": "Gemini Cloud", "native_res": 1024, "default_steps": 1, "default_cfg": 1.0, "supports_negative": False, "prompt_hint": "Detailed natural language"},
    "openai": {"name": "OpenAI Cloud", "native_res": 1024, "default_steps": 1, "default_cfg": 1.0, "supports_negative": False, "prompt_hint": "Detailed natural language"},
    "video": {"name": "Video Gen", "native_res": 1280, "default_steps": 1, "default_cfg": 1.0, "supports_negative": False, "prompt_hint": "Describe the scene, camera motion, action"},
    "unknown": {"name": "Unknown", "native_res": 512, "default_steps": 25, "default_cfg": 7.0, "supports_negative": True, "prompt_hint": "Try tags or natural"},
}

# Gemini image model display names
GEMINI_IMAGE_DISPLAY = {
    "gemini-2.5-flash-image": "Gemini Flash Image  [Cloud]",
    "gemini-3-pro-image-preview": "Gemini Pro Image  [Cloud]",
}

# OpenAI image model display names
OPENAI_IMAGE_DISPLAY = {
    "gpt-image-1": "GPT Image 1  [Cloud]",
    "dall-e-3": "DALL\u00B7E 3  [Cloud]",
}

# Video model display names
SORA_VIDEO_DISPLAY = {
    "sora-2": "Sora 2  [Video]",
    "sora-2-pro": "Sora 2 Pro  [Video]",
}

VEO_VIDEO_DISPLAY = {
    "veo-3.1-generate-preview": "Veo 3.1  [Video]",
    "veo-3.1-fast-generate-preview": "Veo 3.1 Fast  [Video]",
}

ALL_VIDEO_DISPLAY = {**SORA_VIDEO_DISPLAY, **VEO_VIDEO_DISPLAY}

def detect_model_family(model_name: str) -> str:
    name_lower = model_name.lower()
    # Flux first
    if 'flux' in name_lower: return "flux"
    # SDXL includes Pony (SDXL-based)
    if any(x in name_lower for x in ['sdxl', 'xl_', '_xl', 'pony', 'illustrious', 'animagine']):
        return "sdxl"
    # Anime SD 1.5 models
    if any(x in name_lower for x in ['waifu', 'nai', 'anything-v', 'counterfeit']):
        return "pony"
    if any(x in name_lower for x in ['sd2', 'v2-', '768']): return "sd20"
    return "sd15"


class AssetCard(QFrame):
    """Base class for stackable cards."""
    def __init__(self, parent_rack):
        super().__init__()
        self.parent_rack = parent_rack
        self.setFixedHeight(50)
        self.setStyleSheet("QFrame{background:#2D2D30;border:1px solid #3E3E42;border-radius:6px}")
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(8,4,8,4)
    
    def add_delete_btn(self):
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(20,20)
        del_btn.setStyleSheet("background:transparent;color:#A55;border:none")
        del_btn.clicked.connect(lambda: self.parent_rack.remove_card(self))
        self.layout.addWidget(del_btn)


class LoraCard(AssetCard):
    def __init__(self, parent_rack):
        super().__init__(parent_rack)
        
        self.name_combo = QComboBox()
        self.name_combo.setStyleSheet("background:#222;border:1px solid #444;min-width:120px")
        if hasattr(parent_rack, 'available_loras'):
            self.name_combo.addItems(parent_rack.available_loras)
        self.layout.addWidget(self.name_combo, 2)
        
        self.layout.addWidget(QLabel("Str:"))
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 200)
        self.strength_slider.setValue(100)
        self.strength_slider.setFixedWidth(80)
        self.val_label = QLabel("1.00")
        self.strength_slider.valueChanged.connect(lambda v: self.val_label.setText(f"{v/100:.2f}"))
        self.layout.addWidget(self.strength_slider)
        self.layout.addWidget(self.val_label)
        
        self.add_delete_btn()
    
    def get_lora_name(self): return self.name_combo.currentText()
    def get_weight(self): return self.strength_slider.value() / 100.0


class TextEncoderCard(AssetCard):
    def __init__(self, parent_rack):
        super().__init__(parent_rack)
        self.name_combo = QComboBox()
        self.name_combo.setStyleSheet("background:#222;border:1px solid #444;min-width:140px")
        if hasattr(parent_rack, 'available_text_encoders'):
            self.name_combo.addItems(parent_rack.available_text_encoders)
        self.layout.addWidget(self.name_combo, 1)
        
        self.add_delete_btn()

    def get_encoder_name(self): return self.name_combo.currentText()


class VaeCard(AssetCard):
    def __init__(self, parent_rack):
        super().__init__(parent_rack)
        self.setFixedHeight(40) # Slightly smaller
        self.name_combo = QComboBox()
        self.name_combo.setStyleSheet("background:#222;border:1px solid #444;min-width:140px")
        if hasattr(parent_rack, 'available_vaes'):
            self.name_combo.addItems(parent_rack.available_vaes)
        self.layout.addWidget(self.name_combo, 1)
        
        self.add_delete_btn()

    def get_vae_name(self): return self.name_combo.currentText()


class ImageGenView(QWidget):
    RESOLUTIONS = [
        ("512x512 (SD1.5)", 512, 512), ("512x768", 512, 768), ("768x512", 768, 512),
        ("768x768", 768, 768), ("768x1024", 768, 1024), ("1024x768", 1024, 768),
        ("1024x1024 (SDXL)", 1024, 1024), ("1152x896", 1152, 896), ("896x1152", 896, 1152),
        ("1344x768 (16:9)", 1344, 768), ("768x1344 (9:16)", 768, 1344),
    ]
    
    def __init__(self):
        super().__init__()
        self.current_family = "sd15"
        self.available_loras = []
        self.available_vaes = []
        self.available_text_encoders = []
        
        self.main_splitter = QSplitter(Qt.Horizontal)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.main_splitter)
        
        # Left
        left = QWidget()
        self.left_layout = QVBoxLayout(left)
        self.left_layout.setContentsMargins(15,15,10,15)
        self.left_layout.setSpacing(10)
        self._create_model_section()
        self._create_prompt_section()
        self._create_stacks_section() 
        self.main_splitter.addWidget(left)
        
        # Right
        right = QWidget()
        self.right_layout = QVBoxLayout(right)
        self.right_layout.setContentsMargins(10,15,15,15)
        self.right_layout.setSpacing(10)
        self._create_settings_section()
        self._create_output_section()
        self.main_splitter.addWidget(right)
        
        self.main_splitter.setSizes([500,600])
        self.refresh_assets()
    
    def _create_model_section(self):
        grp = QGroupBox("Model")
        grp.setStyleSheet("QGroupBox{border:1px solid #444;border-radius:6px;font-weight:bold;color:#088;margin-top:6px}QGroupBox::title{subcontrol-origin:margin;left:10px}")
        lay = QVBoxLayout(grp)
        
        row = QHBoxLayout()
        self.ckpt_combo = QComboBox()
        self.ckpt_combo.setStyleSheet("padding:5px;background:#252526;color:white;border:1px solid #444")
        self.ckpt_combo.currentTextChanged.connect(self._on_ckpt_changed)
        row.addWidget(self.ckpt_combo, 1)
        
        ref_btn = QPushButton("🔄")
        ref_btn.setFixedSize(28,28)
        ref_btn.clicked.connect(self.refresh_assets)
        row.addWidget(ref_btn)
        lay.addLayout(row)
        
        info = QHBoxLayout()
        self.family_label = QLabel("Family: SD 1.5")
        self.family_label.setStyleSheet("color:#888;font-size:11px")
        self.hint_label = QLabel("")
        self.hint_label.setStyleSheet("color:#666;font-size:10px;font-style:italic")
        info.addWidget(self.family_label)
        info.addStretch()
        info.addWidget(self.hint_label)
        lay.addLayout(info)
        
        self.left_layout.addWidget(grp)
    
    def _create_prompt_section(self):
        # Header with Enhance Button
        header = QHBoxLayout()
        header.addWidget(QLabel("Positive Prompt:", styleSheet="color:#AAA;font-weight:bold"))
        
        self.enhance_btn = QPushButton("✨ Enhance (LLM)")
        self.enhance_btn.setFixedSize(110, 24)
        self.enhance_btn.setCursor(Qt.PointingHandCursor)
        self.enhance_btn.setStyleSheet("""
            QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6a11cb, stop:1 #2575fc); color: white; border: none; border-radius: 12px; font-weight: bold; font-size: 11px; }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7a21db, stop:1 #3585ff); }
        """)
        self.enhance_btn.clicked.connect(self.enhance_prompt)
        header.addWidget(self.enhance_btn)
        
        self.left_layout.addLayout(header)
        
        self.pos_prompt = QTextEdit()
        self.pos_prompt.setPlaceholderText("Describe what you want...")
        self.pos_prompt.setStyleSheet("background:#252526;color:#E0E0E0;border:1px solid #444")
        self.pos_prompt.setFixedHeight(80) # Fixed smaller height
        self.left_layout.addWidget(self.pos_prompt)
        
        self.neg_label = QLabel("Negative Prompt:", styleSheet="color:#AAA;font-weight:bold")
        self.left_layout.addWidget(self.neg_label)
        self.neg_prompt = QTextEdit()
        self.neg_prompt.setText("blurry, bad quality, distorted, ugly, deformed, bad anatomy, watermark")
        self.neg_prompt.setFixedHeight(45)
        self.neg_prompt.setStyleSheet("background:#252526;color:#E0E0E0;border:1px solid #444")
        self.left_layout.addWidget(self.neg_prompt)
    
    def _create_stacks_section(self):
        # 1. Text Encoder Stack
        self._create_generic_stack("Text Encoder Stack", "text_encoder", self.add_text_encoder_card, 100)
        
        # 2. VAE Stack
        self._create_generic_stack("VAE Stack", "vae", self.add_vae_card, 80)

        # 3. LoRA Stack (Expanded)
        self._create_generic_stack("LoRA Stack", "lora", self.add_lora_card, 200)

    def _create_generic_stack(self, title, id_prefix, add_func, max_height):
        grp = QGroupBox(title)
        grp.setStyleSheet("QGroupBox{border:1px solid #444;border-radius:6px;font-weight:bold;color:#088;margin-top:6px}QGroupBox::title{subcontrol-origin:margin;left:10px}")
        lay = QVBoxLayout(grp)
        lay.setContentsMargins(5,12,5,5)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background:transparent;border:none")
        scroll.setMinimumHeight(60) # Ensure it has some presence
        # Remove Max Height to let it expand if needed, or set large
        if max_height:
            scroll.setMaximumHeight(max_height)
        
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setAlignment(Qt.AlignTop)
        container_layout.setSpacing(4)
        setattr(self, f"{id_prefix}_layout", container_layout) # e.g. self.lora_layout
        
        scroll.setWidget(container)
        lay.addWidget(scroll)
        
        add_btn = QPushButton(f"+ Add {title.split(' ')[0]}")
        add_btn.setStyleSheet("background:#333;color:white;border:1px dashed #555;padding:5px")
        add_btn.clicked.connect(add_func)
        lay.addWidget(add_btn)
        
        self.left_layout.addWidget(grp)

    def _create_settings_section(self):
        frame = QFrame()
        frame.setStyleSheet("background-color:#1E1E1E;border-radius:6px")
        grid = QGridLayout(frame)
        grid.setContentsMargins(10,10,10,10)
        grid.setVerticalSpacing(8)
        
        # Resolution
        grid.addWidget(QLabel("Resolution:", styleSheet="color:#CCC"), 0, 0)
        self.res_combo = QComboBox()
        for name, w, h in self.RESOLUTIONS:
            self.res_combo.addItem(name, (w, h))
        self.res_combo.setStyleSheet("background:#252526;color:white;border:1px solid #444")
        grid.addWidget(self.res_combo, 0, 1)
        
        # Sampler
        grid.addWidget(QLabel("Sampler:", styleSheet="color:#CCC"), 0, 2)
        self.sampler_combo = QComboBox()
        for key, name in SAMPLERS:
            self.sampler_combo.addItem(name, key)
        self.sampler_combo.setStyleSheet("background:#252526;color:white;border:1px solid #444")
        grid.addWidget(self.sampler_combo, 0, 3)
        
        # Steps
        grid.addWidget(QLabel("Steps:", styleSheet="color:#CCC"), 1, 0)
        self.steps = QSpinBox()
        self.steps.setRange(1, 100)
        self.steps.setValue(25)
        self.steps.setStyleSheet("background:#252526;color:white;border:1px solid #444")
        grid.addWidget(self.steps, 1, 1)
        
        # CFG
        grid.addWidget(QLabel("CFG:", styleSheet="color:#CCC"), 1, 2)
        self.cfg = QDoubleSpinBox()
        self.cfg.setRange(1.0, 30.0)
        self.cfg.setValue(7.0)
        self.cfg.setSingleStep(0.5)
        self.cfg.setStyleSheet("background:#252526;color:white;border:1px solid #444")
        grid.addWidget(self.cfg, 1, 3)
        
        # Seed
        grid.addWidget(QLabel("Seed:", styleSheet="color:#CCC"), 2, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2147483647)
        self.seed_spin.setValue(-1)
        self.seed_spin.setSpecialValueText("Random")
        self.seed_spin.setStyleSheet("background:#252526;color:white;border:1px solid #444")
        grid.addWidget(self.seed_spin, 2, 1)
        
        # Batch
        grid.addWidget(QLabel("Batch:", styleSheet="color:#CCC"), 2, 2)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 4)
        self.batch_spin.setStyleSheet("background:#252526;color:white;border:1px solid #444")
        self.batch_spin.valueChanged.connect(self.update_tab_count)
        grid.addWidget(self.batch_spin, 2, 3)
        
        # Options
        self.keep_model_check = QCheckBox("Keep model loaded")
        self.keep_model_check.setChecked(True)
        self.keep_model_check.setStyleSheet("color:#AAA")
        grid.addWidget(self.keep_model_check, 3, 0, 1, 2)
        
        self.auto_cfg_check = QCheckBox("Auto-adjust settings")
        self.auto_cfg_check.setChecked(True)
        self.auto_cfg_check.setStyleSheet("color:#AAA")
        grid.addWidget(self.auto_cfg_check, 3, 2, 1, 2)

        # --- Video-specific controls (hidden by default) ---
        self.video_duration_label = QLabel("Duration:", styleSheet="color:#0CC")
        grid.addWidget(self.video_duration_label, 4, 0)
        self.video_duration = QComboBox()
        self.video_duration.addItems(["4s", "8s", "12s"])
        self.video_duration.setCurrentText("8s")
        self.video_duration.setStyleSheet("background:#252526;color:white;border:1px solid #444")
        grid.addWidget(self.video_duration, 4, 1)

        self.video_aspect_label = QLabel("Aspect:", styleSheet="color:#0CC")
        grid.addWidget(self.video_aspect_label, 4, 2)
        self.video_aspect = QComboBox()
        self.video_aspect.addItems(["16:9  (Landscape)", "9:16  (Portrait)"])
        self.video_aspect.setStyleSheet("background:#252526;color:white;border:1px solid #444")
        grid.addWidget(self.video_aspect, 4, 3)

        # Start hidden
        self._toggle_video_controls(False)

        self.right_layout.addWidget(frame)
    
    def _create_output_section(self):
        # --- Live Stats Bar ---
        stats_frame = QFrame()
        stats_frame.setStyleSheet("background:#1A1A1E;border:1px solid #333;border-radius:8px;")
        stats_lay = QHBoxLayout(stats_frame)
        stats_lay.setContentsMargins(12, 8, 12, 8)
        stats_lay.setSpacing(16)

        # Pulsing orb (animated during generation)
        self._stats_orb = QLabel("●")
        self._stats_orb.setFixedSize(18, 18)
        self._stats_orb.setAlignment(Qt.AlignCenter)
        self._stats_orb.setStyleSheet("color:#555;font-size:12px;background:transparent;border:none;")
        stats_lay.addWidget(self._stats_orb)

        # Status text
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color:#888;font-size:12px;font-weight:500;background:transparent;border:none;")
        stats_lay.addWidget(self.status_label)

        stats_lay.addStretch()

        # Elapsed timer
        self._elapsed_label = QLabel("")
        self._elapsed_label.setStyleSheet("color:#0AA;font-size:11px;font-family:'JetBrains Mono','Consolas',monospace;background:transparent;border:none;")
        stats_lay.addWidget(self._elapsed_label)

        # Model badge
        self._model_badge = QLabel("")
        self._model_badge.setStyleSheet("color:#666;font-size:10px;background:transparent;border:none;")
        stats_lay.addWidget(self._model_badge)

        self.right_layout.addWidget(stats_frame)

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m")
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setStyleSheet(
            "QProgressBar{border:1px solid #333;background:#222;border-radius:4px;text-align:center;color:white}"
            "QProgressBar::chunk{background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #066,stop:0.5 #0AA,stop:1 #066);border-radius:3px}"
        )
        self.right_layout.addWidget(self.progress_bar)

        # --- Output Tabs ---
        self.output_tabs = QTabWidget()
        self.output_tabs.setStyleSheet("QTabWidget::pane{border:1px solid #333;background:#111}QTabBar::tab{background:#222;color:#888;padding:6px 12px}QTabBar::tab:selected{background:#333;color:white;border-top:2px solid #088}")
        self.update_tab_count(1)
        self.right_layout.addWidget(self.output_tabs, 1)

        # --- Live Timer State ---
        self._gen_start_time = None
        self._orb_phase = 0
        self._orb_colors = ["#088", "#0AA", "#0CC", "#0EE", "#0CC", "#0AA"]

        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._tick_live_stats)
        # Timer runs at 200ms for smooth orb animation
    
    def _on_ckpt_changed(self, name):
        if not name or "No checkpoint" in name: return

        # Check model type
        is_cloud = self._is_cloud_model(name)
        is_video = self._is_video_model(name)

        if is_video:
            family = "video"
            profile = MODEL_PROFILES["video"]
        elif self._is_gemini_model(name):
            family = "gemini"
            profile = MODEL_PROFILES["gemini"]
        elif self._is_openai_model(name):
            family = "openai"
            profile = MODEL_PROFILES["openai"]
        else:
            family = detect_model_family(name)
            profile = MODEL_PROFILES.get(family, MODEL_PROFILES["unknown"])

        self.current_family = family
        self.family_label.setText(f"Family: {profile['name']}")
        self.hint_label.setText(profile['prompt_hint'])

        self.neg_label.setVisible(profile['supports_negative'])
        self.neg_prompt.setVisible(profile['supports_negative'])

        # Hide local-only controls for cloud/video models
        self._toggle_local_controls(not (is_cloud or is_video))

        # Show/hide video-specific controls
        self._toggle_video_controls(is_video)

        if self.auto_cfg_check.isChecked():
            self.cfg.setValue(profile['default_cfg'])
            self.steps.setValue(profile['default_steps'])

    def _is_gemini_model(self, name: str) -> bool:
        """Check if a display name maps to a Gemini image model."""
        return name in GEMINI_IMAGE_DISPLAY.values() or name.startswith("gemini-")

    def _is_openai_model(self, name: str) -> bool:
        """Check if a display name maps to an OpenAI image model."""
        return name in OPENAI_IMAGE_DISPLAY.values() or name.startswith("dall-e") or name.startswith("gpt-image")

    def _is_cloud_model(self, name: str) -> bool:
        """Check if a display name maps to any cloud image model."""
        return self._is_gemini_model(name) or self._is_openai_model(name)

    def _is_video_model(self, name: str) -> bool:
        """Check if a display name maps to a video generation model."""
        return (name in ALL_VIDEO_DISPLAY.values()
                or name.startswith("sora-") or name.startswith("veo-"))

    def get_gemini_model_id(self) -> str:
        """Get the Gemini API model ID from the display name."""
        display = self.ckpt_combo.currentText()
        for model_id, disp_name in GEMINI_IMAGE_DISPLAY.items():
            if disp_name == display:
                return model_id
        return display

    def get_openai_model_id(self) -> str:
        """Get the OpenAI API model ID from the display name."""
        display = self.ckpt_combo.currentText()
        for model_id, disp_name in OPENAI_IMAGE_DISPLAY.items():
            if disp_name == display:
                return model_id
        return display

    def _toggle_local_controls(self, visible: bool):
        """Show/hide controls that only apply to local models."""
        for widget in [self.res_combo, self.sampler_combo, self.steps,
                       self.cfg, self.seed_spin, self.batch_spin,
                       self.keep_model_check, self.auto_cfg_check]:
            widget.setEnabled(visible)

    def _toggle_video_controls(self, visible: bool):
        """Show/hide video-specific controls (duration, aspect ratio)."""
        for widget in [self.video_duration_label, self.video_duration,
                       self.video_aspect_label, self.video_aspect]:
            widget.setVisible(visible)
    
    def refresh_assets(self):
        config = ConfigManager.load_config()
        img_cfg = config.get("image", {})
        llm_cfg = config.get("llm", {})
        ckpt_dir = Path(img_cfg.get("checkpoint_dir", "models/checkpoints"))
        lora_dir = Path(img_cfg.get("lora_dir", "models/loras"))
        vae_dir = Path(img_cfg.get("vae_dir", "models/vae"))
        # Using configured dir or default
        text_enc_dir = Path(img_cfg.get("text_encoder_dir", "models/text_encoders"))

        # Checkpoints (local)
        ckpts = []
        if ckpt_dir.exists():
            for ext in ["*.safetensors", "*.ckpt", "*.gguf"]:
                ckpts.extend([f.name for f in ckpt_dir.glob(ext)])
        ckpts.sort()

        # Cloud image models (Gemini + OpenAI)
        cloud_display_names = []

        # Gemini
        gemini_api_key = ConfigManager.get_provider_key("gemini")
        if gemini_api_key:
            try:
                from providers.gemini_provider import GeminiProvider
                for mid in GeminiProvider.list_image_models():
                    display = GEMINI_IMAGE_DISPLAY.get(mid, f"{mid}  [Cloud]")
                    cloud_display_names.append(display)
            except Exception as e:
                print(f"[ImageGen] Could not load Gemini image models: {e}")

        # OpenAI (DALL-E)
        openai_api_key = ConfigManager.get_provider_key("openai")
        if openai_api_key:
            try:
                from providers.openai_provider import OpenAIProvider
                for mid in OpenAIProvider.list_image_models():
                    display = OPENAI_IMAGE_DISPLAY.get(mid, f"{mid}  [Cloud]")
                    cloud_display_names.append(display)
            except Exception as e:
                print(f"[ImageGen] Could not load OpenAI image models: {e}")

        # --- Video generation models ---
        video_display_names = []

        # Veo (Gemini key)
        if gemini_api_key:
            try:
                from providers.gemini_provider import GeminiProvider
                for mid in GeminiProvider.list_video_models():
                    display = VEO_VIDEO_DISPLAY.get(mid, f"{mid}  [Video]")
                    video_display_names.append(display)
            except Exception as e:
                print(f"[ImageGen] Could not load Veo video models: {e}")

        # Sora (OpenAI key)
        if openai_api_key:
            try:
                from providers.openai_provider import OpenAIProvider
                for mid in OpenAIProvider.list_video_models():
                    display = SORA_VIDEO_DISPLAY.get(mid, f"{mid}  [Video]")
                    video_display_names.append(display)
            except Exception as e:
                print(f"[ImageGen] Could not load Sora video models: {e}")

        all_models = ckpts + cloud_display_names + video_display_names

        cur = self.ckpt_combo.currentText()
        self.ckpt_combo.blockSignals(True)
        self.ckpt_combo.clear()
        self.ckpt_combo.addItems(all_models if all_models else ["No checkpoints found"])
        if cur in all_models: self.ckpt_combo.setCurrentText(cur)
        self.ckpt_combo.blockSignals(False)
        if all_models: self._on_ckpt_changed(self.ckpt_combo.currentText())
        
        # LoRAs
        self.available_loras = []
        if lora_dir.exists():
            for ext in ["*.safetensors", "*.ckpt"]:
                self.available_loras.extend([f.name for f in lora_dir.glob(ext)])
        self.available_loras.sort()

        # VAEs
        self.available_vaes = []
        if vae_dir.exists():
            for ext in ["*.safetensors", "*.sft", "*.ckpt", "*.pt"]:
                self.available_vaes.extend([f.name for f in vae_dir.glob(ext)])
        self.available_vaes.sort()

        # Text Encoders
        self.available_text_encoders = []
        if text_enc_dir.exists():
            for ext in ["*.safetensors", "*.ckpt", "*.bin", "*.gguf"]: # .bin for some pytorch dumps
                self.available_text_encoders.extend([f.name for f in text_enc_dir.glob(ext)])
        self.available_text_encoders.sort()
    
    def add_lora_card(self):
        if not self.available_loras:
            self.set_status("⚠️ No LoRAs found", "#f59e0b")
            return
        self.lora_layout.addWidget(LoraCard(self))

    def add_vae_card(self):
        if not self.available_vaes:
            self.set_status("⚠️ No VAEs found", "#f59e0b")
            return
        # If single VAE logic is preferred, we could clear previous
        # self.clear_layout(self.vae_layout)
        self.vae_layout.addWidget(VaeCard(self))

    def add_text_encoder_card(self):
        if not self.available_text_encoders:
            self.set_status("⚠️ No Text Encoders found", "#f59e0b")
            return
        self.text_encoder_layout.addWidget(TextEncoderCard(self))
    
    def remove_card(self, card):
        if isinstance(card, LoraCard):
            self.lora_layout.removeWidget(card)
        elif isinstance(card, VaeCard):
            self.vae_layout.removeWidget(card)
        elif isinstance(card, TextEncoderCard):
            self.text_encoder_layout.removeWidget(card)
        card.deleteLater()
    
    def update_tab_count(self, count):
        self.output_tabs.clear()
        for i in range(count):
            tab = QWidget()
            lay = QVBoxLayout(tab)

            # Stacked widget: image label (idx 0) / video player (idx 1)
            stack = QStackedWidget()
            stack.setObjectName("output_stack")

            # Image preview label
            lbl = QLabel(f"Image {i+1}\n(Waiting...)")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color:#555;font-style:italic")
            lbl.setMinimumSize(350, 350)
            lbl.setObjectName("preview_label")
            stack.addWidget(lbl)

            # Video player widget
            video_container = QWidget()
            video_lay = QVBoxLayout(video_container)
            video_lay.setContentsMargins(0, 0, 0, 0)
            video_lay.setSpacing(4)

            if MULTIMEDIA_AVAILABLE:
                video_widget = QVideoWidget()
                video_widget.setMinimumSize(350, 250)
                video_widget.setStyleSheet("background: black;")
                video_widget.setObjectName("video_widget")
                video_lay.addWidget(video_widget, 1)

                # Transport controls
                transport = QHBoxLayout()
                play_btn = QPushButton("▶ Play")
                play_btn.setObjectName("video_play_btn")
                play_btn.setCursor(Qt.PointingHandCursor)
                play_btn.setStyleSheet(
                    "QPushButton{background:#088;color:white;border:none;padding:6px 16px;border-radius:4px;font-weight:bold}"
                    "QPushButton:hover{background:#0AA}"
                    "QPushButton:pressed{background:#066}"
                )
                pause_btn = QPushButton("⏸ Pause")
                pause_btn.setObjectName("video_pause_btn")
                pause_btn.setCursor(Qt.PointingHandCursor)
                pause_btn.setStyleSheet(
                    "QPushButton{background:#444;color:white;border:none;padding:6px 16px;border-radius:4px}"
                    "QPushButton:hover{background:#555}"
                    "QPushButton:pressed{background:#333}"
                )
                stop_btn = QPushButton("⏹ Stop")
                stop_btn.setObjectName("video_stop_btn")
                stop_btn.setCursor(Qt.PointingHandCursor)
                stop_btn.setStyleSheet(
                    "QPushButton{background:#444;color:white;border:none;padding:6px 16px;border-radius:4px}"
                    "QPushButton:hover{background:#555}"
                    "QPushButton:pressed{background:#333}"
                )
                transport.addStretch()
                transport.addWidget(play_btn)
                transport.addWidget(pause_btn)
                transport.addWidget(stop_btn)
                transport.addStretch()
                video_lay.addLayout(transport)
            else:
                no_video = QLabel("Video playback unavailable\n(PySide6 multimedia not installed)")
                no_video.setAlignment(Qt.AlignCenter)
                no_video.setStyleSheet("color:#888;font-style:italic")
                video_lay.addWidget(no_video)

            stack.addWidget(video_container)
            stack.setCurrentIndex(0)  # Default: show image label

            lay.addWidget(stack)
            self.output_tabs.addTab(tab, f"Output {i+1}")
    
    # === API ===
    def is_gemini_selected(self) -> bool:
        """Check if the currently selected model is a Gemini cloud model."""
        return self._is_gemini_model(self.ckpt_combo.currentText())

    def is_openai_selected(self) -> bool:
        """Check if the currently selected model is an OpenAI cloud model."""
        return self._is_openai_model(self.ckpt_combo.currentText())

    def is_video_selected(self) -> bool:
        """Check if the currently selected model is a video generation model."""
        return self._is_video_model(self.ckpt_combo.currentText())

    def get_video_model_id(self) -> str:
        """Get the actual API model ID from a video display name."""
        display = self.ckpt_combo.currentText()
        for model_id, disp_name in ALL_VIDEO_DISPLAY.items():
            if disp_name == display:
                return model_id
        return display

    def get_video_duration(self) -> int:
        """Get selected video duration in seconds."""
        text = self.video_duration.currentText()
        return int(text.replace("s", ""))

    def get_video_aspect(self) -> str:
        """Get selected video aspect ratio."""
        text = self.video_aspect.currentText()
        if "9:16" in text:
            return "9:16"
        return "16:9"

    def get_selected_checkpoint(self):
        """Return selected model. For cloud/video models, returns the API model ID."""
        text = self.ckpt_combo.currentText()
        if self._is_video_model(text):
            return self.get_video_model_id()
        if self._is_gemini_model(text):
            return self.get_gemini_model_id()
        if self._is_openai_model(text):
            return self.get_openai_model_id()
        return text
    def get_resolution(self): return self.res_combo.currentData() or (512, 512)
    def get_steps(self): return self.steps.value()
    def get_cfg_scale(self): return self.cfg.value()
    def get_seed(self): 
        v = self.seed_spin.value()
        return None if v == -1 else v
    def get_sampler(self): return self.sampler_combo.currentData() or "dpm++_2m_karras"
    def get_positive_prompt(self): return self.pos_prompt.toPlainText().strip()
    def get_negative_prompt(self): return self.neg_prompt.toPlainText().strip() if self.neg_prompt.isVisible() else ""
    
    def get_active_loras(self):
        loras = []
        for i in range(self.lora_layout.count()):
            w = self.lora_layout.itemAt(i).widget()
            if isinstance(w, LoraCard):
                loras.append((w.get_lora_name(), w.get_weight()))
        return loras

    def get_active_vae(self):
        # Return the last added VAE, or None
        if self.vae_layout.count() > 0:
            w = self.vae_layout.itemAt(self.vae_layout.count() - 1).widget()
            if isinstance(w, VaeCard):
                return w.get_vae_name()
        return None

    def get_active_text_encoders(self):
        encs = []
        for i in range(self.text_encoder_layout.count()):
            w = self.text_encoder_layout.itemAt(i).widget()
            if isinstance(w, TextEncoderCard):
                encs.append(w.get_encoder_name())
        return encs

    def should_keep_model_loaded(self): return self.keep_model_check.isChecked()
    def get_model_family(self): return self.current_family
    
    def set_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
    
    def set_status(self, text, color="#888"):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color:{color};font-size:12px;font-weight:500;background:transparent;border:none;")

    # --- Live Stats Animation ---
    def start_live_stats(self):
        """Call when generation starts. Starts elapsed timer and orb animation."""
        import time
        self._gen_start_time = time.time()
        self._orb_phase = 0
        # Show model name in badge
        model = self.ckpt_combo.currentText()
        if len(model) > 30:
            model = model[:27] + "..."
        self._model_badge.setText(f"⚡ {model}")
        self._model_badge.setStyleSheet("color:#888;font-size:10px;background:transparent;border:none;")
        self._elapsed_label.setText("0.0s")
        self._live_timer.start(200)

    def stop_live_stats(self):
        """Call when generation finishes. Stops timer and shows final time."""
        self._live_timer.stop()
        # Show final elapsed
        if self._gen_start_time:
            import time
            elapsed = time.time() - self._gen_start_time
            self._elapsed_label.setText(f"{elapsed:.1f}s")
            self._elapsed_label.setStyleSheet("color:#4ade80;font-size:11px;font-family:'JetBrains Mono','Consolas',monospace;background:transparent;border:none;")
        # Set orb to green (done)
        self._stats_orb.setStyleSheet("color:#4ade80;font-size:12px;background:transparent;border:none;")
        self._gen_start_time = None

    def _tick_live_stats(self):
        """Called every 200ms during generation for live updates."""
        import time
        if self._gen_start_time:
            elapsed = time.time() - self._gen_start_time
            self._elapsed_label.setText(f"{elapsed:.1f}s")

        # Animate orb color cycle
        self._orb_phase = (self._orb_phase + 1) % len(self._orb_colors)
        color = self._orb_colors[self._orb_phase]
        # Pulse size
        sizes = [10, 11, 12, 13, 14, 13]
        size = sizes[self._orb_phase % len(sizes)]
        self._stats_orb.setStyleSheet(f"color:{color};font-size:{size}px;background:transparent;border:none;")

    def reset_live_stats(self):
        """Reset stats to idle state."""
        self._live_timer.stop()
        self._gen_start_time = None
        self._elapsed_label.setText("")
        self._model_badge.setText("")
        self._stats_orb.setStyleSheet("color:#555;font-size:12px;background:transparent;border:none;")
    
    def show_output(self, path, tab_index=0):
        """Auto-detect image vs video and show in the correct widget."""
        p = Path(path)
        if p.suffix.lower() in (".mp4", ".webm", ".mov", ".avi"):
            self.show_video(path, tab_index)
        else:
            self.show_image(path, tab_index)

    def show_image(self, path, tab_index=0):
        if tab_index < self.output_tabs.count():
            tab = self.output_tabs.widget(tab_index)
            stack = tab.findChild(QStackedWidget, "output_stack")
            if stack:
                stack.setCurrentIndex(0)  # Switch to image view
            lbl = tab.findChild(QLabel, "preview_label")
            if lbl and Path(path).exists():
                pix = QPixmap(str(path))
                if not pix.isNull():
                    scaled = pix.scaled(
                        lbl.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    lbl.setPixmap(scaled)
                    lbl.setStyleSheet("")

    def show_video(self, path, tab_index=0):
        """Load a video file into the video player widget."""
        if not MULTIMEDIA_AVAILABLE:
            self.set_status("Video playback unavailable — install PySide6-Multimedia", "#f59e0b")
            return

        if tab_index >= self.output_tabs.count():
            return

        tab = self.output_tabs.widget(tab_index)
        stack = tab.findChild(QStackedWidget, "output_stack")
        video_widget = tab.findChild(QVideoWidget, "video_widget")
        if not video_widget:
            self.set_status("Video widget not found", "#f59e0b")
            return

        # Switch to video view FIRST so the widget is visible before playback
        if stack:
            stack.setCurrentIndex(1)

        # Force the video widget to be shown and sized properly
        video_widget.show()
        video_widget.update()

        # Create or retrieve player — store as Python attributes to prevent GC
        if not hasattr(self, '_video_players'):
            self._video_players = {}

        if tab_index not in self._video_players:
            player = QMediaPlayer(self)  # Parent to self for lifecycle
            audio = QAudioOutput(self)
            player.setAudioOutput(audio)
            player.setVideoOutput(video_widget)

            # Debug: connect error signal for troubleshooting
            player.errorOccurred.connect(
                lambda err, msg: print(f"[VideoPlayer] Error: {err} — {msg}")
            )
            player.mediaStatusChanged.connect(
                lambda status: print(f"[VideoPlayer] Status: {status}")
            )

            self._video_players[tab_index] = {"player": player, "audio": audio}

            # Wire transport buttons with active state feedback
            play_btn = tab.findChild(QPushButton, "video_play_btn")
            pause_btn = tab.findChild(QPushButton, "video_pause_btn")
            stop_btn = tab.findChild(QPushButton, "video_stop_btn")

            def _update_transport(state, pb=play_btn, pa=pause_btn, sb=stop_btn):
                """Highlight the active transport button based on playback state."""
                ACTIVE = (
                    "QPushButton{background:#088;color:white;border:none;padding:6px 16px;border-radius:4px;font-weight:bold}"
                    "QPushButton:hover{background:#0AA}"
                    "QPushButton:pressed{background:#066}"
                )
                IDLE = (
                    "QPushButton{background:#444;color:white;border:none;padding:6px 16px;border-radius:4px}"
                    "QPushButton:hover{background:#555}"
                    "QPushButton:pressed{background:#333}"
                )
                # QMediaPlayer.PlaybackState: 0=Stopped, 1=Playing, 2=Paused
                if pb: pb.setStyleSheet(ACTIVE if state == 1 else IDLE)
                if pa: pa.setStyleSheet(ACTIVE if state == 2 else IDLE)
                if sb: sb.setStyleSheet(ACTIVE if state == 0 else IDLE)

            player.playbackStateChanged.connect(_update_transport)

            if play_btn:
                play_btn.clicked.connect(player.play)
            if pause_btn:
                pause_btn.clicked.connect(player.pause)
            if stop_btn:
                stop_btn.clicked.connect(player.stop)

        player = self._video_players[tab_index]["player"]

        # Resolve absolute path for Windows
        abs_path = str(Path(path).resolve())
        print(f"[VideoPlayer] Loading: {abs_path}")

        # Set source and defer play slightly so widget has time to render
        player.setSource(QUrl.fromLocalFile(abs_path))
        QTimer.singleShot(300, player.play)

    # --- LLM ENHANCEMENT LOGIC ---
    def enhance_prompt(self):
        original = self.pos_prompt.toPlainText().strip()
        if not original:
            self.set_status("⚠️ Enter a prompt first", "#f59e0b")
            return

        self.set_status("✨ Asking AI to enhance...", "#2575fc")
        self.enhance_btn.setEnabled(False)
        self.pos_prompt.setDisabled(True)
        
        # Prepare params
        config = ConfigManager.load_config()
        llm_cfg = config.get("llm", {})
        
        # 1. Get Configured Model/Provider
        # Default to a safe local model if nothing configured
        model_name = llm_cfg.get("model", "llama3.2:latest") 
        provider = llm_cfg.get("provider", "Ollama")
        
        # 2. Logic: Force correct provider based on model name
        # If the model name looks like a local model, force VoxAI
        if any(x in model_name.lower() for x in ["llama", "mistral", "gemma", "qwen", "phi", "vicuna"]):
            provider = "VoxAI"
        elif "gemini" in model_name.lower():
            provider = "Gemini"
        else:
            # Default fallback
            provider = "VoxAI"
            
        # 3. Model ID Mapping (Simple version)
        # For VoxAI, we pass the raw filename or name. For Gemini, we map to API ID.
        model_map = {
            "Gemini Pro": "gemini-1.5-pro",
            "GPT-4o": "gpt-4o",
        }
        api_model = model_map.get(model_name, model_name)
        
        print(f"[Enhancer] Using Provider: {provider}, Model: {api_model}")
        
        # Create Worker
        self.enhancer = ChatWorker()
        
        system_prompt = (
            "You are an expert Stable Diffusion prompt engineer. "
            "Rewrite the user's prompt to be highly detailed, listing art styles, lighting, and camera angles. "
            "Output ONLY the comma-separated prompt tags. Do not output any conversational text."
        )
        
        # Resolve correct API key for the chosen provider
        _enhancer_key_map = {"Gemini": "gemini", "OpenAI": "openai"}
        _enhancer_cfg_key = _enhancer_key_map.get(provider, "")
        enhancer_api_key = ConfigManager.get_provider_key(_enhancer_cfg_key) if _enhancer_cfg_key else ""

        self.enhancer.setup(
            provider_type=provider,
            model_name=api_model,
            api_key=enhancer_api_key,
            prompt=f"Enhance this prompt for high quality image generation: '{original}'",
            history=[],
            system_prompt=system_prompt
        )
        
        self.enhancer_accum = ""
        self.enhancer.chunk_received.connect(self._on_enhance_chunk)
        self.enhancer.finished.connect(self._on_enhance_done)
        self.enhancer.error.connect(self._on_enhance_error)
        self.enhancer.start()

    def _on_enhance_chunk(self, chunk):
        self.enhancer_accum += chunk

    def _on_enhance_done(self):
        final = self.enhancer_accum.strip()
        # Clean up any "Here is the prompt:" garbage if LLM ignored instructions
        if ":" in final[:20]: 
            parts = final.split(":", 1)
            if len(parts) > 1: final = parts[1].strip()
            
        self.pos_prompt.setText(final)
        self.pos_prompt.setDisabled(False)
        self.enhance_btn.setEnabled(True)
        self.set_status("✨ Prompt Enhanced!", "#00AA00")

    def _on_enhance_error(self, err):
        print(f"[Enhance Error] {err}")
        self.pos_prompt.setDisabled(False)
        self.enhance_btn.setEnabled(True)
        self.set_status("⚠️ Enhancement Failed", "#FF4444")
