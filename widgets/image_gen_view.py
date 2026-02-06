from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QComboBox, QPushButton, QFrame, QScrollArea, QSlider,
    QSpinBox, QGridLayout, QGroupBox, QDoubleSpinBox, QTabWidget, 
    QProgressBar, QSplitter, QCheckBox, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
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
    "unknown": {"name": "Unknown", "native_res": 512, "default_steps": 25, "default_cfg": 7.0, "supports_negative": True, "prompt_hint": "Try tags or natural"},
}

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
        
        self.right_layout.addWidget(frame)
    
    def _create_output_section(self):
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color:#888;font-size:11px")
        self.right_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m")
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setStyleSheet("QProgressBar{border:1px solid #333;background:#222;border-radius:4px;text-align:center;color:white}QProgressBar::chunk{background:#088;border-radius:3px}")
        self.right_layout.addWidget(self.progress_bar)
        
        self.output_tabs = QTabWidget()
        self.output_tabs.setStyleSheet("QTabWidget::pane{border:1px solid #333;background:#111}QTabBar::tab{background:#222;color:#888;padding:6px 12px}QTabBar::tab:selected{background:#333;color:white;border-top:2px solid #088}")
        self.update_tab_count(1)
        self.right_layout.addWidget(self.output_tabs, 1)
    
    def _on_ckpt_changed(self, name):
        if not name or "No checkpoint" in name: return
        family = detect_model_family(name)
        profile = MODEL_PROFILES.get(family, MODEL_PROFILES["unknown"])
        self.current_family = family
        self.family_label.setText(f"Family: {profile['name']}")
        self.hint_label.setText(profile['prompt_hint'])
        
        self.neg_label.setVisible(profile['supports_negative'])
        self.neg_prompt.setVisible(profile['supports_negative'])
        
        if self.auto_cfg_check.isChecked():
            self.cfg.setValue(profile['default_cfg'])
            self.steps.setValue(profile['default_steps'])
    
    def refresh_assets(self):
        config = ConfigManager.load_config()
        img_cfg = config.get("image", {})
        ckpt_dir = Path(img_cfg.get("checkpoint_dir", "models/checkpoints"))
        lora_dir = Path(img_cfg.get("lora_dir", "models/loras"))
        vae_dir = Path(img_cfg.get("vae_dir", "models/vae"))
        # Using configured dir or default
        text_enc_dir = Path(img_cfg.get("text_encoder_dir", "models/text_encoders"))
        
        # Checkpoints
        ckpts = []
        if ckpt_dir.exists():
            for ext in ["*.safetensors", "*.ckpt", "*.gguf"]:
                ckpts.extend([f.name for f in ckpt_dir.glob(ext)])
        ckpts.sort()
        
        cur = self.ckpt_combo.currentText()
        self.ckpt_combo.blockSignals(True)
        self.ckpt_combo.clear()
        self.ckpt_combo.addItems(ckpts if ckpts else ["No checkpoints found"])
        if cur in ckpts: self.ckpt_combo.setCurrentText(cur)
        self.ckpt_combo.blockSignals(False)
        if ckpts: self._on_ckpt_changed(self.ckpt_combo.currentText())
        
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
            lbl = QLabel(f"Image {i+1}\\n(Waiting...)")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color:#555;font-style:italic")
            lbl.setMinimumSize(350, 350)
            lbl.setObjectName("preview_label")
            lay.addWidget(lbl)
            self.output_tabs.addTab(tab, f"Image {i+1}")
    
    # === API ===
    def get_selected_checkpoint(self): return self.ckpt_combo.currentText()
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
        self.status_label.setStyleSheet(f"color:{color};font-size:11px")
    
    def show_image(self, path, tab_index=0):
        if tab_index < self.output_tabs.count():
            tab = self.output_tabs.widget(tab_index)
            lbl = tab.findChild(QLabel, "preview_label")
            if lbl and Path(path).exists():
                # FIX: Actually load the image from path
                pix = QPixmap(str(path))
                if not pix.isNull():
                    # Scale to fit label while maintaining aspect ratio
                    scaled = pix.scaled(
                        lbl.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    lbl.setPixmap(scaled)
                    lbl.setStyleSheet("")

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
        
        self.enhancer.setup(
            provider_type=provider,
            model_name=api_model,
            api_key=llm_cfg.get("api_key", ""),
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
