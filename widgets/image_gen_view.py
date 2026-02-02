from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QComboBox, QPushButton, QFrame, QScrollArea, QSlider,
    QSpinBox, QGridLayout, QGroupBox, QDoubleSpinBox, QTabWidget, QProgressBar, QSplitter
)
from PySide6.QtCore import Qt, QSize

# =======================================================
# CUSTOM WIDGET: THE "LORA CARD"
# =======================================================
class LoraCard(QFrame):
    def __init__(self, parent_rack):
        super().__init__()
        self.parent_rack = parent_rack
        self.setFixedHeight(60) # Compact height
        self.setStyleSheet("""
            QFrame { background-color: #2D2D30; border: 1px solid #3E3E42; border-radius: 6px; }
            QLabel { border: none; background: transparent; color: #CCC; }
            QSlider::handle:horizontal { background: #008080; }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(10)

        # Name Dropdown
        self.name_combo = QComboBox()
        self.name_combo.addItems(["PixelArt_V3", "Cyborg_Style_XL", "Neon_Lights_LoRA", "Anime_Detailer"])
        self.name_combo.setStyleSheet("background: #222; border: 1px solid #444; min-width: 120px;")
        layout.addWidget(self.name_combo, 2)

        # Strength Slider
        layout.addWidget(QLabel("Str:", styleSheet="font-size: 11px;"))
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 200)
        self.strength_slider.setValue(100)
        self.strength_slider.setFixedWidth(100)

        self.val_label = QLabel("1.0")
        self.val_label.setStyleSheet("font-weight: bold; color: white; min-width: 30px;")
        self.strength_slider.valueChanged.connect(lambda v: self.val_label.setText(str(v/100)))

        layout.addWidget(self.strength_slider)
        layout.addWidget(self.val_label)

        # Delete Button
        delete_btn = QPushButton("✕")
        delete_btn.setFixedSize(20, 20)
        delete_btn.setStyleSheet("background: transparent; color: #AA5555; border: none; font-weight: bold;")
        delete_btn.clicked.connect(self.remove_self)
        layout.addWidget(delete_btn)

    def remove_self(self):
        self.parent_rack.remove_card(self)

# =======================================================
# MAIN VIEW: IMAGE GEN 
# =======================================================
class ImageGenView(QWidget):
    def __init__(self):
        super().__init__()

        # Use a Horizontal Splitter to divide Left (Prompts) and Right (Output)
        # This allows the user to resize the width of the columns
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(2)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.main_splitter)

        # --- LEFT COLUMN (Inputs) ---
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_layout.setContentsMargins(20, 20, 10, 20)
        self.left_layout.setSpacing(15)

        self.create_prompt_section()
        self.create_lora_section()

        self.main_splitter.addWidget(self.left_widget)

        # --- RIGHT COLUMN (Settings & Output) ---
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setContentsMargins(10, 20, 20, 20)
        self.right_layout.setSpacing(15)

        self.create_settings_section()
        self.create_output_section()

        self.main_splitter.addWidget(self.right_widget)

        # Set Initial Split (50% / 50%)
        self.main_splitter.setSizes([600, 600])

    def create_prompt_section(self):
        # 1. Checkpoint & Refiner (Compact)
        ckpt_group = QGroupBox("Checkpoint Model")
        ckpt_group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 6px; font-weight: bold; color: #008080; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; }")
        ckpt_layout = QHBoxLayout(ckpt_group)

        self.ckpt_combo = QComboBox()
        self.ckpt_combo.addItems(["Juggernaut_XL_v9", "Realistic_Vision_V6", "DreamShaper_8", "SDXL_Base_1.0"])
        self.ckpt_combo.setStyleSheet("padding: 5px; background: #252526; color: white; border: 1px solid #444;")

        self.refiner_check = QPushButton("✨ Enable Refiner")
        self.refiner_check.setCheckable(True)
        self.refiner_check.setStyleSheet("""
            QPushButton { background: #333; color: #CCC; border: 1px solid #444; padding: 5px 10px; border-radius: 4px; }
            QPushButton:checked { background: #006666; color: white; border: 1px solid #008080; }
        """)

        ckpt_layout.addWidget(self.ckpt_combo, 1)
        ckpt_layout.addWidget(self.refiner_check)
        self.left_layout.addWidget(ckpt_group)

        # 2. Prompts
        self.left_layout.addWidget(QLabel("Positive Prompt:", styleSheet="color: #AAA; font-weight: bold;"))
        self.pos_prompt = QTextEdit()
        self.pos_prompt.setPlaceholderText("A futuristic cyberpunk detective...")
        self.pos_prompt.setStyleSheet("background: #252526; color: #E0E0E0; border: 1px solid #444;")
        self.left_layout.addWidget(self.pos_prompt, 2) # Stretch factor 2

        self.left_layout.addWidget(QLabel("Negative Prompt:", styleSheet="color: #AAA; font-weight: bold;"))
        self.neg_prompt = QTextEdit()
        self.neg_prompt.setPlaceholderText("blurry, low quality, deformed...")
        self.neg_prompt.setFixedHeight(50) # Fixed small height
        self.neg_prompt.setStyleSheet("background: #252526; color: #E0E0E0; border: 1px solid #444;")
        self.left_layout.addWidget(self.neg_prompt)

    def create_lora_section(self):
        # 3. LoRA Rack
        lora_group = QGroupBox("LoRA Stack")
        lora_group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 6px; font-weight: bold; color: #008080; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; }")
        layout = QVBoxLayout(lora_group)
        layout.setContentsMargins(5, 15, 5, 5)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        self.rack_container = QWidget()
        self.rack_layout = QVBoxLayout(self.rack_container)
        self.rack_layout.setAlignment(Qt.AlignTop)
        self.rack_layout.setSpacing(5)

        scroll.setWidget(self.rack_container)
        layout.addWidget(scroll)

        add_btn = QPushButton("+ Add LoRA")
        add_btn.setStyleSheet("background: #333; color: white; border: 1px dashed #555; padding: 6px;")
        add_btn.clicked.connect(self.add_lora_card)
        layout.addWidget(add_btn)

        self.left_layout.addWidget(lora_group, 1) # Stretch factor 1

    def create_settings_section(self):
        # Settings Grid
        settings_frame = QFrame()
        settings_frame.setStyleSheet("background-color: #1E1E1E; border-radius: 6px;")
        grid = QGridLayout(settings_frame)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setVerticalSpacing(10)

        # Labels
        grid.addWidget(QLabel("Resolution:", styleSheet="color:#CCC"), 0, 0)
        grid.addWidget(QLabel("Batch Size:", styleSheet="color:#CCC"), 1, 0)
        grid.addWidget(QLabel("Steps:", styleSheet="color:#CCC"), 0, 2)
        grid.addWidget(QLabel("CFG Scale:", styleSheet="color:#CCC"), 1, 2)

        # Controls
        self.res_combo = QComboBox()
        self.res_combo.addItems(["1024x1024", "1152x896", "896x1152", "1344x768"])
        self.res_combo.setStyleSheet("background: #252526; color: white; border: 1px solid #444; padding: 2px;")
        grid.addWidget(self.res_combo, 0, 1)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 4)
        self.batch_spin.setStyleSheet("background: #252526; color: white; border: 1px solid #444;")
        self.batch_spin.valueChanged.connect(self.update_tab_count)
        grid.addWidget(self.batch_spin, 1, 1)

        self.steps = QSpinBox()
        self.steps.setValue(30)
        self.steps.setStyleSheet("background: #252526; color: white; border: 1px solid #444;")
        grid.addWidget(self.steps, 0, 3)

        self.cfg = QDoubleSpinBox()
        self.cfg.setValue(7.0)
        self.cfg.setStyleSheet("background: #252526; color: white; border: 1px solid #444;")
        grid.addWidget(self.cfg, 1, 3)

        self.right_layout.addWidget(settings_frame)

    def create_output_section(self):
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: none; background-color: #222; border-radius: 3px; }
            QProgressBar::chunk { background-color: #00AA00; border-radius: 3px; }
        """)
        self.right_layout.addWidget(self.progress_bar)

        # Tabbed Output
        self.output_tabs = QTabWidget()
        self.output_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; background: #111; }
            QTabBar::tab { background: #222; color: #888; padding: 8px 15px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #333; color: white; border-top: 2px solid #008080; }
        """)

        # Initial Tabs
        self.update_tab_count(1)
        self.right_layout.addWidget(self.output_tabs)

    def add_lora_card(self):
        card = LoraCard(self)
        self.rack_layout.addWidget(card)

    def remove_card(self, card):
        self.rack_layout.removeWidget(card)
        card.deleteLater()

    def update_tab_count(self, count):
        self.output_tabs.clear()
        for i in range(count):
            lbl = QLabel(f"Preview Image {i+1}\n(Waiting for generation...)")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #555; font-style: italic;")
            self.output_tabs.addTab(lbl, f"Image {i+1}")