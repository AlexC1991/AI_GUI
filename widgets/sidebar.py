from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFrame, QLabel, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
# --- KEY CHANGE: Import the new panel class ---
from widgets.sidebar_panels import ChatOptionsPanel, SystemInfoPanel, PlaceholderPanel, ImageGenPanel

class Sidebar(QWidget):
    mode_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("Sidebar")
        self.setStyleSheet("""
            #Sidebar { background-color: #252526; border-right: 1px solid #333; }
            
            QPushButton.nav_btn {
                background-color: transparent;
                color: #CCCCCC;
                text-align: left;
                padding: 10px 15px;
                border: none;
                font-size: 14px;
                border-radius: 5px;
                margin: 2px 10px;
            }
            QPushButton.nav_btn:hover { background-color: #3E3E3E; color: white; }
            QPushButton.nav_btn:checked { 
                background-color: #37373D; 
                color: white; 
                border-left: 3px solid #008080; 
            }
            
            QPushButton#SettingsBtn {
                background-color: transparent;
                color: #AAA;
                text-align: left;
                padding: 10px 15px;
                border-top: 1px solid #333;
            }
            QPushButton#SettingsBtn:hover { color: white; background-color: #333; }
        """)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 20, 0, 0)
        self.main_layout.setSpacing(5)

        # 1. TOP NAV
        self.nav_group = []
        self.chat_btn = self.create_nav_button("Chat", "chat")
        self.image_btn = self.create_nav_button("Image Gen", "image")
        self.audio_btn = self.create_nav_button("Audio", "audio")
        self.code_btn = self.create_nav_button("Code / IDE", "code")

        self.chat_btn.setChecked(True)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #333; margin: 15px 10px;")
        self.main_layout.addWidget(line)

        # 2. DYNAMIC AREA
        self.dynamic_area = QWidget()
        self.dynamic_layout = QVBoxLayout(self.dynamic_area)
        self.dynamic_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.dynamic_area, 1)

        # 3. SETTINGS BTN
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setObjectName("SettingsBtn")
        self.settings_btn.clicked.connect(lambda: self.change_mode("settings"))
        self.main_layout.addWidget(self.settings_btn)

        self.current_panel = None
        self.set_active_panel("chat")

    def create_nav_button(self, text, mode_key):
        btn = QPushButton(text)
        btn.setProperty("class", "nav_btn")
        btn.setCheckable(True)
        btn.clicked.connect(lambda: self.change_mode(mode_key))
        self.main_layout.addWidget(btn)
        self.nav_group.append(btn)
        return btn

    def change_mode(self, mode):
        # Update Visual Button State
        if mode == "settings":
            for btn in self.nav_group: btn.setChecked(False)
        else:
            for btn in self.nav_group:
                is_target = False
                if mode == "chat" and "Chat" in btn.text(): is_target = True
                if mode == "image" and "Image" in btn.text(): is_target = True
                if mode == "audio" and "Audio" in btn.text(): is_target = True
                if mode == "code" and "Code" in btn.text(): is_target = True
                btn.setChecked(is_target)

        # Swap Panels
        self.set_active_panel(mode)

        # Notify Main Window
        self.mode_changed.emit(mode)

    def set_active_panel(self, mode):
        # Clear old panel
        if self.current_panel:
            self.dynamic_layout.removeWidget(self.current_panel)
            self.current_panel.deleteLater()
            self.current_panel = None

        # Load New Panel
        if mode == "chat":
            self.current_panel = ChatOptionsPanel()
            # Hook up buttons for MainWindow
            self.change_model_btn = self.current_panel.change_model_btn

        elif mode == "settings":
            self.current_panel = SystemInfoPanel()
            self.change_model_btn = None

        elif mode == "image":
            # --- THE FIX IS HERE ---
            # Loads the correct ImageGenPanel with the "Generate" button
            self.current_panel = ImageGenPanel()

        elif mode == "audio":
            self.current_panel = PlaceholderPanel("Audio Studio")
        elif mode == "code":
            self.current_panel = PlaceholderPanel("Code IDE")

        # Add to layout
        if self.current_panel:
            self.dynamic_layout.addWidget(self.current_panel)

    def update_status(self, state):
        """Pass status updates to the active panel."""
        if self.current_panel and hasattr(self.current_panel, 'update_status'):
            self.current_panel.update_status(state)