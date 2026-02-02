from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget
)
from PySide6.QtCore import QTimer
from widgets.sidebar import Sidebar
from widgets.chat_view import ChatView
from widgets.input_bar import InputBar
from widgets.code_panel import CodePanel
from widgets.settings_view import SettingsView
from widgets.image_gen_view import ImageGenView  # <--- NEW IMPORT
from utils.file_handler import FileHandler

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("VoxAI Orchestrator")
        self.resize(1400, 800)

        central = QWidget()
        self.setCentralWidget(central)

        # --- MAIN LAYOUT (3 Columns: Sidebar | Stack | CodePanel) ---
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. LEFT SIDEBAR
        self.sidebar = Sidebar()
        main_layout.addWidget(self.sidebar)

        # 2. CENTRAL AREA (STACKED WIDGET)
        # This acts like a deck of cards to switch views
        self.stack = QStackedWidget()

        # --- PAGE 0: CHAT INTERFACE ---
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

        self.stack.addWidget(self.chat_container_widget) # Index 0

        # --- PAGE 1: SETTINGS INTERFACE ---
        self.settings_view = SettingsView()
        self.stack.addWidget(self.settings_view) # Index 1

        # --- PAGE 2: IMAGE GEN INTERFACE ---
        self.image_gen_view = ImageGenView()
        self.stack.addWidget(self.image_gen_view) # Index 2

        # Add Stack to Main Layout
        main_layout.addWidget(self.stack, 1)

        # 3. RIGHT CODE PANEL
        self.code_panel = CodePanel()
        main_layout.addWidget(self.code_panel)

        # --- CONNECTIONS ---
        self.file_handler = FileHandler(self)

        # Connect Input Bar
        self.input_bar.send_button.clicked.connect(self.handle_send)
        if hasattr(self.input_bar.input, 'return_pressed'):
            self.input_bar.input.return_pressed.connect(self.handle_send)

        # Initial Button Connections (For Default Chat Mode)
        if self.sidebar.upload_btn:
            self.sidebar.upload_btn.clicked.connect(self.handle_upload)
        if self.sidebar.clear_btn:
            self.sidebar.clear_btn.clicked.connect(self.handle_clear_chat)

        # CRITICAL: View Switching
        self.sidebar.mode_changed.connect(self.handle_mode_switch)

    def handle_mode_switch(self, mode):
        """Switches the central view based on sidebar selection."""
        if mode == "settings":
            self.stack.setCurrentIndex(1) # Show Settings
            self.code_panel.slide_out()
        elif mode == "chat":
            self.stack.setCurrentIndex(0) # Show Chat
        elif mode == "image":
            self.stack.setCurrentIndex(2) # Show Image Gen
            self.code_panel.slide_out()
        else:
            # Fallback for Audio/Code modes
            self.stack.setCurrentIndex(0)

        # --- RECONNECT DYNAMIC SIDEBAR BUTTONS ---
        # 1. Disconnect everything first to avoid duplicates
        try: self.sidebar.upload_btn.clicked.disconnect()
        except: pass
        try: self.sidebar.clear_btn.clicked.disconnect()
        except: pass

        # 2. Reconnect only if we are in Chat mode
        if mode == "chat":
            if self.sidebar.upload_btn:
                self.sidebar.upload_btn.clicked.connect(self.handle_upload)
            if self.sidebar.clear_btn:
                self.sidebar.clear_btn.clicked.connect(self.handle_clear_chat)

        # (Optional) You can add connections for Image Gen buttons (Generate/Abort) here later
        # if mode == "image": ...

    def handle_send(self):
        user_text = self.input_bar.input.toPlainText().strip()
        if not user_text: return

        self.chat_view.chat_display.add_message(user_text, "user")
        self.input_bar.input.clear()

        self.sidebar.update_status("processing")

        if "code" in user_text.lower():
            QTimer.singleShot(500, self.simulate_code_response)
        else:
            QTimer.singleShot(500, self.simulate_text_response)

    def simulate_code_response(self):
        self.sidebar.update_status("thinking")
        self.thinking_bubble = self.chat_view.chat_display.show_thinking()

        script_1 = """# main.py
print("Hello World")
"""
        QTimer.singleShot(1500, lambda: self.code_panel.add_file("main.py", script_1))

        QTimer.singleShot(2000, lambda: self.finalize_ai_response(
            "I've generated the project files. Check the panel."
        ))

    def simulate_text_response(self):
        self.sidebar.update_status("thinking")
        self.thinking_bubble = self.chat_view.chat_display.show_thinking()
        response = "I have processed your request."
        QTimer.singleShot(2000, lambda: self.finalize_ai_response(response))

    def finalize_ai_response(self, text):
        if hasattr(self, 'thinking_bubble'):
            self.chat_view.chat_display.remove_bubble(self.thinking_bubble)
        self.sidebar.update_status("idle")
        self.chat_view.chat_display.stream_ai_message(text)

    def handle_clear_chat(self):
        self.chat_view.chat_display.clear_chat()
        self.sidebar.update_status("idle")
        self.code_panel.slide_out()

    def handle_upload(self):
        file_path = self.file_handler.open_file_dialog()
        if file_path:
            file_data = self.file_handler.process_file(file_path)
            user_msg = f"📂 Uploading File: {file_data['name']}"
            self.chat_view.chat_display.add_message(user_msg, "user")

            if file_data['extension'] in ['.py', '.txt', '.json', '.js']:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    self.code_panel.add_file(file_data['name'], content)
                except: pass

            QTimer.singleShot(1000, lambda: self.finalize_ai_response(
                f"I've loaded {file_data['name']} into the workspace."
            ))