from PySide6.QtWidgets import QWidget, QHBoxLayout, QTextEdit, QPushButton, QFrame, QVBoxLayout
from PySide6.QtCore import Qt, Signal

class MagicInput(QTextEdit):
    return_pressed = Signal()
    resize_needed = Signal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if not (event.modifiers() & Qt.ShiftModifier):
                self.return_pressed.emit()
                return
        super().keyPressEvent(event)
        self.resize_needed.emit()

class InputBar(QWidget):
    def __init__(self):
        super().__init__()

        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.input_wrapper = QFrame()
        self.input_wrapper.setObjectName("InputWrapper")
        self.input_wrapper.setFixedHeight(60)

        self.base_style = "border: 1.5px solid #555; border-radius: 5px;"
        self.input_wrapper.setStyleSheet(f"#InputWrapper {{ background-color: #333333; {self.base_style} }}")

        wrapper_layout = QVBoxLayout(self.input_wrapper)
        wrapper_layout.setContentsMargins(5, 5, 5, 5)

        self.input = MagicInput()
        self.input.setPlaceholderText("Type your message here...")
        self.input.setFrameShape(QFrame.NoFrame)
        self.input.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # --- SIGNALS ---
        self.input.return_pressed.connect(self.trigger_send)
        self.input.resize_needed.connect(self.adjust_height)

        # --- THE FIX: LISTEN FOR TEXT CHANGES ---
        # This ensures that when .clear() is called programmatically, it shrinks back.
        self.input.textChanged.connect(self.adjust_height)

        self.input.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                color: white;
                font-family: Segoe UI, sans-serif;
                font-size: 13px;
                border: none;
            }
        """)

        wrapper_layout.addWidget(self.input)

        self.send_button = QPushButton("Send")
        self.send_button.setFixedWidth(80)
        self.send_button.setFixedHeight(40)

        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #2E2E2E;
                color: white;
                border: 2px solid #444;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #3E3E3E; }
            QPushButton:pressed { background-color: #1E1E1E; }
        """)

        main_layout.addWidget(self.input_wrapper)
        main_layout.addWidget(self.send_button, 0, Qt.AlignBottom)

        self.input.focusInEvent = self._on_focus_in
        self.input.focusOutEvent = self._on_focus_out

    def adjust_height(self):
        doc_height = self.input.document().size().height()
        min_h, max_h = 60, 200
        new_height = doc_height + 20

        if new_height < min_h: new_height = min_h
        final_height = min(new_height, max_h)

        if new_height > max_h:
            self.input.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        else:
            self.input.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.input_wrapper.setFixedHeight(int(final_height))

        # Transparency Logic
        if final_height > 70:
            bg_color = "rgba(51, 51, 51, 150)"
        else:
            bg_color = "#333333"

        current_border = "1.5px solid #008080" if self.input.hasFocus() else "1.5px solid #555"

        self.input_wrapper.setStyleSheet(f"""
            #InputWrapper {{
                background-color: {bg_color};
                border: {current_border};
                border-radius: 5px;
            }}
        """)

    def trigger_send(self):
        self.send_button.click()
        # Note: We rely on the parent (MainWindow) calling input.clear()
        # The textChanged signal connected above will handle the resizing automatically.

    def _on_focus_in(self, event):
        self.adjust_height()
        QTextEdit.focusInEvent(self.input, event)

    def _on_focus_out(self, event):
        style = self.input_wrapper.styleSheet()
        self.input_wrapper.setStyleSheet(style.replace("1.5px solid #008080", "1.5px solid #555"))
        QTextEdit.focusOutEvent(self.input, event)