from PySide6.QtWidgets import QWidget, QHBoxLayout, QTextEdit, QPushButton, QFrame, QVBoxLayout, QLabel
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
    # Signal emitted when clear button is clicked
    clear_requested = Signal()
    
    def __init__(self):
        super().__init__()
        
        # Track generation state
        self._is_generating = False

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

        # --- Common button style ---
        btn_style = """
            QPushButton {
                background-color: #2E2E2E;
                color: white;
                border: 2px solid #444;
                border-radius: 6px;
                font-size: 16px;
            }
            QPushButton:hover { background-color: #3E3E3E; }
            QPushButton:pressed { background-color: #1E1E1E; }
        """
        # --- Button 1: Attachment ---
        self.attach_btn = QPushButton("📎")
        self.attach_btn.setFixedSize(26, 36)
        self.attach_btn.setToolTip("Attach File")
        self.attach_btn.setStyleSheet(btn_style)

        # --- Button 2: Clear Chat ---
        self.clear_btn = QPushButton("🗑️")
        self.clear_btn.setFixedSize(26, 36)
        self.clear_btn.setToolTip("Clear Chat")
        self.clear_btn.setStyleSheet(btn_style)
        self.clear_btn.clicked.connect(self._on_clear_clicked)

        # --- Button 3: Send ---
        self.send_button = QPushButton("Send")
        self.send_button.setFixedSize(80, 36)
        self.send_button.setToolTip("Send Message")
        self.send_button.setStyleSheet(btn_style + "font-weight: bold;")

        # --- Speed Stats Label ---
        self.stats_label = QLabel("")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 11px;
                font-family: 'Segoe UI', monospace;
                padding: 2px 0;
            }
        """)
        self.stats_label.setVisible(False)

        # Layout: input row + stats underneath
        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.addWidget(self.input_wrapper, 1) # Add stretch factor 1 to push buttons right
        input_row.addWidget(self.attach_btn, 0, Qt.AlignBottom)
        input_row.addWidget(self.clear_btn, 0, Qt.AlignBottom)
        input_row.addWidget(self.send_button, 0, Qt.AlignBottom)

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(2)
        outer_layout.addLayout(input_row)
        outer_layout.addWidget(self.stats_label)

        # Replace direct children with the outer layout
        # Remove items from main_layout first
        while main_layout.count():
            main_layout.takeAt(0)
        main_layout.addLayout(outer_layout)

        self.input.focusInEvent = self._on_focus_in
        self.input.focusOutEvent = self._on_focus_out

    def show_speed(self, speed, tokens, duration):
        """Display tokens/sec stats below the input bar."""
        self.stats_label.setText(f"{speed:.1f} t/s  ·  {tokens} tokens  ·  {duration:.1f}s")
        self.stats_label.setVisible(True)

    def clear_speed(self):
        """Hide the speed stats display."""
        self.stats_label.setText("")
        self.stats_label.setVisible(False)

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
    
    # --- Generation State Methods ---
    
    def _on_clear_clicked(self):
        """Handle clear button click."""
        self.clear_requested.emit()
    
    def set_generating(self, is_generating: bool):
        """Update UI state when generation starts/stops.
        Disables input during generation.
        """
        self._is_generating = is_generating
        if is_generating:
            self.send_button.setEnabled(False)
            self.clear_btn.setEnabled(False)
            self.input.setEnabled(False)
            self.input.setPlaceholderText("Generating response...")
        else:
            self.send_button.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.input.setEnabled(True)
            self.input.setPlaceholderText("Type your message here...")
    
    def is_generating(self) -> bool:
        """Check if currently generating."""
        return self._is_generating
