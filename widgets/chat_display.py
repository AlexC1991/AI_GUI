from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QFrame, QScrollBar
from PySide6.QtCore import Qt, QTimer
from widgets.message_bubble import MessageBubble, ThinkingBubble

class ChatDisplay(QScrollArea):
    def __init__(self):
        super().__init__()

        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QWidget { background: transparent; }
            QScrollBar:vertical { background: #121212; width: 8px; }
            QScrollBar::handle:vertical { background: #555; border-radius: 4px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)

        self.container = QWidget()
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setSpacing(15)
        self.layout.setContentsMargins(50, 20, 50, 20)

        self.container.setLayout(self.layout)
        self.setWidget(self.container)

        self.verticalScrollBar().rangeChanged.connect(self.scroll_to_bottom)

    def scroll_to_bottom(self):
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def add_message(self, text, sender="user"):
        bubble = MessageBubble(text, sender)
        self.layout.addWidget(bubble)

    # --- NEW FUNCTION: CLEAR CHAT ---
    def clear_chat(self):
        """Deletes all message bubbles."""
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def show_thinking(self):
        thinking = ThinkingBubble()
        self.layout.addWidget(thinking)
        return thinking

    def remove_bubble(self, bubble_widget):
        self.layout.removeWidget(bubble_widget)
        bubble_widget.deleteLater()

    def stream_ai_message(self, full_text):
        bubble = MessageBubble("", "ai")
        self.layout.addWidget(bubble)

        self.stream_index = 0
        self.stream_text = full_text
        self.current_bubble = bubble

        self.type_timer = QTimer()
        self.type_timer.timeout.connect(self._type_next_char)
        self.type_timer.start(20)

    def _type_next_char(self):
        if self.stream_index < len(self.stream_text):
            current_displayed = self.stream_text[:self.stream_index+1]

            # Change: Use .set_content() which now handles the text_browser internally
            self.current_bubble.set_content(current_displayed)

            self.stream_index += 1

            # Force Scroll to bottom (Important for large text updates)
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        else:
            self.type_timer.stop()