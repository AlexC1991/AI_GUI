from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QFrame, QScrollBar
from PySide6.QtCore import Qt, QTimer
from widgets.message_bubble import MessageBubble, ThinkingBubble, ThinkingSection

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

        # DATA STORAGE
        self.messages = [] # [{'role': 'user', 'content': '...'}, ...]
        
        # THROTTLING (Optimization)
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(50) # Update UI every 50ms max
        self.update_timer.timeout.connect(self._process_buffered_chunk)
        self.buffered_text = ""
        self.is_streaming = False

    def scroll_to_bottom(self):
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def add_message(self, text, sender="user", display_text=None):
        """
        text: The actual content for the LLM.
        display_text: What to show in the UI (optional, defaults to text).
        """
        self.is_streaming = False
        self.update_timer.stop()
        
        # Store in History
        role = "assistant" if sender == "ai" else "user"
        self.messages.append({"role": role, "content": text})
        
        # Show in UI
        content_to_show = display_text if display_text else text
        bubble = MessageBubble(content_to_show, sender)
        self.layout.addWidget(bubble)

    def get_history(self):
        return self.messages

    def clear_chat(self):
        """Deletes all message bubbles and history."""
        self.messages = []
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

    def start_streaming_message(self):
        """Creates a new empty bubble for the AI response."""
        # Add placeholder to history
        self.messages.append({"role": "assistant", "content": ""})
        
        bubble = MessageBubble("", "ai")
        self.layout.addWidget(bubble)
        self.current_bubble = bubble
        self.stream_text = ""
        self.buffered_text = ""
        self.current_bubble.set_content("...") # Initial placeholder
        
        self.is_streaming = True
        self.update_timer.start()
        
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def update_streaming_message(self, chunk):
        """Appends chunk to buffer, processed by timer."""
        if not hasattr(self, 'current_bubble') or self.current_bubble is None:
            self.start_streaming_message()
            
        self.buffered_text += chunk

    def _process_buffered_chunk(self):
        """Called by timer to update UI."""
        if not self.is_streaming or not self.buffered_text:
            return
            
        # Only update if there is new content
        # We append buffer to main text and clear buffer?
        # No, update_streaming_message appends to buffer.
        # But wait, self.stream_text needs to hold TOTAL text.
        
        self.stream_text += self.buffered_text
        self.buffered_text = ""
        
        self.current_bubble.set_content(self.stream_text)
        
        # Update History
        if self.messages:
            self.messages[-1]['content'] = self.stream_text
        
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def end_streaming_message(self):
        """Finalizes the stream."""
        # Process remaining buffer
        self._process_buffered_chunk()
        
        self.update_timer.stop()
        self.is_streaming = False
        self.current_bubble = None
        self.stream_text = ""
        self.buffered_text = ""

    def insert_widget_after_last_message(self, widget):
        """Insert a widget (e.g. FileCardRow) at the end of the chat layout."""
        self.layout.addWidget(widget)

    def rewrite_last_message(self, new_text, history_text=None):
        """Updates the display of the last message bubble.

        If history_text is provided, the message history keeps that value
        (preserving full context for the AI) while the bubble shows new_text.
        """
        if not self.messages: return

        # Update history - keep full content for AI context if provided
        self.messages[-1]['content'] = history_text if history_text else new_text

        # Update display - find the last MessageBubble in layout
        for i in range(self.layout.count() - 1, -1, -1):
            item = self.layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, MessageBubble):
                widget.set_content(new_text)
                break

    def start_thinking_section(self):
        """Create and add a ThinkingSection widget. Returns it for streaming."""
        section = ThinkingSection()
        self.layout.addWidget(section)
        return section

    def end_thinking_section(self, section):
        """Finalize a thinking section (collapse it, stop animation)."""
        if section:
            section.finalize()