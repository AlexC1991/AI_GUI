from PySide6.QtWidgets import QFrame, QVBoxLayout, QSizePolicy
from widgets.chat_display import ChatDisplay

class ChatView(QFrame):  # Changed from QWidget to QFrame for better styling
    def __init__(self):
        super().__init__()

        self.setObjectName("ChatBox")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            #ChatBox {
                border: 5px solid #3C3C3C;   /* Your desired border thickness */
                border-radius: 15px;          /* Your desired corners */
                background-color: #121212;   /* Background color */
            }
        """)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)

        self.chat_display = ChatDisplay()
        self.layout.addWidget(self.chat_display)

        # --- TEST MESSAGES ---
        self.add_message("System initialized.", "ai")
        self.add_message("Border styling moved to wrapper.", "user")

    def add_message(self, text, sender="user"):
        self.chat_display.add_message(text, sender)