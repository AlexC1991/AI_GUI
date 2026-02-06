"""
FileCard - Clickable code file cards for chat messages.

When an AI response contains code blocks, FileCards appear below
the message bubble. Clicking a card opens the CodePanel.
"""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame,
    QPushButton, QFileDialog
)
from PySide6.QtCore import Qt, Signal


LANG_EXTENSIONS = {
    'python': '.py', 'py': '.py', 'python3': '.py',
    'javascript': '.js', 'js': '.js', 'jsx': '.jsx',
    'typescript': '.ts', 'ts': '.ts', 'tsx': '.tsx',
    'go': '.go', 'golang': '.go',
    'rust': '.rs', 'rs': '.rs',
    'c': '.c', 'cpp': '.cpp', 'c++': '.cpp', 'cc': '.cpp',
    'csharp': '.cs', 'cs': '.cs', 'c#': '.cs',
    'java': '.java', 'kotlin': '.kt', 'kt': '.kt',
    'html': '.html', 'css': '.css',
    'json': '.json', 'yaml': '.yaml', 'yml': '.yml',
    'bash': '.sh', 'shell': '.sh', 'sh': '.sh', 'zsh': '.zsh',
    'powershell': '.ps1', 'sql': '.sql',
    'ruby': '.rb', 'php': '.php',
    'swift': '.swift', 'lua': '.lua',
    'text': '.txt', '': '.txt'
}

LANG_DISPLAY = {
    'python': 'Python', 'py': 'Python', 'python3': 'Python',
    'javascript': 'JavaScript', 'js': 'JavaScript',
    'typescript': 'TypeScript', 'ts': 'TypeScript',
    'go': 'Go', 'golang': 'Go',
    'rust': 'Rust', 'rs': 'Rust',
    'c': 'C', 'cpp': 'C++', 'c++': 'C++',
    'csharp': 'C#', 'cs': 'C#',
    'java': 'Java', 'kotlin': 'Kotlin',
    'html': 'HTML', 'css': 'CSS',
    'json': 'JSON', 'yaml': 'YAML',
    'bash': 'Bash', 'shell': 'Shell', 'sh': 'Shell',
    'sql': 'SQL', 'ruby': 'Ruby', 'php': 'PHP',
    'swift': 'Swift', 'lua': 'Lua',
    'text': 'Text', '': 'Text'
}

LANG_ICONS = {
    'python': '\U0001F40D', 'py': '\U0001F40D', 'python3': '\U0001F40D',
    'javascript': '\U0001F310', 'js': '\U0001F310',
    'typescript': '\U0001F4D8', 'ts': '\U0001F4D8',
    'html': '\U0001F310', 'css': '\U0001F3A8',
    'json': '\U0001F4CB', 'yaml': '\U0001F4CB',
    'bash': '\U0001F4BB', 'shell': '\U0001F4BB', 'sh': '\U0001F4BB',
    'sql': '\U0001F5C3',
}


class FileCard(QFrame):
    """A clickable card representing a detected code file."""

    clicked = Signal(str, str, str)  # filename, content, language

    def __init__(self, filename, content, language="text"):
        super().__init__()
        self.filename = filename
        self.content = content
        self.language = language

        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(48)
        self.setMinimumWidth(170)
        self.setMaximumWidth(300)

        self.setObjectName("FileCard")
        self.setStyleSheet("""
            #FileCard {
                background-color: #252526;
                border: 1px solid #444;
                border-radius: 8px;
            }
            #FileCard:hover {
                background-color: #2A2D2E;
                border: 1px solid #4EC9B0;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 8, 4)
        layout.setSpacing(8)

        # File icon
        icon_text = LANG_ICONS.get(language.lower(), '\U0001F4C4')
        icon_label = QLabel(icon_text)
        icon_label.setFixedWidth(20)
        icon_label.setStyleSheet("font-size: 16px; background: transparent; border: none;")
        layout.addWidget(icon_label)

        # Filename + language
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(0)

        name_label = QLabel(filename)
        name_label.setStyleSheet(
            "color: #E0E0E0; font-size: 12px; font-weight: bold; "
            "background: transparent; border: none;"
        )
        info_layout.addWidget(name_label)

        display_name = LANG_DISPLAY.get(language.lower(), language.capitalize() if language else 'Text')
        lang_label = QLabel(display_name)
        lang_label.setStyleSheet(
            "color: #4EC9B0; font-size: 10px; "
            "background: transparent; border: none;"
        )
        info_layout.addWidget(lang_label)

        layout.addLayout(info_layout, 1)

        # Download button
        dl_btn = QPushButton("\u2B73")
        dl_btn.setFixedSize(28, 28)
        dl_btn.setToolTip("Download file")
        dl_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border: 1px solid #444;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #444; }
        """)
        dl_btn.clicked.connect(self._download)
        layout.addWidget(dl_btn)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.filename, self.content, self.language)
        super().mousePressEvent(event)

    def _download(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save File", self.filename, "All Files (*.*)"
        )
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.content)


class FileCardRow(QWidget):
    """Horizontal row of FileCards, placed below a message bubble."""

    def __init__(self):
        super().__init__()
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(50, 4, 50, 8)
        self._layout.setSpacing(8)
        self._layout.setAlignment(Qt.AlignLeft)
        self.setStyleSheet("background: transparent;")

    def add_card(self, filename, content, language="text"):
        """Create and add a FileCard. Returns the card for signal connections."""
        card = FileCard(filename, content, language)
        self._layout.addWidget(card)
        return card
