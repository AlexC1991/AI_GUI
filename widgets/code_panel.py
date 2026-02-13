"""
Code Panel - File viewer for AI-generated code.

Displays code files with syntax highlighting, copy, and download.
Opens when the user clicks a FileCard in the chat.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QFrame, QFileDialog, QComboBox, QLabel
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer


class CodePanel(QWidget):
    """Side panel for displaying code from AI responses."""

    # Language to file extension
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

    # Language display names (proper casing)
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

    def __init__(self):
        super().__init__()

        self.setFixedWidth(0)
        self.target_width = 500
        self.files = {}
        self.file_languages = {}

        self.setStyleSheet("""
            QWidget { background-color: #1E1E1E; border-left: 1px solid #333; }
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 13px;
                border: none;
                padding: 10px;
            }
            QPushButton {
                background-color: #333;
                color: white;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #444; }
            QComboBox {
                background-color: #252526;
                color: white;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 2px 10px;
                min-width: 150px;
                font-weight: bold;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #252526;
                color: white;
                selection-background-color: #008080;
            }
            QLabel { color: #888; font-size: 10px; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setFixedHeight(50)
        header.setStyleSheet("background-color: #252526; border-bottom: 1px solid #333;")

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 0, 10, 0)

        self.file_selector = QComboBox()
        self.file_selector.currentIndexChanged.connect(self._on_file_changed)

        self.lang_label = QLabel("")
        self.lang_label.setStyleSheet("color: #4EC9B0; font-weight: bold; font-size: 11px;")

        self.copy_btn = QPushButton("Copy")
        self.copy_btn.clicked.connect(self.copy_to_clipboard)

        self.download_btn = QPushButton("Download")
        self.download_btn.clicked.connect(self.download_file)

        self.close_btn = QPushButton("\u2715")
        self.close_btn.setFixedWidth(30)
        self.close_btn.setStyleSheet("border: none; background: transparent; font-size: 14px; color: #888;")
        self.close_btn.clicked.connect(self.slide_out)

        header_layout.addWidget(self.file_selector)
        header_layout.addWidget(self.lang_label)
        header_layout.addStretch()
        header_layout.addWidget(self.copy_btn)
        header_layout.addWidget(self.download_btn)
        header_layout.addWidget(self.close_btn)

        layout.addWidget(header)

        self.editor = QTextEdit()
        self.editor.setReadOnly(True)
        layout.addWidget(self.editor)

        self.animation = QPropertyAnimation(self, b"minimumWidth")
        self.animation.setDuration(400)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)

    def _get_extension(self, language: str) -> str:
        return self.LANG_EXTENSIONS.get(language.lower(), '.txt')

    def _get_display_name(self, language: str) -> str:
        return self.LANG_DISPLAY.get(language.lower(), language.capitalize() if language else 'Text')

    # ============================================
    # FILE MANAGEMENT
    # ============================================

    def add_file(self, filename, content, language="", auto_open=True):
        """Add a file to the panel. Set auto_open=False to load silently.
        If filename already exists with different content, appends a counter.
        """
        # Deduplicate: if same name + same content, just select it
        if filename in self.files and self.files[filename] == content:
            self.file_selector.setCurrentText(filename)
            self._on_file_changed()
            if auto_open and self.width() == 0:
                self.slide_in()
            return

        # If same name but different content, make unique
        if filename in self.files:
            base, ext = (filename.rsplit('.', 1) + [''])[:2]
            counter = 2
            while f"{base} ({counter}).{ext}" in self.files:
                counter += 1
            filename = f"{base} ({counter}).{ext}" if ext else f"{base} ({counter})"

        self.files[filename] = content
        if language:
            self.file_languages[filename] = language

        if self.file_selector.findText(filename) == -1:
            self.file_selector.addItem(filename)

        self.file_selector.setCurrentText(filename)
        self._on_file_changed()

        if auto_open and self.width() == 0:
            self.slide_in()

    def _on_file_changed(self):
        """Update editor when file selection changes."""
        filename = self.file_selector.currentText()
        if filename in self.files:
            self.editor.setPlainText(self.files[filename])

            if filename in self.file_languages:
                lang = self.file_languages[filename]
                self.lang_label.setText(self._get_display_name(lang))
            else:
                ext = filename.split('.')[-1] if '.' in filename else ''
                self.lang_label.setText(ext.upper() if ext else 'Text')

    def clear_files(self):
        """Clear all files."""
        self.files.clear()
        self.file_languages.clear()
        self.file_selector.clear()
        self.editor.clear()
        self.lang_label.setText("")

    # ============================================
    # ANIMATION
    # ============================================

    def slide_in(self):
        self.animation.setStartValue(0)
        self.animation.setEndValue(self.target_width)
        self.animation.start()
        self.animation.finished.connect(lambda: self.setFixedWidth(self.target_width))

    def slide_out(self):
        self.setMinimumWidth(self.target_width)
        self.setMaximumWidth(16777215)
        self.animation.setStartValue(self.target_width)
        self.animation.setEndValue(0)
        self.animation.start()
        self.animation.finished.connect(lambda: self.setFixedWidth(0))

    # ============================================
    # FILE OPERATIONS
    # ============================================

    def download_file(self):
        filename = self.file_selector.currentText()
        content = self.files.get(filename, "")

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save File", filename, "All Files (*.*)"
        )
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

    def copy_to_clipboard(self):
        self.editor.selectAll()
        self.editor.copy()

        original_text = self.copy_btn.text()
        self.copy_btn.setText("Copied!")
        QTimer.singleShot(1000, lambda: self.copy_btn.setText(original_text))
