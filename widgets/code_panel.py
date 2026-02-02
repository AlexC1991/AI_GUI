from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QFrame, QFileDialog, QComboBox, QSizePolicy, QGraphicsOpacityEffect
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QSize

class CodePanel(QWidget):
    def __init__(self):
        super().__init__()

        # Start hidden with 0 width for animation
        self.setFixedWidth(0)
        self.target_width = 500

        # Data Store for Files { "script.py": "print('hi')", ... }
        self.files = {}

        # Styling
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
            
            /* DROPDOWN STYLING */
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
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- HEADER ---
        header = QFrame()
        header.setFixedHeight(50)
        header.setStyleSheet("background-color: #252526; border-bottom: 1px solid #333;")

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 0, 10, 0)

        # FILE SELECTOR (Dropdown)
        self.file_selector = QComboBox()
        self.file_selector.currentIndexChanged.connect(self._on_file_changed)

        # Buttons
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.clicked.connect(self.copy_to_clipboard)

        self.download_btn = QPushButton("Download")
        self.download_btn.clicked.connect(self.download_file)

        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedWidth(30)
        self.close_btn.setStyleSheet("border: none; background: transparent; font-size: 14px; color: #888;")
        self.close_btn.clicked.connect(self.slide_out) # Trigger animation on close

        header_layout.addWidget(self.file_selector)
        header_layout.addStretch()
        header_layout.addWidget(self.copy_btn)
        header_layout.addWidget(self.download_btn)
        header_layout.addWidget(self.close_btn)

        layout.addWidget(header)

        # --- EDITOR ---
        self.editor = QTextEdit()
        self.editor.setReadOnly(True)
        layout.addWidget(self.editor)

        # --- ANIMATION SETUP ---
        self.animation = QPropertyAnimation(self, b"minimumWidth")
        self.animation.setDuration(400) # 400ms speed
        self.animation.setEasingCurve(QEasingCurve.OutCubic) # Smooth "slow down at end" effect

        # We animate minimumWidth because fixedWidth locks it. 
        # By animating minWidth, layout adjusts automatically.

    def add_file(self, filename, content):
        """Adds a file to the list. If panel is closed, it opens it."""
        self.files[filename] = content

        # Add to dropdown if not exists
        if self.file_selector.findText(filename) == -1:
            self.file_selector.addItem(filename)

        # Select this new file
        self.file_selector.setCurrentText(filename)

        # Open Panel if closed
        if self.width() == 0:
            self.slide_in()

    def _on_file_changed(self):
        """Updates editor when dropdown selection changes."""
        filename = self.file_selector.currentText()
        if filename in self.files:
            self.editor.setPlainText(self.files[filename])

    def slide_in(self):
        """Animate opening."""
        self.animation.setStartValue(0)
        self.animation.setEndValue(self.target_width)
        self.animation.start()
        # Ensure we set fixed width at end to lock it
        self.animation.finished.connect(lambda: self.setFixedWidth(self.target_width))

    def slide_out(self):
        """Animate closing."""
        # Unset fixed width so it can shrink
        self.setMinimumWidth(self.target_width)
        self.setMaximumWidth(16777215) # Release max width lock

        self.animation.setStartValue(self.target_width)
        self.animation.setEndValue(0)
        self.animation.start()
        self.animation.finished.connect(lambda: self.setFixedWidth(0))

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

        # Flash "Copied!" text
        original_text = self.copy_btn.text()
        self.copy_btn.setText("Copied!")
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1000, lambda: self.copy_btn.setText(original_text))