"""
HuggingFace Authentication Dialog
Handles token-based auth for gated models like Flux
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from pathlib import Path


class HuggingFaceLoginDialog(QDialog):
    """Dialog to authenticate with HuggingFace for gated models."""
    
    TOKEN_URL = "https://huggingface.co/settings/tokens"
    TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"
    
    def __init__(self, parent=None, model_name: str = "this model"):
        super().__init__(parent)
        self.setWindowTitle("HuggingFace Authentication Required")
        self.setMinimumWidth(500)
        self.setModal(True)
        
        self._setup_ui(model_name)
        self._load_existing_token()
    
    def _setup_ui(self, model_name: str):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QLabel(f"🔐 Authentication Required")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #f59e0b;")
        layout.addWidget(header)
        
        # Explanation
        explain = QLabel(
            f"<b>{model_name}</b> is a gated model that requires HuggingFace authentication.<br><br>"
            "To access it, you need a HuggingFace account and access token:<br>"
            "1. Click <b>'Get Token'</b> to open HuggingFace (create account if needed)<br>"
            "2. Create a new token with <b>'Read'</b> permission<br>"
            "3. Copy and paste the token below"
        )
        explain.setWordWrap(True)
        explain.setStyleSheet("color: #ccc; line-height: 1.5;")
        layout.addWidget(explain)
        
        # Token input row
        token_layout = QHBoxLayout()
        
        token_label = QLabel("Token:")
        token_label.setStyleSheet("color: #aaa; font-weight: bold;")
        token_layout.addWidget(token_label)
        
        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("hf_xxxxxxxxxxxxxxxxxxxx")
        self.token_input.setEchoMode(QLineEdit.Password)
        self.token_input.setStyleSheet("""
            QLineEdit {
                background: #252526;
                color: white;
                border: 1px solid #444;
                padding: 8px;
                border-radius: 4px;
                font-family: monospace;
            }
            QLineEdit:focus {
                border: 1px solid #008080;
            }
        """)
        token_layout.addWidget(self.token_input, 1)
        
        # Show/hide toggle
        self.show_btn = QPushButton("👁")
        self.show_btn.setFixedSize(35, 35)
        self.show_btn.setCheckable(True)
        self.show_btn.setStyleSheet("""
            QPushButton {
                background: #333;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QPushButton:checked {
                background: #008080;
            }
        """)
        self.show_btn.toggled.connect(self._toggle_visibility)
        token_layout.addWidget(self.show_btn)
        
        layout.addLayout(token_layout)
        
        # Button row
        btn_layout = QHBoxLayout()
        
        self.get_token_btn = QPushButton("🌐 Get Token")
        self.get_token_btn.setStyleSheet("""
            QPushButton {
                background: #0066cc;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #0077ee;
            }
        """)
        self.get_token_btn.clicked.connect(self._open_token_page)
        btn_layout.addWidget(self.get_token_btn)
        
        btn_layout.addStretch()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background: #333;
                color: #ccc;
                border: 1px solid #444;
                padding: 10px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #444;
            }
        """)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        self.save_btn = QPushButton("✓ Save & Continue")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background: #008080;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #009999;
            }
            QPushButton:disabled {
                background: #333;
                color: #666;
            }
        """)
        self.save_btn.clicked.connect(self._save_token)
        btn_layout.addWidget(self.save_btn)
        
        layout.addLayout(btn_layout)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.status_label)
        
        # Style the dialog
        self.setStyleSheet("""
            QDialog {
                background: #1e1e1e;
            }
        """)
    
    def _toggle_visibility(self, show: bool):
        self.token_input.setEchoMode(
            QLineEdit.Normal if show else QLineEdit.Password
        )
    
    def _open_token_page(self):
        QDesktopServices.openUrl(QUrl(self.TOKEN_URL))
        self.status_label.setText("Browser opened - create a token with 'Read' permission")
        self.status_label.setStyleSheet("color: #0099cc; font-size: 11px;")
    
    def _load_existing_token(self):
        """Load existing token if present."""
        if self.TOKEN_PATH.exists():
            try:
                token = self.TOKEN_PATH.read_text().strip()
                if token:
                    self.token_input.setText(token)
                    self.status_label.setText("Existing token found - it may have expired or lack permissions")
                    self.status_label.setStyleSheet("color: #f59e0b; font-size: 11px;")
            except:
                pass
    
    def _save_token(self):
        """Validate and save the token."""
        token = self.token_input.text().strip()
        
        if not token:
            self.status_label.setText("Please enter a token")
            self.status_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")
            return
        
        if not token.startswith("hf_"):
            self.status_label.setText("Token should start with 'hf_'")
            self.status_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")
            return
        
        try:
            # Create directory if needed
            self.TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            # Save token
            self.TOKEN_PATH.write_text(token)
            
            # Also set for huggingface_hub
            try:
                from huggingface_hub import HfFolder
                HfFolder.save_token(token)
            except:
                pass  # File save is enough
            
            self.status_label.setText("Token saved!")
            self.status_label.setStyleSheet("color: #4ade80; font-size: 11px;")
            
            self.accept()
            
        except Exception as e:
            self.status_label.setText(f"Failed to save: {e}")
            self.status_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")
    
    def get_token(self) -> str:
        """Return the entered token."""
        return self.token_input.text().strip()


def check_hf_auth() -> bool:
    """Check if HuggingFace token exists."""
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        token = token_path.read_text().strip()
        return bool(token and token.startswith("hf_"))
    return False


def request_hf_auth(parent=None, model_name: str = "this model") -> bool:
    """
    Show auth dialog and return True if user authenticated.
    Use this when a gated model fails to load.
    """
    dialog = HuggingFaceLoginDialog(parent, model_name)
    result = dialog.exec()
    return result == QDialog.Accepted
