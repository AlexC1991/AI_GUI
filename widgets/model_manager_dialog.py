from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, 
    QLabel, QProgressBar, QLineEdit, QMessageBox, QTabWidget, QWidget
)
from PySide6.QtCore import Qt
import subprocess
import shutil
import sys
from utils.ollama_helper import get_ollama_path
from backend.ollama_worker import OllamaWorker

# ===================================================================
# DIALOG
# ===================================================================
class ModelManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Local Model Manager (Ollama)")
        self.setFixedSize(600, 550) # Prevent stretching
        self.setStyleSheet("background-color: #1E1E1E; color: #EEE;")
        
        layout = QVBoxLayout(self)
        
        # TABS
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #333; color: #BBB; padding: 8px 12px; }
            QTabBar::tab:selected { background: #006666; color: white; }
        """)
        
        self.tab_installed = QWidget()
        self.tab_market = QWidget()
        
        self._setup_installed_tab()
        self._setup_market_tab()
        
        self.tabs.addTab(self.tab_market, "Marketplace / Download")
        self.tabs.addTab(self.tab_installed, "Installed Models")
        
        layout.addWidget(self.tabs)
        
        # STATUS & CLOSE
        self.status_bar = QLabel("Ready")
        self.status_bar.setWordWrap(True) # Wrap long status messages
        self.status_bar.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self.status_bar)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("padding: 6px; background: #444;")
        layout.addWidget(close_btn, 0, Qt.AlignRight)

    def _setup_installed_tab(self):
        layout = QVBoxLayout(self.tab_installed)
        
        self.list_installed = QListWidget()
        self.list_installed.setStyleSheet("background: #252525; border: 1px solid #333;")
        layout.addWidget(self.list_installed)
        
        refresh_btn = QPushButton("Refresh List")
        refresh_btn.clicked.connect(self.refresh_installed)
        refresh_btn.setStyleSheet("background: #006666; padding: 5px;")
        layout.addWidget(refresh_btn)
        
        self.refresh_installed()

    def _setup_market_tab(self):
        layout = QVBoxLayout(self.tab_market)
        
        # Curated List
        layout.addWidget(QLabel("Popular Models:"))
        self.list_market = QListWidget()
        self.list_market.setStyleSheet("background: #252525; border: 1px solid #333;")
        
        # POPULATE MARKET
        self.market_models = [
            "--- GENERAL / CHAT ---",
            "llama3.2 (3B - Fast & Smart)",
            "llama3.1 (8B - Standard)",
            "mistral-nemo (12B - Very Capable)",
            "gemma2 (9B - Google's Latest)",
            "phi3.5 (3.8B - Microsoft)",
            "mistral (7B - Classic)",
            "openhermes (7B - Creative)",
            
            "--- CODING ---",
            "qwen2.5-coder (7B - Best for Code)",
            "deepseek-coder-v2 (16B - Advanced Code)",
            "codellama (7B - Meta's Code)",
            "starcoder2 (3B - Fast Code)",
            
            "--- VISION (Image Input) ---",
            "llava (7B - Image Description)",
            "moondream (1.8B - Very Tiny Vision)",
            
            "--- FUN / UNCENSORED ---",
            "dolphin-mistral (7B - Uncensored)",
            "hermes3 (8B - Unlocked)",
            "solar (10.7B - Smart)"
        ]
        self.list_market.addItems(self.market_models)
        self.list_market.itemDoubleClicked.connect(lambda: self.custom_input.setText(self.list_market.currentItem().text().split()[0]))
        layout.addWidget(self.list_market)
        
        layout.addSpacing(10)
        
        # Custom Input
        layout.addWidget(QLabel("Or type ANY tag from ollama.com/library:", styleSheet="color: #AAA; margin-top: 10px;"))
        self.custom_input = QLineEdit()
        self.custom_input.setPlaceholderText("e.g. deepseek-coder:6.7b, mistral:latest, llama3:70b")
        self.custom_input.setStyleSheet("padding: 8px; background: #333; border: 1px solid #555; color: white;")
        layout.addWidget(self.custom_input)
        
        # Download Action
        dl_layout = QHBoxLayout()
        self.dl_btn = QPushButton("Download Selected / Typed")
        self.dl_btn.setStyleSheet("background-color: #008080; padding: 10px; font-weight: bold;")
        self.dl_btn.clicked.connect(self.start_download)
        dl_layout.addWidget(self.dl_btn)
        
        layout.addLayout(dl_layout)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("QProgressBar { text-align: center; } QProgressBar::chunk { background-color: #00AAAA; }")
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

    def refresh_installed(self):
        """Runs ollama list asynchronously"""
        self.list_installed.clear()
        self.list_installed.addItem("Loading... (Checking Ollama)")
        
        # Disable button while loading
        sender = self.sender()
        if isinstance(sender, QPushButton):
            sender.setEnabled(False)

        # Start Worker
        self.list_worker = OllamaWorker("list")
        self.list_worker.models_ready.connect(self.on_models_loaded)
        self.list_worker.error.connect(self.on_list_error)
        self.list_worker.finished.connect(lambda: self._enable_refresh_btn(sender))
        self.list_worker.start()

    def _enable_refresh_btn(self, btn):
        if hasattr(self, 'list_installed'):
             # If successful, 'Loading...' is cleared in on_models_loaded
             # If error, it stays or is replaced.
             pass
        if btn and isinstance(btn, QPushButton):
            btn.setEnabled(True)

    def on_models_loaded(self, models):
        self.list_installed.clear()
        if not models:
            self.list_installed.addItem("(No models found)")
            return
            
        for name in models:
            self.list_installed.addItem(name)
            
    def on_list_error(self, err):
        self.list_installed.clear()
        self.list_installed.addItem(f"Error: {err}")
        if "Timeout" in err:
             self.list_installed.addItem("Please Quit Ollama from System Tray.")

    def start_download(self):
        # Determine Model
        model = self.custom_input.text().strip()
        if not model:
            # Check selection
            item = self.list_market.currentItem()
            if item:
                # Extract tag from "Tag (Desc)"
                raw = item.text()
                model = raw.split()[0]
        
        if not model:
            QMessageBox.warning(self, "No Model", "Please select a model or type a tag.")
            return
            
        # Start Thread
        self.worker = OllamaWorker("pull", model)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        
        self.dl_btn.setEnabled(False)
        self.status_bar.setText(f"Pulling {model}...")
        self.progress_bar.setValue(0) # Indeterminate or 0
        self.progress_bar.setRange(0, 0) # Indeterminate for now as parsing % from ollama output is tricky strictly
        
        self.worker.start()

    def on_progress(self, msg):
        # Heuristic parsing for Ollama output
        # Ollama outputs textual progress bars or "writing manifest"
        self.status_bar.setText(msg)
        
        # If msg contains %, try to parse
        if "%" in msg:
            try:
                # Find number before %
                parts = msg.split('%')[0].split()
                num = parts[-1]
                val = int(num)
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(val)
            except:
                pass

    def on_finished(self):
        self.status_bar.setText("Download Complete!")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.dl_btn.setEnabled(True)
        QMessageBox.information(self, "Success", "Model downloaded successfully.")
        self.refresh_installed()

    def on_error(self, err):
        self.status_bar.setText(f"Error: {err}")
        self.dl_btn.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        QMessageBox.critical(self, "Download Failed", f"An error occurred:\n{err}")
