"""
Model Selector Panel - Slide-out panel from left side
Displays available models with names and sizes for selection.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QListWidget, QListWidgetItem, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, QRect
from PySide6.QtGui import QFont
import os


class ModelSelectorPanel(QFrame):
    """
    A slide-out panel that displays available models for selection.
    Slides in from the left side of the main content area.
    """
    model_selected = Signal(dict)  # Emits {display, filename, size} when selected
    panel_closed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ModelSelectorPanel")
        
        # Panel dimensions
        self.panel_width = 320
        self.setFixedWidth(self.panel_width)
        
        # Styling
        self.setStyleSheet("""
            #ModelSelectorPanel {
                background-color: #1E1E1E;
                border-right: 2px solid #006666;
            }
            QLabel#PanelTitle {
                color: #00FFFF;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton#CloseBtn {
                background: transparent;
                color: #888;
                border: none;
                font-size: 18px;
                padding: 5px 10px;
            }
            QPushButton#CloseBtn:hover {
                color: white;
                background-color: #333;
            }
            QListWidget {
                background-color: #252526;
                border: none;
                outline: none;
                font-size: 12px;
            }
            QListWidget::item {
                background-color: #2E2E2E;
                border: 1px solid #3C3C3C;
                border-radius: 4px;
                margin: 2px 6px;
                padding: 6px 8px;
            }
            QListWidget::item:hover {
                background-color: #3E3E3E;
                border-color: #006666;
            }
            QListWidget::item:selected {
                background-color: #006666;
                border-color: #00AAAA;
            }
        """)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QWidget()
        header.setStyleSheet("background-color: #252526; border-bottom: 1px solid #3C3C3C;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 5, 5, 5)
        
        title = QLabel("Select Model")
        title.setObjectName("PanelTitle")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        close_btn = QPushButton("✕")
        close_btn.setObjectName("CloseBtn")
        close_btn.clicked.connect(self.hide_panel)
        header_layout.addWidget(close_btn)
        
        layout.addWidget(header)
        
        # Model list
        self.model_list = QListWidget()
        self.model_list.setSpacing(2)
        self.model_list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.model_list)
        
        # Animation for slide-in/out
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # --- Toggle State (Unity-style) ---
        self._is_open = False
        self._click_state = 0
        
        # Start hidden
        self.hide()
        
    def set_models(self, models):
        """
        Populate the list with models.
        Each model should be a dict with 'display', 'filename', and optionally 'size'.
        """
        self.model_list.clear()
        
        if not models:
            item = QListWidgetItem("No models found")
            item.setFlags(Qt.NoItemFlags)
            self.model_list.addItem(item)
            return
        
        for model in models:
            if isinstance(model, dict):
                display = model.get("display", "Unknown")
                filename = model.get("filename", "")
                
                # Calculate/get size
                size_text = model.get("size", "")
                if not size_text and filename:
                    size_text = self._get_file_size(filename)
                
                # Format display text with size
                if size_text:
                    item_text = f"{display}  [{size_text}]"
                else:
                    item_text = display
                
                # Create simple text item (more reliable than custom widgets)
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, model)
                self.model_list.addItem(item)
            else:
                # Fallback for string entries
                item = QListWidgetItem(str(model))
                item.setData(Qt.UserRole, {"display": str(model), "filename": str(model)})
                self.model_list.addItem(item)
    
    def _create_model_item(self, name, size):
        """Create a widget for a model list item."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Model name
        name_label = QLabel(name)
        name_label.setStyleSheet("color: white; font-size: 13px; font-weight: bold;")
        layout.addWidget(name_label, 1)
        
        # Size badge
        if size:
            size_label = QLabel(size)
            size_label.setStyleSheet("""
                color: #888;
                font-size: 11px;
                background-color: #3C3C3C;
                border-radius: 3px;
                padding: 2px 6px;
            """)
            layout.addWidget(size_label)
        
        return widget
    
    def _get_file_size(self, filepath):
        """Get human-readable file size."""
        try:
            if os.path.exists(filepath):
                size_bytes = os.path.getsize(filepath)
                # Convert to GB/MB
                if size_bytes >= 1024**3:
                    return f"{size_bytes / (1024**3):.1f} GB"
                elif size_bytes >= 1024**2:
                    return f"{size_bytes / (1024**2):.0f} MB"
                else:
                    return f"{size_bytes / 1024:.0f} KB"
        except Exception:
            pass
        return ""
    
    def _on_item_clicked(self, item):
        """Handle model selection."""
        model_data = item.data(Qt.UserRole)
        if model_data:
            self.model_selected.emit(model_data)
            self.hide_panel()
    
    def toggle_panel(self, parent_geometry):
        """
        Toggle panel open/close using Unity-style state machine.
        Call this from the Change Model button.
        """
        self._click_state += 1
        
        # State machine logic
        if self._click_state == 1 and not self._is_open:
            # First click, panel closed -> open it
            self.show_panel(parent_geometry)
        elif self._click_state >= 2 and self._is_open:
            # Second+ click, panel open -> close it
            self.hide_panel()
        
        # Reset state if > 2
        if self._click_state > 2:
            self._click_state = 0
    
    def show_panel(self, parent_geometry):
        """
        Show the panel with slide-in animation.
        parent_geometry: QRect of the parent widget to align to.
        """
        if self._is_open:
            return  # Already open
        
        self._is_open = True
        self._click_state = 1  # Reset to "open" state
        
        # Position: start off-screen to the left, slide in
        start_x = parent_geometry.x() - self.panel_width
        end_x = parent_geometry.x()
        y = parent_geometry.y()
        height = parent_geometry.height()
        
        self.setGeometry(start_x, y, self.panel_width, height)
        self.show()
        self.raise_()
        
        self.animation.setStartValue(QRect(start_x, y, self.panel_width, height))
        self.animation.setEndValue(QRect(end_x, y, self.panel_width, height))
        self.animation.start()
    
    def hide_panel(self):
        """Hide the panel with slide-out animation."""
        if not self._is_open:
            return
        
        self._is_open = False
        self._click_state = 0  # Reset state
        
        current = self.geometry()
        end_x = current.x() - self.panel_width
        
        self.animation.setStartValue(current)
        self.animation.setEndValue(QRect(end_x, current.y(), self.panel_width, current.height()))
        self.animation.finished.connect(self._on_hide_finished)
        self.animation.start()
    
    def _on_hide_finished(self):
        """Called when hide animation finishes."""
        self.hide()
        self.panel_closed.emit()
        # Disconnect to prevent multiple calls
        try:
            self.animation.finished.disconnect(self._on_hide_finished)
        except Exception:
            pass
