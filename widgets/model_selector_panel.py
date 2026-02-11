"""
Model Selector Panel - Slide-out panel with 3 tabs: Local / Cloud / Providers
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFrame, QTabWidget, QStyledItemDelegate
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, QRect, QSize
from PySide6.QtGui import QFont, QPainter, QColor, QFontMetrics, QPen
from PySide6.QtWidgets import QStyle
import os
import re


# -------------------------------------------------------
# Known Model Stats Database
# Maps filename patterns → {params, vram, speed_tps}
# speed_tps = approximate tokens/sec on typical hardware
# vram = approximate VRAM in GB needed to run
# -------------------------------------------------------
MODEL_STATS_DB = {
    # --- Local GGUF models (by param size + quant) ---
    # Tiny (0.5B-1B)
    r"0\.5[bB].*Q4":   {"params": "0.5B", "vram": "~0.5 GB",  "speed": "~120 t/s"},
    r"0\.5[bB].*Q8":   {"params": "0.5B", "vram": "~0.8 GB",  "speed": "~90 t/s"},
    r"1[bB][\-_\.].*Q4": {"params": "1B",  "vram": "~1.0 GB",  "speed": "~80 t/s"},
    r"1[bB][\-_\.].*Q8": {"params": "1B",  "vram": "~1.5 GB",  "speed": "~55 t/s"},
    # Small (3B)
    r"3[bB][\-_\.].*Q4": {"params": "3B",  "vram": "~2.5 GB",  "speed": "~45 t/s"},
    r"3[bB][\-_\.].*Q8": {"params": "3B",  "vram": "~4.0 GB",  "speed": "~30 t/s"},
    # Medium (7-8B)
    r"[78][bB][\-_\.].*Q4":  {"params": "7-8B", "vram": "~5.5 GB",  "speed": "~25 t/s"},
    r"[78][bB][\-_\.].*Q8":  {"params": "7-8B", "vram": "~9.0 GB",  "speed": "~15 t/s"},
    r"[78][bB][\-_\.].*IQ4": {"params": "7-8B", "vram": "~5.0 GB",  "speed": "~25 t/s"},
    # Large (12-14B)
    r"1[234][bB][\-_\.].*Q4": {"params": "12-14B", "vram": "~9 GB",   "speed": "~15 t/s"},
    r"1[234][bB][\-_\.].*Q8": {"params": "12-14B", "vram": "~15 GB",  "speed": "~10 t/s"},
    # XL (27-32B)
    r"2[789][bB][\-_\.].*Q4": {"params": "27-32B", "vram": "~18 GB",  "speed": "~10 t/s"},
    r"3[012][bB][\-_\.].*Q4": {"params": "27-32B", "vram": "~20 GB",  "speed": "~9 t/s"},
    r"3[012][bB][\-_\.].*Q8": {"params": "27-32B", "vram": "~35 GB",  "speed": "~5 t/s"},
    # XXL (70-72B)
    r"7[012][bB][\-_\.].*Q4": {"params": "70-72B", "vram": "~42 GB",  "speed": "~5 t/s"},
    r"7[012][bB][\-_\.].*Q8": {"params": "70-72B", "vram": "~75 GB",  "speed": "~3 t/s"},
    r"7[012][bB][\-_\.].*IQ4": {"params": "70-72B", "vram": "~38 GB", "speed": "~5 t/s"},

    # --- Cloud models (by HF model ID patterns) ---
    r"(?i)qwen.*0\.5[bB]":   {"params": "0.5B", "vram": "~1 GB",   "speed": "~200 t/s"},
    r"(?i)qwen.*1\.5[bB]":   {"params": "1.5B", "vram": "~3 GB",   "speed": "~150 t/s"},
    r"(?i)qwen.*3[bB]":      {"params": "3B",   "vram": "~6 GB",   "speed": "~120 t/s"},
    r"(?i)qwen.*7[bB]":      {"params": "7B",   "vram": "~14 GB",  "speed": "~80 t/s"},
    r"(?i)qwen.*14[bB]":     {"params": "14B",  "vram": "~28 GB",  "speed": "~45 t/s"},
    r"(?i)qwen.*32[bB]":     {"params": "32B",  "vram": "~38 GB",  "speed": "~25 t/s"},
    r"(?i)qwen.*72[bB]":     {"params": "72B",  "vram": "~48 GB",  "speed": "~15 t/s"},
    r"(?i)llama.*3[bB]":     {"params": "3B",   "vram": "~6 GB",   "speed": "~120 t/s"},
    r"(?i)llama.*8[bB]":     {"params": "8B",   "vram": "~16 GB",  "speed": "~70 t/s"},
    r"(?i)llama.*70[bB]":    {"params": "70B",  "vram": "~48 GB",  "speed": "~15 t/s"},
    r"(?i)mistral.*7[bB]":   {"params": "7B",   "vram": "~14 GB",  "speed": "~80 t/s"},
    r"(?i)mixtral.*8x7":     {"params": "46.7B","vram": "~48 GB",  "speed": "~25 t/s"},
    r"(?i)gemma.*2[bB]":     {"params": "2B",   "vram": "~4 GB",   "speed": "~140 t/s"},
    r"(?i)gemma.*9[bB]":     {"params": "9B",   "vram": "~18 GB",  "speed": "~60 t/s"},
    r"(?i)gemma.*27[bB]":    {"params": "27B",  "vram": "~36 GB",  "speed": "~20 t/s"},
    r"(?i)phi.*3.*mini":     {"params": "3.8B", "vram": "~8 GB",   "speed": "~100 t/s"},
    r"(?i)phi.*3.*medium":   {"params": "14B",  "vram": "~28 GB",  "speed": "~45 t/s"},
    r"(?i)deepseek.*7[bB]":  {"params": "7B",   "vram": "~14 GB",  "speed": "~80 t/s"},
    r"(?i)deepseek.*67[bB]": {"params": "67B",  "vram": "~48 GB",  "speed": "~15 t/s"},
}


def lookup_model_stats(model_id: str) -> dict:
    """Look up stats for a model by matching its ID against known patterns."""
    for pattern, stats in MODEL_STATS_DB.items():
        if re.search(pattern, model_id):
            return stats
    return {}


class ModelItemDelegate(QStyledItemDelegate):
    """Custom delegate for rich model list items with stats."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name_font = QFont("Segoe UI", 11, QFont.Bold)
        self._stats_font = QFont("Segoe UI", 9)

    def sizeHint(self, option, index):
        stats = index.data(Qt.UserRole + 1)  # Stats dict
        if stats:
            return QSize(option.rect.width(), 52)
        return QSize(option.rect.width(), 36)

    def paint(self, painter: QPainter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        rect = option.rect.adjusted(6, 2, -6, -2)
        is_selected = bool(option.state & QStyle.State_Selected)
        is_hover = bool(option.state & QStyle.State_MouseOver)
        if is_selected:
            painter.setBrush(QColor("#006666"))
            painter.setPen(QPen(QColor("#00AAAA"), 1))
        elif is_hover:
            painter.setBrush(QColor("#3E3E3E"))
            painter.setPen(QPen(QColor("#006666"), 1))
        else:
            painter.setBrush(QColor("#2E2E2E"))
            painter.setPen(QPen(QColor("#3C3C3C"), 1))
        painter.drawRoundedRect(rect, 4, 4)

        # Model name (top line)
        display = index.data(Qt.DisplayRole) or ""
        painter.setFont(self._name_font)
        painter.setPen(QColor("#E0E0E0"))
        name_rect = rect.adjusted(10, 4, -10, -2)
        painter.drawText(name_rect, Qt.AlignLeft | Qt.AlignTop, display)

        # Stats line (bottom)
        stats = index.data(Qt.UserRole + 1)
        if stats:
            parts = []
            if stats.get("size"):
                parts.append(stats["size"])
            if stats.get("params"):
                parts.append(stats["params"])
            if stats.get("vram"):
                parts.append(f"VRAM {stats['vram']}")
            if stats.get("speed"):
                parts.append(stats["speed"])

            if parts:
                stats_text = "  ·  ".join(parts)
                painter.setFont(self._stats_font)
                painter.setPen(QColor("#00CCCC"))
                stats_rect = rect.adjusted(10, 0, -10, -4)
                painter.drawText(stats_rect, Qt.AlignLeft | Qt.AlignBottom, stats_text)

        painter.restore()


class ModelSelectorPanel(QFrame):
    """
    A slide-out panel with 3 tabs for model selection.
    Tabs: Local / Cloud / Providers
    """
    model_selected = Signal(dict)  # Emits model_data dict when selected
    panel_closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ModelSelectorPanel")

        # Panel dimensions
        self.panel_width = 340
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
            QTabWidget::pane {
                border: none;
                background-color: #252526;
            }
            QTabBar::tab {
                background-color: #2E2E2E;
                color: #888;
                border: 1px solid #3C3C3C;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background-color: #006666;
                color: white;
                border-color: #008080;
            }
            QTabBar::tab:hover:!selected {
                background-color: #3E3E3E;
                color: #CCC;
            }
            QListWidget {
                background-color: #252526;
                border: none;
                outline: none;
                font-size: 12px;
            }
            QListWidget::item {
                background: transparent;
                border: none;
                margin: 0px;
                padding: 0px;
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

        close_btn = QPushButton("\u2715")
        close_btn.setObjectName("CloseBtn")
        close_btn.clicked.connect(self.hide_panel)
        header_layout.addWidget(close_btn)

        layout.addWidget(header)

        # --- Tab Widget ---
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # Custom delegate for rich item rendering
        self._item_delegate = ModelItemDelegate()

        # Tab 1: Local
        self.local_list = QListWidget()
        self.local_list.setSpacing(2)
        self.local_list.setMouseTracking(True)
        self.local_list.setItemDelegate(self._item_delegate)
        self.local_list.itemClicked.connect(self._on_local_clicked)
        self.tabs.addTab(self.local_list, "Local")

        # Tab 2: Cloud
        self.cloud_list = QListWidget()
        self.cloud_list.setSpacing(2)
        self.cloud_list.setMouseTracking(True)
        self.cloud_list.setItemDelegate(self._item_delegate)
        self.cloud_list.itemClicked.connect(self._on_cloud_clicked)
        self.tabs.addTab(self.cloud_list, "Cloud")

        # Tab 3: Providers
        self.provider_list = QListWidget()
        self.provider_list.setSpacing(2)
        self.provider_list.setMouseTracking(True)
        self.provider_list.setItemDelegate(self._item_delegate)
        self.provider_list.itemClicked.connect(self._on_provider_clicked)
        self.tabs.addTab(self.provider_list, "Providers")

        layout.addWidget(self.tabs)

        # Animation for slide-in/out
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)

        # Toggle State
        self._is_open = False
        self._click_state = 0

        # Start hidden
        self.hide()

    # -------------------------------------------------------
    # Population Methods
    # -------------------------------------------------------

    def set_local_models(self, models):
        """Populate Local tab with models from VoxModelWorker.
        Each model: dict with 'display', 'filename', optionally 'size'.
        """
        self.local_list.clear()

        if not models:
            item = QListWidgetItem("No local models found")
            item.setFlags(Qt.NoItemFlags)
            self.local_list.addItem(item)
            return

        for model in models:
            if isinstance(model, dict):
                display = model.get("display", "Unknown")
                filename = model.get("filename", "")
                size_text = model.get("size", "")
                if not size_text and filename:
                    size_text = self._get_file_size(filename)

                item = QListWidgetItem(display)
                item.setData(Qt.UserRole, model)

                # Build stats dict for delegate
                stats = lookup_model_stats(filename)
                stats_display = {}
                if size_text:
                    stats_display["size"] = size_text
                if stats.get("params"):
                    stats_display["params"] = stats["params"]
                if stats.get("vram"):
                    stats_display["vram"] = stats["vram"]
                if stats.get("speed"):
                    stats_display["speed"] = stats["speed"]
                item.setData(Qt.UserRole + 1, stats_display if stats_display else None)

                self.local_list.addItem(item)

    def set_cloud_models(self, models_dict):
        """Populate Cloud tab from config cloud models.
        models_dict: {hf_model_id: display_name, ...}
        """
        self.cloud_list.clear()

        if not models_dict:
            item = QListWidgetItem("No cloud models configured.\nAdd models in Settings.")
            item.setFlags(Qt.NoItemFlags)
            self.cloud_list.addItem(item)
            return

        for hf_id, display_name in models_dict.items():
            item = QListWidgetItem(display_name)
            item.setData(Qt.UserRole, {
                "mode": "cloud",
                "display": display_name,
                "cloud_id": hf_id,
            })

            # Look up stats from HF model ID
            stats = lookup_model_stats(hf_id)
            stats_display = {}
            if stats.get("params"):
                stats_display["params"] = stats["params"]
            if stats.get("vram"):
                stats_display["vram"] = stats["vram"]
            if stats.get("speed"):
                stats_display["speed"] = stats["speed"]
            # Always show "RunPod" tag for cloud
            stats_display["size"] = "RunPod GPU"
            item.setData(Qt.UserRole + 1, stats_display if stats_display else None)

            self.cloud_list.addItem(item)

    def set_provider_models(self, models_list):
        """Populate Providers tab with API model names.
        models_list: list of dicts {"name": str, "provider": str}
                     OR list of plain strings (backward compat → Gemini)
        """
        self.provider_list.clear()

        if not models_list:
            item = QListWidgetItem("No API key configured.\nAdd key in Settings.")
            item.setFlags(Qt.NoItemFlags)
            self.provider_list.addItem(item)
            return

        for entry in models_list:
            # Support both dict format and legacy string format
            if isinstance(entry, dict):
                name = entry["name"]
                provider = entry.get("provider", "Gemini")
            else:
                name = entry
                provider = "Gemini"

            display = f"{name}  [{provider}]"
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, {
                "mode": "provider",
                "display": name,
                "filename": name,
                "provider": provider,
            })
            # Provider models: show provider badge as stats
            item.setData(Qt.UserRole + 1, {"size": f"{provider} API"})
            self.provider_list.addItem(item)

    # For backward compatibility with old MainWindow calls
    def set_models(self, models):
        """Legacy method - routes to set_local_models."""
        self.set_local_models(models)

    def set_active_tab(self, mode):
        """Switch to the tab matching the given execution mode."""
        tab_map = {"local": 0, "cloud": 1, "provider": 2}
        idx = tab_map.get(mode, 0)
        self.tabs.setCurrentIndex(idx)

    # -------------------------------------------------------
    # Click Handlers
    # -------------------------------------------------------

    def _on_local_clicked(self, item):
        model_data = item.data(Qt.UserRole)
        if model_data:
            model_data["mode"] = "local"
            self.model_selected.emit(model_data)
            self.hide_panel()

    def _on_cloud_clicked(self, item):
        model_data = item.data(Qt.UserRole)
        if model_data:
            self.model_selected.emit(model_data)
            self.hide_panel()

    def _on_provider_clicked(self, item):
        model_data = item.data(Qt.UserRole)
        if model_data:
            self.model_selected.emit(model_data)
            self.hide_panel()

    # -------------------------------------------------------
    # Helpers
    # -------------------------------------------------------

    def _get_file_size(self, filepath):
        """Get human-readable file size."""
        try:
            if os.path.exists(filepath):
                size_bytes = os.path.getsize(filepath)
                if size_bytes >= 1024**3:
                    return f"{size_bytes / (1024**3):.1f} GB"
                elif size_bytes >= 1024**2:
                    return f"{size_bytes / (1024**2):.0f} MB"
                else:
                    return f"{size_bytes / 1024:.0f} KB"
        except Exception:
            pass
        return ""

    # -------------------------------------------------------
    # Panel Animation (slide in/out)
    # -------------------------------------------------------

    def toggle_panel(self, parent_geometry):
        """Toggle panel open/close."""
        self._click_state += 1

        if self._click_state == 1 and not self._is_open:
            self.show_panel(parent_geometry)
        elif self._click_state >= 2 and self._is_open:
            self.hide_panel()

        if self._click_state > 2:
            self._click_state = 0

    def show_panel(self, parent_geometry):
        """Show the panel with slide-in animation."""
        if self._is_open:
            return

        self._is_open = True
        self._click_state = 1

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
        self._click_state = 0

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
        try:
            self.animation.finished.disconnect(self._on_hide_finished)
        except Exception:
            pass
