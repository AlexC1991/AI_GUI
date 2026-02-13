"""
Model Selector Panel - Slide-out panel with 4 tabs: Local / Cloud / Providers / Favorites
Providers tab uses a tile grid with drill-down into selected models.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFrame, QTabWidget, QStyledItemDelegate,
    QScrollArea, QGridLayout, QLineEdit, QStackedWidget, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, QRect, QSize
from PySide6.QtGui import QFont, QPainter, QColor, QFontMetrics, QPen, QBrush
from PySide6.QtWidgets import QStyle
import os
import re


# -------------------------------------------------------
# Known Model Stats Database
# Maps filename patterns → {params, vram, speed_tps}
# -------------------------------------------------------
MODEL_STATS_DB = {
    # --- Local GGUF models (by param size + quant) ---
    r"0\.5[bB].*Q4":   {"params": "0.5B", "vram": "~0.5 GB",  "speed": "~120 t/s"},
    r"0\.5[bB].*Q8":   {"params": "0.5B", "vram": "~0.8 GB",  "speed": "~90 t/s"},
    r"1[bB][\-_\.].*Q4": {"params": "1B",  "vram": "~1.0 GB",  "speed": "~80 t/s"},
    r"1[bB][\-_\.].*Q8": {"params": "1B",  "vram": "~1.5 GB",  "speed": "~55 t/s"},
    r"3[bB][\-_\.].*Q4": {"params": "3B",  "vram": "~2.5 GB",  "speed": "~45 t/s"},
    r"3[bB][\-_\.].*Q8": {"params": "3B",  "vram": "~4.0 GB",  "speed": "~30 t/s"},
    r"[78][bB][\-_\.].*Q4":  {"params": "7-8B", "vram": "~5.5 GB",  "speed": "~25 t/s"},
    r"[78][bB][\-_\.].*Q8":  {"params": "7-8B", "vram": "~9.0 GB",  "speed": "~15 t/s"},
    r"[78][bB][\-_\.].*IQ4": {"params": "7-8B", "vram": "~5.0 GB",  "speed": "~25 t/s"},
    r"1[234][bB][\-_\.].*Q4": {"params": "12-14B", "vram": "~9 GB",   "speed": "~15 t/s"},
    r"1[234][bB][\-_\.].*Q8": {"params": "12-14B", "vram": "~15 GB",  "speed": "~10 t/s"},
    r"2[789][bB][\-_\.].*Q4": {"params": "27-32B", "vram": "~18 GB",  "speed": "~10 t/s"},
    r"3[012][bB][\-_\.].*Q4": {"params": "27-32B", "vram": "~20 GB",  "speed": "~9 t/s"},
    r"3[012][bB][\-_\.].*Q8": {"params": "27-32B", "vram": "~35 GB",  "speed": "~5 t/s"},
    r"7[012][bB][\-_\.].*Q4": {"params": "70-72B", "vram": "~42 GB",  "speed": "~5 t/s"},
    r"7[012][bB][\-_\.].*Q8": {"params": "70-72B", "vram": "~75 GB",  "speed": "~3 t/s"},
    r"7[012][bB][\-_\.].*IQ4": {"params": "70-72B", "vram": "~38 GB", "speed": "~5 t/s"},
    # --- Cloud models ---
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

# Unified tile accent color (matches app theme)
TILE_ACCENT = "#006666"

# OpenRouter model ID → friendly company name
OPENROUTER_COMPANY_NAMES = {
    "meta-llama": "Meta",
    "google": "Google",
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "mistralai": "Mistral",
    "microsoft": "Microsoft",
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
    "cohere": "Cohere",
    "nvidia": "NVIDIA",
    "perplexity": "Perplexity",
    "nous": "Nous",
    "nousresearch": "Nous",
    "01-ai": "01.AI",
    "databricks": "Databricks",
    "cognitivecomputations": "Cognitive",
    "x-ai": "xAI",
    "liquid": "Liquid",
    "amazon": "Amazon",
    "ai21": "AI21",
}


def lookup_model_stats(model_id: str) -> dict:
    """Look up stats for a model by matching its ID against known patterns."""
    for pattern, stats in MODEL_STATS_DB.items():
        if re.search(pattern, model_id):
            return stats
    return {}


# Custom data roles
ROLE_MODEL_DATA   = Qt.UserRole      # dict with model info
ROLE_STATS        = Qt.UserRole + 1  # dict with stats for display
ROLE_SECTION      = Qt.UserRole + 2  # bool: True = section header
ROLE_IS_FAVORITE  = Qt.UserRole + 3  # bool: is this model favorited


class ModelItemDelegate(QStyledItemDelegate):
    """Custom delegate for rich model list items with stats and star icon."""

    STAR_SIZE = 16
    STAR_MARGIN = 26  # right edge → star center

    def __init__(self, parent=None, show_stars=True):
        super().__init__(parent)
        self._name_font = QFont("Segoe UI", 11, QFont.Bold)
        self._stats_font = QFont("Segoe UI", 9)
        self._section_font = QFont("Segoe UI", 10, QFont.Bold)
        self._star_font = QFont("Segoe UI", 14)
        self._show_stars = show_stars

    def sizeHint(self, option, index):
        if index.data(ROLE_SECTION):
            return QSize(option.rect.width(), 30)
        stats = index.data(ROLE_STATS)
        if stats:
            return QSize(option.rect.width(), 52)
        return QSize(option.rect.width(), 36)

    def paint(self, painter: QPainter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        # Section header (non-clickable, bold teal label)
        if index.data(ROLE_SECTION):
            rect = option.rect.adjusted(12, 4, -6, -2)
            painter.setFont(self._section_font)
            painter.setPen(QColor("#00CCCC"))
            painter.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, index.data(Qt.DisplayRole) or "")
            painter.restore()
            return

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

        # Star icon (right edge)
        star_right_pad = self.STAR_MARGIN if self._show_stars else 6
        content_right = rect.right() - star_right_pad

        if self._show_stars:
            is_fav = index.data(ROLE_IS_FAVORITE)
            star_x = rect.right() - self.STAR_MARGIN
            star_y = rect.center().y()
            painter.setFont(self._star_font)
            if is_fav:
                painter.setPen(QColor("#FFD700"))
                painter.drawText(star_x, star_y + 6, "\u2605")  # filled star
            else:
                painter.setPen(QColor("#555555"))
                painter.drawText(star_x, star_y + 6, "\u2606")  # outline star

        # Model name (top line)
        display = index.data(Qt.DisplayRole) or ""
        painter.setFont(self._name_font)
        painter.setPen(QColor("#E0E0E0"))
        name_rect = rect.adjusted(10, 4, -(star_right_pad + 4), -2)
        painter.drawText(name_rect, Qt.AlignLeft | Qt.AlignTop, display)

        # Stats line (bottom)
        stats = index.data(ROLE_STATS)
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
                stats_text = "  \u00b7  ".join(parts)
                painter.setFont(self._stats_font)
                painter.setPen(QColor("#00CCCC"))
                stats_rect = rect.adjusted(10, 0, -(star_right_pad + 4), -4)
                painter.drawText(stats_rect, Qt.AlignLeft | Qt.AlignBottom, stats_text)

        painter.restore()

    def star_hit(self, pos, item_rect):
        """Check if a click position hits the star region."""
        if not self._show_stars:
            return False
        star_x = item_rect.right() - 6 - self.STAR_MARGIN - 4
        return pos.x() >= star_x


# -------------------------------------------------------
# Provider Tile — clickable card in the grid
# -------------------------------------------------------

class ProviderTile(QFrame):
    """A ~120x90px clickable tile representing one provider."""
    clicked = Signal(str)  # emits provider config key

    def __init__(self, config_key, display_name, model_count, parent=None):
        super().__init__(parent)
        self._config_key = config_key
        self._display_name = display_name
        self._model_count = model_count
        self._accent = QColor(TILE_ACCENT)
        self._hovered = False

        self.setFixedSize(120, 90)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(2, 2, -2, -2)

        # Background
        if self._hovered:
            p.setBrush(QColor("#3E3E3E"))
            p.setPen(QPen(QColor("#006666"), 2))
        else:
            p.setBrush(QColor("#2A2A2A"))
            p.setPen(QPen(QColor("#3C3C3C"), 1))
        p.drawRoundedRect(r, 8, 8)

        # Colored initial circle
        circle_r = 20
        cx = r.center().x()
        cy = r.top() + 28
        p.setBrush(self._accent)
        p.setPen(Qt.NoPen)
        p.drawEllipse(cx - circle_r, cy - circle_r, circle_r * 2, circle_r * 2)

        # Initial letter
        letter = self._display_name[0].upper()
        p.setFont(QFont("Segoe UI", 16, QFont.Bold))
        p.setPen(QColor("#FFFFFF"))
        p.drawText(cx - circle_r, cy - circle_r, circle_r * 2, circle_r * 2,
                   Qt.AlignCenter, letter)

        # Provider name
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.setPen(QColor("#E0E0E0"))
        p.drawText(r.adjusted(4, 0, -4, -14), Qt.AlignBottom | Qt.AlignHCenter,
                   self._display_name)

        # Model count badge
        badge_text = f"{self._model_count} model{'s' if self._model_count != 1 else ''}"
        badge_font = QFont("Segoe UI", 7)
        p.setFont(badge_font)
        fm = QFontMetrics(badge_font)
        tw = fm.horizontalAdvance(badge_text) + 10
        bx = cx - tw // 2
        by = r.bottom() - 14
        p.setBrush(QColor("#006666"))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(bx, by, tw, 14, 7, 7)
        p.setPen(QColor("#FFFFFF"))
        p.drawText(bx, by, tw, 14, Qt.AlignCenter, badge_text)

        p.end()

    def enterEvent(self, event):
        self._hovered = True
        self.update()

    def leaveEvent(self, event):
        self._hovered = False
        self.update()

    def mousePressEvent(self, event):
        self.clicked.emit(self._config_key)


# -------------------------------------------------------
# Provider Tab Widget — grid + drill-down stack
# -------------------------------------------------------

class ProviderTabWidget(QWidget):
    """Composite widget: search bar + stacked grid/drilldown."""
    model_selected = Signal(dict)
    favorite_toggled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._provider_data = {}  # config_key → {display_name, has_key, models:[dict]}
        self._tiles = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search providers or models...")
        self.search_bar.setStyleSheet("""
            QLineEdit {
                background-color: #2E2E2E;
                border: 1px solid #3C3C3C;
                border-radius: 4px;
                color: #E0E0E0;
                padding: 6px 10px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #006666;
            }
        """)
        self.search_bar.textChanged.connect(self._on_search)
        layout.addWidget(self.search_bar)

        # Stacked widget: page 0 = grid, page 1 = drilldown
        self.stack = QStackedWidget()

        # Page 0: Tile grid
        self._grid_page = QWidget()
        grid_outer = QVBoxLayout(self._grid_page)
        grid_outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._grid_container = QWidget()
        self._grid_container.setStyleSheet("background: transparent;")
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setSpacing(8)
        self._grid_layout.setContentsMargins(8, 8, 8, 8)
        scroll.setWidget(self._grid_container)
        grid_outer.addWidget(scroll)

        self.stack.addWidget(self._grid_page)

        # Page 1: Drilldown
        self._drill_page = QWidget()
        drill_layout = QVBoxLayout(self._drill_page)
        drill_layout.setContentsMargins(0, 4, 0, 0)
        drill_layout.setSpacing(4)

        # Back button + provider header
        back_row = QHBoxLayout()
        self._back_btn = QPushButton("\u2190 Back")
        self._back_btn.setStyleSheet("""
            QPushButton {
                background: #2E2E2E;
                color: #00CCCC;
                border: 1px solid #3C3C3C;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #3E3E3E;
                border-color: #006666;
            }
        """)
        self._back_btn.clicked.connect(self._go_back)
        back_row.addWidget(self._back_btn)

        self._drill_header = QLabel("")
        self._drill_header.setStyleSheet("color: #00FFFF; font-size: 14px; font-weight: bold; padding: 0 8px;")
        back_row.addWidget(self._drill_header)
        back_row.addStretch()
        drill_layout.addLayout(back_row)

        # Model list for drilldown
        self._drill_delegate = ModelItemDelegate(show_stars=True)
        self._drill_list = QListWidget()
        self._drill_list.setSpacing(2)
        self._drill_list.setMouseTracking(True)
        self._drill_list.setItemDelegate(self._drill_delegate)
        self._drill_list.itemClicked.connect(self._on_drill_clicked)
        self._drill_list.setStyleSheet("""
            QListWidget {
                background-color: #252526;
                border: none;
                outline: none;
            }
            QListWidget::item {
                background: transparent;
                border: none;
                margin: 0px;
                padding: 0px;
            }
        """)
        drill_layout.addWidget(self._drill_list)

        self.stack.addWidget(self._drill_page)
        layout.addWidget(self.stack)

        self.stack.setCurrentIndex(0)
        self._current_provider = None

    def set_provider_data(self, grouped):
        """Set provider data from grouped dict.
        grouped: {config_key: {display_name, has_key, models:[model_data_dict]}}
        """
        self._provider_data = grouped
        self._rebuild_grid()

    def _rebuild_grid(self, filter_text=""):
        # Clear old tiles
        for t in self._tiles:
            t.setParent(None)
            t.deleteLater()
        self._tiles.clear()

        # Clear grid layout
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        ft = filter_text.lower()
        row, col = 0, 0
        for cfg_key, data in self._provider_data.items():
            display_name = data["display_name"]
            models = data.get("models", [])

            if not models:
                continue

            # Filter: match provider name or any model name
            if ft:
                name_match = ft in display_name.lower()
                model_match = any(ft in m.get("display", "").lower() for m in models)
                if not name_match and not model_match:
                    continue

            tile = ProviderTile(cfg_key, display_name, len(models))
            tile.clicked.connect(self._drill_into)
            self._grid_layout.addWidget(tile, row, col)
            self._tiles.append(tile)
            col += 1
            if col >= 3:
                col = 0
                row += 1

        # Empty state
        if not self._tiles:
            hint = QLabel("No providers configured.\nAdd API keys and models in Settings.")
            hint.setStyleSheet("color: #666; font-size: 12px; padding: 20px;")
            hint.setAlignment(Qt.AlignCenter)
            hint.setWordWrap(True)
            self._grid_layout.addWidget(hint, 0, 0, 1, 3)

        # Spacer at bottom
        self._grid_layout.setRowStretch(row + 1, 1)

    def _drill_into(self, config_key):
        """Switch to drilldown view for a provider."""
        self._current_provider = config_key
        data = self._provider_data.get(config_key, {})
        self._drill_header.setText(data.get("display_name", config_key))
        self._populate_drill_list(data.get("models", []), config_key)
        self.stack.setCurrentIndex(1)
        self.search_bar.clear()

    def _populate_drill_list(self, models, config_key):
        from utils.config_manager import ConfigManager

        self._drill_list.clear()

        for m in models:
            self._add_model_item(self._drill_list, m, ConfigManager)

    def _add_model_item(self, list_widget, model_data, ConfigManager):
        display = model_data.get("display", "Unknown")
        item = QListWidgetItem(display)
        item.setData(ROLE_MODEL_DATA, model_data)
        item.setData(ROLE_STATS, {"size": f"{model_data.get('provider', '')} API"})
        item.setData(ROLE_IS_FAVORITE, ConfigManager.is_favorite(model_data))
        list_widget.addItem(item)

    def _on_drill_clicked(self, item):
        if item.data(ROLE_SECTION):
            return

        # Check for star click
        list_widget = self._drill_list
        item_rect = list_widget.visualItemRect(item)
        cursor_pos = list_widget.mapFromGlobal(list_widget.cursor().pos())
        if self._drill_delegate.star_hit(cursor_pos, item_rect):
            self._toggle_star(item)
            return

        model_data = item.data(ROLE_MODEL_DATA)
        if model_data:
            self.model_selected.emit(model_data)

    def _toggle_star(self, item):
        from utils.config_manager import ConfigManager
        model_data = item.data(ROLE_MODEL_DATA)
        if not model_data:
            return
        added = ConfigManager.toggle_favorite(model_data)
        item.setData(ROLE_IS_FAVORITE, added)
        self._drill_list.update()
        self.favorite_toggled.emit()

    def _go_back(self):
        self.stack.setCurrentIndex(0)
        self._current_provider = None

    def _on_search(self, text):
        if self.stack.currentIndex() == 0:
            # Filter tiles
            self._rebuild_grid(text)
        else:
            # Filter models in drilldown
            ft = text.lower()
            for i in range(self._drill_list.count()):
                item = self._drill_list.item(i)
                if item.data(ROLE_SECTION):
                    # Show section if any child matches (handled by showing all for now)
                    item.setHidden(bool(ft))
                    continue
                display = (item.data(Qt.DisplayRole) or "").lower()
                item.setHidden(ft not in display)

    def reset_view(self):
        """Reset to grid view (called when panel opens)."""
        self.stack.setCurrentIndex(0)
        self.search_bar.clear()
        self._current_provider = None


class ModelSelectorPanel(QFrame):
    """
    A slide-out panel with 4 tabs for model selection.
    Tabs: Local / Cloud / Providers / Favorites
    """
    model_selected = Signal(dict)
    panel_closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ModelSelectorPanel")

        # Panel dimensions
        self.panel_width = 420
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
                padding: 8px 12px;
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

        # Custom delegates
        self._item_delegate = ModelItemDelegate(show_stars=True)
        self._fav_delegate = ModelItemDelegate(show_stars=True)

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

        # Tab 3: Providers (tile grid + drilldown)
        self.provider_tab = ProviderTabWidget()
        self.provider_tab.model_selected.connect(self._on_provider_model_selected)
        self.provider_tab.favorite_toggled.connect(self.refresh_favorites)
        self.tabs.addTab(self.provider_tab, "Providers")

        # Tab 4: Favorites
        self.fav_list = QListWidget()
        self.fav_list.setSpacing(2)
        self.fav_list.setMouseTracking(True)
        self.fav_list.setItemDelegate(self._fav_delegate)
        self.fav_list.itemClicked.connect(self._on_fav_clicked)
        self.tabs.addTab(self.fav_list, "\u2605 Favorites")

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
        """Populate Local tab with models from VoxModelWorker."""
        self.local_list.clear()

        if not models:
            item = QListWidgetItem("No local models found")
            item.setFlags(Qt.NoItemFlags)
            self.local_list.addItem(item)
            return

        from utils.config_manager import ConfigManager

        for model in models:
            if isinstance(model, dict):
                display = model.get("display", "Unknown")
                filename = model.get("filename", "")
                size_text = model.get("size", "")
                if not size_text and filename:
                    size_text = self._get_file_size(filename)

                model_data = {**model, "mode": "local"}
                item = QListWidgetItem(display)
                item.setData(ROLE_MODEL_DATA, model_data)

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
                item.setData(ROLE_STATS, stats_display if stats_display else None)
                item.setData(ROLE_IS_FAVORITE, ConfigManager.is_favorite(model_data))

                self.local_list.addItem(item)

    def set_cloud_models(self, models_dict):
        """Populate Cloud tab from config cloud models."""
        self.cloud_list.clear()

        if not models_dict:
            item = QListWidgetItem("No cloud models configured.\nAdd models in Settings.")
            item.setFlags(Qt.NoItemFlags)
            self.cloud_list.addItem(item)
            return

        from utils.config_manager import ConfigManager

        for hf_id, display_name in models_dict.items():
            model_data = {
                "mode": "cloud",
                "display": display_name,
                "cloud_id": hf_id,
            }
            item = QListWidgetItem(display_name)
            item.setData(ROLE_MODEL_DATA, model_data)

            stats = lookup_model_stats(hf_id)
            stats_display = {}
            if stats.get("params"):
                stats_display["params"] = stats["params"]
            if stats.get("vram"):
                stats_display["vram"] = stats["vram"]
            if stats.get("speed"):
                stats_display["speed"] = stats["speed"]
            stats_display["size"] = "RunPod GPU"
            item.setData(ROLE_STATS, stats_display if stats_display else None)
            item.setData(ROLE_IS_FAVORITE, ConfigManager.is_favorite(model_data))

            self.cloud_list.addItem(item)

    def set_provider_models(self, grouped_dict):
        """Populate Providers tab with grouped provider data.
        grouped_dict: {config_key: {display_name, has_key, models:[model_data_dict]}}
        """
        self.provider_tab.set_provider_data(grouped_dict)

    def refresh_favorites(self):
        """Reload favorites tab from config."""
        from utils.config_manager import ConfigManager
        self.fav_list.clear()
        favs = ConfigManager.get_favorites()

        if not favs:
            item = QListWidgetItem("No favorites yet.\nStar a model to add it here.")
            item.setFlags(Qt.NoItemFlags)
            self.fav_list.addItem(item)
            return

        for fav in favs:
            display = fav.get("display", "Unknown")
            provider = fav.get("provider", fav.get("mode", ""))
            label = f"{display}  [{provider}]" if provider else display

            item = QListWidgetItem(label)
            item.setData(ROLE_MODEL_DATA, fav)
            item.setData(ROLE_STATS, {"size": f"{provider} API" if provider else ""})
            item.setData(ROLE_IS_FAVORITE, True)  # always gold in favorites tab
            self.fav_list.addItem(item)

    # For backward compatibility
    def set_models(self, models):
        self.set_local_models(models)

    def set_active_tab(self, mode):
        tab_map = {"local": 0, "cloud": 1, "provider": 2, "favorites": 3}
        idx = tab_map.get(mode, 0)
        self.tabs.setCurrentIndex(idx)

    # -------------------------------------------------------
    # Click Handlers
    # -------------------------------------------------------

    def _on_local_clicked(self, item):
        if self._handle_star_click(self.local_list, item, self._item_delegate):
            return
        model_data = item.data(ROLE_MODEL_DATA)
        if model_data:
            model_data["mode"] = "local"
            self.model_selected.emit(model_data)
            self.hide_panel()

    def _on_cloud_clicked(self, item):
        if self._handle_star_click(self.cloud_list, item, self._item_delegate):
            return
        model_data = item.data(ROLE_MODEL_DATA)
        if model_data:
            self.model_selected.emit(model_data)
            self.hide_panel()

    def _on_provider_model_selected(self, model_data):
        self.model_selected.emit(model_data)
        self.hide_panel()

    def _on_fav_clicked(self, item):
        if self._handle_star_click(self.fav_list, item, self._fav_delegate):
            return
        model_data = item.data(ROLE_MODEL_DATA)
        if model_data:
            self.model_selected.emit(model_data)
            self.hide_panel()

    def _handle_star_click(self, list_widget, item, delegate):
        """Check if click was on star, toggle if so. Returns True if handled."""
        item_rect = list_widget.visualItemRect(item)
        cursor_pos = list_widget.mapFromGlobal(list_widget.cursor().pos())
        if delegate.star_hit(cursor_pos, item_rect):
            from utils.config_manager import ConfigManager
            model_data = item.data(ROLE_MODEL_DATA)
            if model_data:
                added = ConfigManager.toggle_favorite(model_data)
                item.setData(ROLE_IS_FAVORITE, added)
                list_widget.update()
                self.refresh_favorites()
            return True
        return False

    # -------------------------------------------------------
    # Helpers
    # -------------------------------------------------------

    def _get_file_size(self, filepath):
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
        self._click_state += 1

        if self._click_state == 1 and not self._is_open:
            self.show_panel(parent_geometry)
        elif self._click_state >= 2 and self._is_open:
            self.hide_panel()

        if self._click_state > 2:
            self._click_state = 0

    def show_panel(self, parent_geometry):
        if self._is_open:
            return

        self._is_open = True
        self._click_state = 1

        # Reset provider tab to grid view
        self.provider_tab.reset_view()
        self.refresh_favorites()

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
        self.hide()
        self.panel_closed.emit()
        try:
            self.animation.finished.disconnect(self._on_hide_finished)
        except Exception:
            pass
