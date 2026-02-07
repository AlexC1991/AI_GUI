from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QTextBrowser, QFrame, QVBoxLayout,
    QLabel, QSizePolicy, QStyle, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QTextCursor
import markdown
from datetime import datetime

# Import Pygments to generate the "Discord-style" colors
from pygments.formatters import HtmlFormatter

# ===================================================================
# 1. CUSTOM TEXT BROWSER (Smart Sizing)
# ===================================================================
class AutoResizingTextBrowser(QTextBrowser):
    def __init__(self):
        super().__init__()
        self.setOpenExternalLinks(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setReadOnly(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("background: transparent; border: none;")
        self.setViewportMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def update_size(self):
        doc = self.document()
        layout = doc.documentLayout()

        # 1. Measure Natural Width
        doc.setTextWidth(-1)
        ideal_width = doc.idealWidth()

        # 2. YOUR LIMITS
        max_w = 500
        min_w = 10

        padding = 30

        final_width = ideal_width + padding

        if final_width > max_w:
            final_width = max_w
        if final_width < min_w:
            final_width = min_w

        # 3. Check Height
        doc.setTextWidth(final_width)
        height = layout.documentSize().height()
        max_h = 600

        # 4. Scrollbar Logic
        if height > max_h:
            scrollbar_width = self.style().pixelMetric(QStyle.PM_ScrollBarExtent)
            final_width += scrollbar_width

            # Re-constrain if adding scrollbar pushed us over
            if final_width > max_w + scrollbar_width:
                final_width = max_w + scrollbar_width

            doc.setTextWidth(final_width)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        else:
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        final_height = min(height, max_h)
        self.setFixedSize(int(final_width), int(final_height))

# ===================================================================
# 2. MESSAGE BUBBLE
# ===================================================================
class MessageBubble(QWidget):
    def __init__(self, text, sender="user"):
        super().__init__()

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.bubble_frame = QFrame()
        self.bubble_frame.setObjectName("BubbleFrame")

        if sender == "user":
            bg_color = "#008080"
            text_color = "white"
            border_radius = "20px 20px 0px 20px"
            border_color = "#B2D8D8"
            self.code_bg = "#004444"
            timestamp_align = Qt.AlignRight
        else:
            bg_color = "#333333"
            text_color = "#E0E0E0"
            border_radius = "20px 20px 20px 0px"
            border_color = "#666666"
            self.code_bg = "#1E1E1E"
            timestamp_align = Qt.AlignLeft

        self.bubble_frame.setStyleSheet(f"""
            #BubbleFrame {{
                background-color: {bg_color};
                border-radius: 20px;
                border: 1.5px solid {border_color};
            }}
            QTextBrowser {{
                color: {text_color};
                font-family: Segoe UI, sans-serif;
                font-size: 14px;
            }}
        """)

        bubble_inner = QVBoxLayout(self.bubble_frame)
        bubble_inner.setContentsMargins(15, 12, 15, 12)

        self.text_browser = AutoResizingTextBrowser()
        self.set_content(text)
        bubble_inner.addWidget(self.text_browser)

        # TIMESTAMP
        current_time = datetime.now().strftime("%I:%M %p")
        self.time_label = QLabel(current_time)
        self.time_label.setAlignment(timestamp_align)
        self.time_label.setStyleSheet(f"""
            color: {text_color};
            font-size: 11px;
            font-weight: bold;
            background: transparent;
            border: none;
            margin-top: 4px;
            margin-left: 10px;
            margin-right: 10px;
        """)

        container_widget = QWidget()
        container_layout = QVBoxLayout(container_widget)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        container_layout.addWidget(self.bubble_frame)
        container_layout.addWidget(self.time_label)

        if sender == "user":
            self.layout.addStretch()
            self.layout.addWidget(container_widget)
        else:
            self.layout.addWidget(container_widget)
            self.layout.addStretch()

    def set_content(self, text):
        if not text:
            self.text_browser.setText("")
            return

        # 1. SETUP MARKDOWN
        # - fenced_code: Enables ``` blocks
        # - nl2br: Makes "Enter" key create a new line
        # - codehilite: Uses Pygments to colorize ONLY inside ``` blocks
        md = markdown.Markdown(extensions=[
            'fenced_code',
            'nl2br',
            'codehilite'
        ], extension_configs={
            'codehilite': {
                'css_class': 'highlight', # The CSS class for the wrapper
                'use_pygments': True,
                'guess_lang': True        # Tries to guess language if you miss it
            }
        })
        md.parser.blockprocessors.deregister('hashheader')

        html_content = md.convert(text)

        # 2. GENERATE PYGMENTS CSS (Monokai Theme - Dark Mode IDE Style)
        # This gets the definitions for .k (keyword), .s (string), etc.
        formatter = HtmlFormatter(style='monokai', nobackground=True)
        pygments_css = formatter.get_style_defs('.highlight')

        # 3. INJECT CSS
        styled_html = f"""
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; font-size: 14px; margin: 0; padding: 0; }}
            p {{ margin: 0; padding: 0; margin-bottom: 6px; }}
            
            /* --- INLINE CODE (`like this`) --- */
            code {{
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                background-color: rgba(255, 255, 255, 0.1); /* Subtle background */
                padding: 2px 4px;
                border-radius: 4px;
            }}

            /* --- CODE BLOCKS (```like this```) --- */
            /* The 'highlight' class is generated by Pygments around fenced blocks */
            .highlight {{
                background-color: {self.code_bg};
                padding: 10px;
                margin: 8px 0;
                border-radius: 6px; 
                border: 1px solid #444;
                overflow: hidden;
            }}
            
            /* The text inside the block */
            .highlight pre {{
                margin: 0;
                line-height: 125%;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                color: #f8f8f2;
                white-space: pre-wrap; 
            }}
            
            /* Overwrite the inline style for blocks so we don't double-box */
            .highlight code {{
                background-color: transparent;
                padding: 0;
                border-radius: 0;
            }}

            a {{ color: #4EC9B0; text-decoration: none; font-weight: bold; }}

            /* --- SYNTAX HIGHLIGHTING COLORS --- */
            {pygments_css}
        </style>
        {html_content}
        """
        self.text_browser.setHtml(styled_html)
        self.text_browser.update_size()

# (ThinkingBubble remains unchanged)
class ThinkingBubble(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 5, 0, 5)

        self.bubble_frame = QFrame()
        self.bubble_frame.setObjectName("ThinkingFrame")
        self.bubble_frame.setStyleSheet("""
            #ThinkingFrame {
                background-color: #333333;
                border-radius: 20px;
                border-bottom-left-radius: 0px; 
                border: 1.5px solid #006666;
            }
        """)

        dots_layout = QHBoxLayout(self.bubble_frame)
        dots_layout.setContentsMargins(15, 10, 15, 10)
        dots_layout.setSpacing(5)

        self.dots = []
        for _ in range(3):
            dot = QLabel("●")
            dot.setStyleSheet("font-size: 10px; color: #666;") 
            dots_layout.addWidget(dot)
            self.dots.append(dot)

        self.layout.addWidget(self.bubble_frame)
        self.layout.addStretch() 

        self.current_dot = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate_dots)
        self.timer.start(300) 

    def animate_dots(self):
        # Reset all dots to gray
        for dot in self.dots:
            dot.setStyleSheet("font-size: 14px; color: #666;") 
        
        # Highlight the current dot (White)
        self.dots[self.current_dot].setStyleSheet("font-size: 14px; color: #FFF;") 
        
        # Move to next dot
        self.current_dot = (self.current_dot + 1) % 3


# ===================================================================
# 4. THINKING SECTION (Compact collapsible reasoning display)
# ===================================================================
class ThinkingSection(QWidget):
    """Compact collapsible section that shows AI reasoning/thinking text.
    Left-aligned and small - doesn't take up the full chat width.
    Auto-hides after the answer is delivered."""

    def __init__(self):
        super().__init__()
        self._expanded = False
        self._thinking_text = ""
        self._finalized = False

        # Outer layout to push content left (like AI bubbles)
        outer_layout = QHBoxLayout(self)
        outer_layout.setContentsMargins(0, 2, 0, 2)
        outer_layout.setSpacing(0)

        # Inner container with fixed max width
        self._container = QWidget()
        self._container.setMaximumWidth(260)
        self._container.setStyleSheet("background: transparent;")

        inner_layout = QVBoxLayout(self._container)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(0)

        # --- Header (clickable) ---
        self.header = QFrame()
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setFixedHeight(28)
        self._set_header_collapsed_style()

        header_inner = QHBoxLayout(self.header)
        header_inner.setContentsMargins(10, 0, 10, 0)
        header_inner.setSpacing(4)

        self.arrow_label = QLabel("\u25B6")
        self.arrow_label.setStyleSheet("color: #666; font-size: 8px; background: transparent; border: none;")
        header_inner.addWidget(self.arrow_label)

        self.title_label = QLabel("Thinking")
        self.title_label.setStyleSheet(
            "color: #888; font-size: 11px; font-style: italic; "
            "background: transparent; border: none;"
        )
        header_inner.addWidget(self.title_label)

        # Animated dots
        self.dots_label = QLabel("")
        self.dots_label.setStyleSheet("color: #555; font-size: 11px; background: transparent; border: none;")
        header_inner.addWidget(self.dots_label)

        header_inner.addStretch()
        self.header.mousePressEvent = self._toggle

        inner_layout.addWidget(self.header)

        # --- Content (hidden by default) ---
        self.content_frame = QFrame()
        self.content_frame.setMaximumWidth(400)
        self.content_frame.setStyleSheet("""
            QFrame {
                background-color: #1E1E1E;
                border: 1px solid #333;
                border-top: none;
                border-radius: 0px 0px 8px 8px;
            }
        """)
        self.content_frame.setVisible(False)

        content_inner = QVBoxLayout(self.content_frame)
        content_inner.setContentsMargins(10, 6, 10, 6)

        self.text_display = QTextBrowser()
        self.text_display.setReadOnly(True)
        self.text_display.setFrameShape(QFrame.NoFrame)
        self.text_display.setMaximumHeight(250)
        self.text_display.setStyleSheet("""
            QTextBrowser {
                background: transparent;
                color: #777;
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
                border: none;
            }
        """)
        content_inner.addWidget(self.text_display)

        inner_layout.addWidget(self.content_frame)

        # Push left
        outer_layout.addWidget(self._container)
        outer_layout.addStretch()

        # Dot animation timer
        self._dot_count = 0
        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._animate_dots)
        self._dot_timer.start(400)

    def _set_header_collapsed_style(self):
        self.header.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
            }
            QFrame:hover { background-color: #2e2e2e; }
        """)

    def _set_header_expanded_style(self):
        self.header.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border: 1px solid #3a3a3a;
                border-radius: 8px 8px 0px 0px;
            }
            QFrame:hover { background-color: #2e2e2e; }
        """)

    def _animate_dots(self):
        if self._finalized:
            self.dots_label.setText("")
            self._dot_timer.stop()
            return
        self._dot_count = (self._dot_count + 1) % 4
        self.dots_label.setText("." * self._dot_count)

    def _toggle(self, event=None):
        self._expanded = not self._expanded
        self.content_frame.setVisible(self._expanded)
        self.arrow_label.setText("\u25BC" if self._expanded else "\u25B6")
        if self._expanded:
            self._set_header_expanded_style()
        else:
            self._set_header_collapsed_style()

    def append_text(self, chunk):
        """Append streaming thinking text."""
        self._thinking_text += chunk
        self.text_display.setPlainText(self._thinking_text)
        cursor = self.text_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.text_display.setTextCursor(cursor)

    def toPlainText(self):
        """Return the raw thinking text."""
        return self._thinking_text

    def finalize(self):
        """Called when thinking is done. Stops animation, shrinks to summary."""
        self._finalized = True
        self.dots_label.setText("")
        self._dot_timer.stop()

        # Count lines for summary
        lines = len([l for l in self._thinking_text.split('\n') if l.strip()])
        self.title_label.setText(f"Thought for {lines} steps")
        self.title_label.setStyleSheet(
            "color: #666; font-size: 11px; background: transparent; border: none;"
        )

        # Collapse if expanded
        if self._expanded:
            self._toggle()