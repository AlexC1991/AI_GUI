from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QTextBrowser, QFrame, QVBoxLayout,
    QLabel, QSizePolicy, QStyle
)
from PySide6.QtCore import Qt, QTimer, QSize
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
        for dot in self.dots:
            dot.setStyleSheet("font-size: 14px; color: #666;")
        self.dots[self.current_dot].setStyleSheet("font-size: 14px; color: #FFF;")
        self.current_dot = (self.current_dot + 1) % 3