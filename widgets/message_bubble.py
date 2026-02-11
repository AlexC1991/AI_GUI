from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QTextBrowser, QFrame, QVBoxLayout,
    QLabel, QSizePolicy, QStyle, QScrollArea, QPushButton
)
from PySide6.QtCore import Qt, QTimer, QSize, Signal
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
        plain = doc.toPlainText().strip()

        from PySide6.QtGui import QFontMetrics, QFont
        fm = QFontMetrics(QFont("Segoe UI", 14))

        max_w = 500
        min_w = 60

        # --- Short message fast path (single-line, no code blocks) ---
        # For tiny messages like "Hey", "Ok", etc. the HTML <style> block
        # inflates idealWidth(). Use pure font metrics instead.
        if "\n" not in plain and len(plain) < 80 and "```" not in plain:
            text_px = fm.horizontalAdvance(plain) + 28  # horizontal padding
            final_width = max(min_w, min(text_px, max_w))
        else:
            # Standard: let Qt layout the document to find natural width
            doc.setTextWidth(-1)
            ideal = doc.idealWidth()
            final_width = max(min_w, min(ideal + 4, max_w))

        # Layout at that width to measure height
        doc.setTextWidth(final_width)
        height = int(doc.size().height()) + 12  # small buffer for descenders

        # No scrollbar ever — show full content (user request)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedSize(int(final_width), height)

# ===================================================================
# 2. MESSAGE BUBBLE
# ===================================================================
class MessageBubble(QWidget):
    # Signals for bubble actions
    pause_requested = Signal()  # Emitted when pause icon clicked (AI bubbles)
    edit_requested = Signal(str)  # Emitted with message content when edit clicked (user bubbles)
    
    def __init__(self, text, sender="user"):
        super().__init__()
        self.sender = sender
        self._raw_text = text  # Store raw text for editing
        
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
        
        # --- Action Icons Row ---
        action_row = QHBoxLayout()
        action_row.setContentsMargins(10, 2, 10, 0)
        action_row.setSpacing(4)
        
        icon_style = """
            QPushButton {
                background: transparent;
                border: none;
                font-size: 14px;
                padding: 2px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
            }
        """
        
        if sender == "ai":
            # AI bubbles get a pause icon (hidden by default, shown during streaming)
            self.pause_btn = QPushButton("⏸")
            self.pause_btn.setStyleSheet(icon_style)
            self.pause_btn.setToolTip("Pause generation")
            self.pause_btn.setFixedSize(24, 24)
            self.pause_btn.clicked.connect(self._on_pause_clicked)
            self.pause_btn.hide()  # Hidden until streaming starts
            action_row.addWidget(self.pause_btn)
            action_row.addStretch()
            self.edit_btn = None
        else:
            # User bubbles get an edit icon
            action_row.addStretch()  # Push to right for user bubbles
            self.edit_btn = QPushButton("✏️")
            self.edit_btn.setStyleSheet(icon_style)
            self.edit_btn.setToolTip("Edit message")
            self.edit_btn.setFixedSize(24, 24)
            self.edit_btn.clicked.connect(self._on_edit_clicked)
            action_row.addWidget(self.edit_btn)
            self.pause_btn = None
        
        container_layout.addLayout(action_row)

        if sender == "user":
            self.layout.addStretch()
            self.layout.addWidget(container_widget)
        else:
            self.layout.addWidget(container_widget)
            self.layout.addStretch()
    
    def _on_pause_clicked(self):
        """Handle pause button click."""
        self.pause_requested.emit()
    
    def _on_edit_clicked(self):
        """Handle edit button click."""
        self.edit_requested.emit(self._raw_text)
    
    def set_streaming(self, is_streaming: bool):
        """Show/hide pause button during streaming (AI bubbles only)."""
        if self.pause_btn:
            self.pause_btn.setVisible(is_streaming)
    
    def set_edit_visible(self, visible: bool):
        """Show/hide edit button (user bubbles only)."""
        if self.edit_btn:
            self.edit_btn.setVisible(visible)

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


# ===================================================================
# 5. PROCESSING INDICATOR (Gemini-Style Animated Sparkle)
# ===================================================================
class ProcessingIndicator(QWidget):
    """Premium search indicator — Claude warmth, OpenAI pulse, Gemini flair.
    Shows during search phases, disappears when answer starts streaming."""

    # Rotating status phrases — personality-driven
    PROCESSING_TEXTS = [
        "Searching the web",
        "Reading sources",
        "Gathering context",
        "Cross-referencing",
        "Analyzing findings",
        "Pulling it together",
        "Checking one more thing",
        "Almost there",
    ]

    # Orb colors for pulse cycle (teal → warm → cool → back)
    ORB_COLORS = [
        "#4fd1c5",  # teal
        "#5eead4",  # mint
        "#38bdf8",  # sky
        "#818cf8",  # indigo
        "#a78bfa",  # violet
        "#818cf8",  # indigo
        "#38bdf8",  # sky
        "#5eead4",  # mint
    ]

    def __init__(self):
        super().__init__()
        self._text_index = 0
        self._orb_frame = 0
        self._pulse_phase = 0
        self._user_text = None  # Stores set_text override

        # Main layout — left-aligned like AI messages
        outer_layout = QHBoxLayout(self)
        outer_layout.setContentsMargins(0, 6, 0, 6)
        outer_layout.setSpacing(0)

        # Container — subtle frosted glass pill
        self._container = QWidget()
        self._container.setMaximumWidth(320)
        self._container.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 32, 38, 0.85);
                border-radius: 20px;
                border: 1px solid rgba(79, 209, 197, 0.25);
            }
        """)

        inner_layout = QHBoxLayout(self._container)
        inner_layout.setContentsMargins(12, 8, 16, 8)
        inner_layout.setSpacing(10)

        # ── Pulsing Orb (OpenAI-style) ──
        self.orb_label = QLabel("●")
        self.orb_label.setFixedSize(20, 20)
        self.orb_label.setAlignment(Qt.AlignCenter)
        self._update_orb()
        inner_layout.addWidget(self.orb_label)

        # ── Status Text ──
        self.text_label = QLabel(self.PROCESSING_TEXTS[0])
        self.text_label.setStyleSheet("""
            color: #b8c5d6;
            font-size: 12px;
            font-weight: 500;
            font-family: 'Segoe UI', 'Inter', sans-serif;
            letter-spacing: 0.3px;
            background: transparent;
            border: none;
        """)
        inner_layout.addWidget(self.text_label)

        # ── Animated Trailing Dots ──
        self.dots_label = QLabel("")
        self.dots_label.setFixedWidth(20)
        self.dots_label.setStyleSheet("""
            color: #4fd1c5;
            font-size: 12px;
            font-weight: 600;
            background: transparent;
            border: none;
        """)
        inner_layout.addWidget(self.dots_label)

        inner_layout.addStretch()

        outer_layout.addWidget(self._container)
        outer_layout.addStretch()

        # ── Animation Timers ──
        self._dot_count = 0
        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._animate_dots)
        self._dot_timer.start(400)

        self._text_timer = QTimer(self)
        self._text_timer.timeout.connect(self._rotate_text)
        self._text_timer.start(2500)

        self._orb_timer = QTimer(self)
        self._orb_timer.timeout.connect(self._animate_orb)
        self._orb_timer.start(200)

    def _update_orb(self):
        """Update the pulsing orb with current color and size."""
        color = self.ORB_COLORS[self._orb_frame % len(self.ORB_COLORS)]
        # Pulse between sizes 10-14px using sine-like pattern
        sizes = [10, 11, 12, 13, 14, 13, 12, 11]
        size = sizes[self._pulse_phase % len(sizes)]
        self.orb_label.setStyleSheet(f"""
            color: {color};
            font-size: {size}px;
            background: transparent;
            border: none;
        """)

    def _animate_orb(self):
        """Cycle orb color and pulse size."""
        self._pulse_phase = (self._pulse_phase + 1) % 8
        # Color shifts slower than pulse
        if self._pulse_phase == 0:
            self._orb_frame = (self._orb_frame + 1) % len(self.ORB_COLORS)
        self._update_orb()

    def _animate_dots(self):
        """Cycle trailing dots: · ·· ··· then blank."""
        self._dot_count = (self._dot_count + 1) % 4
        self.dots_label.setText("·" * self._dot_count)

    def _rotate_text(self):
        """Rotate through status phrases (only if not overridden)."""
        if self._user_text:
            return
        self._text_index = (self._text_index + 1) % len(self.PROCESSING_TEXTS)
        self.text_label.setText(self.PROCESSING_TEXTS[self._text_index])

    def set_text(self, text: str):
        """Override the rotating text with a specific message."""
        self._user_text = text
        self._text_timer.stop()
        self.text_label.setText(text)

    def resume_rotation(self):
        """Resume automatic text rotation after a set_text override."""
        self._user_text = None
        self._text_timer.start(2500)

    def stop(self):
        """Stop all animations (call before removing)."""
        self._dot_timer.stop()
        self._text_timer.stop()
        self._orb_timer.stop()
