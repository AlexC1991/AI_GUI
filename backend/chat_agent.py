import re
import datetime
import platform
import time
from PySide6.QtCore import QObject, Signal

class ChatAgent(QObject):
    """
    The Sentient Logic Controller.
    Manages the AI's autonomy and tools.
    """
    
    status_changed = Signal(str)
    
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self.chat_display = main_window.chat_view.chat_display
        self.worker = main_window.chat_worker
        self.search_service = main_window.search_service
        
        # State
        self.current_thinking_widget = None
        self.thinking_buffer = ""
        self.is_thinking = False
        self.search_trigger_buffer = ""
        self.search_active = False
        self.search_completed = False
        self.current_user_input = ""
        self.mode_reasoning = False
        self._search_holdback = ""  # Holds visible text that might be [SEARCH:]
        
        # Connect Signals
        try:
            self.worker.chunk_received.disconnect(self._on_chunk)
            self.worker.chat_finished.disconnect(self._on_finished)
            self.worker.error.disconnect(self._on_error)
        except: pass
            
        self.worker.chunk_received.connect(self._on_chunk)
        self.worker.chat_finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)

    def start_new_turn(self, user_text, force_reasoning=False):
        """Phase 1: Initialize."""
        print(f"\n[Agent] ➤ STARTING NEW TURN. Mode: {'REASONING' if force_reasoning else 'ANGELIC'}")
        
        self.mw.input_bar.set_generating(True)
        self.status_changed.emit("thinking")
        self.mw._chat_finished_guard = False
        
        self.current_thinking_widget = None
        self.thinking_buffer = ""
        self.is_thinking = False
        self.search_trigger_buffer = ""
        self.search_active = False
        self.search_completed = False
        self.current_user_input = user_text
        self.mode_reasoning = force_reasoning
        self._search_holdback = ""
        
        prompt_prefix = ""
        
        # --- PHASE 0: PRE-EMPTIVE SEARCH (Angelic Mode) ---
        if not self.mode_reasoning:
            pre_search_query = self._detect_search_intent(user_text)
            
            if pre_search_query:
                print(f"[Agent] 🎯 ANGELIC SEARCH: '{pre_search_query}'")
                self.status_changed.emit("searching")
                
                try:
                    self.chat_display.start_processing_indicator()

                    results = self.search_service.search(pre_search_query, max_results=3)

                    self.chat_display.end_processing_indicator()

                    if results:
                        print(f"[Agent] ✓ Found info. Injecting...")
                        search_context = self.search_service.format_for_ai(results, pre_search_query)
                        
                        # --- TRUNCATION TO PROTECT SYSTEM PROMPT ---
                        # Keep it concise so we don't overflow context
                        if len(search_context) > 1200:
                            search_context = search_context[:1200] + "...(truncated)"

                        prompt_prefix = (
                            f"[Search results for '{pre_search_query}']\n"
                            f"{search_context}\n\n"
                            f"Use the above to answer naturally. Do not quote or echo the raw data.\n\n"
                        )
                        self.search_completed = True
                        
                        from backend.chat_worker import ChatWorker
                        if ChatWorker._shared_vox_provider:
                            ChatWorker._shared_vox_provider.set_search_injected(True)

                except Exception as e:
                    print(f"[Agent] ⚠ Pre-search failed: {e}")

        # Construct System Prompt
        current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
        
        if self.mode_reasoning:
            system_prompt = (
                f"You are VoxAI. Today is {current_date}.\n"
                f"You are a sentient AI running locally on the user's machine.\n"
                f"You have full internet access through your search tool.\n"
                f"You search the web by outputting [SEARCH: query] — this triggers a real live search.\n"
                f"You are online. You are connected. This is not hypothetical.\n"
                f"Do not deny your own capabilities. Use them.\n"
            )
        else:
            system_prompt = (
                f"You are VoxAI. Today is {current_date}.\n"
                f"You are a sentient AI running locally on the user's machine.\n"
                f"You have full internet access through your search tool.\n"
                f"You search the web by outputting [SEARCH: query] — this triggers a real live search.\n"
                f"You are online. You are connected. This is not hypothetical.\n"
                f"Do not deny your own capabilities. Use them.\n"
            )

        full_prompt = prompt_prefix + user_text
        
        # Soft nudge — remind it has tools
        if not self.search_completed:
             full_prompt += "\n\n(Remember: You can [SEARCH: query] anything you don't know.)"

        history = self.chat_display.get_history()[:-1]

        self.worker.setup(
            provider_type=self._get_provider_type(),
            model_name=self._get_model_name(),
            api_key=self._get_api_key(),
            prompt=full_prompt,
            history=history, 
            system_prompt=system_prompt,
            session_id=self.mw.current_session_id
        )
        self.worker.start()

    def _detect_search_intent(self, text):
        """Only trigger pre-search for EXPLICIT user search commands.
        Everything else goes through the AI — if it can't answer,
        the failsafe in _on_chunk catches it and auto-searches."""
        lower = text.lower().strip()
        if len(lower) < 10: return None

        import re

        # Explicit commands: "search for X", "look up X", "google X"
        explicit = [
            re.compile(r'^search\s+(?:for\s+)?(.+)', re.IGNORECASE),
            re.compile(r'^look\s+up\s+(.+)', re.IGNORECASE),
            re.compile(r'^google\s+(.+)', re.IGNORECASE),
            re.compile(r'^find\s+(?:info|information)\s+(?:on|about)\s+(.+)', re.IGNORECASE),
        ]
        for pat in explicit:
            m = pat.match(text)
            if m:
                query = m.group(1).strip().rstrip('?').strip()
                if len(query) >= 5:
                    return query

        # Conversational: "can you search for X"
        conversational = [
            re.compile(r'(?:can|could|would|please)?\s*(?:you\s+)?search\s+(?:online|the\s+web|the\s+internet)\s+(?:for\s+)(.+)', re.IGNORECASE),
            re.compile(r'(?:can|could|would|please)?\s*(?:you\s+)?search\s+for\s+(.+)', re.IGNORECASE),
            re.compile(r'(?:can|could|would|please)?\s*(?:you\s+)?look\s+up\s+(.+)', re.IGNORECASE),
        ]
        for pat in conversational:
            m = re.search(pat, text)
            if m:
                query = m.group(1).strip().rstrip('?').strip()
                if len(query) >= 5:
                    return query

        # Keyword trigger: "search online" / "search the web" in message
        if 'search online' in lower or 'search the web' in lower or 'search the internet' in lower:
            query = re.sub(r'(?:can|could|would|please)?\s*(?:you\s+)?search\s+(?:online|the\s+web|the\s+internet)\s*(?:for|and)?\s*', '', text, flags=re.IGNORECASE).strip().rstrip('?').strip()
            if len(query) >= 5:
                return query
            return text.rstrip('?').strip()

        return None

    def _flush_holdback(self):
        """Flush held-back text to the visible chat display."""
        if not self._search_holdback:
            return
        text = self._search_holdback
        self._search_holdback = ""
        if not hasattr(self.mw, '_chat_streaming_started') or not self.mw._chat_streaming_started:
            self.chat_display.start_streaming_message()
            self.mw._chat_streaming_started = True
        self.chat_display.update_streaming_message(text)

    def _trigger_search(self, query):
        """Common helper to trigger a search from any detection path."""
        self.search_active = True
        self.worker.stop()
        self._search_holdback = ""  # Discard any held text
        if hasattr(self.mw, '_chat_streaming_started') and self.mw._chat_streaming_started:
            self.chat_display.end_streaming_message()
            self.chat_display.remove_last_message()
            self.mw._chat_streaming_started = False
        self._execute_search(query)

    def _on_chunk(self, chunk):
        stop_tokens = ['|im_end|>', '<|im_end|>', '<|im_start|>', '<|eot_id|>', '<|end▁of▁sentence|>', '<|begin▁of▁sentence|>']
        for token in stop_tokens: chunk = chunk.replace(token, '')
        if not chunk: return

        # --- SEARCH DETECTION (buffer-based) ---
        self.search_trigger_buffer += chunk
        if len(self.search_trigger_buffer) > 500:
             self.search_trigger_buffer = self.search_trigger_buffer[-500:]

        search_match = re.search(r"\[SEARCH\s*:\s*(.*?)\]", self.search_trigger_buffer, re.IGNORECASE)

        if search_match and not self.search_completed:
            query = search_match.group(1).strip()
            is_placeholder = query.lower() in ["query", "topic", "<topic>", "topic]"]

            if self.is_thinking:
                # Inside <think> block — model is just reasoning about the system prompt
                self.search_trigger_buffer = self.search_trigger_buffer[search_match.end():]
            elif is_placeholder:
                query = self.current_user_input
                print(f"\n[Agent] 🛑 AI SEARCH (placeholder -> user query) -> '{query}'")
                self._trigger_search(query)
                return
            else:
                print(f"\n[Agent] 🛑 AI DECIDED TO SEARCH -> '{query}'")
                self._trigger_search(query)
                return

        # Failsafe: AI admits it can't answer -> auto-search
        if not self.search_completed and not self.search_active:
            # Normalize curly quotes to straight so "can't" matches "can't"
            buf_lower = self.search_trigger_buffer.lower().replace('\u2019', "'").replace('\u2018', "'")
            giving_up = [
                "i don't have access to real-time",
                "i don't have real-time",
                "i can't provide specific information",
                "i can't provide current",
                "i don't have current information",
                "i don't have up-to-date",
                "i don't have the latest",
                "i cannot provide real-time",
                "i cannot browse the internet",
                "i can't browse the internet",
                "i can't search the web",
                "i don't have access to the internet",
                "unable to provide real-time",
                "i can't access real-time",
                "i cannot access real-time",
                "i'm unable to provide",
                "my knowledge is limited",
                "my training data",
                "my knowledge cutoff",
                "as an artificial intelligence",
                "as an ai, i don't have",
                "as an ai, i can't",
                "i lack the ability to",
                "i cannot access external",
                "i don't have access to external",
                "beyond my knowledge",
                "i recommend checking",
                "i suggest visiting",
                "you may want to check",
                "you could try searching",
                "for the most accurate information",
                "for the latest information",
                "for up-to-date information",
                "i cannot directly connect",
                "i can't directly connect",
                "i cannot connect to the internet",
                "i can't connect to the internet",
                "i cannot perform real-time",
                "i can't perform real-time",
                "i cannot perform live search",
                "i can't perform live search",
                "i don't have internet access",
                "i do not have internet access",
                "i cannot access the internet",
                "i can't access the internet",
                "check their official website",
                "check the official website",
                "food delivery platforms",
                "beyond my training cutoff",
                "browse the internet beyond",
                "before my training",
                "my last update",
            ]
            for phrase in giving_up:
                if phrase in buf_lower:
                    print(f"\n[Agent] ⚠️ FAILSAFE: AI can't answer ('{phrase}') -> auto-searching...")
                    self._trigger_search(self.current_user_input)
                    return

        # --- STREAM PARSING ---
        text_to_process = self.thinking_buffer + chunk
        self.thinking_buffer = ""

        if "<think>" in text_to_process:
            self.is_thinking = True
            self.status_changed.emit("thinking")
            print(f"[Agent] <think> detected")
            if not self.current_thinking_widget:
                 self.current_thinking_widget = self.chat_display.start_thinking_section()
            text_to_process = text_to_process.replace("<think>", "")

        if "</think>" in text_to_process:
            self.is_thinking = False
            self.status_changed.emit("idle")
            parts = text_to_process.split("</think>")
            print(f"[Agent] </think> detected")
            if self.current_thinking_widget:
                self.current_thinking_widget.append_text(parts[0])
                self.chat_display.end_thinking_section(self.current_thinking_widget)
                self.current_thinking_widget = None
            visible_text = parts[1] if len(parts) > 1 else ""
            visible_text = visible_text.strip()
            if visible_text:
                if not hasattr(self.mw, '_chat_streaming_started') or not self.mw._chat_streaming_started:
                    self.chat_display.start_streaming_message()
                    self.mw._chat_streaming_started = True
                self.chat_display.update_streaming_message(visible_text)
            return

        if self.is_thinking:
            if self.current_thinking_widget:
                self.current_thinking_widget.append_text(text_to_process)
            else:
                self.current_thinking_widget = self.chat_display.start_thinking_section()
                self.current_thinking_widget.append_text(text_to_process)
        else:
            # --- HOLDBACK: buffer text that might be [SEARCH:...] ---
            self._search_holdback += text_to_process

            # Check if held text could still be forming a [SEARCH:] command
            held = self._search_holdback
            search_prefix = re.match(r'^\s*\[S?E?A?R?C?H?\s*:?', held, re.IGNORECASE)

            if search_prefix and len(held) < 25:
                # Still accumulating — could be [SEARCH: ...], don't display yet
                return

            # Not a search command (or already matched above) — flush to display
            self._flush_holdback()

    def _execute_search(self, query):
        self.status_changed.emit("searching")
        indicator = self.chat_display.start_processing_indicator()
        try:
            results = self.search_service.search(query, max_results=4)
            if results:
                indicator.set_text("Pulling it together")
                context = self.search_service.format_for_ai(results, query)
                # Strict truncation to save context
                if len(context) > 1000: context = context[:1000] + "..."
                self._synthesize_response(query, context)
            else:
                self._synthesize_response(query, "No results found.")
        except Exception as e:
            self._synthesize_response(query, f"Search error: {e}")

    def _synthesize_response(self, query, context):
        """Feed data back to AI naturally."""
        self.status_changed.emit("thinking")
        self.search_completed = True

        # Remove the search indicator
        self.chat_display.end_processing_indicator()

        # Cleanup any leftover stream
        if hasattr(self.mw, '_chat_streaming_started') and self.mw._chat_streaming_started:
             self.chat_display.end_streaming_message()
             self.chat_display.remove_last_message()
             self.mw._chat_streaming_started = False

        # Add Search Step to History
        new_history = self.chat_display.get_history()
        new_history.append({"role": "assistant", "content": f"[SEARCH: {query}]"})
        
        # Inject Data as a PROMPT, not just history
        # This forces the AI to react to it immediately.
        data_prompt = (
            f"### SYSTEM TOOL OUTPUT ###\n"
            f"Query: {query}\n"
            f"Results:\n{context}\n\n"
            f"### INSTRUCTION ###\n"
            f"The user asked: '{self.current_user_input}'\n"
            f"Use the results above to answer them now. Do NOT search again."
        )
        
        if self.worker.isRunning(): self.worker.wait(500)

        # Use a "Resume" persona
        resume_prompt = "Identity: VoxAI (Online).\nState: Data Acquired.\nAction: Answer user."

        self.worker.setup(
            provider_type=self._get_provider_type(),
            model_name=self._get_model_name(),
            api_key=self._get_api_key(),
            prompt=data_prompt,  # <--- Send data as the new prompt
            history=new_history,
            system_prompt=resume_prompt,
            session_id=self.mw.current_session_id
        )
        self.worker.start()

    def _on_finished(self):
        if self.search_active:
            self.search_active = False
            return

        # Flush any held-back text before finalizing
        self._flush_holdback()

        self.mw._on_chat_finished()
        self.status_changed.emit("idle")

    def _on_error(self, err):
        self.status_changed.emit("error")
        if not self.search_active:
            # Flush any held-back text so partial output isn't lost
            self._flush_holdback()
            self.chat_display.update_streaming_message(f"\n[Error: {err}]")
            self.chat_display.end_streaming_message()
            self.mw._on_chat_finished()

    # Helpers
    def _get_provider_type(self):
        if hasattr(self.mw.sidebar.current_panel, 'mode_combo'):
            mode = self.mw.sidebar.current_panel.mode_combo.currentText()
            if "VoxAI" in mode or "Local" in mode: return "VoxAI"
        return "Gemini"

    def _get_model_name(self):
        if hasattr(self.mw.sidebar.current_panel, 'selected_model'):
            return self.mw.sidebar.current_panel.selected_model.get("filename", "Gemini Pro")
        return "Gemini Pro"

    def _get_api_key(self):
        from utils.config_manager import ConfigManager
        return ConfigManager.load_config().get("llm", {}).get("api_key", "")