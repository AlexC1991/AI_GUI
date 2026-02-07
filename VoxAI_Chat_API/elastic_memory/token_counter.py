from typing import List, Dict

class TokenCounter:
    """
    Accurate token counting using the running LLM instance.
    """
    def __init__(self, llm):
        self.llm = llm

    def count(self, text: str) -> int:
        if not text: return 0
        try:
            # Helper to count tokens
            tokens = self.llm.tokenize(text.encode("utf-8", "ignore"))
            return len(tokens)
        except:
            return len(text) // 4 # Rough fallback

    def count_messages(self, messages: List[Dict], template_handler) -> int:
        """
        Formats messages via template then counts tokens.
        This includes special tokens overhead.
        """
        if not messages: return 0
        try:
            if template_handler:
                prompt = template_handler.format_prompt(messages, add_generation_prompt=True)
            else:
                # Basic concatenation
                prompt = ""
                for m in messages:
                    prompt += m["content"]
            
            return self.count(prompt)
        except:
            return 0
