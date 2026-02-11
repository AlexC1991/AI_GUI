"""
Chat Template Handler

Automatically detects model type from filename/metadata and applies
the correct prompt format (ChatML, Llama, Alpaca, etc.)

Supported formats:
- ChatML (Dolphin, OpenHermes, Nous-Hermes, etc.)
- Llama 2/3 (Meta Llama models)
- Alpaca (Stanford Alpaca, Vicuna, etc.)
- Mistral Instruct
- Phi-3
- Qwen
- Zephyr
- Raw (no template, for base models)
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ChatFormat(Enum):
    """Supported chat formats."""
    CHATML = "chatml"
    LLAMA2 = "llama2"
    LLAMA3 = "llama3"
    ALPACA = "alpaca"
    MISTRAL = "mistral"
    PHI3 = "phi3"
    QWEN = "qwen"
    ZEPHYR = "zephyr"
    VICUNA = "vicuna"
    RAW = "raw"


@dataclass
class ChatTemplate:
    """Chat template definition."""
    format: ChatFormat
    system_prefix: str
    system_suffix: str
    user_prefix: str
    user_suffix: str
    assistant_prefix: str
    assistant_suffix: str
    bos_token: str = ""
    eos_token: str = ""
    stop_tokens: List[str] = None
    
    def __post_init__(self):
        if self.stop_tokens is None:
            self.stop_tokens = []


# ============================================
# TEMPLATE DEFINITIONS
# ============================================

TEMPLATES = {
    ChatFormat.CHATML: ChatTemplate(
        format=ChatFormat.CHATML,
        system_prefix="<|im_start|>system\n",
        system_suffix="<|im_end|>\n",
        user_prefix="<|im_start|>user\n",
        user_suffix="<|im_end|>\n",
        assistant_prefix="<|im_start|>assistant\n",
        assistant_suffix="<|im_end|>\n",
        stop_tokens=["<|im_end|>", "<|im_start|>"]
    ),
    
    ChatFormat.LLAMA2: ChatTemplate(
        format=ChatFormat.LLAMA2,
        bos_token="<s>",
        system_prefix="[INST] <<SYS>>\n",
        system_suffix="\n<</SYS>>\n\n",
        user_prefix="",
        user_suffix=" [/INST] ",
        assistant_prefix="",
        assistant_suffix=" </s><s>[INST] ",
        eos_token="</s>",
        stop_tokens=["</s>", "[INST]"]
    ),
    
    ChatFormat.LLAMA3: ChatTemplate(
        format=ChatFormat.LLAMA3,
        bos_token="<|begin_of_text|>",
        system_prefix="<|start_header_id|>system<|end_header_id|>\n\n",
        system_suffix="<|eot_id|>",
        user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
        user_suffix="<|eot_id|>",
        assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
        assistant_suffix="<|eot_id|>",
        eos_token="<|eot_id|>",
        stop_tokens=["<|eot_id|>", "<|end_of_text|>"]
    ),
    
    ChatFormat.ALPACA: ChatTemplate(
        format=ChatFormat.ALPACA,
        system_prefix="### System:\n",
        system_suffix="\n\n",
        user_prefix="### Instruction:\n",
        user_suffix="\n\n",
        assistant_prefix="### Response:\n",
        assistant_suffix="\n\n",
        stop_tokens=["###", "### Instruction:"]
    ),
    
    ChatFormat.MISTRAL: ChatTemplate(
        format=ChatFormat.MISTRAL,
        bos_token="<s>",
        system_prefix="[INST] ",
        system_suffix="\n\n",
        user_prefix="",
        user_suffix=" [/INST]",
        assistant_prefix="",
        assistant_suffix="</s> [INST] ",
        eos_token="</s>",
        stop_tokens=["</s>"]
    ),
    
    ChatFormat.PHI3: ChatTemplate(
        format=ChatFormat.PHI3,
        system_prefix="<|system|>\n",
        system_suffix="<|end|>\n",
        user_prefix="<|user|>\n",
        user_suffix="<|end|>\n",
        assistant_prefix="<|assistant|>\n",
        assistant_suffix="<|end|>\n",
        stop_tokens=["<|end|>", "<|user|>", "<|assistant|>"]
    ),
    
    ChatFormat.QWEN: ChatTemplate(
        format=ChatFormat.QWEN,
        system_prefix="<|im_start|>system\n",
        system_suffix="<|im_end|>\n",
        user_prefix="<|im_start|>user\n",
        user_suffix="<|im_end|>\n",
        assistant_prefix="<|im_start|>assistant\n",
        assistant_suffix="<|im_end|>\n",
        stop_tokens=["<|im_end|>", "<|endoftext|>"]
    ),
    
    ChatFormat.ZEPHYR: ChatTemplate(
        format=ChatFormat.ZEPHYR,
        system_prefix="<|system|>\n",
        system_suffix="</s>\n",
        user_prefix="<|user|>\n",
        user_suffix="</s>\n",
        assistant_prefix="<|assistant|>\n",
        assistant_suffix="</s>\n",
        stop_tokens=["</s>"]
    ),
    
    ChatFormat.VICUNA: ChatTemplate(
        format=ChatFormat.VICUNA,
        system_prefix="SYSTEM: ",
        system_suffix="\n\n",
        user_prefix="USER: ",
        user_suffix="\n",
        assistant_prefix="ASSISTANT: ",
        assistant_suffix="\n",
        stop_tokens=["USER:", "SYSTEM:"]
    ),
    
    ChatFormat.RAW: ChatTemplate(
        format=ChatFormat.RAW,
        system_prefix="",
        system_suffix="\n\n",
        user_prefix="",
        user_suffix="\n",
        assistant_prefix="",
        assistant_suffix="\n",
        stop_tokens=[]
    ),
}


# ============================================
# MODEL NAME TO FORMAT MAPPING
# ============================================

MODEL_FORMAT_PATTERNS = [
    # ChatML models
    (r"dolphin", ChatFormat.CHATML),
    (r"openhermes", ChatFormat.CHATML),
    (r"nous-hermes", ChatFormat.CHATML),
    (r"hermes", ChatFormat.CHATML),
    (r"openchat", ChatFormat.CHATML),
    (r"starling", ChatFormat.CHATML),
    (r"yi-.*chat", ChatFormat.CHATML),
    (r"mythomax", ChatFormat.ALPACA),  # MythoMax uses Alpaca
    (r"noromaid", ChatFormat.ALPACA),  # Noromaid uses Alpaca
    
    # Llama models
    (r"llama-?3", ChatFormat.LLAMA3),
    (r"llama-?2.*chat", ChatFormat.LLAMA2),
    (r"llama-?2.*instruct", ChatFormat.LLAMA2),
    (r"codellama.*instruct", ChatFormat.LLAMA2),
    
    # Mistral models
    (r"mistral.*instruct", ChatFormat.MISTRAL),
    (r"mixtral.*instruct", ChatFormat.MISTRAL),
    (r"mistral-nemo", ChatFormat.MISTRAL),
    
    # Phi / GPT-OSS models (use Phi3 <|end|> style tokens)
    (r"phi-?3", ChatFormat.PHI3),
    (r"phi-?2", ChatFormat.ALPACA),
    (r"gpt.*oss", ChatFormat.PHI3),
    
    # Qwen models
    (r"qwen", ChatFormat.QWEN),
    
    # Zephyr models
    (r"zephyr", ChatFormat.ZEPHYR),
    
    # Vicuna models
    (r"vicuna", ChatFormat.VICUNA),
    
    # Alpaca/general instruct
    (r"alpaca", ChatFormat.ALPACA),
    (r"wizard.*instruct", ChatFormat.ALPACA),
    (r"wizardlm", ChatFormat.VICUNA),
]


class ChatTemplateHandler:
    """
    Handles chat template detection and prompt formatting.
    
    Usage:
        handler = ChatTemplateHandler()
        
        # Auto-detect from model name
        handler.detect_format("dolphin-2.8-mistral-7b.gguf")
        
        # Or set manually
        handler.set_format(ChatFormat.CHATML)
        
        # Format messages
        prompt = handler.format_prompt(messages)
        
        # Get stop tokens for generation
        stops = handler.get_stop_tokens()
    """
    
    def __init__(self, default_format: ChatFormat = ChatFormat.CHATML):
        self.format = default_format
        self.template = TEMPLATES[default_format]
        self._model_name = None
    
    def detect_format(self, model_name: str) -> ChatFormat:
        """
        Detect chat format from model filename.
        
        Args:
            model_name: Model filename or path
            
        Returns:
            Detected ChatFormat
        """
        self._model_name = model_name
        name_lower = model_name.lower()
        
        for pattern, fmt in MODEL_FORMAT_PATTERNS:
            if re.search(pattern, name_lower):
                self.set_format(fmt)
                return fmt
        
        # Default to ChatML for unknown models (most compatible)
        self.set_format(ChatFormat.CHATML)
        return ChatFormat.CHATML
    
    def set_format(self, format: ChatFormat):
        """Set the chat format manually."""
        self.format = format
        self.template = TEMPLATES[format]
    
    def format_prompt(self, messages: List[Dict[str, str]], 
                      add_generation_prompt: bool = True) -> str:
        """
        Format a list of messages into a prompt string.
        
        Args:
            messages: List of {"role": "system/user/assistant", "content": "..."}
            add_generation_prompt: Whether to add the assistant prefix at the end
            
        Returns:
            Formatted prompt string
        """
        prompt = self.template.bos_token
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt += self.template.system_prefix + content + self.template.system_suffix
            elif role == "user":
                prompt += self.template.user_prefix + content + self.template.user_suffix
            elif role == "assistant":
                prompt += self.template.assistant_prefix + content + self.template.assistant_suffix
        
        if add_generation_prompt:
            prompt += self.template.assistant_prefix
        
        return prompt
    
    def format_single_turn(self, user_message: str, 
                           system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Format a single-turn conversation.
        
        Args:
            user_message: The user's message
            system_prompt: System prompt to use
            
        Returns:
            Formatted prompt string
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        return self.format_prompt(messages)
    
    def get_stop_tokens(self) -> List[str]:
        """Get the stop tokens for this format."""
        return self.template.stop_tokens.copy()
    
    def get_eos_token(self) -> str:
        """Get the EOS token for this format."""
        return self.template.eos_token or self.template.assistant_suffix.strip()
    
    def extract_response(self, generated_text: str) -> str:
        """
        Extract just the assistant's response from generated text.
        
        Args:
            generated_text: Raw generated text that may include template tokens
            
        Returns:
            Cleaned response text
        """
        text = generated_text
        
        # Remove any stop tokens that appear at the end
        for stop in self.template.stop_tokens:
            if text.endswith(stop):
                text = text[:-len(stop)]
        
        # Remove assistant prefix if it appears at the start
        if text.startswith(self.template.assistant_prefix):
            text = text[len(self.template.assistant_prefix):]
        
        return text.strip()
    
    def get_format_info(self) -> Dict:
        """Get information about the current format."""
        return {
            "format": self.format.value,
            "model": self._model_name,
            "stop_tokens": self.template.stop_tokens,
            "example": self.format_single_turn("Hello!", "You are helpful.")[:200] + "..."
        }


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def detect_chat_format(model_name: str) -> ChatFormat:
    """Quick function to detect format from model name."""
    handler = ChatTemplateHandler()
    return handler.detect_format(model_name)


def format_for_model(model_name: str, messages: List[Dict[str, str]]) -> str:
    """Quick function to format messages for a specific model."""
    handler = ChatTemplateHandler()
    handler.detect_format(model_name)
    return handler.format_prompt(messages)


def get_stop_tokens_for_model(model_name: str) -> List[str]:
    """Quick function to get stop tokens for a model."""
    handler = ChatTemplateHandler()
    handler.detect_format(model_name)
    return handler.get_stop_tokens()


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("Chat Template Handler Test\n")
    
    test_models = [
        "dolphin-2.8-mistral-7b-v02-Q4_K_M.gguf",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "Mistral-Nemo-12B-Instruct.Q4_K_M.gguf",
        "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "mythomax-l2-13b.Q4_K_M.gguf",
        "noromaid-13b-v0.3.Q4_K_M.gguf",
    ]
    
    handler = ChatTemplateHandler()
    
    for model in test_models:
        fmt = handler.detect_format(model)
        print(f"Model: {model}")
        print(f"  Format: {fmt.value}")
        print(f"  Stop tokens: {handler.get_stop_tokens()}")
        
        # Show example prompt
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi!"}
        ]
        prompt = handler.format_prompt(messages)
        print(f"  Example prompt:\n{prompt[:150]}...")
        print()
