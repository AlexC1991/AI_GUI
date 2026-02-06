"""
VOX API - LLM Chat Engine with Chat Template Support v4

Features:
- Automatic chat template detection (ChatML, Llama, Alpaca, etc.)
- GPU retry with CPU fallback
- Idle timeout and KV cache management
- Proper stop token handling
"""

import os
import time
import gc
import threading
import warnings
from typing import Generator, Dict, List, Optional, Union

# Suppress the LlamaModel cleanup warning
warnings.filterwarnings('ignore', message='.*LlamaModel.*sampler.*')

import llama_cpp
from llama_cpp import Llama

# FIX: Load backends explicitly for newer llama-cpp-python versions
try:
    llama_cpp.ggml_backend_load_all()
    print("[VOX API] Backends loaded via ggml_backend_load_all()")
except Exception:
    pass # Older versions might not have this/need this

# Import chat template handler
try:
    from chat_templates import ChatTemplateHandler, ChatFormat
    TEMPLATES_AVAILABLE = True
except ImportError:
    TEMPLATES_AVAILABLE = False
    print("[VOX API] Warning: chat_templates.py not found, using basic formatting")

import machine_engine_handshake


class VoxAPI:
    """
    A clean API wrapper for the VOX-AI Engine with automatic chat template detection.
    """
    
    _last_load_was_fallback = False
    
    def __init__(self, model_path: str = None, verbose: bool = False, 
                 max_retries: int = 2, retry_delay: float = 1.5,
                 idle_timeout: float = 300.0, chat_format: str = None):
        """
        Initialize the VOX Engine with automatic hardware optimization.
        
        Args:
            model_path: Path to the .gguf model file
            verbose: Enable detailed logging
            max_retries: Number of GPU load attempts before falling back to CPU
            retry_delay: Seconds to wait between retries
            idle_timeout: Seconds of inactivity before clearing KV cache (0 = never)
            chat_format: Force a specific chat format ('chatml', 'llama3', 'alpaca', etc.)
        """
        self.verbose = verbose
        self.history: List[Dict[str, str]] = []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.idle_timeout = idle_timeout
        self.using_fallback = False
        self._forced_format = chat_format
        
        self._last_activity = time.time()
        self._idle_timer: Optional[threading.Timer] = None
        self._is_generating = False
        self._shutdown_requested = False
        
        # Initialize chat template handler
        if TEMPLATES_AVAILABLE:
            self.template_handler = ChatTemplateHandler()
        else:
            self.template_handler = None
        
        # Hardware Handshake
        self.mode, self.phys_cores, self.config = machine_engine_handshake.get_hardware_config()
        self._original_config = self.config.copy()
        
        if self.verbose:
            print(f"[VOX API] Mode: {self.mode}")
            print(f"[VOX API] GPU Layers: {self.config['n_gpu_layers']}, Threads: {self.config['n_threads']}")
            
        self._apply_env_optimizations()
        
        if model_path is None:
            model_path = self._auto_find_model()
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
            
        self.model_name = os.path.basename(model_path)
        self.model_path = model_path
        
        # Detect chat format from model name
        self._setup_chat_format()
        
        if self.verbose:
            print(f"[VOX API] Loading: {self.model_name}")
            if self.template_handler:
                print(f"[VOX API] Chat format: {self.template_handler.format.value}")
        
        self.llm = self._load_model_with_fallback()
        
        self.warmup()
        self._schedule_idle_check()
        
        if self.verbose:
            status = "CPU FALLBACK" if self.using_fallback else self.mode
            print(f"[VOX API] ✓ Ready: {status}")

    def _setup_chat_format(self):
        """Detect and setup the appropriate chat format."""
        if not self.template_handler:
            return
        
        if self._forced_format:
            # User specified a format
            try:
                fmt = ChatFormat(self._forced_format.lower())
                self.template_handler.set_format(fmt)
                if self.verbose:
                    print(f"[VOX API] Using forced chat format: {fmt.value}")
            except ValueError:
                if self.verbose:
                    print(f"[VOX API] Unknown format '{self._forced_format}', auto-detecting...")
                self.template_handler.detect_format(self.model_name)
        else:
            # Auto-detect from model name
            detected = self.template_handler.detect_format(self.model_name)
            if self.verbose:
                print(f"[VOX API] Auto-detected chat format: {detected.value}")

    def _load_model_with_fallback(self) -> Llama:
        """Attempt to load model with GPU, fallback to CPU if needed."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    if self.verbose:
                        print(f"[VOX API] Retry {attempt}/{self.max_retries} - waiting {self.retry_delay}s for VRAM...")
                    gc.collect()
                    time.sleep(self.retry_delay)
                
                config = self._original_config.copy()
                
                llm = Llama(
                    model_path=self.model_path,
                    n_ctx=2048,
                    n_gpu_layers=config['n_gpu_layers'],
                    n_threads=config['n_threads'],
                    n_threads_batch=config['n_threads_batch'],
                    n_batch=config['n_batch'],
                    flash_attn=config['flash_attn'],
                    use_mlock=config['use_mlock'],
                    cache_type_k=config['cache_type_k'],
                    cache_type_v=config['cache_type_v'],
                    use_mmap=True,
                    verbose=self.verbose
                )
                
                VoxAPI._last_load_was_fallback = False
                self.using_fallback = False
                self.config = config
                return llm
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                is_vram_error = any(x in error_str for x in [
                    'vram', 'memory', 'out of memory', 'oom',
                    'vulkan', 'cuda', 'allocation', 'failed to load',
                    'invalid vector subscript', '0 mib free'
                ])
                
                if self.verbose:
                    print(f"[VOX API] Load attempt {attempt + 1} failed: {e}")
                
                if not is_vram_error:
                    break
        
        # CPU Fallback
        if self.verbose:
            print(f"[VOX API] ⚠ GPU loading failed after {self.max_retries} attempts")
            print(f"[VOX API] Falling back to CPU-only mode...")
        
        try:
            cpu_config = self._get_cpu_fallback_config()
            
            llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_gpu_layers=0,
                n_threads=cpu_config['n_threads'],
                n_threads_batch=cpu_config['n_threads_batch'],
                n_batch=cpu_config['n_batch'],
                flash_attn=False,
                use_mlock=cpu_config['use_mlock'],
                cache_type_k='f16',
                cache_type_v='f16',
                use_mmap=True,
                verbose=self.verbose
            )
            
            VoxAPI._last_load_was_fallback = True
            self.using_fallback = True
            self.config = cpu_config
            self.config['n_gpu_layers'] = 0
            self.mode = "CPU (Fallback)"
            
            if self.verbose:
                print(f"[VOX API] ✓ CPU fallback successful")
            
            return llm
            
        except Exception as cpu_error:
            raise RuntimeError(
                f"Failed to load model. GPU error: {last_error}. CPU error: {cpu_error}"
            )
    
    def _get_cpu_fallback_config(self) -> dict:
        """Get optimized CPU-only configuration."""
        cpu_threads = max(1, self.phys_cores - 1)
        return {
            'n_threads': cpu_threads,
            'n_threads_batch': cpu_threads,
            'n_batch': 512,
            'use_mlock': False,
            'flash_attn': False,
        }

    def _apply_env_optimizations(self):
        """Apply environment variables for performance."""
        root_path = os.path.dirname(os.path.abspath(__file__))
        
        if hasattr(os, 'add_dll_directory'): 
            try: 
                os.add_dll_directory(root_path)
            except: 
                pass
            
        if "busy_wait" in self.config:
            os.environ["GGML_VK_FORCE_BUSY_WAIT"] = self.config["busy_wait"]
        
        os.environ["GGML_NUMA"] = "0"
        os.environ["GGML_BACKEND_SEARCH_PATH"] = root_path
        os.environ["LLAMA_CPP_LIB"] = os.path.join(root_path, "llama.dll")

    def _auto_find_model(self) -> str:
        """Find the first .gguf file in ./models."""
        root_path = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(root_path, "models")
        
        if not os.path.exists(models_dir):
            raise FileNotFoundError("Models directory './models' not found")
            
        files = [f for f in os.listdir(models_dir) if f.endswith(".gguf")]
        if not files:
            raise FileNotFoundError("No .gguf models found in ./models")
            
        return os.path.join(models_dir, files[0])

    def warmup(self):
        """Run a silent inference to load weights."""
        if self.verbose: 
            print("[VOX API] Warming up...")
        try:
            # Use raw completion for warmup to avoid template issues
            self.llm(".", max_tokens=1)
        except Exception as e:
            if self.verbose:
                print(f"[VOX API] Warmup warning: {e}")
        self._last_activity = time.time()

    # ============================================
    # IDLE MANAGEMENT
    # ============================================
    
    def _schedule_idle_check(self):
        """Schedule the next idle check."""
        if self.idle_timeout <= 0 or self._shutdown_requested:
            return
        
        if self._idle_timer:
            self._idle_timer.cancel()
        
        self._idle_timer = threading.Timer(
            self.idle_timeout / 2,
            self._check_idle
        )
        self._idle_timer.daemon = True
        self._idle_timer.start()
    
    def _check_idle(self):
        """Check if we've been idle too long."""
        if self._shutdown_requested:
            return
            
        if self._is_generating:
            self._schedule_idle_check()
            return
        
        idle_time = time.time() - self._last_activity
        
        if idle_time >= self.idle_timeout:
            if self.verbose:
                print(f"[VOX API] Idle for {idle_time:.0f}s, resetting KV cache...")
            self.reset_context()
        
        self._schedule_idle_check()
    
    def reset_context(self):
        """Reset the KV cache to free memory."""
        try:
            if hasattr(self.llm, 'reset'):
                self.llm.reset()
            elif hasattr(self.llm, '_ctx') and self.llm._ctx:
                pass
        except Exception as e:
            if self.verbose and 'sampler' not in str(e).lower():
                print(f"[VOX API] Context reset note: {e}")
        
        gc.collect()
        
        if self.verbose:
            print("[VOX API] Context reset, KV cache cleared")
    
    def _touch_activity(self):
        """Mark that activity occurred."""
        self._last_activity = time.time()

    # ============================================
    # CHAT INTERFACE
    # ============================================

    def chat(self, user_message: str, stream: bool = True, 
             system_prompt: str = None) -> Union[str, Generator[str, None, None]]:
        """
        Send a message to the AI and get a response.
        
        Args:
            user_message: The user's message
            stream: Whether to stream the response
            system_prompt: Optional system prompt override
            
        Returns:
            String response if stream=False, Generator if stream=True
        """
        self._touch_activity()
        
        # Initialize history with system prompt if empty
        if not self.history:
            sys_msg = system_prompt or "You are a helpful assistant."
            self.history.append({"role": "system", "content": sys_msg})
            
        self.history.append({"role": "user", "content": user_message})
        
        if stream:
            return self._stream_response()
        else:
            return self._full_response()

    def _format_messages_for_completion(self) -> str:
        """Format messages using the appropriate chat template."""
        if self.template_handler:
            return self.template_handler.format_prompt(self.history, add_generation_prompt=True)
        else:
            # Fallback: basic ChatML format
            prompt = ""
            for msg in self.history:
                role = msg["role"]
                content = msg["content"]
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            return prompt

    def _get_stop_tokens(self) -> List[str]:
        """Get stop tokens for the current model."""
        if self.template_handler:
            return self.template_handler.get_stop_tokens()
        else:
            return ["<|im_end|>", "<|im_start|>"]

    def _stream_response(self) -> Generator[str, None, None]:
        """Internal generator for streaming responses."""
        self._is_generating = True
        full_response = ""
        
        try:
            # Format prompt with proper template
            prompt = self._format_messages_for_completion()
            stop_tokens = self._get_stop_tokens()
            
            if self.verbose:
                print(f"[VOX API] Using stop tokens: {stop_tokens}")
            
            # Use raw completion with proper stop tokens
            stream = self.llm(
                prompt,
                max_tokens=2048,
                temperature=0.7,
                top_k=40,
                repeat_penalty=1.1,
                stop=stop_tokens,
                stream=True,
                echo=False
            )
            
            for output in stream:
                self._touch_activity()
                token = output["choices"][0]["text"]
                
                # Check for stop tokens in the output
                should_stop = False
                for stop in stop_tokens:
                    if stop in token:
                        token = token.split(stop)[0]
                        should_stop = True
                        break
                
                if token:
                    full_response += token
                    yield token
                
                if should_stop:
                    break
                    
            self.history.append({"role": "assistant", "content": full_response})
        
        except Exception as e:
            if self.verbose:
                print(f"[VOX API] Stream error: {e}")
            raise
        
        finally:
            self._is_generating = False
            self._touch_activity()

    def _full_response(self) -> str:
        """Internal method for non-streaming response."""
        self._is_generating = True
        
        try:
            # Format prompt with proper template
            prompt = self._format_messages_for_completion()
            stop_tokens = self._get_stop_tokens()
            
            response = self.llm(
                prompt,
                max_tokens=2048,
                temperature=0.7,
                top_k=40,
                repeat_penalty=1.1,
                stop=stop_tokens,
                stream=False,
                echo=False
            )
            
            text = response["choices"][0]["text"].strip()
            
            # Clean up any stop tokens that slipped through
            for stop in stop_tokens:
                if text.endswith(stop):
                    text = text[:-len(stop)].strip()
            
            self.history.append({"role": "assistant", "content": text})
            return text
        
        finally:
            self._is_generating = False
            self._touch_activity()

    def clear_history(self):
        """Reset conversation context."""
        self.history = []
        self.reset_context()

    def get_stats(self) -> dict:
        """Get info about the loaded model and hardware."""
        stats = {
            "model": self.model_name,
            "mode": self.mode,
            "cores": self.phys_cores,
            "gpu_layers": self.config['n_gpu_layers'],
            "using_fallback": self.using_fallback,
            "idle_timeout": self.idle_timeout,
        }
        
        if self.template_handler:
            stats["chat_format"] = self.template_handler.format.value
        
        return stats
    
    def get_chat_format(self) -> str:
        """Get the current chat format name."""
        if self.template_handler:
            return self.template_handler.format.value
        return "chatml"
    
    def set_chat_format(self, format_name: str):
        """
        Manually set the chat format.
        
        Args:
            format_name: One of 'chatml', 'llama2', 'llama3', 'alpaca', 
                        'mistral', 'phi3', 'qwen', 'zephyr', 'vicuna', 'raw'
        """
        if self.template_handler:
            try:
                from chat_templates import ChatFormat
                fmt = ChatFormat(format_name.lower())
                self.template_handler.set_format(fmt)
                if self.verbose:
                    print(f"[VOX API] Chat format changed to: {format_name}")
            except ValueError:
                if self.verbose:
                    print(f"[VOX API] Unknown format: {format_name}")
    
    def shutdown(self):
        """Clean shutdown."""
        self._shutdown_requested = True
        
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None
        
        gc.collect()
        
        if self.verbose:
            print("[VOX API] Shutdown complete")
    
    @classmethod
    def was_last_load_fallback(cls) -> bool:
        """Check if the last model load used CPU fallback."""
        return cls._last_load_was_fallback


# Usage Example
if __name__ == "__main__":
    print("Testing VOX API with Chat Templates...")
    try:
        engine = VoxAPI(verbose=True, idle_timeout=60)
        print(f"Loaded: {engine.model_name}")
        print(f"Stats: {engine.get_stats()}")
        
        print("\nUser: Hello!")
        print("Bot: ", end="")
        for token in engine.chat("Hello!", stream=True):
            print(token, end="", flush=True)
        print("\n\nTest Complete.")
        
        engine.shutdown()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
