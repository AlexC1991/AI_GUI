import os
import time
import uuid
import msgpack
from typing import List, Dict, Optional

class MemoryStore:
    """
    Persists conversation history using MessagePack.
    Stores metadata like 'priority' tags.
    """
    def __init__(self, session_id: str = None, data_dir: str = "./data"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            
        if not session_id:
            # Generate new session ID based on timestamp
            session_id = f"session_{int(time.time())}"
            
        self.session_id = session_id
        self.file_path = os.path.join(self.data_dir, f"{self.session_id}.msgpack")
        self.messages: List[Dict] = []
        
        self.load()

    def load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "rb") as f:
                    self.messages = msgpack.unpack(f, raw=False)
            except Exception as e:
                print(f"[MemoryStore] Load failed: {e}")
                self.messages = []

    def save(self):
        try:
            with open(self.file_path, "wb") as f:
                msgpack.pack(self.messages, f)
        except Exception as e:
            print(f"[MemoryStore] Save failed: {e}")

    def append_message(self, role: str, content: str, priority: bool = False):
        msg = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "role": role,
            "content": content,
            "priority": priority
        }
        self.messages.append(msg)
        self.save()
        return msg

    def get_recent(self, n: int = 10) -> List[Dict]:
        return self.messages[-n:]

    def get_priorities(self) -> List[Dict]:
        """Return all messages marked as priority."""
        return [m for m in self.messages if m.get("priority")]

    def mark_priority(self, index: int):
        """Mark a message at specific index (from end if negative) as priority."""
        try:
            msg = self.messages[index]
            msg["priority"] = True
            self.save()
            print(f"[MemoryStore] Marked message {index} as priority.")
        except IndexError:
            pass
            
    def get_message_count(self) -> int:
        return len(self.messages)
