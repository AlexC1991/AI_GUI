import gc
from typing import List, Dict

from .token_counter import TokenCounter
from .memory_store import MemoryStore
from .vector_index import VectorIndex

class ContextManager:
    """
    Orchestrates the construction of the prompt context.
    - Manages Token Budget
    - Injects Priority Memories
    - Retrieves RAG Context
    - Maintains Recent History
    """
    def __init__(self, llm_instance, template_handler, n_ctx: int, max_gen_tokens: int = 2048):
        self.llm = llm_instance
        self.template = template_handler
        self.n_ctx = n_ctx
        self.max_gen = max_gen_tokens
        
        # Prevent generation reservation from eating entire context
        if self.max_gen > self.n_ctx * 0.6:
            self.max_gen = int(self.n_ctx * 0.6)
            
        self.safety_margin = 128
        
        # Initialize Components
        self.token_counter = TokenCounter(self.llm)
        self.memory_store = MemoryStore() # Auto-generates session ID
        self.session_id = self.memory_store.session_id
        self.vector_index = VectorIndex(self.session_id)

    def set_session(self, session_id: str):
        """Switch session ID dynamically."""
        if session_id != self.session_id:
            self.session_id = session_id
            self.memory_store = MemoryStore(session_id)
            self.vector_index = VectorIndex(session_id)
            print(f"[ContextManager] Switched to session: {session_id}")

    def prepare_context(self, user_message: str, system_prompt: str) -> List[Dict]:
        """
        Builds the final messages list for the LLM.
        """
        # 1. Start with User Message
        user_msg = {"role": "user", "content": user_message}
        
        # 2. Add System Prompt
        sys_msg = {"role": "system", "content": system_prompt or "You are a helpful assistant."}
        
        # 3. Calculate Available Budget
        # Total - Gen Limit - System - User - Safety
        overhead = self.token_counter.count_messages([sys_msg, user_msg], self.template)
        budget = self.n_ctx - self.max_gen - overhead - self.safety_margin
        
        if budget < 0:
            print("[ContextManager] Warning: User message too long for context window!")
            budget = 0 # Let it truncate naturally or handled by LLM

        final_history = [sys_msg]
        used_tokens = 0
        
        # 4. Add Priority Memories (Limit to 30% of budget)
        priority_budget = int(budget * 0.3)
        priorities = self.memory_store.get_priorities()
        added_priorities = []
        
        for p in reversed(priorities): # Newest priority first
            count = self.token_counter.count(p["content"])
            if used_tokens + count <= priority_budget:
                added_priorities.insert(0, p) # prepend
                used_tokens += count
            else:
                break
        
        # 5. Add Recent Conversation (Continuity)
        # Always try to keep last 2 turns
        recents = self.memory_store.get_recent(6) # Fetch last few
        added_recents = []
        
        # Filter out ones already added as priority
        p_ids = {p["id"] for p in added_priorities}
        recents = [r for r in recents if r["id"] not in p_ids]
        
        for r in reversed(recents):
            count = self.token_counter.count(r["content"])
            if used_tokens + count <= budget:
                added_recents.insert(0, r)
                used_tokens += count
            else:
                break
                
        # 6. Fill remaining budget with RAG
        rag_budget = budget - used_tokens
        if rag_budget > 200: # Minimum useful RAG
            # Query vector DB using user message
            rag_docs = self.vector_index.retrieve(
                user_message, 
                n_results=3, 
                exclude_ids=p_ids.union({r["id"] for r in added_recents})
            )
            
            for doc in rag_docs:
                # Format as system/context injection
                content = f"[Context: {doc['content']}]"
                count = self.token_counter.count(content)
                
                if used_tokens + count <= budget:
                    # Insert RAG context after system prompt
                    final_history.insert(1, {"role": "system", "content": content})
                    used_tokens += count
        
        # Assemble Final
        # [System] + [RAG] + [Priorities] + [Recents] + [User]
        
        # Add priorities (after RAG)
        for p in added_priorities:
            final_history.append({"role": p["role"], "content": p["content"]})
            
        # Add recents
        for r in added_recents:
            final_history.append({"role": r["role"], "content": r["content"]})
            
        # Add current user msg
        
        # [Round 3 Fix] Inject Date as a separate System Message immediately before User Message
        # This ensures it's not lost in long context, without modifying user text.
        import datetime
        current_date_str = datetime.datetime.now().strftime("%A, %B %d, %Y")
        date_msg = {"role": "system", "content": f"Current Date: {current_date_str}"}
        
        final_history.append(date_msg)
        final_history.append(user_msg)
        
        return final_history

    def post_turn(self, user_content: str, assistant_content: str, llm_instance, priority: bool = False):
        """
        Process the turn after generation.
        1. Save to MemoryStore
        2. Index in VectorDB
        3. Flush KV Cache
        """
        # Save User Msg
        u_msg = self.memory_store.append_message("user", user_content, priority=priority)
        self.vector_index.index_message(u_msg["id"], user_content, {"role": "user", "priority": priority})
        
        # Save Assistant Msg
        a_msg = self.memory_store.append_message("assistant", assistant_content)
        self.vector_index.index_message(a_msg["id"], assistant_content, {"role": "assistant"})
        
        # Flush Memory
        if hasattr(llm_instance, 'reset'):
            llm_instance.reset()
        gc.collect()
        # if hasattr(llm_instance, '_ctx'): llm_instance._ctx.kv_cache_clear() # Deep clear if needed
