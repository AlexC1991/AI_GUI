import os
from typing import List, Dict

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    
# Fallback dependencies for simple TF-IDF
import math
import re
from collections import Counter

class VectorIndex:
    """
    RAG Retrieval System.
    Uses ChromaDB if available, otherwise falls back to TF-IDF.
    """
    def __init__(self, session_id: str, db_path: str = "./data/chroma"):
        self.session_id = session_id
        self.chroma_client = None
        self.collection = None
        self.use_fallback = not CHROMA_AVAILABLE
        
        # Local cache for fallback mode
        self.documents = [] # List of (id, text, metadata)
        
        if CHROMA_AVAILABLE:
            try:
                self.chroma_client = chromadb.PersistentClient(path=db_path)
                self.collection = self.chroma_client.get_or_create_collection(
                    name=f"chat_{session_id}"
                )
            except Exception as e:
                print(f"[VectorIndex] ChromaDB init failed ({e}). Using Fallback.")
                self.use_fallback = True

    def index_message(self, msg_id: str, text: str, metadata: Dict = None):
        if not text: return
        
        if not self.use_fallback:
            try:
                self.collection.add(
                    documents=[text],
                    metadatas=[metadata or {}],
                    ids=[msg_id]
                )
            except:
                pass
        else:
            # Fallback storage
            self.documents.append({
                "id": msg_id,
                "text": text,
                "metadata": metadata
            })

    def retrieve(self, query: str, n_results: int = 5, exclude_ids: List[str] = []) -> List[Dict]:
        if not query: return []
        
        results = []
        
        if not self.use_fallback:
            try:
                res = self.collection.query(
                    query_texts=[query],
                    n_results=n_results * 2 # Fetch extra to filter
                )
                
                # Format results
                if res['documents']:
                    for i, doc in enumerate(res['documents'][0]):
                        mid = res['ids'][0][i]
                        meta = res['metadatas'][0][i]
                        if mid not in exclude_ids:
                            results.append({
                                "id": mid,
                                "content": doc,
                                "role": meta.get("role", "unknown"),
                                "metadata": meta
                            })
            except Exception as e:
                print(f"[VectorIndex] Query failed: {e}")
                
        else:
            # TF-IDF Fallback
            results = self._tfidf_search(query, n_results, exclude_ids)
            
        return results[:n_results]

    def _tfidf_search(self, query: str, n: int, exclude_ids: List[str]):
        """Simple memory-based TF-IDF search."""
        if not self.documents: return []
        
        def tokenize(text):
            return re.findall(r'\w+', text.lower())
            
        query_vec = Counter(tokenize(query))
        scores = []
        
        for doc in self.documents:
            if doc["id"] in exclude_ids: continue
            
            doc_vec = Counter(tokenize(doc["text"]))
            
            # Dot product
            dot = sum(query_vec[w] * doc_vec[w] for w in query_vec)
            
            # Magnitude
            mag_q = math.sqrt(sum(c**2 for c in query_vec.values()))
            mag_d = math.sqrt(sum(c**2 for c in doc_vec.values()))
            
            if mag_q * mag_d == 0:
                sim = 0
            else:
                sim = dot / (mag_q * mag_d)
                
            if sim > 0.1: # Threshold
                scores.append((sim, doc))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return [{
            "id": s[1]["id"],
            "content": s[1]["text"],
            "role": s[1]["metadata"].get("role"),
            "score": s[0]
        } for s in scores[:n]]
