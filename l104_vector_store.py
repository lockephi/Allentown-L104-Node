#!/usr/bin/env python3
"""
L104 VECTOR STORE - EVO_48
═══════════════════════════════════════════════════════════════════════════════
High-quality vector embeddings and semantic search.
"""

import hashlib
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


GOD_CODE = 527.5184818492537
PHI = 1.618033988749895


@dataclass
class VectorEntry:
    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]


class L104VectorStore:
    """
    In-memory vector store with cosine similarity search.
    Designed for L104 resonance-aligned embeddings.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.entries: Dict[str, VectorEntry] = {}
        self._encoder = None

    def _get_encoder(self):
        """Lazy load sentence transformer."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                self._encoder = "hash"  # Fallback to hash-based
        return self._encoder

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text."""
        encoder = self._get_encoder()

        if encoder == "hash":
            # Fallback: deterministic hash-based embedding
            embedding = np.zeros(self.embedding_dim)
            text_hash = hashlib.sha256(text.encode()).digest()
            for i, byte in enumerate(text_hash):
                embedding[i % self.embedding_dim] += byte / 255.0
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        else:
            return encoder.encode(text, normalize_embeddings=True)

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add text to vector store."""
        entry_id = hashlib.md5(text.encode()).hexdigest()[:12]
        embedding = self._compute_embedding(text)

        self.entries[entry_id] = VectorEntry(
            id=entry_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {}
        )

        return entry_id

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[VectorEntry, float]]:
        """Search for similar texts."""
        query_embedding = self._compute_embedding(query)

        results = []
        for entry in self.entries.values():
            similarity = float(np.dot(query_embedding, entry.embedding))
            if similarity >= threshold:
                results.append((entry, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def delete(self, entry_id: str) -> bool:
        """Delete entry by ID."""
        if entry_id in self.entries:
            del self.entries[entry_id]
            return True
        return False

    def clear(self):
        """Clear all entries."""
        self.entries.clear()

    @property
    def size(self) -> int:
        return len(self.entries)


# Global instance
_vector_store: Optional[L104VectorStore] = None

def get_vector_store() -> L104VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = L104VectorStore()
    return _vector_store
