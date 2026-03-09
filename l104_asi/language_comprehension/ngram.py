from __future__ import annotations

import re
from typing import Dict, List, Tuple, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from .knowledge_base import KnowledgeNode

from .constants import PHI, TAU

class NGramMatcher:
    """N-gram phrase matcher for multi-word concept recognition.

    Extracts bigrams and trigrams from text and matches them against
    a phrase index built from knowledge base facts. This catches
    multi-word concepts (e.g. "natural selection", "supply and demand")
    that single-word tokenization misses.
    """

    def __init__(self):
        self._phrase_index: Dict[str, List[Tuple[str, float]]] = {}  # phrase -> [(source_key, weight)]
        self._indexed = False

    @staticmethod
    def _extract_ngrams(text: str, n: int = 2) -> List[str]:
        """Extract n-grams from text."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        if len(words) < n:
            return []
        return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

    def build_index(self, knowledge_nodes: Dict[str, 'KnowledgeNode']):
        """Build phrase index from knowledge base nodes."""
        self._phrase_index.clear()
        for key, node in knowledge_nodes.items():
            # Index bigrams and trigrams from facts and definitions
            all_text = [node.definition] + node.facts
            for text in all_text:
                for n in (2, 3):
                    for ngram in self._extract_ngrams(text, n):
                        if ngram not in self._phrase_index:
                            self._phrase_index[ngram] = []
                        weight = 1.0 if n == 3 else 0.7  # Trigrams weighted higher
                        self._phrase_index[ngram].append((key, weight))
        self._indexed = True

    def match(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Match text against phrase index.

        Returns: List of (knowledge_node_key, cumulative_score) tuples.
        """
        if not self._indexed:
            return []

        scores: Dict[str, float] = defaultdict(float)
        # Extract bigrams and trigrams from query
        for n in (2, 3):
            for ngram in self._extract_ngrams(text, n):
                if ngram in self._phrase_index:
                    for key, weight in self._phrase_index[ngram]:
                        scores[key] += weight

        # Rank by cumulative score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def phrase_overlap_score(self, text_a: str, text_b: str) -> float:
        """Compute phrase-level overlap between two texts using shared n-grams."""
        bigrams_a = set(self._extract_ngrams(text_a, 2))
        bigrams_b = set(self._extract_ngrams(text_b, 2))
        trigrams_a = set(self._extract_ngrams(text_a, 3))
        trigrams_b = set(self._extract_ngrams(text_b, 3))

        bi_overlap = len(bigrams_a & bigrams_b) / max(len(bigrams_a | bigrams_b), 1)
        tri_overlap = len(trigrams_a & trigrams_b) / max(len(trigrams_a | trigrams_b), 1)

        # PHI-weighted blend: trigrams contribute more (higher precision)
        return bi_overlap * TAU + tri_overlap * PHI * 0.5


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 3: KNOWLEDGE GRAPH — 70-Subject Structured Knowledge Base
# ═══════════════════════════════════════════════════════════════════════════════

