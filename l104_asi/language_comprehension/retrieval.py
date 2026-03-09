from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple
from collections import Counter

from .constants import PHI, TAU

class BM25Ranker:
    """BM25 (Best Matching 25) relevance ranking algorithm.

    Adapts Okapi BM25 with PHI-weighted parameters for
    optimal passage retrieval performance.

    v2.0: Added stopword filtering to prevent common words from
    inflating all document scores uniformly, destroying discrimination.
    """

    # High-frequency English stopwords that carry minimal discriminative signal
    _STOPWORDS = frozenset({
        'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'shall', 'can',
        'of', 'in', 'to', 'for', 'with', 'on', 'at', 'from', 'by',
        'an', 'as', 'or', 'if', 'it', 'its', 'no', 'not',
        'and', 'but', 'so', 'than', 'that', 'this', 'these', 'those',
        'he', 'she', 'they', 'we', 'you', 'who', 'which', 'what',
        'how', 'when', 'where', 'why', 'all', 'each', 'every',
    })

    def __init__(self, k1: float = None, b: float = None):
        self.k1 = k1 or PHI  # Term saturation (~1.618, typically 1.2-2.0)
        self.b = b or TAU     # Length normalization (~0.618, typically 0.75)
        self._corpus: List[List[str]] = []
        self._doc_freqs: Dict[str, int] = {}
        self._avg_doc_len = 0.0
        self._corpus_size = 0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 with stopword filtering."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [w for w in text.split() if len(w) > 1 and w not in self._STOPWORDS]

    def fit(self, documents: List[str]):
        """Index documents for BM25 ranking."""
        self._corpus = [self._tokenize(doc) for doc in documents]
        self._corpus_size = len(self._corpus)
        self._avg_doc_len = sum(len(d) for d in self._corpus) / max(self._corpus_size, 1)

        self._doc_freqs = {}
        for doc_tokens in self._corpus:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1

    def score(self, query: str) -> List[float]:
        """Score all documents against query."""
        query_tokens = self._tokenize(query)
        scores = []

        for doc_tokens in self._corpus:
            doc_score = 0.0
            doc_len = len(doc_tokens)
            tf_map = Counter(doc_tokens)

            for term in query_tokens:
                if term not in tf_map:
                    continue

                tf = tf_map[term]
                df = self._doc_freqs.get(term, 0)

                # IDF component
                idf = math.log((self._corpus_size - df + 0.5) / (df + 0.5) + 1)

                # TF component with length normalization
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_doc_len, 1))
                )

                doc_score += idf * tf_norm

            scores.append(doc_score)

        return scores

    def rank(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Rank documents by relevance to query.

        Returns: List of (doc_index, score) tuples.
        """
        scores = self.score(query)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 4b: SUBJECT DETECTOR — Auto-detect MMLU subject from question text
# ═══════════════════════════════════════════════════════════════════════════════
