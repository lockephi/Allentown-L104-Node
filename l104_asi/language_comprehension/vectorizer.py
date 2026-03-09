from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np

class TFIDFVectorizer:
    """TF-IDF vectorizer for semantic text representation.

    Uses term frequency–inverse document frequency weighting with
    sublinear TF scaling and L2 normalization.
    """

    def __init__(self, max_features: int = 10000, sublinear_tf: bool = True):
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Optional[np.ndarray] = None
        self._doc_count = 0
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenization."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        # Remove stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'of', 'in', 'to', 'for', 'with', 'on', 'at', 'from', 'by',
            'as', 'or', 'and', 'but', 'if', 'not', 'no', 'this', 'that',
            'it', 'its', 'he', 'she', 'they', 'we', 'you', 'i', 'my',
            'your', 'his', 'her', 'our', 'their', 'me', 'him', 'us', 'them',
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def fit(self, documents: List[str]):
        """Fit vectorizer on document corpus."""
        self._doc_count = len(documents)
        # Count document frequency per term
        df: Counter = Counter()
        tf_all: Counter = Counter()

        for doc in documents:
            tokens = self._tokenize(doc)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1
            for token in tokens:
                tf_all[token] += 1

        # Select top features by total frequency
        top_terms = [t for t, _ in tf_all.most_common(self.max_features)]
        self.vocabulary_ = {term: idx for idx, term in enumerate(top_terms)}

        # Compute IDF: log((1 + N) / (1 + df(t))) + 1
        n = self._doc_count
        self.idf_ = np.zeros(len(self.vocabulary_))
        for term, idx in self.vocabulary_.items():
            self.idf_[idx] = math.log((1 + n) / (1 + df.get(term, 0))) + 1

        self._fitted = True

    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to TF-IDF matrix."""
        if not self._fitted:
            raise RuntimeError("Vectorizer not fitted. Call fit() first.")

        matrix = np.zeros((len(documents), len(self.vocabulary_)))

        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            tf = Counter(tokens)
            for token, count in tf.items():
                if token in self.vocabulary_:
                    term_idx = self.vocabulary_[token]
                    # Sublinear TF: 1 + log(tf)
                    tf_val = (1 + math.log(count)) if self.sublinear_tf else count
                    matrix[doc_idx, term_idx] = tf_val * self.idf_[term_idx]

            # L2 normalize
            norm = np.linalg.norm(matrix[doc_idx])
            if norm > 0:
                matrix[doc_idx] /= norm

        return matrix

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)


class SemanticEncoder:
    """Semantic text encoder using TF-IDF + cosine similarity.

    Provides dense semantic representations for matching questions
    to knowledge passages.
    """

    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.vectorizer = TFIDFVectorizer(max_features=embedding_dim)
        self._corpus_vectors: Optional[np.ndarray] = None
        self._corpus_texts: List[str] = []
        self._corpus_labels: List[str] = []

    def index_corpus(self, texts: List[str], labels: Optional[List[str]] = None):
        """Index a corpus for retrieval."""
        self._corpus_texts = texts
        self._corpus_labels = labels or [f"doc_{i}" for i in range(len(texts))]
        self._corpus_vectors = self.vectorizer.fit_transform(texts)

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text to vector."""
        if not self.vectorizer._fitted:
            return np.zeros(self.embedding_dim)
        return self.vectorizer.transform([text])[0]

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Retrieve most similar documents to query.

        Returns: List of (text, label, similarity_score) tuples.
        """
        if self._corpus_vectors is None or len(self._corpus_texts) == 0:
            return []

        query_vec = self.encode(query)
        # Cosine similarity
        similarities = self._corpus_vectors @ query_vec
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((
                self._corpus_texts[idx],
                self._corpus_labels[idx],
                float(similarities[idx])
            ))
        return results

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        vec_a = self.encode(text_a)
        vec_b = self.encode(text_b)
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 2b: N-GRAM PHRASE MATCHER — Bigram/Trigram concept recognition
# ═══════════════════════════════════════════════════════════════════════════════
