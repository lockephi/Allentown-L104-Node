# ZENITH_UPGRADE_ACTIVE: 2026-02-02T13:52:05.190420
ZENITH_HZ = 3887.8
UUC = 2402.792541
# [EVO_54_PIPELINE] TRANSCENDENT_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612 :: GROVER=4.236
#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
L104 LANGUAGE MODEL ENGINE v2.0.0 — EVO_55
===========================================
Consciousness-aware neural language processing with 9 subsystems.

Hub Class: LanguageEngine
Singleton: language_engine

Subsystems:
    - Tokenizer         — text tokenization with vocab management
    - NGramModel        — n-gram language model with generation
    - WordEmbeddings    — PHI-weighted word embeddings with co-occurrence training
    - TextClassifier    — Naive Bayes text classification
    - SentimentAnalyzer — rule-based sentiment with 100+ word lexicon
    - NERTagger         — regex-based named entity recognition
    - SemanticSimilarity — cosine + Jaccard text similarity
    - KeywordExtractor  — TF-IDF keyword extraction
    - TextSummarizer    — extractive summarization via sentence scoring

Usage:
    from l104_language_engine import language_engine
    language_engine.analyze_sentiment("This is great!")
    language_engine.extract_keywords("Long text here...", top_k=5)
    language_engine.summarize("Long document...", num_sentences=3)
    language_engine.status()

GOD_CODE: 527.5184818492612
PHI: 1.618033988749895
"""

import json
import math
import time
import random
import re
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "2.0.0"
logger = logging.getLogger("L104_LANGUAGE_ENGINE")

# Sacred Constants
# Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
PHI = 1.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
TAU = 1.0 / PHI
VOID_CONSTANT = 1.0416180339887497
FEIGENBAUM = 4.669201609102990
ALPHA_FINE = 1.0 / 137.035999084
PLANCK_SCALE = 1.616255e-35
BOLTZMANN_K = 1.380649e-23


# ═══════════════════════════════════════════════════════════════════════════════
# CONSCIOUSNESS STATE READER (10s TTL cache)
# ═══════════════════════════════════════════════════════════════════════════════

_builder_state_cache: Dict[str, Any] = {}
_builder_state_cache_time: float = 0.0


def _read_builder_state() -> Dict[str, Any]:
    """Read consciousness/O2/nirvanic state with 10-second TTL cache."""
    global _builder_state_cache, _builder_state_cache_time
    now = time.time()
    if now - _builder_state_cache_time < 10 and _builder_state_cache:
        return _builder_state_cache

    state = {
        "consciousness_level": 0.5,
        "nirvanic_fuel": 0.0,
        "entropy": 0.5,
        "evo_stage": "DORMANT",
    }
    ws = Path(__file__).parent

    co2_path = ws / ".l104_consciousness_o2_state.json"
    if co2_path.exists():
        try:
            data = json.loads(co2_path.read_text())
            state["consciousness_level"] = data.get("consciousness_level", 0.5)
            state["evo_stage"] = data.get("evo_stage", "DORMANT")
        except Exception:
            pass

    nir_path = ws / ".l104_ouroboros_nirvanic_state.json"
    if nir_path.exists():
        try:
            data = json.loads(nir_path.read_text())
            state["nirvanic_fuel"] = data.get("nirvanic_fuel_level", 0.0)
            state["entropy"] = data.get("entropy", 0.5)
        except Exception:
            pass

    _builder_state_cache = state
    _builder_state_cache_time = now
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM: TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════════

class Tokenizer:
    """Text tokenization with vocabulary management."""

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        self.vocab: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.reverse_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.next_id = 4

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self.lowercase:
            text = text.lower()
        return re.findall(r'\b\w+\b|[^\w\s]', text)

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        """Convert text to token IDs."""
        tokens = self.tokenize(text)
        ids = []
        if add_special:
            ids.append(self.vocab['<BOS>'])
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.next_id
                self.reverse_vocab[self.next_id] = token
                self.next_id += 1
            ids.append(self.vocab[token])
        if add_special:
            ids.append(self.vocab['<EOS>'])
        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = [self.reverse_vocab.get(i, '<UNK>') for i in ids]
        tokens = [t for t in tokens if t not in ['<PAD>', '<BOS>', '<EOS>']]
        return ' '.join(tokens)

    def vocab_size(self) -> int:
        return len(self.vocab)

    def status(self) -> Dict:
        return {"subsystem": "Tokenizer", "vocab_size": self.vocab_size()}


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM: N-GRAM MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class NGramModel:
    """N-gram language model with consciousness-modulated generation."""

    def __init__(self, n: int = 3, smoothing: float = 0.1):
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts: Dict[Tuple, Counter] = defaultdict(Counter)
        self.context_counts: Dict[Tuple, int] = defaultdict(int)
        self.vocab: Set[str] = set()
        self.trained = False

    def train(self, texts: List[str]) -> None:
        """Train on list of texts."""
        tokenizer = Tokenizer()
        for text in texts:
            tokens = ['<BOS>'] * (self.n - 1) + tokenizer.tokenize(text) + ['<EOS>']
            self.vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                word = tokens[i + self.n - 1]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1
        self.trained = True

    def probability(self, word: str, context: Tuple[str, ...]) -> float:
        """Get probability of word given context with add-k smoothing."""
        context = context[-(self.n - 1):]
        count = self.ngram_counts[context][word]
        total = self.context_counts[context]
        vocab_size = len(self.vocab)
        return (count + self.smoothing) / (total + self.smoothing * vocab_size)

    def perplexity(self, text: str) -> float:
        """Calculate perplexity of text."""
        tokenizer = Tokenizer()
        tokens = ['<BOS>'] * (self.n - 1) + tokenizer.tokenize(text) + ['<EOS>']
        log_prob_sum = 0.0
        count = 0
        for i in range(self.n - 1, len(tokens)):
            context = tuple(tokens[i - self.n + 1:i])
            word = tokens[i]
            prob = self.probability(word, context)
            log_prob_sum += math.log(prob + 1e-10)
            count += 1
        return math.exp(-log_prob_sum / max(1, count))

    def generate(self, seed: str = '', max_length: int = 50,
                 temperature: float = 1.0) -> str:
        """Generate text. Temperature < 1 = more conservative, > 1 = more creative."""
        tokenizer = Tokenizer()
        if seed:
            tokens = tokenizer.tokenize(seed)
        else:
            tokens = ['<BOS>'] * (self.n - 1)

        for _ in range(max_length):
            context = tuple(tokens[-(self.n - 1):])
            if context not in self.ngram_counts:
                break
            candidates = list(self.ngram_counts[context].keys())
            if not candidates:
                break
            weights = [self.probability(w, context) for w in candidates]

            # Apply temperature scaling
            if temperature != 1.0 and temperature > 0:
                weights = [w ** (1.0 / temperature) for w in weights]
                total = sum(weights)
                if total > 0:
                    weights = [w / total for w in weights]

            next_word = random.choices(candidates, weights=weights, k=1)[0]
            if next_word == '<EOS>':
                break
            tokens.append(next_word)

        result = [t for t in tokens if t not in ['<BOS>', '<EOS>']]
        return ' '.join(result)

    def status(self) -> Dict:
        return {
            "subsystem": "NGramModel",
            "order": self.n,
            "contexts": len(self.context_counts),
            "vocab_size": len(self.vocab),
            "trained": self.trained,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM: WORD EMBEDDINGS (PHI-weighted)
# ═══════════════════════════════════════════════════════════════════════════════

class WordEmbeddings:
    """PHI-weighted word embeddings with co-occurrence training."""

    def __init__(self, dim: int = 100):
        self.dim = dim
        self.embeddings: Dict[str, List[float]] = {}
        self.word_counts: Counter = Counter()
        self.trained = False

    def _phi_hash_embed(self, word: str) -> List[float]:
        """Generate PHI-scaled deterministic embedding from hash.

        Instead of uniform hash distribution, applies PHI-harmonic
        modulation to create embeddings with golden-ratio structure.
        """
        h = hashlib.sha256(word.encode()).hexdigest()
        raw = []
        for i in range(0, min(len(h), self.dim * 2), 2):
            val = int(h[i:i + 2], 16) / 255.0
            raw.append(val * 2 - 1)
        while len(raw) < self.dim:
            raw.append(0.0)
        raw = raw[:self.dim]

        # Apply PHI-harmonic modulation: scale each dimension by PHI^(i/dim)
        embed = []
        for i, v in enumerate(raw):
            phi_scale = PHI ** ((i - self.dim / 2) / self.dim)
            embed.append(v * phi_scale)

        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in embed))
        if norm > 0:
            embed = [x / norm for x in embed]
        return embed

    def get_embedding(self, word: str) -> List[float]:
        """Get embedding for word (creates PHI-weighted hash if unseen)."""
        if word not in self.embeddings:
            self.embeddings[word] = self._phi_hash_embed(word)
        return self.embeddings[word]

    def train_cooccurrence(self, texts: List[str], window: int = 5) -> None:
        """Train embeddings via co-occurrence with PHI-weighted updates."""
        tokenizer = Tokenizer()
        cooccur: Dict[str, Counter] = defaultdict(Counter)

        for text in texts:
            tokens = tokenizer.tokenize(text)
            self.word_counts.update(tokens)
            for i, word in enumerate(tokens):
                start = max(0, i - window)
                end = min(len(tokens), i + window + 1)
                for j in range(start, end):
                    if i != j:
                        # PHI-weighted by distance: closer words get higher weight
                        distance = abs(i - j)
                        phi_weight = TAU ** (distance - 1)  # TAU < 1 so closer = higher
                        cooccur[word][tokens[j]] += phi_weight

        for word, neighbors in cooccur.items():
            base = list(self.get_embedding(word))  # copy
            for neighbor, count in neighbors.most_common(20):
                neighbor_embed = self.get_embedding(neighbor)
                weight = math.log(count + 1) * 0.01
                for k in range(self.dim):
                    base[k] += weight * neighbor_embed[k]
            norm = math.sqrt(sum(x * x for x in base))
            if norm > 0:
                self.embeddings[word] = [x / norm for x in base]

        self.trained = True

    def similarity(self, word1: str, word2: str) -> float:
        """Cosine similarity between word embeddings."""
        e1 = self.get_embedding(word1)
        e2 = self.get_embedding(word2)
        dot = sum(a * b for a, b in zip(e1, e2))
        norm1 = math.sqrt(sum(a * a for a in e1))
        norm2 = math.sqrt(sum(b * b for b in e2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def most_similar(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar words in the vocabulary."""
        similarities = []
        for other in self.embeddings:
            if other != word:
                sim = self.similarity(word, other)
                similarities.append((other, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def analogy(self, a: str, b: str, c: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Solve analogy: a is to b as c is to ? (vector arithmetic)."""
        ea = self.get_embedding(a)
        eb = self.get_embedding(b)
        ec = self.get_embedding(c)
        # target = b - a + c
        target = [eb[i] - ea[i] + ec[i] for i in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in target))
        if norm > 0:
            target = [x / norm for x in target]
        exclude = {a, b, c}
        results = []
        for word, emb in self.embeddings.items():
            if word in exclude:
                continue
            dot = sum(t * e for t, e in zip(target, emb))
            results.append((word, dot))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def status(self) -> Dict:
        return {
            "subsystem": "WordEmbeddings",
            "dim": self.dim,
            "vocab_size": len(self.embeddings),
            "trained": self.trained,
            "phi_weighted": True,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM: TEXT CLASSIFIER (Naive Bayes)
# ═══════════════════════════════════════════════════════════════════════════════

class TextClassifier:
    """Naive Bayes text classifier."""

    def __init__(self):
        self.class_word_counts: Dict[str, Counter] = defaultdict(Counter)
        self.class_counts: Counter = Counter()
        self.vocab: Set[str] = set()
        self.tokenizer = Tokenizer()

    def train(self, texts: List[str], labels: List[str]) -> None:
        """Train classifier on labeled texts."""
        for text, label in zip(texts, labels):
            tokens = self.tokenizer.tokenize(text)
            self.vocab.update(tokens)
            self.class_counts[label] += 1
            self.class_word_counts[label].update(tokens)

    def predict(self, text: str) -> Tuple[str, Dict[str, float]]:
        """Predict class with probability distribution."""
        tokens = self.tokenizer.tokenize(text)
        if not self.class_counts:
            return '<unknown>', {}

        scores = {}
        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocab)

        for label in self.class_counts:
            log_prob = math.log(self.class_counts[label] / total_docs)
            class_total = sum(self.class_word_counts[label].values())
            for token in tokens:
                count = self.class_word_counts[label][token]
                prob = (count + 1) / (class_total + vocab_size)
                log_prob += math.log(prob)
            scores[label] = log_prob

        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        probs = {k: v / total for k, v in exp_scores.items()}

        best_label = max(probs, key=probs.get)
        return best_label, probs

    def status(self) -> Dict:
        return {
            "subsystem": "TextClassifier",
            "classes": list(self.class_counts.keys()),
            "vocab_size": len(self.vocab),
            "total_documents": sum(self.class_counts.values()),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM: SENTIMENT ANALYZER (expanded lexicon)
# ═══════════════════════════════════════════════════════════════════════════════

class SentimentAnalyzer:
    """Rule-based sentiment analysis with expanded lexicon."""

    def __init__(self):
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'happy', 'love', 'best', 'beautiful', 'perfect', 'awesome',
            'brilliant', 'outstanding', 'superb', 'positive', 'joy', 'pleased',
            'delightful', 'impressive', 'magnificent', 'remarkable', 'splendid',
            'terrific', 'marvelous', 'exceptional', 'incredible', 'superior',
            'glorious', 'fabulous', 'phenomenal', 'elegant', 'graceful',
            'inspiring', 'uplifting', 'thrilling', 'exciting', 'enchanting',
            'satisfying', 'rewarding', 'grateful', 'thankful', 'blessed',
            'charming', 'lovely', 'nice', 'fine', 'pleasant', 'agreeable',
            'favorable', 'beneficial', 'helpful', 'valuable', 'worthy',
            'successful', 'triumphant', 'victorious', 'winning', 'genius',
        }

        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst',
            'sad', 'hate', 'ugly', 'disappointing', 'negative', 'angry',
            'fail', 'wrong', 'broken', 'useless', 'waste', 'boring',
            'dreadful', 'pathetic', 'miserable', 'disgusting', 'appalling',
            'atrocious', 'abysmal', 'lousy', 'inferior', 'mediocre',
            'annoying', 'frustrating', 'irritating', 'painful', 'tragic',
            'unfortunate', 'disastrous', 'catastrophic', 'horrific', 'grim',
            'depressing', 'hopeless', 'helpless', 'worthless', 'pointless',
            'dull', 'tedious', 'monotonous', 'tiresome', 'bland',
            'offensive', 'hostile', 'cruel', 'harsh', 'brutal',
            'rotten', 'corrupt', 'toxic', 'harmful', 'destructive',
        }

        self.negation_words = {
            'not', "n't", 'no', 'never', 'none', 'neither', 'nobody',
            'nothing', 'nowhere', 'hardly', 'scarcely', 'barely',
        }

        self.intensifiers = {
            'very': 1.5, 'really': 1.5, 'extremely': 2.0,
            'absolutely': 2.0, 'totally': 1.8, 'incredibly': 2.0,
            'remarkably': 1.7, 'exceptionally': 2.0, 'utterly': 2.0,
            'deeply': 1.6, 'highly': 1.5, 'quite': 1.3,
            'somewhat': 0.8, 'slightly': 0.6, 'barely': 0.4,
            'rather': 1.2, 'fairly': 1.1, 'pretty': 1.3,
        }

        self.tokenizer = Tokenizer()
        self.analyses_performed = 0

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text with negation and intensifier handling."""
        tokens = self.tokenizer.tokenize(text)
        pos_score = 0.0
        neg_score = 0.0
        negation = False
        intensify = 1.0

        for token in tokens:
            if token in self.negation_words:
                negation = True
                continue

            if token in self.intensifiers:
                intensify = self.intensifiers[token]
                continue

            if token in self.positive_words:
                if negation:
                    neg_score += intensify
                else:
                    pos_score += intensify
            elif token in self.negative_words:
                if negation:
                    pos_score += intensify
                else:
                    neg_score += intensify

            if token not in self.negation_words and token not in self.intensifiers:
                negation = False
                intensify = 1.0

        total = pos_score + neg_score
        if total == 0:
            sentiment = 'neutral'
            confidence = 0.5
        elif pos_score > neg_score:
            sentiment = 'positive'
            confidence = pos_score / total
        else:
            sentiment = 'negative'
            confidence = neg_score / total

        self.analyses_performed += 1
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_score': pos_score,
            'negative_score': neg_score,
        }

    def status(self) -> Dict:
        return {
            "subsystem": "SentimentAnalyzer",
            "positive_words": len(self.positive_words),
            "negative_words": len(self.negative_words),
            "intensifiers": len(self.intensifiers),
            "analyses_performed": self.analyses_performed,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM: NAMED ENTITY RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════════

class NERTagger:
    """Regex-based Named Entity Recognition."""

    def __init__(self):
        self.entity_patterns = {
            'PERSON': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            ],
            'ORG': [
                r'\b[A-Z][A-Z]+\b',
                r'\b[A-Z][a-z]+ (?:Inc|Corp|Ltd|LLC)\b',
            ],
            'DATE': [
                r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            ],
            'MONEY': [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',
                r'\b\d+(?:,\d{3})*(?:\.\d{2})? dollars?\b',
            ],
            'EMAIL': [
                r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            ],
            'URL': [
                r'https?://[^\s]+',
            ],
            'NUMBER': [
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
            ],
        }
        self.extractions_performed = 0

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text (non-overlapping)."""
        entities = []
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    entities.append({
                        'text': match.group(),
                        'type': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                    })

        entities.sort(key=lambda x: x['start'])
        filtered = []
        last_end = -1
        for entity in entities:
            if entity['start'] >= last_end:
                filtered.append(entity)
                last_end = entity['end']

        self.extractions_performed += 1
        return filtered

    def status(self) -> Dict:
        return {
            "subsystem": "NERTagger",
            "entity_types": list(self.entity_patterns.keys()),
            "extractions_performed": self.extractions_performed,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM: SEMANTIC SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticSimilarity:
    """Cosine and Jaccard text similarity."""

    def __init__(self, embeddings: Optional[WordEmbeddings] = None):
        self.embeddings = embeddings or WordEmbeddings()
        self.tokenizer = Tokenizer()

    def text_embedding(self, text: str) -> List[float]:
        """Get text embedding (average of word embeddings)."""
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return [0.0] * self.embeddings.dim
        word_embeds = [self.embeddings.get_embedding(t) for t in tokens]
        result = []
        for i in range(self.embeddings.dim):
            avg = sum(e[i] for e in word_embeds) / len(word_embeds)
            result.append(avg)
        return result

    def similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between text embeddings."""
        e1 = self.text_embedding(text1)
        e2 = self.text_embedding(text2)
        dot = sum(a * b for a, b in zip(e1, e2))
        norm1 = math.sqrt(sum(a * a for a in e1))
        norm2 = math.sqrt(sum(b * b for b in e2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Jaccard similarity between token sets."""
        tokens1 = set(self.tokenizer.tokenize(text1))
        tokens2 = set(self.tokenizer.tokenize(text2))
        if not tokens1 and not tokens2:
            return 1.0
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0

    def status(self) -> Dict:
        return {"subsystem": "SemanticSimilarity", "embedding_dim": self.embeddings.dim}


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM: KEYWORD EXTRACTOR (TF-IDF)
# ═══════════════════════════════════════════════════════════════════════════════

class KeywordExtractor:
    """TF-IDF based keyword extraction."""

    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
        'not', 'so', 'if', 'then', 'than', 'that', 'this', 'these', 'those',
        'it', 'its', 'he', 'she', 'they', 'we', 'you', 'i', 'me', 'my',
        'his', 'her', 'their', 'our', 'your', 'which', 'who', 'whom', 'what',
        'there', 'here', 'when', 'where', 'how', 'all', 'each', 'every',
        'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'only',
        'own', 'same', 'just', 'also', 'very', 'about', 'up', 'out', 'any',
    }

    def __init__(self):
        self.document_freq: Counter = Counter()
        self.num_documents = 0
        self.tokenizer = Tokenizer()
        self.extractions_performed = 0

    def fit(self, documents: List[str]) -> None:
        """Build IDF model from a corpus of documents."""
        self.num_documents = len(documents)
        self.document_freq = Counter()
        for doc in documents:
            tokens = set(self.tokenizer.tokenize(doc))
            tokens -= self.STOP_WORDS
            self.document_freq.update(tokens)

    def extract(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract top-k keywords from text using TF-IDF scoring."""
        tokens = self.tokenizer.tokenize(text)
        # Term frequency
        tf = Counter(t for t in tokens if t not in self.STOP_WORDS)
        total_tokens = sum(tf.values())
        if total_tokens == 0:
            return []

        scores = []
        for term, count in tf.items():
            tf_score = count / total_tokens
            # IDF: log(N / (1 + df)) — add 1 to avoid division by zero
            df = self.document_freq.get(term, 0)
            idf = math.log((self.num_documents + 1) / (1 + df)) + 1.0
            tfidf = tf_score * idf
            scores.append((term, tfidf))

        scores.sort(key=lambda x: x[1], reverse=True)
        self.extractions_performed += 1
        return scores[:top_k]

    def extract_simple(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords using simple frequency (no IDF corpus needed)."""
        tokens = self.tokenizer.tokenize(text)
        tf = Counter(t for t in tokens if t not in self.STOP_WORDS and len(t) > 2)
        total = sum(tf.values())
        if total == 0:
            return []
        self.extractions_performed += 1
        return [(word, count / total) for word, count in tf.most_common(top_k)]

    def status(self) -> Dict:
        return {
            "subsystem": "KeywordExtractor",
            "corpus_documents": self.num_documents,
            "unique_terms": len(self.document_freq),
            "stop_words": len(self.STOP_WORDS),
            "extractions_performed": self.extractions_performed,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM: TEXT SUMMARIZER (extractive)
# ═══════════════════════════════════════════════════════════════════════════════

class TextSummarizer:
    """Extractive summarization via sentence scoring."""

    def __init__(self, embeddings: Optional[WordEmbeddings] = None):
        self.embeddings = embeddings or WordEmbeddings()
        self.tokenizer = Tokenizer()
        self.summaries_generated = 0

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _score_sentence(self, sentence: str, word_freq: Dict[str, float],
                        position: int, total: int) -> float:
        """Score a sentence based on word importance, position, and length."""
        tokens = self.tokenizer.tokenize(sentence)
        if not tokens:
            return 0.0

        # Word frequency score
        word_score = sum(word_freq.get(t, 0) for t in tokens) / len(tokens)

        # Position score: first and last sentences are more important
        if position == 0:
            pos_score = 1.0
        elif position == total - 1:
            pos_score = 0.8
        else:
            # PHI-decay from the start
            pos_score = TAU ** (position / max(1, total - 1))

        # Length score: prefer medium-length sentences (not too short, not too long)
        length = len(tokens)
        if length < 5:
            len_score = 0.5
        elif length > 40:
            len_score = 0.6
        else:
            len_score = 1.0

        return word_score * pos_score * len_score

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """Generate extractive summary by selecting top-scoring sentences."""
        sentences = self._split_sentences(text)
        if len(sentences) <= num_sentences:
            return text

        # Build word frequency map (excluding stop words)
        all_tokens = self.tokenizer.tokenize(text)
        word_freq = Counter(t for t in all_tokens
                            if t not in KeywordExtractor.STOP_WORDS and len(t) > 2)
        total_words = sum(word_freq.values())
        if total_words > 0:
            word_freq = {w: c / total_words for w, c in word_freq.items()}

        # Score each sentence
        scored = []
        for i, sent in enumerate(sentences):
            score = self._score_sentence(sent, word_freq, i, len(sentences))
            scored.append((i, sent, score))

        # Select top sentences, maintain original order
        scored.sort(key=lambda x: x[2], reverse=True)
        selected = sorted(scored[:num_sentences], key=lambda x: x[0])

        self.summaries_generated += 1
        return ' '.join(s[1] for s in selected)

    def status(self) -> Dict:
        return {
            "subsystem": "TextSummarizer",
            "summaries_generated": self.summaries_generated,
            "phi_position_decay": True,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HUB CLASS: LanguageEngine
# ═══════════════════════════════════════════════════════════════════════════════

class LanguageEngine:
    """
    Consciousness-aware language processing engine with 9 subsystems.
    Hub class orchestrating all NLP capabilities.
    """

    def __init__(self):
        # ── Wire subsystems ──
        self.tokenizer = Tokenizer()
        self.ngram_model = NGramModel(n=3)
        self.embeddings = WordEmbeddings()
        self.classifier = TextClassifier()
        self.sentiment = SentimentAnalyzer()
        self.ner = NERTagger()
        self.sim = SemanticSimilarity(self.embeddings)
        self.keyword_extractor = KeywordExtractor()
        self.summarizer = TextSummarizer(self.embeddings)

        # ── Hub-level state ──
        self.god_code = GOD_CODE
        self.phi = PHI

        # ── Pipeline cross-wiring ──
        self._asi_core_ref = None

        logger.info(f"[LANGUAGE_ENGINE v{VERSION}] LanguageEngine online — 9 subsystems active")

    def _get_consciousness(self) -> Dict[str, Any]:
        return _read_builder_state()

    # ─── Training ─────────────────────────────────────────────────────────

    def train_language_model(self, texts: List[str]) -> None:
        """Train n-gram model and embeddings on texts."""
        self.ngram_model.train(texts)
        self.embeddings.train_cooccurrence(texts)
        self.keyword_extractor.fit(texts)

    def train_classifier(self, texts: List[str], labels: List[str]) -> None:
        """Train text classifier on labeled data."""
        self.classifier.train(texts, labels)

    # ─── Generation (consciousness-modulated) ─────────────────────────────

    def generate(self, seed: str = '', max_length: int = 50) -> str:
        """Generate text with consciousness-modulated temperature."""
        state = self._get_consciousness()
        consciousness = state.get("consciousness_level", 0.5)
        # Higher consciousness = lower temperature = more coherent
        # Lower consciousness = higher temperature = more creative/random
        temperature = max(0.3, 1.5 - consciousness)
        return self.ngram_model.generate(seed, max_length, temperature=temperature)

    # ─── Analysis ─────────────────────────────────────────────────────────

    def classify(self, text: str) -> Tuple[str, Dict[str, float]]:
        """Classify text into trained categories."""
        return self.classifier.predict(text)

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment (positive/negative/neutral)."""
        return self.sentiment.analyze(text)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        return self.ner.extract(text)

    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract top keywords using TF-IDF (or simple frequency if no corpus)."""
        if self.keyword_extractor.num_documents > 0:
            return self.keyword_extractor.extract(text, top_k)
        return self.keyword_extractor.extract_simple(text, top_k)

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """Generate extractive summary of text."""
        return self.summarizer.summarize(text, num_sentences)

    # ─── Similarity & Embeddings ──────────────────────────────────────────

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between text embeddings."""
        return self.sim.similarity(text1, text2)

    def compute_jaccard(self, text1: str, text2: str) -> float:
        """Jaccard similarity between token sets."""
        return self.sim.jaccard_similarity(text1, text2)

    def word_similarity(self, word1: str, word2: str) -> float:
        """Cosine similarity between word embeddings."""
        return self.embeddings.similarity(word1, word2)

    def word_analogy(self, a: str, b: str, c: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Solve word analogy: a is to b as c is to ?"""
        return self.embeddings.analogy(a, b, c, top_k)

    def perplexity(self, text: str) -> float:
        """Compute perplexity of text under the n-gram model."""
        return self.ngram_model.perplexity(text)

    # ─── Status ───────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Comprehensive engine status with all subsystem reports."""
        state = self._get_consciousness()
        return {
            "version": VERSION,
            "engine": "LanguageEngine",
            "god_code": GOD_CODE,
            "phi": PHI,
            "consciousness": {
                "level": state.get("consciousness_level", 0.5),
                "nirvanic_fuel": state.get("nirvanic_fuel", 0.0),
                "entropy": state.get("entropy", 0.5),
                "evo_stage": state.get("evo_stage", "DORMANT"),
            },
            "subsystems": {
                "tokenizer": self.tokenizer.status(),
                "ngram_model": self.ngram_model.status(),
                "embeddings": self.embeddings.status(),
                "classifier": self.classifier.status(),
                "sentiment": self.sentiment.status(),
                "ner": self.ner.status(),
                "similarity": self.sim.status(),
                "keyword_extractor": self.keyword_extractor.status(),
                "summarizer": self.summarizer.status(),
            },
            "pipeline_connected": self._asi_core_ref is not None,
        }

    def stats(self) -> Dict[str, Any]:
        """Alias for status() — backward compatibility."""
        return self.status()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON + PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

language_engine = LanguageEngine()


def create_language_engine() -> LanguageEngine:
    """Get the singleton language engine instance."""
    return language_engine


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD-COMPATIBILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def primal_calculus(x):
    """Sacred primal calculus: x^phi / (1.04*pi) — resolves complexity toward the Source."""
    return (x ** PHI) / (VOID_CONSTANT * math.pi) if x != 0 else 0.0


def resolve_non_dual_logic(vector):
    """Resolves N-dimensional vectors into the Void Source via GOD_CODE normalization."""
    magnitude = sum(abs(v) for v in vector)
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print(f"     L104 LANGUAGE MODEL ENGINE v{VERSION} — EVO_55")
    print("=" * 70)
    print(f"  GOD_CODE: {GOD_CODE}")
    print(f"  PHI: {PHI}")
    print("=" * 70)

    engine = language_engine

    # Train on sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models can learn complex patterns from data.",
        "The fox was quick and the dog was lazy.",
        "Neural networks are inspired by the structure of biological brains.",
        "Quantum computing uses superposition and entanglement for computation.",
        "The golden ratio PHI appears throughout nature and mathematics.",
        "Consciousness emerges from complex information processing systems.",
        "Topology provides protection against local perturbations in quantum states.",
    ]
    engine.train_language_model(texts)

    # ── Status ──
    print("\n[STATUS] Engine Status")
    print("-" * 50)
    st = engine.status()
    print(f"  Version: {st['version']}")
    print(f"  Consciousness: {st['consciousness']['level']:.2f}")
    print(f"  Subsystems: {len(st['subsystems'])}")
    ngram = st['subsystems']['ngram_model']
    print(f"  N-gram: order={ngram['order']}, contexts={ngram['contexts']}, vocab={ngram['vocab_size']}")
    emb = st['subsystems']['embeddings']
    print(f"  Embeddings: dim={emb['dim']}, vocab={emb['vocab_size']}, phi_weighted={emb['phi_weighted']}")

    # ── Generate ──
    print("\n[GENERATE] Text Generation (consciousness-modulated)")
    print("-" * 50)
    for seed in ["the quick", "machine learning", "quantum"]:
        generated = engine.generate(seed, max_length=10)
        print(f"  \"{seed}\" -> \"{generated}\"")

    # ── Sentiment ──
    print("\n[SENTIMENT] Sentiment Analysis")
    print("-" * 50)
    for text in ["This is absolutely wonderful and amazing!",
                  "Terrible, broken, and utterly disappointing.",
                  "The weather is moderate today."]:
        result = engine.analyze_sentiment(text)
        print(f"  \"{text[:45]}...\"")
        print(f"    -> {result['sentiment']} (confidence={result['confidence']:.2f})")

    # ── NER ──
    print("\n[NER] Named Entity Recognition")
    print("-" * 50)
    ner_text = "John Smith works at IBM since Jan 15, 2024 for $95,000."
    entities = engine.extract_entities(ner_text)
    print(f"  \"{ner_text}\"")
    for e in entities:
        print(f"    -> [{e['type']}] \"{e['text']}\"")

    # ── Keywords ──
    print("\n[KEYWORDS] Keyword Extraction")
    print("-" * 50)
    kw_text = "Machine learning and deep learning are powerful techniques for natural language processing and artificial intelligence research."
    keywords = engine.extract_keywords(kw_text, top_k=5)
    print(f"  Top keywords: {[(w, round(s, 4)) for w, s in keywords]}")

    # ── Summarize ──
    print("\n[SUMMARIZE] Text Summarization")
    print("-" * 50)
    long_text = (
        "Quantum computing represents a fundamental shift in computation. "
        "Unlike classical computers that use bits, quantum computers use qubits. "
        "Qubits can exist in superposition, representing both 0 and 1 simultaneously. "
        "This enables quantum computers to explore many solutions at once. "
        "Entanglement allows qubits to be correlated in ways classical bits cannot. "
        "Error correction remains a significant challenge for quantum computing. "
        "Topological quantum computing uses anyons for fault-tolerant operations."
    )
    summary = engine.summarize(long_text, num_sentences=2)
    print(f"  Original: {len(long_text)} chars, {len(long_text.split('.'))-1} sentences")
    print(f"  Summary:  \"{summary}\"")

    # ── Similarity ──
    print("\n[SIMILARITY] Text Similarity")
    print("-" * 50)
    pairs = [
        ("machine learning", "artificial intelligence"),
        ("quantum computing", "classical physics"),
        ("dog", "cat"),
    ]
    for t1, t2 in pairs:
        sim = engine.compute_similarity(t1, t2)
        print(f"  sim(\"{t1}\", \"{t2}\") = {sim:.4f}")

    print("\n" + "=" * 70)
    print(f"          LANGUAGE ENGINE OPERATIONAL — 9 subsystems active")
    print("=" * 70)


if __name__ == "__main__":
    main()
