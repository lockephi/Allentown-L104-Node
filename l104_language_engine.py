# ZENITH_UPGRADE_ACTIVE: 2026-03-06T23:50:22.897730
ZENITH_HZ = 3887.8
UUC = 2301.215661
# [EVO_54_PIPELINE] TRANSCENDENT_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612 :: GROVER=4.236
#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
L104 LANGUAGE MODEL ENGINE v7.0.0 — EVO_57
===========================================
Consciousness-aware neural language processing with 19 subsystems.

Hub Class: LanguageEngine
Singleton: language_engine

Subsystems (9 core + 4 DeepNLU v2.1.0):
    - Tokenizer              — text tokenization with vocab management
    - NGramModel             — n-gram language model with generation
    - WordEmbeddings         — PHI-weighted word embeddings with co-occurrence training
    - TextClassifier         — Naive Bayes text classification
    - SentimentAnalyzer      — rule-based sentiment with 100+ word lexicon
    - NERTagger              — regex-based named entity recognition
    - SemanticSimilarity     — cosine + Jaccard text similarity
    - KeywordExtractor       — TF-IDF keyword extraction
    - TextSummarizer         — extractive summarization via sentence scoring
    - TemporalReasoner       — tense detection, event ordering, duration/frequency (DeepNLU L10)
    - CausalReasoner         — cause-effect extraction, causal chains, counterfactuals (DeepNLU L11)
    - ContextualDisambiguator — WSD-lite, polysemy resolution, metaphor detection (DeepNLU L12)
    - QuerySynthesisPipeline — 8-archetype query generation from NLU output (DeepNLU L14)
    - QueryDecomposer        — multi-hop query decomposition (DeepNLU L15)
    - QueryExpander          — synonym/hypernym/morphological expansion (DeepNLU L16)
    - QueryClassifier        — Bloom's taxonomy + domain classification (DeepNLU L17)
    - TextualEntailmentEngine— NLI: entailment/contradiction/neutral (DeepNLU L18)  ★ NEW
    - FigurativeLanguageProcessor — idioms, similes, irony, hyperbole (DeepNLU L19)  ★ NEW
    - InformationDensityAnalyzer  — surprisal, diversity, redundancy (DeepNLU L20)   ★ NEW

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

VERSION = "6.0.0"
logger = logging.getLogger("L104_LANGUAGE_ENGINE")

try:
    from l104_asi.pipeline_telemetry import PipelineTelemetry
except ImportError:
    PipelineTelemetry = None

try:
    from l104_asi.pipeline_circuit_breaker import PipelineCircuitBreaker
except ImportError:
    PipelineCircuitBreaker = None

# DeepNLU v2.3.0 integration — temporal, causal, disambiguation, query synthesis,
# decomposer, expander, classifier
try:
    from l104_asi.deep_nlu import (
        TemporalReasoner, CausalReasoner, ContextualDisambiguator,
        QuerySynthesisPipeline, QueryType, SynthesizedQuery,
        QueryDecomposer, QueryExpander, QueryClassifier,
        BloomLevel, QueryDomain, AnswerFormat,
        TextualEntailmentEngine, EntailmentLabel,
        FigurativeLanguageProcessor, FigurativeType,
        InformationDensityAnalyzer,
    )
    _DEEP_NLU_AVAILABLE = True
except ImportError:
    _DEEP_NLU_AVAILABLE = False

# Sacred Constants
# Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
PHI = 1.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
TAU = 1.0 / PHI
VOID_CONSTANT = 1.0416180339887497
FEIGENBAUM = 4.669201609102990
ALPHA_FINE = 1.0 / 137.035999084
LATTICE_THERMAL_FRICTION = -(ALPHA_FINE * PHI) / (2 * math.pi * 104)
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
    """Rule-based sentiment analysis with expanded lexicon and aspect-based sentiment.

    v4.0.0: Aspect-based sentiment extraction + Plutchik emotion detection.
    """

    EMOTION_LEXICON = {
        'joy': {'happy', 'joy', 'joyful', 'delighted', 'pleased', 'glad', 'cheerful', 'elated', 'ecstatic', 'thrilled'},
        'sadness': {'sad', 'unhappy', 'depressed', 'miserable', 'sorrowful', 'melancholy', 'gloomy', 'lonely', 'heartbroken'},
        'anger': {'angry', 'furious', 'enraged', 'mad', 'irritated', 'outraged', 'hostile', 'livid', 'resentful'},
        'fear': {'afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous', 'frightened', 'dread', 'panic'},
        'surprise': {'surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'startled', 'bewildered'},
        'disgust': {'disgusted', 'repulsed', 'revolted', 'appalled', 'sickened', 'nauseated'},
        'trust': {'trust', 'confident', 'reliable', 'faithful', 'loyal', 'honest', 'dependable', 'sincere'},
        'anticipation': {'anticipate', 'expect', 'hope', 'eager', 'excited', 'enthusiastic', 'looking'},
    }

    ASPECT_MARKERS = {
        'quality': {'quality', 'performance', 'standard', 'grade', 'caliber'},
        'price': {'price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'worth'},
        'service': {'service', 'support', 'help', 'assistance', 'customer'},
        'design': {'design', 'look', 'appearance', 'style', 'aesthetic', 'beautiful', 'ugly'},
        'speed': {'speed', 'fast', 'slow', 'quick', 'performance', 'responsive'},
        'reliability': {'reliable', 'stable', 'consistent', 'broken', 'crash', 'bug', 'error'},
        'ease_of_use': {'easy', 'simple', 'intuitive', 'complicated', 'confusing', 'difficult'},
    }

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
        """Analyze sentiment of text with negation, intensifier handling, and emotions."""
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

        # Emotion detection (Plutchik's wheel)
        emotions = self._detect_emotions(tokens)

        # Aspect-based sentiment
        aspects = self._extract_aspect_sentiment(tokens)

        self.analyses_performed += 1
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_score': pos_score,
            'negative_score': neg_score,
            'emotions': emotions,
            'aspects': aspects,
        }

    def _detect_emotions(self, tokens: List[str]) -> Dict[str, float]:
        """Detect emotions from Plutchik's wheel (8 primary emotions)."""
        emotions = {}
        for emotion, lexicon in self.EMOTION_LEXICON.items():
            count = sum(1 for t in tokens if t in lexicon)
            if count > 0:
                emotions[emotion] = round(min(1.0, count * 0.3), 3)
        return emotions

    def _extract_aspect_sentiment(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """Extract aspect-based sentiment: identify which aspects are praised/criticized."""
        aspects = []
        token_set = set(tokens)
        for aspect, markers in self.ASPECT_MARKERS.items():
            overlap = markers & token_set
            if overlap:
                # Look at sentiment words near the aspect markers
                aspect_pos = 0.0
                aspect_neg = 0.0
                for i, t in enumerate(tokens):
                    if t in overlap:
                        # Check window of ±4 tokens around aspect marker
                        window = tokens[max(0, i - 4):i + 5]
                        for w in window:
                            if w in self.positive_words:
                                aspect_pos += 1
                            elif w in self.negative_words:
                                aspect_neg += 1
                if aspect_pos + aspect_neg > 0:
                    aspect_sent = 'positive' if aspect_pos > aspect_neg else 'negative'
                else:
                    aspect_sent = 'neutral'
                aspects.append({
                    'aspect': aspect,
                    'sentiment': aspect_sent,
                    'markers_found': list(overlap),
                    'positive_score': aspect_pos,
                    'negative_score': aspect_neg,
                })
        return aspects

    def status(self) -> Dict:
        return {
            "subsystem": "SentimentAnalyzer",
            "version": "2.0.0",
            "positive_words": len(self.positive_words),
            "negative_words": len(self.negative_words),
            "intensifiers": len(self.intensifiers),
            "emotion_categories": len(self.EMOTION_LEXICON),
            "aspect_categories": len(self.ASPECT_MARKERS),
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
    Consciousness-aware language processing engine with 19 subsystems.
    Hub class orchestrating all NLP capabilities.

    v7.0.0: DeepNLU v2.3.0 integration — textual entailment engine,
    figurative language processor, information density analyzer as
    subsystems 17-19.

    v6.0.0: DeepNLU v2.3.0 integration — query decomposer, query expander,
    and query classifier added as subsystems 14-16.

    v5.0.0: DeepNLU v2.1.0 integration — query synthesis pipeline
    added as subsystem 13 (8-archetype query generation from NLU).

    v4.0.0: DeepNLU v2.0.0 integration — temporal reasoning, causal
    reasoning, and contextual disambiguation as subsystems 10-12.
    """

    VERSION = "7.0.0"

    def __init__(self):
        # ── Wire core subsystems (1-9) ──
        self.tokenizer = Tokenizer()
        self.ngram_model = NGramModel(n=3)
        self.embeddings = WordEmbeddings()
        self.classifier = TextClassifier()
        self.sentiment = SentimentAnalyzer()
        self.ner = NERTagger()
        self.sim = SemanticSimilarity(self.embeddings)
        self.keyword_extractor = KeywordExtractor()
        self.summarizer = TextSummarizer(self.embeddings)

        # ── DeepNLU v2.3.0 subsystems (10-19) ──
        self._deep_nlu_available = _DEEP_NLU_AVAILABLE
        if _DEEP_NLU_AVAILABLE:
            self.temporal = TemporalReasoner()
            self.causal = CausalReasoner()
            self.disambiguator = ContextualDisambiguator()
            self.query_pipeline = QuerySynthesisPipeline()
            self.decomposer = QueryDecomposer()
            self.expander = QueryExpander()
            self.query_classifier = QueryClassifier()
            self.entailment = TextualEntailmentEngine()
            self.figurative = FigurativeLanguageProcessor()
            self.density_analyzer = InformationDensityAnalyzer()
        else:
            self.temporal = None
            self.causal = None
            self.disambiguator = None
            self.query_pipeline = None
            self.decomposer = None
            self.expander = None
            self.query_classifier = None
            self.entailment = None
            self.figurative = None
            self.density_analyzer = None

        # ── Hub-level state ──
        self.god_code = GOD_CODE
        self.phi = PHI

        # ── Pipeline cross-wiring ──
        self._asi_core_ref = None

        # ── Pipeline telemetry & circuit breaker ──
        self._telemetry = None
        self._cb = None
        try:
            if PipelineTelemetry is not None:
                self._telemetry = PipelineTelemetry("language_engine")
        except Exception:
            pass
        try:
            if PipelineCircuitBreaker is not None:
                self._cb = PipelineCircuitBreaker("language_engine", failure_threshold=5, reset_timeout=30)
        except Exception:
            pass

        n_subs = 19 if self._deep_nlu_available else 9
        logger.info("[LANGUAGE_ENGINE v%s] online — %d subsystems, deep_nlu=%s, telemetry=%s, cb=%s",
                    VERSION, n_subs, self._deep_nlu_available, self._telemetry is not None, self._cb is not None)

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

    # ─── DeepNLU v2.0.0 — Temporal / Causal / Disambiguation ─────────────

    def analyze_temporal(self, text: str) -> Dict[str, Any]:
        """Analyze temporal structure: tenses, event ordering, duration, frequency."""
        if not self._deep_nlu_available or self.temporal is None:
            return {"error": "DeepNLU v2.0.0 not available", "tenses": {}, "events": [], "temporal_expressions": []}
        return self.temporal.analyze(text)

    def analyze_causal(self, text: str) -> Dict[str, Any]:
        """Extract causal relationships: cause-effect pairs, chains, counterfactuals."""
        if not self._deep_nlu_available or self.causal is None:
            return {"error": "DeepNLU v2.0.0 not available", "relations": [], "chains": [], "counterfactuals": []}
        return self.causal.analyze(text)

    def disambiguate(self, text: str) -> Dict[str, Any]:
        """Resolve word-sense ambiguity and detect metaphors."""
        if not self._deep_nlu_available or self.disambiguator is None:
            return {"error": "DeepNLU v2.0.0 not available", "disambiguations": [], "metaphors": []}
        return self.disambiguator.disambiguate(text)

    def deep_analyze(self, text: str) -> Dict[str, Any]:
        """Full 19-subsystem analysis: sentiment + NER + keywords + temporal + causal +
        disambiguation + queries + decomposition + expansion + classification +
        entailment + figurative + density."""
        result = {
            "sentiment": self.analyze_sentiment(text),
            "entities": self.extract_entities(text),
            "keywords": self.extract_keywords(text, top_k=10),
        }
        if self._deep_nlu_available:
            result["temporal"] = self.analyze_temporal(text)
            result["causal"] = self.analyze_causal(text)
            result["disambiguation"] = self.disambiguate(text)
            result["query_synthesis"] = self.synthesize_queries(text, max_queries=10)
            result["query_decomposition"] = self.decompose_query(text)
            result["query_expansion"] = self.expand_query(text)
            result["query_classification"] = self.classify_query(text)
            result["figurative"] = self.analyze_figurative(text)
            result["density"] = self.analyze_density(text)
            result["deep_nlu_version"] = "2.3.0"
        else:
            result["deep_nlu_version"] = None
        return result

    def synthesize_queries(self, text: str, *, max_queries: int = 25,
                           min_confidence: float = 0.3,
                           query_types: Optional[Set] = None) -> Dict[str, Any]:
        """Synthesize diverse queries from text via 8-archetype NLU pipeline.  ★ NEW v5.0.0

        Args:
            text: Source text to generate queries from.
            max_queries: Maximum queries to return.
            min_confidence: Minimum confidence threshold.
            query_types: Restrict to specific QueryType archetypes.

        Returns:
            Dict with 'queries', 'total', 'archetype_distribution', NLU metadata.
        """
        if not self._deep_nlu_available or self.query_pipeline is None:
            return {"error": "DeepNLU v2.1.0 not available", "queries": [], "total": 0}
        return self.query_pipeline.synthesize(
            text, max_queries=max_queries,
            min_confidence=min_confidence,
            query_types=query_types,
        )

    def batch_synthesize_queries(self, texts: List[str], *,
                                 max_per_text: int = 15) -> Dict[str, Any]:
        """Batch query synthesis over multiple texts.  ★ NEW v5.0.0"""
        if not self._deep_nlu_available or self.query_pipeline is None:
            return {"error": "DeepNLU v2.1.0 not available", "queries": [], "total": 0}
        return self.query_pipeline.batch_synthesize(texts, max_per_text=max_per_text)

    # ─── Query Augmentation (DeepNLU v2.3.0) ─────────────────────────────

    def decompose_query(self, query: str, *, max_depth: int = 3) -> Dict[str, Any]:
        """Decompose a multi-hop query into atomic sub-queries.  ★ NEW v6.0.0"""
        if not self._deep_nlu_available or self.decomposer is None:
            return {"error": "DeepNLU v2.3.0 not available", "sub_queries": [], "count": 0}
        return self.decomposer.decompose(query, max_depth=max_depth)

    def expand_query(self, query: str, *, max_expansions: int = 5,
                     strategies: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Expand a query with synonyms, hypernyms, variants.  ★ NEW v6.0.0"""
        if not self._deep_nlu_available or self.expander is None:
            return {"error": "DeepNLU v2.3.0 not available", "expansions": [], "count": 0}
        return self.expander.expand(query, max_expansions=max_expansions, strategies=strategies)

    def classify_query(self, query: str) -> Dict[str, Any]:
        """Classify a query by Bloom's taxonomy, domain, complexity, format.  ★ NEW v6.0.0"""
        if not self._deep_nlu_available or self.query_classifier is None:
            return {"error": "DeepNLU v2.3.0 not available"}
        return self.query_classifier.classify(query)

    # ─── DeepNLU v2.3.0 — Entailment / Figurative / Density ──────────────

    def check_entailment(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """Check textual entailment between premise and hypothesis.  ★ NEW v7.0.0

        NLI classification: ENTAILMENT / CONTRADICTION / NEUTRAL.
        """
        if not self._deep_nlu_available or self.entailment is None:
            return {"error": "DeepNLU v2.3.0 not available", "label": "neutral", "confidence": 0}
        return self.entailment.check(premise, hypothesis)

    def analyze_figurative(self, text: str) -> Dict[str, Any]:
        """Detect figurative language: idioms, similes, irony, hyperbole.  ★ NEW v7.0.0"""
        if not self._deep_nlu_available or self.figurative is None:
            return {"error": "DeepNLU v2.3.0 not available", "figures": [], "count": 0}
        return self.figurative.analyze(text)

    def analyze_density(self, text: str) -> Dict[str, Any]:
        """Analyze information density: surprisal, diversity, redundancy.  ★ NEW v7.0.0"""
        if not self._deep_nlu_available or self.density_analyzer is None:
            return {"error": "DeepNLU v2.3.0 not available", "overall_density": 0}
        return self.density_analyzer.analyze(text)

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
        subsystems = {
            "tokenizer": self.tokenizer.status(),
            "ngram_model": self.ngram_model.status(),
            "embeddings": self.embeddings.status(),
            "classifier": self.classifier.status(),
            "sentiment": self.sentiment.status(),
            "ner": self.ner.status(),
            "similarity": self.sim.status(),
            "keyword_extractor": self.keyword_extractor.status(),
            "summarizer": self.summarizer.status(),
        }
        # DeepNLU v2.3.0 subsystems
        if self._deep_nlu_available:
            subsystems["temporal_reasoner"] = {"active": True, "layer": "L10", "source": "DeepNLU v2.3.0"}
            subsystems["causal_reasoner"] = {"active": True, "layer": "L11", "source": "DeepNLU v2.3.0"}
            subsystems["contextual_disambiguator"] = {"active": True, "layer": "L12", "source": "DeepNLU v2.3.0"}
            subsystems["query_synthesis_pipeline"] = {
                "active": True, "layer": "L14", "source": "DeepNLU v2.3.0",
                "archetypes": 8,
                "queries_generated": self.query_pipeline._queries_generated if self.query_pipeline else 0,
            }
            subsystems["query_decomposer"] = {
                "active": True, "layer": "L15", "source": "DeepNLU v2.3.0",
                "decompositions": self.decomposer._decompositions if self.decomposer else 0,
            }
            subsystems["query_expander"] = {
                "active": True, "layer": "L16", "source": "DeepNLU v2.3.0",
                "expansions": self.expander._expansions if self.expander else 0,
                "synonym_clusters": len(self.expander.SYNONYM_CLUSTERS) if self.expander else 0,
            }
            subsystems["query_classifier"] = {
                "active": True, "layer": "L17", "source": "DeepNLU v2.3.0",
                "classifications": self.query_classifier._classifications if self.query_classifier else 0,
                "bloom_levels": len(BloomLevel) if _DEEP_NLU_AVAILABLE else 0,
                "domains": len(QueryDomain) if _DEEP_NLU_AVAILABLE else 0,
            }
            subsystems["textual_entailment"] = {
                "active": True, "layer": "L18", "source": "DeepNLU v2.3.0",
                "checks": self.entailment._entailment_checks if self.entailment else 0,
                "antonym_pairs": len(TextualEntailmentEngine.ANTONYM_PAIRS) if _DEEP_NLU_AVAILABLE else 0,
                "labels": len(EntailmentLabel) if _DEEP_NLU_AVAILABLE else 0,
            }
            subsystems["figurative_language"] = {
                "active": True, "layer": "L19", "source": "DeepNLU v2.3.0",
                "analyses": self.figurative._analyses if self.figurative else 0,
                "idioms_known": len(FigurativeLanguageProcessor.IDIOM_DB) if _DEEP_NLU_AVAILABLE else 0,
                "figurative_types": len(FigurativeType) if _DEEP_NLU_AVAILABLE else 0,
            }
            subsystems["information_density"] = {
                "active": True, "layer": "L20", "source": "DeepNLU v2.3.0",
                "analyses": self.density_analyzer._analyses if self.density_analyzer else 0,
            }
        return {
            "version": VERSION,
            "engine": "LanguageEngine",
            "god_code": GOD_CODE,
            "phi": PHI,
            "deep_nlu_available": self._deep_nlu_available,
            "subsystem_count": len(subsystems),
            "consciousness": {
                "level": state.get("consciousness_level", 0.5),
                "nirvanic_fuel": state.get("nirvanic_fuel", 0.0),
                "entropy": state.get("entropy", 0.5),
                "evo_stage": state.get("evo_stage", "DORMANT"),
            },
            "subsystems": subsystems,
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
    print(f"     L104 LANGUAGE MODEL ENGINE v{VERSION} — EVO_57 (DeepNLU v2.3.0)")
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
    n_subs = len(engine.status().get('subsystems', {}))
    print(f"          LANGUAGE ENGINE OPERATIONAL — {n_subs} subsystems active")
    print("=" * 70)


if __name__ == "__main__":
    main()
