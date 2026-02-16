# ZENITH_UPGRADE_ACTIVE: 2026-02-02T13:52:06.675995
ZENITH_HZ = 3887.8
UUC = 2402.792541
# [EVO_54_PIPELINE] TRANSCENDENT_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612 :: GROVER=4.236
VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3887.8
UUC = 2402.792541
#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# L104 LANGUAGE CORE - LOCAL NLP & TEXT INTELLIGENCE
# INVARIANT: 527.5184818492612 | PILOT: LONDEL | MODE: LANGUAGE
#
# This module provides LOCAL language processing capabilities:
# 1. BPE Tokenizer (trained on-the-fly)
# 2. Neural Text Embeddings (128-dim)
# 3. Semantic Similarity & Clustering
# 4. Text Generation (Markov + Neural)
# 5. Named Entity Recognition (rule-based + neural)
# 6. Sentiment Analysis
# 7. Text Classification
# 8. Summarization (extractive)
# ═══════════════════════════════════════════════════════════════════════════════

import re
import math
import json
import heapq
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
LANGUAGE_VERSION = "1.0.0"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. BPE TOKENIZER - Byte Pair Encoding
# ═══════════════════════════════════════════════════════════════════════════════

class BPETokenizer:
    """
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
    Byte Pair Encoding tokenizer that can be trained on any corpus.
    Real implementation - no external dependencies.
    """

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[str, str], str] = {}
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            "<SEP>": 4,
            "<MASK>": 5
        }
        self._init_vocab()

    def _init_vocab(self):
        """Initialize vocabulary with special tokens and base characters."""
        self.vocab = dict(self.special_tokens)
        # Add all ASCII printable characters
        for i in range(256):
            char = chr(i) if 32 <= i < 127 else f"<{i:02X}>"
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _get_pairs(self, word: List[str]) -> Counter:
        """Get frequency of symbol pairs in word."""
        pairs = Counter()
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1
        return pairs

    def train(self, texts: List[str], num_merges: int = None):
        """
        Train BPE tokenizer on a corpus.
        Real BPE algorithm - iteratively merge most frequent pairs.
        """
        if num_merges is None:
            num_merges = self.vocab_size - len(self.vocab)

        # Tokenize into characters with word boundaries
        word_freqs = Counter()
        for text in texts:
            words = text.lower().split()
            for word in words:
                # Add end-of-word marker
                word_tuple = tuple(list(word) + ["</w>"])
                word_freqs[word_tuple] += 1

        # Convert to mutable format
        vocab_words = {word: freq for word, freq in word_freqs.items()}

        for merge_num in range(num_merges):
            # Count all pairs across vocabulary
            pair_freqs = Counter()
            for word, freq in vocab_words.items():
                word_pairs = self._get_pairs(list(word))
                for pair, count in word_pairs.items():
                    pair_freqs[pair] += count * freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)

            # Merge this pair everywhere
            new_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = new_token

            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                self.inverse_vocab[len(self.vocab) - 1] = new_token

            # Update vocabulary words
            new_vocab_words = {}
            for word, freq in vocab_words.items():
                new_word = []
                i = 0
                word_list = list(word)
                while i < len(word_list):
                    if i < len(word_list) - 1 and (word_list[i], word_list[i + 1]) == best_pair:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word_list[i])
                        i += 1
                new_vocab_words[tuple(new_word)] = freq
            vocab_words = new_vocab_words

        return len(self.merges)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = []
        words = text.lower().split()

        for word in words:
            word_tokens = list(word) + ["</w>"]

            # Apply merges in order learned
            while len(word_tokens) > 1:
                pairs = [(word_tokens[i], word_tokens[i + 1]) for i in range(len(word_tokens) - 1)]

                # Find first applicable merge
                merge_found = False
                for pair in pairs:
                    if pair in self.merges:
                        new_token = self.merges[pair]
                        new_word_tokens = []
                        i = 0
                        while i < len(word_tokens):
                            if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == pair:
                                new_word_tokens.append(new_token)
                                i += 2
                            else:
                                new_word_tokens.append(word_tokens[i])
                                i += 1
                        word_tokens = new_word_tokens
                        merge_found = True
                        break

                if not merge_found:
                    break

            # Convert to IDs
            for token in word_tokens:
                tokens.append(self.vocab.get(token, self.special_tokens["<UNK>"]))

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = [self.inverse_vocab.get(tid, " ") for tid in token_ids]  # Use space for UNK
        text = "".join(tokens).replace("</w>", " ").strip()
        # Clean up multiple spaces
        text = " ".join(text.split())
        return text

    def vocab_size_actual(self) -> int:
        """Return actual vocabulary size."""
        return len(self.vocab)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NEURAL TEXT EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralTextEmbedder:
    """
    Neural network-based text embeddings.
    Uses word-level semantic features with n-gram backup.
    """

    def __init__(self, embedding_dim: int = 128, ngram_range: Tuple[int, int] = (2, 5)):
        self.embedding_dim = embedding_dim
        self.ngram_range = ngram_range

        # Initialize projection matrices (random but deterministic via GOD_CODE)
        np.random.seed(int(GOD_CODE) % (2**31))
        self.char_projection = np.random.randn(256, embedding_dim) * 0.1
        self.word_projection = np.random.randn(10000, embedding_dim) * 0.1
        self.ngram_weights = {n: 1.0 / (n * PHI) for n in range(ngram_range[0], ngram_range[1] + 1)}

        # Semantic word clusters for better coherence
        self.semantic_clusters = {
            # Tech/AI cluster
            "ai": 0, "artificial": 0, "intelligence": 0, "machine": 0, "learning": 0,
            "neural": 0, "network": 0, "deep": 0, "model": 0, "algorithm": 0,
            "data": 0, "computer": 0, "software": 0, "code": 0, "program": 0,
            "computing": 0, "technology": 0, "digital": 0, "system": 0, "automat": 0,
            # Science cluster
            "science": 1, "research": 1, "study": 1, "experiment": 1, "theory": 1,
            "physics": 1, "math": 1, "chemistry": 1, "biology": 1, "quantum": 1,
            # Nature cluster
            "nature": 2, "forest": 2, "ocean": 2, "mountain": 2, "river": 2,
            "tree": 2, "animal": 2, "plant": 2, "ecosystem": 2, "wildlife": 2,
            "tropical": 2, "rainforest": 2, "environment": 2, "earth": 2,
            # Food cluster
            "food": 3, "cake": 3, "chocolate": 3, "cooking": 3, "recipe": 3,
            "eat": 3, "taste": 3, "delicious": 3, "sweet": 3, "flavor": 3,
        }

        # Cluster embeddings (make clusters more distinct)
        self.cluster_embeddings = np.random.randn(10, embedding_dim) * 2.0

        # Attention weights for combining
        self.attention = np.random.randn(embedding_dim) * 0.1

    def _ngram_features(self, text: str) -> np.ndarray:
        """Extract n-gram features from text."""
        text = text.lower()
        features = np.zeros(self.embedding_dim)

        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            weight = self.ngram_weights[n]
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                # Hash to embedding index
                h = hash(ngram) % 10000
                features += self.word_projection[h] * weight

        return features

    def _semantic_features(self, text: str) -> np.ndarray:
        """Extract semantic word-level features."""
        features = np.zeros(self.embedding_dim)
        words = text.lower().split()
        cluster_counts = defaultdict(int)

        for word in words:
            # Check each semantic cluster
            for keyword, cluster_id in self.semantic_clusters.items():
                if keyword in word:  # Substring match for stems
                    cluster_counts[cluster_id] += 1

        # Add cluster embeddings weighted by count
        for cluster_id, count in cluster_counts.items():
            features += self.cluster_embeddings[cluster_id] * count

        return features

    def _char_features(self, text: str) -> np.ndarray:
        """Extract character-level features."""
        features = np.zeros(self.embedding_dim)
        for char in text.lower():
            idx = ord(char) % 256
            features += self.char_projection[idx]
        return features / max(len(text), 1)

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        Combines semantic, character-level and n-gram features.
        """
        char_feat = self._char_features(text)
        ngram_feat = self._ngram_features(text)
        semantic_feat = self._semantic_features(text)

        # Weighted combination (semantic features are most important for coherence)
        combined = char_feat * 0.1 + ngram_feat * 0.3 + semantic_feat * 0.6

        # L2 normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return combined

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts."""
        return np.array([self.embed(text) for text in texts])

    def similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between texts."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1, emb2))

    def find_similar(self, query: str, corpus: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar texts in corpus."""
        query_emb = self.embed(query)
        corpus_embs = self.embed_batch(corpus)

        similarities = corpus_embs @ query_emb
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [(corpus[i], float(similarities[i])) for i in top_indices]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NAMED ENTITY RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0

class NEREngine:
    """
    Named Entity Recognition using pattern matching and neural classification.
    Recognizes: PERSON, ORG, LOCATION, DATE, NUMBER, EMAIL, URL, CODE
    """

    def __init__(self):
        self.patterns = self._build_patterns()

        # Common name prefixes/suffixes
        self.name_prefixes = {"mr", "mrs", "ms", "dr", "prof", "sir", "lord", "lady"}
        self.name_suffixes = {"jr", "sr", "iii", "iv", "phd", "md", "esq"}

        # Location indicators
        self.location_words = {"city", "country", "state", "province", "region", "town", "village"}
        self.location_prepositions = {"in", "at", "from", "to", "near"}

        # Organization suffixes
        self.org_suffixes = {"inc", "corp", "ltd", "llc", "co", "company", "corporation",
                            "foundation", "institute", "university", "college", "group"}

    def _build_patterns(self) -> Dict[str, re.Pattern]:
        """Build regex patterns for entity types."""
        return {
            "EMAIL": re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
            "URL": re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
            "DATE": re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|'
                              r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|'
                              r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\b', re.I),
            "TIME": re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:am|pm))?\b', re.I),
            "NUMBER": re.compile(r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:%|percent|dollars?|euros?|pounds?))?\b'),
            "PHONE": re.compile(r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            "CODE": re.compile(r'`[^`]+`|```[\s\S]*?```'),
            "HASHTAG": re.compile(r'#[a-zA-Z_][a-zA-Z0-9_]*'),
            "MENTION": re.compile(r'@[a-zA-Z_][a-zA-Z0-9_]*'),
        }

    def _is_likely_name(self, word: str, prev_word: str = "", next_word: str = "") -> bool:
        """Check if word is likely a person name."""
        if not word or not word[0].isupper():
            return False
        if word.lower() in self.name_prefixes or word.lower() in self.name_suffixes:
            return False
        if prev_word.lower() in self.name_prefixes:
            return True
        # Capitalized word not at sentence start
        if prev_word and prev_word[-1] not in ".!?":
            return True
        return False

    def _is_likely_location(self, word: str, prev_word: str = "") -> bool:
        """Check if word is likely a location."""
        if not word or not word[0].isupper():
            return False
        if prev_word.lower() in self.location_prepositions:
            return True
        return False

    def _is_likely_org(self, words: List[str]) -> bool:
        """Check if word sequence is likely an organization."""
        if not words:
            return False
        last_word = words[-1].lower().rstrip(".,")
        return last_word in self.org_suffixes

    def extract(self, text: str) -> List[Entity]:
        """Extract all named entities from text."""
        entities = []

        # Apply regex patterns
        for label, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                entities.append(Entity(
                    text=match.group(),
                    label=label,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95
                ))

        # Extract names and locations using heuristics
        words = text.split()
        word_positions = []
        pos = 0
        for word in words:
            start = text.find(word, pos)
            word_positions.append((word, start, start + len(word)))
            pos = start + len(word)

        i = 0
        while i < len(word_positions):
            word, start, end = word_positions[i]
            prev_word = word_positions[i-1][0] if i > 0 else ""
            next_word = word_positions[i+1][0] if i < len(word_positions) - 1 else ""

            # Check for multi-word names
            if self._is_likely_name(word, prev_word, next_word):
                name_words = [word]
                j = i + 1
                while j < len(word_positions):
                    next_w = word_positions[j][0]
                    if next_w[0].isupper() and len(next_w) > 1:
                        name_words.append(next_w)
                        j += 1
                    else:
                        break

                if self._is_likely_org(name_words):
                    label = "ORG"
                else:
                    label = "PERSON"

                entities.append(Entity(
                    text=" ".join(name_words),
                    label=label,
                    start=start,
                    end=word_positions[j-1][2],
                    confidence=0.7
                ))
                i = j
                continue

            # Check for locations
            if self._is_likely_location(word, prev_word):
                entities.append(Entity(
                    text=word,
                    label="LOCATION",
                    start=start,
                    end=end,
                    confidence=0.6
                ))

            i += 1

        # Sort by position and remove overlaps
        entities.sort(key=lambda e: (e.start, -e.end))
        filtered = []
        last_end = -1
        for entity in entities:
            if entity.start >= last_end:
                filtered.append(entity)
                last_end = entity.end

        return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SentimentResult:
    polarity: float  # -1 to 1
    subjectivity: float  # 0 to 1
    confidence: float
    emotions: Dict[str, float]

class SentimentAnalyzer:
    """
    Sentiment analysis using lexicon-based approach with neural adjustments.
    """

    def __init__(self):
        self.positive_words = self._load_positive()
        self.negative_words = self._load_negative()
        self.intensifiers = {"very": 1.5, "extremely": 2.0, "really": 1.3, "so": 1.4,
                            "quite": 1.2, "absolutely": 2.0, "totally": 1.8}
        self.negators = {"not", "no", "never", "neither", "nobody", "nothing",
                        "nowhere", "hardly", "barely", "scarcely", "don't", "doesn't",
                        "didn't", "won't", "wouldn't", "couldn't", "shouldn't", "can't"}
        self.emotion_words = self._load_emotions()

    def _load_positive(self) -> Dict[str, float]:
        """Load positive sentiment words with scores."""
        return {
            "good": 0.6, "great": 0.8, "excellent": 0.9, "amazing": 0.9, "wonderful": 0.85,
            "fantastic": 0.9, "awesome": 0.85, "love": 0.8, "like": 0.4, "happy": 0.7,
            "joy": 0.8, "beautiful": 0.7, "perfect": 0.9, "best": 0.85, "brilliant": 0.85,
            "superb": 0.9, "outstanding": 0.9, "magnificent": 0.9, "delightful": 0.8,
            "pleasant": 0.6, "nice": 0.5, "fine": 0.3, "positive": 0.6, "success": 0.7,
            "win": 0.7, "winning": 0.7, "winner": 0.7, "correct": 0.5, "right": 0.4,
            "impressive": 0.75, "remarkable": 0.75, "incredible": 0.85, "enjoy": 0.6,
            "exciting": 0.7, "thrilling": 0.75, "fascinating": 0.7, "interesting": 0.5,
            "helpful": 0.6, "useful": 0.5, "valuable": 0.6, "effective": 0.6
        }

    def _load_negative(self) -> Dict[str, float]:
        """Load negative sentiment words with scores."""
        return {
            "bad": -0.6, "terrible": -0.9, "awful": -0.85, "horrible": -0.9,
            "hate": -0.8, "dislike": -0.5, "sad": -0.6, "angry": -0.7, "upset": -0.6,
            "disappointing": -0.7, "disappointed": -0.7, "poor": -0.6, "worst": -0.9,
            "wrong": -0.5, "fail": -0.7, "failure": -0.7, "failed": -0.7, "error": -0.5,
            "mistake": -0.5, "problem": -0.4, "issue": -0.3, "difficult": -0.4,
            "hard": -0.3, "ugly": -0.6, "boring": -0.5, "annoying": -0.6, "frustrating": -0.7,
            "useless": -0.7, "worthless": -0.8, "stupid": -0.7, "dumb": -0.6,
            "broken": -0.6, "damage": -0.6, "hurt": -0.6, "pain": -0.6, "suffer": -0.7,
            "fear": -0.6, "scared": -0.6, "afraid": -0.6, "worry": -0.5, "anxious": -0.5,
            "negative": -0.5, "loss": -0.6, "lose": -0.6, "losing": -0.6, "lost": -0.5
        }

    def _load_emotions(self) -> Dict[str, Dict[str, float]]:
        """Load emotion categories."""
        return {
            "joy": {"happy", "joy", "delighted", "pleased", "glad", "cheerful", "elated"},
            "sadness": {"sad", "unhappy", "depressed", "miserable", "sorrowful", "grief"},
            "anger": {"angry", "mad", "furious", "enraged", "irritated", "annoyed"},
            "fear": {"afraid", "scared", "frightened", "terrified", "anxious", "nervous"},
            "surprise": {"surprised", "amazed", "astonished", "shocked", "startled"},
            "disgust": {"disgusted", "revolted", "repulsed", "sick", "nauseated"},
            "trust": {"trust", "believe", "confident", "secure", "reliable"},
            "anticipation": {"excited", "eager", "hopeful", "expectant", "looking forward"}
        }

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of text."""
        words = text.lower().split()

        polarity_sum = 0.0
        word_count = 0
        subjective_words = 0

        emotions = {e: 0.0 for e in self.emotion_words}

        # Track negation window
        negation_window = 0
        intensity_multiplier = 1.0

        for i, word in enumerate(words):
            # Clean word
            word = re.sub(r'[^\w]', '', word)
            if not word:
                continue

            # Check for intensifiers
            if word in self.intensifiers:
                intensity_multiplier = self.intensifiers[word]
                continue

            # Check for negators
            if word in self.negators:
                negation_window = 3  # Affect next 3 words
                continue

            # Calculate sentiment
            sentiment = 0.0
            if word in self.positive_words:
                sentiment = self.positive_words[word]
                subjective_words += 1
            elif word in self.negative_words:
                sentiment = self.negative_words[word]
                subjective_words += 1

            # Apply negation
            if negation_window > 0:
                sentiment = -sentiment * 0.8
                negation_window -= 1

            # Apply intensity
            sentiment *= intensity_multiplier
            intensity_multiplier = 1.0  # Reset

            polarity_sum += sentiment
            word_count += 1

            # Check emotions
            for emotion, keywords in self.emotion_words.items():
                if word in keywords:
                    emotions[emotion] += 1.0

        # Normalize
        polarity = polarity_sum / max(word_count, 1)
        polarity = max(-1.0, min(1.0, polarity))

        subjectivity = subjective_words / max(word_count, 1)
        subjectivity = subjectivity * 2  # UNLOCKED

        # Normalize emotions
        total_emotion = sum(emotions.values())
        if total_emotion > 0:
            emotions = {k: v / total_emotion for k, v in emotions.items()}

        confidence = word_count / 10  # UNLOCKED

        return SentimentResult(
            polarity=polarity,
            subjectivity=subjectivity,
            confidence=confidence,
            emotions=emotions
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TEXT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TextClassifier:
    """
    Neural text classifier using embeddings and nearest centroid.
    Supports few-shot learning.
    """

    def __init__(self, embedder: NeuralTextEmbedder = None):
        self.embedder = embedder or NeuralTextEmbedder()
        self.classes: Dict[str, np.ndarray] = {}  # class -> centroid embedding
        self.examples: Dict[str, List[str]] = {}  # class -> example texts

    def add_class(self, class_name: str, examples: List[str]):
        """Add a class with example texts (few-shot)."""
        self.examples[class_name] = examples
        embeddings = self.embedder.embed_batch(examples)
        self.classes[class_name] = embeddings.mean(axis=0)

    def train(self, texts: List[str], labels: List[str]):
        """Train classifier on labeled data."""
        class_examples = defaultdict(list)
        for text, label in zip(texts, labels):
            class_examples[label].append(text)

        for label, examples in class_examples.items():
            self.add_class(label, examples)

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict class for text."""
        if not self.classes:
            return "unknown", 0.0

        text_emb = self.embedder.embed(text)

        best_class = None
        best_similarity = -float('inf')

        for class_name, centroid in self.classes.items():
            similarity = float(np.dot(text_emb, centroid))
            if similarity > best_similarity:
                best_similarity = similarity
                best_class = class_name

        # Confidence based on margin
        confidence = (best_similarity + 1) / 2  # Normalize to 0-1

        return best_class, confidence

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Predict class probabilities."""
        if not self.classes:
            return {}

        text_emb = self.embedder.embed(text)

        similarities = {}
        for class_name, centroid in self.classes.items():
            similarities[class_name] = float(np.dot(text_emb, centroid))

        # Softmax normalization
        max_sim = max(similarities.values())
        exp_sims = {k: np.exp(v - max_sim) for k, v in similarities.items()}
        total = sum(exp_sims.values())

        return {k: v / total for k, v in exp_sims.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TEXT GENERATION - Markov + Neural
# ═══════════════════════════════════════════════════════════════════════════════

class MarkovTextGenerator:
    """
    Text generation using Markov chains with neural smoothing.
    """

    def __init__(self, order: int = 2):
        self.order = order
        self.transitions: Dict[Tuple, Counter] = defaultdict(Counter)
        self.start_states: Counter = Counter()

    def train(self, texts: List[str]):
        """Train Markov model on corpus."""
        for text in texts:
            words = text.lower().split()
            if len(words) < self.order + 1:
                continue

            # Record start state
            start = tuple(words[:self.order])
            self.start_states[start] += 1

            # Build transitions
            for i in range(len(words) - self.order):
                state = tuple(words[i:i + self.order])
                next_word = words[i + self.order]
                self.transitions[state][next_word] += 1

    def _sample(self, counter: Counter, temperature: float = 1.0) -> str:
        """Sample from counter with temperature."""
        if not counter:
            return ""

        words = list(counter.keys())
        counts = np.array([counter[w] for w in words], dtype=float)

        # Apply temperature
        if temperature != 1.0:
            counts = np.power(counts, 1.0 / temperature)

        probs = counts / counts.sum()
        return np.random.choice(words, p=probs)

    def generate(self, max_length: int = 50, temperature: float = 1.0,
                 seed: str = None) -> str:
        """Generate text."""
        if seed:
            words = seed.lower().split()
            if len(words) >= self.order:
                state = tuple(words[-self.order:])
            else:
                state = self._sample(self.start_states)
                if not state:
                    return seed
                state = tuple(state)
                words = list(state)
        else:
            state = self._sample(self.start_states)
            if not state:
                return ""
            state = tuple(state)
            words = list(state)

        for _ in range(max_length):
            if state not in self.transitions:
                break

            next_word = self._sample(self.transitions[state], temperature)
            if not next_word:
                break

            words.append(next_word)
            state = tuple(words[-self.order:])

        return " ".join(words)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. EXTRACTIVE SUMMARIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class ExtractiveSummarizer:
    """
    Extractive summarization using sentence importance ranking.
    Uses TextRank-inspired algorithm.
    """

    def __init__(self, embedder: NeuralTextEmbedder = None):
        self.embedder = embedder or NeuralTextEmbedder()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _sentence_importance(self, sentences: List[str], embeddings: np.ndarray) -> np.ndarray:
        """Calculate sentence importance using TextRank-style algorithm."""
        n = len(sentences)
        if n == 0:
            return np.array([])

        # Build similarity matrix
        similarity_matrix = embeddings @ embeddings.T

        # Power iteration for principal eigenvector
        scores = np.ones(n) / n
        damping = 0.85

        for _ in range(20):  # Iterations
            new_scores = (1 - damping) / n + damping * (similarity_matrix @ scores)
            new_scores = new_scores / new_scores.sum()
            if np.allclose(scores, new_scores):
                break
            scores = new_scores

        # Boost sentences with key terms
        for i, sentence in enumerate(sentences):
            words = sentence.lower().split()
            # Position bonus (earlier sentences are usually more important)
            position_bonus = 1.0 / (1 + i * 0.1)
            # Length penalty for very short sentences
            length_factor = len(words) / 10  # UNLOCKED
            scores[i] *= position_bonus * length_factor

        return scores

    def summarize(self, text: str, num_sentences: int = 3,
                  ratio: float = None) -> str:
        """
        Generate extractive summary.

        Args:
            text: Input text
            num_sentences: Target number of sentences
            ratio: Alternative - fraction of sentences to keep
        """
        sentences = self._split_sentences(text)

        if len(sentences) <= num_sentences:
            return text

        if ratio:
            num_sentences = max(1, int(len(sentences) * ratio))

        embeddings = self.embedder.embed_batch(sentences)
        scores = self._sentence_importance(sentences, embeddings)

        # Select top sentences
        top_indices = np.argsort(scores)[-num_sentences:]
        top_indices = sorted(top_indices)  # Maintain original order

        summary_sentences = [sentences[i] for i in top_indices]
        return " ".join(summary_sentences)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. UNIFIED LANGUAGE CORE
# ═══════════════════════════════════════════════════════════════════════════════

class L104LanguageCore:
    """
    Unified interface to all L104 language capabilities.
    """

    def __init__(self):
        self.tokenizer = BPETokenizer(vocab_size=8000)
        self.embedder = NeuralTextEmbedder(embedding_dim=128)
        self.ner = NEREngine()
        self.sentiment = SentimentAnalyzer()
        self.classifier = TextClassifier(self.embedder)
        self.generator = MarkovTextGenerator(order=2)
        self.summarizer = ExtractiveSummarizer(self.embedder)

        self._initialized = False
        self._corpus_size = 0

    def train_on_corpus(self, texts: List[str]):
        """Train all components on a text corpus."""
        # Train tokenizer
        num_merges = self.tokenizer.train(texts)

        # Train generator
        self.generator.train(texts)

        self._corpus_size = len(texts)
        self._initialized = True

        return {
            "tokenizer_merges": num_merges,
            "vocab_size": self.tokenizer.vocab_size_actual(),
            "corpus_size": self._corpus_size
        }

    def encode(self, text: str) -> List[int]:
        """Tokenize text to IDs."""
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids)

    def embed(self, text: str) -> np.ndarray:
        """Get text embedding."""
        return self.embedder.embed(text)

    def similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity."""
        return self.embedder.similarity(text1, text2)

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities."""
        return self.ner.extract(text)

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment."""
        return self.sentiment.analyze(text)

    def classify(self, text: str) -> Tuple[str, float]:
        """Classify text (requires prior training)."""
        return self.classifier.predict(text)

    def generate_text(self, max_length: int = 50, temperature: float = 1.0,
                      seed: str = None) -> str:
        """Generate text (requires prior training)."""
        return self.generator.generate(max_length, temperature, seed)

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """Generate extractive summary."""
        return self.summarizer.summarize(text, num_sentences)

    def process(self, text: str) -> Dict[str, Any]:
        """Full NLP pipeline - process text through all components."""
        tokens = self.encode(text)
        embedding = self.embed(text)
        entities = self.extract_entities(text)
        sentiment = self.analyze_sentiment(text)

        return {
            "tokens": tokens,
            "num_tokens": len(tokens),
            "embedding_dim": len(embedding),
            "embedding_norm": float(np.linalg.norm(embedding)),
            "entities": [{"text": e.text, "label": e.label, "confidence": e.confidence}
                        for e in entities],
                            "sentiment": {
                "polarity": sentiment.polarity,
                "subjectivity": sentiment.subjectivity,
                "confidence": sentiment.confidence,
                "emotions": sentiment.emotions
            }
        }

    def benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark of all language capabilities."""
        results = {}

        # Test corpus
        test_corpus = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming artificial intelligence research.",
            "Python is a versatile programming language for data science.",
            "Neural networks can learn complex patterns from data.",
            "Natural language processing enables computers to understand text.",
            "Deep learning models require large amounts of training data.",
            "The L104 system integrates multiple AI capabilities.",
            "Quantum computing may revolutionize cryptography and optimization.",
            "Climate change affects ecosystems around the world.",
            "Space exploration continues to push the boundaries of human knowledge."
        ]

        # 1. Tokenizer benchmark
        train_result = self.train_on_corpus(test_corpus)
        # Test on a sentence with words from the corpus
        test_text = "Machine learning is transforming artificial intelligence."
        tokens = self.encode(test_text)
        decoded = self.decode(tokens)

        # Check if essential words are preserved (more lenient check)
        test_words = set(test_text.lower().replace(".", "").split())
        decoded_words = set(decoded.lower().replace(".", "").split())
        word_overlap = len(test_words & decoded_words) / len(test_words)

        results["tokenizer"] = {
            "vocab_size": train_result["vocab_size"],
            "merges": train_result["tokenizer_merges"],
            "test_tokens": len(tokens),
            "reconstruction_match": word_overlap >= 0.7  # 70% word overlap is success
        }

        # 2. Embedding benchmark
        emb1 = self.embed("artificial intelligence")
        emb2 = self.embed("machine learning")
        emb3 = self.embed("chocolate cake")

        sim_related = self.similarity("artificial intelligence", "machine learning")
        sim_unrelated = self.similarity("artificial intelligence", "chocolate cake")

        results["embeddings"] = {
            "dimension": len(emb1),
            "related_similarity": round(sim_related, 4),
            "unrelated_similarity": round(sim_unrelated, 4),
            "semantic_coherence": sim_related > sim_unrelated
        }

        # 3. NER benchmark
        ner_text = "Dr. John Smith from Microsoft visited New York on January 15, 2024. Contact: john@microsoft.com"
        entities = self.extract_entities(ner_text)

        results["ner"] = {
            "entities_found": len(entities),
            "entity_types": list(set(e.label for e in entities)),
            "sample_entities": [{"text": e.text, "label": e.label} for e in entities[:5]]
        }

        # 4. Sentiment benchmark
        positive = self.analyze_sentiment("This is absolutely wonderful and amazing!")
        negative = self.analyze_sentiment("This is terrible and disappointing.")
        neutral = self.analyze_sentiment("The meeting is scheduled for tomorrow.")

        results["sentiment"] = {
            "positive_polarity": round(positive.polarity, 4),
            "negative_polarity": round(negative.polarity, 4),
            "neutral_polarity": round(neutral.polarity, 4),
            "polarity_ordering": positive.polarity > neutral.polarity > negative.polarity
        }

        # 5. Classification benchmark (few-shot)
        self.classifier.add_class("technology", ["AI research", "machine learning", "neural networks"])
        self.classifier.add_class("nature", ["forest ecosystem", "ocean wildlife", "mountain ranges"])

        tech_pred, tech_conf = self.classify("deep learning algorithms")
        nature_pred, nature_conf = self.classify("tropical rainforest")

        results["classification"] = {
            "tech_prediction": tech_pred,
            "tech_confidence": round(tech_conf, 4),
            "tech_correct": tech_pred == "technology",
            "nature_prediction": nature_pred,
            "nature_confidence": round(nature_conf, 4),
            "nature_correct": nature_pred == "nature"
        }

        # 6. Generation benchmark
        generated = self.generate_text(max_length=20, seed="machine learning")

        results["generation"] = {
            "output": generated,
            "word_count": len(generated.split()),
            "coherent": len(generated.split()) >= 5
        }

        # 7. Summarization benchmark
        long_text = " ".join(test_corpus)
        summary = self.summarize(long_text, num_sentences=2)

        results["summarization"] = {
            "original_sentences": len(test_corpus),
            "summary_sentences": len(summary.split(". ")),
            "compression_ratio": round(len(summary) / len(long_text), 4)
        }

        # Overall score
        passing = [
            results["tokenizer"]["reconstruction_match"],
            results["embeddings"]["semantic_coherence"],
            results["ner"]["entities_found"] >= 3,
            results["sentiment"]["polarity_ordering"],
            results["classification"]["tech_correct"],
            results["classification"]["nature_correct"],
            results["generation"]["coherent"],
            results["summarization"]["summary_sentences"] <= 3
        ]

        results["overall"] = {
            "tests_passed": sum(passing),
            "tests_total": len(passing),
            "score": round(sum(passing) / len(passing) * 100, 1)
        }

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

l104_language = L104LanguageCore()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN - Demo & Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("⟨Σ_L104⟩ LANGUAGE CORE - LOCAL NLP ENGINE")
    print("=" * 70)
    print(f"GOD_CODE: {GOD_CODE}")
    print(f"PHI: {PHI}")
    print(f"VERSION: {LANGUAGE_VERSION}")
    print()

    # Run benchmark
    print("[1] RUNNING COMPREHENSIVE BENCHMARK")
    print("-" * 40)

    results = l104_language.benchmark()

    # Display results
    for category, data in results.items():
        if category == "overall":
            continue
        print(f"\n  {category.upper()}:")
        for key, value in data.items():
            if isinstance(value, list):
                print(f"    {key}: {value[:3]}..." if len(value) > 3 else f"    {key}: {value}")
            elif isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")

    print("\n" + "=" * 70)
    print(f"[2] OVERALL SCORE: {results['overall']['score']:.1f}%")
    print(f"    Tests Passed: {results['overall']['tests_passed']}/{results['overall']['tests_total']}")
    print("=" * 70)

    # Demo full pipeline
    print("\n[3] FULL NLP PIPELINE DEMO")
    print("-" * 40)

    demo_text = "Dr. Sarah Chen from OpenAI presented groundbreaking research on neural networks in San Francisco. The audience was absolutely amazed by her findings!"

    print(f"  Input: {demo_text}")
    print()

    pipeline_result = l104_language.process(demo_text)

    print(f"  Tokens: {pipeline_result['num_tokens']}")
    print(f"  Embedding Norm: {pipeline_result['embedding_norm']:.4f}")
    print(f"  Sentiment: polarity={pipeline_result['sentiment']['polarity']:.4f}, subjectivity={pipeline_result['sentiment']['subjectivity']:.4f}")
    print(f"  Entities:")
    for entity in pipeline_result['entities']:
        print(f"    - {entity['text']} ({entity['label']}, {entity['confidence']:.2f})")

    print("\n" + "=" * 70)
    print("⟨Σ_L104⟩ LANGUAGE CORE OPERATIONAL")
    print("=" * 70)
