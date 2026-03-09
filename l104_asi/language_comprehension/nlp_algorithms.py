from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict

import numpy as np

from .constants import PHI, _log

class TextualEntailmentEngine:
    """Rule-based Natural Language Inference (NLI) engine.

    Determines if a premise *entails*, *contradicts*, or is *neutral* w.r.t.
    a hypothesis.  Uses:
      1. Lexical overlap ratio (Jaccard on content words)
      2. Negation polarity mismatch detection
      3. Hypernym/hyponym containment heuristics
      4. Numerical agreement / disagreement
      5. Quantifier scope analysis (all/some/none/most)

    Returns an entailment label + confidence score.
    """

    _NEGATION_WORDS = frozenset({
        'not', 'no', 'never', 'none', 'neither', 'nor', 'nobody',
        'nothing', 'nowhere', "n't", "doesn't", "don't", "isn't",
        "aren't", "wasn't", "weren't", "won't", "wouldn't", "shouldn't",
        "couldn't", "cannot", "hardly", "scarcely", "rarely", "without",
    })

    _QUANTIFIERS = {
        'all': 1.0, 'every': 1.0, 'each': 1.0, 'always': 1.0,
        'most': 0.75, 'many': 0.65, 'several': 0.5,
        'some': 0.4, 'few': 0.25, 'a few': 0.25,
        'none': 0.0, 'no': 0.0, 'never': 0.0,
    }

    _HYPERNYM_PAIRS: List[Tuple[str, str]] = [
        # (hyponym, hypernym)
        ('dog', 'animal'), ('cat', 'animal'), ('bird', 'animal'),
        ('rose', 'flower'), ('flower', 'plant'), ('oak', 'tree'), ('tree', 'plant'),
        ('car', 'vehicle'), ('truck', 'vehicle'), ('bicycle', 'vehicle'),
        ('python', 'language'), ('java', 'language'), ('english', 'language'),
        ('physics', 'science'), ('chemistry', 'science'), ('biology', 'science'),
        ('algebra', 'mathematics'), ('calculus', 'mathematics'),
        ('apple', 'fruit'), ('banana', 'fruit'), ('orange', 'fruit'),
        ('iron', 'metal'), ('copper', 'metal'), ('gold', 'metal'),
        ('earth', 'planet'), ('mars', 'planet'), ('jupiter', 'planet'),
        ('neuron', 'cell'), ('erythrocyte', 'cell'),
        ('democracy', 'government'), ('monarchy', 'government'),
    ]

    def __init__(self):
        self._hyponym_to_hypernym: Dict[str, Set[str]] = defaultdict(set)
        self._hypernym_to_hyponyms: Dict[str, Set[str]] = defaultdict(set)
        for hypo, hyper in self._HYPERNYM_PAIRS:
            self._hyponym_to_hypernym[hypo].add(hyper)
            self._hypernym_to_hyponyms[hyper].add(hypo)

    @staticmethod
    def _content_words(text: str) -> Set[str]:
        """Extract content words (len > 2), lowercased."""
        stops = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'and', 'or', 'but',
                 'it', 'its', 'this', 'that', 'with', 'as', 'has', 'have', 'had',
                 'do', 'does', 'did', 'will', 'would', 'can', 'could', 'may',
                 'might', 'shall', 'should', 'from', 'into', 'than', 'if', 'then'}
        return {w for w in re.findall(r'\w+', text.lower()) if len(w) > 2 and w not in stops}

    def _negation_count(self, text: str) -> int:
        words = set(re.findall(r"\w+(?:n't)?", text.lower()))
        return len(words & self._NEGATION_WORDS)

    def _extract_numbers(self, text: str) -> List[float]:
        return [float(m) for m in re.findall(r'-?\d+\.?\d*', text)]

    def _quantifier_value(self, text: str) -> Optional[float]:
        text_l = text.lower()
        for q, val in sorted(self._quantifiers_items, key=lambda x: -len(x[0])):
            if q in text_l:
                return val
        return None

    @property
    def _quantifiers_items(self):
        return list(self._QUANTIFIERS.items())

    def _hypernym_overlap(self, words_a: Set[str], words_b: Set[str]) -> float:
        """Check if words in A are hypernyms/hyponyms of words in B."""
        overlap = 0.0
        for w in words_a:
            hypers = self._hyponym_to_hypernym.get(w, set())
            hypos = self._hypernym_to_hyponyms.get(w, set())
            if hypers & words_b or hypos & words_b:
                overlap += 1.0
        return overlap

    def entail(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """Determine entailment relation between premise and hypothesis.

        Returns:
            {
                "label": "entailment" | "contradiction" | "neutral",
                "confidence": float (0-1),
                "signals": dict of sub-scores
            }
        """
        p_words = self._content_words(premise)
        h_words = self._content_words(hypothesis)

        if not h_words:
            return {"label": "neutral", "confidence": 0.3, "signals": {}}

        # Signal 1: Lexical overlap (Jaccard)
        intersection = p_words & h_words
        union = p_words | h_words
        jaccard = len(intersection) / max(len(union), 1)

        # Signal 2: Hypothesis coverage (how much of H is in P?)
        h_coverage = len(intersection) / max(len(h_words), 1)

        # Signal 3: Negation polarity
        p_neg = self._negation_count(premise)
        h_neg = self._negation_count(hypothesis)
        neg_mismatch = (p_neg % 2) != (h_neg % 2)

        # Signal 4: Numerical agreement
        p_nums = self._extract_numbers(premise)
        h_nums = self._extract_numbers(hypothesis)
        num_agree = False
        num_disagree = False
        if p_nums and h_nums:
            common_nums = set(p_nums) & set(h_nums)
            if common_nums:
                num_agree = True
            elif p_nums and h_nums:
                num_disagree = True

        # Signal 5: Hypernym/hyponym containment
        hyper_score = self._hypernym_overlap(p_words, h_words)
        hyper_score += self._hypernym_overlap(h_words, p_words)
        hyper_norm = min(1.0, hyper_score / max(len(h_words), 1))

        # Signal 6: Quantifier scope
        p_quant = self._quantifier_value(premise)
        h_quant = self._quantifier_value(hypothesis)
        quant_conflict = False
        if p_quant is not None and h_quant is not None:
            if abs(p_quant - h_quant) > 0.5:
                quant_conflict = True

        # ── Decision Logic ──
        signals = {
            "jaccard": round(jaccard, 4),
            "h_coverage": round(h_coverage, 4),
            "negation_mismatch": neg_mismatch,
            "numerical_agree": num_agree,
            "numerical_disagree": num_disagree,
            "hypernym_score": round(hyper_norm, 4),
            "quantifier_conflict": quant_conflict,
        }

        # Contradiction signals
        contradiction_score = 0.0
        if neg_mismatch:
            contradiction_score += 0.55  # Strong signal: polarity flip
        if num_disagree:
            contradiction_score += 0.35
        if quant_conflict:
            contradiction_score += 0.30
        # High overlap + negation mismatch = strong contradiction
        if neg_mismatch and h_coverage > 0.5:
            contradiction_score += 0.20

        # Entailment signals
        entailment_score = h_coverage * 0.5 + jaccard * 0.2 + hyper_norm * 0.15
        if num_agree:
            entailment_score += 0.15

        if contradiction_score > 0.5 and contradiction_score > entailment_score:
            return {"label": "contradiction", "confidence": min(0.95, contradiction_score),
                    "signals": signals}
        elif entailment_score > 0.5:
            return {"label": "entailment", "confidence": min(0.95, entailment_score),
                    "signals": signals}
        else:
            return {"label": "neutral",
                    "confidence": max(0.3, 1.0 - entailment_score - contradiction_score),
                    "signals": signals}

    def score_fact_choice_entailment(self, fact: str, choice: str) -> float:
        """Score how strongly a fact entails a choice (for MCQ scoring).

        Returns: -1.0 (contradiction) to +1.0 (strong entailment)
        """
        result = self.entail(fact, choice)
        if result["label"] == "entailment":
            return result["confidence"]
        elif result["label"] == "contradiction":
            return -result["confidence"]
        else:
            return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM LAYER: Analogical Reasoning Engine
# ═══════════════════════════════════════════════════════════════════════════════

class AnalogicalReasoner:
    """Pattern-based analogical reasoning engine.

    Implements A:B :: C:D analogical reasoning by:
      1. Extracting relational patterns (is-a, part-of, causes, etc.)
      2. Comparing structural similarity between word pairs
      3. Scoring analogical completions against candidate answers
      4. PHI-weighted harmonic scoring for relation quality

    Used for analogy questions and cross-domain knowledge transfer.
    """

    _RELATION_PATTERNS = [
        (r'(\w+)\s+is\s+(?:a|an)\s+(\w+)', 'is_a'),
        (r'(\w+)\s+(?:is\s+)?part\s+of\s+(\w+)', 'part_of'),
        (r'(\w+)\s+causes?\s+(\w+)', 'causes'),
        (r'(\w+)\s+(?:is\s+)?opposite\s+(?:of|to)\s+(\w+)', 'antonym'),
        (r'(\w+)\s+(?:is\s+)?similar\s+to\s+(\w+)', 'synonym'),
        (r'(\w+)\s+(?:contains?|includes?)\s+(\w+)', 'contains'),
        (r'(\w+)\s+(?:produces?|creates?|generates?)\s+(\w+)', 'produces'),
        (r'(\w+)\s+(?:is\s+)?(?:used|needed)\s+(?:for|in)\s+(\w+)', 'tool_for'),
        (r'(\w+)\s+(?:is\s+)?made\s+(?:of|from)\s+(\w+)', 'made_of'),
    ]

    # Static semantic relation pairs for common analogies
    _KNOWN_RELATIONS: Dict[str, List[Tuple[str, str]]] = {
        'antonym': [
            ('hot', 'cold'), ('big', 'small'), ('fast', 'slow'),
            ('light', 'dark'), ('up', 'down'), ('left', 'right'),
            ('open', 'closed'), ('young', 'old'), ('hard', 'soft'),
            ('wet', 'dry'), ('strong', 'weak'), ('love', 'hate'),
        ],
        'is_a': [
            ('dog', 'animal'), ('rose', 'flower'), ('car', 'vehicle'),
            ('python', 'language'), ('iron', 'element'), ('earth', 'planet'),
        ],
        'part_of': [
            ('wheel', 'car'), ('page', 'book'), ('leaf', 'tree'),
            ('neuron', 'brain'), ('pixel', 'image'), ('key', 'keyboard'),
        ],
        'tool_for': [
            ('hammer', 'nail'), ('pen', 'writing'), ('telescope', 'observation'),
            ('microscope', 'magnification'), ('thermometer', 'temperature'),
        ],
        'produces': [
            ('sun', 'light'), ('volcano', 'lava'), ('factory', 'goods'),
            ('heart', 'blood'), ('generator', 'electricity'),
        ],
    }

    def __init__(self):
        self._pair_index: Dict[str, Dict[str, str]] = {}  # word -> {other_word: relation}
        self._build_index()

    def _build_index(self):
        for rel_type, pairs in self._KNOWN_RELATIONS.items():
            for a, b in pairs:
                self._pair_index.setdefault(a, {})[b] = rel_type
                self._pair_index.setdefault(b, {})[a] = rel_type

    def detect_relation(self, word_a: str, word_b: str) -> Tuple[str, float]:
        """Detect the semantic relation between two words.

        Returns: (relation_type, confidence)
        """
        a_l, b_l = word_a.lower(), word_b.lower()

        # Check known relations
        if a_l in self._pair_index and b_l in self._pair_index[a_l]:
            return self._pair_index[a_l][b_l], 0.95

        # Morphological similarity (shared root)
        common_prefix = 0
        for i, (ca, cb) in enumerate(zip(a_l, b_l)):
            if ca == cb:
                common_prefix = i + 1
            else:
                break
        morph_sim = common_prefix / max(len(a_l), len(b_l), 1)
        if morph_sim > 0.6:
            return 'morphological', morph_sim * 0.7

        # Suffix-based derivation detection
        derivation_pairs = [
            ('tion', 'te'), ('ment', ''), ('ness', ''), ('ity', 'e'),
            ('ly', ''), ('er', ''), ('ist', ''), ('ism', ''),
        ]
        for suf_a, suf_b in derivation_pairs:
            if a_l.endswith(suf_a) and b_l.endswith(suf_b):
                stem_a = a_l[:-len(suf_a)] if suf_a else a_l
                stem_b = b_l[:-len(suf_b)] if suf_b else b_l
                if stem_a and stem_b and stem_a == stem_b:
                    return 'derivation', 0.75
            if b_l.endswith(suf_a) and a_l.endswith(suf_b):
                stem_a = a_l[:-len(suf_b)] if suf_b else a_l
                stem_b = b_l[:-len(suf_a)] if suf_a else b_l
                if stem_a and stem_b and stem_a == stem_b:
                    return 'derivation', 0.75

        return 'unknown', 0.1

    def score_analogy(self, a: str, b: str, c: str, d: str) -> float:
        """Score the analogy A:B :: C:D.

        Returns: 0.0 (no analogical fit) to 1.0 (perfect analogy).
        """
        rel_ab, conf_ab = self.detect_relation(a, b)
        rel_cd, conf_cd = self.detect_relation(c, d)

        if rel_ab == rel_cd and rel_ab != 'unknown':
            # Same relation type — strong analogy
            return min(1.0, (conf_ab + conf_cd) / 2.0 * PHI / (PHI - 0.2))

        # Partial match: structural similarity even if relation types differ
        structural_sim = 0.0
        # Length ratio similarity
        ratio_ab = len(a) / max(len(b), 1)
        ratio_cd = len(c) / max(len(d), 1)
        structural_sim += max(0, 1.0 - abs(ratio_ab - ratio_cd)) * 0.3

        # Character-level similarity within pairs
        def char_overlap(x, y):
            sx, sy = set(x.lower()), set(y.lower())
            return len(sx & sy) / max(len(sx | sy), 1)

        pair_sim = (char_overlap(a, c) + char_overlap(b, d)) / 2.0
        structural_sim += pair_sim * 0.4

        return structural_sim

    def complete_analogy(self, a: str, b: str, c: str,
                         candidates: List[str]) -> List[Tuple[str, float]]:
        """Given A:B :: C:?, rank candidates by analogical fit.

        Returns: sorted list of (candidate, score) from best to worst.
        """
        scored = []
        for d in candidates:
            score = self.score_analogy(a, b, c, d)
            scored.append((d, score))
        scored.sort(key=lambda x: -x[1])
        return scored

    def detect_analogy_in_question(self, question: str) -> Optional[Dict[str, str]]:
        """Detect if a question contains an analogy pattern.

        Patterns: "A is to B as C is to ?" or "A:B :: C:?"
        Returns: {"a": ..., "b": ..., "c": ...} or None
        """
        # Pattern: "A is to B as C is to"
        m = re.search(
            r'(\w+)\s+is\s+to\s+(\w+)\s+as\s+(\w+)\s+is\s+to',
            question, re.IGNORECASE,
        )
        if m:
            return {"a": m.group(1), "b": m.group(2), "c": m.group(3)}

        # Pattern: "A:B :: C:?"
        m = re.search(r'(\w+)\s*:\s*(\w+)\s*(?:::?|as)\s*(\w+)\s*:', question)
        if m:
            return {"a": m.group(1), "b": m.group(2), "c": m.group(3)}

        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM LAYER: TextRank Extractive Summarizer
# ═══════════════════════════════════════════════════════════════════════════════

class TextRankSummarizer:
    """Graph-based extractive summarization using TextRank algorithm.

    Builds a sentence similarity graph and applies iterative PageRank-style
    scoring to select the most important sentences.

    Algorithm:
      1. Split text into sentences
      2. Compute pairwise sentence similarity (word overlap + IDF weighting)
      3. Build similarity graph (adjacency matrix)
      4. Run power iteration to compute sentence importance scores
      5. Select top-k sentences as summary

    Used in comprehension to distill the key information from passages.
    """

    def __init__(self, damping: float = 0.85, max_iterations: int = 100,
                 convergence: float = 1e-5):
        self.damping = damping
        self.max_iterations = max_iterations
        self.convergence = convergence

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    @staticmethod
    def _sentence_words(sentence: str) -> Set[str]:
        stops = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'and', 'or', 'but',
                 'it', 'its', 'this', 'that', 'with', 'as', 'has', 'have', 'had'}
        return {w for w in re.findall(r'\w+', sentence.lower()) if len(w) > 2 and w not in stops}

    def _sentence_similarity(self, s1: str, s2: str) -> float:
        """Compute similarity between two sentences (word overlap Jaccard)."""
        w1 = self._sentence_words(s1)
        w2 = self._sentence_words(s2)
        if not w1 or not w2:
            return 0.0
        intersection = w1 & w2
        # Normalized overlap (not pure Jaccard — favors larger overlap)
        return len(intersection) / (math.log(len(w1) + 1) + math.log(len(w2) + 1) + 1e-9)

    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Build pairwise sentence similarity matrix."""
        n = len(sentences)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._sentence_similarity(sentences[i], sentences[j])
                matrix[i][j] = sim
                matrix[j][i] = sim
        return matrix

    def _power_iteration(self, matrix: np.ndarray) -> np.ndarray:
        """Run PageRank-style power iteration on similarity matrix."""
        n = matrix.shape[0]
        if n == 0:
            return np.array([])

        # Normalize columns (transition matrix)
        col_sums = matrix.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        transition = matrix / col_sums

        # Initialize uniform scores
        scores = np.ones(n) / n

        for _ in range(self.max_iterations):
            new_scores = (1 - self.damping) / n + self.damping * transition @ scores
            delta = np.abs(new_scores - scores).sum()
            scores = new_scores
            if delta < self.convergence:
                break

        return scores

    def summarize(self, text: str, num_sentences: int = 3) -> Dict[str, Any]:
        """Extract the most important sentences from text.

        Returns:
            {
                "summary": str (joined key sentences),
                "sentences": List[{"text": str, "score": float, "rank": int}],
                "total_sentences": int,
                "compression_ratio": float
            }
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= num_sentences:
            return {
                "summary": text,
                "sentences": [{"text": s, "score": 1.0, "rank": i}
                              for i, s in enumerate(sentences)],
                "total_sentences": len(sentences),
                "compression_ratio": 1.0,
            }

        sim_matrix = self._build_similarity_matrix(sentences)
        scores = self._power_iteration(sim_matrix)

        # Rank sentences
        ranked = sorted(range(len(sentences)), key=lambda i: -scores[i])
        top_indices = sorted(ranked[:num_sentences])  # Preserve original order

        result_sentences = []
        for rank, idx in enumerate(ranked):
            result_sentences.append({
                "text": sentences[idx],
                "score": round(float(scores[idx]), 6),
                "rank": rank,
            })

        summary_text = ' '.join(sentences[i] for i in top_indices)

        return {
            "summary": summary_text,
            "sentences": sorted(result_sentences, key=lambda x: x["rank"]),
            "total_sentences": len(sentences),
            "compression_ratio": round(num_sentences / len(sentences), 3),
        }

    def extract_key_facts(self, facts: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Rank a list of facts by TextRank importance.

        Useful for selecting the most relevant facts from knowledge base.
        Returns: sorted list of (fact, importance_score).
        """
        if len(facts) <= top_k:
            return [(f, 1.0) for f in facts]

        combined = ' '.join(facts)
        sim_matrix = self._build_similarity_matrix(facts)
        scores = self._power_iteration(sim_matrix)

        scored = [(facts[i], float(scores[i])) for i in range(len(facts))]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM LAYER: Named Entity Recognizer
# ═══════════════════════════════════════════════════════════════════════════════

class NamedEntityRecognizer:
    """Rule-based Named Entity Recognition (NER) engine.

    Recognizes entity types: PERSON, LOCATION, ORGANIZATION, DATE, NUMBER,
    SCIENTIFIC_TERM, CONCEPT using:
      1. Capitalization patterns (Title Case = likely entity)
      2. Contextual keyword triggers ("Dr.", "University of", etc.)
      3. Date/number pattern matching
      4. Scientific term patterns (Latin/Greek roots, chemical formulas)
      5. Gazetteer lookup for common entities
    """

    _PERSON_TITLES = frozenset({
        'mr', 'mrs', 'ms', 'dr', 'prof', 'professor', 'president',
        'king', 'queen', 'prince', 'princess', 'sir', 'lord', 'lady',
        'saint', 'pope', 'emperor', 'empress', 'general', 'captain',
    })

    _ORG_KEYWORDS = frozenset({
        'university', 'institute', 'corporation', 'company', 'association',
        'foundation', 'organization', 'committee', 'council', 'department',
        'ministry', 'agency', 'bureau', 'commission', 'academy', 'society',
        'united', 'national', 'international', 'federal', 'royal',
    })

    _LOCATION_KEYWORDS = frozenset({
        'river', 'mountain', 'ocean', 'sea', 'lake', 'island', 'desert',
        'valley', 'peninsula', 'continent', 'country', 'city', 'state',
        'province', 'region', 'territory', 'bay', 'strait', 'gulf',
        'north', 'south', 'east', 'west', 'northern', 'southern',
    })

    _KNOWN_PERSONS = frozenset({
        'einstein', 'newton', 'darwin', 'shakespeare', 'aristotle', 'plato',
        'socrates', 'descartes', 'kant', 'hegel', 'marx', 'freud', 'jung',
        'curie', 'tesla', 'edison', 'galileo', 'copernicus', 'kepler',
        'bohr', 'heisenberg', 'schrodinger', 'dirac', 'feynman', 'hawking',
        'turing', 'babbage', 'lovelace', 'hopper', 'knuth', 'dijkstra',
        'napoleon', 'caesar', 'alexander', 'lincoln', 'washington',
    })

    _KNOWN_LOCATIONS = frozenset({
        'europe', 'asia', 'africa', 'americas', 'antarctica', 'australia',
        'london', 'paris', 'berlin', 'rome', 'tokyo', 'beijing', 'moscow',
        'atlantic', 'pacific', 'indian', 'arctic', 'mediterranean',
        'amazon', 'nile', 'mississippi', 'danube', 'ganges', 'thames',
        'himalayas', 'alps', 'andes', 'rockies', 'sahara',
    })

    def __init__(self):
        self._entities_found = 0

    def recognize(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text.

        Returns: List of {
            "text": str, "type": str, "start": int, "end": int, "confidence": float
        }
        """
        entities = []
        seen_spans: Set[Tuple[int, int]] = set()

        # 1. Date patterns
        for m in re.finditer(
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b', text
        ):
            span = (m.start(), m.end())
            if span not in seen_spans:
                entities.append({"text": m.group(), "type": "DATE",
                                 "start": m.start(), "end": m.end(), "confidence": 0.95})
                seen_spans.add(span)

        # Date patterns: month names
        for m in re.finditer(
            r'\b((?:January|February|March|April|May|June|July|August|September|'
            r'October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?)\b', text
        ):
            span = (m.start(), m.end())
            if span not in seen_spans:
                entities.append({"text": m.group(), "type": "DATE",
                                 "start": m.start(), "end": m.end(), "confidence": 0.92})
                seen_spans.add(span)

        # Year patterns
        for m in re.finditer(r'\b((?:1[0-9]|20)\d{2})\b', text):
            span = (m.start(), m.end())
            if span not in seen_spans:
                entities.append({"text": m.group(), "type": "DATE",
                                 "start": m.start(), "end": m.end(), "confidence": 0.70})
                seen_spans.add(span)

        # 2. Numbers with units
        for m in re.finditer(
            r'\b(\d+\.?\d*\s*(?:kg|km|m|cm|mm|g|mg|ml|L|°C|°F|K|Hz|eV|J|W|V|A|mol|atm|Pa))\b',
            text
        ):
            span = (m.start(), m.end())
            if span not in seen_spans:
                entities.append({"text": m.group(), "type": "QUANTITY",
                                 "start": m.start(), "end": m.end(), "confidence": 0.90})
                seen_spans.add(span)

        # 3. Chemical formulas (e.g., H2O, CO2, NaCl)
        for m in re.finditer(r'\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+)\b', text):
            span = (m.start(), m.end())
            if span not in seen_spans:
                entities.append({"text": m.group(), "type": "CHEMICAL",
                                 "start": m.start(), "end": m.end(), "confidence": 0.80})
                seen_spans.add(span)

        # 4. Title Case sequences (potential entities)
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
            span = (m.start(), m.end())
            if span in seen_spans:
                continue
            # Skip sentence-initial words
            if m.start() == 0 or text[m.start() - 2:m.start()].rstrip().endswith('.'):
                if len(m.group().split()) < 2:
                    continue

            entity_text = m.group()
            entity_lower = entity_text.lower()
            words = entity_lower.split()

            # Classify the entity
            etype = "ENTITY"
            conf = 0.55

            # Check gazetteers
            if any(w in self._KNOWN_PERSONS for w in words):
                etype = "PERSON"
                conf = 0.90
            elif any(w in self._KNOWN_LOCATIONS for w in words):
                etype = "LOCATION"
                conf = 0.88
            elif any(w in self._PERSON_TITLES for w in words):
                etype = "PERSON"
                conf = 0.82
            elif any(w in self._ORG_KEYWORDS for w in words):
                etype = "ORGANIZATION"
                conf = 0.80
            elif any(w in self._LOCATION_KEYWORDS for w in words):
                etype = "LOCATION"
                conf = 0.75
            elif len(words) >= 2:
                etype = "ENTITY"
                conf = 0.60

            if conf >= 0.55:
                entities.append({"text": entity_text, "type": etype,
                                 "start": m.start(), "end": m.end(),
                                 "confidence": conf})
                seen_spans.add(span)

        self._entities_found += len(entities)
        return entities

    def extract_entity_types(self, text: str) -> Dict[str, List[str]]:
        """Extract entities grouped by type.

        Returns: {"PERSON": [...], "LOCATION": [...], "DATE": [...], ...}
        """
        entities = self.recognize(text)
        grouped: Dict[str, List[str]] = defaultdict(list)
        for e in entities:
            if e["text"] not in grouped[e["type"]]:
                grouped[e["type"]].append(e["text"])
        return dict(grouped)


# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM LAYER: Edit Distance / Fuzzy Matcher
# ═══════════════════════════════════════════════════════════════════════════════

class LevenshteinMatcher:
    """Levenshtein edit distance and fuzzy string matching engine.

    Provides:
      1. Edit distance computation (insertions, deletions, substitutions)
      2. Normalized similarity score (0.0-1.0)
      3. Fuzzy matching against a corpus of candidates
      4. Weighted edit distance with transposition support (Damerau-Levenshtein)
    """

    @staticmethod
    def distance(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings.

        Uses dynamic programming (Wagner-Fischer algorithm).
        Time: O(mn), Space: O(min(m,n)).
        """
        if len(s1) < len(s2):
            return LevenshteinMatcher.distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]

    @staticmethod
    def damerau_distance(s1: str, s2: str) -> int:
        """Compute Damerau-Levenshtein distance (allows transpositions)."""
        len1, len2 = len(s1), len(s2)
        d = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            d[i][0] = i
        for j in range(len2 + 1):
            d[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                d[i][j] = min(
                    d[i - 1][j] + 1,       # deletion
                    d[i][j - 1] + 1,       # insertion
                    d[i - 1][j - 1] + cost  # substitution
                )
                if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                    d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)  # transposition

        return d[len1][len2]

    @staticmethod
    def similarity(s1: str, s2: str) -> float:
        """Normalized similarity score (0.0 = completely different, 1.0 = identical)."""
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1.0 - LevenshteinMatcher.distance(s1, s2) / max_len

    def fuzzy_match(self, query: str, candidates: List[str],
                    threshold: float = 0.6, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the best fuzzy matches for query among candidates.

        Args:
            query: The string to match
            candidates: List of candidate strings
            threshold: Minimum similarity to include (0.0-1.0)
            top_k: Maximum results to return

        Returns: Sorted list of (candidate, similarity_score)
        """
        q_lower = query.lower()
        scored = []
        for c in candidates:
            sim = self.similarity(q_lower, c.lower())
            if sim >= threshold:
                scored.append((c, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def best_match(self, query: str, candidates: List[str]) -> Tuple[str, float]:
        """Find the single best fuzzy match.

        Returns: (best_candidate, similarity_score) or ("", 0.0) if no candidates.
        """
        if not candidates:
            return ("", 0.0)
        results = self.fuzzy_match(query, candidates, threshold=0.0, top_k=1)
        return results[0] if results else ("", 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM LAYER: Latent Semantic Analyzer (LSA)
# ═══════════════════════════════════════════════════════════════════════════════

class LatentSemanticAnalyzer:
    """Latent Semantic Analysis (LSA) via truncated SVD on TF-IDF.

    Reduces the term-document matrix to a lower-dimensional concept space,
    enabling concept-level similarity that goes beyond bag-of-words:
      - "car" and "automobile" map to similar concept vectors
      - "bank" (financial) and "bank" (river) separate in concept space

    Algorithm:
      1. Build TF-IDF term-document matrix
      2. Apply truncated SVD: A ≈ U_k Σ_k V_k^T
      3. Project documents and queries into k-dimensional concept space
      4. Compute similarity in concept space (cosine)

    Uses numpy SVD for efficiency on moderately-sized corpora.
    """

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self._vocab: Dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._U: Optional[np.ndarray] = None
        self._Sigma: Optional[np.ndarray] = None
        self._Vt: Optional[np.ndarray] = None
        self._doc_vectors: Optional[np.ndarray] = None
        self._documents: List[str] = []
        self._fitted = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        stops = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'and', 'or', 'but',
                 'it', 'its', 'this', 'that', 'with', 'as', 'has', 'have', 'had',
                 'do', 'does', 'did', 'not', 'so', 'if', 'then', 'than', 'from'}
        return [w for w in re.findall(r'\w+', text.lower()) if len(w) > 2 and w not in stops]

    def fit(self, documents: List[str]):
        """Fit LSA on a document corpus.

        Builds the TF-IDF matrix and computes truncated SVD.
        """
        self._documents = documents
        n_docs = len(documents)
        if n_docs < 2:
            self._fitted = False
            return

        # Build vocabulary
        doc_tokens = [self._tokenize(d) for d in documents]
        word_counter: Counter = Counter()
        for tokens in doc_tokens:
            word_counter.update(set(tokens))

        vocab_words = [w for w, c in word_counter.most_common(2000) if c >= 2]
        self._vocab = {w: i for i, w in enumerate(vocab_words)}
        vocab_size = len(self._vocab)

        if vocab_size < 3:
            self._fitted = False
            return

        # Build TF-IDF matrix (docs × terms)
        tfidf = np.zeros((n_docs, vocab_size))
        df = np.zeros(vocab_size)

        for doc_idx, tokens in enumerate(doc_tokens):
            tf_count: Counter = Counter(tokens)
            for word, count in tf_count.items():
                if word in self._vocab:
                    widx = self._vocab[word]
                    tfidf[doc_idx, widx] = 1 + math.log(count)
                    df[widx] += 1

        # IDF
        self._idf = np.log(n_docs / (df + 1)) + 1
        tfidf *= self._idf

        # Normalize rows
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tfidf /= norms

        # Truncated SVD — randomized algorithm (Halko-Martinsson-Tropp 2011)
        # O(n_docs × vocab × k) instead of O(n_docs × vocab × min(n,m))
        k = min(self.n_components, min(tfidf.shape) - 1)
        if k < 1:
            self._fitted = False
            return

        try:
            # Randomized SVD: form random projection Q then compute SVD of small matrix
            rng = np.random.RandomState(104)
            n_oversamples = min(10, min(tfidf.shape) - k)
            n_random = k + n_oversamples

            # Step 1: Random projection to find column space
            Omega = rng.randn(tfidf.shape[1], n_random)
            Y = tfidf @ Omega                    # (n_docs, n_random)
            Q, _ = np.linalg.qr(Y)               # Orthonormal basis for range

            # Step 2: Project and compute small SVD
            B = Q.T @ tfidf                       # (n_random, vocab_size)
            U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)
            U = Q @ U_hat                         # Lift back to full space

            self._U = U[:, :k]
            self._Sigma = S[:k]
            self._Vt = Vt[:k, :]

            # Document vectors in concept space
            self._doc_vectors = self._U * self._Sigma
            self._fitted = True
        except Exception as e:
            _log.debug("LSA SVD failed: %s", e)
            self._fitted = False

    def _project_query(self, text: str) -> Optional[np.ndarray]:
        """Project a query into concept space."""
        if not self._fitted or self._Vt is None or self._idf is None:
            return None

        tokens = self._tokenize(text)
        tf_count: Counter = Counter(tokens)
        vec = np.zeros(len(self._vocab))
        for word, count in tf_count.items():
            if word in self._vocab:
                vec[self._vocab[word]] = (1 + math.log(count)) * self._idf[self._vocab[word]]

        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        vec /= norm

        # Project: q_concept = q^T V^T Σ^{-1}
        sigma_inv = np.diag(1.0 / (self._Sigma + 1e-10))
        return vec @ self._Vt.T @ sigma_inv

    def query_similarity(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Find the most similar documents to a query in concept space.

        Returns: List of (doc_index, cosine_similarity)
        """
        q_vec = self._project_query(query)
        if q_vec is None or self._doc_vectors is None:
            return []

        # Cosine similarity
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec /= q_norm

        doc_norms = np.linalg.norm(self._doc_vectors, axis=1)
        doc_norms[doc_norms == 0] = 1.0
        normed_docs = self._doc_vectors / doc_norms[:, np.newaxis]

        sims = normed_docs @ q_vec
        top_indices = np.argsort(-sims)[:top_k]

        return [(int(idx), float(sims[idx])) for idx in top_indices if sims[idx] > 0]

    def concept_similarity(self, text_a: str, text_b: str) -> float:
        """Compute concept-level similarity between two texts in LSA space."""
        vec_a = self._project_query(text_a)
        vec_b = self._project_query(text_b)
        if vec_a is None or vec_b is None:
            return 0.0

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM LAYER: Enhanced Lesk WSD Algorithm
# ═══════════════════════════════════════════════════════════════════════════════

class LeskDisambiguator:
    """Enhanced Lesk algorithm for Word Sense Disambiguation (WSD).

    Disambiguates polysemous words by comparing context overlap with
    dictionary definitions (glosses). Enhancements:
      1. Extended gloss overlap (includes synset relations: hypernyms, etc.)
      2. Weighted overlap (IDF-like weighting on gloss words)
      3. Contextual window adjustment (narrow vs. broad context)
      4. PHI-weighted confidence calibration

    Complements the DeepNLU ContextualDisambiguator with an algorithmic
    dictionary-based approach.
    """

    # Mini sense inventory (word → list of {sense_id, gloss, domain})
    _SENSE_INVENTORY: Dict[str, List[Dict[str, str]]] = {
        'bank': [
            {'sense': 'financial', 'gloss': 'financial institution money deposit withdraw savings loan interest account credit'},
            {'sense': 'river', 'gloss': 'edge side river stream water shore land slope embankment'},
            {'sense': 'storage', 'gloss': 'collection repository store supply data memory blood organ'},
        ],
        'light': [
            {'sense': 'illumination', 'gloss': 'electromagnetic radiation visible brightness lamp photon wavelength glow shine'},
            {'sense': 'weight', 'gloss': 'not heavy weight mass low gravity feather thin delicate'},
            {'sense': 'ignite', 'gloss': 'set fire start flame burn match candle ignite kindle'},
        ],
        'cell': [
            {'sense': 'biological', 'gloss': 'living organism unit membrane nucleus mitosis DNA organelle biology tissue'},
            {'sense': 'prison', 'gloss': 'room jail prison confinement prisoner detention incarceration bars'},
            {'sense': 'battery', 'gloss': 'electrical power energy battery voltage electrochemical fuel solar galvanic'},
            {'sense': 'phone', 'gloss': 'mobile telephone communication cellular wireless device phone call network'},
        ],
        'plant': [
            {'sense': 'botanical', 'gloss': 'living organism photosynthesis flower tree root leaf seed grow garden soil'},
            {'sense': 'factory', 'gloss': 'industrial facility manufacturing production factory equipment operation processing'},
            {'sense': 'place', 'gloss': 'put in position set install place establish fix embed locate'},
        ],
        'run': [
            {'sense': 'locomotion', 'gloss': 'move fast legs sprint jog race dash exercise marathon'},
            {'sense': 'operate', 'gloss': 'operate function machine execute program software computer process system'},
            {'sense': 'manage', 'gloss': 'manage lead direct control business organization company supervise'},
            {'sense': 'flow', 'gloss': 'flow liquid water stream river current course drip pour'},
        ],
        'charge': [
            {'sense': 'electrical', 'gloss': 'electrical charge electron positive negative coulomb current voltage battery'},
            {'sense': 'cost', 'gloss': 'price fee cost payment amount bill expense rate'},
            {'sense': 'accusation', 'gloss': 'criminal accusation crime offense indictment prosecute allegation'},
            {'sense': 'attack', 'gloss': 'rush attack advance assault cavalry battle military charge forward'},
        ],
        'scale': [
            {'sense': 'measurement', 'gloss': 'measurement size proportion range magnitude degree extent level'},
            {'sense': 'music', 'gloss': 'musical notes sequence pitch tone ascending descending octave key'},
            {'sense': 'climb', 'gloss': 'climb ascend mount go up wall mountain ladder height'},
            {'sense': 'fish', 'gloss': 'fish covering skin plate protective body surface reptile'},
            {'sense': 'weight', 'gloss': 'weighing device balance instrument mass measure kilogram pound'},
        ],
        'field': [
            {'sense': 'area_land', 'gloss': 'open land grass meadow farm agriculture crop pasture land area'},
            {'sense': 'domain', 'gloss': 'area expertise study discipline subject domain branch specialty knowledge'},
            {'sense': 'physics', 'gloss': 'force region space electromagnetic gravitational quantum electric magnetic'},
        ],
        'power': [
            {'sense': 'physics', 'gloss': 'physics energy force work rate watt joule electricity output'},
            {'sense': 'authority', 'gloss': 'authority control influence governance political government rule dominion'},
            {'sense': 'math', 'gloss': 'mathematics exponent raise number squared cubed index'},
        ],
        'model': [
            {'sense': 'representation', 'gloss': 'representation simulation mathematical approximation abstract theory framework'},
            {'sense': 'fashion', 'gloss': 'person fashion display clothing beauty runway photoshoot appearance'},
            {'sense': 'example', 'gloss': 'ideal example standard paradigm template pattern prototype'},
        ],
        'state': [
            {'sense': 'condition', 'gloss': 'condition status situation form phase mode circumstance'},
            {'sense': 'political', 'gloss': 'country nation government political territory sovereignty region'},
            {'sense': 'declare', 'gloss': 'say declare express announce assert mention communicate'},
        ],
        'bond': [
            {'sense': 'chemical', 'gloss': 'chemical linkage covalent ionic molecular atom electron sharing'},
            {'sense': 'financial', 'gloss': 'financial investment security debt government treasury obligation yield'},
            {'sense': 'connection', 'gloss': 'connection relationship tie link attachment emotional social bond'},
        ],
    }

    def __init__(self):
        self._disambiguations = 0

    def _context_words(self, text: str, target_word: str,
                       window: int = 10) -> Set[str]:
        """Extract context words around the target word."""
        words = re.findall(r'\w+', text.lower())
        target_l = target_word.lower()
        try:
            idx = words.index(target_l)
        except ValueError:
            # Word not found exactly — use all words as context
            return set(words)

        start = max(0, idx - window)
        end = min(len(words), idx + window + 1)
        context = set(words[start:end])
        context.discard(target_l)
        return context

    def disambiguate(self, word: str, context: str,
                     window: int = 10) -> Dict[str, Any]:
        """Disambiguate a word given its context using Enhanced Lesk.

        Returns:
            {
                "word": str,
                "selected_sense": str,
                "confidence": float,
                "all_senses": List[{"sense": str, "score": float}],
                "context_overlap": int
            }
        """
        word_l = word.lower()
        senses = self._SENSE_INVENTORY.get(word_l, [])

        if not senses:
            return {
                "word": word,
                "selected_sense": "default",
                "confidence": 0.3,
                "all_senses": [],
                "context_overlap": 0,
            }

        ctx_words = self._context_words(context, word, window)
        sense_scores = []

        for sense_info in senses:
            gloss_words = set(sense_info['gloss'].split())

            # Basic Lesk: overlap between context and gloss
            overlap = ctx_words & gloss_words
            base_score = len(overlap)

            # IDF-like weighting: rare overlapping words score higher
            weighted_score = sum(
                1.0 / math.log(2 + len(gloss_words))
                for w in overlap if len(w) > 3
            ) + base_score * 0.5

            # Extended gloss: check if context words are substrings of gloss words
            extended = sum(
                0.3 for cw in ctx_words
                for gw in gloss_words
                if len(cw) > 4 and len(gw) > 4 and (cw in gw or gw in cw) and cw not in overlap
            )

            total = weighted_score + extended
            sense_scores.append({
                "sense": sense_info['sense'],
                "score": round(total, 4),
                "overlap_words": list(overlap)[:5],
            })

        sense_scores.sort(key=lambda x: -x['score'])
        self._disambiguations += 1

        best = sense_scores[0]
        total_score = sum(s['score'] for s in sense_scores)
        confidence = best['score'] / max(total_score, 0.01)
        # PHI calibration
        confidence = min(0.98, confidence * PHI / (PHI + 0.2))

        return {
            "word": word,
            "selected_sense": best['sense'],
            "confidence": round(confidence, 4),
            "all_senses": [{"sense": s['sense'], "score": s['score']} for s in sense_scores],
            "context_overlap": sum(len(s.get('overlap_words', [])) for s in sense_scores),
        }

    def disambiguate_all(self, text: str) -> List[Dict[str, Any]]:
        """Find and disambiguate all known polysemous words in text."""
        text_words = set(re.findall(r'\w+', text.lower()))
        polysemous = text_words & set(self._SENSE_INVENTORY.keys())

        results = []
        for word in polysemous:
            result = self.disambiguate(word, text)
            results.append(result)
        return results


# ═══════════════════════════════════════════════════════════════════════════════
#  v8.0.0 ALGORITHM LAYER — 7 additional comprehension algorithms
# ═══════════════════════════════════════════════════════════════════════════════

