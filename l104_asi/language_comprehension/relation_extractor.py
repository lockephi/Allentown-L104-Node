from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

class RelationTripleExtractor:
    """Extract and index (subject, predicate, object) triples from natural language facts.

    v5.0 ALGORITHMIC COMPONENT: Replaces brittle regex patterns in _score_choice
    with structured knowledge triples that can be matched algebraically.

    Uses pattern-based extraction for common knowledge forms:
    - "X is/are Y" → (x, is, y)
    - "X wrote Y" → (x, wrote, y)
    - "X stands for Y" → (x, stands_for, y)
    - "X discovered/invented Y" → (x, created, y)
    """

    _RAW_PATTERNS = [
        # v5.1 FIX: Use [^,;.()] negated char classes instead of . to prevent
        # catastrophic backtracking when indexing 10K+ facts.
        (r'(?:the\s+)?([^,;()\n]{2,60})\s+(?:is|are|was|were)\s+(?:the\s+|a\s+|an\s+)?([^,;()\n]{2,80})', 'is'),
        (r'(\w[\w\s]{1,30})\s+wrote\s+([^,;.\n]{2,60})', 'wrote'),
        (r'(\w[\w\s]{1,30})\s+(?:discovered|invented|developed|created|founded|established)\s+([^,;.\n]{2,60})', 'created'),
        (r'(\w{1,10})\s+stands\s+for\s+([^,;.\n]{2,60})', 'stands_for'),
        (r'(\w[\w\s]{1,30})\s+uses?\s+([^,;.\n]{2,40})', 'uses'),
        (r'(\w[\w\s]{1,40})\s+(?:contains?|consists?\s+of)\s+([^,;.\n]{2,60})', 'contains'),
        (r'(\w[\w\s]{1,40})\s+(?:causes?|leads?\s+to|results?\s+in)\s+([^,;.\n]{2,60})', 'causes'),
        (r'(\w[\w\s]{1,30})\s+is\s+(?:located|found|situated)\s+in\s+([^,;.\n]{2,40})', 'located_in'),
        (r'^(\w[\w\s]{1,30}):\s+([^,;.\n]{5,80})', 'defined_as'),
        (r'(\w[\w\s\^\/\*\+\-]{1,30})\s*=\s*([^,;.\n]{2,60})', 'equals'),
        (r'(?:the\s+)?(?:symbol|chemical\s+symbol|formula)\s+(?:for|of)\s+(\w[\w\s]{1,30})\s+is\s+(\w{1,10})', 'symbol_of'),
        (r'(\w[\w\s]{1,30})\s+(?:published|proposed|formulated)\s+([^,;.\n]{2,60})', 'published'),
    ]
    # Pre-compile all patterns once (class-level) for O(10K) fact indexing perf
    COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), pred) for p, pred in _RAW_PATTERNS]

    # Max fact length to regex-parse (longer facts cause backtracking)
    _MAX_FACT_LEN = 300

    def __init__(self):
        self._triples: List[Tuple[str, str, str]] = []
        self._subject_index: Dict[str, List[int]] = defaultdict(list)
        self._object_index: Dict[str, List[int]] = defaultdict(list)
        self._predicate_index: Dict[str, List[int]] = defaultdict(list)

    def extract_from_fact(self, fact: str) -> List[Tuple[str, str, str]]:
        """Extract (subject, predicate, object) triples from a single fact."""
        triples = []
        fact_lower = fact.lower().strip()
        if len(fact_lower) > self._MAX_FACT_LEN:
            fact_lower = fact_lower[:self._MAX_FACT_LEN]
        for compiled_re, predicate in self.COMPILED_PATTERNS:
            m = compiled_re.search(fact_lower)
            if m:
                groups = m.groups()
                if len(groups) >= 2:
                    subj = groups[0].strip()
                    obj = groups[-1].strip()
                    if len(subj) > 1 and len(obj) > 1 and subj != obj:
                        triples.append((subj, predicate, obj))
        return triples

    def index_all_facts(self, facts: List[str]):
        """Extract and index triples from all provided facts."""
        for fact in facts:
            extracted = self.extract_from_fact(fact)
            for subj, pred, obj in extracted:
                idx = len(self._triples)
                self._triples.append((subj, pred, obj))
                for word in subj.split():
                    if len(word) > 2:
                        self._subject_index[word].append(idx)
                self._subject_index[subj].append(idx)
                for word in obj.split():
                    if len(word) > 2:
                        self._object_index[word].append(idx)
                self._object_index[obj].append(idx)
                self._predicate_index[pred].append(idx)

    def query_by_subject(self, subject: str, top_k: int = 10) -> List[Tuple[str, str, str]]:
        """Find triples where the subject matches."""
        results = []
        seen = set()
        subject_lower = subject.lower()
        for word in subject_lower.split():
            if len(word) > 2:
                for idx in self._subject_index.get(word, []):
                    if idx not in seen:
                        seen.add(idx)
                        results.append(self._triples[idx])
        for idx in self._subject_index.get(subject_lower, []):
            if idx not in seen:
                seen.add(idx)
                results.append(self._triples[idx])
        return results[:top_k]

    def query_by_object(self, obj: str, top_k: int = 10) -> List[Tuple[str, str, str]]:
        """Find triples where the object matches."""
        results = []
        seen = set()
        obj_lower = obj.lower()
        for word in obj_lower.split():
            if len(word) > 2:
                for idx in self._object_index.get(word, []):
                    if idx not in seen:
                        seen.add(idx)
                        results.append(self._triples[idx])
        for idx in self._object_index.get(obj_lower, []):
            if idx not in seen:
                seen.add(idx)
                results.append(self._triples[idx])
        return results[:top_k]

    def score_alignment(self, question: str, choice: str) -> float:
        """Score how well a choice aligns with extracted triples for a question.

        Uses structured triple matching: question words match subject,
        choice words match object (or vice versa).
        """
        q_words = {w for w in re.findall(r'\w+', question.lower()) if len(w) > 2}
        c_words = {w for w in re.findall(r'\w+', choice.lower()) if len(w) > 2}
        choice_lower = choice.lower().strip()
        score = 0.0

        for subj, pred, obj in self._triples:
            subj_words = set(subj.split())
            obj_words = set(obj.split())

            q_in_subj = len(q_words & subj_words)
            q_in_obj = len(q_words & obj_words)
            c_in_subj = len(c_words & subj_words)
            c_in_obj = len(c_words & obj_words)

            # Strong: question→subject, choice→object
            if q_in_obj >= 1 and c_in_subj >= 1:
                bonus = q_in_obj * c_in_subj * 0.8
                if choice_lower == subj or choice_lower in subj:
                    bonus *= 2.0
                score += bonus

            # Strong: question→object, choice→subject (reverse)
            if q_in_subj >= 1 and c_in_obj >= 1:
                bonus = q_in_subj * c_in_obj * 0.8
                if choice_lower == obj or choice_lower in obj:
                    bonus *= 2.0
                score += bonus

            # Predicate-specific patterns
            if pred == 'wrote' and q_in_obj >= 1 and c_in_subj >= 1:
                score += 3.0
            elif pred == 'symbol_of' and q_in_subj >= 1 and c_in_obj >= 1:
                score += 3.0
            elif pred == 'stands_for' and any(w in question.lower() for w in subj.split()):
                if choice_lower in obj or obj in choice_lower:
                    score += 3.0

        return score

    def get_status(self) -> Dict[str, Any]:
        """Get triple extractor statistics."""
        return {
            "total_triples": len(self._triples),
            "unique_subjects": len(self._subject_index),
            "unique_objects": len(self._object_index),
            "unique_predicates": len(self._predicate_index),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 4: BM25 RELEVANCE RANKING
# ═══════════════════════════════════════════════════════════════════════════════
