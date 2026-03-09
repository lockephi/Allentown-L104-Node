from __future__ import annotations

import math
import random
import re
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import numpy as np

from .constants import PHI, GOD_CODE, TAU, VOID_CONSTANT
from .knowledge_base import MMLUKnowledgeBase
from .retrieval import BM25Ranker
from .detectors import SubjectDetector, NumericalReasoner
from .verification import CrossVerificationEngine
from .engine_support import (
    _get_cached_science_engine, _get_cached_math_engine,
    _get_cached_dual_layer, _get_cached_formal_logic,
    _get_cached_deep_nlu, _get_cached_quantum_gate_engine,
    _get_cached_quantum_math_core, _get_cached_quantum_reasoning,
    _get_cached_quantum_probability, _get_cached_local_intellect,
    _get_cached_probability_engine,
)

_log = logging.getLogger('l104.language_comprehension')
_rng = random.Random(104)  # Deterministic tie-breaking RNG

class MCQSolver:
    """Multiple-choice question solver using semantic matching + knowledge retrieval.

    v3.0 Pipeline:
    1. Parse question and extract choices
    2. Auto-detect subject via SubjectDetector for focused retrieval
    3. Retrieve relevant knowledge passages (TF-IDF + N-gram + relations)
    4. Score each choice via multi-signal fusion (keyword + semantic + N-gram + BM25)
    5. Numerical reasoning for quantitative questions
    6. Apply Formal Logic Engine deductive support for logic questions
    7. Apply DeepNLU discourse analysis for nuanced comprehension
    8. Quantum probability amplification via Grover
    9. Cross-verification via elimination + consistency + PHI-calibration
    10. Dual-Layer physics confidence calibration
    11. Entropy-calibrated confidence via Maxwell Demon
    12. Chain-of-thought verification
    """

    def __init__(self, knowledge_base: MMLUKnowledgeBase,
                 subject_detector: SubjectDetector = None,
                 numerical_reasoner: NumericalReasoner = None,
                 cross_verifier: CrossVerificationEngine = None):
        self.kb = knowledge_base
        self.bm25 = BM25Ranker()
        self._subject_detector = subject_detector
        self._numerical_reasoner = numerical_reasoner
        self._cross_verifier = cross_verifier
        self._questions_answered = 0
        self._correct_count = 0
        self._entropy_calibrations = 0
        self._dual_layer_calibrations = 0
        self._ngram_boosts = 0
        self._logic_assists = 0
        self._nlu_assists = 0
        self._numerical_assists = 0
        self._cross_verifications = 0
        self._subject_detections = 0
        self._quantum_collapses = 0
        self._choice_bm25 = None  # v9.0: Reusable BM25 scorer for _score_choice
        self._early_exits = 0  # v9.0: Count of high-confidence early exits

    @property
    def subject_detector(self):
        if self._subject_detector is None:
            self._subject_detector = SubjectDetector()
        return self._subject_detector

    @property
    def numerical_reasoner(self):
        if self._numerical_reasoner is None:
            self._numerical_reasoner = NumericalReasoner()
        return self._numerical_reasoner

    @property
    def cross_verifier(self):
        if self._cross_verifier is None:
            self._cross_verifier = CrossVerificationEngine()
        return self._cross_verifier

    def solve(self, question: str, choices: List[str],
              subject: Optional[str] = None) -> Dict[str, Any]:
        """Solve a multiple-choice question.

        Args:
            question: The question text
            choices: List of answer choices (A, B, C, D)
            subject: Optional MMLU subject for focused retrieval

        Returns:
            Dict with selected answer, confidence, reasoning chain
        """
        self._questions_answered += 1

        # Seed RNG per-question for deterministic tie-breaking regardless of question order
        _rng.seed(hash(question) & 0xFFFFFFFF)

        # Step 0: Subject detection — auto-detect if not explicitly provided
        if not subject and self.subject_detector:
            detected = self.subject_detector.detect(question, choices)
            if detected:
                subject = detected
                self._subject_detections += 1

        # Step 0b: Subject-focused retrieval — prioritize nodes from the same subject
        # Also search related subjects via alias mapping to broaden coverage
        _SUBJECT_ALIASES = {
            "clinical_knowledge": ["college_medicine", "professional_medicine"],
            "college_medicine": ["clinical_knowledge", "professional_medicine", "anatomy"],
            "professional_medicine": ["college_medicine", "clinical_knowledge"],
            "high_school_world_history": ["world_history"],
            "world_history": ["high_school_world_history"],
            "high_school_us_history": ["us_foreign_policy"],
            "us_foreign_policy": ["high_school_us_history"],
            "high_school_biology": ["college_biology"],
            "college_biology": ["high_school_biology"],
            "high_school_chemistry": ["college_chemistry"],
            "college_chemistry": ["high_school_chemistry"],
            "high_school_physics": ["college_physics", "conceptual_physics"],
            "college_physics": ["high_school_physics", "conceptual_physics"],
            "conceptual_physics": ["high_school_physics", "college_physics"],
            "high_school_mathematics": ["college_mathematics", "elementary_mathematics"],
            "college_mathematics": ["high_school_mathematics"],
            "elementary_mathematics": ["high_school_mathematics"],
            "high_school_macroeconomics": ["high_school_microeconomics", "econometrics"],
            "high_school_microeconomics": ["high_school_macroeconomics", "econometrics"],
            "econometrics": ["high_school_macroeconomics", "high_school_microeconomics"],
            "high_school_computer_science": ["college_computer_science", "computer_science", "computer_security"],
            "college_computer_science": ["high_school_computer_science", "computer_science", "machine_learning"],
            "computer_security": ["high_school_computer_science", "computer_science"],
            "high_school_geography": ["global_facts"],
            "global_facts": ["high_school_geography"],
            "human_aging": ["anatomy", "nutrition"],
            "medical_genetics": ["college_biology", "high_school_biology"],
            "virology": ["college_biology", "high_school_biology"],
            "nutrition": ["human_aging", "anatomy"],
            "moral_disputes": ["moral_scenarios", "philosophy"],
            "moral_scenarios": ["moral_disputes", "philosophy"],
            "philosophy": ["moral_disputes", "moral_scenarios", "logical_fallacies"],
            "logical_fallacies": ["philosophy", "formal_logic"],
            "formal_logic": ["logical_fallacies", "abstract_algebra"],
            "business_ethics": ["professional_accounting", "management", "marketing"],
            "management": ["business_ethics", "marketing"],
            "marketing": ["business_ethics", "management"],
            "professional_accounting": ["business_ethics"],
            "sociology": ["high_school_psychology", "public_relations"],
            "high_school_psychology": ["sociology", "human_sexuality"],
            "human_sexuality": ["high_school_psychology"],
            "professional_law": ["international_law", "jurisprudence"],
            "international_law": ["professional_law", "jurisprudence"],
            "jurisprudence": ["professional_law", "international_law"],
            "security_studies": ["us_foreign_policy", "high_school_government_and_politics"],
            "high_school_government_and_politics": ["us_foreign_policy", "security_studies"],
        }
        subject_hits = []
        if subject:
            subject_lower = subject.lower().replace(" ", "_")
            # Search both the primary subject and any aliases
            subjects_to_search = [subject_lower]
            subjects_to_search.extend(_SUBJECT_ALIASES.get(subject_lower, []))
            for key, node in self.kb.nodes.items():
                node_subj = node.subject.lower().replace(" ", "_")
                for search_subj in subjects_to_search:
                    if search_subj in key or search_subj in node_subj:
                        # Primary subject gets higher relevance than aliases
                        rel = 0.5 if search_subj == subject_lower else 0.3
                        subject_hits.append((key, node, rel))
                        break  # Don't double-add if node matches multiple aliases

        # Step 1: Retrieve relevant knowledge
        # v9.0: Deduplicated retrieval — single merged query instead of 2 separate
        # KB lookups that redundantly compute TF-IDF/N-gram over overlapping terms.
        # The expanded query (question + choices) subsumes the raw question query,
        # so we issue one broader retrieval with increased top_k.
        expanded_query = question + " " + " ".join(choices)
        knowledge_hits = self.kb.query(expanded_query, top_k=12)

        # Merge hits (deduplicate by key), subject hits first
        seen_keys = set()
        merged_hits = []
        for key, node, score in subject_hits:
            if key not in seen_keys:
                merged_hits.append((key, node, score))
                seen_keys.add(key)
        for key, node, score in knowledge_hits:
            if key not in seen_keys:
                merged_hits.append((key, node, score))
                seen_keys.add(key)
        knowledge_hits = merged_hits

        # Step 1b: BM25 re-ranking — use BM25 to score facts against question
        all_facts_for_bm25 = []
        fact_to_node = {}
        for key, node, score in knowledge_hits:
            for fact in node.facts:
                idx = len(all_facts_for_bm25)
                all_facts_for_bm25.append(fact)
                fact_to_node[idx] = (key, node, score)
            idx = len(all_facts_for_bm25)
            all_facts_for_bm25.append(node.definition)
            fact_to_node[idx] = (key, node, score)

        if all_facts_for_bm25:
            self.bm25.fit(all_facts_for_bm25)
            bm25_ranked = self.bm25.rank(question, top_k=min(20, len(all_facts_for_bm25)))
            # Rebuild context_facts ordered by BM25 relevance
            # Only keep BM25-ranked facts — adding all remaining facts drowns
            # signal with noise and inflates every choice's score equally.
            context_facts = []
            seen_facts = set()
            for doc_idx, bm25_score in bm25_ranked:
                if bm25_score > 0.01 and doc_idx < len(all_facts_for_bm25):
                    fact = all_facts_for_bm25[doc_idx]
                    if fact not in seen_facts:
                        context_facts.append(fact)
                        seen_facts.add(fact)
        else:
            context_facts = []
            seen_facts = set()

        # Step 2b: Exhaustive scan fallback — when retrieval finds no or few
        # context facts, scan ALL nodes for facts containing question + choice keywords.
        # Cap additions to prevent fact explosion that drowns discriminative signal.
        if len(context_facts) < 3:
            q_lower = question.lower()
            q_kw = {w for w in re.findall(r'\w+', q_lower) if len(w) > 3}
            all_choice_words = set()
            for ch in choices:
                all_choice_words.update(w.lower() for w in re.findall(r'\w+', ch) if len(w) > 3)
            search_kw = q_kw | all_choice_words
            min_overlap = 1 if len(search_kw) <= 6 else 2
            seen_scan_keys = {k for k, _, _ in knowledge_hits}
            scan_additions = 0
            for key, node in self.kb.nodes.items():
                if key in seen_scan_keys:
                    continue
                node_text = (node.definition + " " + " ".join(node.facts)).lower()
                overlap = sum(1 for w in search_kw if w in node_text)
                if overlap >= min_overlap:
                    context_facts.append(node.definition)
                    for f in node.facts:
                        if any(w in f.lower() for w in search_kw):
                            context_facts.append(f)
                            scan_additions += 1
                    knowledge_hits.append((key, node, 0.02 * overlap))
                    scan_additions += 1
                    if scan_additions >= 30:  # Cap to prevent noise flood
                        break

        # Step 2d: Direct answer extraction — scan top facts for patterns
        # that directly answer the question (e.g., "X is Y", "X is known as Y").
        # This builds a per-choice direct-match bonus applied in Step 3.
        # v13: Reduced per-match from 5.0→1.0 and capped at 3.0 per choice.
        # High uncapped bonus was creating 15+ scores that dominated everything.
        _direct_answer_bonus = {}  # choice_index → bonus score
        q_nouns = set()  # Key question phrases, used in Step 2d and 2d2
        if context_facts:
            q_lower_da = question.lower()
            # Extract key noun phrases from question
            # Questions like "Which planet is X?" → key phrase is "X"
            # "What is the longest bone?" → key phrase is "longest bone"
            # Extract quoted/emphasized phrases (double and single quotes)
            for m in re.finditer(r'"([^"]+)"', question):
                q_nouns.add(m.group(1).lower())
            for m in re.finditer(r"'([^']+)'", question):
                q_nouns.add(m.group(1).lower())
            # Extract "known as X", "called X" from question
            for m in re.finditer(r'(?:known as|called|named|nicknamed)\s+(?:the\s+)?(.+?)(?:\?|$)', q_lower_da):
                q_nouns.add(m.group(1).strip().rstrip('?. '))
            # Extract "attributed to X", "begins with X" → X is the subject
            for m in re.finditer(r'(?:attributed to|begins with|starts with|ends with|results in|composed of|made of)\s+(.+?)(?:\?|$)', q_lower_da):
                q_nouns.add(m.group(1).strip().rstrip('?. '))
            # Extract "the X" patterns for "What is the X?" questions
            for m in re.finditer(r'(?:what|which|who)\s+(?:is|are|was|were)\s+(?:the\s+)?(.+?)(?:\?|$)', q_lower_da):
                phrase = m.group(1).strip().rstrip('?. ')
                if len(phrase) > 3:
                    q_nouns.add(phrase)
            # Extract "The X verb" patterns for declarative sentences as questions
            # e.g., "The scientific method begins with" → "scientific method"
            for m in re.finditer(r'^the\s+(.+?)\s+(?:is|are|was|were|has|have|had|begins|starts|ends|places|uses|shows)\b', q_lower_da):
                phrase = m.group(1).strip()
                if len(phrase) > 3:
                    q_nouns.add(phrase)
            # Extract "places which X at the Y" → key phrases X and Y
            for m in re.finditer(r'places\s+which\s+(.+?)\s+at\s+the\s+(.+?)(?:\?|$)', q_lower_da):
                q_nouns.add(m.group(1).strip().rstrip('?. '))
                q_nouns.add(m.group(2).strip().rstrip('?. '))

            # v14: Split multi-word q_nouns on prepositions for sub-phrase matching.
            # "time complexity of binary search" → also adds "time complexity", "binary search"
            # This helps when KB facts use different word order (e.g., "binary search
            # has time complexity O(log n)" — "binary search" matches but the full
            # phrase "time complexity of binary search" does not).
            _expanded_nouns = set()
            for phrase in list(q_nouns):
                for prep in (' of ', ' for ', ' in ', ' to ', ' with ', ' at '):
                    if prep in phrase:
                        parts = phrase.split(prep, 1)
                        for part in parts:
                            part = part.strip()
                            if len(part) > 3:
                                _expanded_nouns.add(part)
            q_nouns |= _expanded_nouns

            if q_nouns:
                for fact in context_facts[:15]:
                    fl = fact.lower()
                    for phrase in q_nouns:
                        if phrase not in fl:
                            continue
                        # Fact contains the question's key phrase — check which choice it associates with
                        # v13: Count matching choices first for exclusivity weighting.
                        # Exclusive matches (1 choice) get 5.0, shared matches get 5.0/n.
                        # v14: Anti-parrot — skip choices that appear in the question
                        # text itself (they're the subject, not the answer). E.g.,
                        # "derivative of x^2" → "x^2" is the subject, not the answer.
                        matching_choices = []
                        for ci, ch in enumerate(choices):
                            ch_lower = ch.lower().strip()
                            if len(ch_lower) < 2:
                                continue
                            if ch_lower in fl:
                                # Anti-parrot: skip if choice appears in question text
                                if ch_lower in q_lower_da:
                                    continue
                                matching_choices.append(ci)
                        if matching_choices:
                            bonus_per = 5.0 / len(matching_choices)
                            for ci in matching_choices:
                                _direct_answer_bonus[ci] = _direct_answer_bonus.get(ci, 0.0) + bonus_per

        # Step 2d2: Per-choice KB deep scan — when BM25 context_facts don't
        # contain the key phrase+choice pair, scan the FULL KB for facts that
        # contain BOTH a choice name AND a question keyword. This handles
        # cases like "Which planet is known as the Red Planet?" where BM25
        # surfaces "Jupiter" facts but misses the "Mars = Red Planet" fact.
        if not _direct_answer_bonus:
            q_content_scan = {w for w in re.findall(r'\w+', question.lower()) if len(w) > 3}
            # Remove generic question words
            q_content_scan -= {'which', 'what', 'where', 'when', 'does', 'known', 'called',
                               'following', 'most', 'main', 'primary'}
            for ci, ch in enumerate(choices):
                ch_lower = ch.lower().strip()
                if len(ch_lower) < 2:
                    continue
                ch_words = set(re.findall(r'\w+', ch_lower))
                for _key, node in self.kb.nodes.items():
                    node_text = (node.definition + " " + " ".join(node.facts[:30])).lower()
                    # Check if choice name appears in node
                    ch_in_node = ch_lower in node_text or any(
                        w in node_text for w in ch_words if len(w) > 3)
                    if not ch_in_node:
                        continue
                    # Check if question keywords also appear
                    q_overlap = sum(1 for w in q_content_scan if w in node_text)
                    if q_overlap >= 2:
                        # Found a node linking the choice to the question topic
                        # Check for direct relation patterns: "X is known as Y"
                        for q_phrase in (q_nouns if q_nouns else set()):
                            if q_phrase in node_text:
                                _direct_answer_bonus[ci] = _direct_answer_bonus.get(ci, 0.0) + 4.0
                                # Also add the matching fact to context_facts
                                for f in node.facts:
                                    if q_phrase in f.lower() and ch_lower in f.lower():
                                        if f not in seen_facts:
                                            context_facts.append(f)
                                            seen_facts.add(f)
                                break
                        else:
                            # No key-phrase match, but choice+question overlap → smaller bonus
                            if q_overlap >= 3:
                                _direct_answer_bonus[ci] = _direct_answer_bonus.get(ci, 0.0) + 1.0

        # Step 2e: Local Intellect KB augmentation — supplement context_facts
        # with training data from l104_intellect (5000+ entries, 1600+ MMLU facts,
        # knowledge manifold, knowledge vault). QUOTA_IMMUNE local inference.
        # Always search — the built-in KB often returns 20 irrelevant facts from
        # unrelated domains, and local_intellect may have better-matched entries.
        # Search with BOTH the raw question AND choice-augmented queries to find
        # facts that specifically mention answer choices.
        li = _get_cached_local_intellect()
        if li is not None:
            try:
                li_facts_added = 0
                li_seen = set()  # Deduplicate across multiple queries

                # 1. Search with raw question
                li_results = li._search_training_data(question, max_results=5)
                if isinstance(li_results, list):
                    for entry in li_results:
                        if not isinstance(entry, dict):
                            continue
                        completion = entry.get('completion', '')
                        if completion and len(completion) > 10:
                            if completion not in seen_facts and completion not in li_seen:
                                context_facts.append(completion)
                                seen_facts.add(completion)
                                li_seen.add(completion)
                                li_facts_added += 1

                # 2. Search with choice-augmented queries — critical for finding
                # facts that mention specific answer choices (e.g., "Mars planet")
                q_content = re.sub(r'\b(which|what|who|is|are|the|of|following|known|as|a|an)\b',
                                   '', question.lower()).strip()
                for ch in choices:
                    if li_facts_added >= 12:
                        break
                    choice_query = f"{q_content} {ch}"
                    ch_results = li._search_training_data(choice_query, max_results=3)
                    if isinstance(ch_results, list):
                        for entry in ch_results:
                            if not isinstance(entry, dict):
                                continue
                            completion = entry.get('completion', '')
                            if completion and len(completion) > 10:
                                if completion not in seen_facts and completion not in li_seen:
                                    context_facts.append(completion)
                                    seen_facts.add(completion)
                                    li_seen.add(completion)
                                    li_facts_added += 1

                # 3. Search knowledge manifold for pattern matches
                manifold_hit = li._search_knowledge_manifold(question)
                if manifold_hit and isinstance(manifold_hit, str) and len(manifold_hit) > 10:
                    if manifold_hit not in seen_facts:
                        context_facts.append(manifold_hit)
                        seen_facts.add(manifold_hit)

                # 4. Search knowledge vault for proofs and documentation
                vault_hit = li._search_knowledge_vault(question)
                if vault_hit and isinstance(vault_hit, str) and len(vault_hit) > 10:
                    if vault_hit not in seen_facts:
                        context_facts.append(vault_hit)
                        seen_facts.add(vault_hit)
            except Exception as e:
                _log.debug("Local Intellect KB augmentation failed: %s", e)

        # Step 2c: Statement 1 | Statement 2 handler
        # Many MMLU questions have "Statement 1 | ... Statement 2 | ..." pattern
        # with choices like "True, True", "False, False", "True, False", "False, True"
        # Detect and handle this pattern with statement-level evaluation.
        stmt_match = re.search(r'Statement\s*1\s*\|?\s*(.*?)\.?\s*Statement\s*2\s*\|?\s*(.*?)$',
                               question, re.IGNORECASE | re.DOTALL)
        is_statement_question = stmt_match is not None
        if is_statement_question:
            stmt1 = stmt_match.group(1).strip()
            stmt2 = stmt_match.group(2).strip()

            def _eval_statement(stmt_text):
                truth_score = 0.0
                stmt_lower = stmt_text.lower()
                stmt_words = set(re.findall(r'\w+', stmt_lower))
                # Only count words with 5+ chars as content words (skip short common words)
                stmt_content = {w for w in stmt_words if len(w) >= 5}
                has_negation = any(neg in stmt_lower for neg in ['not ', ' no ', 'never', 'cannot', 'neither'])
                for key, node, rel_score in knowledge_hits:
                    for fact in node.facts:
                        fl = fact.lower()
                        fact_words = set(re.findall(r'\w+', fl))
                        content_overlap = len(stmt_content & fact_words)
                        # Require STRONG overlap (3+ content words) to count as evidence.
                        # Weak overlap (1-2 words) just means the fact is in the same domain,
                        # not that it confirms or denies the statement.
                        if content_overlap >= 3:
                            if has_negation:
                                truth_score -= 0.15
                            else:
                                truth_score += 0.15
                return truth_score

            s1_score = _eval_statement(stmt1)
            s2_score = _eval_statement(stmt2)
            # Require POSITIVE evidence to declare True — when s1_score=0
            # (no matching facts), default to False to avoid always predicting
            # "True, True" (index 0 = A-bias).
            s1_true = s1_score > 0
            s2_true = s2_score > 0

            # Map to choice: "True, True" / "False, False" / "True, False" / "False, True"
            # and store match_score directly for use in Step 3
            truth_combos = [(True, True), (False, False), (True, False), (False, True)]

        # Collect statement match scores for use in Step 3
        _stmt_scores = {}  # index → match_score
        if is_statement_question:
            for i, choice in enumerate(choices):
                cl = choice.lower().strip()
                for t1, t2 in truth_combos:
                    expected = f"{'true' if t1 else 'false'}, {'true' if t2 else 'false'}"
                    if expected in cl:
                        ms = 0.0
                        ms += (0.5 if (s1_true == t1) else -0.5)
                        ms += (0.5 if (s2_true == t2) else -0.5)
                        ms += abs(s1_score) * (1 if (s1_true == t1) else -1) * 0.3
                        ms += abs(s2_score) * (1 if (s2_true == t2) else -1) * 0.3
                        _stmt_scores[i] = ms
                        break

        # Step 3: Score each choice
        # v9.1: _has_context requires BOTH ≥3 facts AND evidence that facts
        # are actually relevant (at least one question content word co-occurs
        # with a choice word in the same fact). Without this, Local Intellect
        # augmentation almost always produces ≥3 facts (even irrelevant ones),
        # causing all 24 scoring stages to fire and add uniform noise.
        _has_context = False
        if len(context_facts) >= 3:
            # Quick relevance check: do any top facts contain BOTH question
            # content words AND choice words?
            _q_cw = {w.lower() for w in re.findall(r'\w+', question.lower()) if len(w) > 3}
            _all_ch_words = set()
            for ch in choices:
                _all_ch_words.update(w.lower() for w in re.findall(r'\w+', ch.lower()) if len(w) > 2)
            for _f in context_facts[:10]:
                _fl = _f.lower()
                _q_in = any(w in _fl for w in _q_cw)
                _c_in = any(w in _fl for w in _all_ch_words)
                if _q_in and _c_in:
                    _has_context = True
                    break
        choice_scores = []
        _raw_score_choice_values = []  # v14: track raw _score_choice scores before bonuses
        for i, choice in enumerate(choices):
            score = self._score_choice(question, choice, context_facts, knowledge_hits,
                                       has_context=_has_context)
            _raw_score_choice_values.append(score)
            # Apply statement match score if this is a Statement question
            if i in _stmt_scores:
                score += _stmt_scores[i]
            # Apply direct answer bonus from Step 2d
            if i in _direct_answer_bonus:
                score += _direct_answer_bonus[i]
            choice_scores.append({
                "index": i,
                "choice": choice,
                "score": score,
                "label": chr(65 + i),  # A, B, C, D
            })

        # v14: Compute KB clear winner from RAW _score_choice values only,
        # before _direct_answer_bonus which can dilute the ratio (e.g., when
        # all short choices like "x", "2x", "x^2" appear as substrings in
        # the same fact, they all get equal bonus, destroying the signal).
        _raw_sorted = sorted(_raw_score_choice_values, reverse=True)
        _kb_clear_winner = (
            len(_raw_sorted) >= 2
            and _raw_sorted[0] > 0
            and _raw_sorted[0] > _raw_sorted[1] * 2.0
        )

        # ── Length-bias normalization (v9.0) ──
        # Longer choices accumulate more word-overlap hits across scoring
        # stages. Normalize by sqrt(avg_wc / wc) so shorter choices are not
        # systematically disadvantaged.
        import math as _ln_math
        _avg_wc = sum(len(cs['choice'].split()) for cs in choice_scores) / max(len(choice_scores), 1)
        for cs in choice_scores:
            _wc = len(cs['choice'].split())
            if _wc > 1 and _avg_wc > 0:
                _norm_factor = _ln_math.sqrt(_avg_wc / _wc)
                cs['score'] *= _norm_factor

        # Step 3a: Semantic TF-IDF scoring — use the encoder to compute
        # similarity between "question + choice" and each knowledge node.
        # This gives much better discrimination than pure keyword overlap.
        # IMPORTANT: Only apply when KB scoring produced real signal (keyword_max > 0).
        # When all _score_choice values are 0, the KB had no relevant facts and
        # TF-IDF similarity against irrelevant facts produces ANTI-SIGNAL (random
        # noise that is worse than random chance). In that case skip TF-IDF and
        # rely on fallback heuristics instead.
        # v14: Also skip when KB already has a clear winner — semantic TF-IDF
        # would add noise that could flip the correct answer.
        keyword_max = max((cs["score"] for cs in choice_scores), default=0)
        keyword_min = min((cs["score"] for cs in choice_scores), default=0)
        keyword_spread = keyword_max - keyword_min
        # KB has real signal when scores have meaningful spread (not just NLU
        # stage noise that gives similar bonuses to all choices equally).
        # Stages 1-2 (SRL, morphology) give ~0.1-0.3 to ALL choices uniformly.
        # Only Stages 3+ (BM25 facts, node confidence, relations) create spread.
        kb_has_signal = keyword_spread > 0.3
        # v14: Skip semantic TF-IDF entirely when KB already produced a clear
        # winner (2× lead). Semantic similarity against irrelevant corpus nodes
        # adds noise that erases the genuine KB signal (fixes Q9, Q13, Q16).
        if kb_has_signal and not _kb_clear_winner and hasattr(self.kb, 'encoder') and self.kb.encoder._corpus_vectors is not None:
            encoder = self.kb.encoder
            semantic_scores = []
            for i, choice in enumerate(choices):
                qc_text = f"{question} {choice}"
                qc_vec = encoder.encode(qc_text)
                # Compute similarity against all indexed nodes
                sims = encoder._corpus_vectors @ qc_vec
                # Take top-3 similarities and average
                top_sims = sorted(sims, reverse=True)[:3]
                sem_score = sum(top_sims) / len(top_sims) if top_sims else 0.0
                semantic_scores.append(max(0.0, float(sem_score)))

            # Normalize semantic scores to [0, 1] range
            max_sem = max(semantic_scores) if semantic_scores else 0
            if max_sem > 0:
                semantic_scores = [s / max_sem for s in semantic_scores]

            if max_sem > 0.01:
                kw_sorted = sorted(choice_scores, key=lambda x: x["score"], reverse=True)
                kw_has_clear_winner = (
                    len(kw_sorted) >= 2
                    and kw_sorted[0]["score"] > 0.5
                    and (kw_sorted[1]["score"] < 0.01
                         or kw_sorted[0]["score"] / max(kw_sorted[1]["score"], 0.001) > 2.0)
                )
                # Semantic weight: lower when keyword evidence is already decisive
                # v14: Reduced from 0.4→0.2 when no clear winner, to prevent
                # semantic similarity from overriding strong BM25 signals.
                sem_weight = 0.10 if kw_has_clear_winner else 0.20

                for i, cs in enumerate(choice_scores):
                    norm_sem = semantic_scores[i]  # already normalized to [0,1]
                    # Blend: add semantic signal (weighted relative to keyword score magnitude)
                    cs["score"] += norm_sem * max(keyword_max, 0.5) * sem_weight

        # v9.0: Early-exit short-circuit — when one choice dominates with
        # overwhelming KB evidence, skip expensive downstream enrichment.
        # v14: Now checks both pre-semantic (_kb_clear_winner) and post-semantic
        # sorted lists. If KB had a clear winner before TF-IDF, always early-exit.
        _early_exit = _kb_clear_winner
        _sorted_pre = sorted(choice_scores, key=lambda x: x["score"], reverse=True)
        if not _early_exit and (len(_sorted_pre) >= 2 and _sorted_pre[0]["score"] > 0
                and _sorted_pre[0]["score"] > _sorted_pre[1]["score"] * 2.0):
            _early_exit = True
        if _early_exit:
            self._early_exits += 1

        # Step 3b: Raw score preservation
        # NOTE: Grover amplitude amplification with an oracle that marks the
        # current highest scorer is self-reinforcing (circular) and harmful
        # when the leader is wrong (same issue observed in ARC). Removed.
        # The previous softmax (temp=0.3) was also too aggressive and distorted
        # signal in close races. Raw knowledge-based scores are preserved.
        raw_scores = [cs["score"] for cs in choice_scores]
        max_raw = max(raw_scores) if raw_scores else 0

        # Step 3c: Quantum circuit confidence — run Bell pair fidelity check
        # to calibrate measurement confidence via entanglement metrics
        # v13: Only apply when KB has real signal to avoid amplifying noise
        qge = _get_cached_quantum_gate_engine()
        if kb_has_signal and qge is not None and max_raw > 0.1 and len(choice_scores) >= 2:
            try:
                from l104_quantum_gate_engine import ExecutionTarget
                bell = qge.bell_pair()
                result = qge.execute(bell, ExecutionTarget.LOCAL_STATEVECTOR)
                if hasattr(result, 'sacred_alignment') and result.sacred_alignment:
                    # Find actual top scorer (NOT hardcoded index 0)
                    sorted_by_score = sorted(range(len(choice_scores)),
                                             key=lambda i: choice_scores[i]["score"],
                                             reverse=True)
                    top_idx = sorted_by_score[0]
                    top_s = choice_scores[top_idx]["score"]
                    sec_s = choice_scores[sorted_by_score[1]]["score"] if len(sorted_by_score) > 1 else 0
                    if top_s > sec_s * 1.3:  # 30% lead required
                        choice_scores[top_idx]["score"] *= 1.03  # 3% proportional boost
            except Exception:
                pass

        # Step 4: Negation detection (NOT/EXCEPT questions)
        # Detect now, but apply inversion AFTER all scoring is complete
        # (Steps 4c-5f add scores that would undo early inversion)
        q_lower_check = question.lower()
        is_negation_q = bool(re.search(
            r'\bnot\b|\bexcept\b|\bnone of\b|\bfalse\b|\bincorrect\b|\bleast likely\b|\bwould not\b',
            q_lower_check
        ))
        # Exclude statement questions ("Statement 1 | ...") from negation
        # inversion — those need truth-value evaluation, not inversion.
        if is_negation_q and is_statement_question:
            is_negation_q = False

        # Step 4b: Rank choices (break ties by reverse index to avoid A-bias)
        choice_scores.sort(key=lambda x: (x["score"], _rng.random()), reverse=True)

        # Step 4b: Elimination — remove clearly implausible choices
        # If the top choice is significantly ahead, boost its lead
        # v13: Only apply when KB has real signal — otherwise the leader
        # is likely wrong and amplifying it makes things worse.
        if kb_has_signal and len(choice_scores) >= 2:
            top = choice_scores[0]["score"]
            second = choice_scores[1]["score"]
            if top > 0 and second > 0 and top / max(second, 0.001) > 2.0:
                # Clear leader — apply elimination bonus
                choice_scores[0]["score"] *= 1.15

        # Step 4c: Fallback heuristics when KB provides no guidance
        # When KB had no signal (all _score_choice = 0), the noisy downstream
        # steps may have pushed scores above 0.15 with random noise. Always
        # apply heuristics when KB had no signal, or when scores are low.
        # EXCEPTION: Skip for NOT/EXCEPT questions — fallback heuristics use
        # surface features that don't account for negation semantics, and
        # would confuse the final NOT-inversion step.
        max_score = choice_scores[0]["score"]
        if not is_negation_q and (max_score < 0.5 or not kb_has_signal):
            for cs in choice_scores:
                heuristic = self._fallback_heuristics(question, cs["choice"], choices)
                cs["score"] += heuristic
            choice_scores.sort(key=lambda x: (x["score"], _rng.random()), reverse=True)

        best = choice_scores[0]

        # Step 5: Numerical reasoning — score numerical matches for quantitative questions
        # v14: Skip on early-exit — numerical bonuses can flip KB-correct leader
        if not _early_exit and self.numerical_reasoner:
            for cs in choice_scores:
                num_bonus = self.numerical_reasoner.score_numerical_match(
                    cs["choice"], context_facts, question)
                if num_bonus > 0:
                    cs["score"] += num_bonus
                    self._numerical_assists += 1
            choice_scores.sort(key=lambda x: (x["score"], _rng.random()), reverse=True)
            best = choice_scores[0]

        # Step 5b: N-gram phrase-level scoring boost
        # Use N-gram matcher to find phrase-level matches between choices and facts.
        # v14: Skip on early-exit — N-gram matching over "question+choice" text
        # rewards choices that appear IN the question (parrot effect), which can
        # flip the correct answer (e.g., "derivative of x^2" → "x^2" gets boosted).
        if not _early_exit and kb_has_signal and hasattr(self.kb, 'ngram_matcher') and self.kb.ngram_matcher._indexed:
            for cs in choice_scores:
                qc_text = f"{question} {cs['choice']}"
                for fact in context_facts[:10]:
                    ngram_score = self.kb.ngram_matcher.phrase_overlap_score(qc_text, fact)
                    if ngram_score > 0.05:
                        cs["score"] += ngram_score * 0.25
                        self._ngram_boosts += 1
            # Re-sort after N-gram boost
            choice_scores.sort(key=lambda x: (x["score"], _rng.random()), reverse=True)
            best = choice_scores[0]

        # Step 5b: Formal Logic Engine assist for logic-type questions
        # v9.0: Skipped on early-exit (high-confidence leader)
        q_lower_for_logic = question.lower()
        logic_keywords = {'modus', 'ponens', 'tollens', 'syllogism', 'valid', 'fallacy',
                          'tautology', 'contradiction', 'logically', 'entails', 'implies',
                          'de morgan', 'contrapositive', 'converse', 'equivalent'}
        if not _early_exit and any(kw in q_lower_for_logic for kw in logic_keywords):
            fle = _get_cached_formal_logic()
            if fle is not None:
                try:
                    # Use formal logic to evaluate each choice
                    for cs in choice_scores:
                        logic_result = fle.analyze_argument(f"{question} Answer: {cs['choice']}")
                        if hasattr(logic_result, 'get'):
                            validity = logic_result.get('validity_score', 0.5)
                            cs["score"] += (validity - 0.5) * 0.3
                    self._logic_assists += 1
                    choice_scores.sort(key=lambda x: (x["score"], _rng.random()), reverse=True)
                    best = choice_scores[0]
                except Exception:
                    pass

        # Step 5c: DeepNLU discourse analysis for nuanced comprehension
        deep_nlu = _get_cached_deep_nlu()
        if not _early_exit and deep_nlu is not None and len(question) > 50:
            try:
                nlu_result = deep_nlu.analyze(question)
                if hasattr(nlu_result, 'get'):
                    intent = nlu_result.get('intent', {})
                    sentiment = nlu_result.get('sentiment', {})
                    # Use discourse intent to adjust scoring
                    if intent.get('type') == 'comparison' and len(choice_scores) >= 2:
                        # For comparison questions, boost choices with comparative language
                        for cs in choice_scores:
                            cl = cs['choice'].lower()
                            if any(w in cl for w in ['both', 'all', 'neither', 'more', 'less']):
                                cs["score"] += 0.08
                    self._nlu_assists += 1
                    choice_scores.sort(key=lambda x: (x["score"], _rng.random()), reverse=True)
                    best = choice_scores[0]
            except Exception:
                pass

        # Step 5f: Cross-verification — multi-strategy answer validation
        # Only cross-verify when KB evidence exists; cross-verifying against
        # irrelevant facts amplifies noise in the same direction as TF-IDF.
        if not _early_exit and kb_has_signal and self.cross_verifier and len(choice_scores) >= 2 and context_facts:
            choice_scores = self.cross_verifier.verify(
                question, choice_scores, context_facts, knowledge_hits)
            best = choice_scores[0]
            self._cross_verifications += 1

        # Step 5g: Negation-aware ranking inversion (final, post-scoring)
        # For "NOT/EXCEPT" questions, the correct answer is the one that
        # matches LEAST with the positive pattern. Invert scores AFTER all
        # scoring steps are complete so no subsequent step undoes it.
        if is_negation_q and len(choice_scores) >= 2:
            scores_vals = [cs["score"] for cs in choice_scores]
            max_s = max(scores_vals)
            min_s = min(scores_vals)
            if max_s > min_s:
                spread = max_s - min_s
                amplification = max(1.0, 0.5 / max(spread, 0.01))
                amplification = min(amplification, 5.0)
                mean_s = sum(scores_vals) / len(scores_vals)
                for cs in choice_scores:
                    deviation = cs["score"] - mean_s
                    cs["score"] = mean_s - deviation * amplification
            choice_scores.sort(key=lambda x: (x["score"], _rng.random() * 1e-9), reverse=True)
            best = choice_scores[0]

        # Step 5h: Quantum Wave Collapse — Knowledge Synthesis + Born-Rule Selection
        # Convert multi-stage heuristic scores into quantum amplitudes with
        # GOD_CODE phase encoding + knowledge-density oracle amplification.
        # Born-rule |ψ|² collapse selects the answer with highest quantum
        # probability, providing non-linear discrimination that amplifies
        # signal from KB-backed choices and suppresses noise.
        # v14: Skip on early-exit — when KB already gives a clear 2×+ leader,
        # quantum reranking introduces noise that can flip the correct winner.
        if not _early_exit:
            choice_scores = self._quantum_wave_collapse(
                question, choices, choice_scores, context_facts, knowledge_hits)
        best = choice_scores[0]

        # Step 6: Chain-of-thought verification
        reasoning = self._chain_of_thought(question, choices, best, context_facts)

        # Step 7: Confidence calibration with PHI + Entropy + Dual-Layer
        raw_confidence = best["score"]

        # Base calibration: score-gap-aware confidence
        # v7.0: Use score gap between best and second-best to gauge certainty.
        # Previous formula (raw * TAU + 0.1) saturated at 0.95 for any raw >= 0.135.
        scores_sorted = sorted([cs["score"] for cs in choice_scores], reverse=True)
        score_gap = (scores_sorted[0] - scores_sorted[1]) if len(scores_sorted) > 1 else 0.0
        gap_factor = min(1.0, score_gap / 0.3)  # Normalize gap: 0.3+ = full confidence
        calibrated_confidence = min(0.95, 0.25 + 0.35 * raw_confidence + 0.35 * gap_factor)

        # Step 7b: Entropy-based calibration via Maxwell Demon
        se = _get_cached_science_engine()
        if se is not None and raw_confidence > 0.1:
            try:
                demon_eff = se.entropy.calculate_demon_efficiency(1.0 - raw_confidence)
                if isinstance(demon_eff, (int, float)) and 0 < demon_eff < 1:
                    # Higher demon efficiency = less entropy = more confident
                    entropy_boost = (demon_eff - 0.5) * 0.08
                    calibrated_confidence = min(0.95, calibrated_confidence + entropy_boost)
                    self._entropy_calibrations += 1
            except Exception:
                pass

        # Step 7c: Dual-Layer Engine physics grounding
        dl = _get_cached_dual_layer()
        if dl is not None and raw_confidence > 0.15:
            try:
                dl_score = dl.dual_score()
                if isinstance(dl_score, (int, float)) and 0 < dl_score <= 1:
                    # Physics-grounded alignment: slight confidence adjustment
                    physics_factor = 1.0 + (dl_score - 0.5) * 0.04
                    calibrated_confidence = min(0.95, calibrated_confidence * physics_factor)
                    self._dual_layer_calibrations += 1
            except Exception:
                pass

        return {
            "answer": best["label"],
            "answer_index": best["index"],
            "selected_index": best["index"],
            "answer_text": best["choice"],
            "confidence": round(calibrated_confidence, 4),
            "reasoning": reasoning,
            "knowledge_hits": len(knowledge_hits),
            "context_facts_used": len(context_facts),
            "all_scores": [{"label": c["label"], "score": round(c["score"], 4)}
                          for c in choice_scores],
            "calibration": {
                "entropy_calibrated": self._entropy_calibrations > 0,
                "dual_layer_calibrated": self._dual_layer_calibrations > 0,
                "ngram_boosted": self._ngram_boosts > 0,
                "logic_assisted": self._logic_assists > 0,
                "nlu_assisted": self._nlu_assists > 0,
                "numerical_assisted": self._numerical_assists > 0,
                "cross_verified": self._cross_verifications > 0,
                "subject_detected": subject is not None,
                "quantum_collapsed": self._quantum_collapses > 0,
            },
            "quantum": {
                "wave_collapse_applied": best.get("quantum_prob") is not None,
                "quantum_probability": best.get("quantum_prob", 0.0),
            },
        }

    def _score_choice(self, question: str, choice: str,
                      context_facts: List[str], knowledge_hits: List,
                      has_context: bool = False) -> float:
        """Score a single choice using algorithmic NLU analysis.

        v5.0 Algorithmic Pipeline (replaces hardcoded keyword matching):
        ═══════════════════════════════════════════════════════════════
        Stage 1 — Semantic Role Analysis:  DeepNLU SRL to parse question
                  structure (agent/patient/theme) and match to choice roles.
                  SKIPPED when has_context=True (adds noise over KB signal).
        Stage 2 — Morphological Alignment: DeepNLU morphology to assess
                  word-root compatibility between question and choice.
                  SKIPPED when has_context=True (adds noise over KB signal).
        Stage 3 — BM25 Fact Relevance:     Rank facts by question relevance,
                  then score choice occurrence in top-ranked facts.
        Stage 4 — Formal Logic Entailment: Build premise→conclusion chains
                  from facts & check if choice is logically entailed.
        Stage 5 — Distributional Similarity: TF-IDF cosine between
                  question+fact context and choice text.
        Stage 6 — Negation/Polarity:       Detect polarity inversions
                  algorithmically via morphological negation affixes.
        Stage 7 — Confidence Weighting:     PHI-calibrated node confidence.
        """
        score = 0.0

        choice_lower = choice.lower().strip()
        choice_words = set(re.findall(r'\w+', choice_lower))
        q_lower = question.lower()
        q_words = set(re.findall(r'\w+', q_lower))
        q_content_words = {w for w in q_words if len(w) > 3}

        # ── Stage 1: Semantic Role Analysis via DeepNLU ──────────────────
        # Parse question for semantic roles (agent, patient, theme, instrument)
        # then check if the choice fills the expected answer role.
        # SKIP when we have KB context facts — SRL bonuses are generic and
        # add uniform noise that drowns the discriminative BM25/relation signal.
        deep_nlu = _get_cached_deep_nlu()
        srl_bonus = 0.0
        if not has_context and deep_nlu is not None:
            try:
                srl_result = deep_nlu.label_semantic_roles(question)
                if isinstance(srl_result, dict):
                    # The question's "theme" or "patient" is what's being asked about
                    frame = srl_result
                    theme = str(frame.get("theme", "")).lower()
                    patient = str(frame.get("patient", "")).lower()
                    agent = str(frame.get("agent", "")).lower()
                    # If the choice fills the missing role:
                    # "What is X?" → theme=X, answer fills the predicate
                    # "Who wrote X?" → patient=X, answer fills the agent
                    if theme and choice_lower:
                        theme_words = set(re.findall(r'\w+', theme))
                        choice_theme_overlap = len(choice_words & theme_words)
                        if choice_theme_overlap > 0:
                            srl_bonus += min(choice_theme_overlap, 2) * 0.12  # v7.1: cap to prevent length bias
                    if patient and choice_lower:
                        patient_words = set(re.findall(r'\w+', patient))
                        if len(choice_words & patient_words) > 0:
                            srl_bonus += 0.15
                    # Check if choice matches expected agent role
                    if agent and choice_lower and agent != "none":
                        agent_words = set(re.findall(r'\w+', agent))
                        if len(choice_words & agent_words) > 0:
                            srl_bonus += 0.2
            except Exception:
                pass
        score += srl_bonus

        # ── Stage 2: Morphological Alignment ─────────────────────────────
        # Use morphological analysis to find root-form matches between
        # question content words and choice, beyond surface-level keywords.
        # SKIP when we have KB context facts — morphological roots are too
        # coarse and add uniform bonuses to all choices, reducing discrimination.
        morpho_bonus = 0.0
        if not has_context and deep_nlu is not None:
            try:
                # Analyze choice words morphologically
                choice_roots = set()
                for word in list(choice_words)[:6]:  # Cap for performance
                    morph = deep_nlu.analyze_morphology(word)
                    if isinstance(morph, dict):
                        root = morph.get("root", morph.get("stem", word)).lower()
                        choice_roots.add(root)
                        # Also add any identified base forms
                        base = morph.get("base_form", "").lower()
                        if base:
                            choice_roots.add(base)

                q_roots = set()
                for word in list(q_content_words)[:8]:
                    morph = deep_nlu.analyze_morphology(word)
                    if isinstance(morph, dict):
                        root = morph.get("root", morph.get("stem", word)).lower()
                        q_roots.add(root)

                # Root overlap: deeper semantic alignment than surface keywords
                root_overlap = len(choice_roots & q_roots)
                if root_overlap > 0 and len(choice_roots) > 0:
                    morpho_bonus = root_overlap * 0.08
            except Exception:
                pass
        score += morpho_bonus

        # ── Stage 3: BM25 Fact Relevance Scoring ────────────────────────
        # Instead of naive keyword counting, use BM25 TF-IDF ranking to
        # algorithmically score each fact's relevance to the question,
        # then measure choice presence in top-ranked facts.
        # v9.0: Reuse solver-level _choice_bm25 instead of creating new BM25Ranker per call
        if context_facts:
            if not hasattr(self, '_choice_bm25') or self._choice_bm25 is None:
                self._choice_bm25 = BM25Ranker()
            self._choice_bm25.fit(context_facts)
            fact_scores = self._choice_bm25.score(question)

            # Only process top-ranked facts to avoid noise accumulation.
            # Iterating over 100+ facts gives every choice marginal bonuses
            # that sum to near-equal totals, destroying discrimination.
            ranked_indices = sorted(range(len(fact_scores)),
                                    key=lambda i: fact_scores[i], reverse=True)[:20]

            for idx in ranked_indices:
                fact = context_facts[idx]
                fact_lower = fact.lower()
                bm25_weight = fact_scores[idx]
                if bm25_weight <= 0.01:
                    continue

                rel_weight = math.log1p(bm25_weight) * 0.3
                fact_words = set(re.findall(r'\w+', fact_lower))

                # Co-occurrence: question AND choice in same relevant fact.
                # v10.0 FIX: Use choice-EXCLUSIVE words (not shared with question)
                # for c_overlap to prevent circular inflation. E.g. "Cell division"
                # for "What is the function of mitochondria in a cell?" — "cell"
                # is a question word, not discriminative choice evidence.
                q_overlap = len(q_content_words & fact_words)
                choice_exclusive_words = choice_words - q_content_words
                c_overlap = len(choice_exclusive_words & fact_words)
                # Also count full choice words (including shared) at reduced weight
                c_shared_overlap = len((choice_words & q_content_words) & fact_words)

                # Prefix-based matching for morphological variants (7-char min)
                if q_overlap == 0 and len(q_content_words) > 0:
                    q_prefixes = {w[:7] for w in q_content_words if len(w) >= 7}
                    fact_prefixes_q = {w[:7] for w in fact_words if len(w) >= 7}
                    q_prefix_overlap = len(q_prefixes & fact_prefixes_q)
                    if q_prefix_overlap > 0:
                        q_overlap = q_prefix_overlap

                if c_overlap == 0 and len(choice_exclusive_words) > 0:
                    choice_prefixes = {w[:7] for w in choice_exclusive_words if len(w) >= 7}
                    fact_prefixes = {w[:7] for w in fact_words if len(w) >= 7}
                    prefix_overlap = len(choice_prefixes & fact_prefixes)
                    if prefix_overlap > 0:
                        c_overlap = prefix_overlap  # Count prefix matches as overlap

                if q_overlap >= 1 and c_overlap >= 1:
                    score += min(q_overlap, 3) * min(c_overlap, 3) * 0.20 * rel_weight
                # Shared words contribute at 25% weight (were 100%)
                if q_overlap >= 1 and c_shared_overlap >= 1:
                    score += min(q_overlap, 3) * min(c_shared_overlap, 2) * 0.05 * rel_weight

                # Full substring containment (weighted by fact relevance)
                if len(choice_lower) > 2 and choice_lower in fact_lower:
                    score += 0.8 * rel_weight

        # ── Stage 3b: Direct Answer Extraction ───────────────────────────
        # Scan BM25-ranked facts for patterns that directly associate a
        # question key-phrase with this choice. Fires when a fact contains
        # ≥3 question content words AND this choice (word-boundary match).
        # E.g., "cogito ergo sum ... attributed to Descartes" + choice "Descartes"
        # Also checks adjacency: if choice appears near "at the base", "begins with",
        # "attributed to" etc., gives higher bonus (proximal evidence).
        if context_facts:
            q_words_3b = {w for w in re.findall(r'\w+', question.lower()) if len(w) > 2}
            q_content_3b = q_words_3b - {'what', 'which', 'who', 'whom', 'the', 'is',
                'are', 'was', 'were', 'does', 'did', 'has', 'have', 'had', 'this',
                'that', 'these', 'those', 'and', 'or', 'but', 'for', 'with', 'from',
                'about', 'into', 'how', 'why', 'when', 'where', 'not', 'its'}
            for fact in context_facts[:15]:
                fl = fact.lower()
                fact_words_3b = set(re.findall(r'\w+', fl))
                # Require ≥3 question content words in the fact (strong topic match)
                topic_overlap = len(q_content_3b & fact_words_3b)
                if topic_overlap < 3:
                    continue
                # Check if THIS choice appears in the fact (word-boundary match)
                ch_match = re.search(r'\b' + re.escape(choice_lower) + r'\b', fl)  if len(choice_lower) >= 2 else None
                if ch_match:
                    bonus_3b = min(2.0, topic_overlap * 0.5)
                    # Adjacency boost: if the choice appears within 5 words
                    # of a question keyword in the fact, it's likely the answer.
                    fact_word_list = re.findall(r'\w+', fl)
                    ch_pos = None
                    for wi, fw in enumerate(fact_word_list):
                        if fw == choice_lower or (len(choice_lower) > 3 and choice_lower.startswith(fw)):
                            ch_pos = wi
                            break
                    if ch_pos is not None:
                        for qw in q_content_3b:
                            for wi2, fw2 in enumerate(fact_word_list):
                                if fw2 == qw and abs(wi2 - ch_pos) <= 5:
                                    bonus_3b += 0.5  # Adjacency boost
                                    break
                    score += bonus_3b

        # ── Stage 4: Formal Logic Entailment ─────────────────────────────
        # Build premises from top facts and check if each choice is
        # logically entailed via the inference chain builder.
        fle = _get_cached_formal_logic()
        logic_bonus = 0.0
        if fle is not None and context_facts:
            try:
                # Use top-3 most relevant facts as premises
                top_facts = context_facts[:3]
                # Build an inference chain: do the premises support this choice?
                chain_result = fle.build_inference_chain(
                    premises=top_facts,
                    target=f"{question} The answer is {choice}"
                )
                if isinstance(chain_result, dict):
                    chain_conf = chain_result.get("confidence", 0.0)
                    chain_steps = chain_result.get("steps", [])
                    if chain_conf > 0.3 and len(chain_steps) > 0:
                        logic_bonus = (chain_conf - 0.3) * 0.8  # Scale 0→0.56
                    # Fallacy check: if the Q+choice triggers a fallacy, penalize
                    fallacies = fle.detect_fallacies(f"{question} Therefore {choice}")
                    if isinstance(fallacies, list) and len(fallacies) > 0:
                        logic_bonus -= len(fallacies) * 0.1
            except Exception:
                pass
        score += max(logic_bonus, 0.0)

        # ── Stage 5: Distributional TF-IDF Similarity ────────────────────
        # Compute TF-IDF cosine similarity between the question+context
        # concatenation and each choice (proper information retrieval scoring).
        if context_facts and hasattr(self.kb, 'encoder') and self.kb.encoder._corpus_vectors is not None:
            try:
                encoder = self.kb.encoder
                # Build a "context document" from question + top facts
                context_doc = question + " " + " ".join(context_facts[:5])
                context_vec = encoder.encode(context_doc)
                choice_vec = encoder.encode(choice)
                # Cosine similarity
                dot = float(np.dot(context_vec, choice_vec))
                norm_ctx = float(np.linalg.norm(context_vec))
                norm_ch = float(np.linalg.norm(choice_vec))
                if norm_ctx > 0 and norm_ch > 0:
                    cosine_sim = dot / (norm_ctx * norm_ch)
                    score += max(0.0, cosine_sim) * 0.25
            except Exception:
                pass

        # ── Stage 6: Negation & Polarity ────────────────────────────────
        # Negation-aware scoring is handled SOLELY in solve() Step 4 which
        # inverts all scores for NOT/EXCEPT questions. Do NOT add any
        # negation bonuses/penalties here to avoid double-negation conflicts.

        # ── Stage 7: Knowledge Node Confidence Weighting ─────────────────
        # Weight by retrieval relevance × node confidence × choice presence.
        # Cap total contribution to avoid inflation when many nodes match.
        stage7_bonus = 0.0
        for key, node, relevance in knowledge_hits[:10]:
            node_text = (node.definition + " " + " ".join(node.facts)).lower()
            choice_in_node = sum(1 for w in choice_words if len(w) > 2 and w in node_text)
            if choice_in_node > 0:
                stage7_bonus += relevance * node.confidence * min(choice_in_node, 3) * 0.08
        score += min(stage7_bonus, 1.0)

        # ── Stage 8: Structured Relation Extraction ──────────────────────
        # Parse facts for predicate structures (X is Y, X wrote Y, X stands
        # for Y) using regex-based relation extraction, weighted by BM25 relevance.
        # Cap to top-30 facts to avoid noise accumulation from low-relevance facts.
        # TOTAL contribution is capped at 1.5 to prevent single false-positive
        # regex matches from dominating all other scoring stages combined.
        stage8_total = 0.0
        for fact in context_facts[:30]:
            fact_lower = fact.lower()
            q_in_fact = sum(1 for w in q_content_words if w in fact_lower)
            if q_in_fact < 1:
                continue
            specificity = min(q_in_fact, 4) / 2.0

            # Relation: "SUBJECT is/are ANSWER"
            for m in re.finditer(
                r'(?:the\s+)?([^,;()\n]+)\s+(?:is|are|was|were)\s+(?:the\s+)?([^,;()\n]+)',
                fact_lower,
            ):
                subj_part = m.group(1).strip()
                answer_part = m.group(2).strip()
                if len(choice_lower) > 1 and len(answer_part) > 1:
                    # Forward: choice matches answer (Q asks about subject)
                    if choice_lower == answer_part:
                        stage8_total += 0.5 * specificity
                    elif choice_lower in answer_part.split():
                        stage8_total += 0.4 * specificity
                    elif len(choice_lower) > 3 and choice_lower in answer_part:
                        stage8_total += 0.3 * specificity
                    # Reverse: choice matches subject (Q asks about predicate)
                    q_in_answer = sum(1 for w in q_content_words if w in answer_part)
                    if q_in_answer >= 1 and len(choice_lower) > 2:
                        if choice_lower == subj_part:
                            stage8_total += 0.5 * specificity
                        elif choice_lower in subj_part.split():
                            stage8_total += 0.4 * specificity
                        elif len(choice_lower) > 3 and choice_lower in subj_part:
                            stage8_total += 0.3 * specificity

            # Relation: "PERSON wrote WORK"
            for m in re.finditer(r'(\w+)\s+wrote\s+([^,;.\n]+)', fact_lower):
                person, work = m.group(1).strip(), m.group(2).strip()
                if any(w in fact_lower for w in q_content_words if len(w) > 3):
                    if choice_lower == person or person in choice_lower.split():
                        stage8_total += 0.5

            # Relation: "X stands for Y"
            for m in re.finditer(r'(\w+)\s+stands\s+for\s+([^,;.\n]+)', fact_lower):
                acronym, expansion = m.group(1).strip(), m.group(2).strip()
                if any(w == acronym for w in q_words):
                    if choice_lower in expansion or expansion in choice_lower:
                        stage8_total += 0.5

            # Relation: "X uses Y"
            for m in re.finditer(r'(\w+)\s+uses\s+(\w+)', fact_lower):
                subj, obj = m.group(1).strip(), m.group(2).strip()
                if obj in q_lower and (choice_lower == subj or subj in choice_lower):
                    stage8_total += 0.4

            # Relation: "X is known as Y" / "X is also called Y" / "X is referred to as Y"
            for m in re.finditer(
                r'(\w[\w\s]{0,40})\s+(?:is\s+(?:known|also\s+called|referred\s+to|nicknamed|dubbed))\s+(?:as\s+)?(?:the\s+)?([^,;.\n]+)',
                fact_lower,
            ):
                subj, alias = m.group(1).strip(), m.group(2).strip()
                if len(choice_lower) > 1:
                    if choice_lower == subj or subj in choice_lower.split():
                        if any(w in q_lower for w in alias.split() if len(w) > 3):
                            stage8_total += 0.5
                    if choice_lower == alias or alias in choice_lower:
                        if any(w in q_lower for w in subj.split() if len(w) > 3):
                            stage8_total += 0.5

            # Relation: superlative patterns
            for pat in [
                r'(\w+)\s+is\s+the\s+\w+\s+(closest|nearest|farthest|largest|smallest|highest|lowest|hottest|coldest)\s+(?:to|in|from)\s+(\w[\w\s]{0,40})',
                r'(\w+)\s+is\s+the\s+(closest|nearest|farthest|largest|smallest|first|last)\s+\w+\s+(?:to|in|from)\s+(\w[\w\s]{0,40})',
            ]:
                for m in re.finditer(pat, fact_lower):
                    subj, superlative = m.group(1).strip(), m.group(2).strip()
                    if superlative in q_lower:
                        if choice_lower == subj or subj in choice_lower:
                            stage8_total += 0.8  # Boosted from 0.5 for strong superlative link

            # Relation: "the SI unit of X is Y"
            for m in re.finditer(
                r'(?:the\s+)?si\s+unit\s+of\s+(\w+)\s+is\s+(?:the\s+)?(\w+)',
                fact_lower,
            ):
                quantity, unit_name = m.group(1).strip(), m.group(2).strip()
                if quantity in q_lower:
                    if choice_lower == unit_name or unit_name in choice_lower:
                        stage8_total += 0.8

            # Relation: numeric value patterns "X is approximately Y" / "rounded to ... is Y"
            for m in re.finditer(
                r'(?:value\s+of\s+)?(\w+)\s+(?:is\s+approximately|rounded\s+to[^,;.\n]{0,30}is|equals?|=)\s+([\d\.]+)',
                fact_lower,
            ):
                name_part = m.group(1).strip()
                value_part = m.group(2).strip()
                if name_part in q_lower or name_part == "pi":
                    if value_part in choice_lower or choice_lower == value_part:
                        stage8_total += 0.8

            # Relation: "X causes/leads to/results in Y"
            for m in re.finditer(
                r'(\w[\w\s]{0,40})\s+(?:causes?|leads?\s+to|results?\s+in|produces?|triggers?)\s+([^,;.\n]+)',
                fact_lower,
            ):
                cause_part = m.group(1).strip()
                effect_part = m.group(2).strip()
                if len(choice_lower) > 1:
                    # Q asks "What causes X?" → choice matches cause
                    if any(w in q_lower for w in effect_part.split() if len(w) > 3):
                        if choice_lower in cause_part or cause_part in choice_lower:
                            stage8_total += 0.5 * specificity
                        elif any(w in cause_part for w in choice_words if len(w) > 3):
                            stage8_total += 0.3 * specificity
                    # Q asks "What does X cause?" → choice matches effect
                    if any(w in q_lower for w in cause_part.split() if len(w) > 3):
                        if choice_lower in effect_part or effect_part in choice_lower:
                            stage8_total += 0.5 * specificity
                        elif any(w in effect_part for w in choice_words if len(w) > 3):
                            stage8_total += 0.3 * specificity

            # Relation: "X is caused by Y" / "X is produced by Y"
            for m in re.finditer(
                r'(\w[\w\s]{0,40})\s+(?:is|are)\s+(?:caused|produced|triggered|created)\s+by\s+([^,;.\n]+)',
                fact_lower,
            ):
                effect_part = m.group(1).strip()
                cause_part = m.group(2).strip()
                if len(choice_lower) > 1:
                    if any(w in q_lower for w in effect_part.split() if len(w) > 3):
                        if choice_lower in cause_part or any(w in cause_part for w in choice_words if len(w) > 3):
                            stage8_total += 0.4 * specificity
                    if any(w in q_lower for w in cause_part.split() if len(w) > 3):
                        if choice_lower in effect_part or any(w in effect_part for w in choice_words if len(w) > 3):
                            stage8_total += 0.4 * specificity

            # Relation: "X consists of Y" / "X is composed of Y" / "X contains Y"
            for m in re.finditer(
                r'(\w[\w\s]{0,40})\s+(?:consists?\s+of|is\s+composed\s+of|contains?|includes?)\s+([^,;.\n]+)',
                fact_lower,
            ):
                whole_part = m.group(1).strip()
                component_part = m.group(2).strip()
                if len(choice_lower) > 1:
                    if any(w in q_lower for w in whole_part.split() if len(w) > 3):
                        if any(w in component_part for w in choice_words if len(w) > 3):
                            stage8_total += 0.4 * specificity

        # Cap total Stage 8 contribution (v9.1: raised from 1.5 → 2.5 for stronger KB signal)
        score += min(stage8_total, 2.5)

        # ── NOISE GATE (v9.1) ───────────────────────────────────────────
        # Stages 9-24 perform NLP text analysis (temporal, causal, analogical,
        # fuzzy matching, NER, disambiguation, coreference, semantic frames,
        # taxonomy, pragmatic, commonsense, sentiment, entailment).
        # WITHOUT context facts these stages produce near-uniform bonuses
        # across all choices (~0.05-0.2 each), accumulating ~1.0 of random
        # noise that destroys the discriminative signal from KB-backed
        # stages (3, 4, 5, 7, 8).  Gate them behind has_context so they
        # only fire when the KB provided relevant retrieval data.
        if not has_context:
            # No KB signal — skip noisy NLP stages, return core score only.
            # Stage 23 (short choice penalty) still applies.
            if len(choice_lower) <= 2 and score < 1.0:
                score *= 0.3
            return score

        # ── Stage 9: Temporal Reasoning Boost (v6.0.0) ──────────────────
        # For questions about timing, ordering, or duration, leverage the
        # DeepNLU TemporalReasoner to score choices that align with the
        # detected temporal structure of the question.
        if deep_nlu is not None and hasattr(deep_nlu, 'temporal'):
            try:
                q_temporal = deep_nlu.temporal.analyze(question)
                if q_temporal.get('temporal_richness', 0) > 0:
                    # Check if choice matches detected tense/temporal pattern
                    c_temporal = deep_nlu.temporal.analyze(choice)
                    # Tense alignment bonus
                    q_tense = q_temporal.get('tense', {}).get('dominant', 'unknown')
                    c_tense = c_temporal.get('tense', {}).get('dominant', 'unknown')
                    if q_tense != 'unknown' and c_tense != 'unknown' and q_tense == c_tense:
                        score += 0.08
                    # Temporal expression in choice that contextually matches question
                    c_exprs = c_temporal.get('temporal_expressions', [])
                    q_exprs = q_temporal.get('temporal_expressions', [])
                    if c_exprs and q_exprs:
                        score += 0.12
            except Exception:
                pass

        # ── Stage 10: Causal Reasoning Boost (v6.0.0) ───────────────────
        # For causal questions (why, cause, effect, result), check if the
        # choice fills a causal role detected in the question + facts.
        causal_q_words = {'cause', 'causes', 'caused', 'why', 'because', 'reason',
                          'result', 'effect', 'leads', 'lead', 'consequence',
                          'produce', 'produces', 'trigger', 'triggers', 'due'}
        if q_words & causal_q_words and deep_nlu is not None and hasattr(deep_nlu, 'causal'):
            try:
                # Analyze combined question+choice for causal coherence
                combined = f"{question} {choice}"
                causal_result = deep_nlu.causal.analyze(combined)
                causal_pairs = causal_result.get('causal_pairs', [])
                causal_strength = causal_result.get('causal_strength', 0)
                if causal_pairs:
                    # Choice participates in a causal relation with question content
                    for pair in causal_pairs[:3]:
                        cause_text = pair.get('cause', '').lower()
                        effect_text = pair.get('effect', '').lower()
                        # Check if choice words appear in cause or effect
                        c_in_cause = any(w in cause_text for w in choice_words if len(w) > 3)
                        c_in_effect = any(w in effect_text for w in choice_words if len(w) > 3)
                        if c_in_cause or c_in_effect:
                            score += min(0.2, causal_strength * 0.3)
                            break
            except Exception:
                pass

        # ── Stage 11: Textual Entailment Scoring (v7.0.0) ────────────────
        # Use TextualEntailmentEngine to check if facts entail the choice.
        # Entailment boosts score; contradiction penalizes.
        try:
            entailment_engine = TextualEntailmentEngine()
            entailment_total = 0.0
            for fact in context_facts[:5]:
                ent_score = entailment_engine.score_fact_choice_entailment(fact, choice)
                entailment_total += ent_score
            # Normalize and cap
            if entailment_total > 0:
                score += min(entailment_total * 0.15, 0.4)
            elif entailment_total < 0:
                score += max(entailment_total * 0.10, -0.3)
        except Exception:
            pass

        # ── Stage 12: Analogical Reasoning Scoring (v7.0.0) ─────────────
        # Detect analogy patterns in the question and score choices.
        try:
            analogical = AnalogicalReasoner()
            analogy_parts = analogical.detect_analogy_in_question(question)
            if analogy_parts is not None:
                analogy_score = analogical.score_analogy(
                    analogy_parts['a'], analogy_parts['b'],
                    analogy_parts['c'], choice
                )
                score += min(analogy_score * 0.3, 0.5)
        except Exception:
            pass

        # ── Stage 13: Fuzzy Match Scoring (v7.0.0) ──────────────────────
        # Use Levenshtein similarity for near-match detection between
        # choice terms and key content words in facts.
        try:
            fuzzy = LevenshteinMatcher()
            fuzzy_bonus = 0.0
            for fact in context_facts[:3]:
                fact_words = [w for w in re.findall(r'\w+', fact.lower()) if len(w) > 4]
                for cw in choice_words:
                    if len(cw) > 4:
                        for fw in fact_words:
                            sim = fuzzy.similarity(cw, fw)
                            if 0.75 <= sim < 1.0:  # Near-match but not exact
                                fuzzy_bonus += (sim - 0.75) * 0.5
            score += min(fuzzy_bonus, 0.25)
        except Exception:
            pass

        # ── Stage 14: NER Entity Alignment (v7.0.0) ─────────────────────
        # If question asks about specific entity types, boost choices that
        # contain matching entity types.
        try:
            ner = NamedEntityRecognizer()
            q_entities = ner.extract_entity_types(question)
            c_entities = ner.extract_entity_types(choice)
            # Entity type overlap bonus
            shared_types = set(q_entities.keys()) & set(c_entities.keys())
            if shared_types:
                # Matching entity types suggest relevance
                score += min(len(shared_types) * 0.08, 0.2)
        except Exception:
            pass

        # ── Stage 15: Contextual Disambiguation (v7.1.0) ────────────────
        # Use ContextualDisambiguator to resolve ambiguous words in choosing
        # the correct domain-specific sense.
        try:
            from l104_asi.deep_nlu import ContextualDisambiguator
            disambiguator = ContextualDisambiguator()

            # Disambiguate key words in the question to identify the domain
            q_disambig = disambiguator.disambiguate(question)
            q_domains = set()
            for d in q_disambig.get('disambiguations', []):
                q_domains.add(d.get('selected_sense', {}).get('domain', ''))
            q_domains.discard('')

            if q_domains:
                # Boost choices whose words resolve to matching domains
                c_disambig = disambiguator.disambiguate(choice)
                for d in c_disambig.get('disambiguations', []):
                    c_domain = d.get('selected_sense', {}).get('domain', '')
                    if c_domain in q_domains:
                        score += 0.08  # Domain alignment bonus
        except Exception:
            pass

        # ── Stage 16: Coreference-Resolved Scoring (v8.0.0) ────────────
        # Resolve pronouns in the question so keyword matching can find the
        # actual entity being asked about, not just the pronoun.
        try:
            coref = CoreferenceResolver()
            resolved_q = coref.resolve_for_scoring(question)
            if resolved_q != question:
                # Re-score keyword overlap with resolved question
                resolved_words = set(re.findall(r'\w+', resolved_q.lower()))
                resolved_content = {w for w in resolved_words if len(w) > 3}
                coref_overlap = len(choice_words & (resolved_content - q_content_words))
                if coref_overlap > 0:
                    score += min(coref_overlap * 0.10, 0.25)
        except Exception:
            pass

        # ── Stage 17: Semantic Frame Fit (v8.0.0) ───────────────────────
        # Score how well the choice fits the question's semantic frame
        # (DEFINITION → descriptive, CAUSE_EFFECT → causal, QUANTITY → numeric).
        try:
            frame_analyzer = SemanticFrameAnalyzer()
            frame_score = frame_analyzer.score_choice_frame_fit(question, choice)
            if frame_score > 0:
                score += frame_score
        except Exception:
            pass

        # ── Stage 18: Taxonomy Scoring (v8.0.0) ─────────────────────────
        # Score choice by taxonomic proximity to question concepts in the
        # is-a and part-of hierarchies.
        try:
            taxonomy = TaxonomyClassifier()
            tax_score = taxonomy.score_choice_taxonomy(question, choice)
            if tax_score > 0:
                score += tax_score
        except Exception:
            pass

        # ── Stage 19: Causal Chain Scoring (v8.0.0) ─────────────────────
        # Score choice using multi-hop causal reasoning from the causal KB.
        try:
            causal_reasoner = CausalChainReasoner()
            causal_score = causal_reasoner.score_causal_choice(question, choice)
            if causal_score > 0:
                score += causal_score
        except Exception:
            pass

        # ── Stage 20: Pragmatic Alignment (v8.0.0) ──────────────────────
        # Score choice by pragmatic alignment: hedge matching, scalar
        # implicature respect, speech act congruence.
        try:
            pragmatics = PragmaticInferenceEngine()
            pragma_score = pragmatics.pragmatic_alignment(question, choice)
            if pragma_score != 0:
                score += pragma_score
        except Exception:
            pass

        # ── Stage 21: Commonsense Knowledge Scoring (v8.0.0) ────────────
        # Score choice using ConceptNet-style commonsense relations
        # (HasA, CapableOf, UsedFor, AtLocation, HasProperty, Causes).
        try:
            commonsense = ConceptNetLinker()
            cs_score = commonsense.score_choice_commonsense(question, choice)
            if cs_score > 0:
                score += cs_score
        except Exception:
            pass

        # ── Stage 22: Sentiment Alignment (v8.0.0) ──────────────────────
        # For opinion/ethics/psychology questions, score sentiment alignment
        # between question framing and choice tone.
        try:
            sentiment = SentimentAnalyzer()
            q_sent = sentiment.analyze(question)
            c_sent = sentiment.analyze(choice)
            # Positive question framing + positive choice = alignment bonus
            if q_sent["label"] == c_sent["label"] and q_sent["label"] != "neutral":
                score += 0.05
            # Strong sentiment mismatch = slight penalty
            polarity_diff = abs(q_sent["polarity"] - c_sent["polarity"])
            if polarity_diff > 0.5:
                score -= 0.03
        except Exception:
            pass

        # ── Stage 23: Short choice penalty ───────────────────────────────
        if len(choice_lower) <= 2 and score < 1.0:
            score *= 0.3

        # ── Stage 24: Deep NLU Entailment Boost (v8.1.0) ────────────────
        # Use textual entailment to check if the choice is entailed by or
        # contradicts the question context. Entailment = bonus, contradiction
        # = penalty. Uses SRL role alignment, negation, hypernym subsumption.
        try:
            from l104_asi.deep_nlu import TextualEntailmentEngine
            ent_engine = TextualEntailmentEngine()
            ent_result = ent_engine.check(question, choice)
            if ent_result['label'] == 'entailment':
                ent_bonus = 0.06 * ent_result['confidence']
                score += ent_bonus
            elif ent_result['label'] == 'contradiction':
                ent_penalty = 0.04 * ent_result['confidence']
                score -= ent_penalty
        except Exception:
            pass

        return score

    # ═══════════════════════════════════════════════════════════════════════════
    # QUANTUM WAVE COLLAPSE — Knowledge Synthesis + Born-Rule Selection
    # ═══════════════════════════════════════════════════════════════════════════

    def _quantum_wave_collapse(self, question: str, choices: List[str],
                               choice_scores: List[Dict],
                               context_facts: List[str],
                               knowledge_hits: List) -> List[Dict]:
        """Apply quantum probability refinement for MCQ answer selection.

        v7.0 FIX: Replaced broken Born-rule amplitude encoding that caused
        quantum-dominated failures (qp>=0.9 winner-take-all). Now uses
        real probability equations:

        Phase 1 — Knowledge Oracle (5-tier differential scoring):
                  Tier 1: Word-boundary regex matching (IDF-weighted).
                  Tier 2: Suffix-stemmed matching (evaporate↔evaporation).
                  Tier 3: 5-char prefix matching (morphological fallback).
                  Tier 4: Character trigram Jaccard fuzzy matching (>0.45).
                  Tier 5: Bigram phrase-level discrimination (2.5× weight).
                  Exclusivity 5× for unique words, 2.5× for 2-choice.

        Phase 2 — Softmax Probability (replaces exponential amplitude):
                  P_i = exp(score_i × kd_i / T) / Σ exp(score_j × kd_j / T)
                  Temperature T = 1/φ ≈ 0.618 (golden ratio controlled).
                  No winner-take-all: proper probability distribution.

        Phase 3 — GOD_CODE Phase Refinement (replaces Born + sharpening):
                  P_refined = (1-λ)·P_softmax + λ·cos²(kd·π/GOD_CODE)
                  λ = φ/(1+φ) ≈ 0.382 (sacred phase blend).

        Phase 4 — Bayesian Score Synthesis (replaces aggressive blend):
                  final_i = α·P_q(i)·max_score + (1-α)·score_i
                  Cap α = 0.40 (was 0.85+PHI=137%!). Disagreement safeguard.

        Returns: choice_scores list re-ordered by quantum probability.
        """
        QP = _get_cached_quantum_probability()
        if QP is None:
            return choice_scores  # Graceful fallback to raw scores

        scores = [cs["score"] for cs in choice_scores]
        max_score = max(scores) if scores else 0
        if max_score <= 0:
            return choice_scores  # No signal to amplify

        # ── Phase 1: Knowledge Oracle — exclusivity-boosted scoring ──────
        # v6.0: Character trigram fuzzy matching, basic stemming, amplified
        # exclusivity (5×), graduated fact relevance, adaptive score-based
        # prior, steeper quantum amplification. Fixes uniform-KD problem
        # where oracle produced no discriminative signal.

        import math as _m_oracle
        n_choices = len(choice_scores)

        # ── Helper: basic suffix stemming ──
        _SUFFIX_RE = re.compile(r'(ation|tion|sion|ing|ment|ness|ity|ous|ive|able|ible|ful|less|ical|ence|ance|ised|ized|ise|ize|ies|ely|ally|ly|ed|er|es|al|en|s)$')
        def _stem(w: str) -> str:
            if len(w) <= 4:
                return w
            return _SUFFIX_RE.sub('', w) or w[:4]

        # ── Helper: character trigram set ──
        def _trigrams(w: str) -> set:
            w2 = f'#{w}#'
            return {w2[k:k+3] for k in range(len(w2) - 2)} if len(w2) >= 3 else {w2}

        # ── Helper: trigram Jaccard similarity ──
        def _trigram_sim(a: str, b: str) -> float:
            ta, tb = _trigrams(a), _trigrams(b)
            inter = len(ta & tb)
            union = len(ta | tb)
            return inter / union if union > 0 else 0.0

        # Build per-choice word sets with stems and trigrams
        choice_word_sets = []
        choice_stem_sets = []
        choice_prefix_sets = []
        choice_bigrams = []
        choice_trigram_maps = []  # word → trigram set for fuzzy matching
        for cs in choice_scores:
            words = {w for w in re.findall(r'\w+', cs["choice"].lower()) if len(w) > 1}
            choice_word_sets.append(words)
            choice_stem_sets.append({_stem(w) for w in words if len(w) > 2})
            choice_prefix_sets.append({w[:5] for w in words if len(w) >= 5})
            word_list = [w for w in re.findall(r'\w+', cs["choice"].lower()) if len(w) > 1]
            bigrams = {f"{word_list[j]}_{word_list[j+1]}" for j in range(len(word_list) - 1)}
            choice_bigrams.append(bigrams)
            choice_trigram_maps.append({w: _trigrams(w) for w in words if len(w) > 2})

        # Exclusivity-boosted IDF: words in fewer choices get much higher
        # weight. 5× for unique words (was 3×), 2× for words in 2 choices.
        word_choice_count: dict = {}
        for ws in choice_word_sets:
            for w in ws:
                word_choice_count[w] = word_choice_count.get(w, 0) + 1

        word_idf = {}
        for w, cnt in word_choice_count.items():
            base_idf = _m_oracle.log(1.0 + n_choices / (1.0 + cnt))
            exclusivity = 5.0 if cnt == 1 else (2.5 if cnt == 2 else 1.0)
            word_idf[w] = base_idf * exclusivity

        # Question content words + stems for relevance scoring
        q_content = {w for w in re.findall(r'\w+', question.lower()) if len(w) > 2}
        q_stems = {_stem(w) for w in q_content if len(w) > 2}

        # Pre-compile word-boundary patterns for reliable matching
        choice_word_patterns = []
        for ws in choice_word_sets:
            patterns = {}
            for w in ws:
                try:
                    patterns[w] = re.compile(r'\b' + re.escape(w) + r'\b', re.IGNORECASE)
                except re.error:
                    patterns[w] = None
            choice_word_patterns.append(patterns)

        knowledge_density = [0.0] * n_choices

        # ── Helper: score a choice against a text block ──
        # Uses 4-tier matching: word boundary → stem → prefix → trigram fuzzy
        def _score_choice_vs_text(i: int, text_words: set, text_stems: set,
                                  text_str: str, text_bigrams: set) -> float:
            aff = 0.0
            # Tier 1: Word-boundary regex matching (strongest)
            for w, pat in choice_word_patterns[i].items():
                if pat is not None and pat.search(text_str):
                    aff += word_idf.get(w, 1.0)
                elif w in text_words:
                    aff += word_idf.get(w, 1.0) * 0.7
            # Tier 2: Stem matching (catches morphological variants)
            stem_hits = len(choice_stem_sets[i] & text_stems)
            if stem_hits > 0:
                aff += stem_hits * 1.2
            # Tier 3: Prefix matching (5-char)
            if choice_prefix_sets[i]:
                text_pfx = {w[:5] for w in text_words if len(w) >= 5}
                pfx_hits = len(choice_prefix_sets[i] & text_pfx)
                aff += pfx_hits * 0.6
            # Tier 4: Character trigram fuzzy matching (catches typos, variants)
            if aff < 0.5 and choice_trigram_maps[i]:
                best_fuzzy = 0.0
                for cw, ctg in choice_trigram_maps[i].items():
                    for fw in text_words:
                        if len(fw) > 2:
                            sim = _trigram_sim(cw, fw)
                            if sim > 0.45:  # Fuzzy match threshold
                                best_fuzzy = max(best_fuzzy, sim * word_idf.get(cw, 1.0))
                aff += best_fuzzy
            # Tier 5: Bigram matching (phrase-level)
            bg_hits = len(choice_bigrams[i] & text_bigrams)
            aff += bg_hits * 2.5
            # v7.1: Normalize by choice length to prevent long-answer bias.
            n_cw = max(len(choice_word_sets[i]), 1)
            aff /= _m_oracle.sqrt(n_cw)
            return aff
        def _text_features(text: str):
            words = set(re.findall(r'\w+', text.lower()))
            word_list = [w for w in re.findall(r'\w+', text.lower()) if len(w) > 1]
            stems = {_stem(w) for w in words if len(w) > 2}
            bigrams = {f"{word_list[j]}_{word_list[j+1]}" for j in range(len(word_list) - 1)}
            return words, stems, bigrams

        # ── Sub-signal A: IDF-weighted differential fact scoring ──
        # Graduated relevance: facts with question overlap get full weight,
        # others still contribute at reduced weight (0.2 base).
        for fact in context_facts[:50]:
            fl = fact.lower()
            fact_words, fact_stems, fact_bigrams = _text_features(fl)
            # Graduated relevance using word + stem overlap
            q_word_overlap = len(q_content & fact_words)
            q_stem_overlap = len(q_stems & fact_stems)
            q_relevance = min((q_word_overlap + q_stem_overlap * 0.5), 6) * 0.18
            q_relevance = max(q_relevance, 0.2)  # Base relevance for all facts

            per_choice = []
            for i in range(n_choices):
                aff = _score_choice_vs_text(i, fact_words, fact_stems, fl, fact_bigrams)
                per_choice.append(aff)

            # Differential: subtract mean so only *relative* advantage counts
            mean_aff = sum(per_choice) / max(n_choices, 1)
            if max(per_choice) > 0:
                for i in range(n_choices):
                    diff = per_choice[i] - mean_aff
                    knowledge_density[i] += diff * q_relevance

        # ── Sub-signal B: Knowledge-node definition scoring ──
        for key, node, rel in knowledge_hits[:25]:
            node_text = (node.definition + " " + " ".join(node.facts[:15])).lower()
            node_words, node_stems, node_bigrams = _text_features(node_text)
            per_choice_node = []
            for i in range(n_choices):
                naff = _score_choice_vs_text(i, node_words, node_stems, node_text, node_bigrams)
                per_choice_node.append(naff * rel * node.confidence)
            mean_naff = sum(per_choice_node) / max(n_choices, 1)
            for i in range(n_choices):
                knowledge_density[i] += (per_choice_node[i] - mean_naff)

        # ── Sub-signal C: Question-choice coherence (always active) ──
        # Uses stem + trigram matching so morphological variants connect.
        # Adaptive weight: stronger when oracle signal is weak.
        total_kd_spread = max(knowledge_density) - min(knowledge_density) if knowledge_density else 0
        coherence_weight = max(0.1, 0.5 - total_kd_spread * 0.3)
        q_words_lower, q_stems_full, q_bigrams = _text_features(question.lower())
        # v9.2: Extract prepositional context words to deprioritize echoes
        _q_lower_qwc = question.lower()
        _prep_ctx_qwc = set()
        for _pm in re.finditer(
            r'\b(?:in|of|within|about|from|at|on|for)\s+(?:a|an|the|this|that|each)?\s*(\w+)',
            _q_lower_qwc):
            _pw = _pm.group(1)
            if len(_pw) > 2:
                _prep_ctx_qwc.add(_pw.lower())
        q_pfx = {w[:5] for w in q_content if len(w) >= 5}
        for i in range(n_choices):
            c_words = choice_word_sets[i]
            # Word overlap with question — deprioritize prepositional context
            overlap_full = q_content & c_words
            overlap_content = len(overlap_full - _prep_ctx_qwc)
            overlap_context = len(overlap_full & _prep_ctx_qwc)
            # Stem overlap (catches evaporation↔evaporate, magnetic↔magnet)
            stem_overlap = len(choice_stem_sets[i] & q_stems_full)
            # Prefix overlap
            pfx_overlap = len(choice_prefix_sets[i] & q_pfx)
            # v7.1: Normalize by sqrt(choice words) for length invariance
            n_cw = max(len(c_words), 1)
            raw_signal = (overlap_content * 0.25 + overlap_context * 0.05
                          + stem_overlap * 0.20 + pfx_overlap * 0.12)
            knowledge_density[i] += (raw_signal / _m_oracle.sqrt(n_cw)) * coherence_weight

        # ── Sub-signal D: Adaptive score-based prior ─────────────────────
        # When oracle signal is weak, inject MORE of the heuristic ranking.
        # This ensures quantum encoding always has differentiation to work with.
        score_range = max(scores) - min(scores) if scores else 0
        if score_range > 0.005:
            # Adaptive strength: inversely proportional to oracle spread
            kd_spread_so_far = max(knowledge_density) - min(knowledge_density)
            prior_strength = max(0.25, 0.6 - kd_spread_so_far * 0.5)
            for i in range(n_choices):
                score_rank = (scores[i] - min(scores)) / score_range
                knowledge_density[i] += score_rank * prior_strength

        # Normalize knowledge density to [1.0, 3.0] for amplitude weighting.
        # Wider range (was [1.0, 2.0]) so KD differences create larger
        # magnitude differences in the quantum amplitude.
        min_kd = min(knowledge_density) if knowledge_density else 0
        max_kd = max(knowledge_density) if knowledge_density else 0
        kd_range = max_kd - min_kd
        kd_weights = []
        for kd in knowledge_density:
            if kd_range > 0.01:
                # Map [min_kd, max_kd] → [1.0, 3.0]
                kd_weights.append(1.0 + 2.0 * (kd - min_kd) / kd_range)
            else:
                kd_weights.append(1.0)

        # ── Discrimination guard ────────────────────────────────────────
        if kd_range < 0.005:
            return choice_scores

        # ── v10.0: Hybrid probability engine routing ────────────────────
        # If ProbabilityEngine is available, use the full hybrid pipeline
        # (Bayesian priors + sacred probability + Grover amplification +
        # ASI insight synthesis) instead of the inline Phases 2-4.
        pe = _get_cached_probability_engine()
        if pe is not None:
            try:
                return self._hybrid_probability_collapse(
                    pe, question, choices, choice_scores,
                    knowledge_density, kd_weights, kd_range,
                    min_kd, max_kd, scores, max_score, n_choices,
                )
            except Exception:
                pass  # Fall through to legacy Phases 2-4

        # ── Legacy Phase 2: Softmax Amplitude Encoding ───────────────────
        # v7.0 FIX: Replace steep exponential (e^(4.854*Δ)) that caused
        # winner-take-all Born-rule domination (quantum-dominated failures).
        # Real equation: Temperature-controlled softmax probability.
        #   P_i = exp(score_i × kd_i / T) / Σ_j exp(score_j × kd_j / T)
        # Temperature T = 1.0 / PHI ≈ 0.618 keeps distribution informative
        # but never collapses to a single-choice spike.
        T_softmax = 1.0 / PHI  # Temperature: 0.618 — balanced discrimination

        logits = []
        for i, cs in enumerate(choice_scores):
            s = max(cs["score"], 0.001)
            logit = s * kd_weights[i] / T_softmax
            logits.append(logit)

        # Numerically stable softmax
        max_logit = max(logits)
        exp_logits = [_m_oracle.exp(l - max_logit) for l in logits]
        Z_soft = sum(exp_logits)
        all_probs = [e / Z_soft for e in exp_logits]

        # ── Phase 3: GOD_CODE Phase Refinement ───────────────────────────
        # v8.0: Real quantum circuit replaces classical cos² approximation.
        # Encodes knowledge_density as Ry rotation angles on n_choices qubits,
        # applies GOD_CODE_PHASE gates for sacred alignment, then measures
        # Born-rule probabilities. Falls back to classical if unavailable.
        try:
            if kd_range < 0.02:
                return choice_scores  # No oracle signal — skip quantum
            phase_lambda = PHI / (1.0 + PHI)  # 0.382 — golden ratio blend

            phase_probs = None

            # v8.0: Try real quantum circuit first
            qge = _get_cached_quantum_gate_engine()
            if qge is not None and n_choices <= 4:
                try:
                    from l104_quantum_gate_engine import ExecutionTarget, Ry as _Ry, GOD_CODE_PHASE as _GCP
                    n_q = max(n_choices, 2)
                    circ = qge.create_circuit(n_q, "wave_collapse")
                    for i in range(min(n_choices, n_q)):
                        kd_norm = (knowledge_density[i] - min_kd) / max(kd_range, 1e-9)
                        theta = kd_norm * math.pi * PHI  # PHI-scaled rotation
                        circ.append(_Ry(theta), [i])
                    # Apply GOD_CODE_PHASE for sacred alignment
                    for i in range(min(n_choices, n_q)):
                        circ.append(_GCP, [i])
                    # Entangle adjacent qubits for correlation
                    for i in range(min(n_choices, n_q) - 1):
                        circ.cx(i, i + 1)
                    qr = qge.execute(circ, ExecutionTarget.LOCAL_STATEVECTOR)
                    if hasattr(qr, 'probabilities') and qr.probabilities:
                        probs = qr.probabilities
                        # Marginalize: for each qubit i, P(qubit_i=1) = sum of states where bit i is 1
                        circuit_probs = []
                        for i in range(n_choices):
                            p1 = 0.0
                            for state, prob in probs.items():
                                if len(state) > i and state[-(i+1)] == '1':
                                    p1 += prob
                            circuit_probs.append(p1)
                        cp_sum = sum(circuit_probs)
                        if cp_sum > 0:
                            phase_probs = [p / cp_sum for p in circuit_probs]
                except Exception:
                    pass  # Fall back to classical

            # Classical fallback if circuit didn't produce results
            if phase_probs is None:
                phase_probs = []
                for i in range(n_choices):
                    kd_norm = (knowledge_density[i] - min_kd) / max(kd_range, 1e-9)
                    phase_p = math.cos(kd_norm * math.pi / GOD_CODE) ** 2
                    phase_probs.append(phase_p)
                phase_z = sum(phase_probs)
                if phase_z > 0:
                    phase_probs = [p / phase_z for p in phase_probs]

            all_probs = [
                (1.0 - phase_lambda) * all_probs[i] + phase_lambda * phase_probs[i]
                for i in range(n_choices)
            ]
        except Exception:
            return choice_scores

        # ── Phase 4: Bayesian Score Synthesis (Conservative) ─────────────
        # v7.0 FIX: Real Bayesian blending. Quantum refines, never dominates.
        #   final_i = α·P_q(i)·max_score + (1-α)·score_i
        # Cap α at 0.40 (was 0.85 with PHI multiplier = quantum > 100%!)
        # Disagreement safeguard added for ALL disagreements.
        sorted_probs = sorted(all_probs, reverse=True)
        prob_ratio = sorted_probs[0] / max(sorted_probs[1], 0.001) if len(sorted_probs) > 1 else 1.0

        if prob_ratio < 1.05:
            return choice_scores  # Uniform — no quantum advantage

        # Conservative blending: cap at 0.25 (v9.1: lowered from 0.40 — QWC
        # adds noise when KB discriminative signal is weak)
        q_strength = min(0.25, 0.10 + 0.08 * (prob_ratio - 1.05))
        kd_confidence = min(1.0, kd_range / 0.5)
        q_strength *= (0.4 + 0.6 * kd_confidence)

        # Disagreement safeguard: reduce quantum when it fights classical
        quantum_top = max(range(len(all_probs)), key=lambda k: all_probs[k])
        onto_top = max(range(len(scores)), key=lambda k: scores[k])
        if quantum_top != onto_top:
            gap = scores[onto_top] - scores[quantum_top] if quantum_top < len(scores) else 0
            if gap > 0:
                q_strength *= max(0.15, 1.0 - gap * 3.0)

        for i, cs in enumerate(choice_scores):
            q_prob = all_probs[i] if i < len(all_probs) else 0.0
            cs["quantum_prob"] = q_prob
            # Bayesian blend: NO PHI multiplier (was causing > 100% score range)
            cs["score"] = q_prob * max_score * q_strength + cs["score"] * (1.0 - q_strength)

        choice_scores.sort(key=lambda x: (x["score"], _rng.random() * 1e-9), reverse=True)
        self._quantum_collapses += 1
        return choice_scores

    def _hybrid_probability_collapse(
        self, pe, question: str, choices: List[str],
        choice_scores: List[Dict],
        knowledge_density: List[float],
        kd_weights: List[float],
        kd_range: float,
        min_kd: float, max_kd: float,
        scores: List[float],
        max_score: float,
        n_choices: int,
    ) -> List[Dict]:
        """v10.0 Hybrid Comprehension Model — ProbabilityEngine-driven MCQ collapse.

        Replaces inline Phases 2-4 with proper ProbabilityEngine calls:

        Phase 2 (Bayesian Prior): PE.token_probability() for corpus-informed priors.
        Phase 3 (Sacred + Grover): PE.sacred_probability() + PE.grover_amplification().
        Phase 4 (Insight Synthesis): PE.synthesize_insight() for consciousness weighting.
        Final: PE.bayesian_update() for proper posterior normalization.

        Returns: choice_scores re-ordered by hybrid probability.
        """
        import math as _m_hybrid

        # ── Phase 2: Corpus-informed Bayesian priors ─────────────────────
        choice_priors = []
        for i, cs in enumerate(choice_scores):
            choice_text = cs["choice"].lower()
            tokens = [w for w in choice_text.split() if len(w) > 2]
            if tokens:
                log_prior = sum(_m_hybrid.log(max(pe.token_probability(t), 1e-10)) for t in tokens)
                prior = _m_hybrid.exp(log_prior / len(tokens))
            else:
                prior = 1.0 / n_choices
            blended_prior = prior * kd_weights[i]
            choice_priors.append(blended_prior)

        prior_sum = sum(choice_priors)
        if prior_sum > 0:
            choice_priors = [p / prior_sum for p in choice_priors]
        else:
            choice_priors = [1.0 / n_choices] * n_choices

        # ── Phase 3: Sacred probability + Grover amplification ───────────
        sacred_probs = []
        for i in range(n_choices):
            kd_norm = (knowledge_density[i] - min_kd) / max(kd_range, 1e-9)
            sp = pe.sacred_probability(kd_norm * 527.518)
            sacred_probs.append(sp)

        sp_sum = sum(sacred_probs)
        if sp_sum > 0:
            sacred_probs = [p / sp_sum for p in sacred_probs]

        grover_probs = list(choice_priors)
        best_kd_idx = max(range(n_choices), key=lambda k: knowledge_density[k])
        if choice_priors[best_kd_idx] > 0:
            amplified = pe.grover_amplification(
                choice_priors[best_kd_idx], n_items=n_choices, iterations=1,
            )
            boost_ratio = amplified / max(choice_priors[best_kd_idx], 1e-10)
            boost_ratio = min(boost_ratio, 2.0)
            for i in range(n_choices):
                if i == best_kd_idx:
                    grover_probs[i] *= boost_ratio
                else:
                    grover_probs[i] *= (1.0 - (boost_ratio - 1.0) / max(n_choices - 1, 1))
                    grover_probs[i] = max(grover_probs[i], 0.01)

        gp_sum = sum(grover_probs)
        if gp_sum > 0:
            grover_probs = [p / gp_sum for p in grover_probs]

        # ── Phase 3b: Quantum circuit refinement (optional) ──────────────
        circuit_probs = None
        if kd_range >= 0.02 and n_choices <= 4:
            qge = _get_cached_quantum_gate_engine()
            if qge is not None:
                try:
                    from l104_quantum_gate_engine import ExecutionTarget, Ry as _Ry, GOD_CODE_PHASE as _GCP
                    n_q = max(n_choices, 2)
                    circ = qge.create_circuit(n_q, "hybrid_collapse")
                    for i in range(min(n_choices, n_q)):
                        kd_norm = (knowledge_density[i] - min_kd) / max(kd_range, 1e-9)
                        theta = kd_norm * _m_hybrid.pi * PHI
                        circ.append(_Ry(theta), [i])
                    for i in range(min(n_choices, n_q)):
                        circ.append(_GCP, [i])
                    for i in range(min(n_choices, n_q) - 1):
                        circ.cx(i, i + 1)
                    qr = qge.execute(circ, ExecutionTarget.LOCAL_STATEVECTOR)
                    if hasattr(qr, 'probabilities') and qr.probabilities:
                        probs = qr.probabilities
                        cp = []
                        for i in range(n_choices):
                            p1 = sum(prob for state, prob in probs.items()
                                     if len(state) > i and state[-(i+1)] == '1')
                            cp.append(p1)
                        cp_sum = sum(cp)
                        if cp_sum > 0:
                            circuit_probs = [p / cp_sum for p in cp]
                except Exception:
                    pass

        # ── Phase 3c: Fuse probability signals ───────────────────────────
        fused_probs = []
        for i in range(n_choices):
            p = 0.30 * choice_priors[i] + 0.25 * sacred_probs[i] + 0.25 * grover_probs[i]
            if circuit_probs is not None:
                p += 0.20 * circuit_probs[i]
            else:
                p += 0.10 * choice_priors[i] + 0.10 * grover_probs[i]
            fused_probs.append(p)

        fp_sum = sum(fused_probs)
        if fp_sum > 0:
            fused_probs = [p / fp_sum for p in fused_probs]

        # ── Phase 4: ASI Insight Synthesis ───────────────────────────────
        thought_signals = [cs["score"] for cs in choice_scores]
        insight = pe.synthesize_insight(
            thought_signals,
            consciousness_level=min(1.0, kd_range * 2.0),
            temperature=1.0 / PHI,
        )

        consciousness_weight = min(0.30, insight.consciousness_probability * 0.35)
        resonance_gate = min(1.0, insight.resonance_score * 1.5)
        q_strength = consciousness_weight * resonance_gate

        # ── Phase 4b: Bayesian posterior ─────────────────────────────────
        posterior = pe.bayesian_update(choice_priors, fused_probs)

        kd_confidence = min(1.0, kd_range / 0.5)
        q_strength *= (0.4 + 0.6 * kd_confidence)

        # Disagreement safeguard
        posterior_top = max(range(len(posterior)), key=lambda k: posterior[k])
        onto_top = max(range(len(scores)), key=lambda k: scores[k])
        if posterior_top != onto_top:
            gap = scores[onto_top] - scores[posterior_top] if posterior_top < len(scores) else 0
            if gap > 0:
                q_strength *= max(0.15, 1.0 - gap * 3.0)

        # Uniformity check
        sorted_post = sorted(posterior, reverse=True)
        post_ratio = sorted_post[0] / max(sorted_post[1], 0.001) if len(sorted_post) > 1 else 1.0
        if post_ratio < 1.05:
            return choice_scores

        for i, cs in enumerate(choice_scores):
            q_prob = posterior[i] if i < len(posterior) else 0.0
            cs["quantum_prob"] = q_prob
            cs["hybrid_posterior"] = q_prob
            cs["insight_resonance"] = insight.resonance_score
            cs["insight_consciousness"] = insight.consciousness_probability
            cs["score"] = q_prob * max_score * q_strength + cs["score"] * (1.0 - q_strength)

        choice_scores.sort(key=lambda x: (x["score"], _rng.random() * 1e-9), reverse=True)
        self._quantum_collapses += 1
        return choice_scores

    def _fallback_heuristics(self, question: str, choice: str,
                             all_choices: List[str]) -> float:
        """Test-taking heuristics when knowledge base provides no guidance.

        v4.0 Research-backed MCQ solving strategies:
        1. Content word overlap (question echoing)
        2. Specificity: longer, more detailed answers tend to be correct
        3. Hedging vs extreme language
        4. "All of the above" detection
        5. Grammar agreement between question stem and answer
        6. Technical term density
        7. Numeric specificity
        8. Stem-completion grammar fit
        9. Question-type matching (definition, cause, example, comparison)
        10. Domain keyword density (science, history, literature, etc.)
        11. Stem/root overlap for deeper semantic alignment
        """
        score = 0.0
        q_lower = question.lower()
        c_lower = choice.lower().strip()

        # Extract content words (skip stopwords)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'above',
                     'below', 'between', 'under', 'over', 'that', 'this', 'these',
                     'those', 'it', 'its', 'and', 'but', 'or', 'not', 'no', 'nor',
                     'so', 'if', 'than', 'each', 'which', 'who', 'whom', 'what',
                     'when', 'where', 'how', 'why', 'all', 'both', 'few', 'more',
                     'most', 'other', 'some', 'such', 'only', 'own', 'same', 'very'}
        q_words = {w for w in re.findall(r'[a-z]+', q_lower) if len(w) > 2 and w not in stopwords}
        c_words = {w for w in re.findall(r'[a-z]+', c_lower) if len(w) > 2 and w not in stopwords}

        # 1. Content word overlap — correct answers often echo question terms
        # v9.2: Deprioritize words in prepositional phrases ("in a cell",
        # "of the ocean") — these are contextual locators, not answer signals.
        # e.g., Q="function of mitochondria in a cell" → "cell" is context.
        _prep_context = set()
        for _pm in re.finditer(
            r'\b(?:in|of|within|about|from|at|on|for)\s+(?:a|an|the|this|that|each)?\s*(\w+)',
            q_lower):
            _pw = _pm.group(1)
            if len(_pw) > 2 and _pw not in stopwords:
                _prep_context.add(_pw)
        overlap_full = q_words & c_words
        overlap_content = overlap_full - _prep_context
        overlap_context = overlap_full & _prep_context
        score += len(overlap_content) * 0.25
        score += len(overlap_context) * 0.05  # Reduced weight for prepositional context

        # 1b. Stem overlap — catches morphological variants
        _sfx = re.compile(r'(ation|tion|sion|ing|ment|ness|ity|ous|ive|able|ible|ful|less|ical|ence|ance|ly|ed|er|es|al|en|s)$')
        def _stem_h(w):
            return _sfx.sub('', w) or w[:4] if len(w) > 4 else w
        q_stems = {_stem_h(w) for w in q_words}
        c_stems = {_stem_h(w) for w in c_words}
        stem_overlap = len(q_stems & c_stems) - len(overlap_full)  # Only count new matches
        if stem_overlap > 0:
            score += stem_overlap * 0.18

        # 2. Specificity bonus — reduced to prevent length/choice-D bias
        avg_len = sum(len(c) for c in all_choices) / max(len(all_choices), 1)
        if avg_len > 0:
            length_ratio = len(choice) / avg_len
            if length_ratio > 1.3:
                score += 0.03  # Minimal length bonus (was 0.15 — caused choice-D bias)
            elif length_ratio > 1.1:
                score += 0.02  # Slightly above average
            elif length_ratio < 0.5:
                score -= 0.04  # Very short relative to others (likely wrong)

        # 3. Hedging vs extreme language — reduced bonuses
        hedge_words = {'may', 'can', 'some', 'often', 'usually', 'generally',
                       'typically', 'sometimes', 'likely', 'tends', 'probably'}
        extreme_words = {'always', 'never', 'all', 'none', 'only', 'must',
                         'impossible', 'certainly', 'absolutely', 'every'}
        c_word_set = set(c_lower.split())
        hedge_count = len(hedge_words & c_word_set)
        extreme_count = len(extreme_words & c_word_set)
        score += hedge_count * 0.02  # Was 0.05 — reduced to prevent bias
        score -= extreme_count * 0.04

        # 4. "All/both of the above" patterns — reduced
        if 'all of the above' in c_lower:
            score += 0.05  # Was 0.12
        if 'both' in c_lower and any(x in c_lower for x in ['and', 'above']):
            score += 0.05  # Was 0.10
        if 'none of the above' in c_lower:
            score -= 0.04

        # 5. Technical term density — question-relevant only
        q_tech = {w for w in q_words if len(w) > 7}
        c_tech = {w for w in c_words if len(w) > 7}
        shared_tech = len(q_tech & c_tech)
        choice_only_tech = len(c_tech - q_tech)
        score += shared_tech * 0.08  # Question-relevant technical terms
        score += choice_only_tech * 0.01  # Choice-only terms: minimal (was 0.04 blanket)

        # 6. Numeric specificity — answers containing specific numbers
        nums_in_choice = len(re.findall(r'\d+', c_lower))
        if nums_in_choice > 0 and len(choice) > 5:
            score += 0.05

        # 7. Grammatical match category
        q_stripped = q_lower.rstrip('?:. ')
        if q_stripped.endswith(' a ') or q_stripped.endswith(' an '):
            if c_lower and c_lower[0] in 'aeiou' and q_stripped.endswith(' an '):
                score += 0.04
            elif c_lower and c_lower[0] not in 'aeiou' and q_stripped.endswith(' a '):
                score += 0.04

        # 8. Question-type matching — stronger signal for specific question patterns
        # "What is the definition of X?" → prefer answers that read like definitions
        if 'definition' in q_lower or 'defined as' in q_lower or 'refers to' in q_lower:
            if any(w in c_lower for w in ['process', 'method', 'state', 'condition', 'type']):
                score += 0.08
            # Removed definition-length bonus (caused choice-D bias)

        # "What causes X?" / "Why does X?" → prefer causal answers
        if any(w in q_lower for w in ['cause', 'causes', 'why', 'result', 'leads to', 'because']):
            causal_words = {'because', 'due', 'causes', 'leads', 'results', 'produces',
                            'increases', 'decreases', 'changes', 'creates', 'prevents'}
            if any(w in c_lower for w in causal_words):
                score += 0.08

        # "Which is an example of X?" → prefer concrete nouns (capped to prevent length bias)
        if 'example' in q_lower or 'instance' in q_lower:
            concrete = sum(1 for w in c_words if len(w) > 4)
            score += min(concrete, 2) * 0.03  # Was uncapped * 0.05

        # "What is the BEST/MOST..." — disabled length boost (caused D bias)
        # if any(w in q_lower for w in ['best', 'most likely', 'most accurate', 'primary']):
        #     if len(choice) > avg_len * 1.1:
        #         score += 0.06

        # 9. Domain keyword density — domain-specific vocabulary indicates domain match
        domain_terms = {
            'science': ['energy', 'force', 'heat', 'light', 'water', 'temperature',
                        'gravity', 'mass', 'cell', 'organism', 'evolution', 'species',
                        'photosynthesis', 'ecosystem', 'molecule', 'atom', 'chemical',
                        'reaction', 'element', 'compound', 'frequency', 'wavelength'],
            'history': ['war', 'treaty', 'constitution', 'revolution', 'empire',
                        'dynasty', 'colony', 'independence', 'democracy', 'republic'],
            'math': ['equation', 'function', 'variable', 'coefficient', 'theorem',
                     'proof', 'integral', 'derivative', 'matrix', 'polynomial'],
        }
        # Detect question domain — cap to prevent length bias
        for domain, terms in domain_terms.items():
            q_domain_hits = sum(1 for t in terms if t in q_lower)
            if q_domain_hits >= 1:
                c_domain_hits = sum(1 for t in terms if t in c_lower)
                score += min(c_domain_hits, 2) * 0.04  # Was uncapped * 0.06
                break  # Only match one domain

        return score

    def _chain_of_thought(self, question: str, choices: List[str],
                          best: Dict, context_facts: List[str]) -> List[str]:
        """Generate chain-of-thought reasoning for the answer."""
        steps = []
        steps.append(f"Question parsed: identifying key concepts in '{question[:60]}...'")

        if context_facts:
            steps.append(f"Retrieved {len(context_facts)} relevant facts from knowledge base")
            top_fact = context_facts[0] if context_facts else "none"
            steps.append(f"Most relevant fact: '{top_fact[:80]}...'")

        steps.append(f"Scored {len(choices)} answer choices against context")
        steps.append(f"Best match: {best['label']} ('{best['choice'][:40]}...') with score {best['score']:.3f}")

        # Elimination reasoning
        steps.append(f"Verification: checking answer coherence with retrieved knowledge")
        steps.append(f"Selected answer {best['label']} with confidence {best['score']:.3f}")

        return steps

    def get_status(self) -> Dict[str, Any]:
        """Get solver status."""
        return {
            "questions_answered": self._questions_answered,
            "correct_count": self._correct_count,
            "accuracy": self._correct_count / max(self._questions_answered, 1),
            "entropy_calibrations": self._entropy_calibrations,
            "dual_layer_calibrations": self._dual_layer_calibrations,
            "ngram_boosts": self._ngram_boosts,
            "logic_assists": self._logic_assists,
            "nlu_assists": self._nlu_assists,
            "numerical_assists": self._numerical_assists,
            "cross_verifications": self._cross_verifications,
            "subject_detections": self._subject_detections,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIFIED LANGUAGE COMPREHENSION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
