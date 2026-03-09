from __future__ import annotations

import math
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from .constants import PHI, TAU, VOID_CONSTANT
from .engine_support import _get_cached_dual_layer, _get_cached_science_engine

_rng = random.Random(104)  # Deterministic tie-breaking RNG

_rng = random.Random(104)  # Deterministic tie-breaking RNG

class CrossVerificationEngine:
    """Multi-strategy answer verification and elimination engine.

    Validates the top-scoring answer by cross-checking against multiple
    independent signals with PHI-calibrated agreement weighting.

    v4.0 Strategies:
      1. Fact-support count with stem/prefix matching (not just exact words)
      2. Mutual information: boost choices that co-occur with question terms
         in facts more than expected by chance
      3. Elimination: negation-window contradiction detection
      4. Inter-choice tiebreaker with specificity + domain-term density
      5. PHI-calibrated agreement with VOID_CONSTANT decay
      6. Confidence gap amplification: widen small leads

    Uses VOID_CONSTANT for decay calibration of verification confidence.
    """

    def __init__(self):
        self._verifications = 0
        self._eliminations = 0
        # Suffix stemmer for morphological matching
        self._suffix_re = re.compile(r'(ation|tion|sion|ing|ment|ness|ity|ous|ive|able|ible|ful|less|ical|ence|ance|ised|ized|ise|ize|ies|ely|ally|ly|ed|er|es|al|en|s)$')

    def _stem(self, w: str) -> str:
        if len(w) <= 4:
            return w
        return self._suffix_re.sub('', w) or w[:4]

    def verify(self, question: str, choice_scores: List[Dict],
               context_facts: List[str],
               knowledge_hits: List) -> List[Dict]:
        """Run cross-verification on scored choices.

        Modifies choice_scores in-place with verification bonuses/penalties.
        Returns the re-sorted choice_scores list.
        """
        self._verifications += 1

        if not choice_scores or not context_facts:
            return choice_scores

        q_lower = question.lower()
        q_words = {w for w in re.findall(r'\w+', q_lower) if len(w) > 3}
        q_stems = {self._stem(w) for w in q_words}

        # Record incoming leader to detect rank distortion
        _incoming_leader = max(range(len(choice_scores)),
                               key=lambda i: choice_scores[i]["score"])

        # === Strategy 1: Fact-support count (with stem matching) ===
        # Count how many distinct facts mention each choice's key terms.
        # Uses stem matching so "evaporation" matches "evaporate" in facts.
        top_facts_for_cv = context_facts[:15]
        for cs in choice_scores:
            c_lower = cs["choice"].lower()
            c_words = {w for w in re.findall(r'\w+', c_lower) if len(w) > 2}
            c_stems = {self._stem(w) for w in c_words if len(w) > 2}
            c_prefixes = {w[:5] for w in c_words if len(w) >= 5}
            support_count = 0
            for fact in top_facts_for_cv:
                fl = fact.lower()
                f_words = set(re.findall(r'\w+', fl))
                f_stems = {self._stem(w) for w in f_words if len(w) > 2}
                f_prefixes = {w[:5] for w in f_words if len(w) >= 5}
                # Question relevance: word OR stem overlap
                q_in_fact = len(q_words & f_words) + len(q_stems & f_stems) * 0.5
                # Choice relevance: word OR stem OR prefix overlap
                c_in_fact = len(c_words & f_words) + len(c_stems & f_stems) * 0.7 + len(c_prefixes & f_prefixes) * 0.3
                if q_in_fact >= 1 and c_in_fact >= 1:
                    support_count += 1
            # Graduated bonus: diminishing returns after 3 supporting facts
            if support_count > 0:
                cs["score"] += min(support_count, 5) * 0.10 * (1.0 / (1.0 + support_count * 0.1))

        # === Strategy 2: Mutual information — co-occurrence signal ===
        # Boost choices where question+choice terms co-occur in same fact
        # significantly more than expected by random chance.
        total_facts = len(top_facts_for_cv)
        for cs in choice_scores:
            c_lower = cs["choice"].lower()
            c_words = {w for w in re.findall(r'\w+', c_lower) if len(w) > 3}
            if not c_words:
                continue
            cooccur = 0
            c_alone = 0
            for fact in top_facts_for_cv:
                fl = fact.lower()
                has_q = any(w in fl for w in q_words)
                has_c = any(w in fl for w in c_words)
                if has_q and has_c:
                    cooccur += 1
                elif has_c:
                    c_alone += 1
            # MI signal: co-occurrence rate above baseline
            if total_facts > 0 and cooccur > 0:
                expected = (cooccur + c_alone) / total_facts * len([1 for f in top_facts_for_cv if any(w in f.lower() for w in q_words)]) / total_facts
                actual = cooccur / total_facts
                mi_boost = max(0, actual - expected) * 2.0
                cs["score"] += min(mi_boost, 0.3)

        # === Strategy 3: Elimination — detect contradicting facts ===
        anti_patterns = {
            "not": -0.2, "never": -0.25, "cannot": -0.2,
            "incorrect": -0.3, "false": -0.15, "wrong": -0.2,
            "except": -0.15, "unlike": -0.1,
        }
        for cs in choice_scores:
            c_lower = cs["choice"].lower()
            for fact in context_facts[:15]:
                fl = fact.lower()
                if c_lower[:15] in fl or any(w in fl for w in c_lower.split() if len(w) > 4):
                    for neg_word, penalty in anti_patterns.items():
                        idx = fl.find(c_lower[:10]) if c_lower[:10] in fl else -1
                        if idx >= 0:
                            window = fl[max(0, idx - 20):idx + len(c_lower) + 20]
                            if neg_word in window:
                                cs["score"] += penalty
                                self._eliminations += 1
                                break

        # === Strategy 4: Inter-choice tiebreaker ===
        choice_scores.sort(key=lambda x: (x["score"], _rng.random()), reverse=True)
        if len(choice_scores) >= 2:
            top = choice_scores[0]["score"]
            second = choice_scores[1]["score"]
            if top > 0 and second > 0 and abs(top - second) / max(top, 0.01) < 0.10:
                # Very close: apply tiebreaker based on specificity + domain terms
                for cs in choice_scores[:2]:
                    specificity = len(cs["choice"]) / 50.0
                    # Domain term density bonus
                    c_words = set(cs["choice"].lower().split())
                    tech_count = sum(1 for w in c_words if len(w) > 7)
                    cs["score"] += (specificity * 0.05 + tech_count * 0.03) * VOID_CONSTANT

        # === Strategy 5: PHI-calibrated agreement ===
        # Only amplify if the leader is the same as the incoming leader
        # (prevents cross-verification from flipping then amplifying a wrong answer)
        choice_scores.sort(key=lambda x: (x["score"], _rng.random()), reverse=True)
        if len(choice_scores) >= 2:
            current_leader = choice_scores[0]["index"]
            if current_leader == choice_scores[_incoming_leader]["index"] if _incoming_leader < len(choice_scores) else True:
                top = choice_scores[0]["score"]
                second = choice_scores[1]["score"]
                if top > second * PHI:  # Golden ratio separation
                    boost = (top - second) * TAU * VOID_CONSTANT * 0.15
                    choice_scores[0]["score"] += min(boost, 0.5)

        # === Strategy 6: Confidence gap amplification ===
        # Only amplify the SAME leader as incoming — if cross-verification
        # flipped the ranking, do NOT amplify the new leader.
        choice_scores.sort(key=lambda x: (x["score"], _rng.random()), reverse=True)
        if len(choice_scores) >= 2:
            current_leader_idx = choice_scores[0]["index"]
            incoming_leader_idx = choice_scores[_incoming_leader]["index"] if _incoming_leader < len(choice_scores) else -1
            if current_leader_idx == incoming_leader_idx:
                top = choice_scores[0]["score"]
                second = choice_scores[1]["score"]
                ratio = top / max(second, 0.001)
                if 1.1 < ratio < 1.5 and top > 0.1:
                    gap = top - second
                    amplified_gap = gap * (PHI ** 0.5)  # √φ ≈ 1.272
                    choice_scores[0]["score"] = second + amplified_gap

        choice_scores.sort(key=lambda x: (x["score"], _rng.random()), reverse=True)
        return choice_scores


# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM LAYER: Textual Entailment Engine
# ═══════════════════════════════════════════════════════════════════════════════
