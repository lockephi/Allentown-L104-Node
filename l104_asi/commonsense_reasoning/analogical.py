"""Layer 5: Analogical Reasoning Engine — A:B :: C:? Pattern Matching."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .constants import PHI, GOD_CODE, VOID_CONSTANT
from .ontology import Concept, ConceptOntology

logger = logging.getLogger(__name__)


class AnalogicalReasoner:
    """Structural analogy engine: A:B :: C:?

    Finds matching relational patterns between concept pairs.
    Used in ARC for questions like "Which is most similar to...?"
    """

    # Relation types for analogical mapping
    RELATION_TYPES = {
        'is_a': lambda o, c: c.name.lower() in [p.lower() for p in (o.lookup(c.name) or Concept(name="")).parents] if o.lookup(c.name) else False,
        'opposite': [
            ("hot", "cold"), ("light", "dark"), ("fast", "slow"), ("big", "small"),
            ("hard", "soft"), ("wet", "dry"), ("solid", "liquid"), ("liquid", "gas"),
            ("predator", "prey"), ("producer", "consumer"), ("acid", "base"),
            ("positive", "negative"), ("north", "south"), ("attract", "repel"),
            ("evaporation", "condensation"), ("melting", "freezing"),
            ("expand", "contract"), ("absorb", "reflect"),
        ],
        'part_whole': [
            ("cell", "organism"), ("leaf", "plant"), ("heart", "circulatory_system"),
            ("electron", "atom"), ("planet", "solar_system"), ("crust", "earth"),
            ("root", "plant"), ("lung", "respiratory_system"), ("gear", "machine"),
        ],
        'cause_effect': [
            ("heat", "melting"), ("cold", "freezing"), ("rain", "flood"),
            ("earthquake", "tsunami"), ("pollution", "global_warming"),
            ("photosynthesis", "oxygen"), ("erosion", "canyon"),
            ("gravity", "falling"), ("friction", "heat"), ("vibration", "sound"),
        ],
        'tool_function': [
            ("thermometer", "temperature"), ("ruler", "length"),
            ("scale", "mass"), ("seismograph", "earthquake"),
            ("telescope", "stars"), ("microscope", "cells"),
            ("barometer", "pressure"), ("anemometer", "wind_speed"),
        ],
    }

    def __init__(self, ontology: ConceptOntology):
        self.ontology = ontology

    def find_analogy(self, a: str, b: str, c: str, candidates: List[str] = None) -> Dict[str, Any]:
        """Solve A:B :: C:? — Find what relates to C the way B relates to A.

        Returns scored candidates.
        """
        # Determine the relationship between A and B
        relationship = self._identify_relationship(a, b)

        if not candidates:
            # Generate candidates from ontology
            c_concept = self.ontology.lookup(c)
            if c_concept:
                candidates = c_concept.related + c_concept.parents + c_concept.parts
            else:
                candidates = []

        # Score each candidate by how well it maps the A:B relationship to C:?
        scored = []
        for d in candidates:
            score = self._score_analogy(a, b, c, d, relationship)
            scored.append((d, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return {
            'a': a, 'b': b, 'c': c,
            'relationship': relationship,
            'best_match': scored[0][0] if scored else None,
            'candidates': scored[:5],
        }

    def _identify_relationship(self, a: str, b: str) -> str:
        """Identify the relationship between two concepts."""
        a_l, b_l = a.lower(), b.lower()

        # Check opposites
        for pair in self.RELATION_TYPES['opposite']:
            if (a_l in pair[0] and b_l in pair[1]) or (a_l in pair[1] and b_l in pair[0]):
                return 'opposite'

        # Check part-whole
        for pair in self.RELATION_TYPES['part_whole']:
            if (a_l in pair[0] and b_l in pair[1]) or (a_l in pair[1] and b_l in pair[0]):
                return 'part_whole'

        # Check cause-effect
        for pair in self.RELATION_TYPES['cause_effect']:
            if (a_l in pair[0] and b_l in pair[1]) or (a_l in pair[1] and b_l in pair[0]):
                return 'cause_effect'

        # Check tool-function
        for pair in self.RELATION_TYPES['tool_function']:
            if (a_l in pair[0] and b_l in pair[1]) or (a_l in pair[1] and b_l in pair[0]):
                return 'tool_function'

        # Check IS-A via ontology
        if self.ontology.is_a(a, b) or self.ontology.is_a(b, a):
            return 'is_a'

        return 'unknown'

    def _score_analogy(self, a: str, b: str, c: str, d: str, relationship: str) -> float:
        """Score how well C:D matches the relationship of A:B."""
        cd_rel = self._identify_relationship(c, d)
        if cd_rel == relationship and relationship != 'unknown':
            return 0.9

        # Partial match: check structural similarity
        score = 0.0

        # Same category
        a_c = self.ontology.lookup(a)
        b_c = self.ontology.lookup(b)
        c_c = self.ontology.lookup(c)
        d_c = self.ontology.lookup(d)

        if a_c and c_c and a_c.category == c_c.category:
            score += 0.2
        if b_c and d_c and b_c.category == d_c.category:
            score += 0.2

        # Word-level similarity
        a_words = set(a.lower().split('_'))
        b_words = set(b.lower().split('_'))
        c_words = set(c.lower().split('_'))
        d_words = set(d.lower().split('_'))

        if a_words & c_words:
            score += 0.1
        if b_words & d_words:
            score += 0.1

        return score


