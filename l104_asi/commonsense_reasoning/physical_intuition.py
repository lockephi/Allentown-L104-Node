"""Layer 3: Physical Intuition — Scientific Common Sense."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from .constants import PHI, GOD_CODE, VOID_CONSTANT
from .ontology import ConceptOntology

logger = logging.getLogger(__name__)


class PhysicalIntuition:
    """Physical intuition rules for reasoning about everyday phenomena.

    Encodes basic physical laws and their observable consequences
    that humans learn through experience.
    """

    # Property inference rules: if concept has property X, what can we conclude
    INFERENCE_RULES = {
        "conducts_heat": {True: "heat transfers through it easily", False: "heat does not transfer through it easily"},
        "conducts_electricity": {True: "electricity flows through it", False: "electricity does not flow through it"},
        "compressible": {True: "volume can be reduced by pressure", False: "volume resists compression"},
        "flows": {True: "takes the shape of its container", False: "maintains its own shape"},
        "has_mass": {True: "affected by gravity"},
        "pulls_down": {True: "objects fall toward ground"},
        "opposes_motion": {True: "slows moving objects"},
        "generates_heat": {True: "produces thermal energy"},
        # ── Expanded inference rules for ARC science coverage ──
        "transparent": {True: "light passes through it", False: "light is blocked by it"},
        "magnetic": {True: "attracted to magnets and can be magnetized", False: "not attracted to magnets"},
        "reflects_light": {True: "light bounces off surface", False: "absorbs most light"},
        "dissolves_in_water": {True: "soluble in water", False: "insoluble in water"},
        "flexible": {True: "can be bent without breaking", False: "rigid and brittle"},
        "renewable": {True: "can be replaced naturally in a short time", False: "takes millions of years to form"},
        "living": {True: "grows, reproduces, and responds to stimuli", False: "does not grow or reproduce"},
        "absorbs_heat": {True: "temperature increases when exposed to heat"},
        "expands_when_heated": {True: "volume increases with temperature"},
        "contracts_when_cooled": {True: "volume decreases with temperature"},
        "insulates": {True: "prevents heat or electricity from flowing through"},
        "biodegradable": {True: "breaks down naturally by organisms"},
        "flammable": {True: "catches fire easily", False: "does not burn easily"},
        "floats_in_water": {True: "less dense than water, floats", False: "denser than water, sinks"},
        "evaporates": {True: "changes from liquid to gas at room temperature"},
        "freezes": {True: "changes from liquid to solid when cooled"},
        "melts": {True: "changes from solid to liquid when heated"},
        "corrodes": {True: "reacts with oxygen/moisture and deteriorates over time"},
        "produces_light": {True: "emits visible light"},
        "produces_sound": {True: "creates vibrations that travel as sound waves"},
        "stores_energy": {True: "can hold potential or chemical energy for later use"},
    }

    def __init__(self, ontology: ConceptOntology):
        self.ontology = ontology

    def infer_properties(self, concept_name: str) -> Dict[str, str]:
        """Infer observable consequences from concept properties."""
        concept = self.ontology.lookup(concept_name)
        if not concept:
            return {}

        inferences = {}
        for prop, value in concept.properties.items():
            if prop in self.INFERENCE_RULES:
                rule = self.INFERENCE_RULES[prop]
                if isinstance(value, bool) and value in rule:
                    inferences[prop] = rule[value]
                elif value in rule:
                    inferences[prop] = rule[value]

        return inferences

    def compare_properties(self, concept_a: str, concept_b: str) -> Dict[str, Any]:
        """Compare properties of two concepts."""
        a = self.ontology.lookup(concept_a)
        b = self.ontology.lookup(concept_b)
        if not a or not b:
            return {'error': 'concept not found'}

        shared = {}
        different = {}
        unique_a = {}
        unique_b = {}

        all_props = set(a.properties.keys()) | set(b.properties.keys())
        for prop in all_props:
            val_a = a.properties.get(prop)
            val_b = b.properties.get(prop)
            if val_a is not None and val_b is not None:
                if val_a == val_b:
                    shared[prop] = val_a
                else:
                    different[prop] = {'a': val_a, 'b': val_b}
            elif val_a is not None:
                unique_a[prop] = val_a
            elif val_b is not None:
                unique_b[prop] = val_b

        return {
            'shared': shared,
            'different': different,
            f'unique_{concept_a}': unique_a,
            f'unique_{concept_b}': unique_b,
        }

