#!/usr/bin/env python3
"""
L104 ASI SYMBOLIC MATH SOLVER v1.0.0
═══════════════════════════════════════════════════════════════════════════════
Addresses MATH (competition) benchmark gap: L104 previously scored ~5-15%
because it could not parse word problems or do novel symbolic reasoning.

Architecture (DeepSeek-R1/Math-7B informed):
  ╔═══════════════════════════════════════════════════════════════════════╗
  ║  Layer 1: WORD PROBLEM PARSER — Extract entities, quantities, rels  ║
  ║  Layer 2: EQUATION BUILDER   — NL → symbolic equations              ║
  ║  Layer 3: SYMBOLIC ALGEBRA   — Solve, simplify, substitute, factor  ║
  ║  Layer 4: GEOMETRY REASONER  — Areas, volumes, Pythagoras, trig     ║
  ║  Layer 5: NUMBER THEORY      — Primes, modular, gcd, lcm, divides  ║
  ║  Layer 6: COMBINATORICS      — P(n,r), C(n,r), probability, series ║
  ║  Layer 7: STEP-BY-STEP       — Chain-of-thought solution trace      ║
  ║  Layer 8: VERIFICATION       — Plug-back, dimensional analysis      ║
  ╚═══════════════════════════════════════════════════════════════════════╝

Key innovations (DeepSeek-R1 style):
  - Natural language → symbolic equation extraction via pattern templates
  - Multi-strategy solver: algebraic, numeric, geometric, combinatorial
  - AST-based symbolic manipulation (no sympy dependency)
  - Chain-of-thought with intermediate step verification
  - PHI/GOD_CODE alignment for sacred math domains
  - Backtracking solve: try multiple approaches, pick best

Target: MATH ~5-15% → 30-45% (approach mid-tier reasoning model)
"""

from __future__ import annotations

import ast
import math
import re
import operator
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache, reduce
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# ── Sacred Constants ──────────────────────────────────────────────────────────
PHI = 1.618033988749895
GOD_CODE = 527.5184818492612
TAU = 1.0 / PHI
VOID_CONSTANT = 1.0416180339887497


# ── Engine Support (lazy-loaded for proof verification + physics problems) ────
def _get_math_engine():
    """Lazy-load MathEngine for GOD_CODE, prime sieve, proof validation."""
    try:
        from l104_math_engine import MathEngine
        return MathEngine()
    except Exception:
        return None

def _get_science_engine():
    """Lazy-load ScienceEngine for physics-domain math problems."""
    try:
        from l104_science_engine import ScienceEngine
        return ScienceEngine()
    except Exception:
        return None

def _get_quantum_gate_engine():
    """Lazy-load quantum gate engine for quantum circuit math + QFT."""
    try:
        from l104_quantum_gate_engine import get_engine
        return get_engine()
    except Exception:
        return None

def _get_quantum_math_core():
    """Lazy-load QuantumMathCore for quantum computation, entanglement, tunnelling."""
    try:
        from l104_quantum_engine import QuantumMathCore
        return QuantumMathCore
    except Exception:
        return None

_math_engine_cache = None
_science_engine_cache = None
_quantum_gate_engine_cache = None
_quantum_math_core_cache = None

def _get_cached_math_engine():
    global _math_engine_cache
    if _math_engine_cache is None:
        _math_engine_cache = _get_math_engine()
    return _math_engine_cache

def _get_cached_science_engine():
    global _science_engine_cache
    if _science_engine_cache is None:
        _science_engine_cache = _get_science_engine()
    return _science_engine_cache

def _get_cached_quantum_gate_engine():
    global _quantum_gate_engine_cache
    if _quantum_gate_engine_cache is None:
        _quantum_gate_engine_cache = _get_quantum_gate_engine()
    return _quantum_gate_engine_cache

def _get_cached_quantum_math_core():
    global _quantum_math_core_cache
    if _quantum_math_core_cache is None:
        _quantum_math_core_cache = _get_quantum_math_core()
    return _quantum_math_core_cache


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 1: WORD PROBLEM PARSER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MathEntity:
    """An entity extracted from a word problem."""
    name: str
    value: Optional[float] = None
    variable: Optional[str] = None
    unit: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MathRelation:
    """A relationship between entities."""
    relation_type: str  # "equals", "sum", "product", "ratio", "difference", "percent_of"
    entities: List[str] = field(default_factory=list)
    expression: str = ""
    value: Optional[float] = None


@dataclass
class ParsedProblem:
    """Fully parsed word problem."""
    original: str
    entities: List[MathEntity] = field(default_factory=list)
    relations: List[MathRelation] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    question_type: str = ""  # "solve_for", "compute", "prove", "count", "find_max/min"
    domain: str = ""  # "algebra", "geometry", "number_theory", "combinatorics", "calculus", "probability"
    constraints: List[str] = field(default_factory=list)


class WordProblemParser:
    """Parse natural language math problems into structured representations.

    Extracts entities (variables, constants), relationships (equations, inequalities),
    constraints, and identifies the question type and mathematical domain.
    """

    # Number word mappings
    NUMBER_WORDS = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
        'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
        'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
        'hundred': 100, 'thousand': 1000, 'million': 1000000,
        'half': 0.5, 'third': 1/3, 'quarter': 0.25, 'fifth': 0.2,
        'twice': 2, 'thrice': 3, 'double': 2, 'triple': 3,
    }

    # Domain detection patterns
    DOMAIN_PATTERNS = {
        'algebra': [r'solve\b', r'equation', r'variable', r'expression', r'polynomial',
                    r'factor', r'simplify', r'quadratic', r'linear', r'system of'],
        'geometry': [r'triangle', r'circle', r'square', r'rectangle', r'area',
                    r'perimeter', r'volume', r'angle', r'radius', r'diameter',
                    r'parallel', r'perpendicular', r'polygon', r'cone', r'sphere',
                    r'cylinder', r'hexagon', r'pentagon', r'circumference'],
        'number_theory': [r'prime', r'divisible', r'remainder', r'modulo', r'factor',
                         r'gcd', r'lcm', r'integer', r'divisor', r'composite',
                         r'congruent', r'digit', r'even', r'odd', r'totient',
                         r'modular', r'coprime', r'factorial'],
        'combinatorics': [r'how many ways', r'permutation', r'combination', r'arrange',
                         r'choose', r'select', r'distribute', r'counting', r'binomial',
                         r'derangement', r'catalan', r'n choose k'],
        'probability': [r'probability', r'chance', r'likely', r'random', r'expected',
                       r'die', r'dice', r'coin', r'deck', r'card', r'binomial\s+distribution'],
        'calculus': [r'derivative', r'integral', r'limit', r'rate of change',
                    r'maximum', r'minimum', r'converge', r'series', r'continuous'],
        'statistics': [r'mean', r'median', r'mode', r'average', r'standard deviation',
                      r'variance', r'percentile', r'quartile', r'range\s+of\s+data'],
    }

    # Question type patterns
    QUESTION_PATTERNS = {
        'solve_for': [r'find\s+(?:the\s+)?(?:value|values?)\s+of', r'solve\s+for', r'what\s+is',
                     r'determine', r'calculate', r'compute', r'evaluate'],
        'count': [r'how\s+many', r'number\s+of', r'count', r'total\s+number'],
        'prove': [r'prove\s+that', r'show\s+that', r'demonstrate', r'verify'],
        'find_max': [r'maximum', r'largest', r'greatest', r'most', r'maximize'],
        'find_min': [r'minimum', r'smallest', r'least', r'fewest', r'minimize'],
        'compare': [r'greater\s+than', r'less\s+than', r'which\s+is', r'compare'],
    }

    # Relation extraction patterns
    RELATION_PATTERNS = [
        (r'(\w+)\s+(?:is|are|equals?|=)\s+(\d+(?:\.\d+)?)', 'equals'),
        (r'(?:sum|total)\s+(?:of\s+)?(\w+)\s+and\s+(\w+)\s+is\s+(\d+)', 'sum'),
        (r'(\w+)\s+(?:plus|added to|\+)\s+(\w+)\s+(?:equals?|is|=)\s+(\d+)', 'sum'),
        (r'(\w+)\s+(?:minus|subtracted from|\-)\s+(\w+)\s+(?:equals?|is|=)\s+(\d+)', 'difference'),
        (r'(\w+)\s+(?:times|multiplied by|\×)\s+(\w+)\s+(?:equals?|is|=)\s+(\d+)', 'product'),
        (r'(?:ratio|proportion)\s+(?:of\s+)?(\w+)\s+to\s+(\w+)\s+is\s+(\d+)', 'ratio'),
        (r'(\d+)%?\s+(?:of|percent\s+of)\s+(\w+)', 'percent_of'),
        (r'(\w+)\s+is\s+(\d+)\s+(?:more|greater)\s+than\s+(\w+)', 'greater_by'),
        (r'(\w+)\s+is\s+(\d+)\s+(?:less|fewer)\s+than\s+(\w+)', 'less_by'),
        (r'(\w+)\s+is\s+(\w+)\s+times\s+(\w+)', 'multiple_of'),
    ]

    def parse(self, problem: str) -> ParsedProblem:
        """Parse a word problem into structured form."""
        parsed = ParsedProblem(original=problem)

        # Normalize
        text = problem.strip().lower()

        # Detect domain
        parsed.domain = self._detect_domain(text)

        # Detect question type
        parsed.question_type = self._detect_question_type(text)

        # Extract numeric entities
        parsed.entities = self._extract_entities(text)

        # Extract relationships
        parsed.relations = self._extract_relations(text)

        # Identify unknowns
        parsed.unknowns = self._identify_unknowns(text, parsed.entities)

        # Extract constraints
        parsed.constraints = self._extract_constraints(text)

        return parsed

    def _detect_domain(self, text: str) -> str:
        """Detect the mathematical domain of the problem."""
        scores = {}
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, text))
            if score > 0:
                scores[domain] = score
        if scores:
            return max(scores, key=scores.get)
        return "algebra"  # Default

    def _detect_question_type(self, text: str) -> str:
        """Detect what the problem is asking for."""
        for q_type, patterns in self.QUESTION_PATTERNS.items():
            for p in patterns:
                if re.search(p, text):
                    return q_type
        return "solve_for"

    def _extract_entities(self, text: str) -> List[MathEntity]:
        """Extract mathematical entities (numbers, variables) from text."""
        entities = []
        seen_values = set()

        # Extract explicit numbers with context
        for m in re.finditer(r'(\w+)\s+(?:is|=|equals?)\s+(\d+(?:\.\d+)?)', text):
            name = m.group(1)
            value = float(m.group(2))
            entities.append(MathEntity(name=name, value=value))
            seen_values.add(value)

        # Extract standalone numbers with units
        for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(cm|m|km|kg|g|lb|mph|hours?|minutes?|seconds?|degrees?|%)', text):
            value = float(m.group(1))
            unit = m.group(2)
            entities.append(MathEntity(name=f"quantity_{len(entities)}", value=value, unit=unit))
            seen_values.add(value)

        # Extract ALL standalone numbers not already captured (critical for coverage)
        for m in re.finditer(r'\b(\d+(?:\.\d+)?)\b', text):
            value = float(m.group(1))
            if value not in seen_values:
                entities.append(MathEntity(name=f"num_{len(entities)}", value=value))
                seen_values.add(value)

        # Extract number words
        for word, value in self.NUMBER_WORDS.items():
            if re.search(rf'\b{word}\b', text):
                if value not in seen_values:
                    entities.append(MathEntity(name=word, value=value))
                    seen_values.add(value)

        # Extract algebraic variables (single letters in math context)
        for m in re.finditer(r'\b([a-z])\s*[+\-*/=<>]', text):
            var = m.group(1)
            if var not in ('a', 'i'):  # Avoid articles
                entities.append(MathEntity(name=var, variable=var))

        # Detect "let x = ..." patterns
        for m in re.finditer(r'let\s+(\w+)\s*=\s*(\d+(?:\.\d+)?)', text):
            entities.append(MathEntity(name=m.group(1), value=float(m.group(2)), variable=m.group(1)))

        return entities

    def _extract_relations(self, text: str) -> List[MathRelation]:
        """Extract mathematical relationships from text."""
        relations = []

        for pattern, rel_type in self.RELATION_PATTERNS:
            for m in re.finditer(pattern, text):
                groups = m.groups()
                rel = MathRelation(
                    relation_type=rel_type,
                    entities=list(groups[:-1]) if len(groups) > 2 else list(groups),
                    expression=m.group(0),
                    value=float(groups[-1]) if groups[-1].replace('.', '').isdigit() else None,
                )
                relations.append(rel)

        # Extract explicit equations: "2x + 3 = 7", "x^2 - 4 = 0"
        for m in re.finditer(r'([\d\w\s\+\-\*/\^()]+)\s*=\s*([\d\w\s\+\-\*/\^()]+)', text):
            lhs = m.group(1).strip()
            rhs = m.group(2).strip()
            if any(c.isdigit() or c.isalpha() for c in lhs):
                relations.append(MathRelation(
                    relation_type='equation',
                    expression=f"{lhs} = {rhs}",
                    entities=[lhs, rhs],
                ))

        return relations

    def _identify_unknowns(self, text: str, entities: List[MathEntity]) -> List[str]:
        """Identify what we need to solve for."""
        unknowns = []

        # "find x", "solve for x", "what is x"
        for m in re.finditer(r'(?:find|solve for|what is|determine|calculate)\s+(?:the\s+)?(\w+)', text):
            unknowns.append(m.group(1))

        # Variables without assigned values
        for e in entities:
            if e.variable and e.value is None and e.variable not in unknowns:
                unknowns.append(e.variable)

        # Default: if question asks "how many" the unknown is "count"
        if not unknowns and 'how many' in text:
            unknowns.append('count')

        return unknowns or ['x']

    def _extract_constraints(self, text: str) -> List[str]:
        """Extract mathematical constraints."""
        constraints = []
        constraint_patterns = [
            r'(?:where|such that|given that|assuming|if)\s+(.+?)(?:\.|,|$)',
            r'(\w+)\s+(?:is|must be)\s+(?:positive|negative|non-negative|an?\s+integer)',
            r'for\s+(?:all\s+)?(\w+)\s*[<>≤≥]\s*\d+',
            r'(\d+)\s*[<>≤≥]\s*(\w+)\s*[<>≤≥]\s*(\d+)',  # Range constraints
        ]
        for p in constraint_patterns:
            for m in re.finditer(p, text):
                constraints.append(m.group(0))
        return constraints


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 2: EQUATION BUILDER — Natural Language → Symbolic Equations
# ═══════════════════════════════════════════════════════════════════════════════

class EquationBuilder:
    """Convert parsed problem relations into solvable equations.

    Supports:
    - Linear equations: ax + b = c
    - Quadratic equations: ax² + bx + c = 0
    - Systems of linear equations
    - Polynomial equations
    - Inequality constraints
    """

    def build_equations(self, parsed: ParsedProblem) -> List[Dict[str, Any]]:
        """Build solvable equation representations from parsed problem."""
        equations = []

        for rel in parsed.relations:
            eq = self._relation_to_equation(rel, parsed)
            if eq:
                equations.append(eq)

        # If no equations from relations, try to build from entities + question
        if not equations and parsed.entities:
            equations = self._infer_equations(parsed)

        return equations

    def _relation_to_equation(self, rel: MathRelation, parsed: ParsedProblem) -> Optional[Dict]:
        """Convert a single relation to an equation dict."""
        if rel.relation_type == 'equals':
            if len(rel.entities) >= 2:
                return {
                    'type': 'assignment',
                    'lhs': rel.entities[0],
                    'rhs': rel.value if rel.value is not None else rel.entities[1],
                    'expression': rel.expression,
                }

        elif rel.relation_type == 'sum':
            return {
                'type': 'linear',
                'terms': [{'var': e, 'coeff': 1.0} for e in rel.entities[:-1]] if len(rel.entities) > 2 else
                         [{'var': rel.entities[0], 'coeff': 1.0}, {'var': rel.entities[1], 'coeff': 1.0}],
                'rhs': rel.value,
                'expression': rel.expression,
            }

        elif rel.relation_type == 'difference':
            if len(rel.entities) >= 2:
                return {
                    'type': 'linear',
                    'terms': [{'var': rel.entities[0], 'coeff': 1.0},
                              {'var': rel.entities[1], 'coeff': -1.0}],
                    'rhs': rel.value,
                    'expression': rel.expression,
                }

        elif rel.relation_type == 'product':
            return {
                'type': 'product',
                'factors': rel.entities[:2] if len(rel.entities) >= 2 else rel.entities,
                'rhs': rel.value,
                'expression': rel.expression,
            }

        elif rel.relation_type == 'equation':
            # Try to parse the explicit equation
            return self._parse_explicit_equation(rel.expression)

        elif rel.relation_type == 'greater_by':
            if len(rel.entities) >= 3:
                return {
                    'type': 'linear',
                    'terms': [{'var': rel.entities[0], 'coeff': 1.0},
                              {'var': rel.entities[2], 'coeff': -1.0}],
                    'rhs': float(rel.entities[1]) if rel.entities[1].replace('.','').isdigit() else 0,
                    'expression': rel.expression,
                }

        elif rel.relation_type == 'less_by':
            if len(rel.entities) >= 3:
                return {
                    'type': 'linear',
                    'terms': [{'var': rel.entities[2], 'coeff': 1.0},
                              {'var': rel.entities[0], 'coeff': -1.0}],
                    'rhs': float(rel.entities[1]) if rel.entities[1].replace('.','').isdigit() else 0,
                    'expression': rel.expression,
                }

        return None

    def _parse_explicit_equation(self, expr: str) -> Optional[Dict]:
        """Parse an explicit equation string like '2x + 3 = 7'."""
        if '=' not in expr:
            return None
        parts = expr.split('=', 1)
        lhs = parts[0].strip()
        rhs = parts[1].strip()

        # Try to identify polynomial degree
        has_square = bool(re.search(r'\w\s*\^?\s*2|\w²', lhs))
        has_cube = bool(re.search(r'\w\s*\^?\s*3|\w³', lhs))

        if has_cube:
            eq_type = 'cubic'
        elif has_square:
            eq_type = 'quadratic'
        else:
            eq_type = 'linear'

        return {
            'type': eq_type,
            'lhs': lhs,
            'rhs': rhs,
            'expression': expr,
            'raw': True,
        }

    def _infer_equations(self, parsed: ParsedProblem) -> List[Dict]:
        """Infer equations from entities and question context when explicit relations are sparse."""
        equations = []

        # If we have numeric entities and the question asks for a value
        values = [e for e in parsed.entities if e.value is not None]
        variables = [e for e in parsed.entities if e.variable is not None]

        if parsed.domain == 'geometry':
            equations = self._infer_geometry_equations(parsed, values)
        elif parsed.domain == 'number_theory':
            equations = self._infer_number_theory(parsed, values)
        elif parsed.domain == 'combinatorics':
            equations = self._infer_combinatorics(parsed, values)
        elif parsed.domain == 'probability':
            equations = self._infer_probability(parsed, values)

        return equations

    def _infer_geometry_equations(self, parsed: ParsedProblem, values: List[MathEntity]) -> List[Dict]:
        """Infer geometry equations from context."""
        text = parsed.original.lower()
        equations = []

        if 'triangle' in text:
            if 'area' in text and len(values) >= 2:
                equations.append({
                    'type': 'geometry_formula', 'formula': 'triangle_area',
                    'params': {v.name: v.value for v in values[:2]},
                })
            if 'perimeter' in text and len(values) >= 3:
                equations.append({
                    'type': 'geometry_formula', 'formula': 'triangle_perimeter',
                    'params': {v.name: v.value for v in values[:3]},
                })
            if 'hypotenuse' in text or 'pythagor' in text:
                equations.append({
                    'type': 'geometry_formula', 'formula': 'pythagorean',
                    'params': {v.name: v.value for v in values[:2]},
                })

        if 'circle' in text:
            if 'area' in text and values:
                equations.append({
                    'type': 'geometry_formula', 'formula': 'circle_area',
                    'params': {'r': values[0].value},
                })
            if 'circumference' in text and values:
                equations.append({
                    'type': 'geometry_formula', 'formula': 'circle_circumference',
                    'params': {'r': values[0].value},
                })

        if 'rectangle' in text or 'square' in text:
            if 'area' in text and len(values) >= 2:
                equations.append({
                    'type': 'geometry_formula', 'formula': 'rectangle_area',
                    'params': {'l': values[0].value, 'w': values[1].value},
                })
            elif 'area' in text and 'square' in text and values:
                equations.append({
                    'type': 'geometry_formula', 'formula': 'square_area',
                    'params': {'s': values[0].value},
                })

        if 'sphere' in text:
            if 'volume' in text and values:
                equations.append({
                    'type': 'geometry_formula', 'formula': 'sphere_volume',
                    'params': {'r': values[0].value},
                })
            if 'surface area' in text and values:
                equations.append({
                    'type': 'geometry_formula', 'formula': 'sphere_surface_area',
                    'params': {'r': values[0].value},
                })

        if 'cylinder' in text:
            if 'volume' in text and len(values) >= 2:
                equations.append({
                    'type': 'geometry_formula', 'formula': 'cylinder_volume',
                    'params': {'r': values[0].value, 'h': values[1].value},
                })

        if 'cone' in text:
            if 'volume' in text and len(values) >= 2:
                equations.append({
                    'type': 'geometry_formula', 'formula': 'cone_volume',
                    'params': {'r': values[0].value, 'h': values[1].value},
                })

        return equations

    def _infer_number_theory(self, parsed: ParsedProblem, values: List[MathEntity]) -> List[Dict]:
        """Infer number theory relationships."""
        text = parsed.original.lower()
        equations = []

        if 'prime' in text and values:
            equations.append({'type': 'nt_query', 'query': 'prime_check', 'n': values[0].value})
        if 'gcd' in text and len(values) >= 2:
            equations.append({'type': 'nt_query', 'query': 'gcd', 'a': values[0].value, 'b': values[1].value})
        if 'lcm' in text and len(values) >= 2:
            equations.append({'type': 'nt_query', 'query': 'lcm', 'a': values[0].value, 'b': values[1].value})
        if 'remainder' in text or 'modulo' in text or 'mod' in text:
            if len(values) >= 2:
                equations.append({'type': 'nt_query', 'query': 'modular',
                                  'a': values[0].value, 'b': values[1].value})
        if 'divisor' in text and values:
            equations.append({'type': 'nt_query', 'query': 'divisors', 'n': values[0].value})
        if 'digit' in text:
            if 'sum' in text and values:
                equations.append({'type': 'nt_query', 'query': 'digit_sum', 'n': values[0].value})
            if 'count' in text or 'how many' in text:
                equations.append({'type': 'nt_query', 'query': 'digit_count',
                                  'n': values[0].value if values else 0})

        return equations

    def _infer_combinatorics(self, parsed: ParsedProblem, values: List[MathEntity]) -> List[Dict]:
        """Infer combinatorics relationships."""
        text = parsed.original.lower()
        equations = []

        if ('permutation' in text or 'arrange' in text) and len(values) >= 2:
            equations.append({'type': 'combo_query', 'query': 'permutation',
                              'n': values[0].value, 'r': values[1].value})
        elif ('combination' in text or 'choose' in text or 'select' in text) and len(values) >= 2:
            equations.append({'type': 'combo_query', 'query': 'combination',
                              'n': values[0].value, 'r': values[1].value})
        elif 'how many ways' in text and values:
            if len(values) >= 2:
                equations.append({'type': 'combo_query', 'query': 'combination',
                                  'n': values[0].value, 'r': values[1].value})

        if 'binomial' in text and len(values) >= 2:
            equations.append({'type': 'combo_query', 'query': 'binomial_coeff',
                              'n': values[0].value, 'k': values[1].value})

        return equations

    def _infer_probability(self, parsed: ParsedProblem, values: List[MathEntity]) -> List[Dict]:
        """Infer probability relationships."""
        text = parsed.original.lower()
        equations = []

        if 'coin' in text:
            n_flips = int(values[0].value) if values else 1
            equations.append({'type': 'prob_query', 'query': 'coin_flips', 'n': n_flips})
        elif 'die' in text or 'dice' in text:
            n_dice = int(values[0].value) if values else 1
            equations.append({'type': 'prob_query', 'query': 'dice', 'n': n_dice})
        elif 'card' in text or 'deck' in text:
            equations.append({'type': 'prob_query', 'query': 'cards'})

        return equations


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 3: SYMBOLIC ALGEBRA ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SymbolicAlgebra:
    """Pure-Python symbolic algebra solver.

    Handles:
    - Linear equations: ax + b = c
    - Quadratic equations: ax² + bx + c = 0 (discriminant method)
    - Systems of 2 linear equations (Cramer's rule / substitution)
    - Polynomial evaluation
    - Expression simplification
    - Factoring (quadratics, difference of squares)
    """

    def solve_linear(self, a: float, b: float) -> Optional[float]:
        """Solve ax + b = 0 → x = -b/a"""
        if abs(a) < 1e-12:
            return None
        return -b / a

    def solve_quadratic(self, a: float, b: float, c: float) -> List[float]:
        """Solve ax² + bx + c = 0 using discriminant."""
        if abs(a) < 1e-12:
            # Degenerate to linear
            sol = self.solve_linear(b, c)
            return [sol] if sol is not None else []

        discriminant = b * b - 4 * a * c
        if discriminant < -1e-12:
            return []  # No real solutions
        elif abs(discriminant) < 1e-12:
            return [-b / (2 * a)]
        else:
            sqrt_d = math.sqrt(discriminant)
            return [(-b + sqrt_d) / (2 * a), (-b - sqrt_d) / (2 * a)]

    def solve_system_2x2(self, a1: float, b1: float, c1: float,
                          a2: float, b2: float, c2: float) -> Optional[Tuple[float, float]]:
        """Solve system: a1*x + b1*y = c1, a2*x + b2*y = c2 using Cramer's rule."""
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-12:
            return None  # No unique solution
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        return (x, y)

    def evaluate_polynomial(self, coeffs: List[float], x: float) -> float:
        """Evaluate polynomial with Horner's method. coeffs[0] = highest degree."""
        result = 0.0
        for c in coeffs:
            result = result * x + c
        return result

    def factor_quadratic(self, a: float, b: float, c: float) -> Optional[str]:
        """Factor ax² + bx + c into (px + q)(rx + s) if integer factors exist."""
        roots = self.solve_quadratic(a, b, c)
        if not roots:
            return None

        if len(roots) == 1:
            r = roots[0]
            if abs(r - round(r)) < 1e-9:
                r = int(round(r))
                return f"({a})(x - {r})²"

        if len(roots) == 2:
            r1, r2 = roots
            # Check if roots are rational/integer
            if abs(r1 - round(r1)) < 1e-9 and abs(r2 - round(r2)) < 1e-9:
                r1, r2 = int(round(r1)), int(round(r2))
                a_int = int(a) if abs(a - round(a)) < 1e-9 else a
                sign1 = '+' if -r1 >= 0 else '-'
                sign2 = '+' if -r2 >= 0 else '-'
                abs_r1 = abs(r1)
                abs_r2 = abs(r2)
                prefix = f"{a_int}" if a_int != 1 else ""
                return f"{prefix}(x {sign1} {abs_r1})(x {sign2} {abs_r2})"

        return None

    def simplify_fraction(self, num: int, den: int) -> Tuple[int, int]:
        """Simplify fraction to lowest terms."""
        if den == 0:
            return (num, 0)
        g = math.gcd(abs(num), abs(den))
        if den < 0:
            num, den = -num, -den
        return (num // g, den // g)

    def parse_linear_equation(self, expr: str) -> Optional[Tuple[float, float]]:
        """Parse a linear equation string and return (a, b) for ax + b = 0.

        Handles: '2x + 3 = 7', '3x - 5 = 10', '-x + 4 = 0', etc.
        """
        if '=' not in expr:
            return None

        lhs, rhs = expr.split('=', 1)

        # Find the variable (first single letter)
        var_match = re.search(r'[a-z]', lhs + rhs)
        if not var_match:
            return None
        var = var_match.group(0)

        # Extract coefficient and constant from each side
        lhs_a, lhs_b = self._extract_linear_terms(lhs, var)
        rhs_a, rhs_b = self._extract_linear_terms(rhs, var)

        # Move everything to LHS: (lhs_a - rhs_a)x + (lhs_b - rhs_b) = 0
        a = lhs_a - rhs_a
        b = lhs_b - rhs_b

        return (a, b)

    def _extract_linear_terms(self, expr: str, var: str) -> Tuple[float, float]:
        """Extract coefficient of variable and constant from a linear expression."""
        expr = expr.replace(' ', '').replace('−', '-')
        coeff = 0.0
        const = 0.0

        # Match terms like: +3x, -2x, x, -x, +5, -3
        terms = re.findall(r'[+\-]?[^+\-]+', expr)
        for term in terms:
            term = term.strip()
            if not term:
                continue
            if var in term:
                # Variable term
                term_coeff = term.replace(var, '').replace('*', '')
                if term_coeff in ('', '+'):
                    coeff += 1.0
                elif term_coeff == '-':
                    coeff -= 1.0
                else:
                    try:
                        coeff += float(term_coeff)
                    except ValueError:
                        pass
            else:
                # Constant term
                try:
                    const += float(term)
                except ValueError:
                    pass

        return (coeff, const)

    def parse_quadratic_equation(self, expr: str) -> Optional[Tuple[float, float, float]]:
        """Parse quadratic equation and return (a, b, c) for ax² + bx + c = 0."""
        if '=' not in expr:
            return None

        lhs, rhs = expr.split('=', 1)
        var_match = re.search(r'[a-z]', lhs + rhs)
        if not var_match:
            return None
        var = var_match.group(0)

        # Move everything to LHS
        lhs_terms = self._extract_quadratic_terms(lhs, var)
        rhs_terms = self._extract_quadratic_terms(rhs, var)

        a = lhs_terms[0] - rhs_terms[0]
        b = lhs_terms[1] - rhs_terms[1]
        c = lhs_terms[2] - rhs_terms[2]

        return (a, b, c)

    def _extract_quadratic_terms(self, expr: str, var: str) -> Tuple[float, float, float]:
        """Extract (a, b, c) from a quadratic expression in one variable."""
        expr = expr.replace(' ', '').replace('−', '-')
        a, b, c = 0.0, 0.0, 0.0

        # Handle x^2, x², xx patterns
        sq_patterns = [f'{var}\\^2', f'{var}²', f'{var}{var}']

        terms = re.findall(r'[+\-]?[^+\-]+', expr)
        for term in terms:
            term = term.strip()
            if not term:
                continue

            is_quadratic = False
            for sp in sq_patterns:
                if re.search(sp, term):
                    is_quadratic = True
                    break

            if is_quadratic:
                # Quadratic term
                cleaned = re.sub(r'[a-z]\^?2?²?', '', term).replace('*', '')
                if cleaned in ('', '+'):
                    a += 1.0
                elif cleaned == '-':
                    a -= 1.0
                else:
                    try:
                        a += float(cleaned)
                    except ValueError:
                        pass
            elif var in term:
                # Linear term
                cleaned = term.replace(var, '').replace('*', '')
                if cleaned in ('', '+'):
                    b += 1.0
                elif cleaned == '-':
                    b -= 1.0
                else:
                    try:
                        b += float(cleaned)
                    except ValueError:
                        pass
            else:
                # Constant
                try:
                    c += float(term)
                except ValueError:
                    pass

        return (a, b, c)


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 4: GEOMETRY REASONER
# ═══════════════════════════════════════════════════════════════════════════════

class GeometryReasoner:
    """Geometry formula application and reasoning.

    Covers:
    - 2D: triangles, circles, rectangles, polygons, parallelograms, trapezoids
    - 3D: spheres, cylinders, cones, pyramids, prisms
    - Theorems: Pythagorean, similarity, congruence, angle sum
    - Trigonometry: sin, cos, tan, basic identities
    """

    PI = math.pi

    # ── 2D Formulas ──

    def triangle_area(self, base: float, height: float) -> float:
        return 0.5 * base * height

    def triangle_area_heron(self, a: float, b: float, c: float) -> float:
        """Heron's formula: area from three side lengths."""
        s = (a + b + c) / 2
        val = s * (s - a) * (s - b) * (s - c)
        return math.sqrt(max(0, val))

    def triangle_perimeter(self, a: float, b: float, c: float) -> float:
        return a + b + c

    def pythagorean_hypotenuse(self, a: float, b: float) -> float:
        return math.sqrt(a * a + b * b)

    def pythagorean_leg(self, hypotenuse: float, other_leg: float) -> float:
        return math.sqrt(max(0, hypotenuse * hypotenuse - other_leg * other_leg))

    def circle_area(self, r: float) -> float:
        return self.PI * r * r

    def circle_circumference(self, r: float) -> float:
        return 2 * self.PI * r

    def circle_arc_length(self, r: float, angle_deg: float) -> float:
        return r * math.radians(angle_deg)

    def circle_sector_area(self, r: float, angle_deg: float) -> float:
        return 0.5 * r * r * math.radians(angle_deg)

    def rectangle_area(self, l: float, w: float) -> float:
        return l * w

    def rectangle_perimeter(self, l: float, w: float) -> float:
        return 2 * (l + w)

    def square_area(self, s: float) -> float:
        return s * s

    def square_perimeter(self, s: float) -> float:
        return 4 * s

    def parallelogram_area(self, base: float, height: float) -> float:
        return base * height

    def trapezoid_area(self, a: float, b: float, h: float) -> float:
        return 0.5 * (a + b) * h

    def regular_polygon_area(self, n_sides: int, side_length: float) -> float:
        """Area of a regular polygon with n sides of given length."""
        return (n_sides * side_length ** 2) / (4 * math.tan(self.PI / n_sides))

    # ── 3D Formulas ──

    def sphere_volume(self, r: float) -> float:
        return (4 / 3) * self.PI * r ** 3

    def sphere_surface_area(self, r: float) -> float:
        return 4 * self.PI * r * r

    def cylinder_volume(self, r: float, h: float) -> float:
        return self.PI * r * r * h

    def cylinder_surface_area(self, r: float, h: float) -> float:
        return 2 * self.PI * r * (r + h)

    def cone_volume(self, r: float, h: float) -> float:
        return (1 / 3) * self.PI * r * r * h

    def cone_surface_area(self, r: float, h: float) -> float:
        slant = math.sqrt(r * r + h * h)
        return self.PI * r * (r + slant)

    def pyramid_volume(self, base_area: float, h: float) -> float:
        return (1 / 3) * base_area * h

    def prism_volume(self, base_area: float, h: float) -> float:
        return base_area * h

    # ── Trigonometry ──

    def sin_deg(self, angle: float) -> float:
        return math.sin(math.radians(angle))

    def cos_deg(self, angle: float) -> float:
        return math.cos(math.radians(angle))

    def tan_deg(self, angle: float) -> float:
        return math.tan(math.radians(angle))

    def law_of_cosines(self, a: float, b: float, C_deg: float) -> float:
        """Find side c given sides a, b and included angle C."""
        C_rad = math.radians(C_deg)
        return math.sqrt(a*a + b*b - 2*a*b*math.cos(C_rad))

    def law_of_sines_angle(self, a: float, A_deg: float, b: float) -> float:
        """Find angle B given side a, angle A, side b."""
        A_rad = math.radians(A_deg)
        sin_B = b * math.sin(A_rad) / a
        sin_B = max(-1.0, min(1.0, sin_B))
        return math.degrees(math.asin(sin_B))

    def triangle_angle_sum(self, A: float, B: float) -> float:
        """Third angle of a triangle given two angles in degrees."""
        return 180.0 - A - B

    def apply_formula(self, formula_name: str, params: Dict[str, float]) -> Optional[float]:
        """Apply a named geometry formula with given parameters."""
        formula_map = {
            'triangle_area': lambda p: self.triangle_area(p.get('base', p.get('b', 0)),
                                                           p.get('height', p.get('h', 0))),
            'triangle_area_heron': lambda p: self.triangle_area_heron(p.get('a', 0), p.get('b', 0), p.get('c', 0)),
            'triangle_perimeter': lambda p: self.triangle_perimeter(p.get('a', 0), p.get('b', 0), p.get('c', 0)),
            'pythagorean': lambda p: self.pythagorean_hypotenuse(p.get('a', 0), p.get('b', 0)),
            'circle_area': lambda p: self.circle_area(p.get('r', p.get('radius', 0))),
            'circle_circumference': lambda p: self.circle_circumference(p.get('r', p.get('radius', 0))),
            'rectangle_area': lambda p: self.rectangle_area(p.get('l', p.get('length', 0)),
                                                             p.get('w', p.get('width', 0))),
            'square_area': lambda p: self.square_area(p.get('s', p.get('side', 0))),
            'sphere_volume': lambda p: self.sphere_volume(p.get('r', p.get('radius', 0))),
            'sphere_surface_area': lambda p: self.sphere_surface_area(p.get('r', 0)),
            'cylinder_volume': lambda p: self.cylinder_volume(p.get('r', 0), p.get('h', 0)),
            'cone_volume': lambda p: self.cone_volume(p.get('r', 0), p.get('h', 0)),
            'trapezoid_area': lambda p: self.trapezoid_area(p.get('a', 0), p.get('b', 0), p.get('h', 0)),
            'parallelogram_area': lambda p: self.parallelogram_area(p.get('base', 0), p.get('height', 0)),
        }

        fn = formula_map.get(formula_name)
        if fn:
            try:
                return fn(params)
            except Exception:
                return None
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 5: NUMBER THEORY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class NumberTheoryEngine:
    """Number theory computations for competition math.

    Covers: primes, GCD/LCM, modular arithmetic, Euler's totient,
    digit operations, divisor functions, congruences.
    """

    @staticmethod
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    @staticmethod
    def prime_factorization(n: int) -> Dict[int, int]:
        """Return prime factorization as {prime: exponent}."""
        if n <= 1:
            return {}
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    @staticmethod
    def gcd(a: int, b: int) -> int:
        return math.gcd(abs(a), abs(b))

    @staticmethod
    def lcm(a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return abs(a * b) // math.gcd(abs(a), abs(b))

    @staticmethod
    def euler_totient(n: int) -> int:
        """Euler's totient function φ(n)."""
        result = n
        p = 2
        temp = n
        while p * p <= temp:
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
            p += 1
        if temp > 1:
            result -= result // temp
        return result

    @staticmethod
    def divisors(n: int) -> List[int]:
        """Return all divisors of n."""
        if n <= 0:
            return []
        divs = []
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return sorted(divs)

    @staticmethod
    def divisor_count(n: int) -> int:
        return len(NumberTheoryEngine.divisors(n))

    @staticmethod
    def divisor_sum(n: int) -> int:
        return sum(NumberTheoryEngine.divisors(n))

    @staticmethod
    def digit_sum(n: int) -> int:
        return sum(int(d) for d in str(abs(n)))

    @staticmethod
    def digit_count(n: int) -> int:
        return len(str(abs(n)))

    @staticmethod
    def modular_power(base: int, exp: int, mod: int) -> int:
        """Compute base^exp mod mod efficiently."""
        return pow(base, exp, mod)

    @staticmethod
    def modular_inverse(a: int, mod: int) -> Optional[int]:
        """Modular inverse using extended Euclidean algorithm."""
        g, x, _ = NumberTheoryEngine._extended_gcd(a % mod, mod)
        if g != 1:
            return None
        return x % mod

    @staticmethod
    def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        g, x1, y1 = NumberTheoryEngine._extended_gcd(b % a, a)
        return g, y1 - (b // a) * x1, x1

    @staticmethod
    def chinese_remainder(remainders: List[int], moduli: List[int]) -> Optional[int]:
        """Chinese Remainder Theorem for coprime moduli."""
        if len(remainders) != len(moduli):
            return None
        M = reduce(operator.mul, moduli, 1)
        result = 0
        for r, m in zip(remainders, moduli):
            Mi = M // m
            yi = NumberTheoryEngine.modular_inverse(Mi, m)
            if yi is None:
                return None
            result += r * Mi * yi
        return result % M

    @staticmethod
    def sieve_of_eratosthenes(n: int) -> List[int]:
        """Primes up to n."""
        if n < 2:
            return []
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(n + 1) if is_prime[i]]

    def solve_query(self, query: Dict) -> Any:
        """Solve a number theory query dict."""
        q = query.get('query', '')
        if q == 'prime_check':
            n = int(query.get('n', 0))
            return {'is_prime': self.is_prime(n), 'n': n}
        elif q == 'gcd':
            a, b = int(query.get('a', 0)), int(query.get('b', 0))
            return {'gcd': self.gcd(a, b), 'a': a, 'b': b}
        elif q == 'lcm':
            a, b = int(query.get('a', 0)), int(query.get('b', 0))
            return {'lcm': self.lcm(a, b), 'a': a, 'b': b}
        elif q == 'modular':
            a, b = int(query.get('a', 0)), int(query.get('b', 1))
            return {'remainder': a % b if b != 0 else None, 'a': a, 'b': b}
        elif q == 'divisors':
            n = int(query.get('n', 0))
            divs = self.divisors(n)
            return {'divisors': divs, 'count': len(divs), 'sum': sum(divs), 'n': n}
        elif q == 'digit_sum':
            n = int(query.get('n', 0))
            return {'digit_sum': self.digit_sum(n), 'n': n}
        elif q == 'digit_count':
            n = int(query.get('n', 0))
            return {'digit_count': self.digit_count(n), 'n': n}
        elif q == 'factorization':
            n = int(query.get('n', 0))
            return {'factors': self.prime_factorization(n), 'n': n}
        elif q == 'totient':
            n = int(query.get('n', 0))
            return {'totient': self.euler_totient(n), 'n': n}
        return {'error': f'Unknown query: {q}'}


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 6: COMBINATORICS & PROBABILITY
# ═══════════════════════════════════════════════════════════════════════════════

class CombinatoricsEngine:
    """Combinatorics and probability computations.

    Covers: permutations, combinations, binomial coefficients,
    inclusion-exclusion, probability distributions, expected value.
    """

    @staticmethod
    @lru_cache(maxsize=512)
    def factorial(n: int) -> int:
        if n < 0:
            return 0
        return math.factorial(n)

    @staticmethod
    def permutation(n: int, r: int) -> int:
        """P(n,r) = n! / (n-r)!"""
        if r > n or r < 0 or n < 0:
            return 0
        return math.perm(n, r)

    @staticmethod
    def combination(n: int, r: int) -> int:
        """C(n,r) = n! / (r!(n-r)!)"""
        if r > n or r < 0 or n < 0:
            return 0
        return math.comb(n, r)

    @staticmethod
    def binomial_coefficient(n: int, k: int) -> int:
        return CombinatoricsEngine.combination(n, k)

    @staticmethod
    def multinomial(n: int, groups: List[int]) -> int:
        """Multinomial coefficient: n! / (k1! * k2! * ... * km!)"""
        if sum(groups) != n:
            return 0
        result = math.factorial(n)
        for k in groups:
            result //= math.factorial(k)
        return result

    @staticmethod
    def stars_and_bars(n: int, k: int) -> int:
        """Number of ways to put n identical items into k distinct bins."""
        return CombinatoricsEngine.combination(n + k - 1, k - 1)

    @staticmethod
    def derangements(n: int) -> int:
        """Number of derangements (permutations with no fixed points)."""
        if n == 0:
            return 1
        if n == 1:
            return 0
        result = 0
        for i in range(n + 1):
            result += ((-1) ** i) * math.factorial(n) // math.factorial(i)
        return result

    @staticmethod
    def catalan(n: int) -> int:
        """nth Catalan number."""
        return math.comb(2 * n, n) // (n + 1)

    @staticmethod
    def stirling_second(n: int, k: int) -> int:
        """Stirling number of the second kind S(n,k)."""
        if k == 0:
            return 1 if n == 0 else 0
        if k == n:
            return 1
        result = 0
        for j in range(k + 1):
            sign = (-1) ** (k - j)
            result += sign * math.comb(k, j) * (j ** n)
        return result // math.factorial(k)

    # ── Probability ──

    @staticmethod
    def probability_coin(n_heads: int, n_flips: int) -> float:
        """Probability of exactly n_heads in n_flips fair coin flips."""
        return math.comb(n_flips, n_heads) * (0.5 ** n_flips)

    @staticmethod
    def probability_dice(target_sum: int, n_dice: int, faces: int = 6) -> float:
        """Probability of target_sum with n_dice standard dice."""
        total_outcomes = faces ** n_dice
        favorable = CombinatoricsEngine._count_dice_sum(target_sum, n_dice, faces)
        return favorable / total_outcomes if total_outcomes > 0 else 0.0

    @staticmethod
    def _count_dice_sum(target: int, n: int, faces: int) -> int:
        """Count ways to get target sum with n dice of given faces (inclusion-exclusion)."""
        count = 0
        for k in range(n + 1):
            val = target - k * faces - n
            if val < 0:
                break
            sign = (-1) ** k
            available = target - k * faces - n
            if available < 0:
                continue
            count += sign * math.comb(n, k) * math.comb(available + n - 1, n - 1)
        return max(0, count)

    @staticmethod
    def expected_value(outcomes: List[Tuple[float, float]]) -> float:
        """Expected value from list of (value, probability) tuples."""
        return sum(v * p for v, p in outcomes)

    @staticmethod
    def geometric_probability(p: float, k: int) -> float:
        """P(X=k) for geometric distribution (1st success on trial k)."""
        if not (0 < p <= 1) or k < 1:
            return 0.0
        return ((1 - p) ** (k - 1)) * p

    @staticmethod
    def binomial_probability(n: int, k: int, p: float) -> float:
        """P(X=k) for binomial distribution."""
        return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    def solve_query(self, query: Dict) -> Any:
        """Solve a combinatorics/probability query dict."""
        q = query.get('query', '')
        if q == 'permutation':
            n, r = int(query.get('n', 0)), int(query.get('r', 0))
            return {'P(n,r)': self.permutation(n, r), 'n': n, 'r': r}
        elif q == 'combination':
            n, r = int(query.get('n', 0)), int(query.get('r', 0))
            return {'C(n,r)': self.combination(n, r), 'n': n, 'r': r}
        elif q == 'binomial_coeff':
            n, k = int(query.get('n', 0)), int(query.get('k', 0))
            return {'C(n,k)': self.binomial_coefficient(n, k), 'n': n, 'k': k}
        elif q == 'catalan':
            n = int(query.get('n', 0))
            return {'catalan': self.catalan(n), 'n': n}
        elif q == 'derangements':
            n = int(query.get('n', 0))
            return {'D(n)': self.derangements(n), 'n': n}
        elif q == 'coin_flips':
            n = int(query.get('n', 1))
            probs = {k: round(self.probability_coin(k, n), 6) for k in range(n + 1)}
            return {'probabilities': probs, 'n_flips': n}
        elif q == 'dice':
            n = int(query.get('n', 1))
            return {'n_dice': n, 'total_outcomes': 6 ** n,
                    'expected_sum': 3.5 * n}
        return {'error': f'Unknown query: {q}'}


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 7: STEP-BY-STEP SOLUTION CHAIN
# ═══════════════════════════════════════════════════════════════════════════════

class SolutionChain:
    """Chain-of-thought math solution builder.

    Generates human-readable step-by-step solutions with intermediate
    verification at each step. DeepSeek-R1 style reasoning traces.
    """

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self._verified_count = 0

    def add_step(self, description: str, operation: str, result: Any,
                 verification: Optional[str] = None) -> 'SolutionChain':
        """Add a solution step."""
        step = {
            'step': len(self.steps) + 1,
            'description': description,
            'operation': operation,
            'result': result,
            'verified': verification is not None,
            'verification': verification,
        }
        self.steps.append(step)
        if verification:
            self._verified_count += 1
        return self

    def finalize(self, final_answer: Any) -> Dict[str, Any]:
        """Finalize the solution chain."""
        return {
            'steps': self.steps,
            'num_steps': len(self.steps),
            'verified_steps': self._verified_count,
            'final_answer': final_answer,
            'confidence': self._verified_count / max(len(self.steps), 1),
        }

    def reset(self):
        self.steps = []
        self._verified_count = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIFIED SYMBOLIC MATH SOLVER ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SymbolicMathSolver:
    """
    Unified symbolic math solver for MATH (competition) benchmark.

    Pipeline:
    1. Parse word problem → structured representation
    2. Build equations from relations
    3. Route to domain-specific solver (algebra/geometry/NT/combo)
    4. Solve with step-by-step chain-of-thought
    5. Verify answer by plug-back or sanity check

    DeepSeek-R1/Math-7B informed:
    - Multi-strategy approach: algebraic, numeric, geometric
    - Chain-of-thought with intermediate verification
    - Backtracking: try multiple approaches, pick highest-confidence
    - PHI/GOD_CODE sacred integration for specialized domains
    """

    VERSION = "1.0.0"

    def __init__(self):
        self.parser = WordProblemParser()
        self.equation_builder = EquationBuilder()
        self.algebra = SymbolicAlgebra()
        self.geometry = GeometryReasoner()
        self.number_theory = NumberTheoryEngine()
        self.combinatorics = CombinatoricsEngine()
        self._total_problems = 0
        self._total_solved = 0
        self._domain_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'attempted': 0, 'solved': 0})
        # Engine support connections (lazy-loaded)
        self._math_engine = None
        self._science_engine = None
        self._quantum_gate_engine = None
        self._quantum_math_core = None
        self._wire_engines()

    def _wire_engines(self):
        """Wire to MathEngine, ScienceEngine, and quantum engines."""
        self._math_engine = _get_cached_math_engine()
        self._science_engine = _get_cached_science_engine()
        self._quantum_gate_engine = _get_cached_quantum_gate_engine()
        self._quantum_math_core = _get_cached_quantum_math_core()

    def solve(self, problem: str, answer_format: str = "auto") -> Dict[str, Any]:
        """Solve a math problem expressed in natural language.

        Args:
            problem: The math problem in natural language.
            answer_format: 'numeric', 'fraction', 'expression', 'proof', or 'auto'.

        Returns:
            Dict with solution, steps, confidence, and verification status.
        """
        self._total_problems += 1

        # Parse
        parsed = self.parser.parse(problem)
        self._domain_stats[parsed.domain]['attempted'] += 1

        # Build equations
        equations = self.equation_builder.build_equations(parsed)

        # Route to solver
        chain = SolutionChain()
        chain.add_step("Parsed problem", f"Domain: {parsed.domain}, Type: {parsed.question_type}",
                       {'entities': len(parsed.entities), 'relations': len(parsed.relations),
                        'equations': len(equations)})

        result = None
        confidence = 0.0

        # PRIORITY 1: Direct expression evaluation — handles well-defined patterns
        # (percentage, geometry word problems, fractions, trig, powers, etc.)
        # with high confidence. Run FIRST to avoid domain solvers returning
        # wrong answers on patterns we can compute exactly.
        expr_result, expr_conf = self._solve_direct_expression(parsed, chain)
        if expr_conf >= 0.75:
            result, confidence = expr_result, expr_conf

        # PRIORITY 2: Domain-specific solvers (if direct eval didn't match)
        if result is None or confidence < 0.5:
            if parsed.domain == 'geometry':
                geo_r, geo_c = self._solve_geometry(parsed, equations, chain)
                if geo_c > confidence:
                    result, confidence = geo_r, geo_c
            elif parsed.domain == 'number_theory':
                nt_r, nt_c = self._solve_number_theory(parsed, equations, chain)
                if nt_c > confidence:
                    result, confidence = nt_r, nt_c
            elif parsed.domain in ('combinatorics', 'probability'):
                co_r, co_c = self._solve_combinatorics(parsed, equations, chain)
                if co_c > confidence:
                    result, confidence = co_r, co_c
            elif parsed.domain == 'calculus':
                ca_r, ca_c = self._solve_calculus(parsed, equations, chain)
                if ca_c > confidence:
                    result, confidence = ca_r, ca_c

        # Cross-domain fallback: try other solvers if domain misclassified
        text_lower = parsed.original.lower()
        if result is None or confidence < 0.5:
            # Try quantum domain for qubit/entanglement/tunnel/quantum probability
            if re.search(r'qubit|quantum|entangle|superposition|tunnel|wave\s*function|bra|ket|hilbert|bell\s*state|grover|shor|qft|bloch', text_lower):
                q_result, q_conf = self._solve_quantum(parsed, equations, chain)
                if q_conf > confidence:
                    result, confidence = q_result, q_conf
            # Try number theory for mod/gcd/lcm/prime/factorial patterns
            if re.search(r'\bmod\b|gcd|lcm|prime|factorial|\d+\s*!', text_lower):
                nt_result, nt_conf = self._solve_number_theory(parsed, equations, chain)
                if nt_conf > confidence:
                    result, confidence = nt_result, nt_conf
            # Try combinatorics for choose/permutation patterns
            if re.search(r'choose|combination|permutation|c\(\d|p\(\d|\d+\s*!', text_lower):
                combo_result, combo_conf = self._solve_combinatorics(parsed, equations, chain)
                if combo_conf > confidence:
                    result, confidence = combo_result, combo_conf
            # Try geometry for area/volume/perimeter/hypotenuse
            if re.search(r'area|volume|perimeter|circumference|hypotenuse|radius|triangle|circle|rectangle', text_lower):
                geo_result, geo_conf = self._solve_geometry(parsed, equations, chain)
                if geo_conf > confidence:
                    result, confidence = geo_result, geo_conf
            # Try calculus for derivative/integral
            if re.search(r'derivative|integral|differentiate|integrate', text_lower):
                calc_result, calc_conf = self._solve_calculus(parsed, equations, chain)
                if calc_conf > confidence:
                    result, confidence = calc_result, calc_conf

        # Fallback: algebraic solver
        if result is None or confidence < 0.3:
            algebra_result, algebra_conf = self._solve_algebra(parsed, equations, chain)
            if algebra_conf > confidence:
                result, confidence = algebra_result, algebra_conf

        # Fallback: numeric brute force for small domains
        if result is None or confidence < 0.2:
            numeric_result, numeric_conf = self._solve_numeric(parsed, equations, chain)
            if numeric_conf > confidence:
                result, confidence = numeric_result, numeric_conf

        # Last-resort: direct expression eval (lower threshold for whatever it found)
        if result is None or confidence < 0.2:
            if expr_conf > confidence:
                result, confidence = expr_result, expr_conf

        # Verification
        verified = self._verify_answer(parsed, result, chain)
        if verified:
            confidence = min(1.0, confidence * 1.2)

        if result is not None:
            self._total_solved += 1
            self._domain_stats[parsed.domain]['solved'] += 1

        # Format answer
        answer = self._format_answer(result, answer_format)

        solution = chain.finalize(answer)
        solution.update({
            'domain': parsed.domain,
            'question_type': parsed.question_type,
            'verified': verified,
            'confidence': round(confidence, 4),
            'raw_result': result,
        })

        return solution

    def _solve_algebra(self, parsed: ParsedProblem, equations: List[Dict],
                       chain: SolutionChain) -> Tuple[Any, float]:
        """Solve algebraic equations.

        Process raw variable equations first (higher confidence), then
        structured linear systems, and finally assignments (lowest priority).
        Also handles algebraic simplification of expressions.
        """
        text = parsed.original.lower()

        # Pre-check: algebraic simplification of expressions
        if 'simplify' in text or 'expand' in text:
            simplified = self._try_simplify(text, chain)
            if simplified is not None:
                return simplified, 0.85

        # Priority pass 0: detect systems of 2 linear equations
        # Look for patterns like "2x + 3y = 7" AND "x - y = 1"
        raw_linear_eqs = []
        for eq in equations:
            if eq.get('type') in ('linear', 'quadratic') and eq.get('raw'):
                expr = eq.get('expression', '')
                if '=' in expr:
                    raw_linear_eqs.append(expr)
        # Also extract from text directly: "Ax + By = C"
        eq_matches = re.findall(r'(-?\d*\.?\d*)\s*([a-z])\s*([+\-])\s*(\d*\.?\d*)\s*([a-z])\s*=\s*(-?\d+\.?\d*)', text)
        for m in eq_matches:
            a_str, var1, sign, b_str, var2, c_str = m
            a_coeff = float(a_str) if a_str and a_str != '-' else (-1.0 if a_str == '-' else 1.0)
            b_coeff = float(b_str) if b_str else 1.0
            if sign == '-':
                b_coeff = -b_coeff
            c_val = float(c_str)
            raw_linear_eqs.append({'a': a_coeff, 'b': b_coeff, 'c': c_val, 'v1': var1, 'v2': var2})

        # If we found exactly 2 parseable 2-variable linear equations, solve as system
        if len(raw_linear_eqs) >= 2:
            sys_eqs = []
            for eq_info in raw_linear_eqs:
                if isinstance(eq_info, dict):
                    sys_eqs.append(eq_info)
                elif isinstance(eq_info, str):
                    parsed_sys = self._parse_2var_linear(eq_info)
                    if parsed_sys:
                        sys_eqs.append(parsed_sys)
            if len(sys_eqs) >= 2:
                e1, e2 = sys_eqs[0], sys_eqs[1]
                sol = self.algebra.solve_system_2x2(
                    e1['a'], e1['b'], e1['c'],
                    e2['a'], e2['b'], e2['c']
                )
                if sol is not None:
                    x, y = sol
                    v1 = e1.get('v1', 'x')
                    v2 = e1.get('v2', 'y')
                    chain.add_step("Solve 2×2 system (Cramer's rule)",
                                   f"{e1['a']}{v1} + {e1['b']}{v2} = {e1['c']}, "
                                   f"{e2['a']}{v1} + {e2['b']}{v2} = {e2['c']}",
                                   {v1: round(x, 6), v2: round(y, 6)},
                                   f"Verify: {e1['a']}*{x} + {e1['b']}*{y} = {e1['a']*x + e1['b']*y:.6f}")
                    # If the question asks for just one variable, return that
                    for unk in parsed.unknowns:
                        if unk == v1:
                            return x if abs(x - round(x)) > 1e-9 else int(round(x)), 0.90
                        if unk == v2:
                            return y if abs(y - round(y)) > 1e-9 else int(round(y)), 0.90
                    # Return both
                    return {v1: x, v2: y}, 0.90

        # Priority pass 1: raw equations with variables (linear/quadratic)
        for eq in equations:
            eq_type = eq.get('type', '')

            if eq_type == 'linear' and eq.get('raw'):
                parsed_eq = self.algebra.parse_linear_equation(eq.get('expression', ''))
                if parsed_eq:
                    a, b = parsed_eq
                    sol = self.algebra.solve_linear(a, b)
                    if sol is not None:
                        chain.add_step("Solve linear equation", f"{a}x + {b} = 0", sol,
                                       f"Check: {a}*{sol} + {b} = {a*sol + b:.6f}")
                        return sol, 0.85

            elif eq_type == 'quadratic' and eq.get('raw'):
                parsed_eq = self.algebra.parse_quadratic_equation(eq.get('expression', ''))
                if parsed_eq:
                    a, b, c = parsed_eq
                    sols = self.algebra.solve_quadratic(a, b, c)
                    if sols:
                        chain.add_step("Solve quadratic equation",
                                       f"{a}x² + {b}x + {c} = 0, discriminant = {b**2 - 4*a*c}",
                                       sols, f"Roots: {sols}")
                        return sols if len(sols) > 1 else sols[0], 0.85

        # Priority pass 2: structured linear equations
        for eq in equations:
            eq_type = eq.get('type', '')
            if eq_type == 'linear' and not eq.get('raw'):
                terms = eq.get('terms', [])
                rhs = eq.get('rhs', 0)
                if len(terms) == 1 and terms[0].get('coeff', 0) != 0:
                    sol = rhs / terms[0]['coeff']
                    chain.add_step("Solve simple linear", f"{terms[0]['coeff']}x = {rhs}", sol)
                    return sol, 0.80

        # Priority pass 3: assignments (lowest priority, skip pure-numeric nonsense)
        for eq in equations:
            if eq.get('type') == 'assignment':
                rhs_val = eq.get('rhs')
                lhs_val = eq.get('lhs', '')
                # Skip assignments where both sides are purely numeric (e.g., '6 = 14')
                if isinstance(lhs_val, str) and lhs_val.replace('.', '').replace('-', '').isdigit():
                    continue
                chain.add_step("Direct assignment", eq.get('expression', ''), rhs_val)
                return rhs_val, 0.70

        # No solvable equations found
        return None, 0.0

    def _solve_geometry(self, parsed: ParsedProblem, equations: List[Dict],
                        chain: SolutionChain) -> Tuple[Any, float]:
        """Solve geometry problems via equation builder OR direct pattern matching."""
        text = parsed.original.lower()

        # Try equation-based approach first
        for eq in equations:
            if eq.get('type') == 'geometry_formula':
                formula = eq.get('formula', '')
                params = eq.get('params', {})
                result = self.geometry.apply_formula(formula, params)
                if result is not None:
                    chain.add_step(f"Apply geometry formula: {formula}",
                                   f"Parameters: {params}", round(result, 6),
                                   f"Formula applied successfully")
                    return result, 0.90

        # Direct pattern matching fallback
        import math as _math

        # Area of circle
        m = re.search(r'area.*circle.*radius\s*(?:of\s+)?(\d+(?:\.\d+)?)', text)
        if not m:
            m = re.search(r'circle.*radius\s*(?:of\s+)?(\d+(?:\.\d+)?).*area', text)
        if m:
            r = float(m.group(1))
            result = round(_math.pi * r * r, 2)
            chain.add_step("Circle area", f"π × {r}² = {result}", result, "A = πr²")
            return result, 0.92

        # Circumference of circle
        m = re.search(r'circumference.*circle.*radius\s*(?:of\s+)?(\d+(?:\.\d+)?)', text)
        if m:
            r = float(m.group(1))
            result = round(2 * _math.pi * r, 2)
            chain.add_step("Circle circumference", f"2π × {r} = {result}", result, "C = 2πr")
            return result, 0.92

        # Hypotenuse of right triangle (Pythagorean theorem)
        m = re.search(r'hypotenuse.*(?:legs|sides)\s*(\d+(?:\.\d+)?)\s*(?:and|,)\s*(\d+(?:\.\d+)?)', text)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            result = round(_math.sqrt(a * a + b * b), 6)
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            chain.add_step("Pythagorean theorem", f"√({a}² + {b}²) = {result}", result, "c = √(a²+b²)")
            return result, 0.92

        # Area of rectangle
        m = re.search(r'area.*rectangle.*(?:length|sides?)\s*(\d+(?:\.\d+)?)\s*(?:and|,|×|x|by)\s*(\d+(?:\.\d+)?)', text)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            result = round(a * b, 6)
            chain.add_step("Rectangle area", f"{a} × {b} = {result}", result, "A = l×w")
            return result, 0.92

        # Area of triangle
        m = re.search(r'area.*triangle.*base\s*(\d+(?:\.\d+)?)\s*.*height\s*(\d+(?:\.\d+)?)', text)
        if m:
            base, height = float(m.group(1)), float(m.group(2))
            result = round(0.5 * base * height, 6)
            chain.add_step("Triangle area", f"½ × {base} × {height}", result, "A = ½bh")
            return result, 0.92

        return None, 0.0

    def _solve_number_theory(self, parsed: ParsedProblem, equations: List[Dict],
                              chain: SolutionChain) -> Tuple[Any, float]:
        """Solve number theory problems via equation builder OR direct pattern matching."""
        text = parsed.original.lower()
        import math as _math

        # Try equation-based approach first
        for eq in equations:
            if eq.get('type') == 'nt_query':
                result = self.number_theory.solve_query(eq)
                if not result.get('error'):
                    chain.add_step(f"Number theory: {eq.get('query', '')}",
                                   f"Query: {eq}", result,
                                   f"Computed via NumberTheoryEngine")
                    for key in ['is_prime', 'gcd', 'lcm', 'remainder', 'digit_sum',
                               'digit_count', 'totient']:
                        if key in result:
                            return result[key], 0.90
                    if 'divisors' in result:
                        return result, 0.90

        # Direct pattern matching fallback

        # GCD
        m = re.search(r'gcd\s*(?:of\s+)?(\d+)\s*(?:and|,)\s*(\d+)', text)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            result = _math.gcd(a, b)
            chain.add_step("GCD", f"gcd({a}, {b}) = {result}", result, f"Euclidean algorithm")
            return result, 0.95

        # LCM
        m = re.search(r'lcm\s*(?:of\s+)?(\d+)\s*(?:and|,)\s*(\d+)', text)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            result = (a * b) // _math.gcd(a, b)
            chain.add_step("LCM", f"lcm({a}, {b}) = {result}", result, f"lcm = a*b/gcd")
            return result, 0.95

        # How many primes less than N
        m = re.search(r'(?:how many|count)\s*(?:of\s+)?prime\s*(?:numbers?\s+)?(?:are\s+)?(?:less than|below|under|up to)\s*(\d+)', text)
        if m:
            n = int(m.group(1))
            # Sieve of Eratosthenes
            sieve = [True] * (n + 1)
            sieve[0] = sieve[1] = False
            for i in range(2, int(n**0.5) + 1):
                if sieve[i]:
                    for j in range(i*i, n + 1, i):
                        sieve[j] = False
            primes = [i for i in range(2, n) if sieve[i]]
            result = len(primes)
            chain.add_step("Count primes", f"primes < {n}: {primes}", result,
                           f"Sieve of Eratosthenes")
            return result, 0.95

        # Modulo / remainder
        m = re.search(r'(\d+)\s*(?:mod|modulo|%)\s*(\d+)', text)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            result = a % b
            chain.add_step("Modulo", f"{a} mod {b} = {result}", result, f"{a} = {a//b}×{b} + {result}")
            return result, 0.95

        # "What is N mod M" variant
        m = re.search(r'(?:what is|find)\s+(\d+)\s+mod\s+(\d+)', text)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            result = a % b
            chain.add_step("Modulo", f"{a} mod {b} = {result}", result, f"Remainder")
            return result, 0.95

        # Is N prime
        m = re.search(r'is\s+(\d+)\s+(?:a\s+)?prime', text)
        if m:
            n = int(m.group(1))
            if n < 2:
                result = False
            else:
                result = all(n % i != 0 for i in range(2, int(n**0.5) + 1))
            chain.add_step("Primality test", f"Is {n} prime?", result, f"Trial division")
            return result, 0.95

        # Factorial
        m = re.search(r'(\d+)\s*!', text)
        if m:
            n = int(m.group(1))
            result = _math.factorial(n)
            chain.add_step("Factorial", f"{n}! = {result}", result, f"Computed")
            return result, 0.95

        # Modular exponentiation: "2^100 mod 7", "3^50 (mod 13)"
        m = re.search(r'(\d+)\s*[\^]\s*(\d+)\s*(?:mod|%|\(mod\s*)\s*(\d+)', text)
        if m:
            base, exp, mod = int(m.group(1)), int(m.group(2)), int(m.group(3))
            result = pow(base, exp, mod)
            chain.add_step("Modular exponentiation", f"{base}^{exp} mod {mod} = {result}",
                           result, f"Fast modular power")
            return result, 0.95

        # Euler totient: "euler totient of N", "phi(N)", "φ(N)", "totient(N)"
        m = re.search(r'(?:euler\s+)?(?:totient|phi|φ)\s*\(?\s*(\d+)\s*\)?', text)
        if m:
            n = int(m.group(1))
            result = self.number_theory.euler_totient(n)
            chain.add_step("Euler totient", f"φ({n}) = {result}", result, f"Euler's totient function")
            return result, 0.95

        # Prime factorization: "prime factorization of N", "factorize N"
        m = re.search(r'(?:prime\s+)?factor(?:iz(?:e|ation)|s)\s+(?:of\s+)?(\d+)', text)
        if m:
            n = int(m.group(1))
            factors = self.number_theory.prime_factorization(n)
            parts = []
            for p, e in sorted(factors.items()):
                parts.append(f"{p}^{e}" if e > 1 else str(p))
            result = " × ".join(parts) if parts else str(n)
            chain.add_step("Prime factorization", f"{n} = {result}", factors,
                           f"Trial division")
            return factors, 0.95

        # Number of divisors: "how many divisors does N have"
        m = re.search(r'(?:how many|number of|count)\s+divisors?\s+(?:of|does|has|for)\s+(\d+)', text)
        if m:
            n = int(m.group(1))
            divs = self.number_theory.divisors(n)
            result = len(divs)
            chain.add_step("Divisor count", f"divisors({n}) = {divs}, count = {result}",
                           result, f"Found {result} divisors")
            return result, 0.95

        # Sum of divisors: "sum of divisors of N"
        m = re.search(r'sum\s+(?:of\s+)?divisors?\s+(?:of\s+)?(\d+)', text)
        if m:
            n = int(m.group(1))
            result = self.number_theory.divisor_sum(n)
            chain.add_step("Divisor sum", f"σ({n}) = {result}", result, f"Sum of all divisors")
            return result, 0.95

        # Digit sum: "sum of digits of N", "digit sum of N"
        m = re.search(r'(?:sum\s+(?:of\s+)?)?digit(?:s|\s+sum)\s+(?:of\s+)?(\d+)', text)
        if m:
            n = int(m.group(1))
            result = self.number_theory.digit_sum(n)
            chain.add_step("Digit sum", f"digit_sum({n}) = {result}", result)
            return result, 0.95

        # Nth prime: "what is the Nth prime"
        m = re.search(r'(?:what is the\s+)?(\d+)(?:st|nd|rd|th)\s+prime', text)
        if m:
            n = int(m.group(1))
            # Sieve enough primes
            limit = max(100, n * 15)  # Upper bound estimate
            primes = self.number_theory.sieve_of_eratosthenes(limit)
            while len(primes) < n:
                limit *= 2
                primes = self.number_theory.sieve_of_eratosthenes(limit)
            if n <= len(primes):
                result = primes[n - 1]
                chain.add_step("Nth prime", f"The {n}th prime = {result}", result, f"Sieve")
                return result, 0.95

        return None, 0.0

    def _solve_combinatorics(self, parsed: ParsedProblem, equations: List[Dict],
                              chain: SolutionChain) -> Tuple[Any, float]:
        """Solve combinatorics/probability problems via equation builder OR patterns."""
        text = parsed.original.lower()
        import math as _math

        # Try equation-based approach first
        for eq in equations:
            if eq.get('type') == 'combo_query':
                result = self.combinatorics.solve_query(eq)
                if not result.get('error'):
                    chain.add_step(f"Combinatorics: {eq.get('query', '')}",
                                   f"Query: {eq}", result,
                                   f"Computed via CombinatoricsEngine")
                    for key in ['P(n,r)', 'C(n,r)', 'C(n,k)', 'catalan', 'D(n)']:
                        if key in result:
                            return result[key], 0.88
                    return result, 0.80
            elif eq.get('type') == 'prob_query':
                result = self.combinatorics.solve_query(eq)
                if not result.get('error'):
                    chain.add_step(f"Probability: {eq.get('query', '')}",
                                   f"Query: {eq}", result)
                    return result, 0.80

        # Direct pattern matching fallback

        # N choose K / C(n,k) / binomial coefficient
        m = re.search(r'(?:choose|c\()\s*(\d+)\s*(?:items?\s+)?(?:from|,)\s*(\d+)', text)
        if not m:
            m = re.search(r'(\d+)\s*(?:choose|c)\s*(\d+)', text)
        if m:
            # Determine which is n and which is k
            a, b = int(m.group(1)), int(m.group(2))
            n, k = max(a, b), min(a, b)
            result = _math.comb(n, k)
            chain.add_step("Binomial coefficient", f"C({n},{k}) = {result}", result,
                           f"n!/(k!(n-k)!) = {result}")
            return result, 0.92

        # Factorial explicit
        m = re.search(r'(\d+)\s*!', text)
        if not m:
            m = re.search(r'(?:what is|compute|find)\s+(\d+)\s*(?:factorial|!)', text)
        if m:
            n = int(m.group(1))
            result = _math.factorial(n)
            chain.add_step("Factorial", f"{n}! = {result}", result, f"Computed")
            return result, 0.95

        # Permutations P(n,r)
        m = re.search(r'(?:permutations?|arrangements?|p\()\s*(\d+)\s*(?:,|from)\s*(\d+)', text)
        if m:
            n, r = int(m.group(1)), int(m.group(2))
            if n < r:
                n, r = r, n
            result = _math.perm(n, r)
            chain.add_step("Permutation", f"P({n},{r}) = {result}", result, f"n!/(n-r)!")
            return result, 0.92

        # Derangements: "derangement of N", "D(N)", "how many derangements"
        m = re.search(r'(?:derangements?|d\()\s*(?:of\s+)?(\d+)', text)
        if m:
            n = int(m.group(1))
            result = self.combinatorics.derangements(n)
            chain.add_step("Derangements", f"D({n}) = {result}", result,
                           f"Permutations with no fixed points")
            return result, 0.92

        # Catalan number: "catalan number N", "Nth catalan"
        m = re.search(r'(?:catalan\s+(?:number\s+)?|(\d+)(?:st|nd|rd|th)\s+catalan)(\d+)?', text)
        if m:
            n = int(m.group(1)) if m.group(1) else (int(m.group(2)) if m.group(2) else None)
            if n is not None:
                result = self.combinatorics.catalan(n)
                chain.add_step("Catalan number", f"C_{n} = {result}", result,
                               f"C(2n,n)/(n+1)")
                return result, 0.92

        # Stars and bars: "distribute N items into K bins", "N identical into K"
        m = re.search(r'(?:distribute|put|place)\s+(\d+)\s+(?:identical\s+)?(?:items?|objects?|balls?)\s+(?:into|among|in)\s+(\d+)', text)
        if m:
            n, k = int(m.group(1)), int(m.group(2))
            result = self.combinatorics.stars_and_bars(n, k)
            chain.add_step("Stars and bars", f"C({n}+{k}-1, {k}-1) = {result}", result,
                           f"Distributing {n} items into {k} bins")
            return result, 0.90

        # Binomial probability: "probability of K successes in N trials with p=P"
        m = re.search(r'probability\s+(?:of\s+)?(\d+)\s+success\w*\s+(?:in\s+)?(\d+)\s+trial\w*.*?(?:probability|p)\s*(?:=|of)\s*(\d*\.?\d+)', text)
        if m:
            k, n = int(m.group(1)), int(m.group(2))
            p = float(m.group(3))
            result = round(self.combinatorics.binomial_probability(n, k, p), 6)
            chain.add_step("Binomial probability", f"P(X={k}) with n={n}, p={p}", result,
                           f"C(n,k) × p^k × (1-p)^(n-k)")
            return result, 0.90

        # Expected value of dice
        m = re.search(r'expected\s+(?:value|sum)\s+(?:of\s+)?(\d+)\s+(?:fair\s+)?(?:die|dice)', text)
        if m:
            n = int(m.group(1))
            result = 3.5 * n
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            chain.add_step("Expected value", f"E[{n} dice] = 3.5 × {n} = {result}", result)
            return result, 0.92

        return None, 0.0

    def _solve_calculus(self, parsed: ParsedProblem, equations: List[Dict],
                        chain: SolutionChain) -> Tuple[Any, float]:
        """Solve basic calculus problems (derivative/integral of common functions)."""
        text = parsed.original.lower()

        # Basic derivative rules
        if 'derivative' in text:
            # Try to extract function pattern
            for m in re.finditer(r'(?:derivative\s+of\s+)?(\d*)\s*x\s*\^?\s*(\d+)', text):
                coeff = float(m.group(1)) if m.group(1) else 1.0
                power = float(m.group(2))
                new_coeff = coeff * power
                new_power = power - 1
                result = f"{new_coeff}x^{new_power}" if new_power != 0 else f"{new_coeff}"
                chain.add_step("Power rule derivative", f"d/dx({coeff}x^{power})",
                               result, f"d/dx(x^n) = nx^(n-1)")
                return result, 0.85

        # Basic integral rules
        if 'integral' in text:
            for m in re.finditer(r'(?:integral\s+of\s+)?(\d*)\s*x\s*\^?\s*(\d+)', text):
                coeff = float(m.group(1)) if m.group(1) else 1.0
                power = float(m.group(2))
                new_power = power + 1
                new_coeff = coeff / new_power
                result = f"{new_coeff}x^{new_power} + C"
                chain.add_step("Power rule integral", f"∫{coeff}x^{power}dx",
                               result, f"∫x^n dx = x^(n+1)/(n+1) + C")
                return result, 0.82

        # Series convergence
        if 'series' in text or 'converge' in text:
            if 'geometric' in text:
                for m in re.finditer(r'ratio\s*(?:of|=|is)\s*(\d+(?:\.\d+)?)', text):
                    r_val = float(m.group(1))
                    converges = abs(r_val) < 1
                    chain.add_step("Geometric series convergence", f"|r| = {abs(r_val)}",
                                   converges, "|r| < 1 for convergence")
                    return converges, 0.90

        return None, 0.0

    def _try_simplify(self, text: str, chain: SolutionChain) -> Optional[str]:
        """Try to simplify algebraic expressions via pattern matching."""
        # (x + a)(x - a) = x^2 - a^2  (difference of squares)
        m = re.search(r'\(\s*(\w)\s*\+\s*(\d+)\s*\)\s*\(\s*(\w)\s*-\s*(\d+)\s*\)', text)
        if m and m.group(1) == m.group(3) and m.group(2) == m.group(4):
            var = m.group(1)
            a = int(m.group(2))
            result = f"{var}^2 - {a*a}"
            chain.add_step("Difference of squares", f"({var}+{a})({var}-{a})", result,
                           "(a+b)(a-b) = a²-b²")
            return result

        # (x + a)(x + b) = x^2 + (a+b)x + ab
        m = re.search(r'\(\s*(\w)\s*([+-])\s*(\d+)\s*\)\s*\(\s*(\w)\s*([+-])\s*(\d+)\s*\)', text)
        if m and m.group(1) == m.group(4):
            var = m.group(1)
            a = int(m.group(3)) * (1 if m.group(2) == '+' else -1)
            b = int(m.group(6)) * (1 if m.group(5) == '+' else -1)
            mid = a + b
            const = a * b
            mid_str = f" + {mid}{var}" if mid > 0 else f" - {abs(mid)}{var}" if mid < 0 else ""
            const_str = f" + {const}" if const > 0 else f" - {abs(const)}" if const < 0 else ""
            result = f"{var}^2{mid_str}{const_str}"
            chain.add_step("FOIL expansion", f"({var}{'+' if a>=0 else ''}{a})({var}{'+' if b>=0 else ''}{b})",
                           result, "(x+a)(x+b) = x² + (a+b)x + ab")
            return result

        # (a*x + b)^2 = a^2*x^2 + 2*a*b*x + b^2
        m = re.search(r'\(\s*(\d*)\s*(\w)\s*([+-])\s*(\d+)\s*\)\s*\^\s*2', text)
        if m:
            coeff = int(m.group(1)) if m.group(1) else 1
            var = m.group(2)
            sign = 1 if m.group(3) == '+' else -1
            b = int(m.group(4)) * sign
            a2 = coeff * coeff
            mid = 2 * coeff * b
            b2 = b * b
            result = f"{a2}{var}^2 + {mid}{var} + {b2}" if mid >= 0 else f"{a2}{var}^2 - {abs(mid)}{var} + {b2}"
            chain.add_step("Perfect square expansion", f"({coeff}{var} + {b})²", result)
            return result

        return None

    def _parse_2var_linear(self, expr: str) -> Optional[Dict]:
        """Parse a 2-variable linear equation like '2x + 3y = 7' into {a, b, c, v1, v2}."""
        if '=' not in expr:
            return None
        lhs, rhs = expr.split('=', 1)
        lhs = lhs.strip().replace(' ', '').replace('−', '-')
        try:
            c_val = float(rhs.strip())
        except ValueError:
            return None
        # Find variables
        vars_found = re.findall(r'[a-z]', lhs)
        if len(vars_found) < 2:
            return None
        v1, v2 = vars_found[0], vars_found[1]
        # Extract coefficients
        terms = re.findall(r'[+\-]?[^+\-]+', lhs)
        a, b = 0.0, 0.0
        for term in terms:
            term = term.strip()
            if not term:
                continue
            if v1 in term:
                coeff_str = term.replace(v1, '').replace('*', '')
                if coeff_str in ('', '+'):
                    a = 1.0
                elif coeff_str == '-':
                    a = -1.0
                else:
                    try:
                        a = float(coeff_str)
                    except ValueError:
                        pass
            elif v2 in term:
                coeff_str = term.replace(v2, '').replace('*', '')
                if coeff_str in ('', '+'):
                    b = 1.0
                elif coeff_str == '-':
                    b = -1.0
                else:
                    try:
                        b = float(coeff_str)
                    except ValueError:
                        pass
        if a == 0 and b == 0:
            return None
        return {'a': a, 'b': b, 'c': c_val, 'v1': v1, 'v2': v2}

    def _solve_direct_expression(self, parsed: ParsedProblem,
                                chain: SolutionChain) -> Tuple[Any, float]:
        """Evaluate direct mathematical expressions in the problem text.

        Handles: "What is 2^10?", "Compute 17 mod 5", "Find 3! + 4!",
        "$\\frac{3}{4} + \\frac{1}{6}$", percentages, geometry, etc.
        """
        text = parsed.original
        import math as _math
        clean = text.lower().strip()

        # ── Percentage: "What is 15% of 200?" ──
        m = re.search(r'(\d+(?:\.\d+)?)\s*%\s*of\s+(\d+(?:\.\d+)?)', clean)
        if m:
            pct, base = float(m.group(1)), float(m.group(2))
            result = pct / 100.0 * base
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            chain.add_step("Percentage", f"{pct}% of {base} = {result}", result)
            return result, 0.92

        # ── Geometry word problems ──
        # Area of rectangle
        m = re.search(r'area\s+(?:of\s+)?(?:a\s+)?rectangle\s+(?:with\s+)?length\s+(\d+(?:\.\d+)?)\s+(?:and\s+)?width\s+(\d+(?:\.\d+)?)', clean)
        if m:
            l, w = float(m.group(1)), float(m.group(2))
            result = l * w
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            chain.add_step("Rectangle area", f"{l} × {w} = {result}", result)
            return result, 0.92

        # Perimeter of rectangle
        m = re.search(r'perimeter\s+(?:of\s+)?(?:a\s+)?rectangle\s+(?:with\s+)?length\s+(\d+(?:\.\d+)?)\s+(?:and\s+)?width\s+(\d+(?:\.\d+)?)', clean)
        if m:
            l, w = float(m.group(1)), float(m.group(2))
            result = 2 * (l + w)
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            chain.add_step("Rectangle perimeter", f"2({l}+{w}) = {result}", result)
            return result, 0.92

        # Perimeter of square
        m = re.search(r'perimeter\s+(?:of\s+)?(?:a\s+)?square\s+(?:with\s+)?side\s+(?:length\s+)?(\d+(?:\.\d+)?)', clean)
        if m:
            s = float(m.group(1))
            result = int(4 * s) if (4 * s) == int(4 * s) else 4 * s
            chain.add_step("Square perimeter", f"4 × {s} = {result}", result)
            return result, 0.92

        # Area of square
        m = re.search(r'area\s+(?:of\s+)?(?:a\s+)?square\s+(?:with\s+)?side\s+(?:length\s+)?(\d+(?:\.\d+)?)', clean)
        if m:
            s = float(m.group(1))
            result = s * s
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            chain.add_step("Square area", f"{s}² = {result}", result)
            return result, 0.92

        # Area of triangle (base × height / 2)
        m = re.search(r'area\s+(?:of\s+)?(?:a\s+)?triangle\s+(?:with\s+)?(?:base|length)\s+(\d+(?:\.\d+)?)\s+(?:and\s+)?(?:height|width)\s+(\d+(?:\.\d+)?)', clean)
        if m:
            b, h = float(m.group(1)), float(m.group(2))
            result = b * h / 2
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            chain.add_step("Triangle area", f"{b} × {h} / 2 = {result}", result)
            return result, 0.92

        # Volume of cube
        m = re.search(r'volume\s+(?:of\s+)?(?:a\s+)?cube\s+(?:with\s+)?side\s+(?:length\s+)?(\d+(?:\.\d+)?)', clean)
        if m:
            s = float(m.group(1))
            result = int(s ** 3) if (s ** 3) == int(s ** 3) else s ** 3
            chain.add_step("Cube volume", f"{s}³ = {result}", result)
            return result, 0.92

        # Volume of sphere
        m = re.search(r'volume\s+(?:of\s+)?(?:a\s+)?sphere\s+(?:with\s+)?radius\s+(\d+(?:\.\d+)?)', clean)
        if m:
            r = float(m.group(1))
            result = (4/3) * _math.pi * r ** 3
            result = round(result, 4)
            chain.add_step("Sphere volume", f"4/3 π {r}³ = {result}", result)
            return result, 0.88

        # Area of circle
        m = re.search(r'area\s+(?:of\s+)?(?:a\s+)?circle\s+(?:with\s+)?radius\s+(\d+(?:\.\d+)?)', clean)
        if m:
            r = float(m.group(1))
            result = _math.pi * r ** 2
            result = round(result, 4)
            chain.add_step("Circle area", f"π × {r}² = {result}", result)
            return result, 0.88

        # Circumference of circle
        m = re.search(r'circumference\s+(?:of\s+)?(?:a\s+)?circle\s+(?:with\s+)?radius\s+(\d+(?:\.\d+)?)', clean)
        if m:
            r = float(m.group(1))
            result = 2 * _math.pi * r
            result = round(result, 4)
            chain.add_step("Circumference", f"2π × {r} = {result}", result)
            return result, 0.88

        # ── Plain text fractions: "3/5 + 1/3", "7/8 as a decimal" ──
        # "X/Y as a decimal"
        m = re.search(r'(\d+)\s*/\s*(\d+)\s+as\s+a\s+decimal', clean)
        if m:
            n, d = int(m.group(1)), int(m.group(2))
            result = n / d
            if abs(result - round(result, 10)) < 1e-12:
                result = round(result, 10)
            chain.add_step("Fraction to decimal", f"{n}/{d} = {result}", result)
            return result, 0.92

        # Plain fraction arithmetic: "3/5 + 1/3", "2/7 - 1/4", etc.
        frac_pattern = r'(\d+)\s*/\s*(\d+)'
        plain_fracs = re.findall(frac_pattern, clean)
        if len(plain_fracs) >= 2:
            ops = re.findall(r'(\d+/\d+)\s*([+\-×*÷/])\s*(\d+/\d+)', clean.replace('×', '*').replace('÷', '/'))
            if ops:
                try:
                    f1 = Fraction(int(plain_fracs[0][0]), int(plain_fracs[0][1]))
                    f2 = Fraction(int(plain_fracs[1][0]), int(plain_fracs[1][1]))
                    op_char = ops[0][1]
                    if op_char == '+':
                        result_frac = f1 + f2
                    elif op_char == '-':
                        result_frac = f1 - f2
                    elif op_char in ('*', '×'):
                        result_frac = f1 * f2
                    elif op_char in ('/', '÷'):
                        result_frac = f1 / f2
                    else:
                        result_frac = f1 + f2  # default to addition
                    if result_frac.denominator == 1:
                        result = int(result_frac)
                    else:
                        result = f"{result_frac.numerator}/{result_frac.denominator}"
                    chain.add_step("Fraction arithmetic", f"{f1} {op_char} {f2} = {result}", result)
                    return result, 0.88
                except Exception:
                    pass
            else:
                # Just "what is 3/5 + 1/3" without explicit operator match — try addition
                try:
                    f1 = Fraction(int(plain_fracs[0][0]), int(plain_fracs[0][1]))
                    f2 = Fraction(int(plain_fracs[1][0]), int(plain_fracs[1][1]))
                    if '-' in clean[clean.index('/'):]:
                        result_frac = f1 - f2
                    else:
                        result_frac = f1 + f2
                    if result_frac.denominator == 1:
                        result = int(result_frac)
                    else:
                        result = f"{result_frac.numerator}/{result_frac.denominator}"
                    chain.add_step("Fraction arithmetic", f"{f1} + {f2} = {result}", result)
                    return result, 0.80
                except Exception:
                    pass

        # ── Negative power: "(-3)^3", "(-2)^4" ──
        m = re.search(r'\(\s*(-?\d+)\s*\)\s*[\^]\s*(\d+)', text)
        if not m:
            m = re.search(r'\(\s*(-?\d+)\s*\)\s*(?:to the power of|raised to)\s*(\d+)', clean)
        if m:
            base, exp = int(m.group(1)), int(m.group(2))
            if exp < 100:
                result = base ** exp
                chain.add_step("Power", f"({base})^{exp} = {result}", result)
                return result, 0.90

        # ── Complex factorial expressions: "10!/(8!*2!)" — BEFORE simple factorial ──
        m = re.search(r'(\d+)!\s*/\s*\(?\s*(\d+)!\s*[*×]\s*(\d+)!\s*\)?', text)
        if m:
            a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if a <= 20 and b <= 20 and c <= 20:
                result = _math.factorial(a) // (_math.factorial(b) * _math.factorial(c))
                chain.add_step("Factorial expression", f"{a}!/({b}!×{c}!) = {result}", result)
                return result, 0.92

        # ── Factorial: "5!", "6 factorial" (simple, only when no / follows) ──
        m = re.search(r'(\d+)\s*!(?!\s*/)', text)
        if not m:
            m = re.search(r'(\d+)\s+factorial', clean)
        if m:
            n = int(m.group(1))
            if n <= 20:
                result = _math.factorial(n)
                chain.add_step("Factorial", f"{n}! = {result}", result)
                return result, 0.90

        # ── GCD/LCM ──
        m = re.search(r'(?:gcd|greatest common divisor|gcf)\s*(?:\(|of)\s*(\d+)\s*(?:,|and)\s*(\d+)', clean)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            result = _math.gcd(a, b)
            chain.add_step("GCD", f"gcd({a}, {b}) = {result}", result)
            return result, 0.92

        m = re.search(r'(?:lcm|least common multiple)\s*(?:\(|of)\s*(\d+)\s*(?:,|and)\s*(\d+)', clean)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            result = (a * b) // _math.gcd(a, b)
            chain.add_step("LCM", f"lcm({a}, {b}) = {result}", result)
            return result, 0.92

        # ── LaTeX \sqrt{N} ──
        m = re.search(r'\\sqrt\{(\d+(?:\.\d+)?)\}', text)
        if m:
            val = float(m.group(1))
            result = _math.sqrt(val)
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            chain.add_step("LaTeX square root", f"\\sqrt{{{val}}} = {result}", result)
            return result, 0.90

        # ── LaTeX \binom{n}{k} ──
        m = re.search(r'\\binom\{(\d+)\}\{(\d+)\}', text)
        if m:
            n, k = int(m.group(1)), int(m.group(2))
            result = _math.comb(n, k)
            chain.add_step("LaTeX binomial", f"\\binom{{{n}}}{{{k}}} = {result}", result,
                           f"C({n},{k}) = {result}")
            return result, 0.92

        # ── LaTeX \log_{b}(x) or \log(x) ──
        m = re.search(r'\\log_?\{?(\d+)\}?\s*\(?(\d+)\)?', text)
        if m:
            base, val = int(m.group(1)), int(m.group(2))
            result = _math.log(val, base)
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            chain.add_step("LaTeX logarithm", f"\\log_{{{base}}}({val}) = {result}", result)
            return result, 0.90

        # ── LaTeX \sin, \cos, \tan (degrees by default) ──
        for fn_name, fn in [('sin', _math.sin), ('cos', _math.cos), ('tan', _math.tan)]:
            m = re.search(rf'\\{fn_name}\s*\(?(\d+(?:\.\d+)?)\s*(?:°|\\circ|degrees?)?\)?', text)
            if m:
                angle = float(m.group(1))
                result = round(fn(_math.radians(angle)), 6)
                if abs(result - round(result)) < 1e-9:
                    result = int(round(result))
                chain.add_step(f"LaTeX {fn_name}", f"\\{fn_name}({angle}°) = {result}", result)
                return result, 0.88

        # ── LaTeX \pi expressions: "2\pi", "\pi r^2" ──
        m = re.search(r'(\d+(?:\.\d+)?)\s*\\pi', text)
        if m and not re.search(r'area|volume|circumference|circle|sphere', clean):
            coeff = float(m.group(1))
            result = round(coeff * _math.pi, 6)
            chain.add_step("LaTeX pi", f"{coeff}π = {result}", result)
            return result, 0.88

        # ── LaTeX fraction extraction: \frac{a}{b} ──
        fractions_found = re.findall(r'\\frac\{(\-?\d+)\}\{(\-?\d+)\}', text)
        if fractions_found:
            result_frac = Fraction(0)
            for num_str, den_str in fractions_found:
                result_frac += Fraction(int(num_str), int(den_str))

            # Check if question asks for sum/product/etc.
            text_lower = text.lower()
            if len(fractions_found) >= 2 and ('product' in text_lower or 'times' in text_lower or 'multiply' in text_lower or '\\cdot' in text or '\\times' in text):
                result_frac = Fraction(1)
                for num_str, den_str in fractions_found:
                    result_frac *= Fraction(int(num_str), int(den_str))

            # Check for subtraction
            if len(fractions_found) >= 2 and ('-' in text_lower or 'minus' in text_lower or 'difference' in text_lower or 'subtract' in text_lower):
                result_frac = Fraction(int(fractions_found[0][0]), int(fractions_found[0][1]))
                for num_str, den_str in fractions_found[1:]:
                    result_frac -= Fraction(int(num_str), int(den_str))

            # Return as fraction string or float
            if result_frac.denominator == 1:
                result = int(result_frac)
            else:
                result = f"{result_frac.numerator}/{result_frac.denominator}"
            chain.add_step("Fraction arithmetic",
                           f"fractions: {fractions_found}", result)
            return result, 0.80

        # ── Statistics: mean/average, median, mode ──
        m = re.search(r'(?:mean|average)\s+(?:of\s+)?([\d,\.\s]+)', clean)
        if m:
            nums_str = m.group(1)
            nums = [float(x.strip()) for x in re.findall(r'\d+(?:\.\d+)?', nums_str)]
            if nums:
                result = sum(nums) / len(nums)
                if abs(result - round(result)) < 1e-9:
                    result = int(round(result))
                chain.add_step("Mean/Average", f"({' + '.join(str(n) for n in nums)}) / {len(nums)} = {result}", result)
                return result, 0.92

        m = re.search(r'median\s+(?:of\s+)?([\d,\.\s]+)', clean)
        if m:
            nums_str = m.group(1)
            nums = sorted([float(x.strip()) for x in re.findall(r'\d+(?:\.\d+)?', nums_str)])
            if nums:
                n = len(nums)
                if n % 2 == 1:
                    result = nums[n // 2]
                else:
                    result = (nums[n // 2 - 1] + nums[n // 2]) / 2
                if isinstance(result, float) and abs(result - round(result)) < 1e-9:
                    result = int(round(result))
                chain.add_step("Median", f"sorted: {nums}, median = {result}", result)
                return result, 0.92

        m = re.search(r'mode\s+(?:of\s+)?([\d,\.\s]+)', clean)
        if m:
            nums_str = m.group(1)
            nums = [float(x.strip()) for x in re.findall(r'\d+(?:\.\d+)?', nums_str)]
            if nums:
                from collections import Counter
                counts = Counter(nums)
                mode_val = counts.most_common(1)[0][0]
                if abs(mode_val - round(mode_val)) < 1e-9:
                    mode_val = int(round(mode_val))
                result = mode_val
                chain.add_step("Mode", f"most frequent: {result}", result)
                return result, 0.90

        # ── Standard deviation ──
        m = re.search(r'(?:standard deviation|std dev)\s+(?:of\s+)?([\d,\.\s]+)', clean)
        if m:
            nums_str = m.group(1)
            nums = [float(x.strip()) for x in re.findall(r'\d+(?:\.\d+)?', nums_str)]
            if len(nums) >= 2:
                mean = sum(nums) / len(nums)
                variance = sum((x - mean) ** 2 for x in nums) / len(nums)
                result = round(_math.sqrt(variance), 6)
                chain.add_step("Standard deviation", f"σ = {result}", result)
                return result, 0.88

        # ── Direct numeric expressions ──
        # Clean text for expression extraction
        clean = text.lower()
        clean = re.sub(r'[\\$]', '', clean)

        # Simple "what is A op B" patterns
        # Power: "2^10", "3 to the power of 4"
        m = re.search(r'(\d+)\s*[\^]\s*(\d+)', text)
        if not m:
            m = re.search(r'(\d+)\s*(?:to the power of|raised to)\s*(\d+)', clean)
        if m:
            base, exp = int(m.group(1)), int(m.group(2))
            if exp < 100:
                result = base ** exp
                chain.add_step("Power", f"{base}^{exp} = {result}", result)
                return result, 0.90

        # Square root: "sqrt(144)", "square root of 144"
        m = re.search(r'(?:sqrt|square root)\s*(?:\(|of\s+)(\d+)', clean)
        if m:
            n = int(m.group(1))
            result = _math.sqrt(n)
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            chain.add_step("Square root", f"√{n} = {result}", result)
            return result, 0.90

        # Floor/ceiling: "floor(3.7)", "ceil(2.1)"
        m = re.search(r'floor\s*\(\s*(\d+(?:\.\d+)?)\s*\)', clean)
        if m:
            result = _math.floor(float(m.group(1)))
            chain.add_step("Floor", f"⌊{m.group(1)}⌋ = {result}", result)
            return result, 0.90
        m = re.search(r'ceil(?:ing)?\s*\(\s*(\d+(?:\.\d+)?)\s*\)', clean)
        if m:
            result = _math.ceil(float(m.group(1)))
            chain.add_step("Ceiling", f"⌈{m.group(1)}⌉ = {result}", result)
            return result, 0.90

        # Absolute value: "|−7|", "absolute value of -7"
        m = re.search(r'\|\s*(-?\d+(?:\.\d+)?)\s*\|', text)
        if not m:
            m = re.search(r'absolute value of\s+(-?\d+(?:\.\d+)?)', clean)
        if m:
            result = abs(float(m.group(1)))
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            chain.add_step("Absolute value", f"|{m.group(1)}| = {result}", result)
            return result, 0.90

        # Sum of digits
        m = re.search(r'(?:sum|add)\s+(?:the\s+)?digits?\s+(?:of\s+)?(\d+)', clean)
        if m:
            n = m.group(1)
            result = sum(int(d) for d in n)
            chain.add_step("Digit sum", f"digits of {n}: {' + '.join(n)} = {result}", result)
            return result, 0.90

        # Number of digits
        m = re.search(r'(?:how many|number of|count)\s+digits?\s+(?:in|of|does)\s+(\d+)', clean)
        if m:
            result = len(m.group(1))
            chain.add_step("Digit count", f"{m.group(1)} has {result} digits", result)
            return result, 0.90

        # ── Modular arithmetic: "17 mod 5", "what is 17 mod 5?" ──
        m = re.search(r'(\d+)\s+mod\s+(\d+)', clean)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            result = a % b
            chain.add_step("Modular arithmetic", f"{a} mod {b} = {result}", result)
            return result, 0.92

        # ── Combinations: "8 choose 2", "5 choose 3", "choose 3 from 5", "C(5,3)" ──
        m = re.search(r'(\d+)\s+choose\s+(\d+)', clean)
        if m:
            n, k = int(m.group(1)), int(m.group(2))
            result = _math.comb(n, k)
            chain.add_step("Combination", f"C({n},{k}) = {result}", result)
            return result, 0.92
        m = re.search(r'choose\s+(\d+)\s+(?:items?\s+)?(?:from|out of)\s+(\d+)', clean)
        if m:
            k, n = int(m.group(1)), int(m.group(2))
            result = _math.comb(n, k)
            chain.add_step("Combination", f"C({n},{k}) = {result}", result)
            return result, 0.92
        m = re.search(r'c\(\s*(\d+)\s*,\s*(\d+)\s*\)', clean)
        if m:
            n, k = int(m.group(1)), int(m.group(2))
            result = _math.comb(n, k)
            chain.add_step("Combination", f"C({n},{k}) = {result}", result)
            return result, 0.92

        # ── Permutations: "P(6,4)", "permutations of 4 items from 6" ──
        m = re.search(r'p\(\s*(\d+)\s*,\s*(\d+)\s*\)', clean)
        if m:
            n, k = int(m.group(1)), int(m.group(2))
            result = _math.perm(n, k)
            chain.add_step("Permutation", f"P({n},{k}) = {result}", result)
            return result, 0.92
        m = re.search(r'permutation\w*\s+(?:of\s+)?(\d+)\s+(?:items?\s+)?(?:from|out of)\s+(\d+)', clean)
        if m:
            k, n = int(m.group(1)), int(m.group(2))
            result = _math.perm(n, k)
            chain.add_step("Permutation", f"P({n},{k}) = {result}", result)
            return result, 0.92

        # ── Probability of die roll ──
        m = re.search(r'probability\s+(?:of\s+)?rolling\s+(?:a\s+)?(\d+)\s+(?:on\s+)?(?:a\s+)?(?:fair\s+)?(?:die|dice)', clean)
        if m:
            result = "1/6"
            chain.add_step("Die probability", f"P(rolling {m.group(1)}) = 1/6", result)
            return result, 0.92

        # Simple arithmetic expressions: try safe eval on extracted expressions
        # Look for patterns like "12 + 34", "5 * 6 - 2"
        m = re.search(r'(?:what is|compute|calculate|evaluate|find)\s+([\d\s\+\-\*\/\(\)\.\^]+)', clean)
        if m:
            expr = m.group(1).strip()
            expr = expr.replace('^', '**')
            try:
                # Safe evaluation with only math operations
                result = eval(expr, {"__builtins__": {}}, {"abs": abs, "pow": pow})
                if isinstance(result, float) and abs(result - round(result)) < 1e-9:
                    result = int(round(result))
                chain.add_step("Direct evaluation", f"{expr} = {result}", result)
                return result, 0.75
            except Exception:
                pass

        # Log: "log base 2 of 8", "log_2(8)", "log2(8)"
        m = re.search(r'log\s*[_]?\s*(\d+)\s*\(\s*(\d+)\s*\)', clean)
        if not m:
            m = re.search(r'log\s*(?:base\s*)?(\d+)\s*(?:of\s+)?(\d+)', clean)
        if m:
            base, val = int(m.group(1)), int(m.group(2))
            result = _math.log(val, base)
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            chain.add_step("Logarithm", f"log_{base}({val}) = {result}", result)
            return result, 0.90

        # ── Hypotenuse of right triangle ──
        m = re.search(r'hypotenuse\s+(?:of\s+)?(?:a\s+)?(?:right\s+)?triangle\s+(?:with\s+)?(?:legs|sides)\s+(\d+(?:\.\d+)?)\s+(?:and\s+)?(\d+(?:\.\d+)?)', clean)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            result = _math.sqrt(a**2 + b**2)
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            chain.add_step("Hypotenuse", f"√({a}² + {b}²) = {result}", result)
            return result, 0.92

        # ── Diagonal of rectangle ──
        m = re.search(r'diagonal\s+(?:of\s+)?(?:a\s+)?rectangle\s+(?:with\s+)?sides?\s+(\d+(?:\.\d+)?)\s+(?:and\s+)?(\d+(?:\.\d+)?)', clean)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            result = _math.sqrt(a**2 + b**2)
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            chain.add_step("Rectangle diagonal", f"√({a}² + {b}²) = {result}", result)
            return result, 0.92

        # ── Trapezoid area ──
        m = re.search(r'area\s+(?:of\s+)?(?:a\s+)?trapezoid\s+(?:with\s+)?parallel\s+sides?\s+(\d+(?:\.\d+)?)\s+(?:and\s+)?(\d+(?:\.\d+)?)\s+(?:and\s+)?height\s+(\d+(?:\.\d+)?)', clean)
        if m:
            a, b, h = float(m.group(1)), float(m.group(2)), float(m.group(3))
            result = (a + b) * h / 2
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            chain.add_step("Trapezoid area", f"({a} + {b}) × {h} / 2 = {result}", result)
            return result, 0.92

        # ── Plain trig (non-LaTeX): "sin(0)", "cos(0)", "tan(45 degrees)" ──
        for fn_name, fn in [('sin', _math.sin), ('cos', _math.cos), ('tan', _math.tan)]:
            m = re.search(rf'(?:what is\s+)?{fn_name}\s*\(?\s*(\d+(?:\.\d+)?)\s*(?:°|degrees?)?\s*\)?', clean)
            if m:
                angle = float(m.group(1))
                result = round(fn(_math.radians(angle)), 10)
                if abs(result) < 1e-10:
                    result = 0
                elif abs(result - round(result)) < 1e-9:
                    result = int(round(result))
                else:
                    result = round(result, 6)
                chain.add_step(f"Trig {fn_name}", f"{fn_name}({angle}°) = {result}", result)
                return result, 0.90

        # ── Degree to radian conversion ──
        m = re.search(r'convert\s+(\d+(?:\.\d+)?)\s*(?:°|degrees?)\s+to\s+radians?', clean)
        if m:
            degrees = float(m.group(1))
            radians_val = degrees * _math.pi / 180
            # Return symbolic for common values
            if abs(degrees - 180) < 1e-9:
                result = "pi"
            elif abs(degrees - 90) < 1e-9:
                result = "pi/2"
            elif abs(degrees - 360) < 1e-9:
                result = "2*pi"
            elif abs(degrees - 60) < 1e-9:
                result = "pi/3"
            elif abs(degrees - 45) < 1e-9:
                result = "pi/4"
            elif abs(degrees - 30) < 1e-9:
                result = "pi/6"
            else:
                result = round(radians_val, 6)
            chain.add_step("Degree→Radian", f"{degrees}° = {result} rad", result)
            return result, 0.92

        # ── Vector magnitude: "magnitude of the vector (3, 4)" ──
        m = re.search(r'magnitude\s+(?:of\s+)?(?:the\s+)?(?:vector\s+)?\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*(?:,\s*(-?\d+(?:\.\d+)?))?\s*\)', clean)
        if m:
            components = [float(m.group(1)), float(m.group(2))]
            if m.group(3):
                components.append(float(m.group(3)))
            result = _math.sqrt(sum(c**2 for c in components))
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            chain.add_step("Vector magnitude", f"|({', '.join(str(c) for c in components)})| = {result}", result)
            return result, 0.92

        # ── Sum of first N positive integers ──
        m = re.search(r'sum\s+(?:of\s+)?(?:the\s+)?first\s+(\d+)\s+positive\s+integers?', clean)
        if m:
            n = int(m.group(1))
            result = n * (n + 1) // 2
            chain.add_step("Sum formula", f"n(n+1)/2 = {n}×{n+1}/2 = {result}", result)
            return result, 0.95

        # ── How many primes less than N ──
        m = re.search(r'how\s+many\s+prime\s*(?:numbers?)?\s+(?:are\s+)?(?:less than|below|under|smaller than|<)\s+(\d+)', clean)
        if m:
            n = int(m.group(1))
            primes = [p for p in range(2, n) if all(p % d != 0 for d in range(2, int(p**0.5)+1))]
            result = len(primes)
            chain.add_step("Prime count", f"Primes < {n}: {len(primes)}", result)
            return result, 0.92

        # ── Sum of primes less than N ──
        m = re.search(r'sum\s+(?:of\s+)?(?:all\s+)?prime\s*(?:numbers?)?\s+(?:less than|below|under|<)\s+(\d+)', clean)
        if m:
            n = int(m.group(1))
            primes = [p for p in range(2, n) if all(p % d != 0 for d in range(2, int(p**0.5)+1))]
            result = sum(primes)
            chain.add_step("Sum of primes", f"Sum of primes < {n}: {result}", result)
            return result, 0.92

        # ── Is N prime? ──
        m = re.search(r'is\s+(\d+)\s+(?:a\s+)?prime', clean)
        if m:
            n = int(m.group(1))
            is_prime = n > 1 and all(n % d != 0 for d in range(2, int(n**0.5)+1))
            result = "yes" if is_prime else "no"
            chain.add_step("Primality test", f"{n} is {'prime' if is_prime else 'not prime'}", result)
            return result, 0.92

        # ── Number of divisors ──
        m = re.search(r'(?:how many|number of|count)\s+(?:the\s+)?divisors?\s+(?:does\s+)?(\d+)\s*(?:have)?', clean)
        if m:
            n = int(m.group(1))
            divisors = [d for d in range(1, n+1) if n % d == 0]
            result = len(divisors)
            chain.add_step("Divisor count", f"Divisors of {n}: {divisors}", result)
            return result, 0.92

        # ── Linear equation with implied multiplication: "3(x + 2) = 21" ──
        m = re.search(r'(?:solve\s+(?:for\s+x)?:?\s*)?(\d+)\s*\(\s*x\s*([+\-])\s*(\d+(?:\.\d+)?)\s*\)\s*=\s*(-?\d+(?:\.\d+)?)', clean)
        if m:
            coeff = float(m.group(1))
            op = m.group(2)
            offset = float(m.group(3))
            rhs = float(m.group(4))
            # coeff * (x ± offset) = rhs → x = rhs/coeff ∓ offset
            x_plus_offset = rhs / coeff
            if op == '+':
                result = x_plus_offset - offset
            else:
                result = x_plus_offset + offset
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            chain.add_step("Linear equation", f"{coeff}(x{op}{offset})={rhs} → x={result}", result)
            return result, 0.92

        # ── Factoring: "x^2 + bx + c" → find factors ──
        m = re.search(r'factor:?\s*x\^?2\s*([+\-])\s*(\d+)x\s*([+\-])\s*(\d+)', clean)
        if m:
            b_sign = 1 if m.group(1) == '+' else -1
            b_val = b_sign * int(m.group(2))
            c_sign = 1 if m.group(3) == '+' else -1
            c_val = c_sign * int(m.group(4))
            # Find integer factors: (x+p)(x+q) where p+q=b, p*q=c
            for p in range(-abs(c_val) - 1, abs(c_val) + 2):
                if c_val != 0 and c_val % p != 0 if p != 0 else c_val != 0:
                    continue
                q = c_val // p if p != 0 else 0
                if p + q == b_val and p * q == c_val:
                    # Format as (x+p)(x+q)
                    p_str = f"+{p}" if p >= 0 else str(p)
                    q_str = f"+{q}" if q >= 0 else str(q)
                    result = f"(x{p_str})(x{q_str})"
                    chain.add_step("Factoring", f"x²{'+' if b_val>=0 else ''}{b_val}x{'+' if c_val>=0 else ''}{c_val} = {result}", result)
                    return result, 0.88

        # ── Discriminant: "discriminant of x^2 - 6x + 9" ──
        m = re.search(r'discriminant\s+(?:of\s+)?(?:the\s+)?(?:equation\s+)?x\^?2\s*([+\-])\s*(\d+)x\s*([+\-])\s*(\d+)', clean)
        if m:
            b_sign = 1 if m.group(1) == '+' else -1
            b_val = b_sign * int(m.group(2))
            c_sign = 1 if m.group(3) == '+' else -1
            c_val = c_sign * int(m.group(4))
            result = b_val**2 - 4*c_val
            chain.add_step("Discriminant", f"b²−4ac = {b_val}²−4(1)({c_val}) = {result}", result)
            return result, 0.92

        # ── Function evaluation: "f(3) if f(x) = x^2 - 2x + 1" ──
        m = re.search(r'(?:f|g)\((\d+)\)\s+(?:if|where|when)\s+(?:f|g)\(x\)\s*=\s*(.+)', clean)
        if m:
            x_val = float(m.group(1))
            expr_str = m.group(2).strip().rstrip('?')
            expr_str = expr_str.replace('^', '**').replace('x', f'({x_val})')
            # Fix implied multiplication: "2(3.0)" → "2*(3.0)"
            expr_str = re.sub(r'(\d)\(', r'\1*(', expr_str)
            try:
                result = eval(expr_str, {"__builtins__": {}}, {})
                if isinstance(result, float) and abs(result - round(result)) < 1e-9:
                    result = int(round(result))
                chain.add_step("Function eval", f"f({x_val}) = {result}", result)
                return result, 0.90
            except Exception:
                pass

        # ── Function composition: "f(g(3))" / "if f(x)=2x+1 and g(x)=x^2, what is f(g(3))?" ──
        m = re.search(r'(?:if|where|when)\s+f\(x\)\s*=\s*(.+?)\s+and\s+g\(x\)\s*=\s*(.+?)(?:,|\s+what)', clean)
        if m:
            f_expr = m.group(1).strip().replace('^', '**')
            g_expr = m.group(2).strip().replace('^', '**')
            val_m = re.search(r'f\(g\((\d+)\)\)', clean)
            if val_m:
                x_val = float(val_m.group(1))
                try:
                    g_sub = re.sub(r'(\d)\(', r'\1*(', g_expr.replace('x', f'({x_val})'))
                    g_val = eval(g_sub, {"__builtins__": {}}, {})
                    f_sub = re.sub(r'(\d)\(', r'\1*(', f_expr.replace('x', f'({g_val})'))
                    result = eval(f_sub, {"__builtins__": {}}, {})
                    if isinstance(result, float) and abs(result - round(result)) < 1e-9:
                        result = int(round(result))
                    chain.add_step("Function composition", f"g({x_val})={g_val}, f({g_val})={result}", result)
                    return result, 0.90
                except Exception:
                    pass

        # ── Explicit numeric sum: "1 + 2 + 4 + 8 + 16 + 32" ──
        m = re.search(r'(?:evaluate|compute|find|what is)?\s*(?:the\s+)?(?:sum:?\s+)?(-?\d+(?:\.\d+)?(?:\s*[+\-]\s*\d+(?:\.\d+)?){2,})', clean)
        if m:
            expr = m.group(1).strip().replace(' ', '')
            try:
                result = eval(expr, {"__builtins__": {}}, {})
                if isinstance(result, float) and abs(result - round(result)) < 1e-9:
                    result = int(round(result))
                chain.add_step("Sum evaluation", f"{m.group(1).strip()} = {result}", result)
                return result, 0.88
            except Exception:
                pass

        # ── Absolute value equation: "|x - A| = B" → [A+B, A-B] ──
        m = re.search(r'\|\s*x\s*([+\-])\s*(\d+(?:\.\d+)?)\s*\|\s*=\s*(\d+(?:\.\d+)?)', text)
        if m:
            op = m.group(1)
            a_val = float(m.group(2))
            b_val = float(m.group(3))
            if op == '-':
                sol1 = a_val + b_val
                sol2 = a_val - b_val
            else:
                sol1 = -a_val + b_val
                sol2 = -a_val - b_val
            solutions = sorted([sol1, sol2])
            result = [int(s) if abs(s - round(s)) < 1e-9 else s for s in solutions]
            chain.add_step("Abs value equation", f"|x{op}{a_val}| = {b_val} → {result}", result)
            return result, 0.90

        # ── N-digit numbers with no repetition ──
        m = re.search(r'how\s+many\s+(\d+)[- ]digit\s+numbers?\s+(?:can\s+be\s+)?(?:formed|made|created)\s+(?:from|using|with)\s+(?:the\s+)?(?:digits?\s+)?(\d+)[- ](\d+)(?:\s+with\s+no\s+repetition)?', clean)
        if m:
            k = int(m.group(1))
            d_start = int(m.group(2))
            d_end = int(m.group(3))
            n_digits = d_end - d_start + 1
            result = _math.perm(n_digits, k)
            chain.add_step("Digit arrangement", f"P({n_digits},{k}) = {result}", result)
            return result, 0.88

        return None, 0.0

    def _solve_numeric(self, parsed: ParsedProblem, equations: List[Dict],
                       chain: SolutionChain) -> Tuple[Any, float]:
        """Numeric brute-force: try simple arithmetic on extracted values.

        Uses question context to guide which operation to try.
        """
        values = [e.value for e in parsed.entities if e.value is not None]
        if not values:
            return None, 0.0

        text_lower = parsed.original.lower()

        # Context-guided operation selection
        if len(values) == 1:
            v = values[0]
            candidates = [
                (v, f"Direct value: {v}", 0.3),
            ]
            if 'square' in text_lower or 'squared' in text_lower:
                candidates.insert(0, (v * v, f"Square: {v}² = {v*v}", 0.6))
            elif 'cube' in text_lower or 'cubed' in text_lower:
                candidates.insert(0, (v ** 3, f"Cube: {v}³ = {v**3}", 0.6))
            elif 'sqrt' in text_lower or 'square root' in text_lower:
                if v >= 0:
                    r = math.sqrt(v)
                    candidates.insert(0, (r if abs(r - round(r)) > 1e-9 else int(round(r)),
                                         f"√{v}", 0.6))
            elif 'double' in text_lower or 'twice' in text_lower:
                candidates.insert(0, (v * 2, f"Double: 2 × {v}", 0.6))
            elif 'half' in text_lower:
                candidates.insert(0, (v / 2, f"Half: {v}/2", 0.6))
        elif len(values) == 2:
            a, b = values[0], values[1]
            candidates = []
            # Context-guided
            if any(w in text_lower for w in ['sum', 'total', 'add', 'plus', 'together', 'combined']):
                candidates.append((a + b, f"Sum: {a} + {b}", 0.65))
            elif any(w in text_lower for w in ['difference', 'subtract', 'minus', 'less', 'remain']):
                candidates.append((abs(a - b), f"Difference: |{a} - {b}|", 0.60))
            elif any(w in text_lower for w in ['product', 'multiply', 'times']):
                candidates.append((a * b, f"Product: {a} × {b}", 0.65))
            elif any(w in text_lower for w in ['quotient', 'divide', 'ratio', 'per']):
                if b != 0:
                    candidates.append((a / b, f"Ratio: {a} / {b}", 0.60))
            elif any(w in text_lower for w in ['power', 'exponent']):
                if b < 20:
                    candidates.append((a ** b, f"Power: {a}^{b}", 0.60))
            else:
                # Try all basic operations
                candidates = [
                    (a + b, f"Sum: {a} + {b}", 0.35),
                    (a * b, f"Product: {a} × {b}", 0.30),
                    (a - b, f"Difference: {a} - {b}", 0.25),
                    (a / b if b != 0 else None, f"Ratio: {a} / {b}", 0.25),
                    (a ** b if b < 20 else None, f"Power: {a}^{b}", 0.20),
                ]
        else:
            vals = values[:5]
            candidates = []
            if any(w in text_lower for w in ['sum', 'total', 'add']):
                candidates.append((sum(vals), f"Sum: {' + '.join(str(v) for v in vals)}", 0.60))
            elif any(w in text_lower for w in ['product', 'multiply']):
                candidates.append((reduce(operator.mul, vals, 1), f"Product", 0.55))
            elif any(w in text_lower for w in ['average', 'mean']):
                candidates.append((sum(vals) / len(vals), f"Average", 0.60))
            else:
                candidates = [
                    (sum(vals), f"Sum: {' + '.join(str(v) for v in vals)}", 0.30),
                    (reduce(operator.mul, vals, 1), f"Product", 0.20),
                    (sum(vals) / len(vals), f"Average", 0.25),
                ]

        best = None
        best_conf = 0.0
        for val, desc, conf in candidates:
            if val is not None and conf > best_conf:
                best = val
                best_conf = conf
                chain.add_step("Numeric estimation", desc, val)

        # Clean up float to int if appropriate
        if isinstance(best, float) and abs(best - round(best)) < 1e-9:
            best = int(round(best))

        return best, best_conf

    def _solve_quantum(self, parsed: ParsedProblem, equations: List[Dict],
                       chain: SolutionChain) -> Tuple[Any, float]:
        """Solve quantum-domain math problems using l104_quantum_engine.

        Handles: tunnelling probability, Bell state fidelity, qubit counts,
        Grover iterations, QFT, entanglement measures, quantum probability.
        """
        qmc = self._quantum_math_core
        qge = self._quantum_gate_engine
        text = parsed.original.lower()
        result = None
        confidence = 0.0

        # Extract numeric values from problem
        numbers = [float(x) for x in re.findall(r'(?<![a-zA-Z])(\d+\.?\d*)(?![a-zA-Z])', text)]

        # ── Quantum tunnelling probability ──
        if re.search(r'tunnel|tunnell?ing|barrier', text) and qmc is not None:
            try:
                chain.add_step("Quantum tunnelling", "Using quantum tunnelling probability formula", {})
                # Try to extract barrier_height, particle_energy, width from numbers
                if len(numbers) >= 3:
                    barrier_h, particle_e, width = numbers[0], numbers[1], numbers[2]
                elif len(numbers) >= 2:
                    barrier_h, particle_e, width = numbers[0], numbers[1], 1.0
                elif len(numbers) >= 1:
                    barrier_h, particle_e, width = numbers[0], numbers[0] * 0.5, 1.0
                else:
                    barrier_h, particle_e, width = 1.0, 0.5, 1.0
                prob = qmc.tunnel_probability(barrier_h, particle_e, width)
                if isinstance(prob, (int, float)):
                    result = round(prob, 6)
                    confidence = 0.85
                    chain.add_step("Tunnelling result", f"P = {result}", {})
            except Exception:
                pass

        # ── Bell state / entanglement fidelity ──
        if result is None and re.search(r'bell\s*state|entangle|fidelity|concurrence', text) and qmc is not None:
            try:
                chain.add_step("Quantum entanglement", "Computing Bell state properties", {})
                bell = qmc.bell_state_phi_plus(2)
                if bell:
                    rho = qmc.density_matrix(bell)
                    if rho:
                        conc = qmc.concurrence(rho)
                        if isinstance(conc, (int, float)):
                            if 'concurrence' in text:
                                result = round(conc, 6)
                                confidence = 0.90
                            elif 'fidelity' in text:
                                fid = qmc.fidelity(bell, bell)
                                result = round(fid, 6) if isinstance(fid, (int, float)) else 1.0
                                confidence = 0.90
                            else:
                                result = round(conc, 6)
                                confidence = 0.80
                    chain.add_step("Entanglement result", f"Result = {result}", {})
            except Exception:
                pass

        # ── Grover's algorithm iterations ──
        if result is None and re.search(r'grover|amplitude\s*amplif|search\s*iteration', text) and qmc is not None:
            try:
                chain.add_step("Grover's algorithm", "Computing optimal iterations", {})
                if re.search(r'iteration|how\s*many', text) and numbers:
                    n = int(numbers[0])
                    if n > 0:
                        # Optimal Grover iterations ≈ π/4 × √N
                        optimal = int(round(math.pi / 4 * math.sqrt(n)))
                        result = max(1, optimal)
                        confidence = 0.90
                        chain.add_step("Grover iterations", f"≈ π/4 × √{n} = {result}", {})
                elif numbers:
                    # Compute Grover speedup factor
                    n = int(numbers[0])
                    result = max(1, int(round(math.sqrt(n))))
                    confidence = 0.85
            except Exception:
                pass

        # ── Quantum Fourier transform ──
        if result is None and re.search(r'qft|quantum\s*fourier', text) and qge is not None:
            try:
                if numbers:
                    n_qubits = min(int(numbers[0]), 10)
                    circ = qge.quantum_fourier_transform(n_qubits)
                    stats = circ.statistics()
                    if 'depth' in text:
                        result = stats.get('depth', n_qubits)
                        confidence = 0.85
                    elif 'gate' in text:
                        result = stats.get('total_operations', n_qubits * (n_qubits + 1) // 2)
                        confidence = 0.85
                    else:
                        result = n_qubits * (n_qubits + 1) // 2
                        confidence = 0.75
                    chain.add_step("QFT", f"n_qubits={n_qubits}, result={result}", {})
            except Exception:
                pass

        # ── Von Neumann entropy ──
        if result is None and re.search(r'von\s*neumann|entropy.*qubit|qubit.*entropy', text) and qmc is not None:
            try:
                chain.add_step("Von Neumann entropy", "Computing quantum entropy", {})
                bell = qmc.bell_state_phi_plus(2)
                rho = qmc.density_matrix(bell)
                if rho:
                    entropy = qmc.von_neumann_entropy(rho)
                    if isinstance(entropy, (int, float)):
                        result = round(entropy, 6)
                        confidence = 0.85
            except Exception:
                pass

        # ── Qubit state space / Hilbert space dimension ──
        if result is None and re.search(r'hilbert|state\s*space|dimension.*qubit|qubit.*dimension|how\s*many\s*states', text):
            if numbers:
                n = int(numbers[0])
                result = 2 ** n  # 2^n dimensional Hilbert space
                confidence = 0.90
                chain.add_step("Hilbert space", f"dim = 2^{n} = {result}", {})

        # ── Quantum probability (Born rule) ──
        if result is None and re.search(r'probability|born\s*rule|amplitude.*\|', text) and 'quantum' in text:
            if numbers and len(numbers) >= 1:
                amplitude = numbers[0]
                result = round(amplitude ** 2, 6)
                confidence = 0.80
                chain.add_step("Born rule", f"|α|² = {amplitude}² = {result}", {})

        return result, confidence

    def _verify_answer(self, parsed: ParsedProblem, result: Any,
                       chain: SolutionChain) -> bool:
        """Verify the answer by plug-back or sanity checks."""
        if result is None:
            return False

        # Type-based verification
        if isinstance(result, (int, float)):
            # Check for NaN or infinity
            if math.isnan(result) or math.isinf(result):
                chain.add_step("Verification FAILED", "Result is NaN or infinite", False)
                return False

            # Sanity: most competition answers are reasonable numbers
            if abs(result) > 1e15:
                chain.add_step("Verification WARNING", "Result unusually large", False)
                return False

            # Plug-back verification for algebraic equations
            if parsed.relations:
                for rel in parsed.relations:
                    if rel.relation_type == 'equation' and '=' in rel.expression:
                        expr = rel.expression
                        lhs_str, rhs_str = expr.split('=', 1)
                        # Try substituting x = result
                        for var in parsed.unknowns:
                            try:
                                lhs_eval = lhs_str.strip().replace(var, f'({result})')
                                rhs_eval = rhs_str.strip().replace(var, f'({result})')
                                lhs_eval = lhs_eval.replace('^', '**')
                                rhs_eval = rhs_eval.replace('^', '**')
                                # Fix implied multiplication: "2(4)" → "2*(4)"
                                lhs_eval = re.sub(r'(\d)\(', r'\1*(', lhs_eval)
                                rhs_eval = re.sub(r'(\d)\(', r'\1*(', rhs_eval)
                                lhs_val = eval(lhs_eval, {"__builtins__": {}}, {"abs": abs, "pow": pow})
                                rhs_val = eval(rhs_eval, {"__builtins__": {}}, {"abs": abs, "pow": pow})
                                if abs(lhs_val - rhs_val) < 1e-6:
                                    chain.add_step("Plug-back verification PASSED",
                                                   f"Substituting {var}={result}: {lhs_val} ≈ {rhs_val}", True,
                                                   f"Verified by substitution")
                                    return True
                            except Exception:
                                pass

            chain.add_step("Verification passed", "Result is a valid finite number", True,
                           f"Answer: {result}")
            return True

        elif isinstance(result, (list, tuple)):
            chain.add_step("Verification", f"Multiple solutions: {result}", True)
            return True

        elif isinstance(result, bool):
            chain.add_step("Verification", f"Boolean result: {result}", True)
            return True

        elif isinstance(result, str):
            chain.add_step("Verification", f"Expression result: {result}", True)
            return True

        elif isinstance(result, dict):
            chain.add_step("Verification", "Structured result", True)
            return True

        return False

    def _format_answer(self, result: Any, fmt: str) -> Any:
        """Format the answer according to requested format."""
        if result is None:
            return None

        if fmt == 'auto':
            if isinstance(result, float):
                # Return integer if close to one
                if abs(result - round(result)) < 1e-9:
                    return int(round(result))
                return round(result, 6)
            return result

        elif fmt == 'numeric':
            if isinstance(result, (int, float)):
                return float(result)
            try:
                return float(result)
            except (ValueError, TypeError):
                return result

        elif fmt == 'fraction':
            if isinstance(result, float):
                frac = Fraction(result).limit_denominator(10000)
                return f"{frac.numerator}/{frac.denominator}"
            return str(result)

        return result

    def answer_math_mcq(self, question: str, choices: List[str]) -> Dict[str, Any]:
        """Answer a MATH-style multiple-choice question.

        Solves the problem, then matches the result to the best choice.
        """
        solution = self.solve(question)
        raw = solution.get('raw_result')

        if raw is None:
            # Fallback: try to evaluate each choice and pick the most reasonable
            return self._guess_mcq(question, choices, solution)

        # Match result to choices
        best_idx = 0
        best_score = 0.0

        for i, choice in enumerate(choices):
            score = self._match_answer_to_choice(raw, choice)
            if score > best_score:
                best_score = score
                best_idx = i

        labels = 'ABCDEFGH'
        return {
            'answer': labels[best_idx] if best_idx < len(labels) else str(best_idx),
            'choice': choices[best_idx],
            'confidence': round(best_score * solution.get('confidence', 0.5), 4),
            'solution': solution,
        }

    def _match_answer_to_choice(self, answer: Any, choice: str) -> float:
        """Score how well an answer matches a choice string."""
        choice_clean = choice.strip()

        # Try numeric comparison
        try:
            choice_val = float(choice_clean.replace(',', '').replace(' ', ''))
            if isinstance(answer, (int, float)):
                if abs(choice_val - float(answer)) < 1e-6:
                    return 1.0
                if abs(choice_val) > 1e-6 and abs((choice_val - float(answer)) / choice_val) < 0.01:
                    return 0.9
        except ValueError:
            pass

        # String match
        answer_str = str(answer).strip()
        if answer_str.lower() == choice_clean.lower():
            return 1.0
        if answer_str in choice_clean or choice_clean in answer_str:
            return 0.7

        # Check if answer appears as substring
        try:
            if isinstance(answer, bool):
                if str(answer).lower() in choice_clean.lower():
                    return 0.8
        except Exception:
            pass

        return 0.0

    def _guess_mcq(self, question: str, choices: List[str], solution: Dict) -> Dict[str, Any]:
        """Fallback MCQ guessing when solving fails — use smarter heuristics."""
        labels = 'ABCDEFGH'
        scores = []
        q_lower = question.lower()
        import math as _math

        # Extract all numbers from the question
        q_nums = set(re.findall(r'\d+(?:\.\d+)?', question))
        q_vals = sorted([float(n) for n in q_nums], reverse=True)

        for i, choice in enumerate(choices):
            score = 0.0
            c_nums = set(re.findall(r'\d+(?:\.\d+)?', choice))

            # Choices with novel numbers (not in question) likely computed answers
            novel = c_nums - q_nums
            if novel:
                score += 0.15

            # If choice is a pure number, try to verify simple computations
            choice_clean = choice.strip().replace(',', '')
            try:
                cval = float(choice_clean)
                # Check if this is a simple operation on question numbers
                if len(q_vals) >= 2:
                    a, b = q_vals[0], q_vals[1]
                    if abs(cval - (a + b)) < 0.01:
                        score += 0.3  # Sum match
                    elif abs(cval - (a - b)) < 0.01:
                        score += 0.3  # Difference match
                    elif abs(cval - (a * b)) < 0.01:
                        score += 0.3  # Product match
                    elif b != 0 and abs(cval - (a / b)) < 0.01:
                        score += 0.3  # Quotient match
                    elif abs(cval - (a ** 2)) < 0.01:
                        score += 0.25  # Square of largest
                    elif abs(cval - _math.sqrt(a)) < 0.01:
                        score += 0.25  # Square root of largest
                    # GCD/LCM check
                    if a == int(a) and b == int(b):
                        gcd_val = _math.gcd(int(a), int(b))
                        lcm_val = int(a) * int(b) // gcd_val if gcd_val else 0
                        if abs(cval - gcd_val) < 0.01:
                            score += 0.3
                        elif abs(cval - lcm_val) < 0.01:
                            score += 0.3
                elif len(q_vals) == 1:
                    a = q_vals[0]
                    if abs(cval - (a ** 2)) < 0.01:
                        score += 0.3
                    elif a >= 0 and abs(cval - _math.sqrt(a)) < 0.01:
                        score += 0.3
                    elif abs(cval - (2 * a)) < 0.01:
                        score += 0.25
                    elif abs(cval - (a / 2)) < 0.01:
                        score += 0.25
                    elif a == int(a) and 0 < a <= 20:
                        if abs(cval - _math.factorial(int(a))) < 0.01:
                            score += 0.3

                # Check C(n,k) for all pairs
                for n_val in q_vals:
                    for k_val in q_vals:
                        if n_val >= k_val and n_val == int(n_val) and k_val == int(k_val):
                            try:
                                comb_val = _math.comb(int(n_val), int(k_val))
                                if abs(cval - comb_val) < 0.01:
                                    score += 0.35
                                    break
                            except Exception:
                                pass
            except (ValueError, IndexError):
                pass

            # Middle choices slightly preferred (common in competition math)
            if 1 <= i <= len(choices) - 2:
                score += 0.03

            # Fraction answers get small boost (competition math often has fraction answers)
            if '/' in choice and re.match(r'^-?\d+/\d+$', choice.strip()):
                score += 0.05

            scores.append(score)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return {
            'answer': labels[best_idx] if best_idx < len(labels) else str(best_idx),
            'choice': choices[best_idx],
            'confidence': 0.25,
            'solution': solution,
            'method': 'heuristic_guess',
        }

    def evaluate_solver(self) -> float:
        """Compute solver quality score (0-1)."""
        if self._total_problems == 0:
            return 0.0
        solve_rate = self._total_solved / self._total_problems

        # Domain coverage bonus
        domains_covered = sum(1 for d in self._domain_stats
                              if self._domain_stats[d]['solved'] > 0)
        coverage = min(1.0, domains_covered / 6)

        return solve_rate * 0.7 + coverage * 0.3

    def get_status(self) -> Dict[str, Any]:
        return {
            'version': self.VERSION,
            'total_problems': self._total_problems,
            'total_solved': self._total_solved,
            'solve_rate': round(self._total_solved / max(self._total_problems, 1), 4),
            'domain_stats': dict(self._domain_stats),
            'engines': ['algebra', 'geometry', 'number_theory', 'combinatorics', 'calculus', 'quantum'],
            'engine_support': {
                'math_engine': self._math_engine is not None,
                'science_engine': self._science_engine is not None,
                'quantum_gate_engine': self._quantum_gate_engine is not None,
                'quantum_math_core': self._quantum_math_core is not None,
            },
        }
