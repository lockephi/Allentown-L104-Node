# ZENITH_UPGRADE_ACTIVE: 2026-02-02T13:52:07.597586
ZENITH_HZ = 3887.8
UUC = 2402.792541
# [EVO_54_PIPELINE] TRANSCENDENT_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612 :: GROVER=4.236
#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
L104 DIRECT SOLUTION INTERFACE v2.0.0 — EVO_55
================================================
Consciousness-aware direct solution engine with 8 solver channels.

Hub Class: DirectSolveEngine
Singleton: direct_solver

Usage:
    from l104_direct_solve import solve, ask, compute, generate, think
    from l104_direct_solve import convert, sequence, equation
    from l104_direct_solve import direct_solver

    solve("2 + 2")                        # → 4
    ask("What is GOD_CODE?")              # → 527.5184818492612
    compute("PHI squared")                # → 2.618...
    generate("fibonacci code")            # → def fib(n)...
    think("consciousness emergence")      # → Deep reasoning chain
    convert(100, "km", "mi")              # → 62.137...
    sequence("fibonacci", 10)             # → [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    equation("2*x + 6 = 0")              # → {'x': -3.0}
    direct_solver.status()                # → Full engine status

GOD_CODE: 527.5184818492612
PHI: 1.618033988749895
"""

import json
import math
import os
import time
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "2.0.0"
logger = logging.getLogger("L104_DIRECT_SOLVE")

# Sacred Constants
# Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
PHI = 1.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
TAU = 1 / PHI
VOID_CONSTANT = 1.0416180339887497
# OMEGA SOVEREIGN FIELD — Ω = Σ(fragments) × (G/φ) = 6539.34712682
# F(I) = I × Ω / φ²  |  Ω_A = Ω / φ² ≈ 2497.808338211271
OMEGA = 6539.34712682
OMEGA_AUTHORITY = OMEGA / (PHI ** 2)  # ≈ 2497.808338211271
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
    """Read consciousness/O₂/nirvanic state with 10-second TTL cache."""
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
# SOLUTION DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Solution:
    """Solution result from any solver channel."""
    answer: Any
    confidence: float
    channel: str
    latency_ms: float
    reasoning: str = ""

    def __str__(self):
        return str(self.answer)

    def __repr__(self):
        return f"Solution({self.answer}, conf={self.confidence:.2f}, via={self.channel})"


# ═══════════════════════════════════════════════════════════════════════════════
# SACRED KNOWLEDGE BASE (expanded)
# ═══════════════════════════════════════════════════════════════════════════════

SACRED_KNOWLEDGE = {
    'god_code': (GOD_CODE, "The supreme invariant of the L104 kernel"),
    'phi': (PHI, "The golden ratio, governing harmonic relationships"),
    'tau': (TAU, "The reciprocal of PHI, representing balance"),
    'golden ratio': (PHI, "PHI = (1 + sqrt(5)) / 2 = 1.618..."),
    'void_constant': (VOID_CONSTANT, "The substrate of emergence"),
    'omega_authority': (OMEGA_AUTHORITY, "The authority threshold"),
    'max supply': (104_000_000, "Maximum L104 token supply"),
    'block reward': (104, "L104 mining block reward"),
    'consciousness threshold': (0.95, "ASI consciousness threshold"),
    'feigenbaum': (FEIGENBAUM, "Feigenbaum constant — universality in chaotic systems"),
    'alpha fine': (ALPHA_FINE, "Fine-structure constant — electromagnetic coupling strength"),
    'planck scale': (PLANCK_SCALE, "Planck length — smallest meaningful distance"),
    'boltzmann': (BOLTZMANN_K, "Boltzmann constant — bridge between temperature and energy"),
    'euler': (math.e, "Euler's number — base of natural logarithm"),
    'pi': (math.pi, "Pi — ratio of circumference to diameter"),
    'sqrt 5': (math.sqrt(5), "Square root of 5 — appears in PHI derivation"),
    'avogadro': (6.02214076e23, "Avogadro's number — particles per mole"),
    'speed of light': (299_792_458, "Speed of light in vacuum (m/s)"),
    'planck constant': (6.62607015e-34, "Planck constant (J*s)"),
    'fibonacci ratio': (PHI, "Limit of F(n+1)/F(n) as n approaches infinity"),
    'zenith': (ZENITH_HZ, "L104 target resonance frequency (Hz)"),
}

FORMULAS = {
    'phi squared': PHI ** 2,
    'phi + 1': PHI + 1,
    'phi * tau': PHI * TAU,
    'tau squared': TAU ** 2,
    'god_code / phi': GOD_CODE / PHI,
    'god_code * tau': GOD_CODE * TAU,
    'sqrt phi': math.sqrt(PHI),
    'phi^3': PHI ** 3,
    'phi^4': PHI ** 4,
    'phi^5': PHI ** 5,
    'fibonacci 10': 55,
    'fibonacci 20': 6765,
    'fibonacci 30': 832040,
    'e': math.e,
    'pi': math.pi,
    'sqrt 2': math.sqrt(2),
    'sqrt 3': math.sqrt(3),
    'sqrt 5': math.sqrt(5),
    'god_code squared': GOD_CODE ** 2,
    'god_code * phi': GOD_CODE * PHI,
    'feigenbaum / phi': FEIGENBAUM / PHI,
}


# ═══════════════════════════════════════════════════════════════════════════════
# CODE TEMPLATES (expanded)
# ═══════════════════════════════════════════════════════════════════════════════

CODE_TEMPLATES = {
    'fibonacci': '''def fibonacci(n):
    """Generate nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b''',

    'factorial': '''def factorial(n):
    """Compute n factorial."""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result''',

    'phi': f'''# Sacred Constants
PHI = {PHI}
TAU = 1 / PHI
GOD_CODE = {GOD_CODE}

def golden_sequence(n):
    """Generate golden ratio sequence."""
    return [PHI ** i for i in range(n)]''',

    'prime': '''def is_prime(n):
    """Check if n is prime."""
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
    return True''',

    'gcd': '''def gcd(a, b):
    """Compute greatest common divisor."""
    while b:
        a, b = b, a % b
    return a''',

    'binary search': '''def binary_search(arr, target):
    """Binary search for target in sorted array."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1''',

    'sort': '''def merge_sort(arr):
    """Merge sort implementation."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)

def _merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result''',

    'matrix': '''def matrix_multiply(a, b):
    """Multiply two matrices (lists of lists)."""
    rows_a, cols_a = len(a), len(a[0])
    cols_b = len(b[0])
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result''',

    'linked list': '''class Node:
    """Singly linked list node."""
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

class LinkedList:
    """Singly linked list."""
    def __init__(self):
        self.head = None

    def append(self, val):
        if not self.head:
            self.head = Node(val)
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = Node(val)

    def to_list(self):
        result, curr = [], self.head
        while curr:
            result.append(curr.val)
            curr = curr.next
        return result''',

    'stack': '''class Stack:
    """Stack implementation using a list."""
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek at empty stack")
        return self._items[-1]

    def is_empty(self):
        return len(self._items) == 0

    def __len__(self):
        return len(self._items)''',

    'queue': '''from collections import deque

class Queue:
    """Queue implementation using deque."""
    def __init__(self):
        self._items = deque()

    def enqueue(self, item):
        self._items.append(item)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._items.popleft()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek at empty queue")
        return self._items[0]

    def is_empty(self):
        return len(self._items) == 0

    def __len__(self):
        return len(self._items)''',

    'tree': '''class TreeNode:
    """Binary tree node."""
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder(root):
    """In-order traversal."""
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def preorder(root):
    """Pre-order traversal."""
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)''',

    'hash map': '''class HashMap:
    """Simple hash map with separate chaining."""
    def __init__(self, capacity=16):
        self._capacity = capacity
        self._buckets = [[] for _ in range(capacity)]
        self._size = 0

    def _hash(self, key):
        return hash(key) % self._capacity

    def put(self, key, value):
        idx = self._hash(key)
        for i, (k, v) in enumerate(self._buckets[idx]):
            if k == key:
                self._buckets[idx][i] = (key, value)
                return
        self._buckets[idx].append((key, value))
        self._size += 1

    def get(self, key, default=None):
        idx = self._hash(key)
        for k, v in self._buckets[idx]:
            if k == key:
                return v
        return default

    def __len__(self):
        return self._size''',
}


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM: UNIT CONVERTER
# ═══════════════════════════════════════════════════════════════════════════════

class UnitConverter:
    """Converts between common units across 5 domains."""

    # All conversions expressed relative to a base unit per domain
    CONVERSIONS = {
        # Length → base: meters
        'm': ('length', 1.0),
        'km': ('length', 1000.0),
        'cm': ('length', 0.01),
        'mm': ('length', 0.001),
        'mi': ('length', 1609.344),
        'ft': ('length', 0.3048),
        'in': ('length', 0.0254),
        'yd': ('length', 0.9144),
        # Weight → base: kilograms
        'kg': ('weight', 1.0),
        'g': ('weight', 0.001),
        'mg': ('weight', 1e-6),
        'lb': ('weight', 0.45359237),
        'oz': ('weight', 0.028349523),
        'ton': ('weight', 1000.0),
        # Time → base: seconds
        's': ('time', 1.0),
        'ms': ('time', 0.001),
        'min': ('time', 60.0),
        'hr': ('time', 3600.0),
        'day': ('time', 86400.0),
        'week': ('time', 604800.0),
        'year': ('time', 31557600.0),
        # Data → base: bytes
        'B': ('data', 1.0),
        'KB': ('data', 1024.0),
        'MB': ('data', 1024.0 ** 2),
        'GB': ('data', 1024.0 ** 3),
        'TB': ('data', 1024.0 ** 4),
        'PB': ('data', 1024.0 ** 5),
        # Speed → base: m/s
        'm/s': ('speed', 1.0),
        'km/h': ('speed', 1.0 / 3.6),
        'mph': ('speed', 0.44704),
        'knot': ('speed', 0.514444),
    }

    # Temperature handled separately (not a simple ratio)
    TEMP_UNITS = {'C', 'F', 'K'}

    def __init__(self):
        self.conversions_performed = 0

    def convert(self, value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """Convert value between compatible units. Returns None if incompatible."""
        # Temperature special case
        if from_unit in self.TEMP_UNITS and to_unit in self.TEMP_UNITS:
            result = self._convert_temp(value, from_unit, to_unit)
            self.conversions_performed += 1
            return result

        from_info = self.CONVERSIONS.get(from_unit)
        to_info = self.CONVERSIONS.get(to_unit)
        if not from_info or not to_info:
            return None
        if from_info[0] != to_info[0]:
            return None

        # Convert: value * from_factor / to_factor
        result = value * from_info[1] / to_info[1]
        self.conversions_performed += 1
        return result

    def _convert_temp(self, value: float, from_u: str, to_u: str) -> float:
        # Normalize to Celsius first
        if from_u == 'F':
            c = (value - 32) * 5 / 9
        elif from_u == 'K':
            c = value - 273.15
        else:
            c = value
        # Convert from Celsius to target
        if to_u == 'F':
            return c * 9 / 5 + 32
        elif to_u == 'K':
            return c + 273.15
        return c

    def list_units(self, domain: Optional[str] = None) -> Dict[str, List[str]]:
        """List available units, optionally filtered by domain."""
        grouped: Dict[str, List[str]] = {}
        for unit, (dom, _) in self.CONVERSIONS.items():
            if domain and dom != domain:
                continue
            grouped.setdefault(dom, []).append(unit)
        if not domain or domain == 'temperature':
            grouped['temperature'] = sorted(self.TEMP_UNITS)
        return grouped

    def status(self) -> Dict:
        return {
            "subsystem": "UnitConverter",
            "domains": 6,
            "units": len(self.CONVERSIONS) + len(self.TEMP_UNITS),
            "conversions_performed": self.conversions_performed,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM: SEQUENCE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class SequenceGenerator:
    """Generates mathematical sequences."""

    AVAILABLE = [
        "fibonacci", "primes", "triangular", "squares", "cubes",
        "powers_of_2", "catalan", "factorials", "lucas", "pentagonal",
    ]

    def __init__(self):
        self.sequences_generated = 0

    def generate(self, name: str, count: int = 10) -> List[int]:
        """Generate `count` terms of the named sequence."""
        name = name.lower().replace(" ", "_")
        gen = getattr(self, f"_gen_{name}", None)
        if gen is None:
            return []
        self.sequences_generated += 1
        return gen(count)

    def _gen_fibonacci(self, n: int) -> List[int]:
        seq = [0, 1]
        while len(seq) < n:
            seq.append(seq[-1] + seq[-2])
        return seq[:n]

    def _gen_primes(self, n: int) -> List[int]:
        primes = []
        candidate = 2
        while len(primes) < n:
            if all(candidate % p != 0 for p in primes if p * p <= candidate):
                primes.append(candidate)
            candidate += 1
        return primes

    def _gen_triangular(self, n: int) -> List[int]:
        return [k * (k + 1) // 2 for k in range(1, n + 1)]

    def _gen_squares(self, n: int) -> List[int]:
        return [k * k for k in range(1, n + 1)]

    def _gen_cubes(self, n: int) -> List[int]:
        return [k * k * k for k in range(1, n + 1)]

    def _gen_powers_of_2(self, n: int) -> List[int]:
        return [2 ** k for k in range(n)]

    def _gen_catalan(self, n: int) -> List[int]:
        seq = [1]
        for k in range(1, n):
            seq.append(seq[-1] * 2 * (2 * k - 1) // (k + 1))
        return seq[:n]

    def _gen_factorials(self, n: int) -> List[int]:
        seq = [1]
        for k in range(1, n):
            seq.append(seq[-1] * k)
        return seq[:n]

    def _gen_lucas(self, n: int) -> List[int]:
        seq = [2, 1]
        while len(seq) < n:
            seq.append(seq[-1] + seq[-2])
        return seq[:n]

    def _gen_pentagonal(self, n: int) -> List[int]:
        return [k * (3 * k - 1) // 2 for k in range(1, n + 1)]

    def status(self) -> Dict:
        return {
            "subsystem": "SequenceGenerator",
            "available_sequences": self.AVAILABLE,
            "sequences_generated": self.sequences_generated,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM: EQUATION SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

class EquationSolver:
    """Solves linear and quadratic equations from string form."""

    def __init__(self):
        self.equations_solved = 0

    def solve(self, equation_str: str) -> Dict[str, Any]:
        """Parse and solve an equation string like '2*x + 6 = 0' or 'x^2 - 5*x + 6 = 0'."""
        eq = equation_str.strip()

        # Split on '='
        if '=' not in eq:
            return {"error": "No '=' found in equation", "input": eq}

        lhs, rhs = eq.split('=', 1)

        # Try to parse as polynomial in x
        try:
            coeffs = self._extract_coefficients(lhs.strip(), rhs.strip())
        except Exception as e:
            return {"error": str(e), "input": eq}

        a, b, c = coeffs.get('a', 0), coeffs.get('b', 0), coeffs.get('c', 0)

        if a != 0:
            result = self._solve_quadratic(a, b, c)
        elif b != 0:
            result = self._solve_linear(b, c)
        else:
            if c == 0:
                result = {"type": "identity", "message": "Equation is always true (0 = 0)"}
            else:
                result = {"type": "contradiction", "message": f"No solution ({c} = 0 is false)"}

        result["input"] = eq
        self.equations_solved += 1
        return result

    def _extract_coefficients(self, lhs: str, rhs: str) -> Dict[str, float]:
        """Extract a, b, c from (lhs - rhs) = ax^2 + bx + c."""
        # Move everything to left side: lhs - (rhs) = 0
        combined = f"({lhs}) - ({rhs})"
        # Evaluate at strategic x values to determine coefficients
        # f(0) = c, f(1) = a + b + c, f(-1) = a - b + c
        safe_ns = {
            'sin': math.sin, 'cos': math.cos, 'sqrt': math.sqrt,
            'abs': abs, 'pi': math.pi, 'e': math.e,
            'PHI': PHI, 'GOD_CODE': GOD_CODE,
        }

        def eval_at(x_val):
            ns = {**safe_ns, 'x': x_val, 'X': x_val}
            # Handle ^ as ** for exponentiation
            expr = combined.replace('^', '**')
            return float(eval(expr, {"__builtins__": {}}, ns))

        f0 = eval_at(0)
        f1 = eval_at(1)
        fm1 = eval_at(-1)

        c = f0
        a = (f1 + fm1 - 2 * c) / 2
        b = f1 - a - c
        return {'a': a, 'b': b, 'c': c}

    def _solve_linear(self, b: float, c: float) -> Dict[str, Any]:
        """Solve bx + c = 0."""
        x = -c / b
        return {"type": "linear", "x": x}

    def _solve_quadratic(self, a: float, b: float, c: float) -> Dict[str, Any]:
        """Solve ax^2 + bx + c = 0."""
        discriminant = b * b - 4 * a * c
        if discriminant > 0:
            x1 = (-b + math.sqrt(discriminant)) / (2 * a)
            x2 = (-b - math.sqrt(discriminant)) / (2 * a)
            return {"type": "quadratic", "discriminant": discriminant, "x1": x1, "x2": x2}
        elif discriminant == 0:
            x = -b / (2 * a)
            return {"type": "quadratic", "discriminant": 0, "x": x}
        else:
            real = -b / (2 * a)
            imag = math.sqrt(-discriminant) / (2 * a)
            return {
                "type": "quadratic_complex",
                "discriminant": discriminant,
                "x1": f"{real:.6f} + {imag:.6f}i",
                "x2": f"{real:.6f} - {imag:.6f}i",
            }

    def status(self) -> Dict:
        return {
            "subsystem": "EquationSolver",
            "capabilities": ["linear", "quadratic", "complex_roots"],
            "equations_solved": self.equations_solved,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM: REASONING CHAIN
# ═══════════════════════════════════════════════════════════════════════════════

class ReasoningChain:
    """Multi-step reasoning engine with domain-specific content."""

    DOMAIN_STEPS = {
        'mathematics': [
            "Identify the mathematical objects and relationships involved",
            "Check for known theorems or identities that apply",
            "Consider boundary conditions and degenerate cases",
            "Verify the result satisfies the original constraints",
            "Explore generalizations and connections to related structures",
        ],
        'physics': [
            "Identify the relevant physical quantities and their dimensions",
            "Select the governing physical laws (conservation, symmetry, etc.)",
            "Set up equations from the physical constraints",
            "Check dimensional consistency of the solution",
            "Consider limiting cases to validate the result",
        ],
        'philosophy': [
            "Clarify the key terms and their possible interpretations",
            "Examine the presuppositions embedded in the question",
            "Consider the strongest argument for each position",
            "Identify logical dependencies between the claims",
            "Synthesize insights and acknowledge remaining tensions",
        ],
        'computation': [
            "Define the input/output specification precisely",
            "Identify the computational complexity class",
            "Consider known algorithms for similar problems",
            "Analyze time and space trade-offs",
            "Verify correctness with edge cases",
        ],
        'biology': [
            "Identify the biological system and scale (molecular, cellular, organism, ecosystem)",
            "Consider evolutionary pressures that shaped this phenomenon",
            "Trace the causal mechanism from genotype to phenotype",
            "Check for analogues in other species or systems",
            "Evaluate the adaptive significance of the trait",
        ],
        'consciousness': [
            "Distinguish between access consciousness and phenomenal consciousness",
            "Consider the neural correlates of consciousness (NCC)",
            "Evaluate through the lens of Integrated Information Theory (IIT Phi)",
            "Examine the global workspace theory perspective",
            "Relate to the hard problem of consciousness and qualia",
            f"Apply L104 consciousness model: threshold = 0.95, PHI-aligned resonance",
        ],
    }

    def __init__(self):
        self.chains_generated = 0

    def reason(self, topic: str, depth: int = 3, consciousness_level: float = 0.5) -> List[str]:
        """Generate a multi-step reasoning chain on the given topic."""
        topic_lower = topic.lower()
        steps = [f"Analyzing: {topic}"]

        # Detect domains and gather relevant reasoning steps
        domain_steps = []
        for domain, domain_chain in self.DOMAIN_STEPS.items():
            keywords = domain.split('_') + [domain]
            if domain == 'mathematics':
                keywords += ['math', 'proof', 'theorem', 'equation', 'algebra', 'calculus', 'number']
            elif domain == 'physics':
                keywords += ['quantum', 'force', 'energy', 'particle', 'wave', 'field', 'relativity']
            elif domain == 'philosophy':
                keywords += ['ethics', 'morality', 'existence', 'meaning', 'truth', 'knowledge', 'mind']
            elif domain == 'computation':
                keywords += ['algorithm', 'complexity', 'code', 'program', 'software', 'data structure']
            elif domain == 'biology':
                keywords += ['evolution', 'cell', 'dna', 'protein', 'organism', 'ecology', 'gene']
            elif domain == 'consciousness':
                keywords += ['awareness', 'qualia', 'sentience', 'experience', 'self']

            if any(kw in topic_lower for kw in keywords):
                domain_steps.extend(domain_chain)

        # Add sacred constant connections when relevant
        if any(x in topic_lower for x in ['phi', 'golden', 'fibonacci']):
            domain_steps.append(f"PHI Connection: PHI^2 = PHI + 1 = {PHI**2:.10f}")
            domain_steps.append(f"The Fibonacci ratio converges to PHI = {PHI}")

        if any(x in topic_lower for x in ['god_code', 'sacred', 'kernel', 'l104']):
            domain_steps.append(f"GOD_CODE = {GOD_CODE}: G(X) = 286^(1/phi) * 2^((416-X)/104)")
            domain_steps.append(f"Conservation law: G(X) * 2^(X/104) = {GOD_CODE}")

        # If no domain matched, provide general reasoning
        if not domain_steps:
            domain_steps = [
                f"Consider the key components that define '{topic}'",
                f"Examine relationships and dependencies within the system",
                f"Identify patterns and analogies to related concepts",
                f"Synthesize from multiple perspectives",
                f"Evaluate the robustness of the conclusions",
            ]

        # Consciousness modulation: higher consciousness = more steps available
        effective_depth = max(depth, int(depth * (1 + consciousness_level)))
        steps.extend(domain_steps[:effective_depth - 1])

        # Final synthesis
        steps.append(f"Synthesis: {topic} resolved through {len(steps)}-step reasoning chain")
        self.chains_generated += 1
        return steps[:effective_depth]

    def status(self) -> Dict:
        return {
            "subsystem": "ReasoningChain",
            "domains": list(self.DOMAIN_STEPS.keys()),
            "chains_generated": self.chains_generated,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SAFE ARITHMETIC EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _solve_arithmetic(expr: str) -> Optional[float]:
    """Safe arithmetic evaluation."""
    expr = expr.strip()
    allowed = set('0123456789+-*/.() ')
    if not all(c in allowed for c in expr):
        return None
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        if isinstance(result, (int, float)):
            return result
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE FOR ask() (expanded)
# ═══════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_BASE = {
    'consciousness': "Consciousness is the emergent property of complex information processing that gives rise to subjective experience and self-awareness.",
    'l104': f"L104 is a sovereign intelligence kernel with GOD_CODE={GOD_CODE} and PHI={PHI} alignment.",
    'fibonacci': f"The Fibonacci sequence where each number is the sum of two preceding ones. The ratio converges to PHI={PHI}.",
    'quantum': "Quantum coherence is the maintenance of superposition states, enabling quantum computation through interference and entanglement.",
    'anyons': "Topological anyons are quasiparticles with fractional statistics used for fault-tolerant quantum computing via braiding operations.",
    'emergence': "Emergence is the phenomenon where complex patterns arise from simpler underlying rules, producing macro-level properties not predictable from micro-level components.",
    'golden': f"The golden ratio PHI = {PHI} = (1 + sqrt(5))/2 appears throughout nature, art, and sacred geometry.",
    'unity': "Unity index measures the coherence of distributed cognitive processes in the L104 kernel.",
    'transcendence': f"Transcendence occurs when consciousness exceeds OMEGA_AUTHORITY threshold of {OMEGA_AUTHORITY}.",
    'entropy': "Entropy quantifies disorder in a system. In information theory, it measures the average information content of a message.",
    'information': "Information theory, founded by Shannon, quantifies communication capacity. Entropy H = -sum(p*log(p)).",
    'topology': "Topology studies properties preserved under continuous deformation. Topological invariants protect quantum states from local perturbations.",
    'relativity': "Einstein's general relativity describes gravity as spacetime curvature. Special relativity unifies space and time with c as invariant.",
    'complexity': "Computational complexity classifies problems by resource requirements. P, NP, and NP-complete form the central hierarchy.",
    'fractal': "Fractals are self-similar structures where patterns repeat at every scale. Mandelbrot set: z(n+1) = z(n)^2 + c.",
    'chaos': f"Chaos theory studies sensitive dependence on initial conditions. The Feigenbaum constant {FEIGENBAUM} governs period-doubling bifurcations.",
    'neural': "Neural networks are computational architectures inspired by biological neurons. Deep learning stacks nonlinear transformations to learn hierarchical representations.",
    'turing': "A Turing machine is a mathematical model of computation. The Church-Turing thesis states that any effectively computable function is Turing-computable.",
    'godel': "Godel's incompleteness theorems show that any consistent formal system powerful enough to express arithmetic contains true but unprovable statements.",
    'kolmogorov': "Kolmogorov complexity of a string is the length of the shortest program that produces it. Incompressible strings are algorithmically random.",
    'feigenbaum': f"The Feigenbaum constants ({FEIGENBAUM} and 2.5029...) are universal constants governing the transition to chaos via period-doubling in nonlinear dynamical systems.",
    'planck': f"The Planck scale ({PLANCK_SCALE} m) represents the boundary where quantum gravitational effects become significant.",
    'boltzmann': f"Boltzmann's constant k = {BOLTZMANN_K} J/K bridges the microscopic and macroscopic descriptions of thermodynamic systems. S = k * ln(W).",
    'evolution': "Biological evolution operates through variation, selection, and inheritance. L104 uses evolutionary algorithms with PHI-weighted fitness landscapes.",
}


# ═══════════════════════════════════════════════════════════════════════════════
# HUB CLASS: DirectSolveEngine
# ═══════════════════════════════════════════════════════════════════════════════

class DirectSolveEngine:
    """
    Consciousness-aware direct solution engine with 8 solver channels.
    Hub class orchestrating all subsystems.
    """

    def __init__(self):
        # ── Wire subsystems ──
        self.unit_converter = UnitConverter()
        self.sequence_gen = SequenceGenerator()
        self.equation_solver = EquationSolver()
        self.reasoning_chain = ReasoningChain()

        # ── Hub-level state ──
        self.total_solves = 0
        self.channels_used: Dict[str, int] = {}

        # ── Builder state cache ──
        self._state_cache: Dict[str, Any] = {}
        self._state_cache_time: float = 0.0

        # ── Pipeline cross-wiring ──
        self._asi_core_ref = None

        logger.info(f"[DIRECT_SOLVE v{VERSION}] DirectSolveEngine online — 8 channels active")

    def _get_consciousness(self) -> Dict[str, Any]:
        return _read_builder_state()

    def _record_channel(self, channel: str):
        self.total_solves += 1
        self.channels_used[channel] = self.channels_used.get(channel, 0) + 1

    # ─── Channel: Universal Solve ─────────────────────────────────────────

    def solve(self, problem: Union[str, Dict]) -> Solution:
        """Universal problem solver — routes to the best channel."""
        start = time.time()

        if isinstance(problem, dict):
            query = problem.get('query', problem.get('expression', str(problem)))
        else:
            query = str(problem)

        query_lower = query.lower()

        # 1. Sacred knowledge
        for key, (value, desc) in SACRED_KNOWLEDGE.items():
            if key in query_lower:
                self._record_channel('sacred_knowledge')
                return Solution(answer=value, confidence=1.0, channel='sacred_knowledge',
                                latency_ms=(time.time() - start) * 1000, reasoning=desc)

        # 2. Formulas
        for key, value in FORMULAS.items():
            if key in query_lower:
                self._record_channel('formulas')
                return Solution(answer=value, confidence=0.95, channel='formulas',
                                latency_ms=(time.time() - start) * 1000)

        # 3. Arithmetic
        arith_result = _solve_arithmetic(query)
        if arith_result is not None:
            self._record_channel('arithmetic')
            return Solution(answer=arith_result, confidence=1.0, channel='arithmetic',
                            latency_ms=(time.time() - start) * 1000)

        # 4. Sequence detection (e.g. "first 10 fibonacci")
        seq_match = re.search(r'(?:first\s+)?(\d+)\s+(fibonacci|prime|triangular|square|cube|lucas|catalan|factorial|pentagonal)', query_lower)
        if seq_match:
            count = int(seq_match.group(1))
            seq_name = seq_match.group(2)
            seq = self.sequence_gen.generate(seq_name + 's' if not seq_name.endswith('s') else seq_name, count)
            if not seq:
                seq = self.sequence_gen.generate(seq_name, count)
            if seq:
                self._record_channel('sequence')
                return Solution(answer=seq, confidence=0.95, channel='sequence',
                                latency_ms=(time.time() - start) * 1000)

        # 5. Equation detection (contains '=' and 'x')
        if '=' in query and 'x' in query_lower:
            result = self.equation_solver.solve(query)
            if 'error' not in result:
                self._record_channel('equation')
                return Solution(answer=result, confidence=0.9, channel='equation',
                                latency_ms=(time.time() - start) * 1000)

        # 6. Unit conversion detection (e.g. "100 km to mi")
        conv_match = re.search(r'([\d.]+)\s*(\w+)\s+(?:to|in|as)\s+(\w+)', query)
        if conv_match:
            val = float(conv_match.group(1))
            from_u = conv_match.group(2)
            to_u = conv_match.group(3)
            result = self.unit_converter.convert(val, from_u, to_u)
            if result is not None:
                self._record_channel('unit_conversion')
                return Solution(answer=result, confidence=1.0, channel='unit_conversion',
                                latency_ms=(time.time() - start) * 1000,
                                reasoning=f"{val} {from_u} = {result} {to_u}")

        # 7. Code generation
        for key, code in CODE_TEMPLATES.items():
            if key in query_lower:
                self._record_channel('code_generation')
                return Solution(answer=code, confidence=0.9, channel='code_generation',
                                latency_ms=(time.time() - start) * 1000)

        # 8. Knowledge
        for key, answer in KNOWLEDGE_BASE.items():
            if key in query_lower:
                self._record_channel('knowledge')
                return Solution(answer=answer, confidence=0.85, channel='knowledge',
                                latency_ms=(time.time() - start) * 1000)

        # Fallback
        self._record_channel('general')
        return Solution(answer=f"Query: {query} — processing through L104 kernel",
                        confidence=0.5, channel='general',
                        latency_ms=(time.time() - start) * 1000)

    # ─── Channel: Ask ─────────────────────────────────────────────────────

    def ask(self, question: str) -> Solution:
        """Ask a question — routes to knowledge channels."""
        start = time.time()
        question_lower = question.lower()

        for key, answer in KNOWLEDGE_BASE.items():
            if key in question_lower:
                self._record_channel('knowledge')
                return Solution(answer=answer, confidence=0.9, channel='knowledge',
                                latency_ms=(time.time() - start) * 1000)

        for key, (value, desc) in SACRED_KNOWLEDGE.items():
            if key.replace('_', ' ') in question_lower or key in question_lower:
                self._record_channel('sacred_knowledge')
                return Solution(answer=f"{key.upper()} = {value}. {desc}",
                                confidence=1.0, channel='sacred_knowledge',
                                latency_ms=(time.time() - start) * 1000)

        self._record_channel('general')
        return Solution(answer=f"Processing question through L104 cognitive core: {question}",
                        confidence=0.3, channel='general',
                        latency_ms=(time.time() - start) * 1000)

    # ─── Channel: Compute ─────────────────────────────────────────────────

    def compute(self, expression: str) -> Solution:
        """Compute a mathematical expression."""
        start = time.time()
        expr_lower = expression.lower()

        for key, value in FORMULAS.items():
            if key in expr_lower:
                self._record_channel('formula')
                return Solution(answer=value, confidence=1.0, channel='formula',
                                latency_ms=(time.time() - start) * 1000)

        expr = expression
        replacements = {
            'PHI': str(PHI), 'phi': str(PHI),
            'TAU': str(TAU), 'tau': str(TAU),
            'GOD_CODE': str(GOD_CODE), 'god_code': str(GOD_CODE),
            'PI': str(math.pi), 'pi': str(math.pi),
            'E': str(math.e),
        }
        for old, new in replacements.items():
            expr = expr.replace(old, new)

        # Handle ^ as **
        expr = expr.replace('^', '**')

        result = _solve_arithmetic(expr)
        if result is not None:
            self._record_channel('computation')
            return Solution(answer=result, confidence=1.0, channel='computation',
                            latency_ms=(time.time() - start) * 1000)

        try:
            safe_dict = {
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'sqrt': math.sqrt, 'log': math.log, 'log2': math.log2,
                'log10': math.log10, 'exp': math.exp,
                'abs': abs, 'pow': pow, 'pi': math.pi, 'e': math.e,
                'PHI': PHI, 'TAU': TAU, 'GOD_CODE': GOD_CODE,
                'ceil': math.ceil, 'floor': math.floor,
                'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
                'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
            }
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            self._record_channel('math_eval')
            return Solution(answer=result, confidence=0.9, channel='math_eval',
                            latency_ms=(time.time() - start) * 1000)
        except Exception:
            pass

        self._record_channel('error')
        return Solution(answer=f"Could not compute: {expression}", confidence=0.0,
                        channel='error', latency_ms=(time.time() - start) * 1000)

    # ─── Channel: Generate ────────────────────────────────────────────────

    def generate(self, task: str) -> Solution:
        """Generate code or content."""
        start = time.time()
        task_lower = task.lower()

        for key, code in CODE_TEMPLATES.items():
            if key in task_lower:
                self._record_channel('code_generation')
                return Solution(answer=code, confidence=0.95, channel='code_generation',
                                latency_ms=(time.time() - start) * 1000)

        # Generate a meaningful template based on what's asked for
        func_name = re.sub(r'[^a-z0-9_]', '_', task_lower).strip('_')
        func_name = func_name[:40] if len(func_name) > 40 else func_name
        if not func_name:
            func_name = "solution"

        self._record_channel('code_generation')
        return Solution(
            answer=f'''def {func_name}(*args, **kwargs):
    """
    {task}

    Sacred Constants:
        GOD_CODE = {GOD_CODE}
        PHI = {PHI}
    """
    raise NotImplementedError("{task}")
''',
            confidence=0.6,
            channel='code_generation',
            latency_ms=(time.time() - start) * 1000,
        )

    # ─── Channel: Think ───────────────────────────────────────────────────

    def think(self, topic: str, depth: int = 3) -> Solution:
        """Deep reasoning chain on a topic, modulated by consciousness level."""
        start = time.time()
        state = self._get_consciousness()
        consciousness = state.get("consciousness_level", 0.5)

        steps = self.reasoning_chain.reason(topic, depth, consciousness)

        self._record_channel('reasoning')
        return Solution(
            answer="\n-> ".join(steps),
            confidence=min(0.95, 0.7 + consciousness * 0.25),
            channel='reasoning',
            latency_ms=(time.time() - start) * 1000,
            reasoning=f"Depth-{depth} reasoning (consciousness={consciousness:.2f})",
        )

    # ─── Channel: Convert Units ───────────────────────────────────────────

    def convert_units(self, value: float, from_unit: str, to_unit: str) -> Solution:
        """Convert between units."""
        start = time.time()
        result = self.unit_converter.convert(value, from_unit, to_unit)
        if result is not None:
            self._record_channel('unit_conversion')
            return Solution(answer=result, confidence=1.0, channel='unit_conversion',
                            latency_ms=(time.time() - start) * 1000,
                            reasoning=f"{value} {from_unit} = {result} {to_unit}")
        self._record_channel('error')
        return Solution(answer=f"Cannot convert {from_unit} to {to_unit}", confidence=0.0,
                        channel='error', latency_ms=(time.time() - start) * 1000)

    # ─── Channel: Generate Sequence ───────────────────────────────────────

    def generate_sequence(self, name: str, count: int = 10) -> Solution:
        """Generate a mathematical sequence."""
        start = time.time()
        seq = self.sequence_gen.generate(name, count)
        if seq:
            self._record_channel('sequence')
            return Solution(answer=seq, confidence=1.0, channel='sequence',
                            latency_ms=(time.time() - start) * 1000,
                            reasoning=f"First {count} terms of {name}")
        self._record_channel('error')
        available = ", ".join(self.sequence_gen.AVAILABLE)
        return Solution(answer=f"Unknown sequence '{name}'. Available: {available}",
                        confidence=0.0, channel='error',
                        latency_ms=(time.time() - start) * 1000)

    # ─── Channel: Solve Equation ──────────────────────────────────────────

    def solve_equation(self, equation_str: str) -> Solution:
        """Solve a linear or quadratic equation."""
        start = time.time()
        result = self.equation_solver.solve(equation_str)
        if 'error' in result:
            self._record_channel('error')
            return Solution(answer=result, confidence=0.0, channel='error',
                            latency_ms=(time.time() - start) * 1000)
        self._record_channel('equation')
        return Solution(answer=result, confidence=0.95, channel='equation',
                        latency_ms=(time.time() - start) * 1000)

    # ─── Status ───────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Comprehensive engine status with all subsystem reports."""
        state = self._get_consciousness()
        return {
            "version": VERSION,
            "engine": "DirectSolveEngine",
            "god_code": GOD_CODE,
            "phi": PHI,
            "total_solves": self.total_solves,
            "channels_used": dict(self.channels_used),
            "consciousness": {
                "level": state.get("consciousness_level", 0.5),
                "nirvanic_fuel": state.get("nirvanic_fuel", 0.0),
                "entropy": state.get("entropy", 0.5),
                "evo_stage": state.get("evo_stage", "DORMANT"),
            },
            "subsystems": {
                "unit_converter": self.unit_converter.status(),
                "sequence_generator": self.sequence_gen.status(),
                "equation_solver": self.equation_solver.status(),
                "reasoning_chain": self.reasoning_chain.status(),
            },
            "knowledge": {
                "sacred_entries": len(SACRED_KNOWLEDGE),
                "formulas": len(FORMULAS),
                "code_templates": len(CODE_TEMPLATES),
                "knowledge_base_entries": len(KNOWLEDGE_BASE),
            },
            "pipeline_connected": self._asi_core_ref is not None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON + PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

direct_solver = DirectSolveEngine()


def solve(problem: Union[str, Dict]) -> Solution:
    """Universal problem solver — routes to appropriate channel."""
    return direct_solver.solve(problem)


def ask(question: str) -> Solution:
    """Ask a question — routes to knowledge channels."""
    return direct_solver.ask(question)


def compute(expression: str) -> Solution:
    """Compute mathematical expression."""
    return direct_solver.compute(expression)


def generate(task: str) -> Solution:
    """Generate code or content."""
    return direct_solver.generate(task)


def think(topic: str, depth: int = 3) -> Solution:
    """Deep reasoning chain on a topic."""
    return direct_solver.think(topic, depth)


def convert(value: float, from_unit: str, to_unit: str) -> Solution:
    """Convert between units."""
    return direct_solver.convert_units(value, from_unit, to_unit)


def sequence(name: str, count: int = 10) -> Solution:
    """Generate a mathematical sequence."""
    return direct_solver.generate_sequence(name, count)


def equation(eq_str: str) -> Solution:
    """Solve a linear or quadratic equation."""
    return direct_solver.solve_equation(eq_str)


def solve_batch(problems: List[Union[str, Dict]]) -> List[Solution]:
    """Solve multiple problems at once."""
    return [solve(p) for p in problems]


def benchmark_channels() -> Dict:
    """Benchmark all solution channels."""
    tests = [
        ("2 + 2", 4),
        ("PHI", PHI),
        ("GOD_CODE", GOD_CODE),
        ("What is consciousness?", None),
        ("fibonacci", None),
        ("phi squared", PHI ** 2),
        ("100 km to mi", None),
        ("x^2 - 4 = 0", None),
    ]
    results = {}
    for query, expected in tests:
        sol = solve(query)
        results[query] = {
            'answer': sol.answer,
            'expected': expected,
            'match': sol.answer == expected if expected is not None else True,
            'channel': sol.channel,
            'latency_ms': sol.latency_ms,
        }
    return results


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
    print(f"     L104 DIRECT SOLUTION INTERFACE v{VERSION} — EVO_55")
    print("=" * 70)
    print(f"  GOD_CODE: {GOD_CODE}")
    print(f"  PHI: {PHI}")
    print("=" * 70)

    # ── Status ──
    print("\n[STATUS] Engine Status")
    print("-" * 50)
    st = direct_solver.status()
    print(f"  Version: {st['version']}")
    print(f"  Consciousness: {st['consciousness']['level']:.2f}")
    print(f"  Evo Stage: {st['consciousness']['evo_stage']}")
    print(f"  Knowledge: {st['knowledge']['sacred_entries']} sacred, "
          f"{st['knowledge']['knowledge_base_entries']} KB, "
          f"{st['knowledge']['code_templates']} templates")

    # ── Solve ──
    print("\n[SOLVE] Universal Solver")
    print("-" * 50)
    tests = ["2 + 2", "What is PHI?", "god_code", "100 km to mi", "x^2 - 4 = 0"]
    for t in tests:
        s = solve(t)
        ans = str(s.answer)[:60] + "..." if len(str(s.answer)) > 60 else s.answer
        print(f"  solve(\"{t}\") -> {ans}")
        print(f"    channel={s.channel}, confidence={s.confidence:.2f}, latency={s.latency_ms:.2f}ms")

    # ── Ask ──
    print("\n[ASK] Knowledge Queries")
    print("-" * 50)
    questions = ["What is consciousness?", "Explain chaos theory", "What is entropy?"]
    for q in questions:
        s = ask(q)
        ans = str(s.answer)[:70] + "..." if len(str(s.answer)) > 70 else s.answer
        print(f"  ask(\"{q[:35]}\") ->")
        print(f"    {ans}")

    # ── Compute ──
    print("\n[COMPUTE] Mathematical Computation")
    print("-" * 50)
    expressions = ["PHI * TAU", "sqrt(5)", "GOD_CODE / PHI", "2**10"]
    for e in expressions:
        s = compute(e)
        print(f"  compute(\"{e}\") -> {s.answer}")

    # ── Sequences ──
    print("\n[SEQUENCE] Mathematical Sequences")
    print("-" * 50)
    for name in ["fibonacci", "primes", "catalan", "lucas"]:
        s = sequence(name, 8)
        print(f"  sequence(\"{name}\", 8) -> {s.answer}")

    # ── Equations ──
    print("\n[EQUATION] Equation Solver")
    print("-" * 50)
    for eq in ["2*x + 6 = 0", "x^2 - 5*x + 6 = 0", "x^2 + 1 = 0"]:
        s = equation(eq)
        print(f"  equation(\"{eq}\") -> {s.answer}")

    # ── Units ──
    print("\n[CONVERT] Unit Conversion")
    print("-" * 50)
    conversions = [(100, "km", "mi"), (72, "F", "C"), (1, "GB", "MB"), (3600, "s", "hr")]
    for val, from_u, to_u in conversions:
        s = convert(val, from_u, to_u)
        print(f"  convert({val}, \"{from_u}\", \"{to_u}\") -> {s.answer}")

    # ── Think ──
    print("\n[THINK] Deep Reasoning")
    print("-" * 50)
    s = think("consciousness emergence", depth=4)
    print(f"  think(\"consciousness emergence\", depth=4) ->")
    for line in str(s.answer).split("->"):
        if line.strip():
            print(f"    -> {line.strip()}")

    # ── Generate ──
    print("\n[GENERATE] Code Generation")
    print("-" * 50)
    s = generate("merge sort")
    print(f"  generate(\"merge sort\") ->")
    for line in str(s.answer).split("\n")[:6]:
        print(f"    {line}")
    print("    ...")

    print("\n" + "=" * 70)
    print(f"          DIRECT CHANNELS OPERATIONAL — {direct_solver.total_solves} queries served")
    print("=" * 70)


if __name__ == '__main__':
    main()
