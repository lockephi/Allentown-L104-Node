#!/usr/bin/env python3
"""
L104 Math Engine — Layer 10: HYPERDIMENSIONAL COMPUTING
══════════════════════════════════════════════════════════════════════════════════
Vector Symbolic Architecture (VSA): 10,000-dimensional hypervectors, holographic
reduced representations, sparse distributed memory, resonator networks,
sequence encoding, and classification.

Consolidates: l104_hyperdimensional_compute.py, l104_hyperdimensional_computing.py.

Import:
  from l104_math_engine.hyperdimensional import HyperdimensionalCompute
"""

import math
import hashlib
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from .constants import (
    PHI, PHI_CONJUGATE, GOD_CODE, PI, VOID_CONSTANT,
    OMEGA, OMEGA_AUTHORITY,
    primal_calculus, resolve_non_dual_logic,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_DIMENSION = 10_000
DENSITY = 0.01          # Sparse vector density

# Try numpy for performance; fall back to pure-Python
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR TYPE
# ═══════════════════════════════════════════════════════════════════════════════

class VectorType(Enum):
    BIPOLAR = "bipolar"         # {-1, +1}
    REAL = "real"               # Continuous ℝ
    SPARSE = "sparse"           # Mostly zeros
    HOLOGRAPHIC = "holographic" # Complex-valued
    BLOCK = "block"             # Block-structured


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERVECTOR — Core data structure
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Hypervector:
    """
    High-dimensional vector: the fundamental unit of hyperdimensional computing.
    Operations: bind (⊗), bundle (+), permute (ρ), similarity (δ).
    """
    data: list
    dimension: int = DEFAULT_DIMENSION
    vtype: VectorType = VectorType.BIPOLAR
    label: str = ""

    @staticmethod
    def random_bipolar(dim: int = DEFAULT_DIMENSION, seed: int = None) -> 'Hypervector':
        """Generate random bipolar {-1, +1} hypervector."""
        if seed is not None:
            random.seed(seed)
        if NUMPY_AVAILABLE:
            data = list(np.random.choice([-1, 1], size=dim).astype(float))
        else:
            data = [random.choice([-1.0, 1.0]) for _ in range(dim)]
        return Hypervector(data=data, dimension=dim, vtype=VectorType.BIPOLAR)

    @staticmethod
    def from_seed(seed, dim: int = DEFAULT_DIMENSION) -> 'Hypervector':
        """Deterministic hypervector from a string or numeric seed via hashing."""
        seed_str = str(seed)
        h = hashlib.sha512(seed_str.encode()).digest()
        rng = random.Random(int.from_bytes(h, 'big'))
        data = [rng.choice([-1.0, 1.0]) for _ in range(dim)]
        return Hypervector(data=data, dimension=dim, vtype=VectorType.BIPOLAR, label=seed_str)

    @staticmethod
    def zeros(dim: int = DEFAULT_DIMENSION) -> 'Hypervector':
        return Hypervector(data=[0.0] * dim, dimension=dim, vtype=VectorType.REAL)

    def bind(self, other: 'Hypervector') -> 'Hypervector':
        """Binding (⊗): element-wise multiplication — creates associations."""
        data = [a * b for a, b in zip(self.data, other.data)]
        return Hypervector(data=data, dimension=self.dimension, vtype=self.vtype)

    def bundle(self, other: 'Hypervector') -> 'Hypervector':
        """Bundling (+): element-wise addition — creates superpositions."""
        data = [a + b for a, b in zip(self.data, other.data)]
        return Hypervector(data=data, dimension=self.dimension, vtype=VectorType.REAL)

    def permute(self, shift: int = 1) -> 'Hypervector':
        """Permutation (ρ): circular shift — encodes sequence position."""
        n = self.dimension
        data = self.data[-shift:] + self.data[:-shift]
        return Hypervector(data=data, dimension=n, vtype=self.vtype)

    def normalize(self) -> 'Hypervector':
        """Normalize to unit magnitude."""
        mag = math.sqrt(sum(x ** 2 for x in self.data))
        if mag == 0:
            return Hypervector(data=list(self.data), dimension=self.dimension, vtype=self.vtype)
        data = [x / mag for x in self.data]
        return Hypervector(data=data, dimension=self.dimension, vtype=VectorType.REAL)

    def bipolarize(self) -> 'Hypervector':
        """Snap to nearest bipolar: sign(x)."""
        data = [1.0 if x >= 0 else -1.0 for x in self.data]
        return Hypervector(data=data, dimension=self.dimension, vtype=VectorType.BIPOLAR)

    def similarity(self, other: 'Hypervector') -> float:
        """Cosine similarity."""
        dot = sum(a * b for a, b in zip(self.data, other.data))
        mag_a = math.sqrt(sum(x ** 2 for x in self.data))
        mag_b = math.sqrt(sum(x ** 2 for x in other.data))
        if mag_a * mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def hamming_similarity(self, other: 'Hypervector') -> float:
        """Hamming similarity for bipolar vectors."""
        matches = sum(1 for a, b in zip(self.data, other.data) if a * b > 0)
        return matches / self.dimension

    def magnitude(self) -> float:
        return math.sqrt(sum(x ** 2 for x in self.data))

    def __xor__(self, other: 'Hypervector') -> 'Hypervector':
        """XOR-like binding via ⊗."""
        return self.bind(other)

    def __add__(self, other: 'Hypervector') -> 'Hypervector':
        """Addition = bundling."""
        return self.bundle(other)


# ═══════════════════════════════════════════════════════════════════════════════
# ITEM MEMORY — Cleanup memory for symbol lookup
# ═══════════════════════════════════════════════════════════════════════════════

class ItemMemory:
    """Associative memory mapping labels to hypervectors."""

    def __init__(self):
        self.memory: Dict[str, Hypervector] = {}

    def store(self, label: str, vector: Hypervector):
        vector.label = label
        self.memory[label] = vector

    def lookup(self, query: Hypervector, top_k: int = 1) -> list:
        """Find nearest stored vectors by cosine similarity."""
        scores = [(label, query.similarity(vec)) for label, vec in self.memory.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def cleanup(self, query: Hypervector) -> Optional[Hypervector]:
        """Return the closest stored vector (cleanup operation)."""
        results = self.lookup(query, top_k=1)
        if results:
            return self.memory.get(results[0][0])
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SPARSE DISTRIBUTED MEMORY (SDM)
# ═══════════════════════════════════════════════════════════════════════════════

class SparseDistributedMemory:
    """
    Kanerva's Sparse Distributed Memory adapted for hyperdimensional computing.
    """

    def __init__(self, num_hard_locations: int = 1000, dim: int = DEFAULT_DIMENSION, radius: int = None):
        self.dim = dim
        self.radius = radius or int(dim * 0.49)  # Hamming radius
        self.hard_locations = [Hypervector.random_bipolar(dim) for _ in range(num_hard_locations)]
        self.counters = [[0] * dim for _ in range(num_hard_locations)]

    def write(self, address: Hypervector, data: Hypervector):
        """Write data at locations within Hamming radius of address."""
        for i, loc in enumerate(self.hard_locations):
            if address.hamming_similarity(loc) > 0.5:
                for j in range(self.dim):
                    self.counters[i][j] += int(data.data[j])

    def read(self, address: Hypervector) -> Hypervector:
        """Read by aggregating counters at matching locations."""
        result = [0.0] * self.dim
        for i, loc in enumerate(self.hard_locations):
            if address.hamming_similarity(loc) > 0.5:
                for j in range(self.dim):
                    result[j] += self.counters[i][j]
        return Hypervector(data=[1.0 if x >= 0 else -1.0 for x in result],
                          dimension=self.dim, vtype=VectorType.BIPOLAR)


# ═══════════════════════════════════════════════════════════════════════════════
# RESONATOR NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class ResonatorNetwork:
    """
    Resonator network for factorization of bound hypervectors.
    Given S = A ⊗ B, and knowing the codebook for A, recover B.
    """

    def __init__(self, codebooks: Dict[str, list]):
        """codebooks: {factor_name: [Hypervector, ...]}"""
        self.codebooks = codebooks

    def factorize(self, composite: Hypervector, max_iterations: int = 100) -> Dict[str, Hypervector]:
        """Iterative resonator factorization."""
        factors = {name: random.choice(vectors) for name, vectors in self.codebooks.items()}
        names = list(self.codebooks.keys())

        for _ in range(max_iterations):
            changed = False
            for name in names:
                # Unbind all other factors
                residual = composite
                for other_name in names:
                    if other_name != name:
                        residual = residual.bind(factors[other_name])
                # Cleanup against codebook
                best_sim = -1.0
                best_vec = factors[name]
                for candidate in self.codebooks[name]:
                    sim = residual.similarity(candidate)
                    if sim > best_sim:
                        best_sim = sim
                        best_vec = candidate
                if best_vec is not factors[name]:
                    factors[name] = best_vec
                    changed = True
            if not changed:
                break

        return factors


# ═══════════════════════════════════════════════════════════════════════════════
# SEQUENCE & RECORD ENCODERS
# ═══════════════════════════════════════════════════════════════════════════════

class SequenceEncoder:
    """Encode ordered sequences into hypervectors using permutation."""

    @staticmethod
    def encode(items: List[Hypervector]) -> Hypervector:
        """Encode sequence by bundling position-shifted items: Σ ρ^i(item_i)."""
        if not items:
            return Hypervector.zeros()
        result = items[0].permute(0)
        for i, item in enumerate(items[1:], 1):
            result = result.bundle(item.permute(i))
        return result

    @staticmethod
    def ngram_encode(items: List[Hypervector], n: int = 3) -> Hypervector:
        """N-gram encoding: bind consecutive n items, bundle all n-grams."""
        if len(items) < n:
            return SequenceEncoder.encode(items)
        ngrams = []
        for i in range(len(items) - n + 1):
            gram = items[i]
            for j in range(1, n):
                gram = gram.bind(items[i + j].permute(j))
            ngrams.append(gram)
        result = ngrams[0]
        for gram in ngrams[1:]:
            result = result.bundle(gram)
        return result


class RecordEncoder:
    """Encode key-value records: R = Σ (key_i ⊗ value_i)."""

    @staticmethod
    def encode(record: Dict[str, Hypervector], key_vectors: Dict[str, Hypervector]) -> Hypervector:
        """Encode a record as bundle of bound key-value pairs."""
        components = []
        for key, value in record.items():
            if key in key_vectors:
                components.append(key_vectors[key].bind(value))
        if not components:
            return Hypervector.zeros()
        result = components[0]
        for c in components[1:]:
            result = result.bundle(c)
        return result

    @staticmethod
    def query(record_vector: Hypervector, key_vector: Hypervector) -> Hypervector:
        """Query: unbind key to recover approximate value."""
        return record_vector.bind(key_vector)


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERDIMENSIONAL COMPUTE — Unified facade
# ═══════════════════════════════════════════════════════════════════════════════

class HyperdimensionalCompute:
    """
    Unified hyperdimensional computing engine:
    vector generation, algebra, memory, encoding, classification.
    """

    def __init__(self, dimension: int = DEFAULT_DIMENSION):
        self.dimension = dimension
        self.item_memory = ItemMemory()
        self.class_prototypes: Dict[str, Hypervector] = {}

    def random_vector(self, seed: str = None) -> Hypervector:
        """Generate a random or seeded hypervector."""
        if seed:
            return Hypervector.from_seed(seed, self.dimension)
        return Hypervector.random_bipolar(self.dimension)

    def bind(self, a: Hypervector, b: Hypervector) -> Hypervector:
        return a.bind(b)

    def bundle(self, vectors: List[Hypervector]) -> Hypervector:
        if not vectors:
            return Hypervector.zeros(self.dimension)
        result = vectors[0]
        for v in vectors[1:]:
            result = result.bundle(v)
        return result

    def encode_sequence(self, items: List[Hypervector]) -> Hypervector:
        return SequenceEncoder.encode(items)

    def encode_record(self, record: Dict[str, Hypervector], key_vectors: Dict[str, Hypervector]) -> Hypervector:
        return RecordEncoder.encode(record, key_vectors)

    def train_classifier(self, label: str, examples: List[Hypervector]):
        """Train a class prototype by bundling examples."""
        prototype = self.bundle(examples).normalize()
        self.class_prototypes[label] = prototype
        self.item_memory.store(label, prototype)

    def classify(self, query: Hypervector) -> str:
        """Classify by nearest prototype."""
        best_label = "unknown"
        best_sim = -1.0
        for label, proto in self.class_prototypes.items():
            sim = query.similarity(proto)
            if sim > best_sim:
                best_sim = sim
                best_label = label
        return best_label

    def sacred_vector(self) -> Hypervector:
        """Generate a GOD_CODE-anchored sacred hypervector."""
        return Hypervector.from_seed(f"GOD_CODE:{GOD_CODE}", self.dimension)

    def status(self) -> dict:
        return {
            "dimension": self.dimension,
            "items_stored": len(self.item_memory.memory),
            "class_prototypes": len(self.class_prototypes),
            "numpy_available": NUMPY_AVAILABLE,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

hyperdimensional_compute = HyperdimensionalCompute()


# ═══════════════════════════════════════════════════════════════════════════════
# RESONATOR NETWORK — Factorization via resonance
# ═══════════════════════════════════════════════════════════════════════════════

class ResonatorNetwork:
    """
    Resonator network for hyperdimensional factorization and analogy.
    Uses iterative vector superposition to converge on stored patterns.
    A resonator network can solve combinatorial problems by exploiting
    high-dimensional interference.
    """

    def __init__(self, codebooks: Dict[str, List[Hypervector]] = None):
        self.codebooks = codebooks or {}
        self.convergence_threshold = 0.99
        self.max_iterations = 100

    def add_codebook(self, name: str, vectors: List[Hypervector]):
        """Add a codebook (set of candidate vectors for one factor)."""
        self.codebooks[name] = vectors

    def factorize(self, target: Hypervector) -> Dict[str, Any]:
        """
        Factorize a target vector into components from each codebook.
        Uses iterative projection: for each codebook, find the best match,
        unbind it, and repeat until convergence.
        """
        if not self.codebooks:
            return {"error": "no codebooks loaded"}
        factors = {}
        residual = target
        similarities = []
        for iteration in range(self.max_iterations):
            changed = False
            for name, candidates in self.codebooks.items():
                # Find best match in this codebook
                best_sim = -1.0
                best_vec = None
                best_idx = -1
                for idx, candidate in enumerate(candidates):
                    sim = residual.similarity(candidate)
                    if sim > best_sim:
                        best_sim = sim
                        best_vec = candidate
                        best_idx = idx
                if best_vec is not None:
                    old = factors.get(name)
                    factors[name] = {"index": best_idx, "similarity": round(best_sim, 6),
                                     "label": best_vec.label}
                    if old is None or old["index"] != best_idx:
                        changed = True
                    # Update residual by unbinding this factor
                    residual = residual.bind(best_vec)  # Self-inverse for bipolar
            total_sim = sum(f["similarity"] for f in factors.values()) / max(len(factors), 1)
            similarities.append(round(total_sim, 6))
            if total_sim > self.convergence_threshold or not changed:
                break
        return {
            "factors": factors,
            "iterations": len(similarities),
            "converged": similarities[-1] > self.convergence_threshold if similarities else False,
            "final_similarity": similarities[-1] if similarities else 0,
            "trajectory": similarities,
        }

    def analogy(self, a: Hypervector, b: Hypervector, c: Hypervector) -> Hypervector:
        """
        Solve analogy: A is to B as C is to ?
        D = C ⊗ B ⊗ A^{-1}  (for bipolar: A^{-1} = A since self-inverse)
        """
        return c.bind(b).bind(a)


resonator_network = ResonatorNetwork()
