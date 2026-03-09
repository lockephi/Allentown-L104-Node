"""
l104_quantum_magic.hyperdimensional — Hyperdimensional computing integration with fallback.

Provides VectorType, Hypervector, HypervectorFactory, HDCAlgebra,
AssociativeMemory, SequenceEncoder — either from l104_hyperdimensional_computing
or standalone fallback implementations.
"""

import math
import hashlib
import random
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum, auto

# ═══════════════════════════════════════════════════════════════════════════════
# HYPERDIMENSIONAL COMPUTING INTEGRATION WITH FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from l104_hyperdimensional_computing import (
        Hypervector, HypervectorFactory, HDCAlgebra,
        AssociativeMemory, SequenceEncoder, VectorType
    )
    HDC_AVAILABLE = True
except ImportError:
    HDC_AVAILABLE = False

    # ═══════════════════════════════════════════════════════════════════════════
    # STANDALONE HDC FALLBACK IMPLEMENTATION
    # ═══════════════════════════════════════════════════════════════════════════

    class VectorType(Enum):
        """Hypervector types"""
        DENSE_BIPOLAR = auto()
        DENSE_REAL = auto()
        SPARSE_BINARY = auto()

    @dataclass
    class Hypervector:
        """Lightweight hypervector implementation"""
        vector: List[float]
        dimension: int
        vector_type: 'VectorType'
        name: str = ""

        def __len__(self) -> int:
            """Return the dimension of the hypervector."""
            return self.dimension

        def copy(self) -> 'Hypervector':
            """Return a deep copy of this hypervector."""
            return Hypervector(
                vector=self.vector.copy(),
                dimension=self.dimension,
                vector_type=self.vector_type,
                name=self.name
            )

    class HypervectorFactory:
        """Factory for creating hypervectors - optimized fallback"""

        def __init__(self, dimension: int = 10000):
            """Initialize hypervector factory with given dimension."""
            self.dimension = dimension
            self._seed_cache: Dict[str, Hypervector] = {}

        def random_bipolar(self, name: str = "") -> Hypervector:
            """Create random bipolar vector using optimized batch generation"""
            vector = [1 if random.random() > 0.5 else -1
                     for _ in range(self.dimension)]
            return Hypervector(vector, self.dimension, VectorType.DENSE_BIPOLAR, name)

        def seed_vector(self, seed: str) -> Hypervector:
            """Deterministic vector from seed with caching"""
            if seed in self._seed_cache:
                return self._seed_cache[seed].copy()

            # Use hash for deterministic seeding
            h = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % (2**32)
            random.seed(h)
            vector = [1 if random.random() > 0.5 else -1
                     for _ in range(self.dimension)]
            random.seed()

            hv = Hypervector(vector, self.dimension, VectorType.DENSE_BIPOLAR, seed)
            self._seed_cache[seed] = hv
            return hv.copy()

        def zeros(self) -> Hypervector:
            """Create a zero-valued hypervector."""
            return Hypervector([0.0] * self.dimension, self.dimension,
                              VectorType.DENSE_REAL, "zeros")

    class HDCAlgebra:
        """HDC algebra operations - optimized fallback"""

        @staticmethod
        def bind(a: Hypervector, b: Hypervector) -> Hypervector:
            """Binding via element-wise multiplication"""
            if a.dimension != b.dimension:
                raise ValueError("Dimension mismatch")
            vector = [a.vector[i] * b.vector[i] for i in range(a.dimension)]
            return Hypervector(vector, a.dimension, a.vector_type,
                              f"bind({a.name},{b.name})")

        @staticmethod
        def bundle(vectors: List[Hypervector]) -> Hypervector:
            """Bundling via majority vote"""
            if not vectors:
                raise ValueError("Empty vector list")
            dim = vectors[0].dimension
            result = [0.0] * dim
            for hv in vectors:
                for i in range(dim):
                    result[i] += hv.vector[i]
            # Threshold to bipolar
            result = [1 if v > 0 else -1 for v in result]
            return Hypervector(result, dim, VectorType.DENSE_BIPOLAR,
                              f"bundle({len(vectors)})")

        @staticmethod
        def permute(hv: Hypervector, shift: int = 1) -> Hypervector:
            """Cyclic permutation"""
            shift = shift % hv.dimension
            vector = hv.vector[-shift:] + hv.vector[:-shift]
            return Hypervector(vector, hv.dimension, hv.vector_type,
                              f"perm({hv.name},{shift})")

        @staticmethod
        def inverse(hv: Hypervector) -> Hypervector:
            """Self-inverse for bipolar vectors"""
            return hv.copy()

        @staticmethod
        def similarity(a: Hypervector, b: Hypervector) -> float:
            """Optimized cosine similarity"""
            if a.dimension != b.dimension:
                raise ValueError("Dimension mismatch")
            dot = sum(a.vector[i] * b.vector[i] for i in range(a.dimension))
            norm_a = math.sqrt(sum(v*v for v in a.vector))
            norm_b = math.sqrt(sum(v*v for v in b.vector))
            return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

    class AssociativeMemory:
        """Simple associative memory fallback"""

        def __init__(self, dimension: int = 10000):
            """Initialize associative memory with given vector dimension."""
            self.dimension = dimension
            self.memory: Dict[str, Hypervector] = {}
            self.algebra = HDCAlgebra()
            self.factory = HypervectorFactory(dimension)

        def store(self, key: str, value: Hypervector) -> None:
            """Store a hypervector with the given key."""
            self.memory[key] = value.copy()

        def retrieve(self, query: Hypervector, threshold: float = 0.3) -> List[Tuple[str, float]]:
            """Retrieve stored vectors above similarity threshold."""
            results = []
            for key, stored in self.memory.items():
                sim = self.algebra.similarity(query, stored)
                if sim >= threshold:
                    results.append((key, sim))
            return sorted(results, key=lambda x: x[1], reverse=True)

    class SequenceEncoder:
        """Sequence encoding fallback"""

        def __init__(self, dimension: int = 10000):
            """Initialize sequence encoder with given vector dimension."""
            self.dimension = dimension
            self.factory = HypervectorFactory(dimension)
            self.algebra = HDCAlgebra()
            self._element_cache: Dict[str, Hypervector] = {}

        def get_element_vector(self, element: Any) -> Hypervector:
            """Get or create a deterministic vector for an element."""
            key = str(element)
            if key not in self._element_cache:
                self._element_cache[key] = self.factory.seed_vector(key)
            return self._element_cache[key]

        def encode_sequence(self, sequence: List[Any]) -> Hypervector:
            """Encode sequence with positional binding"""
            if not sequence:
                return self.factory.zeros()
            vectors = []
            for i, elem in enumerate(sequence):
                elem_vec = self.get_element_vector(elem)
                # Permute by position
                pos_vec = self.algebra.permute(elem_vec, i)
                vectors.append(pos_vec)
            return self.algebra.bundle(vectors)
