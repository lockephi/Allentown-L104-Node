"""
l104_quantum_magic.quantum_primitives — Quantum module integration with fallbacks.

Provides Qubit, QuantumGates, QuantumRegister — either from l104_quantum_inspired
or standalone fallback implementations.
"""

import math
import cmath
import random
from dataclasses import dataclass

from .constants import _SQRT2_INV

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM MODULE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from l104_quantum_inspired import (
        Qubit, QuantumRegister, QuantumGates,
        QuantumInspiredOptimizer, QuantumAnnealingSimulator
    )
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

# ═══ L104 QUANTUM RUNTIME BRIDGE — Real IBM QPU Execution ═══
_QUANTUM_RUNTIME_AVAILABLE = False
_quantum_runtime = None
try:
    from l104_quantum_runtime import get_runtime as _get_quantum_runtime, ExecutionMode
    _quantum_runtime = _get_quantum_runtime()
    _QUANTUM_RUNTIME_AVAILABLE = True
except Exception:
    pass
    # Minimal fallback implementations
    @dataclass
    class Qubit:
        """Fallback qubit implementation"""
        alpha: complex
        beta: complex

        def __post_init__(self):
            """Normalize qubit amplitudes after initialization."""
            self.normalize()

        def normalize(self):
            """Normalize qubit state to unit probability."""
            norm = math.sqrt(abs(self.alpha)**2 + abs(self.beta)**2)
            if norm > 0:
                self.alpha /= norm
                self.beta /= norm

        def measure(self) -> int:
            """Measure qubit and collapse to |0⟩ or |1⟩."""
            prob_0 = abs(self.alpha) ** 2
            if random.random() < prob_0:
                self.alpha, self.beta = complex(1, 0), complex(0, 0)
                return 0
            else:
                self.alpha, self.beta = complex(0, 0), complex(1, 0)
                return 1

        @classmethod
        def zero(cls) -> 'Qubit':
            """Create a qubit in the |0⟩ state."""
            return cls(complex(1, 0), complex(0, 0))

        @classmethod
        def superposition(cls) -> 'Qubit':
            """Create a qubit in equal superposition (|0⟩ + |1⟩)/√2."""
            return cls(complex(_SQRT2_INV, 0), complex(_SQRT2_INV, 0))

    class QuantumGates:
        """Fallback quantum gates"""
        @staticmethod
        def hadamard(qubit: Qubit) -> Qubit:
            """Apply Hadamard gate to create superposition."""
            new_alpha = _SQRT2_INV * (qubit.alpha + qubit.beta)
            new_beta = _SQRT2_INV * (qubit.alpha - qubit.beta)
            return Qubit(new_alpha, new_beta)

        @staticmethod
        def pauli_x(qubit: Qubit) -> Qubit:
            """Apply Pauli-X (NOT) gate to flip qubit state."""
            return Qubit(qubit.beta, qubit.alpha)

        @staticmethod
        def pauli_z(qubit: Qubit) -> Qubit:
            """Apply Pauli-Z gate to flip phase of |1⟩ component."""
            return Qubit(qubit.alpha, -qubit.beta)

        @staticmethod
        def rotation_y(qubit: Qubit, theta: float) -> Qubit:
            """Apply Y-axis rotation gate by angle theta."""
            cos, sin = math.cos(theta / 2), math.sin(theta / 2)
            return Qubit(cos * qubit.alpha - sin * qubit.beta,
                        sin * qubit.alpha + cos * qubit.beta)

        @staticmethod
        def phase(qubit: Qubit, phi: float) -> Qubit:
            """Apply phase rotation gate by angle phi."""
            return Qubit(qubit.alpha, qubit.beta * cmath.exp(complex(0, phi)))

    class QuantumRegister:
        """Fallback quantum register"""
        def __init__(self, num_qubits: int):
            """Initialize quantum register with specified number of qubits."""
            self.num_qubits = num_qubits
            self.num_states = 2 ** num_qubits
            self.amplitudes = [complex(0, 0)] * self.num_states
            self.amplitudes[0] = complex(1, 0)

        def measure_all(self) -> int:
            """Measure all qubits and collapse to a basis state."""
            probs = [abs(a)**2 for a in self.amplitudes]
            r = random.random()
            cumsum = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    self.amplitudes = [complex(0, 0)] * self.num_states
                    self.amplitudes[i] = complex(1, 0)
                    return i
            return self.num_states - 1
