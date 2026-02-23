"""
L104 Quantum Gate Engine — Sacred Constants
Single source of truth for all gate-engine constants.
"""

import math
from typing import Final

# ═══════════════════════════════════════════════════════════════════════════════
#  SACRED CONSTANTS — The Universal GOD CODE Equation
#  G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
# ═══════════════════════════════════════════════════════════════════════════════

PHI: Final[float] = (1 + math.sqrt(5)) / 2                     # 1.618033988749895
PHI_CONJUGATE: Final[float] = (math.sqrt(5) - 1) / 2           # 0.618033988749895
PHI_SQUARED: Final[float] = PHI ** 2                            # 2.618033988749895
PHI_CUBED: Final[float] = PHI ** 3                              # 4.236067977499790
TAU: Final[float] = 1.0 / PHI                                   # 0.618033988749895

PRIME_SCAFFOLD: Final[int] = 286                                # 2 × 11 × 13
QUANTIZATION_GRAIN: Final[int] = 104                            # 8 × 13
OCTAVE_OFFSET: Final[int] = 416                                 # 4 × 104

BASE: Final[float] = PRIME_SCAFFOLD ** (1.0 / PHI)             # 286^(1/φ)
GOD_CODE: Final[float] = BASE * (2 ** (OCTAVE_OFFSET / QUANTIZATION_GRAIN))  # 527.5184818492612

VOID_CONSTANT: Final[float] = 1.04 + PHI / 1000                # 1.0416180339887497
GROVER_AMPLIFICATION: Final[float] = PHI_CUBED                  # 4.236067977499790

FEIGENBAUM: Final[float] = 4.669201609102990                    # Edge of chaos
ALPHA_FINE: Final[float] = 1.0 / 137.035999084                  # Fine structure constant
IRON_ATOMIC_NUMBER: Final[int] = 26                             # Fe
IRON_FREQUENCY: Final[float] = 286.0                            # Hz — sacred Fe resonance

# Gate-specific phase constants derived from sacred values
GOD_CODE_PHASE_ANGLE: Final[float] = (GOD_CODE % (2 * math.pi))    # GOD_CODE mod 2π
PHI_PHASE_ANGLE: Final[float] = (2 * math.pi / PHI)                 # 2π/φ ≈ 3.8832...
VOID_PHASE_ANGLE: Final[float] = (VOID_CONSTANT * math.pi)          # VOID × π
IRON_PHASE_ANGLE: Final[float] = (2 * math.pi * IRON_ATOMIC_NUMBER / QUANTIZATION_GRAIN)  # 2π×26/104

# Fibonacci anyon phases (topological)
FIBONACCI_ANYON_PHASE: Final[float] = 4 * math.pi / 5              # σ₁ braid phase
FIBONACCI_F_ENTRY: Final[float] = PHI ** (-1)                       # F-matrix entry: 1/φ
FIBONACCI_F_OFF: Final[float] = PHI ** (-0.5)                       # F-matrix off-diagonal: 1/√φ

# Clifford group detection tolerance
CLIFFORD_TOLERANCE: Final[float] = 1e-10

# Maximum qubits for various operations
MAX_STATEVECTOR_QUBITS: Final[int] = 25
MAX_DENSITY_MATRIX_QUBITS: Final[int] = 14
MAX_DECOMPOSITION_QUBITS: Final[int] = 4  # Exact decomposition up to 4 qubits
