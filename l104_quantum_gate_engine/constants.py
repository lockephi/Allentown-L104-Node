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

# Gate-specific phase constants — canonical source: god_code_qubit.py (QPU-verified)
try:
    from l104_god_code_simulator.god_code_qubit import (
        GOD_CODE_PHASE as GOD_CODE_PHASE_ANGLE,
        PHI_PHASE as PHI_PHASE_ANGLE,
        VOID_PHASE as VOID_PHASE_ANGLE,
        IRON_PHASE as IRON_PHASE_ANGLE,
    )
except ImportError:
    GOD_CODE_PHASE_ANGLE: Final[float] = (GOD_CODE % (2 * math.pi))    # GOD_CODE mod 2π
    PHI_PHASE_ANGLE: Final[float] = (2 * math.pi / PHI)                 # 2π/φ ≈ 3.8832...
    VOID_PHASE_ANGLE: Final[float] = (VOID_CONSTANT * math.pi)          # VOID × π
    IRON_PHASE_ANGLE: Final[float] = (2 * math.pi * IRON_ATOMIC_NUMBER / QUANTIZATION_GRAIN)  # 2π×26/104

# Fibonacci anyon phases (topological)
FIBONACCI_ANYON_PHASE: Final[float] = 4 * math.pi / 5              # σ₁ braid phase
FIBONACCI_F_ENTRY: Final[float] = PHI ** (-1)                       # F-matrix entry: 1/φ
FIBONACCI_F_OFF: Final[float] = PHI ** (-0.5)                       # F-matrix off-diagonal: 1/√φ

# ═══════════════════════════════════════════════════════════════════════════════
#  TOPOLOGICAL PROTECTION — Error Rate Model (Research v1.0)
#  ε ~ exp(-d/ξ)  where ξ = 1/φ ≈ 0.618 (Fibonacci anyon correlation length)
#  Verified: d=8 → ε = 2.39e-06, d=13 → ε = 7.33e-10
# ═══════════════════════════════════════════════════════════════════════════════

TOPOLOGICAL_CORRELATION_LENGTH: Final[float] = 1.0 / PHI           # ξ = 1/φ ≈ 0.618
TOPOLOGICAL_DEFAULT_DEPTH: Final[int] = 8                           # Default anyon braid depth
TOPOLOGICAL_ERROR_RATE_D8: Final[float] = math.exp(-8 / (1.0 / PHI))  # ε(d=8) ≈ 2.39e-06
TOPOLOGICAL_ERROR_RATE_D13: Final[float] = math.exp(-13 / (1.0 / PHI))  # ε(d=13) ≈ 7.33e-10

# ═══════════════════════════════════════════════════════════════════════════════
#  UNITARY QUANTIZATION — Phase Operator Model
#  U = 2^(E/104) = (2^(1/104))^E  where E ∈ ℤ
#  The phase operator preserves norms: |e^{iθ}| = 1 ∀ θ ∈ ℝ
#  104-TET: 104 steps per octave, each step = 2^(1/104) frequency ratio
# ═══════════════════════════════════════════════════════════════════════════════

FUNDAMENTAL_STEP: Final[float] = 2 ** (1.0 / QUANTIZATION_GRAIN)   # 2^(1/104) ≈ 1.006687
OCTAVE_STEPS: Final[int] = QUANTIZATION_GRAIN                      # 104 steps = one doubling
SEMITONE_STEPS: Final[int] = 8                                      # 8 steps = 1/13 octave (8-fold symmetry)
SEMITONE_RATIO: Final[float] = 2 ** (8.0 / QUANTIZATION_GRAIN)     # 2^(1/13) ≈ 1.054769
FOUR_OCTAVE_OFFSET: Final[float] = 2 ** (OCTAVE_OFFSET / QUANTIZATION_GRAIN)  # 2^4 = 16.0

# ═══════════════════════════════════════════════════════════════════════════════
#  FACTOR-13 UNIFICATION
#  286 = 22 × 13, 104 = 8 × 13, 416 = 32 × 13
#  F(7) = 13 is the shared harmonic root (7th Fibonacci number)
# ═══════════════════════════════════════════════════════════════════════════════

FIBONACCI_7: Final[int] = 13                                        # 7th Fibonacci number
FACTOR_13_286: Final[int] = PRIME_SCAFFOLD // FIBONACCI_7            # 22
FACTOR_13_104: Final[int] = QUANTIZATION_GRAIN // FIBONACCI_7        # 8
FACTOR_13_416: Final[int] = OCTAVE_OFFSET // FIBONACCI_7             # 32

# ═══════════════════════════════════════════════════════════════════════════════
#  14-QUBIT DIAL REGISTER (embedded in 26-qubit Fe(26) iron manifold)
#  a: 3 bits (0-7), b: 4 bits (0-15), c: 3 bits (0-7), d: 4 bits (0-15)
#  Total: 14 qubits → 16,384 unique GOD_CODE configurations
#  With 12 ancilla → 26 qubits = Fe(26) = complete iron electron manifold
# ═══════════════════════════════════════════════════════════════════════════════

DIAL_BITS_A: Final[int] = 3                                         # a: 3 bits (coarse up)
DIAL_BITS_B: Final[int] = 4                                         # b: 4 bits (fine down)
DIAL_BITS_C: Final[int] = 3                                         # c: 3 bits (coarse down)
DIAL_BITS_D: Final[int] = 4                                         # d: 4 bits (octave down)
DIAL_TOTAL_BITS: Final[int] = DIAL_BITS_A + DIAL_BITS_B + DIAL_BITS_C + DIAL_BITS_D  # 14
DIAL_CONFIGURATIONS: Final[int] = 2 ** DIAL_TOTAL_BITS               # 16,384
ANCILLA_QUBITS: Final[int] = IRON_ATOMIC_NUMBER - DIAL_TOTAL_BITS   # 12
IRON_MANIFOLD_QUBITS: Final[int] = IRON_ATOMIC_NUMBER               # 26 = Fe(26)
IRON_MANIFOLD_HILBERT_DIM: Final[int] = 2 ** IRON_ATOMIC_NUMBER     # 67,108,864

# Clifford group detection tolerance
CLIFFORD_TOLERANCE: Final[float] = 1e-10

# Maximum qubits for various operations
MAX_STATEVECTOR_QUBITS: Final[int] = 25
MAX_DENSITY_MATRIX_QUBITS: Final[int] = 14
MAX_DECOMPOSITION_QUBITS: Final[int] = 4  # Exact decomposition up to 4 qubits

# Tensor Network (MPS) simulator constants
MAX_TENSOR_NETWORK_QUBITS: Final[int] = 50           # MPS hard cap
DEFAULT_MPS_BOND_DIM: Final[int] = 1024              # Default χ_max (loosened from 256)
SACRED_MPS_BOND_DIM: Final[int] = 104                 # L104 sacred: 8 × 13
HIGH_FIDELITY_MPS_BOND_DIM: Final[int] = 2048         # High-fidelity mode (loosened from 512)
MPS_SVD_CUTOFF: Final[float] = 1e-16                  # Default SVD truncation threshold (loosened from 1e-12)
PHI_TRUNCATION_BALANCE: Final[float] = 1.0 / PHI      # ≈ 0.618 — sacred truncation ratio
