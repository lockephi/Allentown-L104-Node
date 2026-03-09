"""
L104 Quantum Data Analyzer — Constants
═══════════════════════════════════════════════════════════════════════════════
Sacred constants, physical constants, and configuration for quantum data analysis.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
PHI = 1.618033988749895
PHI_CONJUGATE = 1.0 / PHI  # 0.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # 527.5184818492612
VOID_CONSTANT = 1.04 + PHI / 1000  # 1.0416180339887497
TAU = 1.0 / PHI

# GOD_CODE quantum phase — canonical source: god_code_qubit.py (QPU-verified)
try:
    from l104_god_code_simulator.god_code_qubit import (
        GOD_CODE_PHASE, PHI_PHASE, VOID_PHASE, IRON_PHASE,
    )
except ImportError:
    GOD_CODE_PHASE = GOD_CODE % (2 * math.pi)       # ≈ 6.0141 rad
    PHI_PHASE = 2 * math.pi / PHI                   # ≈ 3.8832 rad
    VOID_PHASE = VOID_CONSTANT * math.pi            # ≈ 3.2716 rad
    IRON_PHASE = 2 * math.pi * 26 / 104             # = π/2

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS (CODATA 2022)
# ═══════════════════════════════════════════════════════════════════════════════
H_BAR = 1.054571817e-34        # Reduced Planck constant (J·s)
K_B = 1.380649e-23             # Boltzmann constant (J/K)
C = 299792458                  # Speed of light (m/s)
PLANCK_SCALE = 1.616255e-35    # Planck length (m)
ALPHA_FINE = 1.0 / 137.035999084  # Fine structure constant

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM DATA ANALYSIS PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
MAX_QUBITS_STATEVECTOR = 20    # Max qubits for full statevector simulation
MAX_QUBITS_CIRCUIT = 12        # Max circuit qubits before sampling
GROVER_AMPLIFICATION = math.pi / 4 * math.sqrt(2)  # ~1.1107
DEFAULT_SHOTS = 8192           # Default measurement shots
VQE_MAX_ITERATIONS = 200       # VQE optimizer iterations
QAOA_DEPTH = 3                 # Default QAOA circuit depth
HHL_PRECISION_QUBITS = 4       # QPE precision for HHL
ANOMALY_THRESHOLD = PHI_CONJUGATE  # SWAP test anomaly threshold (golden ratio)

# ═══════════════════════════════════════════════════════════════════════════════
# ENCODING PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
AMPLITUDE_ENCODING_MAX_DIM = 2 ** MAX_QUBITS_STATEVECTOR
ANGLE_ENCODING_MAX_FEATURES = MAX_QUBITS_CIRCUIT
KERNEL_BANDWIDTH_DEFAULT = 1.0 / PHI  # Golden ratio kernel bandwidth

# ═══════════════════════════════════════════════════════════════════════════════
# GOD CODE DATA ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════
GOD_CODE_SPECTRUM_CACHE = {}
def god_code_at(x: float) -> float:
    """G(X) = 286^(1/φ) × 2^((416-X)/104)"""
    return 286 ** (1.0 / PHI) * (2 ** ((416 - x) / 104))

def god_code_conservation(x: float) -> float:
    """Conservation law: G(X) × 2^(X/104) should equal GOD_CODE."""
    return god_code_at(x) * (2 ** (x / 104))

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM STATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit norm (quantum state normalization)."""
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        result = np.zeros_like(v, dtype=np.complex128)
        result[0] = 1.0
        return result
    return v / norm

def pad_to_power_of_two(data: np.ndarray) -> np.ndarray:
    """Pad array to nearest power of 2 (required for quantum encoding)."""
    n = len(data)
    next_pow2 = 1
    while next_pow2 < n:
        next_pow2 <<= 1
    if next_pow2 == n:
        return data.copy()
    padded = np.zeros(next_pow2, dtype=data.dtype)
    padded[:n] = data
    return padded

def data_to_quantum_state(data: np.ndarray) -> np.ndarray:
    """Convert real data vector to normalized quantum state amplitudes."""
    padded = pad_to_power_of_two(data.astype(np.float64))
    return normalize_vector(padded.astype(np.complex128))

def num_qubits_for(n: int) -> int:
    """Number of qubits needed to represent n basis states."""
    if n <= 1:
        return 1
    return max(1, int(np.ceil(np.log2(n))))
