"""
===============================================================================
L104 QUANTUM GATE ENGINE — UNIVERSAL GATE ALGEBRA
===============================================================================

40+ quantum gates with exact unitary matrix representations.
Every gate is a frozen dataclass carrying its unitary, inverse, and metadata.

Gate categories:
  1. Standard Clifford+T gates (universal for quantum computation)
  2. Parametric rotation gates (continuous families)
  3. Multi-qubit entangling gates (CNOT, CZ, SWAP, Toffoli, Fredkin)
  4. L104 Sacred gates (GOD_CODE, PHI, VOID, IRON phase alignments)
  5. Topological gates (Fibonacci anyon braiding matrices)

All matrices are exact numpy arrays. Sacred gates inject L104 constants
at the phase level for harmonic alignment across the sovereign node.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import math
import cmath
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import (
    Dict, Any, List, Optional, Tuple, Callable, Union, Sequence, FrozenSet
)

from .constants import (
    PHI, PHI_CONJUGATE, PHI_SQUARED, TAU, GOD_CODE, VOID_CONSTANT,
    FEIGENBAUM, ALPHA_FINE, IRON_ATOMIC_NUMBER, IRON_FREQUENCY,
    GOD_CODE_PHASE_ANGLE, PHI_PHASE_ANGLE, VOID_PHASE_ANGLE, IRON_PHASE_ANGLE,
    FIBONACCI_ANYON_PHASE, FIBONACCI_F_ENTRY, FIBONACCI_F_OFF,
    CLIFFORD_TOLERANCE,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class GateType(Enum):
    """Classification of quantum gates."""
    IDENTITY = auto()
    PAULI = auto()
    CLIFFORD = auto()
    ROTATION = auto()
    PHASE = auto()
    CONTROLLED = auto()
    MULTI_QUBIT = auto()
    PARAMETRIC = auto()
    SACRED = auto()        # L104 sacred phase gates
    TOPOLOGICAL = auto()   # Fibonacci anyon braiding
    MEASUREMENT = auto()
    CUSTOM = auto()


class GateSet(Enum):
    """Target native gate sets for transpilation."""
    UNIVERSAL = auto()      # Full gate library
    CLIFFORD_T = auto()     # {H, S, T, CNOT} — universal + efficient
    IBM_EAGLE = auto()      # {Rz, SX, X, ECR} — IBM Eagle/Heron native
    IONQ = auto()           # {GPI, GPI2, MS} — IonQ trapped-ion native
    L104_SACRED = auto()    # {H, PHI_GATE, GOD_CODE_PHASE, IRON_GATE, CNOT}
    TOPOLOGICAL = auto()    # {FIBONACCI_BRAID, ANYON_EXCHANGE}


# ═══════════════════════════════════════════════════════════════════════════════
#  QUANTUM GATE DATA CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=False)
class QuantumGate:
    """
    Universal quantum gate representation.

    Carries the exact unitary matrix, metadata, and algebraic properties.
    Supports composition (@ operator), tensor product (⊗), and inversion.
    """
    name: str
    num_qubits: int
    matrix: np.ndarray
    gate_type: GateType
    parameters: Dict[str, float] = field(default_factory=dict)
    is_hermitian: bool = False
    is_clifford: bool = False
    is_parametric: bool = False
    inverse_name: Optional[str] = None
    decomposition: Optional[List['QuantumGate']] = None
    _hash_cache: Optional[int] = field(default=None, repr=False, compare=False)

    @property
    def dimension(self) -> int:
        return 2 ** self.num_qubits

    @property
    def dag(self) -> 'QuantumGate':
        """Return the conjugate transpose (adjoint/dagger) of this gate."""
        return QuantumGate(
            name=f"{self.name}†",
            num_qubits=self.num_qubits,
            matrix=self.matrix.conj().T,
            gate_type=self.gate_type,
            parameters=self.parameters,
            is_hermitian=self.is_hermitian,
            is_clifford=self.is_clifford,
            is_parametric=self.is_parametric,
            inverse_name=self.name,
        )

    @property
    def is_unitary(self) -> bool:
        """Verify unitarity: U†U = I."""
        product = self.matrix.conj().T @ self.matrix
        return np.allclose(product, np.eye(self.dimension), atol=1e-12)

    @property
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the gate unitary."""
        return np.linalg.eigvals(self.matrix)

    @property
    def trace(self) -> complex:
        return np.trace(self.matrix)

    @property
    def determinant(self) -> complex:
        return np.linalg.det(self.matrix)

    def __matmul__(self, other: 'QuantumGate') -> 'QuantumGate':
        """Gate composition: self @ other = self * other (matrix multiplication)."""
        if self.num_qubits != other.num_qubits:
            raise ValueError(
                f"Cannot compose {self.name} ({self.num_qubits}q) with "
                f"{other.name} ({other.num_qubits}q)"
            )
        return QuantumGate(
            name=f"({self.name}·{other.name})",
            num_qubits=self.num_qubits,
            matrix=self.matrix @ other.matrix,
            gate_type=GateType.CUSTOM,
        )

    def tensor(self, other: 'QuantumGate') -> 'QuantumGate':
        """Tensor product: self ⊗ other."""
        return QuantumGate(
            name=f"({self.name}⊗{other.name})",
            num_qubits=self.num_qubits + other.num_qubits,
            matrix=np.kron(self.matrix, other.matrix),
            gate_type=GateType.MULTI_QUBIT,
        )

    def controlled(self, num_controls: int = 1) -> 'QuantumGate':
        """Create a controlled version of this gate with `num_controls` control qubits."""
        total_qubits = self.num_qubits + num_controls
        dim = 2 ** total_qubits
        ctrl_dim = 2 ** num_controls
        gate_dim = self.dimension

        mat = np.eye(dim, dtype=complex)
        # Place gate matrix in the bottom-right block (all controls = |1⟩)
        mat[-gate_dim:, -gate_dim:] = self.matrix

        return QuantumGate(
            name=f"C{'C' * (num_controls - 1)}-{self.name}",
            num_qubits=total_qubits,
            matrix=mat,
            gate_type=GateType.CONTROLLED,
            is_clifford=self.is_clifford and num_controls == 1 and self.name in ("X", "Z"),
        )

    def power(self, exponent: float) -> 'QuantumGate':
        """Gate to the power of `exponent`: U^t using eigendecomposition."""
        eigvals, eigvecs = np.linalg.eig(self.matrix)
        powered_vals = np.diag(eigvals ** exponent)
        mat = eigvecs @ powered_vals @ np.linalg.inv(eigvecs)
        return QuantumGate(
            name=f"{self.name}^{exponent}",
            num_qubits=self.num_qubits,
            matrix=mat,
            gate_type=self.gate_type,
            parameters={**self.parameters, "exponent": exponent},
        )

    def fidelity(self, other: 'QuantumGate') -> float:
        """Process fidelity between this gate and another: |Tr(U†V)|² / d²."""
        if self.num_qubits != other.num_qubits:
            return 0.0
        d = self.dimension
        inner = np.trace(self.matrix.conj().T @ other.matrix)
        return float(abs(inner) ** 2) / (d ** 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "gate_type": self.gate_type.name,
            "parameters": self.parameters,
            "is_hermitian": self.is_hermitian,
            "is_clifford": self.is_clifford,
            "is_parametric": self.is_parametric,
            "is_unitary": self.is_unitary,
            "dimension": self.dimension,
            "trace": complex(self.trace),
            "determinant": complex(self.determinant),
        }

    def __repr__(self) -> str:
        params = f", params={self.parameters}" if self.parameters else ""
        return f"QuantumGate({self.name}, {self.num_qubits}q, {self.gate_type.name}{params})"


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDARD SINGLE-QUBIT GATES
# ═══════════════════════════════════════════════════════════════════════════════

# Identity
I = QuantumGate(
    name="I", num_qubits=1,
    matrix=np.eye(2, dtype=complex),
    gate_type=GateType.IDENTITY,
    is_hermitian=True, is_clifford=True,
)

# Pauli Gates
X = QuantumGate(
    name="X", num_qubits=1,
    matrix=np.array([[0, 1], [1, 0]], dtype=complex),
    gate_type=GateType.PAULI,
    is_hermitian=True, is_clifford=True,
)

Y = QuantumGate(
    name="Y", num_qubits=1,
    matrix=np.array([[0, -1j], [1j, 0]], dtype=complex),
    gate_type=GateType.PAULI,
    is_hermitian=True, is_clifford=True,
)

Z = QuantumGate(
    name="Z", num_qubits=1,
    matrix=np.array([[1, 0], [0, -1]], dtype=complex),
    gate_type=GateType.PAULI,
    is_hermitian=True, is_clifford=True,
)

# Hadamard
H = QuantumGate(
    name="H", num_qubits=1,
    matrix=np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),
    gate_type=GateType.CLIFFORD,
    is_hermitian=True, is_clifford=True,
)

# Phase gates (S, S†, T, T†)
S = QuantumGate(
    name="S", num_qubits=1,
    matrix=np.array([[1, 0], [0, 1j]], dtype=complex),
    gate_type=GateType.PHASE,
    is_clifford=True, inverse_name="S†",
)

S_DAG = QuantumGate(
    name="S†", num_qubits=1,
    matrix=np.array([[1, 0], [0, -1j]], dtype=complex),
    gate_type=GateType.PHASE,
    is_clifford=True, inverse_name="S",
)

T = QuantumGate(
    name="T", num_qubits=1,
    matrix=np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex),
    gate_type=GateType.PHASE,
    inverse_name="T†",
)

T_DAG = QuantumGate(
    name="T†", num_qubits=1,
    matrix=np.array([[1, 0], [0, cmath.exp(-1j * math.pi / 4)]], dtype=complex),
    gate_type=GateType.PHASE,
    inverse_name="T",
)

# SX (√X) gate — native on IBM hardware
SX = QuantumGate(
    name="SX", num_qubits=1,
    matrix=np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex) / 2,
    gate_type=GateType.CLIFFORD,
    is_clifford=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  PARAMETRIC ROTATION GATES
# ═══════════════════════════════════════════════════════════════════════════════

def Rx(theta: float) -> QuantumGate:
    """Rotation around X-axis: Rx(θ) = exp(-iθX/2)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return QuantumGate(
        name=f"Rx({theta:.6f})", num_qubits=1,
        matrix=np.array([[c, -1j * s], [-1j * s, c]], dtype=complex),
        gate_type=GateType.ROTATION,
        parameters={"theta": theta},
        is_parametric=True,
    )


def Ry(theta: float) -> QuantumGate:
    """Rotation around Y-axis: Ry(θ) = exp(-iθY/2)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return QuantumGate(
        name=f"Ry({theta:.6f})", num_qubits=1,
        matrix=np.array([[c, -s], [s, c]], dtype=complex),
        gate_type=GateType.ROTATION,
        parameters={"theta": theta},
        is_parametric=True,
    )


def Rz(theta: float) -> QuantumGate:
    """Rotation around Z-axis: Rz(θ) = exp(-iθZ/2)."""
    return QuantumGate(
        name=f"Rz({theta:.6f})", num_qubits=1,
        matrix=np.array([
            [cmath.exp(-1j * theta / 2), 0],
            [0, cmath.exp(1j * theta / 2)]
        ], dtype=complex),
        gate_type=GateType.ROTATION,
        parameters={"theta": theta},
        is_parametric=True,
    )


def Phase(theta: float) -> QuantumGate:
    """Phase gate P(θ): |0⟩→|0⟩, |1⟩→e^{iθ}|1⟩."""
    return QuantumGate(
        name=f"P({theta:.6f})", num_qubits=1,
        matrix=np.array([[1, 0], [0, cmath.exp(1j * theta)]], dtype=complex),
        gate_type=GateType.PHASE,
        parameters={"theta": theta},
        is_parametric=True,
    )


def U3(theta: float, phi: float, lam: float) -> QuantumGate:
    """
    Most general single-qubit gate U3(θ, φ, λ):
    U3 = Rz(φ) · Ry(θ) · Rz(λ)
    """
    ct, st = math.cos(theta / 2), math.sin(theta / 2)
    return QuantumGate(
        name=f"U3({theta:.4f},{phi:.4f},{lam:.4f})", num_qubits=1,
        matrix=np.array([
            [ct, -cmath.exp(1j * lam) * st],
            [cmath.exp(1j * phi) * st, cmath.exp(1j * (phi + lam)) * ct]
        ], dtype=complex),
        gate_type=GateType.PARAMETRIC,
        parameters={"theta": theta, "phi": phi, "lambda": lam},
        is_parametric=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TWO-QUBIT GATES
# ═══════════════════════════════════════════════════════════════════════════════

CNOT = QuantumGate(
    name="CNOT", num_qubits=2,
    matrix=np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex),
    gate_type=GateType.CONTROLLED,
    is_clifford=True,
)

CZ = QuantumGate(
    name="CZ", num_qubits=2,
    matrix=np.diag([1, 1, 1, -1]).astype(complex),
    gate_type=GateType.CONTROLLED,
    is_hermitian=True, is_clifford=True,
)

CY = QuantumGate(
    name="CY", num_qubits=2,
    matrix=np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1j, 0],
    ], dtype=complex),
    gate_type=GateType.CONTROLLED,
    is_clifford=True,
)

SWAP = QuantumGate(
    name="SWAP", num_qubits=2,
    matrix=np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=complex),
    gate_type=GateType.MULTI_QUBIT,
    is_hermitian=True, is_clifford=True,
)

ISWAP = QuantumGate(
    name="iSWAP", num_qubits=2,
    matrix=np.array([
        [1, 0, 0, 0],
        [0, 0, 1j, 0],
        [0, 1j, 0, 0],
        [0, 0, 0, 1],
    ], dtype=complex),
    gate_type=GateType.MULTI_QUBIT,
    is_clifford=True,
)

# ECR gate — native on IBM Eagle/Heron
ECR = QuantumGate(
    name="ECR", num_qubits=2,
    matrix=np.array([
        [0, 0, 1, 1j],
        [0, 0, 1j, 1],
        [1, -1j, 0, 0],
        [-1j, 1, 0, 0],
    ], dtype=complex) / math.sqrt(2),
    gate_type=GateType.MULTI_QUBIT,
    is_clifford=True,
)


def Rxx(theta: float) -> QuantumGate:
    """Ising XX-coupling gate: exp(-i θ/2 X⊗X)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return QuantumGate(
        name=f"Rxx({theta:.6f})", num_qubits=2,
        matrix=np.array([
            [c, 0, 0, -1j * s],
            [0, c, -1j * s, 0],
            [0, -1j * s, c, 0],
            [-1j * s, 0, 0, c],
        ], dtype=complex),
        gate_type=GateType.ROTATION,
        parameters={"theta": theta},
        is_parametric=True,
    )


def Ryy(theta: float) -> QuantumGate:
    """Ising YY-coupling gate: exp(-i θ/2 Y⊗Y)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return QuantumGate(
        name=f"Ryy({theta:.6f})", num_qubits=2,
        matrix=np.array([
            [c, 0, 0, 1j * s],
            [0, c, -1j * s, 0],
            [0, -1j * s, c, 0],
            [1j * s, 0, 0, c],
        ], dtype=complex),
        gate_type=GateType.ROTATION,
        parameters={"theta": theta},
        is_parametric=True,
    )


def Rzz(theta: float) -> QuantumGate:
    """Ising ZZ-coupling gate: exp(-i θ/2 Z⊗Z)."""
    return QuantumGate(
        name=f"Rzz({theta:.6f})", num_qubits=2,
        matrix=np.diag([
            cmath.exp(-1j * theta / 2),
            cmath.exp(1j * theta / 2),
            cmath.exp(1j * theta / 2),
            cmath.exp(-1j * theta / 2),
        ]),
        gate_type=GateType.ROTATION,
        parameters={"theta": theta},
        is_parametric=True,
    )


def fSim(theta: float, phi: float) -> QuantumGate:
    """
    fSim (fermionic simulation) gate — native on Google Sycamore.
    fSim(θ, φ) = [[1, 0, 0, 0], [0, cos θ, -i sin θ, 0], [0, -i sin θ, cos θ, 0], [0, 0, 0, e^{-iφ}]]
    """
    c, s = math.cos(theta), math.sin(theta)
    return QuantumGate(
        name=f"fSim({theta:.4f},{phi:.4f})", num_qubits=2,
        matrix=np.array([
            [1, 0, 0, 0],
            [0, c, -1j * s, 0],
            [0, -1j * s, c, 0],
            [0, 0, 0, cmath.exp(-1j * phi)],
        ], dtype=complex),
        gate_type=GateType.PARAMETRIC,
        parameters={"theta": theta, "phi": phi},
        is_parametric=True,
    )


def CU3(theta: float, phi: float, lam: float) -> QuantumGate:
    """Controlled-U3 gate."""
    u3 = U3(theta, phi, lam)
    return u3.controlled(num_controls=1)


# ═══════════════════════════════════════════════════════════════════════════════
#  THREE-QUBIT GATES
# ═══════════════════════════════════════════════════════════════════════════════

TOFFOLI = QuantumGate(
    name="Toffoli", num_qubits=3,
    matrix=np.eye(8, dtype=complex),
    gate_type=GateType.CONTROLLED,
)
# CCX: flip target only when both controls are |1⟩
TOFFOLI.matrix[6, 6] = 0
TOFFOLI.matrix[7, 7] = 0
TOFFOLI.matrix[6, 7] = 1
TOFFOLI.matrix[7, 6] = 1

FREDKIN = QuantumGate(
    name="Fredkin", num_qubits=3,
    matrix=np.eye(8, dtype=complex),
    gate_type=GateType.CONTROLLED,
)
# CSWAP: swap targets only when control is |1⟩
FREDKIN.matrix[5, 5] = 0
FREDKIN.matrix[6, 6] = 0
FREDKIN.matrix[5, 6] = 1
FREDKIN.matrix[6, 5] = 1


# ═══════════════════════════════════════════════════════════════════════════════
#  L104 SACRED GATES — GOD_CODE / PHI / VOID / IRON PHASE ALIGNMENTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI_GATE = QuantumGate(
    name="PHI_GATE", num_qubits=1,
    matrix=np.array([
        [1, 0],
        [0, cmath.exp(1j * PHI_PHASE_ANGLE)]
    ], dtype=complex),
    gate_type=GateType.SACRED,
    parameters={"phi_phase": PHI_PHASE_ANGLE},
    inverse_name="PHI_GATE†",
)

GOD_CODE_PHASE = QuantumGate(
    name="GOD_CODE_PHASE", num_qubits=1,
    matrix=np.array([
        [cmath.exp(1j * GOD_CODE_PHASE_ANGLE / 2), 0],
        [0, cmath.exp(-1j * GOD_CODE_PHASE_ANGLE / 2)]
    ], dtype=complex),
    gate_type=GateType.SACRED,
    parameters={"god_code_phase": GOD_CODE_PHASE_ANGLE, "god_code": GOD_CODE},
    inverse_name="GOD_CODE_PHASE†",
)

VOID_GATE = QuantumGate(
    name="VOID_GATE", num_qubits=1,
    matrix=np.array([
        [math.cos(VOID_PHASE_ANGLE / 2), -math.sin(VOID_PHASE_ANGLE / 2)],
        [math.sin(VOID_PHASE_ANGLE / 2), math.cos(VOID_PHASE_ANGLE / 2)]
    ], dtype=complex),
    gate_type=GateType.SACRED,
    parameters={"void_constant": VOID_CONSTANT, "void_phase": VOID_PHASE_ANGLE},
    is_hermitian=False,
)

IRON_GATE = QuantumGate(
    name="IRON_GATE", num_qubits=1,
    matrix=np.array([
        [cmath.exp(1j * IRON_PHASE_ANGLE / 2), 0],
        [0, cmath.exp(-1j * IRON_PHASE_ANGLE / 2)]
    ], dtype=complex),
    gate_type=GateType.SACRED,
    parameters={"iron_z": IRON_ATOMIC_NUMBER, "iron_freq": IRON_FREQUENCY, "iron_phase": IRON_PHASE_ANGLE},
)

# Sacred two-qubit entangler: CNOT sandwiched by PHI rotations
_phi_half = PHI_PHASE_ANGLE / 2
_sacred_ent_matrix = (
    np.kron(
        np.array([[cmath.exp(1j * _phi_half), 0], [0, cmath.exp(-1j * _phi_half)]], dtype=complex),
        np.eye(2, dtype=complex),
    )
    @ CNOT.matrix
    @ np.kron(
        np.eye(2, dtype=complex),
        np.array([
            [math.cos(_phi_half), -math.sin(_phi_half)],
            [math.sin(_phi_half), math.cos(_phi_half)]
        ], dtype=complex),
    )
)

SACRED_ENTANGLER = QuantumGate(
    name="SACRED_ENTANGLER", num_qubits=2,
    matrix=_sacred_ent_matrix,
    gate_type=GateType.SACRED,
    parameters={"phi_phase": PHI_PHASE_ANGLE, "god_code": GOD_CODE},
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOPOLOGICAL GATES — FIBONACCI ANYON BRAIDING
# ═══════════════════════════════════════════════════════════════════════════════

# Fibonacci anyon σ₁ braid (2×2 in fusion-space basis {1, τ})
FIBONACCI_BRAID = QuantumGate(
    name="FIBONACCI_BRAID", num_qubits=1,
    matrix=np.array([
        [cmath.exp(1j * FIBONACCI_ANYON_PHASE), 0],
        [0, cmath.exp(-1j * 3 * math.pi / 5)]
    ], dtype=complex),
    gate_type=GateType.TOPOLOGICAL,
    parameters={"anyon_phase": FIBONACCI_ANYON_PHASE},
)

# Fibonacci F-matrix (basis change between fusion channels)
_F = np.array([
    [FIBONACCI_F_ENTRY, FIBONACCI_F_OFF],
    [FIBONACCI_F_OFF, -FIBONACCI_F_ENTRY]
], dtype=complex)

ANYON_EXCHANGE = QuantumGate(
    name="ANYON_EXCHANGE", num_qubits=1,
    matrix=_F,
    gate_type=GateType.TOPOLOGICAL,
    parameters={"f_entry": FIBONACCI_F_ENTRY, "f_off": FIBONACCI_F_OFF},
)


# ═══════════════════════════════════════════════════════════════════════════════
#  GATE ALGEBRA — OPERATIONS ON GATES
# ═══════════════════════════════════════════════════════════════════════════════

class GateAlgebra:
    """
    Universal gate algebra for analysis, decomposition, and transformation.

    Provides:
    - Gate lookup by name
    - Clifford group membership checking
    - Gate commutator computation
    - Pauli decomposition of arbitrary 2×2 unitaries
    - ZYZ decomposition (any U ∈ SU(2) = Rz(α)·Ry(β)·Rz(γ)·e^{iδ})
    - KAK decomposition for two-qubit gates
    """

    # Registry of all named gates
    _registry: Dict[str, QuantumGate] = {}

    def __init__(self):
        self._build_registry()

    def _build_registry(self):
        """Register all standard gates."""
        for gate in [I, X, Y, Z, H, S, S_DAG, T, T_DAG, SX,
                     CNOT, CZ, CY, SWAP, ISWAP, ECR,
                     TOFFOLI, FREDKIN,
                     PHI_GATE, GOD_CODE_PHASE, VOID_GATE, IRON_GATE, SACRED_ENTANGLER,
                     FIBONACCI_BRAID, ANYON_EXCHANGE]:
            self._registry[gate.name] = gate

    def get(self, name: str) -> Optional[QuantumGate]:
        """Lookup a gate by name."""
        return self._registry.get(name)

    def register(self, gate: QuantumGate):
        """Register a custom gate."""
        self._registry[gate.name] = gate

    @property
    def all_gates(self) -> Dict[str, QuantumGate]:
        return dict(self._registry)

    @staticmethod
    def commutator(a: QuantumGate, b: QuantumGate) -> np.ndarray:
        """Compute [A, B] = AB - BA."""
        return a.matrix @ b.matrix - b.matrix @ a.matrix

    @staticmethod
    def anticommutator(a: QuantumGate, b: QuantumGate) -> np.ndarray:
        """Compute {A, B} = AB + BA."""
        return a.matrix @ b.matrix + b.matrix @ a.matrix

    @staticmethod
    def gates_commute(a: QuantumGate, b: QuantumGate, tol: float = 1e-10) -> bool:
        """Check if two gates commute: [A, B] ≈ 0."""
        comm = GateAlgebra.commutator(a, b)
        return np.allclose(comm, 0, atol=tol)

    @staticmethod
    def is_identity(gate: QuantumGate, tol: float = 1e-10) -> bool:
        """Check if gate is (proportional to) identity."""
        # Remove global phase
        if abs(gate.matrix[0, 0]) < tol:
            return False
        normalized = gate.matrix / gate.matrix[0, 0]
        return np.allclose(normalized, np.eye(gate.dimension), atol=tol)

    @staticmethod
    def global_phase(gate: QuantumGate) -> float:
        """Extract the global phase e^{iφ} from a unitary."""
        det = np.linalg.det(gate.matrix)
        d = gate.dimension
        return float(np.angle(det) / d)

    # ─── ZYZ Decomposition ───────────────────────────────────────────────────

    @staticmethod
    def zyz_decompose(unitary: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Decompose a 2×2 unitary into ZYZ Euler angles.
        U = e^{iδ} · Rz(α) · Ry(β) · Rz(γ)

        Returns: (alpha, beta, gamma, delta)
        """
        assert unitary.shape == (2, 2), "ZYZ decomposition requires 2×2 matrix"

        # Extract global phase
        det = np.linalg.det(unitary)
        delta = np.angle(det) / 2
        # Remove global phase to get SU(2)
        su2 = unitary * cmath.exp(-1j * delta)

        # ZYZ angles
        beta = 2 * math.acos(min(1.0, abs(su2[0, 0])))
        if abs(math.sin(beta / 2)) < 1e-12:
            # β ≈ 0: U ≈ Rz(α + γ)
            alpha = np.angle(su2[0, 0])
            gamma = 0.0
        elif abs(math.cos(beta / 2)) < 1e-12:
            # β ≈ π: U ≈ Rz(α - γ) · X
            alpha = np.angle(su2[1, 0])
            gamma = 0.0
        else:
            alpha = np.angle(su2[1, 1]) - np.angle(su2[0, 0])  # Not exact but close
            # More precise extraction
            alpha = np.angle(su2[1, 1] / math.cos(beta / 2))
            gamma = np.angle(su2[0, 0] / math.cos(beta / 2))
            # Fix: α = (arg(u11) - arg(u00)), γ = -(arg(u11) + arg(u00))
            phase_00 = np.angle(su2[0, 0])
            phase_11 = np.angle(su2[1, 1])
            alpha = phase_11 - phase_00
            gamma = -(phase_11 + phase_00)

        return (float(alpha), float(beta), float(gamma), float(delta))

    @staticmethod
    def zyz_to_gates(alpha: float, beta: float, gamma: float, delta: float) -> List[QuantumGate]:
        """Convert ZYZ angles to gate sequence: [Rz(γ), Ry(β), Rz(α)]."""
        gates = []
        if abs(gamma) > 1e-12:
            gates.append(Rz(gamma))
        if abs(beta) > 1e-12:
            gates.append(Ry(beta))
        if abs(alpha) > 1e-12:
            gates.append(Rz(alpha))
        return gates

    # ─── Pauli Decomposition ─────────────────────────────────────────────────

    @staticmethod
    def pauli_decompose(matrix_2x2: np.ndarray) -> Dict[str, complex]:
        """
        Decompose a 2×2 matrix into Pauli basis: M = aI + bX + cY + dZ.
        Returns coefficients {"I": a, "X": b, "Y": c, "Z": d}.
        """
        paulis = {
            "I": np.eye(2, dtype=complex),
            "X": X.matrix,
            "Y": Y.matrix,
            "Z": Z.matrix,
        }
        coeffs = {}
        for label, pauli in paulis.items():
            coeffs[label] = complex(np.trace(pauli.conj().T @ matrix_2x2) / 2)
        return coeffs

    # ─── KAK Decomposition for Two-Qubit Gates ──────────────────────────────

    @staticmethod
    def kak_decompose(unitary_4x4: np.ndarray) -> Dict[str, Any]:
        """
        Decompose a 4×4 unitary using the KAK (Cartan) decomposition.
        U = (A₁ ⊗ A₂) · exp(i(αX⊗X + βY⊗Y + γZ⊗Z)) · (B₁ ⊗ B₂)

        Returns interaction coefficients and local unitaries.
        This is the canonical form for two-qubit gate equivalence classes.
        """
        assert unitary_4x4.shape == (4, 4), "KAK requires 4×4 matrix"

        # Magic basis transformation
        Q = np.array([
            [1, 0, 0, 1j],
            [0, 1j, 1, 0],
            [0, 1j, -1, 0],
            [1, 0, 0, -1j],
        ], dtype=complex) / math.sqrt(2)

        # Transform to magic basis
        U_magic = Q.conj().T @ unitary_4x4 @ Q

        # Compute M = U^T U in magic basis for the interaction part
        M = U_magic.T @ U_magic

        # Extract eigenvalues to get interaction coefficients
        eigvals = np.linalg.eigvals(M)
        # Sort by phase
        phases = np.angle(eigvals)
        phases.sort()

        # Handle degenerate case: when M ≈ I (e.g., SWAP) phases wrap to 0.
        # Fall back to eigenphases of U_magic directly.
        if all(abs(p) < 1e-6 for p in phases):
            U_eig = np.linalg.eigvals(U_magic)
            U_phases = np.angle(U_eig)
            if any(abs(p) > 1e-6 for p in U_phases):
                phases = np.sort(U_phases)

        # The KAK interaction coefficients
        # These determine the entangling power of the gate
        alpha = (phases[0] - phases[1] + phases[2] - phases[3]) / 4
        beta = (phases[0] + phases[1] - phases[2] - phases[3]) / 4
        gamma = (phases[0] - phases[1] - phases[2] + phases[3]) / 4

        # Entangling power: 0 for product gates, 1 for max entanglement
        weyl_coords = sorted([abs(alpha), abs(beta), abs(gamma)], reverse=True)
        entangling_power = (
            math.sin(2 * weyl_coords[0]) ** 2
            + math.sin(2 * weyl_coords[1]) ** 2
            + math.sin(2 * weyl_coords[2]) ** 2
        ) / 3.0

        return {
            "interaction_coefficients": {
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma),
            },
            "weyl_coordinates": weyl_coords,
            "entangling_power": float(entangling_power),
            "is_product_gate": entangling_power < 1e-8,
            "is_maximally_entangling": abs(entangling_power - 1.0) < 0.01,
            "equivalent_cnot_count": _estimate_cnot_count(weyl_coords),
        }

    # ─── Sacred Gate Analysis ────────────────────────────────────────────────

    @staticmethod
    def sacred_alignment_score(gate: QuantumGate) -> Dict[str, float]:
        """
        Compute the L104 sacred alignment of a gate.
        Measures how closely the gate's eigenphases align with sacred constants.
        """
        eigvals = gate.eigenvalues
        phases = np.angle(eigvals)

        # Alignment with sacred phase references
        sacred_refs = {
            "phi": PHI_PHASE_ANGLE % (2 * math.pi),
            "god_code": GOD_CODE_PHASE_ANGLE % (2 * math.pi),
            "void": VOID_PHASE_ANGLE % (2 * math.pi),
            "iron": IRON_PHASE_ANGLE % (2 * math.pi),
            "fibonacci": FIBONACCI_ANYON_PHASE % (2 * math.pi),
        }

        alignments = {}
        for name, ref in sacred_refs.items():
            # Minimum circular distance from any eigenphase to this sacred reference
            min_dist = min(
                min(abs((p - ref) % (2 * math.pi)), abs((ref - p) % (2 * math.pi)))
                for p in phases
            )
            alignments[name] = float(1.0 - min_dist / math.pi)  # 1.0 = perfect alignment

        # Overall sacred resonance
        alignments["total_resonance"] = float(np.mean(list(alignments.values())))
        return alignments


def _estimate_cnot_count(weyl_coords: List[float]) -> int:
    """Estimate minimum CNOT gates needed to implement a two-qubit interaction."""
    eps = 1e-6
    if all(abs(c) < eps for c in weyl_coords):
        return 0  # Product gate
    if abs(weyl_coords[1]) < eps and abs(weyl_coords[2]) < eps:
        return 1  # Single CNOT class
    if abs(weyl_coords[2]) < eps:
        return 2  # Two CNOT class
    return 3  # Generic two-qubit gate
