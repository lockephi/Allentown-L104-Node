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
    L104_HERON = auto()     # {Rz, SX, X, ECR} — L104 sovereign Heron-class native
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
    def params(self) -> list:
        """Qiskit-compat: return list of parameter values."""
        return list(self.parameters.values()) if self.parameters else []

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

# SX (√X) gate — native on L104 Heron-class hardware
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
    is_hermitian=True, is_clifford=True,
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

# ECR gate — native on L104 Heron-class hardware
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

_toff_matrix = np.eye(8, dtype=complex)
_toff_matrix[6, 6] = 0; _toff_matrix[7, 7] = 0
_toff_matrix[6, 7] = 1; _toff_matrix[7, 6] = 1

TOFFOLI = QuantumGate(
    name="Toffoli", num_qubits=3,
    matrix=_toff_matrix,
    gate_type=GateType.CONTROLLED,
)

_fred_matrix = np.eye(8, dtype=complex)
_fred_matrix[5, 5] = 0; _fred_matrix[6, 6] = 0
_fred_matrix[5, 6] = 1; _fred_matrix[6, 5] = 1

FREDKIN = QuantumGate(
    name="Fredkin", num_qubits=3,
    matrix=_fred_matrix,
    gate_type=GateType.CONTROLLED,
)


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
#  v2.0: ML ENCODING GATES — SVM Feature Encoding + Classification
# ═══════════════════════════════════════════════════════════════════════════════

def SVM_ENCODING_GATE(theta: float, phi_scale: float = PHI) -> QuantumGate:
    """Composite SVM encoding gate: Ry(theta*PHI) followed by Rz(theta/VOID_CONSTANT).

    Encodes a classical feature value into quantum phase space using
    PHI-scaled Y-rotation and VOID-scaled Z-rotation.

    Args:
        theta: Classical feature value
        phi_scale: Scaling factor (default: PHI)

    Returns:
        QuantumGate with the composed unitary
    """
    ry_angle = theta * phi_scale
    rz_angle = theta / VOID_CONSTANT
    cy, sy = math.cos(ry_angle / 2), math.sin(ry_angle / 2)
    # Ry(ry_angle)
    Ry = np.array([[cy, -sy], [sy, cy]], dtype=complex)
    # Rz(rz_angle)
    Rz = np.array([
        [cmath.exp(-1j * rz_angle / 2), 0],
        [0, cmath.exp(1j * rz_angle / 2)]
    ], dtype=complex)
    matrix = Rz @ Ry
    return QuantumGate(
        name=f"SVM_ENC({theta:.4f})", num_qubits=1,
        matrix=matrix, gate_type=GateType.SACRED,
        parameters={"theta": theta, "phi_scale": phi_scale,
                     "ry_angle": ry_angle, "rz_angle": rz_angle},
    )


def CLASSIFIER_MEASUREMENT_GATE(n_classes: int = 2) -> QuantumGate:
    """Measurement-basis rotation gate for n-class classification output.

    Rotates the measurement basis by π/(n_classes * PHI) to align
    the computational basis with class label boundaries.

    Args:
        n_classes: Number of classification classes

    Returns:
        QuantumGate with the measurement-basis rotation
    """
    angle = math.pi / (n_classes * PHI)
    cy, sy = math.cos(angle / 2), math.sin(angle / 2)
    matrix = np.array([[cy, -sy], [sy, cy]], dtype=complex)
    return QuantumGate(
        name=f"CLASS_MEAS({n_classes})", num_qubits=1,
        matrix=matrix, gate_type=GateType.SACRED,
        parameters={"n_classes": n_classes, "angle": angle},
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

    # ─── Unitary Quantization (Topological Research v1.0) ────────────────────

    @staticmethod
    def verify_unitarity(gate: QuantumGate, tolerance: float = 1e-12) -> Dict[str, Any]:
        """
        Verify U†U = I for a quantum gate (unitary quantization proof).

        The phase operator U = 2^(E/104) preserves norms because:
          U = e^{iθ} where θ = E × ln(2)/104
          |e^{iθ}| = 1 ∀ θ ∈ ℝ → ||U|ψ⟩|| = ||ψ⟩||

        Returns unitarity metrics including max deviation from identity.
        """
        U = gate.matrix
        product = U.conj().T @ U
        identity = np.eye(gate.dimension, dtype=complex)
        deviation = np.max(np.abs(product - identity))
        # Eigenvalue norms (should all be 1.0 for unitary)
        eigvals = gate.eigenvalues
        eigval_norms = [abs(ev) for ev in eigvals]
        norm_deviation = max(abs(n - 1.0) for n in eigval_norms)
        # Determinant magnitude (should be 1.0 for unitary)
        det_mag = abs(np.linalg.det(U))
        return {
            "is_unitary": deviation < tolerance,
            "max_UdagU_deviation": float(deviation),
            "eigenvalue_norms": eigval_norms,
            "max_norm_deviation": float(norm_deviation),
            "determinant_magnitude": float(det_mag),
            "tolerance": tolerance,
            "gate_name": gate.name,
        }

    @staticmethod
    def verify_reversibility(gate: QuantumGate, tolerance: float = 1e-12) -> Dict[str, Any]:
        """
        Verify strict reversibility: U⁻¹ exists and U†U = UU† = I.

        For GOD_CODE phase operator: U⁻¹ = (2^(1/104))^{-E}
        Achieved by negating all dials: G(A,B,C,D)⁻¹ = G(-A,-B,-C,-D)
        """
        U = gate.matrix
        # U†U
        forward = U.conj().T @ U
        # UU†
        backward = U @ U.conj().T
        identity = np.eye(gate.dimension, dtype=complex)
        dev_forward = float(np.max(np.abs(forward - identity)))
        dev_backward = float(np.max(np.abs(backward - identity)))
        return {
            "is_reversible": dev_forward < tolerance and dev_backward < tolerance,
            "forward_deviation": dev_forward,   # max|U†U - I|
            "backward_deviation": dev_backward,  # max|UU† - I|
            "is_normal": dev_forward < tolerance and dev_backward < tolerance,
            "gate_name": gate.name,
        }

    @staticmethod
    def composite_gate_order(gates: List[QuantumGate], max_k: int = 10000) -> Dict[str, Any]:
        """
        Determine the order of a composite gate U = G₁·G₂·...·Gₙ.

        Sacred composite gates have INFINITE ORDER because their eigenphases
        are not rational multiples of π → U^k ≠ I for any finite k.

        Optimization: check eigenphases for rationality first. If any phase
        is an irrational multiple of 2π, the order is infinite — avoids
        brute-forcing up to 10K matrix multiplications.
        """
        # Build composite
        composite = gates[0].matrix.copy()
        for g in gates[1:]:
            composite = composite @ g.matrix
        # Check eigenphases for rationality
        eigvals = np.linalg.eigvals(composite)
        phases_deg = [float(np.degrees(np.angle(ev))) for ev in eigvals]

        # Early exit: check if all eigenphases are rational multiples of 360°
        # (i.e., p/q with small denominator). If any is irrational → infinite order.
        from fractions import Fraction
        all_rational = True
        max_denominator = max_k  # If order exists, it divides lcm of denominators
        for ev in eigvals:
            phase_turns = np.angle(ev) / (2 * np.pi)  # Phase as fraction of full turn
            # Try to approximate as rational p/q
            frac = Fraction(float(phase_turns)).limit_denominator(max_denominator)
            # Check if the approximation is close enough
            if abs(float(frac) - phase_turns) > 1e-10:
                all_rational = False
                break

        if not all_rational:
            return {
                "eigenphases_deg": phases_deg,
                "finite_order": None,
                "is_infinite_order": True,
                "max_k_tested": 0,
                "gate_names": [g.name for g in gates],
                "note": "Irrational eigenphase detected — infinite order",
            }

        # Eigenphases are rational — find exact order via matrix power
        identity = np.eye(composite.shape[0], dtype=complex)
        power = np.eye(composite.shape[0], dtype=complex)
        finite_order = None
        for k in range(1, min(max_k + 1, 10001)):
            power = power @ composite
            if np.allclose(power, identity, atol=1e-10):
                finite_order = k
                break
        return {
            "eigenphases_deg": phases_deg,
            "finite_order": finite_order,
            "is_infinite_order": finite_order is None,
            "max_k_tested": max_k if finite_order is None else finite_order,
            "gate_names": [g.name for g in gates],
        }

    @staticmethod
    def bloch_manifold_state(gate: QuantumGate) -> Dict[str, Any]:
        """
        Compute the Bloch sphere representation of a gate applied to |0⟩.

        X = G(A,B,C,D) maps to a position on the Bloch manifold S²:
          - 286^(1/φ) base sets the radius (amplitude)
          - 2^(E/104) operator rotates the azimuthal angle
          - Together they define a TOPOLOGICALLY PROTECTED state
        """
        if gate.num_qubits != 1:
            return {"error": "Bloch representation requires single-qubit gate"}
        # Apply gate to |0⟩
        state = gate.matrix @ np.array([1, 0], dtype=complex)
        # Bloch coordinates: r = (⟨X⟩, ⟨Y⟩, ⟨Z⟩)
        rho = np.outer(state, state.conj())
        rx = float(2 * rho[0, 1].real)
        ry = float(-2 * rho[0, 1].imag)  # ⟨Y⟩ = Tr(σy ρ) = −2·Im(ρ₀₁)
        rz = float(rho[0, 0].real - rho[1, 1].real)
        r_mag = math.sqrt(rx**2 + ry**2 + rz**2)
        purity = float(np.trace(rho @ rho).real)
        return {
            "bloch_vector": (rx, ry, rz),
            "magnitude": r_mag,
            "is_pure_state": abs(r_mag - 1.0) < 1e-10,
            "purity": purity,
            "theta": math.acos(max(-1, min(1, rz))) if r_mag > 1e-12 else 0.0,
            "phi_angle": math.atan2(ry, rx),
            "gate_name": gate.name,
        }

    # ─── Topological Protection Analysis ─────────────────────────────────────

    @staticmethod
    def topological_error_rate(braid_depth: int) -> Dict[str, float]:
        """
        Compute the topological error rate at a given Fibonacci anyon braid depth.

        Model: ε ~ exp(-d/ξ) where ξ = 1/φ ≈ 0.618
        The exponential suppression with correlation length ξ = 1/φ provides
        robust protection against local perturbations.

        Args:
            braid_depth: Number of anyon braids (d)

        Returns:
            Error rate metrics including suppression quality.
        """
        xi = 1.0 / PHI  # Correlation length ξ = 1/φ
        error_rate = math.exp(-braid_depth / xi)
        # Quality threshold: < 10^-6 is "quantum error correction ready"
        qec_ready = error_rate < 1e-6
        return {
            "braid_depth": braid_depth,
            "correlation_length": xi,
            "error_rate": error_rate,
            "log10_error": math.log10(error_rate) if error_rate > 0 else float('-inf'),
            "qec_ready": qec_ready,
            "suppression_factor": 1.0 / error_rate if error_rate > 0 else float('inf'),
        }

    @staticmethod
    def topological_noise_resilience(gate: QuantumGate, noise_levels: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Analyze how a sacred gate state degrades under thermal noise.

        Applies depolarizing noise ε to the Bloch vector: |r| → (1-ε)|r|
        and measures purity degradation: γ = (1-ε)²|r|² + (1-(1-ε)²)/2
        """
        if gate.num_qubits != 1:
            return {"error": "Noise analysis requires single-qubit gate"}
        if noise_levels is None:
            noise_levels = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]

        bloch = GateAlgebra.bloch_manifold_state(gate)
        r0 = bloch["magnitude"]

        results = []
        for eps in noise_levels:
            r_noisy = r0 * (1 - eps)
            purity = (r_noisy ** 2 + 1) / 2  # For single-qubit depolarizing
            results.append({
                "noise_level": eps,
                "bloch_magnitude": r_noisy,
                "purity": purity,
                "fidelity": (1 + r_noisy) / 2,
            })
        return {
            "gate_name": gate.name,
            "initial_bloch_magnitude": r0,
            "noise_analysis": results,
        }

    @staticmethod
    def full_sacred_analysis(gate: QuantumGate) -> Dict[str, Any]:
        """
        Complete sacred gate analysis combining alignment, unitarity,
        reversibility, Bloch manifold, and topological protection.
        """
        analysis = {
            "gate_name": gate.name,
            "gate_type": gate.gate_type.name if gate.gate_type else "UNKNOWN",
            "num_qubits": gate.num_qubits,
            "sacred_alignment": GateAlgebra.sacred_alignment_score(gate),
            "unitarity": GateAlgebra.verify_unitarity(gate),
            "reversibility": GateAlgebra.verify_reversibility(gate),
        }
        if gate.num_qubits == 1:
            analysis["bloch_manifold"] = GateAlgebra.bloch_manifold_state(gate)
            analysis["noise_resilience"] = GateAlgebra.topological_noise_resilience(gate)
        return analysis


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
