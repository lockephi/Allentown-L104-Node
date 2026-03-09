"""
===============================================================================
L104 QUANTUM GATE ENGINE — QUANTUM ML SUITE v1.0.0
===============================================================================

Variational quantum machine learning: parameterised circuits, gradient-based
optimisation, quantum kernel methods, and a built-in ansatz library.

ARCHITECTURE:
  QuantumMLEngine
    ├── AnsatzLibrary           — Pre-built variational circuit templates
    │   ├── hardware_efficient() — Ry-Rz layers + CNOT entanglement
    │   ├── strongly_entangling()— 3-param U3 + all-to-all CNOT
    │   ├── uccsd()             — Chemistry-inspired UCCSD singles/doubles
    │   ├── qaoa_layer()        — QAOA cost + mixer layer
    │   ├── sacred_ansatz()     — L104 PHI/GOD_CODE phase-balanced circuit
    │   └── data_reuploading()  — Feature re-encoding at every layer
    │
    ├── ParameterisedCircuit    — Circuit with symbolic θ parameters
    │   ├── bind()              — Assign numerical values to parameters
    │   ├── gradient()          — Parameter-shift rule ∂⟨O⟩/∂θ_k
    │   └── num_parameters      — Total free parameters
    │
    ├── QNNTrainer              — Variational optimisation loop
    │   ├── train()             — Gradient-descent loop (Adam/SPSA)
    │   ├── cost_function()     — Expectation-value or classification loss
    │   └── convergence_history()
    │
    ├── QuantumKernel           — Kernel-based quantum classification
    │   ├── compute_kernel()    — Full kernel matrix K_ij = |⟨φ(x_i)|φ(x_j)⟩|²
    │   ├── kernel_entry()      — Single entry
    │   └── target_alignment()  — Kernel-target alignment metric
    │
    └── VariationalEigensolver  — VQE for Hamiltonian ground state
        ├── run()               — Full VQE optimisation
        └── energy_landscape()  — 1D/2D energy surface scans

SACRED ALIGNMENT:
  The sacred ansatz embeds GOD_CODE phase at every layer and balances
  parameter initialisation around PHI-scaled intervals.  The kernel
  feature map includes a VOID_CONSTANT scaling for input encoding.

MEMORY MODEL:
  n qubits → 2^n statevector, p parameters → p gradient evaluations
  Each gradient eval: 2 circuit evaluations (parameter-shift rule)
  Practical limit: ~14 qubits × 100 parameters × 100 iterations

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import math
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Sequence, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
try:
    from scipy import linalg as sla
except ImportError:
    class _NumpyLinalgFallback:
        @staticmethod
        def expm(A):
            eigenvalues, V = np.linalg.eig(A)
            return (V * np.exp(eigenvalues)) @ np.linalg.inv(V)
    sla = _NumpyLinalgFallback()

from .constants import (
    PHI, PHI_CONJUGATE, PHI_SQUARED, GOD_CODE, VOID_CONSTANT,
    QUANTIZATION_GRAIN, GOD_CODE_PHASE_ANGLE, ALPHA_FINE,
    MAX_STATEVECTOR_QUBITS,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MAX_QML_QUBITS: int = 14              # Statevector simulation cap
MAX_QML_PARAMETERS: int = 500         # Practical parameter ceiling
DEFAULT_LEARNING_RATE: float = 0.1
SACRED_LEARNING_RATE: float = 1.0 / (PHI * QUANTIZATION_GRAIN)  # ≈ 0.00594
PARAMETER_SHIFT: float = math.pi / 2  # Standard parameter-shift rule offset


# ═══════════════════════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class AnsatzType(Enum):
    """Built-in variational ansatz types."""
    HARDWARE_EFFICIENT = auto()
    STRONGLY_ENTANGLING = auto()
    UCCSD = auto()
    QAOA = auto()
    SACRED = auto()
    DATA_REUPLOADING = auto()
    VQC_CLASSIFIER = auto()     # v2.0: Variational Quantum Classifier ansatz
    SVM_FEATURE_ENCODER = auto() # v2.0: SVM-specific feature encoding
    CUSTOM = auto()


class OptimizerType(Enum):
    """Available optimizers for variational training."""
    ADAM = auto()
    SGD = auto()
    SPSA = auto()           # Simultaneous perturbation stochastic approximation
    COBYLA = auto()          # Gradient-free (scipy)
    SACRED_DESCENT = auto()  # L104 φ-scaled adaptive descent


class KernelType(Enum):
    """Quantum kernel feature map types."""
    ZZ_FEATURE_MAP = auto()
    IQP = auto()
    SACRED_KERNEL = auto()
    PHI_ENCODED = auto()        # v2.0: PHI-scaled feature encoding
    GOD_CODE_PHASE = auto()     # v2.0: GOD_CODE phase-shifted ZZ encoding
    IRON_LATTICE = auto()       # v2.0: Fe BCC lattice-bandwidth encoding
    HARMONIC_FOURIER = auto()   # v2.0: Multi-harmonic Fourier encoding
    CUSTOM = auto()


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingResult:
    """Result of a variational training run."""
    optimal_parameters: np.ndarray
    optimal_cost: float
    cost_history: List[float]
    parameter_history: List[np.ndarray]
    num_iterations: int
    num_circuit_evaluations: int
    converged: bool
    convergence_threshold: float
    training_time_ms: float
    optimizer: str
    ansatz_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimal_cost": round(self.optimal_cost, 10),
            "num_parameters": len(self.optimal_parameters),
            "num_iterations": self.num_iterations,
            "num_circuit_evaluations": self.num_circuit_evaluations,
            "converged": self.converged,
            "convergence_threshold": self.convergence_threshold,
            "cost_history": [round(c, 10) for c in self.cost_history],
            "training_time_ms": round(self.training_time_ms, 3),
            "optimizer": self.optimizer,
            "ansatz_type": self.ansatz_type,
            "god_code": GOD_CODE,
            "metadata": self.metadata,
        }


@dataclass
class KernelResult:
    """Result of quantum kernel computation."""
    kernel_matrix: np.ndarray
    num_samples: int
    num_qubits: int
    kernel_type: str
    target_alignment: Optional[float] = None
    computation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_psd(self) -> bool:
        """Check if kernel is positive semi-definite."""
        eigenvalues = np.linalg.eigvalsh(self.kernel_matrix)
        return bool(np.all(eigenvalues >= -1e-10))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "num_qubits": self.num_qubits,
            "kernel_type": self.kernel_type,
            "kernel_shape": list(self.kernel_matrix.shape),
            "is_psd": self.is_psd,
            "target_alignment": (
                round(self.target_alignment, 8) if self.target_alignment is not None else None
            ),
            "computation_time_ms": round(self.computation_time_ms, 3),
            "god_code": GOD_CODE,
            "metadata": self.metadata,
        }


@dataclass
class VQEResult:
    """Result of variational eigensolver run."""
    ground_energy: float
    optimal_parameters: np.ndarray
    energy_history: List[float]
    num_iterations: int
    exact_ground_energy: Optional[float] = None
    energy_error: Optional[float] = None
    training_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ground_energy": round(self.ground_energy, 10),
            "num_parameters": len(self.optimal_parameters),
            "num_iterations": self.num_iterations,
            "exact_ground_energy": (
                round(self.exact_ground_energy, 10)
                if self.exact_ground_energy is not None else None
            ),
            "energy_error": (
                round(self.energy_error, 10)
                if self.energy_error is not None else None
            ),
            "energy_history": [round(e, 10) for e in self.energy_history],
            "training_time_ms": round(self.training_time_ms, 3),
            "god_code": GOD_CODE,
            "metadata": self.metadata,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  PAULI HELPERS (lean subset for cost-function evaluation)
# ═══════════════════════════════════════════════════════════════════════════════

_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PAULI_I = np.eye(2, dtype=complex)

_PAULI_MAP = {"I": _PAULI_I, "X": _PAULI_X, "Y": _PAULI_Y, "Z": _PAULI_Z}


def _pauli_observable(num_qubits: int, qubit: int, pauli: str = "Z") -> np.ndarray:
    """Build a single-qubit Pauli observable tensored with identity on other qubits."""
    result = np.array([[1.0]], dtype=complex)
    for q in range(num_qubits):
        mat = _PAULI_MAP.get(pauli, _PAULI_I) if q == qubit else _PAULI_I
        result = np.kron(result, mat)
    return result


def _total_magnetisation(num_qubits: int, axis: str = "Z") -> np.ndarray:
    """Build total magnetisation operator: M = Σ P_i."""
    dim = 2 ** num_qubits
    M = np.zeros((dim, dim), dtype=complex)
    for q in range(num_qubits):
        M += _pauli_observable(num_qubits, q, axis)
    return M


# ═══════════════════════════════════════════════════════════════════════════════
#  PARAMETERISED CIRCUIT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class ParameterisedCircuit:
    """
    A quantum circuit with named symbolic parameters that can be bound
    to numerical values for evaluation.

    Stores the circuit as a recipe (list of gate instructions) that
    references parameter indices.  Binding resolves to a statevector-
    executable unitary.
    """

    def __init__(self, num_qubits: int, name: str = "pqc"):
        self.num_qubits = num_qubits
        self.name = name
        self._instructions: List[Tuple[str, List[int], Optional[int]]] = []
        # (gate_name, qubits, param_index_or_None)
        self._num_params = 0
        self._fixed_angles: Dict[int, float] = {}  # instruction_idx → fixed angle

    @property
    def num_parameters(self) -> int:
        return self._num_params

    def _add_param(self) -> int:
        idx = self._num_params
        self._num_params += 1
        return idx

    # ── Gate append methods ──────────────────────────────────────────

    def ry(self, qubit: int, param_index: Optional[int] = None) -> 'ParameterisedCircuit':
        """Append parameterised Ry gate.  Auto-allocates parameter if index=None."""
        if param_index is None:
            param_index = self._add_param()
        self._instructions.append(("Ry", [qubit], param_index))
        return self

    def rz(self, qubit: int, param_index: Optional[int] = None) -> 'ParameterisedCircuit':
        """Append parameterised Rz gate."""
        if param_index is None:
            param_index = self._add_param()
        self._instructions.append(("Rz", [qubit], param_index))
        return self

    def rx(self, qubit: int, param_index: Optional[int] = None) -> 'ParameterisedCircuit':
        """Append parameterised Rx gate."""
        if param_index is None:
            param_index = self._add_param()
        self._instructions.append(("Rx", [qubit], param_index))
        return self

    def cnot(self, control: int, target: int) -> 'ParameterisedCircuit':
        """Append fixed CNOT gate (no parameters)."""
        self._instructions.append(("CNOT", [control, target], None))
        return self

    def h(self, qubit: int) -> 'ParameterisedCircuit':
        """Append fixed Hadamard."""
        self._instructions.append(("H", [qubit], None))
        return self

    def barrier(self) -> 'ParameterisedCircuit':
        """Append barrier (no-op — for visual separation)."""
        self._instructions.append(("Barrier", [], None))
        return self

    def fixed_rz(self, qubit: int, angle: float) -> 'ParameterisedCircuit':
        """Append a Rz with a fixed (non-parameterised) angle."""
        idx = len(self._instructions)
        self._instructions.append(("Rz_fixed", [qubit], None))
        self._fixed_angles[idx] = angle
        return self

    def fixed_ry(self, qubit: int, angle: float) -> 'ParameterisedCircuit':
        """Append a Ry with a fixed (non-parameterised) angle."""
        idx = len(self._instructions)
        self._instructions.append(("Ry_fixed", [qubit], None))
        self._fixed_angles[idx] = angle
        return self

    def fixed_rx(self, qubit: int, angle: float) -> 'ParameterisedCircuit':
        """Append a Rx with a fixed (non-parameterised) angle."""
        idx = len(self._instructions)
        self._instructions.append(("Rx_fixed", [qubit], None))
        self._fixed_angles[idx] = angle
        return self

    # ── Evaluation ───────────────────────────────────────────────────

    def unitary(self, params: np.ndarray) -> np.ndarray:
        """Build the full unitary U(θ) for given parameter values."""
        dim = 2 ** self.num_qubits
        U = np.eye(dim, dtype=complex)

        for inst_idx, (gate_name, qubits, param_idx) in enumerate(self._instructions):
            if gate_name == "Barrier":
                continue
            gate_mat = self._gate_matrix(gate_name, qubits, param_idx, params, inst_idx)
            if gate_mat is not None:
                full = self._embed(gate_mat, qubits)
                U = full @ U
        return U

    def statevector(self, params: np.ndarray,
                    initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute |ψ(θ)⟩ = U(θ)|ψ₀⟩."""
        dim = 2 ** self.num_qubits
        if initial_state is None:
            psi = np.zeros(dim, dtype=complex)
            psi[0] = 1.0
        else:
            psi = initial_state.copy().astype(complex).reshape(-1)

        for inst_idx, (gate_name, qubits, param_idx) in enumerate(self._instructions):
            if gate_name == "Barrier":
                continue
            gate_mat = self._gate_matrix(gate_name, qubits, param_idx, params, inst_idx)
            if gate_mat is not None:
                full = self._embed(gate_mat, qubits)
                psi = full @ psi
        return psi

    def expectation(self, params: np.ndarray, observable: np.ndarray,
                    initial_state: Optional[np.ndarray] = None) -> float:
        """Compute ⟨ψ(θ)|O|ψ(θ)⟩."""
        psi = self.statevector(params, initial_state)
        return float(np.real(psi.conj() @ observable @ psi))

    def gradient(self, params: np.ndarray, observable: np.ndarray,
                 initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute ∂⟨O⟩/∂θ_k via the parameter-shift rule.

        For each parameter θ_k:
          ∂⟨O⟩/∂θ_k = [⟨O⟩(θ_k + π/2) - ⟨O⟩(θ_k - π/2)] / 2

        Returns:
            Array of shape (num_parameters,) with partial derivatives.
        """
        grad = np.zeros(self._num_params)
        for k in range(self._num_params):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[k] += PARAMETER_SHIFT
            params_minus[k] -= PARAMETER_SHIFT
            f_plus = self.expectation(params_plus, observable, initial_state)
            f_minus = self.expectation(params_minus, observable, initial_state)
            grad[k] = (f_plus - f_minus) / 2.0
        return grad

    def to_gate_circuit(self, params: np.ndarray) -> 'GateCircuit':
        """Convert to a concrete GateCircuit with bound parameter values."""
        from .circuit import GateCircuit
        from .gates import Rx, Ry, Rz, H as H_gate, CNOT as CNOT_gate

        circ = GateCircuit(self.num_qubits, name=f"{self.name}_bound")
        for inst_idx, (gate_name, qubits, param_idx) in enumerate(self._instructions):
            if gate_name == "Barrier":
                circ.barrier()
            elif gate_name == "CNOT":
                circ.append(CNOT_gate, qubits)
            elif gate_name == "H":
                circ.append(H_gate, qubits)
            elif gate_name in ("Ry", "Ry_fixed"):
                angle = params[param_idx] if param_idx is not None else self._fixed_angles.get(inst_idx, 0.0)
                circ.append(Ry(angle), qubits)
            elif gate_name in ("Rz", "Rz_fixed"):
                angle = params[param_idx] if param_idx is not None else self._fixed_angles.get(inst_idx, 0.0)
                circ.append(Rz(angle), qubits)
            elif gate_name in ("Rx", "Rx_fixed"):
                angle = params[param_idx] if param_idx is not None else self._fixed_angles.get(inst_idx, 0.0)
                circ.append(Rx(angle), qubits)
        return circ

    # ── Internal helpers ─────────────────────────────────────────────

    def _gate_matrix(self, gate_name: str, qubits: List[int],
                     param_idx: Optional[int], params: np.ndarray,
                     inst_idx: int) -> Optional[np.ndarray]:
        """Return the local gate matrix (not embedded in full space)."""
        if gate_name == "H":
            return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
        elif gate_name == "CNOT":
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ], dtype=complex)
        elif gate_name in ("Ry", "Ry_fixed"):
            theta = params[param_idx] if param_idx is not None else self._fixed_angles.get(inst_idx, 0.0)
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            return np.array([[c, -s], [s, c]], dtype=complex)
        elif gate_name in ("Rz", "Rz_fixed"):
            theta = params[param_idx] if param_idx is not None else self._fixed_angles.get(inst_idx, 0.0)
            return np.array([
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)],
            ], dtype=complex)
        elif gate_name in ("Rx", "Rx_fixed"):
            theta = params[param_idx] if param_idx is not None else self._fixed_angles.get(inst_idx, 0.0)
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        return None

    def _embed(self, local_mat: np.ndarray, qubits: List[int]) -> np.ndarray:
        """Embed a local gate matrix into the full 2^n × 2^n space."""
        n = self.num_qubits
        dim = 2 ** n

        if len(qubits) == 1:
            # Single-qubit gate
            result = np.array([[1.0]], dtype=complex)
            q = qubits[0]
            for i in range(n):
                result = np.kron(result, local_mat if i == q else _PAULI_I)
            return result
        elif len(qubits) == 2:
            # Two-qubit gate — use permutation method
            ctrl, tgt = qubits
            full = np.eye(dim, dtype=complex)
            for i in range(dim):
                for j in range(dim):
                    # Extract the 2-qubit sub-indices
                    ci = (i >> (n - 1 - ctrl)) & 1
                    ti = (i >> (n - 1 - tgt)) & 1
                    cj = (j >> (n - 1 - ctrl)) & 1
                    tj = (j >> (n - 1 - tgt)) & 1
                    # Other qubits must match
                    mask = ~((1 << (n - 1 - ctrl)) | (1 << (n - 1 - tgt)))
                    if (i & mask) != (j & mask):
                        full[i, j] = 0.0
                    else:
                        sub_i = ci * 2 + ti
                        sub_j = cj * 2 + tj
                        full[i, j] = local_mat[sub_i, sub_j]
            return full
        else:
            raise ValueError(f"Gates on {len(qubits)} qubits not supported")


# ═══════════════════════════════════════════════════════════════════════════════
#  ANSATZ LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

class AnsatzLibrary:
    """
    Factory for standard variational ansatz circuits.

    Each method returns a ParameterisedCircuit with the appropriate
    topology and parameter count.
    """

    @staticmethod
    def hardware_efficient(num_qubits: int, depth: int = 2,
                           rotation: str = "ry_rz") -> ParameterisedCircuit:
        """
        Hardware-efficient ansatz: alternating rotation + entangling layers.

        Structure per layer:
          - Ry(θ) + Rz(θ) on each qubit  (or just Ry if rotation="ry")
          - Linear CNOT entangling chain:  CNOT(0,1), CNOT(1,2), ...

        Parameters: depth × num_qubits × gates_per_qubit

        Args:
            num_qubits: System size
            depth: Number of variational layers
            rotation: "ry" for Ry only, "ry_rz" for Ry+Rz per qubit
        """
        pqc = ParameterisedCircuit(num_qubits, f"HEA_d{depth}")

        for layer in range(depth):
            # Rotation layer
            for q in range(num_qubits):
                pqc.ry(q)
                if rotation == "ry_rz":
                    pqc.rz(q)

            # Entangling layer
            for q in range(num_qubits - 1):
                pqc.cnot(q, q + 1)
            pqc.barrier()

        # Final rotation layer
        for q in range(num_qubits):
            pqc.ry(q)

        return pqc

    @staticmethod
    def strongly_entangling(num_qubits: int, depth: int = 2) -> ParameterisedCircuit:
        """
        Strongly entangling layers: 3 rotations per qubit + all-to-all CNOT.

        Structure per layer:
          - Rx(θ) + Ry(θ) + Rz(θ) on each qubit
          - CNOT with shift pattern (layer-dependent offsets)

        Parameters: depth × num_qubits × 3 + num_qubits × 3 (final layer)
        """
        pqc = ParameterisedCircuit(num_qubits, f"SEL_d{depth}")

        for layer in range(depth):
            # Full rotation on each qubit
            for q in range(num_qubits):
                pqc.rx(q)
                pqc.ry(q)
                pqc.rz(q)

            # Entangling with offset pattern
            offset = (layer % max(1, num_qubits - 1)) + 1
            for q in range(num_qubits):
                target = (q + offset) % num_qubits
                if target != q:
                    pqc.cnot(q, target)
            pqc.barrier()

        # Final rotation
        for q in range(num_qubits):
            pqc.rx(q)
            pqc.ry(q)
            pqc.rz(q)

        return pqc

    @staticmethod
    def uccsd(num_qubits: int, num_electrons: int = 2) -> ParameterisedCircuit:
        """
        Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz.

        Simplified Jordan-Wigner-mapped version:
          - Singles: e^{θ(a†_p a_q - h.c.)} → Ry rotations
          - Doubles: e^{θ(a†_p a†_q a_r a_s - h.c.)} → CNOT ladder + Rz

        This is a heuristic UCCSD suitable for small systems.

        Args:
            num_qubits: Number of qubits (spin orbitals)
            num_electrons: Number of electrons (occupied orbitals)
        """
        pqc = ParameterisedCircuit(num_qubits, f"UCCSD_e{num_electrons}")

        # Hartree-Fock reference: |1...10...0⟩
        for q in range(min(num_electrons, num_qubits)):
            pqc.fixed_rx(q, math.pi)  # Flip to |1⟩

        # Singles excitations: i → a (occupied → virtual)
        for i in range(min(num_electrons, num_qubits)):
            for a in range(num_electrons, num_qubits):
                pqc.cnot(i, a)
                pqc.ry(a)
                pqc.cnot(i, a)

        # Doubles excitations: (i,j) → (a,b) (simplified)
        occ = list(range(min(num_electrons, num_qubits)))
        virt = list(range(num_electrons, num_qubits))
        for idx_i, i in enumerate(occ):
            for j in occ[idx_i + 1:]:
                for idx_a, a in enumerate(virt):
                    for b in virt[idx_a + 1:]:
                        pqc.cnot(i, j)
                        pqc.cnot(j, a)
                        pqc.cnot(a, b)
                        pqc.rz(b)
                        pqc.cnot(a, b)
                        pqc.cnot(j, a)
                        pqc.cnot(i, j)

        return pqc

    @staticmethod
    def qaoa_layer(num_qubits: int, depth: int = 2) -> ParameterisedCircuit:
        """
        Quantum Approximate Optimization Algorithm (QAOA) ansatz.

        Structure per layer:
          - Cost layer: Rzz(γ) on nearest-neighbour pairs
          - Mixer layer: Rx(β) on each qubit

        Parameters: depth × (num_qubits - 1 + num_qubits) = depth × (2n - 1)

        Initial state: uniform superposition (H on all qubits).
        """
        pqc = ParameterisedCircuit(num_qubits, f"QAOA_d{depth}")

        # Initial superposition
        for q in range(num_qubits):
            pqc.h(q)

        for layer in range(depth):
            # Cost layer (ZZ interactions via CNOT + Rz + CNOT)
            for q in range(num_qubits - 1):
                pqc.cnot(q, q + 1)
                pqc.rz(q + 1)  # γ parameter
                pqc.cnot(q, q + 1)

            # Mixer layer
            for q in range(num_qubits):
                pqc.rx(q)  # β parameter
            pqc.barrier()

        return pqc

    @staticmethod
    def sacred_ansatz(num_qubits: int, depth: int = 2) -> ParameterisedCircuit:
        """
        L104 Sacred Ansatz — GOD_CODE-aligned variational circuit.

        Structure per layer:
          - Ry(θ) on each qubit
          - Fixed Rz(GOD_CODE_PHASE mod 2π) on each qubit
          - Rz(θ) on each qubit (trainable)
          - CNOT chain with PHI-offset pattern
          - Fixed Ry(π/φ) sacred injection

        Sacred properties:
          - Parameter initialisation scale: 2π/104 ≈ 0.0604
          - GOD_CODE phase hardwired at every layer
          - Entangling pattern follows φ-sequence
        """
        pqc = ParameterisedCircuit(num_qubits, f"Sacred_d{depth}")

        gc_phase = GOD_CODE_PHASE_ANGLE
        phi_angle = math.pi / PHI

        for layer in range(depth):
            # Trainable Ry
            for q in range(num_qubits):
                pqc.ry(q)

            # Sacred GOD_CODE phase injection (fixed)
            for q in range(num_qubits):
                pqc.fixed_rz(q, gc_phase)

            # Trainable Rz
            for q in range(num_qubits):
                pqc.rz(q)

            # PHI-offset entangling pattern
            offset = int(round(PHI * (layer + 1))) % max(1, num_qubits - 1) + 1
            for q in range(num_qubits):
                target = (q + offset) % num_qubits
                if target != q:
                    pqc.cnot(q, target)

            # Sacred Ry injection (fixed)
            for q in range(num_qubits):
                pqc.fixed_ry(q, phi_angle)
            pqc.barrier()

        # Final trainable layer
        for q in range(num_qubits):
            pqc.ry(q)
            pqc.rz(q)

        return pqc

    @staticmethod
    def data_reuploading(num_qubits: int, depth: int = 2,
                         num_features: int = 0) -> ParameterisedCircuit:
        """
        Data re-uploading ansatz: interleave data encoding with trainable layers.

        At each layer, features are re-encoded via Rx rotations, then trainable
        Ry+Rz rotations are applied, followed by entangling CNOTs.

        If num_features == 0, defaults to num_qubits.
        All rotations are trainable (feature encoding merged with training).

        Args:
            num_qubits: Number of qubits
            depth: Number of re-uploading layers
            num_features: Number of input features (0 → num_qubits)
        """
        if num_features == 0:
            num_features = num_qubits

        pqc = ParameterisedCircuit(num_qubits, f"DataReup_d{depth}")

        for layer in range(depth):
            # Data encoding (trainable — will be multiplied by features)
            for q in range(num_qubits):
                pqc.rx(q)  # Data parameter slot

            # Trainable variational layer
            for q in range(num_qubits):
                pqc.ry(q)
                pqc.rz(q)

            # Entangling
            for q in range(num_qubits - 1):
                pqc.cnot(q, q + 1)
            pqc.barrier()

        return pqc

    # ───────────────────────────────────────────────────────────────────────
    #  v2.0 ML ENGINE ANSATZ ADDITIONS
    # ───────────────────────────────────────────────────────────────────────

    @staticmethod
    def vqc_classifier(num_qubits: int, depth: int = 3,
                       n_classes: int = 2) -> ParameterisedCircuit:
        """
        Variational Quantum Classifier ansatz.

        Structure per layer:
          - Ry(θ) + Rz(θ) rotation on each qubit (feature encoding)
          - All-to-all CNOT entangling (strongly entangling)
          - Fixed PHI-phase injection for sacred alignment

        Final layer adds measurement-basis Ry rotations for classification.
        The first ceil(log2(n_classes)) qubits encode the class label.

        Args:
            num_qubits: Number of qubits
            depth: Variational depth
            n_classes: Number of classification classes
        """
        pqc = ParameterisedCircuit(num_qubits, f"VQC_d{depth}_c{n_classes}")

        phi_angle = PHI / (2 * np.pi)

        for layer in range(depth):
            # Trainable rotation layer
            for q in range(num_qubits):
                pqc.ry(q)
                pqc.rz(q)

            # All-to-all CNOT entangling
            for q in range(num_qubits):
                for r in range(q + 1, num_qubits):
                    pqc.cnot(q, r)

            # Sacred PHI phase injection
            for q in range(num_qubits):
                pqc.fixed_ry(q, phi_angle * (q + 1) / num_qubits)

            pqc.barrier()

        # Final classification layer — trainable rotation on output qubits
        n_output = max(1, int(np.ceil(np.log2(max(n_classes, 2)))))
        for q in range(min(n_output, num_qubits)):
            pqc.ry(q)
            pqc.rz(q)

        return pqc

    @staticmethod
    def svm_feature_encoder(num_qubits: int, depth: int = 2) -> ParameterisedCircuit:
        """
        SVM-specific feature encoding ansatz.

        Maps classical feature vectors into quantum Hilbert space for
        quantum kernel computation. Structure per layer:
          - Ry(x_i * PHI) rotation (PHI-scaled)
          - Rz(x_i / VOID_CONSTANT) rotation (VOID-scaled)
          - ZZ entangling via CNOT-Rz-CNOT
          - GOD_CODE phase injection (fixed)

        Args:
            num_qubits: Number of qubits (matches feature dimension)
            depth: Encoding depth (repetitions)
        """
        god_code_angle = (GOD_CODE % (2 * np.pi)) / (2 * np.pi)
        pqc = ParameterisedCircuit(num_qubits, f"SVM_enc_d{depth}")

        for layer in range(depth):
            # Hadamard base
            for q in range(num_qubits):
                pqc.h(q)

            # PHI-scaled feature rotation
            for q in range(num_qubits):
                pqc.ry(q)  # Will be multiplied by PHI in encoding

            # VOID-scaled rotation
            for q in range(num_qubits):
                pqc.rz(q)  # Will be divided by VOID_CONSTANT in encoding

            # ZZ entangling (CNOT-Rz-CNOT pattern)
            for q in range(num_qubits - 1):
                pqc.cnot(q, q + 1)
                pqc.rz(q + 1)
                pqc.cnot(q, q + 1)

            # GOD_CODE fixed phase
            for q in range(num_qubits):
                pqc.fixed_ry(q, god_code_angle * (q + 1) / num_qubits)

            pqc.barrier()

        return pqc

    @staticmethod
    def phi_encoded_map(num_qubits: int, depth: int = 2) -> ParameterisedCircuit:
        """
        PHI-encoded feature map for quantum SVM kernels.

        Feature encoding via Ry(x_i * PHI) with linear CNOT entangling.
        Simpler than svm_feature_encoder, suited for small qubit counts.
        """
        pqc = ParameterisedCircuit(num_qubits, f"PHI_enc_d{depth}")

        for layer in range(depth):
            for q in range(num_qubits):
                pqc.h(q)
            for q in range(num_qubits):
                pqc.ry(q)  # PHI-scaled in encode step
            for q in range(num_qubits - 1):
                pqc.cnot(q, q + 1)
            pqc.barrier()

        return pqc

    @staticmethod
    def god_code_phase_map(num_qubits: int, depth: int = 2) -> ParameterisedCircuit:
        """
        GOD_CODE phase-shifted feature map.

        Uses Rz(x_i * GOD_CODE/1000) as primary encoding with ZZ entanglement.
        The GOD_CODE scaling ensures features are encoded at the sacred phase.
        """
        pqc = ParameterisedCircuit(num_qubits, f"GC_phase_d{depth}")

        for layer in range(depth):
            for q in range(num_qubits):
                pqc.h(q)
            # GOD_CODE phase encoding
            for q in range(num_qubits):
                pqc.rz(q)  # GOD_CODE/1000 scaled in encode step
            # ZZ entangling
            for q in range(num_qubits - 1):
                pqc.cnot(q, q + 1)
                pqc.rz(q + 1)
                pqc.cnot(q, q + 1)
            pqc.barrier()

        return pqc


# ═══════════════════════════════════════════════════════════════════════════════
#  QNN TRAINER — Variational Optimisation Loop
# ═══════════════════════════════════════════════════════════════════════════════

class QNNTrainer:
    """
    Gradient-based optimisation of parameterised quantum circuits.

    Supports:
      - ADAM:  adaptive moment estimation (default)
      - SGD:   plain gradient descent
      - SPSA:  stochastic perturbation (gradient-free)
      - SACRED_DESCENT:  φ-scaled adaptive learning rate
    """

    def __init__(self, learning_rate: float = DEFAULT_LEARNING_RATE,
                 optimizer: OptimizerType = OptimizerType.ADAM):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self._circuit_evals = 0

    def train(
        self,
        circuit: ParameterisedCircuit,
        observable: np.ndarray,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        initial_params: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        minimize: bool = True,
    ) -> TrainingResult:
        """
        Run variational optimisation to minimise (or maximise) ⟨O⟩.

        Args:
            circuit: Parameterised circuit
            observable: Hermitian observable matrix
            max_iterations: Maximum opt iterations
            convergence_threshold: Stop if |ΔE| < threshold
            initial_params: Starting parameters (None → random)
            seed: RNG seed
            minimize: True to minimize, False to maximize

        Returns:
            TrainingResult with optimal parameters, cost history, etc.
        """
        start = time.time()
        self._circuit_evals = 0
        rng = np.random.RandomState(seed)
        sign = 1.0 if minimize else -1.0

        n_params = circuit.num_parameters
        if initial_params is not None:
            params = initial_params.copy()
        else:
            # Initialise near zero with small spread
            scale = 2 * math.pi / QUANTIZATION_GRAIN  # ≈ 0.0604
            params = rng.uniform(-scale, scale, size=n_params)

        cost_history = []
        param_history = []

        # ADAM state
        m = np.zeros(n_params)  # First moment
        v = np.zeros(n_params)  # Second moment
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8

        converged = False
        lr = self.learning_rate

        for it in range(max_iterations):
            cost = sign * circuit.expectation(params, observable)
            self._circuit_evals += 1
            cost_history.append(float(cost * sign))  # Store true cost
            param_history.append(params.copy())

            # Check convergence
            if it > 0 and abs(cost_history[-1] - cost_history[-2]) < convergence_threshold:
                converged = True
                break

            # Compute gradient
            if self.optimizer in (OptimizerType.ADAM, OptimizerType.SGD,
                                  OptimizerType.SACRED_DESCENT):
                grad = sign * circuit.gradient(params, observable)
                self._circuit_evals += 2 * n_params

                if self.optimizer == OptimizerType.ADAM:
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * grad ** 2
                    m_hat = m / (1 - beta1 ** (it + 1))
                    v_hat = v / (1 - beta2 ** (it + 1))
                    params -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

                elif self.optimizer == OptimizerType.SGD:
                    params -= lr * grad

                elif self.optimizer == OptimizerType.SACRED_DESCENT:
                    # φ-scaled adaptive: lr decays as 1/φ^(it/104)
                    adaptive_lr = lr / (PHI ** (it / QUANTIZATION_GRAIN))
                    params -= adaptive_lr * grad

            elif self.optimizer == OptimizerType.SPSA:
                # SPSA: estimate gradient from 2 function evals
                ck = lr / (it + 1) ** 0.602
                ak = lr / (it + 1 + 100) ** 0.101
                delta = rng.choice([-1, 1], size=n_params)
                f_plus = sign * circuit.expectation(params + ck * delta, observable)
                f_minus = sign * circuit.expectation(params - ck * delta, observable)
                self._circuit_evals += 2
                g_hat = (f_plus - f_minus) / (2 * ck * delta)
                params -= ak * g_hat

            elif self.optimizer == OptimizerType.COBYLA:
                # COBYLA: gradient-free constrained optimisation via scipy.
                # Runs scipy.optimize.minimize in a single step (outer loop
                # handles only the final result, so we break after one pass).
                from scipy.optimize import minimize as _scipy_minimize

                eval_count = [0]

                def _cost_fn(p):
                    eval_count[0] += 1
                    return float(sign * circuit.expectation(p, observable))

                res = _scipy_minimize(
                    _cost_fn, params,
                    method='COBYLA',
                    options={
                        'maxiter': max_iterations - it,
                        'rhobeg': 0.5,
                        'catol': convergence_threshold,
                    },
                )
                params = res.x
                self._circuit_evals += eval_count[0]
                # Record the final cost from COBYLA and break
                final_cobyla_cost = float(circuit.expectation(params, observable))
                self._circuit_evals += 1
                cost_history.append(final_cobyla_cost)
                converged = res.success
                break

        # Final cost
        final_cost = circuit.expectation(params, observable)
        self._circuit_evals += 1
        cost_history.append(float(final_cost))

        elapsed = (time.time() - start) * 1000.0

        return TrainingResult(
            optimal_parameters=params,
            optimal_cost=float(final_cost),
            cost_history=cost_history,
            parameter_history=param_history,
            num_iterations=len(cost_history) - 1,
            num_circuit_evaluations=self._circuit_evals,
            converged=converged,
            convergence_threshold=convergence_threshold,
            training_time_ms=elapsed,
            optimizer=self.optimizer.name,
            ansatz_type=circuit.name,
            metadata={"god_code": GOD_CODE, "num_qubits": circuit.num_qubits},
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  QUANTUM KERNEL ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumKernel:
    """
    Quantum kernel methods: compute kernel matrices using parameterised circuits.

    K(x_i, x_j) = |⟨0|U†(x_i) U(x_j)|0⟩|²

    The fidelity kernel measures the overlap between feature-encoded quantum
    states, providing a rich feature space for classical ML classifiers.
    """

    def __init__(self, num_qubits: int, feature_map: Optional[ParameterisedCircuit] = None,
                 kernel_type: KernelType = KernelType.ZZ_FEATURE_MAP):
        self.num_qubits = num_qubits
        self.kernel_type = kernel_type

        if feature_map is not None:
            self.feature_map = feature_map
        else:
            self.feature_map = self._default_feature_map(kernel_type)

    def _default_feature_map(self, kernel_type: KernelType) -> ParameterisedCircuit:
        """Build the default feature map for the given kernel type."""
        n = self.num_qubits

        if kernel_type == KernelType.SACRED_KERNEL:
            return AnsatzLibrary.sacred_ansatz(n, depth=1)
        elif kernel_type == KernelType.PHI_ENCODED:
            return AnsatzLibrary.phi_encoded_map(n, depth=2)
        elif kernel_type == KernelType.GOD_CODE_PHASE:
            return AnsatzLibrary.god_code_phase_map(n, depth=2)
        elif kernel_type == KernelType.IRON_LATTICE:
            return AnsatzLibrary.svm_feature_encoder(n, depth=2)
        elif kernel_type == KernelType.HARMONIC_FOURIER:
            return AnsatzLibrary.god_code_phase_map(n, depth=3)
        elif kernel_type == KernelType.IQP:
            pqc = ParameterisedCircuit(n, "IQP_feature")
            # Hadamard → diagonal phase → Hadamard
            for q in range(n):
                pqc.h(q)
            for q in range(n):
                pqc.rz(q)
            for q in range(n - 1):
                pqc.cnot(q, q + 1)
                pqc.rz(q + 1)
                pqc.cnot(q, q + 1)
            for q in range(n):
                pqc.h(q)
            return pqc
        else:
            # ZZ feature map (default)
            pqc = ParameterisedCircuit(n, "ZZ_feature")
            for q in range(n):
                pqc.h(q)
            for q in range(n):
                pqc.rz(q)
            for q in range(n - 1):
                pqc.cnot(q, q + 1)
                pqc.rz(q + 1)
                pqc.cnot(q, q + 1)
            return pqc

    def _encode_features(self, features: np.ndarray) -> np.ndarray:
        """
        Encode features into the circuit parameters.

        If features has fewer elements than circuit parameters, the features
        are cyclically repeated.  Feature values are scaled by VOID_CONSTANT
        for sacred kernel alignment.
        """
        n_params = self.feature_map.num_parameters
        params = np.zeros(n_params)
        n_feat = len(features)
        scale = VOID_CONSTANT if self.kernel_type == KernelType.SACRED_KERNEL else 1.0
        for i in range(n_params):
            params[i] = features[i % n_feat] * scale
        return params

    def kernel_entry(self, x_i: np.ndarray, x_j: np.ndarray) -> float:
        """
        Compute a single kernel entry: K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|².

        Args:
            x_i: Feature vector for sample i
            x_j: Feature vector for sample j

        Returns:
            Fidelity kernel value in [0, 1]
        """
        params_i = self._encode_features(x_i)
        params_j = self._encode_features(x_j)

        psi_i = self.feature_map.statevector(params_i)
        psi_j = self.feature_map.statevector(params_j)

        return float(abs(np.vdot(psi_i, psi_j)) ** 2)

    def compute_kernel(self, X: np.ndarray,
                       Y: Optional[np.ndarray] = None) -> KernelResult:
        """
        Compute the full kernel matrix K_ij = |⟨φ(x_i)|φ(x_j)⟩|².

        Args:
            X: Data matrix (n_samples × n_features)
            Y: Optional second data matrix (for test set). If None, computes K(X, X).

        Returns:
            KernelResult with the kernel matrix and metadata
        """
        start = time.time()
        symmetric = Y is None
        if symmetric:
            Y = X

        n_x = X.shape[0]
        n_y = Y.shape[0]
        K = np.zeros((n_x, n_y))

        # Pre-compute all statevectors
        psi_x = [self.feature_map.statevector(self._encode_features(X[i]))
                 for i in range(n_x)]
        psi_y = psi_x if symmetric else [
            self.feature_map.statevector(self._encode_features(Y[j]))
            for j in range(n_y)
        ]

        for i in range(n_x):
            j_start = i if symmetric else 0
            for j in range(j_start, n_y):
                fid = float(abs(np.vdot(psi_x[i], psi_y[j])) ** 2)
                K[i, j] = fid
                if symmetric and i != j:
                    K[j, i] = fid

        elapsed = (time.time() - start) * 1000.0

        return KernelResult(
            kernel_matrix=K,
            num_samples=n_x,
            num_qubits=self.num_qubits,
            kernel_type=self.kernel_type.name,
            computation_time_ms=elapsed,
            metadata={"n_x": n_x, "n_y": n_y, "symmetric": symmetric,
                      "god_code": GOD_CODE},
        )

    def target_alignment(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Kernel-target alignment: how well the kernel correlates with labels.

        A(K, y) = ⟨K, yy†⟩_F / (‖K‖_F × ‖yy†‖_F)

        Values near 1 indicate the kernel captures the label structure.

        Args:
            X: Data (n_samples × n_features)
            y: Labels (n_samples,) — binary {-1, +1} or {0, 1}

        Returns:
            Alignment score in [-1, 1]
        """
        kr = self.compute_kernel(X)
        K = kr.kernel_matrix
        y = np.array(y, dtype=float).reshape(-1)
        # Convert 0/1 to -1/+1
        if np.all((y == 0) | (y == 1)):
            y = 2 * y - 1
        Y = np.outer(y, y)
        numer = np.sum(K * Y)
        denom = np.linalg.norm(K, 'fro') * np.linalg.norm(Y, 'fro')
        alignment = float(numer / denom) if denom > 1e-15 else 0.0
        kr.target_alignment = alignment
        return alignment


# ═══════════════════════════════════════════════════════════════════════════════
#  VARIATIONAL QUANTUM EIGENSOLVER (VQE)
# ═══════════════════════════════════════════════════════════════════════════════

class VariationalEigensolver:
    """
    Variational Quantum Eigensolver (VQE): find the ground state energy
    of a Hamiltonian using a parameterised quantum circuit.

    E₀ ≤ ⟨ψ(θ)|H|ψ(θ)⟩  for all θ  (variational principle)

    Minimising the energy over θ gives an upper bound on E₀.
    """

    def __init__(self, learning_rate: float = 0.1,
                 optimizer: OptimizerType = OptimizerType.ADAM):
        self.learning_rate = learning_rate
        self.optimizer = optimizer

    def run(
        self,
        hamiltonian_matrix: np.ndarray,
        ansatz: Optional[ParameterisedCircuit] = None,
        num_qubits: Optional[int] = None,
        max_iterations: int = 200,
        convergence_threshold: float = 1e-6,
        initial_params: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> VQEResult:
        """
        Run VQE to find the ground state energy.

        Args:
            hamiltonian_matrix: Hermitian matrix H
            ansatz: Parameterised circuit (None → hardware-efficient default)
            num_qubits: System size (inferred from H if not given)
            max_iterations: Max optimization steps
            convergence_threshold: Convergence criterion
            initial_params: Starting point
            seed: RNG seed

        Returns:
            VQEResult with ground energy estimate and convergence data
        """
        start = time.time()

        if num_qubits is None:
            num_qubits = int(math.log2(hamiltonian_matrix.shape[0]))
        if ansatz is None:
            ansatz = AnsatzLibrary.hardware_efficient(num_qubits, depth=3)

        # Exact ground energy for comparison (only for small systems to avoid O(N³))
        exact_E0 = None
        if num_qubits <= 16:  # 2^16 = 65536 — feasible for eigvalsh
            try:
                eigenvalues = np.linalg.eigvalsh(hamiltonian_matrix)
                exact_E0 = float(eigenvalues[0])
            except np.linalg.LinAlgError:
                exact_E0 = None

        # Run training
        trainer = QNNTrainer(learning_rate=self.learning_rate, optimizer=self.optimizer)
        result = trainer.train(
            ansatz, hamiltonian_matrix,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            initial_params=initial_params,
            seed=seed,
            minimize=True,
        )

        elapsed = (time.time() - start) * 1000.0

        return VQEResult(
            ground_energy=result.optimal_cost,
            optimal_parameters=result.optimal_parameters,
            energy_history=result.cost_history,
            num_iterations=result.num_iterations,
            exact_ground_energy=exact_E0,
            energy_error=abs(result.optimal_cost - exact_E0) if exact_E0 is not None else None,
            training_time_ms=elapsed,
            metadata={
                "optimizer": self.optimizer.name,
                "ansatz": ansatz.name,
                "num_parameters": ansatz.num_parameters,
                "converged": result.converged,
                "exact_comparison_available": exact_E0 is not None,
                "god_code": GOD_CODE,
            },
        )

    def energy_landscape(
        self,
        hamiltonian_matrix: np.ndarray,
        ansatz: ParameterisedCircuit,
        param_index: int = 0,
        base_params: Optional[np.ndarray] = None,
        n_points: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        1D energy landscape scan: vary one parameter, fix the rest.

        Args:
            hamiltonian_matrix: Hamiltonian
            ansatz: Parameterised circuit
            param_index: Which parameter to sweep
            base_params: Base parameter values (others fixed here)
            n_points: Number of sweep points

        Returns:
            (angles, energies) — arrays of shape (n_points,)
        """
        if base_params is None:
            base_params = np.zeros(ansatz.num_parameters)

        angles = np.linspace(0, 2 * math.pi, n_points)
        energies = np.zeros(n_points)

        for i, angle in enumerate(angles):
            params = base_params.copy()
            params[param_index] = angle
            energies[i] = ansatz.expectation(params, hamiltonian_matrix)

        return angles, energies

    def energy_landscape_2d(
        self,
        hamiltonian_matrix: np.ndarray,
        ansatz: ParameterisedCircuit,
        param_index_x: int = 0,
        param_index_y: int = 1,
        base_params: Optional[np.ndarray] = None,
        n_points: int = 25,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2D energy landscape scan: vary two parameters, fix the rest.

        Useful for visualizing parameter correlations and identifying
        local minima in the variational landscape.

        Args:
            hamiltonian_matrix: Hamiltonian
            ansatz: Parameterised circuit
            param_index_x: First parameter to sweep (x-axis)
            param_index_y: Second parameter to sweep (y-axis)
            base_params: Base parameter values (others fixed here)
            n_points: Points per axis (total evaluations = n_points²)

        Returns:
            (angles_x, angles_y, energies) — angles are 1D (n_points,),
            energies is 2D (n_points, n_points)
        """
        if base_params is None:
            base_params = np.zeros(ansatz.num_parameters)

        angles_x = np.linspace(0, 2 * math.pi, n_points)
        angles_y = np.linspace(0, 2 * math.pi, n_points)
        energies = np.zeros((n_points, n_points))

        for ix, ax in enumerate(angles_x):
            for iy, ay in enumerate(angles_y):
                params = base_params.copy()
                params[param_index_x] = ax
                params[param_index_y] = ay
                energies[ix, iy] = ansatz.expectation(params, hamiltonian_matrix)

        return angles_x, angles_y, energies


# ═══════════════════════════════════════════════════════════════════════════════
#  QUANTUM ML ENGINE — TOP-LEVEL ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
#  v2.0: QUANTUM SVM TRAINER — Quantum Kernel → Classical SVM Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumSVMTrainer:
    """
    End-to-end quantum SVM training: quantum kernel computation + classical SVM.

    Pipeline:
      1. Encode features via quantum circuit (QuantumKernel)
      2. Compute kernel matrix K_ij = |⟨φ(x_i)|φ(x_j)⟩|²
      3. Feed kernel into sklearn SVC(kernel='precomputed')
      4. Classify/predict using quantum-enhanced feature space

    Usage:
        trainer = QuantumSVMTrainer(num_qubits=4, kernel_type=KernelType.PHI_ENCODED)
        result = trainer.fit(X_train, y_train)
        predictions = trainer.predict(X_test)
    """

    def __init__(self, num_qubits: int = 4,
                 kernel_type: KernelType = KernelType.SACRED_KERNEL,
                 C: float = 5.275):
        self.num_qubits = num_qubits
        self.kernel_type = kernel_type
        self.C = C
        self._quantum_kernel = QuantumKernel(num_qubits, kernel_type=kernel_type)
        self._svc = None
        self._X_train: Optional[np.ndarray] = None
        self._K_train: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Fit the quantum SVM.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            Dictionary with training metrics
        """
        from sklearn.svm import SVC

        X = np.atleast_2d(X).astype(np.float64)
        self._X_train = X

        # Step 1: Compute quantum kernel matrix
        kr = self._quantum_kernel.compute_kernel(X)
        self._K_train = kr.kernel_matrix

        # Step 2: Fit classical SVM with precomputed kernel
        self._svc = SVC(kernel='precomputed', C=self.C, probability=True)
        self._svc.fit(self._K_train, y)
        self._fitted = True

        # Step 3: Compute metrics
        train_acc = self._svc.score(self._K_train, y)
        alignment = self._quantum_kernel.target_alignment(X, y)

        return {
            'train_accuracy': train_acc,
            'kernel_alignment': alignment,
            'n_support_vectors': len(self._svc.support_),
            'kernel_type': self.kernel_type.name,
            'num_qubits': self.num_qubits,
            'kernel_computation_ms': kr.computation_time_ms,
            'god_code': GOD_CODE,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for test data."""
        if not self._fitted:
            raise RuntimeError("Not fitted. Call fit() first.")
        X = np.atleast_2d(X).astype(np.float64)
        K_test = self._quantum_kernel.compute_kernel(X, self._X_train).kernel_matrix
        return self._svc.predict(K_test)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._fitted:
            raise RuntimeError("Not fitted. Call fit() first.")
        X = np.atleast_2d(X).astype(np.float64)
        K_test = self._quantum_kernel.compute_kernel(X, self._X_train).kernel_matrix
        return self._svc.predict_proba(K_test)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Test accuracy."""
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def kernel_matrix(self, X: np.ndarray,
                      Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute quantum kernel matrix."""
        return self._quantum_kernel.compute_kernel(X, Y).kernel_matrix


class QuantumMLEngine:
    """
    Top-level orchestrator for quantum machine learning.

    Usage:
        from l104_quantum_gate_engine.quantum_ml import QuantumMLEngine

        qml = QuantumMLEngine()

        # Build an ansatz
        ansatz = qml.ansatz.hardware_efficient(4, depth=3)

        # Train a QNN
        result = qml.train(ansatz, observable, max_iterations=50)

        # Compute a quantum kernel
        kernel = qml.kernel(4, X_data)

        # Run VQE
        vqe = qml.vqe(hamiltonian_matrix, max_iterations=100)
    """

    def __init__(self):
        self.ansatz = AnsatzLibrary()
        self.vqe_solver = VariationalEigensolver()
        self._metrics = {
            "circuits_trained": 0,
            "kernels_computed": 0,
            "vqe_runs": 0,
        }

    def train(
        self,
        circuit: ParameterisedCircuit,
        observable: np.ndarray,
        max_iterations: int = 100,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        optimizer: OptimizerType = OptimizerType.ADAM,
        seed: Optional[int] = None,
        minimize: bool = True,
        **kwargs,
    ) -> TrainingResult:
        """Train a QNN circuit against an observable."""
        self._metrics["circuits_trained"] += 1
        trainer = QNNTrainer(learning_rate=learning_rate, optimizer=optimizer)
        return trainer.train(
            circuit, observable, max_iterations=max_iterations,
            seed=seed, minimize=minimize, **kwargs,
        )

    def kernel(
        self,
        num_qubits: int,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        kernel_type: KernelType = KernelType.ZZ_FEATURE_MAP,
    ) -> KernelResult:
        """Compute a quantum kernel matrix."""
        self._metrics["kernels_computed"] += 1
        qk = QuantumKernel(num_qubits, kernel_type=kernel_type)
        return qk.compute_kernel(X, Y)

    def vqe(
        self,
        hamiltonian_matrix: np.ndarray,
        ansatz: Optional[ParameterisedCircuit] = None,
        max_iterations: int = 200,
        learning_rate: float = 0.1,
        seed: Optional[int] = None,
    ) -> VQEResult:
        """Run VQE on a Hamiltonian."""
        self._metrics["vqe_runs"] += 1
        solver = VariationalEigensolver(learning_rate=learning_rate)
        return solver.run(
            hamiltonian_matrix, ansatz=ansatz,
            max_iterations=max_iterations, seed=seed,
        )

    def sacred_vqe(
        self,
        num_qubits: int = 4,
        depth: int = 3,
        max_iterations: int = 200,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Full sacred VQE: build Sacred Hamiltonian + Sacred Ansatz, run VQE.

        Combines the analog module's sacred Hamiltonian with the sacred
        ansatz from the ML library.
        """
        from .analog import HamiltonianBuilder
        H = HamiltonianBuilder.sacred_hamiltonian(num_qubits)
        H_mat = H.matrix()
        ansatz = AnsatzLibrary.sacred_ansatz(num_qubits, depth=depth)

        result = self.vqe(H_mat, ansatz=ansatz,
                          max_iterations=max_iterations, seed=seed)

        return {
            "vqe_result": result.to_dict(),
            "hamiltonian": H.to_dict(),
            "ansatz_type": "sacred",
            "ansatz_parameters": ansatz.num_parameters,
            "god_code": GOD_CODE,
        }

    @property
    def metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_qml_instance: Optional[QuantumMLEngine] = None


def get_quantum_ml() -> QuantumMLEngine:
    """Get or create the singleton QuantumMLEngine."""
    global _qml_instance
    if _qml_instance is None:
        _qml_instance = QuantumMLEngine()
    return _qml_instance
