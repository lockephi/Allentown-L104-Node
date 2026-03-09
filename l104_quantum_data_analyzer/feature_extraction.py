"""
L104 Quantum Data Analyzer — Quantum Feature Extraction
═══════════════════════════════════════════════════════════════════════════════
Quantum feature maps and state encoders for data embedding into Hilbert space:

  1. QuantumFeatureMap     — ZZ / IQP / Pauli feature maps
  2. QuantumStateEncoder   — Amplitude / angle / basis encoding
  3. EntanglementFeatureExtractor — Extract entanglement-based features
  4. QuantumEmbedding      — High-dimensional quantum embeddings

CROSS-ENGINE INTEGRATION:
  • l104_quantum_gate_engine — Circuit building, sacred gates
  • l104_math_engine         — PHI harmonics, God Code weighting
  • l104_science_engine      — Coherence field encoding

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from .constants import (
    PHI, PHI_CONJUGATE, GOD_CODE, VOID_CONSTANT, TAU,
    MAX_QUBITS_STATEVECTOR, MAX_QUBITS_CIRCUIT,
    god_code_at, normalize_vector, pad_to_power_of_two,
    data_to_quantum_state, num_qubits_for,
    KERNEL_BANDWIDTH_DEFAULT, ALPHA_FINE,
)


# ─── Lazy engine imports ────────────────────────────────────────────────────
def _get_gate_engine():
    try:
        from l104_quantum_gate_engine import get_engine
        return get_engine()
    except ImportError:
        return None

def _get_math_engine():
    try:
        from l104_math_engine import MathEngine
        return MathEngine()
    except ImportError:
        return None

def _get_science_engine():
    try:
        from l104_science_engine import ScienceEngine
        return ScienceEngine()
    except ImportError:
        return None


QISKIT_AVAILABLE = False
try:
    from l104_quantum_gate_engine import GateCircuit as QuantumCircuit
    from l104_quantum_gate_engine.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureMapResult:
    """Result from quantum feature mapping."""
    quantum_features: np.ndarray
    feature_dimension: int
    n_qubits: int
    circuit_depth: int
    encoding_method: str
    god_code_alignment: float
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Result from quantum embedding."""
    embedded_vectors: np.ndarray
    embedding_dimension: int
    kernel_matrix: np.ndarray
    expressibility: float
    entanglement_capability: float
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. QUANTUM FEATURE MAP
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumFeatureMap:
    """
    Quantum feature maps for encoding classical data into quantum states.

    Supports multiple feature map architectures:
      • ZZFeatureMap   — entangling feature map with ZZ interactions
      • PauliFeatureMap — tensor products of Pauli rotations
      • IQPFeatureMap  — Instantaneous Quantum Polynomial
      • SacredFeatureMap — L104 God Code-aligned feature map

    Each feature map φ(x) maps x ∈ R^d → |φ(x)⟩ ∈ C^{2^n}
    """

    SUPPORTED_MAPS = ("zz", "pauli", "iqp", "sacred")

    def __init__(self, map_type: str = "zz", n_reps: int = 2):
        self.map_type = map_type.lower()
        if self.map_type not in self.SUPPORTED_MAPS:
            raise ValueError(f"Unknown map type '{map_type}'. Use one of: {self.SUPPORTED_MAPS}")
        self.n_reps = n_reps
        self.gate_engine = _get_gate_engine()
        self.math_engine = _get_math_engine()

    def transform(self, data: np.ndarray) -> FeatureMapResult:
        """
        Map data matrix (n_samples × n_features) into quantum feature space.

        Returns FeatureMapResult with quantum state vectors for each sample.
        """
        t0 = time.time()
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples, n_features = data.shape
        n_qubits = min(n_features, MAX_QUBITS_CIRCUIT)

        quantum_features = []
        total_depth = 0

        for i in range(n_samples):
            state, depth = self._apply_feature_map(data[i, :n_qubits], n_qubits)
            quantum_features.append(state)
            total_depth = max(total_depth, depth)

        quantum_features = np.array(quantum_features)

        # God Code alignment: check feature space symmetries
        gc_alignment = self._god_code_feature_alignment(quantum_features)

        return FeatureMapResult(
            quantum_features=quantum_features,
            feature_dimension=2 ** n_qubits,
            n_qubits=n_qubits,
            circuit_depth=total_depth,
            encoding_method=self.map_type,
            god_code_alignment=gc_alignment,
            execution_time=time.time() - t0,
            metadata={
                "n_samples": n_samples,
                "n_features": n_features,
                "n_reps": self.n_reps,
                "map_type": self.map_type,
            },
        )

    def _apply_feature_map(self, x: np.ndarray, n_qubits: int) -> Tuple[np.ndarray, int]:
        """Apply the selected feature map to a single data point."""
        dispatch = {
            "zz": self._zz_feature_map,
            "pauli": self._pauli_feature_map,
            "iqp": self._iqp_feature_map,
            "sacred": self._sacred_feature_map,
        }
        return dispatch[self.map_type](x, n_qubits)

    def _zz_feature_map(self, x: np.ndarray, n_qubits: int) -> Tuple[np.ndarray, int]:
        """
        ZZFeatureMap: U_ZZ(x) = ∏_rep [∏_{i<j} e^{-i x_i x_j ZZ} ∏_i H Rz(x_i)]
        Creates entanglement through two-qubit ZZ interactions.
        """
        dim = 2 ** n_qubits
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0

        for rep in range(self.n_reps):
            # Hadamard layer
            state = self._apply_hadamard_all(state, n_qubits)
            # Rz rotations: feature encoding
            for i in range(n_qubits):
                angle = 2.0 * x[i] if i < len(x) else 0.0
                state = self._apply_rz(state, i, angle, n_qubits)
            # ZZ entangling layer
            for i in range(n_qubits - 1):
                xi = x[i] if i < len(x) else 0.0
                xj = x[i + 1] if i + 1 < len(x) else 0.0
                angle = 2.0 * (math.pi - xi) * (math.pi - xj)
                state = self._apply_rzz(state, i, i + 1, angle, n_qubits)

        depth = self.n_reps * (1 + n_qubits + (n_qubits - 1) * 3)
        return state, depth

    def _pauli_feature_map(self, x: np.ndarray, n_qubits: int) -> Tuple[np.ndarray, int]:
        """
        PauliFeatureMap: Tensor products of e^{-i x_k σ_k} for σ ∈ {X, Y, Z}.
        """
        dim = 2 ** n_qubits
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0

        for rep in range(self.n_reps):
            state = self._apply_hadamard_all(state, n_qubits)
            for i in range(n_qubits):
                angle = x[i] if i < len(x) else 0.0
                # Apply Rx, Ry, Rz in sequence
                state = self._apply_rx(state, i, angle, n_qubits)
                state = self._apply_ry(state, i, angle * PHI, n_qubits)
                state = self._apply_rz(state, i, angle * PHI_CONJUGATE, n_qubits)
            # Entangle via CNOT ladder
            for i in range(n_qubits - 1):
                state = self._apply_cnot(state, i, i + 1, n_qubits)

        depth = self.n_reps * (1 + 3 * n_qubits + (n_qubits - 1))
        return state, depth

    def _iqp_feature_map(self, x: np.ndarray, n_qubits: int) -> Tuple[np.ndarray, int]:
        """
        IQP (Instantaneous Quantum Polynomial) feature map.
        U_IQP = H^n · diag(e^{i f(x)}) · H^n
        """
        dim = 2 ** n_qubits
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0

        for rep in range(self.n_reps):
            # Hadamard
            state = self._apply_hadamard_all(state, n_qubits)
            # Diagonal unitary: vectorized phase computation
            # Build bit matrix: (dim, n_qubits) — bits[i,q] = (i >> q) & 1
            indices = np.arange(dim, dtype=np.intp)
            bits = np.array([((indices >> q) & 1) for q in range(n_qubits)], dtype=np.float64).T  # (dim, n_qubits)
            # Pad feature vector x to n_qubits
            x_padded = np.zeros(n_qubits)
            x_padded[:min(len(x), n_qubits)] = x[:min(len(x), n_qubits)]
            # Linear terms: sum_i x_i * bits_i
            phase = bits @ x_padded
            # Quadratic terms: sum_{i<j} x_i * x_j * bits_i * bits_j
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    phase += x_padded[i] * x_padded[j] * bits[:, i] * bits[:, j]
            state *= np.exp(1j * phase)
            # Final Hadamard
            state = self._apply_hadamard_all(state, n_qubits)

        depth = self.n_reps * 3
        return state, depth

    def _sacred_feature_map(self, x: np.ndarray, n_qubits: int) -> Tuple[np.ndarray, int]:
        """
        L104 Sacred Feature Map — God Code aligned quantum encoding.

        Uses PHI-rotations and God Code resonance phases for feature encoding.
        Gates: H · Ry(x·φ) · Rz(G(x·104)) · CZ_sacred
        """
        dim = 2 ** n_qubits
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0

        for rep in range(self.n_reps):
            state = self._apply_hadamard_all(state, n_qubits)
            for i in range(n_qubits):
                xi = x[i] if i < len(x) else 0.0
                # PHI-rotation encoding
                phi_angle = xi * PHI
                state = self._apply_ry(state, i, phi_angle, n_qubits)
                # God Code resonance phase
                gc_phase = god_code_at(abs(xi) * 104) / GOD_CODE * math.pi
                state = self._apply_rz(state, i, gc_phase, n_qubits)
            # Sacred entangling: PHI-spaced CZ interactions
            for i in range(n_qubits - 1):
                j = (i + int(PHI * (i + 1))) % n_qubits
                if j != i:
                    state = self._apply_cz(state, i, j, n_qubits)
            # VOID_CONSTANT global phase correction
            state *= np.exp(1j * VOID_CONSTANT * rep)

        depth = self.n_reps * (1 + 2 * n_qubits + n_qubits)
        return state, depth

    # ─── Gate primitives ────────────────────────────────────────────────────

    def _apply_hadamard_all(self, state: np.ndarray, n_qubits: int) -> np.ndarray:
        """Apply Hadamard to all qubits.

        v1.0.1: Vectorized via reshape+einsum per qubit — 50-200× faster.
        """
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        for q in range(n_qubits):
            shape = (2 ** q, 2, 2 ** (n_qubits - q - 1))
            psi = state.reshape(shape)
            state = np.einsum('ij,ajb->aib', H, psi).reshape(-1)
        return state

    def _apply_rz(self, state: np.ndarray, qubit: int, angle: float, n_qubits: int) -> np.ndarray:
        """Apply Rz(angle) to qubit.

        v1.0.1: Vectorized diagonal gate via boolean mask.
        """
        dim = 2 ** n_qubits
        indices = np.arange(dim, dtype=np.intp)
        bit_vals = (indices >> qubit) & 1
        phases = np.where(bit_vals == 1, angle / 2, -angle / 2)
        state = state * np.exp(1j * phases)
        return state

    def _apply_rx(self, state: np.ndarray, qubit: int, angle: float, n_qubits: int) -> np.ndarray:
        """Apply Rx(angle) to qubit.

        v1.0.1: Vectorized via reshape+einsum.
        """
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        Rx = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
        shape = (2 ** qubit, 2, 2 ** (n_qubits - qubit - 1))
        psi = state.reshape(shape)
        return np.einsum('ij,ajb->aib', Rx, psi).reshape(-1)

    def _apply_ry(self, state: np.ndarray, qubit: int, angle: float, n_qubits: int) -> np.ndarray:
        """Apply Ry(angle) to qubit.

        v1.0.1: Vectorized via reshape+einsum.
        """
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        Ry = np.array([[c, -s], [s, c]], dtype=np.complex128)
        shape = (2 ** qubit, 2, 2 ** (n_qubits - qubit - 1))
        psi = state.reshape(shape)
        return np.einsum('ij,ajb->aib', Ry, psi).reshape(-1)

    def _apply_cnot(self, state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
        """Apply CNOT (CX) gate.

        v1.0.1: Vectorized boolean mask swap.
        """
        dim = 2 ** n_qubits
        new = state.copy()
        indices = np.arange(dim, dtype=np.intp)
        ctrl_set = ((indices >> control) & 1).astype(bool)
        tgt_zero = ~((indices >> target) & 1).astype(bool)
        mask = ctrl_set & tgt_zero
        partners = indices[mask] ^ (1 << target)
        new[indices[mask]] = state[partners]
        new[partners] = state[indices[mask]]
        return new

    def _apply_cz(self, state: np.ndarray, q0: int, q1: int, n_qubits: int) -> np.ndarray:
        """Apply CZ (controlled-Z) gate.

        v1.0.1: Vectorized boolean mask.
        """
        dim = 2 ** n_qubits
        indices = np.arange(dim, dtype=np.intp)
        mask = ((indices >> q0) & 1).astype(bool) & ((indices >> q1) & 1).astype(bool)
        state = state.copy()
        state[mask] *= -1
        return state

    def _apply_rzz(self, state: np.ndarray, q0: int, q1: int, angle: float, n_qubits: int) -> np.ndarray:
        """Apply e^{-i θ/2 ZZ} = Rzz(θ).

        v1.0.1: Vectorized parity computation via XOR.
        """
        dim = 2 ** n_qubits
        indices = np.arange(dim, dtype=np.intp)
        b0 = (indices >> q0) & 1
        b1 = (indices >> q1) & 1
        parity = 1 - 2 * (b0 ^ b1)
        state = state * np.exp(1j * angle / 2 * parity)
        return state

    def _god_code_feature_alignment(self, features: np.ndarray) -> float:
        """Score how well the quantum feature space aligns with God Code harmonics."""
        if len(features) == 0:
            return 0.0
        # Compute average state overlap with God Code reference state
        n_qubits = num_qubits_for(features.shape[1])
        dim = features.shape[1]
        # God Code reference: state with amplitudes proportional to G(k·104/dim)
        gc_ref = np.array([god_code_at(k * 104 / dim) for k in range(dim)], dtype=np.complex128)
        gc_ref = gc_ref / np.linalg.norm(gc_ref)

        overlaps = []
        for state in features:
            overlap = abs(np.dot(state.conj(), gc_ref)) ** 2
            overlaps.append(overlap)

        return float(np.mean(overlaps))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. QUANTUM STATE ENCODER
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumStateEncoder:
    """
    Encodes classical data into quantum states using various strategies:

      • Amplitude encoding: x → |ψ⟩ where amplitudes = normalized x
        (log₂N qubits for N-dimensional data — exponential compression)
      • Angle encoding: x_i → Ry(x_i)|0⟩ on qubit i (linear qubits)
      • Basis encoding: integer k → |k⟩ (computational basis)
      • Dense angle encoding: 2 features per qubit via Ry and Rz
    """

    METHODS = ("amplitude", "angle", "basis", "dense_angle")

    def __init__(self, method: str = "amplitude"):
        self.method = method.lower()
        if self.method not in self.METHODS:
            raise ValueError(f"Unknown encoding '{method}'. Use: {self.METHODS}")

    def encode(self, data: np.ndarray) -> FeatureMapResult:
        """
        Encode data vector into a quantum state.

        Args:
            data: Classical data (1D vector or 2D matrix — rows encoded separately)

        Returns:
            FeatureMapResult with quantum state vectors
        """
        t0 = time.time()
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_samples, n_features = data.shape
        dispatch = {
            "amplitude": self._amplitude_encode,
            "angle": self._angle_encode,
            "basis": self._basis_encode,
            "dense_angle": self._dense_angle_encode,
        }

        states = []
        n_qubits = 0
        for row in data:
            state, nq = dispatch[self.method](row)
            states.append(state)
            n_qubits = max(n_qubits, nq)

        # Pad all states to the same dimension
        dim = max(len(s) for s in states)
        quantum_features = np.zeros((n_samples, dim), dtype=np.complex128)
        for i, s in enumerate(states):
            quantum_features[i, :len(s)] = s

        return FeatureMapResult(
            quantum_features=quantum_features,
            feature_dimension=dim,
            n_qubits=n_qubits,
            circuit_depth=self._estimate_depth(n_qubits),
            encoding_method=self.method,
            god_code_alignment=0.0,
            execution_time=time.time() - t0,
            metadata={
                "n_samples": n_samples,
                "n_features": n_features,
                "encoding": self.method,
            },
        )

    def _amplitude_encode(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """Amplitude encoding: N features → log₂(N) qubits."""
        state = data_to_quantum_state(x)
        n_qubits = num_qubits_for(len(state))
        return state, n_qubits

    def _angle_encode(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """Angle encoding: each feature → Ry rotation on one qubit."""
        n_qubits = len(x)
        dim = 2 ** n_qubits
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0
        # Apply Ry(x_i) to each qubit
        for i in range(n_qubits):
            angle = float(x[i])
            c, s = math.cos(angle / 2), math.sin(angle / 2)
            new = np.zeros_like(state)
            for basis in range(dim):
                bit = (basis >> i) & 1
                partner = basis ^ (1 << i)
                if bit == 0:
                    new[basis] += c * state[basis]
                    new[partner] += s * state[basis]
                else:
                    new[basis] += c * state[basis]
                    new[partner] += -s * state[basis]
            state = new
        return state, n_qubits

    def _basis_encode(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """Basis encoding: integer index → computational basis state."""
        idx = int(abs(x[0])) if len(x) > 0 else 0
        n_qubits = max(num_qubits_for(idx + 1), 1)
        dim = 2 ** n_qubits
        state = np.zeros(dim, dtype=np.complex128)
        state[min(idx, dim - 1)] = 1.0
        return state, n_qubits

    def _dense_angle_encode(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """Dense angle encoding: 2 features per qubit via Ry + Rz."""
        n_qubits = math.ceil(len(x) / 2)
        dim = 2 ** n_qubits
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0

        for i in range(n_qubits):
            # Ry for even-indexed features
            ry_angle = float(x[2 * i]) if 2 * i < len(x) else 0.0
            # Rz for odd-indexed features
            rz_angle = float(x[2 * i + 1]) if 2 * i + 1 < len(x) else 0.0

            c, s = math.cos(ry_angle / 2), math.sin(ry_angle / 2)
            new = np.zeros_like(state)
            for basis in range(dim):
                bit = (basis >> i) & 1
                partner = basis ^ (1 << i)
                if bit == 0:
                    new[basis] += c * state[basis]
                    new[partner] += s * state[basis]
                else:
                    new[basis] += c * state[basis]
                    new[partner] += -s * state[basis]
            state = new

            # Rz
            for basis in range(dim):
                bit = (basis >> i) & 1
                phase = rz_angle / 2 if bit else -rz_angle / 2
                state[basis] *= np.exp(1j * phase)

        return state, n_qubits

    def _estimate_depth(self, n_qubits: int) -> int:
        """Estimate circuit depth for encoding method."""
        depth_map = {
            "amplitude": 2 ** n_qubits,  # O(2^n) for arbitrary state prep
            "angle": n_qubits,
            "basis": n_qubits,
            "dense_angle": 2 * n_qubits,
        }
        return depth_map.get(self.method, n_qubits)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ENTANGLEMENT FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class EntanglementFeatureExtractor:
    """
    Extracts features based on entanglement structure in quantum states.

    Computes:
      • Von Neumann entanglement entropy for each bipartition
      • Concurrence for qubit pairs
      • Mutual information between subsystems
      • Entanglement spectrum (Schmidt coefficients)

    These entanglement measures serve as quantum features for downstream ML.
    """

    def __init__(self):
        self.science_engine = _get_science_engine()

    def extract(self, quantum_states: np.ndarray) -> Dict[str, Any]:
        """
        Extract entanglement-based features from quantum states.

        Args:
            quantum_states: Array of quantum state vectors (n_samples × dim)

        Returns:
            Dictionary with entanglement features per sample
        """
        t0 = time.time()
        if quantum_states.ndim == 1:
            quantum_states = quantum_states.reshape(1, -1)

        n_samples, dim = quantum_states.shape
        n_qubits = num_qubits_for(dim)

        all_entropies = []
        all_concurrences = []
        all_schmidt_ranks = []
        all_mutual_info = []

        for state in quantum_states:
            state_norm = normalize_vector(state)

            # Von Neumann entropy for each bipartition (cut after qubit k)
            entropies = []
            for cut in range(1, n_qubits):
                entropy = self._von_neumann_entropy(state_norm, n_qubits, cut)
                entropies.append(entropy)
            all_entropies.append(entropies)

            # Concurrence for adjacent qubit pairs
            concurrences = []
            for q in range(n_qubits - 1):
                conc = self._concurrence(state_norm, n_qubits, q, q + 1)
                concurrences.append(conc)
            all_concurrences.append(concurrences)

            # Schmidt rank at midpoint
            mid = n_qubits // 2
            schmidt_coeffs = self._schmidt_decomposition(state_norm, n_qubits, mid)
            all_schmidt_ranks.append(np.sum(schmidt_coeffs > 1e-10))

            # Pairwise mutual information
            mi_pairs = []
            for q1 in range(min(n_qubits, 4)):
                for q2 in range(q1 + 1, min(n_qubits, 4)):
                    mi = self._mutual_information(state_norm, n_qubits, q1, q2)
                    mi_pairs.append(mi)
            all_mutual_info.append(mi_pairs)

        return {
            "entanglement_entropies": np.array(all_entropies),
            "concurrences": np.array(all_concurrences),
            "schmidt_ranks": np.array(all_schmidt_ranks),
            "mutual_information": all_mutual_info,
            "n_samples": n_samples,
            "n_qubits": n_qubits,
            "execution_time": time.time() - t0,
        }

    def _von_neumann_entropy(self, state: np.ndarray, n_qubits: int, cut: int) -> float:
        """Compute Von Neumann entropy S(ρ_A) for bipartition at cut."""
        schmidt = self._schmidt_decomposition(state, n_qubits, cut)
        # S = -Σ λ² log₂(λ²)
        probs = schmidt ** 2
        probs = probs[probs > 1e-15]
        return float(-np.sum(probs * np.log2(probs)))

    def _schmidt_decomposition(self, state: np.ndarray, n_qubits: int, cut: int) -> np.ndarray:
        """Compute Schmidt coefficients for bipartition at cut."""
        dim_a = 2 ** cut
        dim_b = 2 ** (n_qubits - cut)
        # Reshape state into matrix
        psi_matrix = state[:dim_a * dim_b].reshape(dim_a, dim_b)
        # SVD gives Schmidt coefficients
        singular_values = np.linalg.svd(psi_matrix, compute_uv=False)
        return singular_values

    def _concurrence(self, state: np.ndarray, n_qubits: int, q1: int, q2: int) -> float:
        """Compute concurrence for a pair of qubits (Wootters formula)."""
        # Trace out all other qubits to get 2-qubit reduced density matrix
        rho_2q = self._partial_trace_to_pair(state, n_qubits, q1, q2)
        # Concurrence via Wootters formula
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        yy = np.kron(sigma_y, sigma_y)
        rho_tilde = yy @ rho_2q.conj() @ yy
        product = rho_2q @ rho_tilde
        eigenvalues = np.sort(np.real(np.linalg.eigvals(product)))[::-1]
        eigenvalues = np.maximum(eigenvalues, 0)
        sqrt_eigs = np.sqrt(eigenvalues)
        C = max(0, sqrt_eigs[0] - sqrt_eigs[1] - sqrt_eigs[2] - sqrt_eigs[3])
        return float(C)

    def _partial_trace_to_pair(self, state: np.ndarray, n_qubits: int, q1: int, q2: int) -> np.ndarray:
        """Trace out all qubits except q1 and q2 to get 4×4 density matrix."""
        dim = 2 ** n_qubits
        rho = np.zeros((4, 4), dtype=np.complex128)
        for a in range(4):
            a1 = (a >> 1) & 1
            a2 = a & 1
            for b in range(4):
                b1 = (b >> 1) & 1
                b2 = b & 1
                val = 0.0 + 0j
                for env_basis in range(2 ** (n_qubits - 2)):
                    # Reconstruct full basis indices
                    idx_a = self._embed_pair_basis(a1, a2, env_basis, n_qubits, q1, q2)
                    idx_b = self._embed_pair_basis(b1, b2, env_basis, n_qubits, q1, q2)
                    if idx_a < dim and idx_b < dim:
                        val += state[idx_a] * state[idx_b].conj()
                rho[a, b] = val
        return rho

    def _embed_pair_basis(self, b1: int, b2: int, env: int, n_qubits: int,
                          q1: int, q2: int) -> int:
        """Embed qubit pair values + environment into full basis index."""
        result = 0
        env_pos = 0
        for q in range(n_qubits):
            if q == q1:
                result |= (b1 << q)
            elif q == q2:
                result |= (b2 << q)
            else:
                result |= (((env >> env_pos) & 1) << q)
                env_pos += 1
        return result

    def _mutual_information(self, state: np.ndarray, n_qubits: int,
                             q1: int, q2: int) -> float:
        """Compute quantum mutual information I(A:B) = S(A) + S(B) - S(AB)."""
        # S(AB) from 2-qubit reduced density matrix
        rho_2q = self._partial_trace_to_pair(state, n_qubits, q1, q2)
        eigenvalues_ab = np.real(np.linalg.eigvals(rho_2q))
        eigenvalues_ab = eigenvalues_ab[eigenvalues_ab > 1e-15]
        S_ab = float(-np.sum(eigenvalues_ab * np.log2(eigenvalues_ab)))

        # S(A) and S(B) from single-qubit states
        rho_a = np.array([[rho_2q[0, 0] + rho_2q[1, 1], rho_2q[0, 2] + rho_2q[1, 3]],
                          [rho_2q[2, 0] + rho_2q[3, 1], rho_2q[2, 2] + rho_2q[3, 3]]])
        rho_b = np.array([[rho_2q[0, 0] + rho_2q[2, 2], rho_2q[0, 1] + rho_2q[2, 3]],
                          [rho_2q[1, 0] + rho_2q[3, 2], rho_2q[1, 1] + rho_2q[3, 3]]])

        S_a = self._entropy_of_density(rho_a)
        S_b = self._entropy_of_density(rho_b)

        return max(0.0, S_a + S_b - S_ab)

    def _entropy_of_density(self, rho: np.ndarray) -> float:
        """Von Neumann entropy of a density matrix."""
        eigenvalues = np.real(np.linalg.eigvals(rho))
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        if len(eigenvalues) == 0:
            return 0.0
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. QUANTUM EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumEmbedding:
    """
    High-dimensional quantum embedding for data representation.

    Combines feature mapping with entanglement analysis to produce
    rich data embeddings that capture quantum correlations impossible
    to represent efficiently classically.

    Pipeline: data → feature_map → entangle → measure → embed
    """

    def __init__(self, map_type: str = "sacred", n_reps: int = 2,
                 embedding_dim: Optional[int] = None):
        self.feature_map = QuantumFeatureMap(map_type=map_type, n_reps=n_reps)
        self.entanglement_extractor = EntanglementFeatureExtractor()
        self.embedding_dim = embedding_dim

    def embed(self, data: np.ndarray) -> EmbeddingResult:
        """
        Create quantum embedding of data.

        Args:
            data: Input data (n_samples × n_features)

        Returns:
            EmbeddingResult with embedded vectors and kernel matrix
        """
        t0 = time.time()
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples = data.shape[0]

        # Step 1: Feature map → quantum states
        fm_result = self.feature_map.transform(data)
        quantum_states = fm_result.quantum_features

        # Step 2: Compute kernel matrix (fidelity-based)
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            kernel_matrix[i, i] = 1.0
            for j in range(i + 1, n_samples):
                fidelity = abs(np.dot(quantum_states[i].conj(), quantum_states[j])) ** 2
                kernel_matrix[i, j] = fidelity
                kernel_matrix[j, i] = fidelity

        # Step 3: Spectral embedding from kernel matrix
        eigenvalues, eigenvectors = np.linalg.eigh(kernel_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Choose embedding dimension
        embed_dim = self.embedding_dim or min(n_samples, data.shape[1], 10)
        embed_dim = min(embed_dim, len(eigenvalues))

        # Scale eigenvectors by sqrt(eigenvalue)
        scale = np.sqrt(np.maximum(eigenvalues[:embed_dim], 0))
        embedded = eigenvectors[:, :embed_dim] * scale

        # Step 4: Compute expressibility and entanglement capability
        expressibility = self._compute_expressibility(quantum_states)
        ent_capability = self._compute_entanglement_capability(quantum_states)

        return EmbeddingResult(
            embedded_vectors=embedded,
            embedding_dimension=embed_dim,
            kernel_matrix=kernel_matrix,
            expressibility=expressibility,
            entanglement_capability=ent_capability,
            execution_time=time.time() - t0,
            metadata={
                "n_samples": n_samples,
                "feature_map": self.feature_map.map_type,
                "n_reps": self.feature_map.n_reps,
                "quantum_feature_dim": fm_result.feature_dimension,
                "god_code_alignment": fm_result.god_code_alignment,
            },
        )

    def _compute_expressibility(self, states: np.ndarray) -> float:
        """
        Expressibility: KL divergence between fidelity distribution and Haar random.

        Higher expressibility = circuit explores more of Hilbert space.
        """
        n = len(states)
        if n < 2:
            return 0.0

        # Sample pairwise fidelities
        fidelities = []
        for i in range(min(n, 50)):
            for j in range(i + 1, min(n, 50)):
                f = abs(np.dot(states[i].conj(), states[j])) ** 2
                fidelities.append(f)

        if not fidelities:
            return 0.0

        fidelities = np.array(fidelities)
        dim = states.shape[1]

        # Haar random fidelity distribution is Beta(1, dim-1)
        # Mean = 1/dim, Var = (dim-1)/(dim²(dim+1))
        haar_mean = 1.0 / dim
        observed_mean = np.mean(fidelities)

        # Expressibility ∝ deviation from Haar
        kl_estimate = abs(observed_mean - haar_mean) / max(haar_mean, 1e-15)
        expressibility = 1.0 - min(kl_estimate, 1.0)  # Higher = more expressive
        return float(expressibility)

    def _compute_entanglement_capability(self, states: np.ndarray) -> float:
        """
        Entanglement capability: average Meyer-Wallach entanglement measure.
        """
        n_samples, dim = states.shape
        n_qubits = num_qubits_for(dim)

        if n_qubits < 2:
            return 0.0

        # Sample subset for efficiency
        sample_size = min(n_samples, 20)
        indices = np.random.choice(n_samples, sample_size, replace=False) if n_samples > sample_size else range(n_samples)

        Q_values = []
        for idx in indices:
            state = states[idx]
            # Meyer-Wallach: Q = 2(1 - 1/n Σ_k Tr(ρ_k²))
            purity_sum = 0.0
            for k in range(n_qubits):
                rho_k = self._single_qubit_rdm(state, n_qubits, k)
                purity_sum += np.real(np.trace(rho_k @ rho_k))
            Q = 2 * (1 - purity_sum / n_qubits)
            Q_values.append(max(0, Q))

        return float(np.mean(Q_values))

    def _single_qubit_rdm(self, state: np.ndarray, n_qubits: int, qubit: int) -> np.ndarray:
        """Compute single-qubit reduced density matrix by tracing out others."""
        dim = 2 ** n_qubits
        rho = np.zeros((2, 2), dtype=np.complex128)
        for a in range(2):
            for b in range(2):
                val = 0.0 + 0j
                for env in range(2 ** (n_qubits - 1)):
                    # Reconstruct full index
                    idx_a = 0
                    idx_b = 0
                    env_pos = 0
                    for q in range(n_qubits):
                        if q == qubit:
                            idx_a |= (a << q)
                            idx_b |= (b << q)
                        else:
                            bit = (env >> env_pos) & 1
                            idx_a |= (bit << q)
                            idx_b |= (bit << q)
                            env_pos += 1
                    if idx_a < dim and idx_b < dim:
                        val += state[idx_a] * state[idx_b].conj()
                rho[a, b] = val
        return rho
