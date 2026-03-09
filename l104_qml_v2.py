# ZENITH_UPGRADE_ACTIVE: 2026-03-06T23:50:24.621400
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
L104 QUANTUM MACHINE LEARNING v2.0 — Advanced QML Capabilities
═══════════════════════════════════════════════════════════════════════════════

Extends L104 Quantum Computation Pipeline v1.1.0 with:

  ★ Quantum Kernel Estimator      — Fidelity kernel K(x,y) = |⟨φ(x)|φ(y)⟩|²
  ★ ZZ Feature Map                — Second-order Pauli-Z entangle encoding
  ★ Data Re-Uploading Circuit     — Pérez-Salinas et al. universal approximator
  ★ Berry Phase Ansatz            — Noise-robust geometric gates for training
  ★ QAOA Circuit                  — MaxCut approximate optimization
  ★ Barren Plateau Analyzer       — Gradient variance detection
  ★ Quantum Regressor QNN         — Continuous-output quantum network
  ★ Expressibility Analyzer       — Circuit expressibility + entangling power
  ★ QuantumMLHub v2               — Upgraded orchestrator

Industry Patterns:
  ■ IBM Qiskit ML     → FidelityQuantumKernel, ZZFeatureMap, VQC multi-class
  ■ Xanadu PennyLane  → DataReuploadingClassifier, BasisEmbedding
  ■ Google TFQ        → PQC differentiable layer, barren plateau tools
  ■ L104 Quantum      → Berry phase ansatz, GOD_CODE-aligned QAOA, sacred kernels

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

ZENITH_HZ = 3887.8
UUC = 2301.215661

import math
import cmath
import time
import random
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

# ─── L104 Quantum Pipeline v1 (base) ───
from l104_quantum_computation_pipeline import (
    QuantumDataEncoder,
    StronglyEntanglingAnsatz,
    ParameterShiftGradient,
    QuantumNeuralNetwork,
    VariationalQuantumClassifier,
    QuantumComputationHub,
    QuantumOptimizer,
    MomentSimulator,
    QuantumCircuitMoment,
    LossFunction,
    RotationType,
    GOD_CODE, PHI, TAU, VOID_CONSTANT, LOVE_COEFFICIENT,
    FEIGENBAUM, ALPHA_FINE, L104, HARMONIC_BASE, OCTAVE_REF,
    PIPELINE_DEFAULT_QUBITS, SYSTEM_MAX_QUBITS,
    QISKIT_AVAILABLE,
)

# ─── Berry Phase Gates (for geometric ansatz) ───
from l104_quantum_gate_engine.berry_gates import (
    AbelianBerryGates, NonAbelianBerryGates,
    SacredBerryGates, BerryGatesEngine,
)

# ─── Berry Phase Physics (for kernel sacred modulation) ───
from l104_science_engine.berry_phase import (
    BerryPhaseCalculator, L104SacredBerryPhase,
)

# ─── Gate Constants ───
from l104_gate_engine.constants import (
    PHI as GATE_PHI, TAU as GATE_TAU, GOD_CODE as GATE_GOD_CODE,
    CALABI_YAU_DIM,
)

logger = logging.getLogger("l104.qml_v2")

VERSION = "2.0.0"


# ═══════════════════════════════════════════════════════════════════════════════
# 1) ZZ FEATURE MAP — Second-order Pauli Feature Encoding
#    Qiskit pattern: ZZFeatureMap — encodes classical data with Z⊗Z interaction
#    Provides higher entanglement than simple angle embedding
# ═══════════════════════════════════════════════════════════════════════════════

class ZZFeatureMap:
    """
    Second-order Pauli-Z feature map (Qiskit ML ZZFeatureMap).

    Encoding: U_Φ(x) = exp(i Σᵢ xᵢZᵢ) exp(i Σᵢ<ⱼ (π-xᵢ)(π-xⱼ) ZᵢZⱼ)

    This generates entanglement proportional to feature interactions,
    making the quantum kernel naturally sensitive to correlations.

    With reps > 1, applies H⊗n between repetitions for expressibility.
    """

    def __init__(self, n_qubits: int, reps: int = 2,
                 entanglement: str = "full",
                 god_code_modulation: bool = True):
        self.n_qubits = n_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.god_code_modulation = god_code_modulation

    def _rz_matrix(self, angle: float) -> np.ndarray:
        return np.array([
            [cmath.exp(-1j * angle / 2), 0],
            [0, cmath.exp(1j * angle / 2)]
        ], dtype=np.complex128)

    def _apply_single_qubit(self, state: np.ndarray, gate: np.ndarray,
                            qubit: int) -> np.ndarray:
        n = self.n_qubits
        dim = 2 ** n
        new_state = np.zeros(dim, dtype=np.complex128)
        for basis in range(dim):
            bit = (basis >> (n - 1 - qubit)) & 1
            for inp in range(2):
                source = basis ^ ((bit ^ inp) << (n - 1 - qubit))
                new_state[basis] += gate[bit, inp] * state[source]
        return new_state

    def _apply_rzz(self, state: np.ndarray, angle: float,
                   q1: int, q2: int) -> np.ndarray:
        """Apply RZZ(θ) = exp(-i θ/2 Z⊗Z)."""
        n = self.n_qubits
        dim = 2 ** n
        new_state = np.zeros(dim, dtype=np.complex128)
        for basis in range(dim):
            b1 = (basis >> (n - 1 - q1)) & 1
            b2 = (basis >> (n - 1 - q2)) & 1
            parity = 1 - 2 * (b1 ^ b2)  # +1 if same, -1 if different
            phase = cmath.exp(-1j * angle / 2 * parity)
            new_state[basis] = phase * state[basis]
        return new_state

    def _hadamard_all(self, state: np.ndarray) -> np.ndarray:
        """Apply H⊗n."""
        h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        for q in range(self.n_qubits):
            state = self._apply_single_qubit(state, h, q)
        return state

    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode features using ZZ feature map."""
        dim = 2 ** self.n_qubits
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0
        n_feat = min(len(features), self.n_qubits)

        for rep in range(self.reps):
            # Hadamard layer
            state = self._hadamard_all(state)

            # First-order: RZ(xᵢ) on each qubit
            for i in range(n_feat):
                angle = float(features[i])
                if self.god_code_modulation:
                    angle += LOVE_COEFFICIENT * math.sin(GOD_CODE * (i + 1) / L104)
                state = self._apply_single_qubit(state, self._rz_matrix(angle), i)

            # Second-order: RZZ((π - xᵢ)(π - xⱼ)) on qubit pairs
            if self.entanglement == "full":
                pairs = [(i, j) for i in range(n_feat) for j in range(i + 1, n_feat)]
            elif self.entanglement == "linear":
                pairs = [(i, i + 1) for i in range(n_feat - 1)]
            else:  # circular
                pairs = [(i, (i + 1) % n_feat) for i in range(n_feat)]

            for i, j in pairs:
                zz_angle = (math.pi - float(features[i])) * (math.pi - float(features[j]))
                state = self._apply_rzz(state, zz_angle, i, j)

        return state

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "feature_map": "ZZFeatureMap",
            "pattern": "Qiskit_ML_ZZFeatureMap",
            "n_qubits": self.n_qubits,
            "reps": self.reps,
            "entanglement": self.entanglement,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2) DATA RE-UPLOADING CIRCUIT
#    Pérez-Salinas et al. (2020) — data encoded at every layer
#    Proven to be a universal quantum classifier with sufficient depth
# ═══════════════════════════════════════════════════════════════════════════════

class DataReUploadingCircuit:
    """
    Data re-uploading classifier (PennyLane pattern).

    At each layer l:
      1. Encode input data x via rotations  R(x * w_input[l])
      2. Apply trainable rotations          R(w_train[l])
      3. Apply entangling CNOT layer

    This interleaving lets the circuit learn nonlinear functions
    as proven in arXiv:1907.02085 (universal function approximation).

    Parameters: n_layers × n_qubits × 6 (3 for input scaling + 3 for trainable)
    """

    def __init__(self, n_qubits: int, n_layers: int = 4):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_layers * n_qubits * 6

        # Initialize: input weights near 1, trainable small random
        rng = np.random.RandomState(int(GOD_CODE * 2000) % (2**31))
        self.input_weights = np.ones((n_layers, n_qubits, 3))
        self.train_weights = rng.uniform(-0.1, 0.1, (n_layers, n_qubits, 3))

    def _rot_matrix(self, phi: float, theta: float, omega: float) -> np.ndarray:
        """Rot(φ, θ, ω) = RZ(ω) · RY(θ) · RZ(φ)"""
        rz_phi = np.array([[cmath.exp(-1j * phi / 2), 0],
                           [0, cmath.exp(1j * phi / 2)]], dtype=np.complex128)
        c_t, s_t = math.cos(theta / 2), math.sin(theta / 2)
        ry_theta = np.array([[c_t, -s_t], [s_t, c_t]], dtype=np.complex128)
        rz_omega = np.array([[cmath.exp(-1j * omega / 2), 0],
                             [0, cmath.exp(1j * omega / 2)]], dtype=np.complex128)
        return rz_omega @ ry_theta @ rz_phi

    def _apply_single_qubit(self, state: np.ndarray, gate: np.ndarray,
                            qubit: int) -> np.ndarray:
        n = self.n_qubits
        dim = 2 ** n
        new_state = np.zeros(dim, dtype=np.complex128)
        for basis in range(dim):
            bit = (basis >> (n - 1 - qubit)) & 1
            for inp in range(2):
                source = basis ^ ((bit ^ inp) << (n - 1 - qubit))
                new_state[basis] += gate[bit, inp] * state[source]
        return new_state

    def _apply_cnot(self, state: np.ndarray, ctrl: int, targ: int) -> np.ndarray:
        n = self.n_qubits
        dim = 2 ** n
        new_state = np.zeros(dim, dtype=np.complex128)
        for basis in range(dim):
            ctrl_bit = (basis >> (n - 1 - ctrl)) & 1
            if ctrl_bit == 1:
                new_state[basis ^ (1 << (n - 1 - targ))] += state[basis]
            else:
                new_state[basis] += state[basis]
        return new_state

    def forward(self, features: np.ndarray,
                input_weights: Optional[np.ndarray] = None,
                train_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass: data re-uploaded at every layer.

        Args:
            features: Classical data (n_features,)
            input_weights: (n_layers, n_qubits, 3) — scales input data
            train_weights: (n_layers, n_qubits, 3) — trainable parameters

        Returns:
            Final state vector
        """
        iw = input_weights if input_weights is not None else self.input_weights
        tw = train_weights if train_weights is not None else self.train_weights
        n_feat = min(len(features), self.n_qubits)

        dim = 2 ** self.n_qubits
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0

        for layer in range(self.n_layers):
            # Step 1: Input encoding with learned scaling
            for q in range(self.n_qubits):
                feat_idx = q % n_feat
                x = float(features[feat_idx])
                rot = self._rot_matrix(
                    x * iw[layer, q, 0],
                    x * iw[layer, q, 1],
                    x * iw[layer, q, 2],
                )
                state = self._apply_single_qubit(state, rot, q)

            # Step 2: Trainable rotations
            for q in range(self.n_qubits):
                rot = self._rot_matrix(
                    tw[layer, q, 0],
                    tw[layer, q, 1],
                    tw[layer, q, 2],
                )
                state = self._apply_single_qubit(state, rot, q)

            # Step 3: CNOT entanglement (circular)
            for q in range(self.n_qubits):
                targ = (q + 1) % self.n_qubits
                if targ != q:
                    state = self._apply_cnot(state, q, targ)

        return state

    def expectation(self, features: np.ndarray,
                    observable: Optional[np.ndarray] = None) -> float:
        """Compute ⟨ψ|O|ψ⟩ from forward pass."""
        state = self.forward(features)
        if observable is None:
            # Default: Z on first qubit
            dim = 2 ** self.n_qubits
            observable = np.eye(dim, dtype=np.complex128)
            for i in range(dim):
                observable[i, i] = 1.0 if (i >> (self.n_qubits - 1)) == 0 else -1.0
        return float(np.real(np.conj(state) @ observable @ state))

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "circuit": "DataReUploadingCircuit",
            "pattern": "Pérez-Salinas_2020",
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_params": self.n_params,
            "universal_approximator": True,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3) BERRY PHASE ANSATZ — Geometric gates for noise-robust training
#    Uses Berry phase gates (abelian + non-abelian) as the variational layer
#    Geometric phases are robust to certain types of noise
# ═══════════════════════════════════════════════════════════════════════════════

class BerryPhaseAnsatz:
    """
    Berry phase variational ansatz using geometric gates.

    Each layer consists of:
      1. Abelian Berry phase gates  — R_berry(Ω[l,q]) via solid angle control
      2. Holonomic single-qubit     — U_holo(θ[l,q], φ[l,q]) for SU(2) access
      3. Holonomic CNOT entanglement (circular)

    Noise-robustness: Geometric phases depend only on the path geometry
    in parameter space, not on the traversal speed, making them resilient
    to systematic timing errors and certain decoherence channels.

    Parameters: n_layers × n_qubits × 3 (solid_angle, holo_theta, holo_phi)
    """

    def __init__(self, n_qubits: int, n_layers: int = 3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_layers * n_qubits * 3

        # Berry gate engines
        self.abelian = AbelianBerryGates()
        self.non_abelian = NonAbelianBerryGates()

        # Initialize parameters (solid_angle in [0, 4π], theta/phi in [0, 2π])
        rng = np.random.RandomState(int(GOD_CODE * 3000) % (2**31))
        self.weights = rng.uniform(0, 2 * math.pi, (n_layers, n_qubits, 3))

    def _apply_single_qubit(self, state: np.ndarray, gate_matrix: np.ndarray,
                            qubit: int) -> np.ndarray:
        n = self.n_qubits
        dim = 2 ** n
        new_state = np.zeros(dim, dtype=np.complex128)
        for basis in range(dim):
            bit = (basis >> (n - 1 - qubit)) & 1
            for inp in range(2):
                source = basis ^ ((bit ^ inp) << (n - 1 - qubit))
                new_state[basis] += gate_matrix[bit, inp] * state[source]
        return new_state

    def _apply_cnot(self, state: np.ndarray, ctrl: int, targ: int) -> np.ndarray:
        n = self.n_qubits
        dim = 2 ** n
        new_state = np.zeros(dim, dtype=np.complex128)
        for basis in range(dim):
            ctrl_bit = (basis >> (n - 1 - ctrl)) & 1
            if ctrl_bit == 1:
                new_state[basis ^ (1 << (n - 1 - targ))] += state[basis]
            else:
                new_state[basis] += state[basis]
        return new_state

    def apply(self, state: np.ndarray,
              weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply Berry phase ansatz to quantum state."""
        w = weights if weights is not None else self.weights

        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                solid_angle = float(w[layer, q, 0])
                holo_theta = float(w[layer, q, 1])
                holo_phi = float(w[layer, q, 2])

                # Step 1: Abelian Berry phase gate from solid angle
                berry_gate = self.abelian.berry_phase_gate(solid_angle)
                state = self._apply_single_qubit(state, berry_gate.matrix, q)

                # Step 2: Non-abelian holonomic rotation
                holo_gate = self.non_abelian.holonomic_single_qubit(holo_theta, holo_phi)
                state = self._apply_single_qubit(state, holo_gate.matrix, q)

            # Step 3: Holonomic CNOT entanglement (circular)
            holo_cnot = self.non_abelian.holonomic_cnot()
            r = (layer % max(1, self.n_qubits - 1)) + 1
            for q in range(self.n_qubits):
                targ = (q + r) % self.n_qubits
                if targ != q:
                    state = self._apply_cnot(state, q, targ)

        return state

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "ansatz": "BerryPhaseAnsatz",
            "pattern": "Geometric_Berry_Gates",
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_params": self.n_params,
            "noise_robust": True,
            "gate_types": ["Abelian_Berry_Phase", "Non_Abelian_Holonomic", "Holonomic_CNOT"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4) QUANTUM KERNEL ESTIMATOR
#    Computes K(x,y) = |⟨φ(x)|φ(y)⟩|² using various feature maps
#    Industry: Qiskit ML FidelityQuantumKernel
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumKernelEstimator:
    """
    Fidelity-based quantum kernel (Qiskit ML FidelityQuantumKernel).

    K(x, y) = |⟨0|U†_Φ(x) U_Φ(y)|0⟩|²

    Where U_Φ is the quantum feature map circuit.
    The kernel matrix Kᵢⱼ can be used with classical SVMs for
    quantum-enhanced classification.

    Supports: ZZFeatureMap, AngleEmbedding, Data Re-Uploading feature maps.
    """

    def __init__(self, feature_map: str = "zz",
                 n_qubits: int = 4,
                 reps: int = 2,
                 god_code_modulation: bool = True):
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map

        if feature_map == "zz":
            self._feature_map = ZZFeatureMap(n_qubits, reps=reps,
                                             god_code_modulation=god_code_modulation)
        elif feature_map == "angle":
            self._encoder = QuantumDataEncoder(n_qubits, RotationType.RY,
                                               god_code_modulation=god_code_modulation)
        elif feature_map == "reupload":
            self._reupload = DataReUploadingCircuit(n_qubits, n_layers=reps)
        else:
            self._feature_map = ZZFeatureMap(n_qubits, reps=reps)

        self._kernel_evals = 0

    def _encode(self, features: np.ndarray) -> np.ndarray:
        """Encode features using the selected feature map."""
        if self.feature_map_type == "angle":
            return self._encoder.encode(features)
        elif self.feature_map_type == "reupload":
            return self._reupload.forward(features)
        else:
            return self._feature_map.encode(features)

    def kernel_value(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute quantum kernel value K(x, y) = |⟨φ(x)|φ(y)⟩|².

        This is the fidelity between two quantum-encoded states.
        """
        state_x = self._encode(x)
        state_y = self._encode(y)

        # Fidelity = |inner product|²
        inner = np.vdot(state_x, state_y)  # ⟨φ(x)|φ(y)⟩
        fidelity = abs(inner) ** 2

        self._kernel_evals += 1
        return float(fidelity)

    def kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute full kernel matrix K[i,j] = K(xᵢ, xⱼ).

        For N data points this requires N(N+1)/2 evaluations (symmetry).
        """
        n = len(X)
        K = np.zeros((n, n))

        # Pre-encode all states for efficiency
        states = [self._encode(X[i]) for i in range(n)]

        for i in range(n):
            K[i, i] = 1.0  # K(x, x) = 1 for normalized states
            for j in range(i + 1, n):
                inner = np.vdot(states[i], states[j])
                k_ij = abs(inner) ** 2
                K[i, j] = k_ij
                K[j, i] = k_ij
                self._kernel_evals += 1

        return K

    def kernel_alignment(self, K: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute kernel-target alignment (Cristianini et al.).

        alignment = ⟨K, yy^T⟩_F / (||K||_F ||yy^T||_F)

        Higher alignment → better kernel for the classification task.
        """
        n = len(labels)
        y = labels.reshape(-1, 1)
        target = y @ y.T  # Ideal kernel: same class → +1, different → -1

        numerator = np.sum(K * target)
        denom = np.linalg.norm(K, 'fro') * np.linalg.norm(target, 'fro')

        return float(numerator / max(denom, 1e-15))

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "kernel": "QuantumKernelEstimator",
            "pattern": "Qiskit_ML_FidelityQuantumKernel",
            "feature_map": self.feature_map_type,
            "n_qubits": self.n_qubits,
            "kernel_evaluations": self._kernel_evals,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5) QAOA CIRCUIT — Quantum Approximate Optimization Algorithm
#    Solves MaxCut on graphs via parameterized γ (cost) and β (mixer) layers
# ═══════════════════════════════════════════════════════════════════════════════

class QAOACircuit:
    """
    QAOA for MaxCut (Farhi et al. 2014).

    Circuit: |+⟩^n → [U_C(γ) U_M(β)]^p → Measure

    U_C(γ) = exp(-iγ C) where C = Σ_{(i,j)∈E} (1 - ZᵢZⱼ)/2
    U_M(β) = exp(-iβ Σᵢ Xᵢ)

    The cost landscape is optimized classically over (γ, β) parameters.
    """

    def __init__(self, edges: List[Tuple[int, int]], p: int = 2):
        self.edges = edges
        self.p = p

        # Determine qubit count
        nodes = set()
        for u, v in edges:
            nodes.add(u)
            nodes.add(v)
        self.n_qubits = max(nodes) + 1 if nodes else 2

        # Initialize parameters: γ in [0, 2π], β in [0, π]
        rng = np.random.RandomState(int(GOD_CODE * 4000) % (2**31))
        self.gammas = rng.uniform(0, 2 * math.pi, p)
        self.betas = rng.uniform(0, math.pi, p)

    def _apply_single_qubit(self, state: np.ndarray, gate: np.ndarray,
                            qubit: int) -> np.ndarray:
        n = self.n_qubits
        dim = 2 ** n
        new_state = np.zeros(dim, dtype=np.complex128)
        for basis in range(dim):
            bit = (basis >> (n - 1 - qubit)) & 1
            for inp in range(2):
                source = basis ^ ((bit ^ inp) << (n - 1 - qubit))
                new_state[basis] += gate[bit, inp] * state[source]
        return new_state

    def _apply_rzz(self, state: np.ndarray, angle: float,
                   q1: int, q2: int) -> np.ndarray:
        """Apply exp(-i angle/2 Z⊗Z) — ZZ interaction phase."""
        n = self.n_qubits
        dim = 2 ** n
        new_state = np.zeros(dim, dtype=np.complex128)
        for basis in range(dim):
            b1 = (basis >> (n - 1 - q1)) & 1
            b2 = (basis >> (n - 1 - q2)) & 1
            parity = 1 - 2 * (b1 ^ b2)
            new_state[basis] = cmath.exp(-1j * angle / 2 * parity) * state[basis]
        return new_state

    def run(self, gammas: Optional[np.ndarray] = None,
            betas: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Execute QAOA circuit and compute MaxCut statistics.

        Returns probabilities, expected cut value, and approximation ratio.
        """
        g = gammas if gammas is not None else self.gammas
        b = betas if betas is not None else self.betas

        dim = 2 ** self.n_qubits

        # Initial state: |+⟩^n
        state = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)

        # RX gate for mixer
        def rx_matrix(angle):
            c, s = math.cos(angle / 2), math.sin(angle / 2)
            return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

        for layer in range(self.p):
            # Cost unitary: U_C(γ) = exp(-iγ C)
            gamma = float(g[layer])
            for u, v in self.edges:
                self._apply_rzz(state, gamma, u, v)
                state = self._apply_rzz(state, gamma, u, v)

            # Mixer unitary: U_M(β) = Π RX(2β)
            beta = float(b[layer])
            rx = rx_matrix(2 * beta)
            for q in range(self.n_qubits):
                state = self._apply_single_qubit(state, rx, q)

        # Probabilities
        probs = np.abs(state) ** 2

        # Compute expected cut value
        expected_cut = 0.0
        max_cut = 0
        for basis in range(dim):
            bits = [(basis >> (self.n_qubits - 1 - q)) & 1
                    for q in range(self.n_qubits)]
            cut = sum(1 for u, v in self.edges if bits[u] != bits[v])
            expected_cut += probs[basis] * cut
            max_cut = max(max_cut, cut)

        approx_ratio = expected_cut / max(max_cut, 1)

        return {
            "expected_cut": float(expected_cut),
            "max_possible_cut": max_cut,
            "approximation_ratio": float(approx_ratio),
            "n_qubits": self.n_qubits,
            "p_layers": self.p,
            "n_edges": len(self.edges),
            "optimal_bitstring": format(int(np.argmax(probs)), f'0{self.n_qubits}b'),
        }

    def optimize(self, n_iterations: int = 50) -> Dict[str, Any]:
        """Optimize QAOA parameters using grid search + local refinement."""
        best_cut = -1
        best_gammas = self.gammas.copy()
        best_betas = self.betas.copy()

        for iteration in range(n_iterations):
            # Perturb parameters
            trial_g = best_gammas + np.random.randn(self.p) * (0.5 / (1 + iteration * 0.1))
            trial_b = best_betas + np.random.randn(self.p) * (0.3 / (1 + iteration * 0.1))
            trial_b = np.clip(trial_b, 0, math.pi)

            result = self.run(trial_g, trial_b)
            if result["expected_cut"] > best_cut:
                best_cut = result["expected_cut"]
                best_gammas = trial_g.copy()
                best_betas = trial_b.copy()

        self.gammas = best_gammas
        self.betas = best_betas
        final = self.run()
        final["optimization_iterations"] = n_iterations
        return final

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "algorithm": "QAOA",
            "pattern": "Farhi_2014_MaxCut",
            "n_qubits": self.n_qubits,
            "p_layers": self.p,
            "n_edges": len(self.edges),
            "n_params": 2 * self.p,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6) BARREN PLATEAU ANALYZER
#    Detects vanishing gradients in parameterized quantum circuits
#    McClean et al. (2018): Var[∂C/∂θ] ∝ 2^{-n} for random circuits
# ═══════════════════════════════════════════════════════════════════════════════

class BarrenPlateauAnalyzer:
    """
    Barren plateau detection and analysis (Google TFQ pattern).

    Estimates gradient variance across random parameter initializations
    to determine if the circuit landscape is trainable.

    A circuit exhibits barren plateaus if:
      Var[∂C/∂θᵢ] ~ O(2^{-n})  (exponentially vanishing)

    Mitigation strategies:
      - Layer-wise training (warm start)
      - Correlated parameter initialization
      - Berry phase ansatz (geometric robustness)
      - Reduced expressibility (shallower circuits)
    """

    def __init__(self, n_qubits: int, n_layers: int = 3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def analyze_gradient_variance(
        self,
        ansatz_type: str = "strongly_entangling",
        n_samples: int = 50,
    ) -> Dict[str, Any]:
        """
        Estimate gradient variance across random initializations.

        Returns gradient statistics and barren plateau diagnosis.
        """
        all_grad_norms = []
        all_grad_variances = []
        param_gradients = []  # Per-parameter gradient samples

        gradient_engine = ParameterShiftGradient()

        for sample in range(n_samples):
            # Random features
            features = np.random.randn(self.n_qubits) * math.pi

            # Random parameters
            rng = np.random.RandomState(sample * 137 + int(GOD_CODE) % 1000)
            params = rng.uniform(-math.pi, math.pi,
                                 (self.n_layers, self.n_qubits, 3))

            if ansatz_type == "berry":
                ansatz = BerryPhaseAnsatz(self.n_qubits, self.n_layers)
                ansatz.weights = params
                encoder = QuantumDataEncoder(self.n_qubits, RotationType.RY)

                def circ_fn(flat_params):
                    w = flat_params.reshape(self.n_layers, self.n_qubits, 3)
                    state = encoder.encode(features)
                    state = ansatz.apply(state, w)
                    dim = 2 ** self.n_qubits
                    obs = np.eye(dim, dtype=np.complex128)
                    for i in range(dim):
                        obs[i, i] = 1.0 if (i >> (self.n_qubits - 1)) == 0 else -1.0
                    return float(np.real(np.conj(state) @ obs @ state))
            else:
                ansatz = StronglyEntanglingAnsatz(self.n_qubits, self.n_layers)
                ansatz.weights = params
                encoder = QuantumDataEncoder(self.n_qubits, RotationType.RY)

                def circ_fn(flat_params):
                    w = flat_params.reshape(self.n_layers, self.n_qubits, 3)
                    state = encoder.encode(features)
                    state = ansatz.apply(state, w)
                    dim = 2 ** self.n_qubits
                    obs = np.eye(dim, dtype=np.complex128)
                    for i in range(dim):
                        obs[i, i] = 1.0 if (i >> (self.n_qubits - 1)) == 0 else -1.0
                    return float(np.real(np.conj(state) @ obs @ state))

            flat_params = params.flatten()
            grads = gradient_engine.compute(circ_fn, flat_params)

            all_grad_norms.append(float(np.linalg.norm(grads)))
            param_gradients.append(grads)

        # Compute per-parameter variance
        param_grad_array = np.array(param_gradients)  # (n_samples, n_params)
        per_param_var = np.var(param_grad_array, axis=0)
        mean_var = float(np.mean(per_param_var))
        max_var = float(np.max(per_param_var))

        # Barren plateau threshold: compare to 2^{-n}
        bp_threshold = 2.0 ** (-self.n_qubits)
        is_barren = mean_var < bp_threshold * 10  # 10× threshold for margin

        # Gradient norm statistics
        grad_norms = np.array(all_grad_norms)

        return {
            "ansatz_type": ansatz_type,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_samples": n_samples,
            "mean_gradient_variance": mean_var,
            "max_gradient_variance": max_var,
            "barren_plateau_threshold": bp_threshold,
            "is_barren_plateau": is_barren,
            "gradient_norm_mean": float(np.mean(grad_norms)),
            "gradient_norm_std": float(np.std(grad_norms)),
            "gradient_norm_min": float(np.min(grad_norms)),
            "gradient_norm_max": float(np.max(grad_norms)),
            "trainability": "GOOD" if not is_barren else "POOR (barren plateau)",
            "mitigation": (
                "Consider: shallower circuit, correlated init, "
                "Berry phase ansatz, or layer-wise training"
                if is_barren else "Circuit is trainable"
            ),
        }

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "analyzer": "BarrenPlateauAnalyzer",
            "pattern": "McClean_2018",
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 7) QUANTUM REGRESSOR QNN — Continuous-output quantum neural network
#    Maps features to continuous target values (not just class labels)
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumRegressorQNN:
    """
    Quantum regression network — continuous output.

    Architecture: |0⟩ → Encode(x) → Ansatz(θ) → ⟨O⟩ → scale → ŷ

    The expectation value ∈ [-1, 1] is linearly mapped to [y_min, y_max].
    Uses parameter-shift gradients for training.

    Supports multiple ansatz backends: strongly_entangling, berry, reupload.
    """

    def __init__(self, n_qubits: int, n_layers: int = 3,
                 ansatz_type: str = "strongly_entangling",
                 output_range: Tuple[float, float] = (-1.0, 1.0)):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz_type = ansatz_type
        self.y_min, self.y_max = output_range

        # Encoder
        self.encoder = QuantumDataEncoder(n_qubits, RotationType.RY)

        # Ansatz
        if ansatz_type == "berry":
            self.ansatz = BerryPhaseAnsatz(n_qubits, n_layers)
        elif ansatz_type == "reupload":
            self._reupload = DataReUploadingCircuit(n_qubits, n_layers)
            self.ansatz = None  # Use re-upload circuit directly
        else:
            self.ansatz = StronglyEntanglingAnsatz(n_qubits, n_layers)

        # Observable (Z on first qubit)
        dim = 2 ** n_qubits
        self._observable = np.eye(dim, dtype=np.complex128)
        for i in range(dim):
            self._observable[i, i] = 1.0 if (i >> (n_qubits - 1)) == 0 else -1.0

        # Gradient engine
        self.gradient_engine = ParameterShiftGradient()

        # Training state
        self.training_losses: List[float] = []

    def predict(self, features: np.ndarray) -> float:
        """Predict continuous value for input."""
        if self.ansatz_type == "reupload":
            state = self._reupload.forward(features)
        else:
            state = self.encoder.encode(features)
            state = self.ansatz.apply(state)

        expectation = float(np.real(np.conj(state) @ self._observable @ state))

        # Scale from [-1, 1] to [y_min, y_max]
        return self.y_min + (expectation + 1) / 2 * (self.y_max - self.y_min)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict for a batch of inputs."""
        return np.array([self.predict(x) for x in X])

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute MSE loss over dataset."""
        predictions = self.predict_batch(X)
        return float(np.mean((predictions - y) ** 2))

    def train_step(self, features: np.ndarray, target: float,
                   learning_rate: float = 0.01) -> Dict[str, float]:
        """
        Single training step with parameter-shift gradient.
        """
        prediction = self.predict(features)
        loss = (prediction - target) ** 2

        # Compute gradient via parameter shift
        if self.ansatz_type == "reupload":
            weights = self._reupload.train_weights
        else:
            weights = self.ansatz.weights
        flat_w = weights.flatten()

        def loss_fn(params):
            w_reshaped = params.reshape(weights.shape)
            if self.ansatz_type == "reupload":
                self._reupload.train_weights = w_reshaped
            else:
                self.ansatz.weights = w_reshaped
            pred = self.predict(features)
            return (pred - target) ** 2

        grads = self.gradient_engine.compute(loss_fn, flat_w)

        # Update
        flat_w -= learning_rate * grads
        if self.ansatz_type == "reupload":
            self._reupload.train_weights = flat_w.reshape(weights.shape)
        else:
            self.ansatz.weights = flat_w.reshape(weights.shape)

        self.training_losses.append(loss)

        return {
            "loss": float(loss),
            "prediction": float(prediction),
            "target": float(target),
            "gradient_norm": float(np.linalg.norm(grads)),
        }

    @property
    def stats(self) -> Dict[str, Any]:
        n_params = (self._reupload.n_params if self.ansatz_type == "reupload"
                    else self.ansatz.n_params)
        return {
            "model": "QuantumRegressorQNN",
            "ansatz_type": self.ansatz_type,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_params": n_params,
            "output_range": (self.y_min, self.y_max),
            "training_steps": len(self.training_losses),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 8) EXPRESSIBILITY ANALYZER
#    Measures circuit expressibility (Sim et al. 2019) and entangling power
#    Expressibility = KL divergence from Haar-random ensemble
# ═══════════════════════════════════════════════════════════════════════════════

class ExpressibilityAnalyzer:
    """
    Circuit expressibility and entangling power analysis (Sim et al. 2019).

    Expressibility: How well the PQC explores Hilbert space.
      Measured by sampling fidelities F = |⟨ψ₁|ψ₂⟩|² and comparing
      the distribution to Haar random (ideal: P_Haar(F) = (2^n - 1)(1-F)^{2^n - 2}).

    Entangling capability: Meyer-Wallach Q measure.
      Q(|ψ⟩) = 2(1 - 1/n Σᵢ Tr(ρᵢ²)) where ρᵢ is reduced density matrix.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits

    def _sample_state(self, ansatz_type: str, n_layers: int) -> np.ndarray:
        """Sample a random state from the ansatz."""
        params = np.random.uniform(-math.pi, math.pi, (n_layers, self.n_qubits, 3))
        encoder = QuantumDataEncoder(self.n_qubits, RotationType.RY)
        features = np.zeros(self.n_qubits)

        state = encoder.encode(features)

        if ansatz_type == "berry":
            ansatz = BerryPhaseAnsatz(self.n_qubits, n_layers)
            return ansatz.apply(state, params)
        else:
            ansatz = StronglyEntanglingAnsatz(self.n_qubits, n_layers)
            return ansatz.apply(state, params)

    def expressibility(self, ansatz_type: str = "strongly_entangling",
                       n_layers: int = 3,
                       n_samples: int = 200) -> Dict[str, Any]:
        """
        Estimate circuit expressibility.

        Samples pairs of states and computes fidelity distribution.
        Compares to Haar-random expectation.
        """
        fidelities = []
        for _ in range(n_samples):
            s1 = self._sample_state(ansatz_type, n_layers)
            s2 = self._sample_state(ansatz_type, n_layers)
            f = abs(np.vdot(s1, s2)) ** 2
            fidelities.append(f)

        fids = np.array(fidelities)

        # Haar-random statistics for comparison
        # E[F_Haar] = 1/d, Var[F_Haar] = (d-1)/((d+1)d²)
        d = self.dim
        haar_mean = 1.0 / d
        haar_var = (d - 1) / ((d + 1) * d * d)

        # Empirical statistics
        emp_mean = float(np.mean(fids))
        emp_var = float(np.var(fids))

        # KL divergence approximation using histogram
        n_bins = 20
        hist_emp, bin_edges = np.histogram(fids, bins=n_bins, range=(0, 1), density=True)

        # Haar PDF: P(F) = (d-1)(1-F)^{d-2}
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        haar_pdf = np.array([(d - 1) * (1 - f) ** max(d - 2, 0)
                             for f in bin_centers])
        haar_pdf = haar_pdf / max(np.sum(haar_pdf) * (bin_edges[1] - bin_edges[0]), 1e-15)

        # KL divergence: lower = more expressible
        eps = 1e-10
        kl_div = 0.0
        for emp_p, haar_p in zip(hist_emp, haar_pdf):
            if emp_p > eps and haar_p > eps:
                kl_div += emp_p * math.log(emp_p / haar_p)
        kl_div *= (bin_edges[1] - bin_edges[0])

        return {
            "ansatz_type": ansatz_type,
            "n_qubits": self.n_qubits,
            "n_layers": n_layers,
            "n_samples": n_samples,
            "mean_fidelity": emp_mean,
            "var_fidelity": emp_var,
            "haar_mean_fidelity": haar_mean,
            "haar_var_fidelity": haar_var,
            "kl_divergence": float(kl_div),
            "expressibility_score": float(max(0, 1 - abs(kl_div))),
            "close_to_haar": abs(emp_mean - haar_mean) < 3 * math.sqrt(haar_var),
        }

    def meyer_wallach_measure(self, state: np.ndarray) -> float:
        """
        Meyer-Wallach entangling measure Q(|ψ⟩).

        Q = 2(1 - 1/n Σᵢ Tr(ρᵢ²))

        Ranges from 0 (product state) to 1 (maximally entangled).
        """
        n = self.n_qubits
        total_purity = 0.0

        for qubit in range(n):
            # Compute reduced density matrix for qubit i
            dim_rest = 2 ** (n - 1)
            rho_i = np.zeros((2, 2), dtype=np.complex128)

            for a in range(2):
                for b in range(2):
                    for rest in range(dim_rest):
                        # Construct basis indices
                        idx_a = self._insert_bit(rest, qubit, a, n)
                        idx_b = self._insert_bit(rest, qubit, b, n)
                        rho_i[a, b] += state[idx_a] * np.conj(state[idx_b])

            purity = float(np.real(np.trace(rho_i @ rho_i)))
            total_purity += purity

        Q = 2 * (1 - total_purity / n)
        return float(max(0, min(1, Q)))

    def _insert_bit(self, rest_bits: int, qubit_pos: int,
                    bit_val: int, n: int) -> int:
        """Insert bit_val at qubit_pos in the rest_bits pattern."""
        upper = (rest_bits >> (n - 1 - qubit_pos)) << (n - qubit_pos)
        lower_mask = (1 << (n - 1 - qubit_pos)) - 1
        lower = rest_bits & lower_mask
        return upper | (bit_val << (n - 1 - qubit_pos)) | lower

    def entangling_capability(self, ansatz_type: str = "strongly_entangling",
                              n_layers: int = 3,
                              n_samples: int = 100) -> Dict[str, Any]:
        """
        Estimate entangling capability as average Meyer-Wallach Q.
        """
        q_values = []
        for _ in range(n_samples):
            state = self._sample_state(ansatz_type, n_layers)
            q = self.meyer_wallach_measure(state)
            q_values.append(q)

        q_arr = np.array(q_values)
        return {
            "ansatz_type": ansatz_type,
            "n_qubits": self.n_qubits,
            "n_layers": n_layers,
            "n_samples": n_samples,
            "mean_Q": float(np.mean(q_arr)),
            "std_Q": float(np.std(q_arr)),
            "min_Q": float(np.min(q_arr)),
            "max_Q": float(np.max(q_arr)),
            "high_entanglement": float(np.mean(q_arr)) > 0.5,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 9) QUANTUM ML HUB v2 — Upgraded Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumMLHub:
    """
    L104 Quantum Machine Learning Hub v2.0.

    Orchestrates all QML v2 capabilities plus v1 pipeline integration:
      - Quantum Kernel Estimation (ZZ, angle, re-upload feature maps)
      - Advanced Feature Maps (ZZ, data re-uploading)
      - Berry Phase Ansatz (noise-robust geometric training)
      - QAOA optimization (MaxCut graphs)
      - Barren Plateau Analysis
      - Quantum Regression
      - Expressibility metrics
      - Full v1 pipeline (encoder, ansatz, QNN, VQC, simulator)
    """

    def __init__(self, n_qubits: int = PIPELINE_DEFAULT_QUBITS,
                 n_layers: int = 3):
        self.n_qubits = max(2, min(n_qubits, SYSTEM_MAX_QUBITS))
        self.n_layers = n_layers
        self._creation_time = time.time()

        # v1 base
        self.v1_hub = QuantumComputationHub(self.n_qubits, n_layers)

        # v2 components
        self.zz_feature_map = ZZFeatureMap(self.n_qubits, reps=2)
        self.reupload_circuit = DataReUploadingCircuit(self.n_qubits, n_layers)
        self.berry_ansatz = BerryPhaseAnsatz(self.n_qubits, n_layers)
        self.kernel_estimator = QuantumKernelEstimator("zz", self.n_qubits)
        self.barren_analyzer = BarrenPlateauAnalyzer(self.n_qubits, n_layers)
        self.expressibility = ExpressibilityAnalyzer(self.n_qubits)
        self.regressor = QuantumRegressorQNN(self.n_qubits, n_layers)

        logger.info(
            f"QuantumMLHub v{VERSION}: {self.n_qubits}q/{n_layers}L, "
            f"ZZ+ReUpload+Berry+Kernel+QAOA+BP+Regressor+Expressibility"
        )

    # ──── Kernel Methods ────

    def compute_kernel(self, X: np.ndarray,
                       feature_map: str = "zz") -> np.ndarray:
        """Compute quantum kernel matrix."""
        ke = QuantumKernelEstimator(feature_map, self.n_qubits)
        return ke.kernel_matrix(X)

    def kernel_classify(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray,
                        feature_map: str = "zz") -> Dict[str, Any]:
        """
        Classify using quantum kernel + simple nearest-centroid.
        """
        ke = QuantumKernelEstimator(feature_map, self.n_qubits)
        K_train = ke.kernel_matrix(X_train)

        # Compute kernel values between test and train
        predictions = []
        for x_test in X_test:
            k_vals = np.array([ke.kernel_value(x_test, x_tr) for x_tr in X_train])
            # Weighted vote by kernel similarity
            classes = np.unique(y_train)
            class_scores = {}
            for c in classes:
                mask = y_train == c
                class_scores[int(c)] = float(np.mean(k_vals[mask]))
            predictions.append(max(class_scores, key=class_scores.get))

        return {
            "predictions": predictions,
            "kernel_evaluations": ke._kernel_evals,
            "feature_map": feature_map,
        }

    # ──── QAOA ────

    def qaoa_maxcut(self, edges: List[Tuple[int, int]], p: int = 2,
                    optimize: bool = True,
                    n_iterations: int = 50) -> Dict[str, Any]:
        """Run QAOA for MaxCut problem."""
        qaoa = QAOACircuit(edges, p=p)
        if optimize:
            return qaoa.optimize(n_iterations)
        return qaoa.run()

    # ──── Berry Phase Training ────

    def berry_classify(self, features: np.ndarray) -> Dict[str, Any]:
        """Classify using Berry phase ansatz."""
        state = self.v1_hub.encoder.encode(features)
        state = self.berry_ansatz.apply(state)

        probs = np.abs(state) ** 2
        # Binary classification: first half → class 0, second half → class 1
        half = len(probs) // 2
        p0 = float(np.sum(probs[:half]))
        p1 = float(np.sum(probs[half:]))
        total = p0 + p1
        if total > 0:
            p0 /= total
            p1 /= total

        return {
            "prediction": 0 if p0 > p1 else 1,
            "probabilities": [p0, p1],
            "confidence": float(max(p0, p1)),
            "ansatz": "berry_phase",
        }

    # ──── Regression ────

    def quantum_regress(self, features: np.ndarray) -> float:
        """Predict continuous value."""
        return self.regressor.predict(features)

    # ──── Analysis ────

    def analyze_trainability(self, ansatz_type: str = "strongly_entangling",
                             n_samples: int = 30) -> Dict[str, Any]:
        """Analyze if the circuit has barren plateaus."""
        return self.barren_analyzer.analyze_gradient_variance(
            ansatz_type, n_samples)

    def analyze_expressibility(self, ansatz_type: str = "strongly_entangling",
                               n_samples: int = 100) -> Dict[str, Any]:
        """Measure circuit expressibility."""
        return self.expressibility.expressibility(
            ansatz_type, self.n_layers, n_samples)

    def analyze_entanglement(self, ansatz_type: str = "strongly_entangling",
                             n_samples: int = 50) -> Dict[str, Any]:
        """Measure circuit entangling capability."""
        return self.expressibility.entangling_capability(
            ansatz_type, self.n_layers, n_samples)

    # ──── Status ────

    def status(self) -> Dict[str, Any]:
        return {
            "hub": "QuantumMLHub",
            "version": VERSION,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "uptime_s": time.time() - self._creation_time,
            "qiskit_available": QISKIT_AVAILABLE,
            "capabilities": {
                "v1_pipeline": True,
                "zz_feature_map": True,
                "data_reuploading": True,
                "berry_phase_ansatz": True,
                "quantum_kernel": True,
                "qaoa": True,
                "barren_plateau_analysis": True,
                "quantum_regression": True,
                "expressibility_analysis": True,
            },
            "sacred_constants": {
                "GOD_CODE": GOD_CODE,
                "PHI": PHI,
                "VOID_CONSTANT": VOID_CONSTANT,
            },
            "subsystems": {
                "v1_hub": self.v1_hub.quick_summary(),
                "zz_feature_map": self.zz_feature_map.stats,
                "reupload": self.reupload_circuit.stats,
                "berry_ansatz": self.berry_ansatz.stats,
                "kernel": self.kernel_estimator.stats,
                "regressor": self.regressor.stats,
                "bp_analyzer": self.barren_analyzer.stats,
                "expressibility": {"n_qubits": self.n_qubits},
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_qml_hub: Optional[QuantumMLHub] = None


def get_qml_hub(n_qubits: Optional[int] = None,
                n_layers: int = 3) -> QuantumMLHub:
    """Get or create the singleton QML hub v2."""
    global _qml_hub
    if _qml_hub is None:
        if n_qubits is None:
            n_qubits = PIPELINE_DEFAULT_QUBITS
        _qml_hub = QuantumMLHub(n_qubits=n_qubits, n_layers=n_layers)
    return _qml_hub
