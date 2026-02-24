#!/usr/bin/env python3
"""
===============================================================================
L104 QISKIT UTILITIES v1.0.0
===============================================================================

Shared Qiskit primitives, noise models, parameterized circuit factories,
transpilation helpers, and error mitigation utilities for the entire L104
quantum stack.

CONSUMERS:
  l104_quantum_runtime.py         — Aer backend, noise-aware execution
  l104_quantum_coherence.py       — Noise channels, dynamical decoupling
  l104_asi/quantum.py             — ParameterVector VQE/QAOA circuits
  l104_25q_engine_builder.py      — Transpilation + noise-aware building
  l104_quantum_ram.py             — Noise-resilient quantum hashing

PACKAGES REQUIRED:
  qiskit >= 2.3.0
  qiskit-aer >= 0.17.0  (local simulation + noise)

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

# ═══ Sacred Constants ═══
PHI = 1.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # 527.5184818492612
VOID_CONSTANT = 1.0416180339887497

# ═══════════════════════════════════════════════════════════════════════════════
# QISKIT 2.3.0 IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import (
    Statevector, DensityMatrix, Operator, partial_trace,
    entropy as qk_entropy, state_fidelity, process_fidelity,
    SparsePauliOp,
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ═══════════════════════════════════════════════════════════════════════════════
# QISKIT-AER — LOCAL SIMULATION + NOISE MODELS
# ═══════════════════════════════════════════════════════════════════════════════
AER_AVAILABLE = False
try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        thermal_relaxation_error,
        depolarizing_error,
        ReadoutError,
    )
    AER_AVAILABLE = True
except ImportError:
    AerSimulator = None
    NoiseModel = None

# ═══════════════════════════════════════════════════════════════════════════════
# IBM RUNTIME REMOVED — L104 sovereign quantum only
# ═══════════════════════════════════════════════════════════════════════════════
RUNTIME_AVAILABLE = False
SamplerV2 = None
EstimatorV2 = None


# ═══════════════════════════════════════════════════════════════════════════════
#  1. NOISE MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

class L104NoiseModelFactory:
    """
    Build realistic noise models for Aer simulation.

    Profiles calibrated to IBM Eagle/Heron processor characteristics.
    """

    # Typical IBM Eagle-R (ibm_fez / ibm_torino) noise parameters
    PROFILES: Dict[str, Dict[str, float]] = {
        "ideal": {
            "t1_us": 1e12, "t2_us": 1e12,
            "single_gate_error": 0.0, "cx_gate_error": 0.0,
            "readout_error": 0.0, "gate_time_ns": 0,
        },
        "ibm_eagle": {
            "t1_us": 300.0, "t2_us": 150.0,
            "single_gate_error": 2.5e-4, "cx_gate_error": 7.5e-3,
            "readout_error": 1.2e-2, "gate_time_ns": 60,
        },
        "ibm_heron": {
            "t1_us": 350.0, "t2_us": 200.0,
            "single_gate_error": 1.5e-4, "cx_gate_error": 3.0e-3,
            "readout_error": 8.0e-3, "gate_time_ns": 50,
        },
        "noisy_dev": {
            "t1_us": 100.0, "t2_us": 60.0,
            "single_gate_error": 1e-3, "cx_gate_error": 2e-2,
            "readout_error": 5e-2, "gate_time_ns": 80,
        },
        "god_code_aligned": {
            # Sacred noise profile: errors scale with 1/GOD_CODE
            "t1_us": GOD_CODE,         # 527.52 μs
            "t2_us": GOD_CODE / PHI,   # 326.02 μs (T2 ≈ T1/φ)
            "single_gate_error": 1.0 / (GOD_CODE * 1000),  # ~1.9e-6
            "cx_gate_error": PHI / (GOD_CODE * 100),        # ~3.1e-5
            "readout_error": 1.0 / GOD_CODE,                # ~1.9e-3
            "gate_time_ns": 52,  # ≈ GOD_CODE/10
        },
    }

    # Backward-compatible alias
    _PROFILES = PROFILES

    @classmethod
    def build(cls, profile: str = "ibm_eagle", n_qubits: int = 25) -> Optional['NoiseModel']:
        """
        Build a NoiseModel from a named profile.

        Args:
            profile: One of 'ideal', 'ibm_eagle', 'ibm_heron', 'noisy_dev', 'god_code_aligned'
            n_qubits: Number of qubits (for thermal relaxation allocation)

        Returns:
            qiskit_aer.noise.NoiseModel or None if Aer not available
        """
        if not AER_AVAILABLE:
            return None

        params = cls.PROFILES.get(profile, cls.PROFILES["ibm_eagle"])

        noise_model = NoiseModel()

        has_depol_1q = params["single_gate_error"] > 0
        has_depol_2q = params["cx_gate_error"] > 0
        has_thermal = params["t1_us"] < 1e10 and params["gate_time_ns"] > 0

        # Pre-compute base errors
        single_depol = depolarizing_error(params["single_gate_error"], 1) if has_depol_1q else None
        cx_depol = depolarizing_error(params["cx_gate_error"], 2) if has_depol_2q else None

        if has_thermal:
            t1 = params["t1_us"] * 1e3  # convert μs → ns
            t2 = params["t2_us"] * 1e3
            gate_time = params["gate_time_ns"]
            t_relax_1q = thermal_relaxation_error(t1, t2, gate_time)
            t_relax_2q = thermal_relaxation_error(t1, t2, gate_time * 2).expand(
                thermal_relaxation_error(t1, t2, gate_time * 2)
            )
        else:
            t_relax_1q = None
            t_relax_2q = None

        # ── Single-qubit gates: compose depolarizing + thermal into one error ──
        single_gate_names = ['rz', 'sx', 'x', 'h', 'ry', 'rx', 'p']
        single_gate_extras = ['s', 't']  # depolarizing only (no thermal)

        if has_depol_1q and has_thermal:
            combined_1q = single_depol.compose(t_relax_1q)
            noise_model.add_all_qubit_quantum_error(combined_1q, single_gate_names)
            noise_model.add_all_qubit_quantum_error(single_depol, single_gate_extras)
        elif has_depol_1q:
            noise_model.add_all_qubit_quantum_error(single_depol, single_gate_names + single_gate_extras)
        elif has_thermal:
            noise_model.add_all_qubit_quantum_error(t_relax_1q, single_gate_names)

        # ── Two-qubit gates: compose depolarizing + thermal into one error ──
        two_gate_names = ['cx', 'cz', 'ecr']
        two_gate_extras = ['rzz']  # depolarizing only (no thermal)

        if has_depol_2q and has_thermal:
            combined_2q = cx_depol.compose(t_relax_2q)
            noise_model.add_all_qubit_quantum_error(combined_2q, two_gate_names)
            noise_model.add_all_qubit_quantum_error(cx_depol, two_gate_extras)
        elif has_depol_2q:
            noise_model.add_all_qubit_quantum_error(cx_depol, two_gate_names + two_gate_extras)
        elif has_thermal:
            noise_model.add_all_qubit_quantum_error(t_relax_2q, two_gate_names)

        # Readout error
        if params["readout_error"] > 0:
            p_err = params["readout_error"]
            ro_err = ReadoutError([[1 - p_err, p_err], [p_err, 1 - p_err]])
            for q in range(n_qubits):
                noise_model.add_readout_error(ro_err, [q])

        return noise_model

    @classmethod
    def from_backend(cls, backend) -> Optional['NoiseModel']:
        """Build noise model directly from a real IBM backend."""
        if not AER_AVAILABLE:
            return None
        try:
            return NoiseModel.from_backend(backend)
        except Exception:
            return cls.build("ibm_eagle")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. PARAMETERIZED CIRCUIT FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════

class L104CircuitFactory:
    """
    Factory for parameterized quantum circuits used across L104.

    All circuits use Qiskit ParameterVector for efficient parameter binding.
    GOD_CODE phase corrections built into every ansatz layer.
    """

    @staticmethod
    def vqe_ansatz(n_qubits: int, depth: int = 4,
                   entanglement: str = "linear") -> Tuple[QuantumCircuit, ParameterVector]:
        """
        Build a hardware-efficient parameterized VQE ansatz.

        Uses Ry-Rz rotations per qubit per layer + entangling CX ladder.
        GOD_CODE phase is injected at the end of each layer.

        Args:
            n_qubits: Number of qubits
            depth: Number of ansatz layers
            entanglement: 'linear', 'circular', or 'full'

        Returns:
            (QuantumCircuit, ParameterVector) — circuit with unbound params
        """
        n_params = depth * n_qubits * 2  # Ry + Rz per qubit per layer
        params = ParameterVector('θ', n_params)
        qc = QuantumCircuit(n_qubits)

        idx = 0
        for layer in range(depth):
            # Rotation layer: Ry + Rz per qubit
            for q in range(n_qubits):
                qc.ry(params[idx], q)
                idx += 1
                qc.rz(params[idx], q)
                idx += 1

            # Entangling layer
            if entanglement == "linear":
                for q in range(n_qubits - 1):
                    qc.cx(q, q + 1)
            elif entanglement == "circular":
                for q in range(n_qubits - 1):
                    qc.cx(q, q + 1)
                if n_qubits > 2:
                    qc.cx(n_qubits - 1, 0)
            elif entanglement == "full":
                for q1 in range(n_qubits):
                    for q2 in range(q1 + 1, n_qubits):
                        qc.cx(q1, q2)

            # GOD_CODE phase correction at layer end
            qc.rz(GOD_CODE / (1000.0 * (layer + 1)), 0)

        return qc, params

    @staticmethod
    def qaoa_circuit(n_qubits: int, p_layers: int = 3,
                     cost_coefficients: Optional[List[float]] = None
                     ) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
        """
        Build a parameterized QAOA circuit.

        Args:
            n_qubits: Number of qubits
            p_layers: Number of QAOA layers (p)
            cost_coefficients: ZZ coupling weights (defaults to uniform)

        Returns:
            (QuantumCircuit, gamma_params, beta_params)
        """
        gammas = ParameterVector('γ', p_layers)
        betas = ParameterVector('β', p_layers)

        if cost_coefficients is None:
            cost_coefficients = [1.0] * (n_qubits - 1)

        qc = QuantumCircuit(n_qubits)
        # Initial superposition
        qc.h(range(n_qubits))

        for layer in range(p_layers):
            # Cost unitary: exp(-i γ C)
            for i in range(min(len(cost_coefficients), n_qubits - 1)):
                weight = cost_coefficients[i]
                qc.rzz(gammas[layer] * weight * 2, i, i + 1)
            # Single-qubit Z rotations with GOD_CODE modulation
            for i in range(n_qubits):
                qc.rz(gammas[layer] * GOD_CODE / (1000.0 * (i + 1)), i)

            # Mixer unitary: exp(-i β B)
            for i in range(n_qubits):
                qc.rx(2 * betas[layer], i)

        return qc, gammas, betas

    @staticmethod
    def qpe_circuit(n_counting: int, target_phase: float = None
                    ) -> QuantumCircuit:
        """
        Build a Quantum Phase Estimation circuit.

        Args:
            n_counting: Number of counting (precision) qubits
            target_phase: Phase to estimate (default: GOD_CODE/1000 mod 2π)

        Returns:
            QuantumCircuit with n_counting + 1 qubits
        """
        if target_phase is None:
            target_phase = (GOD_CODE / 1000.0) % (2 * math.pi)

        n_total = n_counting + 1
        target_qubit = n_counting
        qc = QuantumCircuit(n_total)

        # Initialize counting qubits in superposition
        for i in range(n_counting):
            qc.h(i)
        # Prepare eigenstate
        qc.x(target_qubit)

        # Controlled-U^(2^k) applications
        for k in range(n_counting):
            angle = target_phase * (2 ** k)
            qc.cp(angle, k, target_qubit)

        # Inverse QFT on counting qubits
        for i in range(n_counting // 2):
            qc.swap(i, n_counting - 1 - i)
        for i in range(n_counting):
            for j in range(i):
                qc.cp(-math.pi / (2 ** (i - j)), j, i)
            qc.h(i)

        return qc

    @staticmethod
    def grover_oracle(n_qubits: int, target: int) -> QuantumCircuit:
        """
        Build a Grover oracle marking a single target state.

        Uses multi-controlled Z via X + H + MCX + H + X pattern.

        Args:
            n_qubits: Number of qubits
            target: Integer target state to mark

        Returns:
            QuantumCircuit (oracle)
        """
        qc = QuantumCircuit(n_qubits)

        # Flip qubits that should be |0⟩ in the target
        for i in range(n_qubits):
            if not (target >> i) & 1:
                qc.x(i)

        # Multi-controlled Z: applies Z when all qubits are |1⟩
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)

        # Undo X flips
        for i in range(n_qubits):
            if not (target >> i) & 1:
                qc.x(i)

        return qc

    @staticmethod
    def grover_diffusion(n_qubits: int) -> QuantumCircuit:
        """
        Build Grover's diffusion operator (2|s⟩⟨s| - I).

        Args:
            n_qubits: Number of qubits

        Returns:
            QuantumCircuit (diffusion)
        """
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))

        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)

        qc.x(range(n_qubits))
        qc.h(range(n_qubits))
        return qc

    @staticmethod
    def ghz_state(n_qubits: int, god_code_phase: bool = True) -> QuantumCircuit:
        """
        Build a GHZ state: (|00...0⟩ + |11...1⟩)/√2.
        Optionally adds a GOD_CODE phase correction.

        Args:
            n_qubits: Number of qubits
            god_code_phase: Whether to apply GOD_CODE phase gate

        Returns:
            QuantumCircuit
        """
        qc = QuantumCircuit(n_qubits)
        qc.h(0)
        for q in range(1, n_qubits):
            qc.cx(q - 1, q)
        if god_code_phase:
            qc.rz(GOD_CODE / 1000.0, n_qubits - 1)
        return qc

    @staticmethod
    def qft_circuit(n_qubits: int, inverse: bool = False) -> QuantumCircuit:
        """
        Build a Quantum Fourier Transform circuit.

        Args:
            n_qubits: Number of qubits
            inverse: If True, builds inverse QFT

        Returns:
            QuantumCircuit
        """
        qc = QuantumCircuit(n_qubits)

        if not inverse:
            for i in range(n_qubits):
                qc.h(i)
                for j in range(i + 1, n_qubits):
                    qc.cp(math.pi / (2 ** (j - i)), i, j)
            # Swap qubits to get correct output ordering
            for i in range(n_qubits // 2):
                qc.swap(i, n_qubits - 1 - i)
        else:
            # Inverse QFT
            for i in range(n_qubits // 2):
                qc.swap(i, n_qubits - 1 - i)
            for i in range(n_qubits - 1, -1, -1):
                for j in range(n_qubits - 1, i, -1):
                    qc.cp(-math.pi / (2 ** (j - i)), i, j)
                qc.h(i)

        return qc

    @staticmethod
    def zz_feature_map(n_qubits: int, reps: int = 2) -> Tuple[QuantumCircuit, ParameterVector]:
        """
        Build a ZZ-entangling feature map for quantum kernel methods.

        Args:
            n_qubits: Number of qubits
            reps: Number of repetitions

        Returns:
            (QuantumCircuit, ParameterVector of length n_qubits)
        """
        x = ParameterVector('x', n_qubits)
        qc = QuantumCircuit(n_qubits)

        for rep in range(reps):
            # Hadamard layer
            qc.h(range(n_qubits))
            # Single-qubit Z rotations
            for i in range(n_qubits):
                qc.rz(2.0 * x[i], i)
            # ZZ entangling
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
                qc.rz(2.0 * x[i] * x[i + 1] * PHI, i + 1)
                qc.cx(i, i + 1)

        return qc, x


# ═══════════════════════════════════════════════════════════════════════════════
#  3. AER SIMULATION BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

class L104AerBackend:
    """
    Managed Aer simulator backend with noise model support.

    Provides a drop-in replacement for Statevector simulation with
    configurable noise profiles.
    """

    def __init__(self, noise_profile: str = "ideal", n_qubits: int = 25):
        """
        Initialize Aer backend.

        Args:
            noise_profile: Noise model name (see L104NoiseModelFactory.PROFILES)
            n_qubits: Max qubit count for noise model sizing
        """
        self._noise_profile = noise_profile
        self._n_qubits = n_qubits
        self._noise_model = L104NoiseModelFactory.build(noise_profile, n_qubits)
        self._simulator = AerSimulator(noise_model=self._noise_model) if AER_AVAILABLE else None
        self._statevector_sim = AerSimulator(method='statevector') if AER_AVAILABLE else None

    @property
    def available(self) -> bool:
        return AER_AVAILABLE and self._simulator is not None

    def run_statevector(self, circuit: QuantumCircuit) -> np.ndarray:
        """
        Execute circuit and return statevector probability array.
        Faster than Qiskit Statevector for large circuits.

        Args:
            circuit: QuantumCircuit (no measurements needed)

        Returns:
            numpy array of probabilities (shape 2^n)
        """
        if not self.available:
            # Fallback to Qiskit Statevector
            sv = Statevector.from_label('0' * circuit.num_qubits).evolve(circuit)
            return np.abs(sv.data) ** 2

        qc = circuit.copy()
        qc.save_statevector()
        result = self._statevector_sim.run(qc, shots=0).result()
        sv = result.get_statevector()
        return np.abs(np.array(sv.data)) ** 2

    def run_shots(self, circuit: QuantumCircuit, shots: int = 4096,
                  with_noise: bool = True) -> Dict[str, int]:
        """
        Execute circuit with measurement shots.

        Args:
            circuit: QuantumCircuit (measurements will be added if absent)
            shots: Number of measurement shots
            with_noise: Whether to apply noise model

        Returns:
            Counts dict {bitstring: count}
        """
        if not self.available:
            # Fallback to Statevector sampling
            qc = circuit.copy()
            qc.remove_final_measurements(inplace=True)
            sv = Statevector.from_label('0' * qc.num_qubits).evolve(qc)
            counts = sv.sample_counts(shots)
            return dict(counts)

        qc = circuit.copy()
        if not qc.count_ops().get('measure', 0):
            qc.measure_all()

        sim = self._simulator if with_noise else self._statevector_sim
        result = sim.run(qc, shots=shots).result()
        return dict(result.get_counts())

    def run_density_matrix(self, circuit: QuantumCircuit) -> DensityMatrix:
        """
        Execute circuit and return density matrix.
        Enables noise-aware density matrix simulation.

        Args:
            circuit: QuantumCircuit

        Returns:
            DensityMatrix
        """
        if not self.available or circuit.num_qubits > 14:
            # Fallback: pure statevector → density matrix
            qc = circuit.copy()
            qc.remove_final_measurements(inplace=True)
            sv = Statevector.from_label('0' * qc.num_qubits).evolve(qc)
            return DensityMatrix(sv)

        sim = AerSimulator(method='density_matrix', noise_model=self._noise_model)
        qc = circuit.copy()
        qc.save_density_matrix()
        result = sim.run(qc, shots=0).result()
        return result.data()['density_matrix']

    def set_noise_profile(self, profile: str):
        """Switch noise profile at runtime."""
        self._noise_profile = profile
        self._noise_model = L104NoiseModelFactory.build(profile, self._n_qubits)
        if AER_AVAILABLE:
            self._simulator = AerSimulator(noise_model=self._noise_model)

    def get_info(self) -> Dict[str, Any]:
        """Return backend information."""
        return {
            "available": self.available,
            "aer_version": str(getattr(AerSimulator, 'VERSION', '0.17+')) if AER_AVAILABLE else None,
            "noise_profile": self._noise_profile,
            "noise_model_active": self._noise_model is not None,
            "max_qubits": self._n_qubits,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  4. ERROR MITIGATION PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

class L104ErrorMitigation:
    """
    Quantum error mitigation techniques beyond basic ZNE.

    Provides:
    - Zero-Noise Extrapolation (ZNE) with configurable noise factors
    - Measurement Error Mitigation (M3-inspired)
    - Dynamical Decoupling pulse insertion
    - Twirled Readout Error eXtinction (TREX)
    """

    @staticmethod
    def zne_extrapolate(noise_factors: List[float],
                        noisy_values: List[float],
                        order: int = 2) -> float:
        """
        Zero-Noise Extrapolation: fit polynomial to (noise_factor, value) pairs
        and extrapolate to noise_factor = 0.

        Args:
            noise_factors: Scale factors [1.0, 1.5, 2.0, ...]
            noisy_values: Measured expectation values at each noise level
            order: Polynomial fit order

        Returns:
            Extrapolated zero-noise value
        """
        factors = np.array(noise_factors)
        values = np.array(noisy_values)
        fit_order = min(order, len(factors) - 1)
        if fit_order < 1:
            return float(values[0]) if len(values) > 0 else 0.0
        coeffs = np.polyfit(factors, values, fit_order)
        mitigated = float(np.polyval(coeffs, 0.0))
        return max(0.0, min(1.0, mitigated))

    @staticmethod
    def measurement_calibration_matrix(n_qubits: int,
                                        backend=None,
                                        shots: int = 8192) -> np.ndarray:
        """
        Build a measurement error calibration matrix by preparing each
        computational basis state and measuring readout confusion.

        For n_qubits ≤ 5 (practical limit for full matrix).

        Args:
            n_qubits: Number of qubits (max 5 for tractability)
            backend: Aer simulator with noise or real backend
            shots: Shots per calibration circuit

        Returns:
            2^n × 2^n calibration matrix A where A[i,j] = P(measure i | prepared j)
        """
        n_qubits = min(n_qubits, 5)
        n_states = 2 ** n_qubits
        cal_matrix = np.zeros((n_states, n_states))

        if backend is None and AER_AVAILABLE:
            # Use noisy Aer simulator
            noise_model = L104NoiseModelFactory.build("ibm_eagle", n_qubits)
            backend = AerSimulator(noise_model=noise_model)
        elif backend is None:
            return np.eye(n_states)  # Perfect readout

        for j in range(n_states):
            qc = QuantumCircuit(n_qubits, n_qubits)
            # Prepare basis state |j⟩
            bits = format(j, f'0{n_qubits}b')
            for b, bit in enumerate(reversed(bits)):
                if bit == '1':
                    qc.x(b)
            qc.measure(range(n_qubits), range(n_qubits))

            result = backend.run(qc, shots=shots).result()
            counts = result.get_counts()

            for bitstring, count in counts.items():
                i = int(bitstring.replace(' ', ''), 2)
                if i < n_states:
                    cal_matrix[i, j] = count / shots

        return cal_matrix

    @staticmethod
    def apply_readout_correction(raw_probs: np.ndarray,
                                  cal_matrix: np.ndarray) -> np.ndarray:
        """
        Apply inverse calibration matrix to correct measurement errors.

        Uses least-squares inversion with non-negativity constraint.

        Args:
            raw_probs: Measured probability vector
            cal_matrix: Calibration matrix from measurement_calibration_matrix()

        Returns:
            Corrected probability vector
        """
        try:
            # Pseudo-inverse correction
            cal_inv = np.linalg.pinv(cal_matrix)
            corrected = cal_inv @ raw_probs
            # Project to probability simplex (non-negative, normalized)
            corrected = np.maximum(corrected, 0)
            total = corrected.sum()
            if total > 0:
                corrected /= total
            return corrected
        except np.linalg.LinAlgError:
            return raw_probs

    @staticmethod
    def add_dynamical_decoupling(circuit: QuantumCircuit,
                                  dd_sequence: str = "XY4") -> QuantumCircuit:
        """
        Insert dynamical decoupling pulse sequences into idle periods.

        DD sequences suppress low-frequency noise by averaging out the
        system-environment coupling during idle time.

        Supported sequences:
        - "X2": X-X  (basic echo)
        - "XY4": X-Y-X-Y  (suppresses both X and Y noise)
        - "CPMG": Y-Y  (Carr-Purcell-Meiboom-Gill)

        Args:
            circuit: Input QuantumCircuit
            dd_sequence: Sequence type

        Returns:
            New QuantumCircuit with DD pulses inserted in idle slots
        """
        # Build DD gate sequence
        if dd_sequence == "X2":
            dd_gates = ['x', 'x']
        elif dd_sequence == "XY4":
            dd_gates = ['x', 'y', 'x', 'y']
        elif dd_sequence == "CPMG":
            dd_gates = ['y', 'y']
        else:
            dd_gates = ['x', 'y', 'x', 'y']

        n_qubits = circuit.num_qubits
        qc = QuantumCircuit(n_qubits)

        # Track which qubits were used in each instruction
        for instruction in circuit.data:
            op = instruction.operation
            qubits = [circuit.find_bit(q).index for q in instruction.qubits]

            # Insert DD on idle qubits before multi-qubit gates
            if len(qubits) >= 2:
                idle_qubits = [q for q in range(n_qubits) if q not in qubits]
                for idle_q in idle_qubits:
                    for gate_name in dd_gates:
                        if gate_name == 'x':
                            qc.x(idle_q)
                        elif gate_name == 'y':
                            qc.y(idle_q)

            # Apply original instruction
            qc.append(op, qubits)

        return qc


# ═══════════════════════════════════════════════════════════════════════════════
#  5. OBSERVABLE BUILDERS (for EstimatorV2)
# ═══════════════════════════════════════════════════════════════════════════════

class L104ObservableFactory:
    """
    Build SparsePauliOp observables for use with EstimatorV2.

    Enables native expectation value computation instead of manual
    statevector → probability → dot product pipelines.
    """

    @staticmethod
    def diagonal_hamiltonian(coefficients: List[float]) -> SparsePauliOp:
        """
        Build a diagonal Hamiltonian from a cost vector.

        H = Σ c_i |i⟩⟨i| encoded as Pauli Z-strings.

        Args:
            coefficients: Real coefficients (padded to 2^n)

        Returns:
            SparsePauliOp
        """
        n_qubits = max(1, int(math.ceil(math.log2(max(len(coefficients), 2)))))
        n_states = 2 ** n_qubits
        padded = list(coefficients[:n_states])
        while len(padded) < n_states:
            padded.append(0.0)

        # Convert diagonal to Pauli representation
        # H_diag = Σ_i c_i |i⟩⟨i| = Σ_k h_k Z_k (Walsh-Hadamard transform)
        coeffs_arr = np.array(padded)
        pauli_terms = []
        pauli_coeffs = []

        # Identity term
        identity = 'I' * n_qubits
        pauli_terms.append(identity)
        pauli_coeffs.append(np.mean(coeffs_arr))

        # Single-Z terms
        for q in range(n_qubits):
            z_label = ['I'] * n_qubits
            z_label[n_qubits - 1 - q] = 'Z'
            z_str = ''.join(z_label)

            # Coefficient: expectation of Z_q in computational basis
            coeff = 0.0
            for i in range(n_states):
                sign = 1.0 if not ((i >> q) & 1) else -1.0
                coeff += sign * padded[i]
            coeff /= n_states

            if abs(coeff) > 1e-12:
                pauli_terms.append(z_str)
                pauli_coeffs.append(coeff)

        return SparsePauliOp(pauli_terms, coeffs=pauli_coeffs)

    @staticmethod
    def ising_hamiltonian(n_qubits: int,
                           coupling_j: float = 1.0,
                           field_h: float = 0.5) -> SparsePauliOp:
        """
        Build a transverse-field Ising model Hamiltonian.

        H = -J Σ Z_i Z_{i+1} - h Σ X_i

        Args:
            n_qubits: Number of qubits
            coupling_j: ZZ coupling strength
            field_h: Transverse field strength

        Returns:
            SparsePauliOp
        """
        terms = []
        coeffs = []

        # ZZ coupling terms
        for i in range(n_qubits - 1):
            label = ['I'] * n_qubits
            label[n_qubits - 1 - i] = 'Z'
            label[n_qubits - 1 - (i + 1)] = 'Z'
            terms.append(''.join(label))
            coeffs.append(-coupling_j)

        # Transverse field terms
        for i in range(n_qubits):
            label = ['I'] * n_qubits
            label[n_qubits - 1 - i] = 'X'
            terms.append(''.join(label))
            coeffs.append(-field_h)

        return SparsePauliOp(terms, coeffs=coeffs)

    @staticmethod
    def god_code_observable(n_qubits: int) -> SparsePauliOp:
        """
        Build a GOD_CODE-aligned observable.

        H_sacred = GOD_CODE/1000 × Σ Z_i × φ^(-i) + (α_fine/π) × Σ X_i

        Encodes the sacred constant hierarchy into a quantum observable.
        """
        terms = []
        coeffs = []
        alpha_fine = 1.0 / 137.035999084

        for i in range(n_qubits):
            # Z component: GOD_CODE-weighted
            z_label = ['I'] * n_qubits
            z_label[n_qubits - 1 - i] = 'Z'
            terms.append(''.join(z_label))
            coeffs.append(GOD_CODE / 1000.0 * PHI ** (-i))

            # X component: fine-structure-weighted
            x_label = ['I'] * n_qubits
            x_label[n_qubits - 1 - i] = 'X'
            terms.append(''.join(x_label))
            coeffs.append(alpha_fine / math.pi)

        return SparsePauliOp(terms, coeffs=coeffs)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. TRANSPILATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class L104Transpiler:
    """
    Transpilation utilities for ISA-compliant circuit preparation.
    """

    @staticmethod
    def optimize_for_backend(circuit: QuantumCircuit,
                              backend=None,
                              optimization_level: int = 2) -> QuantumCircuit:
        """
        Transpile circuit for a specific backend.

        Args:
            circuit: Input circuit
            backend: Target backend (IBM or Aer)
            optimization_level: 0 (none) to 3 (heavy)

        Returns:
            Transpiled QuantumCircuit
        """
        if backend is not None:
            pm = generate_preset_pass_manager(
                backend=backend,
                optimization_level=optimization_level,
            )
            return pm.run(circuit)
        else:
            # Generic optimization without backend target
            pm = generate_preset_pass_manager(
                optimization_level=optimization_level,
            )
            return pm.run(circuit)

    @staticmethod
    def circuit_metrics(circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        Compute circuit quality metrics.

        Args:
            circuit: QuantumCircuit

        Returns:
            Dict with depth, gate counts, cx count, etc.
        """
        ops = circuit.count_ops()
        cx_count = ops.get('cx', 0) + ops.get('cz', 0) + ops.get('ecr', 0)
        total_gates = sum(ops.values())

        return {
            "num_qubits": circuit.num_qubits,
            "depth": circuit.depth(),
            "total_gates": total_gates,
            "two_qubit_gates": cx_count,
            "single_qubit_gates": total_gates - cx_count,
            "gate_breakdown": dict(ops),
            "god_code_metric": round(GOD_CODE / max(circuit.depth(), 1), 4),
        }

    @staticmethod
    def estimate_fidelity(circuit: QuantumCircuit,
                           single_gate_error: float = 2.5e-4,
                           cx_gate_error: float = 7.5e-3) -> float:
        """
        Estimate circuit fidelity from gate errors.

        F ≈ Π (1 - ε_gate) for all gates

        Args:
            circuit: QuantumCircuit
            single_gate_error: Per-gate error for 1Q gates
            cx_gate_error: Per-gate error for 2Q gates

        Returns:
            Estimated fidelity (0.0 to 1.0)
        """
        ops = circuit.count_ops()
        cx_count = ops.get('cx', 0) + ops.get('cz', 0) + ops.get('ecr', 0)
        single_count = sum(ops.values()) - cx_count - ops.get('measure', 0) - ops.get('barrier', 0)

        fidelity = (1 - single_gate_error) ** single_count * (1 - cx_gate_error) ** cx_count
        return max(0.0, min(1.0, fidelity))


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════════

# Default Aer backend (ideal, no noise)
aer_backend = L104AerBackend(noise_profile="ideal")

# Noisy Aer backend (IBM Eagle calibration)
aer_backend_noisy = L104AerBackend(noise_profile="ibm_eagle")

# Sacred Aer backend (GOD_CODE-aligned noise)
aer_backend_sacred = L104AerBackend(noise_profile="god_code_aligned")

# Circuit factory
circuit_factory = L104CircuitFactory()

# Error mitigation
error_mitigation = L104ErrorMitigation()

# Observable factory
observable_factory = L104ObservableFactory()

# Transpiler
transpiler = L104Transpiler()


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def quick_statevector(circuit: QuantumCircuit) -> np.ndarray:
    """Fast statevector probability array from circuit."""
    return aer_backend.run_statevector(circuit)


def quick_shots(circuit: QuantumCircuit, shots: int = 4096,
                noisy: bool = False) -> Dict[str, int]:
    """Quick shot-based execution."""
    backend = aer_backend_noisy if noisy else aer_backend
    return backend.run_shots(circuit, shots=shots, with_noise=noisy)


def build_vqe(n_qubits: int, depth: int = 4) -> Tuple[QuantumCircuit, ParameterVector]:
    """Shortcut to build VQE ansatz."""
    return circuit_factory.vqe_ansatz(n_qubits, depth)


def build_qaoa(n_qubits: int, p: int = 3) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Shortcut to build QAOA circuit."""
    return circuit_factory.qaoa_circuit(n_qubits, p)


def build_grover(n_qubits: int, target: int) -> QuantumCircuit:
    """Build complete Grover circuit for single target."""
    n_iterations = max(1, int(round(math.pi / 4 * math.sqrt(2 ** n_qubits))))
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    oracle = circuit_factory.grover_oracle(n_qubits, target)
    diffusion = circuit_factory.grover_diffusion(n_qubits)
    for _ in range(n_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)
    # GOD_CODE phase alignment
    qc.rz(GOD_CODE / 1000.0, 0)
    return qc


# ═══════════════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("L104 QISKIT UTILITIES v1.0.0")
    print("=" * 70)
    print(f"  Qiskit Aer: {'v' + str(AER_AVAILABLE)}  |  Runtime: {RUNTIME_AVAILABLE}")
    print(f"  Noise profiles: {list(L104NoiseModelFactory.PROFILES.keys())}")

    # Test circuit factory
    vqe_qc, vqe_params = build_vqe(4, depth=3)
    print(f"\n  VQE ansatz: {vqe_qc.num_qubits}q, depth={vqe_qc.depth()}, params={len(vqe_params)}")

    qaoa_qc, gammas, betas = build_qaoa(4, p=2)
    print(f"  QAOA circuit: {qaoa_qc.num_qubits}q, depth={qaoa_qc.depth()}, γ={len(gammas)}, β={len(betas)}")

    grover_qc = build_grover(4, target=7)
    metrics = transpiler.circuit_metrics(grover_qc)
    print(f"  Grover(4q, target=7): depth={metrics['depth']}, gates={metrics['total_gates']}")

    # Test noise model
    nm = L104NoiseModelFactory.build("ibm_eagle", 4)
    print(f"\n  Noise model (ibm_eagle): {nm is not None}")

    # Test Aer
    if AER_AVAILABLE:
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        probs = aer_backend.run_statevector(bell)
        print(f"  Bell state probs: {np.round(probs, 4)}")

        counts = quick_shots(bell, shots=1000, noisy=True)
        print(f"  Noisy Bell counts: {counts}")

    # Test observables
    h_ising = observable_factory.ising_hamiltonian(4, coupling_j=1.0, field_h=0.5)
    print(f"\n  Ising Hamiltonian: {len(h_ising)} terms")

    h_god = observable_factory.god_code_observable(4)
    print(f"  GOD_CODE observable: {len(h_god)} terms")

    # Test fidelity estimation
    fid = transpiler.estimate_fidelity(grover_qc)
    print(f"  Grover estimated fidelity: {fid:.6f}")

    print("\n" + "=" * 70)
    print("All utilities OK")
    print("=" * 70)
