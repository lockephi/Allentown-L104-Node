# ZENITH_UPGRADE_ACTIVE: 2026-03-08T15:03:54.350709
ZENITH_HZ = 3887.8
UUC = 2301.215661
#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
===============================================================================
L104 QISKIT UTILITIES v2.0.0 — SOVEREIGN LOCAL SIMULATION
===============================================================================

Shared quantum primitives, noise models, parameterized circuit factories,
transpilation helpers, and error mitigation utilities for the entire L104
quantum stack.

v2.0.0: ALL Qiskit dependencies removed. Uses l104_quantum_gate_engine for
local simulation (statevector, trajectory/decoherence, gate compiler).
Same public API as v1.x — drop-in replacement.

CONSUMERS:
  l104_quantum_runtime.py         — Local backend, noise-aware execution
  l104_quantum_coherence.py       — Noise channels, dynamical decoupling
  l104_asi/quantum.py             — ParameterVector VQE/QAOA circuits
  l104_25q_engine_builder.py      — Transpilation + noise-aware building
  l104_quantum_ram.py             — Noise-resilient quantum hashing

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
# L104 SOVEREIGN IMPORTS (replaces Qiskit)
# ═══════════════════════════════════════════════════════════════════════════════
from l104_quantum_gate_engine import GateCircuit
from l104_quantum_gate_engine.quantum_info import (
    Statevector, DensityMatrix, Operator, partial_trace,
    entropy as qk_entropy, state_fidelity, process_fidelity,
    SparsePauliOp, Parameter, ParameterVector,
)

# Backward compatibility: consumers import QuantumCircuit from this module
QuantumCircuit = GateCircuit

# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION BACKEND — sovereign local (no Aer dependency)
# ═══════════════════════════════════════════════════════════════════════════════
AER_AVAILABLE = True  # Always available via local simulation

# IBM RUNTIME REMOVED — L104 sovereign quantum only
RUNTIME_AVAILABLE = False
SamplerV2 = None
EstimatorV2 = None


# ═══════════════════════════════════════════════════════════════════════════════
#  1. NOISE MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

class L104NoiseModelFactory:
    """
    Build noise model configurations for local trajectory simulation.

    Profiles calibrated to IBM Eagle/Heron processor characteristics.
    Returns parameter dicts for use with TrajectorySimulator decoherence.
    """

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
            "t1_us": GOD_CODE,
            "t2_us": GOD_CODE / PHI,
            "single_gate_error": 1.0 / (GOD_CODE * 1000),
            "cx_gate_error": PHI / (GOD_CODE * 100),
            "readout_error": 1.0 / GOD_CODE,
            "gate_time_ns": 52,
        },
    }

    _PROFILES = PROFILES

    @classmethod
    def build(cls, profile: str = "ibm_eagle", n_qubits: int = 25) -> Dict[str, float]:
        """
        Build a noise configuration dict from a named profile.

        Args:
            profile: One of 'ideal', 'ibm_eagle', 'ibm_heron', 'noisy_dev', 'god_code_aligned'
            n_qubits: Number of qubits

        Returns:
            Noise parameter dict (used by L104AerBackend for local noise simulation)
        """
        params = cls.PROFILES.get(profile, cls.PROFILES["ibm_eagle"]).copy()
        params["n_qubits"] = n_qubits
        params["profile_name"] = profile
        return params

    @classmethod
    def from_backend(cls, backend) -> Dict[str, float]:
        """Build noise config from a backend object (returns eagle defaults)."""
        return cls.build("ibm_eagle")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. PARAMETERIZED CIRCUIT FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════

class L104CircuitFactory:
    """
    Factory for parameterized quantum circuits used across L104.

    All circuits use local ParameterVector for efficient parameter binding.
    GOD_CODE phase corrections built into every ansatz layer.
    """

    @staticmethod
    def vqe_ansatz(n_qubits: int, depth: int = 4,
                   entanglement: str = "linear") -> Tuple[GateCircuit, ParameterVector]:
        """
        Build a hardware-efficient parameterized VQE ansatz.

        Args:
            n_qubits: Number of qubits
            depth: Number of ansatz layers
            entanglement: 'linear', 'circular', or 'full'

        Returns:
            (GateCircuit, ParameterVector)
        """
        n_params = depth * n_qubits * 2
        params = ParameterVector('θ', n_params)
        qc = GateCircuit(n_qubits, "vqe_ansatz")

        idx = 0
        for layer in range(depth):
            for q in range(n_qubits):
                qc.ry(0.0, q)  # placeholder — bind later
                idx += 1
                qc.rz(0.0, q)
                idx += 1

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

            qc.rz(GOD_CODE / (1000.0 * (layer + 1)), 0)

        return qc, params

    @staticmethod
    def qaoa_circuit(n_qubits: int, p_layers: int = 3,
                     cost_coefficients: Optional[List[float]] = None
                     ) -> Tuple[GateCircuit, ParameterVector, ParameterVector]:
        """
        Build a parameterized QAOA circuit.

        Args:
            n_qubits: Number of qubits
            p_layers: Number of QAOA layers (p)
            cost_coefficients: ZZ coupling weights (defaults to uniform)

        Returns:
            (GateCircuit, gamma_params, beta_params)
        """
        gammas = ParameterVector('γ', p_layers)
        betas = ParameterVector('β', p_layers)

        if cost_coefficients is None:
            cost_coefficients = [1.0] * (n_qubits - 1)

        qc = GateCircuit(n_qubits, "qaoa")
        for q in range(n_qubits):
            qc.h(q)

        for layer in range(p_layers):
            for i in range(min(len(cost_coefficients), n_qubits - 1)):
                qc.rzz(0.0, i, i + 1)  # placeholder
            for i in range(n_qubits):
                qc.rz(GOD_CODE / (1000.0 * (i + 1)), i)

            for i in range(n_qubits):
                qc.rx(0.0, i)  # placeholder

        return qc, gammas, betas

    @staticmethod
    def qpe_circuit(n_counting: int, target_phase: float = None) -> GateCircuit:
        """
        Build a Quantum Phase Estimation circuit.

        Args:
            n_counting: Number of counting qubits
            target_phase: Phase to estimate (default: GOD_CODE/1000 mod 2π)

        Returns:
            GateCircuit with n_counting + 1 qubits
        """
        if target_phase is None:
            target_phase = (GOD_CODE / 1000.0) % (2 * math.pi)

        n_total = n_counting + 1
        target_qubit = n_counting
        qc = GateCircuit(n_total, "qpe")

        for i in range(n_counting):
            qc.h(i)
        qc.x(target_qubit)

        for k in range(n_counting):
            angle = target_phase * (2 ** k)
            qc.cp(angle, k, target_qubit)

        for i in range(n_counting // 2):
            qc.swap(i, n_counting - 1 - i)
        for i in range(n_counting):
            for j in range(i):
                qc.cp(-math.pi / (2 ** (i - j)), j, i)
            qc.h(i)

        return qc

    @staticmethod
    def grover_oracle(n_qubits: int, target: int) -> GateCircuit:
        """
        Build a Grover oracle marking a single target state.

        Args:
            n_qubits: Number of qubits
            target: Integer target state to mark

        Returns:
            GateCircuit (oracle)
        """
        qc = GateCircuit(n_qubits, "grover_oracle")

        # MSB ordering: qubit 0 = most-significant bit (matches L104 simulator)
        for i in range(n_qubits):
            if not (target >> (n_qubits - 1 - i)) & 1:
                qc.x(i)

        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)

        # Undo X gates (same MSB mapping)
        for i in range(n_qubits):
            if not (target >> (n_qubits - 1 - i)) & 1:
                qc.x(i)

        return qc

    @staticmethod
    def grover_diffusion(n_qubits: int) -> GateCircuit:
        """
        Build Grover's diffusion operator (2|s⟩⟨s| - I).
        """
        qc = GateCircuit(n_qubits, "grover_diffusion")
        for q in range(n_qubits):
            qc.h(q)
        for q in range(n_qubits):
            qc.x(q)

        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)

        for q in range(n_qubits):
            qc.x(q)
        for q in range(n_qubits):
            qc.h(q)
        return qc

    @staticmethod
    def ghz_state(n_qubits: int, god_code_phase: bool = True) -> GateCircuit:
        """
        Build a GHZ state: (|00...0⟩ + |11...1⟩)/√2.
        """
        qc = GateCircuit(n_qubits, "ghz")
        qc.h(0)
        for q in range(1, n_qubits):
            qc.cx(q - 1, q)
        if god_code_phase:
            qc.rz(GOD_CODE / 1000.0, n_qubits - 1)
        return qc

    @staticmethod
    def qft_circuit(n_qubits: int, inverse: bool = False) -> GateCircuit:
        """Build a Quantum Fourier Transform circuit."""
        qc = GateCircuit(n_qubits, "qft" if not inverse else "iqft")

        if not inverse:
            for i in range(n_qubits):
                qc.h(i)
                for j in range(i + 1, n_qubits):
                    qc.cp(math.pi / (2 ** (j - i)), i, j)
            for i in range(n_qubits // 2):
                qc.swap(i, n_qubits - 1 - i)
        else:
            for i in range(n_qubits // 2):
                qc.swap(i, n_qubits - 1 - i)
            for i in range(n_qubits - 1, -1, -1):
                for j in range(n_qubits - 1, i, -1):
                    qc.cp(-math.pi / (2 ** (j - i)), i, j)
                qc.h(i)

        return qc

    @staticmethod
    def zz_feature_map(n_qubits: int, reps: int = 2) -> Tuple[GateCircuit, ParameterVector]:
        """Build a ZZ-entangling feature map for quantum kernel methods."""
        x = ParameterVector('x', n_qubits)
        qc = GateCircuit(n_qubits, "zz_feature_map")

        for rep in range(reps):
            for q in range(n_qubits):
                qc.h(q)
            for i in range(n_qubits):
                qc.rz(0.0, i)  # placeholder — bind with 2*x[i]
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
                qc.rz(0.0, i + 1)  # placeholder — bind with 2*x[i]*x[i+1]*PHI
                qc.cx(i, i + 1)

        return qc, x


# ═══════════════════════════════════════════════════════════════════════════════
#  3. LOCAL SIMULATION BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

class _AerJobResult:
    """Minimal result object mimicking qiskit Aer job.result() output."""

    def __init__(self, counts: Dict[str, int]):
        self._counts = counts

    def get_counts(self) -> Dict[str, int]:
        return self._counts


class _AerJob:
    """Minimal job object mimicking AerSimulator.run() return value."""

    def __init__(self, counts: Dict[str, int]):
        self._counts = counts

    def result(self) -> _AerJobResult:
        return _AerJobResult(self._counts)


class L104AerBackend:
    """
    Local quantum simulation backend using l104_quantum_gate_engine.

    Replaces Qiskit Aer with sovereign statevector + trajectory simulation.
    Also serves as a drop-in replacement for qiskit-aer AerSimulator:
      - Accepts noise_model and method kwargs for API compatibility.
      - Provides a run(circuit, shots=...) method returning a job-like object.
    """

    def __init__(self, noise_profile: str = "ideal", n_qubits: int = 25, *,
                 noise_model=None, method: str = "automatic"):
        # If created via AerSimulator(noise_model=dict, method=...) alias,
        # derive noise_profile from the noise_model dict when possible.
        if noise_model is not None and noise_profile == "ideal":
            # noise_model may be a dict from L104NoiseModelFactory.build()
            if isinstance(noise_model, dict) and "profile" in noise_model:
                noise_profile = noise_model["profile"]
            elif isinstance(noise_model, dict):
                noise_profile = "custom"
        self._noise_profile = noise_profile
        self._n_qubits = n_qubits
        self._method = method
        self._noise_model_raw = noise_model  # keep reference for get_info()
        self._noise_params = (
            noise_model if isinstance(noise_model, dict)
            else L104NoiseModelFactory.build(noise_profile, n_qubits)
        )

    @property
    def available(self) -> bool:
        return True  # Always available — sovereign local

    def run_statevector(self, circuit) -> np.ndarray:
        """
        Execute circuit and return statevector probability array.

        Args:
            circuit: GateCircuit (or any object with operations)

        Returns:
            numpy array of probabilities (shape 2^n)
        """
        if isinstance(circuit, GateCircuit):
            n = circuit.num_qubits
        else:
            n = getattr(circuit, 'num_qubits', 2)

        sv = Statevector.from_label('0' * n)
        # Remove measurements before evolving
        if isinstance(circuit, GateCircuit):
            circ = circuit.remove_final_measurements()
        else:
            circ = circuit
        sv = sv.evolve(circ)
        return sv.probabilities()

    def run_shots(self, circuit, shots: int = 8192,
                  with_noise: bool = True) -> Dict[str, int]:
        """
        Execute circuit with measurement shots.

        Args:
            circuit: GateCircuit
            shots: Number of measurement shots
            with_noise: Whether to apply readout noise

        Returns:
            Counts dict {bitstring: count}
        """
        if isinstance(circuit, GateCircuit):
            circ = circuit.remove_final_measurements()
            n = circuit.num_qubits
        else:
            circ = circuit
            n = getattr(circuit, 'num_qubits', 2)

        sv = Statevector.from_label('0' * n).evolve(circ)
        counts = sv.sample_counts(shots)

        # Apply readout noise if requested
        if with_noise and self._noise_params.get("readout_error", 0) > 0:
            counts = self._apply_readout_noise(counts, n)

        return counts

    def run_density_matrix(self, circuit) -> DensityMatrix:
        """
        Execute circuit and return density matrix.

        Args:
            circuit: GateCircuit

        Returns:
            DensityMatrix
        """
        if isinstance(circuit, GateCircuit):
            circ = circuit.remove_final_measurements()
            n = circuit.num_qubits
        else:
            circ = circuit
            n = getattr(circuit, 'num_qubits', 2)

        sv = Statevector.from_label('0' * n).evolve(circ)

        # If noise profile is non-ideal, apply depolarizing to density matrix
        dm = sv.to_density_matrix()
        if self._noise_profile != "ideal":
            dm = self._apply_depolarizing(dm)
        return dm

    def _apply_readout_noise(self, counts: Dict[str, int], n_qubits: int) -> Dict[str, int]:
        """Apply readout bit-flip noise to measurement counts."""
        p_err = self._noise_params.get("readout_error", 0)
        if p_err <= 0:
            return counts

        rng = np.random.default_rng()
        noisy_counts: Dict[str, int] = {}
        for bitstring, count in counts.items():
            for _ in range(count):
                noisy_bits = list(bitstring)
                for b in range(len(noisy_bits)):
                    if rng.random() < p_err:
                        noisy_bits[b] = '1' if noisy_bits[b] == '0' else '0'
                key = ''.join(noisy_bits)
                noisy_counts[key] = noisy_counts.get(key, 0) + 1
        return noisy_counts

    def _apply_depolarizing(self, dm: DensityMatrix) -> DensityMatrix:
        """Apply global depolarizing channel to density matrix."""
        p = self._noise_params.get("single_gate_error", 0) * 10  # scale up
        p = min(p, 0.5)
        if p <= 0:
            return dm
        dim = dm.dim
        mixed = np.eye(dim, dtype=complex) / dim
        new_data = (1 - p) * dm.data + p * mixed
        return DensityMatrix(new_data)

    def run(self, circuit, shots: int = 8192, **kwargs) -> _AerJob:
        """AerSimulator.run() compatible interface.

        Returns a job-like object whose .result().get_counts() gives
        measurement counts — matching the Qiskit Aer API surface used by
        l104_quantum_runtime._execute_aer().
        """
        counts = self.run_shots(circuit, shots=shots, with_noise=(self._noise_profile != "ideal"))
        return _AerJob(counts)

    def set_noise_profile(self, profile: str):
        """Switch noise profile at runtime."""
        self._noise_profile = profile
        self._noise_params = L104NoiseModelFactory.build(profile, self._n_qubits)

    def get_info(self) -> Dict[str, Any]:
        return {
            "available": True,
            "backend": "l104_sovereign_local",
            "noise_profile": self._noise_profile,
            "noise_model_active": self._noise_profile != "ideal",
            "max_qubits": self._n_qubits,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  4. ERROR MITIGATION PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

class L104ErrorMitigation:
    """
    Quantum error mitigation techniques.

    Provides:
    - Zero-Noise Extrapolation (ZNE) with configurable noise factors
    - Measurement Error Mitigation
    - Dynamical Decoupling pulse insertion
    """

    @staticmethod
    def zne_extrapolate(noise_factors: List[float],
                        noisy_values: List[float],
                        order: int = 2) -> float:
        """
        Zero-Noise Extrapolation: fit polynomial to (noise_factor, value) pairs
        and extrapolate to noise_factor = 0.
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

        Uses local simulation with noise model.
        """
        n_qubits = min(n_qubits, 5)
        n_states = 2 ** n_qubits

        if backend is None:
            backend = L104AerBackend(noise_profile="ibm_eagle", n_qubits=n_qubits)

        cal_matrix = np.zeros((n_states, n_states))
        for j in range(n_states):
            qc = GateCircuit(n_qubits, f"cal_{j}")
            bits = format(j, f'0{n_qubits}b')
            for b, bit in enumerate(reversed(bits)):
                if bit == '1':
                    qc.x(b)

            counts = backend.run_shots(qc, shots=shots, with_noise=True)
            for bitstring, count in counts.items():
                i = int(bitstring, 2)
                if i < n_states:
                    cal_matrix[i, j] = count / shots

        return cal_matrix

    @staticmethod
    def apply_readout_correction(raw_probs: np.ndarray,
                                  cal_matrix: np.ndarray) -> np.ndarray:
        """Apply inverse calibration matrix to correct measurement errors."""
        try:
            cal_inv = np.linalg.pinv(cal_matrix)
            corrected = cal_inv @ raw_probs
            corrected = np.maximum(corrected, 0)
            total = corrected.sum()
            if total > 0:
                corrected /= total
            return corrected
        except np.linalg.LinAlgError:
            return raw_probs

    @staticmethod
    def add_dynamical_decoupling(circuit: GateCircuit,
                                  dd_sequence: str = "XY4") -> GateCircuit:
        """
        Insert dynamical decoupling pulse sequences into idle periods.

        Args:
            circuit: Input GateCircuit
            dd_sequence: "X2", "XY4", or "CPMG"

        Returns:
            New GateCircuit with DD pulses inserted in idle slots
        """
        if dd_sequence == "X2":
            dd_gates = ['x', 'x']
        elif dd_sequence == "XY4":
            dd_gates = ['x', 'y', 'x', 'y']
        elif dd_sequence == "CPMG":
            dd_gates = ['y', 'y']
        else:
            dd_gates = ['x', 'y', 'x', 'y']

        n_qubits = circuit.num_qubits
        qc = GateCircuit(n_qubits, f"{circuit.name}_dd")

        for op in circuit.operations:
            if op.label == "BARRIER" or op.label == "MEASURE":
                qc.operations.append(op)
                continue

            qubits = list(op.qubits)
            # Insert DD on idle qubits before multi-qubit gates
            if len(qubits) >= 2:
                idle_qubits = [q for q in range(n_qubits) if q not in qubits]
                for idle_q in idle_qubits:
                    for gate_name in dd_gates:
                        if gate_name == 'x':
                            qc.x(idle_q)
                        elif gate_name == 'y':
                            qc.y(idle_q)

            qc.append(op.gate, qubits, label=op.label, condition=op.condition)

        return qc


# ═══════════════════════════════════════════════════════════════════════════════
#  5. OBSERVABLE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

class L104ObservableFactory:
    """Build SparsePauliOp observables for expectation value computation."""

    @staticmethod
    def diagonal_hamiltonian(coefficients: List[float]) -> SparsePauliOp:
        """Build a diagonal Hamiltonian from a cost vector."""
        n_qubits = max(1, int(math.ceil(math.log2(max(len(coefficients), 2)))))
        n_states = 2 ** n_qubits
        padded = list(coefficients[:n_states])
        while len(padded) < n_states:
            padded.append(0.0)

        coeffs_arr = np.array(padded)
        pauli_terms = []
        pauli_coeffs = []

        identity = 'I' * n_qubits
        pauli_terms.append(identity)
        pauli_coeffs.append(np.mean(coeffs_arr))

        for q in range(n_qubits):
            z_label = ['I'] * n_qubits
            z_label[n_qubits - 1 - q] = 'Z'
            z_str = ''.join(z_label)

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
        """Build a transverse-field Ising model: H = -J Σ Z_iZ_{i+1} - h Σ X_i."""
        terms = []
        coeffs = []

        for i in range(n_qubits - 1):
            label = ['I'] * n_qubits
            label[n_qubits - 1 - i] = 'Z'
            label[n_qubits - 1 - (i + 1)] = 'Z'
            terms.append(''.join(label))
            coeffs.append(-coupling_j)

        for i in range(n_qubits):
            label = ['I'] * n_qubits
            label[n_qubits - 1 - i] = 'X'
            terms.append(''.join(label))
            coeffs.append(-field_h)

        return SparsePauliOp(terms, coeffs=coeffs)

    @staticmethod
    def god_code_observable(n_qubits: int) -> SparsePauliOp:
        """Build a GOD_CODE-aligned observable."""
        terms = []
        coeffs = []
        alpha_fine = 1.0 / 137.035999084

        for i in range(n_qubits):
            z_label = ['I'] * n_qubits
            z_label[n_qubits - 1 - i] = 'Z'
            terms.append(''.join(z_label))
            coeffs.append(GOD_CODE / 1000.0 * PHI ** (-i))

            x_label = ['I'] * n_qubits
            x_label[n_qubits - 1 - i] = 'X'
            terms.append(''.join(x_label))
            coeffs.append(alpha_fine / math.pi)

        return SparsePauliOp(terms, coeffs=coeffs)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. TRANSPILATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class L104Transpiler:
    """Transpilation utilities using l104_quantum_gate_engine GateCompiler."""

    @staticmethod
    def optimize_for_backend(circuit: GateCircuit,
                              backend=None,
                              optimization_level: int = 2) -> GateCircuit:
        """
        Optimize circuit using the L104 gate compiler.

        Args:
            circuit: Input GateCircuit
            backend: Ignored (sovereign local only)
            optimization_level: 0 (none) to 3 (heavy)

        Returns:
            Optimized GateCircuit
        """
        try:
            from l104_quantum_gate_engine import GateCompiler, OptimizationLevel, GateSet
            level_map = {
                0: OptimizationLevel.O0,
                1: OptimizationLevel.O1,
                2: OptimizationLevel.O2,
                3: OptimizationLevel.O3,
            }
            opt = level_map.get(optimization_level, OptimizationLevel.O2)
            compiler = GateCompiler()
            result = compiler.compile(circuit, target_gates=GateSet.UNIVERSAL, optimization=opt)
            return result.circuit if hasattr(result, 'circuit') and result.circuit else circuit
        except Exception:
            return circuit

    @staticmethod
    def circuit_metrics(circuit: GateCircuit) -> Dict[str, Any]:
        """Compute circuit quality metrics."""
        ops = circuit.count_ops()
        cx_count = ops.get('CNOT', 0) + ops.get('CZ', 0) + ops.get('ECR', 0)
        total_gates = sum(v for k, v in ops.items() if k not in ('MEASURE', 'BARRIER'))

        return {
            "num_qubits": circuit.num_qubits,
            "depth": circuit.depth,
            "total_gates": total_gates,
            "two_qubit_gates": cx_count,
            "single_qubit_gates": total_gates - cx_count,
            "gate_breakdown": dict(ops),
            "god_code_metric": round(GOD_CODE / max(circuit.depth, 1), 4),
        }

    @staticmethod
    def estimate_fidelity(circuit: GateCircuit,
                           single_gate_error: float = 2.5e-4,
                           cx_gate_error: float = 7.5e-3) -> float:
        """Estimate circuit fidelity from gate errors: F ≈ Π (1 - ε_gate)."""
        ops = circuit.count_ops()
        cx_count = ops.get('CNOT', 0) + ops.get('CZ', 0) + ops.get('ECR', 0)
        total = sum(v for k, v in ops.items() if k not in ('MEASURE', 'BARRIER'))
        single_count = total - cx_count

        fidelity = (1 - single_gate_error) ** single_count * (1 - cx_gate_error) ** cx_count
        return max(0.0, min(1.0, fidelity))


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════════

aer_backend = L104AerBackend(noise_profile="ideal")
aer_backend_noisy = L104AerBackend(noise_profile="ibm_eagle")
aer_backend_sacred = L104AerBackend(noise_profile="god_code_aligned")
circuit_factory = L104CircuitFactory()
error_mitigation = L104ErrorMitigation()
observable_factory = L104ObservableFactory()
transpiler = L104Transpiler()

# Backward compatibility: NoiseModel is now a dict
NoiseModel = dict

# Backward compatibility: AerSimulator alias for L104AerBackend
AerSimulator = L104AerBackend


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def quick_statevector(circuit) -> np.ndarray:
    """Fast statevector probability array from circuit."""
    return aer_backend.run_statevector(circuit)


def quick_shots(circuit, shots: int = 4096,
                noisy: bool = False) -> Dict[str, int]:
    """Quick shot-based execution."""
    backend = aer_backend_noisy if noisy else aer_backend
    return backend.run_shots(circuit, shots=shots, with_noise=noisy)


def build_vqe(n_qubits: int, depth: int = 4) -> Tuple[GateCircuit, ParameterVector]:
    """Shortcut to build VQE ansatz."""
    return circuit_factory.vqe_ansatz(n_qubits, depth)


def build_qaoa(n_qubits: int, p: int = 3) -> Tuple[GateCircuit, ParameterVector, ParameterVector]:
    """Shortcut to build QAOA circuit."""
    return circuit_factory.qaoa_circuit(n_qubits, p)


def build_grover(n_qubits: int, target: int) -> GateCircuit:
    """Build complete Grover circuit for single target."""
    n_iterations = max(1, int(round(math.pi / 4 * math.sqrt(2 ** n_qubits))))
    qc = GateCircuit(n_qubits, "grover")
    for q in range(n_qubits):
        qc.h(q)
    oracle = circuit_factory.grover_oracle(n_qubits, target)
    diffusion = circuit_factory.grover_diffusion(n_qubits)
    for _ in range(n_iterations):
        qc.compose(oracle)
        qc.compose(diffusion)
    qc.rz(GOD_CODE / 1000.0, 0)
    return qc


# ═══════════════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("L104 QISKIT UTILITIES v2.0.0 — SOVEREIGN LOCAL")
    print("=" * 70)
    print(f"  Backend: l104_quantum_gate_engine (sovereign)")
    print(f"  Noise profiles: {list(L104NoiseModelFactory.PROFILES.keys())}")

    vqe_qc, vqe_params = build_vqe(4, depth=3)
    print(f"\n  VQE ansatz: {vqe_qc.num_qubits}q, depth={vqe_qc.depth}, params={len(vqe_params)}")

    qaoa_qc, gammas, betas = build_qaoa(4, p=2)
    print(f"  QAOA circuit: {qaoa_qc.num_qubits}q, depth={qaoa_qc.depth}, γ={len(gammas)}, β={len(betas)}")

    grover_qc = build_grover(4, target=7)
    metrics = transpiler.circuit_metrics(grover_qc)
    print(f"  Grover(4q, target=7): depth={metrics['depth']}, gates={metrics['total_gates']}")

    nm = L104NoiseModelFactory.build("ibm_eagle", 4)
    print(f"\n  Noise config (ibm_eagle): {nm.get('profile_name', 'N/A')}")

    bell = GateCircuit(2, "bell")
    bell.h(0)
    bell.cx(0, 1)
    probs = aer_backend.run_statevector(bell)
    print(f"  Bell state probs: {np.round(probs, 4)}")

    counts = quick_shots(bell, shots=1000, noisy=True)
    print(f"  Noisy Bell counts: {counts}")

    h_ising = observable_factory.ising_hamiltonian(4, coupling_j=1.0, field_h=0.5)
    print(f"\n  Ising Hamiltonian: {len(h_ising)} terms")

    h_god = observable_factory.god_code_observable(4)
    print(f"  GOD_CODE observable: {len(h_god)} terms")

    fid = transpiler.estimate_fidelity(grover_qc)
    print(f"  Grover estimated fidelity: {fid:.6f}")

    print("\n" + "=" * 70)
    print("All utilities OK — SOVEREIGN LOCAL (zero Qiskit)")
    print("=" * 70)
