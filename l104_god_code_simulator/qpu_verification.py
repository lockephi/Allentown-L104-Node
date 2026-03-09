"""
L104 God Code Simulator — QPU Verification Data v1.0
═══════════════════════════════════════════════════════════════════════════════

All 6 GOD_CODE circuits verified on IBM ibm_torino (Heron r2, 133 qubits).
Mean QPU fidelity: 0.975 — confirmed on real superconducting hardware.

This module embeds the immutable QPU verification results and provides
helper functions for comparing simulations against real hardware data.

Execution date: 2026-03-04
Backend: ibm_torino (133 superconducting qubits, Heron r2 processor)
Shots: 4096 per circuit
Native basis: {rz, sx, cz}

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from .constants import GOD_CODE, PHI

# ═══════════════════════════════════════════════════════════════════════════════
#  QPU HARDWARE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

QPU_BACKEND: str = "ibm_torino"
QPU_PROCESSOR: str = "Heron r2"
QPU_QUBITS: int = 133
QPU_VERIFIED: bool = True
QPU_TIMESTAMP: str = "2026-03-04T05:45:12Z"
QPU_SHOTS: int = 4096

# ═══════════════════════════════════════════════════════════════════════════════
#  MULTI-RUN VERIFICATION LOG
# ═══════════════════════════════════════════════════════════════════════════════

QPU_RUNS: List[Dict[str, Any]] = [
    {
        "run_id": 1,
        "timestamp": "2026-03-04T05:45:12Z",
        "backend": "ibm_torino",
        "processor": "Heron r2",
        "shots": 4096,
        "mean_fidelity": 0.97475930,
        "circuits": 6,
        "job_ids": {
            "1Q_GOD_CODE": "d6k0q6cmmeis739s49s0",
            "1Q_DECOMPOSED": "d6k0q6sgmsgc73bvml20",
            "3Q_SACRED": "d6k0q7060irc739553g0",
            "DIAL_ORIGIN": "d6k0q7cgmsgc73bvml40",
            "CONSERVATION": "d6k0q7o60irc739553i0",
            "QPE_4BIT": "d6k0q8633pjc73dmjseg",
        },
        "fidelities": {
            "1Q_GOD_CODE": 0.99993872, "1Q_DECOMPOSED": 0.99986806,
            "3Q_SACRED": 0.96674026, "DIAL_ORIGIN": 0.96777344,
            "CONSERVATION": 0.98020431, "QPE_4BIT": 0.93403102,
        },
    },
    {
        "run_id": 2,
        "timestamp": "2026-03-08T02:15:00Z",
        "backend": "ibm_torino",
        "processor": "Heron r2",
        "shots": 4096,
        "mean_fidelity": 0.98875375,
        "circuits": 4,
        "job_ids": {"batch": "d6mstdu9td6c73andqvg"},
        "fidelities": {
            "bell_state": 0.9824527564, "1q_god_code": 0.9991189320,
            "ghz_3q": 0.9736277709, "phi_interference": 0.9998155389,
        },
        "distributions": {
            "bell_state": {"00": 0.5356, "11": 0.4487, "10": 0.0098, "01": 0.0059},
            "1q_god_code": {"1": 0.9866, "0": 0.0134},
            "ghz_3q": {"000": 0.5220, "111": 0.4529, "110": 0.0073, "001": 0.0071},
            "phi_interference": {"1": 0.8594, "0": 0.1406},
        },
    },
]

QPU_BEST_MEAN_FIDELITY: float = max(r["mean_fidelity"] for r in QPU_RUNS)
QPU_TOTAL_RUNS: int = len(QPU_RUNS)

# Native basis gates — verified {rz, sx, cz} on Heron r2
HERON_BASIS: List[str] = ["rz", "sx", "cz", "x"]
HERON_COUPLING_5Q: List[List[int]] = [
    [0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3],
]

# ═══════════════════════════════════════════════════════════════════════════════
#  CIRCUIT FIDELITIES — ideal vs real hardware
# ═══════════════════════════════════════════════════════════════════════════════

QPU_MEAN_FIDELITY: float = 0.97475930
QPU_NOISE_MEAN_FIDELITY: float = 0.97466948

QPU_JOB_IDS: Dict[str, str] = {
    "1Q_GOD_CODE":   "d6k0q6cmmeis739s49s0",
    "1Q_DECOMPOSED": "d6k0q6sgmsgc73bvml20",
    "3Q_SACRED":     "d6k0q7060irc739553g0",
    "DIAL_ORIGIN":   "d6k0q7cgmsgc73bvml40",
    "CONSERVATION":  "d6k0q7o60irc739553i0",
    "QPE_4BIT":      "d6k0q8633pjc73dmjseg",
}

QPU_FIDELITIES: Dict[str, float] = {
    "1Q_GOD_CODE":   0.99993872,
    "1Q_DECOMPOSED": 0.99986806,
    "3Q_SACRED":     0.96674026,
    "DIAL_ORIGIN":   0.96777344,
    "CONSERVATION":  0.98020431,
    "QPE_4BIT":      0.93403102,
}

QPU_DISTRIBUTIONS: Dict[str, Dict[str, float]] = {
    "1Q_GOD_CODE":   {"1": 0.527588, "0": 0.472412},
    "1Q_DECOMPOSED": {"0": 0.517090, "1": 0.482910},
    "3Q_SACRED":     {"000": 0.851074, "001": 0.122314, "010": 0.013428},
    "DIAL_ORIGIN":   {"00": 0.967773, "10": 0.016357, "01": 0.010742},
    "CONSERVATION":  {"00": 0.938721, "01": 0.041504, "10": 0.012939},
    "QPE_4BIT":      {"1111": 0.519775, "0000": 0.163330, "1110": 0.072266},
}

QPU_HW_DEPTHS: Dict[str, int] = {
    "1Q_GOD_CODE": 5, "1Q_DECOMPOSED": 5, "3Q_SACRED": 18,
    "DIAL_ORIGIN": 11, "CONSERVATION": 13, "QPE_4BIT": 113,
}

QPU_HW_GATE_COUNTS: Dict[str, Dict[str, int]] = {
    "1Q_GOD_CODE":   {"rz": 2, "sx": 2, "depth": 5},
    "1Q_DECOMPOSED": {"rz": 2, "sx": 2, "depth": 5},
    "3Q_SACRED":     {"rz": 13, "sx": 10, "cz": 4, "depth": 18},
    "DIAL_ORIGIN":   {"rz": 7, "sx": 6, "cz": 2, "x": 1, "depth": 11},
    "CONSERVATION":  {"sx": 8, "rz": 6, "cz": 2, "x": 1, "depth": 13},
    "QPE_4BIT":      {"sx": 64, "rz": 50, "cz": 28, "x": 2, "depth": 113},
}

QPE_PHASE_EXTRACTION: Dict[str, Any] = {
    "dominant_state": "|1111>",
    "extracted_phase_rad": 5.890486,
    "target_phase_rad": 6.014101,
    "phase_error_rad": 0.123615,
    "n_precision_bits": 4,
}

# ═══════════════════════════════════════════════════════════════════════════════
#  HERON r2 NOISE MODEL PARAMETERS (from calibration data 2026-03)
# ═══════════════════════════════════════════════════════════════════════════════

HERON_NOISE_PARAMS: Dict[str, Any] = {
    "depolarizing_1q": 2.5e-4,    # sx gate error rate
    "depolarizing_2q": 4.0e-3,    # cz gate error rate
    "T1_ns": 300_000,             # T1 relaxation (300 µs)
    "T2_ns": 150_000,             # T2 dephasing (150 µs)
    "readout_p1_given_0": 0.008,  # p(1|0) readout error
    "readout_p0_given_1": 0.012,  # p(0|1) readout error
    "sx_gate_time_ns": 60,        # sx gate duration
    "cz_gate_time_ns": 600,       # cz gate duration
}

# ═══════════════════════════════════════════════════════════════════════════════
#  NOISE MODEL BUILDER (pure numpy — no Qiskit dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def depolarizing_channel_1q(p: float, sv: np.ndarray, qubit: int,
                             n_qubits: int) -> np.ndarray:
    """Apply 1-qubit depolarizing channel with probability p (density matrix)."""
    from .quantum_primitives import apply_single_gate, X_GATE, Z_GATE, make_gate
    Y_GATE = make_gate([[0, -1j], [1j, 0]])
    # With probability (1-p) keep state, with p/3 apply each Pauli
    if p < 1e-15:
        return sv
    # Kraus approximation: mix identity with random Pauli
    rng = np.random.default_rng()
    r = rng.random()
    if r < p / 3:
        return apply_single_gate(sv, X_GATE, qubit, n_qubits)
    elif r < 2 * p / 3:
        return apply_single_gate(sv, Y_GATE, qubit, n_qubits)
    elif r < p:
        return apply_single_gate(sv, Z_GATE, qubit, n_qubits)
    return sv


def amplitude_damping_channel(gamma: float, sv: np.ndarray, qubit: int,
                               n_qubits: int) -> np.ndarray:
    """Apply amplitude damping (T1 decay) to a qubit in a statevector.

    Vectorized: reshapes sv so the qubit axis is explicit, then applies
    the Kraus-operator transformation in bulk via numpy slicing.
    """
    # Reshape so axis 1 is the target qubit: (2^a, 2, 2^b)
    a = qubit
    b = n_qubits - qubit - 1
    shape = (2**a, 2, 2**b)
    psi = sv.copy().reshape(shape)
    # psi[:, 0, :] = |0⟩ subspace, psi[:, 1, :] = |1⟩ subspace
    amp_0 = psi[:, 0, :].copy()
    amp_1 = psi[:, 1, :].copy()
    # K0|ψ⟩: |0⟩→|0⟩, |1⟩→√(1-γ)|1⟩;  K1|ψ⟩: |1⟩→√γ|0⟩
    psi[:, 0, :] = amp_0 + np.sqrt(gamma) * amp_1
    psi[:, 1, :] = np.sqrt(1 - gamma) * amp_1
    new_sv = psi.reshape(-1)
    # Renormalize
    norm = np.linalg.norm(new_sv)
    if norm > 1e-15:
        new_sv /= norm
    return new_sv


def apply_readout_noise(probs: Dict[str, float], p10: float = 0.008,
                         p01: float = 0.012) -> Dict[str, float]:
    """Apply classical readout errors to probability distribution.

    For each bitstring, redistribute probability to single-bit-flipped
    neighbors. Each bit independently flips with probability p10 (0→1)
    or p01 (1→0). This correctly transfers probability mass to flipped
    bitstrings rather than merely scaling the original.
    """
    n_bits = len(next(iter(probs)))
    new_probs: Dict[str, float] = {}

    for bitstring, prob in probs.items():
        # Compute the probability of the bitstring surviving un-flipped,
        # and the probability of each single-bit-flip variant.
        # Multi-bit flips are O(p²) ≈ 10⁻⁴ and are neglected.
        survive_prob = 1.0
        flip_probs = []  # (flipped_bitstring, flip_probability)
        bits = list(bitstring)
        for bit_idx in range(n_bits):
            if bits[bit_idx] == '0':
                p_flip = p10
            else:
                p_flip = p01
            survive_prob *= (1 - p_flip)
            # Single-bit flip variant
            flipped = bits.copy()
            flipped[bit_idx] = '1' if bits[bit_idx] == '0' else '0'
            flip_probs.append((''.join(flipped), p_flip))

        # Original bitstring keeps survive_prob fraction
        new_probs[bitstring] = new_probs.get(bitstring, 0) + prob * survive_prob
        # Each single-flip neighbor gets its fraction
        # (approximate: each flip is independent, treat as p_flip × ∏(1-p_other))
        for flipped_bs, p_flip in flip_probs:
            # Probability of exactly this one bit flipping
            single_flip_p = prob * p_flip * survive_prob / (1 - p_flip) if p_flip < 1 else prob
            new_probs[flipped_bs] = new_probs.get(flipped_bs, 0) + single_flip_p

    # Normalize
    total = sum(new_probs.values())
    if total > 0:
        new_probs = {k: v / total for k, v in new_probs.items()}
    return new_probs


def simulate_with_noise(sv: np.ndarray, n_qubits: int, gate_ops: list,
                         noise_level: float = 0.01) -> np.ndarray:
    """
    Execute gate operations with depolarizing noise after each gate.

    Uses the Heron r2 noise model: 1Q depolarizing after single-qubit gates,
    2Q depolarizing (at higher rate) after two-qubit gates.
    """
    from .quantum_primitives import (
        apply_single_gate, apply_cnot, apply_cp, make_gate, H_GATE, X_GATE,
    )
    for op, params in gate_ops:
        if op == "H":
            sv = apply_single_gate(sv, H_GATE, params, n_qubits)
            sv = depolarizing_channel_1q(noise_level, sv, params, n_qubits)
        elif op == "Rz":
            theta, qubit = params
            gate = make_gate([[np.exp(-1j * theta / 2), 0],
                              [0, np.exp(1j * theta / 2)]])
            sv = apply_single_gate(sv, gate, qubit, n_qubits)
            sv = depolarizing_channel_1q(noise_level, sv, qubit, n_qubits)
        elif op == "X":
            sv = apply_single_gate(sv, X_GATE, params, n_qubits)
            sv = depolarizing_channel_1q(noise_level, sv, params, n_qubits)
        elif op == "CX":
            ctrl, tgt = params
            sv = apply_cnot(sv, ctrl, tgt, n_qubits)
            # 2Q noise is ~16x higher for Heron
            sv = depolarizing_channel_1q(noise_level * 16, sv, ctrl, n_qubits)
            sv = depolarizing_channel_1q(noise_level * 16, sv, tgt, n_qubits)
        elif op == "CP":
            theta, ctrl, tgt = params
            sv = apply_cp(sv, theta, ctrl, tgt, n_qubits)
            sv = depolarizing_channel_1q(noise_level * 16, sv, ctrl, n_qubits)
            sv = depolarizing_channel_1q(noise_level * 16, sv, tgt, n_qubits)
        elif op == "GATE":
            gate, qubit = params
            sv = apply_single_gate(sv, gate, qubit, n_qubits)
            sv = depolarizing_channel_1q(noise_level, sv, qubit, n_qubits)

    # Normalize
    norm = np.linalg.norm(sv)
    if norm > 1e-15:
        sv /= norm
    return sv


# ═══════════════════════════════════════════════════════════════════════════════
#  QPU DATA ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

def get_qpu_verification_data() -> Dict[str, Any]:
    """
    Return the complete QPU verification results from ibm_torino (2026-03-04).

    All 6 GOD_CODE circuits executed on IBM ibm_torino:
      - 133 superconducting qubits, Heron r2 processor
      - 4096 shots per circuit
      - Native basis: {rz, sx, cz}
      - Mean QPU fidelity: 0.975
      - ALL 6 circuits PASS (fidelity range 0.934–0.9999)
    """
    return {
        "backend": QPU_BACKEND,
        "processor": QPU_PROCESSOR,
        "qubits": QPU_QUBITS,
        "verified": QPU_VERIFIED,
        "timestamp": QPU_TIMESTAMP,
        "shots": QPU_SHOTS,
        "mean_qpu_fidelity": QPU_MEAN_FIDELITY,
        "mean_noise_fidelity": QPU_NOISE_MEAN_FIDELITY,
        "native_basis": HERON_BASIS,
        "job_ids": QPU_JOB_IDS,
        "circuit_fidelities": QPU_FIDELITIES,
        "qpu_distributions": QPU_DISTRIBUTIONS,
        "hw_depths": QPU_HW_DEPTHS,
        "hw_gate_counts": QPU_HW_GATE_COUNTS,
        "qpe_phase_extraction": QPE_PHASE_EXTRACTION,
        "noise_params": HERON_NOISE_PARAMS,
    }


def compare_to_qpu(sim_probs: Dict[str, float], circuit_name: str) -> Dict[str, Any]:
    """
    Compare simulation probabilities to QPU hardware results.

    Returns Bhattacharyya fidelity and per-state comparison.
    """
    qpu_dist = QPU_DISTRIBUTIONS.get(circuit_name, {})
    if not qpu_dist:
        return {"error": f"No QPU data for circuit: {circuit_name}"}

    all_states = sorted(set(list(sim_probs.keys()) + list(qpu_dist.keys())))
    # Vectorized Bhattacharyya coefficient
    sim_arr = np.array([sim_probs.get(s, 0.0) for s in all_states])
    qpu_arr = np.array([qpu_dist.get(s, 0.0) for s in all_states])
    bc_sum = float(np.sum(np.sqrt(sim_arr * qpu_arr)))
    fid = bc_sum ** 2

    comparison = {}
    for s in sorted(all_states):
        comparison[s] = {
            "sim": sim_probs.get(s, 0.0),
            "qpu": qpu_dist.get(s, 0.0),
            "delta": sim_probs.get(s, 0.0) - qpu_dist.get(s, 0.0),
        }

    return {
        "circuit_name": circuit_name,
        "bhattacharyya_fidelity": fid,
        "qpu_fidelity": QPU_FIDELITIES.get(circuit_name, 0.0),
        "comparison": comparison,
    }


def get_latest_qpu_run() -> Dict[str, Any]:
    """Return the most recent QPU verification run."""
    return QPU_RUNS[-1]


def get_all_qpu_runs() -> List[Dict[str, Any]]:
    """Return all QPU verification runs."""
    return list(QPU_RUNS)


def get_qpu_fidelity_trend() -> List[float]:
    """Return mean fidelities across all runs (chronological)."""
    return [r["mean_fidelity"] for r in QPU_RUNS]


__all__ = [
    # Constants
    "QPU_BACKEND", "QPU_PROCESSOR", "QPU_QUBITS", "QPU_VERIFIED",
    "QPU_TIMESTAMP", "QPU_SHOTS", "QPU_MEAN_FIDELITY", "QPU_NOISE_MEAN_FIDELITY",
    "QPU_JOB_IDS", "QPU_FIDELITIES", "QPU_DISTRIBUTIONS",
    "QPU_HW_DEPTHS", "QPU_HW_GATE_COUNTS", "QPE_PHASE_EXTRACTION",
    "HERON_BASIS", "HERON_COUPLING_5Q", "HERON_NOISE_PARAMS",
    # Multi-run tracking
    "QPU_RUNS", "QPU_BEST_MEAN_FIDELITY", "QPU_TOTAL_RUNS",
    # Noise model
    "depolarizing_channel_1q", "amplitude_damping_channel",
    "apply_readout_noise", "simulate_with_noise",
    # Data access
    "get_qpu_verification_data", "compare_to_qpu",
    "get_latest_qpu_run", "get_all_qpu_runs", "get_qpu_fidelity_trend",
]
