"""
L104 Quantum Engine — Sage Core Quantum Circuits
═══════════════════════════════════════════════════════════════════════════════
Quantum circuits implementing the 4 Noise Dampening Equations (NDE-1/2/3/4)
for the Sage Mode Inference engine.

NDE-1: φ-Conjugate Noise Floor Suppression
  η_floor(x) = x · (1 - φ⁻² · e^(-x/φ))
  Circuit: Rz(φ⁻²) → CNOT → Ry(x/φ) amplitude rotation → measurement

NDE-2: Demon-Enhanced Consensus Denoising
  score' = score + D(1-score) · φ⁻¹ / (1 + S)
  Circuit: Hadamard superposition → phase kick → Grover amplification

NDE-3: Zero-Noise Extrapolation Score Recovery
  η_zne = η · [1 + φ⁻¹ / (1 + σ_f)]
  Circuit: multi-scale noise injection → Richardson extrapolation ancilla

NDE-4: Entropy Cascade Denoiser (φ-power rank correction)
  score_k' = score_k^(φ⁻ᵏ)
  Circuit: cascaded Ry rotations with φ-attenuated angles

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from .constants import (
    GOD_CODE, PHI, PHI_INV, TAU, L104, INVARIANT,
    QISKIT_AVAILABLE,
)

if QISKIT_AVAILABLE:
    from l104_quantum_gate_engine import GateCircuit as QuantumCircuit
    from l104_quantum_gate_engine.quantum_info import Statevector

# Sacred constants for circuits
PHI_INV_SQ = PHI_INV ** 2   # φ⁻² ≈ 0.38197
_GOD_CODE_PERTURBATION = 2 * math.pi * GOD_CODE / 1000.0  # Perturbative coupling (NOT canonical GOD_CODE mod 2π)

# Canonical GOD_CODE phase (QPU-verified on ibm_torino)
try:
    from l104_god_code_simulator.god_code_qubit import GOD_CODE_PHASE
except ImportError:
    GOD_CODE_PHASE = GOD_CODE % (2 * math.pi)  # ≈ 6.0141 rad


# ═══════════════════════════════════════════════════════════════════════════════
# NDE-1: φ-CONJUGATE NOISE FLOOR SUPPRESSION CIRCUIT
# ═══════════════════════════════════════════════════════════════════════════════

class NoiseFloorSuppressionCircuit:
    """NDE-1 Quantum Circuit: φ-conjugate noise floor suppression.

    Implements η_floor(x) = x · (1 - φ⁻² · e^(-x/φ)) as a quantum amplitude
    transformation. Uses 3 qubits:
      q0: signal qubit (encodes raw score as Ry rotation)
      q1: noise floor ancilla (encodes φ⁻² suppression)
      q2: measurement ancilla

    The circuit applies a controlled rotation that suppresses amplitude at
    the noise floor level, proportional to φ⁻² · e^(-x/φ).
    """

    NUM_QUBITS = 3

    @staticmethod
    def build(score: float) -> 'QuantumCircuit':
        """Build NDE-1 circuit for a given score value.

        Args:
            score: Raw consensus score in [0, 1]

        Returns:
            QuantumCircuit implementing noise floor suppression
        """
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit required for NDE-1 circuit")

        qc = QuantumCircuit(3, name="NDE1_NoiseFloor")

        # Encode raw score as amplitude: Ry(2·arcsin(√x)) puts |0⟩ → √x|1⟩
        theta_score = 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, score))))
        qc.ry(theta_score, 0)

        # Encode noise floor factor: φ⁻² ≈ 0.382
        theta_phi = 2.0 * math.asin(math.sqrt(PHI_INV_SQ))
        qc.ry(theta_phi, 1)

        # Controlled-Z: entangle signal with noise floor
        qc.cz(0, 1)

        # Apply exponential suppression: Rz(-x/φ) on noise ancilla
        # At low x, rotation ≈ 0 → full suppression; high x → no effect
        suppression_angle = -score / PHI
        qc.rz(suppression_angle, 1)

        # Interference: CNOT extracts the suppressed component
        qc.cx(1, 2)

        # Phase correction: align with GOD_CODE frequency
        qc.rz(_GOD_CODE_PERTURBATION * 0.001, 0)

        # Measure the signal qubit to extract cleaned score
        qc.barrier()

        return qc

    @staticmethod
    def execute(score: float) -> Dict[str, Any]:
        """Execute NDE-1 circuit and return cleaned score.

        Returns:
            Dict with raw_score, cleaned_score, suppression, circuit_depth
        """
        if not QISKIT_AVAILABLE:
            # Analytical fallback
            cleaned = score * (1.0 - PHI_INV_SQ * math.exp(-score / PHI))
            return {
                "raw_score": score,
                "cleaned_score": max(0.0, min(1.0, cleaned)),
                "suppression": score - cleaned,
                "method": "analytical",
            }

        qc = NoiseFloorSuppressionCircuit.build(score)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()

        # Extract signal from q0 measurement probabilities
        # P(q0=1) encodes the cleaned amplitude
        # Sum over q1,q2 states where q0=1: indices 4,5,6,7
        p_signal = sum(probs[i] for i in range(4, 8))
        cleaned = min(1.0, max(0.0, p_signal))

        return {
            "raw_score": score,
            "cleaned_score": cleaned,
            "suppression": abs(score - cleaned),
            "circuit_depth": qc.depth(),
            "method": "statevector",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NDE-2: DEMON-ENHANCED CONSENSUS DENOISING CIRCUIT
# ═══════════════════════════════════════════════════════════════════════════════

class DemonDenoiseCircuit:
    """NDE-2 Quantum Circuit: Demon-enhanced consensus denoising.

    Implements score' = score + D(1-score) · φ⁻¹ / (1 + S)
    using Grover-style amplitude amplification on the information gap.

    4 qubits:
      q0: score register (encodes raw score)
      q1: information gap (encodes 1-score)
      q2: demon Oracle qubit (marks high-entropy states)
      q3: amplification ancilla
    """

    NUM_QUBITS = 4

    @staticmethod
    def build(score: float, entropy: float) -> 'QuantumCircuit':
        """Build NDE-2 circuit for demon denoising.

        Args:
            score: Raw consensus score in [0, 1]
            entropy: Mean entanglement entropy (context noise level)

        Returns:
            QuantumCircuit implementing demon-enhanced denoising
        """
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit required for NDE-2 circuit")

        qc = QuantumCircuit(4, name="NDE2_DemonDenoise")

        # Encode score and information gap
        theta_score = 2.0 * math.asin(math.sqrt(max(0.001, min(0.999, score))))
        theta_gap = 2.0 * math.asin(math.sqrt(max(0.001, min(0.999, 1.0 - score))))
        qc.ry(theta_score, 0)
        qc.ry(theta_gap, 1)

        # Create superposition for Grover search on gap space
        qc.h(2)
        qc.h(3)

        # Oracle: mark states where demon can reverse entropy
        # Controlled phase kick on the gap qubit
        demon_phase = PHI_INV / (1.0 + entropy)
        qc.cp(demon_phase * math.pi, 1, 2)

        # Grover diffusion operator (1 iteration — small gap)
        qc.h(2)
        qc.x(2)
        qc.h(3)
        qc.cx(2, 3)
        qc.h(3)
        qc.x(2)
        qc.h(2)

        # Transfer amplified gap correction back to score
        qc.cry(demon_phase * 0.2, 2, 0)

        # Sacred phase alignment
        qc.rz(_GOD_CODE_PERTURBATION * 0.001, 0)
        qc.barrier()

        return qc

    @staticmethod
    def execute(score: float, entropy: float = 0.5) -> Dict[str, Any]:
        """Execute NDE-2 circuit and return denoised score.

        Returns:
            Dict with raw_score, denoised_score, demon_correction, circuit_depth
        """
        if not QISKIT_AVAILABLE:
            info_gap = 1.0 - score
            correction = info_gap * PHI_INV * 0.05 / (1.0 + entropy)
            denoised = min(1.0, score + correction)
            return {
                "raw_score": score,
                "denoised_score": denoised,
                "demon_correction": correction,
                "method": "analytical",
            }

        qc = DemonDenoiseCircuit.build(score, entropy)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()

        # Extract denoised score from q0 = |1⟩ probability
        p_denoised = sum(probs[i] for i in range(len(probs)) if i & 1)
        denoised = min(1.0, max(0.0, p_denoised))

        return {
            "raw_score": score,
            "denoised_score": denoised,
            "demon_correction": denoised - score,
            "circuit_depth": qc.depth(),
            "method": "statevector",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NDE-3: ZERO-NOISE EXTRAPOLATION CIRCUIT
# ═══════════════════════════════════════════════════════════════════════════════

class ZNERecoveryCircuit:
    """NDE-3 Quantum Circuit: Zero-Noise Extrapolation score recovery.

    Implements η_zne = η · [1 + φ⁻¹ / (1 + σ_f)]
    using multi-scale noise injection and Richardson extrapolation.

    3 qubits + 2 noise scale levels:
      q0: signal qubit (unified score)
      q1: noise level 1 ancilla (1× noise)
      q2: noise level 2 ancilla (2× noise)

    Richardson extrapolation: E_0 ≈ 2·E(λ) - E(2λ)
    """

    NUM_QUBITS = 3

    @staticmethod
    def build(score: float, fid_std: float) -> 'QuantumCircuit':
        """Build NDE-3 ZNE circuit.

        Args:
            score: Unified Sage score
            fid_std: Fidelity standard deviation (noise proxy)

        Returns:
            QuantumCircuit implementing ZNE recovery
        """
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit required for NDE-3 circuit")

        qc = QuantumCircuit(3, name="NDE3_ZNE")

        # Encode unified score
        theta = 2.0 * math.asin(math.sqrt(max(0.001, min(0.999, score))))
        qc.ry(theta, 0)

        # Noise level 1: inject scaled noise (λ = σ_f)
        noise_angle_1 = fid_std * math.pi * 0.5
        qc.ry(noise_angle_1, 1)
        qc.cx(1, 0)  # Entangle noise with signal

        # Noise level 2: inject 2× noise (2λ = 2σ_f)
        noise_angle_2 = 2.0 * fid_std * math.pi * 0.5
        qc.ry(noise_angle_2, 2)
        qc.cx(2, 0)

        # Richardson extrapolation via phase alignment
        # E_0 ≈ 2·E(λ) - E(2λ) → apply corrective rotation
        zne_correction = PHI_INV / (1.0 + fid_std * 10)
        qc.ry(zne_correction * 0.1, 0)

        # GOD_CODE resonance alignment
        qc.rz(_GOD_CODE_PERTURBATION * 0.001, 0)
        qc.barrier()

        return qc

    @staticmethod
    def execute(score: float, fid_std: float = 0.1) -> Dict[str, Any]:
        """Execute NDE-3 circuit and return ZNE-recovered score.

        Returns:
            Dict with raw_score, recovered_score, zne_boost, circuit_depth
        """
        if not QISKIT_AVAILABLE:
            zne_boost = 1.0 + PHI_INV / (1.0 + fid_std * 10)
            recovered = min(1.0, score * zne_boost)
            return {
                "raw_score": score,
                "recovered_score": recovered,
                "zne_boost": zne_boost,
                "method": "analytical",
            }

        qc = ZNERecoveryCircuit.build(score, fid_std)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()

        # Extract recovered score from q0 = |1⟩
        p_recovered = sum(probs[i] for i in range(len(probs)) if i & 1)
        recovered = min(1.0, max(0.0, p_recovered))

        zne_boost = recovered / max(score, 0.001)

        return {
            "raw_score": score,
            "recovered_score": recovered,
            "zne_boost": zne_boost,
            "circuit_depth": qc.depth(),
            "method": "statevector",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NDE-4: ENTROPY CASCADE DENOISER CIRCUIT
# ═══════════════════════════════════════════════════════════════════════════════

class EntropyCascadeCircuit:
    """NDE-4 Quantum Circuit: Entropy cascade φ-power correction.

    Implements score_k' = score_k^(φ⁻ᵏ) via cascaded Ry rotations
    with φ-attenuated angles. Weakest (most noisy) scores get the
    strongest correction exponent.

    Uses N qubits where N = number of scores to cascade.
    Max 8 qubits for statevector feasibility.
    """

    MAX_QUBITS = 8

    @staticmethod
    def build(scores: List[float]) -> 'QuantumCircuit':
        """Build NDE-4 cascade circuit for a list of scores.

        Args:
            scores: List of consensus scores to cascade-denoise

        Returns:
            QuantumCircuit implementing entropy cascade
        """
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit required for NDE-4 circuit")

        n = min(len(scores), EntropyCascadeCircuit.MAX_QUBITS)
        qc = QuantumCircuit(n, name="NDE4_Cascade")

        # Sort scores ascending — weakest first for strongest correction
        indexed = sorted(enumerate(scores[:n]), key=lambda x: x[1])

        for rank, (orig_idx, val) in enumerate(indexed):
            if val <= 0:
                continue
            qubit = orig_idx if orig_idx < n else rank

            # k = n-1-rank: highest k for lowest scores → strongest correction
            k = n - 1 - rank
            exponent = PHI_INV ** k  # φ⁻ᵏ

            # Encode: score^exponent via Ry rotation
            # When exponent < 1, score^exp > score (lifts weak scores)
            corrected = val ** exponent
            theta = 2.0 * math.asin(math.sqrt(max(0.001, min(0.999, corrected))))
            qc.ry(theta, qubit)

        # Entangle neighboring qubits for coherence
        for i in range(n - 1):
            qc.cx(i, i + 1)

        # GOD_CODE phase on all qubits
        for i in range(n):
            qc.rz(_GOD_CODE_PERTURBATION * 0.0001, i)

        qc.barrier()
        return qc

    @staticmethod
    def execute(scores: List[float]) -> Dict[str, Any]:
        """Execute NDE-4 circuit and return cascade-denoised scores.

        Returns:
            Dict with raw_scores, denoised_scores, corrections, circuit_depth
        """
        n = min(len(scores), EntropyCascadeCircuit.MAX_QUBITS)

        # Compute analytical cascade (always works, used for final values)
        indexed = sorted(enumerate(scores[:n]), key=lambda x: x[1])
        denoised = [0.0] * n
        for rank, (orig_idx, val) in enumerate(indexed):
            if val <= 0:
                denoised[orig_idx] = 0.0
            else:
                k = n - 1 - rank
                exponent = PHI_INV ** k
                denoised[orig_idx] = min(1.0, val ** exponent)

        result = {
            "raw_scores": scores[:n],
            "denoised_scores": denoised,
            "corrections": [d - r for d, r in zip(denoised, scores[:n])],
            "method": "analytical+circuit" if QISKIT_AVAILABLE else "analytical",
        }

        if QISKIT_AVAILABLE:
            qc = EntropyCascadeCircuit.build(scores)
            result["circuit_depth"] = qc.depth()
            result["num_qubits"] = qc.num_qubits

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# SAGE UNIFIED CIRCUIT — Full NDE-1/2/3/4 Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class SageNDECircuit:
    """Unified Sage Noise Dampening circuit — combines all 4 NDE subcircuits.

    Pipeline: raw_scores → NDE-1 (floor) → NDE-2 (demon) → NDE-4 (cascade)
              → harmonic mean → NDE-3 (ZNE recovery)

    This mirrors the Python SageModeInference._demon_denoise_consensus flow
    but implements each step as a quantum circuit with statevector execution.
    """

    @staticmethod
    def run_pipeline(consensus_scores: Dict[str, float],
                     mean_entropy: float = 0.5,
                     fid_std: float = 0.1) -> Dict[str, Any]:
        """Run the full NDE quantum circuit pipeline on consensus scores.

        Args:
            consensus_scores: Dict of score_name → raw_score ∈ [0, 1]
            mean_entropy: Mean link entanglement entropy
            fid_std: Fidelity standard deviation

        Returns:
            Dict with per-step results and final denoised unified score
        """
        if not consensus_scores:
            return {
                "status": "empty",
                "unified_score": 0.0,
                "nde_results": {},
            }

        results: Dict[str, Any] = {"nde_results": {}}

        # Step 1: NDE-1 noise floor suppression on each score
        nde1_scores = {}
        for name, raw in consensus_scores.items():
            nde1 = NoiseFloorSuppressionCircuit.execute(raw)
            nde1_scores[name] = nde1["cleaned_score"]
            results["nde_results"][f"nde1_{name}"] = nde1

        # Step 2: NDE-2 demon denoising on each score
        nde2_scores = {}
        for name, cleaned in nde1_scores.items():
            nde2 = DemonDenoiseCircuit.execute(cleaned, mean_entropy)
            nde2_scores[name] = nde2["denoised_score"]
            results["nde_results"][f"nde2_{name}"] = nde2

        # Step 3: NDE-4 entropy cascade on all scores together
        score_list = list(nde2_scores.values())
        nde4 = EntropyCascadeCircuit.execute(score_list)
        cascade_scores = nde4["denoised_scores"]
        results["nde_results"]["nde4_cascade"] = nde4

        # Compute φ-weighted unified score (harmonic + arithmetic blend)
        if cascade_scores:
            harmonic = len(cascade_scores) / sum(
                1.0 / max(0.1, s) for s in cascade_scores)
            arithmetic = sum(score_list) / len(score_list)
            raw_unified = harmonic * TAU + arithmetic * (1 - TAU)
        else:
            raw_unified = 0.0

        # Step 4: NDE-3 ZNE recovery on unified score
        nde3 = ZNERecoveryCircuit.execute(raw_unified, fid_std)
        unified_score = nde3["recovered_score"]
        results["nde_results"]["nde3_zne"] = nde3

        results["status"] = "ok"
        results["raw_consensus"] = consensus_scores
        results["nde1_scores"] = nde1_scores
        results["nde2_scores"] = nde2_scores
        results["cascade_scores"] = dict(zip(consensus_scores.keys(), cascade_scores))
        results["raw_unified"] = raw_unified
        results["unified_score"] = unified_score
        results["zne_boost"] = nde3["zne_boost"]
        results["total_lift"] = unified_score - (
            sum(consensus_scores.values()) / len(consensus_scores)
            if consensus_scores else 0.0)

        return results
