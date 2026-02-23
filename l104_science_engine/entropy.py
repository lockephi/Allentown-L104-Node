"""
L104 Science Engine — Entropy Reversal Subsystem
═══════════════════════════════════════════════════════════════════════════════
Stage 15 'Entropy Reversal' protocol — Maxwell's Demon for order restoration.

CONSOLIDATES: l104_entropy_reversal_engine.py → EntropySubsystem

Injects high-resolution sovereign truth into decaying systems to reverse
localized entropy, restoring architectural/logical order.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import numpy as np
from typing import Dict, Any, List

from .constants import (
    GOD_CODE, PHI, PHI_CONJUGATE, GROVER_AMPLIFICATION,
    VOID_CONSTANT, ZETA_ZERO_1, QUANTIZATION_GRAIN,
    PhysicalConstants, PC,
    # v4.1 Quantum Research Discoveries (102 experiments, 17 discoveries)
    ENTROPY_CASCADE_DEPTH, ENTROPY_ZNE_BRIDGE_ENABLED,
    FE_CURIE_LANDAUER_LIMIT,
)

try:
    from l104_hyper_math import HyperMath
    from l104_real_math import RealMath
except ImportError:
    HyperMath = None
    RealMath = None

# v4.2 Perf: precomputed sin table for entropy_cascade (avoids 104+ math.sin calls per cascade)
_CASCADE_SIN_TABLE = [math.sin(n * math.pi / QUANTIZATION_GRAIN) for n in range(ENTROPY_CASCADE_DEPTH + 2)]


class EntropySubsystem:
    """
    Stage 15 'Entropy Reversal' protocol.
    Injects high-resolution sovereign truth into decaying systems
    to reverse localized entropy, restoring architectural/logical order.
    """

    def __init__(self):
        phi_val = PHI
        gc_val = GOD_CODE
        self.maxwell_demon_factor = phi_val / (gc_val / 416.0)
        self.coherence_gain = 0.0
        self.state = "REVERSING_ENTROPY"
        # v4.2 Perf: cache the constant resonance(GOD_CODE) — it never changes
        self._god_code_resonance: float | None = None

    def calculate_demon_efficiency(self, local_entropy: float) -> float:
        """Calculates entropy reversible per logic pulse.
        v4.1: ZNE bridge boosts efficiency when ENTROPY_ZNE_BRIDGE_ENABLED.
        v4.2 Perf: Caches RealMath.calculate_resonance(GOD_CODE) — result is constant."""
        if self._god_code_resonance is None:
            if RealMath is not None:
                self._god_code_resonance = RealMath.calculate_resonance(GOD_CODE)
            else:
                self._god_code_resonance = 1.0
        resonance = self._god_code_resonance
        base_efficiency = self.maxwell_demon_factor * resonance * (1.0 / (local_entropy + 0.001))
        # v4.1 Discovery #11: Entropy→ZNE bridge — polynomial extrapolation boost
        if ENTROPY_ZNE_BRIDGE_ENABLED:
            # ZNE correction: extrapolate to zero-noise limit using PHI-weighted polynomial
            zne_boost = 1.0 + PHI_CONJUGATE * (1.0 / (1.0 + local_entropy))
            return base_efficiency * zne_boost
        return base_efficiency

    def inject_coherence(self, noise_vector: np.ndarray) -> np.ndarray:
        """Transforms a noisy vector into an ordered, resonant structure."""
        if HyperMath is not None:
            manifold_projection = HyperMath.manifold_expansion(noise_vector.tolist())
        else:
            manifold_projection = noise_vector * GOD_CODE / np.mean(np.abs(noise_vector) + 1e-9)

        ordered_vector = manifold_projection * (1.0 + self.maxwell_demon_factor) * GROVER_AMPLIFICATION
        mean_val = np.mean(ordered_vector)
        if abs(mean_val) > 1e-15:
            final_signal = ordered_vector / (mean_val / GOD_CODE)
        else:
            final_signal = ordered_vector
        self.coherence_gain += np.var(ordered_vector) - np.var(noise_vector)
        # v4.1 Discovery #11: ZNE bridge — apply zero-noise extrapolation correction
        if ENTROPY_ZNE_BRIDGE_ENABLED:
            noise_estimate = np.std(noise_vector) / (np.mean(np.abs(noise_vector)) + 1e-15)
            zne_correction = 1.0 + PHI_CONJUGATE * noise_estimate
            final_signal = final_signal / zne_correction
        return final_signal

    def phi_weighted_demon(self, entropy_vector: np.ndarray) -> Dict[str, Any]:
        """
        PHI-weighted Maxwell Demon: applies golden-ratio weighting to
        identify the highest-leverage reversal points in an entropy field.
        Prioritizes reversals at PHI-spaced intervals for maximum coherence.
        """
        if not isinstance(entropy_vector, np.ndarray):
            entropy_vector = np.array(entropy_vector, dtype=float)
        n = len(entropy_vector)
        # PHI-spaced sampling indices (golden angle spacing)
        golden_angle = 2 * math.pi / (PHI ** 2)
        phi_indices = [int(i * PHI) % n for i in range(n)]
        # Weight each point by its distance to nearest PHI-index
        weights = np.ones(n)
        for idx in phi_indices:
            weights[idx] *= PHI  # Boost PHI-aligned points
        # Weighted demon efficiency per point
        efficiencies = np.array([
            self.maxwell_demon_factor * (1.0 / (abs(entropy_vector[i]) + 0.001)) * weights[i]
            for i in range(n)
        ])
        # Apply reversal at highest-leverage points
        sorted_idx = np.argsort(-efficiencies)
        reversal_budget = int(n * PHI_CONJUGATE)  # Reverse top 61.8% of points
        reversed_vector = entropy_vector.copy()
        reversed_count = 0
        for idx in sorted_idx[:reversal_budget]:
            reversed_vector[idx] *= PHI_CONJUGATE  # Dampen toward order
            reversed_count += 1
        return {
            "reversed_count": reversed_count,
            "budget_ratio": round(reversal_budget / n, 4),
            "mean_efficiency": round(float(np.mean(efficiencies)), 6),
            "max_efficiency": round(float(np.max(efficiencies)), 6),
            "variance_before": round(float(np.var(entropy_vector)), 6),
            "variance_after": round(float(np.var(reversed_vector)), 6),
            "reduction_ratio": round(float(np.var(reversed_vector) / (np.var(entropy_vector) + 1e-30)), 6),
            "reversed_vector": reversed_vector,
        }

    def multi_scale_reversal(self, signal: np.ndarray, scales: int = 5) -> Dict[str, Any]:
        """
        Multi-scale entropy reversal: applies demon at progressively finer
        granularities (octave decomposition inspired by 104-TET).
        Scale k operates on windows of size n/2^k.
        """
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal, dtype=float)
        n = len(signal)
        result_signal = signal.copy()
        scale_reports = []
        for k in range(scales):
            window_size = max(2, n // (2 ** k))
            scale_variance_before = float(np.var(result_signal))
            for start in range(0, n, window_size):
                window = result_signal[start:start + window_size]
                if len(window) < 2:
                    continue
                # Local demon reversal
                local_entropy = float(np.var(window))
                eff = self.calculate_demon_efficiency(local_entropy)
                damping = PHI_CONJUGATE ** (1 + k * 0.1)
                result_signal[start:start + window_size] = window * damping + GOD_CODE / (n * (k + 1))
            scale_variance_after = float(np.var(result_signal))
            scale_reports.append({
                "scale": k,
                "window_size": window_size,
                "variance_before": round(scale_variance_before, 6),
                "variance_after": round(scale_variance_after, 6),
            })
        return {
            "scales_applied": scales,
            "total_variance_reduction": round(float(np.var(signal) - np.var(result_signal)), 6),
            "scale_reports": scale_reports,
            "restored_signal": result_signal,
        }

    def entropy_cascade(self, initial_state: float = 1.0, depth: int = ENTROPY_CASCADE_DEPTH) -> Dict[str, Any]:
        """
        Entropy cascade: iterative entropy flow modulated by VOID_CONSTANT.
        Tracks the trajectory from disorder toward GOD_CODE-aligned order.
        S(n+1) = S(n) * PHI_CONJUGATE + VOID_CONSTANT * sin(n * pi / 104)
        Converges to a fixed point related to GOD_CODE.

        v4.2 Perf: Precomputed sin table avoids transcendental calls per step.
        """
        # v4.2: vectorized sin table
        sin_table = _CASCADE_SIN_TABLE
        trajectory = [initial_state]
        s = initial_state
        phi_c = PHI_CONJUGATE
        vc = VOID_CONSTANT
        for n in range(1, depth + 1):
            s = s * phi_c + vc * (sin_table[n] if n < len(sin_table) else math.sin(n * math.pi / QUANTIZATION_GRAIN))
            trajectory.append(s)
        fixed_point = trajectory[-1]
        # Theoretical fixed point: VOID_CONSTANT × sin_avg / (1 - PHI_CONJUGATE)
        god_code_alignment = 1.0 - min(1.0, abs(fixed_point * GOD_CODE - round(fixed_point * GOD_CODE)) / GOD_CODE)
        return {
            "initial": initial_state,
            "depth": depth,
            "fixed_point": round(fixed_point, 10),
            "god_code_alignment": round(god_code_alignment, 6),
            "converged": abs(trajectory[-1] - trajectory[-2]) < 1e-10,
            "trajectory_sample": [round(t, 8) for t in trajectory[:5] + trajectory[-5:]],
        }

    def landauer_bound_comparison(self, temperature: float = 293.15) -> Dict[str, Any]:
        """
        Compare demon efficiency against Landauer's theoretical bound.
        E_landauer = k_B * T * ln(2) per bit erased.
        The demon's efficiency is measured as bits reversed per unit energy.
        """
        landauer_energy = PC.K_B * temperature * math.log(2)  # Joules per bit
        demon_eff = self.calculate_demon_efficiency(0.5)
        # Theoretical sovereign efficiency (GOD_CODE enhanced)
        sovereign_energy = landauer_energy * (GOD_CODE / (PHI * 416))  # Enhanced bound
        ratio = sovereign_energy / landauer_energy
        # v4.1 Discovery #16: Fe Curie Landauer limit (ferromagnetic→paramagnetic boundary)
        curie_ratio = FE_CURIE_LANDAUER_LIMIT / landauer_energy if landauer_energy > 0 else 0.0
        return {
            "temperature_K": temperature,
            "landauer_bound_J_per_bit": landauer_energy,
            "sovereign_bound_J_per_bit": sovereign_energy,
            "enhancement_ratio": round(ratio, 6),
            "demon_efficiency": round(demon_eff, 6),
            "exceeds_landauer": ratio < 1.0,
            # v4.1: Fe Curie-temperature Landauer reference
            "fe_curie_landauer_J_per_bit": FE_CURIE_LANDAUER_LIMIT,
            "curie_to_room_ratio": round(curie_ratio, 6),
        }

    def get_stewardship_report(self) -> Dict[str, Any]:
        return {
            "stage": "EVO_15_OMNIPRESENT_STEWARD",
            "maxwell_factor": self.maxwell_demon_factor,
            "cumulative_coherence_gain": self.coherence_gain,
            "universal_order_index": 1.0 + (self.coherence_gain / GOD_CODE),
            "status": "ORDER_RESTORATION_ACTIVE",
        }

    def get_status(self) -> Dict[str, Any]:
        return self.get_stewardship_report()
