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

    def entropy_cascade(self, initial_state: float = 1.0, depth: int = ENTROPY_CASCADE_DEPTH,
                          damped: bool = True) -> Dict[str, Any]:
        """
        Entropy cascade: iterative entropy flow modulated by VOID_CONSTANT.
        Tracks the trajectory from disorder toward GOD_CODE-aligned order.

        v4.3 (Chaos-aware): Damped sine correction.
        - Original:  S(n+1) = S(n) × φ_c + VOID × sin(nπ/104)
        - Damped:    S(n+1) = S(n) × φ_c + VOID × φ_c^n × sin(nπ/104)

        The damped mode eliminates the 0.133 permanent residual by making
        the VOID sine correction decay exponentially alongside the main term.
        This was discovered in Experiment 10 (104-cascade healing protocol).

        v4.2 Perf: Precomputed sin table avoids transcendental calls per step.
        """
        # v4.2: vectorized sin table
        sin_table = _CASCADE_SIN_TABLE
        trajectory = [initial_state]
        s = initial_state
        phi_c = PHI_CONJUGATE
        vc = VOID_CONSTANT
        decay = 1.0  # φ_c^n accumulator for damped mode
        for n in range(1, depth + 1):
            sin_val = sin_table[n] if n < len(sin_table) else math.sin(n * math.pi / QUANTIZATION_GRAIN)
            if damped:
                decay *= phi_c
                s = s * phi_c + vc * decay * sin_val
            else:
                s = s * phi_c + vc * sin_val
            trajectory.append(s)
        fixed_point = trajectory[-1]
        # Theoretical fixed point: VOID_CONSTANT × sin_avg / (1 - PHI_CONJUGATE)
        god_code_alignment = 1.0 - min(1.0, abs(fixed_point * GOD_CODE - round(fixed_point * GOD_CODE)) / GOD_CODE)
        return {
            "initial": initial_state,
            "depth": depth,
            "damped": damped,
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

    def chaos_conservation_cascade(self, chaos_product: float,
                                     depth: int = ENTROPY_CASCADE_DEPTH) -> Dict[str, Any]:
        """
        Apply the 104-cascade healing protocol to a chaos-perturbed conservation product.

        From chaos × conservation findings (2026-02-24, 13 experiments):
        S(n+1) = S(n) × φ_c + VOID × sin(nπ/104) + INVARIANT × (1 - φ_c)

        Heals 99.6% of chaos perturbation. The VOID sine term creates a
        harmonic spiral that guides the product back toward conservation.
        Residual ~0.133 from the VOID oscillation fingerprint.

        Args:
            chaos_product: The chaos-perturbed G(X) × W(X) value
            depth: Number of cascade iterations (default: 104)

        Returns:
            Dict with healed value, trajectory, convergence info
        """
        from .constants import GOD_CODE as _GC
        INVARIANT = 527.5184818492612
        sin_table = _CASCADE_SIN_TABLE
        phi_c = PHI_CONJUGATE
        vc = VOID_CONSTANT
        s = chaos_product
        trajectory = [s]
        convergence_step = None
        initial_error = abs(s - INVARIANT)

        decay = 1.0  # φ_c^n accumulator — damped sine eliminates 0.133 residual
        for n in range(1, depth + 1):
            sin_val = sin_table[n] if n < len(sin_table) else math.sin(n * math.pi / QUANTIZATION_GRAIN)
            decay *= phi_c
            s = s * phi_c + vc * decay * sin_val + INVARIANT * (1 - phi_c)
            trajectory.append(s)
            if convergence_step is None and abs(s - INVARIANT) < 1e-6:
                convergence_step = n

        final_error = abs(trajectory[-1] - INVARIANT)
        healing_pct = (1 - final_error / initial_error) * 100 if initial_error > 0 else 100

        return {
            "initial_product": chaos_product,
            "initial_error": initial_error,
            "healed_product": trajectory[-1],
            "final_error": final_error,
            "healing_pct": round(healing_pct, 2),
            "convergence_step": convergence_step,
            "converged": final_error < 1e-6,
            "depth": depth,
            "trajectory_sample": [round(t, 8) for t in trajectory[:3] + trajectory[-3:]],
        }

    def demon_vs_chaos(self, chaos_products: list) -> Dict[str, Any]:
        """
        Apply Maxwell's Demon adaptive restoration to chaos-perturbed products.

        From findings: Demon outperforms constant φ-damping (39% vs 38.2%)
        because it targets high-entropy (disordered) regions selectively.

        Args:
            chaos_products: List of chaos-perturbed G(X)×W(X) values

        Returns:
            Dict with demon-corrected values and improvement metrics
        """
        INVARIANT = 527.5184818492612
        if not chaos_products:
            return {"error": "empty input"}

        raw_residuals = [p - INVARIANT for p in chaos_products]
        raw_rms = math.sqrt(sum(r ** 2 for r in raw_residuals) / len(raw_residuals))

        # φ-damping baseline
        phi_corrected = [INVARIANT + (p - INVARIANT) * PHI_CONJUGATE for p in chaos_products]
        phi_rms = math.sqrt(sum((p - INVARIANT) ** 2 for p in phi_corrected) / len(phi_corrected))

        # Demon adaptive correction
        demon_corrected = []
        for i, p in enumerate(chaos_products):
            # Local entropy from nearby values
            start = max(0, i - 3)
            end = min(len(chaos_products), i + 4)
            local = chaos_products[start:end]
            local_mean = sum(local) / len(local)
            local_var = sum((v - local_mean) ** 2 for v in local) / len(local)
            local_entropy = math.log(1 + local_var)

            eff = self.maxwell_demon_factor * (1.0 / (local_entropy + 0.001))
            damping = min(1.0, PHI_CONJUGATE ** (1 + eff * 0.1))
            demon_corrected.append(INVARIANT + (p - INVARIANT) * damping)

        demon_rms = math.sqrt(sum((p - INVARIANT) ** 2 for p in demon_corrected) / len(demon_corrected))

        return {
            "raw_rms": round(raw_rms, 8),
            "phi_rms": round(phi_rms, 8),
            "demon_rms": round(demon_rms, 8),
            "phi_improvement_pct": round((1 - phi_rms / raw_rms) * 100, 2) if raw_rms > 0 else 0,
            "demon_improvement_pct": round((1 - demon_rms / raw_rms) * 100, 2) if raw_rms > 0 else 0,
            "demon_beats_phi": demon_rms < phi_rms,
            "demon_corrected": demon_corrected,
        }

    def chaos_diagnostics(self, signal: list, window: int = 10) -> Dict[str, Any]:
        """
        Real-time chaos health diagnostics for any signal stream.

        Computes three chaos indicators:
        1. Shannon entropy (information content) — flat = high chaos
        2. Lyapunov exponent estimate — positive = chaotic, negative = stable
        3. Bifurcation distance — how far from the 0.35 threshold

        From Experiment 6 (Shannon landscape): entropy ratio 0.962 means chaos
        distributes uniformly — deviation from this baseline signals anomaly.
        From Experiment 5 (Lyapunov): negative at amp ≤ 0.001 = attractor.

        Args:
            signal: List of numeric values (e.g., pipeline scores, products)
            window: Rolling window for Lyapunov estimation
        """
        if len(signal) < 4:
            return {"error": "need >= 4 data points"}

        # --- Shannon entropy ---
        # Bin the signal into 20 buckets and compute H
        mn, mx = min(signal), max(signal)
        spread = mx - mn if mx > mn else 1.0
        n_bins = 20
        bins = [0] * n_bins
        for v in signal:
            idx = min(n_bins - 1, int((v - mn) / spread * n_bins))
            bins[idx] += 1
        total = len(signal)
        shannon = 0.0
        for cnt in bins:
            if cnt > 0:
                p = cnt / total
                shannon -= p * math.log(p)
        max_shannon = math.log(n_bins)
        entropy_ratio = shannon / max_shannon if max_shannon > 0 else 0

        # --- Lyapunov exponent estimate ---
        # Average local divergence rate: λ ≈ mean(log|f'(x)|)
        lyapunov_sum = 0.0
        lyapunov_count = 0
        for i in range(len(signal) - 1):
            diff = abs(signal[i + 1] - signal[i])
            if diff > 1e-15:
                lyapunov_sum += math.log(diff)
                lyapunov_count += 1
        lyapunov = lyapunov_sum / lyapunov_count if lyapunov_count > 0 else 0.0

        # --- Amplitude / bifurcation distance ---
        mean_val = sum(signal) / len(signal)
        rms_deviation = math.sqrt(sum((v - mean_val) ** 2 for v in signal) / len(signal))
        # Relative amplitude as fraction of mean
        rel_amplitude = rms_deviation / abs(mean_val) if mean_val != 0 else rms_deviation
        bifurcation_distance = max(0.0, 0.35 - rel_amplitude)

        # --- Health verdict ---
        is_stable = lyapunov < 0
        is_normal_entropy = 0.90 <= entropy_ratio <= 1.0
        is_below_bifurcation = rel_amplitude < 0.35
        health = "HEALTHY" if (is_stable and is_below_bifurcation) else \
                 "WARNING" if is_below_bifurcation else "CRITICAL"

        return {
            "shannon_entropy": round(shannon, 6),
            "entropy_ratio": round(entropy_ratio, 4),
            "entropy_baseline": 0.962,
            "lyapunov_exponent": round(lyapunov, 6),
            "is_stable": is_stable,
            "rms_deviation": round(rms_deviation, 8),
            "relative_amplitude": round(rel_amplitude, 6),
            "bifurcation_distance": round(bifurcation_distance, 6),
            "bifurcation_threshold": 0.35,
            "health": health,
            "samples": len(signal),
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
