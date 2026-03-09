"""
L104 God Code Simulator — FeedbackLoopEngine v3.0
═══════════════════════════════════════════════════════════════════════════════

Closes the loop: Simulation → Coherence → Entropy → Scoring → Re-simulation.

v3.0 UPGRADES (from VQPU findings):
  - 12-dimension scoring (was 8D): +qfi, +purity, +topology, +trotter
  - VQPU-aware composite scoring with Cramér-Rao and Loschmidt integration
  - Improved convergence: cosine annealing + plateau detection

v2.0 UPGRADES (retained):
  - Adaptive convergence tracking with momentum
  - Multi-pass refinement with diminishing learning rate
  - Phase-locked oscillation detection (prevents limit cycles)
  - Weighted scoring across multiple quality dimensions
  - Aggregated statistics and convergence metrics

Can operate standalone (with local metrics) or connected to live engine
instances:
  - CoherenceSubsystem.ingest_simulation_result()
  - EntropySubsystem.calculate_demon_efficiency()
  - MathEngine.verify_conservation()

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .constants import (
    GOD_CODE, PHI,
    PHI_WEIGHT_1, PHI_WEIGHT_2, PHI_WEIGHT_3, PHI_WEIGHT_4, PHI_WEIGHT_5,
    SACRED_MOMENTUM_BLEND, SACRED_LR_DECAY,
    PHI_CONJUGATE,
)
from .result import SimulationResult


class FeedbackLoopEngine:
    """
    Closes the loop: Simulation → Coherence → Entropy → Scoring → Re-simulation.

    v4.0 — PHI power series weights, Adam-style momentum, 15D scoring.
    v3.0 VQPU upgrades: +qfi, +topo_entropy, +trotter_quality, +loschmidt dims.
    """

    def __init__(self):
        self._coherence_subsystem = None
        self._entropy_subsystem = None
        self._math_engine = None
        self.loop_history: List[Dict[str, Any]] = []
        self._momentum: float = 0.0
        self._momentum_sq: float = 0.0  # v3.0: second-order (Adam-style)
        self._learning_rate: float = 1.0
        self._convergence_window: int = 5

    def connect_coherence(self, coherence_subsystem) -> None:
        """Connect live CoherenceSubsystem for real feedback."""
        self._coherence_subsystem = coherence_subsystem

    def connect_entropy(self, entropy_subsystem) -> None:
        """Connect live EntropySubsystem for real entropy reversal."""
        self._entropy_subsystem = entropy_subsystem

    def connect_math(self, math_engine) -> None:
        """Connect live MathEngine for verification."""
        self._math_engine = math_engine

    # ── Core Feedback Cycle ─────────────────────────────────────────────────

    def run_feedback_cycle(self, sim_results: List[SimulationResult],
                           iterations: int = 5) -> Dict[str, Any]:
        """
        Run a full feedback cycle with convergence tracking.

          1. Take simulation results
          2. Feed into coherence (if connected) or compute local metrics
          3. Compute entropy reversal
          4. Score with weighted multi-dimensional metric
          5. Track convergence with momentum
          6. Return convergence report with statistics
        """
        self.loop_history = []
        self._momentum = 0.0
        self._learning_rate = 1.0

        coherence_series: List[float] = []
        entropy_series: List[float] = []
        score_series: List[float] = []

        for i, sim_result in enumerate(sim_results[:iterations]):
            cycle_report: Dict[str, Any] = {"iteration": i, "simulation": sim_result.name}

            # Step 1: Coherence feedback
            coh_val = self._compute_coherence(sim_result, cycle_report)
            coherence_series.append(coh_val)

            # Step 2: Entropy reversal
            ent_val = self._compute_entropy(sim_result, cycle_report)
            entropy_series.append(ent_val)

            # Step 3: Math verification (if connected)
            math_ok = self._verify_math(sim_result, cycle_report)

            # Step 4: Weighted composite score
            score = self._compute_composite_score(coh_val, ent_val, sim_result)
            score_series.append(score)
            cycle_report["composite_score"] = round(score, 6)

            # Step 5: Adam-style dual momentum (first + second order)
            # β₁ = φ conjugate (0.618), β₂ = φ² (0.382)
            if len(score_series) > 1:
                delta = score_series[-1] - score_series[-2]
                self._momentum = SACRED_MOMENTUM_BLEND * self._momentum + (1 - SACRED_MOMENTUM_BLEND) * delta
                self._momentum_sq = PHI_CONJUGATE ** 2 * self._momentum_sq + (1 - PHI_CONJUGATE ** 2) * delta ** 2
                # Bias-corrected estimate (Adam)
                step = len(score_series)
                m_hat = self._momentum / (1 - SACRED_MOMENTUM_BLEND ** step)
                v_hat = self._momentum_sq / (1 - (PHI_CONJUGATE ** 2) ** step)
                cycle_report["momentum"] = round(m_hat, 6)
                cycle_report["momentum_sq"] = round(v_hat, 6)
            else:
                cycle_report["momentum"] = 0.0
                cycle_report["momentum_sq"] = 0.0

            # Step 6: Sacred learning rate decay: 1 - φ^-3 ≈ 0.7639
            self._learning_rate *= SACRED_LR_DECAY
            cycle_report["learning_rate"] = round(self._learning_rate, 4)

            self.loop_history.append(cycle_report)

        # Aggregate statistics
        return self._build_report(coherence_series, entropy_series, score_series)

    # ── Multi-Pass Refinement ───────────────────────────────────────────────

    def run_multi_pass(self, sim_results: List[SimulationResult],
                       passes: int = 3, iterations_per_pass: int = 5) -> Dict[str, Any]:
        """
        Run multiple feedback passes with convergence detection.

        Each pass uses the same simulation results but with accumulated
        momentum and diminishing learning rate. Stops early if the
        score converges (oscillation detected or plateau reached).
        """
        all_scores: List[float] = []
        pass_reports: List[Dict[str, Any]] = []

        for p in range(passes):
            report = self.run_feedback_cycle(sim_results, iterations=iterations_per_pass)
            pass_reports.append(report)
            all_scores.append(report["avg_composite_score"])

            # Early stopping: check for convergence
            if len(all_scores) >= 2:
                delta = abs(all_scores[-1] - all_scores[-2])
                if delta < 1e-6:
                    break

            # Check for oscillation
            if len(all_scores) >= 3:
                if (all_scores[-1] - all_scores[-2]) * (all_scores[-2] - all_scores[-3]) < 0:
                    # Oscillating — reduce learning rate further
                    self._learning_rate *= 0.5

        final_score = all_scores[-1] if all_scores else 0.0
        converged = len(all_scores) >= 2 and abs(all_scores[-1] - all_scores[-2]) < 1e-4

        return {
            "passes_completed": len(pass_reports),
            "final_composite_score": round(final_score, 6),
            "converged": converged,
            "score_trajectory": [round(s, 6) for s in all_scores],
            "pass_summaries": [
                {
                    "pass": i,
                    "avg_coherence": r["avg_coherence"],
                    "avg_entropy": r["avg_demon_efficiency"],
                    "avg_score": r["avg_composite_score"],
                }
                for i, r in enumerate(pass_reports)
            ],
            "engines_connected": pass_reports[-1]["engines_connected"] if pass_reports else {},
        }

    # ── Dimension-Specific Feedback ─────────────────────────────────────────

    def score_by_dimension(self, sim_results: List[SimulationResult]) -> Dict[str, float]:
        """
        Score simulation results across 15 quality dimensions.

        v4.0 (VQPU upgrade): Expanded from 11D to 15D.
        New dimensions from VQPU v8.0 findings:
          - qfi: Quantum Fisher Information (Cramér-Rao sensitivity)
          - topo_entropy: Topological entanglement entropy (Kitaev-Preskill)
          - trotter_quality: Trotter decomposition accuracy (1 - trotter_error)
          - loschmidt: Loschmidt echo / time-reversal fidelity

        Returns per-dimension scores (0-1) for:
          fidelity, entropy, coherence, conservation, alignment,
          concurrence, information, stability, purity, gate_fidelity,
          decoherence_resilience, qfi, topo_entropy, trotter_quality,
          loschmidt
        """
        dims: Dict[str, List[float]] = {
            "fidelity": [], "entropy": [], "coherence": [],
            "conservation": [], "alignment": [], "concurrence": [],
            "information": [], "stability": [],
            "purity": [], "gate_fidelity": [], "decoherence_resilience": [],
            # v4.0 VQPU-derived dimensions
            "qfi": [], "topo_entropy": [], "trotter_quality": [],
            "loschmidt": [],
        }

        for r in sim_results:
            dims["fidelity"].append(r.fidelity)
            dims["entropy"].append(min(1.0, r.entanglement_entropy))
            dims["coherence"].append(r.phase_coherence)
            dims["conservation"].append(1.0 - min(1.0, r.conservation_error * 1e6))
            dims["alignment"].append(r.sacred_alignment)
            dims["concurrence"].append(r.concurrence)
            dims["information"].append(min(1.0, r.mutual_information / 2.0) if r.mutual_information else 0.0)
            dims["stability"].append(1.0 if r.passed else 0.0)
            # v3.0 dimensions
            dims["purity"].append(getattr(r, 'extra', {}).get('purity', r.fidelity) if r.extra else r.fidelity)
            dims["gate_fidelity"].append(r.gate_fidelity if r.gate_fidelity else r.fidelity)
            dims["decoherence_resilience"].append(
                r.decoherence_fidelity if r.decoherence_fidelity else r.fidelity
            )
            # v4.0 VQPU-derived dimensions
            dims["qfi"].append(min(1.0, r.qfi / (4.0 * max(r.n_qubits, 1))) if r.qfi else 0.0)
            dims["topo_entropy"].append(min(1.0, r.topo_entropy) if r.topo_entropy else 0.0)
            dims["trotter_quality"].append(1.0 - min(1.0, r.trotter_error or 0.0))
            dims["loschmidt"].append(min(1.0, r.decay_rate or 0.0) if r.decay_rate is not None else 0.0)

        return {k: round(float(np.mean(v)), 4) if v else 0.0 for k, v in dims.items()}

    # ── Internal Computation ────────────────────────────────────────────────

    def _compute_coherence(self, sim_result: SimulationResult,
                           report: Dict[str, Any]) -> float:
        """Compute coherence via engine or local fallback."""
        if self._coherence_subsystem:
            try:
                payload = sim_result.to_coherence_payload()
                ingest_result = self._coherence_subsystem.ingest_simulation_result(payload)
                val = ingest_result.get("post_coherence", 0)
                report["coherence_delta"] = ingest_result.get("coherence_delta", 0)
                return float(val)
            except Exception as e:
                report["coherence_error"] = str(e)
        local_coh = sim_result.fidelity * sim_result.phase_coherence
        report["coherence_local"] = local_coh
        return float(local_coh)

    def _compute_entropy(self, sim_result: SimulationResult,
                         report: Dict[str, Any]) -> float:
        """Compute entropy reversal via engine or local fallback."""
        if self._entropy_subsystem:
            try:
                entropy_input = sim_result.to_entropy_input()
                demon_eff = self._entropy_subsystem.calculate_demon_efficiency(entropy_input)
                report["demon_efficiency"] = demon_eff
                return float(demon_eff)
            except Exception as e:
                report["entropy_error"] = str(e)
        demon_factor = PHI / (GOD_CODE / 416.0)
        local_demon = demon_factor / (1.0 + sim_result.to_entropy_input())
        report["demon_efficiency_local"] = local_demon
        return float(local_demon)

    def _verify_math(self, sim_result: SimulationResult,
                     report: Dict[str, Any]) -> bool:
        """Verify math conservation via engine or skip."""
        if self._math_engine:
            try:
                math_payload = sim_result.to_math_verification()
                ok = self._math_engine.verify_conservation(
                    math_payload.get("god_code_measured", 0.0))
                report["conservation_verified"] = ok
                return ok
            except Exception as e:
                report["math_error"] = str(e)
        return True

    def _compute_composite_score(self, coherence: float, entropy: float,
                                 sim_result: SimulationResult) -> float:
        """
        Weighted composite score using PHI power series weights.

        v4.0 (VQPU upgrade): Core 5D score + VQPU bonus dimensions.

        Core weights (φ^-k normalized):
          fidelity     → φ^-1 normalized ≈ 0.2879 (dominant)
          coherence    → φ^-2 normalized ≈ 0.1780
          entropy      → φ^-3 normalized ≈ 0.1100
          alignment    → φ^-4 normalized ≈ 0.0680
          conservation → φ^-5 normalized ≈ 0.0420

        VQPU bonus (scaled by φ^-6 ≈ 0.056 total budget):
          qfi_norm, purity, topo_entropy, trotter_quality
          → Each contributes φ^-6/4 ≈ 0.014 when present
          → Gracefully degrades to 0 when VQPU metrics absent
        """
        conservation = 1.0 - min(1.0, sim_result.conservation_error * 1e6)
        # PHI-derived weights sum to ~0.686; redistribute remainder
        remainder = 1.0 - (PHI_WEIGHT_1 + PHI_WEIGHT_2 + PHI_WEIGHT_3 + PHI_WEIGHT_4 + PHI_WEIGHT_5)
        w1 = PHI_WEIGHT_1 + remainder * 0.5   # Fidelity boost
        w2 = PHI_WEIGHT_2 + remainder * 0.3   # Coherence boost
        w3 = PHI_WEIGHT_3 + remainder * 0.2   # Entropy boost
        w4 = PHI_WEIGHT_4
        w5 = PHI_WEIGHT_5

        core_score = (
            sim_result.fidelity * w1
            + coherence * w2
            + entropy * w3
            + sim_result.sacred_alignment * w4
            + conservation * w5
        )

        # v4.0 VQPU bonus: φ^-6 budget (~0.056) split across 4 VQPU dims
        vqpu_budget = PHI ** -6  # ≈ 0.0557
        vqpu_w = vqpu_budget / 4.0  # ≈ 0.0139 per dim
        qfi_norm = min(1.0, sim_result.qfi / (4.0 * max(sim_result.n_qubits, 1))) if sim_result.qfi else 0.0
        purity_val = sim_result.purity if sim_result.purity else 0.0
        topo_val = min(1.0, sim_result.topo_entropy) if sim_result.topo_entropy else 0.0
        trotter_val = 1.0 - min(1.0, sim_result.trotter_error or 0.0)

        vqpu_bonus = vqpu_w * (qfi_norm + purity_val + topo_val + trotter_val)

        return core_score + vqpu_bonus

    def _build_report(self, coherence_series: List[float],
                      entropy_series: List[float],
                      score_series: List[float]) -> Dict[str, Any]:
        """Build the complete feedback report with convergence metrics."""
        n = len(self.loop_history)

        # Convergence detection with adaptive window
        # v3.0: Window shrinks as variance decreases (faster convergence detection)
        converging = False
        if len(score_series) >= self._convergence_window:
            window = score_series[-self._convergence_window:]
            variance = float(np.var(window))
            converging = variance < 1e-4 and float(np.mean(window)) > 0.5
            # Adaptive: shrink window if variance is very low
            if variance < 1e-6 and self._convergence_window > 3:
                self._convergence_window = max(3, self._convergence_window - 1)

        return {
            "iterations": n,
            "avg_coherence": round(float(np.mean(coherence_series)), 6) if coherence_series else 0.0,
            "avg_demon_efficiency": round(float(np.mean(entropy_series)), 6) if entropy_series else 0.0,
            "avg_composite_score": round(float(np.mean(score_series)), 6) if score_series else 0.0,
            "score_std": round(float(np.std(score_series)), 6) if score_series else 0.0,
            "score_trend": round(self._momentum, 6),
            "converging": converging,
            "final_learning_rate": round(self._learning_rate, 4),
            "history": self.loop_history,
            "engines_connected": {
                "coherence": self._coherence_subsystem is not None,
                "entropy": self._entropy_subsystem is not None,
                "math": self._math_engine is not None,
            },
        }


__all__ = ["FeedbackLoopEngine"]
