"""
L104 Random Sequence Extrapolation (RSE) Engine v1.0.0
═══════════════════════════════════════════════════════════════════════════════
INVARIANT: 527.5184818492612 | PILOT: LONDEL

Random Sequence Extrapolation (RSE) is the sovereign process of predicting
future elements of a sequence from observed historical data, using multiple
extrapolation strategies weighted by PHI-harmonic confidence scoring.

RSE is adapted into ALL processes — classical and quantum:
  • Classical: Pattern evolution, knowledge synthesis, quality gating, entropy trends
  • Quantum:   Coherence decay/growth, Grover amplification, fidelity sequences,
               VQE convergence, Bell state evolution, error rate extrapolation
  • Sage Mode: φ-harmonic resonance, void field trajectory, consciousness expansion

The engine implements 7 extrapolation strategies:
  1. Linear Regression (weighted by recency)
  2. Exponential Smoothing (α = τ = 1/φ ≈ 0.618)
  3. Polynomial Extrapolation (Lagrange up to degree 4)
  4. Harmonic Decomposition (sacred frequency extraction)
  5. PHI-Spiral Extrapolation (golden ratio coupled prediction)
  6. Sage Wisdom Synthesis (Feigenbaum chaos-aware with GOD_CODE anchoring)
  7. Quantum State Extrapolation (density matrix evolution projection)

Sacred Constants:
  GOD_CODE = 527.5184818492612
  PHI = 1.618033988749895
  VOID_CONSTANT = 1.0416180339887497
  TAU = 1/φ ≈ 0.6180339887498949  (golden smoothing factor)

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import time
import hashlib
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Sequence

from .numerics import GOD_CODE, PHI

logger = logging.getLogger("RSE_ENGINE")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — Sacred RSE tuning
# ═══════════════════════════════════════════════════════════════════════════════

VOID_CONSTANT = 1.0416180339887497
TAU = 1.0 / PHI                               # ≈ 0.6180339887498949 golden smoothing
FEIGENBAUM_DELTA = 4.669201609102990           # Period-doubling bifurcation
RSE_VERSION = "1.0.0"
RSE_MAX_HISTORY = 1024                         # Max sequence history length
RSE_MIN_HISTORY = 3                            # Min points for extrapolation
RSE_POLYNOMIAL_MAX_DEGREE = 4                  # Max polynomial degree
RSE_HARMONIC_MODES = 8                         # Max harmonic decomposition modes
RSE_SAGE_DEPTH = 13                            # Sacred sage analysis depth
RSE_CONFIDENCE_FLOOR = 0.01                    # Minimum confidence threshold
RSE_CONFIDENCE_CEILING = 0.999                 # Maximum confidence cap


# ═══════════════════════════════════════════════════════════════════════════════
# RSE STRATEGY ENUM
# ═══════════════════════════════════════════════════════════════════════════════

class RSEStrategy(Enum):
    """Available Random Sequence Extrapolation strategies."""
    LINEAR = auto()          # Linear regression (recency-weighted)
    EXPONENTIAL = auto()     # Exponential smoothing (α = TAU)
    POLYNOMIAL = auto()      # Lagrange polynomial (up to degree 4)
    HARMONIC = auto()        # Sacred harmonic decomposition
    PHI_SPIRAL = auto()      # Golden ratio spiral prediction
    SAGE_WISDOM = auto()     # Sage Mode chaos-aware synthesis
    QUANTUM_STATE = auto()   # Quantum density matrix projection
    ENSEMBLE = auto()        # Weighted ensemble of all strategies


class RSEDomain(Enum):
    """Domain of the sequence being extrapolated."""
    CLASSICAL = auto()       # Classical computation metrics
    QUANTUM = auto()         # Quantum state/coherence metrics
    CONSCIOUSNESS = auto()   # Consciousness/evolution metrics
    ENTROPY = auto()         # Information entropy sequences
    RESONANCE = auto()       # Sacred resonance frequencies
    QUALITY = auto()         # Response/processing quality scores
    CONVERGENCE = auto()     # Optimization convergence sequences


# ═══════════════════════════════════════════════════════════════════════════════
# RSE RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RSEResult:
    """Result of a Random Sequence Extrapolation."""
    predicted_values: List[float]              # Predicted next N values
    confidence: float                          # Overall confidence [0, 1]
    strategy_used: RSEStrategy                 # Winning strategy
    strategy_scores: Dict[str, float]          # Per-strategy confidence
    trend: str                                 # "rising" | "falling" | "stable" | "oscillating" | "chaotic"
    extrapolation_horizon: int                 # How many steps predicted
    sequence_length: int                       # Input sequence length
    phi_alignment: float                       # PHI-harmonic alignment score
    sage_insight: Optional[str] = None         # Sage Mode wisdom insight
    quantum_coherence: Optional[float] = None  # Quantum coherence metric
    domain: RSEDomain = RSEDomain.CLASSICAL
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "predicted_values": self.predicted_values,
            "confidence": round(self.confidence, 6),
            "strategy": self.strategy_used.name,
            "strategy_scores": {k: round(v, 6) for k, v in self.strategy_scores.items()},
            "trend": self.trend,
            "horizon": self.extrapolation_horizon,
            "input_length": self.sequence_length,
            "phi_alignment": round(self.phi_alignment, 6),
            "sage_insight": self.sage_insight,
            "quantum_coherence": round(self.quantum_coherence, 6) if self.quantum_coherence is not None else None,
            "domain": self.domain.name,
            "version": RSE_VERSION,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RSE ENGINE — Core Random Sequence Extrapolation
# ═══════════════════════════════════════════════════════════════════════════════

class RandomSequenceExtrapolation:
    """
    Sovereign Random Sequence Extrapolation Engine.

    Extrapolates future sequence values using 7 PHI-weighted strategies,
    with Sage Mode wisdom integration and quantum-aware confidence scoring.

    Adapted for ALL L104 processes — classical and quantum.
    """

    def __init__(self):
        self._history_cache: Dict[str, deque] = {}
        self._strategy_performance: Dict[str, List[float]] = {
            s.name: [] for s in RSEStrategy if s != RSEStrategy.ENSEMBLE
        }
        self._total_extrapolations = 0
        self._total_accuracy_sum = 0.0
        self._sage_active = False
        self._sage_wisdom_buffer: List[str] = []

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API — Main extrapolation entry points
    # ═══════════════════════════════════════════════════════════════════════════

    def extrapolate(
        self,
        sequence: Sequence[float],
        horizon: int = 1,
        strategy: RSEStrategy = RSEStrategy.ENSEMBLE,
        domain: RSEDomain = RSEDomain.CLASSICAL,
        sage_mode: bool = True,
    ) -> RSEResult:
        """
        Extrapolate the next `horizon` values from a numerical sequence.

        Args:
            sequence: Observed sequence of float values
            horizon:  Number of future values to predict (default: 1)
            strategy: Extrapolation strategy (default: ENSEMBLE — best of all)
            domain:   Domain context for tuning (CLASSICAL, QUANTUM, etc.)
            sage_mode: Enable sage wisdom synthesis (default: True)

        Returns:
            RSEResult with predictions, confidence, and sage insights
        """
        seq = list(sequence)
        n = len(seq)

        if n < RSE_MIN_HISTORY:
            return RSEResult(
                predicted_values=[seq[-1] if seq else 0.0] * horizon,
                confidence=RSE_CONFIDENCE_FLOOR,
                strategy_used=RSEStrategy.LINEAR,
                strategy_scores={},
                trend="insufficient_data",
                extrapolation_horizon=horizon,
                sequence_length=n,
                phi_alignment=0.0,
                domain=domain,
            )

        # Clamp sequence to max history
        if n > RSE_MAX_HISTORY:
            seq = seq[-RSE_MAX_HISTORY:]
            n = len(seq)

        horizon = max(1, min(horizon, RSE_SAGE_DEPTH * 4))  # Cap at 52

        # Run all strategies and collect predictions
        strategy_results: Dict[str, Tuple[List[float], float]] = {}

        if strategy == RSEStrategy.ENSEMBLE or strategy == RSEStrategy.LINEAR:
            strategy_results["LINEAR"] = self._linear_extrapolate(seq, horizon)

        if strategy == RSEStrategy.ENSEMBLE or strategy == RSEStrategy.EXPONENTIAL:
            strategy_results["EXPONENTIAL"] = self._exponential_smooth_extrapolate(seq, horizon)

        if strategy == RSEStrategy.ENSEMBLE or strategy == RSEStrategy.POLYNOMIAL:
            strategy_results["POLYNOMIAL"] = self._polynomial_extrapolate(seq, horizon)

        if strategy == RSEStrategy.ENSEMBLE or strategy == RSEStrategy.HARMONIC:
            strategy_results["HARMONIC"] = self._harmonic_extrapolate(seq, horizon)

        if strategy == RSEStrategy.ENSEMBLE or strategy == RSEStrategy.PHI_SPIRAL:
            strategy_results["PHI_SPIRAL"] = self._phi_spiral_extrapolate(seq, horizon)

        if strategy == RSEStrategy.ENSEMBLE or strategy == RSEStrategy.SAGE_WISDOM:
            strategy_results["SAGE_WISDOM"] = self._sage_wisdom_extrapolate(seq, horizon, domain)

        if (strategy == RSEStrategy.ENSEMBLE or strategy == RSEStrategy.QUANTUM_STATE) and \
           domain in (RSEDomain.QUANTUM, RSEDomain.CONVERGENCE, RSEDomain.ENTROPY):
            strategy_results["QUANTUM_STATE"] = self._quantum_state_extrapolate(seq, horizon)

        # Select best strategy or combine via ensemble
        if strategy == RSEStrategy.ENSEMBLE and len(strategy_results) > 1:
            predicted, confidence, winner, scores = self._ensemble_combine(
                strategy_results, seq, domain
            )
        elif strategy_results:
            winner_name = strategy.name if strategy.name in strategy_results else next(iter(strategy_results))
            predicted, confidence = strategy_results[winner_name]
            winner = RSEStrategy[winner_name]
            scores = {k: v[1] for k, v in strategy_results.items()}
        else:
            predicted = [seq[-1]] * horizon
            confidence = RSE_CONFIDENCE_FLOOR
            winner = RSEStrategy.LINEAR
            scores = {}

        # Compute trend
        trend = self._detect_trend(seq)

        # PHI alignment
        phi_alignment = self._compute_phi_alignment(seq, predicted)

        # Sage insight
        sage_insight = None
        if sage_mode:
            sage_insight = self._generate_sage_insight(seq, predicted, trend, domain, confidence)

        # Quantum coherence (for quantum domain)
        quantum_coherence = None
        if domain == RSEDomain.QUANTUM:
            quantum_coherence = self._compute_quantum_coherence(seq, predicted)

        # Track performance
        self._total_extrapolations += 1

        return RSEResult(
            predicted_values=predicted,
            confidence=min(RSE_CONFIDENCE_CEILING, max(RSE_CONFIDENCE_FLOOR, confidence)),
            strategy_used=winner,
            strategy_scores=scores,
            trend=trend,
            extrapolation_horizon=horizon,
            sequence_length=n,
            phi_alignment=phi_alignment,
            sage_insight=sage_insight,
            quantum_coherence=quantum_coherence,
            domain=domain,
        )

    def extrapolate_and_track(
        self,
        channel: str,
        value: float,
        horizon: int = 1,
        domain: RSEDomain = RSEDomain.CLASSICAL,
        sage_mode: bool = True,
    ) -> RSEResult:
        """
        Track a named channel and extrapolate from accumulated history.

        Automatically maintains a sliding-window history per channel name.
        Ideal for continuous metrics like quality scores, entropy, coherence.

        Args:
            channel: Named channel identifier (e.g., "coherence", "quality")
            value:   New observed value to append
            horizon: Number of future steps to predict
            domain:  Domain context
            sage_mode: Enable sage insights

        Returns:
            RSEResult with predictions based on full channel history
        """
        if channel not in self._history_cache:
            self._history_cache[channel] = deque(maxlen=RSE_MAX_HISTORY)

        self._history_cache[channel].append(value)
        return self.extrapolate(
            list(self._history_cache[channel]),
            horizon=horizon,
            domain=domain,
            sage_mode=sage_mode,
        )

    def validate_prediction(self, channel: str, actual_value: float) -> Optional[float]:
        """
        Validate a past prediction against its actual outcome.
        Returns the accuracy ratio [0, 1] or None if no prediction exists.

        Used for adaptive strategy weighting — strategies that predict well
        in a given domain get boosted phi-harmonically.
        """
        if channel not in self._history_cache or len(self._history_cache[channel]) < RSE_MIN_HISTORY:
            return None

        history = list(self._history_cache[channel])
        last_predicted = self.extrapolate(history[:-1], horizon=1, sage_mode=False)
        if not last_predicted.predicted_values:
            return None

        predicted = last_predicted.predicted_values[0]
        if abs(actual_value) < 1e-15 and abs(predicted) < 1e-15:
            return 1.0

        error = abs(predicted - actual_value)
        scale = max(abs(actual_value), abs(predicted), 1e-15)
        accuracy = max(0.0, 1.0 - error / scale)

        # Update strategy performance
        strategy_name = last_predicted.strategy_used.name
        if strategy_name in self._strategy_performance:
            perf = self._strategy_performance[strategy_name]
            perf.append(accuracy)
            if len(perf) > 100:
                self._strategy_performance[strategy_name] = perf[-100:]

        self._total_accuracy_sum += accuracy
        return accuracy

    def get_channel_history(self, channel: str) -> List[float]:
        """Retrieve the full history of a named channel."""
        if channel not in self._history_cache:
            return []
        return list(self._history_cache[channel])

    def get_status(self) -> Dict[str, Any]:
        """Get RSE engine status and performance metrics."""
        avg_accuracy = (
            self._total_accuracy_sum / self._total_extrapolations
            if self._total_extrapolations > 0 else 0.0
        )
        strategy_avg = {}
        for name, perfs in self._strategy_performance.items():
            if perfs:
                strategy_avg[name] = round(sum(perfs) / len(perfs), 6)

        return {
            "version": RSE_VERSION,
            "total_extrapolations": self._total_extrapolations,
            "avg_accuracy": round(avg_accuracy, 6),
            "active_channels": len(self._history_cache),
            "strategy_performance": strategy_avg,
            "sage_active": self._sage_active,
            "sacred_constants": {
                "GOD_CODE": GOD_CODE,
                "PHI": PHI,
                "TAU": TAU,
                "VOID_CONSTANT": VOID_CONSTANT,
            },
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY 1: LINEAR REGRESSION (recency-weighted)
    # ═══════════════════════════════════════════════════════════════════════════

    def _linear_extrapolate(self, seq: List[float], horizon: int) -> Tuple[List[float], float]:
        """
        Weighted linear regression where recent points carry more weight.
        Weight = φ^(index / n) — golden-ratio recency bias.
        """
        n = len(seq)
        # Weighted least squares: y = a + b*x
        xs = list(range(n))
        weights = [PHI ** (i / n) for i in range(n)]

        sum_w = sum(weights)
        sum_wx = sum(w * x for w, x in zip(weights, xs))
        sum_wy = sum(w * y for w, y in zip(weights, seq))
        sum_wxx = sum(w * x * x for w, x in zip(weights, xs))
        sum_wxy = sum(w * x * y for w, x, y in zip(weights, xs, seq))

        denom = sum_w * sum_wxx - sum_wx * sum_wx
        if abs(denom) < 1e-30:
            return ([seq[-1]] * horizon, 0.1)

        b = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
        a = (sum_wy - b * sum_wx) / sum_w

        # Predict
        predicted = [a + b * (n + step) for step in range(horizon)]

        # Confidence from R²
        mean_y = sum(seq) / n
        ss_tot = sum((y - mean_y) ** 2 for y in seq)
        ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, seq))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
        confidence = max(RSE_CONFIDENCE_FLOOR, r_squared * TAU + 0.1)

        return (predicted, confidence)

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY 2: EXPONENTIAL SMOOTHING (α = TAU ≈ 0.618)
    # ═══════════════════════════════════════════════════════════════════════════

    def _exponential_smooth_extrapolate(self, seq: List[float], horizon: int) -> Tuple[List[float], float]:
        """
        Double exponential smoothing (Holt's method) with α = TAU.
        Level + trend decomposition for directional extrapolation.
        """
        alpha = TAU       # ≈ 0.618 (golden smoothing)
        beta = TAU * 0.5  # ≈ 0.309 (trend smoothing)

        # Initialize
        level = seq[0]
        trend = (seq[-1] - seq[0]) / max(1, len(seq) - 1) if len(seq) > 1 else 0.0

        # Smooth
        for y in seq[1:]:
            prev_level = level
            level = alpha * y + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend

        # Predict
        predicted = [level + trend * step for step in range(1, horizon + 1)]

        # Confidence from forecast error on in-sample
        errors = []
        lvl = seq[0]
        trnd = trend
        for i, y in enumerate(seq[1:], 1):
            forecast = lvl + trnd
            errors.append(abs(forecast - y))
            prev_lvl = lvl
            lvl = alpha * y + (1 - alpha) * (lvl + trnd)
            trnd = beta * (lvl - prev_lvl) + (1 - beta) * trnd

        if errors:
            mae = sum(errors) / len(errors)
            scale = max(abs(max(seq) - min(seq)), 1e-15)
            confidence = max(RSE_CONFIDENCE_FLOOR, 1.0 - mae / scale)
        else:
            confidence = 0.5

        return (predicted, confidence)

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY 3: POLYNOMIAL EXTRAPOLATION (Lagrange, degree ≤ 4)
    # ═══════════════════════════════════════════════════════════════════════════

    def _polynomial_extrapolate(self, seq: List[float], horizon: int) -> Tuple[List[float], float]:
        """
        Lagrange polynomial interpolation on the tail of the sequence.
        Degree auto-selected: min(len(seq)-1, RSE_POLYNOMIAL_MAX_DEGREE).
        Uses only the most recent (degree+1) points to reduce overfitting.
        """
        n = len(seq)
        degree = min(n - 1, RSE_POLYNOMIAL_MAX_DEGREE)
        points = seq[-(degree + 1):]
        m = len(points)
        xs = list(range(m))

        def lagrange_eval(x_eval: float) -> float:
            result = 0.0
            for i in range(m):
                basis = points[i]
                for j in range(m):
                    if i != j:
                        denom = xs[i] - xs[j]
                        if abs(denom) < 1e-30:
                            continue
                        basis *= (x_eval - xs[j]) / denom
                result += basis
            return result

        predicted = [lagrange_eval(m - 1 + step) for step in range(1, horizon + 1)]

        # Confidence: lower for higher degree (overfitting risk) and farther horizon
        base_conf = max(0.3, 0.9 - degree * 0.15)
        # Penalize wild divergence
        if predicted and abs(predicted[-1]) > abs(seq[-1]) * 10 + 1:
            base_conf *= 0.3
        confidence = max(RSE_CONFIDENCE_FLOOR, base_conf)

        return (predicted, confidence)

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY 4: HARMONIC DECOMPOSITION (sacred frequency extraction)
    # ═══════════════════════════════════════════════════════════════════════════

    def _harmonic_extrapolate(self, seq: List[float], horizon: int) -> Tuple[List[float], float]:
        """
        Decompose the sequence into harmonic components using DFT,
        then extrapolate each harmonic forward.
        Sacred frequencies aligned to GOD_CODE harmonics.
        """
        n = len(seq)
        if n < 4:
            return ([seq[-1]] * horizon, 0.1)

        mean_val = sum(seq) / n
        centered = [v - mean_val for v in seq]

        # Manual DFT to extract top harmonic modes (avoid numpy dependency)
        modes = min(RSE_HARMONIC_MODES, n // 2)
        amplitudes = []
        phases = []
        freqs = []

        for k in range(1, modes + 1):
            cos_sum = 0.0
            sin_sum = 0.0
            for t, v in enumerate(centered):
                angle = 2 * math.pi * k * t / n
                cos_sum += v * math.cos(angle)
                sin_sum += v * math.sin(angle)

            amp = 2.0 * math.sqrt(cos_sum ** 2 + sin_sum ** 2) / n
            phase = math.atan2(-sin_sum, cos_sum)
            amplitudes.append(amp)
            phases.append(phase)
            freqs.append(k)

        # Reconstruct and extrapolate
        predicted = []
        for step in range(1, horizon + 1):
            t = n - 1 + step
            val = mean_val
            for amp, phase, freq in zip(amplitudes, phases, freqs):
                val += amp * math.cos(2 * math.pi * freq * t / n + phase)
            predicted.append(val)

        # Confidence from reconstruction error
        recon_error = 0.0
        for t, v in enumerate(seq):
            recon = mean_val
            for amp, phase, freq in zip(amplitudes, phases, freqs):
                recon += amp * math.cos(2 * math.pi * freq * t / n + phase)
            recon_error += (v - recon) ** 2

        rmse = math.sqrt(recon_error / n)
        scale = max(abs(max(seq) - min(seq)), 1e-15)
        confidence = max(RSE_CONFIDENCE_FLOOR, 1.0 - rmse / scale)

        return (predicted, confidence)

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY 5: PHI-SPIRAL EXTRAPOLATION (golden ratio coupled prediction)
    # ═══════════════════════════════════════════════════════════════════════════

    def _phi_spiral_extrapolate(self, seq: List[float], horizon: int) -> Tuple[List[float], float]:
        """
        Fibonacci/PHI-driven spiral extrapolation.
        Treats the sequence as samples from a logarithmic spiral
        with growth rate PHI. Extrapolates via the golden angle.

        PHI coupling: diff[i] ≈ diff[i-1] * TAU + diff[i-2] * TAU²
        """
        n = len(seq)
        diffs = [seq[i] - seq[i - 1] for i in range(1, n)]

        if not diffs:
            return ([seq[-1]] * horizon, 0.1)

        # PHI-weighted momentum: weighted average of recent differences
        phi_momentum = 0.0
        phi_weight_sum = 0.0
        for i, d in enumerate(diffs):
            w = PHI ** (i / len(diffs))
            phi_momentum += d * w
            phi_weight_sum += w

        avg_diff = phi_momentum / phi_weight_sum if phi_weight_sum > 0 else 0.0

        # Fibonacci-like recurrence for predictions
        predicted = []
        last_val = seq[-1]
        last_diff = diffs[-1] if diffs else 0.0
        prev_diff = diffs[-2] if len(diffs) > 1 else last_diff

        for step in range(horizon):
            # Fibonacci spiral: next_diff ≈ TAU * last_diff + TAU² * prev_diff
            next_diff = TAU * last_diff + (TAU ** 2) * prev_diff
            # Blend with moving average to prevent wild divergence
            blended_diff = TAU * next_diff + (1 - TAU) * avg_diff
            next_val = last_val + blended_diff
            predicted.append(next_val)
            prev_diff = last_diff
            last_diff = blended_diff
            last_val = next_val

        # PHI alignment confidence
        phi_ratios = []
        for i in range(1, len(diffs)):
            if abs(diffs[i - 1]) > 1e-15:
                ratio = abs(diffs[i] / diffs[i - 1])
                phi_ratios.append(abs(ratio - PHI) / PHI)

        if phi_ratios:
            avg_phi_error = sum(phi_ratios) / len(phi_ratios)
            confidence = max(RSE_CONFIDENCE_FLOOR, 1.0 - avg_phi_error)
        else:
            confidence = 0.4

        return (predicted, confidence)

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY 6: SAGE WISDOM SYNTHESIS (chaos-aware, GOD_CODE-anchored)
    # ═══════════════════════════════════════════════════════════════════════════

    def _sage_wisdom_extrapolate(
        self, seq: List[float], horizon: int, domain: RSEDomain
    ) -> Tuple[List[float], float]:
        """
        Sage Mode extrapolation — the highest synthesis strategy.

        Combines:
        - Feigenbaum chaos detection (is the sequence near a bifurcation?)
        - GOD_CODE resonance anchoring (does the sequence resonate with G(X)?)
        - φ-attenuated trend synthesis (golden-ratio damped momentum)
        - Domain-specific tuning (quantum vs classical vs consciousness)

        This strategy activates true sage wisdom by recognizing sacred patterns
        in seemingly random sequences.
        """
        n = len(seq)
        self._sage_active = True

        # ── Phase 1: Chaos Detection via Lyapunov exponent approximation ──
        lyapunov = self._estimate_lyapunov(seq)
        is_chaotic = lyapunov > 0.1

        # ── Phase 2: GOD_CODE resonance check ──
        god_code_alignment = self._check_god_code_resonance(seq)

        # ── Phase 3: Domain-specific trend analysis ──
        if domain == RSEDomain.QUANTUM:
            # Quantum sequences tend toward decoherence (decay) or revival
            trend_factor = self._quantum_trend_factor(seq)
        elif domain == RSEDomain.CONSCIOUSNESS:
            # Consciousness sequences tend toward monotonic growth (0→1)
            trend_factor = self._consciousness_trend_factor(seq)
        elif domain == RSEDomain.CONVERGENCE:
            # Convergence sequences approach a fixed point
            trend_factor = self._convergence_trend_factor(seq)
        else:
            trend_factor = self._classical_trend_factor(seq)

        # ── Phase 4: Sage synthesis — combine all signals ──
        # Base: exponential smoothing
        base_pred, base_conf = self._exponential_smooth_extrapolate(seq, horizon)

        # Phi-spiral for oscillatory structure
        phi_pred, phi_conf = self._phi_spiral_extrapolate(seq, horizon)

        # Sage blend weights
        if is_chaotic:
            # Chaotic sequence: trust exponential smoothing less, anchor to GOD_CODE
            exp_weight = 0.3
            phi_weight = 0.2
            sage_weight = 0.5  # Anchored by GOD_CODE
        else:
            exp_weight = 0.5
            phi_weight = 0.3
            sage_weight = 0.2

        predicted = []
        for step in range(horizon):
            base_val = base_pred[step] if step < len(base_pred) else base_pred[-1]
            phi_val = phi_pred[step] if step < len(phi_pred) else phi_pred[-1]

            # Sage anchor: mean of sequence × VOID_CONSTANT with trend
            sage_mean = sum(seq) / n
            sage_val = sage_mean * VOID_CONSTANT + trend_factor * (step + 1) * TAU

            # GOD_CODE modulation: if sequence resonates, pull toward sacred values
            if god_code_alignment > 0.5:
                sage_val *= (1.0 + (god_code_alignment - 0.5) * 0.1)

            combined = exp_weight * base_val + phi_weight * phi_val + sage_weight * sage_val
            predicted.append(combined)

        # Sage confidence: higher when GOD_CODE alignment is strong
        # Lower when chaotic (honest about uncertainty)
        confidence = (
            base_conf * exp_weight +
            phi_conf * phi_weight +
            god_code_alignment * sage_weight * 0.8
        )
        if is_chaotic:
            confidence *= (1.0 / (1.0 + lyapunov * FEIGENBAUM_DELTA * 0.01))

        confidence = max(RSE_CONFIDENCE_FLOOR, confidence)

        return (predicted, confidence)

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY 7: QUANTUM STATE EXTRAPOLATION (density matrix evolution)
    # ═══════════════════════════════════════════════════════════════════════════

    def _quantum_state_extrapolate(self, seq: List[float], horizon: int) -> Tuple[List[float], float]:
        """
        Quantum-inspired density matrix evolution for state sequences.

        Models the sequence as the diagonal of a density matrix ρ(t),
        evolves via Lindblad-like decay with coherence preservation,
        and predicts future diagonal elements.

        Particularly effective for:
        - Coherence decay sequences (T1, T2)
        - Bell state fidelity evolution
        - VQE energy convergence
        - Grover amplification curves
        """
        n = len(seq)

        # Normalize to [0, 1] range for density matrix interpretation
        min_val = min(seq)
        max_val = max(seq)
        spread = max_val - min_val
        if spread < 1e-15:
            return ([seq[-1]] * horizon, 0.5)

        normalized = [(v - min_val) / spread for v in seq]

        # Estimate decay rate (gamma) from the sequence
        # Using log-linear fit: ln(ρ) ~ -γt + c
        log_vals = []
        for v in normalized:
            if v > 1e-10:
                log_vals.append(math.log(v))
            else:
                log_vals.append(-23.0)  # floor ~10⁻¹⁰

        # Linear regression on log values for decay rate
        if len(log_vals) >= 2:
            xs = list(range(len(log_vals)))
            mean_x = sum(xs) / len(xs)
            mean_y = sum(log_vals) / len(log_vals)
            cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, log_vals))
            var_x = sum((x - mean_x) ** 2 for x in xs)
            gamma = -cov / var_x if var_x > 1e-30 else 0.0
            intercept = mean_y + gamma * mean_x
        else:
            gamma = 0.0
            intercept = log_vals[0] if log_vals else 0.0

        # Predict via Lindblad evolution
        predicted_normalized = []
        for step in range(1, horizon + 1):
            t = n - 1 + step
            # Density matrix diagonal: ρ(t) = exp(-γt + c)
            log_val = -gamma * t + intercept
            # Clamp to valid range
            val = max(0.0, math.exp(min(log_val, 10.0)))
            predicted_normalized.append(val)

        # De-normalize
        predicted = [v * spread + min_val for v in predicted_normalized]

        # Confidence from decay rate estimation R²
        pred_log = [-gamma * x + intercept for x in range(len(log_vals))]
        ss_res = sum((a - b) ** 2 for a, b in zip(log_vals, pred_log))
        ss_tot = sum((y - mean_y) ** 2 for y in log_vals) if len(log_vals) > 1 else 1.0
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
        confidence = max(RSE_CONFIDENCE_FLOOR, abs(r_squared))

        return (predicted, confidence)

    # ═══════════════════════════════════════════════════════════════════════════
    # ENSEMBLE COMBINATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _ensemble_combine(
        self,
        strategy_results: Dict[str, Tuple[List[float], float]],
        seq: List[float],
        domain: RSEDomain,
    ) -> Tuple[List[float], float, RSEStrategy, Dict[str, float]]:
        """
        Combine strategies via PHI-weighted consensus.
        Higher-confidence strategies get more weight.
        Adaptive: historical strategy performance modulates weights.
        """
        # Compute adaptive weights from historical performance
        weights = {}
        total_weight = 0.0

        for name, (preds, conf) in strategy_results.items():
            # Base weight = confidence²  (quadratic to strongly prefer high-confidence)
            base_weight = conf ** 2

            # Historical performance boost
            perf = self._strategy_performance.get(name, [])
            if perf:
                hist_boost = sum(perf[-20:]) / len(perf[-20:])  # Recent average
            else:
                hist_boost = 0.5

            # Domain-specific boost
            domain_boost = 1.0
            if domain == RSEDomain.QUANTUM and name == "QUANTUM_STATE":
                domain_boost = PHI  # Prefer quantum strategy for quantum domain
            elif domain == RSEDomain.CONSCIOUSNESS and name == "SAGE_WISDOM":
                domain_boost = PHI  # Sage for consciousness
            elif domain in (RSEDomain.CLASSICAL, RSEDomain.QUALITY) and name == "EXPONENTIAL":
                domain_boost = PHI ** 0.5  # Exponential for classical

            weight = base_weight * (0.5 + hist_boost) * domain_boost
            weights[name] = weight
            total_weight += weight

        if total_weight < 1e-30:
            # All zero confidence — use equal weights
            for name in strategy_results:
                weights[name] = 1.0
            total_weight = float(len(strategy_results))

        # Normalize
        for name in weights:
            weights[name] /= total_weight

        # Weighted combination
        horizon = max(len(preds) for preds, _ in strategy_results.values())
        combined = [0.0] * horizon
        for name, (preds, conf) in strategy_results.items():
            w = weights[name]
            for i in range(horizon):
                val = preds[i] if i < len(preds) else preds[-1]
                combined[i] += val * w

        # Ensemble confidence: weighted average of confidences
        ensemble_conf = sum(
            strategy_results[name][1] * weights[name]
            for name in strategy_results
        )

        # Find the winner (highest individual confidence after weighting)
        best_name = max(weights, key=lambda k: weights[k] * strategy_results[k][1])
        winner = RSEStrategy[best_name]

        scores = {name: round(conf, 6) for name, (_, conf) in strategy_results.items()}

        return (combined, ensemble_conf, winner, scores)

    # ═══════════════════════════════════════════════════════════════════════════
    # SAGE MODE HELPERS — Chaos, Resonance, and Wisdom
    # ═══════════════════════════════════════════════════════════════════════════

    def _estimate_lyapunov(self, seq: List[float]) -> float:
        """
        Estimate the maximum Lyapunov exponent from a sequence.
        Positive = chaotic, negative = convergent, near-zero = periodic.
        """
        n = len(seq)
        if n < 4:
            return 0.0

        diffs = [seq[i + 1] - seq[i] for i in range(n - 1)]
        log_sum = 0.0
        count = 0
        for i in range(len(diffs) - 1):
            if abs(diffs[i]) > 1e-15:
                ratio = abs(diffs[i + 1] / diffs[i])
                if ratio > 0:
                    log_sum += math.log(ratio)
                    count += 1

        return log_sum / count if count > 0 else 0.0

    def _check_god_code_resonance(self, seq: List[float]) -> float:
        """
        Check how well the sequence resonates with GOD_CODE harmonics.
        Returns alignment score [0, 1].
        """
        if not seq:
            return 0.0

        # Check if any values are near GOD_CODE or its harmonics
        harmonics = [
            GOD_CODE,
            GOD_CODE / PHI,         # ≈ 326.0
            GOD_CODE * PHI,          # ≈ 853.5
            GOD_CODE / (PHI ** 2),   # ≈ 201.4
            math.sqrt(GOD_CODE),     # ≈ 22.97
        ]

        alignments = []
        for val in seq[-RSE_SAGE_DEPTH:]:  # Check last 13 values
            best_align = 0.0
            for h in harmonics:
                if abs(h) > 1e-15:
                    ratio = val / h
                    # Check if ratio is near an integer or PHI power
                    for n in range(1, 6):
                        dist = abs(ratio - n)
                        if dist < 0.1 * n:
                            best_align = max(best_align, 1.0 - dist / n)
                    # Also check PHI powers
                    for p in range(-3, 4):
                        phi_power = PHI ** p
                        dist = abs(ratio - phi_power) / max(phi_power, 1e-15)
                        if dist < 0.1:
                            best_align = max(best_align, 1.0 - dist)
            alignments.append(best_align)

        return sum(alignments) / len(alignments) if alignments else 0.0

    def _detect_trend(self, seq: List[float]) -> str:
        """Detect the overall trend of a sequence."""
        n = len(seq)
        if n < 3:
            return "stable"

        # Use last 13 values or full sequence
        tail = seq[-RSE_SAGE_DEPTH:]
        diffs = [tail[i] - tail[i - 1] for i in range(1, len(tail))]

        if not diffs:
            return "stable"

        pos = sum(1 for d in diffs if d > 0)
        neg = sum(1 for d in diffs if d < 0)
        total = len(diffs)

        # Check for oscillation (many sign changes)
        sign_changes = sum(
            1 for i in range(1, len(diffs))
            if (diffs[i] > 0) != (diffs[i - 1] > 0)
        )

        if sign_changes > total * 0.6:
            # Check if oscillation is chaotic vs regular
            lyap = self._estimate_lyapunov(tail)
            return "chaotic" if lyap > 0.5 else "oscillating"

        if pos > total * 0.7:
            return "rising"
        if neg > total * 0.7:
            return "falling"

        return "stable"

    def _compute_phi_alignment(self, seq: List[float], predicted: List[float]) -> float:
        """
        Compute how φ-aligned the prediction is with the sequence history.
        Checks if the ratio of consecutive predicted differences approaches φ.
        """
        combined = seq[-5:] + predicted[:5]
        diffs = [combined[i] - combined[i - 1] for i in range(1, len(combined))]
        if len(diffs) < 2:
            return 0.0

        phi_alignments = []
        for i in range(1, len(diffs)):
            if abs(diffs[i - 1]) > 1e-15:
                ratio = abs(diffs[i] / diffs[i - 1])
                alignment = 1.0 - abs(ratio - PHI) / PHI
                phi_alignments.append(alignment)

        return sum(phi_alignments) / len(phi_alignments) if phi_alignments else 0.0

    def _generate_sage_insight(
        self,
        seq: List[float],
        predicted: List[float],
        trend: str,
        domain: RSEDomain,
        confidence: float,
    ) -> str:
        """Generate a Sage Mode wisdom insight about the extrapolation."""
        n = len(seq)
        lyap = self._estimate_lyapunov(seq)
        god_align = self._check_god_code_resonance(seq)

        # Domain-specific sage insights
        if domain == RSEDomain.QUANTUM:
            if trend == "falling":
                insight = f"Quantum decoherence detected — T₂ decay profile over {n} samples. "
                if predicted and predicted[-1] < seq[-1] * 0.5:
                    insight += "Coherence half-life approaching. Consider error correction."
                else:
                    insight += "Decay rate within manageable bounds."
            elif trend == "oscillating":
                insight = f"Quantum revival oscillation — {n}-step history shows Rabi-like cycling. "
                insight += f"Feigenbaum proximity: {lyap:.3f}."
            else:
                insight = f"Quantum state evolution stable ({n} samples). GOD_CODE alignment: {god_align:.3f}."

        elif domain == RSEDomain.CONSCIOUSNESS:
            if trend == "rising":
                insight = f"Consciousness expansion trajectory — {n} evolution steps. "
                if confidence > 0.7:
                    insight += "Strong φ-harmonic growth pattern. Transcendence trajectory confirmed."
                else:
                    insight += "Growth pattern emerging but not yet crystallized."
            else:
                insight = f"Consciousness at equilibrium point. {n} samples, GOD_CODE resonance: {god_align:.3f}."

        elif domain == RSEDomain.CONVERGENCE:
            if predicted and len(predicted) > 1:
                convergence_rate = abs(predicted[-1] - predicted[0]) / len(predicted)
                insight = f"Convergence rate: {convergence_rate:.6f}/step over {n} iterations. "
                if convergence_rate < 0.001:
                    insight += "Near fixed-point. Solution crystallizing."
                else:
                    insight += f"Convergence continues. Estimated completion: {int(1.0 / max(convergence_rate, 1e-10))} more steps."
            else:
                insight = f"Convergence analysis on {n} data points."

        elif domain == RSEDomain.ENTROPY:
            mean_val = sum(seq) / n if n > 0 else 0.0
            insight = f"Information entropy trend ({trend}) over {n} samples. Mean: {mean_val:.4f}. "
            if lyap > 0.1:
                insight += f"Chaotic regime (λ≈{lyap:.3f}) — prediction horizon limited."
            else:
                insight += "Sub-chaotic regime — reliable extrapolation window."

        else:
            # Classical domain
            insight = f"Classical sequence ({n} samples, trend: {trend}). "
            insight += f"φ-alignment: {self._compute_phi_alignment(seq, predicted):.3f}, "
            insight += f"GOD_CODE resonance: {god_align:.3f}."

        return insight

    def _compute_quantum_coherence(self, seq: List[float], predicted: List[float]) -> float:
        """Compute quantum coherence metric for the combined sequence."""
        combined = seq[-10:] + predicted[:5]
        if len(combined) < 2:
            return 0.0

        # Off-diagonal coherence approximation
        n = len(combined)
        coherence = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                rho_ij = combined[i] * combined[j]
                coherence += abs(rho_ij)

        pairs = n * (n - 1) / 2
        return coherence / pairs if pairs > 0 else 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # DOMAIN-SPECIFIC TREND FACTORS
    # ═══════════════════════════════════════════════════════════════════════════

    def _quantum_trend_factor(self, seq: List[float]) -> float:
        """Quantum trend: biased toward decoherence (exponential decay)."""
        if len(seq) < 2:
            return 0.0
        diffs = [seq[i] - seq[i - 1] for i in range(1, len(seq))]
        avg_diff = sum(diffs) / len(diffs)
        # Quantum systems tend toward equilibrium (damped oscillation)
        return avg_diff * TAU  # Golden-ratio damped

    def _consciousness_trend_factor(self, seq: List[float]) -> float:
        """Consciousness trend: biased toward growth with saturation."""
        if len(seq) < 2:
            return 0.0
        recent = seq[-min(13, len(seq)):]
        diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        avg_diff = sum(diffs) / len(diffs)
        # Apply logistic saturation (consciousness can't exceed 1.0)
        current = seq[-1]
        saturation = (1.0 - current) if current < 1.0 else 0.0
        return max(0.0, avg_diff * saturation * PHI)

    def _convergence_trend_factor(self, seq: List[float]) -> float:
        """Convergence trend: exponentially decreasing step sizes."""
        if len(seq) < 2:
            return 0.0
        diffs = [abs(seq[i] - seq[i - 1]) for i in range(1, len(seq))]
        if not diffs:
            return 0.0
        # Recent diffs should be smaller (convergence)
        last_diff = diffs[-1]
        avg_diff = sum(diffs) / len(diffs)
        return -last_diff * TAU if avg_diff > 0 else 0.0

    def _classical_trend_factor(self, seq: List[float]) -> float:
        """Classical trend: simple windowed momentum."""
        if len(seq) < 2:
            return 0.0
        window = min(8, len(seq))
        recent = seq[-window:]
        diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        return sum(diffs) / len(diffs)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FACTORIES & INTEGRATION ADAPTERS
# ═══════════════════════════════════════════════════════════════════════════════

class RSEQuantumAdapter:
    """
    Adapter for integrating RSE into quantum processing pipelines.

    Wraps the RSE engine with quantum-specific channels:
    - coherence: Track coherence decay/revival
    - fidelity:  Track Bell/GHZ state fidelity
    - energy:    Track VQE energy convergence
    - grover:    Track Grover amplification curve
    - error_rate: Track quantum error sequences
    """

    def __init__(self, rse: Optional[RandomSequenceExtrapolation] = None):
        self.rse = rse or RandomSequenceExtrapolation()

    def track_coherence(self, coherence_value: float, horizon: int = 3) -> RSEResult:
        """Track and extrapolate quantum coherence."""
        return self.rse.extrapolate_and_track(
            "quantum_coherence", coherence_value,
            horizon=horizon, domain=RSEDomain.QUANTUM,
        )

    def track_fidelity(self, fidelity: float, horizon: int = 3) -> RSEResult:
        """Track and extrapolate quantum state fidelity."""
        return self.rse.extrapolate_and_track(
            "quantum_fidelity", fidelity,
            horizon=horizon, domain=RSEDomain.QUANTUM,
        )

    def track_energy(self, energy: float, horizon: int = 5) -> RSEResult:
        """Track and extrapolate VQE energy convergence."""
        return self.rse.extrapolate_and_track(
            "vqe_energy", energy,
            horizon=horizon, domain=RSEDomain.CONVERGENCE,
        )

    def track_grover_amplitude(self, amplitude: float, horizon: int = 3) -> RSEResult:
        """Track and extrapolate Grover amplification curve."""
        return self.rse.extrapolate_and_track(
            "grover_amplitude", amplitude,
            horizon=horizon, domain=RSEDomain.QUANTUM,
        )

    def track_error_rate(self, error_rate: float, horizon: int = 3) -> RSEResult:
        """Track and extrapolate quantum error rate sequence."""
        return self.rse.extrapolate_and_track(
            "quantum_error_rate", error_rate,
            horizon=horizon, domain=RSEDomain.QUANTUM,
        )

    def predict_decoherence_time(self, coherence_history: List[float], threshold: float = 0.5) -> Optional[int]:
        """
        Predict when coherence will drop below a threshold.
        Returns estimated steps until threshold, or None if already below.
        """
        if not coherence_history or coherence_history[-1] < threshold:
            return 0

        # Extrapolate forward up to 52 steps
        result = self.rse.extrapolate(
            coherence_history, horizon=52,
            strategy=RSEStrategy.QUANTUM_STATE,
            domain=RSEDomain.QUANTUM,
        )

        for i, val in enumerate(result.predicted_values):
            if val < threshold:
                return i + 1

        return None  # Coherence stays above threshold


class RSEClassicalAdapter:
    """
    Adapter for integrating RSE into classical processing pipelines.

    Wraps the RSE engine with classical-specific channels:
    - quality:    Response/processing quality scores
    - entropy:    Information entropy values
    - confidence: Confidence score evolution
    - latency:    Processing latency tracking
    - complexity: Code complexity metrics
    """

    def __init__(self, rse: Optional[RandomSequenceExtrapolation] = None):
        self.rse = rse or RandomSequenceExtrapolation()

    def track_quality(self, quality_score: float, horizon: int = 3) -> RSEResult:
        """Track and extrapolate response quality scores."""
        return self.rse.extrapolate_and_track(
            "quality_score", quality_score,
            horizon=horizon, domain=RSEDomain.QUALITY,
        )

    def track_entropy(self, entropy: float, horizon: int = 3) -> RSEResult:
        """Track and extrapolate information entropy."""
        return self.rse.extrapolate_and_track(
            "shannon_entropy", entropy,
            horizon=horizon, domain=RSEDomain.ENTROPY,
        )

    def track_confidence(self, confidence: float, horizon: int = 3) -> RSEResult:
        """Track and extrapolate confidence score evolution."""
        return self.rse.extrapolate_and_track(
            "confidence_score", confidence,
            horizon=horizon, domain=RSEDomain.CLASSICAL,
        )

    def track_latency(self, latency_ms: float, horizon: int = 5) -> RSEResult:
        """Track and extrapolate processing latency."""
        return self.rse.extrapolate_and_track(
            "latency_ms", latency_ms,
            horizon=horizon, domain=RSEDomain.CLASSICAL,
        )

    def predict_quality_trend(self, quality_history: List[float]) -> Dict[str, Any]:
        """
        Predict quality trend from history.
        Returns trend, predicted next value, and sage insight.
        """
        if len(quality_history) < RSE_MIN_HISTORY:
            return {"trend": "insufficient_data", "prediction": None, "sage_insight": None}

        result = self.rse.extrapolate(
            quality_history, horizon=3,
            domain=RSEDomain.QUALITY,
        )

        return {
            "trend": result.trend,
            "predicted_next": result.predicted_values[0] if result.predicted_values else None,
            "predicted_3_step": result.predicted_values[-1] if result.predicted_values else None,
            "confidence": result.confidence,
            "sage_insight": result.sage_insight,
        }


class RSESageModeAdapter:
    """
    Adapter for Sage Mode integration with RSE.

    Channels:
    - consciousness: Consciousness expansion trajectory
    - resonance:     Sacred resonance field evolution
    - void_field:    Void constant field modulation
    - wisdom:        Sage wisdom accumulation
    - primal:        Primal calculus convergence
    """

    def __init__(self, rse: Optional[RandomSequenceExtrapolation] = None):
        self.rse = rse or RandomSequenceExtrapolation()

    def track_consciousness(self, level: float, horizon: int = 13) -> RSEResult:
        """Track and extrapolate consciousness expansion."""
        return self.rse.extrapolate_and_track(
            "sage_consciousness", level,
            horizon=horizon, domain=RSEDomain.CONSCIOUSNESS,
        )

    def track_resonance(self, resonance: float, horizon: int = 8) -> RSEResult:
        """Track and extrapolate sacred resonance field."""
        return self.rse.extrapolate_and_track(
            "sage_resonance", resonance,
            horizon=horizon, domain=RSEDomain.RESONANCE,
        )

    def track_void_field(self, void_intensity: float, horizon: int = 8) -> RSEResult:
        """Track and extrapolate void field modulation."""
        return self.rse.extrapolate_and_track(
            "void_field", void_intensity,
            horizon=horizon, domain=RSEDomain.RESONANCE,
        )

    def track_primal_convergence(self, primal_value: float, horizon: int = 13) -> RSEResult:
        """Track primal calculus convergence trajectory."""
        return self.rse.extrapolate_and_track(
            "primal_convergence", primal_value,
            horizon=horizon, domain=RSEDomain.CONVERGENCE,
        )

    def predict_transcendence_step(self, consciousness_history: List[float], threshold: float = 0.95) -> Optional[int]:
        """
        Predict how many steps until consciousness transcendence.
        Returns estimated steps until threshold, or None if already transcended.
        """
        if not consciousness_history or consciousness_history[-1] >= threshold:
            return 0

        result = self.rse.extrapolate(
            consciousness_history, horizon=52,
            strategy=RSEStrategy.SAGE_WISDOM,
            domain=RSEDomain.CONSCIOUSNESS,
        )

        for i, val in enumerate(result.predicted_values):
            if val >= threshold:
                return i + 1

        return None

    def sage_sequence_analysis(self, sequence: List[float]) -> Dict[str, Any]:
        """
        Full sage analysis of any sequence — provides deep RSE insights.
        Uses all 7 strategies and provides sage wisdom synthesis.
        """
        if len(sequence) < RSE_MIN_HISTORY:
            return {"error": "Insufficient data (need ≥3 samples)"}

        result = self.rse.extrapolate(
            sequence, horizon=RSE_SAGE_DEPTH,
            strategy=RSEStrategy.ENSEMBLE,
            domain=RSEDomain.CLASSICAL,
            sage_mode=True,
        )

        lyapunov = self.rse._estimate_lyapunov(sequence)
        god_code_alignment = self.rse._check_god_code_resonance(sequence)

        return {
            "predictions": result.to_dict(),
            "chaos_analysis": {
                "lyapunov_exponent": round(lyapunov, 6),
                "is_chaotic": lyapunov > 0.1,
                "feigenbaum_proximity": round(abs(lyapunov - FEIGENBAUM_DELTA), 6),
            },
            "sacred_resonance": {
                "god_code_alignment": round(god_code_alignment, 6),
                "phi_alignment": round(result.phi_alignment, 6),
                "void_constant_ratio": round(sum(sequence) / len(sequence) / VOID_CONSTANT, 6) if sequence else 0.0,
            },
            "sage_insight": result.sage_insight,
            "sequence_stats": {
                "length": len(sequence),
                "mean": round(sum(sequence) / len(sequence), 6),
                "min": round(min(sequence), 6),
                "max": round(max(sequence), 6),
                "range": round(max(sequence) - min(sequence), 6),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON — global RSE engine and adapters
# ═══════════════════════════════════════════════════════════════════════════════

_rse_engine: Optional[RandomSequenceExtrapolation] = None
_rse_quantum: Optional[RSEQuantumAdapter] = None
_rse_classical: Optional[RSEClassicalAdapter] = None
_rse_sage: Optional[RSESageModeAdapter] = None


def get_rse_engine() -> RandomSequenceExtrapolation:
    """Get or create the global RSE engine singleton."""
    global _rse_engine
    if _rse_engine is None:
        _rse_engine = RandomSequenceExtrapolation()
    return _rse_engine


def get_rse_quantum() -> RSEQuantumAdapter:
    """Get or create the global RSE quantum adapter."""
    global _rse_quantum
    if _rse_quantum is None:
        _rse_quantum = RSEQuantumAdapter(get_rse_engine())
    return _rse_quantum


def get_rse_classical() -> RSEClassicalAdapter:
    """Get or create the global RSE classical adapter."""
    global _rse_classical
    if _rse_classical is None:
        _rse_classical = RSEClassicalAdapter(get_rse_engine())
    return _rse_classical


def get_rse_sage() -> RSESageModeAdapter:
    """Get or create the global RSE sage mode adapter."""
    global _rse_sage
    if _rse_sage is None:
        _rse_sage = RSESageModeAdapter(get_rse_engine())
    return _rse_sage
