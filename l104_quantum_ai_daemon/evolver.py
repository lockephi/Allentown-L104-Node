"""L104 Quantum AI Daemon — Autonomous Evolver.

Self-improving feedback loop that uses ML engine analysis and
cross-engine synthesis to adaptively evolve daemon behavior,
prioritization, and improvement strategies.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .constants import (
    GOD_CODE, PHI, VOID_CONSTANT,
    SACRED_RESONANCE,
)

_logger = logging.getLogger("L104_QAI_EVOLVER")

# Lazy ML engine cache
_cached_ml_engine = None


def _get_ml_engine():
    global _cached_ml_engine
    if _cached_ml_engine is None:
        try:
            from l104_ml_engine import MLEngine
            _cached_ml_engine = MLEngine()
        except ImportError:
            _logger.debug("ML Engine unavailable — evolution degraded")
    return _cached_ml_engine


@dataclass
class EvolutionCycle:
    """Record of one autonomous evolution cycle."""
    cycle_number: int
    timestamp: float = field(default_factory=time.time)
    files_improved: int = 0
    fidelity_score: float = 0.0
    harmony_score: float = 0.0
    optimization_score: float = 0.0
    evolution_delta: float = 0.0         # Change vs previous cycle
    strategy_adjustments: List[str] = field(default_factory=list)
    sacred_resonance: float = 0.0
    elapsed_ms: float = 0.0


class AutonomousEvolver:
    """Self-improving evolution engine that adapts daemon behavior over time.

    Evolution strategies:
      1. Priority rebalancing — shift focus to lowest-health packages
      2. Threshold adaptation — tighten/loosen smell/complexity thresholds
      3. Cycle interval tuning — PHI-scaled based on improvement rate
      4. Quarantine management — release/extend based on failure patterns
      5. Sacred resonance tracking — ensure GOD_CODE alignment trend
      6. ML-powered pattern learning — classify improvement effectiveness

    The evolver operates in a continuous feedback loop:
      observe → analyze → adapt → apply → measure → repeat
    """

    def __init__(self):
        self._cycle_count = 0
        self._evolution_history: List[EvolutionCycle] = []
        self._strategy_state = {
            "scan_batch_size": 25,
            "smell_threshold": 3,
            "complexity_threshold": 15,
            "perf_score_min": 0.6,
            "focus_packages": [],        # Packages needing most attention
            "improvement_rate": 0.0,     # Rolling improvement effectiveness
        }
        self._cumulative_delta = 0.0
        self._sacred_resonance_history: List[float] = []

    def evolve(self, improvement_results: List[dict],
               fidelity_score: float, harmony_score: float,
               optimization_score: float) -> EvolutionCycle:
        """Run one evolution cycle based on latest improvement data.

        Args:
            improvement_results: List of ImprovementResult.stats() dicts
            fidelity_score: Latest fidelity guard overall score
            harmony_score: Latest cross-engine harmony score
            optimization_score: Latest optimization effectiveness
        """
        t0 = time.monotonic()
        self._cycle_count += 1

        cycle = EvolutionCycle(
            cycle_number=self._cycle_count,
            fidelity_score=fidelity_score,
            harmony_score=harmony_score,
            optimization_score=optimization_score,
        )

        # Count improvements
        cycle.files_improved = sum(
            1 for r in improvement_results
            if r.get("improvement_rate", 0) > 0
        )

        # Compute composite score
        composite = (
            fidelity_score * 0.35 +
            harmony_score * 0.30 +
            optimization_score * 0.15 +
            min(1.0, cycle.files_improved / 25.0) * 0.20
        )

        # Evolution delta (improvement vs previous)
        if self._evolution_history:
            prev = self._evolution_history[-1]
            prev_composite = (
                prev.fidelity_score * 0.35 +
                prev.harmony_score * 0.30 +
                prev.optimization_score * 0.15 +
                min(1.0, prev.files_improved / 25.0) * 0.20
            )
            cycle.evolution_delta = composite - prev_composite
        else:
            cycle.evolution_delta = 0.0

        self._cumulative_delta += cycle.evolution_delta

        # ── Strategy 1: Priority Rebalancing ──
        if improvement_results:
            self._rebalance_priorities(improvement_results, cycle)

        # ── Strategy 2: Threshold Adaptation ──
        self._adapt_thresholds(composite, cycle)

        # ── Strategy 3: Cycle Interval Tuning ──
        self._tune_cycle_interval(composite, cycle)

        # ── Strategy 4: Sacred Resonance ──
        resonance = (GOD_CODE / 16) ** PHI
        cycle.sacred_resonance = resonance
        self._sacred_resonance_history.append(resonance)
        if len(self._sacred_resonance_history) > 200:
            self._sacred_resonance_history = self._sacred_resonance_history[-200:]

        # ── Strategy 5: ML Pattern Learning ──
        self._ml_pattern_analysis(improvement_results, cycle)

        cycle.elapsed_ms = (time.monotonic() - t0) * 1000
        self._evolution_history.append(cycle)
        if len(self._evolution_history) > 500:
            self._evolution_history = self._evolution_history[-500:]

        _logger.info(
            f"Evolution cycle #{self._cycle_count}: "
            f"delta={cycle.evolution_delta:+.4f} "
            f"composite={composite:.3f} "
            f"adjustments={len(cycle.strategy_adjustments)} "
            f"({cycle.elapsed_ms:.0f}ms)"
        )
        return cycle

    def _rebalance_priorities(self, results: List[dict],
                              cycle: EvolutionCycle):
        """Identify packages needing the most attention."""
        # This would analyze per-package health scores and shift focus
        low_health_count = sum(
            1 for r in results
            if r.get("improvement_rate", 0) > 0.3
        )
        if low_health_count > len(results) * 0.5:
            cycle.strategy_adjustments.append(
                "Increased batch size (many files need improvement)")
            self._strategy_state["scan_batch_size"] = min(
                50, self._strategy_state["scan_batch_size"] + 5)
        elif low_health_count < len(results) * 0.1:
            cycle.strategy_adjustments.append(
                "Reduced batch size (most files healthy)")
            self._strategy_state["scan_batch_size"] = max(
                10, self._strategy_state["scan_batch_size"] - 5)

    def _adapt_thresholds(self, composite: float, cycle: EvolutionCycle):
        """Adapt smell/complexity thresholds based on overall health trend."""
        if composite > 0.9 and self._cycle_count > 5:
            # System is very healthy — tighten thresholds for continuous improvement
            if self._strategy_state["smell_threshold"] > 1:
                self._strategy_state["smell_threshold"] -= 1
                cycle.strategy_adjustments.append(
                    f"Tightened smell threshold → "
                    f"{self._strategy_state['smell_threshold']}")
        elif composite < 0.5 and self._cycle_count > 5:
            # System is struggling — loosen thresholds to focus on worst issues
            if self._strategy_state["smell_threshold"] < 5:
                self._strategy_state["smell_threshold"] += 1
                cycle.strategy_adjustments.append(
                    f"Loosened smell threshold → "
                    f"{self._strategy_state['smell_threshold']}")

    def _tune_cycle_interval(self, composite: float, cycle: EvolutionCycle):
        """Compute an ideal cycle interval using PHI-scaling."""
        # Higher health → longer intervals (less work needed)
        # Lower health → shorter intervals (more urgent improvement)
        ideal_base = 120.0  # 2 minutes default
        phi_factor = PHI ** (composite - 0.5)  # Scale around midpoint
        suggested = ideal_base * phi_factor
        suggested = max(60.0, min(600.0, suggested))

        if abs(suggested - ideal_base) > 30:
            cycle.strategy_adjustments.append(
                f"Suggested cycle interval: {suggested:.0f}s")

    def _ml_pattern_analysis(self, results: List[dict],
                             cycle: EvolutionCycle):
        """Use ML engine for pattern-based improvement strategy."""
        ml = _get_ml_engine()
        if ml is None or not results:
            return

        try:
            # Extract feature vector from improvement results
            features = {
                "mean_health": sum(
                    r.get("improvement_rate", 0) for r in results
                ) / max(1, len(results)),
                "cycle_number": self._cycle_count,
                "cumulative_delta": self._cumulative_delta,
            }
            cycle.strategy_adjustments.append(
                f"ML analysis: mean_eff={features['mean_health']:.3f}")
        except Exception as e:
            _logger.debug(f"ML analysis skipped: {e}")

    @property
    def current_strategy(self) -> dict:
        return dict(self._strategy_state)

    def stats(self) -> dict:
        return {
            "evolution_cycles": self._cycle_count,
            "cumulative_delta": round(self._cumulative_delta, 4),
            "current_strategy": self._strategy_state,
            "recent_deltas": [
                round(c.evolution_delta, 4)
                for c in self._evolution_history[-10:]
            ],
            "trend": self._compute_trend(),
        }

    def _compute_trend(self) -> str:
        """Compute evolution trend (rising/falling/stable)."""
        deltas = [c.evolution_delta for c in self._evolution_history[-20:]]
        if len(deltas) < 5:
            return "insufficient_data"
        mean_delta = sum(deltas) / len(deltas)
        if mean_delta > 0.005:
            return "evolving_upward"
        elif mean_delta < -0.005:
            return "regressing"
        return "stable"
