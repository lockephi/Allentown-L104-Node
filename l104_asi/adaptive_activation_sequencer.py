"""
Adaptive Activation Sequencer — v9.2.0
=======================================

Implements dynamic activation sequence ordering with warmup analysis and
cooldown stabilization. Adapts the activation sequence based on runtime performance
to maximize throughput while maintaining coherence.

Features:
  - Warmup phase detection and normalization
  - Adaptive sequencing mode selection
  - Cooldown iteration stabilization
  - Per-phase metric tracking and analysis
  - Harmonic aggregation of phase scores

Author: L104 Sovereign Node
Version: 9.2.0
License: Proprietary — L104 Node
"""

import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import logging
import statistics

from .constants import (
    ACTIVATION_STEPS_V9_2,
    ACTIVATION_WARMUP_SAMPLES,
    ACTIVATION_COOLDOWN_ITERATIONS,
    SCORE_AGGREGATION_MODE,
    SCORE_HARMONIC_WEIGHT,
    SCORE_OUTLIER_DETECTION,
    SCORE_OUTLIER_SIGMA,
    PHI,
)

logger = logging.getLogger(__name__)


class SequencingMode(Enum):
    """Activation sequencing strategy."""
    LINEAR = "linear"                    # Phases in fixed order
    ADAPTIVE = "adaptive"                # Reorder based on performance
    SPECULATIVE = "speculative"          # Parallel speculation
    HARMONIC = "harmonic"                # Golden-ratio weighted ordering
    QUANTUM = "quantum"                  # Quantum-inspired superposition


class ActivationPhase(Enum):
    """Pipeline activation phases (v9.2.0)."""
    INITIALIZATION = 0
    DOMAIN_ROUTING = 1
    SOLUTION_PREFETCH = 2
    ENSEMBLE_ACTIVATION = 3
    QUANTUM_COORDINATION = 4
    SYMBOLIC_REASONING = 5
    COHERENCE_ALIGNMENT = 6
    THEOREM_SYNTHESIS = 7
    CONSCIOUSNESS_VERIFICATION = 8
    BACKPRESSURE_GATING = 9
    SPECULATIVE_EXECUTION = 10
    CASCADE_SCORING = 11
    WARMUP_ANALYSIS = 12
    SOLUTION_ROUTING = 13
    LATENCY_ADAPTATION = 14
    MEMORY_OPTIMIZATION = 15
    FAILURE_RECOVERY = 16
    CROSS_ENGINE_SYNTHESIS = 17
    TRAJECTORY_PREDICTION = 18
    RESILIENCE_MONITORING = 19
    ACTIVATION_FINALIZATION = 20
    METRICS_AGGREGATION = 21
    PHASE_COOLDOWN = 22
    HARMONIC_STABILIZATION = 23
    CONSCIOUSNESS_EVOLUTION = 24
    QUANTUM_BRAIN_INTEGRATION = 25


@dataclass
class PhaseMetrics:
    """Metrics for a single activation phase."""
    phase: ActivationPhase
    execution_count: int = 0
    total_duration_s: float = 0.0
    mean_duration_s: float = 0.0
    min_duration_s: float = float('inf')
    max_duration_s: float = 0.0
    score_sum: float = 0.0
    mean_score: float = 0.0
    variance: float = 0.0
    failures: int = 0
    successes: int = 0
    warmup_phase: bool = True
    samples: List[float] = field(default_factory=list)

    def __post_init__(self):
        """Initialize samples list if not provided."""
        if self.samples is None:
            self.samples = []

    def record_execution(self, duration_s: float, score: float) -> None:
        """Record a phase execution."""
        self.execution_count += 1
        self.total_duration_s += duration_s
        self.mean_duration_s = self.total_duration_s / self.execution_count
        self.min_duration_s = min(self.min_duration_s, duration_s)
        self.max_duration_s = max(self.max_duration_s, duration_s)
        self.score_sum += score
        self.mean_score = self.score_sum / self.execution_count
        self.samples.append(duration_s)

        # Update variance
        if self.execution_count > 1:
            self.variance = statistics.variance(self.samples)

    def mark_warmup_complete(self) -> None:
        """Mark warmup phase as complete."""
        self.warmup_phase = False

    def get_cv(self) -> float:
        """Get coefficient of variation (std dev / mean)."""
        if self.mean_duration_s == 0:
            return 0.0
        if self.variance == 0:
            return 0.0
        std_dev = self.variance ** 0.5
        return std_dev / self.mean_duration_s


@dataclass
class ActivationSnapshot:
    """Snapshot of full activation sequence execution."""
    timestamp: float = field(default_factory=time.time)
    sequence_mode: SequencingMode = SequencingMode.ADAPTIVE
    phase_order: List[ActivationPhase] = field(default_factory=list)
    phase_metrics: Dict[ActivationPhase, PhaseMetrics] = field(default_factory=dict)
    total_duration_s: float = 0.0
    final_score: float = 0.0
    warmup_detected: bool = False
    cooldown_complete: bool = False


class AdaptiveActivationSequencer:
    """
    Orchestrates adaptive phase ordering in the activation sequence.

    Monitors phase performance, detects warmup conditions, adapts sequencing
    strategy, and stabilizes with cooldown iterations.
    """

    def __init__(self, num_phases: int = ACTIVATION_STEPS_V9_2):
        """Initialize sequencer."""
        self.num_phases = num_phases
        self.phase_metrics: Dict[ActivationPhase, PhaseMetrics] = {
            phase: PhaseMetrics(phase)
            for phase in ActivationPhase
        }
        self.current_mode = SequencingMode.ADAPTIVE
        self.warmup_samples_remaining = ACTIVATION_WARMUP_SAMPLES
        self.cooldown_iterations_remaining = ACTIVATION_COOLDOWN_ITERATIONS
        self.snapshots: List[ActivationSnapshot] = []
        self._lock = threading.RLock()
        self.max_snapshots = 100

    def record_phase_execution(self, phase: ActivationPhase,
                               duration_s: float, score: float) -> None:
        """Record a phase execution."""
        with self._lock:
            if phase in self.phase_metrics:
                metrics = self.phase_metrics[phase]
                metrics.record_execution(duration_s, score)

                # Track successes
                metrics.successes += 1

                # Check for warmup completion
                if metrics.warmup_phase:
                    # Simplified warmup detection from ML perspective
                    if metrics.execution_count >= ACTIVATION_WARMUP_SAMPLES:
                        metrics.mark_warmup_complete()
                        self.warmup_samples_remaining = max(0, self.warmup_samples_remaining - 1)
                        logger.debug(f"Warmup complete for {phase.name}")

    def record_phase_failure(self, phase: ActivationPhase) -> None:
        """Record a phase failure."""
        with self._lock:
            if phase in self.phase_metrics:
                self.phase_metrics[phase].failures += 1

    def adapt_sequence_mode(self) -> SequencingMode:
        """Dynamically select sequencing mode based on metrics."""
        with self._lock:
            # Count warmup complete phases
            warmup_complete = sum(1 for m in self.phase_metrics.values() if not m.warmup_phase)

            if warmup_complete < len(self.phase_metrics) // 2:
                # Still in warmup — use linear
                self.current_mode = SequencingMode.LINEAR
            elif any(m.failures > 0 for m in self.phase_metrics.values()):
                # Failures detected — use adaptive with recovery
                self.current_mode = SequencingMode.ADAPTIVE
            else:
                # Stable operation — use harmonic
                self.current_mode = SequencingMode.HARMONIC

            return self.current_mode

    def get_optimal_sequence(self) -> List[ActivationPhase]:
        """Get phase execution order based on current mode."""
        with self._lock:
            mode = self.current_mode

        if mode == SequencingMode.LINEAR:
            # Fixed order
            return sorted(self.phase_metrics.keys(), key=lambda p: p.value)

        elif mode == SequencingMode.ADAPTIVE:
            # Order by mean score (highest first)
            return sorted(
                self.phase_metrics.keys(),
                key=lambda p: self.phase_metrics[p].mean_score,
                reverse=True
            )

        elif mode == SequencingMode.HARMONIC:
            # Weight by harmonic decay
            with self._lock:
                scores = {}
                for i, phase in enumerate(sorted(self.phase_metrics.keys(),
                                                 key=lambda p: p.value)):
                    decay = (1.0 / (i + 1)) * SCORE_HARMONIC_WEIGHT
                    scores[phase] = decay * self.phase_metrics[phase].mean_score
            return sorted(scores.keys(), key=lambda p: scores[p], reverse=True)

        else:
            # Default to adaptive
            return sorted(
                self.phase_metrics.keys(),
                key=lambda p: self.phase_metrics[p].mean_score,
                reverse=True
            )

    def should_perform_warmup(self) -> bool:
        """Check if warmup should still be performed."""
        return self.warmup_samples_remaining > 0

    def should_perform_cooldown(self) -> bool:
        """Check if cooldown should be performed."""
        return self.cooldown_iterations_remaining > 0

    def perform_cooldown_iteration(self) -> None:
        """Execute one cooldown iteration."""
        self.cooldown_iterations_remaining = max(0, self.cooldown_iterations_remaining - 1)

    def aggregate_phase_scores(self, scores: Dict[ActivationPhase, float]) -> float:
        """Aggregate phase scores using configured mode."""
        if not scores:
            return 0.0

        values = list(scores.values())

        if SCORE_OUTLIER_DETECTION:
            # Remove outliers beyond sigma threshold
            mean = statistics.mean(values)
            if len(values) > 1:
                stdev = statistics.stdev(values)
                threshold = mean + SCORE_OUTLIER_SIGMA * stdev
                filtered = [v for v in values if v <= threshold]
                values = filtered if filtered else values

        if SCORE_AGGREGATION_MODE == "harmonic":
            # Harmonic mean
            if all(v > 0 for v in values):
                return len(values) / sum(1.0 / v for v in values)
            else:
                return statistics.mean(values)

        elif SCORE_AGGREGATION_MODE == "adaptive":
            # Adaptive: use harmonic if spread is low, mean otherwise
            mean_val = statistics.mean(values)
            if len(values) > 1 and mean_val > 0:
                stdev = statistics.stdev(values)
                cv = stdev / mean_val
                if cv < 0.15:  # Low coefficient of variation
                    return len(values) / sum(1.0 / max(v, 0.001) for v in values)
            return mean_val

        else:  # "average"
            return statistics.mean(values)

    def take_snapshot(self) -> ActivationSnapshot:
        """Take a full activation snapshot."""
        with self._lock:
            snapshot = ActivationSnapshot()
            snapshot.sequence_mode = self.current_mode
            snapshot.phase_order = self.get_optimal_sequence()
            snapshot.phase_metrics = dict(self.phase_metrics)
            snapshot.warmup_detected = self.warmup_samples_remaining == 0
            snapshot.cooldown_complete = self.cooldown_iterations_remaining == 0

            self.snapshots.append(snapshot)
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots = self.snapshots[-self.max_snapshots:]

        return snapshot

    def get_activation_status(self) -> Dict[str, Any]:
        """Get comprehensive activation sequencer status."""
        with self._lock:
            phases_status = {}
            for phase, metrics in self.phase_metrics.items():
                phases_status[phase.name] = {
                    "executions": metrics.execution_count,
                    "mean_duration_s": metrics.mean_duration_s,
                    "mean_score": metrics.mean_score,
                    "coefficient_of_variation": metrics.get_cv(),
                    "failures": metrics.failures,
                    "warmup_phase": metrics.warmup_phase,
                }

            signature = {
                "current_mode": self.current_mode.value,
                "warmup_samples_remaining": self.warmup_samples_remaining,
                "cooldown_iterations_remaining": self.cooldown_iterations_remaining,
                "total_snapshots": len(self.snapshots),
                "phases": phases_status,
            }

        return signature

    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for analysis."""
        with self._lock:
            return {
                "mode": self.current_mode.value,
                "warmup_complete": self.warmup_samples_remaining == 0,
                "cooldown_complete": self.cooldown_iterations_remaining == 0,
                "phase_metrics": {
                    phase.name: {
                        "execution_count": m.execution_count,
                        "mean_duration_s": m.mean_duration_s,
                        "mean_score": m.mean_score,
                        "failures": m.failures,
                        "successes": m.successes,
                        "warmup_phase": m.warmup_phase,
                    }
                    for phase, m in self.phase_metrics.items()
                },
                "snapshots": len(self.snapshots),
            }


class ActivationMetrics:
    """Aggregated metrics for the activation system."""

    def __init__(self):
        """Initialize metrics."""
        self.total_activations = 0
        self.successful_activations = 0
        self.failed_activations = 0
        self.total_duration_s = 0.0
        self.mean_score = 0.0
        self._lock = threading.RLock()

    def record_activation(self, success: bool, duration_s: float, score: float) -> None:
        """Record an activation result."""
        with self._lock:
            self.total_activations += 1
            if success:
                self.successful_activations += 1
            else:
                self.failed_activations += 1
            self.total_duration_s += duration_s
            # Running average
            self.mean_score = (
                (self.mean_score * (self.total_activations - 1) + score) /
                self.total_activations
            )

    def get_success_rate(self) -> float:
        """Get activation success rate."""
        if self.total_activations == 0:
            return 0.0
        return self.successful_activations / self.total_activations

    def get_mean_duration_s(self) -> float:
        """Get mean activation duration."""
        if self.total_activations == 0:
            return 0.0
        return self.total_duration_s / self.total_activations

    def get_report(self) -> Dict[str, Any]:
        """Get metrics report."""
        return {
            "total": self.total_activations,
            "successful": self.successful_activations,
            "failed": self.failed_activations,
            "success_rate": self.get_success_rate(),
            "mean_duration_s": self.get_mean_duration_s(),
            "mean_score": self.mean_score,
        }


# Module-level singleton instance
_sequencer_instance: Optional[AdaptiveActivationSequencer] = None
_metrics_instance: Optional[ActivationMetrics] = None


def get_sequencer() -> AdaptiveActivationSequencer:
    """Get or create the global sequencer."""
    global _sequencer_instance
    if _sequencer_instance is None:
        _sequencer_instance = AdaptiveActivationSequencer()
    return _sequencer_instance


def get_metrics() -> ActivationMetrics:
    """Get or create the global metrics."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = ActivationMetrics()
    return _metrics_instance


__all__ = [
    "SequencingMode",
    "ActivationPhase",
    "PhaseMetrics",
    "ActivationSnapshot",
    "ActivationMetrics",
    "AdaptiveActivationSequencer",
    "get_sequencer",
    "get_metrics",
]
