"""
Adaptive Optimization Engine — v9.2.0
======================================

Provides dynamic latency targeting, memory budget management, and cascading failure recovery
for the ASI Core pipeline. Adapts optimization strategies based on runtime performance metrics
and system resource availability.

Features:
  - Adaptive latency targeting per pipeline stage
  - Memory budget tracking and optimization
  - Cascading failure recovery with retry adaptation
  - Pipeline-wide orchestration and health monitoring

Author: L104 Sovereign Node
Version: 9.2.0
License: Proprietary — L104 Node
"""

import time
import psutil
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import logging

from .constants import (
    TARGET_LATENCY_LCE_MS,
    TARGET_LATENCY_QE_MS,
    TARGET_LATENCY_SM_MS,
    TARGET_LATENCY_COHERENCE_MS,
    TARGET_LATENCY_ACTIVATION_MS,
    TIMEOUT_NORMAL_MULTIPLIER,
    TIMEOUT_DEGRADED_MULTIPLIER,
    TIMEOUT_CRITICAL_MULTIPLIER,
    MAX_MEMORY_PERCENT_ASI,
    MEMORY_SAFETY_MARGIN_PCT,
    ADAPTIVE_MEMORY_THRESHOLD_PCT,
    CASCADE_MAX_RETRY_ADAPTIVE,
    CASCADE_CONFIDENCE_THRESHOLD,
    CASCADE_FAILURE_THRESHOLD,
    SCORE_AGGREGATION_MODE,
    SCORE_HARMONIC_WEIGHT,
    SCORE_OUTLIER_DETECTION,
    SCORE_OUTLIER_SIGMA,
    PHI,
)

logger = logging.getLogger(__name__)


class StageType(Enum):
    """Pipeline stage classification."""
    LCE = "language_code_engineering"
    QUANTUM = "quantum_engineering"
    SYMBOLIC_MATH = "symbolic_math"
    COHERENCE = "coherence_alignment"
    ACTIVATION = "activation_gate"
    OTHER = "other"


@dataclass
class LatencyInfo:
    """Latency tracking for a single execution."""
    stage: StageType
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    target_ms: float = 50.0
    timeout_ms: float = 150.0
    exceeded_target: bool = False
    exceeded_timeout: bool = False

    def finalize(self, end_time: Optional[float] = None) -> None:
        """Mark execution as complete and compute duration."""
        self.end_time = end_time or time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.exceeded_target = self.duration_ms > self.target_ms
        self.exceeded_timeout = self.duration_ms > self.timeout_ms


@dataclass
class MemorySnapshot:
    """System memory status snapshot."""
    timestamp: float = field(default_factory=time.time)
    total_bytes: int = 0
    available_bytes: int = 0
    used_percent: float = 0.0
    asi_usage_percent: float = 0.0  # Estimated ASI portion
    optimization_required: bool = False


class AdaptiveLatencyTargeter:
    """
    Dynamically targets and adapts stage-level latency goals.

    Monitors actual execution times vs. targets and adjusts timeouts
    to be responsive without being overly strict.
    """

    def __init__(self):
        """Initialize latency targeter."""
        self.latencies: Dict[StageType, List[LatencyInfo]] = {
            stage: [] for stage in StageType
        }
        self.target_ms: Dict[StageType, float] = {
            StageType.LCE: TARGET_LATENCY_LCE_MS,
            StageType.QUANTUM: TARGET_LATENCY_QE_MS,
            StageType.SYMBOLIC_MATH: TARGET_LATENCY_SM_MS,
            StageType.COHERENCE: TARGET_LATENCY_COHERENCE_MS,
            StageType.ACTIVATION: TARGET_LATENCY_ACTIVATION_MS,
            StageType.OTHER: 25.0,
        }
        self.timeout_ms: Dict[StageType, float] = {
            stage: self.target_ms[stage] * TIMEOUT_NORMAL_MULTIPLIER
            for stage in StageType
        }
        self._lock = threading.RLock()
        self.max_history = 100  # Keep moving window

    def record_latency(self, info: LatencyInfo) -> None:
        """Record a latency measurement."""
        with self._lock:
            self.latencies[info.stage].append(info)
            # Trim history to max_history entries
            if len(self.latencies[info.stage]) > self.max_history:
                self.latencies[info.stage] = self.latencies[info.stage][-self.max_history:]

    def get_target_for_stage(self, stage: StageType) -> float:
        """Get current target latency for a stage (in ms)."""
        return self.target_ms.get(stage, 50.0)

    def get_timeout_for_stage(self, stage: StageType, degraded: bool = False,
                              critical: bool = False) -> float:
        """Get adaptive timeout for a stage (in ms)."""
        target = self.get_target_for_stage(stage)
        if critical:
            multiplier = TIMEOUT_CRITICAL_MULTIPLIER
        elif degraded:
            multiplier = TIMEOUT_DEGRADED_MULTIPLIER
        else:
            multiplier = TIMEOUT_NORMAL_MULTIPLIER
        return target * multiplier

    def compute_percentile_latency(self, stage: StageType, percentile: float = 95.0) -> Optional[float]:
        """Compute Pth percentile latency for a stage."""
        with self._lock:
            history = self.latencies.get(stage, [])
            if not history:
                return None
            durations = sorted([l.duration_ms for l in history if l.duration_ms is not None])
            if not durations:
                return None
            idx = int((percentile / 100.0) * len(durations))
            return durations[min(idx, len(durations) - 1)]

    def adapt_timeout_to_degraded(self, stage: StageType) -> None:
        """Increase timeout for a stage that's experiencing degradation."""
        with self._lock:
            current = self.timeout_ms.get(stage, 150.0)
            target = self.get_target_for_stage(stage)
            degraded_timeout = target * TIMEOUT_DEGRADED_MULTIPLIER
            self.timeout_ms[stage] = max(current, degraded_timeout)

    def get_health_report(self) -> Dict[str, Any]:
        """Generate latency health report."""
        report = {}
        with self._lock:
            for stage in StageType:
                history = self.latencies.get(stage, [])
                if not history:
                    continue
                durations = [l.duration_ms for l in history if l.duration_ms is not None]
                if not durations:
                    continue
                exceeded_target = sum(1 for l in history if l.exceeded_target)
                exceeded_timeout = sum(1 for l in history if l.exceeded_timeout)
                report[stage.value] = {
                    "samples": len(durations),
                    "mean_ms": sum(durations) / len(durations),
                    "min_ms": min(durations),
                    "max_ms": max(durations),
                    "p95_ms": self.compute_percentile_latency(stage, 95.0),
                    "exceeded_target": exceeded_target,
                    "exceeded_timeout": exceeded_timeout,
                    "target_ms": self.get_target_for_stage(stage),
                    "timeout_ms": self.timeout_ms.get(stage, 0.0),
                }
        return report


class MemoryBudgetOptimizer:
    """
    Monitors and optimizes system memory usage for ASI.

    Tracks memory consumption, enforces budget limits, and triggers
    optimization when approaching thresholds.
    """

    def __init__(self):
        """Initialize memory optimizer."""
        self.memory_snapshots: List[MemorySnapshot] = []
        self.optimization_callbacks: List[Callable[[], None]] = []
        self._lock = threading.RLock()
        self.max_history = 100

    def record_snapshot(self) -> MemorySnapshot:
        """Record current memory state."""
        snap = MemorySnapshot()
        try:
            mem = psutil.virtual_memory()
            snap.total_bytes = mem.total
            snap.available_bytes = mem.available
            snap.used_percent = mem.percent
            # Estimate ASI usage as fraction of non-available memory
            snap.asi_usage_percent = (mem.total - mem.available) / mem.total * 100
            snap.optimization_required = snap.used_percent > ADAPTIVE_MEMORY_THRESHOLD_PCT
        except Exception as e:
            logger.error(f"Failed to record memory snapshot: {e}")

        with self._lock:
            self.memory_snapshots.append(snap)
            if len(self.memory_snapshots) > self.max_history:
                self.memory_snapshots = self.memory_snapshots[-self.max_history:]

        if snap.optimization_required:
            self._trigger_optimization()

        return snap

    def register_optimization_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to run on optimization trigger."""
        with self._lock:
            self.optimization_callbacks.append(callback)

    def _trigger_optimization(self) -> None:
        """Trigger registered optimization callbacks."""
        with self._lock:
            callbacks = list(self.optimization_callbacks)
        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Optimization callback failed: {e}")

    def get_memory_available_percent(self) -> float:
        """Get current available memory percentage."""
        try:
            return 100.0 - psutil.virtual_memory().percent
        except Exception:
            return 50.0

    def is_memory_constrained(self) -> bool:
        """Check if system memory is constrained."""
        return self.get_memory_available_percent() < (100.0 - ADAPTIVE_MEMORY_THRESHOLD_PCT)

    def get_memory_report(self) -> Dict[str, Any]:
        """Generate memory health report."""
        with self._lock:
            if not self.memory_snapshots:
                return {}
            recent = self.memory_snapshots[-10:]

        avg_used = sum(s.used_percent for s in recent) / len(recent)
        return {
            "total_samples": len(self.memory_snapshots),
            "recent_avg_used_percent": avg_used,
            "is_constrained": self.is_memory_constrained(),
            "max_allowed_percent": MAX_MEMORY_PERCENT_ASI,
            "safety_margin_percent": MEMORY_SAFETY_MARGIN_PCT,
        }


class CascadingOptimizationController:
    """
    Controls cascading retries and failure recovery.

    Manages fallback strategies when stages fail, with adaptive retry
    logic that respects confidence thresholds.
    """

    def __init__(self):
        """Initialize cascading controller."""
        self.retry_attempts: Dict[str, int] = {}
        self.failure_counts: Dict[str, int] = {}
        self._lock = threading.RLock()

    def should_cascade(self, stage_id: str, confidence: float) -> bool:
        """Determine if cascading should be attempted."""
        return confidence >= CASCADE_CONFIDENCE_THRESHOLD

    def should_fail_fast(self, stage_id: str, confidence: float) -> bool:
        """Determine if execution should fail immediately."""
        return confidence < CASCADE_FAILURE_THRESHOLD

    def get_max_retries(self, stage_id: str) -> int:
        """Get max retries for a stage, adapted by history."""
        with self._lock:
            retries = CASCADE_MAX_RETRY_ADAPTIVE
            failure_count = self.failure_counts.get(stage_id, 0)
            # Reduce retries if stage consistently fails
            if failure_count > retries:
                retries = max(1, retries - (failure_count // 2))
            return retries

    def record_retry(self, stage_id: str) -> None:
        """Record a retry attempt."""
        with self._lock:
            self.retry_attempts[stage_id] = self.retry_attempts.get(stage_id, 0) + 1

    def record_failure(self, stage_id: str) -> None:
        """Record a failure."""
        with self._lock:
            self.failure_counts[stage_id] = self.failure_counts.get(stage_id, 0) + 1

    def reset_stage(self, stage_id: str) -> None:
        """Reset counters for a stage after recovery."""
        with self._lock:
            self.retry_attempts[stage_id] = 0
            # Don't reset failure_counts — keep history for learning

    def get_cascade_report(self) -> Dict[str, Any]:
        """Generate cascading failure report."""
        with self._lock:
            return {
                "total_retries": sum(self.retry_attempts.values()),
                "total_failures": sum(self.failure_counts.values()),
                "retries_by_stage": dict(self.retry_attempts),
                "failures_by_stage": dict(self.failure_counts),
            }


class PipelineOptimizationOrchestrator:
    """
    Orchestrates all three optimization subsystems.

    Coordinates latency targeting, memory management, and cascading recovery
    to optimize overall pipeline performance while respecting resource constraints.
    """

    def __init__(self):
        """Initialize orchestrator."""
        self.latency_targeter = AdaptiveLatencyTargeter()
        self.memory_optimizer = MemoryBudgetOptimizer()
        self.cascade_controller = CascadingOptimizationController()
        self._lock = threading.RLock()
        self.optimization_enabled = True

    def enable_optimization(self, enabled: bool = True) -> None:
        """Enable/disable optimization engine."""
        with self._lock:
            self.optimization_enabled = enabled

    def should_optimize_stage(self, stage: StageType) -> bool:
        """Check if a stage should be optimized."""
        if not self.optimization_enabled:
            return False
        # Don't optimize if memory constrained, unless it's latency recovery
        if self.memory_optimizer.is_memory_constrained():
            return stage == StageType.ACTIVATION
        return True

    def get_stage_timeout(self, stage: StageType) -> float:
        """Get adaptive timeout for a stage (in ms)."""
        degraded = self.memory_optimizer.is_memory_constrained()
        return self.latency_targeter.get_timeout_for_stage(stage, degraded=degraded)

    def record_execution(self, info: LatencyInfo) -> None:
        """Record execution results."""
        self.latency_targeter.record_latency(info)
        if info.exceeded_timeout:
            self.cascade_controller.record_failure(info.stage.value)
            self.latency_targeter.adapt_timeout_to_degraded(info.stage)

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status."""
        with self._lock:
            return {
                "enabled": self.optimization_enabled,
                "latency": self.latency_targeter.get_health_report(),
                "memory": self.memory_optimizer.get_memory_report(),
                "cascading": self.cascade_controller.get_cascade_report(),
            }

    def periodic_health_check(self) -> None:
        """Run periodic optimization health checks."""
        self.memory_optimizer.record_snapshot()
        status = self.get_optimization_status()
        logger.debug(f"v9.2.0 Optimization Status: {status}")


# Module-level singleton instance
_orchestrator_instance: Optional[PipelineOptimizationOrchestrator] = None


def get_orchestrator() -> PipelineOptimizationOrchestrator:
    """Get or create the global optimization orchestrator."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = PipelineOptimizationOrchestrator()
    return _orchestrator_instance


__all__ = [
    "StageType",
    "LatencyInfo",
    "MemorySnapshot",
    "AdaptiveLatencyTargeter",
    "MemoryBudgetOptimizer",
    "CascadingOptimizationController",
    "PipelineOptimizationOrchestrator",
    "get_orchestrator",
]
