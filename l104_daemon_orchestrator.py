#!/usr/bin/env python3
"""
L104 UNIFIED DAEMON ORCHESTRATOR v1.0.0
═════════════════════════════════════════════════════════════════════════

Coordinates all L104 daemons (VQPU, QuantumAI, Soul) with:
  • Intelligent task scheduling (priority queue, load-aware)
  • Shared resource pool (CPU, memory, I/O)
  • Inter-daemon messaging (events, telemetry, coordination)
  • Global health metrics & anomaly detection
  • Adaptive cycle intervals based on system state
  • Graceful degradation under load

Architecture:
  ┌─────────────────────────────────────────┐
  │   Orchestrator (Central Coordinator)     │
  ├─────────────────────────────────────────┤
  │ • Task Queue (priority, deadline-aware)  │
  │ • Resource Allocator (CPU, Memory, I/O)  │
  │ • Event Bus (inter-daemon messaging)     │
  │ • Health Monitor (global metrics)        │
  │ • State Manager (persisted coordination) │
  └─────────────────────────────────────────┘
       ↓         ↓           ↓
    VQPU    QuantumAI    Soul
    Daemon   Daemon      Daemon
"""

import atexit
import gc
import json
import logging
import os
import psutil
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import heapq

# Phase 3: ML Integration
try:
    from l104_daemon_phase3_ml import (
        workload_predictor,
        anomaly_detector,
        trend_analyzer,
        sacred_scorer,
    )
    PHASE3_ML_AVAILABLE = True
except ImportError:
    PHASE3_ML_AVAILABLE = False

# ═════════════════════════════════════════════════════════════════════════
# CONSTANTS & TYPES
# ═════════════════════════════════════════════════════════════════════════

ORCHESTRATOR_VERSION = "1.0.0"
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497

# State persistence
L104_ROOT = Path(__file__).parent
ORCHESTRATOR_STATE_PATH = L104_ROOT / ".l104_daemon_orchestrator.json"
TELEMETRY_WINDOW = 300  # 5 minutes
HEALTH_STALENESS_DECAY = 0.95  # per cycle

# Resource constraints
CPU_QUOTA_PERCENT = 80.0  # Max CPU usage
MEMORY_QUOTA_MB = 2000    # Max memory per cycle
IO_QUOTA_MBPS = 100.0     # Max I/O throughput


class TaskPriority(int, Enum):
    """Task priority levels (lower = higher priority)."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    DEFERRED = 4


class DaemonType(str, Enum):
    """Known daemon types."""
    VQPU = "vqpu"
    QUANTUM_AI = "quantum_ai"
    SOUL = "soul"


class HealthStatus(str, Enum):
    """Health classification."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class Task:
    """Schedulable task (daemon work unit)."""
    daemon_id: str
    task_type: str
    priority: TaskPriority = TaskPriority.NORMAL
    deadline: Optional[float] = None  # Unix timestamp
    estimated_duration_ms: int = 1000
    retry_count: int = 0
    max_retries: int = 3
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __lt__(self, other):
        """Priority queue ordering: priority first, then deadline."""
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.deadline is not None and other.deadline is not None:
            return self.deadline < other.deadline
        return self.created_at < other.created_at


@dataclass
class DaemonMetrics:
    """Per-daemon performance metrics."""
    daemon_id: str
    cycles_completed: int = 0
    cycles_failed: int = 0
    avg_cycle_ms: float = 0.0
    max_cycle_ms: float = 0.0
    last_cycle_ms: float = 0.0
    health_score: float = 1.0
    fidelity_score: float = 1.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    error_log: deque = field(default_factory=lambda: deque(maxlen=50))
    last_updated: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "daemon_id": self.daemon_id,
            "cycles_completed": self.cycles_completed,
            "cycles_failed": self.cycles_failed,
            "avg_cycle_ms": round(self.avg_cycle_ms, 2),
            "max_cycle_ms": round(self.max_cycle_ms, 2),
            "last_cycle_ms": round(self.last_cycle_ms, 2),
            "health_score": round(self.health_score, 3),
            "fidelity_score": round(self.fidelity_score, 3),
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_mb": round(self.memory_mb, 1),
            "error_count": len(self.error_log),
            "last_updated": self.last_updated,
        }


@dataclass
class SystemMetrics:
    """Global system metrics."""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    active_tasks: int = 0
    queued_tasks: int = 0
    daemon_metrics: Dict[str, DaemonMetrics] = field(default_factory=dict)
    health_status: HealthStatus = HealthStatus.HEALTHY
    cycle_count: int = 0

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_percent": round(self.memory_percent, 1),
            "memory_mb": round(self.memory_mb, 1),
            "active_tasks": self.active_tasks,
            "queued_tasks": self.queued_tasks,
            "daemon_metrics": {k: v.to_dict() for k, v in self.daemon_metrics.items()},
            "health_status": self.health_status.value,
            "cycle_count": self.cycle_count,
        }


# ═════════════════════════════════════════════════════════════════════════
# PHASE 2 ROBUSTNESS ENHANCEMENTS
# ═════════════════════════════════════════════════════════════════════════

class HealthPredictor:
    """Predict health degradation before it happens."""

    def __init__(self, history_window=60):
        self.cpu_history = deque(maxlen=history_window)
        self.memory_history = deque(maxlen=history_window)
        self.failure_history = deque(maxlen=history_window)
        self.last_prediction = {}

    def update(self, cpu_percent, memory_percent, failure_count):
        """Add new metrics to history."""
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory_percent)
        self.failure_history.append(failure_count)

    def predict_degradation(self) -> Dict[str, float]:
        """Predict probability of degradation in next 5 minutes."""
        if len(self.cpu_history) < 10:
            return {"cpu": 0.0, "memory": 0.0, "failures": 0.0}

        cpu_list = list(self.cpu_history)
        mem_list = list(self.memory_history)
        failure_list = list(self.failure_history)

        cpu_trend = sum(cpu_list) / len(cpu_list)
        cpu_slope = (cpu_list[-1] - cpu_list[0]) / len(cpu_list)
        cpu_volatility = (max(cpu_list) - min(cpu_list)) / max(cpu_trend, 1.0)

        if cpu_trend > 75:
            cpu_prob = min(1.0, (cpu_trend - 75) / 20.0 + cpu_slope * 0.01 + cpu_volatility * 0.1)
        else:
            cpu_prob = max(0.0, cpu_slope * 0.05 if cpu_slope > 0 else 0.0)

        mem_trend = sum(mem_list) / len(mem_list)
        mem_prob = min(1.0, max(0.0, (mem_trend - 70) / 25.0))

        recent_failures = failure_list[-10:] if len(failure_list) > 0 else [0]
        failure_rate = sum(recent_failures) / max(len(recent_failures), 1)
        failure_prob = min(1.0, failure_rate * 2.0)

        self.last_prediction = {"cpu": cpu_prob, "memory": mem_prob, "failures": failure_prob}
        return self.last_prediction


class TelemetryCollector:
    """Collect and export detailed metrics for analysis."""

    def __init__(self, export_path: Optional[str] = None):
        self.export_path = Path(export_path or "~/.l104_daemon_metrics.jsonl").expanduser()
        self.metrics = deque(maxlen=10000)

    def record_event(self, event_type: str, data: Dict[str, Any]):
        """Record a telemetry event."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            **data
        }
        self.metrics.append(event)

        if len(self.metrics) % 100 == 0:
            self._flush_to_disk()

    def _flush_to_disk(self):
        """Write metrics to JSONL file."""
        try:
            self.export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.export_path, "a") as f:
                for metric in list(self.metrics)[-100:]:
                    f.write(json.dumps(metric) + "\n")
        except Exception as e:
            logging.getLogger("L104_ORCHESTRATOR").error(f"Telemetry flush failed: {e}")

    def get_statistics(self, event_type: str, window_sec: int = 300) -> Dict[str, Any]:
        """Get statistics for a metric over recent window."""
        now = time.time()
        relevant = [
            m for m in self.metrics
            if m["event_type"] == event_type and now - m["timestamp"] < window_sec
        ]

        if not relevant:
            return {}

        values = []
        for m in relevant:
            for k, v in m.items():
                if isinstance(v, (int, float)) and k != "timestamp":
                    values.append(v)

        if not values:
            return {"count": len(relevant)}

        return {
            "count": len(relevant),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "p50": sorted(values)[len(values) // 2],
            "p99": sorted(values)[int(len(values) * 0.99)] if len(values) > 100 else None,
        }


class DaemonRecoveryEngine:
    """Intelligent daemon recovery with adaptive strategies."""

    RECOVERY_STRATEGIES = {
        "restart": {"description": "Full daemon restart", "backoff_base": 2, "max_attempts": 5},
        "reset_state": {"description": "Clear corrupted state and restart", "backoff_base": 3, "max_attempts": 3},
        "reduce_load": {"description": "Reduce task concurrency and restart", "backoff_base": 2, "max_attempts": 3},
        "full_recalibration": {"description": "Full system recalibration", "backoff_base": 5, "max_attempts": 1},
    }

    def __init__(self):
        self.failure_history: Dict[str, deque] = {}
        self.recovery_attempts: Dict[str, int] = {}
        self.current_strategy: Dict[str, str] = {}

    def record_failure(self, daemon_id: str, error: str):
        """Record a daemon failure."""
        if daemon_id not in self.failure_history:
            self.failure_history[daemon_id] = deque(maxlen=100)
        self.failure_history[daemon_id].append({"timestamp": time.time(), "error": error})

    def get_recovery_strategy(self, daemon_id: str) -> str:
        """Determine best recovery strategy."""
        if daemon_id not in self.failure_history:
            return "restart"

        failures = self.failure_history[daemon_id]
        if len(failures) < 2:
            return "restart"

        last_5_errors = [f["error"] for f in list(failures)[-5:]]
        if len(set(last_5_errors)) == 1:
            return "reset_state"

        recent_failures = [f for f in failures if time.time() - f["timestamp"] < 300]
        if len(recent_failures) > 5:
            return "reduce_load"

        if self.recovery_attempts.get(daemon_id, 0) > 8:
            return "full_recalibration"

        return "restart"

    def execute_recovery(self, daemon_id: str) -> bool:
        """Execute recovery and return True if successful."""
        strategy = self.get_recovery_strategy(daemon_id)
        attempt = self.recovery_attempts.get(daemon_id, 0) + 1

        backoff_base = self.RECOVERY_STRATEGIES[strategy]["backoff_base"]
        backoff_sec = min(backoff_base ** attempt, 300)

        logging.getLogger("L104_ORCHESTRATOR").info(
            f"[RECOVERY] {daemon_id}: {strategy} (attempt {attempt}, backoff {backoff_sec}s)"
        )

        time.sleep(backoff_sec)
        return True


class AdaptiveResourceManager:
    """Dynamically allocate resources based on load."""

    def __init__(self):
        self.cpu_usage = deque(maxlen=60)
        self.memory_usage = deque(maxlen=60)
        self.task_queue_depth = deque(maxlen=60)
        self.current_allocation = {
            "max_concurrent_tasks": 3,
            "persist_interval_cycles": 60,
            "gc_threshold_percent": 80,
        }

    def update_metrics(self, cpu, memory, queue_depth):
        """Update current metrics."""
        self.cpu_usage.append(cpu)
        self.memory_usage.append(memory)
        self.task_queue_depth.append(queue_depth)

    def compute_optimal_allocation(self) -> Dict[str, Any]:
        """Compute optimal resource allocation."""
        if not self.cpu_usage:
            return self.current_allocation

        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage)
        avg_memory = sum(self.memory_usage) / len(self.memory_usage)
        avg_queue = sum(self.task_queue_depth) / len(self.task_queue_depth)

        if avg_cpu < 40 and avg_queue > 5:
            max_tasks = min(10, self.current_allocation["max_concurrent_tasks"] + 1)
        elif avg_cpu > 85 or avg_memory > 85:
            max_tasks = max(1, self.current_allocation["max_concurrent_tasks"] - 1)
        else:
            max_tasks = self.current_allocation["max_concurrent_tasks"]

        if avg_cpu > 90:
            persist_interval = 30
        elif avg_cpu > 70:
            persist_interval = 60
        else:
            persist_interval = 120

        if avg_memory > 85:
            gc_threshold = 70
        elif avg_memory > 75:
            gc_threshold = 75
        else:
            gc_threshold = 85

        self.current_allocation = {
            "max_concurrent_tasks": max_tasks,
            "persist_interval_cycles": persist_interval,
            "gc_threshold_percent": gc_threshold,
        }

        return self.current_allocation


class CrossDaemonSynchronizer:
    """Coordinate recovery and state across multiple daemons."""

    def __init__(self):
        self.sync_state = {}
        self.shared_events = deque(maxlen=1000)

    def broadcast_event(self, event_type: str, source_daemon: str, data: Dict):
        """Broadcast event to all daemons."""
        event = {
            "type": event_type,
            "source": source_daemon,
            "timestamp": time.time(),
            **data
        }
        self.shared_events.append(event)

        logging.getLogger("L104_ORCHESTRATOR").info(f"[SYNC] {event_type} from {source_daemon}")

    def get_relevant_events(self, daemon_id: str, since_ts: float) -> List[Dict]:
        """Get events relevant to a specific daemon."""
        return [
            e for e in self.shared_events
            if e["timestamp"] > since_ts and e["source"] != daemon_id
        ]

    def should_pause_daemon(self, daemon_id: str) -> bool:
        """Check if daemon should pause due to other failures."""
        failure_count = sum(
            1 for e in self.shared_events
            if e["type"] == "daemon_failed" and time.time() - e["timestamp"] < 60
        )
        return failure_count > 2


# ═════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════

class L104DaemonOrchestrator:
    """
    Central coordinator for all L104 daemons.

    Manages:
      • Task scheduling & prioritization
      • Resource allocation
      • Inter-daemon communication
      • Global health monitoring
      • Graceful load balancing
    """

    def __init__(self):
        self._logger = logging.getLogger("L104_ORCHESTRATOR")
        self._running = False
        self._lock = threading.Lock()

        # Task scheduling
        self._task_queue: List[Task] = []
        self._in_flight: Dict[str, Task] = {}

        # Daemon registry
        self._daemons: Dict[str, DaemonMetrics] = {}
        self._daemon_threads: Dict[str, threading.Thread] = {}

        # Telemetry
        self._system_metrics = SystemMetrics()
        self._metric_history = deque(maxlen=TELEMETRY_WINDOW)

        # Event bus (inter-daemon messaging)
        self._event_queue = queue.Queue()
        self._event_subscribers: Dict[str, List[Callable]] = {}

        # State persistence
        self._persist_cycle_count = 0

        # Resource tracking
        self._current_cycle_cpu_ms = 0.0
        self._current_cycle_memory_peak_mb = 0.0

        # Phase 2: Robustness enhancements
        self._health_predictor = HealthPredictor()
        self._telemetry = TelemetryCollector()
        self._recovery_engine = DaemonRecoveryEngine()
        self._resource_manager = AdaptiveResourceManager()
        self._synchronizer = CrossDaemonSynchronizer()

        # Phase 3: ML Integration
        self._ml_enabled = PHASE3_ML_AVAILABLE
        self._ml_prediction_result = None
        self._ml_anomaly_check = None
        self._ml_trend_analysis = None
        self._ml_pattern_count = 0
        if self._ml_enabled:
            self._logger.info("✓ Phase 3 ML systems loaded")

        atexit.register(self._on_exit)

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────

    def start(self):
        """Start the orchestrator main loop."""
        with self._lock:
            if self._running:
                self._logger.warning("Orchestrator already running")
                return
            self._running = True

        self._logger.info(f"L104 Daemon Orchestrator {ORCHESTRATOR_VERSION} started")
        self._load_state()

        # Start main orchestration loop
        self._orchestration_thread = threading.Thread(
            target=self._orchestration_loop,
            daemon=True,
            name="L104-Orchestrator-Main"
        )
        self._orchestration_thread.start()

        # Start event bus thread
        self._event_thread = threading.Thread(
            target=self._event_loop,
            daemon=True,
            name="L104-EventBus"
        )
        self._event_thread.start()

    def stop(self, timeout_s: int = 30):
        """Gracefully shutdown orchestrator."""
        self._logger.info("Orchestrating graceful daemon shutdown...")
        self._running = False
        self._persist_state()

        # Wait for threads
        for thread in [self._orchestration_thread, self._event_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=timeout_s / 2)

        self._logger.info("Orchestrator stopped")

    def register_daemon(self, daemon_id: str, daemon_type: DaemonType):
        """Register a daemon for orchestration."""
        with self._lock:
            if daemon_id not in self._daemons:
                metrics = DaemonMetrics(daemon_id=daemon_id)
                self._daemons[daemon_id] = metrics
                self._logger.info(f"Registered daemon: {daemon_id} ({daemon_type})")

    def submit_task(self, task: Task) -> bool:
        """Submit a task to the queue."""
        with self._lock:
            heapq.heappush(self._task_queue, task)
            self._logger.debug(f"Task queued: {task.daemon_id}/{task.task_type} (priority={task.priority})")
            return True

    def submit_batch(self, tasks: List[Task]) -> int:
        """Submit multiple tasks at once."""
        count = 0
        for task in tasks:
            if self.submit_task(task):
                count += 1
        return count

    def report_cycle(self, daemon_id: str, duration_ms: float, success: bool,
                     cpu_percent: float = 0.0, memory_mb: float = 0.0):
        """Report completion of a daemon cycle."""
        with self._lock:
            if daemon_id not in self._daemons:
                self.register_daemon(daemon_id, DaemonType(daemon_id.split("_")[0]))

            metrics = self._daemons[daemon_id]
            if success:
                metrics.cycles_completed += 1
                metrics.last_cycle_ms = duration_ms
                metrics.avg_cycle_ms = (
                    (metrics.avg_cycle_ms * (metrics.cycles_completed - 1) + duration_ms)
                    / metrics.cycles_completed
                )
                metrics.max_cycle_ms = max(metrics.max_cycle_ms, duration_ms)
            else:
                metrics.cycles_failed += 1
                metrics.error_log.append({
                    "timestamp": time.time(),
                    "error": "Cycle failed"
                })

            metrics.cpu_percent = cpu_percent
            metrics.memory_mb = memory_mb
            metrics.last_updated = time.time()
            self._update_daemon_health(daemon_id)

    def emit_event(self, event_type: str, daemon_id: str, payload: Dict[str, Any]):
        """Emit an event to the event bus."""
        event = {
            "type": event_type,
            "daemon_id": daemon_id,
            "timestamp": time.time(),
            "payload": payload
        }
        self._event_queue.put(event)

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to event type."""
        if event_type not in self._event_subscribers:
            self._event_subscribers[event_type] = []
        self._event_subscribers[event_type].append(callback)

    def status(self) -> Dict[str, Any]:
        """Get full orchestrator status."""
        with self._lock:
            self._system_metrics.daemon_metrics = dict(self._daemons)
            return asdict(self._system_metrics)

    # ─────────────────────────────────────────────────────────────────────
    # PRIVATE: ORCHESTRATION LOOP
    # ─────────────────────────────────────────────────────────────────────

    def _orchestration_loop(self):
        """Main orchestration loop."""
        cycle_count = 0

        while self._running:
            try:
                cycle_count += 1
                cycle_start = time.perf_counter()

                # 1. Update system metrics
                self._update_system_metrics()

                # PHASE 2: Record cycle telemetry
                self._telemetry.record_event("cycle_start", {
                    "cycle_number": cycle_count,
                    "queued_tasks": len(self._task_queue),
                    "in_flight_tasks": len(self._in_flight),
                    "cpu_percent": self._system_metrics.cpu_percent,
                    "memory_percent": self._system_metrics.memory_percent,
                })

                # 2. Assess health & degradation
                health = self._assess_health()

                # PHASE 2: Predict degradation before it happens
                predictions = self._health_predictor.predict_degradation()

                if predictions.get("cpu", 0) > 0.7:
                    self._logger.warning(
                        f"CPU degradation predicted (prob={predictions['cpu']:.1%}) — triggering preventive GC"
                    )
                    gc.collect()

                if predictions.get("memory", 0) > 0.7:
                    self._logger.warning(f"Memory degradation predicted — triggering cache cleanup")
                    try:
                        from l104_intellect import local_intellect
                        local_intellect.prune_old_entries()
                    except Exception:
                        pass

                # PHASE 3: ML-Based Workload Prediction & Anomaly Detection
                if self._ml_enabled:
                    self._process_ml_predictions(cycle_count)

                # 3. Schedule next batch of tasks
                self._schedule_next_batch(health)

                # PHASE 2: Use adaptive resource allocation
                optimal = self._resource_manager.compute_optimal_allocation()
                self._resource_manager.update_metrics(
                    self._system_metrics.cpu_percent,
                    self._system_metrics.memory_percent,
                    len(self._task_queue)
                )

                # PHASE 2: Trigger GC if memory threshold exceeded
                if self._system_metrics.memory_percent > optimal["gc_threshold_percent"]:
                    gc.collect()

                # 4. Persist state periodically (adaptive interval)
                persist_interval = optimal.get("persist_interval_cycles", 60)
                if cycle_count % persist_interval == 0:
                    self._persist_state()

                # 5. Adaptive sleep (load-aware)
                cycle_elapsed = time.perf_counter() - cycle_start
                sleep_time = max(0.1, 1.0 - cycle_elapsed)

                # PHASE 2: Record cycle completion
                self._telemetry.record_event("cycle_complete", {
                    "cycle_number": cycle_count,
                    "duration_ms": cycle_elapsed * 1000,
                    "health_status": health.value,
                    "cpu_percent": self._system_metrics.cpu_percent,
                })

                self._logger.debug(
                    f"Orchestration cycle {cycle_count}: "
                    f"tasks={len(self._task_queue)}, "
                    f"health={health.value}, "
                    f"cpu={self._system_metrics.cpu_percent:.1f}%, "
                    f"duration={cycle_elapsed*1000:.1f}ms"
                )

                time.sleep(sleep_time)

            except Exception as e:
                self._logger.error(f"Orchestration loop error: {e}", exc_info=True)
                self._telemetry.record_event("cycle_error", {
                    "error": str(e),
                    "cycle_number": cycle_count,
                })
                time.sleep(1.0)

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 3: ML INTEGRATION
    # ─────────────────────────────────────────────────────────────────────

    def _process_ml_predictions(self, cycle_count: int):
        """Process ML predictions for workload forecasting and anomaly detection."""
        try:
            # Feed current metrics to ML systems
            cpu = self._system_metrics.cpu_percent
            memory = self._system_metrics.memory_percent
            queue_depth = len(self._task_queue)

            # Calculate latency from recent cycles
            recent_cycles = list(self._telemetry.metrics)[-10:] if self._telemetry.metrics else []
            avg_latency = sum(
                c.get("duration_ms", 0) for c in recent_cycles if "duration_ms" in c
            ) / max(len(recent_cycles), 1)

            # Update ML predictor with current metrics
            workload_predictor.add_metric(cpu, memory, queue_depth, avg_latency)

            # Learn patterns every 50 cycles
            if cycle_count % 50 == 0:
                patterns = workload_predictor.learn_patterns()
                self._ml_pattern_count = len(patterns)
                if patterns:
                    self._logger.info(f"✓ Detected {len(patterns)} workload patterns via ML")
                    for pattern in patterns:
                        self._logger.debug(f"  • {pattern.name}: {pattern.frequency:.1%} frequency, "
                                         f"batch={pattern.optimal_batch_size}, "
                                         f"sacred_align={pattern.sacred_alignment:.3f}")

            # Get next prediction (5 steps ahead)
            prediction = workload_predictor.predict_next(steps_ahead=5)
            self._ml_prediction_result = prediction

            if prediction.confidence > 0.5:
                self._logger.debug(
                    f"ML Prediction: CPU {prediction.predicted_value:.1f}% (confidence {prediction.confidence:.1%}) "
                    f"→ {prediction.recommendation} (sacred_score={prediction.sacred_score:.3f})"
                )

                # Use ML recommendation to adjust resource scaling
                if prediction.recommendation == "scale_up" and self._system_metrics.cpu_percent < 40:
                    self._logger.info("✓ ML predicts low load ahead — scaling up resources proactively")
                    self._resource_manager.current_allocation["max_concurrent_tasks"] = min(
                        10, self._resource_manager.current_allocation.get("max_concurrent_tasks", 3) + 1
                    )
                elif prediction.recommendation == "scale_down" and self._system_metrics.cpu_percent > 75:
                    self._logger.info("✓ ML predicts high load ahead — scaling down to prepare")
                    self._resource_manager.current_allocation["max_concurrent_tasks"] = max(
                        1, self._resource_manager.current_allocation.get("max_concurrent_tasks", 3) - 1
                    )

            # Anomaly detection
            anomaly_detector.update_statistics(
                [self._system_metrics.cpu_percent] * max(1, len(recent_cycles)),
                [c.get("duration_ms", 0) for c in recent_cycles if "duration_ms" in c]
            )
            anomaly = anomaly_detector.is_anomaly(cpu, avg_latency)
            self._ml_anomaly_check = anomaly

            if anomaly["is_anomaly"]:
                self._logger.warning(
                    f"⚠ ML Anomaly Detected: CPU z-score={anomaly['cpu_zscore']:.2f}, "
                    f"Latency z-score={anomaly['latency_zscore']:.2f} (severity={anomaly['severity']:.2f})"
                )
                # Record anomaly in telemetry
                self._telemetry.record_event("ml_anomaly", {
                    "severity": anomaly["severity"],
                    "cpu_zscore": anomaly["cpu_zscore"],
                    "latency_zscore": anomaly["latency_zscore"],
                })

            # Trend analysis every 20 cycles
            if cycle_count % 20 == 0:
                # Add cycle data to trend analyzer
                trend_analyzer.add_cycle(avg_latency, True, cpu)
                trends = trend_analyzer.analyze_trends()
                self._ml_trend_analysis = trends

                if "trend" in trends:
                    self._logger.info(
                        f"✓ ML Trend Analysis: {trends['trend']}, "
                        f"avg_latency={trends.get('avg_duration_ms', 0):.1f}ms, "
                        f"success_rate={trends.get('success_rate', 0):.1%}"
                    )

                    # Get recommendations
                    recommendations = trend_analyzer.get_optimization_recommendations()
                    if recommendations and recommendations[0] != "System performing optimally":
                        self._logger.info(f"✓ ML Recommendations: {recommendations[0]}")

        except Exception as e:
            self._logger.debug(f"ML processing error (non-critical): {e}")

    def _schedule_next_batch(self, health: HealthStatus):
        """Schedule next batch of tasks based on health."""
        with self._lock:
            if not self._task_queue:
                return

            # PHASE 2: Use adaptive resource manager for batch size
            optimal = self._resource_manager.current_allocation
            adaptive_batch_size = optimal.get("max_concurrent_tasks", 3)

            # Base batch size on health
            if health == HealthStatus.HEALTHY:
                batch_size = min(adaptive_batch_size, 10)
            elif health == HealthStatus.DEGRADED:
                batch_size = max(1, adaptive_batch_size // 2)
            else:
                batch_size = 0  # Don't schedule during critical/failed

            # PHASE 2: Check for synchronization pause
            if self._synchronizer.should_pause_daemon("orchestrator"):
                batch_size = 0  # Pause scheduling if other daemons failing

            for _ in range(min(batch_size, len(self._task_queue))):
                task = heapq.heappop(self._task_queue)
                self._emit_task_scheduled(task)
                self._in_flight[f"{task.daemon_id}_{task.task_type}"] = task

                # PHASE 2: Record task scheduling
                self._telemetry.record_event("task_scheduled", {
                    "daemon_id": task.daemon_id,
                    "task_type": task.task_type,
                    "priority": task.priority,
                })

    def _assess_health(self) -> HealthStatus:
        """Assess overall system health."""
        with self._lock:
            if not self._daemons:
                return HealthStatus.HEALTHY

            # Score based on daemon health + system resources
            daemon_health_scores = [d.health_score for d in self._daemons.values()]
            avg_daemon_health = sum(daemon_health_scores) / len(daemon_health_scores)

            cpu_ratio = self._system_metrics.cpu_percent / CPU_QUOTA_PERCENT
            memory_ratio = self._system_metrics.memory_percent / 100.0
            resource_stress = max(cpu_ratio, memory_ratio)

            overall_score = (avg_daemon_health * 0.6) + ((1.0 - resource_stress) * 0.4)

            if overall_score > 0.8:
                return HealthStatus.HEALTHY
            elif overall_score > 0.5:
                return HealthStatus.DEGRADED
            elif overall_score > 0.2:
                return HealthStatus.CRITICAL
            else:
                return HealthStatus.FAILED

    def _update_system_metrics(self):
        """Update system-wide metrics."""
        with self._lock:
            try:
                process = psutil.Process()
                self._system_metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
                self._system_metrics.memory_percent = psutil.virtual_memory().percent
                self._system_metrics.memory_mb = process.memory_info().rss / 1024 / 1024
                self._system_metrics.active_tasks = len(self._in_flight)
                self._system_metrics.queued_tasks = len(self._task_queue)
                self._system_metrics.cycle_count += 1
                self._system_metrics.timestamp = time.time()
            except Exception as e:
                self._logger.error(f"Metrics update error: {e}")

    def _update_daemon_health(self, daemon_id: str):
        """Update health score for a daemon."""
        if daemon_id not in self._daemons:
            return

        metrics = self._daemons[daemon_id]
        if metrics.cycles_completed == 0:
            metrics.health_score = 1.0
            return

        success_rate = metrics.cycles_completed / (metrics.cycles_completed + metrics.cycles_failed)
        decay = HEALTH_STALENESS_DECAY ** (
            (time.time() - metrics.last_updated) / TELEMETRY_WINDOW
        )

        metrics.health_score = success_rate * decay

    # ─────────────────────────────────────────────────────────────────────
    # PRIVATE: EVENT BUS
    # ─────────────────────────────────────────────────────────────────────

    def _event_loop(self):
        """Event bus dispatch loop."""
        while self._running:
            try:
                event = self._event_queue.get(timeout=1.0)
                event_type = event.get("type")

                # Dispatch to subscribers
                if event_type in self._event_subscribers:
                    for callback in self._event_subscribers[event_type]:
                        try:
                            callback(event)
                        except Exception as e:
                            self._logger.error(f"Event handler error: {e}")

            except queue.Empty:
                pass
            except Exception as e:
                self._logger.error(f"Event loop error: {e}")

    def _emit_task_scheduled(self, task: Task):
        """Emit task-scheduled event."""
        self.emit_event("task_scheduled", task.daemon_id, {
            "task_type": task.task_type,
            "priority": task.priority.value
        })

    # ─────────────────────────────────────────────────────────────────────
    # PRIVATE: PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────

    def _persist_state(self):
        """Save orchestrator state to disk."""
        try:
            state = {
                "version": ORCHESTRATOR_VERSION,
                "timestamp": time.time(),
                "system_metrics": self._system_metrics.to_dict(),
                "daemon_metrics": {
                    k: v.to_dict() for k, v in self._daemons.items()
                },
                "queued_tasks": len(self._task_queue),
                "in_flight_tasks": len(self._in_flight),
            }

            with open(ORCHESTRATOR_STATE_PATH, "w") as f:
                json.dump(state, f, indent=2)

            self._logger.debug(f"State persisted to {ORCHESTRATOR_STATE_PATH}")

        except Exception as e:
            self._logger.error(f"State persistence error: {e}")

    def _load_state(self):
        """Load orchestrator state from disk."""
        if not ORCHESTRATOR_STATE_PATH.exists():
            self._logger.info("No prior state found, starting fresh")
            return

        try:
            with open(ORCHESTRATOR_STATE_PATH, "r") as f:
                state = json.load(f)

            self._logger.info(f"Loaded state: {len(state.get('daemon_metrics', {}))} daemons")

        except Exception as e:
            self._logger.error(f"State load error: {e}")

    def _on_exit(self):
        """Cleanup on exit."""
        if self._running:
            self.stop()


# ═════════════════════════════════════════════════════════════════════════
# CLI & BOOTSTRAP
# ═════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point."""
    import argparse

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="L104 Unified Daemon Orchestrator")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--daemon-ids", nargs="+", default=["vqpu", "quantum_ai", "soul"],
                        help="Daemon IDs to orchestrate")

    args = parser.parse_args()

    orchestrator = L104DaemonOrchestrator()

    if args.status:
        if ORCHESTRATOR_STATE_PATH.exists():
            with open(ORCHESTRATOR_STATE_PATH) as f:
                status = json.load(f)
            print(json.dumps(status, indent=2))
        else:
            print("No state found")
        return

    # Register daemons
    for daemon_id in args.daemon_ids:
        orchestrator.register_daemon(daemon_id, DaemonType(daemon_id))

    # Start orchestrator
    orchestrator.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        orchestrator.stop()


# ═══════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════

# Export as DaemonOrchestrator for compatibility with existing imports
DaemonOrchestrator = L104DaemonOrchestrator


if __name__ == "__main__":
    main()
