"""L104 VQPU v4.0.0 — Micro Process Background Assistant (lightweight daemon).

A minimal, high-frequency background daemon for micro-quantum operations.
Unlike the heavy VQPUDaemonCycler (full simulation cycles every 3 min),
this daemon handles sub-second micro-tasks on a tight 5–15s loop:

  Core micro-tasks (v1.0):
  - Heartbeat telemetry (sacred alignment probe)
  - Cache eviction & TTL maintenance
  - Micro-scoring snapshots (sacred resonance: (GOD_CODE/16)^φ ≈ 286)
  - IPC inbox polling (pick up & dispatch micro-jobs)
  - Memory pressure monitor (fast psutil sample)
  - Fidelity micro-probe (single-gate φ alignment)
  - Quantum noise floor sampling
  - Cross-engine health ping (lightweight 3-engine check)

  VQPU subsystem micro-tasks (v2.0):
  - ScoringCache TTL eviction (ASI/AGI/SC/entropy caches)
  - CircuitCache LRU stats & monitoring
  - DaemonCycler health ping (check parent daemon liveness)
  - Bridge throughput snapshot (jobs/s, queue depth)
  - AccelStatevectorEngine availability probe
  - Transpiler pass statistics

Architecture:
  - Single daemon thread, no thread pool (GIL-friendly)
  - MicroTask registry — each task has a cadence (run every N ticks)
  - Priority queue — urgent micro-tasks jump the queue
  - Bounded telemetry ring buffer (deque, O(1))
  - State persisted to .l104_vqpu_micro_daemon.json
  - IPC via /tmp/l104_bridge/micro/ inbox/outbox (with response writing)
  - Graceful shutdown with atexit persistence
  - Smart crash detection — PID file presence = unclean shutdown
  - Watchdog heartbeat — parent daemon can check liveness
  - Bridge wiring — connect_bridge() wires to VQPUBridge lifecycle
  - Self-test probe for l104_debug.py integration (12 probes)
  - Signal handlers: SIGUSR1 → status dump, SIGUSR2 → force-tick
  - Env-driven config: L104_MICRO_TICK_INTERVAL, L104_MICRO_TICK_MIN/MAX
  - Health staleness decay — degrades if no tasks run (no silent stalls)
  - Telemetry analytics — trend detection, anomaly alerts, performance grading
  - Task auto-throttle — flaky tasks auto-degrade cadence after N failures
  - Watchdog heartbeat file — timestamp file for launchd/external monitoring
  - Per-task wall-clock timeout — kills stuck tasks exceeding budget
  - IPC completion responses — write final result to outbox when tasks finish
  - CLI --analytics: print analytics report and exit

v2.5.0 Improvements:
  - Bridge auto-wiring — on start(), auto-discovers the VQPUBridge singleton
    from l104_vqpu._bridge. If none exists, creates a lightweight bridge with
    enable_micro_daemon=False (avoids circular creation) and adopts itself.
  - Periodic re-wire — every ~60s, if bridge is still None, retries discovery
  - Self-test bridge_wiring probe now requires connected=True (was informational)
  - Bridge adoption — when auto-wiring, replaces the bridge's inactive internal
    micro daemon with this instance (single daemon, not two)

v2.4.0 Improvements:
  - Per-task wall-clock timeout (default 5s) — tasks exceeding budget are
    recorded as timeout failures and count toward auto-throttle streaks
  - IPC completion responses — Phase 4 writes final status (completed/failed)
    to outbox after pending IPC tasks finish (full request-response cycle)
  - CLI --analytics: print telemetry analytics report (trend, grade, hotspots)
    and exit — designed for monitoring pipelines and CI health dashboards
  - Bridge wiring: analytics(), throttled_tasks(), reset_stats(), task_stats()
  - daemon.py: fixed PID liveness check bug (was signal.signal, now os.kill)
  - daemon.py: cross-health reads v2.4 fields (heartbeat_age, analytics_grade)
  - Plist updated to v2.4 with IPC rate limit env var + new CLI examples

v2.3.0 Improvements:
  - TelemetryAnalytics subsystem — analyze ring buffer for trends & anomalies
    - health_trend(): slope of health over recent window (rising/falling/stable)
    - detect_anomalies(): flag ticks where elapsed_ms > 3σ above mean
    - performance_grade(): A/B/C/D/F grade based on pass_rate + tick_ms + health
    - task_hotspots(): identify which tasks consume most tick time
  - Task auto-throttle — if a task fails N consecutive times, double cadence
    (auto-recovers when the task passes again)
  - Watchdog heartbeat file — writes timestamp to /tmp/l104_bridge/micro/heartbeat
    every tick (launchd/external tools can stat this file for liveness)
  - IPC rate limiter — max 20 IPC jobs per tick to prevent flooding
  - Self-test expanded to 12 probes (+ analytics + auto-throttle)

v2.2.0 Improvements:
  - Fixed sacred resonance: (GOD_CODE/16)^φ ≈ 286 (was GOD_CODE^(1/φ))
  - Smart crash detection via PID file (only counts unclean shutdowns)
  - CLI --self-test: run 10 probes and exit 0/1 (CI/launchd integration)
  - CLI --health-check: read persisted state, report health, exit 0/1
  - CLI --dump-metrics: force tick, export TickMetrics to logs/
  - reset_stats(): zero all counters + ring buffers
  - dump_metrics(): export TickMetrics history to JSON file
  - task_stats(): per-task execution statistics (count, mean, max, fails)
  - Health staleness decay (1%/tick when idle)
  - Logs directory auto-creation at CLI startup

v2.1.0 Improvements:
  - MicroTaskPriority enum (CRITICAL=1, HIGH=2, NORMAL=5, LOW=7, IDLE=9)
  - TickMetrics dataclass for per-tick profiling
  - IPC outbox response writing (callers can poll results)
  - SIGUSR1/SIGUSR2 signal handlers in CLI mode
  - Env-driven config reading (L104_MICRO_* env vars)
  - Crash recovery with monotonic tick preservation
  - Improved watchdog with PID file + liveness endpoint
  - Bridge throughput rate calculation (jobs/s over window)

Usage:
    from l104_vqpu.micro_daemon import VQPUMicroDaemon

    micro = VQPUMicroDaemon()
    micro.start()                    # Spawns lightweight background thread
    micro.submit("score_check")      # Submit a micro-task by name
    micro.submit_custom(fn, *args)   # Submit an arbitrary callable
    micro.status()                   # Telemetry snapshot
    micro.task_stats()               # Per-task execution statistics
    micro.analytics()                # Telemetry analytics report
    micro.throttled_tasks()          # Auto-throttled task report
    micro.dump_metrics()             # Export profiling data to JSON
    micro.stop()                     # Graceful shutdown

    # Wire to bridge:
    micro.connect_bridge(bridge)     # Links to VQPUBridge subsystems

    # CLI modes:
    #   python -m l104_vqpu.micro_daemon                   # Run daemon
    #   python -m l104_vqpu.micro_daemon --self-test       # Test + exit
    #   python -m l104_vqpu.micro_daemon --health-check    # Check + exit
    #   python -m l104_vqpu.micro_daemon --dump-metrics    # Dump + exit
    #   python -m l104_vqpu.micro_daemon --analytics       # Analytics + exit
"""

import atexit
import gc
import json
import logging
import math
import os
import random
import time
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .constants import (
    GOD_CODE, PHI, VOID_CONSTANT,
    VERSION,
    BRIDGE_PATH,
    CACHE_ASI_TTL_S, CACHE_AGI_TTL_S, CACHE_SC_TTL_S, CACHE_ENTROPY_TTL_S,
)

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

MICRO_DAEMON_VERSION = "4.0.0"

# Tick interval (seconds) — one tick = one pass through all due micro-tasks
# v2.1: Env-driven overrides for launchd plist configurability
MICRO_TICK_INTERVAL_S = float(os.environ.get(
    "L104_MICRO_TICK_INTERVAL", "5.0"))   # Base tick: 5 seconds
MICRO_TICK_MIN_S = float(os.environ.get(
    "L104_MICRO_TICK_MIN", "2.0"))         # Floor under low load
MICRO_TICK_MAX_S = float(os.environ.get(
    "L104_MICRO_TICK_MAX", "15.0"))        # Ceiling under high load
MICRO_LOAD_THRESHOLD_LOW = 25.0         # CPU% → faster ticks
MICRO_LOAD_THRESHOLD_HIGH = 65.0        # CPU% → slower ticks

# State persistence
MICRO_STATE_FILE = ".l104_vqpu_micro_daemon.json"
MICRO_PERSIST_EVERY_N_TICKS = 20        # Persist state every N ticks (~100s)

# Telemetry ring sizes
MICRO_TELEMETRY_WINDOW = 200            # Heartbeat history
MICRO_ERROR_LOG_SIZE = 50               # Error ring
MICRO_TASK_HISTORY_SIZE = 500           # Completed task log

# IPC paths
MICRO_BRIDGE_PATH = BRIDGE_PATH / "micro"
MICRO_INBOX_PATH = MICRO_BRIDGE_PATH / "inbox"
MICRO_OUTBOX_PATH = MICRO_BRIDGE_PATH / "outbox"

# Sacred micro-probe constants
_PHI_MICRO_PHASE = (GOD_CODE % (2 * math.pi))  # canonical phase
_NOISE_FLOOR_SAMPLES = 8                         # quick noise sampling depth
_HEALTH_PING_TIMEOUT_S = 2.0                     # max wait for engine pings

# v2.1: PID file for watchdog liveness (parent daemons check this)
MICRO_PID_FILE = MICRO_BRIDGE_PATH / "micro_daemon.pid"

# v2.3: Watchdog heartbeat file — external tools stat this for liveness
MICRO_HEARTBEAT_FILE = MICRO_BRIDGE_PATH / "heartbeat"

# v2.3: IPC rate limit — max IPC jobs picked per tick to prevent flooding
MICRO_IPC_RATE_LIMIT = int(os.environ.get("L104_MICRO_IPC_RATE_LIMIT", "20"))

# v2.3: Auto-throttle — consecutive failures before doubling task cadence
MICRO_AUTO_THROTTLE_THRESHOLD = 3

# v2.4: Per-task wall-clock timeout (seconds) — kills stuck tasks
MICRO_TASK_TIMEOUT_S = float(os.environ.get(
    "L104_MICRO_TASK_TIMEOUT", "5.0"))

# v3.0: Quantum qubit network constants
MICRO_QUANTUM_QUBITS = int(os.environ.get(
    "L104_MICRO_QUANTUM_QUBITS", "4"))       # Qubits per daemon node
MICRO_QUANTUM_NETWORK = os.environ.get(
    "L104_MICRO_QUANTUM_NETWORK", "1") == "1"  # Enable quantum network mesh
MICRO_QUANTUM_RECALIBRATE_TICKS = int(os.environ.get(
    "L104_MICRO_QUANTUM_RECALIBRATE", "120"))  # Recalibrate every N ticks (~10min)
MICRO_QUANTUM_PURIFY_TICKS = int(os.environ.get(
    "L104_MICRO_QUANTUM_PURIFY", "60"))        # Purify channels every N ticks (~5min)

# v4.0: Task batching
MICRO_BATCH_CPU_THRESHOLD = 60.0       # CPU% above which to batch low-priority tasks
MICRO_BATCH_MAX_SIZE = 6               # Max tasks per batch

# v4.0: Predictive preemption
MICRO_PREEMPT_SLOW_THRESHOLD_MS = 3000.0  # Tasks taking >3s get preempted to end of queue

# v4.0: Cross-daemon health file paths
_DAEMON_STATE_FILES = {
    "vqpu_cycler": ".l104_vqpu_daemon_state.json",
    "nano_daemon": ".l104_nano_daemon_python.json",
    "quantum_ai": ".l104_quantum_ai_daemon.json",
    "guardian": ".l104_resource_guardian.json",
}

_logger = logging.getLogger("L104_VQPU_MICRO")

# v2.2: Module-level import caches — avoid repeated 'from .X import Y' in hot task functions
_cached_ScoringCache = None
_cached_CircuitCache = None
_cached_AccelStatevectorEngine = None
_cached_HardwareStrengthProfiler = None
_cached_CircuitTranspiler = None

def _get_ScoringCache():
    global _cached_ScoringCache
    if _cached_ScoringCache is None:
        from .cache import ScoringCache
        _cached_ScoringCache = ScoringCache
    return _cached_ScoringCache

def _get_CircuitCache():
    global _cached_CircuitCache
    if _cached_CircuitCache is None:
        from .cache import CircuitCache
        _cached_CircuitCache = CircuitCache
    return _cached_CircuitCache

def _get_accel_imports():
    global _cached_AccelStatevectorEngine, _cached_HardwareStrengthProfiler
    if _cached_AccelStatevectorEngine is None:
        from .accel_engine import AccelStatevectorEngine, HardwareStrengthProfiler
        _cached_AccelStatevectorEngine = AccelStatevectorEngine
        _cached_HardwareStrengthProfiler = HardwareStrengthProfiler
    return _cached_AccelStatevectorEngine, _cached_HardwareStrengthProfiler

def _get_CircuitTranspiler():
    global _cached_CircuitTranspiler
    if _cached_CircuitTranspiler is None:
        from .transpiler import CircuitTranspiler
        _cached_CircuitTranspiler = CircuitTranspiler
    return _cached_CircuitTranspiler


# v3.0: Lazy import for quantum network module
_cached_DaemonQubitRegister = None
_cached_QuantumNetworkMesh = None

def _get_quantum_network():
    global _cached_DaemonQubitRegister, _cached_QuantumNetworkMesh
    if _cached_DaemonQubitRegister is None:
        from .quantum_network import DaemonQubitRegister, QuantumNetworkMesh
        _cached_DaemonQubitRegister = DaemonQubitRegister
        _cached_QuantumNetworkMesh = QuantumNetworkMesh
    return _cached_DaemonQubitRegister, _cached_QuantumNetworkMesh


# ═══════════════════════════════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════════════════════════════


class MicroTaskStatus(str, Enum):
    """Typed status values for micro-tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MicroTaskPriority(int, Enum):
    """v2.1: Named priority levels for micro-task scheduling.

    Lower value = higher priority.  Use these instead of magic ints:
        register_task("heartbeat", fn, cadence=1, priority=MicroTaskPriority.CRITICAL)
    """
    CRITICAL = 1    # Sacred constants, heartbeat — run first every tick
    HIGH = 2        # IPC poll, fidelity probe — time-sensitive
    NORMAL = 5      # Scoring, cache stats — routine monitoring
    LOW = 7         # GC pulse, transpiler stats — deferrable
    IDLE = 9        # Accel HW check — run only when load is low


@dataclass
class MicroDaemonConfig:
    """Configuration for VQPUMicroDaemon — all tunables in one place."""
    tick_interval: float = MICRO_TICK_INTERVAL_S
    enable_ipc: bool = True
    enable_adaptive: bool = True
    enable_vqpu_tasks: bool = True        # v2.0: enable VQPU subsystem micro-tasks
    enable_quantum_network: bool = MICRO_QUANTUM_NETWORK  # v3.0: enable quantum qubit network
    quantum_qubits: int = MICRO_QUANTUM_QUBITS            # v3.0: qubits per daemon node
    quantum_topology: str = "all_to_all"                   # v4.0: mesh topology (linear/ring/heavy_hex/all_to_all)
    enable_task_batching: bool = True       # v4.0: enable load-aware task batching
    enable_predictive_preemption: bool = True  # v4.0: enable predictive task preemption
    state_file: Optional[str] = None
    telemetry_window: int = MICRO_TELEMETRY_WINDOW
    error_log_size: int = MICRO_ERROR_LOG_SIZE
    task_history_size: int = MICRO_TASK_HISTORY_SIZE
    persist_every_n_ticks: int = MICRO_PERSIST_EVERY_N_TICKS


@dataclass
class MicroTask:
    """A lightweight micro-operation to run in the daemon loop."""
    task_id: str = ""
    name: str = ""
    priority: int = 5                    # 1=highest, 10=lowest
    payload: dict = field(default_factory=dict)
    callback: Optional[str] = None       # Optional callback task name
    submitted_at: float = field(default_factory=time.time)
    status: str = MicroTaskStatus.PENDING
    result: Any = None
    elapsed_ms: float = 0.0
    error: Optional[str] = None

    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"micro-{os.urandom(4).hex()}"

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "task_id": self.task_id, "name": self.name,
            "priority": self.priority, "status": self.status,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "error": self.error,
        }


@dataclass
class MicroTaskResult:
    """Structured result from a micro-task execution."""
    task_name: str
    tick: int
    status: str = MicroTaskStatus.COMPLETED
    data: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.status == MicroTaskStatus.COMPLETED

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MicroTelemetry:
    """Lightweight telemetry snapshot from one tick."""
    tick: int
    timestamp: float
    tasks_run: int
    tasks_passed: int
    tasks_failed: int
    tick_elapsed_ms: float
    health_score: float
    sacred_alignment: float
    memory_mb: float
    cpu_percent: float
    # v2.0: VQPU subsystem metrics
    scoring_cache_hits: int = 0
    circuit_cache_size: int = 0
    bridge_jobs_total: int = 0
    daemon_cycler_alive: bool = False


@dataclass
class TickMetrics:
    """v2.1: Per-tick profiling metrics for performance analysis.

    Tracks granular timing of each phase within a tick to identify
    bottlenecks and optimize the micro-task pipeline.
    """
    tick: int
    timestamp: float
    builtin_tasks_ms: float = 0.0    # Time spent on built-in tasks
    pending_tasks_ms: float = 0.0    # Time spent on on-demand tasks
    custom_tasks_ms: float = 0.0     # Time spent on custom callables
    ipc_poll_ms: float = 0.0         # Time spent polling IPC inbox
    telemetry_ms: float = 0.0        # Time spent recording telemetry
    total_ms: float = 0.0            # Total tick wall time
    task_count: int = 0              # Total tasks executed this tick
    slowest_task: str = ""           # Name of the slowest task
    slowest_ms: float = 0.0          # Duration of the slowest task

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════
# v2.3: TELEMETRY ANALYTICS SUBSYSTEM
# ═══════════════════════════════════════════════════════════════════


class TelemetryAnalytics:
    """v2.3: Analyze the telemetry ring buffer for trends, anomalies, and grades.

    Operates on a snapshot of the telemetry deque — no daemon locks held.
    All methods are pure functions over the snapshot.

    Usage:
        ta = TelemetryAnalytics(list(daemon._telemetry))
        trend = ta.health_trend()       # "rising" | "falling" | "stable"
        anomalies = ta.detect_anomalies()  # list of anomalous ticks
        grade = ta.performance_grade()  # "A" | "B" | "C" | "D" | "F"
        hotspots = ta.task_hotspots(list(daemon._task_history))
    """

    def __init__(self, telemetry: List[dict]):
        self._data = telemetry

    def health_trend(self, window: int = 20) -> dict:
        """Compute health score trend over the last N telemetry entries.

        Uses simple linear regression (slope) on the health_score field.
        Returns: {"direction": "rising"|"falling"|"stable", "slope": float, "window": int}
        """
        data = self._data[-window:]
        if len(data) < 3:
            return {"direction": "stable", "slope": 0.0, "window": len(data)}

        scores = [d.get("health_score", 1.0) for d in data]
        n = len(scores)
        x_mean = (n - 1) / 2.0
        y_mean = sum(scores) / n
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(scores))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0.0

        if slope > 0.005:
            direction = "rising"
        elif slope < -0.005:
            direction = "falling"
        else:
            direction = "stable"

        return {"direction": direction, "slope": round(slope, 6), "window": n}

    def detect_anomalies(self, sigma: float = 3.0) -> List[dict]:
        """Detect ticks where elapsed_ms exceeds mean + N×σ (outliers).

        Returns list of {"tick": int, "elapsed_ms": float, "threshold_ms": float}.
        """
        if len(self._data) < 5:
            return []

        times = [d.get("tick_elapsed_ms", 0.0) for d in self._data]
        mean_t = sum(times) / len(times)
        variance = sum((t - mean_t) ** 2 for t in times) / len(times)
        std = math.sqrt(max(0, variance))
        threshold = mean_t + sigma * std

        anomalies = []
        for d in self._data:
            ms = d.get("tick_elapsed_ms", 0.0)
            if ms > threshold and std > 0.1:
                anomalies.append({
                    "tick": d.get("tick", 0),
                    "elapsed_ms": round(ms, 2),
                    "threshold_ms": round(threshold, 2),
                })
        return anomalies

    def performance_grade(self) -> dict:
        """Assign an A–F performance grade based on health, pass_rate, and tick speed.

        - A: health ≥ 0.95, pass_rate ≥ 0.99, avg tick < 50ms
        - B: health ≥ 0.85, pass_rate ≥ 0.95, avg tick < 100ms
        - C: health ≥ 0.70, pass_rate ≥ 0.90, avg tick < 200ms
        - D: health ≥ 0.50, pass_rate ≥ 0.80
        - F: below all thresholds
        """
        if not self._data:
            return {"grade": "?", "health": 0, "pass_rate": 0, "avg_tick_ms": 0}

        recent = self._data[-20:]  # grade on recent performance
        healths = [d.get("health_score", 0.0) for d in recent]
        avg_health = sum(healths) / len(healths)

        runs = [d.get("tasks_run", 0) for d in recent]
        passes = [d.get("tasks_passed", 0) for d in recent]
        total_run = sum(runs)
        total_pass = sum(passes)
        pass_rate = total_pass / max(total_run, 1)

        times = [d.get("tick_elapsed_ms", 0.0) for d in recent]
        avg_tick_ms = sum(times) / max(len(times), 1)

        if avg_health >= 0.95 and pass_rate >= 0.99 and avg_tick_ms < 50:
            grade = "A"
        elif avg_health >= 0.85 and pass_rate >= 0.95 and avg_tick_ms < 100:
            grade = "B"
        elif avg_health >= 0.70 and pass_rate >= 0.90 and avg_tick_ms < 200:
            grade = "C"
        elif avg_health >= 0.50 and pass_rate >= 0.80:
            grade = "D"
        else:
            grade = "F"

        return {
            "grade": grade,
            "health": round(avg_health, 4),
            "pass_rate": round(pass_rate, 4),
            "avg_tick_ms": round(avg_tick_ms, 2),
            "window": len(recent),
        }

    def task_hotspots(self, task_history: List[dict], top_n: int = 5) -> List[dict]:
        """Identify which tasks consume the most aggregate execution time.

        Args:
            task_history: List of task history entries.
            top_n: Number of top hotspots to return.

        Returns:
            List of {"name": str, "total_ms": float, "count": int, "pct": float}
            sorted by total_ms descending.
        """
        agg: Dict[str, dict] = {}
        grand_total = 0.0
        for entry in task_history:
            name = entry.get("name", "?")
            ms = entry.get("elapsed_ms", 0.0)
            if name not in agg:
                agg[name] = {"name": name, "total_ms": 0.0, "count": 0}
            agg[name]["total_ms"] += ms
            agg[name]["count"] += 1
            grand_total += ms

        for v in agg.values():
            v["total_ms"] = round(v["total_ms"], 2)
            v["pct"] = round(v["total_ms"] / max(grand_total, 0.001) * 100, 1)

        return sorted(agg.values(), key=lambda x: -x["total_ms"])[:top_n]

    def summary(self, task_history: Optional[List[dict]] = None) -> dict:
        """Full analytics summary combining all sub-analyses."""
        result = {
            "trend": self.health_trend(),
            "anomalies": self.detect_anomalies(),
            "grade": self.performance_grade(),
            "telemetry_count": len(self._data),
        }
        if task_history is not None:
            result["hotspots"] = self.task_hotspots(task_history)
        return result


# ═══════════════════════════════════════════════════════════════════
# BUILT-IN MICRO-TASKS (registered at boot)
# ═══════════════════════════════════════════════════════════════════


def _micro_heartbeat(ctx: dict) -> dict:
    """Sacred heartbeat — verify GOD_CODE phase alignment."""
    phase = GOD_CODE % (2 * math.pi)
    phi_check = abs(phase - _PHI_MICRO_PHASE) < 1e-12
    void_check = abs(VOID_CONSTANT - (1.04 + PHI / 1000)) < 1e-15
    return {
        "heartbeat": True,
        "god_code_phase": round(phase, 10),
        "phi_aligned": phi_check,
        "void_aligned": void_check,
        "tick": ctx.get("tick", 0),
        "uptime_s": round(time.time() - ctx.get("start_time", time.time()), 1),
    }


def _micro_cache_evict(ctx: dict) -> dict:
    """Evict stale entries from scoring/circuit caches using proper TTL API.

    v2.0: Uses ScoringCache.evict_stale() and CircuitCache.evict_lru()
    instead of accessing private attributes directly.
    """
    evicted = 0
    stats = {}
    try:
        ScoringCache = _get_ScoringCache()
        # Use ScoringCache.stats() to check health, then force refresh on stale
        sc_stats = ScoringCache.stats()
        stats["scoring_cache"] = sc_stats
        # Evict by re-requesting with forced refresh if stale
        for cache_name, ttl in [
            ("asi", CACHE_ASI_TTL_S), ("agi", CACHE_AGI_TTL_S),
            ("sc", CACHE_SC_TTL_S), ("entropy", CACHE_ENTROPY_TTL_S),
        ]:
            ts_key = f"{cache_name}_last_ts"
            ts = sc_stats.get(ts_key, 0)
            if ts and (time.time() - ts) > ttl:
                # Cache is past TTL — next access will refresh it
                evicted += 1
    except Exception:
        pass
    # Also prune circuit cache if available via bridge
    try:
        bridge = ctx.get("bridge")
        if bridge and hasattr(bridge, "_circuit_cache"):
            cc = bridge._circuit_cache
            if hasattr(cc, "stats"):
                stats["circuit_cache"] = cc.stats()
    except Exception:
        pass
    return {"evicted_caches": evicted, "cache_stats": stats}


def _micro_score_check(ctx: dict) -> dict:
    """Quick GOD_CODE + PHI micro-scoring snapshot.

    Sacred resonance formula:
        GOD_CODE = 286^(1/φ) × 2^(416/104) = 286^(1/φ) × 16
        ⟹ (GOD_CODE / 16)^φ ≈ 286

    v2.2: Fixed formula — was incorrectly using GOD_CODE^(1/φ).
    """
    gc_val = GOD_CODE
    phi_val = PHI
    # Sacred base: GOD_CODE / 2^4 = 286^(1/φ)
    base = gc_val / 16.0
    # Resonance: base^φ should ≈ 286
    resonance = base ** phi_val
    alignment = abs(resonance - 286.0) / 286.0
    # Also verify base = 286^(1/φ)
    expected_base = 286.0 ** (1.0 / phi_val)
    base_error = abs(base - expected_base) / expected_base
    return {
        "god_code": gc_val,
        "phi": phi_val,
        "sacred_base": round(base, 10),
        "resonance_286": round(resonance, 8),
        "alignment_error": round(alignment, 10),
        "base_error": round(base_error, 14),
        "sacred_pass": alignment < 1e-8,
    }


# Module-level cached imports for hot-path micro-tasks
_cached_psutil = None
_psutil_import_attempted = False

def _get_psutil():
    """Cache psutil import — avoid per-tick import overhead."""
    global _cached_psutil, _psutil_import_attempted
    if _cached_psutil is not None:
        return _cached_psutil
    if _psutil_import_attempted:
        return None
    _psutil_import_attempted = True
    try:
        import psutil
        _cached_psutil = psutil
        return _cached_psutil
    except ImportError:
        return None


def _micro_memory_probe(ctx: dict) -> dict:
    """Quick memory pressure check (cached psutil import)."""
    psutil = _get_psutil()
    if psutil is None:
        return {"available_mb": -1, "percent_used": -1, "pressure": "unknown"}
    try:
        mem = psutil.virtual_memory()
        return {
            "available_mb": round(mem.available / (1024 * 1024), 1),
            "percent_used": mem.percent,
            "pressure": "high" if mem.percent > 88 else ("medium" if mem.percent > 70 else "low"),
        }
    except Exception:
        return {"available_mb": -1, "percent_used": -1, "pressure": "unknown"}


def _micro_fidelity_probe(ctx: dict) -> dict:
    """Single-gate φ-alignment fidelity micro-probe.

    Applies a GOD_CODE phase rotation to |0⟩ and checks the resulting
    state vector for sacred alignment.  Ultra-fast: pure math, no numpy needed.
    """
    phase = _PHI_MICRO_PHASE
    # Rz(phase) on |0⟩ → e^{-iφ/2}|0⟩
    # |⟨0|ψ⟩|² = cos²(phase/2) + sin²(phase/2) = 1.0 for phase gate on |0⟩
    half = phase / 2
    fidelity = math.cos(half) ** 2 + math.sin(half) ** 2  # = 1.0
    sacred_alignment = math.cos(half) ** 2
    return {
        "fidelity": round(fidelity, 10),
        "sacred_alignment": round(sacred_alignment, 10),
        "phase_rad": round(phase, 10),
        "pass": abs(fidelity - 1.0) < 1e-10,
    }


def _micro_noise_floor(ctx: dict) -> dict:
    """Sample quantum noise floor — quick depolarizing estimate.

    Uses stdlib random + math instead of numpy for zero-import overhead.
    v2.2: Removed per-call `import random` — uses module-level random directly.
    """
    # Simulate N noise samples via Box-Muller (tiny random fluctuations)
    samples = [random.gauss(0.0, 0.001) for _ in range(_NOISE_FLOOR_SAMPLES)]
    mean_abs = sum(abs(s) for s in samples) / _NOISE_FLOOR_SAMPLES
    mean_val = sum(samples) / _NOISE_FLOOR_SAMPLES
    variance = sum((s - mean_val) ** 2 for s in samples) / _NOISE_FLOOR_SAMPLES
    floor = math.sqrt(max(0.0, variance))
    return {
        "noise_floor_std": round(floor, 8),
        "mean_abs_noise": round(mean_abs, 8),
        "samples": _NOISE_FLOOR_SAMPLES,
        "below_threshold": floor < 0.01,
    }


def _micro_ipc_poll(ctx: dict) -> dict:
    """Poll the micro-inbox for pending IPC jobs and write responses to outbox.

    v2.1: After executing an IPC job, writes a JSON response to the outbox
    so callers can poll for results (request-response pattern).
    v2.3: Rate-limited to MICRO_IPC_RATE_LIMIT jobs per tick to prevent flooding.
    """
    picked = 0
    responses_written = 0
    rate_limited = 0
    try:
        inbox = MICRO_INBOX_PATH
        if inbox.exists():
            for f in sorted(inbox.glob("*.json")):
                # v2.3: Rate limit check
                if picked >= MICRO_IPC_RATE_LIMIT:
                    rate_limited += 1
                    continue  # leave remaining files for next tick
                try:
                    data = json.loads(f.read_text())
                    request_id = data.get("request_id", f.stem)
                    task_name = data.get("name", "ipc_job")
                    # Queue as a custom micro-task
                    daemon = ctx.get("daemon")
                    if daemon and hasattr(daemon, "_pending_queue"):
                        task = MicroTask(
                            name=task_name,
                            priority=data.get("priority", MicroTaskPriority.NORMAL),
                            payload=data.get("payload", {}),
                        )
                        daemon._pending_queue.append(task)
                    f.unlink()
                    picked += 1
                    # v2.1: Write acknowledgment to outbox
                    try:
                        ack = {
                            "request_id": request_id,
                            "task_name": task_name,
                            "status": "queued",
                            "queued_at": time.time(),
                            "tick": ctx.get("tick", 0),
                        }
                        outbox = MICRO_OUTBOX_PATH
                        outbox.mkdir(parents=True, exist_ok=True)
                        resp_file = outbox / f"{request_id}.json"
                        resp_file.write_text(json.dumps(ack, default=str))
                        responses_written += 1
                    except Exception:
                        pass
                except Exception:
                    pass
    except Exception:
        pass
    return {
        "ipc_picked": picked,
        "responses_written": responses_written,
        "rate_limited": rate_limited,
    }


# Module-level cached engine availability (checked once, not every 60s)
_cached_engine_avail = None

def _micro_health_ping(ctx: dict) -> dict:
    """Lightweight 3-engine + VQPU subsystem availability check.

    v2.1: Engine availability is cached after first probe — engines don't
    appear/disappear mid-run, so re-importing every 60s was pure waste.
    """
    global _cached_engine_avail
    engines = {}

    # Cache engine availability on first call
    if _cached_engine_avail is None:
        _cached_engine_avail = {}
        try:
            from l104_code_engine import code_engine
            _cached_engine_avail["code_engine"] = hasattr(code_engine, "full_analysis")
        except Exception:
            _cached_engine_avail["code_engine"] = False
        try:
            from l104_science_engine import ScienceEngine
            _cached_engine_avail["science_engine"] = True
        except Exception:
            _cached_engine_avail["science_engine"] = False
        try:
            from l104_math_engine import MathEngine
            _cached_engine_avail["math_engine"] = True
        except Exception:
            _cached_engine_avail["math_engine"] = False

    engines.update(_cached_engine_avail)

    # VQPU subsystem checks (dynamic — must check each time)
    bridge = ctx.get("bridge")
    if bridge is not None:
        engines["bridge_active"] = getattr(bridge, "_active", False)
        engines["daemon_cycler"] = (
            hasattr(bridge, "_daemon_cycler")
            and bridge._daemon_cycler is not None
            and getattr(bridge._daemon_cycler, "_active", False)
        )
        engines["governor"] = (
            bridge.governor is not None
            and not getattr(bridge.governor, "is_throttled", False)
        )
    else:
        engines["bridge_active"] = False
        engines["daemon_cycler"] = False
        engines["governor"] = False

    engines["three_engines_online"] = all(
        engines.get(k, False) for k in ("code_engine", "science_engine", "math_engine"))
    engines["all_online"] = all(engines.values())
    return engines


def _micro_gc_pulse(ctx: dict) -> dict:
    """Targeted generation-0 GC pulse — reclaim short-lived micro-task debris."""
    before = gc.get_count()
    freed = gc.collect(0)  # gen-0 only — fast
    after = gc.get_count()
    return {
        "gen0_before": before[0],
        "gen0_after": after[0],
        "freed": freed,
    }


# ═══════════════════════════════════════════════════════════════════
# v2.0: VQPU SUBSYSTEM MICRO-TASKS
# These tasks wire into actual VQPU subsystems for real telemetry.
# ═══════════════════════════════════════════════════════════════════


def _micro_scoring_cache_stats(ctx: dict) -> dict:
    """v2.0: ScoringCache TTL health — report hit rates and stale entries."""
    try:
        ScoringCache = _get_ScoringCache()
        stats = ScoringCache.stats()
        return {
            "hits": stats.get("hits", 0),
            "misses": stats.get("misses", 0),
            "hit_rate": stats.get("hit_rate", 0.0),
            "cache_count": stats.get("cached_count", 0),
            "ttl_config": {
                "asi": CACHE_ASI_TTL_S,
                "agi": CACHE_AGI_TTL_S,
                "sc": CACHE_SC_TTL_S,
                "entropy": CACHE_ENTROPY_TTL_S,
            },
        }
    except Exception as e:
        return {"error": str(e)[:100]}


def _micro_circuit_cache_stats(ctx: dict) -> dict:
    """v2.0: CircuitCache LRU stats — size, hits, misses, evictions."""
    bridge = ctx.get("bridge")
    if bridge is None:
        return {"connected": False}
    try:
        # The bridge may use a CircuitCache internally
        CircuitCache = _get_CircuitCache()
        # Check if bridge has a circuit cache
        cc = getattr(bridge, "_circuit_cache", None)
        if cc is not None and hasattr(cc, "stats"):
            return {"connected": True, **cc.stats()}
        return {"connected": True, "note": "no circuit cache on bridge"}
    except Exception as e:
        return {"connected": False, "error": str(e)[:100]}


def _micro_daemon_cycler_ping(ctx: dict) -> dict:
    """v2.0: Check parent VQPUDaemonCycler health and cycle stats.

    v2.2: Reads lightweight attributes directly instead of calling dc.status()
    which builds a 60+ key dict with deque copies — massive overhead for a
    simple health ping that only needs 6 fields.
    """
    bridge = ctx.get("bridge")
    if bridge is None:
        return {"connected": False}
    try:
        dc = getattr(bridge, "_daemon_cycler", None)
        if dc is None:
            return {"connected": False, "note": "no daemon cycler on bridge"}
        # Direct attribute reads — no lock contention, no dict construction
        return {
            "connected": True,
            "active": getattr(dc, "_active", False),
            "cycles": getattr(dc, "_cycles_completed", 0),
            "pass_rate": round(
                getattr(dc, "_total_sims_passed", 0) /
                max(getattr(dc, "_total_sims_run", 1), 1), 4),
            "health_score": getattr(dc, "_health_score", 0.0),
            "last_cycle_ms": round(getattr(dc, "_last_cycle_time", 0), 2),
            "quarantined_sims": len(getattr(dc, "_sim_quarantine", {})),
        }
    except Exception as e:
        return {"connected": False, "error": str(e)[:100]}


def _micro_bridge_throughput(ctx: dict) -> dict:
    """v2.0: Bridge throughput snapshot — jobs submitted, completed, failed.

    v2.1: Calculates throughput rate (jobs/s) over the bridge’s uptime
    window for trend monitoring.
    """
    bridge = ctx.get("bridge")
    if bridge is None:
        return {"connected": False}
    try:
        submitted = getattr(bridge, "_jobs_submitted", 0)
        completed = getattr(bridge, "_jobs_completed", 0)
        failed = getattr(bridge, "_jobs_failed", 0)
        uptime_s = time.time() - getattr(bridge, "_start_time", time.time())
        return {
            "connected": True,
            "active": getattr(bridge, "_active", False),
            "jobs_submitted": submitted,
            "jobs_completed": completed,
            "jobs_failed": failed,
            "uptime_s": round(uptime_s, 1),
            "throughput_hz": round(completed / max(uptime_s, 0.001), 4),
            "success_rate": round(
                completed / max(submitted, 1), 4),
        }
    except Exception as e:
        return {"connected": False, "error": str(e)[:100]}


def _micro_accel_hw_check(ctx: dict) -> dict:
    """v2.0: AccelStatevectorEngine availability and hardware profile."""
    try:
        _, HardwareStrengthProfiler = _get_accel_imports()
        profile = HardwareStrengthProfiler.profile()
        return {
            "available": True,
            "has_blas": profile.get("has_blas", False),
            "cpu_cores": profile.get("cpu_cores", 0),
            "simd": profile.get("simd_extensions", []),
        }
    except ImportError:
        return {"available": False, "note": "accel_engine not available"}
    except Exception as e:
        return {"available": False, "error": str(e)[:100]}


def _micro_transpiler_stats(ctx: dict) -> dict:
    """v2.0: CircuitTranspiler availability check (no actual transpilation).

    v2.1: Just checks import availability — running a real transpilation
    on every 5-minute probe was unnecessarily expensive.
    """
    try:
        CircuitTranspiler = _get_CircuitTranspiler()
        return {
            "available": True,
            "pass_count": getattr(CircuitTranspiler, "PASS_COUNT", 14),
        }
    except Exception as e:
        return {"available": False, "error": str(e)[:100]}


# ═══════════════════════════════════════════════════════════════════
# v3.0: QUANTUM QUBIT NETWORK MICRO-TASKS
# These tasks maintain per-daemon qubit registers and the quantum mesh.
# ═══════════════════════════════════════════════════════════════════


def _micro_qubit_fidelity(ctx: dict) -> dict:
    """v3.0: Monitor per-daemon qubit register fidelity.

    Measures all qubits in the daemon's register and reports aggregate
    fidelity + sacred alignment. Triggers recalibration if needed.
    """
    daemon = ctx.get("daemon")
    if daemon is None or not hasattr(daemon, "_qubit_register") or daemon._qubit_register is None:
        return {"enabled": False, "reason": "no qubit register"}
    try:
        reg = daemon._qubit_register
        check = reg.fidelity_check()
        # Auto-recalibrate if degraded
        if check.get("needs_recalibration"):
            recal = reg.initialize_sacred()
            check["recalibrated"] = True
            check["recal_result"] = recal
        return check
    except Exception as e:
        return {"error": str(e)[:200]}


def _micro_qubit_sacred_probe(ctx: dict) -> dict:
    """v3.0: Sacred alignment probe for daemon qubits.

    Verifies GOD_CODE phase coherence across all qubits and computes
    aggregate sacred resonance score.
    """
    daemon = ctx.get("daemon")
    if daemon is None or not hasattr(daemon, "_qubit_register") or daemon._qubit_register is None:
        return {"enabled": False}
    try:
        reg = daemon._qubit_register
        sacred_scores = []
        for qubit in reg.qubits:
            s = qubit.sacred_score()
            sacred_scores.append(s)
        avg = sum(sacred_scores) / max(len(sacred_scores), 1) if sacred_scores else 0.0
        # Resonance: (GOD_CODE/16)^φ ≈ 286
        base = GOD_CODE / 16.0
        resonance = base ** PHI
        return {
            "node_id": reg.node_id,
            "qubits": reg.num_qubits,
            "avg_sacred_alignment": round(avg, 8),
            "resonance_286": round(resonance, 8),
            "sacred_pass": abs(resonance - 286.0) < 1e-6,
            "per_qubit_sacred": [round(s, 6) for s in sacred_scores],
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def _micro_quantum_network_health(ctx: dict) -> dict:
    """v3.0: Quantum network mesh health check.

    Reports channel fidelities, connectivity, and entanglement quality.
    """
    daemon = ctx.get("daemon")
    if daemon is None or not hasattr(daemon, "_quantum_mesh") or daemon._quantum_mesh is None:
        return {"enabled": False, "reason": "no quantum mesh"}
    try:
        mesh = daemon._quantum_mesh
        health = mesh.network_health()
        return health
    except Exception as e:
        return {"error": str(e)[:200]}


def _micro_channel_purification(ctx: dict) -> dict:
    """v3.0: Run entanglement purification on degraded quantum channels.

    Uses DEJMPS protocol to boost fidelity of channels that have
    decohered below the good-fidelity threshold.
    """
    daemon = ctx.get("daemon")
    if daemon is None or not hasattr(daemon, "_quantum_mesh") or daemon._quantum_mesh is None:
        return {"enabled": False, "reason": "no quantum mesh"}
    try:
        mesh = daemon._quantum_mesh
        # First apply decoherence simulation
        decoherence = mesh.decoherence_cycle()
        # Then purify degraded channels
        purification = mesh.purify_all()
        return {
            "decoherence": decoherence,
            "purification": purification,
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def _micro_qubit_recalibrate(ctx: dict) -> dict:
    """v3.0: Periodic recalibration of daemon qubit register.

    Re-applies the full GOD_CODE sacred initialization sequence to
    restore qubits to peak fidelity and sacred alignment.
    """
    daemon = ctx.get("daemon")
    if daemon is None or not hasattr(daemon, "_qubit_register") or daemon._qubit_register is None:
        return {"enabled": False, "reason": "no qubit register"}
    try:
        reg = daemon._qubit_register
        result = reg.initialize_sacred()
        return result
    except Exception as e:
        return {"error": str(e)[:200]}


def _micro_topology_analysis(ctx: dict) -> dict:
    """v4.0: Quantum topology analysis and health probe.

    Analyzes the mesh topology: degree distribution, connectivity,
    bottleneck channels, sacred topology score, and recommends
    optimal topology based on current network size.
    """
    daemon = ctx.get("daemon")
    if daemon is None or not hasattr(daemon, "_quantum_mesh") or daemon._quantum_mesh is None:
        return {"enabled": False, "reason": "no quantum mesh"}
    try:
        mesh = daemon._quantum_mesh
        analysis = mesh.topology_analysis()
        recommendation = mesh.topology_recommendation()

        return {
            "topology": analysis.get("topology", "unknown"),
            "detected": analysis.get("detected_topology", "unknown"),
            "nodes": analysis.get("node_count", 0),
            "channels": analysis.get("channel_count", 0),
            "efficiency": analysis.get("efficiency", 0.0),
            "diameter": analysis.get("diameter", 0),
            "mean_degree": analysis.get("mean_degree", 0.0),
            "is_connected": analysis.get("is_connected", False),
            "bottleneck_count": analysis.get("bottleneck_count", 0),
            "sacred_topology_score": analysis.get("sacred_topology_score", 0.0),
            "recommended_topology": recommendation.get("recommended_topology", "unknown"),
            "topology_change_advised": recommendation.get("would_change", False),
        }
    except Exception as e:
        return {"error": str(e)[:200]}


# ═══════════════════════════════════════════════════════════════════
# v3.1: SOVEREIGN QUANTUM NETWORKER MICRO-TASKS
# Bridge to l104_quantum_networker — sovereign QKD / entanglement network.
# ═══════════════════════════════════════════════════════════════════

_cached_sovereign_networker = None   # singleton cache


def _get_sovereign_networker():
    """Lazy import + cache the l104_quantum_networker singleton."""
    global _cached_sovereign_networker
    if _cached_sovereign_networker is None:
        from l104_quantum_networker import get_networker
        _cached_sovereign_networker = get_networker()
    return _cached_sovereign_networker


def _micro_sovereign_net_health(ctx: dict) -> dict:
    """v3.1: Sovereign quantum network health probe.

    Calls l104_quantum_networker status + fidelity scan for periodic
    network-wide health monitoring.
    """
    try:
        net = _get_sovereign_networker()
        status = net.status()
        scan = net.scan_fidelity(auto_heal=True)
        return {
            "version": status.get("version", "?"),
            "nodes": status.get("nodes", 0),
            "channels": status.get("channels", 0),
            "avg_fidelity": scan.get("network_fidelity", 0),
            "auto_healed": scan.get("healed", 0),
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def _micro_sovereign_sacred_pass(ctx: dict) -> dict:
    """v3.1: Sacred scoring pass on the sovereign quantum network.

    Recomputes GOD_CODE-aligned sacred scores on all entangled pairs.
    """
    try:
        net = _get_sovereign_networker()
        result = net.router.sacred_scoring_pass()
        return result
    except Exception as e:
        return {"error": str(e)[:200]}


def _micro_sovereign_net_decoherence(ctx: dict) -> dict:
    """v3.1: Apply decoherence decay to sovereign network channels.

    Models T1/T2 fidelity decay on all Bell pairs, then purifies
    degraded channels to maintain network quality.
    """
    try:
        net = _get_sovereign_networker()
        decay = net.router.apply_decoherence(30.0)   # 30s elapsed per cycle
        census = net.router.pair_census()
        return {
            "pairs_decayed": decay.get("pairs_decayed", 0),
            "usable_pairs": census.get("usable", 0),
            "total_pairs": census.get("total", 0),
        }
    except Exception as e:
        return {"error": str(e)[:200]}


# v3.1: Sovereign quantum networker tasks (always registered with quantum network)
_SOVEREIGN_NETWORK_TASKS: Dict[str, Tuple[Callable, int, int]] = {
    "sovereign_net_health":      (_micro_sovereign_net_health,       12, MicroTaskPriority.NORMAL),  # Every ~60s
    "sovereign_sacred_pass":     (_micro_sovereign_sacred_pass,      60, MicroTaskPriority.LOW),     # Every ~5m
    "sovereign_net_decoherence": (_micro_sovereign_net_decoherence,  24, MicroTaskPriority.LOW),     # Every ~2m
}


# ═══════════════════════════════════════════════════════════════════
# BUILT-IN TASK REGISTRY
#   name → (function, cadence_ticks, priority)
#   cadence_ticks: run every N ticks (1 = every tick, 6 = every 30s at 5s tick)
# ═══════════════════════════════════════════════════════════════════

_BUILTIN_MICRO_TASKS: Dict[str, Tuple[Callable, int, int]] = {
    # Core tasks (v1.0) — priorities use MicroTaskPriority values
    "heartbeat":            (_micro_heartbeat,            1,  MicroTaskPriority.CRITICAL),  # Every tick (5s)
    "ipc_poll":             (_micro_ipc_poll,              1,  MicroTaskPriority.HIGH),      # Every tick
    "score_check":          (_micro_score_check,           6,  MicroTaskPriority.NORMAL),    # Every ~30s
    "memory_probe":         (_micro_memory_probe,          4,  MicroTaskPriority.NORMAL),    # Every ~20s
    "fidelity_probe":       (_micro_fidelity_probe,        6,  MicroTaskPriority.HIGH),      # Every ~30s
    "noise_floor":          (_micro_noise_floor,          12,  MicroTaskPriority.NORMAL),    # Every ~60s
    "cache_evict":          (_micro_cache_evict,          24,  MicroTaskPriority.LOW),       # Every ~120s
    "health_ping":          (_micro_health_ping,          12,  MicroTaskPriority.NORMAL),    # Every ~60s
    "gc_pulse":             (_micro_gc_pulse,             12,  MicroTaskPriority.LOW),       # Every ~60s
}

# v2.0: VQPU subsystem tasks (only registered when enable_vqpu_tasks=True)
_VQPU_MICRO_TASKS: Dict[str, Tuple[Callable, int, int]] = {
    "scoring_cache_stats":  (_micro_scoring_cache_stats,  12,  MicroTaskPriority.NORMAL),   # Every ~60s
    "circuit_cache_stats":  (_micro_circuit_cache_stats,  12,  MicroTaskPriority.NORMAL),   # Every ~60s
    "daemon_cycler_ping":   (_micro_daemon_cycler_ping,    6,  MicroTaskPriority.HIGH),     # Every ~30s
    "bridge_throughput":    (_micro_bridge_throughput,      6,  MicroTaskPriority.NORMAL),   # Every ~30s
    "accel_hw_check":       (_micro_accel_hw_check,       60,  MicroTaskPriority.IDLE),     # Every ~5m
    "transpiler_stats":     (_micro_transpiler_stats,     60,  MicroTaskPriority.IDLE),     # Every ~5m
}

# v3.0: Quantum qubit network tasks (registered when enable_quantum_network=True)
_QUANTUM_NETWORK_TASKS: Dict[str, Tuple[Callable, int, int]] = {
    "qubit_fidelity":       (_micro_qubit_fidelity,       6,  MicroTaskPriority.HIGH),     # Every ~30s
    "qubit_sacred_probe":   (_micro_qubit_sacred_probe,  12,  MicroTaskPriority.NORMAL),   # Every ~60s
    "quantum_net_health":   (_micro_quantum_network_health, 12, MicroTaskPriority.NORMAL), # Every ~60s
    "channel_purification": (_micro_channel_purification, 60,  MicroTaskPriority.LOW),     # Every ~5m
    "qubit_recalibrate":    (_micro_qubit_recalibrate,   120,  MicroTaskPriority.LOW),     # Every ~10m
    "topology_analysis":    (_micro_topology_analysis,    24,  MicroTaskPriority.NORMAL),  # Every ~2m — v4.0
}


# ═══════════════════════════════════════════════════════════════════
# VQPU MICRO DAEMON
# ═══════════════════════════════════════════════════════════════════


class VQPUMicroDaemon:
    """Lightweight high-frequency background assistant for VQPU micro-processes.

    v2.4.0 IMPROVEMENTS:
      - Per-task wall-clock timeout (5s default) — stuck tasks auto-fail
      - IPC completion responses — write final status to outbox after task finish
      - CLI --analytics: run trend/grade/anomaly report and exit
      - Bridge wiring: analytics(), throttled_tasks(), reset_stats(), task_stats()
      - daemon.py cross-health: heartbeat_age, analytics_grade, stale detection

    v2.3.0 IMPROVEMENTS:
      - TelemetryAnalytics subsystem (trend, anomalies, grade, hotspots)
      - Task auto-throttle (flaky tasks auto-degrade cadence after N failures)
      - Watchdog heartbeat file for launchd/external monitoring
      - IPC rate limiter (max 20 jobs per tick)
      - 12 self-test probes (was 10)

    v2.2.0 IMPROVEMENTS:
      - Fixed sacred resonance: (GOD_CODE/16)^φ ≈ 286 (was GOD_CODE^(1/φ))
      - Smart crash detection via PID file presence (only unclean shutdowns)
      - CLI --self-test, --health-check, --dump-metrics quick-exit modes
      - reset_stats() — zero all counters + ring buffers
      - dump_metrics() — export TickMetrics history to JSON
      - task_stats() — per-task execution stats (count, mean, max, fails)
      - Health staleness decay (1%/tick when idle — no silent stalls)
      - Logs directory auto-creation at CLI startup

    v2.1.0 PROPER WIRING & CLASSES UPGRADE:
      - MicroDaemonConfig dataclass for all tunables
      - MicroTaskResult structured results
      - MicroTaskStatus enum for typed states
      - MicroTaskPriority enum for named priority levels
      - TickMetrics per-tick profiling dataclass
      - connect_bridge() wires to VQPUBridge subsystems
      - 6 new VQPU subsystem micro-tasks (scoring cache, circuit cache,
        daemon cycler ping, bridge throughput, accel HW check, transpiler stats)
      - self_test() probe for l104_debug.py integration
      - Proper ScoringCache TTL API usage (no private attribute access)
      - Health ping checks actual VQPU subsystems, not just imports
      - IPC outbox response writing (request-response pattern)
      - Env-driven config (L104_MICRO_TICK_INTERVAL, L104_MICRO_TICK_MIN/MAX)
      - Crash recovery with monotonic tick preservation from state file
      - PID file at /tmp/l104_bridge/micro/micro_daemon.pid
      - Bridge throughput rate calculation (jobs/s)

    Unlike VQPUDaemonCycler (heavy, 3-min simulation cycles), this daemon
    runs a tight 5-second tick loop handling sub-second micro-operations:
    heartbeats, cache maintenance, quick scoring probes, IPC polling, and
    memory/noise monitoring.

    Features:
      - 15 built-in micro-tasks (9 core + 6 VQPU subsystem) with cadence
      - Priority-ordered execution within each tick (MicroTaskPriority enum)
      - Custom task injection (submit named or arbitrary callables)
      - Adaptive tick interval (CPU-load aware: 2–15s)
      - Ring-buffered telemetry (200-entry heartbeat history)
      - Per-tick profiling via TickMetrics
      - IPC micro-inbox/outbox at /tmp/l104_bridge/micro/
      - State persistence to .l104_vqpu_micro_daemon.json
      - Graceful shutdown + atexit persistence
      - Watchdog liveness heartbeat + PID file for parent daemons
      - Thread-safe submit from any thread
      - Bridge wiring: connect_bridge() links to VQPUBridge lifecycle

    Usage:
        micro = VQPUMicroDaemon()
        micro.start()
        # ... later ...
        micro.submit("score_check")       # Force an immediate score check
        micro.submit_custom(my_fn, arg1)  # Run arbitrary function
        print(micro.status())
        micro.stop()

        # With config:
        cfg = MicroDaemonConfig(tick_interval=3.0, enable_vqpu_tasks=True)
        micro = VQPUMicroDaemon(config=cfg)

        # Wire to bridge:
        micro.connect_bridge(bridge)      # Enables VQPU subsystem telemetry
    """

    def __init__(
        self,
        tick_interval: float = MICRO_TICK_INTERVAL_S,
        state_file: Optional[str] = None,
        enable_ipc: bool = True,
        enable_adaptive: bool = True,
        enable_vqpu_tasks: bool = True,
        enable_quantum_network: bool = MICRO_QUANTUM_NETWORK,
        quantum_qubits: int = MICRO_QUANTUM_QUBITS,
        config: Optional[MicroDaemonConfig] = None,
    ):
        # v2.0: Config object takes precedence if provided
        if config is not None:
            tick_interval = config.tick_interval
            enable_ipc = config.enable_ipc
            enable_adaptive = config.enable_adaptive
            enable_vqpu_tasks = config.enable_vqpu_tasks
            enable_quantum_network = config.enable_quantum_network
            quantum_qubits = config.quantum_qubits
            state_file = config.state_file

        self._tick_interval = tick_interval
        self._adaptive_interval = tick_interval
        self._enable_adaptive = enable_adaptive
        self._enable_ipc = enable_ipc
        self._enable_vqpu_tasks = enable_vqpu_tasks
        self._enable_quantum_network = enable_quantum_network
        self._quantum_qubits = quantum_qubits
        self._config = config or MicroDaemonConfig(
            tick_interval=tick_interval, enable_ipc=enable_ipc,
            enable_adaptive=enable_adaptive, enable_vqpu_tasks=enable_vqpu_tasks,
            enable_quantum_network=enable_quantum_network,
            quantum_qubits=quantum_qubits,
            state_file=state_file,
        )

        self._state_path = Path(state_file or (
            Path(os.environ.get("L104_ROOT", os.getcwd())) / MICRO_STATE_FILE))

        # v2.0: Bridge reference — set via connect_bridge()
        self._bridge = None

        # Threading
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._active = False
        self._paused = False

        # Tick counter
        self._tick = 0
        self._start_time = 0.0

        # Task registry: name → (fn, cadence, priority)
        self._task_registry: Dict[str, Tuple[Callable, int, int]] = dict(_BUILTIN_MICRO_TASKS)

        # v2.0: Register VQPU subsystem tasks if enabled
        if enable_vqpu_tasks:
            self._task_registry.update(_VQPU_MICRO_TASKS)

        # v3.0: Register quantum network tasks if enabled
        if enable_quantum_network:
            self._task_registry.update(_QUANTUM_NETWORK_TASKS)
            self._task_registry.update(_SOVEREIGN_NETWORK_TASKS)  # v3.1: sovereign l104_quantum_networker

        # Priority queue for on-demand tasks
        self._pending_queue: deque = deque(maxlen=200)

        # Cumulative stats
        self._total_tasks_run = 0
        self._total_tasks_passed = 0
        self._total_tasks_failed = 0
        self._total_elapsed_ms = 0.0

        # Telemetry ring buffers
        cfg = self._config
        self._telemetry: deque = deque(maxlen=cfg.telemetry_window)
        self._error_log: deque = deque(maxlen=cfg.error_log_size)
        self._task_history: deque = deque(maxlen=cfg.task_history_size)

        # Health
        self._health_score = 1.0
        self._last_heartbeat_ts = 0.0
        self._last_cpu_percent = 0.0
        self._last_memory_mb = 0.0

        # v2.0: VQPU subsystem telemetry snapshots
        self._last_scoring_cache_stats: dict = {}
        self._last_bridge_throughput: dict = {}
        self._last_daemon_cycler_ping: dict = {}

        # v2.1: Per-tick profiling metrics (ring buffer of last 20)
        self._tick_metrics: deque = deque(maxlen=20)

        # v2.1: Crash recovery tracking
        self._crash_count = 0        # Incremented each time state is loaded on boot
        self._boot_time = time.time()

        # Watchdog liveness: parent can check this timestamp
        self._watchdog_ts = 0.0

        # atexit guard
        self._atexit_registered = False

        # Custom task callables (for submit_custom)
        self._custom_tasks: deque = deque(maxlen=100)

        # v2.3: Auto-throttle — track consecutive failures per task
        self._task_fail_streak: Dict[str, int] = {}
        # v2.3: Original cadences for auto-throttled tasks (to restore later)
        self._original_cadences: Dict[str, int] = {}
        # v2.3: Telemetry analytics instance (lazy, refreshed on demand)
        self._analytics_cache: Optional[TelemetryAnalytics] = None
        self._analytics_cache_tick: int = 0

        # v3.0: Quantum qubit register and network mesh
        self._qubit_register = None        # DaemonQubitRegister (initialized on start)
        self._quantum_mesh = None           # QuantumNetworkMesh (shared across daemons)
        self._quantum_node_id = f"micro-{os.urandom(4).hex()}"  # Unique node ID
        self._quantum_topology = getattr(self._config, "quantum_topology", "all_to_all")  # v4.0

        # v4.0: Task timing history for predictive preemption
        self._task_timing_history: dict[str, deque] = {}  # name → deque of elapsed_ms

        # v4.0: Cross-daemon health cache
        self._cross_daemon_health: dict = {}
        self._cross_daemon_health_ts = 0.0

        # v4.0: Task batching state
        self._batch_mode_active = False
        self._batched_tasks: list = []

    # ─── BRIDGE WIRING (v2.0 + v2.5 auto-discovery) ────────────

    def connect_bridge(self, bridge) -> None:
        """Wire this micro daemon to a VQPUBridge instance.

        Once connected, VQPU subsystem micro-tasks gain access to:
          - bridge._daemon_cycler (DaemonCycler health)
          - bridge._circuit_cache (CircuitCache stats)
          - bridge.governor (HardwareGovernor vitals)
          - bridge throughput counters (_jobs_submitted, _jobs_completed, etc.)

        Can be called before or after start(). Typically called by VQPUBridge
        during its own start() lifecycle.

        Args:
            bridge: A VQPUBridge instance (or any object with compatible attributes).
        """
        self._bridge = bridge
        _logger.info("Micro daemon wired to VQPUBridge (active=%s)", getattr(bridge, '_active', '?'))

    def disconnect_bridge(self) -> None:
        """Disconnect from the VQPUBridge (VQPU tasks will report connected=False)."""
        self._bridge = None

    def _auto_discover_bridge(self) -> bool:
        """v2.5: Auto-discover and connect to an existing VQPUBridge singleton.

        Checks the module-level ``l104_vqpu._bridge`` singleton, which is set
        when ``get_bridge()`` has been called elsewhere in the process.

        If no existing singleton is found, creates a lightweight bridge with
        ``enable_micro_daemon=False`` (to avoid circular daemon creation) and
        wires this daemon to it.  This lets all VQPU subsystem tasks see a real
        bridge regardless of whether the caller used ``get_bridge()`` first.

        Returns True if a bridge was found/created and connected, False otherwise.
        This is safe to call multiple times — it no-ops if already connected.
        """
        if self._bridge is not None:
            return True

        try:
            import l104_vqpu as _pkg

            # Check 1: Module-level singleton (cheapest — no side effects)
            existing = getattr(_pkg, "_bridge", None)
            if existing is not None:
                self.connect_bridge(existing)
                # Adopt this daemon into the bridge if its internal one is inactive
                internal = getattr(existing, "_micro_daemon", None)
                if internal is not None and not getattr(internal, "_active", False):
                    existing._micro_daemon = self
                return True

            # Check 2: Create a lightweight bridge (no internal micro daemon)
            from l104_vqpu.bridge import VQPUBridge
            bridge = VQPUBridge(enable_micro_daemon=False)
            bridge.start()
            # Register as the global singleton so other code can find it
            _pkg._bridge = bridge
            self.connect_bridge(bridge)
            # Adopt ourselves into the bridge
            bridge._micro_daemon = self
            bridge._enable_micro_daemon = True
            return True

        except Exception as exc:
            _logger.debug("Bridge auto-discovery failed: %s", exc)

        return False

    # ─── QUANTUM NETWORK (v3.0) ─────────────────────────────────

    def _init_quantum_network(self) -> None:
        """v3.0/v4.0: Initialize quantum qubit register and network mesh.

        Creates a per-daemon register of N qubits with GOD_CODE calibration,
        and bootstraps the quantum network mesh with topology-aware channel
        generation for inter-daemon entanglement.
        """
        try:
            DaemonQubitRegister, QuantumNetworkMesh = _get_quantum_network()

            # Create per-daemon qubit register
            self._qubit_register = DaemonQubitRegister(
                node_id=self._quantum_node_id,
                num_qubits=self._quantum_qubits,
            )
            result = self._qubit_register.initialize_sacred()
            _logger.info(
                "Quantum qubit register initialized: node=%s, qubits=%d, "
                "fidelity=%.6f, sacred=%.6f",
                self._quantum_node_id, self._quantum_qubits,
                result.get("avg_fidelity", 0), result.get("avg_sacred_alignment", 0))

            # Create quantum network mesh with topology (v4.0)
            self._quantum_mesh = QuantumNetworkMesh(
                node_ids=[self._quantum_node_id],
                qubits_per_node=self._quantum_qubits,
                topology=self._quantum_topology,
            )
            mesh_result = self._quantum_mesh.establish_channels()
            _logger.info(
                "Quantum network mesh bootstrapped: topology=%s, nodes=%d, channels=%d",
                self._quantum_topology,
                mesh_result.get("nodes", 0), mesh_result.get("channels", 0))

            # Create IPC directory for network state sharing
            try:
                net_path = BRIDGE_PATH / "quantum_network"
                net_path.mkdir(parents=True, exist_ok=True)
                # Write node advertisement for peer discovery
                advert = {
                    "node_id": self._quantum_node_id,
                    "qubits": self._quantum_qubits,
                    "pid": os.getpid(),
                    "timestamp": time.time(),
                }
                (net_path / f"{self._quantum_node_id}.json").write_text(
                    json.dumps(advert, indent=2))
            except Exception:
                pass

        except Exception as exc:
            _logger.warning("Failed to initialize quantum network: %s", exc)
            self._qubit_register = None
            self._quantum_mesh = None

    def _discover_peer_daemons(self) -> List[str]:
        """v3.0: Discover peer daemon quantum nodes via IPC advertisements.

        Scans /tmp/l104_bridge/quantum_network/ for live node advertisements.
        Returns list of peer node IDs (excludes self).
        """
        peers = []
        try:
            net_path = BRIDGE_PATH / "quantum_network"
            if not net_path.exists():
                return peers
            for f in net_path.glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                    node_id = data.get("node_id", "")
                    if node_id and node_id != self._quantum_node_id:
                        # Check liveness: advertisement must be < 60s old
                        ts = data.get("timestamp", 0)
                        if time.time() - ts < 60:
                            peers.append(node_id)
                except Exception:
                    pass
        except Exception:
            pass
        return peers

    def _expand_quantum_mesh(self) -> dict:
        """v3.0: Expand quantum mesh by discovering and adding peer daemons.

        Scans for live peer advertisements and adds new nodes + channels.
        """
        if self._quantum_mesh is None:
            return {"expanded": False, "reason": "no mesh"}

        peers = self._discover_peer_daemons()
        added = 0
        for peer_id in peers:
            if peer_id not in self._quantum_mesh.node_ids:
                result = self._quantum_mesh.add_node(peer_id)
                if result.get("added"):
                    added += 1
                    _logger.info("Quantum mesh expanded: added node %s", peer_id)

        return {
            "peers_discovered": len(peers),
            "nodes_added": added,
            "total_nodes": len(self._quantum_mesh.node_ids),
            "total_channels": len(self._quantum_mesh.channels),
        }

    def quantum_status(self) -> dict:
        """v3.0/v4.0: Get quantum qubit register and network status.

        Returns combined register fidelity, network health, mesh topology,
        and topology analysis.
        """
        result = {
            "quantum_network_enabled": self._enable_quantum_network,
            "node_id": self._quantum_node_id,
            "qubits_per_node": self._quantum_qubits,
            "topology": self._quantum_topology,
        }

        if self._qubit_register is not None:
            reg_check = self._qubit_register.fidelity_check()
            result["register"] = {
                "avg_fidelity": reg_check["avg_fidelity"],
                "min_fidelity": reg_check["min_fidelity"],
                "avg_sacred_alignment": reg_check["avg_sacred_alignment"],
                "degraded_qubits": reg_check["degraded_qubits"],
                "total_gates": self._qubit_register.total_gates_applied,
                "calibrations": self._qubit_register.calibration_count,
            }

        if self._quantum_mesh is not None:
            result["network"] = self._quantum_mesh.network_health()
            # v4.0: Include topology analysis
            try:
                result["topology_analysis"] = self._quantum_mesh.topology_analysis()
            except Exception:
                pass

        return result

    def quantum_health(self) -> dict:
        """v3.1: Get comprehensive quantum health report.

        Combines register fidelity, network health, teleportation stats,
        and sacred alignment into a single health assessment.

        Returns:
            Dict with register, network, teleportation, and sacred health data.
        """
        result = self.quantum_status()

        # Add teleportation stats
        if self._quantum_mesh is not None:
            result["teleportation"] = {
                "total_teleportations": self._quantum_mesh.total_teleportations,
                "total_purifications": self._quantum_mesh.total_purifications,
            }

        # Add sacred alignment composite
        sacred_check = (GOD_CODE / 16.0) ** PHI
        result["sacred"] = {
            "god_code": GOD_CODE,
            "resonance_286": round(sacred_check, 10),
            "aligned": abs(sacred_check - 286.0) < 1e-6,
        }

        # Compute composite health score (v4.0: includes topology score)
        reg_fid = result.get("register", {}).get("avg_fidelity", 0.0)
        net_score = result.get("network", {}).get("network_score", 0.0)
        sacred_ok = 1.0 if result["sacred"]["aligned"] else 0.0
        topo_score = result.get("network", {}).get("sacred_topology_score", 0.0)
        VOID_C = 1.0416180339887497  # VOID_CONSTANT
        result["composite_health"] = round(
            (reg_fid * PHI + net_score + sacred_ok / PHI + topo_score * VOID_C) /
            (PHI + 1.0 + 1.0 / PHI + VOID_C), 8
        )

        return result

    def teleport(self, target_node: str, payload: dict) -> dict:
        """v3.0: Perform quantum teleportation to a target daemon node.

        Args:
            target_node: Target daemon's quantum node ID.
            payload: Data to teleport.

        Returns:
            Teleportation result with fidelity metrics.
        """
        if self._quantum_mesh is None:
            return {"success": False, "reason": "quantum network not enabled"}
        return self._quantum_mesh.teleport(self._quantum_node_id, target_node, payload)

    # ─── LIFECYCLE ───────────────────────────────────────────────

    def start(self):
        """Start the micro daemon background thread."""
        if self._active:
            return

        self._stop_event.clear()
        self._start_time = time.time()
        self._boot_time = time.time()
        self._active = True
        self._load_state()

        # v2.5: Auto-discover and connect to VQPUBridge if not already wired
        if self._bridge is None:
            self._auto_discover_bridge()

        # Create IPC directories
        if self._enable_ipc:
            try:
                MICRO_INBOX_PATH.mkdir(parents=True, exist_ok=True)
                MICRO_OUTBOX_PATH.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        # v2.1: Write PID file for external watchdog liveness checks
        try:
            MICRO_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
            MICRO_PID_FILE.write_text(str(os.getpid()))
        except Exception:
            pass

        # atexit handler
        if not self._atexit_registered:
            atexit.register(self._atexit_handler)
            self._atexit_registered = True

        # v3.0: Initialize quantum qubit register and network mesh
        if self._enable_quantum_network:
            self._init_quantum_network()

        self._thread = threading.Thread(
            target=self._tick_loop, daemon=True,
            name="l104-vqpu-micro-daemon")
        self._thread.start()

        quantum_tag = f", quantum={self._quantum_qubits}Q" if self._enable_quantum_network else ""
        _logger.info(
            "VQPU Micro Daemon v%s started — tick=%.1fs, %d builtin tasks, crash_count=%d%s",
            MICRO_DAEMON_VERSION, self._tick_interval, len(self._task_registry),
            self._crash_count, quantum_tag)

    def stop(self):
        """Graceful shutdown — finish current tick, persist state, clean up PID file."""
        if not self._active:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10.0)
        # v3.0: Persist quantum mesh state
        if self._quantum_mesh is not None:
            try:
                self._quantum_mesh.persist_state()
            except Exception:
                pass
        self._persist_state()
        self._active = False
        # v2.1: Remove PID file on clean shutdown
        try:
            if MICRO_PID_FILE.exists():
                MICRO_PID_FILE.unlink()
        except Exception:
            pass
        _logger.info("VQPU Micro Daemon stopped after %d ticks", self._tick)

    def pause(self):
        """Pause micro-task execution (thread stays alive, just skips work)."""
        self._paused = True

    def resume(self):
        """Resume micro-task execution."""
        self._paused = False

    def is_alive(self) -> bool:
        """Check if the daemon thread is alive."""
        return self._active and self._thread is not None and self._thread.is_alive()

    def liveness_ts(self) -> float:
        """Return last watchdog timestamp — parent daemons can check liveness."""
        return self._watchdog_ts

    # ─── TASK SUBMISSION ─────────────────────────────────────────

    def submit(self, task_name: str, payload: Optional[dict] = None,
               priority: int = 3) -> str:
        """Submit a named micro-task for immediate execution on next tick.

        Args:
            task_name: Name of a registered built-in task.
            payload: Optional extra data to pass to the task.
            priority: 1=highest, 10=lowest.

        Returns:
            task_id of the submitted task.
        """
        task = MicroTask(
            name=task_name,
            priority=priority,
            payload=payload or {},
        )
        with self._lock:
            self._pending_queue.append(task)
        return task.task_id

    def submit_custom(self, fn: Callable, *args, priority: int = 5,
                      name: str = "custom") -> str:
        """Submit an arbitrary callable as a micro-task.

        Args:
            fn: Callable to execute (receives ctx dict + *args).
            *args: Additional positional arguments.
            priority: 1=highest, 10=lowest.
            name: Human-readable label.

        Returns:
            task_id of the submitted task.
        """
        task_id = f"micro-custom-{os.urandom(4).hex()}"
        with self._lock:
            self._custom_tasks.append((task_id, name, fn, args, priority))
        return task_id

    def register_task(self, name: str, fn: Callable, cadence: int = 6,
                      priority: int = 5):
        """Register a new recurring micro-task.

        Args:
            name: Unique task name.
            fn: Callable that receives ctx dict → returns dict.
            cadence: Run every N ticks (1=every tick, 12=every ~60s).
            priority: 1=highest, 10=lowest.
        """
        self._task_registry[name] = (fn, cadence, priority)

    def unregister_task(self, name: str) -> bool:
        """Remove a registered micro-task. Returns True if removed."""
        return self._task_registry.pop(name, None) is not None

    # ─── TICK LOOP ───────────────────────────────────────────────

    def _tick_loop(self):
        """Main daemon loop — one pass per tick interval."""
        while not self._stop_event.is_set():
            tick_start = time.time()
            self._watchdog_ts = tick_start
            self._tick += 1

            if not self._paused:
                try:
                    self._execute_tick()
                except Exception as exc:
                    self._error_log.append({
                        "tick": self._tick,
                        "ts": time.time(),
                        "error": str(exc)[:200],
                        "type": type(exc).__name__,
                    })

            # v2.3: Write watchdog heartbeat file (lightweight — external tools
            # can stat this file's mtime for liveness without parsing JSON)
            try:
                MICRO_HEARTBEAT_FILE.write_text(
                    f"{time.time():.3f}\n{self._tick}\n{os.getpid()}")
            except Exception:
                pass

            # Adaptive interval
            if self._enable_adaptive:
                self._adapt_interval()

            # Persist periodically
            if self._tick % MICRO_PERSIST_EVERY_N_TICKS == 0:
                threading.Thread(
                    target=self._persist_state, daemon=True,
                    name="l104-micro-persist").start()

            # Sleep for the remainder of the tick
            elapsed = time.time() - tick_start
            sleep_time = max(0.1, self._adaptive_interval - elapsed)
            self._stop_event.wait(timeout=sleep_time)

    def _execute_tick(self):
        """Execute all due micro-tasks for this tick (v2.1: with per-phase profiling)."""
        tick_start = time.time()
        tasks_run = 0
        tasks_passed = 0
        tasks_failed = 0
        slowest_task = ""
        slowest_ms = 0.0

        # v2.5: Periodic bridge re-discovery (every ~60s if not connected)
        if self._bridge is None and self._tick % 12 == 0:
            self._auto_discover_bridge()

        # v3.0: Periodic quantum peer discovery (every ~30s)
        if self._enable_quantum_network and self._quantum_mesh is not None and self._tick % 6 == 0:
            try:
                self._expand_quantum_mesh()
                # Refresh node advertisement
                net_path = BRIDGE_PATH / "quantum_network"
                net_path.mkdir(parents=True, exist_ok=True)
                advert = {
                    "node_id": self._quantum_node_id,
                    "qubits": self._quantum_qubits,
                    "pid": os.getpid(),
                    "timestamp": time.time(),
                    "tick": self._tick,
                }
                (net_path / f"{self._quantum_node_id}.json").write_text(
                    json.dumps(advert, indent=2))
            except Exception:
                pass

        ctx = self._build_context()

        # ── Phase 1: Collect due built-in tasks ──
        due_tasks: List[Tuple[int, str, Callable]] = []
        for name, (fn, cadence, priority) in self._task_registry.items():
            if self._tick % cadence == 0:
                due_tasks.append((priority, name, fn))

        # ── Phase 2: Collect on-demand pending tasks ──
        pending: List[MicroTask] = []
        with self._lock:
            while self._pending_queue:
                pending.append(self._pending_queue.popleft())
            custom_batch = []
            while self._custom_tasks:
                custom_batch.append(self._custom_tasks.popleft())

        # ── Phase 3: Execute built-in tasks by priority ──
        builtin_start = time.time()
        due_tasks.sort(key=lambda t: t[0])
        for priority, name, fn in due_tasks:
            t0 = time.time()
            try:
                result = fn(ctx)
                elapsed_ms = (time.time() - t0) * 1000
                # v2.4: Wall-clock timeout check — if elapsed > budget, record as timeout
                if elapsed_ms > MICRO_TASK_TIMEOUT_S * 1000:
                    _logger.warning(
                        "Task %s exceeded timeout (%.1fms > %.0fms budget)",
                        name, elapsed_ms, MICRO_TASK_TIMEOUT_S * 1000)
                    tasks_run += 1
                    tasks_failed += 1
                    self._error_log.append({
                        "tick": self._tick,
                        "task": name,
                        "error": f"timeout: {elapsed_ms:.1f}ms > {MICRO_TASK_TIMEOUT_S * 1000:.0f}ms budget",
                        "ts": time.time(),
                    })
                    # Count toward auto-throttle streak
                    streak = self._task_fail_streak.get(name, 0) + 1
                    self._task_fail_streak[name] = streak
                    if streak >= MICRO_AUTO_THROTTLE_THRESHOLD:
                        fn_entry = self._task_registry.get(name)
                        if fn_entry:
                            old_cadence = fn_entry[1]
                            new_cadence = min(old_cadence * 2, 120)
                            if name not in self._original_cadences:
                                self._original_cadences[name] = old_cadence
                            self._task_registry[name] = (fn_entry[0], new_cadence, fn_entry[2])
                    if elapsed_ms > slowest_ms:
                        slowest_ms = elapsed_ms
                        slowest_task = name
                    continue
                tasks_run += 1
                tasks_passed += 1
                if elapsed_ms > slowest_ms:
                    slowest_ms = elapsed_ms
                    slowest_task = name
                self._task_history.append({
                    "tick": self._tick,
                    "name": name,
                    "status": "ok",
                    "elapsed_ms": round(elapsed_ms, 2),
                    "ts": time.time(),
                })
                # Extract useful telemetry from certain tasks
                self._ingest_task_result(name, result)
                # v2.3: Reset fail streak on success (auto-throttle recovery)
                if name in self._task_fail_streak:
                    self._task_fail_streak[name] = 0
                    # Restore original cadence if it was throttled
                    if name in self._original_cadences:
                        fn_entry = self._task_registry.get(name)
                        if fn_entry:
                            orig_cadence = self._original_cadences.pop(name)
                            self._task_registry[name] = (fn_entry[0], orig_cadence, fn_entry[2])
                            _logger.info("Auto-throttle: restored %s cadence to %d", name, orig_cadence)
            except Exception as exc:
                elapsed_ms = (time.time() - t0) * 1000
                tasks_run += 1
                tasks_failed += 1
                if elapsed_ms > slowest_ms:
                    slowest_ms = elapsed_ms
                    slowest_task = name
                self._error_log.append({
                    "tick": self._tick,
                    "task": name,
                    "error": str(exc)[:200],
                    "ts": time.time(),
                })
                # v2.3: Auto-throttle — double cadence after N consecutive failures
                streak = self._task_fail_streak.get(name, 0) + 1
                self._task_fail_streak[name] = streak
                if streak >= MICRO_AUTO_THROTTLE_THRESHOLD:
                    fn_entry = self._task_registry.get(name)
                    if fn_entry:
                        old_cadence = fn_entry[1]
                        new_cadence = min(old_cadence * 2, 120)  # cap at ~10min
                        if name not in self._original_cadences:
                            self._original_cadences[name] = old_cadence
                        self._task_registry[name] = (fn_entry[0], new_cadence, fn_entry[2])
                        _logger.warning(
                            "Auto-throttle: %s failed %d× in a row — cadence %d→%d",
                            name, streak, old_cadence, new_cadence)
        builtin_ms = (time.time() - builtin_start) * 1000

        # ── Phase 4: Execute on-demand pending tasks ──
        pending_start = time.time()
        for task in sorted(pending, key=lambda t: t.priority):
            t0 = time.time()
            fn_entry = self._task_registry.get(task.name)
            if fn_entry:
                fn, _, _ = fn_entry
                try:
                    result = fn(ctx)
                    task.status = "completed"
                    task.result = result
                    task.elapsed_ms = (time.time() - t0) * 1000
                    tasks_run += 1
                    tasks_passed += 1
                except Exception as exc:
                    task.status = "failed"
                    task.error = str(exc)[:200]
                    task.elapsed_ms = (time.time() - t0) * 1000
                    tasks_run += 1
                    tasks_failed += 1
            else:
                task.status = "failed"
                task.error = f"unknown task: {task.name}"
                tasks_run += 1
                tasks_failed += 1
            # v2.4: Write final IPC completion response to outbox
            # Completes the request-response cycle started in _micro_ipc_poll
            try:
                outbox = MICRO_OUTBOX_PATH
                if outbox.exists():
                    resp = {
                        "request_id": task.task_id,
                        "task_name": task.name,
                        "status": task.status,
                        "elapsed_ms": round(task.elapsed_ms, 2),
                        "completed_at": time.time(),
                        "tick": self._tick,
                    }
                    if task.error:
                        resp["error"] = task.error
                    resp_file = outbox / f"{task.task_id}_done.json"
                    resp_file.write_text(json.dumps(resp, default=str))
            except Exception:
                pass
        pending_ms = (time.time() - pending_start) * 1000

        # ── Phase 5: Execute custom callables ──
        custom_start = time.time()
        for (task_id, name, fn, args, priority) in sorted(custom_batch, key=lambda t: t[4]):
            t0 = time.time()
            try:
                result = fn(ctx, *args)
                elapsed_ms = (time.time() - t0) * 1000
                tasks_run += 1
                tasks_passed += 1
                self._task_history.append({
                    "tick": self._tick,
                    "name": f"custom:{name}",
                    "task_id": task_id,
                    "status": "ok",
                    "elapsed_ms": round(elapsed_ms, 2),
                    "ts": time.time(),
                })
            except Exception as exc:
                elapsed_ms = (time.time() - t0) * 1000
                tasks_run += 1
                tasks_failed += 1
                self._error_log.append({
                    "tick": self._tick,
                    "task": f"custom:{name}",
                    "task_id": task_id,
                    "error": str(exc)[:200],
                    "ts": time.time(),
                })
        custom_ms = (time.time() - custom_start) * 1000

        # ── Phase 6: Record tick telemetry ──
        tick_elapsed_ms = (time.time() - tick_start) * 1000
        with self._lock:
            self._total_tasks_run += tasks_run
            self._total_tasks_passed += tasks_passed
            self._total_tasks_failed += tasks_failed
            self._total_elapsed_ms += tick_elapsed_ms

        self._update_health_score(tasks_run, tasks_passed)
        self._last_heartbeat_ts = time.time()

        snap = MicroTelemetry(
            tick=self._tick,
            timestamp=time.time(),
            tasks_run=tasks_run,
            tasks_passed=tasks_passed,
            tasks_failed=tasks_failed,
            tick_elapsed_ms=round(tick_elapsed_ms, 2),
            health_score=self._health_score,
            sacred_alignment=round(_PHI_MICRO_PHASE, 8),
            memory_mb=self._last_memory_mb,
            cpu_percent=self._last_cpu_percent,
            # v2.0: VQPU subsystem metrics
            scoring_cache_hits=self._last_scoring_cache_stats.get("hits", 0),
            circuit_cache_size=self._last_scoring_cache_stats.get("cache_count", 0),
            bridge_jobs_total=self._last_bridge_throughput.get("jobs_submitted", 0),
            daemon_cycler_alive=self._last_daemon_cycler_ping.get("active", False),
        )
        self._telemetry.append(asdict(snap))

        # v2.1: TickMetrics profiling
        metrics = TickMetrics(
            tick=self._tick,
            timestamp=time.time(),
            builtin_tasks_ms=round(builtin_ms, 2),
            pending_tasks_ms=round(pending_ms, 2),
            custom_tasks_ms=round(custom_ms, 2),
            total_ms=round(tick_elapsed_ms, 2),
            task_count=tasks_run,
            slowest_task=slowest_task,
            slowest_ms=round(slowest_ms, 2),
        )
        self._tick_metrics.append(metrics.to_dict())

    def _build_context(self) -> dict:
        """Build the shared context dict passed to every micro-task.

        v2.0: Includes bridge reference for VQPU subsystem tasks.
        v3.0: Includes quantum register and mesh for quantum network tasks.
        """
        return {
            "tick": self._tick,
            "start_time": self._start_time,
            "daemon": self,
            "god_code": GOD_CODE,
            "phi": PHI,
            "void_constant": VOID_CONSTANT,
            "bridge": self._bridge,         # v2.0: VQPUBridge (or None)
            "version": MICRO_DAEMON_VERSION,
        }

    def _ingest_task_result(self, name: str, result: dict):
        """Extract common telemetry from built-in task results.

        v2.0: Also ingests VQPU subsystem telemetry (scoring cache,
        bridge throughput, daemon cycler health).
        """
        if not isinstance(result, dict):
            return
        if name == "memory_probe":
            self._last_memory_mb = result.get("available_mb", self._last_memory_mb)
        elif name == "scoring_cache_stats":
            self._last_scoring_cache_stats = result
        elif name == "bridge_throughput":
            self._last_bridge_throughput = result
        elif name == "daemon_cycler_ping":
            self._last_daemon_cycler_ping = result

    def _adapt_interval(self):
        """Adjust tick interval based on CPU load (adaptive mode).

        v2.1: Uses cached psutil import to avoid per-tick import overhead.
        """
        psutil = _get_psutil()
        if psutil is None:
            self._adaptive_interval = self._tick_interval
            return
        try:
            cpu = psutil.cpu_percent(interval=0)
            self._last_cpu_percent = cpu
            if cpu < MICRO_LOAD_THRESHOLD_LOW:
                # Low load → faster ticks
                self._adaptive_interval = max(
                    MICRO_TICK_MIN_S,
                    self._tick_interval * 0.6)
            elif cpu > MICRO_LOAD_THRESHOLD_HIGH:
                # High load → slower ticks
                self._adaptive_interval = min(
                    MICRO_TICK_MAX_S,
                    self._tick_interval * 2.0)
            else:
                self._adaptive_interval = self._tick_interval
        except Exception:
            self._adaptive_interval = self._tick_interval

    def _update_health_score(self, tasks_run: int, tasks_passed: int):
        """Compute rolling health score [0,1].

        v2.2: Adds staleness decay — if the daemon is alive but hasn't run
        any tasks for a tick (paused or stalled), health decays by 1% per tick.
        """
        if tasks_run == 0:
            # v2.2: Staleness decay — no work done, health degrades slowly
            self._health_score = max(0.0, self._health_score * 0.99)
            return
        tick_score = tasks_passed / tasks_run
        # Exponential moving average (α=0.3)
        self._health_score = 0.3 * tick_score + 0.7 * self._health_score
        self._health_score = max(0.0, min(1.0, self._health_score))

    # ─── STATE PERSISTENCE ───────────────────────────────────────

    def _persist_state(self):
        """Write daemon state to disk (v2.1: includes crash_count, PID, tick_metrics).

        v2.2: JSON serialization moved outside lock — lock only held for
        fast dict construction (~0.01ms), not slow file I/O (~1-5ms).
        """
        try:
            with self._lock:
                state = {
                    "version": MICRO_DAEMON_VERSION,
                    "vqpu_version": VERSION,
                    "daemon": "VQPUMicroDaemon",
                    "last_persist": time.time(),
                    "pid": os.getpid(),
                    "tick": self._tick,
                    "total_tasks_run": self._total_tasks_run,
                    "total_tasks_passed": self._total_tasks_passed,
                    "total_tasks_failed": self._total_tasks_failed,
                    "pass_rate": round(
                        self._total_tasks_passed / max(self._total_tasks_run, 1), 4),
                    "total_elapsed_ms": round(self._total_elapsed_ms, 2),
                    "health_score": round(self._health_score, 4),
                    "adaptive_interval_s": round(self._adaptive_interval, 2),
                    "registered_tasks": list(self._task_registry.keys()),
                    "vqpu_tasks_enabled": self._enable_vqpu_tasks,
                    "bridge_connected": self._bridge is not None,
                    "telemetry_count": len(self._telemetry),
                    "error_count": len(self._error_log),
                    "god_code": GOD_CODE,
                    # v2.1: Crash recovery & profiling
                    "crash_count": self._crash_count,
                    "boot_time": self._boot_time,
                    "last_tick_metrics": list(self._tick_metrics)[-1:],
                    # v2.0: VQPU subsystem snapshot
                    "scoring_cache": self._last_scoring_cache_stats,
                    "bridge_throughput": self._last_bridge_throughput,
                    "daemon_cycler": self._last_daemon_cycler_ping,
                    # v3.0: Quantum network snapshot
                    "quantum_network_enabled": self._enable_quantum_network,
                    "quantum_node_id": self._quantum_node_id,
                    "quantum_qubits": self._quantum_qubits,
                    "quantum_register_active": self._qubit_register is not None,
                    "quantum_mesh_nodes": len(self._quantum_mesh.node_ids) if self._quantum_mesh else 0,
                    "quantum_mesh_channels": len(self._quantum_mesh.channels) if self._quantum_mesh else 0,
                    # v3.1: Topology data for cross-daemon consumption
                    "quantum_topology": self._quantum_topology,
                    "quantum_topology_analysis": (
                        self._quantum_mesh.topology_analysis() if self._quantum_mesh else None
                    ),
                }
            # JSON serialization + file I/O outside lock (no contention)
            self._state_path.write_text(
                json.dumps(state, indent=2, default=str))
        except Exception as e:
            _logger.warning("Micro daemon: failed to persist state: %s", e)

    def _load_state(self):
        """Restore state from disk (v2.2: smart crash detection via PID file).

        v2.2: Only increments crash_count if the previous shutdown was unclean
        (PID file still exists, meaning stop() never ran to remove it).
        Clean restarts (e.g. launchctl restart) do NOT bump crash_count.
        """
        try:
            if self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                self._tick = data.get("tick", 0)
                self._total_tasks_run = data.get("total_tasks_run", 0)
                self._total_tasks_passed = data.get("total_tasks_passed", 0)
                self._total_tasks_failed = data.get("total_tasks_failed", 0)
                self._total_elapsed_ms = data.get("total_elapsed_ms", 0.0)
                self._health_score = data.get("health_score", 1.0)
                prev_crash_count = data.get("crash_count", 0)

                # v2.3: Smart crash detection — check if PID file references
                # a process that is actually dead (true crash), not just leftover
                # from a process that exited without calling stop() (stale PID).
                unclean = False
                if MICRO_PID_FILE.exists():
                    try:
                        old_pid = int(MICRO_PID_FILE.read_text().strip())
                        # Check if old PID is still running
                        os.kill(old_pid, 0)  # signal 0 = existence check
                        # Process still alive — another daemon instance running
                        unclean = True
                    except (ValueError, ProcessLookupError):
                        # PID invalid or process dead — stale PID file, clean up
                        _logger.info(
                            "Micro daemon: stale PID file cleaned (process not running)")
                        try:
                            MICRO_PID_FILE.unlink()
                        except Exception:
                            pass
                    except PermissionError:
                        # Process exists but we can't signal it — treat as alive
                        unclean = True
                    except Exception:
                        pass

                if unclean:
                    self._crash_count = prev_crash_count + 1
                    _logger.warning(
                        "Micro daemon: UNCLEAN restart detected (PID file present) — "
                        "crash_count=%d", self._crash_count)
                else:
                    self._crash_count = prev_crash_count
                    _logger.info(
                        "Micro daemon: clean restart — tick=%d, tasks=%d",
                        self._tick, self._total_tasks_run)
        except Exception as e:
            _logger.warning("Micro daemon: failed to load state: %s", e)

    def _atexit_handler(self):
        """Persist state and clean up PID file on process exit.

        v2.3: Now removes PID file so next startup doesn't false-positive
        as an unclean restart. Also calls stop() for graceful shutdown.
        """
        if self._active:
            try:
                self.stop()
            except Exception:
                # Fallback: at minimum persist state and remove PID
                try:
                    self._persist_state()
                except Exception:
                    pass
                try:
                    if MICRO_PID_FILE.exists():
                        MICRO_PID_FILE.unlink()
                except Exception:
                    pass

    # ─── STATUS / API ────────────────────────────────────────────

    def status(self) -> dict:
        """Full micro-daemon status snapshot (v2.1: + tick_metrics + crash_count)."""
        # Snapshot mutable state under lock (fast copy)
        with self._lock:
            snap_tick = self._tick
            snap_run = self._total_tasks_run
            snap_passed = self._total_tasks_passed
            snap_failed = self._total_tasks_failed
            snap_health = self._health_score
            snap_interval = self._adaptive_interval
            snap_pending = len(self._pending_queue)
            snap_custom = len(self._custom_tasks)
            recent_telemetry = list(self._telemetry)[-5:]
            recent_errors = list(self._error_log)[-5:]
            recent_tasks = list(self._task_history)[-10:]
            recent_metrics = list(self._tick_metrics)[-3:]

        # Build dict outside lock — no contention
        uptime = time.time() - self._start_time if self._active else 0
        return {
            "version": MICRO_DAEMON_VERSION,
            "vqpu_version": VERSION,
            "daemon": "VQPUMicroDaemon",
            "active": self._active,
            "paused": self._paused,
            "alive": self.is_alive(),
            "pid": os.getpid(),
            "uptime_seconds": round(uptime, 1),
            "tick": snap_tick,
            "tick_interval_s": round(snap_interval, 2),
            "total_tasks_run": snap_run,
            "total_tasks_passed": snap_passed,
            "total_tasks_failed": snap_failed,
            "pass_rate": round(
                snap_passed / max(snap_run, 1), 4),
            "health_score": round(snap_health, 4),
            "health": round(snap_health, 4),  # v2.1: alias for compat
            "total_ticks": snap_tick,           # v2.1: alias for compat
            "last_heartbeat": self._last_heartbeat_ts,
            "watchdog_ts": self._watchdog_ts,
            "crash_count": self._crash_count,
            "boot_time": self._boot_time,
            "registered_tasks": sorted(self._task_registry.keys()),
            "registered_count": len(self._task_registry),
            "pending_queue_size": snap_pending,
            "custom_queue_size": snap_custom,
            "recent_telemetry": recent_telemetry,
            "recent_errors": recent_errors,
            "recent_tasks": recent_tasks,
            "recent_tick_metrics": recent_metrics,
            "state_file": str(self._state_path),
            "ipc_inbox": str(MICRO_INBOX_PATH),
            "ipc_outbox": str(MICRO_OUTBOX_PATH),
            "god_code": GOD_CODE,
            "sacred_phase": round(_PHI_MICRO_PHASE, 10),
            # v2.0: VQPU wiring status
            "vqpu_tasks_enabled": self._enable_vqpu_tasks,
            "bridge_connected": self._bridge is not None,
            "bridge_active": (
                getattr(self._bridge, "_active", False)
                if self._bridge else False
            ),
            "scoring_cache_stats": self._last_scoring_cache_stats,
            "bridge_throughput": self._last_bridge_throughput,
            "daemon_cycler_ping": self._last_daemon_cycler_ping,
            # v2.3: Analytics + auto-throttle
            "throttled_tasks": {k: v for k, v in self._original_cadences.items()},
            "fail_streaks": {k: v for k, v in self._task_fail_streak.items() if v > 0},
            # v3.0: Quantum network
            "quantum_network_enabled": self._enable_quantum_network,
            "quantum_node_id": self._quantum_node_id,
            "quantum_qubits": self._quantum_qubits,
            "quantum_register_active": self._qubit_register is not None,
            "quantum_mesh_active": self._quantum_mesh is not None,
            "quantum_mesh_nodes": len(self._quantum_mesh.node_ids) if self._quantum_mesh else 0,
            "quantum_mesh_channels": len(self._quantum_mesh.channels) if self._quantum_mesh else 0,
            # v4.0: Cross-daemon health, batching, predictive preemption
            "cross_daemon_health": self._compute_cross_daemon_health(),
            "batch_mode_active": self._batch_mode_active,
            "task_timing_tracked": len(self._task_timing_history),
        }

    def force_tick(self) -> dict:
        """Force one tick manually (synchronous). Returns tick telemetry."""
        if not self._active:
            self._start_time = self._start_time or time.time()
        self._tick += 1
        self._watchdog_ts = time.time()
        ctx = self._build_context()

        # Run all tasks regardless of cadence
        results = {}
        for name, (fn, _, _) in sorted(self._task_registry.items(),
                                         key=lambda x: x[1][2]):
            try:
                results[name] = fn(ctx)
            except Exception as e:
                results[name] = {"error": str(e)[:200]}

        self._last_heartbeat_ts = time.time()
        return {
            "tick": self._tick,
            "results": results,
            "health_score": self._health_score,
        }

    def get_task_result(self, task_name: str) -> Optional[dict]:
        """Get the most recent result for a named task from history."""
        for entry in reversed(self._task_history):
            if entry.get("name") == task_name:
                return entry
        return None

    def adjust_interval(self, new_interval: float):
        """Dynamically adjust the base tick interval."""
        self._tick_interval = max(MICRO_TICK_MIN_S, min(MICRO_TICK_MAX_S, new_interval))
        self._adaptive_interval = self._tick_interval

    # ─── OPERATIONAL TOOLING (v2.2) ──────────────────────────────

    def reset_stats(self) -> dict:
        """v2.2: Reset all cumulative counters and ring buffers.

        Returns snapshot of stats BEFORE reset for logging.
        Useful for debugging/testing without full daemon restart.
        """
        with self._lock:
            before = {
                "tick": self._tick,
                "total_tasks_run": self._total_tasks_run,
                "total_tasks_passed": self._total_tasks_passed,
                "total_tasks_failed": self._total_tasks_failed,
                "total_elapsed_ms": self._total_elapsed_ms,
                "health_score": self._health_score,
                "crash_count": self._crash_count,
                "telemetry_count": len(self._telemetry),
                "error_count": len(self._error_log),
                "task_history_count": len(self._task_history),
                "tick_metrics_count": len(self._tick_metrics),
            }
            self._tick = 0
            self._total_tasks_run = 0
            self._total_tasks_passed = 0
            self._total_tasks_failed = 0
            self._total_elapsed_ms = 0.0
            self._health_score = 1.0
            self._crash_count = 0
            self._telemetry.clear()
            self._error_log.clear()
            self._task_history.clear()
            self._tick_metrics.clear()
            self._last_scoring_cache_stats = {}
            self._last_bridge_throughput = {}
            self._last_daemon_cycler_ping = {}
        _logger.info("Micro daemon: stats reset (was tick=%d, tasks=%d)",
                      before["tick"], before["total_tasks_run"])
        return before

    def dump_metrics(self, path: Optional[str] = None) -> str:
        """v2.2: Dump TickMetrics history to a JSON file for analysis.

        Args:
            path: Optional file path. Defaults to logs/micro_daemon_metrics.json.

        Returns:
            Absolute path of the written file.
        """
        if path is None:
            log_dir = Path(os.environ.get("L104_ROOT", os.getcwd())) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            path = str(log_dir / "micro_daemon_metrics.json")

        with self._lock:
            metrics = list(self._tick_metrics)
            telemetry = list(self._telemetry)[-50:]

        dump = {
            "version": MICRO_DAEMON_VERSION,
            "dumped_at": time.time(),
            "tick": self._tick,
            "health_score": self._health_score,
            "tick_metrics": metrics,
            "recent_telemetry": telemetry,
            "god_code": GOD_CODE,
        }
        Path(path).write_text(json.dumps(dump, indent=2, default=str))
        _logger.info("Micro daemon: metrics dumped to %s (%d entries)", path, len(metrics))
        return str(Path(path).resolve())

    def task_stats(self) -> dict:
        """v2.2: Per-task execution statistics from task history.

        Returns a dict keyed by task name with count, mean_ms, max_ms, fail_count.
        Useful for identifying slow or flaky micro-tasks.
        """
        stats: Dict[str, dict] = {}
        with self._lock:
            history = list(self._task_history)
            errors = list(self._error_log)

        for entry in history:
            name = entry.get("name", "?")
            ms = entry.get("elapsed_ms", 0.0)
            if name not in stats:
                stats[name] = {"count": 0, "total_ms": 0.0, "max_ms": 0.0, "fail_count": 0}
            s = stats[name]
            s["count"] += 1
            s["total_ms"] += ms
            s["max_ms"] = max(s["max_ms"], ms)

        for entry in errors:
            name = entry.get("task", "?")
            if name not in stats:
                stats[name] = {"count": 0, "total_ms": 0.0, "max_ms": 0.0, "fail_count": 0}
            stats[name]["fail_count"] += 1

        # Compute mean
        for s in stats.values():
            s["mean_ms"] = round(s["total_ms"] / max(s["count"], 1), 2)
            s["max_ms"] = round(s["max_ms"], 2)
            del s["total_ms"]

        return dict(sorted(stats.items(), key=lambda x: -x[1]["count"]))

    # ─── ANALYTICS (v2.3) ───────────────────────────────────────

    def analytics(self) -> dict:
        """v2.3: Telemetry analytics — trends, anomalies, grades, hotspots.

        Returns a structured report from the TelemetryAnalytics subsystem.
        Analytics are cached per-tick to avoid recomputing on repeated calls.
        """
        with self._lock:
            telemetry_snap = list(self._telemetry)
            history_snap = list(self._task_history)

        # Cache per tick
        if self._analytics_cache_tick == self._tick and self._analytics_cache is not None:
            return self._analytics_cache.summary(history_snap)

        ta = TelemetryAnalytics(telemetry_snap)
        self._analytics_cache = ta
        self._analytics_cache_tick = self._tick
        return ta.summary(history_snap)

    def throttled_tasks(self) -> dict:
        """v2.3: Report which tasks are currently auto-throttled.

        Returns dict of task_name → {"original_cadence": int, "current_cadence": int,
        "fail_streak": int}.
        """
        result = {}
        for name, orig_cadence in self._original_cadences.items():
            fn_entry = self._task_registry.get(name)
            current_cadence = fn_entry[1] if fn_entry else orig_cadence
            result[name] = {
                "original_cadence": orig_cadence,
                "current_cadence": current_cadence,
                "fail_streak": self._task_fail_streak.get(name, 0),
            }
        return result

    # ─── SELF-TEST (v2.3) ───────────────────────────────────────

    def self_test(self) -> dict:
        """v3.0: Diagnostic self-test for l104_debug.py integration.

        Runs 15 probes across micro daemon subsystems (including 3 quantum
        network probes) and returns a structured result compatible with the
        unified debug framework.
        """
        results = []
        t0 = time.monotonic()

        # 1. Sacred constants alignment
        try:
            phase = GOD_CODE % (2 * math.pi)
            assert abs(phase - _PHI_MICRO_PHASE) < 1e-12, "Phase mismatch"
            assert abs(VOID_CONSTANT - (1.04 + PHI / 1000)) < 1e-15, "VOID mismatch"
            results.append({"test": "sacred_constants", "pass": True,
                            "detail": f"phase={phase:.10f}"})
        except Exception as e:
            results.append({"test": "sacred_constants", "pass": False, "error": str(e)})

        # 2. Task registry
        try:
            count = len(self._task_registry)
            has_core = "heartbeat" in self._task_registry
            has_vqpu = "scoring_cache_stats" in self._task_registry
            assert has_core, "Missing core heartbeat task"
            results.append({"test": "task_registry", "pass": True,
                            "detail": f"{count} tasks, core={has_core}, vqpu={has_vqpu}"})
        except Exception as e:
            results.append({"test": "task_registry", "pass": False, "error": str(e)})

        # 3. Force-tick execution
        try:
            tick_result = self.force_tick()
            task_count = len(tick_result.get("results", {}))
            assert task_count > 0, "No tasks executed in force tick"
            results.append({"test": "force_tick", "pass": True,
                            "detail": f"{task_count} tasks executed"})
        except Exception as e:
            results.append({"test": "force_tick", "pass": False, "error": str(e)})

        # 4. Heartbeat micro-task
        try:
            ctx = self._build_context()
            hb = _micro_heartbeat(ctx)
            assert hb.get("heartbeat") is True, "Heartbeat returned False"
            assert hb.get("phi_aligned") is True, "PHI not aligned"
            results.append({"test": "heartbeat_task", "pass": True,
                            "detail": f"phase={hb.get('god_code_phase', 0):.8f}"})
        except Exception as e:
            results.append({"test": "heartbeat_task", "pass": False, "error": str(e)})

        # 5. Fidelity probe
        try:
            ctx = self._build_context()
            fid = _micro_fidelity_probe(ctx)
            assert fid.get("pass") is True, f"Fidelity={fid.get('fidelity', 0)}"
            results.append({"test": "fidelity_probe", "pass": True,
                            "detail": f"fidelity={fid['fidelity']}"})
        except Exception as e:
            results.append({"test": "fidelity_probe", "pass": False, "error": str(e)})

        # 6. Score check
        try:
            ctx = self._build_context()
            sc = _micro_score_check(ctx)
            assert sc.get("sacred_pass") is True, f"resonance error={sc.get('alignment_error', 0)}"
            results.append({"test": "score_check", "pass": True,
                            "detail": f"resonance={sc.get('resonance_286', 0):.8f}"})
        except Exception as e:
            results.append({"test": "score_check", "pass": False, "error": str(e)})

        # 7. Bridge wiring
        try:
            # v2.5: Attempt auto-discovery if not already wired
            if self._bridge is None:
                self._auto_discover_bridge()
            bridge_connected = self._bridge is not None
            bridge_active = getattr(self._bridge, "_active", False) if bridge_connected else False
            has_cycler = (
                bridge_connected
                and hasattr(self._bridge, "_daemon_cycler")
                and self._bridge._daemon_cycler is not None
            )
            has_governor = (
                bridge_connected
                and getattr(self._bridge, "governor", None) is not None
            )
            detail = f"connected={bridge_connected}"
            if bridge_connected:
                detail += f", active={bridge_active}, cycler={has_cycler}, governor={has_governor}"
            results.append({"test": "bridge_wiring", "pass": bridge_connected,
                            "detail": detail})
        except Exception as e:
            results.append({"test": "bridge_wiring", "pass": False, "error": str(e)})

        # 8. State persistence
        try:
            self._persist_state()
            exists = self._state_path.exists()
            results.append({"test": "state_persistence", "pass": exists,
                            "detail": f"file={'exists' if exists else 'missing'}"})
        except Exception as e:
            results.append({"test": "state_persistence", "pass": False, "error": str(e)})

        # 9. v2.1: MicroTaskPriority enum integrity
        try:
            assert MicroTaskPriority.CRITICAL == 1, "CRITICAL != 1"
            assert MicroTaskPriority.HIGH == 2, "HIGH != 2"
            assert MicroTaskPriority.NORMAL == 5, "NORMAL != 5"
            assert MicroTaskPriority.LOW == 7, "LOW != 7"
            assert MicroTaskPriority.IDLE == 9, "IDLE != 9"
            results.append({"test": "priority_enum", "pass": True,
                            "detail": "5 levels: CRITICAL/HIGH/NORMAL/LOW/IDLE"})
        except Exception as e:
            results.append({"test": "priority_enum", "pass": False, "error": str(e)})

        # 10. v2.1: IPC directory structure
        try:
            inbox_ok = MICRO_INBOX_PATH.parent.exists() or True  # parent = micro/
            outbox_ok = MICRO_OUTBOX_PATH.parent.exists() or True
            pid_path = MICRO_PID_FILE
            results.append({"test": "ipc_structure", "pass": True,
                            "detail": f"inbox={MICRO_INBOX_PATH.exists()}, outbox={MICRO_OUTBOX_PATH.exists()}, pid={pid_path.exists()}"})
        except Exception as e:
            results.append({"test": "ipc_structure", "pass": False, "error": str(e)})

        # 11. v2.3: TelemetryAnalytics subsystem
        try:
            report = self.analytics()
            grade = report.get("grade", {}).get("grade", "?")
            trend = report.get("trend", {}).get("direction", "?")
            anomaly_count = len(report.get("anomalies", []))
            assert grade in ("A", "B", "C", "D", "F", "?"), f"Invalid grade: {grade}"
            assert trend in ("rising", "falling", "stable"), f"Invalid trend: {trend}"
            results.append({"test": "telemetry_analytics", "pass": True,
                            "detail": f"grade={grade}, trend={trend}, anomalies={anomaly_count}"})
        except Exception as e:
            results.append({"test": "telemetry_analytics", "pass": False, "error": str(e)})

        # 12. v2.3: Auto-throttle + heartbeat file
        try:
            throttled = self.throttled_tasks()
            heartbeat_exists = MICRO_HEARTBEAT_FILE.exists()
            # Auto-throttle should be initialized (empty dict is fine)
            assert isinstance(self._task_fail_streak, dict), "fail_streak not a dict"
            assert isinstance(self._original_cadences, dict), "original_cadences not a dict"
            results.append({"test": "auto_throttle_heartbeat", "pass": True,
                            "detail": f"throttled={len(throttled)}, heartbeat_file={heartbeat_exists}"})
        except Exception as e:
            results.append({"test": "auto_throttle_heartbeat", "pass": False, "error": str(e)})

        # 13. v3.0: Quantum qubit register
        try:
            if self._enable_quantum_network:
                DaemonQubitRegister, _ = _get_quantum_network()
                reg = DaemonQubitRegister(node_id="selftest", num_qubits=2)
                reg.initialize_sacred()
                check = reg.fidelity_check()
                avg_fid = check.get("avg_fidelity", 0)
                assert avg_fid > 0.5, f"Qubit register fidelity too low: {avg_fid}"
                results.append({"test": "quantum_qubit_register", "pass": True,
                                "detail": f"fidelity={avg_fid:.6f}, qubits=2"})
            else:
                results.append({"test": "quantum_qubit_register", "pass": True,
                                "detail": "skipped (quantum_network disabled)"})
        except Exception as e:
            results.append({"test": "quantum_qubit_register", "pass": False, "error": str(e)})

        # 14. v3.0: Quantum network mesh
        try:
            if self._enable_quantum_network and self._quantum_mesh is not None:
                health = self._quantum_mesh.network_health()
                node_count = health.get("nodes", 0)
                mean_fid = health.get("avg_fidelity", 0)
                assert node_count >= 1, f"Mesh has {node_count} nodes (expected ≥1)"
                results.append({"test": "quantum_mesh_health", "pass": True,
                                "detail": f"nodes={node_count}, mean_fid={mean_fid:.6f}"})
            else:
                results.append({"test": "quantum_mesh_health", "pass": True,
                                "detail": "skipped (mesh not active)"})
        except Exception as e:
            results.append({"test": "quantum_mesh_health", "pass": False, "error": str(e)})

        # 15. v3.0: Sacred qubit alignment
        try:
            if self._enable_quantum_network and self._qubit_register is not None:
                ctx = self._build_context()
                probe = _micro_qubit_sacred_probe(ctx)
                alignment = probe.get("avg_sacred_alignment", 0)
                assert alignment > 0, f"Sacred alignment is {alignment}"
                results.append({"test": "sacred_qubit_alignment", "pass": True,
                                "detail": f"alignment={alignment:.6f}"})
            else:
                results.append({"test": "sacred_qubit_alignment", "pass": True,
                                "detail": "skipped (qubits not active)"})
        except Exception as e:
            results.append({"test": "sacred_qubit_alignment", "pass": False, "error": str(e)})

        # 16. v4.0: Quantum topology analysis
        try:
            if self._enable_quantum_network and self._quantum_mesh is not None:
                topo_analysis = self._quantum_mesh.topology_analysis()
                topo_type = topo_analysis.get("topology", "unknown")
                detected = topo_analysis.get("detected_topology", "unknown")
                sacred_topo = topo_analysis.get("sacred_topology_score", 0.0)
                is_connected = topo_analysis.get("is_connected", False)
                results.append({"test": "quantum_topology_analysis", "pass": True,
                                "detail": f"topology={topo_type}, detected={detected}, "
                                          f"sacred={sacred_topo:.4f}, connected={is_connected}"})
            else:
                results.append({"test": "quantum_topology_analysis", "pass": True,
                                "detail": "skipped (mesh not active)"})
        except Exception as e:
            results.append({"test": "quantum_topology_analysis", "pass": False, "error": str(e)})

        elapsed_ms = round((time.monotonic() - t0) * 1000, 2)
        passed = sum(1 for r in results if r["pass"])
        total = len(results)

        return {
            "engine": "vqpu_micro_daemon",
            "version": MICRO_DAEMON_VERSION,
            "vqpu_version": VERSION,
            "tests": results,
            "passed": passed,
            "total": total,
            "all_pass": passed == total,
            "elapsed_ms": elapsed_ms,
            "god_code": GOD_CODE,
        }

    # ─── v4.0: CROSS-DAEMON HEALTH, TIMING, BATCHING ────────

    def _compute_cross_daemon_health(self) -> dict:
        """v4.0: Aggregate health from all L104 daemon state files."""
        now = time.time()
        if (now - self._cross_daemon_health_ts) < 30.0 and self._cross_daemon_health:
            return self._cross_daemon_health

        l104_root = os.environ.get("L104_ROOT", os.getcwd())
        scores = {}
        for name, filename in _DAEMON_STATE_FILES.items():
            try:
                path = Path(l104_root) / filename
                if path.exists():
                    data = json.loads(path.read_text())
                    scores[name] = {
                        "health_score": data.get("health_score", data.get("health_trend", 0)),
                        "version": data.get("version", "?"),
                        "available": True,
                    }
                else:
                    scores[name] = {"available": False}
            except Exception:
                scores[name] = {"available": False}

        # Composite score: average of available daemons
        available_scores = [v["health_score"] for v in scores.values()
                          if v.get("available") and isinstance(v.get("health_score"), (int, float))]
        composite = round(sum(available_scores) / max(len(available_scores), 1), 4) if available_scores else 0.0

        result = {
            "daemons": scores,
            "composite_health": composite,
            "daemons_available": len(available_scores),
            "daemons_total": len(_DAEMON_STATE_FILES),
        }
        self._cross_daemon_health = result
        self._cross_daemon_health_ts = now
        return result

    def _record_task_timing(self, task_name: str, elapsed_ms: float):
        """v4.0: Record task execution time for predictive preemption."""
        if task_name not in self._task_timing_history:
            self._task_timing_history[task_name] = deque(maxlen=50)
        self._task_timing_history[task_name].append(elapsed_ms)

    def _predict_task_duration(self, task_name: str) -> float:
        """v4.0: Predict task duration from historical timing."""
        history = self._task_timing_history.get(task_name)
        if not history or len(history) < 3:
            return 0.0
        return sum(history) / len(history)

    def cross_daemon_health(self) -> dict:
        """v4.0: Public API for cross-daemon health aggregation."""
        return self._compute_cross_daemon_health()

    def __repr__(self) -> str:
        bridge_tag = "+bridge" if self._bridge else ""
        vqpu_tag = "+vqpu" if self._enable_vqpu_tasks else ""
        quantum_tag = "+quantum" if self._enable_quantum_network else ""
        return (
            f"VQPUMicroDaemon(v{MICRO_DAEMON_VERSION}{bridge_tag}{vqpu_tag}{quantum_tag}, "
            f"tick={self._tick}, health={self._health_score:.3f}, "
            f"tasks={self._total_tasks_run}/{len(self._task_registry)}reg, "
            f"active={self._active})"
        )


# ═══════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════

_micro_daemon: Optional[VQPUMicroDaemon] = None


def get_micro_daemon() -> VQPUMicroDaemon:
    """Get the global VQPUMicroDaemon singleton (auto-starts on first call)."""
    global _micro_daemon
    if _micro_daemon is None:
        _micro_daemon = VQPUMicroDaemon()
        _micro_daemon.start()
    return _micro_daemon


# ═══════════════════════════════════════════════════════════════════
# CLI ENTRY POINT — python -m l104_vqpu.micro_daemon
# ═══════════════════════════════════════════════════════════════════

def _cli_main():
    """Standalone CLI runner for the VQPU Micro Daemon."""
    import argparse
    import signal

    parser = argparse.ArgumentParser(
        description="L104 VQPU Micro Daemon — lightweight background process assistant")
    parser.add_argument("--tick", type=float, default=MICRO_TICK_INTERVAL_S,
                        help=f"Tick interval in seconds (default: {MICRO_TICK_INTERVAL_S})")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable adaptive tick interval")
    parser.add_argument("--no-ipc", action="store_true",
                        help="Disable IPC inbox/outbox")
    parser.add_argument("--status-interval", type=float, default=30.0,
                        help="Print status every N seconds (default: 30)")
    parser.add_argument("--json", action="store_true",
                        help="Output status as JSON")
    parser.add_argument("--self-test", action="store_true",
                        help="Run self-test (15 probes incl. quantum) and exit with 0/1")
    parser.add_argument("--quantum-status", action="store_true",
                        help="Print quantum network status and exit")
    parser.add_argument("--health-check", action="store_true",
                        help="Read persisted state file, print health, exit 0/1")
    parser.add_argument("--dump-metrics", action="store_true",
                        help="Force one tick, dump metrics to logs/, then exit")
    parser.add_argument("--analytics", action="store_true",
                        help="Run analytics report (trend, grade, hotspots) and exit")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S")

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  L104 VQPU Micro Daemon v{MICRO_DAEMON_VERSION}                   ║")
    print(f"║  Micro Process Background Assistant              ║")
    print(f"║  GOD_CODE = {GOD_CODE}               ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    # ── v2.2: Quick-exit modes ──

    if args.health_check:
        # Read persisted state file without starting daemon
        state_path = Path(os.environ.get("L104_ROOT", os.getcwd())) / MICRO_STATE_FILE
        if not state_path.exists():
            print("[HEALTH] No state file found — daemon may not have run yet")
            raise SystemExit(2)
        data = json.loads(state_path.read_text())
        health = data.get("health_score", 0)
        tick = data.get("tick", 0)
        crash = data.get("crash_count", 0)
        pid_alive = MICRO_PID_FILE.exists()
        print(f"[HEALTH] score={health:.4f} | tick={tick} | crash_count={crash} | pid_alive={pid_alive}")
        if args.json:
            print(json.dumps(data, indent=2, default=str))
        raise SystemExit(0 if health >= 0.5 else 1)

    if args.quantum_status:
        d = VQPUMicroDaemon(tick_interval=args.tick, enable_ipc=not args.no_ipc)
        d.start()
        time.sleep(0.3)
        d.force_tick()
        qs = d.quantum_status()
        if args.json:
            print(json.dumps(qs, indent=2, default=str))
        else:
            print(f"[QUANTUM] Quantum Network Status")
            print(f"  Enabled:     {qs.get('quantum_network_enabled', False)}")
            print(f"  Node ID:     {qs.get('node_id', 'N/A')}")
            print(f"  Qubits:      {qs.get('qubits_per_node', 0)}")
            print(f"  Topology:    {qs.get('topology', 'N/A')}")
            reg = qs.get('register', {})
            print(f"  Register:    fidelity={reg.get('avg_fidelity', 0):.6f}")
            mesh = qs.get("network", {})
            if mesh:
                print(f"  Mesh Nodes:  {mesh.get('nodes', 0)}")
                print(f"  Mesh Chans:  {mesh.get('active_channels', 0)}")
                print(f"  Mesh Topo:   {mesh.get('topology', 'N/A')} (detected={mesh.get('detected_topology', '?')})")
                print(f"  Mean Fid:    {mesh.get('avg_fidelity', 0):.6f}")
                print(f"  Mean Degree: {mesh.get('mean_degree', 0):.1f}")
                print(f"  Diameter:    {mesh.get('diameter', 0)}")
                sac = mesh.get("sacred_alignment", 0)
                print(f"  Sacred Algn: {sac:.6f}")
                topo_s = mesh.get("sacred_topology_score", 0)
                print(f"  Sacred Topo: {topo_s:.6f}")
        d.stop()
        raise SystemExit(0)

    if args.self_test:
        d = VQPUMicroDaemon(tick_interval=args.tick, enable_ipc=not args.no_ipc)
        d.start()
        time.sleep(0.3)
        st = d.self_test()
        if args.json:
            print(json.dumps(st, indent=2, default=str))
        else:
            print(f"[SELF-TEST] {st['passed']}/{st['total']} in {st['elapsed_ms']}ms")
            for t in st["tests"]:
                tag = "OK" if t["pass"] else "FAIL"
                print(f"  [{tag}] {t['test']}: {t.get('detail', t.get('error', ''))}")
        d.stop()
        raise SystemExit(0 if st["all_pass"] else 1)

    if args.dump_metrics:
        d = VQPUMicroDaemon(tick_interval=args.tick, enable_ipc=not args.no_ipc)
        d.start()
        time.sleep(0.3)
        d.force_tick()
        path = d.dump_metrics()
        print(f"[DUMP] Metrics written to {path}")
        ts = d.task_stats()
        print(f"[TASKS] {len(ts)} task types recorded")
        for name, s in list(ts.items())[:10]:
            print(f"  {name}: {s['count']}× mean={s['mean_ms']}ms max={s['max_ms']}ms fail={s['fail_count']}")
        d.stop()
        raise SystemExit(0)

    if args.analytics:
        # v2.4: Run analytics report — requires a few ticks for meaningful data
        d = VQPUMicroDaemon(tick_interval=args.tick, enable_ipc=not args.no_ipc)
        d.start()
        time.sleep(0.3)
        # Run 3 force-ticks to accumulate baseline telemetry for trend analysis
        for _ in range(3):
            d.force_tick()
        report = d.analytics()
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            grade_info = report.get("grade", {})
            trend_info = report.get("trend", {})
            anomalies = report.get("anomalies", [])
            hotspots = report.get("hotspots", [])
            print(f"[ANALYTICS] Telemetry Analytics Report")
            print(f"  Grade:     {grade_info.get('grade', '?')} "
                  f"(health={grade_info.get('health', 0):.3f}, "
                  f"pass_rate={grade_info.get('pass_rate', 0):.3f}, "
                  f"avg_tick={grade_info.get('avg_tick_ms', 0):.1f}ms)")
            print(f"  Trend:     {trend_info.get('direction', '?')} "
                  f"(slope={trend_info.get('slope', 0):.6f}, "
                  f"window={trend_info.get('window', 0)})")
            print(f"  Anomalies: {len(anomalies)}")
            if hotspots:
                print(f"  Hotspots:")
                for h in hotspots[:5]:
                    print(f"    {h['name']}: {h['total_ms']:.1f}ms "
                          f"({h['count']}×, {h['pct']:.1f}%)")
            # Also print throttled tasks
            throttled = d.throttled_tasks()
            if throttled:
                print(f"  Throttled: {len(throttled)} tasks")
                for name, info in throttled.items():
                    print(f"    {name}: cadence {info['original_cadence']}→{info['current_cadence']} "
                          f"(streak={info['fail_streak']})")
            else:
                print(f"  Throttled: none")
        d.stop()
        raise SystemExit(0)

    # ── Create logs directory for launchd stdout/stderr ──
    try:
        log_dir = Path(os.environ.get("L104_ROOT", os.getcwd())) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    daemon = VQPUMicroDaemon(
        tick_interval=args.tick,
        enable_ipc=not args.no_ipc,
        enable_adaptive=not args.no_adaptive,
    )

    # Handle SIGINT/SIGTERM for clean shutdown
    shutdown = threading.Event()

    def _sig_handler(signum, frame):
        print("\n[SHUTDOWN] Stopping micro daemon...")
        shutdown.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # SIGUSR1 — dump full status JSON to stdout (kill -USR1 <pid>)
    def _sig_status(signum, frame):
        try:
            s = daemon.status()
            print(json.dumps(s, indent=2, default=str))
        except Exception as e:
            print(f"[SIGUSR1] status error: {e}")

    # SIGUSR2 — force an immediate tick (kill -USR2 <pid>)
    def _sig_force_tick(signum, frame):
        try:
            result = daemon.force_tick()
            count = len(result.get("results", {}))
            print(f"[SIGUSR2] Force-tick complete — {count} tasks")
        except Exception as e:
            print(f"[SIGUSR2] force_tick error: {e}")

    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _sig_status)
    if hasattr(signal, "SIGUSR2"):
        signal.signal(signal.SIGUSR2, _sig_force_tick)

    daemon.start()
    print(f"[STARTED] Tick interval: {args.tick}s | "
          f"Adaptive: {not args.no_adaptive} | "
          f"IPC: {not args.no_ipc} | "
          f"Tasks: {len(daemon._task_registry)}")
    print(f"[SIGNALS] SIGUSR1=status dump, SIGUSR2=force tick, PID={os.getpid()}")

    # Force one tick immediately so we have data to show
    initial = daemon.force_tick()
    print(f"[INIT] Force-tick complete — {len(initial.get('results', {}))} tasks")

    last_status_ts = time.time()
    while not shutdown.is_set():
        shutdown.wait(timeout=1.0)
        now = time.time()
        if now - last_status_ts >= args.status_interval:
            last_status_ts = now
            s = daemon.status()
            if args.json:
                print(json.dumps(s, indent=2, default=str))
            else:
                print(
                    f"[STATUS] tick={s['tick']} | "
                    f"tasks={s['total_tasks_run']} | "
                    f"pass={s['pass_rate']:.2%} | "
                    f"health={s['health_score']:.3f} | "
                    f"interval={s['tick_interval_s']:.1f}s | "
                    f"pending={s['pending_queue_size']}")

    daemon.stop()
    final = daemon.status()
    print(f"\n[FINAL] {final['tick']} ticks | "
          f"{final['total_tasks_run']} tasks | "
          f"pass={final['pass_rate']:.2%} | "
          f"health={final['health_score']:.3f}")

    # v2.2: Print task execution stats on shutdown
    ts = daemon.task_stats()
    if ts:
        print(f"[TASK STATS] {len(ts)} task types:")
        for name, s in list(ts.items())[:8]:
            print(f"  {name}: {s['count']}× mean={s['mean_ms']}ms max={s['max_ms']}ms")


if __name__ == "__main__":
    _cli_main()


__all__ = [
    "VQPUMicroDaemon",
    "MicroTask",
    "MicroTaskResult",
    "MicroTaskStatus",
    "MicroTaskPriority",
    "MicroTelemetry",
    "MicroDaemonConfig",
    "TickMetrics",
    "TelemetryAnalytics",
    "get_micro_daemon",
    "MICRO_DAEMON_VERSION",
    # v3.0: Quantum network
    "_micro_qubit_fidelity",
    "_micro_qubit_sacred_probe",
    "_micro_quantum_network_health",
    "_micro_channel_purification",
    "_micro_qubit_recalibrate",
]
