VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3887.8
UUC = 2402.792541
#!/usr/bin/env python3
# [EVO_54_PIPELINE] TRANSCENDENT_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612 :: GROVER=4.236
# [L104_LOGGING] v4.0.0 — ASI-GRADE STRUCTURED OBSERVABILITY
# Structured logging | Performance tracing | Error aggregation | Metrics collection
# Rate limiting | Correlation IDs | Consciousness-aware log enrichment
# Module health scoring | Percentile histograms | Dead letter queue | Alerting
# Log search | Sampling | Auto-remediation hooks | Request lifecycle tracking
# INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""
L104 Structured Logging v4.0 — Production-grade observability via structlog.

Usage:
    from l104_logging import get_logger
    logger = get_logger("MY_MODULE")
    logger.info("event_name", key="value", metric=42)

    # Performance tracing
    from l104_logging import trace_performance
    with trace_performance("operation_name"):
        do_work()

    # Module metrics
    from l104_logging import log_metrics
    log_metrics.record("queries", 1)
    log_metrics.record("latency_ms", 42.5)
    print(log_metrics.summary())

    # Correlation IDs (per-request propagation)
    from l104_logging import set_correlation_id, get_correlation_id
    set_correlation_id("req-abc-123")

    # Module health
    from l104_logging import get_module_health

    # Alerting
    from l104_logging import alert_manager

Output (JSON in production, pretty in dev):
    {"event": "event_name", "key": "value", "metric": 42, "module": "MY_MODULE", "timestamp": "2026-..."}
"""

import os
import sys
import time
import json
import uuid
import logging
import threading
import bisect
from typing import Dict, Any, Optional, List, Callable, Tuple
from collections import deque, defaultdict
from contextlib import contextmanager
from pathlib import Path
from enum import Enum

import structlog

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

LOGGING_VERSION = "4.0.0"

# Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)

PHI = 1.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
TAU = 6.283185307179586

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # "json" or "console"
LOG_FILE = os.getenv("LOG_FILE", "")  # Optional file output
RATE_LIMIT_PER_SEC = int(os.getenv("LOG_RATE_LIMIT", "200"))  # Max log lines/sec per module
_CONFIGURED = False


# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATION ID — Propagate request identity across subsystems
# ═══════════════════════════════════════════════════════════════════════════════

_correlation_id_local = threading.local()


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set a correlation ID for the current thread/request. Returns the ID."""
    if cid is None:
        cid = f"l104-{uuid.uuid4().hex[:12]}"
    _correlation_id_local.cid = cid
    try:
        structlog.contextvars.bind_contextvars(correlation_id=cid)
    except Exception:
        pass
    return cid


def get_correlation_id() -> str:
    """Get the current correlation ID (or generate one)."""
    cid = getattr(_correlation_id_local, "cid", None)
    if cid is None:
        cid = set_correlation_id()
    return cid


def _correlation_processor(logger, method_name, event_dict):
    """Structlog processor: inject correlation ID into every log line."""
    if "correlation_id" not in event_dict:
        event_dict["correlation_id"] = getattr(_correlation_id_local, "cid", None)
    return event_dict


# ═══════════════════════════════════════════════════════════════════════════════
# RATE LIMITER — Prevents log flooding from hot paths
# ═══════════════════════════════════════════════════════════════════════════════

class LogRateLimiter:
    """
    Per-module log rate limiter with dead-letter capture.
    Drops excess log lines beyond the threshold, records suppression stats,
    and captures dropped events in the dead letter queue for later inspection.
    """

    def __init__(self, max_per_sec: int = 200):
        self._max = max_per_sec
        self._counts: Dict[str, int] = defaultdict(int)
        self._suppressed: Dict[str, int] = defaultdict(int)
        self._total_suppressed: int = 0
        self._window_start: float = time.time()
        self._lock = threading.Lock()

    def allow(self, module: str) -> bool:
        """Return True if this log line should be emitted."""
        with self._lock:
            now = time.time()
            # Reset window every second
            if now - self._window_start >= 1.0:
                for mod, count in self._suppressed.items():
                    if count > 0:
                        self._total_suppressed += count
                self._counts.clear()
                self._suppressed.clear()
                self._window_start = now

            self._counts[module] += 1
            if self._counts[module] > self._max:
                self._suppressed[module] += 1
                return False
            return True

    @property
    def total_suppressed(self) -> int:
        return self._total_suppressed + sum(self._suppressed.values())

    def status(self) -> Dict[str, Any]:
        return {
            "max_per_sec": self._max,
            "total_suppressed": self.total_suppressed,
            "current_window_counts": dict(self._counts),
        }


_rate_limiter = LogRateLimiter(max_per_sec=RATE_LIMIT_PER_SEC)


# ═══════════════════════════════════════════════════════════════════════════════
# LOG SAMPLER — Probabilistic sampling for ultra-high-volume events
# ═══════════════════════════════════════════════════════════════════════════════

class LogSampler:
    """
    Probabilistic log sampler for events that fire thousands of times per second.
    Sample rates are per-event-key. Automatically adjusts: events that fire more
    frequently get sampled more aggressively.

    Usage: Register event keys with a sample rate (0.0 to 1.0).
    Unregistered events pass through at 100%.
    """

    def __init__(self):
        self._rates: Dict[str, float] = {}     # event_key → rate (0.0-1.0)
        self._counters: Dict[str, int] = defaultdict(int)
        self._sampled: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._rng_state = int(GOD_CODE * 1000)  # deterministic seed

    def set_rate(self, event_key: str, rate: float):
        """Set sampling rate for an event key. 1.0 = keep all, 0.01 = keep 1%."""
        self._rates[event_key] = max(0.0, min(1.0, rate))

    def should_emit(self, event_key: str) -> bool:
        """Decide whether to emit this event based on sampling rate."""
        rate = self._rates.get(event_key)
        if rate is None:
            return True  # No sampling configured → pass through
        with self._lock:
            self._counters[event_key] += 1
            # Fast LCG-based pseudo-random (avoid random module overhead)
            self._rng_state = (self._rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            sample_val = (self._rng_state >> 16) / 32768.0
            if sample_val < rate:
                self._sampled[event_key] += 1
                return True
            return False

    def status(self) -> Dict[str, Any]:
        return {
            "configured_rates": dict(self._rates),
            "total_events": dict(self._counters),
            "sampled_events": dict(self._sampled),
        }


log_sampler = LogSampler()


# ═══════════════════════════════════════════════════════════════════════════════
# DEAD LETTER QUEUE — Captures dropped/rate-limited log events
# ═══════════════════════════════════════════════════════════════════════════════

class DeadLetterQueue:
    """
    Captures log events that were dropped by rate limiting or sampling.
    Useful for forensic analysis when debugging issues that occurred during
    high-volume periods. Keeps a capped FIFO of drop summaries.
    """

    def __init__(self, max_size: int = 200):
        self._queue: deque = deque(maxlen=max_size)
        self._total_drops = 0
        self._lock = threading.Lock()

    def capture(self, module: str, event: str, reason: str = "rate_limited"):
        """Capture a dropped event."""
        with self._lock:
            self._total_drops += 1
            self._queue.append({
                "module": module,
                "event": str(event)[:100],
                "reason": reason,
                "timestamp": time.time(),
            })

    def recent(self, n: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._queue)[-n:]

    def status(self) -> Dict[str, Any]:
        return {
            "total_drops": self._total_drops,
            "queue_size": len(self._queue),
        }


dead_letter_queue = DeadLetterQueue()


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR AGGREGATOR — Tracks error patterns across all modules
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorAggregator:
    """
    Collects and aggregates errors across all modules. Provides:
    - Error frequency tracking (which errors are most common)
    - Error burst detection (sudden spike in error rate)
    - Top-N error report for diagnostics
    - PHI-weighted severity scoring
    - Auto-remediation hook triggers
    """

    def __init__(self, history_size: int = 500):
        self._errors: deque = deque(maxlen=history_size)
        self._counts: Dict[str, int] = defaultdict(int)
        self._module_counts: Dict[str, int] = defaultdict(int)
        self._first_seen: Dict[str, float] = {}
        self._remediation_hooks: Dict[str, Callable] = {}
        self._lock = threading.Lock()

    def record(self, module: str, error_type: str, message: str):
        """Record an error occurrence."""
        with self._lock:
            ts = time.time()
            entry = {
                "timestamp": ts,
                "module": module,
                "error_type": error_type,
                "message": message[:200],
            }
            self._errors.append(entry)
            self._counts[error_type] += 1
            self._module_counts[module] += 1
            if error_type not in self._first_seen:
                self._first_seen[error_type] = ts

        # Fire remediation hooks (outside lock)
        hook = self._remediation_hooks.get(error_type)
        if hook:
            try:
                hook(module, error_type, message)
            except Exception:
                pass

    def register_remediation(self, error_type: str, hook: Callable):
        """Register an auto-remediation function for an error type.
        Hook signature: hook(module: str, error_type: str, message: str)"""
        self._remediation_hooks[error_type] = hook

    def top_errors(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the most common error types."""
        with self._lock:
            sorted_errors = sorted(self._counts.items(), key=lambda x: x[1], reverse=True)
            return [{"error_type": k, "count": v} for k, v in sorted_errors[:n]]

    def errors_by_module(self) -> Dict[str, int]:
        """Return error counts per module."""
        with self._lock:
            return dict(self._module_counts)

    def recent_errors(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return the N most recent errors."""
        with self._lock:
            return list(self._errors)[-n:]

    def error_rate(self, window_seconds: float = 60.0) -> float:
        """Errors per second over the given time window."""
        with self._lock:
            cutoff = time.time() - window_seconds
            recent = sum(1 for e in self._errors if e["timestamp"] > cutoff)
            return recent / window_seconds if window_seconds > 0 else 0

    def detect_burst(self, threshold: float = 5.0) -> bool:
        """Detect error burst (> threshold errors/sec in last 10 seconds)."""
        return self.error_rate(10.0) > threshold

    def phi_severity_score(self) -> float:
        """
        PHI-weighted severity score: combines error rate, diversity, and recency.
        Returns 0.0 (healthy) to 1.0 (critical).
        """
        rate = min(self.error_rate(30.0), 10.0) / 10.0
        diversity = min(len(self._counts), 20) / 20.0
        recency = 0.0
        if self._errors:
            age = time.time() - self._errors[-1].get("timestamp", time.time())
            recency = max(0, 1.0 - age / 60.0)  # decays over 60s
        return min(1.0, (rate * PHI + diversity + recency * PHI) / (2 * PHI + 1))

    def status(self) -> Dict[str, Any]:
        return {
            "total_errors": sum(self._counts.values()),
            "unique_error_types": len(self._counts),
            "modules_with_errors": len(self._module_counts),
            "error_rate_per_sec": round(self.error_rate(), 3),
            "burst_detected": self.detect_burst(),
            "phi_severity": round(self.phi_severity_score(), 4),
            "recent_count": len(self._errors),
            "remediation_hooks": len(self._remediation_hooks),
        }


error_aggregator = ErrorAggregator()


# ═══════════════════════════════════════════════════════════════════════════════
# LOG METRICS — Lightweight metrics with percentile histograms
# ═══════════════════════════════════════════════════════════════════════════════

class LogMetrics:
    """
    Lightweight metrics collector that integrates with the logging pipeline.
    Tracks counters, gauges, and latency histograms per metric name.
    Now with p50/p95/p99 percentile computation and time-windowed rates.
    """

    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._counter_timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()

    def record(self, name: str, value: float = 1.0, metric_type: str = "counter"):
        """Record a metric. Types: counter (additive), gauge (latest), histogram (distribution)."""
        with self._lock:
            if metric_type == "counter":
                self._counters[name] += value
                self._counter_timestamps[name].append(time.time())
            elif metric_type == "gauge":
                self._gauges[name] = value
            elif metric_type == "histogram":
                self._histograms[name].append(value)

    def increment(self, name: str, delta: float = 1.0):
        self.record(name, delta, "counter")

    def set_gauge(self, name: str, value: float):
        self.record(name, value, "gauge")

    def observe(self, name: str, value: float):
        self.record(name, value, "histogram")

    def counter_rate(self, name: str, window_seconds: float = 60.0) -> float:
        """Compute the rate (events/sec) of a counter over the window."""
        with self._lock:
            timestamps = self._counter_timestamps.get(name)
            if not timestamps:
                return 0.0
            cutoff = time.time() - window_seconds
            count = sum(1 for t in timestamps if t > cutoff)
            return count / window_seconds

    def _percentile(self, sorted_vals: List[float], pct: float) -> float:
        """Compute percentile from sorted list."""
        if not sorted_vals:
            return 0.0
        k = (len(sorted_vals) - 1) * (pct / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(sorted_vals):
            return sorted_vals[-1]
        d = k - f
        return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])

    def histogram_stats(self, name: str) -> Dict[str, Any]:
        """Get full histogram stats including percentiles."""
        with self._lock:
            values = self._histograms.get(name)
            if not values or len(values) == 0:
                return {"count": 0}
            vals = sorted(values)
            total = sum(vals)
            return {
                "count": len(vals),
                "avg": round(total / len(vals), 4),
                "min": round(vals[0], 4),
                "max": round(vals[-1], 4),
                "p50": round(self._percentile(vals, 50), 4),
                "p95": round(self._percentile(vals, 95), 4),
                "p99": round(self._percentile(vals, 99), 4),
            }

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all metrics with percentiles."""
        with self._lock:
            result = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {},
            }
            for name, values in self._histograms.items():
                if values:
                    vals = sorted(values)
                    total = sum(vals)
                    result["histograms"][name] = {
                        "count": len(vals),
                        "avg": round(total / len(vals), 4),
                        "min": round(vals[0], 4),
                        "max": round(vals[-1], 4),
                        "p50": round(self._percentile(vals, 50), 4),
                        "p95": round(self._percentile(vals, 95), 4),
                        "p99": round(self._percentile(vals, 99), 4),
                    }
            return result

    def reset(self, name: Optional[str] = None):
        """Reset metrics. If name given, reset only that metric."""
        with self._lock:
            if name:
                self._counters.pop(name, None)
                self._gauges.pop(name, None)
                self._histograms.pop(name, None)
                self._counter_timestamps.pop(name, None)
            else:
                self._counters.clear()
                self._gauges.clear()
                self._histograms.clear()
                self._counter_timestamps.clear()


log_metrics = LogMetrics()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE HEALTH SCORER — Computes health per subsystem module
# ═══════════════════════════════════════════════════════════════════════════════

class ModuleHealthScorer:
    """
    Computes health scores (0.0–1.0) per module based on:
    - Ratio of errors to total log volume
    - Error burst frequency
    - Slow operation counts (from tracer)
    - PHI-weighted composite score
    """

    def __init__(self):
        self._log_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record_log(self, module: str):
        """Called on every log line to track volume per module."""
        with self._lock:
            self._log_counts[module] += 1

    def score(self, module: str) -> float:
        """
        Compute health score for a module. 1.0 = perfect, 0.0 = failing.
        """
        with self._lock:
            total_logs = self._log_counts.get(module, 0)

        error_count = error_aggregator.errors_by_module().get(module, 0)

        if total_logs == 0:
            return 1.0  # No data = assumed healthy

        error_ratio = error_count / max(total_logs, 1)
        # Invert: low error ratio → high score. PHI-weighted for sensitivity
        health = max(0.0, 1.0 - error_ratio * PHI * 5)
        return round(health, 4)

    def all_scores(self) -> Dict[str, float]:
        """Health scores for all known modules."""
        with self._lock:
            modules = set(self._log_counts.keys())
        modules |= set(error_aggregator.errors_by_module().keys())
        return {m: self.score(m) for m in sorted(modules)}

    def degraded_modules(self, threshold: float = 0.7) -> List[str]:
        """Return modules with health below threshold."""
        return [m for m, s in self.all_scores().items() if s < threshold]


_module_health = ModuleHealthScorer()


def get_module_health(module: Optional[str] = None) -> Any:
    """Get health score for one module or all modules."""
    if module:
        return _module_health.score(module)
    return _module_health.all_scores()


# ═══════════════════════════════════════════════════════════════════════════════
# ALERT MANAGER — Structured alerting based on observability signals
# ═══════════════════════════════════════════════════════════════════════════════

class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertManager:
    """
    Evaluates observability signals and fires structured alerts.
    Supports configurable thresholds, cooldown periods, and alert history.
    """

    def __init__(self):
        self._alerts: deque = deque(maxlen=200)
        self._cooldowns: Dict[str, float] = {}
        self._cooldown_seconds = 300.0  # 5 min between repeat alerts
        self._handlers: List[Callable] = []
        self._lock = threading.Lock()

    def register_handler(self, handler: Callable):
        """Register a callback: handler(alert_dict) — called on every new alert."""
        self._handlers.append(handler)

    def fire(self, name: str, level: AlertLevel, message: str, data: Optional[Dict] = None):
        """Fire an alert if not in cooldown."""
        now = time.time()
        with self._lock:
            last_fired = self._cooldowns.get(name, 0)
            if now - last_fired < self._cooldown_seconds:
                return  # In cooldown
            self._cooldowns[name] = now

            alert = {
                "name": name,
                "level": level.value,
                "message": message,
                "data": data or {},
                "timestamp": now,
            }
            self._alerts.append(alert)

        # Fire handlers outside the lock
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception:
                pass

    def check_thresholds(self):
        """Evaluate all observability signals and fire alerts as needed."""
        # Error burst
        if error_aggregator.detect_burst():
            self.fire("error_burst", AlertLevel.CRITICAL,
                      f"Error burst detected: {error_aggregator.error_rate(10.0):.1f} errors/sec",
                      {"rate": error_aggregator.error_rate(10.0)})

        # High PHI severity
        sev = error_aggregator.phi_severity_score()
        if sev > 0.7:
            self.fire("high_severity", AlertLevel.WARNING,
                      f"PHI severity elevated: {sev:.3f}",
                      {"severity": sev})

        # Degraded modules
        degraded = _module_health.degraded_modules(0.5)
        if degraded:
            self.fire("degraded_modules", AlertLevel.WARNING,
                      f"Degraded modules: {', '.join(degraded[:5])}",
                      {"modules": degraded})

        # High rate limiting
        if _rate_limiter.total_suppressed > 1000:
            self.fire("rate_limit_overflow", AlertLevel.INFO,
                      f"Log suppression high: {_rate_limiter.total_suppressed} total drops")

    def recent_alerts(self, n: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._alerts)[-n:]

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_alerts": len(self._alerts),
                "active_cooldowns": len(self._cooldowns),
                "handlers": len(self._handlers),
            }


alert_manager = AlertManager()


# ═══════════════════════════════════════════════════════════════════════════════
# LOG SEARCH — Search through recent log entries by pattern
# ═══════════════════════════════════════════════════════════════════════════════

class LogSearchIndex:
    """
    In-memory index of recent log events for fast search/grep.
    Keeps a rolling window of structured log records.
    """

    def __init__(self, max_entries: int = 2000):
        self._entries: deque = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def index(self, event_dict: Dict[str, Any]):
        """Index a log event for searchability."""
        with self._lock:
            self._entries.append({
                "event": str(event_dict.get("event", "")),
                "module": event_dict.get("module", ""),
                "level": event_dict.get("level", ""),
                "timestamp": event_dict.get("timestamp", time.time()),
                "correlation_id": event_dict.get("correlation_id"),
            })

    def search(self, query: str, module: Optional[str] = None,
               level: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Search logs by text pattern, optionally filtered by module/level."""
        query_lower = query.lower()
        results = []
        with self._lock:
            for entry in reversed(self._entries):
                if query_lower in entry["event"].lower():
                    if module and entry["module"] != module:
                        continue
                    if level and entry["level"] != level:
                        continue
                    results.append(entry)
                    if len(results) >= limit:
                        break
        return results

    def by_correlation(self, cid: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve all log entries for a given correlation ID."""
        results = []
        with self._lock:
            for entry in self._entries:
                if entry.get("correlation_id") == cid:
                    results.append(entry)
                    if len(results) >= limit:
                        break
        return results

    def status(self) -> Dict[str, Any]:
        return {"indexed_entries": len(self._entries)}


_log_search = LogSearchIndex()


def search_logs(query: str, module: Optional[str] = None,
                level: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Search recent log entries by text pattern."""
    return _log_search.search(query, module, level, limit)


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST LIFECYCLE TRACKER — Track request start → processing → complete
# ═══════════════════════════════════════════════════════════════════════════════

class RequestLifecycleTracker:
    """
    Tracks the lifecycle of requests through the system.
    Each request is identified by its correlation ID and progresses through
    phases: STARTED → PROCESSING → COMPLETED / FAILED.
    """

    def __init__(self, max_active: int = 500):
        self._active: Dict[str, Dict[str, Any]] = {}
        self._completed: deque = deque(maxlen=500)
        self._total_completed = 0
        self._total_failed = 0
        self._lock = threading.Lock()

    def start(self, request_id: str, metadata: Optional[Dict] = None):
        with self._lock:
            self._active[request_id] = {
                "start_time": time.time(),
                "phase": "STARTED",
                "metadata": metadata or {},
            }

    def update_phase(self, request_id: str, phase: str):
        with self._lock:
            if request_id in self._active:
                self._active[request_id]["phase"] = phase

    def complete(self, request_id: str, success: bool = True):
        with self._lock:
            req = self._active.pop(request_id, None)
            if req:
                dt = time.time() - req["start_time"]
                record = {
                    "request_id": request_id,
                    "duration_ms": round(dt * 1000, 2),
                    "success": success,
                    "timestamp": time.time(),
                }
                self._completed.append(record)
                if success:
                    self._total_completed += 1
                else:
                    self._total_failed += 1
                log_metrics.observe("request_duration_ms", dt * 1000)

    def active_count(self) -> int:
        return len(self._active)

    def status(self) -> Dict[str, Any]:
        return {
            "active_requests": len(self._active),
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
        }


request_tracker = RequestLifecycleTracker()


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE TRACER — Context manager for timing operations
# ═══════════════════════════════════════════════════════════════════════════════

class PerformanceTracer:
    """
    Records operation timing with nested tracing support, percentile stats,
    and automatic slow-operation alerting.
    Integrates with LogMetrics for histogram recording.
    """

    def __init__(self):
        self._traces: deque = deque(maxlen=2000)
        self._active_traces: Dict[str, float] = {}
        self._operation_stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._lock = threading.Lock()

    @contextmanager
    def trace(self, operation: str, module: str = "SYSTEM"):
        """Context manager for timing an operation."""
        t0 = time.perf_counter()
        trace_id = f"{module}:{operation}:{time.time()}"
        self._active_traces[trace_id] = t0
        try:
            yield
        except Exception as exc:
            dt = time.perf_counter() - t0
            error_aggregator.record(module, type(exc).__name__, str(exc))
            log_metrics.observe(f"{operation}_error_ms", dt * 1000)
            raise
        finally:
            dt = time.perf_counter() - t0
            self._active_traces.pop(trace_id, None)
            duration_ms = round(dt * 1000, 3)
            record = {
                "operation": operation,
                "module": module,
                "duration_ms": duration_ms,
                "timestamp": time.time(),
            }
            with self._lock:
                self._traces.append(record)
                self._operation_stats[operation].append(duration_ms)
            log_metrics.observe(f"{operation}_ms", duration_ms)

            # Alert on very slow operations (>5s)
            if duration_ms > 5000:
                alert_manager.fire(
                    f"slow_op_{operation}", AlertLevel.WARNING,
                    f"Slow operation: {operation} took {duration_ms:.0f}ms in {module}",
                    {"operation": operation, "duration_ms": duration_ms}
                )

    def operation_percentiles(self, operation: str) -> Dict[str, Any]:
        """Get p50/p95/p99 for a specific operation."""
        with self._lock:
            vals = self._operation_stats.get(operation)
            if not vals:
                return {"count": 0}
            sorted_vals = sorted(vals)
            n = len(sorted_vals)
            return {
                "count": n,
                "p50": round(sorted_vals[int(n * 0.50)], 3) if n > 0 else 0,
                "p95": round(sorted_vals[int(n * 0.95)] if n >= 20 else sorted_vals[-1], 3),
                "p99": round(sorted_vals[int(n * 0.99)] if n >= 100 else sorted_vals[-1], 3),
                "avg": round(sum(sorted_vals) / n, 3),
            }

    def all_operation_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get percentile stats for all tracked operations."""
        with self._lock:
            ops = list(self._operation_stats.keys())
        return {op: self.operation_percentiles(op) for op in ops}

    def _slow_ops_unlocked(self, threshold_ms: float = 100.0, n: int = 20) -> List[Dict[str, Any]]:
        slow = [t for t in self._traces if t["duration_ms"] > threshold_ms]
        slow.sort(key=lambda x: x["duration_ms"], reverse=True)
        return slow[:n]

    def slow_operations(self, threshold_ms: float = 100.0, n: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return self._slow_ops_unlocked(threshold_ms, n)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_traces": len(self._traces),
                "active_traces": len(self._active_traces),
                "tracked_operations": len(self._operation_stats),
                "slow_operations": len(self._slow_ops_unlocked()),
            }


_tracer = PerformanceTracer()


@contextmanager
def trace_performance(operation: str, module: str = "SYSTEM"):
    """Module-level convenience function for performance tracing."""
    with _tracer.trace(operation, module):
        yield


def get_operation_stats(operation: Optional[str] = None) -> Any:
    """Get percentile stats for one or all operations."""
    if operation:
        return _tracer.operation_percentiles(operation)
    return _tracer.all_operation_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# CONSCIOUSNESS-AWARE LOG ENRICHER — Adds system state to log context
# ═══════════════════════════════════════════════════════════════════════════════

_consciousness_cache: Dict[str, Any] = {}
_consciousness_cache_time: float = 0.0


def _get_consciousness_level() -> float:
    global _consciousness_cache, _consciousness_cache_time
    now = time.time()
    if now - _consciousness_cache_time < 10 and _consciousness_cache:
        return _consciousness_cache.get("cl", 0.5)
    try:
        path = Path(__file__).parent / ".l104_consciousness_o2_state.json"
        if path.exists():
            data = json.loads(path.read_text())
            cl = data.get("consciousness_level", 0.5)
            _consciousness_cache = {"cl": cl}
            _consciousness_cache_time = now
            return cl
    except Exception:
        pass
    return 0.5


def _consciousness_enricher(logger, method_name, event_dict):
    if method_name in ("warning", "error", "critical"):
        event_dict["consciousness_level"] = round(_get_consciousness_level(), 3)
    return event_dict


def _error_capture_processor(logger, method_name, event_dict):
    if method_name in ("error", "critical", "exception"):
        module = event_dict.get("module", "UNKNOWN")
        event = event_dict.get("event", "")
        error_type = event_dict.get("error_type", event_dict.get("exc_info", type(event).__name__ if event else "UnknownError"))
        if isinstance(error_type, tuple):
            error_type = error_type[0].__name__ if error_type[0] else "UnknownError"
        error_aggregator.record(module, str(error_type), str(event)[:200])
    return event_dict


def _rate_limit_processor(logger, method_name, event_dict):
    module = event_dict.get("module", "GLOBAL")
    if not _rate_limiter.allow(module):
        dead_letter_queue.capture(module, event_dict.get("event", ""), "rate_limited")
        raise structlog.DropEvent
    return event_dict


def _sampling_processor(logger, method_name, event_dict):
    """Structlog processor: probabilistic sampling for high-volume events."""
    event_key = event_dict.get("_sample_key") or event_dict.get("event", "")
    if not log_sampler.should_emit(str(event_key)):
        dead_letter_queue.capture(event_dict.get("module", "UNKNOWN"), str(event_key), "sampled")
        raise structlog.DropEvent
    return event_dict


def _metrics_processor(logger, method_name, event_dict):
    log_metrics.increment(f"log_{method_name}")
    module = event_dict.get("module", "UNKNOWN")
    log_metrics.increment(f"log_{module}_{method_name}")
    _module_health.record_log(module)
    return event_dict


def _search_index_processor(logger, method_name, event_dict):
    """Structlog processor: index all log events for searchability."""
    _log_search.index(event_dict)
    return event_dict


# ═══════════════════════════════════════════════════════════════════════════════
# CORE CONFIGURATION — Structlog + stdlib setup
# ═══════════════════════════════════════════════════════════════════════════════

def _configure_once():
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        _correlation_processor,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _rate_limit_processor,
        _sampling_processor,
        _metrics_processor,
        _error_capture_processor,
        _consciousness_enricher,
        _search_index_processor,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if LOG_FORMAT == "console":
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    if LOG_FILE:
        try:
            fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
            fh.setFormatter(formatter)
            root.addHandler(fh)
        except Exception:
            pass

    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — Backward-compatible + full observability suite
# ═══════════════════════════════════════════════════════════════════════════════

def get_logger(module_name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger bound to a module name.

    Args:
        module_name: Identifier for the subsystem (e.g. "SAGE_CORE", "NEURAL_MESH").

    Returns:
        A structlog BoundLogger with the module name pre-bound.

    This is the primary API — backward compatible with all 8+ importers.
    """
    _configure_once()
    return structlog.get_logger(module=module_name)


def get_observability_status() -> Dict[str, Any]:
    """Return full observability system status."""
    return {
        "version": LOGGING_VERSION,
        "log_level": LOG_LEVEL,
        "log_format": LOG_FORMAT,
        "rate_limit_per_sec": RATE_LIMIT_PER_SEC,
        "errors": error_aggregator.status(),
        "metrics_summary": log_metrics.summary(),
        "tracer": _tracer.status(),
        "sampler": log_sampler.status(),
        "dead_letter": dead_letter_queue.status(),
        "alerts": alert_manager.status(),
        "requests": request_tracker.status(),
        "search_index": _log_search.status(),
        "rate_limiter": _rate_limiter.status(),
        "health": "OPTIMAL" if error_aggregator.phi_severity_score() < 0.3 else
                  "WARNING" if error_aggregator.phi_severity_score() < 0.7 else "CRITICAL",
    }


def get_diagnostics_report() -> Dict[str, Any]:
    """Full diagnostics report for debugging and triage."""
    alert_manager.check_thresholds()  # Evaluate signals before reporting
    return {
        "version": LOGGING_VERSION,
        "top_errors": error_aggregator.top_errors(10),
        "errors_by_module": error_aggregator.errors_by_module(),
        "recent_errors": error_aggregator.recent_errors(10),
        "error_burst": error_aggregator.detect_burst(),
        "phi_severity": error_aggregator.phi_severity_score(),
        "slow_operations": _tracer.slow_operations(threshold_ms=100, n=10),
        "operation_percentiles": _tracer.all_operation_stats(),
        "module_health": _module_health.all_scores(),
        "degraded_modules": _module_health.degraded_modules(),
        "recent_alerts": alert_manager.recent_alerts(10),
        "dead_letter_recent": dead_letter_queue.recent(10),
        "metrics": log_metrics.summary(),
    }
