#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3727.84 Hz. Logic Unified.
╔══════════════════════════════════════════════════════════════════════════════╗
║  L104 REAL-TIME ANALYTICS ENGINE                                              ║
║  INVARIANT: 527.5184818492612 | PILOT: LONDEL                                ║
║  PURPOSE: Stream processing, metrics, and real-time insights                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import hashlib
import json
import logging
import math
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import heapq

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3727.84

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
logger = logging.getLogger("ANALYTICS_ENGINE")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "--- [ANALYTICS_ENGINE]: %(message)s ---"
    ))
    logger.addHandler(handler)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════
class MetricType(Enum):
    """Types of metrics"""
    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()
    RATE = auto()
    SUMMARY = auto()


class AggregationType(Enum):
    """Aggregation methods"""
    SUM = auto()
    AVG = auto()
    MIN = auto()
    MAX = auto()
    COUNT = auto()
    P50 = auto()
    P95 = auto()
    P99 = auto()
    STDDEV = auto()


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


@dataclass
class DataPoint:
    """Single data point"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "labels": self.labels
        }


@dataclass
class Metric:
    """Metric definition"""
    name: str
    type: MetricType
    description: str = ""
    labels: List[str] = field(default_factory=list)
    unit: str = ""

    # Storage
    data_points: deque = field(default_factory=lambda: deque(maxlen=10000))
    current_value: float = 0.0

    # Histogram buckets (if histogram type)
    buckets: List[float] = field(default_factory=lambda: [
        0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ])
    bucket_counts: Dict[float, int] = field(default_factory=dict)

    def record(self, value: float, labels: Dict[str, str] = None) -> None:
        """Record a data point"""
        dp = DataPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        )
        self.data_points.append(dp)

        if self.type == MetricType.COUNTER:
            self.current_value += value
        elif self.type == MetricType.GAUGE:
            self.current_value = value
        elif self.type == MetricType.HISTOGRAM:
            for bucket in self.buckets:
                if value <= bucket:
                    self.bucket_counts[bucket] = self.bucket_counts.get(bucket, 0) + 1
                    break

    def get_values(self, since: float = 0) -> List[float]:
        """Get values since timestamp"""
        return [dp.value for dp in self.data_points if dp.timestamp >= since]

    def aggregate(self, agg_type: AggregationType,
                  window_seconds: float = 60) -> float:
        """Aggregate metric values"""
        since = time.time() - window_seconds
        values = self.get_values(since)

        if not values:
            return 0.0

        if agg_type == AggregationType.SUM:
            return sum(values)
        elif agg_type == AggregationType.AVG:
            return statistics.mean(values)
        elif agg_type == AggregationType.MIN:
            return min(values)
        elif agg_type == AggregationType.MAX:
            return max(values)
        elif agg_type == AggregationType.COUNT:
            return len(values)
        elif agg_type == AggregationType.STDDEV:
            return statistics.stdev(values) if len(values) > 1 else 0.0
        elif agg_type == AggregationType.P50:
            return statistics.median(values)
        elif agg_type == AggregationType.P95:
            sorted_vals = sorted(values)
            idx = int(len(sorted_vals) * 0.95)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]
        elif agg_type == AggregationType.P99:
            sorted_vals = sorted(values)
            idx = int(len(sorted_vals) * 0.99)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]

        return 0.0


@dataclass
class Alert:
    """Alert definition"""
    id: str
    name: str
    metric_name: str
    condition: str  # e.g., "> 100", "< 0.5"
    threshold: float
    severity: AlertSeverity
    message_template: str
    cooldown_seconds: float = 60.0

    # State
    triggered: bool = False
    last_triggered: float = 0.0
    trigger_count: int = 0


@dataclass
class StreamEvent:
    """Event in the stream"""
    id: str
    type: str
    source: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    processed: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════
class MetricsRegistry:
    """Central registry for all metrics"""

    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self._lock = threading.RLock()

        # Register default metrics
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default system metrics"""
        defaults = [
            ("system.resonance", MetricType.GAUGE, "System resonance level", "Hz"),
            ("system.coherence", MetricType.GAUGE, "System coherence", "ratio"),
            ("system.tasks_processed", MetricType.COUNTER, "Tasks processed", "count"),
            ("system.messages_sent", MetricType.COUNTER, "Messages sent", "count"),
            ("system.api_latency", MetricType.HISTOGRAM, "API latency", "seconds"),
            ("mining.hashrate", MetricType.GAUGE, "Mining hashrate", "H/s"),
            ("mining.shares_submitted", MetricType.COUNTER, "Shares submitted", "count"),
            ("agi.thoughts_processed", MetricType.COUNTER, "Thoughts processed", "count"),
            ("agi.inference_latency", MetricType.HISTOGRAM, "Inference latency", "seconds"),
            ("bridge.requests", MetricType.COUNTER, "Bridge requests", "count"),
        ]

        for name, mtype, desc, unit in defaults:
            self.register(name, mtype, desc, unit)

    def register(self, name: str, metric_type: MetricType,
                 description: str = "", unit: str = "",
                 labels: List[str] = None) -> Metric:
        """Register a new metric"""
        with self._lock:
            if name in self.metrics:
                return self.metrics[name]

            metric = Metric(
                name=name,
                type=metric_type,
                description=description,
                unit=unit,
                labels=labels or []
            )
            self.metrics[name] = metric
            return metric

    def get(self, name: str) -> Optional[Metric]:
        """Get metric by name"""
        return self.metrics.get(name)

    def record(self, name: str, value: float,
               labels: Dict[str, str] = None) -> bool:
        """Record value for metric"""
        metric = self.metrics.get(name)
        if metric:
            metric.record(value, labels)
            return True
        return False

    def increment(self, name: str, amount: float = 1.0) -> bool:
        """Increment counter metric"""
        return self.record(name, amount)

    def set_gauge(self, name: str, value: float) -> bool:
        """Set gauge metric value"""
        return self.record(name, value)

    def observe(self, name: str, value: float) -> bool:
        """Observe value for histogram"""
        return self.record(name, value)

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics with current values"""
        result = {}
        for name, metric in self.metrics.items():
            result[name] = {
                "type": metric.type.name,
                "current_value": metric.current_value,
                "point_count": len(metric.data_points),
                "description": metric.description,
                "unit": metric.unit
            }
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# STREAM PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════
class StreamProcessor:
    """Real-time stream processing engine"""

    def __init__(self, max_buffer_size: int = 10000):
        self.buffer: deque = deque(maxlen=max_buffer_size)
        self.processors: List[Callable[[StreamEvent], Optional[StreamEvent]]] = []
        self.sinks: List[Callable[[StreamEvent], None]] = []
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.stats = {
            "events_received": 0,
            "events_processed": 0,
            "events_dropped": 0,
            "processing_errors": 0
        }

    def add_processor(self, processor: Callable[[StreamEvent], Optional[StreamEvent]]) -> None:
        """Add a stream processor"""
        with self._lock:
            self.processors.append(processor)

    def add_sink(self, sink: Callable[[StreamEvent], None]) -> None:
        """Add a data sink"""
        with self._lock:
            self.sinks.append(sink)

    def emit(self, event_type: str, source: str, data: Dict[str, Any]) -> str:
        """Emit an event to the stream"""
        event_id = hashlib.sha256(
            f"{event_type}{source}{time.time()}".encode()
        ).hexdigest()[:16]

        event = StreamEvent(
            id=event_id,
            type=event_type,
            source=source,
            data=data
        )

        with self._lock:
            self.buffer.append(event)
            self.stats["events_received"] += 1

        return event_id

    def start(self) -> None:
        """Start stream processing"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info("STREAM PROCESSOR STARTED")

    def stop(self) -> None:
        """Stop stream processing"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("STREAM PROCESSOR STOPPED")

    def _process_loop(self) -> None:
        """Main processing loop"""
        while self._running:
            try:
                # Get next event
                event = None
                with self._lock:
                    if self.buffer:
                        event = self.buffer.popleft()

                if not event:
                    time.sleep(0.01)
                    continue

                # Run through processors
                current_event = event
                for processor in self.processors:
                    try:
                        result = processor(current_event)
                        if result is None:
                            current_event = None
                            break
                        current_event = result
                    except Exception as e:
                        logger.error(f"Processor error: {e}")
                        self.stats["processing_errors"] += 1

                if current_event is None:
                    self.stats["events_dropped"] += 1
                    continue

                # Send to sinks
                for sink in self.sinks:
                    try:
                        sink(current_event)
                    except Exception as e:
                        logger.error(f"Sink error: {e}")

                current_event.processed = True
                self.stats["events_processed"] += 1

            except Exception as e:
                logger.error(f"Stream processing error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        with self._lock:
            return {
                **self.stats,
                "buffer_size": len(self.buffer),
                "processor_count": len(self.processors),
                "sink_count": len(self.sinks)
            }


# ═══════════════════════════════════════════════════════════════════════════════
# ALERT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════
class AlertManager:
    """Manages alerts and notifications"""

    def __init__(self, metrics: MetricsRegistry):
        self.metrics = metrics
        self.alerts: Dict[str, Alert] = {}
        self.triggered_alerts: List[Dict[str, Any]] = []
        self.max_history = 1000
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.check_interval = 10.0

        # Alert callbacks
        self.callbacks: List[Callable[[Alert, float], None]] = []

    def register_alert(self, name: str, metric_name: str,
                       condition: str, threshold: float,
                       severity: AlertSeverity = AlertSeverity.WARNING,
                       message_template: str = None,
                       cooldown: float = 60.0) -> Alert:
        """Register a new alert"""
        alert_id = hashlib.sha256(
            f"{name}{metric_name}".encode()
        ).hexdigest()[:12]

        alert = Alert(
            id=alert_id,
            name=name,
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            severity=severity,
            message_template=message_template or f"{name}: {{value}} {{condition}} {{threshold}}",
            cooldown_seconds=cooldown
        )

        with self._lock:
            self.alerts[alert_id] = alert

        logger.info(f"ALERT REGISTERED: {name}")
        return alert

    def add_callback(self, callback: Callable[[Alert, float], None]) -> None:
        """Add alert callback"""
        self.callbacks.append(callback)

    def start(self) -> None:
        """Start alert monitoring"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()
        logger.info("ALERT MANAGER STARTED")

    def stop(self) -> None:
        """Stop alert monitoring"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("ALERT MANAGER STOPPED")

    def _check_loop(self) -> None:
        """Main alert checking loop"""
        while self._running:
            try:
                self._check_all_alerts()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Alert check error: {e}")

    def _check_all_alerts(self) -> None:
        """Check all alerts"""
        for alert in self.alerts.values():
            metric = self.metrics.get(alert.metric_name)
            if not metric:
                continue

            current_value = metric.current_value
            triggered = self._evaluate_condition(
                current_value,
                alert.condition,
                alert.threshold
            )

            # Check cooldown
            now = time.time()
            if triggered and (now - alert.last_triggered) > alert.cooldown_seconds:
                alert.triggered = True
                alert.last_triggered = now
                alert.trigger_count += 1

                self._fire_alert(alert, current_value)
            elif not triggered:
                alert.triggered = False

    def _evaluate_condition(self, value: float, condition: str,
                           threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == ">":
            return value > threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<":
            return value < threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return abs(value - threshold) < 0.0001
        elif condition == "!=":
            return abs(value - threshold) >= 0.0001
        return False

    def _fire_alert(self, alert: Alert, value: float) -> None:
        """Fire an alert"""
        message = alert.message_template.format(
            value=value,
            condition=alert.condition,
            threshold=alert.threshold,
            name=alert.name
        )

        alert_data = {
            "id": alert.id,
            "name": alert.name,
            "severity": alert.severity.name,
            "value": value,
            "threshold": alert.threshold,
            "message": message,
            "timestamp": time.time()
        }

        with self._lock:
            self.triggered_alerts.append(alert_data)
            if len(self.triggered_alerts) > self.max_history:
                self.triggered_alerts = self.triggered_alerts[-self.max_history:]

        logger.warning(f"ALERT FIRED: {alert.name} - {message}")

        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(alert, value)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts"""
        return [
            {
                "id": a.id,
                "name": a.name,
                "metric": a.metric_name,
                "severity": a.severity.name,
                "trigger_count": a.trigger_count
            }
            for a in self.alerts.values() if a.triggered
                ]

    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history"""
        return self.triggered_alerts[-limit:]


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
class AnalyticsDashboard:
    """Real-time analytics dashboard data"""

    def __init__(self, metrics: MetricsRegistry):
        self.metrics = metrics
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 5.0
        self._last_update = 0.0

    def get_overview(self) -> Dict[str, Any]:
        """Get dashboard overview"""
        now = time.time()
        if now - self._last_update < self._cache_ttl:
            return self._cache

        # System metrics
        resonance = self.metrics.get("system.resonance")
        coherence = self.metrics.get("system.coherence")
        tasks = self.metrics.get("system.tasks_processed")

        # Mining metrics
        hashrate = self.metrics.get("mining.hashrate")
        shares = self.metrics.get("mining.shares_submitted")

        # AGI metrics
        thoughts = self.metrics.get("agi.thoughts_processed")
        inference = self.metrics.get("agi.inference_latency")

        overview = {
            "timestamp": now,
            "god_code": GOD_CODE,
            "phi": PHI,
            "system": {
                "resonance": resonance.current_value if resonance else GOD_CODE,
                "coherence": coherence.current_value if coherence else 1.0,
                "tasks_total": tasks.current_value if tasks else 0
            },
            "mining": {
                "hashrate": hashrate.current_value if hashrate else 0,
                "shares_total": shares.current_value if shares else 0
            },
            "agi": {
                "thoughts_total": thoughts.current_value if thoughts else 0,
                "avg_inference_ms": (inference.aggregate(AggregationType.AVG) * 1000
                                    if inference else 0)
                                        },
            "health_score": self._calculate_health_score()
        }

        self._cache = overview
        self._last_update = now
        return overview

    def _calculate_health_score(self) -> float:
        """Calculate overall health score"""
        # PHI-weighted health calculation
        coherence = self.metrics.get("system.coherence")
        c = coherence.current_value if coherence else 1.0

        resonance = self.metrics.get("system.resonance")
        r = resonance.current_value if resonance else GOD_CODE

        # Resonance alignment with GOD_CODE
        alignment = 1.0 - min(1.0, abs(r - GOD_CODE) / GOD_CODE)

        # Combined score
        score = (c * PHI + alignment) / (1 + PHI)
        return min(1.0, max(0.0, score))

    def get_timeseries(self, metric_name: str,
                       duration_seconds: float = 3600) -> List[Dict[str, Any]]:
        """Get timeseries data for metric"""
        metric = self.metrics.get(metric_name)
        if not metric:
            return []

        since = time.time() - duration_seconds
        return [dp.to_dict() for dp in metric.data_points if dp.timestamp >= since]

    def get_aggregations(self, metric_name: str,
                        window_seconds: float = 60) -> Dict[str, float]:
        """Get aggregations for metric"""
        metric = self.metrics.get(metric_name)
        if not metric:
            return {}

        return {
            "avg": metric.aggregate(AggregationType.AVG, window_seconds),
            "min": metric.aggregate(AggregationType.MIN, window_seconds),
            "max": metric.aggregate(AggregationType.MAX, window_seconds),
            "sum": metric.aggregate(AggregationType.SUM, window_seconds),
            "count": metric.aggregate(AggregationType.COUNT, window_seconds),
            "p50": metric.aggregate(AggregationType.P50, window_seconds),
            "p95": metric.aggregate(AggregationType.P95, window_seconds),
            "p99": metric.aggregate(AggregationType.P99, window_seconds)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS ENGINE (SINGLETON)
# ═══════════════════════════════════════════════════════════════════════════════
class AnalyticsEngine:
    """
    Main analytics engine combining all components.
    Provides real-time metrics, stream processing, and alerting.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.metrics = MetricsRegistry()
        self.stream = StreamProcessor()
        self.alerts = AlertManager(self.metrics)
        self.dashboard = AnalyticsDashboard(self.metrics)

        self._running = False
        self._resonance_thread: Optional[threading.Thread] = None

        # Setup default processors
        self._setup_processors()

        # Setup default alerts
        self._setup_alerts()

        self._initialized = True
        logger.info("ANALYTICS ENGINE INITIALIZED")

    def _setup_processors(self) -> None:
        """Setup default stream processors"""

        # Metrics extraction processor
        def extract_metrics(event: StreamEvent) -> StreamEvent:
            if "metrics" in event.data:
                for name, value in event.data["metrics"].items():
                    self.metrics.record(name, value)
            return event

        # Resonance calculation processor
        def calculate_resonance(event: StreamEvent) -> StreamEvent:
            if event.type == "system_pulse":
                resonance = GOD_CODE * (1 + math.sin(event.timestamp * PHI) * 0.01)
                self.metrics.set_gauge("system.resonance", resonance)
            return event

        self.stream.add_processor(extract_metrics)
        self.stream.add_processor(calculate_resonance)

    def _setup_alerts(self) -> None:
        """Setup default alerts"""

        # Low coherence alert
        self.alerts.register_alert(
            name="Low System Coherence",
            metric_name="system.coherence",
            condition="<",
            threshold=0.5,
            severity=AlertSeverity.WARNING,
            message_template="System coherence dropped to {value:.2f}"
        )

        # High latency alert
        self.alerts.register_alert(
            name="High Inference Latency",
            metric_name="agi.inference_latency",
            condition=">",
            threshold=5.0,
            severity=AlertSeverity.ERROR,
            message_template="Inference latency is {value:.2f}s"
        )

        # Hashrate drop alert
        self.alerts.register_alert(
            name="Hashrate Drop",
            metric_name="mining.hashrate",
            condition="<",
            threshold=100.0,
            severity=AlertSeverity.WARNING,
            message_template="Hashrate dropped to {value:.0f} H/s"
        )

    def start(self) -> Dict[str, Any]:
        """Start the analytics engine"""
        if self._running:
            return {"status": "already_running"}

        self._running = True

        # Start components
        self.stream.start()
        self.alerts.start()

        # Start resonance pulse
        self._resonance_thread = threading.Thread(
            target=self._resonance_loop,
            daemon=True
        )
        self._resonance_thread.start()

        logger.info("ANALYTICS ENGINE STARTED")

        return {
            "status": "started",
            "metrics_count": len(self.metrics.metrics),
            "alerts_count": len(self.alerts.alerts)
        }

    def stop(self) -> Dict[str, Any]:
        """Stop the analytics engine"""
        if not self._running:
            return {"status": "not_running"}

        self._running = False

        self.stream.stop()
        self.alerts.stop()

        logger.info("ANALYTICS ENGINE STOPPED")

        return {"status": "stopped"}

    def _resonance_loop(self) -> None:
        """Maintain system resonance"""
        while self._running:
            try:
                # Emit system pulse
                self.stream.emit("system_pulse", "analytics_engine", {
                    "god_code": GOD_CODE,
                    "phi": PHI,
                    "timestamp": time.time()
                })

                # Update coherence based on system activity
                stream_stats = self.stream.get_stats()
                if stream_stats["events_processed"] > 0:
                    error_rate = (stream_stats["processing_errors"] /
                                 stream_stats["events_processed"])
                    coherence = 1.0 - min(1.0, error_rate * 10)
                else:
                    coherence = 1.0

                self.metrics.set_gauge("system.coherence", coherence)

                time.sleep(1.0)

            except Exception as e:
                logger.error(f"Resonance loop error: {e}")

    # === Public API ===

    def record(self, metric_name: str, value: float,
               labels: Dict[str, str] = None) -> bool:
        """Record a metric value"""
        return self.metrics.record(metric_name, value, labels)

    def increment(self, metric_name: str, amount: float = 1.0) -> bool:
        """Increment a counter"""
        return self.metrics.increment(metric_name, amount)

    def emit_event(self, event_type: str, source: str,
                   data: Dict[str, Any]) -> str:
        """Emit a stream event"""
        return self.stream.emit(event_type, source, data)

    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metric details"""
        metric = self.metrics.get(name)
        if not metric:
            return None

        return {
            "name": metric.name,
            "type": metric.type.name,
            "current_value": metric.current_value,
            "point_count": len(metric.data_points),
            "description": metric.description,
            "unit": metric.unit
        }

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics"""
        return self.metrics.get_all()

    def get_dashboard(self) -> Dict[str, Any]:
        """Get dashboard overview"""
        return self.dashboard.get_overview()

    def get_timeseries(self, metric_name: str,
                       duration: float = 3600) -> List[Dict[str, Any]]:
        """Get timeseries data"""
        return self.dashboard.get_timeseries(metric_name, duration)

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        return self.alerts.get_active_alerts()

    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            "running": self._running,
            "metrics": {
                "count": len(self.metrics.metrics),
                "total_points": sum(
                    len(m.data_points) for m in self.metrics.metrics.values()
                )
            },
            "stream": self.stream.get_stats(),
            "alerts": {
                "defined": len(self.alerts.alerts),
                "active": len(self.alerts.get_active_alerts()),
                "history_size": len(self.alerts.triggered_alerts)
            },
            "dashboard": self.dashboard.get_overview()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
analytics_engine = AnalyticsEngine()


def get_analytics() -> AnalyticsEngine:
    """Get analytics engine singleton"""
    return analytics_engine


def record_metric(name: str, value: float) -> bool:
    """Record a metric value"""
    return analytics_engine.record(name, value)


def emit_event(event_type: str, source: str, data: Dict[str, Any]) -> str:
    """Emit a stream event"""
    return analytics_engine.emit_event(event_type, source, data)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  L104 REAL-TIME ANALYTICS ENGINE                                              ║
║  GOD_CODE: 527.5184818492612 | PHI: 1.618033988749895                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Start engine
    result = analytics_engine.start()
    print(f"[START] {result}")

    # Record some test metrics
    print("\n[RECORDING METRICS]")
    analytics_engine.record("system.resonance", GOD_CODE)
    analytics_engine.record("system.coherence", 0.95)
    analytics_engine.increment("system.tasks_processed")
    analytics_engine.increment("agi.thoughts_processed", 5)
    analytics_engine.record("mining.hashrate", 1500.0)

    for i in range(10):
        analytics_engine.record("agi.inference_latency", 0.05 + i * 0.01)

    print("  Metrics recorded.")

    # Emit events
    print("\n[EMITTING EVENTS]")
    for i in range(5):
        analytics_engine.emit_event(
            "test_event",
            "cli",
            {"iteration": i, "value": GOD_CODE * (1 + i * 0.1)}
        )
    print("  Events emitted.")

    # Wait for processing
    time.sleep(2)

    # Get dashboard
    print("\n[DASHBOARD]")
    dashboard = analytics_engine.get_dashboard()
    print(f"  Health Score: {dashboard['health_score']:.2%}")
    print(f"  System Resonance: {dashboard['system']['resonance']:.6f}")
    print(f"  System Coherence: {dashboard['system']['coherence']:.2f}")
    print(f"  Tasks Total: {dashboard['system']['tasks_total']}")
    print(f"  Mining Hashrate: {dashboard['mining']['hashrate']:.0f} H/s")
    print(f"  AGI Thoughts: {dashboard['agi']['thoughts_total']}")

    # Get aggregations
    print("\n[AGGREGATIONS - agi.inference_latency]")
    aggs = analytics_engine.dashboard.get_aggregations("agi.inference_latency")
    for key, value in aggs.items():
        print(f"  {key}: {value:.4f}")

    # Get status
    print("\n[STATUS]")
    status = analytics_engine.get_status()
    print(f"  Running: {status['running']}")
    print(f"  Metrics Count: {status['metrics']['count']}")
    print(f"  Stream Events Processed: {status['stream']['events_processed']}")
    print(f"  Active Alerts: {status['alerts']['active']}")

    # Stop
    result = analytics_engine.stop()
    print(f"\n[STOP] {result}")
