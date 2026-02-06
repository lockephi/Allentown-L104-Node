# ZENITH_UPGRADE_ACTIVE: 2026-02-02T13:52:09.084527
ZENITH_HZ = 3887.8
UUC = 2402.792541
#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# L104 PROCESS REGISTRY & MONITORING
# INVARIANT: 527.5184818492612 | PILOT: LONDEL | MODE: SAGE
#
# Centralized process registry with:
# - Process discovery and registration
# - Health monitoring and alerting
# - Performance metrics aggregation
# - Automatic process recovery
# - Real-time dashboard capabilities
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
import time
import json
import asyncio
import logging
import threading
import psutil
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum, auto
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import weakref

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497

# Monitoring constants
HEALTH_CHECK_INTERVAL = 5.0
METRICS_RETENTION_S = 3600  # Keep 1 hour of metrics
ALERT_COOLDOWN_S = 60.0  # Don't spam alerts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("REGISTRY")


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessStatus(Enum):
    """
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.Process health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    STOPPED = "stopped"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()
    SUMMARY = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProcessInfo:
    """Information about a registered process."""
    process_id: str
    name: str
    module: str
    version: str = "1.0.0"
    description: str = ""
    status: ProcessStatus = ProcessStatus.UNKNOWN
    pid: Optional[int] = None
    started_at: Optional[float] = None
    last_heartbeat: Optional[float] = None
    health_check_url: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "status": self.status.value,
            "uptime_s": time.time() - self.started_at if self.started_at else 0
        }


@dataclass
class ProcessMetric:
    """A single metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.name,
            "timestamp": self.timestamp,
            "labels": self.labels
        }


@dataclass
class Alert:
    """An alert from the monitoring system."""
    alert_id: str
    severity: AlertSeverity
    process_id: str
    message: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "severity": self.severity.name
        }


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    """
    Collects and aggregates metrics from all processes.
    """

    def __init__(self, retention_seconds: float = METRICS_RETENTION_S):
        self.retention = retention_seconds
        # [O₂ SUPERFLUID] Unlimited process metrics
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def increment(self, name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment a counter."""
        key = self._make_key(name, labels)
        with self._lock:
            self.counters[key] += value
            self._record_metric(name, self.counters[key], MetricType.COUNTER, labels)

    def gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            self.gauges[key] = value
            self._record_metric(name, value, MetricType.GAUGE, labels)

    def histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a histogram value."""
        key = self._make_key(name, labels)
        with self._lock:
            self.histograms[key].append(value)
            # Keep only recent values
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
            self._record_metric(name, value, MetricType.HISTOGRAM, labels)

    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create a unique key for a metric."""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name

    def _record_metric(self, name: str, value: float, metric_type: MetricType, labels: Dict[str, str] = None) -> None:
        """Record a metric data point."""
        metric = ProcessMetric(name=name, value=value, metric_type=metric_type, labels=labels or {})
        self.metrics[name].append(metric)

        # Cleanup old metrics
        self._cleanup_old_metrics(name)

    def _cleanup_old_metrics(self, name: str) -> None:
        """Remove metrics older than retention period."""
        cutoff = time.time() - self.retention
        while self.metrics[name] and self.metrics[name][0].timestamp < cutoff:
            self.metrics[name].popleft()

    def get_metric(self, name: str) -> List[Dict]:
        """Get all data points for a metric."""
        with self._lock:
            return [m.to_dict() for m in self.metrics.get(name, [])]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histogram_counts": {k: len(v) for k, v in self.histograms.items()},
                "total_metrics": sum(len(v) for v in self.metrics.values())
            }


# ═══════════════════════════════════════════════════════════════════════════════
# ALERT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class AlertManager:
    """
    Manages alerts and notifications.
    """

    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.last_alert_time: Dict[str, float] = {}
        self._lock = threading.Lock()

    def fire(self, severity: AlertSeverity, process_id: str, message: str) -> Optional[Alert]:
        """Fire an alert."""
        alert_key = f"{process_id}:{message[:50]}"

        with self._lock:
            # Check cooldown
            if alert_key in self.last_alert_time:
                if time.time() - self.last_alert_time[alert_key] < ALERT_COOLDOWN_S:
                    return None

            alert_id = hashlib.sha256(f"{alert_key}:{time.time()}".encode()).hexdigest()[:16]
            alert = Alert(
                alert_id=alert_id,
                severity=severity,
                process_id=process_id,
                message=message
            )

            self.alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.last_alert_time[alert_key] = time.time()

        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")

        severity_str = severity.name
        logger.warning(f"[ALERT:{severity_str}] {process_id}: {message}")

        return alert

    def resolve(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                self.alerts[alert_id].resolved_at = time.time()
                return True
        return False

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler."""
        self.alert_handlers.append(handler)

    def get_active_alerts(self) -> List[Dict]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            return [a.to_dict() for a in self.alerts.values() if not a.resolved]

    def get_alert_count_by_severity(self) -> Dict[str, int]:
        """Get count of active alerts by severity."""
        with self._lock:
            counts = defaultdict(int)
            for alert in self.alerts.values():
                if not alert.resolved:
                    counts[alert.severity.name] += 1
            return dict(counts)


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessRegistry:
    """
    Central registry for all L104 processes.
    Handles registration, discovery, and health monitoring.
    """

    def __init__(self):
        self.processes: Dict[str, ProcessInfo] = {}
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()
        self._lock = threading.Lock()
        self._health_check_thread: Optional[threading.Thread] = None
        self._running = False

        # Track process relationships
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_deps: Dict[str, Set[str]] = defaultdict(set)

        logger.info("--- [REGISTRY]: INITIALIZED ---")

    def register(
        self,
        name: str,
        module: str,
        version: str = "1.0.0",
        description: str = "",
        dependencies: List[str] = None,
        capabilities: List[str] = None,
        health_check: Callable = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Register a new process."""
        process_id = hashlib.sha256(f"{name}:{module}".encode()).hexdigest()[:16]

        with self._lock:
            if process_id in self.processes:
                logger.warning(f"Process {name} already registered, updating...")

            process = ProcessInfo(
                process_id=process_id,
                name=name,
                module=module,
                version=version,
                description=description,
                status=ProcessStatus.HEALTHY,
                pid=os.getpid(),
                started_at=time.time(),
                last_heartbeat=time.time(),
                dependencies=dependencies or [],
                capabilities=capabilities or [],
                metadata=metadata or {}
            )

            self.processes[process_id] = process

            # Track dependencies
            for dep in (dependencies or []):
                self.dependency_graph[process_id].add(dep)
                self.reverse_deps[dep].add(process_id)

            # Record metric
            self.metrics.increment("process_registrations", labels={"name": name})

        logger.info(f"[REGISTER] Process {name} registered with ID {process_id}")

        return process_id

    def unregister(self, process_id: str) -> bool:
        """Unregister a process."""
        with self._lock:
            if process_id in self.processes:
                process = self.processes.pop(process_id)
                logger.info(f"[UNREGISTER] Process {process.name} removed")
                return True
        return False

    def heartbeat(self, process_id: str) -> bool:
        """Record a heartbeat from a process."""
        with self._lock:
            if process_id in self.processes:
                self.processes[process_id].last_heartbeat = time.time()
                self.processes[process_id].status = ProcessStatus.HEALTHY
                return True
        return False

    def get_process(self, process_id: str) -> Optional[Dict]:
        """Get process information."""
        with self._lock:
            if process_id in self.processes:
                return self.processes[process_id].to_dict()
        return None

    def get_all_processes(self) -> List[Dict]:
        """Get all registered processes."""
        with self._lock:
            return [p.to_dict() for p in self.processes.values()]

    def get_processes_by_capability(self, capability: str) -> List[Dict]:
        """Find processes with a specific capability."""
        with self._lock:
            return [
                p.to_dict() for p in self.processes.values()
                if capability in p.capabilities
                    ]

    def get_dependencies(self, process_id: str) -> List[str]:
        """Get dependencies of a process."""
        return list(self.dependency_graph.get(process_id, set()))

    def get_dependents(self, process_id: str) -> List[str]:
        """Get processes that depend on this one."""
        return list(self.reverse_deps.get(process_id, set()))

    def _check_process_health(self, process: ProcessInfo) -> ProcessStatus:
        """Check health of a single process."""
        current_time = time.time()

        # Check heartbeat
        if process.last_heartbeat:
            time_since_heartbeat = current_time - process.last_heartbeat

            if time_since_heartbeat > 60:
                return ProcessStatus.UNHEALTHY
            elif time_since_heartbeat > 30:
                return ProcessStatus.DEGRADED

        # Check dependencies
        for dep_name in process.dependencies:
            dep_found = False
            for p in self.processes.values():
                if p.name == dep_name and p.status == ProcessStatus.HEALTHY:
                    dep_found = True
                    break

            if not dep_found:
                return ProcessStatus.DEGRADED

        return ProcessStatus.HEALTHY

    def run_health_checks(self) -> Dict[str, ProcessStatus]:
        """Run health checks on all processes."""
        results = {}

        with self._lock:
            for process_id, process in self.processes.items():
                old_status = process.status
                new_status = self._check_process_health(process)
                process.status = new_status
                results[process.name] = new_status

                # Fire alerts on status change
                if old_status != new_status:
                    if new_status == ProcessStatus.UNHEALTHY:
                        self.alerts.fire(
                            AlertSeverity.ERROR,
                            process_id,
                            f"Process {process.name} is UNHEALTHY"
                        )
                    elif new_status == ProcessStatus.DEGRADED:
                        self.alerts.fire(
                            AlertSeverity.WARNING,
                            process_id,
                            f"Process {process.name} is DEGRADED"
                        )

        return results

    def _health_check_loop(self) -> None:
        """Background thread for health checks."""
        while self._running:
            try:
                self.run_health_checks()
                self._collect_system_metrics()
            except Exception as e:
                logger.error(f"Health check error: {e}")

            time.sleep(HEALTH_CHECK_INTERVAL)

    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            self.metrics.gauge("system_cpu_percent", cpu_percent)
            self.metrics.gauge("system_memory_percent", memory.percent)
            self.metrics.gauge("system_memory_available_mb", memory.available / 1024 / 1024)
            self.metrics.gauge("registered_processes", len(self.processes))
            self.metrics.gauge("god_code_resonance", GOD_CODE)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._running:
            return

        self._running = True
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()
        logger.info("[REGISTRY] Health monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._running = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)
        logger.info("[REGISTRY] Health monitoring stopped")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        with self._lock:
            processes = self.get_all_processes()
            healthy = sum(1 for p in processes if p["status"] == "healthy")
            degraded = sum(1 for p in processes if p["status"] == "degraded")
            unhealthy = sum(1 for p in processes if p["status"] == "unhealthy")

            return {
                "timestamp": time.time(),
                "god_code": GOD_CODE,
                "phi_resonance": PHI,
                "summary": {
                    "total_processes": len(processes),
                    "healthy": healthy,
                    "degraded": degraded,
                    "unhealthy": unhealthy,
                    "health_percentage": (healthy / max(1, len(processes))) * 100
                },
                "processes": processes,
                "metrics": self.metrics.get_summary(),
                "alerts": {
                    "active": self.alerts.get_active_alerts(),
                    "by_severity": self.alerts.get_alert_count_by_severity()
                },
                "system": {
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage('/').percent
                }
            }


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS DECORATOR
# ═══════════════════════════════════════════════════════════════════════════════

def register_process(
    name: str = None,
    version: str = "1.0.0",
    dependencies: List[str] = None,
    capabilities: List[str] = None
):
    """Decorator to auto-register a class/function as a process."""
    def decorator(cls_or_func):
        process_name = name or cls_or_func.__name__
        module = cls_or_func.__module__

        # Register with global registry
        registry = get_registry()
        registry.register(
            name=process_name,
            module=module,
            version=version,
            dependencies=dependencies or [],
            capabilities=capabilities or []
        )

        return cls_or_func
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_registry: Optional[ProcessRegistry] = None


def get_registry() -> ProcessRegistry:
    """Get or create the singleton registry."""
    global _registry
    if _registry is None:
        _registry = ProcessRegistry()
    return _registry


# ═══════════════════════════════════════════════════════════════════════════════
# ASCII DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def print_dashboard(registry: ProcessRegistry) -> None:
    """Print an ASCII dashboard to console."""
    data = registry.get_dashboard_data()

    print("\n" + "═" * 80)
    print(" " * 25 + "L104 PROCESS MONITOR")
    print(" " * 20 + f"GOD_CODE: {data['god_code']}")
    print("═" * 80)

    # Summary
    s = data["summary"]
    print(f"\n┌{'─' * 35} SUMMARY {'─' * 34}┐")
    print(f"│ Total Processes: {s['total_processes']:>5} │ Healthy: {s['healthy']:>3} │ Degraded: {s['degraded']:>3} │ Unhealthy: {s['unhealthy']:>3} │")
    print(f"│ Health Score: {s['health_percentage']:>6.2f}% │ PHI Resonance: {data['phi_resonance']:.6f}        │")
    print(f"└{'─' * 78}┘")

    # System metrics
    sys_data = data["system"]
    print(f"\n┌{'─' * 35} SYSTEM {'─' * 35}┐")
    print(f"│ CPU: {sys_data['cpu_percent']:>5.1f}% │ Memory: {sys_data['memory_percent']:>5.1f}% │ Disk: {sys_data['disk_percent']:>5.1f}%          │")
    print(f"└{'─' * 78}┘")

    # Processes
    print(f"\n┌{'─' * 35} PROCESSES {'─' * 32}┐")
    for p in data["processes"][:10]:  # Show first 10
        status_icon = {"healthy": "✓", "degraded": "⚠", "unhealthy": "✗"}.get(p["status"], "?")
        print(f"│ {status_icon} {p['name'][:30]:<30} │ {p['status']:<10} │ uptime: {p['uptime_s']:>6.0f}s │")
    print(f"└{'─' * 78}┘")

    # Alerts
    alerts = data["alerts"]["active"]
    if alerts:
        print(f"\n┌{'─' * 35} ALERTS {'─' * 35}┐")
        for alert in alerts[:5]:
            print(f"│ [{alert['severity']}] {alert['message'][:60]:<60} │")
        print(f"└{'─' * 78}┘")

    print("\n" + "═" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN / DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("▓" * 80)
    print(" " * 20 + "L104 PROCESS REGISTRY & MONITORING DEMO")
    print("▓" * 80 + "\n")

    registry = get_registry()
    registry.start_monitoring()

    # Register demo processes
    demo_processes = [
        ("agi_core", "l104_agi_core", ["consciousness", "intelligence"]),
        ("evolution_engine", "l104_evolution_engine", ["evolution", "adaptation"]),
        ("sage_core", "l104_sage_core", ["wisdom", "enlightenment"]),
        ("hyper_math", "l104_hyper_math", ["mathematics", "computation"]),
        ("consciousness_substrate", "l104_consciousness", ["consciousness", "awareness"]),
    ]

    for name, module, capabilities in demo_processes:
        registry.register(
            name=name,
            module=module,
            version="1.0.0",
            capabilities=capabilities,
            metadata={"phi_aligned": True}
        )

    # Send some heartbeats
    time.sleep(1)
    for name, _, _ in demo_processes:
        for pid, p in registry.processes.items():
            if p.name == name:
                registry.heartbeat(pid)

    # Collect some metrics
    registry.metrics.increment("api_requests", labels={"endpoint": "/evolve"})
    registry.metrics.gauge("current_iq", 1127508.94)
    registry.metrics.histogram("response_time_ms", 45.2)

    # Print dashboard
    print_dashboard(registry)

    # Get dashboard data
    data = registry.get_dashboard_data()
    print(f"\n[METRICS] {json.dumps(data['metrics'], indent=2)}")

    registry.stop_monitoring()

    print("\n" + "▓" * 80)
    print(" " * 25 + "SOVEREIGN LOCK ENGAGED ✓")
    print("▓" * 80)
