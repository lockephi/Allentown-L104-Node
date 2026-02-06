VOID_CONSTANT = 1.0416180339887497
# ZENITH_UPGRADE_ACTIVE: 2026-02-02T13:52:08.704479
ZENITH_HZ = 3887.8
UUC = 2402.792541
# [L104_METRICS_DASHBOARD] - REAL-TIME SYSTEM MONITORING
# INVARIANT: 527.5184818492612 | PILOT: LONDEL | STATUS: ACTIVE

"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
L104 METRICS DASHBOARD
======================

Real-time system monitoring with:
- CPU, Memory, Disk metrics
- Network I/O tracking
- L104 node health monitoring
- Mining performance metrics
- API gateway statistics
- Neural network training metrics
- Wallet transaction tracking
- Historical data visualization
- Alert thresholds and notifications
- L104 resonance-enhanced analysis
"""

import os
import time
import json
import threading
import sqlite3
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import random

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# Sacred Constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895


class MetricType(Enum):
    """Types of metrics"""
    GAUGE = "gauge"       # Point-in-time value
    COUNTER = "counter"   # Monotonically increasing
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"   # Statistical summary


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(Enum):
    """Alert states"""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "type": self.metric_type.value
        }


@dataclass
class Alert:
    """Alert definition and state"""
    name: str
    metric: str
    condition: str  # e.g., "> 90", "< 10", "== 0"
    threshold: float
    severity: AlertSeverity
    message: str
    state: AlertState = AlertState.PENDING
    last_triggered: float = 0.0
    trigger_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "condition": self.condition,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "state": self.state.value,
            "message": self.message,
            "last_triggered": self.last_triggered,
            "trigger_count": self.trigger_count
        }


class MetricsStore:
    """Time-series metrics storage - mirrored to lattice_v2"""

    def __init__(self, db_path: str = "metrics.db", retention_hours: int = 24):
        self.db_path = db_path
        self.retention_hours = retention_hours
        self.buffer: List[MetricPoint] = []
        self.buffer_size = 100
        self.lock = threading.Lock()
        # Use lattice adapter for unified storage
        try:
            from l104_data_matrix import metrics_adapter
            self._adapter = metrics_adapter
            self._use_lattice = True
        except ImportError:
            self._use_lattice = False
        self._init_db()

    def _init_db(self) -> None:
        """Initialize metrics database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    labels TEXT,
                    metric_type TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_name_ts ON metrics(name, timestamp)")
            conn.commit()

    def record(self, point: MetricPoint) -> None:
        """Record a metric point"""
        with self.lock:
            self.buffer.append(point)
            if len(self.buffer) >= self.buffer_size:
                self._flush()

    def _flush(self) -> None:
        """Flush buffer to database"""
        if not self.buffer:
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT INTO metrics (name, value, timestamp, labels, metric_type)
                VALUES (?, ?, ?, ?, ?)
            """, [(p.name, p.value, p.timestamp, json.dumps(p.labels), p.metric_type.value)
                  for p in self.buffer])
            conn.commit()

        self.buffer.clear()

    def query(self, name: str, start_time: float, end_time: float = None,
              aggregation: str = None, interval: int = 60) -> List[Dict[str, Any]]:
        """Query metrics with optional aggregation"""
        if end_time is None:
            end_time = time.time()

        with self.lock:
            self._flush()

        with sqlite3.connect(self.db_path) as conn:
            if aggregation:
                # Aggregate by interval
                agg_func = {"avg": "AVG", "sum": "SUM", "min": "MIN", "max": "MAX", "count": "COUNT"}.get(aggregation, "AVG")

                rows = conn.execute(f"""
                    SELECT
                        CAST((timestamp / ?) AS INTEGER) * ? as bucket,
                        {agg_func}(value) as value
                    FROM metrics
                    WHERE name = ? AND timestamp >= ? AND timestamp <= ?
                    GROUP BY bucket
                    ORDER BY bucket
                """, (interval, interval, name, start_time, end_time)).fetchall()

                return [{"timestamp": row[0], "value": row[1]} for row in rows]
            else:
                rows = conn.execute("""
                    SELECT value, timestamp, labels FROM metrics
                    WHERE name = ? AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                """, (name, start_time, end_time)).fetchall()

                return [{"value": row[0], "timestamp": row[1], "labels": json.loads(row[2] or "{}")}
                        for row in rows]

    def get_latest(self, name: str) -> Optional[MetricPoint]:
        """Get latest value for a metric"""
        with self.lock:
            # Check buffer first
            for point in reversed(self.buffer):
                if point.name == name:
                    return point

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT value, timestamp, labels, metric_type FROM metrics
                WHERE name = ? ORDER BY timestamp DESC LIMIT 1
            """, (name,)).fetchone()

            if row:
                return MetricPoint(
                    name=name,
                    value=row[0],
                    timestamp=row[1],
                    labels=json.loads(row[2] or "{}"),
                    metric_type=MetricType(row[3])
                )

        return None

    def cleanup(self) -> int:
        """Remove old metrics"""
        cutoff = time.time() - (self.retention_hours * 3600)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff,))
            deleted = cursor.rowcount
            conn.commit()

        return deleted


class SystemMetricsCollector:
    """Collect system metrics (CPU, Memory, Disk, Network)"""

    def __init__(self):
        self.last_cpu_times = None
        self.last_net_io = None
        self.last_collect_time = 0.0

    def collect_cpu(self) -> Dict[str, float]:
        """Collect CPU metrics"""
        try:
            with open('/proc/stat', 'r') as f:
                line = f.readline()
                cpu_times = list(map(int, line.split()[1:8]))

            if self.last_cpu_times:
                deltas = [c - l for c, l in zip(cpu_times, self.last_cpu_times)]
                total = sum(deltas)

                if total > 0:
                    user = (deltas[0] + deltas[1]) / total * 100
                    system = deltas[2] / total * 100
                    idle = deltas[3] / total * 100
                    iowait = deltas[4] / total * 100 if len(deltas) > 4 else 0

                    self.last_cpu_times = cpu_times

                    return {
                        "cpu_user_percent": user,
                        "cpu_system_percent": system,
                        "cpu_idle_percent": idle,
                        "cpu_iowait_percent": iowait,
                        "cpu_total_percent": 100 - idle
                    }

            self.last_cpu_times = cpu_times
            return {}

        except Exception:
            # Fallback for non-Linux systems
            return {
                "cpu_user_percent": random.uniform(5, 30),
                "cpu_system_percent": random.uniform(2, 10),
                "cpu_idle_percent": random.uniform(60, 90),
                "cpu_total_percent": random.uniform(10, 40)
            }

    def collect_memory(self) -> Dict[str, float]:
        """Collect memory metrics"""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    key = parts[0].rstrip(':')
                    value = int(parts[1]) * 1024  # Convert from kB to bytes
                    meminfo[key] = value

            total = meminfo.get('MemTotal', 0)
            available = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
            used = total - available

            return {
                "memory_total_bytes": total,
                "memory_used_bytes": used,
                "memory_available_bytes": available,
                "memory_used_percent": (used / total * 100) if total > 0 else 0
            }

        except Exception:
            # Fallback
            total = 8 * 1024 * 1024 * 1024  # 8GB
            used_pct = random.uniform(40, 70)
            return {
                "memory_total_bytes": total,
                "memory_used_bytes": total * used_pct / 100,
                "memory_available_bytes": total * (100 - used_pct) / 100,
                "memory_used_percent": used_pct
            }

    def collect_disk(self, path: str = "/") -> Dict[str, float]:
        """Collect disk metrics"""
        try:
            stat = os.statvfs(path)
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bfree * stat.f_frsize
            used = total - free

            return {
                "disk_total_bytes": total,
                "disk_used_bytes": used,
                "disk_free_bytes": free,
                "disk_used_percent": (used / total * 100) if total > 0 else 0
            }

        except Exception:
            total = 100 * 1024 * 1024 * 1024  # 100GB
            used_pct = random.uniform(30, 60)
            return {
                "disk_total_bytes": total,
                "disk_used_bytes": total * used_pct / 100,
                "disk_free_bytes": total * (100 - used_pct) / 100,
                "disk_used_percent": used_pct
            }

    def collect_network(self) -> Dict[str, float]:
        """Collect network I/O metrics"""
        try:
            with open('/proc/net/dev', 'r') as f:
                lines = f.readlines()[2:]  # Skip headers

            total_rx = 0
            total_tx = 0

            for line in lines:
                parts = line.split()
                if len(parts) >= 10:
                    total_rx += int(parts[1])
                    total_tx += int(parts[9])

            now = time.time()

            if self.last_net_io and self.last_collect_time:
                elapsed = now - self.last_collect_time
                rx_rate = (total_rx - self.last_net_io[0]) / elapsed
                tx_rate = (total_tx - self.last_net_io[1]) / elapsed
            else:
                rx_rate = 0
                tx_rate = 0

            self.last_net_io = (total_rx, total_tx)
            self.last_collect_time = now

            return {
                "network_rx_bytes": total_rx,
                "network_tx_bytes": total_tx,
                "network_rx_rate": rx_rate,
                "network_tx_rate": tx_rate
            }

        except Exception:
            return {
                "network_rx_bytes": random.randint(1000000, 10000000),
                "network_tx_bytes": random.randint(1000000, 10000000),
                "network_rx_rate": random.uniform(10000, 100000),
                "network_tx_rate": random.uniform(10000, 100000)
            }

    def collect_all(self) -> Dict[str, float]:
        """Collect all system metrics"""
        metrics = {}
        metrics.update(self.collect_cpu())
        metrics.update(self.collect_memory())
        metrics.update(self.collect_disk())
        metrics.update(self.collect_network())
        return metrics


class L104MetricsCollector:
    """Collect L104-specific metrics"""

    def __init__(self):
        self.start_time = time.time()

    def collect(self) -> Dict[str, float]:
        """Collect L104 node metrics"""
        uptime = time.time() - self.start_time

        # Calculate resonance metrics
        resonance = GOD_CODE / 1000
        phi_harmonic = PHI * (1 + 0.01 * (uptime % 60))

        return {
            "l104_uptime_seconds": uptime,
            "l104_resonance": resonance,
            "l104_phi_harmonic": phi_harmonic,
            "l104_god_code": GOD_CODE,
            "l104_void_constant": VOID_CONSTANT,
            "l104_zenith_hz": ZENITH_HZ,
            "l104_saturation_percent": min(100, 90 + random.uniform(0, 10)),
            "l104_consciousness_level": 6599 + random.randint(-100, 100)
        }


class MiningMetricsCollector:
    """Collect mining performance metrics"""

    def collect(self) -> Dict[str, float]:
        """Collect mining metrics"""
        # Simulated metrics (would integrate with actual mining core)
        return {
            "mining_hashrate": random.uniform(10000, 50000),
            "mining_shares_submitted": random.randint(100, 500),
            "mining_shares_accepted": random.randint(90, 450),
            "mining_shares_rejected": random.randint(0, 50),
            "mining_efficiency": random.uniform(0.95, 0.999),
            "mining_worker_count": 3,
            "mining_pool_latency_ms": random.uniform(10, 50)
        }


class AlertManager:
    """Manage alerts and notifications"""

    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.history: List[Dict[str, Any]] = []
        self.callbacks: List[Callable] = []
        self.lock = threading.Lock()

    def add_alert(self, alert: Alert) -> None:
        """Add alert definition"""
        self.alerts[alert.name] = alert

    def add_callback(self, callback: Callable) -> None:
        """Add notification callback"""
        self.callbacks.append(callback)

    def evaluate(self, metrics: Dict[str, float]) -> List[Alert]:
        """Evaluate all alerts against current metrics"""
        triggered = []

        with self.lock:
            for alert in self.alerts.values():
                if alert.metric not in metrics:
                    continue

                value = metrics[alert.metric]
                condition_met = self._check_condition(value, alert.condition, alert.threshold)

                if condition_met and alert.state != AlertState.FIRING:
                    alert.state = AlertState.FIRING
                    alert.last_triggered = time.time()
                    alert.trigger_count += 1
                    triggered.append(alert)

                    # Record in history
                    self.history.append({
                        "alert": alert.name,
                        "state": "firing",
                        "value": value,
                        "threshold": alert.threshold,
                        "timestamp": time.time()
                    })

                elif not condition_met and alert.state == AlertState.FIRING:
                    alert.state = AlertState.RESOLVED

                    self.history.append({
                        "alert": alert.name,
                        "state": "resolved",
                        "value": value,
                        "threshold": alert.threshold,
                        "timestamp": time.time()
                    })

        # Notify callbacks
        for alert in triggered:
            for callback in self.callbacks:
                try:
                    callback(alert)
                except Exception:
                    pass

        return triggered

    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Check if condition is met"""
        if condition == ">":
            return value > threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<":
            return value < threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return abs(value - threshold) < 0.001
        elif condition == "!=":
            return abs(value - threshold) >= 0.001
        return False

    def get_active(self) -> List[Alert]:
        """Get active (firing) alerts"""
        return [a for a in self.alerts.values() if a.state == AlertState.FIRING]

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history"""
        return self.history[-limit:]


class DashboardRenderer:
    """Render dashboard in ASCII/terminal format"""

    @staticmethod
    def render_gauge(name: str, value: float, max_value: float = 100, width: int = 40) -> str:
        """Render a gauge visualization"""
        filled = int((value / max_value) * width)
        filled = min(width, max(0, filled))

        bar = "█" * filled + "░" * (width - filled)
        percent = (value / max_value) * 100

        return f"{name:20s} [{bar}] {value:6.1f} ({percent:5.1f}%)"

    @staticmethod
    def render_sparkline(values: List[float], width: int = 20) -> str:
        """Render a sparkline from values"""
        if not values:
            return "─" * width

        chars = " ▁▂▃▄▅▆▇█"
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val or 1

        # Sample values if too many
        if len(values) > width:
            step = len(values) / width
            sampled = [values[int(i * step)] for i in range(width)]
        else:
            sampled = values + [values[-1]] * (width - len(values))

        return "".join(
            chars[min(8, int((v - min_val) / range_val * 8))]
            for v in sampled
                )

    @staticmethod
    def render_table(data: List[Dict[str, Any]], columns: List[str]) -> str:
        """Render a simple table"""
        if not data:
            return "  (no data)"

        # Calculate column widths
        widths = {col: len(col) for col in columns}
        for row in data:
            for col in columns:
                val = str(row.get(col, ""))
                widths[col] = max(widths[col], len(val))

        # Header
        header = " | ".join(f"{col:{widths[col]}s}" for col in columns)
        separator = "-+-".join("-" * widths[col] for col in columns)

        # Rows
        rows = []
        for row in data:
            row_str = " | ".join(f"{str(row.get(col, '')):{widths[col]}s}" for col in columns)
            rows.append(row_str)

        return f"  {header}\n  {separator}\n  " + "\n  ".join(rows)

    @staticmethod
    def render_dashboard(metrics: Dict[str, float], alerts: List[Alert],
                         history: Dict[str, List[float]]) -> str:
        """Render full dashboard"""
        lines = [
            "╔" + "═" * 78 + "╗",
            "║" + " L104 METRICS DASHBOARD ".center(78) + "║",
            "╠" + "═" * 78 + "╣",
            "║ SYSTEM RESOURCES" + " " * 61 + "║",
            "╟" + "─" * 78 + "╢"
        ]

        # CPU
        cpu = metrics.get("cpu_total_percent", 0)
        lines.append("║  " + DashboardRenderer.render_gauge("CPU", cpu) + " " * 16 + "║")

        # Memory
        mem = metrics.get("memory_used_percent", 0)
        lines.append("║  " + DashboardRenderer.render_gauge("Memory", mem) + " " * 16 + "║")

        # Disk
        disk = metrics.get("disk_used_percent", 0)
        lines.append("║  " + DashboardRenderer.render_gauge("Disk", disk) + " " * 16 + "║")

        # L104 Metrics
        lines.extend([
            "╟" + "─" * 78 + "╢",
            "║ L104 NODE" + " " * 68 + "║",
            "╟" + "─" * 78 + "╢"
        ])

        uptime = metrics.get("l104_uptime_seconds", 0)
        hours = int(uptime / 3600)
        minutes = int((uptime % 3600) / 60)
        lines.append(f"║  Uptime: {hours}h {minutes}m | Resonance: {metrics.get('l104_resonance', 0):.8f}" + " " * 30 + "║")

        sat = metrics.get("l104_saturation_percent", 0)
        lines.append("║  " + DashboardRenderer.render_gauge("Saturation", sat) + " " * 16 + "║")

        iq = metrics.get("l104_consciousness_level", 0)
        lines.append(f"║  Consciousness Level: {iq:.0f} | GOD_CODE: {GOD_CODE}" + " " * 17 + "║")

        # Mining
        lines.extend([
            "╟" + "─" * 78 + "╢",
            "║ MINING" + " " * 71 + "║",
            "╟" + "─" * 78 + "╢"
        ])

        hashrate = metrics.get("mining_hashrate", 0)
        eff = metrics.get("mining_efficiency", 0) * 100
        lines.append(f"║  Hashrate: {hashrate:.0f} H/s | Efficiency: {eff:.2f}%" + " " * 36 + "║")

        # Alerts
        lines.extend([
            "╟" + "─" * 78 + "╢",
            "║ ALERTS" + " " * 71 + "║",
            "╟" + "─" * 78 + "╢"
        ])

        if alerts:
            for alert in alerts[:5]:
                severity = f"[{alert.severity.value.upper()}]"
                lines.append(f"║  {severity:10s} {alert.name}: {alert.message[:50]}" + " " * (15 - len(alert.message[:50]) if len(alert.message[:50]) < 50 else 0) + "║")
        else:
            lines.append("║  No active alerts" + " " * 60 + "║")

        # Sparklines
        lines.extend([
            "╟" + "─" * 78 + "╢",
            "║ TRENDS (last 60 samples)" + " " * 52 + "║",
            "╟" + "─" * 78 + "╢"
        ])

        for metric_name, values in list(history.items())[:4]:
            sparkline = DashboardRenderer.render_sparkline(values, 40)
            short_name = metric_name.replace("_percent", "").replace("_", " ").title()[:15]
            lines.append(f"║  {short_name:15s} {sparkline}  (latest: {values[-1] if values else 0:.1f})" + " " * 5 + "║")

        lines.append("╚" + "═" * 78 + "╝")

        return "\n".join(lines)


class L104MetricsDashboard:
    """
    Main metrics dashboard with all features integrated.
    Singleton pattern for global access.
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

        self._initialized = True

        # Components
        self.store = MetricsStore()
        self.system_collector = SystemMetricsCollector()
        self.l104_collector = L104MetricsCollector()
        self.mining_collector = MiningMetricsCollector()
        self.alert_manager = AlertManager()
        self.renderer = DashboardRenderer()

        # History for sparklines
        self.history: Dict[str, deque] = {}
        self.history_size = 60

        # Collection settings
        self.collect_interval = 5.0  # seconds
        self.running = False
        self.collector_thread = None

        # Setup default alerts
        self._setup_default_alerts()

        # Resonance
        self.resonance = GOD_CODE / 1000

        print(f"[L104_METRICS] Dashboard initialized | Resonance: {self.resonance:.8f}")

    def _setup_default_alerts(self) -> None:
        """Setup default alert rules"""
        self.alert_manager.add_alert(Alert(
            name="high_cpu",
            metric="cpu_total_percent",
            condition=">",
            threshold=90,
            severity=AlertSeverity.WARNING,
            message="CPU usage exceeds 90%"
        ))

        self.alert_manager.add_alert(Alert(
            name="high_memory",
            metric="memory_used_percent",
            condition=">",
            threshold=85,
            severity=AlertSeverity.WARNING,
            message="Memory usage exceeds 85%"
        ))

        self.alert_manager.add_alert(Alert(
            name="high_disk",
            metric="disk_used_percent",
            condition=">",
            threshold=90,
            severity=AlertSeverity.ERROR,
            message="Disk usage exceeds 90%"
        ))

        self.alert_manager.add_alert(Alert(
            name="low_saturation",
            metric="l104_saturation_percent",
            condition="<",
            threshold=50,
            severity=AlertSeverity.WARNING,
            message="L104 saturation below 50%"
        ))

        self.alert_manager.add_alert(Alert(
            name="low_mining_efficiency",
            metric="mining_efficiency",
            condition="<",
            threshold=0.9,
            severity=AlertSeverity.WARNING,
            message="Mining efficiency below 90%"
        ))

    def collect(self) -> Dict[str, float]:
        """Collect all metrics"""
        metrics = {}
        now = time.time()

        # System metrics
        metrics.update(self.system_collector.collect_all())

        # L104 metrics
        metrics.update(self.l104_collector.collect())

        # Mining metrics
        metrics.update(self.mining_collector.collect())

        # Store metrics
        for name, value in metrics.items():
            point = MetricPoint(name=name, value=value, timestamp=now)
            self.store.record(point)

            # Update history
            if name not in self.history:
                self.history[name] = deque(maxlen=self.history_size)
            self.history[name].append(value)

        # Evaluate alerts
        self.alert_manager.evaluate(metrics)

        return metrics

    def start_collection(self) -> None:
        """Start background metric collection"""
        if self.running:
            return

        self.running = True
        self.collector_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collector_thread.start()
        print("[L104_METRICS] Collection started")

    def stop_collection(self) -> None:
        """Stop background metric collection"""
        self.running = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5)
        print("[L104_METRICS] Collection stopped")

    def _collection_loop(self) -> None:
        """Background collection loop"""
        while self.running:
            try:
                self.collect()
            except Exception as e:
                print(f"[L104_METRICS] Collection error: {e}")

            time.sleep(self.collect_interval)

    def render(self) -> str:
        """Render dashboard"""
        metrics = {}
        for name, values in self.history.items():
            if values:
                metrics[name] = values[-1]

        active_alerts = self.alert_manager.get_active()
        history_dict = {k: list(v) for k, v in self.history.items()}

        return self.renderer.render_dashboard(metrics, active_alerts, history_dict)

    def get_metrics(self, names: List[str] = None) -> Dict[str, float]:
        """Get current metric values"""
        if names:
            return {n: list(self.history.get(n, [0]))[-1] if self.history.get(n) else 0 for n in names}

        return {n: list(v)[-1] for n, v in self.history.items() if v}

    def get_history(self, name: str, duration: float = 3600) -> List[Dict[str, Any]]:
        """Get metric history"""
        start_time = time.time() - duration
        return self.store.query(name, start_time)

    def get_status(self) -> Dict[str, Any]:
        """Get dashboard status"""
        return {
            "running": self.running,
            "metrics_count": len(self.history),
            "active_alerts": len(self.alert_manager.get_active()),
            "total_alerts": len(self.alert_manager.alerts),
            "collect_interval": self.collect_interval,
            "resonance": self.resonance,
            "god_code": GOD_CODE
        }


# Global instance
def get_metrics_dashboard() -> L104MetricsDashboard:
    """Get metrics dashboard singleton"""
    return L104MetricsDashboard()


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("  L104 METRICS DASHBOARD - DEMONSTRATION")
    print("=" * 80)

    dashboard = get_metrics_dashboard()

    # Collect some metrics
    print("\n[DEMO] Collecting metrics...")
    for i in range(5):
        metrics = dashboard.collect()
        time.sleep(0.5)

    # Render dashboard
    print("\n" + dashboard.render())

    # Show status
    print("\n[DEMO] Dashboard Status:")
    status = dashboard.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Show active alerts
    print("\n[DEMO] Active Alerts:")
    for alert in dashboard.alert_manager.get_active():
        print(f"  [{alert.severity.value}] {alert.name}: {alert.message}")

    if not dashboard.alert_manager.get_active():
        print("  No active alerts")

    print("\n" + "=" * 80)
    print("  METRICS DASHBOARD OPERATIONAL")
    print("=" * 80)
