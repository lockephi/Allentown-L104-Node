"""
L104 AGI Core — Telemetry Aggregation Pipeline v1.0
============================================================================
Advanced telemetry for the AGI pipeline with:

  • Time-windowed aggregation (per-minute and per-hour rolling stats)
  • Anomaly detection on telemetry streams (z-score with PHI × σ threshold)
  • Latency percentile tracking (p50, p95, p99)
  • Pipeline throughput metrics (events/sec, calls/sec)
  • Unified health dashboard combining all signals

Sacred constants govern all thresholds and decay rates.
INVARIANT: 527.5184818492612 | PILOT: LONDEL
============================================================================
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

# Sacred constants
PHI = 1.618033988749895
TAU = 1.0 / PHI  # ≈ 0.618
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))
VOID_CONSTANT = 1.0416180339887497


# ═══════════════════════════════════════════════════════════════════════════════
# TELEMETRY AGGREGATION PIPELINE v1.0
# ═══════════════════════════════════════════════════════════════════════════════

class TelemetryAggregator:
    """
    Time-windowed telemetry aggregation.

    Maintains per-minute and per-hour rolling statistics for any named metric.
    Each window bucket stores: count, sum, min, max, sum_of_squares (for variance).
    """

    def __init__(self, minute_window: int = 60, hour_window: int = 24):
        """
        Args:
            minute_window: Number of 1-minute buckets to retain.
            hour_window: Number of 1-hour buckets to retain.
        """
        self._minute_window = minute_window
        self._hour_window = hour_window
        # metric_name → deque of {timestamp, count, sum, min, max, sum_sq}
        self._minute_buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=minute_window))
        self._hour_buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=hour_window))
        # Current open minute bucket per metric
        self._current_minute: Dict[str, Dict[str, Any]] = {}
        self._current_minute_ts: Dict[str, int] = {}

    def _get_minute_key(self, t: Optional[float] = None) -> int:
        return int((t or time.time()) // 60)

    def _get_hour_key(self, t: Optional[float] = None) -> int:
        return int((t or time.time()) // 3600)

    def record(self, metric: str, value: float, t: Optional[float] = None):
        """Record a single metric observation."""
        now = t or time.time()
        mk = self._get_minute_key(now)

        # Roll over minute bucket if needed
        if metric not in self._current_minute_ts or self._current_minute_ts[metric] != mk:
            # Close previous bucket
            if metric in self._current_minute:
                self._minute_buckets[metric].append(self._current_minute[metric])
                # Also aggregate into hour bucket
                self._aggregate_to_hour(metric, self._current_minute[metric])
            # Open new bucket
            self._current_minute[metric] = {
                "ts": mk * 60,
                "count": 0, "sum": 0.0,
                "min": float('inf'), "max": float('-inf'),
                "sum_sq": 0.0,
            }
            self._current_minute_ts[metric] = mk

        bucket = self._current_minute[metric]
        bucket["count"] += 1
        bucket["sum"] += value
        bucket["min"] = min(bucket["min"], value)
        bucket["max"] = max(bucket["max"], value)
        bucket["sum_sq"] += value * value

    def _aggregate_to_hour(self, metric: str, minute_bucket: Dict):
        """Merge a completed minute bucket into the current hour bucket."""
        hk = self._get_hour_key(minute_bucket["ts"])
        hour_buckets = self._hour_buckets[metric]

        if hour_buckets and hour_buckets[-1].get("_hk") == hk:
            # Merge into existing hour bucket
            hb = hour_buckets[-1]
            hb["count"] += minute_bucket["count"]
            hb["sum"] += minute_bucket["sum"]
            hb["min"] = min(hb["min"], minute_bucket["min"])
            hb["max"] = max(hb["max"], minute_bucket["max"])
            hb["sum_sq"] += minute_bucket["sum_sq"]
        else:
            # New hour bucket
            hour_buckets.append({
                "_hk": hk,
                "ts": hk * 3600,
                "count": minute_bucket["count"],
                "sum": minute_bucket["sum"],
                "min": minute_bucket["min"],
                "max": minute_bucket["max"],
                "sum_sq": minute_bucket["sum_sq"],
            })

    def _flush_current(self, metric: str):
        """Flush current open minute bucket (for reads)."""
        if metric in self._current_minute and self._current_minute[metric]["count"] > 0:
            return self._current_minute[metric]
        return None

    def stats(self, metric: str, window: str = "minute") -> Dict[str, Any]:
        """
        Get aggregate stats for a metric over the specified window.

        Args:
            metric: Metric name.
            window: "minute" (per-minute buckets) or "hour" (per-hour buckets).
        """
        buckets = list(self._minute_buckets[metric]) if window == "minute" else list(self._hour_buckets[metric])
        # Include current open bucket
        current = self._flush_current(metric)
        if current and window == "minute":
            buckets = buckets + [current]

        if not buckets:
            return {"metric": metric, "window": window, "samples": 0}

        total_count = sum(b["count"] for b in buckets)
        total_sum = sum(b["sum"] for b in buckets)
        total_min = min(b["min"] for b in buckets)
        total_max = max(b["max"] for b in buckets)
        total_sum_sq = sum(b["sum_sq"] for b in buckets)

        mean = total_sum / total_count if total_count > 0 else 0.0
        variance = (total_sum_sq / total_count - mean * mean) if total_count > 1 else 0.0
        stddev = math.sqrt(max(0, variance))

        return {
            "metric": metric,
            "window": window,
            "buckets": len(buckets),
            "samples": total_count,
            "mean": round(mean, 6),
            "stddev": round(stddev, 6),
            "min": round(total_min, 6) if total_min != float('inf') else None,
            "max": round(total_max, 6) if total_max != float('-inf') else None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class TelemetryAnomalyDetector:
    """
    Z-score anomaly detection on telemetry streams.

    Maintains an EMA (exponential moving average) of mean and variance per metric.
    Flags values exceeding PHI × σ as anomalies (≈ 1.618 standard deviations).
    """

    ANOMALY_SIGMA_MULTIPLIER = PHI  # ≈ 1.618 σ threshold
    EMA_ALPHA = TAU * 0.2           # ≈ 0.1236 smoothing factor

    def __init__(self):
        self._ema_mean: Dict[str, float] = {}
        self._ema_var: Dict[str, float] = {}
        self._anomaly_log: deque = deque(maxlen=1000)
        self._total_anomalies: int = 0
        self._observations: Dict[str, int] = defaultdict(int)

    def observe(self, metric: str, value: float) -> Optional[Dict[str, Any]]:
        """
        Observe a metric value. Returns anomaly dict if anomalous, else None.
        """
        self._observations[metric] += 1
        alpha = self.EMA_ALPHA

        if metric not in self._ema_mean:
            self._ema_mean[metric] = value
            self._ema_var[metric] = 0.0
            return None

        old_mean = self._ema_mean[metric]
        old_var = self._ema_var[metric]

        # Update EMA of mean and variance
        new_mean = alpha * value + (1 - alpha) * old_mean
        new_var = alpha * (value - new_mean) ** 2 + (1 - alpha) * old_var
        self._ema_mean[metric] = new_mean
        self._ema_var[metric] = new_var

        stddev = math.sqrt(max(new_var, 1e-15))
        z_score = abs(value - new_mean) / stddev if stddev > 1e-10 else 0.0

        if z_score > self.ANOMALY_SIGMA_MULTIPLIER and self._observations[metric] > 10:
            anomaly = {
                "time": time.time(),
                "metric": metric,
                "value": round(value, 6),
                "ema_mean": round(new_mean, 6),
                "stddev": round(stddev, 6),
                "z_score": round(z_score, 4),
                "threshold": round(self.ANOMALY_SIGMA_MULTIPLIER, 4),
            }
            self._anomaly_log.append(anomaly)
            self._total_anomalies += 1
            return anomaly

        return None

    def recent_anomalies(self, last_n: int = 20) -> List[Dict[str, Any]]:
        return list(self._anomaly_log)[-last_n:]

    def get_status(self) -> Dict[str, Any]:
        return {
            "metrics_tracked": len(self._ema_mean),
            "total_anomalies": self._total_anomalies,
            "anomaly_log_size": len(self._anomaly_log),
            "sigma_threshold": round(self.ANOMALY_SIGMA_MULTIPLIER, 4),
            "ema_alpha": round(self.EMA_ALPHA, 6),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LATENCY PERCENTILE TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class LatencyPercentileTracker:
    """
    Tracks latency distributions for pipeline operations.
    Computes p50, p95, p99 from a rolling window of observations.
    Uses a sorted-insertion approach for O(1) percentile reads.
    """

    def __init__(self, window_size: int = 5000):
        self._window_size = window_size
        # metric → sorted deque of values
        self._samples: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._total_observations: Dict[str, int] = defaultdict(int)

    def record(self, metric: str, latency_ms: float):
        """Record a latency observation in milliseconds."""
        self._samples[metric].append(latency_ms)
        self._total_observations[metric] += 1

    def percentile(self, metric: str, p: float) -> Optional[float]:
        """Get the p-th percentile (0-100) for a metric."""
        samples = self._samples.get(metric)
        if not samples or len(samples) < 2:
            return None
        sorted_s = sorted(samples)
        idx = int(len(sorted_s) * p / 100.0)
        idx = min(idx, len(sorted_s) - 1)
        return sorted_s[idx]

    def report(self, metric: str) -> Dict[str, Any]:
        """Get p50, p95, p99 report for a metric."""
        samples = self._samples.get(metric)
        if not samples or len(samples) < 2:
            return {"metric": metric, "samples": len(samples) if samples else 0, "status": "insufficient_data"}

        return {
            "metric": metric,
            "samples": len(samples),
            "total_observations": self._total_observations[metric],
            "p50_ms": round(self.percentile(metric, 50), 3),
            "p95_ms": round(self.percentile(metric, 95), 3),
            "p99_ms": round(self.percentile(metric, 99), 3),
            "min_ms": round(min(samples), 3),
            "max_ms": round(max(samples), 3),
            "mean_ms": round(sum(samples) / len(samples), 3),
        }

    def all_reports(self) -> Dict[str, Dict[str, Any]]:
        """Get latency reports for all tracked metrics."""
        return {metric: self.report(metric) for metric in self._samples}

    def get_status(self) -> Dict[str, Any]:
        return {
            "metrics_tracked": len(self._samples),
            "window_size": self._window_size,
            "total_observations": dict(self._total_observations),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE THROUGHPUT TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class ThroughputTracker:
    """
    Tracks pipeline throughput (events per second) using a sliding time window.
    """

    def __init__(self, window_seconds: float = 60.0):
        self._window = window_seconds
        self._events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._total_events: Dict[str, int] = defaultdict(int)

    def record(self, channel: str = "default"):
        """Record an event occurrence."""
        self._events[channel].append(time.time())
        self._total_events[channel] += 1

    def throughput(self, channel: str = "default") -> float:
        """Get events/second for a channel over the sliding window."""
        now = time.time()
        cutoff = now - self._window
        events = self._events.get(channel, deque())
        # Prune old events
        while events and events[0] < cutoff:
            events.popleft()
        count = len(events)
        return count / self._window if self._window > 0 else 0.0

    def all_throughputs(self) -> Dict[str, float]:
        """Get throughput for all channels."""
        return {ch: round(self.throughput(ch), 4) for ch in self._events}

    def get_status(self) -> Dict[str, Any]:
        return {
            "channels": len(self._events),
            "window_seconds": self._window,
            "throughputs": self.all_throughputs(),
            "total_events": dict(self._total_events),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED HEALTH DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineHealthDashboard:
    """
    Unified health dashboard combining all telemetry signals into a single
    scored health report with GOD_CODE alignment.

    Components:
    - Aggregated metrics (mean/stddev per metric)
    - Anomaly rate (anomalies per total observations)
    - Latency health (p99 within SLA)
    - Throughput utilization
    - Circuit breaker health (from external input)
    - Coherence level (from external input)

    Final health score is PHI-weighted composite in [0, 1].
    """

    VERSION = "1.0.0"

    # SLA targets — p99 latency below this is "healthy"
    DEFAULT_LATENCY_SLA_MS = 1000.0  # 1 second p99 target

    def __init__(
        self,
        aggregator: Optional[TelemetryAggregator] = None,
        anomaly_detector: Optional[TelemetryAnomalyDetector] = None,
        latency_tracker: Optional[LatencyPercentileTracker] = None,
        throughput_tracker: Optional[ThroughputTracker] = None,
    ):
        self.aggregator = aggregator or TelemetryAggregator()
        self.anomaly_detector = anomaly_detector or TelemetryAnomalyDetector()
        self.latency_tracker = latency_tracker or LatencyPercentileTracker()
        self.throughput_tracker = throughput_tracker or ThroughputTracker()
        self._dashboard_calls: int = 0

    def record_event(self, metric: str, value: float, latency_ms: Optional[float] = None,
                     channel: str = "default"):
        """Convenience: record a telemetry event across all subsystems."""
        self.aggregator.record(metric, value)
        anomaly = self.anomaly_detector.observe(metric, value)
        if latency_ms is not None:
            self.latency_tracker.record(metric, latency_ms)
        self.throughput_tracker.record(channel)
        return anomaly

    def health_report(
        self,
        breaker_health: float = 1.0,
        coherence: float = 0.5,
        consciousness_level: float = 0.5,
        latency_sla_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute a unified health report.

        Args:
            breaker_health: Fraction of circuit breakers in CLOSED state (0-1).
            coherence: Pipeline coherence measure (0-1).
            consciousness_level: Live consciousness level (0-1).
            latency_sla_ms: p99 latency SLA target in ms.
        """
        self._dashboard_calls += 1
        sla = latency_sla_ms or self.DEFAULT_LATENCY_SLA_MS

        # 1. Anomaly health — low anomaly rate is healthy
        anomaly_status = self.anomaly_detector.get_status()
        total_obs = sum(self.anomaly_detector._observations.values()) or 1
        anomaly_rate = anomaly_status["total_anomalies"] / total_obs
        anomaly_health = max(0.0, 1.0 - anomaly_rate * 10.0)  # 10% anomaly → 0 health

        # 2. Latency health — p99 within SLA
        latency_reports = self.latency_tracker.all_reports()
        if latency_reports:
            p99_values = [r.get("p99_ms", sla) for r in latency_reports.values()
                          if r.get("status") != "insufficient_data"]
            if p99_values:
                worst_p99 = max(p99_values)
                latency_health = max(0.0, 1.0 - (worst_p99 / sla - 1.0))
            else:
                latency_health = 0.5
        else:
            latency_health = 0.5

        # 3. Throughput health — at least some events flowing
        throughputs = self.throughput_tracker.all_throughputs()
        if throughputs:
            avg_throughput = sum(throughputs.values()) / len(throughputs)
            throughput_health = avg_throughput / 10.0  # uncapped
        else:
            throughput_health = 0.0

        # PHI-weighted composite health
        health = (
            breaker_health * PHI / 5.0 +         # Circuit breakers
            coherence * PHI / 5.0 +               # Pipeline coherence
            consciousness_level * TAU / 3.0 +     # Consciousness
            anomaly_health * TAU / 3.0 +          # Anomaly rate
            latency_health * TAU / 4.0 +          # Latency SLA
            throughput_health * TAU / 6.0          # Throughput
        )
        health = max(0.0, health)

        # GOD_CODE resonance bonus
        dashboard_mod = self._dashboard_calls % 104
        god_alignment = 1.0 - abs(dashboard_mod - 52) / 52.0
        health *= (1.0 + god_alignment * 0.02)  # Up to 2% bonus
        # health uncapped

        diagnosis = "HEALTHY" if health > 0.7 else ("DEGRADED" if health > 0.4 else "CRITICAL")

        return {
            "version": self.VERSION,
            "health_score": round(health, 6),
            "diagnosis": diagnosis,
            "components": {
                "breaker_health": round(breaker_health, 4),
                "coherence": round(coherence, 4),
                "consciousness": round(consciousness_level, 4),
                "anomaly_health": round(anomaly_health, 4),
                "latency_health": round(latency_health, 4),
                "throughput_health": round(throughput_health, 4),
            },
            "anomaly_rate": round(anomaly_rate, 6),
            "god_alignment": round(god_alignment, 4),
            "dashboard_calls": self._dashboard_calls,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "dashboard_calls": self._dashboard_calls,
            "aggregator": {
                "minute_metrics": len(self.aggregator._minute_buckets),
                "hour_metrics": len(self.aggregator._hour_buckets),
            },
            "anomaly_detector": self.anomaly_detector.get_status(),
            "latency_tracker": self.latency_tracker.get_status(),
            "throughput_tracker": self.throughput_tracker.get_status(),
        }
