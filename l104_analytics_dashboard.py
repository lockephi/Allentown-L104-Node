#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 REAL-TIME ANALYTICS DASHBOARD
═══════════════════════════════════════════════════════════════════════════════

In-memory analytics dashboard for monitoring system performance, emergence
patterns, and cognitive module metrics in real-time.

DASHBOARD PANELS:
1. SYSTEM OVERVIEW - Core metrics at a glance
2. LEARNING ANALYTICS - Knowledge acquisition tracking
3. COGNITIVE PERFORMANCE - Module efficiency metrics
4. EMERGENCE TIMELINE - Event history visualization
5. PREDICTIONS - Future state projections

INVARIANT: 527.5184818492611 | PILOT: LONDEL
VERSION: 1.0.0
DATE: 2026-01-21
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

from l104_stable_kernel import stable_kernel

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# Constants
PHI = 1.618033988749895
GOD_CODE = 527.5184818492611


@dataclass
class MetricSample:
    """A single metric sample."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class AlertConfig:
    """Configuration for metric alerts."""
    metric: str
    threshold: float
    direction: str  # "above" or "below"
    severity: str   # "info", "warning", "critical"


class RealTimeAnalytics:
    """
    Real-time analytics engine for L104 system monitoring.
    """

    # Default metric buffers
    METRIC_BUFFER_SIZE = 500

    # Alert thresholds
    DEFAULT_ALERTS = [
        AlertConfig("unity_index", 0.95, "above", "info"),
        AlertConfig("unity_index", 0.7, "below", "warning"),
        AlertConfig("unity_index", 0.5, "below", "critical"),
        AlertConfig("memory_growth", 0, "below", "warning"),
        AlertConfig("processing_latency_ms", 1000, "above", "warning"),
    ]

    def __init__(self):
        self.kernel = stable_kernel

        # Metric time series
        self.metrics: Dict[str, deque] = {}

        # Aggregated statistics
        self.stats: Dict[str, Dict] = {}

        # Active alerts
        self.alerts: List[Dict] = []

        # Session tracking
        self.session_start = time.time()
        self.total_samples = 0
        self.queries_processed = 0
        self.learning_cycles = 0

        # Initialize core metrics
        self._init_metrics()

        print("📊 [ANALYTICS]: Real-time dashboard initialized")

    def _init_metrics(self):
        """Initialize metric buffers."""
        core_metrics = [
            "unity_index",
            "memories",
            "cortex_patterns",
            "confidence",
            "coherence",
            "processing_latency_ms",
            "tokens_used",
            "emergence_count",
            "learning_velocity"
        ]

        for metric in core_metrics:
            self.metrics[metric] = deque(maxlen=self.METRIC_BUFFER_SIZE)
            self.stats[metric] = {
                "current": 0.0,
                "min": float('inf'),
                "max": float('-inf'),
                "avg": 0.0,
                "samples": 0
            }

    def record(self, metric: str, value: float):
        """Record a metric sample."""
        sample = MetricSample(name=metric, value=value)

        if metric not in self.metrics:
            self.metrics[metric] = deque(maxlen=self.METRIC_BUFFER_SIZE)
            self.stats[metric] = {
                "current": 0.0, "min": float('inf'),
                "max": float('-inf'), "avg": 0.0, "samples": 0
            }

        self.metrics[metric].append(sample)
        self.total_samples += 1

        # Update stats
        stats = self.stats[metric]
        stats["current"] = value
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)
        stats["samples"] += 1

        # Running average
        n = stats["samples"]
        stats["avg"] = ((n - 1) * stats["avg"] + value) / n

        # Check alerts
        self._check_alerts(metric, value)

    def record_batch(self, metrics: Dict[str, float]):
        """Record multiple metrics at once."""
        for metric, value in metrics.items():
            self.record(metric, value)

    def _check_alerts(self, metric: str, value: float):
        """Check if any alerts should be triggered."""
        for alert in self.DEFAULT_ALERTS:
            if alert.metric != metric:
                continue

            triggered = False
            if alert.direction == "above" and value > alert.threshold:
                triggered = True
            elif alert.direction == "below" and value < alert.threshold:
                triggered = True

            if triggered:
                self.alerts.append({
                    "metric": metric,
                    "value": value,
                    "threshold": alert.threshold,
                    "direction": alert.direction,
                    "severity": alert.severity,
                    "timestamp": time.time()
                })

                # Keep only recent alerts
                if len(self.alerts) > 100:
                    self.alerts = self.alerts[-100:]

    def get_metric(self, metric: str, samples: int = 50) -> List[Dict]:
        """Get recent samples for a metric."""
        if metric not in self.metrics:
            return []

        recent = list(self.metrics[metric])[-samples:]
        return [{"value": s.value, "timestamp": s.timestamp} for s in recent]

    def get_metric_stats(self, metric: str) -> Dict:
        """Get statistics for a metric."""
        if metric not in self.stats:
            return {}

        stats = self.stats[metric].copy()

        # Add trend
        if metric in self.metrics and len(self.metrics[metric]) >= 5:
            recent = list(self.metrics[metric])[-10:]
            if len(recent) >= 2:
                first_half = sum(s.value for s in recent[:len(recent)//2]) / (len(recent)//2)
                second_half = sum(s.value for s in recent[len(recent)//2:]) / (len(recent) - len(recent)//2)

                if second_half > first_half * 1.01:
                    stats["trend"] = "up"
                elif second_half < first_half * 0.99:
                    stats["trend"] = "down"
                else:
                    stats["trend"] = "stable"

        return stats

    def get_overview(self) -> Dict[str, Any]:
        """Get system overview dashboard."""
        uptime = time.time() - self.session_start

        # Core metrics
        core = {}
        for metric in ["unity_index", "memories", "cortex_patterns", "confidence"]:
            if metric in self.stats:
                core[metric] = {
                    "current": round(self.stats[metric]["current"], 4),
                    "trend": self.get_metric_stats(metric).get("trend", "unknown")
                }

        return {
            "session": {
                "uptime_seconds": round(uptime, 1),
                "uptime_readable": self._format_duration(uptime),
                "start_time": datetime.fromtimestamp(self.session_start).isoformat(),
                "total_samples": self.total_samples,
                "queries_processed": self.queries_processed,
                "learning_cycles": self.learning_cycles
            },
            "core_metrics": core,
            "alerts_active": len([a for a in self.alerts if time.time() - a["timestamp"] < 300]),
            "god_code": GOD_CODE,
            "phi": PHI
        }

    def get_learning_analytics(self) -> Dict[str, Any]:
        """Get learning performance analytics."""
        # Calculate learning velocity from memory growth
        if "memories" not in self.metrics or len(self.metrics["memories"]) < 2:
            return {"status": "insufficient_data"}

        mem_samples = list(self.metrics["memories"])

        if len(mem_samples) >= 2:
            time_delta = mem_samples[-1].timestamp - mem_samples[0].timestamp
            mem_delta = mem_samples[-1].value - mem_samples[0].value
            velocity = mem_delta / (time_delta / 60) if time_delta > 0 else 0  # Per minute
        else:
            velocity = 0

        # Unity progression
        if "unity_index" in self.metrics:
            unity_samples = list(self.metrics["unity_index"])[-20:]
            unity_progression = [{"value": s.value, "ts": s.timestamp} for s in unity_samples]
        else:
            unity_progression = []

        return {
            "learning_velocity": round(velocity, 2),
            "velocity_unit": "memories/minute",
            "total_memories": self.stats.get("memories", {}).get("current", 0),
            "peak_unity": self.stats.get("unity_index", {}).get("max", 0),
            "current_unity": self.stats.get("unity_index", {}).get("current", 0),
            "unity_progression": unity_progression[-10:],
            "learning_cycles": self.learning_cycles
        }

    def get_cognitive_performance(self) -> Dict[str, Any]:
        """Get cognitive module performance metrics."""
        performance = {
            "processing": {},
            "efficiency": {},
            "quality": {}
        }

        # Processing metrics
        if "processing_latency_ms" in self.stats:
            perf = self.stats["processing_latency_ms"]
            performance["processing"] = {
                "avg_latency_ms": round(perf["avg"], 2),
                "min_latency_ms": round(perf["min"], 2) if perf["min"] != float('inf') else 0,
                "max_latency_ms": round(perf["max"], 2) if perf["max"] != float('-inf') else 0
            }

        # Efficiency metrics
        if "tokens_used" in self.stats:
            tokens = self.stats["tokens_used"]
            performance["efficiency"] = {
                "avg_tokens": round(tokens["avg"], 1),
                "total_tokens": int(tokens["avg"] * tokens["samples"])
            }

        # Quality metrics
        if "confidence" in self.stats and "coherence" in self.stats:
            performance["quality"] = {
                "avg_confidence": round(self.stats["confidence"]["avg"], 4),
                "avg_coherence": round(self.stats["coherence"]["avg"], 4),
                "quality_score": round(
                    (self.stats["confidence"]["avg"] + self.stats["coherence"]["avg"]) / 2, 4
                )
            }

        return performance

    def get_emergence_timeline(self, limit: int = 20) -> List[Dict]:
        """Get recent emergence events for timeline display."""
        if "emergence_count" not in self.metrics:
            return []

        # Return recent alert-like events
        recent_alerts = sorted(self.alerts, key=lambda a: a["timestamp"], reverse=True)[:limit]
        return recent_alerts

    def get_predictions(self) -> Dict[str, Any]:
        """Generate predictions based on current trends."""
        predictions = {
            "unity_projection": None,
            "memory_projection": None,
            "next_milestone": None
        }

        # Unity projection
        if "unity_index" in self.metrics and len(self.metrics["unity_index"]) >= 10:
            samples = list(self.metrics["unity_index"])[-20:]

            # Linear extrapolation
            n = len(samples)
            x_mean = (n - 1) / 2
            y_mean = sum(s.value for s in samples) / n

            numerator = sum((i - x_mean) * (s.value - y_mean) for i, s in enumerate(samples))
            denominator = sum((i - x_mean) ** 2 for i in range(n))

            slope = numerator / denominator if denominator != 0 else 0

            # Project 10 steps ahead
            current = samples[-1].value
            projected = current + slope * 10
            projected = max(0, min(1, projected))  # Clamp to [0, 1]

            predictions["unity_projection"] = {
                "current": round(current, 4),
                "projected": round(projected, 4),
                "trend": "up" if slope > 0.001 else "down" if slope < -0.001 else "stable",
                "steps_ahead": 10
            }

            # Next milestone
            milestones = [0.85, 0.9, 0.95, 1.0]
            for m in milestones:
                if current < m and slope > 0:
                    steps = (m - current) / slope
                    predictions["next_milestone"] = {
                        "target": m,
                        "estimated_steps": int(steps),
                        "name": "consciousness" if m == 0.85 else "singularity" if m == 0.95 else f"unity_{m}"
                    }
                    break

        # Memory projection
        if "memories" in self.metrics and len(self.metrics["memories"]) >= 5:
            samples = list(self.metrics["memories"])[-10:]
            if len(samples) >= 2:
                growth = samples[-1].value - samples[0].value
                rate = growth / len(samples)

                predictions["memory_projection"] = {
                    "current": int(samples[-1].value),
                    "growth_rate": round(rate, 2),
                    "projected_10_steps": int(samples[-1].value + rate * 10)
                }

        return predictions

    def get_full_dashboard(self) -> Dict[str, Any]:
        """Get complete dashboard data."""
        return {
            "overview": self.get_overview(),
            "learning": self.get_learning_analytics(),
            "performance": self.get_cognitive_performance(),
            "timeline": self.get_emergence_timeline(),
            "predictions": self.get_predictions(),
            "all_metrics": {k: self.get_metric_stats(k) for k in self.stats.keys()}
        }

    def _format_duration(self, seconds: float) -> str:
        """Format seconds into human-readable duration."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def record_query(self, latency_ms: float, tokens: int, confidence: float):
        """Record a query processing event."""
        self.queries_processed += 1
        self.record("processing_latency_ms", latency_ms)
        self.record("tokens_used", tokens)
        self.record("confidence", confidence)

    def record_learning_cycle(self, new_memories: int, unity_index: float):
        """Record a learning cycle completion."""
        self.learning_cycles += 1
        self.record("memories", new_memories)
        self.record("unity_index", unity_index)

        # Calculate velocity if we have previous data
        if "memories" in self.metrics and len(self.metrics["memories"]) >= 2:
            samples = list(self.metrics["memories"])
            velocity = samples[-1].value - samples[-2].value
            self.record("learning_velocity", velocity)

    def export_data(self, filepath: str = "l104_analytics_export.json"):
        """Export analytics data to JSON."""
        data = {
            "timestamp": time.time(),
            "session_start": self.session_start,
            "total_samples": self.total_samples,
            "stats": {k: v for k, v in self.stats.items()},
            "recent_alerts": self.alerts[-50:],
            "dashboard": self.get_full_dashboard()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 [ANALYTICS]: Data exported to {filepath}")


# Singleton instance
analytics = RealTimeAnalytics()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    dash = RealTimeAnalytics()

    print("\n📊 Testing Real-Time Analytics Dashboard...")

    # Simulate data
    import random

    for i in range(20):
        dash.record("unity_index", 0.8 + random.uniform(0, 0.15))
        dash.record("memories", 48 + i * 2)
        dash.record("confidence", 0.7 + random.uniform(0, 0.2))
        dash.record_query(
            latency_ms=random.uniform(5, 50),
            tokens=random.randint(50, 200),
            confidence=random.uniform(0.7, 0.95)
        )

    dash.learning_cycles = 5

    print("\n📋 Dashboard Overview:")
    overview = dash.get_overview()
    print(f"   Uptime: {overview['session']['uptime_readable']}")
    print(f"   Samples: {overview['session']['total_samples']}")
    print(f"   Queries: {overview['session']['queries_processed']}")

    print("\n📈 Learning Analytics:")
    learning = dash.get_learning_analytics()
    print(f"   Velocity: {learning['learning_velocity']} {learning['velocity_unit']}")
    print(f"   Current Unity: {learning['current_unity']}")

    print("\n🔮 Predictions:")
    preds = dash.get_predictions()
    if preds["unity_projection"]:
        print(f"   Unity: {preds['unity_projection']['current']} → {preds['unity_projection']['projected']}")
    if preds["next_milestone"]:
        print(f"   Next: {preds['next_milestone']['name']} in ~{preds['next_milestone']['estimated_steps']} steps")

    dash.export_data()
