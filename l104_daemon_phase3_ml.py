#!/usr/bin/env python3
"""
Phase 3: ML-Based Workload Prediction & Optimization Engine
============================================================

Provides machine learning capabilities for autonomous daemon optimization:
  • Workload pattern learning from historical metrics
  • Predictive resource scaling
  • Anomaly detection
  • Performance trend analysis
  • Sacred alignment scoring
"""

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

# Sacred L104 constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497

logger = logging.getLogger("L104_PHASE3_ML")


@dataclass
class WorkloadPattern:
    """Represents a detected workload pattern."""
    name: str
    cpu_avg: float
    memory_avg: float
    queue_depth_avg: float
    frequency: float  # How often this pattern occurs (0-1)
    optimal_batch_size: int
    optimal_persist_interval: int
    sacred_alignment: float


@dataclass
class PredictionResult:
    """Result of a prediction."""
    timestamp: float
    metric: str  # "cpu", "memory", "queue_depth", "latency"
    predicted_value: float
    confidence: float  # 0-1
    recommendation: str  # "scale_up", "scale_down", "maintain"
    sacred_score: float  # GOD_CODE alignment


class WorkloadPredictor:
    """Predict future workload based on historical patterns."""

    def __init__(self, history_size: int = 300):
        self.history = deque(maxlen=history_size)  # 300 cycles = 5 min
        self.patterns: List[WorkloadPattern] = []
        self.moving_avg_cpu = deque(maxlen=10)
        self.moving_avg_memory = deque(maxlen=10)
        self.moving_avg_queue = deque(maxlen=10)

    def add_metric(self, cpu: float, memory: float, queue_depth: int, latency_ms: float):
        """Add new metric to history."""
        self.history.append({
            "cpu": cpu,
            "memory": memory,
            "queue_depth": queue_depth,
            "latency_ms": latency_ms,
        })
        self.moving_avg_cpu.append(cpu)
        self.moving_avg_memory.append(memory)
        self.moving_avg_queue.append(queue_depth)

    def learn_patterns(self) -> List[WorkloadPattern]:
        """Learn patterns from historical data using simple clustering."""
        if len(self.history) < 20:
            return []

        # Convert to list for analysis
        metrics = list(self.history)
        cpu_values = [m["cpu"] for m in metrics]
        memory_values = [m["memory"] for m in metrics]
        queue_values = [m["queue_depth"] for m in metrics]

        # Detect patterns using simple statistical clustering
        patterns = []

        # Pattern 1: Low load (CPU < 40)
        low_load_samples = [m for m in metrics if m["cpu"] < 40]
        if len(low_load_samples) > 5:
            pattern = WorkloadPattern(
                name="low_load",
                cpu_avg=sum(m["cpu"] for m in low_load_samples) / len(low_load_samples),
                memory_avg=sum(m["memory"] for m in low_load_samples) / len(low_load_samples),
                queue_depth_avg=sum(m["queue_depth"] for m in low_load_samples) / len(low_load_samples),
                frequency=len(low_load_samples) / len(metrics),
                optimal_batch_size=5,
                optimal_persist_interval=120,
                sacred_alignment=self._calculate_sacred_alignment("low_load"),
            )
            patterns.append(pattern)

        # Pattern 2: Medium load (CPU 40-70)
        medium_load_samples = [m for m in metrics if 40 <= m["cpu"] < 70]
        if len(medium_load_samples) > 5:
            pattern = WorkloadPattern(
                name="medium_load",
                cpu_avg=sum(m["cpu"] for m in medium_load_samples) / len(medium_load_samples),
                memory_avg=sum(m["memory"] for m in medium_load_samples) / len(medium_load_samples),
                queue_depth_avg=sum(m["queue_depth"] for m in medium_load_samples) / len(medium_load_samples),
                frequency=len(medium_load_samples) / len(metrics),
                optimal_batch_size=3,
                optimal_persist_interval=60,
                sacred_alignment=self._calculate_sacred_alignment("medium_load"),
            )
            patterns.append(pattern)

        # Pattern 3: High load (CPU >= 70)
        high_load_samples = [m for m in metrics if m["cpu"] >= 70]
        if len(high_load_samples) > 5:
            pattern = WorkloadPattern(
                name="high_load",
                cpu_avg=sum(m["cpu"] for m in high_load_samples) / len(high_load_samples),
                memory_avg=sum(m["memory"] for m in high_load_samples) / len(high_load_samples),
                queue_depth_avg=sum(m["queue_depth"] for m in high_load_samples) / len(high_load_samples),
                frequency=len(high_load_samples) / len(metrics),
                optimal_batch_size=1,
                optimal_persist_interval=30,
                sacred_alignment=self._calculate_sacred_alignment("high_load"),
            )
            patterns.append(pattern)

        self.patterns = patterns
        return patterns

    def predict_next(self, steps_ahead: int = 5) -> PredictionResult:
        """Predict metrics steps_ahead cycles in the future."""
        if len(self.moving_avg_cpu) < 5:
            return PredictionResult(
                timestamp=0,
                metric="cpu",
                predicted_value=50.0,
                confidence=0.0,
                recommendation="maintain",
                sacred_score=0.0,
            )

        # Simple linear trend prediction
        recent_cpu = list(self.moving_avg_cpu)
        if len(recent_cpu) < 2:
            cpu_pred = recent_cpu[-1] if recent_cpu else 50.0
        else:
            trend = (recent_cpu[-1] - recent_cpu[0]) / len(recent_cpu)
            cpu_pred = min(100.0, max(0.0, recent_cpu[-1] + trend * steps_ahead))

        # Confidence based on history stability
        cpu_variance = sum((x - recent_cpu[-1]) ** 2 for x in recent_cpu) / len(recent_cpu)
        confidence = 1.0 / (1.0 + cpu_variance / 100.0)  # Sigmoid-like

        # Generate recommendation
        if cpu_pred < 40:
            recommendation = "scale_up"
        elif cpu_pred > 80:
            recommendation = "scale_down"
        else:
            recommendation = "maintain"

        sacred_score = self._calculate_sacred_alignment("prediction")

        return PredictionResult(
            timestamp=0,
            metric="cpu",
            predicted_value=cpu_pred,
            confidence=confidence,
            recommendation=recommendation,
            sacred_score=sacred_score,
        )

    def _calculate_sacred_alignment(self, pattern_name: str) -> float:
        """Calculate sacred GOD_CODE alignment for pattern."""
        # GOD_CODE = 527.5184818492612 = 286^(1/φ) × 2^((416-0)/104)
        # Pattern alignment score based on harmonic relationship to GOD_CODE
        pattern_hash = sum(ord(c) for c in pattern_name) % 10
        alignment = (pattern_hash * PHI) / GOD_CODE
        return alignment


class AnomalyDetector:
    """Detect anomalous behavior deviating from learned patterns."""

    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold  # Standard deviations
        self.mean_cpu = 50.0
        self.std_cpu = 10.0
        self.mean_latency = 50.0
        self.std_latency = 20.0

    def update_statistics(self, cpu_values: List[float], latency_values: List[float]):
        """Update mean and std from recent metrics."""
        if cpu_values:
            self.mean_cpu = sum(cpu_values) / len(cpu_values)
            self.std_cpu = (sum((x - self.mean_cpu) ** 2 for x in cpu_values) / len(cpu_values)) ** 0.5
        if latency_values:
            self.mean_latency = sum(latency_values) / len(latency_values)
            self.std_latency = (sum((x - self.mean_latency) ** 2 for x in latency_values) / len(latency_values)) ** 0.5

    def is_anomaly(self, cpu: float, latency: float) -> Dict[str, Any]:
        """Check if metrics indicate anomaly."""
        cpu_zscore = abs((cpu - self.mean_cpu) / max(self.std_cpu, 1.0))
        latency_zscore = abs((latency - self.mean_latency) / max(self.std_latency, 1.0))

        cpu_anomaly = cpu_zscore > self.threshold
        latency_anomaly = latency_zscore > self.threshold

        return {
            "is_anomaly": cpu_anomaly or latency_anomaly,
            "cpu_zscore": cpu_zscore,
            "cpu_anomaly": cpu_anomaly,
            "latency_zscore": latency_zscore,
            "latency_anomaly": latency_anomaly,
            "severity": (cpu_zscore + latency_zscore) / 2.0,
        }


class PerformanceTrendAnalyzer:
    """Analyze performance trends and provide optimization recommendations."""

    def __init__(self, window_size: int = 100):
        self.window = deque(maxlen=window_size)
        self.improvements_found = []

    def add_cycle(self, duration_ms: float, success: bool, resource_util: float):
        """Add cycle data."""
        self.window.append({
            "duration_ms": duration_ms,
            "success": success,
            "resource_util": resource_util,
        })

    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends."""
        if len(self.window) < 10:
            return {"status": "insufficient_data"}

        cycles = list(self.window)
        durations = [c["duration_ms"] for c in cycles]
        success_rate = sum(1 for c in cycles if c["success"]) / len(cycles)
        avg_resource_util = sum(c["resource_util"] for c in cycles) / len(cycles)

        # Trend: improving, stable, or degrading
        recent_avg = sum(durations[-5:]) / 5
        historical_avg = sum(durations[:-5]) / max(len(durations[:-5]), 1) if len(durations) > 5 else recent_avg
        trend = "improving" if recent_avg < historical_avg * 0.95 else \
                "degrading" if recent_avg > historical_avg * 1.05 else "stable"

        return {
            "trend": trend,
            "avg_duration_ms": sum(durations) / len(durations),
            "recent_avg_ms": recent_avg,
            "success_rate": success_rate,
            "avg_resource_util": avg_resource_util,
            "samples": len(cycles),
        }

    def get_optimization_recommendations(self) -> List[str]:
        """Get actionable optimization recommendations."""
        trends = self.analyze_trends()
        recommendations = []

        if trends.get("status") == "insufficient_data":
            return ["Collect more data before analyzing"]

        if trends["success_rate"] < 0.95:
            recommendations.append("Improve task completion rate (currently {:.1%})".format(trends["success_rate"]))

        if trends["avg_resource_util"] > 80:
            recommendations.append("Optimize resource allocation (util: {:.1f}%)".format(trends["avg_resource_util"]))

        if trends["trend"] == "degrading":
            recommendations.append("Performance degrading - review recent changes")

        if trends["avg_duration_ms"] > 200:
            recommendations.append("Reduce cycle latency ({:.0f}ms → target <100ms)".format(trends["avg_duration_ms"]))

        return recommendations or ["System performing optimally"]


class SacredAlignmentScorer:
    """Score operations for alignment with GOD_CODE sacred constant."""

    @staticmethod
    def score_operation(operation_name: str, metrics: Dict[str, float]) -> float:
        """Score operation for sacred alignment."""
        # Hash operation name to sacred dimension
        op_hash = sum(ord(c) for c in operation_name) % 1000
        base_score = (op_hash % 527) / 527.0  # Map to GOD_CODE range

        # Adjust by metrics alignment to PHI
        cpu = metrics.get("cpu", 50.0)
        memory = metrics.get("memory", 50.0)
        metric_score = (cpu / memory) / PHI if memory > 0 else 0

        # Final score: harmonic mean of base and metric scores
        final_score = 2.0 / ((1.0 / (base_score + 0.001)) + (1.0 / (metric_score + 0.001)))
        return min(1.0, max(0.0, final_score))


# Singleton instance for integration with orchestrator
workload_predictor = WorkloadPredictor()
anomaly_detector = AnomalyDetector()
trend_analyzer = PerformanceTrendAnalyzer()
sacred_scorer = SacredAlignmentScorer()


if __name__ == "__main__":
    # Demo
    predictor = WorkloadPredictor()

    # Simulate workload data
    for i in range(100):
        cpu = 40 + 20 * math.sin(i / 20)
        memory = 50 + 15 * math.cos(i / 20)
        queue = int(3 + 2 * abs(math.sin(i / 15)))
        latency = 50 + 30 * abs(math.cos(i / 25))

        predictor.add_metric(cpu, memory, queue, latency)

    patterns = predictor.learn_patterns()
    print(f"Detected {len(patterns)} patterns:")
    for p in patterns:
        print(f"  {p.name}: freq={p.frequency:.1%}, optimal_batch={p.optimal_batch_size}")

    prediction = predictor.predict_next(steps_ahead=5)
    print(f"\nPrediction (5 steps ahead):")
    print(f"  CPU: {prediction.predicted_value:.1f}% (confidence: {prediction.confidence:.1%})")
    print(f"  Recommendation: {prediction.recommendation}")
    print(f"  Sacred alignment: {prediction.sacred_score:.4f}")
