"""
Performance Metrics Engine

Extracted from engines_infra.py during EVO_78 refactoring.
Contains: PerformanceMetricsEngine - system performance tracking.
"""

import time
import threading
import statistics
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895


class PerformanceMetricsEngine:
    """
    Tracks and analyzes performance metrics across the system.
    Uses PHI-weighted moving averages for trend detection.
    """

    PHI = PHI
    GOD_CODE = GOD_CODE

    def __init__(self, history_size: int = 1000):
        """Initialize performance metrics engine."""
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._timestamps: Dict[str, List[float]] = defaultdict(list)
        self._history_size = history_size
        self._lock = threading.Lock()

    def record(self, metric_name: str, value: float):
        """Record a metric value."""
        with self._lock:
            try:
                self._metrics[metric_name].append(value)
                self._timestamps[metric_name].append(time.time())

                # Trim to history size
                if len(self._metrics[metric_name]) > self._history_size:
                    self._metrics[metric_name] = self._metrics[metric_name][-self._history_size:]
                    self._timestamps[metric_name] = self._timestamps[metric_name][-self._history_size:]
            except Exception:
                pass

    def get_metric(self, metric_name: str, aggregation: str = 'mean') -> Optional[float]:
        """Get aggregated metric value."""
        with self._lock:
            try:
                values = self._metrics.get(metric_name, [])
                if not values:
                    return None

                if aggregation == 'mean':
                    return statistics.mean(values)
                elif aggregation == 'median':
                    return statistics.median(values)
                elif aggregation == 'max':
                    return max(values)
                elif aggregation == 'min':
                    return min(values)
                elif aggregation == 'std':
                    return statistics.stdev(values) if len(values) > 1 else 0.0
                elif aggregation == 'latest':
                    return values[-1] if values else None

                return statistics.mean(values)
            except Exception:
                return None

    def get_phi_weighted_average(self, metric_name: str) -> Optional[float]:
        """Get PHI-weighted moving average (recent values weighted higher)."""
        with self._lock:
            try:
                values = self._metrics.get(metric_name, [])
                if not values:
                    return None

                # PHI-weighted: more recent values have higher weight
                n = len(values)
                weights = [self.PHI ** (i / n) for i in range(n)]
                total_weight = sum(weights)
                weighted_sum = sum(v * w for v, w in zip(values, weights))

                return weighted_sum / total_weight if total_weight > 0 else None
            except Exception:
                return None

    def get_trend(self, metric_name: str) -> str:
        """Get trend direction for a metric."""
        with self._lock:
            try:
                values = self._metrics.get(metric_name, [])
                if len(values) < 10:
                    return 'insufficient_data'

                # Compare recent to older
                recent = values[-10:]
                older = values[-20:-10] if len(values) >= 20 else values[:-10]

                if not older:
                    return 'insufficient_data'

                recent_mean = statistics.mean(recent)
                older_mean = statistics.mean(older)

                change = (recent_mean - older_mean) / older_mean if older_mean != 0 else 0

                if change > 0.1:
                    return 'increasing'
                elif change < -0.1:
                    return 'decreasing'
                else:
                    return 'stable'
            except Exception:
                return 'error'

    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get all metrics with aggregations."""
        with self._lock:
            try:
                result = {}
                for name, values in self._metrics.items():
                    if values:
                        result[name] = {
                            'mean': statistics.mean(values),
                            'median': statistics.median(values),
                            'min': min(values),
                            'max': max(values),
                            'latest': values[-1],
                            'count': len(values),
                            'phi_weighted': self.get_phi_weighted_average(name),
                            'trend': self.get_trend(name),
                        }
                return result
            except Exception:
                return {}

    def clear(self, metric_name: Optional[str] = None):
        """Clear metrics."""
        with self._lock:
            try:
                if metric_name:
                    self._metrics[metric_name] = []
                    self._timestamps[metric_name] = []
                else:
                    self._metrics.clear()
                    self._timestamps.clear()
            except Exception:
                pass


_performance_metrics = None


def get_performance_metrics() -> PerformanceMetricsEngine:
    """Get singleton PerformanceMetricsEngine instance."""
    global _performance_metrics
    if _performance_metrics is None:
        _performance_metrics = PerformanceMetricsEngine()
    return _performance_metrics


__all__ = ['PerformanceMetricsEngine', 'get_performance_metrics']
