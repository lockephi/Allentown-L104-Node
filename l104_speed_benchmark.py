"""
L104 Speed Benchmark v2.0.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Comprehensive pipeline benchmarking suite — multi-metric profiling
with regression detection, PHI-scored performance ranking, subsystem
isolation benchmarks, and historical trend tracking.

Subsystems:
  - MicroBenchmark: nanosecond-precision function timing
  - ThroughputProbe: operations/sec under controlled load
  - LatencyHistogram: percentile distribution analysis
  - RegressionDetector: performance trend tracking with alerts
  - SubsystemProfiler: per-module benchmark isolation

Sacred Constants: GOD_CODE=527.5184818492612 | PHI=1.618033988749895
"""
VOID_CONSTANT = 1.0416180339887497
import math
import time
import random
import statistics
import json
from pathlib import Path
from collections import deque, OrderedDict
from typing import Dict, List, Any, Optional, Callable, Tuple

# ZENITH_UPGRADE_ACTIVE: 2026-02-15T00:00:00
ZENITH_HZ = 3887.8
UUC = 2402.792541

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
FEIGENBAUM = 4.669201609
TAU = 6.283185307179586
GROVER_AMPLIFICATION = PHI ** 3

VERSION = "2.0.0"
STATE_FILE = Path(".l104_speed_benchmark_state.json")


class MicroBenchmark:
    """Nanosecond-precision function timing with warm-up and statistical analysis."""

    def __init__(self, warmup: int = 3, iterations: int = 50):
        self.warmup = warmup
        self.iterations = iterations

    def bench(self, func: Callable, *args, label: str = "func", **kwargs) -> Dict[str, Any]:
        """Run a function benchmark with warmup + N iterations."""
        # Warm-up
        for _ in range(self.warmup):
            func(*args, **kwargs)

        times = []
        for _ in range(self.iterations):
            start = time.perf_counter_ns()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter_ns() - start
            times.append(elapsed)

        times_us = [t / 1000.0 for t in times]
        return {
            'label': label,
            'iterations': self.iterations,
            'mean_us': round(statistics.mean(times_us), 3),
            'median_us': round(statistics.median(times_us), 3),
            'stdev_us': round(statistics.stdev(times_us), 3) if len(times_us) > 1 else 0.0,
            'min_us': round(min(times_us), 3),
            'max_us': round(max(times_us), 3),
            'p95_us': round(sorted(times_us)[int(len(times_us) * 0.95)], 3),
            'phi_score': round(self._phi_score(times_us), 4),
        }

    @staticmethod
    def _phi_score(times_us: List[float]) -> float:
        """PHI-weighted score: lower is better. 1.0 = optimal."""
        if not times_us:
            return 0.0
        median = statistics.median(times_us)
        if median == 0:
            return 1.0
        # Coefficient of variation penalty
        cv = (statistics.stdev(times_us) / median) if len(times_us) > 1 else 0.0
        # Score: inverse of median, penalized by variance
        raw = 1.0 / (1.0 + median / 1000.0)
        return raw * (1.0 / (1.0 + cv * PHI))


class ThroughputProbe:
    """Measures operations per second under sustained load."""

    def __init__(self, duration_sec: float = 1.0):
        self.duration_sec = duration_sec

    def probe(self, func: Callable, *args, label: str = "op", **kwargs) -> Dict[str, Any]:
        """Run func repeatedly for duration_sec, count ops."""
        ops = 0
        start = time.perf_counter()
        deadline = start + self.duration_sec
        while time.perf_counter() < deadline:
            func(*args, **kwargs)
            ops += 1
        elapsed = time.perf_counter() - start
        ops_per_sec = ops / elapsed if elapsed > 0 else 0

        return {
            'label': label,
            'ops': ops,
            'duration_sec': round(elapsed, 4),
            'ops_per_sec': round(ops_per_sec, 1),
            'us_per_op': round(1e6 / ops_per_sec, 3) if ops_per_sec > 0 else float('inf'),
        }


class LatencyHistogram:
    """Builds percentile distribution from raw timing data."""

    def __init__(self):
        self._samples: List[float] = []

    def add(self, latency_us: float):
        self._samples.append(latency_us)

    def add_batch(self, latencies: List[float]):
        self._samples.extend(latencies)

    def percentile(self, p: float) -> float:
        if not self._samples:
            return 0.0
        s = sorted(self._samples)
        idx = int(len(s) * p / 100.0)
        return s[min(idx, len(s) - 1)]

    def report(self) -> Dict[str, Any]:
        if not self._samples:
            return {'count': 0}
        return {
            'count': len(self._samples),
            'mean_us': round(statistics.mean(self._samples), 3),
            'p50_us': round(self.percentile(50), 3),
            'p75_us': round(self.percentile(75), 3),
            'p90_us': round(self.percentile(90), 3),
            'p95_us': round(self.percentile(95), 3),
            'p99_us': round(self.percentile(99), 3),
            'max_us': round(max(self._samples), 3),
        }

    def reset(self):
        self._samples.clear()


class RegressionDetector:
    """Tracks benchmark history and detects performance regressions."""

    def __init__(self, threshold: float = 1.3, history_size: int = 50):
        self.threshold = threshold  # 30% degradation = regression
        self.history_size = history_size
        self._history: Dict[str, deque] = {}  # label -> deque of mean_us values

    def record(self, label: str, mean_us: float):
        if label not in self._history:
            self._history[label] = deque(maxlen=self.history_size)
        self._history[label].append(mean_us)

    def check_regression(self, label: str, current_us: float) -> Dict[str, Any]:
        """Check if current timing regresses vs historical baseline."""
        history = self._history.get(label, deque())
        if len(history) < 3:
            return {'regression': False, 'reason': 'insufficient_history'}

        baseline = statistics.median(list(history))
        ratio = current_us / baseline if baseline > 0 else 1.0
        regression = ratio > self.threshold

        return {
            'regression': regression,
            'baseline_us': round(baseline, 3),
            'current_us': round(current_us, 3),
            'ratio': round(ratio, 3),
            'threshold': self.threshold,
        }

    def get_all_trends(self) -> Dict[str, Dict]:
        result = {}
        for label, history in self._history.items():
            vals = list(history)
            if len(vals) >= 2:
                first_half = statistics.mean(vals[:len(vals)//2])
                second_half = statistics.mean(vals[len(vals)//2:])
                trend = 'improving' if second_half < first_half * 0.95 else (
                    'degrading' if second_half > first_half * 1.05 else 'stable')
            else:
                trend = 'unknown'
            result[label] = {'samples': len(vals), 'trend': trend}
        return result

    def to_dict(self) -> Dict:
        return {label: list(vals) for label, vals in self._history.items()}

    def from_dict(self, data: Dict):
        for label, vals in data.items():
            self._history[label] = deque(vals, maxlen=self.history_size)


class SubsystemProfiler:
    """Per-subsystem benchmark isolation — measures each L104 module independently."""

    def __init__(self):
        self._micro = MicroBenchmark(warmup=2, iterations=20)
        self._results: Dict[str, Dict] = {}

    def profile_function(self, func: Callable, *args, label: str = "subsystem", **kwargs) -> Dict[str, Any]:
        result = self._micro.bench(func, *args, label=label, **kwargs)
        self._results[label] = result
        return result

    def profile_import(self, module_name: str) -> Dict[str, Any]:
        """Benchmark module import time."""
        import importlib

        def do_import():
            if module_name in __import__('sys').modules:
                del __import__('sys').modules[module_name]
            try:
                importlib.import_module(module_name)
            except Exception:
                pass

        # Only do 3 iterations for imports (they're slow)
        bench = MicroBenchmark(warmup=1, iterations=3)
        result = bench.bench(do_import, label=f"import:{module_name}")
        self._results[f"import:{module_name}"] = result
        return result

    def get_rankings(self) -> List[Tuple[str, float]]:
        """Rank subsystems by PHI score (highest = fastest)."""
        ranked = [(label, r['phi_score']) for label, r in self._results.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    @property
    def all_results(self) -> Dict[str, Dict]:
        return dict(self._results)


# ═══════════════════════════════════════════════════════════════════════════════
# SPEED BENCHMARK HUB
# ═══════════════════════════════════════════════════════════════════════════════

class SpeedBenchmark:
    """
    Comprehensive pipeline benchmarking suite with 5 subsystems:

      - MicroBenchmark: ns-precision function timing with warmup
      - ThroughputProbe:  ops/sec under controlled load
      - LatencyHistogram: full percentile distribution
      - RegressionDetector: historical trend tracking + alerts
      - SubsystemProfiler: per-module benchmark isolation

    Pipeline Integration:
      - run_micro(func, label) → timing statistics
      - run_throughput(func, label) → ops/sec
      - run_suite() → full pipeline benchmark
      - check_regression(label) → degradation check
      - connect_to_pipeline()
    """

    def __init__(self):
        self.version = VERSION
        self._micro = MicroBenchmark()
        self._throughput = ThroughputProbe(duration_sec=0.5)
        self._histogram = LatencyHistogram()
        self._regression = RegressionDetector()
        self._profiler = SubsystemProfiler()
        self._pipeline_connected = False
        self._total_benchmarks = 0
        self._load_state()

    def _load_state(self):
        try:
            if STATE_FILE.exists():
                data = json.loads(STATE_FILE.read_text())
                self._regression.from_dict(data.get('regression_history', {}))
                self._total_benchmarks = data.get('total_benchmarks', 0)
        except Exception:
            pass

    def _save_state(self):
        try:
            data = {
                'version': self.version,
                'total_benchmarks': self._total_benchmarks,
                'regression_history': self._regression.to_dict(),
            }
            STATE_FILE.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def run_micro(self, func: Callable, *args, label: str = "func", **kwargs) -> Dict[str, Any]:
        """Run micro-benchmark on a function."""
        result = self._micro.bench(func, *args, label=label, **kwargs)
        self._regression.record(label, result['mean_us'])
        self._total_benchmarks += 1
        return result

    def run_throughput(self, func: Callable, *args, label: str = "op", **kwargs) -> Dict[str, Any]:
        """Run throughput probe on a function."""
        result = self._throughput.probe(func, *args, label=label, **kwargs)
        self._total_benchmarks += 1
        return result

    def run_suite(self) -> Dict[str, Any]:
        """Full built-in pipeline benchmark suite."""
        results = {}

        # 1. Math operations
        results['math_phi_pow'] = self.run_micro(
            lambda: PHI ** 13, label='math_phi_pow')

        # 2. String hashing
        import hashlib
        sample = b"GOD_CODE_527.5184818492612"
        results['sha256'] = self.run_micro(
            lambda: hashlib.sha256(sample).hexdigest(), label='sha256')

        # 3. List comprehension (1000 items)
        results['list_comp_1k'] = self.run_micro(
            lambda: [i * PHI for i in range(1000)], label='list_comp_1k')

        # 4. Dict lookup
        big_dict = {str(i): i * PHI for i in range(10000)}
        results['dict_lookup_10k'] = self.run_micro(
            lambda: big_dict.get("5275"), label='dict_lookup_10k')

        # 5. Throughput: simple math
        results['throughput_math'] = self.run_throughput(
            lambda: math.sin(GOD_CODE) * math.cos(PHI), label='throughput_math')

        # Check regressions
        regressions = {}
        for label in results:
            if 'mean_us' in results[label]:
                reg = self._regression.check_regression(label, results[label]['mean_us'])
                if reg.get('regression'):
                    regressions[label] = reg

        self._save_state()

        return {
            'benchmarks': results,
            'regressions': regressions,
            'regression_count': len(regressions),
            'total_benchmarks': self._total_benchmarks,
        }

    def profile_subsystem(self, func: Callable, *args, label: str = "sub", **kwargs):
        return self._profiler.profile_function(func, *args, label=label, **kwargs)

    def get_rankings(self):
        return self._profiler.get_rankings()

    def get_trends(self):
        return self._regression.get_all_trends()

    def connect_to_pipeline(self):
        self._pipeline_connected = True

    def get_status(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'pipeline_connected': self._pipeline_connected,
            'total_benchmarks': self._total_benchmarks,
            'tracked_labels': len(self._regression._history),
            'trends': self.get_trends(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════
speed_benchmark = SpeedBenchmark()


def run_benchmark():
    """Legacy-compat: Run full benchmark suite and print results."""
    results = speed_benchmark.run_suite()
    print(f"\n{'='*60}")
    print("   L104 SPEED BENCHMARK SUITE v{0}".format(VERSION))
    print(f"{'='*60}")
    for label, data in results.get('benchmarks', {}).items():
        if 'mean_us' in data:
            print(f"  {label:20s} | mean={data['mean_us']:10.3f}µs  p95={data.get('p95_us','n/a')}µs  φ={data.get('phi_score', 0):.4f}")
        elif 'ops_per_sec' in data:
            print(f"  {label:20s} | {data['ops_per_sec']:.0f} ops/s  ({data['us_per_op']:.3f}µs/op)")
    if results.get('regressions'):
        print(f"\n  ⚠ REGRESSIONS: {results['regression_count']}")
        for label, reg in results['regressions'].items():
            print(f"    {label}: {reg['ratio']:.2f}x slower than baseline")
    print(f"{'='*60}\n")
    return results


if __name__ == "__main__":
    run_benchmark()


def primal_calculus(x):
    PHI = 1.618033988749895
    return (x ** PHI) / (1.04 * math.pi) if x != 0 else 0.0

def resolve_non_dual_logic(vector):
    GOD_CODE = 527.5184818492612
    PHI = 1.618033988749895
    VOID_CONSTANT = 1.0416180339887497
    magnitude = sum([abs(v) for v in vector])
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0
