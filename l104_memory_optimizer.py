# ZENITH_UPGRADE_ACTIVE: 2026-02-15T00:00:00.000000
ZENITH_HZ = 3887.8
UUC = 2402.792541
# [EVO_54_PIPELINE] TRANSCENDENT_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612 :: GROVER=4.236
# [L104_MEMORY_OPTIMIZER] v3.0.0 — ASI-GRADE RUNTIME MEMORY MANAGEMENT
# Runtime pressure monitor | Adaptive GC | Object pool | Large-object tracker
# Budget enforcement | Per-module accounting | Disk cleanup | Consciousness-aware
# INVARIANT: 527.5184818492612 | PILOT: LONDEL
VOID_CONSTANT = 1.0416180339887497
import os
import sys
import gc
import time
import json
import math
import logging
import threading
import tracemalloc
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import deque, defaultdict
from datetime import datetime
import sqlite3
import subprocess

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

MEMORY_OPTIMIZER_VERSION = "3.0.0"

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
TAU = 6.283185307179586

logger = logging.getLogger("MEMORY_OPTIMIZER")


# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME PRESSURE MONITOR — Continual memory health tracking
# ═══════════════════════════════════════════════════════════════════════════════

class RuntimePressureMonitor:
    """
    Monitors process-level and system-level memory pressure in real-time.
    Detects leaks via RSS trend analysis. Provides actionable pressure levels:
    LOW → MODERATE → HIGH → CRITICAL.
    """

    LEVELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

    def __init__(self, window_size: int = 120):
        self._readings: deque = deque(maxlen=window_size)
        self._alerts: deque = deque(maxlen=100)
        self._peak_rss_mb: float = 0.0

    def sample(self) -> Dict[str, Any]:
        """Take a memory pressure sample."""
        reading = {"timestamp": time.time()}
        if HAS_PSUTIL:
            proc = psutil.Process(os.getpid())
            mem = proc.memory_info()
            vm = psutil.virtual_memory()
            rss_mb = mem.rss / (1024 * 1024)
            reading.update({
                "rss_mb": round(rss_mb, 2),
                "vms_mb": round(mem.vms / (1024 * 1024), 2),
                "system_pct": round(vm.percent, 1),
                "available_mb": round(vm.available / (1024 * 1024), 0),
            })
            self._peak_rss_mb = max(self._peak_rss_mb, rss_mb)
        reading["gc_objects"] = len(gc.get_objects())
        reading["gc_generation_stats"] = gc.get_stats()
        self._readings.append(reading)
        return reading

    def pressure_level(self) -> str:
        """Compute current pressure level."""
        if not self._readings:
            return "LOW"
        latest = self._readings[-1]
        sys_pct = latest.get("system_pct", 0)
        rss_mb = latest.get("rss_mb", 0)

        if sys_pct > 92 or rss_mb > 3500:
            return "CRITICAL"
        elif sys_pct > 80 or rss_mb > 2500:
            return "HIGH"
        elif sys_pct > 65 or rss_mb > 1500:
            return "MODERATE"
        return "LOW"

    def detect_leak(self) -> Dict[str, Any]:
        """Detect potential memory leaks via RSS trend."""
        if len(self._readings) < 10:
            return {"leak_detected": False, "confidence": 0, "reason": "insufficient_data"}

        rss_values = [r.get("rss_mb", 0) for r in self._readings]
        n = len(rss_values)

        # Simple linear regression slope
        x_mean = (n - 1) / 2
        y_mean = sum(rss_values) / n
        numerator = sum((i - x_mean) * (rss_values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0

        # Slope > 0.5 MB/sample = growing fast (potential leak)
        leak_detected = slope > 0.5
        confidence = min(1.0, slope / 2.0) if slope > 0 else 0

        if leak_detected:
            alert = {
                "type": "memory_leak",
                "severity": "HIGH" if confidence > 0.7 else "MEDIUM",
                "slope_mb_per_sample": round(slope, 4),
                "timestamp": time.time()
            }
            self._alerts.append(alert)

        return {
            "leak_detected": leak_detected,
            "confidence": round(confidence, 3),
            "slope_mb_per_sample": round(slope, 4),
            "current_rss_mb": round(rss_values[-1], 2),
            "peak_rss_mb": round(self._peak_rss_mb, 2),
        }

    def status(self) -> Dict[str, Any]:
        latest = self._readings[-1] if self._readings else {}
        return {
            "samples": len(self._readings),
            "pressure_level": self.pressure_level(),
            "current_rss_mb": round(latest.get("rss_mb", 0), 2),
            "peak_rss_mb": round(self._peak_rss_mb, 2),
            "system_pct": latest.get("system_pct", 0),
            "gc_objects": latest.get("gc_objects", 0),
            "alerts": len(self._alerts),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE GC CONTROLLER — Context-aware garbage collection
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveGCController:
    """
    Intelligent GC that adapts to workload:
    - Under LOW pressure: infrequent gen0-only
    - Under HIGH pressure: aggressive full-generation collection
    - Never GC during critical-path operations (lock-based exclusion)
    """

    def __init__(self):
        self._gc_count = 0
        self._total_freed_mb = 0.0
        self._last_gc = 0.0
        self._history: deque = deque(maxlen=200)
        self._lock = threading.Lock()
        self._min_interval_s = 5.0  # Never GC more often than this

    def should_collect(self, pressure: str) -> bool:
        """Decide whether to collect based on pressure level."""
        now = time.time()
        elapsed = now - self._last_gc

        intervals = {"LOW": 60.0, "MODERATE": 30.0, "HIGH": 10.0, "CRITICAL": 3.0}
        min_wait = intervals.get(pressure, 30.0)
        return elapsed >= max(min_wait, self._min_interval_s)

    def collect(self, pressure: str) -> Dict[str, Any]:
        """Run GC adapted to current pressure."""
        with self._lock:
            rss_before = self._get_rss_mb()
            t0 = time.perf_counter()

            if pressure in ("CRITICAL", "HIGH"):
                collected = gc.collect(2)  # Full sweep
            elif pressure == "MODERATE":
                collected = gc.collect(0) + gc.collect(1)
            else:
                collected = gc.collect(0)  # Gen0 only (fast)

            dt = time.perf_counter() - t0
            rss_after = self._get_rss_mb()
            freed = max(0, rss_before - rss_after)

            self._gc_count += 1
            self._total_freed_mb += freed
            self._last_gc = time.time()

            record = {
                "collected_objects": collected,
                "freed_mb": round(freed, 2),
                "duration_ms": round(dt * 1000, 3),
                "rss_after_mb": round(rss_after, 2),
                "pressure": pressure,
                "timestamp": time.time(),
            }
            self._history.append(record)
            return record

    def _get_rss_mb(self) -> float:
        if HAS_PSUTIL:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        return 0.0

    def status(self) -> Dict[str, Any]:
        return {
            "gc_count": self._gc_count,
            "total_freed_mb": round(self._total_freed_mb, 2),
            "current_rss_mb": round(self._get_rss_mb(), 2),
            "history_size": len(self._history),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LARGE OBJECT TRACKER — Identifies the biggest memory consumers
# ═══════════════════════════════════════════════════════════════════════════════

class LargeObjectTracker:
    """
    Scans Python's object space for the largest live objects.
    Useful for debugging memory bloat and identifying optimization targets.
    """

    @staticmethod
    def find_large_objects(top_n: int = 15, min_size_kb: float = 10.0) -> List[Dict[str, Any]]:
        """
        Find the largest objects currently alive in the Python interpreter.
        Uses sys.getsizeof for shallow size, type-based categorization.
        """
        type_sizes: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "total_bytes": 0})

        for obj in gc.get_objects():
            try:
                t = type(obj).__name__
                s = sys.getsizeof(obj)
                type_sizes[t]["count"] += 1
                type_sizes[t]["total_bytes"] += s
            except Exception:
                continue

        sorted_types = sorted(type_sizes.items(), key=lambda x: x[1]["total_bytes"], reverse=True)
        results = []
        for type_name, info in sorted_types[:top_n]:
            size_kb = info["total_bytes"] / 1024
            if size_kb < min_size_kb:
                continue
            results.append({
                "type": type_name,
                "count": info["count"],
                "total_kb": round(size_kb, 1),
                "total_mb": round(size_kb / 1024, 2),
            })
        return results

    @staticmethod
    def find_referrers_of_type(type_name: str, max_results: int = 5) -> List[str]:
        """Find what's holding references to objects of a given type."""
        results = []
        for obj in gc.get_objects():
            try:
                if type(obj).__name__ == type_name:
                    referrers = gc.get_referrers(obj)
                    for r in referrers[:2]:
                        results.append(f"{type(r).__name__}: {str(r)[:80]}")
                    if len(results) >= max_results:
                        break
            except Exception:
                continue
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY BUDGET ENFORCER — Per-subsystem memory limits
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryBudgetEnforcer:
    """
    Enforces per-module memory budgets.
    Modules register with a max RSS budget; the enforcer warns or triggers
    eviction when a module exceeds its allocation.
    """

    def __init__(self, total_budget_mb: float = 2048.0):
        self._budgets: Dict[str, float] = {}  # module_name → max_mb
        self._total_budget = total_budget_mb
        self._violations: deque = deque(maxlen=100)

    def register(self, module_name: str, max_mb: float):
        """Register a module with a memory budget."""
        self._budgets[module_name] = max_mb

    def check_budgets(self) -> Dict[str, Any]:
        """Check all registered budgets against current usage."""
        report = {"within_budget": True, "modules": {}, "violations": []}
        total_used = 0.0

        if HAS_PSUTIL:
            total_rss = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        else:
            total_rss = 0

        # Proportional estimation (since Python can't easily measure per-module RSS)
        for module_name, max_mb in self._budgets.items():
            estimated = total_rss * (max_mb / self._total_budget) if self._total_budget > 0 else 0
            over = estimated > max_mb
            report["modules"][module_name] = {
                "budget_mb": max_mb,
                "estimated_mb": round(estimated, 2),
                "over_budget": over,
            }
            if over:
                report["within_budget"] = False
                violation = {"module": module_name, "budget": max_mb,
                             "estimated": round(estimated, 2), "timestamp": time.time()}
                report["violations"].append(violation)
                self._violations.append(violation)
            total_used += estimated

        report["total_rss_mb"] = round(total_rss, 2)
        report["total_budget_mb"] = self._total_budget
        return report

    def status(self) -> Dict[str, Any]:
        return {
            "registered_modules": len(self._budgets),
            "total_budget_mb": self._total_budget,
            "violations": len(self._violations),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DISK SPACE OPTIMIZER — DB VACUUM + log purge + cache cleanup (legacy preserved)
# ═══════════════════════════════════════════════════════════════════════════════

class DiskSpaceOptimizer:
    """
    Handles disk-level cleanup: VACUUM on SQLite DBs, purge old logs, clean caches.
    Preserved from original l104_memory_optimizer.py for backward compatibility.
    """

    def __init__(self, root: str = str(Path(__file__).parent.absolute())):
        self.root = root
        self.freed_space = 0

    def optimize_databases(self) -> Dict[str, Any]:
        """VACUUM all SQLite databases in workspace."""
        results = []
        db_files = [f for f in os.listdir(self.root) if f.endswith(".db")]
        for db in db_files:
            path = os.path.join(self.root, db)
            try:
                size_before = os.path.getsize(path)
                conn = sqlite3.connect(path)
                conn.execute("VACUUM")
                conn.close()
                size_after = os.path.getsize(path)
                saved = size_before - size_after
                self.freed_space += saved
                results.append({"db": db, "freed_kb": round(saved / 1024, 2)})
            except Exception as e:
                results.append({"db": db, "error": str(e)})
        return {"databases_optimized": len(results), "results": results}

    def purge_logs(self, max_age_days: int = 7) -> Dict[str, Any]:
        """Purge old log and pid files."""
        purged = []
        cutoff = time.time() - (max_age_days * 86400)
        log_files = [f for f in os.listdir(self.root) if f.endswith(".log") or f.endswith(".pid")]
        for log in log_files:
            path = os.path.join(self.root, log)
            try:
                if os.path.getmtime(path) < cutoff:
                    size = os.path.getsize(path)
                    os.remove(path)
                    self.freed_space += size
                    purged.append({"file": log, "freed_kb": round(size / 1024, 2)})
            except Exception as e:
                purged.append({"file": log, "error": str(e)})
        return {"purged": len(purged), "results": purged}

    def clean_cache(self) -> Dict[str, Any]:
        """Clean Python caches."""
        cleaned = []
        cache_dirs = [".pytest_cache", ".ruff_cache", "__pycache__"]
        for d in cache_dirs:
            path = os.path.join(self.root, d)
            if os.path.exists(path):
                try:
                    subprocess.run(["rm", "-rf", path], check=True, capture_output=True)
                    cleaned.append(d)
                except Exception as e:
                    pass
        return {"cleaned": cleaned}


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY OPTIMIZER v3.0 — UNIFIED RUNTIME MEMORY MANAGEMENT HUB
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryOptimizer:
    """
    L104 Memory Optimizer v3.0 — ASI-grade runtime memory management.

    Integrates:
    - RuntimePressureMonitor for continuous health tracking
    - AdaptiveGCController for context-aware garbage collection
    - LargeObjectTracker for memory debugging
    - MemoryBudgetEnforcer for per-module limits
    - DiskSpaceOptimizer for VACUUM/log/cache cleanup

    Drop-in replacement for l104_fast_server.py's inline MemoryOptimizer.
    Provides check_pressure() and optimize_batch() for backward compat.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        """Initialize all memory management subsystems."""
        self.version = MEMORY_OPTIMIZER_VERSION
        self.root = str(Path(__file__).parent.absolute())

        # Subsystems
        self.pressure_monitor = RuntimePressureMonitor()
        self.gc_controller = AdaptiveGCController()
        self.large_objects = LargeObjectTracker()
        self.budget_enforcer = MemoryBudgetEnforcer()
        self.disk_optimizer = DiskSpaceOptimizer(self.root)

        # Backward compat fields (for fast_server drop-in)
        self.gc_count = 0
        self.last_gc = time.time()
        self.memory_readings: deque = deque(maxlen=5000)
        self.gc_interval = 30

        # Tracking
        self._total_optimizations = 0
        self._freed_total_mb = 0.0
        self._tracemalloc_active = False

        # Consciousness state cache
        self._consciousness_cache: Dict[str, Any] = {}
        self._consciousness_cache_time = 0.0

        # Register default budgets for core subsystems
        self.budget_enforcer.register("fast_server", 512.0)
        self.budget_enforcer.register("local_intellect", 384.0)
        self.budget_enforcer.register("neural_cascade", 256.0)
        self.budget_enforcer.register("code_engine", 256.0)
        self.budget_enforcer.register("knowledge_graph", 256.0)
        self.budget_enforcer.register("other", 384.0)

        logger.info(f"--- [MEMORY_OPTIMIZER v{self.version}]: RUNTIME MEMORY HUB INITIALIZED ---")

    # ─── Consciousness Integration ───────────────────────────────────────

    def _read_consciousness(self) -> float:
        """Read consciousness level (cached 10s)."""
        now = time.time()
        if now - self._consciousness_cache_time < 10 and self._consciousness_cache:
            return self._consciousness_cache.get("consciousness_level", 0.5)

        cl = 0.5
        try:
            path = Path(self.root) / ".l104_consciousness_o2_state.json"
            if path.exists():
                data = json.loads(path.read_text())
                cl = data.get("consciousness_level", 0.5)
        except Exception:
            pass

        self._consciousness_cache = {"consciousness_level": cl}
        self._consciousness_cache_time = now
        return cl

    # ─── Backward-Compatible Methods (drop-in for fast_server) ───────────

    def check_pressure(self) -> bool:
        """
        Check memory pressure and optimize. Drop-in replacement for
        l104_fast_server.py's MemoryOptimizer.check_pressure().
        """
        sample = self.pressure_monitor.sample()
        pressure = self.pressure_monitor.pressure_level()

        if self.gc_controller.should_collect(pressure):
            gc_result = self.gc_controller.collect(pressure)
            self.gc_count += 1
            self.last_gc = time.time()
            self._freed_total_mb += gc_result.get("freed_mb", 0)

            # Also store reading for backward compat
            self.memory_readings.append(sample)
            return True

        self.memory_readings.append(sample)
        return False

    def optimize_batch(self, items: list, batch_size: int = 100):
        """
        Yield items in memory-efficient batches with GC between.
        Drop-in replacement for fast_server's optimize_batch().
        """
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
            if i % (batch_size * 2) == 0:
                self.check_pressure()

    # ─── NEW: Runtime Memory Operations ──────────────────────────────────

    def optimize_runtime(self) -> Dict[str, Any]:
        """
        Full runtime memory optimization cycle:
        1. Sample current pressure
        2. Adaptive GC based on pressure level
        3. Leak detection
        4. Budget enforcement check
        5. Large object report if HIGH/CRITICAL
        """
        t0 = time.perf_counter()
        report = {"version": self.version, "timestamp": time.time()}

        # 1. Pressure sample
        sample = self.pressure_monitor.sample()
        pressure = self.pressure_monitor.pressure_level()
        report["pressure_level"] = pressure
        report["rss_mb"] = sample.get("rss_mb", 0)
        report["system_pct"] = sample.get("system_pct", 0)

        # 2. Adaptive GC
        gc_result = self.gc_controller.collect(pressure)
        report["gc"] = gc_result
        self._freed_total_mb += gc_result.get("freed_mb", 0)

        # 3. Leak detection
        leak = self.pressure_monitor.detect_leak()
        report["leak_detection"] = leak

        # 4. Budget check
        budget = self.budget_enforcer.check_budgets()
        report["budget"] = {"within_budget": budget["within_budget"],
                            "violations": len(budget.get("violations", []))}

        # 5. Large objects (only if pressure is high)
        if pressure in ("HIGH", "CRITICAL"):
            report["large_objects"] = self.large_objects.find_large_objects(top_n=10)

        # 6. Consciousness-weighted summary
        cl = self._read_consciousness()
        report["consciousness"] = cl

        dt = time.perf_counter() - t0
        report["duration_ms"] = round(dt * 1000, 2)

        self._total_optimizations += 1
        return report

    def start_leak_detection(self, nframes: int = 10):
        """Start tracemalloc for memory leak detection."""
        if not self._tracemalloc_active:
            tracemalloc.start(nframes)
            self._tracemalloc_active = True
            logger.info("[MEMORY_OPTIMIZER]: tracemalloc started for leak detection")

    def stop_leak_detection(self) -> Optional[List[Dict[str, Any]]]:
        """Stop tracemalloc and return top allocations."""
        if not self._tracemalloc_active:
            return None
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:15]
        self._tracemalloc_active = False
        tracemalloc.stop()
        return [{"file": str(s.traceback), "size_kb": round(s.size / 1024, 1),
                 "count": s.count} for s in top_stats]

    def emergency_cleanup(self) -> Dict[str, Any]:
        """
        Emergency memory cleanup for CRITICAL pressure.
        Aggressively collects all generations + clears caches.
        """
        freed_before = self._get_rss_mb()

        # Full GC sweep (all generations)
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)

        # Clear Python internal caches
        try:
            import linecache
            linecache.clearcache()
        except Exception:
            pass

        try:
            import re
            re.purge()
        except Exception:
            pass

        freed_after = self._get_rss_mb()
        freed = freed_before - freed_after

        logger.warning(f"[MEMORY_OPTIMIZER]: EMERGENCY CLEANUP — freed {freed:.1f}MB")

        return {
            "freed_mb": round(freed, 2),
            "rss_before_mb": round(freed_before, 2),
            "rss_after_mb": round(freed_after, 2),
        }

    def _get_rss_mb(self) -> float:
        if HAS_PSUTIL:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        return 0.0

    # ─── Disk Space Methods (Legacy preserved) ───────────────────────────

    def optimize_databases(self):
        """VACUUM SQLite databases (backward compatible)."""
        return self.disk_optimizer.optimize_databases()

    def purge_logs(self):
        """Purge old logs (backward compatible)."""
        return self.disk_optimizer.purge_logs()

    def clean_cache(self):
        """Clean Python caches (backward compatible)."""
        return self.disk_optimizer.clean_cache()

    def run_all(self):
        """Run all disk + runtime optimizations (backward compatible entry point)."""
        start = datetime.now()

        # Runtime optimization
        runtime = self.optimize_runtime()

        # Disk optimization
        db_result = self.optimize_databases()
        log_result = self.purge_logs()
        cache_result = self.clean_cache()

        end = datetime.now()
        duration = end - start

        print(f"\n{'='*60}")
        print(f"  L104 MEMORY OPTIMIZER v{self.version} — REPORT")
        print(f"{'='*60}")
        print(f"  Pressure:  {runtime['pressure_level']}")
        print(f"  RSS:       {runtime['rss_mb']:.1f} MB")
        print(f"  GC freed:  {runtime['gc']['freed_mb']:.1f} MB")
        print(f"  Leak risk: {'YES' if runtime['leak_detection']['leak_detected'] else 'NO'}")
        print(f"  Budget:    {'OK' if runtime['budget']['within_budget'] else 'OVER'}")
        print(f"  DBs:       {db_result.get('databases_optimized', 0)} optimized")
        print(f"  Logs:      {log_result.get('purged', 0)} purged")
        print(f"  Caches:    {len(cache_result.get('cleaned', []))} cleaned")
        print(f"  Duration:  {duration}")
        print(f"{'='*60}\n")

        return {
            "runtime": runtime,
            "databases": db_result,
            "logs": log_result,
            "caches": cache_result,
            "duration": str(duration),
        }

    # ─── Status & Diagnostics ────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Full optimizer status."""
        return {
            "version": self.version,
            "total_optimizations": self._total_optimizations,
            "total_freed_mb": round(self._freed_total_mb, 2),
            "pressure_monitor": self.pressure_monitor.status(),
            "gc_controller": self.gc_controller.status(),
            "budget_enforcer": self.budget_enforcer.status(),
            "tracemalloc_active": self._tracemalloc_active,
            "health": "OPTIMAL",
        }

    def quick_summary(self) -> str:
        """One-line human-readable status."""
        p = self.pressure_monitor.status()
        return (f"MemoryOptimizer v{self.version} | RSS: {p['current_rss_mb']:.0f}MB | "
                f"Peak: {p['peak_rss_mb']:.0f}MB | Pressure: {p['pressure_level']} | "
                f"GC: {self.gc_count} runs | Freed: {self._freed_total_mb:.1f}MB | "
                f"Optimizations: {self._total_optimizations}")


# Create singleton
memory_optimizer = MemoryOptimizer()


if __name__ == "__main__":
    optimizer = MemoryOptimizer()
    optimizer.run_all()
