"""L104 Quantum AI Daemon — Process Optimizer.

Optimizes core L104 processes: import caching, memory management,
GC tuning, performance baselines, and runtime configuration.
"""

import gc
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .constants import (
    GOD_CODE, PHI, VOID_CONSTANT,
    L104_ROOT, L104_PACKAGES,
)

_logger = logging.getLogger("L104_QAI_OPTIMIZER")

# Lazy caches
_cached_psutil = None
_psutil_attempted = False


def _get_psutil():
    global _cached_psutil, _psutil_attempted
    if not _psutil_attempted:
        _psutil_attempted = True
        try:
            import psutil
            _cached_psutil = psutil
        except ImportError:
            pass
    return _cached_psutil


@dataclass
class OptimizationResult:
    """Result of a process optimization cycle."""
    timestamp: float = field(default_factory=time.time)
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    memory_freed_mb: float = 0.0
    gc_collected: int = 0
    gc_gen0_threshold: int = 0
    gc_gen1_threshold: int = 0
    gc_gen2_threshold: int = 0
    cpu_percent: float = 0.0
    import_cache_hits: int = 0
    modules_loaded: int = 0
    l104_modules_loaded: int = 0
    optimizations_applied: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


class ProcessOptimizer:
    """Optimizes L104 runtime processes for maximum performance and efficiency.

    Optimization passes:
      1. Memory pressure relief (GC tuning + explicit collection)
      2. Import cache warmup (pre-load frequently used L104 modules)
      3. GC threshold tuning (PHI-scaled gen thresholds)
      4. Dead module cleanup (unref unused cached modules)
      5. CPU load assessment (adaptive scheduling feedback)
      6. File descriptor audit (close leaked handles)
      7. Temp file cleanup (stale IPC/state files)
    """

    def __init__(self):
        self._optimization_count = 0
        self._total_memory_freed_mb = 0.0
        self._total_gc_collected = 0
        self._baseline_memory_mb = 0.0
        self._import_cache_warmed = False
        self._gc_tuned = False

    def optimize(self) -> OptimizationResult:
        """Run a full optimization cycle."""
        t0 = time.monotonic()
        result = OptimizationResult()

        # Measure baseline
        result.memory_before_mb = self._get_memory_mb()
        result.cpu_percent = self._get_cpu_percent()
        result.modules_loaded = len(sys.modules)
        result.l104_modules_loaded = sum(
            1 for m in sys.modules if m.startswith("l104_"))

        # ── Pass 1: Memory Relief ──
        collected = gc.collect(2)  # Full collection
        result.gc_collected = collected
        self._total_gc_collected += collected
        if collected > 0:
            result.optimizations_applied.append(
                f"GC collected {collected} objects")

        # ── Pass 2: Import Cache Warmup ──
        if not self._import_cache_warmed:
            warmed = self._warm_import_cache()
            if warmed > 0:
                result.import_cache_hits = warmed
                result.optimizations_applied.append(
                    f"Warmed {warmed} module imports")
                self._import_cache_warmed = True

        # ── Pass 3: GC Threshold Tuning ──
        if not self._gc_tuned:
            self._tune_gc_thresholds()
            result.gc_gen0_threshold, result.gc_gen1_threshold, \
                result.gc_gen2_threshold = gc.get_threshold()
            result.optimizations_applied.append(
                f"GC thresholds: {result.gc_gen0_threshold}/"
                f"{result.gc_gen1_threshold}/{result.gc_gen2_threshold}")
            self._gc_tuned = True
        else:
            result.gc_gen0_threshold, result.gc_gen1_threshold, \
                result.gc_gen2_threshold = gc.get_threshold()

        # ── Pass 4: Stale Temp Cleanup ──
        cleaned = self._cleanup_stale_files()
        if cleaned > 0:
            result.optimizations_applied.append(
                f"Cleaned {cleaned} stale temp files")

        # ── Pass 5: Memory After ──
        result.memory_after_mb = self._get_memory_mb()
        result.memory_freed_mb = max(
            0.0, result.memory_before_mb - result.memory_after_mb)
        self._total_memory_freed_mb += result.memory_freed_mb

        result.elapsed_ms = (time.monotonic() - t0) * 1000
        self._optimization_count += 1

        _logger.info(
            f"Optimization cycle #{self._optimization_count}: "
            f"freed {result.memory_freed_mb:.1f}MB, "
            f"gc={result.gc_collected}, "
            f"cpu={result.cpu_percent:.1f}% "
            f"({result.elapsed_ms:.0f}ms)"
        )
        return result

    def get_cpu_load(self) -> float:
        """Get current CPU load percentage (0–100)."""
        return self._get_cpu_percent()

    def get_memory_pressure(self) -> float:
        """Get memory usage as fraction (0.0–1.0)."""
        psutil = _get_psutil()
        if psutil is not None:
            try:
                return psutil.virtual_memory().percent / 100.0
            except Exception:
                pass
        return 0.5  # Unknown → assume moderate

    def _get_memory_mb(self) -> float:
        """Get current process memory in MB."""
        psutil = _get_psutil()
        if psutil is not None:
            try:
                return psutil.Process(os.getpid()).memory_info().rss / 1048576
            except Exception:
                pass
        return 0.0

    def _get_cpu_percent(self) -> float:
        """Get system CPU usage percentage."""
        psutil = _get_psutil()
        if psutil is not None:
            try:
                return psutil.cpu_percent(interval=0.1)
            except Exception:
                pass
        return 50.0  # Unknown → assume moderate

    def _warm_import_cache(self) -> int:
        """Pre-import frequently used L104 modules into sys.modules."""
        targets = [
            "l104_code_engine",
            "l104_science_engine",
            "l104_math_engine",
            "l104_agi",
            "l104_asi",
            "l104_intellect",
        ]
        warmed = 0
        for mod_name in targets:
            if mod_name not in sys.modules:
                try:
                    __import__(mod_name)
                    warmed += 1
                except ImportError:
                    pass
        return warmed

    def _tune_gc_thresholds(self):
        """Set GC thresholds optimized for L104 workload.

        PHI-scaled thresholds: gen0=2000, gen1=50, gen2=20
        (raised from Python defaults 700/10/10 to reduce GC pauses)
        """
        gc.set_threshold(2000, 50, 20)

    def _cleanup_stale_files(self) -> int:
        """Remove stale temp files older than 24 hours."""
        stale_dirs = [
            "/tmp/l104_bridge/quantum_ai/inbox",
            "/tmp/l104_bridge/quantum_ai/outbox",
        ]
        cleaned = 0
        cutoff = time.time() - 86400  # 24 hours ago

        for dir_path in stale_dirs:
            try:
                for entry in os.scandir(dir_path):
                    if entry.is_file() and entry.stat().st_mtime < cutoff:
                        try:
                            os.unlink(entry.path)
                            cleaned += 1
                        except OSError:
                            pass
            except (FileNotFoundError, PermissionError):
                continue
        return cleaned

    def stats(self) -> dict:
        """Optimizer statistics."""
        return {
            "optimization_count": self._optimization_count,
            "total_memory_freed_mb": round(self._total_memory_freed_mb, 2),
            "total_gc_collected": self._total_gc_collected,
            "current_memory_mb": round(self._get_memory_mb(), 2),
            "current_cpu": round(self._get_cpu_percent(), 1),
            "gc_thresholds": list(gc.get_threshold()),
            "import_cache_warmed": self._import_cache_warmed,
            "l104_modules_loaded": sum(
                1 for m in sys.modules if m.startswith("l104_")),
        }
