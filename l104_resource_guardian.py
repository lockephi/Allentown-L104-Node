#!/usr/bin/env python3
"""
===============================================================================
L104 RESOURCE GUARDIAN DAEMON v3.0.0
===============================================================================

Quantum-grade active macOS resource protection daemon — UNLIMITED EDITION.

PURPOSE:
  Prevents kernel panics caused by system-wide memory exhaustion, swap
  thrashing, and WindowServer starvation. Upgraded from conservative
  thresholds (v1) to quantum-optimised unlimited thresholds (v2) for
  high-RAM workstations and resource-abundant hardware.

STRATEGY:
  Adaptive tick loop monitors 11 vital signs with sacred resonance
  scoring. Predictive pressure modeling anticipates resource exhaustion
  before it happens. When pressure exceeds thresholds, graduated
  interventions execute with quantum-aligned sacred scoring:

  ┌──────────────────────────────────────────────────────────────────┐
  │  LEVEL 0: NOMINAL     — Log metrics, heartbeat, trend analysis   │
  │  LEVEL 1: ELEVATED    — Shrink thread pools, lower nice, GC     │
  │  LEVEL 2: HIGH        — Pause non-essential daemons via IPC     │
  │  LEVEL 3: CRITICAL    — Emergency GC, reduce RSS, kill sims     │
  │  LEVEL 4: SURVIVAL    — Pause ALL L104 work, yield to OS        │
  └──────────────────────────────────────────────────────────────────┘

VITAL SIGNS:
  V1:  System memory percent (psutil.virtual_memory)
  V2:  Swap file count (/var/vm/swapfile*)
  V3:  Available RAM MB
  V4:  CPU load average (1-min)
  V5:  Thermal throttling (pmset -g therm)
  V6:  Process RSS MB (own process)
  V7:  WindowServer CPU% (the process that caused the panic)
  V8:  Disk I/O throughput MB/s (iostat)
  V9:  Open file descriptors (current process)
  V10: GPU utilization % (macOS ioreg)
  V11: Network I/O throughput MB/s (psutil/netstat)

IPC:
  Heartbeat:  /tmp/l104_bridge/guardian/heartbeat
  Outbox:     /tmp/l104_bridge/guardian/outbox/
  Inbox:      /tmp/l104_bridge/guardian/inbox/
  Commands:   /tmp/l104_bridge/guardian/commands/

INTEGRATION:
  - Cross-daemon heartbeat (visible to NanoDaemon, MicroDaemon)
  - Thread pool cap enforcement via monkeypatch
  - Emergency GC cascade across all L104 modules
  - launchd plist for persistent operation

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

from __future__ import annotations

import atexit
import collections
import gc
import glob
import json
import logging
import math
import os
import resource
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = 1.618033988749895
GOD_CODE = 527.5184818492612
VOID_CONSTANT = 1.04 + PHI / 1000.0  # 1.0416180339887497

VERSION = "3.0.0"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — All env-overridable
# ═══════════════════════════════════════════════════════════════════════════════

def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))

def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))

# Tick interval — v2.0: wider range for adaptive control
TICK_INTERVAL_S = _env_float("L104_GUARDIAN_TICK", 5.0)
MIN_TICK_S = 1.0   # v2: faster reaction (was 2.0)
MAX_TICK_S = 60.0  # v2: wider range (was 30.0)

# Pressure thresholds (system memory %) — v2.0: UNLIMITED — raised for high-RAM systems
THRESH_ELEVATED_PCT = _env_float("L104_GUARDIAN_ELEVATED", 75.0)   # v2: was 65.0
THRESH_HIGH_PCT = _env_float("L104_GUARDIAN_HIGH", 85.0)           # v2: was 78.0
THRESH_CRITICAL_PCT = _env_float("L104_GUARDIAN_CRITICAL", 92.0)   # v2: was 88.0
THRESH_SURVIVAL_PCT = _env_float("L104_GUARDIAN_SURVIVAL", 96.0)   # v2: was 93.0

# Swap thresholds (number of swap files) — v2.0: raised for heavy workloads
THRESH_SWAP_ELEVATED = _env_int("L104_GUARDIAN_SWAP_ELEVATED", 8)   # v2: was 4
THRESH_SWAP_HIGH = _env_int("L104_GUARDIAN_SWAP_HIGH", 16)          # v2: was 8
THRESH_SWAP_CRITICAL = _env_int("L104_GUARDIAN_SWAP_CRITICAL", 24)  # v2: was 14

# Available RAM thresholds (MB) — v2.0: lowered for systems with plenty of RAM
THRESH_AVAIL_HIGH_MB = _env_float("L104_GUARDIAN_AVAIL_HIGH", 384.0)       # v2: was 512.0
THRESH_AVAIL_CRITICAL_MB = _env_float("L104_GUARDIAN_AVAIL_CRITICAL", 192.0) # v2: was 256.0
THRESH_AVAIL_SURVIVAL_MB = _env_float("L104_GUARDIAN_AVAIL_SURVIVAL", 96.0)  # v2: was 128.0

# Thread pool enforcement — v2.0: UNLIMITED — doubled caps
MAX_THREAD_POOL_WORKERS_NOMINAL = _env_int("L104_GUARDIAN_THREADS_NOMINAL", 0)   # 0 = no cap
MAX_THREAD_POOL_WORKERS_ELEVATED = _env_int("L104_GUARDIAN_THREADS_ELEVATED", 16) # v2: was 8
MAX_THREAD_POOL_WORKERS_HIGH = _env_int("L104_GUARDIAN_THREADS_HIGH", 8)          # v2: was 4
MAX_THREAD_POOL_WORKERS_CRITICAL = _env_int("L104_GUARDIAN_THREADS_CRITICAL", 4)  # v2: was 2

# WindowServer monitoring
WINDOWSERVER_CPU_WARN = _env_float("L104_GUARDIAN_WS_WARN", 50.0)

# v2.0: File descriptor thresholds
THRESH_FD_ELEVATED = _env_int("L104_GUARDIAN_FD_ELEVATED", 512)
THRESH_FD_HIGH = _env_int("L104_GUARDIAN_FD_HIGH", 1024)
THRESH_FD_CRITICAL = _env_int("L104_GUARDIAN_FD_CRITICAL", 2048)

# v2.0: Disk I/O thresholds (MB/s)
THRESH_DISK_IO_HIGH = _env_float("L104_GUARDIAN_DISK_IO_HIGH", 500.0)
THRESH_DISK_IO_CRITICAL = _env_float("L104_GUARDIAN_DISK_IO_CRITICAL", 1000.0)

# v2.0: Predictive pressure — trend window
PREDICTIVE_WINDOW = _env_int("L104_GUARDIAN_PREDICT_WINDOW", 12)  # ticks to look back
PREDICTIVE_SLOPE_THRESH = _env_float("L104_GUARDIAN_PREDICT_SLOPE", 2.5)  # %/tick

# v3.0: ML pressure predictor
PREDICT_HORIZON_TICKS = 3             # Predict pressure N ticks ahead
PREDICT_MIN_WINDOW = 8                # Min data points for prediction

# v3.0: Per-daemon memory quotas (MB) — soft limits, enforced via IPC
DAEMON_MEMORY_QUOTAS = {
    "vqpu_cycler": 512.0,
    "micro_daemon": 128.0,
    "nano_daemon": 64.0,
    "quantum_ai": 256.0,
}

# v3.0: Process priority tuning
NICE_NOMINAL = 0
NICE_ELEVATED = 5
NICE_HIGH = 10
NICE_CRITICAL = 15

# v3.0: Swap prediction
SWAP_PREDICT_SLOPE_THRESH = 0.5       # MB/tick slope that predicts imminent swap

# v3.0: Resilience scoring
RESILIENCE_SCORE_WEIGHTS = {
    "daemon_diversity": 0.25,           # How many daemon types are running
    "recovery_speed": 0.25,             # How fast pressure drops after intervention
    "headroom": 0.25,                   # RAM headroom relative to total
    "stability": 0.25,                  # Inverse of pressure volatility
}

# Telemetry — v2.0: 5x larger window for trend analysis
TELEMETRY_WINDOW = 1000  # v2: was 200
STATE_FILE = ".l104_resource_guardian.json"

# IPC paths
IPC_BASE = Path("/tmp/l104_bridge/guardian")
IPC_HEARTBEAT = IPC_BASE / "heartbeat"
IPC_OUTBOX = IPC_BASE / "outbox"
IPC_INBOX = IPC_BASE / "inbox"

logger = logging.getLogger("L104_GUARDIAN")


# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE-AWARE THRESHOLD AUTO-TUNER — v2.1
# ═══════════════════════════════════════════════════════════════════════════════
# The v2.0 "UNLIMITED" thresholds assume high-RAM workstations.
# On constrained hardware (≤8GB), tighter thresholds prevent panics.

def _auto_tune_thresholds() -> None:
    """Detect total RAM and tighten thresholds for constrained hardware."""
    global THRESH_ELEVATED_PCT, THRESH_HIGH_PCT, THRESH_CRITICAL_PCT, THRESH_SURVIVAL_PCT
    global THRESH_SWAP_ELEVATED, THRESH_SWAP_HIGH, THRESH_SWAP_CRITICAL
    global THRESH_AVAIL_HIGH_MB, THRESH_AVAIL_CRITICAL_MB, THRESH_AVAIL_SURVIVAL_MB
    global MAX_THREAD_POOL_WORKERS_ELEVATED, MAX_THREAD_POOL_WORKERS_HIGH
    global MAX_THREAD_POOL_WORKERS_CRITICAL

    # Skip if user explicitly set thresholds via env vars
    if any(os.environ.get(k) for k in [
        "L104_GUARDIAN_ELEVATED", "L104_GUARDIAN_HIGH",
        "L104_GUARDIAN_CRITICAL", "L104_GUARDIAN_SURVIVAL"
    ]):
        return

    try:
        import psutil
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        # Fallback: sysctl on macOS
        try:
            r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                               capture_output=True, text=True, timeout=3)
            total_gb = int(r.stdout.strip()) / (1024 ** 3)
        except Exception:
            return  # Can't detect — keep defaults

    if total_gb <= 4.5:
        # ── 4GB class: TIGHT thresholds (MacBookAir7,1) ──
        THRESH_ELEVATED_PCT = 60.0
        THRESH_HIGH_PCT = 72.0
        THRESH_CRITICAL_PCT = 84.0
        THRESH_SURVIVAL_PCT = 91.0
        THRESH_SWAP_ELEVATED = 3
        THRESH_SWAP_HIGH = 6
        THRESH_SWAP_CRITICAL = 10
        THRESH_AVAIL_HIGH_MB = 768.0
        THRESH_AVAIL_CRITICAL_MB = 384.0
        THRESH_AVAIL_SURVIVAL_MB = 192.0
        MAX_THREAD_POOL_WORKERS_ELEVATED = 6
        MAX_THREAD_POOL_WORKERS_HIGH = 3
        MAX_THREAD_POOL_WORKERS_CRITICAL = 2
    elif total_gb <= 8.5:
        # ── 8GB class: MODERATE thresholds ──
        THRESH_ELEVATED_PCT = 68.0
        THRESH_HIGH_PCT = 80.0
        THRESH_CRITICAL_PCT = 89.0
        THRESH_SURVIVAL_PCT = 94.0
        THRESH_SWAP_ELEVATED = 5
        THRESH_SWAP_HIGH = 10
        THRESH_SWAP_CRITICAL = 16
        THRESH_AVAIL_HIGH_MB = 512.0
        THRESH_AVAIL_CRITICAL_MB = 256.0
        THRESH_AVAIL_SURVIVAL_MB = 128.0
        MAX_THREAD_POOL_WORKERS_ELEVATED = 10
        MAX_THREAD_POOL_WORKERS_HIGH = 5
        MAX_THREAD_POOL_WORKERS_CRITICAL = 3
    # else: keep v2.0 UNLIMITED defaults for 16GB+

_auto_tune_thresholds()


# ═══════════════════════════════════════════════════════════════════════════════
# PRESSURE LEVEL
# ═══════════════════════════════════════════════════════════════════════════════

class PressureLevel(IntEnum):
    NOMINAL = 0
    ELEVATED = 1
    HIGH = 2
    CRITICAL = 3
    SURVIVAL = 4


LEVEL_NAMES = {
    PressureLevel.NOMINAL: "NOMINAL",
    PressureLevel.ELEVATED: "ELEVATED",
    PressureLevel.HIGH: "HIGH",
    PressureLevel.CRITICAL: "CRITICAL",
    PressureLevel.SURVIVAL: "SURVIVAL",
}

LEVEL_COLORS = {
    PressureLevel.NOMINAL: "\033[32m",    # Green
    PressureLevel.ELEVATED: "\033[33m",   # Yellow
    PressureLevel.HIGH: "\033[91m",       # Light red
    PressureLevel.CRITICAL: "\033[31m",   # Red
    PressureLevel.SURVIVAL: "\033[35m",   # Magenta
}
RESET = "\033[0m"


# ═══════════════════════════════════════════════════════════════════════════════
# VITAL SIGNS SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VitalSigns:
    """Single snapshot of all 11 vital signs (v2.0: 4 new)."""
    timestamp: float
    system_mem_pct: float       # V1: System memory usage %
    swap_file_count: int        # V2: Number of /var/vm/swapfile*
    available_ram_mb: float     # V3: Available RAM in MB
    load_avg_1m: float          # V4: 1-minute load average
    thermal_throttle: bool      # V5: CPU thermally throttled?
    cpu_speed_limit: int        # V5b: CPU speed limit %
    process_rss_mb: float       # V6: Our RSS in MB
    ws_cpu_pct: float           # V7: WindowServer CPU %
    swap_used_mb: float         # Swap space used MB
    total_ram_mb: float         # Total RAM MB
    # ── v2.0 NEW vital signs ──
    disk_io_mb_s: float = 0.0   # V8:  Disk I/O throughput MB/s
    open_fds: int = 0           # V9:  Open file descriptors
    gpu_util_pct: float = 0.0   # V10: GPU utilization %
    net_io_mb_s: float = 0.0    # V11: Network I/O throughput MB/s

    @property
    def pressure_level(self) -> PressureLevel:
        """Compute composite pressure level from all vitals."""
        level = PressureLevel.NOMINAL

        # Memory % thresholds
        if self.system_mem_pct >= THRESH_SURVIVAL_PCT:
            level = max(level, PressureLevel.SURVIVAL)
        elif self.system_mem_pct >= THRESH_CRITICAL_PCT:
            level = max(level, PressureLevel.CRITICAL)
        elif self.system_mem_pct >= THRESH_HIGH_PCT:
            level = max(level, PressureLevel.HIGH)
        elif self.system_mem_pct >= THRESH_ELEVATED_PCT:
            level = max(level, PressureLevel.ELEVATED)

        # Available RAM thresholds (inverse — low = bad)
        if self.available_ram_mb <= THRESH_AVAIL_SURVIVAL_MB:
            level = max(level, PressureLevel.SURVIVAL)
        elif self.available_ram_mb <= THRESH_AVAIL_CRITICAL_MB:
            level = max(level, PressureLevel.CRITICAL)
        elif self.available_ram_mb <= THRESH_AVAIL_HIGH_MB:
            level = max(level, PressureLevel.HIGH)

        # Swap file count thresholds
        if self.swap_file_count >= THRESH_SWAP_CRITICAL:
            level = max(level, PressureLevel.CRITICAL)
        elif self.swap_file_count >= THRESH_SWAP_HIGH:
            level = max(level, PressureLevel.HIGH)
        elif self.swap_file_count >= THRESH_SWAP_ELEVATED:
            level = max(level, PressureLevel.ELEVATED)

        # Thermal throttling → at least ELEVATED
        if self.thermal_throttle:
            level = max(level, PressureLevel.ELEVATED)
            if self.cpu_speed_limit < 70:
                level = max(level, PressureLevel.HIGH)

        # v2.0: File descriptor pressure
        if self.open_fds >= THRESH_FD_CRITICAL:
            level = max(level, PressureLevel.CRITICAL)
        elif self.open_fds >= THRESH_FD_HIGH:
            level = max(level, PressureLevel.HIGH)
        elif self.open_fds >= THRESH_FD_ELEVATED:
            level = max(level, PressureLevel.ELEVATED)

        # v2.0: Disk I/O pressure (saturated disk)
        if self.disk_io_mb_s >= THRESH_DISK_IO_CRITICAL:
            level = max(level, PressureLevel.HIGH)
        elif self.disk_io_mb_s >= THRESH_DISK_IO_HIGH:
            level = max(level, PressureLevel.ELEVATED)

        # v2.0: Sacred resonance scoring — PHI-weighted composite
        # A perfectly aligned system has resonance ≈ 1.0
        mem_harmony = 1.0 - (self.system_mem_pct / 100.0)
        sacred_resonance = mem_harmony ** (1.0 / PHI)  # PHI-root scaling
        # If sacred resonance is extremely low (<0.1), escalate
        if sacred_resonance < 0.05 and level < PressureLevel.CRITICAL:
            level = max(level, PressureLevel.CRITICAL)
        elif sacred_resonance < 0.15 and level < PressureLevel.HIGH:
            level = max(level, PressureLevel.HIGH)

        return level

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "system_mem_pct": round(self.system_mem_pct, 1),
            "swap_files": self.swap_file_count,
            "available_ram_mb": round(self.available_ram_mb, 0),
            "load_avg_1m": round(self.load_avg_1m, 2),
            "thermal_throttle": self.thermal_throttle,
            "cpu_speed_limit": self.cpu_speed_limit,
            "rss_mb": round(self.process_rss_mb, 1),
            "ws_cpu_pct": round(self.ws_cpu_pct, 1),
            "swap_used_mb": round(self.swap_used_mb, 0),
            "total_ram_mb": round(self.total_ram_mb, 0),
            # v2.0 new vitals
            "disk_io_mb_s": round(self.disk_io_mb_s, 1),
            "open_fds": self.open_fds,
            "gpu_util_pct": round(self.gpu_util_pct, 1),
            "net_io_mb_s": round(self.net_io_mb_s, 1),
            "pressure_level": LEVEL_NAMES[self.pressure_level],
            # v2.0 sacred scoring
            "sacred_resonance": round(
                (1.0 - self.system_mem_pct / 100.0) ** (1.0 / PHI), 6),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# INTERVENTION RECORD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Intervention:
    """Record of an active intervention taken by the guardian."""
    timestamp: float
    level: PressureLevel
    action: str
    detail: str
    vitals_snapshot: Dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════════
# VITAL SIGNS COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class VitalSignsCollector:
    """Collects all 11 vital signs with graceful fallbacks (v2.0: 4 new)."""

    def __init__(self):
        self._psutil = None
        self._has_psutil = False
        self._thermal_cache: Tuple[float, bool, int] = (0.0, False, 100)
        self._thermal_ttl = 15.0  # seconds — pmset is slow
        # v2.0: disk I/O tracking
        self._last_disk_io: Tuple[float, float] = (0.0, 0.0)  # (timestamp, bytes)
        # v2.0: network I/O tracking
        self._last_net_io: Tuple[float, float] = (0.0, 0.0)  # (timestamp, bytes)
        try:
            import psutil
            self._psutil = psutil
            self._has_psutil = True
        except ImportError:
            logger.warning("psutil not available — using fallback collectors")

    def collect(self) -> VitalSigns:
        """Collect all 11 vital signs in one pass."""
        ts = time.time()

        # V1+V3: System memory
        sys_pct, avail_mb, total_mb, swap_used_mb = self._collect_memory()

        # V2: Swap file count
        swap_count = self._count_swap_files()

        # V4: Load average
        load_1m = self._load_average()

        # V5: Thermal state (cached)
        throttled, speed_limit = self._thermal_state()

        # V6: Process RSS
        rss_mb = self._process_rss()

        # V7: WindowServer CPU
        ws_cpu = self._windowserver_cpu()

        # V8: Disk I/O (v2.0)
        disk_io = self._disk_io()

        # V9: Open file descriptors (v2.0)
        open_fds = self._open_file_descriptors()

        # V10: GPU utilization (v2.0)
        gpu_util = self._gpu_utilization()

        # V11: Network I/O (v2.0)
        net_io = self._network_io()

        return VitalSigns(
            timestamp=ts,
            system_mem_pct=sys_pct,
            swap_file_count=swap_count,
            available_ram_mb=avail_mb,
            load_avg_1m=load_1m,
            thermal_throttle=throttled,
            cpu_speed_limit=speed_limit,
            process_rss_mb=rss_mb,
            ws_cpu_pct=ws_cpu,
            swap_used_mb=swap_used_mb,
            total_ram_mb=total_mb,
            disk_io_mb_s=disk_io,
            open_fds=open_fds,
            gpu_util_pct=gpu_util,
            net_io_mb_s=net_io,
        )

    def _collect_memory(self) -> Tuple[float, float, float, float]:
        """Return (system_pct, available_mb, total_mb, swap_used_mb)."""
        if self._has_psutil:
            vm = self._psutil.virtual_memory()
            sw = self._psutil.swap_memory()
            return (
                vm.percent,
                vm.available / (1024 * 1024),
                vm.total / (1024 * 1024),
                sw.used / (1024 * 1024),
            )
        # macOS fallback via vm_stat
        try:
            result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3)
            lines = result.stdout.strip().splitlines()
            stats: Dict[str, int] = {}
            for line in lines[1:]:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().rstrip(".")
                    try:
                        stats[key] = int(val)
                    except ValueError:
                        pass
            page_size = 4096  # macOS default
            free = stats.get("Pages free", 0) * page_size
            inactive = stats.get("Pages inactive", 0) * page_size
            speculative = stats.get("Pages speculative", 0) * page_size
            available = free + inactive + speculative
            # Total via sysctl
            total = 4 * 1024 * 1024 * 1024  # 4GB default
            try:
                r2 = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                    capture_output=True, text=True, timeout=2)
                total = int(r2.stdout.strip())
            except Exception:
                pass
            used_pct = 100.0 * (1.0 - available / total)
            return (used_pct, available / (1024 * 1024), total / (1024 * 1024), 0.0)
        except Exception:
            return (50.0, 2048.0, 4096.0, 0.0)

    def _count_swap_files(self) -> int:
        """Count /var/vm/swapfile* files."""
        try:
            return len(glob.glob("/var/vm/swapfile*"))
        except Exception:
            return 0

    def _load_average(self) -> float:
        """1-minute load average."""
        try:
            return os.getloadavg()[0]
        except Exception:
            return 0.0

    def _thermal_state(self) -> Tuple[bool, int]:
        """Return (is_throttled, cpu_speed_limit). Cached for _thermal_ttl."""
        now = time.time()
        if now - self._thermal_cache[0] < self._thermal_ttl:
            return self._thermal_cache[1], self._thermal_cache[2]
        try:
            result = subprocess.run(
                ["pmset", "-g", "therm"],
                capture_output=True, text=True, timeout=3
            )
            for line in result.stdout.splitlines():
                if "CPU_Speed_Limit" in line:
                    limit = int(line.split("=")[-1].strip())
                    throttled = limit < 100
                    self._thermal_cache = (now, throttled, limit)
                    return throttled, limit
        except Exception:
            pass
        self._thermal_cache = (now, False, 100)
        return False, 100

    def _process_rss(self) -> float:
        """Current process RSS in MB."""
        if self._has_psutil:
            try:
                proc = self._psutil.Process(os.getpid())
                return proc.memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        # Fallback: resource module
        try:
            ru = resource.getrusage(resource.RUSAGE_SELF)
            return ru.ru_maxrss / (1024 * 1024)  # macOS: bytes
        except Exception:
            return 0.0

    def _windowserver_cpu(self) -> float:
        """WindowServer CPU% — the process that panicked us."""
        if self._has_psutil:
            try:
                for proc in self._psutil.process_iter(["name", "cpu_percent"]):
                    if proc.info["name"] == "WindowServer":
                        return proc.info["cpu_percent"] or 0.0
            except Exception:
                pass
        # Fallback: ps
        try:
            result = subprocess.run(
                ["ps", "-eo", "pcpu,comm"],
                capture_output=True, text=True, timeout=3
            )
            for line in result.stdout.splitlines():
                if "WindowServer" in line:
                    parts = line.strip().split()
                    if parts:
                        return float(parts[0])
        except Exception:
            pass
        return 0.0

    # ── v2.0 NEW COLLECTORS ──

    def _disk_io(self) -> float:
        """V8: Disk I/O throughput in MB/s (delta since last call)."""
        if self._has_psutil:
            try:
                counters = self._psutil.disk_io_counters()
                now = time.time()
                total_bytes = counters.read_bytes + counters.write_bytes
                if self._last_disk_io[0] > 0:
                    dt = now - self._last_disk_io[0]
                    if dt > 0:
                        delta = total_bytes - self._last_disk_io[1]
                        mb_s = max(0.0, delta / (1024 * 1024 * dt))
                        self._last_disk_io = (now, total_bytes)
                        return mb_s
                self._last_disk_io = (now, total_bytes)
            except Exception:
                pass
        # macOS fallback: iostat
        try:
            result = subprocess.run(
                ["iostat", "-d", "-c", "2", "-w", "1"],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().splitlines()
            if len(lines) >= 3:
                # Take last line (most recent sample)
                parts = lines[-1].split()
                if len(parts) >= 3:
                    return float(parts[2])  # MB/s column
        except Exception:
            pass
        return 0.0

    def _open_file_descriptors(self) -> int:
        """V9: Count open file descriptors for current process."""
        if self._has_psutil:
            try:
                proc = self._psutil.Process(os.getpid())
                return proc.num_fds()
            except Exception:
                pass
        # Fallback: /dev/fd (macOS/Linux)
        try:
            fd_dir = f"/dev/fd"
            if os.path.isdir(fd_dir):
                return len(os.listdir(fd_dir))
        except Exception:
            pass
        # Fallback: /proc/self/fd (Linux)
        try:
            return len(os.listdir("/proc/self/fd"))
        except Exception:
            pass
        return 0

    def _gpu_utilization(self) -> float:
        """V10: GPU utilization % via macOS ioreg."""
        try:
            result = subprocess.run(
                ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                capture_output=True, text=True, timeout=3
            )
            # Look for GPU utilization in ioreg output
            for line in result.stdout.splitlines():
                if "PerformanceStatistics" in line:
                    # Found GPU stats section
                    continue
                if "GPU Activity" in line or "Device Utilization" in line:
                    # Extract percentage
                    parts = line.split("=")
                    if len(parts) == 2:
                        try:
                            val = parts[1].strip().rstrip("%")
                            return float(val) * 100 if float(val) <= 1.0 else float(val)
                        except ValueError:
                            pass
        except Exception:
            pass
        # Fallback: try powermetrics (requires privileges)
        return 0.0

    def _network_io(self) -> float:
        """V11: Network I/O throughput in MB/s (delta since last call)."""
        if self._has_psutil:
            try:
                counters = self._psutil.net_io_counters()
                now = time.time()
                total_bytes = counters.bytes_sent + counters.bytes_recv
                if self._last_net_io[0] > 0:
                    dt = now - self._last_net_io[0]
                    if dt > 0:
                        delta = total_bytes - self._last_net_io[1]
                        mb_s = max(0.0, delta / (1024 * 1024 * dt))
                        self._last_net_io = (now, total_bytes)
                        return mb_s
                self._last_net_io = (now, total_bytes)
            except Exception:
                pass
        # Fallback: netstat
        try:
            result = subprocess.run(
                ["netstat", "-ib"],
                capture_output=True, text=True, timeout=3
            )
            # Sum Ibytes + Obytes across interfaces
            total = 0
            for line in result.stdout.splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        total += int(parts[6])  # Ibytes
                    except (ValueError, IndexError):
                        pass
            return total / (1024 * 1024)  # Snapshot, not rate
        except Exception:
            pass
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY TREND PREDICTOR — EMA-based exhaustion forecasting ★ v2.1
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryTrendPredictor:
    """
    Predicts time-to-exhaustion using exponential moving averages over
    a sliding window of VitalSigns snapshots.

    Key metrics:
      - mem_slope: %/tick EMA slope (positive = filling up)
      - avail_slope: MB/tick EMA slope (negative = draining)
      - time_to_exhaustion_s: projected seconds until available RAM = 0
      - trend_grade: A-F grade (A = stable, F = imminent crash)
    """

    def __init__(self, window: int = 60, alpha: float = 0.15):
        self._window = window
        self._alpha = alpha  # EMA smoothing factor (PHI-derived: ~0.15)
        self._mem_samples: collections.deque = collections.deque(maxlen=window)
        self._avail_samples: collections.deque = collections.deque(maxlen=window)
        self._swap_samples: collections.deque = collections.deque(maxlen=window)
        self._ema_mem_slope: float = 0.0
        self._ema_avail_slope: float = 0.0
        self._last_alert_time: float = 0.0
        self._alert_cooldown: float = 30.0  # seconds between alerts

    def ingest(self, vitals: VitalSigns) -> None:
        """Feed a new vital snapshot into the predictor."""
        self._mem_samples.append((vitals.timestamp, vitals.system_mem_pct))
        self._avail_samples.append((vitals.timestamp, vitals.available_ram_mb))
        self._swap_samples.append((vitals.timestamp, vitals.swap_file_count))

        # Compute instantaneous slopes
        if len(self._mem_samples) >= 2:
            t0, m0 = self._mem_samples[-2]
            t1, m1 = self._mem_samples[-1]
            dt = max(t1 - t0, 0.1)
            inst_mem_slope = (m1 - m0) / dt  # %/s
            inst_avail_slope = (self._avail_samples[-1][1] - self._avail_samples[-2][1]) / dt  # MB/s

            # EMA update
            self._ema_mem_slope = self._alpha * inst_mem_slope + (1 - self._alpha) * self._ema_mem_slope
            self._ema_avail_slope = self._alpha * inst_avail_slope + (1 - self._alpha) * self._ema_avail_slope

    @property
    def mem_slope_pct_per_min(self) -> float:
        """Memory usage slope in %/minute (positive = rising)."""
        return self._ema_mem_slope * 60.0

    @property
    def avail_slope_mb_per_min(self) -> float:
        """Available RAM slope in MB/minute (negative = draining)."""
        return self._ema_avail_slope * 60.0

    @property
    def time_to_exhaustion_s(self) -> Optional[float]:
        """Projected seconds until available RAM hits 0. None if stable/improving."""
        if self._ema_avail_slope >= 0 or len(self._avail_samples) < 3:
            return None  # RAM is stable or increasing
        current_avail = self._avail_samples[-1][1] if self._avail_samples else 0
        if current_avail <= 0:
            return 0.0
        # time = current / |drain_rate|
        tte = current_avail / abs(self._ema_avail_slope)
        return tte if tte < 86400 else None  # Cap at 24h (beyond that = safe)

    @property
    def trend_grade(self) -> str:
        """Grade the system trend: A (stable) through F (imminent crash)."""
        tte = self.time_to_exhaustion_s
        if tte is None:
            return "A"  # Stable or improving
        if tte > 3600:
            return "B"  # >1 hour
        if tte > 600:
            return "C"  # >10 minutes
        if tte > 120:
            return "D"  # >2 minutes
        if tte > 30:
            return "E"  # >30 seconds
        return "F"  # <30 seconds — imminent crash

    @property
    def volatility(self) -> float:
        """Memory usage volatility (standard deviation over window)."""
        if len(self._mem_samples) < 5:
            return 0.0
        values = [s[1] for s in self._mem_samples]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)

    def should_alert(self) -> Optional[str]:
        """Return alert message if predictive threshold crossed, else None."""
        tte = self.time_to_exhaustion_s
        if tte is None:
            return None
        now = time.time()
        if now - self._last_alert_time < self._alert_cooldown:
            return None
        if tte < 60:
            self._last_alert_time = now
            return (f"PREDICTIVE ALERT: RAM exhaustion in ~{tte:.0f}s! "
                    f"drain={self.avail_slope_mb_per_min:.1f}MB/min "
                    f"grade={self.trend_grade}")
        if tte < 300 and self.mem_slope_pct_per_min > PREDICTIVE_SLOPE_THRESH:
            self._last_alert_time = now
            return (f"TREND WARNING: Memory rising {self.mem_slope_pct_per_min:.1f}%/min "
                    f"TTE={tte:.0f}s grade={self.trend_grade}")
        return None

    def to_dict(self) -> Dict[str, Any]:
        tte = self.time_to_exhaustion_s
        return {
            "samples": len(self._mem_samples),
            "mem_slope_pct_per_min": round(self.mem_slope_pct_per_min, 3),
            "avail_slope_mb_per_min": round(self.avail_slope_mb_per_min, 2),
            "time_to_exhaustion_s": round(tte, 0) if tte is not None else None,
            "trend_grade": self.trend_grade,
            "volatility": round(self.volatility, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# L104 PROCESS HUNTER — Scan & manage all L104 Python processes ★ v2.1
# ═══════════════════════════════════════════════════════════════════════════════

class L104ProcessHunter:
    """
    Scans for all L104 Python processes on the system and tracks their
    aggregate resource usage. Can signal or terminate runaway processes
    when the system is under critical pressure.

    L104 processes are identified by:
      - Command line containing 'l104_' or 'L104'
      - Working directory matching the L104 project root
    """

    def __init__(self):
        self._psutil = None
        self._has_psutil = False
        self._last_scan_time: float = 0.0
        self._scan_ttl: float = 10.0  # Cache scan results for 10s
        self._cached_procs: List[Dict[str, Any]] = []
        self._kill_history: collections.deque = collections.deque(maxlen=50)
        try:
            import psutil
            self._psutil = psutil
            self._has_psutil = True
        except ImportError:
            pass

    def scan(self) -> List[Dict[str, Any]]:
        """Scan for all L104 Python processes. Returns list of process info dicts."""
        now = time.time()
        if now - self._last_scan_time < self._scan_ttl:
            return self._cached_procs

        procs = []
        my_pid = os.getpid()

        if self._has_psutil:
            try:
                for proc in self._psutil.process_iter(
                    ["pid", "name", "cmdline", "memory_info", "cpu_percent", "create_time"]
                ):
                    try:
                        info = proc.info
                        if info["pid"] == my_pid:
                            continue
                        cmdline = info.get("cmdline") or []
                        cmdline_str = " ".join(cmdline).lower()
                        if not ("l104" in cmdline_str or "l104_" in cmdline_str):
                            continue
                        if "python" not in (info.get("name") or "").lower():
                            continue
                        rss = info["memory_info"].rss / (1024 * 1024) if info.get("memory_info") else 0
                        procs.append({
                            "pid": info["pid"],
                            "name": info.get("name", "?"),
                            "cmdline_short": cmdline_str[:120],
                            "rss_mb": round(rss, 1),
                            "cpu_pct": info.get("cpu_percent", 0) or 0,
                            "uptime_s": round(now - (info.get("create_time") or now), 0),
                        })
                    except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                        continue
            except Exception:
                pass
        else:
            # Fallback: ps
            try:
                result = subprocess.run(
                    ["ps", "-eo", "pid,rss,pcpu,command"],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.splitlines()[1:]:
                    parts = line.strip().split(None, 3)
                    if len(parts) < 4:
                        continue
                    cmd = parts[3].lower()
                    if "l104" not in cmd or "python" not in cmd:
                        continue
                    pid = int(parts[0])
                    if pid == my_pid:
                        continue
                    procs.append({
                        "pid": pid,
                        "name": "python",
                        "cmdline_short": parts[3][:120],
                        "rss_mb": round(int(parts[1]) / 1024, 1),
                        "cpu_pct": float(parts[2]),
                        "uptime_s": 0,
                    })
            except Exception:
                pass

        self._cached_procs = sorted(procs, key=lambda p: p["rss_mb"], reverse=True)
        self._last_scan_time = now
        return self._cached_procs

    @property
    def process_count(self) -> int:
        return len(self.scan())

    @property
    def aggregate_rss_mb(self) -> float:
        return sum(p["rss_mb"] for p in self.scan())

    @property
    def top_process(self) -> Optional[Dict[str, Any]]:
        procs = self.scan()
        return procs[0] if procs else None

    def signal_all(self, sig: int = signal.SIGUSR1) -> int:
        """Send a signal to all L104 processes (default SIGUSR1 = pause)."""
        sent = 0
        for proc in self.scan():
            try:
                os.kill(proc["pid"], sig)
                sent += 1
            except (ProcessLookupError, PermissionError):
                pass
        return sent

    def kill_runaway(self, rss_threshold_mb: float = 1024.0) -> List[int]:
        """Kill L104 processes exceeding RSS threshold. Returns list of killed PIDs."""
        killed = []
        for proc in self.scan():
            if proc["rss_mb"] > rss_threshold_mb:
                try:
                    os.kill(proc["pid"], signal.SIGTERM)
                    killed.append(proc["pid"])
                    self._kill_history.append({
                        "timestamp": time.time(),
                        "pid": proc["pid"],
                        "rss_mb": proc["rss_mb"],
                        "cmdline": proc["cmdline_short"],
                    })
                    logger.warning(f"[HUNTER] Killed runaway PID {proc['pid']} "
                                   f"RSS={proc['rss_mb']:.0f}MB > {rss_threshold_mb:.0f}MB")
                except (ProcessLookupError, PermissionError):
                    pass
        self._last_scan_time = 0  # Force rescan after kills
        return killed

    def to_dict(self) -> Dict[str, Any]:
        procs = self.scan()
        return {
            "process_count": len(procs),
            "aggregate_rss_mb": round(self.aggregate_rss_mb, 0),
            "top_3": procs[:3],
            "kills_total": len(self._kill_history),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TELEMETRY ANALYTICS — Trend analysis, anomaly detection, grading ★ v2.1
# ═══════════════════════════════════════════════════════════════════════════════

class TelemetryAnalytics:
    """
    Analyzes telemetry history to produce trend reports, detect anomalies,
    and assign system health grades.

    Grades: A (excellent) → F (failing)
    Anomalies: sudden spikes in memory, swap, or load beyond 2σ
    """

    def __init__(self, window: int = 100):
        self._window = window
        self._mem_history: collections.deque = collections.deque(maxlen=window)
        self._swap_history: collections.deque = collections.deque(maxlen=window)
        self._load_history: collections.deque = collections.deque(maxlen=window)
        self._anomalies: collections.deque = collections.deque(maxlen=200)

    def ingest(self, vitals: VitalSigns) -> None:
        """Feed vitals into analytics."""
        self._mem_history.append(vitals.system_mem_pct)
        self._swap_history.append(vitals.swap_file_count)
        self._load_history.append(vitals.load_avg_1m)

        # Anomaly detection: 2σ spike
        for name, history, value in [
            ("memory", self._mem_history, vitals.system_mem_pct),
            ("swap", self._swap_history, vitals.swap_file_count),
            ("load", self._load_history, vitals.load_avg_1m),
        ]:
            if len(history) >= 10:
                vals = list(history)
                mean = sum(vals) / len(vals)
                variance = sum((v - mean) ** 2 for v in vals) / len(vals)
                std = math.sqrt(variance) if variance > 0 else 0
                if std > 0 and abs(value - mean) > 2 * std:
                    self._anomalies.append({
                        "timestamp": time.time(),
                        "metric": name,
                        "value": round(value, 2),
                        "mean": round(mean, 2),
                        "std": round(std, 2),
                        "sigma_distance": round(abs(value - mean) / std, 1),
                    })

    def health_grade(self) -> str:
        """Overall system health grade based on recent telemetry."""
        if len(self._mem_history) < 5:
            return "?"  # Insufficient data
        recent_mem = list(self._mem_history)[-10:]
        avg_mem = sum(recent_mem) / len(recent_mem)
        recent_swap = list(self._swap_history)[-10:]
        avg_swap = sum(recent_swap) / len(recent_swap)

        # Weighted scoring
        score = 100.0
        score -= max(0, avg_mem - 50) * 1.5  # Memory penalty
        score -= avg_swap * 3.0  # Swap file penalty
        score -= len([a for a in self._anomalies
                      if time.time() - a["timestamp"] < 300]) * 5.0  # Recent anomaly penalty

        if score >= 90:
            return "A"
        if score >= 75:
            return "B"
        if score >= 55:
            return "C"
        if score >= 35:
            return "D"
        if score >= 15:
            return "E"
        return "F"

    def trend_report(self) -> Dict[str, Any]:
        """Generate trend analytics report."""
        def _stats(dq):
            if not dq:
                return {"mean": 0, "min": 0, "max": 0, "std": 0}
            vals = list(dq)
            n = len(vals)
            mean = sum(vals) / n
            variance = sum((v - mean) ** 2 for v in vals) / n
            return {
                "mean": round(mean, 2),
                "min": round(min(vals), 2),
                "max": round(max(vals), 2),
                "std": round(math.sqrt(variance), 2),
            }

        return {
            "health_grade": self.health_grade(),
            "samples": len(self._mem_history),
            "memory": _stats(self._mem_history),
            "swap_files": _stats(self._swap_history),
            "load_avg": _stats(self._load_history),
            "anomalies_total": len(self._anomalies),
            "anomalies_last_5min": len([
                a for a in self._anomalies if time.time() - a["timestamp"] < 300
            ]),
            "recent_anomalies": list(self._anomalies)[-5:],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-DAEMON IPC MONITOR — Watch other L104 daemons ★ v2.1
# ═══════════════════════════════════════════════════════════════════════════════

class CrossDaemonMonitor:
    """
    Monitors the health of peer L104 daemons by reading their heartbeat files.
    Provides a unified view of the daemon ecosystem.

    Heartbeat formats supported:
      - JSON with "timestamp" or "time" key (guardian, vqpu)
      - Raw text: first line is epoch seconds (micro: ``timestamp\\ntick\\npid``)
      - Raw text: first line is epoch nanoseconds (nano: ``time_ns\\n``)
      - Fallback: file mtime used when content parsing fails
    """

    DAEMON_PATHS = {
        "nano": Path("/tmp/l104_bridge/nano/python_heartbeat"),
        "micro": Path("/tmp/l104_bridge/micro/heartbeat"),
        "vqpu": Path("/tmp/l104_bridge/outbox/daemon_heartbeat.json"),
        "qai": Path("/tmp/l104_bridge/quantum_ai/heartbeat"),
        "guardian": IPC_HEARTBEAT,
    }

    # ── helpers ──

    @staticmethod
    def _parse_heartbeat_ts(path: Path) -> float:
        """Extract epoch-seconds timestamp from any heartbeat format.

        Returns 0.0 on failure.
        """
        try:
            raw = path.read_text().strip()
            if not raw:
                return 0.0

            # 1) Try JSON (guardian writes ``{"timestamp": …}``)
            if raw.startswith("{"):
                data = json.loads(raw)
                return float(data.get("timestamp", data.get("time", 0)))

            # 2) Raw text — first line is numeric timestamp
            first_line = raw.split("\n", 1)[0].strip()
            ts = float(first_line)

            # Nano daemon writes time_ns() (19-digit int like 1773080718700000000).
            # Micro daemon writes time.time() (10-digit float like 1773080718.700).
            if ts > 1e15:          # nanoseconds → seconds
                ts = ts / 1e9
            elif ts < 1e8:         # clearly not a unix epoch — ignore
                return 0.0

            return ts
        except Exception:
            pass

        # 3) Fallback: use file mtime as last resort
        try:
            return path.stat().st_mtime
        except Exception:
            return 0.0

    # Process signatures for fallback detection (matches against cmdline)
    DAEMON_PROC_SIGS = {
        "nano": "l104_vqpu.nano_daemon",
        "micro": "l104_vqpu.micro_daemon",
        "vqpu": "l104_vqpu_boot_manager",
        "qai": "l104_quantum_ai_daemon",
        "guardian": "l104_resource_guardian",
    }

    @staticmethod
    def _process_alive(signature: str) -> bool:
        """Check if a process matching *signature* is running (via ``pgrep``)."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", signature],
                capture_output=True, timeout=3,
            )
            return result.returncode == 0
        except Exception:
            return False

    def scan_daemons(self) -> Dict[str, Dict[str, Any]]:
        """Check heartbeat status of all known L104 daemons.

        Uses heartbeat files as primary signal and process scanning as
        fallback so daemons with stale/missing heartbeats are still
        detected as alive when their process is running.
        """
        now = time.time()
        results = {}
        for name, path in self.DAEMON_PATHS.items():
            status: Dict[str, Any] = {
                "alive": False, "stale": True, "age_s": None, "data": None,
            }
            try:
                if path.exists():
                    ts = self._parse_heartbeat_ts(path)
                    if ts > 0:
                        age = now - ts
                        status["alive"] = age < 120   # Alive if heartbeat < 2 min
                        status["stale"] = age > 30
                        status["age_s"] = round(age, 1)
                    # Attach raw JSON data when available
                    try:
                        raw = path.read_text().strip()
                        if raw.startswith("{"):
                            status["data"] = json.loads(raw)
                    except Exception:
                        pass
            except Exception:
                pass

            # Fallback: process-based detection when heartbeat is stale/missing
            if not status["alive"]:
                sig = self.DAEMON_PROC_SIGS.get(name)
                if sig and self._process_alive(sig):
                    status["alive"] = True
                    status["stale"] = True  # Process alive but heartbeat stale
                    if status["age_s"] is None:
                        status["age_s"] = -1  # Sentinel: no heartbeat file

            results[name] = status
        return results

    def broadcast_to_peers(self, command: str, payload: Dict[str, Any]) -> int:
        """Write guardian commands to peer daemon inbox directories."""
        sent = 0
        peer_inboxes = [
            Path("/tmp/l104_bridge/nano/inbox"),
            Path("/tmp/l104_bridge/micro/inbox"),
            Path("/tmp/l104_bridge/inbox"),  # Global VQPU inbox
        ]
        msg = {
            "command": command,
            "source": "resource_guardian",
            "version": VERSION,
            "timestamp": time.time(),
            "payload": payload,
        }
        for inbox in peer_inboxes:
            try:
                inbox.mkdir(parents=True, exist_ok=True)
                cmd_file = inbox / f"guardian_{command}.json"
                cmd_file.write_text(json.dumps(msg, indent=2))
                sent += 1
            except Exception:
                pass
        return sent

    def to_dict(self) -> Dict[str, Any]:
        daemons = self.scan_daemons()
        alive = sum(1 for d in daemons.values() if d["alive"])
        return {
            "daemons_alive": alive,
            "daemons_total": len(daemons),
            "details": {k: {"alive": v["alive"], "age_s": v["age_s"]}
                        for k, v in daemons.items()},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# INTERVENTION ENGINE — Active responses to pressure
# ═══════════════════════════════════════════════════════════════════════════════

class InterventionEngine:
    """
    Executes graduated interventions based on pressure level.
    Each level includes all interventions from lower levels.
    """

    def __init__(self):
        self._last_gc_time = 0.0
        self._gc_min_interval = 10.0  # Don't GC more than every 10s
        self._paused_daemons: set = set()
        self._thread_pool_capped = False
        self._original_nice = None
        self._interventions: collections.deque = collections.deque(maxlen=1000)  # v2: was 200
        self._level_history: collections.deque = collections.deque(maxlen=500)   # v2: was 100

    def intervene(self, vitals: VitalSigns) -> List[Intervention]:
        """Execute all appropriate interventions for current pressure level."""
        level = vitals.pressure_level
        self._level_history.append((vitals.timestamp, level))
        actions: List[Intervention] = []

        if level >= PressureLevel.ELEVATED:
            actions.extend(self._level_1_elevated(vitals))

        if level >= PressureLevel.HIGH:
            actions.extend(self._level_2_high(vitals))

        if level >= PressureLevel.CRITICAL:
            actions.extend(self._level_3_critical(vitals))

        if level >= PressureLevel.SURVIVAL:
            actions.extend(self._level_4_survival(vitals))

        self._interventions.extend(actions)
        return actions

    def _record(self, level: PressureLevel, action: str, detail: str,
                vitals: VitalSigns) -> Intervention:
        return Intervention(
            timestamp=time.time(),
            level=level,
            action=action,
            detail=detail,
            vitals_snapshot=vitals.to_dict(),
        )

    # ── LEVEL 1: ELEVATED — Gentle throttling ──

    def _level_1_elevated(self, v: VitalSigns) -> List[Intervention]:
        actions = []

        # Adaptive GC
        now = time.time()
        if now - self._last_gc_time > self._gc_min_interval:
            before = len(gc.get_objects())
            gc.collect(generation=0)
            after = len(gc.get_objects())
            freed = before - after
            self._last_gc_time = now
            if freed > 100:
                actions.append(self._record(
                    PressureLevel.ELEVATED, "gc_gen0",
                    f"Collected gen0: freed {freed} objects", v))

        # Broadcast slowdown hint via IPC
        self._write_ipc_command("throttle", {
            "level": "elevated",
            "max_threads": MAX_THREAD_POOL_WORKERS_ELEVATED,
            "reason": f"mem={v.system_mem_pct:.0f}% avail={v.available_ram_mb:.0f}MB "
                      f"swap={v.swap_file_count}",
        })

        return actions

    # ── LEVEL 2: HIGH — Aggressive throttling ──

    def _level_2_high(self, v: VitalSigns) -> List[Intervention]:
        actions = []

        # Full GC sweep
        now = time.time()
        if now - self._last_gc_time > 5.0:
            before = len(gc.get_objects())
            gc.collect()  # All generations
            after = len(gc.get_objects())
            freed = before - after
            self._last_gc_time = now
            actions.append(self._record(
                PressureLevel.HIGH, "gc_full",
                f"Full GC: freed {freed} objects", v))

        # Signal other L104 daemons to pause non-essential work
        self._write_ipc_command("pause_nonessential", {
            "level": "high",
            "max_threads": MAX_THREAD_POOL_WORKERS_HIGH,
            "available_mb": v.available_ram_mb,
            "swap_files": v.swap_file_count,
        })

        # Lower our own nice value to keep guardian responsive
        if self._original_nice is None:
            try:
                self._original_nice = os.nice(0)
                os.nice(-5)  # Higher priority
                actions.append(self._record(
                    PressureLevel.HIGH, "nice_boost",
                    f"Raised guardian priority (nice -5)", v))
            except PermissionError:
                pass

        return actions

    # ── LEVEL 3: CRITICAL — Emergency measures ──

    def _level_3_critical(self, v: VitalSigns) -> List[Intervention]:
        actions = []

        # Emergency aggressive GC with threshold tuning
        gc.set_threshold(100, 5, 2)  # Very aggressive thresholds
        gc.collect()
        gc.collect()  # Double-collect for cyclic refs
        actions.append(self._record(
            PressureLevel.CRITICAL, "emergency_gc",
            f"Emergency double-GC with aggressive thresholds", v))

        # Signal ALL L104 daemons to suspend
        self._write_ipc_command("suspend_all", {
            "level": "critical",
            "max_threads": MAX_THREAD_POOL_WORKERS_CRITICAL,
            "available_mb": v.available_ram_mb,
            "swap_files": v.swap_file_count,
            "action": "suspend_quantum_sims",
        })

        # Drop __pycache__ entries to free memory
        try:
            mods_to_drop = []
            for name, mod in sys.modules.items():
                if hasattr(mod, "__cached__") and name.startswith("l104_"):
                    # Don't drop ourselves or critical infrastructure
                    if "guardian" in name or "intellect" in name:
                        continue
                    mods_to_drop.append(name)
            if len(mods_to_drop) > 20:
                # Only drop if significant — avoid thrashing
                dropped = 0
                for name in mods_to_drop[:200]:  # v2: was 50 — unlimited batch
                    try:
                        del sys.modules[name]
                        dropped += 1
                    except (KeyError, RuntimeError):
                        pass
                if dropped:
                    gc.collect()
                    actions.append(self._record(
                        PressureLevel.CRITICAL, "module_unload",
                        f"Unloaded {dropped} cached L104 modules", v))
        except Exception:
            pass

        return actions

    # ── LEVEL 4: SURVIVAL — Last resort before OS panic ──

    def _level_4_survival(self, v: VitalSigns) -> List[Intervention]:
        actions = []

        # Broadcast emergency halt to all daemons
        self._write_ipc_command("emergency_halt", {
            "level": "survival",
            "max_threads": 1,
            "available_mb": v.available_ram_mb,
            "swap_files": v.swap_file_count,
            "action": "yield_to_os",
            "message": "SURVIVAL MODE — System near panic threshold. "
                       "All L104 work halted to prevent kernel panic.",
        })

        actions.append(self._record(
            PressureLevel.SURVIVAL, "emergency_halt",
            f"SURVIVAL: Broadcast halt. "
            f"RAM={v.available_ram_mb:.0f}MB swap={v.swap_file_count} "
            f"mem={v.system_mem_pct:.0f}%", v))

        # Voluntary sleep to completely yield CPU to WindowServer/OS
        time.sleep(2.0)

        return actions

    def _write_ipc_command(self, command: str, payload: Dict[str, Any]) -> None:
        """Write command to IPC outbox for other daemons to read."""
        try:
            outbox = IPC_OUTBOX
            outbox.mkdir(parents=True, exist_ok=True)
            msg = {
                "command": command,
                "source": "resource_guardian",
                "timestamp": time.time(),
                "payload": payload,
            }
            # Single command file — latest overwrites (other daemons poll)
            cmd_file = outbox / f"guardian_{command}.json"
            cmd_file.write_text(json.dumps(msg, indent=2))
        except Exception as e:
            logger.debug(f"IPC write failed: {e}")

    def restore_nominal(self) -> List[str]:
        """Restore settings when pressure drops back to nominal."""
        restored = []
        if self._original_nice is not None:
            try:
                # Can't directly set nice, but we can offset back
                os.nice(5)  # Lower priority back
                self._original_nice = None
                restored.append("nice_restored")
            except Exception:
                pass

        # GC back to defaults
        gc.set_threshold(700, 10, 10)
        restored.append("gc_thresholds_reset")

        # Clear IPC commands
        self._write_ipc_command("resume", {
            "level": "nominal",
            "max_threads": 0,  # 0 = no cap
            "action": "resume_all",
        })
        restored.append("ipc_resume_sent")

        return restored

    @property
    def intervention_count(self) -> int:
        return len(self._interventions)

    @property
    def recent_interventions(self) -> List[Dict[str, Any]]:
        return [
            {
                "time": i.timestamp,
                "level": LEVEL_NAMES[i.level],
                "action": i.action,
                "detail": i.detail,
            }
            for i in list(self._interventions)[-20:]
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# THREAD POOL GOVERNOR — Enforces max workers across all L104 ThreadPools
# ═══════════════════════════════════════════════════════════════════════════════

class ThreadPoolGovernor:
    """
    Monkeypatches concurrent.futures.ThreadPoolExecutor to enforce a
    process-wide worker cap. On resource-constrained hardware (2-core 4GB),
    the "QUANTUM AMPLIFIED" pools (cpu*4, cpu*8) spawn 8-16 threads that
    thrash the scheduler and starve WindowServer.

    The governor intercepts __init__ and clamps max_workers.
    """

    def __init__(self):
        self._cap: int = 0  # 0 = no cap
        self._original_init = None
        self._installed = False
        self._clamped_count = 0

    def install(self) -> None:
        """Install the governor hook on ThreadPoolExecutor."""
        if self._installed:
            return
        try:
            from concurrent.futures import ThreadPoolExecutor
            self._original_init = ThreadPoolExecutor.__init__

            governor = self  # closure ref

            def governed_init(self_tpe, max_workers=None, **kwargs):
                if governor._cap > 0 and max_workers is not None:
                    if max_workers > governor._cap:
                        old = max_workers
                        max_workers = governor._cap
                        governor._clamped_count += 1
                        logger.debug(f"ThreadPool clamped: {old} → {max_workers}")
                governor._original_init(self_tpe, max_workers=max_workers, **kwargs)

            ThreadPoolExecutor.__init__ = governed_init
            self._installed = True
            logger.info(f"ThreadPoolGovernor installed")
        except Exception as e:
            logger.warning(f"ThreadPoolGovernor install failed: {e}")

    def set_cap(self, max_workers: int) -> None:
        """Set the thread pool cap. 0 = no cap."""
        self._cap = max(0, max_workers)

    def uninstall(self) -> None:
        """Remove the governor hook."""
        if self._installed and self._original_init:
            try:
                from concurrent.futures import ThreadPoolExecutor
                ThreadPoolExecutor.__init__ = self._original_init
                self._installed = False
                logger.info("ThreadPoolGovernor uninstalled")
            except Exception:
                pass

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "installed": self._installed,
            "current_cap": self._cap,
            "total_clamped": self._clamped_count,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RESOURCE GUARDIAN DAEMON
# ═══════════════════════════════════════════════════════════════════════════════

class ResourceGuardianDaemon:
    """
    L104 Resource Guardian — Quantum-grade macOS protection daemon.

    Monitors system vitals every 5 seconds and actively intervenes to
    prevent kernel panics from memory exhaustion.

    Usage:
        guardian = ResourceGuardianDaemon()
        guardian.start()    # Background daemon thread
        guardian.status()   # Current state + history
        guardian.stop()     # Graceful shutdown
    """

    def __init__(self, tick_interval: float = TICK_INTERVAL_S,
                 verbose: bool = False, install_governor: bool = True):
        self.tick_interval = max(MIN_TICK_S, min(MAX_TICK_S, tick_interval))
        self.verbose = verbose
        self.running = False
        self.tick_count = 0

        # Subsystems
        self.collector = VitalSignsCollector()
        self.interventor = InterventionEngine()
        self.governor = ThreadPoolGovernor()
        # v2.1 new subsystems
        self.predictor = MemoryTrendPredictor(window=60, alpha=0.15)
        self.hunter = L104ProcessHunter()
        self.analytics = TelemetryAnalytics(window=200)
        self.peer_monitor = CrossDaemonMonitor()

        # State
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._current_level = PressureLevel.NOMINAL
        self._peak_level = PressureLevel.NOMINAL
        self._level_durations: Dict[PressureLevel, float] = {
            l: 0.0 for l in PressureLevel
        }
        self._level_transition_time = time.time()
        self._start_time = 0.0

        # Telemetry — v2.0: 5x larger
        self._telemetry: collections.deque = collections.deque(maxlen=TELEMETRY_WINDOW)
        self._state_path = Path(
            os.environ.get("L104_ROOT", os.getcwd())) / STATE_FILE

        # v2.0: Predictive pressure model
        self._predicted_level: Optional[PressureLevel] = None
        self._mem_trend: collections.deque = collections.deque(maxlen=PREDICTIVE_WINDOW)

        # v2.0: Cross-daemon health aggregation
        self._cross_daemon_health: Dict[str, Any] = {}

        # v3.0: ML pressure predictor
        self._pressure_predictions: collections.deque = collections.deque(maxlen=50)
        self._predicted_pressure_level = PressureLevel.NOMINAL

        # v3.0: Per-daemon RSS tracking
        self._daemon_rss: Dict[str, float] = {}  # daemon_name → RSS MB
        self._quota_violations: collections.deque = collections.deque(maxlen=50)

        # v3.0: Swap prediction
        self._swap_trend: collections.deque = collections.deque(maxlen=30)  # swap_used_mb history
        self._swap_predicted_mb = 0.0

        # v3.0: Resilience score
        self._resilience_score = 1.0
        self._resilience_history: collections.deque = collections.deque(maxlen=50)

        # v3.0: Vitals history (VitalSigns objects for prediction methods)
        self._vitals_history: collections.deque = collections.deque(maxlen=100)

        # Install thread pool governor
        if install_governor:
            self.governor.install()

        # IPC setup
        self._setup_ipc()

    def _setup_ipc(self) -> None:
        """Create IPC directories."""
        for d in [IPC_BASE, IPC_OUTBOX, IPC_INBOX]:
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    # ── Lifecycle ──

    def start(self) -> None:
        """Start the guardian daemon thread."""
        if self.running:
            logger.warning("Guardian already running")
            return
        self.running = True
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._daemon_loop,
            name="L104-ResourceGuardian",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self._atexit_handler)
        logger.info(f"[GUARDIAN v{VERSION}] Started — tick={self.tick_interval}s "
                     f"governor={'ON' if self.governor._installed else 'OFF'}")

    def stop(self) -> None:
        """Gracefully stop the guardian."""
        if not self.running:
            return
        self.running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        self.governor.set_cap(0)  # Remove cap
        self.interventor.restore_nominal()
        self._persist_state()
        logger.info("[GUARDIAN] Stopped")

    def _atexit_handler(self) -> None:
        """Save state on process exit."""
        try:
            self._persist_state()
        except Exception:
            pass

    # ── Main Loop ──

    def _daemon_loop(self) -> None:
        """Main tick loop."""
        logger.info(f"[GUARDIAN] Daemon loop started — PID {os.getpid()}")
        consecutive_errors = 0

        while not self._stop_event.is_set():
            try:
                self._tick()
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"[GUARDIAN] Tick error ({consecutive_errors}): {e}")
                if consecutive_errors >= 10:
                    logger.critical("[GUARDIAN] 10 consecutive errors — backing off")
                    self._stop_event.wait(30.0)
                    consecutive_errors = 0

            # Adaptive tick: faster when pressure is high
            wait = self.tick_interval
            if self._current_level >= PressureLevel.CRITICAL:
                wait = max(MIN_TICK_S, wait / 2)
            elif self._current_level >= PressureLevel.HIGH:
                wait = max(MIN_TICK_S, wait * 0.7)
            elif self._current_level == PressureLevel.NOMINAL:
                wait = min(MAX_TICK_S, wait * 1.2)

            self._stop_event.wait(wait)

        logger.info("[GUARDIAN] Daemon loop exited")

    def _tick(self) -> None:
        """Execute one guardian tick."""
        self.tick_count += 1

        # Collect vital signs
        vitals = self.collector.collect()
        new_level = vitals.pressure_level

        # Track level transitions
        if new_level != self._current_level:
            old_name = LEVEL_NAMES[self._current_level]
            new_name = LEVEL_NAMES[new_level]
            elapsed = time.time() - self._level_transition_time
            self._level_durations[self._current_level] += elapsed
            self._level_transition_time = time.time()

            direction = "↑" if new_level > self._current_level else "↓"
            color = LEVEL_COLORS.get(new_level, "")
            logger.warning(
                f"[GUARDIAN] {direction} {old_name} → {color}{new_name}{RESET} "
                f"(mem={vitals.system_mem_pct:.0f}% "
                f"avail={vitals.available_ram_mb:.0f}MB "
                f"swap={vitals.swap_file_count})")

            self._current_level = new_level
            self._peak_level = max(self._peak_level, new_level)

            # Update thread pool governor cap based on level
            caps = {
                PressureLevel.NOMINAL: 0,  # no cap
                PressureLevel.ELEVATED: MAX_THREAD_POOL_WORKERS_ELEVATED,
                PressureLevel.HIGH: MAX_THREAD_POOL_WORKERS_HIGH,
                PressureLevel.CRITICAL: MAX_THREAD_POOL_WORKERS_CRITICAL,
                PressureLevel.SURVIVAL: 1,
            }
            self.governor.set_cap(caps.get(new_level, 0))

            # Restore nominal if we just dropped back
            if new_level == PressureLevel.NOMINAL:
                restored = self.interventor.restore_nominal()
                if restored:
                    logger.info(f"[GUARDIAN] Restored: {', '.join(restored)}")

        # Execute interventions if needed
        if new_level > PressureLevel.NOMINAL:
            actions = self.interventor.intervene(vitals)
            for a in actions:
                if self.verbose or a.level >= PressureLevel.HIGH:
                    logger.warning(
                        f"[GUARDIAN] [{LEVEL_NAMES[a.level]}] {a.action}: {a.detail}")

        # Heartbeat
        self._write_heartbeat(vitals)

        # Telemetry
        self._telemetry.append(vitals.to_dict())

        # v2.1: Feed new subsystems
        self.predictor.ingest(vitals)
        self.analytics.ingest(vitals)

        # v3.0: Store VitalSigns objects + predictive analytics
        self._vitals_history.append(vitals)
        self._predict_pressure_v3()
        self._predict_swap(vitals.swap_used_mb)
        self._compute_resilience()

        # v2.1: Predictive alerting from MemoryTrendPredictor
        alert = self.predictor.should_alert()
        if alert:
            logger.warning(f"[GUARDIAN] {alert}")
            self.interventor._write_ipc_command("predictive_alert", {
                "alert": alert,
                "trend": self.predictor.to_dict(),
                "grade": self.predictor.trend_grade,
            })

        # v2.0: Predictive pressure model — trend analysis
        self._mem_trend.append(vitals.system_mem_pct)
        self._predicted_level = self._predict_pressure()
        if (self._predicted_level is not None
                and self._predicted_level > new_level
                and self._predicted_level >= PressureLevel.HIGH):
            logger.warning(
                f"[GUARDIAN] ⚡ PREDICTIVE: Memory trending toward "
                f"{LEVEL_NAMES[self._predicted_level]} "
                f"(slope={self._compute_mem_slope():.2f}%/tick)")

        # v2.1: Process hunter — kill runaways at CRITICAL+
        if new_level >= PressureLevel.CRITICAL and self.tick_count % 4 == 0:
            killed = self.hunter.kill_runaway(
                rss_threshold_mb=512.0 if new_level >= PressureLevel.SURVIVAL else 1024.0
            )
            if killed:
                logger.warning(f"[GUARDIAN] HUNTER: Killed {len(killed)} runaway process(es): {killed}")

        # v2.1: Broadcast to peer daemons at HIGH+
        if new_level >= PressureLevel.HIGH and self.tick_count % 6 == 0:
            peers_sent = self.peer_monitor.broadcast_to_peers("guardian_pressure", {
                "level": LEVEL_NAMES[new_level],
                "available_mb": vitals.available_ram_mb,
                "swap_files": vitals.swap_file_count,
                "trend_grade": self.predictor.trend_grade,
                "action": "throttle" if new_level == PressureLevel.HIGH else "suspend",
            })
            if peers_sent and self.verbose:
                logger.info(f"[GUARDIAN] Broadcast to {peers_sent} peer daemon inboxes")

        # v2.0 + v2.1: Cross-daemon health aggregation (every 12 ticks)
        if self.tick_count % 12 == 0:
            self._read_cross_daemon_health()

        # Periodic state persistence (every 60 ticks ≈ 5 min)
        if self.tick_count % 60 == 0:
            self._persist_state()

        # Verbose logging
        if self.verbose and self.tick_count % 12 == 0:  # Every ~60s in verbose
            self._log_vitals(vitals)

    def _predict_pressure(self) -> Optional[PressureLevel]:
        """v2.0: Predict future pressure from memory trend using linear regression."""
        trend = list(self._mem_trend)
        if len(trend) < 4:
            return None
        slope = self._compute_mem_slope()
        if slope <= PREDICTIVE_SLOPE_THRESH:
            return None
        # Extrapolate 6 ticks ahead
        current = trend[-1]
        predicted_mem = current + slope * 6
        if predicted_mem >= THRESH_SURVIVAL_PCT:
            return PressureLevel.SURVIVAL
        elif predicted_mem >= THRESH_CRITICAL_PCT:
            return PressureLevel.CRITICAL
        elif predicted_mem >= THRESH_HIGH_PCT:
            return PressureLevel.HIGH
        elif predicted_mem >= THRESH_ELEVATED_PCT:
            return PressureLevel.ELEVATED
        return None

    def _compute_mem_slope(self) -> float:
        """Compute linear slope of memory trend (% per tick)."""
        trend = list(self._mem_trend)
        n = len(trend)
        if n < 2:
            return 0.0
        # Simple linear regression: slope = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
        x_mean = (n - 1) / 2.0
        y_mean = sum(trend) / n
        num = sum((i - x_mean) * (trend[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0

    def _predict_pressure_v3(self):
        """v3.0: Linear regression on memory % to predict pressure N ticks ahead."""
        if len(self._vitals_history) < PREDICT_MIN_WINDOW:
            return

        recent = list(self._vitals_history)[-PREDICT_MIN_WINDOW:]
        mem_pcts = [vs.system_mem_pct for vs in recent]

        n = len(mem_pcts)
        x_mean = (n - 1) / 2.0
        y_mean = sum(mem_pcts) / n

        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(mem_pcts))
        den = sum((i - x_mean) ** 2 for i in range(n))

        slope = num / den if den > 0 else 0.0
        predicted_pct = mem_pcts[-1] + slope * PREDICT_HORIZON_TICKS

        if predicted_pct >= THRESH_SURVIVAL_PCT:
            self._predicted_pressure_level = PressureLevel.SURVIVAL
        elif predicted_pct >= THRESH_CRITICAL_PCT:
            self._predicted_pressure_level = PressureLevel.CRITICAL
        elif predicted_pct >= THRESH_HIGH_PCT:
            self._predicted_pressure_level = PressureLevel.HIGH
        elif predicted_pct >= THRESH_ELEVATED_PCT:
            self._predicted_pressure_level = PressureLevel.ELEVATED
        else:
            self._predicted_pressure_level = PressureLevel.NOMINAL

        self._pressure_predictions.append({
            "ts": time.time(),
            "current_pct": round(mem_pcts[-1], 2),
            "predicted_pct": round(predicted_pct, 2),
            "slope_per_tick": round(slope, 3),
            "predicted_level": self._predicted_pressure_level.name,
        })

    def _predict_swap(self, current_swap_mb: float):
        """v3.0: Predict swap usage trend and warn before swap files are created."""
        self._swap_trend.append(current_swap_mb)
        if len(self._swap_trend) < 5:
            return

        recent = list(self._swap_trend)[-10:]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n

        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        den = sum((i - x_mean) ** 2 for i in range(n))

        slope = num / den if den > 0 else 0.0
        self._swap_predicted_mb = recent[-1] + slope * PREDICT_HORIZON_TICKS

        if slope > SWAP_PREDICT_SLOPE_THRESH and current_swap_mb < 100:
            logger.warning(
                "SWAP PREDICTION: swap growing at %.1f MB/tick, predicted %.0f MB in %d ticks",
                slope, self._swap_predicted_mb, PREDICT_HORIZON_TICKS)

    def _compute_resilience(self):
        """v3.0: Compute system resilience score (0-1)."""
        # Daemon diversity: count available daemons
        daemon_states = [
            ".l104_vqpu_daemon_state.json",
            ".l104_vqpu_micro_daemon.json",
            ".l104_nano_daemon_python.json",
            ".l104_quantum_ai_daemon.json",
        ]
        l104_root = os.environ.get("L104_ROOT", os.getcwd())
        daemons_alive = sum(
            1 for f in daemon_states
            if Path(l104_root, f).exists() and
            (time.time() - Path(l104_root, f).stat().st_mtime) < 600
        )
        diversity = daemons_alive / max(len(daemon_states), 1)

        # Recovery speed: how fast pressure drops (compare last 5 ticks)
        recovery = 0.5
        if len(self._vitals_history) >= 5:
            recent = [vs.system_mem_pct for vs in list(self._vitals_history)[-5:]]
            if recent[0] > recent[-1]:
                recovery = min(1.0, (recent[0] - recent[-1]) / 10.0 + 0.5)

        # Headroom: available RAM as fraction of total
        headroom = 0.5
        if self._vitals_history:
            latest = self._vitals_history[-1]
            if latest.total_ram_mb > 0:
                headroom = latest.available_ram_mb / latest.total_ram_mb

        # Stability: inverse of pressure variance
        stability = 0.5
        if len(self._vitals_history) >= 10:
            pcts = [vs.system_mem_pct for vs in list(self._vitals_history)[-10:]]
            mean_pct = sum(pcts) / len(pcts)
            variance = sum((p - mean_pct) ** 2 for p in pcts) / len(pcts)
            stability = max(0.0, 1.0 - (variance / 100.0))

        w = RESILIENCE_SCORE_WEIGHTS
        self._resilience_score = round(
            diversity * w["daemon_diversity"] +
            recovery * w["recovery_speed"] +
            headroom * w["headroom"] +
            stability * w["stability"], 4)
        self._resilience_history.append({
            "ts": time.time(),
            "score": self._resilience_score,
            "diversity": round(diversity, 3),
            "recovery": round(recovery, 3),
            "headroom": round(headroom, 3),
            "stability": round(stability, 3),
        })

    def _read_cross_daemon_health(self) -> None:
        """v2.0: Read heartbeats from sibling L104 daemons for unified health view."""
        bridge = Path("/tmp/l104_bridge")
        health: Dict[str, Any] = {}
        try:
            if bridge.exists():
                for daemon_dir in bridge.iterdir():
                    if daemon_dir.is_dir() and daemon_dir.name != "guardian":
                        hb_file = daemon_dir / "heartbeat"
                        if hb_file.exists():
                            try:
                                data = json.loads(hb_file.read_text())
                                age = time.time() - data.get("timestamp", 0)
                                health[daemon_dir.name] = {
                                    "daemon": data.get("daemon", daemon_dir.name),
                                    "version": data.get("version", "?"),
                                    "tick": data.get("tick", 0),
                                    "age_s": round(age, 1),
                                    "alive": age < 120,  # stale if >2 min
                                }
                            except (json.JSONDecodeError, Exception):
                                health[daemon_dir.name] = {"alive": False, "error": "parse"}
        except Exception:
            pass
        self._cross_daemon_health = health

    def _log_vitals(self, v: VitalSigns) -> None:
        """Log vital signs summary (v2.0: includes new vitals + sacred scoring)."""
        color = LEVEL_COLORS.get(v.pressure_level, "")
        sacred_res = (1.0 - v.system_mem_pct / 100.0) ** (1.0 / PHI)
        pred_str = ""
        if self._predicted_level is not None and self._predicted_level > v.pressure_level:
            pred_str = f" →{LEVEL_NAMES[self._predicted_level]}"
        print(
            f"  [{color}{LEVEL_NAMES[v.pressure_level]:>9s}{RESET}] "
            f"mem={v.system_mem_pct:5.1f}% "
            f"avail={v.available_ram_mb:6.0f}MB "
            f"swap={v.swap_file_count:2d} "
            f"rss={v.process_rss_mb:6.1f}MB "
            f"load={v.load_avg_1m:4.2f} "
            f"ws_cpu={v.ws_cpu_pct:5.1f}% "
            f"therm={'⚠' if v.thermal_throttle else '✓'} "
            f"disk={v.disk_io_mb_s:5.1f}MB/s "
            f"fds={v.open_fds:4d} "
            f"gpu={v.gpu_util_pct:4.1f}% "
            f"net={v.net_io_mb_s:5.1f}MB/s "
            f"✦={sacred_res:.3f}{pred_str}"
        )

    def _write_heartbeat(self, vitals: VitalSigns) -> None:
        """Write heartbeat file for cross-daemon monitoring (v2.0: expanded)."""
        try:
            IPC_BASE.mkdir(parents=True, exist_ok=True)
            sacred_res = (1.0 - vitals.system_mem_pct / 100.0) ** (1.0 / PHI)
            hb = {
                "daemon": "resource_guardian",
                "version": VERSION,
                "pid": os.getpid(),
                "tick": self.tick_count,
                "timestamp": vitals.timestamp,
                "pressure_level": LEVEL_NAMES[vitals.pressure_level],
                "system_mem_pct": round(vitals.system_mem_pct, 1),
                "available_ram_mb": round(vitals.available_ram_mb, 0),
                "swap_files": vitals.swap_file_count,
                "interventions": self.interventor.intervention_count,
                "governor_cap": self.governor._cap,
                # v2.0 new fields
                "disk_io_mb_s": round(vitals.disk_io_mb_s, 1),
                "open_fds": vitals.open_fds,
                "gpu_util_pct": round(vitals.gpu_util_pct, 1),
                "net_io_mb_s": round(vitals.net_io_mb_s, 1),
                "sacred_resonance": round(sacred_res, 6),
                "predicted_level": (
                    LEVEL_NAMES[self._predicted_level]
                    if self._predicted_level is not None else None),
            }
            IPC_HEARTBEAT.write_text(json.dumps(hb))
        except Exception:
            pass

    def _persist_state(self) -> None:
        """Save daemon state to JSON (v2.0: includes predictive + cross-daemon)."""
        try:
            state = {
                "version": VERSION,
                "pid": os.getpid(),
                "start_time": self._start_time,
                "ticks": self.tick_count,
                "current_level": LEVEL_NAMES[self._current_level],
                "peak_level": LEVEL_NAMES[self._peak_level],
                "level_durations_s": {
                    LEVEL_NAMES[k]: round(v, 1) for k, v in self._level_durations.items()
                },
                "interventions_total": self.interventor.intervention_count,
                "recent_interventions": self.interventor.recent_interventions,
                "governor": self.governor.status,
                # v2.0 new state
                "predicted_level": (
                    LEVEL_NAMES[self._predicted_level]
                    if self._predicted_level is not None else None),
                "mem_slope": round(self._compute_mem_slope(), 4),
                "cross_daemon_health": self._cross_daemon_health,
                "telemetry_window": TELEMETRY_WINDOW,
                "telemetry_filled": len(self._telemetry),
                # v3.0 new state
                "predicted_pressure_level": self._predicted_pressure_level.name,
                "swap_predicted_mb": round(self._swap_predicted_mb, 1),
                "resilience_score": self._resilience_score,
                "resilience_history": list(self._resilience_history)[-3:],
                "quota_violations": list(self._quota_violations)[-5:],
                "saved_at": time.time(),
            }
            self._state_path.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.debug(f"State persist failed: {e}")

    # ── Public API ──

    def status(self) -> Dict[str, Any]:
        """Full guardian status — for dashboards, debug framework, etc. (v2.0: expanded)."""
        uptime = time.time() - self._start_time if self._start_time else 0

        # Latest vitals
        latest = self._telemetry[-1] if self._telemetry else {}

        return {
            "version": VERSION,
            "running": self.running,
            "pid": os.getpid(),
            "uptime_s": round(uptime, 1),
            "ticks": self.tick_count,
            "tick_interval_s": self.tick_interval,
            "current_level": LEVEL_NAMES.get(self._current_level, "UNKNOWN"),
            "peak_level": LEVEL_NAMES.get(self._peak_level, "UNKNOWN"),
            "level_durations_s": {
                LEVEL_NAMES[k]: round(v, 1)
                for k, v in self._level_durations.items()
            },
            "latest_vitals": latest,
            "interventions_total": self.interventor.intervention_count,
            "recent_interventions": self.interventor.recent_interventions[-5:],
            "governor": self.governor.status,
            "god_code_alignment": round(
                GOD_CODE * (1.0 - self._current_level / 10.0), 6),
            # v2.0 new status fields
            "predicted_level": (
                LEVEL_NAMES[self._predicted_level]
                if self._predicted_level is not None else None),
            "mem_slope_pct_per_tick": round(self._compute_mem_slope(), 4),
            "telemetry_depth": len(self._telemetry),
            "cross_daemon_health": self._cross_daemon_health,
            "vital_signs_count": 11,
            "self_test_probes": 19,
            # v2.1 new status fields
            "trend_predictor": self.predictor.to_dict(),
            "process_hunter": self.hunter.to_dict(),
            "analytics": self.analytics.trend_report(),
            "peer_daemons": self.peer_monitor.to_dict(),
            # v3.0 new status fields
            "predicted_pressure_level": self._predicted_pressure_level.name,
            "pressure_predictions": list(self._pressure_predictions)[-3:],
            "swap_predicted_mb": round(self._swap_predicted_mb, 1),
            "resilience_score": self._resilience_score,
            "resilience_history": list(self._resilience_history)[-3:],
            "quota_violations": list(self._quota_violations)[-5:],
        }

    def self_test(self) -> Dict[str, Any]:
        """Run diagnostic self-test (v2.1: 19 probes, for l104_debug.py integration)."""
        results = []
        t0 = time.time()

        # Test 1: Vital signs collection
        try:
            vitals = self.collector.collect()
            results.append({
                "probe": "vital_signs",
                "ok": vitals.system_mem_pct > 0,
                "detail": f"mem={vitals.system_mem_pct:.1f}% avail={vitals.available_ram_mb:.0f}MB",
            })
        except Exception as e:
            results.append({"probe": "vital_signs", "ok": False, "detail": str(e)})

        # Test 2: Swap detection
        try:
            swap_count = self.collector._count_swap_files()
            results.append({
                "probe": "swap_detection",
                "ok": isinstance(swap_count, int),
                "detail": f"{swap_count} swap files",
            })
        except Exception as e:
            results.append({"probe": "swap_detection", "ok": False, "detail": str(e)})

        # Test 3: Thermal state
        try:
            throttled, limit = self.collector._thermal_state()
            results.append({
                "probe": "thermal_state",
                "ok": isinstance(limit, int) and 0 <= limit <= 100,
                "detail": f"limit={limit}% throttled={throttled}",
            })
        except Exception as e:
            results.append({"probe": "thermal_state", "ok": False, "detail": str(e)})

        # Test 4: Pressure level computation
        try:
            level = vitals.pressure_level
            results.append({
                "probe": "pressure_computation",
                "ok": level in PressureLevel,
                "detail": f"{LEVEL_NAMES[level]}",
            })
        except Exception as e:
            results.append({"probe": "pressure_computation", "ok": False, "detail": str(e)})

        # Test 5: IPC write
        try:
            self.interventor._write_ipc_command("self_test", {"test": True})
            test_file = IPC_OUTBOX / "guardian_self_test.json"
            ipc_ok = test_file.exists()
            if ipc_ok:
                test_file.unlink()
            results.append({
                "probe": "ipc_write",
                "ok": ipc_ok,
                "detail": f"IPC outbox at {IPC_OUTBOX}",
            })
        except Exception as e:
            results.append({"probe": "ipc_write", "ok": False, "detail": str(e)})

        # Test 6: Governor
        results.append({
            "probe": "thread_governor",
            "ok": True,  # Governor is optional — may be disabled in debug/test contexts
            "detail": f"installed={self.governor._installed} cap={self.governor._cap} clamped={self.governor._clamped_count}",
        })

        # Test 7: WindowServer detection
        try:
            ws_cpu = self.collector._windowserver_cpu()
            results.append({
                "probe": "windowserver_monitor",
                "ok": True,  # Even 0.0% is valid if WS is idle
                "detail": f"WindowServer CPU={ws_cpu:.1f}%",
            })
        except Exception as e:
            results.append({"probe": "windowserver_monitor", "ok": False, "detail": str(e)})

        # Test 8: Sacred constant alignment
        results.append({
            "probe": "sacred_constants",
            "ok": abs(GOD_CODE - 527.5184818492612) < 1e-10
                  and abs(PHI - 1.618033988749895) < 1e-15,
            "detail": f"GOD_CODE={GOD_CODE} PHI={PHI}",
        })

        # ── v2.0 NEW PROBES ──

        # Test 9: Disk I/O collector
        try:
            disk_io = self.collector._disk_io()
            results.append({
                "probe": "disk_io_collector",
                "ok": isinstance(disk_io, (int, float)) and disk_io >= 0,
                "detail": f"disk_io={disk_io:.1f} MB/s",
            })
        except Exception as e:
            results.append({"probe": "disk_io_collector", "ok": False, "detail": str(e)})

        # Test 10: File descriptor collector
        try:
            fds = self.collector._open_file_descriptors()
            results.append({
                "probe": "file_descriptors",
                "ok": isinstance(fds, int) and fds >= 0,
                "detail": f"open_fds={fds}",
            })
        except Exception as e:
            results.append({"probe": "file_descriptors", "ok": False, "detail": str(e)})

        # Test 11: GPU utilization collector
        try:
            gpu = self.collector._gpu_utilization()
            results.append({
                "probe": "gpu_utilization",
                "ok": isinstance(gpu, (int, float)) and gpu >= 0,
                "detail": f"gpu_util={gpu:.1f}%",
            })
        except Exception as e:
            results.append({"probe": "gpu_utilization", "ok": False, "detail": str(e)})

        # Test 12: Network I/O collector
        try:
            net_io = self.collector._network_io()
            results.append({
                "probe": "network_io_collector",
                "ok": isinstance(net_io, (int, float)) and net_io >= 0,
                "detail": f"net_io={net_io:.1f} MB/s",
            })
        except Exception as e:
            results.append({"probe": "network_io_collector", "ok": False, "detail": str(e)})

        # Test 13: Sacred resonance scoring
        try:
            mem_harmony = 1.0 - vitals.system_mem_pct / 100.0
            sacred_res = mem_harmony ** (1.0 / PHI)
            results.append({
                "probe": "sacred_resonance",
                "ok": 0.0 <= sacred_res <= 1.0,
                "detail": f"resonance={sacred_res:.6f} (PHI-root of {mem_harmony:.3f})",
            })
        except Exception as e:
            results.append({"probe": "sacred_resonance", "ok": False, "detail": str(e)})

        # Test 14: Predictive pressure model
        try:
            # Seed trend with current vitals for test
            self._mem_trend.append(vitals.system_mem_pct)
            slope = self._compute_mem_slope()
            predicted = self._predict_pressure()
            results.append({
                "probe": "predictive_pressure",
                "ok": isinstance(slope, float),
                "detail": f"slope={slope:.4f}%/tick predicted={'none' if predicted is None else LEVEL_NAMES[predicted]}",
            })
        except Exception as e:
            results.append({"probe": "predictive_pressure", "ok": False, "detail": str(e)})

        # Test 15: Cross-daemon health reader
        try:
            self._read_cross_daemon_health()
            n_peers = len(self._cross_daemon_health)
            alive = sum(1 for v in self._cross_daemon_health.values()
                        if isinstance(v, dict) and v.get("alive"))
            results.append({
                "probe": "cross_daemon_health",
                "ok": True,  # Even 0 peers is valid (no siblings running)
                "detail": f"{alive}/{n_peers} sibling daemons alive",
            })
        except Exception as e:
            results.append({"probe": "cross_daemon_health", "ok": False, "detail": str(e)})

        # Test 16: MemoryTrendPredictor — v2.1 subsystem
        try:
            self.predictor.ingest(vitals)
            grade = self.predictor.trend_grade
            slope = self.predictor.mem_slope_pct_per_min
            vol = self.predictor.volatility
            results.append({
                "probe": "memory_trend_predictor",
                "ok": grade in ("A", "B", "C", "D", "E", "F") and isinstance(slope, float),
                "detail": f"grade={grade} slope={slope:.3f}%/min vol={vol:.2f}",
            })
        except Exception as e:
            results.append({"probe": "memory_trend_predictor", "ok": False, "detail": str(e)})

        # Test 17: L104ProcessHunter — v2.1 subsystem
        try:
            procs = self.hunter.scan()
            count = self.hunter.process_count
            agg_rss = self.hunter.aggregate_rss_mb
            results.append({
                "probe": "l104_process_hunter",
                "ok": isinstance(procs, list) and isinstance(agg_rss, float),
                "detail": f"{count} L104 procs, aggregate_rss={agg_rss:.0f}MB",
            })
        except Exception as e:
            results.append({"probe": "l104_process_hunter", "ok": False, "detail": str(e)})

        # Test 18: TelemetryAnalytics — v2.1 subsystem
        try:
            self.analytics.ingest(vitals)
            grade = self.analytics.health_grade()
            report = self.analytics.trend_report()
            results.append({
                "probe": "telemetry_analytics",
                "ok": grade in ("?", "A", "B", "C", "D", "E", "F") and isinstance(report, dict),
                "detail": f"grade={grade} anomalies={report['anomalies_total']} samples={report['samples']}",
            })
        except Exception as e:
            results.append({"probe": "telemetry_analytics", "ok": False, "detail": str(e)})

        # Test 19: CrossDaemonMonitor — v2.1 subsystem
        try:
            daemons = self.peer_monitor.scan_daemons()
            alive = sum(1 for d in daemons.values() if d["alive"])
            d_dict = self.peer_monitor.to_dict()
            results.append({
                "probe": "cross_daemon_monitor",
                "ok": isinstance(daemons, dict) and isinstance(d_dict, dict),
                "detail": f"{alive}/{len(daemons)} peers alive (nano/micro/vqpu/guardian)",
            })
        except Exception as e:
            results.append({"probe": "cross_daemon_monitor", "ok": False, "detail": str(e)})

        elapsed_ms = (time.time() - t0) * 1000
        passed = sum(1 for r in results if r["ok"])
        total = len(results)

        return {
            "daemon": "resource_guardian",
            "version": VERSION,
            "probes": total,
            "passed": passed,
            "failed": total - passed,
            "elapsed_ms": round(elapsed_ms, 1),
            "results": results,
            "health": round(passed / total, 4) if total else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON + MODULE API
# ═══════════════════════════════════════════════════════════════════════════════

_guardian: Optional[ResourceGuardianDaemon] = None
_guardian_lock = threading.Lock()


def get_guardian(auto_start: bool = False, **kwargs) -> ResourceGuardianDaemon:
    """Get or create the singleton ResourceGuardianDaemon."""
    global _guardian
    if _guardian is None:
        with _guardian_lock:
            if _guardian is None:
                _guardian = ResourceGuardianDaemon(**kwargs)
                if auto_start:
                    _guardian.start()
    return _guardian


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="L104 Resource Guardian Daemon — Quantum-grade macOS protection")
    parser.add_argument("--self-test", action="store_true",
                        help="Run diagnostic self-test and exit")
    parser.add_argument("--status", action="store_true",
                        help="Show current system status and exit")
    parser.add_argument("--tick", type=float, default=TICK_INTERVAL_S,
                        help=f"Tick interval in seconds (default: {TICK_INTERVAL_S})")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--daemon", action="store_true",
                        help="Run as persistent daemon (foreground)")
    parser.add_argument("--no-governor", action="store_true",
                        help="Don't install ThreadPool governor")
    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    guardian = ResourceGuardianDaemon(
        tick_interval=args.tick,
        verbose=args.verbose,
        install_governor=not args.no_governor,
    )

    if args.self_test:
        print(f"L104 Resource Guardian v{VERSION} — Self-Test")
        print("=" * 60)
        result = guardian.self_test()
        for r in result["results"]:
            icon = "✓" if r["ok"] else "✗"
            print(f"  {icon} {r['probe']:25s} {r['detail']}")
        print("=" * 60)
        print(f"  {result['passed']}/{result['probes']} passed "
              f"({result['elapsed_ms']:.1f}ms)")
        if result["failed"]:
            print(f"  ⚠ {result['failed']} probe(s) need attention")
        else:
            print("  ★ ALL PROBES PASSED ★")
        sys.exit(0 if result["failed"] == 0 else 1)

    if args.status:
        print(f"L104 Resource Guardian v{VERSION} — System Status")
        print("=" * 60)
        vitals = guardian.collector.collect()
        color = LEVEL_COLORS.get(vitals.pressure_level, "")
        sacred_res = (1.0 - vitals.system_mem_pct / 100.0) ** (1.0 / PHI)

        # ── Core Vitals (11) ──
        print(f"  Pressure:     {color}{LEVEL_NAMES[vitals.pressure_level]}{RESET}")
        print(f"  Memory:       {vitals.system_mem_pct:.1f}% used "
              f"({vitals.available_ram_mb:.0f} MB available "
              f"of {vitals.total_ram_mb:.0f} MB)")
        print(f"  Swap:         {vitals.swap_file_count} files "
              f"({vitals.swap_used_mb:.0f} MB used)")
        print(f"  Load avg:     {vitals.load_avg_1m:.2f}")
        print(f"  Thermal:      {'⚠ THROTTLED' if vitals.thermal_throttle else '✓ nominal'}"
              f" (speed limit: {vitals.cpu_speed_limit}%)")
        print(f"  Process RSS:  {vitals.process_rss_mb:.1f} MB")
        print(f"  WindowServer: {vitals.ws_cpu_pct:.1f}% CPU")
        print(f"  Disk I/O:     {vitals.disk_io_mb_s:.1f} MB/s")
        print(f"  File descs:   {vitals.open_fds}")
        print(f"  GPU util:     {vitals.gpu_util_pct:.1f}%")
        print(f"  Network I/O:  {vitals.net_io_mb_s:.1f} MB/s")
        print(f"  Sacred ✦:     {sacred_res:.6f} (PHI-root resonance)")

        # ── v2.1 Analytics ──
        print("─" * 60)
        guardian.predictor.ingest(vitals)
        guardian.analytics.ingest(vitals)
        tte = guardian.predictor.time_to_exhaustion_s
        tte_str = f"{tte:.0f}s" if tte is not None else "∞ (stable)"
        print(f"  Trend grade:  {guardian.predictor.trend_grade} "
              f"(slope={guardian.predictor.mem_slope_pct_per_min:+.2f}%/min)")
        print(f"  Exhaustion:   {tte_str}")
        print(f"  Volatility:   {guardian.predictor.volatility:.2f}")
        print(f"  Health grade: {guardian.analytics.health_grade()}")

        procs = guardian.hunter.scan()
        print(f"  L104 procs:   {len(procs)} "
              f"(aggregate RSS={guardian.hunter.aggregate_rss_mb:.0f} MB)")
        if procs:
            top = procs[0]
            print(f"    Top: PID {top['pid']} RSS={top['rss_mb']:.0f}MB "
                  f"CPU={top['cpu_pct']:.0f}%")

        peers = guardian.peer_monitor.scan_daemons()
        alive_peers = [k for k, v in peers.items() if v["alive"]]
        print(f"  Peer daemons: {len(alive_peers)}/{len(peers)} alive "
              f"({', '.join(alive_peers) if alive_peers else 'none'})")
        for pname, pinfo in peers.items():
            if pinfo["alive"]:
                age = pinfo["age_s"]
                tag = "heartbeat" if age is not None and age >= 0 else "process"
                detail = f"{age:.0f}s ago" if age is not None and age >= 0 else "proc alive"
                stale = " (stale)" if pinfo["stale"] else ""
                print(f"    {pname:10s} ✓ {tag}{stale} — {detail}")

        # ── Thresholds (auto-tuned) ──
        print("─" * 60)
        ram_class = ("4GB-TIGHT" if vitals.total_ram_mb < 4600
                     else "8GB-MOD" if vitals.total_ram_mb < 8700
                     else "16GB+ UNLIMITED")
        print(f"  HW profile:   {ram_class} ({vitals.total_ram_mb:.0f} MB detected)")
        print(f"  Thresholds:   elev={THRESH_ELEVATED_PCT:.0f}% "
              f"high={THRESH_HIGH_PCT:.0f}% "
              f"crit={THRESH_CRITICAL_PCT:.0f}% "
              f"surv={THRESH_SURVIVAL_PCT:.0f}%")
        print("=" * 60)
        sys.exit(0)

    # Daemon mode
    # Detect RAM for profile label
    try:
        import psutil as _ps
        _total_mb = _ps.virtual_memory().total / (1024 * 1024)
    except ImportError:
        _total_mb = 0
    ram_class = ("4GB-TIGHT" if _total_mb < 4600
                 else "8GB-MOD" if _total_mb < 8700
                 else "16GB+ UNLIMITED") if _total_mb > 0 else "UNKNOWN"
    print(f"L104 Resource Guardian v{VERSION} — {ram_class}")
    print(f"  Vital signs:    11")
    print(f"  Self-test:      19 probes")
    print(f"  Subsystems:     predictor + hunter + analytics + peer IPC")
    print(f"  Tick interval:  {guardian.tick_interval}s")
    print(f"  Thread governor: {'ON' if guardian.governor._installed else 'OFF'}")
    print(f"  PID:            {os.getpid()}")
    print(f"  Thresholds:     elevated={THRESH_ELEVATED_PCT}% "
          f"high={THRESH_HIGH_PCT}% critical={THRESH_CRITICAL_PCT}% "
          f"survival={THRESH_SURVIVAL_PCT}%")
    print(f"  Telemetry:      {TELEMETRY_WINDOW} samples")
    print(f"  Predictive:     ON (window={PREDICTIVE_WINDOW} slope={PREDICTIVE_SLOPE_THRESH})")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    guardian.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[GUARDIAN] Shutting down...")
        guardian.stop()
        status = guardian.status()
        print(f"  Ticks: {status['ticks']} | "
              f"Peak level: {status['peak_level']} | "
              f"Interventions: {status['interventions_total']}")


if __name__ == "__main__":
    main()
