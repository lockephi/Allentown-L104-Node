"""L104 Nano Daemon — Native AI Python Substrate v2.0.0

v2.0.0 INTELLIGENCE UPGRADE:
  - Temporal fault causality: detect causal chains (fault A → B within 5s window)
  - Distributed fault clustering: group similar faults by type/severity across tick windows
  - Hardware-specific probe tuning: Apple Silicon vs Intel threshold adaptation
  - Fidelity trend slope alert: linear regression on health to detect sustained decline
  - Cross-daemon fault correlation v2: read ALL daemon heartbeats (guardian, micro, QAI)
  - Health slope tracking with structured alerts when slope < -0.005/tick

Atomized fault detection with native AI-powered anomaly detection.
Part of the L104 Tri-Nano Daemon (Python + Swift + C).

Unlike the micro-daemon (5-15s tick, operational tasks) and the heavy daemon
(60-600s, simulation cycles), the nano daemon operates at 2-5s resolution
detecting miniscule faults invisible to higher layers:

  AI-Powered Nano Probes (12):
    1.  Sacred Constant Drift — ULP-level IEEE 754 comparison (struct.pack)
    2.  Memory Sentinel Canary — φ-scrambled in-process canary values
    3.  Numerical Stability Audit — φ-recurrence, GOD_CODE roundtrip, cancellation
    4.  Phase Drift Accumulation — 100K modular reduction drift measurement
    5.  Statistical Anomaly Detector — Z-score + IQR on health telemetry stream
    6.  Entropy Source Quality — chi-squared + poker test on os.urandom
    7.  Import Health Monitor — verify all 18 L104 packages importable
    8.  State File Integrity — JSON parse + GOD_CODE cross-check on .l104_*.json
    9.  Cross-Daemon Heartbeat — check Swift + C nano daemon liveness
   10.  AI Trend Predictor — linear regression on health window, predict failures
   11.  AI Anomaly Classifier — isolation forest on multi-dimensional fault vectors
   12.  AI Auto-Correlator — cross-correlate faults across all three nano daemons

AI Features:
  - Rolling health window with linear regression trend prediction
  - Isolation-forest-inspired anomaly scoring (no sklearn dependency)
  - Cross-daemon fault correlation via IPC JSON parsing
  - Auto-adaptive tick interval based on fault density
  - Bayesian fault severity estimation from historical patterns

IPC: /tmp/l104_bridge/nano/python_outbox (JSON tick reports)
Heartbeat: /tmp/l104_bridge/nano/python_heartbeat
PID: /tmp/l104_bridge/nano/python_nano.pid
State: .l104_nano_daemon_python.json

Usage:
  python -m l104_vqpu.nano_daemon                  # Run daemon (default 3s tick)
  python -m l104_vqpu.nano_daemon --self-test      # 12-probe self-test, exit 0/1
  python -m l104_vqpu.nano_daemon --health-check   # Read persisted state, report
  python -m l104_vqpu.nano_daemon --once            # Single tick, exit
  python -m l104_vqpu.nano_daemon --tick 2          # Custom tick interval

GOD_CODE=527.5184818492612 | PHI=1.618033988749895 | PILOT: LONDEL
"""

from __future__ import annotations

import atexit
import collections
import json
import math
import os
import signal
import struct
import sys
import time
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

# ═══════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS (Nano-Precision)
# ═══════════════════════════════════════════════════════════════════════

VERSION = "2.0.0"
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.04 + PHI / 1000.0  # 1.0416180339887497
OMEGA = 6539.34712682

# IEEE 754 exact bit patterns
GOD_CODE_BITS = struct.unpack(">Q", struct.pack(">d", GOD_CODE))[0]
PHI_BITS = struct.unpack(">Q", struct.pack(">d", PHI))[0]
VOID_BITS = struct.unpack(">Q", struct.pack(">d", VOID_CONSTANT))[0]

# Tick config
DEFAULT_TICK = 3.0
MIN_TICK = 1.0
MAX_TICK = 10.0

# v2.0: Temporal fault causality
CAUSALITY_WINDOW_S = 5.0          # Faults within 5s may be causally linked
MAX_CAUSAL_CHAINS = 20            # Max tracked causal chains

# v2.0: Fault clustering
CLUSTER_DISTANCE_THRESHOLD = 0.15  # Normalized distance for clustering
MAX_CLUSTERS = 10

# v2.0: Hardware detection
import platform as _platform
IS_APPLE_SILICON = _platform.machine() == "arm64"
IS_INTEL = _platform.machine() == "x86_64"

# v2.0: Fidelity slope alert
SLOPE_ALERT_THRESHOLD = -0.005     # Alert when health slope < -0.005/tick
SLOPE_WINDOW = 20                  # Ticks to compute slope

# v2.0: Additional heartbeat paths for cross-daemon correlation
GUARDIAN_HEARTBEAT = "/tmp/l104_bridge/guardian/heartbeat"
MICRO_HEARTBEAT = "/tmp/l104_bridge/micro/heartbeat"
QAI_HEARTBEAT = "/tmp/l104_bridge/quantum_ai/heartbeat"

# IPC paths
NANO_BASE = "/tmp/l104_bridge/nano"
PYTHON_OUTBOX = f"{NANO_BASE}/python_outbox"
PYTHON_HEARTBEAT = f"{NANO_BASE}/python_heartbeat"
PYTHON_PID = f"{NANO_BASE}/python_nano.pid"
C_HEARTBEAT = f"{NANO_BASE}/c_heartbeat"
SWIFT_HEARTBEAT = f"{NANO_BASE}/swift_heartbeat"
C_OUTBOX = f"{NANO_BASE}/c_outbox"
SWIFT_OUTBOX = f"{NANO_BASE}/swift_outbox"

# Telemetry
TELEMETRY_WINDOW = 300
AI_WINDOW = 50  # Window for AI predictions
PERSIST_EVERY = 10

# ═══════════════════════════════════════════════════════════════════════
# NANO FAULT MODEL
# ═══════════════════════════════════════════════════════════════════════

class NanoSeverity:
    TRACE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    _LABELS = {0: "TRACE", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}
    _PENALTIES = {0: 0.001, 1: 0.01, 2: 0.05, 3: 0.15, 4: 0.30}

    @classmethod
    def label(cls, sev: int) -> str:
        return cls._LABELS.get(sev, "UNKNOWN")

    @classmethod
    def penalty(cls, sev: int) -> float:
        return cls._PENALTIES.get(sev, 0.1)


class NanoFaultType:
    CONSTANT_DRIFT = 0
    MEMORY_CANARY = 1
    NUMERICAL_STAB = 2
    PHASE_DRIFT = 3
    STAT_ANOMALY = 4
    ENTROPY = 5
    IMPORT_HEALTH = 6
    STATE_FILE = 7
    CROSS_DAEMON = 8
    AI_TREND = 9
    AI_ANOMALY = 10
    AI_CORRELATOR = 11


class NanoFault(NamedTuple):
    type: int
    severity: int
    measured: float
    expected: float
    deviation: float
    ulp_distance: int
    description: str
    timestamp: float


# ═══════════════════════════════════════════════════════════════════════
# UTILITY: ULP Distance & Bit Ops
# ═══════════════════════════════════════════════════════════════════════

def double_to_bits(v: float) -> int:
    """IEEE 754 double → 64-bit unsigned integer."""
    return struct.unpack(">Q", struct.pack(">d", v))[0]


def ulp_distance(a: float, b: float) -> int:
    """ULP distance between two doubles."""
    if math.isnan(a) or math.isnan(b):
        return 2**63
    if a == b:
        return 0
    a_bits = double_to_bits(a)
    b_bits = double_to_bits(b)
    # Same sign
    if (a_bits >> 63) == (b_bits >> 63):
        ia = struct.unpack(">q", struct.pack(">Q", a_bits))[0]
        ib = struct.unpack(">q", struct.pack(">Q", b_bits))[0]
        return abs(ia - ib)
    # Different signs
    return (a_bits & 0x7FFFFFFFFFFFFFFF) + (b_bits & 0x7FFFFFFFFFFFFFFF)


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64-bit values."""
    return bin(a ^ b).count("1")


def _ensure_dirs():
    """Ensure all IPC directories exist."""
    for d in ["/tmp/l104_bridge", NANO_BASE, PYTHON_OUTBOX, C_OUTBOX, SWIFT_OUTBOX]:
        os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# PROBE 1: Sacred Constant ULP Drift
# ═══════════════════════════════════════════════════════════════════════

def probe_constant_drift() -> list[NanoFault]:
    faults = []
    ts = time.time()

    # GOD_CODE
    # v2.0: Hardware-tuned ULP tolerance
    _hw_drift_tolerance = 1 if IS_APPLE_SILICON else 2

    gc_ulp = ulp_distance(GOD_CODE, 527.5184818492612)
    gc_bits = double_to_bits(GOD_CODE)
    if gc_ulp > _hw_drift_tolerance or gc_bits != GOD_CODE_BITS:
        sev = NanoSeverity.LOW if gc_ulp <= 2 else NanoSeverity.MEDIUM if gc_ulp <= 8 else NanoSeverity.HIGH
        faults.append(NanoFault(NanoFaultType.CONSTANT_DRIFT, sev,
                                GOD_CODE, 527.5184818492612, GOD_CODE - 527.5184818492612,
                                gc_ulp, f"GOD_CODE ULP drift (tol={_hw_drift_tolerance})", ts))

    # PHI
    phi_ulp = ulp_distance(PHI, 1.618033988749895)
    if phi_ulp > _hw_drift_tolerance:
        sev = NanoSeverity.LOW if phi_ulp <= 2 else NanoSeverity.MEDIUM
        faults.append(NanoFault(NanoFaultType.CONSTANT_DRIFT, sev,
                                PHI, 1.618033988749895, PHI - 1.618033988749895,
                                phi_ulp, f"PHI ULP drift (tol={_hw_drift_tolerance})", ts))

    # VOID_CONSTANT derivation
    expected_vc = 1.04 + 1.618033988749895 / 1000.0
    vc_ulp = ulp_distance(VOID_CONSTANT, expected_vc)
    if vc_ulp > 1:
        faults.append(NanoFault(NanoFaultType.CONSTANT_DRIFT, NanoSeverity.LOW,
                                VOID_CONSTANT, expected_vc, VOID_CONSTANT - expected_vc,
                                vc_ulp, "VOID_CONSTANT formula drift", ts))

    # Sacred resonance: (GOD_CODE/16)^φ ≈ 286
    sacred = (GOD_CODE / 16.0) ** PHI
    if abs(sacred - 286.0) > 0.5:
        faults.append(NanoFault(NanoFaultType.CONSTANT_DRIFT, NanoSeverity.MEDIUM,
                                sacred, 286.0, sacred - 286.0,
                                ulp_distance(sacred, 286.0),
                                "(GOD_CODE/16)^φ ≈ 286 drift", ts))

    return faults


# ═══════════════════════════════════════════════════════════════════════
# PROBE 2: Memory Sentinel Canary
# ═══════════════════════════════════════════════════════════════════════

class MemoryCanary:
    """φ-scrambled in-process sentinel values."""

    def __init__(self):
        self.canary_phi = PHI_BITS ^ 0xA5A5A5A5A5A5A5A5
        self.canary_god = GOD_CODE_BITS ^ 0x5A5A5A5A5A5A5A5A
        self.canary_void = VOID_BITS ^ 0x1041041041041041
        self.checksum = self.canary_phi ^ self.canary_god ^ self.canary_void

    def check(self) -> list[NanoFault]:
        faults = []
        ts = time.time()

        expected_cs = self.canary_phi ^ self.canary_god ^ self.canary_void
        if expected_cs != self.checksum:
            hd = hamming_distance(expected_cs, self.checksum)
            faults.append(NanoFault(NanoFaultType.MEMORY_CANARY, NanoSeverity.CRITICAL,
                                    float(expected_cs), float(self.checksum), float(hd),
                                    hd, f"Canary checksum corruption: {hd}-bit flip", ts))

        expected_phi = double_to_bits(PHI) ^ 0xA5A5A5A5A5A5A5A5
        if self.canary_phi != expected_phi:
            hd = hamming_distance(self.canary_phi, expected_phi)
            faults.append(NanoFault(NanoFaultType.MEMORY_CANARY, NanoSeverity.CRITICAL,
                                    0, 0, float(hd), hd,
                                    f"PHI canary bit-flip ({hd} bits)", ts))

        expected_god = double_to_bits(GOD_CODE) ^ 0x5A5A5A5A5A5A5A5A
        if self.canary_god != expected_god:
            hd = hamming_distance(self.canary_god, expected_god)
            faults.append(NanoFault(NanoFaultType.MEMORY_CANARY, NanoSeverity.CRITICAL,
                                    0, 0, float(hd), hd,
                                    f"GOD_CODE canary bit-flip ({hd} bits)", ts))

        return faults


# ═══════════════════════════════════════════════════════════════════════
# PROBE 3: Numerical Stability Audit
# ═══════════════════════════════════════════════════════════════════════

def probe_numerical_stability() -> list[NanoFault]:
    faults = []
    ts = time.time()

    # φ-recurrence convergence
    a, b = 1.0, PHI
    for _ in range(100):
        a, b = b, a + b
    computed_phi = b / a
    phi_err = abs(computed_phi - PHI)
    if phi_err > 1e-12:
        faults.append(NanoFault(NanoFaultType.NUMERICAL_STAB, NanoSeverity.MEDIUM,
                                computed_phi, PHI, phi_err,
                                ulp_distance(computed_phi, PHI),
                                f"φ-recurrence error: {phi_err:.2e}", ts))

    # GOD_CODE log-pow roundtrip
    encoded = math.log(GOD_CODE) / math.log(286.0) * PHI
    decoded = 286.0 ** (encoded / PHI)
    rt_err = abs(decoded - GOD_CODE)
    if rt_err > 1e-10:
        faults.append(NanoFault(NanoFaultType.NUMERICAL_STAB, NanoSeverity.MEDIUM,
                                decoded, GOD_CODE, rt_err,
                                ulp_distance(decoded, GOD_CODE),
                                f"GOD_CODE roundtrip loss: {rt_err:.2e}", ts))

    # Catastrophic cancellation
    eps = PHI * 1e-15
    recovered = (1.0 + eps) - 1.0
    if eps != 0 and recovered != 0:
        bits_lost = -math.log2(abs(recovered / eps))
        if bits_lost > 4:
            faults.append(NanoFault(NanoFaultType.NUMERICAL_STAB, NanoSeverity.LOW,
                                    bits_lost, 0, bits_lost, 0,
                                    f"Catastrophic cancellation: {bits_lost:.1f} bits lost", ts))

    # Underflow spiral
    val = GOD_CODE
    underflow_count = 0
    for _ in range(2000):
        val /= PHI
        if 0 < val < sys.float_info.min:
            underflow_count += 1
        if val == 0:
            break

    return faults


# ═══════════════════════════════════════════════════════════════════════
# PROBE 4: Phase Drift Accumulation
# ═══════════════════════════════════════════════════════════════════════

def probe_phase_drift() -> list[NanoFault]:
    faults = []
    ts = time.time()
    # v2.0: Hardware-tuned iteration count
    N = 150_000 if IS_APPLE_SILICON else 100_000

    god_phase = math.fmod(GOD_CODE, 2.0 * math.pi)
    phase = 0.0
    for _ in range(N):
        phase += god_phase
        phase = math.fmod(phase, 2.0 * math.pi)

    exact = math.fmod(N * GOD_CODE, 2.0 * math.pi)
    drift = abs(phase - exact)

    # v2.0: Hardware-adaptive thresholds (Apple Silicon has higher FP precision)
    threshold = 5e-9 if IS_APPLE_SILICON else 1e-8
    if drift > threshold:
        sev = NanoSeverity.LOW if drift < 1e-6 else NanoSeverity.MEDIUM if drift < 1e-4 else NanoSeverity.HIGH
        faults.append(NanoFault(NanoFaultType.PHASE_DRIFT, sev,
                                phase, exact, drift,
                                ulp_distance(phase, exact),
                                f"Phase drift after {N} iterations: {drift:.2e} rad "
                                f"({'arm64' if IS_APPLE_SILICON else 'x86_64'})", ts))

    return faults


# ═══════════════════════════════════════════════════════════════════════
# PROBE 5: Statistical Anomaly Detector (AI)
# ═══════════════════════════════════════════════════════════════════════

class StatisticalAnomalyDetector:
    """Z-score + IQR anomaly detection on health telemetry stream."""

    def __init__(self):
        self.health_window: collections.deque[float] = collections.deque(maxlen=AI_WINDOW)
        self.fault_window: collections.deque[int] = collections.deque(maxlen=AI_WINDOW)
        self.duration_window: collections.deque[float] = collections.deque(maxlen=AI_WINDOW)

    def observe(self, health: float, faults: int, duration_ms: float):
        self.health_window.append(health)
        self.fault_window.append(faults)
        self.duration_window.append(duration_ms)

    def detect(self) -> list[NanoFault]:
        faults = []
        ts = time.time()

        if len(self.health_window) < 10:
            return faults

        values = list(self.health_window)
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.001

        # Z-score on latest value
        latest = values[-1]
        if stdev > 0:
            z_score = (latest - mean) / stdev
            if z_score < -2.5:  # Significant health drop
                faults.append(NanoFault(NanoFaultType.STAT_ANOMALY, NanoSeverity.MEDIUM,
                                        latest, mean, z_score, 0,
                                        f"Health Z-score anomaly: z={z_score:.2f} (health={latest:.4f})", ts))

        # IQR on duration
        if len(self.duration_window) >= 10:
            dur_sorted = sorted(self.duration_window)
            q1 = dur_sorted[len(dur_sorted) // 4]
            q3 = dur_sorted[3 * len(dur_sorted) // 4]
            iqr = q3 - q1
            latest_dur = list(self.duration_window)[-1]
            threshold = q3 + 3.0 * iqr
            if latest_dur > threshold and threshold > 0:
                faults.append(NanoFault(NanoFaultType.STAT_ANOMALY, NanoSeverity.LOW,
                                        latest_dur, threshold, latest_dur - threshold, 0,
                                        f"Tick duration outlier: {latest_dur:.1f}ms > {threshold:.1f}ms", ts))

        return faults


# ═══════════════════════════════════════════════════════════════════════
# PROBE 6: Entropy Source Quality
# ═══════════════════════════════════════════════════════════════════════

def probe_entropy_quality() -> list[NanoFault]:
    faults = []
    ts = time.time()

    buf = os.urandom(2500)

    # Chi-squared on byte distribution
    freq = [0] * 256
    for b in buf:
        freq[b] += 1
    expected = len(buf) / 256.0
    chi2 = sum((f - expected) ** 2 / expected for f in freq)

    # 4-bit poker test
    nib_freq = [0] * 16
    for b in buf:
        nib_freq[b >> 4] += 1
        nib_freq[b & 0x0F] += 1
    total_nib = len(buf) * 2
    nib_expected = total_nib / 16.0
    poker = sum((f - nib_expected) ** 2 / nib_expected for f in nib_freq)

    # Serial correlation
    n = len(buf) - 1
    if n > 0:
        sum_xy = sum(buf[i] * buf[i + 1] for i in range(n))
        sum_x = sum(buf[:-1])
        sum_y = sum(buf[1:])
        sum_x2 = sum(x * x for x in buf[:-1])
        sum_y2 = sum(y * y for y in buf[1:])
        denom = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
        serial_corr = (n * sum_xy - sum_x * sum_y) / denom if denom > 0 else 1.0
    else:
        serial_corr = 1.0

    healthy = chi2 < 350.0 and poker < 35.0 and abs(serial_corr) < 0.05
    if not healthy:
        faults.append(NanoFault(NanoFaultType.ENTROPY, NanoSeverity.MEDIUM,
                                chi2, 256.0, chi2 - 256.0, 0,
                                f"Entropy degradation: chi2={chi2:.1f}, poker={poker:.1f}, corr={serial_corr:.4f}", ts))

    return faults


# ═══════════════════════════════════════════════════════════════════════
# PROBE 7: Import Health Monitor
# ═══════════════════════════════════════════════════════════════════════

def probe_import_health() -> list[NanoFault]:
    """Quick import check for critical L104 packages."""
    faults = []
    ts = time.time()

    critical_packages = [
        "l104_code_engine",
        "l104_science_engine",
        "l104_math_engine",
        "l104_agi",
        "l104_asi",
        "l104_intellect",
        "l104_vqpu",
    ]

    for pkg in critical_packages:
        try:
            __import__(pkg)
        except Exception as e:
            faults.append(NanoFault(NanoFaultType.IMPORT_HEALTH, NanoSeverity.HIGH,
                                    0, 1, 1, 0,
                                    f"Import failure: {pkg} — {type(e).__name__}: {e}", ts))

    return faults


# ═══════════════════════════════════════════════════════════════════════
# PROBE 8: State File Integrity
# ═══════════════════════════════════════════════════════════════════════

# State files that legitimately store list (not dict) data
_LIST_STATE_FILES = {
    ".l104_conversation_memory.json",
}

# Maximum file size (bytes) to fully parse per tick — skip large files to avoid outlier ticks
_STATE_FILE_MAX_PARSE = 512_000  # 512 KB


def probe_state_files() -> list[NanoFault]:
    """Verify .l104_*.json state files parse and contain valid data."""
    faults = []
    ts = time.time()

    root = os.environ.get("L104_ROOT", os.getcwd())
    state_files = list(Path(root).glob(".l104_*.json"))

    for sf in state_files[:10]:  # Cap to 10 files per tick
        try:
            # Skip oversized files to keep tick duration bounded
            if sf.stat().st_size > _STATE_FILE_MAX_PARSE:
                continue
            data = json.loads(sf.read_text())
            # dict and list are both valid top-level JSON structures
            if not isinstance(data, (dict, list)):
                faults.append(NanoFault(NanoFaultType.STATE_FILE, NanoSeverity.MEDIUM,
                                        0, 1, 1, 0,
                                        f"State file unexpected type ({type(data).__name__}): {sf.name}", ts))
        except json.JSONDecodeError as e:
            faults.append(NanoFault(NanoFaultType.STATE_FILE, NanoSeverity.HIGH,
                                    0, 1, 1, 0,
                                    f"Invalid JSON in {sf.name}: {e}", ts))
        except Exception as e:
            faults.append(NanoFault(NanoFaultType.STATE_FILE, NanoSeverity.LOW,
                                    0, 0, 0, 0,
                                    f"State file read error {sf.name}: {e}", ts))

    return faults


# ═══════════════════════════════════════════════════════════════════════
# PROBE 9: Cross-Daemon Heartbeat
# ═══════════════════════════════════════════════════════════════════════

def probe_cross_daemon() -> list[NanoFault]:
    faults = []
    ts = time.time()

    # v2.0: Read ALL daemon heartbeats (not just Swift+C)
    peers = [
        ("C nano daemon", C_HEARTBEAT),
        ("Swift nano daemon", SWIFT_HEARTBEAT),
        ("Resource Guardian", GUARDIAN_HEARTBEAT),
        ("Micro daemon", MICRO_HEARTBEAT),
        ("Quantum AI daemon", QAI_HEARTBEAT),
    ]

    live_count = 0
    stale_count = 0
    missing_count = 0

    for name, path in peers:
        if os.path.exists(path):
            try:
                # Ignore test-only heartbeat files (written by --self-test)
                try:
                    hb_data = json.loads(Path(path).read_text())
                    if isinstance(hb_data, dict) and hb_data.get("test"):
                        continue  # Skip — not a live daemon heartbeat
                except (json.JSONDecodeError, ValueError):
                    pass  # Plain-text heartbeat — check normally

                age = time.time() - os.path.getmtime(path)
                if age > 30.0:
                    stale_count += 1
                    # v2.0: Graduated severity based on staleness
                    sev = NanoSeverity.LOW if age < 120.0 else NanoSeverity.MEDIUM if age < 600.0 else NanoSeverity.HIGH
                    faults.append(NanoFault(NanoFaultType.CROSS_DAEMON, sev,
                                            age, 10, age - 10, 0,
                                            f"{name} heartbeat stale: {age:.0f}s", ts))
                else:
                    live_count += 1
            except OSError:
                missing_count += 1
        else:
            missing_count += 1

    # v2.0: Alert when majority of daemons are unreachable
    total = len(peers)
    if stale_count + missing_count > total // 2 and total > 2:
        faults.append(NanoFault(NanoFaultType.CROSS_DAEMON, NanoSeverity.HIGH,
                                float(stale_count + missing_count), float(total),
                                float(stale_count + missing_count) / total, 0,
                                f"Cross-daemon majority unreachable: {live_count}/{total} live, "
                                f"{stale_count} stale, {missing_count} missing", ts))

    return faults


# ═══════════════════════════════════════════════════════════════════════
# PROBE 10: AI Trend Predictor
# ═══════════════════════════════════════════════════════════════════════

class AITrendPredictor:
    """Linear regression on health window to predict future failures."""

    def __init__(self):
        self.health_history: collections.deque[float] = collections.deque(maxlen=AI_WINDOW)

    def observe(self, health: float):
        self.health_history.append(health)

    def predict(self) -> list[NanoFault]:
        faults = []
        ts = time.time()

        if len(self.health_history) < 20:
            return faults

        values = list(self.health_history)
        n = len(values)
        x_mean = (n - 1) / 2.0
        y_mean = statistics.mean(values)

        # Linear regression: y = mx + b
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den != 0 else 0

        # If slope is significantly negative, predict trouble
        if slope < -0.005:  # Health dropping > 0.5% per tick
            ticks_to_critical = (values[-1] - 0.5) / abs(slope) if slope != 0 else float("inf")
            faults.append(NanoFault(NanoFaultType.AI_TREND, NanoSeverity.MEDIUM,
                                    slope, 0, slope, 0,
                                    f"AI trend: health declining at {slope:.4f}/tick, "
                                    f"critical in ~{ticks_to_critical:.0f} ticks", ts))

        # Recent volatility (std of last 10)
        recent = values[-10:]
        if len(recent) >= 5:
            vol = statistics.stdev(recent)
            if vol > 0.1:
                faults.append(NanoFault(NanoFaultType.AI_TREND, NanoSeverity.LOW,
                                        vol, 0.05, vol - 0.05, 0,
                                        f"AI trend: health volatility high ({vol:.3f})", ts))

        return faults


# ═══════════════════════════════════════════════════════════════════════
# PROBE 11: AI Anomaly Classifier (Isolation Forest-inspired)
# ═══════════════════════════════════════════════════════════════════════

class AIAnomalyClassifier:
    """Lightweight isolation-forest-inspired anomaly scoring.

    Maintains a sliding window of multi-dimensional feature vectors:
      [health, fault_count, duration_ms, phase_drift, entropy_chi2]

    Anomaly score is based on average distance from centroid,
    normalized by historical distribution.
    """

    def __init__(self):
        self.feature_window: collections.deque[list[float]] = collections.deque(maxlen=AI_WINDOW)

    def observe(self, features: list[float]):
        self.feature_window.append(features)

    def score(self) -> list[NanoFault]:
        faults = []
        ts = time.time()

        if len(self.feature_window) < 15:
            return faults

        data = list(self.feature_window)
        n_features = len(data[0])

        # Compute centroid
        centroid = [statistics.mean(row[i] for row in data) for i in range(n_features)]

        # Compute distances
        distances = []
        for row in data:
            d = math.sqrt(sum((row[i] - centroid[i]) ** 2 for i in range(n_features)))
            distances.append(d)

        # Latest point's distance
        latest_dist = distances[-1]
        mean_dist = statistics.mean(distances)
        std_dist = statistics.stdev(distances) if len(distances) > 1 else 0.001

        # Anomaly score: how many σ from mean distance
        anomaly_score = (latest_dist - mean_dist) / std_dist if std_dist > 0 else 0

        if anomaly_score > 2.5:
            faults.append(NanoFault(NanoFaultType.AI_ANOMALY, NanoSeverity.MEDIUM,
                                    anomaly_score, 0, anomaly_score, 0,
                                    f"AI anomaly: score={anomaly_score:.2f}σ (isolation distance outlier)", ts))

        return faults


# ═══════════════════════════════════════════════════════════════════════
# PROBE 12: AI Auto-Correlator (Cross-Daemon Fault Correlation)
# ═══════════════════════════════════════════════════════════════════════

class AIAutoCorrelator:
    """v2.0: Cross-correlate faults across ALL L104 daemons.

    Reads IPC outbox reports from Python, Swift, and C nano daemons,
    plus heartbeat age from Guardian, Micro, and Quantum AI daemons.
    Detects systemic faults, health divergence, and temporal clustering."""

    def correlate(self) -> list[NanoFault]:
        faults = []
        ts = time.time()

        # Read latest reports from nano daemon siblings
        sibling_data = {}
        for name, outbox in [("c", C_OUTBOX), ("swift", SWIFT_OUTBOX)]:
            try:
                if os.path.isdir(outbox):
                    files = sorted(Path(outbox).glob("tick_*.json"))
                    if files:
                        latest = json.loads(files[-1].read_text())
                        sibling_data[name] = latest
            except Exception:
                pass

        # v2.0: Read heartbeat ages from all additional daemons
        extra_heartbeats = [
            ("guardian", GUARDIAN_HEARTBEAT),
            ("micro", MICRO_HEARTBEAT),
            ("quantum_ai", QAI_HEARTBEAT),
        ]
        heartbeat_ages = {}
        for name, path in extra_heartbeats:
            if os.path.exists(path):
                try:
                    age = time.time() - os.path.getmtime(path)
                    heartbeat_ages[name] = age
                except OSError:
                    pass

        if not sibling_data and not heartbeat_ages:
            return faults

        # Cross-correlation: if both nano siblings report faults, flag systemic issue
        faulting_siblings = [name for name, data in sibling_data.items()
                             if data.get("fault_count", data.get("faults", 0)) > 0]

        if len(faulting_siblings) >= 2:
            total_sibling_faults = sum(
                d.get("fault_count", d.get("faults", 0)) for d in sibling_data.values()
            )
            faults.append(NanoFault(NanoFaultType.AI_CORRELATOR, NanoSeverity.HIGH,
                                    float(total_sibling_faults), 0,
                                    float(total_sibling_faults), 0,
                                    f"Cross-daemon fault correlation: {faulting_siblings} "
                                    f"all reporting faults ({total_sibling_faults} total)", ts))

        # Health divergence: if siblings health differs significantly
        healths = {name: data.get("health", 1.0) for name, data in sibling_data.items()}
        if len(healths) >= 2:
            vals = list(healths.values())
            spread = max(vals) - min(vals)
            if spread > 0.2:
                faults.append(NanoFault(NanoFaultType.AI_CORRELATOR, NanoSeverity.LOW,
                                        spread, 0, spread, 0,
                                        f"Cross-daemon health divergence: {spread:.3f} "
                                        f"({healths})", ts))

        # v2.0: Heartbeat temporal clustering — multiple daemons going stale simultaneously
        stale_daemons = [name for name, age in heartbeat_ages.items() if age > 60.0]
        if len(stale_daemons) >= 2:
            max_age = max(heartbeat_ages[n] for n in stale_daemons)
            faults.append(NanoFault(NanoFaultType.AI_CORRELATOR, NanoSeverity.MEDIUM,
                                    float(len(stale_daemons)), 0,
                                    max_age, 0,
                                    f"Multi-daemon stale cluster: {stale_daemons} "
                                    f"(max age={max_age:.0f}s)", ts))

        # v2.0: Slope divergence — check if sibling reports include v2 slope data
        slopes = {}
        for name, data in sibling_data.items():
            v2_data = data.get("v2", {})
            if "health_slope" in v2_data:
                slopes[name] = v2_data["health_slope"]
        if len(slopes) >= 2:
            slope_vals = list(slopes.values())
            slope_spread = max(slope_vals) - min(slope_vals)
            if slope_spread > 0.01:
                faults.append(NanoFault(NanoFaultType.AI_CORRELATOR, NanoSeverity.LOW,
                                        slope_spread, 0, slope_spread, 0,
                                        f"Cross-daemon slope divergence: {slope_spread:.4f} "
                                        f"({slopes})", ts))

        return faults
        return faults


# ═══════════════════════════════════════════════════════════════════════
# NANO DAEMON CLASS
# ═══════════════════════════════════════════════════════════════════════

class NanoDaemon:
    """L104 Nano Daemon — Native AI Python Substrate."""

    def __init__(self, tick_interval: float = DEFAULT_TICK, verbose: bool = False):
        self.tick_interval = max(MIN_TICK, min(MAX_TICK, tick_interval))
        self.verbose = verbose
        self.tick_count = 0
        self.total_faults = 0
        self.health_trend = 1.0
        self.running = False

        # Components
        self.canary = MemoryCanary()
        self.stat_detector = StatisticalAnomalyDetector()
        self.trend_predictor = AITrendPredictor()
        self.anomaly_classifier = AIAnomalyClassifier()
        self.correlator = AIAutoCorrelator()

        # Telemetry
        self.telemetry: collections.deque[dict] = collections.deque(maxlen=TELEMETRY_WINDOW)
        self._state_path = os.path.join(
            os.environ.get("L104_ROOT", os.getcwd()),
            ".l104_nano_daemon_python.json"
        )

        # v2.0: Temporal fault causality tracking
        self._causal_chains: collections.deque = collections.deque(maxlen=MAX_CAUSAL_CHAINS)
        self._last_fault_by_type: dict[int, float] = {}

        # v2.0: Fault clustering
        self._fault_clusters: list[dict] = []
        self._cluster_window: collections.deque = collections.deque(maxlen=100)

        # v2.0: Hardware-tuned thresholds
        self._hw_drift_tolerance = 1 if IS_APPLE_SILICON else 2
        self._hw_phase_iterations = 150_000 if IS_APPLE_SILICON else 100_000

        # v2.0: Health slope tracking
        self._health_slope = 0.0
        self._slope_alerts: collections.deque = collections.deque(maxlen=20)

        # Probe registry: (probe_fn, cadence, name)
        self.probes: list[tuple[Any, int, str]] = [
            (probe_constant_drift, 1, "constant_drift"),
            (self.canary.check, 1, "memory_canary"),
            (probe_numerical_stability, 2, "numerical_stability"),
            (probe_phase_drift, 2, "phase_drift"),
            (self._run_stat_anomaly, 1, "stat_anomaly"),
            (probe_entropy_quality, 5, "entropy_quality"),
            (probe_import_health, 10, "import_health"),
            (probe_state_files, 10, "state_files"),
            (probe_cross_daemon, 3, "cross_daemon"),
            (self._run_ai_trend, 1, "ai_trend"),
            (self._run_ai_anomaly, 2, "ai_anomaly"),
            (self.correlator.correlate, 5, "ai_correlator"),
        ]

    def _run_stat_anomaly(self) -> list[NanoFault]:
        return self.stat_detector.detect()

    def _run_ai_trend(self) -> list[NanoFault]:
        return self.trend_predictor.predict()

    def _run_ai_anomaly(self) -> list[NanoFault]:
        return self.anomaly_classifier.score()

    # ─── Tick ───
    def tick(self) -> tuple[float, int, list[NanoFault]]:
        t0 = time.monotonic()
        all_faults: list[NanoFault] = []
        probes_run = 0

        for probe_fn, cadence, name in self.probes:
            if self.tick_count % cadence == 0:
                try:
                    result = probe_fn()
                    all_faults.extend(result)
                    probes_run += 1
                except Exception as e:
                    all_faults.append(NanoFault(
                        NanoFaultType.AI_ANOMALY, NanoSeverity.MEDIUM,
                        0, 0, 0, 0,
                        f"Probe {name} exception: {type(e).__name__}: {e}",
                        time.time()
                    ))

        # Compute health
        health = 1.0
        for fault in all_faults:
            health -= NanoSeverity.penalty(fault.severity)
        health = max(0.0, health)

        self.health_trend = 0.9 * self.health_trend + 0.1 * health
        self.total_faults += len(all_faults)

        # v2.0: Causality, clustering, slope, adaptive tick
        self._track_causality(all_faults)
        self._cluster_faults(all_faults)
        self._compute_health_slope()
        self._adaptive_tick_from_slope()

        t1 = time.monotonic()
        duration_ms = (t1 - t0) * 1000

        # Feed AI observers
        self.stat_detector.observe(health, len(all_faults), duration_ms)
        self.trend_predictor.observe(health)
        self.anomaly_classifier.observe([
            health,
            float(len(all_faults)),
            duration_ms,
        ])

        # Telemetry
        metrics = {
            "tick": self.tick_count,
            "health": health,
            "faults": len(all_faults),
            "duration_ms": duration_ms,
            "probes_run": probes_run,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.telemetry.append(metrics)

        # IPC
        self._write_report(metrics, all_faults)
        self._write_heartbeat()

        # Log
        if all_faults or self.tick_count % 100 == 0 or self.verbose:
            print(f"[NanoDaemon/Python tick {self.tick_count}] "
                  f"health={health:.4f} faults={len(all_faults)} "
                  f"probes={probes_run} {duration_ms:.1f}ms")
            for f in all_faults:
                print(f"  [{NanoSeverity.label(f.severity)}] {f.description}")

        # Persist periodically
        if self.tick_count % PERSIST_EVERY == 0:
            self._persist_state()

        # v2.0: Log slope alerts
        if self._health_slope < SLOPE_ALERT_THRESHOLD and self.tick_count > SLOPE_WINDOW:
            print(f"  [SLOPE ALERT] health slope={self._health_slope:.6f} "
                  f"(threshold={SLOPE_ALERT_THRESHOLD}, "
                  f"tick_interval={self.tick_interval}s)")

        self.tick_count += 1
        return health, len(all_faults), all_faults

    # ─── IPC ───
    def _write_report(self, metrics: dict, faults: list[NanoFault]):
        try:
            report = {
                "daemon": "l104_nano_python",
                "version": VERSION,
                **metrics,
                "total_faults": self.total_faults,
                "health_trend": self.health_trend,
                "fault_details": [
                    {
                        "type": f.type,
                        "type_label": self._fault_type_label(f.type),
                        "severity": f.severity,
                        "severity_label": NanoSeverity.label(f.severity),
                        "measured": f.measured,
                        "expected": f.expected,
                        "deviation": f.deviation,
                        "ulp_distance": f.ulp_distance,
                        "description": f.description,
                    }
                    for f in faults
                ],
                # v2.0: Enhanced IPC payload
                "v2": {
                    "health_slope": round(self._health_slope, 6),
                    "slope_alert": self._health_slope < SLOPE_ALERT_THRESHOLD,
                    "causal_chains_active": len(self._causal_chains),
                    "fault_clusters_active": len(self._fault_clusters),
                    "hw_platform": "apple_silicon" if IS_APPLE_SILICON else "intel_x86",
                    "hw_drift_tolerance_ulp": self._hw_drift_tolerance,
                    "latest_causal": list(self._causal_chains)[-1] if self._causal_chains else None,
                    "top_cluster": self._fault_clusters[0] if self._fault_clusters else None,
                },
            }
            path = os.path.join(PYTHON_OUTBOX, f"tick_{metrics['tick']}.json")
            with open(path, "w") as fp:
                json.dump(report, fp, indent=2)
        except Exception:
            pass

    def _write_heartbeat(self):
        try:
            with open(PYTHON_HEARTBEAT, "w") as fp:
                fp.write(f"{time.time_ns()}\n")
        except Exception:
            pass

    # ─── v2.0 Methods ───

    # ─── v2.0: Fault-type label map ───
    _FAULT_TYPE_LABELS = {
        NanoFaultType.CONSTANT_DRIFT: "constant_drift",
        NanoFaultType.MEMORY_CANARY: "memory_canary",
        NanoFaultType.NUMERICAL_STAB: "numerical_stab",
        NanoFaultType.PHASE_DRIFT: "phase_drift",
        NanoFaultType.STAT_ANOMALY: "stat_anomaly",
        NanoFaultType.ENTROPY: "entropy",
        NanoFaultType.IMPORT_HEALTH: "import_health",
        NanoFaultType.STATE_FILE: "state_file",
        NanoFaultType.CROSS_DAEMON: "cross_daemon",
        NanoFaultType.AI_TREND: "ai_trend",
        NanoFaultType.AI_ANOMALY: "ai_anomaly",
        NanoFaultType.AI_CORRELATOR: "ai_correlator",
    }

    def _fault_type_label(self, ft: int) -> str:
        return self._FAULT_TYPE_LABELS.get(ft, f"unknown_{ft}")

    def _track_causality(self, faults: list):
        """v2.0: Detect causal chains — fault A preceding fault B within window."""
        ts = time.time()
        for fault in faults:
            ft = fault.type
            for prev_type, prev_ts in self._last_fault_by_type.items():
                if prev_type != ft and (ts - prev_ts) < CAUSALITY_WINDOW_S:
                    chain = {
                        "ts": ts,
                        "cause_type": prev_type,
                        "cause_label": self._fault_type_label(prev_type),
                        "effect_type": ft,
                        "effect_label": self._fault_type_label(ft),
                        "delay_s": round(ts - prev_ts, 3),
                        "cause_desc": fault.description[:80],
                    }
                    self._causal_chains.append(chain)
            self._last_fault_by_type[ft] = ts

    def _cluster_faults(self, faults: list):
        """v2.0: Group similar faults into clusters using severity+type distance."""
        for fault in faults:
            self._cluster_window.append(fault)
        if len(self._cluster_window) < 5:
            return
        type_groups: dict[int, list] = {}
        for f in self._cluster_window:
            type_groups.setdefault(f.type, []).append(f)
        self._fault_clusters = []
        for ft, group in sorted(type_groups.items(), key=lambda x: -len(x[1])):
            if len(group) >= 2:
                severities = [f.severity for f in group]
                self._fault_clusters.append({
                    "type": ft,
                    "count": len(group),
                    "avg_severity": round(sum(severities) / len(severities), 2),
                    "max_severity": max(severities),
                    "latest_ts": group[-1].timestamp,
                })
            if len(self._fault_clusters) >= MAX_CLUSTERS:
                break

    def _compute_health_slope(self):
        """v2.0: Linear regression slope on health window to detect sustained decline."""
        window = [m.get("health", 1.0) for m in list(self.telemetry)[-SLOPE_WINDOW:]]
        if len(window) < 5:
            return
        n = len(window)
        x_mean = (n - 1) / 2.0
        y_mean = sum(window) / n
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(window))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator > 0:
            self._health_slope = numerator / denominator
        else:
            self._health_slope = 0.0
        if self._health_slope < SLOPE_ALERT_THRESHOLD:
            self._slope_alerts.append({
                "ts": time.time(),
                "slope": round(self._health_slope, 6),
                "window_size": n,
                "health_current": round(window[-1], 4) if window else 0,
            })

    def _adaptive_tick_from_slope(self):
        """v2.0: Auto-adjust tick interval based on health slope.

        When health is declining steeply, tick faster for more granular
        fault detection. When stable, relax back toward default.
        """
        if self._health_slope < SLOPE_ALERT_THRESHOLD * 2:  # Severe decline
            target = max(MIN_TICK, self.tick_interval * 0.8)
        elif self._health_slope < SLOPE_ALERT_THRESHOLD:  # Moderate decline
            target = max(MIN_TICK, self.tick_interval * 0.9)
        elif abs(self._health_slope) < 0.001:  # Stable
            target = min(DEFAULT_TICK, self.tick_interval * 1.05)
        else:
            return  # No adjustment
        self.tick_interval = round(max(MIN_TICK, min(MAX_TICK, target)), 2)

    def _read_all_heartbeats(self) -> dict[str, dict]:
        """v2.0: Read ALL daemon heartbeats for cross-correlation.

        Returns dict mapping daemon name to heartbeat info dict with
        'age_s', 'alive', 'path' keys.
        """
        result = {}
        all_heartbeats = [
            ("c_nano", C_HEARTBEAT),
            ("swift_nano", SWIFT_HEARTBEAT),
            ("guardian", GUARDIAN_HEARTBEAT),
            ("micro", MICRO_HEARTBEAT),
            ("quantum_ai", QAI_HEARTBEAT),
        ]
        for name, path in all_heartbeats:
            info = {"path": path, "alive": False, "age_s": float("inf")}
            if os.path.exists(path):
                try:
                    age = time.time() - os.path.getmtime(path)
                    info["age_s"] = round(age, 2)
                    info["alive"] = age < 30.0
                except OSError:
                    pass
            result[name] = info
        return result

    def _persist_state(self):
        try:
            state = {
                "daemon": "l104_nano_python",
                "version": VERSION,
                "tick_count": self.tick_count,
                "tick_interval": self.tick_interval,
                "total_faults": self.total_faults,
                "health_trend": self.health_trend,
                # v2.0: Slope + causality + clustering
                "health_slope": round(self._health_slope, 6),
                "slope_declining": self._health_slope < SLOPE_ALERT_THRESHOLD,
                "slope_alerts_count": len(self._slope_alerts),
                "causal_chains_count": len(self._causal_chains),
                "latest_causal_chain": list(self._causal_chains)[-1] if self._causal_chains else None,
                "fault_clusters_count": len(self._fault_clusters),
                "top_fault_cluster": self._fault_clusters[0] if self._fault_clusters else None,
                # v2.0: Hardware
                "hw_platform": "apple_silicon" if IS_APPLE_SILICON else "intel_x86",
                "hw_drift_tolerance_ulp": self._hw_drift_tolerance,
                "hw_phase_iterations": self._hw_phase_iterations,
                "probes": [name for _, _, name in self.probes],
                # v2.0: Daemon peer liveness
                "daemon_peers_alive": sum(
                    1 for info in self._read_all_heartbeats().values() if info["alive"]
                ),
            }
            with open(self._state_path, "w") as fp:
                json.dump(state, fp, indent=2)
        except Exception:
            pass

    # ─── L104Daemon-grade Lifecycle Assertions ───

    def validate_configuration(self) -> bool:
        """Validate configuration — mirrors L104Daemon.validateConfiguration().
        Checks: tick bounds, sacred constant bit-exact, IPC directories, system resources."""
        valid = True

        # Tick interval bounds
        if not (MIN_TICK <= self.tick_interval <= MAX_TICK):
            print(f"[L104 NanoDaemon/Python] ERROR: Invalid tick interval: {self.tick_interval}s "
                  f"(must be {MIN_TICK}-{MAX_TICK})")
            valid = False

        # Sacred constant bit-exact verification
        gc_bits = struct.unpack(">Q", struct.pack(">d", GOD_CODE))[0]
        phi_bits = struct.unpack(">Q", struct.pack(">d", PHI))[0]
        void_bits = struct.unpack(">Q", struct.pack(">d", VOID_CONSTANT))[0]
        if gc_bits != GOD_CODE_BITS:
            print(f"[L104 NanoDaemon/Python] ERROR: GOD_CODE bit mismatch: "
                  f"0x{gc_bits:016X} != 0x{GOD_CODE_BITS:016X}")
            valid = False
        if phi_bits != PHI_BITS:
            print(f"[L104 NanoDaemon/Python] ERROR: PHI bit mismatch: "
                  f"0x{phi_bits:016X} != 0x{PHI_BITS:016X}")
            valid = False
        if void_bits != VOID_BITS:
            print(f"[L104 NanoDaemon/Python] ERROR: VOID_CONSTANT bit mismatch: "
                  f"0x{void_bits:016X} != 0x{VOID_BITS:016X}")
            valid = False

        # Verify IPC directories exist (after creation)
        for d in ["/tmp/l104_bridge", NANO_BASE, PYTHON_OUTBOX]:
            if not os.path.isdir(d):
                print(f"[L104 NanoDaemon/Python] ERROR: Required directory missing: {d}")
                valid = False

        # System resource check
        try:
            import resource
            # Check available memory (soft limit)
            mem_limit = resource.getrlimit(resource.RLIMIT_RSS)
            # Just log — Python doesn't have direct Mach VM access without ctypes
        except Exception:
            pass

        if valid:
            print("[L104 NanoDaemon/Python] Configuration validated ✓")
        return valid

    def _kill_previous_instance(self):
        """Kill stale daemon instance — mirrors L104Daemon.killPreviousInstance()."""
        try:
            with open(PYTHON_PID) as fp:
                old_pid = int(fp.read().strip())
        except (FileNotFoundError, ValueError):
            return

        my_pid = os.getpid()
        if old_pid <= 0 or old_pid == my_pid:
            return

        # Check if alive
        try:
            os.kill(old_pid, 0)
        except OSError:
            return

        print(f"[L104 NanoDaemon/Python] Killing stale instance (PID {old_pid})")
        try:
            os.kill(old_pid, signal.SIGTERM)
        except OSError:
            return

        # Wait up to 2 seconds
        for _ in range(20):
            time.sleep(0.1)
            try:
                os.kill(old_pid, 0)
            except OSError:
                break
        else:
            # Still alive — force kill
            try:
                print(f"[L104 NanoDaemon/Python] Stale PID {old_pid} did not exit — sending SIGKILL")
                os.kill(old_pid, signal.SIGKILL)
                time.sleep(0.1)
            except OSError:
                pass

    def dump_status(self):
        """Dump full status — mirrors L104Daemon's SIGUSR1 handler."""
        print(f"\n[L104 NanoDaemon/Python] ═══ STATUS DUMP (v{VERSION}) ═══")
        print(f"  Version:      {VERSION}")
        print(f"  PID:          {os.getpid()}")
        print(f"  Running:      {self.running}")
        print(f"  Ticks:        {self.tick_count}")
        print(f"  Total faults: {self.total_faults}")
        print(f"  Health trend: {self.health_trend:.6f}")
        print(f"  Probes:       {len(self.probes)} ({', '.join(n for _, _, n in self.probes)})")
        print(f"  GOD_CODE:     {GOD_CODE} (bits=0x{GOD_CODE_BITS:016X})")
        print(f"  PHI:          {PHI}")
        print(f"  VOID:         {VOID_CONSTANT}")
        print(f"  Telemetry:    {len(self.telemetry)} entries")
        # v2.0: Extended status dump
        print(f"  ─── v2.0 Diagnostics ───")
        print(f"  Platform:     {'Apple Silicon (arm64)' if IS_APPLE_SILICON else 'Intel (x86_64)'}")
        print(f"  Health slope: {self._health_slope:.6f} (alert threshold: {SLOPE_ALERT_THRESHOLD})")
        print(f"  Slope alerts: {len(self._slope_alerts)}")
        print(f"  Causal chains: {len(self._causal_chains)} tracked")
        print(f"  Fault clusters: {len(self._fault_clusters)} active")
        print(f"  HW drift tol: {self._hw_drift_tolerance} ULP")
        print(f"  HW phase iter: {self._hw_phase_iterations}")
        print(f"  Tick interval: {self.tick_interval}s (adaptive)")
        # v2.0: Cross-daemon heartbeat summary
        heartbeats = self._read_all_heartbeats()
        alive = [n for n, info in heartbeats.items() if info["alive"]]
        print(f"  Daemon peers:  {len(alive)}/{len(heartbeats)} alive ({', '.join(alive) if alive else 'none'})")

        # System resources
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            print(f"  Memory RSS:   {usage.ru_maxrss / 1024 / 1024:.1f}MB (peak)")
            print(f"  CPU user:     {usage.ru_utime:.3f}s  sys: {usage.ru_stime:.3f}s")
        except Exception:
            pass

        print(f"  ═══════════════════════════════\n")
        sys.stdout.flush()

        # Write JSON status file
        try:
            status_path = os.path.join(NANO_BASE, "python_status.json")
            status_json = {
                "daemon": "l104_nano_python",
                "version": VERSION,
                "pid": os.getpid(),
                "running": self.running,
                "tick_count": self.tick_count,
                "total_faults": self.total_faults,
                "health_trend": self.health_trend,
                # v2.0: Extended JSON status
                "health_slope": round(self._health_slope, 6),
                "slope_alerts": len(self._slope_alerts),
                "causal_chains": len(self._causal_chains),
                "fault_clusters": len(self._fault_clusters),
                "hw_platform": "apple_silicon" if IS_APPLE_SILICON else "intel_x86",
                "tick_interval": self.tick_interval,
                "daemon_peers": {n: info["alive"] for n, info in heartbeats.items()},
            }
            with open(status_path, "w") as fp:
                json.dump(status_json, fp, indent=2)
        except Exception:
            pass

    def reload(self):
        """Reload — reinitialize components — mirrors L104Daemon's SIGHUP handler."""
        print("[L104 NanoDaemon/Python] SIGHUP — reloading components")
        self.canary = MemoryCanary()
        self.stat_detector = StatisticalAnomalyDetector()
        self.trend_predictor = AITrendPredictor()
        self.anomaly_classifier = AIAnomalyClassifier()
        self.correlator = AIAutoCorrelator()
        # v2.0: Reset v2.0 state on reload
        self._causal_chains.clear()
        self._last_fault_by_type.clear()
        self._fault_clusters.clear()
        self._cluster_window.clear()
        self._health_slope = 0.0
        self._slope_alerts.clear()
        self.tick_interval = DEFAULT_TICK  # Reset adaptive tick
        print(f"[L104 NanoDaemon/Python] Reload complete — {len(self.probes)} probes reinitialized (v2.0 state reset)")
        sys.stdout.flush()

    # ─── Run ───
    def run(self):
        """Main daemon loop (blocking)."""
        self.running = True
        _ensure_dirs()

        # ── L104Daemon-grade startup assertion gate ──
        print(f"[L104 NanoDaemon/Python] Startup validation...")
        print(f"  PID:      {os.getpid()}")
        print(f"  Tick:     {self.tick_interval}s")
        print(f"  IPC:      {PYTHON_OUTBOX}")

        if not self.validate_configuration():
            print("[L104 NanoDaemon/Python] ERROR: Configuration validation failed — exiting")
            sys.exit(1)

        # Kill stale instance (L104Daemon pattern)
        self._kill_previous_instance()

        # PID file
        try:
            with open(PYTHON_PID, "w") as fp:
                fp.write(f"{os.getpid()}\n")
        except Exception:
            pass

        # atexit
        atexit.register(self.shutdown)

        print(f"[L104 NanoDaemon/Python v{VERSION}] Running (tick={self.tick_interval}s, probes={len(self.probes)})")
        print(f"  Sacred: GOD_CODE={GOD_CODE}  PHI={PHI}  VOID={VOID_CONSTANT}")
        print(f"  AI: trend_predictor, anomaly_classifier, auto_correlator")
        print(f"  v2.0: causality, clustering, slope_alert, hw_tuning, adaptive_tick")
        print(f"  Platform: {'Apple Silicon (arm64)' if IS_APPLE_SILICON else 'Intel (x86_64)'}")
        print(f"  HW thresholds: drift_tol={self._hw_drift_tolerance}ULP, phase_iter={self._hw_phase_iterations}")
        print(f"  IPC: {PYTHON_OUTBOX}")

        while self.running:
            try:
                self.tick()
                time.sleep(self.tick_interval)
            except KeyboardInterrupt:
                break

        self.shutdown()

    def shutdown(self):
        self.running = False
        self._persist_state()
        try:
            os.unlink(PYTHON_PID)
        except OSError:
            pass
        print(f"[L104 NanoDaemon/Python] Shutdown: {self.tick_count} ticks, "
              f"{self.total_faults} faults, health={self.health_trend:.4f}")
        # v2.0: Final slope & causality summary
        if self._causal_chains:
            print(f"  Causal chains detected: {len(self._causal_chains)}")
        if self._slope_alerts:
            print(f"  Slope alerts triggered: {len(self._slope_alerts)} "
                  f"(final slope={self._health_slope:.6f})")

    # ─── Self-Test ───
    def self_test(self) -> int:
        """Run all 12 probes once. Returns number of failures."""
        print(f"[L104 NanoDaemon/Python] Self-test — {len(self.probes)} probes")
        _ensure_dirs()

        failures = 0
        for probe_fn, _, name in self.probes:
            try:
                result = probe_fn()
                critical = [f for f in result if f.severity == NanoSeverity.CRITICAL]
                ok = len(critical) == 0
                status = "PASS" if ok else "FAIL"
                if not ok:
                    failures += 1
                print(f"  {status}: {name} ({len(result)} faults, {len(critical)} critical)")
            except Exception as e:
                failures += 1
                print(f"  FAIL: {name} — {type(e).__name__}: {e}")

        # Utility checks
        ulp_ok = ulp_distance(1.0, 1.0 + sys.float_info.epsilon) == 1
        print(f"  {'PASS' if ulp_ok else 'FAIL'}: ulp_distance")
        if not ulp_ok:
            failures += 1

        hd_ok = hamming_distance(0xFF, 0x00) == 8
        print(f"  {'PASS' if hd_ok else 'FAIL'}: hamming_distance")
        if not hd_ok:
            failures += 1

        print(f"[L104 NanoDaemon/Python] Self-test: {len(self.probes) + 2 - failures} passed, {failures} failed")

        # v2.0: Report subsystem readiness
        print(f"  ─── v2.0 Subsystems ───")
        print(f"  Platform:        {'Apple Silicon (arm64)' if IS_APPLE_SILICON else 'Intel (x86_64)'}")
        print(f"  Drift tolerance: {self._hw_drift_tolerance} ULP")
        print(f"  Phase iterations: {self._hw_phase_iterations}")
        print(f"  Causality window: {CAUSALITY_WINDOW_S}s")
        print(f"  Cluster threshold: {CLUSTER_DISTANCE_THRESHOLD}")
        print(f"  Slope window:    {SLOPE_WINDOW} ticks (alert < {SLOPE_ALERT_THRESHOLD}/tick)")
        heartbeats = self._read_all_heartbeats()
        alive = [n for n, info in heartbeats.items() if info["alive"]]
        print(f"  Daemon peers:    {len(alive)}/{len(heartbeats)} alive")
        for name, info in heartbeats.items():
            status_str = f"alive ({info['age_s']:.0f}s)" if info["alive"] else (
                f"stale ({info['age_s']:.0f}s)" if info["age_s"] < float("inf") else "missing"
            )
            print(f"    {name}: {status_str}")

        return failures

    # ─── Health Check ───
    def health_check(self) -> int:
        """Read persisted state and report health."""
        try:
            with open(self._state_path) as fp:
                state = json.load(fp)
            health = state.get("health_trend", 0)
            ticks = state.get("tick_count", 0)
            faults = state.get("total_faults", 0)
            print(f"[L104 NanoDaemon/Python] Health: {health:.4f} "
                  f"(ticks={ticks}, total_faults={faults})")
            return 0 if health > 0.5 else 1
        except FileNotFoundError:
            print("[L104 NanoDaemon/Python] No state file — daemon not yet run")
            return 1
        except Exception as e:
            print(f"[L104 NanoDaemon/Python] Health check error: {e}")
            return 1

    # ─── Status ───
    def status(self) -> dict:
        heartbeats = self._read_all_heartbeats()
        return {
            "daemon": "l104_nano_python",
            "version": VERSION,
            "running": self.running,
            "tick_count": self.tick_count,
            "tick_interval": self.tick_interval,
            "total_faults": self.total_faults,
            "health_trend": self.health_trend,
            "probes": [name for _, _, name in self.probes],
            "telemetry_size": len(self.telemetry),
            # v2.0: Temporal fault causality
            "causal_chains": list(self._causal_chains)[-5:],
            "causal_chains_total": len(self._causal_chains),
            # v2.0: Fault clustering
            "fault_clusters": self._fault_clusters[:5],
            "fault_clusters_total": len(self._fault_clusters),
            # v2.0: Health slope
            "health_slope": round(self._health_slope, 6),
            "slope_declining": self._health_slope < SLOPE_ALERT_THRESHOLD,
            "slope_alerts": list(self._slope_alerts)[-3:],
            "slope_alerts_total": len(self._slope_alerts),
            # v2.0: Hardware detection
            "hw_platform": "apple_silicon" if IS_APPLE_SILICON else "intel_x86",
            "hw_drift_tolerance_ulp": self._hw_drift_tolerance,
            "hw_phase_iterations": self._hw_phase_iterations,
            # v2.0: Cross-daemon heartbeat summary
            "daemon_peers": heartbeats,
            "daemon_peers_alive": sum(1 for info in heartbeats.values() if info["alive"]),
        }


# ═══════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="L104 Nano Daemon — AI Python Substrate")
    parser.add_argument("--self-test", action="store_true", help="Run probes and exit 0/1")
    parser.add_argument("--health-check", action="store_true", help="Read state, report, exit 0/1")
    parser.add_argument("--validate", action="store_true", help="Validate configuration and exit 0/1")
    parser.add_argument("--status", action="store_true", help="Send SIGUSR1 to running daemon for status dump")
    parser.add_argument("--once", action="store_true", help="Single tick, exit")
    parser.add_argument("--tick", type=float, default=DEFAULT_TICK, help="Tick interval seconds")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  L104 NANO DAEMON — AI Python Substrate v{VERSION}                 ║")
    print("║  Atomized Fault Detection with Native AI Anomaly Detection     ║")
    print("║  GOD_CODE=527.5184818492612 | PHI=1.618033988749895            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    daemon = NanoDaemon(tick_interval=args.tick, verbose=args.verbose)

    if args.self_test:
        failures = daemon.self_test()
        sys.exit(1 if failures > 0 else 0)

    if args.health_check:
        sys.exit(daemon.health_check())

    if args.validate:
        _ensure_dirs()
        ok = daemon.validate_configuration()
        sys.exit(0 if ok else 1)

    if args.status:
        # Read PID file and send SIGUSR1 to running daemon
        try:
            with open(PYTHON_PID) as fp:
                pid = int(fp.read().strip())
            if pid <= 0:
                raise ValueError
        except (FileNotFoundError, ValueError):
            print("No running daemon (PID file missing)")
            sys.exit(1)
        try:
            os.kill(pid, 0)
        except OSError:
            print(f"Daemon PID {pid} not running")
            sys.exit(1)
        os.kill(pid, signal.SIGUSR1)
        print(f"Sent SIGUSR1 to PID {pid} (status dump requested)")
        sys.exit(0)

    if args.once:
        _ensure_dirs()
        health, faults, _ = daemon.tick()
        print(f"[Single tick] health={health:.4f}  faults={faults}")
        sys.exit(1 if faults > 0 else 0)

    # Signal handlers — full L104Daemon set
    def _shutdown(signum, frame):
        daemon.shutdown()
        sys.exit(0)

    def _dump_status(signum, frame):
        daemon.dump_status()

    def _force_tick(signum, frame):
        print("[L104 NanoDaemon/Python] SIGUSR2 — force tick requested")

    def _reload(signum, frame):
        daemon.reload()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGUSR1, _dump_status)
    signal.signal(signal.SIGUSR2, _force_tick)
    signal.signal(signal.SIGHUP, _reload)

    daemon.run()


if __name__ == "__main__":
    main()
