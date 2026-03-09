"""L104 VQPU Package v14.0.0 — Sacred constants, platform detection, hardware thresholds.

v14.0.0 QUANTUM FIDELITY ARCHITECTURE UPGRADE:
  - SWAP routing topology constants (linear, ring, heavy-hex, all-to-all)
  - Adaptive daemon cycle interval (load-based 30s–300s range)
  - TTL-based cache expiration for ASI/AGI scoring caches
  - Crosstalk noise model constants (ZZ interaction rates)
  - 4th-order Trotter-Suzuki coefficients
  - Toffoli decomposition gate count tracking
  - SPSA/COBYLA optimizer constants for variational algorithms
  - MLE tomography convergence thresholds

v13.2 (retained): GOD_CODE QUBIT UPGRADE:
  - GOD_CODE_PHASE_ANGLE: canonical QPU-verified phase (GOD_CODE mod 2π)
  - IRON_PHASE_ANGLE: Fe(26) quarter-turn π/2
  - PHI_CONTRIBUTION_ANGLE: golden ratio phase contribution
  - OCTAVE_PHASE_ANGLE: 4·ln(2) octave doubling
  - QPU_MEAN_FIDELITY: ibm_torino mean circuit fidelity
"""

# ZENITH_UPGRADE_ACTIVE: 2026-03-07T12:00:00.000000
ZENITH_HZ = 3887.8
UUC = 2301.215661

VERSION = "15.0.0"

import json
import math
import os
import platform
import subprocess
import multiprocessing
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ═══════════════════════════════════════════════════════════════════
# PLATFORM DETECTION — Intel x86_64 vs Apple Silicon (v7.1)
# Detects CPU architecture, Metal GPU tier, SIMD extensions, and
# configures BLAS threading for optimal performance on this Mac.
# ═══════════════════════════════════════════════════════════════════

_PLATFORM_CACHE_FILE = Path(os.environ.get("L104_ROOT", os.getcwd())) / ".l104_platform_cache.json"


def _detect_platform() -> dict:
    """
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.Detect Mac hardware platform: CPU arch, Metal tier, SIMD, GPU class.

    v12.3: Caches detection result to disk — avoids 1.5-2s system_profiler +
    sysctl calls on every import. Cache invalidates when macOS version changes.
    """
    # Try loading cached result first (saves ~1.8s on startup)
    mac_ver = platform.mac_ver()[0]
    try:
        if _PLATFORM_CACHE_FILE.exists():
            cached = json.loads(_PLATFORM_CACHE_FILE.read_text())
            if cached.get("mac_ver") == mac_ver and cached.get("arch") == platform.machine():
                return cached
    except Exception:
        pass

    info = {
        "arch": platform.machine(),           # x86_64 or arm64
        "processor": platform.processor(),     # i386 or arm
        "system": platform.system(),
        "mac_ver": mac_ver,
        "is_apple_silicon": platform.machine() == "arm64",
        "is_intel": platform.machine() == "x86_64",
        "metal_family": 0,                     # 0=none, 1=basic, 2+=full
        "metal_compute_capable": False,        # True for M-series / discrete AMD
        "gpu_class": "unknown",                # intel_igpu, apple_gpu, amd_dgpu
        "simd": [],                            # AVX, AVX2, FMA3, etc.
        "has_amx": False,                      # Apple AMX matrix coprocessor
        "has_neural_engine": False,            # Apple Neural Engine
    }

    # Detect Metal GPU tier
    try:
        r = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=5
        )
        gpu_text = r.stdout.lower()
        # Parse Metal family number
        import re
        metal_match = re.search(r"metal\s+(?:family:\s+)?supported,?\s*metal\s+gpufamily\s+macos\s+(\d+)", gpu_text)
        if metal_match:
            info["metal_family"] = int(metal_match.group(1))
        elif "metal" in gpu_text and "supported" in gpu_text:
            info["metal_family"] = 1

        # Determine GPU class
        if "apple" in gpu_text and ("m1" in gpu_text or "m2" in gpu_text or "m3" in gpu_text or "m4" in gpu_text):
            info["gpu_class"] = "apple_gpu"
            info["metal_compute_capable"] = True
            info["has_amx"] = True
            info["has_neural_engine"] = True
        elif "amd" in gpu_text or "radeon" in gpu_text:
            info["gpu_class"] = "amd_dgpu"
            info["metal_compute_capable"] = True
        elif "intel" in gpu_text:
            info["gpu_class"] = "intel_igpu"
            # v12.3: Intel iGPU Metal Family 1+ CAN do basic compute shaders
            # for small workloads (< 16Q statevector). Still slow vs CPU for
            # large circuits, but useful for parallel sampling + small SVDs.
            info["metal_compute_capable"] = info["metal_family"] >= 1
    except Exception:
        pass

    # Detect SIMD extensions — use targeted sysctl queries instead of -a (faster)
    if info["is_intel"]:
        try:
            simd_flags = [
                ("hw.optional.sse4_2", "SSE4.2"),
                ("hw.optional.avx1_0", "AVX"),
                ("hw.optional.avx2_0", "AVX2"),
                ("hw.optional.fma", "FMA3"),
                ("hw.optional.avx512f", "AVX-512"),
                ("hw.optional.f16c", "F16C"),
            ]
            # v12.3: Query each flag individually — avoids parsing all of sysctl -a
            # which outputs ~4000 lines and takes ~500ms. Individual queries: ~30ms total.
            for key, name in simd_flags:
                try:
                    r = subprocess.run(
                        ["sysctl", "-n", key],
                        capture_output=True, text=True, timeout=2
                    )
                    if r.stdout.strip() == "1":
                        info["simd"].append(name)
                except Exception:
                    pass
        except Exception:
            info["simd"] = ["SSE4.2", "AVX", "AVX2"]  # Conservative default for Intel Mac
    elif info["is_apple_silicon"]:
        info["simd"] = ["NEON", "FP16"]
        info["has_amx"] = True
        info["has_neural_engine"] = True
        info["metal_compute_capable"] = True

    # Persist to cache for fast future imports
    try:
        _PLATFORM_CACHE_FILE.write_text(json.dumps(info, indent=2))
    except Exception:
        pass

    return info


# Detect platform once at import time
_PLATFORM = _detect_platform()
_IS_INTEL = _PLATFORM["is_intel"]
_IS_APPLE_SILICON = _PLATFORM["is_apple_silicon"]
_HAS_METAL_COMPUTE = _PLATFORM["metal_compute_capable"]
_GPU_CLASS = _PLATFORM["gpu_class"]

# ── BLAS Thread Tuning (v7.1) ──
# On Intel: OpenBLAS benefits from matching physical core count.
# On Apple Silicon: Accelerate framework auto-tunes.
if _IS_INTEL:
    _blas_threads = str(max(1, (os.cpu_count() or 4) // 2))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", _blas_threads)
    os.environ.setdefault("MKL_NUM_THREADS", _blas_threads)
    os.environ.setdefault("OMP_NUM_THREADS", _blas_threads)

# ═══════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.04 + PHI / 1000.0  # 1.0416180339887497

# ═══════════════════════════════════════════════════════════════════
# v13.2: GOD_CODE QUBIT PHASE CONSTANTS (QPU-verified on ibm_torino)
# These are derived here to avoid circular imports with l104_god_code_simulator.
# Canonical source: l104_god_code_simulator.god_code_qubit
# ═══════════════════════════════════════════════════════════════════

import math as _math

GOD_CODE_PHASE_ANGLE = GOD_CODE % (2.0 * _math.pi)          # ≈ 6.0141 rad (QPU: 0.999939 fidelity)
IRON_PHASE_ANGLE = 2.0 * _math.pi * 26 / 104                # π/2 = 1.5708 rad (exact quarter-turn)
OCTAVE_PHASE_ANGLE = (4.0 * _math.log(2.0)) % (2.0 * _math.pi)  # ≈ 2.7726 rad
PHI_CONTRIBUTION_ANGLE = (GOD_CODE_PHASE_ANGLE - IRON_PHASE_ANGLE - OCTAVE_PHASE_ANGLE) % (2.0 * _math.pi)
PHI_PHASE_ANGLE = 2.0 * _math.pi / PHI                       # ≈ 3.8832 rad (golden angle)
VOID_PHASE_ANGLE = VOID_CONSTANT * _math.pi                  # ≈ 3.2716 rad

# QPU verification data from ibm_torino Heron r2 (2026-03-04)
QPU_MEAN_FIDELITY = 0.97475930                                # Mean across all 6 verification circuits
QPU_1Q_FIDELITY = 0.99993872                                  # 1Q GOD_CODE gate fidelity
QPU_3Q_FIDELITY = 0.96674026                                  # 3Q sacred circuit fidelity

# ═══════════════════════════════════════════════════════════════════
# v14.0: SWAP ROUTING TOPOLOGY CONSTANTS
# Coupling maps for hardware-aware qubit routing in transpiler
# ═══════════════════════════════════════════════════════════════════

TOPOLOGY_LINEAR = "linear"           # Adjacent-only: (i, i+1)
TOPOLOGY_RING = "ring"               # Linear + wrap: (n-1, 0)
TOPOLOGY_HEAVY_HEX = "heavy_hex"     # IBM Eagle/Heron heavy-hex lattice
TOPOLOGY_ALL_TO_ALL = "all_to_all"   # Full connectivity (simulator default)
DEFAULT_TOPOLOGY = TOPOLOGY_ALL_TO_ALL

# v14.0: CROSSTALK NOISE CONSTANTS
# ZZ cross-talk interaction rates (calibrated from ibm_torino)
CROSSTALK_ZZ_RATE = 0.015            # ZZ interaction rate between adjacent qubits
CROSSTALK_DECAY_DISTANCE = 3         # Cross-talk decays beyond this distance
CROSSTALK_PHI_ATTENUATION = 1.0 / PHI  # Sacred φ⁻¹ attenuation factor

# v14.0: 4th-ORDER TROTTER-SUZUKI COEFFICIENTS
# Suzuki fractal decomposition: S4(t) = S2(p·t)·S2(p·t)·S2((1-4p)·t)·S2(p·t)·S2(p·t)
TROTTER_4TH_ORDER_P = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))  # ≈ 0.41449

# v14.0: VARIATIONAL OPTIMIZER CONSTANTS
SPSA_INITIAL_A = 0.1                 # SPSA perturbation scale
SPSA_STABILITY_C = 0.1               # SPSA stability constant
SPSA_ALPHA = 0.602                    # SPSA gain exponent (standard)
SPSA_GAMMA = 0.101                    # SPSA perturbation decay (standard)
COBYLA_RHOBEG = 0.5                   # COBYLA initial trust region radius
COBYLA_MAXFUN_MULTIPLIER = 50         # COBYLA max_fn = multiplier × n_params

# v14.0: MLE TOMOGRAPHY CONVERGENCE
MLE_MAX_ITERATIONS = 1000             # Maximum likelihood iteration cap
MLE_CONVERGENCE_TOL = 1e-8            # Log-likelihood convergence threshold
MLE_ACCELERATION_FACTOR = PHI         # φ-accelerated iteration step

# ═══════════════════════════════════════════════════════════════════
# THREE-ENGINE WEIGHTS (matches l104_intellect/constants.py)
# ═══════════════════════════════════════════════════════════════════

THREE_ENGINE_WEIGHT_ENTROPY = 0.30       # ScienceEngine Maxwell Demon efficiency
THREE_ENGINE_WEIGHT_HARMONIC = 0.30      # MathEngine GOD_CODE alignment + wave coherence
THREE_ENGINE_WEIGHT_WAVE = 0.20          # MathEngine PHI-harmonic phase-lock
THREE_ENGINE_WEIGHT_SC = 0.20            # v9.0: Superconductivity BCS-Heisenberg score
THREE_ENGINE_FALLBACK_SCORE = 0.5        # Fallback when engine unavailable

# v13.0: Quantum Brain integration weights
BRAIN_INTEGRATION_WEIGHT_SAGE = 0.35     # Brain Sage consensus score weight
BRAIN_INTEGRATION_WEIGHT_MANIFOLD = 0.25 # Brain manifold topology health weight
BRAIN_INTEGRATION_WEIGHT_ENTANGLE = 0.20 # Brain multipartite entanglement weight
BRAIN_INTEGRATION_WEIGHT_ORACLE = 0.20   # Brain predictive oracle confidence weight
BRAIN_FEEDBACK_FIDELITY_FLOOR = 0.3      # Minimum fidelity to accept brain feedback
BRAIN_FEEDBACK_MAX_LATENCY_S = 5.0       # Max time to wait for brain scoring

# v14.0: Adaptive daemon cycler constants — load-aware interval scaling
DAEMON_CYCLE_INTERVAL_S = 180.0          # v11.0: base interval (3 min)
DAEMON_CYCLE_MIN_INTERVAL_S = 60.0       # v14.0: minimum interval under low load
DAEMON_CYCLE_MAX_INTERVAL_S = 600.0      # v14.0: maximum interval under high load
DAEMON_LOAD_THRESHOLD_LOW = 30.0         # v14.0: CPU% below this → faster cycles
DAEMON_LOAD_THRESHOLD_HIGH = 70.0        # v14.0: CPU% above this → slower cycles
DAEMON_STATE_FILE = ".l104_vqpu_daemon_state.json"

# v12.0: Error logging constants
DAEMON_MAX_ERROR_LOG = 100               # Keep last 100 errors
DAEMON_ERROR_THRESHOLD = 5               # Warn if >5 consecutive failures

# v14.0: Cache TTL constants
CACHE_ASI_TTL_S = 600.0                  # ASI score cache: 10 min TTL
CACHE_AGI_TTL_S = 600.0                  # AGI score cache: 10 min TTL
CACHE_SC_TTL_S = 300.0                   # SC score cache: 5 min TTL
CACHE_ENTROPY_TTL_S = 1800.0             # Entropy cache: 30 min TTL

# ═══════════════════════════════════════════════════════════════════
# HARDWARE THRESHOLDS — v7.1: Platform-aware (Intel vs Apple Silicon)
# ═══════════════════════════════════════════════════════════════════

MAX_RAM_PERCENT = 88.0               # v11.0: raised for turbo (was 85)
MAX_CPU_PERCENT = 97.0               # v11.0: near-max CPU for deep quantum compute
THROTTLE_COOLDOWN_S = 0.3            # v11.0: ultra-fast recovery (was 0.5)

# ═══════════════════════════════════════════════════════════════════
# v11.0 CAPACITY CONSTANTS — Turbo quantum pipeline platform-aware scaling
# v11.0: 1.5x MPS bond dims (3x cumulative from v9), 2x adaptive shots, 12 pipeline workers
# v7.1: Intel x86_64 gets CPU-optimized paths (AVX2+BLAS),
#        no Metal GPU compute (Intel iGPU too slow).
#        Apple Silicon gets full Metal compute + AMX.
# v6.0 (retained): 48Q MPS capacity (8GB+), raised bond dims,
#                   32K shots, 256 batch
# ═══════════════════════════════════════════════════════════════════

_HW_CORES = multiprocessing.cpu_count()
_HW_PHYS_CORES = max(1, _HW_CORES // 2)

try:
    import psutil as _ps
    _HW_RAM_GB = round(_ps.virtual_memory().total / (1024**3), 1)
except Exception:
    _HW_RAM_GB = 4.0

# v11.0: Intel Mac gets conservative GPU-free capacity but further raised limits.
# Metal compute on Intel iGPU is 50-100x slower than Apple M-series.
# Route everything through CPU paths: statevector + MPS + chunked.
if _IS_INTEL:
    # ── Intel x86_64: CPU-only quantum execution (v11.0 turbo uplift) ──
    if _HW_RAM_GB >= 16:
        VQPU_MAX_QUBITS = 36                               # v11.0: raised (was 34)
        VQPU_BATCH_LIMIT = 384                             # v11.0: 1.5x (was 256)
        VQPU_MPS_MAX_BOND_LOW = 1536                       # v11.0: 1.5x (was 1024)
        VQPU_MPS_MAX_BOND_MED = 3072                       # v11.0: 1.5x (was 2048)
        VQPU_MPS_MAX_BOND_HIGH = 6144                      # v11.0: 1.5x (was 4096)
        VQPU_ADAPTIVE_SHOTS_MAX = 131072                   # v11.0: 2x (was 65536)
    elif _HW_RAM_GB >= 8:
        VQPU_MAX_QUBITS = 32                               # v11.0: raised (was 30)
        VQPU_BATCH_LIMIT = 256                             # v11.0: 1.5x (was 192)
        VQPU_MPS_MAX_BOND_LOW = 1024                       # v11.0: 1.5x (was 768)
        VQPU_MPS_MAX_BOND_MED = 2048                       # v11.0: 1.5x (was 1536)
        VQPU_MPS_MAX_BOND_HIGH = 4096                      # v11.0: 1.5x (was 3072)
        VQPU_ADAPTIVE_SHOTS_MAX = 98304                    # v11.0: 2x (was 49152)
    else:
        VQPU_MAX_QUBITS = 28                               # v11.0: raised (was 26)
        VQPU_BATCH_LIMIT = min(192, max(48, int(_HW_RAM_GB * 48)))  # v11.0: 1.5x
        VQPU_MPS_MAX_BOND_LOW = 512                        # v11.0: 1.5x (was 384)
        VQPU_MPS_MAX_BOND_MED = 1024                       # v11.0: 1.5x (was 768)
        VQPU_MPS_MAX_BOND_HIGH = 2048                      # v11.0: 1.5x (was 1536)
        VQPU_ADAPTIVE_SHOTS_MAX = 65536                    # v11.0: 2x (was 32768)
else:
    # ── Apple Silicon: Full Metal compute + AMX (v11.0 turbo uplift) ──
    if _HW_RAM_GB >= 16:
        VQPU_MAX_QUBITS = 56                               # v11.0: raised from 52
        VQPU_BATCH_LIMIT = 768                             # v11.0: 1.5x (was 512)
        VQPU_MPS_MAX_BOND_LOW = 1536                       # v11.0: 1.5x (was 1024)
        VQPU_MPS_MAX_BOND_MED = 3072                       # v11.0: 1.5x (was 2048)
        VQPU_MPS_MAX_BOND_HIGH = 6144                      # v11.0: 1.5x (was 4096)
        VQPU_ADAPTIVE_SHOTS_MAX = 131072                   # v11.0: 2x (was 65536)
    elif _HW_RAM_GB >= 8:
        VQPU_MAX_QUBITS = 48                               # v11.0: raised from 44
        VQPU_BATCH_LIMIT = 512                             # v11.0: 1.5x (was 384)
        VQPU_MPS_MAX_BOND_LOW = 1024                       # v11.0: 1.5x (was 768)
        VQPU_MPS_MAX_BOND_MED = 2048                       # v11.0: 1.5x (was 1536)
        VQPU_MPS_MAX_BOND_HIGH = 4096                      # v11.0: 1.5x (was 3072)
        VQPU_ADAPTIVE_SHOTS_MAX = 98304                    # v11.0: 2x (was 49152)
    else:
        VQPU_MAX_QUBITS = 44                               # v11.0: raised from 40
        VQPU_BATCH_LIMIT = min(384, max(48, int(_HW_RAM_GB * 48)))  # v11.0: 1.5x
        VQPU_MPS_MAX_BOND_LOW = 512                        # v11.0: 1.5x (was 384)
        VQPU_MPS_MAX_BOND_MED = 1024                       # v11.0: 1.5x (was 768)
        VQPU_MPS_MAX_BOND_HIGH = 2048                      # v11.0: 1.5x (was 1536)
        VQPU_ADAPTIVE_SHOTS_MAX = 65536                    # v11.0: 2x (was 32768)

VQPU_ADAPTIVE_SHOTS_MIN = 256
# v12.3: Use all physical cores (min 2) — was under-counting on 2-core Macs
# The pipeline is I/O-bound (file writes + transpile), so hyperthreading helps
VQPU_PIPELINE_WORKERS = min(12, max(2, _HW_CORES))        # v12.3: use logical cores, min 2
VQPU_STABILIZER_MAX_QUBITS = 10000                        # v6.0: Clifford-only unlimited regime
VQPU_DB_RESEARCH_QUBITS = min(16, VQPU_MAX_QUBITS)       # v11.0: raised (was 14)

# v7.1: CPU/MPS crossover — qubit count below which CPU statevector is optimal.
# Above this: route to MPS (low/med entanglement) or chunked CPU (high entanglement).
# Intel x86_64: Higher crossover (16Q → 512MB sv fits in RAM with AVX2+OpenBLAS).
#   CPU statevector is fast on Intel for <16Q thanks to AVX2+FMA3 vectorization.
#   Above 16Q: MPS is more memory-efficient if entanglement is bounded.
# Apple Silicon: Lower crossover (10Q) — Metal GPU wins above ~10Q.
# v11.0: CPU/MPS crossover — auto-tuned at first run, fallback to heuristic.
if _IS_INTEL:
    if _HW_RAM_GB >= 16:
        VQPU_GPU_CROSSOVER = 28                            # v11.0: raised (was 26)
    elif _HW_RAM_GB >= 8:
        VQPU_GPU_CROSSOVER = 26                            # v11.0: raised (was 24)
    else:
        VQPU_GPU_CROSSOVER = 22                            # v11.0: raised (was 20)
else:
    VQPU_GPU_CROSSOVER = 14                                # v11.0: raised (was 12)

# v7.1: MPS fallback target when bond dim explodes
# Intel: fall back to chunked CPU statevector (no Metal compute)
# Apple Silicon: fall back to Metal GPU (fast compute shaders)
VQPU_MPS_FALLBACK_TARGET = "chunked_cpu" if _IS_INTEL else "metal_gpu"

# ═══════════════════════════════════════════════════════════════════
# IPC PATHS
# ═══════════════════════════════════════════════════════════════════

BRIDGE_PATH = Path("/tmp/l104_bridge")
INBOX_PATH = BRIDGE_PATH / "inbox"
OUTBOX_PATH = BRIDGE_PATH / "outbox"
TELEMETRY_PATH = BRIDGE_PATH / "telemetry"
THROTTLE_SIGNAL = BRIDGE_PATH / "throttle.signal"
