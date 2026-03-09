"""L104 Quantum AI Daemon — Sacred constants and configuration."""

import math
import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# SACRED CONSTANTS — immutable, GOD_CODE-aligned
# ═══════════════════════════════════════════════════════════════════

DAEMON_VERSION = "2.0.0"

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.04 + PHI / 1000  # 1.0416180339887497
OMEGA = 6539.34712682
TAU = 2 * math.pi
FEIGENBAUM = 4.669201609102990

# Sacred daemon tuning — all derived from GOD_CODE
SACRED_RESONANCE = (GOD_CODE / 16) ** PHI  # ≈ 286 (Iron harmonic)
PHI_MICRO_PHASE = GOD_CODE % TAU           # Canonical phase angle
GOLDEN_TICK = PHI * 10                     # ≈ 16.18s sacred tick

# ═══════════════════════════════════════════════════════════════════
# DAEMON TIMING — adaptive cycle configuration
# ═══════════════════════════════════════════════════════════════════

CYCLE_INTERVAL_S = float(os.environ.get(
    "L104_QAI_CYCLE_INTERVAL", "120.0"))      # Full improvement cycle: 2 min
CYCLE_MIN_INTERVAL_S = float(os.environ.get(
    "L104_QAI_CYCLE_MIN", "60.0"))             # Floor: 1 min under low load
CYCLE_MAX_INTERVAL_S = float(os.environ.get(
    "L104_QAI_CYCLE_MAX", "600.0"))            # Ceiling: 10 min under high load
LOAD_THRESHOLD_LOW = 20.0                      # CPU% → faster cycles
LOAD_THRESHOLD_HIGH = 70.0                     # CPU% → slower cycles

# Scan settings
SCAN_BATCH_SIZE = int(os.environ.get(
    "L104_QAI_SCAN_BATCH", "25"))              # Files per improvement cycle
MAX_FILE_SIZE_KB = 500                         # Skip files > 500KB (data files)
SCAN_DEPTH_MAX = 5                             # Max directory recursion depth

# ═══════════════════════════════════════════════════════════════════
# STATE + PERSISTENCE
# ═══════════════════════════════════════════════════════════════════

L104_ROOT = Path(os.environ.get("L104_ROOT", os.getcwd()))
STATE_FILE = ".l104_quantum_ai_daemon.json"
STATE_PATH = L104_ROOT / STATE_FILE
PERSIST_EVERY_N_CYCLES = 5                     # Persist state every N cycles
PID_FILE = Path("/tmp/l104_bridge/quantum_ai/daemon.pid")
HEARTBEAT_FILE = Path("/tmp/l104_bridge/quantum_ai/heartbeat")
LOG_DIR = L104_ROOT / "logs" / "quantum_ai_daemon"
IPC_PATH = Path("/tmp/l104_bridge/quantum_ai")
INBOX_PATH = IPC_PATH / "inbox"
OUTBOX_PATH = IPC_PATH / "outbox"

# ═══════════════════════════════════════════════════════════════════
# IMPROVEMENT THRESHOLDS — when to trigger improvement actions
# ═══════════════════════════════════════════════════════════════════

SMELL_THRESHOLD = 3              # Trigger auto-fix if ≥ N code smells
COMPLEXITY_THRESHOLD = 15        # Flag functions with cyclomatic > N
PERF_SCORE_MIN = 0.6             # Trigger optimization if perf < N
FIDELITY_MIN = 0.85              # Sacred alignment minimum
COHERENCE_MIN = 0.7              # Cross-engine coherence minimum
HEALTH_STALENESS_DECAY = 0.01   # 1% decay per cycle when idle

# ═══════════════════════════════════════════════════════════════════
# TELEMETRY
# ═══════════════════════════════════════════════════════════════════

TELEMETRY_WINDOW = 500           # Ring buffer size
ERROR_LOG_SIZE = 100             # Error history
IMPROVEMENT_HISTORY_SIZE = 1000  # Completed improvement log
ANOMALY_SIGMA = 3.0              # Standard deviations for anomaly detection

# ═══════════════════════════════════════════════════════════════════
# QUARANTINE + RESILIENCE
# ═══════════════════════════════════════════════════════════════════

QUARANTINE_THRESHOLD = 3         # Consecutive failures → quarantine
QUARANTINE_CYCLES = 10           # Cycles to skip quarantined file
MAX_RETRY = 1                    # Retries per file per cycle
CIRCUIT_BREAKER_BASE_S = 30.0   # Base backoff on breaker trip
CIRCUIT_BREAKER_MAX_S = 1800.0  # Max 30-min backoff

# ═══════════════════════════════════════════════════════════════════
# PACKAGES + FILE PATTERNS
# ═══════════════════════════════════════════════════════════════════

L104_PACKAGES = [
    "l104_agi", "l104_asi", "l104_intellect", "l104_server",
    "l104_code_engine", "l104_science_engine", "l104_math_engine",
    "l104_ml_engine", "l104_quantum_gate_engine", "l104_quantum_engine",
    "l104_numerical_engine", "l104_gate_engine", "l104_god_code_simulator",
    "l104_quantum_data_analyzer", "l104_search", "l104_simulator",
    "l104_audio_simulation", "l104_vqpu",
]

# Root shim files (backward compat)
L104_ROOT_SHIMS = [
    "l104_agi_core.py", "l104_asi_core.py", "l104_local_intellect.py",
    "l104_fast_server.py", "l104_quantum_link_builder.py",
    "l104_quantum_numerical_builder.py", "l104_logic_gate_builder.py",
]

# Files to never modify (sacred, config, or generated)
IMMUTABLE_FILES = {
    "constants.py", "__init__.py", "setup.py", "pyproject.toml",
}

# File extensions to process
PROCESSABLE_EXTENSIONS = {".py"}
