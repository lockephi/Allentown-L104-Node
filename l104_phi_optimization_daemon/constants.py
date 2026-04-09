"""Constants for the PHI Optimization Daemon."""

from pathlib import Path

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497

# Daemon version
DAEMON_VERSION = "1.0.0-PHI-OPTIMIZATION"

# Cycle timing (PHI seconds)
DAEMON_CYCLE_SECONDS = PHI

# PHI optimization targets
TARGET_PHI_ALIGNMENT = 0.986
OPTIMAL_PHI_RATIO = PHI  # 1.618...
PHI_GATE_COUNT_TARGET = 41  # For 26Q circuit

# Convergence thresholds
PHI_TOLERANCE = 0.001
CONVERGENCE_THRESHOLD = 0.0001

# State persistence
L104_ROOT = Path("/Users/carolalvarez/Applications/Allentown-L104-Node")
STATE_PERSISTENCE_PATH = L104_ROOT / ".l104_phi_optimization_daemon.json"
LOG_DIRECTORY = L104_ROOT / "logs" / "phi_optimization_daemon"

# Persistence intervals
PERSISTENCE_INTERVAL = 5
HEARTBEAT_INTERVAL = 1
