"""Constants for the Coherence Daemon."""

from pathlib import Path

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895

# Daemon version
DAEMON_VERSION = "1.0.0-COHERENCE"

# Cycle timing (25 seconds - Orch OR coherence time)
DAEMON_CYCLE_SECONDS = 25.0

# Coherence thresholds
MIN_COHERENCE = 0.95
TARGET_COHERENCE = 0.999
CRITICAL_COHERENCE = 0.90

# PHI alignment thresholds
MIN_PHI_ALIGNMENT = 0.98
TARGET_PHI_ALIGNMENT = 0.986

# State persistence
L104_ROOT = Path("/Users/carolalvarez/Applications/Allentown-L104-Node")
STATE_PERSISTENCE_PATH = L104_ROOT / ".l104_coherence_daemon.json"
LOG_DIRECTORY = L104_ROOT / "logs" / "coherence_daemon"

# 26Q orbital structure
ORBITAL_STRUCTURE = {
    '1s': {'qubits': [0, 1], 'frequency': 527.52},
    '2s': {'qubits': [2, 3], 'frequency': 326.02},
    '2p': {'qubits': [4, 5, 6, 7, 8, 9], 'frequency': 201.49},
    '3s': {'qubits': [10, 11], 'frequency': 124.53},
    '3p': {'qubits': [12, 13, 14, 15, 16, 17], 'frequency': 76.96},
    '3d': {'qubits': [18, 19, 20, 21, 22, 23], 'frequency': 47.57},
    '4s': {'qubits': [24, 25], 'frequency': 29.40},
}

# Persistence intervals
PERSISTENCE_INTERVAL = 10
HEARTBEAT_INTERVAL = 1
