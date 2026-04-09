"""Constants for the Cross-Engine Daemon."""

from pathlib import Path

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895

# Daemon version
DAEMON_VERSION = "1.0.0-CROSS-ENGINE"

# Cycle timing
DAEMON_CYCLE_SECONDS = 30  # 30 seconds

# Entanglement mesh
ENGINES = ['code', 'science', 'math']
MESH_CHANNELS = 6  # Full mesh between 3 engines

# Sync thresholds
TARGET_SYNC_COHERENCE = 0.999
MIN_SYNC_COHERENCE = 0.95

# State persistence
L104_ROOT = Path("/Users/carolalvarez/Applications/Allentown-L104-Node")
STATE_PERSISTENCE_PATH = L104_ROOT / ".l104_cross_engine_daemon.json"
LOG_DIRECTORY = L104_ROOT / "logs" / "cross_engine_daemon"

# Persistence intervals
PERSISTENCE_INTERVAL = 5
HEARTBEAT_INTERVAL = 1
