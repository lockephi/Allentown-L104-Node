"""Constants for the Self-Healing Daemon."""

from pathlib import Path

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895

# Daemon version
DAEMON_VERSION = "1.0.0-SELF-HEALING"

# Cycle timing (PHI seconds * 10)
DAEMON_CYCLE_SECONDS = PHI * 10

# Response time targets
TARGET_RESPONSE_TIME_MS = 50  # 50ms for critical issues
MAX_RESPONSE_TIME_MS = 100   # 100ms max

# Healing thresholds
PHI_DRIFT_THRESHOLD = 0.01
COHERENCE_DROP_THRESHOLD = 0.05
ERROR_RATE_THRESHOLD = 0.1

# State persistence
L104_ROOT = Path("/Users/carolalvarez/Applications/Allentown-L104-Node")
STATE_PERSISTENCE_PATH = L104_ROOT / ".l104_self_healing_daemon.json"
LOG_DIRECTORY = L104_ROOT / "logs" / "self_healing_daemon"

# Persistence intervals
PERSISTENCE_INTERVAL = 5
HEARTBEAT_INTERVAL = 1
