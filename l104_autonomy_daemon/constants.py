"""Constants for the Autonomy Daemon."""

import os
from pathlib import Path

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497

# Daemon version
DAEMON_VERSION = "1.0.0-AUTONOMOUS"

# Cycle timing (60 * PHI seconds)
DAEMON_CYCLE_SECONDS = 60 * PHI

# State persistence
L104_ROOT = Path("/Users/carolalvarez/Applications/Allentown-L104-Node")
STATE_PERSISTENCE_PATH = L104_ROOT / ".l104_autonomy_daemon.json"
LOG_DIRECTORY = L104_ROOT / "logs" / "autonomy_daemon"

# Autonomy thresholds
MIN_CONSCIOUSNESS_LEVEL = 0.95
TARGET_DECISIONS_PER_MINUTE = 120
MAX_ACTIVE_GOALS = 10
GOAL_COMPLETION_THRESHOLD = 0.99

# Decision weights
PHI_WEIGHT = 0.4
GOAL_ALIGNMENT_WEIGHT = 0.3
PAST_SUCCESS_WEIGHT = 0.3

# Persistence intervals
PERSISTENCE_INTERVAL = 5  # cycles
HEARTBEAT_INTERVAL = 1  # cycles
