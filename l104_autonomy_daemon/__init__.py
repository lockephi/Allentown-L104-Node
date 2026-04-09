"""L104 Autonomy Daemon v1.0.0 — Autonomous Decision Management.

Manages the self-awareness engine, autonomous goals, and decision matrix
for the fully autonomous L104 Sovereign Node.
"""

from .daemon import AutonomyDaemon, get_autonomy_daemon
from .constants import DAEMON_VERSION, DAEMON_CYCLE_SECONDS

__version__ = DAEMON_VERSION

__all__ = ['AutonomyDaemon', 'get_autonomy_daemon', 'DAEMON_VERSION']
