"""L104 Cross-Engine Daemon v1.0.0 — Three-Engine Entanglement Management.

Manages the 6-channel entanglement mesh between Code, Science, and Math
engines with 26Q consciousness integration.
"""

from .daemon import CrossEngineDaemon, get_cross_engine_daemon
from .constants import DAEMON_VERSION

__version__ = DAEMON_VERSION

__all__ = ['CrossEngineDaemon', 'get_cross_engine_daemon', 'DAEMON_VERSION']
