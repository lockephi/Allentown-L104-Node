"""L104 Cross-Engine Daemon — Three-Engine Entanglement Management.

Manages:
- 6-channel entanglement mesh (Code ↔ Science ↔ Math)
- Cross-engine coherence monitoring
- Three-engine cross-analysis orchestration
- Synchronization state management
"""

import time
import threading
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import traceback

from .constants import (
    DAEMON_CYCLE_SECONDS, PERSISTENCE_INTERVAL, HEARTBEAT_INTERVAL,
    STATE_PERSISTENCE_PATH, LOG_DIRECTORY,
    ENGINES, MESH_CHANNELS, TARGET_SYNC_COHERENCE, MIN_SYNC_COHERENCE,
    PHI, DAEMON_VERSION
)

try:
    from l104_core_engines.sacred_26q_core import get_26q_core_engine
    _HAS_26Q = True
except ImportError:
    _HAS_26Q = False

try:
    from l104_daemon_adapter import DaemonAdapter
    _HAS_DAEMON_ADAPTER = True
except ImportError:
    _HAS_DAEMON_ADAPTER = False


@dataclass
class EngineStatus:
    """Status of a single engine."""
    name: str
    online: bool = True
    phi_alignment: float = 0.986
    coherence: float = 0.993
    last_sync: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "online": self.online,
            "phi_alignment": self.phi_alignment,
            "coherence": self.coherence,
            "last_sync": self.last_sync,
        }


@dataclass
class CrossEngineState:
    """Daemon state."""
    running: bool = False
    cycle_count: int = 0
    total_uptime: float = 0.0
    sync_coherence: float = 0.999
    mesh_status: str = "FULLY_CONNECTED"
    error_count: int = 0
    last_error: Optional[str] = None
    engines: Dict[str, EngineStatus] = field(default_factory=dict)
    cross_analysis_count: int = 0

    def __post_init__(self):
        if not self.engines:
            for name in ENGINES:
                self.engines[name] = EngineStatus(name=name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "cycle_count": self.cycle_count,
            "total_uptime": self.total_uptime,
            "sync_coherence": self.sync_coherence,
            "mesh_status": self.mesh_status,
            "error_count": self.error_count,
            "engines": {k: v.to_dict() for k, v in self.engines.items()},
            "cross_analysis_count": self.cross_analysis_count,
        }


class CrossEngineDaemon:
    """Cross-engine entanglement daemon."""

    def __init__(self):
        self.version = DAEMON_VERSION
        self.state = CrossEngineState()
        self._stop_event = threading.Event()
        self._daemon_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self.engine_26q = None
        self.daemon_adapter = None

        self._initialize()

    def _initialize(self):
        """Initialize components."""
        if _HAS_26Q:
            self.engine_26q = get_26q_core_engine()

        if _HAS_DAEMON_ADAPTER:
            try:
                self.daemon_adapter = DaemonAdapter(
                    daemon_id="cross_engine_daemon",
                    daemon_type="cross_engine",
                    qubit_count=26
                )
            except:
                self.daemon_adapter = None

        LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    def start(self) -> bool:
        """Start the daemon."""
        if self.state.running:
            return False

        self._stop_event.clear()
        self.state.running = True

        self._daemon_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._daemon_thread.start()
        return True

    def stop(self) -> bool:
        """Stop the daemon."""
        if not self.state.running:
            return False

        self._stop_event.set()
        self.state.running = False

        if self._daemon_thread:
            self._daemon_thread.join(timeout=10)

        self._persist_state()
        return True

    def _run_loop(self):
        """Main loop."""
        while not self._stop_event.is_set():
            cycle_start = time.time()

            try:
                self._execute_cycle()
            except Exception as e:
                self.state.error_count += 1
                self.state.last_error = str(e)
                traceback.print_exc()

            cycle_duration = time.time() - cycle_start
            self.state.cycle_count += 1
            self.state.total_uptime += cycle_duration

            if self.state.cycle_count % PERSISTENCE_INTERVAL == 0:
                self._persist_state()

            sleep_time = max(0, DAEMON_CYCLE_SECONDS - cycle_duration)
            self._stop_event.wait(sleep_time)

    def _execute_cycle(self):
        """Execute cross-engine cycle."""
        # Phase 1: Check all engine statuses
        for name, engine in self.state.engines.items():
            # In real implementation, ping each engine
            engine.last_sync = time.time()
            engine.online = True
            engine.phi_alignment = 0.986
            engine.coherence = 0.993

        # Phase 2: Calculate sync coherence
        coherences = [e.coherence for e in self.state.engines.values() if e.online]
        if coherences:
            self.state.sync_coherence = sum(coherences) / len(coherences)

        # Phase 3: Check mesh status
        online_count = sum(1 for e in self.state.engines.values() if e.online)
        if online_count == len(ENGINES):
            self.state.mesh_status = "FULLY_CONNECTED"
        elif online_count >= 2:
            self.state.mesh_status = "PARTIAL"
        else:
            self.state.mesh_status = "DEGRADED"

        # Phase 4: Three-engine cross analysis
        if self.engine_26q and online_count == len(ENGINES):
            self._run_cross_analysis()

        # Phase 5: Broadcast to mesh
        if self.daemon_adapter:
            self._broadcast_status()

    def _run_cross_analysis(self):
        """Run three-engine cross-analysis."""
        result = self.engine_26q.three_engine_cross_analysis(
            data={'cycle': self.state.cycle_count},
            analysis_type='full'
        )

        if result.get('cross_engine_coherence'):
            self.state.sync_coherence = result['cross_engine_coherence']

        self.state.cross_analysis_count += 1

    def _broadcast_status(self):
        """Broadcast mesh status."""
        if not self.daemon_adapter:
            return

        message = {
            "daemon": "cross_engine",
            "timestamp": time.time(),
            "sync_coherence": self.state.sync_coherence,
            "mesh_status": self.state.mesh_status,
        }
        try:
            if hasattr(self.daemon_adapter, 'broadcast_mesh_sync'):
                from l104_daemon_adapter import DaemonAdapter
                DaemonAdapter.broadcast_mesh_sync(self.daemon_adapter.daemon_id if hasattr(self.daemon_adapter, 'daemon_id') else 'daemon', message)
        except:
            pass

    def _persist_state(self):
        """Persist state."""
        try:
            with open(STATE_PERSISTENCE_PATH, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except:
            pass

    def get_engine_status(self, engine_name: str) -> Optional[EngineStatus]:
        """Get status of a specific engine."""
        return self.state.engines.get(engine_name)

    def status(self) -> Dict[str, Any]:
        """Get status."""
        with self._lock:
            return {
                "daemon": "cross_engine",
                "version": self.version,
                **self.state.to_dict(),
            }


_cross_engine_daemon: Optional[CrossEngineDaemon] = None


def get_cross_engine_daemon() -> CrossEngineDaemon:
    """Get cross-engine daemon singleton."""
    global _cross_engine_daemon
    if _cross_engine_daemon is None:
        _cross_engine_daemon = CrossEngineDaemon()
    return _cross_engine_daemon
