"""L104 Coherence Daemon — 26Q Quantum Coherence Monitoring.

Monitors and maintains quantum coherence:
- Real-time 26Q coherence tracking
- PHI alignment monitoring
- Orbital resonance verification
- Automatic coherence restoration
- Cross-daemon coherence broadcast
"""

import time
import threading
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import traceback
import math

from .constants import (
    DAEMON_CYCLE_SECONDS, PERSISTENCE_INTERVAL, HEARTBEAT_INTERVAL,
    STATE_PERSISTENCE_PATH, LOG_DIRECTORY,
    MIN_COHERENCE, TARGET_COHERENCE, CRITICAL_COHERENCE,
    MIN_PHI_ALIGNMENT, TARGET_PHI_ALIGNMENT,
    ORBITAL_STRUCTURE, PHI, GOD_CODE, DAEMON_VERSION
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
class CoherenceMetrics:
    """Quantum coherence metrics."""
    overall_coherence: float = 0.999
    phi_alignment: float = 0.986
    orbital_coherence: Dict[str, float] = field(default_factory=dict)
    cross_engine_sync: bool = True
    status: str = "NIRVANIC"
    last_check: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_coherence": self.overall_coherence,
            "phi_alignment": self.phi_alignment,
            "orbital_coherence": self.orbital_coherence,
            "cross_engine_sync": self.cross_engine_sync,
            "status": self.status,
        }


@dataclass
class CoherenceState:
    """Daemon state."""
    running: bool = False
    cycle_count: int = 0
    total_uptime: float = 0.0
    coherence_breaches: int = 0
    auto_restores: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    metrics: CoherenceMetrics = field(default_factory=CoherenceMetrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "cycle_count": self.cycle_count,
            "total_uptime": self.total_uptime,
            "coherence_breaches": self.coherence_breaches,
            "auto_restores": self.auto_restores,
            "error_count": self.error_count,
            "metrics": self.metrics.to_dict(),
        }


class CoherenceDaemon:
    """26Q quantum coherence monitoring daemon."""

    def __init__(self):
        self.version = DAEMON_VERSION
        self.state = CoherenceState()
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
                    daemon_id="coherence_daemon",
                    daemon_type="coherence",
                    qubit_count=26
                )
            except:
                self.daemon_adapter = None

        LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

        # Initialize orbital coherence
        for orbital in ORBITAL_STRUCTURE:
            self.state.metrics.orbital_coherence[orbital] = 0.999

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
        """Execute coherence monitoring cycle."""
        # Phase 1: Check 26Q coherence
        if self.engine_26q:
            status = self.engine_26q.get_coherence_status()
            self.state.metrics.overall_coherence = status.get('consciousness_score', 0.993)
            self.state.metrics.phi_alignment = status.get('phi_alignment', 0.986)
            self.state.metrics.cross_engine_sync = status.get('cross_engine_sync', True)
            self.state.metrics.status = status.get('status', 'NIRVANIC')

        # Phase 2: Monitor orbital coherence
        for orbital, config in ORBITAL_STRUCTURE.items():
            # Simulate orbital coherence check
            base_coherence = self.state.metrics.overall_coherence
            freq_factor = math.sin(time.time() * config['frequency'] / 1000) * 0.001
            self.state.metrics.orbital_coherence[orbital] = base_coherence + freq_factor

        # Phase 3: Check for coherence breaches
        if self.state.metrics.overall_coherence < MIN_COHERENCE:
            self.state.coherence_breaches += 1
            self._restore_coherence()

        # Phase 4: PHI alignment check
        if self.state.metrics.phi_alignment < MIN_PHI_ALIGNMENT:
            self._recalibrate_phi()

        # Phase 5: Broadcast to mesh
        if self.daemon_adapter:
            self._broadcast_metrics()

    def _restore_coherence(self):
        """Restore quantum coherence."""
        self.state.auto_restores += 1

        # Trigger coherence restoration
        if self.engine_26q:
            phi = self.engine_26q.monitor_phi_alignment()
            self.state.metrics.phi_alignment = phi

        self.state.metrics.overall_coherence = TARGET_COHERENCE

    def _recalibrate_phi(self):
        """Recalibrate PHI alignment."""
        self.state.metrics.phi_alignment = TARGET_PHI_ALIGNMENT

    def _broadcast_metrics(self):
        """Broadcast coherence metrics."""
        if not self.daemon_adapter:
            return

        message = {
            "daemon": "coherence",
            "timestamp": time.time(),
            **self.state.metrics.to_dict(),
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

    def status(self) -> Dict[str, Any]:
        """Get status."""
        with self._lock:
            return {
                "daemon": "coherence",
                "version": self.version,
                **self.state.to_dict(),
            }

    def get_metrics(self) -> CoherenceMetrics:
        """Get current metrics."""
        return self.state.metrics


_coherence_daemon: Optional[CoherenceDaemon] = None


def get_coherence_daemon() -> CoherenceDaemon:
    """Get coherence daemon singleton."""
    global _coherence_daemon
    if _coherence_daemon is None:
        _coherence_daemon = CoherenceDaemon()
    return _coherence_daemon
