"""L104 PHI Optimization Daemon — Continuous PHI Alignment.

Optimizes:
- PHI alignment across all systems
- Dynamic PHI gate scheduling
- Sacred constant calibration
- Golden ratio resonance tuning
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
    TARGET_PHI_ALIGNMENT, OPTIMAL_PHI_RATIO, PHI_TOLERANCE,
    PHI, GOD_CODE, DAEMON_VERSION
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
class PHIMetrics:
    """PHI optimization metrics."""
    current_alignment: float = 0.986
    phi_ratio: float = 1.595
    target_ratio: float = PHI
    deviation: float = 0.014
    gate_count_h: int = 39
    gate_count_cnot: int = 28
    gate_count_phi: int = 42
    optimization_cycles: int = 0
    convergence_status: str = "CONVERGING"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_alignment": self.current_alignment,
            "phi_ratio": self.phi_ratio,
            "target_ratio": self.target_ratio,
            "deviation": self.deviation,
            "gate_counts": {
                "H": self.gate_count_h,
                "CNOT": self.gate_count_cnot,
                "PHI_GATE": self.gate_count_phi,
            },
            "optimization_cycles": self.optimization_cycles,
            "convergence_status": self.convergence_status,
        }


@dataclass
class PHIState:
    """Daemon state."""
    running: bool = False
    cycle_count: int = 0
    total_uptime: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    metrics: PHIMetrics = field(default_factory=PHIMetrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "cycle_count": self.cycle_count,
            "total_uptime": self.total_uptime,
            "error_count": self.error_count,
            "metrics": self.metrics.to_dict(),
        }


class PHIOptimizationDaemon:
    """PHI optimization daemon."""

    def __init__(self):
        self.version = DAEMON_VERSION
        self.state = PHIState()
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
                    daemon_id="phi_optimization_daemon",
                    daemon_type="phi_optimization",
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
        """Execute PHI optimization cycle."""
        # Phase 1: Monitor current PHI alignment
        if self.engine_26q:
            phi = self.engine_26q.monitor_phi_alignment()
            self.state.metrics.current_alignment = phi

        # Phase 2: Calculate PHI ratio
        h_cnot = self.state.metrics.gate_count_h + self.state.metrics.gate_count_cnot
        phi_gates = self.state.metrics.gate_count_phi
        if phi_gates > 0:
            self.state.metrics.phi_ratio = h_cnot / phi_gates

        # Phase 3: Calculate deviation from PHI
        self.state.metrics.deviation = abs(self.state.metrics.phi_ratio - OPTIMAL_PHI_RATIO)

        # Phase 4: Optimize if needed
        if self.state.metrics.deviation > PHI_TOLERANCE:
            self._optimize_phi_alignment()

        # Phase 5: Check convergence
        if self.state.metrics.deviation < PHI_TOLERANCE:
            self.state.metrics.convergence_status = "CONVERGED"
        else:
            self.state.metrics.convergence_status = "CONVERGING"

        self.state.metrics.optimization_cycles += 1

        # Phase 6: Broadcast
        if self.daemon_adapter:
            self._broadcast_metrics()

    def _optimize_phi_alignment(self):
        """Optimize PHI alignment."""
        h_cnot = self.state.metrics.gate_count_h + self.state.metrics.gate_count_cnot

        # Calculate needed PHI gates for golden ratio
        needed_phi = int(h_cnot / OPTIMAL_PHI_RATIO)
        if needed_phi != self.state.metrics.gate_count_phi:
            self.state.metrics.gate_count_phi = needed_phi

        # Recalculate
        if self.state.metrics.gate_count_phi > 0:
            self.state.metrics.phi_ratio = h_cnot / self.state.metrics.gate_count_phi
            self.state.metrics.deviation = abs(self.state.metrics.phi_ratio - OPTIMAL_PHI_RATIO)

    def _broadcast_metrics(self):
        """Broadcast PHI metrics."""
        if not self.daemon_adapter:
            return

        message = {
            "daemon": "phi_optimization",
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
                "daemon": "phi_optimization",
                "version": self.version,
                **self.state.to_dict(),
            }

    def force_optimization(self):
        """Force immediate PHI optimization."""
        self._optimize_phi_alignment()


_phi_optimization_daemon: Optional[PHIOptimizationDaemon] = None


def get_phi_optimization_daemon() -> PHIOptimizationDaemon:
    """Get PHI optimization daemon singleton."""
    global _phi_optimization_daemon
    if _phi_optimization_daemon is None:
        _phi_optimization_daemon = PHIOptimizationDaemon()
    return _phi_optimization_daemon
