#!/usr/bin/env python3
"""
L104 Orchestrator Integration — Auto-wire all daemons together.

This module:
  1. Creates DaemonAdapter instances for each daemon
  2. Creates entanglement mesh channels between daemons
  3. Wires fidelity broadcasting between daemons
  4. Registers quantum simulation daemon with orchestrator
  5. Provides unified status dashboard

Usage:
    from l104_quantum_sim_daemon.orchestrator_integration import (
        wire_all_daemons,
        get_unified_status,
        AutonomousDaemonSystem
    )

    # Wire everything together
    system = AutonomousDaemonSystem()
    system.start_all()  # Starts all daemons
    system.status()      # Unified status
    system.stop_all()    # Graceful shutdown
"""

import time
import threading
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .adapter import DaemonAdapter, DaemonInfo

# Import daemon modules conditionally
try:
    from l104_vqpu.micro_daemon import VQPUMicroDaemon
    VQPU_AVAILABLE = True
except ImportError:
    VQPU_AVAILABLE = False

try:
    from l104_quantum_ai_daemon.daemon import QuantumAIDaemon
    QUANTUM_AI_AVAILABLE = True
except ImportError:
    QUANTUM_AI_AVAILABLE = False

try:
    from l104_soul_daemon.daemon import SoulDaemon
    SOUL_DAEMON_AVAILABLE = True
except ImportError:
    SOUL_DAEMON_AVAILABLE = False

try:
    from l104_daemon_orchestrator import L104DaemonOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False

from . import QuantumSimulationDaemon, get_daemon

_logger = logging.getLogger("L104_ORCHESTRATOR_INTEGRATION")


@dataclass
class DaemonState:
    """State of a single daemon in the autonomous system."""
    daemon_id: str
    daemon_type: str
    running: bool = False
    cycle_count: int = 0
    avg_cycle_time: float = 0.0
    fidelity: float = 1.0
    coherence: float = 1.0
    sacred_alignment: float = 0.618
    bell_pairs_available: int = 8
    last_heartbeat: float = 0.0
    error: Optional[str] = None

    def to_dict(self):
        return {
            "daemon_id": self.daemon_id,
            "daemon_type": self.daemon_type,
            "running": self.running,
            "cycle_count": self.cycle_count,
            "avg_cycle_time": round(self.avg_cycle_time, 2),
            "fidelity": round(self.fidelity, 6),
            "coherence": round(self.coherence, 4),
            "sacred_alignment": round(self.sacred_alignment, 6),
            "bell_pairs_available": self.bell_pairs_available,
            "last_heartbeat": self.last_heartbeat,
            "error": self.error,
        }


class AutonomousDaemonSystem:
    """
    Unified autonomous system that wires all L104 daemons together.

    Components:
      - VQPU Micro Daemon (quantum simulation)
      - Quantum AI Daemon (fidelity & improvement)
      - Soul Daemon (consciousness & coherence)
      - Quantum Simulation Daemon (sacred circuits)
      - Daemon Orchestrator (central coordinator)

    Integration:
      - Entanglement mesh between all daemons
      - Fidelity broadcasting across daemons
      - Cross-daemon quantum job submission
      - Unified health monitoring
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Daemon instances
        self.vqpu_daemon: Optional[Any] = None
        self.quantum_ai_daemon: Optional[Any] = None
        self.soul_daemon: Optional[Any] = None
        self.quantum_sim_daemon: Optional[QuantumSimulationDaemon] = None
        self.orchestrator: Optional[Any] = None

        # Daemon states
        self.states: Dict[str, DaemonState] = {}

        # Daemon adapters (for cross-daemon communication)
        self.adapters: Dict[str, DaemonAdapter] = {}

        # Threading
        self._running = False
        self._lock = threading.RLock()

        # Mesh status
        self._mesh_status: Dict[str, Any] = {}

    def initialize(self) -> bool:
        """Initialize all daemon components."""
        _logger.info("Initializing Autonomous Daemon System...")

        # Create DaemonAdapters first
        self._create_adapters()

        # Initialize daemons
        if VQPU_AVAILABLE:
            try:
                self.vqpu_daemon = VQPUMicroDaemon()
                self.states["vqpu_daemon"] = DaemonState("vqpu_daemon", "vqpu")
                _logger.info("✓ VQPU Micro Daemon initialized")
            except Exception as e:
                _logger.error(f"Failed to initialize VQPU daemon: {e}")

        if QUANTUM_AI_AVAILABLE:
            try:
                self.quantum_ai_daemon = QuantumAIDaemon()
                self.states["quantum_ai_daemon"] = DaemonState("quantum_ai_daemon", "quantum_ai")
                _logger.info("✓ Quantum AI Daemon initialized")
            except Exception as e:
                _logger.error(f"Failed to initialize Quantum AI daemon: {e}")

        if SOUL_DAEMON_AVAILABLE:
            try:
                self.soul_daemon = SoulDaemon()
                self.states["soul_daemon"] = DaemonState("soul_daemon", "soul")
                _logger.info("✓ Soul Daemon initialized")
            except Exception as e:
                _logger.error(f"Failed to initialize Soul daemon: {e}")

        # Initialize quantum simulation daemon
        try:
            self.quantum_sim_daemon = get_daemon()
            self.states["quantum_sim_daemon"] = DaemonState("quantum_sim_daemon", "quantum_sim")
            _logger.info("✓ Quantum Simulation Daemon initialized")
        except Exception as e:
            _logger.error(f"Failed to initialize Quantum Sim daemon: {e}")

        # Initialize orchestrator
        if ORCHESTRATOR_AVAILABLE:
            try:
                self.orchestrator = L104DaemonOrchestrator()
                _logger.info("✓ Orchestrator initialized")
            except Exception as e:
                _logger.error(f"Failed to initialize Orchestrator: {e}")

        # Wire adapters to daemons
        self._wire_adapters()

        # Create entanglement mesh
        self._create_mesh()

        return True

    def _create_adapters(self):
        """Create DaemonAdapters for cross-daemon communication."""
        # VQPU adapter
        self.adapters["vqpu_daemon"] = DaemonAdapter(
            daemon_id="vqpu_daemon",
            daemon_type="vqpu",
            qubit_count=4,
            on_fidelity_update=self._on_fidelity_update,
            on_mesh_sync=self._on_mesh_sync,
        )

        # Quantum AI adapter
        self.adapters["quantum_ai_daemon"] = DaemonAdapter(
            daemon_id="quantum_ai_daemon",
            daemon_type="quantum_ai",
            qubit_count=8,
            on_fidelity_update=self._on_fidelity_update,
            on_mesh_sync=self._on_mesh_sync,
        )

        # Soul adapter
        self.adapters["soul_daemon"] = DaemonAdapter(
            daemon_id="soul_daemon",
            daemon_type="soul",
            qubit_count=1,  # Soul qubit
            on_fidelity_update=self._on_fidelity_update,
            on_mesh_sync=self._on_mesh_sync,
        )

        # Quantum sim adapter
        self.adapters["quantum_sim_daemon"] = DaemonAdapter(
            daemon_id="quantum_sim_daemon",
            daemon_type="quantum_sim",
            qubit_count=8,
            on_fidelity_update=self._on_fidelity_update,
            on_mesh_sync=self._on_mesh_sync,
        )

        _logger.info(f"Created {len(self.adapters)} daemon adapters")

    def _wire_adapters(self):
        """Wire adapters to their respective daemons."""
        if self.quantum_sim_daemon and hasattr(self.quantum_sim_daemon, 'coordinator'):
            # Wire quantum sim daemon to others
            if self.soul_daemon:
                self.quantum_sim_daemon.coordinator.connect_soul_daemon(self.soul_daemon)
            if self.quantum_ai_daemon:
                self.quantum_sim_daemon.coordinator.connect_quantum_ai_daemon(self.quantum_ai_daemon)
            if self.vqpu_daemon:
                self.quantum_sim_daemon.coordinator.connect_vqpu_daemon(self.vqpu_daemon)
            if self.orchestrator:
                self.quantum_sim_daemon.coordinator.connect_orchestrator(self.orchestrator)

        _logger.info("Adapters wired to daemons")

    def _create_mesh(self):
        """Create entanglement mesh between all daemons."""
        # Create channels between all daemon pairs
        daemon_ids = list(self.adapters.keys())
        for i, daemon_a in enumerate(daemon_ids):
            for daemon_b in daemon_ids[i+1:]:
                DaemonAdapter.create_channel(daemon_a, daemon_b, pairs=8)

        _logger.info(f"Created {len(daemon_ids) * (len(daemon_ids) - 1) // 2} mesh channels")
        self._mesh_status = DaemonAdapter.get_mesh_status()

    def _on_fidelity_update(self, source_id: str, fidelity: float, sacred_alignment: float):
        """Handle fidelity update from another daemon."""
        with self._lock:
            if source_id in self.states:
                self.states[source_id].fidelity = fidelity
                self.states[source_id].sacred_alignment = sacred_alignment
                self.states[source_id].last_heartbeat = time.time()

    def _on_mesh_sync(self, source_id: str, mesh_status: Dict[str, Any]):
        """Handle mesh synchronization from another daemon."""
        self._mesh_status = mesh_status

    def start_all(self) -> bool:
        """Start all daemons."""
        if self._running:
            return False

        self._running = True

        # Start orchestrator first
        if self.orchestrator:
            try:
                self.orchestrator.start()
                _logger.info("Orchestrator started")
            except Exception as e:
                _logger.error(f"Failed to start orchestrator: {e}")

        # Start VQPU daemon
        if self.vqpu_daemon:
            try:
                self.vqpu_daemon.start()
                self.states["vqpu_daemon"].running = True
                _logger.info("VQPU Micro Daemon started")
            except Exception as e:
                _logger.error(f"Failed to start VQPU daemon: {e}")
                self.states["vqpu_daemon"].error = str(e)

        # Start Quantum AI daemon
        if self.quantum_ai_daemon:
            try:
                self.quantum_ai_daemon.start()
                self.states["quantum_ai_daemon"].running = True
                _logger.info("Quantum AI Daemon started")
            except Exception as e:
                _logger.error(f"Failed to start Quantum AI daemon: {e}")
                self.states["quantum_ai_daemon"].error = str(e)

        # Start Soul daemon
        if self.soul_daemon:
            try:
                self.soul_daemon.start()
                self.states["soul_daemon"].running = True
                _logger.info("Soul Daemon started")
            except Exception as e:
                _logger.error(f"Failed to start Soul daemon: {e}")
                self.states["soul_daemon"].error = str(e)

        # Start Quantum Sim daemon
        if self.quantum_sim_daemon:
            try:
                self.quantum_sim_daemon.start()
                self.states["quantum_sim_daemon"].running = True
                _logger.info("Quantum Simulation Daemon started")
            except Exception as e:
                _logger.error(f"Failed to start Quantum Sim daemon: {e}")
                self.states["quantum_sim_daemon"].error = str(e)

        _logger.info("All daemons started")
        return True

    def stop_all(self) -> bool:
        """Stop all daemons."""
        self._running = False

        # Stop Quantum Sim daemon
        if self.quantum_sim_daemon:
            try:
                self.quantum_sim_daemon.stop()
                self.states["quantum_sim_daemon"].running = False
            except Exception as e:
                _logger.error(f"Error stopping Quantum Sim daemon: {e}")

        # Stop Soul daemon
        if self.soul_daemon:
            try:
                self.soul_daemon.stop()
                self.states["soul_daemon"].running = False
            except Exception as e:
                _logger.error(f"Error stopping Soul daemon: {e}")

        # Stop Quantum AI daemon
        if self.quantum_ai_daemon:
            try:
                self.quantum_ai_daemon.stop()
                self.states["quantum_ai_daemon"].running = False
            except Exception as e:
                _logger.error(f"Error stopping Quantum AI daemon: {e}")

        # Stop VQPU daemon
        if self.vqpu_daemon:
            try:
                self.vqpu_daemon.stop()
                self.states["vqpu_daemon"].running = False
            except Exception as e:
                _logger.error(f"Error stopping VQPU daemon: {e}")

        # Stop orchestrator last
        if self.orchestrator:
            try:
                self.orchestrator.stop()
            except Exception as e:
                _logger.error(f"Error stopping orchestrator: {e}")

        _logger.info("All daemons stopped")
        return True

    def status(self) -> Dict[str, Any]:
        """Get unified status of all daemons."""
        with self._lock:
            daemon_states = {k: v.to_dict() for k, v in self.states.items()}

            # Update from adapters
            for daemon_id, adapter in self.adapters.items():
                if daemon_id in daemon_states:
                    daemon_states[daemon_id]["coherence"] = adapter.info.coherence
                    daemon_states[daemon_id]["bell_pairs_available"] = adapter.info.bell_pairs_available

            return {
                "running": self._running,
                "daemons": daemon_states,
                "mesh": self._mesh_status,
                "adapter_registry": len(DaemonAdapter._registry),
                "orchestrator_connected": self.orchestrator is not None,
                "timestamp": time.time(),
            }

    def run_quantum_simulation(self, sim_type: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Run a quantum simulation through the quantum sim daemon."""
        if not self.quantum_sim_daemon:
            return None

        try:
            result = self.quantum_sim_daemon.run_simulation(sim_type, **kwargs)
            return result.to_dict()
        except Exception as e:
            _logger.error(f"Simulation failed: {e}")
            return None

    def teleport_state(self, from_daemon: str, to_daemon: str, state: List[complex]) -> Optional[List[complex]]:
        """Teleport quantum state between daemons via entanglement mesh."""
        return DaemonAdapter.teleport_state(from_daemon, to_daemon, state)

    def broadcast_fidelity(self, source_daemon: str, fidelity: float, sacred_alignment: float):
        """Broadcast fidelity metrics to all daemons."""
        DaemonAdapter.broadcast_fidelity(source_daemon, fidelity, sacred_alignment)

    def get_daemon(self, daemon_id: str) -> Optional[Any]:
        """Get a specific daemon instance."""
        if daemon_id == "vqpu_daemon":
            return self.vqpu_daemon
        elif daemon_id == "quantum_ai_daemon":
            return self.quantum_ai_daemon
        elif daemon_id == "soul_daemon":
            return self.soul_daemon
        elif daemon_id == "quantum_sim_daemon":
            return self.quantum_sim_daemon
        elif daemon_id == "orchestrator":
            return self.orchestrator
        return None


# Singleton instance
_autonomous_system: Optional[AutonomousDaemonSystem] = None


def get_autonomous_system() -> AutonomousDaemonSystem:
    """Get or create the singleton autonomous system instance."""
    global _autonomous_system
    if _autonomous_system is None:
        _autonomous_system = AutonomousDaemonSystem()
    return _autonomous_system


def wire_all_daemons() -> AutonomousDaemonSystem:
    """Initialize and wire all daemons together."""
    system = get_autonomous_system()
    system.initialize()
    return system


def get_unified_status() -> Dict[str, Any]:
    """Get unified status of all daemons."""
    system = get_autonomous_system()
    return system.status()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="L104 Autonomous Daemon System")
    parser.add_argument("--status", action="store_true", help="Print status and exit")
    parser.add_argument("--start", action="store_true", help="Start all daemons")
    parser.add_argument("--stop", action="store_true", help="Stop all daemons")
    args = parser.parse_args()

    system = get_autonomous_system()

    if args.status:
        system.initialize()
        print(json.dumps(system.status(), indent=2))
    elif args.start:
        system.initialize()
        system.start_all()
        print("All daemons started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            system.stop_all()
    elif args.stop:
        system.stop_all()
        print("All daemons stopped.")
    else:
        parser.print_help()