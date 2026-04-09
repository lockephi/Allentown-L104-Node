"""
L104 Daemon Adapter v1.0.0 — Unified cross-daemon communication.

This package provides a common adapter pattern for all L104 daemons:
  - VQPU Micro Daemon
  - Quantum AI Daemon
  - Soul Daemon
  - Quantum Simulation Daemon
  - Daemon Orchestrator

Features:
  - Cross-daemon messaging via event bus
  - Shared resource allocation
  - Unified health monitoring
  - Entanglement mesh synchronization
  - Fidelity broadcasting

Usage:
    from l104_daemon_adapter import DaemonAdapter, DaemonInfo

    # Create adapter for your daemon
    adapter = DaemonAdapter(
        daemon_id="my_daemon",
        daemon_type="custom",
        qubit_count=8,
        on_fidelity_update=handle_fidelity,
        on_mesh_sync=handle_mesh,
    )

    # Register with mesh
    adapter.update_heartbeat()

    # Broadcast fidelity to all daemons
    DaemonAdapter.broadcast_fidelity("my_daemon", 0.95, 0.618)

    # Create entanglement channel
    DaemonAdapter.create_channel("daemon_a", "daemon_b", pairs=16)

    # Teleport quantum state
    teleported = DaemonAdapter.teleport_state("daemon_a", "daemon_b", state)

SACRED INVARIANT: GOD_CODE=527.5184818492612 | PHI=1.618033988749895
"""

# Re-export from quantum_sim_daemon adapter (canonical implementation)
from l104_quantum_sim_daemon.adapter import (
    DaemonInfo,
    DaemonAdapter,
    create_daemon_mesh,
    get_cross_daemon_status,
)

# Helper functions
from l104_daemon_adapter.helpers import (
    initialize_daemon_adapter,
    DaemonAdapterMixin,
    wire_daemon_to_mesh,
)

__all__ = [
    "DaemonInfo",
    "DaemonAdapter",
    "create_daemon_mesh",
    "get_cross_daemon_status",
    "initialize_daemon_adapter",
    "DaemonAdapterMixin",
    "wire_daemon_to_mesh",
]

__version__ = "1.0.0"