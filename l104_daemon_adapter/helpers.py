"""
L104 Daemon Adapter Helpers — Initialization utilities for daemon adapters.

Provides helper functions for daemons to initialize and use the DaemonAdapter
for cross-daemon communication.
"""

import logging
from typing import Any, Callable, Dict, Optional

_logger = logging.getLogger("L104_DAEMON_ADAPTER_HELPERS")

# Sacred constants (re-exported for convenience)
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895


def initialize_daemon_adapter(
    daemon_id: str,
    daemon_type: str,
    qubit_count: int = 8,
    on_fidelity_update: Optional[Callable] = None,
    on_mesh_sync: Optional[Callable] = None,
    on_quantum_job: Optional[Callable] = None,
) -> Optional[Any]:
    """
    Initialize a DaemonAdapter for cross-daemon communication.

    Args:
        daemon_id: Unique identifier for this daemon (e.g., "quantum_ai_daemon")
        daemon_type: Type of daemon (e.g., "quantum_ai", "soul", "vqpu", "quantum_sim")
        qubit_count: Number of qubits this daemon manages
        on_fidelity_update: Callback for fidelity updates from other daemons
        on_mesh_sync: Callback for mesh synchronization from other daemons
        on_quantum_job: Callback for quantum jobs from other daemons

    Returns:
        DaemonAdapter instance or None if import failed

    Usage:
        adapter = initialize_daemon_adapter(
            daemon_id="soul_daemon",
            daemon_type="soul",
            qubit_count=1,
            on_fidelity_update=self._on_fidelity_update,
            on_mesh_sync=self._on_mesh_sync,
        )
    """
    try:
        from l104_daemon_adapter import DaemonAdapter
        adapter = DaemonAdapter(
            daemon_id=daemon_id,
            daemon_type=daemon_type,
            qubit_count=qubit_count,
            on_fidelity_update=on_fidelity_update,
            on_mesh_sync=on_mesh_sync,
            on_quantum_job=on_quantum_job,
        )
        _logger.info(f"DaemonAdapter initialized for {daemon_id} ({daemon_type})")
        return adapter
    except ImportError as e:
        _logger.warning(f"DaemonAdapter not available: {e}")
        return None


class DaemonAdapterMixin:
    """
    Mixin class providing daemon adapter functionality.

    Inherit from this class to add adapter capabilities to your daemon:

        class MyDaemon(DaemonAdapterMixin):
            def __init__(self):
                super().__init__()
                self._init_adapter(
                    daemon_id="my_daemon",
                    daemon_type="custom",
                    qubit_count=4,
                )

            def _on_fidelity_update(self, source_id, fidelity, sacred_alignment):
                # Handle fidelity update from another daemon
                pass

            def _on_mesh_sync(self, source_id, mesh_status):
                # Handle mesh synchronization
                pass
    """

    def _init_adapter(
        self,
        daemon_id: str,
        daemon_type: str,
        qubit_count: int = 8,
    ):
        """Initialize the adapter with callbacks."""
        self._adapter = initialize_daemon_adapter(
            daemon_id=daemon_id,
            daemon_type=daemon_type,
            qubit_count=qubit_count,
            on_fidelity_update=self._on_fidelity_update if hasattr(self, '_on_fidelity_update') else None,
            on_mesh_sync=self._on_mesh_sync if hasattr(self, '_on_mesh_sync') else None,
            on_quantum_job=self._on_quantum_job if hasattr(self, '_on_quantum_job') else None,
        )

    def update_adapter_heartbeat(self):
        """Update adapter heartbeat timestamp."""
        if self._adapter:
            self._adapter.update_heartbeat()

    def update_adapter_fidelity(self, fidelity: float, sacred_alignment: float):
        """Update adapter fidelity metrics and broadcast to mesh."""
        if self._adapter:
            self._adapter.update_fidelity(fidelity, sacred_alignment)
            from l104_daemon_adapter import DaemonAdapter
            DaemonAdapter.broadcast_fidelity(
                self._adapter.info.daemon_id,
                fidelity,
                sacred_alignment
            )

    def update_adapter_coherence(self, coherence: float):
        """Update adapter coherence score."""
        if self._adapter:
            self._adapter.update_coherence(coherence)

    def get_mesh_status(self) -> Dict[str, Any]:
        """Get current mesh status."""
        from l104_daemon_adapter import DaemonAdapter
        return DaemonAdapter.get_mesh_status()

    def create_mesh_channel(self, target_daemon: str, pairs: int = 8) -> Optional[str]:
        """Create entanglement channel with another daemon."""
        if self._adapter:
            from l104_daemon_adapter import DaemonAdapter
            return DaemonAdapter.create_channel(
                self._adapter.info.daemon_id,
                target_daemon,
                pairs=pairs
            )
        return None

    def teleport_to_daemon(self, target_daemon: str, state: list) -> Optional[list]:
        """Teleport quantum state to another daemon."""
        if self._adapter:
            from l104_daemon_adapter import DaemonAdapter
            return DaemonAdapter.teleport_state(
                self._adapter.info.daemon_id,
                target_daemon,
                state
            )
        return None


def wire_daemon_to_mesh(
    daemon,
    daemon_id: str,
    daemon_type: str,
    qubit_count: int = 8,
) -> bool:
    """
    Wire a daemon to the entanglement mesh.

    This is a convenience function that:
    1. Creates a DaemonAdapter for the daemon
    2. Registers it in the mesh
    3. Sets up the daemon's adapter attribute

    Args:
        daemon: The daemon instance (will have daemon._adapter set)
        daemon_id: Unique identifier
        daemon_type: Type of daemon
        qubit_count: Number of qubits

    Returns:
        True if successful, False otherwise
    """
    try:
        adapter = initialize_daemon_adapter(
            daemon_id=daemon_id,
            daemon_type=daemon_type,
            qubit_count=qubit_count,
            on_fidelity_update=getattr(daemon, '_on_fidelity_update', None),
            on_mesh_sync=getattr(daemon, '_on_mesh_sync', None),
            on_quantum_job=getattr(daemon, '_on_quantum_job', None),
        )
        if adapter:
            daemon._adapter = adapter
            return True
        return False
    except Exception as e:
        _logger.error(f"Failed to wire daemon {daemon_id}: {e}")
        return False