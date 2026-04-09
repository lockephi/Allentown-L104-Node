"""
L104 Daemon Adapter — Unified interface for cross-daemon communication.

Provides a common adapter pattern for all L104 daemons:
  - VQPU Micro Daemon
  - Quantum AI Daemon
  - Soul Daemon
  - Quantum Simulation Daemon (new)
  - Orchestrator

This enables:
  - Cross-daemon messaging via event bus
  - Shared resource allocation
  - Unified health monitoring
  - Entanglement mesh synchronization
  - Fidelity broadcasting
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import json

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895

_logger = logging.getLogger("L104_DAEMON_ADAPTER")


@dataclass
class DaemonInfo:
    """Information about a registered daemon."""
    daemon_id: str
    daemon_type: str
    qubit_count: int
    coherence: float = 1.0
    last_heartbeat: float = field(default_factory=time.time)
    bell_pairs_available: int = 8
    fidelity: float = 1.0
    sacred_alignment: float = 0.618

    def to_dict(self):
        return {
            "daemon_id": self.daemon_id,
            "daemon_type": self.daemon_type,
            "qubit_count": self.qubit_count,
            "coherence": round(self.coherence, 4),
            "last_heartbeat": self.last_heartbeat,
            "bell_pairs_available": self.bell_pairs_available,
            "fidelity": round(self.fidelity, 6),
            "sacred_alignment": round(self.sacred_alignment, 6),
        }


class DaemonAdapter:
    """
    Adapter for cross-daemon communication.

    Each daemon uses this adapter to:
    1. Register with the orchestrator
    2. Send/receive messages via event bus
    3. Share metrics (fidelity, coherence, sacred alignment)
    4. Synchronize entanglement mesh
    5. Request Bell pairs from other daemons
    """

    _registry: Dict[str, 'DaemonAdapter'] = {}
    _mesh_channels: Dict[str, Dict] = {}
    _event_bus: Dict[str, List[Callable]] = {}
    _lock = threading.RLock()

    def __init__(
        self,
        daemon_id: str,
        daemon_type: str,
        qubit_count: int = 4,
        on_fidelity_update: Optional[Callable] = None,
        on_mesh_sync: Optional[Callable] = None,
        on_quantum_job: Optional[Callable] = None,
    ):
        self.info = DaemonInfo(
            daemon_id=daemon_id,
            daemon_type=daemon_type,
            qubit_count=qubit_count,
        )
        self.on_fidelity_update = on_fidelity_update
        self.on_mesh_sync = on_mesh_sync
        self.on_quantum_job = on_quantum_job

        # Register this adapter
        with self._lock:
            DaemonAdapter._registry[daemon_id] = self

        _logger.info(f"DaemonAdapter: Registered {daemon_id} ({daemon_type})")

    @classmethod
    def get_adapter(cls, daemon_id: str) -> Optional['DaemonAdapter']:
        """Get adapter by daemon ID."""
        return cls._registry.get(daemon_id)

    @classmethod
    def get_all_adapters(cls) -> Dict[str, 'DaemonAdapter']:
        """Get all registered adapters."""
        return cls._registry.copy()

    @classmethod
    def broadcast_fidelity(cls, source_id: str, fidelity: float, sacred_alignment: float):
        """Broadcast fidelity metrics to all connected daemons."""
        with cls._lock:
            for daemon_id, adapter in cls._registry.items():
                if daemon_id != source_id and adapter.on_fidelity_update:
                    try:
                        adapter.on_fidelity_update(source_id, fidelity, sacred_alignment)
                    except Exception as e:
                        _logger.error(f"Failed to send fidelity to {daemon_id}: {e}")

    @classmethod
    def broadcast_mesh_sync(cls, source_id: str, mesh_status: Dict[str, Any]):
        """Broadcast mesh synchronization to all daemons."""
        with cls._lock:
            for daemon_id, adapter in cls._registry.items():
                if daemon_id != source_id and adapter.on_mesh_sync:
                    try:
                        adapter.on_mesh_sync(source_id, mesh_status)
                    except Exception as e:
                        _logger.error(f"Failed to send mesh sync to {daemon_id}: {e}")

    @classmethod
    def create_channel(cls, node_a: str, node_b: str, pairs: int = 8) -> str:
        """Create an entanglement channel between two daemons."""
        channel_id = f"{node_a}:{node_b}"
        with cls._lock:
            cls._mesh_channels[channel_id] = {
                "node_a": node_a,
                "node_b": node_b,
                "bell_pairs": pairs,
                "fidelity": 0.95,
                "created": time.time(),
            }
        _logger.info(f"Mesh: Created channel {channel_id} with {pairs} Bell pairs")
        return channel_id

    @classmethod
    def teleport_state(cls, from_node: str, to_node: str, state: List[complex]) -> Optional[List[complex]]:
        """Teleport quantum state via Bell pair channel."""
        channel_id = f"{from_node}:{to_node}"
        reverse_id = f"{to_node}:{from_node}"

        with cls._lock:
            channel = cls._mesh_channels.get(channel_id) or cls._mesh_channels.get(reverse_id)
            if not channel or channel["bell_pairs"] < 1:
                return None

            channel["bell_pairs"] -= 1
            fidelity = channel["fidelity"]

        # Apply PHI-attenuated fidelity
        teleported = [complex(c.real * fidelity, c.imag * fidelity) for c in state]
        return teleported

    @classmethod
    def replenish_channels(cls):
        """Replenish Bell pairs in all channels."""
        with cls._lock:
            for channel_id in cls._mesh_channels:
                cls._mesh_channels[channel_id]["bell_pairs"] = min(
                    16, cls._mesh_channels[channel_id]["bell_pairs"] + 4
                )

    @classmethod
    def get_mesh_status(cls) -> Dict[str, Any]:
        """Get overall mesh status."""
        with cls._lock:
            return {
                "nodes": len(cls._registry),
                "channels": len(cls._mesh_channels),
                "total_bell_pairs": sum(c["bell_pairs"] for c in cls._mesh_channels.values()),
                "avg_fidelity": sum(c["fidelity"] for c in cls._mesh_channels.values()) / max(1, len(cls._mesh_channels)),
            }

    def update_heartbeat(self):
        """Update heartbeat timestamp."""
        self.info.last_heartbeat = time.time()

    def update_fidelity(self, fidelity: float, sacred_alignment: float):
        """Update fidelity metrics."""
        self.info.fidelity = fidelity
        self.info.sacred_alignment = sacred_alignment
        self.update_heartbeat()

    def update_coherence(self, coherence: float):
        """Update coherence score."""
        self.info.coherence = coherence
        self.update_heartbeat()

    def receive_fidelity(self, source_id: str, fidelity: float, sacred_alignment: float):
        """Receive fidelity update from another daemon."""
        if self.on_fidelity_update:
            self.on_fidelity_update(source_id, fidelity, sacred_alignment)

    def receive_mesh_sync(self, source_id: str, mesh_status: Dict[str, Any]):
        """Receive mesh synchronization from another daemon."""
        if self.on_mesh_sync:
            self.on_mesh_sync(source_id, mesh_status)

    def receive_quantum_job(self, job: Dict[str, Any]):
        """Receive quantum job from another daemon."""
        if self.on_quantum_job:
            self.on_quantum_job(job)

    def submit_quantum_job(self, target_id: str, job: Dict[str, Any]) -> bool:
        """Submit a quantum job to another daemon."""
        target = DaemonAdapter.get_adapter(target_id)
        if target and target.on_quantum_job:
            target.receive_quantum_job(job)
            return True
        return False

    # === Event Bus ===

    @classmethod
    def subscribe(cls, event_type: str, handler: Callable):
        """Subscribe to an event type."""
        with cls._lock:
            if event_type not in cls._event_bus:
                cls._event_bus[event_type] = []
            cls._event_bus[event_type].append(handler)

    @classmethod
    def unsubscribe(cls, event_type: str, handler: Callable):
        """Unsubscribe from an event type."""
        with cls._lock:
            if event_type in cls._event_bus and handler in cls._event_bus[event_type]:
                cls._event_bus[event_type].remove(handler)

    @classmethod
    def publish(cls, event_type: str, data: Dict[str, Any]):
        """Publish an event to all subscribers."""
        with cls._lock:
            handlers = cls._event_bus.get(event_type, []).copy()

        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                _logger.error(f"Event handler error: {e}")

    def to_dict(self) -> Dict[str, Any]:
        return self.info.to_dict()

    # ------------------------------------------------------------------
    # Lifecycle hooks — called by quantum AI / sim daemons (EVO_76)
    # ------------------------------------------------------------------

    def on_cycle_start(self) -> None:
        """Called at the beginning of each daemon cycle."""
        self.update_heartbeat()
        self.publish("cycle_start", {
            "daemon_id": self.info.daemon_id,
            "timestamp": time.time(),
        })

    def on_cycle_end(self, success: bool, duration_ms: float,
                     cpu_percent: float = 0.0, memory_mb: float = 0.0) -> None:
        """Called at the end of each daemon cycle."""
        self.publish("cycle_end", {
            "daemon_id": self.info.daemon_id,
            "success": success,
            "duration_ms": duration_ms,
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "timestamp": time.time(),
        })

    def emit_error(self, event_type: str, message: str, level: str = "error") -> None:
        """Publish a daemon error event to the mesh."""
        self.publish("daemon_error", {
            "daemon_id": self.info.daemon_id,
            "event_type": event_type,
            "message": message,
            "level": level,
            "timestamp": time.time(),
        })

    def emit_fidelity_alert(self, fidelity: float, trending: str = "stable",
                            sim_count: int = 0, error_count: int = 0) -> None:
        """Broadcast fidelity metrics to the mesh."""
        sacred_alignment = fidelity * 0.618033988749895  # PHI-inverse weighting
        self.update_fidelity(fidelity, sacred_alignment)
        self.publish("fidelity_alert", {
            "daemon_id": self.info.daemon_id,
            "fidelity": fidelity,
            "trending": trending,
            "sim_count": sim_count,
            "error_count": error_count,
            "sacred_alignment": sacred_alignment,
            "timestamp": time.time(),
        })


# === Convenience functions ===

def create_daemon_mesh() -> None:
    """Create entanglement channels between all registered daemons."""
    adapters = list(DaemonAdapter.get_all_adapters().values())
    for i, adapter_a in enumerate(adapters):
        for adapter_b in adapters[i+1:]:
            DaemonAdapter.create_channel(adapter_a.info.daemon_id, adapter_b.info.daemon_id)

    DaemonAdapter.broadcast_mesh_sync("orchestrator", DaemonAdapter.get_mesh_status())


def get_cross_daemon_status() -> Dict[str, Any]:
    """Get status of all daemons and mesh."""
    return {
        "daemons": {da_id: adapter.to_dict() for da_id, adapter in DaemonAdapter.get_all_adapters().items()},
        "mesh": DaemonAdapter.get_mesh_status(),
    }