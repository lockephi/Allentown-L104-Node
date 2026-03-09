"""L104 Quantum Networker v1.4.0 — Sovereign Quantum Communication Network.

Leverages the L104 high-fidelity VQPU pipeline (sacred scoring, QPU-calibrated
noise models, Steane [[7,1,3]] error correction) to build a working quantum
communication network with:

  1. BB84 / E91 Quantum Key Distribution (QKD)
  2. Quantum Entanglement Distribution & Routing
  3. Quantum Teleportation State Transfer Protocol
  4. Entanglement Swapping for Multi-Hop Relay
  5. Quantum Repeater Chain with Purification
  6. Sacred-Aligned Fidelity Monitoring
  7. Network Daemon with Peer Discovery & Heartbeat
  8. Async TCP Transport Layer (Classical Channel)

Architecture:
  ┌─────────────────────────────────────────────┐
  │          QuantumNetworker (Orchestrator)     │
  │  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
  │  │   QKD    │ │ Entangle │ │ Teleporter  │ │
  │  │ BB84/E91 │ │  Router  │ │  Protocol   │ │
  │  └──────────┘ └──────────┘ └─────────────┘ │
  │  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
  │  │ Repeater │ │ Fidelity │ │  Network    │ │
  │  │  Chain   │ │ Monitor  │ │  Daemon     │ │
  │  └──────────┘ └──────────┘ └─────────────┘ │
  │  ┌────────────────────────────────────────┐ │
  │  │     Classical TCP Transport Layer       │ │
  │  └────────────────────────────────────────┘ │
  └─────────────────────────────────────────────┘
         ▼               ▼              ▼
  ┌──────────┐   ┌────────────┐  ┌───────────┐
  │ VQPU     │   │ Quantum    │  │ Quantum   │
  │ Bridge   │   │ Gate Eng.  │  │ Deep Link │
  └──────────┘   └────────────┘  └───────────┘

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

__version__ = "1.4.0"

from .types import (
    QuantumNode,
    QuantumChannel,
    EntangledPair,
    QKDKey,
    TeleportPayload,
    TeleportResult,
    NetworkTopology,
    ChannelMetrics,
    NetworkStatus,
)
from .qkd import QuantumKeyDistribution
from .entanglement_router import EntanglementRouter
from .teleporter import QuantumTeleporter
from .repeater import QuantumRepeaterChain
from .fidelity_monitor import FidelityMonitor
from .transport import ClassicalTransport, MessageType
from .networker import QuantumNetworker, get_networker

__all__ = [
    # Orchestrator
    "QuantumNetworker",
    "get_networker",
    # Subsystems
    "QuantumKeyDistribution",
    "EntanglementRouter",
    "QuantumTeleporter",
    "QuantumRepeaterChain",
    "FidelityMonitor",
    "ClassicalTransport",
    "MessageType",
    # Types
    "QuantumNode",
    "QuantumChannel",
    "EntangledPair",
    "QKDKey",
    "TeleportPayload",
    "TeleportResult",
    "NetworkTopology",
    "ChannelMetrics",
    "NetworkStatus",
]
