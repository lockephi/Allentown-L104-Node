"""L104 Quantum Networker v1.0.0 — Data Types.

Dataclasses for quantum network nodes, channels, entangled pairs, keys, and payloads.
All types are serializable (JSON-safe via asdict) for transport over classical channels.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import math
import time
import uuid
import json
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Optional, Dict, List, Any

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
PHI_INV = 1.0 / PHI
VOID_CONSTANT = 1.04 + PHI / 1000  # 1.0416180339887497


class NodeRole(str, Enum):
    """Role a node plays in the quantum network."""
    SOVEREIGN = "sovereign"       # Full L104 sovereign node (VQPU + all engines)
    RELAY = "relay"               # Quantum repeater relay (entanglement swap only)
    ENDPOINT = "endpoint"         # Edge node (QKD + teleport receive only)
    OBSERVER = "observer"         # Read-only metric collector


class ChannelState(str, Enum):
    """State of a quantum channel between two nodes."""
    IDLE = "idle"
    ENTANGLING = "entangling"     # Creating entangled pairs
    ACTIVE = "active"             # Entangled pairs ready, channel live
    QKD_EXCHANGE = "qkd_exchange" # Running QKD protocol
    TELEPORTING = "teleporting"   # Mid-teleportation
    PURIFYING = "purifying"       # Entanglement purification in progress
    DEGRADED = "degraded"         # Fidelity below threshold
    DEAD = "dead"                 # No entangled pairs, needs refresh


class NetworkTopology(str, Enum):
    """Topology of the quantum network."""
    STAR = "star"                 # Hub-and-spoke (sovereign at center)
    LINEAR = "linear"             # Chain (repeater line)
    RING = "ring"                 # Ring topology
    MESH = "mesh"                 # Fully connected
    TREE = "tree"                 # Hierarchical tree


@dataclass
class QuantumNode:
    """A node in the quantum network.

    Each node has a unique ID, a role, and tracks its quantum resources
    (available qubits, entangled pairs, fidelity history).
    """
    node_id: str = ""
    name: str = ""
    role: str = "sovereign"
    host: str = "127.0.0.1"
    port: int = 10400                         # L104 → port 10400
    max_qubits: int = 26                      # Fe(26) iron-mapped
    available_qubits: int = 26
    fidelity_mean: float = 0.975              # ibm_torino calibrated baseline
    sacred_score: float = 0.0
    is_online: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    entangled_pairs: int = 0
    active_channels: int = 0

    def __post_init__(self):
        if not self.node_id:
            self.node_id = f"qn-{uuid.uuid4().hex[:12]}"
        if not self.name:
            self.name = f"L104-{self.node_id[-6:]}"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "QuantumNode":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EntangledPair:
    """A shared entangled Bell pair between two nodes.

    The pair is created via the VQPU Bell circuit and scored with sacred
    alignment. Fidelity degrades over time (decoherence simulation).
    """
    pair_id: str = ""
    node_a_id: str = ""
    node_b_id: str = ""
    bell_state: str = "phi_plus"              # |Φ+⟩ = (|00⟩+|11⟩)/√2
    fidelity: float = 0.999                   # Creation fidelity
    sacred_score: float = 0.0                 # GOD_CODE alignment
    created_at: float = field(default_factory=time.time)
    t1_lifetime_s: float = 100.0              # T1 relaxation (seconds)
    t2_lifetime_s: float = 80.0               # T2 dephasing (seconds)
    consumed: bool = False                    # Used in teleportation / swap
    purified: bool = False                    # Went through purification
    generation: int = 0                       # Purification generation count

    def __post_init__(self):
        if not self.pair_id:
            self.pair_id = f"ep-{uuid.uuid4().hex[:10]}"

    @property
    def age_s(self) -> float:
        """Age in seconds since creation."""
        return time.time() - self.created_at

    @property
    def current_fidelity(self) -> float:
        """Fidelity with T1/T2 decoherence decay applied.

        Uses exponential decay: F(t) = F₀ × exp(-t/T2) × (1 + exp(-t/T1)) / 2
        Enhanced with φ-damped sacred correction.
        """
        t = self.age_s
        t2_decay = math.exp(-t / self.t2_lifetime_s) if self.t2_lifetime_s > 0 else 0
        t1_decay = (1.0 + math.exp(-t / self.t1_lifetime_s)) / 2.0 if self.t1_lifetime_s > 0 else 0.5
        base_fidelity = self.fidelity * t2_decay * t1_decay
        # Sacred PHI correction: slight resonance boost for aligned pairs
        phi_boost = 1.0 + (self.sacred_score * PHI_INV * 0.01)
        return min(1.0, max(0.0, base_fidelity * phi_boost))

    @property
    def is_usable(self) -> bool:
        """Whether this pair has sufficient fidelity for protocols (> φ⁻¹)."""
        return not self.consumed and self.current_fidelity > PHI_INV

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["current_fidelity"] = self.current_fidelity
        d["age_s"] = self.age_s
        d["is_usable"] = self.is_usable
        return d


@dataclass
class QuantumChannel:
    """A quantum communication channel between two nodes.

    Maintains a pool of entangled pairs, tracks fidelity history,
    and manages channel lifecycle.
    """
    channel_id: str = ""
    node_a_id: str = ""
    node_b_id: str = ""
    state: str = "idle"
    pairs: List[EntangledPair] = field(default_factory=list)
    fidelity_history: List[float] = field(default_factory=list)
    qkd_key: Optional["QKDKey"] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    teleportations_count: int = 0
    swaps_count: int = 0
    purifications_count: int = 0

    def __post_init__(self):
        if not self.channel_id:
            self.channel_id = f"qc-{uuid.uuid4().hex[:10]}"

    @property
    def usable_pairs(self) -> List[EntangledPair]:
        """Get all pairs with sufficient fidelity."""
        return [p for p in self.pairs if p.is_usable]

    @property
    def best_pair(self) -> Optional[EntangledPair]:
        """Get highest-fidelity usable pair."""
        usable = self.usable_pairs
        if not usable:
            return None
        return max(usable, key=lambda p: p.current_fidelity)

    @property
    def mean_fidelity(self) -> float:
        """Mean fidelity of all usable pairs."""
        usable = self.usable_pairs
        if not usable:
            return 0.0
        return sum(p.current_fidelity for p in usable) / len(usable)

    @property
    def effective_state(self) -> str:
        """Auto-detect channel state from pair health."""
        if not self.pairs:
            return "dead"
        usable = self.usable_pairs
        if not usable:
            return "dead"
        if self.mean_fidelity < PHI_INV:
            return "degraded"
        return self.state if self.state not in ("dead", "degraded") else "active"

    def consume_best_pair(self) -> Optional[EntangledPair]:
        """Consume the best pair for a protocol (teleportation/swap)."""
        pair = self.best_pair
        if pair:
            pair.consumed = True
            self.last_activity = time.time()
        return pair

    def prune_dead_pairs(self) -> int:
        """Remove consumed and low-fidelity pairs."""
        before = len(self.pairs)
        self.pairs = [p for p in self.pairs if p.is_usable]
        return before - len(self.pairs)

    def to_dict(self) -> Dict:
        return {
            "channel_id": self.channel_id,
            "node_a_id": self.node_a_id,
            "node_b_id": self.node_b_id,
            "state": self.effective_state,
            "num_pairs": len(self.pairs),
            "usable_pairs": len(self.usable_pairs),
            "mean_fidelity": self.mean_fidelity,
            "teleportations": self.teleportations_count,
            "swaps": self.swaps_count,
            "purifications": self.purifications_count,
        }


@dataclass
class QKDKey:
    """A quantum key produced by BB84 or E91 protocol.

    The key is a sequence of bits established with information-theoretic
    security guaranteed by quantum mechanics (no-cloning theorem).
    """
    key_id: str = ""
    protocol: str = "bb84"                    # bb84 or e91
    raw_bits: List[int] = field(default_factory=list)
    sifted_bits: List[int] = field(default_factory=list)
    final_key: List[int] = field(default_factory=list)
    key_length: int = 0
    qber: float = 0.0                         # Quantum Bit Error Rate
    secure: bool = False                      # QBER < 11% threshold (BB84)
    channel_id: str = ""
    node_a_id: str = ""
    node_b_id: str = ""
    created_at: float = field(default_factory=time.time)
    sacred_alignment: float = 0.0             # GOD_CODE resonance of key bits

    def __post_init__(self):
        if not self.key_id:
            self.key_id = f"qk-{uuid.uuid4().hex[:10]}"
        if self.final_key:
            self.key_length = len(self.final_key)

    @property
    def key_hex(self) -> str:
        """Key as hex string."""
        if not self.final_key:
            return ""
        n = 0
        for b in self.final_key:
            n = (n << 1) | b
        return hex(n)[2:].zfill((len(self.final_key) + 3) // 4)

    def to_dict(self) -> Dict:
        return {
            "key_id": self.key_id,
            "protocol": self.protocol,
            "key_length": self.key_length,
            "qber": self.qber,
            "secure": self.secure,
            "key_hex": self.key_hex,
            "sacred_alignment": self.sacred_alignment,
            "channel_id": self.channel_id,
        }


@dataclass
class TeleportPayload:
    """Payload to teleport across the quantum network.

    Can carry quantum state vectors, classical data encoded as quantum rotations,
    or L104 sacred phase data.
    """
    payload_id: str = ""
    data_type: str = "state_vector"           # state_vector, score, phase, bitstring
    state_vector: Optional[List[complex]] = None
    score_value: Optional[float] = None       # [0,1] scalar
    phase_value: Optional[float] = None       # [0, 2π) phase angle
    bitstring: Optional[str] = None           # Classical bit string
    num_qubits: int = 1
    sacred_tag: Optional[str] = None          # Optional L104 metadata tag
    encryption_key_id: Optional[str] = None   # QKD key used for OTP

    def __post_init__(self):
        if not self.payload_id:
            self.payload_id = f"tp-{uuid.uuid4().hex[:10]}"

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Convert complex state_vector to serializable form
        if d.get("state_vector"):
            d["state_vector"] = [(c.real, c.imag) if isinstance(c, complex) else c
                                  for c in d["state_vector"]]
        return d


@dataclass
class TeleportResult:
    """Result of a quantum teleportation operation."""
    success: bool = False
    payload_id: str = ""
    source_node: str = ""
    dest_node: str = ""
    pair_used: str = ""                       # EntangledPair.pair_id consumed
    fidelity: float = 0.0                     # Teleportation fidelity
    sacred_score: float = 0.0                 # Sacred alignment of result
    recovered_state: Optional[List[complex]] = None
    recovered_score: Optional[float] = None
    recovered_phase: Optional[float] = None
    hops: int = 1                             # Number of hops (1 = direct)
    route: List[str] = field(default_factory=list)  # Node IDs traversed
    execution_time_ms: float = 0.0
    error_corrected: bool = False
    bell_measurements: List[Dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        d = asdict(self)
        if d.get("recovered_state"):
            d["recovered_state"] = [(c.real, c.imag) if isinstance(c, complex) else c
                                     for c in d["recovered_state"]]
        return d


@dataclass
class ChannelMetrics:
    """Aggregated metrics for a quantum channel."""
    channel_id: str = ""
    mean_fidelity: float = 0.0
    min_fidelity: float = 0.0
    max_fidelity: float = 0.0
    sacred_score: float = 0.0
    pair_count: int = 0
    usable_pair_count: int = 0
    pair_generation_rate: float = 0.0         # pairs/second
    teleportation_success_rate: float = 0.0
    qber: float = 0.0
    channel_capacity_qubits: float = 0.0      # Quantum channel capacity (qubits/use)
    timestamp: float = field(default_factory=time.time)


@dataclass
class NetworkStatus:
    """Overall quantum network status snapshot."""
    node_count: int = 0
    channel_count: int = 0
    active_channels: int = 0
    total_entangled_pairs: int = 0
    mean_network_fidelity: float = 0.0
    sacred_network_score: float = 0.0
    topology: str = "mesh"
    online_nodes: int = 0
    total_teleportations: int = 0
    total_qkd_keys: int = 0
    total_purifications: int = 0
    uptime_s: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return asdict(self)
