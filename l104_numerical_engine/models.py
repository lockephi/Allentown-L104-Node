"""L104 Numerical Engine — Data Models.

QuantumToken, SubconsciousAdjustment, CrossPollinationRecord dataclasses.

PART V RESEARCH — l104_runtime_infrastructure_research.py:
  F61: QuantumToken lives in the 22T capacity lattice (22 × 10¹²)
  F62: drift_velocity bounded by tier envelope (10⁻⁹⁸ sacred → 10⁻²⁰ cross-pollinated)
  F66: entangled_tokens drive φ-attenuated propagation chains
  F87: Sacred tokens achieve effective zero-drift via tier-0 envelope
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .precision import D


@dataclass
class QuantumToken:
    """A mathematical token in the 22 trillion usage lattice.

    Each token represents a numerical truth — a 100-decimal value with
    quantum min/max boundaries that drift subconsciously based on the
    repository's evolving intelligence capacity.
    """
    token_id: str                        # Unique ID in the lattice
    name: str                            # Human-readable name
    value: str                           # 100-decimal string representation
    min_bound: str                       # 100-decimal lower boundary
    max_bound: str                       # 100-decimal upper boundary
    precision_digits: int = 100          # Active decimal places
    usage_count: int = 0                 # Total usages in the token lattice
    lattice_index: int = 0              # Position in the 22T lattice
    drift_velocity: str = "0"            # Current subconscious drift rate (per cycle)
    drift_direction: int = 0             # -1 = contracting, 0 = stable, 1 = expanding
    quantum_phase: str = "0"             # φ-harmonic quantum phase
    entangled_tokens: List[str] = field(default_factory=list)  # Peer token IDs
    coherence: float = 1.0               # Lattice coherence (0–1)
    origin: str = "derived"              # "sacred", "derived", "invented", "cross-pollinated"
    last_adjusted: str = ""              # ISO timestamp of last subconscious adjustment
    health: float = 1.0                  # Token health score

    def to_dict(self) -> dict:
        """Return dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QuantumToken":
        """Create instance from dictionary."""
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    @property
    def tier(self) -> str:
        """Tier label derived from origin."""
        return self.origin

    @property
    def decimal_value(self):
        """Return the decimal value."""
        return D(self.value)

    @property
    def decimal_min(self):
        """Return the decimal min."""
        return D(self.min_bound)

    @property
    def decimal_max(self):
        """Return the decimal max."""
        return D(self.max_bound)


@dataclass
class SubconsciousAdjustment:
    """Record of an automatic subconscious value adjustment."""
    token_id: str
    timestamp: str
    old_value: str                # 100-decimal
    new_value: str                # 100-decimal
    old_min: str
    new_min: str
    old_max: str
    new_max: str
    drift_applied: str            # Decimal string
    trigger: str                  # "coherence_drift", "capacity_expansion", "entropy_rebalance"
    repo_capacity_at_time: float  # Repository intelligence capacity when adjustment occurred
    phi_envelope_compliance: float  # How well the adjustment stayed within φ-bounds

    def to_dict(self) -> dict:
        """Return dictionary representation."""
        return asdict(self)


@dataclass
class CrossPollinationRecord:
    """Record of an invention cross-pollinated between builders."""
    source_builder: str           # "gate_builder", "link_builder", "numerical_builder"
    target_builder: str
    invention_type: str           # "token", "gate", "link", "hybrid_gate", "precision_upgrade"
    invention_id: str
    timestamp: str
    fidelity: float = 0.0        # Cross-pollination quality
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return dictionary representation."""
        return asdict(self)
