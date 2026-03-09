"""L104 Gate Engine — Data models: LogicGate, GateLink, ChronologEntry."""

import math
from dataclasses import dataclass, field, asdict
from typing import List

from .constants import (
    PHI, TAU, GOD_CODE, EULER_GAMMA, DRIFT_ENVELOPE,
)


@dataclass
class LogicGate:
    """Represents a single logic gate implementation across the system.

    ★ v5.0 Quantum Min/Max Dynamism:
      - dynamic_value: φ-derived gate value that evolves each cycle
      - min_bound / max_bound: auto-adjusting quantum boundaries
      - drift_velocity: current rate of value change per cycle
      - drift_direction: +1 (expanding) or -1 (compacting)
      - quantum_phase: oscillation phase in [0, 2π)
      - evolution_count: how many cycles this gate has evolved through
    """
    name: str
    language: str                     # python, swift, javascript
    source_file: str                  # relative path
    line_number: int
    gate_type: str                    # function, class, method
    signature: str                    # Full function/class signature
    parameters: List[str] = field(default_factory=list)
    docstring: str = ""
    complexity: int = 0               # Cyclomatic complexity estimate
    entropy_score: float = 0.0        # φ-modulated entropy of the gate
    quantum_links: List[str] = field(default_factory=list)  # Cross-file links
    hash: str = ""                    # Content hash for change detection
    last_seen: str = ""               # ISO timestamp
    test_status: str = "untested"     # untested, passed, failed
    # ★ v5.0 Quantum Min/Max Dynamism fields
    dynamic_value: float = 0.0        # φ-derived evolving value
    min_bound: float = 0.0            # Auto-adjusting lower boundary
    max_bound: float = 0.0            # Auto-adjusting upper boundary
    drift_velocity: float = 0.0       # Rate of change per cycle
    drift_direction: int = 1          # +1 expand, -1 compact
    quantum_phase: float = 0.0        # Oscillation phase [0, 2π)
    evolution_count: int = 0          # Cycles evolved
    resonance_score: float = 0.0      # Alignment with sacred constants

    def __post_init__(self):
        """Initialize dynamic bounds from complexity and entropy if not set."""
        if self.dynamic_value == 0.0 and (self.complexity > 0 or self.entropy_score > 0):
            self._initialize_dynamism()

    def _initialize_dynamism(self):
        """Compute initial dynamic value and bounds from gate properties."""
        # Dynamic value = φ-weighted combination of complexity + entropy
        self.dynamic_value = (
            self.complexity * PHI +
            self.entropy_score * TAU +
            len(self.quantum_links) * EULER_GAMMA * 0.1
        )
        # Bounds = ± φ envelope around the dynamic value
        envelope = max(abs(self.dynamic_value) * DRIFT_ENVELOPE["amplitude"], PHI)
        self.min_bound = self.dynamic_value - envelope
        self.max_bound = self.dynamic_value + envelope
        # Initial phase from hash seed
        if self.hash:
            seed = int(self.hash[:8], 16) if self.hash else 0
            self.quantum_phase = (seed % 10000) / 10000.0 * 2 * math.pi
        # Initial drift velocity
        self.drift_velocity = DRIFT_ENVELOPE["max_velocity"] * math.sin(self.quantum_phase)
        # Resonance with God Code
        if self.dynamic_value > 0:
            self.resonance_score = abs(math.cos(
                self.dynamic_value * math.pi / GOD_CODE
            ))

    def evolve(self):
        """Evolve this gate's dynamic value by one cycle — φ-harmonic drift."""
        self.evolution_count += 1
        # Phase advance
        self.quantum_phase = (
            self.quantum_phase + DRIFT_ENVELOPE["frequency"] * 0.1
        ) % (2 * math.pi)
        # Drift velocity with damping
        target_velocity = DRIFT_ENVELOPE["max_velocity"] * math.sin(self.quantum_phase)
        damping = DRIFT_ENVELOPE["damping"]
        self.drift_velocity = (
            self.drift_velocity * (1 - damping) + target_velocity * damping
        )
        # Apply drift
        new_value = self.dynamic_value + self.drift_velocity * self.drift_direction
        # Boundary enforcement — bounce off walls
        if new_value > self.max_bound:
            new_value = self.max_bound
            self.drift_direction = -1
        elif new_value < self.min_bound:
            new_value = self.min_bound
            self.drift_direction = 1
        self.dynamic_value = new_value
        # Update resonance
        if abs(self.dynamic_value) > 1e-10:
            self.resonance_score = abs(math.cos(
                self.dynamic_value * math.pi / GOD_CODE
            ))
        # Adaptive bounds — expand if gate is performing well
        if self.test_status == "passed":
            expansion = PHI * 0.001 * self.evolution_count
            self.max_bound += expansion
            self.min_bound -= expansion * TAU

    def to_dict(self) -> dict:
        """Serialize this LogicGate to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LogicGate":
        """Deserialize a LogicGate from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GateLink:
    """Quantum link between two gate implementations."""
    source_gate: str
    target_gate: str
    link_type: str          # "calls", "inherits", "mirrors", "entangles"
    strength: float = 1.0   # φ-weighted link strength
    evidence: str = ""      # Where was this link detected


@dataclass
class ChronologEntry:
    """Chronological record of gate changes."""
    timestamp: str
    gate_name: str
    event: str              # "discovered", "modified", "removed", "test_passed", "test_failed"
    details: str = ""
    file_hash: str = ""
