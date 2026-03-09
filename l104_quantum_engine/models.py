"""
L104 Quantum Engine — Data Models
═══════════════════════════════════════════════════════════════════════════════
Extracted from l104_quantum_link_builder.py v5.0.0 → v6.0.0 package decomposition.
"""

import math
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .constants import (
    PHI, PHI_GROWTH, GOD_CODE, GOD_CODE_HZ, LINK_DRIFT_ENVELOPE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantumLink:
    """A quantum link between two implementations across the codebase.

    ★ v4.0 Quantum Min/Max Dynamism:
      - dynamic_value: composite φ-derived link value that evolves
      - min_bound / max_bound: auto-adjusting quantum boundaries
      - drift_velocity: rate of value change per cycle
      - drift_direction: +1 or -1
      - quantum_phase: oscillation phase [0, 2π)
      - evolution_count: cycles evolved
    """
    source_file: str
    source_symbol: str
    source_line: int
    target_file: str
    target_symbol: str
    target_line: int
    link_type: str             # "entanglement", "tunneling", "teleportation",
                               # "grover_chain", "epr_pair", "mirror", "bridge",
                               # "fourier", "braiding", "spooky_action"
    fidelity: float = 1.0      # 0.0–1.0 quantum fidelity of link
    strength: float = 1.0      # φ-weighted link strength
    coherence_time: float = 0.0  # Estimated decoherence time (sec)
    entanglement_entropy: float = 0.0  # S = -Tr(ρ log ρ)
    bell_violation: float = 0.0   # CHSH value (>2 = quantum, >2.828 = max)
    noise_resilience: float = 0.0  # 0–1 decoherence resistance
    last_verified: str = ""
    test_status: str = "untested"  # untested, passed, stressed, failed, upgraded
    upgrade_applied: str = ""
    # ★ v4.0 Quantum Min/Max Dynamism fields
    dynamic_value: float = 0.0       # Combined φ-derived evolving value
    min_bound: float = 0.0           # Auto-adjusting lower boundary
    max_bound: float = 0.0           # Auto-adjusting upper boundary
    drift_velocity: float = 0.0      # Rate of change per cycle
    drift_direction: int = 1         # +1 expand, -1 compact
    quantum_phase: float = 0.0       # Oscillation phase [0, 2π)
    evolution_count: int = 0         # Cycles evolved
    resonance_score: float = 0.0     # God Code alignment

    def __post_init__(self):
        """Initialize dynamism from fidelity/strength if not already set."""
        if self.dynamic_value == 0.0 and (self.fidelity > 0 or self.strength > 0):
            self._initialize_dynamism()

    def _initialize_dynamism(self):
        """Compute initial dynamic value and bounds from link properties."""
        # Dynamic value = φ-weighted combination of fidelity + strength + entropy
        self.dynamic_value = (
            self.fidelity * PHI_GROWTH +
            self.strength * PHI +
            self.entanglement_entropy * 0.5 +
            self.bell_violation * 0.1
        )
        # Bounds
        env_amp = LINK_DRIFT_ENVELOPE["amplitude"]
        envelope = max(abs(self.dynamic_value) * env_amp, PHI)
        self.min_bound = self.dynamic_value - envelope
        self.max_bound = self.dynamic_value + envelope
        # Phase from link_id hash
        lid = f"{self.source_file}:{self.source_symbol}"
        seed = int(hashlib.sha256(lid.encode()).hexdigest()[:8], 16)
        self.quantum_phase = (seed % 10000) / 10000.0 * 2 * math.pi
        # Initial velocity
        self.drift_velocity = LINK_DRIFT_ENVELOPE["max_velocity"] * math.sin(self.quantum_phase)
        # Resonance
        if self.dynamic_value > 0:
            self.resonance_score = abs(math.cos(self.dynamic_value * math.pi / GOD_CODE))

    def evolve(self):
        """Evolve this link's dynamic value by one φ-harmonic cycle."""
        self.evolution_count += 1
        # Phase advance
        self.quantum_phase = (
            self.quantum_phase + LINK_DRIFT_ENVELOPE["frequency"] * 0.1
        ) % (2 * math.pi)
        # Velocity
        target_v = LINK_DRIFT_ENVELOPE["max_velocity"] * math.sin(self.quantum_phase)
        d = LINK_DRIFT_ENVELOPE["damping"]
        self.drift_velocity = self.drift_velocity * (1 - d) + target_v * d
        # Apply drift
        new_val = self.dynamic_value + self.drift_velocity * self.drift_direction
        # Bounce off bounds
        if new_val > self.max_bound:
            new_val = self.max_bound
            self.drift_direction = -1
        elif new_val < self.min_bound:
            new_val = self.min_bound
            self.drift_direction = 1
        self.dynamic_value = new_val
        # Also drift fidelity and strength within safe ranges
        f_drift = LINK_DRIFT_ENVELOPE["fidelity_drift_scale"] * math.sin(self.quantum_phase)
        s_drift = LINK_DRIFT_ENVELOPE["strength_drift_scale"] * math.cos(self.quantum_phase)
        self.fidelity = max(0.0, min(1.0, self.fidelity + f_drift))
        self.strength = max(0.0, self.strength + s_drift)
        # Update resonance
        if abs(self.dynamic_value) > 1e-10:
            self.resonance_score = abs(math.cos(self.dynamic_value * math.pi / GOD_CODE))
        # Adaptive bounds for high-performing links
        if self.test_status in ("passed", "upgraded") and self.evolution_count > 3:
            self.max_bound += PHI * 0.0003
            self.min_bound -= PHI * 0.0002

    @property
    def link_id(self) -> str:
        """Generate unique link identifier from source and target."""
        return f"{self.source_file}:{self.source_symbol}↔{self.target_file}:{self.target_symbol}"

    def to_dict(self) -> dict:
        """Convert quantum link to dictionary representation."""
        d = asdict(self)
        d["link_id"] = self.link_id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "QuantumLink":
        """Reconstruct a QuantumLink from a dictionary."""
        d.pop("link_id", None)
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass

class StressTestResult:
    """Result of a quantum stress test on a link."""
    link_id: str
    test_type: str           # "grover_flood", "decoherence_attack", "noise_injection",
                             # "tunnel_barrier", "bell_violation", "entanglement_swap"
    iterations: int = 0
    passed: bool = False
    fidelity_before: float = 0.0
    fidelity_after: float = 0.0
    degradation_rate: float = 0.0
    recovery_time: float = 0.0
    details: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        """Convert stress test result to dictionary."""
        return asdict(self)


@dataclass

class CrossModalLink:
    """Cross-modal quantum link spanning Python↔Swift↔JS boundaries."""
    python_symbol: str
    swift_symbol: str
    modal_coherence: float = 0.0     # Cross-language coherence
    api_bridge_active: bool = False
    shared_constants: List[str] = field(default_factory=list)
    protocol_alignment: float = 0.0  # 0–1 protocol compatibility
    data_format_match: float = 0.0   # JSON/dict structure similarity

    def to_dict(self) -> dict:
        """Convert cross-modal link to dictionary."""
        return asdict(self)


@dataclass

class ChronoEntry:
    """Chronological record of a quantum link event."""
    timestamp: str
    event_type: str       # "created", "upgraded", "repaired", "degraded",
                          # "enlightened", "stress_tested", "cross_pollinated",
                          # "consciousness_shift", "stochastic_invented"
    link_id: str
    before_fidelity: float = 0.0
    after_fidelity: float = 0.0
    before_strength: float = 0.0
    after_strength: float = 0.0
    details: str = ""
    sacred_alignment: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM MIN/MAX DYNAMISM ENGINE — Link Subconscious Monitoring & Evolution
# ═══════════════════════════════════════════════════════════════════════════════

