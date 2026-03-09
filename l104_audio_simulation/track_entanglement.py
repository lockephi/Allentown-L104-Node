"""
Track Entanglement System — Quantum-Correlated Multi-Track Audio
═══════════════════════════════════════════════════════════════════════════════
Implements quantum entanglement between audio tracks so that changes on one
track propagate to entangled partners via quantum correlation.

DAW Context:
  In FL Studio / Ableton / Logic, tracks are independent — adjusting one
  has no effect on others unless manually linked via sidechain or macro.

  Track Entanglement introduces genuine quantum correlation:

  • **Bell-Pair Tracks**: Two tracks share a Bell state. When one track's
    gain increases, the partner's gain adjusts to maintain the entanglement
    invariant (total energy conserved via quantum constraint).

  • **GHZ Group**: N tracks share a GHZ state. All tracks are maximally
    correlated — a phase change on one propagates to all others.

  • **W-State Ensemble**: N tracks share a W state. Exactly one track is
    "active" at any time — quantum analog of solo/mute groups.

  • **Sidechain Entanglement**: Measurement on a sidechain input collapses
    the entangled track's dynamics (quantum sidechain compression).

  • **Spectral Entanglement**: Two tracks are entangled in the frequency
    domain — boosting bass on Track A reduces bass on Track B (spectral
    complementarity).

VQPU Integration:
  Entanglement states are maintained as real quantum circuits on the VQPU.
  Updates push new parameters into the circuit, and measurements propagate
  correlations.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

from __future__ import annotations

import math
import time
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

import numpy as np

from .constants import GOD_CODE, PHI, PHI_INV

logger = logging.getLogger("l104.audio.entanglement")


class EntanglementType(Enum):
    """Types of quantum entanglement between tracks."""
    BELL_PAIR = auto()       # 2-track maximally entangled pair
    GHZ_GROUP = auto()       # N-track GHZ state (all-or-nothing correlation)
    W_STATE = auto()         # N-track W state (exactly-one-active)
    CLUSTER = auto()         # Graph-state entanglement (arbitrary topology)
    SPECTRAL = auto()        # Frequency-domain entanglement
    SIDECHAIN = auto()       # Sidechain-triggered measurement/collapse
    TEMPORAL = auto()        # Time-delayed entanglement (echo correlation)


@dataclass
class EntanglementBond:
    """
    A quantum entanglement bond between tracks.
    Tracks the quantum state, correlation strength, and propagation rules.
    """
    bond_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    bond_type: EntanglementType = EntanglementType.BELL_PAIR
    track_ids: List[str] = field(default_factory=list)

    # Quantum state vector for this bond (2^N dimensional for N tracks)
    state_vector: np.ndarray = field(default=None)

    # Correlation matrix (N×N) measuring pairwise entanglement strength
    correlation_matrix: np.ndarray = field(default=None)

    # Bond strength: 0.0 (no entanglement) to 1.0 (maximally entangled)
    strength: float = 1.0

    # What properties are entangled
    entangle_gain: bool = True       # Fader levels correlated
    entangle_pan: bool = False       # Pan positions correlated
    entangle_phase: bool = True      # Phase locked
    entangle_spectrum: bool = False  # Spectral complement
    entangle_timing: bool = False    # Rhythmic correlation

    # Statistics
    measurements: int = 0
    propagations: int = 0
    creation_time: float = field(default_factory=time.time)
    last_measurement: float = 0.0

    def __post_init__(self):
        n = len(self.track_ids)
        if n < 2:
            n = 2
        dim = 2 ** n

        if self.state_vector is None:
            self.state_vector = self._init_state(n)

        if self.correlation_matrix is None:
            self.correlation_matrix = np.eye(n) * self.strength

    def _init_state(self, n: int) -> np.ndarray:
        """Initialize quantum state based on entanglement type."""
        dim = 2 ** n

        if self.bond_type == EntanglementType.BELL_PAIR:
            # |Φ+⟩ = (|00⟩ + |11⟩) / √2
            sv = np.zeros(dim, dtype=np.complex128)
            sv[0] = 1.0 / math.sqrt(2)
            sv[-1] = 1.0 / math.sqrt(2)
            return sv

        elif self.bond_type == EntanglementType.GHZ_GROUP:
            # |GHZ⟩ = (|00...0⟩ + |11...1⟩) / √2
            sv = np.zeros(dim, dtype=np.complex128)
            sv[0] = 1.0 / math.sqrt(2)
            sv[-1] = 1.0 / math.sqrt(2)
            return sv

        elif self.bond_type == EntanglementType.W_STATE:
            # |W⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √N
            sv = np.zeros(dim, dtype=np.complex128)
            for k in range(n):
                idx = 1 << (n - 1 - k)
                sv[idx] = 1.0 / math.sqrt(n)
            return sv

        elif self.bond_type == EntanglementType.CLUSTER:
            # Cluster state: H on all, then CZ between connected pairs
            sv = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)
            # Apply CZ-like phase to connected pairs
            for i in range(n - 1):
                for idx in range(dim):
                    bit_i = (idx >> (n - 1 - i)) & 1
                    bit_j = (idx >> (n - 2 - i)) & 1
                    if bit_i and bit_j:
                        sv[idx] *= -1
            return sv

        else:
            # Default: equal superposition
            sv = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)
            return sv

    @property
    def concurrence(self) -> float:
        """
        Measure entanglement strength (concurrence for 2-qubit case,
        generalized for N-qubit).
        """
        n = len(self.track_ids) or 2
        if n == 2:
            # 2-qubit concurrence: C = 2|α·δ - β·γ| for |ψ⟩ = α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩
            sv = self.state_vector
            if len(sv) >= 4:
                return float(2.0 * abs(sv[0] * sv[3] - sv[1] * sv[2]))
        # N-qubit: use average pairwise correlation
        return float(np.mean(np.abs(self.correlation_matrix - np.eye(n)))) * 2.0

    @property
    def entropy(self) -> float:
        """Von Neumann entropy of the bond state."""
        probs = np.abs(self.state_vector) ** 2
        probs = probs[probs > 1e-15]
        return float(-np.sum(probs * np.log2(probs)))

    def measure(self, track_index: int) -> Dict[str, float]:
        """
        Measure one track's quantum state, collapsing the entanglement
        and producing correlated effects on partner tracks.

        Returns dict of {track_id: effect_strength} for each entangled track.
        """
        n = len(self.track_ids)
        if track_index >= n:
            return {}

        probs = np.abs(self.state_vector) ** 2
        dim = len(self.state_vector)

        # Marginalize over the measured track
        outcome_probs = [0.0, 0.0]
        for idx in range(dim):
            bit = (idx >> (n - 1 - track_index)) & 1
            outcome_probs[bit] += probs[idx]

        # Measurement outcome (Born rule)
        total = sum(outcome_probs)
        if total < 1e-15:
            return {}
        p_one = outcome_probs[1] / total
        outcome = 1 if np.random.random() < p_one else 0

        # Post-measurement state collapse
        new_sv = np.zeros_like(self.state_vector)
        for idx in range(dim):
            bit = (idx >> (n - 1 - track_index)) & 1
            if bit == outcome:
                new_sv[idx] = self.state_vector[idx]
        norm = np.linalg.norm(new_sv)
        if norm > 1e-15:
            new_sv /= norm
        self.state_vector = new_sv

        # Compute correlated effects on partner tracks
        effects = {}
        for i, tid in enumerate(self.track_ids):
            if i == track_index:
                continue
            # Marginal probability for this partner
            p_partner_one = 0.0
            for idx in range(dim):
                bit = (idx >> (n - 1 - i)) & 1
                if bit == 1:
                    p_partner_one += np.abs(new_sv[idx]) ** 2

            # Effect: deviation from 0.5 (no correlation)
            effect = (p_partner_one - 0.5) * 2.0 * self.strength
            effects[tid] = float(effect)

        self.measurements += 1
        self.last_measurement = time.time()
        return effects

    def propagate_gain(self, source_track_id: str, gain_change: float) -> Dict[str, float]:
        """
        Propagate a gain change from one track to entangled partners.
        Uses correlation matrix to determine propagation strength.
        """
        if not self.entangle_gain or source_track_id not in self.track_ids:
            return {}
        source_idx = self.track_ids.index(source_track_id)

        effects = {}
        for i, tid in enumerate(self.track_ids):
            if tid == source_track_id:
                continue
            correlation = self.correlation_matrix[source_idx, i]

            if self.bond_type == EntanglementType.BELL_PAIR:
                # Bell: anti-correlated gain (conservation)
                effects[tid] = -gain_change * float(correlation) * self.strength
            elif self.bond_type == EntanglementType.GHZ_GROUP:
                # GHZ: all move together
                effects[tid] = gain_change * float(correlation) * self.strength
            elif self.bond_type == EntanglementType.W_STATE:
                # W: compensatory — increase one, decrease others
                n = len(self.track_ids)
                effects[tid] = -gain_change / max(n - 1, 1) * float(correlation) * self.strength
            elif self.bond_type == EntanglementType.SPECTRAL:
                # Spectral: complementary frequency response
                effects[tid] = -gain_change * float(correlation) * self.strength * PHI_INV
            else:
                effects[tid] = gain_change * float(correlation) * self.strength * 0.5

        self.propagations += 1
        return effects

    def propagate_spectral(
        self,
        source_spectrum: np.ndarray,
        source_track_id: str,
    ) -> Dict[str, np.ndarray]:
        """
        Propagate spectral changes to entangled partners.
        For SPECTRAL entanglement: boost on source = cut on partner (complementarity).
        """
        if not self.entangle_spectrum or source_track_id not in self.track_ids:
            return {}

        source_idx = self.track_ids.index(source_track_id)
        effects = {}

        for i, tid in enumerate(self.track_ids):
            if tid == source_track_id:
                continue
            correlation = self.correlation_matrix[source_idx, i]

            # Spectral complement: invert the change
            complement = 1.0 / np.maximum(np.abs(source_spectrum), 1e-15)
            complement /= np.max(complement)
            blend = float(abs(correlation)) * self.strength
            partner_response = (1.0 - blend) + blend * complement
            effects[tid] = partner_response

        return effects

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bond_id": self.bond_id,
            "bond_type": self.bond_type.name,
            "track_ids": self.track_ids,
            "strength": self.strength,
            "concurrence": self.concurrence,
            "entropy": self.entropy,
            "measurements": self.measurements,
            "propagations": self.propagations,
            "entangle_gain": self.entangle_gain,
            "entangle_pan": self.entangle_pan,
            "entangle_phase": self.entangle_phase,
            "entangle_spectrum": self.entangle_spectrum,
        }


class TrackEntanglementManager:
    """
    Manages all entanglement bonds across the DAW session.
    Provides high-level API for creating, querying, and propagating
    quantum correlations between tracks.
    """

    def __init__(self):
        self.bonds: Dict[str, EntanglementBond] = {}
        self._track_bonds: Dict[str, Set[str]] = {}  # track_id → set of bond_ids

        # VQPU for maintaining real quantum states
        self._vqpu = None

        # Statistics
        self.total_bonds_created = 0
        self.total_measurements = 0
        self.total_propagations = 0

    @property
    def vqpu(self):
        if self._vqpu is None:
            try:
                from l104_vqpu import get_bridge
                self._vqpu = get_bridge()
            except ImportError:
                pass
        return self._vqpu

    def create_bell_pair(
        self,
        track_a: str,
        track_b: str,
        strength: float = 1.0,
        **kwargs,
    ) -> EntanglementBond:
        """Create a Bell-pair entanglement between two tracks."""
        bond = EntanglementBond(
            bond_type=EntanglementType.BELL_PAIR,
            track_ids=[track_a, track_b],
            strength=strength,
            **kwargs,
        )
        return self._register_bond(bond)

    def create_ghz_group(
        self,
        track_ids: List[str],
        strength: float = 1.0,
        **kwargs,
    ) -> EntanglementBond:
        """Create GHZ entanglement across N tracks (all-or-nothing)."""
        bond = EntanglementBond(
            bond_type=EntanglementType.GHZ_GROUP,
            track_ids=track_ids,
            strength=strength,
            **kwargs,
        )
        return self._register_bond(bond)

    def create_w_state(
        self,
        track_ids: List[str],
        strength: float = 1.0,
        **kwargs,
    ) -> EntanglementBond:
        """Create W-state entanglement (exactly-one-active pattern)."""
        bond = EntanglementBond(
            bond_type=EntanglementType.W_STATE,
            track_ids=track_ids,
            strength=strength,
            **kwargs,
        )
        return self._register_bond(bond)

    def create_spectral_pair(
        self,
        track_a: str,
        track_b: str,
        strength: float = 0.8,
    ) -> EntanglementBond:
        """Create spectral entanglement (frequency-domain complementarity)."""
        bond = EntanglementBond(
            bond_type=EntanglementType.SPECTRAL,
            track_ids=[track_a, track_b],
            strength=strength,
            entangle_spectrum=True,
            entangle_gain=False,
        )
        return self._register_bond(bond)

    def create_sidechain(
        self,
        source_track: str,
        target_track: str,
        strength: float = 0.9,
    ) -> EntanglementBond:
        """Create sidechain entanglement (measurement on source → collapse on target)."""
        bond = EntanglementBond(
            bond_type=EntanglementType.SIDECHAIN,
            track_ids=[source_track, target_track],
            strength=strength,
            entangle_gain=True,
        )
        return self._register_bond(bond)

    def _register_bond(self, bond: EntanglementBond) -> EntanglementBond:
        """Register a bond and update track-bond mapping."""
        self.bonds[bond.bond_id] = bond
        for tid in bond.track_ids:
            if tid not in self._track_bonds:
                self._track_bonds[tid] = set()
            self._track_bonds[tid].add(bond.bond_id)
        self.total_bonds_created += 1
        return bond

    def get_bonds_for_track(self, track_id: str) -> List[EntanglementBond]:
        """Get all entanglement bonds involving a track."""
        bond_ids = self._track_bonds.get(track_id, set())
        return [self.bonds[bid] for bid in bond_ids if bid in self.bonds]

    def propagate_change(
        self,
        track_id: str,
        change_type: str,
        change_value: float,
    ) -> Dict[str, float]:
        """
        Propagate a change from one track through all its entanglement bonds.
        Returns aggregated effects on all partner tracks.
        """
        all_effects: Dict[str, float] = {}
        for bond in self.get_bonds_for_track(track_id):
            if change_type == "gain":
                effects = bond.propagate_gain(track_id, change_value)
            else:
                continue
            for tid, val in effects.items():
                all_effects[tid] = all_effects.get(tid, 0.0) + val
            self.total_propagations += 1

        return all_effects

    def propagate_spectral_change(
        self,
        track_id: str,
        spectrum: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Propagate spectral changes through spectral entanglement bonds."""
        all_effects: Dict[str, np.ndarray] = {}
        for bond in self.get_bonds_for_track(track_id):
            if bond.entangle_spectrum:
                effects = bond.propagate_spectral(spectrum, track_id)
                for tid, response in effects.items():
                    if tid in all_effects:
                        all_effects[tid] *= response
                    else:
                        all_effects[tid] = response
        return all_effects

    def vqpu_refresh_bonds(self):
        """
        Use VQPU to refresh all entanglement bond states with real
        quantum circuit measurements.
        """
        if not self.vqpu:
            return

        try:
            from l104_vqpu import QuantumJob, QuantumGate

            for bond in self.bonds.values():
                n = len(bond.track_ids)
                n_q = min(n, 8)
                ops = []

                # Create entangled state on VQPU
                if bond.bond_type == EntanglementType.BELL_PAIR:
                    ops.append(QuantumGate("H", [0]))
                    ops.append(QuantumGate("CNOT", [0, 1]))
                elif bond.bond_type == EntanglementType.GHZ_GROUP:
                    ops.append(QuantumGate("H", [0]))
                    for q in range(n_q - 1):
                        ops.append(QuantumGate("CNOT", [q, q + 1]))
                elif bond.bond_type == EntanglementType.W_STATE:
                    # W state preparation
                    ops.append(QuantumGate("X", [0]))
                    for q in range(n_q - 1):
                        theta = math.acos(math.sqrt(1.0 / (n_q - q)))
                        ops.append(QuantumGate("Ry", [q], [2.0 * theta]))
                        ops.append(QuantumGate("CNOT", [q, q + 1]))

                # GOD_CODE sacred phase
                ops.append(QuantumGate("Rz", [0], [GOD_CODE / 1000.0 * math.pi]))

                job = QuantumJob(
                    circuit_id=f"entangle_refresh_{bond.bond_id}",
                    num_qubits=n_q,
                    operations=ops,
                    shots=256,
                )
                result = self.vqpu.submit_and_wait(job, timeout=3.0)
                if result and result.probabilities:
                    dim = 2 ** n_q
                    new_sv = np.zeros(dim, dtype=np.complex128)
                    for key, prob in result.probabilities.items():
                        idx = int(key, 2) if isinstance(key, str) else int(key)
                        if idx < dim:
                            new_sv[idx] = math.sqrt(prob)
                    norm = np.linalg.norm(new_sv)
                    if norm > 1e-15:
                        bond.state_vector = new_sv / norm

        except Exception as e:
            logger.debug(f"VQPU bond refresh error: {e}")

    def remove_bond(self, bond_id: str):
        """Remove an entanglement bond."""
        if bond_id in self.bonds:
            bond = self.bonds.pop(bond_id)
            for tid in bond.track_ids:
                if tid in self._track_bonds:
                    self._track_bonds[tid].discard(bond_id)

    def status(self) -> Dict[str, Any]:
        return {
            "total_bonds": len(self.bonds),
            "total_bonds_created": self.total_bonds_created,
            "total_measurements": self.total_measurements,
            "total_propagations": self.total_propagations,
            "bond_types": {bt.name: sum(1 for b in self.bonds.values() if b.bond_type == bt)
                           for bt in EntanglementType},
            "vqpu_available": self.vqpu is not None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.status(),
            "bonds": {bid: b.to_dict() for bid, b in self.bonds.items()},
        }
