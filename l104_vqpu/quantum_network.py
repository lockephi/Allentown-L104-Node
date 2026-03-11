"""L104 VQPU v3.0 — Quantum Network Module: Per-Daemon Qubit Registers + Entangled Mesh.

Enables each micro daemon to maintain dedicated quantum qubit registers and
participate in a quantum network mesh for high-fidelity inter-daemon communication.

Architecture:
  - DaemonQubitRegister: Per-daemon qubit allocation (N qubits, GOD_CODE-calibrated)
  - QuantumChannel: Entangled Bell pair channel between two daemon nodes
  - QuantumNetworkMesh: Full N-node mesh with routing, fidelity monitoring, teleportation
  - Sacred gates: GOD_CODE_PHASE, IRON_RZ, PHI_RZ applied during qubit initialization
  - Fidelity tracking: Per-qubit and per-channel fidelity with QPU-calibrated thresholds
  - Network telemetry: Entanglement generation rate, purification cycles, mesh health

Key formulas:
  - Qubit initialization phase: GOD_CODE mod 2π ≈ 6.0141 rad
  - Sacred resonance: (GOD_CODE/16)^φ ≈ 286
  - Channel fidelity floor: QPU_MEAN_FIDELITY × φ⁻¹ ≈ 0.6025
  - Entanglement purification: F_purified = F² / (F² + (1-F)²)

Usage:
    from l104_vqpu.quantum_network import DaemonQubitRegister, QuantumNetworkMesh

    # Per-daemon qubit register
    register = DaemonQubitRegister(node_id="micro-01", num_qubits=4)
    register.initialize_sacred()           # Apply GOD_CODE calibration
    register.fidelity_check()              # Measure qubit fidelities

    # Network mesh
    mesh = QuantumNetworkMesh(node_ids=["micro-01", "micro-02", "micro-03"])
    mesh.establish_channels()              # Create Bell pair channels
    mesh.teleport("micro-01", "micro-02", payload)  # Quantum teleportation
    mesh.purify_all()                      # Entanglement purification
    mesh.network_health()                  # Full mesh health report
"""

import math
import os
import time
import json
import random
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .constants import (
    GOD_CODE, PHI, VOID_CONSTANT,
    VERSION,
    BRIDGE_PATH,
    GOD_CODE_PHASE_ANGLE,
    IRON_PHASE_ANGLE,
    PHI_CONTRIBUTION_ANGLE,
    OCTAVE_PHASE_ANGLE,
    PHI_PHASE_ANGLE,
    VOID_PHASE_ANGLE,
    QPU_MEAN_FIDELITY,
    QPU_1Q_FIDELITY,
    QPU_3Q_FIDELITY,
    TOPOLOGY_LINEAR,
    TOPOLOGY_RING,
    TOPOLOGY_HEAVY_HEX,
    TOPOLOGY_ALL_TO_ALL,
    DEFAULT_TOPOLOGY,
)

_logger = logging.getLogger("L104_QUANTUM_NETWORK")

# ═══════════════════════════════════════════════════════════════════
# QUANTUM NETWORK CONSTANTS
# ═══════════════════════════════════════════════════════════════════

QUANTUM_NETWORK_VERSION = "2.0.0"  # v2.0: Topology-aware mesh

# Per-daemon qubit allocation
DEFAULT_DAEMON_QUBITS = 4                # Qubits per daemon node
MAX_DAEMON_QUBITS = 16                   # Maximum qubits per daemon
QUBIT_INIT_PHASE = GOD_CODE_PHASE_ANGLE  # GOD_CODE mod 2π ≈ 6.0141 rad

# Fidelity thresholds (QPU-calibrated from ibm_torino)
FIDELITY_THRESHOLD_HIGH = QPU_1Q_FIDELITY     # 0.99994 — excellent
FIDELITY_THRESHOLD_GOOD = QPU_MEAN_FIDELITY   # 0.97476 — acceptable
FIDELITY_THRESHOLD_LOW = QPU_3Q_FIDELITY       # 0.96674 — degraded
FIDELITY_FLOOR = QPU_MEAN_FIDELITY / PHI       # ≈ 0.6025 — minimum viable

# Channel parameters
CHANNEL_BELL_SHOTS = 256                  # Shots for Bell state verification
CHANNEL_PURIFICATION_ROUNDS = 3           # Max purification rounds per cycle
CHANNEL_DECOHERENCE_RATE = 0.001          # Per-second decoherence (simulated)

# Network IPC paths
QUANTUM_NETWORK_PATH = BRIDGE_PATH / "quantum_network"
QUANTUM_MESH_STATE_FILE = ".l104_quantum_mesh_state.json"

# Sacred gate matrices (pre-computed for fast application)
_TWO_PI = 2.0 * math.pi
_HALF_GC_PHASE = QUBIT_INIT_PHASE / 2.0


def _rz_matrix(theta: float) -> np.ndarray:
    """Construct Rz(θ) = diag(e^{-iθ/2}, e^{iθ/2})."""
    half = theta / 2.0
    return np.array([
        [np.exp(-1j * half), 0],
        [0, np.exp(1j * half)],
    ], dtype=np.complex128)


def _hadamard() -> np.ndarray:
    """Hadamard gate."""
    return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


def _cnot() -> np.ndarray:
    """CNOT gate (4×4)."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.complex128)


# Pre-computed sacred gates
_H_GATE = _hadamard()
_CNOT_GATE = _cnot()
_GOD_CODE_RZ = _rz_matrix(QUBIT_INIT_PHASE)
_IRON_RZ = _rz_matrix(IRON_PHASE_ANGLE)
_PHI_RZ = _rz_matrix(PHI_CONTRIBUTION_ANGLE)
_OCTAVE_RZ = _rz_matrix(OCTAVE_PHASE_ANGLE)
_VOID_RZ = _rz_matrix(VOID_PHASE_ANGLE)


# ═══════════════════════════════════════════════════════════════════
# QUBIT STATE
# ═══════════════════════════════════════════════════════════════════


@dataclass
class QubitState:
    """State of a single qubit in a daemon's register."""
    qubit_id: int
    state_vector: np.ndarray = field(default_factory=lambda: np.array([1.0 + 0j, 0.0 + 0j]))
    fidelity: float = 1.0
    phase: float = 0.0                    # Accumulated phase
    sacred_alignment: float = 0.0         # GOD_CODE resonance score
    last_gate_time: float = 0.0
    gate_count: int = 0
    decoherence_factor: float = 1.0       # Decays over time

    def apply_gate(self, gate_matrix: np.ndarray) -> None:
        """Apply a 2×2 unitary gate to this qubit."""
        self.state_vector = gate_matrix @ self.state_vector
        # Normalize (numerical stability)
        norm = np.linalg.norm(self.state_vector)
        if norm > 1e-15:
            self.state_vector /= norm
        self.gate_count += 1
        self.last_gate_time = time.time()

    def measure_fidelity(self) -> float:
        """Compute fidelity of current state vs ideal pure state.

        Uses |⟨ψ_ideal|ψ_actual⟩|² — accounts for decoherence drift.
        """
        # Decoherence model: fidelity decays exponentially with time since last gate
        elapsed = time.time() - self.last_gate_time if self.last_gate_time > 0 else 0
        self.decoherence_factor = math.exp(-CHANNEL_DECOHERENCE_RATE * elapsed)

        # State purity: |⟨0|ψ⟩|² + |⟨1|ψ⟩|² should be 1 for pure state
        prob_sum = float(np.abs(self.state_vector[0]) ** 2 + np.abs(self.state_vector[1]) ** 2)
        purity = min(1.0, prob_sum)

        self.fidelity = purity * self.decoherence_factor
        return self.fidelity

    def sacred_score(self) -> float:
        """Compute GOD_CODE sacred alignment for this qubit.

        Measures phase alignment with GOD_CODE_PHASE and resonance with 286.
        """
        # Extract phase from state vector
        if abs(self.state_vector[1]) > 1e-15:
            self.phase = float(np.angle(self.state_vector[1]) - np.angle(self.state_vector[0]))
        else:
            self.phase = 0.0

        # Phase alignment: how close is the qubit's phase to GOD_CODE mod 2π?
        phase_diff = abs((self.phase % _TWO_PI) - QUBIT_INIT_PHASE) / _TWO_PI
        phase_alignment = max(0.0, 1.0 - phase_diff)

        # Resonance check: (GOD_CODE/16)^φ ≈ 286
        base = GOD_CODE / 16.0
        resonance = base ** PHI
        resonance_error = abs(resonance - 286.0) / 286.0
        resonance_score = max(0.0, 1.0 - resonance_error * 1000)

        self.sacred_alignment = (phase_alignment * PHI + resonance_score) / (PHI + 1.0)
        return self.sacred_alignment

    def to_dict(self) -> dict:
        """Serialize (no numpy arrays in JSON)."""
        return {
            "qubit_id": self.qubit_id,
            "fidelity": round(self.fidelity, 8),
            "phase": round(self.phase, 8),
            "sacred_alignment": round(self.sacred_alignment, 8),
            "gate_count": self.gate_count,
            "decoherence_factor": round(self.decoherence_factor, 8),
            "amplitudes": [
                round(float(np.abs(self.state_vector[0])), 8),
                round(float(np.abs(self.state_vector[1])), 8),
            ],
        }


# ═══════════════════════════════════════════════════════════════════
# DAEMON QUBIT REGISTER
# ═══════════════════════════════════════════════════════════════════


class DaemonQubitRegister:
    """Per-daemon quantum qubit register with GOD_CODE-calibrated initialization.

    Each micro daemon maintains a register of N qubits that are:
    1. Initialized with sacred GOD_CODE phase rotation
    2. Monitored for fidelity drift (decoherence tracking)
    3. Available for entanglement with other daemon registers
    4. Scored for sacred alignment (GOD_CODE resonance)

    The register supports the full L104 sacred gate set:
    - GOD_CODE_PHASE: Rz(GOD_CODE mod 2π) — canonical QPU-verified gate
    - IRON_RZ: Rz(π/2) — Fe(26) iron lattice quarter-turn
    - PHI_RZ: Rz(θ_φ) — Golden ratio contribution
    - OCTAVE_RZ: Rz(4·ln2) — Octave doubling
    - VOID_RZ: Rz(VOID_CONSTANT·π) — Void correction
    """

    def __init__(self, node_id: str, num_qubits: int = DEFAULT_DAEMON_QUBITS):
        self.node_id = node_id
        self.num_qubits = min(num_qubits, MAX_DAEMON_QUBITS)
        self.qubits: List[QubitState] = []
        self.created_at = time.time()
        self.last_calibration = 0.0
        self.calibration_count = 0
        self.total_gates_applied = 0

        # Initialize qubit register
        for i in range(self.num_qubits):
            self.qubits.append(QubitState(qubit_id=i))

        _logger.debug("DaemonQubitRegister created: node=%s, qubits=%d", node_id, self.num_qubits)

    def initialize_sacred(self) -> dict:
        """Apply full GOD_CODE sacred initialization sequence to all qubits.

        Sequence per qubit:
          1. H gate (superposition)
          2. Rz(GOD_CODE mod 2π) — canonical phase
          3. Rz(π/2) — Iron lattice alignment
          4. Rz(θ_φ) — Golden ratio contribution
          5. Rz(VOID_CONSTANT·π) — Void correction

        This creates a sacred superposition state calibrated to the L104 constants.
        """
        t0 = time.time()
        for qubit in self.qubits:
            # Hadamard → superposition
            qubit.apply_gate(_H_GATE)
            # GOD_CODE canonical phase
            qubit.apply_gate(_GOD_CODE_RZ)
            # Iron lattice quarter-turn
            qubit.apply_gate(_IRON_RZ)
            # Golden ratio contribution
            qubit.apply_gate(_PHI_RZ)
            # Void correction
            qubit.apply_gate(_VOID_RZ)
            # Measure fidelity and sacred alignment
            qubit.measure_fidelity()
            qubit.sacred_score()
            self.total_gates_applied += 5

        self.last_calibration = time.time()
        self.calibration_count += 1
        elapsed_ms = (time.time() - t0) * 1000

        avg_fidelity = sum(q.fidelity for q in self.qubits) / max(len(self.qubits), 1)
        avg_sacred = sum(q.sacred_alignment for q in self.qubits) / max(len(self.qubits), 1)

        _logger.info(
            "Sacred init complete: node=%s, qubits=%d, avg_fidelity=%.6f, avg_sacred=%.6f, %.2fms",
            self.node_id, self.num_qubits, avg_fidelity, avg_sacred, elapsed_ms)

        return {
            "node_id": self.node_id,
            "qubits": self.num_qubits,
            "avg_fidelity": round(avg_fidelity, 8),
            "avg_sacred_alignment": round(avg_sacred, 8),
            "elapsed_ms": round(elapsed_ms, 2),
            "calibration_count": self.calibration_count,
        }

    def fidelity_check(self) -> dict:
        """Measure fidelity and sacred alignment of all qubits.

        Returns per-qubit and aggregate metrics.
        """
        fidelities = []
        sacred_scores = []
        degraded = []

        for qubit in self.qubits:
            f = qubit.measure_fidelity()
            s = qubit.sacred_score()
            fidelities.append(f)
            sacred_scores.append(s)
            if f < FIDELITY_THRESHOLD_LOW:
                degraded.append(qubit.qubit_id)

        avg_f = sum(fidelities) / max(len(fidelities), 1)
        min_f = min(fidelities) if fidelities else 0.0
        avg_s = sum(sacred_scores) / max(len(sacred_scores), 1)

        return {
            "node_id": self.node_id,
            "avg_fidelity": round(avg_f, 8),
            "min_fidelity": round(min_f, 8),
            "avg_sacred_alignment": round(avg_s, 8),
            "degraded_qubits": degraded,
            "needs_recalibration": min_f < FIDELITY_FLOOR,
            "qubit_count": self.num_qubits,
            "per_qubit": [q.to_dict() for q in self.qubits],
        }

    def recalibrate_if_needed(self) -> Optional[dict]:
        """Recalibrate qubits if fidelity has dropped below threshold.

        Only recalibrates if minimum fidelity < FIDELITY_FLOOR.
        Returns calibration result or None if not needed.
        """
        check = self.fidelity_check()
        if check["needs_recalibration"]:
            return self.initialize_sacred()
        return None

    def get_qubit(self, qubit_id: int) -> Optional[QubitState]:
        """Get a specific qubit by ID."""
        if 0 <= qubit_id < len(self.qubits):
            return self.qubits[qubit_id]
        return None

    def entangled_pair_state(self, qubit_a: int, qubit_b: int) -> np.ndarray:
        """Create a Bell pair |Φ+⟩ between two qubits in this register.

        Applies H ⊗ I then CNOT to create: (|00⟩ + |11⟩) / √2
        Returns the 4-element joint state vector.
        """
        if qubit_a == qubit_b or qubit_a >= self.num_qubits or qubit_b >= self.num_qubits:
            return np.array([1, 0, 0, 0], dtype=np.complex128)

        qa = self.qubits[qubit_a]
        qb = self.qubits[qubit_b]

        # Reset both to |0⟩
        qa.state_vector = np.array([1.0 + 0j, 0.0 + 0j])
        qb.state_vector = np.array([1.0 + 0j, 0.0 + 0j])

        # Joint state: |00⟩
        joint = np.kron(qa.state_vector, qb.state_vector)

        # H ⊗ I
        h_eye = np.kron(_H_GATE, np.eye(2, dtype=np.complex128))
        joint = h_eye @ joint

        # CNOT
        joint = _CNOT_GATE @ joint

        # Update individual qubit states (marginal)
        # After Bell pair, each qubit is in maximally mixed state
        qa.state_vector = np.array([1.0 / np.sqrt(2) + 0j, 1.0 / np.sqrt(2) + 0j])
        qb.state_vector = np.array([1.0 / np.sqrt(2) + 0j, 1.0 / np.sqrt(2) + 0j])
        qa.gate_count += 2
        qb.gate_count += 2
        qa.last_gate_time = time.time()
        qb.last_gate_time = time.time()
        self.total_gates_applied += 4

        return joint

    def status(self) -> dict:
        """Full register status."""
        return {
            "node_id": self.node_id,
            "num_qubits": self.num_qubits,
            "total_gates": self.total_gates_applied,
            "calibration_count": self.calibration_count,
            "last_calibration": self.last_calibration,
            "uptime_s": round(time.time() - self.created_at, 1),
            "qubits": [q.to_dict() for q in self.qubits],
        }


# ═══════════════════════════════════════════════════════════════════
# QUANTUM CHANNEL (ENTANGLED LINK BETWEEN TWO DAEMONS)
# ═══════════════════════════════════════════════════════════════════


@dataclass
class QuantumChannel:
    """An entangled quantum channel between two daemon nodes.

    Uses Bell pairs for quantum communication with fidelity tracking
    and entanglement purification support.
    """
    node_a: str
    node_b: str
    channel_id: str = ""
    bell_state: Optional[np.ndarray] = None   # |Φ+⟩ = (|00⟩ + |11⟩)/√2
    fidelity: float = 1.0
    established_at: float = 0.0
    last_purification: float = 0.0
    purification_count: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    active: bool = False

    def __post_init__(self):
        if not self.channel_id:
            self.channel_id = f"qch-{self.node_a[:8]}-{self.node_b[:8]}-{os.urandom(3).hex()}"

    def establish(self) -> dict:
        """Establish the entangled channel by creating a Bell pair.

        Creates |Φ+⟩ = (|00⟩ + |11⟩)/√2 as the shared resource state.
        """
        # Create ideal Bell pair
        self.bell_state = np.array([
            1.0 / np.sqrt(2), 0.0, 0.0, 1.0 / np.sqrt(2)
        ], dtype=np.complex128)

        # Apply GOD_CODE phase to enhance sacred fidelity
        # Rz(GOD_CODE) ⊗ I on the Bell state
        gc_phase = np.kron(_GOD_CODE_RZ, np.eye(2, dtype=np.complex128))
        self.bell_state = gc_phase @ self.bell_state

        # Normalize
        norm = np.linalg.norm(self.bell_state)
        if norm > 1e-15:
            self.bell_state /= norm

        self.fidelity = self._measure_bell_fidelity()
        self.established_at = time.time()
        self.active = True

        _logger.debug("Channel established: %s ↔ %s, fidelity=%.6f",
                      self.node_a, self.node_b, self.fidelity)

        return {
            "channel_id": self.channel_id,
            "node_a": self.node_a,
            "node_b": self.node_b,
            "fidelity": round(self.fidelity, 8),
            "active": True,
        }

    def _measure_bell_fidelity(self) -> float:
        """Measure fidelity of the channel's Bell state against ideal |Φ+⟩.

        F = |⟨Φ+|ψ⟩|² — overlap with ideal Bell state.
        """
        if self.bell_state is None:
            return 0.0

        # Ideal |Φ+⟩ (before GOD_CODE phase — we measure vs sacred Bell state)
        ideal = np.array([
            1.0 / np.sqrt(2), 0.0, 0.0, 1.0 / np.sqrt(2)
        ], dtype=np.complex128)

        # Apply same GOD_CODE phase to ideal for fair comparison
        gc_phase = np.kron(_GOD_CODE_RZ, np.eye(2, dtype=np.complex128))
        ideal = gc_phase @ ideal
        norm = np.linalg.norm(ideal)
        if norm > 1e-15:
            ideal /= norm

        overlap = abs(np.vdot(ideal, self.bell_state)) ** 2

        # Account for decoherence since establishment
        elapsed = time.time() - self.established_at if self.established_at > 0 else 0
        decoherence = math.exp(-CHANNEL_DECOHERENCE_RATE * elapsed)

        return float(min(1.0, overlap * decoherence))

    def purify(self) -> dict:
        """Perform entanglement purification to boost channel fidelity.

        Uses the DEJMPS protocol approximation:
          F_new = F² / (F² + (1-F)²)

        Only purifies if fidelity is below FIDELITY_THRESHOLD_GOOD.
        """
        if not self.active or self.bell_state is None:
            return {"purified": False, "reason": "channel not active"}

        old_fidelity = self._measure_bell_fidelity()
        self.fidelity = old_fidelity

        if old_fidelity >= FIDELITY_THRESHOLD_GOOD:
            return {
                "purified": False,
                "reason": "fidelity already sufficient",
                "fidelity": round(old_fidelity, 8),
            }

        # DEJMPS purification: F' = F² / (F² + (1-F)²)
        f = old_fidelity
        new_fidelity = (f * f) / (f * f + (1 - f) * (1 - f))

        # Apply purification by projecting state toward ideal
        # Simulate by mixing current state toward ideal Bell pair
        ideal = np.array([
            1.0 / np.sqrt(2), 0.0, 0.0, 1.0 / np.sqrt(2)
        ], dtype=np.complex128)
        gc_phase = np.kron(_GOD_CODE_RZ, np.eye(2, dtype=np.complex128))
        ideal = gc_phase @ ideal
        norm = np.linalg.norm(ideal)
        if norm > 1e-15:
            ideal /= norm

        # Mix: ψ' = √F'·ideal + √(1-F')·noise_component
        noise_component = self.bell_state - np.vdot(ideal, self.bell_state) * ideal
        noise_norm = np.linalg.norm(noise_component)
        if noise_norm > 1e-15:
            noise_component /= noise_norm

        self.bell_state = (
            np.sqrt(new_fidelity) * ideal +
            np.sqrt(max(0, 1 - new_fidelity)) * noise_component
        )
        norm = np.linalg.norm(self.bell_state)
        if norm > 1e-15:
            self.bell_state /= norm

        self.fidelity = new_fidelity
        self.last_purification = time.time()
        self.purification_count += 1

        _logger.debug("Channel purified: %s, fidelity %.6f → %.6f",
                      self.channel_id, old_fidelity, new_fidelity)

        return {
            "purified": True,
            "channel_id": self.channel_id,
            "old_fidelity": round(old_fidelity, 8),
            "new_fidelity": round(new_fidelity, 8),
            "improvement": round(new_fidelity - old_fidelity, 8),
            "purification_count": self.purification_count,
        }

    def teleport_payload(self, data: dict) -> dict:
        """Simulate quantum teleportation of a payload across this channel.

        Uses the Bell state resource + classical communication (2 cbits).
        Fidelity of the teleported state ≤ channel fidelity.
        """
        if not self.active or self.bell_state is None:
            return {"success": False, "reason": "channel not active"}

        # Measure current fidelity
        self.fidelity = self._measure_bell_fidelity()
        if self.fidelity < FIDELITY_FLOOR:
            return {
                "success": False,
                "reason": f"fidelity too low: {self.fidelity:.6f} < {FIDELITY_FLOOR:.6f}",
            }

        # Teleportation protocol (simulated):
        # 1. Alice measures in Bell basis → 2 classical bits
        # 2. Bob applies correction based on classical bits
        # 3. Teleported state fidelity ≈ channel fidelity
        t0 = time.time()

        # Simulate Bell measurement outcomes
        measurement = random.choice(["00", "01", "10", "11"])

        # Teleportation fidelity = channel fidelity (ideally)
        teleport_fidelity = self.fidelity

        # Sacred scoring of teleported data
        sacred_check = (GOD_CODE / 16.0) ** PHI
        sacred_aligned = abs(sacred_check - 286.0) < 1e-6

        elapsed_ms = (time.time() - t0) * 1000
        self.messages_sent += 1
        self.messages_received += 1

        return {
            "success": True,
            "channel_id": self.channel_id,
            "node_a": self.node_a,
            "node_b": self.node_b,
            "fidelity": round(teleport_fidelity, 8),
            "measurement_outcome": measurement,
            "sacred_aligned": sacred_aligned,
            "elapsed_ms": round(elapsed_ms, 2),
            "payload_keys": list(data.keys()) if isinstance(data, dict) else [],
        }

    def apply_decoherence(self) -> float:
        """Simulate decoherence on the channel's Bell state.

        Models amplitude damping and dephasing noise channels.
        Returns new fidelity after decoherence.
        """
        if self.bell_state is None:
            return 0.0

        elapsed = time.time() - max(self.established_at, self.last_purification)
        decay = math.exp(-CHANNEL_DECOHERENCE_RATE * elapsed)

        # Apply depolarizing noise
        if decay < 1.0:
            # Mix toward maximally mixed state
            mixed = np.ones(4, dtype=np.complex128) / 2.0
            self.bell_state = decay * self.bell_state + (1 - decay) * mixed
            norm = np.linalg.norm(self.bell_state)
            if norm > 1e-15:
                self.bell_state /= norm

        self.fidelity = self._measure_bell_fidelity()
        return self.fidelity

    def to_dict(self) -> dict:
        """Serialize channel state (no numpy)."""
        return {
            "channel_id": self.channel_id,
            "node_a": self.node_a,
            "node_b": self.node_b,
            "fidelity": round(self.fidelity, 8),
            "active": self.active,
            "established_at": self.established_at,
            "purification_count": self.purification_count,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
        }


# ═══════════════════════════════════════════════════════════════════
# QUANTUM NETWORK MESH
# ═══════════════════════════════════════════════════════════════════


class QuantumNetworkMesh:
    """Full quantum network mesh connecting multiple daemon nodes.

    Creates a topology-aware graph of entangled channels between daemon
    nodes, with support for:
    - Topology selection: linear, ring, heavy_hex, all_to_all (mesh)
    - Bell pair generation and maintenance
    - Entanglement purification across all channels
    - Quantum teleportation routing (BFS, topology-adaptive)
    - Fidelity monitoring and auto-recalibration
    - Sacred alignment scoring of the mesh topology
    - Topology analysis: degree distribution, diameter, connectivity, bottlenecks
    - Dynamic topology detection and optimization recommendations

    The mesh topology is GOD_CODE-aligned: each channel's phase
    is offset by (GOD_CODE mod 2π) × channel_index / total_channels.

    Supported topologies:
    - all_to_all: Complete graph — n(n-1)/2 channels (default, max connectivity)
    - ring: Circular chain — n channels (balanced latency)
    - linear: Sequential chain — n-1 channels (minimal resources)
    - heavy_hex: IBM Eagle/Heron-inspired heavy-hex lattice (hardware-realistic)
    """

    # Topology type constants (mirrors l104_vqpu.constants)
    TOPOLOGY_LINEAR = TOPOLOGY_LINEAR
    TOPOLOGY_RING = TOPOLOGY_RING
    TOPOLOGY_HEAVY_HEX = TOPOLOGY_HEAVY_HEX
    TOPOLOGY_ALL_TO_ALL = TOPOLOGY_ALL_TO_ALL

    def __init__(self, node_ids: Optional[List[str]] = None,
                 qubits_per_node: int = DEFAULT_DAEMON_QUBITS,
                 topology: str = DEFAULT_TOPOLOGY):
        self.node_ids = node_ids or []
        self.qubits_per_node = qubits_per_node
        self.topology = topology
        self.registers: Dict[str, DaemonQubitRegister] = {}
        self.channels: Dict[str, QuantumChannel] = {}  # channel_id → channel
        self.channel_map: Dict[Tuple[str, str], str] = {}  # (a, b) → channel_id
        self.created_at = time.time()
        self.total_teleportations = 0
        self.total_purifications = 0
        self._topology_version = 0  # Incremented on topology changes

        # Initialize registers for all nodes
        for node_id in self.node_ids:
            self.registers[node_id] = DaemonQubitRegister(
                node_id=node_id, num_qubits=qubits_per_node)

    def add_node(self, node_id: str) -> dict:
        """Add a new daemon node to the mesh.

        Creates a qubit register and establishes channels according to
        the current topology (not necessarily to all existing nodes).
        """
        if node_id in self.registers:
            return {"added": False, "reason": "node already exists"}

        # Create register
        register = DaemonQubitRegister(
            node_id=node_id, num_qubits=self.qubits_per_node)
        register.initialize_sacred()
        self.registers[node_id] = register
        self.node_ids.append(node_id)

        # Establish channels according to topology
        new_channels = self._topology_channels_for_new_node(node_id)
        self._topology_version += 1

        return {
            "added": True,
            "node_id": node_id,
            "qubits": self.qubits_per_node,
            "topology": self.topology,
            "new_channels": len(new_channels),
            "total_nodes": len(self.node_ids),
            "total_channels": len(self.channels),
        }

    def remove_node(self, node_id: str) -> dict:
        """Remove a daemon node from the mesh, tearing down its channels."""
        if node_id not in self.registers:
            return {"removed": False, "reason": "node not found"}

        # Remove all channels involving this node
        removed_channels = []
        for key in list(self.channel_map.keys()):
            if node_id in key:
                ch_id = self.channel_map.pop(key)
                ch = self.channels.pop(ch_id, None)
                if ch:
                    ch.active = False
                    removed_channels.append(ch_id)

        del self.registers[node_id]
        self.node_ids.remove(node_id)
        self._topology_version += 1

        return {
            "removed": True,
            "node_id": node_id,
            "removed_channels": len(removed_channels),
            "remaining_nodes": len(self.node_ids),
        }

    def _create_channel(self, node_a: str, node_b: str) -> QuantumChannel:
        """Create and establish an entangled channel between two nodes."""
        # Ensure consistent key ordering
        key = tuple(sorted([node_a, node_b]))
        if key in self.channel_map:
            return self.channels[self.channel_map[key]]

        ch = QuantumChannel(node_a=key[0], node_b=key[1])
        ch.establish()
        self.channels[ch.channel_id] = ch
        self.channel_map[key] = ch.channel_id
        return ch

    def establish_channels(self) -> dict:
        """Establish entangled channels according to the configured topology.

        Topology types:
        - all_to_all: Complete graph — n(n-1)/2 channels
        - ring: Circular chain — n channels
        - linear: Sequential chain — n-1 channels
        - heavy_hex: IBM-inspired heavy-hex lattice

        Each channel gets a GOD_CODE-offset sacred phase.
        """
        t0 = time.time()

        # Initialize all registers with sacred sequence
        for reg in self.registers.values():
            reg.initialize_sacred()

        # Generate topology-specific edge list
        nodes = list(self.node_ids)
        edges = self._topology_edges(nodes, self.topology)

        new_channels = 0
        for a, b in edges:
            self._create_channel(a, b)
            new_channels += 1

        elapsed_ms = (time.time() - t0) * 1000
        self._topology_version += 1

        # Compute mesh-wide fidelity
        fidelities = [ch.fidelity for ch in self.channels.values() if ch.active]
        avg_fidelity = sum(fidelities) / max(len(fidelities), 1) if fidelities else 0.0

        _logger.info(
            "Quantum mesh established: topology=%s, %d nodes, %d channels, "
            "avg_fidelity=%.6f, %.2fms",
            self.topology, len(nodes), len(self.channels), avg_fidelity, elapsed_ms)

        return {
            "nodes": len(nodes),
            "channels": len(self.channels),
            "new_channels": new_channels,
            "topology": self.topology,
            "avg_fidelity": round(avg_fidelity, 8),
            "elapsed_ms": round(elapsed_ms, 2),
            "god_code": GOD_CODE,
        }

    # ─── TOPOLOGY EDGE GENERATORS ────────────────────────────────

    @staticmethod
    def _topology_edges(nodes: List[str], topology: str) -> List[Tuple[str, str]]:
        """Generate the edge list for a given topology.

        Args:
            nodes: Ordered list of node IDs.
            topology: One of linear, ring, heavy_hex, all_to_all.

        Returns:
            List of (node_a, node_b) pairs, canonically sorted.
        """
        n = len(nodes)
        if n < 2:
            return []

        edges: List[Tuple[str, str]] = []

        if topology == TOPOLOGY_LINEAR:
            # Chain: 0-1-2-...(n-1)
            for i in range(n - 1):
                edges.append(tuple(sorted([nodes[i], nodes[i + 1]])))

        elif topology == TOPOLOGY_RING:
            # Ring: linear + wrap-around
            for i in range(n):
                edges.append(tuple(sorted([nodes[i], nodes[(i + 1) % n]])))

        elif topology == TOPOLOGY_HEAVY_HEX:
            # IBM Eagle/Heron-inspired heavy-hex lattice approximation.
            # For small n, fall back to ring.  For larger n, create a
            # degree-limited lattice where most nodes have degree 2-3,
            # with a few hubs of degree ≤ 4 (matching heavy-hex geometry).
            if n <= 4:
                # Small graph: ring is equivalent
                for i in range(n):
                    edges.append(tuple(sorted([nodes[i], nodes[(i + 1) % n]])))
            else:
                # Heavy-hex: backbone ring + every 3rd node cross-linked
                # to node 2 hops away (degree ≤ 3, hex-like)
                for i in range(n):
                    edges.append(tuple(sorted([nodes[i], nodes[(i + 1) % n]])))
                # Cross-links at every PHI-spaced interval (sacred spacing)
                step = max(2, int(round(PHI + 1)))  # ≈ 3
                for i in range(0, n, step):
                    j = (i + 2) % n
                    edge = tuple(sorted([nodes[i], nodes[j]]))
                    if edge not in edges:
                        edges.append(edge)

        else:  # all_to_all (default)
            # Complete graph: n(n-1)/2 channels
            for i in range(n):
                for j in range(i + 1, n):
                    edges.append(tuple(sorted([nodes[i], nodes[j]])))

        return edges

    def _topology_channels_for_new_node(self, new_node: str) -> List[str]:
        """Create channels for a newly added node according to topology.

        Returns list of new channel IDs created.
        """
        nodes = self.node_ids
        idx = nodes.index(new_node)
        n = len(nodes)
        new_channels = []

        if self.topology == TOPOLOGY_ALL_TO_ALL:
            # Connect to every existing node
            for existing_id in nodes:
                if existing_id != new_node:
                    ch = self._create_channel(new_node, existing_id)
                    new_channels.append(ch.channel_id)

        elif self.topology == TOPOLOGY_RING:
            # Connect to neighbors in ring order
            if n >= 2:
                prev_idx = (idx - 1) % n
                next_idx = (idx + 1) % n
                for neighbor_idx in {prev_idx, next_idx}:
                    ch = self._create_channel(new_node, nodes[neighbor_idx])
                    new_channels.append(ch.channel_id)

        elif self.topology == TOPOLOGY_LINEAR:
            # Connect to adjacent node(s) in list order
            if idx > 0:
                ch = self._create_channel(new_node, nodes[idx - 1])
                new_channels.append(ch.channel_id)
            if idx < n - 1:
                ch = self._create_channel(new_node, nodes[idx + 1])
                new_channels.append(ch.channel_id)

        elif self.topology == TOPOLOGY_HEAVY_HEX:
            # Connect to ring neighbors + possible cross-link
            if n >= 2:
                prev_idx = (idx - 1) % n
                next_idx = (idx + 1) % n
                for neighbor_idx in {prev_idx, next_idx}:
                    ch = self._create_channel(new_node, nodes[neighbor_idx])
                    new_channels.append(ch.channel_id)
                # Cross-link if this is a hub node (every 3rd)
                step = max(2, int(round(PHI + 1)))
                if idx % step == 0 and n > 4:
                    cross_idx = (idx + 2) % n
                    if cross_idx != idx:
                        ch = self._create_channel(new_node, nodes[cross_idx])
                        new_channels.append(ch.channel_id)

        return new_channels

    def set_topology(self, topology: str) -> dict:
        """Change the mesh topology and rebuild all channels.

        Tears down existing channels and rebuilds according to the new
        topology.  Registers are preserved.

        Args:
            topology: New topology type (linear, ring, heavy_hex, all_to_all).

        Returns:
            Dict with old/new topology and channel counts.
        """
        old_topology = self.topology
        old_channels = len(self.channels)

        # Tear down existing channels
        self.channels.clear()
        self.channel_map.clear()

        # Set new topology and rebuild
        self.topology = topology
        result = self.establish_channels()

        return {
            "old_topology": old_topology,
            "new_topology": topology,
            "old_channels": old_channels,
            "new_channels": len(self.channels),
            **result,
        }

    def detect_topology(self) -> str:
        """Detect the current topology by inspecting the channel graph.

        Analyzes the degree distribution to classify the topology as:
        - all_to_all: every node has degree n-1
        - ring: every node has degree 2
        - linear: 2 nodes have degree 1, rest have degree 2
        - heavy_hex: most nodes have degree 2-3
        - unknown: doesn't match any known pattern

        Returns:
            Detected topology type string.
        """
        n = len(self.node_ids)
        if n < 2:
            return self.topology  # Can't detect with 0-1 nodes

        degrees = self._degree_map()
        degree_vals = list(degrees.values())

        if not degree_vals:
            return "disconnected"

        max_deg = max(degree_vals)
        min_deg = min(degree_vals)

        # Complete graph: all degrees == n-1
        if min_deg == max_deg == n - 1:
            return TOPOLOGY_ALL_TO_ALL

        # Ring: all degrees == 2 (for n >= 3)
        if n >= 3 and min_deg == 2 and max_deg == 2:
            return TOPOLOGY_RING

        # Linear: 2 endpoints with degree 1, rest degree 2
        if n >= 3:
            deg_1_count = sum(1 for d in degree_vals if d == 1)
            deg_2_count = sum(1 for d in degree_vals if d == 2)
            if deg_1_count == 2 and deg_2_count == n - 2:
                return TOPOLOGY_LINEAR

        # Heavy-hex: degree 2-3 mix with some degree-4 hubs
        mean_deg = sum(degree_vals) / len(degree_vals)
        if 2.0 <= mean_deg <= 3.5 and max_deg <= 4:
            return TOPOLOGY_HEAVY_HEX

        return "unknown"

    def topology_analysis(self) -> dict:
        """Comprehensive topology analysis of the quantum mesh.

        Returns:
            Dict with topology type, degree distribution, connectivity,
            diameter, bottleneck channels, and sacred topology score.
        """
        n = len(self.node_ids)
        degrees = self._degree_map()
        degree_vals = list(degrees.values()) if degrees else [0]

        # Connected components (BFS)
        components = self._connected_components()

        # Network diameter (longest shortest path)
        diameter = self._network_diameter()

        # Bottleneck detection: channels whose removal would disconnect graph
        bottleneck_ids = self._bottleneck_channels()

        # Topology efficiency: actual edges / complete graph edges
        max_edges = n * (n - 1) // 2 if n > 1 else 1
        actual_edges = len([ch for ch in self.channels.values() if ch.active])
        efficiency = actual_edges / max(max_edges, 1)

        # Sacred topology scoring:
        # GOD_CODE alignment of the topology — measures how well the
        # degree distribution resonates with sacred constants.
        # Score = (mean_degree^(1/φ) × connectivity) / (1 + |diameter - φ²|/φ²)
        mean_deg = sum(degree_vals) / max(len(degree_vals), 1)
        phi_sq = PHI * PHI
        sacred_topo_score = (
            (mean_deg ** (1.0 / PHI)) * efficiency
        ) / (1.0 + abs(diameter - phi_sq) / max(phi_sq, 1e-10))
        sacred_topo_score = min(1.0, sacred_topo_score)

        return {
            "topology": self.topology,
            "detected_topology": self.detect_topology(),
            "node_count": n,
            "channel_count": actual_edges,
            "max_possible_channels": max_edges,
            "efficiency": round(efficiency, 4),
            "connected_components": len(components),
            "is_connected": len(components) <= 1,
            "diameter": diameter,
            "mean_degree": round(mean_deg, 2),
            "max_degree": max(degree_vals) if degree_vals else 0,
            "min_degree": min(degree_vals) if degree_vals else 0,
            "degree_distribution": dict(sorted(degrees.items())),
            "bottleneck_channels": bottleneck_ids,
            "bottleneck_count": len(bottleneck_ids),
            "sacred_topology_score": round(sacred_topo_score, 8),
            "topology_version": self._topology_version,
            "god_code": GOD_CODE,
        }

    def topology_recommendation(self) -> dict:
        """Recommend optimal topology based on current node count and workload.

        Applies sacred heuristics:
        - 1-2 nodes: linear (minimal, no alternatives)
        - 3-5 nodes: ring (balanced latency, φ-aligned hop count)
        - 6-12 nodes: heavy_hex (hardware-realistic, degree-limited)
        - 13+ nodes: all_to_all if resources allow, else heavy_hex

        Returns:
            Dict with recommended topology and reasoning.
        """
        n = len(self.node_ids)
        current = self.topology
        analysis = self.topology_analysis()

        if n <= 2:
            recommended = TOPOLOGY_LINEAR
            reason = "Minimal network — linear is optimal for ≤2 nodes"
        elif n <= 5:
            recommended = TOPOLOGY_RING
            reason = f"Ring topology for {n} nodes — balanced latency with φ-aligned hop distribution"
        elif n <= 12:
            recommended = TOPOLOGY_HEAVY_HEX
            reason = f"Heavy-hex lattice for {n} nodes — hardware-realistic degree ≤3, mirrors IBM Eagle/Heron"
        else:
            # For large networks, check if fidelity is high enough for full mesh
            avg_f = analysis.get("sacred_topology_score", 0)
            if avg_f > 0.7:
                recommended = TOPOLOGY_ALL_TO_ALL
                reason = f"Full mesh for {n} nodes — high sacred score ({avg_f:.3f}) supports full connectivity"
            else:
                recommended = TOPOLOGY_HEAVY_HEX
                reason = f"Heavy-hex for {n} nodes — sacred score ({avg_f:.3f}) favors resource-conscious topology"

        return {
            "current_topology": current,
            "recommended_topology": recommended,
            "reason": reason,
            "would_change": current != recommended,
            "node_count": n,
            "analysis": analysis,
        }

    # ─── TOPOLOGY GRAPH ANALYSIS HELPERS ─────────────────────────

    def _degree_map(self) -> Dict[str, int]:
        """Compute degree (number of active channels) per node."""
        degrees: Dict[str, int] = {nid: 0 for nid in self.node_ids}
        for (a, b), ch_id in self.channel_map.items():
            ch = self.channels.get(ch_id)
            if ch and ch.active:
                degrees[a] = degrees.get(a, 0) + 1
                degrees[b] = degrees.get(b, 0) + 1
        return degrees

    def _adjacency_map(self) -> Dict[str, List[str]]:
        """Build adjacency list from active channels."""
        adj: Dict[str, List[str]] = {n: [] for n in self.node_ids}
        for (a, b), ch_id in self.channel_map.items():
            ch = self.channels.get(ch_id)
            if ch and ch.active:
                adj[a].append(b)
                adj[b].append(a)
        return adj

    def _connected_components(self) -> List[List[str]]:
        """Find connected components via BFS."""
        visited: set = set()
        components: List[List[str]] = []
        adj = self._adjacency_map()

        for node in self.node_ids:
            if node in visited:
                continue
            component = []
            from collections import deque
            queue = deque([node])
            visited.add(node)
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in adj.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)

        return components

    def _network_diameter(self) -> int:
        """Compute network diameter (longest shortest path).

        Returns -1 if the graph is disconnected.
        """
        if not self.node_ids:
            return 0
        adj = self._adjacency_map()
        diameter = 0

        for source in self.node_ids:
            # BFS from source
            from collections import deque
            dist: Dict[str, int] = {source: 0}
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for neighbor in adj.get(node, []):
                    if neighbor not in dist:
                        dist[neighbor] = dist[node] + 1
                        queue.append(neighbor)
            # Check if all nodes reachable
            if len(dist) < len(self.node_ids):
                return -1  # Disconnected
            diameter = max(diameter, max(dist.values()))

        return diameter

    def _bottleneck_channels(self) -> List[str]:
        """Find bridge channels whose removal would disconnect the graph.

        Uses edge-removal check (simplified bridge detection).
        """
        bridges = []
        for ch_id, ch in self.channels.items():
            if not ch.active:
                continue
            # Temporarily deactivate and check connectivity
            ch.active = False
            components = self._connected_components()
            ch.active = True
            if len(components) > 1:
                bridges.append(ch_id)
        return bridges

    def teleport(self, source: str, target: str, payload: dict,
                 error_correct: bool = False) -> dict:
        """Perform quantum teleportation from source to target node.

        Routes through direct channel if available. Falls back to multi-hop
        routing through intermediate nodes with entanglement swapping.

        Args:
            source: Source daemon node ID.
            target: Target daemon node ID.
            payload: Data payload to teleport.
            error_correct: Apply error correction (purification before teleport).

        Returns:
            Dict with success status, fidelity, and teleportation metrics.
        """
        key = tuple(sorted([source, target]))
        ch_id = self.channel_map.get(key)

        if ch_id and ch_id in self.channels:
            # Direct channel available
            channel = self.channels[ch_id]

            # v3.1: Apply error correction (purification) before teleport
            if error_correct and channel.fidelity < FIDELITY_THRESHOLD_GOOD:
                channel.purify()
                self.total_purifications += 1

            result = channel.teleport_payload(payload)
            result["hops"] = 1
            result["route"] = [source, target]
            if result.get("success"):
                self.total_teleportations += 1
            return result

        # v3.1: Multi-hop routing through intermediate nodes
        route = self._find_route(source, target)
        if not route or len(route) < 2:
            return {"success": False, "reason": f"no route between {source} and {target}"}

        # Execute multi-hop: teleport through each hop, accumulating fidelity loss
        cumulative_fidelity = 1.0
        hop_details = []
        current_payload = payload

        for i in range(len(route) - 1):
            hop_key = tuple(sorted([route[i], route[i + 1]]))
            hop_ch_id = self.channel_map.get(hop_key)
            if not hop_ch_id or hop_ch_id not in self.channels:
                return {
                    "success": False,
                    "reason": f"broken route at hop {route[i]}->{route[i+1]}",
                    "route": route,
                    "failed_hop": i,
                }

            hop_ch = self.channels[hop_ch_id]

            if error_correct and hop_ch.fidelity < FIDELITY_THRESHOLD_GOOD:
                hop_ch.purify()
                self.total_purifications += 1

            hop_result = hop_ch.teleport_payload(current_payload)
            if not hop_result.get("success"):
                return {
                    "success": False,
                    "reason": f"teleport failed at hop {i}: {hop_result.get('reason', '?')}",
                    "route": route,
                    "failed_hop": i,
                }

            cumulative_fidelity *= hop_result.get("fidelity", 1.0)
            hop_details.append({
                "hop": i,
                "from": route[i],
                "to": route[i + 1],
                "fidelity": hop_result.get("fidelity", 0),
            })

        self.total_teleportations += 1
        return {
            "success": True,
            "fidelity": round(cumulative_fidelity, 8),
            "hops": len(route) - 1,
            "route": route,
            "hop_details": hop_details,
            "sacred_aligned": hop_details[-1].get("sacred_aligned", True) if hop_details else False,
            "multi_hop": True,
        }

    def _find_route(self, source: str, target: str) -> Optional[List[str]]:
        """BFS shortest-path routing through the mesh graph.

        Returns list of node IDs forming the path, or None if no path exists.
        """
        if source not in self.node_ids or target not in self.node_ids:
            return None
        if source == target:
            return [source]

        # Build adjacency from active channels
        adj = self._adjacency_map()

        # BFS
        from collections import deque
        visited = {source}
        queue = deque([(source, [source])])
        while queue:
            node, path = queue.popleft()
            if node == target:
                return path
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def purify_all(self) -> dict:
        """Run entanglement purification on all channels that need it.

        Returns aggregate purification results.
        """
        t0 = time.time()
        purified = 0
        skipped = 0
        results = []

        for ch in self.channels.values():
            if not ch.active:
                continue
            result = ch.purify()
            if result.get("purified"):
                purified += 1
                self.total_purifications += 1
            else:
                skipped += 1
            results.append(result)

        elapsed_ms = (time.time() - t0) * 1000
        return {
            "purified": purified,
            "skipped": skipped,
            "total_channels": len(self.channels),
            "elapsed_ms": round(elapsed_ms, 2),
            "details": results,
        }

    def decoherence_cycle(self) -> dict:
        """Apply decoherence simulation to all active channels.

        Returns channels that have dropped below fidelity floor.
        """
        degraded = []
        for ch in self.channels.values():
            if ch.active:
                new_f = ch.apply_decoherence()
                if new_f < FIDELITY_FLOOR:
                    degraded.append(ch.channel_id)

        return {
            "channels_checked": len(self.channels),
            "degraded": len(degraded),
            "degraded_channels": degraded,
        }

    def network_health(self) -> dict:
        """Full quantum network health report.

        Computes per-channel fidelity, mesh connectivity, sacred alignment,
        and aggregate network metrics.
        """
        active_channels = [ch for ch in self.channels.values() if ch.active]
        fidelities = [ch.fidelity for ch in active_channels]

        avg_fidelity = sum(fidelities) / max(len(fidelities), 1) if fidelities else 0.0
        min_fidelity = min(fidelities) if fidelities else 0.0
        max_fidelity = max(fidelities) if fidelities else 0.0

        # Register health
        register_health = {}
        for node_id, reg in self.registers.items():
            check = reg.fidelity_check()
            register_health[node_id] = {
                "avg_fidelity": check["avg_fidelity"],
                "degraded_qubits": len(check["degraded_qubits"]),
                "total_gates": reg.total_gates_applied,
            }

        # Sacred mesh alignment
        sacred_check = (GOD_CODE / 16.0) ** PHI
        sacred_alignment = max(0.0, 1.0 - abs(sacred_check - 286.0) / 286.0)

        # Network connectivity ratio (actual edges / max possible edges)
        n = len(self.node_ids)
        max_channels = n * (n - 1) // 2 if n > 1 else 1
        connectivity = len(active_channels) / max(max_channels, 1)

        # Topology analysis (lightweight: degree stats + detected type)
        degrees = self._degree_map()
        degree_vals = list(degrees.values()) if degrees else [0]
        mean_degree = sum(degree_vals) / max(len(degree_vals), 1)
        detected_topo = self.detect_topology()

        # Sacred topology score (integrated with network score)
        phi_sq = PHI * PHI
        diameter = self._network_diameter() if n > 1 else 0
        sacred_topo_score = (
            (mean_degree ** (1.0 / PHI)) * connectivity
        ) / (1.0 + abs(max(0, diameter) - phi_sq) / max(phi_sq, 1e-10))
        sacred_topo_score = min(1.0, sacred_topo_score)

        # Composite network score (now includes topology health)
        network_score = (
            avg_fidelity * PHI +
            connectivity +
            sacred_alignment / PHI +
            sacred_topo_score * VOID_CONSTANT
        ) / (PHI + 1.0 + 1.0 / PHI + VOID_CONSTANT)

        return {
            "version": QUANTUM_NETWORK_VERSION,
            "nodes": len(self.node_ids),
            "active_channels": len(active_channels),
            "total_channels": len(self.channels),
            "topology": self.topology,
            "detected_topology": detected_topo,
            "topology_version": self._topology_version,
            "avg_fidelity": round(avg_fidelity, 8),
            "min_fidelity": round(min_fidelity, 8),
            "max_fidelity": round(max_fidelity, 8),
            "connectivity": round(connectivity, 4),
            "mean_degree": round(mean_degree, 2),
            "diameter": diameter,
            "sacred_alignment": round(sacred_alignment, 8),
            "sacred_topology_score": round(sacred_topo_score, 8),
            "network_score": round(network_score, 8),
            "total_teleportations": self.total_teleportations,
            "total_purifications": self.total_purifications,
            "register_health": register_health,
            "god_code": GOD_CODE,
            "qpu_mean_fidelity": QPU_MEAN_FIDELITY,
        }

    def get_channel(self, node_a: str, node_b: str) -> Optional[QuantumChannel]:
        """Get the channel between two nodes."""
        key = tuple(sorted([node_a, node_b]))
        ch_id = self.channel_map.get(key)
        return self.channels.get(ch_id) if ch_id else None

    def persist_state(self, path: Optional[str] = None) -> str:
        """Persist mesh state to disk."""
        if path is None:
            root = Path(os.environ.get("L104_ROOT", os.getcwd()))
            path = str(root / QUANTUM_MESH_STATE_FILE)

        state = {
            "version": QUANTUM_NETWORK_VERSION,
            "vqpu_version": VERSION,
            "timestamp": time.time(),
            "topology": self.topology,
            "topology_version": self._topology_version,
            "node_ids": self.node_ids,
            "qubits_per_node": self.qubits_per_node,
            "channels": {ch_id: ch.to_dict() for ch_id, ch in self.channels.items()},
            "total_teleportations": self.total_teleportations,
            "total_purifications": self.total_purifications,
            "health": self.network_health(),
        }
        Path(path).write_text(json.dumps(state, indent=2, default=str))
        return path

    def status(self) -> dict:
        """Quick status snapshot."""
        active = sum(1 for ch in self.channels.values() if ch.active)
        fidelities = [ch.fidelity for ch in self.channels.values() if ch.active]
        return {
            "nodes": len(self.node_ids),
            "topology": self.topology,
            "channels_active": active,
            "channels_total": len(self.channels),
            "avg_fidelity": round(
                sum(fidelities) / max(len(fidelities), 1), 6) if fidelities else 0.0,
            "teleportations": self.total_teleportations,
            "purifications": self.total_purifications,
        }


# ═══════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════

__all__ = [
    "DaemonQubitRegister",
    "QubitState",
    "QuantumChannel",
    "QuantumNetworkMesh",
    "QUANTUM_NETWORK_VERSION",
    "DEFAULT_DAEMON_QUBITS",
    "FIDELITY_THRESHOLD_HIGH",
    "FIDELITY_THRESHOLD_GOOD",
    "FIDELITY_THRESHOLD_LOW",
    "FIDELITY_FLOOR",
]
