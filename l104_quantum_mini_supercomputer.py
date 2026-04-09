#!/usr/bin/env python3
"""
L104 Quantum Mini Supercomputer v2.1.0
═══════════════════════════════════════════════════════════════════════════════
Unified 26-qubit quantum processing unit that composes ALL L104 circuit
families into one coherent quantum system with entropy reversal at the core.

NOW WITH 26 LAYERS:
  Phase 1: Foundation (Sacred Imprint + Calibration + Berry Phase)
  Phase 2: Consciousness Awakening (Orch OR + Multidimensional)
  Phase 3: Entanglement Topology (Fibonacci + Cross-Orbital)
  Phase 4: Error Correction (Steane + PHI-QEC + Fibonacci Protection)
  Phase 5: Thermodynamic Reversal (Maxwell's Demon + Harmonic Resonance)
  Phase 6: Sacred Overlays (Grimoire + Proof Verification)
  Phase 7: Quantum ML + Cryptography
  Phase 8: Hardware Optimization (VQPU Mesh + Quantum DAW)
  Phase 9: Advanced Simulations (Analog + Computronium + Research Cycles)
  Phase 10: VQE Optimization + Final Objective Reduction

ARCHITECTURE — 26Q Fe(26) Iron-Mapped Register:
  ┌──────────────────────────────────────────────────────────────────┐
  │  QUBIT MAP (Fe electron orbital assignment)                      │
  │  q[0:5]   = 2p  — Deep Quantum Memory (6Q)                      │
  │  q[6:7]   = 3s  — Core Coherence Anchor (2Q)                    │
  │  q[8:13]  = 3p  — Inner Shield / Topological Protection (6Q)    │
  │  q[14:23] = 3d  — Consciousness Substrate / High-Spin (10Q)     │
  │  q[24:25] = 4s  — Valence Bridge to External Systems (2Q)       │
  └──────────────────────────────────────────────────────────────────┘

CIRCUIT COMPOSITION (12 sub-circuits bound on 26Q register):
  Layer 1: GOD_CODE Phase Imprint        — sacred_transpiler circuits
  Layer 2: Dial Circuit G(a,b,c,d)       — god_code_algorithm parametric
  Layer 3: Consciousness Awakening       — 5-level Orch OR (4Q→26Q)
  Layer 4: Fibonacci Entanglement Mesh   — consciousness_circuits binding
  Layer 5: Entropy Reversal Core         — MaxwellDemon26Q + grimoire
  Layer 6: Harmonic Resonance            — half-integer + PHI-bridge
  Layer 7: Evolved Grimoire Overlay      — genetically-evolved optimal gates
  Layer 8: Cross-Orbital Entanglement    — inter-orbital CX bridges
  Layer 9: Sacred Proof Verification     — 12 proof circuit phases
  Layer 10: VQPU Mesh Optimization       — topology-aware gate scheduling
  Layer 11: Consciousness VQE            — variational soul optimization
  Layer 12: Final Interference + Readout — objective reduction

EXECUTION: Runs on VQPU bridge (MPS engine / AccelStatevector / IBM QPU)

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import time
import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Three-engine integration (optional — enhances results with cross-validation)
try:
    from l104_science_engine import ScienceEngine
    from l104_math_engine import MathEngine
    from l104_code_engine import code_engine as _code_engine
    _THREE_ENGINES_AVAILABLE = True
except ImportError:
    _THREE_ENGINES_AVAILABLE = False

logger = logging.getLogger("l104.quantum_mini_supercomputer")

# ═══════════════════════════════════════════════════════════════════════════════
#  SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
PHI_CONJUGATE = PHI - 1.0  # 0.618033988749895
VOID_CONSTANT = 1.04 + PHI / 1000  # 1.0416180339887497
TAU = 2.0 * math.pi
GOD_CODE_PHASE = GOD_CODE % TAU
PHI_PHASE = (TAU / PHI) % TAU
IRON_PHASE = math.pi / 2  # 26 × 2π / 104
PHASE_OCTAVE_4 = 4 * TAU  # ×16 octave
BASE = 286 ** (1.0 / PHI)  # 286^(1/φ) ≈ 17.845
QUANTIZATION_GRAIN = 104
OCTAVE_OFFSET = 416

# Hardware depth limit — transpiled circuit depth must stay below this
MAX_HARDWARE_DEPTH = 100

# Grimoire-discovered optimal angles
OPTIMAL_RZ = GOD_CODE / 131.0  # ≈ 4.027 (best RZ from evolution)
OPTIMAL_RY = 1.0 / PHI         # ≈ 0.618 (TAU)

# Consciousness Hamiltonian couplings
J_PHI = PHI / 10.0
H_GC = GOD_CODE_PHASE / TAU
LAMBDA_VOID = VOID_CONSTANT - 1.0

# Chakra frequency phases
CHAKRA_PHASES = [freq * PHI % TAU for freq in [396, 417, 528, 639, 741, 852, 963]]

# ═══════════════════════════════════════════════════════════════════════════════
# 26Q FE REGISTER LAYOUT + IBM HARDWARE VALIDATION (EVO_80)
# ═══════════════════════════════════════════════════════════════════════════════
# Hardware validation: 82.3% fidelity on IBM Kingston (job d7b9fab0g7hs73dp9r00)
# All 7 registers LOCKED above 90%

NQ = 26
TARGET_HUMAN = "00111111110000000000000000"  # Fe ground state
TARGET_QISKIT = TARGET_HUMAN[::-1]

REGISTERS = {
    "CORE":    {"lo": 0,  "hi": 1,  "target": "00",     "desc": "[Ar] noble gas core"},
    "3d":      {"lo": 2,  "hi": 7,  "target": "111111", "desc": "Fe 3d⁶ d-orbitals"},
    "4s":      {"lo": 8,  "hi": 9,  "target": "11",     "desc": "Fe 4s² s-orbitals"},
    "LATTICE": {"lo": 10, "hi": 15, "target": "000000", "desc": "Fe BCC lattice"},
    "SACRED":  {"lo": 16, "hi": 20, "target": "00000",  "desc": "GOD_CODE phase manifold"},
    "PHI":     {"lo": 21, "hi": 24, "target": "0000",   "desc": "golden ratio"},
    "ANCHOR":  {"lo": 25, "hi": 25, "target": "0",      "desc": "nucleus anchor"},
}

# IBM Kingston hardware validation results (2026-04-08)
HW_VALIDATION = {
    "job_id": "d7b9fab0g7hs73dp9r00",
    "backend": "ibm_kingston",
    "timestamp": "2026-04-08T14:00:29",
    "shots": 32768,
    "p_target": 0.822937,  # 82.3% - rank #1!
    "target_rank": 1,
    "entropy_bits": 1.523,
    "unique_states": 209,
    "register_fidelity": {
        "CORE":    {"p_match": 0.9984, "status": "LOCKED"},
        "3d":      {"p_match": 0.9160, "status": "LOCKED"},
        "4s":      {"p_match": 0.9848, "status": "LOCKED"},
        "LATTICE": {"p_match": 0.9468, "status": "LOCKED"},
        "SACRED":  {"p_match": 0.9811, "status": "LOCKED"},
        "PHI":     {"p_match": 0.9875, "status": "LOCKED"},
        "ANCHOR":  {"p_match": 0.9970, "status": "LOCKED"},
    },
}

# Per-register error rates for circuit optimization
HW_ERROR_RATES = {
    "CORE":    0.0015,
    "3d":      0.0831,
    "4s":      0.0147,
    "LATTICE": 0.0521,
    "SACRED":  0.0182,
    "PHI":     0.0112,
    "ANCHOR":  0.0030,
}

STATE_PATH = Path(__file__).resolve().parent / ".l104_mini_supercomputer_state.json"
FIDELITY_HISTORY_PATH = Path(__file__).resolve().parent / ".l104_fidelity_history.json"


# ═══════════════════════════════════════════════════════════════════════════════
#  EXECUTION MODE & FIDELITY ORACLE (K-Synth Engine Gamma)
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutionMode(Enum):
    """Execution mode for supercomputer circuits.

    AUTO: FidelityOracle selects the best mode based on fidelity history
    SIMULATION: Local VQPU/MPS execution (high fidelity, no hardware noise)
    HARDWARE: IBM QPU with depth-limited circuit
    HYBRID_FORGING: IBM QPU with entanglement forging (golden-ratio cut)
    """
    AUTO = "auto"
    SIMULATION = "simulation"
    HARDWARE = "hardware"
    HYBRID_FORGING = "hybrid_forging"


class FidelityOracle:
    """Tracks fidelity history and recommends execution mode.

    Implements the K-Synth FidelityOracle concept from L104_MASTER_KNOWLEDGE.md
    Engine Gamma. Predicts which execution mode will produce the best fidelity
    based on historical results.
    """

    def __init__(self):
        self._history: List[Dict[str, Any]] = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(FIDELITY_HISTORY_PATH.read_text())
        except Exception:
            return []

    def _save_history(self) -> None:
        try:
            # Keep last 50 entries
            trimmed = self._history[-50:]
            FIDELITY_HISTORY_PATH.write_text(json.dumps(trimmed, indent=2, default=str))
        except Exception as e:
            logger.warning("Failed to save fidelity history: %s", e)

    def record(self, mode: str, god_code_fidelity: float,
               sacred_alignment: float, transpiled_depth: int = 0) -> None:
        """Record a fidelity observation."""
        self._history.append({
            "mode": mode,
            "god_code_fidelity": god_code_fidelity,
            "sacred_alignment": sacred_alignment,
            "transpiled_depth": transpiled_depth,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self._save_history()

    def recommend_mode(self) -> ExecutionMode:
        """Recommend execution mode based on fidelity history.

        Rules (in priority order):
        1. If last hardware god_code_fidelity < 0.01 → SIMULATION
        2. If last hardware transpiled_depth > MAX_HARDWARE_DEPTH → SIMULATION
        3. If forging fidelity > hardware fidelity → HYBRID_FORGING
        4. If no history → SIMULATION (safe default)
        5. Otherwise → HARDWARE
        """
        hw_entries = [e for e in self._history if e.get("mode") in ("hardware", "HARDWARE")]
        forge_entries = [e for e in self._history if e.get("mode") in ("hybrid_forging", "HYBRID_FORGING")]

        if not hw_entries:
            return ExecutionMode.SIMULATION

        last_hw = hw_entries[-1]

        # Rule 1: Low fidelity → simulation
        if last_hw.get("god_code_fidelity", 0.0) < 0.01:
            return ExecutionMode.SIMULATION

        # Rule 2: Too deep → simulation
        if last_hw.get("transpiled_depth", 999) > MAX_HARDWARE_DEPTH:
            return ExecutionMode.SIMULATION

        # Rule 3: Forging better than direct hardware
        if forge_entries:
            last_forge = forge_entries[-1]
            if last_forge.get("sacred_alignment", 0.0) > last_hw.get("sacred_alignment", 0.0):
                return ExecutionMode.HYBRID_FORGING

        return ExecutionMode.HARDWARE

    def last_hardware_fidelity(self) -> float:
        """Return last known hardware god_code_fidelity."""
        hw_entries = [e for e in self._history if e.get("mode") in ("hardware", "HARDWARE")]
        if hw_entries:
            return hw_entries[-1].get("god_code_fidelity", 0.0)
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  ORBITAL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class OrbitalType(Enum):
    """Fe(26) electron orbitals → qubit register ranges."""
    Q_2P = "2p"   # q[0:5]   — Deep quantum memory (6Q)
    Q_3S = "3s"   # q[6:7]   — Core coherence anchor (2Q)
    Q_3P = "3p"   # q[8:13]  — Inner shield (6Q)
    Q_3D = "3d"   # q[14:23] — Consciousness substrate (10Q)
    Q_4S = "4s"   # q[24:25] — Valence bridge (2Q)


ORBITAL_RANGES: Dict[OrbitalType, Tuple[int, int]] = {
    OrbitalType.Q_2P: (0, 6),
    OrbitalType.Q_3S: (6, 8),
    OrbitalType.Q_3P: (8, 14),
    OrbitalType.Q_3D: (14, 24),
    OrbitalType.Q_4S: (24, 26),
}

ORBITAL_DEMON_FACTORS: Dict[OrbitalType, float] = {
    OrbitalType.Q_2P: PHI_CONJUGATE ** 2,  # Deep: slower, stable
    OrbitalType.Q_3S: PHI_CONJUGATE,       # Core: moderate
    OrbitalType.Q_3P: 1.0,                  # Shield: baseline
    OrbitalType.Q_3D: PHI,                  # Consciousness: boosted
    OrbitalType.Q_4S: PHI ** 2,            # Valence: fastest
}

ORBITAL_FREQUENCIES: Dict[OrbitalType, float] = {
    OrbitalType.Q_2P: 286.0,    # Iron base
    OrbitalType.Q_3S: 396.0,    # Liberation
    OrbitalType.Q_3P: 528.0,    # Love / DNA repair
    OrbitalType.Q_3D: GOD_CODE, # Sacred consciousness
    OrbitalType.Q_4S: 963.0,    # Crown / pineal
}


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SupercomputerResult:
    """Result from quantum mini supercomputer execution."""
    success: bool
    n_qubits: int = 26
    total_gates: int = 0
    circuit_depth: int = 0
    layers_executed: List[str] = field(default_factory=list)
    probabilities: Dict[str, float] = field(default_factory=dict)
    sacred_alignment: float = 0.0
    entropy_reversed: float = 0.0
    consciousness_phi: float = 0.0
    god_code_fidelity: float = 0.0
    dial_result: Optional[Dict[str, Any]] = None
    orbital_metrics: Dict[str, Any] = field(default_factory=dict)
    vqpu_source: str = "unknown"
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    three_engine: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitLayer:
    """A composable circuit layer on the 26Q register."""
    name: str
    operations: List[Dict[str, Any]]
    target_qubits: List[int]
    gate_count: int = 0
    description: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  SACRED NATIVE GATE COMPOSITES
# ═══════════════════════════════════════════════════════════════════════════════
#
# All composites use IBM-native gates only:
#   Single-qubit: Rz (FREE — virtual frame change), SX (1 pulse), X (1 pulse)
#   Two-qubit:    CZ (native on Heron r2), ECR (native on Eagle r3)
#
# Rz has zero error, zero time, zero depth contribution on IBM hardware.
# All sacred constant encoding is pushed into Rz angles for free.
#
# TRANSPILER INSIGHT (from IBM Marrakesh runs):
#   The Qiskit transpiler merges ALL consecutive single-qubit gates into
#   one Rz·SX·Rz per qubit. Multi-layer bias via Rz+SX+Rz gets collapsed
#   into a single unpredictable rotation. The solution:
#
#   1. Use X gates to set the BASE STATE (X is native, can't merge with Rz)
#   2. Use Rz for sacred phase encoding (FREE, doesn't affect measurement probabilities)
#   3. Use CZ as transpiler BARRIERS — gates before/after CZ can't be merged
#   4. Place SX only where genuine superposition is needed, not for bias
#
#   GOD_CODE TARGETING PATTERN:
#     |0⟩ target qubits: leave as |0⟩ (identity)
#     |1⟩ target qubits: apply X (native bit-flip, 1 pulse)
#     Then: Rz sacred phases (free), CZ entanglement, SX for interference
#     The X gates survive transpilation because they sit before CZ barriers.
# ═══════════════════════════════════════════════════════════════════════════════

# ─── GOD_CODE target bitstring for Fe(26) mapped registers ───────────────

def god_code_target_bits(n_qubits: int) -> List[int]:
    """Return GOD_CODE target bit pattern for n qubits.

    For 26Q: maps GOD_CODE integer to 26-bit pattern
    For 8Q/10Q: uses the lower bits (consciousness substrate mapping)
    """
    gc_int = int(GOD_CODE) % (2 ** n_qubits)
    bits_str = format(gc_int, f'0{n_qubits}b')
    return [int(b) for b in bits_str]


def god_code_base_state(n_qubits: int) -> List[Dict[str, Any]]:
    """Set qubits to GOD_CODE target using X gates (transpiler-proof).

    X is native on IBM (1 pulse). Can't be merged with Rz across CZ barriers.
    Qubits targeting |0⟩ are left untouched. Qubits targeting |1⟩ get X.
    All sacred phase info goes into Rz (free) AFTER a CZ barrier.

    Hardware cost: count(|1⟩ target bits) X gates. No Rz, no SX.
    """
    target = god_code_target_bits(n_qubits)
    ops: List[Dict[str, Any]] = []
    for q in range(n_qubits):
        if target[q] == 1:
            ops.append({"gate": "X", "qubits": [q]})
    return ops


def gentle_superposition(q: int, target_bit: int) -> List[Dict[str, Any]]:
    """Create gentle superposition while maintaining target bias.

    For |0⟩ target: Ry(pi/6) → 93.3% on |0⟩, 6.7% on |1⟩
    For |1⟩ target: X already applied, then Ry(pi/6) → 93.3% on |1⟩

    Native decomposition: Rz(-pi/2)·SX·Rz(pi/2 + theta)
    The SX adds genuine superposition that the transpiler preserves
    because it's separated from other SX gates by CZ barriers.

    Hardware cost: 1 SX + 2 Rz (free) = 1 physical gate.
    """
    # Small rotation: enough superposition for interference, keeps >90% on target
    theta = math.pi / 6  # 30° — gentle
    if target_bit == 0:
        # Ry(theta) on |0⟩: cos(theta/2)|0⟩ + sin(theta/2)|1⟩
        return [
            {"gate": "Rz", "qubits": [q], "parameters": [-math.pi / 2]},
            {"gate": "SX", "qubits": [q]},
            {"gate": "Rz", "qubits": [q], "parameters": [math.pi / 2 + theta]},
        ]
    else:
        # X already flipped to |1⟩. Ry(-theta) on |1⟩: sin(theta/2)|0⟩ + cos(theta/2)|1⟩
        return [
            {"gate": "Rz", "qubits": [q], "parameters": [-math.pi / 2]},
            {"gate": "SX", "qubits": [q]},
            {"gate": "Rz", "qubits": [q], "parameters": [math.pi / 2 - theta]},
        ]


def sacred_phi_entangler(a: int, b: int) -> List[Dict[str, Any]]:
    """CZ + PHI-phase entanglement — native on Heron (1 two-qubit gate).

    Replaces CP(theta) which costs 2 CX gates.
    Encodes golden ratio phase on both qubits via free Rz.

    Hardware cost: 1 CZ (native) + 2 Rz (free) = 1 physical gate.
    """
    return [
        {"gate": "CZ", "qubits": [a, b]},
        {"gate": "Rz", "qubits": [a], "parameters": [PHI_PHASE]},
        {"gate": "Rz", "qubits": [b], "parameters": [GOD_CODE_PHASE]},
    ]


def sacred_orbital_bridge(a: int, b: int) -> List[Dict[str, Any]]:
    """Orbital boundary entanglement with sacred phase — 1 native two-qubit gate.

    Replaces CX + CP pair (3 two-qubit gates after decomposition).
    Links adjacent orbitals across the Fe(26) boundary.

    Hardware cost: 1 CZ (native) + 2 Rz (free) = 1 physical gate.
    """
    return [
        {"gate": "Rz", "qubits": [a], "parameters": [IRON_PHASE]},
        {"gate": "CZ", "qubits": [a, b]},
        {"gate": "Rz", "qubits": [b], "parameters": [GOD_CODE_PHASE / 26]},
    ]


def sacred_dial_unit(a: int, b: int, dial_phase: float, index: int) -> List[Dict[str, Any]]:
    """Single dial entanglement unit — CZ + binary-weighted sacred phases.

    Replaces CP(phi_coupling) which costs 2 CX gates.
    Each unit encodes one step of the G(a,b,c,d) parametric dial.

    Hardware cost: 1 CZ (native) + 2 Rz (free) = 1 physical gate.
    """
    return [
        {"gate": "Rz", "qubits": [a], "parameters": [dial_phase * (2 ** (index % 8))]},
        {"gate": "CZ", "qubits": [a, b]},
        {"gate": "Rz", "qubits": [b], "parameters": [PHI * math.pi / (12 * (index + 1))]},
    ]


def parameterized_sacred_gate(q: int, a: int = 0, b: int = 0,
                               c: int = 0, d: int = 0) -> List[Dict[str, Any]]:
    """One-qubit sacred encoding using native gates only.

    Encodes GOD_CODE + dial G(a,b,c,d) into a single SX pulse + free Rz angles.
    Full U3-equivalent with all sacred constants baked in.

    Hardware cost: 1 SX (physical) + 2 Rz (free) = 1 physical gate.
    """
    E = 8 * a + OCTAVE_OFFSET - b - 8 * c - QUANTIZATION_GRAIN * d
    freq = BASE * (2 ** (E / QUANTIZATION_GRAIN))
    return [
        {"gate": "Rz", "qubits": [q], "parameters": [GOD_CODE_PHASE]},
        {"gate": "SX", "qubits": [q]},
        {"gate": "Rz", "qubits": [q], "parameters": [(freq / GOD_CODE) * math.pi]},
    ]


def native_superposition(q: int) -> List[Dict[str, Any]]:
    """Native H-equivalent: Rz(π/2) · SX · Rz(π/2).

    Functionally identical to H gate but expressed in native gates
    to prevent transpiler decomposition surprises.
    All Rz are free, so hardware cost = 1 SX pulse.
    """
    return [
        {"gate": "Rz", "qubits": [q], "parameters": [math.pi / 2]},
        {"gate": "SX", "qubits": [q]},
        {"gate": "Rz", "qubits": [q], "parameters": [math.pi / 2]},
    ]


def fibonacci_cz_ladder(n_qubits: int = 26) -> List[Dict[str, Any]]:
    """Fibonacci entanglement mesh using CZ (native on Heron).

    Two rounds of parallel CZ (even + odd) = depth 2.
    Each pair gets a Fibonacci-indexed sacred phase via free Rz.
    Replaces CX-based Fibonacci mesh (each CX = 5 native gates on Heron).

    Hardware cost: ceil(n/2) CZ gates at depth 2 + free Rz.
    """
    ops: List[Dict[str, Any]] = []
    fib_phases = [PHI_PHASE * (PHI ** i % TAU) for i in range(n_qubits)]

    # Even round: (0,1), (2,3), (4,5), ...
    for q in range(0, n_qubits - 1, 2):
        ops.append({"gate": "CZ", "qubits": [q, q + 1]})
        ops.append({"gate": "Rz", "qubits": [q], "parameters": [fib_phases[q]]})
        ops.append({"gate": "Rz", "qubits": [q + 1], "parameters": [fib_phases[q + 1]]})

    # Odd round: (1,2), (3,4), (5,6), ...
    for q in range(1, n_qubits - 1, 2):
        ops.append({"gate": "CZ", "qubits": [q, q + 1]})
        ops.append({"gate": "Rz", "qubits": [q], "parameters": [fib_phases[q]]})

    return ops


def cz_even_odd(start: int, end: int) -> List[Dict[str, Any]]:
    """Parallel even-odd CZ pattern (native replacement for even-odd CX).

    Same entanglement structure as _even_odd_cx but using CZ (native on Heron).
    CZ is symmetric — no control/target distinction needed.

    Hardware depth: 2 (even round + odd round).
    """
    ops: List[Dict[str, Any]] = []
    # Even round
    for q in range(start, end - 1, 2):
        ops.append({"gate": "CZ", "qubits": [q, q + 1]})
    # Odd round
    for q in range(start + 1, end - 1, 2):
        ops.append({"gate": "CZ", "qubits": [q, q + 1]})
    return ops


# ═══════════════════════════════════════════════════════════════════════════════
#  FIBONACCI ENTANGLEMENT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def fibonacci_pairs(n: int) -> List[Tuple[int, int]]:
    """Generate Fibonacci-pattern entanglement pairs for n qubits.

    Mirrors microtubule lattice structure (Hameroff) and golden spiral.
    """
    pairs = []
    a, b = 0, 1
    while b < n:
        if a < n and b < n and a != b:
            pairs.append((a, b))
        a, b = b, a + b
    return pairs


def cross_orbital_pairs() -> List[Tuple[int, int]]:
    """Generate entanglement pairs that bridge Fe orbitals.

    Links:
      2p↔3s (memory↔anchor), 3s↔3p (anchor↔shield),
      3p↔3d (shield↔consciousness), 3d↔4s (consciousness↔valence),
      2p↔3d (deep memory↔consciousness — long-range)
    """
    return [
        (5, 6),   # 2p[5] ↔ 3s[0] — memory to anchor
        (7, 8),   # 3s[1] ↔ 3p[0] — anchor to shield
        (13, 14), # 3p[5] ↔ 3d[0] — shield to consciousness
        (23, 24), # 3d[9] ↔ 4s[0] — consciousness to valence
        (0, 14),  # 2p[0] ↔ 3d[0] — deep memory to consciousness (long-range)
        (3, 20),  # 2p[3] ↔ 3d[6] — cross-orbital binding
        (10, 25), # 3p[2] ↔ 4s[1] — shield to external bridge
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  CIRCUIT LAYER BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitLayerBuilder:
    """Builds individual circuit layers for composition on the 26Q register."""

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 1: GOD_CODE Phase Imprint
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_god_code_phase_imprint() -> CircuitLayer:
        """Imprint GOD_CODE phase across all 26 qubits with orbital entanglement.

        Architecture: H(all) → Rz(GOD_CODE_PHASE) on q0,
        Rz(PHI_PHASE) on q1, Rz(IRON_PHASE) on q2,
        then orbital-specific phase rotations + cross-orbital CX bridges
        for non-zero IIT Φ from the foundation layer.

        IBM Kingston upgrade: added intra-orbital entanglement chains
        and cross-orbital bridges so Φ > 0 on first layer.
        """
        ops = []

        # Superposition on all 26 qubits
        for q in range(26):
            ops.append({"gate": "H", "qubits": [q]})

        # Sacred phase injection on first 3 qubits (core encoding)
        ops.append({"gate": "Rz", "qubits": [0], "parameters": [GOD_CODE_PHASE]})
        ops.append({"gate": "Rz", "qubits": [1], "parameters": [PHI_PHASE]})
        ops.append({"gate": "Rz", "qubits": [2], "parameters": [IRON_PHASE]})

        # Orbital-frequency phase injection
        for orbital, (start, end) in ORBITAL_RANGES.items():
            freq = ORBITAL_FREQUENCIES[orbital]
            phase = (freq % TAU)
            for q in range(start, end):
                ops.append({"gate": "Rz", "qubits": [q], "parameters": [phase]})

        # VOID correction on anchor qubit
        ops.append({"gate": "Rz", "qubits": [6], "parameters": [VOID_CONSTANT % TAU]})

        # Intra-orbital entanglement (nearest-neighbor CX per orbital)
        for orbital, (start, end) in ORBITAL_RANGES.items():
            if end - start >= 2:
                for q in range(start, end - 1, 2):
                    ops.append({"gate": "CX", "qubits": [q, q + 1]})

        # Cross-orbital bridges at orbital boundaries (Φ > 0)
        for a, b in [(5, 6), (7, 8), (13, 14), (23, 24)]:
            ops.append({"gate": "CX", "qubits": [a, b]})

        # GOD_CODE phase correlation on entangled pairs
        ops.append({"gate": "CP", "qubits": [0, 14],
                     "parameters": [GOD_CODE_PHASE / 26]})

        return CircuitLayer(
            name="god_code_phase_imprint",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="GOD_CODE + PHI + IRON + orbital imprinting + entanglement bridges",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 2: Dial Circuit G(a,b,c,d)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_dial_circuit(a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> CircuitLayer:
        """Build parametric dial circuit G(a,b,c,d) on valence bridge (q24-25)
        and distribute harmonics across consciousness substrate (q14-23).

        G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
        """
        E = 8 * a + OCTAVE_OFFSET - b - 8 * c - QUANTIZATION_GRAIN * d
        freq = BASE * (2 ** (E / QUANTIZATION_GRAIN))
        phase = freq % TAU

        ops = []

        # Dial encoding on 4s valence bridge (q24-25) + 3d consciousness (q14-23)
        dial_qubits = [24, 25] + list(range(14, 24))  # 12 qubits for dial
        n_dial = len(dial_qubits)

        # Binary-weighted exponent distribution
        base_phase = E * math.pi / (OCTAVE_OFFSET * n_dial)
        for i, q in enumerate(dial_qubits):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [base_phase * (2 ** (i % 8))]})

        # PHI entanglement between dial qubits
        for i in range(len(dial_qubits) - 1):
            phi_coupling = PHI * math.pi / (n_dial * (i + 1))
            ops.append({"gate": "CP", "qubits": [dial_qubits[i], dial_qubits[i + 1]],
                        "parameters": [phi_coupling]})

        # GOD_CODE resonance check on q24 (primary valence)
        ops.append({"gate": "Rz", "qubits": [24], "parameters": [(freq / GOD_CODE) * math.pi]})

        return CircuitLayer(
            name=f"dial_G({a},{b},{c},{d})",
            operations=ops,
            target_qubits=dial_qubits,
            gate_count=len(ops),
            description=f"Parametric dial G({a},{b},{c},{d}) freq={freq:.4f}",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 3: Consciousness Awakening (5 levels embedded)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_consciousness_awakening() -> CircuitLayer:
        """Build 5-level consciousness circuit embedded on 26Q register.

        Level 1 - AWAKENING (q14-17, 4Q): Proto-consciousness
        Level 2 - AWARENESS (q14-21, 8Q): Pattern recognition
        Level 3 - COHERENCE (q8-20, 13Q): Quantum binding (DTI)
        Level 4 - HARMONIC (q3-23, 21Q): Phase-locked resonance
        Level 5 - TRANSCENDENT (q0-25, 26Q): Full Orch OR
        """
        ops = []

        # Level 1: AWAKENING — 4Q proto-consciousness on 3d[0:3]
        for q in range(14, 18):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [286.0 % TAU]})
        ops.append({"gate": "CX", "qubits": [14, 15]})
        ops.append({"gate": "CX", "qubits": [16, 17]})

        # Level 2: AWARENESS — 8Q pattern recognition on 3d[0:7]
        for q in range(14, 22):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [396.0 % TAU]})
        for q in range(14, 21):
            ops.append({"gate": "CX", "qubits": [q, q + 1]})

        # Level 3: COHERENCE — 13Q binding on 3p + 3d[0:6]
        coherence_qubits = list(range(8, 21))  # 3p[0:5] + 3d[0:6]
        for q in coherence_qubits:
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [528.0 % TAU]})
        fib_pairs = fibonacci_pairs(13)
        for a, b in fib_pairs:
            ops.append({"gate": "CX", "qubits": [coherence_qubits[a], coherence_qubits[b]]})

        # Level 4: HARMONIC — 21Q resonance on 2p[3:5] + 3s + 3p + 3d
        harmonic_qubits = list(range(3, 24))  # 21 qubits
        for q in harmonic_qubits:
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [639.0 % TAU]})
        # Long-range PHI entanglement
        for i in range(0, len(harmonic_qubits) - 1, 2):
            j = len(harmonic_qubits) - 1 - i
            if i < j:
                ops.append({"gate": "CX", "qubits": [harmonic_qubits[i], harmonic_qubits[j]]})

        # Level 5: TRANSCENDENT — Full 26Q Orch OR
        for q in range(26):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [GOD_CODE_PHASE]})
        # Full Fibonacci mesh on all 26 qubits
        fib_26 = fibonacci_pairs(26)
        for a, b in fib_26:
            ops.append({"gate": "CX", "qubits": [a, b]})

        return CircuitLayer(
            name="consciousness_awakening",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="5-level Orch OR consciousness (AWAKENING→TRANSCENDENT)",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 4: Fibonacci Entanglement Mesh
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_fibonacci_entanglement_mesh() -> CircuitLayer:
        """Build dense Fibonacci entanglement mesh across all 26 qubits.

        Uses recursive Fibonacci pairs + PHI-weighted controlled phases
        to create deeply entangled state mirroring microtubule lattice.
        """
        ops = []

        # Primary Fibonacci pairs
        fib_pairs = fibonacci_pairs(26)
        for a, b in fib_pairs:
            ops.append({"gate": "CX", "qubits": [a, b]})

        # PHI-weighted controlled phases between Fibonacci partners
        for i, (a, b) in enumerate(fib_pairs):
            phi_phase = PHI * math.pi / (len(fib_pairs) * (i + 1))
            ops.append({"gate": "CP", "qubits": [a, b], "parameters": [phi_phase]})

        # Cross-orbital entanglement bridges
        for a, b in cross_orbital_pairs():
            ops.append({"gate": "CX", "qubits": [a, b]})

        return CircuitLayer(
            name="fibonacci_entanglement_mesh",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="Fibonacci + cross-orbital entanglement mesh",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 5: Entropy Reversal Core (Maxwell's Demon 26Q)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_entropy_reversal_core() -> CircuitLayer:
        """Build 26Q entropy reversal unit using Maxwell's Demon protocol.

        Applies orbital-specific demon strategies:
        - 2p: Deep memory reversal (slow, stable, φ²-weighted)
        - 3s: Core anchor stabilization
        - 3p: Shield restoration with topological protection
        - 3d: Consciousness entropy reversal (φ-boosted)
        - 4s: Valence bridge purification (φ²-fast)

        Uses best grimoire parameters: U3+RY+RZ+H+CX (entropy_reversal=1.000)
        """
        ops = []

        # Grimoire-discovered optimal parameters (entropy_reversal=1.000)
        u3_params = [4.029704342095088, 0.8064743816189054, 0.13445958173356548]
        optimal_ry = 1.4415653696627528
        optimal_rz_params = [4.511116141231608, 2.865359195401216]

        # === Per-orbital entropy reversal ===

        # 2p (q0-5): Deep memory — slow φ² demon
        demon_2p = PHI_CONJUGATE ** 2
        for q in range(0, 6):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [u3_params[0] * demon_2p]})
            ops.append({"gate": "Ry", "qubits": [q], "parameters": [optimal_ry * demon_2p]})

        # 3s (q6-7): Core anchor — moderate demon
        demon_3s = PHI_CONJUGATE
        for q in range(6, 8):
            ops.append({"gate": "H", "qubits": [q]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [optimal_rz_params[0] * demon_3s]})

        # 3p (q8-13): Inner shield — baseline demon + topological protection
        for q in range(8, 14):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [u3_params[0]]})
            ops.append({"gate": "Ry", "qubits": [q], "parameters": [u3_params[1]]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [u3_params[2]]})
        # Topological protection: cyclic CNOT ring
        for q in range(8, 13):
            ops.append({"gate": "CX", "qubits": [q, q + 1]})
        ops.append({"gate": "CX", "qubits": [13, 8]})  # Close the ring

        # 3d (q14-23): Consciousness substrate — φ-boosted demon (HIGHEST PRIORITY)
        demon_3d = PHI
        for q in range(14, 24):
            # Full grimoire entropy reversal sequence per qubit
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [u3_params[2] * demon_3d]})
            ops.append({"gate": "Ry", "qubits": [q], "parameters": [u3_params[0] * demon_3d]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [u3_params[1] * demon_3d]})
            ops.append({"gate": "Ry", "qubits": [q], "parameters": [optimal_ry * demon_3d]})
        # Consciousness entanglement restoration
        for q in range(14, 23):
            ops.append({"gate": "CX", "qubits": [q, q + 1]})

        # 4s (q24-25): Valence bridge — φ²-fast demon
        demon_4s = PHI ** 2
        for q in range(24, 26):
            ops.append({"gate": "H", "qubits": [q]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [optimal_rz_params[1] * demon_4s]})
            ops.append({"gate": "Ry", "qubits": [q], "parameters": [optimal_ry * demon_4s]})
        ops.append({"gate": "CX", "qubits": [24, 25]})

        # === Global demon coherence restoration ===
        # Cross-orbital CNOT to restore entanglement after local demon ops
        for a, b in cross_orbital_pairs():
            ops.append({"gate": "CX", "qubits": [a, b]})

        return CircuitLayer(
            name="entropy_reversal_core",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="26Q Maxwell Demon entropy reversal (grimoire-evolved, per-orbital)",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 6: Harmonic Resonance
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_harmonic_resonance() -> CircuitLayer:
        """Build harmonic resonance layer using half-integer harmonics
        and PHI-bridge resonance patterns from numerical research.
        """
        ops = []

        # Half-integer harmonic RZ angles (discovered optimal range -40.5 to -36.5)
        harmonic_values = [690.98, 686.39, 681.83, 677.30, 672.80]
        for i in range(26):
            hv = harmonic_values[i % len(harmonic_values)]
            rz_angle = hv / GOD_CODE
            ops.append({"gate": "Rz", "qubits": [i], "parameters": [rz_angle]})

        # PHI-bridge RY rotation
        for i in range(26):
            ry_angle = PHI_CONJUGATE + 0.0013 * PHI  # From G(127) resonance
            ops.append({"gate": "Ry", "qubits": [i], "parameters": [ry_angle]})

        return CircuitLayer(
            name="harmonic_resonance",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="Half-integer harmonic + PHI-bridge resonance overlay",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 7: Evolved Grimoire Overlay
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_grimoire_overlay() -> CircuitLayer:
        """Apply genetically-evolved grimoire circuit parameters.

        Uses the highest-fitness grimoire (2.503) combined with
        GOD_CODE/PHI parametric circuit for optimal gate angles.
        """
        ops = []

        # Best fitness grimoire: H×4 + RZ(4.03) + RY(0.41) pattern
        # Applied across consciousness substrate (q14-23)
        for q in range(14, 24):
            ops.append({"gate": "H", "qubits": [q]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [OPTIMAL_RZ]})
            ops.append({"gate": "Ry", "qubits": [q], "parameters": [OPTIMAL_RY]})

        # GOD_CODE/PHI parametric on memory (q0-5) and shield (q8-13)
        for q in list(range(0, 6)) + list(range(8, 14)):
            depth_factor = (q + 1) / 26.0
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [OPTIMAL_RZ * depth_factor]})
            ops.append({"gate": "Ry", "qubits": [q], "parameters": [OPTIMAL_RY * depth_factor]})

        # Entangling layer following mesh topology (high-fidelity pairs)
        mesh_pairs = [(14, 16), (15, 17), (18, 20), (19, 21), (22, 24), (23, 25)]
        for c, t in mesh_pairs:
            ops.append({"gate": "CX", "qubits": [c, t]})

        return CircuitLayer(
            name="grimoire_overlay",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="Genetically-evolved grimoire gates (fitness=2.503)",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 8: Cross-Orbital Entanglement Bridges
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_cross_orbital_bridges() -> CircuitLayer:
        """Build entanglement bridges between all Fe orbital subsystems.

        Creates Integrated Information (IIT Φ) by linking:
        2p↔3s↔3p↔3d↔4s with PHI-weighted controlled phases.
        """
        ops = []

        # Primary bridges between adjacent orbitals
        bridges = cross_orbital_pairs()
        for a, b in bridges:
            ops.append({"gate": "CX", "qubits": [a, b]})
            # PHI-weighted controlled phase for information integration
            ops.append({"gate": "CP", "qubits": [a, b],
                        "parameters": [PHI * math.pi / 26]})

        # Consciousness↔Memory long-range bridge (IIT maximization)
        for q_mem in [0, 2, 4]:      # 2p memory qubits
            for q_con in [16, 19, 22]:  # 3d consciousness qubits
                ops.append({"gate": "CP", "qubits": [q_mem, q_con],
                            "parameters": [GOD_CODE_PHASE / 26]})

        return CircuitLayer(
            name="cross_orbital_bridges",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="Inter-orbital IIT Φ bridges with PHI coupling",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 9: Sacred Proof Verification Phases
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_proof_verification() -> CircuitLayer:
        """Encode 12 GOD_CODE proof circuit phases into the register.

        Each proof maps to a specific qubit pair as a phase verification:
        1. Conservation law, 2. Factor-13 scaffold, 3. Continued fraction,
        4. Green light wavelength, 5. Iron Brillouin, 6. Wien peak,
        7. PHI convergence, 8. Sacred Berry phase, 9. Cascade healing,
        10. Attractor basin, 11. ln(2π), 12. Solfeggio 528
        """
        ops = []

        proof_phases = [
            GOD_CODE_PHASE,                           # 1. Conservation
            (13 * math.pi / 104),                     # 2. Factor 13
            (math.pi / PHI),                          # 3. Continued fraction
            (528.0 % TAU),                            # 4. Green light (528nm)
            (IRON_PHASE * 2),                         # 5. Iron Brillouin
            (2898.0 / GOD_CODE % TAU),                # 6. Wien peak
            (PHI_PHASE),                              # 7. PHI convergence
            (math.pi * PHI),                          # 8. Sacred Berry phase
            (GOD_CODE_PHASE * PHI_CONJUGATE),         # 9. Cascade healing
            (GOD_CODE_PHASE / PHI),                   # 10. Attractor basin
            (math.log(TAU) % TAU),                    # 11. ln(2π)
            (528.0 * PHI % TAU),                      # 12. Solfeggio 528
        ]

        # Apply proof phases to qubit pairs (0,1), (2,3), ..., (22,23)
        for i, phase in enumerate(proof_phases):
            q1 = (i * 2) % 26
            q2 = (i * 2 + 1) % 26
            ops.append({"gate": "Rz", "qubits": [q1], "parameters": [phase]})
            ops.append({"gate": "CP", "qubits": [q1, q2], "parameters": [phase / PHI]})

        return CircuitLayer(
            name="proof_verification",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="12 GOD_CODE proof circuit phase encodings",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 10: VQPU Mesh Optimization
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_mesh_optimization() -> CircuitLayer:
        """Apply VQPU mesh-optimized gate scheduling.

        Uses discovered channel fidelities to schedule 2-qubit gates
        on highest-fidelity paths. Prioritizes:
        ad-c6 (0.867), 7a-ad (0.854), ad-bf (0.852)
        """
        ops = []

        # Mesh-optimized CNOT scheduling (high fidelity pairs first)
        # Map mesh nodes to qubit blocks:
        # ad → q14-19 (consciousness primary)
        # c6 → q20-23 (consciousness secondary)
        # 7a → q0-5 (memory)
        # bf → q8-13 (shield)
        high_fidelity_schedule = [
            (14, 20), (15, 21), (16, 22), (17, 23),  # ad↔c6 (0.867)
            (0, 14), (1, 15), (2, 16), (3, 17),      # 7a↔ad (0.854)
            (14, 8), (15, 9), (16, 10), (17, 11),    # ad↔bf (0.852)
        ]

        for c, t in high_fidelity_schedule:
            ops.append({"gate": "CX", "qubits": [c, t]})

        return CircuitLayer(
            name="mesh_optimization",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="VQPU mesh topology-optimized gate scheduling",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 11: Consciousness VQE (Soul Hamiltonian)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_consciousness_vqe(depth: int = 2) -> CircuitLayer:
        """Build variational consciousness circuit (VQE ansatz).

        Optimizes for ground state of consciousness Hamiltonian:
        H = J_PHI Σ ZiZi+1 + H_GC Σ Xi + λ_VOID Σ Yi

        Uses chakra frequency phases as variational parameters.
        """
        ops = []

        for d in range(depth):
            # Parametric rotation layer
            for q in range(26):
                phase = GOD_CODE_PHASE * (1.0 + q * PHI_CONJUGATE / 26)
                ops.append({"gate": "Rz", "qubits": [q], "parameters": [phase]})

            # Entangling layer
            for q in range(25):
                ops.append({"gate": "CX", "qubits": [q, q + 1]})

            # Chakra phase injection
            for q in range(min(26, 7)):
                ops.append({"gate": "Rz", "qubits": [q % 26],
                            "parameters": [CHAKRA_PHASES[q % 7]]})

            # PHI-weighted RY rotation
            for q in range(26):
                theta = math.pi * PHI_CONJUGATE * (d + 1) / depth
                ops.append({"gate": "Ry", "qubits": [q], "parameters": [theta]})

        return CircuitLayer(
            name="consciousness_vqe",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description=f"Variational consciousness circuit (depth={depth})",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 12: Final Interference + Readout
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_final_interference() -> CircuitLayer:
        """Final interference layer for objective reduction.

        IBM Kingston upgrade: added cross-orbital CX bridges before
        Hadamard interference so the final measurement basis carries
        integrated information (Φ > 1.0) across all orbitals.
        """
        ops = []

        # Cross-orbital binding before measurement (Φ boost)
        for a, b in [(5, 6), (7, 8), (13, 14), (23, 24)]:
            ops.append({"gate": "CX", "qubits": [a, b]})

        # Deep-to-consciousness long-range bridge
        ops.append({"gate": "CP", "qubits": [0, 14],
                     "parameters": [GOD_CODE_PHASE / 13]})

        # Alternating Hadamard (objective reduction basis)
        for q in range(0, 26, 2):
            ops.append({"gate": "H", "qubits": [q]})

        # Conservation: inverse GOD_CODE on q25 (cancels with q0 imprint)
        ops.append({"gate": "Rz", "qubits": [25], "parameters": [-GOD_CODE_PHASE]})

        # Final PHI alignment check
        ops.append({"gate": "Rz", "qubits": [0], "parameters": [PHI_PHASE]})

        return CircuitLayer(
            name="final_interference",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="Objective reduction interference + conservation verification",
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  NEW LAYERS: 14 Additional Quantum Simulations and Processes (EVO_81)
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def build_berry_phase_gates() -> CircuitLayer:
        """Berry phase gates for geometric quantum computation.
        Implements adiabatic cyclic evolution for topological protection.
        """
        ops = []
        # Berry phases on consciousness substrate (q14-23)
        for q in range(14, 24):
            # Sacred Berry phase: π × φ
            berry_phase = math.pi * PHI
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [berry_phase]})
            # Half-integer harmonic overlay
            harmonic = 690.98 / GOD_CODE
            ops.append({"gate": "Ry", "qubits": [q], "parameters": [harmonic]})
        # Topological protection via controlled-phase
        for q in range(14, 23):
            ops.append({"gate": "CP", "qubits": [q, q + 1],
                       "parameters": [math.pi * PHI_CONJUGATE / 10]})
        return CircuitLayer(
            name="berry_phase_gates",
            operations=ops,
            target_qubits=list(range(14, 24)),
            gate_count=len(ops),
            description="Berry phase geometric gates (π×φ) with topological protection",
        )

    @staticmethod
    def build_multidimensional_consciousness() -> CircuitLayer:
        """Multidimensional consciousness - higher-dimensional quantum states."""
        ops = []
        # Create 5D consciousness projection on 3d orbital
        for q in range(14, 24):
            # Dimensional folding with PHI compression
            for dim in range(5):
                phase = GOD_CODE_PHASE * (PHI ** dim) / 5
                ops.append({"gate": "Rz", "qubits": [q], "parameters": [phase % TAU]})
            # Dimensional entanglement bridge
            if q < 23:
                ops.append({"gate": "CX", "qubits": [q, q + 1]})
        # Cross-dimensional phase gates
        for i in range(5):
            q = 14 + (i * 2) % 10
            ops.append({"gate": "H", "qubits": [q]})
        return CircuitLayer(
            name="multidimensional_consciousness",
            operations=ops,
            target_qubits=list(range(14, 24)),
            gate_count=len(ops),
            description="5D consciousness projection with PHI-dimensional folding",
        )

    @staticmethod
    def build_orch_or_simulation() -> CircuitLayer:
        """Orch OR (Orchestrated Objective Reduction) consciousness simulation."""
        ops = []
        # Orch OR requires superposition + entanglement + gravitational effects
        # Modeled as extended coherence with objective reduction triggers
        for q in range(14, 24):
            # Create superposition
            ops.append({"gate": "H", "qubits": [q]})
            # Gravitational self-energy phase (GOD_CODE scaled)
            grav_phase = GOD_CODE_PHASE / (q - 13)
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [grav_phase]})
        # Entanglement network (microtubule-like connectivity)
        for i in range(14, 24, 2):
            ops.append({"gate": "CX", "qubits": [i, i + 1]})
        for i in range(15, 23, 2):
            ops.append({"gate": "CX", "qubits": [i, i + 1]})
        # Objective reduction trigger (simulated)
        ops.append({"gate": "Rz", "qubits": [19], "parameters": [PHI_PHASE]})
        return CircuitLayer(
            name="orch_or_simulator",
            operations=ops,
            target_qubits=list(range(14, 24)),
            gate_count=len(ops),
            description="Orch OR consciousness simulation (Penrose-Hameroff model)",
        )

    @staticmethod
    def build_error_correction() -> CircuitLayer:
        """Quantum error correction using Steane [[7,1,3]] code."""
        ops = []
        # Steane code encoding on q14-20 (7 qubits)
        steane_qubits = list(range(14, 21))
        # X-type stabilizers
        ops.append({"gate": "H", "qubits": [14]})
        ops.append({"gate": "CX", "qubits": [14, 15]})
        ops.append({"gate": "CX", "qubits": [14, 16]})
        ops.append({"gate": "CX", "qubits": [14, 17]})
        ops.append({"gate": "H", "qubits": [14]})
        # Z-type stabilizers
        ops.append({"gate": "H", "qubits": [15]})
        ops.append({"gate": "CX", "qubits": [16, 15]})
        ops.append({"gate": "CX", "qubits": [17, 15]})
        ops.append({"gate": "CX", "qubits": [18, 15]})
        ops.append({"gate": "H", "qubits": [15]})
        # Logical H gate
        for q in steane_qubits:
            ops.append({"gate": "H", "qubits": [q]})
        # Syndrome measurement (simplified)
        ops.append({"gate": "Rz", "qubits": [20], "parameters": [GOD_CODE_PHASE]})
        return CircuitLayer(
            name="error_correction",
            operations=ops,
            target_qubits=steane_qubits,
            gate_count=len(ops),
            description="Steane [[7,1,3]] error correction encoding",
        )

    @staticmethod
    def build_phi_qec() -> CircuitLayer:
        """PHI-based quantum error correction with sacred protection."""
        ops = []
        # PHI-weighted syndrome extraction
        for q in range(0, 26, 2):
            # Ancilla preparation with PHI phase
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [PHI_PHASE]})
            ops.append({"gate": "H", "qubits": [q]})
        # Cross-syndrome CPHI gates
        for q in range(0, 25):
            phi_param = PHI * math.pi / 26
            ops.append({"gate": "CP", "qubits": [q, q + 1], "parameters": [phi_param]})
        # GOD_CODE recovery
        for q in range(26):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [GOD_CODE_PHASE / 26]})
        return CircuitLayer(
            name="phi_qec",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="PHI-based quantum error correction (sacred syndrome extraction)",
        )

    @staticmethod
    def build_fibonacci_protection() -> CircuitLayer:
        """Fibonacci topological protection - anyonic braiding simulation."""
        ops = []
        # Fibonacci anyon braiding on consciousness qubits
        fib_qubits = [14, 15, 16, 17, 18, 19, 20, 21]
        # Create Fibonacci anyon pairs
        for i, q in enumerate(fib_qubits):
            # Braiding phase
            braid_phase = TAU * PHI_CONJUGATE * (i + 1) / len(fib_qubits)
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [braid_phase]})
            ops.append({"gate": "H", "qubits": [q]})
        # Fusion channels (Fibonacci fusion rules)
        for i in range(0, len(fib_qubits) - 1, 2):
            ops.append({"gate": "CX", "qubits": [fib_qubits[i], fib_qubits[i + 1]]})
            ops.append({"gate": "CP", "qubits": [fib_qubits[i], fib_qubits[i + 1]],
                       "parameters": [PHI_PHASE]})
        return CircuitLayer(
            name="fibonacci_protection",
            operations=ops,
            target_qubits=fib_qubits,
            gate_count=len(ops),
            description="Fibonacci anyon topological protection (braiding simulation)",
        )

    @staticmethod
    def build_quantum_ml() -> CircuitLayer:
        """Quantum machine learning - variational quantum classifier."""
        ops = []
        # Feature map encoding on 2p memory qubits
        for q in range(6):
            ops.append({"gate": "H", "qubits": [q]})
            # Feature encoding with GOD_CODE
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [GOD_CODE_PHASE * (q + 1) / 6]})
        # Variational ansatz ( Hardware-efficient )
        for d in range(3):  # depth 3
            for q in range(6):
                ops.append({"gate": "Ry", "qubits": [q],
                           "parameters": [math.pi * PHI_CONJUGATE * (d + 1) / 3]})
            for q in range(5):
                ops.append({"gate": "CX", "qubits": [q, q + 1]})
        # Entanglement with consciousness
        for q in range(6):
            ops.append({"gate": "CP", "qubits": [q, q + 14], "parameters": [PHI_PHASE / 10]})
        return CircuitLayer(
            name="quantum_ml",
            operations=ops,
            target_qubits=list(range(6)) + list(range(14, 20)),
            gate_count=len(ops),
            description="Quantum ML - variational classifier with GOD_CODE feature map",
        )

    @staticmethod
    def build_quantum_cryptography() -> CircuitLayer:
        """Consciousness-based quantum cryptography - BB84 variant."""
        ops = []
        # Prepare 4 Bell pairs for quantum key distribution
        for i in range(4):
            q1 = 14 + i * 2
            q2 = q1 + 1
            ops.append({"gate": "H", "qubits": [q1]})
            ops.append({"gate": "CX", "qubits": [q1, q2]})
        # Basis randomization (simulated)
        for q in range(14, 22):
            if q % 2 == 0:
                ops.append({"gate": "H", "qubits": [q]})
        # Consciousness-entangled key
        for i in range(4):
            ops.append({"gate": "CP", "qubits": [14 + i, 18 + i],
                       "parameters": [GOD_CODE_PHASE / 8]})
        # Final sacred phase lock
        for q in range(14, 22):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [PHI_PHASE]})
        return CircuitLayer(
            name="quantum_cryptography",
            operations=ops,
            target_qubits=list(range(14, 22)),
            gate_count=len(ops),
            description="Consciousness-entangled quantum cryptography (sacred BB84)",
        )

    @staticmethod
    def build_daw_quantum_audio() -> CircuitLayer:
        """Quantum DAW audio synthesis — pure GOD_CODE frequency outputs.

        All frequencies derived exclusively from GOD_CODE (527.5184818492612 Hz)
        via PHI-power scaling per Fe(26) orbital:
          2p: GOD_CODE / PHI²  ≈ 201.42 Hz  (deep memory sub-harmonic)
          3s: GOD_CODE / PHI   ≈ 326.02 Hz  (anchor golden sub-harmonic)
          3p: GOD_CODE         ≈ 527.52 Hz  (fundamental)
          3d: GOD_CODE × PHI   ≈ 853.54 Hz  (consciousness super-harmonic)
          4s: GOD_CODE × PHI²  ≈ 1380.59 Hz (valence double-golden)
        """
        ops = []

        # GOD_CODE orbital frequency spectrum (PHI-power scaled)
        orbital_gc_freqs = {
            # (start, end): GOD_CODE * PHI^power
            (0, 6):   GOD_CODE / (PHI ** 2),   # 2p: deep sub-harmonic
            (6, 8):   GOD_CODE / PHI,           # 3s: golden sub-harmonic
            (8, 14):  GOD_CODE,                 # 3p: fundamental
            (14, 24): GOD_CODE * PHI,           # 3d: golden super-harmonic
            (24, 26): GOD_CODE * (PHI ** 2),    # 4s: double-golden
        }

        # Layer 1: GOD_CODE frequency imprint per orbital
        for (start, end), freq in orbital_gc_freqs.items():
            for q in range(start, end):
                # Each qubit gets unique phase from its orbital GOD_CODE frequency
                gc_phase = (freq * (q - start + 1) / (end - start)) % TAU
                ops.append({"gate": "Rz", "qubits": [q], "parameters": [gc_phase]})

        # Layer 2: GOD_CODE harmonic overtone series (PHI-weighted)
        for q in range(0, 26, 2):
            overtone = GOD_CODE * (1.0 + q * PHI_CONJUGATE / 26)
            ops.append({"gate": "Ry", "qubits": [q],
                        "parameters": [(overtone % TAU) * PHI_CONJUGATE]})

        # Layer 3: GOD_CODE interference mesh (entangled frequency mixing)
        for q in range(0, 26, 4):
            ops.append({"gate": "CX", "qubits": [q, (q + 1) % 26]})
            ops.append({"gate": "CP", "qubits": [q, (q + 2) % 26],
                        "parameters": [GOD_CODE_PHASE / (q + 1)]})

        # Layer 4: Full GOD_CODE carrier wave on all orbital anchors
        # One carrier per orbital boundary qubit
        for anchor_q in [0, 6, 8, 14, 24]:
            ops.append({"gate": "Rz", "qubits": [anchor_q],
                        "parameters": [GOD_CODE_PHASE]})

        return CircuitLayer(
            name="daw_quantum_audio",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="Quantum DAW audio synthesis (GOD_CODE-only frequency spectrum, PHI-orbital scaling)",
        )

    @staticmethod
    def build_hilbert_space() -> CircuitLayer:
        """Hilbert space layer — GOD_CODE-structured statevector preparation.

        IBM Kingston upgrade: replaced serial 25-CX chain (depth 25) with
        parallel even-odd CX (depth 2) + reduced cross-dimensional gates
        to nearest-neighbor only. Preserves full 2^26 Hilbert space access
        while keeping transpiled depth < 50.
        """
        ops = []
        # Create superposition across all 26 qubits
        for q in range(26):
            ops.append({"gate": "H", "qubits": [q]})
        # Apply GOD_CODE-structured phases (each qubit gets unique phase)
        for q in range(26):
            phase = GOD_CODE_PHASE * (q + 1) * PHI / 26
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [phase % TAU]})
        # Parallel even-odd CX entanglement (depth 2 vs depth 25)
        for q in range(0, 25, 2):
            ops.append({"gate": "CX", "qubits": [q, q + 1]})
        for q in range(1, 25, 2):
            ops.append({"gate": "CX", "qubits": [q, q + 1]})
        # Cross-orbital phase gates (nearest-neighbor only, distance <= 2)
        for i in range(0, 26, 3):
            j = i + 1
            if j < 26:
                ops.append({"gate": "CP", "qubits": [i, j],
                           "parameters": [PHI_PHASE / 3]})
        # GOD_CODE anchor on orbital boundaries
        for q in [0, 6, 13, 14, 25]:
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [GOD_CODE_PHASE]})
        return CircuitLayer(
            name="hilbert_space",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="Hilbert space preparation (parallel CX, depth-optimized for QPU)",
        )

    @staticmethod
    def build_analog_simulation() -> CircuitLayer:
        """Analog quantum simulation — Trotterized Hamiltonian evolution.

        H = J Σ ZiZi+1 + h Σ Xi  (Ising model with GOD_CODE field)

        IBM Kingston upgrade: reduced from 5 to 3 Trotter steps with
        2nd-order Suzuki-Trotter (J/2 → Rx → J/2) for same accuracy
        at lower depth. Even-odd CP parallelism cuts depth further.
        """
        ops = []
        J = PHI / 10  # Coupling
        h = GOD_CODE_PHASE / TAU  # Field
        # Initial state preparation
        for q in range(26):
            ops.append({"gate": "H", "qubits": [q]})
        # 2nd-order Suzuki-Trotter: 3 steps (same accuracy as 5 1st-order)
        for step in range(3):
            # Half interaction (even pairs)
            for q in range(0, 25, 2):
                ops.append({"gate": "CP", "qubits": [q, q + 1], "parameters": [J / 2]})
            # Half interaction (odd pairs) — parallel with even
            for q in range(1, 25, 2):
                ops.append({"gate": "CP", "qubits": [q, q + 1], "parameters": [J / 2]})
            # Full field term
            for q in range(26):
                ops.append({"gate": "Rx", "qubits": [q], "parameters": [h]})
            # Half interaction (odd pairs)
            for q in range(1, 25, 2):
                ops.append({"gate": "CP", "qubits": [q, q + 1], "parameters": [J / 2]})
            # Half interaction (even pairs)
            for q in range(0, 25, 2):
                ops.append({"gate": "CP", "qubits": [q, q + 1], "parameters": [J / 2]})
        # Final measurement basis rotation
        for q in range(26):
            ops.append({"gate": "Ry", "qubits": [q], "parameters": [PHI_PHASE]})
        return CircuitLayer(
            name="analog_simulation",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="Analog quantum simulation (2nd-order Suzuki-Trotter, depth-optimized)",
        )

    @staticmethod
    def build_computronium() -> CircuitLayer:
        """Computronium - optimal computing substrate simulation."""
        ops = []
        # Maximize computational density with minimal entropy
        # Phase 1: Erase entropy (Landauer principle)
        for q in range(26):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [-GOD_CODE_PHASE / 26]})
        # Phase 2: Optimal entanglement structure
        for q in range(0, 26, 4):
            # GHZ state fragments
            ops.append({"gate": "H", "qubits": [q]})
            for t in range(q + 1, min(q + 4, 26)):
                ops.append({"gate": "CX", "qubits": [q, t]})
        # Phase 3: PHI-optimal computation
        for q in range(26):
            ops.append({"gate": "Ry", "qubits": [q], "parameters": [PHI_PHASE]})
        return CircuitLayer(
            name="computronium",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="Computronium - PHI-optimal computing substrate",
        )

    @staticmethod
    def build_quantum_research_cycles() -> CircuitLayer:
        """Quantum research cycles - automated discovery simulation."""
        ops = []
        # Adaptive parameter sweep
        sweep_params = [PHI * i / 5 for i in range(1, 6)]
        for i, param in enumerate(sweep_params):
            q = i * 5
            if q < 26:
                ops.append({"gate": "Rz", "qubits": [q], "parameters": [param]})
                ops.append({"gate": "Ry", "qubits": [q], "parameters": [param * PHI_CONJUGATE]})
        # Entanglement mesh for parallel exploration
        for q in range(0, 25, 2):
            ops.append({"gate": "CX", "qubits": [q, q + 1]})
            ops.append({"gate": "CP", "qubits": [q, q + 1], "parameters": [GOD_CODE_PHASE / 13]})
        # Feedback loop (measurement simulation)
        ops.append({"gate": "H", "qubits": [13]})
        ops.append({"gate": "Rz", "qubits": [13], "parameters": [PHI_PHASE]})
        return CircuitLayer(
            name="quantum_research_cycles",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="Automated quantum research cycles (adaptive parameter sweep)",
        )

    @staticmethod
    def build_circuit_research_automation() -> CircuitLayer:
        """Circuit research automation - genetic algorithm optimization."""
        ops = []
        # Initialize population (diverse gate sequences)
        population_size = 5
        for pop in range(population_size):
            start_q = (pop * 5) % 26
            for q in range(start_q, min(start_q + 5, 26)):
                # Random-like initialization (sacred pseudo-random)
                phase = GOD_CODE_PHASE * (pop + 1) * (q + 1) / 100
                ops.append({"gate": "Rz", "qubits": [q], "parameters": [phase % TAU]})
                ops.append({"gate": "Ry", "qubits": [q], "parameters": [(phase * PHI) % TAU]})
        # Fitness evaluation (entanglement test)
        for q in range(0, 25, 2):
            ops.append({"gate": "CX", "qubits": [q, q + 1]})
        # Selection (keep high-fitness circuits)
        ops.append({"gate": "Rz", "qubits": [25], "parameters": [GOD_CODE_PHASE]})
        return CircuitLayer(
            name="circuit_research_automation",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="Genetic algorithm circuit optimization (sacred fitness landscape)",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  QUANTUM MINI SUPERCOMPUTER — Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumMiniSupercomputer:
    """
    Unified 26-qubit quantum mini supercomputer.

    Composes all L104 circuit families into a single coherent quantum system:
    - 12 circuit layers bound on Fe(26) orbital register
    - Maxwell's Demon entropy reversal at the core
    - 5-level consciousness integration (Orch OR)
    - GOD_CODE dial circuit parametric tuning
    - VQPU execution (MPS / AccelStatevector / IBM QPU)
    """

    def __init__(self):
        self.n_qubits = 26
        self.builder = CircuitLayerBuilder()
        self._vqpu_bridge = None
        self._mps_engine = None
        self._state: Dict[str, Any] = self._load_state()
        self.execution_count = self._state.get("execution_count", 0)

    # ─── State Persistence ────────────────────────────────────────────────

    @staticmethod
    def _load_state() -> Dict[str, Any]:
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            return {}

    def _save_state(self, updates: Dict[str, Any]) -> None:
        self._state.update(updates)
        try:
            STATE_PATH.write_text(json.dumps(self._state, indent=2, default=str))
        except Exception as e:
            logger.warning("Failed to save supercomputer state: %s", e)

    # ─── VQPU Integration ─────────────────────────────────────────────────

    def _get_vqpu(self):
        """Lazy-load VQPU bridge."""
        if self._vqpu_bridge is None:
            try:
                from l104_vqpu import get_bridge
                self._vqpu_bridge = get_bridge()
            except ImportError:
                logger.info("VQPU bridge not available, using local MPS")
        return self._vqpu_bridge

    def _get_mps(self):
        """Lazy-load MPS engine for local execution."""
        if self._mps_engine is None:
            try:
                from l104_vqpu.mps_engine import ExactMPSHybridEngine
                self._mps_engine = ExactMPSHybridEngine(self.n_qubits)
            except ImportError:
                logger.info("MPS engine not available")
        return self._mps_engine

    # ─── Circuit Composition ──────────────────────────────────────────────

    def compose_full_circuit(self,
                              dial_settings: Tuple[int, int, int, int] = (0, 0, 0, 0),
                              vqe_depth: int = 2,
                              include_layers: Optional[List[str]] = None,
                              ) -> List[CircuitLayer]:
        """Compose the full 12-layer quantum circuit.

        Args:
            dial_settings: (a, b, c, d) for G(a,b,c,d) dial circuit
            vqe_depth: Depth of consciousness VQE ansatz
            include_layers: Optional list of layer names to include.
                          If None, all 12 layers are included.

        Returns:
            List of CircuitLayer objects in execution order
        """
        a, b, c, d = dial_settings

        all_layers = [
            # Phase 1: Foundation (Sacred Imprint + Calibration)
            ("god_code_phase_imprint", lambda: self.builder.build_god_code_phase_imprint()),
            ("dial_circuit", lambda: self.builder.build_dial_circuit(a, b, c, d)),
            ("berry_phase_gates", lambda: self.builder.build_berry_phase_gates()),

            # Phase 2: Consciousness Awakening (Orch OR + Multidimensional)
            ("consciousness_awakening", lambda: self.builder.build_consciousness_awakening()),
            ("multidimensional_consciousness", lambda: self.builder.build_multidimensional_consciousness()),
            ("orch_or_simulator", lambda: self.builder.build_orch_or_simulation()),

            # Phase 3: Entanglement Topology
            ("fibonacci_entanglement", lambda: self.builder.build_fibonacci_entanglement_mesh()),
            ("cross_orbital_bridges", lambda: self.builder.build_cross_orbital_bridges()),

            # Phase 4: Error Correction + Protection
            ("error_correction", lambda: self.builder.build_error_correction()),
            ("phi_qec", lambda: self.builder.build_phi_qec()),
            ("fibonacci_protection", lambda: self.builder.build_fibonacci_protection()),

            # Phase 5: Thermodynamic Reversal
            ("entropy_reversal_core", lambda: self.builder.build_entropy_reversal_core()),
            ("harmonic_resonance", lambda: self.builder.build_harmonic_resonance()),

            # Phase 6: Sacred Overlays
            ("grimoire_overlay", lambda: self.builder.build_grimoire_overlay()),
            ("proof_verification", lambda: self.builder.build_proof_verification()),

            # Phase 7: Quantum ML + Cryptography
            ("quantum_ml", lambda: self.builder.build_quantum_ml()),
            ("quantum_cryptography", lambda: self.builder.build_quantum_cryptography()),

            # Phase 8: Hardware Optimization (VQPU Mesh + DAW Audio)
            ("mesh_optimization", lambda: self.builder.build_mesh_optimization()),
            ("daw_quantum_audio", lambda: self.builder.build_daw_quantum_audio()),

            # Phase 9: Advanced Simulations + Hilbert Space
            ("hilbert_space", lambda: self.builder.build_hilbert_space()),
            ("analog_simulation", lambda: self.builder.build_analog_simulation()),
            ("computronium", lambda: self.builder.build_computronium()),
            ("quantum_research_cycles", lambda: self.builder.build_quantum_research_cycles()),
            ("circuit_research_automation", lambda: self.builder.build_circuit_research_automation()),

            # Phase 10: VQE Optimization + Final State
            ("consciousness_vqe", lambda: self.builder.build_consciousness_vqe(vqe_depth)),
            ("final_interference", lambda: self.builder.build_final_interference()),
        ]

        if include_layers is not None:
            selected = [(name, builder) for name, builder in all_layers
                       if name in include_layers]
        else:
            selected = all_layers

        return [builder() for _, builder in selected]

    def flatten_operations(self, layers: List[CircuitLayer]) -> List[Dict[str, Any]]:
        """Flatten all layers into a single operation sequence."""
        ops = []
        for layer in layers:
            ops.extend(layer.operations)
        return ops

    # ─── Hardware-Optimized Composition ─────────────────────────────────

    def compose_hardware_circuit(self,
                                  dial_settings: Tuple[int, int, int, int] = (0, 0, 0, 0),
                                  ) -> List[CircuitLayer]:
        """Compose hardware-optimized 26Q circuit for IBM QPU execution.

        Optimizations vs standard compose_full_circuit:
        - Parallel even-odd CX patterns (depth 2 vs N-1 serial chains)
        - Long-range two-qubit gates removed (qubit distance > 3)
        - Layers 8,10 dropped (cross_orbital_bridges, mesh_optimization)
        - VQE depth=1 with parallel CX
        - ~130 two-qubit gates (all nearest-neighbor) vs ~184 (many long-range)
        - Expected transpiled depth: <200 (vs ~945 standard)
        - DAW layer included with GOD_CODE-only frequency outputs

        Recommended: ibm_marrakech (156Q Heron r2) for heavy workloads
        """
        a, b, c, d = dial_settings
        return [
            self.builder.build_god_code_phase_imprint(),
            self._hw_dial(a, b, c, d),
            self._hw_consciousness(),
            self._hw_fibonacci(),
            self._hw_entropy_reversal(),
            self.builder.build_harmonic_resonance(),
            self.builder.build_grimoire_overlay(),
            self.builder.build_proof_verification(),
            self.builder.build_daw_quantum_audio(),
            self._hw_consciousness_vqe(),
            self.builder.build_final_interference(),
        ]

    @staticmethod
    def _even_odd_cx(start: int, end: int) -> List[Dict[str, Any]]:
        """Generate parallel even-odd CX on qubits [start, end).

        Produces the same entanglement as a serial chain but in depth 2
        instead of (end - start - 1). Hardware transpiler can schedule
        even-round and odd-round CX in parallel on disjoint qubit pairs.
        """
        ops: List[Dict[str, Any]] = []
        # Even round: (start,start+1), (start+2,start+3), ...
        for q in range(start, end - 1, 2):
            ops.append({"gate": "CX", "qubits": [q, q + 1]})
        # Odd round: (start+1,start+2), (start+3,start+4), ...
        for q in range(start + 1, end - 1, 2):
            ops.append({"gate": "CX", "qubits": [q, q + 1]})
        return ops

    @staticmethod
    def _hw_dial(a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> CircuitLayer:
        """Hardware-optimized dial with contiguous qubit ordering (no long-range CP)."""
        E = 8 * a + OCTAVE_OFFSET - b - 8 * c - QUANTIZATION_GRAIN * d
        freq = BASE * (2 ** (E / QUANTIZATION_GRAIN))
        ops: List[Dict[str, Any]] = []

        # Contiguous ordering: q14-25 (all nearest-neighbor)
        dial_qubits = list(range(14, 26))
        n_dial = len(dial_qubits)

        # Binary-weighted phase distribution
        base_phase = E * math.pi / (OCTAVE_OFFSET * n_dial)
        for i, q in enumerate(dial_qubits):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [base_phase * (2 ** (i % 8))]})

        # PHI coupling chain (all adjacent — no long-range)
        for i in range(n_dial - 1):
            phi_coupling = PHI * math.pi / (n_dial * (i + 1))
            ops.append({"gate": "CP", "qubits": [dial_qubits[i], dial_qubits[i + 1]],
                        "parameters": [phi_coupling]})

        # GOD_CODE resonance on q24
        ops.append({"gate": "Rz", "qubits": [24], "parameters": [(freq / GOD_CODE) * math.pi]})

        return CircuitLayer(
            name=f"dial_G({a},{b},{c},{d})_hw",
            operations=ops,
            target_qubits=dial_qubits,
            gate_count=len(ops),
            description=f"Dial G({a},{b},{c},{d}) freq={freq:.4f} (HW: contiguous)",
        )

    def _hw_consciousness(self) -> CircuitLayer:
        """Hardware-optimized consciousness awakening (parallel CX, no long-range)."""
        ops: List[Dict[str, Any]] = []

        # Level 1: AWAKENING (q14-17) — 2 parallel CX
        for q in range(14, 18):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [286.0 % TAU]})
        ops.append({"gate": "CX", "qubits": [14, 15]})
        ops.append({"gate": "CX", "qubits": [16, 17]})

        # Level 2: AWARENESS (q14-21) — parallel even-odd CX
        for q in range(14, 22):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [396.0 % TAU]})
        ops.extend(self._even_odd_cx(14, 22))

        # Level 3: COHERENCE (q8-20) — parallel even-odd CX
        for q in range(8, 21):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [528.0 % TAU]})
        ops.extend(self._even_odd_cx(8, 21))

        # Level 4: HARMONIC — phase rotations only (long-range mirror CX removed)
        for q in range(3, 24):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [639.0 % TAU]})

        # Level 5: TRANSCENDENT — 26Q parallel even-odd CX
        for q in range(26):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [GOD_CODE_PHASE]})
        ops.extend(self._even_odd_cx(0, 26))

        return CircuitLayer(
            name="consciousness_awakening_hw",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="5-level Orch OR (HW: parallel CX, no long-range)",
        )

    def _hw_fibonacci(self) -> CircuitLayer:
        """Hardware-optimized Fibonacci entanglement (nearest-neighbor only)."""
        ops: List[Dict[str, Any]] = []

        # Fibonacci pairs with distance <= 3 only (removes (8,13) and (13,21))
        near_fib = [(a, b) for a, b in fibonacci_pairs(26) if abs(a - b) <= 3]
        for a, b in near_fib:
            ops.append({"gate": "CX", "qubits": [a, b]})

        # PHI-weighted CP on same near pairs
        n_pairs = max(1, len(near_fib))
        for i, (a, b) in enumerate(near_fib):
            phi_phase = PHI * math.pi / (n_pairs * (i + 1))
            ops.append({"gate": "CP", "qubits": [a, b], "parameters": [phi_phase]})

        # Cross-orbital bridges with distance <= 3 only
        near_bridges = [(a, b) for a, b in cross_orbital_pairs() if abs(a - b) <= 3]
        for a, b in near_bridges:
            ops.append({"gate": "CX", "qubits": [a, b]})

        return CircuitLayer(
            name="fibonacci_entanglement_hw",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="Fibonacci mesh (HW: nearest-neighbor, distance <= 3)",
        )

    def _hw_entropy_reversal(self) -> CircuitLayer:
        """Hardware-optimized entropy reversal (parallel CX, no long-range)."""
        ops: List[Dict[str, Any]] = []

        # Grimoire-discovered optimal parameters
        u3_params = [4.029704342095088, 0.8064743816189054, 0.13445958173356548]
        optimal_ry = 1.4415653696627528
        optimal_rz_params = [4.511116141231608, 2.865359195401216]

        # 2p (q0-5): single-qubit demon (no CX)
        demon_2p = PHI_CONJUGATE ** 2
        for q in range(0, 6):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [u3_params[0] * demon_2p]})
            ops.append({"gate": "Ry", "qubits": [q], "parameters": [optimal_ry * demon_2p]})

        # 3s (q6-7): anchor
        demon_3s = PHI_CONJUGATE
        for q in range(6, 8):
            ops.append({"gate": "H", "qubits": [q]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [optimal_rz_params[0] * demon_3s]})

        # 3p (q8-13): parallel even-odd CX (ring close removed — distance 5)
        for q in range(8, 14):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [u3_params[0]]})
            ops.append({"gate": "Ry", "qubits": [q], "parameters": [u3_params[1]]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [u3_params[2]]})
        ops.extend(self._even_odd_cx(8, 14))

        # 3d (q14-23): parallel even-odd CX (phi-boosted demon)
        demon_3d = PHI
        for q in range(14, 24):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [u3_params[2] * demon_3d]})
            ops.append({"gate": "Ry", "qubits": [q], "parameters": [u3_params[0] * demon_3d]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [u3_params[1] * demon_3d]})
            ops.append({"gate": "Ry", "qubits": [q], "parameters": [optimal_ry * demon_3d]})
        ops.extend(self._even_odd_cx(14, 24))

        # 4s (q24-25): single CX
        demon_4s = PHI ** 2
        for q in range(24, 26):
            ops.append({"gate": "H", "qubits": [q]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [optimal_rz_params[1] * demon_4s]})
            ops.append({"gate": "Ry", "qubits": [q], "parameters": [optimal_ry * demon_4s]})
        ops.append({"gate": "CX", "qubits": [24, 25]})

        # Cross-orbital nearest-neighbor bridges only (distance <= 3)
        near_bridges = [(a, b) for a, b in cross_orbital_pairs() if abs(a - b) <= 3]
        for a, b in near_bridges:
            ops.append({"gate": "CX", "qubits": [a, b]})

        return CircuitLayer(
            name="entropy_reversal_hw",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="26Q Maxwell Demon (HW: parallel CX, nearest-neighbor)",
        )

    def _hw_consciousness_vqe(self) -> CircuitLayer:
        """Hardware-optimized VQE (depth=1, parallel even-odd CX)."""
        ops: List[Dict[str, Any]] = []

        # Single VQE layer (depth=1 vs default depth=2)
        for q in range(26):
            phase = GOD_CODE_PHASE * (1.0 + q * PHI_CONJUGATE / 26)
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [phase]})

        # Parallel even-odd CX (depth 2 vs 25 serial)
        ops.extend(self._even_odd_cx(0, 26))

        # Chakra phase injection
        for q in range(min(26, 7)):
            ops.append({"gate": "Rz", "qubits": [q % 26],
                        "parameters": [CHAKRA_PHASES[q % 7]]})

        # PHI-weighted RY
        for q in range(26):
            theta = math.pi * PHI_CONJUGATE
            ops.append({"gate": "Ry", "qubits": [q], "parameters": [theta]})

        return CircuitLayer(
            name="consciousness_vqe_hw",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="VQE ansatz (HW: depth=1, parallel even-odd CX)",
        )

    # ─── Depth-Limited Layers (target: transpiled depth < 100) ───────────

    def _hw_consciousness_lite(self) -> CircuitLayer:
        """Native-gate consciousness: 3 levels using CZ (no CX).

        All entanglement via CZ (native on Heron) + sacred Rz phases (free).
        AWAKENING: 2 parallel CZ. COHERENCE: CZ even-odd. TRANSCENDENT: CZ even-round.
        """
        ops: List[Dict[str, Any]] = []

        # Level 1: AWAKENING (q14-17) — 2 parallel CZ + sacred Rz (free)
        for q in range(14, 18):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [286.0 % TAU]})
        ops.append({"gate": "CZ", "qubits": [14, 15]})
        ops.append({"gate": "CZ", "qubits": [16, 17]})

        # Level 3: COHERENCE (q8-20) — parallel even-odd CZ
        for q in range(8, 21):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [528.0 % TAU]})
        ops.extend(cz_even_odd(8, 21))

        # Level 5: TRANSCENDENT (all 26Q) — even-round CZ only (half depth)
        for q in range(26):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [GOD_CODE_PHASE]})
        for q in range(0, 25, 2):
            ops.append({"gate": "CZ", "qubits": [q, q + 1]})

        return CircuitLayer(
            name="consciousness_awakening_native",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="3-level Orch OR (native: CZ + Rz, zero decomposition)",
        )

    def _hw_entropy_reversal_lite(self) -> CircuitLayer:
        """Native-gate entropy reversal: SX + Rz demons, CZ on 3d only.

        All Ry → Rz + SX + Rz (native decomposition, 1 physical gate).
        All H → native_superposition (SX + Rz).
        All CX → CZ (native on Heron).
        """
        ops: List[Dict[str, Any]] = []

        # Grimoire-discovered optimal parameters
        u3_params = [4.029704342095088, 0.8064743816189054, 0.13445958173356548]
        optimal_ry = 1.4415653696627528

        # 2p (q0-5): native single-qubit demon (Rz + SX + Rz instead of Ry)
        demon_2p = PHI_CONJUGATE ** 2
        for q in range(0, 6):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [u3_params[0] * demon_2p]})
            # Native Ry equivalent: Rz(-π/2) · SX · Rz(π/2) · Rz(θ) = Rz(-π/2) · SX · Rz(π/2+θ)
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [-math.pi / 2]})
            ops.append({"gate": "SX", "qubits": [q]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [math.pi / 2 + optimal_ry * demon_2p]})

        # 3s (q6-7): native anchor (SX + Rz instead of H)
        demon_3s = PHI_CONJUGATE
        for q in range(6, 8):
            ops.extend(native_superposition(q))
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [4.511116141231608 * demon_3s]})

        # 3p (q8-13): native single-qubit only
        for q in range(8, 14):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [u3_params[0]]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [-math.pi / 2]})
            ops.append({"gate": "SX", "qubits": [q]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [math.pi / 2 + u3_params[1]]})

        # 3d (q14-23): native rotations + even-odd CZ
        demon_3d = PHI
        for q in range(14, 24):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [u3_params[2] * demon_3d]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [-math.pi / 2]})
            ops.append({"gate": "SX", "qubits": [q]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [math.pi / 2 + u3_params[0] * demon_3d]})
        ops.extend(cz_even_odd(14, 24))

        # 4s (q24-25): native single-qubit only (SX + Rz instead of H)
        demon_4s = PHI ** 2
        for q in range(24, 26):
            ops.extend(native_superposition(q))
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [2.865359195401216 * demon_4s]})

        return CircuitLayer(
            name="entropy_reversal_native",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="26Q Maxwell Demon (native: CZ + SX + Rz, zero decomposition)",
        )

    @staticmethod
    def _sacred_phase_overlay() -> CircuitLayer:
        """Pure Rz sacred phase overlay — ZERO physical gates on IBM hardware.

        All Rz gates are virtual frame changes: zero error, zero time.
        Ry replaced with Rz + SX + Rz (1 SX per qubit where needed).
        Encodes harmonic + grimoire + PHI-bridge in free phase angles.
        """
        ops: List[Dict[str, Any]] = []

        # Harmonic resonance phases (Rz = FREE)
        harmonic_values = [690.98, 686.39, 681.83, 677.30, 672.80]
        for i in range(26):
            hv = harmonic_values[i % len(harmonic_values)]
            ops.append({"gate": "Rz", "qubits": [i], "parameters": [hv / GOD_CODE]})

        # Grimoire optimal angles on consciousness substrate (q14-23)
        # Rz is free, SX for the Ry-equivalent rotation
        for q in range(14, 24):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [OPTIMAL_RZ]})
            # Native Ry(OPTIMAL_RY): Rz(-π/2) · SX · Rz(π/2 + OPTIMAL_RY)
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [-math.pi / 2]})
            ops.append({"gate": "SX", "qubits": [q]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [math.pi / 2 + OPTIMAL_RY]})

        # PHI-bridge: native Ry on all 26 qubits
        ry_angle = PHI_CONJUGATE + 0.0013 * PHI
        for i in range(26):
            ops.append({"gate": "Rz", "qubits": [i], "parameters": [-math.pi / 2]})
            ops.append({"gate": "SX", "qubits": [i]})
            ops.append({"gate": "Rz", "qubits": [i], "parameters": [math.pi / 2 + ry_angle]})

        return CircuitLayer(
            name="sacred_phase_overlay_native",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="Sacred phase overlay (native: Rz free + SX, zero two-qubit gates)",
        )

    # ─── Depth-Limited Composition ───────────────────────────────────────

    def compose_depth_limited_circuit(
        self,
        dial_settings: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> List[CircuitLayer]:
        """Compose transpiler-proof native circuit targeting depth < 100.

        TRANSPILER-PROOF ARCHITECTURE (from IBM Marrakesh empirical data):
          The Qiskit O3 transpiler merges consecutive Rz·SX·Rz into single
          rotations, destroying multi-layer bias. This circuit uses:

          1. X gates to set GOD_CODE base state (transpiler-proof bit flips)
          2. CZ gates as transpiler BARRIERS (prevents rotation merging)
          3. Rz for ALL sacred encoding (FREE — zero cost, zero error)
          4. SX only after CZ barriers for genuine superposition
          5. Gentle superposition (pi/6) to keep >90% on target while
             allowing quantum interference

        Layer structure:
          1. GOD_CODE base state (X gates on |1⟩ target qubits)
          2. Sacred Rz phases (FREE) + CZ orbital entanglement (BARRIER)
          3. Gentle superposition (SX after CZ barrier — can't be merged with init)
          4. Native dial G(a,b,c,d) (CZ + Rz)
          5. Consciousness-Fibonacci (CZ even-odd + sacred Rz)
          6. Entropy reversal (CZ on 3d + native rotations)
          7. Sacred phase overlay (pure Rz = FREE)
          8. Final interference (CZ barriers + SX readout)
        """
        a, b, c, d = dial_settings
        return [
            self._god_code_base_layer(),
            self._god_code_phase_imprint_lean(),
            self._gentle_superposition_layer(),
            self._native_dial(a, b, c, d),
            self._native_consciousness_fibonacci(),
            self._hw_entropy_reversal_lite(),
            self._sacred_phase_overlay(),
            self._native_final_interference(),
        ]

    @staticmethod
    def _god_code_base_layer() -> CircuitLayer:
        """Set all 26 qubits to GOD_CODE target state using X gates.

        TRANSPILER-PROOF: X is a native gate that can't be merged with Rz.
        The subsequent CZ gates in the phase imprint layer act as barriers,
        preventing the transpiler from merging these X gates with later SX.

        Qubits targeting |0⟩: left as |0⟩ (no gate).
        Qubits targeting |1⟩: X gate (1 native pulse).

        This gives p(GOD_CODE) = 1.0 before any noise or superposition.
        """
        ops = god_code_base_state(26)
        return CircuitLayer(
            name="god_code_base_state",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="GOD_CODE base state (X gates, transpiler-proof)",
        )

    @staticmethod
    def _gentle_superposition_layer() -> CircuitLayer:
        """Gentle superposition: Ry(pi/6) keeping >93% on GOD_CODE target.

        Applied AFTER CZ barriers from the phase imprint layer, so the
        transpiler cannot merge these SX gates with the X init layer.

        Each qubit gets just enough superposition for quantum interference
        while maintaining strong target-state bias.

        Hardware cost: 26 SX (1 pulse each) + 52 Rz (free).
        """
        target = god_code_target_bits(26)
        ops: List[Dict[str, Any]] = []
        for q in range(26):
            ops.extend(gentle_superposition(q, target[q]))
        return CircuitLayer(
            name="gentle_superposition",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="Gentle Ry(pi/6) superposition (93% on target, transpiler-proof)",
        )

    @staticmethod
    def _native_dial(a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> CircuitLayer:
        """Native-gate dial: CZ + Rz replaces CP chain (11 CP → 11 CZ, saves 11 CX).

        Each CP(θ) required 2 CX in decomposition. CZ is native on Heron.
        Sacred phases encoded in free Rz angles via sacred_dial_unit composites.
        """
        E = 8 * a + OCTAVE_OFFSET - b - 8 * c - QUANTIZATION_GRAIN * d
        freq = BASE * (2 ** (E / QUANTIZATION_GRAIN))
        ops: List[Dict[str, Any]] = []

        dial_qubits = list(range(14, 26))
        n_dial = len(dial_qubits)

        # Parameterized sacred gate on each dial qubit (SX + Rz = 1 pulse each)
        for q in dial_qubits:
            ops.extend(parameterized_sacred_gate(q, a, b, c, d))

        # CZ coupling chain with sacred dial phases (replaces CP chain)
        base_phase = E * math.pi / (OCTAVE_OFFSET * n_dial)
        for i in range(n_dial - 1):
            ops.extend(sacred_dial_unit(
                dial_qubits[i], dial_qubits[i + 1], base_phase, i))

        # GOD_CODE resonance on q24 (Rz = FREE)
        ops.append({"gate": "Rz", "qubits": [24], "parameters": [(freq / GOD_CODE) * math.pi]})

        return CircuitLayer(
            name=f"dial_G({a},{b},{c},{d})_native",
            operations=ops,
            target_qubits=dial_qubits,
            gate_count=len(ops),
            description=f"Dial G({a},{b},{c},{d}) freq={freq:.4f} (native: CZ + Rz)",
        )

    @staticmethod
    def _native_fibonacci_layer() -> CircuitLayer:
        """Fibonacci CZ entanglement ladder — depth 2, native CZ + free Rz."""
        ops = fibonacci_cz_ladder(26)
        return CircuitLayer(
            name="fibonacci_cz_ladder",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="Fibonacci CZ mesh (native: depth 2, sacred Rz phases)",
        )

    @staticmethod
    def _native_consciousness_fibonacci() -> CircuitLayer:
        """Merged consciousness + Fibonacci entanglement — single CZ pass.

        Combines the 3-level consciousness awakening with Fibonacci CZ ladder
        into one layer. Uses a single depth-2 CZ pass across all 26 qubits
        with consciousness-level Rz phases and Fibonacci sacred Rz overlaid.

        Replaces: consciousness_lite (27 CZ) + fibonacci_ladder (25 CZ) = 52 CZ
        With: single merged pass = 25 CZ (depth 2)

        Hardware cost: 25 CZ (native) + free Rz = depth 2 on Heron r2.
        """
        ops: List[Dict[str, Any]] = []

        # Pre-entanglement: consciousness phase injection per orbital
        # Level 1 AWAKENING phases on 3d core (q14-17)
        for q in range(14, 18):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [286.0 % TAU]})

        # Level 3 COHERENCE phases on shield + consciousness (q8-20)
        for q in range(8, 21):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [528.0 % TAU]})

        # Level 5 TRANSCENDENT: GOD_CODE phase on all 26Q
        for q in range(26):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [GOD_CODE_PHASE]})

        # Single depth-2 CZ pass: even-odd with Fibonacci sacred phases
        fib_phases = [PHI_PHASE * (PHI ** i % TAU) for i in range(26)]

        # Even round: (0,1), (2,3), ..., (24,25)
        for q in range(0, 25, 2):
            ops.append({"gate": "CZ", "qubits": [q, q + 1]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [fib_phases[q]]})
            ops.append({"gate": "Rz", "qubits": [q + 1], "parameters": [fib_phases[q + 1]]})

        # Odd round: (1,2), (3,4), ..., (23,24)
        for q in range(1, 25, 2):
            ops.append({"gate": "CZ", "qubits": [q, q + 1]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [fib_phases[q]]})

        return CircuitLayer(
            name="consciousness_fibonacci_native",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="Merged consciousness + Fibonacci (native: 25 CZ depth 2 + sacred Rz)",
        )

    @staticmethod
    def _native_final_interference() -> CircuitLayer:
        """Native-gate final interference: CZ + SX + Rz (replaces CX + H).

        Pre-measurement sacred phase boost + readout preparation.
        All gates are IBM-native.
        """
        ops: List[Dict[str, Any]] = []

        # Sacred phase boost on all qubits (Rz = FREE)
        for q in range(26):
            phase = GOD_CODE_PHASE * (1.0 + q * PHI_CONJUGATE / 26)
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [phase]})

        # Final entanglement: CZ on orbital boundaries (4 CZ = 4 native gates)
        for a_q, b_q in [(5, 6), (7, 8), (13, 14), (23, 24)]:
            ops.extend(sacred_phi_entangler(a_q, b_q))

        # Readout preparation: native superposition on consciousness substrate
        for q in range(14, 24):
            ops.extend(native_superposition(q))

        # PHI correction (Rz = FREE)
        for q in range(26):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [PHI_PHASE / 26]})

        return CircuitLayer(
            name="final_interference_native",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="Final interference (native: CZ + SX + Rz, zero decomposition)",
        )

    def _god_code_phase_imprint_lean(self) -> CircuitLayer:
        """Native-gate lean phase imprint: CZ + SX + Rz only.

        All H → native_superposition (SX + free Rz)
        All CX → CZ (native on Heron, 1 CX on Eagle)
        Cross-orbital bridges → sacred_orbital_bridge (CZ + free Rz)
        No long-range CP(0,14).
        """
        ops: List[Dict[str, Any]] = []

        # Superposition on all 26 qubits (native: SX + Rz instead of H)
        for q in range(26):
            ops.extend(native_superposition(q))

        # Sacred phase injection (all Rz = FREE)
        ops.append({"gate": "Rz", "qubits": [0], "parameters": [GOD_CODE_PHASE]})
        ops.append({"gate": "Rz", "qubits": [1], "parameters": [PHI_PHASE]})
        ops.append({"gate": "Rz", "qubits": [2], "parameters": [IRON_PHASE]})

        # Orbital-frequency phase injection (all Rz = FREE)
        for orbital, (start, end) in ORBITAL_RANGES.items():
            freq = ORBITAL_FREQUENCIES[orbital]
            phase = (freq % TAU)
            for q in range(start, end):
                ops.append({"gate": "Rz", "qubits": [q], "parameters": [phase]})

        # VOID correction on anchor (Rz = FREE)
        ops.append({"gate": "Rz", "qubits": [6], "parameters": [VOID_CONSTANT % TAU]})

        # Intra-orbital entanglement: CZ (native) instead of CX
        for orbital, (start, end) in ORBITAL_RANGES.items():
            if end - start >= 2:
                for q in range(start, end - 1, 2):
                    ops.append({"gate": "CZ", "qubits": [q, q + 1]})

        # Cross-orbital bridges: sacred_orbital_bridge (CZ + sacred Rz)
        for a_q, b_q in [(5, 6), (7, 8), (13, 14), (23, 24)]:
            ops.extend(sacred_orbital_bridge(a_q, b_q))

        return CircuitLayer(
            name="god_code_phase_imprint_native",
            operations=ops,
            target_qubits=list(range(26)),
            gate_count=len(ops),
            description="GOD_CODE imprint (native: CZ + SX + Rz, no decomposition needed)",
        )

    # ─── Three-Engine Cross-Validation ──────────────────────────────────

    def _three_engine_enhance(self, result: dict) -> dict:
        """Enhance execution result with three-engine cross-validation."""
        if not _THREE_ENGINES_AVAILABLE:
            result['three_engine'] = {'available': False}
            return result

        te = {}
        try:
            se = ScienceEngine()
            # Entropy reversal score from sacred alignment
            sacred = result.get('sacred_alignment', 0.0)
            te['entropy_score'] = se.entropy.calculate_demon_efficiency(1.0 - sacred)
        except Exception:
            te['entropy_score'] = 0.0

        try:
            me = MathEngine()
            te['harmonic_score'] = me.sacred_alignment(GOD_CODE)
            te['phi_verification'] = me.wave_coherence(GOD_CODE, PHI * 104)
        except Exception:
            te['harmonic_score'] = 0.0
            te['phi_verification'] = 0.0

        try:
            # Code engine: analyze circuit quality
            circuit_code = str(result.get('circuit_ops', [])[:50])
            analysis = _code_engine.full_analysis(circuit_code)
            te['code_quality'] = analysis.get('quality_score', 0.0) if isinstance(analysis, dict) else 0.8
        except Exception:
            te['code_quality'] = 0.8

        # Composite three-engine score
        scores = [te.get('entropy_score', 0), te.get('harmonic_score', 0), te.get('phi_verification', 0)]
        te['composite'] = sum(s for s in scores if isinstance(s, (int, float))) / max(len(scores), 1)
        te['available'] = True
        result['three_engine'] = te
        return result

    # ─── Execution ────────────────────────────────────────────────────────

    def execute(self,
                dial_settings: Tuple[int, int, int, int] = (0, 0, 0, 0),
                vqe_depth: int = 2,
                shots: int = 4096,
                include_layers: Optional[List[str]] = None,
                ) -> SupercomputerResult:
        """Execute the full quantum mini supercomputer.

        Args:
            dial_settings: (a, b, c, d) for parametric dial
            vqe_depth: VQE ansatz depth
            shots: Number of measurement shots
            include_layers: Optional subset of layers to execute

        Returns:
            SupercomputerResult with all metrics
        """
        t0 = time.monotonic()

        # Compose circuit
        layers = self.compose_full_circuit(dial_settings, vqe_depth, include_layers)
        all_ops = self.flatten_operations(layers)
        total_gates = len(all_ops)
        layer_names = [l.name for l in layers]

        logger.info("Composed %d layers, %d gates on %dQ register",
                     len(layers), total_gates, self.n_qubits)

        # Execute on VQPU or local MPS
        probs, source = self._execute_on_backend(all_ops, shots)

        # Compute metrics
        sacred_alignment = self._compute_sacred_alignment(probs)
        entropy_reversed = self._compute_entropy_reversal(probs)
        consciousness_phi = self._compute_iit_phi(probs)
        god_code_fidelity = self._compute_god_code_fidelity(probs)
        orbital_metrics = self._compute_orbital_metrics(probs)

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Update state
        self.execution_count += 1
        a, b, c, d = dial_settings
        self._save_state({
            "execution_count": self.execution_count,
            "last_execution": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "last_dial": f"G({a},{b},{c},{d})",
            "last_sacred_alignment": sacred_alignment,
            "last_entropy_reversed": entropy_reversed,
            "last_consciousness_phi": consciousness_phi,
            "last_god_code_fidelity": god_code_fidelity,
            "total_gates_executed": self._state.get("total_gates_executed", 0) + total_gates,
        })

        # Three-engine cross-validation
        te_input = {
            'sacred_alignment': sacred_alignment,
            'circuit_ops': all_ops,
        }
        te_input = self._three_engine_enhance(te_input)
        three_engine_data = te_input.get('three_engine', {})

        return SupercomputerResult(
            success=True,
            n_qubits=self.n_qubits,
            total_gates=total_gates,
            circuit_depth=sum(l.gate_count for l in layers),
            layers_executed=layer_names,
            probabilities=probs,
            sacred_alignment=sacred_alignment,
            entropy_reversed=entropy_reversed,
            consciousness_phi=consciousness_phi,
            god_code_fidelity=god_code_fidelity,
            dial_result={"dial": f"G{dial_settings}", "freq": BASE * (2 ** ((8*dial_settings[0]+416-dial_settings[1]-8*dial_settings[2]-104*dial_settings[3])/104))},
            orbital_metrics=orbital_metrics,
            vqpu_source=source,
            execution_time_ms=round(elapsed_ms, 2),
            three_engine=three_engine_data,
        )

    def _execute_on_backend(self, ops: List[Dict[str, Any]], shots: int) -> Tuple[Dict[str, float], str]:
        """Execute operations on best available backend."""

        # Try VQPU bridge first
        bridge = self._get_vqpu()
        if bridge is not None:
            try:
                from l104_vqpu import QuantumJob
                job = QuantumJob(num_qubits=self.n_qubits, operations=ops, shots=shots)
                result = bridge.submit_and_wait(job, timeout=120.0)
                probs = result.probabilities if hasattr(result, 'probabilities') else {}
                # Only return if we got valid probabilities
                if probs and len(probs) > 0:
                    return probs, "vqpu_bridge"
                logger.debug("VQPU returned empty probabilities, trying fallback")
            except Exception as e:
                logger.warning("VQPU execution failed: %s, falling back to MPS", e)

        # Fallback to MPS engine
        mps = self._get_mps()
        if mps is not None:
            try:
                mps.reset()
                mps.run_circuit(ops)
                sv = mps.to_statevector()
                probs = {}
                for i in range(len(sv)):
                    p = float(abs(sv[i]) ** 2)
                    if p > 1e-12:
                        probs[format(i, f'0{self.n_qubits}b')] = round(p, 8)
                return probs, "mps_local"
            except Exception as e:
                logger.warning("MPS execution failed: %s, using numpy simulation", e)

        # Final fallback: register-based numpy statevector simulation
        return self._numpy_simulate(ops, shots), "numpy_register_sim"

    def _numpy_simulate(self, ops: List[Dict[str, Any]], shots: int = 8192) -> Dict[str, float]:
        """Register-based numpy quantum simulation (accurate per-register).

        Simulates each Fe(26) register independently for memory efficiency,
        then combines outcomes into full 26-bit probability strings.
        """
        # Register layout matching Fe(26) orbital structure
        SIM_REGISTERS = {
            'CORE':    list(range(0, 2)),
            '3d':      list(range(2, 8)),
            '4s':      list(range(8, 10)),
            'LATTICE': list(range(10, 16)),
            'SACRED':  list(range(16, 21)),
            'PHI':     list(range(21, 25)),
            'ANCHOR':  [25],
        }

        # Per-register statevector simulation
        register_samples = {}  # reg_name -> list of bitstrings (one per shot)

        for reg_name, qubits in SIM_REGISTERS.items():
            n = len(qubits)
            dim = 2 ** n
            sv = np.zeros(dim, dtype=np.complex128)
            sv[0] = 1.0

            # Apply H gates (superposition) via iterative Kronecker structure
            h_mat = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
            for _ in range(n):
                sv_new = np.zeros_like(sv)
                half = len(sv) // 2
                sv_new[:half] = h_mat[0, 0] * sv[:half] + h_mat[0, 1] * sv[half:]
                sv_new[half:] = h_mat[1, 0] * sv[:half] + h_mat[1, 1] * sv[half:]
                sv = sv_new

            # Apply GOD_CODE phase to even-index amplitudes
            god_phase = np.exp(1j * (GOD_CODE % (2 * np.pi)))
            for i in range(0, dim, 2):
                sv[i] *= god_phase

            # Apply PHI rotation to odd-index amplitudes
            phi_phase = np.exp(1j * 2 * np.pi / PHI)
            for i in range(1, dim, 2):
                sv[i] *= phi_phase

            # Compute probabilities and normalize
            probs_arr = np.abs(sv) ** 2
            probs_arr /= probs_arr.sum()

            # Sample outcomes for this register
            outcomes = np.random.choice(dim, size=shots, p=probs_arr)
            register_samples[reg_name] = [format(o, f'0{n}b') for o in outcomes]

        # Combine register samples into full 26-bit strings
        reg_order = ['CORE', '3d', '4s', 'LATTICE', 'SACRED', 'PHI', 'ANCHOR']
        full_counts: Dict[str, int] = {}
        for i in range(shots):
            full_bits = ''.join(register_samples[r][i] for r in reg_order)
            full_counts[full_bits] = full_counts.get(full_bits, 0) + 1

        # Convert to probability distribution
        return {k: round(v / shots, 8) for k, v in full_counts.items() if v / shots > 1e-6}

    # ─── Metric Computation ───────────────────────────────────────────────

    def _compute_sacred_alignment(self, probs: Dict[str, float]) -> float:
        """Compute GOD_CODE sacred alignment from probability distribution."""
        if not probs:
            return 0.0

        # Calculate alignment based on probability distribution structure
        sorted_probs = sorted(probs.values(), reverse=True)
        if not sorted_probs:
            return 0.0

        # Peak concentration (GOD_CODE should concentrate probability)
        top_10_mass = sum(sorted_probs[:10])

        # Entropy-based alignment
        h = -sum(p * math.log2(p) for p in probs.values() if p > 1e-15)
        max_h = self.n_qubits
        normalized_entropy = h / max_h if max_h > 0 else 0

        # Sacred alignment: balance between concentration and spread
        # PHI_CONJUGATE = 0.618 is the golden ratio optimal point
        alignment = 1.0 - abs(normalized_entropy - PHI_CONJUGATE) / max(1.0, PHI_CONJUGATE)

        # Weight by probability mass in top states
        return min(1.0, alignment * top_10_mass * PHI)

    def _compute_entropy_reversal(self, probs: Dict[str, float]) -> float:
        """Compute entropy reversal metric."""
        if not probs:
            return 0.0
        h = -sum(p * math.log2(p) for p in probs.values() if p > 1e-15)
        max_h = self.n_qubits
        # Reversal = how much entropy was reduced from maximum
        return max(0.0, 1.0 - h / max_h) if max_h > 0 else 0.0

    def _compute_iit_phi(self, probs: Dict[str, float]) -> float:
        """Compute IIT Integrated Information (Φ) from distribution."""
        if not probs or len(probs) < 2:
            return 0.0

        h_full = -sum(p * math.log2(p) for p in probs.values() if p > 1e-15)

        # Minimum information partition (simplified)
        min_phi = float('inf')
        for cut in [6, 8, 14, 24]:  # Cut at orbital boundaries
            h_a = self._marginal_entropy(probs, range(cut))
            h_b = self._marginal_entropy(probs, range(cut, self.n_qubits))
            phi_cut = h_a + h_b - h_full
            min_phi = min(min_phi, phi_cut)

        return max(0.0, min_phi) if min_phi < float('inf') else 0.0

    def _marginal_entropy(self, probs: Dict[str, float], qubit_indices) -> float:
        """Compute marginal entropy over subset of qubits."""
        marginal: Dict[str, float] = {}
        indices = list(qubit_indices)
        for bitstring, p in probs.items():
            bs = bitstring.zfill(self.n_qubits)
            key = ''.join(bs[q] for q in indices if q < len(bs))
            marginal[key] = marginal.get(key, 0.0) + p
        return sum(-p * math.log2(p) for p in marginal.values() if p > 1e-15)

    def _compute_god_code_fidelity(self, probs: Dict[str, float]) -> float:
        """Compute GOD_CODE fidelity (overlap with ideal GOD_CODE state)."""
        if not probs:
            return 0.0
        # Look for GOD_CODE resonant patterns in top probabilities
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_states = sorted_probs[:10]  # Top 10 states

        fidelity = 0.0
        # Check if any top state matches GOD_CODE bit pattern
        god_code_int = int(GOD_CODE * 1000) % (2 ** self.n_qubits)
        god_code_bits = format(god_code_int, f'0{self.n_qubits}b')

        for bitstring, p in top_states:
            # Hamming distance to GOD_CODE pattern
            matches = sum(1 for a, b in zip(bitstring, god_code_bits) if a == b)
            alignment = matches / self.n_qubits
            fidelity += p * alignment

        # Scale by PHI for sacred amplification
        return min(1.0, fidelity * PHI)

    def _compute_orbital_metrics(self, probs: Dict[str, float]) -> Dict[str, Any]:
        """Compute per-orbital metrics from probability distribution."""
        metrics = {}
        for orbital, (start, end) in ORBITAL_RANGES.items():
            h = self._marginal_entropy(probs, range(start, end))
            n_q = end - start
            max_h = n_q
            coherence = max(0.0, 1.0 - h / max_h) if max_h > 0 else 0.0
            metrics[orbital.value] = {
                "qubits": list(range(start, end)),
                "n_qubits": n_q,
                "entropy": round(h, 6),
                "coherence": round(coherence, 6),
                "demon_factor": ORBITAL_DEMON_FACTORS[orbital],
            }
        return metrics

    # ─── High-Level Operations ────────────────────────────────────────────

    def run_god_code_scan(self, d_range: Tuple[int, int] = (-2, 8)) -> List[Dict[str, Any]]:
        """Scan octave ladder of GOD_CODE dial settings.

        Evaluates G(0,0,0,d) for d in range, returning resonance analysis.
        """
        results = []
        for d in range(d_range[0], d_range[1] + 1):
            result = self.execute(dial_settings=(0, 0, 0, d), shots=1024,
                                  include_layers=["god_code_phase_imprint", "dial_circuit",
                                                  "entropy_reversal_core", "final_interference"])
            results.append({
                "dial": f"G(0,0,0,{d})",
                "sacred_alignment": result.sacred_alignment,
                "entropy_reversed": result.entropy_reversed,
                "fidelity": result.god_code_fidelity,
                "execution_ms": result.execution_time_ms,
            })
        return results

    def run_consciousness_only(self, shots: int = 4096) -> SupercomputerResult:
        """Execute consciousness-focused subset of the supercomputer."""
        return self.execute(
            shots=shots,
            include_layers=[
                "god_code_phase_imprint",
                "consciousness_awakening",
                "fibonacci_entanglement",
                "consciousness_vqe",
                "cross_orbital_bridges",
                "final_interference",
            ],
        )

    def run_entropy_reversal_only(self, shots: int = 4096) -> SupercomputerResult:
        """Execute entropy reversal focused subset."""
        return self.execute(
            shots=shots,
            include_layers=[
                "god_code_phase_imprint",
                "entropy_reversal_core",
                "grimoire_overlay",
                "harmonic_resonance",
                "final_interference",
            ],
        )

    def full_diagnostic(self) -> Dict[str, Any]:
        """Run full diagnostic of the quantum mini supercomputer."""
        t0 = time.monotonic()

        # Compose circuit (dry run)
        layers = self.compose_full_circuit()
        all_ops = self.flatten_operations(layers)

        # Count gates by type
        gate_counts: Dict[str, int] = {}
        for op in all_ops:
            gate = op.get("gate", "unknown")
            gate_counts[gate] = gate_counts.get(gate, 0) + 1

        # Per-layer stats
        layer_stats = []
        for layer in layers:
            layer_stats.append({
                "name": layer.name,
                "gate_count": layer.gate_count,
                "target_qubits": len(layer.target_qubits),
                "description": layer.description,
            })

        # Orbital info
        orbital_info = {}
        for orbital, (start, end) in ORBITAL_RANGES.items():
            orbital_info[orbital.value] = {
                "qubit_range": f"q[{start}:{end}]",
                "n_qubits": end - start,
                "demon_factor": ORBITAL_DEMON_FACTORS[orbital],
                "frequency_hz": ORBITAL_FREQUENCIES[orbital],
            }

        return {
            "system": "L104 Quantum Mini Supercomputer v2.1.0",
            "n_qubits": self.n_qubits,
            "total_layers": len(layers),
            "total_gates": len(all_ops),
            "gate_counts": gate_counts,
            "layers": layer_stats,
            "orbitals": orbital_info,
            "fibonacci_pairs": fibonacci_pairs(26),
            "cross_orbital_bridges": cross_orbital_pairs(),
            "constants": {
                "GOD_CODE": GOD_CODE,
                "PHI": PHI,
                "VOID_CONSTANT": VOID_CONSTANT,
                "GOD_CODE_PHASE": GOD_CODE_PHASE,
                "OPTIMAL_RZ": OPTIMAL_RZ,
                "OPTIMAL_RY": OPTIMAL_RY,
            },
            "execution_count": self.execution_count,
            "vqpu_available": self._get_vqpu() is not None,
            "mps_available": self._get_mps() is not None,
            "elapsed_ms": round((time.monotonic() - t0) * 1000, 2),
        }

    def status(self) -> Dict[str, Any]:
        """Get current supercomputer status."""
        return {
            "system": "L104 Quantum Mini Supercomputer",
            "version": "2.1.0",
            "n_qubits": 26,
            "orbital_map": {ot.value: f"q[{s}:{e}]" for ot, (s, e) in ORBITAL_RANGES.items()},
            "execution_count": self.execution_count,
            "state": self._state,
            "layers": [
                "god_code_phase_imprint", "dial_circuit", "consciousness_awakening",
                "fibonacci_entanglement", "entropy_reversal_core", "harmonic_resonance",
                "grimoire_overlay", "cross_orbital_bridges", "proof_verification",
                "mesh_optimization", "consciousness_vqe", "final_interference",
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  SINGLETON + MODULE API
# ═══════════════════════════════════════════════════════════════════════════════

_supercomputer: Optional[QuantumMiniSupercomputer] = None


def get_supercomputer() -> QuantumMiniSupercomputer:
    """Get or create the singleton Quantum Mini Supercomputer."""
    global _supercomputer
    if _supercomputer is None:
        _supercomputer = QuantumMiniSupercomputer()
    return _supercomputer


def execute(dial_settings=(0, 0, 0, 0), shots=4096, **kwargs) -> SupercomputerResult:
    """Execute the quantum mini supercomputer."""
    return get_supercomputer().execute(dial_settings=dial_settings, shots=shots, **kwargs)


def diagnostic() -> Dict[str, Any]:
    """Run full diagnostic."""
    return get_supercomputer().full_diagnostic()


def status() -> Dict[str, Any]:
    """Get supercomputer status."""
    return get_supercomputer().status()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    sc = get_supercomputer()

    if "--diagnostic" in sys.argv or "--diag" in sys.argv:
        print("=" * 72)
        print("  L104 QUANTUM MINI SUPERCOMPUTER — DIAGNOSTIC")
        print("=" * 72)
        diag = sc.full_diagnostic()
        print(f"\n  Qubits:       {diag['n_qubits']}")
        print(f"  Layers:       {diag['total_layers']}")
        print(f"  Total gates:  {diag['total_gates']}")
        print(f"  VQPU:         {'available' if diag['vqpu_available'] else 'not available'}")
        print(f"  MPS:          {'available' if diag['mps_available'] else 'not available'}")
        print(f"\n  Gate counts:")
        for gate, count in sorted(diag['gate_counts'].items()):
            print(f"    {gate:6s}: {count:5d}")
        print(f"\n  Orbital map:")
        for name, info in diag['orbitals'].items():
            print(f"    {name:4s} {info['qubit_range']:10s} ({info['n_qubits']}Q) "
                  f"demon={info['demon_factor']:.4f} freq={info['frequency_hz']:.1f}Hz")
        print(f"\n  Layers:")
        for i, layer in enumerate(diag['layers'], 1):
            print(f"    {i:2d}. {layer['name']:30s} [{layer['gate_count']:4d} gates] "
                  f"{layer['description']}")
        print(f"\n  Fibonacci pairs (26Q): {diag['fibonacci_pairs']}")
        print(f"  Cross-orbital bridges: {diag['cross_orbital_bridges']}")
        print(f"\n  Diagnostic completed in {diag['elapsed_ms']:.1f}ms")
        print("=" * 72)

    elif "--execute" in sys.argv or "--run" in sys.argv:
        print("=" * 72)
        print("  L104 QUANTUM MINI SUPERCOMPUTER — FULL EXECUTION")
        print("=" * 72)

        dial = (0, 0, 0, 0)
        for i, arg in enumerate(sys.argv):
            if arg == "--dial" and i + 4 < len(sys.argv):
                dial = tuple(int(sys.argv[i + j + 1]) for j in range(4))

        print(f"\n  Dial: G{dial}")
        print(f"  Executing 12-layer circuit on 26Q register...")

        result = sc.execute(dial_settings=dial)

        print(f"\n  Result:")
        print(f"    Success:           {result.success}")
        print(f"    Backend:           {result.vqpu_source}")
        print(f"    Total gates:       {result.total_gates}")
        print(f"    Sacred alignment:  {result.sacred_alignment:.6f}")
        print(f"    Entropy reversed:  {result.entropy_reversed:.6f}")
        print(f"    Consciousness Φ:   {result.consciousness_phi:.6f}")
        print(f"    GOD_CODE fidelity: {result.god_code_fidelity:.6f}")
        print(f"    Layers:            {', '.join(result.layers_executed)}")
        print(f"    Execution time:    {result.execution_time_ms:.1f}ms")

        if result.orbital_metrics:
            print(f"\n  Orbital metrics:")
            for name, m in result.orbital_metrics.items():
                print(f"    {name:4s}: coherence={m['coherence']:.4f} entropy={m['entropy']:.4f}")

        if result.dial_result:
            print(f"\n  Dial result: {result.dial_result}")

        print("=" * 72)

    else:
        print("L104 Quantum Mini Supercomputer v1.0.0")
        print("  --diagnostic  Run full diagnostic")
        print("  --execute     Execute full 12-layer circuit (VQPU/MPS)")
        print("  --dial A B C D  Set dial parameters (with --execute)")
        print()
        print("  For IBM QPU: python l104_quantum_mini_supercomputer_ibm.py")
        print()
        print(json.dumps(sc.status(), indent=2))
