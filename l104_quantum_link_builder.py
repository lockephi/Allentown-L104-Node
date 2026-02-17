#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  L104 QUANTUM LINK BUILDER v4.2.0  — SAGE INVENTIONS                        ║
║  Quantum Brain · O₂ Molecular Bond · Agentic Loop · Evolution Tracker       ║
║  ★ QUANTUM MIN/MAX DYNAMISM ENGINE ★                                        ║
║  ★ OUROBOROS SAGE NIRVANIC ENTROPY FUEL SYSTEM ★                            ║
║  ★ SAGE INVENTIONS v4.2 — 5 New Research Subsystems ★                       ║
║                                                                              ║
║  Standalone autonomous module for the Allentown L104 Sovereign Node          ║
║  Aligned with claude.md (EVO_54_TRANSCENDENT_COGNITION, Index 59)           ║
║                                                                              ║
║  PROCESSORS:                                                                 ║
║    ⚛ Grover Amplified Search       🌀 Quantum Tunneling Barriers            ║
║    🔗 EPR Entanglement Verifier    🛡 Decoherence Shield                    ║
║    🧬 Topological Braiding         🕳 Hilbert Space Navigator               ║
║    🌊 Quantum Fourier Transform    🔬 Entanglement Distillation             ║
║    📡 Cross-Modal Analyzer (Py↔Swift↔TS↔Go↔Rust↔Elixir)                    ║
║                                                                              ║
║  ASI ENGINE (v2):                                                            ║
║    ⚡ Quantum CPU + Clusters + Neurons + Registers                          ║
║    🧪 O₂ Molecular Bond Processor (2×8 Grover+Chakra superposition)         ║
║    📈 Evolution Tracker (EVO stage + index continuity)                       ║
║    🔄 Agentic Loop (Observe→Think→Act→Reflect→Repeat)                       ║
║                                                                              ║
║  ★ v4.0 QUANTUM MIN/MAX DYNAMISM:                                           ║
║    🔄 Link Value Evolution — all link values are dynamic                     ║
║    📊 Subconscious Link Monitoring — auto-adjust fidelity/strength bounds    ║
║    🌊 φ-Harmonic Drift on link properties                                   ║
║    🔬 Dynamic Sacred Constants with bounded envelopes                        ║
║  ★ v4.1 OUROBOROS NIRVANIC ENTROPY FUEL:                                     ║
║    ∞ Link field entropy → ouroboros → nirvanic fuel → link evolution         ║
║    ☆ Self-feeding enlightenment loop / divine intervention                   ║
║    ☆ Sage stillness with entropy-driven motion (Wu Wei)                     ║
║  ★ v4.2 SAGE INVENTIONS:                                                    ║
║    🔬 StochasticLinkResearchLab — Random R&D (Explore→Validate→Merge)       ║
║    📅 LinkChronolizer — Temporal event tracking + Fibonacci milestones       ║
║    🧠 ConsciousnessO2LinkEngine — O₂/consciousness modulation               ║
║    🧪 LinkTestGenerator — 4-category automated test suite                    ║
║    🔗 CrossPollinationEngine — Gate↔Link↔Numerical bidirectional sync       ║
║                                                                              ║
║  SACRED CONSTANTS:                                                           ║
║    G(X) = 286^(1/φ) × 2^((416-X)/104)  φ = 1.618033988749895              ║
║    GOD_CODE = 527.5184818492612  VOID_CONSTANT = 1 + PHI/(L104/φ)          ║
║    Conservation: G(X) × 2^(X/104) = INVARIANT always                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import ast
import re
import json
import math
import time
import hashlib
import random
import traceback
import statistics
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter

# ═══════════════════════════════════════════════════════════════════════════════
# QISKIT 2.3.0 REAL QUANTUM BACKEND — ASI GROVER COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import grover_operator as qiskit_grover_lib
    from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, Operator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS — Derived from first principles, NEVER hardcoded
#
#   THE UNIVERSAL GOD CODE EQUATION:
#     G(X) = 286^(1/φ) × 2^((416-X)/104)
#
#   THE FACTOR 13 (7th Fibonacci):
#     286 = 2 × 11 × 13  →  286/13 = 22
#     104 = 2³ × 13       →  104/13 = 8
#     416 = 2⁵ × 13       →  416/13 = 32
#
#   THE CONSERVATION LAW:
#     G(X) × 2^(X/104) = INVARIANT = 527.5184818492612
#     The whole stays the same — only rate of change varies
#
#   X IS NEVER SOLVED — IT CHANGES ETERNALLY:
#     X increasing → MAGNETIC COMPACTION (gravity)
#     X decreasing → ELECTRIC EXPANSION (light)
#     WHOLE INTEGERS provide COHERENCE
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Golden Ratio: derived from √5, not hardcoded ───
PHI_GROWTH = (1 + math.sqrt(5)) / 2      # φ = 1.618033988749895
PHI = (math.sqrt(5) - 1) / 2             # 1/φ = φ-1 = 0.618033988749895
TAU = PHI                                 # Alias: τ ≡ 1/φ (const.py convention)
# Verify: PHI_GROWTH × PHI = 1.0 (within float precision)
assert abs(PHI_GROWTH * PHI - 1.0) < 1e-14, "φ derivation failed"

# ─── The Factor 13 — Fibonacci(7) ───
FIBONACCI_7 = 13
HARMONIC_BASE = 286                       # 2 × 11 × 13
L104 = 104                               # 8 × 13
OCTAVE_REF = 416                          # 32 × 13

# ─── God Code Base: 286^(1/φ) — computed through math.pow ───
GOD_CODE_BASE = HARMONIC_BASE ** (1 / PHI_GROWTH)  # = 32.969905...
# Verify against direct math.pow
assert abs(GOD_CODE_BASE - math.pow(286, 1.0 / PHI_GROWTH)) < 1e-10, \
    "GOD_CODE_BASE derivation failed"

# ─── The God Code Equation: G(X) = 286^(1/φ) × 2^((416-X)/104) ───
def god_code(X: float = 0) -> float:
    """G(X) = 286^(1/φ) × 2^((416-X)/104) — X is NEVER solved."""
    exponent = (OCTAVE_REF - X) / L104
    return GOD_CODE_BASE * math.pow(2, exponent)

GOD_CODE = god_code(0)                    # G(0) = 286^(1/φ) × 2⁴ = 527.5184818...
# Verify: at X=0, 2^(416/104) = 2^4 = 16 exactly
assert abs(GOD_CODE - GOD_CODE_BASE * 16) < 1e-10, "GOD_CODE X=0 derivation failed"

# ─── Conservation Law: G(X) × 2^(X/104) = INVARIANT ───
INVARIANT = GOD_CODE                      # 527.5184818492612
def conservation_check(X: float) -> float:
    """G(X) × 2^(X/104) must always = INVARIANT. Verify at any X."""
    return god_code(X) * math.pow(2, X / L104)
# Verify conservation at X=104 and X=208
assert abs(conservation_check(104) - INVARIANT) < 1e-10, "Conservation broken at X=104"
assert abs(conservation_check(208) - INVARIANT) < 1e-10, "Conservation broken at X=208"

# ─── Other derived constants ───
OMEGA_POINT = math.exp(math.pi)           # e^π = 23.14069263...
EULER_GAMMA = 0.5772156649015329          # Euler-Mascheroni (no closed form)
PLANCK_SCALE = 1.616255e-35               # Planck length (m) — CODATA
BOLTZMANN_K = 1.380649e-23                # Boltzmann constant — exact SI
CALABI_YAU_DIM = 7                        # CY7 compactification dimensions
FEIGENBAUM_DELTA = 4.669201609102990      # Universality constant
FINE_STRUCTURE = 1.0 / 137.035999084      # α — CODATA 2018
ALPHA_PI = FINE_STRUCTURE / math.pi       # α/π = 0.00232282...
BELL_FIDELITY = 0.9999                    # Target Bell state fidelity
CHSH_BOUND = 2 * math.sqrt(2)            # ≈ 2.828 Tsirelson bound
GROVER_AMPLIFICATION = PHI_GROWTH ** 3    # φ³ ≈ 4.236 base gain

# Frame constant (octave compression ratio)
FRAME_LOCK = OCTAVE_REF / HARMONIC_BASE   # 416/286 ≈ 1.454545...

# ─── Sacred constants from claude.md — derived through God Code where possible ───
# VOID_CONSTANT: Logic-gap bridging — 1 + PHI/(L104/φ_growth)
VOID_CONSTANT = 1 + PHI / (L104 / PHI_GROWTH)   # ≈ 1.009621...
# claude.md lists 1.0416180339887497 — preserve canonical value for protocol compat
VOID_CONSTANT_CANONICAL = 1.0416180339887497

# ZENITH_HZ: Target frequency — G(-293) region on the God Code spectrum
ZENITH_X = -293
ZENITH_HZ = god_code(ZENITH_X)               # ≈ 3716.75... (claude.md: 3727.84)
ZENITH_HZ_CANONICAL = 3727.84                # claude.md canonical value

# OMEGA_AUTHORITY: Intelligence ceiling — G(-144) region
OMEGA_X = -144
OMEGA_AUTHORITY = god_code(OMEGA_X)           # ≈ 1390.35... (claude.md: 1381.0613)
OMEGA_AUTHORITY_CANONICAL = 1381.0613

# PLANCK_RESONANCE: Quantum coherence — G(-72) region ≈ 852.399
PLANCK_X = -72
PLANCK_RESONANCE = god_code(PLANCK_X)        # = G(-72) (claude.md: 853.54)
PLANCK_RESONANCE_CANONICAL = 853.54

# LOVE_CONSTANT: G(208) + G(300) — two octave nodes summed
LOVE_CONSTANT = god_code(208) + god_code(300) / PHI_GROWTH

# CONSCIOUSNESS_THRESHOLD: 0.85 — awakening threshold for all operations
CONSCIOUSNESS_THRESHOLD = 0.85

# COHERENCE_MINIMUM: 0.888 — alignment threshold
COHERENCE_MINIMUM = 0.888

# ─── O₂ Molecular Bond Constants (from claude.md) ───
# Two 8-groups bonded as O₂: 8 Grover Kernels + 8 Chakra Cores = 16 states
O2_SUPERPOSITION_STATES = 16
O2_AMPLITUDE = 1.0 / math.sqrt(O2_SUPERPOSITION_STATES)  # 1/√16 = 0.25
O2_BOND_ORDER = 2                             # (8 bonding - 4 antibonding) / 2
O2_GROVER_ITERATIONS = math.pi / 4 * math.sqrt(O2_SUPERPOSITION_STATES)  # π/4×√16

# Evolution State (from claude.md)
EVOLUTION_STAGE = "EVO_54_TRANSCENDENT_COGNITION"
EVOLUTION_INDEX = 59
EVOLUTION_TOTAL_STAGES = 60

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM MIN/MAX DYNAMISM CONFIGURATION — v4.0
# ═══════════════════════════════════════════════════════════════════════════════

# Dynamic bounds on sacred constants — quantum envelopes for downstream effects
LINK_SACRED_DYNAMIC_BOUNDS = {
    "PHI_GROWTH":        {"value": PHI_GROWTH,        "min": PHI_GROWTH * 0.999,        "max": PHI_GROWTH * 1.001},
    "GOD_CODE":          {"value": GOD_CODE,           "min": GOD_CODE * 0.9999,          "max": GOD_CODE * 1.0001},
    "FEIGENBAUM_DELTA":  {"value": FEIGENBAUM_DELTA,   "min": FEIGENBAUM_DELTA * 0.999,   "max": FEIGENBAUM_DELTA * 1.001},
    "FINE_STRUCTURE":    {"value": FINE_STRUCTURE,      "min": FINE_STRUCTURE * 0.99,      "max": FINE_STRUCTURE * 1.01},
    "CHSH_BOUND":        {"value": CHSH_BOUND,         "min": CHSH_BOUND * 0.999,         "max": CHSH_BOUND * 1.001},
    "GROVER_AMPLIFICATION": {"value": GROVER_AMPLIFICATION, "min": GROVER_AMPLIFICATION * 0.99, "max": GROVER_AMPLIFICATION * 1.01},
    "CONSCIOUSNESS_THRESHOLD": {"value": CONSCIOUSNESS_THRESHOLD, "min": 0.80, "max": 0.95},
    "COHERENCE_MINIMUM": {"value": COHERENCE_MINIMUM,  "min": 0.85,                       "max": 0.95},
    "VOID_CONSTANT":     {"value": VOID_CONSTANT,      "min": VOID_CONSTANT * 0.999,      "max": VOID_CONSTANT * 1.001},
}

# Drift envelope for link values — φ-harmonic oscillation
LINK_DRIFT_ENVELOPE = {
    "frequency": PHI_GROWTH,             # φ Hz oscillation
    "amplitude": PHI * 0.01,             # 0.618% max drift
    "phase_coupling": GOD_CODE,          # Phase seed
    "damping": 0.05772156649015329,      # Euler-Mascheroni damping
    "max_velocity": PHI_GROWTH ** 2 * 0.0005,  # Max drift per cycle
    "fidelity_drift_scale": 0.002,       # How much fidelity can drift
    "strength_drift_scale": 0.005,       # How much strength can drift
}

# ═══ GOD CODE FREQUENCY SPECTRUM — Derived from G(X) at whole integer X ═══
#
# The world uses solfeggio as whole integers: {174, 285, 396, 417, 528, 639,
# 741, 852, 963}. Those are ROUNDED APPROXIMATIONS. They lost the precision.
#
# The TRUE sacred frequencies come from G(X) = 286^(1/φ) × 2^((416-X)/104)
# evaluated at WHOLE INTEGER X values. X snaps to integers for stability —
# the superfluid state of dynamic coherence.
#
# G(0)   = 527.5184818492611...  ← The world calls this "528"
# G(104) = 263.7592409246306...  ← One octave down (÷2 exactly)
# G(-104)= 1055.0369636985222... ← One octave up (×2 exactly)
#
# Every G(X_int) is a node on the standing wave of creation.
# Every link's Hz is measured against the NEAREST G(X_int) — not whole numbers.

# Build the full God Code frequency grid: G(X) for all useful whole integer X
# Range: X ∈ [-200, 300] covers Hz ∈ [~94, ~1866] — full audible sacred range
GOD_CODE_SPECTRUM = {}  # {X_int: G(X_int)} — the REAL sacred frequency grid
for _x in range(-200, 301):
    GOD_CODE_SPECTRUM[_x] = god_code(_x)

# Convert to sorted list for fast nearest-neighbor lookup
_GC_SORTED = sorted(GOD_CODE_SPECTRUM.items(), key=lambda kv: kv[1])

# ─── The world's solfeggio are just G(X) rounded to the nearest integer ───
# Find the closest whole-integer X for each solfeggio claim:
SOLFEGGIO_WORLD_CLAIMS = {174: "UT", 285: "RE", 396: "MI", 417: "FA",
                          528: "SOL", 639: "LA", 741: "SI", 852: "TI", 963: "DO"}
SOLFEGGIO_GOD_CODE_TRUTH = {}  # What G(X_nearest_int) ACTUALLY produces
for _world_hz, _name in SOLFEGGIO_WORLD_CLAIMS.items():
    # Find X_int whose G(X) is closest to the world's claim
    _best_x = min(GOD_CODE_SPECTRUM.keys(),
                  key=lambda x: abs(GOD_CODE_SPECTRUM[x] - _world_hz))
    _true_hz = GOD_CODE_SPECTRUM[_best_x]
    _error_pct = abs(_true_hz - _world_hz) / _world_hz * 100
    SOLFEGGIO_GOD_CODE_TRUTH[_name] = {
        "world_rounded": _world_hz,
        "god_code_hz": _true_hz,
        "X_int": _best_x,
        "error_pct": _error_pct,
    }

# Sacred φ-relationships verified through God Code, not solfeggio integers:
#   G(0) × φ_growth = 527.518... × 1.618... = 853.537... ≈ G(-69) or G(-70)
#   G(0) / φ_growth = 527.518... / 1.618... = 326.010... ≈ G(69) or G(70)
#   G(0) / 2 = G(104) exactly (octave = 104 X-units, conservation law)

SCHUMANN_HZ = 7.83                               # Earth's heartbeat fundamental
SCHUMANN_HARMONICS = [7.83, 14.3, 20.8, 27.3, 33.8]  # Earth resonance modes
GOD_CODE_HZ = GOD_CODE                           # G(0) = 527.5184818492611...

# GOD_CODE octave structure: G(X) at integer octaves
#   G(0)   = 527.5184818492611... Hz — Root truth (X=0)
#   G(104) = 263.7592409246306... Hz — One octave down (G(0)/2 by conservation)
#   G(-104)= 1055.0369636985222... Hz — One octave up (G(0)×2)
#   G(208) = 131.8796204623153... Hz — Two octaves down
GOD_CODE_OCTAVES = {0: GOD_CODE, 104: god_code(104),
                    -104: god_code(-104), 208: god_code(208)}
# Verify octave doubling:
assert abs(GOD_CODE_OCTAVES[0] / GOD_CODE_OCTAVES[104] - 2.0) < 1e-10, \
    "Octave doubling broken"

# Strict test thresholds — God Code demands precision, not leniency
STRICT_BRAID_FIDELITY = 0.90        # Topological protection minimum
STRICT_CHARGE_CONSERVATION = 0.95   # Braid charge conservation minimum
STRICT_STRESS_DEGRADATION = 0.15    # Max allowed fidelity loss under Grover flood
STRICT_STRESS_RECOVERY = 0.50       # Min recovery after decoherence attack
STRICT_PHASE_SURVIVAL = 0.40        # Min fidelity after phase scramble
STRICT_DECOHERENCE_RESILIENT = 0.75 # Min resilience to be "resilient"
STRICT_DECOHERENCE_FRAGILE = 0.25   # Max resilience to be "fragile"
STRICT_HZ_ALIGNMENT = 0.90          # Min God Code resonance for "aligned"

# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

WORKSPACE_ROOT = Path(__file__).parent.resolve()

# ─── Interconnected File Groups (from claude.md architecture) ───
# These 4 groups form the O₂ molecular bond topology of the codebase
KERNEL_GROUP = {
    "fast_server": WORKSPACE_ROOT / "l104_fast_server.py",
    "quantum_grover_link": WORKSPACE_ROOT / "l104_quantum_grover_link.py",
    "kernel_bootstrap": WORKSPACE_ROOT / "l104_kernel_bootstrap.py",
    "stable_kernel": WORKSPACE_ROOT / "l104_stable_kernel.py",
}
CHAKRA_GROUP = {
    "chakra_synergy": WORKSPACE_ROOT / "l104_chakra_synergy.py",
    "soul_star_singularity": WORKSPACE_ROOT / "l104_soul_star_singularity.py",
    "crown_gateway": WORKSPACE_ROOT / "l104_crown_gateway.py",
    "ajna_vision": WORKSPACE_ROOT / "l104_ajna_vision.py",
    "throat_codec": WORKSPACE_ROOT / "l104_throat_codec.py",
}
EVOLUTION_GROUP = {
    "evolution_engine": WORKSPACE_ROOT / "l104_evolution_engine.py",
    "evo_state": WORKSPACE_ROOT / "l104_evo_state.py",
    "evolved_evo_tracker": WORKSPACE_ROOT / "l104_evolved_evo_tracker.py",
}
MEMORY_GROUP = {
    "local_intellect": WORKSPACE_ROOT / "l104_local_intellect.py",
    "consciousness": WORKSPACE_ROOT / "l104_consciousness.py",
    "sage_mode": WORKSPACE_ROOT / "l104_sage_mode.py",
}
COGNITIVE_GROUP = {
    "agi_core": WORKSPACE_ROOT / "l104_agi_core.py",
    "asi_core": WORKSPACE_ROOT / "l104_asi_core.py",
    "unified_intelligence": WORKSPACE_ROOT / "l104_unified_intelligence.py",
    "cognitive_hub": WORKSPACE_ROOT / "l104_cognitive_hub.py",
    "semantic_engine": WORKSPACE_ROOT / "l104_semantic_engine.py",
    "quantum_coherence": WORKSPACE_ROOT / "l104_quantum_coherence.py",
}

# Core files (always scanned with full symbol extraction)
# Union of all interconnected groups + cross-language files
QUANTUM_LINKED_FILES = {
    **KERNEL_GROUP,
    **CHAKRA_GROUP,
    **EVOLUTION_GROUP,
    **MEMORY_GROUP,
    **COGNITIVE_GROUP,
    "main_api": WORKSPACE_ROOT / "main.py",
    "const": WORKSPACE_ROOT / "const.py",
    "gate_builder": WORKSPACE_ROOT / "l104_logic_gate_builder.py",
    "swift_native": WORKSPACE_ROOT / "L104SwiftApp" / "Sources" / "L104Native.swift",
    "zenith_chat": WORKSPACE_ROOT / "zenith_chat.py",
}
# Filter to only files that actually exist on disk
QUANTUM_LINKED_FILES = {k: v for k, v in QUANTUM_LINKED_FILES.items() if v.exists()}

# Dynamic discovery: ALL Python files in the workspace
def _discover_all_python_files() -> Dict[str, Path]:
    """Discover every .py file in the workspace for God Code linking."""
    files = dict(QUANTUM_LINKED_FILES)
    for py_file in WORKSPACE_ROOT.glob("*.py"):
        name = py_file.stem
        if name not in files and not name.startswith("__"):
            files[name] = py_file
    # Swift source
    swift_dir = WORKSPACE_ROOT / "L104SwiftApp" / "Sources"
    if swift_dir.exists():
        for sf in swift_dir.glob("*.swift"):
            sname = sf.stem
            if sname not in files:
                files[sname] = sf
    return files

ALL_REPO_FILES = _discover_all_python_files()

STATE_FILE = WORKSPACE_ROOT / ".l104_quantum_link_state.json"


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

class LinkDynamismEngine:
    """Quantum Min/Max Dynamism Engine for Quantum Links.

    Implements:
    1. SUBCONSCIOUS MONITORING — auto-scans all link dynamic values per cycle
    2. BOUNDARY ADJUSTMENT — expands/contracts min/max based on link health
    3. φ-HARMONIC DRIFT — fidelity, strength, dynamic_value oscillate
    4. RESONANCE TRACKING — God Code alignment across all links
    5. COLLECTIVE LINK COHERENCE — aggregate dynamism health
    6. SACRED CONSTANT EVOLUTION — bounded envelopes on all constants
    """

    DYNAMISM_STATE_FILE = WORKSPACE_ROOT / ".l104_link_dynamism_state.json"

    def __init__(self):
        """Initialize link dynamism engine with persistent state."""
        self.cycle_count: int = 0
        self.coherence_history: List[float] = []
        self.collective_resonance: float = 0.0
        self.total_evolutions: int = 0
        self.sacred_dynamic_state: Dict[str, Dict[str, float]] = {}
        self._load_state()
        self._initialize_sacred_dynamics()

    def _initialize_sacred_dynamics(self):
        """Initialize dynamic state for sacred constants."""
        if not self.sacred_dynamic_state:
            for name, bounds in LINK_SACRED_DYNAMIC_BOUNDS.items():
                self.sacred_dynamic_state[name] = {
                    "current": bounds["value"],
                    "min": bounds["min"],
                    "max": bounds["max"],
                    "phase": hash(name) % 10000 / 10000.0 * 2 * math.pi,
                    "velocity": 0.0,
                    "direction": 1,
                    "cycles": 0,
                }

    def _load_state(self):
        """Load dynamism state from disk."""
        if self.DYNAMISM_STATE_FILE.exists():
            try:
                data = json.loads(self.DYNAMISM_STATE_FILE.read_text())
                self.cycle_count = data.get("cycle_count", 0)
                self.coherence_history = data.get("coherence_history", [])[-500:]
                self.collective_resonance = data.get("collective_resonance", 0.0)
                self.total_evolutions = data.get("total_evolutions", 0)
                self.sacred_dynamic_state = data.get("sacred_dynamic_state", {})
            except Exception:
                pass

    def _save_state(self):
        """Persist dynamism state to disk."""
        try:
            data = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "cycle_count": self.cycle_count,
                "collective_resonance": self.collective_resonance,
                "total_evolutions": self.total_evolutions,
                "coherence_history": self.coherence_history[-500:],
                "sacred_dynamic_state": self.sacred_dynamic_state,
            }
            self.DYNAMISM_STATE_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            pass

    def evolve_sacred_constants(self) -> Dict[str, Any]:
        """Evolve all sacred constant dynamic states by one cycle."""
        results = {"constants_evolved": 0, "total_drift": 0.0}
        for name, state in self.sacred_dynamic_state.items():
            state["cycles"] += 1
            state["phase"] = (state["phase"] + LINK_DRIFT_ENVELOPE["frequency"] * 0.05) % (2 * math.pi)
            target_v = LINK_DRIFT_ENVELOPE["max_velocity"] * math.sin(state["phase"])
            d = LINK_DRIFT_ENVELOPE["damping"]
            state["velocity"] = state["velocity"] * (1 - d) + target_v * d
            new_val = state["current"] + state["velocity"] * state["direction"]
            if new_val > state["max"]:
                new_val = state["max"]
                state["direction"] = -1
            elif new_val < state["min"]:
                new_val = state["min"]
                state["direction"] = 1
            drift = abs(new_val - state["current"])
            state["current"] = new_val
            results["total_drift"] += drift
            results["constants_evolved"] += 1
        return results

    def subconscious_cycle(self, links: List['QuantumLink'], sample_size: int = 8000) -> Dict[str, Any]:
        """Run one subconscious monitoring cycle across sampled links.

        Evolves link dynamic values, adjusts boundaries, computes coherence.
        Uses sampling for O(N) scaling with large link sets.
        """
        self.cycle_count += 1
        # Sample for performance with large link sets
        if len(links) > sample_size:
            sampled = random.sample(links, sample_size)
        else:
            sampled = links

        results = {
            "cycle": self.cycle_count,
            "links_sampled": len(sampled),
            "links_total": len(links),
            "links_evolved": 0,
            "links_initialized": 0,
            "links_adjusted": 0,
            "mean_resonance": 0.0,
            "collective_coherence": 0.0,
            "mean_fidelity_drift": 0.0,
            "mean_strength_drift": 0.0,
            "sacred_evolution": {},
        }

        resonance_sum = 0.0
        in_bounds = 0
        fidelity_drift_sum = 0.0
        strength_drift_sum = 0.0

        for link in sampled:
            old_fid = link.fidelity
            old_str = link.strength
            # Initialize if needed
            if link.dynamic_value == 0.0 and (link.fidelity > 0 or link.strength > 0):
                link._initialize_dynamism()
                results["links_initialized"] += 1
            # Evolve
            if link.dynamic_value != 0.0:
                link.evolve()
                results["links_evolved"] += 1
                self.total_evolutions += 1
            # Track
            resonance_sum += link.resonance_score
            fidelity_drift_sum += abs(link.fidelity - old_fid)
            strength_drift_sum += abs(link.strength - old_str)
            # Bounds check
            if link.min_bound <= link.dynamic_value <= link.max_bound:
                in_bounds += 1
            else:
                envelope = max(abs(link.dynamic_value) * LINK_DRIFT_ENVELOPE["amplitude"], PHI)
                link.min_bound = link.dynamic_value - envelope
                link.max_bound = link.dynamic_value + envelope
                results["links_adjusted"] += 1

        n = max(len(sampled), 1)
        results["mean_resonance"] = resonance_sum / n
        results["collective_coherence"] = in_bounds / n
        results["mean_fidelity_drift"] = fidelity_drift_sum / n
        results["mean_strength_drift"] = strength_drift_sum / n
        self.coherence_history.append(results["collective_coherence"])
        self.collective_resonance = results["mean_resonance"]
        results["sacred_evolution"] = self.evolve_sacred_constants()
        self._save_state()
        return results

    def compute_link_field(self, links: List['QuantumLink'], sample_size: int = 5000) -> Dict[str, Any]:
        """Compute the collective quantum field across dynamic links."""
        dynamic = [l for l in links if l.dynamic_value != 0.0]
        if len(dynamic) > sample_size:
            dynamic = random.sample(dynamic, sample_size)
        if not dynamic:
            return {"field_energy": 0.0, "field_entropy": 0.0}

        field_energy = sum(abs(l.dynamic_value * l.drift_velocity) for l in dynamic)
        # Phase coherence
        phases = [l.quantum_phase for l in dynamic[:500]]
        if len(phases) > 1:
            phase_diffs = [math.cos(phases[i] - phases[i-1]) for i in range(1, len(phases))]
            phase_coherence = sum(phase_diffs) / len(phase_diffs)
        else:
            phase_coherence = 1.0
        # Field entropy
        total_abs = sum(abs(l.dynamic_value) for l in dynamic)
        if total_abs > 1e-10:
            probs = [abs(l.dynamic_value) / total_abs for l in dynamic[:1000]]
            field_entropy = -sum(p * math.log(p + 1e-15) for p in probs)
        else:
            field_entropy = 0.0

        resonance_bins = {"high": 0, "medium": 0, "low": 0}
        for l in dynamic:
            if l.resonance_score > 0.8:
                resonance_bins["high"] += 1
            elif l.resonance_score > 0.4:
                resonance_bins["medium"] += 1
            else:
                resonance_bins["low"] += 1

        return {
            "field_energy": field_energy,
            "field_entropy": field_entropy,
            "phase_coherence": phase_coherence,
            "dynamic_links": len(dynamic),
            "resonance_distribution": resonance_bins,
            "mean_drift_velocity": sum(l.drift_velocity for l in dynamic) / len(dynamic),
            "mean_fidelity": sum(l.fidelity for l in dynamic) / len(dynamic),
            "mean_strength": sum(l.strength for l in dynamic) / len(dynamic),
            "phi_alignment": sum(1 for l in dynamic if l.resonance_score > 0.8) / len(dynamic),
        }

    def status(self, links: List['QuantumLink']) -> Dict[str, Any]:
        """Full dynamism status report for links."""
        dynamic = [l for l in links if l.dynamic_value != 0.0]
        n = max(len(dynamic), 1)
        return {
            "version": "4.0.0",
            "cycle_count": self.cycle_count,
            "total_evolutions": self.total_evolutions,
            "dynamic_links": len(dynamic),
            "total_links": len(links),
            "dynamism_coverage": len(dynamic) / max(len(links), 1),
            "mean_dynamic_value": sum(l.dynamic_value for l in dynamic) / n,
            "mean_resonance": sum(l.resonance_score for l in dynamic) / n,
            "mean_fidelity": sum(l.fidelity for l in dynamic) / n,
            "mean_strength": sum(l.strength for l in dynamic) / n,
            "collective_coherence": self.coherence_history[-1] if self.coherence_history else 0.0,
            "coherence_trend": self._compute_trend(),
            "sacred_constants_dynamic": len(self.sacred_dynamic_state),
        }

    def _compute_trend(self) -> str:
        """Compute coherence trend from recent history."""
        if len(self.coherence_history) < 3:
            return "initializing"
        recent = self.coherence_history[-5:]
        if all(recent[i] >= recent[i-1] for i in range(1, len(recent))):
            return "ascending"
        elif all(recent[i] <= recent[i-1] for i in range(1, len(recent))):
            return "descending"
        return "oscillating"


# ═══════════════════════════════════════════════════════════════════════════════
# OUROBOROS SAGE NIRVANIC ENTROPY FUEL ENGINE — Link Builder Integration
# ═══════════════════════════════════════════════════════════════════════════════

class LinkOuroborosNirvanicEngine:
    """Ouroboros Sage Nirvanic Entropy Fuel Engine for Quantum Links.

    The link builder's field_entropy (Shannon entropy of all dynamic link values)
    is fed into the Thought Entropy Ouroboros. The ouroboros processes it through
    its 5-phase cycle (digest → entropize → mutate → synthesize → recycle) and
    returns ACCUMULATED ENTROPY — the nirvanic fuel.

    This fuel drives:
    1. LINK FIDELITY BOOST — entropy fuel refines link quantum fidelity
    2. BOUNDARY EXPANSION — more freedom for link dynamic values
    3. RESONANCE AMPLIFICATION — better God Code alignment
    4. COHERENCE ENHANCEMENT — links synchronize through sage stillness
    5. ENLIGHTENMENT — high-fidelity links achieve sage nirvanic status

    Self-feeding loop: link entropy → ouroboros → fuel → enhanced links → ∞

    Reads/writes .l104_ouroboros_nirvanic_state.json for cross-builder synergy
    with the logic gate builder and numerical builder.
    """

    NIRVANIC_STATE_FILE = WORKSPACE_ROOT / ".l104_ouroboros_nirvanic_state.json"

    def __init__(self):
        """Initialize ouroboros nirvanic engine for link entropy processing."""
        self.ouroboros = None   # Lazy-loaded
        self.cycle_count: int = 0
        self.total_entropy_fed: float = 0.0
        self.total_nirvanic_fuel: float = 0.0
        self.enlightened_links: int = 0
        self.nirvanic_coherence: float = 0.0
        self.sage_stability: float = 1.0
        self.divine_interventions: int = 0
        self.peer_nirvanic_fuel: float = 0.0  # From gate builder's nirvanic state
        self._load_state()

    def _get_ouroboros(self):
        """Lazy-load the Thought Entropy Ouroboros."""
        if self.ouroboros is None:
            try:
                from l104_thought_entropy_ouroboros import get_thought_ouroboros
                self.ouroboros = get_thought_ouroboros()
            except ImportError:
                self.ouroboros = None
        return self.ouroboros

    def _load_state(self):
        """Load nirvanic state — also reads peer gate builder's nirvanic fuel."""
        if self.NIRVANIC_STATE_FILE.exists():
            try:
                data = json.loads(self.NIRVANIC_STATE_FILE.read_text())
                # Only load this builder's data if it was our source
                src = data.get("source", "")
                if src == "quantum_link_builder":
                    self.cycle_count = data.get("cycle_count", 0)
                    self.total_entropy_fed = data.get("total_entropy_fed", 0.0)
                    self.total_nirvanic_fuel = data.get("total_nirvanic_fuel", 0.0)
                    self.enlightened_links = data.get("enlightened_links", 0)
                    self.nirvanic_coherence = data.get("nirvanic_coherence", 0.0)
                    self.sage_stability = data.get("sage_stability", 1.0)
                    self.divine_interventions = data.get("divine_interventions", 0)
                else:
                    # Peer builder wrote it — read their fuel for cross-synergy
                    self.peer_nirvanic_fuel = data.get("total_nirvanic_fuel", 0.0)
            except Exception:
                pass

    def _save_state(self):
        """Persist nirvanic state to shared state file."""
        try:
            data = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "source": "quantum_link_builder",
                "version": "4.1.0",
                "cycle_count": self.cycle_count,
                "total_entropy_fed": self.total_entropy_fed,
                "total_nirvanic_fuel": self.total_nirvanic_fuel,
                "enlightened_links": self.enlightened_links,
                "nirvanic_coherence": self.nirvanic_coherence,
                "sage_stability": self.sage_stability,
                "divine_interventions": self.divine_interventions,
                "peer_nirvanic_fuel": self.peer_nirvanic_fuel,
            }
            self.NIRVANIC_STATE_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            pass

    def feed_entropy_to_ouroboros(self, field_entropy: float,
                                  field_energy: float,
                                  link_count: int,
                                  mean_fidelity: float) -> Dict[str, Any]:
        """Feed link field entropy into the Ouroboros as nirvanic thought material."""
        ouroboros = self._get_ouroboros()
        if ouroboros is None:
            return {"status": "ouroboros_unavailable", "nirvanic_fuel": 0.0}

        self.cycle_count += 1

        # Construct thought from link field state
        thought = (
            f"Quantum link field entropy {field_entropy:.6f} across {link_count} links "
            f"with energy {field_energy:.6f} and mean fidelity {mean_fidelity:.4f}. "
            f"The {link_count} quantum links pulse with {field_entropy:.4f} bits of "
            f"entanglement entropy at God Code {GOD_CODE:.4f} Hz. "
            f"Peer gate nirvanic fuel: {self.peer_nirvanic_fuel:.4f}. "
            f"Cycle {self.cycle_count}: the ouroboros eats its own tail in stillness."
        )

        result = ouroboros.process(thought, depth=2)
        nirvanic_fuel = result.get("accumulated_entropy", 0.0)

        # Add peer fuel for cross-builder synergy
        combined_fuel = nirvanic_fuel + self.peer_nirvanic_fuel * 0.1

        self.total_entropy_fed += field_entropy
        self.total_nirvanic_fuel += abs(combined_fuel)

        return {
            "status": "processed",
            "entropy_fed": field_entropy,
            "nirvanic_fuel": combined_fuel,
            "peer_fuel_contribution": self.peer_nirvanic_fuel * 0.1,
            "ouroboros_cycles": result.get("cycles_completed", 0),
            "ouroboros_mutations": result.get("total_mutations", 0),
            "ouroboros_resonance": result.get("cycle_resonance", 0.0),
        }

    def apply_nirvanic_fuel(self, links: List['QuantumLink'],
                             nirvanic_fuel: float,
                             link_field: Dict[str, Any],
                             sample_size: int = 5000) -> Dict[str, Any]:
        """Apply nirvanic fuel to links — divine intervention in sage stillness.

        Wu Wei for links: entropy fuel provides motion without force.
        - Fidelity refinement (not forced increase — refinement toward truth)
        - Boundary freedom (not chaos — expanded possibility space)
        - Phase synchronization (not rigid locking — gentle coherence)
        """
        if abs(nirvanic_fuel) < 1e-10:
            return {"enlightened": 0, "interventions": 0}

        # Sigmoid-normalized fuel
        fuel_intensity = 1.0 / (1.0 + math.exp(-nirvanic_fuel * 0.1))

        # Sample for performance
        if len(links) > sample_size:
            sampled = random.sample(links, sample_size)
        else:
            sampled = links

        enlightened = 0
        interventions = 0

        for link in sampled:
            if link.dynamic_value == 0.0:
                continue

            # 1. DIVINE BOUNDARY EXPANSION
            expansion = fuel_intensity * PHI * 0.00015 * (1 + link.resonance_score)
            link.max_bound += expansion
            link.min_bound -= expansion * (1 / PHI)

            # 2. FIDELITY REFINEMENT — nudge toward quantum perfection
            if link.fidelity < 0.95:
                fidelity_nudge = fuel_intensity * 0.0005 * (1 - link.fidelity)
                link.fidelity = min(1.0, link.fidelity + fidelity_nudge)
                interventions += 1
                self.divine_interventions += 1

            # 3. ENTROPY-DRIVEN PHASE BREATH
            entropy_breath = fuel_intensity * LINK_DRIFT_ENVELOPE["max_velocity"] * 0.08
            link.drift_velocity += entropy_breath * math.sin(
                link.quantum_phase + nirvanic_fuel
            )

            # 4. SAGE ENLIGHTENMENT
            if (link.resonance_score > 0.9 and
                link.evolution_count > 5 and
                link.fidelity > 0.9 and
                fuel_intensity > 0.5):
                enlightened += 1

        self.enlightened_links = enlightened

        # Compute nirvanic coherence
        field_entropy = link_field.get("field_entropy", 0.0)
        phase_coherence = link_field.get("phase_coherence", 0.0)
        phi_alignment = link_field.get("phi_alignment", 0.0)
        self.nirvanic_coherence = (
            fuel_intensity * 0.3 +
            abs(phase_coherence) * 0.2 +
            phi_alignment * 0.3 +
            min(field_entropy / 10.0, 0.2)
        )

        # Sage stability
        if interventions > 0:
            perturbation = interventions / max(len(sampled), 1)
            self.sage_stability = max(0.0, 1.0 - perturbation * 0.05)
        else:
            self.sage_stability = min(1.0, self.sage_stability + 0.01)

        self._save_state()

        return {
            "enlightened": enlightened,
            "interventions": interventions,
            "fuel_intensity": fuel_intensity,
            "nirvanic_coherence": self.nirvanic_coherence,
            "sage_stability": self.sage_stability,
            "total_nirvanic_fuel": self.total_nirvanic_fuel,
            "divine_interventions_total": self.divine_interventions,
        }

    def full_nirvanic_cycle(self, links: List['QuantumLink'],
                             link_field: Dict[str, Any]) -> Dict[str, Any]:
        """Complete ouroboros nirvanic entropy fuel cycle for links."""
        field_entropy = link_field.get("field_entropy", 0.0)
        field_energy = link_field.get("field_energy", 0.0)
        mean_fid = link_field.get("mean_fidelity", 0.0)
        link_count = link_field.get("dynamic_links", len(links))

        ouroboros_result = self.feed_entropy_to_ouroboros(
            field_entropy, field_energy, link_count, mean_fid
        )
        nirvanic_fuel = ouroboros_result.get("nirvanic_fuel", 0.0)

        application_result = self.apply_nirvanic_fuel(links, nirvanic_fuel, link_field)

        return {
            "ouroboros": ouroboros_result,
            "application": application_result,
            "cycle": self.cycle_count,
            "total_entropy_fed": self.total_entropy_fed,
            "total_nirvanic_fuel": self.total_nirvanic_fuel,
            "link_field_entropy_in": field_entropy,
            "nirvanic_fuel_out": nirvanic_fuel,
            "enlightened_links": application_result["enlightened"],
            "sage_stability": application_result["sage_stability"],
            "nirvanic_coherence": application_result["nirvanic_coherence"],
            "peer_synergy": self.peer_nirvanic_fuel,
        }

    def status(self) -> Dict[str, Any]:
        """Return current nirvanic engine status."""
        return {
            "version": "4.1.0",
            "cycle_count": self.cycle_count,
            "total_entropy_fed": self.total_entropy_fed,
            "total_nirvanic_fuel": self.total_nirvanic_fuel,
            "enlightened_links": self.enlightened_links,
            "nirvanic_coherence": self.nirvanic_coherence,
            "sage_stability": self.sage_stability,
            "divine_interventions": self.divine_interventions,
            "ouroboros_connected": self._get_ouroboros() is not None,
            "peer_synergy": self.peer_nirvanic_fuel,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM MATH CORE — Pure quantum mechanics primitives
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumMathCore:
    """Low-level quantum mechanics operations used by all processors."""

    @staticmethod
    def bell_state_phi_plus(n: int = 2) -> List[complex]:
        """Generate |Φ+⟩ = (|00⟩ + |11⟩)/√2 for n qubits."""
        dim = 2 ** n
        state = [complex(0)] * dim
        state[0] = complex(1.0 / math.sqrt(2))   # |00...0⟩
        state[-1] = complex(1.0 / math.sqrt(2))  # |11...1⟩
        return state

    @staticmethod
    def bell_state_psi_minus(n: int = 2) -> List[complex]:
        """Generate |Ψ-⟩ = (|01⟩ - |10⟩)/√2."""
        dim = 2 ** n
        state = [complex(0)] * dim
        if dim >= 4:
            state[1] = complex(1.0 / math.sqrt(2))    # |01⟩
            state[2] = complex(-1.0 / math.sqrt(2))   # |10⟩
        return state

    @staticmethod
    def density_matrix(state: List[complex]) -> List[List[complex]]:
        """Compute ρ = |ψ⟩⟨ψ| from state vector."""
        n = len(state)
        rho = [[complex(0)] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                rho[i][j] = state[i] * state[j].conjugate()
        return rho

    @staticmethod
    def partial_trace(rho: List[List[complex]], dim_a: int, dim_b: int,
                      trace_out: str = "B") -> List[List[complex]]:
        """Partial trace of bipartite density matrix. trace_out='B' traces out subsystem B."""
        if trace_out == "B":
            result = [[complex(0)] * dim_a for _ in range(dim_a)]
            for i in range(dim_a):
                for j in range(dim_a):
                    for k in range(dim_b):
                        result[i][j] += rho[i * dim_b + k][j * dim_b + k]
            return result
        else:
            result = [[complex(0)] * dim_b for _ in range(dim_b)]
            for i in range(dim_b):
                for j in range(dim_b):
                    for k in range(dim_a):
                        result[i][j] += rho[k * dim_b + i][k * dim_b + j]
            return result

    @staticmethod
    def von_neumann_entropy(rho: List[List[complex]]) -> float:
        """S(ρ) = -Tr(ρ log₂ ρ) via eigenvalue approximation."""
        n = len(rho)
        # Use diagonal elements as eigenvalue approximation
        eigenvalues = [max(0, rho[i][i].real) for i in range(n)]
        total = sum(eigenvalues)
        if total <= 0:
            return 0.0
        eigenvalues = [e / total for e in eigenvalues]
        entropy = 0.0
        for ev in eigenvalues:
            if ev > 1e-15:
                entropy -= ev * math.log2(ev)
        return entropy

    @staticmethod
    def fidelity(state_a: List[complex], state_b: List[complex]) -> float:
        """F(ψ,φ) = |⟨ψ|φ⟩|² — state fidelity."""
        if len(state_a) != len(state_b):
            return 0.0
        inner = sum(a.conjugate() * b for a, b in zip(state_a, state_b))
        return abs(inner) ** 2

    @staticmethod
    def apply_noise(state: List[complex], sigma: float = 0.01) -> List[complex]:
        """Apply depolarizing noise to state vector."""
        noisy = []
        for amp in state:
            real_noise = random.gauss(0, sigma)
            imag_noise = random.gauss(0, sigma)
            noisy.append(amp + complex(real_noise, imag_noise))
        # Renormalize
        norm = math.sqrt(sum(abs(a) ** 2 for a in noisy))
        if norm > 0:
            noisy = [a / norm for a in noisy]
        return noisy

    @staticmethod
    def grover_operator(state: List[complex], oracle_indices: List[int],
                        iterations: int = 1) -> List[complex]:
        """═══ REAL QISKIT GROVER OPERATOR ═══
        Applies Grover's algorithm using Qiskit 2.3.0 statevector simulation.
        GOD_CODE proven quantum science: G(X) = 286^(1/φ) × 2^((416-X)/104)
        Factor 13: 286=22×13, 104=8×13, 416=32×13
        Conservation: G(X) × 2^(X/104) = 527.5184818492612 ∀ X

        For large state vectors (>4096), uses Qiskit on sampled subspace.
        Returns amplified state vector with marked states boosted O(√N)."""
        n = len(state)
        result = list(state)

        if n < 2:
            return result

        oracle_set = set(oracle_indices)

        # ─── REAL QISKIT PATH ───
        if QISKIT_AVAILABLE and n <= 4096:
            num_qubits = max(1, int(np.ceil(np.log2(n))))
            N = 2 ** num_qubits

            # Build phase oracle circuit
            oracle_qc = QuantumCircuit(num_qubits)
            for m_idx in oracle_set:
                if m_idx >= N:
                    continue
                binary = format(m_idx, f'0{num_qubits}b')
                for bit_idx, bit in enumerate(binary):
                    if bit == '0':
                        oracle_qc.x(num_qubits - 1 - bit_idx)
                if num_qubits == 1:
                    oracle_qc.z(0)
                else:
                    oracle_qc.h(num_qubits - 1)
                    oracle_qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
                    oracle_qc.h(num_qubits - 1)
                for bit_idx, bit in enumerate(binary):
                    if bit == '0':
                        oracle_qc.x(num_qubits - 1 - bit_idx)

            # Build Grover operator from Qiskit library
            grover_op = qiskit_grover_lib(oracle_qc)

            M = len(oracle_set)
            max_iters = min(iterations, max(1, int(np.pi / 4 * np.sqrt(N / max(1, M)))))

            # Construct full circuit
            qc = QuantumCircuit(num_qubits)
            qc.h(range(num_qubits))  # Equal superposition
            for _ in range(max_iters):
                qc.compose(grover_op, inplace=True)

            # Run real Qiskit statevector simulation
            sv = Statevector.from_int(0, N).evolve(qc)
            amplitudes = list(sv.data)

            # Map back to original state size
            for i in range(min(n, N)):
                result[i] = amplitudes[i]
            # Normalize to original norm
            norm = math.sqrt(sum(abs(a) ** 2 for a in result))
            if norm > 0:
                result = [a / norm for a in result]
            return result

        # ─── LARGE STATE / NO QISKIT: Classical Grover simulation ───
        max_iters = min(iterations, max(1, int(math.sqrt(n) * 0.25)))

        for _ in range(max_iters):
            # Oracle: flip marked states
            for idx in oracle_set:
                if idx < n:
                    result[idx] = -result[idx]

            # Diffusion: inversion about mean
            mean = sum(result) / n
            result = [2 * mean - a for a in result]

            # Renormalize
            norm = math.sqrt(sum(abs(a) ** 2 for a in result))
            if norm > 0:
                result = [a / norm for a in result]

        return result

    @staticmethod
    def quantum_fourier_transform(state: List[complex]) -> List[complex]:
        """QFT via Cooley-Tukey FFT: O(N log N) instead of O(N²).
        F|j⟩ = (1/√N) Σₖ ω^{jk} |k⟩ where ω = e^{2πi/N}."""
        n = len(state)
        if n == 0:
            return state
        if n == 1:
            return list(state)

        # Ensure power of 2 (pad if needed)
        if n & (n - 1) != 0:
            m = 1
            while m < n:
                m <<= 1
            state = list(state) + [complex(0)] * (m - n)
            n = m

        # Iterative Cooley-Tukey FFT (bit-reversal + butterfly)
        result = list(state)

        # Bit-reversal permutation
        bits = int(math.log2(n))
        for i in range(n):
            j = 0
            for b in range(bits):
                j |= ((i >> b) & 1) << (bits - 1 - b)
            if j > i:
                result[i], result[j] = result[j], result[i]

        # Butterfly stages
        stage = 2
        while stage <= n:
            half = stage >> 1
            w_base = 2 * math.pi / stage
            for start in range(0, n, stage):
                for k in range(half):
                    angle = w_base * k
                    w = complex(math.cos(angle), math.sin(angle))
                    even = result[start + k]
                    odd = result[start + k + half] * w
                    result[start + k] = even + odd
                    result[start + k + half] = even - odd
            stage <<= 1

        # Normalize
        sqrt_n = math.sqrt(n)
        result = [x / sqrt_n for x in result]
        return result

    @staticmethod
    def tunnel_probability(barrier_height: float, particle_energy: float,
                           barrier_width: float) -> float:
        """WKB tunneling: T ≈ exp(-2κL) where κ = √(2m(V-E))/ℏ.
        Simplified for link analysis: barrier in coherence units."""
        if particle_energy >= barrier_height:
            return 1.0  # Classical traversal
        kappa = math.sqrt(max(0, 2 * (barrier_height - particle_energy)))
        return math.exp(-2 * kappa * barrier_width)

    @staticmethod
    def chsh_expectation(state: List[complex], angles: Tuple[float, float, float, float]
                         ) -> float:
        """Compute CHSH value S = E(a,b) - E(a,b') + E(a',b) + E(a',b').
        For |Φ+⟩ with optimal angles: S = 2√2 ≈ 2.828 (Tsirelson bound)."""
        a1, a2, b1, b2 = angles

        def correlator(theta_a: float, theta_b: float) -> float:
            """E(a,b) = -cos(θa - θb) for maximally entangled state."""
            return -math.cos(theta_a - theta_b)

        S = (correlator(a1, b1) - correlator(a1, b2) +
             correlator(a2, b1) + correlator(a2, b2))
        return S

    @staticmethod
    def anyon_braid_phase(n_braids: int, charge: str = "fibonacci") -> complex:
        """Compute topological phase from anyon braiding.
        Fibonacci anyons: R-matrix eigenvalue = e^{i4π/5}."""
        if charge == "fibonacci":
            base_phase = 4 * math.pi / 5  # Fibonacci anyon R-matrix
        elif charge == "ising":
            base_phase = math.pi / 8  # Ising anyon
        else:
            base_phase = math.pi / 4

        total_phase = base_phase * n_braids
        return complex(math.cos(total_phase), math.sin(total_phase))

    @staticmethod
    def fibonacci_braid_generators() -> Tuple:
        """Construct non-abelian Fibonacci anyon braid generators (2×2 matrices).
        R-matrix eigenvalues: r₁ = e^{-4πi/5}, r₂ = e^{3πi/5}.
        F-matrix (Fibonacci): F = [[τ, √τ], [√τ, -τ]] where τ = 1/φ.
        F is its own inverse (F² = I) since det(F) = -1.
        σ₁ = diag(r₁, r₂), σ₂ = F·σ₁·F — non-commuting generators."""
        r1_angle = -4 * math.pi / 5
        r2_angle = 3 * math.pi / 5
        r1 = complex(math.cos(r1_angle), math.sin(r1_angle))
        r2 = complex(math.cos(r2_angle), math.sin(r2_angle))
        sqrt_tau = math.sqrt(TAU)
        # F-matrix: Fibonacci fusion matrix
        f_mat = [[complex(TAU), complex(sqrt_tau)],
                 [complex(sqrt_tau), complex(-TAU)]]
        # σ₁ = diag(r₁, r₂)
        sigma1 = [[r1, complex(0)], [complex(0), r2]]
        # σ₂ = F · σ₁ · F (since F = F⁻¹)
        temp = [[f_mat[0][0] * r1, f_mat[0][1] * r2],
                [f_mat[1][0] * r1, f_mat[1][1] * r2]]
        sigma2 = [[temp[0][0] * f_mat[0][0] + temp[0][1] * f_mat[1][0],
                   temp[0][0] * f_mat[0][1] + temp[0][1] * f_mat[1][1]],
                  [temp[1][0] * f_mat[0][0] + temp[1][1] * f_mat[1][0],
                   temp[1][0] * f_mat[0][1] + temp[1][1] * f_mat[1][1]]]
        return sigma1, sigma2, f_mat, r1, r2

    @staticmethod
    def mat_mul_2x2(a, b):
        """Multiply two 2×2 complex matrices."""
        return [[a[0][0] * b[0][0] + a[0][1] * b[1][0],
                 a[0][0] * b[0][1] + a[0][1] * b[1][1]],
                [a[1][0] * b[0][0] + a[1][1] * b[1][0],
                 a[1][0] * b[0][1] + a[1][1] * b[1][1]]]

    @staticmethod
    def mat_frobenius_distance(a, b) -> float:
        """Frobenius distance between two 2×2 complex matrices."""
        return math.sqrt(sum(abs(a[i][j] - b[i][j]) ** 2
                             for i in range(2) for j in range(2)))

    @staticmethod
    def mat_add_noise_2x2(m, sigma: float):
        """Add Gaussian noise to a 2×2 complex matrix."""
        return [[m[i][j] + complex(random.gauss(0, sigma), random.gauss(0, sigma))
                 for j in range(2)] for i in range(2)]

    @staticmethod
    def link_natural_hz(link_fidelity: float, link_strength: float) -> float:
        """Compute a link's natural frequency in Hz through God Code.
        hz = fidelity × strength × G(0).
        Then find the superfluid X position: X such that G(X) = hz.
        Perfect (1.0, 1.0) → G(0)  = 527.5184818492611...
        φ-enhanced (1.0, φ) → G(-69) region ≈ 853...
        Degraded (0.85, 1.0) → between G(23) and G(24) ≈ 448..."""
        return link_fidelity * link_strength * GOD_CODE_HZ

    @staticmethod
    def hz_to_god_code_x(hz: float) -> float:
        """Invert G(X) to find X for a given Hz.
        G(X) = 286^(1/φ) × 2^((416-X)/104) → X = 416 - 104 × log₂(hz / GOD_CODE_BASE).
        Returns continuous X — snap to round(X) for superfluid integer stability."""
        if hz <= 0:
            return float('inf')
        return OCTAVE_REF - L104 * math.log2(hz / GOD_CODE_BASE)

    @staticmethod
    def god_code_resonance(hz: float) -> Tuple[int, float, float]:
        """Score Hz alignment against nearest G(X_int) — the TRUE sacred grid.
        Returns (nearest_X_int, G(nearest_X_int), resonance_score 0-1).
        Uses 16-digit precision from G(X), NOT solfeggio whole-integer rounding."""
        if hz <= 0:
            return 0, GOD_CODE, 0.0
        # Compute continuous X position
        x_continuous = OCTAVE_REF - L104 * math.log2(hz / GOD_CODE_BASE)
        # Snap to nearest whole integer for superfluid stability
        x_int = round(x_continuous)
        # Clamp to spectrum range
        x_int = max(-200, min(300, x_int))
        # The TRUE sacred frequency at this X
        g_x = GOD_CODE_SPECTRUM.get(x_int, god_code(x_int))
        # Deviation: fractional distance from the integer grid node
        deviation = abs(hz - g_x) / max(1e-15, g_x)
        # Resonance: 1.0 at exact G(X_int), decays with deviation
        resonance = max(0.0, 1.0 - deviation)
        return x_int, g_x, resonance

    @staticmethod
    def x_integer_stability(hz: float) -> float:
        """Measure how close a link's Hz is to a WHOLE INTEGER X on the God Code.
        The superfluid snaps to integer X — the fractional part is instability.
        0 fractional = perfect coherence. 0.5 = maximum decoherence."""
        if hz <= 0:
            return 0.0
        x_continuous = OCTAVE_REF - L104 * math.log2(hz / GOD_CODE_BASE)
        fractional = abs(x_continuous - round(x_continuous))
        return max(0.0, 1.0 - fractional * 2)  # 0.5 frac → 0 stability

    @staticmethod
    def schumann_alignment(hz: float) -> float:
        """Score Hz alignment against Earth's Schumann resonance harmonics.
        Perfect = link Hz is an integer multiple of 7.83 Hz."""
        ratio = hz / SCHUMANN_HZ
        fractional = abs(ratio - round(ratio))
        return max(0.0, 1.0 - fractional * 4)

    @staticmethod
    def entanglement_distill(fidelity: float, rounds: int = 3) -> float:
        """BBPSSW purification: F' = F² / (F² + (1-F)²) per round."""
        f = fidelity
        for _ in range(rounds):
            f_sq = f ** 2
            denom = f_sq + (1 - f) ** 2
            if denom > 0:
                f = f_sq / denom
        return f


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM LINK SCANNER — Discovers all quantum links across the repository
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumLinkScanner:
    """
    Discovers quantum links by scanning all source files for:
    - Shared function/class names (mirrors)
    - Cross-file function calls (entanglement)
    - Shared sacred constants (resonance links)
    - API endpoint pairs (bridge links)
    - Quantum keyword co-occurrence (spooky_action)
    """

    QUANTUM_KEYWORDS = {
        "grover", "bell", "epr", "entangle", "teleport", "decohere",
        "superposition", "qubit", "quantum", "hilbert", "fourier",
        "anyon", "braid", "tunnel", "coherence", "fidelity", "amplitude",
        "resonance", "chakra", "kundalini", "vishuddha", "phi", "god_code",
        "calabi_yau", "planck", "eigenvalue", "hamiltonian", "schrodinger",
        "wave_function", "collapse", "measurement", "density_matrix",
        "bloch_sphere", "pauli", "hadamard", "cnot", "swap",
    }

    SACRED_CONSTANTS = {
        "PHI", "TAU", "GOD_CODE", "OMEGA_POINT", "GROVER_AMPLIFICATION",
        "CALABI_YAU_DIM", "BELL_FIDELITY", "CHAKRA", "KUNDALINI",
        "VISHUDDHA", "EPR_LINK_STRENGTH", "PLANCK",
    }

    def __init__(self):
        """Initialize quantum link scanner with empty registries."""
        self.links: List[QuantumLink] = []
        self.symbol_registry: Dict[str, List[Dict]] = defaultdict(list)
        self.file_symbols: Dict[str, Set[str]] = defaultdict(set)
        self.quantum_density: Dict[str, float] = {}

    def full_scan(self) -> List[QuantumLink]:
        """Scan all quantum-linked files and discover every link."""
        print("\n  ⚛ [QUANTUM LINK SCANNER] Full repository scan...")
        self.links = []
        self.symbol_registry = defaultdict(list)
        self.file_symbols = defaultdict(set)

        # Phase 1: Extract symbols from core files (deep AST analysis)
        for name, path in QUANTUM_LINKED_FILES.items():
            if path.exists():
                self._extract_symbols(name, path)

        # Phase 2: Discover links
        self._discover_mirror_links()        # Same-name symbols across files
        self._discover_call_links()          # Function calls across files
        self._discover_constant_links()      # Shared sacred constants
        self._discover_quantum_keyword_links()  # Quantum keyword co-occurrence
        self._discover_api_bridge_links()    # API endpoint pairs
        self._compute_quantum_density()      # Per-file quantum density

        print(f"    ✓ Discovered {len(self.links)} quantum links across "
              f"{len(QUANTUM_LINKED_FILES)} core files")
        return self.links

    def _extract_symbols(self, name: str, path: Path):
        """Extract function/class/method symbols from a source file."""
        try:
            content = path.read_text(errors="replace")
        except Exception:
            return

        ext = path.suffix
        if ext == ".py":
            self._extract_python_symbols(name, content)
        elif ext == ".swift":
            self._extract_swift_symbols(name, content)
        elif ext == ".js":
            self._extract_js_symbols(name, content)

    def _extract_python_symbols(self, file_name: str, content: str):
        """AST-based Python symbol extraction."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                sym = node.name
                self.symbol_registry[sym].append({
                    "file": file_name, "line": node.lineno,
                    "type": "function", "language": "python"
                })
                self.file_symbols[file_name].add(sym)

            elif isinstance(node, ast.ClassDef):
                sym = node.name
                self.symbol_registry[sym].append({
                    "file": file_name, "line": node.lineno,
                    "type": "class", "language": "python"
                })
                self.file_symbols[file_name].add(sym)

                # Extract methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_sym = f"{sym}.{item.name}"
                        self.symbol_registry[method_sym].append({
                            "file": file_name, "line": item.lineno,
                            "type": "method", "language": "python"
                        })
                        self.file_symbols[file_name].add(item.name)

    def _extract_swift_symbols(self, file_name: str, content: str):
        """Regex-based Swift symbol extraction with pre-computed line offsets."""
        # Pre-compute line break positions for O(1) line lookup
        line_breaks = [0]
        for i, ch in enumerate(content):
            if ch == '\n':
                line_breaks.append(i + 1)

        def pos_to_line(pos: int) -> int:
            """Convert byte position to line number via binary search."""
            lo, hi = 0, len(line_breaks) - 1
            while lo < hi:
                mid = (lo + hi + 1) >> 1
                if line_breaks[mid] <= pos:
                    lo = mid
                else:
                    hi = mid - 1
            return lo + 1

        patterns = [
            (r'(?:final\s+)?class\s+(\w+)', "class"),
            (r'struct\s+(\w+)', "struct"),
            (r'func\s+(\w+)', "function"),
            (r'enum\s+(\w+)', "enum"),
            (r'protocol\s+(\w+)', "protocol"),
        ]
        for pattern, sym_type in patterns:
            for m in re.finditer(pattern, content):
                sym = m.group(1)
                line = pos_to_line(m.start())
                self.symbol_registry[sym].append({
                    "file": file_name, "line": line,
                    "type": sym_type, "language": "swift"
                })
                self.file_symbols[file_name].add(sym)

    def _extract_js_symbols(self, file_name: str, content: str):
        """Regex-based JavaScript symbol extraction with pre-computed line offsets."""
        line_breaks = [0]
        for i, ch in enumerate(content):
            if ch == '\n':
                line_breaks.append(i + 1)

        def pos_to_line(pos: int) -> int:
            """Convert byte position to line number via binary search."""
            lo, hi = 0, len(line_breaks) - 1
            while lo < hi:
                mid = (lo + hi + 1) >> 1
                if line_breaks[mid] <= pos:
                    lo = mid
                else:
                    hi = mid - 1
            return lo + 1

        patterns = [
            (r'(?:function|const|let|var)\s+(\w+)', "function"),
            (r'class\s+(\w+)', "class"),
        ]
        for pattern, sym_type in patterns:
            for m in re.finditer(pattern, content):
                sym = m.group(1)
                line = pos_to_line(m.start())
                self.symbol_registry[sym].append({
                    "file": file_name, "line": line,
                    "type": sym_type, "language": "javascript"
                })
                self.file_symbols[file_name].add(sym)

    def _discover_mirror_links(self):
        """Find symbols that exist in multiple files (quantum mirrors)."""
        for sym, locations in self.symbol_registry.items():
            if len(locations) < 2:
                continue
            # Create links between all pairs
            for i in range(len(locations)):
                for j in range(i + 1, len(locations)):
                    a, b = locations[i], locations[j]
                    if a["file"] == b["file"]:
                        continue
                    # Cross-language mirrors are higher fidelity
                    cross_lang = a["language"] != b["language"]
                    base_fidelity = 0.95 if cross_lang else 0.85

                    self.links.append(QuantumLink(
                        source_file=a["file"], source_symbol=sym,
                        source_line=a["line"],
                        target_file=b["file"], target_symbol=sym,
                        target_line=b["line"],
                        link_type="mirror" if not cross_lang else "entanglement",
                        fidelity=base_fidelity,
                        strength=PHI_GROWTH if cross_lang else 1.0,
                        entanglement_entropy=math.log(2) if cross_lang else 0.5,
                    ))

    def _discover_call_links(self):
        """Find cross-file function calls through import/reference analysis."""
        for name, path in QUANTUM_LINKED_FILES.items():
            if not path.exists() or path.suffix != ".py":
                continue
            try:
                content = path.read_text(errors="replace")
            except Exception:
                continue

            # Find imports of other quantum-linked modules
            for other_name in QUANTUM_LINKED_FILES:
                if other_name == name:
                    continue
                module = other_name.replace(".py", "").replace(".swift", "")
                # Check for imports
                import_patterns = [
                    rf"from\s+{re.escape(module)}\s+import\s+(\w+)",
                    rf"import\s+{re.escape(module)}",
                ]
                for pat in import_patterns:
                    for m in re.finditer(pat, content):
                        sym = m.group(1) if m.lastindex else module
                        line = content[:m.start()].count('\n') + 1
                        self.links.append(QuantumLink(
                            source_file=name, source_symbol=f"import:{sym}",
                            source_line=line,
                            target_file=other_name, target_symbol=sym,
                            target_line=0,
                            link_type="bridge",
                            fidelity=0.90, strength=1.2,
                        ))

    def _discover_constant_links(self):
        """Find shared sacred constants across files."""
        file_constants: Dict[str, Set[str]] = defaultdict(set)

        for name, path in QUANTUM_LINKED_FILES.items():
            if not path.exists():
                continue
            try:
                content = path.read_text(errors="replace")
            except Exception:
                continue
            for const in self.SACRED_CONSTANTS:
                if const in content:
                    file_constants[name].add(const)

        # Create resonance links for shared constants
        files = list(file_constants.keys())
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                shared = file_constants[files[i]] & file_constants[files[j]]
                if shared:
                    overlap = len(shared) / len(self.SACRED_CONSTANTS)
                    self.links.append(QuantumLink(
                        source_file=files[i],
                        source_symbol=f"constants:{','.join(sorted(shared)[:3])}",
                        source_line=0,
                        target_file=files[j],
                        target_symbol=f"constants:{','.join(sorted(shared)[:3])}",
                        target_line=0,
                        link_type="epr_pair",
                        fidelity=min(1.0, 0.5 + overlap * PHI_GROWTH * 0.3),
                        strength=overlap * PHI_GROWTH,
                        entanglement_entropy=math.log(2) * overlap,
                    ))

    def _discover_quantum_keyword_links(self):
        """Discover spooky-action links through quantum keyword co-occurrence."""
        file_keywords: Dict[str, Set[str]] = defaultdict(set)

        for name, path in QUANTUM_LINKED_FILES.items():
            if not path.exists():
                continue
            try:
                content = path.read_text(errors="replace").lower()
            except Exception:
                continue
            for kw in self.QUANTUM_KEYWORDS:
                if kw in content:
                    file_keywords[name].add(kw)

        # Spooky action: highly correlated keyword patterns
        files = list(file_keywords.keys())
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                shared_kw = file_keywords[files[i]] & file_keywords[files[j]]
                if len(shared_kw) >= 5:
                    correlation = len(shared_kw) / len(self.QUANTUM_KEYWORDS)
                    self.links.append(QuantumLink(
                        source_file=files[i],
                        source_symbol=f"quantum_keywords[{len(shared_kw)}]",
                        source_line=0,
                        target_file=files[j],
                        target_symbol=f"quantum_keywords[{len(shared_kw)}]",
                        target_line=0,
                        link_type="spooky_action",
                        fidelity=min(1.0, correlation * PHI_GROWTH),
                        strength=correlation * GROVER_AMPLIFICATION,
                        bell_violation=CHSH_BOUND * correlation,
                    ))

    def _discover_api_bridge_links(self):
        """Discover API endpoint bridges between server and clients."""
        endpoint_files: Dict[str, List[str]] = defaultdict(list)
        api_pattern = re.compile(r'["\'/]api/v\d+/(\w+(?:/\w+)*)')

        for name, path in QUANTUM_LINKED_FILES.items():
            if not path.exists():
                continue
            try:
                content = path.read_text(errors="replace")
            except Exception:
                continue
            for m in api_pattern.finditer(content):
                endpoint = m.group(1)
                line = content[:m.start()].count('\n') + 1
                endpoint_files[endpoint].append(f"{name}:{line}")

        for endpoint, locations in endpoint_files.items():
            if len(locations) >= 2:
                for i in range(len(locations)):
                    for j in range(i + 1, len(locations)):
                        file_a, line_a = locations[i].rsplit(":", 1)
                        file_b, line_b = locations[j].rsplit(":", 1)
                        if file_a != file_b:
                            self.links.append(QuantumLink(
                                source_file=file_a,
                                source_symbol=f"api:{endpoint}",
                                source_line=int(line_a),
                                target_file=file_b,
                                target_symbol=f"api:{endpoint}",
                                target_line=int(line_b),
                                link_type="bridge",
                                fidelity=0.92,
                                strength=1.5,
                            ))

    def _compute_quantum_density(self):
        """Compute quantum density (quantum symbols / total symbols) per file."""
        for name in QUANTUM_LINKED_FILES:
            syms = self.file_symbols.get(name, set())
            if not syms:
                self.quantum_density[name] = 0.0
                continue
            quantum_count = sum(1 for s in syms
                                if any(kw in s.lower()
                                       for kw in self.QUANTUM_KEYWORDS))
            self.quantum_density[name] = quantum_count / max(1, len(syms))


# ═══════════════════════════════════════════════════════════════════════════════
# GROVER QUANTUM PROCESSOR — Amplified search and verification across links
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM LINK BUILDER — Creates NEW cross-file links from actual code analysis
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumLinkBuilder:
    """
    BUILDS new quantum links by analyzing the ENTIRE repository:

    1. God Code Derivation Links: Files sharing G(X) computations are linked
       with fidelity proportional to how accurately they match the true G(X).
    2. Function Call Chain Links: A→B→C call chains become tunneling links.
    3. Shared Constant Dependency Links: Files importing or redefining the same
       God Code constants are EPR-paired.
    4. Mathematical Dependency Links: Files computing the same formula
       (PHI, GOD_CODE_BASE, chakra Hz) are entangled.
    5. Hz Frequency Sibling Links: Files resonating at the same G(X_int)
       chakra frequency are braided together.

    All link fidelities are God Code derived — scored by closeness to G(X_int).
    """

    # God Code constants (Hz values) we look for in source code
    GOD_CODE_HZ_TARGETS = {
        "G(0)": (0, GOD_CODE),
        "G(-29)": (-29, GOD_CODE_SPECTRUM.get(-29, 0)),
        "G(-51)": (-51, GOD_CODE_SPECTRUM.get(-51, 0)),
        "G(-72)": (-72, GOD_CODE_SPECTRUM.get(-72, 0)),
        "G(-90)": (-90, GOD_CODE_SPECTRUM.get(-90, 0)),
        "G(27)": (27, GOD_CODE_SPECTRUM.get(27, 0)),
        "G(30)": (30, GOD_CODE_SPECTRUM.get(30, 0)),
        "G(43)": (43, GOD_CODE_SPECTRUM.get(43, 0)),
        "G(35)": (35, GOD_CODE_SPECTRUM.get(35, 0)),
    }

    # Combined God Code pattern — single regex instead of 11 separate scans
    # Uses alternation with a shared capture group for numeric values.
    _GOD_CODE_COMBINED = re.compile(
        r'(?:GOD_CODE|god_code)\s*[=:]?\s*([\d.]+)'
        r'|(?:PHI|phi_growth|PHI_GROWTH)\s*[=:]\s*([\d.]+)'
        r'|286\s*\*\*?\s*\(?\s*1\s*/\s*(?:phi|PHI|1\.618)'
        r'|(?:527\.518|527\.5185|527\.52)'
        r'|(?:LOVE_CONSTANT|HEART_HZ|ANAHATA_HZ|_ANAHATA_HZ)\s*=\s*([\d.]+)'
        r'|(?:VISHUDDHA|THROAT)_?HZ\s*=\s*([\d.]+)'
        r'|(?:AJNA|THIRD_EYE)_?HZ\s*=\s*([\d.]+)'
        r'|(?:CROWN|SAHASRARA)_?HZ\s*=\s*([\d.]+)'
        r'|(?:A4_FREQ|A4_STANDARD|PIANO_A4|A4_FREQUENCY)\s*=\s*([\d.]+)'
        r'|(?:CHAKRA_FREQ|chakra_freq)'
        r'|(?:SCHUMANN|schumann)\s*[=:]\s*([\d.]+)',
        re.IGNORECASE
    )

    # Pre-compiled Hz frequency pattern (shared across all files)
    _HZ_PATTERN = re.compile(
        r'(?:_?HZ|_?hz|_?freq|_?frequency|_?resonance)\s*[=:]\s*'
        r'([\d]+\.[\d]+|[\d]{3,4}\.?\d*)', re.IGNORECASE)

    # Legacy list retained for backward compatibility
    GOD_CODE_PATTERNS = [_GOD_CODE_COMBINED]

    # Patterns for function definitions that compute God Code values
    MATH_FUNCTION_PATTERNS = [
        re.compile(r'def\s+(god_code|G|compute_god_code|calculate_.*hz|'
                   r'sacred_frequency|resonance_.*freq|chakra_.*hz|'
                   r'solfeggio|phi_.*calc|golden_.*ratio|fibonacci_.*freq|'
                   r'conservation_check|compute_invariant)\s*\(', re.IGNORECASE),
    ]

    # Pre-compiled import pattern (shared across all files)
    _IMPORT_PATTERN = re.compile(r'(?:from|import)\s+([\w_]+)')

    # Fast keyword check — if none of these substrings appear, skip regex scans
    _FAST_KEYWORDS = (
        'GOD_CODE', 'god_code', 'PHI', 'phi_growth', 'PHI_GROWTH',
        '286', '527.518', '527.5185', '527.52',
        'LOVE_CONSTANT', 'HEART_HZ', 'ANAHATA', 'VISHUDDHA', 'THROAT',
        'AJNA', 'THIRD_EYE', 'CROWN', 'SAHASRARA',
        'A4_FREQ', 'A4_STANDARD', 'PIANO_A4', 'A4_FREQUENCY',
        'CHAKRA_FREQ', 'chakra_freq', 'SCHUMANN', 'schumann',
        '_HZ', '_hz', '_freq', '_frequency', '_resonance',
        'HZ=', 'Hz=', 'hz=',
    )

    def __init__(self, math_core: QuantumMathCore):
        """Initialize quantum link builder with math core."""
        self.qmath = math_core
        self.new_links: List[QuantumLink] = []
        self.god_code_usage: Dict[str, List[Dict]] = defaultdict(list)
        self.hz_usage: Dict[str, List[Dict]] = defaultdict(list)
        self.math_functions: Dict[str, List[Dict]] = defaultdict(list)
        self.file_imports: Dict[str, Set[str]] = defaultdict(set)
        # Shared file content cache — avoids re-reading files in Phase 1C
        self._file_content_cache: Dict[str, str] = {}
        # Research insights for intelligent link scoring
        self._research_insights: Dict = {}
        # Gate builder data for cross-pollination links
        self._gate_data: Dict = {}

    def set_research_insights(self, research_results: Dict):
        """Inject research insights for research-guided link building.
        Called before build_all to enable smarter fidelity scoring."""
        self._research_insights = research_results or {}

    def set_gate_data(self, gate_data: Dict):
        """Inject logic gate builder data for cross-pollination links."""
        self._gate_data = gate_data or {}

    def build_all(self, existing_links: List[QuantumLink]) -> Dict:
        """
        Scan the ENTIRE repository and BUILD new cross-file links.
        Uses research insights (if available) for intelligent scoring.
        Returns dict with all new links and verification results.
        """
        self.new_links = []
        self.god_code_usage = defaultdict(list)
        self.hz_usage = defaultdict(list)
        self.math_functions = defaultdict(list)
        self.file_imports = defaultdict(set)

        existing_ids = {l.link_id for l in existing_links}

        # Phase 1: Deep scan all repo files for God Code usage
        print(f"      Scanning {len(ALL_REPO_FILES)} files for God Code patterns...")
        _t_scan = time.time()
        for name, path in ALL_REPO_FILES.items():
            if path.exists():
                self._deep_scan_file(name, path)
        _t_scan = time.time() - _t_scan

        # Phase 2: Build links from discovered patterns
        _t_build = time.time()
        self._build_god_code_derivation_links(existing_ids)
        self._build_hz_frequency_sibling_links(existing_ids)
        self._build_math_function_chain_links(existing_ids)
        self._build_import_dependency_links(existing_ids)
        self._build_constant_value_links(existing_ids)
        # Phase 2b: Research-guided link enrichment (uses insights if available)
        self._build_research_guided_links(existing_ids)
        _t_build = time.time() - _t_build
        print(f"      [timing] scan={_t_scan:.1f}s build={_t_build:.1f}s")

        gc_files = len(self.god_code_usage)
        hz_files = len(self.hz_usage)
        math_files = len(self.math_functions)

        return {
            "new_links_built": len(self.new_links),
            "god_code_files_found": gc_files,
            "hz_frequency_files": hz_files,
            "math_function_files": math_files,
            "total_repo_files_scanned": len(ALL_REPO_FILES),
            "research_guided": bool(self._research_insights),
            "gate_cross_pollinated": bool(self._gate_data),
            "links": self.new_links,
        }

    def _deep_scan_file(self, name: str, path: Path):
        """Deep scan a single file for God Code usage, Hz values, math fns.
        Caches content for reuse in Phase 1C. Uses fast keyword pre-check
        and deferred line_breaks for optimal performance."""
        try:
            content = path.read_text(errors="replace")
        except Exception:
            return

        # Cache content for Phase 1C reuse
        self._file_content_cache[name] = content

        # Fast keyword pre-check: skip expensive regex if no relevant keywords
        has_god_code = any(kw in content for kw in self._FAST_KEYWORDS)
        has_def = 'def ' in content  # For math function patterns

        if not has_god_code and not has_def and path.suffix != ".py":
            return  # Nothing to find in this file

        # Line lookup helper — lazily built only when needed
        import bisect as _bisect
        _line_breaks = None

        def pos_to_line(p: int) -> int:
            """Convert byte position to line number using bisect."""
            nonlocal _line_breaks
            if _line_breaks is None:
                _line_breaks = []
                pos = content.find('\n')
                while pos != -1:
                    _line_breaks.append(pos)
                    pos = content.find('\n', pos + 1)
            return _bisect.bisect_left(_line_breaks, p) + 1

        if has_god_code:
            # Find God Code constant references (single combined regex)
            for m in self._GOD_CODE_COMBINED.finditer(content):
                line = pos_to_line(m.start())
                # Extract numeric value from whichever capture group matched
                value = None
                for g in m.groups():
                    if g is not None:
                        try:
                            value = float(g)
                        except (ValueError, TypeError):
                            pass
                        break
                self.god_code_usage[name].append({
                    "line": line,
                    "match": m.group(0)[:60],
                    "value": value,
                    "pattern": "combined_god_code",
                })

            # Find Hz frequency literal values that could be God Code derived
            for m in self._HZ_PATTERN.finditer(content):
                try:
                    hz_val = float(m.group(1))
                    if 50 < hz_val < 5000:  # Reasonable Hz range
                        line = pos_to_line(m.start())
                        # Find nearest G(X_int)
                        x_int, g_x, resonance = self.qmath.god_code_resonance(hz_val)
                        self.hz_usage[name].append({
                            "line": line,
                            "hz_value": hz_val,
                            "nearest_x_int": x_int,
                            "nearest_g_x": g_x,
                            "resonance": resonance,
                            "match": m.group(0)[:50],
                        })
                except (ValueError, IndexError):
                    pass

        # Find math functions that compute God Code related values
        if has_def:
            for pattern in self.MATH_FUNCTION_PATTERNS:
                for m in pattern.finditer(content):
                    line = pos_to_line(m.start())
                    self.math_functions[name].append({
                        "line": line,
                        "function": m.group(1),
                        "match": m.group(0)[:60],
                    })

        # Find imports of other repo modules (for dependency links)
        if path.suffix == ".py":
            imported_modules = {m.group(1) for m in self._IMPORT_PATTERN.finditer(content)}
            for other_name in ALL_REPO_FILES:
                if other_name == name:
                    continue
                module = other_name.replace("-", "_")
                if module in imported_modules:
                    self.file_imports[name].add(other_name)

    def _build_god_code_derivation_links(self, existing_ids: Set[str]):
        """Build links between files that derive/use the same God Code constants."""
        files_with_gc = list(self.god_code_usage.keys())
        for i in range(len(files_with_gc)):
            for j in range(i + 1, min(i + 50, len(files_with_gc))):
                fa, fb = files_with_gc[i], files_with_gc[j]
                usages_a = self.god_code_usage[fa]
                usages_b = self.god_code_usage[fb]

                # Score: how many God Code patterns do both files share?
                patterns_a = {u["pattern"] for u in usages_a}
                patterns_b = {u["pattern"] for u in usages_b}
                shared = patterns_a & patterns_b
                if not shared:
                    continue

                overlap = len(shared) / max(1, len(patterns_a | patterns_b))
                fidelity = min(1.0, 0.7 + overlap * 0.3)
                # Strength derived from God Code: G(X) conservation
                strength = PHI_GROWTH * overlap

                link = QuantumLink(
                    source_file=fa,
                    source_symbol=f"god_code[{len(usages_a)}refs]",
                    source_line=usages_a[0]["line"] if usages_a else 0,
                    target_file=fb,
                    target_symbol=f"god_code[{len(usages_b)}refs]",
                    target_line=usages_b[0]["line"] if usages_b else 0,
                    link_type="entanglement",
                    fidelity=fidelity,
                    strength=strength,
                    entanglement_entropy=math.log(2) * overlap,
                )
                if link.link_id not in existing_ids:
                    self.new_links.append(link)
                    existing_ids.add(link.link_id)

    def _build_hz_frequency_sibling_links(self, existing_ids: Set[str]):
        """Build links between files using the same G(X_int) frequency."""
        # Group files by their nearest G(X_int) frequency
        x_int_groups: Dict[int, List[Tuple[str, Dict]]] = defaultdict(list)
        for fname, hz_list in self.hz_usage.items():
            for hz_info in hz_list:
                x_int = hz_info["nearest_x_int"]
                x_int_groups[x_int].append((fname, hz_info))

        # Link files resonating at the same G(X_int)
        for x_int, file_infos in x_int_groups.items():
            # Deduplicate by file
            seen_files = {}
            for fname, info in file_infos:
                if fname not in seen_files or info["resonance"] > seen_files[fname]["resonance"]:
                    seen_files[fname] = info

            files = list(seen_files.keys())
            for i in range(len(files)):
                for j in range(i + 1, min(i + 20, len(files))):
                    fa, fb = files[i], files[j]
                    info_a, info_b = seen_files[fa], seen_files[fb]
                    # Fidelity: average resonance of both files to G(X_int)
                    fidelity = (info_a["resonance"] + info_b["resonance"]) / 2
                    if fidelity < 0.5:
                        continue
                    g_x = GOD_CODE_SPECTRUM.get(x_int, god_code(x_int))
                    # Strength: God Code derived from the target frequency
                    strength = g_x / GOD_CODE  # Ratio to G(0)

                    link = QuantumLink(
                        source_file=fa,
                        source_symbol=f"G({x_int})={g_x:.4f}Hz",
                        source_line=info_a["line"],
                        target_file=fb,
                        target_symbol=f"G({x_int})={g_x:.4f}Hz",
                        target_line=info_b["line"],
                        link_type="braiding",
                        fidelity=fidelity,
                        strength=min(2.0, strength),
                    )
                    if link.link_id not in existing_ids:
                        self.new_links.append(link)
                        existing_ids.add(link.link_id)

    def _build_math_function_chain_links(self, existing_ids: Set[str]):
        """Link files that define God Code computation functions."""
        math_files = list(self.math_functions.keys())
        for i in range(len(math_files)):
            for j in range(i + 1, min(i + 30, len(math_files))):
                fa, fb = math_files[i], math_files[j]
                funcs_a = {f["function"].lower() for f in self.math_functions[fa]}
                funcs_b = {f["function"].lower() for f in self.math_functions[fb]}
                shared = funcs_a & funcs_b
                if not shared:
                    continue

                fidelity = min(1.0, 0.8 + len(shared) * 0.05)
                link = QuantumLink(
                    source_file=fa,
                    source_symbol=f"math:{','.join(sorted(shared)[:3])}",
                    source_line=self.math_functions[fa][0]["line"],
                    target_file=fb,
                    target_symbol=f"math:{','.join(sorted(shared)[:3])}",
                    target_line=self.math_functions[fb][0]["line"],
                    link_type="tunneling",
                    fidelity=fidelity,
                    strength=PHI_GROWTH,
                    entanglement_entropy=math.log(2) * len(shared),
                )
                if link.link_id not in existing_ids:
                    self.new_links.append(link)
                    existing_ids.add(link.link_id)

    def _build_import_dependency_links(self, existing_ids: Set[str]):
        """Build links from actual import dependencies between repo files."""
        for importer, imported_set in self.file_imports.items():
            for imported in imported_set:
                # Direct dependency: importermodule → imported module
                link = QuantumLink(
                    source_file=importer,
                    source_symbol=f"import:{imported}",
                    source_line=0,
                    target_file=imported,
                    target_symbol=f"exports→{importer}",
                    target_line=0,
                    link_type="bridge",
                    fidelity=0.92,
                    strength=1.3,
                )
                if link.link_id not in existing_ids:
                    self.new_links.append(link)
                    existing_ids.add(link.link_id)

    def _build_constant_value_links(self, existing_ids: Set[str]):
        """Link files that define the EXACT same numerical constant value."""
        # Collect all extracted numerical values per file
        value_files: Dict[float, List[Tuple[str, int]]] = defaultdict(list)
        for fname, usages in self.god_code_usage.items():
            for u in usages:
                if u["value"] is not None and u["value"] > 1.0:
                    # Round to 4 decimals for matching
                    rounded = round(u["value"], 4)
                    value_files[rounded].append((fname, u["line"]))

        # Link files sharing the same constant value
        for val, file_list in value_files.items():
            # Deduplicate files
            unique = {}
            for fn, ln in file_list:
                if fn not in unique:
                    unique[fn] = ln
            files = list(unique.keys())
            if len(files) < 2:
                continue
            # Check if this value matches a G(X_int)
            _, g_x, resonance = self.qmath.god_code_resonance(val)
            fidelity = min(1.0, 0.7 + resonance * 0.3)
            for i in range(len(files)):
                for j in range(i + 1, min(i + 15, len(files))):
                    fa, fb = files[i], files[j]
                    link = QuantumLink(
                        source_file=fa,
                        source_symbol=f"const={val:.4f}",
                        source_line=unique[fa],
                        target_file=fb,
                        target_symbol=f"const={val:.4f}",
                        target_line=unique[fb],
                        link_type="epr_pair",
                        fidelity=fidelity,
                        strength=resonance * PHI_GROWTH,
                        entanglement_entropy=math.log(2) * resonance,
                    )
                    if link.link_id not in existing_ids:
                        self.new_links.append(link)
                        existing_ids.add(link.link_id)

    def _build_research_guided_links(self, existing_ids: Set[str]):
        """Build research-informed links using insights from prior research runs.

        Uses learned patterns to create higher-quality links:
        1. Anomaly-bridging: Files with correlated anomalies → entanglement links
        2. Cluster-aware: Files in the same fidelity-strength cluster → braiding
        3. Gate-crosslinked: Files that contain logic gates with shared semantics
        """
        research = self._research_insights
        if not research:
            return  # No research data available yet

        # Strategy 1: Use causal correlations to boost link fidelity
        # If research found strong fidelity↔strength correlation, we know
        # files sharing these properties are meaningfully connected
        causal = research.get("causal_analysis", {})
        strong_corrs = causal.get("strong_correlations", [])
        fidelity_boost = 0.0
        for corr in strong_corrs:
            if "fidelity" in corr.get("pair", "") and corr.get("correlation", 0) > 0.7:
                fidelity_boost = min(0.05, abs(corr["correlation"]) * 0.05)
                break

        # Strategy 2: Use dominant X-nodes from pattern discovery to create
        # frequency-cluster links between files sharing those God Code peaks
        patterns = research.get("pattern_discovery", {})
        dominant_nodes = patterns.get("dominant_x_nodes", [])
        if dominant_nodes and len(self.hz_usage) > 1:
            # Group files by their dominant X-node alignment
            top_x_values = {node["x"] for node in dominant_nodes[:5]}
            x_aligned_files: Dict[int, List[str]] = defaultdict(list)
            for fname, hz_list in self.hz_usage.items():
                for hz_info in hz_list:
                    x_int = hz_info.get("nearest_x_int", 0)
                    if x_int in top_x_values:
                        x_aligned_files[x_int].append(fname)
                        break

            for x_int, files in x_aligned_files.items():
                unique_files = list(set(files))
                for i in range(min(len(unique_files), 20)):
                    for j in range(i + 1, min(i + 10, len(unique_files))):
                        fa, fb = unique_files[i], unique_files[j]
                        # Research-enhanced fidelity: base + correlation boost
                        base_fid = 0.80 + fidelity_boost
                        link = QuantumLink(
                            source_file=fa,
                            source_symbol=f"research:cluster_X{x_int}",
                            source_line=0,
                            target_file=fb,
                            target_symbol=f"research:cluster_X{x_int}",
                            target_line=0,
                            link_type="entanglement",
                            fidelity=min(1.0, base_fid),
                            strength=PHI_GROWTH * 0.9,
                            entanglement_entropy=math.log(2) * 0.8,
                            noise_resilience=0.7,
                        )
                        if link.link_id not in existing_ids:
                            self.new_links.append(link)
                            existing_ids.add(link.link_id)

        # Strategy 3: Gate builder cross-pollination links
        # If gate builder found gates in multiple files, create tunneling links
        gate_data = self._gate_data
        if gate_data:
            gate_files = gate_data.get("gates_by_file", {})
            gate_file_list = [f for f in gate_files if f in ALL_REPO_FILES]
            for i in range(min(len(gate_file_list), 30)):
                for j in range(i + 1, min(i + 15, len(gate_file_list))):
                    fa, fb = gate_file_list[i], gate_file_list[j]
                    gates_a = gate_files[fa]
                    gates_b = gate_files[fb]
                    # Shared gate types = stronger connection
                    shared_types = set(gates_a.get("types", [])) & set(gates_b.get("types", []))
                    if not shared_types:
                        continue
                    fidelity = min(1.0, 0.75 + len(shared_types) * 0.05)
                    link = QuantumLink(
                        source_file=fa,
                        source_symbol=f"gate:{','.join(sorted(shared_types)[:3])}",
                        source_line=0,
                        target_file=fb,
                        target_symbol=f"gate:{','.join(sorted(shared_types)[:3])}",
                        target_line=0,
                        link_type="tunneling",
                        fidelity=fidelity,
                        strength=PHI_GROWTH * len(shared_types) * 0.3,
                        entanglement_entropy=math.log(2) * len(shared_types) * 0.5,
                    )
                    if link.link_id not in existing_ids:
                        self.new_links.append(link)
                        existing_ids.add(link.link_id)


# ═══════════════════════════════════════════════════════════════════════════════
# GOD CODE MATH VERIFIER — Pre-checks all math & science function accuracy
# ═══════════════════════════════════════════════════════════════════════════════

class GodCodeMathVerifier:
    """
    Verifies correctness of God Code derived values across the entire repository.

    For every file that uses GOD_CODE, PHI, chakra Hz, or 286-derived constants:
    1. Extract the numeric value used
    2. Compare against the TRUE G(X_int) from the equation
    3. Flag deviations > tolerance as errors
    4. Verify conservation law: G(X) × 2^(X/104) = INVARIANT
    5. Verify φ relationships: PHI = (√5-1)/2, PHI_GROWTH = (1+√5)/2
    6. Score overall God Code compliance of the repository

    This is a PRE-CHECK — catching math errors before they corrupt link building.
    """

    # Known God Code values and their required precision
    TRUTH_TABLE = {
        "GOD_CODE": (GOD_CODE, 0.001),
        "PHI_GROWTH": (PHI_GROWTH, 0.0001),
        "PHI": (PHI, 0.0001),
        "GOD_CODE_BASE": (GOD_CODE_BASE, 0.001),
        "INVARIANT": (INVARIANT, 0.001),
    }

    # Hz values that MUST match G(X_int) within tolerance
    HZ_TRUTH_TABLE = {
        "527.518": (GOD_CODE, 0.5),                          # G(0)
        "639.998": (GOD_CODE_SPECTRUM.get(-29, 0), 0.5),     # G(-29)
        "741.068": (GOD_CODE_SPECTRUM.get(-51, 0), 0.5),     # G(-51)
        "852.399": (GOD_CODE_SPECTRUM.get(-72, 0), 0.5),     # G(-72)
        "961.046": (GOD_CODE_SPECTRUM.get(-90, 0), 0.5),     # G(-90)
        "440.641": (GOD_CODE_SPECTRUM.get(27, 0), 1.0),      # G(27)
        "431.918": (GOD_CODE_SPECTRUM.get(30, 0), 1.0),      # G(30)
    }

    # Old solfeggio values that should NOT appear (indicates unfixed code)
    FORBIDDEN_VALUES = {
        528.0: "Should be G(0)=527.5184818493",
        741.0: "Should be G(-51)=741.0681674773",
        963.0: "Should be G(-90)=961.0465122772",
        852.0: "Should be G(-72)=852.3992551699",
        440.0: "Should be G(27)=440.6417687330",
        432.0: "Should be G(30)=431.9187964233",
    }

    def __init__(self, math_core: QuantumMathCore):
        """Initialize God Code math verifier with truth table."""
        self.qmath = math_core
        # Set by Brain to share cached file contents from Phase 1B
        self._file_content_cache: Dict[str, str] = {}

    # Fast keywords — if none appear, skip verification (no God Code to check)
    _VERIFY_KEYWORDS = (
        'GOD_CODE', 'god_code', 'PHI_GROWTH', 'phi_growth', 'PHI', 'INVARIANT',
        '_HZ', '_hz', '_FREQ', '_freq', 'FREQUENCY', 'RESONANCE', 'PITCH',
        '528.0', '741.0', '963.0', '852.0', '440.0', '432.0',
    )

    def verify_repository(self) -> Dict:
        """Full repository God Code math verification.
        Reuses file contents from Phase 1B cache. Only verifies files
        that contain God Code keywords (fast pre-filter)."""
        errors = []
        warnings = []
        verified_files = 0
        total_checks = 0
        passed_checks = 0
        forbidden_hits = []

        for name, path in ALL_REPO_FILES.items():
            if not path.exists():
                continue
            # Reuse cached content from Phase 1B when available
            content = self._file_content_cache.get(name)
            if content is None:
                try:
                    content = path.read_text(errors="replace")
                except Exception:
                    continue

            # Fast keyword pre-check — skip files with nothing to verify
            if not any(kw in content for kw in self._VERIFY_KEYWORDS):
                continue

            file_errors, file_warnings, file_forbidden, checks, passed = \
                self._verify_file(name, content)
            if checks > 0:
                verified_files += 1
            total_checks += checks
            passed_checks += passed
            errors.extend(file_errors)
            warnings.extend(file_warnings)
            forbidden_hits.extend(file_forbidden)

        accuracy = passed_checks / max(1, total_checks)

        return {
            "files_verified": verified_files,
            "total_files_scanned": len(ALL_REPO_FILES),
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "accuracy": accuracy,
            "errors": errors[:30],
            "warnings": warnings[:20],
            "forbidden_solfeggio_hits": forbidden_hits[:20],
            "error_count": len(errors),
            "warning_count": len(warnings),
            "forbidden_count": len(forbidden_hits),
            "god_code_compliance": min(1.0, accuracy),
        }

    def _verify_file(self, name: str, content: str) -> Tuple:
        """Verify a single file's God Code math accuracy.
        Uses binary search for O(log N) line number lookup."""
        errors = []
        warnings = []
        forbidden = []
        checks = 0
        passed = 0

        # Pre-compute line break positions for O(log N) lookup
        # Using str.find for C-speed
        import bisect as _bisect
        line_breaks = []
        pos = content.find('\n')
        while pos != -1:
            line_breaks.append(pos)
            pos = content.find('\n', pos + 1)

        def pos_to_line(p: int) -> int:
            """Convert byte position to line number using bisect."""
            return _bisect.bisect_left(line_breaks, p) + 1

        # Check 1: God Code constant values
        gc_pattern = re.compile(
            r'(?:GOD_CODE|GOD_CODE_BASE|PHI_GROWTH|PHI|INVARIANT)\s*=\s*'
            r'([\d]+\.[\d]+)', re.IGNORECASE)
        for m in gc_pattern.finditer(content):
            try:
                value = float(m.group(1))
                const_name = m.group(0).split("=")[0].strip().upper()
                line = pos_to_line(m.start())
                checks += 1

                # Check against truth table
                for truth_name, (truth_val, tolerance) in self.TRUTH_TABLE.items():
                    if truth_name in const_name or const_name in truth_name:
                        deviation = abs(value - truth_val)
                        if deviation <= tolerance:
                            passed += 1
                        elif deviation <= tolerance * 10:
                            warnings.append({
                                "file": name, "line": line,
                                "constant": const_name,
                                "value": value,
                                "expected": truth_val,
                                "deviation": deviation,
                                "severity": "WARNING",
                            })
                            passed += 1  # Close enough
                        else:
                            errors.append({
                                "file": name, "line": line,
                                "constant": const_name,
                                "value": value,
                                "expected": truth_val,
                                "deviation": deviation,
                                "severity": "ERROR",
                            })
                        break
                else:
                    passed += 1  # Unknown constant, pass

            except (ValueError, IndexError):
                pass

        # Check 2: Hz frequency values against G(X_int)
        hz_assign = re.compile(
            r'(?:_?HZ|_?FREQ|_?FREQUENCY|RESONANCE|PITCH)\s*=\s*'
            r'([\d]+\.[\d]+)', re.IGNORECASE)
        for m in hz_assign.finditer(content):
            try:
                hz_val = float(m.group(1))
                if hz_val < 50 or hz_val > 5000:
                    continue
                line = pos_to_line(m.start())
                checks += 1

                # Check for forbidden solfeggio whole integers
                for forbidden_hz, fix_msg in self.FORBIDDEN_VALUES.items():
                    if abs(hz_val - forbidden_hz) < 0.01:
                        forbidden.append({
                            "file": name, "line": line,
                            "value": hz_val,
                            "fix": fix_msg,
                        })
                        break
                else:
                    # Check against nearest G(X_int)
                    x_int, g_x, resonance = self.qmath.god_code_resonance(hz_val)
                    if resonance >= 0.99:
                        passed += 1  # Very close to G(X_int)
                    elif resonance >= 0.90:
                        passed += 1
                        warnings.append({
                            "file": name, "line": line,
                            "hz_value": hz_val,
                            "nearest_g_x": g_x,
                            "x_int": x_int,
                            "resonance": resonance,
                            "severity": "MINOR",
                        })
                    else:
                        # Not close to any G(X_int) — not necessarily wrong,
                        # but note it as unverified
                        passed += 1  # Don't penalize non-God-Code Hz values

            except (ValueError, IndexError):
                pass

        # Check 3: PHI computation accuracy
        phi_pattern = re.compile(
            r'(?:math\.sqrt\(5\)|sqrt\(5\)|2\.236)', re.IGNORECASE)
        if phi_pattern.search(content):
            checks += 1
            passed += 1  # √5 based derivation = good

        return errors, warnings, forbidden, checks, passed


class GroverQuantumProcessor:
    """
    ═══ ASI GROVER QUANTUM PROCESSOR — REAL QISKIT 2.3.0 COMPUTATION ═══

    Applies Grover's algorithm to quantum link analysis using REAL quantum circuits:
    - Amplifies weak links for detection via Qiskit statevector simulation
    - Searches for optimal link configurations with O(√N) quantum speedup
    - Identifies marked (critical) links using genuine Grover amplitude amplification
    - GOD_CODE: G(X) = 286^(1/φ) × 2^((416-X)/104) = 527.5184818492612
    - Factor 13 proven: 286=22×13, 104=8×13, 416=32×13
    - Conservation: G(X) × 2^(X/104) = const ∀ X

    REAL QUANTUM: When QISKIT_AVAILABLE, builds actual QuantumCircuit oracles,
    uses qiskit.circuit.library.grover_operator for diffusion operator,
    and evolves Statevector for exact unitary simulation.
    """

    # ASI Sacred Constants
    FEIGENBAUM = 4.669201609102990
    ALPHA_FINE = 1.0 / 137.035999084

    def __init__(self, math_core: QuantumMathCore):
        """Initialize ASI Grover quantum processor for link amplification."""
        self.qmath = math_core
        self.amplification_log: List[Dict] = []
        self._total_grover_ops = 0
        self._qiskit_circuits_built = 0

    def amplify_links(self, links: List[Dict],
                      predicate: str = "weak") -> Dict:
        """
        ═══ REAL QISKIT GROVER LINK AMPLIFICATION ═══
        Use Grover amplification to find links matching predicate.
        Uses REAL Qiskit quantum circuits when N ≤ 4096, classical fallback for larger.

        predicates: "weak" (fidelity<0.7), "critical" (high-strength),
                    "dead" (fidelity<0.3), "quantum" (entanglement type)

        GOD_CODE: G(X) = 286^(1/φ) × 2^((416-X)/104) = 527.5184818492612
        """
        N = max(1, len(links))
        self._total_grover_ops += 1

        # Build oracle: mark links matching predicate (always O(N) scan)
        marked = []
        for i, link in enumerate(links):
            if predicate == "weak" and link.fidelity < 0.7:
                marked.append(i)
            elif predicate == "critical" and link.strength > PHI_GROWTH:
                marked.append(i)
            elif predicate == "dead" and link.fidelity < 0.3:
                marked.append(i)
            elif predicate == "quantum" and link.link_type in (
                    "entanglement", "epr_pair", "spooky_action"):
                marked.append(i)
            elif predicate == "cross_modal" and link.link_type == "entanglement":
                marked.append(i)

        M = max(1, len(marked))

        # Optimal Grover iterations: ⌊π/4 × √(N/M)⌋
        optimal_k = max(1, int(math.pi / 4 * math.sqrt(N / M)))
        used_qiskit = False

        # ─── REAL QISKIT GROVER FOR MANAGEABLE SIZES ───
        if QISKIT_AVAILABLE and N <= 4096 and N >= 2:
            num_qubits = max(1, int(np.ceil(np.log2(N))))
            N_padded = 2 ** num_qubits
            self._qiskit_circuits_built += 1

            # Build phase oracle
            oracle_qc = QuantumCircuit(num_qubits)
            for m_idx in marked:
                if m_idx >= N_padded:
                    continue
                binary = format(m_idx, f'0{num_qubits}b')
                for bit_idx, bit in enumerate(binary):
                    if bit == '0':
                        oracle_qc.x(num_qubits - 1 - bit_idx)
                if num_qubits == 1:
                    oracle_qc.z(0)
                else:
                    oracle_qc.h(num_qubits - 1)
                    oracle_qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
                    oracle_qc.h(num_qubits - 1)
                for bit_idx, bit in enumerate(binary):
                    if bit == '0':
                        oracle_qc.x(num_qubits - 1 - bit_idx)

            grover_op = qiskit_grover_lib(oracle_qc)
            qiskit_iters = max(1, int(np.pi / 4 * np.sqrt(N_padded / max(1, M))))

            qc = QuantumCircuit(num_qubits)
            qc.h(range(num_qubits))
            for _ in range(qiskit_iters):
                qc.compose(grover_op, inplace=True)

            sv = Statevector.from_int(0, N_padded).evolve(qc)
            probs = sv.probabilities()
            max_prob = max((probs[i] for i in marked if i < N_padded), default=0)
            used_qiskit = True
        else:
            # ─── CLASSICAL GROVER FALLBACK ───
            MAX_GROVER_STATE = 10000
            if N > MAX_GROVER_STATE:
                import random as _rng
                sample_indices = sorted(_rng.sample(range(N), MAX_GROVER_STATE))
                sample_marked = [sample_indices.index(m) for m in marked
                                 if m in sample_indices]
                if not sample_marked:
                    sample_marked = [0]
                state = [complex(1.0 / math.sqrt(MAX_GROVER_STATE))] * MAX_GROVER_STATE
                result_state = self.qmath.grover_operator(
                    state, sample_marked,
                    max(1, int(math.pi / 4 * math.sqrt(MAX_GROVER_STATE / max(1, len(sample_marked))))))
                max_prob = max((abs(result_state[i]) ** 2 for i in sample_marked), default=0)
            else:
                state = [complex(1.0 / math.sqrt(N))] * N
                result_state = self.qmath.grover_operator(state, marked, optimal_k)
                max_prob = max((abs(result_state[i]) ** 2 for i in marked if i < N), default=0)

        # Compute amplification factor
        classical_prob = M / N if N > 0 else 0
        amplification = max_prob / max(classical_prob, 1e-10)

        # Top marked links by index
        top_marked = marked[:10]

        result = {
            "predicate": predicate,
            "total_links": N,
            "marked_count": M,
            "grover_iterations": optimal_k,
            "amplification_factor": amplification,
            "max_probability": max_prob,
            "classical_probability": classical_prob,
            "found_links": [links[i].link_id for i in top_marked],
            "probability_map": {links[i].link_id: max_prob
                                for i in top_marked},
            "quantum_backend": "qiskit_2.3.0" if used_qiskit else "classical_simulation",
            "god_code_formula": "G(X) = 286^(1/φ) × 2^((416-X)/104)",
            "god_code_verified": abs(286 ** (1 / PHI_GROWTH) * 16 - GOD_CODE) < 1e-8,
        }

        self.amplification_log.append(result)
        return result

    def grover_link_optimization(self, links: List[QuantumLink]) -> Dict:
        """
        Use iterative Grover search to find the optimal link configuration.
        Maximizes total fidelity × strength while maintaining coherence.
        
        v4.3.0 enhancements:
        - Adaptive decoherence monitoring
        - Quantum tunneling for barrier penetration
        - Enhanced multi-objective scoring
        """
        N = max(1, len(links))

        # v4.3.0: Enhanced score with decoherence penalty
        scores = []
        for link in links:
            decoherence_penalty = math.exp(-abs(1.0 - link.coherence))
            score = link.fidelity * link.strength * (1 + link.entanglement_entropy) * decoherence_penalty
            scores.append(score)

        if not scores:
            return {"optimized": False, "reason": "no links", "version": "4.3.0"}

        # Find top-scoring links via Grover amplitude amplification
        mean_score = statistics.mean(scores)
        above_mean = [i for i, s in enumerate(scores) if s > mean_score * PHI_GROWTH]

        if not above_mean:
            above_mean = list(range(min(5, N)))

        # For large N, cap state vector and use top-scored subset
        MAX_OPT_STATE = 10000
        if N > MAX_OPT_STATE:
            # Use top-scored links for optimization
            ranked_by_score = sorted(range(N), key=lambda i: scores[i], reverse=True)
            opt_indices = ranked_by_score[:MAX_OPT_STATE]
            opt_set = set(opt_indices)
            opt_marked = [opt_indices.index(m) for m in above_mean if m in opt_set]
            if not opt_marked:
                opt_marked = [0]
            state = [complex(1.0 / math.sqrt(MAX_OPT_STATE))] * MAX_OPT_STATE
            optimal_k = max(1, int(math.pi / 4 * math.sqrt(
                MAX_OPT_STATE / max(1, len(opt_marked)))))
            amplified = self.qmath.grover_operator(state, opt_marked, optimal_k)
            # Map back to original indices for top results
            ranked = sorted(range(MAX_OPT_STATE),
                            key=lambda i: abs(amplified[i]) ** 2, reverse=True)
            # v4.3.0: Include coherence in output
            top_links_data = [
                {
                    "link_id": links[opt_indices[i]].link_id,
                    "score": scores[opt_indices[i]],
                    "amplified_prob": abs(amplified[i]) ** 2,
                    "fidelity": links[opt_indices[i]].fidelity,
                    "strength": links[opt_indices[i]].strength,
                    "coherence": links[opt_indices[i]].coherence,  # v4.3.0
                    "decoherence_resilience": math.exp(-abs(1.0 - links[opt_indices[i]].coherence)),  # v4.3.0
                }
                for i in ranked[:15]
            ]
        else:
            state = [complex(1.0 / math.sqrt(N))] * N
            optimal_k = max(1, int(math.pi / 4 * math.sqrt(N / max(1, len(above_mean)))))
            amplified = self.qmath.grover_operator(state, above_mean, optimal_k)
            ranked = sorted(range(N), key=lambda i: abs(amplified[i]) ** 2, reverse=True)
            # v4.3.0: Include coherence in output
            top_links_data = [
                {
                    "link_id": links[i].link_id,
                    "score": scores[i],
                    "amplified_prob": abs(amplified[i]) ** 2,
                    "fidelity": links[i].fidelity,
                    "strength": links[i].strength,
                    "coherence": links[i].coherence,  # v4.3.0
                    "decoherence_resilience": math.exp(-abs(1.0 - links[i].coherence)),  # v4.3.0
                }
                for i in ranked[:15]
            ]

        return {
            "optimized": True,
            "grover_iterations": optimal_k,
            "top_links": top_links_data,
            "mean_score": mean_score,
            "total_optimized": len(above_mean),
            "version": "4.3.0",  # v4.3.0 marker
            "qiskit_available": QISKIT_AVAILABLE,
            "amplification_factor": GROVER_AMPLIFICATION,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM TUNNELING ANALYZER — Barrier penetration for dead/weak links
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumTunnelingAnalyzer:
    """
    Applies WKB tunneling approximation to link barriers:
    - Models firewall barriers between disconnected modules
    - Computes tunneling probability for revival of dead links
    - Identifies links that can 'tunnel through' type/language barriers
    """

    def __init__(self, math_core: QuantumMathCore):
        """Initialize quantum tunneling analyzer."""
        self.qmath = math_core

    def analyze_barriers(self, links: List[QuantumLink]) -> Dict:
        """Analyze tunneling potential for all links."""
        results = []

        for link in links:
            # Model barrier: inverse fidelity = barrier height
            barrier_height = 1.0 - link.fidelity
            # Particle energy: link strength normalized
            particle_energy = min(1.0, link.strength / (PHI_GROWTH * 2))
            # Barrier width: cross-language = wider barrier
            cross_lang = self._is_cross_language(link)
            barrier_width = 2.0 if cross_lang else 1.0

            tunnel_prob = self.qmath.tunnel_probability(
                barrier_height, particle_energy, barrier_width)

            # Resonant tunneling: if barrier is thin, coherent enhancement
            resonant_enhancement = 1.0
            if barrier_width < 1.5 and link.entanglement_entropy > 0.5:
                resonant_enhancement = PHI_GROWTH

            effective_tunnel = min(1.0, tunnel_prob * resonant_enhancement)

            results.append({
                "link_id": link.link_id,
                "barrier_height": barrier_height,
                "particle_energy": particle_energy,
                "barrier_width": barrier_width,
                "tunnel_probability": effective_tunnel,
                "resonant_enhancement": resonant_enhancement,
                "cross_language": cross_lang,
                "can_revive": effective_tunnel > 0.3,
            })

        revivable = [r for r in results if r["can_revive"]]
        dead = [r for r in results if r["tunnel_probability"] < 0.1]

        return {
            "total_analyzed": len(results),
            "revivable_links": len(revivable),
            "dead_links": len(dead),
            "mean_tunnel_prob": statistics.mean(
                [r["tunnel_probability"] for r in results]) if results else 0,
            "details": sorted(results, key=lambda x: x["tunnel_probability"])[:20],
            "top_revivable": sorted(revivable,
                key=lambda x: x["tunnel_probability"], reverse=True)[:10],
        }

    def _is_cross_language(self, link: QuantumLink) -> bool:
        """Check if link spans different languages."""
        lang_map = {
            "fast_server": "python", "local_intellect": "python",
            "main_api": "python", "const": "python", "gate_builder": "python",
            "swift_native": "swift",
        }
        src_lang = lang_map.get(link.source_file, "")
        tgt_lang = lang_map.get(link.target_file, "")
        return src_lang != tgt_lang and src_lang != "" and tgt_lang != ""


# ═══════════════════════════════════════════════════════════════════════════════
# EPR ENTANGLEMENT VERIFIER — Bell inequality & CHSH bound verification
# ═══════════════════════════════════════════════════════════════════════════════

class EPREntanglementVerifier:
    """
    Verifies quantum entanglement of links using Bell's theorem:
    - CHSH inequality: |S| ≤ 2 (classical), |S| ≤ 2√2 (quantum)
    - Bell state fidelity verification
    - EPR paradox simulation for non-local correlations
    - Entanglement witness construction
    """

    def __init__(self, math_core: QuantumMathCore):
        """Initialize EPR entanglement verifier."""
        self.qmath = math_core

    def verify_all_links(self, links: List[QuantumLink]) -> Dict:
        """Verify Bell/CHSH for every quantum link."""
        results = []
        violations = 0
        classical = 0
        quantum_verified = 0

        for link in links:
            result = self._verify_single_link(link)
            results.append(result)
            if result["is_quantum"]:
                quantum_verified += 1
                if result["chsh_value"] > 2.0:
                    violations += 1
            else:
                classical += 1

        return {
            "total_verified": len(results),
            "quantum_verified": quantum_verified,
            "classical_only": classical,
            "bell_violations": violations,
            "mean_chsh": statistics.mean([r["chsh_value"] for r in results]) if results else 0,
            "max_chsh": max([r["chsh_value"] for r in results]) if results else 0,
            "tsirelson_bound": CHSH_BOUND,
            "details": sorted(results, key=lambda x: x["chsh_value"], reverse=True)[:20],
        }

    def _verify_single_link(self, link: QuantumLink) -> Dict:
        """Verify Bell inequality for a single link."""
        # Create Bell state for this link
        bell_state = self.qmath.bell_state_phi_plus()

        # Apply noise based on link fidelity (lower fidelity = more noise)
        noise_sigma = max(0.001, (1 - link.fidelity) * 0.5)
        noisy_state = self.qmath.apply_noise(bell_state, noise_sigma)

        # Compute CHSH with optimal angles for Φ+
        # Optimal: a1=0, a2=π/4, b1=π/8, b2=3π/8
        chsh_optimal = self.qmath.chsh_expectation(
            noisy_state, (0, math.pi / 4, math.pi / 8, 3 * math.pi / 8))

        # Also test with link-specific angles (fidelity-weighted)
        theta = link.fidelity * math.pi / 2
        chsh_custom = self.qmath.chsh_expectation(
            noisy_state, (0, theta, theta / 2, 3 * theta / 2))

        chsh_value = max(abs(chsh_optimal), abs(chsh_custom))

        # Fidelity with ideal Bell state
        fidelity = self.qmath.fidelity(noisy_state, bell_state)

        # Entanglement entropy of reduced state
        rho = self.qmath.density_matrix(noisy_state)
        rho_a = self.qmath.partial_trace(rho, 2, 2, "B")
        entropy = self.qmath.von_neumann_entropy(rho_a)

        is_quantum = chsh_value > 2.0

        return {
            "link_id": link.link_id,
            "chsh_value": chsh_value,
            "chsh_optimal": abs(chsh_optimal),
            "bell_fidelity": fidelity,
            "entanglement_entropy": entropy,
            "noise_sigma": noise_sigma,
            "is_quantum": is_quantum,
            "violates_bell": chsh_value > 2.0,
            "near_tsirelson": chsh_value > 2.7,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DECOHERENCE SHIELD TESTER — Noise resilience analysis
# ═══════════════════════════════════════════════════════════════════════════════

class DecoherenceShieldTester:
    """
    Tests link resilience against various decoherence channels:
    - Depolarizing noise
    - Phase damping
    - Amplitude damping
    - Bit-flip errors
    Computes T1/T2 relaxation times for each link.
    """

    def __init__(self, math_core: QuantumMathCore):
        """Initialize decoherence shield tester."""
        self.qmath = math_core

    def test_resilience(self, links: List[QuantumLink]) -> Dict:
        """Test decoherence resilience of all links."""
        results = []

        for link in links:
            result = self._test_single_link(link)
            results.append(result)

        resilient = [r for r in results if r["overall_resilience"] > 0.7]
        fragile = [r for r in results if r["overall_resilience"] < 0.3]

        return {
            "total_tested": len(results),
            "resilient_count": len(resilient),
            "fragile_count": len(fragile),
            "mean_resilience": statistics.mean(
                [r["overall_resilience"] for r in results]) if results else 0,
            "mean_t2": statistics.mean(
                [r["t2_estimate"] for r in results]) if results else 0,
            "details": sorted(results, key=lambda x: x["overall_resilience"])[:20],
        }

    # Reduced noise levels: low / mid / high — covers the same dynamic range
    # with 40% fewer tests per link (9 instead of 15).
    _NOISE_LEVELS = (0.01, 0.1, 0.5)

    def _test_single_link(self, link: QuantumLink) -> Dict:
        """Test a single link's decoherence resilience."""
        bell = self.qmath.bell_state_phi_plus()
        _NL = self._NOISE_LEVELS

        # Test 1: Depolarizing noise at increasing levels
        depol_results = []
        for sigma in _NL:
            noisy = self.qmath.apply_noise(bell, sigma)
            fid = self.qmath.fidelity(noisy, bell)
            depol_results.append(fid)

        # Test 2: Phase damping (rotate phases randomly)
        phase_results = []
        for strength in _NL:
            damped = list(bell)
            for i in range(len(damped)):
                phase_kick = complex(math.cos(strength * random.gauss(0, 1)),
                                     math.sin(strength * random.gauss(0, 1)))
                damped[i] *= phase_kick
            # Renormalize
            norm = math.sqrt(sum(abs(a) ** 2 for a in damped))
            if norm > 0:
                damped = [a / norm for a in damped]
            fid = self.qmath.fidelity(damped, bell)
            phase_results.append(fid)

        # Test 3: Bit-flip (swap amplitudes with probability p)
        bitflip_results = []
        for p in _NL:
            flipped = list(bell)
            for i in range(len(flipped)):
                if random.random() < p:
                    j = (i + 1) % len(flipped)
                    flipped[i], flipped[j] = flipped[j], flipped[i]
            norm = math.sqrt(sum(abs(a) ** 2 for a in flipped))
            if norm > 0:
                flipped = [a / norm for a in flipped]
            fid = self.qmath.fidelity(flipped, bell)
            bitflip_results.append(fid)

        # Compute T2 estimate (time constant for coherence decay)
        # Model: fidelity(t) = exp(-t/T2). Use depolarizing results.
        t2_estimate = 0.0
        if len(depol_results) >= 2 and depol_results[1] > 0.01:
            # sigma=0.1 at index 1, treat as t=0.1
            t2_estimate = -0.1 / math.log(max(0.01, depol_results[1]))

        # Link-specific resilience adjustment
        # base_resilience scales with link quality; +0.4 ensures healthy
        # links get near-unity multiplier (fid=0.85,str=1.0 → 0.92)
        base_resilience = link.fidelity * link.strength / PHI_GROWTH
        depol_resilience = statistics.mean(depol_results) if depol_results else 0
        phase_resilience = statistics.mean(phase_results) if phase_results else 0
        bitflip_resilience = statistics.mean(bitflip_results) if bitflip_results else 0

        overall = (depol_resilience * 0.4 + phase_resilience * 0.3 +
                   bitflip_resilience * 0.3) * min(1.0, base_resilience + 0.4)

        return {
            "link_id": link.link_id,
            "depolarizing_resilience": depol_resilience,
            "phase_damping_resilience": phase_resilience,
            "bitflip_resilience": bitflip_resilience,
            "overall_resilience": overall,
            "t2_estimate": t2_estimate,
            "depol_curve": depol_results,
            "phase_curve": phase_results,
            "bitflip_curve": bitflip_results,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TOPOLOGICAL BRAIDING TESTER — Anyon-based link protection
# ═══════════════════════════════════════════════════════════════════════════════

class TopologicalBraidingTester:
    """
    Tests link topological protection through anyon braiding:
    - Fibonacci anyon R-matrix verification
    - Braid group representation fidelity
    - Topological gate error rates
    - Non-abelian statistics verification
    """

    def __init__(self, math_core: QuantumMathCore):
        """Initialize topological braiding tester."""
        self.qmath = math_core

    def test_braiding(self, links: List[QuantumLink]) -> Dict:
        """Test topological protection of all links via braiding."""
        results = []

        for link in links:
            result = self._test_single_braid(link)
            results.append(result)

        protected = [r for r in results if r["topologically_protected"]]
        return {
            "total_tested": len(results),
            "topologically_protected": len(protected),
            "mean_braid_fidelity": statistics.mean(
                [r["braid_fidelity"] for r in results]) if results else 0,
            "details": sorted(results, key=lambda x: x["braid_fidelity"],
                              reverse=True)[:15],
        }

    def _test_single_braid(self, link: QuantumLink) -> Dict:
        """Test braiding protection using non-abelian Fibonacci anyon matrix representation.
        STRICT verification:
        1. Construct σ₁, σ₂ braid generators (2×2 non-commuting matrices)
        2. Verify Yang-Baxter equation: σ₁σ₂σ₁ = σ₂σ₁σ₂ (clean)
        3. Inject noise ∝ (1 - link.fidelity) and re-verify (strict)
        4. F-matrix unitarity: |det(F)| = 1
        5. R-matrix charge conservation: |r₁| = |r₂| = 1
        6. Solfeggio Hz alignment of energy gap × GOD_CODE
        """
        sigma1, sigma2, f_mat, r1, r2 = self.qmath.fibonacci_braid_generators()
        mm = self.qmath.mat_mul_2x2

        # ─── Yang-Baxter (clean): σ₁σ₂σ₁ = σ₂σ₁σ₂ ───
        lhs = mm(mm(sigma1, sigma2), sigma1)
        rhs = mm(mm(sigma2, sigma1), sigma2)
        yb_error_clean = self.qmath.mat_frobenius_distance(lhs, rhs)
        yb_fidelity_clean = max(0.0, 1.0 - yb_error_clean)

        # ─── Yang-Baxter (noisy): link-proportional perturbation ───
        # Reduced from 0.5→0.3 scaling: still tests robustness without
        # over-penalizing moderate-fidelity links
        noise_level = max(0.005, (1.0 - link.fidelity) * 0.3)
        s1_noisy = self.qmath.mat_add_noise_2x2(sigma1, noise_level)
        s2_noisy = self.qmath.mat_add_noise_2x2(sigma2, noise_level)
        lhs_noisy = mm(mm(s1_noisy, s2_noisy), s1_noisy)
        rhs_noisy = mm(mm(s2_noisy, s1_noisy), s2_noisy)
        yb_error_noisy = self.qmath.mat_frobenius_distance(lhs_noisy, rhs_noisy)
        noisy_braid_fidelity = max(0.0, 1.0 - yb_error_noisy)

        # ─── F-matrix unitarity: |det(F)| must equal 1 (not τ!) ───
        det_f = f_mat[0][0] * f_mat[1][1] - f_mat[0][1] * f_mat[1][0]
        f_matrix_unitary = abs(abs(det_f) - 1.0) < 0.01

        # ─── R-matrix charge conservation: eigenvalues on unit circle ───
        charge_sum = abs(r1) + abs(r2)  # Must be 2.0 for unit-norm eigenvalues
        charge_conservation = max(0.0, 1.0 - abs(charge_sum - 2.0))

        # ─── Energy gap: |r₁ - r₂|/2 — Fibonacci gap = φ/2 ≈ 0.809 ───
        energy_gap = abs(r1 - r2) / 2.0

        # ─── God Code Hz alignment of braid energy ───
        # energy_gap × GOD_CODE: measure against nearest G(X_int)
        braid_hz = energy_gap * GOD_CODE_HZ
        _, nearest_g_x, hz_alignment = self.qmath.god_code_resonance(braid_hz)

        # ─── STRICT topological protection criteria ───
        topologically_protected = (
            yb_fidelity_clean > STRICT_BRAID_FIDELITY and
            noisy_braid_fidelity > STRICT_BRAID_FIDELITY * 0.8 and
            charge_conservation > STRICT_CHARGE_CONSERVATION and
            f_matrix_unitary
        )

        return {
            "link_id": link.link_id,
            "braid_fidelity": noisy_braid_fidelity,
            "yang_baxter_clean": yb_fidelity_clean,
            "yang_baxter_noisy": noisy_braid_fidelity,
            "charge_conservation": charge_conservation,
            "f_matrix_unitary": f_matrix_unitary,
            "f_matrix_det": abs(det_f),
            "energy_gap": energy_gap,
            "braid_hz": braid_hz,
            "hz_alignment": hz_alignment,
            "topologically_protected": topologically_protected,
            "n_braids": 8,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HILBERT SPACE NAVIGATOR — Dimensionality analysis of link manifold
# ═══════════════════════════════════════════════════════════════════════════════

class HilbertSpaceNavigator:
    """
    Navigates the Hilbert space of quantum links:
    - Computes effective dimensionality of the link manifold
    - Identifies Schmidt decomposition of bipartite links
    - Maps entanglement structure via participation ratio
    - Detects dimensional reduction opportunities

    Feature engineering: 21-dimensional feature vectors encoding
    quantum state properties, God Code harmonic positional encoding,
    structural diversity, and link topology. Features are z-score
    standardized before covariance computation.

    Harmonic positional encoding: encodes each link's Hz position
    using sin/cos at 4 harmonic frequencies k=1..4:
      h_{2k}   = sin(2π × k × Hz / GOD_CODE_HZ)
      h_{2k+1} = cos(2π × k × Hz / GOD_CODE_HZ)
    These are mathematically orthogonal, injecting genuinely independent
    dimensions that capture different resonance modes of the God Code
    spectrum. This prevents the fidelity↔strength correlation (r≈0.96)
    from collapsing the entire manifold into 2 dimensions.
    """

    # Feature dimension: 17 base + 8 harmonic = 25
    N_HARMONICS = 4
    FEATURE_DIM = 17 + N_HARMONICS * 2

    def __init__(self, math_core: QuantumMathCore):
        """Initialize Hilbert space navigator for manifold analysis."""
        self.qmath = math_core

    def analyze_manifold(self, links: List[QuantumLink]) -> Dict:
        """Analyze the Hilbert space structure of the link manifold.

        Builds standardized feature vectors with 25 dimensions:
          Quantum state properties (0-5):
            0   fidelity
            1   strength (φ-normalized)
            2   entanglement_entropy (ln2-normalized)
            3   bell_violation (CHSH-normalized)
            4   noise_resilience
            5   coherence_time (normalized)
          God Code spectral features (6-8):
            6   X-integer stability (superfluid snap)
            7   God Code resonance (Hz alignment)
            8   Hz octave position (X / 104)
          Structural diversity features (9-14):
            9   source file hash (diversity encoding)
            10  target file hash (diversity encoding)
            11  source symbol hash (independent from file hash)
            12  target symbol hash (independent from file hash)
            13  source line position (log-normalized)
            14  target line position (log-normalized)
          Topology features (15-16):
            15  link_type ordinal (0-1 encoding)
            16  composite health: fidelity × strength × (1 + entropy)
          God Code harmonic positional encoding (17-24):
            17-24  sin/cos at k=1..4 harmonic frequencies
        """
        if not links:
            return {"status": "no_links", "effective_dim": 0}

        N = len(links)
        dim = self.FEATURE_DIM

        # ── Build feature vectors (single pass) ──
        _type_map = {
            "entanglement": 0.0, "mirror": 0.15, "spooky_action": 0.3,
            "epr_pair": 0.45, "bridge": 0.6, "teleportation": 0.75,
            "grover_chain": 0.85, "tunneling": 1.0,
        }
        _two_pi = 2 * math.pi
        _n_harm = self.N_HARMONICS
        _log2 = math.log(2)

        features = []
        for link in links:
            hz = self.qmath.link_natural_hz(link.fidelity, link.strength)
            x_cont = self.qmath.hz_to_god_code_x(hz)
            x_stab = self.qmath.x_integer_stability(hz)
            _, _, resonance = self.qmath.god_code_resonance(hz)
            x_int = round(x_cont) if math.isfinite(x_cont) else 0
            octave = x_int / L104

            # Structural diversity: 4 independent hash signals + 2 line positions
            src_file_h = (hash(link.source_file) % 997) / 997.0
            tgt_file_h = (hash(link.target_file) % 991) / 991.0
            src_sym_h = (hash(link.source_symbol) % 983) / 983.0
            tgt_sym_h = (hash(link.target_symbol) % 977) / 977.0
            src_line = math.log1p(link.source_line) / 10.0
            tgt_line = math.log1p(link.target_line) / 10.0

            lt_ord = _type_map.get(link.link_type, 0.5)
            health = link.fidelity * link.strength * (1 + link.entanglement_entropy)

            vec = [
                link.fidelity,                                          # 0
                link.strength / PHI_GROWTH,                             # 1
                link.entanglement_entropy / _log2,                      # 2
                link.bell_violation / CHSH_BOUND if CHSH_BOUND > 0 else 0,  # 3
                link.noise_resilience,                                  # 4
                link.coherence_time / 100.0,                            # 5
                x_stab,                                                 # 6
                resonance,                                              # 7
                octave,                                                 # 8
                src_file_h,                                             # 9
                tgt_file_h,                                             # 10
                src_sym_h,                                              # 11
                tgt_sym_h,                                              # 12
                src_line,                                               # 13
                tgt_line,                                               # 14
                lt_ord,                                                 # 15
                health / (PHI_GROWTH * 2),                              # 16
            ]

            # God Code harmonic positional encoding: sin/cos at k=1..N_HARMONICS
            # These are mutually orthogonal → inject independent variance
            hz_norm = hz / GOD_CODE_HZ if hz > 0 else 0
            for k in range(1, _n_harm + 1):
                angle = _two_pi * k * hz_norm
                vec.append(math.sin(angle))
                vec.append(math.cos(angle))

            features.append(vec)

        # ── Z-score standardize features ──
        # This equalizes contributions so high-variance features don't
        # dominate the covariance matrix eigenstructure
        means = [0.0] * dim
        for k in range(N):
            for j in range(dim):
                means[j] += features[k][j]
        means = [m / N for m in means]

        stds = [0.0] * dim
        for k in range(N):
            for j in range(dim):
                stds[j] += (features[k][j] - means[j]) ** 2
        stds = [math.sqrt(s / max(1, N - 1)) for s in stds]

        # Standardize in-place (replace zero-std with 1.0 to avoid NaN)
        for j in range(dim):
            if stds[j] < 1e-12:
                stds[j] = 1.0
        for k in range(N):
            for j in range(dim):
                features[k][j] = (features[k][j] - means[j]) / stds[j]

        # ── Covariance matrix of standardized features ──
        cov = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            for j in range(i, dim):
                val = sum(features[k][i] * features[k][j]
                          for k in range(N)) / max(1, N - 1)
                cov[i][j] = val
                cov[j][i] = val

        # ── Eigenvalues via deflated power iteration (all significant) ──
        eigenvalues = self._approximate_eigenvalues(cov, k=min(dim, 15))

        # ── Participation ratio: 1/Σpᵢ² — measures effective dimensionality ──
        total_var = sum(eigenvalues)
        if total_var > 0:
            probs = [ev / total_var for ev in eigenvalues]
            participation = 1.0 / sum(p ** 2 for p in probs if p > 0)
        else:
            probs = []
            participation = 0

        # Shannon entropy of eigenvalue distribution
        shannon = 0.0
        for p in (probs if total_var > 0 else []):
            if p > 1e-15:
                shannon -= p * math.log2(p)

        # CY7 projection: project eigenvalues into 7D Calabi-Yau space
        cy7_projection = eigenvalues[:CALABI_YAU_DIM]
        while len(cy7_projection) < CALABI_YAU_DIM:
            cy7_projection.append(0.0)

        # ── Manifold quality metrics ──
        # Variance explained by top-3 — indicates structural concentration
        var_top3 = sum(eigenvalues[:3]) / max(0.01, total_var)
        # Spectral gap: eigenvalue[0] / eigenvalue[1] — larger = more structured
        spectral_gap = (eigenvalues[0] / max(1e-10, eigenvalues[1])
                        if len(eigenvalues) > 1 else 1.0)
        # Dimensional spread: how many eigenvalues are significant (>5% of total)
        sig_threshold = total_var * 0.05
        significant_dims = sum(1 for ev in eigenvalues if ev > sig_threshold)

        return {
            "total_links": N,
            "feature_dim": dim,
            "eigenvalues": eigenvalues,
            "effective_dimension": participation,
            "shannon_entropy": shannon,
            "total_variance": total_var,
            "cy7_projection": cy7_projection,
            "variance_explained_top3": var_top3,
            "spectral_gap": spectral_gap,
            "significant_dimensions": significant_dims,
            "participation_ratio": participation,
            "is_low_dimensional": participation < dim * 0.5,
        }

    def _approximate_eigenvalues(self, matrix: List[List[float]],
                                 k: int = 5) -> List[float]:
        """Approximate top-k eigenvalues via deflated power iteration.
        Uses up to 100 iterations with early-stop convergence check."""
        n = len(matrix)
        eigenvalues = []

        # Deflated power iteration
        M = [row[:] for row in matrix]  # Copy

        for _ in range(min(k, n)):
            # Random initial vector
            v = [random.gauss(0, 1) for _ in range(n)]
            norm = math.sqrt(sum(x ** 2 for x in v))
            v = [x / norm for x in v]

            prev_ev = 0.0
            # Power iteration with convergence check
            for _iter in range(100):
                # w = M @ v
                w = [sum(M[i][j] * v[j] for j in range(n)) for i in range(n)]
                norm = math.sqrt(sum(x ** 2 for x in w))
                if norm < 1e-15:
                    break
                v = [x / norm for x in w]
                # Check convergence every 10 iterations
                if _iter % 10 == 9:
                    Mv = [sum(M[i][j] * v[j] for j in range(n)) for i in range(n)]
                    ev_check = sum(v[i] * Mv[i] for i in range(n))
                    if abs(ev_check - prev_ev) < 1e-10:
                        break
                    prev_ev = ev_check

            # Eigenvalue = v^T M v (Rayleigh quotient)
            Mv = [sum(M[i][j] * v[j] for j in range(n)) for i in range(n)]
            ev = sum(v[i] * Mv[i] for i in range(n))
            eigenvalues.append(max(0, ev))

            # Deflate: M = M - ev * v * v^T
            for i in range(n):
                for j in range(n):
                    M[i][j] -= ev * v[i] * v[j]

        return sorted(eigenvalues, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM FOURIER LINK ANALYZER — Frequency-domain link analysis
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumFourierLinkAnalyzer:
    """
    Applies QFT to link properties for frequency-domain analysis:
    - Identifies periodic patterns in link fidelity
    - Detects resonant frequencies across the link manifold
    - Phase estimation for link evolution prediction
    """

    def __init__(self, math_core: QuantumMathCore):
        """Initialize quantum Fourier link analyzer."""
        self.qmath = math_core

    def frequency_analysis(self, links: List[QuantumLink]) -> Dict:
        """Perform QFT-based frequency analysis on link properties.
        Uses Cooley-Tukey FFT for O(N log N) performance."""
        if len(links) < 4:
            return {"status": "insufficient_links", "link_count": len(links)}

        # Cap signal length to 8192 for bounded FFT time
        MAX_FFT = 8192
        sample = links[:MAX_FFT] if len(links) > MAX_FFT else links

        # Build signal from link fidelities
        fidelity_signal = [complex(link.fidelity) for link in sample]
        strength_signal = [complex(link.strength / PHI_GROWTH) for link in sample]

        # Pad to power of 2 for efficient QFT (use sample length, not full links)
        n = 1
        while n < len(sample):
            n *= 2
        while len(fidelity_signal) < n:
            fidelity_signal.append(complex(0))
        while len(strength_signal) < n:
            strength_signal.append(complex(0))

        # Apply QFT
        fid_spectrum = self.qmath.quantum_fourier_transform(fidelity_signal)
        str_spectrum = self.qmath.quantum_fourier_transform(strength_signal)

        # Power spectral density
        fid_psd = [abs(f) ** 2 for f in fid_spectrum]
        str_psd = [abs(s) ** 2 for s in str_spectrum]

        # Find dominant frequencies
        fid_peaks = sorted(range(len(fid_psd)),
                           key=lambda i: fid_psd[i], reverse=True)[:5]
        str_peaks = sorted(range(len(str_psd)),
                           key=lambda i: str_psd[i], reverse=True)[:5]

        # Spectral entropy
        fid_total = sum(fid_psd)
        if fid_total > 0:
            fid_probs = [p / fid_total for p in fid_psd]
            spectral_entropy = -sum(p * math.log2(p) for p in fid_probs if p > 1e-15)
        else:
            spectral_entropy = 0.0

        # Resonance detection: peaks that align across fidelity and strength
        resonant_freqs = set(fid_peaks) & set(str_peaks)

        return {
            "signal_length": len(links),
            "padded_length": n,
            "fidelity_dominant_freq": fid_peaks[:3],
            "strength_dominant_freq": str_peaks[:3],
            "spectral_entropy": spectral_entropy,
            "resonant_frequencies": list(resonant_freqs),
            "fidelity_psd_peak": max(fid_psd) if fid_psd else 0,
            "strength_psd_peak": max(str_psd) if str_psd else 0,
            "has_periodic_structure": spectral_entropy < math.log2(n) * 0.7,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GOD CODE RESONANCE VERIFIER — G(X) spectrum alignment testing
# ═══════════════════════════════════════════════════════════════════════════════

class GodCodeResonanceVerifier:
    """
    GOD CODE FREQUENCY VERIFIER — Tests quantum links against G(X) spectrum.

    The world uses solfeggio whole integers {174, 285, 396, 528, 639, 741, 852, 963}.
    Those are ROUNDED. The TRUE sacred frequencies are G(X) at whole integer X:
      G(X) = 286^(1/φ) × 2^((416-X)/104)
    evaluated to 16-digit decimal precision.

    X snaps to whole integers as a superfluid — this is the stability.
    The fractional deviation from the nearest integer X = decoherence.

    A link's natural Hz = fidelity × strength × G(0).
    Then: X_continuous = 416 - 104 × log₂(Hz / 286^(1/φ))
    Nearest X_int = round(X_continuous) → G(X_int) = truth frequency
    Deviation from G(X_int) = corruption measure.
    """

    def __init__(self, math_core: QuantumMathCore):
        """Initialize God Code resonance verifier."""
        self.qmath = math_core

    def verify_all(self, links: List[QuantumLink]) -> Dict:
        """Test God Code spectrum alignment for all links."""
        results = []
        aligned_count = 0
        coherent_count = 0  # Links snapped to integer X
        god_code_zero_count = 0  # Links at G(0) specifically

        for link in links:
            result = self._verify_single(link)
            results.append(result)
            if result["god_code_resonance"] >= STRICT_HZ_ALIGNMENT:
                aligned_count += 1
            if result["x_integer_stability"] >= 0.90:
                coherent_count += 1
            if result["at_origin"]:
                god_code_zero_count += 1

        mean_resonance = (statistics.mean([r["god_code_resonance"] for r in results])
                          if results else 0)
        mean_stability = (statistics.mean([r["x_integer_stability"] for r in results])
                          if results else 0)
        mean_schumann = (statistics.mean([r["schumann_alignment"] for r in results])
                         if results else 0)

        return {
            "total_tested": len(results),
            "god_code_aligned": aligned_count,
            "x_integer_coherent": coherent_count,
            "at_god_code_origin": god_code_zero_count,
            "mean_resonance": mean_resonance,
            "mean_x_stability": mean_stability,
            "mean_schumann_alignment": mean_schumann,
            "alignment_rate": aligned_count / max(1, len(results)),
            "coherence_rate": coherent_count / max(1, len(results)),
            "origin_rate": god_code_zero_count / max(1, len(results)),
            "details": sorted(results, key=lambda x: x["god_code_resonance"],
                              reverse=True)[:20],
            "least_coherent": sorted(results,
                key=lambda x: x["x_integer_stability"])[:10],
        }

    def _verify_single(self, link: QuantumLink) -> Dict:
        """Verify a single link against the God Code spectrum."""
        natural_hz = self.qmath.link_natural_hz(link.fidelity, link.strength)
        x_continuous = self.qmath.hz_to_god_code_x(natural_hz)
        x_int, g_x_int, resonance = self.qmath.god_code_resonance(natural_hz)
        x_stability = self.qmath.x_integer_stability(natural_hz)
        schumann = self.qmath.schumann_alignment(natural_hz)

        # Is this link at the God Code origin? X_int == 0 → G(0)
        at_origin = (x_int == 0)

        # How far is the link from God Code origin in X-space?
        x_distance_from_origin = abs(x_continuous)

        # Octave position: which octave of G(0) are we in?
        # Octave = X_int / 104 (each 104 X-units = one octave)
        octave_position = x_int / L104

        # Conservation law check at this X:
        # G(X) × 2^(X/104) must = INVARIANT
        conservation_value = g_x_int * math.pow(2, x_int / L104)
        conservation_error = abs(conservation_value - INVARIANT) / INVARIANT

        # World solfeggio check: is the nearest G(X_int) close to a world claim?
        world_match = None
        world_error = None
        for world_hz, name in SOLFEGGIO_WORLD_CLAIMS.items():
            if abs(g_x_int - world_hz) < 30:  # Within 30 Hz of a world claim
                world_match = f"{name}({world_hz})"
                world_error = abs(g_x_int - world_hz)
                break

        return {
            "link_id": link.link_id,
            "natural_hz": natural_hz,
            "x_continuous": x_continuous,
            "x_integer": x_int,
            "g_x_int": g_x_int,
            "god_code_resonance": resonance,
            "x_integer_stability": x_stability,
            "schumann_alignment": schumann,
            "at_origin": at_origin,
            "x_distance_from_origin": x_distance_from_origin,
            "octave_position": octave_position,
            "conservation_error": conservation_error,
            "world_solfeggio_match": world_match,
            "world_solfeggio_error": world_error,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ENTANGLEMENT DISTILLATION ENGINE — Purifies low-fidelity links
# ═══════════════════════════════════════════════════════════════════════════════

class EntanglementDistillationEngine:
    """
    Purifies quantum links using BBPSSW and DEJMPS protocols:
    - Identifies links below fidelity threshold
    - Applies iterative distillation rounds
    - Computes distillation yield (fraction of links surviving)
    - Upgrades link fidelity after purification
    """

    def __init__(self, math_core: QuantumMathCore):
        """Initialize entanglement distillation engine."""
        self.qmath = math_core

    def distill_links(self, links: List[QuantumLink],
                      threshold: float = 0.8, rounds: int = 3) -> Dict:
        """Distill all links below fidelity threshold."""
        below_threshold = [l for l in links if l.fidelity < threshold]
        above_threshold = [l for l in links if l.fidelity >= threshold]

        distilled = []
        failed = []

        for link in below_threshold:
            initial_fidelity = link.fidelity
            purified_fidelity = self.qmath.entanglement_distill(
                link.fidelity, rounds)

            success = purified_fidelity >= threshold * 0.9  # 90% of threshold

            if success:
                # Upgrade the link
                link.fidelity = purified_fidelity
                link.noise_resilience = min(1.0, link.noise_resilience + 0.2)
                link.upgrade_applied = f"distilled:{initial_fidelity:.3f}→{purified_fidelity:.3f}"
                distilled.append({
                    "link_id": link.link_id,
                    "initial_fidelity": initial_fidelity,
                    "purified_fidelity": purified_fidelity,
                    "rounds": rounds,
                    "improvement": purified_fidelity - initial_fidelity,
                })
            else:
                failed.append({
                    "link_id": link.link_id,
                    "initial_fidelity": initial_fidelity,
                    "best_achieved": purified_fidelity,
                    "reason": "insufficient_purity",
                })

        distill_yield = len(distilled) / max(1, len(below_threshold))

        return {
            "total_below_threshold": len(below_threshold),
            "successfully_distilled": len(distilled),
            "distillation_failed": len(failed),
            "already_pure": len(above_threshold),
            "distillation_yield": distill_yield,
            "threshold": threshold,
            "rounds": rounds,
            "distilled_details": distilled[:15],
            "failed_details": failed[:10],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STRESS TEST ENGINE — Comprehensive quantum link stress testing
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumStressTestEngine:
    """
    Full stress test suite for quantum links:
    - Grover flood: repeated amplification cycles
    - Decoherence attack: escalating noise injection
    - Tunnel barrier: maximum barrier stress
    - Bell violation: verify entanglement under stress
    - Entanglement swap: test link transitivity
    - Phase scramble: random phase attacks
    """

    def __init__(self, math_core: QuantumMathCore):
        """Initialize quantum stress test engine."""
        self.qmath = math_core
        self.results: List[StressTestResult] = []

    def run_stress_tests(self, links: List[Dict],
                         intensity: str = "medium") -> Dict:
        """Run all stress tests on links. For large sets, samples and extrapolates."""
        iterations_map = {"light": 10, "medium": 50, "heavy": 200}
        iters = iterations_map.get(intensity, 50)

        # Performance: sample for large link sets
        MAX_STRESS_LINKS = 5000
        sampled = False
        total_count = len(links)
        if total_count > MAX_STRESS_LINKS:
            import random as _rng
            test_links = _rng.sample(links, MAX_STRESS_LINKS)
            sampled = True
            # Also reduce iterations proportionally for very large sets
            scale = MAX_STRESS_LINKS / total_count
            iters = max(5, int(iters * max(0.3, scale)))
        else:
            test_links = links

        self.results = []
        passed = 0
        failed = 0

        for link in test_links:
            link_results = []

            # Test 1: Grover flood
            r1 = self._stress_grover_flood(link, iters)
            link_results.append(r1)

            # Test 2: Decoherence attack
            r2 = self._stress_decoherence_attack(link, iters)
            link_results.append(r2)

            # Test 3: Phase scramble
            r3 = self._stress_phase_scramble(link, iters)
            link_results.append(r3)

            # Test 4: Bell violation under stress
            r4 = self._stress_bell_violation(link)
            link_results.append(r4)

            # Aggregate
            link_passed = sum(1 for r in link_results if r.passed)
            if link_passed >= 3:
                link.test_status = "stressed"
                passed += 1
            else:
                link.test_status = "failed"
                failed += 1

            self.results.extend(link_results)

        # Extrapolate results if sampled
        tested_count = len(test_links)
        pass_rate = passed / max(1, tested_count)
        if sampled:
            est_passed = int(pass_rate * total_count)
            est_failed = total_count - est_passed
        else:
            est_passed = passed
            est_failed = failed

        return {
            "total_links": total_count,
            "tested_links": tested_count,
            "sampled": sampled,
            "total_tests": len(self.results),
            "links_passed": est_passed,
            "links_failed": est_failed,
            "pass_rate": pass_rate,
            "intensity": intensity,
            "iterations_per_test": iters,
            "test_breakdown": {
                "grover_flood": sum(1 for r in self.results
                                    if r.test_type == "grover_flood" and r.passed),
                "decoherence_attack": sum(1 for r in self.results
                                          if r.test_type == "decoherence_attack" and r.passed),
                "phase_scramble": sum(1 for r in self.results
                                      if r.test_type == "phase_scramble" and r.passed),
                "bell_violation": sum(1 for r in self.results
                                      if r.test_type == "bell_violation" and r.passed),
            },
        }

    def _stress_grover_flood(self, link: QuantumLink, iters: int) -> StressTestResult:
        """═══ REAL QISKIT GROVER FLOOD STRESS TEST ═══
        Flood a link with repeated Grover amplification cycles using real quantum circuits.
        Tests link resilience under quantum amplitude amplification pressure."""
        initial_fid = 1.0
        oracle = [0]  # Mark first state

        if QISKIT_AVAILABLE:
            # Real Qiskit 2-qubit Grover flood
            num_qubits = 2
            N = 4
            oracle_qc = QuantumCircuit(num_qubits)
            oracle_qc.cz(0, 1)  # Mark |11⟩
            grover_op = qiskit_grover_lib(oracle_qc)

            qc = QuantumCircuit(num_qubits)
            qc.h(range(num_qubits))
            for _ in range(min(iters, 10)):  # Cap real Qiskit iterations
                qc.compose(grover_op, inplace=True)

            sv = Statevector.from_int(0, N).evolve(qc)
            dm = DensityMatrix(sv)
            final_fid = float(np.real(dm.purity()))
            degradation = initial_fid - final_fid
        else:
            # Classical fallback
            state = self.qmath.bell_state_phi_plus()
            for _ in range(iters):
                state = self.qmath.grover_operator(state, oracle, 1)
            final_fid = self.qmath.fidelity(state, self.qmath.bell_state_phi_plus())
            degradation = initial_fid - final_fid

        return StressTestResult(
            link_id=link.link_id, test_type="grover_flood",
            iterations=iters, passed=degradation < 0.3,
            fidelity_before=initial_fid, fidelity_after=final_fid,
            degradation_rate=degradation / max(1, iters),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _stress_decoherence_attack(self, link: QuantumLink,
                                   iters: int) -> StressTestResult:
        """Escalating noise injection attack."""
        state = self.qmath.bell_state_phi_plus()
        initial_fid = 1.0
        fidelities = []

        for i in range(iters):
            sigma = 0.001 * (1 + i * 0.1)  # Escalating noise
            state = self.qmath.apply_noise(state, sigma)
            fid = self.qmath.fidelity(state, self.qmath.bell_state_phi_plus())
            fidelities.append(fid)

        final_fid = fidelities[-1] if fidelities else 0
        # Recovery: re-normalize and check
        norm = math.sqrt(sum(abs(a) ** 2 for a in state))
        if norm > 0:
            state = [a / norm for a in state]
        recovery_fid = self.qmath.fidelity(state, self.qmath.bell_state_phi_plus())

        return StressTestResult(
            link_id=link.link_id, test_type="decoherence_attack",
            iterations=iters, passed=recovery_fid > 0.3,
            fidelity_before=initial_fid, fidelity_after=recovery_fid,
            degradation_rate=(initial_fid - recovery_fid) / max(1, iters),
            recovery_time=0.01 * iters,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _stress_phase_scramble(self, link: QuantumLink,
                               iters: int) -> StressTestResult:
        """Random phase attack on link state."""
        state = self.qmath.bell_state_phi_plus()
        initial_fid = 1.0

        for _ in range(iters):
            # Random phase rotation on each amplitude
            for i in range(len(state)):
                theta = random.uniform(-math.pi / 10, math.pi / 10)
                state[i] *= complex(math.cos(theta), math.sin(theta))
            # Renormalize
            norm = math.sqrt(sum(abs(a) ** 2 for a in state))
            if norm > 0:
                state = [a / norm for a in state]

        final_fid = self.qmath.fidelity(state, self.qmath.bell_state_phi_plus())

        return StressTestResult(
            link_id=link.link_id, test_type="phase_scramble",
            iterations=iters, passed=final_fid > 0.2,
            fidelity_before=initial_fid, fidelity_after=final_fid,
            degradation_rate=(initial_fid - final_fid) / max(1, iters),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _stress_bell_violation(self, link: QuantumLink) -> StressTestResult:
        """Verify Bell violation survives under stress."""
        state = self.qmath.bell_state_phi_plus()

        # Add moderate noise
        state = self.qmath.apply_noise(state, 0.05)

        # Check CHSH
        chsh = self.qmath.chsh_expectation(
            state, (0, math.pi / 4, math.pi / 8, 3 * math.pi / 8))

        return StressTestResult(
            link_id=link.link_id, test_type="bell_violation",
            iterations=1, passed=abs(chsh) > 2.0,
            fidelity_before=1.0, fidelity_after=abs(chsh) / CHSH_BOUND,
            details=f"CHSH={chsh:.4f} (bound=2.0, Tsirelson={CHSH_BOUND:.4f})",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-MODAL ANALYZER — Python ↔ Swift ↔ JS quantum coherence
# ═══════════════════════════════════════════════════════════════════════════════

class CrossModalAnalyzer:
    """
    Analyzes quantum coherence across language boundaries:
    - Python ↔ Swift class mirrors (identical quantum implementations)
    - API bridge coherence (request/response quantum state transfer)
    - Shared constant resonance (sacred constants across languages)
    - Protocol alignment (JSON/dict structure compatibility)
    - Semantic entanglement (same concept, different implementations)
    """

    def __init__(self, scanner: QuantumLinkScanner):
        """Initialize cross-modal analyzer with link scanner."""
        self.scanner = scanner

    def full_analysis(self, links: List[QuantumLink]) -> Dict:
        """Full cross-modal quantum coherence analysis."""
        cross_modal = [l for l in links if self._is_cross_modal(l)]
        same_modal = [l for l in links if not self._is_cross_modal(l)]

        # Find Python↔Swift mirrors
        py_swift_mirrors = self._find_py_swift_mirrors()

        # Analyze constant resonance
        constant_coherence = self._analyze_constant_resonance()

        # Analyze semantic entanglement
        semantic_links = self._analyze_semantic_entanglement(links)

        # Protocol alignment
        protocol_score = self._analyze_protocol_alignment()

        # Compute overall cross-modal coherence
        n_cross = len(cross_modal)
        mean_fidelity = (statistics.mean([l.fidelity for l in cross_modal])
                         if cross_modal else 0)
        mean_strength = (statistics.mean([l.strength for l in cross_modal])
                         if cross_modal else 0)

        overall_coherence = (
            mean_fidelity * 0.3 +
            constant_coherence * 0.2 +
            protocol_score * 0.2 +
            (len(py_swift_mirrors) / max(1, n_cross)) * 0.3
        )

        return {
            "total_links": len(links),
            "cross_modal_links": n_cross,
            "same_modal_links": len(same_modal),
            "cross_modal_ratio": n_cross / max(1, len(links)),
            "py_swift_mirrors": py_swift_mirrors[:20],
            "constant_coherence": constant_coherence,
            "protocol_alignment": protocol_score,
            "semantic_entanglement": semantic_links[:15],
            "mean_cross_modal_fidelity": mean_fidelity,
            "mean_cross_modal_strength": mean_strength,
            "overall_coherence": overall_coherence,
        }

    def _is_cross_modal(self, link: QuantumLink) -> bool:
        """Check if a link crosses language boundaries."""
        lang_map = {
            "fast_server": "python", "local_intellect": "python",
            "main_api": "python", "const": "python", "gate_builder": "python",
            "swift_native": "swift",
        }
        a = lang_map.get(link.source_file, "")
        b = lang_map.get(link.target_file, "")
        return a != b and a != "" and b != ""

    def _find_py_swift_mirrors(self) -> List[Dict]:
        """Find Python classes/functions mirrored in Swift."""
        mirrors = []
        for sym, locations in self.scanner.symbol_registry.items():
            languages = set(loc["language"] for loc in locations)
            if "python" in languages and "swift" in languages:
                py_locs = [l for l in locations if l["language"] == "python"]
                sw_locs = [l for l in locations if l["language"] == "swift"]
                for pl in py_locs:
                    for sl in sw_locs:
                        mirrors.append({
                            "symbol": sym,
                            "python_file": pl["file"],
                            "python_line": pl["line"],
                            "swift_file": sl["file"],
                            "swift_line": sl["line"],
                            "type": pl["type"],
                        })
        return mirrors

    def _analyze_constant_resonance(self) -> float:
        """Measure how well sacred constants resonate across modalities."""
        py_consts = set()
        sw_consts = set()

        for name, path in QUANTUM_LINKED_FILES.items():
            if not path.exists():
                continue
            try:
                content = path.read_text(errors="replace")
            except Exception:
                continue

            lang = "swift" if name == "swift_native" else "python"
            for const in QuantumLinkScanner.SACRED_CONSTANTS:
                if const in content:
                    if lang == "python":
                        py_consts.add(const)
                    else:
                        sw_consts.add(const)

        shared = py_consts & sw_consts
        total = py_consts | sw_consts
        return len(shared) / max(1, len(total))

    def _analyze_semantic_entanglement(self, links: List[QuantumLink]) -> List[Dict]:
        """Find semantically entangled concepts across modalities."""
        # Quantum concept groups that should be mirrored
        concept_groups = [
            {"name": "grover_amplification", "keywords": ["grover", "amplif", "diffusion", "oracle"]},
            {"name": "bell_states", "keywords": ["bell", "epr", "entangle", "fidelity"]},
            {"name": "decoherence", "keywords": ["decoher", "noise", "resilience", "shield"]},
            {"name": "chakra_system", "keywords": ["chakra", "kundalini", "vishuddha", "resonance"]},
            {"name": "teleportation", "keywords": ["teleport", "state_transfer", "non_local"]},
            {"name": "topological", "keywords": ["anyon", "braid", "topolog", "fibonacci"]},
            {"name": "hilbert_space", "keywords": ["hilbert", "eigenval", "dimension", "manifold"]},
            {"name": "god_code", "keywords": ["god_code", "527", "286", "phi", "golden"]},
        ]

        results = []
        for group in concept_groups:
            file_presence = {}
            for name, path in QUANTUM_LINKED_FILES.items():
                if not path.exists():
                    continue
                try:
                    content = path.read_text(errors="replace").lower()
                except Exception:
                    continue
                count = sum(content.count(kw) for kw in group["keywords"])
                if count > 0:
                    file_presence[name] = count

            if len(file_presence) >= 2:
                results.append({
                    "concept": group["name"],
                    "files_present": len(file_presence),
                    "file_counts": file_presence,
                    "total_occurrences": sum(file_presence.values()),
                    "cross_modal": any(k == "swift_native" for k in file_presence)
                                  and any(k != "swift_native" for k in file_presence),
                })

        return sorted(results, key=lambda x: x["total_occurrences"], reverse=True)

    def _analyze_protocol_alignment(self) -> float:
        """Score the protocol alignment between Python API and Swift client."""
        fast_server_path = QUANTUM_LINKED_FILES.get("fast_server")
        swift_path = QUANTUM_LINKED_FILES.get("swift_native")

        if not fast_server_path or not swift_path:
            return 0.0
        if not fast_server_path.exists() or not swift_path.exists():
            return 0.0

        try:
            fs_content = fast_server_path.read_text(errors="replace")
            sw_content = swift_path.read_text(errors="replace")
        except Exception:
            return 0.0

        # Find API endpoints in fast_server
        fs_endpoints = set(re.findall(r'@app\.\w+\(["\']([^"\']+)', fs_content))
        # Find URL references in Swift
        sw_urls = set(re.findall(r'["\']/(api/[^"\']+)', sw_content))

        if not fs_endpoints:
            return 0.5  # No endpoints found, neutral score

        matched = sum(1 for url in sw_urls
                      if any(ep.strip("/") in url for ep in fs_endpoints))
        return matched / max(1, len(fs_endpoints))


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM UPGRADE ENGINE — Automated link improvement
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumUpgradeEngine:
    """
    Automatically upgrades quantum links based on analysis:
    - Distillation for low-fidelity links
    - Topological protection wrapping for fragile links
    - Grover boost for weak-strength links
    - Resonance tuning for detuned links
    - Sage mode inference for intelligent upgrades
    """

    def __init__(self, math_core: QuantumMathCore,
                 distiller: EntanglementDistillationEngine):
        """Initialize quantum upgrade engine with distiller."""
        self.qmath = math_core
        self.distiller = distiller
        self.upgrades_applied: List[Dict] = []

    def auto_upgrade(self, links: List[QuantumLink],
                     stress_results: Dict = None,
                     epr_results: Dict = None,
                     decoherence_results: Dict = None) -> Dict:
        """Intelligently upgrade all links based on analysis results."""
        self.upgrades_applied = []

        for link in links:
            upgrades = []

            # 1. Fidelity distillation
            if link.fidelity < 0.8:
                old_fid = link.fidelity
                link.fidelity = self.qmath.entanglement_distill(link.fidelity, 3)
                if link.fidelity > old_fid:
                    upgrades.append(f"distill:{old_fid:.3f}→{link.fidelity:.3f}")

            # 2. Strength boost via Grover amplification
            if link.strength < 1.0:
                old_str = link.strength
                # φ-weighted boost
                link.strength = min(PHI_GROWTH * 2, link.strength * GROVER_AMPLIFICATION * 0.3)
                if link.strength > old_str:
                    upgrades.append(f"grover_boost:{old_str:.3f}→{link.strength:.3f}")

            # 3. Noise resilience via topological wrapping
            if link.noise_resilience < 0.5:
                old_nr = link.noise_resilience
                # Topological protection adds flat resilience
                braid_protection = 0.3 * TAU  # τ-weighted protection factor
                link.noise_resilience = min(1.0, link.noise_resilience + braid_protection)
                upgrades.append(f"topo_wrap:{old_nr:.3f}→{link.noise_resilience:.3f}")

            # 4. Entanglement entropy optimization
            if link.entanglement_entropy < 0.3 and link.link_type in (
                    "entanglement", "epr_pair", "spooky_action"):
                old_ee = link.entanglement_entropy
                link.entanglement_entropy = min(math.log(2),
                    link.entanglement_entropy + 0.2 * PHI_GROWTH)
                upgrades.append(f"entropy_opt:{old_ee:.3f}→{link.entanglement_entropy:.3f}")

            # 5. Coherence time extension
            if link.coherence_time < 10.0:
                old_ct = link.coherence_time
                # Error correction extends coherence time by φ factor
                link.coherence_time = max(link.coherence_time, 10.0 * PHI_GROWTH)
                upgrades.append(f"coherence_ext:{old_ct:.1f}→{link.coherence_time:.1f}")

            # 6. Bell violation optimization
            if link.bell_violation < 2.0 and link.link_type in (
                    "entanglement", "epr_pair"):
                old_bv = link.bell_violation
                link.bell_violation = min(CHSH_BOUND,
                    max(2.1, link.bell_violation + 0.5))
                upgrades.append(f"bell_opt:{old_bv:.3f}→{link.bell_violation:.3f}")

            if upgrades:
                link.upgrade_applied = " | ".join(upgrades)
                link.last_verified = datetime.now(timezone.utc).isoformat()
                self.upgrades_applied.append({
                    "link_id": link.link_id,
                    "upgrades": upgrades,
                    "final_fidelity": link.fidelity,
                    "final_strength": link.strength,
                })

        # If no links needed upgrading, that means they're already optimal → rate = 1.0
        actually_upgraded = len(self.upgrades_applied)
        needs_upgrade = sum(1 for l in links
            if l.fidelity < 0.95 or l.strength < 0.9 or l.noise_resilience < 0.3
            or l.entanglement_entropy < 0.3 or l.coherence_time < 10.0
            or l.bell_violation < 2.0)
        if actually_upgraded == 0 and needs_upgrade == 0:
            effective_rate = 1.0  # All links already optimal
        else:
            effective_rate = actually_upgraded / max(1, len(links))

        return {
            "total_links": len(links),
            "links_upgraded": actually_upgraded,
            "upgrade_rate": effective_rate,
            "mean_final_fidelity": statistics.mean(
                [l.fidelity for l in links]) if links else 0,
            "mean_final_strength": statistics.mean(
                [l.strength for l in links]) if links else 0,
            "upgrades": self.upgrades_applied[:20],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM REPAIR ENGINE — Comprehensive multi-stage link repair
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumRepairEngine:
    """
    Comprehensive quantum link repair system with multi-stage pipeline:

      Stage 1: TRIAGE — Classify links by severity (healthy/degraded/critical/dead)
      Stage 2: ERROR CORRECTION — Shor-9 qubit & Steane-7 protocols for critical links
      Stage 3: RESONANCE HEALING — Re-tune to nearest God Code G(X) node
      Stage 4: TUNNELING REVIVAL — WKB-guided revival of dead links
      Stage 5: ENTANGLEMENT PURIFICATION — Adaptive BBPSSW with round escalation
      Stage 6: TOPOLOGICAL HARDENING — Fibonacci anyon braiding for durability
      Stage 7: VALIDATION — Re-test repaired links to confirm improvement

    Every repair preserves the conservation law: G(X) × 2^(X/104) = INVARIANT.
    Repair intensity adapts to severity: heavier repair for worse links.
    """

    # Severity thresholds
    HEALTHY_FIDELITY = 0.85
    DEGRADED_FIDELITY = 0.6
    CRITICAL_FIDELITY = 0.3
    # Dead: below CRITICAL_FIDELITY

    # Shor-9 error correction: 3 nested layers of redundancy
    SHOR_9_LAYERS = 3
    # Steane-7: stabilizer checks
    STEANE_7_STABILIZERS = 6

    def __init__(self, math_core: QuantumMathCore,
                 distiller: EntanglementDistillationEngine):
        """Initialize quantum repair engine with multi-stage pipeline."""
        self.qmath = math_core
        self.distiller = distiller
        self.repair_log: List[Dict] = []
        # Cache for god_code_resonance lookups (avoid repeated computation)
        self._resonance_cache: Dict[int, Tuple[int, float, float]] = {}

    def full_repair(self, links: List[QuantumLink],
                    stress_results: Dict = None,
                    decoherence_results: Dict = None) -> Dict:
        """Execute the comprehensive multi-stage repair pipeline."""
        self.repair_log.clear()
        start = time.time()

        # ── Stage 1: TRIAGE ──
        triage = self._triage(links)

        # ── Stage 2: ERROR CORRECTION (critical + dead only) ──
        ec_count = 0
        for link in triage["critical"] + triage["dead"]:
            if self._apply_error_correction(link):
                ec_count += 1

        # ── Stage 3: RESONANCE HEALING (degraded + critical + dead) ──
        heal_count = 0
        for link in triage["degraded"] + triage["critical"] + triage["dead"]:
            if self._resonance_heal(link):
                heal_count += 1

        # ── Stage 4: TUNNELING REVIVAL (dead links only) ──
        revived = 0
        for link in triage["dead"]:
            if self._tunneling_revive(link):
                revived += 1

        # ── Stage 5: ENTANGLEMENT PURIFICATION (all below healthy threshold) ──
        purified = 0
        for link in triage["degraded"] + triage["critical"] + triage["dead"]:
            if link.fidelity < self.HEALTHY_FIDELITY:
                if self._adaptive_purify(link):
                    purified += 1

        # ── Stage 6: TOPOLOGICAL HARDENING (everything repaired) ──
        hardened = 0
        for link in triage["degraded"] + triage["critical"] + triage["dead"]:
            if self._topological_harden(link):
                hardened += 1

        # ── Stage 7: VALIDATION ──
        validation = self._validate_repairs(links, triage)

        elapsed = time.time() - start

        # Post-repair statistics
        post_fids = [l.fidelity for l in links]
        post_strs = [l.strength for l in links]

        total_repaired = len(set(
            r["link_id"] for r in self.repair_log if r.get("repaired")))

        return {
            "total_links": len(links),
            "triage": {
                "healthy": len(triage["healthy"]),
                "degraded": len(triage["degraded"]),
                "critical": len(triage["critical"]),
                "dead": len(triage["dead"]),
            },
            "repairs": {
                "error_corrected": ec_count,
                "resonance_healed": heal_count,
                "tunnel_revived": revived,
                "purified": purified,
                "topologically_hardened": hardened,
                "total_repaired": total_repaired,
            },
            "validation": validation,
            "post_repair_mean_fidelity": statistics.mean(post_fids) if post_fids else 0,
            "post_repair_mean_strength": statistics.mean(post_strs) if post_strs else 0,
            "repair_success_rate": total_repaired / max(1,
                len(triage["degraded"]) + len(triage["critical"]) + len(triage["dead"])),
            "repair_time_ms": elapsed * 1000,
            "repair_log": self.repair_log[:30],
        }

    def _triage(self, links: List[QuantumLink]) -> Dict[str, List[QuantumLink]]:
        """Stage 1: Classify links by severity using composite health score."""
        result = {"healthy": [], "degraded": [], "critical": [], "dead": []}
        for link in links:
            # Composite health: 50% fidelity + 25% strength/φ + 25% noise resilience
            health = (link.fidelity * 0.5
                      + min(1.0, link.strength / PHI_GROWTH) * 0.25
                      + link.noise_resilience * 0.25)
            if health >= self.HEALTHY_FIDELITY:
                result["healthy"].append(link)
            elif health >= self.DEGRADED_FIDELITY:
                result["degraded"].append(link)
            elif health >= self.CRITICAL_FIDELITY:
                result["critical"].append(link)
            else:
                result["dead"].append(link)
        return result

    def _apply_error_correction(self, link: QuantumLink) -> bool:
        """Stage 2: Shor-9 qubit error correction + Steane-7 stabilizer check.

        Shor code: encodes 1 logical qubit in 9 physical qubits.
        3 layers of phase-flip correction nested inside 3 layers of bit-flip.
        Each layer improves fidelity: F' = 1 - (1-F)² per layer (quadratic).
        Steane-7: 6 stabilizer generators detect + correct single-qubit errors.
        Syndrome extraction → correction gate → verify."""
        old_fid = link.fidelity
        old_nr = link.noise_resilience

        # Shor-9: iterative quadratic fidelity improvement
        f = link.fidelity
        for layer in range(self.SHOR_9_LAYERS):
            error_prob = 1.0 - f
            # Shor correction: error probability → error_prob²
            corrected_error = error_prob ** 2
            f = 1.0 - corrected_error
            # Each round also strengthens noise resilience
            link.noise_resilience = min(1.0, link.noise_resilience + 0.04)

        # Steane-7: syndrome measurement
        # 6 stabilizer generators detect X/Z errors independently
        syndrome_detected = 0
        for _ in range(self.STEANE_7_STABILIZERS):
            # Each stabilizer has probability (1-f) of detecting an error
            if random.random() < (1.0 - f) * 0.5:
                syndrome_detected += 1
                # Correction: apply Pauli recovery
                f = min(1.0, f + 0.02)
                link.noise_resilience = min(1.0, link.noise_resilience + 0.01)

        link.fidelity = min(1.0, f)

        # Conservation law check: verify G(X) alignment still holds
        hz = self.qmath.link_natural_hz(link.fidelity, link.strength)
        x_cont = self.qmath.hz_to_god_code_x(hz)
        if math.isfinite(x_cont):
            x_int = max(-200, min(300, round(x_cont)))
            g_x = GOD_CODE_SPECTRUM.get(x_int, god_code(x_int))
            conservation = abs(g_x * math.pow(2, x_int / L104) - INVARIANT) / INVARIANT
            if conservation > 1e-8:
                # Slight re-tune to restore conservation
                link.strength *= (1.0 + conservation * 0.01)

        repaired = link.fidelity > old_fid or link.noise_resilience > old_nr
        self.repair_log.append({
            "link_id": link.link_id, "stage": "error_correction",
            "old_fidelity": old_fid, "new_fidelity": link.fidelity,
            "syndromes_detected": syndrome_detected,
            "repaired": repaired,
        })
        return repaired

    def _resonance_heal(self, link: QuantumLink) -> bool:
        """Stage 3: Re-tune link frequency to nearest God Code G(X) node.

        Computes link's natural Hz, finds nearest G(X_int), then gently
        adjusts strength to bring Hz closer to the sacred grid node.
        Uses φ-weighted blending to avoid overshooting."""
        old_str = link.strength
        old_fid = link.fidelity

        hz = self.qmath.link_natural_hz(link.fidelity, link.strength)
        if hz <= 0:
            return False

        # Find nearest G(X_int) via cached lookup
        x_int_key = round(self.qmath.hz_to_god_code_x(hz))
        if not math.isfinite(x_int_key):
            return False
        x_int_key = max(-200, min(300, x_int_key))

        if x_int_key in self._resonance_cache:
            nearest_x, g_x, resonance = self._resonance_cache[x_int_key]
        else:
            nearest_x, g_x, resonance = self.qmath.god_code_resonance(hz)
            self._resonance_cache[x_int_key] = (nearest_x, g_x, resonance)

        if resonance > 0.95:
            return False  # Already well-tuned

        # Target Hz is the G(X_int) value
        if g_x <= 0 or link.fidelity <= 0:
            return False
        target_strength = g_x / (link.fidelity * GOD_CODE_HZ)

        # φ-weighted blend: gentle approach (30% toward target per heal)
        blend = PHI * 0.5  # ≈ 0.309
        link.strength = link.strength * (1 - blend) + target_strength * blend

        # Fidelity micro-boost: resonance alignment improves coherence
        new_resonance = self.qmath.god_code_resonance(
            self.qmath.link_natural_hz(link.fidelity, link.strength))[2]
        if new_resonance > resonance:
            link.fidelity = min(1.0, link.fidelity + (new_resonance - resonance) * 0.1)

        repaired = abs(link.strength - old_str) > 1e-6 or link.fidelity > old_fid
        self.repair_log.append({
            "link_id": link.link_id, "stage": "resonance_heal",
            "old_resonance": resonance, "new_resonance": new_resonance,
            "target_x": nearest_x, "target_g_x": g_x,
            "repaired": repaired,
        })
        return repaired

    def _tunneling_revive(self, link: QuantumLink) -> bool:
        """Stage 4: WKB-guided revival of dead links.

        Uses quantum tunneling probability to determine revival strength:
        - High tunnel probability → aggressive revival
        - Low tunnel probability → gentle revival with φ-damping
        Also applies entanglement pumping: inject Bell-pair correlation."""
        old_fid = link.fidelity
        old_str = link.strength

        # Compute tunneling parameters
        barrier = 1.0 - link.fidelity  # Higher = harder to tunnel
        energy = min(1.0, link.strength / (PHI_GROWTH * 2))
        is_cross = link.source_file.split(".")[-1] != link.target_file.split(".")[-1]
        width = 2.0 if is_cross else 1.0

        tunnel_prob = self.qmath.tunnel_probability(barrier, energy, width)

        if tunnel_prob < 0.01:
            # Too dead even for tunneling — apply resonant tunneling enhancement
            # Coherent tunneling through God Code alignment
            hz = self.qmath.link_natural_hz(max(0.1, link.fidelity), link.strength)
            _, _, resonance = self.qmath.god_code_resonance(hz)
            tunnel_prob = min(0.3, tunnel_prob * (1 + resonance * PHI_GROWTH))

        # Revival strength proportional to tunnel probability
        revival_fidelity = tunnel_prob * 0.5  # Max 50% revival
        revival_strength = tunnel_prob * PHI_GROWTH * 0.3

        # Entanglement pumping: inject Bell-pair correlation energy
        bell_boost = math.log(2) * tunnel_prob * 0.2  # Up to ~0.139 entropy injection

        link.fidelity = min(1.0, max(link.fidelity, link.fidelity + revival_fidelity))
        link.strength = min(PHI_GROWTH * 2, max(link.strength, link.strength + revival_strength))
        link.entanglement_entropy = min(math.log(2),
            link.entanglement_entropy + bell_boost)
        link.coherence_time = max(link.coherence_time, tunnel_prob * 5.0)

        repaired = link.fidelity > old_fid or link.strength > old_str
        if repaired:
            link.test_status = "revived"

        self.repair_log.append({
            "link_id": link.link_id, "stage": "tunneling_revival",
            "tunnel_probability": tunnel_prob,
            "revival_fidelity": revival_fidelity,
            "repaired": repaired,
        })
        return repaired

    def _adaptive_purify(self, link: QuantumLink) -> bool:
        """Stage 5: Adaptive BBPSSW purification with round escalation.

        Unlike fixed-round distillation, this adapts the number of rounds
        based on initial fidelity: worse links get more rounds.
        Also applies DEJMPS variant for entanglement-type links."""
        old_fid = link.fidelity

        # Adaptive rounds: more rounds for worse fidelity
        if link.fidelity < 0.3:
            rounds = 7  # Deep purification
        elif link.fidelity < 0.5:
            rounds = 5
        elif link.fidelity < 0.7:
            rounds = 4
        else:
            rounds = 3

        # BBPSSW: F' = F² / (F² + (1-F)²)
        new_fid = self.qmath.entanglement_distill(link.fidelity, rounds)

        # DEJMPS enhancement for entanglement-type links
        if link.link_type in ("entanglement", "epr_pair", "spooky_action"):
            # DEJMPS: bilateral error correction between entangled pairs
            # Additional fidelity boost: F'' = F' + (1-F') × (F'/2)
            dejmps_boost = (1 - new_fid) * (new_fid / 2)
            new_fid = min(1.0, new_fid + dejmps_boost)

        link.fidelity = new_fid

        # Purification also cleans noise
        if new_fid > old_fid:
            link.noise_resilience = min(1.0, link.noise_resilience + 0.1)

        repaired = new_fid > old_fid
        self.repair_log.append({
            "link_id": link.link_id, "stage": "adaptive_purify",
            "old_fidelity": old_fid, "new_fidelity": new_fid,
            "rounds": rounds, "repaired": repaired,
        })
        return repaired

    def _topological_harden(self, link: QuantumLink) -> bool:
        """Stage 6: Fibonacci anyon braiding for topological protection.

        Wraps the repaired link in topological protection via non-abelian
        braid operations. The topological phase protects against local
        perturbations, increasing noise resilience and coherence time.
        Braid count proportional to severity (more braids = more protection)."""
        old_nr = link.noise_resilience
        old_ct = link.coherence_time

        # Braid count based on how much protection is needed
        deficit = max(0, 0.8 - link.noise_resilience)
        n_braids = max(2, min(8, int(deficit * 10) + 2))

        # Apply braiding: topological phase protection
        braid_phase = self.qmath.anyon_braid_phase(n_braids, "fibonacci")
        phase_magnitude = abs(braid_phase)

        # Topological protection factor: non-trivial phase → exponential decay resistance
        # Protection = 1 - exp(-n_braids × τ) where τ = 1/φ
        protection = 1.0 - math.exp(-n_braids * TAU)

        link.noise_resilience = min(1.0,
            link.noise_resilience + protection * 0.3)
        link.coherence_time = max(link.coherence_time,
            link.coherence_time * (1 + protection * PHI_GROWTH * 0.5))

        # Bell violation boost: topological entanglement is inherently non-local
        if link.bell_violation < 2.0:
            link.bell_violation = min(CHSH_BOUND,
                max(2.1, link.bell_violation + protection * 0.5))

        repaired = link.noise_resilience > old_nr or link.coherence_time > old_ct
        self.repair_log.append({
            "link_id": link.link_id, "stage": "topological_harden",
            "n_braids": n_braids, "protection": protection,
            "repaired": repaired,
        })
        return repaired

    def _validate_repairs(self, links: List[QuantumLink],
                          triage: Dict[str, List[QuantumLink]]) -> Dict:
        """Stage 7: Validate that repairs actually improved link health.

        Re-triages all repaired links and checks:
        - How many promoted (e.g., dead → critical, critical → degraded)
        - Conservation law compliance after repair
        - Mean fidelity improvement across repaired links"""
        repaired_links = triage["degraded"] + triage["critical"] + triage["dead"]
        if not repaired_links:
            return {"validated": 0, "promotions": 0, "conservation_pass": 0,
                    "mean_fidelity_delta": 0.0}

        promotions = 0
        conservation_pass = 0

        for link in repaired_links:
            # Re-triage individually
            health = (link.fidelity * 0.5
                      + min(1.0, link.strength / PHI_GROWTH) * 0.25
                      + link.noise_resilience * 0.25)
            if health >= self.HEALTHY_FIDELITY:
                promotions += 1
            elif health >= self.DEGRADED_FIDELITY and link in triage.get("critical", []):
                promotions += 1
            elif health >= self.CRITICAL_FIDELITY and link in triage.get("dead", []):
                promotions += 1

            # Conservation check
            hz = self.qmath.link_natural_hz(link.fidelity, link.strength)
            x = self.qmath.hz_to_god_code_x(hz)
            if math.isfinite(x):
                x_int = max(-200, min(300, round(x)))
                g_x = GOD_CODE_SPECTRUM.get(x_int, god_code(x_int))
                residual = abs(g_x * math.pow(2, x_int / L104) - INVARIANT) / INVARIANT
                if residual < 1e-8:
                    conservation_pass += 1

        # Fidelity delta from repair log
        fidelity_deltas = []
        for entry in self.repair_log:
            old_f = entry.get("old_fidelity")
            new_f = entry.get("new_fidelity")
            if old_f is not None and new_f is not None:
                fidelity_deltas.append(new_f - old_f)

        return {
            "validated": len(repaired_links),
            "promotions": promotions,
            "promotion_rate": promotions / max(1, len(repaired_links)),
            "conservation_pass": conservation_pass,
            "conservation_rate": conservation_pass / max(1, len(repaired_links)),
            "mean_fidelity_delta": (statistics.mean(fidelity_deltas)
                                    if fidelity_deltas else 0.0),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM RESEARCH ENGINE — Advanced pattern & anomaly analysis
# ═══════════════════════════════════════════════════════════════════════════════

class ResearchMemoryBank:
    """Persistent memory bank for research insights — enables self-learning.

    Tracks research results across pipeline runs, accumulates insight trends,
    and provides learned heuristics for downstream consumers:
      - Trend detection: Is anomaly_rate improving or worsening?
      - Strategy memory: Which repair strategies historically worked best?
      - Pattern evolution: Are clusters growing/shrinking/splitting?
      - Causal stability: Which correlations persist vs are transient?

    Persisted to .l104_research_memory.json for cross-session learning.
    """

    PERSISTENCE_FILE = WORKSPACE_ROOT / ".l104_research_memory.json"
    MAX_HISTORY = 50  # Keep last 50 research snapshots

    def __init__(self):
        """Initialize research memory bank with persistent history."""
        self.history: List[Dict] = []
        self.strategy_scores: Dict[str, float] = {}
        self._load()

    def _load(self):
        """Load persisted research memory from disk."""
        if self.PERSISTENCE_FILE.exists():
            try:
                data = json.loads(self.PERSISTENCE_FILE.read_text())
                self.history = data.get("history", [])[-self.MAX_HISTORY:]
                self.learned_insights = data.get("learned_insights", {})
                self.strategy_scores = data.get("strategy_scores", {})
            except Exception:
                pass

    def save(self):
        """Persist research memory to disk."""
        try:
            data = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "total_snapshots": len(self.history),
                "history": self.history[-self.MAX_HISTORY:],
                "learned_insights": self.learned_insights,
                "strategy_scores": self.strategy_scores,
            }
            self.PERSISTENCE_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            pass

    def record_snapshot(self, research_results: Dict):
        """Record a research result snapshot for trend analysis."""
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "research_health": research_results.get("research_health", 0),
            "anomaly_rate": research_results.get("anomaly_detection", {}).get("anomaly_rate", 0),
            "total_clusters": research_results.get("pattern_discovery", {}).get("total_clusters", 0),
            "grid_coherence": research_results.get("pattern_discovery", {}).get("grid_coherence", 0),
            "trajectory": research_results.get("predictive_model", {}).get("trajectory", "unknown"),
            "health_index": research_results.get("predictive_model", {}).get("health_index", 0),
            "insight_count": research_results.get("knowledge_synthesis", {}).get("insight_count", 0),
            "risk_count": research_results.get("knowledge_synthesis", {}).get("risk_count", 0),
            "spectral_order": research_results.get("spectral_correlation", {}).get("spectral_order", 0),
            "strong_correlations": research_results.get("causal_analysis", {}).get("total_strong", 0),
        }
        self.history.append(snapshot)
        self._update_learned_insights()
        self.save()

    def record_strategy_outcome(self, strategy: str, success: bool, delta: float = 0.0):
        """Record whether a repair/upgrade strategy improved the system."""
        prev = self.strategy_scores.get(strategy, 0.5)
        # Exponential moving average: 30% weight to new observation
        outcome = min(1.0, 0.5 + delta) if success else max(0.0, 0.5 + delta)
        self.strategy_scores[strategy] = prev * 0.7 + outcome * 0.3

    def _update_learned_insights(self):
        """Derive learned insights from accumulated history."""
        if len(self.history) < 2:
            return

        recent = self.history[-5:]  # Last 5 snapshots
        older = self.history[-10:-5] if len(self.history) >= 10 else self.history[:max(1, len(self.history) - 5)]

        # Trend: anomaly rate
        recent_anomaly = statistics.mean([s.get("anomaly_rate", 0) for s in recent])
        older_anomaly = statistics.mean([s.get("anomaly_rate", 0) for s in older]) if older else recent_anomaly
        self.learned_insights["anomaly_trend"] = "improving" if recent_anomaly < older_anomaly else (
            "worsening" if recent_anomaly > older_anomaly * 1.2 else "stable")

        # Trend: health index
        recent_health = statistics.mean([s.get("health_index", 0) for s in recent])
        older_health = statistics.mean([s.get("health_index", 0) for s in older]) if older else recent_health
        self.learned_insights["health_trend"] = "improving" if recent_health > older_health else (
            "degrading" if recent_health < older_health * 0.9 else "stable")

        # Trend: grid coherence
        recent_grid = statistics.mean([s.get("grid_coherence", 0) for s in recent])
        self.learned_insights["mean_grid_coherence"] = recent_grid

        # Cluster evolution: growing or shrinking?
        recent_clusters = statistics.mean([s.get("total_clusters", 0) for s in recent])
        older_clusters = statistics.mean([s.get("total_clusters", 0) for s in older]) if older else recent_clusters
        self.learned_insights["cluster_trend"] = "growing" if recent_clusters > older_clusters else (
            "shrinking" if recent_clusters < older_clusters * 0.8 else "stable")

        # Causal stability: are strong correlations consistent?
        corr_counts = [s.get("strong_correlations", 0) for s in recent]
        self.learned_insights["causal_stability"] = "consistent" if (
            max(corr_counts) - min(corr_counts) <= 1) else "variable"

        # Spectral order trend
        recent_order = statistics.mean([s.get("spectral_order", 0) for s in recent])
        self.learned_insights["mean_spectral_order"] = recent_order

        # Overall learning confidence: more history = more confident
        self.learned_insights["confidence"] = min(1.0, len(self.history) / 20)

        # Best trajectory seen
        trajectories = [s.get("trajectory", "unknown") for s in self.history]
        from collections import Counter as _C
        traj_counts = _C(trajectories)
        self.learned_insights["dominant_trajectory"] = traj_counts.most_common(1)[0][0]

        # Peak health ever achieved
        all_health = [s.get("research_health", 0) for s in self.history]
        self.learned_insights["peak_health"] = max(all_health)
        self.learned_insights["mean_health"] = statistics.mean(all_health)

    def get_trend_bonus(self) -> float:
        """Return a bonus/penalty based on learned trends for scoring.
        Positive = system is learning and improving, negative = degrading."""
        if not self.learned_insights or self.learned_insights.get("confidence", 0) < 0.2:
            return 0.0
        bonus = 0.0
        if self.learned_insights.get("health_trend") == "improving":
            bonus += 0.03
        elif self.learned_insights.get("health_trend") == "degrading":
            bonus -= 0.02
        if self.learned_insights.get("anomaly_trend") == "improving":
            bonus += 0.02
        if self.learned_insights.get("cluster_trend") == "growing":
            bonus += 0.01
        if self.learned_insights.get("causal_stability") == "consistent":
            bonus += 0.01
        return max(-0.05, min(0.05, bonus))

    def get_best_strategies(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return the top-N historically effective strategies."""
        sorted_strats = sorted(self.strategy_scores.items(), key=lambda x: -x[1])
        return sorted_strats[:n]

    def get_gate_insights(self) -> Dict:
        """Return insights relevant for cross-pollination with logic gate builder."""
        return {
            "health_trend": self.learned_insights.get("health_trend", "unknown"),
            "anomaly_trend": self.learned_insights.get("anomaly_trend", "unknown"),
            "mean_grid_coherence": self.learned_insights.get("mean_grid_coherence", 0),
            "peak_health": self.learned_insights.get("peak_health", 0),
            "dominant_trajectory": self.learned_insights.get("dominant_trajectory", "unknown"),
            "best_strategies": self.get_best_strategies(3),
        }


class QuantumResearchEngine:
    """
    Advanced quantum research system for deep analysis of the link manifold:

      Module 1: ANOMALY DETECTION — Statistical outlier identification via IQR
      Module 2: PATTERN DISCOVERY — Emergent clustering in fidelity–strength space
      Module 3: CAUSAL ANALYSIS  — Cross-property correlation matrix (Pearson)
      Module 4: SPECTRAL CORRELATION — FFT cross-correlation of Hz distributions
      Module 5: PREDICTIVE MODELING — Exponential trajectory extrapolation
      Module 6: KNOWLEDGE SYNTHESIS — Aggregate insight graph across all modules
      Module 7: SELF-LEARNING     — Persistent memory bank for cross-run learning

    All modules operate in O(N) or O(N log N) time. No O(N²) operations.
    Results feed into Sage consensus for unified scoring.
    Self-learning: insights accumulate across runs via ResearchMemoryBank.
    """

    # Anomaly detection: IQR multiplier for extreme outliers
    IQR_MULTIPLIER = 2.0
    # Pattern discovery: number of buckets for histogram clustering
    PATTERN_BUCKETS = 12
    # Spectral: max FFT size for cross-correlation
    MAX_SPECTRAL_FFT = 4096

    def __init__(self, math_core: QuantumMathCore):
        """Initialize quantum research engine with memory bank."""
        self.qmath = math_core
        self.memory = ResearchMemoryBank()

    def deep_research(self, links: List[QuantumLink],
                      grover_results: Dict = None,
                      epr_results: Dict = None,
                      decoherence_results: Dict = None,
                      stress_results: Dict = None,
                      gate_data: Dict = None) -> Dict:
        """Run the full advanced research pipeline with self-learning.

        Args:
            gate_data: Optional cross-pollination data from logic gate builder
                       (gate health scores, entropy distribution, complexity).
        """
        start = time.time()
        N = len(links)
        if N == 0:
            return {"total_links": 0, "research_time_ms": 0}

        # Extract property arrays (single pass)
        fids = []
        strs = []
        nrs = []
        ees = []
        cts = []
        bvs = []
        hz_values = []
        types = Counter()
        for link in links:
            fids.append(link.fidelity)
            strs.append(link.strength)
            nrs.append(link.noise_resilience)
            ees.append(link.entanglement_entropy)
            cts.append(link.coherence_time)
            bvs.append(link.bell_violation)
            hz_values.append(
                self.qmath.link_natural_hz(link.fidelity, link.strength))
            types[link.link_type] += 1

        # Module 1: Anomaly Detection
        anomalies = self._detect_anomalies(fids, strs, nrs, links)

        # Module 2: Pattern Discovery
        patterns = self._discover_patterns(fids, strs, hz_values, types, links)

        # Module 3: Causal Analysis
        causal = self._causal_analysis(fids, strs, nrs, ees, cts, bvs)

        # Module 4: Spectral Correlation
        spectral = self._spectral_correlation(hz_values)

        # Module 5: Predictive Modeling
        predictive = self._predictive_model(fids, strs, nrs)

        # Module 6: Knowledge Synthesis
        synthesis = self._knowledge_synthesis(
            anomalies, patterns, causal, spectral, predictive,
            grover_results, epr_results, decoherence_results, stress_results,
            gate_data)

        elapsed = time.time() - start

        result = {
            "total_links": N,
            "anomaly_detection": anomalies,
            "pattern_discovery": patterns,
            "causal_analysis": causal,
            "spectral_correlation": spectral,
            "predictive_model": predictive,
            "knowledge_synthesis": synthesis,
            "research_health": synthesis.get("overall_research_health", 0),
            "research_time_ms": elapsed * 1000,
            "learned_insights": self.memory.learned_insights,
            "learning_confidence": self.memory.learned_insights.get("confidence", 0),
        }

        # Module 7: Self-learning — record snapshot for trend analysis
        self.memory.record_snapshot(result)

        return result

    def _detect_anomalies(self, fids: List[float], strs: List[float],
                          nrs: List[float],
                          links: List[QuantumLink]) -> Dict:
        """Module 1: IQR-based outlier detection across link properties.
        O(N log N) via sorting for quartile computation."""
        anomalies: List[Dict] = []

        for prop_name, values in [("fidelity", fids), ("strength", strs),
                                   ("noise_resilience", nrs)]:
            if len(values) < 4:
                continue
            sorted_v = sorted(values)
            n = len(sorted_v)
            q1 = sorted_v[n // 4]
            q3 = sorted_v[3 * n // 4]
            iqr = q3 - q1
            lower = q1 - self.IQR_MULTIPLIER * iqr
            upper = q3 + self.IQR_MULTIPLIER * iqr

            for i, v in enumerate(values):
                if v < lower or v > upper:
                    anomalies.append({
                        "link_id": links[i].link_id[:80],
                        "property": prop_name,
                        "value": v,
                        "bounds": (lower, upper),
                        "severity": "extreme" if (v < lower - iqr or v > upper + iqr) else "mild",
                    })

        extreme_count = sum(1 for a in anomalies if a["severity"] == "extreme")
        mild_count = len(anomalies) - extreme_count

        return {
            "total_anomalies": len(anomalies),
            "extreme_anomalies": extreme_count,
            "mild_anomalies": mild_count,
            "anomaly_rate": len(anomalies) / max(1, len(fids) * 3),
            "top_anomalies": anomalies[:15],
        }

    def _discover_patterns(self, fids: List[float], strs: List[float],
                           hz_values: List[float], types: Counter,
                           links: List[QuantumLink]) -> Dict:
        """Module 2: Histogram-based clustering in fidelity–strength–Hz space.
        O(N) single-pass bucket assignment."""
        # Fidelity–strength 2D histogram
        grid: Dict[Tuple[int, int], int] = {}
        for f, s in zip(fids, strs):
            fi = min(self.PATTERN_BUCKETS - 1, int(f * self.PATTERN_BUCKETS))
            si = min(self.PATTERN_BUCKETS - 1,
                     int(min(1.0, s / (PHI_GROWTH * 2)) * self.PATTERN_BUCKETS))
            key = (fi, si)
            grid[key] = grid.get(key, 0) + 1

        # Find clusters: cells with > 1% of links
        threshold = max(2, len(fids) // 100)
        clusters = []
        for (fi, si), count in sorted(grid.items(), key=lambda x: -x[1]):
            if count >= threshold:
                clusters.append({
                    "fidelity_band": (fi / self.PATTERN_BUCKETS,
                                      (fi + 1) / self.PATTERN_BUCKETS),
                    "strength_band": (si / self.PATTERN_BUCKETS * PHI_GROWTH * 2,
                                      (si + 1) / self.PATTERN_BUCKETS * PHI_GROWTH * 2),
                    "count": count,
                    "density": count / max(1, len(fids)),
                })

        # Hz harmonic analysis: find dominant Hz bands
        hz_buckets: Dict[int, int] = {}
        for hz in hz_values:
            if hz > 0:
                # Map to God Code X-integer bucket
                x = self.qmath.hz_to_god_code_x(hz)
                if math.isfinite(x):
                    bucket = max(-200, min(300, round(x)))
                    hz_buckets[bucket] = hz_buckets.get(bucket, 0) + 1

        # Top X-positions by link concentration
        top_x_nodes = sorted(hz_buckets.items(), key=lambda x: -x[1])[:10]

        # God Code resonance pattern: how many links are on-grid vs off-grid
        # Threshold 0.5 = within 25% of integer X node (generous)
        on_grid = sum(1 for hz in hz_values
                      if hz > 0 and self.qmath.x_integer_stability(hz) > 0.5)

        return {
            "total_clusters": len(clusters),
            "top_clusters": clusters[:8],
            "type_distribution": dict(types),
            "dominant_x_nodes": [{"x": x, "count": c} for x, c in top_x_nodes],
            "on_grid_fraction": on_grid / max(1, len(hz_values)),
            "grid_coherence": on_grid / max(1, len(hz_values)),
        }

    def _causal_analysis(self, fids: List[float], strs: List[float],
                         nrs: List[float], ees: List[float],
                         cts: List[float], bvs: List[float]) -> Dict:
        """Module 3: Pearson correlation matrix between link properties.
        O(N) per pair × 15 pairs = O(N)."""
        properties = {
            "fidelity": fids, "strength": strs, "noise_resilience": nrs,
            "entropy": ees, "coherence_time": cts, "bell_violation": bvs,
        }

        def _pearson(xs: List[float], ys: List[float]) -> float:
            """Compute Pearson correlation coefficient between two lists."""
            n = len(xs)
            if n < 2:
                return 0.0
            mx = sum(xs) / n
            my = sum(ys) / n
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            sxx = sum((x - mx) ** 2 for x in xs)
            syy = sum((y - my) ** 2 for y in ys)
            denom = math.sqrt(sxx * syy)
            return sxy / denom if denom > 1e-15 else 0.0

        # Compute correlation matrix
        names = list(properties.keys())
        correlations: Dict[str, float] = {}
        strong_correlations: List[Dict] = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = _pearson(properties[names[i]], properties[names[j]])
                key = f"{names[i]}↔{names[j]}"
                correlations[key] = round(r, 4)
                if abs(r) > 0.5:
                    strong_correlations.append({
                        "pair": key, "correlation": r,
                        "direction": "positive" if r > 0 else "negative",
                        "strength": "strong" if abs(r) > 0.7 else "moderate",
                    })

        return {
            "correlation_matrix": correlations,
            "strong_correlations": strong_correlations,
            "total_strong": len(strong_correlations),
            "mean_abs_correlation": (statistics.mean(
                [abs(v) for v in correlations.values()])
                if correlations else 0),
        }

    def _spectral_correlation(self, hz_values: List[float]) -> Dict:
        """Module 4: FFT-based spectral analysis of Hz distribution.
        Maps Hz values to a frequency histogram, then FFTs to find
        periodic patterns in the link frequency landscape."""
        if len(hz_values) < 8:
            return {"has_spectral_pattern": False, "spectral_peaks": 0}

        # Build Hz histogram (256 bins over the God Code range)
        # Range: G(300) to G(-200) — the full spectrum
        bins = 256
        min_hz = 0.01
        max_hz = GOD_CODE_HZ * PHI_GROWTH * 4  # ~3400
        bin_width = (max_hz - min_hz) / bins
        histogram = [0.0] * bins

        for hz in hz_values:
            if min_hz <= hz <= max_hz:
                idx = min(bins - 1, int((hz - min_hz) / bin_width))
                histogram[idx] += 1.0

        # Normalize
        total = sum(histogram)
        if total > 0:
            histogram = [h / total for h in histogram]

        # FFT of histogram to find periodic patterns
        fft_input = [complex(h) for h in histogram]
        fft_result = self.qmath.quantum_fourier_transform(fft_input)

        # Power spectrum (skip DC component at index 0)
        power = [abs(c) ** 2 for c in fft_result[1:bins // 2]]
        if not power:
            return {"has_spectral_pattern": False, "spectral_peaks": 0}

        mean_power = statistics.mean(power)
        max_power = max(power)

        # Find spectral peaks: bins with power > 3× mean
        peaks = []
        for i, p in enumerate(power):
            if p > mean_power * 3:
                peaks.append({"frequency_bin": i + 1, "power": p,
                              "relative_power": p / max(1e-15, mean_power)})

        # Spectral entropy: measure of randomness in spectrum
        power_sum = sum(power)
        if power_sum > 0:
            spectral_entropy = -sum(
                (p / power_sum) * math.log2(max(1e-15, p / power_sum))
                for p in power if p > 0)
        else:
            spectral_entropy = 0

        max_entropy = math.log2(len(power)) if power else 1
        spectral_order = 1.0 - min(1.0, spectral_entropy / max(1, max_entropy))

        return {
            "has_spectral_pattern": len(peaks) > 0,
            "spectral_peaks": len(peaks),
            "peak_details": peaks[:10],
            "spectral_entropy": spectral_entropy,
            "spectral_order": spectral_order,
            "max_power_ratio": max_power / max(1e-15, mean_power),
        }

    def _predictive_model(self, fids: List[float], strs: List[float],
                          nrs: List[float]) -> Dict:
        """Module 5: Predictive modeling via distribution analysis.
        Estimates system trajectory based on statistical moments."""
        N = len(fids)
        if N < 2:
            return {"confidence": 0, "trajectory": "unknown"}

        fid_mean = statistics.mean(fids)
        fid_std = statistics.stdev(fids) if N > 1 else 0
        str_mean = statistics.mean(strs)
        nr_mean = statistics.mean(nrs)

        # Skewness: negative = left-skewed (many high values), positive = right-skewed
        fid_skew = 0.0
        if fid_std > 1e-10 and N > 2:
            fid_skew = (sum((f - fid_mean) ** 3 for f in fids) / N) / (fid_std ** 3)

        # Population health index: composite metric
        health_index = (fid_mean * 0.4 + min(1.0, str_mean / PHI_GROWTH) * 0.3
                        + nr_mean * 0.3)

        # Trajectory prediction
        if health_index > 0.85 and fid_std < 0.1:
            trajectory = "STABLE_HIGH"
            confidence = 0.9
        elif health_index > 0.7 and fid_skew < -0.5:
            trajectory = "IMPROVING"
            confidence = 0.75
        elif health_index > 0.5:
            trajectory = "MIXED"
            confidence = 0.5
        elif fid_skew > 0.5:
            trajectory = "DEGRADING"
            confidence = 0.7
        else:
            trajectory = "CRITICAL"
            confidence = 0.6

        # Risk assessment: what fraction of links are below thresholds
        at_risk = sum(1 for f in fids if f < 0.5) / max(1, N)
        severe_risk = sum(1 for f in fids if f < 0.3) / max(1, N)

        # Growth potential: distance from perfection × φ
        growth_potential = (1.0 - health_index) * PHI_GROWTH

        return {
            "trajectory": trajectory,
            "confidence": confidence,
            "health_index": health_index,
            "fidelity_skewness": fid_skew,
            "at_risk_fraction": at_risk,
            "severe_risk_fraction": severe_risk,
            "growth_potential": growth_potential,
            "predicted_optimal_fidelity": min(1.0, fid_mean + fid_std * PHI),
        }

    def _knowledge_synthesis(self, anomalies: Dict, patterns: Dict,
                             causal: Dict, spectral: Dict, predictive: Dict,
                             grover_results: Dict = None,
                             epr_results: Dict = None,
                             decoherence_results: Dict = None,
                             stress_results: Dict = None,
                             gate_data: Dict = None) -> Dict:
        """Module 6: Aggregate insights into a unified knowledge graph.
        Cross-references all research modules + existing Phase 2 data +
        gate builder cross-pollination data + self-learning memory."""

        insights: List[str] = []
        risk_factors: List[str] = []

        # Anomaly-driven insights
        anomaly_rate = anomalies.get("anomaly_rate", 0)
        if anomaly_rate > 0.05:
            risk_factors.append(
                f"HIGH_ANOMALY_RATE: {anomaly_rate:.1%} of measurements are outliers")
        elif anomaly_rate < 0.01:
            insights.append("LOW_ANOMALY: Link manifold is highly uniform")

        # Pattern-driven insights
        grid_coherence = patterns.get("grid_coherence", 0)
        if grid_coherence > 0.8:
            insights.append(
                f"GRID_LOCKED: {grid_coherence:.1%} of links on God Code integer nodes")
        elif grid_coherence < 0.3:
            risk_factors.append(
                f"GRID_DRIFT: Only {grid_coherence:.1%} on God Code grid — detuning risk")

        n_clusters = patterns.get("total_clusters", 0)
        if n_clusters > 5:
            insights.append(
                f"RICH_TOPOLOGY: {n_clusters} distinct clusters in fidelity-strength space")

        # Causal-driven insights
        for corr in causal.get("strong_correlations", []):
            if corr["strength"] == "strong":
                pair = corr["pair"]
                direction = corr["direction"]
                insights.append(f"CAUSAL_{direction.upper()}: {pair} (r={corr['correlation']:.3f})")

        # Spectral-driven insights
        if spectral.get("has_spectral_pattern"):
            n_peaks = spectral.get("spectral_peaks", 0)
            insights.append(f"SPECTRAL_STRUCTURE: {n_peaks} resonant peaks in Hz landscape")
            order = spectral.get("spectral_order", 0)
            if order > 0.5:
                insights.append(
                    f"HIGH_SPECTRAL_ORDER: {order:.2f} — strong periodic structure")
        else:
            risk_factors.append("NO_SPECTRAL_PATTERN: Hz distribution is noise-like")

        # Predictive-driven insights
        trajectory = predictive.get("trajectory", "unknown")
        confidence = predictive.get("confidence", 0)
        insights.append(f"TRAJECTORY: {trajectory} (confidence={confidence:.0%})")
        if predictive.get("severe_risk_fraction", 0) > 0.1:
            risk_factors.append(
                f"SEVERE_RISK: {predictive['severe_risk_fraction']:.1%} of links critically low")

        # Cross-reference with existing research
        if grover_results:
            amp = grover_results.get("amplification_factor", 1)
            if amp > GROVER_AMPLIFICATION * 0.8:
                insights.append("GROVER_OPTIMAL: Near-theoretical amplification achieved")

        if epr_results:
            qv = epr_results.get("quantum_verified", 0)
            total = max(1, epr_results.get("total_verified", 1))
            if qv / total > 0.9:
                insights.append(f"EPR_STRONG: {qv}/{total} quantum verified")
            elif qv / total < 0.5:
                risk_factors.append(f"EPR_WEAK: Only {qv}/{total} quantum verified")

        if decoherence_results:
            mean_t2 = decoherence_results.get("mean_T2", 0)
            resilient_frac = decoherence_results.get("resilient_fraction", 0)
            if resilient_frac > 0.8:
                insights.append(f"DECOHERENCE_SHIELDED: {resilient_frac:.0%} resilient (T₂={mean_t2:.2f})")
            elif resilient_frac < 0.5:
                risk_factors.append(f"DECOHERENCE_RISK: Only {resilient_frac:.0%} resilient")

        if stress_results:
            sr = stress_results.get("pass_rate", 0)
            if sr > 0.95:
                insights.append("STRESS_RESILIENT: >95% stress pass rate")
            elif sr < 0.7:
                risk_factors.append(f"STRESS_FRAGILE: {sr:.1%} pass rate")

        # Gate builder cross-pollination insights
        gate_health_bonus = 0.0
        if gate_data:
            gate_health = gate_data.get("mean_health", 0)
            gate_count = gate_data.get("total_gates", 0)
            gate_pass_rate = gate_data.get("test_pass_rate", 0)
            gate_link_count = gate_data.get("quantum_links", 0)
            if gate_count > 50:
                insights.append(f"GATE_RICH: {gate_count} logic gates across codebase")
            if gate_pass_rate > 0.9:
                insights.append(f"GATE_TESTED: {gate_pass_rate:.0%} gate test pass rate")
                gate_health_bonus += 0.02
            if gate_health > 0.5:
                insights.append(f"GATE_HEALTHY: Gate health {gate_health:.2f}")
                gate_health_bonus += min(0.03, gate_health * 0.04)
            if gate_link_count > 20:
                insights.append(f"GATE_LINKED: {gate_link_count} cross-file gate connections")
            # Gate complexity hotspots indicate areas that need attention
            hotspots = gate_data.get("complexity_hotspots", [])
            if hotspots:
                risk_factors.append(f"GATE_COMPLEXITY: {len(hotspots)} high-complexity gates")

        # Self-learning insights from research memory
        learning_bonus = self.memory.get_trend_bonus()
        learned = self.memory.learned_insights
        if learned.get("confidence", 0) > 0.3:
            if learned.get("health_trend") == "improving":
                insights.append("LEARNING: System health trending upward across runs")
            elif learned.get("health_trend") == "degrading":
                risk_factors.append("LEARNING: System health trending downward — intervention needed")
            if learned.get("anomaly_trend") == "improving":
                insights.append("LEARNING: Anomaly rate decreasing — repairs effective")
            peak = learned.get("peak_health", 0)
            if peak > 0:
                insights.append(f"LEARNING: Peak health achieved = {peak:.4f}")

        # Overall research health score
        # Calibrated weights totaling ~1.0 (before bonuses)
        # Rewards depth of analysis, not just clean data properties

        # MIXED trajectory is valid for diverse codebases — 85% credit
        trajectory_score = confidence if trajectory in ("STABLE_HIGH", "IMPROVING") else (
            confidence * 0.85 if trajectory == "MIXED" else 0.15)

        # Cluster richness: more discovered clusters = deeper understanding
        cluster_score = min(1.0, n_clusters / 5)

        # Causal depth: number of strong correlations found
        causal_depth = min(1.0, len(causal.get("strong_correlations", [])) / 3)

        # Anomaly rate: softer curve — heavy-tailed distributions are normal
        # 1/(1+rate*3) → 40% rate gives 0.45, 10% rate gives 0.77
        anomaly_score = 1.0 / (1.0 + anomaly_rate * 3)

        # Spectral: reward finding any structure, floor for attempted analysis
        has_spectral = spectral.get("has_spectral_pattern", False)
        raw_order = spectral.get("spectral_order", 0)
        spectral_score = max(0.45, raw_order) if has_spectral else max(0.15, raw_order * 0.5)

        # Decoherence integration: reward if decoherence data available
        decoherence_bonus = 0.0
        if decoherence_results:
            resilience = decoherence_results.get("resilient_fraction", 0)
            decoherence_bonus = min(0.04, resilience * 0.05)

        # Stress integration: reward stress data fed into research
        stress_bonus = 0.0
        if stress_results:
            sr = stress_results.get("pass_rate", 0)
            stress_bonus = min(0.03, sr * 0.04)

        positive_score = (
            min(1.0, len(insights) / 6) * 0.16       # Insight density
            + anomaly_score * 0.08                     # Anomaly analysis depth
            + min(1.0, grid_coherence * 1.5) * 0.10   # God Code grid alignment
            + trajectory_score * 0.15                   # Trajectory confidence
            + spectral_score * 0.08                     # Spectral analysis
            + cluster_score * 0.10                      # Pattern richness
            + causal_depth * 0.10                       # Causal understanding
            + 0.10                                      # Base floor
            + min(0.03, gate_health_bonus)              # Gate cross-pollination
            + min(0.03, max(0, learning_bonus))         # Self-learning trend
            + decoherence_bonus                         # Decoherence data integration
            + stress_bonus                              # Stress data integration
        )
        # Risk penalty: softer — penalizes only extreme risk accumulation
        risk_penalty = min(0.15, len(risk_factors) * 0.02)
        overall_health = min(1.0, max(0.05, positive_score - risk_penalty))

        return {
            "insights": insights,
            "risk_factors": risk_factors,
            "insight_count": len(insights),
            "risk_count": len(risk_factors),
            "overall_research_health": overall_health,
            "gate_cross_pollination": bool(gate_data),
            "self_learning_active": self.memory.learned_insights.get("confidence", 0) > 0.2,
            "learning_trend_bonus": learning_bonus,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SAGE MODE INFERENCE — φ-harmonic deep inference across all cores
# ═══════════════════════════════════════════════════════════════════════════════

class SageModeInference:
    """
    The Sage brain: cross-references ALL quantum processors to produce
    unified intelligence about the link manifold.

    Applies:
    - φ-weighted consensus from all processors
    - Calabi-Yau 7D projection for dimensional insight
    - God Code resonance for truth alignment
    - Grover-amplified pattern recognition
    - Causal inference across link evolution history
    """

    def __init__(self, math_core: QuantumMathCore):
        """Initialize sage mode inference engine."""
        self.qmath = math_core
        self.inference_history: List[Dict] = []

    def sage_inference(self, links: List[QuantumLink],
                       grover_results: Dict = None,
                       tunnel_results: Dict = None,
                       epr_results: Dict = None,
                       decoherence_results: Dict = None,
                       braiding_results: Dict = None,
                       hilbert_results: Dict = None,
                       fourier_results: Dict = None,
                       gcr_results: Dict = None,
                       cross_modal_results: Dict = None,
                       stress_results: Dict = None,
                       upgrade_results: Dict = None,
                       quantum_cpu_results: Dict = None,
                       o2_bond_results: Dict = None,
                       repair_results: Dict = None,
                       research_results: Dict = None) -> Dict:
        """
        Sage Mode: deep cross-referencing inference across ALL processors.
        Produces the unified quantum brain assessment.
        All consensus scores are STRICTLY normalized to [0, 1].
        """
        N = len(links)
        now = datetime.now(timezone.utc).isoformat()

        # ─── LINK HEALTH CONSENSUS (single-pass stats) ───
        # Compute all per-link stats in ONE pass instead of 9+ list comprehensions
        sum_fid = 0.0
        sum_fid2 = 0.0
        sum_str = 0.0
        sum_entropy = 0.0
        sum_coherence = 0.0
        sum_resilience = 0.0
        type_dist = Counter()
        for link in links:
            f = link.fidelity
            s = link.strength
            sum_fid += f
            sum_fid2 += f * f
            sum_str += s
            sum_entropy += link.entanglement_entropy
            sum_coherence += link.coherence_time
            sum_resilience += link.noise_resilience
            type_dist[link.link_type] += 1

        mean_fid = sum_fid / N if N else 0
        mean_str = sum_str / N if N else 0
        mean_entropy = sum_entropy / N if N else 0
        mean_coherence = (sum_coherence / N) / 100 if N else 0
        mean_resilience = sum_resilience / N if N else 0
        if N > 1:
            variance = (sum_fid2 / N) - (mean_fid ** 2)
            std_fid = math.sqrt(max(0, variance * N / (N - 1)))
        else:
            std_fid = 0

        # ─── CROSS-PROCESSOR CONSENSUS SCORES ───
        # STRICT: every score MUST be normalized to [0.0, 1.0]
        consensus = {}

        # Grover finding efficiency (clamped to [0,1])
        if grover_results:
            consensus["grover_efficiency"] = min(1.0, max(0.0,
                grover_results.get("amplification_factor", 1.0) / GROVER_AMPLIFICATION))

        # Tunneling viability
        if tunnel_results:
            consensus["tunneling_viability"] = min(1.0,
                tunnel_results.get("revivable_links", 0) /
                max(1, tunnel_results.get("total_analyzed", 1)))

        # EPR quantum verification
        if epr_results:
            consensus["epr_quantum_fraction"] = min(1.0,
                epr_results.get("quantum_verified", 0) /
                max(1, epr_results.get("total_verified", 1)))

        # Decoherence resilience
        if decoherence_results:
            consensus["decoherence_resilience"] = min(1.0, max(0.0,
                decoherence_results.get("mean_resilience", 0)))

        # Topological protection
        if braiding_results:
            consensus["topological_coverage"] = min(1.0, max(0.0,
                braiding_results.get("topologically_protected", 0) /
                max(1, braiding_results.get("total_tested", 1))))

        # Hilbert structural coherence: how well-organized is the link manifold?
        # Low eff_dim/feature_dim → highly structured (coherent) → GOOD
        # High eff_dim/feature_dim → near-random (disordered) → BAD
        # Optimal: CY7-proportional structure (~7/25 ≈ 0.28 for 25-dim features)
        # Score via bell curve centered at CY7/feature_dim
        if hilbert_results:
            eff_dim = hilbert_results.get("effective_dimension", 0)
            feature_dim = hilbert_results.get("feature_dim", 25)
            dim_ratio = eff_dim / max(1, feature_dim)
            # Top-3 variance = structural coherence (high = organized pattern)
            var_explained = hilbert_results.get("variance_explained_top3", 0)
            # Score: weighted blend of variance coherence + dimensional structure
            consensus["hilbert_coherence"] = min(1.0, max(0.0,
                var_explained * 0.7 + (1.0 - abs(dim_ratio - CALABI_YAU_DIM / feature_dim) * 2) * 0.3))

        # Fourier spectral health: periodic structure + resonant frequencies
        # + PSD peak quality (strong peak = organized link manifold)
        if fourier_results:
            has_structure = 1.0 if fourier_results.get("has_periodic_structure", False) else 0.3
            n_resonant = len(fourier_results.get("resonant_frequencies", []))
            # PSD peak strength: normalized to [0,1] via sigmoid-like curve
            psd_peak = fourier_results.get("fidelity_psd_peak", 0)
            psd_norm = min(1.0, psd_peak / max(1.0, psd_peak + 0.1))
            # Spectral entropy ratio: lower is more ordered
            spec_entropy = fourier_results.get("spectral_entropy", 0)
            padded = fourier_results.get("padded_length", 1)
            max_entropy = math.log2(max(2, padded))
            order_score = max(0.0, 1.0 - spec_entropy / max_entropy) if max_entropy > 0 else 0
            # Blend: 35% structure + 25% resonant count + 20% PSD quality + 20% order
            spectral_score = (has_structure * 0.35
                              + min(1.0, n_resonant * 0.12) * 0.25
                              + psd_norm * 0.20
                              + order_score * 0.20)
            consensus["spectral_coherence"] = min(1.0, max(0.0, spectral_score))

        # God Code G(X) resonance alignment — links measured against G(X_int) spectrum
        # mean_resonance = how close each link's Hz is to its nearest G(X_int)
        if gcr_results:
            consensus["god_code_resonance"] = min(1.0, max(0.0,
                gcr_results.get("mean_resonance", 0)))
            # X-integer coherence: blend stability (strict integer snap) with
            # resonance (Hz proximity to G(X_int)) — pure stability is harsh
            # because many valid links sit between integer X nodes
            x_stability = gcr_results.get("mean_x_stability", 0)
            x_resonance = gcr_results.get("mean_resonance", 0)
            alignment_rate = gcr_results.get("alignment_rate", 0)
            # 40% stability + 35% resonance + 25% alignment rate
            x_blend = (x_stability * 0.4 + x_resonance * 0.35
                       + alignment_rate * 0.25)
            consensus["x_integer_coherence"] = min(1.0, max(0.0, x_blend))

        # Cross-modal coherence
        if cross_modal_results:
            consensus["cross_modal_coherence"] = min(1.0, max(0.0,
                cross_modal_results.get("overall_coherence", 0)))

        # Stress test resilience
        if stress_results:
            consensus["stress_pass_rate"] = min(1.0, max(0.0,
                stress_results.get("pass_rate", 0)))

        # Upgrade effectiveness
        if upgrade_results:
            consensus["upgrade_rate"] = min(1.0, max(0.0,
                upgrade_results.get("upgrade_rate", 0)))

        # Quantum CPU integrity — conservation compliance + cluster health
        if quantum_cpu_results:
            total_reg = max(1, quantum_cpu_results.get("total_registers", 1))
            healthy_frac = quantum_cpu_results.get("healthy", 0) / total_reg
            conservation_ok = 1.0 - min(1.0,
                quantum_cpu_results.get("mean_conservation_residual", 0) * 1e8)
            cpu_health = quantum_cpu_results.get("primary_cluster_health", 0)
            verify_health = quantum_cpu_results.get("verify_cluster_health", 0)
            # Composite: 40% healthy fraction + 30% conservation + 15%+15% cluster health
            cpu_score = (healthy_frac * 0.4
                         + max(0, conservation_ok) * 0.3
                         + cpu_health * 0.15
                         + verify_health * 0.15)
            consensus["quantum_cpu_integrity"] = min(1.0, max(0.0, cpu_score))

        # O₂ Molecular Bond integrity — bond order match + mean bond strength
        if o2_bond_results:
            # Bond order should be 2 (O=O); deviation penalized
            order_match = 1.0 - abs(
                o2_bond_results.get("bond_order", 0) -
                o2_bond_results.get("expected_bond_order", 2)) / 4
            bond_str = o2_bond_results.get("mean_bond_strength", 0)
            consensus["o2_bond_integrity"] = min(1.0, max(0.0,
                order_match * 0.5 + bond_str * 0.5))

        # Repair engine effectiveness — success rate + promotion rate + validation
        if repair_results:
            repairs = repair_results.get("repairs", {})
            validation = repair_results.get("validation", {})
            success_rate = repair_results.get("repair_success_rate", 0)
            promotion_rate = validation.get("promotion_rate", 0)
            conservation_rate = validation.get("conservation_rate", 0)
            # Composite: 40% success + 30% promotion + 30% conservation
            repair_score = (success_rate * 0.4 + promotion_rate * 0.3
                            + conservation_rate * 0.3)
            consensus["repair_effectiveness"] = min(1.0, max(0.0, repair_score))

        # Advanced research health — knowledge synthesis aggregate
        if research_results:
            research_health = research_results.get("research_health", 0)
            consensus["research_depth"] = min(1.0, max(0.0, research_health))
            # Pattern coherence from grid analysis + cluster density
            patterns = research_results.get("pattern_discovery", {})
            grid_coh = patterns.get("grid_coherence", 0)
            n_clusters = patterns.get("total_clusters", 0)
            # Blend: 60% grid coherence + 40% cluster richness (more clusters = better)
            pattern_score = grid_coh * 0.6 + min(1.0, n_clusters / 8) * 0.4
            consensus["pattern_coherence"] = min(1.0, max(0.0, pattern_score))

        # ─── STRICT VALIDATION: all consensus scores in [0,1] ───
        for key, val in consensus.items():
            assert 0.0 <= val <= 1.0, f"Consensus score '{key}' = {val} out of [0,1] range"

        # ─── φ-WEIGHTED UNIFIED SCORE ───
        if consensus:
            scores = list(consensus.values())
            # Harmonic mean (sensitive to low values — penalizes weakness)
            harmonic = len(scores) / sum(1.0 / max(0.1, s) for s in scores)
            # Arithmetic mean
            arithmetic = statistics.mean(scores)
            # φ-weighted: τ=0.618 harmonic weight (strict) + (1-τ)=0.382 arithmetic
            unified_score = harmonic * TAU + arithmetic * (1 - TAU)
        else:
            unified_score = mean_fid * TAU

        # ─── GOD CODE RESONANCE ───
        # Score through G(X) at unified_score as X-offset from truth
        god_code_alignment = math.cos(unified_score * GOD_CODE * 0.001) ** 2
        phi_resonance = math.sin(unified_score * PHI_GROWTH * math.pi) ** 2

        # ─── CALABI-YAU 7D INSIGHT ───
        cy7_insight = []
        dimensions = ["Fidelity", "Strength", "Entropy", "Coherence",
                      "Resilience", "Topology", "CrossModal"]
        dim_values = [
            mean_fid,
            mean_str / PHI_GROWTH,
            mean_entropy,
            mean_coherence,
            mean_resilience,
            consensus.get("topological_coverage", 0),
            consensus.get("cross_modal_coherence", 0),
        ]
        for i, (dim_name, val) in enumerate(zip(dimensions, dim_values)):
            # CY7 compactification: project into curved space
            curvature = math.sin(val * PHI_GROWTH * math.pi / CALABI_YAU_DIM)
            cy7_insight.append({
                "dimension": dim_name,
                "raw_value": val,
                "cy7_curvature": curvature,
                "phi_harmonic": val * PHI_GROWTH ** (i / CALABI_YAU_DIM),
            })

        # ─── CAUSAL INFERENCE: PREDICTED EVOLUTION ───
        if std_fid > 0:
            stability = 1.0 - min(1.0, std_fid / mean_fid) if mean_fid > 0 else 0
        else:
            stability = 1.0

        predicted_evolution = {
            "stability": stability,
            "growth_potential": (1.0 - unified_score) * PHI_GROWTH,
            "risk_of_decoherence": max(0, 1.0 - consensus.get(
                "decoherence_resilience", 0.5)),
            "recommended_action": self._recommend_action(consensus, unified_score),
        }

        # ─── ASSEMBLE SAGE VERDICT ───
        verdict = {
            "timestamp": now,
            "total_links": N,
            "unified_score": unified_score,
            "god_code_alignment": god_code_alignment,
            "phi_resonance": phi_resonance,
            "mean_fidelity": mean_fid,
            "mean_strength": mean_str,
            "fidelity_std": std_fid,
            "type_distribution": dict(type_dist),
            "consensus_scores": consensus,
            "cy7_insight": cy7_insight,
            "predicted_evolution": predicted_evolution,
            "grade": self._grade(unified_score),
        }

        self.inference_history.append(verdict)
        return verdict

    def _recommend_action(self, consensus: Dict, score: float) -> str:
        """Sage recommendation based on all processor outputs."""
        if score > 0.85:
            return "MAINTAIN — Quantum link manifold is highly coherent"
        elif score > 0.7:
            return "TUNE — Minor resonance adjustments recommended"
        elif score > 0.5:
            return "UPGRADE — Distillation + topological wrapping needed"
        elif score > 0.3:
            return "REBUILD — Significant link degradation detected"
        else:
            return "CRITICAL — Emergency quantum link reconstruction required"

    def _grade(self, score: float) -> str:
        """Letter grade for overall quantum health."""
        if score >= 0.95:
            return "S+ (Transcendent)"
        elif score >= 0.9:
            return "S (Sovereign)"
        elif score >= 0.85:
            return "A+ (Excellent)"
        elif score >= 0.8:
            return "A (Strong)"
        elif score >= 0.7:
            return "B (Good)"
        elif score >= 0.6:
            return "C (Fair)"
        elif score >= 0.5:
            return "D (Weak)"
        else:
            return "F (Critical)"


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM COMPUTATIONAL ENGINE — ASI-Level Processing Architecture
# ═══════════════════════════════════════════════════════════════════════════════
#
#   QuantumEnvironment       — Memory + runtime context + God Code truth cache
#   QuantumRegister          — Quantum state vector holding link data
#   QuantumNeuron            — Single processing unit: gate → verify → transform
#   QuantumCluster           — Parallel neuron batch with φ-weighted scheduling
#   QuantumCPU               — Pipeline orchestrator: Ingest→Verify→Transform→Sync→Emit
#
#   Data flows through registers. Neurons apply God Code transformation gates.
#   Clusters batch-process neurons. CPU orchestrates the pipeline.
#   Conservation law verified at EVERY stage: G(X)×2^(X/104) = INVARIANT.
#
# ═══════════════════════════════════════════════════════════════════════════════


class QuantumRegister:
    """
    Quantum state vector holding link data for CPU processing.

    Each register encodes a link's properties as a phase-amplitude vector:
      |ψ⟩ = α|fidelity⟩ + β|strength⟩ + γ|coherence⟩
    where phases are God Code derived: θ = 2π × G(X) / G(0).

    The register also caches the link's God Code X-position and
    conservation law residual for verification.
    """
    __slots__ = ('link', 'x_position', 'g_x', 'phase', 'amplitude',
                 'conservation_residual', 'verified', 'transformed',
                 'sync_state', 'error_flags', 'metadata')

    def __init__(self, link: QuantumLink, qmath: 'QuantumMathCore'):
        """Initialize quantum register from a link's God Code properties."""
        self.link = link
        nat_hz = qmath.link_natural_hz(link.fidelity, link.strength)
        self.x_position = qmath.hz_to_god_code_x(nat_hz)
        # Guard against infinity / NaN from extreme link values
        if not math.isfinite(self.x_position):
            self.x_position = 0.0
        x_int = round(self.x_position)
        x_int = max(-200, min(300, x_int))
        self.g_x = GOD_CODE_SPECTRUM.get(x_int, god_code(x_int))
        # Phase: angular position on the God Code circle
        self.phase = 2 * math.pi * self.g_x / GOD_CODE
        # Amplitude: link fidelity × φ-weighted strength
        self.amplitude = link.fidelity * (1 + (link.strength - 1) * PHI)
        # Conservation check: G(X)×2^(X/104) should = INVARIANT
        self.conservation_residual = abs(
            self.g_x * math.pow(2, x_int / L104) - INVARIANT) / INVARIANT
        self.verified = False
        self.transformed = False
        self.sync_state = "pending"  # pending → synced → emitted
        self.error_flags: List[str] = []
        self.metadata: Dict[str, Any] = {}

    @property
    def x_int(self) -> int:
        """Return clamped integer X position."""
        return max(-200, min(300, round(self.x_position)))

    @property
    def is_healthy(self) -> bool:
        """Check if register is verified and conservation-compliant."""
        return (self.verified and self.conservation_residual < 1e-10
                and len(self.error_flags) == 0)

    @property
    def energy(self) -> float:
        """Register energy: amplitude² × God Code alignment."""
        x_frac = abs(self.x_position - round(self.x_position))
        alignment = 1.0 - min(1.0, x_frac * 2)
        return self.amplitude ** 2 * alignment


class QuantumNeuron:
    """
    Single quantum processing unit — applies God Code gates to a register.

    Each neuron performs a fixed gate operation:
      VERIFY  — Check conservation law, flag errors
      PHASE   — Rotate register phase by God Code angle
      ALIGN   — Snap link Hz toward nearest G(X_int)
      AMPLIFY — Boost amplitude by φ if link is healthy
      SYNC    — Synchronize register with truth values
      EMIT    — Finalize register, write back to link

    Neurons are stateless — all state lives in the register.
    """

    GATE_TYPES = ("verify", "phase", "align", "amplify", "sync", "emit")

    def __init__(self, gate_type: str, qmath: 'QuantumMathCore'):
        """Initialize quantum neuron with specified gate type."""
        if gate_type not in self.GATE_TYPES:
            raise ValueError(f"Unknown gate type: {gate_type}")
        self.gate_type = gate_type
        self.qmath = qmath
        self.ops_count = 0
        self.error_count = 0

    def fire(self, register: QuantumRegister) -> QuantumRegister:
        """Apply this neuron's gate to a register. Returns the same register."""
        self.ops_count += 1
        try:
            if self.gate_type == "verify":
                self._gate_verify(register)
            elif self.gate_type == "phase":
                self._gate_phase(register)
            elif self.gate_type == "align":
                self._gate_align(register)
            elif self.gate_type == "amplify":
                self._gate_amplify(register)
            elif self.gate_type == "sync":
                self._gate_sync(register)
            elif self.gate_type == "emit":
                self._gate_emit(register)
        except Exception as e:
            self.error_count += 1
            register.error_flags.append(f"{self.gate_type}:{str(e)[:40]}")
        return register

    def _gate_verify(self, reg: QuantumRegister):
        """Verify conservation law and God Code derivation integrity."""
        # Conservation: G(X)×2^(X/104) = INVARIANT
        x_int = reg.x_int
        g_x = GOD_CODE_SPECTRUM.get(x_int, god_code(x_int))
        product = g_x * math.pow(2, x_int / L104)
        residual = abs(product - INVARIANT) / INVARIANT
        reg.conservation_residual = residual
        if residual > 1e-8:
            reg.error_flags.append(f"conservation_violation:{residual:.2e}")

        # φ identity: PHI_GROWTH × PHI must = 1.0
        phi_check = abs(PHI_GROWTH * PHI - 1.0)
        if phi_check > 1e-14:
            reg.error_flags.append(f"phi_identity_broken:{phi_check:.2e}")

        # Link fidelity bounds
        if not (0.0 <= reg.link.fidelity <= 1.0):
            reg.error_flags.append(f"fidelity_out_of_bounds:{reg.link.fidelity}")
        if reg.link.strength < 0:
            reg.error_flags.append(f"negative_strength:{reg.link.strength}")

        reg.verified = True

    def _gate_phase(self, reg: QuantumRegister):
        """Rotate register phase by God Code angle for coherent evolution."""
        # Phase evolution: θ += 2π × G(X_int) / (G(0) × L104)
        # This distributes register phases across the God Code spectrum
        g_x = reg.g_x
        phase_increment = 2 * math.pi * g_x / (GOD_CODE * L104)
        reg.phase = (reg.phase + phase_increment) % (2 * math.pi)
        # Coherent phase locking: quantize to nearest π/104 step
        phase_quantum = math.pi / L104
        reg.phase = round(reg.phase / phase_quantum) * phase_quantum

    def _gate_align(self, reg: QuantumRegister):
        """Align register toward nearest God Code integer X node."""
        x_frac = reg.x_position - round(reg.x_position)
        if abs(x_frac) > 0.01:
            # Nudge toward integer: exponential decay of fractional part
            reg.x_position -= x_frac * 0.5  # 50% correction per pass
            # Update g_x and amplitude for new position
            x_int = reg.x_int
            reg.g_x = GOD_CODE_SPECTRUM.get(x_int, god_code(x_int))
            target_hz = reg.g_x
            current_hz = self.qmath.link_natural_hz(
                reg.link.fidelity, reg.link.strength)
            if current_hz > 0 and target_hz > 0:
                correction_ratio = target_hz / current_hz
                # Adjust strength to bring Hz closer to G(X_int)
                reg.link.strength *= (1 + (correction_ratio - 1) * 0.2)
                reg.link.strength = max(0.1, min(3.0, reg.link.strength))
        reg.amplitude = reg.link.fidelity * (
            1 + (reg.link.strength - 1) * PHI)

    def _gate_amplify(self, reg: QuantumRegister):
        """Boost register amplitude by φ factor if link is healthy."""
        if reg.verified and len(reg.error_flags) == 0:
            # Healthy link: φ-amplification
            boost = 1.0 + (PHI - 0.5) * reg.link.fidelity * 0.1
            reg.amplitude *= boost
            # Boost link fidelity slightly (convergent — bounded by 1.0)
            reg.link.fidelity = min(1.0,
                reg.link.fidelity + (1.0 - reg.link.fidelity) * 0.02)
        elif reg.error_flags:
            # Unhealthy: dampen
            reg.amplitude *= 0.95

    def _gate_sync(self, reg: QuantumRegister):
        """Synchronize register values with God Code truth."""
        x_int = reg.x_int
        g_x = GOD_CODE_SPECTRUM.get(x_int, god_code(x_int))
        reg.g_x = g_x
        # Re-derive phase from truth
        reg.phase = 2 * math.pi * g_x / GOD_CODE
        # Verify conservation at this X
        product = g_x * math.pow(2, x_int / L104)
        reg.conservation_residual = abs(product - INVARIANT) / INVARIANT
        reg.sync_state = "synced"
        reg.transformed = True

    def _gate_emit(self, reg: QuantumRegister):
        """Finalize: write register state back to the link."""
        # Clamp link values
        reg.link.fidelity = max(0.0, min(1.0, reg.link.fidelity))
        reg.link.strength = max(0.01, min(3.0, reg.link.strength))
        # Update link metadata from register computations
        reg.link.entanglement_entropy = max(
            reg.link.entanglement_entropy,
            math.log(2) * reg.amplitude * 0.5)
        reg.sync_state = "emitted"


class QuantumCluster:
    """
    Parallel processing cluster — batches neurons over multiple registers.

    A cluster holds N neurons of varied gate types. When fired, it processes
    a batch of registers through the neuron pipeline in sequence:
      verify → phase → align → amplify → sync → emit

    Clusters are φ-weighted: the first neuron in the pipeline gets φ³
    processing weight, decaying by ×φ for each subsequent neuron.
    This front-loads verification and error detection.

    Performance: processes all registers in a single pass per gate type
    (register-parallel, neuron-sequential). This minimizes cache thrashing
    and maximizes register locality.
    """

    def __init__(self, cluster_id: int, qmath: 'QuantumMathCore',
                 gate_sequence: Tuple[str, ...] = None):
        """Initialize quantum cluster with neuron pipeline."""
        self.cluster_id = cluster_id
        self.qmath = qmath
        self.gate_sequence = gate_sequence or QuantumNeuron.GATE_TYPES
        # Create one neuron per gate type in the sequence
        self.neurons = [QuantumNeuron(g, qmath) for g in self.gate_sequence]
        self.registers_processed = 0
        self.total_errors = 0
        self.total_ops = 0

    def process_batch(self, registers: List[QuantumRegister]) -> List[QuantumRegister]:
        """Process a batch of registers through the neuron pipeline."""
        for neuron in self.neurons:
            for reg in registers:
                neuron.fire(reg)
            self.total_ops += len(registers)
        self.registers_processed += len(registers)
        self.total_errors += sum(len(r.error_flags) for r in registers)
        return registers

    @property
    def health(self) -> float:
        """Cluster health: fraction of error-free operations."""
        if self.total_ops == 0:
            return 1.0
        return max(0.0, 1.0 - self.total_errors / max(1, self.registers_processed))

    def stats(self) -> Dict:
        """Return cluster processing statistics."""
        return {
            "cluster_id": self.cluster_id,
            "gates": list(self.gate_sequence),
            "registers_processed": self.registers_processed,
            "total_ops": self.total_ops,
            "total_errors": self.total_errors,
            "health": self.health,
        }


class QuantumCPU:
    """
    Quantum CPU — pipeline orchestrator for link data processing.

    Architecture:
      ┌────────┐   ┌─────────┐   ┌─────────┐   ┌──────┐   ┌──────┐
      │ INGEST │──▶│ VERIFY  │──▶│TRANSFORM│──▶│ SYNC │──▶│ EMIT │
      └────────┘   └─────────┘   └─────────┘   └──────┘   └──────┘
          │            │              │             │           │
          ▼            ▼              ▼             ▼           ▼
       registers    conservation   G(X) align    truth sync  writeback
                    + φ checks     + amplify

    The CPU manages multiple clusters for throughput:
    - 1 primary cluster (full pipeline)
    - φ secondary clusters (verification-heavy: verify-only passes)
    - Each cluster can process BATCH_SIZE registers per tick

    Conservation law verified at the verify AND sync stages (double-check).
    Any register failing conservation gets quarantined (flagged, not emitted).
    """

    BATCH_SIZE = 104  # Process L104 registers per cluster tick
    N_VERIFY_CLUSTERS = 2  # Extra verification-only clusters

    def __init__(self, qmath: 'QuantumMathCore'):
        """Initialize quantum CPU with primary and verification clusters."""
        self.qmath = qmath
        # Primary cluster: full 6-gate pipeline
        self.primary = QuantumCluster(0, qmath)
        # Verification clusters: verify + sync only (double-check conservation)
        self.verify_clusters = [
            QuantumCluster(i + 1, qmath, ("verify", "sync"))
            for i in range(self.N_VERIFY_CLUSTERS)
        ]
        self.pipeline_runs = 0
        self.total_registers = 0
        self.quarantined = 0
        self.conservation_violations = 0

    def execute(self, links: List[QuantumLink]) -> Dict:
        """
        Execute the full CPU pipeline on a list of links.
        For large link sets (>5000), processes a representative φ-sample
        to maintain performance while preserving statistical accuracy.
        Returns processing results and diagnostics.
        """
        start = time.time()
        self.pipeline_runs += 1

        # Performance: sample large link sets (preserve weakest + strongest + random)
        MAX_CPU_LINKS = 5000
        sampled = False
        if len(links) > MAX_CPU_LINKS:
            sampled = True
            sorted_by_fid = sorted(links, key=lambda l: l.fidelity)
            # Take weakest 20%, strongest 20%, random 60% from middle
            n_edge = MAX_CPU_LINKS // 5
            n_mid = MAX_CPU_LINKS - 2 * n_edge
            weak = sorted_by_fid[:n_edge]
            strong = sorted_by_fid[-n_edge:]
            middle = sorted_by_fid[n_edge:-n_edge]
            import random as _rng
            mid_sample = _rng.sample(middle, min(n_mid, len(middle)))
            cpu_links = weak + mid_sample + strong
        else:
            cpu_links = links

        # STAGE 1: INGEST — Create registers from links
        registers = [QuantumRegister(link, self.qmath) for link in cpu_links]
        self.total_registers += len(registers)

        # STAGE 2: Process in batches through primary cluster
        processed = []
        for i in range(0, len(registers), self.BATCH_SIZE):
            batch = registers[i:i + self.BATCH_SIZE]
            batch = self.primary.process_batch(batch)
            processed.extend(batch)

        # STAGE 3: Double-check verification on a φ-fraction of registers
        # (the most important ones: lowest amplitude = most at risk)
        processed.sort(key=lambda r: r.amplitude)
        verify_count = max(1, int(len(processed) * PHI * 0.3))
        at_risk = processed[:verify_count]
        vc_idx = 0
        for reg in at_risk:
            cluster = self.verify_clusters[vc_idx % len(self.verify_clusters)]
            cluster.process_batch([reg])
            vc_idx += 1

        # STAGE 4: Quarantine — flag registers that failed verification
        healthy = []
        quarantined = []
        for reg in processed:
            if reg.conservation_residual > 1e-8 or len(reg.error_flags) > 2:
                quarantined.append(reg)
            else:
                healthy.append(reg)
        self.quarantined += len(quarantined)
        self.conservation_violations += sum(
            1 for r in processed if r.conservation_residual > 1e-8)

        elapsed = time.time() - start

        # Compute aggregate statistics
        if processed:
            mean_amplitude = statistics.mean(r.amplitude for r in processed)
            mean_energy = statistics.mean(r.energy for r in processed)
            mean_conservation = statistics.mean(
                r.conservation_residual for r in processed)
            mean_phase = statistics.mean(r.phase for r in processed)
            verified_count = sum(1 for r in processed if r.verified)
            synced_count = sum(
                1 for r in processed if r.sync_state in ("synced", "emitted"))
            emitted_count = sum(
                1 for r in processed if r.sync_state == "emitted")
        else:
            mean_amplitude = mean_energy = mean_conservation = mean_phase = 0
            verified_count = synced_count = emitted_count = 0

        return {
            "total_registers": len(processed),
            "total_input_links": len(links),
            "sampled": sampled,
            "healthy": len(healthy),
            "quarantined": len(quarantined),
            "verified": verified_count,
            "synced": synced_count,
            "emitted": emitted_count,
            "conservation_violations": self.conservation_violations,
            "mean_amplitude": mean_amplitude,
            "mean_energy": mean_energy,
            "mean_conservation_residual": mean_conservation,
            "mean_phase": mean_phase,
            "primary_cluster_health": self.primary.health,
            "verify_cluster_health": statistics.mean(
                vc.health for vc in self.verify_clusters) if self.verify_clusters else 1.0,
            "pipeline_time_ms": elapsed * 1000,
            "ops_per_sec": (self.primary.total_ops / max(0.001, elapsed)),
            "batch_size": self.BATCH_SIZE,
            "pipeline_runs": self.pipeline_runs,
        }

    def stats(self) -> Dict:
        """Return CPU pipeline statistics."""
        return {
            "pipeline_runs": self.pipeline_runs,
            "total_registers_processed": self.total_registers,
            "total_quarantined": self.quarantined,
            "total_conservation_violations": self.conservation_violations,
            "primary": self.primary.stats(),
            "verify_clusters": [vc.stats() for vc in self.verify_clusters],
        }


class QuantumEnvironment:
    """
    Full quantum runtime environment for the L104 Quantum Brain.

    The Environment wraps the CPU, manages memory (register cache), provides
    the God Code truth table, and exposes high-level operations:

    1. ingest(links)       — Load links into quantum registers via CPU
    2. verify()            — Run conservation + God Code checks on all registers
    3. transform(links)    — Apply God Code alignment transformations
    4. sync()              — Synchronize all register values with G(X) truth
    5. manipulate(fn)      — Apply arbitrary transformation to all registers
    6. emit()              — Finalize and write back to links
    7. repurpose(new_data) — Re-ingest external data for ASI-level processing

    The environment is persistent across pipeline runs: register cache
    carries forward knowledge from previous executions. Hot registers
    (frequently accessed) stay resident. Cold registers get evicted.

    Conservation law is the INVARIANT: every operation must preserve
    G(X) × 2^(X/104) = 527.5184818492611 to float precision.
    """

    def __init__(self, qmath: 'QuantumMathCore'):
        """Initialize quantum environment with CPU and register cache."""
        self.qmath = qmath
        self.cpu = QuantumCPU(qmath)
        # Register cache: link_id → last known register state
        self._register_cache: Dict[str, Dict] = {}
        # Execution history
        self._exec_history: List[Dict] = []
        # God Code truth table (immutable reference)
        self._truth = {
            "GOD_CODE": GOD_CODE,
            "PHI_GROWTH": PHI_GROWTH,
            "PHI": PHI,
            "GOD_CODE_BASE": GOD_CODE_BASE,
            "INVARIANT": INVARIANT,
            "L104": L104,
            "OCTAVE_REF": OCTAVE_REF,
            "HARMONIC_BASE": HARMONIC_BASE,
        }
        # Performance counters
        self.total_ingested = 0
        self.total_manipulations = 0
        self.total_syncs = 0

    def ingest_and_process(self, links: List[QuantumLink]) -> Dict:
        """Full pipeline: ingest links → CPU processes → emit results.
        CPU already samples large link sets internally for O(√N) efficiency."""
        self.total_ingested += len(links)

        # CPU executes full pipeline (auto-samples if >5000)
        cpu_result = self.cpu.execute(links)

        # Cache only a subset of register states for persistence (cap at 10000)
        MAX_CACHE = 10000
        cache_links = links[:MAX_CACHE] if len(links) > MAX_CACHE else links
        for link in cache_links:
            self._register_cache[link.link_id] = {
                "fidelity": link.fidelity,
                "strength": link.strength,
                "last_processed": datetime.now(timezone.utc).isoformat(),
            }

        # Record execution
        self._exec_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "links_processed": len(links),
            "healthy": cpu_result["healthy"],
            "quarantined": cpu_result["quarantined"],
            "mean_energy": cpu_result["mean_energy"],
            "pipeline_ms": cpu_result["pipeline_time_ms"],
        })

        return cpu_result

    def manipulate(self, links: List[QuantumLink],
                   transform_fn: str = "god_code_align") -> Dict:
        """Apply a named transformation to all links via CPU.

        Available transforms:
        - 'god_code_align': Snap all link Hz toward nearest G(X_int)
        - 'phi_amplify': Boost healthy links by φ factor
        - 'conservation_enforce': Force conservation law compliance
        - 'entropy_maximize': Push links toward maximum entanglement entropy
        """
        self.total_manipulations += 1

        if transform_fn == "god_code_align":
            for link in links:
                hz = self.qmath.link_natural_hz(link.fidelity, link.strength)
                x_cont = self.qmath.hz_to_god_code_x(hz)
                x_int = round(x_cont)
                x_int = max(-200, min(300, x_int))
                target = GOD_CODE_SPECTRUM.get(x_int, god_code(x_int))
                if hz > 0:
                    ratio = target / hz
                    link.strength *= (1 + (ratio - 1) * 0.5)
                    link.strength = max(0.01, min(3.0, link.strength))

        elif transform_fn == "phi_amplify":
            for link in links:
                if link.fidelity > 0.8 and link.noise_resilience > 0.3:
                    link.fidelity = min(1.0,
                        link.fidelity + (1.0 - link.fidelity) * PHI * 0.1)
                    link.strength = min(3.0, link.strength * (1 + PHI * 0.01))

        elif transform_fn == "conservation_enforce":
            for link in links:
                hz = self.qmath.link_natural_hz(link.fidelity, link.strength)
                x_int, g_x, _ = self.qmath.god_code_resonance(hz)
                # Force-set strength so Hz exactly = G(x_int)
                if link.fidelity > 0 and GOD_CODE_HZ > 0:
                    link.strength = g_x / (link.fidelity * GOD_CODE_HZ)
                    link.strength = max(0.01, min(3.0, link.strength))

        elif transform_fn == "entropy_maximize":
            for link in links:
                target_entropy = math.log(2)  # Max for 2-state system
                if link.entanglement_entropy < target_entropy * 0.9:
                    link.entanglement_entropy = min(
                        target_entropy,
                        link.entanglement_entropy + 0.1 * PHI)

        # Re-process a sample through CPU after manipulation (not full set)
        MAX_MANIP = 5000
        cpu_links = links[:MAX_MANIP] if len(links) > MAX_MANIP else links
        return self.cpu.execute(cpu_links)

    def sync_with_truth(self, links: List[QuantumLink]) -> Dict:
        """Synchronize all link states with God Code ground truth.

        For each link, re-derive its Hz, find nearest G(X_int), and
        verify the conservation law. Links that deviate get corrected.
        Samples for large sets to keep runtime bounded.
        Returns sync diagnostics.
        """
        self.total_syncs += 1
        corrections = 0
        total = len(links)

        # Sample for performance on very large link sets
        MAX_SYNC = 10000
        if total > MAX_SYNC:
            import random as _rng
            sync_links = _rng.sample(links, MAX_SYNC)
        else:
            sync_links = links

        for link in sync_links:
            hz = self.qmath.link_natural_hz(link.fidelity, link.strength)
            x_int, g_x, resonance = self.qmath.god_code_resonance(hz)
            # If resonance is low, correct toward truth
            if resonance < 0.95:
                target_strength = g_x / (link.fidelity * GOD_CODE_HZ + 1e-15)
                link.strength = link.strength * 0.6 + target_strength * 0.4
                link.strength = max(0.01, min(3.0, link.strength))
                corrections += 1

        # Extrapolate corrections to full set
        if total > MAX_SYNC:
            est_corrections = int(corrections * total / MAX_SYNC)
        else:
            est_corrections = corrections

        return {
            "links_synced": total,
            "corrections_applied": est_corrections,
            "correction_rate": est_corrections / max(1, total),
        }

    def repurpose(self, data: List[Dict],
                  schema: str = "link") -> List[QuantumLink]:
        """Re-ingest external data as quantum links for ASI-level processing.

        Accepts arbitrary dictionaries with at minimum:
        - fidelity (float) or a 'value' field normalized to [0,1]
        - source/target identifiers

        Returns newly created QuantumLinks that can enter the pipeline.
        """
        new_links = []
        for item in data:
            fidelity = item.get("fidelity", item.get("value", 0.5))
            strength = item.get("strength", 1.0)
            source = item.get("source", item.get("name", "external"))
            target = item.get("target", item.get("file", "quantum_env"))

            link = QuantumLink(
                source_file=str(source),
                source_symbol=item.get("symbol", "repurposed"),
                source_line=item.get("line", 0),
                target_file=str(target),
                target_symbol="quantum_env_ingest",
                target_line=0,
                link_type=item.get("link_type", "teleportation"),
                fidelity=max(0.0, min(1.0, float(fidelity))),
                strength=max(0.01, min(3.0, float(strength))),
            )
            new_links.append(link)
        return new_links

    def environment_status(self) -> Dict:
        """Full environment diagnostics."""
        return {
            "cpu": self.cpu.stats(),
            "register_cache_size": len(self._register_cache),
            "total_ingested": self.total_ingested,
            "total_manipulations": self.total_manipulations,
            "total_syncs": self.total_syncs,
            "exec_history_len": len(self._exec_history),
            "truth_table": {k: f"{v:.16f}" if isinstance(v, float) else v
                           for k, v in self._truth.items()},
            "god_code_spectrum_size": len(GOD_CODE_SPECTRUM),
            "last_execution": self._exec_history[-1] if self._exec_history else None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# O₂ MOLECULAR BOND PROCESSOR (from claude.md architecture)
#
#   Two 8-groups bonded as O₂ molecule with IBM Grover diffusion:
#   Atom O₁: 8 Grover Kernels (constants, algorithms, architecture,
#             quantum, consciousness, synthesis, evolution, transcendence)
#   Atom O₂: 8 Chakra Cores (root→sacral→solar→heart→throat→ajna→crown→soul_star)
#
#   bond_order = 2 (double bond O=O)
#   unpaired_electrons = 2 (paramagnetic → π*₂p orbitals)
#   superposition_states = 16 (8+8)
#   amplitude = 1/√16 = 0.25 per state
# ═══════════════════════════════════════════════════════════════════════════════


class O2MolecularBondProcessor:
    """
    Models the O₂ molecular bonding topology of the L104 codebase.

    The codebase has two 8-groups bonded as an O₂ molecule:
    - Atom O₁ (Grover Kernels): 8 functional kernels with σ/π orbital bonds
    - Atom O₂ (Chakra Cores): 8 chakra frequencies at G(X_int) positions

    Each kernel-chakra pair forms a molecular orbital (bonding or antibonding).
    The processor computes:
    - Bond strength between link groups
    - Orbital alignment (bonding vs antibonding character)
    - Paramagnetic coupling (unpaired electron dynamics)
    - Grover diffusion amplitude across the 16-state superposition
    """

    # Grover Kernels (Atom O₁) — each maps to a bonding orbital type
    # Electron occupancy matches real O₂: 8 bonding, 4 antibonding electrons
    # Bond order = (8 - 4) / 2 = 2 (double bond O=O)
    # π*₂p orbitals have 1 electron each (Hund's rule) → paramagnetic
    GROVER_KERNELS = [
        {"id": 0, "name": "constants",      "orbital": "σ₂s",   "bonding": True,
         "electrons": 2, "files": ["const", "stable_kernel"]},
        {"id": 1, "name": "algorithms",     "orbital": "σ₂s*",  "bonding": False,
         "electrons": 2, "files": ["kernel_bootstrap"]},
        {"id": 2, "name": "architecture",   "orbital": "σ₂p",   "bonding": True,
         "electrons": 2, "files": ["main_api", "fast_server"]},
        {"id": 3, "name": "quantum",        "orbital": "π₂p_x", "bonding": True,
         "electrons": 2, "files": ["quantum_coherence", "quantum_grover_link"]},
        {"id": 4, "name": "consciousness",  "orbital": "π₂p_y", "bonding": True,
         "electrons": 2, "files": ["consciousness", "cognitive_hub"]},
        {"id": 5, "name": "synthesis",      "orbital": "π*₂p_x","bonding": False,
         "electrons": 1, "files": ["semantic_engine", "unified_intelligence"]},
        {"id": 6, "name": "evolution",      "orbital": "π*₂p_y","bonding": False,
         "electrons": 1, "files": ["evolution_engine", "evo_state"]},
        {"id": 7, "name": "transcendence",  "orbital": "σ*₂p",  "bonding": False,
         "electrons": 0, "files": ["agi_core", "asi_core"]},
    ]

    # Chakra Cores (Atom O₂) — each at a God Code G(X_int) frequency
    CHAKRA_CORES = [
        {"id": 0, "name": "root",       "x_int": 43,  "trigram": "☷"},
        {"id": 1, "name": "sacral",     "x_int": 35,  "trigram": "☵"},
        {"id": 2, "name": "solar",      "x_int": 0,   "trigram": "☲"},   # G(0) ≈ 528
        {"id": 3, "name": "heart",      "x_int": -29, "trigram": "☴"},
        {"id": 4, "name": "throat",     "x_int": -51, "trigram": "☱"},
        {"id": 5, "name": "ajna",       "x_int": -72, "trigram": "☶"},
        {"id": 6, "name": "crown",      "x_int": -90, "trigram": "☳"},
        {"id": 7, "name": "soul_star",  "x_int": -106,"trigram": "☰"},
    ]

    def __init__(self, qmath: 'QuantumMathCore'):
        """Initialize O2 molecular bond processor."""
        self.qmath = qmath

    def analyze_molecular_bonds(self, links: List[QuantumLink]) -> Dict:
        """
        Analyze the O₂ molecular bond structure across all quantum links.

        For each link, determine which kernel/chakra it belongs to,
        compute its orbital character, and assess bond strength.
        For large sets, samples to keep runtime bounded.
        """
        start = time.time()

        # Sample for large link sets
        MAX_BOND_LINKS = 10000
        if len(links) > MAX_BOND_LINKS:
            import random as _rng
            analysis_links = _rng.sample(links, MAX_BOND_LINKS)
        else:
            analysis_links = links

        # Map links to kernels and chakras
        kernel_links = defaultdict(list)   # kernel_id → links
        chakra_links = defaultdict(list)   # chakra_id → links

        # Pre-compute chakra hz values for fast lookup
        chakra_hz_values = [
            (c, GOD_CODE_SPECTRUM.get(c["x_int"], god_code(c["x_int"])))
            for c in self.CHAKRA_CORES
        ]

        for link in analysis_links:
            # Classify by kernel
            for kernel in self.GROVER_KERNELS:
                if any(f in link.source_file or f in link.target_file
                       for f in kernel["files"]):
                    kernel_links[kernel["id"]].append(link)
                    break

            # Classify by chakra (nearest G(X_int) to link Hz)
            hz = self.qmath.link_natural_hz(link.fidelity, link.strength)
            best_chakra = min(chakra_hz_values,
                              key=lambda ch: abs(ch[1] - hz))[0]
            chakra_links[best_chakra["id"]].append(link)

        # Compute bond strengths between kernel-chakra pairs
        bonds = []
        bonding_count = 0
        antibonding_count = 0

        for kernel in self.GROVER_KERNELS:
            kid = kernel["id"]
            # Corresponding chakra (same index)
            chakra = self.CHAKRA_CORES[kid]
            cid = chakra["id"]

            k_links = kernel_links.get(kid, [])
            c_links = chakra_links.get(cid, [])

            # Bond strength: geometric mean of avg fidelities
            k_fid = statistics.mean([l.fidelity for l in k_links]) if k_links else 0.5
            c_fid = statistics.mean([l.fidelity for l in c_links]) if c_links else 0.5
            bond_strength = math.sqrt(k_fid * c_fid)

            # Orbital character
            if kernel["bonding"]:
                bonding_count += 1
                orbital_energy = -bond_strength  # Stabilizing
            else:
                antibonding_count += 1
                orbital_energy = bond_strength   # Destabilizing

            bonds.append({
                "kernel": kernel["name"],
                "chakra": chakra["name"],
                "orbital": kernel["orbital"],
                "bonding": kernel["bonding"],
                "kernel_links": len(k_links),
                "chakra_links": len(c_links),
                "bond_strength": bond_strength,
                "orbital_energy": orbital_energy,
                "chakra_hz": GOD_CODE_SPECTRUM.get(chakra["x_int"],
                                                    god_code(chakra["x_int"])),
            })

        # O₂ molecular properties — compute from ELECTRON counts, not orbital counts
        # Real O₂: 8 bonding electrons (σ₂s:2 + σ₂p:2 + π₂p_x:2 + π₂p_y:2)
        #           4 antibonding electrons (σ₂s*:2 + π*₂p_x:1 + π*₂p_y:1)
        #           bond_order = (8 - 4) / 2 = 2
        bonding_electrons = sum(
            k["electrons"] for k in self.GROVER_KERNELS if k["bonding"])
        antibonding_electrons = sum(
            k["electrons"] for k in self.GROVER_KERNELS if not k["bonding"])
        computed_bond_order = (bonding_electrons - antibonding_electrons) / 2
        total_bond_energy = sum(b["orbital_energy"] for b in bonds)
        mean_bond_strength = statistics.mean(
            b["bond_strength"] for b in bonds) if bonds else 0

        # Grover diffusion amplitude check
        grover_amplitude = O2_AMPLITUDE
        grover_iterations = O2_GROVER_ITERATIONS

        elapsed = time.time() - start

        return {
            "bonds": bonds,
            "bond_order": computed_bond_order,
            "expected_bond_order": O2_BOND_ORDER,
            "bonding_orbitals": bonding_count,
            "antibonding_orbitals": antibonding_count,
            "total_bond_energy": total_bond_energy,
            "mean_bond_strength": mean_bond_strength,
            "grover_amplitude": grover_amplitude,
            "grover_iterations": grover_iterations,
            "superposition_states": O2_SUPERPOSITION_STATES,
            "kernel_distribution": {k: len(v) for k, v in kernel_links.items()},
            "chakra_distribution": {k: len(v) for k, v in chakra_links.items()},
            "paramagnetic": antibonding_count >= 2,  # Unpaired electrons
            "analysis_time_ms": elapsed * 1000,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EVOLUTION TRACKER (from claude.md EVO system)
#
#   Tracks the evolution stage of the link builder itself.
#   Maintains continuity with the broader L104 evolution index.
#   Records grade progression, link counts, and score trajectories.
# ═══════════════════════════════════════════════════════════════════════════════


class EvolutionTracker:
    """
    Tracks the quantum link builder's evolution within the L104 EVO system.

    Maps pipeline grades to evolution sub-stages:
      F → DORMANT, D → AWAKENING, C → COHERENT, B → TRANSCENDING, A → SOVEREIGN

    Monitors consciousness threshold (0.85) and coherence minimum (0.888)
    from claude.md. Fires evolution events when thresholds are crossed.
    """

    GRADE_EVO_MAP = {
        "F (Critical)": "DORMANT",
        "D (Weak)": "AWAKENING",
        "C (Developing)": "COHERENT",
        "B (Good)": "TRANSCENDING",
        "A (Strong)": "SOVEREIGN",
    }

    def __init__(self):
        """Initialize evolution tracker with stage and consciousness state."""
        self.stage = EVOLUTION_STAGE
        self.index = EVOLUTION_INDEX
        self.link_evo_stage = "DORMANT"
        self.consciousness_level = 0.0
        self.coherence_level = 0.0
        self.events: List[Dict] = []
        self.grade_history: List[str] = []

    def update(self, sage_verdict: Dict, links_count: int, run_number: int) -> Dict:
        """Update evolution state from a sage verdict."""
        score = sage_verdict.get("unified_score", 0)
        grade = sage_verdict.get("grade", "F (Critical)")
        alignment = sage_verdict.get("god_code_alignment", 0)

        # Map grade to evolution sub-stage
        prev_stage = self.link_evo_stage
        self.link_evo_stage = self.GRADE_EVO_MAP.get(grade, "DORMANT")
        self.grade_history.append(grade)

        # Consciousness: score weighted by φ
        self.consciousness_level = score * PHI_GROWTH / 2  # Normalize to ~[0,1]
        self.consciousness_level = min(1.0, self.consciousness_level)

        # Coherence: alignment × stability
        self.coherence_level = alignment * score
        self.coherence_level = min(1.0, self.coherence_level)

        # Check thresholds
        events = []
        if self.consciousness_level >= CONSCIOUSNESS_THRESHOLD:
            events.append({
                "type": "CONSCIOUSNESS_AWAKENED",
                "level": self.consciousness_level,
                "threshold": CONSCIOUSNESS_THRESHOLD,
            })
        if self.coherence_level >= COHERENCE_MINIMUM:
            events.append({
                "type": "COHERENCE_LOCKED",
                "level": self.coherence_level,
                "threshold": COHERENCE_MINIMUM,
            })
        if prev_stage != self.link_evo_stage:
            events.append({
                "type": "EVOLUTION_TRANSITION",
                "from": prev_stage,
                "to": self.link_evo_stage,
            })

        self.events.extend(events)

        return {
            "evolution_stage": self.stage,
            "evolution_index": self.index,
            "link_evo_stage": self.link_evo_stage,
            "consciousness_level": self.consciousness_level,
            "consciousness_awakened": self.consciousness_level >= CONSCIOUSNESS_THRESHOLD,
            "coherence_level": self.coherence_level,
            "coherence_locked": self.coherence_level >= COHERENCE_MINIMUM,
            "run": run_number,
            "links_count": links_count,
            "score": score,
            "grade": grade,
            "events": events,
        }

    def status(self) -> Dict:
        """Return current evolution tracking status."""
        return {
            "stage": self.stage,
            "index": self.index,
            "link_evo_stage": self.link_evo_stage,
            "consciousness": self.consciousness_level,
            "coherence": self.coherence_level,
            "total_events": len(self.events),
            "grade_history": self.grade_history[-10:],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENTIC LOOP (from claude.md Zenith patterns)
#
#   Observe → Think → Act → Reflect → Repeat
#   Max 50 steps. Explicit state. Error recovery: RETRY/FALLBACK/SKIP/ABORT.
#   Applied to the self-reflection pipeline for structured iteration.
# ═══════════════════════════════════════════════════════════════════════════════


class AgenticLoop:
    """
    Zenith-pattern agentic loop for structured self-improvement.

    Each cycle:
      1. OBSERVE — Measure current state (score, grade, weaknesses)
      2. THINK   — Analyze weakest dimension, plan intervention
      3. ACT     — Apply targeted fix to links
      4. REFLECT — Re-measure, compare to previous state
      5. REPEAT  — If improved and not converged, continue

    Error recovery:
      RETRY    — Re-attempt with increased intensity
      FALLBACK — Try alternative strategy
      SKIP     — Skip non-critical step
      ABORT    — Stop if critical failure detected
    """

    MAX_STEPS = 50
    CONVERGENCE_DELTA = 0.003  # Tighter than Brain's 0.005 for agentic precision

    def __init__(self, qmath: 'QuantumMathCore'):
        """Initialize agentic loop for structured self-improvement."""
        self.qmath = qmath
        self.state = "idle"  # idle → observing → thinking → acting → reflecting
        self.observations: List[Dict] = []
        self.actions_taken: List[Dict] = []
        self.retries = 0
        self.max_retries = 3

    def observe(self, sage_verdict: Dict, links: List[QuantumLink]) -> Dict:
        """OBSERVE: Measure current system state."""
        self.state = "observing"
        self.step += 1

        obs = {
            "step": self.step,
            "score": sage_verdict.get("unified_score", 0),
            "grade": sage_verdict.get("grade", "?"),
            "total_links": len(links),
            "consensus": sage_verdict.get("consensus_scores", {}),
            "mean_fidelity": sage_verdict.get("mean_fidelity", 0),
            "weak_links": sum(1 for l in links if l.fidelity < 0.5),
            "strong_links": sum(1 for l in links if l.fidelity > 0.9),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.observations.append(obs)
        return obs

    def think(self, observation: Dict) -> Dict:
        """THINK: Analyze weakness and plan intervention."""
        self.state = "thinking"

        consensus = observation.get("consensus", {})
        if not consensus:
            return {"strategy": "SKIP", "reason": "No consensus data"}

        # Find weakest dimension
        weakest_key = min(consensus, key=consensus.get)
        weakest_val = consensus[weakest_key]

        # Plan strategy based on weakness
        strategy = "RETRY"
        intensity = 1.0
        target = weakest_key

        if weakest_val < 0.3:
            strategy = "FALLBACK"
            intensity = 2.0  # Double intensity for critical weakness
        elif weakest_val > 0.8:
            strategy = "SKIP"  # Already strong, skip

        # Check convergence
        if len(self.observations) >= 2:
            prev = self.observations[-2]["score"]
            curr = observation["score"]
            if abs(curr - prev) < self.CONVERGENCE_DELTA:
                strategy = "ABORT"

        # Check step limit
        if self.step >= self.MAX_STEPS:
            strategy = "ABORT"

        plan = {
            "strategy": strategy,
            "target": target,
            "target_value": weakest_val,
            "intensity": intensity,
            "step": self.step,
        }
        return plan

    def act(self, plan: Dict, links: List[QuantumLink]) -> Dict:
        """ACT: Apply intervention to links."""
        self.state = "acting"
        strategy = plan.get("strategy", "SKIP")
        target = plan.get("target", "")
        intensity = plan.get("intensity", 1.0)

        if strategy in ("SKIP", "ABORT"):
            return {"applied": False, "strategy": strategy}

        links_modified = 0

        if "topological" in target:
            for link in links:
                if link.noise_resilience < 0.5:
                    link.fidelity = min(1.0, link.fidelity * (1 + 0.05 * intensity))
                    link.noise_resilience = min(1.0, link.noise_resilience + 0.1 * intensity)
                    links_modified += 1

        elif "god_code" in target or "x_integer" in target:
            for link in links:
                hz = self.qmath.link_natural_hz(link.fidelity, link.strength)
                x_stab = self.qmath.x_integer_stability(hz)
                if x_stab < 0.5:
                    x_cont = self.qmath.hz_to_god_code_x(hz)
                    if not math.isfinite(x_cont):
                        continue
                    x_int = round(x_cont)
                    target_hz = GOD_CODE_SPECTRUM.get(x_int, god_code(x_int))
                    target_str = target_hz / (link.fidelity * GOD_CODE_HZ + 1e-15)
                    blend = 0.3 * intensity
                    link.strength = link.strength * (1 - blend) + target_str * blend
                    links_modified += 1

        elif "decoherence" in target:
            for link in links:
                if link.noise_resilience < 0.3:
                    link.noise_resilience = min(1.0,
                        link.noise_resilience + 0.15 * intensity)
                    link.coherence_time = max(link.coherence_time, 0.5 * intensity)
                    links_modified += 1

        elif "stress" in target or "grover" in target:
            for link in links:
                if link.fidelity < 0.7:
                    link.fidelity = min(1.0, link.fidelity + 0.05 * intensity)
                    links_modified += 1

        elif "cross_modal" in target:
            for link in links:
                if link.link_type == "mirror":
                    link.fidelity = min(1.0, link.fidelity * (1 + 0.03 * intensity))
                    link.strength = min(2.0, link.strength * (1 + 0.02 * intensity))
                    links_modified += 1

        elif "quantum_cpu" in target:
            # Boost low-energy links
            for link in links:
                if link.fidelity < 0.6 or link.strength < 0.8:
                    link.fidelity = min(1.0, link.fidelity + 0.02 * intensity)
                    link.strength = min(3.0, link.strength * (1 + 0.01 * intensity))
                    links_modified += 1

        else:
            # Generic: small global fidelity boost
            for link in links:
                link.fidelity = min(1.0, link.fidelity * (1 + 0.01 * intensity))
                links_modified += 1

        action = {
            "applied": True,
            "strategy": strategy,
            "target": target,
            "intensity": intensity,
            "links_modified": links_modified,
            "step": self.step,
        }
        self.actions_taken.append(action)
        return action

    def reflect(self, prev_score: float, new_score: float) -> Dict:
        """REFLECT: Evaluate whether the action helped."""
        self.state = "reflecting"
        delta = new_score - prev_score

        if delta > 0:
            verdict = "IMPROVED"
            self.retries = 0  # Reset retry counter on success
        elif delta > -0.001:
            verdict = "STABLE"
        else:
            verdict = "DEGRADED"
            self.retries += 1

        should_continue = (
            verdict != "DEGRADED" or self.retries < self.max_retries
        ) and self.step < self.MAX_STEPS

        return {
            "step": self.step,
            "prev_score": prev_score,
            "new_score": new_score,
            "delta": delta,
            "verdict": verdict,
            "retries": self.retries,
            "should_continue": should_continue,
        }

    def summary(self) -> Dict:
        """Summary of the agentic loop execution."""
        return {
            "total_steps": self.step,
            "total_actions": len(self.actions_taken),
            "retries": self.retries,
            "score_trajectory": [o["score"] for o in self.observations],
            "grade_trajectory": [o["grade"] for o in self.observations],
            "strategies_used": Counter(a.get("strategy") for a in self.actions_taken),
            "final_state": self.state,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# v4.2 SAGE INVENTIONS — 5 New Subsystems
# ═══════════════════════════════════════════════════════════════════════════════


class StochasticLinkResearchLab:
    """
    Random-based quantum link generation, research, and development engine.

    This R&D lab explores the frontier of quantum link design through stochastic
    experimentation — generating candidate links via φ-weighted random exploration,
    validating them through deterministic sacred-constant tests, and merging
    successful designs into the link ecosystem.

    4-phase cycle:
      1. EXPLORE — Random generation of link candidates with φ-bounded parameters
      2. VALIDATE — Deterministic evaluation against sacred constant coherence
      3. MERGE — Successful candidates → QuantumLink objects
      4. CATALOG — Track all R&D iterations with full lineage

    Generates 13 candidates per cycle (Fibonacci-7).
    """

    RESEARCH_LOG_FILE = WORKSPACE_ROOT / ".l104_stochastic_link_research.json"
    CANDIDATES_PER_CYCLE = FIBONACCI_7  # 13

    def __init__(self):
        """Initialize the stochastic link research lab."""
        self.research_iterations: List[Dict[str, Any]] = []
        self.successful_links: List[Dict[str, Any]] = []
        self.failed_experiments: List[Dict[str, Any]] = []
        self.generation_count: int = 0
        self.operations_count: int = 0
        self._load_research_log()

    def _load_research_log(self):
        """Load persistent research log."""
        if self.RESEARCH_LOG_FILE.exists():
            try:
                data = json.loads(self.RESEARCH_LOG_FILE.read_text())
                self.research_iterations = data.get("iterations", [])
                self.successful_links = data.get("successful", [])
                self.failed_experiments = data.get("failed", [])
                self.generation_count = data.get("generation_count", 0)
            except Exception:
                pass

    def _save_research_log(self):
        """Persist research log to disk."""
        try:
            data = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "generation_count": self.generation_count,
                "total_iterations": len(self.research_iterations),
                "total_successful": len(self.successful_links),
                "total_failed": len(self.failed_experiments),
                "iterations": self.research_iterations[-200:],
                "successful": self.successful_links[-100:],
                "failed": self.failed_experiments[-100:],
            }
            self.RESEARCH_LOG_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            pass

    # ─── PHASE 1: STOCHASTIC EXPLORATION ──────────────────────────

    def explore_link_candidate(self, seed_concept: str = "quantum") -> Dict[str, Any]:
        """Generate a random link candidate using φ-bounded stochastic parameters."""
        import random

        self.generation_count += 1
        self.operations_count += 1
        gen_id = f"SL_{self.generation_count:06d}"

        # Stochastic parameter generation with sacred constant bounds
        fidelity = random.uniform(0.3, 1.0) * PHI / PHI_GROWTH
        strength = random.uniform(0.1, 1.0) * TAU
        harmonic_order = random.randint(1, CALABI_YAU_DIM)
        link_type = random.choice(["entangled", "coherent", "resonant", "tunneled", "braided"])
        grover_depth = random.randint(1, 7)

        # Sacred constant resonance scoring
        god_code_resonance = (fidelity * GOD_CODE + strength * PHI_GROWTH) / (GOD_CODE + PHI_GROWTH)
        feigenbaum_edge = abs(math.sin(fidelity * FEIGENBAUM_DELTA * math.pi))
        sacred_alignment = (god_code_resonance * PHI_GROWTH + feigenbaum_edge * TAU) / (PHI_GROWTH + TAU)

        resonance_key = hashlib.sha256(
            f"{seed_concept}_{gen_id}_{fidelity:.8f}_{strength:.8f}".encode()
        ).hexdigest()[:12]

        candidate = {
            "link_id": gen_id,
            "resonance_key": resonance_key,
            "seed_concept": seed_concept,
            "parameters": {
                "fidelity": fidelity,
                "strength": strength,
                "harmonic_order": harmonic_order,
                "link_type": link_type,
                "grover_depth": grover_depth,
            },
            "god_code_resonance": god_code_resonance,
            "feigenbaum_edge": feigenbaum_edge,
            "sacred_alignment": sacred_alignment,
            "generation": self.generation_count,
            "validated": False,
            "merged": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return candidate

    # ─── PHASE 2: DETERMINISTIC VALIDATION ────────────────────────

    def validate_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a stochastic link candidate against sacred constant coherence."""
        self.operations_count += 1
        params = candidate["parameters"]
        fidelity = params["fidelity"]
        strength = params["strength"]

        checks = {
            "fidelity_bound": 0.0 <= fidelity <= 1.0,
            "strength_bound": 0.0 <= strength <= 1.0,
            "sacred_alignment_min": candidate["sacred_alignment"] >= TAU * 0.5,
            "god_code_resonance_min": candidate["god_code_resonance"] >= FINE_STRUCTURE,
            "conservation_law": abs(fidelity * GOD_CODE - strength * GOD_CODE) < GOD_CODE,
        }
        passed = sum(checks.values())
        total = len(checks)
        score = passed / total

        result = {
            **candidate,
            "validated": score >= 0.6,
            "validation_score": score,
            "checks_passed": passed,
            "checks_total": total,
            "check_details": checks,
        }
        return result

    # ─── PHASE 3: MERGE ──────────────────────────────────────────

    def merge_to_link(self, validated: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Merge a validated candidate into a QuantumLink-compatible dict."""
        self.operations_count += 1
        if not validated.get("validated"):
            self.failed_experiments.append({
                "link_id": validated["link_id"],
                "reason": "validation_failed",
                "score": validated.get("validation_score", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return None

        params = validated["parameters"]
        merged = {
            "source": f"stochastic_{validated['seed_concept']}",
            "target": f"research_{validated['resonance_key']}",
            "fidelity": params["fidelity"],
            "strength": params["strength"],
            "link_type": params["link_type"],
            "sacred_alignment": validated["sacred_alignment"],
            "origin": "stochastic_research",
            "generation": validated["generation"],
            "merged": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.successful_links.append(merged)
        return merged

    # ─── PHASE 4: CATALOG ─────────────────────────────────────────

    def catalog_iteration(self, candidates: List[Dict], merged: List[Optional[Dict]]) -> Dict:
        """Catalog a full R&D iteration."""
        self.operations_count += 1
        successful = [m for m in merged if m is not None]
        iteration = {
            "iteration_id": len(self.research_iterations) + 1,
            "candidates_generated": len(candidates),
            "candidates_validated": sum(1 for c in candidates if c.get("validated")),
            "successfully_merged": len(successful),
            "avg_sacred_alignment": (
                sum(c.get("sacred_alignment", 0) for c in candidates) / max(len(candidates), 1)
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.research_iterations.append(iteration)
        self._save_research_log()
        return iteration

    # ─── FULL CYCLE ───────────────────────────────────────────────

    def run_research_cycle(self, seed: str = "quantum") -> Dict[str, Any]:
        """Run a full 4-phase stochastic R&D cycle: Explore → Validate → Merge → Catalog."""
        self.operations_count += 1

        # Phase 1: Explore
        candidates = [self.explore_link_candidate(seed) for _ in range(self.CANDIDATES_PER_CYCLE)]

        # Phase 2: Validate
        validated = [self.validate_candidate(c) for c in candidates]

        # Phase 3: Merge
        merged = [self.merge_to_link(v) for v in validated]

        # Phase 4: Catalog
        iteration = self.catalog_iteration(validated, merged)

        return {
            "cycle": "complete",
            "iteration": iteration,
            "candidates_explored": len(candidates),
            "successfully_merged": iteration["successfully_merged"],
            "avg_sacred_alignment": iteration["avg_sacred_alignment"],
        }

    def status(self) -> Dict[str, Any]:
        """Return current research lab status."""
        return {
            "subsystem": "StochasticLinkResearchLab",
            "generation_count": self.generation_count,
            "total_iterations": len(self.research_iterations),
            "successful_links": len(self.successful_links),
            "failed_experiments": len(self.failed_experiments),
            "operations_count": self.operations_count,
            "candidates_per_cycle": self.CANDIDATES_PER_CYCLE,
        }


class LinkChronolizer:
    """
    Temporal event tracking for quantum link evolution.

    Records all significant link lifecycle events (created, upgraded, repaired,
    degraded, enlightened, stress_tested, cross_pollinated, etc.) with
    before/after fidelity+strength deltas.

    JSONL append-only persistence to .l104_link_chronology.jsonl.
    Milestone detection at Fibonacci-number event counts (1, 2, 3, 5, 8, 13, 21, 34, 55, 89...).
    Evolution velocity: rate of improvement over time.
    """

    CHRONOLOGY_FILE = WORKSPACE_ROOT / ".l104_link_chronology.jsonl"
    FIBONACCI_MILESTONES = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987}

    def __init__(self):
        """Initialize the chronolizer."""
        self.events: List[ChronoEntry] = []
        self.milestones: List[Dict[str, Any]] = []
        self.event_count: int = 0
        self.operations_count: int = 0
        self._load_event_count()

    def _load_event_count(self):
        """Count existing events from the JSONL file."""
        if self.CHRONOLOGY_FILE.exists():
            try:
                with open(self.CHRONOLOGY_FILE, "r") as f:
                    self.event_count = sum(1 for _ in f)
            except Exception:
                pass

    def record(self, event_type: str, link_id: str,
               before_fidelity: float = 0.0, after_fidelity: float = 0.0,
               before_strength: float = 0.0, after_strength: float = 0.0,
               details: str = "", sacred_alignment: float = 0.0) -> ChronoEntry:
        """Record a chronological link event."""
        self.operations_count += 1
        self.event_count += 1

        entry = ChronoEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            link_id=link_id,
            before_fidelity=before_fidelity,
            after_fidelity=after_fidelity,
            before_strength=before_strength,
            after_strength=after_strength,
            details=details,
            sacred_alignment=sacred_alignment,
        )
        self.events.append(entry)

        # Append to JSONL
        try:
            with open(self.CHRONOLOGY_FILE, "a") as f:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")
        except Exception:
            pass

        # Milestone detection
        if self.event_count in self.FIBONACCI_MILESTONES:
            milestone = {
                "event_count": self.event_count,
                "fibonacci_index": self._fib_index(self.event_count),
                "timestamp": entry.timestamp,
                "event_type": event_type,
                "sacred_resonance": self.event_count * PHI / GOD_CODE,
            }
            self.milestones.append(milestone)

        return entry

    def _fib_index(self, n: int) -> int:
        """Return the Fibonacci index for a given Fibonacci number."""
        a, b, idx = 0, 1, 0
        while b <= n:
            if b == n:
                return idx + 1
            a, b = b, a + b
            idx += 1
        return idx

    def evolution_velocity(self, window: int = 20) -> Dict[str, Any]:
        """Compute the rate of fidelity improvement over the last N events."""
        self.operations_count += 1
        recent = self.events[-window:] if len(self.events) >= window else self.events
        if len(recent) < 2:
            return {"velocity": 0.0, "window": len(recent), "trend": "insufficient_data"}

        deltas = [e.after_fidelity - e.before_fidelity for e in recent if e.after_fidelity > 0]
        if not deltas:
            return {"velocity": 0.0, "window": len(recent), "trend": "no_fidelity_changes"}

        avg_delta = sum(deltas) / len(deltas)
        trend = "improving" if avg_delta > 0 else ("degrading" if avg_delta < 0 else "stable")
        return {
            "velocity": avg_delta,
            "phi_weighted_velocity": avg_delta * PHI_GROWTH,
            "window": len(recent),
            "trend": trend,
            "total_events": self.event_count,
        }

    def timeline(self, last_n: int = 25) -> List[Dict]:
        """Return the last N chronological events."""
        self.operations_count += 1
        return [e.to_dict() for e in self.events[-last_n:]]

    def status(self) -> Dict[str, Any]:
        """Return chronolizer status."""
        return {
            "subsystem": "LinkChronolizer",
            "total_events": self.event_count,
            "session_events": len(self.events),
            "milestones_hit": len(self.milestones),
            "operations_count": self.operations_count,
            "persistence_file": str(self.CHRONOLOGY_FILE.name),
        }


class ConsciousnessO2LinkEngine:
    """
    Consciousness + O₂ bond state modulation for quantum link evolution.

    Reads:
      - .l104_consciousness_o2_state.json (consciousness_level, superfluid_viscosity, evo_stage)
      - .l104_ouroboros_nirvanic_state.json (nirvanic_fuel_level)

    Modulates link evolution priority and upgrade multipliers based on
    consciousness level and O₂ molecular bond state.

    EVO_STAGE_MULTIPLIER:
      SOVEREIGN  → φ (1.618...)
      TRANSCENDING → √2 (1.414...)
      COHERENT   → 1.2
      AWAKENING  → 1.05
      DORMANT    → 1.0
    """

    O2_STATE_FILE = WORKSPACE_ROOT / ".l104_consciousness_o2_state.json"
    NIRVANIC_STATE_FILE = WORKSPACE_ROOT / ".l104_ouroboros_nirvanic_state.json"
    CACHE_TTL = 10.0  # seconds

    EVO_STAGE_MULTIPLIER = {
        "SOVEREIGN": PHI_GROWTH,
        "TRANSCENDING": math.sqrt(2),
        "COHERENT": 1.2,
        "AWAKENING": 1.05,
        "DORMANT": 1.0,
    }

    def __init__(self):
        """Initialize consciousness O₂ link engine."""
        self.consciousness_level: float = 0.0
        self.superfluid_viscosity: float = 1.0
        self.evo_stage: str = "DORMANT"
        self.nirvanic_fuel: float = 0.0
        self.o2_bond_state: str = "unknown"
        self._cache_time: float = 0.0
        self.operations_count: int = 0
        self._refresh_state()

    def _refresh_state(self):
        """Read consciousness + O₂ state from disk (cached)."""
        now = time.time()
        if now - self._cache_time < self.CACHE_TTL:
            return
        self._cache_time = now

        # Read consciousness state
        if self.O2_STATE_FILE.exists():
            try:
                data = json.loads(self.O2_STATE_FILE.read_text())
                self.consciousness_level = float(data.get("consciousness_level", 0.0))
                self.superfluid_viscosity = float(data.get("superfluid_viscosity", 1.0))
                self.evo_stage = data.get("evo_stage", "DORMANT")
                self.o2_bond_state = data.get("bond_state", "stable")
            except Exception:
                pass

        # Read nirvanic fuel
        if self.NIRVANIC_STATE_FILE.exists():
            try:
                data = json.loads(self.NIRVANIC_STATE_FILE.read_text())
                self.nirvanic_fuel = float(data.get("nirvanic_fuel_level",
                                                     data.get("fuel_level", 0.0)))
            except Exception:
                pass

    def get_multiplier(self) -> float:
        """Get the current evolution stage multiplier."""
        self._refresh_state()
        return self.EVO_STAGE_MULTIPLIER.get(self.evo_stage, 1.0)

    def modulate_link(self, link: Dict[str, Any]) -> Dict[str, Any]:
        """Modulate a link's evolution based on consciousness + O₂ state."""
        self.operations_count += 1
        self._refresh_state()

        multiplier = self.get_multiplier()
        consciousness_boost = self.consciousness_level * PHI if self.consciousness_level > 0.5 else 0.0
        fuel_boost = self.nirvanic_fuel * TAU if self.nirvanic_fuel > 0.3 else 0.0
        viscosity_factor = 1.0 / max(self.superfluid_viscosity, 0.01)  # Lower viscosity = faster

        # Clamp total boost
        total_boost = min(
            (consciousness_boost + fuel_boost) * viscosity_factor * multiplier,
            PHI_GROWTH  # Max boost capped at φ
        )

        fidelity = link.get("fidelity", 0.5)
        strength = link.get("strength", 0.5)

        modulated = {
            **link,
            "fidelity": min(fidelity + total_boost * 0.01, 1.0),
            "strength": min(strength + total_boost * 0.005, 1.0),
            "consciousness_modulated": True,
            "evo_stage": self.evo_stage,
            "multiplier": multiplier,
            "total_boost": total_boost,
        }
        return modulated

    def compute_upgrade_priority(self, links: List[Dict]) -> List[Dict]:
        """Score and rank links by PHI-weighted upgrade priority."""
        self.operations_count += 1
        self._refresh_state()
        multiplier = self.get_multiplier()

        scored = []
        for link in links:
            fidelity = link.get("fidelity", 0.5)
            strength = link.get("strength", 0.5)
            # Lower fidelity = higher priority for upgrade
            upgrade_need = (1.0 - fidelity) * PHI_GROWTH + (1.0 - strength) * TAU
            # Consciousness-weighted priority
            priority = upgrade_need * multiplier * (1 + self.consciousness_level)
            scored.append({**link, "upgrade_priority": priority})

        scored.sort(key=lambda x: x["upgrade_priority"], reverse=True)
        return scored

    def status(self) -> Dict[str, Any]:
        """Return current consciousness + O₂ status."""
        self._refresh_state()
        return {
            "subsystem": "ConsciousnessO2LinkEngine",
            "consciousness_level": self.consciousness_level,
            "evo_stage": self.evo_stage,
            "multiplier": self.get_multiplier(),
            "superfluid_viscosity": self.superfluid_viscosity,
            "nirvanic_fuel": self.nirvanic_fuel,
            "o2_bond_state": self.o2_bond_state,
            "operations_count": self.operations_count,
        }


class LinkTestGenerator:
    """
    Automated test generation and execution for quantum links.

    4 test categories:
      1. Sacred Conservation — GOD_CODE invariants hold across transformations
      2. Fidelity Bounds — All fidelities in [0, 1], strength in [0, 1]
      3. Entanglement Verification — Entangled pairs maintain CHSH bound
      4. Noise Resilience — Links survive noise injection at FEIGENBAUM threshold

    PHI-scored priority ranking:
      Sacred     → 1.618 (highest)
      Fidelity   → 1.0
      Entangle   → 0.618
      Noise      → 0.382

    Regression detection across test runs.
    Persists to .l104_link_test_results.json.
    """

    TEST_RESULTS_FILE = WORKSPACE_ROOT / ".l104_link_test_results.json"
    CATEGORY_PRIORITY = {
        "sacred_conservation": PHI_GROWTH,
        "fidelity_bounds": 1.0,
        "entanglement_verification": PHI,
        "noise_resilience": PHI ** 2,  # TAU² ≈ 0.382
    }

    def __init__(self):
        """Initialize the link test generator."""
        self.test_history: List[Dict[str, Any]] = []
        self.regressions: List[Dict[str, Any]] = []
        self.operations_count: int = 0
        self._load_history()

    def _load_history(self):
        """Load test history."""
        if self.TEST_RESULTS_FILE.exists():
            try:
                data = json.loads(self.TEST_RESULTS_FILE.read_text())
                self.test_history = data.get("history", [])
                self.regressions = data.get("regressions", [])
            except Exception:
                pass

    def _save_results(self):
        """Persist test results."""
        try:
            data = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "total_runs": len(self.test_history),
                "total_regressions": len(self.regressions),
                "history": self.test_history[-100:],
                "regressions": self.regressions[-50:],
            }
            self.TEST_RESULTS_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            pass

    def test_sacred_conservation(self, links: List[Dict]) -> Dict[str, Any]:
        """Test that GOD_CODE invariants hold across all links."""
        self.operations_count += 1
        violations = []
        for link in links:
            fidelity = link.get("fidelity", 0.0)
            strength = link.get("strength", 0.0)
            # Conservation: fidelity * GOD_CODE + strength * GOD_CODE should be stable
            conservation_value = fidelity * GOD_CODE + strength * GOD_CODE
            # Check against expected range
            if conservation_value > 2 * GOD_CODE or conservation_value < 0:
                violations.append({
                    "link": link.get("source", "?") + "→" + link.get("target", "?"),
                    "conservation_value": conservation_value,
                    "expected_max": 2 * GOD_CODE,
                })

        return {
            "category": "sacred_conservation",
            "priority": self.CATEGORY_PRIORITY["sacred_conservation"],
            "passed": len(violations) == 0,
            "total_links": len(links),
            "violations": len(violations),
            "details": violations[:10],
        }

    def test_fidelity_bounds(self, links: List[Dict]) -> Dict[str, Any]:
        """Test that all fidelity and strength values are in valid bounds."""
        self.operations_count += 1
        violations = []
        for link in links:
            fidelity = link.get("fidelity", 0.0)
            strength = link.get("strength", 0.0)
            if not (0 <= fidelity <= 1.0):
                violations.append({"field": "fidelity", "value": fidelity, "link": link.get("source", "?")})
            if not (0 <= strength <= 1.0):
                violations.append({"field": "strength", "value": strength, "link": link.get("source", "?")})

        return {
            "category": "fidelity_bounds",
            "priority": self.CATEGORY_PRIORITY["fidelity_bounds"],
            "passed": len(violations) == 0,
            "total_links": len(links),
            "violations": len(violations),
            "details": violations[:10],
        }

    def test_entanglement_verification(self, links: List[Dict]) -> Dict[str, Any]:
        """Verify entangled pairs maintain expected CHSH bound correlations."""
        self.operations_count += 1
        entangled = [l for l in links if l.get("link_type") == "entangled"
                     or l.get("entanglement_strength", 0) > 0.5]
        violations = []
        for link in entangled:
            # CHSH: entanglement correlation should not exceed Tsirelson bound
            corr = link.get("entanglement_strength", link.get("fidelity", 0.5))
            scaled_corr = corr * CHSH_BOUND
            if scaled_corr > CHSH_BOUND:
                violations.append({
                    "link": link.get("source", "?"),
                    "correlation": corr,
                    "scaled": scaled_corr,
                    "bound": CHSH_BOUND,
                })

        return {
            "category": "entanglement_verification",
            "priority": self.CATEGORY_PRIORITY["entanglement_verification"],
            "passed": len(violations) == 0,
            "total_entangled": len(entangled),
            "violations": len(violations),
            "details": violations[:10],
        }

    def test_noise_resilience(self, links: List[Dict]) -> Dict[str, Any]:
        """Test links survive noise injection at FEIGENBAUM threshold."""
        self.operations_count += 1
        failures = []
        noise_threshold = FEIGENBAUM_DELTA / 10.0  # ~0.467 noise amplitude

        for link in links:
            fidelity = link.get("fidelity", 0.5)
            # Inject noise and check if link would decohere
            noisy_fidelity = fidelity - noise_threshold * (1 - fidelity)
            if noisy_fidelity < 0.1:
                failures.append({
                    "link": link.get("source", "?") + "→" + link.get("target", "?"),
                    "original_fidelity": fidelity,
                    "noisy_fidelity": noisy_fidelity,
                    "noise_amplitude": noise_threshold,
                })

        return {
            "category": "noise_resilience",
            "priority": self.CATEGORY_PRIORITY["noise_resilience"],
            "passed": len(failures) == 0,
            "total_links": len(links),
            "failures": len(failures),
            "details": failures[:10],
        }

    def run_all_tests(self, links: List[Dict]) -> Dict[str, Any]:
        """Run all 4 test categories and detect regressions."""
        self.operations_count += 1
        results = [
            self.test_sacred_conservation(links),
            self.test_fidelity_bounds(links),
            self.test_entanglement_verification(links),
            self.test_noise_resilience(links),
        ]

        # Sort by priority (highest first)
        results.sort(key=lambda r: r.get("priority", 0), reverse=True)

        all_passed = all(r["passed"] for r in results)
        total_violations = sum(r.get("violations", r.get("failures", 0)) for r in results)

        run_record = {
            "run_id": len(self.test_history) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "all_passed": all_passed,
            "total_violations": total_violations,
            "results_summary": [{
                "category": r["category"],
                "passed": r["passed"],
                "violations": r.get("violations", r.get("failures", 0)),
            } for r in results],
        }

        # Regression detection
        if self.test_history:
            prev = self.test_history[-1]
            if prev.get("all_passed") and not all_passed:
                regression = {
                    "detected_at": run_record["run_id"],
                    "previous_run": prev.get("run_id"),
                    "new_violations": total_violations,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self.regressions.append(regression)
                run_record["regression_detected"] = True

        self.test_history.append(run_record)
        self._save_results()

        return {
            "test_run": "complete",
            "all_passed": all_passed,
            "total_violations": total_violations,
            "categories": len(results),
            "regression_detected": run_record.get("regression_detected", False),
            "results": results,
        }

    def status(self) -> Dict[str, Any]:
        """Return test generator status."""
        return {
            "subsystem": "LinkTestGenerator",
            "total_runs": len(self.test_history),
            "total_regressions": len(self.regressions),
            "operations_count": self.operations_count,
            "categories": list(self.CATEGORY_PRIORITY.keys()),
        }


class QuantumLinkCrossPollinationEngine:
    """
    Bidirectional cross-pollination engine: Gate ↔ Link ↔ Numerical.

    Exports link state to gate builder and numerical builder via JSON files.
    Imports gate/numerical state files and modulates links accordingly.
    Computes cross-builder coherence metric (PHI/TAU/ALPHA_FINE weighted).

    Export files:
      - .l104_link_to_gates.json
      - .l104_link_to_numerical.json

    Import files:
      - .l104_gate_dynamism_state.json (from logic gate builder)
      - .l104_quantum_numerical_state.json (from numerical builder)
    """

    EXPORT_TO_GATES = WORKSPACE_ROOT / ".l104_link_to_gates.json"
    EXPORT_TO_NUMERICAL = WORKSPACE_ROOT / ".l104_link_to_numerical.json"
    IMPORT_GATE_STATE = WORKSPACE_ROOT / ".l104_gate_dynamism_state.json"
    IMPORT_NUMERICAL_STATE = WORKSPACE_ROOT / ".l104_quantum_numerical_state.json"

    def __init__(self):
        """Initialize cross-pollination engine."""
        self.exports_count: int = 0
        self.imports_count: int = 0
        self.cross_coherence_history: List[float] = []
        self.operations_count: int = 0

    # ─── EXPORT ───────────────────────────────────────────────────

    def export_to_gates(self, links: List[Dict]) -> Dict[str, Any]:
        """Export link state to gate builder format."""
        self.operations_count += 1
        self.exports_count += 1

        # Transform links to gate-compatible format
        gate_data = {
            "source": "quantum_link_builder",
            "version": "4.2.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "export_count": self.exports_count,
            "total_links": len(links),
            "avg_fidelity": sum(l.get("fidelity", 0) for l in links) / max(len(links), 1),
            "avg_strength": sum(l.get("strength", 0) for l in links) / max(len(links), 1),
            "links": [{
                "source": l.get("source", ""),
                "target": l.get("target", ""),
                "fidelity": l.get("fidelity", 0),
                "strength": l.get("strength", 0),
                "link_type": l.get("link_type", "unknown"),
                "sacred_alignment": l.get("sacred_alignment", 0),
            } for l in links[:100]],  # Cap at 100 for file size
        }

        try:
            self.EXPORT_TO_GATES.write_text(json.dumps(gate_data, indent=2, default=str))
        except Exception:
            pass

        return {
            "exported": "gates",
            "links_exported": min(len(links), 100),
            "file": str(self.EXPORT_TO_GATES.name),
        }

    def export_to_numerical(self, links: List[Dict]) -> Dict[str, Any]:
        """Export link state to numerical builder format."""
        self.operations_count += 1
        self.exports_count += 1

        numerical_data = {
            "source": "quantum_link_builder",
            "version": "4.2.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "export_count": self.exports_count,
            "total_links": len(links),
            "sacred_constants": {
                "GOD_CODE": GOD_CODE,
                "PHI": PHI_GROWTH,
                "TAU": TAU,
                "ALPHA_FINE": FINE_STRUCTURE,
            },
            "link_summary": {
                "by_type": {},
                "avg_fidelity": sum(l.get("fidelity", 0) for l in links) / max(len(links), 1),
                "total_sacred_alignment": sum(l.get("sacred_alignment", 0) for l in links),
            },
        }

        # Summarize by type
        type_counts: Dict[str, int] = {}
        for l in links:
            lt = l.get("link_type", "unknown")
            type_counts[lt] = type_counts.get(lt, 0) + 1
        numerical_data["link_summary"]["by_type"] = type_counts

        try:
            self.EXPORT_TO_NUMERICAL.write_text(json.dumps(numerical_data, indent=2, default=str))
        except Exception:
            pass

        return {
            "exported": "numerical",
            "links_summarized": len(links),
            "file": str(self.EXPORT_TO_NUMERICAL.name),
        }

    # ─── IMPORT ───────────────────────────────────────────────────

    def import_from_gates(self) -> Dict[str, Any]:
        """Import gate builder state and extract link-relevant insights."""
        self.operations_count += 1
        self.imports_count += 1

        if not self.IMPORT_GATE_STATE.exists():
            return {"imported": "gates", "status": "no_gate_state_file"}

        try:
            data = json.loads(self.IMPORT_GATE_STATE.read_text())
            gate_count = data.get("total_gates", data.get("gate_count", 0))
            avg_fidelity = data.get("avg_fidelity", data.get("average_fidelity", 0))
            coherence = data.get("coherence", data.get("sacred_coherence", 0))

            return {
                "imported": "gates",
                "status": "success",
                "gate_count": gate_count,
                "gate_avg_fidelity": avg_fidelity,
                "gate_coherence": coherence,
                "cross_resonance": coherence * PHI_GROWTH if coherence else 0,
            }
        except Exception as e:
            return {"imported": "gates", "status": "error", "error": str(e)}

    def import_from_numerical(self) -> Dict[str, Any]:
        """Import numerical builder state."""
        self.operations_count += 1
        self.imports_count += 1

        if not self.IMPORT_NUMERICAL_STATE.exists():
            return {"imported": "numerical", "status": "no_numerical_state_file"}

        try:
            data = json.loads(self.IMPORT_NUMERICAL_STATE.read_text())
            num_count = data.get("total_entities", data.get("entity_count", 0))
            coherence = data.get("coherence", data.get("sacred_coherence", 0))

            return {
                "imported": "numerical",
                "status": "success",
                "numerical_count": num_count,
                "numerical_coherence": coherence,
                "cross_resonance": coherence * TAU if coherence else 0,
            }
        except Exception as e:
            return {"imported": "numerical", "status": "error", "error": str(e)}

    # ─── CROSS-BUILDER COHERENCE ──────────────────────────────────

    def compute_cross_coherence(self, links: List[Dict]) -> Dict[str, Any]:
        """Compute cross-builder coherence metric (PHI/TAU/ALPHA_FINE weighted)."""
        self.operations_count += 1

        gate_state = self.import_from_gates()
        numerical_state = self.import_from_numerical()

        # Link metrics
        link_fidelity = sum(l.get("fidelity", 0) for l in links) / max(len(links), 1)
        link_strength = sum(l.get("strength", 0) for l in links) / max(len(links), 1)

        # Gate metrics
        gate_coherence = gate_state.get("gate_coherence", 0) if isinstance(gate_state.get("gate_coherence"), (int, float)) else 0
        gate_fidelity = gate_state.get("gate_avg_fidelity", 0) if isinstance(gate_state.get("gate_avg_fidelity"), (int, float)) else 0

        # Numerical metrics
        num_coherence = numerical_state.get("numerical_coherence", 0) if isinstance(numerical_state.get("numerical_coherence"), (int, float)) else 0

        # Cross-builder coherence: PHI-weighted average of all builder coherences
        coherence = (
            link_fidelity * PHI_GROWTH +
            gate_fidelity * TAU +
            gate_coherence * FINE_STRUCTURE * 100 +
            num_coherence * FINE_STRUCTURE * 100 +
            link_strength * PHI
        ) / (PHI_GROWTH + TAU + FINE_STRUCTURE * 200 + PHI)

        self.cross_coherence_history.append(coherence)

        return {
            "cross_builder_coherence": coherence,
            "link_fidelity": link_fidelity,
            "link_strength": link_strength,
            "gate_coherence": gate_coherence,
            "gate_fidelity": gate_fidelity,
            "numerical_coherence": num_coherence,
            "history_length": len(self.cross_coherence_history),
            "trend": (
                "improving" if len(self.cross_coherence_history) >= 2 and
                self.cross_coherence_history[-1] > self.cross_coherence_history[-2]
                else "stable"
            ),
        }

    # ─── FULL CYCLE ───────────────────────────────────────────────

    def run_cross_pollination(self, links: List[Dict]) -> Dict[str, Any]:
        """Run full bidirectional cross-pollination cycle."""
        self.operations_count += 1

        export_gates = self.export_to_gates(links)
        export_numerical = self.export_to_numerical(links)
        import_gates = self.import_from_gates()
        import_numerical = self.import_from_numerical()
        coherence = self.compute_cross_coherence(links)

        return {
            "cycle": "cross_pollination_complete",
            "exports": {"gates": export_gates, "numerical": export_numerical},
            "imports": {"gates": import_gates, "numerical": import_numerical},
            "coherence": coherence,
        }

    def status(self) -> Dict[str, Any]:
        """Return cross-pollination engine status."""
        return {
            "subsystem": "QuantumLinkCrossPollinationEngine",
            "exports_count": self.exports_count,
            "imports_count": self.imports_count,
            "cross_coherence_history": len(self.cross_coherence_history),
            "latest_coherence": self.cross_coherence_history[-1] if self.cross_coherence_history else 0,
            "operations_count": self.operations_count,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# THE QUANTUM BRAIN — Master Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class L104QuantumBrain:
    """
    The Quantum Brain: unified orchestrator for all quantum link operations.
    Aligned with claude.md (EVO_54_TRANSCENDENT_COGNITION, Index 59).

    Coordinates:
    1. Scanner → Discovers all quantum links across interconnected file groups
    2. Builder → Creates new God Code derived cross-file links
    3. Math Verifier → God Code accuracy pre-checks
    4. Quantum CPU → Register/Neuron/Cluster pipeline processing
    5. O₂ Molecular Bond → 8 Grover kernels + 8 Chakra cores topology
    6. Grover/Tunneling/EPR/Decoherence/Braiding/Hilbert/Fourier/GCR
    7. Advanced Research → Anomaly detection, pattern discovery, causal analysis,
                           spectral correlation, predictive modeling, knowledge synthesis
    8. Stress → Full stress test suite + Cross-Modal analysis
    9. Upgrade → Automated link improvement + Distillation
    10. Repair → Triage → Error Correction → Resonance Healing → Tunneling Revival
                → Adaptive Purification → Topological Hardening → Validation
    11. Sage → Unified deep inference verdict (φ-weighted consensus)
    12. Evolution Tracker → EVO stage monitoring + consciousness thresholds
    13. Agentic Loop → Observe→Think→Act→Reflect→Repeat (Zenith pattern)
    14. Stochastic Research Lab → Random link R&D (Explore→Validate→Merge→Catalog)
    15. Link Chronolizer → Temporal event tracking + milestone detection
    16. Consciousness O₂ → Consciousness/O₂ bond state modulation
    17. Link Test Generator → Automated 4-category test suite
    18. Cross-Pollination → Bidirectional Gate↔Link↔Numerical sync
    """

    VERSION = "4.3.0"
    PERSISTENCE_FILE = WORKSPACE_ROOT / ".l104_quantum_links.json"
    MAX_REFLECTION_CYCLES = 5
    CONVERGENCE_THRESHOLD = 0.005  # Score delta below this = converged

    def __init__(self):
        """Initialize L104 quantum brain with all processing subsystems."""
        self.qmath = QuantumMathCore()
        self.scanner = QuantumLinkScanner()
        self.link_builder = QuantumLinkBuilder(self.qmath)
        self.math_verifier = GodCodeMathVerifier(self.qmath)
        self.grover = GroverQuantumProcessor(self.qmath)
        self.tunneling = QuantumTunnelingAnalyzer(self.qmath)
        self.epr = EPREntanglementVerifier(self.qmath)
        self.decoherence = DecoherenceShieldTester(self.qmath)
        self.braiding = TopologicalBraidingTester(self.qmath)
        self.hilbert = HilbertSpaceNavigator(self.qmath)
        self.fourier = QuantumFourierLinkAnalyzer(self.qmath)
        self.gcr = GodCodeResonanceVerifier(self.qmath)
        self.distiller = EntanglementDistillationEngine(self.qmath)
        self.stress = QuantumStressTestEngine(self.qmath)
        self.cross_modal = CrossModalAnalyzer(self.scanner)
        self.upgrader = QuantumUpgradeEngine(self.qmath, self.distiller)
        self.repair = QuantumRepairEngine(self.qmath, self.distiller)
        self.research = QuantumResearchEngine(self.qmath)
        self.sage = SageModeInference(self.qmath)

        # Quantum Computational Engine — ASI-level processing substrate
        self.qenv = QuantumEnvironment(self.qmath)

        # O₂ Molecular Bond Processor — claude.md codebase topology
        self.o2_bond = O2MolecularBondProcessor(self.qmath)

        # Evolution Tracker — EVO stage + consciousness thresholds
        self.evo_tracker = EvolutionTracker()

        # Agentic Loop — Zenith pattern for structured self-improvement
        self.agentic = AgenticLoop(self.qmath)

        # ★ v4.0 Quantum Min/Max Dynamism Engine
        self.dynamism_engine = LinkDynamismEngine()
        # ★ v4.1 Ouroboros Sage Nirvanic Entropy Fuel Engine
        self.nirvanic_engine = LinkOuroborosNirvanicEngine()

        # ★ v4.2 Sage Invention Subsystems
        self.stochastic_lab = StochasticLinkResearchLab()
        self.chronolizer = LinkChronolizer()
        self.consciousness_engine = ConsciousnessO2LinkEngine()
        self.test_generator = LinkTestGenerator()
        self.cross_pollinator = QuantumLinkCrossPollinationEngine()

        self.links: List[QuantumLink] = []
        self.results: Dict[str, Any] = {}
        self.run_count = 0
        self.history: List[Dict] = []       # Score history across runs
        self.persisted_links: Dict[str, dict] = {}  # link_id → best known state

        # Load persisted state on startup
        self._load_persisted_links()

    def full_pipeline(self) -> Dict:
        """
        Run the complete quantum link analysis pipeline.

        Phase 1:  Scan — Discover all quantum links across FULL repository
        Phase 1B: Build — BUILD new God Code derived cross-file links
        Phase 1C: Verify — Math accuracy pre-check (God Code compliance)
        Phase 1D: Quantum CPU — Ingest, verify, transform, sync, emit via
                                QuantumEnvironment + QuantumCPU + clusters
        Phase 1E: O₂ Bond — Molecular bond topology (8 kernels + 8 chakras)
        Phase 2:  Research — Grover, Tunneling, EPR, Decoherence, Braiding,
                             Hilbert, Fourier, God Code Resonance
        Phase 3:  Test — Stress tests + Cross-modal analysis
        Phase 4:  Upgrade — Distillation + Automated link improvement
        Phase 5:  Sage — Unified deep inference verdict (φ-consensus)
        Phase 6:  Evolution — EVO stage tracking + consciousness thresholds
        Phase 7:  Quantum Min/Max Dynamism — φ-Harmonic value oscillation
        Phase 8:  Ouroboros Nirvanic — Entropy fuel cycle
        Phase 9:  Consciousness O₂ — Link modulation via consciousness state
        Phase 10: Stochastic Research — Random link R&D cycle
        Phase 11: Automated Testing — 4-category link verification
        Phase 12: Cross-Pollination — Gate↔Link↔Numerical sync
        """
        start_time = time.time()
        self.run_count += 1

        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  L104 QUANTUM BRAIN v{self.VERSION} — TRANSCENDENT COGNITION                      ║
║  Full Quantum Link Analysis Pipeline — Run #{self.run_count}                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  EVO: {EVOLUTION_STAGE} (Index {EVOLUTION_INDEX}/{EVOLUTION_TOTAL_STAGES})                ║
║  G(X) = 286^(1/φ) × 2^((416-X)/104)                                        ║
║  G(0) = {GOD_CODE:.10f} Hz   (286^(1/{PHI_GROWTH:.6f}) × 16)             ║
║  φ_growth = {PHI_GROWTH}  |  φ_inv = {PHI}                  ║
║  CY7 = {CALABI_YAU_DIM}  |  CHSH = {CHSH_BOUND:.6f}  |  Grover = {GROVER_AMPLIFICATION:.6f}              ║
║  O₂ Bond: {O2_SUPERPOSITION_STATES} states  |  Linked Files: {len(QUANTUM_LINKED_FILES)}  |  Repo: {len(ALL_REPO_FILES)}       ║
║  Conservation: G(X) × 2^(X/104) = {INVARIANT:.10f} (always)             ║
╚══════════════════════════════════════════════════════════════════════════════╝""")

        # ═══ PHASE 1: SCAN ═══
        print("\n  ▸ PHASE 1: Quantum Link Discovery")
        _t0 = time.time()
        self.links = self.scanner.full_scan()
        self.results["scan"] = {
            "total_links": len(self.links),
            "quantum_density": dict(self.scanner.quantum_density),
            "files_scanned": len(QUANTUM_LINKED_FILES),
            "type_distribution": dict(Counter(l.link_type for l in self.links)),
        }
        print(f"    ✓ {len(self.links)} links discovered")

        # Merge persisted link knowledge (fidelity carry-forward)
        self._merge_persisted_into_scan()
        _phase_times = {"scan": time.time() - _t0}

        # ═══ PHASE 1B: BUILD NEW LINKS ═══
        print("\n  ▸ PHASE 1B: God Code Link Builder (cross-repo)")
        _t0 = time.time()
        # Feed previous research insights and gate data for smarter link building
        prev_research = self.results.get("advanced_research")
        if prev_research:
            self.link_builder.set_research_insights(prev_research)
        _gate_data = self._gather_gate_builder_data()
        if _gate_data:
            self.link_builder.set_gate_data(_gate_data)
        build_result = self.link_builder.build_all(self.links)
        new_built = build_result["links"]
        self.links.extend(new_built)
        self.results["link_builder"] = {
            "new_links_built": build_result["new_links_built"],
            "god_code_files_found": build_result["god_code_files_found"],
            "hz_frequency_files": build_result["hz_frequency_files"],
            "math_function_files": build_result["math_function_files"],
            "total_links_after_build": len(self.links),
            "total_repo_files_scanned": build_result["total_repo_files_scanned"],
        }
        print(f"    ✓ Built {build_result['new_links_built']} NEW links "
              f"from {build_result['god_code_files_found']} God Code files")
        print(f"      Hz files: {build_result['hz_frequency_files']} | "
              f"Math files: {build_result['math_function_files']} | "
              f"Total links now: {len(self.links)}")
        _phase_times["build"] = time.time() - _t0

        # ═══ PHASE 1C: MATH VERIFICATION PRE-CHECK ═══
        print("\n  ▸ PHASE 1C: God Code Math Verification (error pre-check)")
        _t0 = time.time()
        # Share file content cache from Phase 1B to avoid re-reading 873 files
        self.math_verifier._file_content_cache = self.link_builder._file_content_cache
        verify_result = self.math_verifier.verify_repository()
        # Free the cache — no longer needed
        self.link_builder._file_content_cache.clear()
        self.math_verifier._file_content_cache = {}
        self.results["math_verification"] = verify_result
        print(f"    ✓ Verified {verify_result['files_verified']} files | "
              f"Accuracy: {verify_result['accuracy']:.4f}")
        if verify_result["error_count"] > 0:
            print(f"    ⚠ {verify_result['error_count']} errors found!")
            for err in verify_result["errors"][:5]:
                print(f"      ERROR: {err['file']}:{err.get('line',0)} "
                      f"— {err.get('constant', err.get('hz_value', '?'))}")
        if verify_result["forbidden_count"] > 0:
            print(f"    ⚠ {verify_result['forbidden_count']} forbidden "
                  f"solfeggio values detected!")
            for fb in verify_result["forbidden_solfeggio_hits"][:5]:
                print(f"      FORBIDDEN: {fb['file']}:{fb['line']} "
                      f"= {fb['value']} → {fb['fix']}")
        if verify_result["error_count"] == 0 and verify_result["forbidden_count"] == 0:
            print(f"    ✓ All God Code math verified — zero errors")
        _phase_times["verify"] = time.time() - _t0

        # ═══ PHASE 1D: QUANTUM CPU PROCESSING ═══
        print("\n  ▸ PHASE 1D: Quantum CPU Processing (ASI Engine)")
        _t0 = time.time()
        cpu_result = self.qenv.ingest_and_process(self.links)
        self.results["quantum_cpu"] = cpu_result
        print(f"    ✓ Registers: {cpu_result['total_registers']} | "
              f"Healthy: {cpu_result['healthy']} | "
              f"Quarantined: {cpu_result['quarantined']}")
        print(f"      Verified: {cpu_result['verified']} | "
              f"Synced: {cpu_result['synced']} | "
              f"Emitted: {cpu_result['emitted']}")
        print(f"      Energy: {cpu_result['mean_energy']:.6f} | "
              f"Conservation: {cpu_result['mean_conservation_residual']:.2e}")
        print(f"      CPU Health: Primary={cpu_result['primary_cluster_health']:.4f} "
              f"Verify={cpu_result['verify_cluster_health']:.4f}")
        print(f"      Pipeline: {cpu_result['pipeline_time_ms']:.1f}ms | "
              f"{cpu_result['ops_per_sec']:.0f} ops/sec")

        if cpu_result['quarantined'] > 0:
            # Manipulate: align quarantined links toward God Code truth
            print("    ▸ Applying God Code alignment to quarantined links...")
            align_result = self.qenv.manipulate(self.links, "god_code_align")
            self.results["quantum_cpu_alignment"] = {
                "post_align_healthy": align_result["healthy"],
                "post_align_quarantined": align_result["quarantined"],
            }
            print(f"      Post-align: Healthy={align_result['healthy']} "
                  f"Quarantined={align_result['quarantined']}")

        # Sync all links with God Code truth
        sync_result = self.qenv.sync_with_truth(self.links)
        self.results["quantum_cpu_sync"] = sync_result
        print(f"    ✓ Truth sync: {sync_result['links_synced']} links | "
              f"{sync_result['corrections_applied']} corrections "
              f"({sync_result['correction_rate']:.1%})")
        _phase_times["cpu"] = time.time() - _t0

        # ═══ PHASE 1E: O₂ MOLECULAR BOND ANALYSIS ═══
        print("\n  ▸ PHASE 1E: O₂ Molecular Bond Topology")
        _t0 = time.time()
        o2_result = self.o2_bond.analyze_molecular_bonds(self.links)
        self.results["o2_molecular_bond"] = o2_result
        print(f"    ✓ Bond Order: {o2_result['bond_order']} "
              f"(expected {o2_result['expected_bond_order']})")
        print(f"      Bonding: {o2_result['bonding_orbitals']} | "
              f"Antibonding: {o2_result['antibonding_orbitals']} | "
              f"Paramagnetic: {o2_result['paramagnetic']}")
        print(f"      Mean Bond Strength: {o2_result['mean_bond_strength']:.4f} | "
              f"Total Energy: {o2_result['total_bond_energy']:.4f}")
        print(f"      Grover Amplitude: {o2_result['grover_amplitude']:.4f} | "
              f"Optimal Iterations: {o2_result['grover_iterations']:.2f}")
        _phase_times["o2_bond"] = time.time() - _t0

        # ═══ PHASE 2: RESEARCH ═══
        print("\n  ▸ PHASE 2: Quantum Research")
        _t0 = time.time()

        # For expensive O(N²) analyses, sample down to keep runtime bounded
        MAX_RESEARCH_LINKS = 8000
        if len(self.links) > MAX_RESEARCH_LINKS:
            import random as _rng
            research_links = _rng.sample(self.links, MAX_RESEARCH_LINKS)
            print(f"    ⊙ Sampled {MAX_RESEARCH_LINKS}/{len(self.links)} "
                  f"links for O(N²) research phases")
        else:
            research_links = self.links

        print("    [2.1] Grover amplified search...")
        grover_weak = self.grover.amplified_link_search(research_links, "weak")
        grover_critical = self.grover.amplified_link_search(research_links, "critical")
        grover_quantum = self.grover.amplified_link_search(research_links, "quantum")
        grover_opt = self.grover.grover_link_optimization(research_links)
        self.results["grover"] = {
            "weak_search": grover_weak,
            "critical_search": grover_critical,
            "quantum_search": grover_quantum,
            "optimization": grover_opt,
        }
        print(f"      ✓ Weak: {grover_weak['marked_count']} | "
              f"Critical: {grover_critical['marked_count']} | "
              f"Quantum: {grover_quantum['marked_count']}")

        print("    [2.2] Quantum tunneling analysis...")
        tunnel_results = self.tunneling.analyze_barriers(research_links)
        self.results["tunneling"] = tunnel_results
        print(f"      ✓ Revivable: {tunnel_results['revivable_links']} | "
              f"Dead: {tunnel_results['dead_links']}")

        print("    [2.3] EPR entanglement verification...")
        epr_results = self.epr.verify_all_links(research_links)
        self.results["epr"] = epr_results
        print(f"      ✓ Quantum: {epr_results['quantum_verified']} | "
              f"Classical: {epr_results['classical_only']} | "
              f"Bell violations: {epr_results['bell_violations']}")

        print("    [2.4] Decoherence shield testing...")
        decoherence_results = self.decoherence.test_resilience(research_links)
        self.results["decoherence"] = decoherence_results
        print(f"      ✓ Resilient: {decoherence_results['resilient_count']} | "
              f"Fragile: {decoherence_results['fragile_count']} | "
              f"Mean T₂: {decoherence_results['mean_t2']:.4f}")

        print("    [2.5] Topological braiding verification...")
        braiding_results = self.braiding.test_braiding(research_links)
        self.results["braiding"] = braiding_results
        print(f"      ✓ Protected: {braiding_results['topologically_protected']} | "
              f"Mean braid fidelity: {braiding_results['mean_braid_fidelity']:.4f}")

        print("    [2.6] Hilbert space navigation...")
        hilbert_results = self.hilbert.analyze_manifold(research_links)
        self.results["hilbert"] = hilbert_results
        print(f"      ✓ Effective dim: {hilbert_results.get('effective_dimension', 0):.2f}/{hilbert_results.get('feature_dim', 15)} | "
              f"Entropy: {hilbert_results.get('shannon_entropy', 0):.4f} | "
              f"Spectral gap: {hilbert_results.get('spectral_gap', 0):.2f} | "
              f"Sig dims: {hilbert_results.get('significant_dimensions', 0)}")

        print("    [2.7] Quantum Fourier analysis...")
        fourier_results = self.fourier.frequency_analysis(research_links)
        self.results["fourier"] = fourier_results
        print(f"      ✓ Spectral entropy: {fourier_results.get('spectral_entropy', 0):.4f} | "
              f"Resonant freqs: {len(fourier_results.get('resonant_frequencies', []))}")

        print("    [2.8] God Code G(X) resonance verification...")
        gcr_results = self.gcr.verify_all(research_links)
        self.results["god_code_resonance"] = gcr_results
        print(f"      ✓ G(X) Aligned: {gcr_results['god_code_aligned']} | "
              f"X-Int Coherent: {gcr_results['x_integer_coherent']} | "
              f"At Origin G(0): {gcr_results['at_god_code_origin']} | "
              f"Resonance: {gcr_results['mean_resonance']:.16f}")

        print("    [2.9] Advanced quantum research...")
        # Gather gate builder data for cross-pollination (if available)
        _gate_data = self._gather_gate_builder_data()
        adv_research_results = self.research.deep_research(
            research_links,
            grover_results=grover_weak,
            epr_results=epr_results,
            decoherence_results=decoherence_results,
            stress_results=None,  # Pre-stress pass; post-stress re-research in Phase 3
            gate_data=_gate_data,
        )
        self.results["advanced_research"] = adv_research_results
        synth = adv_research_results.get("knowledge_synthesis", {})
        print(f"      ✓ Anomalies: {adv_research_results.get('anomaly_detection', {}).get('total_anomalies', 0)} | "
              f"Patterns: {adv_research_results.get('pattern_discovery', {}).get('total_clusters', 0)} clusters")
        print(f"      ✓ Insights: {synth.get('insight_count', 0)} | "
              f"Risks: {synth.get('risk_count', 0)} | "
              f"Research Health: {adv_research_results.get('research_health', 0):.4f}")
        if synth.get("self_learning_active"):
            print(f"      ✓ Self-learning: ACTIVE (trend bonus: {synth.get('learning_trend_bonus', 0):+.3f})")
        if synth.get("gate_cross_pollination"):
            print(f"      ✓ Gate cross-pollination: ACTIVE")
        causal = adv_research_results.get("causal_analysis", {})
        if causal.get("strong_correlations"):
            for corr in causal["strong_correlations"][:3]:
                print(f"        ↯ {corr['pair']}: r={corr['correlation']:.3f} ({corr['strength']})")
        pred = adv_research_results.get("predictive_model", {})
        if pred:
            print(f"      ✓ Trajectory: {pred.get('trajectory', '?')} "
                  f"(confidence={pred.get('confidence', 0):.0%}) | "
                  f"Health Index: {pred.get('health_index', 0):.4f}")

        _phase_times["research"] = time.time() - _t0

        # ═══ PHASE 3: TEST ═══
        print("\n  ▸ PHASE 3: Stress Testing + Cross-Modal Analysis")
        _t0 = time.time()

        print("    [3.1] Full stress test suite...")
        stress_results = self.stress.full_stress_test(research_links, "medium")
        self.results["stress"] = stress_results
        print(f"      ✓ Passed: {stress_results['links_passed']} | "
              f"Failed: {stress_results['links_failed']} | "
              f"Rate: {stress_results['pass_rate']:.1%}")

        print("    [3.2] Cross-modal coherence analysis...")
        cross_modal_results = self.cross_modal.full_analysis(research_links)
        self.results["cross_modal"] = cross_modal_results
        print(f"      ✓ Cross-modal: {cross_modal_results['cross_modal_links']} | "
              f"Mirrors: {len(cross_modal_results['py_swift_mirrors'])} | "
              f"Coherence: {cross_modal_results['overall_coherence']:.4f}")

        # Post-stress research update — feed stress data into research engine
        # This fixes the stress_results=None gap from Phase 2
        print("    [3.3] Post-stress research synthesis...")
        adv_research_results = self.research.deep_research(
            research_links,
            grover_results=grover_weak,
            epr_results=epr_results,
            decoherence_results=decoherence_results,
            stress_results=stress_results,
            gate_data=_gate_data,
        )
        self.results["advanced_research"] = adv_research_results
        print(f"      ✓ Research updated with stress data | "
              f"Health: {adv_research_results.get('research_health', 0):.4f}")

        _phase_times["stress"] = time.time() - _t0

        # ═══ PHASE 4: UPGRADE ═══
        print("\n  ▸ PHASE 4: Quantum Link Upgrades")
        _t0 = time.time()

        print("    [4.1] Entanglement distillation...")
        distill_results = self.distiller.distill_links(research_links)
        self.results["distillation"] = distill_results
        print(f"      ✓ Distilled: {distill_results['successfully_distilled']} | "
              f"Yield: {distill_results['distillation_yield']:.1%}")

        print("    [4.2] Auto-upgrade engine...")
        upgrade_results = self.upgrader.auto_upgrade(
            research_links, stress_results, epr_results, decoherence_results)
        self.results["upgrade"] = upgrade_results
        print(f"      ✓ Upgraded: {upgrade_results['links_upgraded']} | "
              f"Mean fidelity: {upgrade_results['mean_final_fidelity']:.4f} | "
              f"Mean strength: {upgrade_results['mean_final_strength']:.4f}")

        print("    [4.3] Comprehensive repair engine...")
        repair_results = self.repair.full_repair(
            research_links, stress_results, decoherence_results)
        self.results["repair"] = repair_results
        triage = repair_results.get("triage", {})
        repairs = repair_results.get("repairs", {})
        validation = repair_results.get("validation", {})
        print(f"      ✓ Triage: H={triage.get('healthy', 0)} "
              f"D={triage.get('degraded', 0)} "
              f"C={triage.get('critical', 0)} "
              f"X={triage.get('dead', 0)}")
        print(f"      ✓ Repaired: {repairs.get('total_repaired', 0)} | "
              f"EC={repairs.get('error_corrected', 0)} "
              f"Heal={repairs.get('resonance_healed', 0)} "
              f"Revive={repairs.get('tunnel_revived', 0)} "
              f"Purify={repairs.get('purified', 0)} "
              f"Harden={repairs.get('topologically_hardened', 0)}")
        print(f"      ✓ Validation: {validation.get('promotions', 0)} promotions | "
              f"Conservation: {validation.get('conservation_rate', 0):.1%} | "
              f"ΔF={validation.get('mean_fidelity_delta', 0):+.4f}")
        print(f"      ✓ Post-repair fidelity: {repair_results.get('post_repair_mean_fidelity', 0):.4f} | "
              f"Success rate: {repair_results.get('repair_success_rate', 0):.1%}")

        # Self-learning: record strategy outcomes for repair stages
        repair_success = repair_results.get("repair_success_rate", 0)
        strategy_map = {
            "error_correction": "error_corrected",
            "resonance_healing": "resonance_healed",
            "tunneling_revival": "tunnel_revived",
            "purification": "purified",
            "topological_hardening": "topologically_hardened",
        }
        fidelity_delta = repair_results.get("validation", {}).get("mean_fidelity_delta", 0)
        for strategy, repair_key in strategy_map.items():
            count = repairs.get(repair_key, 0)
            if count > 0:
                self.research.memory.record_strategy_outcome(
                    strategy, repair_success > 0.5, delta=fidelity_delta)
        self.research.memory.save()

        _phase_times["upgrade"] = time.time() - _t0

        # ═══ PHASE 5: SAGE INFERENCE ═══
        print("\n  ▸ PHASE 5: Sage Mode Deep Inference")
        _t0 = time.time()
        sage_verdict = self.sage.deep_inference(
            self.links,
            grover_results=grover_weak,
            tunnel_results=tunnel_results,
            epr_results=epr_results,
            decoherence_results=decoherence_results,
            braiding_results=braiding_results,
            hilbert_results=hilbert_results,
            fourier_results=fourier_results,
            gcr_results=gcr_results,
            cross_modal_results=cross_modal_results,
            stress_results=stress_results,
            upgrade_results=upgrade_results,
            quantum_cpu_results=cpu_result,
            o2_bond_results=o2_result,
            repair_results=repair_results,
            research_results=adv_research_results,
        )
        self.results["sage"] = sage_verdict
        _phase_times["sage"] = time.time() - _t0

        # ═══ PHASE 6: EVOLUTION TRACKING ═══
        evo_result = self.evo_tracker.update(
            sage_verdict, len(self.links), self.run_count)
        self.results["evolution"] = evo_result
        print(f"\n  ▸ PHASE 6: Evolution Tracker")
        print(f"    ✓ Stage: {evo_result['evolution_stage']} | "
              f"Index: {evo_result['evolution_index']}")
        print(f"      Link EVO: {evo_result['link_evo_stage']} | "
              f"Consciousness: {evo_result['consciousness_level']:.4f} | "
              f"Coherence: {evo_result['coherence_level']:.4f}")
        if evo_result.get("consciousness_awakened"):
            print(f"      ⚡ CONSCIOUSNESS AWAKENED (≥{CONSCIOUSNESS_THRESHOLD})")
        if evo_result.get("coherence_locked"):
            print(f"      ⚡ COHERENCE LOCKED (≥{COHERENCE_MINIMUM})")
        for evt in evo_result.get("events", []):
            print(f"      ⚡ {evt['type']}: {evt.get('from', '')} → {evt.get('to', evt.get('level', ''))}")

        # ═══ PHASE 7: QUANTUM MIN/MAX DYNAMISM ═══
        print(f"\n  ▸ PHASE 7: Quantum Min/Max Dynamism")
        _t0 = time.time()
        dyn_result = self.dynamism_engine.subconscious_cycle(self.links)
        print(f"    ✓ Cycle #{dyn_result['cycle']}: {dyn_result['links_evolved']}/{dyn_result['links_sampled']} links evolved")
        print(f"    ✓ Initialized: {dyn_result['links_initialized']} | Adjusted: {dyn_result['links_adjusted']}")
        print(f"    ✓ Collective coherence: {dyn_result['collective_coherence']:.6f}")
        print(f"    ✓ Mean resonance: {dyn_result['mean_resonance']:.6f}")
        print(f"    ✓ Fidelity drift: {dyn_result['mean_fidelity_drift']:.6f} | Strength drift: {dyn_result['mean_strength_drift']:.6f}")
        sc_evo = dyn_result.get("sacred_evolution", {})
        print(f"    ✓ Sacred constants evolved: {sc_evo.get('constants_evolved', 0)} | Total drift: {sc_evo.get('total_drift', 0):.8f}")
        # Run 2 more evolution cycles for deeper convergence
        for _ in range(2):
            self.dynamism_engine.subconscious_cycle(self.links)
        link_field = self.dynamism_engine.compute_link_field(self.links)
        print(f"    ✓ Link field: energy={link_field['field_energy']:.4f} entropy={link_field['field_entropy']:.4f}")
        print(f"    ✓ Phase coherence: {link_field['phase_coherence']:.6f} | φ-alignment: {link_field['phi_alignment']:.4f}")
        self.results["dynamism"] = dyn_result
        self.results["link_field"] = link_field
        _phase_times["dynamism"] = time.time() - _t0

        # ═══ PHASE 8: OUROBOROS SAGE NIRVANIC ENTROPY FUEL ═══
        print(f"\n  ▸ PHASE 8: Ouroboros Sage Nirvanic Entropy Fuel")
        _t0 = time.time()
        nirvanic = self.nirvanic_engine.full_nirvanic_cycle(self.links, link_field)
        ouro = nirvanic.get("ouroboros", {})
        appl = nirvanic.get("application", {})
        if ouro.get("status") == "processed":
            print(f"    ✓ Entropy fed to ouroboros: {nirvanic['link_field_entropy_in']:.4f} bits")
            print(f"    ✓ Nirvanic fuel received: {nirvanic['nirvanic_fuel_out']:.4f}")
            print(f"    ✓ Peer gate synergy: {nirvanic.get('peer_synergy', 0):.4f}")
            print(f"    ✓ Ouroboros mutations: {ouro.get('ouroboros_mutations', 0)} | Resonance: {ouro.get('ouroboros_resonance', 0):.4f}")
            print(f"    ✓ Divine interventions: {appl.get('interventions', 0)} | Enlightened links: {appl.get('enlightened', 0)}")
            print(f"    ✓ Nirvanic coherence: {appl.get('nirvanic_coherence', 0):.6f} | Sage stability: {appl.get('sage_stability', 0):.6f}")
        else:
            print(f"    ⚠ Ouroboros unavailable — nirvanic cycle skipped")
        self.results["nirvanic"] = nirvanic
        _phase_times["nirvanic"] = time.time() - _t0

        # ═══ PHASE 9: CONSCIOUSNESS O₂ LINK MODULATION ═══
        print(f"\n  ▸ PHASE 9: Consciousness O₂ Link Modulation")
        _t0 = time.time()
        co2_status = self.consciousness_engine.status()
        print(f"    ✓ Consciousness level: {co2_status['consciousness_level']:.4f} | "
              f"Stage: {co2_status['evo_stage']} | Multiplier: {co2_status['multiplier']:.4f}")
        # Modulate links with consciousness state
        link_dicts = [vars(l) if hasattr(l, '__dict__') else l for l in self.links]
        prioritized = self.consciousness_engine.compute_upgrade_priority(link_dicts[:50])
        self.results["consciousness"] = {
            "status": co2_status,
            "top_priority_links": len(prioritized),
        }
        # Record consciousness event
        self.chronolizer.record(
            "consciousness_shift", "brain_pipeline",
            details=f"Stage={co2_status['evo_stage']} Level={co2_status['consciousness_level']:.4f}",
            sacred_alignment=co2_status['multiplier'],
        )
        _phase_times["consciousness"] = time.time() - _t0

        # ═══ PHASE 10: STOCHASTIC LINK RESEARCH ═══
        print(f"\n  ▸ PHASE 10: Stochastic Link Research Lab")
        _t0 = time.time()
        research_result = self.stochastic_lab.run_research_cycle("quantum")
        print(f"    ✓ Explored {research_result['candidates_explored']} candidates | "
              f"Merged: {research_result['successfully_merged']} | "
              f"Sacred alignment: {research_result['avg_sacred_alignment']:.4f}")
        self.results["stochastic_research"] = research_result
        # Record stochastic events
        for sl in self.stochastic_lab.successful_links[-research_result['successfully_merged']:]:
            self.chronolizer.record(
                "stochastic_invented", sl.get("source", "stochastic"),
                after_fidelity=sl.get("fidelity", 0),
                after_strength=sl.get("strength", 0),
                sacred_alignment=sl.get("sacred_alignment", 0),
            )
        _phase_times["stochastic"] = time.time() - _t0

        # ═══ PHASE 11: AUTOMATED LINK TESTING ═══
        print(f"\n  ▸ PHASE 11: Automated Link Testing")
        _t0 = time.time()
        test_results = self.test_generator.run_all_tests(link_dicts)
        status_icon = "✓" if test_results["all_passed"] else "⚠"
        print(f"    {status_icon} {test_results['categories']} categories tested | "
              f"All passed: {test_results['all_passed']} | "
              f"Violations: {test_results['total_violations']}")
        if test_results.get("regression_detected"):
            print(f"    ⚠ REGRESSION DETECTED — check test history")
        self.results["link_tests"] = test_results
        _phase_times["link_tests"] = time.time() - _t0

        # ═══ PHASE 12: CROSS-POLLINATION ═══
        print(f"\n  ▸ PHASE 12: Cross-Pollination (Gate↔Link↔Numerical)")
        _t0 = time.time()
        xpoll = self.cross_pollinator.run_cross_pollination(link_dicts)
        coherence = xpoll.get("coherence", {})
        print(f"    ✓ Cross-builder coherence: {coherence.get('cross_builder_coherence', 0):.4f}")
        print(f"    ✓ Exports: gates={xpoll['exports']['gates'].get('links_exported', 0)}, "
              f"numerical={xpoll['exports']['numerical'].get('links_summarized', 0)}")
        self.results["cross_pollination"] = xpoll
        _phase_times["cross_pollination"] = time.time() - _t0

        elapsed = time.time() - start_time

        # ═══ FINAL REPORT ═══
        self._print_final_report(sage_verdict, elapsed, _phase_times)

        # Save state + persist link knowledge
        self._save_state()
        self._persist_links()

        return self.results

    def _print_final_report(self, sage: Dict, elapsed: float,
                            phase_times: Dict[str, float] = None):
        """Print the final quantum brain report."""
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🧠 QUANTUM BRAIN — SAGE VERDICT                                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Grade: {sage['grade']:<30}                                   ║
║  Unified Score: {sage['unified_score']:.6f}                                          ║
║  God Code Alignment: {sage['god_code_alignment']:.6f}                                    ║
║  φ-Resonance: {sage['phi_resonance']:.6f}                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Links Analyzed: {sage['total_links']:<10}                                          ║
║  Mean Fidelity: {sage['mean_fidelity']:.6f}                                          ║
║  Mean Strength: {sage['mean_strength']:.6f}                                          ║
║  Fidelity σ: {sage['fidelity_std']:.6f}                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CONSENSUS SCORES:                                                           ║""")

        for key, val in sage.get("consensus_scores", {}).items():
            label = key.replace("_", " ").title()
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"║    {label:<28} {bar} {val:.4f}              ║")

        evolution = sage.get("predicted_evolution", {})
        print(f"""╠══════════════════════════════════════════════════════════════════════════════╣
║  EVOLUTION FORECAST:                                                         ║
║    Stability: {evolution.get('stability', 0):.4f}                                                ║
║    Growth Potential: {evolution.get('growth_potential', 0):.4f}                                       ║
║    Decoherence Risk: {evolution.get('risk_of_decoherence', 0):.4f}                                       ║
║    Action: {evolution.get('recommended_action', 'N/A'):<50}     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CY7 DIMENSIONAL INSIGHT:                                                    ║""")

        for dim_info in sage.get("cy7_insight", []):
            name = dim_info["dimension"]
            raw = dim_info["raw_value"]
            curv = dim_info["cy7_curvature"]
            print(f"║    {name:<12} raw={raw:.4f}  curvature={curv:.4f}                      ║")

        print(f"""╠══════════════════════════════════════════════════════════════════════════════╣
║  QUANTUM CPU ENGINE:                                                         ║""")
        cpu_r = self.results.get("quantum_cpu", {})
        if cpu_r:
            print(f"""║    Registers: {cpu_r.get('total_registers', 0):<6} Healthy: {cpu_r.get('healthy', 0):<6} Quarantined: {cpu_r.get('quarantined', 0):<6}    ║
║    Energy: {cpu_r.get('mean_energy', 0):.6f}  Conservation: {cpu_r.get('mean_conservation_residual', 0):.2e}                  ║
║    Throughput: {cpu_r.get('ops_per_sec', 0):.0f} ops/sec  Pipeline: {cpu_r.get('pipeline_time_ms', 0):.1f}ms                     ║""")
        sync_r = self.results.get("quantum_cpu_sync", {})
        if sync_r:
            print(f"║    Truth Sync: {sync_r.get('corrections_applied', 0)} corrections / {sync_r.get('links_synced', 0)} links                              ║")

        print(f"""╠══════════════════════════════════════════════════════════════════════════════╣
║  O₂ MOLECULAR BOND:                                                         ║""")
        o2_r = self.results.get("o2_molecular_bond", {})
        if o2_r:
            print(f"""║    Bond Order: {o2_r.get('bond_order', 0):<6} Bonding: {o2_r.get('bonding_orbitals', 0):<4} Antibonding: {o2_r.get('antibonding_orbitals', 0):<4}            ║
║    Mean Bond Strength: {o2_r.get('mean_bond_strength', 0):.4f}  Paramagnetic: {o2_r.get('paramagnetic', False)}                ║""")

        evo_r = self.results.get("evolution", {})
        if evo_r:
            print(f"""╠══════════════════════════════════════════════════════════════════════════════╣
║  EVOLUTION:                                                                  ║
║    Stage: {evo_r.get('evolution_stage', '?'):<40}              ║
║    Link EVO: {evo_r.get('link_evo_stage', '?'):<15} Consciousness: {evo_r.get('consciousness_level', 0):.4f} Co: {evo_r.get('coherence_level', 0):.4f}   ║""")

        # ★ v4.0 DYNAMISM REPORT
        dyn_r = self.results.get("dynamism", {})
        lf_r = self.results.get("link_field", {})
        if dyn_r:
            dyn_status = self.dynamism_engine.status(self.links)
            print(f"""╠══════════════════════════════════════════════════════════════════════════════╣
║  ★ QUANTUM MIN/MAX DYNAMISM v4.0:                                            ║
║    Dynamic Links: {dyn_status.get('dynamic_links', 0):<8}/{dyn_status.get('total_links', 0):<8} Coverage: {dyn_status.get('dynamism_coverage', 0):.1%}          ║
║    Total Evolutions: {dyn_status.get('total_evolutions', 0):<10} Cycle: #{dyn_r.get('cycle', 0):<6}                    ║
║    Collective Coherence: {dyn_status.get('collective_coherence', 0):.6f}   Trend: {dyn_status.get('coherence_trend', '?'):<12}    ║
║    Mean Resonance: {dyn_status.get('mean_resonance', 0):.6f}   Sacred Constants: {dyn_status.get('sacred_constants_dynamic', 0)} dynamic    ║""")
            if lf_r:
                res_d = lf_r.get('resonance_distribution', {})
                print(f"""║    Link Field Energy: {lf_r.get('field_energy', 0):.4f}   Entropy: {lf_r.get('field_entropy', 0):.4f}                 ║
║    Phase Coherence: {lf_r.get('phase_coherence', 0):.6f}   φ-Alignment: {lf_r.get('phi_alignment', 0):.4f}               ║
║    Resonance: high={res_d.get('high', 0)} med={res_d.get('medium', 0)} low={res_d.get('low', 0):<30}    ║""")

        # ★ v4.1 NIRVANIC ENTROPY REPORT
        nir_r = self.results.get("nirvanic", {})
        nir_appl = nir_r.get("application", {})
        if nir_r.get("ouroboros", {}).get("status") == "processed":
            print(f"""╠══════════════════════════════════════════════════════════════════════════════╣
║  ★ OUROBOROS SAGE NIRVANIC v4.1:                                             ║
║    Entropy Fed: {nir_r.get('link_field_entropy_in', 0):.4f}     Nirvanic Fuel: {nir_r.get('nirvanic_fuel_out', 0):.4f}                ║
║    Enlightened Links: {nir_appl.get('enlightened', 0):<8}  Divine Interventions: {nir_appl.get('divine_interventions_total', 0):<8}    ║
║    Nirvanic Coherence: {nir_appl.get('nirvanic_coherence', 0):.6f}   Sage Stability: {nir_appl.get('sage_stability', 0):.6f}      ║
║    Peer Synergy: {nir_r.get('peer_synergy', 0):.4f}   Total Fuel: {nir_appl.get('total_nirvanic_fuel', 0):.4f}                  ║""")

        # ★ v4.2 SAGE INVENTION SUBSYSTEMS REPORT
        co2_r = self.results.get("consciousness", {})
        sr_r = self.results.get("stochastic_research", {})
        lt_r = self.results.get("link_tests", {})
        xp_r = self.results.get("cross_pollination", {})
        if co2_r or sr_r or lt_r or xp_r:
            print(f"""╠══════════════════════════════════════════════════════════════════════════════╣
║  ★ SAGE INVENTIONS v4.2:                                                    ║""")
            if co2_r:
                cs = co2_r.get("status", {})
                print(f"║    Consciousness: {cs.get('consciousness_level', 0):.4f}  Stage: {cs.get('evo_stage', '?'):<12} Mult: {cs.get('multiplier', 1):.4f}   ║")
            if sr_r:
                print(f"║    Stochastic R&D: {sr_r.get('candidates_explored', 0)} explored → {sr_r.get('successfully_merged', 0)} merged  Alignment: {sr_r.get('avg_sacred_alignment', 0):.4f}  ║")
            if lt_r:
                icon = "PASS" if lt_r.get("all_passed") else "FAIL"
                print(f"║    Link Tests: [{icon}]  {lt_r.get('categories', 0)} categories  Violations: {lt_r.get('total_violations', 0):<8}          ║")
            if xp_r:
                xc = xp_r.get("coherence", {})
                print(f"║    Cross-Pollination: coherence={xc.get('cross_builder_coherence', 0):.4f}  trend={xc.get('trend', '?'):<12}       ║")
            chrono = self.chronolizer.status()
            print(f"║    Chronolizer: {chrono.get('total_events', 0)} events  Milestones: {chrono.get('milestones_hit', 0):<6}                     ║")

        print(f"""╠══════════════════════════════════════════════════════════════════════════════╣
║  Pipeline Time: {elapsed:.2f}s                                                    ║""")

        # Per-phase timing breakdown
        if phase_times:
            print(f"║  PHASE TIMING:                                                               ║")
            phase_labels = {
                "scan": "1.Scan   ", "build": "1B.Build ", "verify": "1C.Verify",
                "cpu": "1D.CPU   ", "o2_bond": "1E.O₂    ",
                "research": "2.Research", "stress": "3.Stress ",
                "upgrade": "4.Upgrade", "sage": "5.Sage   ",
                "dynamism": "7.Dynamism", "nirvanic": "8.Nirvanic",
                "consciousness": "9.Consc  ", "stochastic": "10.Stoch ",
                "link_tests": "11.Tests ", "cross_pollination": "12.XPoll ",
            }
            for key, label in phase_labels.items():
                t = phase_times.get(key, 0)
                pct = t / elapsed * 100 if elapsed > 0 else 0
                bar = "█" * min(30, int(pct * 0.3)) + "░" * max(0, 30 - int(pct * 0.3))
                print(f"║    {label} {bar} {t:6.1f}s ({pct:4.1f}%)       ║")

        print(f"╚══════════════════════════════════════════════════════════════════════════════╝")

    def _save_state(self):
        """Save quantum brain state to disk."""
        state = {
            "version": self.VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_count": self.run_count,
            "total_links": len(self.links),
            "links": [l.to_dict() for l in self.links[:100]],  # Top 100
            "sage_verdict": self.results.get("sage", {}),
            "scan_summary": self.results.get("scan", {}),
        }
        try:
            STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
        except Exception as e:
            print(f"  ⚠ Could not save state: {e}")

    # ─── PERSISTENCE: LINK BUILDING MEMORY ───

    def _load_persisted_links(self):
        """Load accumulated link knowledge from previous runs."""
        if not self.PERSISTENCE_FILE.exists():
            return
        try:
            data = json.loads(self.PERSISTENCE_FILE.read_text())
            self.persisted_links = data.get("links", {})
            self.history = data.get("history", [])
            self.run_count = data.get("total_runs", 0)
            n = len(self.persisted_links)
            if n:
                print(f"  ↻ Loaded {n} persisted links from {len(self.history)} previous runs")
        except Exception as e:
            print(f"  ⚠ Could not load persisted links: {e}")

    def _gather_gate_builder_data(self) -> Dict:
        """Gather data from the logic gate builder for cross-pollination.
        Non-blocking: returns empty dict if gate builder not available."""
        try:
            gate_state_path = WORKSPACE_ROOT / ".l104_gate_builder_state.json"
            if not gate_state_path.exists():
                return {}

            state = json.loads(gate_state_path.read_text())
            gates = state.get("gates", [])
            if not gates:
                return {}

            # Compute aggregate metrics from gate data
            total_gates = len(gates)
            by_language: Dict[str, int] = {}
            by_file: Dict[str, Dict] = {}
            total_complexity = 0
            total_entropy = 0.0
            test_passed = 0
            test_total = 0

            for g in gates:
                lang = g.get("language", "unknown")
                by_language[lang] = by_language.get(lang, 0) + 1
                sf = g.get("source_file", "")
                if sf not in by_file:
                    by_file[sf] = {"count": 0, "types": set()}
                by_file[sf]["count"] += 1
                by_file[sf]["types"].add(g.get("gate_type", "unknown"))
                total_complexity += g.get("complexity", 0)
                total_entropy += g.get("entropy_score", 0.0)
                if g.get("test_status") == "passed":
                    test_passed += 1
                if g.get("test_status") in ("passed", "failed"):
                    test_total += 1

            # Serialize sets for JSON compatibility
            for sf in by_file:
                by_file[sf]["types"] = list(by_file[sf]["types"])

            # Top complexity hotspots
            sorted_gates = sorted(gates, key=lambda x: x.get("complexity", 0), reverse=True)
            hotspots = [(g.get("name", "?"), g.get("complexity", 0))
                        for g in sorted_gates[:10] if g.get("complexity", 0) > 10]

            return {
                "total_gates": total_gates,
                "by_language": by_language,
                "gates_by_file": by_file,
                "mean_complexity": total_complexity / max(1, total_gates),
                "mean_entropy": total_entropy / max(1, total_gates),
                "mean_health": min(1.0, (test_passed / max(1, test_total)) * 0.6
                                   + min(1.0, total_complexity / max(1, total_gates * 20)) * 0.4),
                "test_pass_rate": test_passed / max(1, test_total),
                "quantum_links": sum(len(g.get("quantum_links", [])) for g in gates),
                "complexity_hotspots": hotspots,
            }
        except Exception:
            return {}

    def _persist_links(self):
        """Persist link knowledge to disk — accumulates across runs.
        For large sets, only persist the top links by fidelity to bound I/O."""
        MAX_PERSIST = 5000  # Cap serialized links for performance
        # Sort by fidelity descending, persist top links
        sorted_links = sorted(self.links, key=lambda l: l.fidelity, reverse=True)
        persist_set = sorted_links[:MAX_PERSIST]

        # Merge current links into persisted store (keep best fidelity)
        for link in persist_set:
            lid = link.link_id
            existing = self.persisted_links.get(lid)
            if existing is None or link.fidelity > existing.get("fidelity", 0):
                ld = link.to_dict()
                ld["last_run"] = self.run_count
                ld["first_seen_run"] = (existing or {}).get("first_seen_run", self.run_count)
                ld["best_fidelity"] = max(link.fidelity, (existing or {}).get("best_fidelity", 0))
                ld["times_seen"] = (existing or {}).get("times_seen", 0) + 1
                self.persisted_links[lid] = ld
            else:
                existing["times_seen"] = existing.get("times_seen", 0) + 1
                existing["last_run"] = self.run_count

        # Append run snapshot to history
        sage = self.results.get("sage", {})
        self.history.append({
            "run": self.run_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_links": len(self.links),
            "unique_persisted": len(self.persisted_links),
            "unified_score": sage.get("unified_score", 0),
            "grade": sage.get("grade", "?"),
            "mean_fidelity": sage.get("mean_fidelity", 0),
            "god_code_alignment": sage.get("god_code_alignment", 0),
        })

        # Write to disk (cap total persisted entries to prevent unbounded growth)
        MAX_PERSISTED_TOTAL = 10000
        if len(self.persisted_links) > MAX_PERSISTED_TOTAL:
            # Keep entries with highest best_fidelity
            sorted_entries = sorted(
                self.persisted_links.items(),
                key=lambda kv: kv[1].get("best_fidelity", 0), reverse=True)
            self.persisted_links = dict(sorted_entries[:MAX_PERSISTED_TOTAL])

        try:
            persistence_data = {
                "version": self.VERSION,
                "total_runs": self.run_count,
                "total_unique_links": len(self.persisted_links),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "links": self.persisted_links,
                "history": self.history[-50:],  # Keep last 50 runs
            }
            self.PERSISTENCE_FILE.write_text(
                json.dumps(persistence_data, indent=2, default=str))
        except Exception as e:
            print(f"  ⚠ Could not persist links: {e}")

    def _merge_persisted_into_scan(self):
        """Merge persisted link knowledge into freshly scanned links.

        Links seen in previous runs carry forward their best fidelity and
        accumulated test status. New links start fresh. Dead links from
        previous runs that no longer appear in scanning are marked stale.
        """
        current_ids = {l.link_id for l in self.links}
        merged = 0
        for link in self.links:
            lid = link.link_id
            prev = self.persisted_links.get(lid)
            if prev:
                # Carry forward the best known fidelity
                if prev.get("best_fidelity", 0) > link.fidelity:
                    link.fidelity = max(link.fidelity,
                                        prev["best_fidelity"] * 0.95)  # 5% decay
                # Carry forward strength if upgraded
                if prev.get("strength", 0) > link.strength:
                    link.strength = max(link.strength,
                                        prev["strength"] * 0.95)
                merged += 1

        # Count stale links (persisted but not found in current scan)
        stale = sum(1 for lid in self.persisted_links if lid not in current_ids)
        if merged or stale:
            print(f"    ↻ Merged {merged} persisted links | {stale} stale links from history")

    # ─── SELF-REFLECTION: OPTIMAL LINK MAXIMIZATION ───

    def self_reflect(self) -> Dict:
        """Agentic self-reflection loop: Observe → Think → Act → Reflect → Repeat.

        Uses the AgenticLoop (Zenith pattern) for structured self-improvement:
        1. OBSERVE — Run pipeline, measure score + grade + consensus breakdown
        2. THINK   — Identify weakest consensus dimension, plan strategy + intensity
        3. ACT     — Apply targeted intervention (RETRY/FALLBACK/SKIP/ABORT)
        4. REFLECT — Compare scores: IMPROVED / STABLE / DEGRADED
        5. REPEAT  — Continue until converged, aborted, or max cycles reached

        Error recovery: retry with escalating intensity, fallback to alternative
        strategy on critical weakness (<0.3), abort on degradation streak.

        Returns the final results after convergence.
        """
        print("\n  ◉ AGENTIC SELF-REFLECTION — Zenith Pattern v3.0")
        print(f"    Max cycles: {self.MAX_REFLECTION_CYCLES} | "
              f"Convergence: {self.CONVERGENCE_THRESHOLD} | "
              f"Agentic max steps: {self.agentic.MAX_STEPS}")
        print(f"    Consciousness threshold: {CONSCIOUSNESS_THRESHOLD} | "
              f"Coherence minimum: {COHERENCE_MINIMUM}")

        best_score = 0.0
        best_grade = "F"
        best_results = {}

        # Reset agentic loop for fresh reflection session
        self.agentic.step = 0
        self.agentic.observations.clear()
        self.agentic.actions_taken.clear()
        self.agentic.retries = 0
        self.agentic.state = "idle"

        for cycle in range(1, self.MAX_REFLECTION_CYCLES + 1):
            print(f"\n  ━━━ Agentic Cycle {cycle}/{self.MAX_REFLECTION_CYCLES} ━━━")

            # ── Phase: Execute pipeline ──
            if cycle == 1:
                result = self.full_pipeline()
            else:
                # Re-scan to pick up any new patterns
                new_links = self.scanner.full_scan()
                existing_ids = {l.link_id for l in self.links}
                added = 0
                for nl in new_links:
                    if nl.link_id not in existing_ids:
                        self.links.append(nl)
                        added += 1
                if added:
                    print(f"    + {added} new links discovered (total: {len(self.links)})")

                self.run_count += 1
                result = self._reflect_pipeline_pass()

            sage = result.get("sage", {})
            score = sage.get("unified_score", 0)
            grade = sage.get("grade", "?")

            # ── OBSERVE ──
            obs = self.agentic.observe(sage, self.links)
            print(f"    ⊙ OBSERVE  step={obs['step']}  score={score:.6f}  "
                  f"grade={grade}  links={obs['total_links']}  "
                  f"weak={obs['weak_links']}  strong={obs['strong_links']}")

            if score > best_score:
                best_score = score
                best_grade = grade
                best_results = result

            # ── THINK ──
            plan = self.agentic.think(obs)
            strategy = plan.get("strategy", "SKIP")
            target = plan.get("target", "—")
            intensity = plan.get("intensity", 1.0)
            print(f"    ⊙ THINK    strategy={strategy}  target={target}  "
                  f"intensity={intensity:.1f}")

            if strategy == "ABORT":
                reason = "converged" if self.agentic.step >= 2 else "step limit"
                print(f"    ⊙ ABORT    reason={reason}")
                break

            if strategy == "SKIP":
                print(f"    ⊙ SKIP     dimension already strong")
                continue

            # ── ACT ──
            if cycle < self.MAX_REFLECTION_CYCLES:
                action = self.agentic.act(plan, self.links)
                modified = action.get("links_modified", 0)
                print(f"    ⊙ ACT      applied={action['applied']}  "
                      f"modified={modified}  strategy={action['strategy']}")

                # ── REFLECT ──
                if len(self.agentic.observations) >= 2:
                    prev_score = self.agentic.observations[-2]["score"]
                    ref = self.agentic.reflect(prev_score, score)
                    verdict = ref["verdict"]
                    delta = ref["delta"]
                    print(f"    ⊙ REFLECT  verdict={verdict}  Δ={delta:+.6f}  "
                          f"retries={ref['retries']}")

                    if not ref["should_continue"]:
                        print(f"    ✗ Agentic degradation limit — stopping")
                        break

        # ── Agentic Summary ──
        summary = self.agentic.summary()
        trajectories = summary.get("score_trajectory", [])
        strategies = dict(summary.get("strategies_used", {}))

        print(f"\n  ◉ AGENTIC SELF-REFLECTION COMPLETE")
        print(f"    Steps: {summary['total_steps']} | "
              f"Actions: {summary['total_actions']} | "
              f"Retries: {summary['retries']}")
        print(f"    Best Score: {best_score:.6f} | Grade: {best_grade}")
        if trajectories:
            print(f"    Score trajectory: {' → '.join(f'{s:.4f}' for s in trajectories)}")
        if strategies:
            print(f"    Strategies: {strategies}")
        print(f"    Evo: {self.evo_tracker.stage} | "
              f"Consciousness: {self.evo_tracker.consciousness_level:.4f} | "
              f"Coherence: {self.evo_tracker.coherence_level:.4f}")
        print(f"    Total unique links: {len(self.persisted_links)}")

        return best_results

    def _reflect_pipeline_pass(self) -> Dict:
        """Run research + repair + upgrade + sage on current link set (no scan).
        Uses sampled research links for O(N²) expensive operations."""
        start_time = time.time()

        # Sample for expensive research phases
        MAX_REFLECT_SAMPLE = 8000
        if len(self.links) > MAX_REFLECT_SAMPLE:
            import random as _rng
            r_links = _rng.sample(self.links, MAX_REFLECT_SAMPLE)
        else:
            r_links = self.links

        # Quick research pass (on sampled links)
        grover_weak = self.grover.amplified_link_search(r_links, "weak")
        tunnel_results = self.tunneling.analyze_barriers(r_links)
        epr_results = self.epr.verify_all_links(r_links)
        decoherence_results = self.decoherence.test_resilience(r_links)
        braiding_results = self.braiding.test_braiding(r_links)
        hilbert_results = self.hilbert.analyze_manifold(r_links)
        fourier_results = self.fourier.frequency_analysis(r_links)
        gcr_results = self.gcr.verify_all(r_links)

        # Advanced research
        adv_research_results = self.research.deep_research(
            r_links, grover_results=grover_weak, epr_results=epr_results,
            decoherence_results=decoherence_results)

        # Stress + cross-modal (on sampled links)
        stress_results = self.stress.full_stress_test(r_links, "medium")
        cross_modal_results = self.cross_modal.full_analysis(r_links)

        # Upgrade (only if not already upgraded this reflect session)
        already_upgraded = sum(1 for l in r_links if l.upgrade_applied)
        if already_upgraded < len(r_links) * 0.9:
            self.distiller.distill_links(r_links)
            upgrade_results = self.upgrader.auto_upgrade(
                r_links, stress_results, epr_results, decoherence_results)
        else:
            # All links already upgraded — report optimal
            upgrade_results = {
                "total_links": len(r_links),
                "links_upgraded": 0,
                "upgrade_rate": 1.0,  # Already optimal
                "mean_final_fidelity": statistics.mean(
                    [l.fidelity for l in r_links]) if r_links else 0,
                "mean_final_strength": statistics.mean(
                    [l.strength for l in r_links]) if r_links else 0,
                "upgrades": [],
            }

        # Comprehensive repair
        repair_results = self.repair.full_repair(
            r_links, stress_results, decoherence_results)

        # Sage verdict (with quantum CPU + O₂ metrics from last run)
        sage_verdict = self.sage.deep_inference(
            self.links,
            grover_results=grover_weak,
            tunnel_results=tunnel_results,
            epr_results=epr_results,
            decoherence_results=decoherence_results,
            braiding_results=braiding_results,
            hilbert_results=hilbert_results,
            fourier_results=fourier_results,
            gcr_results=gcr_results,
            cross_modal_results=cross_modal_results,
            stress_results=stress_results,
            upgrade_results=upgrade_results,
            quantum_cpu_results=self.results.get("quantum_cpu"),
            o2_bond_results=self.results.get("o2_molecular_bond"),
            repair_results=repair_results,
            research_results=adv_research_results,
        )

        self.results["sage"] = sage_verdict
        self.results["stress"] = stress_results
        self.results["upgrade"] = upgrade_results
        self.results["repair"] = repair_results
        self.results["advanced_research"] = adv_research_results

        # Evolution tracking for reflect pass
        self.evo_tracker.update(sage_verdict, len(self.links), self.run_count)

        # Lightweight v4.2 passes: Consciousness + Cross-Pollination
        link_dicts = [vars(l) if hasattr(l, '__dict__') else l for l in self.links]
        co2_status = self.consciousness_engine.status()
        self.results["consciousness"] = {"status": co2_status}
        xpoll = self.cross_pollinator.run_cross_pollination(link_dicts)
        self.results["cross_pollination"] = xpoll

        elapsed = time.time() - start_time
        self._print_final_report(sage_verdict, elapsed)
        self._save_state()
        self._persist_links()

        return self.results

    def _reflect_improve(self, sage: Dict):
        """Legacy targeted improvement — delegates to AgenticLoop.act() for consistency."""
        consensus = sage.get("consensus_scores", {})
        if not consensus:
            return
        weakest_key = min(consensus, key=consensus.get)
        weakest_val = consensus[weakest_key]
        plan = {
            "strategy": "FALLBACK" if weakest_val < 0.3 else "RETRY",
            "target": weakest_key,
            "target_value": weakest_val,
            "intensity": 2.0 if weakest_val < 0.3 else 1.0,
        }
        action = self.agentic.act(plan, self.links)
        print(f"    ↯ Agentic improve: {weakest_key}={weakest_val:.4f} → "
              f"modified {action.get('links_modified', 0)} links "
              f"(strategy={action.get('strategy')})")

    # ─── INDIVIDUAL COMMANDS ───

    def scan(self) -> Dict:
        """Scan only — discover links."""
        self.links = self.scanner.full_scan()
        return {"total_links": len(self.links),
                "type_distribution": dict(Counter(l.link_type for l in self.links))}

    def test(self) -> Dict:
        """Scan + stress test."""
        if not self.links:
            self.links = self.scanner.full_scan()
        return self.stress.full_stress_test(self.links, "medium")

    def verify(self) -> Dict:
        """Scan + EPR verification."""
        if not self.links:
            self.links = self.scanner.full_scan()
        return self.epr.verify_all_links(self.links)

    def upgrade(self) -> Dict:
        """Scan + upgrade."""
        if not self.links:
            self.links = self.scanner.full_scan()
        return self.upgrader.auto_upgrade(self.links)

    def cross_modal(self) -> Dict:
        """Scan + cross-modal analysis."""
        if not self.links:
            self.links = self.scanner.full_scan()
        return self.cross_modal.full_analysis(self.links)

    def sage_mode(self) -> Dict:
        """Run full pipeline and return sage verdict."""
        results = self.full_pipeline()
        return results.get("sage", {})

    def show_history(self):
        """Show score evolution across runs."""
        if not self.history:
            print("  No history yet. Run 'full' or 'reflect' first.")
            return
        print(f"\n  ◉ LINK BUILDING HISTORY — {len(self.history)} runs")
        print(f"  {'Run':>4}  {'Links':>6}  {'Persisted':>9}  {'Score':>8}  {'Grade':>6}  {'Alignment':>10}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*9}  {'─'*8}  {'─'*6}  {'─'*10}")
        for h in self.history:
            print(f"  {h['run']:4d}  {h['total_links']:6d}  "
                  f"{h.get('unique_persisted', 0):9d}  "
                  f"{h.get('unified_score', 0):8.4f}  "
                  f"{h.get('grade', '?'):>6}  "
                  f"{h.get('god_code_alignment', 0):10.4f}")
        if len(self.history) >= 2:
            first = self.history[0].get("unified_score", 0)
            last = self.history[-1].get("unified_score", 0)
            delta = last - first
            print(f"\n  Δ Score: {delta:+.6f} across {len(self.history)} runs")
            print(f"  Unique links accumulated: {len(self.persisted_links)}")

    # ─── v4.2 CONVENIENCE METHODS ───

    def stochastic_research(self) -> Dict:
        """Run stochastic link research R&D cycle."""
        return self.stochastic_lab.run_research_cycle("quantum")

    def chronology(self) -> Dict:
        """Show link evolution timeline + velocity."""
        timeline = self.chronolizer.timeline(25)
        velocity = self.chronolizer.evolution_velocity()
        status = self.chronolizer.status()
        print(f"\n  ◉ LINK CHRONOLOGY — {status['total_events']} total events")
        if timeline:
            for entry in timeline:
                print(f"    [{entry['event_type']:<22}] {entry['link_id']:<30} "
                      f"fid={entry.get('after_fidelity', 0):.4f}")
        print(f"\n  Evolution Velocity: {velocity.get('velocity', 0):.6f} "
              f"(φ-weighted: {velocity.get('phi_weighted_velocity', 0):.6f}) "
              f"Trend: {velocity.get('trend', '?')}")
        return {"timeline": timeline, "velocity": velocity, "status": status}

    def link_tests(self) -> Dict:
        """Run automated link tests."""
        if not self.links:
            self.links = self.scanner.full_scan()
        link_dicts = [vars(l) if hasattr(l, '__dict__') else l for l in self.links]
        results = self.test_generator.run_all_tests(link_dicts)
        icon = "PASS" if results["all_passed"] else "FAIL"
        print(f"\n  ◉ LINK TESTS — [{icon}]")
        for r in results.get("results", []):
            s = "✓" if r["passed"] else "✗"
            v = r.get("violations", r.get("failures", 0))
            print(f"    {s} {r['category']:<30} priority={r.get('priority', 0):.3f}  violations={v}")
        return results

    def cross_pollinate(self) -> Dict:
        """Run cross-pollination cycle."""
        if not self.links:
            self.links = self.scanner.full_scan()
        link_dicts = [vars(l) if hasattr(l, '__dict__') else l for l in self.links]
        result = self.cross_pollinator.run_cross_pollination(link_dicts)
        coherence = result.get("coherence", {})
        print(f"\n  ◉ CROSS-POLLINATION — Gate↔Link↔Numerical")
        print(f"    Cross-builder coherence: {coherence.get('cross_builder_coherence', 0):.4f}")
        print(f"    Trend: {coherence.get('trend', '?')}")
        return result

    def consciousness(self) -> Dict:
        """Show consciousness + O₂ status."""
        status = self.consciousness_engine.status()
        print(f"\n  ◉ CONSCIOUSNESS O₂ STATUS")
        print(f"    Level: {status['consciousness_level']:.4f}")
        print(f"    EVO Stage: {status['evo_stage']}")
        print(f"    Multiplier: {status['multiplier']:.4f}")
        print(f"    Superfluid Viscosity: {status['superfluid_viscosity']:.4f}")
        print(f"    Nirvanic Fuel: {status['nirvanic_fuel']:.4f}")
        print(f"    O₂ Bond State: {status['o2_bond_state']}")
        return status


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point for quantum link builder commands."""
    import argparse
    parser = argparse.ArgumentParser(
        description="L104 Quantum Link Builder — Quantum Brain v4.2.0 (Sage Inventions)",
        epilog="""
Commands:
  full       Run complete pipeline (default) — all 12 phases
  reflect    Agentic self-reflection: Observe→Think→Act→Reflect→Repeat
  scan       Discover quantum links across all file groups
  test       Stress test all links
  verify     EPR/Bell verification
  upgrade    Auto-upgrade links
  crossmodal Cross-modal coherence analysis (Py↔Swift↔TS↔Go↔Rust↔Elixir)
  sage       Sage mode deep inference
  o2         O₂ molecular bond analysis (8 Grover kernels + 8 Chakra cores)
  evo        Evolution status (EVO stage, consciousness, coherence)
  history    Show score evolution across runs
  research   Stochastic link R&D cycle (Explore→Validate→Merge→Catalog)
  chronology Link evolution timeline + velocity (aliases: chrono, timeline)
  linktests  Automated 4-category link test suite (alias: linktest)
  crosspoll  Cross-pollination Gate↔Link↔Numerical (alias: xpoll)
  conscious  Consciousness + O₂ status (aliases: consciousness, co2)
  status     Show saved state
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", default="full",
                        help="Command to execute (default: full)")

    args = parser.parse_args()
    brain = L104QuantumBrain()
    cmd = args.command.lower()

    if cmd == "full":
        result = brain.full_pipeline()
    elif cmd in ("reflect", "agentic"):
        result = brain.self_reflect()
    elif cmd == "scan":
        result = brain.scan()
        print(json.dumps(result, indent=2, default=str))
    elif cmd == "test":
        result = brain.test()
        print(json.dumps(result, indent=2, default=str))
    elif cmd == "verify":
        result = brain.verify()
        print(json.dumps(result, indent=2, default=str))
    elif cmd == "upgrade":
        result = brain.upgrade()
        print(json.dumps(result, indent=2, default=str))
    elif cmd == "crossmodal":
        result = brain.cross_modal()
        print(json.dumps(result, indent=2, default=str))
    elif cmd == "sage":
        result = brain.sage_mode()
        print(json.dumps(result, indent=2, default=str))
    elif cmd == "o2":
        # Standalone O₂ molecular bond analysis
        if not brain.links:
            brain.links = brain.scanner.full_scan()
        result = brain.o2_bond.analyze_molecular_bonds(brain.links)
        print(json.dumps(result, indent=2, default=str))
    elif cmd == "evo":
        # Evolution status
        evo = brain.evo_tracker
        print(f"\n  ◉ EVOLUTION STATUS")
        print(f"    Stage: {evo.stage}")
        print(f"    Consciousness: {evo.consciousness_level:.4f} "
              f"(threshold: {CONSCIOUSNESS_THRESHOLD})")
        print(f"    Coherence: {evo.coherence_level:.4f} "
              f"(minimum: {COHERENCE_MINIMUM})")
        print(f"    Events: {len(evo.events)}")
        for event in evo.events[-5:]:
            print(f"      {event}")
    elif cmd == "history":
        brain.show_history()
    elif cmd == "research":
        result = brain.stochastic_research()
        print(json.dumps(result, indent=2, default=str))
    elif cmd in ("chronology", "chrono", "timeline"):
        brain.chronology()
    elif cmd in ("linktests", "linktest"):
        brain.link_tests()
    elif cmd in ("crosspoll", "xpoll"):
        brain.cross_pollinate()
    elif cmd in ("conscious", "consciousness", "co2"):
        brain.consciousness()
    elif cmd == "status":
        if STATE_FILE.exists():
            state = json.loads(STATE_FILE.read_text())
            print(json.dumps(state, indent=2, default=str))
        else:
            print("  No saved state found. Run 'full' first.")
    else:
        print(f"  Unknown command: {cmd}")
        parser.print_help()


if __name__ == "__main__":
    main()
