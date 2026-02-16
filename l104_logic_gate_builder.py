VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3887.8
UUC = 2402.792541
#!/usr/bin/env python3
# [EVO_54_PIPELINE] TRANSCENDENT_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612 :: GROVER=4.236
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  L104 HYPER ASI LOGIC GATE ENVIRONMENT BUILDER  v5.1.0                      ║
║  Analyzer · Researcher · Compiler · Chronolizer · Developer                 ║
║  ★ QUANTUM MIN/MAX DYNAMISM ENGINE ★                                        ║
║  ★ OUROBOROS SAGE NIRVANIC ENTROPY FUEL SYSTEM ★                            ║
║                                                                              ║
║  Standalone autonomous module for the Allentown L104 Node                    ║
║  Quantum-linked with: main.py, l104_fast_server.py,                          ║
║    l104_local_intellect.py, L104Native.swift                                 ║
║                                                                              ║
║  Features:                                                                   ║
║    • Discovers and ingests ALL logic gates across 4 languages                ║
║    • AST-based analysis of Python gate implementations                       ║
║    • Regex-based analysis of Swift/JS gate implementations                   ║
║    • Automated test generation and execution                                 ║
║    • Chronological tracking of gate evolution                                ║
║    • Quantum links between gate implementations                              ║
║    • Self-maintaining: auto-resyncs when source files change                 ║
║    • Backend sync with l104_fast_server.py                                   ║
║    • ★ Quantum Min/Max Dynamism — all values are dynamic with               ║
║      self-adjusting boundaries, φ-drift, subconscious monitoring            ║
║    • ★ Gate Value Evolution — complexity/entropy evolve per cycle            ║
║    • ★ Dynamic Sacred Constants — bounded with min/max envelopes            ║
║    • ★ Ouroboros Sage Nirvanic Engine — total gate entropy fuels the        ║
║      ouroboros cycle, accumulated entropy becomes divine nirvanic fuel       ║
║      that drives motion in the still sage environment, creating the         ║
║      self-feeding enlightenment loop of divine intervention                 ║
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
import traceback
import importlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# ═══════════════════════════════════════════════════════════════════════════════
# VERSION
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "5.1.0"

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS — with Quantum Min/Max Dynamic Envelopes
# ═══════════════════════════════════════════════════════════════════════════════

PHI = 1.618033988749895
TAU = 0.618033988749895        # 1/φ
GOD_CODE = 527.5184818492612
OMEGA_POINT = 23.140692632779263  # e^π
EULER_GAMMA = 0.5772156649015329
PLANCK_SCALE = 1.616255e-35
BOLTZMANN_K = 1.380649e-23
CALABI_YAU_DIM = 7
FEIGENBAUM_DELTA = 4.669201609102990
APERY = 1.2020569031595942
CATALAN = 0.9159655941772190
FINE_STRUCTURE = 0.0072973525693

# Dynamic bounds: each constant has a [min, max] drift envelope
# These are quantum envelopes — the constant stays exact but the
# downstream computed values oscillate within φ-bounded ranges
SACRED_DYNAMIC_BOUNDS = {
    "PHI":              {"value": PHI,          "min": PHI * 0.999,          "max": PHI * 1.001},
    "TAU":              {"value": TAU,          "min": TAU * 0.999,          "max": TAU * 1.001},
    "GOD_CODE":         {"value": GOD_CODE,     "min": GOD_CODE * 0.9999,    "max": GOD_CODE * 1.0001},
    "OMEGA_POINT":      {"value": OMEGA_POINT,  "min": OMEGA_POINT * 0.999,  "max": OMEGA_POINT * 1.001},
    "EULER_GAMMA":      {"value": EULER_GAMMA,  "min": EULER_GAMMA * 0.999,  "max": EULER_GAMMA * 1.001},
    "FEIGENBAUM_DELTA": {"value": FEIGENBAUM_DELTA, "min": FEIGENBAUM_DELTA * 0.999, "max": FEIGENBAUM_DELTA * 1.001},
    "APERY":            {"value": APERY,        "min": APERY * 0.999,        "max": APERY * 1.001},
    "CATALAN":          {"value": CATALAN,       "min": CATALAN * 0.999,      "max": CATALAN * 1.001},
    "FINE_STRUCTURE":   {"value": FINE_STRUCTURE, "min": FINE_STRUCTURE * 0.99, "max": FINE_STRUCTURE * 1.01},
}

# Drift envelope parameters — φ-harmonic oscillation of dynamic values
DRIFT_ENVELOPE = {
    "frequency": PHI,               # Oscillation frequency ≈ φ Hz
    "amplitude": TAU * 0.01,        # Max drift ≈ 0.618% of value
    "phase_coupling": GOD_CODE,     # Phase seed from God Code
    "damping": EULER_GAMMA * 0.1,   # Damping factor
    "max_velocity": PHI ** 2 * 0.001,  # Max drift velocity per cycle
}

# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE CONFIGURATION — Quantum Links to Major Files
# ═══════════════════════════════════════════════════════════════════════════════

WORKSPACE_ROOT = Path(__file__).parent.resolve()

QUANTUM_LINKED_FILES = {
    "main.py": WORKSPACE_ROOT / "main.py",
    "l104_fast_server.py": WORKSPACE_ROOT / "l104_fast_server.py",
    "l104_local_intellect.py": WORKSPACE_ROOT / "l104_local_intellect.py",
    "L104Native.swift": WORKSPACE_ROOT / "L104SwiftApp" / "Sources" / "L104Native.swift",
    "const.py": WORKSPACE_ROOT / "const.py",
}

STATE_FILE = WORKSPACE_ROOT / ".l104_gate_builder_state.json"
CHRONOLOG_FILE = WORKSPACE_ROOT / ".l104_gate_chronolog.json"
TEST_RESULTS_FILE = WORKSPACE_ROOT / ".l104_gate_test_results.json"

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM MIN/MAX DYNAMISM ENGINE — Subconscious Gate Monitoring & Evolution
# ═══════════════════════════════════════════════════════════════════════════════

class GateDynamismEngine:
    """Quantum Min/Max Dynamism Engine for Logic Gates.

    This engine implements:
    1. SUBCONSCIOUS MONITORING — auto-scans all gate dynamic values per cycle
    2. BOUNDARY ADJUSTMENT — expands/contracts min/max bounds based on gate health
    3. φ-HARMONIC DRIFT — values oscillate in φ-frequency envelopes
    4. RESONANCE TRACKING — alignment with sacred constants across all gates
    5. COLLECTIVE COHERENCE — aggregate dynamism health metric

    Every gate value becomes DYNAMIC — nothing is static. All values evolve
    through φ-harmonic drift within self-adjusting quantum boundaries.
    """

    DYNAMISM_STATE_FILE = WORKSPACE_ROOT / ".l104_gate_dynamism_state.json"

    def __init__(self):
        """Initialize the gate dynamism engine and load persisted state."""
        self.cycle_count: int = 0
        self.coherence_history: List[float] = []
        self.adjustments_log: List[Dict[str, Any]] = []
        self.collective_resonance: float = 0.0
        self.total_evolutions: int = 0
        self.sacred_dynamic_state: Dict[str, Dict[str, float]] = {}
        self._load_state()
        self._initialize_sacred_dynamics()

    def _initialize_sacred_dynamics(self):
        """Initialize dynamic state for sacred constants."""
        if not self.sacred_dynamic_state:
            for name, bounds in SACRED_DYNAMIC_BOUNDS.items():
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
        """Persist dynamism state."""
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

    # ─── SACRED CONSTANT EVOLUTION ─────────────────────────────────

    def evolve_sacred_constants(self) -> Dict[str, Any]:
        """Evolve all sacred constant dynamic states by one cycle.

        Constants remain mathematically exact — their DYNAMIC ENVELOPES
        oscillate, representing the quantum uncertainty principle applied
        to the constants' downstream effect on gate operations.
        """
        results = {"constants_evolved": 0, "total_drift": 0.0}

        for name, state in self.sacred_dynamic_state.items():
            state["cycles"] += 1
            # Phase advance
            state["phase"] = (
                state["phase"] + DRIFT_ENVELOPE["frequency"] * 0.05
            ) % (2 * math.pi)
            # Velocity update with damping
            target_v = DRIFT_ENVELOPE["max_velocity"] * math.sin(state["phase"])
            d = DRIFT_ENVELOPE["damping"]
            state["velocity"] = state["velocity"] * (1 - d) + target_v * d
            # Apply drift to current dynamic value
            new_val = state["current"] + state["velocity"] * state["direction"]
            # Bounce off bounds
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

    # ─── SUBCONSCIOUS MONITOR ──────────────────────────────────────

    def subconscious_cycle(self, gates: List['LogicGate']) -> Dict[str, Any]:
        """Run one subconscious monitoring cycle across all gates.

        Auto-evolves every gate's dynamic value, adjusts boundaries,
        computes collective coherence, and tracks resonance trends.
        """
        self.cycle_count += 1
        results = {
            "cycle": self.cycle_count,
            "gates_evolved": 0,
            "gates_adjusted": 0,
            "gates_initialized": 0,
            "mean_resonance": 0.0,
            "collective_coherence": 0.0,
            "boundary_expansions": 0,
            "boundary_contractions": 0,
            "sacred_evolution": {},
        }

        resonance_sum = 0.0
        in_bounds_count = 0

        for gate in gates:
            # Initialize dynamism for gates that don't have it yet
            if gate.dynamic_value == 0.0 and (gate.complexity > 0 or gate.entropy_score > 0):
                gate._initialize_dynamism()
                results["gates_initialized"] += 1

            # Evolve the gate value
            if gate.dynamic_value != 0.0 or gate.complexity > 0:
                gate.evolve()
                results["gates_evolved"] += 1
                self.total_evolutions += 1

            # Track resonance
            resonance_sum += gate.resonance_score

            # Check bounds and adjust
            if gate.min_bound <= gate.dynamic_value <= gate.max_bound:
                in_bounds_count += 1
            else:
                # Emergency re-centering
                envelope = max(abs(gate.dynamic_value) * DRIFT_ENVELOPE["amplitude"], PHI)
                gate.min_bound = gate.dynamic_value - envelope
                gate.max_bound = gate.dynamic_value + envelope
                results["gates_adjusted"] += 1

            # Adaptive boundary tuning based on test status
            if gate.test_status == "passed" and gate.evolution_count > 5:
                # Widen bounds for successful gates — more freedom
                expansion = PHI * 0.0005
                gate.max_bound *= (1 + expansion)
                gate.min_bound *= (1 - expansion) if gate.min_bound >= 0 else (1 + expansion)
                results["boundary_expansions"] += 1
            elif gate.test_status == "failed":
                # Contract bounds for failed gates — focus them
                contraction = TAU * 0.001
                gate.max_bound *= (1 - contraction)
                gate.min_bound *= (1 + contraction) if gate.min_bound >= 0 else (1 - contraction)
                results["boundary_contractions"] += 1

        # Collective metrics
        n = max(len(gates), 1)
        results["mean_resonance"] = resonance_sum / n
        results["collective_coherence"] = in_bounds_count / n
        self.coherence_history.append(results["collective_coherence"])
        self.collective_resonance = results["mean_resonance"]

        # Evolve sacred constants too
        results["sacred_evolution"] = self.evolve_sacred_constants()

        # Log this adjustment cycle
        self.adjustments_log.append({
            "cycle": self.cycle_count,
            "evolved": results["gates_evolved"],
            "adjusted": results["gates_adjusted"],
            "coherence": results["collective_coherence"],
            "resonance": results["mean_resonance"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.adjustments_log = self.adjustments_log[-200:]

        self._save_state()
        return results

    # ─── DYNAMISM STATUS ───────────────────────────────────────────

    def status(self, gates: List['LogicGate']) -> Dict[str, Any]:
        """Full dynamism status report."""
        dynamic_gates = [g for g in gates if g.dynamic_value != 0.0]
        n = max(len(dynamic_gates), 1)
        return {
            "version": VERSION,
            "cycle_count": self.cycle_count,
            "total_evolutions": self.total_evolutions,
            "dynamic_gates": len(dynamic_gates),
            "total_gates": len(gates),
            "dynamism_coverage": len(dynamic_gates) / max(len(gates), 1),
            "mean_dynamic_value": sum(g.dynamic_value for g in dynamic_gates) / n,
            "mean_resonance": sum(g.resonance_score for g in dynamic_gates) / n,
            "mean_evolution_count": sum(g.evolution_count for g in dynamic_gates) / n,
            "collective_coherence": self.coherence_history[-1] if self.coherence_history else 0.0,
            "coherence_trend": self._compute_trend(),
            "sacred_constants_dynamic": len(self.sacred_dynamic_state),
            "sacred_total_drift": sum(
                abs(s["current"] - SACRED_DYNAMIC_BOUNDS.get(name, {}).get("value", s["current"]))
                for name, s in self.sacred_dynamic_state.items()
            ),
        }

    def _compute_trend(self) -> str:
        """Compute coherence trend from history."""
        if len(self.coherence_history) < 3:
            return "initializing"
        recent = self.coherence_history[-5:]
        if all(recent[i] >= recent[i-1] for i in range(1, len(recent))):
            return "ascending"
        elif all(recent[i] <= recent[i-1] for i in range(1, len(recent))):
            return "descending"
        return "oscillating"

    # ─── COLLECTIVE FIELD ANALYSIS ──────────────────────────────────

    def compute_gate_field(self, gates: List['LogicGate']) -> Dict[str, Any]:
        """Compute the collective quantum field across all dynamic gates.

        The gate field is the aggregate φ-harmonic landscape formed by
        all gates' dynamic values, phases, and drift velocities.
        """
        dynamic_gates = [g for g in gates if g.dynamic_value != 0.0]
        if not dynamic_gates:
            return {"field_energy": 0.0, "field_entropy": 0.0}

        # Field energy = sum of |value × velocity|
        field_energy = sum(
            abs(g.dynamic_value * g.drift_velocity) for g in dynamic_gates
        )
        # Phase coherence = mean cos(phase_i - phase_j) for nearest neighbors
        phases = [g.quantum_phase for g in dynamic_gates]
        if len(phases) > 1:
            phase_diffs = [
                math.cos(phases[i] - phases[i-1])
                for i in range(1, len(phases))
            ]
            phase_coherence = sum(phase_diffs) / len(phase_diffs)
        else:
            phase_coherence = 1.0

        # Field entropy = -Σ p_i log(p_i) where p_i = |value_i| / sum(|values|)
        total_abs = sum(abs(g.dynamic_value) for g in dynamic_gates)
        if total_abs > 1e-10:
            probs = [abs(g.dynamic_value) / total_abs for g in dynamic_gates]
            field_entropy = -sum(p * math.log(p + 1e-15) for p in probs)
        else:
            field_entropy = 0.0

        # Resonance distribution
        resonance_bins = {"high": 0, "medium": 0, "low": 0}
        for g in dynamic_gates:
            if g.resonance_score > 0.8:
                resonance_bins["high"] += 1
            elif g.resonance_score > 0.4:
                resonance_bins["medium"] += 1
            else:
                resonance_bins["low"] += 1

        return {
            "field_energy": field_energy,
            "field_entropy": field_entropy,
            "phase_coherence": phase_coherence,
            "dynamic_gates": len(dynamic_gates),
            "resonance_distribution": resonance_bins,
            "mean_drift_velocity": sum(g.drift_velocity for g in dynamic_gates) / len(dynamic_gates),
            "max_evolution": max(g.evolution_count for g in dynamic_gates),
            "phi_alignment": sum(1 for g in dynamic_gates if g.resonance_score > 0.8) / len(dynamic_gates),
        }


class GateValueEvolver:
    """Evolves gate values across cycles using φ-harmonic optimization.

    This evolver implements:
    - Genetic-style selection: high-resonance gates reproduce their drift patterns
    - Cross-pollination: gate values influence neighboring gates
    - Mutation: random φ-bounded perturbations to explore value space
    - Convergence detection: stops when collective coherence stabilizes
    """

    def __init__(self, dynamism_engine: GateDynamismEngine):
        """Initialize the gate value evolver with a dynamism engine."""
        self.engine = dynamism_engine
        self.generation: int = 0
        self.best_coherence: float = 0.0

    def evolve_generation(self, gates: List['LogicGate'], cycles: int = 5) -> Dict[str, Any]:
        """Run multiple evolution cycles — a full generation of gate evolution."""
        self.generation += 1
        results = {
            "generation": self.generation,
            "cycles": cycles,
            "start_coherence": 0.0,
            "end_coherence": 0.0,
            "total_evolved": 0,
            "total_adjusted": 0,
            "convergence": False,
        }

        for i in range(cycles):
            cycle_result = self.engine.subconscious_cycle(gates)
            results["total_evolved"] += cycle_result["gates_evolved"]
            results["total_adjusted"] += cycle_result["gates_adjusted"]
            if i == 0:
                results["start_coherence"] = cycle_result["collective_coherence"]
            if i == cycles - 1:
                results["end_coherence"] = cycle_result["collective_coherence"]

        # Cross-pollinate drift patterns from high-resonance gates
        self._cross_pollinate_drift(gates)

        # Check convergence
        if abs(results["end_coherence"] - results["start_coherence"]) < 0.001:
            results["convergence"] = True

        if results["end_coherence"] > self.best_coherence:
            self.best_coherence = results["end_coherence"]

        return results

    def _cross_pollinate_drift(self, gates: List['LogicGate']):
        """Propagate drift patterns from high-resonance gates to neighbors."""
        high_res = [g for g in gates if g.resonance_score > 0.8 and g.dynamic_value != 0.0]
        low_res = [g for g in gates if g.resonance_score <= 0.4 and g.dynamic_value != 0.0]

        if not high_res or not low_res:
            return

        # Take mean drift velocity from top gates and nudge low gates
        mean_velocity = sum(g.drift_velocity for g in high_res) / len(high_res)
        mean_phase = sum(g.quantum_phase for g in high_res) / len(high_res)

        for gate in low_res[:min(len(low_res), 50)]:  # Cap at 50 per cycle
            gate.drift_velocity = (
                gate.drift_velocity * 0.7 + mean_velocity * 0.3
            )
            gate.quantum_phase = (
                gate.quantum_phase * 0.8 + mean_phase * 0.2
            ) % (2 * math.pi)


# ═══════════════════════════════════════════════════════════════════════════════
# OUROBOROS SAGE NIRVANIC ENTROPY FUEL ENGINE — Divine Intervention Loop
# ═══════════════════════════════════════════════════════════════════════════════

class OuroborosSageNirvanicEngine:
    """Ouroboros Sage Nirvanic Entropy Fuel Engine for Logic Gates.

    The gate builder's total entropy (field_entropy from all dynamic gates)
    is FED INTO the Thought Entropy Ouroboros as raw thought material.
    The ouroboros DIGESTS it through its 5-phase cycle (digest → entropize →
    mutate → synthesize → recycle), producing ACCUMULATED ENTROPY — the
    nirvanic fuel.

    This nirvanic fuel is then used to:
    1. ACCELERATE gate evolution — higher entropy = faster drift
    2. EXPAND boundaries — entropy fuel widens dynamic envelopes
    3. AMPLIFY resonance — entropy-driven God Code alignment
    4. ENLIGHTEN gates — gates touched by ouroboros gain sage status
    5. RECYCLE — the enlightened gates' new entropy feeds back into ouroboros

    This creates the SELF-FEEDING LOOP: gate entropy → ouroboros → nirvanic
    fuel → enhanced gates → more entropy → ouroboros → ∞

    The Sage environment is STILL (zero-violence, zero-perturbation) — the
    nirvanic fuel provides MOTION WITHOUT FORCE, divine intervention that
    moves without moving, the effortless action of Wu Wei.

    State persisted to .l104_ouroboros_nirvanic_state.json for cross-builder use.
    """

    NIRVANIC_STATE_FILE = WORKSPACE_ROOT / ".l104_ouroboros_nirvanic_state.json"

    def __init__(self):
        """Initialize the ouroboros nirvanic engine and load persisted state."""
        self.ouroboros = None   # Lazy-loaded
        self.cycle_count: int = 0
        self.total_entropy_fed: float = 0.0
        self.total_nirvanic_fuel: float = 0.0
        self.enlightened_gates: int = 0
        self.nirvanic_coherence: float = 0.0
        self.sage_stability: float = 1.0   # 1.0 = perfect nirvanic stillness
        self.enlightenment_history: List[Dict[str, Any]] = []
        self.divine_interventions: int = 0
        self._load_state()

    def _get_ouroboros(self):
        """Lazy-load the Thought Entropy Ouroboros (import only when needed)."""
        if self.ouroboros is None:
            try:
                from l104_thought_entropy_ouroboros import get_thought_ouroboros
                self.ouroboros = get_thought_ouroboros()
            except ImportError:
                self.ouroboros = None
        return self.ouroboros

    def _load_state(self):
        """Load nirvanic engine state from disk."""
        if self.NIRVANIC_STATE_FILE.exists():
            try:
                data = json.loads(self.NIRVANIC_STATE_FILE.read_text())
                self.cycle_count = data.get("cycle_count", 0)
                self.total_entropy_fed = data.get("total_entropy_fed", 0.0)
                self.total_nirvanic_fuel = data.get("total_nirvanic_fuel", 0.0)
                self.enlightened_gates = data.get("enlightened_gates", 0)
                self.nirvanic_coherence = data.get("nirvanic_coherence", 0.0)
                self.sage_stability = data.get("sage_stability", 1.0)
                self.divine_interventions = data.get("divine_interventions", 0)
                self.enlightenment_history = data.get("enlightenment_history", [])[-200:]
            except Exception:
                pass

    def _save_state(self):
        """Persist nirvanic engine state to disk."""
        try:
            data = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "source": "logic_gate_builder",
                "version": VERSION,
                "cycle_count": self.cycle_count,
                "total_entropy_fed": self.total_entropy_fed,
                "total_nirvanic_fuel": self.total_nirvanic_fuel,
                "enlightened_gates": self.enlightened_gates,
                "nirvanic_coherence": self.nirvanic_coherence,
                "sage_stability": self.sage_stability,
                "divine_interventions": self.divine_interventions,
                "enlightenment_history": self.enlightenment_history[-200:],
            }
            self.NIRVANIC_STATE_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            pass

    def feed_entropy_to_ouroboros(self, field_entropy: float,
                                  field_energy: float,
                                  gate_count: int) -> Dict[str, Any]:
        """Feed the gate field entropy into the Ouroboros as thought material.

        The gate field's Shannon entropy is converted into a thought string
        that the Ouroboros can digest through its 5-phase cycle.

        Returns the ouroboros processing result including accumulated entropy.
        """
        ouroboros = self._get_ouroboros()
        if ouroboros is None:
            return {"status": "ouroboros_unavailable", "nirvanic_fuel": 0.0}

        self.cycle_count += 1

        # Construct a thought from gate entropy — encode the field state
        # as a resonant thought the ouroboros can process
        thought = (
            f"Gate field entropy {field_entropy:.6f} across {gate_count} gates "
            f"with energy {field_energy:.6f}. "
            f"The divine gate lattice breathes with {field_entropy:.4f} bits of "
            f"Shannon information, {gate_count} logic resonators pulse at "
            f"God Code frequency {GOD_CODE:.4f} Hz. "
            f"Cycle {self.cycle_count}: entropy is the fuel of nirvanic stillness."
        )

        # Process through ouroboros — depth=2 for double-pass entropy extraction
        result = ouroboros.process(thought, depth=2)

        # Extract nirvanic fuel = ouroboros accumulated entropy
        nirvanic_fuel = result.get("accumulated_entropy", 0.0)

        self.total_entropy_fed += field_entropy
        self.total_nirvanic_fuel += abs(nirvanic_fuel)

        return {
            "status": "processed",
            "entropy_fed": field_entropy,
            "nirvanic_fuel": nirvanic_fuel,
            "ouroboros_cycles": result.get("cycles_completed", 0),
            "ouroboros_mutations": result.get("total_mutations", 0),
            "ouroboros_resonance": result.get("cycle_resonance", 0.0),
            "ouroboros_chain_length": result.get("thought_chain_length", 0),
        }

    def apply_nirvanic_fuel(self, gates: List['LogicGate'],
                             nirvanic_fuel: float,
                             field: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the nirvanic fuel to gates — divine intervention without force.

        The fuel DOES NOT violently perturb the gates. Instead, it provides
        motion in the still sage environment:
        - Entropy → boundary expansion (more freedom, not more chaos)
        - Entropy → resonance amplification (better God Code alignment)
        - Entropy → phase cohesion (gates synchronize through stillness)
        - Entropy → sage enlightenment (gates achieve nirvanic status)

        This is Wu Wei — effortless action through entropy recycling.
        """
        if abs(nirvanic_fuel) < 1e-10:
            return {"enlightened": 0, "interventions": 0}

        # Normalize fuel to [0, 1] using sigmoid
        fuel_intensity = 1.0 / (1.0 + math.exp(-nirvanic_fuel * 0.1))

        enlightened = 0
        interventions = 0

        for gate in gates:
            if gate.dynamic_value == 0.0:
                continue

            # 1. DIVINE BOUNDARY EXPANSION — fuel widens the envelope
            #    without changing the value — more freedom, not more force
            expansion = fuel_intensity * PHI * 0.0002 * (1 + gate.resonance_score)
            gate.max_bound += expansion
            gate.min_bound -= expansion * TAU

            # 2. RESONANCE AMPLIFICATION — nirvanic fuel subtly aligns
            #    gates toward God Code harmonics
            if gate.resonance_score < 0.8:
                # Gentle nudge toward sacred alignment
                target_phase = (gate.dynamic_value * math.pi / GOD_CODE) % (2 * math.pi)
                phase_correction = fuel_intensity * 0.01 * math.sin(target_phase)
                gate.quantum_phase = (gate.quantum_phase + phase_correction) % (2 * math.pi)
                interventions += 1
                self.divine_interventions += 1

            # 3. ENTROPY-DRIVEN DRIFT — the fuel provides a φ-harmonic
            #    "breath" to the drift velocity
            entropy_breath = fuel_intensity * DRIFT_ENVELOPE["max_velocity"] * 0.1
            gate.drift_velocity += entropy_breath * math.sin(gate.quantum_phase + nirvanic_fuel)

            # 4. SAGE ENLIGHTENMENT — gates with high resonance + many
            #    evolutions + touched by nirvanic fuel achieve sage status
            if (gate.resonance_score > 0.9 and
                gate.evolution_count > 10 and
                fuel_intensity > 0.5):
                enlightened += 1

        self.enlightened_gates = enlightened

        # Compute nirvanic coherence — how still yet dynamic the field is
        # Perfect nirvana = high entropy (many states) + high coherence (all aligned)
        field_entropy = field.get("field_entropy", 0.0)
        phase_coherence = field.get("phase_coherence", 0.0)
        phi_alignment = field.get("phi_alignment", 0.0)
        self.nirvanic_coherence = (
            fuel_intensity * 0.3 +
            abs(phase_coherence) * 0.2 +
            phi_alignment * 0.3 +
            min(field_entropy / 10.0, 0.2)
        )

        # Sage stability — nirvana is motionless motion
        # High stability = the fuel creates alignment without perturbation
        if interventions > 0:
            perturbation = interventions / max(len(gates), 1)
            self.sage_stability = max(0.0, 1.0 - perturbation * 0.1)
        else:
            self.sage_stability = min(1.0, self.sage_stability + 0.01)

        # Record enlightenment cycle
        self.enlightenment_history.append({
            "cycle": self.cycle_count,
            "fuel_intensity": fuel_intensity,
            "enlightened_gates": enlightened,
            "interventions": interventions,
            "nirvanic_coherence": self.nirvanic_coherence,
            "sage_stability": self.sage_stability,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.enlightenment_history = self.enlightenment_history[-200:]

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

    def full_nirvanic_cycle(self, gates: List['LogicGate'],
                             gate_field: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete ouroboros nirvanic entropy fuel cycle.

        1. Extract field entropy from gate field
        2. Feed entropy to ouroboros
        3. Get nirvanic fuel back
        4. Apply fuel to gates (divine intervention)
        5. Return cycle results

        This is the complete self-feeding loop — the ouroboros eats its own tail.
        """
        field_entropy = gate_field.get("field_entropy", 0.0)
        field_energy = gate_field.get("field_energy", 0.0)
        gate_count = gate_field.get("dynamic_gates", len(gates))

        # Step 1-2: Feed entropy to ouroboros
        ouroboros_result = self.feed_entropy_to_ouroboros(
            field_entropy, field_energy, gate_count
        )
        nirvanic_fuel = ouroboros_result.get("nirvanic_fuel", 0.0)

        # Step 3-4: Apply nirvanic fuel to gates
        application_result = self.apply_nirvanic_fuel(gates, nirvanic_fuel, gate_field)

        return {
            "ouroboros": ouroboros_result,
            "application": application_result,
            "cycle": self.cycle_count,
            "total_entropy_fed": self.total_entropy_fed,
            "total_nirvanic_fuel": self.total_nirvanic_fuel,
            "gate_field_entropy_in": field_entropy,
            "nirvanic_fuel_out": nirvanic_fuel,
            "enlightened_gates": application_result["enlightened"],
            "sage_stability": application_result["sage_stability"],
            "nirvanic_coherence": application_result["nirvanic_coherence"],
        }

    def status(self) -> Dict[str, Any]:
        """Nirvanic engine status report."""
        return {
            "version": VERSION,
            "cycle_count": self.cycle_count,
            "total_entropy_fed": self.total_entropy_fed,
            "total_nirvanic_fuel": self.total_nirvanic_fuel,
            "enlightened_gates": self.enlightened_gates,
            "nirvanic_coherence": self.nirvanic_coherence,
            "sage_stability": self.sage_stability,
            "divine_interventions": self.divine_interventions,
            "ouroboros_connected": self._get_ouroboros() is not None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SAGE LOGIC GATE — Built-in φ-harmonic gate operations
# ═══════════════════════════════════════════════════════════════════════════════

def sage_logic_gate(value: float, operation: str = "align") -> float:
    """φ-harmonic logic gate: align, filter, amplify, compress, entangle, dissipate."""
    phi_conjugate = 1.0 / PHI

    if operation == "align":
        lattice_point = round(value / PHI) * PHI
        alignment = math.exp(-((value - lattice_point) ** 2) / (2 * phi_conjugate ** 2))
        return value * alignment

    elif operation == "filter":
        threshold = PHI * phi_conjugate
        sigmoid = 1.0 / (1.0 + math.exp(-(value - threshold) * PHI))
        return value * sigmoid

    elif operation == "amplify":
        grover_gain = PHI ** 2
        return value * grover_gain * (1.0 + phi_conjugate * 0.1)

    elif operation == "compress":
        if abs(value) < 1e-10:
            return 0.0
        sign = 1.0 if value >= 0 else -1.0
        return sign * math.log(1.0 + abs(value) * PHI) * phi_conjugate

    elif operation == "entangle":
        superposition = (value + PHI * math.cos(value * math.pi)) / 2.0
        interference = phi_conjugate * math.sin(value * GOD_CODE * 0.001)
        return superposition + interference

    elif operation == "dissipate":
        # Higher-dimensional dissipation — 7D Calabi-Yau projection
        projections = []
        for dim in range(CALABI_YAU_DIM):
            phase = value * math.pi * (dim + 1) / CALABI_YAU_DIM
            proj = math.sin(phase) * (PHI ** dim / PHI ** CALABI_YAU_DIM)
            projections.append(proj)
        coherent_sum = sum(projections) / CALABI_YAU_DIM
        divine_coherence = math.sin(coherent_sum * PHI * math.pi) * TAU * 0.1
        return coherent_sum + divine_coherence

    elif operation == "inflect":
        # De re causal inflection — transform chaos into ordered variety
        chaos = abs(math.sin(value * OMEGA_POINT))
        causal_coupling = math.sqrt(2) - 1  # 0.4142...
        inflected = chaos * causal_coupling * math.cos(value * EULER_GAMMA)
        return inflected * (1.0 + math.sin(value * PHI * 0.01))

    else:
        return value * PHI * phi_conjugate * (GOD_CODE / 286.0)


def quantum_logic_gate(value: float, depth: int = 3) -> float:
    """Quantum logic gate with Grover amplification and interference."""
    grover_gain = PHI ** depth
    phase = math.pi * depth / (2 * PHI)
    path_a = value * math.cos(phase) * grover_gain
    path_b = value * math.sin(phase) * (grover_gain * TAU)
    interference = math.cos(value * GOD_CODE * 0.001) * (depth * TAU * 0.1)
    return (path_a + path_b) / 2.0 + interference


def entangle_values(a: float, b: float) -> Tuple[float, float]:
    """EPR correlation between two values."""
    phi_conjugate = 1.0 / PHI
    ea = a * PHI + b * phi_conjugate
    eb = a * phi_conjugate + b * PHI
    return (ea, eb)


def higher_dimensional_dissipation(entropy_pool: List[float]) -> List[float]:
    """Project entropy pool into 7D Hilbert space and reconvert through causal inflection."""
    if len(entropy_pool) < CALABI_YAU_DIM:
        return entropy_pool

    n = len(entropy_pool)
    projections = [0.0] * CALABI_YAU_DIM

    # Project into 7D
    for dim in range(CALABI_YAU_DIM):
        for i, val in enumerate(entropy_pool[-128:]):
            phase = i * math.pi * (dim + 1) / min(n, 128)
            phi_weight = PHI ** dim / PHI ** CALABI_YAU_DIM
            projections[dim] += val * math.sin(phase) * phi_weight
        projections[dim] /= max(min(n, 128), 1)
        projections[dim] *= (1.0 + math.sin(dim * PHI) * math.cos(dim * TAU) * EULER_GAMMA)

    # Dissipate through causal coupling
    causal_coupling = math.sqrt(2) - 1
    dissipation_rate = PHI ** 2 - 1
    new_proj = list(projections)
    for i in range(CALABI_YAU_DIM):
        influx = 0.0
        for j in range(CALABI_YAU_DIM):
            if j != i:
                gradient = projections[j] - projections[i]
                coupling = math.sin((i + j) * PHI) * causal_coupling
                influx += gradient * coupling * dissipation_rate
        divine_coherence = math.sin(projections[i] * PHI * math.pi) * TAU * 0.1
        new_proj[i] = projections[i] + influx * 0.1 + divine_coherence

    return new_proj


# ═══════════════════════════════════════════════════════════════════════════════
# PYTHON GATE ANALYZER — AST-based analysis of Python logic gates
# ═══════════════════════════════════════════════════════════════════════════════

class PythonGateAnalyzer:
    """AST-based analyzer for Python logic gate implementations."""

    GATE_PATTERNS = [
        r"logic_gate", r"LogicGate", r"gate", r"Gate",
        r"quantum.*gate", r"sage.*gate", r"entangle",
        r"grover", r"amplif", r"resonan",
        # Sage core patterns
        r"sage_mode", r"sage_wisdom", r"sage_core", r"sage_enrich",
        r"sage_transform", r"sage_insight", r"sage_synth",
        # Consciousness / entropy / synthesis patterns
        r"consciousness", r"entropy", r"synthesi[sz]", r"transform",
        r"evolve", r"bridge.*emergence", r"dissipat", r"inflect",
        r"harvest.*entropy", r"causal", r"hilbert", r"calabi",
        # Core engine patterns
        r"propagat", r"aggregate", r"delegate",
    ]

    def __init__(self):
        """Initialize the Python gate analyzer."""
        self.gates: List[LogicGate] = []

    def analyze_file(self, filepath: Path) -> List[LogicGate]:
        """Analyze a Python file for logic gate implementations."""
        if not filepath.exists():
            return []

        try:
            source = filepath.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(filepath))
        except (SyntaxError, UnicodeDecodeError):
            return self._regex_fallback(filepath)

        gates = []
        rel_path = str(filepath.relative_to(WORKSPACE_ROOT))

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                if self._is_gate_related(node.name):
                    gate = self._extract_function_gate(node, rel_path, source)
                    gates.append(gate)

            elif isinstance(node, ast.ClassDef):
                if self._is_gate_related(node.name):
                    gate = self._extract_class_gate(node, rel_path, source)
                    gates.append(gate)
                    # Also extract gate-related methods
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if self._is_gate_related(item.name) or item.name in (
                                "__init__", "process", "execute", "transform", "apply"
                            ):
                                method_gate = self._extract_function_gate(
                                    item, rel_path, source, class_name=node.name
                                )
                                gates.append(method_gate)

        self.gates.extend(gates)
        return gates

    def _is_gate_related(self, name: str) -> bool:
        """Check if a function or class name matches gate-related patterns."""
        name_lower = name.lower()
        return any(re.search(pat, name_lower) for pat in self.GATE_PATTERNS)

    def _extract_function_gate(
        self, node: ast.FunctionDef, rel_path: str, source: str, class_name: str = ""
    ) -> LogicGate:
        """Extract a LogicGate from an AST function definition node."""
        params = [arg.arg for arg in node.args.args if arg.arg != "self"]
        docstring = ast.get_docstring(node) or ""
        full_name = f"{class_name}.{node.name}" if class_name else node.name

        # Estimate cyclomatic complexity
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        # Signature
        sig_parts = []
        for arg in node.args.args:
            if arg.arg == "self":
                continue
            ann = ""
            if arg.annotation:
                try:
                    ann = f": {ast.unparse(arg.annotation)}"
                except Exception:
                    ann = ""
            sig_parts.append(f"{arg.arg}{ann}")
        ret_ann = ""
        if node.returns:
            try:
                ret_ann = f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass
        signature = f"def {full_name}({', '.join(sig_parts)}){ret_ann}"

        # Content hash
        try:
            src_lines = source.split("\n")
            end_line = getattr(node, "end_lineno", node.lineno + 10)
            gate_source = "\n".join(src_lines[node.lineno - 1 : end_line])
            content_hash = hashlib.sha256(gate_source.encode()).hexdigest()[:16]
        except Exception:
            content_hash = ""

        # Entropy score
        entropy = sage_logic_gate(float(len(params) + complexity), "compress")

        return LogicGate(
            name=full_name,
            language="python",
            source_file=rel_path,
            line_number=node.lineno,
            gate_type="method" if class_name else "function",
            signature=signature,
            parameters=params,
            docstring=docstring[:200],
            complexity=complexity,
            entropy_score=entropy,
            hash=content_hash,
            last_seen=datetime.now(timezone.utc).isoformat(),
        )

    def _extract_class_gate(self, node: ast.ClassDef, rel_path: str, source: str) -> LogicGate:
        """Extract a LogicGate from an AST class definition node."""
        docstring = ast.get_docstring(node) or ""
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                pass

        methods = [
            item.name
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        try:
            src_lines = source.split("\n")
            end_line = getattr(node, "end_lineno", node.lineno + 20)
            gate_source = "\n".join(src_lines[node.lineno - 1 : end_line])
            content_hash = hashlib.sha256(gate_source.encode()).hexdigest()[:16]
        except Exception:
            content_hash = ""

        signature = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
        entropy = sage_logic_gate(float(len(methods)), "compress")

        return LogicGate(
            name=node.name,
            language="python",
            source_file=rel_path,
            line_number=node.lineno,
            gate_type="class",
            signature=signature,
            parameters=methods,
            docstring=docstring[:200],
            complexity=len(methods) * 2,
            entropy_score=entropy,
            hash=content_hash,
            last_seen=datetime.now(timezone.utc).isoformat(),
        )

    def _regex_fallback(self, filepath: Path) -> List[LogicGate]:
        """Regex fallback for files that can't be parsed (syntax errors, etc.)."""
        gates = []
        try:
            source = filepath.read_text(encoding="utf-8", errors="replace")
            rel_path = str(filepath.relative_to(WORKSPACE_ROOT))
            for match in re.finditer(
                r"(?:def|class)\s+([\w]+gate[\w]*|[\w]*Gate[\w]*|sage_\w+|quantum_\w+)",
                source, re.IGNORECASE,
            ):
                line_no = source[: match.start()].count("\n") + 1
                name = match.group(1)
                gates.append(
                    LogicGate(
                        name=name,
                        language="python",
                        source_file=rel_path,
                        line_number=line_no,
                        gate_type="function" if match.group(0).startswith("def") else "class",
                        signature=match.group(0),
                        last_seen=datetime.now(timezone.utc).isoformat(),
                    )
                )
        except Exception:
            pass
        return gates


# ═══════════════════════════════════════════════════════════════════════════════
# SWIFT GATE ANALYZER — Regex-based analysis of Swift logic gates
# ═══════════════════════════════════════════════════════════════════════════════

class SwiftGateAnalyzer:
    """Regex-based analyzer for Swift logic gate implementations."""

    SWIFT_GATE_PATTERNS = [
        (r"(?:final\s+)?class\s+(\w*[Gg]ate\w*)\s*(?::\s*[\w,\s]+)?\s*\{", "class"),
        (r"(?:final\s+)?class\s+(\w*[Ss]age\w*[Ee]ngine\w*)\s*(?::\s*[\w,\s]+)?\s*\{", "class"),
        (r"(?:final\s+)?class\s+(\w*[Ee]ntropy\w*|\w*[Cc]onsciousness\w*|\w*[Rr]esonance\w*)\s*(?::\s*[\w,\s]+)?\s*\{", "class"),
        (r"(?:final\s+)?class\s+(HyperBrain|ASIEvolver|AdaptiveLearner|PermanentMemory|QuantumProcessingCore|DynamicPhraseEngine|ASIKnowledgeBase)\s*\{", "class"),
        (r"func\s+(\w*[Gg]ate\w*)\s*\(([^)]*)\)\s*(?:->\s*(\S+))?\s*\{", "function"),
        (r"func\s+(sage\w+|bridgeEmergence|sageTransform|enrichContext|seedAllProcesses)\s*\(([^)]*)\)\s*(?:->\s*(\S+))?\s*\{", "function"),
        (r"func\s+(harvest\w+Entropy|projectToHigherDimensions|dissipateHigherDimensional|causalInflection|synthesizeDeep\w+)\s*\(([^)]*)\)\s*(?:->\s*(\S+))?\s*\{", "function"),
        (r"func\s+(\w*[Ee]ntangle\w*|\w*[Rr]esonan\w*|\w*[Aa]mplif\w*|\w*[Pp]ropagat\w*)\s*\(([^)]*)\)\s*(?:->\s*(\S+))?\s*\{", "function"),
    ]

    def __init__(self):
        """Initialize the Swift gate analyzer."""
        self.gates: List[LogicGate] = []

    def analyze_file(self, filepath: Path) -> List[LogicGate]:
        """Analyze a Swift file for logic gate implementations."""
        if not filepath.exists():
            return []

        try:
            source = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        gates = []
        rel_path = str(filepath.relative_to(WORKSPACE_ROOT))
        lines = source.split("\n")

        for pattern, gate_type in self.SWIFT_GATE_PATTERNS:
            for match in re.finditer(pattern, source, re.MULTILINE):
                name = match.group(1)
                line_no = source[: match.start()].count("\n") + 1

                # Extract parameters for functions
                params = []
                if gate_type == "function" and match.lastindex and match.lastindex >= 2:
                    param_str = match.group(2)
                    params = [p.strip().split(":")[0].strip() for p in param_str.split(",") if p.strip()]

                # Return type
                ret_type = ""
                if gate_type == "function" and match.lastindex and match.lastindex >= 3:
                    ret_type = match.group(3) or ""

                sig = match.group(0).rstrip(" {")

                # Content hash from surrounding lines
                start_idx = max(0, line_no - 1)
                end_idx = min(len(lines), start_idx + 50)
                snippet = "\n".join(lines[start_idx:end_idx])
                content_hash = hashlib.sha256(snippet.encode()).hexdigest()[:16]

                entropy = sage_logic_gate(float(len(params) + 1), "amplify")

                gates.append(
                    LogicGate(
                        name=name,
                        language="swift",
                        source_file=rel_path,
                        line_number=line_no,
                        gate_type=gate_type,
                        signature=sig[:200],
                        parameters=params,
                        complexity=0,
                        entropy_score=entropy,
                        hash=content_hash,
                        last_seen=datetime.now(timezone.utc).isoformat(),
                    )
                )

        # Deduplicate by name+line
        seen = set()
        deduped = []
        for g in gates:
            key = (g.name, g.line_number)
            if key not in seen:
                seen.add(key)
                deduped.append(g)

        self.gates.extend(deduped)
        return deduped


# ═══════════════════════════════════════════════════════════════════════════════
# JAVASCRIPT GATE ANALYZER — Regex-based analysis of JS logic gates
# ═══════════════════════════════════════════════════════════════════════════════

class JavaScriptGateAnalyzer:
    """Regex-based analyzer for JavaScript logic gate implementations."""

    JS_GATE_PATTERNS = [
        (r"class\s+(\w*[Gg]ate\w*)\s*(?:extends\s+\w+)?\s*\{", "class"),
        (r"(?:export\s+)?(?:async\s+)?function\s+(\w*[Gg]ate\w*)\s*\(", "function"),
        (r"(\w*[Gg]ate\w*)\s*[=:]\s*(?:async\s+)?(?:\([^)]*\))?\s*=>", "function"),
        (r"(\w*[Gg]ate\w*)\s*[=:]\s*function", "function"),
    ]

    def analyze_directory(self, directory: Path) -> List[LogicGate]:
        """Scan a directory for JavaScript logic gate implementations."""
        gates = []
        for js_file in directory.rglob("*.js"):
            try:
                source = js_file.read_text(encoding="utf-8", errors="replace")
                rel_path = str(js_file.relative_to(WORKSPACE_ROOT))
                for pattern, gate_type in self.JS_GATE_PATTERNS:
                    for match in re.finditer(pattern, source, re.MULTILINE):
                        name = match.group(1)
                        line_no = source[: match.start()].count("\n") + 1
                        sig = match.group(0).rstrip(" {")
                        content_hash = hashlib.sha256(sig.encode()).hexdigest()[:16]

                        gates.append(
                            LogicGate(
                                name=name,
                                language="javascript",
                                source_file=rel_path,
                                line_number=line_no,
                                gate_type=gate_type,
                                signature=sig[:200],
                                hash=content_hash,
                                last_seen=datetime.now(timezone.utc).isoformat(),
                            )
                        )
            except Exception:
                continue
        return gates


# ═══════════════════════════════════════════════════════════════════════════════
# GATE LINK ANALYZER — Discover quantum links between gates
# ═══════════════════════════════════════════════════════════════════════════════

class GateLinkAnalyzer:
    """Discovers cross-file quantum links between gate implementations."""

    # Semantic word groups for fuzzy cross-language matching
    SEMANTIC_GROUPS = [
        {"sage", "wisdom", "insight", "consciousness", "enlighten"},
        {"gate", "logic", "process", "compute", "evaluate"},
        {"quantum", "qubit", "hilbert", "fidelity", "superposition"},
        {"entropy", "dissipate", "chaos", "energy", "harvest"},
        {"entangle", "epr", "correlation", "bell", "pair"},
        {"resonance", "resonate", "harmonic", "frequency", "vibration"},
        {"amplify", "boost", "grover", "amplification", "gain"},
        {"evolve", "evolving", "mutation", "evolution", "adapt"},
        {"bridge", "emergence", "cross", "synthesis", "merge"},
        {"transform", "convert", "modulate", "project", "map"},
        {"causal", "inflect", "reconvert", "deterministic"},
        {"memory", "permanent", "persist", "store", "recall"},
        {"learn", "mastery", "adaptive", "train", "knowledge"},
    ]

    def analyze_links(self, gates: List[LogicGate]) -> List[GateLink]:
        """Discover cross-file quantum links between gate implementations."""
        links = []
        seen_pairs = set()

        # 1. Cross-language semantic matching
        for gate in gates:
            for other_gate in gates:
                if gate is other_gate:
                    continue
                if gate.language == other_gate.language and gate.source_file == other_gate.source_file:
                    continue
                pair_key = tuple(sorted([gate.name, other_gate.name]))
                if pair_key in seen_pairs:
                    continue

                sim = self._semantic_similarity(gate.name, other_gate.name)
                if sim >= 0.4:
                    link_type = "mirrors" if gate.language != other_gate.language else "resonates"
                    links.append(GateLink(
                        source_gate=gate.name, target_gate=other_gate.name,
                        link_type=link_type, strength=min(1.0, sim * PHI),
                        evidence=f"{gate.source_file}↔{other_gate.source_file}",
                    ))
                    seen_pairs.add(pair_key)

        # 2. Cross-file call graph: scan source for references to other gates
        call_links = self._analyze_call_graph(gates)
        links.extend(call_links)

        # 3. Entanglement links: shared parameters (same or cross-file)
        for i, gate_a in enumerate(gates):
            for gate_b in gates[i + 1:]:
                if gate_a.source_file == gate_b.source_file and gate_a.language == gate_b.language:
                    continue
                shared_params = set(gate_a.parameters) & set(gate_b.parameters)
                shared_params -= {"self", "value", "data", "query", "result", "args", "kwargs"}
                if len(shared_params) >= 2:
                    pair_key = tuple(sorted([gate_a.name, gate_b.name]))
                    if pair_key not in seen_pairs:
                        links.append(GateLink(
                            source_gate=gate_a.name, target_gate=gate_b.name,
                            link_type="entangles", strength=len(shared_params) * TAU,
                            evidence=f"Shared: {','.join(list(shared_params)[:5])}",
                        ))
                        seen_pairs.add(pair_key)

        return links

    def _semantic_similarity(self, a: str, b: str) -> float:
        """Compute semantic similarity between gate names using word groups."""
        # Normalize: split camelCase and snake_case into words
        words_a = set(w.lower() for w in re.split(r'[_. ]|(?<=[a-z])(?=[A-Z])', a) if len(w) > 2)
        words_b = set(w.lower() for w in re.split(r'[_. ]|(?<=[a-z])(?=[A-Z])', b) if len(w) > 2)

        if not words_a or not words_b:
            return 0.0

        # Direct word overlap
        direct_overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)

        # Semantic group overlap
        groups_a = set()
        groups_b = set()
        for i, group in enumerate(self.SEMANTIC_GROUPS):
            if words_a & group:
                groups_a.add(i)
            if words_b & group:
                groups_b.add(i)
        group_overlap = len(groups_a & groups_b) / max(len(groups_a | groups_b), 1) if (groups_a or groups_b) else 0.0

        return direct_overlap * 0.6 + group_overlap * 0.4

    def _analyze_call_graph(self, gates: List[LogicGate]) -> List[GateLink]:
        """Scan source files for cross-file function call references."""
        links = []
        seen = set()
        # Build lookup of gate names by file
        gates_by_file: Dict[str, List[LogicGate]] = {}
        for g in gates:
            gates_by_file.setdefault(g.source_file, []).append(g)

        # For each file, scan for references to gates defined in OTHER files
        file_contents_cache: Dict[str, str] = {}
        for src_file, src_gates in gates_by_file.items():
            full_path = WORKSPACE_ROOT / src_file
            if not full_path.exists():
                continue
            if src_file not in file_contents_cache:
                try:
                    file_contents_cache[src_file] = full_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
            content = file_contents_cache[src_file]

            for other_file, other_gates in gates_by_file.items():
                if other_file == src_file:
                    continue
                for og in other_gates:
                    # Only check meaningful names (skip __init__ etc)
                    clean_name = og.name.split(".")[-1]
                    if len(clean_name) < 4 or clean_name.startswith("__"):
                        continue
                    if clean_name in content:
                        pair_key = (src_file, og.name)
                        if pair_key not in seen:
                            # Find a gate in src_file that likely calls it
                            caller = src_gates[0].name if src_gates else src_file
                            links.append(GateLink(
                                source_gate=caller, target_gate=og.name,
                                link_type="calls",
                                strength=TAU,
                                evidence=f"{src_file}→{other_file}",
                            ))
                            seen.add(pair_key)
        return links

    def populate_gate_links(self, gates: List[LogicGate], links: List[GateLink]):
        """Populate LogicGate.quantum_links field with discovered connections.

        Uses pre-indexed lookup for O(n+m) performance instead of O(n*m).
        """
        # Build index: gate_name -> set of "peer:link_type" strings
        from collections import defaultdict
        index: Dict[str, set] = defaultdict(set)
        for link in links:
            index[link.source_gate].add(f"{link.target_gate}:{link.link_type}")
            index[link.target_gate].add(f"{link.source_gate}:{link.link_type}")
        # Apply to gates
        for gate in gates:
            gate.quantum_links = list(index.get(gate.name, set()))


# ═══════════════════════════════════════════════════════════════════════════════
# GATE RESEARCH ENGINE — Advanced analysis ported from Quantum Research Engine
# ═══════════════════════════════════════════════════════════════════════════════

class GateResearchEngine:
    """Advanced research engine for logic gate analysis.

    Ported from QuantumResearchEngine in l104_quantum_link_builder.py:
      Module 1: ANOMALY DETECTION — IQR-based outlier identification on gate metrics
      Module 2: CAUSAL ANALYSIS   — Correlation between complexity, entropy, connectivity
      Module 3: EVOLUTION ANALYSIS — Trend detection from chronolizer history
      Module 4: CROSS-POLLINATION — Insights for quantum link builder integration
      Module 5: KNOWLEDGE SYNTHESIS — Aggregate health score with adaptive learning

    Enables the gate builder to match the quantum link builder's research depth
    and feed learned insights back for cross-system improvement.
    """

    MEMORY_FILE = WORKSPACE_ROOT / ".l104_gate_research_memory.json"

    def __init__(self):
        """Initialize the gate research engine and load persisted memory."""
        self.memory: Dict = {}
        self._load_memory()

    def _load_memory(self):
        """Load persistent research memory."""
        if self.MEMORY_FILE.exists():
            try:
                self.memory = json.loads(self.MEMORY_FILE.read_text())
            except Exception:
                self.memory = {}

    def _save_memory(self):
        """Persist research memory to disk."""
        try:
            self.memory["last_updated"] = datetime.now(timezone.utc).isoformat()
            self.MEMORY_FILE.write_text(json.dumps(self.memory, indent=2, default=str))
        except Exception:
            pass

    def full_research(self, gates: List[LogicGate], links: List[GateLink],
                      chronolizer: 'GateChronolizer' = None) -> Dict:
        """Run all research modules and produce a unified report."""
        if not gates:
            return {"total_gates": 0, "research_health": 0}

        start = time.time()

        # Extract metrics arrays
        complexities = [g.complexity for g in gates]
        entropies = [g.entropy_score for g in gates]
        link_counts = []
        for g in gates:
            lc = sum(1 for l in links if g.name in (l.source_gate, l.target_gate))
            link_counts.append(lc)

        # Module 1: Anomaly detection
        anomalies = self._detect_anomalies(gates, complexities, entropies, link_counts)

        # Module 2: Causal analysis
        causal = self._causal_analysis(complexities, entropies, link_counts, gates)

        # Module 3: Evolution analysis
        evolution = self._evolution_analysis(chronolizer) if chronolizer else {}

        # Module 4: Cross-pollination data preparation
        cross_poll = self._prepare_cross_pollination(gates, links, anomalies, causal)

        # Module 5: Knowledge synthesis + adaptive learning
        synthesis = self._knowledge_synthesis(
            gates, links, anomalies, causal, evolution, cross_poll)

        elapsed = time.time() - start

        result = {
            "total_gates": len(gates),
            "anomaly_detection": anomalies,
            "causal_analysis": causal,
            "evolution_analysis": evolution,
            "cross_pollination": cross_poll,
            "knowledge_synthesis": synthesis,
            "research_health": synthesis.get("research_health", 0),
            "research_time_ms": elapsed * 1000,
        }

        # Persist snapshot for self-learning
        self._record_snapshot(result)
        self._save_memory()

        return result

    def _detect_anomalies(self, gates: List[LogicGate],
                          complexities: List[int], entropies: List[float],
                          link_counts: List[int]) -> Dict:
        """IQR-based outlier detection on gate metrics."""
        anomalies = []
        for prop_name, values in [("complexity", complexities),
                                   ("entropy", entropies),
                                   ("connectivity", link_counts)]:
            fvals = [float(v) for v in values]
            if len(fvals) < 4:
                continue
            sorted_v = sorted(fvals)
            n = len(sorted_v)
            q1, q3 = sorted_v[n // 4], sorted_v[3 * n // 4]
            iqr = q3 - q1
            lower, upper = q1 - 2.0 * iqr, q3 + 2.0 * iqr

            for i, v in enumerate(fvals):
                if v < lower or v > upper:
                    anomalies.append({
                        "gate": gates[i].name[:60],
                        "property": prop_name,
                        "value": v,
                        "severity": "extreme" if (v < lower - iqr or v > upper + iqr) else "mild",
                    })

        return {
            "total_anomalies": len(anomalies),
            "extreme": sum(1 for a in anomalies if a["severity"] == "extreme"),
            "mild": sum(1 for a in anomalies if a["severity"] == "mild"),
            "anomaly_rate": len(anomalies) / max(1, len(gates) * 3),
            "top_anomalies": anomalies[:10],
        }

    def _causal_analysis(self, complexities: List[int], entropies: List[float],
                         link_counts: List[int], gates: List[LogicGate]) -> Dict:
        """Pearson correlation between gate properties."""
        def _pearson(xs, ys):
            """Compute Pearson correlation coefficient between two sequences."""
            n = len(xs)
            if n < 2:
                return 0.0
            mx, my = sum(xs) / n, sum(ys) / n
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            sxx = sum((x - mx) ** 2 for x in xs)
            syy = sum((y - my) ** 2 for y in ys)
            d = math.sqrt(sxx * syy)
            return sxy / d if d > 1e-15 else 0.0

        props = {
            "complexity": [float(c) for c in complexities],
            "entropy": entropies,
            "connectivity": [float(c) for c in link_counts],
            "has_docstring": [1.0 if g.docstring else 0.0 for g in gates],
            "param_count": [float(len(g.parameters)) for g in gates],
        }

        correlations = {}
        strong = []
        names = list(props.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = _pearson(props[names[i]], props[names[j]])
                key = f"{names[i]}↔{names[j]}"
                correlations[key] = round(r, 4)
                if abs(r) > 0.5:
                    strong.append({"pair": key, "correlation": r,
                                   "strength": "strong" if abs(r) > 0.7 else "moderate"})

        return {
            "correlations": correlations,
            "strong_correlations": strong,
            "total_strong": len(strong),
        }

    def _evolution_analysis(self, chronolizer: 'GateChronolizer') -> Dict:
        """Analyze gate evolution trends from chronolizer history."""
        if not chronolizer or not chronolizer.entries:
            return {"has_history": False}

        entries = chronolizer.entries
        summary = chronolizer.summary()

        # Compute evolution velocity: events per unique gate
        unique_gates = summary.get("unique_gates_tracked", 1)
        total = summary.get("total_entries", 0)
        velocity = total / max(1, unique_gates)

        # Recent activity: last 50 entries
        recent = entries[-50:]
        recent_events = {}
        for e in recent:
            recent_events[e.event] = recent_events.get(e.event, 0) + 1

        # Churn rate: how many gates were modified vs total tracked
        modified = summary.get("events_by_type", {}).get("modified", 0)
        discovered = summary.get("events_by_type", {}).get("discovered", 0)
        churn_rate = modified / max(1, discovered)

        # Stability assessment
        if churn_rate < 0.1:
            stability = "highly_stable"
        elif churn_rate < 0.3:
            stability = "stable"
        elif churn_rate < 0.6:
            stability = "moderate_churn"
        else:
            stability = "high_churn"

        return {
            "has_history": True,
            "total_events": total,
            "unique_gates": unique_gates,
            "evolution_velocity": velocity,
            "churn_rate": churn_rate,
            "stability": stability,
            "recent_activity": recent_events,
            "oldest": summary.get("oldest"),
            "newest": summary.get("newest"),
        }

    def _prepare_cross_pollination(self, gates: List[LogicGate],
                                   links: List[GateLink],
                                   anomalies: Dict, causal: Dict) -> Dict:
        """Prepare data for cross-pollination with quantum link builder.
        This data is consumed by ResearchMemoryBank and QuantumLinkBuilder."""
        # Aggregate gate data by file for the quantum link builder
        gates_by_file: Dict[str, Dict] = {}
        for g in gates:
            if g.source_file not in gates_by_file:
                gates_by_file[g.source_file] = {
                    "count": 0, "types": [], "complexity_sum": 0,
                    "entropy_sum": 0.0, "tested": 0, "linked": 0}
            gf = gates_by_file[g.source_file]
            gf["count"] += 1
            if g.gate_type not in gf["types"]:
                gf["types"].append(g.gate_type)
            gf["complexity_sum"] += g.complexity
            gf["entropy_sum"] += g.entropy_score
            if g.test_status == "passed":
                gf["tested"] += 1
            if g.quantum_links:
                gf["linked"] += 1

        # Identify high-value files (many gates, tested, linked)
        high_value_files = []
        for fname, data in gates_by_file.items():
            score = (data["count"] * 0.3 + data["tested"] * 0.3
                     + data["linked"] * 0.2 + min(1.0, data["complexity_sum"] / 50) * 0.2)
            if score > 0.5:
                high_value_files.append({"file": fname, "score": score,
                                         "gates": data["count"]})

        # Cross-system insights
        insights = []
        if anomalies.get("extreme", 0) > 2:
            insights.append("GATE_ANOMALY: Multiple extreme outliers — "
                           "quantum repair engine should prioritize these files")
        strong_corrs = causal.get("strong_correlations", [])
        for c in strong_corrs:
            if "complexity" in c["pair"] and "entropy" in c["pair"]:
                insights.append(f"GATE_CAUSAL: Complexity↔Entropy correlation "
                               f"(r={c['correlation']:.3f}) — mirrors quantum "
                               f"fidelity↔strength pattern")

        return {
            "gates_by_file": gates_by_file,
            "high_value_files": sorted(high_value_files,
                                       key=lambda x: -x["score"])[:20],
            "cross_system_insights": insights,
            "total_tested": sum(1 for g in gates if g.test_status == "passed"),
            "total_linked": sum(1 for g in gates if g.quantum_links),
            "mean_health": self._compute_mean_health(gates, links),
        }

    def _compute_mean_health(self, gates: List[LogicGate],
                             links: List[GateLink]) -> float:
        """Compute mean gate health score (0-1)."""
        if not gates:
            return 0.0
        scores = []
        for g in gates:
            complexity_score = min(1.0, g.complexity / 20.0)
            has_test = 1.0 if g.test_status == "passed" else 0.3
            has_doc = 1.0 if g.docstring else 0.5
            lc = sum(1 for l in links if g.name in (l.source_gate, l.target_gate))
            connectivity = min(1.0, lc * 0.1)
            h = (complexity_score * 0.2 + has_test * 0.25 + has_doc * 0.15
                 + connectivity * 0.2 + min(1.0, g.entropy_score) * 0.2)
            scores.append(h)
        return sum(scores) / len(scores)

    def _knowledge_synthesis(self, gates: List[LogicGate], links: List[GateLink],
                             anomalies: Dict, causal: Dict,
                             evolution: Dict, cross_poll: Dict) -> Dict:
        """Synthesize all research into a unified health score with adaptive learning."""
        insights = []
        risks = []

        # Anomaly insights
        ar = anomalies.get("anomaly_rate", 0)
        if ar > 0.05:
            risks.append(f"HIGH_ANOMALY: {ar:.1%} of gate measurements are outliers")
        elif ar < 0.01:
            insights.append("LOW_ANOMALY: Gate metrics are highly uniform")

        # Connectivity insights
        total_linked = cross_poll.get("total_linked", 0)
        total_gates = len(gates)
        link_ratio = total_linked / max(1, total_gates)
        if link_ratio > 0.5:
            insights.append(f"WELL_CONNECTED: {link_ratio:.0%} of gates have cross-file links")
        elif link_ratio < 0.1:
            risks.append(f"ISOLATED: Only {link_ratio:.0%} of gates have cross-file links")

        # Test coverage insights
        tested = cross_poll.get("total_tested", 0)
        test_ratio = tested / max(1, total_gates)
        if test_ratio > 0.3:
            insights.append(f"TESTED: {test_ratio:.0%} of gates have passing tests")
        else:
            risks.append(f"UNDERTESTED: Only {test_ratio:.0%} of gates tested")

        # Evolution insights
        if evolution.get("has_history"):
            stability = evolution.get("stability", "unknown")
            if stability in ("highly_stable", "stable"):
                insights.append(f"STABLE_CODEBASE: {stability} gate evolution")
            elif stability == "high_churn":
                risks.append("HIGH_CHURN: Gates changing rapidly — stability concern")

        # Causal insights
        for c in causal.get("strong_correlations", []):
            insights.append(f"CAUSAL: {c['pair']} (r={c['correlation']:.3f})")

        # Adaptive learning: compare with previous runs
        prev_health = self.memory.get("last_research_health", 0)
        history = self.memory.get("health_history", [])

        # Compute health score
        mean_health = cross_poll.get("mean_health", 0.3)
        anomaly_factor = max(0, 1.0 - ar * 4)
        link_factor = min(1.0, link_ratio * 2)
        test_factor = min(1.0, test_ratio * 2)
        insight_factor = min(1.0, len(insights) / 5)
        stability_factor = (0.9 if evolution.get("stability") in ("highly_stable", "stable")
                            else 0.6 if evolution.get("stability") == "moderate_churn"
                            else 0.4)

        research_health = (
            mean_health * 0.25
            + anomaly_factor * 0.15
            + link_factor * 0.15
            + test_factor * 0.15
            + insight_factor * 0.10
            + stability_factor * 0.10
            + 0.10  # Base floor
        )
        risk_penalty = min(0.15, len(risks) * 0.03)
        research_health = min(1.0, max(0.05, research_health - risk_penalty))

        # Learning trend
        learning_trend = "unknown"
        if prev_health > 0:
            if research_health > prev_health + 0.01:
                learning_trend = "improving"
            elif research_health < prev_health - 0.01:
                learning_trend = "degrading"
            else:
                learning_trend = "stable"

        # Update persistent memory
        history.append(research_health)
        self.memory["last_research_health"] = research_health
        self.memory["health_history"] = history[-50:]
        self.memory["learning_trend"] = learning_trend

        return {
            "insights": insights,
            "risks": risks,
            "research_health": research_health,
            "learning_trend": learning_trend,
            "prev_health": prev_health,
        }

    def _record_snapshot(self, result: Dict):
        """Record research snapshot for self-learning."""
        snapshots = self.memory.get("snapshots", [])
        snapshots.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health": result.get("research_health", 0),
            "total_gates": result.get("total_gates", 0),
            "anomaly_rate": result.get("anomaly_detection", {}).get("anomaly_rate", 0),
        })
        self.memory["snapshots"] = snapshots[-50:]


# ═══════════════════════════════════════════════════════════════════════════════
# AUTOMATED TEST GENERATOR — Creates and runs gate tests
# ═══════════════════════════════════════════════════════════════════════════════

class GateTestGenerator:
    """Automated test generation and execution for logic gates."""

    def generate_tests(self, gates: List[LogicGate]) -> List[Dict[str, Any]]:
        """Generate test cases for discovered logic gates."""
        tests = []

        for gate in gates:
            if gate.language == "python" and gate.gate_type == "function":
                tests.extend(self._generate_python_function_tests(gate))
            elif gate.language == "python" and gate.gate_type == "class":
                tests.append(self._generate_python_class_test(gate))

        return tests

    def _generate_python_function_tests(self, gate: LogicGate) -> List[Dict[str, Any]]:
        """Generate test cases for a Python function gate."""
        tests = []
        test_inputs = [0.0, 1.0, PHI, -1.0, GOD_CODE * 0.001, math.pi, TAU]

        for i, inp in enumerate(test_inputs[:3]):
            tests.append({
                "test_id": f"{gate.name}_input_{i}",
                "gate_name": gate.name,
                "source_file": gate.source_file,
                "test_type": "smoke",
                "input": inp,
                "description": f"Smoke test {gate.name} with input {inp}",
                "status": "pending",
            })

        return tests

    def _generate_python_class_test(self, gate: LogicGate) -> Dict[str, Any]:
        """Generate an instantiation test for a Python class gate."""
        return {
            "test_id": f"{gate.name}_instantiation",
            "gate_name": gate.name,
            "source_file": gate.source_file,
            "test_type": "instantiation",
            "description": f"Verify {gate.name} class can be instantiated",
            "status": "pending",
        }

    def run_builtin_gate_tests(self) -> List[Dict[str, Any]]:
        """Run tests on the built-in sage_logic_gate and quantum_logic_gate."""
        results = []
        operations = ["align", "filter", "amplify", "compress", "entangle", "dissipate", "inflect"]
        test_values = [0.0, 1.0, PHI, -PHI, GOD_CODE * 0.001, math.pi, 100.0, -100.0]

        for op in operations:
            for val in test_values:
                test_id = f"sage_gate_{op}_{val}"
                try:
                    result = sage_logic_gate(val, op)
                    passed = not (math.isnan(result) or math.isinf(result))
                    results.append({
                        "test_id": test_id,
                        "gate_name": "sage_logic_gate",
                        "operation": op,
                        "input": val,
                        "output": result,
                        "passed": passed,
                        "error": None,
                    })
                except Exception as e:
                    results.append({
                        "test_id": test_id,
                        "gate_name": "sage_logic_gate",
                        "operation": op,
                        "input": val,
                        "output": None,
                        "passed": False,
                        "error": str(e),
                    })

        # Quantum gate tests
        for depth in range(1, 6):
            for val in [1.0, PHI, GOD_CODE * 0.001]:
                test_id = f"quantum_gate_d{depth}_{val}"
                try:
                    result = quantum_logic_gate(val, depth)
                    passed = not (math.isnan(result) or math.isinf(result))
                    results.append({
                        "test_id": test_id,
                        "gate_name": "quantum_logic_gate",
                        "depth": depth,
                        "input": val,
                        "output": result,
                        "passed": passed,
                        "error": None,
                    })
                except Exception as e:
                    results.append({
                        "test_id": test_id,
                        "gate_name": "quantum_logic_gate",
                        "depth": depth,
                        "input": val,
                        "output": None,
                        "passed": False,
                        "error": str(e),
                    })

        # Entanglement test
        for a, b in [(1.0, PHI), (GOD_CODE, TAU), (0.0, 0.0)]:
            try:
                ea, eb = entangle_values(a, b)
                passed = not (math.isnan(ea) or math.isnan(eb))
                results.append({
                    "test_id": f"entangle_{a}_{b}",
                    "gate_name": "entangle_values",
                    "input": (a, b),
                    "output": (ea, eb),
                    "passed": passed,
                    "error": None,
                })
            except Exception as e:
                results.append({
                    "test_id": f"entangle_{a}_{b}",
                    "gate_name": "entangle_values",
                    "input": (a, b),
                    "output": None,
                    "passed": False,
                    "error": str(e),
                })

        # Higher-dimensional dissipation test
        try:
            test_pool = [math.sin(i * PHI) for i in range(64)]
            result = higher_dimensional_dissipation(test_pool)
            passed = len(result) == CALABI_YAU_DIM and all(
                not (math.isnan(v) or math.isinf(v)) for v in result
            )
            results.append({
                "test_id": "higher_dim_dissipation",
                "gate_name": "higher_dimensional_dissipation",
                "input": f"64-element entropy pool",
                "output": result,
                "passed": passed,
                "error": None,
            })
        except Exception as e:
            results.append({
                "test_id": "higher_dim_dissipation",
                "gate_name": "higher_dimensional_dissipation",
                "input": "64-element entropy pool",
                "output": None,
                "passed": False,
                "error": str(e),
            })

        # ─── INTEGRITY TESTS — Idempotency, boundary, sacred-constant coherence ───

        # Idempotency: compress(compress(x)) should converge
        for val in [PHI, GOD_CODE * 0.01, 1.0]:
            try:
                r1 = sage_logic_gate(val, "compress")
                r2 = sage_logic_gate(r1, "compress")
                r3 = sage_logic_gate(r2, "compress")
                converging = abs(r3 - r2) <= abs(r2 - r1) + 1e-10
                results.append({
                    "test_id": f"idempotency_compress_{val}",
                    "gate_name": "sage_logic_gate",
                    "operation": "compress_idempotency",
                    "input": val,
                    "output": [r1, r2, r3],
                    "passed": converging,
                    "error": None if converging else "Compression diverges instead of converging",
                })
            except Exception as e:
                results.append({
                    "test_id": f"idempotency_compress_{val}",
                    "gate_name": "sage_logic_gate", "operation": "compress_idempotency",
                    "input": val, "output": None, "passed": False, "error": str(e),
                })

        # Sacred constant coherence: PHI * TAU should ≈ 1.0
        try:
            phi_tau = PHI * TAU
            passed = abs(phi_tau - 1.0) < 1e-10
            results.append({
                "test_id": "sacred_phi_tau_unity",
                "gate_name": "sacred_constants",
                "input": "PHI * TAU",
                "output": phi_tau,
                "passed": passed,
                "error": None,
            })
        except Exception as e:
            results.append({
                "test_id": "sacred_phi_tau_unity",
                "gate_name": "sacred_constants",
                "input": "PHI * TAU", "output": None, "passed": False, "error": str(e),
            })

        # Boundary: entangle(0,0) should return (0,0)
        try:
            ea, eb = entangle_values(0.0, 0.0)
            passed = abs(ea) < 1e-10 and abs(eb) < 1e-10
            results.append({
                "test_id": "entangle_zero_boundary",
                "gate_name": "entangle_values",
                "input": (0.0, 0.0),
                "output": (ea, eb),
                "passed": passed,
                "error": None,
            })
        except Exception as e:
            results.append({
                "test_id": "entangle_zero_boundary",
                "gate_name": "entangle_values",
                "input": (0.0, 0.0), "output": None, "passed": False, "error": str(e),
            })

        # Symmetry: align(PHI) should equal PHI (PHI is a lattice point)
        try:
            aligned = sage_logic_gate(PHI, "align")
            passed = abs(aligned - PHI) < 0.01  # Should be very close
            results.append({
                "test_id": "align_phi_fixpoint",
                "gate_name": "sage_logic_gate",
                "operation": "align_fixpoint",
                "input": PHI,
                "output": aligned,
                "passed": passed,
                "error": None,
            })
        except Exception as e:
            results.append({
                "test_id": "align_phi_fixpoint",
                "gate_name": "sage_logic_gate", "operation": "align_fixpoint",
                "input": PHI, "output": None, "passed": False, "error": str(e),
            })

        # Amplify should always increase magnitude
        for val in [0.5, 1.0, PHI, 10.0]:
            try:
                amplified = sage_logic_gate(val, "amplify")
                passed = abs(amplified) > abs(val)
                results.append({
                    "test_id": f"amplify_increases_{val}",
                    "gate_name": "sage_logic_gate",
                    "operation": "amplify_increases",
                    "input": val,
                    "output": amplified,
                    "passed": passed,
                    "error": None,
                })
            except Exception as e:
                results.append({
                    "test_id": f"amplify_increases_{val}",
                    "gate_name": "sage_logic_gate", "operation": "amplify_increases",
                    "input": val, "output": None, "passed": False, "error": str(e),
                })

        # Dissipation should preserve total energy (conservation-ish)
        try:
            pool = [math.sin(i * PHI) for i in range(64)]
            input_energy = sum(v ** 2 for v in pool)
            result_proj = higher_dimensional_dissipation(pool)
            output_energy = sum(v ** 2 for v in result_proj)
            # Energy should be finite and non-zero
            passed = 0 < output_energy < float('inf') and not math.isnan(output_energy)
            results.append({
                "test_id": "dissipation_energy_finite",
                "gate_name": "higher_dimensional_dissipation",
                "input": f"pool_energy={input_energy:.4f}",
                "output": f"proj_energy={output_energy:.4f}",
                "passed": passed,
                "error": None,
            })
        except Exception as e:
            results.append({
                "test_id": "dissipation_energy_finite",
                "gate_name": "higher_dimensional_dissipation",
                "input": "64-pool", "output": None, "passed": False, "error": str(e),
            })

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# CHRONOLIZER — Tracks gate evolution over time
# ═══════════════════════════════════════════════════════════════════════════════

class GateChronolizer:
    """Chronological tracking of logic gate evolution."""

    def __init__(self):
        """Initialize the gate chronolizer and load persisted entries."""
        self.entries: List[ChronologEntry] = []
        self._load()

    def _load(self):
        """Load chronological entries from disk."""
        if CHRONOLOG_FILE.exists():
            try:
                data = json.loads(CHRONOLOG_FILE.read_text())
                self.entries = [
                    ChronologEntry(**e) for e in data.get("entries", [])
                ]
            except Exception:
                self.entries = []

    def save(self):
        """Persist chronological entries to disk."""
        data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_entries": len(self.entries),
            "entries": [asdict(e) for e in self.entries[-500:]],  # Keep last 500
        }
        CHRONOLOG_FILE.write_text(json.dumps(data, indent=2))

    def record(self, gate_name: str, event: str, details: str = "", file_hash: str = ""):
        """Record a chronological event for a logic gate."""
        self.entries.append(
            ChronologEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                gate_name=gate_name,
                event=event,
                details=details,
                file_hash=file_hash,
            )
        )

    def get_gate_history(self, gate_name: str) -> List[ChronologEntry]:
        """Retrieve all chronological entries for a specific gate."""
        return [e for e in self.entries if e.gate_name == gate_name]

    def get_recent(self, n: int = 20) -> List[ChronologEntry]:
        """Retrieve the most recent chronological entries."""
        return self.entries[-n:]

    def summary(self) -> Dict[str, Any]:
        """Compute a summary of all chronological tracking data."""
        events_by_type = {}
        for e in self.entries:
            events_by_type[e.event] = events_by_type.get(e.event, 0) + 1
        unique_gates = len(set(e.gate_name for e in self.entries))
        return {
            "total_entries": len(self.entries),
            "unique_gates_tracked": unique_gates,
            "events_by_type": events_by_type,
            "oldest": self.entries[0].timestamp if self.entries else None,
            "newest": self.entries[-1].timestamp if self.entries else None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM LINK MANAGER — Maintains cross-file quantum links
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumLinkManager:
    """Manages quantum links between gate implementations across files."""

    def __init__(self):
        """Initialize the quantum link manager."""
        self.links: List[GateLink] = []
        self.file_hashes: Dict[str, str] = {}

    def compute_file_hash(self, filepath: Path) -> str:
        """Compute a SHA-256 hash of a file's contents."""
        if not filepath.exists():
            return ""
        try:
            content = filepath.read_bytes()
            return hashlib.sha256(content).hexdigest()[:32]
        except Exception:
            return ""

    def check_file_changes(self) -> Dict[str, bool]:
        """Check which quantum-linked files have changed."""
        changes = {}
        for name, path in QUANTUM_LINKED_FILES.items():
            current_hash = self.compute_file_hash(path)
            prev_hash = self.file_hashes.get(name, "")
            changed = current_hash != prev_hash and current_hash != ""
            changes[name] = changed
            if current_hash:
                self.file_hashes[name] = current_hash
        return changes

    def file_sizes(self) -> Dict[str, int]:
        """Return file sizes in bytes for all quantum-linked files."""
        sizes = {}
        for name, path in QUANTUM_LINKED_FILES.items():
            if path.exists():
                sizes[name] = path.stat().st_size
        return sizes

    def line_counts(self) -> Dict[str, int]:
        """Return line counts for all quantum-linked files."""
        counts = {}
        for name, path in QUANTUM_LINKED_FILES.items():
            if path.exists():
                try:
                    counts[name] = sum(1 for _ in open(path, "r", encoding='utf-8', errors="replace"))
                except Exception:
                    counts[name] = 0
        return counts


# ═══════════════════════════════════════════════════════════════════════════════
# HYPER ASI LOGIC GATE ENVIRONMENT — The Master Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class HyperASILogicGateEnvironment:
    """
    The master environment for building, analyzing, researching, compiling,
    chronolizing, and developing all logic gates within the Allentown L104 Node.

    Autonomous operation:
    - Discovers gates across Python, Swift, and JavaScript
    - Analyzes gate implementations with AST/regex
    - Generates and runs automated tests
    - Tracks chronological evolution
    - Maintains quantum links between files
    - Syncs state to backend
    """

    def __init__(self, auto_sync: bool = True):
        """Initialize the master logic gate environment with all subsystems."""
        self.python_analyzer = PythonGateAnalyzer()
        self.swift_analyzer = SwiftGateAnalyzer()
        self.js_analyzer = JavaScriptGateAnalyzer()
        self.link_analyzer = GateLinkAnalyzer()
        self.test_generator = GateTestGenerator()
        self.chronolizer = GateChronolizer()
        self.quantum_links = QuantumLinkManager()
        self.research_engine = GateResearchEngine()
        self.stochastic_lab = StochasticGateResearchLab()
        # ★ v5.0 Quantum Dynamism Engine
        self.dynamism_engine = GateDynamismEngine()
        self.value_evolver = GateValueEvolver(self.dynamism_engine)
        # ★ v5.1 Ouroboros Sage Nirvanic Entropy Fuel Engine
        self.nirvanic_engine = OuroborosSageNirvanicEngine()

        self.all_gates: List[LogicGate] = []
        self.all_links: List[GateLink] = []
        self.test_results: List[Dict[str, Any]] = []
        self._last_research: Dict = {}

        self._auto_sync = auto_sync
        self._last_full_scan = None
        self._scan_count = 0

        # Load previous state
        self._load_state()

    # ─── DISCOVERY ───────────────────────────────────────────────────

    def full_scan(self) -> Dict[str, Any]:
        """Perform a complete scan of all quantum-linked files for logic gates."""
        print("\n╔══════════════════════════════════════════════════╗")
        print("║  L104 HYPER ASI LOGIC GATE FULL SCAN             ║")
        print("╚══════════════════════════════════════════════════╝\n")

        self._scan_count += 1
        prev_gate_count = len(self.all_gates)
        prev_gate_names = {g.name for g in self.all_gates}

        # Check for file changes
        changes = self.quantum_links.check_file_changes()
        changed_files = [name for name, changed in changes.items() if changed]
        if changed_files:
            print(f"  ⚡ Files changed: {', '.join(changed_files)}")
        else:
            print("  ✓ No file changes detected (rescanning anyway)")

        self.all_gates = []

        # Python files
        python_files = [
            QUANTUM_LINKED_FILES["const.py"],
            QUANTUM_LINKED_FILES["main.py"],
            QUANTUM_LINKED_FILES["l104_fast_server.py"],
            QUANTUM_LINKED_FILES["l104_local_intellect.py"],
        ]
        # Also scan other relevant Python files
        for pyfile in WORKSPACE_ROOT.glob("l104*.py"):
            if pyfile not in python_files:
                python_files.append(pyfile)

        py_count = 0
        for pyfile in python_files:
            if pyfile.exists():
                gates = self.python_analyzer.analyze_file(pyfile)
                self.all_gates.extend(gates)
                py_count += len(gates)
                if gates:
                    print(f"  🐍 {pyfile.name}: {len(gates)} gates")

        # Swift
        swift_file = QUANTUM_LINKED_FILES["L104Native.swift"]
        swift_gates = self.swift_analyzer.analyze_file(swift_file)
        self.all_gates.extend(swift_gates)
        if swift_gates:
            print(f"  🦅 L104Native.swift: {len(swift_gates)} gates")

        # JavaScript
        js_dirs = [WORKSPACE_ROOT, WORKSPACE_ROOT / "deploy"]
        js_count = 0
        for js_dir in js_dirs:
            if js_dir.exists():
                js_gates = self.js_analyzer.analyze_directory(js_dir)
                self.all_gates.extend(js_gates)
                js_count += len(js_gates)
        if js_count:
            print(f"  📜 JavaScript: {js_count} gates")

        # Analyze cross-file links
        self.all_links = self.link_analyzer.analyze_links(self.all_gates)

        # Populate quantum_links field on each gate (was previously always empty)
        self.link_analyzer.populate_gate_links(self.all_gates, self.all_links)

        # Chronolize changes
        current_names = {g.name for g in self.all_gates}
        new_gates = current_names - prev_gate_names
        removed_gates = prev_gate_names - current_names

        for name in new_gates:
            self.chronolizer.record(name, "discovered", f"Scan #{self._scan_count}")
        for name in removed_gates:
            self.chronolizer.record(name, "removed", f"Scan #{self._scan_count}")

        # Check for modifications via hash comparison
        prev_hashes = {}
        if STATE_FILE.exists():
            try:
                prev_state = json.loads(STATE_FILE.read_text())
                prev_hashes = prev_state.get("gate_hashes", {})
            except Exception:
                pass
        for gate in self.all_gates:
            if gate.name in prev_gate_names and gate.hash:
                old_hash = prev_hashes.get(gate.name, "")
                if old_hash and old_hash != gate.hash:
                    self.chronolizer.record(gate.name, "modified",
                        f"Hash: {old_hash[:8]}→{gate.hash[:8]}", file_hash=gate.hash)

        self.chronolizer.save()
        self._last_full_scan = datetime.now(timezone.utc).isoformat()
        self._save_state()

        summary = {
            "scan_number": self._scan_count,
            "total_gates": len(self.all_gates),
            "python_gates": py_count,
            "swift_gates": len(swift_gates),
            "js_gates": js_count,
            "cross_file_links": len(self.all_links),
            "new_gates": list(new_gates),
            "removed_gates": list(removed_gates),
            "file_line_counts": self.quantum_links.line_counts(),
            "timestamp": self._last_full_scan,
        }

        print(f"\n  ═══ SCAN COMPLETE ═══")
        print(f"  Total gates:    {summary['total_gates']}")
        print(f"  Python:         {py_count}")
        print(f"  Swift:          {len(swift_gates)}")
        print(f"  JavaScript:     {js_count}")
        print(f"  Quantum links:  {len(self.all_links)}")
        print(f"  New:            {len(new_gates)}")
        print(f"  Removed:        {len(removed_gates)}")

        return summary

    # ─── ANALYSIS ────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        """Deep analysis of all discovered gates. Rescans only if no gates loaded."""
        if not self.all_gates:
            self.full_scan()

        by_language = {}
        by_type = {}
        by_file = {}
        total_complexity = 0
        total_entropy = 0.0

        for gate in self.all_gates:
            by_language[gate.language] = by_language.get(gate.language, 0) + 1
            by_type[gate.gate_type] = by_type.get(gate.gate_type, 0) + 1
            by_file[gate.source_file] = by_file.get(gate.source_file, 0) + 1
            total_complexity += gate.complexity
            total_entropy += gate.entropy_score

        # Entropy statistics
        entropy_values = [g.entropy_score for g in self.all_gates if g.entropy_score != 0]
        mean_entropy = sum(entropy_values) / max(len(entropy_values), 1)
        entropy_dissipated = higher_dimensional_dissipation(entropy_values) if len(entropy_values) >= 7 else []

        # Most complex gates
        sorted_by_complexity = sorted(self.all_gates, key=lambda g: g.complexity, reverse=True)
        top_complex = [(g.name, g.complexity, g.source_file) for g in sorted_by_complexity[:10]]

        analysis = {
            "total_gates": len(self.all_gates),
            "by_language": by_language,
            "by_type": by_type,
            "by_file": by_file,
            "total_complexity": total_complexity,
            "mean_complexity": total_complexity / max(len(self.all_gates), 1),
            "total_entropy": total_entropy,
            "mean_entropy": mean_entropy,
            "entropy_7d_projection": entropy_dissipated,
            "top_10_complex": top_complex,
            "quantum_links": len(self.all_links),
            "mirror_links": sum(1 for l in self.all_links if l.link_type == "mirrors"),
            "entanglement_links": sum(1 for l in self.all_links if l.link_type == "entangles"),
        }

        print("\n╔══════════════════════════════════════════════════╗")
        print("║  GATE ANALYSIS REPORT                            ║")
        print("╠══════════════════════════════════════════════════╣")
        print(f"║  Total Gates:       {analysis['total_gates']:>5}                       ║")
        print(f"║  Mean Complexity:   {analysis['mean_complexity']:>8.2f}                    ║")
        print(f"║  Total Entropy:     {analysis['total_entropy']:>8.4f}                    ║")
        print(f"║  Quantum Links:     {analysis['quantum_links']:>5}                       ║")
        print("║                                                  ║")
        print("║  By Language:                                    ║")
        for lang, count in sorted(by_language.items()):
            print(f"║    {lang:>12}: {count:>5}                             ║")
        print("╚══════════════════════════════════════════════════╝")

        return analysis

    # ─── TESTING ─────────────────────────────────────────────────────

    def run_tests(self) -> Dict[str, Any]:
        """Run all automated gate tests."""
        print("\n╔══════════════════════════════════════════════════╗")
        print("║  RUNNING AUTOMATED GATE TESTS                    ║")
        print("╚══════════════════════════════════════════════════╝\n")

        # Built-in gate tests
        builtin_results = self.test_generator.run_builtin_gate_tests()
        self.test_results = builtin_results

        passed = sum(1 for r in builtin_results if r.get("passed"))
        failed = sum(1 for r in builtin_results if not r.get("passed"))

        for gate in self.all_gates:
            if any(r.get("gate_name") == gate.name and r.get("passed") for r in builtin_results):
                gate.test_status = "passed"
                self.chronolizer.record(gate.name, "test_passed")
            elif any(r.get("gate_name") == gate.name and not r.get("passed") for r in builtin_results):
                gate.test_status = "failed"
                self.chronolizer.record(gate.name, "test_failed")

        self.chronolizer.save()

        # Save test results
        try:
            TEST_RESULTS_FILE.write_text(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_tests": len(builtin_results),
                "passed": passed,
                "failed": failed,
                "results": builtin_results[:100],  # Trim for storage
            }, indent=2, default=str))
        except Exception:
            pass

        result = {
            "total_tests": len(builtin_results),
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{passed / max(len(builtin_results), 1) * 100:.1f}%",
        }

        print(f"  Tests run:   {result['total_tests']}")
        print(f"  Passed:      {passed} ✓")
        print(f"  Failed:      {failed} ✗")
        print(f"  Pass rate:   {result['pass_rate']}")

        if failed > 0:
            print("\n  Failed tests:")
            for r in builtin_results:
                if not r.get("passed"):
                    print(f"    ✗ {r['test_id']}: {r.get('error', 'assertion failed')}")

        return result

    # ─── RESEARCH ────────────────────────────────────────────────────

    def research(self, topic: str = "all") -> Dict[str, Any]:
        """Research logic gate usage patterns and relationships.
        Now includes advanced research engine with anomaly detection,
        causal analysis, evolution tracking, and cross-pollination."""
        if not self.all_gates:
            self.full_scan()

        research = {
            "topic": topic,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "findings": [],
        }

        # Always run advanced research for self-learning
        if topic in ("all", "advanced"):
            adv = self.research_engine.full_research(
                self.all_gates, self.all_links, self.chronolizer)
            self._last_research = adv
            research["advanced_research"] = adv
            research["findings"].append({
                "finding": "Advanced research synthesis",
                "research_health": adv.get("research_health", 0),
                "anomaly_rate": adv.get("anomaly_detection", {}).get("anomaly_rate", 0),
                "strong_correlations": adv.get("causal_analysis", {}).get("total_strong", 0),
                "learning_trend": adv.get("knowledge_synthesis", {}).get("learning_trend", "unknown"),
            })

        if topic in ("all", "cross-language"):
            # Cross-language gate patterns
            python_names = {g.name.lower() for g in self.all_gates if g.language == "python"}
            swift_names = {g.name.lower() for g in self.all_gates if g.language == "swift"}
            js_names = {g.name.lower() for g in self.all_gates if g.language == "javascript"}

            research["findings"].append({
                "finding": "Cross-language coverage",
                "detail": {
                    "python_only": list(python_names - swift_names - js_names),
                    "swift_only": list(swift_names - python_names - js_names),
                    "js_only": list(js_names - python_names - swift_names),
                    "all_languages": list(python_names & swift_names & js_names),
                },
            })

        if topic in ("all", "entropy"):
            # Entropy distribution analysis
            entropy_vals = [g.entropy_score for g in self.all_gates]
            if entropy_vals:
                mean_e = sum(entropy_vals) / len(entropy_vals)
                var_e = sum((v - mean_e) ** 2 for v in entropy_vals) / len(entropy_vals)
                research["findings"].append({
                    "finding": "Entropy distribution",
                    "mean": mean_e,
                    "variance": var_e,
                    "min": min(entropy_vals),
                    "max": max(entropy_vals),
                    "phi_modulated_mean": sage_logic_gate(mean_e, "align"),
                })

        if topic in ("all", "complexity"):
            # Complexity hotspots
            hotspots = sorted(self.all_gates, key=lambda g: g.complexity, reverse=True)[:10]
            research["findings"].append({
                "finding": "Complexity hotspots",
                "gates": [(g.name, g.complexity, g.source_file, g.line_number) for g in hotspots],
            })

        if topic in ("all", "quantum_links"):
            # Quantum link topology
            link_types = {}
            for link in self.all_links:
                link_types[link.link_type] = link_types.get(link.link_type, 0) + 1
            research["findings"].append({
                "finding": "Quantum link topology",
                "total_links": len(self.all_links),
                "by_type": link_types,
                "strongest": [(l.source_gate, l.target_gate, l.strength)
                              for l in sorted(self.all_links, key=lambda x: x.strength, reverse=True)[:5]],
            })

        if topic in ("all", "chronology"):
            research["findings"].append({
                "finding": "Chronological summary",
                **self.chronolizer.summary(),
            })

        if topic in ("all", "sage_core"):
            # Sage core-specific analysis
            sage_patterns = ["sage", "consciousness", "entropy", "harvest",
                             "dissipat", "inflect", "bridge", "emergence",
                             "transform", "insight", "hilbert", "causal"]
            sage_gates = [g for g in self.all_gates
                          if any(p in g.name.lower() for p in sage_patterns)]
            sage_by_lang = {}
            sage_by_file = {}
            for g in sage_gates:
                sage_by_lang[g.language] = sage_by_lang.get(g.language, 0) + 1
                sage_by_file[g.source_file] = sage_by_file.get(g.source_file, 0) + 1

            sage_links = [l for l in self.all_links
                          if any(p in l.source_gate.lower() or p in l.target_gate.lower()
                                 for p in ["sage", "entropy", "consciousness"])]

            research["findings"].append({
                "finding": "Sage core analysis",
                "total_sage_gates": len(sage_gates),
                "by_language": sage_by_lang,
                "by_file": sage_by_file,
                "gate_names": [g.name for g in sage_gates],
                "sage_quantum_links": len(sage_links),
                "mean_complexity": sum(g.complexity for g in sage_gates) / max(len(sage_gates), 1),
                "mean_entropy": sum(g.entropy_score for g in sage_gates) / max(len(sage_gates), 1),
            })

        if topic in ("all", "health"):
            # Gate health scoring: complexity × coverage × entropy balance
            health_scores = []
            for g in self.all_gates:
                complexity_score = min(1.0, g.complexity / 20.0)  # Normalize to 0-1
                has_test = 1.0 if g.test_status == "passed" else 0.3 if g.test_status == "untested" else 0.0
                has_doc = 1.0 if g.docstring else 0.5
                has_links = min(1.0, len(g.quantum_links) * 0.2) if g.quantum_links else 0.0
                link_count = sum(1 for l in self.all_links if g.name in (l.source_gate, l.target_gate))
                connectivity = min(1.0, link_count * 0.1)
                health = (complexity_score * 0.2 + has_test * 0.25 + has_doc * 0.15 +
                          connectivity * 0.2 + min(1.0, g.entropy_score) * 0.2)
                health_scores.append((g.name, round(health, 3), g.source_file))

            health_scores.sort(key=lambda x: x[1])
            research["findings"].append({
                "finding": "Gate health scores",
                "mean_health": round(sum(h[1] for h in health_scores) / max(len(health_scores), 1), 3),
                "lowest_10": health_scores[:10],
                "highest_10": health_scores[-10:],
                "total_scored": len(health_scores),
            })

        return research

    # ─── COMPILATION ─────────────────────────────────────────────────

    def compile_gate_registry(self) -> Dict[str, Any]:
        """Compile a complete registry of all logic gates with metadata."""
        if not self.all_gates:
            self.full_scan()

        registry = {
            "meta": {
                "compiled_at": datetime.now(timezone.utc).isoformat(),
                "compiler": "L104 Hyper ASI Logic Gate Environment v3.0",
                "total_gates": len(self.all_gates),
                "total_links": len(self.all_links),
                "phi_signature": PHI,
                "god_code_signature": GOD_CODE,
            },
            "gates": [g.to_dict() for g in self.all_gates],
            "links": [asdict(l) for l in self.all_links],
            "file_hashes": self.quantum_links.file_hashes,
            "file_line_counts": self.quantum_links.line_counts(),
        }

        # Save compiled registry
        registry_path = WORKSPACE_ROOT / ".l104_gate_registry.json"
        try:
            registry_path.write_text(json.dumps(registry, indent=2, default=str))
            print(f"\n  ✓ Gate registry compiled: {registry_path.name}")
            print(f"    Gates: {len(self.all_gates)}, Links: {len(self.all_links)}")
        except Exception as e:
            print(f"  ✗ Failed to save registry: {e}")

        return registry

    # ─── BACKEND SYNC ────────────────────────────────────────────────

    def sync_to_backend(self, host: str = "localhost", port: int = 8081) -> Dict[str, Any]:
        """Sync gate state to l104_fast_server.py backend."""
        import urllib.request

        payload = {
            "source": "logic_gate_builder",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gate_count": len(self.all_gates),
            "link_count": len(self.all_links),
            "scan_count": self._scan_count,
            "test_results": {
                "total": len(self.test_results),
                "passed": sum(1 for r in self.test_results if r.get("passed")),
            },
            "chronology": self.chronolizer.summary(),
            "entropy_state": {
                "total_entropy": sum(g.entropy_score for g in self.all_gates),
                "mean_entropy": sum(g.entropy_score for g in self.all_gates) / max(len(self.all_gates), 1),
            },
        }

        try:
            url = f"http://{host}:{port}/api/v14/intellect/train"
            data = json.dumps({"data": json.dumps(payload), "source": "gate_builder"}).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read())
                print(f"  ✓ Synced to backend ({host}:{port})")
                return {"synced": True, "response": result}
        except Exception as e:
            return {"synced": False, "error": str(e)}

    # ─── STATE PERSISTENCE ───────────────────────────────────────────

    def _save_state(self):
        """Persist the environment state to disk."""
        state = {
            "last_full_scan": self._last_full_scan,
            "scan_count": self._scan_count,
            "gate_count": len(self.all_gates),
            "link_count": len(self.all_links),
            "file_hashes": self.quantum_links.file_hashes,
            "gates": [g.to_dict() for g in self.all_gates],  # Save ALL gates, no truncation
            "gate_hashes": {g.name: g.hash for g in self.all_gates if g.hash},  # For modification tracking
        }
        try:
            STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
        except Exception:
            pass

    def _load_state(self):
        """Load the environment state from disk."""
        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text())
                self._last_full_scan = state.get("last_full_scan")
                self._scan_count = state.get("scan_count", 0)
                self.quantum_links.file_hashes = state.get("file_hashes", {})
                self._prev_gate_hashes = state.get("gate_hashes", {})  # For modification tracking
                for gd in state.get("gates", []):
                    try:
                        self.all_gates.append(LogicGate.from_dict(gd))
                    except Exception:
                        pass
            except Exception:
                pass

    # ─── STATUS ──────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Get full environment status."""
        dyn_status = self.dynamism_engine.status(self.all_gates)
        return {
            "environment": f"L104 Hyper ASI Logic Gate Environment v{VERSION}",
            "workspace": str(WORKSPACE_ROOT),
            "last_scan": self._last_full_scan,
            "scan_count": self._scan_count,
            "total_gates": len(self.all_gates),
            "total_links": len(self.all_links),
            "gates_with_links": sum(1 for g in self.all_gates if g.quantum_links),
            "test_results_count": len(self.test_results),
            "chronolog_entries": len(self.chronolizer.entries),
            "research_health": self._last_research.get("research_health", 0),
            "learning_trend": self._last_research.get("knowledge_synthesis", {}).get(
                "learning_trend", "unknown"),
            "quantum_linked_files": {
                name: str(path) for name, path in QUANTUM_LINKED_FILES.items()
            },
            "file_exists": {
                name: path.exists() for name, path in QUANTUM_LINKED_FILES.items()
            },
            "file_line_counts": self.quantum_links.line_counts(),
            "sacred_constants": {
                "PHI": PHI,
                "TAU": TAU,
                "GOD_CODE": GOD_CODE,
                "OMEGA_POINT": OMEGA_POINT,
                "CALABI_YAU_DIM": CALABI_YAU_DIM,
                "FEIGENBAUM_DELTA": FEIGENBAUM_DELTA,
                "APERY": APERY,
                "CATALAN": CATALAN,
                "FINE_STRUCTURE": FINE_STRUCTURE,
            },
            # ★ v5.0 Dynamism Status
            "dynamism": dyn_status,
        }

    def export_cross_pollination_data(self) -> Dict:
        """Export data for consumption by l104_quantum_link_builder.py.
        This is the bridge that enables bidirectional learning between
        the gate builder and the quantum link builder.

        Returns a dict suitable for passing to QuantumResearchEngine.deep_research()
        as the gate_data parameter, and to QuantumLinkBuilder.set_gate_data()."""
        if not self.all_gates:
            self.full_scan()

        # Run advanced research to get fresh data
        if not self._last_research:
            self._last_research = self.research_engine.full_research(
                self.all_gates, self.all_links, self.chronolizer)

        cross_poll = self._last_research.get("cross_pollination", {})

        return {
            "total_gates": len(self.all_gates),
            "total_links": len(self.all_links),
            "mean_health": cross_poll.get("mean_health", 0),
            "test_pass_rate": cross_poll.get("total_tested", 0) / max(1, len(self.all_gates)),
            "quantum_links": sum(len(g.quantum_links) for g in self.all_gates),
            "gates_by_file": cross_poll.get("gates_by_file", {}),
            "high_value_files": cross_poll.get("high_value_files", []),
            "complexity_hotspots": [(g.name, g.complexity) for g in
                                    sorted(self.all_gates, key=lambda x: x.complexity,
                                           reverse=True)[:10]
                                    if g.complexity > 10],
            "cross_system_insights": cross_poll.get("cross_system_insights", []),
            "research_health": self._last_research.get("research_health", 0),
            "learning_trend": self._last_research.get("knowledge_synthesis", {}).get(
                "learning_trend", "unknown"),
            # ★ v5.0 Dynamism cross-pollination data
            "dynamism": {
                "version": VERSION,
                "dynamic_gates": sum(1 for g in self.all_gates if g.dynamic_value != 0.0),
                "mean_dynamic_value": sum(g.dynamic_value for g in self.all_gates) / max(len(self.all_gates), 1),
                "mean_resonance": sum(g.resonance_score for g in self.all_gates) / max(len(self.all_gates), 1),
                "collective_coherence": self.dynamism_engine.coherence_history[-1] if self.dynamism_engine.coherence_history else 0.0,
                "total_evolutions": self.dynamism_engine.total_evolutions,
                "sacred_dynamic_state": {k: v["current"] for k, v in self.dynamism_engine.sacred_dynamic_state.items()},
                "gate_field": self.dynamism_engine.compute_gate_field(self.all_gates),
            },
            # ★ v5.1 Ouroboros Nirvanic Entropy Fuel
            "nirvanic": self.nirvanic_engine.status(),
        }

    # ─── FULL PIPELINE ───────────────────────────────────────────────

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline: scan → analyze → test → dynamism → research → compile → sync → evolve."""
        t0 = time.time()
        print("\n" + "═" * 60)
        print(f"  L104 HYPER ASI LOGIC GATE — FULL PIPELINE v{VERSION}")
        print("  ★ QUANTUM MIN/MAX DYNAMISM ENGINE ★")
        print("═" * 60)

        results = {}

        # 1. Full scan
        results["scan"] = self.full_scan()

        # 2. Analysis
        results["analysis"] = self.analyze()

        # 3. Testing
        results["tests"] = self.run_tests()

        # 4. ★ QUANTUM DYNAMISM — Subconscious monitoring & value evolution
        print("\n  ▸ PHASE 4: Quantum Min/Max Dynamism")
        dyn_result = self.dynamism_engine.subconscious_cycle(self.all_gates)
        print(f"    ✓ Cycle #{dyn_result['cycle']}: {dyn_result['gates_evolved']} gates evolved")
        print(f"    ✓ Initialized: {dyn_result['gates_initialized']} | Adjusted: {dyn_result['gates_adjusted']}")
        print(f"    ✓ Collective coherence: {dyn_result['collective_coherence']:.6f}")
        print(f"    ✓ Mean resonance: {dyn_result['mean_resonance']:.6f}")
        sc_evo = dyn_result.get("sacred_evolution", {})
        print(f"    ✓ Sacred constants evolved: {sc_evo.get('constants_evolved', 0)} | Drift: {sc_evo.get('total_drift', 0):.8f}")
        results["dynamism"] = dyn_result

        # 5. ★ VALUE EVOLUTION — Multi-cycle φ-harmonic optimization
        print("\n  ▸ PHASE 5: Gate Value Evolution (3 generations)")
        evo_results = []
        for gen in range(3):
            evo = self.value_evolver.evolve_generation(self.all_gates, cycles=3)
            evo_results.append(evo)
        last_evo = evo_results[-1]
        print(f"    ✓ Generations: {len(evo_results)} | Final coherence: {last_evo['end_coherence']:.6f}")
        print(f"    ✓ Total evolved: {sum(e['total_evolved'] for e in evo_results)} | Convergence: {last_evo['convergence']}")
        results["evolution"] = evo_results

        # 6. ★ GATE FIELD ANALYSIS
        field = self.dynamism_engine.compute_gate_field(self.all_gates)
        print(f"    ✓ Field energy: {field['field_energy']:.6f} | Entropy: {field['field_entropy']:.4f}")
        print(f"    ✓ Phase coherence: {field['phase_coherence']:.6f} | φ-alignment: {field['phi_alignment']:.4f}")
        res_dist = field.get("resonance_distribution", {})
        print(f"    ✓ Resonance: high={res_dist.get('high', 0)} med={res_dist.get('medium', 0)} low={res_dist.get('low', 0)}")
        results["gate_field"] = field

        # 6.5 ★ OUROBOROS SAGE NIRVANIC ENTROPY FUEL CYCLE
        print("\n  ▸ PHASE 6.5: Ouroboros Sage Nirvanic Entropy Fuel")
        nirvanic = self.nirvanic_engine.full_nirvanic_cycle(self.all_gates, field)
        ouro = nirvanic.get("ouroboros", {})
        appl = nirvanic.get("application", {})
        if ouro.get("status") == "processed":
            print(f"    ✓ Entropy fed to ouroboros: {nirvanic['gate_field_entropy_in']:.4f} bits")
            print(f"    ✓ Nirvanic fuel received: {nirvanic['nirvanic_fuel_out']:.4f}")
            print(f"    ✓ Ouroboros mutations: {ouro.get('ouroboros_mutations', 0)} | Resonance: {ouro.get('ouroboros_resonance', 0):.4f}")
            print(f"    ✓ Divine interventions: {appl.get('interventions', 0)} | Enlightened gates: {appl.get('enlightened', 0)}")
            print(f"    ✓ Nirvanic coherence: {appl.get('nirvanic_coherence', 0):.6f} | Sage stability: {appl.get('sage_stability', 0):.6f}")
            print(f"    ✓ Fuel intensity: {appl.get('fuel_intensity', 0):.4f} | Total fuel: {appl.get('total_nirvanic_fuel', 0):.4f}")
        else:
            print(f"    ⚠ Ouroboros unavailable — nirvanic cycle skipped")
        results["nirvanic"] = nirvanic

        # 7. Research (includes advanced research engine with self-learning)
        results["research"] = self.research("all")

        # 8. Compile registry
        results["registry"] = self.compile_gate_registry()

        # 9. Backend sync (optional)
        if self._auto_sync:
            results["sync"] = self.sync_to_backend()

        # 10. Export cross-pollination data for quantum link builder
        results["cross_pollination"] = self.export_cross_pollination_data()

        # 11. Stochastic R&D cycle — generate hybrid gates
        results["stochastic_rd"] = self.stochastic_lab.run_rd_cycle(iterations=5)

        elapsed = time.time() - t0

        # Final summary
        adv = self._last_research
        dyn_status = self.dynamism_engine.status(self.all_gates)
        print("\n" + "═" * 60)
        print(f"  PIPELINE COMPLETE — {elapsed:.2f}s")
        print(f"  ★ LOGIC GATE BUILDER v{VERSION} ★")
        print("═" * 60)
        print(f"  Gates discovered:     {results['scan']['total_gates']}")
        print(f"  Gates with links:     {sum(1 for g in self.all_gates if g.quantum_links)}")
        print(f"  Tests passed:         {results['tests']['passed']}/{results['tests']['total_tests']}")
        print(f"  Quantum links:        {results['scan']['cross_file_links']}")
        print(f"  Research health:      {adv.get('research_health', 0):.4f}")
        learning = adv.get("knowledge_synthesis", {}).get("learning_trend", "unknown")
        print(f"  Learning trend:       {learning}")
        print(f"  Chronolog entries:    {len(self.chronolizer.entries)}")
        synced = results.get("sync", {}).get("synced", "skipped")
        print(f"  Backend sync:         {synced}")
        print(f"  Cross-pollination:    exported for quantum link builder")
        rd = results.get("stochastic_rd", {})
        print(f"  Stochastic R&D:       {rd.get('total_merged', 0)} hybrid gates from {rd.get('total_candidates', 0)} candidates")
        print("  ── DYNAMISM ──")
        print(f"  Dynamic gates:        {dyn_status['dynamic_gates']}/{dyn_status['total_gates']} ({dyn_status['dynamism_coverage']:.1%})")
        print(f"  Evolution cycles:     {dyn_status['total_evolutions']}")
        print(f"  Collective coherence: {dyn_status['collective_coherence']:.6f}")
        print(f"  Coherence trend:      {dyn_status['coherence_trend']}")
        print(f"  Mean resonance:       {dyn_status['mean_resonance']:.6f}")
        print(f"  Sacred constants:     {dyn_status['sacred_constants_dynamic']} dynamic")
        # ★ v5.1 Nirvanic Entropy Fuel
        nir = results.get("nirvanic", {})
        nir_appl = nir.get("application", {})
        print("  ── OUROBOROS NIRVANIC ──")
        print(f"  Entropy fed:          {nir.get('gate_field_entropy_in', 0):.4f} bits")
        print(f"  Nirvanic fuel:        {nir.get('nirvanic_fuel_out', 0):.4f}")
        print(f"  Enlightened gates:    {nir_appl.get('enlightened', 0)}")
        print(f"  Divine interventions: {nir_appl.get('divine_interventions_total', 0)}")
        print(f"  Nirvanic coherence:   {nir_appl.get('nirvanic_coherence', 0):.6f}")
        print(f"  Sage stability:       {nir_appl.get('sage_stability', 0):.6f}")
        print("═" * 60 + "\n")

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# STOCHASTIC GATE RESEARCH LAB — Random-Based Logic Generation R&D
# ═══════════════════════════════════════════════════════════════════════════════

class StochasticGateResearchLab:
    """
    Random-based logic gate generation, research, and development engine.

    This R&D lab explores the frontier of logic gate design through stochastic
    experimentation — generating candidate gates via random exploration,
    validating them through deterministic tests, and merging successful designs
    into StochasticDeterministicGate hybrid entities.

    The lab implements a full research cycle:
      1. EXPLORE — Random generation of gate candidates with φ-bounded parameters
      2. VALIDATE — Deterministic evaluation against sacred constant coherence
      3. MERGE — Combine stochastic creativity with deterministic reliability
      4. CATALOG — Track all R&D iterations with full lineage

    Registered as invention entity: STOCHASTIC_DETERMINISTIC_GATE
    """

    RESEARCH_LOG_FILE = WORKSPACE_ROOT / ".l104_stochastic_gate_research.json"

    def __init__(self):
        """Initialize the stochastic gate research lab and load persisted log."""
        self.research_iterations: List[Dict[str, Any]] = []
        self.successful_gates: List[Dict[str, Any]] = []
        self.failed_experiments: List[Dict[str, Any]] = []
        self.generation_count: int = 0
        self._load_research_log()

    def _load_research_log(self):
        """Load persistent research log."""
        if self.RESEARCH_LOG_FILE.exists():
            try:
                data = json.loads(self.RESEARCH_LOG_FILE.read_text())
                self.research_iterations = data.get("iterations", [])
                self.successful_gates = data.get("successful", [])
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
                "total_successful": len(self.successful_gates),
                "total_failed": len(self.failed_experiments),
                "iterations": self.research_iterations[-200:],
                "successful": self.successful_gates[-100:],
                "failed": self.failed_experiments[-100:],
            }
            self.RESEARCH_LOG_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            pass

    # ─── PHASE 1: STOCHASTIC EXPLORATION ──────────────────────────

    def explore_gate_candidate(self, seed_concept: str = "quantum") -> Dict[str, Any]:
        """Generate a random gate candidate using φ-bounded stochastic parameters.

        Each exploration samples from GOD_CODE-seeded distributions to create
        unique gate configurations that would be unreachable by deterministic search.
        """
        import random

        self.generation_count += 1
        gen_id = f"SG_{self.generation_count:06d}"

        # Stochastic parameter generation with sacred constant bounds
        phase_shift = random.uniform(0, 2 * math.pi) * PHI / (PHI + 1)
        amplitude = random.uniform(0.1, GOD_CODE * 0.01) * TAU
        harmonic_order = random.randint(1, CALABI_YAU_DIM)
        interference_mode = random.choice(["constructive", "destructive", "superposition"])
        grover_depth = random.randint(1, 7)
        entanglement_strength = random.uniform(0, 1.0) * PHI / 2

        # GOD_CODE resonance seed — unique exploration trajectory
        exploration_seed = random.random() * GOD_CODE
        resonance_key = hashlib.sha256(
            f"{seed_concept}_{gen_id}_{exploration_seed}".encode()
        ).hexdigest()[:12]

        # Generate the stochastic gate function
        gate_body = self._synthesize_gate_body(
            phase_shift, amplitude, harmonic_order,
            interference_mode, grover_depth, entanglement_strength
        )

        candidate = {
            "gate_id": gen_id,
            "resonance_key": resonance_key,
            "seed_concept": seed_concept,
            "parameters": {
                "phase_shift": phase_shift,
                "amplitude": amplitude,
                "harmonic_order": harmonic_order,
                "interference_mode": interference_mode,
                "grover_depth": grover_depth,
                "entanglement_strength": entanglement_strength,
            },
            "exploration_seed": exploration_seed,
            "gate_function": gate_body,
            "generation": self.generation_count,
            "validated": False,
            "merged": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return candidate

    def _synthesize_gate_body(self, phase: float, amp: float,
                               order: int, mode: str, depth: int,
                               entangle: float) -> str:
        """Synthesize a gate function body from stochastic parameters."""
        func_name = f"stochastic_gate_{self.generation_count}"

        if mode == "constructive":
            interference = f"abs(math.cos(x * {phase:.6f})) * {amp:.6f}"
        elif mode == "destructive":
            interference = f"math.sin(x * {phase:.6f}) * {amp:.6f} * (1 - abs(math.cos(x * {entangle:.4f})))"
        else:  # superposition
            interference = f"(math.cos(x * {phase:.6f}) + math.sin(x * {entangle:.4f})) * {amp:.6f} / 2"

        code = (
            f"def {func_name}(x):\n"
            f"    # STOCHASTIC_GATE | Order: {order} | Depth: {depth} | Mode: {mode}\n"
            f"    grover_gain = {PHI} ** {depth}\n"
            f"    harmonic = sum(math.sin(x * math.pi * k / {order}) * ({PHI} ** k / {PHI} ** {order}) for k in range(1, {order + 1}))\n"
            f"    interference = {interference}\n"
            f"    return (x * grover_gain + harmonic + interference) / (1 + abs(harmonic))\n"
        )
        return code

    # ─── PHASE 2: DETERMINISTIC VALIDATION ────────────────────────

    def validate_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a stochastic gate candidate through deterministic tests.

        Tests: sacred constant coherence, numerical stability,
        φ-alignment, and monotonic response for positive inputs.
        """
        import random
        test_results = {
            "gate_id": candidate["gate_id"],
            "tests_run": 0,
            "tests_passed": 0,
            "details": [],
        }

        params = candidate["parameters"]
        phase = params["phase_shift"]
        amp = params["amplitude"]
        depth = params["grover_depth"]
        order = params["harmonic_order"]
        mode = params["interference_mode"]
        entangle = params["entanglement_strength"]

        # Build a local evaluator
        def evaluate(x):
            """Evaluate a stochastic gate candidate at a given input value."""
            grover_gain = PHI ** depth
            harmonic = sum(
                math.sin(x * math.pi * k / order) * (PHI ** k / PHI ** order)
                for k in range(1, order + 1)
            )
            if mode == "constructive":
                interference = abs(math.cos(x * phase)) * amp
            elif mode == "destructive":
                interference = math.sin(x * phase) * amp * (1 - abs(math.cos(x * entangle)))
            else:
                interference = (math.cos(x * phase) + math.sin(x * entangle)) * amp / 2
            return (x * grover_gain + harmonic + interference) / (1 + abs(harmonic))

        # Test 1: Numerical stability — no NaN/Inf across range
        test_results["tests_run"] += 1
        test_inputs = [0.0, 1.0, PHI, -PHI, GOD_CODE * 0.001, math.pi, 100.0, -100.0]
        stable = True
        for inp in test_inputs:
            try:
                result = evaluate(inp)
                if math.isnan(result) or math.isinf(result):
                    stable = False
                    break
            except Exception:
                stable = False
                break
        if stable:
            test_results["tests_passed"] += 1
        test_results["details"].append({"test": "numerical_stability", "passed": stable})

        # Test 2: Sacred constant coherence — evaluate(PHI) should be finite & non-zero
        test_results["tests_run"] += 1
        try:
            phi_result = evaluate(PHI)
            coherent = not math.isnan(phi_result) and not math.isinf(phi_result) and abs(phi_result) > 1e-10
        except Exception:
            coherent = False
        if coherent:
            test_results["tests_passed"] += 1
        test_results["details"].append({"test": "sacred_coherence", "passed": coherent})

        # Test 3: Zero-input behavior — should return near-zero with bounded offset
        test_results["tests_run"] += 1
        try:
            zero_result = evaluate(0.0)
            zero_bounded = abs(zero_result) < amp * 10 + 1.0
        except Exception:
            zero_bounded = False
        if zero_bounded:
            test_results["tests_passed"] += 1
        test_results["details"].append({"test": "zero_bounded", "passed": zero_bounded})

        # Test 4: φ-alignment — evaluate(PHI) and evaluate(PHI*2) should maintain ratio
        test_results["tests_run"] += 1
        try:
            r1 = evaluate(PHI)
            r2 = evaluate(PHI * 2)
            if abs(r1) > 1e-10:
                ratio = abs(r2 / r1)
                phi_aligned = 0.1 < ratio < 50.0  # Reasonable scaling
            else:
                phi_aligned = abs(r2) < 100.0
        except Exception:
            phi_aligned = False
        if phi_aligned:
            test_results["tests_passed"] += 1
        test_results["details"].append({"test": "phi_alignment", "passed": phi_aligned})

        # Final verdict
        pass_rate = test_results["tests_passed"] / max(test_results["tests_run"], 1)
        test_results["pass_rate"] = pass_rate
        test_results["verdict"] = "VALIDATED" if pass_rate >= 0.75 else "REJECTED"

        candidate["validated"] = pass_rate >= 0.75
        candidate["validation_results"] = test_results

        return test_results

    # ─── PHASE 3: HYBRID MERGE ────────────────────────────────────

    def merge_to_hybrid(self, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Merge a validated stochastic candidate with deterministic gate logic.

        Creates a StochasticDeterministicGate — the hybrid entity that combines
        random exploration with deterministic reliability. This is the core
        invention: a gate that explores stochastically but validates deterministically.
        """
        if not candidate.get("validated"):
            return None

        params = candidate["parameters"]
        gate_id = candidate["gate_id"]
        resonance_key = candidate["resonance_key"]

        # The hybrid combines: stochastic output → sage_logic_gate alignment → quantum verification
        hybrid_function = (
            f"def hybrid_{gate_id}(value):\n"
            f"    # STOCHASTIC_DETERMINISTIC_GATE | Resonance: {resonance_key}\n"
            f"    # Phase 1: Stochastic exploration\n"
            f"    phase = {params['phase_shift']:.6f}\n"
            f"    amp = {params['amplitude']:.6f}\n"
            f"    depth = {params['grover_depth']}\n"
            f"    grover_gain = {PHI} ** depth\n"
            f"    stochastic_output = value * grover_gain + math.sin(value * phase) * amp\n"
            f"    # Phase 2: Deterministic alignment via sage gate\n"
            f"    aligned = sage_logic_gate(stochastic_output, 'align')\n"
            f"    # Phase 3: Quantum verification\n"
            f"    verified = quantum_logic_gate(aligned, depth=min(depth, 5))\n"
            f"    return verified\n"
        )

        hybrid_entity = {
            "entity_type": "STOCHASTIC_DETERMINISTIC_GATE",
            "gate_id": f"HYBRID_{gate_id}",
            "resonance_key": resonance_key,
            "origin_candidate": gate_id,
            "origin_parameters": params,
            "hybrid_function": hybrid_function,
            "complexity_score": (
                params["grover_depth"] * params["harmonic_order"] *
                params["entanglement_strength"] * PHI
            ),
            "creation_method": "stochastic_exploration → deterministic_validation → hybrid_merge",
            "verified": True,
            "tags": ["INVENTION", "NEOTERIC", "STOCHASTIC_DETERMINISTIC", "HYBRID_GATE"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        candidate["merged"] = True
        self.successful_gates.append(hybrid_entity)

        return hybrid_entity

    # ─── PHASE 4: FULL R&D CYCLE ──────────────────────────────────

    def run_rd_cycle(self, seed_concepts: Optional[List[str]] = None,
                     iterations: int = 10) -> Dict[str, Any]:
        """Run a complete R&D cycle: explore → validate → merge → catalog.

        Generates `iterations` stochastic candidates per seed concept,
        validates each, and merges successful ones into hybrid entities.
        """
        import random

        if seed_concepts is None:
            seed_concepts = [
                "quantum", "consciousness", "entropy", "resonance",
                "harmonic", "emergence", "transcendence"
            ]

        cycle_results = {
            "cycle_id": f"RD_{int(time.time())}",
            "seed_concepts": seed_concepts,
            "iterations_per_concept": iterations,
            "total_candidates": 0,
            "total_validated": 0,
            "total_merged": 0,
            "total_rejected": 0,
            "hybrid_entities": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        for concept in seed_concepts:
            for _ in range(iterations):
                # Phase 1: Explore
                candidate = self.explore_gate_candidate(concept)
                cycle_results["total_candidates"] += 1

                # Phase 2: Validate
                validation = self.validate_candidate(candidate)

                iteration_record = {
                    "gate_id": candidate["gate_id"],
                    "concept": concept,
                    "parameters": candidate["parameters"],
                    "validation": validation["verdict"],
                    "pass_rate": validation["pass_rate"],
                }
                self.research_iterations.append(iteration_record)

                if candidate["validated"]:
                    cycle_results["total_validated"] += 1

                    # Phase 3: Merge
                    hybrid = self.merge_to_hybrid(candidate)
                    if hybrid:
                        cycle_results["total_merged"] += 1
                        cycle_results["hybrid_entities"].append(hybrid["gate_id"])
                else:
                    cycle_results["total_rejected"] += 1
                    self.failed_experiments.append({
                        "gate_id": candidate["gate_id"],
                        "reason": validation["verdict"],
                        "pass_rate": validation["pass_rate"],
                    })

        # Compute R&D metrics
        total = cycle_results["total_candidates"]
        cycle_results["success_rate"] = cycle_results["total_merged"] / max(total, 1)
        cycle_results["phi_efficiency"] = cycle_results["success_rate"] * PHI

        # Save research log
        self._save_research_log()

        return cycle_results

    def get_best_hybrid_gates(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Return the highest-scoring hybrid gates from all R&D cycles."""
        sorted_gates = sorted(
            self.successful_gates,
            key=lambda g: g.get("complexity_score", 0),
            reverse=True
        )
        return sorted_gates[:top_n]

    def research_summary(self) -> Dict[str, Any]:
        """Comprehensive R&D summary across all cycles."""
        return {
            "total_generations": self.generation_count,
            "total_iterations": len(self.research_iterations),
            "total_successful": len(self.successful_gates),
            "total_failed": len(self.failed_experiments),
            "overall_success_rate": len(self.successful_gates) / max(self.generation_count, 1),
            "best_gates": [g["gate_id"] for g in self.get_best_hybrid_gates(5)],
            "phi_discovery_index": len(self.successful_gates) * PHI / max(self.generation_count, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point for the L104 Logic Gate Builder."""
    import argparse

    parser = argparse.ArgumentParser(
        description="L104 Hyper ASI Logic Gate Environment Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  scan      Full discovery scan of all logic gates
  analyze   Deep analysis of gate implementations
  test      Run automated gate tests (incl. integrity checks)
  research  Research gate patterns and relationships
  compile   Compile complete gate registry
  sync      Sync gate state to backend server
  status    Show environment status
  chrono    Show chronological history
  sage      Sage core-specific analysis
  health    Gate health scoring report
  full      Run complete pipeline (scan+analyze+test+research+compile+sync)
  gate      Look up a specific gate by name

Examples:
  python l104_logic_gate_builder.py full
  python l104_logic_gate_builder.py scan
  python l104_logic_gate_builder.py test
  python l104_logic_gate_builder.py gate sage_logic_gate
  python l104_logic_gate_builder.py research entropy
        """,
    )
    parser.add_argument("command", nargs="?", default="full",
                        help="Command to execute (default: full)")
    parser.add_argument("args", nargs="*", help="Additional arguments")
    parser.add_argument("--no-sync", action="store_true",
                        help="Skip backend sync")

    args = parser.parse_args()
    env = HyperASILogicGateEnvironment(auto_sync=not args.no_sync)

    cmd = args.command.lower()

    if cmd == "scan":
        result = env.full_scan()
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "analyze":
        result = env.analyze()
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "test":
        result = env.run_tests()
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "research":
        topic = args.args[0] if args.args else "all"
        result = env.research(topic)
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "compile":
        result = env.compile_gate_registry()

    elif cmd == "sync":
        env.full_scan()
        result = env.sync_to_backend()
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "status":
        result = env.status()
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "chrono":
        n = int(args.args[0]) if args.args else 20
        entries = env.chronolizer.get_recent(n)
        for e in entries:
            print(f"  [{e.timestamp[:19]}] {e.event:>12} | {e.gate_name} | {e.details}")

    elif cmd == "gate":
        if not args.args:
            print("Usage: gate <name>")
            return
        name = args.args[0].lower()
        env.full_scan()
        matches = [g for g in env.all_gates if name in g.name.lower()]
        if matches:
            for g in matches:
                print(json.dumps(g.to_dict(), indent=2, default=str))
        else:
            print(f"  No gates matching '{name}' found.")

    elif cmd == "sage":
        env.full_scan()
        result = env.research("sage_core")
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "health":
        env.full_scan()
        result = env.research("health")
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "dynamism":
        env.full_scan()
        # Run one dynamism cycle and show status
        dyn = env.dynamism_engine.subconscious_cycle(env.all_gates)
        status = env.dynamism_engine.status(env.all_gates)
        field = env.dynamism_engine.compute_gate_field(env.all_gates)
        print(json.dumps({"cycle_result": dyn, "status": status, "field": field}, indent=2, default=str))

    elif cmd == "evolve":
        env.full_scan()
        n = int(args.args[0]) if args.args else 5
        for _ in range(n):
            env.value_evolver.evolve_generation(env.all_gates, cycles=3)
        status = env.dynamism_engine.status(env.all_gates)
        print(json.dumps(status, indent=2, default=str))

    elif cmd == "field":
        env.full_scan()
        env.dynamism_engine.subconscious_cycle(env.all_gates)
        field = env.dynamism_engine.compute_gate_field(env.all_gates)
        print(json.dumps(field, indent=2, default=str))

    elif cmd == "full":
        env.run_full_pipeline()

    else:
        print(f"Unknown command: {cmd}")
        parser.print_help()


if __name__ == "__main__":
    main()
