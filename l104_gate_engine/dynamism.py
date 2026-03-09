"""L104 Gate Engine — Dynamism engine and value evolver."""

import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List

from .constants import (
    VERSION, PHI, TAU, DRIFT_ENVELOPE, SACRED_DYNAMIC_BOUNDS, WORKSPACE_ROOT,
)
from .models import LogicGate


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

    def subconscious_cycle(self, gates: List[LogicGate]) -> Dict[str, Any]:
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

    def status(self, gates: List[LogicGate]) -> Dict[str, Any]:
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

    def compute_gate_field(self, gates: List[LogicGate]) -> Dict[str, Any]:
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

    def evolve_generation(self, gates: List[LogicGate], cycles: int = 5) -> Dict[str, Any]:
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

    def _cross_pollinate_drift(self, gates: List[LogicGate]):
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
