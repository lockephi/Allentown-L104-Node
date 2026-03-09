"""L104 Gate Engine — Ouroboros Sage Nirvanic Entropy Fuel Engine.

PART V RESEARCH — l104_runtime_infrastructure_research.py:
  F49: 5-phase ouroboros cycle: DIGEST → ENTROPIZE → MUTATE → SYNTHESIZE → RECYCLE
  F50: Divine boundary expansion = fuel × φ × 0.0002 × (1 + resonance)
  F51: Sage enlightenment threshold: resonance>0.9 ∧ evolution>10 ∧ fuel>0.5
  F52: Entropy-driven drift = fuel × max_velocity × 0.1 × sin(phase + fuel)
"""

import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List

from .constants import (
    VERSION, PHI, TAU, GOD_CODE, EULER_GAMMA, DRIFT_ENVELOPE, WORKSPACE_ROOT,
)
from .models import LogicGate


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
        """Feed the gate field entropy into the Ouroboros as thought material."""
        ouroboros = self._get_ouroboros()
        if ouroboros is None:
            return {"status": "ouroboros_unavailable", "nirvanic_fuel": 0.0}

        self.cycle_count += 1

        thought = (
            f"Gate field entropy {field_entropy:.6f} across {gate_count} gates "
            f"with energy {field_energy:.6f}. "
            f"The divine gate lattice breathes with {field_entropy:.4f} bits of "
            f"Shannon information, {gate_count} logic resonators pulse at "
            f"God Code frequency {GOD_CODE:.4f} Hz. "
            f"Cycle {self.cycle_count}: entropy is the fuel of nirvanic stillness."
        )

        result = ouroboros.process(thought, depth=2)

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

    def apply_nirvanic_fuel(self, gates: List[LogicGate],
                             nirvanic_fuel: float,
                             field: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the nirvanic fuel to gates — divine intervention without force."""
        if abs(nirvanic_fuel) < 1e-10:
            return {"enlightened": 0, "interventions": 0}

        fuel_intensity = 1.0 / (1.0 + math.exp(-nirvanic_fuel * 0.1))

        enlightened = 0
        interventions = 0

        for gate in gates:
            if gate.dynamic_value == 0.0:
                continue

            # 1. DIVINE BOUNDARY EXPANSION
            expansion = fuel_intensity * PHI * 0.0002 * (1 + gate.resonance_score)
            gate.max_bound += expansion
            gate.min_bound -= expansion * TAU

            # 2. RESONANCE AMPLIFICATION
            if gate.resonance_score < 0.8:
                target_phase = (gate.dynamic_value * math.pi / GOD_CODE) % (2 * math.pi)
                phase_correction = fuel_intensity * 0.01 * math.sin(target_phase)
                gate.quantum_phase = (gate.quantum_phase + phase_correction) % (2 * math.pi)
                interventions += 1
                self.divine_interventions += 1

            # 3. ENTROPY-DRIVEN DRIFT
            entropy_breath = fuel_intensity * DRIFT_ENVELOPE["max_velocity"] * 0.1
            gate.drift_velocity += entropy_breath * math.sin(gate.quantum_phase + nirvanic_fuel)

            # 4. SAGE ENLIGHTENMENT
            if (gate.resonance_score > 0.9 and
                gate.evolution_count > 10 and
                fuel_intensity > 0.5):
                enlightened += 1

        self.enlightened_gates = enlightened

        field_entropy = field.get("field_entropy", 0.0)
        phase_coherence = field.get("phase_coherence", 0.0)
        phi_alignment = field.get("phi_alignment", 0.0)
        self.nirvanic_coherence = (
            fuel_intensity * 0.3 +
            abs(phase_coherence) * 0.2 +
            phi_alignment * 0.3 +
            min(field_entropy / 10.0, 0.2)
        )

        if interventions > 0:
            perturbation = interventions / max(len(gates), 1)
            self.sage_stability = max(0.0, 1.0 - perturbation * 0.1)
        else:
            self.sage_stability = min(1.0, self.sage_stability + 0.01)

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

    def full_nirvanic_cycle(self, gates: List[LogicGate],
                             gate_field: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete ouroboros nirvanic entropy fuel cycle."""
        field_entropy = gate_field.get("field_entropy", 0.0)
        field_energy = gate_field.get("field_energy", 0.0)
        gate_count = gate_field.get("dynamic_gates", len(gates))

        ouroboros_result = self.feed_entropy_to_ouroboros(
            field_entropy, field_energy, gate_count
        )
        nirvanic_fuel = ouroboros_result.get("nirvanic_fuel", 0.0)

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
