"""
L104 Quantum Engine — Dynamism & Nirvanic Entropy Engines
═══════════════════════════════════════════════════════════════════════════════
LinkDynamismEngine: Quantum Min/Max value evolution (v4.0)
LinkOuroborosNirvanicEngine: Entropy → Ouroboros → Nirvanic fuel (v4.1)
"""

import json
import math
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from .constants import (
    PHI, PHI_GROWTH, GOD_CODE, LINK_DRIFT_ENVELOPE,
    LINK_SACRED_DYNAMIC_BOUNDS, WORKSPACE_ROOT,
)
from .models import QuantumLink


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

