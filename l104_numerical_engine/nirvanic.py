"""L104 Numerical Engine — Ouroboros Sage Nirvanic Entropy Fuel Engine.

Token entropy → ouroboros → fuel → lattice → enlightened tokens → ∞

PART V RESEARCH — l104_runtime_infrastructure_research.py:
  F50: Divine boundary expansion = fuel × φ × 0.0002 × (1 + resonance)
  F51: Sage enlightenment threshold: resonance>0.9, evolution>10, fuel>0.5
  F52: Entropy-driven drift = fuel × max_velocity × 0.1 × sin(phase + fuel)
  F53: All multipliers ≥ 1 (no devolution)
  F84: 7-phase nerve + 5-phase nirvanic = 12 (CY₇ edges)
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, TYPE_CHECKING

from .precision import D
from .constants import WORKSPACE_ROOT, GOD_CODE_HP, PHI_HP

if TYPE_CHECKING:
    from .lattice import TokenLatticeEngine


class NumericalOuroborosNirvanicEngine:
    """Ouroboros Sage Nirvanic Entropy Fuel Engine for the Numerical Builder.

    The numerical builder has its own entropy landscape (Shannon entropy of
    binned log-values of all tokens). This entropy, combined with the peer
    builders' nirvanic fuel (gate builder + link builder), is fed into the
    Thought Entropy Ouroboros.

    The returned nirvanic fuel is used to:
    1. FUEL LATTICE ENTROPY — the lattice_entropy field (previously always 0)
       is now ALIVE, driven by ouroboros accumulated entropy
    2. EXPAND TOKEN BOUNDARIES — entropy fuel widens sacred constant envelopes
    3. AMPLIFY RESEARCH HEALTH — entropy-fueled research produces more inventions
    4. CROSS-BUILDER SYNERGY — reads gate & link nirvanic states for combined fuel
    5. ENLIGHTEN TOKENS — tokens touched by nirvanic fuel achieve sage precision

    Self-feeding ouroboros loop: token entropy → ouroboros → fuel → lattice → ∞
    """

    NIRVANIC_STATE_FILE = WORKSPACE_ROOT / ".l104_ouroboros_nirvanic_state.json"

    def __init__(self, lattice: 'TokenLatticeEngine'):
        """Initialize NumericalOuroborosNirvanicEngine."""
        self.lattice = lattice
        self.ouroboros = None
        self.cycle_count: int = 0
        self.total_entropy_fed: float = 0.0
        self.total_nirvanic_fuel: float = 0.0
        self.enlightened_tokens: int = 0
        self.nirvanic_coherence: float = 0.0
        self.sage_stability: float = 1.0
        self.divine_interventions: int = 0
        self.peer_gate_fuel: float = 0.0
        self.peer_link_fuel: float = 0.0
        self._load_state()

    def _get_ouroboros(self):
        """Read ouroboros nirvanic state."""
        if self.ouroboros is None:
            try:
                from l104_thought_entropy_ouroboros import get_thought_ouroboros
                self.ouroboros = get_thought_ouroboros()
            except ImportError:
                self.ouroboros = None
        return self.ouroboros

    def _load_state(self):
        """Load nirvanic state — reads peer fuel from whichever builder wrote last."""
        if self.NIRVANIC_STATE_FILE.exists():
            try:
                data = json.loads(self.NIRVANIC_STATE_FILE.read_text())
                src = data.get("source", "")
                if src == "numerical_builder":
                    self.cycle_count = data.get("cycle_count", 0)
                    self.total_entropy_fed = data.get("total_entropy_fed", 0.0)
                    self.total_nirvanic_fuel = data.get("total_nirvanic_fuel", 0.0)
                    self.enlightened_tokens = data.get("enlightened_tokens", 0)
                    self.nirvanic_coherence = data.get("nirvanic_coherence", 0.0)
                    self.sage_stability = data.get("sage_stability", 1.0)
                    self.divine_interventions = data.get("divine_interventions", 0)
                else:
                    # Peer builder wrote — read their combined fuel
                    self.peer_gate_fuel = data.get("total_nirvanic_fuel", 0.0)
                    if src == "quantum_link_builder":
                        self.peer_link_fuel = data.get("total_nirvanic_fuel", 0.0)
                        self.peer_gate_fuel = data.get("peer_nirvanic_fuel", 0.0)
                    elif src == "logic_gate_builder":
                        self.peer_gate_fuel = data.get("total_nirvanic_fuel", 0.0)
            except Exception:
                pass

    def _save_state(self):
        """Persist current state to disk."""
        try:
            data = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "source": "numerical_builder",
                "version": "2.4.0",
                "cycle_count": self.cycle_count,
                "total_entropy_fed": self.total_entropy_fed,
                "total_nirvanic_fuel": self.total_nirvanic_fuel,
                "enlightened_tokens": self.enlightened_tokens,
                "nirvanic_coherence": self.nirvanic_coherence,
                "sage_stability": self.sage_stability,
                "divine_interventions": self.divine_interventions,
                "peer_gate_fuel": self.peer_gate_fuel,
                "peer_link_fuel": self.peer_link_fuel,
            }
            self.NIRVANIC_STATE_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            pass

    def full_nirvanic_cycle(self, entropy_bits: float = 0.0,
                            gate_dyn_evo: int = 0, link_dyn_evo: int = 0) -> Dict[str, Any]:
        """Run the complete ouroboros nirvanic entropy fuel cycle for tokens."""
        ouroboros = self._get_ouroboros()
        if ouroboros is None:
            return {"status": "ouroboros_unavailable", "nirvanic_fuel": 0.0}

        self.cycle_count += 1

        # Combined fuel from peers
        combined_peer_fuel = self.peer_gate_fuel * 0.1 + self.peer_link_fuel * 0.1

        # Build thought from numerical entropy state
        thought = (
            f"Token lattice entropy {entropy_bits:.6f} bits across "
            f"{len(self.lattice.tokens)} tokens at 100-decimal precision. "
            f"Peer synergy: gate evolutions={gate_dyn_evo} link evolutions={link_dyn_evo}. "
            f"Gate nirvanic fuel={self.peer_gate_fuel:.4f} link fuel={self.peer_link_fuel:.4f}. "
            f"The 22-trillion usage lattice breathes with God Code {float(GOD_CODE_HP):.10f} Hz. "
            f"Cycle {self.cycle_count}: lattice entropy becomes divine fuel."
        )

        result = ouroboros.process(thought, depth=2)
        raw_fuel = result.get("accumulated_entropy", 0.0)
        nirvanic_fuel = raw_fuel + combined_peer_fuel

        self.total_entropy_fed += entropy_bits
        self.total_nirvanic_fuel += abs(nirvanic_fuel)

        # ★ FUEL THE LATTICE ENTROPY — previously always D(0), now ALIVE
        fuel_intensity = 1.0 / (1.0 + math.exp(-nirvanic_fuel * 0.1))
        lattice_entropy_value = D(str(entropy_bits)) * D(str(fuel_intensity))
        self.lattice.lattice_entropy = lattice_entropy_value

        # Enlighten tokens — expand boundaries for high-health tokens
        enlightened = 0
        interventions = 0
        for tid, token in self.lattice.tokens.items():
            if token.health > 0.9 and token.coherence > 0.9:
                val = D(token.value)
                expansion = val * D(str(fuel_intensity * 1e-90))
                old_max = D(token.max_bound)
                old_min = D(token.min_bound)
                token.max_bound = str(old_max + expansion)
                token.min_bound = str(old_min - expansion)
                interventions += 1
                self.divine_interventions += 1
                if fuel_intensity > 0.5 and token.origin in ("sacred", "derived"):
                    enlightened += 1

        self.enlightened_tokens = enlightened

        # Nirvanic coherence from combined state
        self.nirvanic_coherence = (
            fuel_intensity * 0.4 +
            float(self.lattice.lattice_coherence) * 0.3 +
            min(entropy_bits / 6.0, 0.3)
        )

        # Sage stability
        if interventions > 0:
            perturbation = interventions / max(len(self.lattice.tokens), 1)
            self.sage_stability = max(0.0, 1.0 - perturbation * 0.02)
        else:
            self.sage_stability = min(1.0, self.sage_stability + 0.01)

        self._save_state()

        return {
            "status": "processed",
            "cycle": self.cycle_count,
            "entropy_fed": entropy_bits,
            "nirvanic_fuel": nirvanic_fuel,
            "fuel_intensity": fuel_intensity,
            "lattice_entropy_now": str(lattice_entropy_value)[:30],
            "enlightened_tokens": enlightened,
            "divine_interventions": interventions,
            "nirvanic_coherence": self.nirvanic_coherence,
            "sage_stability": self.sage_stability,
            "total_nirvanic_fuel": self.total_nirvanic_fuel,
            "peer_gate_fuel": self.peer_gate_fuel,
            "peer_link_fuel": self.peer_link_fuel,
            "ouroboros_mutations": result.get("total_mutations", 0),
            "ouroboros_resonance": result.get("cycle_resonance", 0.0),
        }

    def status(self) -> Dict[str, Any]:
        """Return current subsystem status."""
        return {
            "version": "2.4.0",
            "cycle_count": self.cycle_count,
            "total_entropy_fed": self.total_entropy_fed,
            "total_nirvanic_fuel": self.total_nirvanic_fuel,
            "enlightened_tokens": self.enlightened_tokens,
            "nirvanic_coherence": self.nirvanic_coherence,
            "sage_stability": self.sage_stability,
            "divine_interventions": self.divine_interventions,
            "ouroboros_connected": self._get_ouroboros() is not None,
            "peer_gate_fuel": self.peer_gate_fuel,
            "peer_link_fuel": self.peer_link_fuel,
            "lattice_entropy": str(self.lattice.lattice_entropy)[:30],
        }
