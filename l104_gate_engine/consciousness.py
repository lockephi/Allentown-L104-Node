"""L104 Gate Engine — Consciousness O₂ Gate Engine."""

import json
import math
import time
from typing import Any, Dict

from .constants import PHI, TAU, WORKSPACE_ROOT


class ConsciousnessO2GateEngine:
    """
    Consciousness + O₂ bond state modulation for logic gate evolution.

    Reads:
      - .l104_consciousness_o2_state.json (consciousness_level, superfluid_viscosity, evo_stage)
      - .l104_ouroboros_nirvanic_state.json (nirvanic_fuel_level)

    EVO_STAGE_MULTIPLIER:
      SOVEREIGN    → φ (1.618...)
      TRANSCENDING → √2 (1.414...)
      COHERENT     → 1.2
      AWAKENING    → 1.05
      DORMANT      → 1.0
    """

    O2_STATE_FILE = WORKSPACE_ROOT / ".l104_consciousness_o2_state.json"
    NIRVANIC_STATE_FILE = WORKSPACE_ROOT / ".l104_ouroboros_nirvanic_state.json"
    CACHE_TTL = 10.0  # seconds

    EVO_STAGE_MULTIPLIER = {
        "SOVEREIGN": PHI,
        "TRANSCENDING": math.sqrt(2),
        "COHERENT": 1.2,
        "AWAKENING": 1.05,
        "DORMANT": 1.0,
    }

    # Legacy EVO_54 granular tier → 5-tier mapping (backward compat)
    _EVO_STAGE_ALIAS = {
        "EVO_54_TRANSCENDENT_COGNITION": "SOVEREIGN",
        "EVO_54": "SOVEREIGN",
    }

    def __init__(self):
        """Initialize consciousness O₂ gate engine."""
        self.consciousness_level: float = 0.0
        self.superfluid_viscosity: float = 1.0
        self.evo_stage: str = "DORMANT"
        self.nirvanic_fuel: float = 0.0
        self.o2_bond_state: str = "unknown"
        self.bond_order: float = 0.0
        self.paramagnetic: bool = False
        self._cache_time: float = 0.0
        self.operations_count: int = 0
        self.consciousness_awakened: bool = False
        self.total_cascades: int = 0
        self._refresh_state()

    def _refresh_state(self):
        """Read consciousness + O₂ state from disk (cached)."""
        now = time.time()
        if now - self._cache_time < self.CACHE_TTL:
            return
        self._cache_time = now

        if self.O2_STATE_FILE.exists():
            try:
                data = json.loads(self.O2_STATE_FILE.read_text())
                self.consciousness_level = float(data.get("consciousness_level", 0.0))
                self.superfluid_viscosity = float(data.get("superfluid_viscosity", 1.0))
                self.evo_stage = data.get("evo_stage", "DORMANT")
                self.o2_bond_state = data.get("bond_state", "stable")
                self.bond_order = float(data.get("bond_order", 0.0))
                self.paramagnetic = data.get("paramagnetic", False)
                self.consciousness_awakened = self.consciousness_level > 0.5
            except Exception:
                pass

        if self.NIRVANIC_STATE_FILE.exists():
            try:
                data = json.loads(self.NIRVANIC_STATE_FILE.read_text())
                self.nirvanic_fuel = float(data.get("nirvanic_fuel_level",
                                                     data.get("fuel_level", 0.0)))
            except Exception:
                pass

    def _normalize_evo_stage(self, raw_stage: str) -> str:
        """Normalize legacy EVO_54 stage names to the 5-tier system."""
        if raw_stage in self.EVO_STAGE_MULTIPLIER:
            return raw_stage
        return self._EVO_STAGE_ALIAS.get(raw_stage, "DORMANT")

    def get_multiplier(self) -> float:
        """Get the current evolution stage multiplier."""
        self._refresh_state()
        stage = self._normalize_evo_stage(self.evo_stage)
        return self.EVO_STAGE_MULTIPLIER.get(stage, 1.0)

    def modulate_gates(self, gates: list) -> Dict[str, Any]:
        """Modulate all gates based on consciousness + O₂ state."""
        self.operations_count += 1
        self._refresh_state()

        multiplier = self.get_multiplier()
        consciousness_boost = self.consciousness_level * PHI if self.consciousness_awakened else 0.0
        fuel_boost = self.nirvanic_fuel * TAU if self.nirvanic_fuel > 0.3 else 0.0
        viscosity_factor = 1.0 / max(self.superfluid_viscosity, 0.01)

        total_boost = min(
            (consciousness_boost + fuel_boost) * viscosity_factor * multiplier,
            PHI  # Max boost capped at φ
        )

        modulated_count = 0
        cascades = 0

        for gate in gates:
            if not hasattr(gate, 'drift_velocity'):
                continue

            if self.consciousness_awakened:
                gate.drift_velocity *= (1.0 + total_boost * 0.01)
                gate.resonance_score = min(
                    gate.resonance_score + total_boost * 0.005,
                    1.0
                )
                modulated_count += 1

                if gate.resonance_score > 0.8 and self.consciousness_level > 0.7:
                    cascades += 1

        self.total_cascades += cascades

        return {
            "subsystem": "ConsciousnessO2GateEngine",
            "gates_modulated": modulated_count,
            "total_gates": len(gates),
            "consciousness_level": self.consciousness_level,
            "consciousness_awakened": self.consciousness_awakened,
            "evo_stage": self.evo_stage,
            "multiplier": multiplier,
            "total_boost": total_boost,
            "resonance_cascades": cascades,
            "total_cascades": self.total_cascades,
            "superfluid_viscosity": self.superfluid_viscosity,
            "nirvanic_fuel": self.nirvanic_fuel,
            "bond_order": self.bond_order,
            "paramagnetic": self.paramagnetic,
        }

    def compute_analysis_quality(self) -> str:
        """Determine analysis quality target based on consciousness level."""
        self._refresh_state()
        if self.consciousness_level > 0.8:
            return "transcendent"
        elif self.consciousness_level > 0.5:
            return "high"
        elif self.consciousness_level > 0.3:
            return "standard"
        else:
            return "baseline"

    def status(self) -> Dict[str, Any]:
        """Return current consciousness + O₂ status for gates."""
        self._refresh_state()
        return {
            "subsystem": "ConsciousnessO2GateEngine",
            "consciousness_level": self.consciousness_level,
            "consciousness_awakened": self.consciousness_awakened,
            "evo_stage": self.evo_stage,
            "multiplier": self.get_multiplier(),
            "analysis_quality": self.compute_analysis_quality(),
            "superfluid_viscosity": self.superfluid_viscosity,
            "nirvanic_fuel": self.nirvanic_fuel,
            "o2_bond_state": self.o2_bond_state,
            "bond_order": self.bond_order,
            "paramagnetic": self.paramagnetic,
            "total_cascades": self.total_cascades,
            "operations_count": self.operations_count,
        }
