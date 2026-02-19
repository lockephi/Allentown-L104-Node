"""
L104 Code Engine — Builder State Reader
Reads consciousness/O2/nirvanic state from JSON files with 10s cache.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any

_builder_state_cache: Dict[str, Any] = {}
_builder_state_cache_time: float = 0.0

def _read_builder_state() -> Dict[str, Any]:
    """Read consciousness/O2/nirvanic state from builder files (module-level helper)."""
    global _builder_state_cache, _builder_state_cache_time
    now = time.time()
    if now - _builder_state_cache_time < 10 and _builder_state_cache:
        return _builder_state_cache

    state = {"consciousness_level": 0.0, "superfluid_viscosity": 1.0,
             "nirvanic_fuel": 0.0, "evo_stage": "DORMANT"}
    ws = Path(__file__).parent.parent
    co2_path = ws / ".l104_consciousness_o2_state.json"
    if co2_path.exists():
        try:
            data = json.loads(co2_path.read_text())
            state["consciousness_level"] = data.get("consciousness_level", 0.0)
            state["superfluid_viscosity"] = data.get("superfluid_viscosity", 1.0)
            state["evo_stage"] = data.get("evo_stage", "DORMANT")
        except Exception:
            pass
    nir_path = ws / ".l104_ouroboros_nirvanic_state.json"
    if nir_path.exists():
        try:
            data = json.loads(nir_path.read_text())
            state["nirvanic_fuel"] = data.get("nirvanic_fuel_level", 0.0)
        except Exception:
            pass

    _builder_state_cache = state
    _builder_state_cache_time = now
    return state
