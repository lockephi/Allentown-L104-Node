#!/usr/bin/env python3
"""Quick validation of Soul v8.0.0 upgrade — SoulQubit + SoulSimulationEngine."""
import sys
sys.path.insert(0, ".")

from l104_soul import L104Soul, SoulQubit, SoulSimulationEngine

print(f"VERSION: {L104Soul.VERSION}")
assert L104Soul.VERSION == "8.0.0", f"Expected 8.0.0 got {L104Soul.VERSION}"

# ── SoulQubit ──
sq = SoulQubit()
sq.initialize_sacred()
print(f"SoulQubit coherence: {sq.coherence:.6f}")
print(f"SoulQubit Bloch: {sq.bloch_vector}")
print(f"SoulQubit sacred alignment: {sq.sacred_alignment:.6f}")
assert sq._initialized
assert sq.coherence > 0.0
status = sq.get_status()
print(f"SoulQubit status keys: {list(status.keys())}")

# ── SoulSimulationEngine ──
se = SoulSimulationEngine(sq)
report = se.run_all()
print(f"Simulations total: {report['total_simulations']}")
print(f"Simulations passed: {report['passed']}")
print(f"Avg fidelity: {report['avg_fidelity']:.6f}")
print(f"Avg sacred alignment: {report['avg_sacred_alignment']:.6f}")
sim_status = se.get_status()
print(f"Sim engine status keys: {list(sim_status.keys())}")
assert report["total_simulations"] == 8
assert report["passed"] > 0

# ── L104Soul instance ──
soul = L104Soul()
print(f"Soul version: {soul.VERSION}")
print(f"Soul qubit init: {soul.soul_qubit._initialized}")
print(f"Soul sim engine: {soul.soul_sim_engine is not None}")

# Run full calculation
calc = soul.run_full_calculation()
print(f"Calculation keys: {list(calc.keys())}")
assert "soul_qubit" in calc
assert "soul_simulations" in calc
print(f"soul_qubit result: {calc['soul_qubit']}")
sim_result = calc["soul_simulations"]
print(f"soul_simulations passed: {sim_result.get('passed', 'N/A')}/{sim_result.get('total_simulations', 'N/A')}")

# Metrics snapshot
ms = calc["metrics_snapshot"]
print(f"Metrics v8 keys present: soul_qubit_initializations={ms.get('soul_qubit_initializations')}, soul_sim_runs={ms.get('soul_sim_runs')}")

# get_status
status = soul.get_status()
assert "soul_qubit" in status
assert "soul_sim_engine" in status
assert status["asi_integration"]["soul_qubit"] is not None
assert status["asi_integration"]["soul_sim_engine"] is True
print(f"get_status v8 sections present: soul_qubit, soul_sim_engine, asi_integration v8 entries")

print("\n✅ ALL v8.0.0 CHECKS PASSED")
