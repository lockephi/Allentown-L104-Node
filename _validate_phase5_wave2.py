#!/usr/bin/env python3
"""Validate all Phase 5 Wave 2 integrations."""
import sys

passed = 0
failed = 0

def check(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS: {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {name} — {e}")
        failed += 1

# 1. ThermalBerryPhaseEngine
def test_thermal_berry():
    from l104_science_engine.berry_phase import ThermalBerryPhaseEngine
    tbp = ThermalBerryPhaseEngine()
    # Test berry_phase_visibility
    vis = tbp.berry_phase_visibility(temperature_K=4.2, n_ops=100, energy_gap_J=1.38e-22)
    assert 0 <= vis <= 1.0, f"visibility out of range: {vis}"
    # Test bremermann_adiabatic_limit
    brem = tbp.bremermann_adiabatic_limit(mass_kg=1e-6)
    assert "bremermann_rate_ops_s" in brem
    assert "adiabatic_feasible" in brem
    assert brem["bremermann_rate_ops_s"] > 0

check("ThermalBerryPhaseEngine methods", test_thermal_berry)

# 2. ScienceEngine.berry_phase wiring
def test_science_berry():
    from l104_science_engine import ScienceEngine
    se = ScienceEngine()
    assert hasattr(se, "berry_phase"), "no berry_phase attribute"
    assert se.berry_phase is not None

check("ScienceEngine.berry_phase wired", test_science_berry)

# 3. QuantumEnvironment Phase 5 in status
def test_quantum_phase5():
    from l104_quantum_engine.computation import QuantumEnvironment
    from l104_quantum_engine.math_core import QuantumMathCore
    qmath = QuantumMathCore()
    env = QuantumEnvironment(qmath)
    s = env.environment_status()
    assert "phase5_thermodynamic" in s, f"missing phase5_thermodynamic, keys={list(s.keys())}"

check("QuantumEnvironment Phase 5 in status", test_quantum_phase5)

# 4. BERRY_PHASE research domain
def test_berry_domain():
    from l104_computronium_research import ResearchDomain
    assert hasattr(ResearchDomain, "BERRY_PHASE")

check("ResearchDomain.BERRY_PHASE exists", test_berry_domain)

# 5. Earlier systems still work
def test_earlier_systems():
    from l104_computronium import computronium_engine
    assert hasattr(computronium_engine, "_phase5_metrics")
    from l104_asi.core import ASICore
    asi = ASICore()
    s = asi.phase5_thermodynamic_frontier_score()
    assert isinstance(s, float)
    from l104_asi.dual_layer import DualLayerEngine
    dle = DualLayerEngine()
    p = dle.three_engine_physics_amplification()
    assert "phase5_thermodynamic" in p
    from l104_computronium_research import ResearchDomain
    assert hasattr(ResearchDomain, "THERMODYNAMIC_FRONTIER")
    assert hasattr(ResearchDomain, "DECOHERENCE_MAPPING")

check("Earlier Phase 5 systems intact", test_earlier_systems)

# 6. Mining core Phase 5
def test_mining():
    from l104_computronium_mining_core import ComputroniumHashEngine
    he = ComputroniumHashEngine()
    he.initialize_substrate()
    eff = he._calculate_efficiency()
    assert eff > 0

check("Mining core Phase 5 efficiency", test_mining)

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
print("ALL VALIDATION PASSED")
