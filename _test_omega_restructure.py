#!/usr/bin/env python3
"""Smoke tests for OMEGA dual-layer restructure."""

from l104_god_code_dual_layer import (
    consciousness, physics, physics_v3, derive, derive_both, status,
    OMEGA, OMEGA_AUTHORITY, GOD_CODE, GOD_CODE_V3, EVOLUTION_HERITAGE,
    CONSCIOUSNESS_TO_PHYSICS_BRIDGE, full_integrity_check
)

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")

print("=== LAYER 1: CONSCIOUSNESS (GOD_CODE equation) ===")
c = consciousness()
print(f"  consciousness(0,0,0,0) = {c:.10f}")
check("GOD_CODE value", abs(c - 527.5184818492612) < 1e-6)

print("\n=== LAYER 2: PHYSICS (OMEGA equation) ===")
p = physics()
print(f"  physics() = {p}")
check("physics returns dict", isinstance(p, dict))
check("has omega key", "omega" in p)
check("OMEGA value", abs(p.get("omega", 0) - 6539.34712682) < 0.01)
check("has field_strength", "field_strength" in p)
check("has omega_authority", "omega_authority" in p)

print("\n=== PHYSICS V3 (sub-tool backward compat) ===")
pv3 = physics_v3(0, 0, 0, 0)
print(f"  physics_v3(0,0,0,0) = {pv3:.10f}")
check("GOD_CODE_V3 value", abs(pv3 - 45.41141298077539) < 0.001)

print("\n=== EVOLUTION_HERITAGE ===")
print(f"  keys: {list(EVOLUTION_HERITAGE.keys())}")
check("has original", "original" in EVOLUTION_HERITAGE)
check("has v3_superparticular", "v3_superparticular" in EVOLUTION_HERITAGE)
check("has omega_sovereign_field", "omega_sovereign_field" in EVOLUTION_HERITAGE)

print("\n=== CONSCIOUSNESS_TO_PHYSICS_BRIDGE ===")
print(f"  keys: {list(CONSCIOUSNESS_TO_PHYSICS_BRIDGE.keys())}")
check("has omega_sovereign_field", "omega_sovereign_field" in CONSCIOUSNESS_TO_PHYSICS_BRIDGE)
check("has god_code_generates_omega", "god_code_generates_omega" in CONSCIOUSNESS_TO_PHYSICS_BRIDGE)

print("\n=== STATUS ===")
s = status()
print(f"  version: {s['version']}")
print(f"  architecture: {s['architecture']}")
print(f"  layer2 equation: {s['layer2_physics']['equation'][:70]}...")
check("version 3.0.0", s["version"] == "3.0.0")
check("OMEGA in layer2 equation", "OMEGA" in s["layer2_physics"]["equation"] or "Ω" in s["layer2_physics"]["equation"])
check("layer2 has OMEGA constant", "OMEGA" in s["layer2_physics"])

print("\n=== INTEGRITY CHECK ===")
ic = full_integrity_check()
print(f"  {ic['checks_passed']}/{ic['total_checks']} passed, all_passed={ic['all_passed']}")
check("integrity passes", ic["all_passed"])

print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
if failed == 0:
    print("=== ALL SMOKE TESTS PASSED ===")
else:
    print(f"!!! {failed} TESTS FAILED !!!")
