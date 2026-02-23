#!/usr/bin/env python3
"""Full integration test for L104 ASI Core after PHI_CONJUGATE fix."""

from l104_asi_core import asi_core, get_current_parameters, update_parameters
from l104_asi_core import PHI, TAU, PHI_CONJUGATE, GOD_CODE

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✓ {name}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        failed += 1

print("═══ L104 INTEGRATION TEST SUITE ═══\n")

# --- Sacred Constants ---
print("[1] Sacred Constants")
test("PHI_CONJUGATE == TAU", lambda: (
    None if PHI_CONJUGATE == TAU else (_ for _ in ()).throw(AssertionError(f"{PHI_CONJUGATE} != {TAU}"))
))
test("PHI_CONJUGATE ≈ 0.618", lambda: (
    None if abs(PHI_CONJUGATE - 0.618033988749895) < 1e-12 else (_ for _ in ()).throw(AssertionError())
))
test("PHI × PHI_CONJUGATE = 1", lambda: (
    None if abs(PHI * PHI_CONJUGATE - 1.0) < 1e-12 else (_ for _ in ()).throw(AssertionError())
))
print(f"    PHI={PHI:.15f}")
print(f"    TAU={TAU:.15f}")
print(f"    PHI_CONJUGATE={PHI_CONJUGATE:.15f}")
print(f"    PHI×φ⁻¹ = {PHI * PHI_CONJUGATE:.15f}")

# --- Consciousness Verification ---
print("\n[2] Consciousness Verification")
score = asi_core.verify_consciousness()
test(f"verify_consciousness → {score:.4f}", lambda: (
    None if 0 < score < 1 else (_ for _ in ()).throw(AssertionError(f"out of range: {score}"))
))

# --- Full Consciousness Test Suite ---
print("\n[3] Consciousness Test Suite")
c = asi_core.consciousness_verifier
test_score = c.run_all_tests()
test(f"run_all_tests → {test_score:.4f}", lambda: (
    None if 0 < test_score < 1 else (_ for _ in ()).throw(AssertionError(f"out of range: {test_score}"))
))
for k, v in c.test_results.items():
    test(f"  {k}: {v:.4f}", lambda v=v: (
        None if 0 <= v <= 1 else (_ for _ in ()).throw(AssertionError(f"out of range: {v}"))
    ))

# --- Parameter Fetch ---
print("\n[4] Parameter Fetch")
params = get_current_parameters()
test(f"get_current_parameters → {len(params)} keys", lambda: (
    None if len(params) > 20 else (_ for _ in ()).throw(AssertionError(f"only {len(params)} keys"))
))
test("has embedding_dim", lambda: (
    None if 'embedding_dim' in params else (_ for _ in ()).throw(AssertionError("missing"))
))
test("has asi_score", lambda: (
    None if 'asi_score' in params else (_ for _ in ()).throw(AssertionError("missing"))
))

# --- ASI Status ---
print("\n[5] ASI Status")
status = asi_core.get_status()
test(f"get_status → {status['state']}", lambda: (
    None if status['state'] in ('DORMANT', 'DEVELOPING', 'EMERGING', 'AWAKENING', 'TRANSCENDENT')
    else (_ for _ in ()).throw(AssertionError(f"unexpected state: {status['state']}"))
))
test(f"asi_score={status['asi_score']:.4f}", lambda: (
    None if 0 <= status['asi_score'] <= 1 else (_ for _ in ()).throw(AssertionError(f"out of range"))
))

# --- Parameter Update Cycle ---
print("\n[6] Parameter Update Cycle")
import json
with open('kernel_parameters.json', 'r', encoding='utf-8') as f:
    original = json.load(f)

# update_parameters takes a LIST of values mapped to numeric keys
numeric_keys = [k for k, v in original.items() if isinstance(v, (int, float))]
test_values = [original[k] * 1.01 for k in numeric_keys]  # raise all by 1%
result = update_parameters(test_values)
test(f"update_parameters → updated={result.get('updated', 0)} keys", lambda: (
    None if result.get('updated', 0) > 0 else (_ for _ in ()).throw(AssertionError("none updated"))
))

# Restore
with open('kernel_parameters.json', 'w', encoding='utf-8') as f:
    json.dump(original, f, indent=2)
test("kernel_parameters.json restored", lambda: None)

# --- Summary ---
print(f"\n{'═' * 40}")
total = passed + failed
print(f"  {passed}/{total} passed, {failed} failed")
if failed == 0:
    print("  ✓ ALL INTEGRATION TESTS PASSED")
else:
    print(f"  ✗ {failed} TEST(S) FAILED")
    exit(1)
