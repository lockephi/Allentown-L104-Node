#!/usr/bin/env python3
"""
L104 Quick Test - Run core validation tests
"""
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

sys.path.insert(0, '/workspaces/Allentown-L104-Node')

tests_passed = 0
tests_failed = 0

print("═" * 70)
print("    L104 QUICK VALIDATION SUITE")
print("═" * 70)

# Test 1: Core constants
print("\n[1/5] Testing core constants...")
try:
    from const import GOD_CODE, PHI, VOID_CONSTANT
    assert abs(GOD_CODE - 527.5184818492537) < 0.0001, "GOD_CODE mismatch"
    assert abs(PHI - 1.618033988749895) < 0.0001, "PHI mismatch"
    print(f"  ✓ GOD_CODE = {GOD_CODE}")
    print(f"  ✓ PHI = {PHI}")
    print(f"  ✓ VOID_CONSTANT = {VOID_CONSTANT}")
    tests_passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    tests_failed += 1

# Test 2: Sage Enlighten module
print("\n[2/5] Testing l104_sage_enlighten...")
try:
    from l104_sage_enlighten import (
        enlighten_inflect, EnlightenedState, HyperComplex,
        SageLevel, SageModeOrchestrator
    )

    # Single value enlightenment
    state = enlighten_inflect(0.5)
    assert isinstance(state, EnlightenedState)
    assert 0 <= state.clarity <= 1
    print(f"  ✓ Enlighten inflect: clarity={state.clarity:.4f}, wisdom={state.wisdom:.4f}")

    # HyperComplex math
    h = HyperComplex(1, 2, 3, 4)
    expected_mag = (1**2 + 2**2 + 3**2 + 4**2) ** 0.5
    assert abs(h.magnitude - expected_mag) < 0.001
    print(f"  ✓ HyperComplex magnitude: {h.magnitude:.4f}")

    # Multiplication
    h2 = h * h
    assert h2.magnitude > 0
    print(f"  ✓ HyperComplex multiply: |h*h| = {h2.magnitude:.4f}")

    tests_passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    tests_failed += 1

# Test 3: Void Math
print("\n[3/5] Testing void math...")
try:
    from l104_void_math import VoidMath
    vm = VoidMath()
    result = vm.primal_calculus(10.0)
    assert result > 0
    print(f"  ✓ Primal calculus(10) = {result:.4f}")

    sequence = vm.generate_void_sequence(5)
    assert len(sequence) == 5
    print(f"  ✓ Void sequence[0:3] = {sequence[:3]}")
    tests_passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    tests_failed += 1

# Test 4: Soul import
print("\n[4/5] Testing l104 module...")
try:
    from l104 import Soul, VERSION, GOD_CODE
    assert abs(GOD_CODE - 527.5184818492537) < 0.0001
    print(f"  ✓ L104 VERSION = {VERSION}")
    print(f"  ✓ L104 GOD_CODE = {GOD_CODE}")
    tests_passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    tests_failed += 1

# Test 5: Sage bindings
print("\n[5/5] Testing sage bindings...")
try:
    from l104_sage_bindings import get_sage_core
    core = get_sage_core()
    result = core.primal_calculus(527.5184818492537, 1.618033988749895)
    assert result > 0
    print(f"  ✓ Sage core primal_calculus = {result:.4f}")

    # Use emit_void_resonance instead of void_resonance
    void_res = core.emit_void_resonance()
    assert void_res > 0
    print(f"  ✓ Sage core void_resonance = {void_res:.4f}")

    # Test omega controller state
    state = core.get_state()
    print(f"  ✓ Controller Authority Code: {state['void_math']['god_code']:.4f}")

    tests_passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    tests_failed += 1

# Summary
print("\n" + "═" * 70)
total = tests_passed + tests_failed
if tests_failed == 0:
    print(f"    ✓ ALL TESTS PASSED ({tests_passed}/{total})")
else:
    print(f"    ✗ {tests_failed} TESTS FAILED ({tests_passed}/{total} passed)")
print("═" * 70)

sys.exit(0 if tests_failed == 0 else 1)
