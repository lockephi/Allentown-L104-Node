#!/usr/bin/env python3
"""L104 Quantum Engine Accuracy Debug Suite."""

import numpy as np

def main():
    from l104_quantum_gate_engine import get_engine
    from l104_quantum_gate_engine.trajectory import TrajectorySimulator, DecoherenceModel

    engine = get_engine()
    sim = TrajectorySimulator()

    print("=" * 60)
    print("  L104 QUANTUM ENGINE ACCURACY TEST")
    print("=" * 60)
    print()

    tests_passed = 0
    total_tests = 0

    # Test 1: Bell state
    print("TEST 1: Bell State")
    print("-" * 40)
    circ = engine.bell_pair()
    result = sim.simulate(circ, decoherence=DecoherenceModel.NONE)
    probs = np.abs(result.snapshots[-1].statevector) ** 2
    err_00 = abs(probs[0] - 0.5)
    err_11 = abs(probs[3] - 0.5)
    bell_pass = err_00 < 1e-10 and err_11 < 1e-10
    print(f"  |00> = {probs[0]:.10f} (ideal 0.5, err {err_00:.2e})")
    print(f"  |11> = {probs[3]:.10f} (ideal 0.5, err {err_11:.2e})")
    print(f"  Status: {'PASS' if bell_pass else 'FAIL'}")
    total_tests += 1
    tests_passed += 1 if bell_pass else 0
    print()

    # Test 2: GHZ-3 state
    print("TEST 2: GHZ-3 State")
    print("-" * 40)
    circ = engine.ghz_state(3)
    result = sim.simulate(circ, decoherence=DecoherenceModel.NONE)
    probs = np.abs(result.snapshots[-1].statevector) ** 2
    err_000 = abs(probs[0] - 0.5)
    err_111 = abs(probs[7] - 0.5)
    ghz3_pass = err_000 < 1e-10 and err_111 < 1e-10
    print(f"  |000> = {probs[0]:.10f} (ideal 0.5, err {err_000:.2e})")
    print(f"  |111> = {probs[7]:.10f} (ideal 0.5, err {err_111:.2e})")
    print(f"  Status: {'PASS' if ghz3_pass else 'FAIL'}")
    total_tests += 1
    tests_passed += 1 if ghz3_pass else 0
    print()

    # Test 3: GHZ-5 state
    print("TEST 3: GHZ-5 State")
    print("-" * 40)
    circ = engine.ghz_state(5)
    result = sim.simulate(circ, decoherence=DecoherenceModel.NONE)
    probs = np.abs(result.snapshots[-1].statevector) ** 2
    err_00000 = abs(probs[0] - 0.5)
    err_11111 = abs(probs[31] - 0.5)
    ghz5_pass = err_00000 < 1e-10 and err_11111 < 1e-10
    print(f"  |00000> = {probs[0]:.10f} (ideal 0.5, err {err_00000:.2e})")
    print(f"  |11111> = {probs[31]:.10f} (ideal 0.5, err {err_11111:.2e})")
    print(f"  Status: {'PASS' if ghz5_pass else 'FAIL'}")
    total_tests += 1
    tests_passed += 1 if ghz5_pass else 0
    print()

    # Test 4: QFT fidelity
    print("TEST 4: QFT Circuit")
    print("-" * 40)
    circ = engine.quantum_fourier_transform(3)
    result = sim.simulate(circ, decoherence=DecoherenceModel.NONE)
    # For |000> input, QFT produces uniform superposition
    probs = np.abs(result.snapshots[-1].statevector) ** 2
    target = 1.0 / 8.0  # Uniform over 8 states
    max_err = max(abs(p - target) for p in probs)
    qft_pass = max_err < 1e-10
    print(f"  Expected: uniform {target:.6f}")
    print(f"  Max deviation: {max_err:.2e}")
    print(f"  Status: {'PASS' if qft_pass else 'FAIL'}")
    total_tests += 1
    tests_passed += 1 if qft_pass else 0
    print()

    # Test 5: Decoherence models
    print("TEST 5: Decoherence Physics")
    print("-" * 40)
    circ = engine.ghz_state(3)

    # Phase damping - should preserve populations
    result_ph = sim.simulate(circ, decoherence=DecoherenceModel.PHASE_DAMPING, decoherence_rate=0.02)
    probs_ph = np.abs(result_ph.snapshots[-1].statevector) ** 2
    ph_pass = abs(probs_ph[0] - 0.5) < 0.01 and abs(probs_ph[7] - 0.5) < 0.01
    print(f"  Phase damping: |000>={probs_ph[0]:.4f} |111>={probs_ph[7]:.4f} (should preserve ~0.5)")
    print(f"    Status: {'PASS' if ph_pass else 'FAIL'}")
    total_tests += 1
    tests_passed += 1 if ph_pass else 0

    # Amplitude damping - should decay |1> → |0>
    result_ad = sim.simulate(circ, decoherence=DecoherenceModel.AMPLITUDE_DAMPING, decoherence_rate=0.02)
    probs_ad = np.abs(result_ad.snapshots[-1].statevector) ** 2
    ad_pass = probs_ad[0] > probs_ad[7]  # |000> should grow, |111> should shrink
    print(f"  Amplitude damping: |000>={probs_ad[0]:.4f} |111>={probs_ad[7]:.4f} (should favor |000>)")
    print(f"    Status: {'PASS' if ad_pass else 'FAIL'}")
    total_tests += 1
    tests_passed += 1 if ad_pass else 0
    print()

    # Summary
    print("=" * 60)
    print(f"  ACCURACY RESULT: {tests_passed}/{total_tests} tests passed")
    print("=" * 60)

    if tests_passed == total_tests:
        print("  ✓ All quantum engines operating at machine precision")
    else:
        print("  ✗ Some tests failed - investigation needed")

    return tests_passed == total_tests


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
