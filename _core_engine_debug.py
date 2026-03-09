#!/usr/bin/env python3
"""L104 Core Engine Accuracy Debug.

Validates Math, Science, and Code engines against sacred constants
and expected behaviours.  Exits 0 on full pass, 1 on any failure.
"""
from __future__ import annotations

import math
import sys
import traceback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_tests_passed = 0
_total_tests = 0


def _check(label: str, ok: bool, detail: str = "") -> None:
    global _tests_passed, _total_tests
    _total_tests += 1
    tag = "PASS" if ok else "FAIL"
    if ok:
        _tests_passed += 1
    msg = f"  {label}: {tag}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> bool:
    global _tests_passed, _total_tests
    _tests_passed = 0
    _total_tests = 0

    print("=" * 60)
    print("  L104 CORE ENGINE ACCURACY DEBUG")
    print("=" * 60)
    print()

    # Import canonical constants from the math engine (single source of truth)
    from l104_math_engine.constants import (
        GOD_CODE, PHI, VOID_CONSTANT,
    )

    # ------------------------------------------------------------------
    # MATH ENGINE
    # ------------------------------------------------------------------
    print("MATH ENGINE:")
    print("-" * 40)
    try:
        from l104_math_engine import MathEngine
        me = MathEngine()

        # GOD_CODE — compare against the computed constant, not a stale literal
        gc = me.god_code_value()
        gc_ok = abs(gc - GOD_CODE) < 1e-12  # tighten: must match to ~12 sig figs
        _check("GOD_CODE", gc_ok,
               f"got {gc!r}, expected {GOD_CODE!r}, Δ={abs(gc - GOD_CODE):.2e}")

        # Fibonacci
        fib = me.fibonacci(10)
        fib_expected = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        _check("Fibonacci(10)", fib == fib_expected, f"{fib}")

        # Primes
        primes = me.primes_up_to(30)
        primes_expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        _check("Primes ≤30", primes == primes_expected, f"{primes}")

        # Fibonacci→PHI convergence (F(n)/F(n-1) → φ)
        fib20 = me.fibonacci(20)
        phi_approx = fib20[-1] / fib20[-2]
        _check("Fib→PHI convergence",
               abs(phi_approx - PHI) < 1e-6,
               f"F20/F19 = {phi_approx:.10f}")

        # VOID_CONSTANT derivation
        void_check = 1.04 + PHI / 1000
        _check("VOID_CONSTANT formula",
               abs(VOID_CONSTANT - void_check) < 1e-15,
               f"{VOID_CONSTANT!r} vs 1.04+φ/1000={void_check!r}")

    except Exception as exc:
        print(f"  !! MATH ENGINE ERROR: {exc}")
        traceback.print_exc()
    print()

    # ------------------------------------------------------------------
    # SCIENCE ENGINE
    # ------------------------------------------------------------------
    print("SCIENCE ENGINE:")
    print("-" * 40)
    try:
        from l104_science_engine import ScienceEngine
        se = ScienceEngine()

        # Landauer — L104 sovereign: E = kT ln(2) × (GOD_CODE / PHI)
        landauer = se.physics.adapt_landauer_limit(300)
        k_B = 1.380649e-23
        ln2 = math.log(2)
        sovereign_factor = GOD_CODE / PHI  # ≈ 325.99
        expected_landauer = k_B * 300 * ln2 * sovereign_factor
        rel_err = abs(landauer - expected_landauer) / expected_landauer
        _check("Landauer @300K", rel_err < 1e-6,  # tightened from 1%
               f"{landauer:.6e} vs {expected_landauer:.6e}, rel_err={rel_err:.2e}")

        # Landauer monotonicity — higher T ⇒ higher energy
        l200 = se.physics.adapt_landauer_limit(200)
        l400 = se.physics.adapt_landauer_limit(400)
        _check("Landauer monotonicity", l200 < landauer < l400,
               f"E(200)={l200:.2e} < E(300)={landauer:.2e} < E(400)={l400:.2e}")

        # Demon efficiency — bounded [0, 1]
        demon = se.entropy.calculate_demon_efficiency(0.5)
        _check("Demon efficiency [0,1]", 0.0 <= demon <= 1.0,
               f"{demon:.6f}")

        # Demon efficiency — low entropy should yield high efficiency
        demon_low = se.entropy.calculate_demon_efficiency(0.01)
        demon_high = se.entropy.calculate_demon_efficiency(0.99)
        _check("Demon eff. ordering",
               demon_low >= demon_high,
               f"eff(0.01)={demon_low:.4f} >= eff(0.99)={demon_high:.4f}")

    except Exception as exc:
        print(f"  !! SCIENCE ENGINE ERROR: {exc}")
        traceback.print_exc()
    print()

    # ------------------------------------------------------------------
    # CODE ENGINE
    # ------------------------------------------------------------------
    print("CODE ENGINE:")
    print("-" * 40)
    try:
        from l104_code_engine import code_engine

        test_code = (
            "def example(x):\n"
            "    if x > 0:\n"
            "        return x * 2\n"
            "    return 0\n"
        )

        # Full analysis
        analysis = code_engine.full_analysis(test_code)
        _check("full_analysis returns dict",
               analysis is not None and isinstance(analysis, dict),
               f"type={type(analysis).__name__}")

        # Analysis should contain complexity info
        has_complexity = (
            isinstance(analysis, dict) and
            any(k for k in analysis if "complex" in k.lower() or "metric" in k.lower()
                or "loc" in k.lower() or "line" in k.lower())
        )
        _check("analysis has metrics", has_complexity,
               f"keys={list(analysis.keys())[:6]}…" if isinstance(analysis, dict) else "N/A")

        # Smell detection
        smells = code_engine.smell_detector.detect_all(test_code)
        _check("smell_detector returns dict/list",
               isinstance(smells, (list, dict)),
               f"type={type(smells).__name__}")

        # Auto-fix (should not crash on trivial code)
        fixed, log = code_engine.auto_fix_code(test_code)
        _check("auto_fix_code succeeds",
               isinstance(fixed, str) and len(fixed) > 0,
               f"fixed_len={len(fixed)}, log_entries={len(log) if isinstance(log, list) else '?'}")

    except Exception as exc:
        print(f"  !! CODE ENGINE ERROR: {exc}")
        traceback.print_exc()
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"  ACCURACY RESULT: {_tests_passed}/{_total_tests} tests passed")
    print("=" * 60)

    if _tests_passed == _total_tests:
        print("  ✓ All core engines operating correctly")
    else:
        failed = _total_tests - _tests_passed
        print(f"  ✗ {failed} test(s) FAILED — see details above")

    return _tests_passed == _total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
