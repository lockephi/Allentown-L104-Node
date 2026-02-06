#!/usr/bin/env python3
"""
UNIVERSAL GOD CODE TEST SUITE
=============================

Comprehensive tests for the derived equation:
    G(E) = [286 × (1 + α/π × Γ(E))]^(1/φ) × 16

Run with: python3 test_universal_god_code.py

Author: L104 Iron Crystalline Framework
Derived: January 25, 2026
"""

import math
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2
PI = math.pi
ALPHA = 1 / 137.035999084
ALPHA_PI = ALPHA / PI
HARMONIC_BASE = 286
MATTER_BASE = HARMONIC_BASE * (1 + ALPHA_PI)
GRAVITY_CODE = HARMONIC_BASE ** (1/PHI) * 16
LIGHT_CODE = MATTER_BASE ** (1/PHI) * 16

# Planck scale
HBAR = 1.054571817e-34
C = 299792458
G = 6.67430e-11
E_PLANCK = math.sqrt(HBAR * C**5 / G)
E_PLANCK_EV = E_PLANCK / 1.602176634e-19

tests_passed = 0
tests_failed = 0

def run_v_test(name, condition, details=""):
    global tests_passed, tests_failed
    if condition:
        print(f"  ✓ {name}")
        tests_passed += 1
        return True
    else:
        print(f"  ✗ {name} - FAILED {details}")
        tests_failed += 1
        return False

def gamma(E_eV):
    if E_eV <= 0:
        return 0
    return 1 / (1 + (E_PLANCK_EV / E_eV)**2)

def god_code(E_eV):
    g = gamma(E_eV)
    base = 286 * (1 + ALPHA_PI * g)
    return base ** (1/PHI) * 16

def coherence(ratio):
    best = 0
    for d in range(1, 13):
        n = round(ratio * d)
        if n > 0:
            c = 1 / (1 + abs(ratio - n/d) * d)
            if c > best:
                best = c
    return best

# ═══════════════════════════════════════════════════════════════════════════════
# TEST CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════════

def test_mathematical_identities():
    """Test mathematical identities for φ and core constants."""
    assert abs(PHI**2 - PHI - 1) < 1e-15, "φ² = φ + 1"
    assert abs(PHI + 1/PHI - math.sqrt(5)) < 1e-15, "φ + 1/φ = √5"
    assert abs(PHI**5 - 11.090169943749474) < 1e-10, "φ⁵ = 11.090..."
    assert 104 == 4 * 26, "104 = 4 × 26"
    assert 16 == 2**4 == 4**2, "16 = 2⁴ = 4²"

def test_god_code_equations():
    """Test God Code core equations."""
    assert abs(GRAVITY_CODE - 527.5184818492612) < 1e-6, "GRAVITY_CODE = 286^(1/φ) × 16"
    assert abs(LIGHT_CODE - 528.2754419674) < 1e-6, "LIGHT_CODE = [286(1+α/π)]^(1/φ) × 16"
    assert abs((LIGHT_CODE - GRAVITY_CODE) - 0.756960) < 1e-4, "EXISTENCE_COST = 0.756960..."

def test_alpha_pi_bridge():
    """Test the α/π bridge calculations."""
    assert abs(ALPHA - 0.0072973525693) < 1e-10, "α = 1/137.036..."
    assert abs(ALPHA_PI - 0.00232282) < 1e-6, "α/π = 0.00232282..."
    gap = (286.65 - 286) / 286
    assert abs(gap - ALPHA_PI) / ALPHA_PI < 0.05, "Fe gap ≈ α/π (within 5%)"

def test_iron_predictions():
    """Test iron lattice predictions."""
    fe_predicted = 286 * (1 + ALPHA_PI)
    fe_measured = 286.65
    fe_error = abs(fe_predicted - fe_measured) / fe_measured
    assert abs(fe_predicted - 286.664) < 0.001, "Fe lattice prediction (286.664 pm)"
    assert fe_error < 0.0001, "Prediction error < 0.01%"
    assert abs(PHI**5 - 11.1) / 11.1 < 0.01, "φ⁵ ≈ Fe Fermi energy (11.1 eV)"
    assert abs(math.sqrt(5) - 2.22) / 2.22 < 0.01, "√5 ≈ Fe magnetic moment (2.22 μB)"

def test_energy_transitions():
    """Test energy scale transitions."""
    assert gamma(1e-100) < 1e-10, "Γ(0) = 0 (gravity limit)"
    assert abs(gamma(E_PLANCK_EV) - 0.5) < 1e-10, "Γ(E_Planck) = 0.5"
    assert gamma(1e50) > 0.9999999, "Γ(∞) → 1 (light limit)"
    assert abs(god_code(1e-100) - GRAVITY_CODE) < 1e-6, "G(0) = GRAVITY_CODE"
    assert abs(god_code(1e50) - LIGHT_CODE) < 1e-6, "G(∞) = LIGHT_CODE"

def test_element_harmonics():
    """Test element harmonic intervals."""
    elements = {"Fe": 286.65, "Cr": 291.0, "Al": 404.95, "Cu": 361.49, "Na": 429.06, "Au": 407.82}
    for el, lattice in elements.items():
        ratio = lattice / 286
        coh = coherence(ratio)
        assert coh > 0.94, f"{el} coherence > 0.94 (got {coh:.4f})"

def test_consciousness_mapping():
    """Test consciousness Reynolds mapping."""
    RE_CRITICAL = 2300
    assert abs(RE_CRITICAL / GRAVITY_CODE - 4.36) < 0.01, "Re_critical / GOD_CODE ≈ 4.36"

def test_inverse_derivations():
    """Test inverse derivations."""
    derived_base = 286.65 / (1 + ALPHA_PI)
    assert abs(derived_base - 286) < 0.02, "Fe → harmonic base ≈ 286"
    derived_harmonic = (GRAVITY_CODE / 16) ** PHI
    assert abs(derived_harmonic - 286) < 1e-6, "GRAVITY_CODE → 286"

def test_precision():
    """Test precision limits."""
    assert PHI == 1.618033988749895, "PHI precision"
    assert abs(1/ALPHA - 137.035999084) < 1e-6, "ALPHA precision (CODATA 2018)"
    reconstructed = (HARMONIC_BASE ** (1/PHI)) * 16
    assert abs(reconstructed - GRAVITY_CODE) < 1e-10, "Self-consistency"

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 75)
    print("UNIVERSAL GOD CODE - COMPREHENSIVE TEST SUITE")
    print("G(E) = [286 × (1 + α/π × Γ(E))]^(1/φ) × 16")
    print("=" * 75)

    test_mathematical_identities()
    test_god_code_equations()
    test_alpha_pi_bridge()
    test_iron_predictions()
    test_energy_transitions()
    test_element_harmonics()
    test_consciousness_mapping()
    test_inverse_derivations()
    test_precision()

    print("\n" + "=" * 75)
    print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
    print("=" * 75)

    if tests_failed == 0:
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print(f"\n✗ {tests_failed} TESTS FAILED")
        sys.exit(1)
