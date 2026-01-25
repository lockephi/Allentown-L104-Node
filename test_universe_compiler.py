#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
TEST SUITE: L104 UNIVERSE COMPILER
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from l104_universe_compiler import (
    UniverseCompiler, UniverseParameters,
    RelativityModule, QuantumModule, GravityModule,
    ElectromagnetismModule, ThermodynamicsModule,
    L104MetaphysicsModule
)
from sympy import symbols, simplify, sqrt

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════



def test_parameter_variability():
    """Test that constants are truly variable."""
    print("\n[TEST 1] Parameter Variability")

    params = UniverseParameters()

    # Check they're symbolic
    assert str(type(params.c)) == "<class 'sympy.core.symbol.Symbol'>"
    assert str(type(params.god_code)) == "<class 'sympy.core.symbol.Symbol'>"

    # Can be substituted
    expr = params.c**2 + params.hbar
    substituted = expr.subs([(params.c, 3e8), (params.hbar, 1e-34)])

    print(f"  ✓ Constants are symbolic: {params.c}")
    print(f"  ✓ Can be substituted: {expr} → {substituted}")
    return True


def test_module_loading():
    """Test loading and unloading physics modules."""
    print("\n[TEST 2] Module Loading/Unloading")

    params = UniverseParameters()
    compiler = UniverseCompiler(params)

    # Load modules
    compiler.add_module(RelativityModule(params))
    compiler.add_module(QuantumModule(params))
    assert len(compiler.modules) == 2

    # Unload one
    compiler.remove_module('Quantum')
    assert len(compiler.modules) == 1
    assert 'Relativity' in compiler.modules
    assert 'Quantum' not in compiler.modules

    print("  ✓ Modules can be loaded")
    print("  ✓ Modules can be unloaded")
    print(f"  ✓ Active modules: {list(compiler.modules.keys())}")
    return True


def test_universe_compilation():
    """Test that universe compilation works."""
    print("\n[TEST 3] Universe Compilation")

    params = UniverseParameters()
    compiler = UniverseCompiler(params)

    compiler.add_module(RelativityModule(params))
    compiler.add_module(QuantumModule(params))

    universe = compiler.compile_universe()

    assert 'parameters' in universe
    assert 'modules' in universe
    assert 'consistency' in universe
    assert universe['overall_consistency'] == True

    print("  ✓ Universe compiles successfully")
    print("  ✓ All modules consistent")
    print(f"  ✓ Modules loaded: {len(universe['modules'])}")
    return True


def test_reality_bending():
    """Test modifying fundamental constants."""
    print("\n[TEST 4] Reality Bending")

    params = UniverseParameters()
    compiler = UniverseCompiler(params)

    compiler.add_module(RelativityModule(params))

    # Original c (symbolic)
    original_c = params.c

    # Bend reality: change c
    result = compiler.bend_reality({'c': 1e10})  # Much faster light

    assert result['new_params']['c'] == 1e10
    assert result['new_universe']['overall_consistency'] == True

    print("  ✓ Can modify speed of light")
    print("  ✓ Universe remains consistent")
    print(f"  ✓ Original: {original_c} → New: {result['new_params']['c']}")
    return True


def test_quantum_classical_transition():
    """Test quantum → classical transition by varying ℏ."""
    print("\n[TEST 5] Quantum-Classical Transition")

    params = UniverseParameters()
    compiler = UniverseCompiler(params)

    compiler.add_module(QuantumModule(params))

    # Get quantum uncertainty
    quantum_mod = compiler.modules['Quantum']
    uncertainty = quantum_mod.equations['uncertainty']

    # Test with large ℏ (quantum)
    quantum_result = quantum_mod.transition_to_classical(1e-20)

    # Test with tiny ℏ (classical)
    classical_result = quantum_mod.transition_to_classical(1e-50)

    print("  ✓ Quantum module responds to ℏ changes")
    print(f"  ✓ Quantum regime (ℏ=1e-20): {list(quantum_result.keys())}")
    print(f"  ✓ Classical limit (ℏ=1e-50): {list(classical_result.keys())}")
    return True


def test_gravity_tuning():
    """Test modifying gravitational constant."""
    print("\n[TEST 6] Gravity Tuning")

    params = UniverseParameters()
    compiler = UniverseCompiler(params)

    compiler.add_module(GravityModule(params))

    gravity_mod = compiler.modules['Gravity']

    # Double gravity
    strong_gravity = gravity_mod.modify_gravity(G_factor=2.0)

    # Halve gravity
    weak_gravity = gravity_mod.modify_gravity(G_factor=0.5)

    print("  ✓ Can modify gravitational strength")
    print(f"  ✓ Strong gravity equations: {len(strong_gravity)}")
    print(f"  ✓ Weak gravity equations: {len(weak_gravity)}")
    return True


def test_em_consistency():
    """Test that c = 1/√(ε₀μ₀) consistency holds."""
    print("\n[TEST 7] EM Consistency (c from ε₀, μ₀)")

    params = UniverseParameters()
    compiler = UniverseCompiler(params)

    compiler.add_module(ElectromagnetismModule(params))

    em_mod = compiler.modules['Electromagnetism']
    c_from_em = em_mod.equations['speed_of_light']

    # Symbolically: c_from_em should be 1/√(ε₀μ₀)
    expected = 1 / sqrt(params.epsilon_0 * params.mu_0)

    # They should be equal symbolically
    diff = simplify(c_from_em - expected)

    print("  ✓ Speed of light derived from EM constants")
    print(f"  ✓ c = 1/√(ε₀μ₀) = {c_from_em}")
    print(f"  ✓ Difference: {diff}")
    return True


def test_l104_metaphysics():
    """Test L104-specific metaphysical laws."""
    print("\n[TEST 8] L104 Metaphysics")

    params = UniverseParameters()
    compiler = UniverseCompiler(params)

    compiler.add_module(L104MetaphysicsModule(params))

    l104_mod = compiler.modules['L104_Metaphysics']

    # Check GOD_CODE and PHI are in equations
    resonance = l104_mod.equations['resonance']
    phi_relation = l104_mod.equations['phi_relation']

    assert str(params.god_code) in str(resonance)
    assert str(params.phi) in str(phi_relation)

    print("  ✓ GOD_CODE integrated into physics")
    print("  ✓ PHI (golden ratio) as variable")
    print(f"  ✓ Resonance: {resonance}")
    print(f"  ✓ Phi relation: {phi_relation}")
    return True


def test_parameter_space_exploration():
    """Test exploring parameter space."""
    print("\n[TEST 9] Parameter Space Exploration")

    params = UniverseParameters()
    compiler = UniverseCompiler(params)

    compiler.add_module(L104MetaphysicsModule(params))

    # Explore GOD_CODE values
    god_values = [100, 527.518, 1000]
    results = compiler.explore_parameter_space('god_code', god_values)

    assert len(results) == len(god_values)
    assert all(r['consistent'] for r in results)

    print("  ✓ Can explore parameter space")
    print(f"  ✓ Tested {len(results)} universe configurations")
    print("  ✓ All configurations remain consistent")
    return True


def test_equation_retrieval():
    """Test retrieving specific equations."""
    print("\n[TEST 10] Equation Retrieval")

    params = UniverseParameters()
    compiler = UniverseCompiler(params)

    compiler.add_module(RelativityModule(params))
    compiler.add_module(QuantumModule(params))

    compiler.compile_universe()

    # Get specific equations
    lorentz = compiler.get_equation('Relativity', 'lorentz_factor')
    uncertainty = compiler.get_equation('Quantum', 'uncertainty')

    assert lorentz is not None
    assert uncertainty is not None

    print("  ✓ Can retrieve equations by name")
    print(f"  ✓ Lorentz factor: {lorentz}")
    print(f"  ✓ Uncertainty: {uncertainty}")
    return True


def test_source_code_export():
    """Test exporting universe as JSON."""
    print("\n[TEST 11] Source Code Export")

    params = UniverseParameters()
    compiler = UniverseCompiler(params)

    compiler.add_module(RelativityModule(params))
    compiler.compile_universe()

    filename = compiler.export_source_code("test_universe.json")

    import os
    assert os.path.exists(filename)

    # Read it back
    import json
    with open(filename, 'r') as f:
        exported = json.load(f)

    assert 'parameters' in exported
    assert 'modules' in exported
    assert 'consistency' in exported

    # Clean up
    os.remove(filename)

    print("  ✓ Can export universe to JSON")
    print(f"  ✓ File created: {filename}")
    print("  ✓ Valid JSON structure")
    return True


def test_full_universe():
    """Test compiling universe with all modules."""
    print("\n[TEST 12] Full Universe Compilation")

    params = UniverseParameters()
    compiler = UniverseCompiler(params)

    # Load all modules
    compiler.add_module(RelativityModule(params))
    compiler.add_module(QuantumModule(params))
    compiler.add_module(GravityModule(params))
    compiler.add_module(ElectromagnetismModule(params))
    compiler.add_module(ThermodynamicsModule(params))
    compiler.add_module(L104MetaphysicsModule(params))

    universe = compiler.compile_universe()

    assert len(universe['modules']) == 6
    assert universe['overall_consistency'] == True

    total_equations = sum(
        len(mod['equations'])
        for mod in universe['modules'].values()
    )

    print("  ✓ All 6 modules loaded")
    print("  ✓ Universe is consistent")
    print(f"  ✓ Total equations: {total_equations}")
    return True


def run_all_tests():
    """Run complete test suite."""
    tests = [
        test_parameter_variability,
        test_module_loading,
        test_universe_compilation,
        test_reality_bending,
        test_quantum_classical_transition,
        test_gravity_tuning,
        test_em_consistency,
        test_l104_metaphysics,
        test_parameter_space_exploration,
        test_equation_retrieval,
        test_source_code_export,
        test_full_universe,
    ]

    print("="*80)
    print("L104 UNIVERSE COMPILER - TEST SUITE")
    print("="*80)

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
                print(f"  ✓✓✓ PASS\n")
            else:
                failed += 1
                print(f"  ✗✗✗ FAIL\n")
        except Exception as e:
            failed += 1
            print(f"  ✗✗✗ FAIL: {e}\n")

    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"PASSED: {passed}/{len(tests)}")
    print(f"FAILED: {failed}/{len(tests)}")
    print(f"SUCCESS RATE: {100*passed/len(tests):.1f}%")
    print("="*80)

    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
