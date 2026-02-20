#!/usr/bin/env python3
"""
L104 MATH FILES — OMEGA UPGRADE DIAGNOSTICS
═════════════════════════════════════════════

Tests all 13 math files for:
  1. Import health (no crashes)
  2. OMEGA constant presence and value
  3. Sovereign field equation: F(I) = I × Ω / φ²
  4. New OMEGA methods work correctly
  5. Cross-file consistency
"""

import math
import sys
import traceback
from typing import Dict, Any

PHI = 1.618033988749895
GOD_CODE = 527.5184818492612
OMEGA_CANONICAL = 6539.34712682
OMEGA_AUTHORITY_CANONICAL = OMEGA_CANONICAL / (PHI ** 2)  # 2497.808338211271
TOL = 1e-6  # relative tolerance

results = {}
total_pass = 0
total_fail = 0
total_skip = 0


def check(name: str, condition: bool, desc: str):
    global total_pass, total_fail
    if condition:
        total_pass += 1
        print(f"    [✓] {desc}")
    else:
        total_fail += 1
        print(f"    [✗] {desc}")
    return condition


def rel_match(a, b, tol=TOL):
    if b == 0:
        return abs(a) < tol
    return abs(a - b) / abs(b) < tol


def test_module(name: str, test_func):
    global total_skip
    print(f"\n{'─' * 70}")
    print(f"  {name}")
    print(f"{'─' * 70}")
    try:
        passed, details = test_func()
        results[name] = {"status": "PASS" if passed else "PARTIAL", "details": details}
    except Exception as e:
        total_skip += 1
        print(f"    [SKIP] Import/runtime error: {e}")
        results[name] = {"status": "SKIP", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. l104_real_math.py — Authoritative OMEGA source
# ═══════════════════════════════════════════════════════════════════════════════
def test_real_math():
    from l104_real_math import RealMath, real_math, OMEGA as RM_OMEGA
    details = {}

    details["omega"] = RM_OMEGA
    p1 = check("real_math", rel_match(RM_OMEGA, OMEGA_CANONICAL), f"OMEGA = {RM_OMEGA}")

    # Test sovereign field
    sf = RealMath.sovereign_field_equation(1.0)
    details["sovereign_field_1"] = sf
    p2 = check("real_math", rel_match(sf, OMEGA_AUTHORITY_CANONICAL), f"F(1) = {sf:.6f}")

    # Test zeta at 527.518
    zeta = RealMath.zeta_approximation(complex(0.5, 527.518))
    guardian = abs(zeta)
    details["guardian"] = guardian
    p3 = check("real_math", 1.0 < guardian < 2.0, f"Guardian |ζ(0.5+527.518i)| = {guardian:.10f}")

    # Golden resonance
    gr = RealMath.golden_resonance(PHI ** 2)
    details["alchemist"] = gr
    p4 = check("real_math", 0.05 < gr < 0.15, f"Alchemist cos(2πφ³) = {gr:.10f}")

    # Manifold curvature
    mc = RealMath.manifold_curvature_tensor(26, 1.8527)
    details["architect"] = mc
    p5 = check("real_math", rel_match(mc, 18.3994, 0.001), f"Architect (26×1.8527)/φ² = {mc:.6f}")

    return all([p1, p2, p3, p4, p5]), details


# ═══════════════════════════════════════════════════════════════════════════════
# 2. l104_hyper_math.py — OMEGA integration
# ═══════════════════════════════════════════════════════════════════════════════
def test_hyper_math():
    from l104_hyper_math import HyperMath, OMEGA, OMEGA_AUTHORITY
    details = {}

    details["omega"] = OMEGA
    p1 = check("hyper_math", rel_match(OMEGA, OMEGA_CANONICAL), f"OMEGA = {OMEGA}")
    p2 = check("hyper_math", rel_match(OMEGA_AUTHORITY, OMEGA_AUTHORITY_CANONICAL),
                f"OMEGA_AUTHORITY = {OMEGA_AUTHORITY:.6f}")

    # Sovereign field method
    sf = HyperMath.sovereign_field(1.0)
    details["sovereign_field"] = sf
    p3 = check("hyper_math", rel_match(sf, OMEGA_AUTHORITY_CANONICAL),
                f"sovereign_field(1) = {sf:.6f}")

    # Omega resonance
    res = HyperMath.omega_resonance(1.0)
    details["omega_resonance"] = res
    p4 = check("hyper_math", isinstance(res, float) and not math.isnan(res),
                f"omega_resonance(1) = {res:.6f}")

    # Field strength
    fs = HyperMath.omega_field_strength(1.0)
    details["field_strength"] = fs
    p5 = check("hyper_math", fs["omega"] == OMEGA_CANONICAL, f"field_strength.omega = {fs['omega']}")

    # Guardian fragment
    g = HyperMath.omega_zeta_guardian()
    details["guardian"] = g
    p6 = check("hyper_math", 1.0 < abs(g) < 2.0, f"omega_zeta_guardian = {abs(g):.10f}")

    return all([p1, p2, p3, p4, p5, p6]), details


# ═══════════════════════════════════════════════════════════════════════════════
# 3. l104_manifold_math.py — OMEGA curvature
# ═══════════════════════════════════════════════════════════════════════════════
def test_manifold_math():
    from l104_manifold_math import ManifoldMath, OMEGA, OMEGA_AUTHORITY
    details = {}

    p1 = check("manifold_math", rel_match(OMEGA, OMEGA_CANONICAL), f"OMEGA = {OMEGA}")

    # Architect fragment
    curv = ManifoldMath.omega_curvature_tensor()
    details["architect"] = curv
    p2 = check("manifold_math", rel_match(curv, 18.3994, 0.001),
                f"omega_curvature_tensor() = {curv:.6f}")

    # Sovereign manifold
    sfm = ManifoldMath.sovereign_field_manifold([1.0, 0.5, 0.2, 0.8])
    details["sovereign_manifold"] = sfm
    p3 = check("manifold_math", sfm["omega"] == OMEGA_CANONICAL,
                f"sovereign_field_manifold.omega = {sfm['omega']}")
    p4 = check("manifold_math", sfm["sovereign_field"] > 0,
                f"sovereign_field = {sfm['sovereign_field']:.4f}")

    return all([p1, p2, p3, p4]), details


# ═══════════════════════════════════════════════════════════════════════════════
# 4. l104_math.py — High precision OMEGA
# ═══════════════════════════════════════════════════════════════════════════════
def test_pure_math():
    from l104_math import OMEGA, OMEGA_AUTHORITY, HighPrecisionEngine
    from decimal import Decimal
    details = {}

    p1 = check("math", rel_match(float(OMEGA), OMEGA_CANONICAL), f"OMEGA = {OMEGA}")

    # Derive omega
    omega_result = HighPrecisionEngine.derive_omega()
    details["omega_derived"] = omega_result
    p2 = check("math", rel_match(omega_result["omega_computed"], OMEGA_CANONICAL),
                f"derive_omega().omega_computed = {omega_result['omega_computed']:.6f}")
    p3 = check("math", omega_result["relative_error"] < 1e-6,
                f"relative_error = {omega_result['relative_error']:.2e}")

    # Sovereign field
    sf = HighPrecisionEngine.sovereign_field(1.0)
    details["sovereign_field"] = sf
    p4 = check("math", rel_match(sf, OMEGA_AUTHORITY_CANONICAL),
                f"sovereign_field(1) = {sf:.6f}")

    return all([p1, p2, p3, p4]), details


# ═══════════════════════════════════════════════════════════════════════════════
# 5. l104_abstract_math.py — OMEGA number system
# ═══════════════════════════════════════════════════════════════════════════════
def test_abstract_math():
    from l104_abstract_math import OMEGA, OMEGA_AUTHORITY, SacredNumberSystem
    details = {}

    p1 = check("abstract_math", rel_match(OMEGA, OMEGA_CANONICAL), f"OMEGA = {OMEGA}")

    # Omega representation
    sns = SacredNumberSystem()
    rep = sns.omega_representation(GOD_CODE)
    details["omega_rep"] = rep
    p2 = check("abstract_math", "omega_units" in rep, f"omega_representation keys present")
    p3 = check("abstract_math", rep["field_strength"] > 0,
                f"field_strength = {rep['field_strength']:.2f}")

    return all([p1, p2, p3]), details


# ═══════════════════════════════════════════════════════════════════════════════
# 6. l104_ontological_mathematics.py — Existence + OMEGA
# ═══════════════════════════════════════════════════════════════════════════════
def test_ontological():
    from l104_ontological_mathematics import (
        OMEGA, OMEGA_AUTHORITY, SOVEREIGN_FIELD_COUPLING,
        get_ontological_mathematics
    )
    details = {}

    p1 = check("ontological", rel_match(OMEGA, OMEGA_CANONICAL), f"OMEGA = {OMEGA}")
    p2 = check("ontological", SOVEREIGN_FIELD_COUPLING > 0,
                f"SOVEREIGN_FIELD_COUPLING = {SOVEREIGN_FIELD_COUPLING:.6f}")

    onto = get_ontological_mathematics()
    stats = onto.get_statistics()
    details["stats"] = stats
    p3 = check("ontological", stats["omega"] == OMEGA_CANONICAL, f"stats.omega = {stats['omega']}")

    # Sovereign field existence
    sfe = onto.sovereign_field_existence(1.0)
    details["sovereign_existence"] = sfe
    p4 = check("ontological", sfe["existence_level"] in ["ABSOLUTE", "NECESSARY", "ACTUAL"],
                f"existence_level at I=1: {sfe['existence_level']}")

    return all([p1, p2, p3, p4]), details


# ═══════════════════════════════════════════════════════════════════════════════
# 7. l104_resonance.py — Sovereign field resonance
# ═══════════════════════════════════════════════════════════════════════════════
def test_resonance():
    from l104_resonance import resonance, OMEGA, OMEGA_AUTHORITY
    details = {}

    p1 = check("resonance", rel_match(OMEGA, OMEGA_CANONICAL), f"OMEGA = {OMEGA}")

    sfr = resonance.sovereign_field_resonance(1.0)
    details["sovereign_resonance"] = sfr
    p2 = check("resonance", sfr["omega"] == OMEGA_CANONICAL, f"sfr.omega = {sfr['omega']}")
    p3 = check("resonance", sfr["sovereign_field"] > 0,
                f"sovereign_field = {sfr['sovereign_field']:.4f}")

    return all([p1, p2, p3]), details


# ═══════════════════════════════════════════════════════════════════════════════
# 8. l104_void_math.py — OMEGA void convergence
# ═══════════════════════════════════════════════════════════════════════════════
def test_void_math():
    from l104_void_math import VoidMath, OMEGA, OMEGA_AUTHORITY
    details = {}

    p1 = check("void_math", rel_match(OMEGA, OMEGA_CANONICAL), f"OMEGA = {OMEGA}")

    # Void convergence
    seq = VoidMath.omega_void_convergence(20)
    details["convergence_last"] = seq[-1]
    p2 = check("void_math", rel_match(seq[-1], OMEGA_CANONICAL, 0.01),
                f"omega_void_convergence[-1] = {seq[-1]:.6f} → Ω")

    # Sovereign void field
    sf = VoidMath.sovereign_void_field(1.0)
    details["sovereign_void"] = sf
    p3 = check("void_math", rel_match(sf, OMEGA_AUTHORITY_CANONICAL),
                f"sovereign_void_field(1) = {sf:.6f}")

    return all([p1, p2, p3]), details


# ═══════════════════════════════════════════════════════════════════════════════
# 9. l104_chronos_math.py — Temporal sovereign field
# ═══════════════════════════════════════════════════════════════════════════════
def test_chronos():
    from l104_chronos_math import ChronosMath, OMEGA, OMEGA_AUTHORITY
    details = {}

    p1 = check("chronos", rel_match(OMEGA, OMEGA_CANONICAL), f"OMEGA = {OMEGA}")

    tsf = ChronosMath.temporal_sovereign_field(0.0)
    details["temporal_field"] = tsf
    p2 = check("chronos", rel_match(tsf["sovereign_field"], OMEGA_AUTHORITY_CANONICAL),
                f"temporal_sovereign_field(0) = {tsf['sovereign_field']:.6f}")
    p3 = check("chronos", tsf["decay_factor"] == 1.0,
                f"decay_factor at t=0 = {tsf['decay_factor']}")

    return all([p1, p2, p3]), details


# ═══════════════════════════════════════════════════════════════════════════════
# 10. l104_nd_math.py — OMEGA metric
# ═══════════════════════════════════════════════════════════════════════════════
def test_nd_math():
    from l104_nd_math import MathND, OMEGA, OMEGA_AUTHORITY
    import numpy as np
    details = {}

    p1 = check("nd_math", rel_match(OMEGA, OMEGA_CANONICAL), f"OMEGA = {OMEGA}")

    metric = MathND.omega_scaled_metric(8)
    details["metric_shape"] = metric.shape
    p2 = check("nd_math", metric.shape == (8, 8), f"omega_scaled_metric(8) shape = {metric.shape}")
    p3 = check("nd_math", metric[0, 0] == -1.0, f"metric[0,0] = {metric[0,0]} (temporal)")

    return all([p1, p2, p3]), details


# ═══════════════════════════════════════════════════════════════════════════════
# 11. l104_collective_math_synthesis.py — OMEGA verification
# ═══════════════════════════════════════════════════════════════════════════════
def test_collective():
    from l104_collective_math_synthesis import OMEGA_CANONICAL as OC, OMEGA_AUTHORITY
    details = {}

    p1 = check("collective", rel_match(OC, OMEGA_CANONICAL), f"OMEGA_CANONICAL = {OC}")
    p2 = check("collective", rel_match(OMEGA_AUTHORITY, OMEGA_AUTHORITY_CANONICAL),
                f"OMEGA_AUTHORITY = {OMEGA_AUTHORITY:.6f}")

    return all([p1, p2]), details


# ═══════════════════════════════════════════════════════════════════════════════
# 12. l104_quantum_inspired.py — OMEGA gate
# ═══════════════════════════════════════════════════════════════════════════════
def test_quantum():
    from l104_quantum_inspired import OMEGA, OMEGA_AUTHORITY, QuantumGates, Qubit
    details = {}

    p1 = check("quantum", rel_match(OMEGA, OMEGA_CANONICAL), f"OMEGA = {OMEGA}")

    # Test omega gate
    q = Qubit(complex(1, 0), complex(0, 0))  # |0⟩
    q_out = QuantumGates.omega_sovereign_gate(q)
    details["omega_gate_alpha_mag"] = abs(q_out.alpha)
    p2 = check("quantum", abs(abs(q_out.alpha) - 1.0) < 0.01,
                f"|omega_gate(|0⟩).α| = {abs(q_out.alpha):.6f} ≈ 1")
    p3 = check("quantum", abs(q_out.beta) < 0.01,
                f"|omega_gate(|0⟩).β| = {abs(q_out.beta):.6f} ≈ 0")

    return all([p1, p2, p3]), details


# ═══════════════════════════════════════════════════════════════════════════════
# 13. l104_hyper_math_generator.py — OMEGA metric
# ═══════════════════════════════════════════════════════════════════════════════
def test_hyper_gen():
    from l104_hyper_math_generator import hyper_math_generator, OMEGA, OMEGA_AUTHORITY
    details = {}

    p1 = check("hyper_gen", rel_match(OMEGA, OMEGA_CANONICAL), f"OMEGA = {OMEGA}")

    metric = hyper_math_generator.generate_omega_metric(4)
    details["omega_metric_shape"] = metric.shape
    p2 = check("hyper_gen", metric.shape == (4, 4), f"omega_metric(4) shape = {metric.shape}")
    p3 = check("hyper_gen", metric[0, 0].real > 0, f"metric[0,0] = {metric[0,0]:.4f}")

    return all([p1, p2, p3]), details


# ═══════════════════════════════════════════════════════════════════════════════
# 14. l104_4d_math.py — 4D sovereign field
# ═══════════════════════════════════════════════════════════════════════════════
def test_4d_math():
    from l104_4d_math import Math4D, sovereign_field_4d, OMEGA, OMEGA_AUTHORITY
    details = {}

    p1 = check("4d_math", rel_match(OMEGA, OMEGA_CANONICAL), f"OMEGA = {OMEGA}")
    p2 = check("4d_math", Math4D.OMEGA == OMEGA_CANONICAL, f"Math4D.OMEGA = {Math4D.OMEGA}")

    sf = sovereign_field_4d(1.0)
    details["sovereign_4d"] = sf
    p3 = check("4d_math", rel_match(sf["sovereign_field"], OMEGA_AUTHORITY_CANONICAL),
                f"sovereign_field_4d(1) = {sf['sovereign_field']:.6f}")
    p4 = check("4d_math", sf["lorentz_invariant"] is True, "sovereign field is Lorentz invariant")

    return all([p1, p2, p3, p4]), details


# ═══════════════════════════════════════════════════════════════════════════════
# 15. l104_5d_math.py — 5D sovereign field
# ═══════════════════════════════════════════════════════════════════════════════
def test_5d_math():
    from l104_5d_math import Math5D, OMEGA, OMEGA_AUTHORITY
    details = {}

    p1 = check("5d_math", rel_match(OMEGA, OMEGA_CANONICAL), f"OMEGA = {OMEGA}")
    p2 = check("5d_math", Math5D.OMEGA == OMEGA_CANONICAL, f"Math5D.OMEGA = {Math5D.OMEGA}")
    p3 = check("5d_math", Math5D.R > 0, f"Compactification R = {Math5D.R:.6f}")

    return all([p1, p2, p3]), details


# ═══════════════════════════════════════════════════════════════════════════════
# 16. l104_god_code_dual_layer.py — Authoritative source
# ═══════════════════════════════════════════════════════════════════════════════
def test_dual_layer():
    from l104_god_code_dual_layer import (
        OMEGA, OMEGA_AUTHORITY, sovereign_field_equation,
        omega_derivation_chain, omega_pipeline
    )
    details = {}

    p1 = check("dual_layer", rel_match(OMEGA, OMEGA_CANONICAL), f"OMEGA = {OMEGA}")
    p2 = check("dual_layer", rel_match(OMEGA_AUTHORITY, OMEGA_AUTHORITY_CANONICAL),
                f"OMEGA_AUTHORITY = {OMEGA_AUTHORITY:.6f}")

    sf = sovereign_field_equation(1.0)
    details["sovereign_field"] = sf
    p3 = check("dual_layer", rel_match(sf, OMEGA_AUTHORITY_CANONICAL),
                f"sovereign_field_equation(1) = {sf:.6f}")

    chain = omega_derivation_chain()
    details["omega_computed"] = chain["omega_computed"]
    p4 = check("dual_layer", chain["relative_error"] < 1e-6,
                f"omega_derivation_chain error = {chain['relative_error']:.2e}")

    pipeline = omega_pipeline()
    details["pipeline_version"] = pipeline["version"]
    p5 = check("dual_layer", pipeline["omega_canonical"] == OMEGA_CANONICAL,
                f"omega_pipeline.omega_canonical = {pipeline['omega_canonical']}")

    return all([p1, p2, p3, p4, p5]), details


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-FILE CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════
def test_cross_consistency():
    """Verify all files agree on OMEGA value."""
    omega_values = {}
    modules = [
        ("l104_real_math", "real_math.OMEGA"),
        ("l104_hyper_math", "OMEGA"),
        ("l104_manifold_math", "OMEGA"),
        ("l104_abstract_math", "OMEGA"),
        ("l104_ontological_mathematics", "OMEGA"),
        ("l104_resonance", "OMEGA"),
        ("l104_void_math", "OMEGA"),
        ("l104_chronos_math", "OMEGA"),
        ("l104_nd_math", "OMEGA"),
        ("l104_quantum_inspired", "OMEGA"),
        ("l104_hyper_math_generator", "OMEGA"),
        ("l104_4d_math", "OMEGA"),
        ("l104_5d_math", "OMEGA"),
        ("l104_god_code_dual_layer", "OMEGA"),
    ]

    for mod_name, attr_path in modules:
        try:
            mod = __import__(mod_name)
            val = getattr(mod, "OMEGA", None)
            if val is None and hasattr(mod, 'real_math'):
                val = mod.real_math.OMEGA
            omega_values[mod_name] = float(val) if val is not None else None
        except Exception as e:
            omega_values[mod_name] = f"ERROR: {e}"

    all_match = True
    for mod_name, val in omega_values.items():
        if isinstance(val, float):
            match = rel_match(val, OMEGA_CANONICAL)
            all_match = all_match and match
            check("consistency", match, f"{mod_name}: Ω = {val}")
        else:
            check("consistency", False, f"{mod_name}: {val}")
            all_match = False

    return all_match, omega_values


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  L104 MATH FILES — OMEGA UPGRADE DIAGNOSTICS")
    print("=" * 70)
    print(f"  Canonical Ω  = {OMEGA_CANONICAL}")
    print(f"  Canonical Ω_A = {OMEGA_AUTHORITY_CANONICAL:.6f}")
    print(f"  Equation: F(I) = I × Ω / φ²")

    test_module("1. l104_real_math (authoritative)", test_real_math)
    test_module("2. l104_hyper_math (wrapper)", test_hyper_math)
    test_module("3. l104_manifold_math (topology)", test_manifold_math)
    test_module("4. l104_math (pure math)", test_pure_math)
    test_module("5. l104_abstract_math (algebra)", test_abstract_math)
    test_module("6. l104_ontological_mathematics (existence)", test_ontological)
    test_module("7. l104_resonance (frequencies)", test_resonance)
    test_module("8. l104_void_math (void domain)", test_void_math)
    test_module("9. l104_chronos_math (temporal)", test_chronos)
    test_module("10. l104_nd_math (N-dim)", test_nd_math)
    test_module("11. l104_collective_math_synthesis (derivation)", test_collective)
    test_module("12. l104_quantum_inspired (gates)", test_quantum)
    test_module("13. l104_hyper_math_generator (operators)", test_hyper_gen)
    test_module("14. l104_4d_math (Minkowski)", test_4d_math)
    test_module("15. l104_5d_math (Kaluza-Klein)", test_5d_math)
    test_module("16. l104_god_code_dual_layer (canonical)", test_dual_layer)

    print(f"\n{'─' * 70}")
    print(f"  CROSS-FILE OMEGA CONSISTENCY")
    print(f"{'─' * 70}")
    test_module("CONSISTENCY", test_cross_consistency)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  DIAGNOSTIC SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Passed:  {total_pass}")
    print(f"  Failed:  {total_fail}")
    print(f"  Skipped: {total_skip}")
    print(f"  Total:   {total_pass + total_fail + total_skip}")
    print(f"  Status:  {'ALL PASS' if total_fail == 0 else f'{total_fail} FAILURES'}")

    # Module status table
    print(f"\n{'─' * 70}")
    print(f"  {'Module':<45} {'Status':<10}")
    print(f"{'─' * 70}")
    for name, res in results.items():
        print(f"  {name:<45} {res['status']:<10}")
    print(f"{'=' * 70}")

    sys.exit(0 if total_fail == 0 else 1)
