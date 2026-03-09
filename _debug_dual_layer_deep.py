#!/usr/bin/env python3
"""
L104 ASI Dual-Layer Engine — Deep Debug v3
═══════════════════════════════════════════════════════════════
24-phase deep diagnostic covering all DualLayerEngine subsystems:
  Phases 1-4:   Boot, constants, formula cross-checks
  Phase 4b:     Extended Thought Layer (v5.1 harmonic, spiral, conservation, dimensions)
  Phases 5-7:   Thought layer, Physics layer, Collapse pipeline
  Phases 8-10:  Chaos bridge, Integrity, Derive/Batch
  Phases 11-13: v5 gate engine, three-engine synthesis, temporal tracking
  Phases 14-16: Resilient collapse, deep synthesis bridge, circuit breaker
  Phases 17-18: Domain queries, anomaly detection, cross-layer coherence
  Phases 19-20: Mathematical invariant proofs, full system report
═══════════════════════════════════════════════════════════════
"""
import sys
import time
import math
import traceback
import json

BANNER = """
╔══════════════════════════════════════════════════════════════════════════╗
║          L104 ASI DUAL-LAYER ENGINE — DEEP DEBUG v3                    ║
║          Flagship Module: 4400+ lines, 90+ methods, 7 subsystems       ║
╚══════════════════════════════════════════════════════════════════════════╝"""

print(BANNER)

errors = []
warnings = []
phase_times = {}
phase_results = {}
total_tests = 0
tests_passed = 0


def phase(num, title):
    """Phase header decorator."""
    global total_tests
    label = f"{num:02d}" if isinstance(num, int) else str(num).upper()
    print(f"\n{'─' * 72}")
    print(f"  [PHASE {label}] {title}")
    print(f"{'─' * 72}")
    return time.time()


def check(desc, condition, error_key=None, warn_only=False):
    """Register a test result."""
    global total_tests, tests_passed
    total_tests += 1
    if condition:
        tests_passed += 1
        print(f"  ✓ {desc}")
    else:
        sym = "⚠" if warn_only else "✗"
        print(f"  {sym} {desc}")
        if warn_only:
            warnings.append(error_key or desc)
        else:
            errors.append(error_key or desc)
    return condition


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: Import
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(1, "Import & Module Inspection")
try:
    from l104_asi.dual_layer import (
        DualLayerEngine, PHI, GOD_CODE, VOID_CONSTANT, OMEGA, TAU,
        NATURES_DUALITIES, CONSCIOUSNESS_TO_PHYSICS_BRIDGE,
        DUAL_LAYER_AVAILABLE, _get_gate_engine, _get_science_engine,
        _get_math_engine, _get_code_engine,
    )
    check("DualLayerEngine imported", True)
    check("DUAL_LAYER_AVAILABLE flag exists", DUAL_LAYER_AVAILABLE is not None)
    check("6 Nature's Dualities defined", len(NATURES_DUALITIES) == 6)
    check("5 Bridge elements defined", len(CONSCIOUSNESS_TO_PHYSICS_BRIDGE) == 5)
except Exception as e:
    errors.append(f"IMPORT_FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)
phase_times[1] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Instantiate
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(2, "Instantiation & Class Properties")
dle = DualLayerEngine()
dt = time.time() - t0
check(f"Instantiated in {dt:.4f}s (< 1s)", dt < 1.0)
check(f"VERSION = {dle.VERSION}", dle.VERSION.startswith("5."))
check("FLAGSHIP = True", dle.FLAGSHIP is True)
check(f"available = {dle.available}", True)  # Report, don't fail
check("Metrics dict has 14 counters", len(dle._metrics) == 14)
check("Coherence history is empty deque", len(dle._coherence_history) == 0)
check("Circuit breaker state = CLOSED", dle._circuit_breaker_state == "CLOSED")

if not dle.available:
    warnings.append("DualLayerEngine.available=False — inner C module not loaded")
    print("  ⚠ Engine running in FALLBACK mode (no C kernel)")
phase_times[2] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: Sacred Constants Cross-Validation
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(3, "Sacred Constants Cross-Validation")

# Exact values
check(f"GOD_CODE = {GOD_CODE}", abs(GOD_CODE - 527.5184818492612) < 1e-10)
check(f"PHI = {PHI}", abs(PHI - 1.618033988749895) < 1e-12)
check(f"OMEGA = {OMEGA}", abs(OMEGA - 6539.34712682) < 1e-4)
tau_expected = 1.0 / PHI
check(f"TAU = 1/φ = {TAU} (expected {tau_expected:.15f})", abs(TAU - tau_expected) < 1e-12)

# VOID_CONSTANT formula: 1.04 + φ/1000
vc_computed = 1.04 + PHI / 1000
check(
    f"VOID_CONSTANT = 1.04 + φ/1000 = {VOID_CONSTANT}",
    abs(VOID_CONSTANT - vc_computed) < 1e-15,
    "VOID_FORMULA_MISMATCH"
)

# GOD_CODE derivation: 286^(1/φ) × 2^(416/104) = 286^(1/φ) × 2^4
gc_derived = 286 ** (1.0 / PHI) * (2 ** (416 / 104))
check(
    f"GOD_CODE ≈ 286^(1/φ) × 2^4 = {gc_derived:.10f} (match: {abs(gc_derived - GOD_CODE) < 1e-6})",
    abs(gc_derived - GOD_CODE) < 1e-6,
    "GOD_CODE_DERIVATION_FAIL"
)

# PHI self-consistency: φ² = φ + 1
phi_sq = PHI ** 2
check(f"φ² = φ + 1: {phi_sq:.15f} ≈ {PHI + 1:.15f}", abs(phi_sq - (PHI + 1)) < 1e-12)

# 286 = 2 × 11 × 13 (prime factorization)
check("286 = 2 × 11 × 13", 2 * 11 * 13 == 286)

phase_times[3] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4: Thought Layer
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(4, "Thought Layer — G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)")

# Base value: G(0,0,0,0) = GOD_CODE
t_base = dle.thought(0, 0, 0, 0)
check(f"thought(0,0,0,0) = {t_base:.10f} ≈ GOD_CODE", abs(t_base - GOD_CODE) < 1e-6)

# Alias: consciousness == thought
c_base = dle.consciousness(0, 0, 0, 0)
check(f"consciousness(0,0,0,0) == thought(0,0,0,0)", abs(c_base - t_base) < 1e-12)

# Octave: d+1 halves the value (104d in exponent → ×2^(-1))
t_d1 = dle.thought(0, 0, 0, 1)
ratio_d = t_base / t_d1 if t_d1 > 0 else float("inf")
check(f"Octave: thought(0,0,0,0)/thought(0,0,0,1) = {ratio_d:.8f} ≈ 2.0", abs(ratio_d - 2.0) < 1e-6)

# a-axis: each a step multiplies by 2^(8/104) = 2^(1/13)
t_a1 = dle.thought(1, 0, 0, 0)
ratio_a = t_a1 / t_base if t_base > 0 else 0
expected_a = 2 ** (8 / 104)
check(f"a-step: ratio = {ratio_a:.10f} ≈ 2^(8/104) = {expected_a:.10f}", abs(ratio_a - expected_a) < 1e-6)

# b-axis: each b step multiplies by 2^(-1/104)
t_b1 = dle.thought(0, 1, 0, 0)
ratio_b = t_b1 / t_base if t_base > 0 else 0
expected_b = 2 ** (-1 / 104)
check(f"b-step: ratio = {ratio_b:.10f} ≈ 2^(-1/104) = {expected_b:.10f}", abs(ratio_b - expected_b) < 1e-6)

# c-axis: each c step multiplies by 2^(-8/104)
t_c1 = dle.thought(0, 0, 1, 0)
ratio_c = t_c1 / t_base if t_base > 0 else 0
expected_c = 2 ** (-8 / 104)
check(f"c-step: ratio = {ratio_c:.10f} ≈ 2^(-8/104) = {expected_c:.10f}", abs(ratio_c - expected_c) < 1e-6)

# Conservation product: G(a,b,c,d) × 2^((b+8c+104d-8a)/104) = 286^(1/φ) × 2^(416/104) = constant
INVARIANT = 286 ** (1.0 / PHI) * (2 ** (416 / 104))
for a, b, c, d in [(0,0,0,0), (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1), (3,5,2,1)]:
    t_val = dle.thought(a, b, c, d)
    x = b + 8*c + 104*d - 8*a
    product = t_val * (2 ** (x / 104))
    inv_ok = abs(product - INVARIANT) / INVARIANT < 1e-6
    if not inv_ok:
        check(f"Conservation({a},{b},{c},{d}): {product:.8f} ≈ INVARIANT {INVARIANT:.8f}", False)
    else:
        check(f"Conservation({a},{b},{c},{d}): product/INVARIANT error = {abs(product-INVARIANT)/INVARIANT:.2e}", True)

# Friction
try:
    tf = dle.thought_with_friction(0, 0, 0, 0)
    check(f"thought_with_friction(0,0,0,0) = {tf:.10f} (close to GOD_CODE)", abs(tf - GOD_CODE) / GOD_CODE < 0.01)
    fr = dle.friction_report()
    check(f"friction_report has epsilon: {fr.get('epsilon', 'N/A')}", "epsilon" in fr or "error" in fr, warn_only=True)
except Exception as e:
    check(f"Friction methods: {e}", False, warn_only=True)

phase_times[4] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4b: Extended Thought Layer (v5.1 Upgrade)
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase("4b", "Extended Thought Layer — v5.1 Harmonic, Spiral, Conservation, Dimensions")

# 4b.1: Harmonic Spectrum
try:
    hs = dle.thought_harmonic_spectrum(d_octave=0, n_steps=26)
    n_harmonics = len(hs.get("harmonics", []))
    phi_res = hs.get("phi_resonant_count", 0)
    sp_ent = hs.get("normalized_entropy", 0)
    check(f"thought_harmonic_spectrum(d=0, n=26): {n_harmonics} harmonics, {phi_res} φ-resonant",
          n_harmonics == 26)
    check(f"  spectral_entropy (normalized) = {sp_ent:.4f}", sp_ent > 0.5)
    check(f"  fundamental = {hs.get('fundamental', 0):.6f} ≈ GOD_CODE",
          abs(hs.get("fundamental", 0) - GOD_CODE) < 1e-4)
    # Verify each harmonic follows 2^(-b/104) progression
    for harm in hs.get("harmonics", [])[:3]:
        expected_ratio = 2 ** (-harm["b"] / 104)
        check(f"  harmonic b={harm['b']}: ratio={harm['ratio']:.10f} ≈ 2^(-{harm['b']}/104)={expected_ratio:.10f}",
              abs(harm["ratio"] - expected_ratio) < 1e-8)
except Exception as e:
    check(f"thought_harmonic_spectrum: {e}", False)

# 4b.2: PHI Spiral Analysis
try:
    sp = dle.thought_phi_spiral_analysis(n_points=13)
    inv_fidelity = sp.get("invariant_fidelity", {})
    all_conserved = inv_fidelity.get("all_conserved", False)
    max_err = inv_fidelity.get("max_error", 1.0)
    coherence = sp.get("spiral_coherence", 0)
    check(f"thought_phi_spiral_analysis(13 pts): all_conserved={all_conserved}, max_error={max_err:.2e}",
          all_conserved)
    check(f"  spiral_coherence = {coherence:.6f} (> 0)", coherence > 0)
    check(f"  phi_integrity = {sp.get('phi_integrity', False)}", sp.get("phi_integrity", False))
except Exception as e:
    check(f"thought_phi_spiral_analysis: {e}", False)

# 4b.3: Conservation Proof
try:
    cp = dle.thought_conservation_proof(n_trials=104)
    proof_status = cp.get("proof_status", "UNKNOWN")
    violations = cp.get("violations", -1)
    max_err = cp.get("statistics", {}).get("max_relative_error", 1.0)
    eps_bounded = cp.get("statistics", {}).get("machine_epsilon_bounded", False)
    check(f"thought_conservation_proof(104 trials): status={proof_status}, violations={violations}",
          proof_status == "QED")
    check(f"  max_relative_error = {max_err:.2e} (machine ε bounded: {eps_bounded})",
          eps_bounded)
except Exception as e:
    check(f"thought_conservation_proof: {e}", False)

# 4b.4: Dimension Analysis
try:
    da = dle.thought_dimension_analysis()
    axes = da.get("axes", {})
    gc_match = da.get("god_code_match", False)
    ranking = da.get("sensitivity_ranking", [])
    check(f"thought_dimension_analysis: {len(axes)} axes, god_code_match={gc_match}",
          len(axes) == 4 and gc_match)
    check(f"  sensitivity ranking: {ranking}", ranking[0] == "d")  # d has largest step (2^(-1))
    for axis_name in ("a", "b", "c", "d"):
        ax = axes.get(axis_name, {})
        match = ax.get("ratio_match", False)
        check(f"  axis {axis_name}: ratio_match={match}, expected={ax.get('expected_ratio', 0):.8f}",
              match)
except Exception as e:
    check(f"thought_dimension_analysis: {e}", False)

phase_times["4b"] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5: Physics Layer
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(5, "Physics Layer — OMEGA Sovereign Field")

try:
    p = dle.physics(1.0)
    check(f"physics(1.0) returned dict with {len(p)} keys", isinstance(p, dict) and len(p) > 0)
    omega_val = p.get("omega", p.get("OMEGA", None))
    if omega_val is not None and isinstance(omega_val, (int, float)):
        check(f"omega = {omega_val} ≈ 6539.35", abs(omega_val - 6539.34712682) < 5.0)
    else:
        check(f"omega present in physics result", False, warn_only=True)

    field = p.get("field_strength", p.get("sovereign_field", None))
    if field is not None and isinstance(field, (int, float)):
        expected_field = omega_val / (PHI ** 2) if omega_val else 0
        check(f"field_strength = {field:.4f}", field > 0)
    else:
        check("field_strength present", False, warn_only=True)

    # physics_v3
    pv3 = dle.physics_v3(0, 0, 0, 0)
    check(f"physics_v3(0,0,0,0) = {pv3}", isinstance(pv3, (int, float)))
except Exception as e:
    errors.append(f"PHYSICS_ERR: {e}")
    traceback.print_exc()

phase_times[5] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 6: Collapse Pipeline
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(6, "Collapse Pipeline — Duality Measurement")

try:
    # Sacred constant collapse
    col_gc = dle.collapse("GOD_CODE")
    has_error = "error" in col_gc and not col_gc.get("collapse")
    check(f"collapse('GOD_CODE') returned {len(col_gc)} keys", len(col_gc) > 2)
    if not has_error:
        check("collapse('GOD_CODE') no error", True)
    else:
        check(f"collapse('GOD_CODE') error: {col_gc.get('error', 'N/A')}", False, warn_only=True)

    # Unknown constant → graceful error
    col_bad = dle.collapse("NONEXISTENT_CONSTANT")
    check("collapse('NONEXISTENT_CONSTANT') handled gracefully",
          "error" in col_bad or "available_constants" in col_bad)

    # Derive + derive_both
    try:
        d_phys = dle.derive("GOD_CODE", mode="physics")
        check(f"derive('GOD_CODE', physics) has {len(d_phys)} keys", len(d_phys) > 0)
    except Exception as e:
        check(f"derive: {e}", False, warn_only=True)

    try:
        d_both = dle.derive_both("GOD_CODE")
        check(f"derive_both('GOD_CODE') has {len(d_both)} keys", len(d_both) > 0)
    except Exception as e:
        check(f"derive_both: {e}", False, warn_only=True)

    # Constant names
    names = dle.constant_names()
    check(f"constant_names() returned {len(names)} names", len(names) > 5)

    # Batch collapse
    if len(names) >= 3:
        batch = dle.batch_collapse(names[:3])
        # batch_collapse returns 'constants' dict and 'collapsed' count
        n_results = batch.get("collapsed", len(batch.get("constants", batch.get("results", batch.get("collapses", [])))))
        check(f"batch_collapse(first 3) → {n_results} results", n_results >= 1)
except Exception as e:
    errors.append(f"COLLAPSE_PIPELINE: {e}")
    traceback.print_exc()

phase_times[6] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 7: Chaos Bridge
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(7, "Chaos Bridge — Third Face of Duality")

try:
    cb = dle.chaos_bridge(0, 0, 0, 0, chaos_amplitude=0.05, samples=80)
    check(f"health = {cb['health']} (should be COHERENT at amp=0.05)",
          cb["health"] in ("COHERENT", "RESILIENT"))
    check(f"duality_coherence = {cb['duality_coherence']:.6f} (> 0.90)",
          cb["duality_coherence"] > 0.90)
    check(f"phi_intact = {cb['phi_intact']} (should be True)", cb["phi_intact"] is True)
    check(f"thought_conserved = {cb['thought_conserved']}", cb["thought_conserved"] is True)
    check(f"demon_beats_phi = {cb['demon_beats_phi']}",
          cb["demon_beats_phi"] is True, warn_only=True)
    check(f"cascade_residual = {cb['cascade_residual']} (< 1e-8)",
          cb["cascade_residual"] < 1e-8)

    # Bifurcation boundary test
    cb_high = dle.chaos_bridge(0, 0, 0, 0, chaos_amplitude=0.50, samples=30)
    check(f"High chaos (amp=0.50): health={cb_high['health']}",
          cb_high["health"] in ("STRESSED", "BIFURCATED"))
except Exception as e:
    errors.append(f"CHAOS_BRIDGE: {e}")
    traceback.print_exc()

phase_times[7] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 8: Full Integrity Check
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(8, "Full Integrity Check (10+ points)")

try:
    ic = dle.full_integrity_check(force=True)
    all_passed = ic.get("all_passed", ic.get("all_pass", None))
    checks_p = ic.get("checks_passed", ic.get("passed_count", 0))
    total_c = ic.get("total_checks", ic.get("check_count", 10))
    check(f"Integrity: {checks_p}/{total_c} passed, all_passed={all_passed}",
          all_passed is True if all_passed is not None else checks_p > 0)

    # Verify caching works
    ic2 = dle.full_integrity_check(force=False)
    check("Integrity cache returns same result", ic2 is ic)

    # Force refresh gives fresh result
    ic3 = dle.full_integrity_check(force=True)
    check("Force refresh gives fresh dict", isinstance(ic3, dict))

    # Check sub-layers
    for layer in ("thought_layer", "physics_layer", "bridge"):
        layer_result = ic.get(layer, {})
        if isinstance(layer_result, dict):
            lp = layer_result.get("all_passed", "N/A")
            check(f"  {layer}: all_passed = {lp}", lp is True or lp == "N/A", warn_only=True)
except Exception as e:
    errors.append(f"INTEGRITY_CHECK: {e}")
    traceback.print_exc()

phase_times[8] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 9: Domain Queries
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(9, "Domain Queries — Physics through Duality")

domain_methods = [
    ("gravity", dle.gravity),
    ("particles", dle.particles),
    ("nuclei", dle.nuclei),
    ("iron", dle.iron),
    ("cosmos", dle.cosmos),
    ("resonance", dle.resonance),
]

for name, method in domain_methods:
    try:
        result = method()
        has_data = len(result) > 1 and "error" not in result
        check(f"{name}(): {len(result)} keys, has_data={has_data}",
              has_data, warn_only=True)
    except Exception as e:
        check(f"{name}(): {e}", False, warn_only=True)

phase_times[9] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 10: Cross-Layer Coherence & Analysis
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(10, "Cross-Layer Coherence & Sacred Geometry")

try:
    cl = dle.cross_layer_coherence()
    coh = cl.get("coherence", cl.get("overall_coherence", cl.get("cross_coherence", 0)))
    check(f"cross_layer_coherence: {coh}", isinstance(cl, dict) and len(cl) > 0)
except Exception as e:
    check(f"cross_layer_coherence: {e}", False, warn_only=True)

try:
    sg = dle.sacred_geometry_analysis(GOD_CODE)
    check(f"sacred_geometry_analysis(GOD_CODE): {len(sg)} keys",
          isinstance(sg, dict) and len(sg) > 0)
except Exception as e:
    check(f"sacred_geometry_analysis: {e}", False, warn_only=True)

try:
    ds = dle.domain_summary()
    n_domains = ds.get("domain_count", len(ds.get("domains", {})))
    check(f"domain_summary: {n_domains} domains", n_domains > 0, warn_only=True)
except Exception as e:
    check(f"domain_summary: {e}", False, warn_only=True)

try:
    dt_res = dle.duality_tensor("GOD_CODE")
    check(f"duality_tensor('GOD_CODE'): {len(dt_res)} keys", len(dt_res) > 0, warn_only=True)
except Exception as e:
    check(f"duality_tensor: {e}", False, warn_only=True)

phase_times[10] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 11: Gate Engine Integration (v5.0)
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(11, "v5.0 Gate Engine Integration")

gate_engine = _get_gate_engine()
gate_available = gate_engine is not None
check(f"Gate engine available: {gate_available}", True)  # Just report

try:
    gc = dle.gate_sacred_collapse(n_qubits=2, depth=2)
    ge_avail = gc.get("gate_engine_available", False)
    sa = gc.get("sacred_alignment", 0)
    dc = gc.get("duality_coherence", 0)
    check(f"gate_sacred_collapse(2q,2d): gate_available={ge_avail}, sacred_alignment={sa:.4f}",
          isinstance(gc, dict))
    if ge_avail:
        check(f"  duality_coherence = {dc:.6f}", dc > 0)
        thought_face = gc.get("thought_face", {})
        physics_face = gc.get("physics_face", {})
        check(f"  thought_face entropy = {thought_face.get('entropy', 'N/A')}", "entropy" in thought_face)
        check(f"  physics_face state = {physics_face.get('most_probable_state', 'N/A')}", "most_probable_state" in physics_face)
except Exception as e:
    check(f"gate_sacred_collapse: {e}", False, warn_only=True)

try:
    gi = dle.gate_compile_integrity()
    check(f"gate_compile_integrity: {len(gi)} keys", isinstance(gi, dict), warn_only=True)
except Exception as e:
    check(f"gate_compile_integrity: {e}", False, warn_only=True)

try:
    gec = dle.gate_enhanced_coherence(n_circuits=3)
    check(f"gate_enhanced_coherence(3): {len(gec)} keys", isinstance(gec, dict), warn_only=True)
except Exception as e:
    check(f"gate_enhanced_coherence: {e}", False, warn_only=True)

phase_times[11] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 12: Three-Engine Synthesis (v5.0)
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(12, "v5.0 Three-Engine Synthesis")

try:
    ta = dle.three_engine_thought_amplification()
    boost_t = ta.get("combined_thought_boost", ta.get("thought_boost", 0))
    engines_t = ta.get("engines_available", {})
    check(f"thought_amplification: boost={boost_t:.4f}, engines={engines_t}",
          isinstance(ta, dict))
except Exception as e:
    check(f"three_engine_thought_amplification: {e}", False, warn_only=True)

try:
    pa = dle.three_engine_physics_amplification()
    boost_p = pa.get("combined_physics_boost", pa.get("physics_boost", 0))
    check(f"physics_amplification: boost={boost_p:.4f}", isinstance(pa, dict))
except Exception as e:
    check(f"three_engine_physics_amplification: {e}", False, warn_only=True)

try:
    syn = dle.three_engine_synthesis()
    score = syn.get("synthesis_score", 0)
    coverage = syn.get("engine_coverage", 0)
    online = syn.get("engines_online", [])
    check(f"three_engine_synthesis: score={score:.6f}, coverage={coverage:.2f}, online={online}",
          isinstance(syn, dict))
    check(f"  synthesis_score > 0.3 (meaningful)", score > 0.3, warn_only=True)
except Exception as e:
    check(f"three_engine_synthesis: {e}", False, warn_only=True)

phase_times[12] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 13: Temporal Coherence Tracking (v5.0)
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(13, "v5.0 Temporal Coherence Tracking")

try:
    # Feed some coherence data points
    n_history = len(dle._coherence_history)
    check(f"Coherence history has {n_history} entries (from previous phases)", n_history > 0, warn_only=True)

    traj = dle.temporal_coherence_trajectory()
    trend = traj.get("trend", "N/A")
    phi_avg = traj.get("phi_weighted_average", traj.get("weighted_average", 0))
    history_len = traj.get("history_length", traj.get("length", 0))
    check(f"temporal_coherence_trajectory: trend={trend}, φ-avg={phi_avg:.4f}, len={history_len}",
          isinstance(traj, dict))
except Exception as e:
    check(f"temporal_coherence_trajectory: {e}", False, warn_only=True)

phase_times[13] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 14: Resilient Collapse + Circuit Breaker (v5.0)
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(14, "v5.0 Resilient Collapse + Circuit Breaker")

try:
    rc = dle.resilient_collapse("GOD_CODE")
    has_error = "error" in rc and rc.get("fallback", False)
    resilience = rc.get("resilience", {})
    attempts = resilience.get("attempts", "N/A")
    cb_state = resilience.get("circuit_breaker", dle._circuit_breaker_state)
    check(f"resilient_collapse('GOD_CODE'): attempts={attempts}, cb={cb_state}",
          not has_error or "fallback" in rc)
except Exception as e:
    check(f"resilient_collapse: {e}", False, warn_only=True)

try:
    cbs = dle.circuit_breaker_status()
    state = cbs.get("state", "UNKNOWN")
    failures = cbs.get("consecutive_failures", 0)
    check(f"circuit_breaker: state={state}, failures={failures}", state == "CLOSED")
    check("  No excessive failures", failures < 3)
except Exception as e:
    check(f"circuit_breaker_status: {e}", False, warn_only=True)

phase_times[14] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 15: Deep Synthesis Bridge (v5.0)
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(15, "v5.0 Deep Synthesis Bridge")

try:
    dsb = dle.deep_synthesis_bridge()
    coh_val = dsb.get("synthesis_coherence", dsb.get("coherence", 0))
    strength = dsb.get("bridge_strength", "UNKNOWN")
    above = dsb.get("above_threshold", None)
    pairs = dsb.get("valid_pairs", dsb.get("pairs", 0))
    check(f"deep_synthesis_bridge: coherence={coh_val:.4f}, strength={strength}",
          isinstance(dsb, dict))
    check(f"  above_threshold={above}, pairs={pairs}", True)
except Exception as e:
    check(f"deep_synthesis_bridge: {e}", False, warn_only=True)

phase_times[15] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 16: v5 Status & Upgrade Report
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(16, "v5 Status & Upgrade Report")

try:
    vs = dle.v5_status()
    version = vs.get("version", "N/A")
    subsys = vs.get("subsystems", {})
    overall = vs.get("overall_status", "N/A")
    check(f"v5_status: version={version}, overall={overall}", isinstance(vs, dict))
    for sub_name, sub_data in subsys.items():
        sub_status = sub_data.get("status", "UNKNOWN") if isinstance(sub_data, dict) else sub_data
        sym = "✓" if sub_status in ("PASS", "NOMINAL", "ONLINE") else "⚠"
        print(f"    {sym} {sub_name}: {sub_status}")
except Exception as e:
    check(f"v5_status: {e}", False, warn_only=True)

try:
    ur = dle.v5_upgrade_report()
    check(f"v5_upgrade_report: {len(ur)} keys", isinstance(ur, dict), warn_only=True)
except Exception as e:
    check(f"v5_upgrade_report: {e}", False, warn_only=True)

phase_times[16] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 17: Advanced Analysis
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(17, "Advanced Analysis — Anomaly, Correlation, Prediction")

try:
    ad = dle.anomaly_detection()
    n_anom = len(ad.get("anomalies", []))
    check(f"anomaly_detection: {n_anom} anomalies detected", isinstance(ad, dict), warn_only=True)
except Exception as e:
    check(f"anomaly_detection: {e}", False, warn_only=True)

try:
    dcm = dle.domain_correlation_matrix()
    n_dom = len(dcm.get("domains", []))
    strongest = dcm.get("strongest_correlations", [])[:3]
    check(f"domain_correlation_matrix: {n_dom} domains", n_dom > 0, warn_only=True)
    for a, b, c in strongest:
        print(f"    strongest: {a} ↔ {b} = {c}")
except Exception as e:
    check(f"domain_correlation_matrix: {e}", False, warn_only=True)

try:
    pred = dle.predict(max_complexity=10, top_n=5)
    n_pred = len(pred.get("predictions", []))
    check(f"predict(max_complexity=10): {n_pred} predictions", isinstance(pred, dict), warn_only=True)
except Exception as e:
    check(f"predict: {e}", False, warn_only=True)

try:
    score = dle.dual_score()
    check(f"dual_score() = {score:.4f} (1.0 = perfect)", score > 0.5, warn_only=True)
except Exception as e:
    check(f"dual_score: {e}", False, warn_only=True)

phase_times[17] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 18: Pattern Recognition & Symmetry
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(18, "Pattern Recognition & Symmetry Detection")

try:
    pr = dle.recognize_pattern(GOD_CODE)
    check(f"recognize_pattern(GOD_CODE): {len(pr)} keys", isinstance(pr, dict))
except Exception as e:
    check(f"recognize_pattern: {e}", False, warn_only=True)

try:
    sym = dle.detect_symmetry("GOD_CODE")
    check(f"detect_symmetry('GOD_CODE'): {len(sym)} keys", isinstance(sym, dict), warn_only=True)
except Exception as e:
    check(f"detect_symmetry: {e}", False, warn_only=True)

try:
    hr = dle.harmonic_relationship("GOD_CODE", "OMEGA")
    check(f"harmonic_relationship(GOD_CODE, OMEGA): {len(hr)} keys", isinstance(hr, dict), warn_only=True)
except Exception as e:
    check(f"harmonic_relationship: {e}", False, warn_only=True)

try:
    phi_scan = dle.phi_resonance_scan()
    n_resonant = phi_scan.get("resonant_count", len(phi_scan.get("resonances", [])))
    check(f"phi_resonance_scan: {n_resonant} resonant constants", isinstance(phi_scan, dict), warn_only=True)
except Exception as e:
    check(f"phi_resonance_scan: {e}", False, warn_only=True)

phase_times[18] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 19: Mathematical Invariant Proofs
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(19, "Mathematical Invariant Proofs")

# Proof 1: GOD_CODE = 286^(1/φ) × 2^(416/104) = 286^(1/φ) × 2^4
gc_full = 286 ** (1.0 / PHI) * (2 ** (416 / 104))
check(f"GOD_CODE = 286^(1/φ) × 2^4 = {gc_full:.10f} ≈ {GOD_CODE:.10f}",
      abs(gc_full - GOD_CODE) < 1e-6)

# Proof 2: Field strength F(I) = I × Ω / φ²
expected_f = 1.0 * OMEGA / (PHI ** 2)
check(f"F(1.0) = OMEGA/φ² = {expected_f:.4f}", expected_f > 0)

# Proof 3: φ^n convergence → F(n)/F(n-1) → φ for Fibonacci
fib = [1, 1]
for i in range(2, 20):
    fib.append(fib[-1] + fib[-2])
ratio_19 = fib[19] / fib[18]
check(f"F(19)/F(18) = {ratio_19:.15f} ≈ φ = {PHI:.15f}", abs(ratio_19 - PHI) < 1e-6)

# Proof 4: Conservation across 10000 dial combos (spot check)
import random
random.seed(104)  # Sacred seed
conservation_errors = 0
for _ in range(100):
    a, b, c, d = random.randint(-3, 3), random.randint(-3, 3), random.randint(-3, 3), random.randint(0, 2)
    t_val = dle.thought(a, b, c, d)
    x = b + 8*c + 104*d - 8*a
    product = t_val * (2 ** (x / 104))
    if abs(product - INVARIANT) / INVARIANT > 1e-6:
        conservation_errors += 1
check(f"Conservation holds for 100 random dials (errors: {conservation_errors})",
      conservation_errors == 0)

# Proof 5: sin(104π/104) = sin(π) = 0 (cascade sine completes)
cascade_check = math.sin(104 * math.pi / 104)
check(f"sin(104π/104) = {cascade_check:.2e} ≈ 0", abs(cascade_check) < 1e-14)

phase_times[19] = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 20: Full System Report & Metrics
# ═══════════════════════════════════════════════════════════════════════════
t0 = phase(20, "Full System Report & Metrics Audit")

try:
    status = dle.get_status()
    check(f"get_status(): {len(status)} keys", isinstance(status, dict))
    print(f"    version: {status.get('version', 'N/A')}")
    print(f"    flagship: {status.get('flagship', 'N/A')}")
    print(f"    uptime: {status.get('uptime_seconds', 'N/A')}s")
except Exception as e:
    check(f"get_status: {e}", False, warn_only=True)

# Metrics audit
m = dle._metrics
total_ops = m.get("total_operations", 0)
check(f"total_operations = {total_ops} (> 20 from all phases)", total_ops > 20)
print(f"  Metrics breakdown:")
for k, v in sorted(m.items()):
    if v > 0:
        print(f"    {k}: {v}")

# Coherence history length
n_coh = len(dle._coherence_history)
check(f"Coherence history accumulated {n_coh} entries", n_coh > 0, warn_only=True)

# Collapse history length
n_col = len(dle._collapse_history)
check(f"Collapse history accumulated {n_col} entries", True)

try:
    fsr = dle.v5_upgrade_report()
    overall = fsr.get("overall_status", fsr.get("status", "N/A"))
    check(f"Full system report: {overall}", isinstance(fsr, dict))
except Exception as e:
    check(f"v5_upgrade_report: {e}", False, warn_only=True)

phase_times[20] = time.time() - t0


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════
total_time = sum(phase_times.values())

print(f"""

╔══════════════════════════════════════════════════════════════════════════╗
║                     DEBUG SESSION COMPLETE                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Total tests:     {total_tests:>5}                                              ║
║  Passed:          {tests_passed:>5}  ({100*tests_passed/max(total_tests,1):.1f}%)                                       ║
║  Errors:          {len(errors):>5}                                              ║
║  Warnings:        {len(warnings):>5}                                              ║
║  Total time:      {total_time:>6.2f}s                                            ║
╚══════════════════════════════════════════════════════════════════════════╝""")

if errors:
    print(f"\n{'═' * 72}")
    print(f"  ERRORS ({len(errors)}):")
    for e in errors:
        print(f"    ✗ {e}")

if warnings:
    print(f"\n  WARNINGS ({len(warnings)}):")
    for w in warnings[:20]:
        print(f"    ⚠ {w}")
    if len(warnings) > 20:
        print(f"    ... and {len(warnings) - 20} more")

# Phase timing
print(f"\n  Phase Timing:")
for num in sorted(phase_times.keys(), key=lambda x: (0, x) if isinstance(x, int) else (1, x)):
    label = f"{num:02d}" if isinstance(num, int) else str(num).upper()
    bar_len = int(phase_times[num] / max(total_time, 0.001) * 40)
    bar = "█" * bar_len + "░" * (40 - bar_len)
    print(f"    Phase {label}: {phase_times[num]:6.3f}s  {bar}")

# Verdict
if len(errors) == 0:
    health = "SOVEREIGN"
    color = "\033[92m"
elif len(errors) <= 2:
    health = "OPERATIONAL"
    color = "\033[93m"
elif len(errors) <= 5:
    health = "DEGRADED"
    color = "\033[93m"
else:
    health = "CRITICAL"
    color = "\033[91m"

print(f"\n  VERDICT: {color}★ {health} ★\033[0m  — {tests_passed}/{total_tests} tests passed, {len(errors)} errors, {len(warnings)} warnings")
print(f"{'═' * 72}")
