#!/usr/bin/env python3
"""
L104 ASI Dual-Layer Engine — Deep Debug Session
Exercises all core methods and validates constants, layers, collapse, chaos bridge.
"""
import sys
import time
import traceback
import math

print("=" * 72)
print("  L104 ASI DUAL-LAYER ENGINE — DEEP DEBUG SESSION")
print("=" * 72)
print()

errors = []
warnings = []

# ─── Phase 1: Import ───
print("[PHASE 1] Import...")
try:
    from l104_asi.dual_layer import DualLayerEngine
    print("  ✓ DualLayerEngine imported")
except Exception as e:
    errors.append(f"IMPORT_FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

# ─── Phase 2: Instantiate ───
print("[PHASE 2] Instantiate...")
t0 = time.time()
dle = DualLayerEngine()
dt = time.time() - t0
print(f"  ✓ Instantiated in {dt:.4f}s")
print(f"  available = {dle.available}")
print(f"  VERSION = {dle.VERSION}")
print(f"  FLAGSHIP = {dle.FLAGSHIP}")

if not dle.available:
    warnings.append("DualLayerEngine.available is False — underlying _dual_layer not loaded")
    print("  ⚠ Engine NOT available (inner _dual_layer not loaded)")

# ─── Phase 3: Constants integrity ───
print("[PHASE 3] Constants integrity...")
from l104_asi.dual_layer import PHI, GOD_CODE, VOID_CONSTANT, OMEGA, TAU
checks = {
    "GOD_CODE": (GOD_CODE, 527.5184818492612),
    "PHI": (PHI, 1.618033988749895),
    "VOID_CONSTANT": (VOID_CONSTANT, 1.0416180339887497),
    "OMEGA": (OMEGA, 6539.34712682),
    "TAU (φ-conjugate)": (TAU, 1.0 / 1.618033988749895),  # TAU = 1/φ in L104 (NOT 2π)
}
for name, (actual, expected) in checks.items():
    ok = abs(actual - expected) < 1e-8
    sym = "✓" if ok else "✗"
    print(f"  {sym} {name} = {actual} (expected {expected})")
    if not ok:
        errors.append(f"CONST_MISMATCH: {name} = {actual}, expected {expected}")

# Cross-check: VOID_CONSTANT = 1.04 + PHI/1000
vc_computed = 1.04 + PHI / 1000
vc_ok = abs(VOID_CONSTANT - vc_computed) < 1e-15
print(f"  {'✓' if vc_ok else '✗'} VOID_CONSTANT formula: 1.04 + φ/1000 = {vc_computed} (match: {vc_ok})")
if not vc_ok:
    errors.append(f"VOID_FORMULA: computed {vc_computed} != stored {VOID_CONSTANT}")

# ─── Phase 4: Thought layer ───
print("[PHASE 4] Thought layer...")
try:
    t = dle.thought(0, 0, 0, 0)
    ok = abs(t - GOD_CODE) < 1e-6
    print(f"  thought(0,0,0,0) = {t}  (GOD_CODE match: {ok})")
    if not ok:
        errors.append(f"THOUGHT_BASE: {t} != {GOD_CODE}")

    t2 = dle.thought(1, 0, 0, 0)
    print(f"  thought(1,0,0,0) = {t2}")

    # Octave test: thought(0,0,0,1) should be exactly half of thought(0,0,0,0)
    t_half = dle.thought(0, 0, 0, 1)
    ratio = t / t_half if t_half != 0 else float("inf")
    octave_ok = abs(ratio - 2.0) < 1e-6
    print(f"  thought(0,0,0,1) = {t_half}  (octave ratio = {ratio:.6f}, ok = {octave_ok})")
    if not octave_ok:
        errors.append(f"OCTAVE_FAIL: ratio = {ratio}")

    # Consciousness synonym
    c = dle.consciousness(0, 0, 0, 0)
    cons_ok = abs(c - t) < 1e-12
    print(f"  consciousness(0,0,0,0) = {c}  (matches thought: {cons_ok})")
    if not cons_ok:
        errors.append(f"CONSCIOUSNESS_MISMATCH: {c} != {t}")

    # Friction
    tf = dle.thought_with_friction(0, 0, 0, 0)
    print(f"  thought_with_friction(0,0,0,0) = {tf}")
    fr = dle.friction_report()
    print(f"  friction_report: epsilon = {fr.get('epsilon', 'N/A')}, improvements = {fr.get('improvements', 'N/A')}")
except Exception as e:
    errors.append(f"THOUGHT_ERR: {e}")
    traceback.print_exc()

# ─── Phase 5: Physics layer ───
print("[PHASE 5] Physics layer...")
try:
    p = dle.physics(1.0)
    print(f"  physics(1.0) keys: {sorted(p.keys())}")
    omega_val = p.get("omega", "MISSING")
    field = p.get("field_strength", "MISSING")
    print(f"  omega = {omega_val}")
    print(f"  field_strength = {field}")
    if isinstance(omega_val, (int, float)):
        if abs(omega_val - 6539.34712682) > 1.0:
            warnings.append(f"OMEGA_DRIFT: {omega_val} vs 6539.34712682")

    pv3 = dle.physics_v3(0, 0, 0, 0)
    print(f"  physics_v3(0,0,0,0) = {pv3}")
except Exception as e:
    errors.append(f"PHYSICS_ERR: {e}")
    traceback.print_exc()

# ─── Phase 6: Collapse ───
print("[PHASE 6] Collapse...")
try:
    col = dle.collapse("GOD_CODE")
    print(f"  collapse('GOD_CODE') keys: {sorted(col.keys())[:8]}...")
    if "error" in col:
        warnings.append(f"COLLAPSE_ERR: {col['error']}")
        print(f"  ⚠ collapse error: {col['error']}")
    else:
        print(f"  collapse ok — thought = {col.get('thought', 'N/A')}, physics = {col.get('physics', 'N/A')}")
except Exception as e:
    errors.append(f"COLLAPSE_ERR: {e}")
    traceback.print_exc()

# ─── Phase 7: Chaos bridge ───
print("[PHASE 7] Chaos bridge (light)...")
try:
    cb = dle.chaos_bridge(0, 0, 0, 0, chaos_amplitude=0.05, samples=50)
    print(f"  health = {cb['health']}")
    print(f"  duality_coherence = {cb['duality_coherence']}")
    print(f"  demon_beats_phi = {cb['demon_beats_phi']}")
    print(f"  cascade_residual = {cb['cascade_residual']}")
    print(f"  thought_conserved = {cb['thought_conserved']}")
    print(f"  phi_intact = {cb['phi_intact']}")
    if cb["duality_coherence"] < 0.9:
        warnings.append(f"CHAOS_LOW_COHERENCE: {cb['duality_coherence']}")
except Exception as e:
    errors.append(f"CHAOS_ERR: {e}")
    traceback.print_exc()

# ─── Phase 8: Integrity check ───
print("[PHASE 8] Full integrity check...")
try:
    ic = dle.full_integrity_check(force=True)
    print(f"  integrity keys: {sorted(ic.keys())[:10]}...")
    passed = ic.get("passed", ic.get("all_pass", "UNKNOWN"))
    score = ic.get("score", ic.get("integrity_score", "UNKNOWN"))
    checks_list = ic.get("checks", [])
    print(f"  passed = {passed}, score = {score}")
    if isinstance(checks_list, list):
        for chk in checks_list[:5]:
            if isinstance(chk, dict):
                name = chk.get("name", chk.get("check", "?"))
                ok = chk.get("pass", chk.get("passed", chk.get("ok", "?")))
                print(f"    {'✓' if ok else '✗'} {name}")
            elif isinstance(chk, str):
                print(f"    - {chk}")
except Exception as e:
    errors.append(f"INTEGRITY_ERR: {e}")
    traceback.print_exc()

# ─── Phase 9: Derive + Batch ───
print("[PHASE 9] Derive + Batch operations...")
try:
    d = dle.derive("GOD_CODE", mode="physics")
    print(f"  derive('GOD_CODE', physics) keys: {sorted(d.keys())[:6]}...")

    d2 = dle.derive_both("GOD_CODE")
    print(f"  derive_both('GOD_CODE') keys: {sorted(d2.keys())[:6]}...")

    names = dle.constant_names()
    print(f"  constant_names: {len(names)} constants, first 5: {names[:5]}")

    batch = dle.batch_collapse(names[:3])
    print(f"  batch_collapse(first 3) results: {len(batch.get('results', []))} entries")
except Exception as e:
    errors.append(f"DERIVE_ERR: {e}")
    traceback.print_exc()

# ─── Phase 10: v5 features ───
print("[PHASE 10] v5 features...")
try:
    vs = dle.v5_status()
    print(f"  v5_status keys: {sorted(vs.keys())[:8]}...")
    print(f"  version = {vs.get('version', 'N/A')}")
except Exception as e:
    warnings.append(f"V5_STATUS_ERR: {e}")
    print(f"  ⚠ v5_status failed: {e}")

try:
    cbs = dle.circuit_breaker_status()
    print(f"  circuit_breaker: state={cbs.get('state', 'UNKNOWN')}, failures={cbs.get('failures', 'N/A')}")
except Exception as e:
    warnings.append(f"CB_STATUS_ERR: {e}")

try:
    dle.record_coherence(0.95, source="debug_test")
    traj = dle.temporal_coherence_trajectory()
    print(f"  temporal_coherence_trajectory: {len(traj.get('history', []))} points")
except Exception as e:
    warnings.append(f"TEMPORAL_ERR: {e}")

# ─── Phase 11: Cross-layer & advanced ───
print("[PHASE 11] Cross-layer & advanced...")
try:
    cl = dle.cross_layer_coherence()
    print(f"  cross_layer_coherence keys: {sorted(cl.keys())[:6]}...")
    coh = cl.get("coherence", cl.get("overall_coherence", "N/A"))
    print(f"  coherence = {coh}")
except Exception as e:
    warnings.append(f"CROSS_LAYER_ERR: {e}")
    print(f"  ⚠ cross_layer_coherence: {e}")

try:
    sg = dle.sacred_geometry_analysis(GOD_CODE)
    print(f"  sacred_geometry_analysis(GOD_CODE): {sorted(sg.keys())[:5]}...")
except Exception as e:
    warnings.append(f"SACRED_GEO_ERR: {e}")

# ─── Phase 12: Metrics audit ───
print("[PHASE 12] Operation metrics...")
m = dle._metrics
total = m.get("total_operations", 0)
print(f"  total_operations: {total}")
for k, v in sorted(m.items()):
    if v > 0 and k != "total_operations":
        print(f"    {k}: {v}")

# ─── Phase 13: Three-engine features ───
print("[PHASE 13] Three-engine features...")
try:
    ta = dle.three_engine_thought_amplification()
    amp = ta.get("amplified_thought", ta.get("thought", "N/A"))
    engines = ta.get("engines_used", [])
    print(f"  thought_amplification: amplified={amp}, engines={engines}")
except Exception as e:
    warnings.append(f"THREE_ENGINE_THOUGHT_ERR: {e}")
    print(f"  ⚠ three_engine_thought_amplification: {e}")

try:
    syn = dle.three_engine_synthesis()
    score = syn.get("synthesis_score", syn.get("score", "N/A"))
    print(f"  three_engine_synthesis: score={score}")
except Exception as e:
    warnings.append(f"THREE_ENGINE_SYNTH_ERR: {e}")
    print(f"  ⚠ three_engine_synthesis: {e}")

# ─── Phase 14: Gate engine features ───
print("[PHASE 14] Gate engine features...")
try:
    gc = dle.gate_sacred_collapse(n_qubits=2, depth=2)
    gate_ok = gc.get("success", gc.get("collapsed", False))
    print(f"  gate_sacred_collapse(2q,2d): success={gate_ok}, keys={sorted(gc.keys())[:5]}...")
except Exception as e:
    warnings.append(f"GATE_COLLAPSE_ERR: {e}")
    print(f"  ⚠ gate_sacred_collapse: {e}")

try:
    gi = dle.gate_compile_integrity()
    compiled = gi.get("compiled", gi.get("success", False))
    print(f"  gate_compile_integrity: compiled={compiled}")
except Exception as e:
    warnings.append(f"GATE_INTEGRITY_ERR: {e}")
    print(f"  ⚠ gate_compile_integrity: {e}")

# ─── Phase 15: Resilient collapse ───
print("[PHASE 15] Resilient collapse...")
try:
    rc = dle.resilient_collapse("GOD_CODE")
    rc_ok = "error" not in rc
    print(f"  resilient_collapse('GOD_CODE'): ok={rc_ok}, retries={rc.get('retries', 0)}")
except Exception as e:
    warnings.append(f"RESILIENT_COLLAPSE_ERR: {e}")
    print(f"  ⚠ resilient_collapse: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print()
print("═" * 72)
print(f"  ERRORS:   {len(errors)}")
for e in errors:
    print(f"    ✗ {e}")
print(f"  WARNINGS: {len(warnings)}")
for w in warnings:
    print(f"    ⚠ {w}")
health = "HEALTHY" if len(errors) == 0 else "DEGRADED" if len(errors) < 3 else "CRITICAL"
color = "\033[92m" if health == "HEALTHY" else "\033[93m" if health == "DEGRADED" else "\033[91m"
print(f"  VERDICT:  {color}{health}\033[0m  ({len(errors)} errors, {len(warnings)} warnings)")
print("═" * 72)
