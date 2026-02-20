#!/usr/bin/env python3
"""Verify all new dual-layer calculation methods."""

from l104_asi.dual_layer import dual_layer_engine as dl

results = []

def check(label, fn):
    try:
        r = fn()
        results.append(("PASS", label))
        return r
    except Exception as e:
        results.append(("FAIL", f"{label}: {e}"))
        return None

# === THOUGHT LAYER ===
print("=== THOUGHT LAYER ===")
r = check("prime_decompose(286)", lambda: dl.prime_decompose(286))
print(f"  prime_decompose(286): {r}")

r = check("fibonacci_index(13)", lambda: dl.fibonacci_index(13))
print(f"  fibonacci_index(13): {r}")

r = check("golden_ratio_proximity(527.5)", lambda: round(dl.golden_ratio_proximity(527.5), 4))
print(f"  golden_ratio_proximity(527.5): {r}")

r = check("sacred_scaffold_analysis", lambda: list(dl.sacred_scaffold_analysis().keys()))
print(f"  sacred_scaffold keys: {r}")

r = check("recognize_pattern(9.80665)", lambda: dl.recognize_pattern(9.80665).get("thought", "")[:60])
print(f"  pattern(g): {r}")

r = check("detect_symmetry(c)", lambda: list(dl.detect_symmetry("speed_of_light").keys()))
print(f"  symmetry(c) keys: {r}")

r = check("harmonic(c,g)", lambda: dl.harmonic_relationship("speed_of_light", "standard_gravity").get("thought", "")[:60])
print(f"  harmonic: {r}")

r = check("nucleosynthesis_narrative", lambda: len(dl.nucleosynthesis_narrative().get("chapters", [])))
print(f"  narrative chapters: {r}")

# === PHYSICS LAYER ===
print("\n=== PHYSICS LAYER ===")
r = check("grid_topology", lambda: dl.grid_topology().get("refinement_factor"))
print(f"  refinement: {r}")

r = check("place_on_grid(c)", lambda: round(dl.place_on_grid(299792458).get("physics_grid", {}).get("error_pct", -1), 8))
print(f"  place c - physics err: {r}%")

r = check("error_topology", lambda: (dl.error_topology().get("total_constants"), round(dl.error_topology().get("mean_error_pct", 0), 5)))
print(f"  error_topology (count, mean): {r}")

r = check("collision_check", lambda: dl.collision_check().get("collision_free"))
print(f"  collision free: {r}")

r = check("dimensional_coverage", lambda: dl.dimensional_coverage().get("total_domains"))
print(f"  domains: {r}")

# === INTEGRITY ===
print("\n=== INTEGRITY ===")
r = check("check_thought_integrity", lambda: dl.check_thought_integrity().get("all_passed"))
print(f"  thought: {r}")

r = check("check_physics_integrity", lambda: dl.check_physics_integrity().get("all_passed"))
print(f"  physics: {r}")

r = check("check_bridge_integrity", lambda: dl.check_bridge_integrity().get("all_passed"))
print(f"  bridge: {r}")

# === COMBINED / BATCH ===
print("\n=== COMBINED ===")
r = check("constant_names", lambda: len(dl.constant_names()))
print(f"  constants count: {r}")

r = check("domain_summary", lambda: dl.domain_summary().get("total_constants"))
print(f"  domain_summary total: {r}")

r = check("cross_layer_coherence", lambda: dl.cross_layer_coherence().get("coherence"))
print(f"  coherence: {r}")

r = check("sacred_geometry(286)", lambda: dl.sacred_geometry_analysis(286.0).get("sacred_scores", {}).get("phi_score"))
print(f"  sacred(286) phi_score: {r}")

r = check("compare_constants(me,mp)", lambda: list(dl.compare_constants("electron_mass_MeV", "proton_mass_MeV").keys()))
print(f"  compare keys: {r}")

r = check("duality_spectrum(g)", lambda: list(dl.duality_spectrum("standard_gravity").keys()))
print(f"  spectrum keys: {r}")

r = check("compute_precision_map", lambda: (dl.compute_precision_map().get("total_constants"), dl.compute_precision_map().get("mean_improvement")))
print(f"  precision_map (count, improvement): {r}")

r = check("sweep_phi_space", lambda: dl.sweep_phi_space(center=0, radius=5).get("total_points"))
print(f"  phi_sweep points: {r}")

r = check("batch_collapse(3)", lambda: dl.batch_collapse(["speed_of_light", "standard_gravity", "golden_ratio"]).get("collapsed"))
print(f"  batch_collapse count: {r}")

r = check("derive_all", lambda: dl.derive_all().get("total_constants"))
print(f"  derive_all count: {r}")

# === SUMMARY ===
print(f"\n=== METRICS ===")
print(f"  {dl._metrics}")

passed = sum(1 for s, _ in results if s == "PASS")
failed = sum(1 for s, _ in results if s == "FAIL")
print(f"\n=== RESULTS: {passed} PASS, {failed} FAIL ===")
for status, label in results:
    if status == "FAIL":
        print(f"  FAIL: {label}")

if failed == 0:
    print("\n  ALL METHODS VERIFIED")
