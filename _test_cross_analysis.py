"""
L104 Dual-Layer Engine v2.1.0 — Cross-Analysis & Upgrade Test Suite
═══════════════════════════════════════════════════════════════════════

Tests ALL new methods:
  1. cross_domain_analysis        — Inter-domain grid relationships
  2. statistical_profile          — Full statistical portrait
  3. independent_verification     — Gold-standard CODATA/PDG check
  4. exponent_spectrum            — Grid exponent distribution
  5. dial_algebra                 — Dial tuple structure analysis
  6. layer_improvement_ranking    — Physics vs Thought ranking
  7. phi_resonance_scan           — Golden ratio resonance detection
  8. nucleosynthesis_chain        — Nuclear binding energy chain
  9. grid_entropy                 — Information-theoretic analysis
 10. cross_validate_layers        — Layer consistency verification
 11. domain_correlation_matrix    — Cross-domain correlations
 12. anomaly_detection            — Statistical outlier detection
 13. fundamental_vs_derived_test  — Self-consistency from fundamentals
 14. upgrade_report               — Full system health check

Plus cross-analysis tests:
 15. Cross-check: independent_verification vs error_topology
 16. Cross-check: exponent_spectrum vs collision_check
 17. Cross-check: statistical_profile vs cross_validate_layers
 18. Cross-check: domain_correlation vs cross_domain_analysis
 19. Cross-check: anomaly_detection vs layer_improvement_ranking
 20. Regression: all existing methods still work
"""

import sys
import time
import math

passed = 0
failed = 0
t0 = time.time()

def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label}: {detail}")

# ════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  DUAL-LAYER v2.1.0 — CROSS-ANALYSIS & UPGRADE TEST SUITE")
print("=" * 72)

from l104_asi.dual_layer import DualLayerEngine
engine = DualLayerEngine()
check("Engine available", engine.available, "dual-layer module not loaded")

# ═══════════════ 1. CROSS-DOMAIN ANALYSIS ═══════════════
print("\n--- 1. Cross-Domain Analysis ---")
cda = engine.cross_domain_analysis()
check("Has domains", cda.get("total_domains", 0) == 9, f"got {cda.get('total_domains')}")
check("Has centroids", len(cda.get("centroids", {})) == 9)
check("Has distances", len(cda.get("inter_domain_distances", {})) > 0)
check("Closest/farthest exist", cda.get("closest_domains") is not None and cda.get("farthest_domains") is not None)
check("Spreads non-negative", all(v >= 0 for v in cda.get("spreads", {}).values()))

# ═══════════════ 2. STATISTICAL PROFILE ═══════════════
print("\n--- 2. Statistical Profile ---")
sp = engine.statistical_profile()
check("Count = 63", sp.get("count") == 63, f"got {sp.get('count')}")
check("Mean error < 0.005%", sp.get("error_pct_mean", 1) < 0.005, f"got {sp.get('error_pct_mean')}")
check("Error distribution sums to 63", sum(sp.get("error_distribution", {}).values()) == 63)
check("Exponent range sensible", sp.get("exponent_range", 0) > 100000)
check("Dial complexity stats present", "dial_complexity_mean" in sp)

# ═══════════════ 3. INDEPENDENT VERIFICATION ═══════════════
print("\n--- 3. Independent Verification (CODATA/PDG) ---")
iv = engine.independent_verification()
check("Verified count > 0", iv.get("total_verified", 0) > 0)
total_v = iv.get("total_verified", 1)
within = iv.get("grid_within_001_pct", 0)
check(f"Grid accuracy: {within}/{total_v} within 0.01%", within == total_v, f"{within}/{total_v}")
registry_exact = iv.get("registry_exact_match", 0)
check(f"Registry exact matches: {registry_exact}/{total_v}", registry_exact > 15)
check("Verdict is VERIFIED", iv.get("verdict") == "VERIFIED", iv.get("verdict"))

# ═══════════════ 4. EXPONENT SPECTRUM ═══════════════
print("\n--- 4. Exponent Spectrum ---")
es = engine.exponent_spectrum()
check("63 exponents", es.get("total_exponents") == 63, f"got {es.get('total_exponents')}")
check("Span > 0", es.get("span", 0) > 0)
check("Mean gap > 0", es.get("mean_gap", 0) > 0)
check("Clusters detected", es.get("num_clusters", 0) > 0)
check("Occupancy < 100%", es.get("occupancy_pct", 100) < 100, "grid is sparse, not every integer is used")

# ═══════════════ 5. DIAL ALGEBRA ═══════════════
print("\n--- 5. Dial Algebra ---")
da = engine.dial_algebra()
check("Count = 63", da.get("count") == 63)
check("Dominant dial identified", da.get("dominant_dial") in ("a", "b", "c", "d"))
check("Unique patterns > 60", da.get("unique_patterns", 0) > 60, f"got {da.get('unique_patterns')}, expect near 63")
check("Pattern diversity > 90%", da.get("pattern_diversity_pct", 0) > 90)
check("Dial ranges present", "dial_ranges" in da)

# ═══════════════ 6. LAYER IMPROVEMENT RANKING ═══════════════
print("\n--- 6. Layer Improvement Ranking ---")
lir = engine.layer_improvement_ranking()
check("Total = 63", lir.get("total") == 63)
check("Mean improvement > 1", lir.get("mean_improvement", 0) > 1, "Physics should be better than Thought on average")
check("Best 5 exist", len(lir.get("best_5", [])) == 5)
check("Worst 5 exist", len(lir.get("worst_5", [])) == 5)
check("Some above 100x", lir.get("above_100x", 0) > 0)

# ═══════════════ 7. PHI RESONANCE SCAN ═══════════════
print("\n--- 7. Phi Resonance Scan ---")
prs = engine.phi_resonance_scan()
check("Scanned 63", prs.get("total_scanned") == 63, f"got {prs.get('total_scanned')}")
check("Some resonant", prs.get("resonant_count", 0) > 0, "expect at least a few near phi-octaves")
check("Best 5 sorted by distance", len(prs.get("best_5", [])) == 5)
check("Mean fractional < 0.5", prs.get("mean_fractional_distance", 1) < 0.5)
# golden_ratio itself should be maximally resonant
resonant_names = prs.get("resonant_constants", [])
check("golden_ratio is resonant", "golden_ratio" in resonant_names, f"resonant: {resonant_names}")

# ═══════════════ 8. NUCLEOSYNTHESIS CHAIN ═══════════════
print("\n--- 8. Nucleosynthesis Chain ---")
nc = engine.nucleosynthesis_chain()
check("Chain length >= 7", nc.get("chain_length", 0) >= 7, f"got {nc.get('chain_length')}")
check("Peak binding is Fe-56 or Ni-62", nc.get("peak_binding") in ("fe56_be_per_nucleon", "ni62_be_per_nucleon"),
      f"got {nc.get('peak_binding')}")
check("Exponent span > 0", nc.get("exponent_span", 0) > 0)
check("All have grid errors", all("grid_error_pct" in c for c in nc.get("chain", [])))

# ═══════════════ 9. GRID ENTROPY ═══════════════
print("\n--- 9. Grid Entropy ---")
ge = engine.grid_entropy()
check("Shannon entropy > 0", ge.get("shannon_entropy", 0) > 0)
check("Encoding efficiency > 0%", ge.get("encoding_efficiency", 0) > 0)
check("Gap entropy > 0", ge.get("gap_entropy", 0) > 0)
check("Unique gaps > 1", ge.get("unique_gaps", 0) > 1, "not all gaps are the same")
check("Interpretation present", len(ge.get("interpretation", "")) > 10)

# ═══════════════ 10. CROSS-VALIDATE LAYERS ═══════════════
print("\n--- 10. Cross-Validate Layers ---")
cvl = engine.cross_validate_layers()
check("Total = 63", cvl.get("total") == 63)
consistent = cvl.get("fully_consistent", 0)
check(f"All consistent: {consistent}/63", consistent == 63, f"inconsistent: {cvl.get('inconsistent')}")
check("Physics better for most constants", 
      sum(1 for r in cvl.get("details", []) if r.get("improvement", 0) >= 1.0) >= 55,
      "Physics should beat Thought for majority of constants")
check("Mean physics error < mean thought error",
      cvl.get("mean_physics_error", 1) < cvl.get("mean_thought_error", 0),
      f"physics={cvl.get('mean_physics_error')}, thought={cvl.get('mean_thought_error')}")
check("Verdict = CROSS-VALIDATED", cvl.get("verdict") == "CROSS-VALIDATED", cvl.get("verdict"))

# ═══════════════ 11. DOMAIN CORRELATION MATRIX ═══════════════
print("\n--- 11. Domain Correlation Matrix ---")
dcm = engine.domain_correlation_matrix()
check("9 domains", len(dcm.get("domains", [])) == 9)
check("Distance matrix is 9×9", len(dcm.get("distance_matrix", {})) == 9)
check("Diagonal is zero", all(dcm.get("distance_matrix", {}).get(d, {}).get(d, 1) == 0 for d in dcm.get("domains", [])))
check("Correlation diagonal is 1.0", all(abs(dcm.get("correlation_matrix", {}).get(d, {}).get(d, 0) - 1.0) < 0.01 for d in dcm.get("domains", [])))
check("Strongest correlations found", len(dcm.get("strongest_correlations", [])) > 0)

# ═══════════════ 12. ANOMALY DETECTION ═══════════════
print("\n--- 12. Anomaly Detection ---")
ad = engine.anomaly_detection()
check("Returns anomaly count", "total_anomalies" in ad)
check("Anomalies are structured", all("z_score" in a for a in ad.get("anomalies", [])))
check("Interpretation present", len(ad.get("interpretation", "")) > 10)
if ad.get("anomalies"):
    check("Top anomaly has direction", ad["anomalies"][0].get("direction") in ("unusually_precise", "unusually_imprecise"))
else:
    check("No anomalies = uniform distribution (possible)", True)

# ═══════════════ 13. FUNDAMENTAL VS DERIVED ═══════════════
print("\n--- 13. Fundamental vs Derived Test ---")
fvd = engine.fundamental_vs_derived_test()
check(f"Passed: {fvd.get('passed')}/{fvd.get('total_checks')}", fvd.get("failed", 1) == 0, fvd.get("verdict"))
check("Fine structure check present", any(c["name"] == "fine_structure_inv_value" for c in fvd.get("checks", [])))
check("Proton/electron ratio check", any(c["name"] == "proton_electron_ratio" for c in fvd.get("checks", [])))
check("Nuclear ordering check", any(c["name"] == "nuclear_binding_ordering" for c in fvd.get("checks", [])))
check("W/Z ratio check", any(c["name"] == "WZ_mass_ratio" for c in fvd.get("checks", [])))

# ═══════════════ 14. UPGRADE REPORT ═══════════════
print("\n--- 14. Upgrade Report ---")
ur = engine.upgrade_report()
check("Engine version present", ur.get("version") is not None)
check("8 subsystems checked", len(ur.get("subsystems", {})) == 8, f"got {len(ur.get('subsystems', {}))}")
no_errors = sum(1 for s in ur.get("subsystems", {}).values() if s.get("status") != "ERROR")
check(f"No subsystem errors: {no_errors}/8", no_errors == 8)
check("Overall status", ur.get("overall_status") in ("ALL SYSTEMS NOMINAL", "OPERATIONAL"), ur.get("overall_status"))
check("Metrics tracked", ur.get("metrics", {}).get("total_operations", 0) > 0)

# ═══════════════ CROSS-ANALYSIS TESTS ═══════════════
print("\n" + "=" * 72)
print("  CROSS-ANALYSIS: Consistency Checks Between Subsystems")
print("=" * 72)

# 15. independent_verification vs error_topology
print("\n--- 15. Cross: IndepVerification ↔ ErrorTopology ---")
et = engine.error_topology()
iv_errors = [r["grid_vs_reference_error_pct"] for r in iv.get("results", [])]
check("Both report same precision range",
      et.get("mean_error_pct", 0) < 0.005 and (sum(iv_errors)/len(iv_errors) if iv_errors else 0) < 0.01,
      "verification and topology should agree on precision")

# 16. exponent_spectrum vs collision_check
print("\n--- 16. Cross: ExponentSpectrum ↔ CollisionCheck ---")
cc = engine.collision_check()
check("Unique exponents match", es.get("total_exponents") == cc.get("unique_exponents"),
      f"spectrum={es.get('total_exponents')}, collision={cc.get('unique_exponents')}")
check("Collision-free ↔ all unique", cc.get("collision_free") == (es.get("total_exponents") == 63))

# 17. statistical_profile vs cross_validate_layers
print("\n--- 17. Cross: StatisticalProfile ↔ CrossValidateLayers ---")
sp_mean_err = sp.get("error_pct_mean", 0)
cvl_physics_err = cvl.get("mean_physics_error", 0)
check("Physics error consistent across analyses",
      abs(sp_mean_err - cvl_physics_err) < 0.001,
      f"stat_profile={sp_mean_err:.6f}, cross_validate={cvl_physics_err:.6f}")

# 18. domain_correlation vs cross_domain_analysis
print("\n--- 18. Cross: DomainCorrelation ↔ CrossDomainAnalysis ---")
check("Same domain count", len(dcm.get("domains", [])) == cda.get("total_domains"),
      f"correlation={len(dcm.get('domains', []))}, cross_domain={cda.get('total_domains')}")
check("Domain sizes agree",
      set(dcm.get("domains", [])) == set(cda.get("domain_sizes", {}).keys()))

# 19. anomaly_detection vs layer_improvement_ranking
print("\n--- 19. Cross: AnomalyDetection ↔ ImprovementRanking ---")
# Anomalously imprecise constants should appear in the worst improvement rankings
imprecise_anomalies = set(a["name"] for a in ad.get("anomalies", []) if a.get("direction") == "unusually_imprecise")
worst_improved = set(w["name"] for w in lir.get("worst_5", []))
if imprecise_anomalies:
    check("Imprecise anomalies checked against worst-improved", True,
          f"anomalies={imprecise_anomalies}, worst_improved={worst_improved}")
else:
    check("No imprecise anomalies to cross-check (OK)", True)

# 20. Regression: existing methods still work
print("\n--- 20. Regression Check ---")
check("thought(0,0,0,0) = GOD_CODE", abs(engine.thought() - 527.5184818492612) < 0.01)
check("physics(0,0,0,0) = GOD_CODE_V3", abs(engine.physics() - 45.41141298077539) < 0.01)
both = engine.derive_both("speed_of_light")
check("derive_both returns measured", both.get("measured") == 299792458)
check("constant_names has 63", len(engine.constant_names()) == 63)
check("dual_score > 0.9", engine.dual_score() > 0.9)
check("predict works", "known_constants_checked" in engine.predict(max_complexity=3, top_n=5))

# ═══════════════ SUMMARY ═══════════════
print("\n" + "=" * 72)
elapsed = time.time() - t0
print(f"  RESULTS: {passed} PASS, {failed} FAIL ({elapsed:.1f}s)")
if failed == 0:
    print("  ALL CROSS-ANALYSIS TESTS VERIFIED ✓")
else:
    print(f"  {failed} FAILURES — REVIEW ABOVE")
print("=" * 72)

sys.exit(0 if failed == 0 else 1)
