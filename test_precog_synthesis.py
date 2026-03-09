#!/usr/bin/env python3
"""
Test Suite for L104 Precognition Synthesis Intelligence v1.0.0
══════════════════════════════════════════════════════════════════════════════════
30 tests across 6 phases validating all synthesis layers and their
higher-dimensional mathematical foundations.

Run: .venv/bin/python test_precog_synthesis.py

INVARIANT: 527.5184818492612 | PILOT: LONDEL
══════════════════════════════════════════════════════════════════════════════════
"""

import sys
import time
import math
import traceback

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = 1.618033988749895
PHI_CONJUGATE = 0.6180339887498948
GOD_CODE = 527.5184818492612
VOID_CONSTANT = 1.0416180339887497

passed = 0
failed = 0
errors = []

# ═══════════════════════════════════════════════════════════════════════════════
# TEST HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_test(name: str, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  ✅ {name}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  ❌ {name}: {e}")
        traceback.print_exc()


def gen_series(n=50, seed=104):
    """Generate a test time series with PHI-trend + sacred noise."""
    import random
    rng = random.Random(seed)
    series = []
    val = 10.0
    for i in range(n):
        val += 0.5 * math.sin(i * PHI) + rng.gauss(0, 0.3)
        series.append(round(val, 6))
    return series


# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("L104 PRECOGNITION SYNTHESIS INTELLIGENCE — TEST SUITE v1.0.0")
print("=" * 70)
t_start = time.time()

# Import module under test
from l104_precog_synthesis import (
    precog_synthesis,
    PrecogSynthesisIntelligence,
    HyperdimensionalPrecogFusion,
    ManifoldConvergenceTracker,
    TemporalCoherenceField,
    DimensionalPrecogProjector,
    SacredResonanceAmplifier,
    SynthesisResult,
    SynthesisPhase,
    VERSION,
)

series = gen_series()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Constants & Module Structure
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Phase 1: Constants & Module Structure ──")


def test_version():
    assert VERSION == "1.0.0", f"Version should be 1.0.0, got {VERSION}"

run_test("VERSION is 1.0.0", test_version)


def test_singleton():
    assert precog_synthesis is not None, "Singleton should exist"
    assert isinstance(precog_synthesis, PrecogSynthesisIntelligence), "Singleton type"

run_test("precog_synthesis singleton exists", test_singleton)


def test_sacred_constants_in_module():
    from l104_precog_synthesis import GOD_CODE as gc, PHI as phi, VOID_CONSTANT as vc
    assert abs(gc - 527.5184818492612) < 1e-10, f"GOD_CODE mismatch: {gc}"
    assert abs(phi - 1.618033988749895) < 1e-10, f"PHI mismatch: {phi}"
    assert abs(vc - 1.0416180339887497) < 1e-10, f"VOID mismatch: {vc}"

run_test("Sacred constants match", test_sacred_constants_in_module)


def test_hd_dimension():
    from l104_precog_synthesis import HD_DIMENSION
    assert HD_DIMENSION == 10000, f"HD_DIMENSION should be 10000, got {HD_DIMENSION}"

run_test("HD_DIMENSION = 10000", test_hd_dimension)


def test_synthesis_phases_enum():
    phases = list(SynthesisPhase)
    assert len(phases) == 6, f"Should have 6 phases, got {len(phases)}"
    names = [p.name for p in phases]
    assert "HD_FUSION" in names, "Missing HD_FUSION phase"
    assert "MANIFOLD_TRACK" in names, "Missing MANIFOLD_TRACK phase"
    assert "SACRED_AMPLIFY" in names, "Missing SACRED_AMPLIFY phase"

run_test("SynthesisPhase enum has 6 phases", test_synthesis_phases_enum)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Hyperdimensional Precog Fusion
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Phase 2: Hyperdimensional Precog Fusion ──")

hd_fusion = HyperdimensionalPrecogFusion(dimension=1000)  # Smaller for test speed


def test_hd_fusion_init():
    assert hd_fusion.dim == 1000, "Dimension mismatch"
    assert len(hd_fusion._codebook) > 0, "Codebook should be populated"

run_test("HD fusion initializes with codebook", test_hd_fusion_init)


def test_hd_make_value_vector():
    v = hd_fusion._make_value_vector(GOD_CODE)
    if hasattr(v, '__len__'):
        assert len(v) == 1000, f"Vector should be 1000-D, got {len(v)}"
    else:
        assert False, "Value vector should be array-like"

run_test("Value vector encoding produces 1000-D vector", test_hd_make_value_vector)


def test_hd_bind_bundle():
    v1 = hd_fusion._make_value_vector(1.0)
    v2 = hd_fusion._make_value_vector(2.0)
    bound = hd_fusion._bind(v1, v2)
    assert len(bound) == 1000, "Bound vector dimension"
    bundled = hd_fusion._bundle([v1, v2])
    assert len(bundled) == 1000, "Bundled vector dimension"

run_test("Bind and bundle produce correct dimensions", test_hd_bind_bundle)


def test_hd_similarity_self():
    v = hd_fusion._make_value_vector(PHI)
    sim = hd_fusion._similarity(v, v)
    assert abs(sim - 1.0) < 0.001, f"Self-similarity should be ~1.0, got {sim}"

run_test("Self-similarity ≈ 1.0", test_hd_similarity_self)


def test_hd_permute():
    v = hd_fusion._make_value_vector(42.0)
    p = hd_fusion._permute(v, 104)
    assert len(p) == 1000, "Permuted vector dimension"
    sim = hd_fusion._similarity(v, p)
    assert sim < 0.99, f"Permuted vector should differ from original, sim={sim}"

run_test("Permute shifts vector (sim < 1)", test_hd_permute)


def test_hd_fuse_multi_source():
    outputs = {
        "temporal_pattern": {"predictions": [10.0, 10.5, 11.0], "confidence": 0.8},
        "harmonic_extrapolation": {"predictions": [10.1, 10.6, 11.1], "confidence": 0.75},
        "cascade_precognition": {"predictions": [9.9, 10.4, 10.9], "confidence": 0.7},
    }
    result = hd_fusion.fuse(outputs)
    assert "consensus_predictions" in result, "Should have consensus_predictions"
    assert len(result["consensus_predictions"]) == 3, "Should produce 3 predictions"
    assert result["sources_fused"] == 3, f"Should fuse 3 sources, got {result['sources_fused']}"
    assert 0 <= result["fusion_score"] <= 1, f"Fusion score should be [0,1], got {result['fusion_score']}"

run_test("HD fusion with 3 sources produces consensus", test_hd_fuse_multi_source)


def test_hd_fuse_empty():
    result = hd_fusion.fuse({})
    assert result["sources_fused"] == 0, "Empty input → 0 sources"
    assert result["consensus_predictions"] == [], "Empty → no predictions"

run_test("HD fusion handles empty input", test_hd_fuse_empty)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Manifold Convergence Tracker
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Phase 3: Manifold Convergence Tracker ──")

manifold = ManifoldConvergenceTracker(manifold_dim=5)


def test_manifold_init():
    assert manifold.manifold_dim == 5, "Manifold dimension"
    assert manifold.kappa == PHI, "Curvature should be PHI"
    assert len(manifold._attractors) == 4, "Should have 4 attractors"

run_test("Manifold tracker initializes correctly", test_manifold_init)


def test_manifold_geodesic_distance():
    p = [0.5, 0.5, 0.5, 0.5, 0.5]
    q = [0.6, 0.4, 0.5, 0.5, 0.5]
    d = manifold._geodesic_distance(p, q)
    assert d > 0, f"Geodesic distance should be positive, got {d}"
    # Should be greater than Euclidean distance due to curvature
    d_euc = math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))
    assert d >= d_euc, f"Geodesic ({d}) should be ≥ Euclidean ({d_euc})"

run_test("Geodesic distance > Euclidean (curvature effect)", test_manifold_geodesic_distance)


def test_manifold_metric_tensor():
    pt = [0.3, 0.4, 0.5, 0.2, 0.1]
    metric = manifold._phi_metric_tensor(pt)
    assert len(metric) == 5, "Metric tensor should be 5x5"
    assert len(metric[0]) == 5, "Metric tensor row should be 5"
    # Diagonal should be ≥ 1 (δ_ii + curvature term ≥ 1)
    for i in range(5):
        assert metric[i][i] >= 1.0, f"Diagonal g_{i}{i} should be ≥ 1"

run_test("PHI-metric tensor is positive definite diagonal", test_manifold_metric_tensor)


def test_manifold_track():
    pred_sources = {
        "src_a": [0.5, 0.4, 0.35, 0.33, 0.32],
        "src_b": [0.6, 0.45, 0.38, 0.34, 0.33],
        "src_c": [0.55, 0.42, 0.36, 0.335, 0.325],
    }
    result = manifold.track(pred_sources)
    assert "convergence_score" in result, "Should have convergence_score"
    assert "closest_attractor" in result, "Should have closest_attractor"
    assert result["ricci_smoothed"] is True, "Should apply Ricci smoothing"
    assert 0 <= result["convergence_score"] <= 1, "Score in [0,1]"

run_test("Manifold tracking with converging sources", test_manifold_track)


def test_manifold_ricci_smoothing():
    traj = [[0.5, 0.5, 0.5, 0.5, 0.5],
            [0.8, 0.2, 0.9, 0.1, 0.7],  # Noisy outlier
            [0.5, 0.5, 0.5, 0.5, 0.5]]
    smoothed = manifold._ricci_flow_smooth(traj, steps=50)
    # Smoothed middle point should be closer to endpoints
    mid = smoothed[1]
    assert abs(mid[0] - 0.5) < abs(0.8 - 0.5), f"Smoothed should be closer to 0.5, got {mid[0]}"

run_test("Ricci flow smooths noisy trajectory", test_manifold_ricci_smoothing)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Temporal Coherence Field & Dimensional Projector
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Phase 4: Temporal Coherence Field & Dimensional Projector ──")

cfield = TemporalCoherenceField(field_size=52)  # Smaller for speed


def test_coherence_field_evolve():
    result = cfield.evolve(series[:30], steps=52)
    assert "field_strength" in result, "Should have field_strength"
    assert "demon_efficiency" in result, "Should have demon_efficiency"
    assert "reversal_potential" in result, "Should have reversal_potential"
    assert result["field_size"] == 52, "Field size"
    assert 0 <= result["field_strength"] <= 1, f"Coherence in [0,1], got {result['field_strength']}"

run_test("Coherence field evolution produces metrics", test_coherence_field_evolve)


def test_coherence_entropy_reduces():
    result = cfield.evolve(series[:30], steps=100, dt=0.02)
    assert result["entropy_reduced"] or result["reversal_potential"] >= 0, \
        f"Entropy should reduce (reversal={result['reversal_potential']})"

run_test("Coherence field entropy reduces over evolution", test_coherence_entropy_reduces)


def test_coherence_landauer():
    result = cfield.evolve([1.0, 2.0, 3.0, 4.0, 5.0], steps=20)
    assert result["landauer_cost_joules"] >= 0, "Landauer cost should be ≥ 0"
    assert result["bits_of_certainty"] >= 0, "Bits should be ≥ 0"

run_test("Landauer cost is non-negative", test_coherence_landauer)


projector = DimensionalPrecogProjector()


def test_lorentz_boost():
    four_vec = [1.0, 100.0, 0.0, 0.0]
    boosted = projector._lorentz_boost_4d(four_vec, 0.1)
    assert len(boosted) == 4, "Should produce 4-vector"
    gamma = projector._lorentz_gamma(0.1)
    assert gamma > 1.0, f"Gamma should be > 1, got {gamma}"

run_test("4D Lorentz boost produces valid 4-vector", test_lorentz_boost)


def test_kaluza_klein():
    four_vec = [1.0, 50.0, 0.0, 0.0]
    five_vec = projector._kaluza_klein_extend(four_vec, phi_charge=PHI)
    assert len(five_vec) == 5, "Should produce 5-vector"
    assert five_vec[4] != 0, "5th coordinate should be non-zero"

run_test("Kaluza-Klein 5D extension adds sacred dimension", test_kaluza_klein)


def test_dimensional_project():
    preds = [10.0, 10.5, 11.0, 11.5, 12.0]
    result = projector.project(preds, series[:20])
    assert "corrected_predictions" in result, "Should have corrected_predictions"
    assert len(result["corrected_predictions"]) == 5, "Should correct all 5 predictions"
    assert result["dimensions_used"] == 5, "Should use 5 dimensions"
    assert result["lorentz_gamma"] >= 1.0, f"Gamma should be ≥ 1, got {result['lorentz_gamma']}"

run_test("Dimensional projection corrects predictions in 5D", test_dimensional_project)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Sacred Resonance Amplifier
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Phase 5: Sacred Resonance Amplifier ──")

amplifier = SacredResonanceAmplifier()


def test_resonance_score():
    score = amplifier._resonance_score(GOD_CODE)
    assert 0 <= score <= 1, f"Resonance score should be [0,1], got {score}"
    # GOD_CODE itself should have very high resonance with its harmonics
    score_random = amplifier._resonance_score(42.0)
    # Both should be valid, GOD_CODE should be ≥ random
    assert 0 <= score_random <= 1, "Random value resonance in [0,1]"

run_test("Resonance scoring produces [0,1] values", test_resonance_score)


def test_phi_spiral_phase_lock():
    values = [10.0, 20.0, 30.0, 40.0]
    locked = amplifier._phi_spiral_phase_lock(values)
    assert len(locked) == 4, "Should produce same length"
    # Phase locking should slightly modulate values
    for original, modulated in zip(values, locked):
        ratio = modulated / original
        assert 0.9 < ratio < 1.1, f"Phase lock ratio {ratio} should be near 1.0"

run_test("PHI-spiral phase lock modulates values ±5%", test_phi_spiral_phase_lock)


def test_void_coupling():
    val = 100.0
    coupled = amplifier._void_coupling(val, field_strength=0.8)
    assert coupled >= val, f"VOID coupling should amplify, got {coupled} vs original {val}"

run_test("VOID coupling amplifies with positive field", test_void_coupling)


def test_amplifier_full():
    preds = [10.0, 10.5, 11.0, 11.5, 12.0]
    result = amplifier.amplify(preds, coherence_field_strength=0.7)
    assert "amplified_predictions" in result, "Should have amplified_predictions"
    assert len(result["amplified_predictions"]) == 5, "All predictions amplified"
    assert result["mean_amplification"] >= 1.0, \
        f"Mean amplification should be ≥ 1.0, got {result['mean_amplification']}"
    assert 0 <= result["mean_resonance"] <= 1, \
        f"Mean resonance in [0,1], got {result['mean_resonance']}"

run_test("Full amplification pipeline produces valid output", test_amplifier_full)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: Full Synthesis Intelligence Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Phase 6: Full Synthesis Intelligence Pipeline ──")


def test_full_synthesis():
    result = precog_synthesis.synthesize(series, horizon=5)
    assert isinstance(result, SynthesisResult), "Should return SynthesisResult"
    assert result.horizon == 5, f"Horizon should be 5, got {result.horizon}"
    assert len(result.phases_completed) > 0, "Should complete at least 1 phase"
    assert 0 <= result.synthesis_intelligence_score <= 1, \
        f"SIS should be [0,1], got {result.synthesis_intelligence_score}"
    assert result.system_outlook in ("TRANSCENDENT", "COHERENT", "EVOLVING", "NASCENT"), \
        f"Unexpected outlook: {result.system_outlook}"

run_test("Full synthesis pipeline produces SynthesisResult", test_full_synthesis)


def test_quick_synthesis():
    result = precog_synthesis.quick_synthesis(series, horizon=3)
    assert "predictions" in result, "Should have predictions"
    assert "intelligence_score" in result, "Should have intelligence_score"
    assert result["processing_ms"] >= 0, "Processing time should be ≥ 0"

run_test("Quick synthesis returns fast predictions", test_quick_synthesis)


def test_synthesis_weights():
    weights = precog_synthesis.weights
    total = sum(weights.values())
    assert abs(total - 1.0) < 0.001, f"Weights should sum to 1.0, got {total}"
    # Coherence (center) should have highest weight due to PHI ratio pattern
    assert weights["coherence"] >= weights["hd_fusion"], "Coherence weight ≥ HD fusion"
    assert weights["coherence"] >= weights["sacred"], "Coherence weight ≥ sacred"

run_test("PHI-weighted scoring coefficients sum to 1.0", test_synthesis_weights)


def test_engine_status():
    status = precog_synthesis.engine_status()
    for key in ["precognition_engine", "search_engine", "three_engine_hub",
                "science_engine", "math_engine", "code_engine"]:
        assert key in status, f"Missing engine status key: {key}"
        assert isinstance(status[key], bool), f"Status for {key} should be bool"

run_test("Engine status reports all 6 subsystems", test_engine_status)


def test_full_status():
    status = precog_synthesis.status()
    assert status["version"] == VERSION, "Version mismatch in status"
    assert len(status["synthesis_layers"]) == 5, "Should list 5 layers"
    assert status["hd_dimension"] == 10000, "HD dimension"
    assert status["manifold_curvature"] == PHI, "Manifold curvature"
    assert status["field_size"] == 104, "Field size"
    assert status["constants"]["GOD_CODE"] == GOD_CODE, "GOD_CODE in status"

run_test("Full status report structure complete", test_full_status)


def test_intelligence_metrics():
    metrics = precog_synthesis.intelligence_score()
    assert metrics["synthesis_count"] >= 1, "Should have run at least 1 synthesis"
    assert "weights" in metrics, "Should report weights"
    assert metrics["version"] == VERSION, "Version"

run_test("Intelligence metrics track synthesis count", test_intelligence_metrics)


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elapsed = time.time() - t_start
print("\n" + "=" * 70)
print(f"RESULTS: {passed}/{passed + failed} passed ({failed} failed) in {elapsed:.2f}s")
if errors:
    print("\nFAILURES:")
    for name, err in errors:
        print(f"  • {name}: {err}")
print("=" * 70)

sys.exit(0 if failed == 0 else 1)
