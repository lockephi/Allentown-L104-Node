#!/usr/bin/env python3
"""Functional validation for Round 3 improvements across 5 packages."""

import sys
import time
import traceback
import numpy as np

PASS = 0
FAIL = 0

def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL += 1
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
# 1. KMeans silhouette bug fix — uses actual training data now
# ═══════════════════════════════════════════════════════════════════════
def test_kmeans_silhouette():
    from l104_ml_engine.clustering import L104KMeans
    rng = np.random.default_rng(42)
    # Well-separated clusters for clear silhouette > 0
    X = np.vstack([rng.normal(0, 0.3, (30, 2)),
                   rng.normal(5, 0.3, (30, 2)),
                   rng.normal(10, 0.3, (30, 2))])
    km = L104KMeans(n_clusters=3, init='k-means++')
    km.fit(X)
    coh = km.cluster_coherence()
    assert isinstance(coh, float), f"Expected float, got {type(coh)}"
    assert 0.5 < coh <= 1.0, f"Expected high coherence for well-separated data, got {coh}"
    # Verify _X_scaled is stored
    assert hasattr(km, '_X_scaled'), "fit() must store _X_scaled"
    assert km._X_scaled.shape[0] == 90, f"Expected 90 rows, got {km._X_scaled.shape[0]}"

test("KMeans silhouette uses training data", test_kmeans_silhouette)


# ═══════════════════════════════════════════════════════════════════════
# 2. Grover PHI-damping — selective non-target damping
# ═══════════════════════════════════════════════════════════════════════
def test_grover_damping():
    from l104_search.search_algorithms import QuantumGroverSearch
    gs = QuantumGroverSearch()
    items = list(range(16))
    target_val = 7
    result = gs.search(items, oracle=lambda x: x == target_val)
    assert result.found, "Grover should find target"
    top = result.top_match()
    assert top is not None, "Should have a top match"
    assert top["item"] == target_val, f"Top match should be {target_val}, got {top['item']}"
    # With selective damping, target probability should be significantly above 1/N
    assert top["probability"] > 1.0 / len(items), (
        f"Target prob {top['probability']:.4f} should exceed uniform {1/len(items):.4f}"
    )

test("Grover selective PHI-damping", test_grover_damping)


# ═══════════════════════════════════════════════════════════════════════
# 3. Resonance cache thread-safety — now LRUCache, not plain dict
# ═══════════════════════════════════════════════════════════════════════
def test_resonance_cache():
    from l104_intellect.cache import LRUCache, _RESONANCE_CACHE
    assert isinstance(_RESONANCE_CACHE, LRUCache), (
        f"_RESONANCE_CACHE should be LRUCache, got {type(_RESONANCE_CACHE)}"
    )
    # Test get/set round-trip
    _RESONANCE_CACHE.set("resonance", 42.0)
    val = _RESONANCE_CACHE.get("resonance")
    assert val == 42.0, f"Expected 42.0, got {val}"
    # Verify TTL = 0.5s
    assert _RESONANCE_CACHE._ttl == 0.5, f"Expected TTL=0.5, got {_RESONANCE_CACHE._ttl}"
    _RESONANCE_CACHE.clear()

test("Resonance cache thread-safe LRUCache", test_resonance_cache)


# ═══════════════════════════════════════════════════════════════════════
# 4. SVM PSD kernel correction — harmonic kernel no longer crashes
# ═══════════════════════════════════════════════════════════════════════
def test_svm_psd_correction():
    from l104_ml_engine.svm import L104SVM
    rng = np.random.default_rng(104)
    X = rng.normal(0, 1, (30, 4))
    y = np.array([0]*15 + [1]*15)
    # Use harmonic kernel (most likely to be non-PSD)
    svm = L104SVM(mode='classify', kernel='harmonic_kernel')
    svm.fit(X, y)
    preds = svm.predict(X)
    assert preds.shape == (30,), f"Predictions shape mismatch: {preds.shape}"
    acc = np.mean(preds == y)
    assert acc >= 0.4, f"Accuracy too low: {acc}"

test("SVM PSD kernel correction (harmonic)", test_svm_psd_correction)


# ═══════════════════════════════════════════════════════════════════════
# 5. VQC early stopping — should terminate before max_iterations
# ═══════════════════════════════════════════════════════════════════════
def test_vqc_early_stopping():
    from l104_ml_engine.quantum_classifiers import VariationalQuantumClassifier
    rng = np.random.default_rng(42)
    # Simple 2-class problem, 2 features
    X = rng.normal(0, 1, (20, 2))
    y = (X[:, 0] > 0).astype(int)
    vqc = VariationalQuantumClassifier(n_qubits=2, depth=1, n_classes=2)
    result = vqc.fit(X, y, max_iterations=100, patience=5)
    assert 'early_stopped' in result, "Result must include early_stopped flag"
    assert 'max_iterations' in result, "Result must include max_iterations"
    # If early stopped, actual iterations < max
    if result['early_stopped']:
        assert result['n_iterations'] < result['max_iterations'], (
            f"early_stopped=True but iters={result['n_iterations']} >= max={result['max_iterations']}"
        )

test("VQC early stopping", test_vqc_early_stopping)


# ═══════════════════════════════════════════════════════════════════════
# 6. Search constants — GOD_CODE canonical value
# ═══════════════════════════════════════════════════════════════════════
def test_search_constants():
    from l104_search import search_algorithms as sa
    assert sa.GOD_CODE == 527.5184818492612, f"GOD_CODE drift: {sa.GOD_CODE}"
    assert sa.VOID_CONSTANT == 1.0416180339887497, f"VOID_CONSTANT drift: {sa.VOID_CONSTANT}"

test("Search constants canonical", test_search_constants)


# ═══════════════════════════════════════════════════════════════════════
# 7. SearchResult serialization — to_dict / from_dict / to_json
# ═══════════════════════════════════════════════════════════════════════
def test_search_result_serialization():
    import json
    from l104_search.search_algorithms import SearchResult
    sr = SearchResult(
        found=True, query="test", matches=[{"item": 1}],
        score=0.95, sacred_alignment=0.88,
        strategy="quantum_grover", iterations=5,
        elapsed_ms=1.23, entropy_delta=-0.5,
        coherence=0.9, dimensions_searched=16,
    )
    d = sr.to_dict()
    assert d["found"] is True
    assert d["score"] == 0.95
    assert "phi_quality" in d
    # JSON round-trip
    j = sr.to_json()
    d2 = json.loads(j)
    assert d2["strategy"] == "quantum_grover"
    # from_dict round-trip
    sr2 = SearchResult.from_dict(d)
    assert sr2.found == sr.found
    assert sr2.score == sr.score
    assert sr2.strategy == sr.strategy

test("SearchResult to_dict/from_dict/to_json", test_search_result_serialization)


# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*60}")
sys.exit(1 if FAIL > 0 else 0)
