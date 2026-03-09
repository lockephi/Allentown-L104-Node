#!/usr/bin/env python3
"""Quick smoke test for search/precog ASI + Comprehension + Learning integration."""
import math
import sys

def test_asi_core():
    print("=== 1. ASI Core Search/Precog Integration ===")
    from l104_asi.core import ASICore
    core = ASICore()

    se = core._get_search_engine()
    pe = core._get_precognition_engine()
    hub = core._get_three_engine_search_hub()
    print(f"  Search Engine: {se is not None}")
    print(f"  Precognition Engine: {pe is not None}")
    print(f"  Three-Engine Hub: {hub is not None}")
    assert se is not None, "Search engine not loaded"
    assert pe is not None, "Precognition engine not loaded"
    assert hub is not None, "Three-engine hub not loaded"

    sp_score = core.search_precog_score()
    pa_score = core.precognition_accuracy_score()
    print(f"  search_precog_score: {sp_score:.4f}")
    print(f"  precognition_accuracy: {pa_score:.4f}")
    assert 0.0 <= sp_score <= 1.0
    assert 0.0 <= pa_score <= 1.0

    data = list(range(100))
    result = core.search(data, lambda x: x == 42, algorithm="grover")
    print(f"  search(42): found={result.get('found')}, value={result.get('value')}")
    assert result.get("found"), f"ASI search channel failed: {result}"

    series = [math.sin(i * 0.5) for i in range(20)]
    precog = core.precognize(series, horizon=3)
    print(f"  precognize: trend={precog.get('trend', '?')}")

    status = core.search_precog_status()
    print(f"  status keys: {sorted(status.keys())}")
    assert "search_engine" in status
    print("  OK\n")


def test_comprehension():
    print("=== 2. Language Comprehension Engine ===")
    from l104_asi.language_comprehension import LanguageComprehensionEngine
    from l104_asi.language_comprehension import _get_cached_search_engine, _get_cached_precognition_engine, _get_cached_three_engine_hub

    # Verify module-level loaders work
    se = _get_cached_search_engine()
    pe = _get_cached_precognition_engine()
    hub = _get_cached_three_engine_hub()
    print(f"  Cached Search Engine: {se is not None}")
    print(f"  Cached Precognition Engine: {pe is not None}")
    print(f"  Cached Three-Engine Hub: {hub is not None}")
    assert se is not None, "Search engine cache loader failed"
    assert pe is not None, "Precognition engine cache loader failed"
    assert hub is not None, "Three-engine hub cache loader failed"

    # Verify __init__ wires slots correctly (without full initialize — KB init is too slow)
    lce = LanguageComprehensionEngine()
    assert hasattr(lce, '_search_engine'), "Missing _search_engine slot"
    assert hasattr(lce, '_precognition_engine'), "Missing _precognition_engine slot"
    assert hasattr(lce, '_three_engine_hub'), "Missing _three_engine_hub slot"
    assert hasattr(lce, '_search_enhanced_queries'), "Missing _search_enhanced_queries slot"
    assert lce._search_enhanced_queries == 0
    print(f"  __init__ slots: OK (all 4 present)")

    # Verify source code structure: search wiring in initialize, comprehend, get_status
    import inspect
    src = inspect.getsource(lce.initialize)
    assert '_get_cached_search_engine' in src, "initialize() missing search engine wiring"
    assert '_get_cached_precognition_engine' in src, "initialize() missing precognition wiring"
    print(f"  initialize() wiring: OK")

    src_c = inspect.getsource(lce.comprehend)
    assert 'search_augmented' in src_c, "comprehend() missing search_augmented"
    assert 'precog_insight' in src_c, "comprehend() missing precognition_insight"
    print(f"  comprehend() new fields: OK")

    src_s = inspect.getsource(lce.get_status)
    assert 'v10_search_precog' in src_s, "get_status() missing v10_search_precog"
    assert '26_search_augmentation' in src_s, "get_status() missing layer 26"
    assert '27_precognition_insight' in src_s, "get_status() missing layer 27"
    assert '28_three_engine_search_hub' in src_s, "get_status() missing layer 28"
    print(f"  get_status() v10 sections: OK")

    src_t = inspect.getsource(lce.three_engine_comprehension_score)
    assert 'search_precog_score' in src_t or 'sp_components' in src_t, "three_engine_comprehension_score() missing search/precog"
    assert '_search_engine' in src_t, "three_engine_comprehension_score() missing search check"
    print(f"  three_engine_comprehension_score() search/precog: OK")
    print("  OK\n")


def test_learning():
    print("=== 3. Learning Intellect ===")
    from l104_server.learning.intellect import intellect

    print(f"  Search Engine wired: {intellect._search_engine is not None}")
    print(f"  Precognition Engine wired: {intellect._precognition_engine is not None}")
    print(f"  Three-Engine Hub wired: {intellect._three_engine_search_hub is not None}")

    # Test predict_future_state includes precog data
    future = intellect.predict_future_state(steps=5)
    print(f"  predict_future_state keys: {sorted(future.keys())}")
    has_precog_forecast = "precognition_forecast" in future
    has_three_engine = "three_engine_forecast" in future
    print(f"  precognition_forecast present: {has_precog_forecast}")
    print(f"  three_engine_forecast present: {has_three_engine}")

    # Test search_precog_status
    sp_status = intellect.search_precog_status()
    print(f"  search_precog_status: {sp_status}")
    assert "search_engine" in sp_status
    assert "precognition_engine" in sp_status
    print("  OK\n")


def main():
    passed = 0
    failed = 0
    for name, fn in [("ASI Core", test_asi_core), ("Comprehension", test_comprehension), ("Learning", test_learning)]:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"{'=' * 60}")
    print(f"  Integration Smoke Test: {passed}/{passed+failed} passed")
    if failed == 0:
        print("  ★ ALL INTEGRATION TESTS PASSED ★")
    else:
        print(f"  ⚠ {failed} test(s) failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
