#!/usr/bin/env python3
"""Precog Synthesis Intelligence — ASI Integration Test Suite."""
import math

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    print(f"── {name} ──")
    try:
        fn()
        passed += 1
        print("  ✅ PASS")
    except Exception as e:
        failed += 1
        print(f"  ❌ FAIL: {e}")

print("═══ PRECOG SYNTHESIS INTELLIGENCE — ASI WIRING TEST ═══\n")

# Test 1
def t1():
    from l104_asi import asi_core
    synth = asi_core._get_precog_synthesis()
    assert synth is not None, "Failed to load precog synthesis"
    print(f"  Synthesis loaded: {synth is not None}")
test("ASI Core precog synthesis loader", t1)

# Test 2
def t2():
    from l104_asi import asi_core
    score = asi_core.precog_synthesis_intelligence_score()
    print(f"  Score: {score:.6f}")
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
test("precog_synthesis_intelligence_score", t2)

# Test 3
def t3():
    from l104_asi import asi_core
    series = [527.5 + 1.618 * math.sin(i * 0.3) for i in range(30)]
    result = asi_core.synthesize_precognition(series, horizon=5)
    assert "predictions" in result, f"Missing predictions, got: {sorted(result.keys())}"
    assert "synthesis_intelligence_score" in result, "Missing SIS"
    print(f"  SIS: {result['synthesis_intelligence_score']:.6f}")
    print(f"  Predictions: {len(result['predictions'])}")
test("synthesize_precognition direct channel", t3)

# Test 4
def t4():
    from l104_asi import asi_core
    status = asi_core.search_precog_status()
    assert status["version"] == "21.0.0", f"Wrong version: {status['version']}"
    assert "precog_synthesis" in status, "Missing precog_synthesis key"
    assert status["precog_synthesis"] is True
    assert "precog_synthesis_intelligence" in status["scores"]
    print(f"  Version: {status['version']}")
    print(f"  Synth score: {status['scores']['precog_synthesis_intelligence']}")
    print(f"  Synth runs: {status['metrics']['precog_synthesis_runs']}")
test("search_precog_status v21.0", t4)

# Test 5
def t5():
    from l104_server.learning.intellect import intellect
    li_status = intellect.search_precog_status()
    assert li_status["precog_synthesis"] is True
    assert "synthesis_augmented_predictions" in li_status
    print(f"  Synthesis wired: {li_status['precog_synthesis']}")
test("Learning Intellect wiring", t5)

print(f"\n═══ RESULTS: {passed}/{passed+failed} passed ({failed} failed) ═══")
