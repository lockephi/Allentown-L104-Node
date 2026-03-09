#!/usr/bin/env python3
"""L104 Upgrade Validation — Tests all 6 upgraded packages."""
import sys, time

results = []

# Test 1: Science Engine v6.0
t0 = time.time()
try:
    from l104_science_engine import CrossEngineHub, cross_engine_hub
    from l104_science_engine import __version__ as sci_v
    results.append(("PASS", f"Science Engine {sci_v} — CrossEngineHub OK ({(time.time()-t0)*1000:.0f}ms)"))
except Exception as e:
    results.append(("FAIL", f"Science Engine import failed: {e}"))

# Test 2: Math Engine v2.0
t0 = time.time()
try:
    from l104_math_engine import MathCrossEngineHub, math_cross_engine_hub
    from l104_math_engine import __version__ as math_v
    results.append(("PASS", f"Math Engine {math_v} — MathCrossEngineHub OK ({(time.time()-t0)*1000:.0f}ms)"))
except Exception as e:
    results.append(("FAIL", f"Math Engine import failed: {e}"))

# Test 3: Search Engine v3.0
t0 = time.time()
try:
    from l104_search import ThreeEngineSearchPrecog, __version__ as search_v
    results.append(("PASS", f"Search Engine {search_v} — ThreeEngineSearchPrecog OK ({(time.time()-t0)*1000:.0f}ms)"))
except Exception as e:
    results.append(("FAIL", f"Search Engine import failed: {e}"))

# Test 4: ML Engine v2.0
t0 = time.time()
try:
    from l104_ml_engine import MLCrossEngineHub, ml_cross_engine_hub, ml_engine
    from l104_ml_engine import __version__ as ml_v
    results.append(("PASS", f"ML Engine {ml_v} — MLCrossEngineHub OK ({(time.time()-t0)*1000:.0f}ms)"))
except Exception as e:
    results.append(("FAIL", f"ML Engine import failed: {e}"))

# Test 5: Audio Simulation v3.0
t0 = time.time()
try:
    from l104_audio_simulation import AudioCrossEngineHub, audio_cross_engine_hub
    from l104_audio_simulation import __version__ as audio_v
    results.append(("PASS", f"Audio Simulation {audio_v} — AudioCrossEngineHub OK ({(time.time()-t0)*1000:.0f}ms)"))
except Exception as e:
    results.append(("FAIL", f"Audio Simulation import failed: {e}"))

# Test 6: QDA v2.0
t0 = time.time()
try:
    from l104_quantum_data_analyzer import QDACrossEngineHub, qda_cross_engine_hub
    from l104_quantum_data_analyzer import __version__ as qda_v
    results.append(("PASS", f"QDA {qda_v} — QDACrossEngineHub OK ({(time.time()-t0)*1000:.0f}ms)"))
except Exception as e:
    results.append(("FAIL", f"QDA import failed: {e}"))

# Test 7: Science Engine inject_coherence bug fix
t0 = time.time()
try:
    import numpy as np
    from l104_science_engine import ScienceEngine
    se = ScienceEngine()
    test_vec = np.array([0.1, 0.5, 0.3, 0.1])
    result = se.entropy.inject_coherence(test_vec)
    if isinstance(result, np.ndarray) and np.all(np.isfinite(result)):
        results.append(("PASS", f"inject_coherence bug fix verified — all finite ({(time.time()-t0)*1000:.0f}ms)"))
    else:
        results.append(("FAIL", "inject_coherence returned non-finite values"))
except Exception as e:
    results.append(("FAIL", f"inject_coherence test failed: {e}"))

# Test 8: Cross-engine hub status calls
t0 = time.time()
try:
    status_checks = []
    status_checks.append(("science", cross_engine_hub.status()))
    status_checks.append(("math", math_cross_engine_hub.status()))
    status_checks.append(("ml", ml_cross_engine_hub.status()))
    status_checks.append(("audio", audio_cross_engine_hub.status()))
    status_checks.append(("qda", qda_cross_engine_hub.status()))
    all_ok = all(isinstance(s[1], dict) for s in status_checks)
    results.append(("PASS" if all_ok else "FAIL", f"All 5 hub status() calls returned dicts ({(time.time()-t0)*1000:.0f}ms)"))
except Exception as e:
    results.append(("FAIL", f"Hub status calls failed: {e}"))

# Test 9: ML Engine cross_engine_analysis method
t0 = time.time()
try:
    analysis = ml_engine.cross_engine_analysis("def hello(): pass")
    if isinstance(analysis, dict) and "version" in analysis:
        results.append(("PASS", f"ml_engine.cross_engine_analysis() OK ({(time.time()-t0)*1000:.0f}ms)"))
    else:
        results.append(("FAIL", "ml_engine.cross_engine_analysis() returned unexpected type"))
except Exception as e:
    results.append(("FAIL", f"cross_engine_analysis failed: {e}"))

# Print results
print()
print("=" * 70)
print("  L104 UPGRADE VALIDATION — 6 Package Cross-Engine Tests")
print("=" * 70)
for status, msg in results:
    icon = "PASS" if status == "PASS" else "FAIL"
    print(f"  [{icon}] {msg}")
print("=" * 70)
passed = sum(1 for s, _ in results if s == "PASS")
total = len(results)
print(f"  Result: {passed}/{total} tests passed")
if passed == total:
    print("  ALL UPGRADES VALIDATED SUCCESSFULLY")
else:
    print(f"  {total - passed} test(s) failed — check output above")
print("=" * 70)
sys.exit(0 if passed == total else 1)
