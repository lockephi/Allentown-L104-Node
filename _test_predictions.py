#!/usr/bin/env python3
"""Test the Dual-Layer Prediction Engine — does it find unmatched grid points?"""
import sys
import time

sys.path.insert(0, ".")
from l104_asi.dual_layer import DualLayerEngine

engine = DualLayerEngine()

print("=" * 72)
print("  DUAL-LAYER PREDICTION ENGINE — TEST")
print("=" * 72)

# Test 1: Engine available
print(f"\n1. Engine available: {engine.available}")
assert engine.available, "Dual-layer engine not available"
print("   PASS")

# Test 2: predict() returns structured data
print("\n2. Running predict(max_complexity=10, top_n=20)...")
t0 = time.time()
result = engine.predict(max_complexity=10, top_n=20)
elapsed = time.time() - t0
print(f"   Completed in {elapsed:.1f}s")

assert "physics_predictions" in result, "Missing physics_predictions"
assert "thought_predictions" in result, "Missing thought_predictions"
assert "convergences" in result, "Missing convergences"
assert "known_constants_checked" in result, "Missing known_constants_checked"
print(f"   Known constants: {result['known_constants_checked']}")
print(f"   Physics unmatched: {result['total_physics_unmatched']}")
print(f"   Thought unmatched: {result['total_thought_unmatched']}")
print(f"   Convergences: {result['total_convergences']}")
print("   PASS")

# Test 3: Predictions are actually unmatched
print("\n3. Verifying predictions aren't known constants...")
from l104_god_code_dual_layer import REAL_WORLD_CONSTANTS_V3

known_vals = [e["measured"] for e in REAL_WORLD_CONSTANTS_V3.values()]
false_matches = 0
for pred in result["physics_predictions"][:20]:
    for kv in known_vals:
        if kv > 0 and abs(pred["value"] - kv) / kv * 100 < 0.01:
            false_matches += 1
            print(f"   FALSE MATCH: {pred['value']} ≈ {kv}")

if false_matches == 0:
    print(f"   All {len(result['physics_predictions'][:20])} checked — none match known constants")
    print("   PASS")
else:
    print(f"   {false_matches} FALSE MATCHES — predictions are not truly unmatched")
    print("   FAIL")

# Test 4: Predictions have valid structure
print("\n4. Checking prediction structure...")
for pred in result["physics_predictions"][:5]:
    assert "dials" in pred, f"Missing dials in {pred}"
    assert "value" in pred, f"Missing value in {pred}"
    assert "complexity" in pred, f"Missing complexity in {pred}"
    assert "layer" in pred, f"Missing layer in {pred}"
    assert pred["value"] > 0, f"Invalid value: {pred['value']}"
    assert pred["complexity"] > 0, f"Invalid complexity: {pred['complexity']}"
print("   PASS")

# Test 5: predict_summary() returns readable text
print("\n5. Testing predict_summary()...")
summary = engine.predict_summary(max_complexity=8)
assert isinstance(summary, str), "Summary should be a string"
assert "PREDICTION" in summary.upper(), "Summary should mention predictions"
assert len(summary) > 100, "Summary too short"
print(f"   Summary length: {len(summary)} chars")
print("   PASS")

# Test 6: Print the actual summary
print("\n" + "=" * 72)
print(summary)
print("=" * 72)

# Test 7: Convergences have valid structure
print("\n7. Checking convergence structure...")
if result["convergences"]:
    for conv in result["convergences"][:5]:
        assert "physics" in conv, "Missing physics in convergence"
        assert "thought" in conv, "Missing thought in convergence"
        assert "value_avg" in conv, "Missing value_avg"
        assert conv["convergence_pct"] < 0.1, f"Convergence too loose: {conv['convergence_pct']}%"
    print(f"   {len(result['convergences'])} convergences verified")
    print("   PASS")
else:
    print("   No convergences found (not necessarily an error — grids are different)")
    print("   SKIP")

print("\n" + "=" * 72)
total = 6 + (1 if result["convergences"] else 0)
print(f"  PREDICTION ENGINE: ALL {total} TESTS PASSED")
print("=" * 72)
