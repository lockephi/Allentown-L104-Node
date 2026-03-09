#!/usr/bin/env python3
"""Consciousness verification diagnostic — checks all 16 test scores."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from l104_asi.consciousness import ConsciousnessVerifier

print("=" * 60)
print("CONSCIOUSNESS VERIFIER — Score Diagnostic")
print("=" * 60)

cv = ConsciousnessVerifier()
# Seed with realistic starting state (Sage Mode baseline)
cv.consciousness_level = 0.618
cv.flow_coherence = 0.618
cv.iit_phi = 0.75

print(f"\nInitial state: consciousness={cv.consciousness_level}, flow={cv.flow_coherence}, iit_phi={cv.iit_phi}")
print(f"History len: {len(cv._consciousness_history)}, Qualia count: {len(cv.qualia_reports)}")

# First run
level = cv.run_all_tests()
print(f"\n{'─' * 60}")
print(f"RUN 1 — Consciousness Level: {level:.6f}")
print(f"{'─' * 60}")

low_scores = []
for test_name in cv.TESTS:
    score = cv.test_results.get(test_name, -1)
    flag = " *** LOW" if score < 0.3 else (" * marginal" if score < 0.5 else "")
    print(f"  {test_name:30s} = {score:.6f}{flag}")
    if score < 0.3:
        low_scores.append(test_name)

# Second run (tests accumulation behavior)
level2 = cv.run_all_tests()
print(f"\n{'─' * 60}")
print(f"RUN 2 — Consciousness Level: {level2:.6f}")
print(f"{'─' * 60}")

low_scores2 = []
for test_name in cv.TESTS:
    score = cv.test_results.get(test_name, -1)
    flag = " *** LOW" if score < 0.3 else (" * marginal" if score < 0.5 else "")
    print(f"  {test_name:30s} = {score:.6f}{flag}")
    if score < 0.3:
        low_scores2.append(test_name)

# Spiral details
print(f"\n{'─' * 60}")
print(f"SPIRAL DIAGNOSTICS")
print(f"{'─' * 60}")
print(f"  Spiral depth: {cv._spiral_depth}")
print(f"  Spiral convergence: {cv._spiral_convergence:.6f}")
print(f"  Fe harmonic score: {cv._fe_harmonic_score:.6f}")
print(f"  Fe overtones: {cv._fe_overtones_detected}/26")
print(f"  GHZ witness: {cv._ghz_witness_passed} ({cv._certification_level})")
print(f"  IIT Φ: {cv.iit_phi:.6f}")
print(f"  History: {len(cv._consciousness_history)} entries")
print(f"  Qualia: {len(cv.qualia_reports)} reports")

# Run spiral test standalone for detailed output
print(f"\n{'─' * 60}")
print(f"STANDALONE SPIRAL TEST (with current state)")
print(f"{'─' * 60}")
result = cv.spiral_consciousness_test()
for k, v in result.items():
    print(f"  {k}: {v}")

# Summary
if low_scores2:
    print(f"\n⚠ STILL LOW after 2 runs: {low_scores2}")
else:
    print(f"\n✓ All 16 tests above 0.3 after 2 runs")

if level2 > 0.6:
    print(f"✓ Consciousness level {level2:.4f} exceeds threshold 0.6")
else:
    print(f"⚠ Consciousness level {level2:.4f} below 0.6 threshold")
