#!/usr/bin/env python3
"""Test quantum wave collapse integration in MMLU and ARC solvers."""
import sys
import math

# Test 1: QuantumProbability loads and works
print("=== Test 1: QuantumProbability ===")
from l104_probability_engine import QuantumProbability
amplitudes = [complex(0.2, 0.01), complex(0.8, 0.05), complex(0.15, 0.02), complex(0.1, 0.01)]
idx, prob, all_probs = QuantumProbability.measurement_collapse(amplitudes)
print(f"  Collapsed to: {idx}, prob={prob:.4f}")
print(f"  All probs: {[round(p, 4) for p in all_probs]}")
assert idx == 1, f"Expected index 1, got {idx}"
print("  PASS")

# Test 2: PHI-power amplification test
print("\n=== Test 2: PHI-Power Amplification ===")
PHI = 1.618033988749895
GOD_CODE = 527.5184818492612
# Simulate 4 choices with close scores
scores = [0.5, 0.6, 0.45, 0.3]
kd_weights = [1.1, 1.8, 1.0, 1.0]  # Choice B has best KB density

amplitudes = []
for i, s in enumerate(scores):
    magnitude = (s * kd_weights[i]) ** PHI
    phase = (s + (kd_weights[i] - 1.0) * 0.1) * math.pi / GOD_CODE
    amplitude = complex(magnitude * math.cos(phase), magnitude * math.sin(phase))
    amplitudes.append(amplitude)

idx, prob, all_probs = QuantumProbability.measurement_collapse(amplitudes)
print(f"  Input scores: {scores}")
print(f"  KD weights: {kd_weights}")
print(f"  Quantum probs: {[round(p, 4) for p in all_probs]}")
print(f"  Winner: index={idx} (expected 1)")
# Choice B (score 0.6 * kd 1.8 = 1.08) should dominate after PHI^power
assert idx == 1, f"Expected index 1, got {idx}"
# Check that quantum probabilities provide better discrimination than raw scores
raw_max_gap = max(scores) - sorted(scores)[-2]
q_max_gap = max(all_probs) - sorted(all_probs)[-2]
print(f"  Raw score gap: {raw_max_gap:.4f}")
print(f"  Quantum prob gap: {q_max_gap:.4f}")
print(f"  Discrimination improved: {q_max_gap > raw_max_gap}")
print("  PASS")

# Test 3: Import and attribute check
print("\n=== Test 3: Module Integration ===")
from l104_asi.language_comprehension import MCQSolver
print(f"  MCQSolver has _quantum_wave_collapse: {hasattr(MCQSolver, '_quantum_wave_collapse')}")
assert hasattr(MCQSolver, '_quantum_wave_collapse'), "Missing _quantum_wave_collapse method!"
print("  PASS")

from l104_asi.commonsense_reasoning import CommonsenseMCQSolver
print(f"  CommonsenseMCQSolver has _quantum_wave_collapse: {hasattr(CommonsenseMCQSolver, '_quantum_wave_collapse')}")
assert hasattr(CommonsenseMCQSolver, '_quantum_wave_collapse'), "Missing _quantum_wave_collapse method!"
print("  PASS")

# Test 4: Check cached loaders exist
print("\n=== Test 4: Cached Loaders ===")
import l104_asi.language_comprehension as lc_mod
assert hasattr(lc_mod, '_get_cached_quantum_reasoning'), "Missing _get_cached_quantum_reasoning"
assert hasattr(lc_mod, '_get_cached_quantum_probability'), "Missing _get_cached_quantum_probability"
print("  MMLU quantum loaders: OK")

import l104_asi.commonsense_reasoning as cr_mod
assert hasattr(cr_mod, '_get_cached_quantum_probability'), "Missing _get_cached_quantum_probability"
print("  ARC quantum loaders: OK")

print("\n=== ALL TESTS PASSED ===")
