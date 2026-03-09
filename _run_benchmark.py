#!/usr/bin/env python3
"""Run benchmark and print results."""
import json, sys, os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import logging
logging.disable(logging.WARNING)

from l104_asi.benchmark_harness import BenchmarkHarness

h = BenchmarkHarness()
results = h.run_all(online=True)

print("\n" + "="*60)
print("  BENCHMARK RESULTS (with Quantum Integration)")
print("="*60)

bm = results.get("benchmarks", {})
for k, v in bm.items():
    if isinstance(v, dict):
        score = v.get("score", v.get("accuracy", "?"))
        total = v.get("total", "?")
        correct = v.get("correct", v.get("passed", "?"))
        if isinstance(score, float):
            print(f"  {k:12s}: {score*100:6.1f}%  ({correct}/{total})")
        else:
            print(f"  {k:12s}: {score}")

print(f"\n  COMPOSITE:     {results.get('composite_score', 0)*100:.1f}%")
print(f"  VERDICT:       {results.get('verdict', '?')}")
print(f"  GOD_CODE +     {results.get('god_code_bonus', 0)*100:.2f}%")
print("="*60)

# Also check detailed_results
dr = results.get("detailed_results", {})
if dr:
    print("\nDetailed sub-scores:")
    for k, v in dr.items():
        if isinstance(v, dict):
            score = v.get("score", v.get("accuracy", "?"))
            if isinstance(score, float):
                print(f"  {k}: {score*100:.1f}%")
            else:
                print(f"  {k}: {score}")

# Save to file
with open("_benchmark_quantum_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nResults saved to _benchmark_quantum_results.json")
