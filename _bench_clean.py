#!/usr/bin/env python3
"""Run full benchmark — clean, no hardcoded data."""
import json, sys, os, logging
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
logging.disable(logging.WARNING)

from l104_asi.benchmark_harness import BenchmarkHarness
h = BenchmarkHarness()
results = h.run_all(online=True)

print("\n" + "="*60)
print("  BENCHMARK RESULTS (Pattern-Based, No Hardcoded Data)")
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
print("="*60)

with open("_benchmark_clean_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
