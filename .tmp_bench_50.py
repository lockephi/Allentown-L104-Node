#!/usr/bin/env python3
"""Quick 50-question MMLU benchmark to test improvements."""
import os, sys, time, json
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import logging
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l104_asi.benchmark_harness import BenchmarkHarness

print("=" * 60)
print("  L104 Quick Benchmark — 50 MMLU + 25 ARC")
print("=" * 60)

t0 = time.time()
h = BenchmarkHarness()
results = h.run_all(online=True, mmlu_count=50, arc_count=25)
elapsed = time.time() - t0

bm = results.get("benchmarks", {})
print(f"\n{'=' * 60}")
print(f"  RESULTS ({elapsed:.1f}s)")
print(f"{'=' * 60}")
for name, data in bm.items():
    if isinstance(data, dict):
        acc = data.get("accuracy", data.get("score", "N/A"))
        count = data.get("total", data.get("count", "?"))
        print(f"  {name}: {acc} ({count} questions)")
print()
