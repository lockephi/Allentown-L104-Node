#!/usr/bin/env python3
"""Full benchmark — all 4 engines, sequential. Writes result to /tmp/full_bench.json"""
import os, sys, json, time
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import logging
logging.disable(logging.WARNING)

from l104_asi.benchmark_harness import BenchmarkHarness

print("Starting full benchmark...", flush=True)
t0 = time.time()
h = BenchmarkHarness()
results = h.run_all(online=True)
elapsed = time.time() - t0

print(f"\n{'='*60}", flush=True)
print(f"  BENCHMARK RESULTS ({elapsed:.0f}s)", flush=True)
print(f"{'='*60}", flush=True)

bm = results.get("benchmarks", {})
for k, v in bm.items():
    if isinstance(v, dict):
        score = v.get("score", v.get("accuracy", "?"))
        total = v.get("total", "?")
        correct = v.get("correct", v.get("passed", "?"))
        if isinstance(score, float):
            print(f"  {k:12s}: {score*100:6.1f}%  ({correct}/{total})", flush=True)
        else:
            print(f"  {k:12s}: {score}", flush=True)

print(f"\n  COMPOSITE:     {results.get('composite_score', 0)*100:.1f}%", flush=True)
print(f"  VERDICT:       {results.get('verdict', '?')}", flush=True)

with open('/tmp/full_bench.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("Result saved to /tmp/full_bench.json", flush=True)
