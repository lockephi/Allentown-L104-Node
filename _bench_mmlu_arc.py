#!/usr/bin/env python3
"""Run MMLU + ARC benchmarks only (skip HumanEval/MATH which are already 100%)."""
import json, os, time, logging
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
logging.disable(logging.WARNING)

from l104_asi.benchmark_harness import BenchmarkHarness

h = BenchmarkHarness()
start = time.time()

print("=" * 60)
print("  MMLU + ARC Benchmark (Online — HuggingFace)")
print("=" * 60)

# Fetch from HuggingFace
from l104_asi.benchmark_harness import _HuggingFaceFetcher as fetcher

# MMLU — 500 questions
print("\n[1/4] Fetching MMLU data...")
mmlu_data = fetcher.fetch_mmlu(max_questions=500)
print(f"       Got {len(mmlu_data)} MMLU questions")

print("[2/4] Running MMLU evaluation...")
t0 = time.time()
mmlu_result = h._mmlu.evaluate(mmlu_data)
mmlu_time = time.time() - t0
mmlu_score = mmlu_result.get("score", mmlu_result.get("accuracy", 0))
mmlu_correct = mmlu_result.get("correct", 0)
mmlu_total = mmlu_result.get("total", len(mmlu_data))
print(f"       MMLU: {mmlu_score*100:.1f}% ({mmlu_correct}/{mmlu_total}) in {mmlu_time:.1f}s")

# ARC — 1000 questions (500 easy + 500 challenge)
print("\n[3/4] Fetching ARC data...")
arc_data = fetcher.fetch_arc(max_questions=500, include_easy=True)
print(f"       Got {len(arc_data)} ARC questions")

print("[4/4] Running ARC evaluation...")
t0 = time.time()
arc_result = h._arc.evaluate(arc_data)
arc_time = time.time() - t0
arc_score = arc_result.get("score", arc_result.get("accuracy", 0))
arc_correct = arc_result.get("correct", 0)
arc_total = arc_result.get("total", len(arc_data))
print(f"       ARC: {arc_score*100:.1f}% ({arc_correct}/{arc_total}) in {arc_time:.1f}s")

elapsed = time.time() - start

print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"  MMLU:  {mmlu_score*100:6.1f}%  ({mmlu_correct}/{mmlu_total})  [was 29.6%]")
print(f"  ARC:   {arc_score*100:6.1f}%  ({arc_correct}/{arc_total})  [was 28.7%]")
print(f"  Time:  {elapsed:.0f}s")
print("=" * 60)

# Save detailed results
report = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "mmlu": mmlu_result,
    "arc": arc_result,
    "elapsed_seconds": elapsed,
    "previous": {"mmlu": 0.296, "arc": 0.287},
}
with open("_bench_mmlu_arc_results.json", "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"\nDetailed results saved to _bench_mmlu_arc_results.json")
