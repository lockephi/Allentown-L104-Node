#!/usr/bin/env python3
"""Run 500 MMLU + 250 ARC benchmark for real accuracy measurement."""
import os, sys, time, json, collections
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import logging
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l104_asi.benchmark_harness import BenchmarkHarness

print("=" * 60)
print("  L104 Benchmark — 500 MMLU + 250 ARC")
print("=" * 60)

t0 = time.time()
h = BenchmarkHarness()
results = h.run_all(online=True, mmlu_count=500, arc_count=250)
elapsed = time.time() - t0

bm = results.get("benchmarks", {})
print(f"\n{'=' * 60}")
print(f"  RESULTS ({elapsed:.1f}s)")
print(f"{'=' * 60}")
for name, data in bm.items():
    if isinstance(data, dict):
        acc = data.get("accuracy", data.get("score", "N/A"))
        count = data.get("total", data.get("count", "?"))
        correct = data.get("correct", "?")
        print(f"  {name}: {acc} ({correct}/{count})")
        # Show all keys for MMLU
        if name == "MMLU":
            print(f"    Keys: {list(data.keys())}")
            if "error" in data:
                print(f"    Error: {data['error']}")

# MMLU subject breakdown
mmlu = bm.get("MMLU", {})
by_subj = mmlu.get("by_subject", {})
if by_subj:
    print(f"\n  MMLU per-subject ({len(by_subj)} subjects):")
    for subj, acc in sorted(by_subj.items(), key=lambda x: x[1], reverse=True):
        print(f"    {subj}: {acc*100:.0f}%")
else:
    print(f"  No by_subject data found")

# Check prediction distribution from details
details = mmlu.get("details", [])
if details:
    import collections
    pred_dist = collections.Counter()
    exp_dist = collections.Counter()
    for d in details:
        pred_dist[d.get("predicted", -1)] += 1
        exp_dist[d.get("expected", -1)] += 1
    print(f"\n  MMLU prediction dist: {dict(sorted(pred_dist.items()))}")
    print(f"  MMLU expected dist:   {dict(sorted(exp_dist.items()))}")
    # Subjects
    subjects = set(d.get("subject", "?") for d in details)
    print(f"  Subjects ({len(subjects)}): {sorted(subjects)[:10]}")
print()
