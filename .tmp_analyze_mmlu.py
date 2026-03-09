#!/usr/bin/env python3
"""Analyze MMLU prediction distribution and failure patterns."""
import os, sys, time, json, collections
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import logging
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l104_asi.benchmark_harness import BenchmarkHarness, _HuggingFaceFetcher

h = BenchmarkHarness()
# Fetch MMLU questions — get 200 and sample 50 diversely
mmlu_data = _HuggingFaceFetcher.fetch_mmlu(max_questions=200)
print(f"Fetched {len(mmlu_data)} MMLU questions")

# Sample 1 from each subject, then fill the rest randomly
import random
subjects_seen = {}
diverse_sample = []
for d in mmlu_data:
    subj = d.get("subject", "unknown")
    if subj not in subjects_seen:
        subjects_seen[subj] = []
    subjects_seen[subj].append(d)

# Take up to 3 from each subject to ensure diversity
for subj, items in subjects_seen.items():
    diverse_sample.extend(items[:3])
if len(diverse_sample) > 100:
    random.shuffle(diverse_sample)
    diverse_sample = diverse_sample[:100]
print(f"Diverse sample: {len(diverse_sample)} questions from {len(subjects_seen)} subjects")

# Run evaluation
result = h._mmlu.evaluate(diverse_sample)
details = result.get("details", [])

# Prediction distribution
pred_dist = collections.Counter()
expected_dist = collections.Counter()
for d in details:
    pred_dist[d.get("predicted", -1)] += 1
    expected_dist[d.get("expected", -1)] += 1

print(f"\nAccuracy: {result['correct']}/{result['total']} ({result['score']*100:.1f}%)")
print(f"\nPrediction distribution (idx): {dict(sorted(pred_dist.items()))}")
print(f"Expected distribution (idx):   {dict(sorted(expected_dist.items()))}")

# Check if answer distribution is biased
print(f"\nPredicted labels: ", end="")
for i in range(4):
    pct = pred_dist.get(i, 0) / max(len(details), 1) * 100
    print(f"{chr(65+i)}={pred_dist.get(i,0)} ({pct:.0f}%) ", end="")
print()
print(f"Expected labels:  ", end="")
for i in range(4):
    pct = expected_dist.get(i, 0) / max(len(details), 1) * 100
    print(f"{chr(65+i)}={expected_dist.get(i,0)} ({pct:.0f}%) ", end="")
print()

# By-subject accuracy
print(f"\nPer-subject accuracy:")
by_subj = result.get("by_subject", {})
for subj, acc in sorted(by_subj.items(), key=lambda x: x[1], reverse=True):
    count = sum(1 for d in details if d.get("subject") == subj)
    print(f"  {subj}: {acc*100:.0f}% ({count}q)")

# Sample failures
print(f"\nSample failures:")
count = 0
for d in details:
    if d.get("correct"):
        continue
    count += 1
    if count > 10:
        break
    q = d.get("question", "")[:90]
    pred = chr(65 + d["predicted"]) if d["predicted"] >= 0 else "?"
    exp = chr(65 + d["expected"]) if d["expected"] >= 0 else "?"
    subj = d.get("subject", "?")
    print(f"  [{count}] Pred={pred} Exp={exp} [{subj}] {q}")
