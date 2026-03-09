#!/usr/bin/env python3
"""Analyze benchmark results from _benchmark_quantum_results.json."""
import json
from collections import defaultdict, Counter

with open("_benchmark_quantum_results.json") as f:
    data = json.load(f)

# MMLU
mmlu = data["detailed_results"]["MMLU"]
print("MMLU score:", data["benchmarks"]["MMLU"])
details = mmlu.get("details", [])
print("Detail entries:", len(details))
if details:
    print("Sample entry:", json.dumps(details[0])[:300])

# Per-subject accuracy
subj_correct = defaultdict(int)
subj_total = defaultdict(int)
for d in details:
    subj = d.get("subject", "unknown")
    subj_total[subj] += 1
    if d.get("correct"):
        subj_correct[subj] += 1

print("\nPer-subject accuracy (%d subjects):" % len(subj_total))
for subj in sorted(subj_total.keys(), key=lambda s: subj_correct[s]/max(subj_total[s],1), reverse=True):
    t = subj_total[subj]
    c = subj_correct[subj]
    pct = 100*c/t if t else 0
    print("  %s: %d/%d (%.0f%%)" % (subj, c, t, pct))

# Predicted distribution
pred_dist = Counter(d.get("predicted", "?") for d in details)
print("\nMMKU predicted distribution:", dict(pred_dist.most_common()))

exp_dist = Counter(d.get("expected", "?") for d in details)
print("MMLU expected distribution:", dict(exp_dist.most_common()))

# ARC
print("\nARC score:", data["benchmarks"]["ARC"])
arc = data["detailed_results"]["ARC"]
arc_details = arc.get("details", [])
print("ARC detail entries:", len(arc_details))
if arc_details:
    arc_correct = sum(1 for d in arc_details if d.get("correct"))
    print("ARC correct: %d/%d" % (arc_correct, len(arc_details)))
    print("Sample ARC entry:", json.dumps(arc_details[0])[:300])

    arc_pred = Counter(d.get("predicted", "?") for d in arc_details)
    print("\nARC predicted distribution:", dict(arc_pred.most_common()))
    arc_exp = Counter(d.get("expected", "?") for d in arc_details)
    print("ARC expected distribution:", dict(arc_exp.most_common()))

# Show 5 wrong MMLU questions
print("\n5 sample WRONG MMLU answers:")
wrong = [d for d in details if not d.get("correct")]
for d in wrong[:5]:
    print("  Q: %s" % d.get("question", "?")[:100])
    print("  Subj: %s | Predicted: %s | Expected: %s" % (
        d.get("subject", "?"), d.get("predicted", "?"), d.get("expected", "?")))
    print()

# Show 5 wrong ARC
print("5 sample WRONG ARC answers:")
arc_wrong = [d for d in arc_details if not d.get("correct")]
for d in arc_wrong[:5]:
    print("  Q: %s" % d.get("question", "?")[:100])
    print("  Predicted: %s | Expected: %s" % (d.get("predicted", "?"), d.get("expected", "?")))
    print()
