#!/usr/bin/env python3
"""Analyze benchmark failure patterns."""
import json

with open("benchmark_online_results.json") as f:
    d = json.load(f)

# Count MMLU by subject
mmlu_details = d.get("detailed_results", {}).get("MMLU", {}).get("details", [])
by_subj = {}
for item in mmlu_details:
    s = item.get("subject", "unknown")
    if s not in by_subj:
        by_subj[s] = {"correct": 0, "total": 0}
    by_subj[s]["total"] += 1
    if item.get("correct"):
        by_subj[s]["correct"] += 1

print("=== MMLU BY SUBJECT ===")
for s in sorted(by_subj, key=lambda x: by_subj[x]["correct"] / max(by_subj[x]["total"], 1)):
    info = by_subj[s]
    pct = 100 * info["correct"] / max(info["total"], 1)
    print(f"  {s:40s}: {info['correct']:3d}/{info['total']:3d} ({pct:5.1f}%)")

# Show 5 sample wrong answers with choices
wrong = [x for x in mmlu_details if not x.get("correct")][:5]
print()
print("=== SAMPLE WRONG MMLU ===")
for w in wrong:
    q = w.get("question", "")[:100]
    print(f"  Q: {q}")
    print(f"    Exp={w.get('expected')} Pred={w.get('predicted')} Subj={w.get('subject')}")

# Count ARC by looking at score distribution
arc_details = d.get("detailed_results", {}).get("ARC", {}).get("details", [])
arc_wrong = [x for x in arc_details if not x.get("correct")]
arc_right = [x for x in arc_details if x.get("correct")]
print()
print(f"=== ARC SUMMARY ===")
print(f"  Correct: {len(arc_right)}")
print(f"  Wrong: {len(arc_wrong)}")

# Show 5 sample wrong ARC answers
print()
print("=== SAMPLE WRONG ARC ===")
for w in arc_wrong[:5]:
    q = w.get("question", "")[:100]
    print(f"  Q: {q}")
    print(f"    Exp={w.get('expected')} Pred={w.get('predicted')}")

# Show score patterns for wrong answers
print()
print("=== PREDICTION DISTRIBUTION FOR WRONG MMLU ===")
pred_counts = {}
for w in [x for x in mmlu_details if not x.get("correct")]:
    p = w.get("predicted", -1)
    pred_counts[p] = pred_counts.get(p, 0) + 1
for p in sorted(pred_counts):
    print(f"  Predicted {p}: {pred_counts[p]} times ({100*pred_counts[p]/len([x for x in mmlu_details if not x.get('correct')]):.1f}%)")

print()
print("=== PREDICTION DISTRIBUTION FOR WRONG ARC ===")
pred_counts = {}
for w in arc_wrong:
    p = w.get("predicted", -1)
    pred_counts[p] = pred_counts.get(p, 0) + 1
for p in sorted(pred_counts):
    print(f"  Predicted {p}: {pred_counts[p]} times ({100*pred_counts[p]/len(arc_wrong):.1f}%)")
