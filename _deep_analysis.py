#!/usr/bin/env python3
"""Deep analysis of scoring patterns — understand knowledge vs accuracy relationship."""
import json

with open("benchmark_online_results.json") as f:
    d = json.load(f)

# MMLU: Analyze relationship between knowledge hits and correctness
mmlu_details = d.get("detailed_results", {}).get("MMLU", {}).get("details", [])

# Split by subject to find best and worst performing
by_subj = {}
for item in mmlu_details:
    s = item.get("subject", "unknown")
    if s not in by_subj:
        by_subj[s] = {"correct": 0, "total": 0, "items": []}
    by_subj[s]["total"] += 1
    if item.get("correct"):
        by_subj[s]["correct"] += 1
    by_subj[s]["items"].append(item)

print("=== MMLU DETAILED ANALYSIS ===")
subjects_sorted = sorted(by_subj.items(), key=lambda x: x[1]["correct"]/max(x[1]["total"],1), reverse=True)
for s, info in subjects_sorted:
    pct = 100*info["correct"]/max(info["total"],1)
    print(f"  {s:40s}: {info['correct']:3d}/{info['total']:3d} ({pct:5.1f}%)")

# Count how many subjects are above vs below random
above_random = sum(1 for _, info in subjects_sorted if info["correct"]/max(info["total"],1) > 0.30)
at_random = sum(1 for _, info in subjects_sorted if 0.20 <= info["correct"]/max(info["total"],1) <= 0.30)
below_random = sum(1 for _, info in subjects_sorted if info["correct"]/max(info["total"],1) < 0.20)
print(f"\n  Above 30%: {above_random} subjects")
print(f"  At random (20-30%): {at_random} subjects")
print(f"  Below 20%: {below_random} subjects")

# Check prediction distribution for ALL answers
print("\n=== PREDICTION DISTRIBUTION (ALL MMLU) ===")
pred_all = {}
exp_all = {}
for item in mmlu_details:
    p = item.get("predicted", -1)
    e = item.get("expected", -1)
    pred_all[p] = pred_all.get(p, 0) + 1
    exp_all[e] = exp_all.get(e, 0) + 1
print("  Predicted distribution:", {k: f"{100*v/len(mmlu_details):.1f}%" for k, v in sorted(pred_all.items())})
print("  Expected distribution: ", {k: f"{100*v/len(mmlu_details):.1f}%" for k, v in sorted(exp_all.items())})

# ARC analysis
arc_details = d.get("detailed_results", {}).get("ARC", {}).get("details", [])
print("\n=== ARC PREDICTION DISTRIBUTION ===")
pred_all = {}
exp_all = {}
for item in arc_details:
    p = item.get("predicted", -1)
    e = item.get("expected", -1)
    pred_all[p] = pred_all.get(p, 0) + 1
    exp_all[e] = exp_all.get(e, 0) + 1
print("  Predicted distribution:", {k: f"{100*v/len(arc_details):.1f}%" for k, v in sorted(pred_all.items())})
print("  Expected distribution: ", {k: f"{100*v/len(arc_details):.1f}%" for k, v in sorted(exp_all.items())})

# Show correct answer distribution for ARC wrongs only
wrong_arc = [x for x in arc_details if not x.get("correct")]
print(f"\n=== WRONG ARC ANSWERS ({len(wrong_arc)} total) ===")
pred_wrong = {}
for item in wrong_arc:
    p = item.get("predicted", -1)
    pred_wrong[p] = pred_wrong.get(p, 0) + 1
print("  Wrong predictions:", {k: f"{v} ({100*v/len(wrong_arc):.1f}%)" for k, v in sorted(pred_wrong.items())})

# And for MMLU wrongs
wrong_mmlu = [x for x in mmlu_details if not x.get("correct")]
print(f"\n=== WRONG MMLU ANSWERS ({len(wrong_mmlu)} total) ===")
pred_wrong = {}
for item in wrong_mmlu:
    p = item.get("predicted", -1)
    pred_wrong[p] = pred_wrong.get(p, 0) + 1
print("  Wrong predictions:", {k: f"{v} ({100*v/len(wrong_mmlu):.1f}%)" for k, v in sorted(pred_wrong.items())})
