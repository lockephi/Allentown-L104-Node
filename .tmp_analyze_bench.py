#!/usr/bin/env python3
"""Analyze benchmark weakness patterns."""
import json
from collections import Counter

with open("benchmark_online_results.json") as f:
    data = json.load(f)

# MMLU analysis
mmlu = data.get("detailed_results", {}).get("MMLU", {})
by_sub = mmlu.get("by_subject", {})
details = mmlu.get("details", [])

print("=== MMLU By Subject (worst → best) ===")
sorted_subs = sorted(by_sub.items(), key=lambda x: x[1])
for s, v in sorted_subs[:8]:
    print(f"  {s}: {v:.1%}")
print("  ...")
for s, v in sorted_subs[-5:]:
    print(f"  {s}: {v:.1%}")

# Statement questions
stmts = [d for d in details if "statement" in d.get("question", "").lower()[:20]]
stmt_correct = sum(1 for d in stmts if d.get("correct"))
print(f"\nStatement Qs: {stmt_correct}/{len(stmts)} correct ({stmt_correct/max(len(stmts),1):.0%})")

# Prediction distribution
preds = [d.get("predicted", 0) for d in details]
pred_counts = Counter(preds)
print(f"\nMMLU prediction distribution: {dict(pred_counts)}")
# Expected distribution
expected = [d.get("expected", 0) for d in details]
exp_counts = Counter(expected)
print(f"MMLU expected distribution:   {dict(exp_counts)}")

# Analyze index bias
print("\nMMLU accuracy by predicted index:")
for idx in range(4):
    predicted_as_idx = [d for d in details if d.get("predicted") == idx]
    correct_of_those = sum(1 for d in predicted_as_idx if d.get("correct"))
    print(f"  Predicted {idx}: {correct_of_those}/{len(predicted_as_idx)} ({correct_of_those/max(len(predicted_as_idx),1):.0%})")

# ARC analysis
arc = data.get("detailed_results", {}).get("ARC", {})
arc_details = arc.get("details", [])
arc_preds = [d.get("predicted", 0) for d in arc_details]
arc_pred_counts = Counter(arc_preds)
arc_exp = [d.get("expected", 0) for d in arc_details]
arc_exp_counts = Counter(arc_exp)
print(f"\n=== ARC Analysis ===")
print(f"ARC prediction distribution: {dict(arc_pred_counts)}")
print(f"ARC expected distribution:   {dict(arc_exp_counts)}")
arc_correct = sum(1 for d in arc_details if d.get("correct"))
print(f"ARC correct: {arc_correct}/{len(arc_details)}")

# ARC index accuracy
print("\nARC accuracy by predicted index:")
for idx in range(5):
    predicted_as_idx = [d for d in arc_details if d.get("predicted") == idx]
    correct_of_those = sum(1 for d in predicted_as_idx if d.get("correct"))
    if predicted_as_idx:
        print(f"  Predicted {idx}: {correct_of_those}/{len(predicted_as_idx)} ({correct_of_those/max(len(predicted_as_idx),1):.0%})")

# Sample some wrong MMLU questions to understand failure patterns
print("\n=== Sample MMLU Failures ===")
wrong = [d for d in details if not d.get("correct")]
for d in wrong[:8]:
    print(f"  Q: {d['question'][:80]}...")
    print(f"    Expected: {d['expected']} Predicted: {d['predicted']} Subj: {d.get('subject','?')}")

# Sample wrong ARC
print("\n=== Sample ARC Failures ===")
wrong_arc = [d for d in arc_details if not d.get("correct")]
for d in wrong_arc[:8]:
    print(f"  Q: {d.get('question','?')[:80]}...")
    print(f"    Expected: {d['expected']} Predicted: {d['predicted']}")
