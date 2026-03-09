#!/usr/bin/env python3
"""Analyze ARC failures to find systematic patterns."""
import json
from collections import Counter

d = json.load(open('benchmark_full_online_results.json'))
arc = d['detailed_results']['ARC']['details']

# Analyze when we predict each index vs expected
confusion = Counter()  # (predicted, expected) pairs
for x in arc:
    if not x.get('correct'):
        p = x.get('predicted', -1)
        e = x.get('expected', -1)
        confusion[(p, e)] += 1

print("Confusion pairs (pred→exp): count")
for (p, e), cnt in confusion.most_common(20):
    print(f"  Pred {p} → Exp {e}: {cnt}")

# Check if longer answers systematically win
wrong = [x for x in arc if not x.get('correct')]
print(f"\nTotal wrong: {len(wrong)}")

# Check the predicted vs expected index distributions for wrong answers
wrong_pred_higher = sum(1 for x in wrong if x['predicted'] > x['expected'])
wrong_pred_lower = sum(1 for x in wrong if x['predicted'] < x['expected'])
wrong_pred_same_bucket = sum(1 for x in wrong if x['predicted'] == x['expected'])
print(f"Wrong answers - pred > exp: {wrong_pred_higher}, pred < exp: {wrong_pred_lower}")
