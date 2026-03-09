#!/usr/bin/env python3
"""Analyze v8 benchmark bias by ARC category."""
import json
from collections import Counter

data = json.load(open('benchmark_full_online_results.json'))
arc = data['detailed_results']['ARC']['details']

for cat in ['arc_challenge', 'arc_easy']:
    items = [r for r in arc if r.get('category') == cat]
    correct = sum(1 for r in items if r['correct'])
    got_dist = dict(sorted(Counter(r['predicted'] for r in items).items()))
    exp_dist = dict(sorted(Counter(r['expected'] for r in items).items()))
    spread = max(got_dist.values()) - min(got_dist.values())
    print(f"\n{cat}: {correct}/{len(items)} = {100*correct/len(items):.1f}%")
    print(f"  Got: {got_dist}  spread={spread}")
    print(f"  Exp: {exp_dist}")
