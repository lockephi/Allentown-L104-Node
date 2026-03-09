#!/usr/bin/env python3
"""Analyze v17 benchmark results — distribution and spread."""
import json

with open('benchmark_full_online_results.json') as f:
    r = json.load(f)

b = r['benchmarks']
for k, v in b.items():
    print(f'{k}: {v["score"]:.1%} ({v["correct"]}/{v["total"]})')

# Estimated composite with HE at previous 66.5%
mmlu_s = b['MMLU']['score']
arc_s = b['ARC']['score']
math_s = b['MATH']['score']
he_s = 0.665  # from v16
composite = 0.25 * mmlu_s + 0.30 * he_s + 0.25 * math_s + 0.20 * arc_s
print(f'Estimated Composite (with HE=66.5%): {composite:.1%}')

# ARC per category
arc_det = r['detailed_results']['ARC']
if 'by_category' in arc_det:
    for cat, data in arc_det['by_category'].items():
        if isinstance(data, dict):
            print(f'  {cat}: {data.get("correct",0)}/{data.get("total",0)} = {data.get("score",0):.1%}')

# Try to get prediction distribution from the detailed results
# Check what keys are available in the category data
print("\nARC detail keys:", list(arc_det.keys()))
if 'results' in arc_det:
    results = arc_det['results']
    if isinstance(results, list) and len(results) > 0:
        print("Result item keys:", list(results[0].keys()))

# MMLU
mmlu_det = r['detailed_results']['MMLU']
print("\nMMLU detail keys:", list(mmlu_det.keys()))
if 'by_category' in mmlu_det:
    cats = mmlu_det['by_category']
    zero_cats = [k for k, v in cats.items() if isinstance(v, dict) and v.get('correct', 0) == 0]
    high_cats = [(k, v.get('score', 0)) for k, v in cats.items() if isinstance(v, dict) and v.get('score', 0) > 0.4]
    print(f"  Subjects at 0%: {len(zero_cats)} - {zero_cats}")
    print(f"  Subjects >40%: {len(high_cats)} - {sorted(high_cats, key=lambda x: -x[1])}")
