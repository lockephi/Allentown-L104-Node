#!/usr/bin/env python3
"""Analyze v17 with details - distributions, categories, failures."""
import json

with open('benchmark_full_online_results.json') as f:
    r = json.load(f)

b = r['benchmarks']
for k, v in b.items():
    print(f'{k}: {v["score"]:.1%} ({v["correct"]}/{v["total"]})')

he_s = 0.665
composite = 0.25 * b['MMLU']['score'] + 0.30 * he_s + 0.25 * b['MATH']['score'] + 0.20 * b['ARC']['score']
print(f'Est. Composite (HE=66.5%): {composite:.1%}')

# ARC
arc_det = r['detailed_results']['ARC']
if 'by_category' in arc_det:
    print('\nARC by category:')
    for cat, data in arc_det['by_category'].items():
        if isinstance(data, dict):
            c = data.get('correct', 0)
            t = data.get('total', 0)
            print(f'  {cat}: {c}/{t} = {c/max(t,1):.1%}')

# ARC details - distribution
if 'details' in arc_det:
    details = arc_det['details']
    if isinstance(details, list):
        preds = [0,0,0,0]
        for d in details:
            idx = d.get('predicted_index', d.get('answer_index', -1))
            if isinstance(idx, int) and 0 <= idx < 4:
                preds[idx] += 1
        if sum(preds) > 0:
            print(f'  ARC dist: {preds}, spread: {max(preds)-min(preds)}')
        else:
            # check what keys are in detail items
            if len(details) > 0:
                print(f'  Detail keys: {list(details[0].keys())}')
                print(f'  Sample: {details[0]}')

# MMLU
mmlu_det = r['detailed_results']['MMLU']
if 'by_subject' in mmlu_det:
    subjects = mmlu_det['by_subject']
    zero_cats = [k for k, v in subjects.items() if isinstance(v, dict) and v.get('correct', 0) == 0]
    high_cats = [(k, v.get('correct',0), v.get('total',0)) for k, v in subjects.items()
                 if isinstance(v, dict) and v.get('total',0) > 0 and v.get('correct',0)/v.get('total',1) > 0.4]
    print(f'\nMMLU at 0%: {len(zero_cats)} subjects: {zero_cats}')
    print(f'MMLU >40%: {len(high_cats)} subjects: {sorted(high_cats, key=lambda x: -x[1]/max(x[2],1))}')

if 'details' in mmlu_det:
    details = mmlu_det['details']
    if isinstance(details, list):
        preds = [0,0,0,0]
        for d in details:
            idx = d.get('predicted_index', d.get('answer_index', -1))
            if isinstance(idx, int) and 0 <= idx < 4:
                preds[idx] += 1
        if sum(preds) > 0:
            print(f'  MMLU dist: {preds}, spread: {max(preds)-min(preds)}')
        elif len(details) > 0:
            print(f'  MMLU detail keys: {list(details[0].keys())}')
