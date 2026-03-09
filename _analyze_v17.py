#!/usr/bin/env python3
"""Analyze v17 benchmark results."""
import json

with open('benchmark_full_online_results.json') as f:
    r = json.load(f)

b = r['benchmarks']
for k, v in b.items():
    print(f'{k}: {v["score"]:.1%} ({v["correct"]}/{v["total"]})')
print(f'Composite: {r["composite_score"]:.1%}')

d = r['detailed_results']
if 'arc' in d:
    arc_preds = [0,0,0,0]
    easy_c = easy_t = ch_c = ch_t = 0
    for q in d['arc']:
        idx = q.get('predicted_index', q.get('answer_index', 0))
        if isinstance(idx, int) and idx < 4:
            arc_preds[idx] += 1
        cat = q.get('category', '')
        correct = q.get('correct', False)
        if 'easy' in str(cat).lower():
            easy_t += 1
            if correct: easy_c += 1
        else:
            ch_t += 1
            if correct: ch_c += 1
    print(f'ARC Easy: {easy_c}/{easy_t} = {easy_c/max(easy_t,1):.1%}')
    print(f'ARC Challenge: {ch_c}/{ch_t} = {ch_c/max(ch_t,1):.1%}')
    print(f'ARC dist: {arc_preds}, spread: {max(arc_preds)-min(arc_preds)}')
if 'mmlu' in d:
    mmlu_preds = [0,0,0,0]
    for q in d['mmlu']:
        idx = q.get('predicted_index', q.get('answer_index', 0))
        if isinstance(idx, int) and idx < 4:
            mmlu_preds[idx] += 1
    print(f'MMLU dist: {mmlu_preds}, spread: {max(mmlu_preds)-min(mmlu_preds)}')
