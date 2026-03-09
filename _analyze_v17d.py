#!/usr/bin/env python3
"""Final v17 analysis with distributions."""
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
details = arc_det.get('details', [])
preds = [0,0,0,0]
easy_c = easy_t = ch_c = ch_t = 0
for d in details:
    idx = d.get('predicted', -1)
    if isinstance(idx, int) and 0 <= idx < 4:
        preds[idx] += 1
    cat = d.get('category', '')
    correct = d.get('correct', False)
    if 'easy' in str(cat).lower():
        easy_t += 1
        if correct: easy_c += 1
    else:
        ch_t += 1
        if correct: ch_c += 1
print(f'\nARC Easy: {easy_c}/{easy_t} = {easy_c/max(easy_t,1):.1%}')
print(f'ARC Challenge: {ch_c}/{ch_t} = {ch_c/max(ch_t,1):.1%}')
print(f'ARC dist: {preds}, spread: {max(preds)-min(preds)}')

# MMLU
mmlu_det = r['detailed_results']['MMLU']
details = mmlu_det.get('details', [])
preds = [0,0,0,0]
for d in details:
    idx = d.get('predicted', -1)
    if isinstance(idx, int) and 0 <= idx < 4:
        preds[idx] += 1
print(f'MMLU dist: {preds}, spread: {max(preds)-min(preds)}')

# MMLU by subject - sort by score
subjects = mmlu_det.get('by_subject', {})
subj_data = []
for k, v in subjects.items():
    if isinstance(v, dict):
        c = v.get('correct', 0)
        t = v.get('total', 0)
        subj_data.append((k, c, t, c/max(t,1)))
subj_data.sort(key=lambda x: -x[3])
print(f'\nMMLU top 10:')
for name, c, t, score in subj_data[:10]:
    print(f'  {name}: {c}/{t} = {score:.0%}')
print(f'MMLU bottom 5:')
for name, c, t, score in subj_data[-5:]:
    print(f'  {name}: {c}/{t} = {score:.0%}')
