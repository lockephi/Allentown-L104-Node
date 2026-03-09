#!/usr/bin/env python3
"""Analyze v15 benchmark results."""
import json
from collections import Counter

d = json.load(open('benchmark_full_online_results.json'))
arc = d['detailed_results']['ARC']['details']
easy = [x for x in arc if 'easy' in str(x.get('category','')).lower()]
chal = [x for x in arc if 'challenge' in str(x.get('category','')).lower()]
e_c = sum(1 for x in easy if x.get('correct'))
c_c = sum(1 for x in chal if x.get('correct'))
print(f'Easy: {e_c}/{len(easy)} = {e_c/len(easy)*100:.1f}%')
print(f'Challenge: {c_c}/{len(chal)} = {c_c/len(chal)*100:.1f}%')

pred = Counter(x.get('predicted','?') for x in arc)
exp = Counter(x.get('expected','?') for x in arc)
print(f'Pred dist: {sorted(pred.items())}')
print(f'Exp  dist: {sorted(exp.items())}')
spread = sum(abs(pred.get(k,0) - exp.get(k,0)) for k in set(list(pred.keys()) + list(exp.keys())))
print(f'Spread: {spread}')

# MMLU breakdown
mmlu = d['detailed_results']['MMLU']['details']
subjects = Counter()
subject_correct = Counter()
for x in mmlu:
    s = x.get('subject', 'unknown')
    subjects[s] += 1
    if x.get('correct'):
        subject_correct[s] += 1

print(f'\nMMLU: {sum(subject_correct.values())}/{sum(subjects.values())}')
print('Top subjects:')
for s, n in sorted(subjects.items(), key=lambda x: subject_correct[x[0]]/max(x[1],1), reverse=True)[:10]:
    c = subject_correct[s]
    print(f'  {s}: {c}/{n} = {c/n*100:.0f}%')
print('Bottom subjects:')
for s, n in sorted(subjects.items(), key=lambda x: subject_correct[x[0]]/max(x[1],1))[:10]:
    c = subject_correct[s]
    print(f'  {s}: {c}/{n} = {c/n*100:.0f}%')

# MMLU pred distribution
mpred = Counter(x.get('predicted','?') for x in mmlu)
mexp = Counter(x.get('expected','?') for x in mmlu)
print(f'\nMMLU Pred: {sorted(mpred.items())}')
print(f'MMLU Exp:  {sorted(mexp.items())}')
mspread = sum(abs(mpred.get(k,0) - mexp.get(k,0)) for k in set(list(mpred.keys()) + list(mexp.keys())))
print(f'MMLU Spread: {mspread}')
