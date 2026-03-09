#!/usr/bin/env python3
"""Analyze v18 benchmark results."""
import json
from collections import Counter

with open('benchmark_full_online_results.json') as f:
    r = json.load(f)

# ARC distribution
arc = r['detailed_results']['ARC']['details']

pred_dist = Counter(d['predicted'] for d in arc)
exp_dist = Counter(d['expected'] for d in arc)
print('=== ARC Choice Distribution ===')
print(f'Predicted: {dict(sorted(pred_dist.items()))}')
print(f'Expected:  {dict(sorted(exp_dist.items()))}')
spread = max(pred_dist.values()) - min(pred_dist.values())
print(f'Pred Spread: {spread}')

# ARC per category
easy = [d for d in arc if d.get('category') == 'arc_easy']
chal = [d for d in arc if d.get('category') == 'arc_challenge']
easy_correct = sum(1 for d in easy if d['correct'])
chal_correct = sum(1 for d in chal if d['correct'])
print(f'\nARC Easy: {easy_correct}/{len(easy)} = {easy_correct/len(easy)*100:.1f}%')
print(f'ARC Challenge: {chal_correct}/{len(chal)} = {chal_correct/len(chal)*100:.1f}%')

# ARC per-choice accuracy
print('\nARC Per-Answer Accuracy:')
for c in range(4):
    subset = [d for d in arc if d['expected'] == c]
    correct = sum(1 for d in subset if d['correct'])
    pct = correct/len(subset)*100 if subset else 0
    print(f'  When answer={c}: {correct}/{len(subset)} = {pct:.1f}%')

# MMLU distribution
mmlu = r['detailed_results']['MMLU']['details']
pred_dist_m = Counter(d['predicted'] for d in mmlu)
exp_dist_m = Counter(d['expected'] for d in mmlu)
print(f'\n=== MMLU Choice Distribution ===')
print(f'Predicted: {dict(sorted(pred_dist_m.items()))}')
print(f'Expected:  {dict(sorted(exp_dist_m.items()))}')
spread_m = max(pred_dist_m.values()) - min(pred_dist_m.values())
print(f'Pred Spread: {spread_m}')

# MMLU per-choice accuracy
print('\nMMLU Per-Answer Accuracy:')
for c in range(4):
    subset = [d for d in mmlu if d['expected'] == c]
    correct = sum(1 for d in subset if d['correct'])
    pct = correct/len(subset)*100 if subset else 0
    print(f'  When answer={c}: {correct}/{len(subset)} = {pct:.1f}%')

# ARC confusion matrix
wrong_arc = [d for d in arc if not d['correct']]
confusion = Counter()
for d in wrong_arc:
    confusion[(d['expected'], d['predicted'])] += 1
print(f'\n=== ARC Confusion (top 12) ===')
for (exp, pred), cnt in confusion.most_common(12):
    print(f'  Expected {exp} -> Predicted {pred}: {cnt} times')

# ARC: analyze failure themes via keywords
print(f'\n=== ARC Failure Theme Analysis ===')
themes = {
    'temperature/heat': ['temperature', 'heat', 'cold', 'warm', 'cool', 'thermal', 'boiling', 'freezing'],
    'force/motion': ['force', 'motion', 'speed', 'velocity', 'acceleration', 'push', 'pull', 'gravity'],
    'life science': ['cell', 'organism', 'species', 'plant', 'animal', 'predator', 'prey', 'habitat'],
    'earth science': ['rock', 'mineral', 'fossil', 'earthquake', 'volcano', 'erosion', 'weather', 'climate'],
    'energy': ['energy', 'solar', 'electricity', 'battery', 'power', 'fuel', 'renewable'],
    'water cycle': ['evaporation', 'condensation', 'precipitation', 'water cycle', 'rain', 'cloud'],
    'light/sound': ['light', 'sound', 'wave', 'reflection', 'refraction', 'shadow', 'lens'],
    'matter': ['solid', 'liquid', 'gas', 'matter', 'mixture', 'solution', 'dissolve', 'chemical'],
    'experiment': ['experiment', 'hypothesis', 'variable', 'control', 'data', 'measure', 'observe'],
}
for theme, keywords in themes.items():
    theme_wrong = [d for d in wrong_arc if any(k in d['question'].lower() for k in keywords)]
    theme_all = [d for d in arc if any(k in d['question'].lower() for k in keywords)]
    theme_correct = len(theme_all) - len(theme_wrong)
    if theme_all:
        pct = theme_correct / len(theme_all) * 100
        print(f'  {theme:20s}: {theme_correct}/{len(theme_all)} = {pct:.1f}%')

# Score range analysis - how close are scores?
print(f'\n=== Score Margin Analysis ===')
# Can't get scores from JSON, but can analyze if wrong predictions cluster
wrong_by_pred = Counter(d['predicted'] for d in wrong_arc)
print(f'Wrong predictions cluster: {dict(sorted(wrong_by_pred.items()))}')

# Compare v18 to v17 expectations
print(f'\n=== v18 Summary ===')
print(f"ARC: {r['benchmarks']['ARC']['score']*100:.1f}%")
print(f"MMLU: {r['benchmarks']['MMLU']['score']*100:.1f}%")
print(f"MATH: {r['benchmarks']['MATH']['score']*100:.1f}%")
print(f"HumanEval: {r['benchmarks']['HumanEval']['score']*100:.1f}%")
print(f"Composite: {r['composite_score']*100:.1f}%")
