#!/usr/bin/env python3
"""Deep ARC failure analysis for v18."""
import json
from collections import Counter

with open('benchmark_full_online_results.json') as f:
    r = json.load(f)

arc = r['detailed_results']['ARC']['details']
wrong = [d for d in arc if not d['correct']]

# 1. Choice 3 over-prediction analysis
print('=== CHOICE 3 OVER-PREDICTION ===')
c3_wrong = [d for d in wrong if d['predicted'] == 3]
print(f'Total wrong choice-3 predictions: {len(c3_wrong)}')
# Are these questions where choice 3 tends to be longest?
# Can't tell from JSON alone, but let's see the expected distribution
c3_exp = Counter(d['expected'] for d in c3_wrong)
print(f'True answers when we wrongly predicted 3: {dict(sorted(c3_exp.items()))}')

# 2. Sample failures from weak themes
themes = {
    'temperature/heat': ['temperature', 'heat', 'cold', 'warm', 'cool', 'thermal', 'boiling', 'freezing'],
    'water cycle': ['evaporation', 'condensation', 'precipitation', 'water cycle', 'rain', 'cloud'],
    'force/motion': ['force', 'motion', 'speed', 'velocity', 'acceleration', 'push', 'pull', 'gravity'],
}

for theme, keywords in themes.items():
    theme_wrong = [d for d in wrong if any(k in d['question'].lower() for k in keywords)]
    print(f'\n=== {theme.upper()} FAILURES ({len(theme_wrong)}) ===')
    for d in theme_wrong[:8]:
        q = d['question'][:120]
        print(f'  Q: {q}...')
        print(f'    Exp={d["expected"]} Got={d["predicted"]} Cat={d.get("category","")}')

# 3. Analyze answer length patterns
# Group by whether predicted answer is the longest in its question group
# We can't get choices from the JSON, but we can check if choice 3 is systematically longer

# 4. Easy vs Challenge failure rate by theme
print('\n=== EASY vs CHALLENGE by theme ===')
all_themes = {
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
for theme, keywords in all_themes.items():
    theme_qs = [d for d in arc if any(k in d['question'].lower() for k in keywords)]
    easy_qs = [d for d in theme_qs if d.get('category') == 'arc_easy']
    chal_qs = [d for d in theme_qs if d.get('category') == 'arc_challenge']
    easy_c = sum(1 for d in easy_qs if d['correct'])
    chal_c = sum(1 for d in chal_qs if d['correct'])
    e_pct = easy_c/len(easy_qs)*100 if easy_qs else 0
    c_pct = chal_c/len(chal_qs)*100 if chal_qs else 0
    print(f'  {theme:20s}: Easy {easy_c}/{len(easy_qs)}={e_pct:.0f}%  Chal {chal_c}/{len(chal_qs)}={c_pct:.0f}%')

# 5. Which questions have expected=0 but we predict 3? (biggest confusion pair)
print('\n=== Expected=0, Predicted=3 samples (65 cases) ===')
exp0_pred3 = [d for d in wrong if d['expected'] == 0 and d['predicted'] == 3]
for d in exp0_pred3[:10]:
    q = d['question'][:120]
    print(f'  [{d.get("category","")}] {q}')
