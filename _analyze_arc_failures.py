#!/usr/bin/env python3
"""Deep analysis of ARC failures to identify fixable patterns."""
import json, os, re
from collections import Counter
os.chdir(os.path.dirname(os.path.abspath(__file__)))

r = json.load(open('benchmark_full_online_results.json'))
arc_det = r['detailed_results']['ARC']['details']

# Separate correct/wrong
correct = [d for d in arc_det if d.get('correct')]
wrong = [d for d in arc_det if not d.get('correct')]

print(f"ARC: {len(correct)}/{len(arc_det)} correct ({len(correct)/len(arc_det)*100:.1f}%)")
print(f"  Challenge: {sum(1 for d in correct if d.get('category')=='arc_challenge')}/{sum(1 for d in arc_det if d.get('category')=='arc_challenge')}")
print(f"  Easy: {sum(1 for d in correct if d.get('category')=='arc_easy')}/{sum(1 for d in arc_det if d.get('category')=='arc_easy')}")

# ── Answer distribution analysis ──
print("\n── Answer Distribution ──")
exp_dist = Counter(d.get('expected') for d in arc_det)
got_dist = Counter(d.get('predicted') for d in arc_det)
print(f"  Expected: {dict(sorted(exp_dist.items()))}")
print(f"  Got:      {dict(sorted(got_dist.items()))}")

# Check if there's a systematic bias
print("\n── Bias check: Expected vs Got frequencies per choice ──")
for i in range(5):
    e = exp_dist.get(i, 0)
    g = got_dist.get(i, 0)
    if e > 0 or g > 0:
        print(f"  Choice {i}: Expected {e} times, Got {g} times (delta: {g-e:+d})")

# ── Confusion matrix ──
print("\n── Confusion Matrix (Exp x Got) ──")
conf = {}
for d in arc_det:
    e = d.get('expected', -1)
    g = d.get('predicted', -1)
    if e not in conf:
        conf[e] = Counter()
    conf[e][g] += 1

header = "     " + "".join(f"  G={i}" for i in range(5))
print(header)
for e in sorted(conf.keys()):
    row = f"E={e}:"
    for g in range(5):
        row += f"  {conf[e].get(g, 0):4d}"
    print(row)

# ── Question types in failures ──
print(f"\n── Content analysis of {len(wrong)} ARC failures ──")
# Look for question keywords
keyword_map = {
    'science/biology': ['cell', 'organism', 'plant', 'animal', 'species', 'gene', 'dna', 'evolution', 'ecosystem', 'food chain', 'photosynthesis'],
    'physics': ['force', 'energy', 'mass', 'gravity', 'speed', 'light', 'heat', 'temperature', 'magnet', 'electric', 'motion', 'wave'],
    'chemistry': ['element', 'chemical', 'molecule', 'atom', 'compound', 'reaction', 'solution', 'acid', 'metal', 'gas'],
    'earth_science': ['rock', 'mineral', 'weather', 'climate', 'erosion', 'earthquake', 'volcano', 'fossil', 'soil', 'ocean', 'atmosphere'],
    'astronomy': ['star', 'planet', 'moon', 'sun', 'solar', 'orbit', 'space', 'earth'],
    'experiment/method': ['experiment', 'hypothesis', 'variable', 'control', 'observe', 'measure', 'investigate', 'test', 'data', 'evidence'],
}

for cat, keywords in keyword_map.items():
    matches = [d for d in wrong if any(kw in d.get('question', '').lower() for kw in keywords)]
    total_in_cat = [d for d in arc_det if any(kw in d.get('question', '').lower() for kw in keywords)]
    correct_in_cat = [d for d in correct if any(kw in d.get('question', '').lower() for kw in keywords)]
    if total_in_cat:
        pct = len(correct_in_cat) / len(total_in_cat) * 100
        print(f"  {cat:>20}: {len(correct_in_cat)}/{len(total_in_cat)} = {pct:.0f}% correct ({len(matches)} wrong)")

# ── Look at choice length correlation ──
print("\n── Choice that's correct: avg word count ──")
for cat in ['arc_challenge', 'arc_easy']:
    cat_q = [d for d in arc_det if d.get('category') == cat]
    if not cat_q:
        continue
    # We don't have the actual choices in details, just predicted/expected indices

# ── Sample hard failures by category ──
challenge_wrong = [d for d in wrong if d.get('category') == 'arc_challenge']
easy_wrong = [d for d in wrong if d.get('category') == 'arc_easy']

print(f"\n── ARC-Challenge failures ({len(challenge_wrong)}) — 30 samples ──")
for d in challenge_wrong[:30]:
    q = d.get('question', '')[:100]
    print(f"  Q: {q}")
    print(f"    Exp={d.get('expected')}  Got={d.get('predicted')}")

print(f"\n── ARC-Easy failures ({len(easy_wrong)}) — 30 samples ──")
for d in easy_wrong[:30]:
    q = d.get('question', '')[:100]
    print(f"  Q: {q}")
    print(f"    Exp={d.get('expected')}  Got={d.get('predicted')}")
