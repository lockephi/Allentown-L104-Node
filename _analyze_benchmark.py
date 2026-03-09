#!/usr/bin/env python3
"""Analyze the saved benchmark results."""
import json, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

r = json.load(open('benchmark_full_online_results.json'))
print('Mode:', r.get('mode'))
print('Total Q:', r.get('total_questions'))
print('Composite:', r.get('composite_score'))
print('Elapsed:', r.get('elapsed_seconds'), 's')
print()

for name, bm in r.get('benchmarks', {}).items():
    c = bm.get('correct', 0)
    t = bm.get('total', 0)
    s = bm.get('score', 0)
    print(f"  {name:>12}: {c}/{t} = {s*100:.1f}%")

print()

# MMLU per-subject
from collections import Counter
mmlu_det = r.get('detailed_results', {}).get('MMLU', {}).get('details', [])
print(f"MMLU details count: {len(mmlu_det)}")
if mmlu_det:
    subj_c, subj_t = Counter(), Counter()
    for d in mmlu_det:
        s = d.get('subject', 'unknown')
        subj_t[s] += 1
        if d.get('correct'):
            subj_c[s] += 1
    print(f"MMLU subjects: {len(subj_t)}")
    for s in sorted(subj_t.keys()):
        c, t = subj_c[s], subj_t[s]
        pct = c / t * 100 if t else 0
        marker = " ✗" if pct < 30 else ""
        print(f"    {s:>42}: {c}/{t} = {pct:5.1f}%{marker}")

# ARC per-category
arc_det = r.get('detailed_results', {}).get('ARC', {}).get('details', [])
print(f"\nARC details count: {len(arc_det)}")
if arc_det:
    cat_c, cat_t = Counter(), Counter()
    for d in arc_det:
        c = d.get('category', 'unknown')
        cat_t[c] += 1
        if d.get('correct'):
            cat_c[c] += 1
    for c in sorted(cat_t.keys()):
        cr, t = cat_c[c], cat_t[c]
        pct = cr / t * 100 if t else 0
        print(f"    {c:>20}: {cr}/{t} = {pct:5.1f}%")

# MATH
math_det = r.get('detailed_results', {}).get('MATH', {}).get('details', [])
print(f"\nMATH details count: {len(math_det)}")
if math_det:
    dom_c, dom_t = Counter(), Counter()
    for d in math_det:
        dm = d.get('domain', 'unknown')
        dom_t[dm] += 1
        if d.get('correct'):
            dom_c[dm] += 1
    for dm in sorted(dom_t.keys()):
        c, t = dom_c[dm], dom_t[dm]
        pct = c / t * 100 if t else 0
        print(f"    {dm:>25}: {c}/{t} = {pct:5.1f}%")

# HumanEval
he_det = r.get('detailed_results', {}).get('HumanEval', {}).get('details', [])
print(f"\nHumanEval details count: {len(he_det)}")
if he_det:
    passed = sum(1 for d in he_det if d.get('passed'))
    print(f"  Passed: {passed}/{len(he_det)}")

# Top ARC failures
if arc_det:
    wrong_arc = [d for d in arc_det if not d.get('correct')]
    print(f"\n── ARC Failures ({len(wrong_arc)}) — first 25 ──")
    for d in wrong_arc[:25]:
        q = d.get('question', '')[:80]
        exp = d.get('expected', d.get('answer', '?'))
        got = d.get('predicted', d.get('chosen', '?'))
        print(f"  [{d.get('category','?')}] {q}")
        print(f"    Exp={exp}  Got={got}")

# Top MMLU failures
if mmlu_det:
    wrong_mmlu = [d for d in mmlu_det if not d.get('correct')]
    print(f"\n── MMLU Failures ({len(wrong_mmlu)}) — first 25 ──")
    for d in wrong_mmlu[:25]:
        q = d.get('question', '')[:80]
        exp = d.get('expected', d.get('answer', '?'))
        got = d.get('predicted', d.get('chosen', '?'))
        print(f"  [{d.get('subject','?')[:20]}] {q}")
        print(f"    Exp={exp}  Got={got}")
