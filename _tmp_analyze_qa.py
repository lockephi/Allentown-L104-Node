#!/usr/bin/env python3
"""Analyze MMLU and ARC question distributions and sample failures."""
import os, sys, json
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import logging; logging.disable(logging.WARNING)
sys.path.insert(0, '.')
import importlib
bh = importlib.import_module('l104_asi.benchmark_harness')

print("=== MMLU Subject Distribution ===", flush=True)
mmlu = bh._HuggingFaceFetcher.fetch_mmlu(max_questions=500)
from collections import Counter
subjects = Counter(q.get('subject', 'unknown') for q in mmlu)
print(f"Total: {len(mmlu)} questions, {len(subjects)} subjects")
for subj, cnt in subjects.most_common(20):
    print(f"  {subj}: {cnt}")

# Run MMLU engine on a sample to see what it gets wrong
lc_mod = importlib.import_module('l104_asi.language_comprehension')
engine = lc_mod.LanguageComprehensionEngine()

correct = 0
wrong_samples = []
for i, item in enumerate(mmlu[:100]):
    q = item['question']
    choices = item['choices']
    answer_idx = item['answer']
    result = engine.answer_mcq(q, choices)
    sel = result.get('selected_index', result.get('answer_index', -1))
    if sel == answer_idx:
        correct += 1
    else:
        wrong_samples.append({
            'q': q[:100],
            'subj': item.get('subject', '?'),
            'correct': choices[answer_idx][:50] if answer_idx < len(choices) else '?',
            'selected': choices[sel][:50] if 0 <= sel < len(choices) else '?',
            'conf': result.get('confidence', 0)
        })

print(f"\nMMLU Sample: {correct}/100 = {correct}%")
print(f"\nSample wrong answers (first 10):")
for s in wrong_samples[:10]:
    print(f"  [{s['subj']}] Q: {s['q']}")
    print(f"    Correct: {s['correct']} | Selected: {s['selected']} | Conf: {s['conf']:.3f}")
    print()

# ARC analysis
print("\n=== ARC Question Analysis ===", flush=True)
arc = bh._HuggingFaceFetcher.fetch_arc(max_questions=200)
print(f"Total: {len(arc)} questions")
print(f"Avg choices: {sum(len(q['choices']) for q in arc)/len(arc):.1f}")

cr_mod = importlib.import_module('l104_asi.commonsense_reasoning')
arc_engine = cr_mod.CommonsenseReasoningEngine()

arc_correct = 0
arc_wrong = []
for item in arc[:100]:
    q = item['question']
    choices = item['choices']
    answer_idx = item['answer']
    result = arc_engine.answer_mcq(q, choices)
    sel = result.get('selected_index', result.get('answer_index', -1))
    if sel == answer_idx:
        arc_correct += 1
    else:
        arc_wrong.append({
            'q': q[:120],
            'correct': choices[answer_idx][:50] if answer_idx < len(choices) else '?',
            'selected': choices[sel][:50] if 0 <= sel < len(choices) else '?'
        })

print(f"ARC Sample: {arc_correct}/100 = {arc_correct}%")
print(f"\nSample wrong ARC (first 10):")
for s in arc_wrong[:10]:
    print(f"  Q: {s['q']}")
    print(f"    Correct: {s['correct']} | Selected: {s['selected']}")
    print()
