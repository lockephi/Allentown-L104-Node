#!/usr/bin/env python3
"""MMLU-only benchmark."""
import os, sys, json
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import logging
logging.disable(logging.WARNING)

from l104_asi.benchmark_harness import _HuggingFaceFetcher
import importlib

print("Fetching MMLU data...")
data = _HuggingFaceFetcher.fetch_mmlu(max_questions=200)
print(f"Fetched {len(data)} questions")

if not data:
    print("No MMLU data fetched. Exiting.")
    sys.exit(1)

lc_mod = importlib.import_module('l104_asi.language_comprehension')
engine = lc_mod.LanguageComprehensionEngine()

correct = 0
total = 0
subject_stats = {}
for item in data:
    q = item['question']
    choices = item['choices']
    answer_idx = item['answer']
    subj = item.get('subject', 'unknown')
    result = engine.answer_mcq(q, choices)
    sel = result.get('selected_index', result.get('answer_index', -1))
    is_correct = (sel == answer_idx)
    if is_correct:
        correct += 1
    total += 1

    if subj not in subject_stats:
        subject_stats[subj] = [0, 0]
    subject_stats[subj][1] += 1
    if is_correct:
        subject_stats[subj][0] += 1

    if total % 50 == 0:
        print(f"  Progress: {total}/{len(data)} — {correct}/{total} = {correct/total*100:.1f}%")

print(f"\nMMLU RESULT: {correct}/{total} = {correct/total*100:.1f}%")
print("\nBy subject:")
for subj, (c, t) in sorted(subject_stats.items()):
    print(f"  {subj:30s}: {c}/{t} = {c/t*100:.0f}%")
