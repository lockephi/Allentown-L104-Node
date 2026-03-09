#!/usr/bin/env python3
"""ARC-only benchmark."""
import os, sys, json
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import logging
logging.disable(logging.WARNING)

from l104_asi.benchmark_harness import _HuggingFaceFetcher
import importlib

print("Fetching ARC data...")
data = _HuggingFaceFetcher.fetch_arc(max_questions=200)
print(f"Fetched {len(data)} questions")

cr_mod = importlib.import_module('l104_asi.commonsense_reasoning')
engine = cr_mod.CommonsenseReasoningEngine()

correct = 0
total = 0
wrong_samples = []
for item in data:
    q = item['question']
    choices = item['choices']
    answer_idx = item['answer']
    result = engine.answer_mcq(q, choices)
    sel = result.get('selected_index', result.get('answer_index', -1))
    is_correct = (sel == answer_idx)
    if is_correct:
        correct += 1
    else:
        if len(wrong_samples) < 10:
            wrong_samples.append({
                'q': q[:80],
                'correct': choices[answer_idx] if answer_idx < len(choices) else '?',
                'picked': choices[sel] if 0 <= sel < len(choices) else '?',
            })
    total += 1

    if total % 50 == 0:
        print(f"  Progress: {total}/{len(data)} — {correct}/{total} = {correct/total*100:.1f}%")

print(f"\nARC RESULT: {correct}/{total} = {correct/total*100:.1f}%")
print("\nSample wrong answers:")
for w in wrong_samples[:5]:
    print(f"  Q: {w['q']}")
    print(f"  Correct: {w['correct']}, Picked: {w['picked']}")
    print()
