#!/usr/bin/env python3
"""Diagnose ARC failures — identify which question types fail most."""
import os, sys, json, time
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import logging; logging.disable(logging.WARNING)

from l104_asi.benchmark_harness import _HuggingFaceFetcher

# Fetch ARC data
arc_data = _HuggingFaceFetcher.fetch_arc()
print(f"ARC questions: {len(arc_data)}")

# Init engine
import importlib
cr_mod = importlib.import_module('l104_asi.commonsense_reasoning')
CommonsenseReasoningEngine = cr_mod.CommonsenseReasoningEngine
eng = CommonsenseReasoningEngine()
eng.initialize()

correct = 0
total = 0
failures = []

for i, q in enumerate(arc_data[:100]):
    question = q.get("question", "")
    choices = q.get("choices", [])
    answer_idx = q.get("answer", 0)

    if not choices:
        continue

    total += 1
    try:
        result = eng.answer_mcq(question, choices)
        predicted_idx = result.get("selected_index", result.get("answer_index", -1))
        is_correct = (predicted_idx == answer_idx)
    except Exception as e:
        predicted_idx = -1
        is_correct = False

    if is_correct:
        correct += 1
    else:
        failures.append({
            "idx": i,
            "question": question[:100],
            "correct_answer": choices[answer_idx] if answer_idx < len(choices) else "?",
            "predicted": choices[predicted_idx] if 0 <= predicted_idx < len(choices) else "?",
            "choices": choices,
        })

print(f"\n=== ARC: {correct}/{total} correct ({correct/total*100:.1f}%) ===")

print("\n=== SAMPLE FAILURES ===")
for f in failures[:15]:
    print(f"\n  [{f['idx']}] Q: {f['question']}")
    print(f"  Choices: {f['choices']}")
    print(f"  Correct: {f['correct_answer']} | Predicted: {f['predicted']}")
