#!/usr/bin/env python3
"""Directly trace scoring for regression question idx 87 (hybrid car)."""
import sys
sys.path.insert(0, '.')

from datasets import load_dataset
ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")

# Get the hybrid car question
row = ds[87]
q = row['question']
choices = row['choices']['text']
labels = row['choices']['label']
answer_key = row['answerKey']

print(f"Question: {q}")
print(f"Correct: {answer_key}")
for l, c in zip(labels, choices):
    print(f"  {l}: {c}")

# Import and run engine (same as _diag_broader.py)
from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
engine = CommonsenseReasoningEngine()

result = engine.answer_mcq(q, choices)

print(f"\n=== SOLVER OUTPUT ===")
pred_idx = result.get('selected_index', 0)
pred_label = labels[pred_idx] if pred_idx < len(labels) else '?'
print(f"Selected: {pred_label} ({choices[pred_idx]})")
print(f"Expected: {answer_key} ({choices[labels.index(answer_key)]})")
print(f"Correct: {pred_label == answer_key}")
print(f"Scores: {result.get('all_scores', {})}")
print(f"Confidence: {result.get('confidence', 'N/A')}")

# Also test idx 94
print("\n" + "="*60)
row = ds[94]
q = row['question']
choices = row['choices']['text']
labels = row['choices']['label']
answer_key = row['answerKey']

print(f"Question: {q}")
print(f"Correct: {answer_key}")
for l, c in zip(labels, choices):
    print(f"  {l}: {c}")

result = engine.answer_mcq(q, choices)
print(f"\n=== SOLVER OUTPUT ===")
pred_idx = result.get('selected_index', 0)
pred_label = labels[pred_idx] if pred_idx < len(labels) else '?'
print(f"Selected: {pred_label} ({choices[pred_idx]})")
print(f"Expected: {answer_key} ({choices[labels.index(answer_key)]})")
print(f"Correct: {pred_label == answer_key}")
print(f"Scores: {result.get('all_scores', {})}")
