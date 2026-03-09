#!/usr/bin/env python3
"""Diagnose a single ARC question."""
import os, logging
os.chdir(os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.WARNING)
os.environ["L104_QUIET"] = "1"

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
engine = CommonsenseReasoningEngine()

# Test cases to diagnose
tests = [
    {
        "q": "Which device produces mechanical energy?",
        "choices": ["a toaster", "an electric motor", "a light bulb", "a solar panel"],
        "answer": 1
    },
    {
        "q": "If you place a thermometer into a glass of boiling water, what temperature should the thermometer read?",
        "choices": ["32°C", "100°C", "200°C", "212°C"],
        "answer": 1
    },
    {
        "q": "Plants use sunlight to make",
        "choices": ["minerals.", "seeds.", "food.", "carbon dioxide."],
        "answer": 2
    },
    {
        "q": "All living and nonliving material is composed of",
        "choices": ["water", "minerals", "elements", "cells"],
        "answer": 2
    },
]

for t in tests:
    q = t["q"]
    choices = t["choices"]
    exp = t["answer"]
    result = engine.answer_mcq(q, choices)
    pred = result.get('selected_index', result.get('answer_index', 0))
    ok = "✓" if pred == exp else "✗"
    print(f"\n{ok} Q: {q}")
    print(f"  Expected: {choices[exp]} | Got: {choices[pred]}")
    print(f"  All scores: {result.get('all_scores', {})}")
    # Show concepts found
    concepts = engine.mcq_solver._extract_concepts(q.lower())
    print(f"  Concepts: {concepts}")
    # Show per-choice ontology scores
    q_lower = q.lower()
    for i, ch in enumerate(choices):
        ch_concepts = engine.mcq_solver._extract_concepts(ch.lower())
        onto_score = engine.mcq_solver._score_choice(q, ch, concepts, [])
        print(f"    [{chr(65+i)}] '{ch}' → onto={onto_score:.4f} ch_concepts={ch_concepts}")
