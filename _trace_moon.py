#!/usr/bin/env python3
"""Deep trace of Moon phases scoring to find what's boosting B."""
import sys, time, os
# sys.stderr = open(os.devnull, 'w')

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
eng = CommonsenseReasoningEngine()
eng.initialize()

q = 'What causes the phases of the Moon?'
choices = [
    'The position of the Moon relative to the Earth and the Sun',
    'The shadow of the Earth on the Moon',
    'The distance of the Moon from the Sun',
    'The rotation of the Moon on its axis',
]

solver = eng.mcq_solver

# Get concept extraction
q_concepts = solver.ontology.extract_concepts(q)
print(f'Q concepts: {[c.name for c in q_concepts]}')

for i, ch in enumerate(choices):
    label = 'ABCD'[i]
    ch_concepts = solver.ontology.extract_concepts(ch)
    print(f'\n{label}: "{ch[:60]}..."')
    print(f'  Concepts: {[c.name for c in ch_concepts]}')
    for cc in ch_concepts:
        if cc.properties:
            print(f'    {cc.name}.properties = {dict(cc.properties)}')

# Now run solve with monkey-patched _score_choice to trace
original_score = solver._score_choice

def traced_score(question, q_words, choice, choice_idx, q_concepts, all_choices, causal_matches, **kw):
    result = original_score(question, q_words, choice, choice_idx, q_concepts, all_choices, causal_matches, **kw)
    label = 'ABCD'[choice_idx]
    print(f'\n  _score_choice({label})={result:.4f}')
    return result

solver._score_choice = traced_score
result = solver.solve(q, choices)
print(f'\nFinal answer: {result.get("answer")}')
print(f'Scores: {result.get("all_scores", result.get("scores", "?"))}')
# Check if choice_scores is in result
for k, v in result.items():
    if 'score' in k.lower():
        print(f'  {k}: {v}')
