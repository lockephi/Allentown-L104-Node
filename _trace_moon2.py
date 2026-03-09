#!/usr/bin/env python3
"""Deep trace of Moon phases: inject print statements into solve pipeline."""
import sys, time, os, re

t0 = time.time()
print('Loading...', flush=True)
from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
eng = CommonsenseReasoningEngine()
eng.initialize()
print(f'Loaded in {time.time()-t0:.1f}s', flush=True)

q = 'What causes the phases of the Moon?'
choices = [
    'The position of the Moon relative to the Earth and the Sun',
    'The shadow of the Earth on the Moon',
    'The distance of the Moon from the Sun',
    'The rotation of the Moon on its axis',
]

solver = eng.mcq_solver

# Monkey-patch solve to trace pipeline stages
import math as _cac_math

_orig_solve = solver.solve.__func__ if hasattr(solver.solve, '__func__') else None

# Instead, let's just trace key metrics by calling subsystems directly
q_lower = q.lower()
concepts = solver._extract_concepts(q_lower)
print(f'\nQ concepts from question: {concepts}')

_q_concepts = set(concepts)
for ci, ch in enumerate(choices):
    ch_concepts = solver._extract_concepts(ch.lower())
    added = 0
    for c in ch_concepts:
        if c not in concepts and added < 3:
            concepts.append(c)
            added += 1
print(f'All concepts (incl choices): {concepts}')

causal_matches = solver.causal.query(q_lower, top_k=8)
print(f'\nCausal matches ({len(causal_matches)}):')
for rule, score in causal_matches:
    print(f'  score={score:.3f} cond="{rule.condition[:40]}" eff="{rule.effect[:40]}" kw={rule.keywords}')

max_causal = max((s for _, s in causal_matches), default=0.0)
print(f'Max causal score: {max_causal:.4f}')
print(f'Compression threshold: 0.7')
print(f'Compression will trigger: {max_causal >= 0.7}')

# Call _score_choice for each choice
print('\n--- RAW _score_choice ---')
raw_scores = []
for i, ch in enumerate(choices):
    s = solver._score_choice(q, ch, concepts, causal_matches)
    raw_scores.append(s)
    label = 'ABCD'[i]
    print(f'  {label}: raw_score={s:.4f}  "{ch[:50]}"')

# Simulate compression
print('\n--- AFTER COMPRESSION ---')
compressed = list(raw_scores)
if max_causal >= 0.7 and len(choices) >= 2:
    _max_s = max(compressed)
    _min_s = min(compressed)
    if _max_s > 0 and _max_s - _min_s > 0.5:
        compressed = [_cac_math.sqrt(s) if s > 0 else s for s in compressed]
        print('  COMPRESSION APPLIED (sqrt)')
    else:
        print('  No compression (range too narrow)')
else:
    print(f'  No compression (max_causal={max_causal:.4f} < 0.7)')

for i, s in enumerate(compressed):
    print(f'  {"ABCD"[i]}: {s:.4f}')

# Simulate length normalization
print('\n--- AFTER LENGTH NORM ---')
wcs = [len(ch.split()) for ch in choices]
avg_wc = sum(wcs) / len(wcs)
print(f'  Word counts: {wcs}, avg={avg_wc:.1f}')
normed = []
for i, s in enumerate(compressed):
    wc = wcs[i]
    if wc > 1 and avg_wc > 0:
        factor = max(0.70, min(1.30, (avg_wc / wc) ** 0.55))
        normed.append(s * factor)
        print(f'  {"ABCD"[i]}: {s:.4f} * {factor:.4f} = {s*factor:.4f}  (wc={wc})')
    else:
        normed.append(s)
        print(f'  {"ABCD"[i]}: {s:.4f} (no norm, wc={wc})')

# Show who wins at each stage
print('\n--- WINNER AT EACH STAGE ---')
for stage, scores_list in [('raw', raw_scores), ('compressed', compressed), ('normed', normed)]:
    winner_idx = max(range(4), key=lambda i: scores_list[i])
    print(f'  {stage}: {"ABCD"[winner_idx]} = {scores_list[winner_idx]:.4f}')

# Now run actual solve
print('\n--- ACTUAL SOLVE RESULT ---')
result = solver.solve(q, choices)
print(f'  Answer: {result.get("answer")}')
if 'all_scores' in result:
    print(f'  All scores: {result["all_scores"]}')
