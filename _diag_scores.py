#!/usr/bin/env python3
"""Score breakdown for failing ARC questions"""
import sys
sys.path.insert(0, '.')
from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine

engine = CommonsenseReasoningEngine()
engine.initialize()
solver = engine.mcq_solver

tests = [
    ("At which temperature does water freeze?", ['0 degrees Celsius', '32 degrees Celsius', '100 degrees Celsius', '212 degrees Celsius'], 0),
    ("Plants use sunlight to make", ['oxygen', 'minerals', 'food.', 'water.'], 2),
    ("All living and nonliving material is composed of", ['cells', 'elements', 'water', 'oxygen'], 1),
    ("Where do plants get most of the energy they need to live and grow?", ['air', 'soil', 'water', 'sunlight'], 3),
    ("Which characteristic describes the texture of a kitten's fur?", ['gray', 'warm', 'long', 'soft'], 3),
]

for q, choices, expected in tests:
    result = solver.solve(q, choices)
    pred = result.get('answer_index', -1)
    all_scores = result.get('all_scores', {})
    status = '✓' if pred == expected else '✗'
    print(f"\n{status} Q: {q}")
    # First show what all_scores looks like
    for i, c in enumerate(choices):
        label = chr(65 + i)
        s = all_scores.get(label, 0)
        markers = []
        if i == expected:
            markers.append('EXP')
        if i == pred:
            markers.append('GOT')
        m = ' '.join(markers)
        if isinstance(s, dict):
            print(f"  {label}: {c:35s} {s}  {m}")
        else:
            print(f"  {label}: {c:35s} score={float(s):6.3f}  {m}")

    concepts = result.get('concepts_found', [])
    print(f"  Concepts: {concepts}")

    # Show calibration details
    cal = result.get('calibration', {})
    if isinstance(cal, dict):
        for k, v in cal.items():
            if k not in ('version',):
                print(f"  cal.{k}: {v}")

    # Show quantum details
    qd = result.get('quantum', {})
    if isinstance(qd, dict):
        for k, v in list(qd.items())[:5]:
            print(f"  quantum.{k}: {v}")
    print()
