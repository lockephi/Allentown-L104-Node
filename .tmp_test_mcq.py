#!/usr/bin/env python3
"""Quick MCQ solver test."""
import json, sys, os
os.environ['LOG_LEVEL'] = 'ERROR'

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
e = CommonsenseReasoningEngine()
e.initialize()

questions = [
    ("What force causes objects to fall toward the ground?",
     ['friction', 'magnetism', 'gravity', 'electricity'], 2),
    ("Which is NOT a state of matter?",
     ['solid', 'liquid', 'energy', 'gas'], 2),
    ("A student wants to know how much water a container holds. Which tool should the student use?",
     ['ruler', 'graduated cylinder', 'thermometer', 'balance'], 1),
    ("What is the first step of the scientific method?",
     ['experiment', 'hypothesis', 'observation', 'conclusion'], 2),
    ("Which of the following is a conductor of electricity?",
     ['rubber', 'wood', 'copper', 'glass'], 2),
    ("What causes the seasons on Earth?",
     ['distance from the sun', 'the tilt of Earth on its axis', 'the speed of Earth orbit', 'the size of the sun'], 1),
    ("Which part of the cell is known as the powerhouse?",
     ['nucleus', 'cell wall', 'mitochondria', 'chloroplast'], 2),
    ("What is the process by which plants make their own food?",
     ['respiration', 'photosynthesis', 'fermentation', 'digestion'], 1),
    ("Tropical plant fossils have been found on a cold island. What does this suggest?",
     ['The plants moved there recently', 'The island once had a warmer climate', 'The fossils are fake', 'Cold weather grows tropical plants'], 1),
    ("If Earth rotated faster, what would happen to the length of a day?",
     ['It would stay the same', 'It would get longer', 'It would get shorter', 'Days would disappear'], 2),
]

correct = 0
total = len(questions)
for i, (q, choices, expected) in enumerate(questions):
    r = e.answer_mcq(q, choices)
    got = r['answer_index']
    ok = "OK" if got == expected else "WRONG"
    if got == expected:
        correct += 1
    print(f"Q{i+1} [{ok}] expected={expected} got={got} conf={r['confidence']:.3f}")
    print(f"   Scores: {r['all_scores']}")
    print(f"   Concepts: {r['concepts_found'][:8]}")
    print()

print(f"\n=== RESULT: {correct}/{total} = {correct/total*100:.1f}% ===")
