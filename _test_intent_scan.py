#!/usr/bin/env python3
"""Test v5.3 concepts + Section 5b intent-directed property scanning."""
import os, sys
os.environ['L104_QUIET'] = '1'
import logging; logging.disable(logging.WARNING)

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine

e = CommonsenseReasoningEngine()
e.initialize()
print(f"Concepts: {len(e.ontology.concepts)}, Rules: {len(e.causal.rules)}")

# Verify new concepts exist
for c in ['carbon_dioxide', 'organic_compound', 'circulatory_system', 'endocrine_system',
          'respiratory_system', 'electron', 'speed', 'acceleration', 'pie_chart']:
    found = c in e.ontology.concepts
    print(f"  {'✓' if found else '✗'} concept: {c}")

# Test key questions that were failing
tests = [
    ("Marshall learned that plants need a substance from the air to make their own food. Which substance do plants need from the air?",
     ["water vapor", "oxygen", "nitrogen", "carbon dioxide"], 3),
    ("The speed at which sound waves travel depends on the",
     ["size of the object making the sound.", "direction the sound is moving.",
      "type of material through which the sound travels.",
      "amount of energy that originally produced the sound."], 2),
    ("What allows a light bulb to give off light?",
     ["the light bulb generating heat energy",
      "the current flowing through the wire to the light bulb",
      "the shade around the light bulb absorbing light",
      "the batteries providing solar energy to the light bulb"], 1),
    ("Which elements make up most organic compounds?",
     ["iron, oxygen, nickel, copper", "carbon, hydrogen, oxygen, nitrogen",
      "aluminum, chlorine, iron, sulfur", "sodium, calcium, potassium, magnesium"], 1),
    ("The endocrine system relies on which body system for the transport of hormones to their target organs?",
     ["digestive", "circulatory", "respiratory", "immune"], 1),
]

correct = 0
for i, (q, choices, exp) in enumerate(tests):
    r = e.answer_mcq(q, choices)
    pred = r.get('selected_index', -1)
    ok = pred == exp
    if ok:
        correct += 1
    mark = '✓' if ok else '✗'
    scores = r.get('choice_scores', [])
    score_str = " | ".join(f"{s:.3f}" for s in scores) if scores else "N/A"
    print(f"\n{mark} Q{i}: Got={pred} Exp={exp}")
    print(f"  Q: {q[:80]}")
    print(f"  Choices: {[c[:30] for c in choices]}")
    print(f"  Scores: [{score_str}]")

print(f"\n{'='*60}")
print(f"Result: {correct}/{len(tests)} correct")
