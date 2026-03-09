#!/usr/bin/env python3
"""Debug the 3 failing MCQs — trace KB hits, direct-answer extraction, cross-verification."""
import sys
from l104_asi.language_comprehension import LanguageComprehensionEngine

lce = LanguageComprehensionEngine()
solver = lce.mcq_solver

failures = [
    ("What is the largest organ in the human body?", ["Heart", "Liver", "Skin", "Brain"], "Skin"),
    ("What gas do plants absorb from the atmosphere?", ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"], "Carbon dioxide"),
    ("What is the speed of light in vacuum approximately?", ["300,000 km/s", "150,000 km/s", "450,000 km/s", "600,000 km/s"], "300,000 km/s"),
]

for q, choices, expected in failures:
    print(f"\n{'='*80}")
    print(f"Q: {q}")
    print(f"Expected: {expected}")
    r = solver.solve(q, choices)
    print(f"Got: {r['answer_text']} ({r['answer']})")
    print(f"Confidence: {r['confidence']}")
    print(f"Knowledge hits: {r['knowledge_hits']}, Context facts: {r['context_facts_used']}")
    print(f"All scores:")
    for s in r['all_scores']:
        print(f"  {s['label']}: {s['score']}")
    print(f"Calibration: {r['calibration']}")
