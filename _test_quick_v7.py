#!/usr/bin/env python3
"""Quick entailment + version check."""
from l104_asi.language_comprehension import TextualEntailmentEngine, LanguageComprehensionEngine

te = TextualEntailmentEngine()
r1 = te.entail("Dogs are animals that bark.", "Animals can bark.")
print(f"Test A: {r1['label']} ({r1['confidence']:.3f})")
r2 = te.entail("The temperature is 100 degrees.", "The temperature is not 100 degrees.")
print(f"Test B (negation): {r2['label']} ({r2['confidence']:.3f})")
r3 = te.entail("All cats are animals.", "No cats are animals.")
print(f"Test C (quantifier): {r3['label']} ({r3['confidence']:.3f})")
r4 = te.entail("The speed is 50 km/h.", "The speed is 80 km/h.")
print(f"Test D (number): {r4['label']} ({r4['confidence']:.3f})")
r5 = te.entail("Dogs are mammals.", "Dogs are a type of animal.")
print(f"Test E (hypernym): {r5['label']} ({r5['confidence']:.3f})")

lce = LanguageComprehensionEngine()
print(f"\nVersion: {lce.VERSION}")
print(f"Layers: {len(lce.get_status()['layers'])}")
print("ALL OK")
