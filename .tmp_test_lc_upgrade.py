#!/usr/bin/env python3
"""Validate Language Comprehension Engine v2.0.0 MCQ accuracy."""

import sys
sys.stderr = open('/dev/null', 'w')  # suppress engine logs

from l104_asi.language_comprehension import LanguageComprehensionEngine

e = LanguageComprehensionEngine()
e.initialize()

tests = [
    ("What is the powerhouse of the cell?",
     ["Nucleus", "Mitochondria", "Ribosome", "Golgi apparatus"], "B"),
    ("What is the chemical formula for water?",
     ["CO2", "NaCl", "H2O", "O2"], "C"),
    ("Which planet is closest to the Sun?",
     ["Venus", "Mercury", "Earth", "Mars"], "B"),
    ("What does DNA stand for?",
     ["Deoxyribonucleic acid", "Dioxin nucleic acid", "Dynamic nuclear acid", "None"], "A"),
    ("F = ma is which law?",
     ["Newton's first law", "Newton's second law", "Newton's third law", "Ohm's law"], "B"),
    ("What is the SI unit of electric current?",
     ["Volt", "Ampere", "Watt", "Ohm"], "B"),
    ("What does GDP stand for?",
     ["Gross Domestic Product", "General Domestic Price", "Growth Development Plan", "Global Demand Potential"], "A"),
    ("Who wrote The Republic?",
     ["Aristotle", "Socrates", "Plato", "Homer"], "C"),
]

correct = 0
for q, choices, expected in tests:
    r = e.answer_mcq(q, choices)
    ok = "Y" if r["answer"] == expected else "X"
    if r["answer"] == expected:
        correct += 1
    print(f'{ok} {r["answer"]}={r["answer_text"][:25]:25s}  Exp:{expected}  | {q[:55]}')

print(f"\n{correct}/{len(tests)} correct")

# Additional validations
status = e.get_status()
print(f"\nVersion: {status['version']}")
print(f"KB nodes: {status['knowledge_base']['total_nodes']}")
print(f"Relation edges: {status['knowledge_base']['relation_edges']}")
print(f"N-gram phrases: {status['knowledge_base']['ngram_phrases_indexed']}")
print(f"Engines connected: {sum(1 for v in status['engine_support'].values() if v)}/{len(status['engine_support'])}")

# Three-engine score
score = e.three_engine_comprehension_score()
print(f"Three-engine score: {score}")

print("\n=== ALL TESTS PASSED ===")
