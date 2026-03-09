#!/usr/bin/env python3
"""Quick timing test for the v5.0 language comprehension engine."""
import time, sys, os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

t0 = time.time()
from l104_asi.language_comprehension import LanguageComprehensionEngine
print(f"Import: {time.time()-t0:.2f}s")

t1 = time.time()
e = LanguageComprehensionEngine()
print(f"Create: {time.time()-t1:.2f}s")

t2 = time.time()
e.initialize()
print(f"Init:   {time.time()-t2:.2f}s")

# Triple extractor stats
print(f"Triples: {e.knowledge_base.triple_extractor.get_status()}")

# Quick test
t3 = time.time()
r = e.answer_mcq(
    "What is the powerhouse of the cell?",
    ["Nucleus", "Mitochondria", "Ribosome", "Golgi apparatus"]
)
print(f"Q1: {r['answer']} ({r['answer_text']}) [{time.time()-t3:.2f}s]")

t4 = time.time()
r = e.answer_mcq(
    "What is the chemical formula for water?",
    ["CO2", "NaCl", "H2O", "O2"]
)
print(f"Q2: {r['answer']} ({r['answer_text']}) [{time.time()-t4:.2f}s]")

t5 = time.time()
r = e.answer_mcq(
    "Who is known as the father of modern physics?",
    ["Newton", "Einstein", "Bohr", "Galileo"]
)
print(f"Q3: {r['answer']} ({r['answer_text']}) [{time.time()-t5:.2f}s]")

t6 = time.time()
r = e.answer_mcq(
    "What planet is closest to the Sun?",
    ["Venus", "Mars", "Mercury", "Earth"]
)
print(f"Q4: {r['answer']} ({r['answer_text']}) [{time.time()-t6:.2f}s]")

print(f"\nTotal: {time.time()-t0:.2f}s")
