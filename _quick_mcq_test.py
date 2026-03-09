#!/usr/bin/env python3
"""Quick single MCQ test to measure timing."""
import time, os, sys
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import logging
logging.disable(logging.WARNING)

print("Loading engine...")
t0 = time.time()
from l104_asi.language_comprehension import LanguageComprehensionEngine
e = LanguageComprehensionEngine()
print(f"Engine loaded in {time.time()-t0:.1f}s, VERSION={e.VERSION}")

print("Running 1 MCQ...")
t1 = time.time()
r = e.answer_mcq("What is the capital of France?", ["London", "Paris", "Berlin", "Madrid"])
t2 = time.time()
print(f"MCQ time: {t2-t1:.2f}s")
print(f"Selected idx={r.get('selected_index')}, answer={r.get('selected_answer')}")
sys.exit(0)
