#!/usr/bin/env python3
"""Profile MCQ to find the bottleneck."""
import time, os, sys, cProfile, pstats, io
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import logging
logging.disable(logging.WARNING)

print("Loading engine...")
t0 = time.time()
from l104_asi.language_comprehension import LanguageComprehensionEngine
e = LanguageComprehensionEngine()
print(f"Engine loaded in {time.time()-t0:.1f}s, VERSION={e.VERSION}")

print("Profiling 1 MCQ...")
pr = cProfile.Profile()
pr.enable()
t1 = time.time()
r = e.answer_mcq("What is the capital of France?", ["London", "Paris", "Berlin", "Madrid"])
t2 = time.time()
pr.disable()
print(f"MCQ time: {t2-t1:.2f}s")
print(f"Selected idx={r.get('selected_index')}")

# Print top 30 cumtime functions
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(40)
print(s.getvalue())
