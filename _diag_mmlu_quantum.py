#!/usr/bin/env python3
"""Quick test: MMLU with quantum disabled vs enabled."""
import sys, os, re, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(__file__))

from l104_asi.benchmark_harness import _HuggingFaceFetcher

# Fetch 100 questions
samples = _HuggingFaceFetcher.fetch_mmlu(max_questions=100)
print(f"Got {len(samples)} samples")

# Test with quantum enabled
from l104_asi.language_comprehension import LanguageComprehensionEngine

engine = LanguageComprehensionEngine()
engine.initialize()

# Monkey-patch the quantum to be a no-op
original_qwc = engine.mcq_solver._quantum_wave_collapse

def noop_quantum(q, ch, cs, cf, kh):
    return cs

# Run without quantum
engine.mcq_solver._quantum_wave_collapse = noop_quantum
correct_no_q = 0
for s in samples:
    r = engine.answer_mcq(s["question"], s["choices"], subject=s.get("subject"))
    if r.get("selected_index", r.get("answer_index", -1)) == s["answer"]:
        correct_no_q += 1

# Run with quantum
engine.mcq_solver._quantum_wave_collapse = original_qwc
correct_with_q = 0
for s in samples:
    r = engine.answer_mcq(s["question"], s["choices"], subject=s.get("subject"))
    if r.get("selected_index", r.get("answer_index", -1)) == s["answer"]:
        correct_with_q += 1

total = len(samples)
print(f"\nNo quantum:   {correct_no_q}/{total} = {100*correct_no_q/total:.1f}%")
print(f"With quantum: {correct_with_q}/{total} = {100*correct_with_q/total:.1f}%")
print(f"Quantum delta: {correct_with_q - correct_no_q:+d}")
