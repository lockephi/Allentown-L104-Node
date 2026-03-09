#!/usr/bin/env python3
"""Quick MMLU test."""
import sys
from l104_asi.language_comprehension import LanguageComprehensionEngine
lce = LanguageComprehensionEngine()
try:
    r = lce.answer_mcq('What is the derivative of x^2?', ['2x', 'x^2', 'x', '2'], subject='unknown')
    print(f'OK: answer={r.get("answer")} idx={r.get("selected_index")}', flush=True)
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}', flush=True)
