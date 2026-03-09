#!/usr/bin/env python3
"""Debug the 3 remaining MMLU misses with full scoring trace."""
import sys, os, re, logging
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
logging.disable(logging.CRITICAL)
sys.stderr = open(os.devnull, 'w')
sys.path.insert(0, '.')

from l104_asi.language_comprehension import LanguageComprehensionEngine
from l104_asi.language_comprehension.mcq_solver import MCQSolver
sys.stderr = sys.__stderr__

# Patch _score_choice to capture per-choice scores
_orig_score = MCQSolver._score_choice
_score_log = []
def _patched_score(self, question, choice, context_facts, knowledge_hits, has_context=False):
    score = _orig_score(self, question, choice, context_facts, knowledge_hits, has_context=has_context)
    _score_log.append((choice, score))
    return score
MCQSolver._score_choice = _patched_score

# Patch solve to capture internals
_orig_solve = MCQSolver.solve
_solve_internals = {}
def _patched_solve(self, question, choices, subject=None):
    result = _orig_solve(self, question, choices, subject=subject)
    # Extract the all_scores from result
    _solve_internals['all_scores'] = result.get('all_scores', [])
    _solve_internals['result'] = result
    return result
MCQSolver.solve = _patched_solve

lce = LanguageComprehensionEngine()

questions = [
    {"q": "The term 'cogito ergo sum' is attributed to", "c": ["Kant", "Descartes", "Hume", "Locke"], "a": 1, "s": "philosophy"},
    {"q": "Maslow's hierarchy of needs places which need at the base?", "c": ["Safety", "Physiological", "Love", "Esteem"], "a": 1, "s": "psychology"},
    {"q": "The scientific method begins with", "c": ["hypothesis", "observation", "experiment", "conclusion"], "a": 1, "s": "science"},
]

for qi, qdata in enumerate(questions):
    q = qdata["q"]
    choices = qdata["c"]
    correct_idx = qdata["a"]
    subject = qdata["s"]

    _score_log.clear()
    _solve_internals.clear()
    result = lce.answer_mcq(q, choices, subject=subject)
    predicted = result.get("selected_index", -1)
    ok = "✓" if predicted == correct_idx else "✗"

    print(f"\n{'='*70}")
    print(f"{ok} Q: {q}")
    print(f"  Expected: [{correct_idx}] {choices[correct_idx]}")
    print(f"  Got:      [{predicted}] {choices[predicted] if 0<=predicted<len(choices) else '?'}")

    # Show _score_choice raw scores
    print(f"\n  _score_choice raw scores:")
    for ch, s in sorted(_score_log, key=lambda x: x[1], reverse=True):
        mark = " ◀ CORRECT" if ch == choices[correct_idx] else ""
        mark2 = " ◀ PICKED" if ch == (choices[predicted] if 0<=predicted<len(choices) else "") else ""
        print(f"    {s:8.4f}  {ch}{mark}{mark2}")

    # Show final all_scores from result
    print(f"\n  Final all_scores (after all post-processing):")
    for item in _solve_internals.get('all_scores', []):
        ch_label = item.get('label', '?')
        ch_score = item.get('score', 0)
        print(f"    {ch_label}: {ch_score:.4f}")

    # Show confidence and calibration
    print(f"  Confidence: {result.get('confidence', 0):.4f}")
    calib = result.get('calibration', {})
    print(f"  Calibration: early_exit={calib}")

print("\nDone.")
