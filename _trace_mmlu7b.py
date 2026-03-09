#!/usr/bin/env python3
"""Deep trace: patch MCQSolver._score_choice to log per-choice scores."""
import sys, os, logging, math
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
logging.disable(logging.CRITICAL)
sys.stderr = open(os.devnull, 'w')
sys.path.insert(0, '.')

from l104_asi.language_comprehension import LanguageComprehensionEngine
from l104_asi.language_comprehension.mcq_solver import MCQSolver
sys.stderr = sys.__stderr__

# Monkey-patch _score_choice to capture scores
_orig_score = MCQSolver._score_choice
_score_log = []

def _patched_score(self, question, choice, context_facts, knowledge_hits,
                    has_context=False):
    score = _orig_score(self, question, choice, context_facts, knowledge_hits,
                        has_context=has_context)
    _score_log.append((choice, score))
    return score

MCQSolver._score_choice = _patched_score

# Also patch solve() to capture _direct_answer_bonus
_orig_solve = MCQSolver.solve
_captured_dab = {}
_captured_cf = []
def _patched_solve(self, question, choices, subject=None):
    result = _orig_solve(self, question, choices, subject=subject)
    # Capture internals via a re-run of just the bonus computation
    # (hacky but avoids modifying production code)
    return result
MCQSolver.solve = _patched_solve

lce = LanguageComprehensionEngine()

questions = [
    {"q": "What is the derivative of x^2 with respect to x?", "c": ["x", "2x", "x^2", "2"], "a": 1, "s": "mathematics"},
    {"q": "Which data structure uses FIFO ordering?", "c": ["Stack", "Queue", "Tree", "Graph"], "a": 1, "s": "computer_science"},
    {"q": "What is the time complexity of binary search?", "c": ["O(1)", "O(log n)", "O(n)", "O(n log n)"], "a": 1, "s": "computer_science"},
    {"q": "What is the pH of a neutral solution at 25°C?", "c": ["0", "7", "14", "1"], "a": 1, "s": "chemistry"},
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
    result = lce.answer_mcq(q, choices, subject=subject)
    predicted = result.get("selected_index", -1)
    ok = "✓" if predicted == correct_idx else "✗"

    print(f"\n{'='*70}")
    print(f"{ok} Q: {q}")
    print(f"  Expected: [{correct_idx}] {choices[correct_idx]}")
    print(f"  Got:      [{predicted}] {choices[predicted] if 0<=predicted<len(choices) else '?'}")

    # Show all choice scores from the patch
    if _score_log:
        scores_sorted = sorted(_score_log, key=lambda x: x[1], reverse=True)
        print(f"  _score_choice() raw:")
        for choice_text, s in scores_sorted:
            marker = " ◀ CORRECT" if choice_text == choices[correct_idx] else ""
            marker2 = " ◀ PICKED" if choice_text == (choices[predicted] if 0<=predicted<len(choices) else "") else ""
            print(f"    {s:8.4f}  {choice_text}{marker}{marker2}")

    # Show final scores from result
    all_scores = result.get("all_scores", [])
    if all_scores:
        print(f"  Final all_scores:")
        for sc in sorted(all_scores, key=lambda x: x.get("score",0), reverse=True):
            print(f"    {sc.get('score',0):8.4f}  {sc.get('label','?')}")

    # Show context_facts_used count and knowledge hits
    print(f"  context_facts_used={result.get('context_facts_used',0)}, knowledge_hits={result.get('knowledge_hits',0)}")

    # Also check what KB facts are retrieved
    kb = lce.knowledge_base
    if hasattr(lce.mcq_solver, '_kb_retriever') and lce.mcq_solver._kb_retriever:
        retriever = lce.mcq_solver._kb_retriever
        facts = retriever.retrieve(q, top_k=10)
        if facts:
            print(f"  Top KB facts:")
            for fi, f in enumerate(facts[:5]):
                print(f"    [{fi}] {f[:100]}")

print("\nDone.")
