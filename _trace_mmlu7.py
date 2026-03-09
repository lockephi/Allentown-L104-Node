#!/usr/bin/env python3
"""Trace the 7 missed MMLU questions to see score breakdowns."""
import sys, os, logging
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
logging.disable(logging.CRITICAL)
sys.stderr = open(os.devnull, 'w')
sys.path.insert(0, '.')

from l104_asi.language_comprehension import LanguageComprehensionEngine
sys.stderr = sys.__stderr__

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

    result = lce.answer_mcq(q, choices, subject=subject)
    predicted = result.get("selected_index", -1)
    ok = "✓" if predicted == correct_idx else "✗"

    print(f"\n{'='*70}")
    print(f"{ok} Q: {q}")
    print(f"  Expected: [{correct_idx}] {choices[correct_idx]}")
    print(f"  Got:      [{predicted}] {choices[predicted] if 0<=predicted<len(choices) else '?'}")
    print(f"  Confidence: {result.get('confidence', 0):.4f}")

    # Try to get score details from reasoning chain
    chain = result.get("reasoning_chain", [])
    if chain:
        for step in chain[-5:]:  # Last 5 reasoning steps
            print(f"  Chain: {step}")

    # If there's a scores breakdown
    scores = result.get("scores", result.get("choice_scores", {}))
    if scores:
        print(f"  Scores: {scores}")

print("\nDone.")
