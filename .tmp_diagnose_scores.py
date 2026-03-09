#!/usr/bin/env python3
"""Diagnose why MMLU scores are flat / biased toward index 0."""
import sys, os, logging
logging.disable(logging.WARNING)

from l104_asi.language_comprehension import LanguageComprehensionEngine

lce = LanguageComprehensionEngine()
lce.initialize()

# Test with known MMLU questions from the failures
test_cases = [
    {
        "q": "The longest bone in the human body is the:",
        "c": ["Humerus", "Femur", "Tibia", "Fibula"],
        "expected": 1,  # B = Femur
    },
    {
        "q": "Which planet is known as the Red Planet?",
        "c": ["Venus", "Mars", "Jupiter", "Saturn"],
        "expected": 1,  # B = Mars
    },
    {
        "q": "Statement 1 | A factor group of a non-Abelian group is non-Abelian. Statement 2 | Every group of order p^2 where p is prime is Abelian.",
        "c": ["True, True", "True, False", "False, True", "False, False"],
        "expected": 2,  # C = False, True
    },
    {
        "q": "Which of the following is NOT a function of the liver?",
        "c": ["Bile production", "Insulin production", "Detoxification", "Glycogen storage"],
        "expected": 1,  # B = Insulin production
    },
    {
        "q": "The speed of light in vacuum is approximately:",
        "c": ["3 × 10^6 m/s", "3 × 10^8 m/s", "3 × 10^10 m/s", "3 × 10^12 m/s"],
        "expected": 1,  # B
    },
]

correct = 0
for i, tc in enumerate(test_cases):
    r = lce.answer_mcq(tc["q"], tc["c"])
    predicted = r["answer_index"]
    is_correct = predicted == tc["expected"]
    correct += is_correct
    mark = "✓" if is_correct else "✗"

    print(f"\n{mark} Q{i+1}: {tc['q'][:70]}...")
    print(f"  Predicted: {r['answer']} (idx={predicted}), Expected: {chr(65+tc['expected'])} (idx={tc['expected']})")
    print(f"  Confidence: {r['confidence']}")

    # Show all scores
    all_scores = r.get("all_scores", [])
    for s in all_scores:
        marker = " ← PREDICTED" if s.get("label") == r["answer"] else ""
        exp_marker = " ← EXPECTED" if s.get("label") == chr(65+tc["expected"]) else ""
        print(f"    {s['label']}: {s['score']:.4f}{marker}{exp_marker}")

    print(f"  KB hits: {r.get('knowledge_hits', 0)}, Facts used: {r.get('context_facts_used', 0)}")
    cal = r.get("calibration", {})
    print(f"  Calibration: {cal}")

print(f"\nOverall: {correct}/{len(test_cases)}")
