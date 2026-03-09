#!/usr/bin/env python3
"""MMLU Debug — trace direct_answer_bonus and other stage scores on embedded samples."""
import sys, os, re, math, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(__file__))

from l104_asi.language_comprehension import LanguageComprehensionEngine
from l104_asi.benchmark_harness import MMLU_SAMPLES

engine = LanguageComprehensionEngine()
engine.initialize()
solver = engine.mcq_solver

# Monkey-patch quantum out for cleaner analysis
solver._quantum_wave_collapse = lambda q, ch, cs, cf, kh: cs

# Test the 15 built-in samples
correct = 0
for i, s in enumerate(MMLU_SAMPLES):
    q = s["question"]
    choices = s["choices"]
    expected = s["answer"]
    subject = s.get("subject", "unknown")

    result = engine.answer_mcq(q, choices, subject=subject)
    predicted = result.get("selected_index", result.get("answer_index", -1))
    is_correct = predicted == expected

    status = "OK" if is_correct else "WRONG"
    if is_correct:
        correct += 1

    print(f"[{i+1}] {status} | {subject} | Q: {q[:70]}")
    print(f"  Predicted: {chr(65+predicted)}={choices[predicted][:40]}, Expected: {chr(65+expected)}={choices[expected][:40]}")
    for sc in result.get("all_scores", []):
        marker = " <<<" if sc['label'] == chr(65 + expected) else ""
        print(f"    {sc['label']}: {sc['score']:.4f}{marker}")
    print()

print(f"\nResult: {correct}/{len(MMLU_SAMPLES)} = {100*correct/len(MMLU_SAMPLES):.0f}%")

# Also test harder custom questions from different MMLU domains
hard_questions = [
    {"question": "What is the conclusion in a valid disjunctive syllogism when one disjunct is denied?",
     "choices": ["The major premise is true", "The other disjunct is true", "Both disjuncts are false", "The argument is invalid"],
     "answer": 1, "subject": "logical_fallacies"},
    {"question": "In Keynesian economics, a liquidity trap occurs when",
     "choices": ["interest rates are very high", "interest rates are at or near zero", "inflation is high", "unemployment is low"],
     "answer": 1, "subject": "high_school_macroeconomics"},
    {"question": "Which of the following is NOT a characteristic of a perfectly competitive market?",
     "choices": ["Many buyers and sellers", "Product differentiation", "Free entry and exit", "Perfect information"],
     "answer": 1, "subject": "high_school_microeconomics"},
    {"question": "A completely submerged object always displaces its own",
     "choices": ["weight of fluid", "volume of fluid", "density of fluid", "mass of fluid"],
     "answer": 1, "subject": "conceptual_physics"},
    {"question": "The term for a strategy of finding workable solutions that satisfice is",
     "choices": ["Optimizing", "Satisficing", "Maximizing", "Minimizing"],
     "answer": 1, "subject": "management"},
]

print("\n=== HARD Custom Questions ===")
for i, s in enumerate(hard_questions):
    q = s["question"]
    choices = s["choices"]
    expected = s["answer"]
    subject = s.get("subject", "unknown")

    result = engine.answer_mcq(q, choices, subject=subject)
    predicted = result.get("selected_index", result.get("answer_index", -1))
    is_correct = predicted == expected

    status = "OK" if is_correct else "WRONG"
    print(f"[H{i+1}] {status} | Q: {q[:70]}")
    print(f"  Predicted: {chr(65+predicted)}={choices[predicted][:40]}, Expected: {chr(65+expected)}={choices[expected][:40]}")
    for sc in result.get("all_scores", []):
        marker = " <<<" if sc['label'] == chr(65 + expected) else ""
        print(f"    {sc['label']}: {sc['score']:.4f}{marker}")
    print()
