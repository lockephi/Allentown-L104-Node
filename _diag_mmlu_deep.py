#!/usr/bin/env python3
"""MMLU Deep Debug — trace individual scoring stages for sample failures."""
import sys, os, re, math, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(__file__))

from l104_asi.benchmark_harness import _HuggingFaceFetcher
from l104_asi.language_comprehension import LanguageComprehensionEngine, MCQSolver

engine = LanguageComprehensionEngine()
engine.initialize()
solver = engine.mcq_solver

# Instrument _score_choice to get per-stage breakdowns
original_score = solver._score_choice

def instrumented_score(question, choice, context_facts, knowledge_hits, has_context=False):
    """Call original and also compute per-stage scores"""
    total = original_score(question, choice, context_facts, knowledge_hits, has_context=has_context)

    # Compute stage breakdowns manually
    choice_lower = choice.lower().strip()
    choice_words = set(re.findall(r'\w+', choice_lower))
    q_lower = question.lower()
    q_words = set(re.findall(r'\w+', q_lower))
    q_content_words = {w for w in q_words if len(w) > 3}

    # Check if choice text appears in question (self-referential)
    choice_in_question = choice_lower in q_lower or any(
        len(w) > 4 and w in q_lower for w in choice_words
    )

    return total

# Fetch 50 questions
samples = _HuggingFaceFetcher.fetch_mmlu(max_questions=50)
print(f"Got {len(samples)} samples\n")

correct = 0
wrong = 0

# Monkey-patch quantum out for cleaner analysis
solver._quantum_wave_collapse = lambda q, ch, cs, cf, kh: cs

for s in samples:
    q = s["question"]
    choices = s["choices"]
    expected = s["answer"]
    subject = s.get("subject", "unknown")

    result = engine.answer_mcq(q, choices, subject=subject)
    predicted = result.get("selected_index", result.get("answer_index", -1))

    if predicted == expected:
        correct += 1
        continue

    wrong += 1
    if wrong > 15:
        continue

    # Show detail
    print(f"{'='*80}")
    print(f"[{wrong}] {subject}")
    print(f"Q: {q[:150]}")
    for ci, ch in enumerate(choices):
        marker = " *" if ci == expected else ""
        print(f"  {chr(65+ci)}: {ch[:80]}{marker}")

    # Check explicit patterns
    q_lower = q.lower()
    for ci, ch in enumerate(choices):
        ch_lower = ch.lower().strip()
        # Check if choice appears in question
        if len(ch_lower) > 10 and ch_lower[:20] in q_lower:
            print(f"  ** Choice {chr(65+ci)} text appears in question! (self-ref)")

    print(f"  Predicted: {chr(65+predicted)}, Expected: {chr(65+expected)}")
    print(f"  KB hits: {result.get('knowledge_hits',0)}, Facts: {result.get('context_facts_used',0)}")
    for sc in result.get("all_scores", []):
        marker = " <<<" if sc['label'] == chr(65 + expected) else ""
        print(f"    {sc['label']}: {sc['score']:.4f}{marker}")

total = len(samples)
print(f"\n{'='*80}")
print(f"RESULT: {correct}/{total} = {100*correct/max(total,1):.1f}%")
print(f"Wrong: {wrong}")
