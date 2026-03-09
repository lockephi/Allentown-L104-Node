#!/usr/bin/env python3
"""MMLU Diagnostic v2 — sample 100 questions, analyze failure patterns."""
import sys, os, re, json, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(__file__))

from l104_asi.language_comprehension import LanguageComprehensionEngine
from l104_asi.benchmark_harness import _HuggingFaceFetcher

engine = LanguageComprehensionEngine()
engine.initialize()

# Fetch 100 MMLU questions (balanced across subjects)
print("Fetching 100 MMLU questions...")
samples = _HuggingFaceFetcher.fetch_mmlu(max_questions=100)
print(f"Got {len(samples)} samples\n")

correct = 0
wrong_examples = []
subj_stats = {}  # subject -> {correct, total}

for s in samples:
    q = s["question"]
    choices = s["choices"]
    expected = s["answer"]
    subject = s.get("subject", "unknown")

    result = engine.answer_mcq(q, choices, subject=subject)
    predicted = result.get("selected_index", result.get("answer_index", -1))
    is_correct = predicted == expected

    if subject not in subj_stats:
        subj_stats[subject] = {"correct": 0, "total": 0}
    subj_stats[subject]["total"] += 1

    if is_correct:
        correct += 1
        subj_stats[subject]["correct"] += 1
    else:
        wrong_examples.append({
            "q": q[:150],
            "subject": subject,
            "expected": expected,
            "predicted": predicted,
            "exp_choice": choices[expected][:80] if expected < len(choices) else "?",
            "pred_choice": choices[predicted][:80] if 0 <= predicted < len(choices) else "?",
            "scores": result.get("all_scores", []),
            "kb_hits": result.get("knowledge_hits", 0),
            "facts": result.get("context_facts_used", 0),
            "quantum": result.get("quantum", {}),
        })

total = len(samples)
print(f"=== MMLU RESULT: {correct}/{total} = {100*correct/max(total,1):.1f}% ===\n")

# Subject breakdown
print("--- Subject accuracy (worst first) ---")
for subj, stats in sorted(subj_stats.items(), key=lambda x: x[1]["correct"]/max(x[1]["total"],1)):
    pct = 100 * stats["correct"] / max(stats["total"], 1)
    print(f"  {subj:45s} {stats['correct']}/{stats['total']} = {pct:.0f}%")

# Answer position distribution (all wrong)
pos_counts = [0, 0, 0, 0]
for w in wrong_examples:
    p = w['predicted']
    if 0 <= p < 4:
        pos_counts[p] += 1
print(f"\nPredicted dist (wrong): A={pos_counts[0]} B={pos_counts[1]} C={pos_counts[2]} D={pos_counts[3]}")

# Signal stats
zero_kb = sum(1 for w in wrong_examples if w['kb_hits'] == 0)
low_facts = sum(1 for w in wrong_examples if w['facts'] < 3)
high_facts = sum(1 for w in wrong_examples if w['facts'] >= 10)
q_applied = sum(1 for w in wrong_examples if w['quantum'].get('wave_collapse_applied', False))
print(f"Zero KB: {zero_kb}/{len(wrong_examples)}")
print(f"Low facts (<3): {low_facts}/{len(wrong_examples)}")
print(f"High facts (≥10): {high_facts}/{len(wrong_examples)}")
print(f"Quantum applied: {q_applied}/{len(wrong_examples)}")

# Show 15 wrong examples
print(f"\n--- Wrong examples (first 15) ---")
for i, w in enumerate(wrong_examples[:15]):
    print(f"\n[{i+1}] {w['subject']}")
    print(f"  Q: {w['q']}")
    print(f"  Expected: [{w['expected']}] {w['exp_choice']}")
    print(f"  Got:      [{w['predicted']}] {w['pred_choice']}")
    print(f"  KB={w['kb_hits']} Facts={w['facts']}")
    for sc in w['scores']:
        marker = " <<<" if sc['label'] == chr(65 + w['expected']) else ""
        print(f"    {sc['label']}: {sc['score']:.4f}{marker}")
