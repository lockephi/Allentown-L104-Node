#!/usr/bin/env python3
"""Targeted diagnostic: run 20 MMLU questions with quantum wave collapse logging."""
import sys, os, json, time, copy

# Monkey-patch the quantum collapse to log what it does
original_method = None
collapse_log = []

def patched_collapse(self, question, choices, choice_scores, context_facts, knowledge_hits):
    """Wrapper that logs inputs/outputs of quantum wave collapse."""
    before = copy.deepcopy(choice_scores)
    result = original_method(self, question, choices, choice_scores, context_facts, knowledge_hits)

    entry = {
        'question': question[:80],
        'before': [(cs['choice'][:25], round(cs['score'], 4)) for cs in before],
        'after': [(cs['choice'][:25], round(cs['score'], 4), round(cs.get('quantum_prob', 0), 4)) for cs in result],
        'kd_available': len(context_facts),
        'node_hits': len(knowledge_hits),
    }
    collapse_log.append(entry)
    return result

# Import the solver
from l104_asi.language_comprehension import MCQSolver, MMLUKnowledgeBase

# Monkey-patch
original_method = MCQSolver._quantum_wave_collapse
MCQSolver._quantum_wave_collapse = patched_collapse

# Now run some questions
kb = MMLUKnowledgeBase()
solver = MCQSolver(kb)

# Fetch a few MMLU questions
from l104_asi.benchmark_harness import _HuggingFaceFetcher

print("Fetching 20 MMLU questions...")
questions_raw = _HuggingFaceFetcher.fetch_mmlu(max_questions=20)
# Normalize to expected format
questions = []
for q in questions_raw:
    ans_idx = q.get('answer', 0)
    choices = q.get('choices', [])
    correct_letter = chr(65 + ans_idx) if isinstance(ans_idx, int) and ans_idx < len(choices) else str(ans_idx)
    questions.append({
        'question': q['question'],
        'choices': choices,
        'correct': correct_letter,
        'subject': q.get('subject', ''),
    })
print(f"Got {len(questions)} questions")

correct = 0
for i, q in enumerate(questions):
    result = solver.solve(q['question'], q['choices'])
    predicted = result.get('answer', '')
    is_correct = predicted == q['correct']
    if is_correct:
        correct += 1

    mark = 'Y' if is_correct else 'N'
    print(f"\n[{mark}] Q{i+1}: {q['question'][:60]}...")
    print(f"    Correct: {q['correct']}")
    print(f"    Predicted: {predicted}")

    # Show quantum log for this question
    if collapse_log:
        entry = collapse_log[-1]
        print(f"    KB facts: {entry['kd_available']}, Node hits: {entry['node_hits']}")
        print(f"    Before: {entry['before']}")
        print(f"    After:  {entry['after']}")

print(f"\n{'='*60}")
print(f"Score: {correct}/{len(questions)} = {100*correct/len(questions):.1f}%")
print(f"Quantum collapses: {len(collapse_log)}")

# Summary statistics
if collapse_log:
    avg_facts = sum(e['kd_available'] for e in collapse_log) / len(collapse_log)
    avg_nodes = sum(e['node_hits'] for e in collapse_log) / len(collapse_log)
    # Count how many times quantum changed the top choice
    changed = 0
    for e in collapse_log:
        before_top = e['before'][0][0] if e['before'] else None
        after_top = e['after'][0][0] if e['after'] else None
        if before_top != after_top:
            changed += 1
    print(f"Avg KB facts: {avg_facts:.1f}")
    print(f"Avg node hits: {avg_nodes:.1f}")
    print(f"Top choice changed by quantum: {changed}/{len(collapse_log)}")
