#!/usr/bin/env python3
"""Deep trace of 3 failing MCQs — debug score_choice, context_facts, direct_answer_bonus."""
import sys, re
from l104_asi.language_comprehension import LanguageComprehensionEngine

lce = LanguageComprehensionEngine()
solver = lce.mcq_solver

failures = [
    ("What is the largest organ in the human body?", ["Heart", "Liver", "Skin", "Brain"], "Skin"),
    ("What gas do plants absorb from the atmosphere?", ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"], "Carbon dioxide"),
    ("What is the speed of light in vacuum approximately?", ["300,000 km/s", "150,000 km/s", "450,000 km/s", "600,000 km/s"], "300,000 km/s"),
]

for q, choices, expected in failures:
    print(f"\n{'='*80}")
    print(f"Q: {q}")
    print(f"Choices: {choices}")
    print(f"Expected: {expected}")

    # Manually trace the solve() path
    # Step 1: KB query
    expanded_query = q + " " + " ".join(choices)
    knowledge_hits = solver.kb.query(expanded_query, top_k=12)
    print(f"\nKB hits ({len(knowledge_hits)}):")
    for key, node, score in knowledge_hits[:5]:
        print(f"  {key}: score={score:.3f}, {len(node.facts)} facts")
        for f in node.facts[:3]:
            print(f"    - {f[:100]}")

    # Check _has_context
    r = solver.solve(q, choices)
    print(f"\nResult: {r['answer_text']} ({r['answer']})")
    print(f"All scores: {r['all_scores']}")
    print(f"Knowledge hits: {r['knowledge_hits']}, Context facts: {r['context_facts_used']}")

    # Numerical reasoning check
    if solver.numerical_reasoner:
        print(f"\nNumerical reasoning:")
        for ch in choices:
            nums = solver.numerical_reasoner._parse_choice_numbers(ch)
            ns = solver.numerical_reasoner.score_numerical_match(ch, [], q)
            print(f"  {ch}: parsed nums={nums}, score_with_empty_facts={ns}")

    # Check q_nouns extraction
    q_lower = q.lower()
    q_nouns = set()
    for m in re.finditer(r'"([^"]+)"', q):
        q_nouns.add(m.group(1).lower())
    for m in re.finditer(r'(?:known as|called|named|nicknamed)\s+(?:the\s+)?(.+?)(?:\?|$)', q_lower):
        q_nouns.add(m.group(1).strip().rstrip('?. '))
    for m in re.finditer(r'(?:what|which|who)\s+(?:is|are|was|were)\s+(?:the\s+)?(.+?)(?:\?|$)', q_lower):
        phrase = m.group(1).strip().rstrip('?. ')
        if len(phrase) > 3:
            q_nouns.add(phrase)
    print(f"\nq_nouns extracted: {q_nouns}")
