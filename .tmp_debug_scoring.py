#!/usr/bin/env python3
"""Deep debug: trace exactly what facts/scores each question gets."""
import sys, os, re, logging
logging.disable(logging.WARNING)

from l104_asi.language_comprehension import LanguageComprehensionEngine

lce = LanguageComprehensionEngine()
lce.initialize()
solver = lce.mcq_solver

# Focus on the failing questions
test_cases = [
    {
        "q": "Which planet is known as the Red Planet?",
        "c": ["Venus", "Mars", "Jupiter", "Saturn"],
        "expected": 1,
    },
    {
        "q": "Which of the following is NOT a function of the liver?",
        "c": ["Bile production", "Insulin production", "Detoxification", "Glycogen storage"],
        "expected": 1,
    },
]

for tc in test_cases:
    q = tc["q"]
    choices = tc["c"]
    print(f"\n{'='*70}")
    print(f"Q: {q}")
    print(f"Choices: {choices}")
    print(f"Expected: {chr(65+tc['expected'])} ({choices[tc['expected']]})")

    # Step 1: Get knowledge hits
    knowledge_hits = solver.kb.query(q, top_k=8)
    print(f"\n  Knowledge hits: {len(knowledge_hits)}")
    for key, node, score in knowledge_hits[:5]:
        print(f"    [{score:.3f}] {key}: {node.definition[:80]}...")

    # Step 1b: BM25 re-rank
    all_facts = []
    for key, node, s in knowledge_hits:
        for fact in node.facts:
            all_facts.append(fact)
        all_facts.append(node.definition)

    from l104_asi.language_comprehension import BM25Ranker
    bm25 = BM25Ranker()
    bm25.fit(all_facts)
    bm25_ranked = bm25.rank(q, top_k=min(20, len(all_facts)))

    context_facts = []
    seen = set()
    for doc_idx, bm25_score in bm25_ranked:
        if bm25_score > 0.01 and doc_idx < len(all_facts):
            fact = all_facts[doc_idx]
            if fact not in seen:
                context_facts.append(fact)
                seen.add(fact)

    print(f"\n  Context facts after BM25 ({len(context_facts)}):")
    for i, f in enumerate(context_facts[:10]):
        print(f"    [{i}] {f[:100]}")

    # Step 2d: Direct answer extraction check
    q_lower_da = q.lower()
    q_nouns = set()
    for m in re.finditer(r'(?:known as|called|named|nicknamed)\s+(?:the\s+)?(.+?)(?:\?|$)', q_lower_da):
        q_nouns.add(m.group(1).strip().rstrip('?. '))
    for m in re.finditer(r'(?:what|which|who)\s+(?:is|are|was|were)\s+(?:the\s+)?(.+?)(?:\?|$)', q_lower_da):
        phrase = m.group(1).strip().rstrip('?. ')
        if len(phrase) > 3:
            q_nouns.add(phrase)
    print(f"\n  Key phrases extracted: {q_nouns}")

    if q_nouns:
        for fact in context_facts[:15]:
            fl = fact.lower()
            for phrase in q_nouns:
                if phrase in fl:
                    print(f"    MATCH: phrase '{phrase}' in fact: {fact[:100]}")
                    for ci, ch in enumerate(choices):
                        if ch.lower().strip() in fl:
                            print(f"      → Choice {chr(65+ci)} ({ch}) also in fact!")

    # Stage 3: Score each choice individually
    print(f"\n  Per-choice scores:")
    for ci, ch in enumerate(choices):
        s = solver._score_choice(q, ch, context_facts, knowledge_hits)
        print(f"    {chr(65+ci)} ({ch}): {s:.4f}")

    # Check for negation
    is_neg = bool(re.search(r'\bnot\b|\bexcept\b', q.lower()))
    print(f"\n  Is NOT question: {is_neg}")
