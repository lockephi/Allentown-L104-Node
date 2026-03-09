"""Targeted MCQ diagnostic: trace per-choice scores for failing questions."""
import sys, time, re, math
sys.path.insert(0, ".")

# Suppress noisy imports
import logging
logging.disable(logging.WARNING)

print("Booting LCE (this takes ~80s)...")
t0 = time.time()
from l104_asi.language_comprehension import LanguageComprehensionEngine
lce = LanguageComprehensionEngine()
print(f"Booted in {time.time()-t0:.0f}s\n")

solver = lce.mcq_solver

# The failing questions
FAILING = [
    {
        "q": "What is the primary function of mitochondria in a cell?",
        "choices": ["Protein synthesis", "Energy production", "Cell division", "DNA replication"],
        "expected": 1,  # B
    },
    {
        "q": "Which planet is known as the Red Planet?",
        "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
        "expected": 1,  # B
    },
    {
        "q": "What is the speed of light in vacuum?",
        "choices": ["3 × 10^6 m/s", "3 × 10^8 m/s", "3 × 10^10 m/s", "3 × 10^4 m/s"],
        "expected": 1,  # B
    },
    {
        "q": "Who wrote 'Romeo and Juliet'?",
        "choices": ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
        "expected": 1,  # B
    },
    {
        "q": "What is the main gas in Earth's atmosphere?",
        "choices": ["Oxygen", "Carbon dioxide", "Nitrogen", "Hydrogen"],
        "expected": 2,  # C
    },
]

for case in FAILING:
    q = case["q"]
    choices = case["choices"]
    expected_idx = case["expected"]

    print(f"{'='*70}")
    print(f"Q: {q}")
    print(f"Expected: [{chr(65+expected_idx)}] {choices[expected_idx]}")
    print()

    # Step 1: Check what KB retrieval finds
    expanded_query = q + " " + " ".join(choices)
    kb_hits = solver.kb.query(expanded_query, top_k=12)
    print(f"KB hits: {len(kb_hits)}")
    for key, node, score in kb_hits[:5]:
        print(f"  [{score:.3f}] {key}: {node.definition[:60]}...")

    # Step 2: Check BM25 fact ranking
    all_facts = []
    for key, node, score in kb_hits:
        for fact in node.facts:
            all_facts.append(fact)
        all_facts.append(node.definition)

    if all_facts:
        solver.bm25.fit(all_facts)
        bm25_ranked = solver.bm25.rank(q, top_k=10)
        print(f"\nTop BM25 facts for question:")
        for doc_idx, bm25_score in bm25_ranked[:5]:
            if bm25_score > 0.01:
                print(f"  [{bm25_score:.3f}] {all_facts[doc_idx][:80]}...")

    # Step 3: Per-choice _score_choice breakdown
    context_facts = []
    seen = set()
    if all_facts:
        for doc_idx, bm25_score in bm25_ranked:
            if bm25_score > 0.01 and doc_idx < len(all_facts):
                f = all_facts[doc_idx]
                if f not in seen:
                    context_facts.append(f)
                    seen.add(f)

    print(f"\nContext facts for scoring: {len(context_facts)}")
    has_ctx = len(context_facts) >= 3

    print(f"\nPer-choice _score_choice:")
    for i, ch in enumerate(choices):
        # Pass knowledge_hits instead of kb_hits
        sc = solver._score_choice(q, ch, context_facts, knowledge_hits=kb_hits, has_context=has_ctx)
        marker = "★" if i == expected_idx else " "
        print(f"  {marker} [{chr(65+i)}] score={sc:.4f}  {ch}")

    # Step 4: Fallback heuristics
    print(f"\nFallback heuristics (applied when max_score < 0.5 or no KB signal):")
    for i, ch in enumerate(choices):
        h = solver._fallback_heuristics(q, ch, choices)
        marker = "★" if i == expected_idx else " "
        print(f"  {marker} [{chr(65+i)}] heuristic={h:.4f}  {ch}")

    # Step 5: Direct answer extraction
    print(f"\nDirect answer bonus scan:")
    q_lower_da = q.lower()
    q_nouns = set()
    for m in re.finditer(r'"([^"]+)"', q):
        q_nouns.add(m.group(1).lower())
    for m in re.finditer(r'(?:known as|called|named|nicknamed)\s+(?:the\s+)?(.+?)(?:\?|$)', q_lower_da):
        q_nouns.add(m.group(1).strip().rstrip('?. '))
    for m in re.finditer(r'(?:what|which|who)\s+(?:is|are|was|were)\s+(?:the\s+)?(.+?)(?:\?|$)', q_lower_da):
        phrase = m.group(1).strip().rstrip('?. ')
        if len(phrase) > 3:
            q_nouns.add(phrase)
    print(f"  Question key phrases: {q_nouns}")

    if q_nouns and context_facts:
        da_bonus = {}
        for fact in context_facts[:15]:
            fl = fact.lower()
            for phrase in q_nouns:
                if phrase not in fl:
                    continue
                matching = []
                for ci, ch in enumerate(choices):
                    if ch.lower().strip() in fl:
                        matching.append(ci)
                if matching:
                    bonus_per = 5.0 / len(matching)
                    for ci in matching:
                        da_bonus[ci] = da_bonus.get(ci, 0.0) + bonus_per
                    print(f"  Fact: '{fact[:70]}...'")
                    print(f"    Matches: {[chr(65+c) for c in matching]}, bonus_each={bonus_per:.1f}")
        print(f"  Direct answer bonuses: {da_bonus}")
    else:
        print(f"  (no key phrases or no context facts)")

    # Step 6: Full solve
    t1 = time.time()
    result = solver.solve(q, choices)
    t2 = time.time()
    print(f"\nFull solve result ({t2-t1:.1f}s):")
    print(f"  Selected: [{result['answer']}] {result['answer_text']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  All scores: {result['all_scores']}")
    correct = result['answer_index'] == expected_idx
    print(f"  {'CORRECT ✓' if correct else 'WRONG ✗'}")
    print()

print("Done.")
