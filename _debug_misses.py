"""Debug the 2 remaining MCQ misses: Q20 (inclined plane) and MMLU Q09 (pi)."""
import sys
import re

# ── DEBUG Q20: Inclined Plane (ARC) ──
print("=" * 60)
print("DEBUG: Q20 — Inclined Plane (ARC)")
print("=" * 60)

from l104_asi.commonsense_reasoning import (
    CommonsenseMCQSolver, ConceptOntology, CausalReasoningEngine,
    PhysicalIntuition, AnalogicalReasoner, TemporalReasoningEngine,
    CrossVerificationEngine
)

ont = ConceptOntology()
causal = CausalReasoningEngine()
phys = PhysicalIntuition(ont)
analog = AnalogicalReasoner(ont)
temporal = TemporalReasoningEngine()
verifier = CrossVerificationEngine(ont, causal, temporal)
solver = CommonsenseMCQSolver(ont, causal, phys, analog, temporal, verifier)

q = "An inclined plane is a flat surface that is:"
choices = ["round like a wheel", "at an angle", "very smooth", "made of metal"]
result = solver.solve(q, choices, subject="science")
print(f"  Answer: {result['answer']} = {result['choice']}")
print(f"  Scores: {result['all_scores']}")
print(f"  Concepts: {result['concepts_found']}")
print(f"  Causal rules used: {result['causal_rules_used']}")

# Debug _score_choice directly
q_lower = q.lower()
concepts = solver._extract_concepts(q_lower)
causal.build()
causal_matches = causal.query(q_lower, top_k=8)
print(f"\n  Extracted concepts: {concepts}")
print(f"  Causal matches ({len(causal_matches)}):")
for rule, sc in causal_matches:
    print(f"    cond='{rule.condition[:50]}' eff='{rule.effect[:50]}' score={sc}")

print(f"\n  Per-choice scoring:")
for i, ch in enumerate(choices):
    sc = solver._score_choice(q_lower, ch.lower(), concepts, causal_matches)
    fb = solver._fallback_heuristics(q, ch, choices)
    print(f"    [{i}] {ch:25s} score={sc:.4f}  fallback={fb:.4f}  total={sc+fb:.4f}")

# Check fact table matching
print(f"\n  Fact table check:")
q_words = set(re.findall(r'\w+', q_lower))
for i, ch in enumerate(choices):
    c_words = set(re.findall(r'\w+', ch.lower()))
    best = 0.0
    best_entry = None
    for q_pat, a_pat, boost in solver._fact_table:
        q_hits = sum(1 for w in q_pat if w in q_words)
        q_ratio = q_hits / len(q_pat) if q_pat else 0
        a_hits = sum(1 for w in a_pat if w in c_words)
        a_ratio = a_hits / len(a_pat) if a_pat else 0
        min_a = 2 if len(a_pat) >= 4 else 1
        if q_ratio >= 0.6 and a_hits >= min_a:
            fact_score = boost * q_ratio * (a_ratio * 0.7 + 0.3)
            if fact_score > best:
                best = fact_score
                best_entry = (q_pat, a_pat, boost, q_hits, a_hits)
    if best > 0:
        print(f"    [{i}] {ch:25s} fact_boost={best:.4f} match={best_entry}")
    else:
        print(f"    [{i}] {ch:25s} fact_boost=0 (no match)")

# ── DEBUG MMLU Q09: Pi ──
print("\n" + "=" * 60)
print("DEBUG: MMLU Q09 — Pi value")
print("=" * 60)

from l104_asi.language_comprehension import MCQSolver, MMLUKnowledgeBase
kb = MMLUKnowledgeBase()
solver2 = MCQSolver(knowledge_base=kb)

q2 = "What is the value of pi rounded to two decimal places?"
choices2 = ["3.12", "3.14", "3.16", "3.18"]
result2 = solver2.solve(q2, choices2, subject="elementary_mathematics")
print(f"  Answer: {result2['answer']} = {result2.get('choice', '?')}")
print(f"  Scores: {result2.get('all_scores', {})}")

# Check what knowledge was retrieved
print(f"\n  Knowledge retrieval debug:")
facts = kb.query(q2, subject="elementary_mathematics", top_k=10)
print(f"  Retrieved {len(facts)} knowledge hits:")
for key, node, rel in facts[:5]:
    print(f"    [{key}] rel={rel:.3f}")
    print(f"      def: {node.definition[:80]}")
    print(f"      facts ({len(node.facts)}): {[f[:60] for f in node.facts[:3]]}")

# Check if pi node exists
print(f"\n  Pi node check:")
for key in ["pi", "pi_constant", "mathematical_constants", "circle", "elementary_mathematics"]:
    node = kb.nodes.get(key)
    if node:
        print(f"    [{key}] def='{node.definition[:80]}' facts={len(node.facts)}")
        for f in node.facts:
            if "pi" in f.lower() or "3.14" in f:
                print(f"      MATCH: '{f[:80]}'")

# Check per-choice _score_choice
print(f"\n  Per-choice scoring:")
context_facts_raw = [(key, node, rel) for key, node, rel in facts]
context_facts = []
for key, node, rel in context_facts_raw:
    context_facts.append(node.definition)
    context_facts.extend(node.facts)
for i, ch in enumerate(choices2):
    sc = solver2._score_choice(q2, ch, context_facts, facts)
    print(f"    [{i}] {ch:10s} score={sc:.4f}")
