#!/usr/bin/env python3
"""Trace why Q2 'Red Planet' predicts Jupiter instead of Mars."""
import sys, os, re, logging
logging.disable(logging.WARNING)

from l104_asi.language_comprehension import LanguageComprehensionEngine

lce = LanguageComprehensionEngine()
lce.initialize()
s = lce.mcq_solver

q = "Which planet is known as the Red Planet?"
choices = ["Venus", "Mars", "Jupiter", "Saturn"]

# Get context facts
knowledge_hits = s.kb.query(q, top_k=8)
expanded_query = q + " " + " ".join(choices)
extra_hits = s.kb.query(expanded_query, top_k=4)

seen_keys = set()
merged = []
for key, node, score in knowledge_hits:
    if key not in seen_keys:
        merged.append((key, node, score))
        seen_keys.add(key)
for key, node, score in extra_hits:
    if key not in seen_keys:
        merged.append((key, node, score))
        seen_keys.add(key)
knowledge_hits = merged

# BM25 re-rank
from l104_asi.language_comprehension import BM25Ranker
all_facts = []
for key, node, scr in knowledge_hits:
    for fact in node.facts:
        all_facts.append(fact)
    all_facts.append(node.definition)

bm25 = BM25Ranker()
bm25.fit(all_facts)
bm25_ranked = bm25.rank(q, top_k=min(20, len(all_facts)))
context_facts = []
seen = set()
for idx, sc in bm25_ranked:
    if sc > 0.01 and idx < len(all_facts):
        f = all_facts[idx]
        if f not in seen:
            context_facts.append(f)
            seen.add(f)

print(f"Context facts: {len(context_facts)}")

# Score each choice via _score_choice
print("\n=== _score_choice (Stage 1-9) ===")
for i, ch in enumerate(choices):
    scr = s._score_choice(q, ch, context_facts, knowledge_hits)
    print(f"  {chr(65+i)} ({ch}): {scr:.4f}")

# Now let's trace Step 3a (Semantic TF-IDF)
print("\n=== Semantic TF-IDF (Step 3a) ===")
keyword_max = 0  # All are 0
kb_has_signal = keyword_max > 0.01
print(f"  kb_has_signal = {kb_has_signal} → TF-IDF {'SKIPPED' if not kb_has_signal else 'APPLIED'}")

if hasattr(s.kb, 'encoder') and s.kb.encoder._corpus_vectors is not None:
    encoder = s.kb.encoder
    import numpy as np
    for i, ch in enumerate(choices):
        qc_text = f"{q} {ch}"
        qc_vec = encoder.encode(qc_text)
        sims = encoder._corpus_vectors @ qc_vec
        top_sims = sorted(sims, reverse=True)[:3]
        sem_score = sum(top_sims) / len(top_sims) if top_sims else 0.0
        print(f"  {chr(65+i)} ({ch}): raw_sem={sem_score:.4f}, top3={[round(float(s), 4) for s in sorted(sims, reverse=True)[:3]]}")

# Test fallback heuristics
print("\n=== Fallback heuristics ===")
for i, ch in enumerate(choices):
    h = s._fallback_heuristics(q, ch, choices)
    print(f"  {chr(65+i)} ({ch}): {h:.4f}")

print("\n=== Full solve() result ===")
r = lce.answer_mcq(q, choices)
for sc in r["all_scores"]:
    marker = " ← PREDICTED" if sc["label"] == r["answer"] else ""
    print(f"  {sc['label']}: {sc['score']:.4f}{marker}")
print(f"  kb_has_signal: {any(sc['score'] > 0.01 for sc in r['all_scores'])}")
