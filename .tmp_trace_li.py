#!/usr/bin/env python3
"""Trace what local_intellect returns for failing questions."""
import sys, os, logging
logging.disable(logging.WARNING)

from l104_intellect import local_intellect as li

test_queries = [
    "Which planet is known as the Red Planet?",
    "Which of the following is NOT a function of the liver?",
    "Mars Red Planet",
    "liver function bile insulin detoxification",
    "planet Mars Venus Jupiter Saturn",
]

for q in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {q}")

    # Search training data
    results = li._search_training_data(q, max_results=5)
    if results:
        print(f"  Training data ({len(results)} results):")
        for r in results[:5]:
            if isinstance(r, dict):
                prompt = r.get('prompt', '')[:60]
                completion = r.get('completion', '')[:80]
                category = r.get('category', 'unknown')
                print(f"    [{category}] P: {prompt}")
                print(f"         C: {completion}")
    else:
        print("  No training data results")

    # Search knowledge manifold
    manifold = li._search_knowledge_manifold(q)
    if manifold:
        print(f"  Manifold: {str(manifold)[:120]}")
    else:
        print(f"  Manifold: None")

    # Search knowledge vault
    vault = li._search_knowledge_vault(q)
    if vault:
        print(f"  Vault: {str(vault)[:120]}")
    else:
        print(f"  Vault: None")
