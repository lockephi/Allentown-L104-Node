#!/usr/bin/env python3
"""Verify KB health post-training: search quality, category distribution, BM25 index."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from l104_intellect import local_intellect

# Force materialization
li = local_intellect
li._ensure_training_index()

td = li.training_data
print(f"=== KB SIZE: {len(td)} entries ===\n")

# Category distribution
cats = {}
sources = {}
for entry in td:
    cat = entry.get("category", "unknown")
    src = entry.get("source", "unknown")
    cats[cat] = cats.get(cat, 0) + 1
    sources[src] = sources.get(src, 0) + 1

print(f"Categories: {len(cats)}")
for c, n in sorted(cats.items(), key=lambda x: -x[1])[:20]:
    print(f"  {c}: {n}")

print(f"\nSources: {len(sources)}")
for s, n in sorted(sources.items(), key=lambda x: -x[1])[:20]:
    print(f"  {s}: {n}")

# BM25 search quality
print("\n=== SEARCH QUALITY (BM25) ===")
test_queries = [
    ("GOD_CODE", "GOD_CODE"),
    ("primal calculus", "primal"),
    ("quantum gate", "quantum"),
    ("Maxwell Demon", "maxwell"),
    ("Fibonacci", "fibonacci"),
    ("Landauer limit", "landauer"),
    ("CUDA kernel", "cuda"),
    ("Lorentz boost", "lorentz"),
    ("Science Engine", "science_engine"),
    ("Code Engine", "code_engine"),
    ("Quantum Gate Engine", "gate_engine"),
    ("assembly kernel", "asm"),
]

passed = 0
for query, expect_in_result in test_queries:
    results = li._search_training_data(query, max_results=3)
    if results:
        top = results[0]
        q_text = top.get("question", "") + " " + top.get("answer", "") + " " + top.get("category", "")
        hit = expect_in_result.lower() in q_text.lower()
        status = "PASS" if hit else "WEAK"
        if hit:
            passed += 1
        print(f"  [{status}] '{query}' -> top result category='{top.get('category','?')}' q='{str(top.get('question',''))[:60]}'")
    else:
        print(f"  [MISS] '{query}' -> no results")

print(f"\nSearch quality: {passed}/{len(test_queries)} ({100*passed/len(test_queries):.0f}%)")

# Check kernel-specific entries
print("\n=== KERNEL/ENGINE KB ENTRIES ===")
kernel_cats = [c for c in cats if "kernel" in c.lower() or "engine" in c.lower() or "native" in c.lower()]
for c in sorted(kernel_cats):
    print(f"  {c}: {cats[c]}")

print("\n=== DONE ===")
