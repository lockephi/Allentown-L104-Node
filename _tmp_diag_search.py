#!/usr/bin/env python3
"""Diagnose BM25 search misses for GOD_CODE and Fibonacci."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from l104_intellect import local_intellect
li = local_intellect
li._ensure_training_index()

# Check what "god_code" tokenizes to
import re

def clean_term(w):
    return ''.join(c for c in w.lower() if c.isalnum())

# Test tokenization
for q in ["GOD_CODE", "Fibonacci", "Maxwell Demon"]:
    terms = []
    for w in q.lower().split():
        cleaned = clean_term(w)
        if len(cleaned) > 2 and cleaned not in li._TRAINING_SEARCH_STOP:
            terms.append(cleaned)
    print(f"Query '{q}' -> terms: {terms}")

    for t in terms:
        if t in li.training_index:
            entries = li.training_index[t]
            print(f"  Index '{t}': {len(entries)} entries")
            for e in entries[:3]:
                p = e.get("prompt", "")[:60]
                cat = e.get("category", "?")
                print(f"    cat={cat} prompt='{p}'")
        else:
            print(f"  Index '{t}': NOT FOUND")
            # Find similar terms
            similar = [k for k in li.training_index if t[:4] in k][:5]
            if similar:
                print(f"    Similar terms: {similar}")

# Direct check: is "godcode" in the index?
print(f"\n'godcode' in index: {'godcode' in li.training_index}")
print(f"'fibonacci' in index: {'fibonacci' in li.training_index}")
print(f"'god' in index: {'god' in li.training_index}")

# Check the sacred_core_kb entries directly
print("\n=== Sacred core entries ===")
sacred = [e for e in li.training_data if e.get("source") == "sacred_core_kb"]
print(f"Count: {len(sacred)}")
for e in sacred:
    p = e.get("prompt", "")[:60]
    cat = e.get("category", "?")
    first_word = clean_term(p.split()[0]) if p.split() else "?"
    print(f"  [{cat}] '{p}'")
