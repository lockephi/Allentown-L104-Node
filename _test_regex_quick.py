#!/usr/bin/env python3
"""Quick regex backtracking test — find the exact problem."""
import re
import time
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

spec = importlib.util.spec_from_file_location('kd', 'l104_asi/knowledge_data.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

all_facts = []
for node in mod.KNOWLEDGE_NODES:
    all_facts.extend(node.get('facts', []))
print(f'Total facts: {len(all_facts)}')

# Test ONLY the first (most dangerous) pattern
pattern = r'(?:the\s+)?(.{2,60}?)\s+(?:is|are|was|were)\s+(?:the\s+|a\s+|an\s+)?(.{2,80}?)(?:\s*[\(\,\.\;]|$)'

# Find facts that are slow
slow = []
for i, fact in enumerate(all_facts):
    fl = fact.lower().strip()
    start = time.time()
    try:
        list(re.finditer(pattern, fl, re.IGNORECASE))
    except Exception:
        pass
    elapsed = time.time() - start
    if elapsed > 0.01:  # more than 10ms
        slow.append((i, elapsed, fl[:100]))
    if i % 2000 == 0:
        print(f"  checked {i}/{len(all_facts)}...")
    # Safety: abort if any single fact takes > 5s
    if elapsed > 5:
        print(f"  ABORT: fact [{i}] took {elapsed:.1f}s!")
        print(f"  Content: {fl[:200]}")
        break

print(f"\nSlow facts (>10ms): {len(slow)}")
for idx, t, f in slow[:20]:
    print(f"  [{idx}] {t*1000:.0f}ms: {f}")
