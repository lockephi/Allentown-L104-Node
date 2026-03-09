#!/usr/bin/env python3
"""Test regex patterns against KB facts to find problematic ones."""
import re
import sys
import time
import signal
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load KB facts directly
spec = importlib.util.spec_from_file_location('kd', 'l104_asi/knowledge_data.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

all_facts = []
for node in mod.KNOWLEDGE_NODES:
    all_facts.extend(node.get('facts', []))
print(f'Total facts: {len(all_facts)}')

# The regex patterns from TripleExtractor
PATTERNS = [
    (r'(?:the\s+)?(.{2,60}?)\s+(?:is|are|was|were)\s+(?:the\s+|a\s+|an\s+)?(.{2,80}?)(?:\s*[\(\,\.\;]|$)', 'is'),
    (r'(\w[\w\s]{1,30}?)\s+wrote\s+(.{2,60}?)(?:\s*[\,\.\;]|$)', 'wrote'),
    (r'(\w[\w\s]{1,30}?)\s+(?:discovered|invented|developed|created|founded|established)\s+(.{2,60}?)(?:\s*[\,\.\;]|$)', 'created'),
    (r'(\w{1,10})\s+stands\s+for\s+(.{2,60}?)(?:\s*$)', 'stands_for'),
    (r'(\w[\w\s]{1,30}?)\s+uses?\s+(.{2,40}?)(?:\s*[\,\.\;]|$)', 'uses'),
    (r'(\w[\w\s]{1,40}?)\s+(?:contains?|consists?\s+of)\s+(.{2,60}?)(?:\s*[\,\.\;]|$)', 'contains'),
    (r'(\w[\w\s]{1,40}?)\s+(?:causes?|leads?\s+to|results?\s+in)\s+(.{2,60}?)(?:\s*[\,\.\;]|$)', 'causes'),
    (r'(\w[\w\s]{1,30}?)\s+is\s+(?:located|found|situated)\s+in\s+(.{2,40}?)(?:\s*[\,\.\;]|$)', 'located_in'),
    (r'^(\w[\w\s]{1,30}?):\s+(.{5,80}?)(?:\s*$)', 'defined_as'),
    (r'(\w[\w\s\^\/\*\+\-]{1,30}?)\s*=\s*(.{2,60}?)(?:\s*[\,\.\;]|$)', 'equals'),
    (r'(?:the\s+)?(?:symbol|chemical\s+symbol|formula)\s+(?:for|of)\s+(\w[\w\s]{1,30}?)\s+is\s+(\w{1,10})', 'symbol_of'),
    (r'(\w[\w\s]{1,30}?)\s+(?:published|proposed|formulated)\s+(.{2,60}?)(?:\s*[\,\.\;]|$)', 'published'),
]

def timeout_handler(signum, frame):
    raise TimeoutError("regex timeout")

signal.signal(signal.SIGALRM, handler=timeout_handler)

slow_facts = []
error_facts = []
long_facts = 0

for i, fact in enumerate(all_facts):
    fact_lower = fact.lower().strip()
    if len(fact_lower) > 500:
        long_facts += 1
    for pat_idx, (pattern, pred) in enumerate(PATTERNS):
        try:
            signal.alarm(3)
            list(re.finditer(pattern, fact_lower, re.IGNORECASE))
            signal.alarm(0)
        except TimeoutError:
            slow_facts.append((i, pat_idx, pred, fact[:120]))
            signal.alarm(0)
        except re.error as e:
            error_facts.append((i, pat_idx, str(e), fact[:120]))
            signal.alarm(0)

signal.alarm(0)

print(f"\nLong facts (>500 chars): {long_facts}")
print(f"Regex errors: {len(error_facts)}")
for idx, pidx, err, f in error_facts[:10]:
    print(f"  fact[{idx}] pattern[{pidx}]: {err}")
    print(f"    {f}")

print(f"\nSlow (>3s timeout): {len(slow_facts)}")
for idx, pidx, pred, f in slow_facts[:10]:
    print(f"  fact[{idx}] pattern[{pidx}] ({pred}): {f}")

# Time the full indexing
print(f"\n--- Timing full triple extraction on {len(all_facts)} facts ---")
start = time.time()
count = 0
for fact in all_facts:
    fact_lower = fact.lower().strip()
    for pattern, predicate in PATTERNS:
        for m in re.finditer(pattern, fact_lower, re.IGNORECASE):
            count += 1
elapsed = time.time() - start
print(f"Extracted {count} triples in {elapsed:.1f}s ({elapsed/len(all_facts)*1000:.1f}ms/fact)")
