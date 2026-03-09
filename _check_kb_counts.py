#!/usr/bin/env python3
"""Quick check of KB facts count — per-category breakdown via the knowledge_data package."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from l104_asi.knowledge_data import (
    KNOWLEDGE_NODES, CROSS_SUBJECT_RELATIONS,
    STEM_NODES, HUMANITIES_NODES, SOCIAL_SCIENCES_NODES, OTHER_NODES,
    summary,
)

print("═" * 60)
print("  L104 Knowledge Base — Counts Check")
print("═" * 60)

# Per-category counts
for label, nodes in [("STEM", STEM_NODES), ("Humanities", HUMANITIES_NODES),
                     ("Social Sciences", SOCIAL_SCIENCES_NODES), ("Other", OTHER_NODES)]:
    facts = sum(len(n.get("facts", [])) for n in nodes)
    print(f"  {label:20s}: {len(nodes):4d} nodes, {facts:6d} facts")

# Totals
total_facts = sum(len(n.get("facts", [])) for n in KNOWLEDGE_NODES)
print(f"  {'TOTAL':20s}: {len(KNOWLEDGE_NODES):4d} nodes, {total_facts:6d} facts")
print(f"  Cross-subject relations: {len(CROSS_SUBJECT_RELATIONS)}")

# Verify assembled list == sum of parts
parts = len(STEM_NODES) + len(HUMANITIES_NODES) + len(SOCIAL_SCIENCES_NODES) + len(OTHER_NODES)
assert parts == len(KNOWLEDGE_NODES), f"Assembly mismatch: {parts} != {len(KNOWLEDGE_NODES)}"

# Registry summary
s = summary()
print(f"\nRegistry summary: {s['total_nodes']} nodes, {s['total_facts']} facts, {s['total_relations']} relations")
print("✅ All counts verified")
