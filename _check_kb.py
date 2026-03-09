#!/usr/bin/env python3
"""Check KB contents directly for the 3 problem facts."""
import sys, os, logging
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
logging.disable(logging.CRITICAL)
sys.stderr = open(os.devnull, 'w')
sys.path.insert(0, '.')

from l104_asi.language_comprehension.knowledge_base import KnowledgeBase
sys.stderr = sys.__stderr__

try:
    kb = KnowledgeBase()
    kb.initialize()
except Exception as e:
    print(f"KB init error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"Total nodes: {len(kb.nodes)}")
print(f"Total facts: {kb._total_facts}")

# Check specific nodes
for key in sorted(kb.nodes.keys()):
    node = kb.nodes[key]
    all_text = (node.definition + " " + " ".join(node.facts)).lower()
    if 'cogito' in all_text:
        print(f"\ncogito found in: {key} (subject={node.subject}, {len(node.facts)} facts)")
        for f in node.facts:
            if 'cogito' in f.lower():
                print(f"  FACT: {f[:150]}")
    if 'maslow' in all_text and 'physiological' in all_text:
        print(f"\nmaslow+physiological in: {key} (subject={node.subject}, {len(node.facts)} facts)")
        for f in node.facts:
            if 'maslow' in f.lower() or 'physiological' in f.lower():
                print(f"  FACT: {f[:150]}")
    if 'scientific method' in all_text and 'observation' in all_text:
        print(f"\nsci.method+observation in: {key} (subject={node.subject}, {len(node.facts)} facts)")
        for f in node.facts:
            if 'scientific method' in f.lower() or 'observation' in f.lower():
                print(f"  FACT: {f[:150]}")

# Check subject mapping for 'science', 'philosophy', 'psychology'
print("\n=== Subject-matching nodes ===")
for subj_search in ['science', 'philosophy', 'psychology']:
    matching = []
    for key, node in kb.nodes.items():
        node_subj = node.subject.lower().replace(" ", "_")
        if subj_search in key or subj_search in node_subj:
            matching.append(key)
    print(f"\n  Subject '{subj_search}' matches {len(matching)} nodes:")
    for k in matching[:15]:
        print(f"    {k}")

print("\nDone.")
