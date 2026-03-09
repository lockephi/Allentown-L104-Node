#!/usr/bin/env python3
"""Check the 5 specific facts we added."""
import sys, os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)

from l104_asi.language_comprehension import LanguageComprehensionEngine
lce = LanguageComprehensionEngine()
lce.answer_mcq("test?", ["a","b","c","d"])  # trigger init

# Find the KB
mcq = None
for attr in dir(lce):
    obj = getattr(lce, attr)
    if hasattr(obj, 'kb') and hasattr(obj.kb, 'nodes') and len(obj.kb.nodes) > 0:
        mcq = obj
        break

kb = mcq.kb

# Check science_literacy node
key = "miscellaneous/science_literacy"
if key in kb.nodes:
    node = kb.nodes[key]
    print(f"science_literacy: {len(node.facts)} facts")
    for i, f in enumerate(node.facts):
        print(f"  [{i}] {f[:120]}")
else:
    print(f"Node {key} not found!")
    # Search for it
    for k in sorted(kb.nodes.keys()):
        if 'science' in k and 'literacy' in k:
            print(f"  Found: {k}")

# Also check philosophy nodes for cogito
print("\n=== Philosophy nodes ===")
for k in sorted(kb.nodes.keys()):
    if 'philosophy' in k:
        node = kb.nodes[k]
        for f in node.facts:
            if 'cogito' in f.lower() or 'descartes' in f.lower():
                print(f"  {k}: {f[:120]}")

# Check psychology for Maslow at base
print("\n=== Psychology nodes with Maslow ===")
for k in sorted(kb.nodes.keys()):
    node = kb.nodes[k]
    for f in node.facts:
        if 'maslow' in f.lower():
            print(f"  {k}: {f[:150]}")
