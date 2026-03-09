#!/usr/bin/env python3
"""Check KB after actual query to ensure it's initialized."""
import sys, os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)

from l104_asi.language_comprehension import LanguageComprehensionEngine
lce = LanguageComprehensionEngine()

# Trigger KB init by running a query
lce.answer_mcq("Test question?", ["A", "B", "C", "D"])

# Now check KB
mcq = None
for attr in dir(lce):
    obj = getattr(lce, attr)
    if hasattr(obj, 'kb') and hasattr(obj.kb, 'nodes') and len(obj.kb.nodes) > 0:
        mcq = obj
        break

if mcq is None:
    print("Still can't find initialized KB")
    # Try nested attributes
    for attr in dir(lce):
        obj = getattr(lce, attr)
        if hasattr(obj, 'kb'):
            print(f"  lce.{attr}.kb has {len(obj.kb.nodes)} nodes")
        if hasattr(obj, 'solver') and hasattr(obj.solver, 'kb'):
            print(f"  lce.{attr}.solver.kb has {len(obj.solver.kb.nodes)} nodes")
        if hasattr(obj, '_mcq_solver') and hasattr(obj._mcq_solver, 'kb'):
            print(f"  lce.{attr}._mcq_solver.kb has {len(obj._mcq_solver.kb.nodes)} nodes")
    sys.exit(1)

kb = mcq.kb
print(f"Total nodes: {len(kb.nodes)}")
print(f"Total facts: {kb._total_facts}")

# Check specific facts
for key in sorted(kb.nodes.keys()):
    node = kb.nodes[key]
    all_text = (node.definition + " " + " ".join(node.facts)).lower()
    if 'cogito' in all_text:
        print(f"\ncogito in: {key} ({len(node.facts)} facts)")
        for f in node.facts:
            if 'cogito' in f.lower():
                print(f"  {f[:150]}")
    if 'maslow' in all_text and 'physiological' in all_text:
        print(f"\nmaslow+phys in: {key} ({len(node.facts)} facts)")
        for f in node.facts:
            fl = f.lower()
            if 'maslow' in fl or 'physiological' in fl:
                print(f"  {f[:150]}")
    if 'scientific method' in all_text and 'begins' in all_text:
        print(f"\nsci method+begins in: {key} ({len(node.facts)} facts)")
        for f in node.facts:
            if 'scientific method' in f.lower() or 'begins' in f.lower():
                print(f"  {f[:150]}")

# Subject matching
print("\n=== Subject matching for 'science' ===")
for k in sorted(kb.nodes.keys()):
    if 'science' in k:
        n = kb.nodes[k]
        print(f"  {k} ({n.subject}, {len(n.facts)} facts)")

print("\n=== Subject matching for 'philosophy' ===")
for k in sorted(kb.nodes.keys()):
    if 'philosophy' in k:
        n = kb.nodes[k]
        print(f"  {k} ({n.subject}, {len(n.facts)} facts)")

print("\n=== Subject matching for 'psychology' ===")
for k in sorted(kb.nodes.keys()):
    if 'psychology' in k or 'maslow' in k:
        n = kb.nodes[k]
        print(f"  {k} ({n.subject}, {len(n.facts)} facts)")

# Check nodes with 'miscellaneous' in key
print("\n=== Miscellaneous nodes ===")
for k in sorted(kb.nodes.keys()):
    if 'miscellaneous' in k:
        n = kb.nodes[k]
        print(f"  {k} ({n.subject}, {len(n.facts)} facts)")

print("\nDone.")
