#!/usr/bin/env python3
"""Check KB contents via LanguageComprehensionEngine."""
import sys, os, re
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, '.')

# Suppress all logging
import logging
logging.disable(logging.CRITICAL)

try:
    from l104_asi.language_comprehension import LanguageComprehensionEngine
    lce = LanguageComprehensionEngine()

    # Access KB through the MCQ solver
    mcq = lce._mcq_solver if hasattr(lce, '_mcq_solver') else None
    if mcq is None:
        # Try other attribute names
        for attr in dir(lce):
            obj = getattr(lce, attr)
            if hasattr(obj, 'kb'):
                mcq = obj
                break

    if mcq is None:
        print("Could not find MCQ solver with KB")
        for attr in dir(lce):
            if not attr.startswith('_'):
                print(f"  lce.{attr}: {type(getattr(lce, attr))}")
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
                if 'maslow' in f.lower() or 'physiological' in f.lower():
                    print(f"  {f[:150]}")
        if 'scientific method' in all_text and ('observation' in all_text or 'begins' in all_text):
            print(f"\nsci method in: {key} ({len(node.facts)} facts)")
            for f in node.facts:
                if 'scientific method' in f.lower() or ('observation' in f.lower() and 'begins' in f.lower()):
                    print(f"  {f[:150]}")

    # Check subject mapping
    print("\n=== Subject matching ===")
    for s in ['science', 'philosophy', 'psychology']:
        matches = [k for k in kb.nodes if s in k]
        print(f"'{s}' in key: {len(matches)} - {matches[:10]}")

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
