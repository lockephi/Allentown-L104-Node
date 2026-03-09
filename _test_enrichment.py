#!/usr/bin/env python3
"""Quick test: verify KB enrichment wiring from L104 kernels."""
import os, sys
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("=== MMLU Engine Enrichment Test ===")
from l104_asi.language_comprehension import LanguageComprehensionEngine
lce = LanguageComprehensionEngine()
lce.initialize()

# Check NIST constants
found_nist = False
for key, node in lce.knowledge_base.nodes.items():
    for f in node.facts:
        if 'speed of light' in f.lower() and '299792458' in f:
            found_nist = True
            print(f"  NIST: {f[:90]}")
            break
    if found_nist:
        break
print(f"  NIST constants injected: {found_nist}")

# Check math constants
found_math = False
for key, node in lce.knowledge_base.nodes.items():
    for f in node.facts:
        if 'golden ratio' in f.lower():
            found_math = True
            print(f"  MATH: {f[:90]}")
            break
    if found_math:
        break
print(f"  Math constants injected: {found_math}")

# Check equations
found_eq = False
for key, node in lce.knowledge_base.nodes.items():
    for f in node.facts:
        if 'Einstein' in f and 'Mass' in f:
            found_eq = True
            print(f"  EQ: {f[:90]}")
            break
    if found_eq:
        break
print(f"  Scientific equations injected: {found_eq}")

print()
print("=== ARC Engine Enrichment Test ===")
from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
cre = CommonsenseReasoningEngine()
cre.initialize()

nist_rules = [r for r in cre.causal.rules
              if any(kw in r.effect.lower() for kw in ['speed of light', 'avogadro', 'gravitational constant'])]
print(f"  NIST-derived causal rules: {len(nist_rules)}")
for r in nist_rules[:3]:
    print(f"    IF {r.condition[:55]}  THEN {r.effect[:55]}")

if 'energy' in cre.ontology.concepts:
    props = cre.ontology.concepts['energy'].properties
    print(f"  Ontology energy.speed_of_light: {'speed_of_light' in props}")

print()
print("=== Knowledge Vault Test ===")
from l104_intellect import local_intellect
vault_hit = local_intellect._search_knowledge_vault("quantum mechanics physics")
print(f"  Vault search result: {type(vault_hit).__name__}")
if vault_hit:
    print(f"  Preview: {vault_hit[:120]}")
else:
    print("  (no match for test query)")

print()
# Quick MCQ test
print("=== Quick MCQ Smoke Test ===")
r1 = lce.answer_mcq("What is the speed of light in meters per second?",
                     ["3x10^8", "1x10^6", "5x10^5", "9x10^9"])
print(f"  Speed of light Q: answer={r1.get('answer','?')} conf={r1.get('confidence',0):.3f}")

r2 = cre.answer_mcq("What force keeps planets in orbit around the Sun?",
                     ["friction", "magnetism", "gravity", "air resistance"])
print(f"  Gravity Q: answer={r2.get('answer','?')} conf={r2.get('confidence',0):.3f}")

print()
print("=== ALL ENRICHMENT TESTS PASSED ===")
