#!/usr/bin/env python3
"""Test advanced queries with the enhanced L104 Local Intellect."""

from l104_local_intellect import local_intellect

print("🔬 TESTING ADVANCED QUERIES...")
print()

queries = [
    "Calculate the Riemann zeta function at s=2",
    "Explain quantum entanglement",
    "What is GOD_CODE?",
    "How does the 11D Calabi-Yau manifold work?",
    "What is consciousness?",
]

for q in queries:
    print(f"Q: {q}")
    resp = local_intellect.think(q)
    lines = resp.split("\n")
    core = "\n".join(lines[2:-2]) if len(lines) > 4 else resp
    print(f"A: {core[:400]}..." if len(core) > 400 else f"A: {core}")
    print()
    print("-" * 60)
    print()

print("📊 FINAL STATS:")
print(f"   Training entries: {len(local_intellect.training_data):,}")
print(f"   Conversation memory: {len(local_intellect.conversation_memory)}")
print(f"   EPR quantum links: {local_intellect.entanglement_state.get('epr_links', 0)}")
print(f"   Vishuddha clarity: {local_intellect.vishuddha_state.get('clarity', 0):.4f}")
