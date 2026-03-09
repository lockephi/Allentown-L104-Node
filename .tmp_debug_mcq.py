"""Debug MCQ scoring — full solve() pipeline for failing questions."""
import sys, json
sys.path.insert(0, ".")
from l104_asi.language_comprehension import LanguageComprehensionEngine

engine = LanguageComprehensionEngine()

# ---- Test 1: Powerhouse of the cell ----
q1 = "What is the powerhouse of the cell?"
choices1 = ["Nucleus", "Mitochondria", "Ribosome", "Golgi apparatus"]

print("=" * 70)
print(f"Q: {q1}")
r1 = engine.mcq_solver.solve(q1, choices1)
print(f"Answer: {r1['answer']}={r1['answer_text']}")
print(f"All scores: {r1['all_scores']}")
print(f"Calibration: {r1['calibration']}")
print()

# ---- Test 2: Who wrote The Republic ----
q2 = "Who wrote The Republic?"
choices2 = ["Aristotle", "Socrates", "Plato", "Confucius"]

print("=" * 70)
print(f"Q: {q2}")
r2 = engine.mcq_solver.solve(q2, choices2)
print(f"Answer: {r2['answer']}={r2['answer_text']}")
print(f"All scores: {r2['all_scores']}")
print(f"Knowledge hits: {r2['knowledge_hits']}, Context facts: {r2['context_facts_used']}")

# Check if epistemology node gets retrieved
print("\n--- KB retrieval debug for Republic ---")
hits = engine.knowledge_base.query(q2, top_k=20)
for key, node, score in hits:
    has_plato = any("plato" in f.lower() for f in node.facts)
    has_republic = any("republic" in f.lower() for f in node.facts)
    if has_plato or has_republic:
        print(f"  ** {key} (score={score:.4f}) - Plato:{has_plato} Republic:{has_republic}")

# Expanded query
expanded = q2 + " " + " ".join(choices2)
hits2 = engine.knowledge_base.query(expanded, top_k=8)
for key, node, score in hits2:
    has_plato = any("plato" in f.lower() for f in node.facts)
    has_republic = any("republic" in f.lower() for f in node.facts)
    if has_plato or has_republic:
        print(f"  EX {key} (score={score:.4f}) - Plato:{has_plato} Republic:{has_republic}")

# Check epistemology node directly
if "epistemology" in engine.knowledge_base.nodes:
    print("\n--- Epistemology node facts ---")
    for f in engine.knowledge_base.nodes["epistemology"].facts:
        print(f"  {f[:80]}")
else:
    print("  epistemology node NOT found")
