"""Audit LCE pipeline for improvement opportunities."""
import sys, time
t0 = time.time()

from l104_asi.language_comprehension import LanguageComprehensionEngine, _get_cached_deep_nlu
lce = LanguageComprehensionEngine()
lce.initialize()
print(f"Boot: {time.time()-t0:.1f}s")

solver = lce.mcq_solver

# 1. Check API methods on layers that had debug mismatches
print("\n=== LAYER API AUDIT ===")
nr = solver.numerical_reasoner
print("NumericalReasoner:", [m for m in dir(nr) if not m.startswith('_')])

for layer_key in ['22_taxonomy_classification', '23_causal_chain_reasoning',
                   '25_commonsense_knowledge']:
    obj = lce.layers.get(layer_key)
    if obj:
        print(f"{layer_key}: {[m for m in dir(obj) if not m.startswith('_')]}")
    else:
        print(f"{layer_key}: NOT FOUND")

dnlu = _get_cached_deep_nlu()
if dnlu:
    print("DeepNLU:", [m for m in dir(dnlu) if not m.startswith('_')])

# 2. KB coverage gaps — sample questions from different domains
print("\n=== KB COVERAGE AUDIT ===")
test_qs = [
    ("What is the chemical formula for water?", ["H2O", "CO2", "NaCl", "O2"], "A"),
    ("Which organ pumps blood in the human body?", ["Brain", "Heart", "Liver", "Lungs"], "B"),
    ("What is the largest ocean on Earth?", ["Atlantic", "Indian", "Pacific", "Arctic"], "C"),
    ("In which year did World War II end?", ["1943", "1944", "1945", "1946"], "C"),
    ("What is the boiling point of water at sea level?", ["50°C", "100°C", "150°C", "200°C"], "B"),
    ("Which element has the chemical symbol 'Fe'?", ["Iron", "Fluorine", "Francium", "Fermium"], "A"),
    ("What is the square root of 144?", ["10", "12", "14", "16"], "B"),
    ("Who painted the Mona Lisa?", ["Michelangelo", "Leonardo da Vinci", "Raphael", "Donatello"], "B"),
    ("What is the capital of France?", ["London", "Paris", "Berlin", "Madrid"], "B"),
    ("Which planet is closest to the Sun?", ["Venus", "Mercury", "Mars", "Earth"], "B"),
    ("What is DNA an abbreviation for?", ["Deoxyribonucleic acid", "Dinitrogen acid", "Dioxin nucleic acid", "Denatured acid"], "A"),
    ("What force keeps planets in orbit around the Sun?", ["Electromagnetic", "Gravity", "Nuclear", "Friction"], "B"),
]

correct = 0
total = len(test_qs)
failures = []
for q, choices, expected in test_qs:
    t1 = time.time()
    result = solver.solve(q, choices)
    elapsed = time.time() - t1
    got = result["answer"]
    ok = got == expected
    if ok:
        correct += 1
    else:
        failures.append((q, expected, got, result["all_scores"]))
    status = "✓" if ok else "✗"
    print(f"  {status} [{got}] expect=[{expected}] {elapsed:.1f}s  {q[:60]}")

print(f"\n  Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
if failures:
    print(f"\n  === FAILURES ===")
    for q, exp, got, scores in failures:
        print(f"  Q: {q}")
        print(f"    Expected: {exp}, Got: {got}")
        print(f"    Scores: {scores}")

# 3. Performance profiling — measure per-stage timings
print("\n=== PERFORMANCE PROFILE ===")
import time
q = "What is the largest continent by area?"
choices = ["Africa", "Asia", "Europe", "North America"]
t1 = time.time()
result = solver.solve(q, choices)
total_ms = (time.time() - t1) * 1000
print(f"  Total solve time: {total_ms:.0f}ms")
print(f"  Answer: {result['answer']} ({result['answer_text']})")
print(f"  Knowledge hits: {result['knowledge_hits']}")
print(f"  Context facts: {result['context_facts_used']}")
print(f"  Calibration: {result['calibration']}")

print(f"\nDone in {time.time()-t0:.1f}s total")
