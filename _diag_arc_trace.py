"""Trace ARC scoring for specific missed questions to find why ontology isn't helping."""
import sys, os, re, time
sys.path.insert(0, os.path.dirname(__file__))

# Suppress engine output
os.environ['L104_QUIET'] = '1'

print("Initializing engines (this takes ~120s)...")
t0 = time.time()
from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
engine = CommonsenseReasoningEngine()
engine.initialize()
print(f"Engine ready in {time.time()-t0:.0f}s")

solver = engine.mcq_solver

# Test questions that are being missed
MISSED = [
    {
        "q": "Which planet is known as the Red Planet?",
        "choices": ["Mars", "Venus", "Jupiter", "Saturn"],
        "expected": "Mars"
    },
    {
        "q": "Sound travels fastest through which medium?",
        "choices": ["Steel", "Water", "Air", "Vacuum"],
        "expected": "Steel"
    },
    {
        "q": "An animal that eats only plants is called",
        "choices": ["Herbivore", "Carnivore", "Omnivore", "Decomposer"],
        "expected": "Herbivore"
    },
    {
        "q": "The moon's phases are caused by",
        "choices": ["the moon's position relative to Earth and Sun",
                     "Earth's shadow", "the sun's rotation", "lunar eclipses"],
        "expected": "the moon's position relative to Earth and Sun"
    },
    {
        "q": "Which body system is responsible for fighting infections?",
        "choices": ["Immune", "Nervous", "Digestive", "Skeletal"],
        "expected": "Immune"
    },
]

for item in MISSED:
    q = item["q"]
    choices = item["choices"]
    expected = item["expected"]

    print(f"\n{'='*70}")
    print(f"Q: {q}")
    print(f"Choices: {choices}")
    print(f"Expected: {expected}")
    print(f"{'='*70}")

    q_lower = q.lower()

    # 1. What concepts are extracted from question?
    q_concepts = solver._extract_concepts(q_lower)
    print(f"\n[1] Question concepts: {q_concepts}")

    # 2. What concepts are extracted from each choice?
    for ci, ch in enumerate(choices):
        ch_concepts = solver._extract_concepts(ch.lower())
        print(f"    Choice {ci} '{ch}' concepts: {ch_concepts}")

    # 3. Check ontology for each relevant concept
    all_concept_keys = set(q_concepts)
    for ch in choices:
        for c in solver._extract_concepts(ch.lower()):
            all_concept_keys.add(c)

    for ck in all_concept_keys:
        concept = solver.ontology.concepts.get(ck)
        if concept:
            print(f"\n    Ontology '{ck}': cat={concept.category}, props={dict(concept.properties)}")
        else:
            print(f"\n    Ontology '{ck}': NOT FOUND")

    # 4. Get causal rules
    causal_matches = solver.causal.query(q_lower, top_k=8)
    print(f"\n[2] Causal matches ({len(causal_matches)}):")
    for rule, score in causal_matches[:5]:
        print(f"    score={score:.3f}: IF {rule.condition} → {rule.effect}")

    # 5. Score each choice individually
    concepts = solver._extract_concepts(q_lower)
    _q_concepts = set(concepts)
    for ci, ch in enumerate(choices):
        ch_concepts = solver._extract_concepts(ch.lower())
        for c in ch_concepts:
            if c not in concepts:
                concepts.append(c)

    print(f"\n[3] Per-choice scores from _score_choice:")
    for ci, ch in enumerate(choices):
        score = solver._score_choice(q, ch, list(concepts), causal_matches)
        marker = " ★" if ch == expected else ""
        print(f"    [{ci}] '{ch}': {score:.4f}{marker}")

    # 6. Run full solve to see final answer
    result = solver.solve(q, choices)
    answer_label = result.get("answer", "?")
    answer_idx = result.get("answer_index", 0)
    answer_text = choices[answer_idx] if answer_idx < len(choices) else "?"
    all_scores = result.get("all_scores", {})
    print(f"\n[4] Full solve result: {answer_label} = '{answer_text}'")
    print(f"    All scores: {all_scores}")
    print(f"    Calibration: {result.get('calibration', {})}")
    print(f"    Concepts found: {result.get('concepts_found', [])}")

    print(f"\n    {'✓ CORRECT' if answer_text == expected else '✗ WRONG (got: ' + answer_text + ')'}")

print("\n\nDone.")
