"""Quick regression test on known Easy questions."""
from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
e = CommonsenseReasoningEngine()

tests = [
    ("Which substance is taken in from the air?",
     ['oxygen', 'water', 'carbon dioxide', 'nitrogen'], 2),
    ("Which type of energy does a person use to pedal a bicycle?",
     ['chemical', 'electrical', 'sound', 'light'], 0),
    ("What do plants give off during photosynthesis?",
     ['carbon dioxide', 'nitrogen', 'oxygen', 'hydrogen'], 2),
    ("What causes sound?",
     ['sunlight', 'magnetism', 'vibrations', 'gravity'], 2),
    ("Which is an example of an adaptation?",
     ['a dog learning tricks', 'a plant growing tall', 'a bird migrating', 'a cactus having spines'], 3),
    ("What is the main function of the roots of a plant?",
     ['to absorb water and nutrients', 'to produce food', 'to attract pollinators', 'to store seeds'], 0),
    ("Which of these describes a chemical change?",
     ['ice melting', 'paper tearing', 'wood burning', 'sugar dissolving'], 2),
    ("What property of air makes it different from solids?",
     ['Air is a mixture of gases.', 'Air is a liquid.', 'Air is a solid.', 'Air has mass.'], 0),
]

correct = 0
for q, choices, expected in tests:
    r = e.answer_mcq(q, choices)
    got = r.get('selected_index', -1)
    ok = got == expected
    if ok:
        correct += 1
    mark = "OK" if ok else "FAIL"
    print(f"  [{mark}] {q[:60]}")
    if not ok:
        print(f"       Got[{got}]: {choices[got]}")
        print(f"       Exp[{expected}]: {choices[expected]}")

print(f"\n{correct}/{len(tests)} correct")
