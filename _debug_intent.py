#!/usr/bin/env python3
"""Debug the two remaining failing questions to see score details."""
import os, sys
os.environ['L104_QUIET'] = '1'
import logging; logging.disable(logging.WARNING)

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine

e = CommonsenseReasoningEngine()
e.initialize()

tests = [
    ("The speed at which sound waves travel depends on the",
     ["size of the object making the sound.", "direction the sound is moving.",
      "type of material through which the sound travels.",
      "amount of energy that originally produced the sound."], 2),
    ("What allows a light bulb to give off light?",
     ["the light bulb generating heat energy",
      "the current flowing through the wire to the light bulb",
      "the shade around the light bulb absorbing light",
      "the batteries providing solar energy to the light bulb"], 1),
]

for i, (q, choices, exp) in enumerate(tests):
    print(f"\n{'='*70}")
    print(f"Q{i}: {q}")
    print(f"Expected: {exp} = {choices[exp]}")

    # Get the solver internal state
    solver = e.mcq_solver
    r = solver.solve(q, choices)
    pred = r.get('selected_index', -1)
    print(f"Got: {pred} = {choices[pred]}")

    # Show detailed scores from pipeline
    details = r.get('details', {})
    print(f"\nFinal scores: {r.get('final_scores', 'N/A')}")

    # Check what concepts were found
    ql = q.lower()
    matched_concepts = []
    for cname, cprops in e.ontology.concepts.items():
        if cname in ql:
            matched_concepts.append(cname)
    print(f"Concepts in question: {matched_concepts}")

    # Check causal rules
    causal_results = e.causal.query(q)
    if causal_results:
        print(f"Causal results ({len(causal_results)}):")
        for cr in causal_results[:5]:
            print(f"  {cr}")
    else:
        print("No causal results")

    # Check each choice score individually
    print(f"\nPer-choice scoring detail:")
    for ci, ch in enumerate(choices):
        ch_lower = ch.lower()
        relevant = []
        for cname, concept in e.ontology.concepts.items():
            if cname in ql:
                for pk, pv in concept.properties.items():
                    pv_str = str(pv).lower()
                    ch_words = set(ch_lower.split()) - {'the', 'a', 'an', 'of', 'in', 'to', 'is', 'by', 'for', 'on', 'at', 'and', 'or', 'that', 'which', 'through'}
                    for w in ch_words:
                        if len(w) > 3 and w in pv_str:
                            relevant.append(f"{cname}.{pk}={pv_str[:60]} (matched '{w}')")
        if relevant:
            print(f"  [{ci}] {ch[:50]}  -> concept matches:")
            for rm in relevant:
                print(f"      {rm}")
        else:
            print(f"  [{ci}] {ch[:50]}  -> no concept property matches")

    # Show all properties for question-relevant concepts
    print(f"\nAll properties for matched concepts:")
    for cname, concept in e.ontology.concepts.items():
        if cname in ql:
            print(f"  {cname}: {dict(concept.properties)}")
