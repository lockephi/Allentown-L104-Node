#!/usr/bin/env python3
"""Diagnose what the ontology actually contains for key concepts."""
import os, logging
os.chdir(os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.WARNING)
os.environ["L104_QUIET"] = "1"

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
engine = CommonsenseReasoningEngine()
engine.initialize()

ontology = engine.mcq_solver.ontology
causal = engine.mcq_solver.causal

# Check key concepts
for name in ['water', 'boiling', 'temperature', 'sound', 'vibration',
             'photosynthesis', 'oxygen', 'atom', 'molecule', 'compound',
             'friction', 'energy', 'kinetic_energy', 'potential_energy',
             'cell', 'plant_cell', 'reptile', 'fish', 'acceleration',
             'force', 'solid', 'liquid', 'gas', 'erosion', 'weathering',
             'chemical_weathering', 'carbon', 'carbon_atom', 'motor',
             'electric_motor', 'light_bulb', 'inertia', 'gravity']:
    c = ontology.concepts.get(name)
    if c:
        props_str = str(c.properties)
        print(f"\n{'='*50}")
        print(f"CONCEPT: {name} (cat={c.category})")
        print(f"  Props: {props_str[:200]}")
    else:
        print(f"\n[MISSING] {name}")

# Check causal rules count and examples
print(f"\n{'='*50}")
print(f"Total causal rules: {len(causal.rules)}")
for i, rule in enumerate(causal.rules[:5]):
    print(f"  Rule {i}: {rule.condition} -> {rule.effect}")

# Check what concepts are available
all_concepts = list(ontology.concepts.keys())
print(f"\nTotal concepts: {len(all_concepts)}")
print(f"Sample: {all_concepts[:30]}")
