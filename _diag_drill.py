#!/usr/bin/env python3
"""Deep drill into water-freeze and tides failures."""
import re, sys
sys.stderr = open('/dev/null', 'w')

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
eng = CommonsenseReasoningEngine()
eng.initialize()

# Drill 1: Water freeze - why does "volume" concept inflate D?
print("=== WATER FREEZE DRILL ===")
q = "what happens to water when it freezes?"
vol_concept = eng.ontology.concepts.get('volume')
freeze_concept = eng.ontology.concepts.get('freeze')
water_concept = eng.ontology.concepts.get('water')
if vol_concept:
    print(f"'volume' concept: cat={vol_concept.category}, props={dict(list(vol_concept.properties.items())[:5])}")
else:
    print("'volume' concept: NOT FOUND")
if freeze_concept:
    print(f"'freeze' concept: NOT FOUND AS 'freeze', checking 'freezing'...")
freeze_concept = eng.ontology.concepts.get('freezing')
if freeze_concept:
    print(f"'freezing' concept: props={dict(freeze_concept.properties.items())}")
if water_concept:
    print(f"'water' concept: props={dict(list(water_concept.properties.items())[:8])}")

# Check what concept properties contain "volume"
print("\nConcepts mentioning 'volume':")
for key, concept in eng.ontology.concepts.items():
    props_str = str(concept.properties).lower()
    if 'volume' in props_str:
        print(f"  {key}: {props_str[:120]}")

# Check what concept properties contain "expand"
print("\nConcepts mentioning 'expand':")
for key, concept in eng.ontology.concepts.items():
    props_str = str(concept.properties).lower()
    if 'expand' in props_str:
        print(f"  {key}: {props_str[:120]}")

# Drill 2: Tides - why rotation inflates so much
print("\n=== TIDES DRILL ===")
earth = eng.ontology.concepts.get('earth')
if earth:
    print(f"'earth' concept properties:")
    for k, v in earth.properties.items():
        print(f"  {k}: {v}")
    print(f"  related: {earth.related}")
    print(f"  parts: {earth.parts}")

rotation = eng.ontology.concepts.get('rotation')
if rotation:
    print(f"'rotation' concept: {rotation.properties}")
else:
    print("'rotation' concept: NOT FOUND")

moon = eng.ontology.concepts.get('moon')
if moon:
    print(f"'moon' concept properties:")
    for k, v in moon.properties.items():
        print(f"  {k}: {v}")

gravity = eng.ontology.concepts.get('gravity')
if gravity:
    print(f"'gravity' concept properties:")
    for k, v in gravity.properties.items():
        print(f"  {k}: {v}")

# Check what choice words match which concepts
print("\nChoice 'Earth's rotation' matching:")
for key in ['earth', 'rotation']:
    c = eng.ontology.concepts.get(key)
    if c:
        props_str = str(c.properties).lower()
        for w in ['rotation', 'rotate', 'tide', 'tides', 'moon', 'gravity', 'earth']:
            if w in props_str:
                print(f"  '{w}' found in {key} properties")

print("\nChoice 'Moon's gravity' matching:")
for key in ['moon', 'gravity']:
    c = eng.ontology.concepts.get(key)
    if c:
        props_str = str(c.properties).lower()
        for w in ['tide', 'tides', 'cause', 'causes', 'moon', 'gravity', 'earth']:
            if w in props_str:
                print(f"  '{w}' found in {key} properties")
