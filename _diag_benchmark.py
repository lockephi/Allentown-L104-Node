#!/usr/bin/env python3
"""Quick diagnostic to understand MMLU and ARC scoring patterns."""
import os, sys, logging
logging.disable(logging.WARNING)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l104_asi.language_comprehension import LanguageComprehensionEngine
from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine

lce = LanguageComprehensionEngine()
cre = CommonsenseReasoningEngine()

# --- MMLU diagnostic ---
mmlu_samples = [
    ("What is the capital of France?", ["Berlin", "Paris", "London", "Madrid"], 1, "geography"),
    ("Which planet is closest to the Sun?", ["Venus", "Mercury", "Mars", "Earth"], 1, "astronomy"),
    ("What is the powerhouse of the cell?", ["Nucleus", "Ribosome", "Mitochondria", "Cell wall"], 2, "biology"),
    ("Water boils at what temperature at sea level?", ["50C", "100C", "150C", "200C"], 1, "physics"),
    ("Which gas do plants absorb?", ["Oxygen", "Carbon dioxide", "Nitrogen", "Hydrogen"], 1, "biology"),
    ("What is 2+2?", ["3", "4", "5", "6"], 1, "math"),
    ("Who painted the Mona Lisa?", ["Michelangelo", "Da Vinci", "Raphael", "Rembrandt"], 1, "art"),
    ("What is the largest ocean?", ["Atlantic", "Indian", "Pacific", "Arctic"], 2, "geography"),
]

print("=== MMLU DIAGNOSTIC ===")
mmlu_correct = 0
for q, choices, expected, subj in mmlu_samples:
    r = lce.answer_mcq(q, choices)
    pred = r.get("answer_index", r.get("selected_index", -1))
    ok = pred == expected
    if ok:
        mmlu_correct += 1
    scores = r.get("all_scores", {})
    khits = r.get("knowledge_hits", 0)
    print(f"  [{'OK' if ok else 'FAIL'}] {subj}: pred={pred} exp={expected} khits={khits} scores={scores}")

print(f"MMLU sample: {mmlu_correct}/{len(mmlu_samples)} ({100*mmlu_correct/len(mmlu_samples):.0f}%)")

# --- ARC diagnostic ---
arc_samples = [
    ("What happens when ice is heated?", ["It stays frozen", "It melts", "It gets colder", "It evaporates immediately"], 1, "physics"),
    ("Which is a renewable energy source?", ["Coal", "Oil", "Solar", "Natural gas"], 2, "energy"),
    ("What do plants need to make food?", ["Darkness", "Sunlight", "Only water", "Only soil"], 1, "biology"),
    ("Which force pulls objects toward Earth?", ["Friction", "Magnetism", "Gravity", "Wind"], 2, "physics"),
    ("What is the main gas in Earth atmosphere?", ["Oxygen", "Nitrogen", "Carbon dioxide", "Helium"], 1, "earth_sci"),
    ("A thermometer measures what?", ["Weight", "Speed", "Temperature", "Volume"], 2, "tools"),
    ("What causes day and night?", ["Moon orbit", "Earth rotation", "Sun movement", "Seasons"], 1, "astronomy"),
    ("Mammals are warm-blooded and have what?", ["Scales", "Feathers", "Fur or hair", "Shells"], 2, "biology"),
]

print()
print("=== ARC DIAGNOSTIC ===")
arc_correct = 0
for q, choices, expected, subj in arc_samples:
    r = cre.answer_mcq(q, choices)
    pred = r.get("answer_index", r.get("selected_index", -1))
    ok = pred == expected
    if ok:
        arc_correct += 1
    scores = r.get("all_scores", {})
    concepts = r.get("concepts_found", [])
    causal = r.get("causal_rules_used", 0)
    print(f"  [{'OK' if ok else 'FAIL'}] {subj}: pred={pred} exp={expected} concepts={concepts[:5]} causal={causal} scores={scores}")

print(f"ARC sample: {arc_correct}/{len(arc_samples)} ({100*arc_correct/len(arc_samples):.0f}%)")
