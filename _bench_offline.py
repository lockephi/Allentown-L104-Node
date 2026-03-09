#!/usr/bin/env python3
"""Quick offline MMLU+ARC benchmark — uses hardcoded samples + extended test set.
No HuggingFace access required. Tests scoring pipeline fixes."""
import json, os, sys, time, re
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import logging
logging.disable(logging.CRITICAL)

import io
_old = sys.stderr
sys.stderr = io.StringIO()

print("=" * 60, flush=True)
print("  Offline MMLU+ARC Scoring Validation", flush=True)
print("=" * 60, flush=True)
print("[INIT] Loading engines...", flush=True)
t0 = time.time()

from l104_asi.language_comprehension import LanguageComprehensionEngine
from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine

sys.stderr = _old
print(f"[INIT] Engines loaded in {time.time()-t0:.0f}s\n", flush=True)

# ═══ MMLU Extended Test Set ═══
mmlu_questions = [
    # STEM
    {"q": "What is the SI unit of force?", "c": ["watt", "joule", "newton", "pascal"], "a": 2, "s": "physics"},
    {"q": "Newton's second law of motion states that F equals", "c": ["mv", "ma", "mg", "mc²"], "a": 1, "s": "physics"},
    {"q": "The powerhouse of the cell is the", "c": ["nucleus", "ribosome", "mitochondria", "endoplasmic reticulum"], "a": 2, "s": "biology"},
    {"q": "Which philosopher wrote 'The Republic'?", "c": ["Aristotle", "Socrates", "Plato", "Kant"], "a": 2, "s": "philosophy"},
    {"q": "What is the chemical formula for water?", "c": ["CO2", "H2O", "NaCl", "O2"], "a": 1, "s": "chemistry"},
    {"q": "The speed of light in vacuum is approximately", "c": ["3×10⁶ m/s", "3×10⁸ m/s", "3×10¹⁰ m/s", "3×10⁴ m/s"], "a": 1, "s": "physics"},
    {"q": "In economics, what does GDP stand for?", "c": ["Gross Domestic Product", "General Development Program", "Government Debt Portfolio", "Global Distribution Plan"], "a": 0, "s": "economics"},
    {"q": "Which planet is closest to the Sun?", "c": ["Venus", "Mercury", "Mars", "Earth"], "a": 1, "s": "astronomy"},
    {"q": "What is the derivative of x^2 with respect to x?", "c": ["x", "2x", "x^2", "2"], "a": 1, "s": "mathematics"},
    {"q": "What is the sum of the angles in a triangle?", "c": ["90 degrees", "180 degrees", "270 degrees", "360 degrees"], "a": 1, "s": "mathematics"},
    {"q": "Which data structure uses FIFO ordering?", "c": ["Stack", "Queue", "Tree", "Graph"], "a": 1, "s": "computer_science"},
    {"q": "What molecule carries genetic information in most organisms?", "c": ["RNA", "Protein", "DNA", "Lipid"], "a": 2, "s": "biology"},
    {"q": "What is the time complexity of binary search?", "c": ["O(1)", "O(log n)", "O(n)", "O(n log n)"], "a": 1, "s": "computer_science"},
    {"q": "In what year did World War II end?", "c": ["1943", "1944", "1945", "1946"], "a": 2, "s": "history"},
    {"q": "Pavlov's experiments on dogs demonstrated which type of learning?", "c": ["Operant conditioning", "Classical conditioning", "Observational learning", "Habituation"], "a": 1, "s": "psychology"},
    # Harder STEM
    {"q": "What is the pH of a neutral solution at 25°C?", "c": ["0", "7", "14", "1"], "a": 1, "s": "chemistry"},
    {"q": "Which element has the atomic number 1?", "c": ["Helium", "Hydrogen", "Lithium", "Carbon"], "a": 1, "s": "chemistry"},
    {"q": "The Pythagorean theorem relates to which type of triangle?", "c": ["Equilateral", "Isosceles", "Right", "Scalene"], "a": 2, "s": "mathematics"},
    {"q": "DNA replication is described as", "c": ["conservative", "semi-conservative", "dispersive", "random"], "a": 1, "s": "biology"},
    {"q": "Which gas makes up approximately 78% of Earth's atmosphere?", "c": ["Oxygen", "Carbon dioxide", "Nitrogen", "Argon"], "a": 2, "s": "astronomy"},
    # Humanities
    {"q": "Who painted the Mona Lisa?", "c": ["Michelangelo", "Leonardo da Vinci", "Raphael", "Donatello"], "a": 1, "s": "art"},
    {"q": "The French Revolution began in which year?", "c": ["1776", "1789", "1804", "1815"], "a": 1, "s": "history"},
    {"q": "Who wrote 'Hamlet'?", "c": ["Christopher Marlowe", "William Shakespeare", "Ben Jonson", "John Milton"], "a": 1, "s": "literature"},
    {"q": "Which ancient civilization built the pyramids at Giza?", "c": ["Greek", "Roman", "Egyptian", "Persian"], "a": 2, "s": "history"},
    {"q": "The term 'cogito ergo sum' is attributed to", "c": ["Kant", "Descartes", "Hume", "Locke"], "a": 1, "s": "philosophy"},
    # Social Sciences
    {"q": "What is the basic unit of heredity?", "c": ["Chromosome", "Gene", "Cell", "Allele"], "a": 1, "s": "genetics"},
    {"q": "Supply and demand curves intersect at the", "c": ["maximum price", "equilibrium price", "minimum price", "average price"], "a": 1, "s": "economics"},
    {"q": "Maslow's hierarchy of needs places which need at the base?", "c": ["Safety", "Physiological", "Love", "Esteem"], "a": 1, "s": "psychology"},
    {"q": "The scientific method begins with", "c": ["hypothesis", "observation", "experiment", "conclusion"], "a": 1, "s": "science"},
    {"q": "What is the largest organ of the human body?", "c": ["Liver", "Brain", "Skin", "Heart"], "a": 2, "s": "anatomy"},
]

# ═══ ARC Test Set (commonsense science) ═══
arc_questions = [
    {"q": "Which of the following is a renewable source of energy?", "c": ["coal", "natural gas", "solar", "petroleum"], "a": 2},
    {"q": "What happens to water when it freezes?", "c": ["It contracts", "It expands", "It evaporates", "It stays the same"], "a": 1},
    {"q": "Which force keeps planets in orbit around the Sun?", "c": ["Friction", "Magnetism", "Gravity", "Electricity"], "a": 2},
    {"q": "What is the main function of the roots of a plant?", "c": ["Photosynthesis", "Absorbing water and nutrients", "Producing seeds", "Releasing oxygen"], "a": 1},
    {"q": "Sound travels fastest through which medium?", "c": ["Air", "Water", "Steel", "Vacuum"], "a": 2},
    {"q": "What process do plants use to convert sunlight into food?", "c": ["Respiration", "Fermentation", "Photosynthesis", "Decomposition"], "a": 2},
    {"q": "Which of the following is NOT a fossil fuel?", "c": ["Coal", "Natural gas", "Wind", "Oil"], "a": 2},
    {"q": "The moon's phases are caused by", "c": ["Earth's shadow", "the moon's rotation", "the moon's position relative to Earth and Sun", "solar eclipses"], "a": 2},
    {"q": "What type of rock is formed from cooled lava?", "c": ["Sedimentary", "Igneous", "Metamorphic", "Limestone"], "a": 1},
    {"q": "Which body system is responsible for fighting infections?", "c": ["Digestive", "Immune", "Nervous", "Skeletal"], "a": 1},
    {"q": "A food web shows", "c": ["the weather in an area", "how energy flows through an ecosystem", "the age of organisms", "population growth"], "a": 1},
    {"q": "What causes the seasons on Earth?", "c": ["Distance from the Sun", "Earth's tilted axis", "The moon's gravity", "Solar flares"], "a": 1},
    {"q": "Which state of matter has a definite shape and volume?", "c": ["Gas", "Liquid", "Solid", "Plasma"], "a": 2},
    {"q": "An animal that eats only plants is called", "c": ["Carnivore", "Herbivore", "Omnivore", "Decomposer"], "a": 1},
    {"q": "What is the chemical symbol for gold?", "c": ["Go", "Gd", "Au", "Ag"], "a": 2},
    {"q": "Which layer of Earth is the thinnest?", "c": ["Inner core", "Outer core", "Mantle", "Crust"], "a": 3},
    {"q": "Static electricity is caused by", "c": ["moving electrons", "transfer of electrons between objects", "magnetic fields", "nuclear reactions"], "a": 1},
    {"q": "What is the role of decomposers in an ecosystem?", "c": ["Produce food", "Break down dead organisms", "Pollinate flowers", "Regulate temperature"], "a": 1},
    {"q": "Which planet is known as the Red Planet?", "c": ["Jupiter", "Mars", "Venus", "Saturn"], "a": 1},
    {"q": "Condensation is the process of", "c": ["liquid turning to gas", "gas turning to liquid", "solid turning to liquid", "liquid turning to solid"], "a": 1},
]

# ═══ Run MMLU ═══
print(f"[MMLU] Testing {len(mmlu_questions)} questions...", flush=True)
lce = LanguageComprehensionEngine()
correct_mmlu = 0
for i, q in enumerate(mmlu_questions):
    try:
        result = lce.answer_mcq(q["q"], q["c"], subject=q.get("s", "unknown"))
        predicted = result.get("selected_index", result.get("answer_index", -1))
        ok = predicted == q["a"]
    except Exception as e:
        predicted = -1
        ok = False
    if ok:
        correct_mmlu += 1
    mark = "✓" if ok else "✗"
    expected_label = q["c"][q["a"]]
    predicted_label = q["c"][predicted] if 0 <= predicted < len(q["c"]) else "?"
    if not ok:
        print(f"  {mark} Q{i+1}: {q['q'][:60]}  expected={expected_label}  got={predicted_label}", flush=True)
    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(mmlu_questions)}] running accuracy: {correct_mmlu/(i+1)*100:.0f}%", flush=True)

mmlu_acc = correct_mmlu / len(mmlu_questions)
print(f"[MMLU] Result: {mmlu_acc*100:.1f}% ({correct_mmlu}/{len(mmlu_questions)})", flush=True)

# ═══ Run ARC ═══
print(f"\n[ARC] Testing {len(arc_questions)} questions...", flush=True)
cre = CommonsenseReasoningEngine()
correct_arc = 0
for i, q in enumerate(arc_questions):
    try:
        result = cre.answer_mcq(q["q"], q["c"])
        predicted = result.get("selected_index", result.get("answer_index", -1))
        ok = predicted == q["a"]
    except Exception as e:
        predicted = -1
        ok = False
    if ok:
        correct_arc += 1
    mark = "✓" if ok else "✗"
    expected_label = q["c"][q["a"]]
    predicted_label = q["c"][predicted] if 0 <= predicted < len(q["c"]) else "?"
    if not ok:
        print(f"  {mark} Q{i+1}: {q['q'][:60]}  expected={expected_label}  got={predicted_label}", flush=True)
    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(arc_questions)}] running accuracy: {correct_arc/(i+1)*100:.0f}%", flush=True)

arc_acc = correct_arc / len(arc_questions)
print(f"[ARC] Result: {arc_acc*100:.1f}% ({correct_arc}/{len(arc_questions)})", flush=True)

# ═══ Summary ═══
elapsed = time.time() - t0
print(f"\n{'='*60}", flush=True)
print(f"  MMLU: {mmlu_acc*100:5.1f}%  ({correct_mmlu}/{len(mmlu_questions)})  [was 29.6% online]", flush=True)
print(f"  ARC:  {arc_acc*100:5.1f}%  ({correct_arc}/{len(arc_questions)})  [was 28.7% online]", flush=True)
print(f"  Time: {elapsed:.0f}s", flush=True)
print(f"{'='*60}", flush=True)
