#!/usr/bin/env python3
"""Diagnostic: analyze MMLU/ARC scoring pipeline issues."""
import sys, os, json, time

# ── Part 1: KB Coverage ──
print("=" * 70)
print("PART 1: Knowledge Base Coverage Analysis")
print("=" * 70)

from l104_asi.knowledge_data import KNOWLEDGE_NODES, CROSS_SUBJECT_RELATIONS
from collections import Counter

total_nodes = len(KNOWLEDGE_NODES)
total_facts = sum(len(n.get("facts", [])) for n in KNOWLEDGE_NODES)
subjects = set(n["subject"] for n in KNOWLEDGE_NODES)
subj_counts = Counter(n["subject"] for n in KNOWLEDGE_NODES)

print(f"Total knowledge nodes: {total_nodes}")
print(f"Total facts: {total_facts}")
print(f"Subjects covered: {len(subjects)}")
print(f"Cross-subject relations: {len(CROSS_SUBJECT_RELATIONS)}")
print()
print("Top 15 subjects by node count:")
for s, c in subj_counts.most_common(15):
    f_count = sum(len(n.get("facts", [])) for n in KNOWLEDGE_NODES if n["subject"] == s)
    print(f"  {s}: {c} nodes, {f_count} facts")

# Known MMLU subjects from harness
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_physics", "computer_security", "conceptual_physics",
    "electrical_engineering", "elementary_mathematics", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_mathematics", "high_school_physics", "high_school_statistics",
    "machine_learning", "medical_genetics", "formal_logic",
    "high_school_european_history", "high_school_us_history",
    "high_school_world_history", "international_law", "jurisprudence",
    "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
    "prehistory", "professional_law", "world_religions",
    "econometrics", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_microeconomics", "high_school_psychology", "human_sexuality",
    "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy", "business_ethics", "clinical_knowledge",
    "college_medicine", "global_facts", "human_aging", "management",
    "marketing", "miscellaneous", "nutrition", "professional_accounting",
    "professional_medicine", "virology",
]

missing = [s for s in MMLU_SUBJECTS if s not in subjects]
print(f"\nMMLU subjects NOT in KB ({len(missing)}/{len(MMLU_SUBJECTS)}):")
for s in sorted(missing):
    print(f"  - {s}")

# ── Part 2: Sample MMLU questions ──
print("\n" + "=" * 70)
print("PART 2: MMLU Sample Question Scoring Analysis")
print("=" * 70)

# Simple test questions with known answers to trace scoring behavior
test_questions = [
    {
        "question": "What is the SI unit of force?",
        "choices": ["watt", "joule", "newton", "pascal"],
        "correct": 2,  # newton
        "subject": "college_physics",
    },
    {
        "question": "Newton's second law of motion states that F equals",
        "choices": ["mv", "ma", "mg", "mc²"],
        "correct": 1,  # ma
        "subject": "college_physics",
    },
    {
        "question": "The powerhouse of the cell is the",
        "choices": ["nucleus", "ribosome", "mitochondria", "endoplasmic reticulum"],
        "correct": 2,  # mitochondria
        "subject": "college_biology",
    },
    {
        "question": "Which philosopher wrote 'The Republic'?",
        "choices": ["Aristotle", "Socrates", "Plato", "Kant"],
        "correct": 2,  # Plato
        "subject": "philosophy",
    },
    {
        "question": "What is the chemical formula for water?",
        "choices": ["CO2", "H2O", "NaCl", "O2"],
        "correct": 1,  # H2O
        "subject": "college_chemistry",
    },
    {
        "question": "The speed of light in vacuum is approximately",
        "choices": ["3×10⁶ m/s", "3×10⁸ m/s", "3×10¹⁰ m/s", "3×10⁴ m/s"],
        "correct": 1,  # 3×10⁸ m/s
        "subject": "college_physics",
    },
    {
        "question": "In economics, what does GDP stand for?",
        "choices": ["Gross Domestic Product", "General Development Program",
                    "Government Debt Portfolio", "Global Distribution Plan"],
        "correct": 0,  # Gross Domestic Product
        "subject": "high_school_macroeconomics",
    },
    {
        "question": "Which planet is closest to the Sun?",
        "choices": ["Venus", "Mercury", "Mars", "Earth"],
        "correct": 1,  # Mercury
        "subject": "astronomy",
    },
]

from l104_asi.language_comprehension import LanguageComprehensionEngine

print("Initializing LanguageComprehensionEngine...")
t0 = time.time()
engine = LanguageComprehensionEngine()
print(f"  Init took {time.time()-t0:.1f}s")

correct = 0
total = len(test_questions)

for tq in test_questions:
    result = engine.answer_mcq(tq["question"], tq["choices"], subject=tq["subject"])
    selected = result.get("selected_index", result.get("answer_index", -1))
    is_correct = selected == tq["correct"]
    if is_correct:
        correct += 1

    print(f"\nQ: {tq['question']}")
    print(f"  Choices: {tq['choices']}")
    print(f"  Correct: {chr(65+tq['correct'])} ({tq['choices'][tq['correct']]})")
    print(f"  Selected: {chr(65+selected)} ({tq['choices'][selected] if 0 <= selected < len(tq['choices']) else '?'})" +
          (" ✓" if is_correct else " ✗"))
    print(f"  Confidence: {result.get('confidence', 'N/A')}")
    if "all_scores" in result:
        for s in result["all_scores"]:
            marker = "←" if s.get("index") == tq["correct"] else ""
            print(f"    {s['label']}: {s['score']:.4f} {marker}")

print(f"\nMMLU sample accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

# ── Part 3: ARC Sample Questions ──
print("\n" + "=" * 70)
print("PART 3: ARC Sample Question Scoring Analysis")
print("=" * 70)

arc_questions = [
    {
        "question": "Which of the following is a renewable source of energy?",
        "choices": ["coal", "natural gas", "solar", "petroleum"],
        "correct": 2,  # solar
    },
    {
        "question": "What force keeps the Earth in orbit around the Sun?",
        "choices": ["friction", "gravity", "magnetism", "electricity"],
        "correct": 1,  # gravity
    },
    {
        "question": "Which organ in the human body pumps blood?",
        "choices": ["liver", "brain", "heart", "kidney"],
        "correct": 2,  # heart
    },
    {
        "question": "What happens to water when it is heated to 100 degrees Celsius?",
        "choices": ["it freezes", "it evaporates", "it boils", "it condenses"],
        "correct": 2,  # it boils
    },
    {
        "question": "Which of the following is NOT a state of matter?",
        "choices": ["solid", "liquid", "energy", "gas"],
        "correct": 2,  # energy
    },
    {
        "question": "Photosynthesis takes place in which part of a plant?",
        "choices": ["roots", "stem", "leaves", "flowers"],
        "correct": 2,  # leaves
    },
]

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine

print("Initializing CommonsenseReasoningEngine...")
t0 = time.time()
arc_engine = CommonsenseReasoningEngine()
print(f"  Init took {time.time()-t0:.1f}s")

arc_correct = 0
arc_total = len(arc_questions)

for tq in arc_questions:
    result = arc_engine.answer_mcq(tq["question"], tq["choices"])
    selected = result.get("selected_index", result.get("answer_index", -1))
    is_correct = selected == tq["correct"]
    if is_correct:
        arc_correct += 1

    print(f"\nQ: {tq['question']}")
    print(f"  Correct: {chr(65+tq['correct'])} ({tq['choices'][tq['correct']]})")
    print(f"  Selected: {chr(65+selected)} ({tq['choices'][selected] if 0 <= selected < len(tq['choices']) else '?'})" +
          (" ✓" if is_correct else " ✗"))
    print(f"  Confidence: {result.get('confidence', 'N/A')}")
    if "all_scores" in result:
        for s in result["all_scores"]:
            marker = "←" if s.get("index") == tq["correct"] else ""
            print(f"    {s['label']}: {s['score']:.4f} {marker}")

print(f"\nARC sample accuracy: {arc_correct}/{arc_total} ({100*arc_correct/arc_total:.1f}%)")
print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
