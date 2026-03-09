#!/usr/bin/env python3
"""Fast diagnostic: trace MCQ scoring behavior without heavy engine init."""
import sys, os, re, time, math

# Suppress logging noise
import logging
logging.disable(logging.CRITICAL)

print("=" * 70)
print("FAST DIAGNOSTIC: MCQ Scoring Trace")
print("=" * 70)

# Direct KB check
from l104_asi.knowledge_data import KNOWLEDGE_NODES, CROSS_SUBJECT_RELATIONS
print(f"KB: {len(KNOWLEDGE_NODES)} nodes, {sum(len(n.get('facts',[])) for n in KNOWLEDGE_NODES)} facts, {len(CROSS_SUBJECT_RELATIONS)} relations")

print("\nInitializing LanguageComprehensionEngine...")
t0 = time.time()
from l104_asi.language_comprehension import LanguageComprehensionEngine
engine = LanguageComprehensionEngine()
print(f"  LCE init: {time.time()-t0:.1f}s")

# Test questions covering different subjects
test_qs = [
    ("What is the SI unit of force?", ["watt", "joule", "newton", "pascal"], 2, "college_physics"),
    ("Newton's second law states F equals", ["mv", "ma", "mg", "mc squared"], 1, "college_physics"),
    ("The powerhouse of the cell is the", ["nucleus", "ribosome", "mitochondria", "endoplasmic reticulum"], 2, "college_biology"),
    ("What is the chemical formula for water?", ["CO2", "H2O", "NaCl", "O2"], 1, "college_chemistry"),
    ("Which planet is closest to the Sun?", ["Venus", "Mercury", "Mars", "Earth"], 1, "astronomy"),
    ("In economics, GDP stands for", ["Gross Domestic Product", "General Development Program", "Government Debt Portfolio", "Global Distribution Plan"], 0, "high_school_macroeconomics"),
    ("The largest organ of the human body is the", ["liver", "brain", "skin", "heart"], 2, "anatomy"),
    ("DNA stands for", ["deoxyribonucleic acid", "dinitrogen acid", "dynamic nuclear assembly", "dense nucleic alloy"], 0, "college_biology"),
]

print(f"\nTesting {len(test_qs)} MMLU questions...")
correct = 0
for q, choices, ans, subj in test_qs:
    t1 = time.time()
    result = engine.answer_mcq(q, choices, subject=subj)
    dt = time.time() - t1
    sel = result.get("selected_index", result.get("answer_index", -1))
    ok = sel == ans
    if ok:
        correct += 1

    print(f"\n{'✓' if ok else '✗'} [{dt:.1f}s] Q: {q}")
    print(f"  Correct={chr(65+ans)} Selected={chr(65+sel)} Conf={result.get('confidence', '?')}")
    if "all_scores" in result:
        for s in result["all_scores"]:
            mark = " ← correct" if s.get("index") == ans else ""
            sel_mark = " ★ selected" if s.get("index") == sel else ""
            print(f"    {s['label']}: {s['score']:.4f}{mark}{sel_mark}")

print(f"\nMMLU accuracy: {correct}/{len(test_qs)} ({100*correct/len(test_qs):.1f}%)")

# ARC test
print("\n" + "=" * 70)
print("ARC Questions")
print("=" * 70)

print("Initializing CommonsenseReasoningEngine...")
t0 = time.time()
from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
arc_engine = CommonsenseReasoningEngine()
print(f"  CRE init: {time.time()-t0:.1f}s")

arc_qs = [
    ("Which is a renewable source of energy?", ["coal", "natural gas", "solar", "petroleum"], 2),
    ("What force keeps Earth in orbit around the Sun?", ["friction", "gravity", "magnetism", "electricity"], 1),
    ("Which organ pumps blood?", ["liver", "brain", "heart", "kidney"], 2),
    ("What happens to water heated to 100 degrees Celsius?", ["freezes", "evaporates", "boils", "condenses"], 2),
    ("Photosynthesis occurs in which part of a plant?", ["roots", "stem", "leaves", "flowers"], 2),
    ("What do plants need to make food?", ["darkness", "sunlight", "cold", "wind"], 1),
]

arc_correct = 0
for q, choices, ans in arc_qs:
    t1 = time.time()
    result = arc_engine.answer_mcq(q, choices)
    dt = time.time() - t1
    sel = result.get("selected_index", result.get("answer_index", -1))
    ok = sel == ans
    if ok:
        arc_correct += 1

    print(f"\n{'✓' if ok else '✗'} [{dt:.1f}s] Q: {q}")
    print(f"  Correct={chr(65+ans)} Selected={chr(65+sel)} Conf={result.get('confidence', '?')}")
    if "all_scores" in result:
        for s in result["all_scores"]:
            mark = " ← correct" if s.get("index") == ans else ""
            sel_mark = " ★ selected" if s.get("index") == sel else ""
            print(f"    {s['label']}: {s['score']:.4f}{mark}{sel_mark}")

print(f"\nARC accuracy: {arc_correct}/{len(arc_qs)} ({100*arc_correct/len(arc_qs):.1f}%)")
print("\nDONE")
