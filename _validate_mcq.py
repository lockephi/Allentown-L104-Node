#!/usr/bin/env python3
"""Quick MCQ validation — 8 core questions."""
from l104_asi.language_comprehension import LanguageComprehensionEngine

lce = LanguageComprehensionEngine()

mcqs = [
    ("What is the chemical symbol for gold?", ["Au", "Ag", "Fe", "Cu"], "Au"),
    ("Which planet is known as the Red Planet?", ["Venus", "Mars", "Jupiter", "Saturn"], "Mars"),
    ("What is the powerhouse of the cell?", ["Nucleus", "Ribosome", "Mitochondria", "Golgi apparatus"], "Mitochondria"),
    ("Who wrote Romeo and Juliet?", ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"], "William Shakespeare"),
    ("What is the largest organ in the human body?", ["Heart", "Liver", "Skin", "Brain"], "Skin"),
    ("What gas do plants absorb from the atmosphere?", ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"], "Carbon dioxide"),
    ("What is the most abundant gas in Earth atmosphere?", ["Oxygen", "Carbon dioxide", "Nitrogen", "Argon"], "Nitrogen"),
    ("What is the speed of light in vacuum approximately?", ["300,000 km/s", "150,000 km/s", "450,000 km/s", "600,000 km/s"], "300,000 km/s"),
]

correct = 0
for q, choices, ans in mcqs:
    r = lce.mcq_solver.solve(q, choices)
    pick = r.get("answer_text", "")
    ok = pick == ans
    correct += int(ok)
    mark = "OK" if ok else "FAIL"
    lbl = r.get("answer", "?")
    conf = r.get("confidence", 0)
    print(f"[{mark}] {q[:50]:50s} => {lbl}={pick} (conf={conf})")
print(f"\nTotal: {correct}/{len(mcqs)}")
