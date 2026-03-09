#!/usr/bin/env python3
"""Quick diagnostic: test specific ARC questions that should be easy."""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.WARNING)
os.environ["L104_QUIET"] = "1"

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine

engine = CommonsenseReasoningEngine()

tests = [
    {
        "q": "Plants use sunlight to make",
        "choices": ["carbon dioxide and water", "water and minerals", "sugar and oxygen", "soil and minerals"],
        "expected": 2,
    },
    {
        "q": "Where do plants get most of the energy they need to live and grow?",
        "choices": ["air", "soil", "water", "sunlight"],
        "expected": 3,
    },
    {
        "q": "Which best describes the structure of an atom?",
        "choices": ["a core of protons and neutrons surrounded by electrons", "a core of electrons and protons surrounded by neutrons", "a core of protons surrounded by neutrons and electrons", "a core of neutrons and electrons surrounded by protons"],
        "expected": 0,  # typically A
    },
    {
        "q": "Using a softball bat to hit a softball is an example of using which simple machine?",
        "choices": ["a pulley", "a lever", "an inclined plane", "a wheel and axle"],
        "expected": 1,
    },
    {
        "q": "Which process best explains how the Grand Canyon became so wide?",
        "choices": ["volcanic eruptions", "erosion", "earthquakes", "glaciation"],
        "expected": 1,
    },
    {
        "q": "At which temperature does water freeze?",
        "choices": ["0°C", "32°F", "100°C", "212°F"],
        "expected": 0,
    },
    {
        "q": "A dish of sugar water was left on a window sill. One week later, there were only sugar crystals left. What happened?",
        "choices": ["The sugar froze", "The water condensed", "The sugar melted", "The water evaporated"],
        "expected": 3,
    },
]

correct = 0
for t in tests:
    result = engine.answer_mcq(t["q"], t["choices"])
    predicted = result.get("answer_index", -1)
    ok = predicted == t["expected"]
    correct += int(ok)
    marker = "OK" if ok else "FAIL"
    print(f"[{marker}] Q: {t['q'][:60]}")
    print(f"       Exp={t['expected']} Got={predicted}  choice='{t['choices'][predicted][:40]}'")
    # Show scores
    scores = result.get("choice_scores", [])
    for cs in scores:
        print(f"         [{cs.get('index','-')}] {cs.get('score',0):.3f} '{cs.get('choice','')[:30]}'")
    print()

print(f"\n{correct}/{len(tests)} correct")
