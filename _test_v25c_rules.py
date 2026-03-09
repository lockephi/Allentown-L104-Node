#!/usr/bin/env python3
"""Quick test which rules fire for the 6 failing questions."""
import re, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Test each failure pattern
tests = [
    {
        "name": "Nitrogen/Chesapeake",
        "q": "Excess nitrogen fertilizers sometimes drain into waterways that flow into the Chesapeake Bay. This nitrogen may cause algae blooms, which reduce dissolved oxygen in the water. How does nitrogen negatively affect the Chesapeake Bay?",
        "choices": ["fish births increase", "fish populations decrease", "sediment on the bottom of the bay decreases", "the rate of water runoff into the bay increases"],
        "answer": 1,  # B
    },
    {
        "name": "H2O neutralization",
        "q": "Sodium bicarbonate (NaHCO_{3}) will neutralize stomach acid (HCl) in a double replacement reaction as follows: NaHCO_{3} + HCl -> NaCl + CO_{2} + \\Box What is the product necessary to make this reaction complete?",
        "choices": ["2HO", "HO_{2}", "H_{2}O_{2}", "H_{2}O"],
        "answer": 3,  # D
    },
    {
        "name": "Average speed",
        "q": "In one day, a family in a car rode for 2 hours, stopped for 3 hours, and then rode for another 5 hours. During the day, the family traveled a total distance of 400 kilometers. What was their average speed for the whole trip?",
        "choices": ["10 km/h", "20 km/h", "40 km/h", "50 km/h"],
        "answer": 2,  # C
    },
    {
        "name": "Energy transfer",
        "q": "All organisms depend on the transfer of energy to survive. Which best shows the energy transfer between animals in a shoreline ecosystem?",
        "choices": ["Fish -> Plants -> Birds", "Plants -> Birds -> Fish", "Plants -> Fish -> Birds", "Fish -> Birds -> Plants"],
        "answer": 2,  # C
    },
    {
        "name": "Recycle/reuse",
        "q": "In an experiment studying phototropism, students grew bean plants in labeled cardboard milk cartons. Afterward, the plants and soil were properly discarded. Which instruction line BEST conserves the remaining resources?",
        "choices": ["recycle the markers, reuse the milk cartons", "reuse the markers, discard the milk cartons", "discard the markers, reuse the milk cartons", "reuse the markers, recycle the milk cartons"],
        "answer": 3,  # D
    },
    {
        "name": "Work = F x d",
        "q": "Work is a product of force and distance. Which of the following is an example of work?",
        "choices": ["sitting at a desk", "pushing on a wall", "riding a bike", "reading a book"],
        "answer": 2,  # C
    },
    {
        "name": "Hybrid car braking (regression check)",
        "q": "A certain type of hybrid car utilizes a braking system in which energy is recovered during braking. This reclaims energy that was previously lost. This is an example of which energy conversion?",
        "choices": ["chemical energy converted to kinetic energy", "kinetic energy converted to potential energy", "thermal energy converted to kinetic energy", "kinetic energy converted to thermal energy"],
        "answer": 1,  # B
    },
    {
        "name": "Heating liquids (regression check)",
        "q": "A student heats the same amount of two different liquids over Bunsen burners. Each liquid is held in the same type of container and has the same starting temperature. Liquid A reaches the boiling point first. Compared with liquid A, liquid B will",
        "choices": ["evaporate sooner.", "take longer to increase in temperature.", "reach a higher boiling point.", "change color when heated."],
        "answer": 1,  # B
    },
]

# Now let's import and get the specific rules
sys.path.insert(0, '.')
import logging; logging.disable(logging.WARNING)
os.environ["L104_QUIET"] = "1"

# Load the rules directly
from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
engine = CommonsenseReasoningEngine()

# Find _SCIENCE_RULES from the _score_choice method source or just run the engine
for test in tests:
    print(f"\n{'='*70}")
    print(f"TEST: {test['name']}")
    print(f"Q: {test['q'][:80]}...")

    result = engine.answer_mcq(test['q'], test['choices'])
    scores = result.get('all_scores', {})

    labels = ['A', 'B', 'C', 'D']
    print(f"Expected: {labels[test['answer']]} = {test['choices'][test['answer']]}")

    # Sort by score
    scored = []
    for i, c in enumerate(test['choices']):
        label = labels[i]
        # Try different score formats
        sc = scores.get(label, scores.get(c, scores.get(i, 0)))
        scored.append((label, c, sc))
    scored.sort(key=lambda x: -x[2])

    for label, c, sc in scored:
        marker = " <<<" if labels[test['answer']] == label else ""
        print(f"  {label}: {c[:40]:40s} = {sc:.4f}{marker}")

    # Check if correct
    if scored[0][0] == labels[test['answer']]:
        print("  ✓ CORRECT")
    elif scored[0][2] == scored[1][2]:
        print(f"  ✗ TIED — got {scored[0][0]} instead of {labels[test['answer']]}")
    else:
        print(f"  ✗ WRONG — got {scored[0][0]} instead of {labels[test['answer']]}")
