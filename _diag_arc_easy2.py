#!/usr/bin/env python3
"""Quick debug: test specific failing questions."""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.WARNING)
os.environ["L104_QUIET"] = "1"

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
engine = CommonsenseReasoningEngine()

tests = [
    {
        "q": "What causes sound?",
        "choices": ["sunlight", "vibrations", "x-rays", "pitch"],
        "expected": 1,
    },
    {
        "q": "Marshall learned that plants need a substance from the air to make their own food. What does a plant take from the air in order to make food?",
        "choices": ["carbon dioxide", "hydrogen", "nitrogen", "oxygen"],
        "expected": 0,
    },
    {
        "q": "During science class, a teacher explains that the samples the students are studying are made of two or more minerals. What is the teacher describing?",
        "choices": ["gases", "rocks", "elements", "molecules"],
        "expected": 1,
    },
    {
        "q": "In the United States, windmills have been used for farming for hundreds of years. Which simple machine is the most important part of a windmill?",
        "choices": ["lever", "pulley", "inclined plane", "wheel and axle"],
        "expected": 3,
    },
    {
        "q": "Antarctica is a continent at Earth's south pole. Fossils of tropical fern plants were discovered in Antarctica even though tropical fern plants only grow in warm areas. Which statement best explains the presence of fossils of tropical fern plants in Antarctica?",
        "choices": ["Millions of years ago, Antarctica was in a warmer location on Earth.",
                     "Recently, a natural disaster killed all of the fern plants in Antarctica.",
                     "Recently, birds stopped carrying the seeds of fern plants to Antarctica.",
                     "Millions of years ago, Earth's south pole was much colder than it is today."],
        "expected": 0,
    },
]

correct = 0
for t in tests:
    result = engine.answer_mcq(t['q'], t['choices'])
    got = result.get('selected_index', -1)
    ok = got == t['expected']
    correct += ok
    status = '[OK]  ' if ok else '[FAIL]'
    print(f"{status} Q: {t['q'][:80]}")
    print(f"       Exp={t['expected']}({t['choices'][t['expected']][:30]}) Got={got}({t['choices'][got][:30] if 0<=got<len(t['choices']) else '?'})")
print(f"\n{correct}/{len(tests)} correct")
