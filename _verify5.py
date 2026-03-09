#!/usr/bin/env python3
"""Quick verification of 5 ARC diagnostic questions."""
import sys, time, os
sys.stderr = open(os.devnull, 'w')

t0 = time.time()
print('Loading engine...', flush=True)
from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
eng = CommonsenseReasoningEngine()
eng.initialize()
print(f'Engine loaded in {time.time()-t0:.1f}s', flush=True)

questions = [
    ('Which planet is known as the Red Planet?',
     ['Mars', 'Venus', 'Jupiter', 'Saturn']),
    ('Sound travels fastest through which material?',
     ['Steel', 'Water', 'Air', 'Vacuum']),
    ('What type of consumer eats only plants?',
     ['Herbivore', 'Carnivore', 'Omnivore', 'Decomposer']),
    ('What causes the phases of the Moon?',
     ['The position of the Moon relative to the Earth and the Sun',
      'The shadow of the Earth on the Moon',
      'The distance of the Moon from the Sun',
      'The rotation of the Moon on its axis']),
    ('Which body system is mainly responsible for fighting infections?',
     ['Immune system', 'Digestive system', 'Nervous system', 'Skeletal system']),
]

labels = 'ABCDEFGHIJ'
correct_count = 0
for q, ch in questions:
    r = eng.mcq_solver.solve(q, ch)
    ans = r.get('answer', '?')
    correct_label = 'A'  # correct answer is always first choice
    ok = ans == correct_label
    if ok:
        correct_count += 1
    tag = 'CORRECT' if ok else 'WRONG'
    # Show what the answer maps to
    ans_idx = labels.index(ans) if ans in labels else -1
    ans_text = ch[ans_idx] if 0 <= ans_idx < len(ch) else '?'
    print(f'  {tag}: Q="{q[:55]}..."', flush=True)
    print(f'         Expected=A ({ch[0][:40]})', flush=True)
    print(f'         Got={ans} ({ans_text[:40]})', flush=True)

print(f'\nResult: {correct_count}/5 correct ({correct_count*100//5}%)')
print(f'Total time: {time.time()-t0:.1f}s')
