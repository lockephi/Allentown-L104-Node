#!/usr/bin/env python3
"""Debug fact table matching for specific questions."""
import os, re, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["L104_QUIET"] = "1"
import logging; logging.disable(logging.WARNING)
from l104_asi.commonsense_reasoning import CommonsenseMCQSolver

s = CommonsenseMCQSolver._stem_sc

print("=== Stemmer tests ===")
words = ['causes', 'cause', 'sound', 'vibrations', 'vibration',
         'minerals', 'mineral', 'rocks', 'rock', 'two', 'studying',
         'sunlight']
for w in words:
    print(f"  {w:15s} -> {s(w)}")

print("\n=== Matching debug: 'What causes sound?' ===")
q = "what causes sound"
q_words_set = set(re.findall(r'\w+', q.lower()))
q_stems_set = {s(w) for w in q_words_set if len(w) > 2}
print(f"Q words: {q_words_set}")
print(f"Q stems: {q_stems_set}")

entries = [
    (["cause", "sound"], ["vibration", "vibrate"], 4.0),
    (["causes", "sound"], ["vibration", "vibrate"], 4.0),
    (["sound", "cause"], ["vibration", "vibrate"], 4.0),
]
for q_pat, a_pat, boost in entries:
    q_hits = 0
    for w in q_pat:
        if w in q_words_set:
            q_hits += 1
            print(f"  exact match: {w}")
        elif len(w) > 2 and s(w) in q_stems_set:
            q_hits += 0.85
            print(f"  stem match: {w} -> {s(w)}")
        else:
            print(f"  NO match: {w} (stem={s(w)}, q_stems={q_stems_set})")
    q_ratio = q_hits / len(q_pat)
    print(f"  q_ratio={q_ratio:.2f} for pattern {q_pat}")

    # Check choices
    choices = ["sunlight", "vibrations", "x-rays", "pitch"]
    for c in choices:
        c_words = set(re.findall(r'\w+', c.lower()))
        c_stems = {s(w) for w in c_words if len(w) > 2}
        a_hits = 0
        for w in a_pat:
            if w in c_words:
                a_hits += 1
            elif len(w) > 2 and s(w) in c_stems:
                a_hits += 0.85
        if a_hits > 0:
            print(f"    choice '{c}': a_hits={a_hits}")
    print()

print("\n=== Matching debug: minerals/rocks ===")
q2 = "During science class, a teacher explains that the samples the students are studying are made of two or more minerals. What is the teacher describing?"
q2_words = set(re.findall(r'\w+', q2.lower()))
q2_stems = {s(w) for w in q2_words if len(w) > 2}
entries2 = [
    (["two", "mineral", "what"], ["rock"], 4.0),
    (["mineral", "describe", "sample"], ["rock"], 3.5),
    (["mineral", "studying"], ["rock"], 3.5),
]
for q_pat, a_pat, boost in entries2:
    q_hits = 0
    for w in q_pat:
        if w in q2_words:
            q_hits += 1
            print(f"  exact match: {w}")
        elif len(w) > 2 and s(w) in q2_stems:
            q_hits += 0.85
            print(f"  stem match: {w} -> {s(w)}")
        else:
            print(f"  NO match: {w} (stem={s(w)})")
    q_ratio = q_hits / len(q_pat)
    print(f"  q_ratio={q_ratio:.2f} for pattern {q_pat}\n")
