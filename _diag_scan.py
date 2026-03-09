#!/usr/bin/env python3
"""Quick scan of Easy q100-299 and Challenge q50-149 failures."""
import requests, sys
sys.path.insert(0, ".")
from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
cr = CommonsenseReasoningEngine()
API = "https://datasets-server.huggingface.co/rows"

def scan(config, offset, length, label):
    p = {"dataset": "allenai/ai2_arc", "config": config, "split": "test", "offset": offset, "length": length}
    rows = requests.get(API, params=p, timeout=30).json().get("rows", [])
    correct = 0
    total = 0
    failures = []
    for i, row in enumerate(rows):
        r = row["row"]
        q = r["question"]; choices = r["choices"]["text"]; labels = r["choices"]["label"]
        expected = labels.index(r["answerKey"]) if r["answerKey"] in labels else -1
        result = cr.answer_mcq(q, choices)
        got = result.get("answer_index", -1)
        total += 1
        if got == expected:
            correct += 1
        else:
            failures.append((offset+i, q[:85], choices[expected][:45] if expected >= 0 else "?",
                           choices[got][:45] if 0 <= got < len(choices) else "?", expected, got))
    print(f"\n{label}: {correct}/{total} = {100*correct/total:.0f}%")
    print(f"Failures ({len(failures)}):")
    for idx, qs, exp, got, ei, gi in failures[:15]:
        print(f"  q{idx}: {qs}")
        print(f"    EXP[{ei}]: {exp}")
        print(f"    GOT[{gi}]: {got}")

scan("ARC-Easy", 200, 100, "Easy q200-299")
scan("ARC-Easy", 300, 100, "Easy q300-399")
scan("ARC-Challenge", 50, 100, "Challenge q50-149")
