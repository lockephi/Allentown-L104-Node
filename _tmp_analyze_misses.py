#!/usr/bin/env python3
"""Analyze missed benchmark questions to identify knowledge gaps."""
import json

with open("_bench_online_100_results.json") as f:
    data = json.load(f)

print("=== ARC MISSES ===")
arc_misses = [r for r in data["arc"]["results"] if not r["ok"]]
for m in arc_misses[:25]:
    q = m["question"][:130]
    got = m["got"]
    exp = m["expected"]
    print(f"  [{exp}->{got}] {q}")
print(f"Total ARC misses: {len(arc_misses)}/{len(data['arc']['results'])}")

print("\n=== MMLU MISSES ===")
mmlu_misses = [r for r in data["mmlu"]["results"] if not r["ok"]]
for m in mmlu_misses[:25]:
    q = m["question"][:130]
    got = m["got"]
    exp = m["expected"]
    subj = m.get("subject", "?")
    print(f"  [{exp}->{got}] ({subj}) {q}")
print(f"Total MMLU misses: {len(mmlu_misses)}/{len(data['mmlu']['results'])}")
