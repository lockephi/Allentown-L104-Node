#!/usr/bin/env python3
"""Show full text of missed psychology and ARC questions."""
import json

with open("_bench_online_100_results.json") as f:
    d = json.load(f)

print("=== MMLU PSYCHOLOGY MISSES ===")
for r in d["mmlu"]["results"]:
    if "psychol" in r.get("subject", "") and not r["ok"]:
        q = r["question"][:250]
        print(f"Q{r['idx']:02d} [{r['got']}->{r['expected']}] {q}")
        print()

print("\n=== ARC MISSES ===")
for r in d["arc"]["results"]:
    if not r["ok"]:
        q = r["question"][:250]
        print(f"Q{r['idx']:02d} [{r['got']}->{r['expected']}] {q}")
        print()
