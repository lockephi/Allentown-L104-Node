#!/usr/bin/env python3
import json

with open("benchmark_online_results.json") as f:
    r = json.load(f)

print("MODE:", r["mode"])
print("\nSOURCES:")
for k, v in r["sources"].items():
    print(f"  {k}: {v}")

print("\nBENCHMARK SCORES:")
for k, v in r["benchmarks"].items():
    print(f"  {k}: {v['correct']}/{v['total']} = {v['score']*100:.1f}%")

print(f"\nCOMPOSITE: {r['composite_score']*100:.1f}%")
print(f"VERDICT: {r['verdict']}")
print(f"ELAPSED: {r['elapsed_seconds']}s")

dr = r.get("detailed_results", {})

mmlu = dr.get("MMLU", {})
if mmlu.get("by_subject"):
    print("\nMMLU BY SUBJECT:")
    for subj, score in sorted(mmlu["by_subject"].items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {subj}: {score*100:.0f}%")

math_r = dr.get("MATH", {})
if math_r.get("by_domain"):
    print("\nMATH BY DOMAIN:")
    for domain, score in sorted(math_r["by_domain"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {domain}: {score*100:.0f}%")

arc_r = dr.get("ARC", {})
if arc_r.get("by_category"):
    print("\nARC BY CATEGORY:")
    for cat, score in sorted(arc_r["by_category"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {score*100:.0f}%")

# HumanEval details
he = dr.get("HumanEval", {})
passed_list = [d for d in he.get("details", []) if d.get("passed")]
print(f"\nHUMANEVAL: {len(passed_list)}/{he.get('total', 0)} passed")
if passed_list[:5]:
    print("  Passed examples:", [p.get("task_id", p.get("entry_point", "?")) for p in passed_list[:10]])
