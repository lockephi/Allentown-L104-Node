#!/usr/bin/env python3
"""Read and summarize the intensive training report."""
import json

with open("intensive_training_report.json") as f:
    r = json.load(f)

print(f"initial_training_data: {r['initial_training_data']}")
print(f"start_time: {r['start_time']}")
print()

phases = r["phases"]
print(f"Phase keys: {list(phases.keys())}")

for pk, pv in phases.items():
    print(f"\n--- {pk} ---")
    if isinstance(pv, dict):
        for k2, v2 in pv.items():
            if isinstance(v2, list):
                print(f"  {k2}: list[{len(v2)}]")
            elif isinstance(v2, dict):
                print(f"  {k2}: dict{list(v2.keys())[:8]}")
            else:
                sv = str(v2)
                print(f"  {k2}: {sv[:200]}")
    else:
        print(f"  Value: {str(pv)[:200]}")

print()
print("=== FINAL ===")
final = r["final"]
if isinstance(final, dict):
    for k2, v2 in final.items():
        if isinstance(v2, list):
            print(f"  {k2}: list[{len(v2)}]")
        elif isinstance(v2, dict):
            print(f"  {k2}: dict{list(v2.keys())[:8]}")
        else:
            sv = str(v2)
            print(f"  {k2}: {sv[:200]}")
