#!/usr/bin/env python3
"""Run God Code Simulator — full diagnostic."""
from l104_god_code_simulator import god_code_simulator
import json, time

print("=== GOD CODE SIMULATOR STATUS ===")
status = god_code_simulator.get_status()
print(json.dumps(status, indent=2, default=str))

print("\n=== RUNNING ALL SIMULATIONS ===")
t0 = time.time()
report = god_code_simulator.run_all()
elapsed = time.time() - t0

print(f"\nTotal: {report['total']} | Passed: {report['passed']} | Failed: {report['failed']}")
print(f"Pass rate: {report['pass_rate']:.1%}")
print(f"Elapsed: {elapsed:.2f}s ({report['total_elapsed_ms']:.1f}ms internal)")

print("\n=== CATEGORY BREAKDOWN ===")
for cat, data in report["categories"].items():
    icon = "PASS" if data["passed"] == data["total"] else "WARN"
    print(f"  [{icon}] {cat}: {data['passed']}/{data['total']}")

print("\n=== FAILED SIMULATIONS ===")
failed = [r for r in report["results"] if not r.passed]
if not failed:
    print("  None - all passed!")
else:
    for r in failed:
        print(f"  FAIL [{r.category}] {r.name}: {r.detail[:150]}")

print("\n=== ALL RESULTS (with key metrics) ===")
for r in report["results"]:
    extra_str = ""
    if r.extra:
        key_items = {}
        for k, v in list(r.extra.items())[:3]:
            if isinstance(v, float):
                key_items[k] = round(v, 6)
            elif isinstance(v, (int, bool, str)):
                key_items[k] = v
        if key_items:
            extra_str = f" | {key_items}"
    icon = "PASS" if r.passed else "FAIL"
    print(f"  [{icon}] {r.name} ({r.elapsed_ms:.1f}ms) fidelity={r.fidelity:.4f}{extra_str}")

# Sweep tests
print("\n=== PARAMETRIC SWEEP TEST (dial_a) ===")
sweep = god_code_simulator.parametric_sweep("dial_a", start=0, stop=4)
if isinstance(sweep, list):
    print(f"  Sweep points: {len(sweep)}")
    for pt in sweep[:3]:
        if isinstance(pt, dict):
            print(f"    {pt}")
elif isinstance(sweep, dict):
    print(f"  {json.dumps(sweep, indent=2, default=str)[:300]}")

# Feedback test
print("\n=== FEEDBACK LOOP TEST ===")
fb = god_code_simulator.run_feedback_loop(iterations=3)
print(f"  Iterations: {fb.get('iterations', '?')}")
print(f"  Keys: {list(fb.keys())[:10]}")

# Score dimensions
print("\n=== SCORE DIMENSIONS (15D) ===")
dims = god_code_simulator.score_dimensions()
for dim, score in sorted(dims.items()):
    bar = "#" * int(score * 20) if isinstance(score, (int, float)) else ""
    print(f"  {dim:30s}: {score:.4f} {bar}" if isinstance(score, (int, float)) else f"  {dim}: {score}")

print("\n=== DONE ===")
