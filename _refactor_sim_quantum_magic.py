#!/usr/bin/env python3
"""
Refactor Simulation — l104_quantum_magic.py (5,725 lines)
Runs Code Engine: refactor analysis, smell detection, performance prediction,
dead code archaeology, and full analysis to find the best refactoring strategy.
"""

import json
import time
import sys

sys.path.insert(0, ".")
from l104_code_engine import code_engine

with open("l104_quantum_magic.py", "r") as f:
    source = f.read()

line_count = source.count("\n") + 1
print("=" * 70)
print(f"  REFACTOR SIMULATION — l104_quantum_magic.py ({line_count:,} lines)")
print("=" * 70)

results = {}
t_total = time.time()

# ── 1. Refactor Engine Analysis ──
print("\n[1/6] REFACTOR ENGINE ANALYSIS")
t = time.time()
try:
    refactor = code_engine.refactor_engine.refactor_analyze(source)
    results["refactor"] = refactor
    if isinstance(refactor, dict):
        for k, v in refactor.items():
            if isinstance(v, list):
                print(f"  {k}: {len(v)} items")
                for item in v[:8]:
                    print(f"    → {item}")
            elif isinstance(v, (int, float, str, bool)):
                print(f"  {k}: {v}")
    else:
        print(f"  Result: {str(refactor)[:500]}")
except Exception as e:
    print(f"  ERROR: {e}")
print(f"  [{time.time() - t:.2f}s]")

# ── 2. Code Smell Detection ──
print("\n[2/6] CODE SMELL DETECTION")
t = time.time()
try:
    smells = code_engine.smell_detector.detect_all(source)
    results["smells"] = smells
    if isinstance(smells, dict):
        for k, v in smells.items():
            if isinstance(v, list):
                print(f"  {k}: {len(v)} items")
                for item in v[:5]:
                    print(f"    → {item}")
            elif isinstance(v, (int, float, str, bool)):
                print(f"  {k}: {v}")
    else:
        print(f"  Result: {str(smells)[:500]}")
except Exception as e:
    print(f"  ERROR: {e}")
print(f"  [{time.time() - t:.2f}s]")

# ── 3. Performance Prediction ──
print("\n[3/6] PERFORMANCE PREDICTION")
t = time.time()
try:
    perf = code_engine.perf_predictor.predict_performance(source)
    results["performance"] = perf
    if isinstance(perf, dict):
        for k, v in perf.items():
            if isinstance(v, list):
                print(f"  {k}: {len(v)} items")
                for item in v[:5]:
                    if isinstance(item, dict):
                        print(f"    → {item.get('type', item.get('name', str(item)[:80]))}")
                    else:
                        print(f"    → {str(item)[:80]}")
            elif isinstance(v, dict):
                print(f"  {k}:")
                for sk, sv in list(v.items())[:8]:
                    print(f"    {sk}: {sv}")
            elif isinstance(v, (int, float, str, bool)):
                print(f"  {k}: {v}")
    else:
        print(f"  Result: {str(perf)[:500]}")
except Exception as e:
    print(f"  ERROR: {e}")
print(f"  [{time.time() - t:.2f}s]")

# ── 4. Dead Code Archaeology ──
print("\n[4/6] DEAD CODE ARCHAEOLOGY")
t = time.time()
try:
    dead = code_engine.excavator.excavate(source)
    results["dead_code"] = dead
    if isinstance(dead, dict):
        for k, v in dead.items():
            if isinstance(v, list):
                print(f"  {k}: {len(v)} items")
                for item in v[:5]:
                    if isinstance(item, dict):
                        print(f"    → {item.get('name', item.get('type', str(item)[:80]))}")
                    else:
                        print(f"    → {str(item)[:80]}")
            elif isinstance(v, (int, float, str, bool)):
                print(f"  {k}: {v}")
    else:
        print(f"  Result: {str(dead)[:500]}")
except Exception as e:
    print(f"  ERROR: {e}")
print(f"  [{time.time() - t:.2f}s]")

# ── 5. Full Code Analysis ──
print("\n[5/6] FULL CODE ANALYSIS")
t = time.time()
try:
    analysis = code_engine.full_analysis(source)
    results["full_analysis"] = analysis
    if isinstance(analysis, dict):
        for k, v in analysis.items():
            if isinstance(v, list):
                print(f"  {k}: {len(v)} items")
            elif isinstance(v, dict):
                # Print scalar values from dicts
                scalars = {sk: sv for sk, sv in v.items() if isinstance(sv, (int, float, str, bool))}
                if scalars:
                    print(f"  {k}: {scalars}")
                else:
                    print(f"  {k}: {{...}} ({len(v)} keys)")
            elif isinstance(v, (int, float, str, bool)):
                print(f"  {k}: {v}")
    else:
        print(f"  Result: {str(analysis)[:500]}")
except Exception as e:
    print(f"  ERROR: {e}")
print(f"  [{time.time() - t:.2f}s]")

# ── 6. Auto-Fix Analysis ──
print("\n[6/6] AUTO-FIX ANALYSIS")
t = time.time()
try:
    fixed, log = code_engine.auto_fix_code(source)
    results["auto_fix"] = {"log": log, "changed_lines": abs(fixed.count("\n") - source.count("\n"))}
    if isinstance(log, list):
        print(f"  Fixes applied: {len(log)}")
        for item in log[:10]:
            if isinstance(item, dict):
                print(f"    → {item.get('fix', item.get('name', str(item)[:80]))}")
            else:
                print(f"    → {str(item)[:80]}")
    elif isinstance(log, dict):
        for k, v in log.items():
            print(f"  {k}: {v}")
    else:
        print(f"  Log: {str(log)[:500]}")
except Exception as e:
    print(f"  ERROR: {e}")
print(f"  [{time.time() - t:.2f}s]")

total_time = time.time() - t_total
print(f"\n{'=' * 70}")
print(f"  SIMULATION COMPLETE — {total_time:.2f}s total")
print(f"{'=' * 70}")

# Save JSON report
report_path = "_refactor_sim_quantum_magic_report.json"
try:
    # Make serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)[:200]
    with open(report_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\n  Report saved: {report_path}")
except Exception as e:
    print(f"  Could not save report: {e}")
