#!/usr/bin/env python3
"""L104 God Code Simulator — Final Verification Report"""
import time
from l104_god_code_simulator import god_code_simulator

print("=" * 70)
print("L104 GOD CODE SIMULATOR — FINAL VERIFICATION")
print("=" * 70)

# 1. Run all simulations
t0 = time.time()
report = god_code_simulator.run_all()
elapsed = time.time() - t0

passed_n = report["passed"]
failed_n = report["failed"]
total = report["total"]
results = report["results"]  # list of SimulationResult

print(f"\n[SIMULATIONS] {passed_n}/{total} passed, {failed_n} failed  ({elapsed:.1f}s)")
if failed_n:
    print("  FAILURES:")
    for r in results:
        if not r.passed:
            print(f"    - {r.name}: {r.detail}")

# 2. Category breakdown
cats = report.get("categories", {})
print("\n[CATEGORIES]")
for cat, info in sorted(cats.items()):
    p, t = info["passed"], info["total"]
    status = "PASS" if p == t else "PARTIAL"
    print(f"  {cat:20s}: {p}/{t} {status}")

# 3. Performance (top 5 slowest)
times_list = [(r.name, r.elapsed_ms) for r in results]
times_list.sort(key=lambda x: -x[1])
print("\n[PERFORMANCE] Top 5 slowest:")
for name, ms in times_list[:5]:
    print(f"  {name:35s}: {ms:8.1f}ms")
print(f"  Total wall time: {elapsed:.1f}s")

# 4. 15D scoring
print("\n[15D SCORING]")
dims = god_code_simulator.score_dimensions()
total_score = 0
for dim, score in sorted(dims.items()):
    bar = "#" * int(score * 40)
    total_score += score
    print(f"  {dim:20s}: {score:.4f}  {bar}")
avg = total_score / len(dims) if dims else 0
print(f"  {'─' * 32}")
print(f"  {'AVERAGE':20s}: {avg:.4f}")

# 5. Parametric sweep
print("\n[PARAMETRIC SWEEP]")
try:
    sweep = god_code_simulator.parametric_sweep("dial_a", start=0, stop=4)
    print(f"  dial_a sweep: {len(sweep)} points")
    if sweep:
        errors = [s.get("error", 0) for s in sweep]
        g_vals = [s.get("G", 0) for s in sweep]
        all_pass = all(s.get("passed", False) for s in sweep)
        print(f"  G range: {min(g_vals):.4f} — {max(g_vals):.4f}")
        print(f"  max error: {max(errors):.2e}")
        print(f"  all conserved: {all_pass}")
except Exception as e:
    print(f"  sweep error: {e}")

# 6. Adaptive optimize
print("\n[ADAPTIVE OPTIMIZE]")
try:
    opt = god_code_simulator.adaptive_optimize(target_fidelity=0.99, nq=2, depth=3)
    print(f"  converged: {opt.get('converged', '?')}")
    print(f"  best fidelity: {opt.get('best_fidelity', 0):.6f}")
    print(f"  iterations: {opt.get('iterations', '?')}")
except Exception as e:
    print(f"  optimize error: {e}")

# 7. Feedback loop
print("\n[FEEDBACK LOOP]")
try:
    fb = god_code_simulator.run_feedback_loop(iterations=3)
    print(f"  iterations: {fb.get('iterations', '?')}")
    print(f"  avg composite score: {fb.get('avg_composite_score', 0):.4f}")
    print(f"  avg coherence: {fb.get('avg_coherence', 0):.4f}")
    print(f"  avg demon efficiency: {fb.get('avg_demon_efficiency', 0):.4f}")
    print(f"  score std: {fb.get('score_std', 0):.4f}")
    print(f"  converging: {fb.get('converging', '?')}")
    print(f"  engines connected: {fb.get('engines_connected', {})}")
except Exception as e:
    print(f"  feedback error: {e}")

print(f"\n{'=' * 70}")
rate = passed_n / total * 100 if total else 0
print(f"FINAL: {passed_n}/{total} PASS ({rate:.1f}%) | 15D avg={avg:.4f} | {elapsed:.1f}s")
print("=" * 70)
