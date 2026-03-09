#!/usr/bin/env python3
"""Quick verification of Round 1 upgrades."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l104_god_code_simulator.optimizer import AdaptiveOptimizer

opt = AdaptiveOptimizer()

# Test 1: Zero noise — all strategies should be ~1.0
print("=== ZERO NOISE TEST ===")
res = opt.optimize_noise_resilience(nq=2, noise_level=0.0)
all_ok = True
for s in res["strategies"]:
    ok = s["fidelity"] > 0.999
    if not ok:
        all_ok = False
    tag = "OK" if ok else "FAIL"
    print(f"  [{tag}] {s['strategy']:25s}: {s['fidelity']:.10f}")
print(f"  {'PASS' if all_ok else 'FAIL'}: Zero noise identity preservation")

# Test 2: Optimizer convergence
print("\n=== OPTIMIZER CONVERGENCE ===")
for nq in [2, 3, 4]:
    opt2 = AdaptiveOptimizer(target_fidelity=0.99, max_iterations=200)
    r2 = opt2.optimize_sacred_circuit(nq=nq, depth=2)
    print(f"  {nq}Q: best={r2['best_fidelity']:.6f}, conv={r2['converged']}, "
          f"params=depth={r2['best_params']['depth']}, "
          f"phase={r2['best_params']['phase_scale']:.4f}")

# Test 3: Protection strategy differentiation
print("\n=== PROTECTION STRATEGIES ===")
for noise in [0.1, 0.3, 0.5]:
    res2 = opt.optimize_noise_resilience(nq=3, noise_level=noise)
    fids = [s["fidelity"] for s in res2["strategies"]]
    spread = max(fids) - min(fids)
    print(f"  noise={noise}: best={res2['best_strategy']}, "
          f"spread={spread:.4f}, fids=[{', '.join(f'{f:.4f}' for f in fids)}]")

# Test 4: Full suite quick check
print("\n=== FULL SUITE CHECK ===")
from l104_god_code_simulator import god_code_simulator
report = god_code_simulator.run_all()
print(f"  {report['passed']}/{report['total']} pass ({report['pass_rate']*100:.1f}%)")
if report["failed"] > 0:
    for r in report["results"]:
        if not r.passed:
            print(f"  FAIL: {r.name} — {r.detail[:80]}")

print(f"\nDONE")
