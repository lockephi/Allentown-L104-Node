#!/usr/bin/env python3
"""Quick verification of Round 2 upgrades."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l104_god_code_simulator.optimizer import AdaptiveOptimizer, _build_target_state, _evaluate_params

opt = AdaptiveOptimizer()

# Test 1: Composite metric ceiling fix
print("=== COMPOSITE METRIC ===")
for nq in [2, 3, 4]:
    target = _build_target_state(nq)
    c = _evaluate_params((1.0, 1.0, 1.0, 1.0), nq, target)
    print(f"  {nq}Q canonical: composite={c:.6f} (should be ~1.0)")

# Test 2: Zero noise — should still all be 1.0
print("\n=== ZERO NOISE ===")
res = opt.optimize_noise_resilience(nq=2, noise_level=0.0)
for s in res["strategies"]:
    tag = "OK" if s["fidelity"] > 0.999 else "FAIL"
    print(f"  [{tag}] {s['strategy']:25s}: {s['fidelity']:.10f}")

# Test 3: 3Q protection (bit_flip fix)
print("\n=== 3Q PROTECTION (noise=0.1) ===")
res2 = opt.optimize_noise_resilience(nq=3, noise_level=0.1)
for s in res2["strategies"]:
    tag = "OK" if s["fidelity"] > 0.5 else "LOW"
    print(f"  [{tag}] {s['strategy']:25s}: {s['fidelity']:.6f}")
print(f"  Best: {res2['best_strategy']} ({res2['best_fidelity']:.6f})")

# Test 4: Optimizer convergence with fixed metric
print("\n=== OPTIMIZER CONVERGENCE ===")
for nq in [2, 3, 4]:
    opt2 = AdaptiveOptimizer(target_fidelity=0.99, max_iterations=150)
    r = opt2.optimize_sacred_circuit(nq=nq, depth=2)
    print(f"  {nq}Q: best={r['best_fidelity']:.6f}, conv={r['converged']}, "
          f"params=d={r['best_params']['depth']}, ps={r['best_params']['phase_scale']:.4f}")

# Test 5: Strategy spread at multiple noise levels
print("\n=== STRATEGY SPREAD ===")
for noise in [0.1, 0.3, 0.5]:
    res3 = opt.optimize_noise_resilience(nq=3, noise_level=noise)
    fids = {s["strategy"]: s["fidelity"] for s in res3["strategies"]}
    spread = max(fids.values()) - min(fids.values())
    print(f"  noise={noise}: spread={spread:.4f}, best={res3['best_strategy']}")
    for name, f in fids.items():
        tag = "***" if name == res3["best_strategy"] else "   "
        print(f"    {tag} {name:25s}: {f:.6f}")

print("\nDONE")
