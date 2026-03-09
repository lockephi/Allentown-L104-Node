#!/usr/bin/env python3
"""Test v3.0 new strategies."""
from l104_god_code_simulator.optimizer import AdaptiveOptimizer
from l104_god_code_simulator import god_code_simulator as gs

opt = AdaptiveOptimizer()

print("=== Zero-noise identity test ===")
for name, fn in [("zeno_freeze", opt._zeno_freeze_strategy),
                 ("composite_shield", opt._composite_shield_strategy)]:
    fid = fn(nq=2, noise_level=0.0)
    print(f"  {name}: fid={fid:.8f} {'PASS' if fid > 0.999 else 'FAIL'}")

print("\n=== Noise=0.1 comparison ===")
for name, fn in [("raw", opt._raw_strategy),
                 ("dynamical_decouple", opt._dynamical_decoupling_strategy),
                 ("sacred_encode", opt._sacred_encode_strategy),
                 ("phi_braided_echo", opt._phi_braided_echo_strategy),
                 ("zeno_freeze", opt._zeno_freeze_strategy),
                 ("composite_shield", opt._composite_shield_strategy)]:
    fid = fn(nq=2, noise_level=0.1)
    print(f"  {name:25s}: fid={fid:.6f}")

print("\n=== Noise=0.5 comparison ===")
for name, fn in [("raw", opt._raw_strategy),
                 ("dynamical_decouple", opt._dynamical_decoupling_strategy),
                 ("sacred_encode", opt._sacred_encode_strategy),
                 ("phi_braided_echo", opt._phi_braided_echo_strategy),
                 ("zeno_freeze", opt._zeno_freeze_strategy),
                 ("composite_shield", opt._composite_shield_strategy)]:
    fid = fn(nq=2, noise_level=0.5)
    print(f"  {name:25s}: fid={fid:.6f}")

print("\n=== Full optimize_noise_resilience at noise=0.3 ===")
r = gs.optimize_noise_resilience(nq=2, noise_level=0.3)
for s in r["strategies"]:
    print(f"  {s['strategy']:25s} fid={s['fidelity']:.6f}")
print(f"\n  Best: {r['best_strategy']} (fid={r['best_fidelity']:.6f})")

print("\n=== Full regression: 35 sims (skip transpiler) ===")
import time
t0 = time.time()
fail = []
for name in gs.catalog.list_all():
    if "transpil" in name.lower():
        continue
    r = gs.run(name)
    if not r.passed:
        fail.append(name)
        print(f"  FAIL: {name} fid={r.fidelity:.4f}")
elapsed = time.time() - t0
total = len(gs.catalog.list_all()) - 2
print(f"  {total - len(fail)}/{total} pass ({elapsed:.1f}s)")
if fail:
    print(f"  FAILURES: {fail}")
else:
    print("  ALL PASS")
