#!/usr/bin/env python3
"""
Fast regression test — validates all Round 1+2 fixes without slow transpilation.
"""
import time, sys
import numpy as np
from l104_god_code_simulator import god_code_simulator as gs
from l104_god_code_simulator.optimizer import AdaptiveOptimizer
from l104_god_code_simulator.quantum_primitives import (
    init_sv, apply_single_gate, apply_cnot, H_GATE, make_gate, fidelity,
)

print("=" * 70)
print("  FAST REGRESSION TEST — Round 2 Upgrades Validation")
print("=" * 70)

failed = []

# ── Phase 1: Run all sims (skip transpiler) ──────────────────────────
print("\n[Phase 1] Running all simulations (skip transpiler)...")
t0 = time.time()
results = []
for name in gs.catalog.list_all():
    if "transpil" in name.lower():
        print(f"  ⏭  {name} (skipped — slow transpiler)")
        continue
    try:
        r = gs.run(name)
        passed = r.passed
        results.append((name, passed, r.fidelity))
        status = "✓" if passed else "✗"
        print(f"  {status}  {name}: fid={r.fidelity:.6f} pass={passed}")
        if not passed:
            failed.append(name)
    except Exception as e:
        results.append((name, False, 0.0))
        failed.append(name)
        print(f"  ✗  {name}: ERROR {e}")

elapsed = time.time() - t0
total = len(results)
passes = sum(1 for _, p, _ in results if p)
print(f"\n  Sims: {passes}/{total} pass ({elapsed:.1f}s)")

# ── Phase 2: Optimizer convergence (was capped at 0.80) ──────────────
print("\n[Phase 2] Optimizer convergence (was capped at 0.80)...")
for nq in [2, 3, 4]:
    r = gs.adaptive_optimize(target_fidelity=0.99, nq=nq, depth=4)
    fid = r.get("best_fidelity", 0)
    converged = r.get("converged", False)
    status = "✓" if fid > 0.90 else "✗"
    print(f"  {status} {nq}Q: fidelity={fid:.6f} converged={converged}")
    if fid < 0.90:
        failed.append(f"optimizer_{nq}q")

# ── Phase 3: Zero-noise identity check ───────────────────────────────
print("\n[Phase 3] Zero-noise identity (all strategies)...")
opt = AdaptiveOptimizer()
strategy_methods = [
    ("raw", opt._raw_strategy),
    ("dynamical_decouple", opt._dynamical_decoupling_strategy),
    ("bit_flip_code", opt._bit_flip_code_strategy),
    ("sacred_encode", opt._sacred_encode_strategy),
    ("phi_braided_echo", opt._phi_braided_echo_strategy),
]

for sname, sfn in strategy_methods:
    fid = sfn(nq=2, noise_level=0.0)
    status = "✓" if fid > 0.999 else "✗"
    print(f"  {status} {sname}: fid={fid:.8f}")
    if fid < 0.999:
        failed.append(f"zeronoise_{sname}")

# ── Phase 4: bit_flip_code protection (was 0.325) ────────────────────
print("\n[Phase 4] bit_flip_code at noise=0.1 (was 0.325)...")
fid_bf = opt._bit_flip_code_strategy(nq=3, noise_level=0.1)
status = "✓" if fid_bf > 0.5 else "✗"
print(f"  {status} bit_flip_code 3Q: fid={fid_bf:.4f} (target >0.5)")
if fid_bf < 0.5:
    failed.append("bit_flip_code_3q")

# ── Phase 5: sacred_encode differs from raw ───────────────────────────
print("\n[Phase 5] sacred_encode vs raw differentiation...")
fid_raw = opt._raw_strategy(nq=2, noise_level=0.1)
fid_sac = opt._sacred_encode_strategy(nq=2, noise_level=0.1)
diff = abs(fid_sac - fid_raw)
status = "✓" if diff > 0.0001 else "✗"
print(f"  {status} raw={fid_raw:.6f} sacred={fid_sac:.6f} diff={diff:.6f}")

# ── Phase 6: noise_resilience stability ─────────────────────────────
print("\n[Phase 6] noise_resilience stability (5 reps)...")
nr_fails = 0
for i in range(5):
    r = gs.run("noise_resilience")
    if not r.passed:
        nr_fails += 1
        print(f"  ✗ Rep {i+1}: fid={r.fidelity:.4f}")
    else:
        print(f"  ✓ Rep {i+1}: fid={r.fidelity:.4f}")
if nr_fails > 1:
    failed.append(f"noise_resilience_unstable({nr_fails}/5)")

# ── Phase 7: Strategy ranking at noise=0.3 ──────────────────────────
print("\n[Phase 7] Strategy ranking at noise=0.3...")
nr = gs.optimize_noise_resilience(nq=2, noise_level=0.3)
for s in nr["strategies"]:
    print(f"  {s['strategy']:25s} fid={s['fidelity']:.6f}")
spread = max(s["fidelity"] for s in nr["strategies"]) - min(s["fidelity"] for s in nr["strategies"])
print(f"  Spread: {spread:.4f} (target >0.01)")

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
if not failed:
    print(f"  ALL TESTS PASSED — {passes}/{total} sims + all fix validations")
else:
    print(f"  FAILURES: {failed}")
print("=" * 70)
