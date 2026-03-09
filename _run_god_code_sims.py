#!/usr/bin/env python3
"""Run all God Code Simulator simulations."""

from l104_god_code_simulator import god_code_simulator

print("=" * 70)
print("  L104 GOD CODE SIMULATOR - Full Simulation Run")
print("=" * 70)

# Run all 23 simulations across 4 categories
report = god_code_simulator.run_all()

# Summary header
print(f"\n  Version:   {report['version']}")
print(f"  Total:     {report['total']}")
print(f"  Passed:    {report['passed']}")
print(f"  Failed:    {report['failed']}")
print(f"  Pass Rate: {report['pass_rate']*100:.1f}%")
print(f"  Time:      {report['total_elapsed_ms']:.1f} ms")

# Category breakdown
print(f"\n{'='*70}")
print("  CATEGORY BREAKDOWN")
print(f"{'='*70}")
for cat, stats in report["categories"].items():
    icon = "OK" if stats["passed"] == stats["total"] else "!!"
    print(f"  [{icon}] {cat.upper():12s}  {stats['passed']}/{stats['total']} passed")

# Per-simulation results
print(f"\n{'='*70}")
print("  SIMULATION RESULTS")
print(f"{'='*70}")
results = report["results"]
for r in results:
    icon = "PASS" if r.passed else "FAIL"
    mark = "+" if r.passed else "X"
    line = f"  [{mark}] {icon} | {r.name:40s} | fidelity={r.fidelity:.4f}"
    if r.god_code_measured > 0:
        line += f"  god_code={r.god_code_measured:.4f}"
    if r.sacred_alignment > 0:
        line += f"  sacred={r.sacred_alignment:.4f}"
    if r.entanglement_entropy > 0:
        line += f"  entanglement={r.entanglement_entropy:.4f}"
    if r.elapsed_ms > 0:
        line += f"  ({r.elapsed_ms:.1f}ms)"
    print(line)
    if not r.passed and r.detail:
        print(f"        -> {r.detail[:100]}")

# Run parametric sweeps
print(f"\n\n{'='*70}")
print("  PARAMETRIC SWEEPS")
print(f"{'='*70}")

for sweep_name in ["dial_a", "noise", "depth"]:
    print(f"\n--- {sweep_name.upper()} Sweep ---")
    try:
        if sweep_name == "dial_a":
            sweep = god_code_simulator.parametric_sweep(sweep_name, start=0, stop=8)
        else:
            sweep = god_code_simulator.parametric_sweep(sweep_name)

        if isinstance(sweep, dict):
            for k, v in sweep.items():
                print(f"  {k}: {v}")
        elif isinstance(sweep, list):
            for i, s in enumerate(sweep):
                if hasattr(s, "fidelity"):
                    print(f"  step {i}: fidelity={s.fidelity:.4f} passed={s.passed}")
                else:
                    print(f"  step {i}: {s}")
        else:
            print(f"  result: {sweep}")
    except Exception as e:
        print(f"  Error: {e}")

# Adaptive optimization
print(f"\n\n{'='*70}")
print("  ADAPTIVE CIRCUIT OPTIMIZATION")
print(f"{'='*70}")
try:
    opt = god_code_simulator.adaptive_optimize(target_fidelity=0.99, nq=4, depth=4)
    if isinstance(opt, dict):
        for k, v in opt.items():
            print(f"  {k}: {v}")
    else:
        print(f"  Result: {opt}")
except Exception as e:
    print(f"  Error: {e}")

print("\n--- Noise Resilience Optimization ---")
try:
    noise_opt = god_code_simulator.optimize_noise_resilience(nq=2, noise_level=0.1)
    if isinstance(noise_opt, dict):
        for k, v in noise_opt.items():
            print(f"  {k}: {v}")
    else:
        print(f"  Result: {noise_opt}")
except Exception as e:
    print(f"  Error: {e}")

print(f"\n{'='*70}")
print("  DONE")
print(f"{'='*70}")
