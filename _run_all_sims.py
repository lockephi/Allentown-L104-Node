#!/usr/bin/env python3
"""Run all L104 God Code simulations + parametric sweeps + adaptive optimization."""

from l104_god_code_simulator import god_code_simulator

report = god_code_simulator.run_all()

cats = {}
for name, r in report.items():
    cat = r.get("category", "unknown") if isinstance(r, dict) else "unknown"
    cats.setdefault(cat, []).append(name)

print("=" * 60)
print("  GOD CODE SIMULATOR — ALL SIMULATIONS")
print("=" * 60)

for cat, names in sorted(cats.items()):
    print(f"\n  [{cat.upper()}] ({len(names)} simulations)")
    for n in names:
        r = report[n]
        fid = r.get("fidelity", r.get("god_code_fidelity", "")) if isinstance(r, dict) else ""
        god = r.get("god_code_alignment", r.get("alignment", "")) if isinstance(r, dict) else ""
        parts = []
        if fid:
            parts.append(f"fidelity={fid}")
        if god:
            parts.append(f"alignment={god}")
        extra = " | ".join(parts)
        print(f"    OK  {n}  {extra}")

print(f"\n  Total: {len(report)} simulations completed")

# Parametric sweeps
print("\n" + "=" * 60)
print("  PARAMETRIC SWEEPS")
print("=" * 60)

sweep = god_code_simulator.parametric_sweep("dial_a", start=0, stop=4)
print(f"  dial_a sweep: {len(sweep) if isinstance(sweep, (list, dict)) else 'done'} points")

noise = god_code_simulator.parametric_sweep("noise")
print(f"  noise sweep:  {len(noise) if isinstance(noise, (list, dict)) else 'done'} points")

depth = god_code_simulator.parametric_sweep("depth")
print(f"  depth sweep:  {len(depth) if isinstance(depth, (list, dict)) else 'done'} points")

# Adaptive optimization
print("\n" + "=" * 60)
print("  ADAPTIVE OPTIMIZATION")
print("=" * 60)

opt = god_code_simulator.adaptive_optimize(target_fidelity=0.99, nq=4, depth=4)
if isinstance(opt, dict):
    fid_res = opt.get("best_fidelity", opt.get("fidelity", "N/A"))
    iters = opt.get("iterations", opt.get("steps", "N/A"))
    print(f"  target=0.99  |  achieved={fid_res}  |  iterations={iters}")
else:
    print(f"  result: {opt}")

# Noise resilience
opt2 = god_code_simulator.optimize_noise_resilience(nq=2, noise_level=0.1)
if isinstance(opt2, dict):
    nr_fid = opt2.get("best_fidelity", opt2.get("fidelity", "N/A"))
    print(f"  noise_resilience (nq=2, noise=0.1): fidelity={nr_fid}")
else:
    print(f"  noise_resilience: {opt2}")

print("\n" + "=" * 60)
print("  ALL SIMULATIONS COMPLETE")
print("=" * 60)
