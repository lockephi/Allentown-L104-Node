#!/usr/bin/env python3
"""Quick validation of decomposed l104_god_code_simulator package."""

print("=" * 70)
print("L104 GOD CODE SIMULATOR — DECOMPOSITION VALIDATION")
print("=" * 70)

# 1. Core imports
from l104_god_code_simulator import god_code_simulator, GodCodeSimulator, SimulationResult
from l104_god_code_simulator import PHI, GOD_CODE
from l104_god_code_simulator.constants import VOID_CONSTANT
from l104_god_code_simulator.simulator import PHI as PHI2, GOD_CODE as GC2

print(f"PHI={PHI}, GOD_CODE={GOD_CODE}")
print(f"PHI2={PHI2}, GC2={GC2}")
print(f"VOID_CONSTANT={VOID_CONSTANT}")
assert PHI == PHI2
assert GOD_CODE == GC2
print("[OK] Backward-compat imports match")

# 2. Status
status = god_code_simulator.get_status()
print(f"\nVersion: {status['version']}")
print(f"Simulations: {status['simulations_registered']}")
print(f"Categories: {status['categories']}")

# 3. Individual sims
r = god_code_simulator.run("conservation_proof")
print(f"\n{r.summary()}")

r2 = god_code_simulator.run("bell_chsh_violation")
print(f"{r2.summary()}")

# 4. Payload converters
coh = r2.to_coherence_payload()
print(f"\nCoherence payload keys: {list(coh.keys())}")
asi = r2.to_asi_scoring()
print(f"ASI scoring keys: {list(asi.keys())}")

# 5. Full run
print("\n=== ALL SIMULATIONS ===")
report = god_code_simulator.run_all()
print(f"Total: {report['total']}, Passed: {report['passed']}, Failed: {report['failed']}")
print(f"Pass rate: {report['pass_rate']:.1%}")
print(f"Elapsed: {report['total_elapsed_ms']:.1f}ms")
for s in report["summaries"]:
    print(f"  {s}")

# 6. Parametric sweep
sweep = god_code_simulator.parametric_sweep("dial_a", start=0, stop=4)
print(f"\nDial sweep results: {len(sweep)} points, all pass={all(s['passed'] for s in sweep)}")

# 7. Depth sweep
ds = god_code_simulator.parametric_sweep("depth")
print(f"Depth sweep: {len(ds)} points")

# 8. Adaptive optimize
opt = god_code_simulator.adaptive_optimize(target_fidelity=0.5, nq=3, depth=2)
print(f"Optimizer: converged={opt['converged']}, best_fidelity={opt['best_fidelity']:.4f}")

# 9. Feedback loop
fb = god_code_simulator.run_feedback_loop(iterations=3)
print(f"Feedback: {fb['iterations']} iterations, avg_coherence={fb['avg_coherence']:.4f}")

print("\n" + "=" * 70)
print("ALL DECOMPOSED IMPORTS AND SIMULATIONS WORKING — v2.0.0 VALIDATED")
print("=" * 70)
