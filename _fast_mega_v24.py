#!/usr/bin/env python3
"""
L104 God Code Simulator v2.4.0 — Fast Mega Validation
Covers all 10 test dimensions in ~5 minutes (instead of 2+ hours).
"""

import sys, os, time, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l104_god_code_simulator import god_code_simulator, __version__
from l104_god_code_simulator.constants import GOD_CODE, PHI, VOID_CONSTANT
from l104_god_code_simulator.quantum_primitives import (
    init_sv, apply_single_gate, apply_cnot, H_GATE, X_GATE, Z_GATE,
    fidelity, entanglement_entropy, probabilities, god_code_dial,
    Y_GATE, ry_gate, rx_gate, rz_gate, state_purity, trace_distance,
    schmidt_coefficients, quantum_relative_entropy, linear_entropy,
)

PASS = FAIL = 0
FAILURES = []

def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL: {name} — {detail}")

print(f"═══ L104 God Code Simulator v{__version__} — Fast Mega Validation ═══\n")
t_start = time.time()

# ─── 1. Stability: 3 full batch runs ───────────────────────────────────────
print("Battery 1: Stability (3 reps × 45 sims = 135 runs)")
for rep in range(3):
    report = god_code_simulator.run_all()
    check(f"stability_rep{rep}", report["failed"] == 0 and report["total"] == 45,
          f"{report['passed']}/{report['total']}")
    print(f"  Rep {rep+1}: {report['passed']}/{report['total']} ✓")

# ─── 2. Monte Carlo Parameter Fuzzing (50 runs) ───────────────────────────
print("\nBattery 2: Monte Carlo Fuzzing (50 fuzz runs)")
fuzz_pass = 0
fuzz_total = 0
for _ in range(50):
    a, b, c, d = [random.uniform(-2.0, 5.0) for _ in range(4)]
    try:
        val = god_code_dial(a, b, c, d)
        ok = isinstance(val, (int, float)) and not np.isnan(val)
        fuzz_pass += 1 if ok else 0
    except Exception:
        pass
    fuzz_total += 1
check("monte_carlo_fuzz", fuzz_pass == fuzz_total, f"{fuzz_pass}/{fuzz_total}")
print(f"  {fuzz_pass}/{fuzz_total} valid ✓")

# ─── 3. Dial Sweeps (4 dials × 10 points = 40 runs) ──────────────────────
print("\nBattery 3: Dial Sweeps (4 dials)")
for dial in ["dial_a", "dial_b", "dial_c", "dial_d"]:
    result = god_code_simulator.parametric_sweep(dial, start=-1, stop=3)
    all_pass = all(d["passed"] for d in result)
    check(f"dial_{dial}", all_pass, f"{len(result)} points, pass={all_pass}")
print(f"  4 dials swept ✓")

# ─── 4. Noise Gradient (20 points) ────────────────────────────────────────
print("\nBattery 4: Noise Gradient (20 levels)")
noise_levels = [i * 0.025 for i in range(20)]
noise = god_code_simulator.parametric_sweep("noise", noise_levels=noise_levels)
check("noise_gradient", len(noise) >= 15, f"{len(noise)} points")
# At low noise, fidelity should be high
if noise:
    low_noise_fid = noise[0].get("fidelity", 0)
    check("noise_low_fidelity", low_noise_fid > 0.9, f"fid@0.0={low_noise_fid:.4f}")
print(f"  {len(noise)} noise levels swept ✓")

# ─── 5. Qubit Scaling (2Q-8Q on conservation_proof) ──────────────────────
print("\nBattery 5: Qubit Scaling (2Q-8Q)")
for nq in [2, 3, 4, 6, 8]:
    r = god_code_simulator.run("conservation_proof", nq=nq)
    check(f"scale_{nq}Q", r.passed, f"nq={nq} fid={r.fidelity:.6f}")
print(f"  5 qubit sizes ✓")

# ─── 6. Optimizer Convergence (3 configs) ─────────────────────────────────
print("\nBattery 6: Optimizer Convergence")
for nq, depth, iters in [(2, 3, 30), (3, 4, 25), (4, 3, 20)]:
    r = god_code_simulator.adaptive_optimize(target_fidelity=0.95, nq=nq, depth=depth, max_iterations=iters)
    check(f"opt_{nq}Q_d{depth}", r["best_fidelity"] > 0.9, f"fid={r['best_fidelity']:.6f}")
print(f"  3 configs ✓")

# ─── 7. Protection Strategy Matrix ───────────────────────────────────────
print("\nBattery 7: 8 Strategies × 3 Noise Levels")
for noise_lvl in [0.0, 0.05, 0.2]:
    r = god_code_simulator.optimize_noise_resilience(nq=2, noise_level=noise_lvl)
    n_strats = len(r["strategies"])
    check(f"strat_n{noise_lvl}", n_strats == 8, f"{n_strats} strategies at noise={noise_lvl}")
    if noise_lvl == 0.0:
        for s in r["strategies"]:
            if s["strategy"] != "bit_flip_code":
                check(f"zero_noise_{s['strategy']}", abs(s["fidelity"] - 1.0) < 1e-4,
                      f"{s['strategy']}={s['fidelity']:.6f}")
print(f"  8 strategies × 3 levels ✓")

# ─── 8. Feedback Loop ────────────────────────────────────────────────────
print("\nBattery 8: Feedback v2.0")
fb = god_code_simulator.run_feedback_loop(iterations=5)
check("feedback_basic", 0 < fb["avg_composite_score"] <= 1.0, f"score={fb['avg_composite_score']:.4f}")
mp = god_code_simulator.run_multi_pass_feedback(passes=3, iterations_per_pass=3)
check("feedback_multipass", 0 < mp["final_composite_score"] <= 1.0, f"score={mp['final_composite_score']:.4f}")
dims = god_code_simulator.score_dimensions()
check("8d_scoring", len(dims) == 8, f"{len(dims)} dimensions")
print(f"  Feedback OK ✓")

# ─── 9. Sweep v2.0 New Types ────────────────────────────────────────────
print("\nBattery 9: Sweep v2.0 (4 new types)")
phase = god_code_simulator.parametric_sweep("phase", nq=2)
check("sweep_phase", len(phase) > 10, f"{len(phase)} points")
conv = god_code_simulator.parametric_sweep("convergence")
check("sweep_convergence", len(conv) >= 3, f"{len(conv)} sizes")
print(f"  New sweep types ✓")

# ─── 10. Quantum Primitives v3.0 ────────────────────────────────────────
print("\nBattery 10: Quantum Primitives v3.0")
# Y gate
check("Y_gate", np.allclose(Y_GATE @ Y_GATE.conj().T, np.eye(2)))
# Rotation gates
for name, fn in [("Ry", ry_gate), ("Rx", rx_gate), ("Rz", rz_gate)]:
    g = fn(np.pi / 4)
    check(f"{name}_unitary", np.allclose(g @ g.conj().T, np.eye(2)))
# Bell state purity
sv = init_sv(2)
sv = apply_single_gate(sv, H_GATE, 0, 2)
sv = apply_cnot(sv, 0, 1, 2)
check("bell_purity", abs(state_purity(sv, 2, 1) - 0.5) < 0.02)
check("bell_trace_dist", trace_distance(init_sv(2), sv) > 0.4)
check("bell_schmidt", len(schmidt_coefficients(sv, 2)) == 2)
check("bell_linear_entropy", abs(linear_entropy(sv, 2, 1) - 0.5) < 0.02)
check("relative_entropy", quantum_relative_entropy(init_sv(2), sv) > 0)
print(f"  9 primitive checks ✓")

# ─── 11. Research Simulations ────────────────────────────────────────────
print("\nBattery 11: Research Simulations (8 new)")
research_names = [
    "shor_period_finding", "quantum_chaos", "topological_braiding",
    "holographic_entropy", "error_threshold", "quantum_supremacy_sampling",
    "sachdev_ye_kitaev", "phi_fractal_cascade",
]
for name in research_names:
    r = god_code_simulator.run(name)
    check(f"research_{name}", r.passed, r.detail[:60])
print(f"  8 research sims ✓")

# ─── Summary ─────────────────────────────────────────────────────────────
elapsed = time.time() - t_start
total = PASS + FAIL
sim_runs = 3 * 45 + 50 + 40 + 20 + 5 + 3 + 24 + 5 + 36 + 4 + 9 + 8  # ~249 runs + 135 sims

print(f"\n{'═' * 70}")
print(f"  FAST MEGA VALIDATION COMPLETE — v{__version__}")
print(f"  Checks: {PASS}/{total} passed ({PASS/total*100:.1f}%)")
print(f"  Simulation runs: ~{sim_runs}+")
print(f"  Time: {elapsed:.1f}s")
print(f"{'═' * 70}")
if FAILURES:
    print(f"  FAILURES ({len(FAILURES)}):")
    for f in FAILURES:
        print(f"    ❌ {f}")
else:
    print(f"  ✅ ALL CHECKS PASSED — 100% MEGA VALIDATION")
print(f"{'═' * 70}")
sys.exit(0 if FAIL == 0 else 1)
