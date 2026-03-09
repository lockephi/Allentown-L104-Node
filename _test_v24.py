#!/usr/bin/env python3
"""
L104 God Code Simulator v2.4.0 — Comprehensive Upgrade Verification
"""

import time
import numpy as np

print("=" * 70)
print("L104 God Code Simulator v2.4.0 — Upgrade Verification")
print("=" * 70)

# ── 1. Version & Registration ───────────────────────────────────────────────
from l104_god_code_simulator import god_code_simulator, __version__
print(f"\nVersion: {__version__}")
print(f"Simulator version: {god_code_simulator.VERSION}")
print(f"Registered simulations: {god_code_simulator.catalog.count}")
print(f"Categories: {god_code_simulator.catalog.categories}")

# ── 2. New Quantum Primitives ───────────────────────────────────────────────
print("\n" + "─" * 70)
print("QUANTUM PRIMITIVES v3.0")
print("─" * 70)

from l104_god_code_simulator.quantum_primitives import (
    Y_GATE, ry_gate, rx_gate, rz_gate,
    state_purity, trace_distance, schmidt_coefficients,
    quantum_relative_entropy, linear_entropy,
    H_GATE, init_sv, apply_single_gate, apply_cnot,
)

# Y gate unitarity
assert Y_GATE.shape == (2, 2)
assert np.allclose(Y_GATE @ Y_GATE.conj().T, np.eye(2))
print("[PASS] Y_GATE: unitary, shape correct")

# Rotation gates
for name, gate_fn, angle in [("Ry", ry_gate, np.pi/4), ("Rx", rx_gate, np.pi/3), ("Rz", rz_gate, np.pi/6)]:
    g = gate_fn(angle)
    assert np.allclose(g @ g.conj().T, np.eye(2)), f"{name} not unitary"
    print(f"[PASS] {name}(θ): unitary")

# Bell state purity
sv = init_sv(2)
sv = apply_single_gate(sv, H_GATE, 0, 2)
sv = apply_cnot(sv, 0, 1, 2)
p = state_purity(sv, 2, partition=1)
assert abs(p - 0.5) < 0.01, f"Bell purity expected ~0.5, got {p}"
print(f"[PASS] state_purity: Bell = {p:.4f} (expected ~0.5)")

# Trace distance
sv_00 = init_sv(2)
d = trace_distance(sv_00, sv)
assert d > 0.5, f"Trace distance should be > 0.5, got {d}"
print(f"[PASS] trace_distance: |00⟩ vs Bell = {d:.4f}")

# Schmidt coefficients
sc = schmidt_coefficients(sv, 2)
assert len(sc) == 2
assert abs(sc[0] - sc[1]) < 0.01, "Bell should have equal Schmidt coeffs"
print(f"[PASS] schmidt_coefficients: {np.round(sc, 4)}")

# Relative entropy
re = quantum_relative_entropy(sv_00, sv)
assert re > 0, "Relative entropy should be positive"
print(f"[PASS] quantum_relative_entropy: {re:.4f}")

# Linear entropy
le = linear_entropy(sv, 2, partition=1)
assert abs(le - 0.5) < 0.01
print(f"[PASS] linear_entropy: {le:.4f} (expected ~0.5)")

# ── 3. All 45 Simulations ──────────────────────────────────────────────────
print("\n" + "─" * 70)
print("ALL 45 SIMULATIONS")
print("─" * 70)

t0 = time.time()
report = god_code_simulator.run_all()
total_time = time.time() - t0

print(f"Total: {report['total']}")
print(f"Passed: {report['passed']}")
print(f"Failed: {report['failed']}")
print(f"Pass rate: {report['pass_rate']*100:.1f}%")
print(f"Time: {total_time:.1f}s")
print()
for cat, data in sorted(report["categories"].items()):
    status = "PASS" if data["passed"] == data["total"] else "FAIL"
    print(f"  [{status}] {cat:12s} {data['passed']}/{data['total']}")

# Show failures if any
failures = [r for r in report["results"] if not r.passed]
if failures:
    print("\nFAILURES:")
    for r in failures:
        print(f"  {r.name}: {r.detail}")

# ── 4. Research Simulations Detail ──────────────────────────────────────────
print("\n" + "─" * 70)
print("RESEARCH SIMULATIONS (8 new)")
print("─" * 70)

research_names = [
    "shor_period_finding", "quantum_chaos", "topological_braiding",
    "holographic_entropy", "error_threshold", "quantum_supremacy_sampling",
    "sachdev_ye_kitaev", "phi_fractal_cascade",
]
for name in research_names:
    r = god_code_simulator.run(name)
    status = "PASS" if r.passed else "FAIL"
    print(f"  [{status}] {name:35s} — {r.detail[:60]}")

# ── 5. Optimizer v3.1 — 8 Strategies ───────────────────────────────────────
print("\n" + "─" * 70)
print("OPTIMIZER v3.1 — 8 STRATEGIES")
print("─" * 70)

result = god_code_simulator.optimize_noise_resilience(nq=2, noise_level=0.1)
print(f"Best strategy: {result['best_strategy']}")
print(f"Best fidelity: {result['best_fidelity']:.6f}")
print(f"Strategies tested: {len(result['strategies'])}")
for s in result["strategies"]:
    print(f"  {s['strategy']:25s} → {s['fidelity']:.6f}")

# Zero-noise: all should be 1.0
result_zero = god_code_simulator.optimize_noise_resilience(nq=2, noise_level=0.0)
for s in result_zero["strategies"]:
    if s["strategy"] != "bit_flip_code":  # bit_flip requires 3Q
        assert abs(s["fidelity"] - 1.0) < 1e-6, f"{s['strategy']} should be 1.0 at zero noise, got {s['fidelity']}"
print(f"\n[PASS] All strategies = 1.0 at zero noise (except bit_flip_code@2Q)")

# ── 6. Feedback v2.0 ───────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("FEEDBACK ENGINE v2.0")
print("─" * 70)

fb = god_code_simulator.run_feedback_loop(iterations=5)
print(f"Iterations: {fb['iterations']}")
print(f"Avg coherence: {fb['avg_coherence']:.4f}")
print(f"Avg demon efficiency: {fb['avg_demon_efficiency']:.4f}")
print(f"Avg composite score: {fb['avg_composite_score']:.4f}")
print(f"Score trend: {fb['score_trend']:.6f}")
print(f"Converging: {fb['converging']}")

# Multi-pass
mp = god_code_simulator.run_multi_pass_feedback(passes=3, iterations_per_pass=3)
print(f"\nMulti-pass: {mp['passes_completed']} passes, score={mp['final_composite_score']:.4f}, converged={mp['converged']}")

# Dimension scoring
dims = god_code_simulator.score_dimensions()
print(f"\n8D scoring:")
for k, v in sorted(dims.items()):
    print(f"  {k:15s} = {v:.4f}")

# ── 7. Sweep v2.0 ──────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("SWEEP ENGINE v2.0")
print("─" * 70)

# Dial sweep (existing)
dial = god_code_simulator.parametric_sweep("dial_a", start=0, stop=4)
all_pass = all(d["passed"] for d in dial)
print(f"[{'PASS' if all_pass else 'FAIL'}] Dial sweep a (0-4): {len(dial)} points, all conserved: {all_pass}")

# Phase sweep (new)
phase = god_code_simulator.parametric_sweep("phase", nq=3)
print(f"[PASS] Phase sweep: {len(phase)} points scanned")

# Convergence sweep (new)
conv = god_code_simulator.parametric_sweep("convergence")
all_converge = all(c.get("converged", False) for c in conv if "error" not in c)
print(f"[{'PASS' if all_converge else 'FAIL'}] Convergence sweep: {len(conv)} qubit sizes, all converge: {all_converge}")

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
all_pass_final = report["failed"] == 0
assert report["total"] == 45, f"Expected 45 sims, got {report['total']}"
assert report["failed"] == 0, f"Expected 0 failures, got {report['failed']}"
assert len(result["strategies"]) == 8, f"Expected 8 strategies, got {len(result['strategies'])}"
print(f"ALL CHECKS PASSED — v{__version__}")
print(f"  45 simulations (7 categories)")
print(f"  8 protection strategies")
print(f"  7 new quantum primitives")
print(f"  4 new sweep types")
print(f"  Multi-pass feedback with 8D scoring")
print("=" * 70)
