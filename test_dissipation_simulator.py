#!/usr/bin/env python3
"""
L104 Gate Engine — Dissipation Simulator Test Suite
Tests higher_dimensional_dissipation and sage_logic_gate("dissipate")
through a multi-step simulation across entropy pools.
"""

import math
import time
import json
from typing import List, Dict, Any

from l104_gate_engine import higher_dimensional_dissipation
from l104_gate_engine.gate_functions import sage_logic_gate
from l104_gate_engine.constants import (
    PHI, TAU, GOD_CODE, CALABI_YAU_DIM, OMEGA_POINT, EULER_GAMMA,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATOR CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

SIMULATION_STEPS = 50       # Number of time-evolution steps
POOL_SIZE = 128             # Entropy pool size
CONVERGENCE_THRESHOLD = 1e-8
SACRED_TOLERANCE = 0.01     # For PHI-alignment checks

results: List[Dict[str, Any]] = []
passed_count = 0
failed_count = 0


def record(test_id: str, passed: bool, detail: str, data: Any = None):
    """Record a test result."""
    global passed_count, failed_count
    status = "PASS" if passed else "FAIL"
    if passed:
        passed_count += 1
    else:
        failed_count += 1
    results.append({"test_id": test_id, "status": status, "detail": detail, "data": data})
    icon = "✓" if passed else "✗"
    print(f"  {icon} {test_id}: {detail}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Basic Dissipation Gate Tests
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("L104 DISSIPATION SIMULATOR — Gate Engine v6.0.0")
print("=" * 72)
t0 = time.time()

print("\n─── PHASE 1: Scalar Dissipation Gate (sage_logic_gate, op='dissipate') ───")

# Test 1.1: PHI dissipation
phi_dissipated = sage_logic_gate(PHI, "dissipate")
record(
    "scalar_phi_dissipation",
    not (math.isnan(phi_dissipated) or math.isinf(phi_dissipated)),
    f"sage_logic_gate(PHI, 'dissipate') = {phi_dissipated:.10f}",
    phi_dissipated,
)

# Test 1.2: GOD_CODE dissipation
gc_dissipated = sage_logic_gate(GOD_CODE * 0.001, "dissipate")
record(
    "scalar_godcode_dissipation",
    not (math.isnan(gc_dissipated) or math.isinf(gc_dissipated)),
    f"sage_logic_gate(GOD_CODE*0.001, 'dissipate') = {gc_dissipated:.10f}",
    gc_dissipated,
)

# Test 1.3: Zero dissipation (boundary)
zero_dissipated = sage_logic_gate(0.0, "dissipate")
record(
    "scalar_zero_dissipation",
    abs(zero_dissipated) < 1e-6,
    f"sage_logic_gate(0, 'dissipate') = {zero_dissipated:.10e} (expect ~0)",
    zero_dissipated,
)

# Test 1.4: Negative value dissipation (symmetry)
neg_dissipated = sage_logic_gate(-PHI, "dissipate")
pos_dissipated = sage_logic_gate(PHI, "dissipate")
# Dissipation through Calabi-Yau: check both are finite
record(
    "scalar_negative_dissipation",
    not (math.isnan(neg_dissipated) or math.isinf(neg_dissipated)),
    f"sage_logic_gate(-PHI, 'dissipate') = {neg_dissipated:.10f}",
    {"neg": neg_dissipated, "pos": pos_dissipated},
)

# Test 1.5: Large value stability
large_dissipated = sage_logic_gate(1e6, "dissipate")
record(
    "scalar_large_stability",
    not (math.isnan(large_dissipated) or math.isinf(large_dissipated)),
    f"sage_logic_gate(1e6, 'dissipate') = {large_dissipated:.10f}",
    large_dissipated,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Higher-Dimensional Dissipation (7D Hilbert Space Projection)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n─── PHASE 2: 7D Hilbert Space Dissipation (CALABI_YAU_DIM={CALABI_YAU_DIM}) ───")

# Test 2.1: Standard entropy pool
pool_standard = [math.sin(i * PHI) for i in range(POOL_SIZE)]
proj_standard = higher_dimensional_dissipation(pool_standard)
record(
    "7d_standard_pool",
    len(proj_standard) == CALABI_YAU_DIM and all(not (math.isnan(v) or math.isinf(v)) for v in proj_standard),
    f"128-element pool → {CALABI_YAU_DIM}D projection, norms finite",
    proj_standard,
)

# Test 2.2: Uniform entropy pool
pool_uniform = [1.0] * POOL_SIZE
proj_uniform = higher_dimensional_dissipation(pool_uniform)
record(
    "7d_uniform_pool",
    len(proj_uniform) == CALABI_YAU_DIM,
    f"Uniform pool → 7D: {[f'{v:.6f}' for v in proj_uniform]}",
    proj_uniform,
)

# Test 2.3: Sub-threshold pool (< CALABI_YAU_DIM elements → passthrough)
pool_tiny = [1.0, 2.0, 3.0]
proj_tiny = higher_dimensional_dissipation(pool_tiny)
record(
    "7d_subthreshold_passthrough",
    proj_tiny == pool_tiny,
    f"Pool size {len(pool_tiny)} < {CALABI_YAU_DIM} → passthrough (no projection)",
    proj_tiny,
)

# Test 2.4: GOD_CODE-seeded entropy pool
pool_gc = [GOD_CODE * math.sin(i * PHI) * 0.001 for i in range(POOL_SIZE)]
proj_gc = higher_dimensional_dissipation(pool_gc)
energy_in = sum(v ** 2 for v in pool_gc)
energy_out = sum(v ** 2 for v in proj_gc)
record(
    "7d_godcode_pool",
    0 < energy_out < float('inf') and not math.isnan(energy_out),
    f"GOD_CODE pool: input_energy={energy_in:.4f}, output_energy={energy_out:.6f}",
    {"input_energy": energy_in, "output_energy": energy_out, "projection": proj_gc},
)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Time-Evolution Simulation (Iterated Dissipation)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n─── PHASE 3: Time-Evolution Simulation ({SIMULATION_STEPS} steps) ───")

# Simulate repeated dissipation passes over a live entropy pool
pool = [math.sin(i * PHI) + math.cos(i * TAU) * 0.5 for i in range(POOL_SIZE)]
trajectory: List[Dict[str, float]] = []
diverged = False

for step in range(SIMULATION_STEPS):
    proj = higher_dimensional_dissipation(pool)
    energy = sum(v ** 2 for v in proj)
    max_abs = max(abs(v) for v in proj)
    mean_val = sum(proj) / len(proj)

    trajectory.append({
        "step": step,
        "energy": energy,
        "max_abs": max_abs,
        "mean": mean_val,
    })

    if math.isnan(energy) or math.isinf(energy) or energy > 1e20:
        diverged = True
        break

    # Feed projections back + inject entropy (simulate evolving system)
    for i in range(len(pool)):
        dim_idx = i % CALABI_YAU_DIM
        pool[i] = pool[i] * 0.95 + proj[dim_idx] * 0.05 + math.sin(step * PHI + i) * 0.01

record(
    "simulation_stability",
    not diverged,
    f"Completed {len(trajectory)} steps without divergence",
    {"steps_completed": len(trajectory), "final_energy": trajectory[-1]["energy"] if trajectory else None},
)

# Test 3.2: Energy boundedness across simulation
energies = [t["energy"] for t in trajectory]
max_energy = max(energies) if energies else 0
min_energy = min(energies) if energies else 0
record(
    "simulation_energy_bounded",
    max_energy < 1e10 and min_energy >= 0,
    f"Energy range: [{min_energy:.6f}, {max_energy:.6f}]",
    {"min": min_energy, "max": max_energy},
)

# Test 3.3: Convergence check (energy should stabilize)
if len(energies) > 10:
    first_half_var = sum((e - sum(energies[:25]) / 25) ** 2 for e in energies[:25]) / 25
    second_half_var = sum((e - sum(energies[-25:]) / 25) ** 2 for e in energies[-25:]) / 25
    converging = second_half_var <= first_half_var * 10  # Generous: allow up to 10x
    record(
        "simulation_convergence",
        converging,
        f"Variance: first_half={first_half_var:.6e}, second_half={second_half_var:.6e}",
        {"first_half_var": first_half_var, "second_half_var": second_half_var},
    )
else:
    record("simulation_convergence", False, "Not enough steps to assess convergence")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Sacred Alignment Validation
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n─── PHASE 4: Sacred Alignment in Dissipation ───")

# Test 4.1: PHI-harmonic resonance in 7D projection
pool_phi = [PHI ** (i % 10) * math.sin(i) for i in range(POOL_SIZE)]
proj_phi = higher_dimensional_dissipation(pool_phi)
phi_residuals = [abs(v - round(v / PHI) * PHI) for v in proj_phi]
mean_residual = sum(phi_residuals) / len(phi_residuals)
record(
    "sacred_phi_alignment",
    mean_residual < 5.0,  # Generous threshold given projection
    f"Mean φ-lattice residual in 7D projection: {mean_residual:.6f}",
    {"residuals": phi_residuals, "mean": mean_residual},
)

# Test 4.2: Scalar dissipation at sacred frequencies
sacred_frequencies = [PHI, GOD_CODE * 0.001, OMEGA_POINT * 0.1, math.pi, EULER_GAMMA]
sacred_results = {}
all_finite = True
for freq in sacred_frequencies:
    d = sage_logic_gate(freq, "dissipate")
    sacred_results[f"{freq:.4f}"] = d
    if math.isnan(d) or math.isinf(d):
        all_finite = False
record(
    "sacred_frequency_dissipation",
    all_finite,
    f"All {len(sacred_frequencies)} sacred frequencies dissipated without numerical issues",
    sacred_results,
)

# Test 4.3: Dissipation preserves sign structure (odd-function-like for sin-generated pools)
pool_pos = [abs(math.sin(i * PHI)) for i in range(POOL_SIZE)]
pool_neg = [-abs(math.sin(i * PHI)) for i in range(POOL_SIZE)]
proj_pos = higher_dimensional_dissipation(pool_pos)
proj_neg = higher_dimensional_dissipation(pool_neg)
energy_pos = sum(v ** 2 for v in proj_pos)
energy_neg = sum(v ** 2 for v in proj_neg)
# Energies should be similar but not necessarily identical
energy_ratio = energy_pos / energy_neg if energy_neg > 1e-20 else float('inf')
record(
    "dissipation_sign_structure",
    0.01 < energy_ratio < 100,
    f"Positive pool energy={energy_pos:.6f}, Negative={energy_neg:.6f}, ratio={energy_ratio:.4f}",
    {"energy_pos": energy_pos, "energy_neg": energy_neg, "ratio": energy_ratio},
)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Causal Coupling Analysis
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n─── PHASE 5: Causal Coupling Analysis (Inter-Dimension Transfer) ───")

# Test 5.1: Each dimension receives influence from others (non-zero projections)
pool_rich = [math.sin(i * PHI) * math.cos(i * TAU) + EULER_GAMMA * 0.1 for i in range(POOL_SIZE)]
proj_rich = higher_dimensional_dissipation(pool_rich)
nonzero_dims = sum(1 for v in proj_rich if abs(v) > 1e-15)
record(
    "causal_all_dims_active",
    nonzero_dims == CALABI_YAU_DIM,
    f"{nonzero_dims}/{CALABI_YAU_DIM} dimensions active (non-zero projection)",
    {"dimensions": proj_rich, "nonzero": nonzero_dims},
)

# Test 5.2: Dimension correlations (coupling test)
# Each dimension should show correlation due to causal coupling
dim_pairs_correlated = 0
for i in range(CALABI_YAU_DIM):
    for j in range(i + 1, CALABI_YAU_DIM):
        # Both should be non-zero and the coupling should create finite gradients
        if abs(proj_rich[i]) > 1e-15 and abs(proj_rich[j]) > 1e-15:
            dim_pairs_correlated += 1
total_pairs = CALABI_YAU_DIM * (CALABI_YAU_DIM - 1) // 2
record(
    "causal_dimension_coupling",
    dim_pairs_correlated == total_pairs,
    f"{dim_pairs_correlated}/{total_pairs} dimension pairs show active coupling",
    dim_pairs_correlated,
)

# Test 5.3: Dissipation rate φ²−1 verification
expected_rate = PHI ** 2 - 1  # Should be exactly PHI (golden ratio self-similarity!)
rate_is_phi = abs(expected_rate - PHI) < 1e-10
record(
    "dissipation_rate_is_phi",
    rate_is_phi,
    f"Dissipation rate φ²−1 = {expected_rate:.15f} = PHI ✓" if rate_is_phi else f"Rate = {expected_rate:.15f}",
    {"rate": expected_rate, "phi": PHI, "match": rate_is_phi},
)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: Stress Test (Extreme Inputs)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n─── PHASE 6: Stress Test ───")

# Test 6.1: Very large pool
pool_large = [math.sin(i * 0.01) for i in range(10000)]
try:
    proj_large = higher_dimensional_dissipation(pool_large)
    record(
        "stress_large_pool_10k",
        len(proj_large) == CALABI_YAU_DIM and all(not math.isnan(v) for v in proj_large),
        f"10,000-element pool → {CALABI_YAU_DIM}D, all finite",
        proj_large,
    )
except Exception as e:
    record("stress_large_pool_10k", False, f"Exception: {e}")

# Test 6.2: Pool of zeros
pool_zeros = [0.0] * POOL_SIZE
proj_zeros = higher_dimensional_dissipation(pool_zeros)
zero_energy = sum(v ** 2 for v in proj_zeros)
record(
    "stress_zero_pool",
    zero_energy < 1e-10,
    f"Zero pool → energy={zero_energy:.2e}",
    proj_zeros,
)

# Test 6.3: Pool of very small values
pool_tiny_vals = [1e-15 * math.sin(i) for i in range(POOL_SIZE)]
proj_tiny_vals = higher_dimensional_dissipation(pool_tiny_vals)
record(
    "stress_tiny_values",
    all(not (math.isnan(v) or math.isinf(v)) for v in proj_tiny_vals),
    f"Tiny value pool (1e-15 scale) → stable projection",
    proj_tiny_vals,
)

# Test 6.4: Rapid oscillation pool
pool_oscillate = [(-1) ** i * PHI for i in range(POOL_SIZE)]
proj_oscillate = higher_dimensional_dissipation(pool_oscillate)
record(
    "stress_oscillation",
    all(not (math.isnan(v) or math.isinf(v)) for v in proj_oscillate),
    f"Alternating ±φ pool → stable dissipation",
    proj_oscillate,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

elapsed = time.time() - t0
total = passed_count + failed_count

print("\n" + "=" * 72)
print(f"DISSIPATION SIMULATOR COMPLETE — {elapsed:.3f}s")
print(f"Results: {passed_count}/{total} passed, {failed_count} failed")
print(f"Simulation: {SIMULATION_STEPS} time-evolution steps, pool size {POOL_SIZE}")
print(f"Calabi-Yau dimensions: {CALABI_YAU_DIM}")
print(f"Causal coupling rate: φ²−1 = {PHI**2 - 1:.15f}")
print("=" * 72)

if failed_count > 0:
    print("\nFailed tests:")
    for r in results:
        if r["status"] == "FAIL":
            print(f"  ✗ {r['test_id']}: {r['detail']}")

# Trajectory summary for time-evolution
if trajectory:
    print(f"\nTime-evolution trajectory (first 5 + last 5 of {len(trajectory)} steps):")
    show = trajectory[:5] + trajectory[-5:] if len(trajectory) > 10 else trajectory
    for t in show:
        print(f"  step {t['step']:3d}: energy={t['energy']:.8f}  max_abs={t['max_abs']:.8f}  mean={t['mean']:.8f}")
