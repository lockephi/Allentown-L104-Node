#!/usr/bin/env python3
"""
L104 Simulation Speed Benchmark v5.1
═══════════════════════════════════════════════════════════════════════════════

Benchmarks the v5.1 performance optimizations across all simulation layers:

  1. reconstruct_density_matrix: O(4^n) outer product vs O(16^n) Pauli loop
  2. trotter_evolution: composed step operator (13× fewer matmuls/step)
  3. topological_entanglement_entropy: SVD-based (no density matrix)
  4. singlet_projection: reduced 2-qubit DM (3× fewer gate applications)
  5. build_unitary: direct permutation matrices (no column loop)
  6. GodCodeSimulator.run_all: parallel via ThreadPoolExecutor
  7. ParametricSweepEngine.phase_sweep: parallel 104-point grid
  8. Simulator.prob/conditional_prob: vectorized numpy masking
  9. _expm_approx: eigendecomposition fallback (no Taylor loop)
 10. apply_cnot/cp/swap: cached index arrays (zero alloc after warmup)

Usage:
    .venv/bin/python _bench_simulation_speed.py
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import sys
import os
import json
import math
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _timer(fn, label, warmup=1, runs=3):
    """Run fn with warmup and return (median_ms, result)."""
    for _ in range(warmup):
        result = fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000)
    median_ms = sorted(times)[len(times) // 2]
    return median_ms, result


def bench_reconstruct_density_matrix():
    """Benchmark: density matrix reconstruction (was O(16^n) Pauli loop)."""
    from l104_god_code_simulator.quantum_primitives import (
        init_sv, apply_single_gate, apply_cnot, H_GATE,
        reconstruct_density_matrix,
    )
    results = {}
    for nq in [2, 3, 4, 5]:
        sv = init_sv(nq)
        sv = apply_single_gate(sv, H_GATE, 0, nq)
        for q in range(nq - 1):
            sv = apply_cnot(sv, q, q + 1, nq)

        ms, res = _timer(lambda: reconstruct_density_matrix(sv, nq), f"recon_{nq}q")
        results[f"{nq}q"] = {
            "ms": round(ms, 3),
            "purity": round(res["purity"], 4),
            "is_pure": res["is_pure"],
        }
        print(f"  reconstruct_density_matrix({nq}q): {ms:.3f} ms  purity={res['purity']:.4f}")
    return results


def bench_trotter_evolution():
    """Benchmark: Trotter evolution with composed step operator."""
    from l104_god_code_simulator.quantum_primitives import iron_lattice_heisenberg
    results = {}
    for n_sites in [3, 4, 5]:
        for steps in [10, 20]:
            ms, res = _timer(
                lambda ns=n_sites, st=steps: iron_lattice_heisenberg(
                    n_sites=ns, trotter_steps=st
                ),
                f"trotter_{n_sites}s_{steps}st",
            )
            key = f"{n_sites}sites_{steps}steps"
            results[key] = {
                "ms": round(ms, 3),
                "energy": round(res["energy"], 6),
                "magnetization": round(res["magnetization"], 6),
            }
            print(f"  iron_lattice_heisenberg({n_sites}s, {steps}st): {ms:.3f} ms")
    return results


def bench_topological_entropy():
    """Benchmark: topological entanglement entropy (SVD-based)."""
    from l104_god_code_simulator.quantum_primitives import (
        init_sv, apply_single_gate, apply_cnot, H_GATE,
        topological_entanglement_entropy,
    )
    results = {}
    for nq in [3, 4, 6, 8]:
        sv = init_sv(nq)
        sv = apply_single_gate(sv, H_GATE, 0, nq)
        for q in range(nq - 1):
            sv = apply_cnot(sv, q, q + 1, nq)
        ms, res = _timer(
            lambda: topological_entanglement_entropy(sv, nq),
            f"topo_{nq}q",
        )
        results[f"{nq}q"] = {
            "ms": round(ms, 3),
            "gamma": round(res["topological_entropy"], 6),
        }
        print(f"  topological_entropy({nq}q): {ms:.3f} ms  γ={res['topological_entropy']:.4f}")
    return results


def bench_singlet_projection():
    """Benchmark: singlet projection (reduced 2-qubit DM)."""
    from l104_god_code_simulator.quantum_primitives import (
        init_sv, apply_single_gate, H_GATE,
        singlet_projection, cooper_pair_correlation,
    )
    results = {}
    for nq in [4, 6, 8]:
        sv = init_sv(nq)
        for q in range(nq):
            sv = apply_single_gate(sv, H_GATE, q, nq)
        ms, res = _timer(
            lambda: cooper_pair_correlation(sv, nq),
            f"cooper_{nq}q",
        )
        results[f"{nq}q"] = {
            "ms": round(ms, 3),
            "avg_singlet": round(res["avg_singlet_fraction"], 4),
        }
        print(f"  cooper_pair_correlation({nq}q): {ms:.3f} ms")
    return results


def bench_build_unitary():
    """Benchmark: build_unitary with direct matrix construction."""
    from l104_god_code_simulator.quantum_primitives import build_unitary
    results = {}
    for nq in [3, 4, 5]:
        ops = []
        ops.append(("H", 0))
        for q in range(nq - 1):
            ops.append(("CX", (q, q + 1)))
        ops.append(("Rz", (0.5, 0)))
        ops.append(("CP", (0.25, 0, 1)))
        if nq > 2:
            ops.append(("SWAP", (0, nq - 1)))

        ms, U = _timer(lambda: build_unitary(nq, ops), f"unitary_{nq}q")
        is_unitary = np.allclose(U @ U.conj().T, np.eye(2**nq), atol=1e-10)
        results[f"{nq}q"] = {
            "ms": round(ms, 3),
            "is_unitary": is_unitary,
            "dim": 2**nq,
        }
        print(f"  build_unitary({nq}q, {len(ops)} gates): {ms:.3f} ms  unitary={is_unitary}")
    return results


def bench_god_code_run_all():
    """Benchmark: GodCodeSimulator.run_all() with parallel execution."""
    from l104_god_code_simulator import god_code_simulator
    ms, report = _timer(lambda: god_code_simulator.run_all(), "run_all", warmup=0, runs=1)
    passed = report["passed"]
    total = report["total"]
    results = {
        "ms": round(ms, 1),
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / max(total, 1), 3),
    }
    print(f"  god_code_simulator.run_all(): {ms:.1f} ms  {passed}/{total} passed")
    return results


def bench_phase_sweep():
    """Benchmark: phase_sweep with parallel 104-point grid."""
    from l104_god_code_simulator.sweep import ParametricSweepEngine
    sweep = ParametricSweepEngine()
    ms, res = _timer(lambda: sweep.phase_sweep(), "phase_sweep", warmup=0, runs=1)
    results = {
        "ms": round(ms, 1),
        "points": len(res),
    }
    print(f"  phase_sweep(104 points): {ms:.1f} ms")
    return results


def bench_simulator_prob():
    """Benchmark: Simulator prob() and conditional_prob() vectorized."""
    from l104_simulator.simulator import QuantumCircuit, Simulator
    results = {}
    for nq in [6, 8, 10, 12]:
        qc = QuantumCircuit(nq, "bench")
        qc.h(0)
        for q in range(nq - 1):
            qc.cx(q, q + 1)
        sim = Simulator()
        res = sim.run(qc)

        ms_prob, _ = _timer(
            lambda: [res.prob(q, 0) for q in range(nq)],
            f"prob_{nq}q",
        )
        ms_cond, _ = _timer(
            lambda: res.conditional_prob(0, 0, 1, 0),
            f"cond_prob_{nq}q",
        )
        results[f"{nq}q"] = {
            "prob_ms": round(ms_prob, 3),
            "cond_prob_ms": round(ms_cond, 3),
        }
        print(f"  prob({nq}q): {ms_prob:.3f} ms  cond_prob: {ms_cond:.3f} ms")
    return results


def bench_expm_approx():
    """Benchmark: _expm_approx eigendecomposition fallback."""
    from l104_god_code_simulator.quantum_primitives import _expm_approx
    results = {}
    for dim in [4, 8, 16, 32]:
        H = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        H = (H + H.conj().T) / 2  # Hermitian
        M = -1j * 0.1 * H
        ms, res = _timer(lambda: _expm_approx(M), f"expm_{dim}")
        results[f"{dim}x{dim}"] = {"ms": round(ms, 3)}
        print(f"  _expm_approx({dim}×{dim}): {ms:.3f} ms")
    return results


def bench_apply_cnot_cached():
    """Benchmark: apply_cnot with cached index arrays."""
    from l104_god_code_simulator.quantum_primitives import init_sv, apply_cnot
    results = {}
    for nq in [4, 6, 8, 10]:
        sv = init_sv(nq)
        sv[0] = 0.5
        sv[1] = 0.5
        sv[2] = 0.5
        sv[3] = 0.5
        n = np.linalg.norm(sv)
        sv /= n
        ms, _ = _timer(
            lambda: apply_cnot(sv, 0, 1, nq),
            f"cnot_{nq}q",
            runs=10,
        )
        results[f"{nq}q"] = {"ms": round(ms, 4)}
        print(f"  apply_cnot({nq}q): {ms:.4f} ms")
    return results


def bench_meissner():
    """Benchmark: meissner_susceptibility with reduced field points."""
    from l104_god_code_simulator.quantum_primitives import (
        iron_lattice_heisenberg, meissner_susceptibility,
    )
    ms, res = _timer(
        lambda: meissner_susceptibility(
            lambda **kw: iron_lattice_heisenberg(n_sites=4, **kw),
            n_qubits=4,
        ),
        "meissner_4q",
        warmup=0,
        runs=1,
    )
    results = {
        "ms": round(ms, 1),
        "chi": round(res["susceptibility_chi"], 6),
        "field_points": len(res["field_values"]),
    }
    print(f"  meissner_susceptibility(4q): {ms:.1f} ms  χ={res['susceptibility_chi']:.4f}")
    return results


def main():
    print("=" * 72)
    print("L104 Simulation Speed Benchmark v5.1")
    print("=" * 72)

    all_results = {}
    benchmarks = [
        ("reconstruct_density_matrix", bench_reconstruct_density_matrix),
        ("trotter_evolution", bench_trotter_evolution),
        ("topological_entropy", bench_topological_entropy),
        ("singlet_projection", bench_singlet_projection),
        ("build_unitary", bench_build_unitary),
        ("apply_cnot_cached", bench_apply_cnot_cached),
        ("expm_approx", bench_expm_approx),
        ("simulator_prob", bench_simulator_prob),
        ("meissner_susceptibility", bench_meissner),
        ("phase_sweep_parallel", bench_phase_sweep),
        ("god_code_run_all", bench_god_code_run_all),
    ]

    total_t0 = time.perf_counter()

    for name, fn in benchmarks:
        print(f"\n▸ {name}")
        try:
            all_results[name] = fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[name] = {"error": str(e)}

    total_ms = (time.perf_counter() - total_t0) * 1000

    print(f"\n{'=' * 72}")
    print(f"Total benchmark time: {total_ms:.0f} ms")
    print(f"{'=' * 72}")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "_bench_simulation_speed_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
