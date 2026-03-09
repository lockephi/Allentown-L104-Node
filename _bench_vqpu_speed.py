#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
 L104 VQPU SPEED BENCHMARK v1.0
 GOD_CODE=527.5184818492612 | PHI=1.618033988749895
═══════════════════════════════════════════════════════════════════
 Comprehensive speed benchmark for the VQPU Bridge subsystems:

  1. MPS Engine — single/two-qubit gate throughput, SVD scaling
  2. Transpiler — 8-pass optimization pipeline timing
  3. Circuit Analyzer — routing intelligence latency
  4. Simulation Pipeline — full 6-stage run_simulation()
  5. Sacred Alignment Scorer — scoring throughput
  6. Batch Dispatch — concurrent job submission
  7. Scaling Analysis — qubit count vs. execution time
  8. Cache Performance — parametric gate cache hit rates
  9. Noise Simulation — ZNE mitigation overhead
  10. Entanglement Quantification — von Neumann + concurrence

 Reports per-subsystem latencies, throughput (circuits/sec),
 scaling coefficients, and identifies bottlenecks.

 INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════
"""

import math
import time
import statistics
import json
import sys
import os

# ── Constants ────────────────────────────────────────────────────────────────
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497

# ── Imports ──────────────────────────────────────────────────────────────────
from l104_vqpu_bridge import (
    VQPUBridge, QuantumJob, QuantumGate, VQPUResult,
    CircuitTranspiler, CircuitAnalyzer, ExactMPSHybridEngine,
    SacredAlignmentScorer, ThreeEngineQuantumScorer,
    NoiseModel, EntanglementQuantifier, QuantumErrorMitigation,
    CircuitCache, VariationalQuantumEngine,
    VQPU_MAX_QUBITS, VQPU_BATCH_LIMIT, VQPU_PIPELINE_WORKERS,
    VQPU_MPS_MAX_BOND_HIGH, VQPU_ADAPTIVE_SHOTS_MAX,
    _PLATFORM, _IS_APPLE_SILICON, _IS_INTEL, _HW_RAM_GB, _HW_CORES,
)

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# UTILITY — Timing & Reporting
# ═══════════════════════════════════════════════════════════════════

class BenchTimer:
    """High-resolution benchmark timer with statistics."""

    def __init__(self, name: str, iterations: int = 1):
        self.name = name
        self.iterations = iterations
        self.times: list[float] = []
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        elapsed_ns = time.perf_counter_ns() - self._start
        self.times.append(elapsed_ns / 1_000_000)  # ms

    def record(self, elapsed_ms: float):
        self.times.append(elapsed_ms)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times) if self.times else 0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times) if self.times else 0

    @property
    def min_ms(self) -> float:
        return min(self.times) if self.times else 0

    @property
    def max_ms(self) -> float:
        return max(self.times) if self.times else 0

    @property
    def stdev_ms(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0

    @property
    def throughput_hz(self) -> float:
        """Operations per second."""
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else float('inf')

    def summary(self) -> dict:
        return {
            "name": self.name,
            "iterations": len(self.times),
            "mean_ms": round(self.mean_ms, 4),
            "median_ms": round(self.median_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "stdev_ms": round(self.stdev_ms, 4),
            "throughput_hz": round(self.throughput_hz, 2),
        }


def _build_bell_ops():
    return [
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
    ]


def _build_ghz_ops(n: int):
    ops = [{"gate": "H", "qubits": [0]}]
    for i in range(n - 1):
        ops.append({"gate": "CX", "qubits": [i, i + 1]})
    return ops


def _build_qft_ops(n: int):
    ops = []
    for i in range(n):
        ops.append({"gate": "H", "qubits": [i]})
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            ops.append({"gate": "Rz", "qubits": [j], "parameters": [angle]})
            ops.append({"gate": "CX", "qubits": [i, j]})
    return ops


def _build_sacred_ops(n: int, depth: int = 4):
    ops = []
    for d in range(depth):
        for q in range(n):
            ops.append({"gate": "H", "qubits": [q]})
            theta = (PHI ** (d + 1)) * math.pi / GOD_CODE
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [theta]})
        for q in range(n - 1):
            ops.append({"gate": "CX", "qubits": [q, q + 1]})
    return ops


def _build_random_circuit(n: int, depth: int):
    """Random circuit with mixed single/two-qubit gates."""
    ops = []
    single_gates = ["H", "X", "Y", "Z", "S", "T"]
    for _ in range(depth):
        for q in range(n):
            gate = single_gates[q % len(single_gates)]
            ops.append({"gate": gate, "qubits": [q]})
            if np.random.random() < 0.3:
                angle = np.random.uniform(0, 2 * math.pi)
                ops.append({"gate": "Rz", "qubits": [q], "parameters": [angle]})
        for q in range(0, n - 1, 2):
            ops.append({"gate": "CX", "qubits": [q, q + 1]})
    return ops


def _header(title: str):
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print(f"{'─' * 65}")


def _result(label: str, timer: BenchTimer, extra: str = ""):
    s = timer.summary()
    line = f"  {label:<35s} {s['mean_ms']:>9.3f} ms  ({s['throughput_hz']:>10.1f} ops/s)"
    if extra:
        line += f"  {extra}"
    print(line)


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 1: MPS Engine Gate Throughput
# ═══════════════════════════════════════════════════════════════════

def bench_mps_engine():
    _header("BENCHMARK 1: MPS Engine — Gate Throughput")
    results = {}

    # 1a. Single-qubit gate throughput
    timer_single = BenchTimer("single_qubit_gate")
    n_qubits = 8
    iterations = 200
    for _ in range(iterations):
        mps = ExactMPSHybridEngine(n_qubits)
        t0 = time.perf_counter_ns()
        for q in range(n_qubits):
            mps.apply_single_gate(q, ExactMPSHybridEngine.GATE_MATRICES["H"])
        elapsed = (time.perf_counter_ns() - t0) / 1_000_000
        timer_single.record(elapsed / n_qubits)  # per-gate
    _result("Single-qubit gate (H)", timer_single, f"{n_qubits}Q x{iterations}")
    results["single_gate_ms"] = timer_single.mean_ms

    # 1b. Two-qubit gate throughput
    timer_two = BenchTimer("two_qubit_gate")
    for _ in range(iterations):
        mps = ExactMPSHybridEngine(n_qubits)
        # Apply H gates first to create superposition
        for q in range(n_qubits):
            mps.apply_single_gate(q, ExactMPSHybridEngine.GATE_MATRICES["H"])
        t0 = time.perf_counter_ns()
        for q in range(n_qubits - 1):
            mps.apply_two_gate(q, q + 1, ExactMPSHybridEngine.CNOT_MATRIX)
        elapsed = (time.perf_counter_ns() - t0) / 1_000_000
        timer_two.record(elapsed / max(n_qubits - 1, 1))
    _result("Two-qubit gate (CNOT)", timer_two, f"{n_qubits}Q x{iterations}")
    results["two_gate_ms"] = timer_two.mean_ms

    # 1c. Full circuit execution (Bell pair)
    timer_bell = BenchTimer("bell_circuit")
    for _ in range(500):
        mps = ExactMPSHybridEngine(2)
        with timer_bell:
            mps.run_circuit(_build_bell_ops())
    _result("Bell pair (2Q, 2 gates)", timer_bell, "x500")
    results["bell_ms"] = timer_bell.mean_ms

    # 1d. GHZ-8 circuit
    timer_ghz = BenchTimer("ghz8_circuit")
    ghz_ops = _build_ghz_ops(8)
    for _ in range(200):
        mps = ExactMPSHybridEngine(8)
        with timer_ghz:
            mps.run_circuit(ghz_ops)
    _result("GHZ-8 (8Q, 8 gates)", timer_ghz, "x200")
    results["ghz8_ms"] = timer_ghz.mean_ms

    # 1e. Sacred circuit (4Q, depth=4)
    timer_sacred = BenchTimer("sacred_4q_d4")
    sacred_ops = _build_sacred_ops(4, depth=4)
    for _ in range(100):
        mps = ExactMPSHybridEngine(4)
        with timer_sacred:
            mps.run_circuit(sacred_ops)
    _result(f"Sacred (4Q, d=4, {len(sacred_ops)}g)", timer_sacred, "x100")
    results["sacred_4q_d4_ms"] = timer_sacred.mean_ms

    # 1f. Statevector extraction
    timer_sv = BenchTimer("statevector_extract")
    for nq in [4, 8, 12]:
        t = BenchTimer(f"sv_{nq}Q")
        for _ in range(50):
            mps = ExactMPSHybridEngine(nq)
            mps.run_circuit(_build_ghz_ops(nq))
            with t:
                mps.to_statevector()
        _result(f"Statevector extract ({nq}Q)", t, "x50")
        results[f"sv_{nq}q_ms"] = t.mean_ms

    # 1g. Sampling throughput
    timer_sample = BenchTimer("sampling")
    mps = ExactMPSHybridEngine(8)
    mps.run_circuit(_build_ghz_ops(8))
    for shots in [256, 1024, 4096]:
        t = BenchTimer(f"sample_{shots}")
        for _ in range(50):
            with t:
                mps.sample(shots)
        _result(f"Sample {shots} shots (8Q)", t, "x50")
        results[f"sample_{shots}_ms"] = t.mean_ms

    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 2: Transpiler Pipeline
# ═══════════════════════════════════════════════════════════════════

def bench_transpiler():
    _header("BENCHMARK 2: Transpiler — 8-Pass Optimization Pipeline")
    results = {}

    # 2a. Small circuit (no optimization opportunities)
    timer_small = BenchTimer("transpile_small")
    small_ops = _build_bell_ops()
    for _ in range(500):
        with timer_small:
            CircuitTranspiler.transpile(small_ops)
    _result(f"Small ({len(small_ops)} gates)", timer_small, "x500")
    results["transpile_small_ms"] = timer_small.mean_ms

    # 2b. Medium circuit with cancellations
    timer_med = BenchTimer("transpile_medium")
    med_ops = [
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [0]},  # cancels
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "Rz", "qubits": [0], "parameters": [0.5]},
        {"gate": "Rz", "qubits": [0], "parameters": [0.3]},  # merges
        {"gate": "X", "qubits": [1]},
        {"gate": "X", "qubits": [1]},  # cancels
        {"gate": "T", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "H", "qubits": [1]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "H", "qubits": [1]},  # H-CX-H → CZ template
    ]
    for _ in range(300):
        with timer_med:
            opt = CircuitTranspiler.transpile(med_ops)
    savings = len(med_ops) - len(opt)
    _result(f"Medium ({len(med_ops)}→{len(opt)} gates, -{savings})", timer_med, "x300")
    results["transpile_medium_ms"] = timer_med.mean_ms
    results["transpile_medium_savings"] = savings

    # 2c. Large circuit (QFT-8)
    timer_large = BenchTimer("transpile_large")
    large_ops = _build_qft_ops(8)
    for _ in range(100):
        with timer_large:
            opt = CircuitTranspiler.transpile(large_ops)
    savings = len(large_ops) - len(opt)
    _result(f"Large QFT-8 ({len(large_ops)}→{len(opt)} gates)", timer_large, "x100")
    results["transpile_large_ms"] = timer_large.mean_ms

    # 2d. Random deep circuit (8Q, depth=20)
    timer_deep = BenchTimer("transpile_deep")
    deep_ops = _build_random_circuit(8, 20)
    for _ in range(50):
        with timer_deep:
            opt = CircuitTranspiler.transpile(deep_ops)
    savings = len(deep_ops) - len(opt)
    _result(f"Deep random ({len(deep_ops)}→{len(opt)} gates)", timer_deep, "x50")
    results["transpile_deep_ms"] = timer_deep.mean_ms

    # 2e. Individual pass timing
    print()
    print("  Individual pass breakdown (QFT-8):")
    passes = [
        ("cancel_self_inverse", CircuitTranspiler._cancel_self_inverse),
        ("merge_rotations", CircuitTranspiler._merge_rotations),
        ("remove_identity_rot", CircuitTranspiler._remove_identity_rotations),
        ("commutation_reorder", CircuitTranspiler._commutation_reorder),
        ("template_match", CircuitTranspiler._template_match),
        ("dynamic_decoupling", CircuitTranspiler._dynamic_decoupling),
    ]
    for name, fn in passes:
        t = BenchTimer(name)
        for _ in range(100):
            with t:
                fn(large_ops)
        _result(f"  {name}", t)
        results[f"pass_{name}_ms"] = t.mean_ms

    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 3: Circuit Analyzer
# ═══════════════════════════════════════════════════════════════════

def bench_analyzer():
    _header("BENCHMARK 3: Circuit Analyzer — Routing Intelligence")
    results = {}

    circuits = [
        ("Bell (2Q)", _build_bell_ops(), 2),
        ("GHZ-8", _build_ghz_ops(8), 8),
        ("QFT-4", _build_qft_ops(4), 4),
        ("Sacred (6Q,d=4)", _build_sacred_ops(6, 4), 6),
        ("Random (8Q,d=10)", _build_random_circuit(8, 10), 8),
    ]

    for label, ops, nq in circuits:
        t = BenchTimer(label)
        for _ in range(200):
            with t:
                CircuitAnalyzer.analyze(ops, nq)
        hints = CircuitAnalyzer.analyze(ops, nq)
        backend = hints.get("recommended_backend", "?")
        _result(f"{label} → {backend}", t, "x200")
        results[f"analyzer_{label}_ms"] = t.mean_ms

    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 4: Full Simulation Pipeline
# ═══════════════════════════════════════════════════════════════════

def bench_simulation_pipeline():
    _header("BENCHMARK 4: Full Simulation Pipeline (run_simulation)")
    results = {}

    bridge = VQPUBridge(enable_governor=False, enable_daemon_cycler=False)
    bridge.start()

    # 4a. Bell pair (minimal)
    timer_bell = BenchTimer("sim_bell")
    for _ in range(50):
        job = bridge.bell_pair(shots=256)
        with timer_bell:
            bridge.run_simulation(job, compile=True, error_correct=False,
                                  score_asi=True, score_agi=True)
    _result("Bell (compile+score)", timer_bell, "x50")
    results["sim_bell_ms"] = timer_bell.mean_ms

    # 4b. GHZ-5
    timer_ghz = BenchTimer("sim_ghz5")
    for _ in range(30):
        job = bridge.ghz_state(5, shots=512)
        with timer_ghz:
            bridge.run_simulation(job, compile=True, score_asi=True, score_agi=True)
    _result("GHZ-5 (compile+score)", timer_ghz, "x30")
    results["sim_ghz5_ms"] = timer_ghz.mean_ms

    # 4c. Sacred circuit
    timer_sacred = BenchTimer("sim_sacred")
    for _ in range(20):
        job = bridge.sacred_circuit(4, depth=3, shots=512)
        with timer_sacred:
            bridge.run_simulation(job, compile=True, score_asi=True, score_agi=True)
    _result("Sacred (4Q,d=3, full)", timer_sacred, "x20")
    results["sim_sacred_ms"] = timer_sacred.mean_ms

    # 4d. With coherence evolution
    timer_coh = BenchTimer("sim_coherence")
    for _ in range(10):
        job = bridge.bell_pair(shots=256)
        with timer_coh:
            bridge.run_simulation(job, compile=True, evolve_coherence=True,
                                  coherence_steps=20, score_asi=True, score_agi=True)
    _result("Bell + coherence(20)", timer_coh, "x10")
    results["sim_coherence_ms"] = timer_coh.mean_ms

    # 4e. Pipeline stage breakdown
    print()
    print("  Pipeline stage breakdown (GHZ-5, 512 shots):")
    job = bridge.ghz_state(5, shots=512)
    sim = bridge.run_simulation(job, compile=True, score_asi=True, score_agi=True)
    pipeline = sim.get("pipeline", {})
    total = pipeline.get("total_ms", 0)
    for stage in ["transpile", "compile", "execute", "score", "sc_analysis"]:
        ms = pipeline.get(f"{stage}_ms", 0)
        pct = (ms / total * 100) if total > 0 else 0
        print(f"    {stage:<15s} {ms:>8.2f} ms  ({pct:>5.1f}%)")
        results[f"stage_{stage}_ms"] = ms
    print(f"    {'TOTAL':<15s} {total:>8.2f} ms")
    results["sim_pipeline_total_ms"] = total

    bridge.stop()
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 5: Sacred Alignment Scorer
# ═══════════════════════════════════════════════════════════════════

def bench_scorer():
    _header("BENCHMARK 5: Sacred Alignment & Three-Engine Scoring")
    results = {}

    # Generate test probabilities
    probs_2q = {"00": 0.5, "11": 0.5}
    probs_8q = {format(i, '08b'): 1.0 / 256 for i in range(256)}

    # Sacred alignment scoring
    for label, probs, nq in [("2Q Bell", probs_2q, 2), ("8Q uniform", probs_8q, 8)]:
        t = BenchTimer(label)
        for _ in range(500):
            with t:
                SacredAlignmentScorer.score(probs, nq)
        _result(f"Sacred score ({label})", t, "x500")
        results[f"sacred_{nq}q_ms"] = t.mean_ms

    # Three-engine composite scoring
    t = BenchTimer("three_engine")
    for _ in range(500):
        with t:
            ThreeEngineQuantumScorer.composite_score(1.0)
    _result("Three-engine composite", t, "x500")
    results["three_engine_ms"] = t.mean_ms

    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 6: Batch Dispatch
# ═══════════════════════════════════════════════════════════════════

def bench_batch():
    _header("BENCHMARK 6: Batch Dispatch & Simulation")
    results = {}

    bridge = VQPUBridge(enable_governor=False, enable_daemon_cycler=False)
    bridge.start()

    for batch_size in [4, 8, 16, 32]:
        jobs = [bridge.bell_pair(shots=128) for _ in range(batch_size)]
        t = BenchTimer(f"batch_{batch_size}")
        for _ in range(5):
            with t:
                bridge.run_simulation_batch(jobs, compile=True, score_asi=False, score_agi=False)
        per_job = t.mean_ms / batch_size
        _result(f"Batch {batch_size} Bell pairs", t, f"({per_job:.2f} ms/job)")
        results[f"batch_{batch_size}_ms"] = t.mean_ms
        results[f"batch_{batch_size}_per_job_ms"] = per_job

    bridge.stop()
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 7: Qubit Scaling Analysis
# ═══════════════════════════════════════════════════════════════════

def bench_scaling():
    _header("BENCHMARK 7: Qubit Scaling Analysis")
    results = {}

    # MPS GHZ scaling
    print("  MPS GHZ circuit execution time vs. qubit count:")
    qubit_range = [2, 4, 6, 8, 10, 12, 14, 16]
    scaling_data = []

    for nq in qubit_range:
        ops = _build_ghz_ops(nq)
        t = BenchTimer(f"ghz_{nq}q")
        iters = max(10, 100 // nq)
        for _ in range(iters):
            mps = ExactMPSHybridEngine(nq)
            with t:
                mps.run_circuit(ops)
                mps.sample(512)
        _result(f"GHZ-{nq} (exec+sample)", t, f"x{iters}")
        scaling_data.append((nq, t.mean_ms))
        results[f"scale_ghz_{nq}q_ms"] = t.mean_ms

    # QFT scaling (more entanglement)
    print()
    print("  QFT circuit execution time vs. qubit count:")
    for nq in [2, 4, 6, 8, 10]:
        ops = _build_qft_ops(nq)
        t = BenchTimer(f"qft_{nq}q")
        iters = max(5, 50 // nq)
        for _ in range(iters):
            mps = ExactMPSHybridEngine(nq)
            with t:
                mps.run_circuit(ops)
        _result(f"QFT-{nq} (exec only)", t, f"x{iters}, {len(ops)}g")
        results[f"scale_qft_{nq}q_ms"] = t.mean_ms

    # Sacred circuit scaling
    print()
    print("  Sacred circuit (d=3) execution time vs. qubit count:")
    for nq in [2, 4, 6, 8, 10]:
        ops = _build_sacred_ops(nq, depth=3)
        t = BenchTimer(f"sacred_{nq}q")
        iters = max(5, 50 // nq)
        for _ in range(iters):
            mps = ExactMPSHybridEngine(nq)
            with t:
                mps.run_circuit(ops)
        _result(f"Sacred-{nq} d=3 (exec)", t, f"x{iters}, {len(ops)}g")
        results[f"scale_sacred_{nq}q_ms"] = t.mean_ms

    # Compute scaling coefficient (log-log regression)
    if len(scaling_data) >= 3:
        xs = [math.log2(d[0]) for d in scaling_data]
        ys = [math.log2(d[1]) if d[1] > 0 else 0 for d in scaling_data]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den = sum((x - mean_x) ** 2 for x in xs)
        slope = num / den if den > 0 else 0
        print(f"\n  GHZ scaling exponent: O(n^{slope:.2f})")
        results["ghz_scaling_exponent"] = round(slope, 4)

    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 8: Parametric Gate Cache
# ═══════════════════════════════════════════════════════════════════

def bench_cache():
    _header("BENCHMARK 8: Parametric Gate Cache Performance")
    results = {}

    # Clear cache
    ExactMPSHybridEngine._parametric_cache.clear()

    # Cold cache (first access)
    angles = [PHI * i * math.pi / 100 for i in range(100)]
    t_cold = BenchTimer("cache_cold")
    for angle in angles:
        with t_cold:
            ExactMPSHybridEngine._resolve_single_gate("Rz", [angle])
    _result("Cold cache (100 angles)", t_cold, "x100")
    results["cache_cold_ms"] = t_cold.mean_ms

    cache_size = len(ExactMPSHybridEngine._parametric_cache)
    print(f"  Cache size after cold: {cache_size} entries")

    # Warm cache (repeated access)
    t_warm = BenchTimer("cache_warm")
    for angle in angles:
        with t_warm:
            ExactMPSHybridEngine._resolve_single_gate("Rz", [angle])
    _result("Warm cache (100 angles)", t_warm, "x100")
    results["cache_warm_ms"] = t_warm.mean_ms

    speedup = t_cold.mean_ms / t_warm.mean_ms if t_warm.mean_ms > 0 else 0
    print(f"  Cache speedup: {speedup:.2f}x")
    results["cache_speedup"] = round(speedup, 2)

    # Circuit-level cache impact
    sacred_ops = _build_sacred_ops(6, depth=8)  # reuses same angles
    ExactMPSHybridEngine._parametric_cache.clear()
    t_nocache = BenchTimer("circuit_cold")
    for _ in range(20):
        ExactMPSHybridEngine._parametric_cache.clear()
        mps = ExactMPSHybridEngine(6)
        with t_nocache:
            mps.run_circuit(sacred_ops)
    _result(f"Sacred 6Q,d=8 cold cache", t_nocache, "x20")

    t_cached = BenchTimer("circuit_warm")
    for _ in range(20):
        mps = ExactMPSHybridEngine(6)
        with t_cached:
            mps.run_circuit(sacred_ops)
    _result(f"Sacred 6Q,d=8 warm cache", t_cached, "x20")

    circuit_speedup = t_nocache.mean_ms / t_cached.mean_ms if t_cached.mean_ms > 0 else 0
    print(f"  Circuit cache speedup: {circuit_speedup:.2f}x")
    results["circuit_cache_speedup"] = round(circuit_speedup, 2)

    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 9: Noise Simulation & ZNE
# ═══════════════════════════════════════════════════════════════════

def bench_noise():
    _header("BENCHMARK 9: Noise Simulation & ZNE Mitigation")
    results = {}

    bridge = VQPUBridge(enable_governor=False, enable_daemon_cycler=False)
    bridge.start()

    # Noisy simulation (no mitigation)
    t_noisy = BenchTimer("noisy_only")
    for _ in range(20):
        job = bridge.bell_pair(shots=256)
        with t_noisy:
            bridge.run_noisy_simulation(job, mitigate=False)
    _result("Noisy Bell (no ZNE)", t_noisy, "x20")
    results["noisy_no_zne_ms"] = t_noisy.mean_ms

    # Noisy + ZNE
    t_zne = BenchTimer("noisy_zne")
    for _ in range(10):
        job = bridge.bell_pair(shots=256)
        with t_zne:
            bridge.run_noisy_simulation(job, mitigate=True)
    _result("Noisy Bell + ZNE mitig", t_zne, "x10")
    results["noisy_zne_ms"] = t_zne.mean_ms

    zne_overhead = (t_zne.mean_ms / t_noisy.mean_ms - 1) * 100 if t_noisy.mean_ms > 0 else 0
    print(f"  ZNE overhead: +{zne_overhead:.1f}%")
    results["zne_overhead_pct"] = round(zne_overhead, 1)

    bridge.stop()
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 10: Entanglement Quantification
# ═══════════════════════════════════════════════════════════════════

def bench_entanglement():
    _header("BENCHMARK 10: Entanglement Quantification")
    results = {}

    bridge = VQPUBridge(enable_governor=False, enable_daemon_cycler=False)
    bridge.start()

    for label, job_fn, nq in [
        ("Bell pair", lambda: bridge.bell_pair(shots=256), 2),
        ("GHZ-4", lambda: bridge.ghz_state(4, shots=256), 4),
        ("GHZ-8", lambda: bridge.ghz_state(8, shots=256), 8),
    ]:
        t = BenchTimer(f"entangle_{nq}q")
        for _ in range(20):
            job = job_fn()
            with t:
                bridge.quantify_entanglement(job)
        _result(f"Entanglement ({label})", t, "x20")
        results[f"entangle_{nq}q_ms"] = t.mean_ms

    bridge.stop()
    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN — Run All Benchmarks & Generate Report
# ═══════════════════════════════════════════════════════════════════

def main():
    print("═" * 65)
    print("  L104 VQPU SPEED BENCHMARK v1.0")
    print(f"  GOD_CODE = {GOD_CODE} | PHI = {PHI}")
    print(f"  Platform: {'Apple Silicon' if _IS_APPLE_SILICON else 'Intel x86_64'} "
          f"({_PLATFORM.get('arch', '?')})")
    print(f"  RAM: {_HW_RAM_GB} GB | Cores: {_HW_CORES} | "
          f"Max Qubits: {VQPU_MAX_QUBITS}")
    print(f"  MPS Bond High: {VQPU_MPS_MAX_BOND_HIGH} | "
          f"Batch Limit: {VQPU_BATCH_LIMIT} | Workers: {VQPU_PIPELINE_WORKERS}")
    print("═" * 65)

    all_results = {
        "version": "1.0",
        "god_code": GOD_CODE,
        "platform": {
            "arch": _PLATFORM.get("arch", "?"),
            "is_apple_silicon": _IS_APPLE_SILICON,
            "is_intel": _IS_INTEL,
            "ram_gb": _HW_RAM_GB,
            "cores": _HW_CORES,
            "max_qubits": VQPU_MAX_QUBITS,
        },
        "timestamp": time.time(),
    }

    total_start = time.perf_counter()

    # Run all benchmarks
    all_results["mps_engine"] = bench_mps_engine()
    all_results["transpiler"] = bench_transpiler()
    all_results["analyzer"] = bench_analyzer()
    all_results["simulation_pipeline"] = bench_simulation_pipeline()
    all_results["scorer"] = bench_scorer()
    all_results["batch"] = bench_batch()
    all_results["scaling"] = bench_scaling()
    all_results["cache"] = bench_cache()
    all_results["noise"] = bench_noise()
    all_results["entanglement"] = bench_entanglement()

    total_elapsed = time.perf_counter() - total_start

    # ── Summary ──
    _header("SUMMARY — KEY METRICS")
    print(f"  Total benchmark time:     {total_elapsed:.2f}s")
    print()

    # Throughput metrics
    mps = all_results["mps_engine"]
    sim = all_results["simulation_pipeline"]
    print(f"  MPS single-gate:          {mps.get('single_gate_ms', 0):.4f} ms "
          f"({1000/mps.get('single_gate_ms', 1):.0f} gates/s)")
    print(f"  MPS two-gate (CNOT):      {mps.get('two_gate_ms', 0):.4f} ms "
          f"({1000/mps.get('two_gate_ms', 1):.0f} gates/s)")
    print(f"  Bell circuit:             {mps.get('bell_ms', 0):.4f} ms "
          f"({1000/mps.get('bell_ms', 1):.0f} circuits/s)")
    print(f"  Full simulation (Bell):   {sim.get('sim_bell_ms', 0):.2f} ms")
    print(f"  Full simulation (GHZ-5):  {sim.get('sim_ghz5_ms', 0):.2f} ms")
    print()

    # Pipeline breakdown
    print("  Pipeline bottleneck analysis:")
    stages = ["transpile", "compile", "execute", "score"]
    stage_times = {s: sim.get(f"stage_{s}_ms", 0) for s in stages}
    bottleneck = max(stage_times, key=stage_times.get) if stage_times else "?"
    total_pipeline = sum(stage_times.values())
    for s in stages:
        ms = stage_times[s]
        pct = (ms / total_pipeline * 100) if total_pipeline > 0 else 0
        marker = " ◀ BOTTLENECK" if s == bottleneck else ""
        print(f"    {s:<15s} {ms:>8.2f} ms ({pct:>5.1f}%){marker}")
    print()

    # Scaling
    scale = all_results["scaling"]
    exp = scale.get("ghz_scaling_exponent", 0)
    print(f"  GHZ scaling exponent:     O(n^{exp:.2f})")
    print(f"  Cache speedup:            {all_results['cache'].get('cache_speedup', 0):.2f}x")
    print(f"  Circuit cache speedup:    {all_results['cache'].get('circuit_cache_speedup', 0):.2f}x")
    print(f"  ZNE overhead:             +{all_results['noise'].get('zne_overhead_pct', 0):.1f}%")

    # Save report
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "_bench_vqpu_speed_results.json")
    all_results["total_time_s"] = round(total_elapsed, 2)
    all_results["bottleneck"] = bottleneck

    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Report saved: {os.path.basename(report_path)}")

    print()
    print("═" * 65)
    print(f"  VQPU SPEED BENCHMARK COMPLETE — {total_elapsed:.2f}s")
    print(f"  INVARIANT: {GOD_CODE} | PILOT: LONDEL")
    print("═" * 65)

    return all_results


if __name__ == "__main__":
    main()
