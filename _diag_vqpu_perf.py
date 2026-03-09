#!/usr/bin/env python3
"""VQPU Performance Diagnostic — find what's slowing down the MacBook."""
import time
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("L104 VQPU PERFORMANCE DIAGNOSTIC")
print("=" * 60)

# 0. Platform info
import platform
print(f"\nPlatform: {platform.machine()} / {platform.processor()}")
print(f"Python: {sys.version}")

import psutil
mem = psutil.virtual_memory()
print(f"RAM: {mem.total/1024**3:.1f}GB total, {mem.available/1024**3:.1f}GB free, {mem.percent}% used")
print(f"CPU cores: {psutil.cpu_count()}")

# 1. Import overhead
print("\n--- IMPORT OVERHEAD ---")
t0 = time.monotonic()
import l104_vqpu.constants as c
t1 = time.monotonic()
print(f"constants module: {(t1-t0)*1000:.1f}ms")
print(f"  is_intel={c._IS_INTEL} is_apple_silicon={c._IS_APPLE_SILICON}")
print(f"  metal_compute={c._HAS_METAL_COMPUTE} gpu_class={c._GPU_CLASS}")
print(f"  max_qubits={c.VQPU_MAX_QUBITS} gpu_crossover={c.VQPU_GPU_CROSSOVER}")
print(f"  mps_fallback={c.VQPU_MPS_FALLBACK_TARGET}")
print(f"  pipeline_workers={c.VQPU_PIPELINE_WORKERS}")
print(f"  metal_family={c._PLATFORM.get('metal_family', 0)}")

t0 = time.monotonic()
from l104_vqpu.mps_engine import ExactMPSHybridEngine
t1 = time.monotonic()
print(f"MPS engine import: {(t1-t0)*1000:.1f}ms")

t0 = time.monotonic()
from l104_vqpu.transpiler import CircuitTranspiler, CircuitAnalyzer
t1 = time.monotonic()
print(f"Transpiler import: {(t1-t0)*1000:.1f}ms")

t0 = time.monotonic()
from l104_vqpu.three_engine import ThreeEngineQuantumScorer, EngineIntegration
t1 = time.monotonic()
print(f"Three-engine import: {(t1-t0)*1000:.1f}ms")

# 2. MPS Engine Performance
print("\n--- MPS ENGINE ---")
import numpy as np

H = np.array([[1,1],[1,-1]], dtype=np.complex128) / np.sqrt(2)
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex128).reshape(2,2,2,2)

# Bell state
t0 = time.monotonic()
mps = ExactMPSHybridEngine(2)
mps.apply_single_gate(0, H)
mps.apply_two_gate(0, 1, CNOT)
counts = mps.sample(1024)
print(f"Bell (2Q): {(time.monotonic()-t0)*1000:.1f}ms")

# GHZ-8
t0 = time.monotonic()
mps8 = ExactMPSHybridEngine(8)
for q in range(8):
    mps8.apply_single_gate(q, H)
for q in range(7):
    mps8.apply_two_gate(q, q+1, CNOT)
counts8 = mps8.sample(1024)
print(f"GHZ-8 (8Q): {(time.monotonic()-t0)*1000:.1f}ms, peak_chi={mps8._peak_chi}")

# SV extraction at different sizes
for nq in [4, 8, 12]:
    t0 = time.monotonic()
    m = ExactMPSHybridEngine(nq)
    m.apply_single_gate(0, H)
    if nq >= 2:
        m.apply_two_gate(0, 1, CNOT)
    sv = m.to_statevector()
    print(f"SV extraction {nq}Q: {(time.monotonic()-t0)*1000:.1f}ms, sv_size={len(sv)}")

# 3. Transpiler performance
print("\n--- TRANSPILER ---")
small_ops = [{"gate": "H", "qubits": [0]}, {"gate": "CX", "qubits": [0, 1]}]
medium_ops = []
for i in range(50):
    medium_ops.append({"gate": "H", "qubits": [i % 4]})
    medium_ops.append({"gate": "CX", "qubits": [i % 4, (i+1) % 4]})
    medium_ops.append({"gate": "Rz", "qubits": [i % 4], "parameters": [0.5]})
    medium_ops.append({"gate": "Rz", "qubits": [i % 4], "parameters": [0.3]})

t0 = time.monotonic()
r = CircuitTranspiler.transpile(small_ops)
print(f"Small (2 ops): {(time.monotonic()-t0)*1000:.3f}ms -> {len(r)} ops")

t0 = time.monotonic()
r = CircuitTranspiler.transpile(medium_ops)
print(f"Medium (200 ops): {(time.monotonic()-t0)*1000:.3f}ms -> {len(r)} ops (saved {200-len(r)})")

# 4. Scoring bottleneck
print("\n--- SCORING BOTTLENECK ---")

# SC score (first call = most expensive)
t0 = time.monotonic()
try:
    sc = ThreeEngineQuantumScorer.sc_score()
    print(f"SC score (UNCACHED): {(time.monotonic()-t0)*1000:.1f}ms => {sc:.4f}")
except Exception as e:
    print(f"SC score FAILED: {e} ({(time.monotonic()-t0)*1000:.1f}ms)")

# Entropy score
t0 = time.monotonic()
try:
    es = ThreeEngineQuantumScorer.entropy_score(1.0)
    print(f"Entropy score (UNCACHED): {(time.monotonic()-t0)*1000:.1f}ms => {es:.4f}")
except Exception as e:
    print(f"Entropy score FAILED: {e} ({(time.monotonic()-t0)*1000:.1f}ms)")

# Harmonic score
t0 = time.monotonic()
try:
    hs = ThreeEngineQuantumScorer.harmonic_score()
    print(f"Harmonic score (UNCACHED): {(time.monotonic()-t0)*1000:.1f}ms => {hs:.4f}")
except Exception as e:
    print(f"Harmonic score FAILED: {e} ({(time.monotonic()-t0)*1000:.1f}ms)")

# Wave score
t0 = time.monotonic()
try:
    ws = ThreeEngineQuantumScorer.wave_score()
    print(f"Wave score (UNCACHED): {(time.monotonic()-t0)*1000:.1f}ms => {ws:.4f}")
except Exception as e:
    print(f"Wave score FAILED: {e} ({(time.monotonic()-t0)*1000:.1f}ms)")

# Full composite (should be cached now)
t0 = time.monotonic()
try:
    comp = ThreeEngineQuantumScorer.composite_score(1.0)
    print(f"Composite (CACHED): {(time.monotonic()-t0)*1000:.1f}ms")
    print(f"  Result: {comp}")
except Exception as e:
    print(f"Composite FAILED: {e}")

# 5. Full pipeline test
print("\n--- FULL PIPELINE (run_simulation) ---")
from l104_vqpu.bridge import VQPUBridge
from l104_vqpu.types import QuantumJob

bridge = VQPUBridge(enable_daemon_cycler=False, enable_governor=False)

bell_job = QuantumJob(num_qubits=2, operations=[
    {"gate": "H", "qubits": [0]},
    {"gate": "CX", "qubits": [0, 1]},
])

# First run (cold — loads all engines)
t0 = time.monotonic()
sim1 = bridge.run_simulation(bell_job, compile=False, score_asi=False, score_agi=False)
t1 = time.monotonic()
print(f"Pipeline Bell (cold, no compile/ASI/AGI): {(t1-t0)*1000:.1f}ms")
for key in ["transpile_ms", "execute_ms", "score_ms", "total_ms"]:
    v = sim1.get("pipeline", {}).get(key, "N/A")
    print(f"  {key}: {v}")

# Second run (warm)
t0 = time.monotonic()
sim2 = bridge.run_simulation(bell_job, compile=False, score_asi=False, score_agi=False)
t1 = time.monotonic()
print(f"Pipeline Bell (warm): {(t1-t0)*1000:.1f}ms")
for key in ["transpile_ms", "execute_ms", "score_ms", "total_ms"]:
    v = sim2.get("pipeline", {}).get(key, "N/A")
    print(f"  {key}: {v}")

# With compile + ASI + AGI
t0 = time.monotonic()
sim3 = bridge.run_simulation(bell_job, compile=True, score_asi=True, score_agi=True)
t1 = time.monotonic()
print(f"Pipeline Bell (full scoring): {(t1-t0)*1000:.1f}ms")
for key in ["transpile_ms", "compile_ms", "execute_ms", "score_ms", "sc_analysis_ms", "total_ms"]:
    v = sim3.get("pipeline", {}).get(key, "N/A")
    print(f"  {key}: {v}")

# 6. Check NoiseModel readout performance
print("\n--- NOISE MODEL ---")
from l104_vqpu.scoring import NoiseModel
nm = NoiseModel(readout_error_rate=0.01)
test_counts = {"00": 10000, "11": 10000}

t0 = time.monotonic()
noisy = nm.apply_readout_noise(test_counts, 2)
print(f"Readout noise (20K shots, 2Q): {(time.monotonic()-t0)*1000:.1f}ms")

test_counts_big = {"0000": 50000, "1111": 50000}
t0 = time.monotonic()
noisy_big = nm.apply_readout_noise(test_counts_big, 4)
print(f"Readout noise (100K shots, 4Q): {(time.monotonic()-t0)*1000:.1f}ms")

# 7. Entanglement quantification
print("\n--- ENTANGLEMENT ---")
try:
    from l104_vqpu.entanglement import EntanglementQuantifier
    mps2 = ExactMPSHybridEngine(2)
    mps2.apply_single_gate(0, H)
    mps2.apply_two_gate(0, 1, CNOT)
    sv2 = mps2.to_statevector()
    t0 = time.monotonic()
    eq = EntanglementQuantifier(sv2, 2)
    vne = eq.von_neumann_entropy(0)
    conc = eq.concurrence()
    print(f"Entanglement 2Q: {(time.monotonic()-t0)*1000:.1f}ms (VNE={vne:.4f}, C={conc:.4f})")

    mps8 = ExactMPSHybridEngine(8)
    for q in range(8):
        mps8.apply_single_gate(q, H)
    for q in range(7):
        mps8.apply_two_gate(q, q+1, CNOT)
    sv8 = mps8.to_statevector()
    t0 = time.monotonic()
    eq8 = EntanglementQuantifier(sv8, 8)
    vne8 = eq8.von_neumann_entropy(0)
    print(f"Entanglement 8Q: {(time.monotonic()-t0)*1000:.1f}ms (VNE={vne8:.4f})")
except Exception as e:
    print(f"Entanglement error: {e}")

# 8. Memory final state
mem = psutil.virtual_memory()
print(f"\nFinal RAM: {mem.available/1024**3:.1f}GB free, {mem.percent}% used")

bridge.stop()
print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
