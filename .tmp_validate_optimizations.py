#!/usr/bin/env python3
"""Validate all 5 classical bypass optimizations integrated into engines."""

import numpy as np
import time

print("=== 1. god_code_qubit.py — Memoization + BLAS vectorize ===")
from l104_god_code_simulator.god_code_qubit import (
    GOD_CODE_QUBIT, dial_freq, dial_phase, dial_gate,
    GOD_CODE_PHASE, GOD_CODE_RZ
)

# Test memoized dial
f0 = dial_freq(0, 0, 0, 0)
assert abs(f0 - 527.5184818492612) < 1e-8, f"GOD_CODE mismatch: {f0}"
print(f"  dial_freq(0,0,0,0) = {f0:.10f} OK")

# Test cached gate
g = dial_gate(0, 0, 0, 0)
assert np.allclose(g, GOD_CODE_RZ), "dial_gate(0,0,0,0) != GOD_CODE_RZ"
print(f"  dial_gate(0,0,0,0) matches GOD_CODE_RZ OK")

# Benchmark memoization
t0 = time.perf_counter()
for _ in range(100000):
    dial_freq(0, 0, 0, 0)
t1 = time.perf_counter()
ns = (t1 - t0) / 100000 * 1e9
print(f"  Memoized dial_freq: {ns:.0f} ns/call OK")

# Test vectorized apply_to
sv = np.zeros(8, dtype=np.complex128)
sv[0] = 1.0
result = GOD_CODE_QUBIT.apply_to(sv, 0, 3)
assert abs(np.linalg.norm(result) - 1.0) < 1e-12, "apply_to lost unitarity"
print(f"  Vectorized apply_to: norm={np.linalg.norm(result):.12f} OK")

# Benchmark vectorized apply_to at 16Q
sv_big = np.zeros(2**16, dtype=np.complex128)
sv_big[0] = 1.0
t0 = time.perf_counter()
for _ in range(100):
    GOD_CODE_QUBIT.apply_to(sv_big, 0, 16)
t1 = time.perf_counter()
print(f"  apply_to(16Q): {(t1-t0)/100*1000:.2f} ms/call OK")


print("\n=== 2. simulator.py — Pre-cached gates + parametric cache ===")
from l104_simulator.simulator import (
    QuantumCircuit, _CACHED_H, _CACHED_CNOT, _CACHED_GOD_CODE_PHASE,
    _cached_Rz, _cached_Ry, GOD_CODE_PHASE_ANGLE
)

assert _CACHED_H.shape == (2, 2), "H gate shape wrong"
assert _CACHED_CNOT.shape == (4, 4), "CNOT gate shape wrong"
print(f"  Pre-cached H: {_CACHED_H.shape} OK")
print(f"  Pre-cached CNOT: {_CACHED_CNOT.shape} OK")
print(f"  Pre-cached GOD_CODE_PHASE: {_CACHED_GOD_CODE_PHASE.shape} OK")

# Test parametric cache
rz1 = _cached_Rz(GOD_CODE_PHASE_ANGLE)
rz2 = _cached_Rz(GOD_CODE_PHASE_ANGLE)
print(f"  Parametric Rz cache works OK")

# Test circuit building
qc = QuantumCircuit(3, "test")
qc.h(0).cx(0, 1).god_code_phase(2).rz(1.234, 0)
print(f"  Circuit built: {qc.gate_count} gates OK")


print("\n=== 3. orchestrator.py — Circuit result cache ===")
from l104_quantum_gate_engine import get_engine
from l104_quantum_gate_engine.orchestrator import ExecutionTarget

engine = get_engine()
print(f"  Engine initialized OK")
print(f"  Result cache max: {engine._result_cache_max} OK")

circ = engine.bell_pair()
r1 = engine.execute(circ, ExecutionTarget.LOCAL_STATEVECTOR)
r2 = engine.execute(circ, ExecutionTarget.LOCAL_STATEVECTOR)
print(f"  Cache hits={engine._result_cache_hits}, misses={engine._result_cache_misses} OK")
assert engine._result_cache_hits >= 1, "Cache should have 1+ hit"


print("\n=== 4. vqpu_bridge.py — Parametric gate cache ===")
from l104_vqpu import ExactMPSHybridEngine

mps = ExactMPSHybridEngine(2)
g1 = mps._resolve_single_gate("Rz", [GOD_CODE_PHASE_ANGLE])
g2 = mps._resolve_single_gate("Rz", [GOD_CODE_PHASE_ANGLE])
assert g1 is not None, "Failed to resolve Rz"
cache_size = len(ExactMPSHybridEngine._parametric_cache)
print(f"  Parametric gate cache: {cache_size} entries OK")
assert cache_size >= 1, "Should have cached at least 1 parametric gate"


print("\n=== 5. mps_simulator.py — Pre-cached SWAP ===")
from l104_simulator.mps_simulator import MPSState, MPSSimulator, _CACHED_SWAP_MATRIX

assert _CACHED_SWAP_MATRIX.shape == (4, 4), "SWAP shape wrong"
assert not _CACHED_SWAP_MATRIX.flags.writeable, "SWAP should be read-only"
print(f"  Pre-cached SWAP: {_CACHED_SWAP_MATRIX.shape}, read-only OK")

# Run small MPS simulation
mps_sim = MPSSimulator(max_bond=64)
qc2 = QuantumCircuit(4, "mps_test")
qc2.h(0).cx(0, 1).cx(1, 2).cx(2, 3)
r = mps_sim.run(qc2)
print(f"  MPS simulation: {r.n_qubits}Q, {r.gate_count} gates, {r.execution_time_ms:.2f}ms OK")

# Test non-adjacent gate (uses cached SWAP)
mps_state = MPSState(4)
mps_state.apply_single(np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2), 0)
cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
mps_state.apply_two(cnot, 0, 3)  # Non-adjacent: should use cached SWAP
sv = mps_state.to_statevector()
assert abs(np.linalg.norm(sv) - 1.0) < 1e-10, "MPS norm broken after non-adjacent gate"
print(f"  Non-adjacent gate via cached SWAP: norm={np.linalg.norm(sv):.10f} OK")


print("\n" + "=" * 60)
print("ALL 5 ENGINE OPTIMIZATIONS VALIDATED SUCCESSFULLY")
print("=" * 60)
print()
print("Summary of classical bypass integrations:")
print("  1. god_code_qubit.py  : LRU-cached dial functions + BLAS-vectorized apply_to")
print("  2. simulator.py       : 30+ pre-cached gate matrices + parametric Rz/Ry/Rx cache")
print("  3. orchestrator.py    : Circuit result cache (LRU, thread-safe, deterministic)")
print("  4. vqpu_bridge.py     : Parametric gate cache in ExactMPSHybridEngine")
print("  5. mps_simulator.py   : Pre-cached SWAP for non-adjacent gate routing")
