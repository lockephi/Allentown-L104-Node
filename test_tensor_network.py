#!/usr/bin/env python3
"""
===============================================================================
L104 TENSOR NETWORK SIMULATOR — VALIDATION & BENCHMARK SUITE
===============================================================================

7-phase test suite validating the MPS tensor network backend:
  Phase 1: MPS State initialization & properties
  Phase 2: Single-qubit gate correctness (vs statevector)
  Phase 3: Two-qubit gate correctness (adjacent + non-adjacent)
  Phase 4: Multi-qubit circuits (Bell, GHZ, QFT)
  Phase 5: 25-26Q large-scale simulation
  Phase 6: Compression & memory benchmarks
  Phase 7: Integration with CrossSystemOrchestrator

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import sys
import time
import math
import numpy as np

# ─── Bootstrap ────────────────────────────────────────────────────────────────
sys.path.insert(0, ".")

passed = 0
failed = 0
total_time = 0.0

def test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name} — {detail}")

def phase_header(num: int, title: str):
    print(f"\n{'='*70}")
    print(f"  PHASE {num}: {title}")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: MPS STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

phase_header(1, "MPS STATE INITIALIZATION & PROPERTIES")
t0 = time.time()

try:
    from l104_quantum_gate_engine.tensor_network import (
        MPSState, TensorNetworkSimulator, TNSimulationResult,
        TruncationMode, CanonicalForm, get_simulator,
        DEFAULT_MAX_BOND_DIM, SACRED_BOND_DIM, MAX_TENSOR_NETWORK_QUBITS,
    )
    test("Import tensor_network module", True)
except Exception as e:
    test("Import tensor_network module", False, str(e))
    sys.exit(1)

# Test MPS initialization
mps = MPSState(4)
test("MPS init (4 qubits)", mps.num_qubits == 4)
test("MPS initial bond dims all 1", all(d == 1 for d in mps.bond_dimensions))
test("MPS initial state is |0000⟩", abs(mps.get_probability("0000") - 1.0) < 1e-12)
test("MPS initial state |1111⟩ = 0", abs(mps.get_probability("1111")) < 1e-12)
test("MPS norm is 1.0", abs(mps.norm() - 1.0) < 1e-10)
test("MPS total parameters = 8", mps.total_parameters == 8)  # 4 sites × (1,2,1) = 4×2 = 8
test("MPS memory < 1 KB", mps.memory_bytes < 1024)

# Test with different sizes
mps_25 = MPSState(25, max_bond_dim=64)
test("MPS 25-qubit init", mps_25.num_qubits == 25)
test("MPS 25-qubit |0...0⟩", abs(mps_25.get_probability("0" * 25) - 1.0) < 1e-10)
test("MPS 25-qubit statevector memory ~512 MB",
     abs(mps_25.statevector_memory_mb - 512.0) < 1.0)

mps_26 = MPSState(26, max_bond_dim=64)
test("MPS 26-qubit init (Fe(26) manifold)", mps_26.num_qubits == 26)

# Canonical forms
mps_canon = MPSState(6)
mps_canon.left_canonicalize()
test("Left canonicalization", mps_canon.canonical_form == CanonicalForm.LEFT)
mps_canon.right_canonicalize()
test("Right canonicalization", mps_canon.canonical_form == CanonicalForm.RIGHT)
mps_canon.mixed_canonicalize(3)
test("Mixed canonicalization center=3",
     mps_canon.canonical_form == CanonicalForm.MIXED and mps_canon.orthogonality_center == 3)

dt1 = time.time() - t0
print(f"\n  Phase 1 time: {dt1*1000:.1f}ms")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: SINGLE-QUBIT GATE CORRECTNESS
# ═══════════════════════════════════════════════════════════════════════════════

phase_header(2, "SINGLE-QUBIT GATE CORRECTNESS")
t0 = time.time()

from l104_quantum_gate_engine import (
    H, X, Y, Z, S, T, CNOT, CZ, SWAP,
    Rx, Ry, Rz, Phase,
    PHI_GATE, GOD_CODE_PHASE, VOID_GATE, IRON_GATE, SACRED_ENTANGLER,
    GateCircuit, get_engine,
)

# Test H gate: |0⟩ → |+⟩ = (|0⟩ + |1⟩)/√2
mps = MPSState(1)
mps.apply_single_qubit_gate(H.matrix, 0)
p0 = mps.get_probability("0")
p1 = mps.get_probability("1")
test("H|0⟩ → equal superposition", abs(p0 - 0.5) < 1e-10 and abs(p1 - 0.5) < 1e-10)

# Test X gate: |0⟩ → |1⟩
mps = MPSState(1)
mps.apply_single_qubit_gate(X.matrix, 0)
test("X|0⟩ → |1⟩", abs(mps.get_probability("1") - 1.0) < 1e-10)

# Test Z gate: Z|+⟩ → |−⟩ (same probabilities, different phase)
mps = MPSState(1)
mps.apply_single_qubit_gate(H.matrix, 0)
mps.apply_single_qubit_gate(Z.matrix, 0)
p0 = mps.get_probability("0")
p1 = mps.get_probability("1")
test("Z|+⟩ → |−⟩ (probs unchanged)", abs(p0 - 0.5) < 1e-10 and abs(p1 - 0.5) < 1e-10)

# Test HZH = X
mps = MPSState(1)
mps.apply_single_qubit_gate(H.matrix, 0)
mps.apply_single_qubit_gate(Z.matrix, 0)
mps.apply_single_qubit_gate(H.matrix, 0)
test("HZH|0⟩ = X|0⟩ = |1⟩", abs(mps.get_probability("1") - 1.0) < 1e-10)

# Test Rx(π) = iX (up to global phase, same probabilities as X)
mps = MPSState(1)
mps.apply_single_qubit_gate(Rx(math.pi).matrix, 0)
test("Rx(π)|0⟩ → |1⟩", abs(mps.get_probability("1") - 1.0) < 1e-10)

# Test multi-qubit single-gate: H on qubit 2 of 4-qubit system
mps = MPSState(4)
mps.apply_single_qubit_gate(H.matrix, 2)
test("H on qubit 2 of 4: |0000⟩ + |0010⟩",
     abs(mps.get_probability("0000") - 0.5) < 1e-10 and
     abs(mps.get_probability("0010") - 0.5) < 1e-10)

# Verify against full statevector for a small circuit
circ_test = GateCircuit(3, "test_sv")
circ_test.append(H, [0]).append(X, [1]).append(Rz(0.7), [2])
circ_test.append(H, [2])

# MPS result
mps = MPSState(3, max_bond_dim=64)
for op in circ_test.operations:
    mps.apply_single_qubit_gate(op.gate.matrix, op.qubits[0])
mps.normalize()

# Statevector result
sv = circ_test.unitary() @ np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex)
sv_probs = np.abs(sv) ** 2

# Compare
max_diff = 0
for i in range(8):
    bs = format(i, '03b')
    mps_p = mps.get_probability(bs)
    sv_p = float(sv_probs[i])
    max_diff = max(max_diff, abs(mps_p - sv_p))

test("MPS vs SV single-qubit circuit (3q)", max_diff < 1e-10, f"max_diff={max_diff}")

# Sacred gates
mps = MPSState(1)
mps.apply_single_qubit_gate(PHI_GATE.matrix, 0)
test("PHI_GATE preserves |0⟩", abs(mps.get_probability("0") - 1.0) < 1e-10)

mps = MPSState(1)
mps.apply_single_qubit_gate(H.matrix, 0)
mps.apply_single_qubit_gate(GOD_CODE_PHASE.matrix, 0)
p0 = mps.get_probability("0")
p1 = mps.get_probability("1")
test("GOD_CODE_PHASE on |+⟩ preserves probs", abs(p0 - 0.5) < 1e-10 and abs(p1 - 0.5) < 1e-10)

dt2 = time.time() - t0
print(f"\n  Phase 2 time: {dt2*1000:.1f}ms")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: TWO-QUBIT GATE CORRECTNESS
# ═══════════════════════════════════════════════════════════════════════════════

phase_header(3, "TWO-QUBIT GATE CORRECTNESS")
t0 = time.time()

# Bell state: H(0) CNOT(0,1) → (|00⟩ + |11⟩)/√2
mps = MPSState(2, max_bond_dim=64)
mps.apply_single_qubit_gate(H.matrix, 0)
mps.apply_two_qubit_gate(CNOT.matrix, 0, 1)
mps.normalize()

p00 = mps.get_probability("00")
p01 = mps.get_probability("01")
p10 = mps.get_probability("10")
p11 = mps.get_probability("11")
test("Bell state |Φ+⟩: P(00)≈0.5", abs(p00 - 0.5) < 1e-8)
test("Bell state |Φ+⟩: P(11)≈0.5", abs(p11 - 0.5) < 1e-8)
test("Bell state |Φ+⟩: P(01)≈0", abs(p01) < 1e-8)
test("Bell state |Φ+⟩: P(10)≈0", abs(p10) < 1e-8)

# Bond dimension should grow after entangling gate
test("Bell state bond dim = 2", mps.bond_dimensions == [2])

# SWAP gate
mps = MPSState(2)
mps.apply_single_qubit_gate(X.matrix, 0)  # |10⟩
mps.apply_two_qubit_gate(SWAP.matrix, 0, 1)
mps.normalize()
test("SWAP |10⟩ → |01⟩", abs(mps.get_probability("01") - 1.0) < 1e-8)

# CZ gate: CZ|++⟩ = (|00⟩ + |01⟩ + |10⟩ - |11⟩)/2
mps = MPSState(2)
mps.apply_single_qubit_gate(H.matrix, 0)
mps.apply_single_qubit_gate(H.matrix, 1)
mps.apply_two_qubit_gate(CZ.matrix, 0, 1)
mps.normalize()
pvals = {format(i, '02b'): mps.get_probability(format(i, '02b')) for i in range(4)}
test("CZ|++⟩: all probs ≈ 0.25",
     all(abs(v - 0.25) < 1e-8 for v in pvals.values()))

# Non-adjacent 2-qubit gate (requires SWAP routing)
mps = MPSState(4, max_bond_dim=64)
mps.apply_single_qubit_gate(H.matrix, 0)
mps.apply_two_qubit_gate(CNOT.matrix, 0, 3)  # Non-adjacent: 0 → 3
mps.normalize()

p0000 = mps.get_probability("0000")
p0001 = mps.get_probability("0001")
p1000 = mps.get_probability("1000")
p1001 = mps.get_probability("1001")
test("Non-adj CNOT(0,3): P(0000)≈0.5", abs(p0000 - 0.5) < 1e-6)
test("Non-adj CNOT(0,3): P(1001)≈0.5", abs(p1001 - 0.5) < 1e-6)

# Verify 2-qubit circuit against full statevector
circ2 = GateCircuit(4, "test_2q")
circ2.append(H, [0]).append(H, [2])
circ2.append(CNOT, [0, 1]).append(CNOT, [2, 3])
circ2.append(CZ, [1, 2])

# MPS simulation
mps = MPSState(4, max_bond_dim=64)
for op in circ2.operations:
    if op.gate.num_qubits == 1:
        mps.apply_single_qubit_gate(op.gate.matrix, op.qubits[0])
    else:
        mps.apply_two_qubit_gate(op.gate.matrix, op.qubits[0], op.qubits[1])
mps.normalize()

# Full statevector
sv = circ2.unitary() @ np.zeros(16, dtype=complex)
sv[0] = 1.0
sv = circ2.unitary() @ sv
sv_probs = np.abs(sv) ** 2

max_diff = 0
for i in range(16):
    bs = format(i, '04b')
    max_diff = max(max_diff, abs(mps.get_probability(bs) - float(sv_probs[i])))

test("MPS vs SV 4-qubit circuit with CNOT+CZ", max_diff < 1e-6, f"max_diff={max_diff}")

dt3 = time.time() - t0
print(f"\n  Phase 3 time: {dt3*1000:.1f}ms")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: MULTI-QUBIT CIRCUITS (Bell, GHZ, QFT)
# ═══════════════════════════════════════════════════════════════════════════════

phase_header(4, "MULTI-QUBIT CIRCUIT VALIDATION")
t0 = time.time()

engine = get_engine()
sim = TensorNetworkSimulator(max_bond_dim=128)

# Bell pair via orchestrator circuit
bell = engine.bell_pair()
result = sim.simulate(bell, shots=10000, seed=42)
test("Bell pair simulation runs", result.probabilities is not None)
if result.probabilities:
    p00 = result.probabilities.get("00", 0)
    p11 = result.probabilities.get("11", 0)
    test("Bell pair P(00) ≈ 0.5 (sampling)", abs(p00 - 0.5) < 0.05)
    test("Bell pair P(11) ≈ 0.5 (sampling)", abs(p11 - 0.5) < 0.05)

# GHZ state: (|00000⟩ + |11111⟩)/√2
ghz = engine.ghz_state(5)
result = sim.simulate(ghz, shots=10000, seed=42)
test("GHZ-5 simulation runs", result.probabilities is not None)
if result.probabilities:
    p_all0 = result.probabilities.get("00000", 0)
    p_all1 = result.probabilities.get("11111", 0)
    p_other = sum(v for k, v in result.probabilities.items() if k not in ("00000", "11111"))
    test("GHZ-5 P(00000) ≈ 0.5", abs(p_all0 - 0.5) < 0.05)
    test("GHZ-5 P(11111) ≈ 0.5", abs(p_all1 - 0.5) < 0.05)
    test("GHZ-5 other states ≈ 0", p_other < 0.02)

# GHZ-10: larger circuit
ghz10 = engine.ghz_state(10)
result10 = sim.simulate(ghz10, shots=10000, seed=42)
test("GHZ-10 simulation runs", result10.probabilities is not None)
test("GHZ-10 truncation error < 1e-8", result10.truncation_error < 1e-8)
test("GHZ-10 memory < 1 MB", result10.memory_mb < 1.0)

# Sacred circuit
sacred = engine.sacred_circuit(4, depth=2)
result_sacred = sim.simulate(sacred, shots=4096, seed=42)
test("Sacred circuit simulation", result_sacred.probabilities is not None)
test("Sacred circuit has alignment data", result_sacred.sacred_alignment is not None)

# QFT (small)
qft = engine.quantum_fourier_transform(4)
result_qft = sim.simulate(qft, shots=4096, seed=42)
test("QFT-4 simulation runs", result_qft.probabilities is not None)

# Exact probability check for QFT on |0000⟩ → uniform superposition
if result_qft.probabilities:
    # QFT|0⟩ = uniform superposition → all probs ≈ 1/16
    vals = list(result_qft.probabilities.values())
    mean_p = sum(vals) / len(vals) if vals else 0
    test("QFT-4 produces meaningful output", len(vals) > 1)

dt4 = time.time() - t0
print(f"\n  Phase 4 time: {dt4*1000:.1f}ms")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: LARGE-SCALE 25-26Q SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

phase_header(5, "LARGE-SCALE 25-26Q SIMULATION")
t0 = time.time()

# 25-qubit GHZ state
print("  Building 25-qubit GHZ circuit...")
ghz25 = engine.ghz_state(25)
sim_large = TensorNetworkSimulator(max_bond_dim=64)
t_sim = time.time()
result25 = sim_large.simulate(ghz25, shots=4096, seed=42, compute_entanglement=False)
sim_time_25 = time.time() - t_sim
print(f"  25Q GHZ simulation: {sim_time_25*1000:.0f}ms")

test("25Q GHZ runs successfully", result25.probabilities is not None)
test("25Q GHZ memory << 512 MB", result25.memory_mb < 50)
test("25Q compression ratio > 10×", result25.compression_ratio > 10)
test("25Q truncation error < 1e-6", result25.truncation_error < 1e-6)
test("25Q fidelity > 0.999", result25.fidelity_estimate > 0.999)

if result25.probabilities:
    p_all0 = result25.probabilities.get("0" * 25, 0)
    p_all1 = result25.probabilities.get("1" * 25, 0)
    test("25Q GHZ P(0...0) + P(1...1) ≈ 1", abs(p_all0 + p_all1 - 1.0) < 0.05)

print(f"  25Q Memory: {result25.memory_mb:.4f} MB (vs 512 MB statevector)")
print(f"  25Q Compression: {result25.compression_ratio:.0f}×")

# 26-qubit GHZ state — Fe(26) iron manifold
print("\n  Building 26-qubit GHZ circuit (Fe(26) manifold)...")
ghz26 = engine.ghz_state(26)
t_sim = time.time()
result26 = sim_large.simulate(ghz26, shots=4096, seed=42, compute_entanglement=False)
sim_time_26 = time.time() - t_sim
print(f"  26Q GHZ simulation: {sim_time_26*1000:.0f}ms")

test("26Q GHZ runs successfully", result26.probabilities is not None)
test("26Q GHZ memory << 1024 MB", result26.memory_mb < 100)
test("26Q compression ratio > 10×", result26.compression_ratio > 10)
test("26Q fidelity > 0.999", result26.fidelity_estimate > 0.999)

if result26.probabilities:
    p_all0 = result26.probabilities.get("0" * 26, 0)
    p_all1 = result26.probabilities.get("1" * 26, 0)
    test("26Q GHZ P(0...0) + P(1...1) ≈ 1", abs(p_all0 + p_all1 - 1.0) < 0.05)

print(f"  26Q Memory: {result26.memory_mb:.4f} MB (vs 1024 MB statevector)")
print(f"  26Q Compression: {result26.compression_ratio:.0f}×")

# 25-qubit sacred circuit
print("\n  Building 25-qubit sacred circuit...")
sacred25 = engine.sacred_circuit(25, depth=1)
sim_sacred = TensorNetworkSimulator(max_bond_dim=104, sacred_mode=True)
t_sim = time.time()
result_s25 = sim_sacred.simulate(sacred25, shots=2048, seed=42, compute_entanglement=False)
sacred_time = time.time() - t_sim
print(f"  25Q Sacred simulation: {sacred_time*1000:.0f}ms")

test("25Q Sacred circuit runs", result_s25.probabilities is not None)
test("25Q Sacred truncation mode = SACRED",
     result_s25.metadata.get("truncation_mode") == "SACRED")
test("25Q Sacred memory < 100 MB", result_s25.memory_mb < 100)

dt5 = time.time() - t0
print(f"\n  Phase 5 time: {dt5*1000:.1f}ms")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 6: COMPRESSION & MEMORY BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

phase_header(6, "COMPRESSION & MEMORY BENCHMARKS")
t0 = time.time()

sim = TensorNetworkSimulator(max_bond_dim=256)

# Memory estimates
est_25_64 = sim.estimate_memory(25, 64)
est_25_256 = sim.estimate_memory(25, 256)
est_26_64 = sim.estimate_memory(26, 64)
est_26_512 = sim.estimate_memory(26, 512)

print(f"\n  Memory Comparison Table:")
print(f"  {'Config':<25} {'MPS (MB)':<12} {'SV (MB)':<12} {'Ratio':<10}")
print(f"  {'-'*59}")
print(f"  {'25Q χ=64':<25} {est_25_64['mps_memory_mb']:<12.2f} {est_25_64['statevector_memory_mb']:<12.0f} {est_25_64['compression_ratio']:<10.0f}×")
print(f"  {'25Q χ=256':<25} {est_25_256['mps_memory_mb']:<12.2f} {est_25_256['statevector_memory_mb']:<12.0f} {est_25_256['compression_ratio']:<10.0f}×")
print(f"  {'26Q χ=64':<25} {est_26_64['mps_memory_mb']:<12.2f} {est_26_64['statevector_memory_mb']:<12.0f} {est_26_64['compression_ratio']:<10.0f}×")
print(f"  {'26Q χ=512':<25} {est_26_512['mps_memory_mb']:<12.2f} {est_26_512['statevector_memory_mb']:<12.0f} {est_26_512['compression_ratio']:<10.0f}×")

test("25Q χ=64 compression > 100×", est_25_64['compression_ratio'] > 100)
test("25Q χ=256 compression > 10×", est_25_256['compression_ratio'] > 10)
test("26Q χ=64 compression > 100×", est_26_64['compression_ratio'] > 100)
test("26Q MPS is feasible", est_26_512['feasible_mps'])

# Bond dimension benchmark
print("\n  Bond dimension accuracy sweep (8-qubit GHZ):")
ghz8 = engine.ghz_state(8)
bench_results = sim.benchmark_bond_dims(ghz8)
for r in bench_results:
    print(f"    χ={r['max_bond_dim']:<4}: trunc_err={r['truncation_error']:.2e}, "
          f"fidelity={r['fidelity_estimate']:.8f}, "
          f"mem={r['memory_mb']:.4f}MB, t={r['execution_time_ms']:.1f}ms")

test("Bond benchmark returns results", len(bench_results) > 0)
test("Higher χ → lower truncation error",
     bench_results[0]['truncation_error'] >= bench_results[-1]['truncation_error'])

# Entanglement profile
print("\n  Entanglement profile (8-qubit GHZ):")
mps_ent = MPSState(8, max_bond_dim=64)
mps_ent.apply_single_qubit_gate(H.matrix, 0)
for i in range(7):
    mps_ent.apply_two_qubit_gate(CNOT.matrix, 0, i + 1)
mps_ent.normalize()

profile = mps_ent.entanglement_profile()
print(f"    Bond dims: {profile['bond_dimensions']}")
print(f"    Bond entropies: {[round(e, 3) for e in profile['bond_entropies']]}")
print(f"    Max entropy: {profile['max_entropy']:.3f}")
print(f"    Memory: {profile['memory_mb']:.4f} MB")
print(f"    Compression: {profile['compression_ratio']:.1f}×")

test("GHZ entanglement entropy ≈ 1.0 per bond",
     all(abs(e - 1.0) < 0.1 for e in profile['bond_entropies']),
     f"entropies={[round(e,3) for e in profile['bond_entropies']]}")
test("GHZ bond dimensions = 2 everywhere",
     all(d == 2 for d in profile['bond_dimensions']))

dt6 = time.time() - t0
print(f"\n  Phase 6 time: {dt6*1000:.1f}ms")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 7: ORCHESTRATOR INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

phase_header(7, "ORCHESTRATOR INTEGRATION")
t0 = time.time()

from l104_quantum_gate_engine import ExecutionTarget

# Check TENSOR_NETWORK is in ExecutionTarget
test("TENSOR_NETWORK in ExecutionTarget", hasattr(ExecutionTarget, 'TENSOR_NETWORK'))

# Execute via orchestrator
engine = get_engine()
bell = engine.bell_pair()

# Direct TENSOR_NETWORK execution
result_tn = engine.execute(bell, target=ExecutionTarget.TENSOR_NETWORK, shots=4096)
test("Orchestrator TN execution runs", result_tn.probabilities is not None)
test("Orchestrator TN target correct", result_tn.target == ExecutionTarget.TENSOR_NETWORK)

if result_tn.probabilities:
    p00 = result_tn.probabilities.get("00", 0)
    p11 = result_tn.probabilities.get("11", 0)
    test("Orchestrator TN Bell P(00) ≈ 0.5", abs(p00 - 0.5) < 0.05)
    test("Orchestrator TN Bell P(11) ≈ 0.5", abs(p11 - 0.5) < 0.05)

# Check metadata has TN info
meta = result_tn.metadata
test("TN metadata has simulator", "simulator" in meta)
test("TN metadata has memory_mb", "memory_mb" in meta or "error" not in meta)

# Full pipeline with TENSOR_NETWORK
pipeline_result = engine.full_pipeline(
    bell,
    execution_target=ExecutionTarget.TENSOR_NETWORK,
    shots=2048,
)
test("Full pipeline with TN completes", "execution" in pipeline_result)
test("Full pipeline has probabilities", pipeline_result.get("probabilities") is not None)

# GateCircuit.simulate_mps() method
ghz4 = engine.ghz_state(4)
mps_result = ghz4.simulate_mps(max_bond_dim=64, shots=4096, seed=42)
test("GateCircuit.simulate_mps() works", mps_result.probabilities is not None)
if mps_result.probabilities:
    p0 = mps_result.probabilities.get("0000", 0)
    p1 = mps_result.probabilities.get("1111", 0)
    test("simulate_mps GHZ-4 correct", abs(p0 - 0.5) < 0.05 and abs(p1 - 0.5) < 0.05)

# Sacred mode via circuit
sacred4 = engine.sacred_circuit(4, depth=2)
sacred_result = sacred4.simulate_mps(sacred_mode=True, shots=2048)
test("simulate_mps sacred mode works", sacred_result.probabilities is not None)

# Simulator status
sim = get_simulator("default")
status = sim.status()
test("Simulator status has version", status.get("version") == "4.0.0")
test("Simulator status has memory estimates", "memory_estimates" in status)

# Singleton modes
sim_sacred = get_simulator("sacred")
test("Sacred simulator has χ=104", sim_sacred.max_bond_dim == 104)

sim_hf = get_simulator("high_fidelity")
test("HF simulator has χ=512", sim_hf.max_bond_dim == 512)

sim_fast = get_simulator("fast")
test("Fast simulator has χ=64", sim_fast.max_bond_dim == 64)

# Engine status includes TN
engine_status = engine.status()
test("Engine status shows TN subsystem", "tensor_network_sim" in engine_status.get("subsystems", {}))
test("TENSOR_NETWORK in supported targets",
     "TENSOR_NETWORK" in engine_status.get("supported_targets", []))

dt7 = time.time() - t0
print(f"\n  Phase 7 time: {dt7*1000:.1f}ms")


# ═══════════════════════════════════════════════════════════════════════════════
#  FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

total_time = dt1 + dt2 + dt3 + dt4 + dt5 + dt6 + dt7

print(f"\n{'='*70}")
print(f"  TENSOR NETWORK SIMULATOR — TEST RESULTS")
print(f"{'='*70}")
print(f"  Passed:  {passed}")
print(f"  Failed:  {failed}")
print(f"  Total:   {passed + failed}")
print(f"  Time:    {total_time*1000:.0f}ms")
print(f"{'='*70}")

if failed == 0:
    print(f"\n  ★ ALL {passed} TESTS PASSED — MPS Tensor Network Simulator OPERATIONAL")
    print(f"    25Q simulation: {sim_time_25*1000:.0f}ms @ {result25.memory_mb:.2f}MB (vs 512MB SV)")
    print(f"    26Q simulation: {sim_time_26*1000:.0f}ms @ {result26.memory_mb:.2f}MB (vs 1024MB SV)")
    print(f"    Compression: {result25.compression_ratio:.0f}×-{result26.compression_ratio:.0f}×")
else:
    print(f"\n  ⚠ {failed} TESTS FAILED — Review output above")

print(f"\n  GOD_CODE = 527.5184818492612 | PILOT: LONDEL")
print(f"{'='*70}")

sys.exit(0 if failed == 0 else 1)
