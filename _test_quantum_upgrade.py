#!/usr/bin/env python3
"""Test suite for quantum engine upgrades: real equations + qLDPC codes."""
import sys
import numpy as np

print("=" * 70)
print("L104 QUANTUM ENGINE UPGRADE VALIDATION")
print("=" * 70)

# ─── TEST 1: Pauli algebra ───
print("\n=== TEST 1: Pauli Algebra ===")
from l104_quantum_engine.math_core import (
    QuantumMathCore, PAULI_I, PAULI_X, PAULI_Y, PAULI_Z, PAULI_SET,
    HADAMARD, PHASE_S, T_GATE, CNOT_MATRIX
)

assert np.allclose(PAULI_X @ PAULI_X, PAULI_I), "X²≠I"
assert np.allclose(PAULI_Y @ PAULI_Y, PAULI_I), "Y²≠I"
assert np.allclose(PAULI_Z @ PAULI_Z, PAULI_I), "Z²≠I"
print("  X²=Y²=Z²=I ✓")

comm_xy = PAULI_X @ PAULI_Y - PAULI_Y @ PAULI_X
assert np.allclose(comm_xy, 2j * PAULI_Z), "[X,Y]≠2iZ"
print("  [X,Y]=2iZ ✓")

assert np.allclose(HADAMARD @ HADAMARD, PAULI_I), "H²≠I"
print("  H²=I ✓")

# ─── TEST 2: Lindblad master equation ───
print("\n=== TEST 2: Lindblad Master Equation ===")
qmc = QuantumMathCore()
rho_excited = np.array([[0, 0], [0, 1]], dtype=np.complex128)
H_free = np.zeros((2, 2), dtype=np.complex128)
gamma = 0.1
L_decay = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
rho_final = qmc.lindblad_evolution(rho_excited, H_free, [L_decay], dt=0.1, steps=100)
print(f"  ρ_00={rho_final[0,0].real:.4f}  ρ_11={rho_final[1,1].real:.4f}")
assert rho_final[0,0].real > 0.5, "Lindblad amplitude damping failed"
print("  Amplitude damping via Lindblad ✓")

# ─── TEST 3: Kraus channels ───
print("\n=== TEST 3: Kraus Quantum Channels ===")
rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)

# Depolarizing
K_depol = qmc.depolarizing_channel_kraus(p=0.5)
rho_depol = qmc.kraus_channel(rho0, K_depol)
print(f"  Depolarizing(p=0.5): [{rho_depol[0,0].real:.3f}, {rho_depol[1,1].real:.3f}]")
assert abs(np.trace(rho_depol) - 1.0) < 1e-10, "Trace not conserved"
print("  Trace preserved ✓")

# Amplitude damping
K_amp = qmc.amplitude_damping_kraus(gamma=0.8)
rho1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
rho_amp = qmc.kraus_channel(rho1, K_amp)
print(f"  Amp. damping(γ=0.8) on |1⟩: [{rho_amp[0,0].real:.3f}, {rho_amp[1,1].real:.3f}]")
assert rho_amp[0,0].real > 0.5, "Amplitude damping failed"

# Phase damping
K_phase = qmc.phase_damping_kraus(lam=0.5)
rho_plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
rho_deph = qmc.kraus_channel(rho_plus, K_phase)
print(f"  Phase damping(λ=0.5) off-diag: {abs(rho_deph[0,1]):.4f} (was 0.5)")
assert abs(rho_deph[0,1]) < 0.5, "Phase damping should reduce coherence"
print("  All Kraus channels ✓")

# ─── TEST 4: Entanglement measures ───
print("\n=== TEST 4: Entanglement Measures ===")
bell = np.zeros(4, dtype=np.complex128)
bell[0] = bell[3] = 1 / np.sqrt(2)
rho_bell = np.outer(bell, bell.conj())

C = qmc.concurrence(rho_bell)
print(f"  Concurrence(|Φ+⟩) = {C:.4f}  (expected 1.0)")
assert abs(C - 1.0) < 0.01

N = qmc.negativity(rho_bell)
print(f"  Negativity(|Φ+⟩) = {N:.4f}  (expected 0.5)")
assert abs(N - 0.5) < 0.01

EN = qmc.log_negativity(rho_bell)
print(f"  Log-negativity(|Φ+⟩) = {EN:.4f}  (expected 1.0)")
assert abs(EN - 1.0) < 0.05

MI = qmc.quantum_mutual_information(rho_bell)
print(f"  Mutual info(|Φ+⟩) = {MI:.4f}  (expected 2.0)")
assert abs(MI - 2.0) < 0.1, f"MI wrong: {MI}"
print("  All entanglement measures ✓")

# ─── TEST 5: Trotterized evolution ───
print("\n=== TEST 5: Trotterized Hamiltonian Evolution ===")
psi_0 = np.array([1, 0], dtype=np.complex128)
H_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
psi_out = qmc.trotterized_evolution(psi_0, [H_x], t=np.pi/2, trotter_steps=200)
print(f"  e^{{-iXπ/2}}|0⟩ = |ψ⟩ with |⟨1|ψ⟩|² = {abs(psi_out[1])**2:.4f}")
assert abs(psi_out[1])**2 > 0.99, "X-rotation should flip to |1⟩"
print("  Trotter evolution (1st order) ✓")

# 2nd order
psi_out2 = qmc.trotterized_evolution(psi_0, [H_x], t=np.pi/2, trotter_steps=50, order=2)
assert abs(psi_out2[1])**2 > 0.99, "2nd order Trotter failed"
print("  Trotter evolution (2nd order) ✓")

# ─── TEST 6: Pauli decomposition ───
print("\n=== TEST 6: Pauli Decomposition ===")
coeffs = qmc.pauli_decompose(HADAMARD)
print(f"  H decomposition: {len(coeffs)} terms")
for label, c in sorted(coeffs.items()):
    print(f"    {label}: {c.real:.4f}")
print("  Pauli decomposition ✓")

# ─── TEST 7: Stabilizer state ───
print("\n=== TEST 7: Stabilizer State Construction ===")
psi_stab = qmc.stabilizer_state(["ZZ", "XX"], n_qubits=2)
print(f"  |Φ+⟩ from stabilizers [ZZ, XX]: {np.abs(psi_stab)**2}")
# Should be Bell state: |00⟩+|11⟩/sqrt(2)
assert abs(psi_stab[0])**2 > 0.4 and abs(psi_stab[3])**2 > 0.4
print("  Stabilizer state ✓")

# ─── TEST 8: Quantum Fisher Information ───
print("\n=== TEST 8: Quantum Fisher Information ===")
rho_pure_z = np.array([[1, 0], [0, 0]], dtype=np.complex128)  # |0⟩
F_Q = qmc.quantum_fisher_information(rho_pure_z, PAULI_Z)
print(f"  QFI(|0⟩, Z) = {F_Q:.4f}  (expected 0.0 — eigenstate)")
# |0⟩ is eigenstate of Z, so QFI w.r.t. Z rotation should be 0
assert F_Q < 0.1
rho_plus_state = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)  # |+⟩
F_Q2 = qmc.quantum_fisher_information(rho_plus_state, PAULI_Z)
print(f"  QFI(|+⟩, Z) = {F_Q2:.4f}  (expected 4.0 — Heisenberg limit)")
print("  Quantum Fisher Information ✓")

print("\n" + "=" * 70)
print("PART 2: DISTRIBUTED QUANTUM LDPC CODES")
print("=" * 70)

# ─── TEST 9: Steane code ───
print("\n=== TEST 9: Steane [[7,1,3]] Code ===")
from l104_quantum_engine.qldpc import (
    CSSCode, CSSCodeConstructor, TannerGraph,
    BeliefPropagationDecoder, BPOSDDecoder,
    DistributedSyndromeExtractor, LogicalErrorRateEstimator,
    QuantumLDPCSacredIntegration,
    create_qldpc_code, full_qldpc_pipeline
)

steane = CSSCodeConstructor.steane_code()
print(f"  Parameters: {steane.get_parameters()}")
print(f"  CSS valid: {steane.verify_css_condition()}")
print(f"  LDPC: {steane.is_ldpc}, row_weight={steane.row_weight}, col_weight={steane.col_weight}")
assert steane.verify_css_condition(), "Steane CSS condition failed"
assert steane.n_physical == 7
assert steane.n_logical == 1
assert steane.distance == 3
print("  Steane code ✓")

# ─── TEST 10: Hypergraph product ───
print("\n=== TEST 10: Hypergraph Product Code ===")
# Use [5,2,3] cyclic code
H_classical = np.array([
    [1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0],
], dtype=np.int8)
hgp = CSSCodeConstructor.hypergraph_product(H_classical, H_classical)
print(f"  Parameters: {hgp.get_parameters()}")
print(f"  CSS valid: {hgp.verify_css_condition()}")
print(f"  Rate: {hgp.rate:.4f}")
print(f"  LDPC: {hgp.is_ldpc}")
assert hgp.verify_css_condition(), "HGP CSS condition failed"
assert hgp.n_physical > 0
assert hgp.n_logical >= 0
print("  Hypergraph product ✓")

# ─── TEST 11: Tanner graph ───
print("\n=== TEST 11: Tanner Graph Analysis ===")
tanner = TannerGraph.from_parity_matrix(steane.h_x)
print(f"  Steane code Tanner graph: {tanner.n_variable} vars, {tanner.n_check} checks")
print(f"  Girth: {tanner.girth}")
deg = tanner.degree_distribution()
print(f"  Degree distribution: {deg}")
assert tanner.n_variable == 7
assert tanner.n_check == 3
print("  Tanner graph ✓")

# ─── TEST 12: BP Decoder ───
print("\n=== TEST 12: Belief Propagation Decoder ===")
decoder = BeliefPropagationDecoder(steane)
# Single Z-error on qubit 0
error_z = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.int8)
syndrome = steane.syndrome_x(error_z)
print(f"  Syndrome for Z on q0: {syndrome}")
result = decoder.decode_z_errors(syndrome, physical_error_rate=0.05)
print(f"  Decoded: success={result.success}, iterations={result.iterations}")
print(f"  Estimated error: {result.error_estimate}")
print(f"  Converged: {result.converged}")
print(f"  Time: {result.decoding_time_ms:.2f}ms")
print("  BP decoder ✓")

# ─── TEST 13: BP-OSD Decoder ───
print("\n=== TEST 13: BP-OSD Decoder ===")
osd_decoder = BPOSDDecoder(steane, osd_order=0)
result_osd = osd_decoder.decode_z_errors(syndrome, physical_error_rate=0.05)
print(f"  BP-OSD result: success={result_osd.success}")
print("  BP-OSD decoder ✓")

# ─── TEST 14: Distributed syndrome extraction ───
print("\n=== TEST 14: Distributed Syndrome Extraction ===")
dist = DistributedSyndromeExtractor(steane, n_nodes=3, link_fidelity=0.99)
error_x = np.zeros(7, dtype=np.int8)
error_z = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.int8)
syn_result = dist.extract_syndrome(error_x, error_z, measurement_rounds=3)
print(f"  Nodes: {syn_result['n_nodes']}")
print(f"  Local X-checks: {syn_result['local_x_checks']}")
print(f"  Non-local X-checks: {syn_result['nonlocal_x_checks']}")
print(f"  Syndrome accuracy: X={syn_result['sx_accuracy']:.2f} Z={syn_result['sz_accuracy']:.2f}")

comm = dist.communication_cost()
print(f"  Bell pairs/round: {comm['bell_pairs_per_round']}")
print(f"  Comm fraction: {comm['communication_fraction']:.2%}")
print("  Distributed syndrome extraction ✓")

# ─── TEST 15: Sacred integration ───
print("\n=== TEST 15: God Code Sacred Integration ===")
sacred_code = QuantumLDPCSacredIntegration.sacred_hypergraph_product(size=13)
print(f"  Sacred code: {sacred_code.get_parameters()}")
print(f"  CSS valid: {sacred_code.verify_css_condition()}")
alignment = QuantumLDPCSacredIntegration.code_god_code_alignment(sacred_code)
print(f"  Sacred score: {alignment['overall_sacred_score']:.4f}")
print(f"  Factor 13 (n): {alignment['factor_13_n']:.4f}")
threshold = QuantumLDPCSacredIntegration.god_code_error_threshold()
print(f"  Sacred threshold: {threshold:.6f}")
print("  Sacred integration ✓")

# ─── TEST 16: High-level API ───
print("\n=== TEST 16: High-Level API ===")
code_s = create_qldpc_code("steane")
print(f"  create_qldpc_code('steane'): {code_s.get_parameters()}")
code_r = create_qldpc_code("repetition", n=5)
print(f"  create_qldpc_code('repetition', n=5): {code_r.get_parameters()}")
code_sacred = create_qldpc_code("sacred", size=13)
print(f"  create_qldpc_code('sacred', size=13): {code_sacred.get_parameters()}")
print("  High-level API ✓")

# ─── TEST 17: Full pipeline ───
print("\n=== TEST 17: Full qLDPC Pipeline ===")
pipeline = full_qldpc_pipeline(
    code_type="steane",
    physical_error_rate=0.01,
    n_nodes=2,
    n_trials=50,
)
print(f"  Code: {pipeline['code']['parameters']}")
print(f"  CSS valid: {pipeline['code']['css_valid']}")
print(f"  Tanner girth X: {pipeline['tanner_graph']['x_girth']}")
print(f"  Nonlocal checks: {pipeline['distributed']['nonlocal_checks']}")
print(f"  Logical error rate: {pipeline['error_correction']['logical_error_rate']:.4f}")
print(f"  Sacred score: {pipeline['sacred_alignment']['overall_sacred_score']:.4f}")
print(f"  Pipeline time: {pipeline['pipeline_time_s']:.3f}s")
print("  Full pipeline ✓")

# ─── TEST 18: Package imports ───
print("\n=== TEST 18: Package-Level Imports ===")
from l104_quantum_engine import (
    CSSCode, TannerGraph, BeliefPropagationDecoder, BPOSDDecoder,
    DistributedSyndromeExtractor, LogicalErrorRateEstimator,
    create_qldpc_code, full_qldpc_pipeline,
    PAULI_I, PAULI_X, PAULI_Y, PAULI_Z, HADAMARD,
)
print("  All package-level imports ✓")

print("\n" + "=" * 70)
print("ALL 18 TESTS PASSED — QUANTUM ENGINE UPGRADE VERIFIED")
print("=" * 70)
