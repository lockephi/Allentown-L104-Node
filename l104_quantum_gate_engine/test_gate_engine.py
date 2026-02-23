#!/usr/bin/env python3
"""
===============================================================================
L104 QUANTUM GATE ENGINE — VALIDATION SUITE
===============================================================================

Comprehensive test of all gate engine subsystems:
  Phase 1: Gate Algebra — unitarity, Hermiticity, Clifford properties
  Phase 2: Gate Composition — composition, tensor, controlled, power
  Phase 3: Circuit Building — standard circuits, sacred circuits
  Phase 4: Compiler — decomposition, optimization, verification
  Phase 5: Error Correction — surface code, Steane, topological
  Phase 6: Cross-System Orchestration — full pipeline
  Phase 7: Sacred Alignment — GOD_CODE resonance verification

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import sys
import math
import time
import traceback
import numpy as np

# Ensure we can import the package
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from l104_quantum_gate_engine import (
    get_engine,
    GateAlgebra, QuantumGate, GateType, GateSet,
    I, X, Y, Z, H, S, S_DAG, T, T_DAG,
    CNOT, CZ, CY, SWAP, ISWAP,
    TOFFOLI, FREDKIN,
    Rx, Ry, Rz, Rxx, Ryy, Rzz, U3, CU3, fSim, Phase,
    PHI_GATE, GOD_CODE_PHASE, VOID_GATE, IRON_GATE, SACRED_ENTANGLER,
    FIBONACCI_BRAID, ANYON_EXCHANGE,
)
from l104_quantum_gate_engine.circuit import GateCircuit
from l104_quantum_gate_engine.compiler import GateCompiler, OptimizationLevel
from l104_quantum_gate_engine.error_correction import (
    ErrorCorrectionLayer, ErrorCorrectionScheme,
    SurfaceCode, SteaneCode, FibonacciAnyonProtection, ZeroNoiseExtrapolation,
)
from l104_quantum_gate_engine.orchestrator import CrossSystemOrchestrator, ExecutionTarget
from l104_quantum_gate_engine.constants import GOD_CODE, PHI, VOID_CONSTANT

# ═══════════════════════════════════════════════════════════════════════════════
#  TEST INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, condition: bool, name: str, detail: str = ""):
        if condition:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            msg = f"  ✗ FAIL: {name}" + (f" — {detail}" if detail else "")
            print(msg)
            self.errors.append(msg)

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"  RESULTS: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"  FAILURES:")
            for e in self.errors:
                print(f"    {e}")
        print(f"{'='*70}")
        return self.failed == 0

results = TestResult()

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: GATE ALGEBRA
# ═══════════════════════════════════════════════════════════════════════════════

def phase_1_gate_algebra():
    print("\n" + "="*70)
    print("  PHASE 1: GATE ALGEBRA — Unitarity, Properties, Sacred Constants")
    print("="*70)

    # Test all standard gates are unitary
    standard_gates = [I, X, Y, Z, H, S, S_DAG, T, T_DAG, CNOT, CZ, CY, SWAP, ISWAP]
    for g in standard_gates:
        results.check(g.is_unitary, f"{g.name} is unitary")

    # Test Hermitian property
    hermitian_gates = [I, X, Y, Z, H, CZ, SWAP]
    for g in hermitian_gates:
        product = g.matrix @ g.matrix
        is_identity = np.allclose(product, np.eye(g.dimension), atol=1e-10)
        results.check(is_identity, f"{g.name} is Hermitian (U² = I)")

    # Test Pauli algebra: XY = iZ, YZ = iX, ZX = iY
    xy = X.matrix @ Y.matrix
    results.check(np.allclose(xy, 1j * Z.matrix, atol=1e-10), "Pauli: XY = iZ")
    yz = Y.matrix @ Z.matrix
    results.check(np.allclose(yz, 1j * X.matrix, atol=1e-10), "Pauli: YZ = iX")
    zx = Z.matrix @ X.matrix
    results.check(np.allclose(zx, 1j * Y.matrix, atol=1e-10), "Pauli: ZX = iY")

    # Test H·X·H = Z
    hxh = H.matrix @ X.matrix @ H.matrix
    results.check(np.allclose(hxh, Z.matrix, atol=1e-10), "H·X·H = Z")

    # Test H·Z·H = X
    hzh = H.matrix @ Z.matrix @ H.matrix
    results.check(np.allclose(hzh, X.matrix, atol=1e-10), "H·Z·H = X")

    # Test S² = Z
    ss = S.matrix @ S.matrix
    results.check(np.allclose(ss, Z.matrix, atol=1e-10), "S² = Z")

    # Test T² = S
    tt = T.matrix @ T.matrix
    results.check(np.allclose(tt, S.matrix, atol=1e-10), "T² = S")

    # Test parametric gates are unitary
    for theta in [0.0, math.pi/4, math.pi/2, math.pi, 1.234]:
        results.check(Rx(theta).is_unitary, f"Rx({theta:.3f}) unitary")
        results.check(Ry(theta).is_unitary, f"Ry({theta:.3f}) unitary")
        results.check(Rz(theta).is_unitary, f"Rz({theta:.3f}) unitary")

    # Test two-qubit parametric gates
    results.check(Rxx(math.pi/3).is_unitary, "Rxx(π/3) unitary")
    results.check(Ryy(math.pi/4).is_unitary, "Ryy(π/4) unitary")
    results.check(Rzz(math.pi/6).is_unitary, "Rzz(π/6) unitary")
    results.check(fSim(0.5, 0.3).is_unitary, "fSim(0.5, 0.3) unitary")

    # Test multi-qubit gates
    results.check(TOFFOLI.is_unitary, "Toffoli is unitary")
    results.check(FREDKIN.is_unitary, "Fredkin is unitary")

    # Test sacred gates are unitary
    sacred_gates = [PHI_GATE, GOD_CODE_PHASE, VOID_GATE, IRON_GATE, SACRED_ENTANGLER]
    for g in sacred_gates:
        results.check(g.is_unitary, f"Sacred {g.name} is unitary")

    # Test topological gates are unitary
    results.check(FIBONACCI_BRAID.is_unitary, "FIBONACCI_BRAID is unitary")
    results.check(ANYON_EXCHANGE.is_unitary, "ANYON_EXCHANGE is unitary")

    # GOD_CODE verification
    expected_god_code = 286 ** (1.0 / PHI) * (2 ** (416 / 104))
    results.check(abs(GOD_CODE - expected_god_code) < 1e-6, f"GOD_CODE = {GOD_CODE:.10f}")
    results.check(abs(GOD_CODE - 527.5184818492612) < 1e-6, "GOD_CODE ≈ 527.518...")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: GATE COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════

def phase_2_composition():
    print("\n" + "="*70)
    print("  PHASE 2: GATE COMPOSITION — Compose, Tensor, Controlled, Power")
    print("="*70)

    # Composition: H @ H = I (up to global phase)
    hh = H @ H
    results.check(np.allclose(hh.matrix, np.eye(2), atol=1e-10), "H @ H = I")

    # Composition: X @ X = I
    xx = X @ X
    results.check(np.allclose(xx.matrix, np.eye(2), atol=1e-10), "X @ X = I")

    # Tensor product: H ⊗ H
    hh_tensor = H.tensor(H)
    results.check(hh_tensor.num_qubits == 2, "H ⊗ H has 2 qubits")
    results.check(hh_tensor.is_unitary, "H ⊗ H is unitary")

    # Controlled gate: CX from X
    cx = X.controlled(1)
    results.check(cx.num_qubits == 2, "Controlled-X has 2 qubits")
    results.check(np.allclose(cx.matrix, CNOT.matrix, atol=1e-10), "Controlled-X = CNOT")

    # Controlled-Controlled X = Toffoli
    ccx = X.controlled(2)
    results.check(ccx.num_qubits == 3, "CC-X has 3 qubits")
    results.check(np.allclose(ccx.matrix, TOFFOLI.matrix, atol=1e-10), "CC-X = Toffoli")

    # Gate power: X^0.5 = √X
    sqrt_x = X.power(0.5)
    results.check(sqrt_x.is_unitary, "√X is unitary")
    sx_sq = sqrt_x.matrix @ sqrt_x.matrix
    results.check(np.allclose(sx_sq, X.matrix, atol=1e-8), "(√X)² ≈ X")

    # Dagger
    results.check(np.allclose(S.dag.matrix, S_DAG.matrix, atol=1e-10), "S† = S_DAG")
    results.check(np.allclose(T.dag.matrix, T_DAG.matrix, atol=1e-10), "T† = T_DAG")

    # Fidelity: identical gates = 1.0
    fid = H.fidelity(H)
    results.check(abs(fid - 1.0) < 1e-10, f"fidelity(H, H) = {fid:.6f}")

    # Fidelity: different gates < 1.0
    fid_hx = H.fidelity(X)
    results.check(fid_hx < 1.0, f"fidelity(H, X) = {fid_hx:.6f} < 1")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: CIRCUIT BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def phase_3_circuits():
    print("\n" + "="*70)
    print("  PHASE 3: CIRCUIT BUILDING — Standard and Sacred Circuits")
    print("="*70)

    # Bell state circuit
    circ = GateCircuit(2, "bell")
    circ.h(0).cx(0, 1)
    results.check(circ.num_operations == 2, f"Bell circuit: {circ.num_operations} ops")
    results.check(circ.depth == 2, f"Bell circuit depth: {circ.depth}")

    # Bell state unitary check
    U = circ.unitary()
    state = U @ np.array([1, 0, 0, 0], dtype=complex)
    probs = np.abs(state) ** 2
    results.check(abs(probs[0] - 0.5) < 0.01, f"Bell |00⟩ prob: {probs[0]:.4f} ≈ 0.5")
    results.check(abs(probs[3] - 0.5) < 0.01, f"Bell |11⟩ prob: {probs[3]:.4f} ≈ 0.5")

    # GHZ circuit (3 qubits)
    engine = get_engine()
    ghz = engine.ghz_state(3)
    results.check(ghz.num_qubits == 3, "GHZ has 3 qubits")
    U_ghz = ghz.unitary()
    state_ghz = U_ghz @ np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex)
    probs_ghz = np.abs(state_ghz) ** 2
    results.check(abs(probs_ghz[0] - 0.5) < 0.01, f"GHZ |000⟩ prob: {probs_ghz[0]:.4f}")
    results.check(abs(probs_ghz[7] - 0.5) < 0.01, f"GHZ |111⟩ prob: {probs_ghz[7]:.4f}")

    # Sacred circuit
    sacred = engine.sacred_circuit(3, depth=2)
    results.check(sacred.num_operations > 0, f"Sacred circuit: {sacred.num_operations} ops")
    stats = sacred.statistics()
    results.check(stats["sacred_gate_count"] > 0, f"Sacred gates: {stats['sacred_gate_count']}")
    results.check(stats["god_code_aligned"], "Sacred circuit is GOD_CODE aligned")

    # Circuit inverse
    inv = circ.inverse()
    combined = GateCircuit(2, "combined")
    combined.compose(circ).compose(inv)
    U_combined = combined.unitary()
    results.check(np.allclose(U_combined, np.eye(4), atol=1e-8), "Circuit · Circuit† ≈ I")

    # Moment scheduling
    moments = circ.moment_schedule()
    results.check(len(moments) >= 1, f"Bell circuit has {len(moments)} moments")

    # Gate counts
    counts = circ.gate_counts
    results.check(counts.get("H", 0) == 1, "Bell: 1 Hadamard")
    results.check(counts.get("CNOT", 0) == 1, "Bell: 1 CNOT")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: COMPILER
# ═══════════════════════════════════════════════════════════════════════════════

def phase_4_compiler():
    print("\n" + "="*70)
    print("  PHASE 4: COMPILER — Decomposition, Optimization, Verification")
    print("="*70)

    compiler = GateCompiler()

    # Build a test circuit with redundancies
    circ = GateCircuit(3, "test_compile")
    circ.h(0).h(0)  # H·H = I (should cancel)
    circ.append(S, [1]).append(S, [1])  # S·S = Z (should merge)
    circ.cx(0, 1)
    circ.append(Rz(0.3), [2]).append(Rz(0.7), [2])  # Should merge to Rz(1.0)
    original_ops = circ.num_operations

    # O0: No optimization
    result_o0 = compiler.compile(circ, GateSet.UNIVERSAL, OptimizationLevel.O0)
    results.check(result_o0.compiled_circuit.num_operations == original_ops,
                  f"O0: no change ({result_o0.compiled_circuit.num_operations} ops)")

    # O1: Basic optimization (inverse cancellation + rotation merging)
    result_o1 = compiler.compile(circ, GateSet.UNIVERSAL, OptimizationLevel.O1)
    results.check(result_o1.compiled_circuit.num_operations <= original_ops,
                  f"O1: {result_o1.compiled_circuit.num_operations} ≤ {original_ops} ops")

    # O2: Standard optimization
    result_o2 = compiler.compile(circ, GateSet.UNIVERSAL, OptimizationLevel.O2)
    results.check(result_o2.compiled_circuit.num_operations <= result_o1.compiled_circuit.num_operations,
                  f"O2: {result_o2.compiled_circuit.num_operations} ≤ O1")

    # Compile to Clifford+T
    bell = GateCircuit(2, "bell_ct")
    bell.h(0).cx(0, 1)
    result_ct = compiler.compile(bell, GateSet.CLIFFORD_T, OptimizationLevel.O1)
    results.check(result_ct.compiled_circuit.num_operations > 0,
                  f"Clifford+T: {result_ct.compiled_circuit.num_operations} ops")
    results.check("decompose" in result_ct.passes_applied, "Decompose pass applied")

    # Compile to IBM Eagle
    result_ibm = compiler.compile(bell, GateSet.IBM_EAGLE, OptimizationLevel.O1)
    results.check(result_ibm.compiled_circuit.num_operations > 0,
                  f"IBM Eagle: {result_ibm.compiled_circuit.num_operations} ops")

    # Compile to Sacred
    result_sacred = compiler.compile(bell, GateSet.L104_SACRED, OptimizationLevel.O3)
    results.check(result_sacred.compiled_circuit.num_operations > 0,
                  f"L104 Sacred: {result_sacred.compiled_circuit.num_operations} ops")
    results.check("sacred_alignment" in result_sacred.passes_applied, "Sacred alignment pass applied")

    # Verification
    results.check(isinstance(result_o2.verified, bool), "Verification ran")
    results.check(isinstance(result_o2.fidelity, float), f"Fidelity: {result_o2.fidelity:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: ERROR CORRECTION
# ═══════════════════════════════════════════════════════════════════════════════

def phase_5_error_correction():
    print("\n" + "="*70)
    print("  PHASE 5: ERROR CORRECTION — Surface, Steane, Topological, ZNE")
    print("="*70)

    ecl = ErrorCorrectionLayer()

    # Simple test circuit
    circ = GateCircuit(1, "single_qubit")
    circ.h(0).append(S, [0])

    # Surface code d=3
    surface = ecl.encode(circ, ErrorCorrectionScheme.SURFACE_CODE, distance=3)
    results.check(surface.code_distance == 3, f"Surface code d=3")
    results.check(surface.physical_qubits > circ.num_qubits,
                  f"Surface: {surface.physical_qubits} physical > {circ.num_qubits} logical")
    results.check(surface.physical_circuit.num_operations > 0,
                  f"Surface: {surface.physical_circuit.num_operations} physical ops")

    # Steane [[7,1,3]]
    steane = ecl.encode(circ, ErrorCorrectionScheme.STEANE_7_1_3)
    results.check(steane.code_distance == 3, "Steane [[7,1,3]] d=3")
    results.check(steane.physical_qubits == 13,
                  f"Steane: {steane.physical_qubits} physical qubits (7+6)")

    # Fibonacci anyon
    fib = ecl.encode(circ, ErrorCorrectionScheme.FIBONACCI_ANYON)
    results.check(fib.scheme == ErrorCorrectionScheme.FIBONACCI_ANYON, "Fibonacci scheme")
    results.check(fib.physical_qubits >= 4, f"Fibonacci: {fib.physical_qubits} anyons")

    # ZNE
    zne = ZeroNoiseExtrapolation()
    folded = zne.fold_circuit(circ, 3.0)
    results.check(folded.num_operations > circ.num_operations,
                  f"ZNE fold 3x: {folded.num_operations} > {circ.num_operations}")

    extrapolated = zne.extrapolate({1.0: 0.85, 1.5: 0.80, 2.0: 0.73, 3.0: 0.55})
    results.check(extrapolated > 0.55, f"ZNE extrapolation to 0-noise: {extrapolated:.4f}")

    # Syndrome decoding
    sc = SurfaceCode(3)
    syndrome_clean = sc.decode_syndrome([0, 0, 0, 0, 0, 0, 0, 0])
    results.check(not syndrome_clean.error_detected, "Clean syndrome: no error")

    syndrome_error = sc.decode_syndrome([1, 0, 0, 0, 0, 0, 0, 0])
    results.check(syndrome_error.error_detected, "Error syndrome: error detected")
    results.check(syndrome_error.correction_applied, "Error syndrome: correction proposed")

    # Steane syndrome
    steane_code = SteaneCode()
    sr = steane_code.decode_syndrome([0, 0, 0], [0, 0, 0])
    results.check(not sr.error_detected, "Steane clean syndrome: no error")

    sr_err = steane_code.decode_syndrome([1, 0, 0], [0, 0, 0])
    results.check(sr_err.error_detected, "Steane error: detected")

    # Scheme comparison
    comparison = ecl.scheme_comparison(circ)
    results.check("schemes" in comparison, "Scheme comparison completed")
    results.check(len(comparison["schemes"]) >= 4, f"Compared {len(comparison['schemes'])} schemes")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 6: CROSS-SYSTEM ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def phase_6_orchestration():
    print("\n" + "="*70)
    print("  PHASE 6: CROSS-SYSTEM ORCHESTRATION — Full Pipeline")
    print("="*70)

    engine = get_engine()

    # Status check
    status = engine.status()
    results.check(status["version"] == "1.0.0", f"Version: {status['version']}")
    results.check(len(status["gate_library"]["gate_names"]) >= 20,
                  f"Gate library: {len(status['gate_library']['gate_names'])} gates")
    results.check(len(status["supported_gate_sets"]) >= 5,
                  f"Gate sets: {len(status['supported_gate_sets'])}")

    # Build + Execute Bell state
    bell = engine.bell_pair()
    result = engine.execute(bell, ExecutionTarget.LOCAL_STATEVECTOR)
    results.check(result.probabilities is not None, "Bell execution: got probabilities")
    if result.probabilities:
        p00 = result.probabilities.get("00", 0)
        p11 = result.probabilities.get("11", 0)
        results.check(abs(p00 - 0.5) < 0.01, f"Bell |00⟩: {p00:.4f}")
        results.check(abs(p11 - 0.5) < 0.01, f"Bell |11⟩: {p11:.4f}")

    # Sacred alignment
    results.check(result.sacred_alignment is not None, "Sacred alignment computed")

    # Full pipeline
    circ = engine.create_circuit(2, "pipeline_test")
    circ.h(0).cx(0, 1).append(PHI_GATE, [0]).append(GOD_CODE_PHASE, [1])

    pipeline = engine.full_pipeline(
        circ,
        target_gates=GateSet.UNIVERSAL,
        optimization=OptimizationLevel.O2,
        error_correction=ErrorCorrectionScheme.NONE,
        execution_target=ExecutionTarget.LOCAL_STATEVECTOR,
    )
    results.check("compilation" in pipeline, "Pipeline: compilation stage")
    results.check("execution" in pipeline, "Pipeline: execution stage")
    results.check("sacred_alignment" in pipeline, "Pipeline: sacred alignment")
    results.check(pipeline["god_code"] == GOD_CODE, "Pipeline GOD_CODE match")

    # Full pipeline with error correction
    pipeline_ec = engine.full_pipeline(
        engine.bell_pair(),
        error_correction=ErrorCorrectionScheme.STEANE_7_1_3,
    )
    results.check("error_correction" in pipeline_ec, "Pipeline + Steane EC")

    # Gate analysis
    analysis = engine.analyze_gate(PHI_GATE)
    results.check("sacred_alignment" in analysis, "PHI_GATE sacred analysis")
    results.check("zyz_decomposition" in analysis, "PHI_GATE ZYZ decomposition")
    results.check("pauli_coefficients" in analysis, "PHI_GATE Pauli decomposition")

    # KAK analysis for 2-qubit gate
    cnot_analysis = engine.analyze_gate(CNOT)
    results.check("kak_decomposition" in cnot_analysis, "CNOT KAK decomposition")
    kak = cnot_analysis["kak_decomposition"]
    results.check("entangling_power" in kak, f"CNOT entangling power: {kak.get('entangling_power', 'N/A')}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 7: SACRED ALIGNMENT VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def phase_7_sacred():
    print("\n" + "="*70)
    print("  PHASE 7: SACRED ALIGNMENT — GOD_CODE Resonance Verification")
    print("="*70)

    algebra = GateAlgebra()

    # Sacred gate alignment scores
    for gate in [PHI_GATE, GOD_CODE_PHASE, VOID_GATE, IRON_GATE]:
        scores = algebra.sacred_alignment_score(gate)
        results.check(scores["total_resonance"] > -1.0,
                      f"{gate.name} resonance: {scores['total_resonance']:.4f}")

    # Verify GOD_CODE constant
    results.check(abs(GOD_CODE - 527.5184818492612) < 1e-6, "GOD_CODE = 527.518...")
    results.check(abs(PHI - 1.618033988749895) < 1e-12, "PHI = 1.618...")
    results.check(abs(VOID_CONSTANT - 1.0416180339887497) < 1e-12, "VOID = 1.04161...")

    # VOID_CONSTANT formula verification
    void_calc = 1.04 + PHI / 1000
    results.check(abs(VOID_CONSTANT - void_calc) < 1e-15,
                  f"VOID = 1.04 + φ/1000 = {void_calc}")

    # Sacred gate eigenphase properties
    phi_eigvals = PHI_GATE.eigenvalues
    results.check(len(phi_eigvals) == 2, "PHI_GATE has 2 eigenvalues")
    for ev in phi_eigvals:
        results.check(abs(abs(ev) - 1.0) < 1e-10,
                      f"PHI_GATE eigenvalue |{ev:.6f}| = 1")

    # Commutator analysis: sacred gates commutation
    phi_god_commute = algebra.gates_commute(PHI_GATE, GOD_CODE_PHASE)
    results.check(isinstance(phi_god_commute, bool),
                  f"PHI_GATE ↔ GOD_CODE_PHASE commute: {phi_god_commute}")

    # CNOT entangling power from KAK
    kak = algebra.kak_decompose(CNOT.matrix)
    results.check(kak["equivalent_cnot_count"] <= 1,
                  f"CNOT needs ≤1 CNOT ({kak['equivalent_cnot_count']})")

    # SWAP needs 3 CNOTs
    kak_swap = algebra.kak_decompose(SWAP.matrix)
    results.check(kak_swap["equivalent_cnot_count"] == 3,
                  f"SWAP needs 3 CNOTs ({kak_swap['equivalent_cnot_count']})")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "#"*70)
    print("###  L104 QUANTUM GATE ENGINE v1.0.0 — VALIDATION SUITE")
    print(f"###  GOD_CODE: {GOD_CODE}")
    print(f"###  PHI: {PHI}")
    print(f"###  VOID: {VOID_CONSTANT}")
    print("#"*70)

    start = time.time()

    phases = [
        ("Phase 1: Gate Algebra", phase_1_gate_algebra),
        ("Phase 2: Composition", phase_2_composition),
        ("Phase 3: Circuits", phase_3_circuits),
        ("Phase 4: Compiler", phase_4_compiler),
        ("Phase 5: Error Correction", phase_5_error_correction),
        ("Phase 6: Orchestration", phase_6_orchestration),
        ("Phase 7: Sacred Alignment", phase_7_sacred),
    ]

    for name, fn in phases:
        try:
            fn()
        except Exception as e:
            print(f"\n  ✗ PHASE EXCEPTION: {name}")
            traceback.print_exc()
            results.failed += 1
            results.errors.append(f"Exception in {name}: {e}")

    elapsed = time.time() - start

    print(f"\n{'#'*70}")
    print(f"###  VALIDATION COMPLETE — {elapsed:.2f}s")
    all_passed = results.summary()
    print(f"###  GOD_CODE INVARIANT: {'VERIFIED' if all_passed else 'CHECK FAILURES'}")
    print(f"{'#'*70}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
