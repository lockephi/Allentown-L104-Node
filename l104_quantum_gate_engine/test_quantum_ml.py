"""
===============================================================================
L104 QUANTUM GATE ENGINE — QUANTUM ML SUITE TEST SUITE
===============================================================================

12-phase, 86-test validation of the Quantum ML Suite (v8.0).

Phases:
   1. Import & Singleton                         (7 tests)
   2. ParameterisedCircuit — Construction         (7 tests)
   3. ParameterisedCircuit — Evaluation           (8 tests)
   4. Ansatz Library — Hardware Efficient         (7 tests)
   5. Ansatz Library — Advanced Ansätze           (7 tests)
   6. Ansatz Library — Sacred Ansatz              (7 tests)
   7. QNN Trainer — Gradient Optimisation         (8 tests)
   8. Quantum Kernel — Matrix Computation         (7 tests)
   9. Quantum Kernel — Alignment & Types          (7 tests)
  10. Variational Eigensolver — VQE               (8 tests)
  11. GateCircuit Convenience Methods             (6 tests)
  12. Orchestrator — QUANTUM_ML Target            (7 tests)

Run:
    .venv/bin/python -m l104_quantum_gate_engine.test_quantum_ml

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import sys
import math
import time
import traceback
import numpy as np

# ─── Counters ────────────────────────────────────────────────────────────────

_pass = 0
_fail = 0
_errors: list = []


def ok(label: str, condition: bool, detail: str = ""):
    global _pass, _fail
    if condition:
        _pass += 1
        print(f"  ✅ {label}")
    else:
        _fail += 1
        msg = f"  ❌ {label}" + (f"  — {detail}" if detail else "")
        print(msg)
        _errors.append(msg)


def phase(num: int, title: str):
    print(f"\n{'═'*70}")
    print(f"  PHASE {num}: {title}")
    print(f"{'═'*70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: IMPORT & SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_1():
    phase(1, "IMPORT & SINGLETON")

    # 1.1 — Core classes import
    try:
        from l104_quantum_gate_engine.quantum_ml import (
            QuantumMLEngine, ParameterisedCircuit, AnsatzLibrary,
            QNNTrainer, QuantumKernel, VariationalEigensolver,
        )
        ok("1.1  Core classes import", True)
    except Exception as e:
        ok("1.1  Core classes import", False, str(e))
        return

    # 1.2 — Data classes import
    try:
        from l104_quantum_gate_engine.quantum_ml import (
            TrainingResult, KernelResult, VQEResult,
            AnsatzType, OptimizerType, KernelType,
        )
        ok("1.2  Data classes & enums import", True)
    except Exception as e:
        ok("1.2  Data classes & enums import", False, str(e))

    # 1.3 — Constants import
    try:
        from l104_quantum_gate_engine.quantum_ml import (
            MAX_QML_QUBITS, SACRED_LEARNING_RATE, PARAMETER_SHIFT,
        )
        ok("1.3  Constants import", MAX_QML_QUBITS == 14)
    except Exception as e:
        ok("1.3  Constants import", False, str(e))

    # 1.4 — Singleton
    try:
        from l104_quantum_gate_engine.quantum_ml import get_quantum_ml
        s1 = get_quantum_ml()
        s2 = get_quantum_ml()
        ok("1.4  Singleton identity", s1 is s2)
    except Exception as e:
        ok("1.4  Singleton identity", False, str(e))

    # 1.5 — __init__.py re-export
    try:
        from l104_quantum_gate_engine import (
            QuantumMLEngine, ParameterisedCircuit, AnsatzLibrary,
            get_quantum_ml, MAX_QML_QUBITS, SACRED_LEARNING_RATE,
        )
        ok("1.5  Package re-export (__init__)", True)
    except Exception as e:
        ok("1.5  Package re-export (__init__)", False, str(e))

    # 1.6 — Enum values
    try:
        ok("1.6  AnsatzType members ≥ 6",
           len(AnsatzType) >= 6,
           f"got {len(AnsatzType)}")
    except Exception as e:
        ok("1.6  AnsatzType members ≥ 6", False, str(e))

    # 1.7 — Sacred learning rate value
    try:
        from l104_quantum_gate_engine.quantum_ml import SACRED_LEARNING_RATE
        from l104_quantum_gate_engine.constants import PHI, QUANTIZATION_GRAIN
        expected = 1.0 / (PHI * QUANTIZATION_GRAIN)
        ok("1.7  SACRED_LEARNING_RATE = 1/(φ×104)",
           abs(SACRED_LEARNING_RATE - expected) < 1e-12,
           f"got {SACRED_LEARNING_RATE:.8f}, expected {expected:.8f}")
    except Exception as e:
        ok("1.7  SACRED_LEARNING_RATE = 1/(φ×104)", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: PARAMETERISED CIRCUIT — CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_2():
    phase(2, "PARAMETERISED CIRCUIT — CONSTRUCTION")

    from l104_quantum_gate_engine.quantum_ml import ParameterisedCircuit

    # 2.1 — Basic construction
    pqc = ParameterisedCircuit(2, "test_pqc")
    ok("2.1  Construction (2 qubits)", pqc.num_qubits == 2 and pqc.num_parameters == 0)

    # 2.2 — Auto-parameter allocation with Ry
    pqc.ry(0).ry(1)
    ok("2.2  Ry auto-alloc (2 params)", pqc.num_parameters == 2)

    # 2.3 — Rz adds more parameters
    pqc.rz(0)
    ok("2.3  Rz auto-alloc (3 params)", pqc.num_parameters == 3)

    # 2.4 — Rx adds parameter
    pqc.rx(1)
    ok("2.4  Rx auto-alloc (4 params)", pqc.num_parameters == 4)

    # 2.5 — CNOT does not add parameters
    before = pqc.num_parameters
    pqc.cnot(0, 1)
    ok("2.5  CNOT adds 0 params", pqc.num_parameters == before)

    # 2.6 — Fixed gates do not add parameters
    before = pqc.num_parameters
    pqc.fixed_rz(0, 1.23).fixed_ry(1, 0.5).fixed_rx(0, 0.7)
    ok("2.6  Fixed gates add 0 params", pqc.num_parameters == before)

    # 2.7 — H does not add parameters
    before = pqc.num_parameters
    pqc.h(0).h(1)
    ok("2.7  H adds 0 params", pqc.num_parameters == before)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: PARAMETERISED CIRCUIT — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_3():
    phase(3, "PARAMETERISED CIRCUIT — EVALUATION")

    from l104_quantum_gate_engine.quantum_ml import ParameterisedCircuit

    # 3.1 — Statevector is normalised
    pqc = ParameterisedCircuit(2, "test_sv")
    pqc.ry(0).ry(1).cnot(0, 1)
    params = np.array([0.5, 1.2])
    sv = pqc.statevector(params)
    ok("3.1  Statevector normalised",
       abs(np.linalg.norm(sv) - 1.0) < 1e-10,
       f"norm={np.linalg.norm(sv)}")

    # 3.2 — Statevector dimension = 2^n
    ok("3.2  Statevector dim = 2^n", len(sv) == 4)

    # 3.3 — Unitary is unitary
    U = pqc.unitary(params)
    diff = np.linalg.norm(U @ U.conj().T - np.eye(4))
    ok("3.3  Unitary U†U = I",
       diff < 1e-10, f"‖U†U - I‖ = {diff:.2e}")

    # 3.4 — Zero params → |00⟩ for identity rotations
    pqc2 = ParameterisedCircuit(2, "zero_test")
    pqc2.ry(0).ry(1)
    zero_params = np.array([0.0, 0.0])
    sv_zero = pqc2.statevector(zero_params)
    ok("3.4  Zero params → |00⟩ state",
       abs(sv_zero[0] - 1.0) < 1e-10,
       f"|00⟩ amp = {sv_zero[0]}")

    # 3.5 — Ry(π) on qubit 0 → |10⟩
    pqc3 = ParameterisedCircuit(2, "pi_test")
    pqc3.ry(0)
    pi_params = np.array([math.pi])
    sv_pi = pqc3.statevector(pi_params)
    # |10⟩ = index 2 for 2-qubit system
    ok("3.5  Ry(π)|00⟩ → |10⟩",
       abs(abs(sv_pi[2]) - 1.0) < 1e-10,
       f"|10⟩ amp = {abs(sv_pi[2]):.6f}")

    # 3.6 — Expectation of Z on |0⟩ = +1
    from l104_quantum_gate_engine.quantum_ml import _pauli_observable
    pqc4 = ParameterisedCircuit(1, "exp_test")
    pqc4.ry(0)
    Z = _pauli_observable(1, 0, "Z")
    exp_val = pqc4.expectation(np.array([0.0]), Z)
    ok("3.6  ⟨0|Z|0⟩ = +1",
       abs(exp_val - 1.0) < 1e-10, f"got {exp_val:.6f}")

    # 3.7 — Expectation of Z on |1⟩ = -1  (Ry(π)|0⟩ = |1⟩)
    exp_val_pi = pqc4.expectation(np.array([math.pi]), Z)
    ok("3.7  ⟨1|Z|1⟩ = -1",
       abs(exp_val_pi + 1.0) < 1e-10, f"got {exp_val_pi:.6f}")

    # 3.8 — Gradient shape matches parameters
    pqc5 = ParameterisedCircuit(2, "grad_test")
    pqc5.ry(0).rz(1).cnot(0, 1).ry(1)
    p5 = np.array([0.3, 0.7, 1.1])
    Z0 = _pauli_observable(2, 0, "Z")
    grad = pqc5.gradient(p5, Z0)
    ok("3.8  Gradient shape = (n_params,)",
       grad.shape == (3,), f"shape={grad.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: ANSATZ LIBRARY — HARDWARE EFFICIENT
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_4():
    phase(4, "ANSATZ LIBRARY — HARDWARE EFFICIENT")

    from l104_quantum_gate_engine.quantum_ml import AnsatzLibrary, ParameterisedCircuit

    # 4.1 — HEA default (ry_rz)
    hea = AnsatzLibrary.hardware_efficient(3, depth=2)
    ok("4.1  HEA builds (3q, d=2)", isinstance(hea, ParameterisedCircuit))

    # 4.2 — Parameter count: depth × n × 2 (ry+rz) + n (final ry)
    expected_params = 2 * 3 * 2 + 3  # 15
    ok("4.2  HEA param count = 15",
       hea.num_parameters == expected_params,
       f"got {hea.num_parameters}")

    # 4.3 — HEA with ry-only rotation
    hea_ry = AnsatzLibrary.hardware_efficient(3, depth=2, rotation="ry")
    expected_ry = 2 * 3 * 1 + 3  # 9
    ok("4.3  HEA ry-only param count = 9",
       hea_ry.num_parameters == expected_ry,
       f"got {hea_ry.num_parameters}")

    # 4.4 — HEA statevector is normalised
    sv = hea.statevector(np.zeros(hea.num_parameters))
    ok("4.4  HEA zero-params statevector normalised",
       abs(np.linalg.norm(sv) - 1.0) < 1e-10)

    # 4.5 — Different depths give different param counts
    hea_d1 = AnsatzLibrary.hardware_efficient(4, depth=1)
    hea_d3 = AnsatzLibrary.hardware_efficient(4, depth=3)
    ok("4.5  Depth 1 < depth 3 params",
       hea_d1.num_parameters < hea_d3.num_parameters)

    # 4.6 — HEA name contains depth
    ok("4.6  HEA name contains 'd2'", "d2" in hea.name)

    # 4.7 — Unitary from HEA is valid
    U = hea.unitary(np.random.RandomState(42).randn(hea.num_parameters))
    diff = np.linalg.norm(U @ U.conj().T - np.eye(2**3))
    ok("4.7  HEA random-param unitary valid",
       diff < 1e-9, f"‖U†U-I‖={diff:.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: ANSATZ LIBRARY — ADVANCED ANSÄTZE
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_5():
    phase(5, "ANSATZ LIBRARY — ADVANCED ANSÄTZE")

    from l104_quantum_gate_engine.quantum_ml import AnsatzLibrary

    # 5.1 — Strongly entangling layers
    sel = AnsatzLibrary.strongly_entangling(3, depth=2)
    ok("5.1  Strongly entangling builds (3q, d=2)",
       sel.num_parameters > 0)

    # 5.2 — SEL param count: depth × n × 3 + n × 3
    expected = 2 * 3 * 3 + 3 * 3  # 27
    ok("5.2  SEL param count = 27",
       sel.num_parameters == expected,
       f"got {sel.num_parameters}")

    # 5.3 — UCCSD builds
    uccsd = AnsatzLibrary.uccsd(4, num_electrons=2)
    ok("5.3  UCCSD builds (4q, 2e)",
       uccsd.num_parameters > 0,
       f"params={uccsd.num_parameters}")

    # 5.4 — QAOA builds
    qaoa = AnsatzLibrary.qaoa_layer(4, depth=2)
    ok("5.4  QAOA builds (4q, d=2)", qaoa.num_parameters > 0)

    # 5.5 — QAOA param count: depth × (n-1 + n) = depth × (2n-1)
    expected_qaoa = 2 * (4 - 1 + 4)  # 14
    ok("5.5  QAOA params = 14",
       qaoa.num_parameters == expected_qaoa,
       f"got {qaoa.num_parameters}")

    # 5.6 — Data re-uploading
    dr = AnsatzLibrary.data_reuploading(3, depth=2)
    ok("5.6  Data reuploading builds", dr.num_parameters > 0)

    # 5.7 — Data reuploading param count: depth × n × 3  (rx + ry + rz)
    expected_dr = 2 * 3 * 3  # 18
    ok("5.7  Data reuploading params = 18",
       dr.num_parameters == expected_dr,
       f"got {dr.num_parameters}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 6: ANSATZ LIBRARY — SACRED ANSATZ
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_6():
    phase(6, "ANSATZ LIBRARY — SACRED ANSATZ")

    from l104_quantum_gate_engine.quantum_ml import AnsatzLibrary, ParameterisedCircuit
    from l104_quantum_gate_engine.constants import GOD_CODE, PHI, GOD_CODE_PHASE_ANGLE

    # 6.1 — Sacred ansatz builds
    sa = AnsatzLibrary.sacred_ansatz(3, depth=2)
    ok("6.1  Sacred ansatz builds (3q, d=2)",
       isinstance(sa, ParameterisedCircuit))

    # 6.2 — Has GOD_CODE phase fixed angles
    has_gc_phase = any(
        abs(angle - GOD_CODE_PHASE_ANGLE) < 1e-10
        for angle in sa._fixed_angles.values()
    )
    ok("6.2  Contains GOD_CODE phase angles", has_gc_phase)

    # 6.3 — Has PHI-based fixed angles (π/φ)
    phi_angle = math.pi / PHI
    has_phi = any(
        abs(angle - phi_angle) < 1e-10
        for angle in sa._fixed_angles.values()
    )
    ok("6.3  Contains π/φ fixed angles", has_phi)

    # 6.4 — Name contains 'Sacred'
    ok("6.4  Name contains 'Sacred'", "Sacred" in sa.name)

    # 6.5 — Param count: depth × n × 2 (ry + rz) + n × 2 (final ry+rz)
    expected = 2 * 3 * 2 + 3 * 2  # 18
    ok("6.5  Sacred param count = 18",
       sa.num_parameters == expected,
       f"got {sa.num_parameters}")

    # 6.6 — Statevector normalised with random sacred params
    rng = np.random.RandomState(104)
    params = rng.uniform(-0.06, 0.06, sa.num_parameters)
    sv = sa.statevector(params)
    ok("6.6  Sacred statevector normalised",
       abs(np.linalg.norm(sv) - 1.0) < 1e-10)

    # 6.7 — Sacred ansatz unitary is valid
    U = sa.unitary(params)
    diff = np.linalg.norm(U @ U.conj().T - np.eye(2**3))
    ok("6.7  Sacred unitary valid",
       diff < 1e-9, f"‖U†U-I‖={diff:.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 7: QNN TRAINER — GRADIENT OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_7():
    phase(7, "QNN TRAINER — GRADIENT OPTIMISATION")

    from l104_quantum_gate_engine.quantum_ml import (
        QNNTrainer, AnsatzLibrary, OptimizerType, _pauli_observable,
    )

    # Build a simple 2-qubit problem: minimise ⟨Z₀⟩
    Z0 = _pauli_observable(2, 0, "Z")
    ansatz = AnsatzLibrary.hardware_efficient(2, depth=2)

    # 7.1 — ADAM training runs
    trainer = QNNTrainer(learning_rate=0.1, optimizer=OptimizerType.ADAM)
    result = trainer.train(ansatz, Z0, max_iterations=30, seed=42, minimize=True)
    ok("7.1  ADAM training completes",
       result.num_iterations > 0)

    # 7.2 — Cost decreases (minimising ⟨Z₀⟩ should reach -1)
    ok("7.2  ADAM cost decreases",
       result.cost_history[-1] < result.cost_history[0],
       f"first={result.cost_history[0]:.4f}, last={result.cost_history[-1]:.4f}")

    # 7.3 — Optimal cost < 0  (⟨Z₀⟩ minimum is -1)
    ok("7.3  ADAM optimal cost < 0",
       result.optimal_cost < 0,
       f"optimal={result.optimal_cost:.4f}")

    # 7.4 — TrainingResult.to_dict() works
    d = result.to_dict()
    ok("7.4  to_dict() has required keys",
       all(k in d for k in ["optimal_cost", "num_iterations", "optimizer", "god_code"]))

    # 7.5 — GOD_CODE in result (approximate check)
    ok("7.5  GOD_CODE in result ≈ 527.518",
       abs(d["god_code"] - 527.518) < 0.1,
       f"got {d['god_code']}")

    # 7.6 — SGD training
    trainer_sgd = QNNTrainer(learning_rate=0.05, optimizer=OptimizerType.SGD)
    result_sgd = trainer_sgd.train(ansatz, Z0, max_iterations=20, seed=42)
    ok("7.6  SGD training completes", result_sgd.num_iterations > 0)

    # 7.7 — SPSA training
    trainer_spsa = QNNTrainer(learning_rate=0.2, optimizer=OptimizerType.SPSA)
    result_spsa = trainer_spsa.train(ansatz, Z0, max_iterations=20, seed=42)
    ok("7.7  SPSA training completes", result_spsa.num_iterations > 0)

    # 7.8 — SACRED_DESCENT training
    trainer_sacred = QNNTrainer(learning_rate=0.1, optimizer=OptimizerType.SACRED_DESCENT)
    result_sacred = trainer_sacred.train(ansatz, Z0, max_iterations=20, seed=42)
    ok("7.8  SACRED_DESCENT training completes", result_sacred.num_iterations > 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 8: QUANTUM KERNEL — MATRIX COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_8():
    phase(8, "QUANTUM KERNEL — MATRIX COMPUTATION")

    from l104_quantum_gate_engine.quantum_ml import QuantumKernel, KernelType

    qk = QuantumKernel(2, kernel_type=KernelType.ZZ_FEATURE_MAP)

    # Generate toy data
    rng = np.random.RandomState(42)
    X = rng.randn(5, 2)

    # 8.1 — Single kernel entry
    k_val = qk.kernel_entry(X[0], X[1])
    ok("8.1  Kernel entry computable",
       0.0 <= k_val <= 1.0,
       f"K(0,1)={k_val:.6f}")

    # 8.2 — Self-kernel = 1
    k_self = qk.kernel_entry(X[0], X[0])
    ok("8.2  Self-kernel K(x,x) = 1",
       abs(k_self - 1.0) < 1e-10,
       f"K(x,x)={k_self:.8f}")

    # 8.3 — Full kernel matrix
    kr = qk.compute_kernel(X)
    K = kr.kernel_matrix
    ok("8.3  Kernel matrix shape = (5, 5)",
       K.shape == (5, 5))

    # 8.4 — Diagonal = 1
    diag_ok = all(abs(K[i, i] - 1.0) < 1e-10 for i in range(5))
    ok("8.4  Kernel diagonal = 1", diag_ok)

    # 8.5 — Symmetric
    sym_err = np.linalg.norm(K - K.T)
    ok("8.5  Kernel symmetric",
       sym_err < 1e-10, f"‖K-K†‖={sym_err:.2e}")

    # 8.6 — Positive semi-definite
    ok("8.6  Kernel PSD", kr.is_psd)

    # 8.7 — KernelResult.to_dict()
    d = kr.to_dict()
    ok("8.7  to_dict() has keys",
       all(k in d for k in ["num_samples", "kernel_type", "is_psd", "god_code"]))


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 9: QUANTUM KERNEL — ALIGNMENT & TYPES
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_9():
    phase(9, "QUANTUM KERNEL — ALIGNMENT & TYPES")

    from l104_quantum_gate_engine.quantum_ml import QuantumKernel, KernelType

    rng = np.random.RandomState(42)
    X = rng.randn(8, 2)
    y = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    # 9.1 — Target alignment computes
    qk_zz = QuantumKernel(2, kernel_type=KernelType.ZZ_FEATURE_MAP)
    alignment = qk_zz.target_alignment(X, y)
    ok("9.1  Target alignment computes",
       -1.0 <= alignment <= 1.0,
       f"alignment={alignment:.4f}")

    # 9.2 — IQP kernel builds
    qk_iqp = QuantumKernel(2, kernel_type=KernelType.IQP)
    kr_iqp = qk_iqp.compute_kernel(X[:4])
    ok("9.2  IQP kernel matrix computes",
       kr_iqp.kernel_matrix.shape == (4, 4))

    # 9.3 — IQP kernel PSD
    ok("9.3  IQP kernel PSD", kr_iqp.is_psd)

    # 9.4 — Sacred kernel builds
    qk_s = QuantumKernel(2, kernel_type=KernelType.SACRED_KERNEL)
    kr_s = qk_s.compute_kernel(X[:4])
    ok("9.4  Sacred kernel matrix computes",
       kr_s.kernel_matrix.shape == (4, 4))

    # 9.5 — Sacred kernel PSD
    ok("9.5  Sacred kernel PSD", kr_s.is_psd)

    # 9.6 — Sacred kernel diagonal = 1
    diag_ok = all(abs(kr_s.kernel_matrix[i, i] - 1.0) < 1e-10 for i in range(4))
    ok("9.6  Sacred kernel diagonal = 1", diag_ok)

    # 9.7 — Asymmetric kernel (X vs Y)
    Y = rng.randn(3, 2)
    kr_asym = qk_zz.compute_kernel(X[:4], Y)
    ok("9.7  Asymmetric kernel shape (4,3)",
       kr_asym.kernel_matrix.shape == (4, 3))


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 10: VARIATIONAL EIGENSOLVER — VQE
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_10():
    phase(10, "VARIATIONAL EIGENSOLVER — VQE")

    from l104_quantum_gate_engine.quantum_ml import (
        VariationalEigensolver, AnsatzLibrary, _total_magnetisation,
    )

    # Simple 2-qubit Hamiltonian: -Z₀Z₁ - 0.5(X₀ + X₁)  (transverse-field Ising)
    n = 2
    dim = 4
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)

    ZZ = np.kron(Z, Z)
    XI = np.kron(X, I)
    IX = np.kron(I, X)
    H_ising = -ZZ - 0.5 * (XI + IX)

    exact_eigs = np.linalg.eigvalsh(H_ising)
    exact_E0 = float(exact_eigs[0])

    # 10.1 — VQE runs
    vqe = VariationalEigensolver(learning_rate=0.15)
    result = vqe.run(H_ising, max_iterations=80, seed=42)
    ok("10.1  VQE completes",
       result.num_iterations > 0)

    # 10.2 — Ground energy ≥ exact (variational upper bound)
    ok("10.2  VQE E ≥ exact E₀ (within tolerance)",
       result.ground_energy >= exact_E0 - 0.01,
       f"VQE={result.ground_energy:.4f}, exact={exact_E0:.4f}")

    # 10.3 — Energy error reasonable (< 1.0 for 80 iterations)
    ok("10.3  Energy error < 1.0",
       result.energy_error is not None and result.energy_error < 1.0,
       f"error={result.energy_error}")

    # 10.4 — Energy history decreases (first → last)
    ok("10.4  Energy history trend downward",
       result.energy_history[-1] < result.energy_history[0],
       f"first={result.energy_history[0]:.4f}, last={result.energy_history[-1]:.4f}")

    # 10.5 — VQEResult.to_dict() works
    d = result.to_dict()
    ok("10.5  to_dict() has keys",
       all(k in d for k in ["ground_energy", "exact_ground_energy", "energy_error", "god_code"]))

    # 10.6 — GOD_CODE in result  (approximate)
    ok("10.6  GOD_CODE ≈ 527.518",
       abs(d["god_code"] - 527.518) < 0.1,
       f"got {d['god_code']}")

    # 10.7 — Energy landscape scan (use random base to break symmetry)
    ansatz = AnsatzLibrary.hardware_efficient(2, depth=1)
    rng = np.random.RandomState(42)
    base_p = rng.uniform(-1, 1, ansatz.num_parameters)
    angles, energies = vqe.energy_landscape(
        H_ising, ansatz, param_index=0, base_params=base_p, n_points=20,
    )
    ok("10.7  Energy landscape shape",
       len(angles) == 20 and len(energies) == 20)

    # 10.8 — Energy landscape has variation
    e_range = max(energies) - min(energies)
    ok("10.8  Energy landscape varies",
       e_range > 0.01, f"range={e_range:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 11: GATE CIRCUIT CONVENIENCE METHODS
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_11():
    phase(11, "GATE CIRCUIT CONVENIENCE METHODS")

    from l104_quantum_gate_engine.circuit import GateCircuit
    from l104_quantum_gate_engine.quantum_ml import ParameterisedCircuit

    circ = GateCircuit(3, "conv_test")

    # 11.1 — build_qnn() returns ParameterisedCircuit
    qnn = circ.build_qnn(depth=2, ansatz="hardware_efficient")
    ok("11.1  build_qnn() returns ParameterisedCircuit",
       isinstance(qnn, ParameterisedCircuit))

    # 11.2 — QNN matches circuit qubit count
    ok("11.2  QNN qubits = circuit qubits",
       qnn.num_qubits == 3)

    # 11.3 — build_qnn() with sacred ansatz
    qnn_sacred = circ.build_qnn(depth=1, ansatz="sacred")
    ok("11.3  Sacred QNN builds",
       isinstance(qnn_sacred, ParameterisedCircuit) and "Sacred" in qnn_sacred.name)

    # 11.4 — build_qnn() with strongly_entangling
    qnn_sel = circ.build_qnn(depth=1, ansatz="strongly_entangling")
    ok("11.4  Strongly entangling QNN builds",
       isinstance(qnn_sel, ParameterisedCircuit))

    # 11.5 — train_vqe() runs with default sacred Hamiltonian
    try:
        circ2 = GateCircuit(2, "vqe_conv")
        vqe_result = circ2.train_vqe(max_iterations=20, seed=42)
        ok("11.5  train_vqe() returns VQEResult",
           hasattr(vqe_result, "ground_energy"))
    except Exception as e:
        ok("11.5  train_vqe() returns VQEResult", False, str(e))

    # 11.6 — train_vqe() with custom Hamiltonian
    try:
        H_simple = -np.kron(
            np.array([[1,0],[0,-1]], dtype=complex),
            np.array([[1,0],[0,-1]], dtype=complex),
        )
        circ3 = GateCircuit(2, "vqe_custom")
        vqe2 = circ3.train_vqe(hamiltonian_matrix=H_simple, max_iterations=20, seed=42)
        ok("11.6  train_vqe() with custom H",
           hasattr(vqe2, "ground_energy"))
    except Exception as e:
        ok("11.6  train_vqe() with custom H", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 12: ORCHESTRATOR — QUANTUM_ML TARGET
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_12():
    phase(12, "ORCHESTRATOR — QUANTUM_ML TARGET")

    # 12.1 — QUANTUM_ML target exists
    try:
        from l104_quantum_gate_engine.orchestrator import ExecutionTarget
        _ = ExecutionTarget.QUANTUM_ML
        ok("12.1  ExecutionTarget.QUANTUM_ML exists", True)
    except Exception as e:
        ok("12.1  ExecutionTarget.QUANTUM_ML exists", False, str(e))
        return

    # 12.2 — get_engine() works
    try:
        from l104_quantum_gate_engine import get_engine
        engine = get_engine()
        ok("12.2  get_engine() returns orchestrator", engine is not None)
    except Exception as e:
        ok("12.2  get_engine() returns orchestrator", False, str(e))
        return

    # 12.3 — Execute with QUANTUM_ML target
    try:
        from l104_quantum_gate_engine.circuit import GateCircuit
        circ = GateCircuit(2, "qml_test")
        circ.h(0).cx(0, 1)
        result = engine.execute(circ, ExecutionTarget.QUANTUM_ML)
        ok("12.3  Execute QUANTUM_ML target",
           result is not None and result.target == ExecutionTarget.QUANTUM_ML)
    except Exception as e:
        ok("12.3  Execute QUANTUM_ML target", False, str(e))

    # 12.4 — Result metadata has ground_energy
    try:
        ok("12.4  Result has ground_energy",
           "ground_energy" in result.metadata,
           f"keys={list(result.metadata.keys())}")
    except Exception as e:
        ok("12.4  Result has ground_energy", False, str(e))

    # 12.5 — Result metadata has GOD_CODE
    try:
        gc = result.metadata.get("god_code", 0)
        ok("12.5  Metadata GOD_CODE ≈ 527.518",
           abs(gc - 527.518) < 0.1,
           f"got {gc}")
    except Exception as e:
        ok("12.5  Metadata GOD_CODE ≈ 527.518", False, str(e))

    # 12.6 — QuantumMLEngine.sacred_vqe()
    try:
        from l104_quantum_gate_engine.quantum_ml import QuantumMLEngine
        qml = QuantumMLEngine()
        svqe = qml.sacred_vqe(num_qubits=2, depth=2, max_iterations=30, seed=42)
        ok("12.6  sacred_vqe() returns dict",
           isinstance(svqe, dict) and "vqe_result" in svqe)
    except Exception as e:
        ok("12.6  sacred_vqe() returns dict", False, str(e))

    # 12.7 — QuantumMLEngine metrics tracked
    try:
        m = qml.metrics
        ok("12.7  Metrics tracked (vqe_runs > 0)",
           m.get("vqe_runs", 0) > 0,
           f"metrics={m}")
    except Exception as e:
        ok("12.7  Metrics tracked", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║  L104 QUANTUM GATE ENGINE — QUANTUM ML SUITE TEST SUITE            ║
║  12 phases · 86 tests · v8.0                                       ║
╚══════════════════════════════════════════════════════════════════════╝"""
    print(banner)

    t0 = time.time()

    phases = [
        test_phase_1,
        test_phase_2,
        test_phase_3,
        test_phase_4,
        test_phase_5,
        test_phase_6,
        test_phase_7,
        test_phase_8,
        test_phase_9,
        test_phase_10,
        test_phase_11,
        test_phase_12,
    ]

    for fn in phases:
        try:
            fn()
        except Exception as e:
            print(f"\n  💥 PHASE CRASHED: {e}")
            traceback.print_exc()
            global _fail
            _fail += 1

    elapsed = time.time() - t0

    print(f"\n{'═'*70}")
    print(f"  RESULTS: {_pass} passed · {_fail} failed · {_pass + _fail} total · {elapsed:.2f}s")
    print(f"{'═'*70}")

    if _errors:
        print("\n  FAILURES:")
        for e in _errors:
            print(f"    {e}")

    if _fail == 0:
        print("\n  🏆 ALL TESTS PASSED — Quantum ML Suite v8.0 VERIFIED")
    else:
        print(f"\n  ⚠️  {_fail} test(s) failed")

    sys.exit(0 if _fail == 0 else 1)


if __name__ == "__main__":
    main()
