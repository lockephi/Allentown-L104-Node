"""
===============================================================================
L104 QUANTUM GATE ENGINE — ANALOG QUANTUM SIMULATOR TEST SUITE
===============================================================================

12-phase, 80-test validation of the Analog Quantum Simulator (v7.0).

Phases:
   1. Import & Singleton                         (6 tests)
   2. Pauli Algebra & Matrix Construction         (8 tests)
   3. Hamiltonian Builder — Built-in Models       (8 tests)
   4. Hamiltonian Properties — Spectrum & Ground  (6 tests)
   5. Exact Evolution — Unitarity & Conservation  (7 tests)
   6. Trotter Engine — Product Formulas           (7 tests)
   7. Trotter Benchmark — Error Scaling           (7 tests)
   8. Observable Engine — Expectation Values      (7 tests)
   9. Circuit Generation — trotterise_to_circuit  (6 tests)
  10. GateCircuit Convenience Methods             (6 tests)
  11. Orchestrator — ANALOG_SIM Target            (6 tests)
  12. Sacred Hamiltonian & Alignment              (6 tests)

Run:
    .venv/bin/python -m l104_quantum_gate_engine.test_analog_sim

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

    # 1.1 — Core module import
    try:
        from l104_quantum_gate_engine.analog import (
            AnalogSimulator, HamiltonianBuilder, ExactEvolution,
            TrotterEngine, TrotterBenchmark, ObservableEngine,
        )
        ok("1.1  Core classes import", True)
    except Exception as e:
        ok("1.1  Core classes import", False, str(e))
        return

    # 1.2 — Data classes import
    try:
        from l104_quantum_gate_engine.analog import (
            Hamiltonian, HamiltonianTerm, EvolutionResult,
            TrotterBenchmarkResult, HamiltonianType, TrotterOrder,
        )
        ok("1.2  Data classes import", True)
    except Exception as e:
        ok("1.2  Data classes import", False, str(e))

    # 1.3 — Constants import
    try:
        from l104_quantum_gate_engine.analog import (
            MAX_ANALOG_QUBITS, SACRED_COUPLING, SACRED_FIELD,
        )
        ok("1.3  Constants import", MAX_ANALOG_QUBITS == 14)
    except Exception as e:
        ok("1.3  Constants import", False, str(e))

    # 1.4 — Singleton
    try:
        from l104_quantum_gate_engine.analog import get_analog_simulator
        s1 = get_analog_simulator()
        s2 = get_analog_simulator()
        ok("1.4  Singleton identity", s1 is s2)
    except Exception as e:
        ok("1.4  Singleton identity", False, str(e))

    # 1.5 — Package-level import
    try:
        from l104_quantum_gate_engine import (
            AnalogSimulator, HamiltonianBuilder, TrotterOrder,
            get_analog_simulator, MAX_ANALOG_QUBITS,
        )
        ok("1.5  Package-level re-exports", True)
    except Exception as e:
        ok("1.5  Package-level re-exports", False, str(e))

    # 1.6 — trotterise_to_circuit import
    try:
        from l104_quantum_gate_engine.analog import trotterise_to_circuit
        ok("1.6  trotterise_to_circuit import", callable(trotterise_to_circuit))
    except Exception as e:
        ok("1.6  trotterise_to_circuit import", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: PAULI ALGEBRA & MATRIX CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_2():
    phase(2, "PAULI ALGEBRA & MATRIX CONSTRUCTION")

    from l104_quantum_gate_engine.analog import (
        _pauli_string_matrix, _PAULI, HamiltonianTerm,
    )

    # 2.1 — Pauli matrices are unitary
    for name, mat in _PAULI.items():
        prod = mat @ mat.conj().T
        ok(f"2.1  {name} is unitary", np.allclose(prod, np.eye(2)))

    # 2.2 — Pauli matrices are Hermitian
    for name, mat in _PAULI.items():
        ok(f"2.2  {name} is Hermitian", np.allclose(mat, mat.conj().T))

    # INDIVIDUAL TESTS 2.3-2.8

    # 2.3 — Single-qubit Z on qubit 0 of 2-qubit system
    mat = _pauli_string_matrix([(0, "Z")], 2)
    expected = np.kron(_PAULI["Z"], _PAULI["I"])
    ok("2.3  Z⊗I matrix", np.allclose(mat, expected))

    # 2.4 — Two-qubit ZZ
    mat = _pauli_string_matrix([(0, "Z"), (1, "Z")], 2)
    expected = np.kron(_PAULI["Z"], _PAULI["Z"])
    ok("2.4  Z⊗Z matrix", np.allclose(mat, expected))

    # 2.5 — Three-qubit X⊗I⊗Z
    mat = _pauli_string_matrix([(0, "X"), (2, "Z")], 3)
    expected = np.kron(np.kron(_PAULI["X"], _PAULI["I"]), _PAULI["Z"])
    ok("2.5  X⊗I⊗Z matrix", np.allclose(mat, expected))

    # 2.6 — Empty paulis list → full identity
    mat = _pauli_string_matrix([], 3)
    ok("2.6  Empty → identity", np.allclose(mat, np.eye(8)))

    # 2.7 — Pauli string matrix dimension
    mat = _pauli_string_matrix([(1, "Y")], 4)
    ok("2.7  4-qubit dim = 16×16", mat.shape == (16, 16))

    # 2.8 — XX eigenvalues
    mat = _pauli_string_matrix([(0, "X"), (1, "X")], 2)
    evals = sorted(np.linalg.eigvalsh(mat).real)
    ok("2.8  XX eigenvalues ±1", np.allclose(evals, [-1, -1, 1, 1]))


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: HAMILTONIAN BUILDER — BUILT-IN MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_3():
    phase(3, "HAMILTONIAN BUILDER — BUILT-IN MODELS")

    from l104_quantum_gate_engine.analog import (
        HamiltonianBuilder, HamiltonianType,
    )
    hb = HamiltonianBuilder

    # 3.1 — Transverse Ising construction
    H = hb.transverse_ising(4, J=1.0, h=0.5)
    ok("3.1  Ising type", H.hamiltonian_type == HamiltonianType.TRANSVERSE_ISING)

    # 3.2 — Ising term count: 3 ZZ bonds + 4 X fields = 7
    ok("3.2  Ising terms", H.num_terms == 7, f"got {H.num_terms}")

    # 3.3 — Ising matrix is Hermitian
    mat = H.matrix()
    ok("3.3  Ising Hermitian", np.allclose(mat, mat.conj().T))

    # 3.4 — Heisenberg XXX
    H = hb.heisenberg_xxx(3, J=1.0)
    ok("3.4  XXX type", H.hamiltonian_type == HamiltonianType.HEISENBERG_XXX)

    # 3.5 — XXX term count: 2 bonds × 3 Paulis = 6
    ok("3.5  XXX terms", H.num_terms == 6, f"got {H.num_terms}")

    # 3.6 — XXZ includes delta
    H = hb.heisenberg_xxz(3, J=1.0, delta=0.5)
    ok("3.6  XXZ metadata", H.metadata.get("delta") == 0.5)

    # 3.7 — Hubbard construction
    H = hb.hubbard_1d(4, t_hop=1.0, U=2.0)
    ok("3.7  Hubbard type", H.hamiltonian_type == HamiltonianType.HUBBARD)

    # 3.8 — Hubbard Hermitian
    mat = H.matrix()
    ok("3.8  Hubbard Hermitian", np.allclose(mat, mat.conj().T))


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: HAMILTONIAN PROPERTIES — SPECTRUM & GROUND STATE
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_4():
    phase(4, "HAMILTONIAN PROPERTIES — SPECTRUM & GROUND STATE")

    from l104_quantum_gate_engine.analog import HamiltonianBuilder

    H = HamiltonianBuilder.transverse_ising(3, J=1.0, h=0.5)

    # 4.1 — Full eigenspectrum size
    evals, evecs = H.eigenspectrum()
    ok("4.1  Eigenspectrum size", len(evals) == 8, f"got {len(evals)}")

    # 4.2 — Eigenvalues are real
    ok("4.2  Eigenvalues real", np.allclose(evals.imag, 0))

    # 4.3 — Eigenvectors are orthonormal
    prod = evecs.conj().T @ evecs
    ok("4.3  Eigenvectors orthonormal", np.allclose(prod, np.eye(8), atol=1e-10))

    # 4.4 — Ground state energy matches smallest eigenvalue
    E0, gs = H.ground_state()
    ok("4.4  Ground energy matches", abs(E0 - evals[0]) < 1e-10)

    # 4.5 — Ground state is normalised
    ok("4.5  Ground state norm", abs(np.linalg.norm(gs) - 1.0) < 1e-10)

    # 4.6 — Energy gap is positive
    gap = H.energy_gap()
    ok("4.6  Energy gap > 0", gap > 0, f"gap={gap}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: EXACT EVOLUTION — UNITARITY & CONSERVATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_5():
    phase(5, "EXACT EVOLUTION — UNITARITY & CONSERVATION")

    from l104_quantum_gate_engine.analog import (
        HamiltonianBuilder, ExactEvolution,
    )

    H = HamiltonianBuilder.transverse_ising(3, J=1.0, h=0.5)
    psi0 = np.zeros(8, dtype=complex)
    psi0[0] = 1.0

    # 5.1 — Unitary is unitary
    U = ExactEvolution.unitary(H, t=1.0)
    prod = U @ U.conj().T
    ok("5.1  U(t) is unitary", np.allclose(prod, np.eye(8), atol=1e-10))

    # 5.2 — Evolved state normalised
    psi_t = ExactEvolution.evolve(H, psi0, t=1.0)
    ok("5.2  Evolved norm = 1", abs(np.linalg.norm(psi_t) - 1.0) < 1e-10)

    # 5.3 — U(0) = Identity
    U0 = ExactEvolution.unitary(H, t=0.0)
    ok("5.3  U(0) = I", np.allclose(U0, np.eye(8), atol=1e-10))

    # 5.4 — Time series returns correct point count
    times = np.linspace(0, 2.0, 20).tolist()
    result = ExactEvolution.time_series(H, psi0, times)
    ok("5.4  Time series point count", len(result.states) == 20)

    # 5.5 — Energy constant throughout evolution
    energies = result.energies
    ok("5.5  Energy conserved", max(energies) - min(energies) < 1e-10,
       f"spread={max(energies)-min(energies):.2e}")

    # 5.6 — First state is psi0
    ok("5.6  State at t=0 ≈ ψ₀", np.allclose(result.states[0], psi0, atol=1e-10))

    # 5.7 — Composition: U(t1+t2) = U(t2)·U(t1)
    U1 = ExactEvolution.unitary(H, t=0.5)
    U2 = ExactEvolution.unitary(H, t=0.7)
    U12 = ExactEvolution.unitary(H, t=1.2)
    ok("5.7  U(t1+t2) ≈ U(t2)·U(t1)", np.allclose(U2 @ U1, U12, atol=1e-8))


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 6: TROTTER ENGINE — PRODUCT FORMULAS
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_6():
    phase(6, "TROTTER ENGINE — PRODUCT FORMULAS")

    from l104_quantum_gate_engine.analog import (
        HamiltonianBuilder, ExactEvolution, TrotterEngine, TrotterOrder,
    )

    H = HamiltonianBuilder.transverse_ising(3, J=1.0, h=0.5)
    psi0 = np.zeros(8, dtype=complex)
    psi0[0] = 1.0
    t = 1.0

    psi_exact = ExactEvolution.evolve(H, psi0, t)

    # 6.1 — First-order Trotter converges
    psi_t1 = TrotterEngine.evolve(H, psi0, t, 50, TrotterOrder.FIRST)
    fid1 = abs(np.vdot(psi_exact, psi_t1)) ** 2
    ok("6.1  1st-order F > 0.9", fid1 > 0.9, f"F={fid1:.6f}")

    # 6.2 — Second-order Trotter is more accurate
    psi_t2 = TrotterEngine.evolve(H, psi0, t, 50, TrotterOrder.SECOND)
    fid2 = abs(np.vdot(psi_exact, psi_t2)) ** 2
    ok("6.2  2nd-order F > 0.99", fid2 > 0.99, f"F={fid2:.6f}")

    # 6.3 — Fourth-order Trotter very accurate
    psi_t4 = TrotterEngine.evolve(H, psi0, t, 20, TrotterOrder.FOURTH)
    fid4 = abs(np.vdot(psi_exact, psi_t4)) ** 2
    ok("6.3  4th-order F > 0.999", fid4 > 0.999, f"F={fid4:.6f}")

    # 6.4 — Order hierarchy
    ok("6.4  F(4th) ≥ F(2nd) ≥ F(1st)", fid4 >= fid2 >= fid1 * 0.99,
       f"F1={fid1:.6f} F2={fid2:.6f} F4={fid4:.6f}")

    # 6.5 — Trotter preserves normalisation
    ok("6.5  Trotter norm = 1", abs(np.linalg.norm(psi_t2) - 1.0) < 1e-6)

    # 6.6 — More steps → better fidelity
    psi_few = TrotterEngine.evolve(H, psi0, t, 5, TrotterOrder.SECOND)
    psi_many = TrotterEngine.evolve(H, psi0, t, 100, TrotterOrder.SECOND)
    fid_few = abs(np.vdot(psi_exact, psi_few)) ** 2
    fid_many = abs(np.vdot(psi_exact, psi_many)) ** 2
    ok("6.6  More steps → higher F", fid_many > fid_few,
       f"few={fid_few:.6f} many={fid_many:.6f}")

    # 6.7 — Gate count estimate is positive
    gc = TrotterEngine.gate_count_per_step(H, TrotterOrder.SECOND)
    ok("6.7  Gate count > 0", gc > 0, f"gates_per_step={gc}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 7: TROTTER BENCHMARK — ERROR SCALING
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_7():
    phase(7, "TROTTER BENCHMARK — ERROR SCALING")

    from l104_quantum_gate_engine.analog import (
        HamiltonianBuilder, TrotterBenchmark, TrotterOrder,
    )

    H = HamiltonianBuilder.transverse_ising(3, J=1.0, h=0.5)
    psi0 = np.zeros(8, dtype=complex)
    psi0[0] = 1.0
    steps = [2, 5, 10, 20, 50, 100]

    # 7.1 — Benchmark completes for 1st order
    r1 = TrotterBenchmark.fidelity_vs_steps(H, psi0, 1.0, steps, TrotterOrder.FIRST)
    ok("7.1  1st-order benchmark", len(r1.fidelities) == len(steps))

    # 7.2 — Benchmark completes for 2nd order
    r2 = TrotterBenchmark.fidelity_vs_steps(H, psi0, 1.0, steps, TrotterOrder.SECOND)
    ok("7.2  2nd-order benchmark", len(r2.fidelities) == len(steps))

    # 7.3 — Fidelity increases with more steps
    ok("7.3  Fidelity monotone", r2.fidelities[-1] >= r2.fidelities[0],
       f"first={r2.fidelities[0]:.6f} last={r2.fidelities[-1]:.6f}")

    # 7.4 — Error decreases
    ok("7.4  Error decreases", r2.errors[-1] <= r2.errors[0],
       f"first={r2.errors[0]:.2e} last={r2.errors[-1]:.2e}")

    # 7.5 — Error scaling exponent exists
    ok("7.5  Scaling exponent computed",
       r2.error_scaling_exponent is not None and r2.error_scaling_exponent > 0,
       f"exp={r2.error_scaling_exponent}")

    # 7.6 — Gate counts scale with steps
    ok("7.6  Gate counts scale", r2.gate_counts[-1] > r2.gate_counts[0])

    # 7.7 — Serialisation works
    d = r2.to_dict()
    ok("7.7  to_dict has god_code", "god_code" in d and abs(d["god_code"] - 527.518) < 0.1)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 8: OBSERVABLE ENGINE — EXPECTATION VALUES
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_8():
    phase(8, "OBSERVABLE ENGINE — EXPECTATION VALUES")

    from l104_quantum_gate_engine.analog import (
        ObservableEngine, HamiltonianBuilder, ExactEvolution, _PAULI,
        _pauli_string_matrix,
    )

    # |0⟩ state
    psi0 = np.array([1, 0], dtype=complex)

    # 8.1 — ⟨0|Z|0⟩ = +1
    val = ObservableEngine.expectation(psi0, _PAULI["Z"])
    ok("8.1  ⟨0|Z|0⟩ = 1", abs(val - 1.0) < 1e-10, f"got {val}")

    # 8.2 — ⟨0|X|0⟩ = 0
    val = ObservableEngine.expectation(psi0, _PAULI["X"])
    ok("8.2  ⟨0|X|0⟩ = 0", abs(val) < 1e-10, f"got {val}")

    # 8.3 — ⟨+|X|+⟩ = 1  (|+⟩ = (|0⟩+|1⟩)/√2)
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    val = ObservableEngine.expectation(plus, _PAULI["X"])
    ok("8.3  ⟨+|X|+⟩ = 1", abs(val - 1.0) < 1e-10, f"got {val}")

    # 8.4 — Pauli expectation over multi-qubit state
    psi00 = np.array([1, 0, 0, 0], dtype=complex)
    val = ObservableEngine.pauli_expectation(psi00, 0, "Z", 2)
    ok("8.4  ⟨00|Z₀|00⟩ = 1", abs(val - 1.0) < 1e-10)

    # 8.5 — Magnetisation of |000⟩
    psi000 = np.zeros(8, dtype=complex); psi000[0] = 1.0
    mag = ObservableEngine.magnetisation(psi000, 3, "Z")
    ok("8.5  M_Z(|000⟩) = 1", abs(mag - 1.0) < 1e-10, f"got {mag}")

    # 8.6 — Energy matches Hamiltonian expectation
    H = HamiltonianBuilder.transverse_ising(3, J=1.0, h=0.5)
    E = ObservableEngine.energy(psi000, H)
    H_mat = H.matrix()
    E_direct = float(np.real(psi000.conj() @ H_mat @ psi000))
    ok("8.6  Energy consistent", abs(E - E_direct) < 1e-10)

    # 8.7 — Two-point correlator
    corr = ObservableEngine.two_point_correlator(psi000, 0, 1, "Z", 3)
    ok("8.7  ZZ correlator (product state) ≈ 0", abs(corr) < 1e-10,
       f"got {corr}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 9: CIRCUIT GENERATION — trotterise_to_circuit
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_9():
    phase(9, "CIRCUIT GENERATION — trotterise_to_circuit")

    from l104_quantum_gate_engine.analog import (
        HamiltonianBuilder, trotterise_to_circuit, TrotterOrder,
    )

    H = HamiltonianBuilder.transverse_ising(3, J=1.0, h=0.5)

    # 9.1 — Circuit builds successfully
    circ = trotterise_to_circuit(H, t=1.0, n_steps=5, order=TrotterOrder.FIRST)
    ok("9.1  Circuit created", circ is not None)

    # 9.2 — Correct qubit count
    ok("9.2  Qubit count = 3", circ.num_qubits == 3)

    # 9.3 — Has operations
    ok("9.3  Has operations", circ.num_operations > 0,
       f"ops={circ.num_operations}")

    # 9.4 — Second-order has more gates
    circ2 = trotterise_to_circuit(H, t=1.0, n_steps=5, order=TrotterOrder.SECOND)
    ok("9.4  2nd-order more gates", circ2.num_operations >= circ.num_operations,
       f"1st={circ.num_operations} 2nd={circ2.num_operations}")

    # 9.5 — Fourth-order builds
    circ4 = trotterise_to_circuit(H, t=1.0, n_steps=3, order=TrotterOrder.FOURTH)
    ok("9.5  4th-order builds", circ4.num_operations > 0)

    # 9.6 — More steps → more gates
    circ_few = trotterise_to_circuit(H, t=1.0, n_steps=2, order=TrotterOrder.SECOND)
    circ_many = trotterise_to_circuit(H, t=1.0, n_steps=10, order=TrotterOrder.SECOND)
    ok("9.6  More steps → more gates",
       circ_many.num_operations > circ_few.num_operations)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 10: GATECIRCUIT CONVENIENCE METHODS
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_10():
    phase(10, "GATECIRCUIT CONVENIENCE METHODS")

    from l104_quantum_gate_engine import GateCircuit, H as Hgate

    circ = GateCircuit(3, "test_analog_conv")
    circ.h(0).h(1).h(2)

    # 10.1 — analog_evolve method exists
    ok("10.1 analog_evolve exists", hasattr(circ, 'analog_evolve'))

    # 10.2 — analog_evolve returns EvolutionResult
    result = circ.analog_evolve(t=1.0, n_points=10)
    ok("10.2 analog_evolve returns result", hasattr(result, 'states'))

    # 10.3 — Correct number of time points
    ok("10.3 Time points = 10", len(result.states) == 10)

    # 10.4 — trotterise method exists
    ok("10.4 trotterise exists", hasattr(circ, 'trotterise'))

    # 10.5 — trotterise returns a circuit
    trotter_circ = circ.trotterise(t=1.0, n_steps=5, order=2)
    ok("10.5 trotterise returns GateCircuit",
       type(trotter_circ).__name__ == 'GateCircuit')

    # 10.6 — Trotter circuit has correct qubit count
    ok("10.6 Trotter qubits = 3", trotter_circ.num_qubits == 3)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 11: ORCHESTRATOR — ANALOG_SIM TARGET
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_11():
    phase(11, "ORCHESTRATOR — ANALOG_SIM TARGET")

    from l104_quantum_gate_engine import ExecutionTarget, get_engine, GateCircuit

    # 11.1 — ANALOG_SIM target exists
    ok("11.1 ANALOG_SIM in targets", hasattr(ExecutionTarget, 'ANALOG_SIM'))

    # 11.2 — Engine instantiates
    engine = get_engine()
    ok("11.2 Engine creates", engine is not None)

    # 11.3 — Execute with ANALOG_SIM
    circ = GateCircuit(3, "test_analog_exec")
    circ.h(0).cx(0, 1).cx(1, 2)
    try:
        result = engine.execute(circ, ExecutionTarget.ANALOG_SIM)
        ok("11.3 ANALOG_SIM executes", result is not None)
    except Exception as e:
        ok("11.3 ANALOG_SIM executes", False, str(e))

    # 11.4 — Result has metadata
    ok("11.4 Has metadata", result.metadata is not None and len(result.metadata) > 0)

    # 11.5 — Metadata has energy gap
    ok("11.5 Has energy_gap", "energy_gap" in result.metadata,
       f"keys={list(result.metadata.keys())[:5]}")

    # 11.6 — Result target correct
    ok("11.6 Target = ANALOG_SIM", result.target == ExecutionTarget.ANALOG_SIM)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 12: SACRED HAMILTONIAN & ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_12():
    phase(12, "SACRED HAMILTONIAN & ALIGNMENT")

    from l104_quantum_gate_engine.analog import (
        HamiltonianBuilder, AnalogSimulator, ObservableEngine,
        SACRED_COUPLING, SACRED_FIELD,
    )
    from l104_quantum_gate_engine.constants import GOD_CODE, PHI

    # 12.1 — Sacred coupling formula
    expected_J = GOD_CODE / (104 * PHI)
    ok("12.1 Sacred coupling matches", abs(SACRED_COUPLING - expected_J) < 1e-10,
       f"J={SACRED_COUPLING} expected={expected_J}")

    # 12.2 — Sacred field formula
    expected_h = PHI / 104
    ok("12.2 Sacred field matches", abs(SACRED_FIELD - expected_h) < 1e-10)

    # 12.3 — Sacred Hamiltonian builds
    H = HamiltonianBuilder.sacred_hamiltonian(4)
    ok("12.3 Sacred H builds", H.num_terms > 0)

    # 12.4 — Sacred Hamiltonian is Hermitian
    mat = H.matrix()
    ok("12.4 Sacred H Hermitian", np.allclose(mat, mat.conj().T))

    # 12.5 — Full sacred analysis runs
    sim = AnalogSimulator()
    analysis = sim.sacred_analysis(num_qubits=3, t=1.0)
    ok("12.5 Sacred analysis completes",
       "energy_gap" in analysis and "eigenvalues" in analysis)

    # 12.6 — Sacred metadata has god_code
    ok("12.6 Sacred analysis god_code",
       analysis.get("god_code") == GOD_CODE)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║  L104 QUANTUM GATE ENGINE — ANALOG QUANTUM SIMULATOR TEST SUITE  (v7.0)    ║
║  12 phases · 80 tests · continuous Hamiltonian evolution                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    t0 = time.time()

    phases = [
        test_phase_1, test_phase_2, test_phase_3, test_phase_4,
        test_phase_5, test_phase_6, test_phase_7, test_phase_8,
        test_phase_9, test_phase_10, test_phase_11, test_phase_12,
    ]

    for fn in phases:
        try:
            fn()
        except Exception as e:
            global _fail
            _fail += 1
            _errors.append(f"  💥 PHASE CRASH: {fn.__name__} — {e}")
            traceback.print_exc()

    elapsed = time.time() - t0
    total = _pass + _fail

    print(f"\n{'═'*70}")
    print(f"  RESULTS: {_pass}/{total} passed  ({_fail} failed)  [{elapsed:.2f}s]")
    if _errors:
        print(f"\n  FAILURES:")
        for err in _errors:
            print(f"    {err}")
    print(f"{'═'*70}")
    print(f"  GOD_CODE = 527.5184818492612 | PILOT: LONDEL")
    print(f"{'═'*70}\n")

    sys.exit(0 if _fail == 0 else 1)


if __name__ == "__main__":
    main()
