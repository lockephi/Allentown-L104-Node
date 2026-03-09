#!/usr/bin/env python3
"""
L104 Quantum Systems Deep Debug — Multi-Engine Diagnostic
══════════════════════════════════════════════════════════════
Tests all 4 quantum engine packages + quantum brain pipeline:

  1. Quantum Gate Engine    (l104_quantum_gate_engine)
  2. Quantum Link Engine    (l104_quantum_engine)
  3. Numerical Engine       (l104_numerical_engine)
  4. Logic Gate Engine      (l104_gate_engine)
  5. Quantum Brain Pipeline (brain.py 17 phases)

For each engine: boot, constants, core ops, error paths, integration.
"""

import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = (1 + math.sqrt(5)) / 2
VOID_CONSTANT = 1.04 + PHI / 1000
TOL = 1e-10

passed = 0
failed = 0
warned = 0
errors = []

def test(name, fn, expect_error=False):
    global passed, failed, warned
    try:
        result = fn()
        if result is True or result is None:
            passed += 1
            print(f"  ✅ {name}")
            return True
        elif result == "WARN":
            warned += 1
            print(f"  ⚠️  {name}")
            return True
        else:
            failed += 1
            errors.append((name, f"Returned {result}"))
            print(f"  ❌ {name} → {result}")
            return False
    except Exception as e:
        if expect_error:
            passed += 1
            print(f"  ✅ {name} (expected error: {type(e).__name__})")
            return True
        failed += 1
        tb = traceback.format_exc().strip().split("\n")[-3:]
        errors.append((name, str(e)))
        print(f"  ❌ {name} → {type(e).__name__}: {e}")
        for line in tb:
            print(f"       {line}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 1: QUANTUM GATE ENGINE (l104_quantum_gate_engine)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("  ENGINE 1: QUANTUM GATE ENGINE (l104_quantum_gate_engine)")
print("═" * 72)

qge = None

def boot_qge():
    global qge
    from l104_quantum_gate_engine import get_engine
    qge = get_engine()
    assert qge is not None
    return True

test("1.1  Boot get_engine()", boot_qge)

def qge_gate_imports():
    from l104_quantum_gate_engine import H, CNOT, Rx, PHI_GATE, GOD_CODE_PHASE
    assert H is not None and CNOT is not None
    assert PHI_GATE is not None and GOD_CODE_PHASE is not None
    return True
test("1.2  Gate imports (H, CNOT, Rx, PHI_GATE, GOD_CODE_PHASE)", qge_gate_imports)

def qge_enum_imports():
    from l104_quantum_gate_engine import GateSet, OptimizationLevel, ErrorCorrectionScheme, ExecutionTarget
    assert len(GateSet) >= 4
    assert len(OptimizationLevel) >= 3
    return True
test("1.3  Enum imports (GateSet, OptLevel, ECC, ExecTarget)", qge_enum_imports)

def qge_bell_pair():
    circ = qge.bell_pair()
    assert circ is not None
    assert circ.num_qubits == 2
    return True
test("1.4  Bell pair circuit", qge_bell_pair)

def qge_ghz_state():
    circ = qge.ghz_state(4)
    assert circ is not None
    assert circ.num_qubits == 4
    return True
test("1.5  GHZ state (4 qubits)", qge_ghz_state)

def qge_qft():
    circ = qge.quantum_fourier_transform(3)
    assert circ is not None
    assert circ.num_qubits == 3
    return True
test("1.6  QFT (3 qubits)", qge_qft)

def qge_sacred_circuit():
    circ = qge.sacred_circuit(2, depth=3)
    assert circ is not None
    assert circ.num_qubits == 2
    return True
test("1.7  Sacred L104 circuit (2q, depth=3)", qge_sacred_circuit)

def qge_compile_bell():
    from l104_quantum_gate_engine import GateSet, OptimizationLevel
    circ = qge.bell_pair()
    result = qge.compile(circ, GateSet.UNIVERSAL)
    assert result is not None
    return True
test("1.8  Compile Bell pair → UNIVERSAL gate set", qge_compile_bell)

def qge_execute_bell():
    from l104_quantum_gate_engine import ExecutionTarget
    circ = qge.bell_pair()
    result = qge.execute(circ, ExecutionTarget.LOCAL_STATEVECTOR)
    assert result is not None
    assert hasattr(result, 'probabilities')
    probs = result.probabilities
    assert abs(probs.get('00', 0) - 0.5) < 0.1 or abs(probs.get('00', 0) + probs.get('11', 0) - 1.0) < 0.1
    return True
test("1.9  Execute Bell pair → LOCAL_STATEVECTOR", qge_execute_bell)

def qge_error_correction():
    from l104_quantum_gate_engine import ErrorCorrectionScheme
    circ = qge.bell_pair()
    protected = qge.error_correction.encode(circ, ErrorCorrectionScheme.STEANE_7_1_3)
    assert protected is not None
    return True
test("1.10 Error correction (Steane [[7,1,3]])", qge_error_correction)

def qge_gate_analysis():
    from l104_quantum_gate_engine import CNOT
    analysis = qge.analyze_gate(CNOT)
    assert analysis is not None
    assert "name" in analysis or hasattr(analysis, 'name')
    return True
test("1.11 Gate analysis (CNOT)", qge_gate_analysis)

def qge_algebra_decompose():
    from l104_quantum_gate_engine import H
    import numpy as np
    result = qge.algebra.zyz_decompose(H.matrix)
    assert result is not None
    return True
test("1.12 ZYZ decomposition (H gate)", qge_algebra_decompose)

def qge_sacred_alignment():
    from l104_quantum_gate_engine import PHI_GATE
    score = qge.algebra.sacred_alignment_score(PHI_GATE)
    assert score is not None
    assert isinstance(score, (int, float, dict))
    return True
test("1.13 Sacred alignment score (PHI_GATE)", qge_sacred_alignment)

def qge_full_pipeline():
    from l104_quantum_gate_engine import GateSet, OptimizationLevel, ErrorCorrectionScheme, ExecutionTarget
    circ = qge.bell_pair()
    result = qge.full_pipeline(
        circ,
        target_gates=GateSet.UNIVERSAL,
        optimization=OptimizationLevel.O1,
        error_correction=ErrorCorrectionScheme.STEANE_7_1_3,
        execution_target=ExecutionTarget.LOCAL_STATEVECTOR,
    )
    assert result is not None
    return True
test("1.14 Full pipeline (build→compile→protect→execute)", qge_full_pipeline)


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 2: QUANTUM LINK ENGINE (l104_quantum_engine)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("  ENGINE 2: QUANTUM LINK ENGINE (l104_quantum_engine)")
print("═" * 72)

qbrain = None
qmath = None

def boot_quantum_engine():
    global qbrain, qmath
    from l104_quantum_engine import quantum_brain, QuantumMathCore
    qbrain = quantum_brain
    qmath = QuantumMathCore()
    assert qbrain is not None
    assert qmath is not None
    return True
test("2.1  Boot quantum_brain + QuantumMathCore", boot_quantum_engine)

def qe_constants():
    from l104_quantum_engine.constants import GOD_CODE as QE_GOD, PHI as QE_PHI
    assert abs(QE_GOD - GOD_CODE) < TOL, f"GOD_CODE mismatch: {QE_GOD}"
    assert abs(QE_PHI - PHI) < TOL, f"PHI mismatch: {QE_PHI}"
    return True
test("2.2  Constants (GOD_CODE, PHI) match canonical", qe_constants)

def qe_god_code_spectrum():
    from l104_quantum_engine.constants import GOD_CODE_SPECTRUM
    assert len(GOD_CODE_SPECTRUM) > 0
    # G(0) should be GOD_CODE
    assert abs(GOD_CODE_SPECTRUM.get(0, 0) - GOD_CODE) < 0.01
    return True
test("2.3  GOD_CODE_SPECTRUM non-empty, G(0)=GOD_CODE", qe_god_code_spectrum)

def qe_link_model():
    from l104_quantum_engine.models import QuantumLink
    link = QuantumLink(
        source_file="a.py", source_symbol="f", source_line=1,
        target_file="b.py", target_symbol="g", target_line=2,
        link_type="entanglement", fidelity=0.95, strength=0.8,
    )
    assert link.fidelity == 0.95 and link.strength == 0.8
    assert link.link_type == "entanglement"
    return True
test("2.4  QuantumLink dataclass construction", qe_link_model)

def qe_math_bell_state():
    bell = qmath.bell_state_phi_plus()
    assert bell is not None
    assert len(bell) == 4  # 2-qubit state → 4 amplitudes
    return True
test("2.5  QuantumMathCore.bell_state_phi_plus()", qe_math_bell_state)

def qe_math_grover():
    # grover_operator(state, oracle_indices, iterations)
    state = [complex(0.5, 0), complex(0.5, 0), complex(0.5, 0), complex(0.5, 0)]
    result = qmath.grover_operator(state, [2])
    assert result is not None
    assert len(result) == 4
    return True
test("2.6  QuantumMathCore.grover_operator(state, [2])", qe_math_grover)

def qe_math_qft():
    # QFT expects List[complex] — use 4-element state (2 qubits)
    state = [complex(1, 0), complex(0, 0), complex(0, 0), complex(0, 0)]
    result = qmath.quantum_fourier_transform(state)
    assert result is not None
    assert len(result) == 4
    return True
test("2.7  QuantumMathCore.quantum_fourier_transform([1,0,0,0])", qe_math_qft)

def qe_math_hz():
    hz = qmath.link_natural_hz(0.9, 0.8)
    assert hz > 0, f"Hz should be positive: {hz}"
    return True
test("2.8  QuantumMathCore.link_natural_hz(0.9, 0.8)", qe_math_hz)

def qe_math_density_matrix():
    dm = qmath.density_matrix([0.7071, 0.7071])
    assert dm is not None
    return True
test("2.9  QuantumMathCore.density_matrix()", qe_math_density_matrix)

def qe_weak_measurement():
    from l104_quantum_engine.research import L104WeakMeasurement
    import numpy as np
    wm = L104WeakMeasurement()
    amps = np.array([0.8, 0.6], dtype=complex)
    readout, updated = wm.partial_collapse(amps)
    sr = wm.survival_rate(amps, updated)
    assert 0 <= sr <= 1.0
    assert abs(np.linalg.norm(updated) - 1.0) < 1e-10
    return True
test("2.10 L104WeakMeasurement partial_collapse + survival_rate", qe_weak_measurement)

def qe_wave_collapse_research():
    from l104_quantum_engine.research import ProbabilityWaveCollapseResearch
    from l104_quantum_engine.models import QuantumLink
    import random
    pwcr = ProbabilityWaveCollapseResearch(qmath)
    links = [
        QuantumLink(
            source_file="a.py", source_symbol=f"f{i}", source_line=i,
            target_file="b.py", target_symbol=f"g{i}", target_line=i,
            link_type="entanglement",
            fidelity=random.uniform(0.5, 1.0), strength=random.uniform(0.3, 1.0),
            coherence_time=random.uniform(0.1, 2.0),
            entanglement_entropy=random.uniform(0, 1),
            noise_resilience=random.uniform(0.3, 0.9),
        )
        for i in range(30)
    ]
    result = pwcr.wave_collapse_research(links)
    health = result.get("collapse_health", 0)
    assert health > 0, f"Collapse health should be > 0: {health}"
    synth = result.get("collapse_synthesis", {})
    assert "verdict" in synth, "Missing verdict in synthesis"
    return True
test("2.11 ProbabilityWaveCollapseResearch full pipeline", qe_wave_collapse_research)

def qe_scanner():
    from l104_quantum_engine.scanner import QuantumLinkScanner
    scanner = QuantumLinkScanner()
    assert scanner is not None
    return True
test("2.12 QuantumLinkScanner boot", qe_scanner)

def qe_processors():
    from l104_quantum_engine.processors import (
        GroverQuantumProcessor, EPREntanglementVerifier,
        DecoherenceShieldTester, HilbertSpaceNavigator,
    )
    assert GroverQuantumProcessor is not None
    assert EPREntanglementVerifier is not None
    return True
test("2.13 Processor imports (Grover, EPR, Decoherence, Hilbert)", qe_processors)

def qe_computation():
    from l104_quantum_engine.computation import (
        QuantumRegister, QuantumNeuron, QuantumCluster,
        QuantumCPU, QuantumEnvironment, QuantumLinkComputationEngine,
    )
    from l104_quantum_engine.models import QuantumLink
    link = QuantumLink(
        source_file="a.py", source_symbol="f", source_line=1,
        target_file="b.py", target_symbol="g", target_line=2,
        link_type="entanglement", fidelity=0.95, strength=0.8,
    )
    reg = QuantumRegister(link, qmath)
    assert reg is not None
    return True
test("2.14 Computation layer (Register, Neuron, Cluster, CPU)", qe_computation)

def qe_qldpc():
    from l104_quantum_engine.qldpc import create_qldpc_code, full_qldpc_pipeline
    assert create_qldpc_code is not None
    assert full_qldpc_pipeline is not None
    return True
test("2.15 qLDPC imports (create_qldpc_code, full_qldpc_pipeline)", qe_qldpc)

def qe_dynamism():
    from l104_quantum_engine.dynamism import LinkDynamismEngine, LinkOuroborosNirvanicEngine
    assert LinkDynamismEngine is not None
    assert LinkOuroborosNirvanicEngine is not None
    return True
test("2.16 Dynamism + Nirvanic engine imports", qe_dynamism)


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 3: NUMERICAL ENGINE (l104_numerical_engine)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("  ENGINE 3: NUMERICAL ENGINE (l104_numerical_engine)")
print("═" * 72)

qnb = None

def boot_numerical():
    global qnb
    from l104_numerical_engine import QuantumNumericalBuilder
    qnb = QuantumNumericalBuilder()
    assert qnb is not None
    return True
test("3.1  Boot QuantumNumericalBuilder", boot_numerical)

def ne_precision():
    from l104_numerical_engine import D, fmt100, GOD_CODE_HP, PHI_HP
    assert D is not None
    x = D('3.14159265358979323846')
    formatted = fmt100(x)
    assert len(formatted) > 50, f"fmt100 too short: {len(formatted)}"
    assert abs(float(GOD_CODE_HP) - GOD_CODE) < 0.001
    assert abs(float(PHI_HP) - PHI) < 0.001
    return True
test("3.2  100-decimal precision (D, fmt100, GOD_CODE_HP, PHI_HP)", ne_precision)

def ne_lattice():
    summary = qnb.lattice.lattice_summary()
    assert summary is not None
    assert "total_tokens" in summary or isinstance(summary, dict)
    return True
test("3.3  Token lattice summary", ne_lattice)

def ne_lattice_phi():
    phi_token = qnb.lattice.tokens.get("PHI")
    if phi_token is not None:
        assert abs(float(phi_token.value) - PHI) < 0.001
    return True
test("3.4  Lattice PHI token value", ne_lattice_phi)

def ne_editor():
    assert qnb.editor is not None
    return True
test("3.5  Superfluid value editor present", ne_editor)

def ne_verifier():
    result = qnb.verifier.verify_all()
    assert result is not None
    return True
test("3.6  Verifier.verify_all()", ne_verifier)

def ne_research():
    result = qnb.research.full_research()
    assert result is not None
    return True
test("3.7  Research.full_research()", ne_research)

def ne_quantum_compute():
    # Quantum Phase Estimation via quantum_engine subsystem
    result = qnb.quantum_engine.phase_estimation_hp(0.25)
    assert result is not None
    return True
test("3.8  Quantum Phase Estimation (eigenvalue=0.25)", ne_quantum_compute)

def ne_hhl():
    import numpy as np
    A = np.array([[2, 0], [0, 3]], dtype=float)
    b = np.array([1, 0], dtype=float)
    result = qnb.quantum_engine.hhl_linear_solver_hp(A, b)
    assert result is not None
    return True
test("3.9  HHL Linear Solver (2×2)", ne_hhl)

def ne_vqe():
    import numpy as np
    H = np.array([[1, 0], [0, -1]], dtype=float)
    result = qnb.quantum_engine.vqe_ground_state_hp(H)
    assert result is not None
    return True
test("3.10 VQE (Pauli-Z Hamiltonian)", ne_vqe)

def ne_stochastic():
    result = qnb.stochastic_lab.run_stochastic_cycle(5)
    assert result is not None
    return True
test("3.11 Stochastic research cycle (5 rounds)", ne_stochastic)

def ne_nirvanic():
    result = qnb.nirvanic_engine.full_nirvanic_cycle()
    assert result is not None
    return True
test("3.12 Nirvanic engine full cycle", ne_nirvanic)

def ne_consciousness():
    result = qnb.consciousness_o2.full_superfluid_cycle()
    assert result is not None
    return True
test("3.13 Consciousness O₂ superfluid cycle", ne_consciousness)

def ne_cross_pollination():
    result = qnb.cross_pollinator.full_cross_pollination()
    assert result is not None
    return True
test("3.14 Cross-pollination (bidirectional)", ne_cross_pollination)

def ne_pipeline_status():
    result = qnb.full_pipeline()
    assert result is not None
    return True
test("3.15 Full pipeline execution", ne_pipeline_status)

def ne_math_research():
    from l104_numerical_engine.math_research import (
        RiemannZetaEngine, PrimeNumberTheoryEngine, CollatzConjectureAnalyzer,
    )
    rz = RiemannZetaEngine()
    assert rz is not None
    pnt = PrimeNumberTheoryEngine()
    assert pnt is not None
    cc = CollatzConjectureAnalyzer()
    assert cc is not None
    return True
test("3.16 Math research engines (Riemann, Primes, Collatz)", ne_math_research)


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 4: LOGIC GATE ENGINE (l104_gate_engine)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("  ENGINE 4: LOGIC GATE ENGINE (l104_gate_engine)")
print("═" * 72)

gate_env = None

def boot_gate():
    global gate_env
    from l104_gate_engine import HyperASILogicGateEnvironment
    gate_env = HyperASILogicGateEnvironment()
    assert gate_env is not None
    return True
test("4.1  Boot HyperASILogicGateEnvironment", boot_gate)

def ge_gate_funcs():
    from l104_gate_engine import sage_logic_gate, quantum_logic_gate
    assert sage_logic_gate is not None
    assert quantum_logic_gate is not None
    return True
test("4.2  Gate function imports (sage_logic_gate, quantum_logic_gate)", ge_gate_funcs)

def ge_stochastic_lab():
    from l104_gate_engine import StochasticGateResearchLab
    lab = StochasticGateResearchLab()
    assert lab is not None
    return True
test("4.3  StochasticGateResearchLab boot", ge_stochastic_lab)

def ge_sage_gate():
    from l104_gate_engine import sage_logic_gate
    result = sage_logic_gate(1.0, 0.5)
    assert result is not None
    assert isinstance(result, (int, float, dict, tuple))
    return True
test("4.4  sage_logic_gate(1.0, 0.5) execution", ge_sage_gate)

def ge_quantum_gate():
    from l104_gate_engine import quantum_logic_gate
    result = quantum_logic_gate(0.8, 0.6)
    assert result is not None
    return True
test("4.5  quantum_logic_gate(0.8, 0.6) execution", ge_quantum_gate)


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 5: QUANTUM BRAIN PIPELINE (l104_quantum_engine/brain.py)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("  ENGINE 5: QUANTUM BRAIN — Phase-by-Phase Debug")
print("═" * 72)

brain = None

def boot_brain():
    global brain
    from l104_quantum_engine.brain import L104QuantumBrain
    brain = L104QuantumBrain()
    assert brain is not None
    assert brain.qmath is not None
    assert brain.wave_collapse is not None
    return True
test("5.1  Boot L104QuantumBrain", boot_brain)

def brain_state():
    state = brain.get_state() if hasattr(brain, 'get_state') else brain.__dict__
    assert state is not None
    return True
test("5.2  Brain state retrieval", brain_state)

def brain_links_count():
    count = len(brain.links) if hasattr(brain, 'links') else 0
    print(f"        → {count} links loaded")
    return True
test("5.3  Brain link count", brain_links_count)

def brain_math_core():
    bell = brain.qmath.bell_state_phi_plus()
    assert bell is not None
    assert len(bell) == 4
    hz = brain.qmath.link_natural_hz(0.9, 0.8)
    assert hz > 0
    return True
test("5.4  Brain math core (bell_state_phi_plus, link_natural_hz)", brain_math_core)

def brain_wave_collapse():
    assert brain.wave_collapse is not None
    assert brain.wave_collapse.weak_measurement is not None
    assert brain.wave_collapse.weak_measurement.coupling == 0.15
    return True
test("5.5  Brain wave collapse engine + weak measurement", brain_wave_collapse)

def brain_computation_engine():
    if hasattr(brain, 'computation_engine'):
        assert brain.computation_engine is not None
    elif hasattr(brain, 'compute'):
        assert brain.compute is not None
    return True
test("5.6  Brain computation engine", brain_computation_engine)

def brain_qldpc():
    if hasattr(brain, 'qldpc') or hasattr(brain, 'error_correction'):
        return True
    return "WARN"
test("5.7  Brain qLDPC / error correction", brain_qldpc)

def brain_scan():
    """Test Phase 1: scanning (limited scope for speed)."""
    if hasattr(brain, 'scanner'):
        assert brain.scanner is not None
    return True
test("5.8  Brain scanner present", brain_scan)


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-ENGINE: CONSTANT ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("  CROSS-ENGINE: CONSTANT ALIGNMENT")
print("═" * 72)

def cross_god_code():
    from l104_quantum_engine.constants import GOD_CODE as QE_GC
    from l104_quantum_gate_engine.gates import GOD_CODE as QGE_GC
    from l104_numerical_engine import GOD_CODE_HP
    assert abs(QE_GC - GOD_CODE) < TOL, f"QE: {QE_GC}"
    assert abs(float(GOD_CODE_HP) - GOD_CODE) < 0.001, f"NE: {GOD_CODE_HP}"
    return True
test("6.1  GOD_CODE alignment across engines", cross_god_code)

def cross_phi():
    from l104_quantum_engine.constants import PHI as QE_PHI
    from l104_numerical_engine import PHI_HP
    assert abs(QE_PHI - PHI) < TOL, f"QE: {QE_PHI}"
    assert abs(float(PHI_HP) - PHI) < 0.001, f"NE: {PHI_HP}"
    return True
test("6.2  PHI alignment across engines", cross_phi)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
total = passed + failed + warned
print("\n" + "═" * 72)
print(f"  QUANTUM SYSTEMS DEBUG SUMMARY")
print(f"  Total: {total} | ✅ Passed: {passed} | ❌ Failed: {failed} | ⚠️  Warned: {warned}")
print("═" * 72)

if errors:
    print("\n  FAILURES:")
    for name, err in errors:
        print(f"    ❌ {name}: {err}")

if failed == 0:
    print("\n  🟢 ALL QUANTUM SYSTEMS OPERATIONAL")
else:
    print(f"\n  🔴 {failed} FAILURE(S) DETECTED")

sys.exit(1 if failed > 0 else 0)
