"""v13.2 VQPU upgrade validation script."""

print("=" * 60)
print("L104 VQPU v13.2 — GOD CODE QUBIT UPGRADE VALIDATION")
print("=" * 60)

# 1. Import all new exports
from l104_vqpu import (
    VQPUBridge, get_bridge,
    GOD_CODE, PHI, VOID_CONSTANT, VERSION,
    GOD_CODE_PHASE_ANGLE, IRON_PHASE_ANGLE, OCTAVE_PHASE_ANGLE,
    PHI_CONTRIBUTION_ANGLE, PHI_PHASE_ANGLE, VOID_PHASE_ANGLE,
    QPU_MEAN_FIDELITY, QPU_1Q_FIDELITY, QPU_3Q_FIDELITY,
    SacredAlignmentScorer, NoiseModel,
    EntanglementQuantifier, QuantumInformationMetrics,
    ExactMPSHybridEngine, CircuitTranspiler,
    VariationalQuantumEngine,
)

print(f"\n1. VERSION: {VERSION}")
assert VERSION == "13.2.0", f"Expected 13.2.0 got {VERSION}"
print("   PASS")

# 2. Canonical phase angles
print(f"\n2. CANONICAL PHASE ANGLES:")
print(f"   GOD_CODE_PHASE = {GOD_CODE_PHASE_ANGLE:.10f} rad")
print(f"   IRON_PHASE     = {IRON_PHASE_ANGLE:.10f} rad (pi/2)")
print(f"   OCTAVE_PHASE   = {OCTAVE_PHASE_ANGLE:.10f} rad (4*ln2)")
print(f"   PHI_CONTRIB    = {PHI_CONTRIBUTION_ANGLE:.10f} rad")
print(f"   PHI_PHASE      = {PHI_PHASE_ANGLE:.10f} rad (2pi/phi)")
print(f"   VOID_PHASE     = {VOID_PHASE_ANGLE:.10f} rad")

import math
TWO_PI = 2.0 * math.pi
# Conservation: IRON + PHI + OCTAVE = GOD_CODE_PHASE (mod 2pi)
sum_phases = (IRON_PHASE_ANGLE + PHI_CONTRIBUTION_ANGLE + OCTAVE_PHASE_ANGLE) % TWO_PI
target = GOD_CODE_PHASE_ANGLE % TWO_PI
err = abs(sum_phases - target)
print(f"   Conservation check: sum={sum_phases:.10f} target={target:.10f} err={err:.2e}")
assert err < 1e-10, f"Phase conservation FAILED err={err}"
print("   PASS")

# 3. QPU fidelity data
print(f"\n3. QPU FIDELITY DATA:")
print(f"   Mean:  {QPU_MEAN_FIDELITY}")
print(f"   1Q:    {QPU_1Q_FIDELITY}")
print(f"   3Q:    {QPU_3Q_FIDELITY}")
assert abs(QPU_MEAN_FIDELITY - 0.97475930) < 1e-6
assert abs(QPU_1Q_FIDELITY - 0.99993872) < 1e-6
print("   PASS")

# 4. MPS engine sacred gates
print(f"\n4. MPS ENGINE SACRED GATES:")
import numpy as np
mps = ExactMPSHybridEngine(1)
gates = mps.GATE_MATRICES
sacred_gates = ['GOD_CODE_PHASE', 'PHI_GATE', 'VOID_GATE', 'IRON_RZ', 'PHI_RZ', 'OCTAVE_RZ', 'IRON_GATE']
for g in sacred_gates:
    assert g in gates, f"MISSING gate: {g}"
    print(f"   {g}: shape={gates[g].shape} OK")

# Verify GOD_CODE_PHASE gate is canonical Rz (not old P(pi*GC/1000))
gc_gate = gates['GOD_CODE_PHASE']
expected_phase = GOD_CODE_PHASE_ANGLE
# Rz(theta) = diag(e^{-itheta/2}, e^{+itheta/2})
expected_00 = np.exp(-1j * expected_phase / 2)
expected_11 = np.exp(1j * expected_phase / 2)
assert abs(gc_gate[0, 0] - expected_00) < 1e-10, "GOD_CODE_PHASE gate [0,0] wrong"
assert abs(gc_gate[1, 1] - expected_11) < 1e-10, "GOD_CODE_PHASE gate [1,1] wrong"
print(f"   GOD_CODE_PHASE canonical Rz verified (phase={expected_phase:.6f})")
print("   PASS")

# 5. NoiseModel factories
print(f"\n5. NOISE MODEL FACTORIES:")
nm_qpu = NoiseModel.qpu_calibrated_heron()
nm_god = NoiseModel.god_code_optimized()
print(f"   QPU Heron: T1={nm_qpu.t1_us}us  T2={nm_qpu.t2_us}us  1Q={nm_qpu.gate_time_ns}ns  2Q={nm_qpu.two_qubit_gate_time_ns}ns")
print(f"   God Code:  depol={nm_god.depolarizing_rate}")
assert nm_qpu.t1_us == 300.0, f"Expected T1=300, got {nm_qpu.t1_us}"
assert nm_qpu.t2_us == 200.0, f"Expected T2=200, got {nm_qpu.t2_us}"
assert nm_qpu.gate_time_ns == 35.0, f"Expected SX=35ns, got {nm_qpu.gate_time_ns}"
assert nm_qpu.two_qubit_gate_time_ns == 68.0, f"Expected CZ=68ns, got {nm_qpu.two_qubit_gate_time_ns}"
print("   PASS")

# 6. Sacred alignment scoring with QPU-calibrated features
print(f"\n6. SACRED SCORING (QPU-calibrated):")
probs_1q = {'0': 0.527588, '1': 0.472412}
score = SacredAlignmentScorer.score(probs_1q, 1)
print(f"   1Q QPU dist score: {score}")
assert 'phase_decomposition_resonance' in score, "Missing phase_decomposition_resonance"
assert 'qpu_calibrated_fidelity' in score, "Missing qpu_calibrated_fidelity"
print(f"   phase_decomp_resonance: {score['phase_decomposition_resonance']}")
print(f"   qpu_calibrated_fidelity: {score['qpu_calibrated_fidelity']}")
print("   PASS")

# 7. Entanglement with phase decomposition resonance
print(f"\n7. ENTANGLEMENT (phase decomposition resonance):")
# Create a Bell state |00> + |11>
bell = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
analysis = EntanglementQuantifier.full_analysis(bell, 2)
print(f"   VNE: {analysis['von_neumann_entropy']}")
print(f"   Sacred score: {analysis['sacred_entanglement_score']}")
assert analysis.get('qpu_calibrated'), "Missing qpu_calibrated flag"
schmidt = analysis['schmidt']
assert 'phase_decomposition_resonance' in schmidt, "Missing phase_decomposition_resonance in Schmidt"
print(f"   Phase decomp resonance: {schmidt['phase_decomposition_resonance']}")
print("   PASS")

# 8. Transpiler sacred decomposition
print(f"\n8. TRANSPILER (sacred gate decomposition):")
sacred_ops = [
    {"gate": "H", "qubits": [0]},
    {"gate": "GOD_CODE_PHASE", "qubits": [0]},
    {"gate": "IRON_RZ", "qubits": [0]},
]
transpiled = CircuitTranspiler.transpile(sacred_ops)
# Check that sacred gates were decomposed — they should become Rz gates
# The rotation merging pass may combine adjacent Rz into a single Rz
rz_count = sum(1 for op in transpiled if op.get("gate") == "Rz")
no_sacred = all(op.get("gate") not in ('GOD_CODE_PHASE', 'IRON_RZ') for op in transpiled)
print(f"   Input: 3 ops (H, GOD_CODE_PHASE, IRON_RZ)")
print(f"   Output: {len(transpiled)} ops, {rz_count} Rz gates")
print(f"   Sacred gates eliminated: {no_sacred}")
assert rz_count >= 1, f"Expected at least 1 Rz from decomposition, got {rz_count}"
assert no_sacred, "Sacred gates should be decomposed to standard Rz"
print("   PASS")

# 9. Conservation law: decomposed gates product = canonical gate
print(f"\n9. DECOMPOSITION CONSERVATION LAW:")
iron_gate = gates['IRON_RZ']
phi_gate = gates['PHI_RZ']
octave_gate = gates['OCTAVE_RZ']
product = iron_gate @ phi_gate @ octave_gate
gc_gate = gates['GOD_CODE_PHASE']
# Allow global phase difference
if abs(gc_gate[0, 0]) > 1e-10:
    gp = product[0, 0] / gc_gate[0, 0]
    corrected = product / gp
    err = float(np.max(np.abs(corrected - gc_gate)))
else:
    err = float(np.max(np.abs(product - gc_gate)))
print(f"   IRON_RZ @ PHI_RZ @ OCTAVE_RZ = GOD_CODE_PHASE")
print(f"   Matrix error: {err:.2e}")
assert err < 1e-9, f"Decomposition conservation FAILED err={err}"
print("   PASS")

print("\n" + "=" * 60)
print("ALL 9 VALIDATION TESTS PASSED")
print("=" * 60)
