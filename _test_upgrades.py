#!/usr/bin/env python3
"""Verification test for topological unitary research upgrades across all packages."""

import math
import sys

PASS = 0
FAIL = 0

def check(label, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {label}")
    else:
        FAIL += 1
        print(f"  ✗ {label}")


print("=" * 70)
print("TEST 1: l104_quantum_gate_engine/constants.py")
print("=" * 70)
from l104_quantum_gate_engine.constants import (
    TOPOLOGICAL_CORRELATION_LENGTH, TOPOLOGICAL_ERROR_RATE_D8,
    TOPOLOGICAL_ERROR_RATE_D13, FUNDAMENTAL_STEP, SEMITONE_RATIO,
    DIAL_BITS_A, DIAL_BITS_B, DIAL_BITS_C, DIAL_BITS_D,
    DIAL_TOTAL_BITS, DIAL_CONFIGURATIONS, IRON_MANIFOLD_QUBITS,
    IRON_MANIFOLD_HILBERT_DIM, ANCILLA_QUBITS,
    FIBONACCI_7, FACTOR_13_286, FACTOR_13_104, FACTOR_13_416,
    OCTAVE_STEPS, SEMITONE_STEPS, FOUR_OCTAVE_OFFSET,
    TOPOLOGICAL_DEFAULT_DEPTH, PHI,
)
check("ξ = 1/φ ≈ 0.618", abs(TOPOLOGICAL_CORRELATION_LENGTH - 1/PHI) < 1e-12)
check("Error d=8 ≈ 2.39e-06", abs(TOPOLOGICAL_ERROR_RATE_D8 - math.exp(-8*PHI)) < 1e-10)
check("Error d=13 ≈ 7.33e-10", abs(TOPOLOGICAL_ERROR_RATE_D13 - math.exp(-13*PHI)) < 1e-14)
check("Step 2^(1/104) ≈ 1.00669", abs(FUNDAMENTAL_STEP - 2**(1/104)) < 1e-12)
check("Semitone 2^(1/13) ≈ 1.05477", abs(SEMITONE_RATIO - 2**(8/104)) < 1e-12)
check("Dial bits a=3, b=4, c=3, d=4", (DIAL_BITS_A, DIAL_BITS_B, DIAL_BITS_C, DIAL_BITS_D) == (3,4,3,4))
check("Total dial bits = 14", DIAL_TOTAL_BITS == 14)
check("Dial configs = 16384", DIAL_CONFIGURATIONS == 16384)
check("Iron manifold = 26Q", IRON_MANIFOLD_QUBITS == 26)
check("Hilbert dim = 2^26", IRON_MANIFOLD_HILBERT_DIM == 2**26)
check("Ancilla = 12", ANCILLA_QUBITS == 12)
check("Fibonacci(7) = 13", FIBONACCI_7 == 13)
check("Factor-13: 286/13=22", FACTOR_13_286 == 22)
check("Factor-13: 104/13=8", FACTOR_13_104 == 8)
check("Factor-13: 416/13=32", FACTOR_13_416 == 32)
check("Octave steps = 104", OCTAVE_STEPS == 104)
check("Semitone steps = 8", SEMITONE_STEPS == 8)
check("4-octave = 16.0", abs(FOUR_OCTAVE_OFFSET - 16.0) < 1e-12)
check("Default depth = 8", TOPOLOGICAL_DEFAULT_DEPTH == 8)

print()
print("=" * 70)
print("TEST 2: l104_quantum_gate_engine/gates.py — GateAlgebra")
print("=" * 70)
from l104_quantum_gate_engine.gates import (
    GateAlgebra, GOD_CODE_PHASE, PHI_GATE, VOID_GATE, IRON_GATE,
    FIBONACCI_BRAID, ANYON_EXCHANGE,
)
algebra = GateAlgebra()

# Unitarity
for gate in [GOD_CODE_PHASE, PHI_GATE, VOID_GATE, IRON_GATE]:
    u = algebra.verify_unitarity(gate)
    check(f"U†U=I ({gate.name})", u["is_unitary"])

# Reversibility
for gate in [GOD_CODE_PHASE, PHI_GATE, VOID_GATE, IRON_GATE]:
    r = algebra.verify_reversibility(gate)
    check(f"Reversible ({gate.name})", r["is_reversible"])

# Bloch manifold
b = algebra.bloch_manifold_state(GOD_CODE_PHASE)
check("GOD_CODE Bloch pure state", b["is_pure_state"])
check("GOD_CODE Bloch |r|=1", abs(b["magnitude"] - 1.0) < 1e-10)

# Topological error rate
t8 = algebra.topological_error_rate(8)
check("Topo error d=8 ≈ 2.39e-6", abs(t8["error_rate"] - 2.389869e-06) < 1e-8)
t9 = algebra.topological_error_rate(9)
check("Topo error d=9 QEC ready", t9["qec_ready"])
t13 = algebra.topological_error_rate(13)
check("Topo error d=13 < 1e-9", t13["error_rate"] < 1e-9)

# Noise resilience
n = algebra.topological_noise_resilience(GOD_CODE_PHASE)
check("Noise analysis has 6 levels", len(n["noise_analysis"]) == 6)
check("Zero-noise purity ≈ 1.0", abs(n["noise_analysis"][0]["purity"] - 1.0) < 0.01)

# Composite order
co = algebra.composite_gate_order([GOD_CODE_PHASE, PHI_GATE, VOID_GATE, IRON_GATE])
check("Sacred composite infinite order", co["is_infinite_order"])

# Full sacred analysis
fa = algebra.full_sacred_analysis(GOD_CODE_PHASE)
check("Full analysis has all keys", all(k in fa for k in ["sacred_alignment", "unitarity", "reversibility", "bloch_manifold", "noise_resilience"]))

print()
print("=" * 70)
print("TEST 3: l104_simulator — TopologicalProtectionVerifier")
print("=" * 70)
from l104_simulator.algorithms import TopologicalProtectionVerifier
tpv = TopologicalProtectionVerifier()

r1 = tpv.verify_unitarity()
check("All sacred gates unitary", r1["all_unitary"])

r2 = tpv.verify_norm_preservation()
check("Norm preserved across 100 states", r2["norm_preserved"])

r3 = tpv.verify_non_dissipative_loop(depth=500)
check("Non-dissipative after 500 iterations", r3["non_dissipative"])

r4 = tpv.verify_topological_error_rate()
check("QEC threshold depth found", r4["qec_threshold_depth"] is not None)
check("QEC threshold = 9", r4["qec_threshold_depth"] == 9 if r4["qec_threshold_depth"] else False)

r5 = tpv.verify_bloch_manifold()
check("GOD_CODE Bloch pure state", r5["is_pure"])

r6 = tpv.verify_conservation_law()
check("Conservation law verified", r6["all_pass"])

r_all = tpv.verify_all()
check("Full verification PASS", r_all.success)
check("5/5 tests passed", r_all.details["tests_passed"] == 5)

print()
print("=" * 70)
print("TEST 4: l104_simulator — SacredEigenvalueSolver (enhanced)")
print("=" * 70)
from l104_simulator.algorithms import SacredEigenvalueSolver
ses = SacredEigenvalueSolver()
sr = ses.analyze()
check("Sacred Eigen success", sr.success)
check("Has topological_error_rate", "topological_error_rate" in sr.result)
check("Has bloch_magnitude", "bloch_magnitude" in sr.result)
check("Topo error rate in details", "topological_error_rate_d8" in sr.details)
check("Non-dissipative flag", sr.details.get("non_dissipative") == True)

print()
print("=" * 70)
print("TEST 5: l104_science_engine/constants.py")
print("=" * 70)
from l104_science_engine.constants import (
    TOPOLOGICAL_CORRELATION_LENGTH as SCI_TOPO_XI,
    TOPOLOGICAL_ERROR_RATE_D8 as SCI_TOPO_D8,
    TOPOLOGICAL_ERROR_RATE_D13 as SCI_TOPO_D13,
    FIBONACCI_BRAID_PHASE, FIBONACCI_F_MATRIX_ENTRY,
    UNITARY_PHASE_STEP, UNITARY_SEMITONE, UNITARY_FOUR_OCTAVE,
    FIBONACCI_7 as SCI_FIB7, FACTOR_13_SCAFFOLD, FACTOR_13_GRAIN, FACTOR_13_OFFSET,
    DIAL_BITS_A as SCI_DA, DIAL_TOTAL_QUBITS, DIAL_CONFIGURATIONS as SCI_DC,
    DIAL_ANCILLA_QUBITS, DIAL_IRON_MANIFOLD_DIM,
)
check("ξ = 1/φ", abs(SCI_TOPO_XI - 1/PHI) < 1e-12)
check("Topo error d=8", abs(SCI_TOPO_D8 - math.exp(-8*PHI)) < 1e-10)
check("Topo error d=13", abs(SCI_TOPO_D13 - math.exp(-13*PHI)) < 1e-14)
check("Fibonacci braid = 4π/5", abs(FIBONACCI_BRAID_PHASE - 4*math.pi/5) < 1e-12)
check("F-matrix entry = 1/φ", abs(FIBONACCI_F_MATRIX_ENTRY - 1/PHI) < 1e-12)
check("Unitary step ≈ 1.00669", abs(UNITARY_PHASE_STEP - 2**(1/104)) < 1e-12)
check("Semitone ≈ 1.05477", abs(UNITARY_SEMITONE - 2**(8/104)) < 1e-12)
check("4-octave = 16", abs(UNITARY_FOUR_OCTAVE - 16.0) < 1e-12)
check("F(7) = 13", SCI_FIB7 == 13)
check("286/13 = 22", FACTOR_13_SCAFFOLD == 22)
check("104/13 = 8", FACTOR_13_GRAIN == 8)
check("416/13 = 32", FACTOR_13_OFFSET == 32)
check("Dial a=3 bits", SCI_DA == 3)
check("Total dial = 14Q", DIAL_TOTAL_QUBITS == 14)
check("Dial configs = 16384", SCI_DC == 16384)
check("Ancilla = 12", DIAL_ANCILLA_QUBITS == 12)
check("Iron dim = 2^26", DIAL_IRON_MANIFOLD_DIM == 2**26)

print()
print("=" * 70)
print("TEST 6: l104_math_engine/constants.py")
print("=" * 70)
from l104_math_engine.constants import (
    TOPOLOGICAL_CORRELATION_LENGTH as MATH_TOPO_XI,
    TOPOLOGICAL_DEFAULT_DEPTH as MATH_TOPO_DEPTH,
    TOPOLOGICAL_ERROR_D8, TOPOLOGICAL_ERROR_D13,
    FIBONACCI_BRAID_PHASE as MATH_BRAID,
    UNITARY_PHASE_STEP as MATH_STEP,
    UNITARY_SEMITONE_RATIO, UNITARY_FOUR_OCTAVE as MATH_4OCT,
    DIAL_REGISTER_BITS, DIAL_CONFIGURATIONS as MATH_DC,
    IRON_MANIFOLD_QUBITS as MATH_IRON_Q, IRON_MANIFOLD_HILBERT_DIM as MATH_IRON_DIM,
)
check("ξ = 1/φ", abs(MATH_TOPO_XI - 1/PHI) < 1e-12)
check("Default depth = 8", MATH_TOPO_DEPTH == 8)
check("Topo error d=8", abs(TOPOLOGICAL_ERROR_D8 - math.exp(-8*PHI)) < 1e-10)
check("Topo error d=13", abs(TOPOLOGICAL_ERROR_D13 - math.exp(-13*PHI)) < 1e-14)
check("Braid = 4π/5", abs(MATH_BRAID - 4*math.pi/5) < 1e-12)
check("Step = 2^(1/104)", abs(MATH_STEP - 2**(1/104)) < 1e-12)
check("Semitone = 2^(1/13)", abs(UNITARY_SEMITONE_RATIO - 2**(8/104)) < 1e-12)
check("4-octave = 16", abs(MATH_4OCT - 16.0) < 1e-12)
check("Dial bits = 14", DIAL_REGISTER_BITS == 14)
check("Dial configs = 16384", MATH_DC == 16384)
check("Iron Q = 26", MATH_IRON_Q == 26)
check("Iron dim = 2^26", MATH_IRON_DIM == 2**26)

print()
print("=" * 70)
print("TEST 7: l104_math_engine/god_code.py — Topological Methods")
print("=" * 70)
from l104_math_engine.god_code import GodCodeEquation

# Phase operator
po = GodCodeEquation.phase_operator(0, 0, 0, 0)
check("Phase op (0,0,0,0) = 16.0", abs(po - 16.0) < 1e-12)
po1 = GodCodeEquation.phase_operator(1, 0, 0, 0)
check("Phase op (1,0,0,0) > 16", po1 > 16)

# Norm preservation
np_result = GodCodeEquation.verify_norm_preservation(1, 2, 1, 1)
check("Phase norm = 1 (unit circle)", np_result["norm_preserved"])
check("Reversible (e^{iθ}e^{-iθ}=1)", np_result["reversible"])
check("Non-dissipative", np_result["non_dissipative"])

# Topological error rate
te = GodCodeEquation.topological_error_rate(8)
check("Topo error d=8 ≈ 2.39e-6", abs(te["error_rate"] - 2.389869e-06) < 1e-8)
te9 = GodCodeEquation.topological_error_rate(9)
check("Topo error d=9 QEC ready", te9["qec_ready"])
te13 = GodCodeEquation.topological_error_rate(13)
check("Topo error d=13 < 1e-9", te13["error_rate"] < 1e-9)

# Bloch manifold
bm = GodCodeEquation.bloch_manifold_mapping(0, 0, 0, 0)
check("Bloch pure state", bm["is_pure"])
check("GOD_CODE value correct", abs(bm["god_code_value"] - 527.5184818492612) < 1e-6)

# Dial register info
dr = GodCodeEquation.dial_register_info()
check("14-qubit dial register", dr["total_dial_qubits"] == 14)
check("16384 configurations", dr["total_configurations"] == 16384)
check("26Q iron manifold", dr["iron_manifold_qubits"] == 26)

print()
print("=" * 70)
print("TEST 8: Existing tests still pass")
print("=" * 70)
# Run existing god_code test suite
g0 = GodCodeEquation.evaluate(0, 0, 0, 0)
check("G(0,0,0,0) = 527.518...", abs(g0 - 527.5184818492612) < 1e-6)
check("Conservation law X=0", GodCodeEquation.verify_conservation(0))
check("Conservation law X=104", GodCodeEquation.verify_conservation(104))
check("Conservation law X=416", GodCodeEquation.verify_conservation(416))
check("Conservation law X=-104", GodCodeEquation.verify_conservation(-104))
props = GodCodeEquation.equation_properties()
check("Properties has factor_13", "factor_13" in props)

print()
print("═" * 70)
print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
print("═" * 70)

sys.exit(1 if FAIL > 0 else 0)
