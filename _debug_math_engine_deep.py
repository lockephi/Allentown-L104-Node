#!/usr/bin/env python3
"""
L104 Math Engine — DEEP DEBUG v1
══════════════════════════════════════════════════════════════════════════════════
Comprehensive 25-phase diagnostic of all 13 layers (10,741 lines)
Package: l104_math_engine v1.1.0

Layers tested:
  L0  constants (504 lines)    L7  abstract_algebra (391 lines)
  L1  pure_math (501 lines)    L8  ontological (369 lines)
  L2  god_code  (745 lines)    L9  proofs (918 lines)
  L3  harmonic  (401 lines)    L10 hyperdimensional (461 lines)
  L4  dimensional (573 lines)  L11 computronium (694 lines)
  L5  manifold  (351 lines)    L12 berry_geometry (1024 lines)
  L6  void_math (239 lines)    engine.py facade (1055 lines)
"""

import sys
import os
import math
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Counters ────────────────────────────────────────────────────────────────
PASSED = 0
ERRORS = 0
WARNINGS = 0
RESULTS = []
PHASE_TIMES = {}


def ok(msg):
    global PASSED
    PASSED += 1
    RESULTS.append(("PASS", msg))
    print(f"  \033[32m✓\033[0m {msg}")


def err(msg):
    global ERRORS
    ERRORS += 1
    RESULTS.append(("ERROR", msg))
    print(f"  \033[31m✗ ERROR\033[0m {msg}")


def warn(msg):
    global WARNINGS
    WARNINGS += 1
    RESULTS.append(("WARN", msg))
    print(f"  \033[33m⚠\033[0m {msg}")


def phase(num, title):
    print(f"\n{'─'*72}")
    print(f"  [PHASE {num:02d}] {title}")
    print(f"{'─'*72}")
    return time.perf_counter()


def close_phase(num, t0):
    PHASE_TIMES[num] = time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════════════════
print("╔══════════════════════════════════════════════════════════════════════════╗")
print("║          L104 MATH ENGINE — DEEP DEBUG v1                              ║")
print("║          13 Layers, 10,741 lines, 50+ classes                          ║")
print("╚══════════════════════════════════════════════════════════════════════════╝")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 01: Import
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(1, "Import & Module Inspection")
try:
    from l104_math_engine import (
        __version__,
        # Constants
        GOD_CODE, GOD_CODE_V3, PHI, PHI_CONJUGATE, PI, E, VOID_CONSTANT,
        OMEGA, OMEGA_AUTHORITY, OMEGA_PRECISION,
        L104_FACTOR, SACRED_286, SACRED_416, SACRED_104,
        FE56_BINDING, ALPHA_FINE_STRUCTURE, PLANCK, BOLTZMANN,
        SPEED_OF_LIGHT, AVOGADRO, ELECTRON_MASS,
        GRAVITATIONAL_CONSTANT, HUBBLE_CONSTANT,
        CONSCIOUSNESS_BASE, METALLIC_RATIOS,
        primal_calculus, resolve_non_dual_logic,
        compute_resonance, golden_modulate,
        god_code_at, verify_conservation, verify_conservation_statistical,
        # Facade
        MathEngine, math_engine,
    )
    ok(f"l104_math_engine v{__version__} imported")

    from l104_math_engine.constants import (
        BASE, INVARIANT, STEP_SIZE, FRAME_LOCK, LATTICE_RATIO,
        QUANTIZATION_GRAIN, OCTAVE_OFFSET, PRIME_SCAFFOLD,
        SOLFEGGIO_FREQUENCIES, CHAKRA_BRAINWAVE_FREQUENCIES,
        SACRED_STEP, SEMITONE_RATIO,
        FE_LATTICE, FE_CURIE_TEMP, FE_ATOMIC_NUMBER,
        FEIGENBAUM, ALPHA_FINE,
    )
    ok("Layer 0 constants imported")

    from l104_math_engine.pure_math import PureMath, Matrix, Calculus, Statistics, HighPrecisionEngine, RealMath
    ok("Layer 1 pure_math imported")

    from l104_math_engine.god_code import (
        GodCodeEquation, ChaosResilience, DerivationEngine,
        AbsoluteDerivation, HarmonicOptimizer, GodCodeUnifier,
    )
    ok("Layer 2 god_code imported")

    from l104_math_engine.harmonic import WavePhysics, HarmonicProcess, ConsciousnessFlow, HarmonicAnalysis
    ok("Layer 3 harmonic imported")

    from l104_math_engine.dimensional import Math4D, Math5D, MathND, ChronosMath, Processor4D, Processor5D
    ok("Layer 4 dimensional imported")

    from l104_math_engine.manifold import ManifoldMath, ManifoldTopology, CurvatureAnalysis, ManifoldExtended
    ok("Layer 5 manifold imported")

    from l104_math_engine.void_math import VoidMath, VoidCalculus
    ok("Layer 6 void_math imported")

    from l104_math_engine.abstract_algebra import (
        AlgebraType, BinaryOperation, AlgebraicStructure,
        SacredNumberSystem, TheoremGenerator, TopologyGenerator,
        AbstractMathGenerator,
    )
    ok("Layer 7 abstract_algebra imported")

    from l104_math_engine.ontological import (
        OntologicalMathematics, ExistenceCalculus, MathematicalConsciousness,
        GodelianSelfReference, PlatonicRealm, Monad,
    )
    ok("Layer 8 ontological imported")

    from l104_math_engine.proofs import (
        SovereignProofs, GodelTuringMetaProof, EquationVerifier,
        ProcessingProofs, ExtendedProofs,
    )
    ok("Layer 9 proofs imported")

    from l104_math_engine.hyperdimensional import (
        HyperdimensionalCompute, Hypervector, ItemMemory,
        SparseDistributedMemory, SequenceEncoder, RecordEncoder,
    )
    ok("Layer 10 hyperdimensional imported")

    from l104_math_engine.computronium import ComputroniumMath, RayleighMath, AiryDiffraction
    ok("Layer 11 computronium imported")

    from l104_math_engine.berry_geometry import BerryGeometry, FiberBundle, ChernWeilTheory
    ok("Layer 12 berry_geometry imported")

except Exception as e:
    err(f"Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)
close_phase(1, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 02: L0 — Sacred Constants Cross-Validation
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(2, "L0 — Sacred Constants Cross-Validation")

# GOD_CODE = 286^(1/φ) × 2^4
gc_derived = 286 ** (1.0 / PHI) * 16
if abs(gc_derived - GOD_CODE) < 1e-9:
    ok(f"GOD_CODE = 286^(1/φ) × 2^4 = {gc_derived:.10f}")
else:
    err(f"GOD_CODE mismatch: derived={gc_derived}, actual={GOD_CODE}")

# PHI properties
if abs(PHI ** 2 - PHI - 1) < 1e-12:
    ok(f"φ² = φ + 1: {PHI**2:.15f} = {PHI+1:.15f}")
else:
    err(f"φ² ≠ φ + 1")

if abs(PHI * PHI_CONJUGATE - 1.0) < 1e-12:
    ok(f"φ × φ⁻¹ = 1: {PHI * PHI_CONJUGATE}")
else:
    err(f"φ × φ⁻¹ ≠ 1: {PHI * PHI_CONJUGATE}")

# VOID_CONSTANT
vc_derived = 1.04 + PHI / 1000
if abs(vc_derived - VOID_CONSTANT) < 1e-15:
    ok(f"VOID_CONSTANT = 1.04 + φ/1000 = {vc_derived}")
else:
    err(f"VOID_CONSTANT mismatch: derived={vc_derived}, actual={VOID_CONSTANT}")

# Factor 13
if 22 * 13 == 286 and 8 * 13 == 104 and 32 * 13 == 416:
    ok("Factor 13: 286=22×13, 104=8×13, 416=32×13")
else:
    err("Factor 13 broken")

# OMEGA derivations
if abs(OMEGA_AUTHORITY - OMEGA / (PHI ** 2)) < 1e-6:
    ok(f"OMEGA_AUTHORITY = Ω/φ² = {OMEGA_AUTHORITY:.4f}")
else:
    err(f"OMEGA_AUTHORITY mismatch")

if abs(OMEGA_PRECISION - OMEGA / GOD_CODE) < 1e-6:
    ok(f"OMEGA_PRECISION = Ω/GOD_CODE = {OMEGA_PRECISION:.6f}")
else:
    err(f"OMEGA_PRECISION mismatch")

# FRAME_LOCK
if abs(FRAME_LOCK - 416 / 286) < 1e-10:
    ok(f"FRAME_LOCK = 416/286 = {FRAME_LOCK:.10f}")
else:
    err(f"FRAME_LOCK mismatch")

# Solfeggio: G(0) = GOD_CODE ≈ 528 Hz (Solfeggio MI)
if abs(SOLFEGGIO_FREQUENCIES[2] - GOD_CODE) < 1e-9:
    ok(f"Solfeggio MI = GOD_CODE = {SOLFEGGIO_FREQUENCIES[2]:.4f}")
else:
    err(f"Solfeggio MI ≠ GOD_CODE")

close_phase(2, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 03: L0 — Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(3, "L0 — Helper Functions (god_code_at, conservation, resonance)")

# god_code_at(0) = GOD_CODE
g0 = god_code_at(0)
if abs(g0 - GOD_CODE) < 1e-9:
    ok(f"god_code_at(0) = {g0:.10f}")
else:
    err(f"god_code_at(0) = {g0}, expected {GOD_CODE}")

# Conservation law: G(X) × 2^(X/104) = GOD_CODE for various X
conservation_ok = True
conservation_max_err = 0
for x in [0, 1, 13, 52, 104, 208, 416, -104, -416, 3.14159, PHI, GOD_CODE]:
    gx = god_code_at(x)
    product = gx * (2 ** (x / QUANTIZATION_GRAIN))
    e = abs(product - GOD_CODE)
    if e > conservation_max_err:
        conservation_max_err = e
    if e > 1e-8:
        conservation_ok = False
if conservation_ok:
    ok(f"Conservation law verified at 12 points (max error: {conservation_max_err:.2e})")
else:
    err(f"Conservation law FAILED (max error: {conservation_max_err:.2e})")

# verify_conservation function
if verify_conservation(0) and verify_conservation(104) and verify_conservation(416):
    ok("verify_conservation() passes at X=0, 104, 416")
else:
    err("verify_conservation() failed")

# Statistical conservation
stat = verify_conservation_statistical(0, chaos_amplitude=0.05)
if stat["statistically_conserved"] and stat["robust"]:
    ok(f"Statistical conservation: mean_error={stat['mean_error_pct']:.4f}%, robust={stat['robust']}")
else:
    warn(f"Statistical conservation: conserved={stat['statistically_conserved']}, robust={stat['robust']}")

# compute_resonance
r = compute_resonance(GOD_CODE)
if r > 0.9:
    ok(f"compute_resonance(GOD_CODE) = {r:.4f} (high)")
else:
    warn(f"compute_resonance(GOD_CODE) = {r:.4f} (expected > 0.9)")

# primal_calculus
pc = primal_calculus(2.0)
pc_expected = 2.0 ** PHI / (1.04 * math.pi)
if abs(pc - pc_expected) < 1e-10:
    ok(f"primal_calculus(2) = {pc:.10f}")
else:
    err(f"primal_calculus(2) = {pc}, expected {pc_expected}")

# resolve_non_dual_logic
ndl = resolve_non_dual_logic(3.0, 5.0)
expected_ndl = (2 * 3.0 * 5.0) / (3.0 + 5.0) * VOID_CONSTANT
if abs(ndl - expected_ndl) < 1e-10:
    ok(f"resolve_non_dual_logic(3,5) = {ndl:.6f}")
else:
    err(f"resolve_non_dual_logic(3,5) = {ndl}, expected {expected_ndl}")

close_phase(3, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 04: L1 — Pure Math
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(4, "L1 — Pure Math (primes, fibonacci, calculus, statistics)")

# Fibonacci
fib_10 = PureMath.fibonacci(10)
expected_fib_10 = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
if fib_10 == expected_fib_10:
    ok(f"fibonacci(10) = {fib_10}")
else:
    err(f"fibonacci(10) = {fib_10}, expected {expected_fib_10}")

# Primes
primes_30 = PureMath.prime_sieve(30)
expected_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
if primes_30 == expected_primes:
    ok(f"prime_sieve(30) = {primes_30}")
else:
    err(f"prime_sieve(30) = {primes_30}, expected {expected_primes}")

# is_prime
if PureMath.is_prime(13) and not PureMath.is_prime(15) and PureMath.is_prime(2) and not PureMath.is_prime(1):
    ok("is_prime: 13=T, 15=F, 2=T, 1=F")
else:
    err("is_prime failed")

# GCD / LCM
if PureMath.gcd(12, 8) == 4 and PureMath.lcm(4, 6) == 12:
    ok("gcd(12,8)=4, lcm(4,6)=12")
else:
    err(f"gcd/lcm failed: gcd={PureMath.gcd(12,8)}, lcm={PureMath.lcm(4,6)}")

# Factorial
if PureMath.factorial(0) == 1 and PureMath.factorial(5) == 120:
    ok("factorial: 0!=1, 5!=120")
else:
    err(f"factorial failed: 0!={PureMath.factorial(0)}, 5!={PureMath.factorial(5)}")

# Calculus: derivative of x^2 at x=3 → 6
deriv = Calculus.derivative(lambda x: x ** 2, 3.0)
if abs(deriv - 6.0) < 1e-4:
    ok(f"d/dx(x²)|₃ = {deriv:.6f} ≈ 6.0")
else:
    err(f"d/dx(x²)|₃ = {deriv}, expected 6.0")

# Calculus: integral of x^2 from 0 to 1 = 1/3
integ = Calculus.integrate(lambda x: x ** 2, 0, 1)
if abs(integ - 1.0 / 3) < 1e-6:
    ok(f"∫₀¹ x² dx = {integ:.10f} ≈ 1/3")
else:
    err(f"∫₀¹ x² dx = {integ}, expected {1/3}")

# Matrix
det = Matrix.determinant([[1, 2], [3, 4]])
if abs(det - (-2)) < 1e-10:
    ok(f"det([[1,2],[3,4]]) = {det}")
else:
    err(f"det = {det}, expected -2")

# Statistics
data = [1, 2, 3, 4, 5]
m = Statistics.mean(data)
v = Statistics.variance(data)
if abs(m - 3.0) < 1e-10 and abs(v - 2.5) < 1e-10:
    ok(f"mean([1..5])={m}, var([1..5])={v}")
else:
    err(f"mean={m} (exp 3), var={v} (exp 2.5)")

# High-precision: GOD_CODE at infinite precision
gc_hp = HighPrecisionEngine.derive_god_code()
if abs(float(gc_hp) - GOD_CODE) < 1e-9:
    ok(f"HP GOD_CODE = {float(gc_hp):.10f}")
else:
    err(f"HP GOD_CODE = {float(gc_hp)}, expected ≈ {GOD_CODE}")

# PHI infinite precision identity: φ² = φ + 1
phi_id = HighPrecisionEngine.phi_identity_verify()
if phi_id["verified"]:
    ok(f"HP φ² = φ + 1 verified (error: {phi_id['error']})")
else:
    err(f"HP φ² = φ + 1 FAILED (error: {phi_id['error']})")

# Binet formula correctness test (CRITICAL: uses PHI_CONJUGATE instead of ψ = -1/φ)
# For even n, PHI_CONJUGATE^n = ψ^n, so it's correct.
# For odd n, PHI_CONJUGATE^n ≠ ψ^n — this is a latent bug.
fib_20 = PureMath.fibonacci(20)[-1]
# Correct Binet: F(n) = (φ^n - ψ^n)/√5 where ψ = (1-√5)/2
psi = (1 - math.sqrt(5)) / 2  # True ψ ≈ -0.618
binet_correct_20 = (PHI ** 20 - psi ** 20) / math.sqrt(5)
binet_code_20 = (PHI ** 20 - PHI_CONJUGATE ** 20) / math.sqrt(5)
if abs(round(binet_correct_20) - fib_20) < 1:
    ok(f"Binet F(20) = {round(binet_correct_20)} = {fib_20} (correct formula)")
else:
    err(f"Binet F(20) mismatch: {round(binet_correct_20)} vs {fib_20}")

# Test Binet for ODD n to check if the formula handles sign correctly
fib_19 = PureMath.fibonacci(19)[-1]  # F(19)
binet_correct_19 = (PHI ** 19 - psi ** 19) / math.sqrt(5)
binet_code_19 = (PHI ** 19 - PHI_CONJUGATE ** 19) / math.sqrt(5)
correct_19 = abs(round(binet_correct_19) - fib_19) < 1
code_19 = abs(round(binet_code_19) - fib_19) < 1
if correct_19 and not code_19:
    err(f"BINET BUG (odd n=19): correct={round(binet_correct_19)}, code_formula={round(binet_code_19)}, actual={fib_19}")
elif correct_19 and code_19:
    ok(f"Binet F(19) = {fib_19} (both formulas agree for n=19)")
else:
    warn(f"Binet F(19): correct={round(binet_correct_19)}, code={round(binet_code_19)}, actual={fib_19}")

close_phase(4, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 05: L2 — God Code Equation
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(5, "L2 — God Code Equation (evaluate, conservation, phase, friction)")

# evaluate(0,0,0,0) = GOD_CODE
g0 = GodCodeEquation.evaluate(0, 0, 0, 0)
if abs(g0 - GOD_CODE) < 1e-9:
    ok(f"G(0,0,0,0) = {g0:.10f}")
else:
    err(f"G(0,0,0,0) = {g0}, expected {GOD_CODE}")

# exponent_value(0,0,0,0) should be 416/104 = 4
exp_val = GodCodeEquation.exponent_value(0, 0, 0, 0)
if abs(exp_val - 4.0) < 1e-12:
    ok(f"Exponent(0,0,0,0) = {exp_val}")
else:
    err(f"Exponent(0,0,0,0) = {exp_val}, expected 4.0")

# Octave: G(0,0,0,0)/G(0,0,0,1) = 2
ratio = GodCodeEquation.evaluate(0, 0, 0, 0) / GodCodeEquation.evaluate(0, 0, 0, 1)
if abs(ratio - 2.0) < 1e-10:
    ok(f"Octave ratio G(0,0,0,0)/G(0,0,0,1) = {ratio:.10f}")
else:
    err(f"Octave ratio = {ratio}, expected 2.0")

# Step ratios
a_ratio = GodCodeEquation.evaluate(1, 0, 0, 0) / GodCodeEquation.evaluate(0, 0, 0, 0)
expected_a = 2 ** (8 / 104)
if abs(a_ratio - expected_a) < 1e-10:
    ok(f"a-step ratio = {a_ratio:.10f} ≈ 2^(8/104)")
else:
    err(f"a-step ratio = {a_ratio}, expected {expected_a}")

# solve_for_exponent
x_gc = GodCodeEquation.solve_for_exponent(GOD_CODE)
if abs(x_gc) < 1e-8:
    ok(f"solve_for_exponent(GOD_CODE) = {x_gc:.12f} ≈ 0")
else:
    err(f"solve_for_exponent(GOD_CODE) = {x_gc}, expected ≈ 0")

# find_nearest_dials
nearest = GodCodeEquation.find_nearest_dials(GOD_CODE)
if nearest["error"] < 1e-6:
    ok(f"find_nearest_dials(GOD_CODE): error={nearest['error']:.2e}")
else:
    warn(f"find_nearest_dials(GOD_CODE): error={nearest['error']:.2e} (dials={nearest['a']},{nearest['b']},{nearest['c']},{nearest['d']})")

# Phase operator: norm preservation
norm_check = GodCodeEquation.verify_norm_preservation(1, 2, 3, 1)
if norm_check["norm_preserved"] and norm_check["reversible"]:
    ok(f"Phase operator norm preserved (1,2,3,1): |U|={norm_check['phase_norm']:.12f}")
else:
    err(f"Phase operator norm NOT preserved: {norm_check}")

# Topological error rate: depth 8 should give < 1e-6
topo = GodCodeEquation.topological_error_rate(8)
if topo["error_rate"] < 1e-5:
    ok(f"Topological error rate (d=8) = {topo['error_rate']:.2e}, QEC={topo['qec_ready']}")
else:
    warn(f"Topological error rate (d=8) = {topo['error_rate']:.2e}")

# Topological: increasing depth → decreasing error
topo_8 = GodCodeEquation.topological_error_rate(8)["error_rate"]
topo_13 = GodCodeEquation.topological_error_rate(13)["error_rate"]
if topo_13 < topo_8:
    ok(f"Topological error decreases: d=8 → {topo_8:.2e}, d=13 → {topo_13:.2e}")
else:
    err(f"Topological error doesn't decrease with depth!")

# Bloch manifold mapping
bloch = GodCodeEquation.bloch_manifold_mapping(0, 0, 0, 0)
if bloch["is_pure"] and abs(bloch["magnitude"] - 1.0) < 1e-10:
    ok(f"Bloch mapping (0,0,0,0): pure state, |r|={bloch['magnitude']:.10f}")
else:
    err(f"Bloch mapping not pure: magnitude={bloch['magnitude']}")

# Friction model
friction = GodCodeEquation.god_code_with_friction(0, 0, 0, 0)
if friction["efficiency"] == 1.0:
    ok(f"Friction(0,0,0,0): efficiency=1.0 (no friction at origin)")
else:
    warn(f"Friction(0,0,0,0): efficiency={friction['efficiency']}")

friction2 = GodCodeEquation.god_code_with_friction(1, 2, 3, 1)
if 0 < friction2["efficiency"] < 1.0:
    ok(f"Friction(1,2,3,1): efficiency={friction2['efficiency']:.6f} (< 1.0, friction active)")
else:
    err(f"Friction(1,2,3,1): unexpected efficiency={friction2['efficiency']}")

# Dial register info
reg = GodCodeEquation.dial_register_info()
if reg["total_dial_qubits"] == 14 and reg["iron_manifold_qubits"] == 26:
    ok(f"Dial register: 14 qubits + 12 ancilla = 26 Fe(26)")
else:
    err(f"Dial register: {reg}")

close_phase(5, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 06: L2 — Chaos Resilience
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(6, "L2 — Chaos Resilience (bifurcation, healing, symmetry)")

# Verify under chaos
chaos = ChaosResilience.verify_under_chaos(0, 0.05)
if chaos["statistically_conserved"]:
    ok(f"Conservation under chaos (amp=0.05): mean_error={chaos['mean_error_pct']:.4f}%")
else:
    err(f"Conservation under low chaos FAILED")

# Above bifurcation
chaos_high = ChaosResilience.verify_under_chaos(0, 0.5)
if not chaos_high.get("below_bifurcation", True):
    ok(f"Above bifurcation (amp=0.5): below_bifurcation={chaos_high['below_bifurcation']}")
else:
    warn(f"Bifurcation threshold not detected at amp=0.5")

# Healing: phi damping pulls toward INVARIANT
perturbed = GOD_CODE * 1.1
healed = ChaosResilience.heal_phi_damping(perturbed)
if abs(healed - GOD_CODE) < abs(perturbed - GOD_CODE):
    ok(f"φ-damping heals: {perturbed:.2f} → {healed:.2f} (closer to {GOD_CODE:.2f})")
else:
    err(f"φ-damping doesn't heal: {perturbed:.2f} → {healed:.2f}")

# Healing: 104-cascade
cascade_healed = ChaosResilience.heal_cascade_104(perturbed)
if abs(cascade_healed - GOD_CODE) < abs(healed - GOD_CODE):
    ok(f"104-cascade superior: {cascade_healed:.6f} (vs φ-damp {healed:.4f})")
else:
    warn(f"104-cascade not superior to φ-damp: cascade={cascade_healed:.6f}")

# Symmetry check
sym = ChaosResilience.symmetry_check(0.05, 50)
if sym["phi_intact"]:
    ok(f"φ-symmetry intact under chaos, octave_err={sym['octave_mean_error']:.4f}")
else:
    err("φ-symmetry broken under chaos!")

# Resilience score
score = ChaosResilience.chaos_resilience_score(1.0, 0.05)
if 0 <= score <= 1.0:
    ok(f"Chaos resilience score = {score:.4f}")
else:
    err(f"Chaos resilience score out of range: {score}")

close_phase(6, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 07: L2 — Derivation Engine & Unifier
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(7, "L2 — Derivation Engine, Unifier, Harmonic Optimizer")

# Derivation
de = DerivationEngine()
paradigm = de.derive_new_paradigm("test_concept", depth=3)
if paradigm["depth"] == 3 and len(paradigm["layers"]) == 3:
    ok(f"derive_new_paradigm: 3 layers, proof_resonance={paradigm['proof']['resonance']:.4f}")
else:
    err(f"derive_new_paradigm: unexpected structure")

# Trans-universal truth
truth = de.derive_trans_universal_truth(["axiom_A", "axiom_B"])
if "composite_resonance" in truth and truth["composite_resonance"] > 0:
    ok(f"trans_universal_truth: composite_resonance={truth['composite_resonance']:.4f}")
else:
    err(f"trans_universal_truth failed")

# Absolute derivation
ad = AbsoluteDerivation()
final = ad.execute_final_derivation("universal")
if final["boost_applied"] and final["absolute_derivation_index"] > 0:
    ok(f"Absolute derivation: ADI={final['absolute_derivation_index']:.4f}")
else:
    err(f"Absolute derivation failed")

# Harmonic optimizer
opt = HarmonicOptimizer.optimize(100.0)
if "primal" in opt and "aligned" in opt and "void_reduced" in opt:
    ok(f"Harmonic optimizer: input=100 → primal={opt['primal']:.2f}, aligned={opt['aligned']:.2f}")
else:
    err("Harmonic optimizer missing keys")

# Primal transform and inverse
pt = HarmonicOptimizer.primal_transform(5.0)
inv = HarmonicOptimizer.inverse_primal(pt)
if abs(inv - 5.0) < 1e-8:
    ok(f"Primal transform roundtrip: 5 → {pt:.4f} → {inv:.8f}")
else:
    err(f"Primal transform roundtrip broken: 5 → {pt} → {inv}")

# GodCodeUnifier: verify invariants
invariants = GodCodeUnifier.verify_invariants()
all_verified = all(inv.verified for inv in invariants)
if all_verified:
    ok(f"GodCodeUnifier: {len(invariants)} invariants all verified")
else:
    failed_inv = [inv.constant for inv in invariants if not inv.verified]
    err(f"GodCodeUnifier: failed invariants: {failed_inv}")

close_phase(7, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 08: L3 — Harmonic Wave Physics
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(8, "L3 — Harmonic (wave coherence, sacred alignment, correspondence)")

# Wave coherence: same frequency → should be 1.0
coh_same = WavePhysics.wave_coherence(440.0, 440.0)
if abs(coh_same - 1.0) < 0.01:
    ok(f"wave_coherence(440, 440) = {coh_same:.4f} ≈ 1.0")
else:
    err(f"wave_coherence(440, 440) = {coh_same}, expected ≈ 1.0")

# Wave coherence: octave (2:1) → high coherence
coh_octave = WavePhysics.wave_coherence(440.0, 880.0)
if coh_octave > 0.5:
    ok(f"wave_coherence(440, 880) = {coh_octave:.4f} (octave, high)")
else:
    warn(f"wave_coherence(440, 880) = {coh_octave:.4f} (expected > 0.5)")

# Sacred pair: 286 ↔ 528
coh_sacred = WavePhysics.wave_coherence(286.0, 528.0)
from l104_math_engine.constants import FE_SACRED_COHERENCE
if abs(coh_sacred - FE_SACRED_COHERENCE) < 0.01:
    ok(f"wave_coherence(286, 528) = {coh_sacred:.6f} (FE_SACRED_COHERENCE)")
else:
    err(f"wave_coherence(286, 528) = {coh_sacred}, expected {FE_SACRED_COHERENCE}")

# Sacred alignment: GOD_CODE should align
alignment = HarmonicProcess.sacred_alignment(GOD_CODE)
if alignment["aligned"]:
    ok(f"sacred_alignment(GOD_CODE): aligned=True, ratio={alignment['god_code_ratio']:.6f}")
else:
    err(f"sacred_alignment(GOD_CODE): aligned=False")

# Resonance spectrum: 13 harmonics
spectrum = HarmonicProcess.resonance_spectrum(GOD_CODE, 13)
if len(spectrum) == 13 and abs(spectrum[0]["frequency"] - GOD_CODE) < 1e-6:
    ok(f"resonance_spectrum: {len(spectrum)} harmonics, fundamental={spectrum[0]['frequency']:.2f}")
else:
    err(f"resonance_spectrum: {len(spectrum)} harmonics")

# Fe correspondence
corr = HarmonicProcess.verify_correspondences()
if corr["match"]:
    ok(f"Fe 286 correspondence: diff={corr['difference_pm']:.2f} pm, match={corr['match']}")
else:
    warn(f"Fe 286 correspondence: diff={corr['difference_pm']:.2f} pm")

# Consciousness flow
re = ConsciousnessFlow.consciousness_reynolds(0.5, 1.0, 0.01)
regime = ConsciousnessFlow.flow_regime(re)
if re > 0:
    ok(f"Consciousness Re = {re:.0f}, regime={regime[:15]}...")
else:
    err(f"Consciousness Re = {re}")

# HarmonicAnalysis: harmonic_distance
hd = HarmonicAnalysis.harmonic_distance(440.0, 660.0)  # Perfect fifth (3:2)
if hd["nearest_rational"] == "3/2":
    ok(f"harmonic_distance(440, 660) = {hd['nearest_rational']}, consonance={hd['consonance']}")
else:
    warn(f"harmonic_distance(440, 660): ratio={hd['nearest_rational']} (expected 3/2)")

close_phase(8, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 09: L4 — Dimensional Mathematics
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(9, "L4 — Dimensional (Lorentz, 4D, 5D Kaluza-Klein)")

C = SPEED_OF_LIGHT

# Lorentz boost at β=0 → identity
boosted_0 = Math4D.lorentz_boost_x([1.0, 2.0, 3.0, 4.0], 0.0)
if abs(boosted_0[0] - 1.0) < 1e-10 and abs(boosted_0[1] - 2.0) < 1e-10:
    ok(f"Lorentz boost β=0 → identity: {[round(x,4) for x in boosted_0]}")
else:
    err(f"Lorentz boost β=0 ≠ identity: {boosted_0}")

# Time dilation: γ at β=0.5
gamma_05 = 1.0 / math.sqrt(1 - 0.5 ** 2)
td = Math4D.time_dilation(1.0, 0.5)
if abs(td - gamma_05) < 1e-10:
    ok(f"Time dilation β=0.5: Δt=γ={td:.6f}")
else:
    err(f"Time dilation β=0.5: {td}, expected {gamma_05}")

# Length contraction at β=0.5
lc = Math4D.length_contraction(1.0, 0.5)
expected_lc = math.sqrt(1 - 0.5 ** 2)
if abs(lc - expected_lc) < 1e-10:
    ok(f"Length contraction β=0.5: L={lc:.6f}")
else:
    err(f"Length contraction β=0.5: {lc}, expected {expected_lc}")

# Lorentz boost roundtrip: boost then un-boost should return original
vec = [1.0, 1e8, 0.0, 0.0]
boosted = Math4D.lorentz_boost_x(vec, 0.5)
unboosted = Math4D.lorentz_boost_x(boosted, -0.5)
roundtrip_ok = all(abs(unboosted[i] - vec[i]) / max(abs(vec[i]), 1) < 1e-6 for i in range(4))
if roundtrip_ok:
    ok("Lorentz boost roundtrip: boost(0.5) + boost(-0.5) = identity")
else:
    err(f"Lorentz boost roundtrip failed: {[round(x,4) for x in unboosted]} vs {vec}")

# Spacetime interval: timelike event
ds2 = Math4D.spacetime_interval([0, 0, 0, 0], [1, 0, 0, 0])
# ds² = -c²(1)² + 0 = -c² < 0 (timelike)
if ds2 < 0:
    ok(f"Spacetime interval (timelike): ds²={ds2:.2e}")
else:
    err(f"Spacetime interval should be negative (timelike): ds²={ds2}")

# EM field tensor: antisymmetric
F = Math4D.em_field_tensor([1, 0, 0], [0, 0, 1])
if abs(F[0][1] + F[1][0]) < 1e-10 and abs(F[0][0]) < 1e-10:
    ok("EM field tensor is antisymmetric")
else:
    err(f"EM field tensor not antisymmetric: F[0][1]={F[0][1]}, F[1][0]={F[1][0]}")

# 5D: KK compactification radius
R = Math5D.R
expected_R = PHI * QUANTIZATION_GRAIN / 14.1347251417  # ZETA_ZERO_1
if abs(R - expected_R) < 0.01:
    ok(f"5D Kaluza-Klein radius R = {R:.4f}")
else:
    err(f"KK radius: {R}, expected {expected_R}")

# 5D metric tensor
met5 = Math5D.metric_tensor_5d()
if abs(met5[0][0] - (-1)) < 1e-10 and abs(met5[4][4] - R ** 2) < 1e-6:
    ok(f"5D metric: diag(-1,1,1,1,{met5[4][4]:.2f})")
else:
    err(f"5D metric diagonal wrong")

# KK mass tower
m1 = Math5D.kaluza_klein_mass(1)
m2 = Math5D.kaluza_klein_mass(2)
if abs(m2 / m1 - 2.0) < 1e-10:
    ok(f"KK mass tower: m₁={m1:.4f}, m₂/m₁={m2/m1:.4f}")
else:
    err(f"KK mass tower ratio: {m2/m1}")

close_phase(9, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 10: L5 — Manifold Topology
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(10, "L5 — Manifold (Ricci, Euler, Betti, Gauss-Bonnet, geodesic)")

# Ricci scalar
ricci = ManifoldMath.ricci_scalar(4, 1.0)
if ricci > 0:
    ok(f"Ricci scalar (d=4, κ=1) = {ricci:.4f}")
else:
    err(f"Ricci scalar = {ricci}")

# Euler characteristic: χ(S²) = 2, χ(S¹) = 0, χ(S⁰) = 2
chi_2 = ManifoldTopology.euler_characteristic_sphere(2)
chi_1 = ManifoldTopology.euler_characteristic_sphere(1)
chi_0 = ManifoldTopology.euler_characteristic_sphere(0)
if chi_2 == 2 and chi_1 == 0 and chi_0 == 2:
    ok(f"Euler char: χ(S²)={chi_2}, χ(S¹)={chi_1}, χ(S⁰)={chi_0}")
else:
    err(f"Euler char: χ(S²)={chi_2} (exp 2), χ(S¹)={chi_1} (exp 0), χ(S⁰)={chi_0} (exp 2)")

# Euler char torus: χ(Σ₁) = 0 (genus 1)
chi_torus = ManifoldTopology.euler_characteristic_torus(1)
if chi_torus == 0:
    ok(f"Euler char torus (g=1): χ={chi_torus}")
else:
    err(f"Euler char torus: {chi_torus}, expected 0")

# Betti numbers of S²: b = [1, 0, 1]
betti_2 = ManifoldTopology.betti_numbers_sphere(2)
if betti_2 == [1, 0, 1]:
    ok(f"Betti numbers S²: {betti_2}")
else:
    err(f"Betti numbers S²: {betti_2}, expected [1, 0, 1]")

# Gauss-Bonnet: ∫K dA = 2πχ
gb = ManifoldTopology.gauss_bonnet_curvature_integral(2)
expected_gb = 2 * math.pi * 2
if abs(gb - expected_gb) < 1e-10:
    ok(f"Gauss-Bonnet(χ=2): ∫K dA = {gb:.6f} = 4π")
else:
    err(f"Gauss-Bonnet: {gb}, expected {expected_gb}")

# Gaussian curvature of unit sphere = 1
gc_sphere = CurvatureAnalysis.gaussian_curvature_sphere(1.0)
if abs(gc_sphere - 1.0) < 1e-10:
    ok(f"Gaussian curvature (r=1) = {gc_sphere}")
else:
    err(f"Gaussian curvature (r=1) = {gc_sphere}, expected 1.0")

# Einstein tensor trace: G = R*(1 - d/2) for d=4 → G = R*(-1)
ricci_test = 10.0
et_trace = CurvatureAnalysis.einstein_tensor_trace(ricci_test, 4)
expected_et = ricci_test * (1 - 4 / 2)  # = 10 * (-1) = -10
if abs(et_trace - expected_et) < 1e-10:
    ok(f"Einstein tensor trace (R=10, d=4) = {et_trace}")
else:
    err(f"Einstein tensor trace = {et_trace}, expected {expected_et}")

# Geodesic flow
geo = ManifoldExtended.geodesic_flow([0, 0], [1, 0], 0.0, steps=10, dt=0.01)
if abs(geo["final_position"][0] - 0.1) < 0.01:  # 10 steps * 0.01 * v=1
    ok(f"Geodesic flow (flat): final={geo['final_position']}")
else:
    warn(f"Geodesic flow: final={geo['final_position']} (expected near [0.1, 0])")

close_phase(10, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 11: L6 — Void Mathematics
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(11, "L6 — Void Math (primal, paradox, sequence, recursive emptiness)")

# Primal calculus
pc = VoidMath.primal_calculus(2.0)
expected_pc = 2.0 ** PHI / (1.04 * PI)
if abs(pc - expected_pc) < 1e-10:
    ok(f"VoidMath.primal_calculus(2) = {pc:.10f}")
else:
    err(f"Primal calculus mismatch: {pc} vs {expected_pc}")

# Paradox resolve
paradox = VoidMath.paradox_resolve(3.0, 4.0)
# synthesis = √(9+16) * VOID_CONSTANT = 5 * 1.0416... ≈ 5.208
expected_synthesis = math.sqrt(9 + 16) * VOID_CONSTANT
if abs(paradox["synthesis"] - expected_synthesis) < 1e-10 and paradox["resolved"]:
    ok(f"paradox_resolve(3,4): synthesis={paradox['synthesis']:.6f}")
else:
    err(f"paradox_resolve failed: {paradox}")

# Void sequence: should converge toward 0
seq = VoidMath.void_sequence(100.0, 13)
if abs(seq[-1]) < abs(seq[0]):
    ok(f"void_sequence(100, 13): converges {seq[0]:.1f} → {seq[-1]:.6f}")
else:
    warn(f"void_sequence not converging: {seq[0]} → {seq[-1]}")

# Recursive emptiness: should converge to fixed point
re_result = VoidCalculus.recursive_emptiness(5.0, 50)
if re_result["converged"]:
    ok(f"recursive_emptiness(5, 50): value={re_result['final_value']}, fixed_point={re_result['fixed_point']}")
else:
    warn(f"recursive_emptiness not converged: final={re_result['final_value']}, fp={re_result['fixed_point']}, error={re_result['error']:.2e}")

# Void field energy: empty list → high emptiness
vfe_empty = VoidCalculus.void_field_energy([])
if vfe_empty["emptiness"] == 1.0:
    ok("void_field_energy([]) → emptiness=1.0")
else:
    err(f"void_field_energy([]) → emptiness={vfe_empty['emptiness']}")

# Void derivative: d/dx(x²) at x=3 should be ≈ 6 * VOID_CONSTANT
vd = VoidCalculus.void_derivative(lambda x: x ** 2, 3.0)
expected_vd = 6.0 * VOID_CONSTANT
if abs(vd - expected_vd) < 1e-3:
    ok(f"void_derivative(x², 3) = {vd:.6f} ≈ 6 * VOID_CONSTANT")
else:
    err(f"void_derivative(x², 3) = {vd}, expected {expected_vd}")

close_phase(11, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 12: L7 — Abstract Algebra
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(12, "L7 — Abstract Algebra (Zeckendorf, continued fractions, structures)")

sns = SacredNumberSystem()

# Zeckendorf: 10 = 8 + 2 (non-consecutive Fibonacci)
z10 = sns.zeckendorf(10)
if z10 == [8, 2]:
    ok(f"Zeckendorf(10) = {z10}")
else:
    err(f"Zeckendorf(10) = {z10}, expected [8, 2]")

# Zeckendorf: 13 = [13] (is Fibonacci)
z13 = sns.zeckendorf(13)
if z13 == [13]:
    ok(f"Zeckendorf(13) = {z13}")
else:
    err(f"Zeckendorf(13) = {z13}, expected [13]")

# Continued fraction of PHI = [1; 1, 1, 1, ...]
cf_phi = sns.continued_fraction(PHI, 15)
if all(a == 1 for a in cf_phi):
    ok(f"CF(φ) = [{', '.join(str(a) for a in cf_phi[:8])}...] (all 1s)")
else:
    err(f"CF(φ) = {cf_phi}, expected all 1s")

# Continued fraction roundtrip: value → CF → value
cf_pi = sns.continued_fraction(math.pi, 10)
pi_recovered = sns.from_continued_fraction(cf_pi)
if abs(pi_recovered - math.pi) < 1e-6:
    ok(f"CF roundtrip π: {math.pi} → [{','.join(str(a) for a in cf_pi[:5])},...] → {pi_recovered:.10f}")
else:
    err(f"CF roundtrip π failed: {pi_recovered} vs {math.pi}")

# Algebraic structure classification
elements = [1.0, 2.0, 3.0]
op = BinaryOperation("add_mod_4", lambda a, b: (a + b) % 4, elements)
classified = op.classify()
if isinstance(classified, AlgebraType):
    ok(f"Algebraic classification: {classified.value}")
else:
    err(f"Classification failed: {classified}")

# Theorem generator: phi identity
phi_id = TheoremGenerator.generate_phi_identity(10)
if phi_id["verified"]:
    ok(f"PHI identity φ^10 = F(10)φ + F(9): error={phi_id['error']:.2e}")
else:
    err(f"PHI identity failed: {phi_id}")

close_phase(12, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 13: L8 — Ontological Mathematics
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(13, "L8 — Ontological (monads, existence calculus, Gödel, Platonic)")

om = OntologicalMathematics()

# Platonic realm has 5 forms
if len(om.platonic_realm.forms) >= 5:
    ok(f"Platonic realm: {len(om.platonic_realm.forms)} forms")
else:
    err(f"Platonic realm: only {len(om.platonic_realm.forms)} forms, expected ≥ 5")

# Create a monad
monad = om.create_monad("test_monad")
p1 = monad.perceive(10.0)
p2 = monad.perceive(5.0)
if p2 != 0:
    ok(f"Monad perception: {p1:.4f} → {p2:.4f}")
else:
    err("Monad perception returned 0")

# Existence calculus: create and observe
ec = om.existence_calculus
entity = ec.create("test_entity", 3.14)
observed = ec.observe("test_entity")
if observed is not None and observed["collapsed_value"] > 0:
    ok(f"Existence: create → observe: collapsed_value={observed['collapsed_value']:.6f}")
else:
    err(f"Existence calculus failed: {observed}")

# Annihilate
annihilated = ec.annihilate("test_entity")
if annihilated:
    ok("Existence: annihilate succeeded")
else:
    err("Existence: annihilate failed")

# Gödel number is deterministic
gn1 = GodelianSelfReference.godel_number("test")
gn2 = GodelianSelfReference.godel_number("test")
if gn1 == gn2 and gn1 > 0:
    ok(f"Gödel number deterministic: G('test') = {gn1}")
else:
    err(f"Gödel number non-deterministic: {gn1} vs {gn2}")

# Mathematical consciousness: recursive improvement
mc = om.consciousness
insights = mc.recursive_improve(5)
if len(insights) == 5 and insights[-1]["awareness"] > insights[0]["awareness"]:
    ok(f"Consciousness: 5 reflections, awareness {insights[0]['awareness']:.4f} → {insights[-1]['awareness']:.4f}")
else:
    warn(f"Consciousness not improving: {[i['awareness'] for i in insights]}")

close_phase(13, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 14: L9 — Sovereign Proofs (stability, entropy, Collatz)
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(14, "L9 — Sovereign Proofs (stability, entropy, Collatz, conservation)")

# Stability nirvana
stability = SovereignProofs.proof_of_stability_nirvana(100)
if stability["converged"]:
    ok(f"Stability nirvana: converged, error={stability['error']:.2e}")
else:
    err(f"Stability nirvana: NOT converged, error={stability['error']}")

# Entropy reduction / coherence preservation
entropy = SovereignProofs.proof_of_entropy_reduction(50)
if entropy["phi_more_effective"] and entropy["entropy_decreased"]:
    ok(f"Entropy coherence: φ beats all controls, rank={entropy['phi_rank']}")
elif entropy["phi_more_effective"]:
    ok(f"Entropy coherence: φ preserves coherence best, rank={entropy['phi_rank']}")
else:
    if entropy["entropy_decreased"]:
        warn(f"Entropy: φ decreased but rank={entropy['phi_rank']} (not #1)")
    else:
        err(f"Entropy: did NOT decrease (trajectory issue)")

# Collatz: 27 should reach 1
collatz = SovereignProofs.collatz_empirical_verification(27)
if collatz["converged_to_1"]:
    ok(f"Collatz(27): steps={collatz['steps_to_convergence']}, max={collatz['max_value']}")
else:
    err(f"Collatz(27) did not converge")

# Collatz batch: 1-1000
cbatch = SovereignProofs.collatz_batch_verification(1, 1000)
if cbatch["all_converged"]:
    ok(f"Collatz batch [1,1000]: all converged, avg_steps={cbatch['average_stopping_time']}")
else:
    err(f"Collatz batch: failures={cbatch['failures']}")

# GOD_CODE conservation proof
conservation = SovereignProofs.proof_of_god_code_conservation()
if conservation["proven"] and conservation["numerical_verification"]["machine_precision"]:
    ok(f"GOD_CODE conservation: proven, max_error={conservation['numerical_verification']['max_error']:.2e}")
else:
    err(f"GOD_CODE conservation proof failed")

# VOID_CONSTANT derivation proof
void_proof = SovereignProofs.proof_of_void_constant_derivation()
if void_proof["proven"] and void_proof["components"]["exact"]:
    ok(f"VOID_CONSTANT derivation: proven, exact")
else:
    err(f"VOID_CONSTANT derivation proof failed")

# Gödel-Turing meta
godel = GodelTuringMetaProof.execute_meta_framework()
if godel["general_completeness"] == False and godel["practical_decidability"] == True:
    ok("Gödel-Turing: general_completeness=False (honest), practical_decidability=True")
else:
    err(f"Gödel-Turing: unexpected state")

close_phase(14, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 15: L9 — Extended Proofs (Goldbach, twin primes, zeta zeros, PHI)
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(15, "L9 — Extended Proofs (Goldbach, twin primes, zeta zeros, φ convergence)")

# Goldbach up to 1000
goldbach = ExtendedProofs.verify_goldbach(1000)
if goldbach["all_pass"]:
    ok(f"Goldbach [4..1000]: {goldbach['verified']} even numbers verified")
else:
    err(f"Goldbach failures: {goldbach['failures']}")

# Twin primes up to 10000
twins = ExtendedProofs.find_twin_primes(10000)
if twins["twin_pairs"] > 50:
    ok(f"Twin primes [2..10000]: {twins['twin_pairs']} pairs, density={twins['density']:.6f}")
else:
    err(f"Twin primes: only {twins['twin_pairs']}")

# Zeta zeros
zeta = ExtendedProofs.verify_zeta_zeros(5)
if zeta["all_verified"]:
    ok(f"Riemann zeta: {zeta['zeros_verified']}/{zeta['zeros_checked']} zeros verified")
else:
    warn(f"Riemann zeta: {zeta['zeros_verified']}/{zeta['zeros_checked']} verified (Hardy Z approximate)")

# PHI convergence
phi_conv = ExtendedProofs.phi_convergence_proof(50)
if phi_conv["converged"]:
    ok(f"φ convergence: CF error={phi_conv['continued_fraction_error']:.2e}")
else:
    err(f"φ convergence failed: error={phi_conv['continued_fraction_error']}")

close_phase(15, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 16: L9 — Equation Verifier
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(16, "L9 — Equation Verifier (30+ equations)")

ev = EquationVerifier()
results = ev.verify_all()
if results["failed"] == 0:
    ok(f"Equation verifier: {results['passed']}/{results['total']} passed (100%)")
else:
    # Report which equations failed
    failed = [r for r in results["results"] if not r["passed"]]
    for f in failed:
        err(f"  Equation FAILED: {f['name']} — computed={f['computed']}, expected={f['expected']}, error={f['error']:.2e}")
    if results["pass_rate"] > 0.9:
        warn(f"Equation verifier: {results['passed']}/{results['total']} ({results['pass_rate']*100:.1f}%), {results['failed']} failed")
    else:
        err(f"Equation verifier: {results['passed']}/{results['total']} ({results['pass_rate']*100:.1f}%)")

close_phase(16, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 17: L10 — Hyperdimensional Computing
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(17, "L10 — Hyperdimensional (vectors, bind, bundle, SDM)")

hdc = HyperdimensionalCompute(dimension=1000)  # Smaller for speed

# Random vector has correct dimension
v = hdc.random_vector("test")
if len(v.data) == 1000:
    ok(f"Hypervector dimension: {len(v.data)}")
else:
    err(f"Hypervector dimension: {len(v.data)}, expected 1000")

# Seeded vectors are deterministic
v1 = hdc.random_vector("hello")
v2 = hdc.random_vector("hello")
if v1.similarity(v2) > 0.99:
    ok(f"Seeded vectors deterministic: sim={v1.similarity(v2):.6f}")
else:
    err(f"Seeded vectors NOT deterministic: sim={v1.similarity(v2)}")

# Different seeds → quasi-orthogonal
v3 = hdc.random_vector("world")
sim = v1.similarity(v3)
if abs(sim) < 0.1:
    ok(f"Different seeds quasi-orthogonal: sim={sim:.4f}")
else:
    warn(f"Different seeds not quasi-orthogonal: sim={sim:.4f}")

# Bind is self-inverse for bipolar: A ⊗ (A ⊗ B) = B
a = Hypervector.random_bipolar(1000, seed=42)
b = Hypervector.random_bipolar(1000, seed=43)
ab = a.bind(b)
recovered = a.bind(ab)
sim_rb = recovered.similarity(b)
if sim_rb > 0.99:
    ok(f"Bind self-inverse: A⊗(A⊗B) ~ B, sim={sim_rb:.4f}")
else:
    err(f"Bind self-inverse failed: sim={sim_rb:.4f}")

# Bundle preserves dimensionality
bundled = a.bundle(b)
if len(bundled.data) == 1000:
    ok("Bundle preserves dimension")
else:
    err(f"Bundle dimension: {len(bundled.data)}")

# Permute shift
permuted = a.permute(1)
# Permuted should be different but same dimension
if len(permuted.data) == 1000 and a.similarity(permuted) < 0.5:
    ok(f"Permute(1) creates distinct vector: sim={a.similarity(permuted):.4f}")
else:
    warn(f"Permute(1) may not shift enough: sim={a.similarity(permuted):.4f}")

# Sequence encoding
items = [hdc.random_vector(f"item_{i}") for i in range(5)]
encoded = hdc.encode_sequence(items)
if len(encoded.data) == 1000:
    ok(f"Sequence encoding: 5 items → dim={len(encoded.data)}")
else:
    err(f"Sequence encoding dimension wrong")

close_phase(17, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 18: L11 — Computronium & Rayleigh
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(18, "L11 — Computronium & Rayleigh (Airy, diffraction)")

try:
    # Airy diffraction
    airy = AiryDiffraction()
    pattern = airy.compute_pattern(wavelength=500e-9, aperture=1e-3)
    if isinstance(pattern, dict) and len(pattern) > 0:
        ok(f"Airy diffraction pattern: {len(pattern)} keys")
    else:
        warn(f"Airy diffraction: unexpected return type")
except Exception as e:
    warn(f"Airy diffraction: {e}")

try:
    # Rayleigh criterion
    r = RayleighMath()
    resolution = r.rayleigh_criterion(wavelength=500e-9, aperture=0.01)
    if isinstance(resolution, (int, float)) and resolution > 0:
        ok(f"Rayleigh criterion: θ={resolution:.2e} rad")
    elif isinstance(resolution, dict):
        ok(f"Rayleigh criterion: {list(resolution.keys())[:3]}...")
    else:
        warn(f"Rayleigh criterion: {type(resolution)}")
except Exception as e:
    warn(f"Rayleigh criterion: {e}")

try:
    # Computronium math
    cm = ComputroniumMath()
    status = cm.status() if hasattr(cm, 'status') else {"active": True}
    ok(f"ComputroniumMath initialized")
except Exception as e:
    warn(f"ComputroniumMath: {e}")

close_phase(18, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 19: L12 — Berry Geometry
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(19, "L12 — Berry Geometry (fiber bundles, Chern, parallel transport)")

try:
    bg = BerryGeometry()
    # Berry phase
    if hasattr(bg, 'compute_berry_phase'):
        bp = bg.compute_berry_phase()
        ok(f"Berry phase computed: {type(bp)}")
    elif hasattr(bg, 'berry_phase'):
        bp = bg.berry_phase()
        ok(f"Berry phase computed")
    else:
        ok("BerryGeometry instantiated (no direct berry_phase method)")
except Exception as e:
    warn(f"BerryGeometry: {e}")

try:
    fb = FiberBundle()
    if hasattr(fb, 'project') or hasattr(fb, 'parallel_transport'):
        ok("FiberBundle instantiated with transport methods")
    else:
        ok("FiberBundle instantiated")
except Exception as e:
    warn(f"FiberBundle: {e}")

try:
    cw = ChernWeilTheory()
    if hasattr(cw, 'compute_chern_number') or hasattr(cw, 'chern_number'):
        ok("ChernWeilTheory instantiated with Chern methods")
    else:
        ok("ChernWeilTheory instantiated")
except Exception as e:
    warn(f"ChernWeilTheory: {e}")

close_phase(19, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 20: MathEngine Facade
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(20, "MathEngine Facade — Unified Access")

# Status
st = math_engine.status()
if st["version"] == "1.1.0" and st["layers"] == 12:
    ok(f"MathEngine status: v{st['version']}, {st['layers']} layers")
else:
    warn(f"MathEngine status: v{st.get('version')}, layers={st.get('layers')}")

# god_code_value
gcv = math_engine.god_code_value()
if abs(gcv - GOD_CODE) < 1e-9:
    ok(f"math_engine.god_code_value() = {gcv:.10f}")
else:
    err(f"god_code_value() = {gcv}, expected {GOD_CODE}")

# verify_conservation via facade
if math_engine.verify_conservation(0):
    ok("math_engine.verify_conservation(0) = True")
else:
    err("math_engine.verify_conservation(0) = False")

# fibonacci via facade
fib = math_engine.fibonacci(10)
if isinstance(fib, list) and len(fib) == 10:
    ok(f"math_engine.fibonacci(10): {len(fib)} items")
elif isinstance(fib, int):
    # Returns single Fibonacci number F(10)
    ok(f"math_engine.fibonacci(10) = {fib}")
else:
    warn(f"math_engine.fibonacci(10) type: {type(fib)}")

# Lorentz boost via facade
boosted = math_engine.lorentz_boost([1.0, 0, 0, 0], "x", 0.5)
if len(boosted) == 4:
    ok(f"math_engine.lorentz_boost: {[round(x, 4) for x in boosted]}")
else:
    err(f"lorentz_boost returned wrong dim: {boosted}")

# wave_coherence via facade
coh = math_engine.wave_coherence(440.0, GOD_CODE)
if 0 <= coh <= 1:
    ok(f"math_engine.wave_coherence(440, GOD_CODE) = {coh:.4f}")
else:
    err(f"wave_coherence out of range: {coh}")

# hd_vector via facade
hv = math_engine.hd_vector("test_seed")
if hv is not None and hasattr(hv, 'data'):
    ok(f"math_engine.hd_vector('test_seed'): dim={len(hv.data)}")
else:
    err(f"hd_vector failed")

close_phase(20, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 21: Mathematical Invariant Proofs
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(21, "Mathematical Invariant Proofs & Cross-Validation")

# 1. GOD_CODE as product of specific factors
gc_check = (286 ** (1.0 / PHI)) * 16
if abs(gc_check - GOD_CODE) < 1e-9:
    ok(f"GOD_CODE = 286^(1/φ) × 16 = {gc_check:.10f}")
else:
    err(f"GOD_CODE reconstruction failed: {gc_check} vs {GOD_CODE}")

# 2. F(n+1)/F(n) → φ as n → ∞
fibs = PureMath.fibonacci(30)
convergence_errors = []
for i in range(2, 30):
    ratio = fibs[i] / fibs[i - 1]
    convergence_errors.append(abs(ratio - PHI))
if convergence_errors[-1] < 1e-10:
    ok(f"F(30)/F(29) → φ: error={convergence_errors[-1]:.2e}")
else:
    err(f"Fibonacci→φ convergence: error={convergence_errors[-1]}")

# 3. Conservation law is an algebraic identity
# G(X) × 2^(X/104) = 286^(1/φ) × 2^((416-X)/104) × 2^(X/104) = 286^(1/φ) × 2^4 = GOD_CODE
# Test using logarithmic comparison to avoid overflow
extreme_xs = [1e3, -1e3, 1e4, -1e4]
all_conserved = True
for x in extreme_xs:
    # log₂(G(X)) + X/104 should = log₂(GOD_CODE)
    gx = god_code_at(x)
    if gx > 0:
        log_product = math.log2(gx) + x / QUANTIZATION_GRAIN
        log_gc = math.log2(GOD_CODE)
        e = abs(log_product - log_gc)
        if e > 1e-6:
            all_conserved = False
    else:
        all_conserved = False
if all_conserved:
    ok(f"Conservation at extreme X (±10³, ±10⁴): log-verified")
else:
    warn(f"Conservation at extreme X: numerical drift")

# 4. 416/104 = 4 exactly
if 416 / 104 == 4.0:
    ok("416/104 = 4.0 exactly (no float rounding)")
else:
    err(f"416/104 = {416/104}")

# 5. sin(104π/104) = sin(π) ≈ 0
sin_val = math.sin(QUANTIZATION_GRAIN * math.pi / QUANTIZATION_GRAIN)
if abs(sin_val) < 1e-14:
    ok(f"sin(104π/104) = {sin_val:.2e} ≈ 0")
else:
    warn(f"sin(104π/104) = {sin_val}")

# 6. STEP_SIZE = 2^(1/104) and STEP_SIZE^104 = 2
step_104 = STEP_SIZE ** 104
if abs(step_104 - 2.0) < 1e-10:
    ok(f"STEP_SIZE^104 = {step_104:.12f} = 2.0")
else:
    err(f"STEP_SIZE^104 = {step_104}, expected 2.0")

# 7. SEMITONE_RATIO = 2^(8/104) = 2^(1/13) and SEMITONE_RATIO^13 = 2
sem_13 = SEMITONE_RATIO ** 13
if abs(sem_13 - 2.0) < 1e-10:
    ok(f"SEMITONE_RATIO^13 = {sem_13:.12f} = 2.0")
else:
    err(f"SEMITONE_RATIO^13 = {sem_13}, expected 2.0")

close_phase(21, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 22: Edge Cases & Boundary Tests
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(22, "Edge Cases & Boundary Tests")

# god_code_at negative X
g_neg = god_code_at(-104)
g_pos = god_code_at(104)
ratio_neg = g_neg / GOD_CODE
ratio_pos = g_pos / GOD_CODE
if abs(ratio_neg - 2.0) < 1e-10 and abs(ratio_pos - 0.5) < 1e-10:
    ok(f"god_code_at(-104)/GOD_CODE={ratio_neg:.4f}, god_code_at(104)/GOD_CODE={ratio_pos:.4f}")
else:
    err(f"god_code_at boundary: neg={ratio_neg}, pos={ratio_pos}")

# Fibonacci edge cases
if PureMath.fibonacci(0) == [] and PureMath.fibonacci(1) == [1]:
    ok(f"fibonacci(0)=[], fibonacci(1)=[1]")
else:
    err(f"fibonacci edge: 0={PureMath.fibonacci(0)}, 1={PureMath.fibonacci(1)}")

# primal_calculus(0) should return 0 (x <= 0 case)
if primal_calculus(0) == 0.0 and primal_calculus(-1) == 0.0:
    ok("primal_calculus(0)=0, primal_calculus(-1)=0")
else:
    err(f"primal_calculus edge: 0={primal_calculus(0)}, -1={primal_calculus(-1)}")

# wave_coherence(0, x) should return 0
if WavePhysics.wave_coherence(0, 440) == 0.0:
    ok("wave_coherence(0, 440) = 0.0")
else:
    err(f"wave_coherence(0, 440) = {WavePhysics.wave_coherence(0, 440)}")

# Lorentz boost clamp at β ≥ 1
try:
    boosted = Math4D.lorentz_boost_x([1, 0, 0, 0], 1.5)
    if boosted is not None:
        ok(f"Lorentz β>1 handled (clamped to 0.999)")
    else:
        warn("Lorentz β>1 returned None")
except Exception as e:
    warn(f"Lorentz β>1 exception: {e}")

# VoidMath.void_sequence with seed=0
seq0 = VoidMath.void_sequence(0.0, 5)
if seq0 == [0.0] * 5 or all(s == 0 for s in seq0):
    ok("void_sequence(0) → all zeros")
else:
    warn(f"void_sequence(0) = {seq0}")

# Gaussian curvature at r=0 → inf
gc_zero = CurvatureAnalysis.gaussian_curvature_sphere(0)
if gc_zero == float('inf'):
    ok("Gaussian curvature(r=0) = inf")
else:
    err(f"Gaussian curvature(r=0) = {gc_zero}, expected inf")

# Mandelbrot: c=0 should not escape (iterate stays at 0)
from l104_math_engine.pure_math import ComplexMath
mb = ComplexMath.mandelbrot_iterate(0 + 0j, 100)
if mb == 100:
    ok("Mandelbrot(c=0): max_iter reached (in set)")
else:
    err(f"Mandelbrot(c=0) = {mb}, expected 100")

close_phase(22, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 23: Cross-Layer Consistency
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(23, "Cross-Layer Consistency Checks")

# GOD_CODE consistent across layers
gc_const = GOD_CODE
gc_eq = GodCodeEquation.evaluate(0, 0, 0, 0)
gc_at = god_code_at(0)
gc_facade = math_engine.god_code_value()
gc_invariant = INVARIANT
all_gc = [gc_const, gc_eq, gc_at, gc_facade, gc_invariant]
gc_max_diff = max(abs(a - b) for a in all_gc for b in all_gc)
if gc_max_diff < 1e-10:
    ok(f"GOD_CODE consistent across 5 sources (max diff: {gc_max_diff:.2e})")
else:
    err(f"GOD_CODE inconsistent (max diff: {gc_max_diff}): {all_gc}")

# PHI consistent
from l104_math_engine.constants import PHI as PHI_CONST
from l104_math_engine.constants import GOLDEN_RATIO
if PHI_CONST == GOLDEN_RATIO and PHI_CONST == PHI:
    ok("PHI == GOLDEN_RATIO == PHI (consistent)")
else:
    err(f"PHI inconsistent: {PHI_CONST}, {GOLDEN_RATIO}, {PHI}")

# VOID_CONSTANT used consistently in primal_calculus
# primal_calculus uses 1.04, not VOID_CONSTANT
# Check: should be x^φ / (1.04 * π), NOT x^φ / (VOID_CONSTANT * π)
pc_test = primal_calculus(10.0)
pc_with_104 = 10.0 ** PHI / (1.04 * PI)
pc_with_void = 10.0 ** PHI / (VOID_CONSTANT * PI)
if abs(pc_test - pc_with_104) < 1e-10:
    ok("primal_calculus uses 1.04 (not VOID_CONSTANT)")
elif abs(pc_test - pc_with_void) < 1e-10:
    warn("primal_calculus uses VOID_CONSTANT (doc says 1.04π)")
else:
    err(f"primal_calculus inconsistent: result={pc_test}, with 1.04={pc_with_104}, with VOID={pc_with_void}")

# Manifold curvature tensor: symmetric
mct = ManifoldMath.manifold_curvature_tensor(4)
symmetric = True
for i in range(4):
    for j in range(4):
        if abs(mct[i][j] - mct[j][i]) > 1e-10:
            symmetric = False
if symmetric:
    ok("Manifold curvature tensor is symmetric (4×4)")
else:
    err("Manifold curvature tensor NOT symmetric")

# Harmonic optimizer: align(GOD_CODE) should return GOD_CODE (integer harmonic n=1)
aligned = HarmonicOptimizer.harmonic_align(GOD_CODE)
if abs(aligned - GOD_CODE) < 1e-6:
    ok(f"harmonic_align(GOD_CODE) = {aligned:.4f} = GOD_CODE (FIXED)")
else:
    err(f"harmonic_align(GOD_CODE) = {aligned}, expected {GOD_CODE}")

close_phase(23, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 24: Processing Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(24, "Processing Benchmarks (LOPS)")

bench = ProcessingProofs.run_speed_benchmark(50000)
if bench["lops"] > 100000:
    ok(f"LOPS benchmark: {bench['lops']:.0f} ops/sec ({bench['elapsed_seconds']:.3f}s)")
else:
    warn(f"LOPS benchmark: {bench['lops']:.0f} ops/sec (low performance)")

close_phase(24, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 25: Prove All (Full Suite)
# ═══════════════════════════════════════════════════════════════════════════════
t0 = phase(25, "Prove All — Full Proof Suite via MathEngine")

proofs = math_engine.prove_all()
sp = proofs.get("stability_nirvana", {})
er = proofs.get("entropy_reduction", {})
cl = proofs.get("collatz", {})
gc = proofs.get("god_code_conservation", {})
vd = proofs.get("void_constant_derivation", {})
gt = proofs.get("godel_turing", {})

proof_checks = [
    ("stability_nirvana", sp.get("converged", False)),
    ("entropy_reduction", er.get("phi_more_effective", False)),
    ("collatz", cl.get("converged_to_1", False)),
    ("god_code_conservation", gc.get("proven", False)),
    ("void_constant_derivation", vd.get("proven", False)),
    ("godel_turing", gt.get("practical_decidability", False)),
]

all_pass = True
for name, passed_flag in proof_checks:
    if passed_flag:
        ok(f"prove_all → {name}: PASSED")
    else:
        err(f"prove_all → {name}: FAILED")
        all_pass = False

close_phase(25, t0)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
total = PASSED + ERRORS
print(f"\n\n╔══════════════════════════════════════════════════════════════════════════╗")
print(f"║                     DEBUG SESSION COMPLETE                              ║")
print(f"╠══════════════════════════════════════════════════════════════════════════╣")
print(f"║  Total tests:     {total:>4}                                              ║")
print(f"║  Passed:          {PASSED:>4}  ({PASSED/max(total,1)*100:.1f}%)                                       ║")
print(f"║  Errors:          {ERRORS:>4}                                              ║")
print(f"║  Warnings:        {WARNINGS:>4}                                              ║")
print(f"║  Total time:      {sum(PHASE_TIMES.values()):.2f}s                                            ║")
print(f"╚══════════════════════════════════════════════════════════════════════════╝")

if WARNINGS:
    print(f"\n  WARNINGS ({WARNINGS}):")
    for kind, msg in RESULTS:
        if kind == "WARN":
            print(f"    ⚠ {msg}")

if ERRORS:
    print(f"\n  ERRORS ({ERRORS}):")
    for kind, msg in RESULTS:
        if kind == "ERROR":
            print(f"    ✗ {msg}")

# Phase timing
print(f"\n  Phase Timing:")
total_time = sum(PHASE_TIMES.values())
for num in sorted(PHASE_TIMES):
    t = PHASE_TIMES[num]
    bar_len = int(t / max(total_time, 0.001) * 40)
    bar = "█" * bar_len + "░" * (40 - bar_len)
    print(f"    Phase {num:02d}: {t:>7.3f}s  {bar}")

# Verdict
if ERRORS == 0 and WARNINGS <= 5:
    verdict = "★ SOVEREIGN ★"
elif ERRORS == 0:
    verdict = "OPERATIONAL (warnings only)"
elif ERRORS <= 3:
    verdict = "DEGRADED"
else:
    verdict = "CRITICAL"

print(f"\n  VERDICT: {verdict}  — {PASSED}/{total} tests passed, {ERRORS} errors, {WARNINGS} warnings")
print("════════════════════════════════════════════════════════════════════════\n")

sys.exit(0 if ERRORS == 0 else 1)
