#!/usr/bin/env python3
"""
L104 Science Engine — Deep Debug Suite v1.0
═══════════════════════════════════════════════════════════════════════════════
25-phase comprehensive debug of l104_science_engine/ v5.1.0 (7,913 lines)

MODULES TESTED:
  Layer 0: constants.py      (254 lines)  — Sacred + CODATA constants
  Layer 1: physics.py        (383 lines)  — PhysicsSubsystem
  Layer 2: entropy.py        (554 lines)  — Maxwell's Demon entropy reversal
  Layer 3: multidimensional.py (213 lines) — N-dim relativistic processing
  Layer 4: coherence.py      (781 lines)  — Topological coherence
  Layer 5: quantum_25q.py    (775 lines)  — 25Q circuit templates + memory
  Layer 5a: berry_phase.py   (1678 lines) — Berry geometric phase
  Layer 5b: computronium.py  (941 lines)  — Fundamental computation limits
  Layer 6: bridge.py         (519 lines)  — Math↔Science↔Quantum bridge
  Layer 7: engine.py         (1536 lines) — Master orchestrator

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import sys, math, cmath, time, traceback
import numpy as np

# ── Global counters ──
PASS = 0
FAIL = 0
WARN = 0
BUGS = []

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        BUGS.append((name, detail))
        print(f"  ❌ {name}  ← {detail}")

def warn(name, detail=""):
    global WARN
    WARN += 1
    print(f"  ⚠️  {name}: {detail}")

def phase(num, title):
    print(f"\n{'═'*70}")
    print(f"  PHASE {num}: {title}")
    print(f"{'═'*70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: Imports & Boot
# ═══════════════════════════════════════════════════════════════════════════════

phase(1, "IMPORTS & BOOT")

try:
    from l104_science_engine import (
        # Constants
        GOD_CODE, PHI, PHI_CONJUGATE, PHI_SQUARED, PHI_CUBED,
        PRIME_SCAFFOLD, QUANTIZATION_GRAIN, OCTAVE_OFFSET,
        BASE, STEP_SIZE, VOID_CONSTANT,
        OMEGA, OMEGA_AUTHORITY, ZETA_ZERO_1, FEIGENBAUM, ALPHA_FINE,
        PhysicalConstants, PC, IronConstants, Fe, HeliumConstants, He4,
        QuantumBoundary, QB,
        # Subsystems
        PhysicsSubsystem,
        EntropySubsystem,
        MultiDimensionalSubsystem,
        CoherenceSubsystem,
        # Quantum
        GodCodeQuantumConvergence,
        CircuitTemplates25Q,
        MemoryValidator,
        QuantumCircuitScience,
        # Berry phase
        BerryPhaseSubsystem,
        BerryPhaseCalculator,
        ChernNumberEngine,
        MolecularBerryPhase,
        AharonovBohmEngine,
        PancharatnamPhase,
        # Computronium
        ComputroniumSubsystem,
        # Bridge & Engine
        ScienceBridge, bridge,
        ScienceEngine, science_engine,
    )
    check("All imports succeeded", True)
except Exception as e:
    check("All imports succeeded", False, str(e))
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: Sacred Constants Cross-Validation
# ═══════════════════════════════════════════════════════════════════════════════

phase(2, "SACRED CONSTANTS CROSS-VALIDATION")

check("GOD_CODE = 527.5184818492612",
      abs(GOD_CODE - 527.5184818492612) < 1e-10,
      f"got {GOD_CODE}")

check("PHI = (1+√5)/2",
      abs(PHI - (1 + math.sqrt(5)) / 2) < 1e-15,
      f"got {PHI}")

check("PHI_CONJUGATE = (√5-1)/2",
      abs(PHI_CONJUGATE - (math.sqrt(5) - 1) / 2) < 1e-15,
      f"got {PHI_CONJUGATE}")

check("PHI × PHI_CONJUGATE = 1",
      abs(PHI * PHI_CONJUGATE - 1.0) < 1e-14,
      f"product={PHI * PHI_CONJUGATE}")

check("PHI² = PHI + 1",
      abs(PHI_SQUARED - PHI - 1.0) < 1e-14,
      f"PHI²={PHI_SQUARED}, PHI+1={PHI+1}")

check("VOID_CONSTANT = 1.04 + PHI/1000",
      abs(VOID_CONSTANT - (1.04 + PHI / 1000)) < 1e-15,
      f"got {VOID_CONSTANT}")

check("PRIME_SCAFFOLD = 286",
      PRIME_SCAFFOLD == 286, f"got {PRIME_SCAFFOLD}")

check("QUANTIZATION_GRAIN = 104",
      QUANTIZATION_GRAIN == 104, f"got {QUANTIZATION_GRAIN}")

check("OCTAVE_OFFSET = 416",
      OCTAVE_OFFSET == 416, f"got {OCTAVE_OFFSET}")

check("OCTAVE_OFFSET = 4 × QUANTIZATION_GRAIN",
      OCTAVE_OFFSET == 4 * QUANTIZATION_GRAIN,
      f"{OCTAVE_OFFSET} vs {4 * QUANTIZATION_GRAIN}")

check("BASE = 286^(1/φ)",
      abs(BASE - 286 ** (1.0 / PHI)) < 1e-10,
      f"got {BASE}")

check("STEP_SIZE = 2^(1/104)",
      abs(STEP_SIZE - 2 ** (1.0 / 104)) < 1e-14,
      f"got {STEP_SIZE}")

check("GOD_CODE = BASE × 2^4",
      abs(GOD_CODE - BASE * 16.0) < 1e-10,
      f"BASE×16={BASE*16}, GOD_CODE={GOD_CODE}")

# Conservation law: G(x) × 2^(x/104) = GOD_CODE for all x
for x in [0, 52, 104, 208, 416]:
    g_x = BASE * (2 ** ((OCTAVE_OFFSET - x) / QUANTIZATION_GRAIN))
    product = g_x * (2 ** (x / QUANTIZATION_GRAIN))
    check(f"Conservation G({x})×2^({x}/104) = GOD_CODE",
          abs(product - GOD_CODE) < 1e-10,
          f"product={product:.12f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: Physical Constants (CODATA 2022)
# ═══════════════════════════════════════════════════════════════════════════════

phase(3, "PHYSICAL CONSTANTS (CODATA 2022)")

check("Boltzmann k_B = 1.380649e-23 (exact SI)",
      abs(PC.K_B - 1.380649e-23) < 1e-30, f"got {PC.K_B}")

check("Planck h = 6.62607015e-34 (exact SI)",
      abs(PC.H - 6.62607015e-34) < 1e-41, f"got {PC.H}")

check("Speed of light c = 299792458 (exact SI)",
      PC.C == 299792458, f"got {PC.C}")

# CODATA stores ℏ rounded to 10 sig figs; accept ~1e-43 tolerance
check("ℏ ≈ h/(2π) (CODATA precision)",
      abs(PC.H_BAR - PC.H / (2 * math.pi)) < 1e-43,
      f"H_BAR={PC.H_BAR}, h/2π={PC.H/(2*math.pi)}")

check("Fine structure α ≈ 1/137",
      abs(1.0 / PC.ALPHA - 137.036) < 0.001,
      f"1/α = {1.0/PC.ALPHA}")

check("ALPHA_FINE ≈ α",
      abs(ALPHA_FINE - PC.ALPHA) < 1e-6,
      f"ALPHA_FINE={ALPHA_FINE}, PC.ALPHA={PC.ALPHA}")

# Iron constants
check("Fe atomic number = 26",
      Fe.ATOMIC_NUMBER == 26, f"got {Fe.ATOMIC_NUMBER}")

check("Fe BCC lattice ≈ 286.65 pm",
      abs(Fe.BCC_LATTICE_PM - 286.65) < 0.1, f"got {Fe.BCC_LATTICE_PM}")

check("Fe Curie temp = 1043 K",
      Fe.CURIE_TEMP == 1043.0, f"got {Fe.CURIE_TEMP}")

# Helium constants
check("He-4 mass number = 4",
      He4.MASS_NUMBER == 4, f"got {He4.MASS_NUMBER}")

# 104 = Fe(26) × He-4(4)
check("QUANTIZATION_GRAIN = Fe × He-4",
      QUANTIZATION_GRAIN == Fe.ATOMIC_NUMBER * He4.MASS_NUMBER,
      f"26×4={Fe.ATOMIC_NUMBER * He4.MASS_NUMBER}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: QuantumBoundary — 25Q↔26Q Integrity
# ═══════════════════════════════════════════════════════════════════════════════

phase(4, "QUANTUM BOUNDARY — 25Q↔26Q INTEGRITY")

check("QB.N_QUBITS = 26 (Iron Completion)",
      QB.N_QUBITS == 26, f"got {QB.N_QUBITS}")

check("QB.HILBERT_DIM = 2^26",
      QB.HILBERT_DIM == 2**26, f"got {QB.HILBERT_DIM}")

check("QB.STATEVECTOR_BYTES = 2^26 × 16",
      QB.STATEVECTOR_BYTES == 2**26 * 16,
      f"got {QB.STATEVECTOR_BYTES}")

sv_mb = QB.STATEVECTOR_BYTES / (1024 * 1024)
check("QB.STATEVECTOR_MB = 1024 (26Q)",
      QB.STATEVECTOR_MB == 1024,
      f"got {QB.STATEVECTOR_MB}, computed={sv_mb}")

check("QB.BYTES_PER_AMPLITUDE = 16",
      QB.BYTES_PER_AMPLITUDE == 16, f"got {QB.BYTES_PER_AMPLITUDE}")

# Legacy 25Q references
check("QB.N_QUBITS_25 = 25 (legacy)",
      QB.N_QUBITS_25 == 25, f"got {QB.N_QUBITS_25}")

check("QB.HILBERT_DIM_25 = 2^25 (legacy)",
      QB.HILBERT_DIM_25 == 2**25, f"got {QB.HILBERT_DIM_25}")

check("QB.STATEVECTOR_MB_25 = 512 (legacy)",
      QB.STATEVECTOR_MB_25 == 512, f"got {QB.STATEVECTOR_MB_25}")

# Iron completion bridge
check("IRON_QUBIT_BRIDGE = 0 (Fe(26) - 26Q = 0)",
      QB.IRON_QUBIT_BRIDGE == 0,
      f"got {QB.IRON_QUBIT_BRIDGE}")

check("GOD_CODE_TO_1024 ≈ 0.5151",
      abs(QB.GOD_CODE_TO_1024 - GOD_CODE / 1024.0) < 1e-10,
      f"got {QB.GOD_CODE_TO_1024}")

check("GOD_CODE_TO_512 ≈ 1.0303 (legacy)",
      abs(QB.GOD_CODE_TO_512 - GOD_CODE / 512.0) < 1e-10,
      f"got {QB.GOD_CODE_TO_512}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: GodCodeQuantumConvergence — 26Q UPGRADE BUG CHECK
# ═══════════════════════════════════════════════════════════════════════════════

phase(5, "GOD_CODE QUANTUM CONVERGENCE — 26Q BUG CHECK")

conv = GodCodeQuantumConvergence.analyze()

# The core question: does analyze() produce correct results?
mem_used = conv.get("memory_mb", 0)
ratio = conv.get("ratio", 0)
excess_pct = conv.get("excess_above_parity_pct", 0)

print(f"  [INFO] Convergence uses memory_mb = {mem_used} (25Q legacy)")
print(f"  [INFO] ratio = GOD_CODE / memory_mb = {ratio:.8f}")
print(f"  [INFO] excess_above_parity_pct = {excess_pct:.4f}%")
print(f"  [INFO] 26Q iron ratio = {conv.get('iron_memory_ratio', 'N/A')}")

# After fix: mem_used = 512 (25Q legacy), ratio = 1.030, excess = 3.03%
check("Convergence ratio > 1.0 (quantum advantage)",
      ratio > 1.0,
      f"ratio={ratio:.8f} (should be ~1.030 for 512MB)")

check("Excess above parity is positive",
      excess_pct > 0,
      f"excess={excess_pct:.4f}% (should be ~3.03%)")

# The reconstruction check should always be ~0
check("GOD_CODE reconstruction check ≈ 0",
      conv.get("reconstruction_check", 1) < 1e-10,
      f"reconstruction_error={conv.get('reconstruction_check', 'N/A')}")

# Iron qubit bridge interpretation
iron_bridge = conv.get("iron_qubit_bridge", {})
fe_diff = iron_bridge.get("difference", -1)
n_qubits_used = iron_bridge.get("n_qubits", 0)
print(f"  [INFO] Iron bridge: Fe({Fe.ATOMIC_NUMBER}) - {n_qubits_used}Q = {fe_diff}")
check("Iron bridge uses current QB.N_QUBITS",
      n_qubits_used == QB.N_QUBITS,
      f"analyze() uses {n_qubits_used}Q but QB.N_QUBITS={QB.N_QUBITS}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 6: CircuitTemplates25Q — Qubit Count Consistency
# ═══════════════════════════════════════════════════════════════════════════════

phase(6, "CIRCUIT TEMPLATES — QUBIT COUNT CONSISTENCY")

ghz = CircuitTemplates25Q.ghz()
grover = CircuitTemplates25Q.grover()
vqe = CircuitTemplates25Q.vqe()

ghz_qubits = ghz.get("n_qubits", 0)
grover_qubits = grover.get("n_qubits", 0)
vqe_qubits = vqe.get("n_qubits", 0)
ghz_mem = ghz.get("memory_mb", 0)

print(f"  [INFO] GHZ template: n_qubits={ghz_qubits}, memory={ghz_mem}MB")
print(f"  [INFO] Grover template: n_qubits={grover_qubits}")
print(f"  [INFO] VQE template: n_qubits={vqe_qubits}")

# BUG: If QB.N_QUBITS=26, templates say "25-qubit GHZ" but produce 26Q circuits
check("GHZ name matches actual qubit count",
      str(ghz_qubits) in ghz.get("name", ""),
      f"name='{ghz.get('name','')}' but n_qubits={ghz_qubits}")

check("GHZ description matches actual qubit count",
      str(ghz_qubits) in ghz.get("description", ""),
      f"description mentions '25-qubit' but n_qubits={ghz_qubits}")

# Memory should match actual qubit count
expected_mem = 2**ghz_qubits * 16 / (1024*1024)
check("GHZ memory_mb matches qubit count",
      abs(ghz_mem - expected_mem) < 1,
      f"memory_mb={ghz_mem} but {ghz_qubits}Q needs {expected_mem:.0f}MB")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 7: MemoryValidator.validate_512mb() — 26Q Integrity
# ═══════════════════════════════════════════════════════════════════════════════

phase(7, "MEMORY VALIDATOR — 512MB CHECK")

mem_val = MemoryValidator.validate_512mb()
exact_512 = mem_val.get("statevector_exact_512", None)
sv_mb_val = mem_val.get("statevector_mb", 0)
fits_1gb = mem_val.get("fits_in_1gb", None)

print(f"  [INFO] validate_512mb() → sv_mb={sv_mb_val}, exact_512={exact_512}")

# BUG: statevector_exact_512 will be False since QB is 26Q (1024MB)
check("Statevector size matches reality",
      exact_512 == True,
      f"exact_512={exact_512}, sv_mb={sv_mb_val} (QB now 26Q = 1024MB)")

check("Total system fits in expected RAM",
      mem_val.get("fits_in_2gb", False) == True,
      f"total_mb={mem_val.get('total_estimated_mb', 'N/A')}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 8: Bridge Status — 512MB Reference
# ═══════════════════════════════════════════════════════════════════════════════

phase(8, "BRIDGE STATUS — 26Q INTEGRITY")

br_status = bridge.status()
br_512 = br_status.get("512mb_exact", None)
print(f"  [INFO] bridge.status() → 512mb_exact={br_512}")

# BUG: bridge checks QB.STATEVECTOR_MB == 512 but it's 1024
check("Bridge 512mb_exact is True",
      br_512 == True,
      f"512mb_exact={br_512} (QB.STATEVECTOR_MB={QB.STATEVECTOR_MB})")

# Conservation check should always work
cons = br_status.get("conservation_law", None)
check("Bridge conservation law valid",
      cons == True, f"conservation_law={cons}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 9: Bridge conservation_sweep — Points Parameter Bug
# ═══════════════════════════════════════════════════════════════════════════════

phase(9, "BRIDGE CONSERVATION SWEEP — PARAMETER BUG")

sweep_5 = bridge.conservation_sweep(points=5)
sweep_10 = bridge.conservation_sweep(points=10)

pts_5 = sweep_5.get("points_checked", 0)
pts_10 = sweep_10.get("points_checked", 0)

print(f"  [INFO] sweep(points=5) → points_checked={pts_5}")
print(f"  [INFO] sweep(points=10) → points_checked={pts_10}")

# BUG: conservation_sweep ignores `points` parameter — always checks 5 fixed points
check("conservation_sweep(points=5) respects parameter",
      pts_5 == 5, f"requested 5, got {pts_5}")

# This would fail if points is ignored and always returns 5
check("conservation_sweep(points=10) respects parameter",
      pts_10 == 10,
      f"requested 10, got {pts_10} (parameter likely ignored)")

# But conservation itself should pass
check("All conservation checks pass",
      sweep_5.get("all_conserved", False) == True,
      f"all_conserved={sweep_5.get('all_conserved')}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 10: Physics Subsystem
# ═══════════════════════════════════════════════════════════════════════════════

phase(10, "PHYSICS SUBSYSTEM")

phys = PhysicsSubsystem()

# Landauer limit at room temp
landauer = phys.adapt_landauer_limit(293.15)
expected_base = PC.K_B * 293.15 * math.log(2)
expected = expected_base * (GOD_CODE / PHI)
check("Landauer limit positive",
      landauer > 0, f"got {landauer}")
check("Landauer limit formula correct",
      abs(landauer - expected) < 1e-25,
      f"got {landauer}, expected {expected}")

# Bohr resonance
bohr = phys.calculate_bohr_resonance(1)
check("Bohr radius positive",
      bohr > 0, f"got {bohr}")

# Photon resonance
photon = phys.calculate_photon_resonance()
check("Photon resonance is finite",
      math.isfinite(photon), f"got {photon}")

# Casimir force
casimir = phys.calculate_casimir_force(1e-6, 1e-4)
check("Casimir force is attractive (negative)",
      casimir.get("casimir_force_N", 0) < 0,
      f"got {casimir.get('casimir_force_N')}")

# Unruh temperature
unruh = phys.calculate_unruh_temperature(9.81)
check("Unruh temperature at g is tiny",
      0 < unruh.get("unruh_temperature_K", 0) < 1e-15,
      f"got {unruh.get('unruh_temperature_K')}")

# Wien peak — sun should peak near 500nm (visible)
wien = phys.calculate_wien_peak(5778.0)
peak_nm = wien.get("peak_wavelength_nm", 0)
check("Sun peak wavelength 400-600nm",
      400 < peak_nm < 600,
      f"got {peak_nm:.1f}nm")

# Iron lattice Hamiltonian
ham = phys.iron_lattice_hamiltonian(25, 293.15, 1.0)
check("Hamiltonian j_coupling positive",
      ham.get("j_coupling_J", 0) > 0,
      f"j={ham.get('j_coupling_J')}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 11: Entropy Subsystem — Maxwell's Demon
# ═══════════════════════════════════════════════════════════════════════════════

phase(11, "ENTROPY SUBSYSTEM — MAXWELL'S DEMON")

entropy = EntropySubsystem()

# Demon efficiency
for s in [0.1, 1.0, 5.0, 10.0]:
    eff = entropy.calculate_demon_efficiency(s)
    check(f"Demon efficiency at S={s} in [0,1]",
          0.0 <= eff <= 1.0,
          f"got {eff:.6f}")

# Inject coherence
noise = np.random.randn(64)
ordered = entropy.inject_coherence(noise)
check("inject_coherence returns same shape",
      ordered.shape == noise.shape,
      f"input={noise.shape}, output={ordered.shape}")

# Entropy cascade (damped)
cascade = entropy.entropy_cascade(1.0, 104, damped=True)
check("Damped cascade converges",
      cascade.get("converged", False) == True,
      f"converged={cascade.get('converged')}, fixed_point={cascade.get('fixed_point')}")

# Chaos conservation cascade
chaos_healed = entropy.chaos_conservation_cascade(530.0, 104)
check("Chaos cascade heals > 95%",
      chaos_healed.get("healing_pct", 0) > 95,
      f"healing={chaos_healed.get('healing_pct')}%")

# Phi-weighted demon
phi_demon = entropy.phi_weighted_demon(np.abs(np.random.randn(32)) + 0.1)
check("phi_weighted_demon reduces variance",
      phi_demon.get("reduction_ratio", 2) < 1.0,
      f"reduction={phi_demon.get('reduction_ratio')}")

# Multi-scale reversal
multi = entropy.multi_scale_reversal(np.random.randn(64), scales=3)
check("Multi-scale reduces total variance",
      multi.get("total_variance_reduction", -1) > 0,
      f"reduction={multi.get('total_variance_reduction')}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 12: Multidimensional Subsystem
# ═══════════════════════════════════════════════════════════════════════════════

phase(12, "MULTIDIMENSIONAL SUBSYSTEM")

md = MultiDimensionalSubsystem()

check("Default dimension = 11",
      md.dimension == 11, f"got {md.dimension}")

check("Metric is 11×11",
      md.metric.shape == (11, 11),
      f"got {md.metric.shape}")

# Metric signature: should be Lorentzian (-,+,+,...,+)
sig = md.metric_signature_analysis()
check("Metric is Lorentzian (1 timelike)",
      sig.get("is_lorentzian", False) == True,
      f"timelike={sig.get('timelike_dims')}, sig={sig.get('signature_string')}")

# Lorentz boost
point = np.ones(11)
boosted = md.apply_lorentz_boost(point, 1e5, axis=1)
check("Lorentz boost changes point",
      not np.allclose(point, boosted),
      "boost had no effect")

# Process vector
v = np.random.randn(11)
processed = md.process_vector(v)
check("process_vector returns 11-dim",
      len(processed) == 11, f"got len={len(processed)}")

# Projection
proj = md.project(3)
check("project(3) returns 3-dim",
      len(proj) == 3, f"got len={len(proj)}")

# Ricci scalar
ricci = md.ricci_scalar_estimate()
check("Ricci scalar is finite",
      math.isfinite(ricci), f"got {ricci}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 13: Coherence Subsystem
# ═══════════════════════════════════════════════════════════════════════════════

phase(13, "COHERENCE SUBSYSTEM")

coh = CoherenceSubsystem()

# Initialize
seeds = ["GOD_CODE", "PHI", "Iron lattice", "Entropy reversal", "25Q boundary"]
init = coh.initialize(seeds)
check("Coherence initializes",
      init.get("dimension", 0) == len(seeds),
      f"dimension={init.get('dimension')}")

check("Energy is normalized to 1",
      abs(init.get("energy", 0) - 1.0) < 0.01,
      f"energy={init.get('energy')}")

# Evolve — global phase rotation should preserve coherence (initial ≈ final)
evolve = coh.evolve(5)
init_c = evolve.get("initial_coherence", 0)
final_c = evolve.get("final_coherence", 0)
print(f"  [INFO] Evolve: initial={init_c}, final={final_c}, threshold={GOD_CODE/1000*PHI_CONJUGATE:.4f}")
check("Evolve preserves coherence (final ≈ initial)",
      abs(final_c - init_c) < 0.01 or final_c >= init_c * 0.95,
      f"initial={init_c}, final={final_c}, drop={init_c - final_c:.6f}")

# Anchor
anchor = coh.anchor(1.0)
check("Anchor created snapshot",
      anchor.get("snapshots", 0) >= 1,
      f"snapshots={anchor.get('snapshots')}")

# Discover
disc = coh.discover()
check("Discover returns field_size > 0",
      disc.get("field_size", 0) > 0,
      f"field_size={disc.get('field_size')}")

# Golden angle spectrum
spectrum = coh.golden_angle_spectrum()
check("Golden angle spectrum produced",
      spectrum.get("field_size", 0) > 0,
      f"field_size={spectrum.get('field_size')}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 14: Berry Phase Calculator
# ═══════════════════════════════════════════════════════════════════════════════

phase(14, "BERRY PHASE — CALCULATOR")

bp_calc = BerryPhaseCalculator()

# Spin-1/2 Berry phase: hemisphere → γ = -π
spin_result = bp_calc.spin_half_berry_phase(2 * math.pi)
check("Spin-1/2 hemisphere: γ = -π",
      abs(spin_result.phase - (-math.pi)) < 1e-10,
      f"got γ={spin_result.phase:.8f}, expected -π={-math.pi:.8f}")

# Spin-1/2: full sphere → γ = -2π → normalized to 0
spin_full = bp_calc.spin_half_berry_phase(4 * math.pi)
# -4π/2 = -2π, normalized to [-π,π] → 0
check("Spin-1/2 full sphere: γ ≈ 0 (mod 2π)",
      abs(spin_full.phase) < 1e-10 or abs(abs(spin_full.phase) - 2*math.pi) < 1e-10,
      f"got γ={spin_full.phase:.8f}")

# Discrete Berry phase: create a loop of states on Bloch sphere
n_pts = 100
states = []
for i in range(n_pts):
    theta = math.pi / 3  # cone half-angle
    phi_angle = 2 * math.pi * i / n_pts
    state = np.array([
        math.cos(theta / 2),
        math.sin(theta / 2) * cmath.exp(1j * phi_angle)
    ])
    states.append(state)

disc_result = bp_calc.discrete_berry_phase(states)
expected_gamma = -math.pi * (1 - math.cos(math.pi / 3))  # solid angle / 2
check(f"Discrete Berry phase for cone: γ ≈ {expected_gamma:.4f}",
      abs(disc_result.phase - expected_gamma) < 0.1,
      f"got γ={disc_result.phase:.4f}, expected ≈{expected_gamma:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 15: Berry Phase — Molecular (Conical Intersection)
# ═══════════════════════════════════════════════════════════════════════════════

phase(15, "BERRY PHASE — MOLECULAR (CONICAL INTERSECTION)")

import cmath

mol = MolecularBerryPhase()

# Enclosing CI → γ = π
ci_enclosed = mol.conical_intersection_phase(True)
check("CI enclosed: γ = π",
      abs(ci_enclosed.phase - math.pi) < 1e-10,
      f"got γ={ci_enclosed.phase}")

# Not enclosing → γ = 0
ci_none = mol.conical_intersection_phase(False)
check("CI not enclosed: γ = 0",
      abs(ci_none.phase) < 1e-10,
      f"got γ={ci_none.phase}")

# Two-level conical model: explicit calculation should give π
two_level = mol.two_level_conical_model(200)
check("Two-level conical: γ ≈ π",
      abs(abs(two_level.phase) - math.pi) < 0.05,
      f"got |γ|={abs(two_level.phase):.4f}, expected π={math.pi:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 16: Berry Phase — Aharonov-Bohm
# ═══════════════════════════════════════════════════════════════════════════════

phase(16, "BERRY PHASE — AHARONOV-BOHM")

ab = AharonovBohmEngine()

# 1 flux quantum → γ = 2π
flux_quantum = PC.H / PC.Q_E
ab_result = ab.aharonov_bohm_phase(flux_quantum)
check("1 flux quantum: γ = 2π",
      abs(ab_result.phase - 2 * math.pi) < 1e-6,
      f"got γ={ab_result.phase:.6f}, expected 2π={2*math.pi:.6f}")

# 0 flux → γ = 0
ab_zero = ab.aharonov_bohm_phase(0.0)
check("0 flux: γ = 0",
      abs(ab_zero.phase) < 1e-10,
      f"got γ={ab_zero.phase}")

# Flux quantization
fq = ab.flux_quantization(1)
check("SC flux quantum = h/2e",
      abs(fq["sc_flux_quantum_Wb"] - PC.H / (2 * PC.Q_E)) < 1e-25,
      f"got {fq['sc_flux_quantum_Wb']}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 17: Berry Phase — Pancharatnam (Polarization)
# ═══════════════════════════════════════════════════════════════════════════════

phase(17, "BERRY PHASE — PANCHARATNAM")

pan = PancharatnamPhase()

# Orthogonal Stokes: equilateral triangle on Poincaré sphere
S1 = np.array([1, 0, 0], dtype=float)
S2 = np.array([0, 1, 0], dtype=float)
S3 = np.array([0, 0, 1], dtype=float)

pan_result = pan.geodesic_triangle_phase([S1, S2, S3])
# Solid angle of octant = π/2, so γ = π/4
check("Pancharatnam for orthogonal triple: γ ≈ π/4",
      abs(pan_result.phase - math.pi / 4) < 0.1,
      f"got γ={pan_result.phase:.4f}, expected π/4={math.pi/4:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 18: Computronium Subsystem
# ═══════════════════════════════════════════════════════════════════════════════

phase(18, "COMPUTRONIUM — FUNDAMENTAL LIMITS")

comp = ComputroniumSubsystem()

# Bremermann limit
br = comp.bremermann_limit(1.0)
check("Bremermann bits/s > 10^49",
      br.get("max_bits_per_sec", 0) > 1e49,
      f"got {br.get('max_bits_per_sec', 0):.3e}")

# Margolus-Levitin
ml = comp.margolus_levitin(mass_kg=1.0)
check("ML ops/s > 10^49",
      ml.get("max_ops_per_sec", 0) > 1e49,
      f"got {ml.get('max_ops_per_sec', 0):.3e}")

# ML = 2× Bremermann for same mass
check("ML = 2× Bremermann",
      abs(ml.get("max_ops_per_sec", 0) / br.get("max_bits_per_sec", 1) - 2.0) < 0.01,
      f"ratio={ml.get('max_ops_per_sec', 0) / br.get('max_bits_per_sec', 1):.4f}")

# Landauer at room temp
land = comp.landauer_erasure(293.15, 1)
check("Landauer energy ≈ 2.8e-21 J/bit",
      abs(land.get("energy_per_bit_J", 0) - 2.805e-21) < 1e-23,
      f"got {land.get('energy_per_bit_J', 0):.4e}")

# Bekenstein bound
bek = comp.bekenstein_bound(1.0, mass_kg=1.0)
check("Bekenstein bits > 10^42",
      bek.get("max_bits", 0) > 1e42,
      f"got {bek.get('max_bits', 0):.3e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 19: Engine Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

phase(19, "ENGINE ORCHESTRATOR")

se = science_engine

check("Engine version is 5.1.0",
      se.VERSION == "5.1.0", f"got {se.VERSION}")

status = se.get_full_status()
check("Full status has all subsystems",
      all(k in status for k in ["physics", "entropy", "multidim", "coherence",
                                  "berry_phase", "computronium", "quantum_circuit"]),
      f"keys={list(status.keys())}")

# Research cycle
research = se.perform_research_cycle("ADVANCED_PHYSICS")
check("Research cycle produces resonance",
      "resonance_alignment" in research,
      f"keys={list(research.keys())}")

# 512MB validation through engine
val_512 = se.validate_512mb()
check("Engine validate_512mb() returns dict",
      isinstance(val_512, dict), f"got {type(val_512)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 20: Cross-Module Constant Consistency
# ═══════════════════════════════════════════════════════════════════════════════

phase(20, "CROSS-MODULE CONSTANT CONSISTENCY")

# Check GOD_CODE is the same everywhere it's used
from l104_science_engine.constants import GOD_CODE as c_gc
from l104_science_engine.physics import GOD_CODE as p_gc
from l104_science_engine.entropy import GOD_CODE as e_gc
from l104_science_engine.coherence import GOD_CODE as co_gc
from l104_science_engine.bridge import GOD_CODE as b_gc

check("GOD_CODE identical: constants == physics",
      c_gc == p_gc, f"constants={c_gc}, physics={p_gc}")
check("GOD_CODE identical: constants == entropy",
      c_gc == e_gc, f"constants={c_gc}, entropy={e_gc}")
check("GOD_CODE identical: constants == coherence",
      c_gc == co_gc, f"constants={c_gc}, coherence={co_gc}")
check("GOD_CODE identical: constants == bridge",
      c_gc == b_gc, f"constants={c_gc}, bridge={b_gc}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 21: Entropy Quantum Equations (Q2, Q3, Q5)
# ═══════════════════════════════════════════════════════════════════════════════

phase(21, "ENTROPY QUANTUM EQUATIONS")

# Q2: cascade trajectory
traj = entropy.entropy_cascade_trajectory(5.0)
sorted_frac = traj.get("sorted_fraction", 0)
check("Q2: sorted_fraction ∈ (0, 1)",
      0 < sorted_frac < 1,
      f"sorted_fraction={sorted_frac}")

check("Q2: S_final < S_0",
      traj.get("S_final", 10) < traj.get("S_0", 10),
      f"S_0={traj.get('S_0')}, S_final={traj.get('S_final')}")

# Q3: void energy equilibrium
void_eq = entropy.void_energy_equilibrium(1.0, 30)
check("Q3: equilibrium converges",
      void_eq.get("converged", False) == True,
      f"converged={void_eq.get('converged')}, V∞={void_eq.get('V_infinity_simulated')}")

check("Q3: bounded",
      void_eq.get("bounded", False) == True,
      f"bounded={void_eq.get('bounded')}")

# Q5: ZNE analysis
zne = entropy.zne_analysis(2.0)
check("Q5: ZNE boost > 1",
      zne.get("zne_boost_factor", 0) > 1.0,
      f"boost={zne.get('zne_boost_factor')}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 22: Topological Constants Validation
# ═══════════════════════════════════════════════════════════════════════════════

phase(22, "TOPOLOGICAL CONSTANTS")

from l104_science_engine.constants import (
    TOPOLOGICAL_CORRELATION_LENGTH,
    TOPOLOGICAL_ERROR_RATE_D8,
    TOPOLOGICAL_ERROR_RATE_D13,
    FIBONACCI_BRAID_PHASE,
    FIBONACCI_F_MATRIX_ENTRY,
)

check("Topo correlation length = 1/φ",
      abs(TOPOLOGICAL_CORRELATION_LENGTH - PHI_CONJUGATE) < 1e-15,
      f"got {TOPOLOGICAL_CORRELATION_LENGTH}")

check("Topo error D=8 ≈ 2.39e-06",
      abs(TOPOLOGICAL_ERROR_RATE_D8 - math.exp(-8 / PHI_CONJUGATE)) < 1e-15,
      f"got {TOPOLOGICAL_ERROR_RATE_D8}")

check("Topo error D=13 ≈ 7.33e-10",
      abs(TOPOLOGICAL_ERROR_RATE_D13 - math.exp(-13 / PHI_CONJUGATE)) < 1e-15,
      f"got {TOPOLOGICAL_ERROR_RATE_D13}")

check("Fibonacci braid phase = 4π/5",
      abs(FIBONACCI_BRAID_PHASE - 4 * math.pi / 5) < 1e-15,
      f"got {FIBONACCI_BRAID_PHASE}")

check("Fibonacci F-matrix entry = 1/φ",
      abs(FIBONACCI_F_MATRIX_ENTRY - PHI_CONJUGATE) < 1e-15,
      f"got {FIBONACCI_F_MATRIX_ENTRY}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 23: Unitary Quantization Model
# ═══════════════════════════════════════════════════════════════════════════════

phase(23, "UNITARY QUANTIZATION MODEL")

from l104_science_engine.constants import (
    UNITARY_PHASE_STEP, UNITARY_SEMITONE, UNITARY_FOUR_OCTAVE,
    FIBONACCI_7, FACTOR_13_SCAFFOLD, FACTOR_13_GRAIN, FACTOR_13_OFFSET,
    DIAL_BITS_A, DIAL_BITS_B, DIAL_BITS_C, DIAL_BITS_D,
    DIAL_TOTAL_QUBITS, DIAL_CONFIGURATIONS,
)

check("Unitary phase step = 2^(1/104)",
      abs(UNITARY_PHASE_STEP - STEP_SIZE) < 1e-15,
      f"got {UNITARY_PHASE_STEP}")

check("Four-octave multiplier = 16",
      abs(UNITARY_FOUR_OCTAVE - 16.0) < 1e-10,
      f"got {UNITARY_FOUR_OCTAVE}")

check("Fibonacci(7) = 13",
      FIBONACCI_7 == 13, f"got {FIBONACCI_7}")

check("286/13 = 22",
      FACTOR_13_SCAFFOLD == 22, f"got {FACTOR_13_SCAFFOLD}")

check("104/13 = 8",
      FACTOR_13_GRAIN == 8, f"got {FACTOR_13_GRAIN}")

check("Dial total = 14 qubits",
      DIAL_TOTAL_QUBITS == 14, f"got {DIAL_TOTAL_QUBITS}")

check("Dial configs = 2^14 = 16384",
      DIAL_CONFIGURATIONS == 16384, f"got {DIAL_CONFIGURATIONS}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 24: Chaos Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

phase(24, "CHAOS DIAGNOSTICS")

# Stable signal
stable = [GOD_CODE + 0.001 * i for i in range(100)]
diag = entropy.chaos_diagnostics(stable)
check("Stable signal → HEALTHY",
      diag.get("health", "") == "HEALTHY",
      f"health={diag.get('health')}, lyapunov={diag.get('lyapunov_exponent')}")

# Chaotic signal
import random
random.seed(42)
chaotic = [GOD_CODE * (1 + 0.5 * random.gauss(0, 1)) for _ in range(100)]
diag_c = entropy.chaos_diagnostics(chaotic)
check("Chaotic signal → not HEALTHY",
      diag_c.get("health", "HEALTHY") != "HEALTHY",
      f"health={diag_c.get('health')}, amplitude={diag_c.get('relative_amplitude')}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 25: Demon vs Chaos
# ═══════════════════════════════════════════════════════════════════════════════

phase(25, "DEMON VS CHAOS")

chaos_products = [GOD_CODE * (1 + 0.1 * math.sin(i * 0.3)) for i in range(50)]
dvc = entropy.demon_vs_chaos(chaos_products)

check("Demon improvement > 0%",
      dvc.get("demon_improvement_pct", 0) > 0,
      f"improvement={dvc.get('demon_improvement_pct')}%")

check("Phi improvement > 0%",
      dvc.get("phi_improvement_pct", 0) > 0,
      f"improvement={dvc.get('phi_improvement_pct')}%")


# ═══════════════════════════════════════════════════════════════════════════════
#  FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*70}")
print(f"  FINAL REPORT — l104_science_engine/ v5.1.0 DEEP DEBUG")
print(f"{'═'*70}")
print(f"  PASSED : {PASS}")
print(f"  FAILED : {FAIL}")
print(f"  WARNED : {WARN}")
print(f"  TOTAL  : {PASS + FAIL}")
print(f"{'═'*70}")

if BUGS:
    print(f"\n  🐛 BUGS FOUND ({len(BUGS)}):")
    for i, (name, detail) in enumerate(BUGS, 1):
        print(f"    {i}. {name}")
        print(f"       → {detail}")

verdict = "SOVEREIGN" if FAIL == 0 else "NEEDS FIX" if FAIL <= 5 else "CRITICAL"
print(f"\n  VERDICT: {verdict}")
print(f"{'═'*70}\n")

sys.exit(0 if FAIL == 0 else 1)
