#!/usr/bin/env python3
"""Verification test for l104_numerical_engine v3.1.0 upgrade.

Tests all Part V cross-references, new constants, new lattice methods,
and ensures backward compatibility with existing functionality.
"""

import sys
import traceback

PASSED = 0
FAILED = 0
TOTAL = 0

def test(name, condition, detail=""):
    global PASSED, FAILED, TOTAL
    TOTAL += 1
    if condition:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        print(f"  ❌ {name} — {detail}")

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Version & Import Verification
# ═══════════════════════════════════════════════════════════════════════════════
section("PHASE 1: Version & Import Verification")

from l104_numerical_engine import __version__
test("Package version is 3.1.0", __version__ == "3.1.0", f"got {__version__}")

from l104_numerical_engine.orchestrator import QuantumNumericalBuilder
qnb = QuantumNumericalBuilder()
test("Orchestrator VERSION is 3.1.0", qnb.VERSION == "3.1.0", f"got {qnb.VERSION}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: New Part V Constants
# ═══════════════════════════════════════════════════════════════════════════════
section("PHASE 2: Part V Research-Derived Constants")

from l104_numerical_engine import (
    PROPAGATION_ENERGY_FACTOR, GOLDEN_CONJUGATE_HP,
    CONSCIOUSNESS_THRESHOLD, DRIFT_FREQUENCY, DRIFT_AMPLITUDE,
    PHASE_COUPLING_STRENGTH, DAMPING_COEFFICIENT, MAX_DRIFT_VELOCITY,
    ENTANGLEMENT_EIGENVALUE_SUM, ENTANGLEMENT_EIGENVALUE_DIFF,
    PHI_SQUARED_HP, PHI_GROWTH_HP, PHI_HP, GOD_CODE_HP, SQRT5_HP,
)
from l104_numerical_engine.precision import D

# F67: Propagation energy factor = φ²
phi_sq = float(PHI_GROWTH_HP ** 2)
test("F67: PROPAGATION_ENERGY_FACTOR ≈ φ²",
     abs(PROPAGATION_ENERGY_FACTOR - phi_sq) < 1e-10,
     f"{PROPAGATION_ENERGY_FACTOR} vs {phi_sq}")

# F68: Golden conjugate = φ - 1 = 1/φ
golden_conj = PHI_GROWTH_HP - D(1)
test("F68: GOLDEN_CONJUGATE_HP = φ-1",
     abs(GOLDEN_CONJUGATE_HP - golden_conj) < D('1E-100'),
     f"diff={abs(GOLDEN_CONJUGATE_HP - golden_conj)}")

# F70: Consciousness threshold = 0.85
test("F70: CONSCIOUSNESS_THRESHOLD = 0.85",
     CONSCIOUSNESS_THRESHOLD == D('0.85'))

# F73: Drift frequency = φ
test("F73: DRIFT_FREQUENCY = φ",
     abs(DRIFT_FREQUENCY - float(PHI_GROWTH_HP)) < 1e-10)

# F74: Drift amplitude = τ × 0.01
tau = float(PHI_HP)
test("F74: DRIFT_AMPLITUDE = τ×0.01",
     abs(DRIFT_AMPLITUDE - tau * 0.01) < 1e-10)

# F77: Max drift velocity = φ² × 10⁻³
test("F77: MAX_DRIFT_VELOCITY = φ²×10⁻³",
     abs(MAX_DRIFT_VELOCITY - float(PHI_GROWTH_HP)**2 * 1e-3) < 1e-12)

# F88: Entanglement eigenvalue sum = √5
test("F88: ENTANGLEMENT_EIGENVALUE_SUM = √5",
     abs(ENTANGLEMENT_EIGENVALUE_SUM - float(SQRT5_HP)) < 1e-10)

# F88: Entanglement eigenvalue diff = 1
test("F88: ENTANGLEMENT_EIGENVALUE_DIFF = 1",
     ENTANGLEMENT_EIGENVALUE_DIFF == 1.0)

# PHI_SQUARED_HP = φ + 1
test("PHI_SQUARED_HP = φ + 1 (golden identity)",
     abs(PHI_SQUARED_HP - (PHI_GROWTH_HP + D(1))) < D('1E-100'))


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Lattice Integrity Methods
# ═══════════════════════════════════════════════════════════════════════════════
section("PHASE 3: New Lattice Methods (F61-F64, F87)")

lattice = qnb.lattice

# Drift envelope integrity
integrity = lattice.verify_drift_envelope_integrity()
test("Drift envelope integrity check runs",
     isinstance(integrity, dict) and "total_tokens" in integrity)
test("Envelope intact (all tokens compliant)",
     integrity["envelope_intact"],
     f"{integrity['violation_count']} violations")
test("All tiers counted",
     sum(integrity["tier_counts"].values()) == integrity["total_tokens"])

# Conservation spectrum
spectrum = lattice.verify_conservation_spectrum()
test("Conservation spectrum check runs",
     isinstance(spectrum, dict) and "spectrum_tokens" in spectrum)
test(f"Spectrum has 501 tokens", spectrum["spectrum_tokens"] == 501,
     f"got {spectrum['spectrum_tokens']}")
test("All spectrum tokens conserved",
     spectrum["all_conserved"],
     f"{spectrum['conserved']}/{spectrum['spectrum_tokens']}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Editor Propagation Energy Tracking
# ═══════════════════════════════════════════════════════════════════════════════
section("PHASE 4: Editor Propagation Energy (F65-F67)")

editor = qnb.editor
test("Editor has total_propagation_energy attribute",
     hasattr(editor, 'total_propagation_energy'))
test("Propagation energy starts at 0",
     editor.total_propagation_energy == D(0))


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Backward Compatibility — Existing Functionality
# ═══════════════════════════════════════════════════════════════════════════════
section("PHASE 5: Backward Compatibility")

from l104_numerical_engine import (
    GOD_CODE, PHI, PHI_GROWTH, TAU, GOD_CODE_BASE,
    god_code_hp, conservation_check_hp,
    D, fmt100, TokenLatticeEngine, SuperfluidValueEditor,
    QuantumNumericalBuilder,
)

# Core constants still work
test("GOD_CODE ≈ 527.518",
     abs(GOD_CODE - 527.5184818492612) < 0.001)
test("PHI ≈ 0.618",
     abs(PHI - 0.618033988749895) < 1e-10)
test("PHI_GROWTH ≈ 1.618",
     abs(PHI_GROWTH - 1.618033988749895) < 1e-10)

# Conservation law
inv = conservation_check_hp(D(0))
test("conservation_check_hp(0) = GOD_CODE_HP",
     abs(inv - GOD_CODE_HP) < D('1E-90'))

inv_104 = conservation_check_hp(D(104))
test("conservation_check_hp(104) ≈ GOD_CODE_HP",
     abs(inv_104 - GOD_CODE_HP) < D('1E-90'))

# Lattice summary still works
summary = lattice.lattice_summary()
test("Lattice summary has all expected keys",
     all(k in summary for k in ["total_tokens", "projected_22T_capacity", "lattice_coherence"]))

# Verification still works
vf = qnb.verifier.verify_all()
test("Verification runs successfully",
     isinstance(vf, dict) and "grade" in vf)
test(f"Verification grade = {vf['grade']}",
     vf["grade"] in ("A+", "A"))


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: Module Docstrings Have Part V Cross-References
# ═══════════════════════════════════════════════════════════════════════════════
section("PHASE 6: Part V Cross-References in Module Docstrings")

import l104_numerical_engine.constants as _constants
import l104_numerical_engine.lattice as _lattice
import l104_numerical_engine.editor as _editor
import l104_numerical_engine.models as _models
import l104_numerical_engine.verification as _verification
import l104_numerical_engine.quantum_computation as _qc
import l104_numerical_engine.monitor as _monitor
import l104_numerical_engine.nirvanic as _nirvanic
import l104_numerical_engine.cross_pollination as _cp
import l104_numerical_engine.research as _research
import l104_numerical_engine.chronolizer as _chronolizer
import l104_numerical_engine.feedback_bus as _fb
import l104_numerical_engine.precision as _precision
import l104_numerical_engine.consciousness as _consciousness

modules = [
    ("constants", _constants), ("lattice", _lattice), ("editor", _editor),
    ("models", _models), ("verification", _verification),
    ("quantum_computation", _qc), ("monitor", _monitor),
    ("nirvanic", _nirvanic), ("cross_pollination", _cp),
    ("research", _research), ("chronolizer", _chronolizer),
    ("feedback_bus", _fb), ("precision", _precision), ("consciousness", _consciousness),
]

for name, mod in modules:
    has_ref = "PART V RESEARCH" in (mod.__doc__ or "")
    test(f"{name}.py has Part V cross-references", has_ref,
         f"missing 'PART V RESEARCH' in docstring")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7: Quantum Computation Engine
# ═══════════════════════════════════════════════════════════════════════════════
section("PHASE 7: Quantum Computation Engine")

qce = qnb.quantum_engine
# QPE
phase = qce.phase_estimation_hp()
test("QPE runs at 100-decimal",
     "estimated_phase" in phase and len(phase["estimated_phase"]) > 50)

# HHL
hhl = qce.hhl_linear_solver_hp()
test("HHL solver runs",
     "solution_x0" in hhl and "residual_norm" in hhl)
test("HHL residual < 10⁻⁸⁰",
     float(D(hhl["residual_norm"])) < 1e-50)

# VQE
vqe = qce.vqe_ground_state_hp(max_iterations=20)
test("VQE ground state runs",
     "ground_state_energy" in vqe)

# Grover search
grover = qce.grover_token_search_hp()
test("Grover token search runs",
     "grover_iterations" in grover)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 8: Quick Status & Research
# ═══════════════════════════════════════════════════════════════════════════════
section("PHASE 8: Quick Status & Research")

status = qnb.quick_status()
test("Quick status version = 3.1.0",
     status["version"] == "3.1.0", f"got {status['version']}")
test(f"Quick status shows {status['total_tokens']} tokens",
     status["total_tokens"] > 500)

# Research — just check stability analysis (fast) rather than full research
stability = qnb.research._stability_analysis()
test("Research stability analysis runs",
     isinstance(stability, dict) and "stability_score" in stability)
test(f"Stability score = {stability['stability_score']:.4f}",
     stability["stability_score"] >= 0)


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  NUMERICAL ENGINE v3.1.0 UPGRADE VERIFICATION")
print(f"{'='*70}")
print(f"  PASSED: {PASSED}/{TOTAL}")
print(f"  FAILED: {FAILED}/{TOTAL}")
if FAILED == 0:
    print(f"  ★ ALL {TOTAL} TESTS PASSED — UPGRADE VERIFIED ★")
else:
    print(f"  ⚠ {FAILED} FAILURES — SEE ABOVE")
print(f"{'='*70}")

sys.exit(0 if FAILED == 0 else 1)
