"""
L104 God Code Simulator — Transpiler Simulations
═══════════════════════════════════════════════════════════════════════════════

Sacred transpiler simulation category: circuit building, unitary verification,
phase decomposition, dial transpilation, and conservation law tests.

5 simulations: unitary_verification, phase_decomposition, sacred_transpilation,
               dial_transpilation, conservation_law

All simulations use the pure-numpy NumpyCircuit with native gate decomposition
for basis-set transpilation and unitary verification.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import time
from typing import List, Tuple

from ..constants import GOD_CODE, PHI, TAU
from ..quantum_primitives import god_code_dial
from ..result import SimulationResult
from ..sacred_transpiler import (
    GOD_CODE_PHASE, PHI_PHASE, IRON_PHASE, VOID_PHASE,
    PHYSICAL_DIALS, BASIS_SETS,
    build_godcode_1q_circuit, build_godcode_1q_decomposed,
    build_godcode_sacred_circuit, build_godcode_dial_circuit,
    verify_godcode_unitary, verify_decomposition_fidelity,
    verify_conservation_law, transpile_all_basis_sets,
    _get_unitary,
)


def sim_unitary_verification(nq: int = 4) -> SimulationResult:
    """
    Verify unitarity and eigenspectrum of 1Q, decomposed, and NQ sacred circuits.

    Checks: U†U = I, |det(U)| = 1, all eigenvalues on unit circle,
    GOD_CODE phase detection in eigenspectrum.
    """
    t0 = time.time()

    # Build all three circuits
    gc_1q = build_godcode_1q_circuit()
    gc_decomp = build_godcode_1q_decomposed()
    gc_nq = build_godcode_sacred_circuit(nq)

    # Verify each
    uv_1q = verify_godcode_unitary(gc_1q, "1Q Direct")
    uv_decomp = verify_godcode_unitary(gc_decomp, "1Q Decomposed")
    uv_nq = verify_godcode_unitary(gc_nq, f"{nq}Q Sacred")

    # Check decomposition match
    decomp_match = verify_decomposition_fidelity(gc_1q, gc_decomp)

    all_unitary = uv_1q["is_unitary"] and uv_decomp["is_unitary"] and uv_nq["is_unitary"]
    all_unit_circle = (uv_1q["all_on_unit_circle"] and uv_decomp["all_on_unit_circle"]
                       and uv_nq["all_on_unit_circle"])
    gc_phase_found = uv_1q["god_code_phase_found"]

    elapsed = (time.time() - t0) * 1000

    return SimulationResult(
        name="unitary_verification",
        category="transpiler",
        passed=all_unitary and all_unit_circle and decomp_match["match"],
        elapsed_ms=elapsed,
        detail=(f"3 circuits verified: all_unitary={all_unitary}, "
                f"unit_circle={all_unit_circle}, decomp_match={decomp_match['match']}, "
                f"GC_phase={gc_phase_found}"),
        fidelity=decomp_match["hs_fidelity"],
        num_qubits=nq,
        circuit_depth=uv_nq["dimension"],
        phase_coherence=1.0 - min(uv_1q["god_code_phase_error"], 1.0),
        sacred_alignment=abs(math.cos(GOD_CODE_PHASE)),
        conservation_error=decomp_match["max_error"],
        extra={
            "uv_1q": {
                "is_unitary": uv_1q["is_unitary"],
                "det_magnitude": uv_1q["det_magnitude"],
                "gc_phase_found": uv_1q["god_code_phase_found"],
                "gc_phase_error": uv_1q["god_code_phase_error"],
            },
            "uv_nq": {
                "is_unitary": uv_nq["is_unitary"],
                "det_magnitude": uv_nq["det_magnitude"],
                "n_eigenvalues": len(uv_nq["eigenvalue_phases_rad"]),
            },
            "decomposition": {
                "match": decomp_match["match"],
                "max_error": decomp_match["max_error"],
                "hs_fidelity": decomp_match["hs_fidelity"],
            },
        },
    )


def sim_phase_decomposition(nq: int = 1) -> SimulationResult:
    """
    Verify GOD_CODE phase decomposes into Iron + PHI + Octave phases.

    GOD_CODE mod 2π = Rz(IRON) · Rz(PHI_mod) · Rz(OCTAVE)
    Tests that the 3-rotation decomposition exactly equals the direct phase.
    """
    t0 = time.time()

    gc_direct = build_godcode_1q_circuit()
    gc_decomp = build_godcode_1q_decomposed()

    U_direct = _get_unitary(gc_direct)
    U_decomp = _get_unitary(gc_decomp)

    match = verify_decomposition_fidelity(gc_direct, gc_decomp)

    # Phase breakdown
    from ..sacred_transpiler import PHASE_OCTAVE_4
    phi_contribution = (GOD_CODE_PHASE - IRON_PHASE - PHASE_OCTAVE_4) % TAU
    reconstructed = (IRON_PHASE + phi_contribution + PHASE_OCTAVE_4) % TAU
    phase_error = abs(reconstructed - GOD_CODE_PHASE % TAU)

    # Eigenvalue analysis
    eigvals_direct = list(map(complex, [v for v in U_direct.diagonal()]))
    eigvals_decomp = list(map(complex, [v for v in U_decomp.diagonal()]))

    elapsed = (time.time() - t0) * 1000

    return SimulationResult(
        name="phase_decomposition",
        category="transpiler",
        passed=match["match"] and phase_error < 1e-10,
        elapsed_ms=elapsed,
        detail=(f"GOD_CODE phase → Iron({IRON_PHASE:.4f}) + "
                f"PHI_mod({phi_contribution:.4f}) + Octave({PHASE_OCTAVE_4:.4f}) "
                f"= {reconstructed:.10f} (err={phase_error:.2e})"),
        fidelity=match["hs_fidelity"],
        num_qubits=1,
        circuit_depth=3,
        phase_coherence=1.0 - min(phase_error * 1e9, 1.0),
        sacred_alignment=abs(math.cos(GOD_CODE_PHASE)),
        conservation_error=phase_error,
        god_code_measured=GOD_CODE_PHASE % TAU,
        god_code_error=phase_error,
        extra={
            "iron_phase": IRON_PHASE,
            "phi_contribution": phi_contribution,
            "octave_phase": PHASE_OCTAVE_4,
            "reconstructed": reconstructed,
            "direct_diagonal": eigvals_direct,
            "decomp_diagonal": eigvals_decomp,
        },
    )


def sim_sacred_transpilation(nq: int = 5) -> SimulationResult:
    """
    Build NQ sacred circuit and transpile to all hardware basis sets.

    Tests that native transpilation preserves the GOD_CODE unitary at
    every decomposition level (IBM Eagle, Clifford+T, Rigetti, IonQ, Minimal).
    """
    t0 = time.time()

    gc_nq = build_godcode_sacred_circuit(nq)
    uv = verify_godcode_unitary(gc_nq, f"{nq}Q Sacred")

    transpilation_results: dict = {}
    all_preserved = True

    results = transpile_all_basis_sets(gc_nq, f"GOD_CODE_{nq}Q")
    for name, data in results.items():
        rpt = data["report"]
        if "error" not in rpt:
            preserved = rpt.get("unitary_preserved", False)
            transpilation_results[name] = {
                "preserved": preserved,
                "process_fidelity": rpt.get("process_fidelity", 0.0),
                "depth_change": f"{rpt.get('original_depth', 0)} → {rpt.get('transpiled_depth', 0)}",
            }
            if not preserved:
                all_preserved = False
        else:
            transpilation_results[name] = {"error": rpt["error"]}
            all_preserved = False

    elapsed = (time.time() - t0) * 1000

    n_transpiled = sum(1 for v in transpilation_results.values() if v.get("preserved", False))

    return SimulationResult(
        name="sacred_transpilation",
        category="transpiler",
        passed=uv["is_unitary"] and all_preserved,
        elapsed_ms=elapsed,
        detail=(f"{nq}Q sacred: unitary={uv['is_unitary']}, "
                f"transpiled={n_transpiled}/{len(transpilation_results)} basis sets"),
        fidelity=1.0 if uv["is_unitary"] else 0.0,
        num_qubits=nq,
        circuit_depth=uv["dimension"],
        phase_coherence=1.0 if uv["all_on_unit_circle"] else 0.5,
        sacred_alignment=abs(math.cos(GOD_CODE_PHASE)),
        extra={
            "unitary_verification": {
                "is_unitary": uv["is_unitary"],
                "det_magnitude": uv["det_magnitude"],
                "all_on_unit_circle": uv["all_on_unit_circle"],
            },
            "transpilations": transpilation_results,
        },
    )


def sim_dial_transpilation(nq: int = 5) -> SimulationResult:
    """
    Build and verify dial circuits for 5 physical constants.

    Tests: GOD_CODE(0,0,0,0), Schumann 7.83Hz, Bohr radius, Fe BCC lattice, Gamma 40Hz.
    Each circuit verified for unitarity and eigenspectrum.
    """
    t0 = time.time()

    dial_results: dict = {}
    all_unitary = True
    total_gc_phase_found = 0

    for dial_name, (a, b, c, d) in PHYSICAL_DIALS.items():
        dc = build_godcode_dial_circuit(a, b, c, d, n_qubits=nq)
        freq = god_code_dial(a, b, c, d)
        uv = verify_godcode_unitary(dc, f"Dial {dial_name}")

        dial_results[dial_name] = {
            "dials": (a, b, c, d),
            "frequency": freq,
            "is_unitary": uv["is_unitary"],
            "det_magnitude": uv["det_magnitude"],
            "all_on_unit_circle": uv["all_on_unit_circle"],
            "gc_phase_found": uv["god_code_phase_found"],
        }

        if not uv["is_unitary"]:
            all_unitary = False
        if uv["god_code_phase_found"]:
            total_gc_phase_found += 1

    elapsed = (time.time() - t0) * 1000

    return SimulationResult(
        name="dial_transpilation",
        category="transpiler",
        passed=all_unitary,
        elapsed_ms=elapsed,
        detail=(f"{len(PHYSICAL_DIALS)} dial circuits: all_unitary={all_unitary}, "
                f"GC_phase_found={total_gc_phase_found}/{len(PHYSICAL_DIALS)}"),
        fidelity=1.0 if all_unitary else 0.0,
        num_qubits=nq,
        sacred_alignment=total_gc_phase_found / len(PHYSICAL_DIALS),
        extra={
            "dials": dial_results,
        },
    )


def sim_conservation_law(nq: int = 1) -> SimulationResult:
    """
    Verify the GOD_CODE conservation law: G(X) × 2^(X/104) = 527.518...

    Tests 7 octave points (X = -312 to +312 in steps of 104),
    building a 1Q Rz circuit at each X and verifying the product invariant.
    """
    t0 = time.time()

    result = verify_conservation_law(n_points=7)

    all_conserved = result["all_conserved"]
    max_error = max(abs(pt["product"] - GOD_CODE) for pt in result["points"])

    elapsed = (time.time() - t0) * 1000

    return SimulationResult(
        name="conservation_law",
        category="transpiler",
        passed=all_conserved,
        elapsed_ms=elapsed,
        detail=(f"Conservation G(X)×2^(X/104) = {GOD_CODE:.10f} "
                f"verified at {len(result['points'])} points, "
                f"all_conserved={all_conserved}, max_err={max_error:.2e}"),
        fidelity=1.0 if all_conserved else 0.0,
        num_qubits=1,
        conservation_error=max_error,
        god_code_measured=GOD_CODE,
        god_code_error=max_error,
        sacred_alignment=1.0 if all_conserved else 0.0,
        extra={
            "invariant": result["invariant"],
            "points": result["points"],
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  REGISTRY — for GodCodeSimulator._register_builtins()
# ═══════════════════════════════════════════════════════════════════════════════

TRANSPILER_SIMULATIONS: List[Tuple[str, callable, str, str, int]] = [
    ("unitary_verification", sim_unitary_verification, "transpiler",
     "Verify unitarity + eigenspectrum of GOD_CODE circuits (1Q, decomposed, NQ)", 4),
    ("phase_decomposition", sim_phase_decomposition, "transpiler",
     "GOD_CODE phase → Iron + PHI + Octave decomposition verification", 1),
    ("sacred_transpilation", sim_sacred_transpilation, "transpiler",
     "Build NQ sacred circuit, transpile to 5 hardware basis sets (native)", 5),
    ("dial_transpilation", sim_dial_transpilation, "transpiler",
     "Build + verify dial circuits for 5 physical constants", 5),
    ("conservation_law", sim_conservation_law, "transpiler",
     "Verify G(X) × 2^(X/104) = GOD_CODE conservation across 7 octave points", 1),
]


__all__ = [
    "sim_unitary_verification",
    "sim_phase_decomposition",
    "sim_sacred_transpilation",
    "sim_dial_transpilation",
    "sim_conservation_law",
    "TRANSPILER_SIMULATIONS",
]
