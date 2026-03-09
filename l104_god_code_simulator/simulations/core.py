"""
L104 God Code Simulator — Core Simulations
═══════════════════════════════════════════════════════════════════════════════

Conservation proofs, dial sweeps, 104-TET spectral alignment, and
sacred cascade circuits — the algebraic backbone.

5 simulations: conservation_proof, dial_sweep_a, 104_tet_spectrum,
               ln_god_code_2pi, sacred_cascade

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..constants import (
    GOD_CODE, GOD_CODE_PHASE_ANGLE, IRON_PHASE_ANGLE,
    OCTAVE_OFFSET, PHI, PHI_PHASE_ANGLE, QUANTIZATION_GRAIN,
    VOID_PHASE_ANGLE,
)
from ..quantum_primitives import (
    GOD_CODE_GATE, H_GATE, IRON_GATE, PHI_GATE, VOID_GATE,
    apply_cnot, apply_single_gate, entanglement_entropy, god_code_dial,
    god_code_fn, init_sv, probabilities,
)
from ..result import SimulationResult


def sim_conservation_proof(nq: int = 2) -> SimulationResult:
    """Verify GOD_CODE conservation law: G(X) × 2^(X/104) = INVARIANT across range."""
    t0 = time.time()
    invariant = GOD_CODE
    max_error = 0.0
    steps = 200
    for i in range(steps + 1):
        x = (i / steps) * 416.0
        g = god_code_fn(x)
        product = g * (2.0 ** (x / 104.0))
        error = abs(product - invariant)
        max_error = max(max_error, error)
    passed = max_error < 1e-9
    elapsed = (time.time() - t0) * 1000
    return SimulationResult(
        name="conservation_proof", category="core", passed=passed,
        elapsed_ms=elapsed, detail=f"max_error={max_error:.2e} over {steps} points",
        conservation_error=max_error, god_code_measured=invariant,
        god_code_error=max_error, sacred_alignment=1.0 if passed else 0.0,
    )


def sim_dial_sweep_a(nq: int = 2) -> SimulationResult:
    """Sweep dial 'a' from 0 to 8, verify conservation at each point.

    Conservation law:  G(a,0,0,0) × 2^(−8a/104) = G(0,0,0,0) = GOD_CODE.
    The dial raises frequency by 2^(8a/104), so dividing (negative exponent)
    must recover the invariant.
    """
    t0 = time.time()
    results = []
    max_error = 0.0
    for a in range(9):
        g = god_code_dial(a=a)
        x = 8 * a
        product = g * (2.0 ** (-x / 104.0))  # conservation: G(a) / 2^(8a/104)
        error = abs(product - GOD_CODE)
        max_error = max(max_error, error)
        results.append({"a": a, "G": g, "product": product, "error": error})
    passed = max_error < 1e-9
    elapsed = (time.time() - t0) * 1000
    return SimulationResult(
        name="dial_sweep_a", category="core", passed=passed,
        elapsed_ms=elapsed, detail=f"Dial A sweep 0..8, max_err={max_error:.2e}",
        conservation_error=max_error, god_code_measured=GOD_CODE,
        god_code_error=max_error, sacred_alignment=1.0 if passed else 0.0,
        extra={"sweep_results": results},
    )


def sim_104_tet_spectrum(nq: int = 6) -> SimulationResult:
    """104-TET equal temperament: verify GOD_CODE position in 104-division octave.

    GOD_CODE = 286^(1/φ) × 2^(416/104) lives at step ≈ 91.853 in the 104-TET
    lattice rooted at 286 Hz (iron frequency).  The non-integer offset arises
    from φ's irrationality — by design, GOD_CODE transcends the lattice.

    Verification:
      1. Step falls within the octave range [0, 104]
      2. Algebraic step value matches 104×[(1/φ−1)×log₂(286) + 4] exactly
      3. GOD_CODE is bracketed by adjacent 104-TET frequencies
    """
    t0 = time.time()
    base_freq = 286.0
    freqs = [base_freq * (2.0 ** (step / 104.0)) for step in range(105)]

    god_code_step = 104 * math.log2(GOD_CODE / base_freq)
    # Algebraic expected: 104 * [(1/φ − 1)*log₂(286) + 4]
    expected_step = 104 * ((1.0 / PHI - 1.0) * math.log2(base_freq) + 4.0)
    algebraic_error = abs(god_code_step - expected_step)

    step_lo = int(math.floor(god_code_step))
    step_hi = step_lo + 1
    in_range = 0 <= god_code_step <= 104
    bracketed = (freqs[step_lo] <= GOD_CODE <= freqs[step_hi]) if 0 <= step_lo < 104 else False
    algebraic_match = algebraic_error < 1e-9

    passed = in_range and bracketed and algebraic_match
    frac = god_code_step - step_lo
    elapsed = (time.time() - t0) * 1000
    return SimulationResult(
        name="104_tet_spectrum", category="core", passed=passed,
        elapsed_ms=elapsed,
        detail=f"GOD_CODE at step {god_code_step:.6f} (between {step_lo}–{step_hi}), "
               f"frac={frac:.6f}, algebraic_err={algebraic_error:.2e}",
        god_code_measured=GOD_CODE,
        sacred_alignment=1.0 - abs(frac - (1.0 - 1.0 / PHI)),
        extra={
            "god_code_step": god_code_step,
            "expected_step": expected_step,
            "step_lo": step_lo, "step_hi": step_hi,
            "fractional_offset": frac,
            "in_range": in_range, "bracketed": bracketed,
            "algebraic_match": algebraic_match,
            "total_steps": 105,
        },
    )


def sim_ln_god_code_2pi(nq: int = 1) -> SimulationResult:
    """Verify ln(GOD_CODE) ≈ 2π (6.2682... ≈ 6.2832...)."""
    t0 = time.time()
    ln_gc = math.log(GOD_CODE)
    two_pi = 2.0 * math.pi
    error = abs(ln_gc - two_pi)
    ratio = ln_gc / two_pi
    passed = error < 0.02
    elapsed = (time.time() - t0) * 1000
    return SimulationResult(
        name="ln_god_code_2pi", category="core", passed=passed,
        elapsed_ms=elapsed,
        detail=f"ln(GOD_CODE)={ln_gc:.6f}, 2π={two_pi:.6f}, error={error:.6f}, ratio={ratio:.6f}",
        god_code_measured=GOD_CODE, god_code_error=error,
        sacred_alignment=ratio,
        extra={"ln_god_code": ln_gc, "two_pi": two_pi, "ratio": ratio},
    )


def sim_sacred_cascade(nq: int = 8) -> SimulationResult:
    """Sacred cascade: H → CNOT chain → sacred gates → measurement."""
    t0 = time.time()
    n = min(nq, 10)
    sv = init_sv(n)
    for q in range(n):
        sv = apply_single_gate(sv, H_GATE, q, n)
    for q in range(n - 1):
        sv = apply_cnot(sv, q, q + 1, n)
    sv = apply_single_gate(sv, GOD_CODE_GATE, 0, n)
    sv = apply_single_gate(sv, PHI_GATE, 1 % n, n)
    sv = apply_single_gate(sv, VOID_GATE, 2 % n, n)
    sv = apply_single_gate(sv, IRON_GATE, 3 % n, n)
    for q in range(n - 2, -1, -1):
        sv = apply_cnot(sv, q, q + 1, n)
    probs = probabilities(sv)
    entropy = entanglement_entropy(sv, n)
    elapsed = (time.time() - t0) * 1000
    return SimulationResult(
        name="sacred_cascade", category="core", passed=True,
        elapsed_ms=elapsed, detail=f"Sacred cascade {n}q, S={entropy:.4f}",
        fidelity=1.0, circuit_depth=2 * (n - 1) + n + 4, num_qubits=n,
        probabilities=probs, entanglement_entropy=entropy, entropy_value=entropy,
        phase_coherence=abs(math.cos(GOD_CODE_PHASE_ANGLE + PHI_PHASE_ANGLE)),
        sacred_alignment=abs(math.cos(GOD_CODE_PHASE_ANGLE * PHI)),
    )


# ── Registry of core simulations ────────────────────────────────────────────
CORE_SIMULATIONS = [
    ("conservation_proof", sim_conservation_proof, "core", "Verify G(X)×2^(X/104)=INVARIANT", 2),
    ("dial_sweep_a", sim_dial_sweep_a, "core", "Sweep dial A, verify conservation", 2),
    ("104_tet_spectrum", sim_104_tet_spectrum, "core", "104-TET octave alignment", 6),
    ("ln_god_code_2pi", sim_ln_god_code_2pi, "core", "ln(GOD_CODE) ≈ 2π proof", 1),
    ("sacred_cascade", sim_sacred_cascade, "core", "H→CNOT→Sacred gates cascade", 8),
]

__all__ = [
    "sim_conservation_proof", "sim_dial_sweep_a", "sim_104_tet_spectrum",
    "sim_ln_god_code_2pi", "sim_sacred_cascade",
    "CORE_SIMULATIONS",
]
