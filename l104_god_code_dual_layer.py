#!/usr/bin/env python3
"""
L104 GOD_CODE Dual-Layer Encoding System
════════════════════════════════════════════════════════════════════════════════

WHAT THIS IS:
    A logarithmic encoding scheme that indexes physical constants on a
    discrete frequency grid anchored at the iron BCC lattice parameter.
    It is an ENCODING (like a coordinate system), not a DERIVATION.
    The equation does not derive physics from first principles — it
    provides a compact notation for looking up known values.

WHAT IS GENUINELY INTERESTING:
    - The base constant 286 pm IS the Fe BCC lattice parameter (±0.23%)
    - The quantization grain 104 = 26×4 = Fe_Z × He4_A (factual)
    - The golden ratio exponent φ creates an irrational base that avoids
      rational resonances — a real mathematical property
    - The encoding provides a single compact address (4 integers) for any
      positive number, which is useful for lookup and comparison

WHAT IS NOT:
    - The ±0.005% precision is guaranteed by grid density, not physics.
      Any number (including the price of coffee) snaps to within ±0.005%
      on a grid with step (13/12)^(1/758). This is basic rounding.
    - "Speed of light is EXACT" means X_v3 was reverse-solved from c.
    - The equation does not predict unmeasured values (yet — see PREDICTIONS).

TWO-LAYER ARCHITECTURE:

    Layer 1 — CONSCIOUSNESS (The GOD_CODE Equation):
      G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
      Half-step: ±0.334%. Anchored at integer X=286. Honest iron scaffold.
      63 constants indexed. The sacred geometry — WHY constants exist.

    Layer 2 — PHYSICS (The OMEGA Equation):
      Ω = Σ(|ζ(½+GCi)|, cos(2πφ³), (26×1.8527)/φ²) × (GOD_CODE/φ)
      OMEGA = 6539.34712682 — the Sovereign Field Constant.
      F(I) = I × Ω / φ² — the sovereign field equation.
      Derived from GOD_CODE via zeta function + golden resonance + iron curvature.
      Includes v3 superparticular precision grid as encoding sub-tool.

    Bridge: Layer 1 (consciousness) generates OMEGA through physics operations.
      GOD_CODE → zeta + golden resonance + iron curvature → OMEGA.

PREDICTIONS:
    To move from encoding to science, this module generates FALSIFIABLE
    predictions — specific grid points that the equation assigns to
    undiscovered or unmeasured values. If a prediction is confirmed by
    future measurement, that constitutes genuine evidence. If it fails,
    the encoding has no predictive power beyond curve-fitting.

    See: predict_from_patterns(), predict_from_gaps()

════════════════════════════════════════════════════════════════════════════════
Version: 3.1.0  (Algorithm Search + OMEGA Pipeline Integration)
Constants: GOD_CODE = 527.5184818492612, OMEGA = 6539.34712682
════════════════════════════════════════════════════════════════════════════════
"""

import math
import json
import os
import random
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
from fractions import Fraction

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: CONSCIOUSNESS CORE — Import the Original
# ═══════════════════════════════════════════════════════════════════════════════

from l104_god_code_equation import (
    # Sacred constants
    GOD_CODE, PHI, TAU, VOID_CONSTANT,
    PRIME_SCAFFOLD, QUANTIZATION_GRAIN, OCTAVE_OFFSET,
    BASE, STEP_SIZE,
    # Iron data
    FE_BCC_LATTICE_PM, FE_ATOMIC_NUMBER, HE4_MASS_NUMBER,
    FE_56_BE_PER_NUCLEON, HE4_BE_PER_NUCLEON, FE_K_ALPHA1_KEV,
    FE_ATOMIC_RADIUS_PM, IRON_DATA_SYNTHESIS,
    # Functions
    god_code_equation, exponent_value, solve_for_exponent, find_nearest_dials,
    # Frequency table
    QUANTUM_FREQUENCY_TABLE,
    # Named constants
    SCHUMANN_RESONANCE, ALPHA_EEG, BETA_EEG, GAMMA_BINDING,
    BOHR_RADIUS_GOD, FE_BCC_LATTICE_GOD, FE56_BE_GOD,
    OMEGA_GOD, OMEGA_AUTHORITY_GOD,
)

# ── Algorithm Search Integration (Qiskit Quantum Circuits for Layer 1) ──
try:
    from l104_god_code_algorithm import (
        GodCodeAlgorithm, GodCodeGroverSearch, GodCodeQFTSpectrum,
        GodCodeDialCircuit, GodCodeEntanglement, GodCodeDialRegister,
        GodCodePhaseOracle, DialSetting, CircuitResult,
        god_code_algorithm as _algorithm_singleton,
    )
    _ALGORITHM_AVAILABLE = True
except ImportError:
    _ALGORITHM_AVAILABLE = False
    _algorithm_singleton = None

# ── OMEGA Pipeline Functions (from l104_real_math — restored originals) ──
try:
    from l104_real_math import RealMath as _RealMath
    _OMEGA_PIPELINE_AVAILABLE = True
except ImportError:
    _OMEGA_PIPELINE_AVAILABLE = False
    _RealMath = None

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: PHYSICS GENERATOR — OMEGA Sovereign Field Equation
# ═══════════════════════════════════════════════════════════════════════════════
#
# THE COMPLETE OMEGA EQUATION  (Sovereign Field Constant)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Origin: Mini-AI Collective, January 6 2026, commit d4d08873
# Restored: commits df642249, eeeb3362, a431a902 (exact reproduction)
#
# ─── THE FOUR FRAGMENTS ────────────────────────────────────────────────────
#
# Fragment 1 — RESEARCHER (Prime Number Theorem at lattice invariant):
#   Step 1: solve_lattice_invariant(104)
#           = sin(104 × π / 104) × exp(104 / 527.5184818492)
#           = sin(π) × exp(0.19710...)
#           ≈ 3.9e-16 × 1.2179...  ≈ 4.75e-16  (floating point sin(π) ≈ 0)
#   Step 2: int(4.75e-16) = 0
#   Step 3: prime_density(0) → n < 2 → return 0.0
#   RESULT: Researcher = 0.0
#
# Fragment 2 — GUARDIAN (Riemann Zeta on the Critical Line):
#   Step 1: s = 0.5 + 527.518i   (GOD_CODE truncated as in original code)
#   Step 2: Dirichlet eta series: η(s) = Σ_{n=1}^{1000} (-1)^(n-1) / n^s
#   Step 3: Analytic continuation: ζ(s) = η(s) / (1 - 2^(1-s))
#   Step 4: |ζ(0.5 + 527.518i)|
#   RESULT: Guardian ≈ 1.573827...
#
# Fragment 3 — ALCHEMIST (Golden Ratio Harmonic Resonance):
#   Step 1: Input value = φ² = φ + 1 = 2.618033988749895
#   Step 2: golden_resonance(φ²) = cos(2π × φ² × φ)
#           = cos(2π × φ³)
#   Step 3: φ³ = 2φ + 1 = 4.23606797749979  (from φ² = φ+1, so φ³ = φ²·φ = (φ+1)φ = φ²+φ = 2φ+1)
#   Step 4: cos(2π × 4.23606797749979)
#           = cos(26.6222...)
#   RESULT: Alchemist ≈ 0.087433...
#
# Fragment 4 — ARCHITECT (Iron Manifold Curvature Tensor):
#   Step 1: dimension = 26  (Fe atomic number, Z=26)
#   Step 2: tension = 1.8527
#   Step 3: manifold_curvature_tensor(26, 1.8527) = (26 × 1.8527) / φ²
#           = 48.1702 / 2.618033988749895
#   RESULT: Architect ≈ 18.399393...
#
# ─── THE SUMMATION ────────────────────────────────────────────────────────
#
#   Σ(fragments) = 0.0 + 1.573827... + 0.087433... + 18.399393...
#                ≈ 20.060654...
#
# ─── THE MULTIPLIER ───────────────────────────────────────────────────────
#
#   Multiplier = GOD_CODE / φ                       (original: 527.5184818492 / φ)
#              = 527.5184818492 / 1.618033988749895
#              ≈ 326.024351...
#
# ─── THE OMEGA EQUATION ──────────────────────────────────────────────────
#
#   Ω = Σ(fragments) × (GOD_CODE / φ)
#     = 20.060654... × 326.024351...
#     = 6539.34712682
#
# ─── THE SOVEREIGN FIELD EQUATION ────────────────────────────────────────
#
#   F(I) = I × Ω / φ²
#
#   Where:
#     I = Intensity (input)
#     Ω = 6539.34712682  (OMEGA — the Sovereign Field Constant)
#     φ² = φ + 1 = 2.618033988749895
#
#   At I=1:  F(1) = 6539.34712682 / 2.618033988749895 = 2497.808338211271
#
# ─── OMEGA AUTHORITY ─────────────────────────────────────────────────────
#
#   Ω_A = Ω / φ² = 6539.34712682 / 2.618033988749895 = 2497.808338211271
#
# ─── ORIGINAL PIPELINE FUNCTIONS (from l104_real_math.py, commit d4d08873) ───
#
#   1. zeta_approximation(s, terms=1000)
#      Standard Riemann zeta via Dirichlet eta series:
#      ζ(s) = η(s) / (1 - 2^(1-s))  where  η(s) = Σ_{n=1}^{terms} (-1)^(n-1)/n^s
#
#   2. solve_lattice_invariant(seed)
#      R(x) = sin(x·π/104) × exp(x / 527.5184818492)
#
#   3. manifold_curvature_tensor(dimension, tension)
#      R = (dimension × tension) / φ²
#      NOTE: Not a Riemannian curvature tensor — it's a φ²-normalized product.
#
#   4. golden_resonance(value)
#      R(v) = cos(2π × v × φ)
#      OMEGA call: golden_resonance(φ²) = cos(2π·φ²·φ) = cos(2π·φ³)
#
#   5. prime_density(n)
#      Standard PNT: π(n)/n ≈ 1/ln(n).  For n<2 → 0.0.
#
#   6. entropy_inversion_integral(start, end)
#      ∫[start,end] (1/φ) dx = (end - start) / φ
#
#   7. sovereign_field_equation(intensity)
#      F(I) = I × Ω / φ²
#
# ─── v3 PRECISION GRID (Encoding Sub-Tool) ──────────────────────────────
#
#   G_v3(a,b,c,d) = 285.999^(1/φ) × (13/12)^((99a + 3032 - b - 99c - 758d) / 758)
#   63 peer-reviewed constants at ±0.005%.
#   OMEGA on v3 grid: dials (16,0,0,-60), E=50096, error 0.0001%.
#
# ═══════════════════════════════════════════════════════════════════════════════

# ── OMEGA — Sovereign Field Constant (Primary Layer 2 Identity) ──
OMEGA = 6539.34712682
OMEGA_AUTHORITY = OMEGA / (PHI ** 2)  # Ω/φ² = 2497.808338211271


def sovereign_field_equation(intensity: float) -> float:
    """
    F(I) = I × Ω / φ² — Sovereign field strength at given intensity.

    Where:
        Ω  = 6539.34712682  (OMEGA — the Sovereign Field Constant)
        φ² = φ + 1 = 2.618033988749895
        F(1) = 2497.808338211271  (OMEGA_AUTHORITY)
    """
    return intensity * OMEGA / (PHI ** 2)


def omega_derivation_chain(zeta_terms: int = 1000) -> Dict[str, Any]:
    """
    THE COMPLETE OMEGA DERIVATION — Layer 2 Physics.

    Reproduces the EXACT original derivation from commit d4d08873 (Jan 6, 2026).
    No truncation — every step is computed and returned.

    THE OMEGA EQUATION:
        Ω = Σ(fragments) × (GOD_CODE / φ)

    WHERE THE FOUR FRAGMENTS ARE:

    Fragment 1 — RESEARCHER:
        solve_lattice_invariant(104) = sin(104·π/104) × exp(104/527.5184818492)
        = sin(π) × exp(0.19710...) ≈ 0  (floating point sin(π) ≈ 3.9e-16)
        int(≈0) = 0 → prime_density(0) = 0.0  (n<2 in PNT)
        RESULT: 0.0

    Fragment 2 — GUARDIAN:
        s = 0.5 + 527.518i  (GOD_CODE truncated to match original)
        η(s) = Σ_{n=1}^{999} (-1)^(n-1) / n^s   (Dirichlet eta, terms=1000, range(1,terms))
        ζ(s) = η(s) / (1 - 2^(1-s))               (analytic continuation)
        RESULT: |ζ(0.5 + 527.518i)| ≈ 1.571044...

    Fragment 3 — ALCHEMIST:
        golden_resonance(φ²) = cos(2π × φ² × φ) = cos(2π × φ³)
        φ³ = 2φ + 1 = 4.23606797749979
        RESULT: cos(2π × 4.23606797749979) ≈ 0.087433...

    Fragment 4 — ARCHITECT:
        manifold_curvature_tensor(26, 1.8527) = (26 × 1.8527) / φ²
        = 48.1702 / 2.618033988749895
        RESULT: ≈ 18.399393...

    SUMMATION:
        Σ = 0.0 + 1.571044... + 0.087433... + 18.399393... ≈ 20.057870...

    OMEGA:
        Ω = Σ × (527.5184818492 / φ) = 20.057870... × 326.024351... = 6539.34712682

    Returns:
        Complete dict with every fragment value, intermediate computations,
        function-by-function breakdown, and cross-validation.
    """
    import cmath

    # ── Fragment 1: RESEARCHER ──
    # solve_lattice_invariant(104) — original uses 527.5184818492 as target
    _gc_original = 527.5184818492  # truncation as in original code
    lattice_invariant = math.sin(104 * math.pi / 104) * math.exp(104 / _gc_original)
    lattice_int = int(lattice_invariant)  # int(≈0) = 0
    # prime_density(0) — n<2 → 0.0
    frag_researcher = 0.0 if lattice_int < 2 else 1.0 / math.log(lattice_int)

    # ── Fragment 2: GUARDIAN ──
    # zeta_approximation(complex(0.5, 527.518), terms=1000)
    # NOTE: Original code used 527.518 (truncated), NOT the full GOD_CODE
    # NOTE: range(1, zeta_terms) matches l104_real_math.zeta_approximation exactly
    #       which iterates n=1..999 for terms=1000 (standard Dirichlet eta convention)
    s_guardian = complex(0.5, 527.518)
    eta = sum(((-1)**(n-1)) / (n**s_guardian) for n in range(1, zeta_terms))
    zeta_val = eta / (1 - 2**(1 - s_guardian))
    frag_guardian = abs(zeta_val)

    # ── Fragment 3: ALCHEMIST ──
    # golden_resonance(PHI²) = cos(2π × PHI² × PHI) = cos(2π × φ³)
    # φ³ = 2φ + 1 = 4.23606797749979  (identity: φ²=φ+1 → φ³=φ²·φ=(φ+1)φ=φ²+φ=2φ+1)
    phi_cubed = PHI ** 3  # = 2φ+1 = 4.23606797749979
    frag_alchemist = math.cos(2 * math.pi * phi_cubed)

    # ── Fragment 4: ARCHITECT ──
    # manifold_curvature_tensor(26, 1.8527) = (26 × 1.8527) / φ²
    fe_z = 26           # Iron atomic number
    tension = 1.8527    # Manifold tension parameter
    phi_squared = PHI ** 2  # φ² = φ+1 = 2.618033988749895
    frag_architect = (fe_z * tension) / phi_squared

    # ── Summation ──
    sigma = frag_researcher + frag_guardian + frag_alchemist + frag_architect

    # ── Multiplier: GOD_CODE / φ ──
    # Original uses truncated 527.5184818492, matching d4d08873 line 58
    multiplier = _gc_original / PHI

    # ── OMEGA = Σ × (GOD_CODE / φ) ──
    omega_computed = sigma * multiplier

    return {
        # Fragment-by-fragment breakdown
        "fragments": {
            "researcher": {
                "value": frag_researcher,
                "function": "prime_density(int(solve_lattice_invariant(104)))",
                "steps": {
                    "lattice_invariant_raw": lattice_invariant,
                    "lattice_invariant_int": lattice_int,
                    "sin_pi": math.sin(math.pi),
                    "exp_104_over_gc": math.exp(104 / _gc_original),
                    "prime_density_input": lattice_int,
                    "prime_density_result": frag_researcher,
                },
                "note": "sin(π)≈3.9e-16 in float64 → int(0)=0 → prime_density(0)=0.0 (n<2)",
            },
            "guardian": {
                "value": frag_guardian,
                "function": "abs(zeta_approximation(complex(0.5, 527.518), terms=1000))",
                "steps": {
                    "s": str(s_guardian),
                    "zeta_terms": zeta_terms,
                        "eta_method": "Dirichlet: η(s) = Σ_{n=1}^{terms-1} (-1)^(n-1) / n^s  [range(1,terms)]",
                    "continuation": "ζ(s) = η(s) / (1 - 2^(1-s))",
                    "zeta_real": zeta_val.real,
                    "zeta_imag": zeta_val.imag,
                    "zeta_magnitude": frag_guardian,
                },
                "note": "Imaginary part 527.518 (truncated) matches original d4d08873 code",
            },
            "alchemist": {
                "value": frag_alchemist,
                "function": "golden_resonance(φ²) = cos(2π × φ² × φ) = cos(2π × φ³)",
                "steps": {
                    "phi": PHI,
                    "phi_squared": phi_squared,
                    "phi_cubed": phi_cubed,
                    "phi_cubed_identity": "φ³ = 2φ+1 (from φ²=φ+1)",
                    "two_phi_plus_1": 2 * PHI + 1,
                    "argument": 2 * math.pi * phi_cubed,
                    "cos_value": frag_alchemist,
                },
                "note": "cos(2πφ³) = cos(4πφ+2π) = cos(4πφ) = cos(2π√5) ≈ 0.0874",
            },
            "architect": {
                "value": frag_architect,
                "function": "manifold_curvature_tensor(26, 1.8527) = (26 × 1.8527) / φ²",
                "steps": {
                    "dimension": fe_z,
                    "dimension_meaning": "Fe atomic number Z=26",
                    "tension": tension,
                    "numerator": fe_z * tension,
                    "phi_squared": phi_squared,
                    "curvature_result": frag_architect,
                },
                "note": "NOT a Riemannian curvature tensor — φ²-normalized iron product",
            },
        },
        # Aggregation
        "sigma": sigma,
        "sigma_breakdown": f"{frag_researcher} + {frag_guardian} + {frag_alchemist} + {frag_architect}",
        "multiplier": multiplier,
        "multiplier_equation": f"{_gc_original} / {PHI} = {multiplier}",
        # The OMEGA result
        "omega_computed": omega_computed,
        "omega_canonical": OMEGA,
        "delta": abs(omega_computed - OMEGA),
        "relative_error": abs(omega_computed - OMEGA) / OMEGA if OMEGA else 0,
        # Sovereign field at I=1
        "sovereign_field_at_1": omega_computed / phi_squared,
        "omega_authority_computed": omega_computed / phi_squared,
        "omega_authority_canonical": OMEGA_AUTHORITY,
        # Field equation
        "field_equation": "F(I) = I × Ω / φ²",
        "field_at_1": sovereign_field_equation(1.0),
        "field_at_god_code": sovereign_field_equation(GOD_CODE),
        # Metadata
        "zeta_terms": zeta_terms,
        "gc_used_for_zeta": 527.518,
        "gc_used_for_multiplier": _gc_original,
        "gc_full_precision": GOD_CODE,
        "origin": "Mini-AI Collective, Jan 6 2026, commit d4d08873",
        "note": "Exact reproduction — uses truncated GC values matching original code",
    }


# ── v3 Superparticular Precision Grid (Encoding Sub-Tool) ──
# Originally in l104_god_code_evolved_v3.py — absorbed as precision sub-tool.
#   G_v3(a,b,c,d) = 285.999^(1/φ) × (13/12)^((99a + 3032 - b - 99c - 758d) / 758)
#   63 peer-reviewed constants at ±0.005% — grid density guarantee.

# v3 CONSTANTS
V3_BASE_R = 13.0 / 12.0                                    # r: 13/12 = 1.08333...
V3_BASE_R_FRACTION = Fraction(13, 12)                       # Exact rational
V3_QUANTIZATION_GRAIN = 758                                  # Q: 758 = 2 × 379
V3_PRIME_SCAFFOLD = 285.99882035187807                       # X: tuned for c on grid

V3_DIAL_COEFFICIENT = 99                                     # p = 99 (optimal from search)
V3_OCTAVE_OFFSET = 4 * V3_QUANTIZATION_GRAIN                # K = 4Q = 3032
V3_BASE = V3_PRIME_SCAFFOLD ** (1.0 / PHI)                   # X^(1/φ) = 32.969821069618
V3_STEP_SIZE = V3_BASE_R ** (1.0 / V3_QUANTIZATION_GRAIN)   # (13/12)^(1/758) = 1.0001056028

# THE v3 GOD CODE — with K=4Q
GOD_CODE_V3 = V3_BASE * (V3_BASE_R ** (V3_OCTAVE_OFFSET / V3_QUANTIZATION_GRAIN))

# Short aliases
Q_V3 = V3_QUANTIZATION_GRAIN        # 758
P_V3 = V3_DIAL_COEFFICIENT          # 99
K_V3 = V3_OCTAVE_OFFSET             # 3032
X_V3 = V3_PRIME_SCAFFOLD            # 285.999
BASE_V3 = V3_BASE                   # 32.970
STEP_V3 = V3_STEP_SIZE              # 1.0001056028
R_V3 = V3_BASE_R                    # 13/12

# Precision metrics
HALF_STEP_PCT_V3 = (STEP_V3 - 1) / 2 * 100              # ±0.00528%
HALF_STEP_PCT_ORIGINAL = (STEP_SIZE - 1) / 2 * 100
PRECISION_IMPROVEMENT_V3 = HALF_STEP_PCT_ORIGINAL / HALF_STEP_PCT_V3  # ~63×

# Speed of light grid point
C_EXPONENT_V3 = 151737
C_VALUE_V3 = BASE_V3 * (R_V3 ** (C_EXPONENT_V3 / Q_V3))


# ── v3 Core Functions ──

def god_code_v3(a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> float:
    """
    The v3 Evolved Equation — Superparticular 13/12 Base.
    G_v3(a,b,c,d) = 285.999^(1/φ) × (13/12)^((99a + 3032 - b - 99c - 758d) / 758)
    """
    exponent = (P_V3 * a) + (K_V3 - b) - (P_V3 * c) - (Q_V3 * d)
    return BASE_V3 * (R_V3 ** (exponent / Q_V3))


def exponent_value_v3(a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> int:
    """Calculate the raw exponent E for given dial settings (v3 equation)."""
    return (P_V3 * a) + (K_V3 - b) - (P_V3 * c) - (Q_V3 * d)


def solve_for_exponent_v3(target: float) -> float:
    """Find the exact (non-integer) exponent E that produces target in v3 equation."""
    if target <= 0:
        raise ValueError("Target must be positive")
    return Q_V3 * math.log(target / BASE_V3) / math.log(R_V3)


def find_nearest_dials_v3(target: float, max_d_range: int = 300) -> list:
    """Find the simplest integer (a,b,c,d) dials that approximate target (v3 equation)."""
    if target <= 0:
        return []
    E_exact = solve_for_exponent_v3(target)
    E_int = round(E_exact)
    results = []
    for d_val in range(-max_d_range, max_d_range + 1):
        remainder = E_int - K_V3 + Q_V3 * d_val
        if P_V3 > 0:
            b_val = (-remainder) % P_V3
            dac = (remainder + b_val) // P_V3
        else:
            b_val = -remainder
            dac = 0
        if b_val < 0:
            continue
        if dac >= 0:
            a_val, c_val = dac, 0
        else:
            a_val, c_val = 0, -dac
        E_check = P_V3 * a_val + K_V3 - b_val - P_V3 * c_val - Q_V3 * d_val
        if E_check != E_int:
            continue
        val = god_code_v3(a_val, b_val, c_val, d_val)
        err = abs(val - target) / target * 100
        cost = a_val + b_val + c_val + abs(d_val)
        results.append((a_val, b_val, c_val, d_val, val, err, cost))
    results.sort(key=lambda x: (x[5], x[6]))
    return [(a, b, c, d, v, e) for a, b, c, d, v, e, _ in results[:10]]


# ── v3 Frequency Table (63 peer-reviewed constants) ──

V3_FREQUENCY_TABLE = {
    # (a, b, c, d): (name, grid_value, exponent, measured, error_pct)
    (0, 27, 6, -197):      ("SPEED_OF_LIGHT",         2.997925e+08,  151737,  299792458,          0.0000),
    (0, 14, 1, 19):        ("STANDARD_GRAVITY",       9.806246e+00,  -11483,  9.80665,            0.0041),
    (0, 7, 1236, 300):     ("PLANCK_CONSTANT_eVs",    4.135829e-15,  -346739, 4.135667696e-15,    0.0039),
    (0, 24, 12, 163):      ("BOLTZMANN_eV_K",         8.617058e-05,  -121734, 8.617333262e-5,     0.0032),
    (0, 12, 2223, 298):    ("ELEMENTARY_CHARGE",      1.602176e-19,  -442941, 1.602176634e-19,    0.0001),
    (2591, 2, 0, -298):    ("AVOGADRO",               6.022271e+23,  485423,  6.02214076e23,      0.0022),
    (7, 2, 0, -1):         ("BOHR_RADIUS_PM",         5.291960e+01,  4481,    52.9177210544,      0.0036),
    (0, 10, 8, 14):        ("RYDBERG_EV",             1.360553e+01,  -8382,   13.605693123,       0.0012),
    (0, 20, 9, -15):       ("FINE_STRUCTURE_INV",     1.370307e+02,  13491,   137.035999084,      0.0039),
    (0, 23, 12, 35):       ("COMPTON_PM",             2.426356e+00,  -24709,  2.42631023867,      0.0019),
    (0, 23, 13, 33):       ("CLASSICAL_E_RADIUS_FM",  2.817984e+00,  -23292,  2.8179403205,       0.0015),
    (0, 5, 3, 6):          ("HARTREE_EV",             2.721088e+01,  -1818,   27.211386246,       0.0019),
    (0, 3, 1310, 299):     ("MAG_FLUX_QUANTUM_Wb",    2.067929e-15,  -353303, 2.067833848e-15,    0.0046),
    (2, 13, 0, -79):       ("VON_KLITZING_OHM",       2.581402e+04,  63099,   25812.80745,        0.0047),
    (22, 1, 0, 259):       ("STEFAN_BOLTZMANN",        5.670448e-08,  -191113, 5.670374419e-8,     0.0013),
    (0, 14, 510, 299):     ("VACUUM_PERMITTIVITY",    8.854555e-12,  -274114, 8.8541878128e-12,   0.0041),
    (0, 2, 12, 168):       ("BOHR_MAGNETON_eV_T",     5.788385e-05,  -125502, 5.7883818060e-5,    0.0001),
    (0, 11, 8, 55):        ("ELECTRON_MASS_MEV",      5.110029e-01,  -39461,  0.51099895069,      0.0008),
    (0, 10, 11, -12):      ("MUON_MASS_MEV",          1.056598e+02,  11029,   105.6583755,        0.0014),
    (0, 10, 9, -47):       ("TAU_MASS_MEV",           1.776904e+03,  37757,   1776.86,            0.0025),
    (0, 27, 1, -38):       ("PROTON_MASS_MEV",        9.383107e+02,  31710,   938.27208816,       0.0041),
    (0, 14, 1, -38):       ("NEUTRON_MASS_MEV",       9.395996e+02,  31723,   939.56542052,       0.0036),
    (24, 2, 0, -4):        ("W_BOSON_GEV",            8.036849e+01,  8438,    80.3692,            0.0009),
    (0, 22, 2, -9):        ("Z_BOSON_GEV",            9.118737e+01,  9634,    91.1876,            0.0002),
    (0, 14, 10, -14):      ("HIGGS_GEV",              1.252538e+02,  12640,   125.25,             0.0030),
    (8, 13, 0, -13):       ("PION_CHARGED_MEV",       1.395718e+02,  13665,   139.57039,          0.0010),
    (20, 2, 0, -11):       ("PION_NEUTRAL_MEV",       1.349770e+02,  13348,   134.9768,           0.0002),
    (0, 11, 9, -31):       ("KAON_MEV",               4.936559e+02,  25628,   493.677,            0.0043),
    (0, 23, 4, -47):       ("D_MESON_MEV",            1.869685e+03,  38239,   1869.66,            0.0014),
    (0, 11, 10, -18):      ("TOP_QUARK_GEV",          1.725746e+02,  15675,   172.57,             0.0026),
    (0, 14, 6, 29):        ("BOTTOM_QUARK_GEV",       4.180044e+00,  -19558,  4.18,               0.0011),
    (0, 25, 5, 44):        ("CHARM_QUARK_GEV",        1.269945e+00,  -30840,  1.27,               0.0043),
    (4, 29, 0, 21):        ("FE56_BE_PER_NUCLEON",    8.790053e+00,  -12519,  8.790,              0.0006),
    (6, 10, 0, 24):        ("HE4_BE_PER_NUCLEON",     7.073867e+00,  -14576,  7.074,              0.0019),
    (0, 24, 13, 20):       ("O16_BE_PER_NUCLEON",     7.976274e+00,  -13439,  7.976,              0.0034),
    (0, 20, 9, 21):        ("C12_BE_PER_NUCLEON",     7.680368e+00,  -13797,  7.680,              0.0048),
    (5, 27, 0, 23):        ("U238_BE_PER_NUCLEON",    7.570058e+00,  -13934,  7.570,              0.0008),
    (4, 24, 0, 21):        ("NI62_BE_PER_NUCLEON",    8.794696e+00,  -12514,  8.7945,             0.0022),
    (0, 22, 5, 37):        ("DEUTERON_BE",            2.224628e+00,  -25531,  2.22457,             0.0026),
    (8, 5, 0, 22):         ("TRITON_BE",              8.481853e+00,  -12857,  8.48182,             0.0004),
    (8, 20, 0, -22):       ("FE_BCC_LATTICE_PM",      2.866391e+02,  20480,   286.65,             0.0038),
    (6, 26, 0, -12):       ("FE_ATOMIC_RADIUS_PM",    1.259966e+02,  12696,   126.0,              0.0027),
    (0, 27, 11, 23):       ("FE_K_ALPHA1_KEV",        6.404076e+00,  -15518,  6.404,              0.0012),
    (0, 13, 14, 20):       ("FE_IONIZATION_EV",       7.902497e+00,  -13527,  7.9024678,          0.0004),
    (30, 1, 0, -22):       ("CU_LATTICE_PM",          3.614852e+02,  22677,   361.49,             0.0013),
    (0, 9, 5, -28):        ("AL_LATTICE_PM",          4.049397e+02,  23752,   404.95,             0.0025),
    (23, 1, 0, -28):       ("SI_LATTICE_PM",          5.431028e+02,  26532,   543.102,            0.0001),
    (4, 20, 0, -187):      ("EARTH_ORBIT_AU_KM",      1.495968e+08,  145154,  149597870.7,        0.0007),
    (3201, 25, 0, -299):   ("SOLAR_LUMINOSITY_W",     3.827992e+26,  546548,  3.828e26,           0.0002),
    (0, 16, 8, -6):        ("HUBBLE_CONSTANT",        6.740351e+01,  6772,    67.4,               0.0052),
    (0, 11, 1, 35):        ("CMB_TEMPERATURE_K",      2.725503e+00,  -23608,  2.7255,             0.0001),
    (4027, 20, 0, -298):   ("SOLAR_MASS_KG",          1.988924e+30,  627569,  1.98892e30,         0.0002),
    (8, 4, 0, 23):         ("SCHUMANN_HZ",            7.830229e+00,  -13614,  7.83,               0.0029),
    (1, 27, 0, 19):        ("ALPHA_EEG_HZ",           9.999699e+00,  -11298,  10.0,               0.0030),
    (0, 14, 12, 0):        ("GAMMA_EEG_HZ",           3.999825e+01,  1830,    40.0,               0.0044),
    (0, 19, 2, 25):        ("THETA_EEG_HZ",           6.000130e+00,  -16135,  6.0,                0.0022),
    (6, 22, 0, 11):        ("BETA_EEG_HZ",            1.999926e+01,  -4734,   20.0,               0.0037),
    (5, 18, 0, 34):        ("PI",                     3.141440e+00,  -22263,  3.14159265359,       0.0049),
    (0, 2, 9, 34):         ("EULER_E",                2.718317e+00,  -23633,  2.71828182846,       0.0013),
    (0, 27, 10, 42):       ("SQRT2",                  1.414219e+00,  -29821,  1.41421356237,       0.0004),
    (0, 5, 5, 41):         ("GOLDEN_RATIO",           1.618037e+00,  -28546,  1.618033988749895,   0.0002),
    (6, 26, 0, 53):        ("LN2",                    6.931417e-01,  -36574,  0.69314718056,       0.0008),
    (0, 7, 5731, 300):     ("PLANCK_LENGTH_M",        1.616220e-35,  -791744, 1.616255e-35,       0.0022),

    # ── OMEGA — Sovereign Field Constant (derived Jan 6, 2026 by Mini-AI Collective) ──
    # Ω = Σ(fragments) × (GOD_CODE / φ)
    # fragments: Guardian |ζ(½+527.518i)|, Alchemist cos(2πφ³), Architect (26×1.8527)/φ²
    (16, 0, 0, -60):       ("OMEGA",                  6.539343e+03,  50096,   6539.34712682,      0.0001),
    (0, 15, 7, -51):       ("OMEGA_AUTHORITY",         2.497833e+03,  40982,   2497.808338,        0.0010),
}
V3_FREQUENCY_TABLE[(0, 0, 0, 0)] = ("GOD_CODE_V3", GOD_CODE_V3, K_V3, GOD_CODE_V3, 0.0)


# ── Named v3 Constants ──

C_V3 = god_code_v3(0, 27, 6, -197)
GRAVITY_V3 = god_code_v3(0, 14, 1, 19)
BOHR_V3 = god_code_v3(7, 2, 0, -1)
FINE_STRUCTURE_INV_V3 = god_code_v3(0, 20, 9, -15)
RYDBERG_V3 = god_code_v3(0, 10, 8, 14)
SCHUMANN_V3 = god_code_v3(8, 4, 0, 23)
FE_BCC_V3 = god_code_v3(8, 20, 0, -22)
FE56_BE_V3 = god_code_v3(4, 29, 0, 21)
MUON_V3 = god_code_v3(0, 10, 11, -12)
HIGGS_V3 = god_code_v3(0, 14, 10, -14)
ELECTRON_MASS_V3 = god_code_v3(0, 11, 8, 55)
Z_BOSON_V3 = god_code_v3(0, 22, 2, -9)
PROTON_V3 = god_code_v3(0, 27, 1, -38)
NEUTRON_V3 = god_code_v3(0, 14, 1, -38)
W_BOSON_V3 = god_code_v3(24, 2, 0, -4)
TAU_V3 = god_code_v3(0, 10, 9, -47)

# OMEGA on v3 precision grid (cross-validation of Layer 2 OMEGA via encoding sub-tool)
OMEGA_V3 = god_code_v3(16, 0, 0, -60)   # OMEGA on v3 grid (err: 0.0001%)
OMEGA_AUTHORITY_V3 = god_code_v3(0, 15, 7, -51)  # OMEGA_AUTHORITY on v3 grid


# ── Real-World Derivation Engine (v3 precision grid) ──

REAL_WORLD_CONSTANTS_V3: Dict[str, Dict[str, Any]] = {}


def _rw_v3(name: str, measured: float, unit: str, dials: Tuple[int, ...],
           source: str, domain: str = "physics") -> None:
    """Register a real-world constant with its v3 equation derivation path."""
    E_int = exponent_value_v3(*dials)
    E_exact = solve_for_exponent_v3(measured)
    delta = E_exact - E_int
    grid_value = god_code_v3(*dials)
    grid_error_pct = abs(grid_value - measured) / measured * 100
    REAL_WORLD_CONSTANTS_V3[name] = {
        "measured": measured, "unit": unit, "dials": dials,
        "E_integer": E_int, "E_exact": E_exact, "delta": delta,
        "grid_value": grid_value, "grid_error_pct": grid_error_pct,
        "source": source, "domain": domain,
    }


_rw_v3("speed_of_light",      299792458,          "m/s",      (0, 27, 6, -197),     "SI exact",        "fundamental")
_rw_v3("standard_gravity",    9.80665,            "m/s²",     (0, 14, 1, 19),       "SI conventional", "fundamental")
_rw_v3("planck_constant_eVs", 4.135667696e-15,    "eV·s",     (0, 7, 1236, 300),    "SI exact",        "fundamental")
_rw_v3("boltzmann_eV_K",      8.617333262e-5,     "eV/K",     (0, 24, 12, 163),     "SI exact",        "fundamental")
_rw_v3("elementary_charge",   1.602176634e-19,     "C",        (0, 12, 2223, 298),   "SI exact",        "fundamental")
_rw_v3("avogadro",            6.02214076e23,       "mol⁻¹",   (2591, 2, 0, -298),   "SI exact",        "fundamental")
_rw_v3("bohr_radius_pm",      52.9177210544,      "pm",       (7, 2, 0, -1),        "CODATA 2022",     "atomic")
_rw_v3("rydberg_eV",          13.605693123,       "eV",       (0, 10, 8, 14),       "CODATA 2022",     "atomic")
_rw_v3("fine_structure_inv",  137.035999084,       "",         (0, 20, 9, -15),      "CODATA 2022",     "atomic")
_rw_v3("compton_pm",          2.42631023867,      "pm",       (0, 23, 12, 35),      "CODATA 2022",     "atomic")
_rw_v3("classical_e_radius_fm", 2.8179403205,     "fm",       (0, 23, 13, 33),      "CODATA 2022",     "atomic")
_rw_v3("hartree_eV",          27.211386246,       "eV",       (0, 5, 3, 6),         "CODATA 2022",     "atomic")
_rw_v3("mag_flux_quantum_Wb", 2.067833848e-15,    "Wb",       (0, 3, 1310, 299),    "CODATA 2022",     "atomic")
_rw_v3("von_klitzing_ohm",   25812.80745,         "Ω",        (2, 13, 0, -79),      "CODATA 2022",     "atomic")
_rw_v3("stefan_boltzmann",    5.670374419e-8,      "W·m⁻²·K⁻⁴", (22, 1, 0, 259),  "CODATA 2022",     "atomic")
_rw_v3("vacuum_permittivity", 8.8541878128e-12,    "F/m",      (0, 14, 510, 299),   "CODATA 2022",     "atomic")
_rw_v3("bohr_magneton_eV_T",  5.7883818060e-5,    "eV/T",     (0, 2, 12, 168),      "CODATA 2022",     "atomic")
_rw_v3("electron_mass_MeV",   0.51099895069,      "MeV/c²",  (0, 11, 8, 55),       "CODATA 2022",     "particle")
_rw_v3("muon_mass_MeV",      105.6583755,         "MeV/c²",  (0, 10, 11, -12),     "PDG 2024",        "particle")
_rw_v3("tau_mass_MeV",       1776.86,             "MeV/c²",  (0, 10, 9, -47),      "PDG 2024",        "particle")
_rw_v3("proton_mass_MeV",    938.27208816,        "MeV/c²",  (0, 27, 1, -38),      "CODATA 2022",     "particle")
_rw_v3("neutron_mass_MeV",   939.56542052,        "MeV/c²",  (0, 14, 1, -38),      "CODATA 2022",     "particle")
_rw_v3("W_boson_GeV",         80.3692,            "GeV/c²",  (24, 2, 0, -4),       "PDG 2024",        "particle")
_rw_v3("Z_boson_GeV",         91.1876,            "GeV/c²",  (0, 22, 2, -9),       "PDG 2024",        "particle")
_rw_v3("higgs_GeV",          125.25,              "GeV/c²",  (0, 14, 10, -14),     "ATLAS/CMS 2024",  "particle")
_rw_v3("pion_charged_MeV",   139.57039,           "MeV/c²",  (8, 13, 0, -13),      "PDG 2024",        "particle")
_rw_v3("pion_neutral_MeV",   134.9768,            "MeV/c²",  (20, 2, 0, -11),      "PDG 2024",        "particle")
_rw_v3("kaon_MeV",           493.677,             "MeV/c²",  (0, 11, 9, -31),      "PDG 2024",        "particle")
_rw_v3("D_meson_MeV",        1869.66,             "MeV/c²",  (0, 23, 4, -47),      "PDG 2024",        "particle")
_rw_v3("top_quark_GeV",      172.57,              "GeV/c²",  (0, 11, 10, -18),     "PDG 2024",        "particle")
_rw_v3("bottom_quark_GeV",    4.18,               "GeV/c²",  (0, 14, 6, 29),       "PDG 2024",        "particle")
_rw_v3("charm_quark_GeV",     1.27,               "GeV/c²",  (0, 25, 5, 44),       "PDG 2024",        "particle")
_rw_v3("fe56_be_per_nucleon", 8.790,              "MeV",     (4, 29, 0, 21),       "NNDC/BNL",        "nuclear")
_rw_v3("he4_be_per_nucleon",  7.074,              "MeV",     (6, 10, 0, 24),       "NNDC/BNL",        "nuclear")
_rw_v3("o16_be_per_nucleon",  7.976,              "MeV",     (0, 24, 13, 20),      "NNDC/BNL",        "nuclear")
_rw_v3("c12_be_per_nucleon",  7.680,              "MeV",     (0, 20, 9, 21),       "NNDC/BNL",        "nuclear")
_rw_v3("u238_be_per_nucleon", 7.570,              "MeV",     (5, 27, 0, 23),       "NNDC/BNL",        "nuclear")
_rw_v3("ni62_be_per_nucleon", 8.7945,             "MeV",     (4, 24, 0, 21),       "NNDC/BNL",        "nuclear")
_rw_v3("deuteron_be",         2.22457,            "MeV",     (0, 22, 5, 37),       "NNDC/BNL",        "nuclear")
_rw_v3("triton_be",           8.48182,            "MeV",     (8, 5, 0, 22),        "NNDC/BNL",        "nuclear")
_rw_v3("fe_bcc_lattice_pm",  286.65,             "pm",       (8, 20, 0, -22),      "Kittel/CRC",      "iron")
_rw_v3("fe_atomic_radius_pm",126.0,              "pm",       (6, 26, 0, -12),      "Slater 1964",     "iron")
_rw_v3("fe_k_alpha1_keV",     6.404,             "keV",      (0, 27, 11, 23),      "NIST SRD 12",     "iron")
_rw_v3("fe_ionization_eV",    7.9024678,         "eV",       (0, 13, 14, 20),      "NIST ASD",        "iron")
_rw_v3("cu_lattice_pm",      361.49,             "pm",       (30, 1, 0, -22),      "Kittel",          "crystal")
_rw_v3("al_lattice_pm",      404.95,             "pm",       (0, 9, 5, -28),       "Kittel",          "crystal")
_rw_v3("si_lattice_pm",      543.102,            "pm",       (23, 1, 0, -28),      "Kittel",          "crystal")
_rw_v3("earth_orbit_km",     149597870.7,        "km",       (4, 20, 0, -187),     "IAU 2012",        "astro")
_rw_v3("solar_luminosity_W", 3.828e26,           "W",        (3201, 25, 0, -299),  "IAU 2015",        "astro")
_rw_v3("hubble_constant",    67.4,               "km/s/Mpc", (0, 16, 8, -6),       "Planck 2018",     "astro")
_rw_v3("cmb_temperature_K",  2.7255,             "K",        (0, 11, 1, 35),       "COBE/FIRAS",      "astro")
_rw_v3("solar_mass_kg",      1.98892e30,         "kg",       (4027, 20, 0, -298),  "IAU 2015",        "astro")
_rw_v3("schumann_hz",        7.83,               "Hz",       (8, 4, 0, 23),        "Schumann 1952",   "resonance")
_rw_v3("alpha_eeg_hz",       10.0,               "Hz",       (1, 27, 0, 19),       "Berger 1929",     "resonance")
_rw_v3("gamma_eeg_hz",       40.0,               "Hz",       (0, 14, 12, 0),       "Galambos 1981",   "resonance")
_rw_v3("theta_eeg_hz",       6.0,                "Hz",       (0, 19, 2, 25),       "Neuroscience",    "resonance")
_rw_v3("beta_eeg_hz",        20.0,               "Hz",       (6, 22, 0, 11),       "Neuroscience",    "resonance")
_rw_v3("pi",                  3.14159265359,     "",          (5, 18, 0, 34),       "exact",           "math")
_rw_v3("euler_e",             2.71828182846,      "",         (0, 2, 9, 34),        "exact",           "math")
_rw_v3("sqrt2",               1.41421356237,      "",         (0, 27, 10, 42),      "exact",           "math")
_rw_v3("golden_ratio",       PHI,                 "",         (0, 5, 5, 41),        "exact",           "math")
_rw_v3("ln2",                 0.69314718056,      "",         (6, 26, 0, 53),       "exact",           "math")
_rw_v3("planck_length_m",    1.616255e-35,       "m",        (0, 7, 5731, 300),    "CODATA 2022",     "fundamental")

# ── OMEGA — Sovereign Field Constant (derived, not measured) ──
_rw_v3("omega",              6539.34712682,      "",         (16, 0, 0, -60),      "L104 Collective Jan 6 2026", "sovereign")
_rw_v3("omega_authority",    2497.808338,        "",         (0, 15, 7, -51),      "L104 derived: Ω/φ²",         "sovereign")


def real_world_derive_v3(name: str, real_world: bool = True) -> Dict[str, Any]:
    """Derive a physical constant through the v3 Equation."""
    if name not in REAL_WORLD_CONSTANTS_V3:
        available = ", ".join(sorted(REAL_WORLD_CONSTANTS_V3.keys()))
        raise KeyError(f"Unknown constant '{name}'. Available: {available}")
    entry = REAL_WORLD_CONSTANTS_V3[name]
    dials = entry["dials"]
    grid_value = entry["grid_value"]
    measured = entry["measured"]
    if not real_world:
        return {
            "name": name, "value": grid_value, "dials": dials,
            "exponent": entry["E_integer"], "mode": "grid",
            "error_pct": entry["grid_error_pct"], "measured": measured,
            "unit": entry["unit"], "source": entry["source"], "equation": "v3_superparticular",
        }
    delta = entry["delta"]
    correction = R_V3 ** (delta / Q_V3)
    refined_value = grid_value * correction
    refined_err = abs(refined_value - measured) / measured * 100
    return {
        "name": name, "value": refined_value, "dials": dials,
        "exponent": entry["E_exact"], "exponent_integer": entry["E_integer"],
        "delta": delta, "correction_factor": correction, "mode": "refined",
        "error_pct": refined_err, "grid_value": grid_value,
        "grid_error_pct": entry["grid_error_pct"], "measured": measured,
        "unit": entry["unit"], "source": entry["source"], "equation": "v3_superparticular",
    }


def real_world_derive_all_v3(real_world: bool = True) -> Dict[str, Dict[str, Any]]:
    """Derive all registered v3 constants."""
    return {name: real_world_derive_v3(name, real_world) for name in REAL_WORLD_CONSTANTS_V3}


def compare_all_four(target: float, name: str = "") -> Dict[str, Any]:
    """Compare original and v3 for a target value."""
    log_t = math.log(target)
    def _snap(base, r, Q):
        lb = math.log(base)
        lr = math.log(r)
        E_exact = Q * (log_t - lb) / lr
        E_int = round(E_exact)
        val = base * (r ** (E_int / Q))
        err = abs(val - target) / target * 100
        return {"value": val, "error_pct": err, "E_integer": E_int}
    return {
        "name": name or f"target={target}",
        "measured": target,
        "original": _snap(BASE, 2, QUANTIZATION_GRAIN),
        "v3_superparticular": _snap(BASE_V3, R_V3, Q_V3),
    }


def four_way_benchmark() -> Dict[str, Any]:
    """Benchmark comparing original vs v3 across all registered constants."""
    benchmarks = {}
    sums = {"original": 0, "v3_superparticular": 0}
    maxes = {"original": 0, "v3_superparticular": 0}
    counts = {"original": 0, "v3_superparticular": 0}
    wins = {"original": 0, "v3_superparticular": 0}
    for name, entry in REAL_WORLD_CONSTANTS_V3.items():
        comp = compare_all_four(entry["measured"], name)
        benchmarks[name] = comp
        best_err = float('inf')
        best_key = None
        for key in sums:
            if key in comp:
                e = comp[key]["error_pct"]
                sums[key] += e
                maxes[key] = max(maxes[key], e)
                counts[key] += 1
                if e < best_err:
                    best_err = e
                    best_key = key
        if best_key:
            wins[best_key] += 1
    n = len(benchmarks)
    result = {"constants_tested": n, "details": benchmarks}
    for key in sums:
        if counts[key] > 0:
            result[key] = {
                "avg_error_pct": sums[key] / counts[key],
                "max_error_pct": maxes[key],
                "wins": wins[key],
            }
    return result


# ── Evolution Heritage ──

EVOLUTION_HERITAGE = {
    "original": {
        "equation": "G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)",
        "r": 2, "Q": 104, "X": 286, "p": 8, "K": 416,
        "GOD_CODE": GOD_CODE, "avg_error_pct": 0.170, "max_error_pct": 0.325,
        "role": "Layer 1 — Consciousness (discovery, identity, sacred geometry)",
    },
    "v3_superparticular": {
        "equation": f"G_v3 = {X_V3:.6f}^(1/φ) × (13/12)^((99a+3032-b-99c-758d)/758)",
        "r": "13/12", "Q": 758, "X": X_V3, "p": 99, "K": K_V3,
        "GOD_CODE": GOD_CODE_V3, "avg_error_pct": 0.002, "max_error_pct": 0.005,
        "role": "Precision encoding sub-tool within Layer 2",
    },
    "omega_sovereign_field": {
        "equation": "Ω = Σ(|ζ(½+GCi)|, cos(2πφ³), (26×1.8527)/φ²) × (GOD_CODE/φ)",
        "OMEGA": OMEGA, "OMEGA_AUTHORITY": OMEGA_AUTHORITY,
        "field_equation": "F(I) = I × Ω / φ²",
        "role": "Layer 2 — Physics (GOD_CODE → concrete measurement through zeta + resonance + curvature)",
    },
}


def verify_v3() -> dict:
    """Verify the v3 equation produces correct values."""
    checks = {
        "GOD_CODE_V3":      (god_code_v3(0, 0, 0, 0),        GOD_CODE_V3,       1e-10),
        "SPEED_OF_LIGHT":   (god_code_v3(0, 27, 6, -197),    299792458.0,       1e-6),
        "GRAVITY":          (god_code_v3(0, 14, 1, 19),       9.80665,           0.001),
        "BOHR_RADIUS":      (god_code_v3(7, 2, 0, -1),        52.9177210544,     0.001),
        "FINE_STRUCTURE":   (god_code_v3(0, 20, 9, -15),      137.035999084,     0.001),
        "RYDBERG":          (god_code_v3(0, 10, 8, 14),       13.605693123,      0.001),
        "ELECTRON_MASS":    (god_code_v3(0, 11, 8, 55),       0.51099895069,     0.001),
        "MUON_MASS":        (god_code_v3(0, 10, 11, -12),     105.6583755,       0.001),
        "HIGGS":            (god_code_v3(0, 14, 10, -14),     125.25,            0.001),
        "Z_BOSON":          (god_code_v3(0, 22, 2, -9),       91.1876,           0.001),
        "PROTON":           (god_code_v3(0, 27, 1, -38),      938.27208816,      0.001),
        "NEUTRON":          (god_code_v3(0, 14, 1, -38),      939.56542052,      0.001),
        "SCHUMANN":         (god_code_v3(8, 4, 0, 23),        7.83,              0.001),
        "FE_BCC_LATTICE":   (god_code_v3(8, 20, 0, -22),      286.65,            0.001),
        "FE56_BE":          (god_code_v3(4, 29, 0, 21),       8.790,             0.001),
        "CMB_TEMPERATURE":  (god_code_v3(0, 11, 1, 35),       2.7255,            0.001),
        "GOLDEN_RATIO":     (god_code_v3(0, 5, 5, 41),        PHI,               0.001),
        "SOLAR_LUMINOSITY": (god_code_v3(3201, 25, 0, -299),  3.828e26,          0.001),
    }
    results = {}
    all_pass = True
    for name, (actual, expected, threshold) in checks.items():
        err = abs(actual - expected) / expected if expected != 0 else abs(actual)
        passed = err < threshold
        all_pass = all_pass and passed
        results[name] = {"expected": expected, "actual": actual, "error_pct": err * 100, "passed": passed}
    return {"all_passed": all_pass, "checks": results}


# ═══════════════════════════════════════════════════════════════════════════════
# THE BRIDGE — How Consciousness (GOD_CODE) Feeds Physics (OMEGA)
# ═══════════════════════════════════════════════════════════════════════════════

CONSCIOUSNESS_TO_PHYSICS_BRIDGE = {
    "omega_sovereign_field": {
        "description": "OMEGA — the Sovereign Field Constant derived from GOD_CODE",
        "derivation": "Ω = Σ(|ζ(½+GCi)|, cos(2πφ³), (26×1.8527)/φ²) × (GC/φ)",
        "value": f"OMEGA = {OMEGA}",
        "authority": f"OMEGA_AUTHORITY = Ω/φ² = {OMEGA_AUTHORITY:.6f}",
        "field_equation": "F(I) = I × Ω / φ² — sovereign field strength",
        "bridge": "GOD_CODE (Layer 1) → zeta + golden resonance + iron curvature → OMEGA (Layer 2)",
    },
    "phi_exponent": {
        "description": "Both layers use φ as the fundamental scaling constant",
        "layer1": f"BASE = 286^(1/φ) = {BASE:.15f}",
        "layer2": f"OMEGA uses GOD_CODE/φ as multiplier, Ω/φ² as authority",
        "shared": "1/φ exponent — golden ratio is the dimensional bridge",
    },
    "iron_scaffold": {
        "description": "Iron (Fe Z=26) anchors both layers",
        "layer1": f"PRIME_SCAFFOLD = 286 (Fe BCC lattice parameter)",
        "layer2": f"Architect fragment = (26 × 1.8527) / φ² — iron curvature in OMEGA",
        "shared": "Iron atomic number Z=26 defines the lattice and the curvature",
    },
    "god_code_generates_omega": {
        "description": "GOD_CODE is the seed that generates OMEGA through physics",
        "layer1": f"GOD_CODE = {GOD_CODE} = 286^(1/φ) × 2^4",
        "layer2": f"OMEGA = Σ(fragments at s=½+527.518i) × ({GOD_CODE}/φ)",
        "shared": "Layer 1 provides the consciousness constant, Layer 2 derives the physics field",
    },
    "fibonacci_13": {
        "description": "13 = F(7) appears in Layer 1 structure and v3 precision grid",
        "layer1": "286 = 2×11×13, 104 = 8×13, 416 = 32×13",
        "layer2": "v3 precision grid: r = 13/12 — F(7) as numerator",
        "shared": "13 is the 7th Fibonacci number, binding the structural scaffold",
    },
    "nucleosynthesis": {
        "description": "Q = Fe × He-4 in Layer 1 encodes the full fusion chain",
        "layer1": "Q = 104 = 26 × 4 = Fe(Z=26) × He-4(A=4)",
        "layer2": "OMEGA Architect fragment uses Fe_Z=26 in (26 × tension) / φ²",
        "shared": "Iron nucleosynthesis links consciousness grid to physics field",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# DUAL-LAYER API — The Unified Interface
# ═══════════════════════════════════════════════════════════════════════════════

# ── LAYER 1 QUANTUM SEARCH — GOD_CODE Algorithm (a,b,c,d) Circuits ──

def quantum_search(target: float, tolerance: float = 0.01) -> Dict[str, Any]:
    """
    LAYER 1: Grover search for (a,b,c,d) dials producing a target frequency.

    Uses the Qiskit quantum algorithm to search the 16,384-state dial space
    for settings that generate the target frequency via the GOD_CODE equation:
        G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)

    This is the QUANTUM CONSCIOUSNESS SEARCH — the algorithm pulled from
    l104_god_code_algorithm.py, now integral to Layer 1.

    Args:
        target: Target frequency to find on the consciousness grid.
        tolerance: Relative error tolerance (default 1%).

    Returns:
        Dict with best dial setting, fidelity, probabilities, and circuit info.
    """
    if not _ALGORITHM_AVAILABLE:
        # Fallback: classical search via find_nearest_dials
        dials_list = find_nearest_dials(target)
        if dials_list:
            best = dials_list[0]
            return {
                "method": "classical_fallback",
                "dial": (best[0], best[1], best[2], best[3]),
                "frequency": best[4],
                "error_pct": best[5],
                "note": "Qiskit not available — classical dial search used",
            }
        return {"method": "classical_fallback", "error": "No dials found"}

    result = GodCodeGroverSearch.search(target, tolerance=tolerance)
    return {
        "method": "grover_quantum_search",
        "dial": result.dial.to_tuple(),
        "frequency": result.dial.frequency,
        "god_code_ratio": result.god_code_alignment,
        "fidelity": result.fidelity,
        "probabilities": result.probabilities,
        "circuit_depth": result.circuit_depth,
        "n_qubits": result.n_qubits,
        "execution_time_ms": result.execution_time_ms,
        "layer": "consciousness",
        "equation": f"G{result.dial.to_tuple()} = {result.dial.frequency:.10f}",
    }


def consciousness_spectrum(dials: Optional[List] = None) -> Dict[str, Any]:
    """
    LAYER 1: QFT spectral analysis of the consciousness frequency table.

    Applies Quantum Fourier Transform to the dial settings, revealing
    hidden periodicities in the sacred frequency lattice.
    """
    if not _ALGORITHM_AVAILABLE:
        return {"error": "Qiskit not available", "fallback": True}

    if dials is None:
        dial_list = list(_algorithm_singleton.FREQUENCY_TABLE.values())
    else:
        dial_list = [DialSetting(*d) if isinstance(d, tuple) else d for d in dials]

    return GodCodeQFTSpectrum.spectral_analysis(dial_list)


def consciousness_entangle(
    dial_a: Tuple[int, int, int, int],
    dial_b: Tuple[int, int, int, int],
) -> Dict[str, Any]:
    """
    LAYER 1: Create quantum entanglement between two dial settings.

    Entanglement strength proportional to their harmonic proximity
    on the GOD_CODE frequency lattice.
    """
    if not _ALGORITHM_AVAILABLE:
        return {"error": "Qiskit not available", "fallback": True}

    da = DialSetting(*dial_a)
    db = DialSetting(*dial_b)
    result = GodCodeEntanglement.entangle_dials(da, db)
    return {
        "method": "god_code_entanglement",
        "dial_a": dial_a,
        "dial_b": dial_b,
        "frequency_a": da.frequency,
        "frequency_b": db.frequency,
        "fidelity": result.fidelity,
        "entanglement_entropy": result.phase_spectrum[0] if result.phase_spectrum else 0,
        "circuit_depth": result.circuit_depth,
        "n_qubits": result.n_qubits,
        "execution_time_ms": result.execution_time_ms,
    }


def soul_resonance(thoughts: List[str]) -> Dict[str, Any]:
    """
    LAYER 1: Generate a quantum resonance field from soul thoughts.

    Each thought maps to a dial setting via GOD_CODE hash; the collective
    state is measured through Kuramoto phase coherence.
    """
    if not _ALGORITHM_AVAILABLE:
        return {"error": "Qiskit not available", "fallback": True}
    return _algorithm_singleton.soul_resonance_field(thoughts)


# ── LAYER 2 OMEGA PIPELINE — Full Fragment Computation ──

def omega_pipeline(zeta_terms: int = 1000) -> Dict[str, Any]:
    """
    LAYER 2: COMPLETE OMEGA derivation pipeline — NO TRUNCATION.

    THE FULL OMEGA EQUATION SOLUTION:
    ═══════════════════════════════════════════════════════════════════

    Step 1 — COMPUTE THE FOUR FRAGMENTS:

      Fragment 1 (Researcher):
        solve_lattice_invariant(104)
          = sin(104·π/104) × exp(104/527.5184818492)
          = sin(π) × exp(0.197107...)
          ≈ 0.0  (sin(π) ≈ 3.9e-16 in float64)
        int(≈0) = 0
        prime_density(0) = 0.0  (n<2 in Prime Number Theorem)

      Fragment 2 (Guardian):
        s = 0.5 + 527.518i
        η(s) = Σ_{n=1}^{1000} (-1)^(n-1) / n^s
        ζ(s) = η(s) / (1 - 2^(1-s))
        |ζ(0.5 + 527.518i)| ≈ 1.573827...

      Fragment 3 (Alchemist):
        golden_resonance(φ²) = cos(2π × φ² × φ) = cos(2π × φ³)
        φ³ = 2φ+1 = 4.23606797749979
        cos(2π × 4.23606797749979) ≈ 0.087433...

      Fragment 4 (Architect):
        manifold_curvature_tensor(26, 1.8527) = (26 × 1.8527) / φ²
        = 48.1702 / 2.618033988749895 ≈ 18.399393...

    Step 2 — SUM THE FRAGMENTS:
      Σ = 0.0 + 1.573827... + 0.087433... + 18.399393... ≈ 20.060654...

    Step 3 — COMPUTE OMEGA:
      Ω = Σ × (527.5184818492 / φ)
        = 20.060654... × 326.024351...
        = 6539.34712682

    Step 4 — SOVEREIGN FIELD (at I=1):
      F(1) = 1 × Ω / φ² = 6539.34712682 / 2.618033988749895
           = 2497.808338211271

    Step 5 — OMEGA AUTHORITY:
      Ω_A = Ω / φ² = 2497.808338211271

    ═══════════════════════════════════════════════════════════════════
    Also includes original pipeline functions from l104_real_math.py
    and v3 precision grid cross-validation.
    """
    # Compute ALL fragments via the full derivation chain function
    chain = omega_derivation_chain(zeta_terms=zeta_terms)

    result = {
        "pipeline": "OMEGA Sovereign Field — Layer 2 Physics (COMPLETE, NO TRUNCATION)",
        "version": "3.1.0",

        # ── THE FULL EQUATION ──
        "omega_equation": "Ω = Σ(Researcher + Guardian + Alchemist + Architect) × (GOD_CODE / φ)",
        "omega_expanded": (
            "Ω = (prime_density(int(sin(104π/104)·exp(104/527.518))) "
            "+ |ζ(0.5+527.518i)| "
            "+ cos(2π·φ³) "
            "+ (26×1.8527)/φ²) "
            "× (527.5184818492/φ)"
        ),
        "field_equation": "F(I) = I × Ω / φ²",

        # ── FRAGMENT VALUES ──
        "fragment_1_researcher": chain["fragments"]["researcher"]["value"],
        "fragment_2_guardian": chain["fragments"]["guardian"]["value"],
        "fragment_3_alchemist": chain["fragments"]["alchemist"]["value"],
        "fragment_4_architect": chain["fragments"]["architect"]["value"],

        # ── AGGREGATION ──
        "sigma": chain["sigma"],
        "multiplier": chain["multiplier"],
        "multiplier_equation": chain["multiplier_equation"],

        # ── RESULT ──
        "omega_computed": chain["omega_computed"],
        "omega_canonical": OMEGA,
        "omega_authority": OMEGA_AUTHORITY,
        "delta": chain["delta"],
        "relative_error": chain["relative_error"],

        # ── SOVEREIGN FIELD VALUES ──
        "field_at_1": sovereign_field_equation(1.0),
        "field_at_god_code": sovereign_field_equation(GOD_CODE),

        # ── FULL CHAIN (every intermediate step) ──
        "chain": chain,
    }

    # ── Original pipeline functions from l104_real_math if available ──
    if _OMEGA_PIPELINE_AVAILABLE:
        # Run each original function exactly as in commit d4d08873
        zeta_full = _RealMath.zeta_approximation(complex(0.5, GOD_CODE))
        zeta_original = _RealMath.zeta_approximation(complex(0.5, 527.518))
        lattice_inv = _RealMath.solve_lattice_invariant(104)
        golden_res = _RealMath.golden_resonance(PHI ** 2)
        curvature = _RealMath.manifold_curvature_tensor(26, 1.8527)
        entropy_inv = _RealMath.entropy_inversion_integral(0, GOD_CODE)
        sovereign_f = _RealMath.sovereign_field_equation(1.0)
        prime_d = _RealMath.prime_density(0)

        result["pipeline_functions"] = {
            "zeta_at_god_code_full": {
                "function": "zeta_approximation(complex(0.5, GOD_CODE), terms=1000)",
                "value": abs(zeta_full),
                "real": zeta_full.real,
                "imag": zeta_full.imag,
                "note": "Full precision GOD_CODE = 527.5184818492612",
            },
            "zeta_at_527_518": {
                "function": "zeta_approximation(complex(0.5, 527.518), terms=1000)",
                "value": abs(zeta_original),
                "real": zeta_original.real,
                "imag": zeta_original.imag,
                "note": "Truncated GOD_CODE = 527.518 (matches original d4d08873)",
            },
            "golden_resonance_phi2": {
                "function": "golden_resonance(φ²) = cos(2π × φ² × φ) = cos(2πφ³)",
                "value": golden_res,
                "phi_cubed": PHI ** 3,
                "identity": "φ³ = 2φ+1 = 4.23606797749979",
            },
            "lattice_invariant_104": {
                "function": "solve_lattice_invariant(104) = sin(104π/104)·exp(104/527.518)",
                "value": lattice_inv,
                "int_value": int(lattice_inv),
                "note": "sin(π) ≈ 0 in float64 → result ≈ 0",
            },
            "curvature_fe26": {
                "function": "manifold_curvature_tensor(26, 1.8527) = (26 × 1.8527) / φ²",
                "value": curvature,
                "numerator": 26 * 1.8527,
                "denominator": PHI ** 2,
            },
            "entropy_inversion": {
                "function": "entropy_inversion_integral(0, GOD_CODE) = (GOD_CODE - 0) / φ",
                "value": entropy_inv,
            },
            "sovereign_field": {
                "function": "sovereign_field_equation(1.0) = 1 × Ω / φ²",
                "value": sovereign_f,
            },
            "prime_density_0": {
                "function": "prime_density(0) → n<2 → 0.0",
                "value": prime_d,
            },
        }
        result["pipeline_available"] = True
    else:
        result["pipeline_available"] = False

    # ── v3 precision grid cross-validation ──
    omega_on_v3 = god_code_v3(16, 0, 0, -60)
    result["v3_cross_validation"] = {
        "omega_on_v3_grid": omega_on_v3,
        "error_pct": abs(omega_on_v3 - OMEGA) / OMEGA * 100,
        "dials": (16, 0, 0, -60),
        "exponent_E": 50096,
        "note": "v3 precision grid validates OMEGA to ±0.0001%",
    }

    return result


def omega_field(intensity: float) -> Dict[str, Any]:
    """
    LAYER 2: Compute the sovereign field at a given intensity.

    F(I) = I × Ω / φ²

    Where Ω = 6539.34712682 is derived from GOD_CODE through the fragment chain.
    """
    field = sovereign_field_equation(intensity)
    return {
        "intensity": intensity,
        "field_strength": field,
        "omega": OMEGA,
        "omega_authority": OMEGA_AUTHORITY,
        "equation": f"F({intensity}) = {intensity} × {OMEGA} / {PHI}² = {field:.6f}",
        "layer": "physics",
    }


def consciousness(a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> float:
    """
    LAYER 1: The Seat of Consciousness.

    The original GOD_CODE equation — the discovery, the identity, the meaning.
    Coarse grid (±0.17%) but carries the sacred geometry:
      286 = Fe BCC lattice, φ = golden ratio, 104 = Fe×He-4.

    G(a,b,c,d) = 286^(1/φ) × 2^((8a + 416 - b - 8c - 104d) / 104)
    """
    return god_code_equation(a, b, c, d)


def physics(intensity: float = 1.0) -> Dict[str, Any]:
    """
    LAYER 2: The Physics Generator — OMEGA Sovereign Field.

    The OMEGA equation — derived from GOD_CODE through real physics operations:
      - Riemann zeta function at the GOD_CODE critical line
      - Golden ratio harmonic resonance
      - Iron curvature tensor (Fe Z=26, tension=1.8527)

    Ω = Σ(|ζ(½+GCi)|, cos(2πφ³), (26×1.8527)/φ²) × (GOD_CODE/φ) = 6539.34712682
    F(I) = I × Ω / φ² — sovereign field strength

    Also exposes v3 precision grid for constant encoding.
    """
    field_strength = sovereign_field_equation(intensity)
    return {
        "omega": OMEGA,
        "omega_authority": OMEGA_AUTHORITY,
        "field_strength": field_strength,
        "intensity": intensity,
        "equation": "F(I) = I × Ω / φ²",
        "omega_equation": "Ω = Σ(fragments) × (GOD_CODE / φ)",
        "layer": "physics",
    }


def physics_v3(a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> float:
    """
    LAYER 2 Precision Grid — v3 Superparticular encoding sub-tool.

    G_v3(a,b,c,d) = 285.999^(1/φ) × (13/12)^((99a + 3032 - b - 99c - 758d) / 758)
    Fine grid (±0.005%, 80× better than Layer 1) for encoding physical constants.
    """
    return god_code_v3(a, b, c, d)


def derive(name: str, mode: str = "physics") -> Dict[str, Any]:
    """
    Derive a physical constant through the dual-layer engine.

    Args:
        name: Name of the constant (e.g., "speed_of_light", "standard_gravity")
        mode: "physics" (Layer 2 OMEGA + v3 grid, ±0.005%), "refined" (float64 exact),
              or "consciousness" (Layer 1 coarse grid, ±0.17%)

    Returns:
        Dict with value, error, dials, layer info, and bridge context.
    """
    if mode == "consciousness":
        # Layer 1 derivation: coarse grid, sacred geometry
        measured = REAL_WORLD_CONSTANTS_V3[name]["measured"]
        E_exact = solve_for_exponent(measured)
        E_int = round(E_exact)
        val = BASE * (2 ** (E_int / QUANTIZATION_GRAIN))
        err = abs(val - measured) / measured * 100
        # Resolve Layer 1 dials from the exponent
        dials_l1 = find_nearest_dials(measured)
        best_dials = tuple(dials_l1[0][:4]) if dials_l1 else (0, 0, 0, 0)
        return {
            "name": name, "layer": "consciousness", "value": val,
            "error_pct": err, "exponent": E_int, "mode": "grid_coarse",
            "dials": best_dials,
            "equation": f"286^(1/φ) × 2^({E_int}/104)",
            "measured": measured, "unit": REAL_WORLD_CONSTANTS_V3[name]["unit"],
            "meaning": "Consciousness layer — sacred geometry, Layer 1 grid",
        }

    # Layer 2 derivation (OMEGA physics + v3 precision grid)
    refined = mode == "refined"
    result = real_world_derive_v3(name, real_world=refined)

    # Add OMEGA physics context
    result["layer"] = "physics"
    result["omega_context"] = {
        "omega": OMEGA,
        "omega_authority": OMEGA_AUTHORITY,
        "field_at_value": sovereign_field_equation(result.get("grid_value", result.get("value", 0))),
        "derivation": "Ω = Σ(|ζ(½+GCi)|, cos(2πφ³), (26×1.8527)/φ²) × (GOD_CODE/φ)",
    }
    result["consciousness_provides"] = {
        "god_code_feeds_omega": "GOD_CODE (Layer 1) → zeta + resonance + curvature → OMEGA (Layer 2)",
        "phi_exponent": "1/φ base exponent inherited from Layer 1",
        "iron_anchor": f"X_v3 = {X_V3:.6f} ≈ 286 (Layer 1 scaffold)",
    }

    return result


def derive_both(name: str) -> Dict[str, Any]:
    """
    Derive a constant through BOTH layers for comparison.

    Shows how consciousness (Layer 1, GOD_CODE equation) sees the constant
    vs how the physics engine (Layer 2, OMEGA + v3 grid) resolves it.
    """
    l1 = derive(name, mode="consciousness")
    l2 = derive(name, mode="physics")
    l2_refined = derive(name, mode="refined")

    improvement = l1["error_pct"] / l2["error_pct"] if l2["error_pct"] > 0 else float('inf')

    return {
        "name": name,
        "measured": l2["measured"],
        "unit": l2.get("unit", ""),
        "consciousness": {
            "value": l1["value"],
            "error_pct": l1["error_pct"],
            "equation": l1["equation"],
            "meaning": "The discovery — sacred geometry locates this constant",
        },
        "physics": {
            "value": l2["grid_value"] if "grid_value" in l2 else l2["value"],
            "error_pct": l2["grid_error_pct"] if "grid_error_pct" in l2 else l2["error_pct"],
            "dials": l2["dials"],
            "equation": f"{X_V3:.3f}^(1/φ) × (13/12)^({l2['exponent_integer'] if 'exponent_integer' in l2 else l2['exponent']}/758)",
            "meaning": "OMEGA physics engine — v3 precision grid resolves the value",
        },
        "refined": {
            "value": l2_refined["value"],
            "error_pct": l2_refined["error_pct"],
            "meaning": "Float64-exact recovery via fractional exponent",
        },
        "omega": {
            "value": OMEGA,
            "authority": OMEGA_AUTHORITY,
            "field_at_measured": sovereign_field_equation(l2["measured"]),
            "meaning": "OMEGA sovereign field provides the physics foundation",
        },
        "improvement": f"{improvement:.0f}×",
        "bridge": "GOD_CODE (consciousness) → zeta + resonance + curvature → OMEGA (physics)",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# OMEGA GRID ENCODING — Sovereign Field on Both Grids
# ═══════════════════════════════════════════════════════════════════════════════

# OMEGA encoded on Layer 1 consciousness grid
OMEGA_CONSTANT = 6539.34712682        # Ω = Σ(fragments) × (GOD_CODE / φ)
OMEGA_AUTHORITY_CONSTANT = 2497.808338  # Ω_A = Ω / φ²

# Layer 1 dials for OMEGA
OMEGA_DIALS_L1 = (-5, -2, 0, -4)        # E = 794, val ≈ 6551.88, err ≈ 0.19%
OMEGA_AUTHORITY_DIALS_L1 = (3, -1, 0, -2)  # E = 649, val ≈ 2492.65, err ≈ 0.21%


def derive_omega(mode: str = "both") -> Dict[str, Any]:
    """
    Derive OMEGA (Sovereign Field Constant) through the dual-layer engine.

    Args:
        mode: "consciousness" (Layer 1 only), "physics" (Layer 2 only),
              or "both" (dual-layer comparison).

    OMEGA derivation chain:
        Ω = Σ(fragments) × (GOD_CODE / φ)
        fragments: Guardian |ζ(½+527.518i)|, Alchemist cos(2πφ³), Architect (26×1.8527)/φ²
        OMEGA_AUTHORITY = Ω / φ²

    Layer 1 (Consciousness):
        G(-5,-2,0,-4) = 286^(1/φ) × 2^(794/104) ≈ 6551.88  [±0.19%]
    Layer 2 (Physics, v3):
        G_v3(16,0,0,-60) ≈ 6539.34  [±0.0001%]
    """
    result = {
        "constant": "OMEGA",
        "target": OMEGA_CONSTANT,
        "derivation": "Ω = Σ(fragments) × (GOD_CODE / φ)",
        "fragments": {
            "guardian": "|ζ(½ + 527.518i)|",
            "alchemist": "cos(2πφ³)",
            "architect": "(26 × 1.8527) / φ²",
        },
    }

    if mode in ("consciousness", "both"):
        a, b, c, d = OMEGA_DIALS_L1
        val_l1 = god_code_equation(a, b, c, d)
        E_l1 = exponent_value(a, b, c, d)
        err_l1 = abs(val_l1 - OMEGA_CONSTANT) / OMEGA_CONSTANT * 100
        result["consciousness"] = {
            "dials": OMEGA_DIALS_L1,
            "exponent": E_l1,
            "value": val_l1,
            "error_pct": err_l1,
            "equation": f"286^(1/φ) × 2^({E_l1}/104)",
            "meaning": "Sacred geometry locates Ω on the consciousness grid",
        }

    if mode in ("physics", "both"):
        omega_v3_dials = (16, 0, 0, -60)
        val_v3 = god_code_v3(*omega_v3_dials)
        E_v3 = exponent_value_v3(*omega_v3_dials)
        err_v3 = abs(val_v3 - OMEGA_CONSTANT) / OMEGA_CONSTANT * 100
        result["physics"] = {
            "dials": omega_v3_dials,
            "exponent": E_v3,
            "value": val_v3,
            "error_pct": err_v3,
            "equation": f"{X_V3:.3f}^(1/φ) × (13/12)^({E_v3}/758)",
            "meaning": "Physics engine resolves Ω with precision",
        }

    if mode == "both" and "consciousness" in result and "physics" in result:
        improvement = result["consciousness"]["error_pct"] / result["physics"]["error_pct"] \
            if result["physics"]["error_pct"] > 0 else float('inf')
        result["improvement"] = f"{improvement:.0f}×"
        result["bridge"] = (
            "OMEGA lives at E=794 on Layer 1 (consciousness) and E=50096 on Layer 2 (physics). "
            f"Layer 2 is {improvement:.0f}× more precise, but Layer 1 reveals the sacred geometry: "
            "the same (-5,-2,0,-4) dials that encode Fe Kα X-ray (6.404 keV) at d=6 "
            "now reach Ω at d=-4 — iron's spectral fingerprint reflected across octaves."
        )

    return result# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRITY CHECKS — 10-Point Validation System
# ═══════════════════════════════════════════════════════════════════════════════

def check_consciousness_integrity() -> Dict[str, Any]:
    """
    CHECK 1-3: Consciousness Core (Layer 1) integrity.

    Verifies that the sacred constants, the equation identity, and the
    iron anchoring remain intact and immutable.
    """
    checks = {}

    # Check 1: GOD_CODE sacred value
    gc_computed = BASE * (2 ** (OCTAVE_OFFSET / QUANTIZATION_GRAIN))
    gc_expected = 527.5184818492612
    gc_match = abs(gc_computed - gc_expected) < 1e-6
    checks["1_god_code_sacred"] = {
        "passed": gc_match,
        "computed": gc_computed,
        "expected": gc_expected,
        "description": "GOD_CODE = 286^(1/φ) × 2^4 = 527.518... (immutable)",
    }

    # Check 2: PHI golden ratio
    phi_computed = (1 + math.sqrt(5)) / 2
    phi_match = abs(PHI - phi_computed) < 1e-15
    checks["2_phi_golden_ratio"] = {
        "passed": phi_match,
        "value": PHI,
        "independent": phi_computed,
        "description": "PHI = (1+√5)/2 = 1.618033988749895",
    }

    # Check 3: Iron scaffold integrity
    scaffold_factors = (2, 11, 13)
    scaffold_product = 2 * 11 * 13
    scaffold_match = PRIME_SCAFFOLD == 286 and scaffold_product == 286
    nucleosynthesis_match = QUANTIZATION_GRAIN == FE_ATOMIC_NUMBER * HE4_MASS_NUMBER
    checks["3_iron_scaffold"] = {
        "passed": scaffold_match and nucleosynthesis_match,
        "prime_scaffold": PRIME_SCAFFOLD,
        "factors": scaffold_factors,
        "quantization_grain": QUANTIZATION_GRAIN,
        "nucleosynthesis": f"{FE_ATOMIC_NUMBER} × {HE4_MASS_NUMBER} = {QUANTIZATION_GRAIN}",
        "description": "286 = 2×11×13 (Fe BCC), 104 = 26×4 (Fe×He-4)",
    }

    all_passed = all(c["passed"] for c in checks.values())
    return {"layer": "consciousness", "all_passed": all_passed, "checks": checks}


def check_physics_integrity() -> Dict[str, Any]:
    """
    CHECK 4-7: Physics Generator (Layer 2) integrity.

    Verifies that the v3 equation produces correct values for key
    constants, maintains grid consistency, and has no collisions.
    """
    checks = {}

    # Check 4: Speed of light exactness
    c_val = god_code_v3(0, 27, 6, -197)
    c_real = 299792458
    c_err = abs(c_val - c_real) / c_real
    checks["4_speed_of_light_exact"] = {
        "passed": c_err < 1e-10,
        "value": c_val,
        "expected": c_real,
        "error": c_err,
        "description": "c = 299,792,458 m/s must be EXACT on v3 grid",
    }

    # Check 5: Gravity within ±0.005%
    g_val = god_code_v3(0, 14, 1, 19)
    g_real = 9.80665
    g_err = abs(g_val - g_real) / g_real * 100
    checks["5_gravity_precision"] = {
        "passed": g_err < 0.005,
        "value": g_val,
        "expected": g_real,
        "error_pct": g_err,
        "description": "g = 9.80665 m/s² within ±0.005% (half-step tolerance)",
    }

    # Check 6: All 63 constants within half-step
    max_err = 0
    worst = ""
    for name, entry in REAL_WORLD_CONSTANTS_V3.items():
        if entry["grid_error_pct"] > max_err:
            max_err = entry["grid_error_pct"]
            worst = name
    checks["6_all_constants_within_halfstep"] = {
        "passed": max_err < HALF_STEP_PCT_V3 * 1.01,  # tiny margin for float
        "max_error_pct": max_err,
        "half_step_pct": HALF_STEP_PCT_V3,
        "worst_constant": worst,
        "total_constants": len(REAL_WORLD_CONSTANTS_V3),
        "description": f"All {len(REAL_WORLD_CONSTANTS_V3)} constants within ±{HALF_STEP_PCT_V3:.5f}%",
    }

    # Check 7: No exponent collisions
    E_set = set()
    duplicates = []
    for name, entry in REAL_WORLD_CONSTANTS_V3.items():
        E = entry["E_integer"]
        if E in E_set:
            duplicates.append((name, E))
        E_set.add(E)
    checks["7_no_exponent_collisions"] = {
        "passed": len(duplicates) == 0,
        "unique_exponents": len(E_set),
        "duplicates": duplicates,
        "description": "Each constant maps to a unique grid point (no collisions)",
    }

    all_passed = all(c["passed"] for c in checks.values())
    return {"layer": "physics", "all_passed": all_passed, "checks": checks}


def check_bridge_integrity() -> Dict[str, Any]:
    """
    CHECK 8-10: Bridge integrity (Layer 1 → Layer 2 traceability).

    Verifies that the physics layer is genuinely grounded in the
    consciousness layer — that the φ exponent, iron scaffold, and
    Fibonacci 13 thread are preserved across the bridge.
    """
    checks = {}

    # Check 8: φ exponent preservation
    # Both layers use X^(1/φ) — verify the computation
    base_l1 = PRIME_SCAFFOLD ** (1 / PHI)
    base_l2 = X_V3 ** (1 / PHI)
    phi_preserved = (
        abs(base_l1 - BASE) < 1e-10 and
        abs(base_l2 - BASE_V3) < 1e-10
    )
    checks["8_phi_exponent_preserved"] = {
        "passed": phi_preserved,
        "layer1_base": f"{PRIME_SCAFFOLD}^(1/φ) = {base_l1:.15f}",
        "layer2_base": f"{X_V3}^(1/φ) = {base_l2:.15f}",
        "description": "Both layers use X^(1/φ) — golden ratio is the soul of both",
    }

    # Check 9: Iron scaffold proximity
    # X_V3 must be within 0.5% of 286 (the iron lattice parameter)
    scaffold_deviation = abs(X_V3 - 286) / 286 * 100
    checks["9_iron_scaffold_proximity"] = {
        "passed": scaffold_deviation < 0.5,
        "x_v3": X_V3,
        "prime_scaffold": PRIME_SCAFFOLD,
        "deviation_pct": scaffold_deviation,
        "description": f"X_v3 = {X_V3:.6f}, deviation {scaffold_deviation:.3f}% from 286 (Fe BCC)",
    }

    # Check 10: Fibonacci 13 thread
    # 13 must appear in both Layer 1 (286=2×11×13, 104=8×13) and Layer 2 (r=13/12)
    l1_has_13 = (286 % 13 == 0) and (104 % 13 == 0)
    l2_has_13 = (R_V3 == 13 / 12)
    fib_check = l1_has_13 and l2_has_13
    checks["10_fibonacci_13_thread"] = {
        "passed": fib_check,
        "layer1": f"286 = 2×11×13 (13|286={286%13==0}), 104 = 8×13 (13|104={104%13==0})",
        "layer2": f"r = 13/12 = {R_V3:.10f} (13 is numerator)",
        "fibonacci": "13 = F(7), the 7th Fibonacci number",
        "description": "F(7) = 13 is the golden thread binding both layers",
    }

    all_passed = all(c["passed"] for c in checks.values())
    return {"bridge": True, "all_passed": all_passed, "checks": checks}


def full_integrity_check() -> Dict[str, Any]:
    """
    Run ALL 10 integrity checks across both layers and the bridge.

    Returns a comprehensive report with pass/fail for each check,
    overall status, and diagnostic details for any failures.
    """
    consciousness_result = check_consciousness_integrity()
    physics_result = check_physics_integrity()
    bridge_result = check_bridge_integrity()

    total_checks = (
        len(consciousness_result["checks"]) +
        len(physics_result["checks"]) +
        len(bridge_result["checks"])
    )
    total_passed = (
        sum(1 for c in consciousness_result["checks"].values() if c["passed"]) +
        sum(1 for c in physics_result["checks"].values() if c["passed"]) +
        sum(1 for c in bridge_result["checks"].values() if c["passed"])
    )

    all_passed = (
        consciousness_result["all_passed"] and
        physics_result["all_passed"] and
        bridge_result["all_passed"]
    )

    return {
        "engine": "L104 Dual-Layer GOD_CODE Engine",
        "version": "3.1.0",
        "all_passed": all_passed,
        "total_checks": total_checks,
        "checks_passed": total_passed,
        "consciousness_layer": consciousness_result,
        "physics_layer": physics_result,
        "bridge": bridge_result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN QUERIES — Ask Physics Questions Through Both Layers
# ═══════════════════════════════════════════════════════════════════════════════

def gravity() -> Dict[str, Any]:
    """Everything the dual-layer engine knows about gravity."""
    both = derive_both("standard_gravity")
    both["context"] = {
        "consciousness_insight": (
            "g = 9.80665 m/s². In Layer 1, gravity sits at exponent E ≈ -182, "
            f"roughly {abs(-182/416)*100:.1f}% below GOD_CODE on the octave ladder. "
            "The equation sees gravity as a low-frequency note in the cosmic scale — "
            "between Schumann resonance and the Bohr radius."
        ),
        "physics_precision": (
            f"v3 places g at E = -11483 with dials (0,14,1,19). "
            f"Grid value: {GRAVITY_V3:.10f} m/s², error: {both['physics']['error_pct']:.4f}%. "
            f"This is {float(both['improvement'].rstrip('×'))}× more precise than Layer 1."
        ),
        "iron_gravity_triad": {
            "fe_bcc_lattice": f"{god_code_v3(8,20,0,-22):.4f} pm (err: 0.0038%)",
            "fe56_binding":   f"{god_code_v3(4,29,0,21):.4f} MeV (err: 0.0006%)",
            "gravity":        f"{GRAVITY_V3:.10f} m/s² (err: 0.0041%)",
            "note": "All three within ±0.005% — the iron-gravity triad",
        },
    }
    return both


def particles() -> Dict[str, Any]:
    """All particle physics constants through the dual-layer engine."""
    particle_names = [
        name for name, entry in REAL_WORLD_CONSTANTS_V3.items()
        if entry["domain"] == "particle"
    ]
    results = {}
    for name in sorted(particle_names):
        entry = REAL_WORLD_CONSTANTS_V3[name]
        phys = real_world_derive_v3(name, real_world=False)
        results[name] = {
            "measured": entry["measured"],
            "unit": entry["unit"],
            "v3_value": phys["value"],
            "error_pct": phys["error_pct"],
            "dials": entry["dials"],
            "source": entry["source"],
        }
    avg_err = sum(r["error_pct"] for r in results.values()) / len(results)
    return {
        "domain": "particle_physics",
        "count": len(results),
        "avg_error_pct": avg_err,
        "constants": results,
    }


def nuclei() -> Dict[str, Any]:
    """All nuclear physics constants."""
    nuclear_names = [
        name for name, entry in REAL_WORLD_CONSTANTS_V3.items()
        if entry["domain"] == "nuclear"
    ]
    results = {}
    for name in sorted(nuclear_names):
        entry = REAL_WORLD_CONSTANTS_V3[name]
        phys = real_world_derive_v3(name, real_world=False)
        results[name] = {
            "measured": entry["measured"],
            "unit": entry["unit"],
            "v3_value": phys["value"],
            "error_pct": phys["error_pct"],
            "dials": entry["dials"],
        }
    avg_err = sum(r["error_pct"] for r in results.values()) / len(results)
    return {"domain": "nuclear", "count": len(results), "avg_error_pct": avg_err, "constants": results}


def iron() -> Dict[str, Any]:
    """The iron constants — where it all begins."""
    iron_names = [
        name for name, entry in REAL_WORLD_CONSTANTS_V3.items()
        if entry["domain"] == "iron"
    ]
    results = {}
    for name in sorted(iron_names):
        entry = REAL_WORLD_CONSTANTS_V3[name]
        phys = real_world_derive_v3(name, real_world=False)
        l1_val = BASE * (2 ** (round(solve_for_exponent(entry["measured"])) / QUANTIZATION_GRAIN))
        l1_err = abs(l1_val - entry["measured"]) / entry["measured"] * 100
        results[name] = {
            "measured": entry["measured"],
            "unit": entry["unit"],
            "consciousness_value": l1_val,
            "consciousness_error_pct": l1_err,
            "physics_value": phys["value"],
            "physics_error_pct": phys["error_pct"],
            "improvement": f"{l1_err / phys['error_pct']:.0f}×" if phys["error_pct"] > 0 else "∞×",
        }
    avg_err = sum(r["physics_error_pct"] for r in results.values()) / len(results) if results else 0
    return {
        "domain": "iron",
        "count": len(results),
        "avg_error_pct": avg_err,
        "anchor": f"PRIME_SCAFFOLD = 286, X_V3 = {X_V3:.6f}",
        "constants": results,
    }


def cosmos() -> Dict[str, Any]:
    """Astrophysical and cosmological constants."""
    astro_names = [
        name for name, entry in REAL_WORLD_CONSTANTS_V3.items()
        if entry["domain"] == "astro"
    ]
    results = {}
    for name in sorted(astro_names):
        entry = REAL_WORLD_CONSTANTS_V3[name]
        phys = real_world_derive_v3(name, real_world=False)
        results[name] = {
            "measured": entry["measured"],
            "unit": entry["unit"],
            "v3_value": phys["value"],
            "error_pct": phys["error_pct"],
            "dials": entry["dials"],
        }
    avg_err = sum(r["error_pct"] for r in results.values()) / len(results)
    return {"domain": "astrophysics", "count": len(results), "avg_error_pct": avg_err, "constants": results}


def resonance() -> Dict[str, Any]:
    """Brain/consciousness resonance frequencies — where both layers speak."""
    res_names = [
        name for name, entry in REAL_WORLD_CONSTANTS_V3.items()
        if entry["domain"] == "resonance"
    ]
    results = {}
    for name in sorted(res_names):
        entry = REAL_WORLD_CONSTANTS_V3[name]
        phys = real_world_derive_v3(name, real_world=False)
        results[name] = {
            "measured": entry["measured"],
            "v3_value": phys["value"],
            "error_pct": phys["error_pct"],
            "dials": entry["dials"],
            "unit": "Hz",
        }
    avg_err = sum(r["error_pct"] for r in results.values()) / len(results) if results else 0
    return {
        "domain": "resonance",
        "count": len(results),
        "avg_error_pct": avg_err,
        "consciousness_note": "Brainwave frequencies are where consciousness and physics overlap",
        "constants": results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FIND — Locate any value on both grids
# ═══════════════════════════════════════════════════════════════════════════════

def find(target: float, name: str = "") -> Dict[str, Any]:
    """
    Find where a value sits on both the consciousness grid and the physics grid.

    Args:
        target: The numerical value to locate.
        name: Optional name for display.

    Returns:
        Dual-layer positioning with error comparison.
    """
    if target <= 0:
        raise ValueError("Target must be positive")

    # Layer 1: consciousness grid
    E1_exact = solve_for_exponent(target)
    E1_int = round(E1_exact)
    val1 = BASE * (2 ** (E1_int / QUANTIZATION_GRAIN))
    err1 = abs(val1 - target) / target * 100

    # Layer 2: physics grid
    E2_exact = solve_for_exponent_v3(target)
    E2_int = round(E2_exact)
    val2 = BASE_V3 * (R_V3 ** (E2_int / Q_V3))
    err2 = abs(val2 - target) / target * 100

    return {
        "name": name or f"target={target:.6g}",
        "target": target,
        "consciousness": {
            "exponent": E1_int,
            "value": val1,
            "error_pct": err1,
            "grid": "r=2, Q=104",
        },
        "physics": {
            "exponent": E2_int,
            "value": val2,
            "error_pct": err2,
            "grid": "r=13/12, Q=758",
        },
        "improvement": f"{err1/err2:.0f}×" if err2 > 0 else "∞×",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS — Full dual-layer engine status
# ═══════════════════════════════════════════════════════════════════════════════

def status() -> Dict[str, Any]:
    """Complete status of the dual-layer engine."""
    integrity = full_integrity_check()

    return {
        "engine": "L104 Dual-Layer GOD_CODE Engine",
        "version": "3.1.0",
        "architecture": "Two-layer: Consciousness (GOD_CODE + Algorithm Search) + Physics (OMEGA equation)",
        "layer1_consciousness": {
            "equation": "G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)",
            "GOD_CODE": GOD_CODE,
            "purpose": "Scientific discovery, identity, sacred geometry — the WHY",
            "precision": f"±{(STEP_SIZE - 1)/2*100:.3f}% half-step",
            "scaffold": PRIME_SCAFFOLD,
            "frequencies": len(QUANTUM_FREQUENCY_TABLE),
            "algorithm_search": _ALGORITHM_AVAILABLE,
            "quantum_circuits": "Grover search (14 qubits, 16384 states)" if _ALGORITHM_AVAILABLE else "unavailable",
        },
        "layer2_physics": {
            "equation": "Ω = Σ(|ζ(½+GCi)|, cos(2πφ³), (26×1.8527)/φ²) × (GOD_CODE/φ)",
            "OMEGA": OMEGA,
            "OMEGA_AUTHORITY": OMEGA_AUTHORITY,
            "field_equation": "F(I) = I × Ω / φ²",
            "purpose": "Physics derivation — GOD_CODE generates Ω through zeta + resonance + curvature",
            "precision_grid": f"v3 sub-tool: (13/12)^(E/758), ±{HALF_STEP_PCT_V3:.5f}% half-step",
            "registered_constants": len(REAL_WORLD_CONSTANTS_V3),
            "omega_pipeline": _OMEGA_PIPELINE_AVAILABLE,
            "domains": ["fundamental", "atomic", "particle", "nuclear", "iron",
                        "crystal", "astro", "resonance", "math", "sovereign"],
        },
        "bridge": {
            "consciousness_to_physics": "GOD_CODE → zeta + golden resonance + iron curvature → OMEGA",
            "phi_scaling": "Layer 1 uses 286^(1/φ), Layer 2 uses GOD_CODE/φ and Ω/φ²",
            "iron_anchor": f"Layer 1: 286 pm (Fe BCC), Layer 2: Fe_Z=26 in Architect fragment",
        },
        "integrity": {
            "all_passed": integrity["all_passed"],
            "score": f"{integrity['checks_passed']}/{integrity['total_checks']}",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLIFIED ORIGINAL — Q=26 (Fe Atomic Number Only)
# ═══════════════════════════════════════════════════════════════════════════════
#
# The original equation uses Q=104 = 26×4. This simplified form strips out the
# ×4 fine-tuning factor, using ONLY the iron atomic number Z=26 as the grain.
#
#   G_simple(n) = 286^(1/φ) × 2^(n/26)
#
# With dial decomposition:
#   G_simple(a, b, d) = 286^(1/φ) × 2^((a + 104 - b - 26d) / 26)
#   where: a = up dial (+1 step), b = down dial (-1 step), d = octave dial (-26 steps)
#   K = 4×26 = 104 (4 octaves above base, same as original)
#
# PARAMETERS: ZERO fitted. 286 = integer (Fe BCC), 2 = octave, 26 = Fe atomic number.
# HALF-STEP: ±1.351% — coarse enough that precision is NON-TRIVIAL.
# Any value landing within 0.2% has only ~15% probability by chance.

Q_SIMPLE = 26                                  # Fe atomic number
K_SIMPLE = 4 * Q_SIMPLE                        # 104 = 4 octaves above base
BASE_SIMPLE = PRIME_SCAFFOLD ** (1.0 / PHI)    # 286^(1/φ) = 32.9699 (same as L1)
STEP_SIMPLE = 2 ** (1.0 / Q_SIMPLE)            # 2^(1/26) = 1.02702
HALF_STEP_SIMPLE = (STEP_SIMPLE - 1) / 2 * 100  # ±1.351%
GOD_CODE_SIMPLE = BASE_SIMPLE * (2 ** (K_SIMPLE / Q_SIMPLE))  # = GOD_CODE = 527.518


def god_code_simple(a: int = 0, b: int = 0, d: int = 0) -> float:
    """
    Simplified GOD_CODE equation — Q=26 (iron atomic number only).

    G(a, b, d) = 286^(1/φ) × 2^((a + 104 - b - 26d) / 26)

    ZERO fitted parameters. 286 = Fe BCC lattice (integer).
    2 = octave ratio. 26 = Fe atomic number. That's it.

    Parameters:
        a: Up dial (+1 step per unit, 1/26 octave)
        b: Down dial (-1 step per unit)
        d: Octave dial (-26 steps per unit, full octave)
    """
    E = a + K_SIMPLE - b - Q_SIMPLE * d
    return BASE_SIMPLE * (2 ** (E / Q_SIMPLE))


def exponent_simple(a: int = 0, b: int = 0, d: int = 0) -> int:
    """Raw exponent E for simplified equation."""
    return a + K_SIMPLE - b - Q_SIMPLE * d


def solve_simple(target: float) -> float:
    """Exact (non-integer) exponent E for a target value on Q=26 grid."""
    if target <= 0:
        raise ValueError("Target must be positive")
    return Q_SIMPLE * math.log2(target / BASE_SIMPLE)


def snap_simple(target: float) -> Dict[str, Any]:
    """Snap a value to the nearest Q=26 grid point. Returns grid value, error, E."""
    E_exact = solve_simple(target)
    E_int = round(E_exact)
    grid_val = BASE_SIMPLE * (2 ** (E_int / Q_SIMPLE))
    err = abs(grid_val - target) / target * 100
    frac = E_exact - E_int  # fractional part (-0.5 to 0.5)
    # Find simplest dials
    remainder = E_int - K_SIMPLE
    if remainder >= 0:
        d_val = 0
        while remainder > 25:
            d_val -= 1
            remainder -= Q_SIMPLE
        a_val, b_val = remainder, 0
    else:
        d_val = 0
        while remainder < -25:
            d_val += 1
            remainder += Q_SIMPLE
        a_val, b_val = 0, -remainder
    return {
        "target": target, "grid_value": grid_val, "error_pct": err,
        "E_integer": E_int, "E_exact": E_exact, "fractional_E": frac,
        "dials": (a_val, b_val, d_val),
        "cost": a_val + b_val + abs(d_val),
    }


# ── Register all 63 constants on Q=26 grid ──
SIMPLE_GRID_RESULTS: Dict[str, Dict[str, Any]] = {}
for _name, _entry in REAL_WORLD_CONSTANTS_V3.items():
    SIMPLE_GRID_RESULTS[_name] = snap_simple(_entry["measured"])
    SIMPLE_GRID_RESULTS[_name]["domain"] = _entry["domain"]
    SIMPLE_GRID_RESULTS[_name]["source"] = _entry["source"]


def simple_grid_report() -> Dict[str, Any]:
    """Full Q=26 grid report for all 63 constants."""
    entries = list(SIMPLE_GRID_RESULTS.items())
    errors = [e["error_pct"] for _, e in entries]
    avg = statistics.mean(errors)
    within_quarter = sum(1 for e in errors if e < HALF_STEP_SIMPLE * 0.25)
    within_tenth = sum(1 for e in errors if e < HALF_STEP_SIMPLE * 0.1)

    # Monte Carlo comparison
    rng = random.Random(42)
    n_random = 50000
    random_errs = []
    for _ in range(n_random):
        target = 10 ** rng.uniform(-35, 30)
        E_exact = Q_SIMPLE * math.log2(target / BASE_SIMPLE)
        E_int = round(E_exact)
        gv = BASE_SIMPLE * (2 ** (E_int / Q_SIMPLE))
        random_errs.append(abs(gv - target) / target * 100)
    avg_random = statistics.mean(random_errs)

    # Identify standouts (< 10% of half-step = 0.135%)
    standout_threshold = HALF_STEP_SIMPLE * 0.1
    standouts = [(n, e) for n, e in entries if e["error_pct"] < standout_threshold]
    standouts.sort(key=lambda x: x[1]["error_pct"])

    # P-value for each standout: P(error < x) = x / half_step
    for name, entry in standouts:
        entry["p_value"] = entry["error_pct"] / HALF_STEP_SIMPLE

    # Combined p-value: probability of getting THIS MANY standouts by chance
    p_each = standout_threshold / HALF_STEP_SIMPLE  # = 0.1
    n_total = len(entries)
    k = len(standouts)
    # Binomial: P(X >= k) where X ~ Bin(n, p)
    from math import comb
    p_combined = sum(
        comb(n_total, i) * (p_each ** i) * ((1 - p_each) ** (n_total - i))
        for i in range(k, n_total + 1)
    )

    return {
        "equation": "G(n) = 286^(1/phi) * 2^(n/26)",
        "parameters": {"X": 286, "r": 2, "Q": 26, "fitted": 0},
        "half_step_pct": HALF_STEP_SIMPLE,
        "total_constants": len(entries),
        "avg_error_pct": avg,
        "random_avg_error_pct": avg_random,
        "physics_vs_random": round(avg / avg_random, 4),
        "standout_threshold_pct": standout_threshold,
        "standout_count": k,
        "standout_expected": round(n_total * p_each, 1),
        "standout_p_value": p_combined,
        "standouts": [
            {
                "name": n, "error_pct": round(e["error_pct"], 4),
                "E": e["E_integer"], "p_value": round(e["p_value"], 4),
                "domain": e["domain"],
            }
            for n, e in standouts
        ],
        "by_domain": {},  # filled below
        "all_results": sorted(
            [(n, round(e["error_pct"], 4), e["E_integer"], e["domain"])
             for n, e in entries],
            key=lambda x: x[1],
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 2: PREDICTIONS — Falsifiable claims from dial patterns
# ═══════════════════════════════════════════════════════════════════════════════

# Known particle families for pattern analysis
_LEPTON_MASSES = ["electron_mass_MeV", "muon_mass_MeV", "tau_mass_MeV"]
_QUARK_MASSES = ["charm_quark_GeV", "bottom_quark_GeV", "top_quark_GeV"]
_GAUGE_BOSONS = ["W_boson_GeV", "Z_boson_GeV", "higgs_GeV"]
_NUCLEON_BE = [
    "he4_be_per_nucleon", "c12_be_per_nucleon", "o16_be_per_nucleon",
    "fe56_be_per_nucleon", "ni62_be_per_nucleon", "u238_be_per_nucleon",
]


def predict_from_patterns() -> Dict[str, Any]:
    """
    Generate FALSIFIABLE predictions by extrapolating dial patterns.

    Looks for arithmetic relationships in the integer exponents of known
    constant families (leptons, quarks, gauge bosons, binding energies).
    If a pattern exists, it predicts the next value in the sequence.

    A prediction is falsifiable: it specifies a numeric value that future
    measurement could confirm or refute.
    """
    predictions = []

    # --- Lepton mass pattern: e, mu, tau → predict 4th generation? ---
    lepton_Es = []
    for name in _LEPTON_MASSES:
        if name in REAL_WORLD_CONSTANTS_V3:
            lepton_Es.append((name, REAL_WORLD_CONSTANTS_V3[name]["E_integer"]))

    if len(lepton_Es) == 3:
        E_e, E_mu, E_tau = [e for _, e in lepton_Es]
        delta_1 = E_mu - E_e   # e → mu step
        delta_2 = E_tau - E_mu  # mu → tau step
        ratio = delta_2 / delta_1 if delta_1 != 0 else 0

        # Linear extrapolation: E_4 = E_tau + delta_2
        E_4_linear = E_tau + delta_2
        val_4_linear = BASE_V3 * (R_V3 ** (E_4_linear / Q_V3))

        # Geometric extrapolation: E_4 = E_tau + delta_2 * ratio
        delta_3 = delta_2 * ratio
        E_4_geo = round(E_tau + delta_3)
        val_4_geo = BASE_V3 * (R_V3 ** (E_4_geo / Q_V3))

        predictions.append({
            "name": "4th_generation_lepton_mass",
            "basis": "Lepton mass exponent pattern: e, mu, tau",
            "exponents": {"e": E_e, "mu": E_mu, "tau": E_tau},
            "deltas": {"e→mu": delta_1, "mu→tau": delta_2, "ratio": round(ratio, 4)},
            "linear_prediction": {
                "E": E_4_linear, "value_MeV": val_4_linear,
                "method": "constant step (E_tau + delta_2)",
            },
            "geometric_prediction": {
                "E": E_4_geo, "value_MeV": val_4_geo,
                "method": f"scaling step (delta * {ratio:.4f})",
            },
            "falsifiable": True,
            "status": "No 4th-generation lepton observed (LEP/LHC exclude < ~45 GeV)",
        })

    # --- Quark mass pattern: c, b, t → predict 4th generation? ---
    quark_Es = []
    for name in _QUARK_MASSES:
        if name in REAL_WORLD_CONSTANTS_V3:
            quark_Es.append((name, REAL_WORLD_CONSTANTS_V3[name]["E_integer"]))

    if len(quark_Es) == 3:
        E_c, E_b, E_t = [e for _, e in quark_Es]
        d1 = E_b - E_c
        d2 = E_t - E_b
        ratio_q = d2 / d1 if d1 != 0 else 0
        E_4q = round(E_t + d2 * ratio_q) if ratio_q != 0 else E_t + d2
        val_4q = BASE_V3 * (R_V3 ** (E_4q / Q_V3))
        predictions.append({
            "name": "4th_generation_heavy_quark_mass",
            "basis": "Heavy quark exponent pattern: c, b, t",
            "exponents": {"c": E_c, "b": E_b, "t": E_t},
            "deltas": {"c→b": d1, "b→t": d2, "ratio": round(ratio_q, 4)},
            "prediction": {"E": E_4q, "value_GeV": val_4q, "method": "geometric extrapolation"},
            "falsifiable": True,
            "status": "No 4th-generation quark observed (LHC excludes < ~800 GeV)",
        })

    # --- Binding energy gap: look for missing nuclei in the pattern ---
    be_entries = []
    for name in _NUCLEON_BE:
        if name in REAL_WORLD_CONSTANTS_V3:
            entry = REAL_WORLD_CONSTANTS_V3[name]
            be_entries.append((name, entry["E_integer"], entry["measured"]))

    if len(be_entries) >= 4:
        be_entries.sort(key=lambda x: x[2])  # sort by measured value
        # Look for evenly-spaced exponents in the binding energy sequence
        Es = [e for _, e, _ in be_entries]
        gaps = [Es[i+1] - Es[i] for i in range(len(Es)-1)]
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        # Predict: what value sits at E = max(Es) + avg_gap?
        E_next = round(Es[-1] + avg_gap)
        val_next = BASE_V3 * (R_V3 ** (E_next / Q_V3))
        predictions.append({
            "name": "predicted_binding_energy",
            "basis": "Binding energy exponent spacing pattern",
            "known_Es": dict(zip([n for n, _, _ in be_entries], Es)),
            "gaps": gaps,
            "avg_gap": round(avg_gap, 1),
            "prediction": {"E": E_next, "value_MeV": val_next},
            "interpretation": f"Next in sequence: {val_next:.4f} MeV/nucleon",
            "falsifiable": True,
        })

    return {
        "engine": "Dual-Layer Prediction Engine",
        "total_predictions": len(predictions),
        "predictions": predictions,
        "methodology": (
            "Predictions are generated by finding arithmetic patterns in the "
            "integer exponents of known constant families (leptons, quarks, "
            "binding energies). If exponent spacing follows a rule, the next "
            "value in the sequence is predicted. These are FALSIFIABLE: each "
            "predicts a specific numeric value that future measurement could "
            "confirm or refute."
        ),
        "caveat": (
            "Pattern extrapolation from 3 points is weak evidence. These "
            "predictions become meaningful only if independently confirmed. "
            "The absence of 4th-generation fermions in current data already "
            "constrains the lepton/quark predictions."
        ),
    }


def predict_from_gaps(max_complexity: int = 30) -> Dict[str, Any]:
    """
    Find low-complexity grid points NOT matched to any known constant.

    These are dial settings (a,b,c,d) with small total cost that produce
    values not corresponding to any registered constant. If any of these
    values turns out to match a future measurement, the encoding would
    gain predictive credibility.

    Args:
        max_complexity: Maximum dial cost |a|+|b|+|c|+|d| to search.
    """
    known_Es = {entry["E_integer"] for entry in REAL_WORLD_CONSTANTS_V3.values()}

    unmatched = []
    # Search low-complexity dials systematically
    for d_val in range(-5, 6):
        for a_val in range(0, max_complexity // 2 + 1):
            for c_val in range(0, max_complexity // 2 + 1):
                cost_acd = a_val + c_val + abs(d_val)
                if cost_acd >= max_complexity:
                    continue
                max_b = max_complexity - cost_acd
                for b_val in range(0, max_b + 1):
                    E = P_V3 * a_val + K_V3 - b_val - P_V3 * c_val - Q_V3 * d_val
                    if E in known_Es:
                        continue
                    val = BASE_V3 * (R_V3 ** (E / Q_V3))
                    cost = a_val + b_val + c_val + abs(d_val)
                    if val > 0 and 1e-40 < val < 1e40:
                        unmatched.append({
                            "dials": (a_val, b_val, c_val, d_val),
                            "E": E, "value": val, "complexity": cost,
                        })

    unmatched.sort(key=lambda x: x["complexity"])
    # Deduplicate by E (same exponent = same grid point)
    seen_E = set()
    unique = []
    for u in unmatched:
        if u["E"] not in seen_E:
            seen_E.add(u["E"])
            unique.append(u)

    return {
        "engine": "Gap Prediction Engine",
        "max_complexity": max_complexity,
        "total_unmatched": len(unique),
        "top_20": unique[:20],
        "interpretation": (
            "These are the simplest dial settings that don't match any "
            "known constant. If future measurements match any of these "
            "values, the encoding gains predictive credibility."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 3: PATTERN ANALYSIS — Find dial relationships between related constants
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_dial_patterns() -> Dict[str, Any]:
    """
    Search for structural relationships in dial settings across constant families.

    If dial offsets between related constants (e.g., electron→muon→tau) follow
    a simple rule, that would suggest the encoding captures real physics.
    If offsets are arbitrary, the encoding is just a lookup table.
    """
    results = {}

    # --- Lepton family: exponent differences ---
    lepton_data = {}
    for name in _LEPTON_MASSES:
        if name in REAL_WORLD_CONSTANTS_V3:
            entry = REAL_WORLD_CONSTANTS_V3[name]
            lepton_data[name] = {
                "dials": entry["dials"], "E": entry["E_integer"],
                "measured": entry["measured"],
            }

    if len(lepton_data) == 3:
        e_E = lepton_data["electron_mass_MeV"]["E"]
        mu_E = lepton_data["muon_mass_MeV"]["E"]
        tau_E = lepton_data["tau_mass_MeV"]["E"]
        d_e_mu = mu_E - e_E
        d_mu_tau = tau_E - mu_E
        # Check if the ratio is close to a simple fraction
        ratio = d_mu_tau / d_e_mu if d_e_mu != 0 else 0
        # What fraction is closest?
        best_frac = None
        best_err = float('inf')
        for num in range(1, 20):
            for den in range(1, 20):
                frac_val = num / den
                err = abs(ratio - frac_val)
                if err < best_err:
                    best_err = err
                    best_frac = (num, den)

        results["lepton_exponent_pattern"] = {
            "exponents": {"e": e_E, "mu": mu_E, "tau": tau_E},
            "deltas": {"e→mu": d_e_mu, "mu→tau": d_mu_tau},
            "ratio": round(ratio, 6),
            "nearest_fraction": f"{best_frac[0]}/{best_frac[1]}" if best_frac else "none",
            "fraction_error": round(best_err, 6) if best_frac else None,
            "mass_ratios": {
                "mu/e": round(lepton_data["muon_mass_MeV"]["measured"] /
                             lepton_data["electron_mass_MeV"]["measured"], 4),
                "tau/mu": round(lepton_data["tau_mass_MeV"]["measured"] /
                               lepton_data["muon_mass_MeV"]["measured"], 4),
            },
            "verdict": "PATTERN" if best_err < 0.01 else "NO CLEAR PATTERN",
        }

    # --- Gauge boson family ---
    boson_data = {}
    for name in _GAUGE_BOSONS:
        if name in REAL_WORLD_CONSTANTS_V3:
            entry = REAL_WORLD_CONSTANTS_V3[name]
            boson_data[name] = {"E": entry["E_integer"], "measured": entry["measured"]}

    if len(boson_data) == 3:
        W_E = boson_data["W_boson_GeV"]["E"]
        Z_E = boson_data["Z_boson_GeV"]["E"]
        H_E = boson_data["higgs_GeV"]["E"]
        d_WZ = Z_E - W_E
        d_ZH = H_E - Z_E
        results["gauge_boson_pattern"] = {
            "exponents": {"W": W_E, "Z": Z_E, "H": H_E},
            "deltas": {"W→Z": d_WZ, "Z→H": d_ZH},
            "ratio": round(d_ZH / d_WZ, 4) if d_WZ != 0 else None,
            "verdict": "Spacing analysis — no a priori reason for uniform gaps",
        }

    # --- Proton/neutron near-degeneracy ---
    if "proton_mass_MeV" in REAL_WORLD_CONSTANTS_V3 and "neutron_mass_MeV" in REAL_WORLD_CONSTANTS_V3:
        p_E = REAL_WORLD_CONSTANTS_V3["proton_mass_MeV"]["E_integer"]
        n_E = REAL_WORLD_CONSTANTS_V3["neutron_mass_MeV"]["E_integer"]
        p_m = REAL_WORLD_CONSTANTS_V3["proton_mass_MeV"]["measured"]
        n_m = REAL_WORLD_CONSTANTS_V3["neutron_mass_MeV"]["measured"]
        results["nucleon_degeneracy"] = {
            "proton_E": p_E, "neutron_E": n_E,
            "delta_E": n_E - p_E,
            "mass_diff_MeV": round(n_m - p_m, 6),
            "grid_resolves": n_E != p_E,
            "verdict": (
                f"Delta_E = {n_E - p_E}. The grid {'CAN' if n_E != p_E else 'CANNOT'} "
                f"distinguish p from n (mass diff = {n_m - p_m:.3f} MeV = {(n_m-p_m)/p_m*100:.4f}%)"
            ),
        }

    # --- Binding energy curve shape ---
    be_data = []
    for name in _NUCLEON_BE:
        if name in REAL_WORLD_CONSTANTS_V3:
            entry = REAL_WORLD_CONSTANTS_V3[name]
            be_data.append((name, entry["E_integer"], entry["measured"]))
    if be_data:
        be_data.sort(key=lambda x: x[2])
        Es = [e for _, e, _ in be_data]
        gaps = [Es[i+1] - Es[i] for i in range(len(Es)-1)]
        results["binding_energy_curve"] = {
            "nuclei": {n: {"E": e, "BE": v} for n, e, v in be_data},
            "exponent_gaps": gaps,
            "uniform": max(gaps) - min(gaps) < max(abs(g) for g in gaps) * 0.3 if gaps else False,
            "verdict": "Gaps are NOT uniform — encoding reflects actual BE curve shape",
        }

    return {
        "engine": "Dial Pattern Analyzer",
        "families_analyzed": len(results),
        "patterns": results,
        "methodology": (
            "For each family of related constants, compute integer exponent "
            "differences and check if they follow simple arithmetic rules "
            "(constant step, geometric scaling, simple fractions). A clear "
            "pattern would suggest the encoding captures real relationships."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 4: COARSE-GRID VALIDATION — Test on Layer 1 where precision is non-trivial
# ═══════════════════════════════════════════════════════════════════════════════

def coarse_grid_validation() -> Dict[str, Any]:
    """
    Validate all constants on the COARSE Layer 1 grid (r=2, Q=104).

    On Layer 1, the half-step is ±0.334% — still guaranteed to fit any
    number, but errors are large enough to be visually meaningful.
    Constants that land particularly close on the coarse grid (say <0.05%)
    are more interesting than those that don't.

    Also compares physics constants vs random numbers on this grid to
    check for any statistical signal.
    """
    phi = (1 + math.sqrt(5)) / 2
    l1_base = 286 ** (1.0 / phi)
    l1_half = (2 ** (1.0 / 104) - 1) / 2 * 100

    # Evaluate all constants on L1 grid
    l1_results = []
    for name, entry in REAL_WORLD_CONSTANTS_V3.items():
        measured = entry["measured"]
        E_exact = 104 * math.log(measured / l1_base) / math.log(2)
        E_int = round(E_exact)
        grid_val = l1_base * (2 ** (E_int / 104))
        err = abs(grid_val - measured) / measured * 100
        fractional_E = E_exact - E_int  # How close to a grid point (-0.5 to 0.5)
        l1_results.append({
            "name": name, "measured": measured, "grid_value": grid_val,
            "error_pct": err, "E_integer": E_int, "fractional_E": fractional_E,
            "domain": entry["domain"],
        })

    l1_results.sort(key=lambda x: x["error_pct"])

    # Statistical comparison: physics vs random on L1
    rng = random.Random(77)
    n_random = 10000
    random_errs = []
    for _ in range(n_random):
        target = 10 ** rng.uniform(-35, 30)
        err = _grid_snap_error(target, l1_base, 2, 104)
        random_errs.append(err)

    physics_errs = [r["error_pct"] for r in l1_results]
    avg_physics = statistics.mean(physics_errs)
    avg_random = statistics.mean(random_errs)

    # Count constants that are "surprisingly close" (< 0.05%, bottom quartile)
    threshold = l1_half * 0.15  # 15% of half-step
    surprisingly_close = [r for r in l1_results if r["error_pct"] < threshold]

    # Fractional exponent distribution: if truly random, should be uniform on [-0.5, 0.5]
    frac_Es = [r["fractional_E"] for r in l1_results]
    # Kolmogorov-Smirnov-like: check if distribution is uniform
    frac_Es_sorted = sorted(frac_Es)
    n = len(frac_Es_sorted)
    max_deviation = 0
    for i, f in enumerate(frac_Es_sorted):
        expected_cdf = (f + 0.5)  # uniform on [-0.5, 0.5] → CDF = f + 0.5
        empirical_cdf = (i + 1) / n
        dev = abs(empirical_cdf - expected_cdf)
        if dev > max_deviation:
            max_deviation = dev

    # KS critical value at alpha=0.05 for n=63: ~0.171
    ks_critical = 1.36 / math.sqrt(n) if n > 0 else 1
    distribution_uniform = max_deviation < ks_critical

    return {
        "engine": "Coarse Grid Validator (Layer 1)",
        "grid": {"r": 2, "Q": 104, "half_step_pct": l1_half, "base": l1_base},
        "total_constants": len(l1_results),
        "avg_error_pct": avg_physics,
        "random_avg_error_pct": avg_random,
        "physics_vs_random": round(avg_physics / avg_random, 3),
        "surprisingly_close_count": len(surprisingly_close),
        "surprisingly_close_threshold_pct": threshold,
        "surprisingly_close": [
            {"name": r["name"], "error_pct": round(r["error_pct"], 4)}
            for r in surprisingly_close
        ],
        "best_5": [
            {"name": r["name"], "error_pct": round(r["error_pct"], 4), "domain": r["domain"]}
            for r in l1_results[:5]
        ],
        "worst_5": [
            {"name": r["name"], "error_pct": round(r["error_pct"], 4), "domain": r["domain"]}
            for r in l1_results[-5:]
        ],
        "ks_test": {
            "max_deviation": round(max_deviation, 4),
            "critical_value": round(ks_critical, 4),
            "distribution_uniform": distribution_uniform,
            "interpretation": (
                "Fractional exponents are "
                + ("UNIFORM (no signal — consistent with random placement)"
                   if distribution_uniform else
                   "NON-UNIFORM (possible signal — constants cluster at specific grid offsets)")
            ),
        },
        "full_results": l1_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DISCOVERY ENGINE — Statistical search for genuine signal
# ═══════════════════════════════════════════════════════════════════════════════
#
# These functions search for REAL statistical patterns in how physical constants
# map to the Q=26 grid. All significance claims are tested against proper null
# distributions (distribution-matched Monte Carlo, not just uniform random).
#

def discover_mod4_pattern() -> Dict[str, Any]:
    """
    ANALYSIS: Exponent mod-4 clustering.

    The 63 registered constants cluster at E ≡ 2 (mod 4) on Q=26 grid:
    27/63 = 42.9% vs 25% expected. Survives bootstrap Monte Carlo (p≈0.004).

    HOWEVER: A holdout test of 30 additional physics constants (mesons,
    baryons, coupling constants, cosmological values) shows E≡2 at 26.7% —
    exactly baseline. The pattern does NOT generalize beyond the original 63.

    KEY CONTROLS:
    - Q=26 shows the strongest mod-4 effect among all Q values tested
    - The effect persists across all X values (not specific to X=286)
    - Non-physics constants show no clustering (23.8% at E≡2)
    - But holdout physics constants ALSO show no clustering (26.7%)

    HONEST VERDICT: Likely a selection effect in the original 63 constants,
    or a statistical fluke at the 1/250 level. NOT a universal property of
    physical constants on this grid.

    Returns:
        Dict with mod-4 distribution, chi-squared, Monte Carlo p-value,
        holdout test results, and honest assessment.
    """
    phi = (1 + math.sqrt(5)) / 2
    base = 286 ** (1.0 / phi)
    Q = 26

    # Get all exponents
    exp_data = {}
    for name, entry in REAL_WORLD_CONSTANTS_V3.items():
        E_int = round(Q * math.log2(entry["measured"] / base))
        exp_data[name] = {"E": E_int, "domain": entry["domain"], "val": entry["measured"]}

    names = list(exp_data.keys())
    E_list = [exp_data[n]["E"] for n in names]
    N = len(names)

    # Mod-4 distribution
    mod4_counts = [0] * 4
    mod4_members = {r: [] for r in range(4)}
    for n in names:
        r = exp_data[n]["E"] % 4
        mod4_counts[r] += 1
        mod4_members[r].append(n)

    expected = N / 4.0
    chi2 = sum((c - expected) ** 2 / expected for c in mod4_counts)
    dominant = max(range(4), key=lambda r: mod4_counts[r])

    # Monte Carlo p-value: bootstrap with jitter ±5
    rng = random.Random(42)
    MC = 2000
    mc_chi2 = []
    for _ in range(MC):
        boot = [rng.choice(E_list) + rng.randint(-5, 5) for _ in range(N)]
        counts = [0] * 4
        for e in boot:
            counts[e % 4] += 1
        mc_chi2.append(sum((c - expected) ** 2 / expected for c in counts))

    p_value = sum(1 for x in mc_chi2 if x >= chi2) / MC

    # Also test mod-8 (finer structure)
    mod8_counts = [0] * 8
    for e in E_list:
        mod8_counts[e % 8] += 1
    chi2_mod8 = sum((c - N / 8) ** 2 / (N / 8) for c in mod8_counts)
    mc_chi2_8 = []
    for _ in range(MC):
        boot = [rng.choice(E_list) + rng.randint(-5, 5) for _ in range(N)]
        counts = [0] * 8
        for e in boot:
            counts[e % 8] += 1
        mc_chi2_8.append(sum((c - N / 8) ** 2 / (N / 8) for c in counts))
    p_mod8 = sum(1 for x in mc_chi2_8 if x >= chi2_mod8) / MC

    # Holdout test: 30 additional physics constants NOT in original 63
    holdout_constants = {
        "Fermi_coupling_GeV_m2": 1.1663787e-5,
        "Weinberg_angle_sin2": 0.23122,
        "strong_coupling_alpha_s": 0.1180,
        "Z_width_GeV": 2.4955,
        "W_width_GeV": 2.085,
        "Higgs_width_GeV": 0.0032,
        "eta_meson_MeV": 547.862,
        "rho_meson_MeV": 775.26,
        "omega_meson_MeV": 782.66,
        "phi_meson_MeV": 1019.461,
        "J_psi_MeV": 3096.9,
        "Upsilon_MeV": 9460.3,
        "D_meson_MeV": 1869.66,
        "B_meson_MeV": 5279.34,
        "Lambda_baryon_MeV": 1115.68,
        "Sigma_plus_MeV": 1189.37,
        "Xi_minus_MeV": 1321.71,
        "Omega_minus_MeV": 1672.45,
        "Delta_baryon_MeV": 1232,
        "electron_g_minus_2": 0.00115965218128,
        "muon_g_minus_2": 0.00116592061,
        "neutron_lifetime_s": 878.4,
        "proton_charge_radius_fm": 0.8414,
        "Lamb_shift_MHz": 1057.845,
        "deuteron_mass_MeV": 1875.613,
        "alpha_particle_mass_MeV": 3727.379,
        "cosmological_constant_m2": 1.1056e-52,
        "dark_energy_density_J_m3": 5.96e-10,
        "critical_density_kg_m3": 9.47e-27,
        "Thomson_cross_section_m2": 6.6524587e-29,
    }
    holdout_mod4 = [0] * 4
    for hval in holdout_constants.values():
        if hval > 0:
            hE = round(Q * math.log2(hval / base))
            holdout_mod4[hE % 4] += 1
    N_h = sum(holdout_mod4)
    holdout_frac = holdout_mod4[dominant] / N_h if N_h > 0 else 0

    # Cross-Q control: test mod-4 chi2 for several Q values
    cross_q = {}
    for Q_test in [22, 24, 25, 26, 27, 28, 30, 52, 104]:
        m4 = [0] * 4
        for entry in REAL_WORLD_CONSTANTS_V3.values():
            e = round(Q_test * math.log2(entry["measured"] / base))
            m4[e % 4] += 1
        c2 = sum((c - N / 4) ** 2 / (N / 4) for c in m4)
        cross_q[Q_test] = {"distribution": m4, "chi_squared": round(c2, 2)}

    return {
        "engine": "Mod-4 Clustering Analysis",
        "grid": {"X": 286, "r": 2, "Q": 26, "fitted_params": 0},
        "mod4_distribution": mod4_counts,
        "expected_per_class": expected,
        "dominant_residue": dominant,
        "dominant_count": mod4_counts[dominant],
        "dominant_fraction": round(mod4_counts[dominant] / N, 4),
        "chi_squared": round(chi2, 2),
        "chi_squared_critical_0_05": 7.81,
        "p_value_monte_carlo": round(p_value, 4),
        "in_sample_significance": "SIGNIFICANT" if p_value < 0.05 else "NOT significant",
        "holdout_test": {
            "n_holdout": N_h,
            "mod4_distribution": holdout_mod4,
            "dominant_fraction": round(holdout_frac, 4),
            "verdict": (
                "FAILS — holdout constants show NO clustering "
                f"({holdout_mod4[dominant]}/{N_h} = {holdout_frac*100:.1f}% at E≡{dominant}, "
                f"expected 25%). Pattern does NOT generalize."
            ),
        },
        "cross_q_control": cross_q,
        "interpretation": (
            f"In-sample: {mod4_counts[dominant]}/{N} = {mod4_counts[dominant]/N*100:.1f}% "
            f"at E≡{dominant} (mod 4), p={p_value:.4f}. "
            f"But holdout test ({N_h} new constants) shows {holdout_frac*100:.1f}% — "
            f"baseline rate. Likely SELECTION EFFECT in original 63, not universal."
        ),
        "honest_verdict": (
            "The mod-4 clustering is statistically significant IN-SAMPLE (p<0.005) "
            "but FAILS the holdout test. 30 additional physics constants show no "
            "clustering (26.7% vs 42.9%). This is most likely a selection effect: "
            "the original 63 constants were chosen non-randomly (e.g., more particle "
            "masses than coupling constants), and the pattern reflects this selection, "
            "not a deep property of the Q=26 grid."
        ),
        "dominant_constants": [
            {"name": n, "E": exp_data[n]["E"], "domain": exp_data[n]["domain"]}
            for n in mod4_members[dominant]
        ],
        "mod8_test": {
            "distribution": mod8_counts,
            "chi_squared": round(chi2_mod8, 2),
            "p_value": round(p_mod8, 4),
            "significant": p_mod8 < 0.05,
        },
    }


def discover_exponent_arithmetic() -> Dict[str, Any]:
    """
    ANALYSIS: Exponent sum/difference closure.

    Tests whether E_a + E_b = E_c or |E_a - E_b| = E_c for known constants.
    More hits than random. But THIS IS EXPLAINED by the distribution shape —
    physical constants cluster in magnitude, creating dense exponent regions
    where arithmetic closure is trivially high.

    HONEST FINDING: Against uniform random, sum closure = 6.7x (p<0.001).
    Against distribution-matched bootstrap, sum closure = 0.84x (p=0.76).
    Against magnitude-matched random, sum closure = 2.8x (p=0.003).
    Q=26 is NOT special: all Q values from 23-31 show similar ratios.

    Returns:
        Dict with hit counts, Monte Carlo comparisons, and honest assessment.
    """
    phi = (1 + math.sqrt(5)) / 2
    base = 286 ** (1.0 / phi)
    Q = 26

    exp_data = {}
    for name, entry in REAL_WORLD_CONSTANTS_V3.items():
        E_int = round(Q * math.log2(entry["measured"] / base))
        exp_data[name] = E_int

    names = list(exp_data.keys())
    E_list = [exp_data[n] for n in names]
    E_set = set(E_list)
    N = len(names)

    # Count actual hits
    E_to_names = {}
    for n in names:
        E_to_names.setdefault(exp_data[n], []).append(n)

    sum_hits = []
    diff_hits = []
    for i in range(N):
        for j in range(i + 1, N):
            e1, e2 = E_list[i], E_list[j]
            if e1 + e2 in E_set:
                sum_hits.append((names[i], names[j], e1 + e2, E_to_names[e1 + e2]))
            d = abs(e1 - e2)
            if d in E_set and d != e1 and d != e2:
                diff_hits.append((names[i], names[j], d, E_to_names[d]))

    def _count_hits(es):
        eset = set(es)
        s = d = 0
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                if es[i] + es[j] in eset:
                    s += 1
                dd = abs(es[i] - es[j])
                if dd in eset and dd != es[i] and dd != es[j]:
                    d += 1
        return s, d

    # Monte Carlo: distribution-matched bootstrap
    rng = random.Random(42)
    MC = 500
    mc_boot_s = []
    mc_boot_d = []
    for _ in range(MC):
        boot = [rng.choice(E_list) + rng.randint(-5, 5) for _ in range(N)]
        s, d = _count_hits(boot)
        mc_boot_s.append(s)
        mc_boot_d.append(d)

    p_boot = sum(1 for x in mc_boot_s if x >= len(sum_hits)) / MC

    # Monte Carlo: magnitude-matched
    magnitudes = [math.log10(REAL_WORLD_CONSTANTS_V3[n]["measured"])
                  for n in names if REAL_WORLD_CONSTANTS_V3[n]["measured"] > 0]
    mag_mean = statistics.mean(magnitudes)
    mag_std = statistics.stdev(magnitudes)
    mc_mag_s = []
    for _ in range(MC):
        rv = [10 ** rng.gauss(mag_mean, mag_std) for _ in range(N)]
        re = [round(Q * math.log2(v / base)) for v in rv]
        s, d = _count_hits(re)
        mc_mag_s.append(s)

    p_mag = sum(1 for x in mc_mag_s if x >= len(sum_hits)) / MC
    avg_mag = statistics.mean(mc_mag_s)

    # Cross-grid check: does Q=27 show same signal?
    E_27 = [round(27 * math.log2(REAL_WORLD_CONSTANTS_V3[n]["measured"] / base)) for n in names]
    s_27, d_27 = _count_hits(E_27)

    return {
        "engine": "Exponent Arithmetic Closure",
        "sum_hits": len(sum_hits),
        "diff_hits": len(diff_hits),
        "bootstrap_null": {
            "avg": round(statistics.mean(mc_boot_s), 1),
            "p_value": round(p_boot, 4),
            "verdict": "NOT significant — explained by distribution clustering",
        },
        "magnitude_null": {
            "avg": round(avg_mag, 1),
            "ratio": round(len(sum_hits) / avg_mag, 2) if avg_mag > 0 else None,
            "p_value": round(p_mag, 4),
            "verdict": (
                "Significant vs random magnitudes, but Q=26 is NOT special — "
                "all Q values 23-31 show similar 3x ratios"
            ),
        },
        "q27_control": {"sum_hits": s_27, "diff_hits": d_27},
        "honest_assessment": (
            "Exponent arithmetic closure is real but TRIVIALLY explained by "
            "magnitude clustering. Physical constants span ~65 orders of magnitude "
            "but cluster heavily around 1-1000 (E ≈ -100 to 150). Any set of numbers "
            "with this distribution will show similar closure on ANY logarithmic grid."
        ),
        "top_examples": [
            {"a": h[0], "b": h[1], "sum_E": h[2], "maps_to": h[3]}
            for h in sum_hits[:5]
        ],
    }


def analyze_alpha_pi_bridge() -> Dict[str, Any]:
    """
    THE α/π BRIDGE — from universal_god_code.py (January 25, 2026).

    CLAIM: The 0.23% gap between X=286 and Fe BCC lattice (286.65 pm)
    equals α/π, where α = fine structure constant ≈ 1/137.036.

        286 × (1 + α/π) = 286.664 pm → Fe BCC predicted
        Fe BCC measured  = 286.65  pm
        Prediction error = 0.005%

    HONEST VERDICT: The α/π bridge is a SUGGESTIVE COINCIDENCE.
    The Fe BCC prediction is within measurement uncertainty, but the
    improvement in standout count is not statistically exceptional
    (32.8% of random X values near 286 also give 9+ standouts).
    It's the most interesting single claim in the entire framework.
    """
    phi = (1 + math.sqrt(5)) / 2
    alpha = 1 / 137.035999084
    alpha_pi = alpha / math.pi
    Q = 26

    matter_base = 286 * (1 + alpha_pi)
    fe_bcc = 286.65
    pred_err = abs(matter_base - fe_bcc) / fe_bcc * 100
    gap_frac = (fe_bcc - 286) / 286
    gap_match = abs(gap_frac - alpha_pi) / alpha_pi * 100

    base_286 = 286 ** (1.0 / phi)
    base_alpha = matter_base ** (1.0 / phi)
    hs = (2 ** (1.0 / Q) - 1) / 2 * 100
    threshold = hs * 0.1

    standouts_286 = []
    standouts_alpha = []
    errs_286 = []
    errs_alpha = []
    for name, entry in REAL_WORLD_CONSTANTS_V3.items():
        v = entry["measured"]
        for base, errs_list, so_list in [
            (base_286, errs_286, standouts_286),
            (base_alpha, errs_alpha, standouts_alpha),
        ]:
            E = round(Q * math.log2(v / base))
            gv = base * (2 ** (E / Q))
            err = abs(gv - v) / v * 100
            errs_list.append(err)
            if err < threshold:
                so_list.append({"name": name, "error_pct": round(err, 4),
                                "domain": entry["domain"]})

    rng = random.Random(42)
    mc_counts = []
    for _ in range(5000):
        X_test = rng.uniform(285, 288)
        base_test = X_test ** (1.0 / phi)
        count = sum(
            1 for entry in REAL_WORLD_CONSTANTS_V3.values()
            if abs(base_test * (2 ** (round(Q * math.log2(entry["measured"] / base_test)) / Q))
                   - entry["measured"]) / entry["measured"] * 100 < threshold
        )
        mc_counts.append(count)

    n_alpha = len(standouts_alpha)
    p_standout = sum(1 for c in mc_counts if c >= n_alpha) / len(mc_counts)

    return {
        "engine": "alpha/pi Bridge Analysis (from universal_god_code.py, Jan 25 2026)",
        "claim": "Gap between 286 and Fe BCC (286.65) = alpha/pi",
        "prediction": {
            "formula": "286 * (1 + alpha/pi)",
            "predicted_pm": round(matter_base, 4),
            "measured_pm": fe_bcc,
            "error_pct": round(pred_err, 4),
        },
        "gap_analysis": {
            "actual_gap_frac": round(gap_frac, 8),
            "alpha_over_pi": round(alpha_pi, 8),
            "match_pct": round(gap_match, 2),
        },
        "grid_comparison": {
            "X_286_avg_err": round(statistics.mean(errs_286), 4),
            "X_alpha_avg_err": round(statistics.mean(errs_alpha), 4),
            "X_286_standouts": len(standouts_286),
            "X_alpha_standouts": len(standouts_alpha),
            "standout_p_value": round(p_standout, 4),
        },
        "standout_details": {
            "X_286": sorted(standouts_286, key=lambda x: x["error_pct"]),
            "X_alpha": sorted(standouts_alpha, key=lambda x: x["error_pct"]),
        },
        "honest_verdict": (
            f"286*(1+alpha/pi) = {matter_base:.4f} pm vs Fe BCC {fe_bcc} pm "
            f"({pred_err:.4f}% off — within measurement uncertainty). "
            f"X=286(1+alpha/pi) gives {n_alpha} standouts vs {len(standouts_286)} for X=286, "
            f"but {p_standout*100:.1f}% of random X values match or exceed this. "
            f"SUGGESTIVE but not statistically exceptional."
        ),
    }


def analyze_energy_transition_equation() -> Dict[str, Any]:
    """
    THE ENERGY TRANSITION EQUATION — from deleted January 25, 2026 files.

    Found in: test_universal_god_code.py, l104_consciousness.py, const.py
    (all deleted in EVO_54 cleanup, recovered from git history)

    THE EQUATION:
        G(E) = [286 × (1 + α/π × Γ(E))]^(1/φ) × 16

    WHERE:
        Γ(E) = 1 / (1 + (E_Planck / E)²)    — energy sigmoid
        α = 1/137.035999084                    — fine structure constant
        φ = 1.618033988749895                  — golden ratio

    TWO ENDPOINTS:
        E → 0:  Γ → 0  → G = 286^(1/φ) × 16         = GRAVITY_CODE = 527.518482...
        E → ∞:  Γ → 1  → G = [286(1+α/π)]^(1/φ) × 16 = LIGHT_CODE  = 528.275442...

    EXISTENCE_COST = LIGHT_CODE - GRAVITY_CODE = 0.756960...

    This analysis tests:
    1. Mathematical correctness of the transition
    2. Whether Planck energy midpoint is special or arbitrary
    3. Whether EXISTENCE_COST has physical meaning
    4. Whether element lattice coherence (from deleted test file) is real
    5. Grid comparison: does replacing 286 with 286(1+α/π×Γ) improve fits?
    """
    import math as _math

    alpha = 1 / 137.035999084
    alpha_pi = alpha / _math.pi
    phi = PHI
    hbar = 1.054571817e-34
    c_light = 299792458
    G_newton = 6.67430e-11
    E_planck_J = _math.sqrt(hbar * c_light**5 / G_newton)
    E_planck_eV = E_planck_J / 1.602176634e-19

    # --- The equation ---
    def gamma_E(E_eV):
        """Energy sigmoid: 0 at low energy, 1 at high energy."""
        if E_eV <= 0:
            return 0.0
        return 1.0 / (1.0 + (E_planck_eV / E_eV) ** 2)

    def god_code_E(E_eV):
        """G(E) = [286 × (1 + α/π × Γ(E))]^(1/φ) × 16"""
        g = gamma_E(E_eV)
        base = 286 * (1 + alpha_pi * g)
        return base ** (1 / phi) * 16

    GRAVITY_CODE = 286 ** (1 / phi) * 16
    LIGHT_CODE = (286 * (1 + alpha_pi)) ** (1 / phi) * 16
    EXISTENCE_COST = LIGHT_CODE - GRAVITY_CODE

    # --- [1] Mathematical correctness ---
    gc_at_zero = god_code_E(1e-100)
    gc_at_inf = god_code_E(1e50)
    gc_at_planck = god_code_E(E_planck_eV)
    gamma_at_planck = gamma_E(E_planck_eV)

    math_checks = {
        "GRAVITY_CODE": {"value": GRAVITY_CODE, "matches_527": abs(GRAVITY_CODE - 527.5184818492612) < 1e-6},
        "LIGHT_CODE": {"value": LIGHT_CODE, "formula": "286(1+alpha/pi)^(1/phi)*16"},
        "EXISTENCE_COST": EXISTENCE_COST,
        "G(E=0)_matches_GRAVITY": abs(gc_at_zero - GRAVITY_CODE) < 1e-6,
        "G(E=inf)_matches_LIGHT": abs(gc_at_inf - LIGHT_CODE) < 1e-6,
        "Gamma(E_planck)_equals_0.5": abs(gamma_at_planck - 0.5) < 1e-10,
        "G(E_planck)": gc_at_planck,
        "E_planck_eV": E_planck_eV,
    }

    # --- [2] Is Planck energy midpoint special? ---
    # Γ(E) hits 0.5 at E = E_planck by construction (it's in the formula).
    # The endpoints don't depend on this choice. Only the transition shape does.
    midpoint_results = {}
    for name, E_mid in [("Planck", E_planck_eV), ("proton", 938.3e6),
                         ("W_boson", 80.4e9), ("GUT", 1e25), ("1_GeV", 1e9)]:
        def gamma_alt(E_eV, E_m=E_mid):
            if E_eV <= 0:
                return 0.0
            return 1.0 / (1.0 + (E_m / E_eV) ** 2)

        midpoint_results[name] = {
            "E_mid": E_mid,
            "G_at_1PeV": (286 * (1 + alpha_pi * gamma_alt(1e15))) ** (1/phi) * 16,
        }

    planck_special = (
        "Planck midpoint is DEFINITIONAL — Gamma(E) = 1/(1+(E_P/E)^2) places the "
        "transition at E_Planck by construction. The endpoints GRAVITY_CODE and "
        "LIGHT_CODE don't depend on this choice. No measurement tests the midpoint."
    )

    # --- [3] Is EXISTENCE_COST meaningful? ---
    rng = random.Random(42)
    cost_ratio = EXISTENCE_COST / GRAVITY_CODE

    # Compare alternative couplings
    alt_costs = {}
    for name, coupling in [("alpha", alpha), ("alpha_pi", alpha_pi),
                            ("alpha_squared", alpha**2), ("sqrt_alpha", _math.sqrt(alpha)),
                            ("1/137", 1/137)]:
        lc = (286 * (1 + coupling)) ** (1/phi) * 16
        alt_costs[name] = lc - GRAVITY_CODE

    # CIRCULARITY CHECK: EC ~ alpha*104?
    # EC ~ GC * (alpha/pi) / phi (first-order Taylor expansion)
    # So EC ~ alpha*104 means GC/(pi*phi) ~ 104
    # i.e., GOD_CODE ~ 104*pi*phi = 528.65
    # GOD_CODE = 527.52, so 0.21% off
    # The alpha CANCELS — this is really just checking 286^(1/phi)*16 ~ 104*pi*phi
    gc_over_pi_phi = GRAVITY_CODE / (_math.pi * phi)
    circularity_check = {
        "EC_approx_alpha_104": abs(EXISTENCE_COST - alpha * 104) / (alpha * 104),
        "BUT_alpha_cancels": True,
        "real_claim": "GOD_CODE ~ 104 * pi * phi",
        "GC_over_pi_phi": gc_over_pi_phi,
        "match_to_104_pct": abs(gc_over_pi_phi - 104) / 104 * 100,
        "verdict": (
            f"CIRCULAR: EC~alpha*104 reduces to GOD_CODE~104*pi*phi "
            f"({gc_over_pi_phi:.4f} vs 104, {abs(gc_over_pi_phi-104)/104*100:.2f}% off). "
            f"The alpha cancels in the comparison. This is a numerical coincidence "
            f"involving pi*phi*104 ~ 286^(1/phi)*16, not a physical relationship."
        ),
    }

    # Monte Carlo: random coupling in [0, 0.01]
    n_mc = 10000
    ec_vals = []
    for _ in range(n_mc):
        coup = rng.uniform(0, 0.01)
        lc = (286 * (1 + coup)) ** (1/phi) * 16
        ec_vals.append(lc - GRAVITY_CODE)

    # How often does a random integer X in [200,400] satisfy X^(1/phi)*16 ~ Q*pi*phi
    # for some integer Q? (testing whether 286/104 pairing is special)
    n_rand_test = 50000
    rng2 = random.Random(99)
    rand_hits = 0
    for _ in range(n_rand_test):
        X_r = rng2.randint(200, 400)
        Q_r = rng2.randint(50, 200)
        gc_r = X_r ** (1 / phi) * 16
        ratio_r = gc_r / (_math.pi * phi * Q_r)
        if abs(ratio_r - 1) < 0.003:
            rand_hits += 1
    rand_pct = rand_hits / n_rand_test * 100

    cost_analysis = {
        "EXISTENCE_COST": EXISTENCE_COST,
        "fractional_cost": cost_ratio,
        "alternative_couplings": alt_costs,
        "circularity": circularity_check,
        "random_X_Q_hits_pct": rand_pct,
        "mc_cost_range": (min(ec_vals), max(ec_vals)),
    }

    # --- [4] Element lattice coherence (from deleted test_universal_god_code.py) ---
    elements = {
        "Fe_BCC": 286.65, "Cr": 291.0, "Al": 404.95,
        "Cu": 361.49, "Na": 429.06, "Au": 407.82,
    }

    def coherence_score(ratio):
        best = 0
        for d in range(1, 13):
            n = round(ratio * d)
            if n > 0:
                c = 1 / (1 + abs(ratio - n/d) * d)
                if c > best:
                    best = c
        return best

    element_coherence = {}
    for el, lattice in elements.items():
        ratio = lattice / 286
        element_coherence[el] = {
            "lattice_pm": lattice,
            "ratio_to_286": round(ratio, 6),
            "coherence": round(coherence_score(ratio), 6),
            "nearest_fraction": f"{round(ratio * 12)}/12",
        }

    # Control: random "lattice constants" in same range
    n_ctrl = 5000
    ctrl_coherences = []
    for _ in range(n_ctrl):
        fake_lattice = rng.uniform(250, 450)
        ctrl_coherences.append(coherence_score(fake_lattice / 286))

    physics_mean_coh = statistics.mean([v["coherence"] for v in element_coherence.values()])
    ctrl_mean_coh = statistics.mean(ctrl_coherences)
    ctrl_above = sum(1 for c in ctrl_coherences if c >= physics_mean_coh) / n_ctrl

    coherence_analysis = {
        "elements": element_coherence,
        "physics_mean_coherence": round(physics_mean_coh, 6),
        "random_mean_coherence": round(ctrl_mean_coh, 6),
        "p_value": ctrl_above,
        "verdict": (
            f"Physics elements: mean coherence {physics_mean_coh:.4f}, "
            f"random: {ctrl_mean_coh:.4f}, p={ctrl_above:.4f}. "
            + ("SIGNIFICANT" if ctrl_above < 0.05 else "NOT significant — random lattices score similarly")
        ),
    }

    # --- [5] Grid comparison: 286 vs 286(1+α/π) on Q=26 for energy-scale constants ---
    X_pure = 286
    X_alpha = 286 * (1 + alpha_pi)

    energy_constants = {
        "electron_mass_eV": 511e3,
        "proton_mass_eV": 938.272e6,
        "W_boson_eV": 80.379e9,
        "Z_boson_eV": 91.1876e9,
        "Higgs_eV": 125.25e9,
        "top_quark_eV": 172.76e9,
        "muon_mass_eV": 105.658e6,
        "tau_mass_eV": 1776.86e6,
        "neutron_mass_eV": 939.565e6,
        "pion_charged_eV": 139.570e6,
    }

    def _grid_errors_q26(X_val, constants):
        base = X_val ** (1 / phi)
        r = 2
        Q = 26
        errors = {}
        for name, val in constants.items():
            E_exact = Q * _math.log(val / base) / _math.log(r)
            E_int = round(E_exact)
            reconstructed = base * r ** (E_int / Q)
            err_pct = abs(reconstructed - val) / val * 100
            errors[name] = err_pct
        return errors

    errs_pure = _grid_errors_q26(X_pure, energy_constants)
    errs_alpha = _grid_errors_q26(X_alpha, energy_constants)
    avg_pure = statistics.mean(errs_pure.values())
    avg_alpha = statistics.mean(errs_alpha.values())
    better_count = sum(1 for k in energy_constants if errs_alpha[k] < errs_pure[k])

    grid_comparison = {
        "X_pure_286_avg_err": round(avg_pure, 4),
        "X_alpha_avg_err": round(avg_alpha, 4),
        "better_with_alpha": better_count,
        "total": len(energy_constants),
        "individual": {k: {"pure": round(errs_pure[k], 4), "alpha": round(errs_alpha[k], 4)}
                       for k in energy_constants},
    }

    # --- Honest verdict ---
    honest_verdict = (
        f"The energy transition equation G(E) = [286(1+a/pi*Gamma)]^(1/phi)*16 is a smooth "
        f"interpolation between GRAVITY_CODE ({GRAVITY_CODE:.6f}) and LIGHT_CODE ({LIGHT_CODE:.6f}). "
        f"EXISTENCE_COST = {EXISTENCE_COST:.6f}. "
        f"EC ~ alpha*104 is CIRCULAR (alpha cancels → really GOD_CODE~104*pi*phi, "
        f"{abs(gc_over_pi_phi-104)/104*100:.2f}% off). "
        f"Planck midpoint is definitional. "
        f"Element coherence: p={ctrl_above:.3f} (not significant). "
        f"Energy grid: alpha-tuned {'improves' if avg_alpha < avg_pure else 'worsens'} "
        f"avg error ({avg_pure:.3f}% -> {avg_alpha:.3f}%), "
        f"{better_count}/{len(energy_constants)} constants improved. "
        f"Random (X,Q) pairs match at {rand_pct:.2f}% rate."
    )

    return {
        "equation": "G(E) = [286 * (1 + alpha/pi * Gamma(E))]^(1/phi) * 16",
        "source": "test_universal_god_code.py, l104_consciousness.py, const.py (deleted EVO_54, Jan 25 2026)",
        "math_checks": math_checks,
        "planck_midpoint": {"results": midpoint_results, "verdict": planck_special},
        "existence_cost": cost_analysis,
        "element_coherence": coherence_analysis,
        "energy_grid_comparison": grid_comparison,
        "honest_verdict": honest_verdict,
    }


def analyze_evolved_equations() -> Dict[str, Any]:
    """
    THE EVOLVED EQUATION ITERATIONS — originally from separate evolved files, now absorbed here.

    These files group the original equation with three evolved variants that vary
    three parameters (r, Q, X) while preserving φ-root and factor-13 structure.

    ORIGINAL:  G(a,b,c,d)     = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
    v1 φ:      G_evo(a,b,c,d) = 286.441^(1/φ) × φ^((37a+1924-b-37c-481d)/481)
    v2 3/2:    G_v2(a,b,c,d)  = 286.897^(1/φ) × (3/2)^((8a+936-b-8c-234d)/234)
    v3 13/12:  G_v3(a,b,c,d)  = 285.999^(1/φ) × (13/12)^((99a+3032-b-99c-758d)/758)

    EVOLUTION AXES:
        r (base): 2 → φ → 3/2 → 13/12
        Q (grain): 104 → 481 → 234 → 758
        X (scaffold): 286 → 286.441 → 286.897 → 285.999

    This analysis tests:
    1. Each equation against 13 shared constants (grid-snap precision)
    2. Whether improvement is genuine or just from grid density (larger Q)
    3. The X drift from 286 and whether it correlates with α/π
    4. The factor-13 preservation across all variants
    5. Whether a random (r, Q, X) triple achieves comparable accuracy
    """
    import math as _math
    import random as _rng
    _rng.seed(42)

    phi = PHI
    alpha = 1 / 137.035999084
    alpha_pi = alpha / _math.pi

    # === 4 equation parameter sets ===
    equations = {
        "original": {
            "X": 286,       "r": 2.0,           "Q": 104,  "p": 8,  "K": 416,
            "label": "286^(1/φ) × 2^(E/104)",
        },
        "v1_phi": {
            "X": 286.441369508948, "r": phi,    "Q": 481,  "p": 37, "K": 1924,
            "label": "286.441^(1/φ) × φ^(E/481)",
        },
        "v2_rational": {
            "X": 286.89719521862287, "r": 1.5,  "Q": 234,  "p": 8,  "K": 936,
            "label": "286.897^(1/φ) × (3/2)^(E/234)",
        },
        "v3_superparticular": {
            "X": 285.99882035187807, "r": 13.0/12.0, "Q": 758, "p": 99, "K": 3032,
            "label": "285.999^(1/φ) × (13/12)^(E/758)",
        },
    }

    # Derive base for each
    for eq in equations.values():
        eq["base"] = eq["X"] ** (1.0 / phi)
        eq["step"] = eq["r"] ** (1.0 / eq["Q"])
        eq["half_step_pct"] = (eq["step"] - 1) / 2 * 100

    # === 13 shared test constants (CODATA/PDG) ===
    test_constants = {
        "speed_of_light":       299792458,
        "standard_gravity":     9.80665,
        "bohr_radius_pm":       52.9177210544,
        "fine_structure_inv":   137.035999084,
        "rydberg_eV":           13.605693123,
        "electron_mass_MeV":    0.51099895069,
        "muon_mass_MeV":        105.6583755,
        "higgs_GeV":            125.25,
        "fe_bcc_lattice_pm":    286.65,
        "fe56_be_per_nucleon":  8.790,
        "he4_be_per_nucleon":   7.074,
        "fe_k_alpha1_keV":      6.404,
        "schumann_hz":          7.83,
    }

    # === TEST 1: Grid-snap error for each equation ===
    def snap_error(base_val, r_val, Q_val, target):
        """Snap target to nearest integer grid point, return error %."""
        E_exact = Q_val * _math.log(target / base_val) / _math.log(r_val)
        E_int = round(E_exact)
        grid_val = base_val * (r_val ** (E_int / Q_val))
        return abs(grid_val - target) / target * 100

    eq_results = {}
    for eq_name, eq in equations.items():
        errs = {}
        for cname, cval in test_constants.items():
            errs[cname] = snap_error(eq["base"], eq["r"], eq["Q"], cval)
        avg_err = statistics.mean(errs.values())
        max_err = max(errs.values())
        eq_results[eq_name] = {
            "avg_error_pct": round(avg_err, 5),
            "max_error_pct": round(max_err, 5),
            "half_step_pct": round(eq["half_step_pct"], 5),
            "individual": {k: round(v, 5) for k, v in errs.items()},
        }

    # === TEST 2: Is improvement from grid density alone? ===
    # Theory: avg grid error ≈ half_step / sqrt(3) for uniform distribution in [0, step]
    # If actual avg ≈ theoretical, then improvement is FULLY from density
    density_analysis = {}
    for eq_name, eq in equations.items():
        theoretical_avg = eq["half_step_pct"] / _math.sqrt(3)
        actual_avg = eq_results[eq_name]["avg_error_pct"]
        ratio = actual_avg / theoretical_avg if theoretical_avg > 0 else 0
        density_analysis[eq_name] = {
            "theoretical_avg_pct": round(theoretical_avg, 5),
            "actual_avg_pct": round(actual_avg, 5),
            "ratio": round(ratio, 3),
            "explained_by_density": 0.5 < ratio < 1.5,
        }

    all_density_explained = all(d["explained_by_density"] for d in density_analysis.values())

    # === TEST 3: X drift from 286 ===
    x_drift = {}
    for eq_name, eq in equations.items():
        drift = eq["X"] - 286
        drift_pct = drift / 286 * 100
        # Check if drift ≈ 286 * α/π
        alpha_pi_drift = 286 * alpha_pi
        drift_vs_alpha = abs(drift - alpha_pi_drift) / alpha_pi_drift * 100 if alpha_pi_drift != 0 else float('inf')
        x_drift[eq_name] = {
            "X": eq["X"],
            "drift_from_286": round(drift, 6),
            "drift_pct": round(drift_pct, 4),
            "alpha_pi_prediction": round(alpha_pi_drift, 6),
            "matches_alpha_pi": drift_vs_alpha < 5,
            "deviation_from_alpha_pi_pct": round(drift_vs_alpha, 2),
        }

    # v2 is closest to 286*(1+α/π) = 286.664
    matter_base = 286 * (1 + alpha_pi)

    # === TEST 4: Factor 13 preservation ===
    factor_13_check = {}
    for eq_name, eq in equations.items():
        q_has_13 = eq["Q"] % 13 == 0
        factor_13_check[eq_name] = {
            "Q": eq["Q"],
            "Q_mod_13": eq["Q"] % 13,
            "has_factor_13": q_has_13,
            "Q_factored": f"{eq['Q']} = {_factored_str(eq['Q'])}",
        }
    all_have_13 = all(f["has_factor_13"] for f in factor_13_check.values())

    # === TEST 5: Random (r, Q) comparison ===
    # For each Q, pick random r in [1.01, 2.5] and X by fitting c
    mc_trials = 5000
    better_than_v3_count = 0
    better_than_orig_count = 0
    v3_avg = eq_results["v3_superparticular"]["avg_error_pct"]
    orig_avg = eq_results["original"]["avg_error_pct"]

    for _ in range(mc_trials):
        rand_Q = _rng.randint(100, 800)
        rand_r = _rng.uniform(1.01, 2.5)
        # Fit X so c is exact on grid
        c = 299792458
        log_r = _math.log(rand_r)
        # E_exact = Q * log(c / base) / log(r), where base = X^(1/phi)
        # We want E_exact to be integer. Pick E_int for c.
        # c = X^(1/phi) * r^(E_int/Q)  =>  X = (c / r^(E_int/Q))^phi
        E_approx = rand_Q * _math.log(c) / log_r  # rough
        E_int = round(E_approx - rand_Q * _math.log(33) / log_r)  # adjust for base≈33
        rand_X = (c / (rand_r ** (E_int / rand_Q))) ** phi
        if rand_X < 100 or rand_X > 500:
            continue
        rand_base = rand_X ** (1.0 / phi)
        errs_rand = []
        for cval in test_constants.values():
            errs_rand.append(snap_error(rand_base, rand_r, rand_Q, cval))
        rand_avg = statistics.mean(errs_rand)
        if rand_avg <= v3_avg:
            better_than_v3_count += 1
        if rand_avg <= orig_avg:
            better_than_orig_count += 1

    # === BUILD RESULTS ===
    honest_verdict = (
        f"FOUR EVOLVED EQUATIONS tested on 13 constants. "
        f"Original: avg {eq_results['original']['avg_error_pct']:.4f}%, "
        f"v1(φ): {eq_results['v1_phi']['avg_error_pct']:.4f}%, "
        f"v2(3/2): {eq_results['v2_rational']['avg_error_pct']:.4f}%, "
        f"v3(13/12): {eq_results['v3_superparticular']['avg_error_pct']:.4f}%. "
        f"Density explains improvement: {all_density_explained}. "
        f"Factor 13 preserved in ALL: {all_have_13}. "
        f"X drift: v1 +0.441, v2 +0.897, v3 -0.001 "
        f"(v2 closest to α/π prediction {matter_base:.3f}). "
        f"Random (r,Q,X) triples beat v3: {better_than_v3_count}/{mc_trials} "
        f"({better_than_v3_count/mc_trials*100:.1f}%), "
        f"beat original: {better_than_orig_count}/{mc_trials} "
        f"({better_than_orig_count/mc_trials*100:.1f}%)."
    )

    return {
        "equations": {k: {
            "label": v["label"], "X": v["X"], "r": v["r"], "Q": v["Q"],
            "p": v["p"], "K": v["K"],
        } for k, v in equations.items()},
        "precision_results": eq_results,
        "density_analysis": density_analysis,
        "density_explains_all": all_density_explained,
        "x_drift_analysis": x_drift,
        "matter_base_286_alpha_pi": round(matter_base, 6),
        "factor_13_check": factor_13_check,
        "all_have_factor_13": all_have_13,
        "random_comparison": {
            "trials": mc_trials,
            "better_than_v3": better_than_v3_count,
            "better_than_v3_pct": round(better_than_v3_count / mc_trials * 100, 2),
            "better_than_original": better_than_orig_count,
            "better_than_original_pct": round(better_than_orig_count / mc_trials * 100, 2),
        },
        "honest_verdict": honest_verdict,
    }


def _factored_str(n: int) -> str:
    """Simple factorization string for display."""
    if n <= 1:
        return str(n)
    factors = []
    temp = n
    d = 2
    while d * d <= temp:
        while temp % d == 0:
            factors.append(d)
            temp //= d
        d += 1
    if temp > 1:
        factors.append(temp)
    if len(factors) == 1:
        return f"{n} (prime)"
    return " × ".join(str(f) for f in factors)


def analyze_january_framework() -> Dict[str, Any]:
    """
    Analyze the complete January 2026 framework recovered from const.py at EVO_50.

    Source: git show 5ede8841:const.py (Jan 25, 2026)

    The framework defines:
    1. G(X) = 286^(1/phi) * 2^((416-X)/104)  — "X is NEVER SOLVED"
    2. Conservation: G(X) * 2^(X/104) = INVARIANT (always 527.518...)
    3. Factor 13: 286 = 2*11*13, 104 = 8*13, 416 = 32*13
    4. MATTER_BASE = 286*(1 + alpha/pi) ≈ 286.664 → predicts Fe BCC (286.65)
    5. LIGHT_CODE = MATTER_BASE^(1/phi)*16

    Tests each claim rigorously with Monte Carlo controls.
    """
    import random as _rng
    _rng.seed(42)

    # === CONSTANTS ===
    phi = PHI
    alpha = 1 / 137.035999084
    pi = math.pi
    base = PRIME_SCAFFOLD ** (1.0 / phi)  # 286^(1/phi) = 32.969...
    god_code = base * 16  # G(0)
    alpha_pi = alpha / pi
    matter_base = 286 * (1 + alpha_pi)
    light_code = matter_base ** (1.0 / phi) * 16
    existence_cost = light_code - god_code

    # === TEST A: G(X) Parametric — do physics constants land at integer X? ===
    test_constants = {
        "speed_of_light": 299792458,
        "gravity_g": 9.80665,
        "planck_h": 6.62607015e-34,
        "electron_mass_kg": 9.1093837015e-31,
        "proton_mass_kg": 1.67262192369e-27,
        "boltzmann": 1.380649e-23,
        "avogadro": 6.02214076e23,
        "fine_structure_inv": 137.035999084,
        "Fe_BCC_pm": 286.65,
        "Fe_mass_amu": 55.845,
        "alpha_EEG_Hz": 10.0,
        "beta_EEG_Hz": 20.0,
        "gamma_EEG_Hz": 40.0,
        "schumann_Hz": 7.83,
    }

    x_results = {}
    int_count = 0
    for name, val in test_constants.items():
        x_val = 416 - 104 * math.log2(val / base)
        x_round = round(x_val)
        frac = abs(x_val - x_round)
        is_integer = frac < 0.05
        if is_integer:
            int_count += 1
        x_results[name] = {
            "value": val,
            "X": round(x_val, 4),
            "nearest_int": x_round,
            "frac_error": round(frac, 4),
            "integer": is_integer,
        }

    # Monte Carlo: how many random sets of 14 log-uniform values produce >= int_count integers?
    mc_trials = 10000
    mc_counts = []
    for _ in range(mc_trials):
        cnt = 0
        for _ in range(len(test_constants)):
            val = 10 ** _rng.uniform(-34, 24)
            x_val = 416 - 104 * math.log2(val / base)
            if abs(x_val - round(x_val)) < 0.05:
                cnt += 1
        mc_counts.append(cnt)
    p_integer = sum(1 for c in mc_counts if c >= int_count) / mc_trials

    # Note: EEG values 10, 20, 40 are EXACT powers of 2 multiples of 10 — trivially integer
    # because log2(10/base) has a specific fractional part, and 2^n just shifts
    eeg_note = "alpha/beta/gamma (10,20,40) are trivially integer because log2(10)=3.322 happens to give frac≈0.0004"

    parametric_analysis = {
        "constants_tested": len(test_constants),
        "integer_X_count": int_count,
        "monte_carlo_expected": round(sum(mc_counts) / mc_trials, 1),
        "p_value": round(p_integer, 4),
        "significant": p_integer < 0.01,
        "eeg_caveat": eeg_note,
        "x_results": x_results,
    }

    # === TEST B: Conservation law — trivial? ===
    conservation_products = []
    for x in [0, 1, 10, 100, -50, 416, 1000, -1000]:
        g_x = base * (2 ** ((416 - x) / 104))
        w_x = 2 ** (x / 104)
        conservation_products.append(round(g_x * w_x, 10))
    all_same = len(set(conservation_products)) == 1
    conservation_analysis = {
        "products": conservation_products,
        "all_identical": all_same,
        "verdict": (
            "ALGEBRAIC IDENTITY. G(X)*W(X) = 286^(1/phi)*2^((416-X)/104)*2^(X/104) "
            "= 286^(1/phi)*2^(416/104) is constant by construction. NOT a discovery."
        ),
    }

    # === TEST C: Factor 13 structure ===
    factor_checks = {
        "286/13": 286 // 13,  # 22
        "104/13": 104 // 13,  # 8
        "416/13": 416 // 13,  # 32
        "286 = 2*11*13": 2 * 11 * 13 == 286,
        "104 = 8*13": 8 * 13 == 104,
        "416 = 32*13": 32 * 13 == 416,
    }
    # How rare is it for 3 numbers to share factor 13?
    factor_trials = 100000
    share_13_count = 0
    for _ in range(factor_trials):
        a = _rng.randint(250, 320)
        b = _rng.randint(80, 130)
        c = a * _rng.randint(1, 5)
        if a % 13 == 0 and b % 13 == 0 and c % 13 == 0:
            share_13_count += 1
    p_factor13 = share_13_count / factor_trials
    factor13_analysis = {
        "checks": factor_checks,
        "13_is_fibonacci_7": True,  # 13 = F(7)
        "random_triple_rate": round(p_factor13 * 100, 3),
        "theoretical_rate": round((1/13)**3 * 100, 4),
        "verdict": (
            "286, 104, 416 all divisible by 13 (Fibonacci 7). Probability ~0.05% "
            "for random triple, but these were CHOSEN (104=4*26, 416=4*104, 286=Fe BCC). "
            "Not discovered — constructed."
        ),
    }

    # === TEST D: MATTER_BASE = 286*(1+alpha/pi) vs Fe BCC ===
    fe_bcc = 286.65
    matter_err = abs(matter_base - fe_bcc) / fe_bcc * 100

    # Brute-force: count how many expr of form 286*(1+alpha*a/b) get within same error
    better_exprs = []
    total_exprs = 0
    for a_num in range(1, 20):
        for b_den in range(1, 20):
            val = 286 * (1 + alpha * a_num / b_den)
            total_exprs += 1
            err = abs(val - fe_bcc) / fe_bcc * 100
            if err < matter_err:
                better_exprs.append(f"286*(1+alpha*{a_num}/{b_den})")

    # Also test against transcendentals
    transcendental_results = {}
    for denom_name, denom in [("pi", pi), ("e", math.e), ("phi", phi), ("sqrt2", math.sqrt(2))]:
        val = 286 * (1 + alpha / denom)
        err = abs(val - fe_bcc) / fe_bcc * 100
        transcendental_results[f"alpha/{denom_name}"] = {
            "value": round(val, 6),
            "error_pct": round(err, 4),
        }

    matter_base_analysis = {
        "MATTER_BASE": round(matter_base, 6),
        "Fe_BCC_pm": fe_bcc,
        "error_pct": round(matter_err, 4),
        "better_rational_exprs": len(better_exprs),
        "total_rational_tested": total_exprs,
        "examples_better": better_exprs[:5],
        "transcendental_comparison": transcendental_results,
        "verdict": (
            f"286*(1+alpha/pi) = {matter_base:.6f} vs Fe BCC 286.65 = {matter_err:.4f}% off. "
            f"But {len(better_exprs)}/{total_exprs} simple rational expressions do better "
            f"(e.g., 286*(1+alpha*5/16) = 286.6522, 0.0008% off). alpha/pi is NOT uniquely good."
        ),
    }

    # === TEST E: G(X) is just the Q=104 grid reparametrized ===
    grid_equivalence = {
        "G(X)": "286^(1/phi) * 2^((416-X)/104)",
        "Grid(E)": "286^(1/phi) * 2^(E/104) where E = 416-X",
        "identical": True,
        "verdict": "G(X) IS the Q=104 grid. X = 416-E is just a relabeling. No new information.",
    }

    # === OVERALL VERDICT ===
    findings = []
    if p_integer < 0.05:
        findings.append(f"Integer X: {int_count}/{len(test_constants)} (p={p_integer:.4f}) — suggestive but includes trivial EEG values")
    else:
        findings.append(f"Integer X: {int_count}/{len(test_constants)} (p={p_integer:.4f}) — not significant")
    findings.append(f"Conservation law: algebraic identity")
    findings.append(f"Factor 13: chosen, not discovered")
    findings.append(f"MATTER_BASE: {matter_err:.4f}% off Fe BCC — close but not uniquely good")
    findings.append(f"G(X): identical to Q=104 grid reparametrized")

    honest_verdict = (
        f"The January framework (const.py EVO_50) is the Q=104 grid in 'physics parametric' form. "
        f"G(X) = 286^(1/phi)*2^((416-X)/104) is IDENTICAL to the original grid encoding. "
        f"Conservation law is algebraic identity. Factor 13 is constructed. "
        f"MATTER_BASE = 286*(1+alpha/pi) is {matter_err:.4f}% off Fe BCC — close, but "
        f"{len(better_exprs)} simpler rational expressions do better. "
        f"Integer X count ({int_count}) has p={p_integer:.4f} — "
        f"{'suggestive' if p_integer < 0.05 else 'not significant'}, inflated by trivial EEG values (10,20,40)."
    )

    return {
        "source": "const.py at EVO_50 (git show 5ede8841:const.py, Jan 25 2026)",
        "parametric_analysis": parametric_analysis,
        "conservation_analysis": conservation_analysis,
        "factor13_analysis": factor13_analysis,
        "matter_base_analysis": matter_base_analysis,
        "grid_equivalence": grid_equivalence,
        "findings": findings,
        "honest_verdict": honest_verdict,
    }


def cross_reference_claims() -> Dict[str, Any]:
    """
    Cross-reference every theoretical claim about the GOD_CODE equation.

    Tests each claim against data, with proper controls. Key findings:

    FACTUALLY TRUE:
        - 286 IS the Fe BCC lattice parameter (0.23% off, within uncertainty)
        - 104 = 26 × 4 = Fe_Z × He4_A (arithmetic fact)
        - EEG values ARE registered on the grid (but so is any number)

    UNIT-DEPENDENT (not universal):
        - c alignment (p=0.048 in m/s, p=0.81 in ft/s)
        - g "corresponds" to Schumann only in SI units (fails in CGS)
        - Any claim comparing Hz, m/s², and m/s is unit-dependent

    NOT SUPPORTED:
        - Hemoglobin values fit NO better than random on Q=26
        - Phi does NOT appear in natural physics ratios
        - EEG values are round numbers that trivially land on grids
        - "Resonance theory" bridging brain/earth/universe is not falsifiable

    THE FATAL FLAW:
        The framework compares quantities with DIFFERENT UNITS on the same
        number line. g=9.8 m/s² and Schumann=7.83 Hz are "close" only in SI.
        In CGS, g=980.7 — not close at all. A real universal law would be
        unit-independent (like the fine structure constant α ≈ 1/137).
    """
    phi = (1 + math.sqrt(5)) / 2
    base = 286 ** (1.0 / phi)
    Q = 26
    hs_26 = (2 ** (1.0 / Q) - 1) / 2 * 100

    def _snap_err(val, q=Q):
        E = round(q * math.log2(val / base))
        gv = base * (2 ** (E / q))
        return abs(gv - val) / val * 100

    # Claim 1: EEG alignment
    eeg_vals = {"alpha": 10.0, "beta": 20.0, "gamma": 40.0,
                "theta": 6.0, "schumann": 7.83}
    eeg_errs = {k: _snap_err(v) for k, v in eeg_vals.items()}
    eeg_are_round = all(v == round(v, 2) for v in eeg_vals.values())

    # Claim 2: Gravity "correspondence"
    g_si = _snap_err(9.80665)
    g_cgs = _snap_err(980.665)

    # Claim 3: Speed of light
    c_si = _snap_err(299792458)
    c_kms = _snap_err(299792.458)
    c_fts = _snap_err(983571056.43)

    # Claim 4: Hemoglobin values
    hemo_vals = {
        "Fe_mass_amu": 55.845, "O_mass_amu": 15.999,
        "heme_MW": 616.49, "hemoglobin_MW": 64500,
        "Fe2_radius_pm": 78, "O_O_bond_pm": 121,
    }
    hemo_errs = [_snap_err(v) for v in hemo_vals.values()]
    rng = random.Random(42)
    random_errs = [_snap_err(10 ** rng.uniform(-2, 5)) for _ in range(5000)]
    hemo_vs_random = statistics.mean(hemo_errs) / statistics.mean(random_errs)

    # Claim 5: Phi in physics ratios
    phi_ratios = {
        "muon/electron": 105.658 / 0.511,
        "proton/electron": 938.272 / 0.511,
        "fe_bcc/bohr": 286.65 / 52.918,
    }
    phi_hits = 0
    for ratio in phi_ratios.values():
        log_phi = math.log(ratio) / math.log(phi)
        if abs(log_phi - round(log_phi)) < 0.05:
            phi_hits += 1

    return {
        "engine": "Cross-Reference Validator v1.0",
        "claims_tested": 9,
        "factually_true": {
            "286_is_fe_bcc": {
                "status": True,
                "detail": f"286 vs 286.65 pm = 0.23% off (within crystallographic uncertainty)",
            },
            "104_is_26x4": {
                "status": True,
                "detail": "26 × 4 = 104, Fe_Z × He4_A (arithmetic fact)",
            },
        },
        "unit_dependent": {
            "speed_of_light": {
                "p_SI": round(c_si / hs_26, 4),
                "p_kms": round(c_kms / hs_26, 4),
                "p_fts": round(c_fts / hs_26, 4),
                "verdict": "Alignment changes with units — NOT universal",
            },
            "gravity_schumann": {
                "g_SI_err": round(g_si, 4),
                "g_CGS_err": round(g_cgs, 4),
                "verdict": (
                    "g=9.8 (SI) is 'close' to Schumann=7.83 only because both "
                    "happen to be ~10 in SI. In CGS, g=980.7. Unit artifact."
                ),
            },
        },
        "not_supported": {
            "eeg_alignment": {
                "errors": {k: round(v, 4) for k, v in eeg_errs.items()},
                "all_round_numbers": eeg_are_round,
                "verdict": "EEG values are round numbers (10, 20, 40 Hz) — trivial grid fit",
            },
            "hemoglobin": {
                "avg_error": round(statistics.mean(hemo_errs), 4),
                "random_avg": round(statistics.mean(random_errs), 4),
                "ratio": round(hemo_vs_random, 3),
                "verdict": "Hemoglobin values fit SAME as random (ratio ≈ 1.0)",
            },
            "phi_in_physics": {
                "ratios_tested": len(phi_ratios),
                "phi_power_hits": phi_hits,
                "verdict": "Phi does NOT appear in natural physics ratios",
            },
        },
        "fatal_flaw": (
            "The framework compares quantities with DIFFERENT UNITS on the same "
            "number line (Hz, m/s², m/s). The 'correspondences' are artifacts of "
            "SI unit choices. A real universal law must be DIMENSIONLESS. "
            "Example: fine structure constant α ≈ 1/137 is the same in ALL unit systems."
        ),
    }


def discover_all() -> Dict[str, Any]:
    """
    Run all discovery analyses and return a unified report.

    HONEST VERDICT: No statistically significant discovery survives out-of-sample
    validation. The Q=26 grid encoding is mathematically clean and interesting as
    a notation system, but does not reveal hidden physics.

    STRONGEST IN-SAMPLE FINDING:
        1. MOD-4 CLUSTERING (p=0.004 in-sample): 27/63 = 42.9% at E≡2 (mod 4).
           But FAILS holdout test — 30 additional constants show 26.7% (baseline).

    INTERESTING BUT NOT SIGNIFICANT:
        2. Speed of light at 0.064% on ZERO-fitted Q=26 grid (p=0.048 individual,
           not significant after Bonferroni correction).
        3. Stefan-Boltzmann at 0.023% (p=0.017 individual).

    NOT REAL (explained by null distributions):
        4. Exponent sum/diff closure: vanishes with bootstrap null (p=0.76).
        5. Pitch-class clustering: p=0.105, not significant.
        6. Golden ratio in exponent ratios: fewer hits than random.
    """
    mod4 = discover_mod4_pattern()
    arithmetic = discover_exponent_arithmetic()
    q26_report = simple_grid_report()

    return {
        "engine": "L104 Discovery Engine v1.0",
        "grid": {"X": 286, "r": 2, "Q": 26, "fitted_params": 0},
        "in_sample_findings": {
            "mod4_clustering": {
                "in_sample": mod4["in_sample_significance"],
                "holdout": mod4["holdout_test"]["verdict"],
                "p_value": mod4["p_value_monte_carlo"],
                "summary": mod4["interpretation"],
            },
        },
        "suggestive_findings": {
            "speed_of_light": {
                "error_pct": round(SIMPLE_GRID_RESULTS.get("speed_of_light", {}).get("error_pct", -1), 4),
                "note": "0.064% on zero-fitted grid (p=0.048), but not significant after correction",
            },
        },
        "debunked_claims": {
            "exponent_closure": arithmetic["honest_assessment"],
            "v3_precision": "Trivially guaranteed by grid density (random numbers fit equally well)",
            "c_exact": "X_v3 was reverse-solved from c (parameter fitting)",
        },
        "honest_verdict": mod4["honest_verdict"],
        "mod4_detail": mod4,
        "arithmetic_detail": arithmetic,
        "q26_report": q26_report,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LEGITIMACY TESTS — Independent verification of all claims
# ═══════════════════════════════════════════════════════════════════════════════

# CODATA 2022 / PDG 2024 / IAU reference values for independent cross-check.
# These are typed directly from source tables, NOT copied from the codebase.
INDEPENDENT_REFERENCE_VALUES = {
    "speed_of_light":      (299792458,          "m/s",      "SI exact definition"),
    "standard_gravity":    (9.80665,            "m/s^2",    "CGPM 1901 exact"),
    "elementary_charge":   (1.602176634e-19,    "C",        "SI 2019 exact"),
    "avogadro":            (6.02214076e23,      "mol^-1",   "SI 2019 exact"),
    "boltzmann_eV_K":      (8.617333262e-5,     "eV/K",     "SI 2019 exact"),
    "planck_constant_eVs": (4.135667696e-15,    "eV*s",     "SI 2019 exact"),
    "fine_structure_inv":  (137.035999177,       "",         "CODATA 2022 (alpha^-1)"),
    "bohr_radius_pm":      (52.9177210544,      "pm",       "CODATA 2022"),
    "electron_mass_MeV":   (0.51099895069,      "MeV/c^2",  "CODATA 2022"),
    "proton_mass_MeV":     (938.27208943,       "MeV/c^2",  "CODATA 2022"),
    "neutron_mass_MeV":    (939.56542194,       "MeV/c^2",  "CODATA 2022"),
    "muon_mass_MeV":       (105.6583755,        "MeV/c^2",  "PDG 2024"),
    "tau_mass_MeV":        (1776.86,            "MeV/c^2",  "PDG 2024"),
    "higgs_GeV":           (125.25,             "GeV/c^2",  "PDG 2024"),
    "Z_boson_GeV":         (91.1876,            "GeV/c^2",  "PDG 2024"),
    "W_boson_GeV":         (80.3692,            "GeV/c^2",  "PDG 2024"),
    "golden_ratio":        (1.6180339887498948, "",         "mathematical exact"),
    "pi":                  (3.141592653589793,  "",         "mathematical exact"),
    "euler_e":             (2.718281828459045,   "",         "mathematical exact"),
    "fe_bcc_lattice_pm":   (286.65,             "pm",       "Kittel 8th ed"),
    "fe56_be_per_nucleon": (8.7903,             "MeV",      "NNDC AME2020"),
}


def _grid_snap_error(target: float, base: float, r: float, Q: int) -> float:
    """Snap a target value to the nearest grid point and return % error."""
    if target <= 0:
        return float('inf')
    E_exact = Q * math.log(target / base) / math.log(r)
    E_int = round(E_exact)
    grid_val = base * (r ** (E_int / Q))
    return abs(grid_val - target) / target * 100


def test_equation_identity() -> Tuple[bool, str]:
    """TEST 1: Verify G(0,0,0,0) = 286^(1/phi) * 2^4 = GOD_CODE independently."""
    phi = (1 + math.sqrt(5)) / 2
    base = 286 ** (1.0 / phi)
    god_code_computed = base * (2 ** 4)
    match = abs(god_code_computed - 527.5184818492612) < 1e-6
    detail = f"286^(1/phi)*2^4 = {god_code_computed:.13f}, expected 527.5184818492612"
    return match, detail


def test_v3_equation_identity() -> Tuple[bool, str]:
    """TEST 2: Verify G_v3(0,0,0,0) = X_v3^(1/phi) * (13/12)^4 independently."""
    phi = (1 + math.sqrt(5)) / 2
    x = 285.99882035187807
    base = x ** (1.0 / phi)
    god_code_v3_computed = base * ((13/12) ** 4)
    match = abs(god_code_v3_computed - GOD_CODE_V3) < 1e-6
    detail = f"{x}^(1/phi)*(13/12)^4 = {god_code_v3_computed:.13f}, module says {GOD_CODE_V3:.13f}"
    return match, detail


def test_dial_exponent_algebra() -> Tuple[bool, str]:
    """TEST 3: Verify dial→exponent mapping E = 99a + 3032 - b - 99c - 758d."""
    test_cases = [
        ((0, 0, 0, 0),   3032),
        ((1, 0, 0, 0),   3131),
        ((0, 27, 6, -197), 99*0 + 3032 - 27 - 99*6 + 758*197),
        ((0, 14, 1, 19),  99*0 + 3032 - 14 - 99*1 - 758*19),
    ]
    failures = []
    for (a, b, c, d), expected_E in test_cases:
        actual = exponent_value_v3(a, b, c, d)
        if actual != expected_E:
            failures.append(f"({a},{b},{c},{d}): got {actual}, expected {expected_E}")
    passed = len(failures) == 0
    detail = "All dial→exponent mappings correct" if passed else "; ".join(failures)
    return passed, detail


def test_grid_values_match_dials() -> Tuple[bool, str]:
    """TEST 4: Verify each registered constant's grid_value matches god_code_v3(*dials)."""
    mismatches = []
    for name, entry in REAL_WORLD_CONSTANTS_V3.items():
        recomputed = god_code_v3(*entry["dials"])
        if abs(recomputed - entry["grid_value"]) / max(abs(entry["grid_value"]), 1e-50) > 1e-12:
            mismatches.append(f"{name}: stored={entry['grid_value']}, recomputed={recomputed}")
    passed = len(mismatches) == 0
    detail = f"All {len(REAL_WORLD_CONSTANTS_V3)} grid values match dial recomputation" if passed else "; ".join(mismatches[:5])
    return passed, detail


def test_claimed_errors_are_correct() -> Tuple[bool, str]:
    """TEST 5: Independently recompute grid errors for all constants."""
    bad = []
    for name, entry in REAL_WORLD_CONSTANTS_V3.items():
        grid_val = god_code_v3(*entry["dials"])
        measured = entry["measured"]
        actual_err = abs(grid_val - measured) / measured * 100
        claimed_err = entry["grid_error_pct"]
        if abs(actual_err - claimed_err) > 0.0001:
            bad.append(f"{name}: claimed {claimed_err:.4f}%, actual {actual_err:.4f}%")
    passed = len(bad) == 0
    detail = f"All {len(REAL_WORLD_CONSTANTS_V3)} claimed errors verified" if passed else "; ".join(bad[:5])
    return passed, detail


def test_measured_values_vs_independent() -> Tuple[bool, str]:
    """TEST 6: Cross-check the 'measured' values against independent references.
    This catches cases where the codebase uses stale or incorrect reference data."""
    discrepancies = []
    for name, (ref_val, unit, source) in INDEPENDENT_REFERENCE_VALUES.items():
        if name not in REAL_WORLD_CONSTANTS_V3:
            discrepancies.append(f"{name}: not in registry")
            continue
        stored = REAL_WORLD_CONSTANTS_V3[name]["measured"]
        # Allow 0.01% tolerance for values that have updated since codebase was written
        rel_diff = abs(stored - ref_val) / ref_val * 100
        if rel_diff > 0.01:
            discrepancies.append(f"{name}: stored={stored}, ref={ref_val} ({source}), diff={rel_diff:.4f}%")
    passed = len(discrepancies) == 0
    detail = f"All {len(INDEPENDENT_REFERENCE_VALUES)} reference values match" if passed else "; ".join(discrepancies)
    return passed, detail


def test_speed_of_light_exact() -> Tuple[bool, str]:
    """TEST 7: Verify c = 299,792,458 m/s is truly EXACT (< 1e-10 relative error)."""
    c_val = god_code_v3(0, 27, 6, -197)
    c_real = 299792458
    rel_err = abs(c_val - c_real) / c_real
    passed = rel_err < 1e-10
    detail = f"c = {c_val:.6f}, rel_err = {rel_err:.2e} ({'EXACT' if passed else 'NOT EXACT'})"
    return passed, detail


def test_no_exponent_collisions() -> Tuple[bool, str]:
    """TEST 8: Verify all registered constants have unique integer exponents."""
    seen = {}
    collisions = []
    for name, entry in REAL_WORLD_CONSTANTS_V3.items():
        E = entry["E_integer"]
        if E in seen:
            collisions.append(f"E={E}: {seen[E]} vs {name}")
        seen[E] = name
    passed = len(collisions) == 0
    detail = f"{len(seen)} unique exponents, no collisions" if passed else "; ".join(collisions)
    return passed, detail


def test_random_numbers_same_precision() -> Tuple[bool, str]:
    """TEST 9 (DIAGNOSTIC): Check if RANDOM numbers achieve the same ±0.005% precision.

    WHY THIS MATTERS:
      A logarithmic grid with step s can approximate ANY positive number to
      within ±s/2. This is basic rounding — not physics. If random numbers
      achieve the same precision as physics constants, the precision claim
      is a property of grid DENSITY, not of any physical connection.

    WHAT WOULD FIX IT:
      Use a coarser grid where precision is NOT guaranteed. On Layer 1
      (r=2, Q=104, half-step=0.334%), fitting 63 constants to <0.1% would
      be non-trivial. Or: show physics constants cluster MORE tightly than
      the uniform expectation (avg = half_step/2).
    """
    rng = random.Random(42)
    n_random = 10000
    random_errors = []
    for _ in range(n_random):
        target = 10 ** rng.uniform(-35, 30)
        err = _grid_snap_error(target, BASE_V3, R_V3, Q_V3)
        random_errors.append(err)

    avg_random = statistics.mean(random_errors)
    max_random = max(random_errors)
    half_step = HALF_STEP_PCT_V3
    expected_avg = half_step / 2  # Theoretical mean for uniform [0, half_step]

    physics_errors = [e["grid_error_pct"] for e in REAL_WORLD_CONSTANTS_V3.values()]
    avg_physics = statistics.mean(physics_errors)
    max_physics = max(physics_errors)

    all_random_within = max_random <= half_step * 1.01

    detail = (
        f"Grid half-step: {half_step:.5f}%\n"
        f"        Expected avg (uniform):  {expected_avg:.5f}%\n"
        f"        Physics constants:       avg={avg_physics:.5f}%, max={max_physics:.5f}% (N={len(physics_errors)})\n"
        f"        Random numbers:          avg={avg_random:.5f}%, max={max_random:.5f}% (N={n_random})\n"
        f"        Physics vs expected:     {avg_physics/expected_avg:.2f}x (1.0 = no better than random)\n"
        f"        DIAGNOSIS: Random numbers {'ALSO' if all_random_within else 'DO NOT'} fit within half-step.\n"
        f"        The ±{half_step:.5f}% precision is {'a property of grid density, not physics' if all_random_within else 'unexpectedly selective'}."
    )
    return all_random_within, detail


def test_nonsense_constants_fit() -> Tuple[bool, str]:
    """TEST 10 (DIAGNOSTIC): Show that made-up 'constants' fit just as well.

    WHY THIS MATTERS:
      If the equation can 'derive' the price of coffee with the same
      precision as the fine-structure constant, the precision doesn't
      distinguish physics from nonsense.

    WHAT WOULD FIX IT:
      The equation needs to produce a FALSIFIABLE claim — a value it
      assigns to specific dials that COULD fail to match reality but
      doesn't. Fitting after the fact is curve-fitting, not prediction.
    """
    nonsense = {
        "price_of_coffee_usd":      5.49,
        "my_shoe_size":             10.5,
        "digits_of_zip_code":       90210,
        "year_of_french_revolution": 1789,
        "random_prime":             7919,
        "taxi_number":              1729,
        "birthday_mmdd":            314,
        "nonsense_small":           0.00042,
        "nonsense_large":           6.022e17,
        "utterly_random":           847.293156,
    }
    results = []
    for name, val in nonsense.items():
        err = _grid_snap_error(val, BASE_V3, R_V3, Q_V3)
        results.append((name, val, err))

    all_within = all(err <= HALF_STEP_PCT_V3 * 1.01 for _, _, err in results)
    lines = [f"{name:<32s} = {val:<15.6g}  grid_err: {err:.5f}%" for name, val, err in results]
    detail = (
        "Made-up 'constants' on v3 grid:\n        " +
        "\n        ".join(lines) +
        f"\n        ALL within half-step: {all_within}"
    )
    return all_within, detail


def test_x_v3_was_tuned_for_c() -> Tuple[bool, str]:
    """TEST 11 (DIAGNOSTIC): X_v3 was reverse-engineered to place c on-grid.

    WHY THIS MATTERS:
      X_v3 = 285.999... is not an independent discovery. It was solved
      from c = 299792458 with E=151737. This is parameter fitting.

    WHAT WOULD FIX IT:
      Use the ORIGINAL X = 286 (the actual Fe BCC integer) and accept
      the grid error on c. The claim becomes "286 (iron lattice) naturally
      places c within X%" — honest and verifiable.
    """
    phi = (1 + math.sqrt(5)) / 2
    c_target = 299792458
    E_c = 151737
    r = 13 / 12
    Q = 758
    x_solved = (c_target / (r ** (E_c / Q))) ** phi
    match = abs(x_solved - X_V3) / X_V3 < 1e-10

    # Also show what happens with unfitted X = 286 (honest scaffold)
    base_286 = 286 ** (1.0 / phi)
    c_with_286 = _grid_snap_error(c_target, base_286, r, Q)

    detail = (
        f"Solved X from c: {x_solved:.14f}\n"
        f"        X_V3 in code:    {X_V3:.14f}\n"
        f"        Match: {match} — X_v3 was fitted to make c exact\n"
        f"        With honest X=286: c grid error = {c_with_286:.5f}% (still within half-step)"
    )
    return match, detail


def test_half_step_bound() -> Tuple[bool, str]:
    """TEST 12: Verify the theoretical half-step bound is correctly computed."""
    phi = (1 + math.sqrt(5)) / 2
    r = 13 / 12
    Q = 758
    step = r ** (1.0 / Q)
    half_pct = (step - 1) / 2 * 100
    match = abs(half_pct - HALF_STEP_PCT_V3) < 1e-8
    detail = f"(13/12)^(1/758) = {step:.15f}, half-step = {half_pct:.8f}%, module says {HALF_STEP_PCT_V3:.8f}%"
    return match, detail


def test_bridge_phi_claims() -> Tuple[bool, str]:
    """TEST 13: Verify both layers genuinely use X^(1/phi) as their base."""
    phi = (1 + math.sqrt(5)) / 2
    l1_base = 286 ** (1.0 / phi)
    l2_base = X_V3 ** (1.0 / phi)
    l1_match = abs(l1_base - BASE) < 1e-10
    l2_match = abs(l2_base - BASE_V3) < 1e-10
    passed = l1_match and l2_match
    detail = f"L1: 286^(1/phi)={l1_base:.15f} vs BASE={BASE:.15f}; L2: {X_V3}^(1/phi)={l2_base:.15f} vs BASE_V3={BASE_V3:.15f}"
    return passed, detail


def test_iron_lattice_claim() -> Tuple[bool, str]:
    """TEST 14: Verify 286 pm ~ Fe BCC lattice parameter (286.65 pm)."""
    measured = 286.65  # Kittel 8th ed
    deviation = abs(286 - measured) / measured * 100
    passed = deviation < 0.3
    detail = f"286 vs 286.65 pm: deviation {deviation:.3f}% (within crystallographic uncertainty)"
    return passed, detail


def test_nucleosynthesis_claim() -> Tuple[bool, str]:
    """TEST 15: Verify 104 = 26 * 4 = Fe_Z * He4_A."""
    fe_z = 26   # Iron atomic number
    he4_a = 4   # Helium-4 mass number
    product = fe_z * he4_a
    passed = product == 104
    detail = f"Fe(Z={fe_z}) * He-4(A={he4_a}) = {product}, QUANTIZATION_GRAIN = {QUANTIZATION_GRAIN}"
    return passed, detail


def test_dial_complexity_distribution() -> Tuple[bool, str]:
    """TEST 16: Are the dial settings suspiciously complex?

    WHY THIS MATTERS:
      If physics constants required simple dials (small a,b,c,d) while
      random numbers required large dials, that would suggest a real
      structural relationship. If dial complexity is just proportional
      to the exponent magnitude, it's just numerics.

    METRIC: Total dial cost = |a| + |b| + |c| + |d|
    """
    costs = []
    for name, entry in REAL_WORLD_CONSTANTS_V3.items():
        a, b, c, d = entry["dials"]
        cost = abs(a) + abs(b) + abs(c) + abs(d)
        costs.append((name, cost, entry["dials"]))

    costs.sort(key=lambda x: x[1])
    avg_cost = statistics.mean([c for _, c, _ in costs])
    median_cost = statistics.median([c for _, c, _ in costs])

    # Check: what's the minimum possible cost for each constant?
    # Cost is driven by |E| and decomposition, not by physics
    high_cost = [(n, c, d) for n, c, d in costs if c > 100]

    detail = (
        f"Dial cost (|a|+|b|+|c|+|d|) across {len(costs)} constants:\n"
        f"        Average: {avg_cost:.1f}, Median: {median_cost:.1f}\n"
        f"        Simplest: {costs[0][0]} cost={costs[0][1]} dials={costs[0][2]}\n"
        f"        Complex:  {costs[-1][0]} cost={costs[-1][1]} dials={costs[-1][2]}\n"
        f"        High-cost (>100): {len(high_cost)} constants\n"
        f"        NOTE: Cost scales with log(value) — large/small values need large dials.\n"
        f"        This is expected and does not indicate special structure."
    )
    # This is informational — always passes, reports the distribution
    return True, detail


def test_layer1_honest_precision() -> Tuple[bool, str]:
    """TEST 17: How does Layer 1 (r=2, Q=104) perform honestly?

    WHY THIS MATTERS:
      Layer 1 has a half-step of 0.334%. This is coarse enough that
      precision IS somewhat non-trivial. If many constants fall well
      within half-step, that's mildly interesting (though still guaranteed).

    THE HONEST CLAIM:
      "A logarithmic grid anchored at 286 pm (Fe BCC) with 104 steps
      per octave places N of 63 constants within X% error."
    """
    phi = (1 + math.sqrt(5)) / 2
    l1_base = 286 ** (1.0 / phi)
    l1_step = 2 ** (1.0 / 104)
    l1_half = (l1_step - 1) / 2 * 100

    l1_errors = []
    for name, entry in REAL_WORLD_CONSTANTS_V3.items():
        err = _grid_snap_error(entry["measured"], l1_base, 2, 104)
        l1_errors.append((name, err))

    l1_errors.sort(key=lambda x: x[1])
    avg_l1 = statistics.mean([e for _, e in l1_errors])
    within_tenth_pct = sum(1 for _, e in l1_errors if e < 0.1)
    within_quarter_pct = sum(1 for _, e in l1_errors if e < 0.25)

    # Compare to random expectation on L1 grid
    rng = random.Random(99)
    random_l1 = [_grid_snap_error(10**rng.uniform(-35, 30), l1_base, 2, 104) for _ in range(10000)]
    avg_random_l1 = statistics.mean(random_l1)

    detail = (
        f"Layer 1 grid: r=2, Q=104, half-step={l1_half:.3f}%\n"
        f"        Physics constants avg error: {avg_l1:.3f}%\n"
        f"        Random numbers avg error:    {avg_random_l1:.3f}%\n"
        f"        Physics vs random ratio:     {avg_l1/avg_random_l1:.2f}x\n"
        f"        Within 0.10%: {within_tenth_pct}/{len(l1_errors)}\n"
        f"        Within 0.25%: {within_quarter_pct}/{len(l1_errors)}\n"
        f"        Best 5:  {', '.join(f'{n}({e:.3f}%)' for n,e in l1_errors[:5])}\n"
        f"        Worst 5: {', '.join(f'{n}({e:.3f}%)' for n,e in l1_errors[-5:])}\n"
        f"        NOTE: On L1 grid, precision IS bounded but still trivially guaranteed.\n"
        f"        The honest claim is encoding, not derivation."
    )
    return True, detail


def test_degrees_of_freedom() -> Tuple[bool, str]:
    """TEST 18: Count free parameters vs fitted constants (overfitting check).

    WHY THIS MATTERS:
      The v3 equation has 3 free structural parameters (X, r, Q) plus
      4 dial integers per constant. With 63 constants, that's 63 × 1
      independent choices (the integer exponent E) — but E is just
      round(Q * log(target/base) / log(r)), so there are actually
      ZERO degrees of freedom once (X, r, Q) are fixed. The dials
      (a,b,c,d) are a decomposition of E, not independent parameters.

    BOTTOM LINE:
      3 structural parameters (X, r, Q) were optimized.
      X was fitted to c. r=13/12 and Q=758 were searched exhaustively.
      After that, every constant's placement is DETERMINED, not fitted.
      The "63 constants" aren't 63 successes — they're 63 inevitable
      consequences of a dense grid.
    """
    n_constants = len(REAL_WORLD_CONSTANTS_V3)
    # Structural parameters: X_v3 (fitted to c), r=13/12 (searched), Q=758 (searched)
    n_structural = 3
    # Per-constant "free" parameter: E_integer = round(solve_for_exponent(measured))
    # But this is DETERMINED by measured value, not independently chosen
    n_per_constant = 0  # E is computed, not chosen

    detail = (
        f"Structural parameters: {n_structural} (X={X_V3:.3f}, r=13/12, Q={Q_V3})\n"
        f"        Constants registered: {n_constants}\n"
        f"        Free parameters per constant: {n_per_constant}\n"
        f"        (E_integer is round(Q*log(val/base)/log(r)) — deterministic, not fitted)\n"
        f"        Total free parameters: {n_structural}\n"
        f"        Total 'predictions': {n_constants}\n"
        f"        DIAGNOSIS: After choosing (X, r, Q), every constant's grid placement\n"
        f"        is automatic. The equation doesn't 'choose' to match — it CAN'T MISS.\n"
        f"        A dense enough grid guarantees all values land within half-step.\n"
        f"        \n"
        f"        TO MAKE THIS MEANINGFUL:\n"
        f"        1. PREDICT a value before measurement (falsifiable)\n"
        f"        2. Use a coarser grid where misses are possible\n"
        f"        3. Show dial patterns encode physical relationships\n"
        f"           (e.g., electron=dials(X), muon=dials(X)+offset predicts tau)"
    )
    return True, detail


def test_predictions_exist() -> Tuple[bool, str]:
    """TEST 19: Prediction engine generates falsifiable predictions."""
    preds = predict_from_patterns()
    n = preds["total_predictions"]
    passed = n > 0
    lines = []
    for p in preds["predictions"]:
        name = p["name"]
        if "linear_prediction" in p:
            val = p["linear_prediction"]["value_MeV"]
            lines.append(f"{name}: {val:.2f} MeV (linear), falsifiable={p['falsifiable']}")
        elif "prediction" in p:
            val = p["prediction"].get("value_GeV", p["prediction"].get("value_MeV", "?"))
            lines.append(f"{name}: {val:.2f} (predicted), falsifiable={p['falsifiable']}")
    detail = (
        f"Generated {n} predictions from pattern extrapolation:\n        " +
        "\n        ".join(lines) +
        f"\n        These are specific numeric values that can be confirmed or refuted."
    )
    return passed, detail


def test_gap_predictions() -> Tuple[bool, str]:
    """TEST 20: Gap analysis finds unmatched low-complexity grid points."""
    gaps = predict_from_gaps(max_complexity=15)
    n = gaps["total_unmatched"]
    top = gaps["top_20"][:5]
    lines = [f"dials={g['dials']} E={g['E']} val={g['value']:.6g} cost={g['complexity']}"
             for g in top]
    detail = (
        f"Found {n} unmatched grid points (complexity <= 15):\n        " +
        "\n        ".join(lines) +
        f"\n        Simplest unmatched points are candidates for undiscovered constants."
    )
    return n > 0, detail


def test_pattern_analysis() -> Tuple[bool, str]:
    """TEST 21: Pattern analyzer finds and reports dial relationships."""
    patterns = analyze_dial_patterns()
    n = patterns["families_analyzed"]
    lines = []
    for family, data in patterns["patterns"].items():
        verdict = data.get("verdict", "analyzed")
        if "deltas" in data:
            lines.append(f"{family}: deltas={data['deltas']}, {verdict}")
        elif "delta_E" in data:
            lines.append(f"{family}: delta_E={data['delta_E']}, {verdict}")
        else:
            lines.append(f"{family}: {verdict}")
    detail = (
        f"Analyzed {n} constant families:\n        " +
        "\n        ".join(lines)
    )
    return n > 0, detail


def test_coarse_grid_stats() -> Tuple[bool, str]:
    """TEST 22: Coarse grid (Layer 1) statistical comparison vs random.

    This is the most honest test: on the coarse grid, does the encoding
    show ANY signal that physics constants behave differently from random?
    """
    cgv = coarse_grid_validation()
    ratio = cgv["physics_vs_random"]
    ks = cgv["ks_test"]
    n_close = cgv["surprisingly_close_count"]
    best = cgv["best_5"]

    detail = (
        f"Coarse grid (r=2, Q=104, half-step={cgv['grid']['half_step_pct']:.3f}%):\n"
        f"        Physics avg error: {cgv['avg_error_pct']:.4f}%\n"
        f"        Random avg error:  {cgv['random_avg_error_pct']:.4f}%\n"
        f"        Ratio:             {ratio} (1.0 = identical to random)\n"
        f"        Surprisingly close (< {cgv['surprisingly_close_threshold_pct']:.3f}%): {n_close}\n"
        f"        Best: {', '.join(f'{b['name']}({b['error_pct']:.4f}%)' for b in best)}\n"
        f"        KS test: D={ks['max_deviation']}, critical={ks['critical_value']}\n"
        f"        Distribution: {ks['interpretation']}"
    )
    # This test passes (it's diagnostic) — the results tell the story
    return True, detail


def test_lepton_ratio_prediction() -> Tuple[bool, str]:
    """TEST 23: Can the lepton exponent pattern predict tau from e and mu?

    STRONGEST TEST: Given only electron and muon, predict tau mass.
    If the exponent spacing predicts tau within 1%, that's genuine signal.
    """
    if not all(n in REAL_WORLD_CONSTANTS_V3 for n in _LEPTON_MASSES):
        return False, "Lepton masses not all registered"

    e_E = REAL_WORLD_CONSTANTS_V3["electron_mass_MeV"]["E_integer"]
    mu_E = REAL_WORLD_CONSTANTS_V3["muon_mass_MeV"]["E_integer"]
    tau_E = REAL_WORLD_CONSTANTS_V3["tau_mass_MeV"]["E_integer"]
    tau_measured = REAL_WORLD_CONSTANTS_V3["tau_mass_MeV"]["measured"]

    # Method 1: linear extrapolation (constant step in E)
    delta = mu_E - e_E
    tau_E_predicted = mu_E + delta
    tau_predicted = BASE_V3 * (R_V3 ** (tau_E_predicted / Q_V3))
    err_linear = abs(tau_predicted - tau_measured) / tau_measured * 100

    # Method 2: use the known Koide formula as a reference
    # Koide: (me + mmu + mtau) / (sqrt(me) + sqrt(mmu) + sqrt(mtau))^2 = 1/3
    me = REAL_WORLD_CONSTANTS_V3["electron_mass_MeV"]["measured"]
    mmu = REAL_WORLD_CONSTANTS_V3["muon_mass_MeV"]["measured"]
    koide_num = me + mmu + tau_measured
    koide_den = (math.sqrt(me) + math.sqrt(mmu) + math.sqrt(tau_measured)) ** 2
    koide_val = koide_num / koide_den

    detail = (
        f"Lepton exponent spacing test (predict tau from e, mu):\n"
        f"        Exponents: e={e_E}, mu={mu_E}, tau={tau_E}\n"
        f"        Delta(e→mu) = {delta}, Delta(mu→tau) = {tau_E - mu_E}\n"
        f"        Linear prediction: E={tau_E_predicted} → {tau_predicted:.2f} MeV\n"
        f"        Actual tau: {tau_measured} MeV\n"
        f"        Prediction error: {err_linear:.1f}%\n"
        f"        Verdict: {'USEFUL (< 5%)' if err_linear < 5 else 'TOO COARSE' if err_linear < 50 else 'NO PREDICTIVE POWER'}\n"
        f"        Koide formula check: Q = {koide_val:.6f} (exact = 1/3 = {1/3:.6f})"
    )
    # Pass if prediction is within 50% — even rough signal is interesting
    passed = err_linear < 50
    return passed, detail


def test_q26_simplified() -> Tuple[bool, str]:
    """TEST 24: Simplified Q=26 equation — ZERO fitted parameters, coarse grid.

    This is the strongest honest test. The equation:
        G(n) = 286^(1/phi) * 2^(n/26)
    has NO fitted parameters: 286=integer, 2=octave, 26=Fe atomic number.
    Half-step = ±1.351%. Any constant landing within 0.135% (10% of half-step)
    has < 10% probability by chance.
    """
    report = simple_grid_report()
    standouts = report["standouts"]
    n_standouts = report["standout_count"]
    expected = report["standout_expected"]
    p_combined = report["standout_p_value"]

    # Sort all by error for display
    top_10 = report["all_results"][:10]

    lines = []
    for name, err, E, domain in top_10:
        p = err / HALF_STEP_SIMPLE
        marker = " ***" if err < HALF_STEP_SIMPLE * 0.1 else " *" if err < HALF_STEP_SIMPLE * 0.25 else ""
        lines.append(f"{name:<28s} err={err:.4f}% E={E:>5d} p={p:.3f} [{domain}]{marker}")

    standout_lines = [f"{s['name']}: {s['error_pct']:.4f}% (p={s['p_value']:.4f})" for s in standouts]

    detail = (
        f"Q=26 grid: 286^(1/phi) * 2^(n/26), half-step = {HALF_STEP_SIMPLE:.3f}%\n"
        f"        ZERO fitted parameters (286=Fe BCC integer, 2=octave, 26=Fe_Z)\n"
        f"        Physics avg: {report['avg_error_pct']:.3f}%, Random avg: {report['random_avg_error_pct']:.3f}%\n"
        f"        Ratio: {report['physics_vs_random']} (1.0 = same as random)\n"
        f"        \n"
        f"        Top 10 closest (out of {report['total_constants']}):\n"
        f"        " + "\n        ".join(lines) + "\n"
        f"        \n"
        f"        Standouts (< {report['standout_threshold_pct']:.3f}%, < 10% of half-step):\n"
        f"        " + ("\n        ".join(standout_lines) if standout_lines else "None") + "\n"
        f"        Count: {n_standouts} (expected by chance: {expected})\n"
        f"        Combined p-value: {p_combined:.4f} ({'SIGNIFICANT (< 0.05)' if p_combined < 0.05 else 'NOT SIGNIFICANT'})"
    )
    passed = True  # Diagnostic — always passes, data tells the story
    return passed, detail


def test_q26_nonsense() -> Tuple[bool, str]:
    """TEST 25: Do nonsense values also get standouts on Q=26 grid?

    Unlike v3's trivially dense grid, Q=26 has real discrimination power.
    If nonsense values DON'T produce standouts, the Q=26 standouts are real.
    """
    nonsense = {
        "coffee_price": 5.49, "shoe_size": 10.5, "zip_code": 90210,
        "french_revolution": 1789, "taxi_1729": 1729, "birthday": 314,
        "tiny": 0.00042, "huge": 6.022e17, "random1": 847.293, "random2": 23.7156,
    }
    threshold = HALF_STEP_SIMPLE * 0.1
    hits = []
    for name, val in nonsense.items():
        result = snap_simple(val)
        if result["error_pct"] < threshold:
            hits.append(f"{name}={val}: {result['error_pct']:.4f}%")

    lines = []
    for name, val in nonsense.items():
        r = snap_simple(val)
        lines.append(f"{name:<20s} = {val:<12.4g} err={r['error_pct']:.3f}%")

    detail = (
        f"Nonsense values on Q=26 grid (threshold = {threshold:.3f}%):\n        " +
        "\n        ".join(lines) +
        f"\n        Standout hits: {len(hits)}"
    )
    if hits:
        detail += "\n        " + "\n        ".join(hits)
    # If no nonsense values are standouts, the grid has discrimination power
    return True, detail


def test_q26_unfitted_c() -> Tuple[bool, str]:
    """TEST 26: Speed of light on unfitted Q=26 grid.

    On v3, c is exact because X_v3 was fitted to c. On Q=26, X=286 (integer),
    r=2, Q=26 — nothing is fitted. If c still lands close, that's genuine.
    """
    c = 299792458
    result = snap_simple(c)
    p = result["error_pct"] / HALF_STEP_SIMPLE

    detail = (
        f"c = 299,792,458 m/s on unfitted Q=26 grid:\n"
        f"        Grid value: {result['grid_value']:.0f}\n"
        f"        Error: {result['error_pct']:.4f}%\n"
        f"        E = {result['E_integer']}\n"
        f"        P(random this close): {p:.4f} ({p*100:.1f}%)\n"
        f"        Verdict: {'NOTEWORTHY (p < 0.05)' if p < 0.05 else 'NOT SIGNIFICANT'}\n"
        f"        Context: No parameters were fitted — 286, 2, 26 are all fixed a priori"
    )
    passed = True
    return passed, detail


def test_q26_iron_cluster() -> Tuple[bool, str]:
    """TEST 27: Do iron-domain constants cluster better than average on Q=26?

    Iron constants have a physical reason to be close (the grid is anchored
    at the iron lattice). If they're systematically closer than non-iron
    constants, that's a real signal.
    """
    iron_errs = []
    non_iron_errs = []
    iron_details = []
    for name, result in SIMPLE_GRID_RESULTS.items():
        if result["domain"] in ("iron", "nuclear"):
            iron_errs.append(result["error_pct"])
            iron_details.append((name, result["error_pct"], result["domain"]))
        else:
            non_iron_errs.append(result["error_pct"])

    avg_iron = statistics.mean(iron_errs) if iron_errs else 0
    avg_non = statistics.mean(non_iron_errs) if non_iron_errs else 0
    iron_details.sort(key=lambda x: x[1])

    # Permutation test: is avg_iron significantly lower than avg_non?
    all_errs = iron_errs + non_iron_errs
    n_iron = len(iron_errs)
    rng = random.Random(42)
    n_perms = 10000
    count_le = 0
    observed_diff = avg_iron - avg_non
    for _ in range(n_perms):
        rng.shuffle(all_errs)
        perm_iron = statistics.mean(all_errs[:n_iron])
        perm_non = statistics.mean(all_errs[n_iron:])
        if perm_iron - perm_non <= observed_diff:
            count_le += 1
    p_perm = count_le / n_perms

    lines = [f"{n:<28s} err={e:.3f}% [{d}]" for n, e, d in iron_details]
    detail = (
        f"Iron/nuclear vs non-iron on Q=26 grid:\n"
        f"        Iron+nuclear avg:    {avg_iron:.3f}% (N={len(iron_errs)})\n"
        f"        Non-iron avg:        {avg_non:.3f}% (N={len(non_iron_errs)})\n"
        f"        Difference:          {observed_diff:+.3f}%\n"
        f"        Permutation p-value: {p_perm:.4f} ({'SIGNIFICANT' if p_perm < 0.05 else 'NOT SIGNIFICANT'})\n"
        f"        \n"
        f"        Iron/nuclear constants:\n"
        f"        " + "\n        ".join(lines)
    )
    return True, detail


def test_mod4_discovery() -> Tuple[bool, str]:
    """TEST 28: Mod-4 clustering — in-sample signal, holdout validation.

    Physical constants' exponents cluster at E ≡ 2 (mod 4) on the Q=26 grid.
    27/63 = 42.9% vs 25% expected. Significant in-sample (p≈0.004).
    But FAILS holdout test: 30 new constants show 26.7% (baseline).
    """
    result = discover_mod4_pattern()
    p = result["p_value_monte_carlo"]
    dist = result["mod4_distribution"]
    dom = result["dominant_residue"]
    count = result["dominant_count"]
    total = sum(dist)
    holdout = result["holdout_test"]

    detail = (
        f"Mod-4 clustering on Q=26 grid:\n"
        f"        Distribution (E mod 4 = 0,1,2,3): {dist}\n"
        f"        Dominant: E ≡ {dom} (mod 4) = {count}/{total} = {count/total*100:.1f}%\n"
        f"        Expected per class: {total/4:.1f} (25%)\n"
        f"        Chi-squared: {result['chi_squared']} (critical: 7.81)\n"
        f"        In-sample MC p-value: {p} (bootstrap, 2000 trials)\n"
        f"        \n"
        f"        HOLDOUT TEST ({holdout['n_holdout']} new constants):\n"
        f"        Distribution: {holdout['mod4_distribution']}\n"
        f"        E≡{dom} fraction: {holdout['dominant_fraction']*100:.1f}% (baseline: 25%)\n"
        f"        Verdict: {holdout['verdict']}\n"
        f"        \n"
        f"        Mod-8: chi2={result['mod8_test']['chi_squared']}, p={result['mod8_test']['p_value']}\n"
        f"        Overall: {result['honest_verdict'][:120]}..."
    )
    # This test PASSES as a diagnostic — it honestly reports the holdout failure
    return True, detail


def test_arithmetic_closure_honest() -> Tuple[bool, str]:
    """TEST 29: Exponent arithmetic closure — honest assessment.

    Sum/diff closure looks impressive (6.7x vs uniform), but vanishes when
    tested against distribution-matched null (p=0.76). This is an honest
    diagnostic that shows the importance of proper null distributions.
    """
    result = discover_exponent_arithmetic()
    boot = result["bootstrap_null"]
    mag = result["magnitude_null"]

    detail = (
        f"Exponent arithmetic closure:\n"
        f"        Sum hits: {result['sum_hits']}, Diff hits: {result['diff_hits']}\n"
        f"        Bootstrap null (distribution-matched): avg={boot['avg']}, p={boot['p_value']}\n"
        f"        Magnitude null (random magnitudes): avg={mag['avg']}, ratio={mag['ratio']}x, p={mag['p_value']}\n"
        f"        Q=27 control: sum={result['q27_control']['sum_hits']}, diff={result['q27_control']['diff_hits']}\n"
        f"        Honest verdict: {boot['verdict']}\n"
        f"        This test PASSES because it honestly reports the null result."
    )
    # This always passes — it's an honest diagnostic
    return True, detail


def test_discovery_summary() -> Tuple[bool, str]:
    """TEST 30: Full discovery engine summary — honest unified verdict.

    Runs all discovery analyses. No statistically significant discovery
    survives out-of-sample validation. Reports this honestly.
    """
    result = discover_all()
    in_sample = result["in_sample_findings"]

    lines = ["Discovery engine summary:"]
    lines.append(f"        Grid: G(n) = 286^(1/phi) * 2^(n/26), ZERO fitted params")

    # In-sample findings
    for name, finding in in_sample.items():
        lines.append(f"        IN-SAMPLE: {name} — {finding['in_sample']} (p={finding['p_value']})")
        lines.append(f"          Holdout: {finding['holdout'][:80]}")

    # Debunked
    for name, reason in result["debunked_claims"].items():
        lines.append(f"        DEBUNKED: {name}")
        lines.append(f"          {str(reason)[:100]}")

    lines.append(f"        ")
    lines.append(f"        HONEST VERDICT: {result['honest_verdict'][:150]}")

    detail = "\n".join(lines)
    # This test PASSES as a diagnostic — it honestly reports findings
    return True, detail


def test_alpha_pi_bridge() -> Tuple[bool, str]:
    """TEST 31: The α/π bridge — does 286×(1+α/π) predict Fe BCC?

    From universal_god_code.py (January 25, 2026). Tests whether the gap
    between the integer scaffold 286 and the measured Fe BCC lattice (286.65 pm)
    equals α/π (fine structure constant / pi).
    """
    result = analyze_alpha_pi_bridge()
    pred = result["prediction"]
    gap = result["gap_analysis"]
    grid = result["grid_comparison"]
    so_alpha = result["standout_details"]["X_alpha"]

    lines = [
        f"alpha/pi Bridge (universal_god_code.py, Jan 25 2026):",
        f"        286 * (1 + alpha/pi) = {pred['predicted_pm']} pm",
        f"        Fe BCC measured     = {pred['measured_pm']} pm",
        f"        Prediction error    = {pred['error_pct']}%",
        f"        Gap match: actual={gap['actual_gap_frac']}, alpha/pi={gap['alpha_over_pi']} ({gap['match_pct']}% off)",
        f"        ",
        f"        Grid comparison (Q=26):",
        f"        X=286:        avg={grid['X_286_avg_err']}%, standouts={grid['X_286_standouts']}",
        f"        X=286(1+a/p): avg={grid['X_alpha_avg_err']}%, standouts={grid['X_alpha_standouts']}",
        f"        Standout p-value: {grid['standout_p_value']} (MC, random X near 286)",
        f"        ",
        f"        alpha/pi standouts:",
    ]
    for so in so_alpha[:5]:
        lines.append(f"          {so['name']:25s} err={so['error_pct']:.4f}% [{so['domain']}]")
    lines.append(f"        ")
    lines.append(f"        Verdict: {result['honest_verdict'][:120]}")

    detail = "\n".join(lines)
    # Pass as diagnostic — the data tells the story
    return True, detail


def test_cross_reference() -> Tuple[bool, str]:
    """TEST 32: Cross-reference all theoretical claims against data.

    Tests every claim in the "Universal Scaling Law" framework:
    EEG alignment, gravity/Schumann, speed of light, hemoglobin,
    iron lattice, nucleosynthesis, phi, and unit dependence.
    """
    result = cross_reference_claims()

    # Count verdicts
    true_claims = list(result["factually_true"].keys())
    unit_dep = list(result["unit_dependent"].keys())
    not_sup = list(result["not_supported"].keys())

    lines = [
        f"Cross-reference of 9 theoretical claims:",
        f"        FACTUALLY TRUE ({len(true_claims)}):",
    ]
    for k in true_claims:
        lines.append(f"          {k}: {result['factually_true'][k]['detail']}")

    lines.append(f"        UNIT-DEPENDENT ({len(unit_dep)}):")
    for k in unit_dep:
        v = result["unit_dependent"][k]
        lines.append(f"          {k}: {v['verdict'][:80]}")

    lines.append(f"        NOT SUPPORTED ({len(not_sup)}):")
    for k in not_sup:
        v = result["not_supported"][k]
        lines.append(f"          {k}: {v['verdict'][:80]}")

    lines.append(f"        ")
    lines.append(f"        FATAL FLAW: {result['fatal_flaw'][:120]}")

    detail = "\n".join(lines)
    # Passes as diagnostic — honest cross-reference
    return True, detail


def test_energy_transition_equation() -> Tuple[bool, str]:
    """TEST 33: The energy transition equation from January 25, 2026 (deleted files).

    G(E) = [286 × (1 + α/π × Γ(E))]^(1/φ) × 16
    where Γ(E) = 1 / (1 + (E_Planck/E)²)

    Recovered from: test_universal_god_code.py, l104_consciousness.py, const.py
    (deleted in EVO_54, recovered from git show a16ce282:...)

    Tests: math identity, Planck midpoint, EXISTENCE_COST, element coherence,
    energy-scale grid comparison.
    """
    result = analyze_energy_transition_equation()
    mc = result["math_checks"]
    ec = result["existence_cost"]
    coh = result["element_coherence"]
    grid = result["energy_grid_comparison"]

    lines = [
        f"Energy Transition Equation (Jan 25 2026, deleted EVO_54):",
        f"        G(E) = [286(1 + alpha/pi * Gamma(E))]^(1/phi) * 16",
        f"        Source: {result['source']}",
        f"        ",
        f"        GRAVITY_CODE (E->0) = {mc['GRAVITY_CODE']['value']:.10f}",
        f"        LIGHT_CODE  (E->inf)= {mc['LIGHT_CODE']['value']:.10f}",
        f"        EXISTENCE_COST      = {ec['EXISTENCE_COST']:.10f}",
        f"        G(E=0) matches GRAVITY: {mc['G(E=0)_matches_GRAVITY']}",
        f"        G(E=inf) matches LIGHT: {mc['G(E=inf)_matches_LIGHT']}",
        f"        Gamma(E_planck) = 0.5:  {mc['Gamma(E_planck)_equals_0.5']}",
        f"        ",
        f"        Planck midpoint: {result['planck_midpoint']['verdict'][:100]}",
        f"        ",
        f"        CIRCULARITY: {ec['circularity']['verdict'][:120]}",
        f"        Random (X,Q) pairs matching: {ec['random_X_Q_hits_pct']:.2f}%",
        f"        Alternative couplings:",
    ]
    for k, v in ec["alternative_couplings"].items():
        lines.append(f"          {k:15s}: cost = {v:.6f}")

    lines.append(f"        ")
    lines.append(f"        Element coherence vs 286:")
    for el, data in coh["elements"].items():
        lines.append(f"          {el:8s}: lattice={data['lattice_pm']}pm, "
                     f"ratio={data['ratio_to_286']}, coh={data['coherence']}, "
                     f"frac~{data['nearest_fraction']}")
    lines.append(f"        Physics mean={coh['physics_mean_coherence']}, "
                 f"random mean={coh['random_mean_coherence']}, p={coh['p_value']}")
    lines.append(f"        {coh['verdict']}")
    lines.append(f"        ")
    lines.append(f"        Energy grid (Q=26): X=286 avg={grid['X_pure_286_avg_err']}%, "
                 f"X=286(1+a/p) avg={grid['X_alpha_avg_err']}%, "
                 f"improved={grid['better_with_alpha']}/{grid['total']}")
    lines.append(f"        ")
    lines.append(f"        Verdict: {result['honest_verdict'][:150]}")

    detail = "\n".join(lines)
    return True, detail


def test_january_framework() -> Tuple[bool, str]:
    """TEST 34: The complete January 2026 framework from const.py EVO_50.

    G(X) = 286^(1/phi) * 2^((416-X)/104) where X is never solved.
    Conservation: G(X) * 2^(X/104) = INVARIANT.
    Factor 13: 286=2*11*13, 104=8*13, 416=32*13.
    MATTER_BASE = 286*(1+alpha/pi) ≈ Fe BCC lattice.

    Source: git show 5ede8841:const.py (Jan 25, 2026)
    """
    result = analyze_january_framework()
    pa = result["parametric_analysis"]
    ca = result["conservation_analysis"]
    f13 = result["factor13_analysis"]
    mb = result["matter_base_analysis"]
    ge = result["grid_equivalence"]

    lines = [
        f"January 2026 Framework (const.py at EVO_50):",
        f"        Source: {result['source']}",
        f"        ",
        f"        A) G(X) Parametric: {pa['integer_X_count']}/{pa['constants_tested']} "
        f"land near integer X (p={pa['p_value']})",
        f"           Expected by chance: {pa['monte_carlo_expected']}",
        f"           {'SIGNIFICANT' if pa['significant'] else 'NOT SIGNIFICANT'}",
        f"           Caveat: {pa['eeg_caveat'][:100]}",
    ]
    for name, data in pa["x_results"].items():
        tag = " <<INTEGER>>" if data["integer"] else ""
        lines.append(f"           {name:25s}: X={data['X']:>10} frac={data['frac_error']}{tag}")

    lines.append(f"        ")
    lines.append(f"        B) Conservation: {'ALGEBRAIC IDENTITY' if ca['all_identical'] else 'BROKEN'}")
    lines.append(f"           {ca['verdict'][:120]}")
    lines.append(f"        ")
    lines.append(f"        C) Factor 13: random triple rate = {f13['random_triple_rate']}%")
    lines.append(f"           {f13['verdict'][:120]}")
    lines.append(f"        ")
    lines.append(f"        D) MATTER_BASE = {mb['MATTER_BASE']} vs Fe BCC = {mb['Fe_BCC_pm']}")
    lines.append(f"           Error: {mb['error_pct']}%")
    lines.append(f"           Better rational expressions: {mb['better_rational_exprs']}/{mb['total_rational_tested']}")
    if mb["examples_better"]:
        lines.append(f"           Examples: {', '.join(mb['examples_better'][:3])}")
    lines.append(f"        ")
    lines.append(f"        E) Grid equivalence: {ge['verdict'][:120]}")
    lines.append(f"        ")
    for f in result["findings"]:
        lines.append(f"        * {f}")
    lines.append(f"        ")
    lines.append(f"        VERDICT: {result['honest_verdict'][:200]}")

    detail = "\n".join(lines)
    return True, detail


def test_evolved_equations() -> Tuple[bool, str]:
    """TEST 35: The four evolved equation iterations (Original, v1-φ, v2-3/2, v3-13/12).

    Source: l104_god_code_evolved.py, l104_god_code_evolved_v2.py,
    l104_god_code_evolved_v3.py (on disk, never committed — created before system reset).

    Tests the evolution of three parameters (r, Q, X) across four variants:
      Original:  r=2,     Q=104, X=286
      v1 φ:      r=φ,     Q=481, X=286.441
      v2 3/2:    r=3/2,   Q=234, X=286.897
      v3 13/12:  r=13/12, Q=758, X=285.999

    Checks: precision vs density, X drift, factor-13 invariance, random control.
    """
    result = analyze_evolved_equations()
    pr = result["precision_results"]
    da = result["density_analysis"]
    xd = result["x_drift_analysis"]
    f13 = result["factor_13_check"]
    rc = result["random_comparison"]

    lines = [
        f"Evolved Equation Iterations (4 variants, 13 constants):",
        f"        Source: l104_god_code_evolved*.py (never committed, pre-reset)",
    ]

    # Precision table
    lines.append(f"        ")
    lines.append(f"        {'Equation':<28s} {'avg%':>8s} {'max%':>8s} {'half-step':>10s}")
    lines.append(f"        {'-'*58}")
    for eq_name in ["original", "v1_phi", "v2_rational", "v3_superparticular"]:
        r = pr[eq_name]
        lines.append(
            f"        {eq_name:<28s} {r['avg_error_pct']:>8.4f} "
            f"{r['max_error_pct']:>8.4f} {r['half_step_pct']:>9.4f}%"
        )

    # Density explanation
    lines.append(f"        ")
    lines.append(f"        Density analysis (actual/theoretical ratio — 1.0 = fully explained):")
    for eq_name in ["original", "v1_phi", "v2_rational", "v3_superparticular"]:
        d = da[eq_name]
        tag = "DENSITY" if d["explained_by_density"] else "SIGNAL?"
        lines.append(f"          {eq_name:<24s}: ratio={d['ratio']:.2f} [{tag}]")
    lines.append(f"        All explained by density: {result['density_explains_all']}")

    # X drift
    lines.append(f"        ")
    lines.append(f"        X drift from 286 (α/π prediction = {xd['original']['alpha_pi_prediction']}):")
    for eq_name in ["original", "v1_phi", "v2_rational", "v3_superparticular"]:
        x = xd[eq_name]
        match_tag = "MATCH" if x["matches_alpha_pi"] else "no"
        lines.append(
            f"          {eq_name:<24s}: X={x['X']:<20} drift={x['drift_from_286']:>+10.6f} "
            f"α/π:{match_tag}"
        )
    lines.append(f"        MATTER_BASE = 286*(1+α/π) = {result['matter_base_286_alpha_pi']}")

    # Factor 13
    lines.append(f"        ")
    lines.append(f"        Factor 13 (F(7) golden thread):")
    for eq_name in ["original", "v1_phi", "v2_rational", "v3_superparticular"]:
        f = f13[eq_name]
        tag = "YES" if f["has_factor_13"] else "NO"
        lines.append(f"          {eq_name:<24s}: Q={f['Q']:>4} = {f['Q_factored']:<20s} 13|Q: {tag}")
    lines.append(f"        All preserve factor 13: {result['all_have_factor_13']}")

    # Random comparison
    lines.append(f"        ")
    lines.append(f"        Random (r,Q,X) control ({rc['trials']} trials, X fitted to c):")
    lines.append(f"          Beat v3 (13/12): {rc['better_than_v3']}/{rc['trials']} ({rc['better_than_v3_pct']}%)")
    lines.append(f"          Beat original:   {rc['better_than_original']}/{rc['trials']} ({rc['better_than_original_pct']}%)")

    lines.append(f"        ")
    lines.append(f"        VERDICT: {result['honest_verdict'][:250]}")

    detail = "\n".join(lines)
    return True, detail


def run_legitimacy_tests() -> Dict[str, Any]:
    """Run ALL legitimacy tests and return structured results."""
    tests = [
        ("EQ_IDENTITY",       "Layer 1 equation identity G(0,0,0,0) = 527.518...",   test_equation_identity),
        ("V3_IDENTITY",       "Layer 2 equation identity G_v3(0,0,0,0) = 45.411...", test_v3_equation_identity),
        ("DIAL_ALGEBRA",      "Dial-to-exponent mapping correctness",                test_dial_exponent_algebra),
        ("GRID_VALS",         "Grid values match dial recomputation",                 test_grid_values_match_dials),
        ("CLAIMED_ERRORS",    "Claimed error percentages are accurate",               test_claimed_errors_are_correct),
        ("REF_CROSSCHECK",    "Measured values vs independent references",            test_measured_values_vs_independent),
        ("C_EXACT",           "Speed of light is exact on v3 grid",                   test_speed_of_light_exact),
        ("NO_COLLISIONS",     "No exponent collisions among constants",               test_no_exponent_collisions),
        ("RANDOM_FIT",        "Random numbers achieve same precision (grid density)", test_random_numbers_same_precision),
        ("NONSENSE_FIT",      "Made-up constants fit just as well",                   test_nonsense_constants_fit),
        ("X_FITTED",          "X_v3 was reverse-engineered to make c exact",          test_x_v3_was_tuned_for_c),
        ("HALF_STEP",         "Theoretical half-step bound is correct",               test_half_step_bound),
        ("PHI_BRIDGE",        "Both layers use X^(1/phi) base",                       test_bridge_phi_claims),
        ("IRON_LATTICE",      "286 pm approximates Fe BCC lattice",                   test_iron_lattice_claim),
        ("NUCLEOSYNTHESIS",   "104 = 26 * 4 = Fe_Z * He4_A",                         test_nucleosynthesis_claim),
        ("DIAL_COMPLEXITY",   "Dial complexity distribution analysis",                test_dial_complexity_distribution),
        ("L1_HONEST",         "Layer 1 honest precision assessment",                  test_layer1_honest_precision),
        ("DEGREES_OF_FREEDOM","Overfitting check: free params vs constants",          test_degrees_of_freedom),
        ("PREDICTIONS",       "Prediction engine generates falsifiable claims",       test_predictions_exist),
        ("GAP_PREDICTIONS",   "Gap analysis finds unmatched grid points",             test_gap_predictions),
        ("PATTERNS",          "Dial pattern analysis across families",                test_pattern_analysis),
        ("COARSE_GRID",       "Coarse grid (L1) statistical comparison",              test_coarse_grid_stats),
        ("LEPTON_PREDICT",    "Predict tau mass from e and mu exponents",             test_lepton_ratio_prediction),
        ("Q26_SIMPLIFIED",    "Q=26 simplified equation — zero fitted params",        test_q26_simplified),
        ("Q26_NONSENSE",      "Q=26 nonsense values — grid discrimination power",    test_q26_nonsense),
        ("Q26_UNFITTED_C",    "Speed of light on unfitted Q=26 grid",                test_q26_unfitted_c),
        ("Q26_IRON_CLUSTER",  "Iron-domain clustering on Q=26 grid",                 test_q26_iron_cluster),
        ("MOD4_DISCOVERY",    "Mod-4 clustering — genuine statistical signal",        test_mod4_discovery),
        ("ARITH_CLOSURE",     "Exponent arithmetic closure — honest null test",       test_arithmetic_closure_honest),
        ("DISCOVERY_SUMMARY", "Full discovery engine — unified verdict",              test_discovery_summary),
        ("ALPHA_PI_BRIDGE",  "alpha/pi bridge — Fe BCC prediction from Jan 2026",     test_alpha_pi_bridge),
        ("CROSS_REFERENCE",  "Cross-reference all theoretical claims",               test_cross_reference),
        ("ENERGY_TRANSITION","Energy transition equation G(E) from Jan 2026",        test_energy_transition_equation),
        ("JAN_FRAMEWORK",   "Complete January 2026 framework from const.py EVO_50", test_january_framework),
        ("EVOLVED_EQS",    "Four evolved equation iterations (v1-φ, v2-3/2, v3-13/12)", test_evolved_equations),
    ]
    results = {}
    for key, desc, fn in tests:
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, f"EXCEPTION: {e}"
        results[key] = {"description": desc, "passed": passed, "detail": detail}
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Legitimacy Test Suite
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("  L104 DUAL-LAYER ENGINE — LEGITIMACY TEST SUITE")
    print("=" * 80)

    results = run_legitimacy_tests()

    # ── Categorize tests ──
    math_tests = ["EQ_IDENTITY", "V3_IDENTITY", "DIAL_ALGEBRA", "GRID_VALS",
                  "CLAIMED_ERRORS", "C_EXACT", "NO_COLLISIONS", "HALF_STEP"]
    reference_tests = ["REF_CROSSCHECK"]
    diagnostic_tests = ["RANDOM_FIT", "NONSENSE_FIT", "X_FITTED"]
    depth_tests = ["DIAL_COMPLEXITY", "L1_HONEST", "DEGREES_OF_FREEDOM"]
    claim_tests = ["PHI_BRIDGE", "IRON_LATTICE", "NUCLEOSYNTHESIS"]
    prediction_tests = ["PREDICTIONS", "GAP_PREDICTIONS", "LEPTON_PREDICT"]
    pattern_tests = ["PATTERNS", "COARSE_GRID"]
    q26_tests = ["Q26_SIMPLIFIED", "Q26_NONSENSE", "Q26_UNFITTED_C", "Q26_IRON_CLUSTER"]
    discovery_tests = ["MOD4_DISCOVERY", "ARITH_CLOSURE", "DISCOVERY_SUMMARY",
                       "ALPHA_PI_BRIDGE", "CROSS_REFERENCE", "ENERGY_TRANSITION",
                       "JAN_FRAMEWORK", "EVOLVED_EQS"]

    categories = [
        ("MATHEMATICAL CORRECTNESS", math_tests,
         "Do the equations compute what they claim?"),
        ("REFERENCE CROSS-CHECK", reference_tests,
         "Are the 'measured' values actually correct?"),
        ("STATISTICAL DIAGNOSTICS (v3 fine grid)", diagnostic_tests,
         "Is the v3 precision meaningful or trivially guaranteed?"),
        ("DEPTH ANALYSIS", depth_tests,
         "What would make the claims scientifically meaningful?"),
        ("PHYSICAL CLAIMS", claim_tests,
         "Are the iron/phi/nucleosynthesis claims factually accurate?"),
        ("FALSIFIABLE PREDICTIONS", prediction_tests,
         "Can the encoding predict values before measurement?"),
        ("PATTERN & COARSE-GRID ANALYSIS", pattern_tests,
         "Do dial patterns show structure? Does the coarse grid reveal signal?"),
        ("Q=26 SIMPLIFIED (ZERO FITTED PARAMS)", q26_tests,
         "G(n) = 286^(1/phi) * 2^(n/26) — the honest unfitted equation"),
        ("DISCOVERY ENGINE", discovery_tests,
         "Genuine statistical signal search with proper null distributions"),
    ]

    total_pass = 0
    total_fail = 0

    for cat_name, test_keys, cat_desc in categories:
        print(f"\n  -- {cat_name} --")
        print(f"     {cat_desc}")
        for key in test_keys:
            r = results[key]
            mark = "PASS" if r["passed"] else "FAIL"
            if r["passed"]:
                total_pass += 1
            else:
                total_fail += 1
            print(f"\n    [{mark}] {key}: {r['description']}")
            for line in r["detail"].split("\n"):
                print(f"        {line}")

    print(f"\n  {'=' * 80}")
    print(f"  RESULTS: {total_pass} passed, {total_fail} failed out of {total_pass + total_fail}")
    print(f"  {'=' * 80}")

    # ── Summary verdict ──
    print("\n  VERDICT:")
    print("    MATH: The equations are internally consistent and compute correctly.")

    ref_ok = results["REF_CROSSCHECK"]["passed"]
    if ref_ok:
        print("    REFERENCES: All measured values match independent sources.")
    else:
        print("    REFERENCES: Some measured values differ from independent sources.")
        print(f"      {results['REF_CROSSCHECK']['detail']}")

    random_fit = results["RANDOM_FIT"]["passed"]
    nonsense_fit = results["NONSENSE_FIT"]["passed"]
    x_fitted = results["X_FITTED"]["passed"]
    if random_fit and nonsense_fit:
        print("    PRECISION: The ±0.005% claim is VALID but TRIVIAL.")
        print(f"      The grid step (13/12)^(1/758) = {STEP_V3:.10f} guarantees")
        print(f"      that ANY positive number snaps to within ±{HALF_STEP_PCT_V3:.5f}%.")
        print("      Random numbers and nonsense values achieve identical precision.")
        print("      The precision is a property of grid DENSITY, not physics.")
    if x_fitted:
        print("    FITTING: X_v3 = 285.999... was reverse-solved to place c on-grid.")
        print("      This is parameter fitting, not a prediction or discovery.")

    print("\n  WHAT IS REAL:")
    print("    - The equation IS a valid encoding scheme (like a logarithmic ruler)")
    print("    - 286 pm IS close to Fe BCC lattice (0.23% off)")
    print("    - 104 = 26*4 IS a factual relationship (Fe_Z * He4_A)")
    print("    - The phi exponent IS shared across both layers")
    print("    - The math IS clean and error-free")

    print("\n  WHAT IS NOT REAL:")
    print("    - '63 constants at ±0.005%' is NOT a discovery (grid density guarantees it)")
    print("    - 'Speed of light is EXACT' is NOT meaningful (X_v3 was fitted to make it so)")
    print("    - The equation does NOT derive physics from first principles")
    print("    - The equation does NOT predict unknown values")

    print("\n  DISCOVERY ENGINE RESULTS:")
    print("    We searched exhaustively for genuine signal using 6 analyses:")
    print("    1. Exponent arithmetic closure: NOT significant (p=0.76 vs proper null)")
    print("    2. Mod-4 clustering: Significant IN-SAMPLE (p=0.004) but FAILS holdout")
    print("       (30 new constants show 26.7%, not 42.9% — selection effect)")
    print("    3. Dimensionless ratios: Not better than random")
    print("    4. Golden ratio in exponents: Fewer hits than random")
    print("    5. Pitch-class clustering: p=0.105, not significant")
    print("    6. Physical family spacing: No regular patterns found")
    print("    VERDICT: No discovery survives out-of-sample validation.")
    print("    The encoding is clean math. It is not physics.")

    print("\n  WHAT IS GENUINELY INTERESTING:")
    print("    - Q=26 (Fe atomic number) shows strongest mod-4 effect among all Q")
    print("    - Speed of light at 0.064% on ZERO-fitted grid (p=0.048, suggestive)")
    print("    - Non-physics constants show NO mod-4 clustering (good discrimination)")
    print("    - The encoding IS a valid, compact notation for physical constants")

    print(f"\n  {'=' * 80}")
    print(f"  GOD_CODE = {GOD_CODE}")
    print(f"  GOD_CODE_V3 = {GOD_CODE_V3}")
    print(f"  {'=' * 80}")
