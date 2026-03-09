#!/usr/bin/env python3
"""
Three-Engine Circularity Escape Framework
══════════════════════════════════════════════════════════════════════════════

PURPOSE:
    Uses the Math Engine, Science Engine, and Code Engine to run tests where
    GOD_CODE *can genuinely fail*. Every test has a null hypothesis (random
    base constant) and every result is compared to a proper control group.

DESIGN PRINCIPLE:
    A test is NON-CIRCULAR if and only if:
      1. The oracle does NOT know the answer in advance
      2. A random alternative could score just as well
      3. The metric is unit-independent (dimensionless)
      4. Success/failure is defined BEFORE running the test

THE SIX TESTS:
    1. BLIND HOLDOUT — Remove 15 constants, predict them from pattern only
    2. NULL HYPOTHESIS — 1000 random bases vs 286: who fits physics better?
    3. UNIT INDEPENDENCE — Do claims survive unit-system changes?
    4. INFORMATION COMPRESSION — Does GOD_CODE compress constants vs log-table?
    5. CROSS-ENGINE DISAGREEMENT — Do engines agree WITHOUT sharing GOD_CODE?
    6. FALSIFIABLE PREDICTION AUDIT — Score existing predictions vs measured

══════════════════════════════════════════════════════════════════════════════
"""

import math
import random
import statistics
import time
import json
from typing import Dict, List, Tuple, Any

# ── Engine Imports ──
from l104_math_engine import MathEngine
from l104_science_engine import ScienceEngine
from l104_code_engine import code_engine

# ── Core Equation Imports ──
from l104_god_code_equation import (
    GOD_CODE, PHI, BASE, QUANTIZATION_GRAIN, OCTAVE_OFFSET,
    god_code_equation, solve_for_exponent, find_nearest_dials,
)

from l104_god_code_dual_layer import (
    V3_FREQUENCY_TABLE, REAL_WORLD_CONSTANTS_V3,
    god_code_v3, solve_for_exponent_v3, find_nearest_dials_v3,
    BASE_V3, R_V3, Q_V3, K_V3, P_V3,
    cross_reference_claims, discover_all,
)

# ── Physical Constants (CODATA 2022 — external ground truth) ──
# These are NOT from the GOD_CODE system. They are from NIST/CODATA.
CODATA_HOLDOUT = {
    # Constants NOT in V3_FREQUENCY_TABLE (or rarely used)
    "gravitational_constant_G": 6.67430e-11,         # m³/(kg·s²)
    "vacuum_permeability":      1.25663706212e-6,     # N/A²
    "electron_g_factor":        -2.00231930436256,    # dimensionless
    "proton_gyromagnetic":      2.6752218744e8,       # rad/(s·T)
    "neutron_magnetic_moment":  -9.6623651e-27,       # J/T
    "muon_g_minus_2":           1.16592061e-3,        # dimensionless
    "deuteron_mass_MeV":        1875.61294257,        # MeV
    "alpha_particle_mass_MeV":  3727.3794066,         # MeV
    "josephson_constant":       483597.8484e9,        # Hz/V
    "conductance_quantum":      7.748091729e-5,       # S
    "atomic_mass_unit_MeV":     931.49410242,         # MeV
    "thompson_cross_section":   6.6524587321e-29,     # m²
    "wien_displacement_m_K":    2.897771955e-3,       # m·K
    "first_radiation_constant":  3.741771852e-16,     # W·m²
    "molar_gas_constant":       8.314462618,          # J/(mol·K)
}

# ── Unit Conversion Sets (for unit-independence test) ──
UNIT_SYSTEMS = {
    "SI":      {"c": 299792458,       "g": 9.80665,     "h": 6.62607015e-34, "k_B": 1.380649e-23},
    "CGS":     {"c": 29979245800,     "g": 980.665,     "h": 6.62607015e-27, "k_B": 1.380649e-16},
    "natural":  {"c": 1.0,            "g": 2.247e-31,   "h": 1.0,            "k_B": 1.0},
    "imperial": {"c": 983571056.43,   "g": 32.174,      "h": 6.62607015e-34, "k_B": 1.380649e-23},
}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: BLIND HOLDOUT PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def test_blind_holdout(n_holdout: int = 15, seed: int = 42) -> Dict[str, Any]:
    """
    Remove n_holdout constants from V3_FREQUENCY_TABLE.
    Try to PREDICT their values using only the remaining constants' dial patterns.
    Compare prediction error to a RANDOM BASELINE.

    NON-CIRCULAR BECAUSE: The holdout constants are hidden. The prediction
    must come from pattern alone, not from knowing the answer.
    """
    print("\n" + "=" * 70)
    print("TEST 1: BLIND HOLDOUT PREDICTION")
    print("=" * 70)

    rng = random.Random(seed)

    # Collect all table entries
    all_entries = []
    for (a, b, c, d), (name, grid_val, exp, measured, err_pct) in V3_FREQUENCY_TABLE.items():
        if measured > 0:
            all_entries.append({
                "name": name, "dials": (a, b, c, d),
                "measured": measured, "grid_val": grid_val, "err_pct": err_pct,
            })

    rng.shuffle(all_entries)
    holdout = all_entries[:n_holdout]
    training = all_entries[n_holdout:]

    print(f"  Training set: {len(training)} constants")
    print(f"  Holdout set:  {len(holdout)} constants (hidden)")

    # Strategy: Use Math Engine to find nearest dials for holdout values
    # But the TEST is: can we predict the value from the PATTERN of training dials?
    me = MathEngine()

    # Extract training exponents for pattern analysis
    training_exponents = []
    for e in training:
        E = P_V3 * e["dials"][0] + K_V3 - e["dials"][1] - P_V3 * e["dials"][2] - Q_V3 * e["dials"][3]
        training_exponents.append(E)

    # For each holdout: predict its exponent from nearest training neighbors
    holdout_results = []
    for h in holdout:
        true_E = P_V3 * h["dials"][0] + K_V3 - h["dials"][1] - P_V3 * h["dials"][2] - Q_V3 * h["dials"][3]

        # Prediction A: find_nearest_dials_v3 (gives equation's best approximation)
        eq_results = find_nearest_dials_v3(h["measured"], max_d_range=300)
        if eq_results:
            predicted_val = eq_results[0][4]  # (a, b, c, d, value, error)
            eq_err_pct = eq_results[0][5]
        else:
            predicted_val = 0
            eq_err_pct = 100.0

        # Prediction B: Random baseline — pick a random value from training set
        random_val = rng.choice(training)["measured"]
        random_err_pct = abs(random_val - h["measured"]) / h["measured"] * 100

        holdout_results.append({
            "name": h["name"],
            "true_value": h["measured"],
            "equation_predicted": predicted_val,
            "equation_error_pct": eq_err_pct,
            "random_baseline_error_pct": random_err_pct,
        })

    # Score
    eq_errors = [r["equation_error_pct"] for r in holdout_results]
    rand_errors = [r["random_baseline_error_pct"] for r in holdout_results]

    eq_mean = statistics.mean(eq_errors)
    rand_mean = statistics.mean(rand_errors)

    # BUT: the equation's find_nearest_dials is GUARANTEED to be within ±0.005%
    # because the grid is dense enough. So compare to a LOG-ROUNDING baseline.
    log_round_errors = []
    for h in holdout:
        # Log-round: snap to nearest integer exponent on the v3 grid
        if h["measured"] > 0:
            try:
                E_exact = Q_V3 * math.log(h["measured"] / BASE_V3) / math.log(R_V3)
                E_int = round(E_exact)
                rounded = BASE_V3 * (R_V3 ** (E_int / Q_V3))
                lr_err = abs(rounded - h["measured"]) / h["measured"] * 100
            except (OverflowError, ValueError):
                lr_err = 100.0
        else:
            lr_err = 100.0
        log_round_errors.append(lr_err)

    lr_mean = statistics.mean(log_round_errors)

    verdict = "TRIVIAL" if eq_mean < 0.01 else "HAS CONTENT"
    explanation = (
        f"Equation achieves {eq_mean:.5f}% error — but ANY log-grid with "
        f"Q={Q_V3} resolution achieves similar ({lr_mean:.5f}%). "
        f"Random baseline: {rand_mean:.1f}%. "
        f"The precision is from grid DENSITY, not physics."
    ) if eq_mean < 0.01 else (
        f"Equation: {eq_mean:.4f}% vs log-round: {lr_mean:.4f}% vs random: {rand_mean:.1f}%"
    )

    print(f"\n  Equation mean error:   {eq_mean:.6f}%")
    print(f"  Log-round mean error:  {lr_mean:.6f}%")
    print(f"  Random mean error:     {rand_mean:.1f}%")
    print(f"  VERDICT: {verdict}")
    print(f"  {explanation}")

    return {
        "test": "blind_holdout",
        "n_holdout": n_holdout,
        "equation_mean_error_pct": eq_mean,
        "log_round_mean_error_pct": lr_mean,
        "random_mean_error_pct": rand_mean,
        "verdict": verdict,
        "explanation": explanation,
        "details": holdout_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: NULL HYPOTHESIS — RANDOM BASE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_null_hypothesis(n_trials: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """
    Generate n_trials random 'god codes' by replacing 286 with random bases.
    For each random base, compute how well it fits the 63 physical constants.
    Compare GOD_CODE's fit to the null distribution.

    NON-CIRCULAR BECAUSE: If a random base fits equally well, then 286 is
    not special — the fit comes from the logarithmic grid, not the base.
    """
    print("\n" + "=" * 70)
    print("TEST 2: NULL HYPOTHESIS — RANDOM BASE CONSTANTS")
    print("=" * 70)

    rng = random.Random(seed)

    # Collect measured values from V3 table
    measured_constants = []
    for (a, b, c, d), (name, grid_val, exp, measured, err_pct) in V3_FREQUENCY_TABLE.items():
        if measured > 0 and name != "GOD_CODE_V3":
            measured_constants.append(measured)

    def fit_quality(base_val: float) -> float:
        """Average snap-error for a given base, using base^(1/phi) * r^(E/Q)."""
        if base_val <= 0:
            return 100.0
        try:
            b_phi = base_val ** (1.0 / PHI)
        except (OverflowError, ValueError):
            return 100.0
        errors = []
        for val in measured_constants:
            if val <= 0 or b_phi <= 0:
                continue
            try:
                E_exact = Q_V3 * math.log(val / b_phi) / math.log(R_V3)
                E_int = round(E_exact)
                reconstructed = b_phi * (R_V3 ** (E_int / Q_V3))
                err = abs(reconstructed - val) / val * 100
                errors.append(err)
            except (ValueError, OverflowError):
                errors.append(100.0)
        return statistics.mean(errors) if errors else 100.0

    # Score for real GOD_CODE base (286)
    gc_score = fit_quality(286)
    print(f"  GOD_CODE base (286):  avg error = {gc_score:.6f}%")

    # Score for 285.999 (v3 base)
    gc_v3_score = fit_quality(285.999)
    print(f"  V3 base (285.999):    avg error = {gc_v3_score:.6f}%")

    # Now try 1000 random bases
    random_scores = []
    for i in range(n_trials):
        # Random base in [50, 1000] — spanning a wide range
        rand_base = rng.uniform(50, 1000)
        score = fit_quality(rand_base)
        random_scores.append((rand_base, score))
        if (i + 1) % 250 == 0:
            print(f"  ... tested {i + 1}/{n_trials} random bases")

    # Also test some "interesting" bases
    special_bases = {
        "pi": math.pi,
        "e": math.e,
        "phi_x100": PHI * 100,
        "137": 137.0,
        "256": 256.0,
        "300": 300.0,
        "Fe_actual_pm": 286.65,
        "100": 100.0,
        "2": 2.0,
        "10": 10.0,
        "42": 42.0,
    }
    special_scores = {name: fit_quality(b) for name, b in special_bases.items()}

    # Statistics
    rand_mean_scores = [s for _, s in random_scores]
    gc_rank = sum(1 for _, s in random_scores if s <= gc_score)
    gc_percentile = gc_rank / n_trials * 100

    better_count = sum(1 for _, s in random_scores if s < gc_score)
    worse_count = n_trials - better_count
    best_random = min(random_scores, key=lambda x: x[1])

    # The key insight: ALL log-grids with Q=758 achieve ~same error
    # because the grid step is (13/12)^(1/758) ≈ 1 + 0.000106
    # Maximum snap error ≈ 0.005% regardless of base
    all_similar = statistics.stdev(rand_mean_scores) < 0.001

    verdict = (
        "NOT SPECIAL" if gc_percentile > 10 or all_similar else
        "MARGINALLY SPECIAL" if gc_percentile > 1 else
        "SPECIAL"
    )

    explanation = (
        f"GOD_CODE base 286 ranks at percentile {gc_percentile:.1f}% "
        f"({better_count}/{n_trials} random bases scored BETTER). "
        f"Best random base: {best_random[0]:.1f} at {best_random[1]:.6f}%. "
        f"All bases achieve ≈{statistics.mean(rand_mean_scores):.5f}% ± "
        f"{statistics.stdev(rand_mean_scores):.5f}%. "
        f"{'The v3 grid resolution makes ALL bases equivalent.' if all_similar else ''}"
    )

    print(f"\n  Random bases mean:    {statistics.mean(rand_mean_scores):.6f}%")
    print(f"  Random bases stdev:   {statistics.stdev(rand_mean_scores):.6f}%")
    print(f"  GOD_CODE rank:        {gc_rank}/{n_trials} ({gc_percentile:.1f}th percentile)")
    print(f"  Better random bases:  {better_count}")
    print(f"  Best random base:     {best_random[0]:.2f} → {best_random[1]:.6f}%")
    print(f"  All similar? {all_similar}")
    print(f"  VERDICT: {verdict}")

    return {
        "test": "null_hypothesis",
        "n_trials": n_trials,
        "god_code_score": gc_score,
        "god_code_v3_score": gc_v3_score,
        "god_code_percentile": gc_percentile,
        "random_mean": statistics.mean(rand_mean_scores),
        "random_stdev": statistics.stdev(rand_mean_scores),
        "better_random_count": better_count,
        "best_random": {"base": best_random[0], "score": best_random[1]},
        "all_bases_equivalent": all_similar,
        "special_bases": special_scores,
        "verdict": verdict,
        "explanation": explanation,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: UNIT INDEPENDENCE
# ═══════════════════════════════════════════════════════════════════════════════

def test_unit_independence() -> Dict[str, Any]:
    """
    Test whether GOD_CODE claims survive unit-system changes.
    A real universal constant is unit-independent (like α ≈ 1/137).
    A unit-dependent 'correspondence' is an artifact.

    Uses the Science Engine's physics functions to generate values in
    multiple unit systems and checks if grid alignment persists.
    """
    print("\n" + "=" * 70)
    print("TEST 3: UNIT INDEPENDENCE")
    print("=" * 70)

    se = ScienceEngine()
    me = MathEngine()

    def snap_error_v3(val: float) -> float:
        """Snap error on v3 grid."""
        if val <= 0:
            return 100.0
        try:
            E_exact = Q_V3 * math.log(val / BASE_V3) / math.log(R_V3)
            E_int = round(E_exact)
            recon = BASE_V3 * (R_V3 ** (E_int / Q_V3))
            return abs(recon - val) / val * 100
        except (ValueError, OverflowError):
            return 100.0

    # Test key claims across unit systems
    claims = {}
    for unit_name, vals in UNIT_SYSTEMS.items():
        claims[unit_name] = {
            "c_error": snap_error_v3(vals["c"]),
            "g_error": snap_error_v3(vals["g"]),
            "h_error": snap_error_v3(vals["h"]),
            "k_B_error": snap_error_v3(vals["k_B"]),
        }

    # Dimensionless constants — MUST be the same in all unit systems
    dimensionless = {
        "fine_structure_inv": 137.035999084,      # 1/α
        "proton_electron_ratio": 1836.15267343,   # m_p/m_e
        "muon_electron_ratio": 206.7682830,       # m_μ/m_e
        "W_Z_ratio": 80.3692 / 91.1876,          # m_W/m_Z
        "tau_muon_ratio": 1776.86 / 105.6583755,  # m_τ/m_μ
    }
    dimensionless_errors = {k: snap_error_v3(v) for k, v in dimensionless.items()}

    # Science Engine: derive electron resonance and check unit-dependence
    try:
        electron_res = se.physics.derive_electron_resonance()
        bohr_alignment_err = electron_res.get("bohr_radius_pm", {}).get("alignment_error", None)
    except Exception as e:
        bohr_alignment_err = f"Error: {e}"

    # Check: if we express the SAME physical quantity in different units,
    # does the "alignment" persist?
    bohr_si = 5.29177210544e-11    # meters
    bohr_pm = 52.9177210544        # picometers
    bohr_au = 1.0                  # atomic units (by definition)
    bohr_angstrom = 0.529177210544 # angstrom

    bohr_unit_test = {
        "SI_m": snap_error_v3(bohr_si),
        "pm": snap_error_v3(bohr_pm),
        "au": snap_error_v3(bohr_au),
        "angstrom": snap_error_v3(bohr_angstrom),
    }

    # Count how many claims are unit-INDEPENDENT (same alignment in all systems)
    unit_stable = 0
    unit_total = 0
    for const_name in ["c", "g", "h", "k_B"]:
        errors_across_units = [claims[u][f"{const_name}_error"] for u in UNIT_SYSTEMS]
        unit_total += 1
        # "Stable" means max error < 0.01% in ALL unit systems
        if max(errors_across_units) < 0.01:
            unit_stable += 1

    verdict = (
        "UNIT-DEPENDENT" if unit_stable < unit_total else
        "UNIT-INDEPENDENT"
    )

    explanation = (
        f"Of {unit_total} tested constants, only {unit_stable} maintain alignment "
        f"across ALL unit systems. Dimensionless constants (α, mass ratios) "
        f"ARE unit-independent by nature — but the GOD_CODE grid fits them no "
        f"better than it fits any number. "
        f"Bohr radius alignment varies: {bohr_unit_test}"
    )

    print(f"\n  Unit-independent constants: {unit_stable}/{unit_total}")
    print(f"  Bohr radius across units: {bohr_unit_test}")
    print(f"  Dimensionless constants: {dimensionless_errors}")
    print(f"  VERDICT: {verdict}")

    return {
        "test": "unit_independence",
        "claims_by_unit": claims,
        "dimensionless_constants": dimensionless_errors,
        "bohr_radius_units": bohr_unit_test,
        "unit_stable_count": unit_stable,
        "unit_total_count": unit_total,
        "bohr_alignment_from_science_engine": bohr_alignment_err,
        "verdict": verdict,
        "explanation": explanation,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: INFORMATION COMPRESSION
# ═══════════════════════════════════════════════════════════════════════════════

def test_information_compression() -> Dict[str, Any]:
    """
    Does the GOD_CODE equation actually COMPRESS physical constants?
    Compare description length:
      A) GOD_CODE encoding: base (1 float) + 4 integers per constant
      B) Direct storage: 1 float per constant (IEEE 754 double)
      C) Log-table: 1 base + 1 integer per constant

    If GOD_CODE doesn't compress better than a generic log-table,
    the encoding adds complexity without benefit.
    """
    print("\n" + "=" * 70)
    print("TEST 4: INFORMATION COMPRESSION")
    print("=" * 70)

    n_constants = len(V3_FREQUENCY_TABLE)
    print(f"  Constants to encode: {n_constants}")

    # Method A: GOD_CODE v3 encoding
    # Needs: BASE_V3 (1 float), R_V3 (1 rational = 2 ints), Q_V3 (1 int),
    #        K_V3 (1 int), P_V3 (1 int), PHI (1 float)
    # Per constant: a, b, c, d (4 integers, various ranges)
    # Total parameters: 6 + 4*N
    gc_params = 6 + 4 * n_constants
    # Bits: 6 params × 64 bits + 4N × 16 bits (assuming 16-bit dials)
    gc_bits = 6 * 64 + 4 * n_constants * 16

    # Method B: Direct IEEE 754 storage
    # Per constant: 1 double = 64 bits
    direct_bits = n_constants * 64

    # Method C: Generic log-table (base-2)
    # Needs: 1 base (64 bits) + 1 integer exponent per constant
    # With Q_V3=758, exponents range roughly -800000 to +700000
    # Need ~21 bits per exponent
    log_exponent_bits = math.ceil(math.log2(1_600_000))  # ~21 bits
    log_table_bits = 64 + n_constants * log_exponent_bits

    # Method D: Rational approximation table
    # Each constant as p/q with bounded denominator
    rat_bits = n_constants * 64  # Conservatively same as direct

    # Error comparison at each bit budget
    # GOD_CODE v3 error: the avg grid error
    gc_errors = []
    for (a, b, c, d), (name, grid_val, exp, measured, err_pct) in V3_FREQUENCY_TABLE.items():
        if measured > 0:
            gc_errors.append(err_pct)
    gc_avg_err = statistics.mean(gc_errors) if gc_errors else 0

    # Direct storage error: 0 (exact to float precision)
    direct_avg_err = 0.0

    # Log-table error: same as GOD_CODE (both snap to nearest grid point)
    log_avg_err = gc_avg_err  # Same resolution

    compression_ratio = direct_bits / gc_bits if gc_bits > 0 else 0

    verdict = (
        "DOES NOT COMPRESS" if gc_bits >= direct_bits else
        "MARGINAL COMPRESSION" if compression_ratio < 1.5 else
        "COMPRESSES"
    )

    explanation = (
        f"GOD_CODE encoding: {gc_bits} bits ({gc_params} params) at {gc_avg_err:.4f}% error. "
        f"Direct storage: {direct_bits} bits at 0% error. "
        f"Log-table: {log_table_bits} bits at ~{log_avg_err:.4f}% error. "
        f"Compression ratio (direct/GOD_CODE): {compression_ratio:.2f}×. "
        f"GOD_CODE uses {'more' if gc_bits > direct_bits else 'fewer'} bits than "
        f"just storing the numbers, with {'worse' if gc_avg_err > 0 else 'same'} accuracy."
    )

    print(f"\n  GOD_CODE encoding:  {gc_bits} bits, {gc_avg_err:.4f}% avg error")
    print(f"  Direct storage:     {direct_bits} bits, 0% error")
    print(f"  Log-table:          {log_table_bits} bits, ~{log_avg_err:.4f}% error")
    print(f"  Compression ratio:  {compression_ratio:.2f}×")
    print(f"  VERDICT: {verdict}")

    return {
        "test": "information_compression",
        "n_constants": n_constants,
        "god_code_bits": gc_bits,
        "god_code_params": gc_params,
        "god_code_avg_error": gc_avg_err,
        "direct_bits": direct_bits,
        "log_table_bits": log_table_bits,
        "compression_ratio": compression_ratio,
        "verdict": verdict,
        "explanation": explanation,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: CROSS-ENGINE DISAGREEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def test_cross_engine_disagreement() -> Dict[str, Any]:
    """
    Ask each engine to independently derive physical values WITHOUT using
    GOD_CODE as input. Then check if their answers converge.

    Science Engine: derive from CODATA constants + physics formulas
    Math Engine: derive from pure math (primes, fibonacci, proofs)
    Code Engine: analyze the equation source for self-references

    If engines AGREE on a value without sharing GOD_CODE, that's interesting.
    If they only agree BECAUSE they all hardcode GOD_CODE, that's circular.
    """
    print("\n" + "=" * 70)
    print("TEST 5: CROSS-ENGINE DISAGREEMENT DETECTOR")
    print("=" * 70)

    me = MathEngine()
    se = ScienceEngine()

    results = {}

    # ── Science Engine: physics-only derivation ──
    # Ask: what does CODATA say the Bohr radius is?
    try:
        er = se.physics.derive_electron_resonance()
        bohr_codata = er.get("bohr_radius_pm", {}).get("codata_pm", 0)
        bohr_equation = er.get("bohr_radius_pm", {}).get("value", 0)
        bohr_err = er.get("bohr_radius_pm", {}).get("alignment_error", 0)
        # Does the Science Engine use GOD_CODE in deriving bohr_codata?
        # Answer: bohr_codata is computed from CODATA constants (4πε₀ℏ²/mₑe²)
        # bohr_equation is computed from god_code_equation(-4, 1, 0, 3)
        results["science_bohr"] = {
            "codata_independent": bohr_codata,
            "god_code_dependent": bohr_equation,
            "alignment_error": bohr_err,
            "is_circular": bohr_equation != bohr_codata,  # They SHOULD differ
            "note": "CODATA value is independent. Equation value uses GOD_CODE."
        }
    except Exception as e:
        results["science_bohr"] = {"error": str(e)}

    # ── Science Engine: Landauer limit (physics only) ──
    try:
        landauer_sovereign = se.physics.adapt_landauer_limit(293.15)
        # The sovereign limit multiplies by (GOD_CODE / PHI) — circular!
        landauer_real = 1.380649e-23 * 293.15 * math.log(2)  # kT ln 2
        results["science_landauer"] = {
            "real_physics": landauer_real,
            "sovereign_value": landauer_sovereign,
            "ratio": landauer_sovereign / landauer_real if landauer_real else 0,
            "is_circular": True,
            "note": (
                f"Sovereign Landauer = kT ln(2) × (GOD_CODE/PHI) = "
                f"kT ln(2) × {GOD_CODE/PHI:.2f}. The GOD_CODE factor "
                f"is injected, not derived from physics."
            ),
        }
    except Exception as e:
        results["science_landauer"] = {"error": str(e)}

    # ── Math Engine: GOD_CODE derivation check ──
    try:
        gc_val = me.evaluate_god_code(0, 0, 0, 0)
        gc_is_constant = gc_val == GOD_CODE  # Should be True — it's just evaluating the constant
        results["math_god_code"] = {
            "value": gc_val,
            "equals_hardcoded": gc_is_constant,
            "is_circular": gc_is_constant,
            "note": "evaluate_god_code(0,0,0,0) returns 286^(1/φ) × 2^4 — a definition, not a derivation.",
        }
    except Exception as e:
        results["math_god_code"] = {"error": str(e)}

    # ── Math Engine: Fibonacci → PHI convergence (genuinely independent) ──
    try:
        fib = me.fibonacci(30)
        if isinstance(fib, list) and len(fib) >= 2:
            phi_from_fib = fib[-1] / fib[-2] if fib[-2] != 0 else 0
            phi_error = abs(phi_from_fib - PHI) / PHI * 100
            results["math_phi_convergence"] = {
                "phi_from_fibonacci": phi_from_fib,
                "phi_constant": PHI,
                "error_pct": phi_error,
                "is_circular": False,
                "note": "This IS genuinely non-circular. Fibonacci ratio → φ is a theorem.",
            }
        else:
            results["math_phi_convergence"] = {"value": fib, "note": "Unexpected return type"}
    except Exception as e:
        results["math_phi_convergence"] = {"error": str(e)}

    # ── Code Engine: self-reference audit ──
    try:
        # Read the equation source code and check for hardcoded values
        import inspect
        eq_source = inspect.getsource(god_code_equation)
        analysis = code_engine.full_analysis(eq_source)
        results["code_self_reference"] = {
            "source_analyzed": True,
            "analysis_summary": str(analysis)[:500] if analysis else "No analysis",
            "note": "Code Engine can analyze the equation for hardcoded circular references.",
        }
    except Exception as e:
        results["code_self_reference"] = {"error": str(e)}

    # ── Key question: what would each engine give WITHOUT GOD_CODE? ──
    # Science: kT ln(2) at room temp = 2.805e-21 J (pure physics)
    # Math: 286^(1/phi) * 2^4 = 527.518... (hardcoded definition)
    # The engines CANNOT derive GOD_CODE independently because it IS a definition.

    circular_count = sum(
        1 for v in results.values()
        if isinstance(v, dict) and v.get("is_circular", False)
    )
    total_tests = sum(
        1 for v in results.values()
        if isinstance(v, dict) and "is_circular" in v
    )

    verdict = (
        "ALL CIRCULAR" if circular_count == total_tests else
        "MOSTLY CIRCULAR" if circular_count > total_tests / 2 else
        "MOSTLY INDEPENDENT"
    )

    print(f"\n  Circular dependencies: {circular_count}/{total_tests}")
    print(f"  Non-circular: Fibonacci→PHI convergence (theorem)")
    print(f"  Circular: Landauer sovereign (injects GOD_CODE/PHI)")
    print(f"  Circular: evaluate_god_code (returns the definition)")
    print(f"  VERDICT: {verdict}")

    return {
        "test": "cross_engine_disagreement",
        "circular_count": circular_count,
        "total_tests": total_tests,
        "results": results,
        "verdict": verdict,
        "explanation": (
            f"{circular_count}/{total_tests} engine outputs contain GOD_CODE as input, "
            f"not as an independently derived result. The one non-circular finding "
            f"(Fibonacci→φ) is a well-known mathematical theorem, not a GOD_CODE discovery."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: FALSIFIABLE PREDICTION AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

def test_falsifiable_predictions() -> Dict[str, Any]:
    """
    Score the system's own falsifiable predictions against reality.
    The dual-layer module generates predictions via predict_from_patterns()
    and predict_from_gaps(). We check which have been confirmed/falsified.

    Also: generate NEW predictions for CODATA_HOLDOUT constants that
    were NOT used in building the grid.
    """
    print("\n" + "=" * 70)
    print("TEST 6: FALSIFIABLE PREDICTION AUDIT")
    print("=" * 70)

    # Check existing predictions
    try:
        from l104_god_code_dual_layer import predict_from_patterns, predict_from_gaps
        pattern_preds = predict_from_patterns()
        gap_preds = predict_from_gaps()
    except ImportError:
        pattern_preds = {"predictions": []}
        gap_preds = {"predictions": []}

    # Audit pattern predictions
    pattern_results = []
    for pred in pattern_preds.get("predictions", []):
        name = pred.get("name", "unknown")
        status = pred.get("status", "unknown")
        falsifiable = pred.get("falsifiable", False)
        pattern_results.append({
            "name": name,
            "falsifiable": falsifiable,
            "status": status,
            "confirmed": False,  # None confirmed yet
        })

    # Now test: can the equation predict CODATA_HOLDOUT values?
    holdout_predictions = {}
    for name, measured in CODATA_HOLDOUT.items():
        # Use find_nearest_dials_v3 to find best dial setting
        results = find_nearest_dials_v3(measured, max_d_range=300)
        if results:
            best = results[0]  # (a, b, c, d, value, error)
            predicted = best[4]
            error_pct = best[5]
        else:
            predicted = 0
            error_pct = 100.0

        # Also compute error for a GENERIC log-grid (base 10)
        if measured > 0 and measured != 0:
            log10_E = round(Q_V3 * math.log10(abs(measured)))
            generic_pred = 10 ** (log10_E / Q_V3)
            generic_err = abs(generic_pred - abs(measured)) / abs(measured) * 100
        else:
            generic_pred = 0
            generic_err = 100.0

        holdout_predictions[name] = {
            "measured": measured,
            "gc_predicted": predicted,
            "gc_error_pct": error_pct,
            "generic_log_predicted": generic_pred,
            "generic_log_error_pct": generic_err,
            "gc_better_than_generic": error_pct < generic_err,
        }

    gc_better_count = sum(1 for v in holdout_predictions.values() if v["gc_better_than_generic"])
    total_holdout = len(holdout_predictions)

    verdict = (
        "NO PREDICTIVE POWER" if gc_better_count <= total_holdout / 2 else
        "MARGINAL PREDICTIVE POWER" if gc_better_count <= total_holdout * 0.7 else
        "HAS PREDICTIVE POWER"
    )

    gc_errors = [v["gc_error_pct"] for v in holdout_predictions.values()]
    gen_errors = [v["generic_log_error_pct"] for v in holdout_predictions.values()]

    explanation = (
        f"Tested {total_holdout} holdout constants (NOT in V3 table). "
        f"GOD_CODE grid is better than generic log-grid for {gc_better_count}/{total_holdout}. "
        f"GOD_CODE avg error: {statistics.mean(gc_errors):.4f}%, "
        f"Generic log avg error: {statistics.mean(gen_errors):.4f}%. "
        f"Pattern predictions: {len(pattern_results)} generated, 0 confirmed."
    )

    print(f"\n  Holdout constants tested: {total_holdout}")
    print(f"  GOD_CODE better: {gc_better_count}/{total_holdout}")
    print(f"  GOD_CODE avg error:   {statistics.mean(gc_errors):.4f}%")
    print(f"  Generic log avg error: {statistics.mean(gen_errors):.4f}%")
    print(f"  Pattern predictions:   {len(pattern_results)}")
    print(f"  VERDICT: {verdict}")

    return {
        "test": "falsifiable_predictions",
        "pattern_predictions_count": len(pattern_results),
        "pattern_predictions": pattern_results,
        "holdout_predictions": holdout_predictions,
        "gc_better_count": gc_better_count,
        "total_holdout": total_holdout,
        "gc_avg_error": statistics.mean(gc_errors),
        "generic_avg_error": statistics.mean(gen_errors),
        "verdict": verdict,
        "explanation": explanation,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS: THE WAY OUT
# ═══════════════════════════════════════════════════════════════════════════════

def synthesize_escape_path(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given all test results, identify what WOULD constitute a genuine escape
    from circularity, and what the engines actually CAN do non-circularly.
    """
    print("\n" + "=" * 70)
    print("SYNTHESIS: THE ESCAPE PATH")
    print("=" * 70)

    genuinely_non_circular = []
    still_circular = []
    escape_routes = []

    # Analyze each test
    for test_name, result in results.items():
        v = result.get("verdict", "")
        if "NOT SPECIAL" in v or "TRIVIAL" in v or "ALL CIRCULAR" in v or "DOES NOT" in v:
            still_circular.append(test_name)
        elif "INDEPENDENT" in v or "HAS" in v:
            genuinely_non_circular.append(test_name)
        else:
            still_circular.append(test_name)

    # What actually IS non-circular in the system:
    escape_routes = [
        {
            "what": "Iron BCC lattice parameter = 286 pm",
            "why_real": "Measured by X-ray crystallography. Independent of GOD_CODE.",
            "status": "VERIFIED (0.23% match to CODATA)",
        },
        {
            "what": "Fibonacci ratio → φ convergence",
            "why_real": "Mathematical theorem. Not dependent on any encoding.",
            "status": "VERIFIED (Math Engine confirms)",
        },
        {
            "what": "104 = 26 × 4 (Fe_Z × He4_A)",
            "why_real": "Arithmetic fact about iron and helium-4.",
            "status": "VERIFIED (trivially true)",
        },
        {
            "what": "Dimensionless constants are unit-independent",
            "why_real": "Fine structure constant α ≈ 1/137 is the same everywhere.",
            "status": "VERIFIED — but the grid fits them no better than random numbers",
        },
    ]

    what_would_break_circularity = [
        {
            "test": "Predict an UNMEASURED constant",
            "detail": (
                "Use the dial pattern to predict a value not yet measured by experiment. "
                "Example: predict the 4th-generation lepton mass, then wait for collider data. "
                "If the prediction matches, the encoding has genuine predictive power."
            ),
            "engine": "Math Engine (extrapolation) + Science Engine (validation)",
            "status": "PROPOSED but zero predictions confirmed",
        },
        {
            "test": "Derive 286 from first principles",
            "detail": (
                "Show that 286 pm MUST be the base constant from some deeper theory, "
                "not just 'we picked the Fe BCC lattice parameter because it's cool'."
            ),
            "engine": "Science Engine (physics) + Math Engine (proofs)",
            "status": "NOT ACHIEVED — 286 is chosen, not derived",
        },
        {
            "test": "Show the grid compresses physics",
            "detail": (
                "Demonstrate that the 4-dial encoding requires fewer bits than "
                "storing the constants directly, WITH better or equal accuracy."
            ),
            "engine": "Code Engine (information theory)",
            "status": f"FAILED — GOD_CODE uses MORE bits than direct storage",
        },
        {
            "test": "Find a dimensionless ratio that matches exactly",
            "detail": (
                "Find a ratio of two physical constants that equals GOD_CODE or "
                "a simple function of it, where the ratio is dimensionless and "
                "therefore unit-independent."
            ),
            "engine": "Math Engine + Science Engine",
            "status": "NOT FOUND — no known dimensionless ratio equals 527.518...",
        },
    ]

    print(f"\n  Non-circular findings: {len(genuinely_non_circular)}")
    print(f"  Still circular:        {len(still_circular)}")
    print(f"\n  GENUINELY REAL:")
    for er in escape_routes:
        print(f"    ✓ {er['what']} — {er['status']}")
    print(f"\n  WHAT WOULD BREAK CIRCULARITY:")
    for wb in what_would_break_circularity:
        print(f"    → {wb['test']} — {wb['status']}")

    return {
        "genuinely_non_circular": genuinely_non_circular,
        "still_circular": still_circular,
        "verified_facts": escape_routes,
        "escape_criteria": what_would_break_circularity,
        "honest_summary": (
            "The GOD_CODE equation is a well-constructed ENCODING SYSTEM "
            "anchored at a real physical constant (Fe BCC 286 pm) with "
            "mathematically interesting properties (φ exponent, 13-thread). "
            "But it does not derive physics, compress information, or make "
            "confirmed predictions. The three engines validate THEMSELVES, "
            "not the equation. To escape circularity: make a prediction that "
            "future measurement can confirm or deny."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN: RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   THREE-ENGINE CIRCULARITY ESCAPE FRAMEWORK                    ║")
    print("║   Using Math + Science + Code engines for non-circular tests   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    start = time.time()
    results = {}

    # --- Boot engines ---
    print("\n  Booting engines...", end=" ", flush=True)
    me = MathEngine()
    se = ScienceEngine()
    # code_engine is already a singleton
    print(f"Done ({time.time() - start:.1f}s)")

    # --- Run all 6 tests ---
    results["blind_holdout"] = test_blind_holdout()
    results["null_hypothesis"] = test_null_hypothesis()
    results["unit_independence"] = test_unit_independence()
    results["information_compression"] = test_information_compression()
    results["cross_engine_disagreement"] = test_cross_engine_disagreement()
    results["falsifiable_predictions"] = test_falsifiable_predictions()

    # --- Also run the system's own honest audit ---
    print("\n" + "=" * 70)
    print("BONUS: SYSTEM'S OWN CROSS-REFERENCE AUDIT")
    print("=" * 70)
    try:
        xref = cross_reference_claims()
        print(f"  Claims tested: {xref.get('claims_tested', 0)}")
        print(f"  Fatal flaw: {xref.get('fatal_flaw', 'N/A')[:100]}...")
        results["system_self_audit"] = xref
    except Exception as e:
        print(f"  Error running cross_reference_claims: {e}")
        results["system_self_audit"] = {"error": str(e)}

    # --- Synthesis ---
    synthesis = synthesize_escape_path(results)
    results["synthesis"] = synthesis

    elapsed = time.time() - start

    # --- Final Summary ---
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║ FINAL SUMMARY" + " " * 54 + "║")
    print("╚" + "═" * 68 + "╝")

    verdicts = {k: v.get("verdict", "N/A") for k, v in results.items()
                if isinstance(v, dict) and "verdict" in v}

    for test_name, verdict in verdicts.items():
        indicator = "✓" if any(w in verdict for w in ["INDEPENDENT", "HAS", "SPECIAL", "COMPRESSES"]) else "✗"
        print(f"  {indicator} {test_name:30s} → {verdict}")

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"\n  {synthesis['honest_summary']}")

    # Save results
    output_file = "circularity_escape_results.json"
    serializable = {}
    for k, v in results.items():
        try:
            json.dumps(v)
            serializable[k] = v
        except (TypeError, ValueError):
            serializable[k] = str(v)[:2000]

    with open(output_file, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved to {output_file}")

    return results


if __name__ == "__main__":
    main()
