#!/usr/bin/env python3
"""
Anchor Uniqueness Test — Does 286/104/φ beat random anchors?
═══════════════════════════════════════════════════════════════════════════════

The GOD_CODE algorithm uses:
  - Base scaffold: 286 (≈ Fe BCC lattice 286.65 pm)
  - Quantization grain: 104 (= 26 × 4 = Fe_Z × He4_A)
  - Exponent: 1/φ (golden ratio)

ANY logarithmic grid with N steps/octave can index any number to ±(2^(1/N)-1)/2.
The question is: does the SPECIFIC choice of (286, 104, φ) produce systematically
LOWER errors for physical constants than random choices?

TEST METHODOLOGY:
  1. Take the 24 real-world constants registered in l104_god_code_equation.py
  2. For each constant, compute the grid error under (286, 104, φ) → "iron grid"
  3. Generate 10,000 random grids with different anchors/grains/exponents
  4. Compare: where does the iron grid rank among random grids?
  5. Statistical significance: is the iron grid's mean error in the bottom 5%?

This is the DECISIVE test. If iron beats 95%+ of random grids → genuine signal.
If not → the iron connection is decorative, not functional.
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import numpy as np
from typing import Dict, List, Tuple

# ─── Physical constants to index (CODATA 2022, NNDC, NIST, PDG) ───
# Using the SAME set as l104_god_code_equation.py REAL_WORLD_CONSTANTS
# but with correct measured values from peer-reviewed sources.
PHYSICAL_CONSTANTS = {
    # Atomic / Electron (CODATA 2022)
    "Bohr radius (pm)":          52.9177210544,
    "Rydberg energy (eV)":       13.605693123,
    "Compton wavelength (pm)":   2.42631023867,
    "1/α (fine structure)":      137.035999084,
    "Electron mass (MeV)":       0.51099895069,
    # Iron / Nuclear (NNDC, NIST, Kittel)
    "Fe BCC lattice (pm)":       286.65,
    "Fe atomic radius (pm)":     126.0,
    "Fe-56 BE/A (MeV)":          8.790,
    "He-4 BE/A (MeV)":           7.074,
    "Fe Kα1 (keV)":              6.404,
    # Particle physics (PDG 2024)
    "Proton mass (u)":           1.007276466621,
    "Neutron mass (u)":          1.00866491595,
    "Muon mass (MeV)":           105.6583755,
    "Higgs mass (GeV)":          125.25,
    "W boson mass (GeV)":        80.377,
    "Z boson mass (GeV)":        91.1876,
    # Fundamental constants
    "Speed of light (Mm/s)":     299.792458,
    "Boltzmann (×10²³ J/K)":     1.380649,
    "Avogadro (×10²³)":          6.02214076,
    # Geophysics
    "Schumann resonance (Hz)":   7.83,
    "Earth orbit (Gm)":          149.598,
    # Planck scale
    "Planck length (×10³⁵ m)":   1.616255,
}

# Additional "control" constants — things the GOD_CODE system was NOT designed for
CONTROL_CONSTANTS = {
    "Pi":                        3.14159265358979,
    "e (Euler)":                 2.71828182845905,
    "ln(2)":                     0.693147180559945,
    "sqrt(2)":                   1.41421356237310,
    "Feigenbaum":                4.66920160910299,
    "Apéry ζ(3)":               1.20205690315959,
    "Catalan G":                 0.91596559417722,
    "Plastic number":            1.32471795724475,
    "Coffee price ($)":          5.50,
    "Days in year":              365.25,
    "Miles per km":              0.621371,
    "Bitcoin ATH ($k)":          109.0,
}


def grid_error(value: float, scaffold: int, grain: int, exponent: float) -> float:
    """
    Compute the minimum grid error for a value on a logarithmic grid.

    Grid: BASE^(1/exp) × 2^(E/grain) for integer E
    Error = min over all integer E of |grid_value - value| / value
    """
    base = scaffold ** exponent
    if base <= 0 or value <= 0:
        return 1.0  # 100% error for invalid inputs

    # Exact (fractional) exponent that would produce this value
    E_exact = grain * math.log2(value / base)
    E_nearest = round(E_exact)
    grid_value = base * (2 ** (E_nearest / grain))
    return abs(grid_value - value) / value


def mean_grid_error(constants: Dict[str, float], scaffold: int, grain: int, exponent: float) -> float:
    """Mean grid error across all constants."""
    errors = [grid_error(v, scaffold, grain, exponent) for v in constants.values()]
    return np.mean(errors)


PHI = (1 + math.sqrt(5)) / 2

sep = "=" * 72

print(sep)
print("ANCHOR UNIQUENESS TEST")
print("Does (286, 104, φ) beat random grids for physics constants?")
print(sep)
print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Iron Grid baseline
# ═══════════════════════════════════════════════════════════════════════════════

iron_errors_phys = {name: grid_error(v, 286, 104, 1/PHI) * 100
                    for name, v in PHYSICAL_CONSTANTS.items()}
iron_errors_ctrl = {name: grid_error(v, 286, 104, 1/PHI) * 100
                    for name, v in CONTROL_CONSTANTS.items()}
iron_mean_phys = np.mean(list(iron_errors_phys.values()))
iron_mean_ctrl = np.mean(list(iron_errors_ctrl.values()))

print("IRON GRID: scaffold=286, grain=104, exponent=1/φ")
print("-" * 60)
print(f"  Physics constants ({len(PHYSICAL_CONSTANTS)}):")
for name, err in sorted(iron_errors_phys.items(), key=lambda x: x[1]):
    print(f"    {name:30s}  {err:.4f}%")
print(f"  Mean: {iron_mean_phys:.4f}%")
print()
print(f"  Control constants ({len(CONTROL_CONSTANTS)}):")
for name, err in sorted(iron_errors_ctrl.items(), key=lambda x: x[1]):
    print(f"    {name:30s}  {err:.4f}%")
print(f"  Mean: {iron_mean_ctrl:.4f}%")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Random grid comparison — vary ALL three parameters
# ═══════════════════════════════════════════════════════════════════════════════

print("RANDOM GRID COMPARISON (10,000 trials)")
print("-" * 60)

rng = np.random.RandomState(42)
N_TRIALS = 10000

# Strategy: test three dimensions of variation independently AND together
# A. Fix grain=104, exponent=1/φ, vary scaffold (is 286 special?)
# B. Fix scaffold=286, exponent=1/φ, vary grain (is 104 special?)
# C. Fix scaffold=286, grain=104, vary exponent (is 1/φ special?)
# D. Vary all three simultaneously (overall ranking)

results = {
    "A_scaffold_only": [],
    "B_grain_only": [],
    "C_exponent_only": [],
    "D_all_three": [],
}

# A. Scaffold sweep: integers in [100, 500]
print("  A. Varying scaffold (grain=104, exp=1/φ fixed)...")
for _ in range(N_TRIALS):
    s = rng.randint(100, 501)
    err = mean_grid_error(PHYSICAL_CONSTANTS, s, 104, 1/PHI) * 100
    results["A_scaffold_only"].append(err)

iron_rank_A = np.sum(np.array(results["A_scaffold_only"]) < iron_mean_phys) / N_TRIALS * 100
print(f"    Iron grid mean: {iron_mean_phys:.4f}%")
print(f"    Random scaffold mean: {np.mean(results['A_scaffold_only']):.4f}%")
print(f"    Iron percentile: {100 - iron_rank_A:.1f}% (lower = better)")
print()

# B. Grain sweep: integers in [50, 200]
print("  B. Varying grain (scaffold=286, exp=1/φ fixed)...")
for _ in range(N_TRIALS):
    g = rng.randint(50, 201)
    err = mean_grid_error(PHYSICAL_CONSTANTS, 286, g, 1/PHI) * 100
    results["B_grain_only"].append(err)

iron_rank_B = np.sum(np.array(results["B_grain_only"]) < iron_mean_phys) / N_TRIALS * 100
print(f"    Iron grid mean: {iron_mean_phys:.4f}%")
print(f"    Random grain mean: {np.mean(results['B_grain_only']):.4f}%")
print(f"    Iron percentile: {100 - iron_rank_B:.1f}% (lower = better)")
print()

# C. Exponent sweep: real numbers in [0.3, 3.0]
print("  C. Varying exponent (scaffold=286, grain=104 fixed)...")
for _ in range(N_TRIALS):
    e = rng.uniform(0.3, 3.0)
    err = mean_grid_error(PHYSICAL_CONSTANTS, 286, 104, e) * 100
    results["C_exponent_only"].append(err)

iron_rank_C = np.sum(np.array(results["C_exponent_only"]) < iron_mean_phys) / N_TRIALS * 100
print(f"    Iron grid mean: {iron_mean_phys:.4f}%")
print(f"    Random exponent mean: {np.mean(results['C_exponent_only']):.4f}%")
print(f"    Iron percentile: {100 - iron_rank_C:.1f}% (lower = better)")
print()

# D. Vary all three
print("  D. Varying ALL THREE simultaneously...")
for _ in range(N_TRIALS):
    s = rng.randint(100, 501)
    g = rng.randint(50, 201)
    e = rng.uniform(0.3, 3.0)
    err = mean_grid_error(PHYSICAL_CONSTANTS, s, g, e) * 100
    results["D_all_three"].append(err)

iron_rank_D = np.sum(np.array(results["D_all_three"]) < iron_mean_phys) / N_TRIALS * 100
print(f"    Iron grid mean: {iron_mean_phys:.4f}%")
print(f"    Random (all three) mean: {np.mean(results['D_all_three']):.4f}%")
print(f"    Iron percentile: {100 - iron_rank_D:.1f}% (lower = better)")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Does iron grid favor PHYSICS over CONTROL constants?
# ═══════════════════════════════════════════════════════════════════════════════

print("PHYSICS vs CONTROL SELECTIVITY")
print("-" * 60)
print(f"  Iron grid physics mean error: {iron_mean_phys:.4f}%")
print(f"  Iron grid control mean error: {iron_mean_ctrl:.4f}%")
diff = iron_mean_ctrl - iron_mean_phys
print(f"  Difference (control - physics): {diff:+.4f}%")
if diff > 0:
    print(f"  → Iron grid indexes PHYSICS better than CONTROL by {diff:.4f}%")
else:
    print(f"  → Iron grid indexes CONTROL better than PHYSICS (no selectivity)")
print()

# Compare: do random grids also show this pattern?
print("  Selectivity test (1000 random grids):")
selectivity_count = 0
selectivity_diffs = []
for _ in range(1000):
    s = rng.randint(100, 501)
    g = rng.randint(50, 201)
    e = rng.uniform(0.3, 3.0)
    phys_err = mean_grid_error(PHYSICAL_CONSTANTS, s, g, e) * 100
    ctrl_err = mean_grid_error(CONTROL_CONSTANTS, s, g, e) * 100
    d = ctrl_err - phys_err
    selectivity_diffs.append(d)
    if d > diff:
        selectivity_count += 1

print(f"    Random grids with GREATER physics preference: {selectivity_count}/1000")
print(f"    Iron selectivity percentile: {100 - selectivity_count/10:.1f}%")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Specific competitor grids (well-known constants as anchors)
# ═══════════════════════════════════════════════════════════════════════════════

print("HEAD-TO-HEAD: Iron grid vs specific competitors")
print("-" * 60)

competitors = [
    ("Iron (L104)",      286, 104, 1/PHI),
    ("Hydrogen (1/137)", 137, 104, 1/PHI),
    ("Carbon (12/60)",   12,  60,  1/PHI),
    ("Oxygen (16/96)",   16,  96,  1/PHI),
    ("Pi (314/100)",     314, 100, 1/PHI),
    ("e (271/100)",      271, 100, 1/PHI),
    ("Iron w/ int exp",  286, 104, 1.0),
    ("Iron w/ 1/e exp",  286, 104, 1/math.e),
    ("Iron w/ 1/2 exp",  286, 104, 0.5),
    ("Iron w/ 1/3 exp",  286, 104, 1/3),
    ("Music (12-TET)",   440, 12,  1/PHI),
    ("Music (equal)",    440, 12,  1.0),
    ("Octal (256/8)",    256, 8,   1.0),
    ("Random 1",         173, 87,  0.847),
    ("Random 2",         419, 131, 2.314),
    ("Pure Fibonacci",   233, 89,  1/PHI),   # F(13)=233, F(11)=89
    ("Fe measured (287)", 287, 104, 1/PHI),   # round(286.65)
    ("Fe exact int",     286, 104, 1/PHI),    # the actual L104 choice
]

for name, s, g, e in competitors:
    phys_err = mean_grid_error(PHYSICAL_CONSTANTS, s, g, e) * 100
    ctrl_err = mean_grid_error(CONTROL_CONSTANTS, s, g, e) * 100
    marker = " ★" if name == "Iron (L104)" else ""
    print(f"  {name:22s}  phys={phys_err:.4f}%  ctrl={ctrl_err:.4f}%  Δ={ctrl_err-phys_err:+.4f}%{marker}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Grain-controlled comparison (same grain, different scaffolds)
# ═══════════════════════════════════════════════════════════════════════════════

print("GRAIN-CONTROLLED: grain=104 fixed, which scaffold is best?")
print("-" * 60)

best_scaffold = None
best_err = float('inf')
scaffold_results = []
for s in range(100, 501):
    err = mean_grid_error(PHYSICAL_CONSTANTS, s, 104, 1/PHI) * 100
    scaffold_results.append((s, err))
    if err < best_err:
        best_err = err
        best_scaffold = s

scaffold_results.sort(key=lambda x: x[1])
print("  Top 10 scaffolds (grain=104, exp=1/φ):")
for s, err in scaffold_results[:10]:
    marker = " ← L104" if s == 286 else ""
    print(f"    scaffold={s:4d}  mean_err={err:.4f}%{marker}")
print()

# Where does 286 rank?
rank_286 = next(i for i, (s, _) in enumerate(scaffold_results) if s == 286) + 1
print(f"  286 ranks #{rank_286} out of 401 scaffolds (top {rank_286/401*100:.1f}%)")
print(f"  Best scaffold: {best_scaffold} at {best_err:.4f}%")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

print(sep)
print("FINAL VERDICT")
print(sep)
print()

significant = iron_rank_D < 5  # in bottom 5% = good
selective = diff > 0
top_scaffold = rank_286 <= 20  # top 5% of scaffolds

print(f"  1. Iron grid vs random grids (all params varied):")
print(f"     Percentile: {100 - iron_rank_D:.1f}% (lower = rarer)")
if iron_rank_D < 5:
    print(f"     → SIGNIFICANT: Iron grid beats 95%+ of random grids")
elif iron_rank_D < 25:
    print(f"     → MODERATE: Iron grid is above average but not exceptional")
else:
    print(f"     → NOT SIGNIFICANT: Random grids perform comparably")
print()

print(f"  2. Physics selectivity (does iron favor physics over arbitrary constants?):")
print(f"     Physics mean: {iron_mean_phys:.4f}%, Control mean: {iron_mean_ctrl:.4f}%")
if selective:
    print(f"     → YES: Iron grid indexes physics {diff:.4f}% better than control")
else:
    print(f"     → NO: Iron grid shows no preference for physics constants")
print()

print(f"  3. Scaffold ranking (grain=104, exp=1/φ fixed):")
print(f"     286 ranks #{rank_286}/401 (top {rank_286/401*100:.1f}%)")
if top_scaffold:
    print(f"     → SIGNIFICANT: 286 is one of the best scaffolds for physics")
else:
    print(f"     → NOT SIGNIFICANT: Many scaffolds work as well or better")
print()

print(f"  ╔══════════════════════════════════════════════════════════════╗")
if significant and selective and top_scaffold:
    print(f"  ║  RESULT: The iron anchor IS special for physics indexing.  ║")
    print(f"  ║  The (286, 104, φ) choice outperforms random alternatives ║")
    print(f"  ║  AND shows selectivity for physics over arbitrary numbers. ║")
elif significant or top_scaffold:
    print(f"  ║  RESULT: Partial signal — iron grid performs above average ║")
    print(f"  ║  but the advantage is {'selective' if selective else 'not selective'} for physics constants.  ║")
else:
    print(f"  ║  RESULT: No special advantage — random grids match iron.  ║")
    print(f"  ║  The iron connection is thematic, not functional.         ║")
print(f"  ╚══════════════════════════════════════════════════════════════╝")
print()
