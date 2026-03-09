#!/usr/bin/env python3
"""Correlation analysis: mass constants vs non-mass constants on GOD_CODE v3 grid (P=64)."""
import math
import statistics

PHI = 1.618033988749895
X = 286; R = 2; Q = 416; P = 64; K = 1664
BASE = X ** (1.0 / PHI)

def solve_E(target):
    return Q * math.log(target / BASE) / math.log(R)

def snap(target):
    E_exact = solve_E(target)
    E_int = round(E_exact)
    val = BASE * (R ** (E_int / Q))
    err = abs(val - target) / target * 100
    return E_int, val, err

# Categorize all 65 constants
MASS = {
    "ELECTRON_MASS_MEV":     0.51099895069,
    "MUON_MASS_MEV":         105.6583755,
    "TAU_MASS_MEV":          1776.86,
    "PROTON_MASS_MEV":       938.27208816,
    "NEUTRON_MASS_MEV":      939.56542052,
    "W_BOSON_GEV":           80.3692,
    "Z_BOSON_GEV":           91.1876,
    "HIGGS_GEV":             125.25,
    "PION_CHARGED_MEV":      139.57039,
    "PION_NEUTRAL_MEV":      134.9768,
    "KAON_MEV":              493.677,
    "D_MESON_MEV":           1869.66,
    "TOP_QUARK_GEV":         172.57,
    "BOTTOM_QUARK_GEV":      4.18,
    "CHARM_QUARK_GEV":       1.27,
    "SOLAR_MASS_KG":         1.98892e30,
}

BINDING_ENERGY = {
    "FE56_BE_PER_NUCLEON":   8.790,
    "HE4_BE_PER_NUCLEON":    7.074,
    "O16_BE_PER_NUCLEON":    7.976,
    "C12_BE_PER_NUCLEON":    7.680,
    "U238_BE_PER_NUCLEON":   7.570,
    "NI62_BE_PER_NUCLEON":   8.7945,
    "DEUTERON_BE":           2.22457,
    "TRITON_BE":             8.48182,
}

FUNDAMENTAL = {
    "SPEED_OF_LIGHT":        299792458,
    "STANDARD_GRAVITY":      9.80665,
    "PLANCK_CONSTANT_eVs":   4.135667696e-15,
    "BOLTZMANN_eV_K":        8.617333262e-5,
    "ELEMENTARY_CHARGE":     1.602176634e-19,
    "AVOGADRO":              6.02214076e23,
    "PLANCK_LENGTH_M":       1.616255e-35,
}

ATOMIC = {
    "BOHR_RADIUS_PM":        52.9177210544,
    "RYDBERG_EV":            13.605693123,
    "FINE_STRUCTURE_INV":    137.035999084,
    "COMPTON_PM":            2.42631023867,
    "CLASSICAL_E_RADIUS_FM": 2.8179403205,
    "HARTREE_EV":            27.211386246,
    "MAG_FLUX_QUANTUM_Wb":   2.067833848e-15,
    "VON_KLITZING_OHM":      25812.80745,
    "STEFAN_BOLTZMANN":       5.670374419e-8,
    "VACUUM_PERMITTIVITY":    8.8541878128e-12,
    "BOHR_MAGNETON_eV_T":    5.7883818060e-5,
}

IRON_CRYSTAL = {
    "FE_BCC_LATTICE_PM":     286.65,
    "FE_ATOMIC_RADIUS_PM":   126.0,
    "FE_K_ALPHA1_KEV":       6.404,
    "FE_IONIZATION_EV":      7.9024678,
    "CU_LATTICE_PM":         361.49,
    "AL_LATTICE_PM":         404.95,
    "SI_LATTICE_PM":         543.102,
}

ASTRO = {
    "EARTH_ORBIT_AU_KM":     149597870.7,
    "SOLAR_LUMINOSITY_W":    3.828e26,
    "HUBBLE_CONSTANT":       67.4,
    "CMB_TEMPERATURE_K":     2.7255,
}

RESONANCE = {
    "SCHUMANN_HZ":           7.83,
    "ALPHA_EEG_HZ":          10.0,
    "GAMMA_EEG_HZ":          40.0,
    "THETA_EEG_HZ":          6.0,
    "BETA_EEG_HZ":           20.0,
}

MATH_CONST = {
    "PI":                    3.14159265359,
    "EULER_E":               2.71828182846,
    "SQRT2":                 1.41421356237,
    "GOLDEN_RATIO":          1.618033988749895,
    "LN2":                   0.69314718056,
}

SOVEREIGN = {
    "OMEGA":                 6539.34712682,
    "OMEGA_AUTHORITY":       2497.808338,
}

categories = {
    "Mass (16)":        MASS,
    "Binding E (8)":    BINDING_ENERGY,
    "Fundamental (7)":  FUNDAMENTAL,
    "Atomic (11)":      ATOMIC,
    "Iron/Crystal (7)": IRON_CRYSTAL,
    "Astro (4)":        ASTRO,
    "Resonance (5)":    RESONANCE,
    "Math (5)":         MATH_CONST,
    "Sovereign (2)":    SOVEREIGN,
}

print("=" * 80)
print("GOD_CODE v3 (grain=416, P=64) — CATEGORY CORRELATION ANALYSIS")
print("=" * 80)

# Per-category error stats
print(f"\n{'Category':<20s} {'N':>3s} {'Mean%':>8s} {'Med%':>8s} {'Max%':>8s} {'Min%':>8s} {'StdDev':>8s}")
print("-" * 65)

cat_errors = {}
all_errors = []
for cat_name, constants in categories.items():
    errs = []
    for name, val in constants.items():
        _, _, err = snap(val)
        errs.append(err)
        all_errors.append((err, name, cat_name))
    cat_errors[cat_name] = errs
    avg = statistics.mean(errs)
    med = statistics.median(errs)
    mx = max(errs)
    mn = min(errs)
    sd = statistics.stdev(errs) if len(errs) > 1 else 0
    print(f"{cat_name:<20s} {len(errs):>3d} {avg:>8.4f} {med:>8.4f} {mx:>8.4f} {mn:>8.4f} {sd:>8.4f}")

# Overall
all_err_vals = [e for e, _, _ in all_errors]
print(f"\n{'ALL (65)':<20s} {len(all_err_vals):>3d} {statistics.mean(all_err_vals):>8.4f} "
      f"{statistics.median(all_err_vals):>8.4f} {max(all_err_vals):>8.4f} "
      f"{min(all_err_vals):>8.4f} {statistics.stdev(all_err_vals):>8.4f}")

# Mass vs non-mass comparison
mass_errs = cat_errors["Mass (16)"]
non_mass_errs = [e for cat, errs in cat_errors.items() if cat != "Mass (16)" for e in errs]

print("\n" + "=" * 80)
print("MASS vs NON-MASS COMPARISON")
print("=" * 80)
print(f"  Mass constants (16):     mean={statistics.mean(mass_errs):.4f}%  median={statistics.median(mass_errs):.4f}%  max={max(mass_errs):.4f}%")
print(f"  Non-mass constants (49): mean={statistics.mean(non_mass_errs):.4f}%  median={statistics.median(non_mass_errs):.4f}%  max={max(non_mass_errs):.4f}%")
print(f"  Ratio (mass/non-mass):   {statistics.mean(mass_errs)/statistics.mean(non_mass_errs):.3f}×")

# Is the difference statistically significant? Simple permutation test
import random
random.seed(104)
observed_diff = statistics.mean(mass_errs) - statistics.mean(non_mass_errs)
combined = mass_errs + non_mass_errs
n_mass = len(mass_errs)
n_perm = 10000
count_ge = 0
for _ in range(n_perm):
    random.shuffle(combined)
    perm_mass = combined[:n_mass]
    perm_non = combined[n_mass:]
    perm_diff = statistics.mean(perm_mass) - statistics.mean(perm_non)
    if perm_diff >= observed_diff:
        count_ge += 1
p_value = count_ge / n_perm
print(f"\n  Permutation test (10K trials):")
print(f"    Observed diff (mass - non-mass): {observed_diff:+.4f}%")
print(f"    p-value: {p_value:.4f}")
print(f"    Significant at 0.05: {'YES' if p_value < 0.05 else 'NO'}")

# Exponent magnitude analysis — do masses cluster in a particular E range?
print("\n" + "=" * 80)
print("EXPONENT (E) DISTRIBUTION BY CATEGORY")
print("=" * 80)
print(f"\n{'Category':<20s} {'Mean E':>10s} {'Med E':>10s} {'Range':>20s} {'|E| mean':>10s}")
print("-" * 75)

for cat_name, constants in categories.items():
    Es = [round(solve_E(v)) for v in constants.values()]
    abs_Es = [abs(e) for e in Es]
    me = statistics.mean(Es)
    med = statistics.median(Es)
    rng = f"[{min(Es)}, {max(Es)}]"
    abs_me = statistics.mean(abs_Es)
    print(f"{cat_name:<20s} {me:>10.0f} {med:>10.0f} {rng:>20s} {abs_me:>10.0f}")

# Grid residual pattern — are mass errors randomly distributed or biased?
print("\n" + "=" * 80)
print("GRID RESIDUAL ANALYSIS (fractional part of E_exact)")
print("=" * 80)
print("  If residuals are uniform in [0,1), errors are purely from grid density.")
print("  Clustering near 0 or 0.5 would indicate structural alignment.\n")

for cat_name, constants in categories.items():
    residuals = []
    for val in constants.values():
        E_exact = solve_E(val)
        frac = E_exact - round(E_exact)  # in [-0.5, 0.5]
        residuals.append(frac)
    abs_residuals = [abs(r) for r in residuals]
    avg_abs_res = statistics.mean(abs_residuals)
    # Expected for uniform: 0.25
    print(f"  {cat_name:<20s}: mean|residual|={avg_abs_res:.4f}  (uniform expectation: 0.2500)")

# Degeneracy check — which constants share the same grid point?
print("\n" + "=" * 80)
print("GRID-POINT DEGENERACIES (same E_int → same grid value)")
print("=" * 80)

all_grid = {}
for cat_name, constants in categories.items():
    for name, val in constants.items():
        E_int = round(solve_E(val))
        if E_int not in all_grid:
            all_grid[E_int] = []
        all_grid[E_int].append((name, val, cat_name))

degen_count = 0
for E_int, entries in sorted(all_grid.items()):
    if len(entries) > 1:
        degen_count += 1
        names = ", ".join(f"{n} ({c})" for n, _, c in entries)
        grid_val = BASE * (R ** (E_int / Q))
        print(f"  E={E_int:>7d}  grid={grid_val:.4e}  →  {names}")

if degen_count == 0:
    print("  No degeneracies found.")
else:
    print(f"\n  Total degenerate grid points: {degen_count}")

# Cross-category error correlation with log(value)
print("\n" + "=" * 80)
print("ERROR vs LOG(VALUE) CORRELATION")
print("=" * 80)

log_vals = []
err_vals = []
for cat_name, constants in categories.items():
    for name, val in constants.items():
        _, _, err = snap(val)
        log_vals.append(math.log10(val))
        err_vals.append(err)

# Pearson correlation
n = len(log_vals)
mean_x = sum(log_vals) / n
mean_y = sum(err_vals) / n
cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_vals, err_vals)) / n
std_x = (sum((x - mean_x)**2 for x in log_vals) / n) ** 0.5
std_y = (sum((y - mean_y)**2 for y in err_vals) / n) ** 0.5
r = cov / (std_x * std_y) if std_x > 0 and std_y > 0 else 0

print(f"  Pearson r(log10(value), error%): {r:.4f}")
print(f"  Interpretation: {'no' if abs(r) < 0.1 else 'weak' if abs(r) < 0.3 else 'moderate' if abs(r) < 0.5 else 'strong'} correlation")
print(f"  → Grid error is {'independent of' if abs(r) < 0.2 else 'weakly correlated with' if abs(r) < 0.4 else 'correlated with'} the magnitude of the constant")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
