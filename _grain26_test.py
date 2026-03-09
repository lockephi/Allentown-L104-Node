#!/usr/bin/env python3
"""Test grain=26 (Fe atomic number) as the simpler GOD_CODE grain."""

import math
import numpy as np

PHI = (1 + math.sqrt(5)) / 2

PHYSICAL_CONSTANTS = {
    "Bohr radius (pm)": 52.9177210544,
    "Rydberg energy (eV)": 13.605693123,
    "Compton wavelength (pm)": 2.42631023867,
    "1/alpha (fine structure)": 137.035999084,
    "Electron mass (MeV)": 0.51099895069,
    "Fe BCC lattice (pm)": 286.65,
    "Fe atomic radius (pm)": 126.0,
    "Fe-56 BE/A (MeV)": 8.790,
    "He-4 BE/A (MeV)": 7.074,
    "Fe Ka1 (keV)": 6.404,
    "Proton mass (u)": 1.007276466621,
    "Neutron mass (u)": 1.00866491595,
    "Muon mass (MeV)": 105.6583755,
    "Higgs mass (GeV)": 125.25,
    "W boson mass (GeV)": 80.377,
    "Z boson mass (GeV)": 91.1876,
    "Speed of light (Mm/s)": 299.792458,
    "Boltzmann (x1e23 J/K)": 1.380649,
    "Avogadro (x1e23)": 6.02214076,
    "Schumann resonance (Hz)": 7.83,
    "Earth orbit (Gm)": 149.598,
    "Planck length (x1e35 m)": 1.616255,
}

CONTROL_CONSTANTS = {
    "Pi": 3.14159265358979,
    "e (Euler)": 2.71828182845905,
    "ln(2)": 0.693147180559945,
    "sqrt(2)": 1.41421356237310,
    "Feigenbaum": 4.66920160910299,
    "Apery zeta(3)": 1.20205690315959,
    "Catalan G": 0.91596559417722,
    "Plastic number": 1.32471795724475,
    "Coffee price": 5.50,
    "Days in year": 365.25,
    "Miles per km": 0.621371,
    "Bitcoin ATH (k)": 109.0,
}


def grid_error(value, scaffold, grain, exponent):
    base = scaffold ** exponent
    if base <= 0 or value <= 0:
        return 1.0
    E_exact = grain * math.log2(value / base)
    E_nearest = round(E_exact)
    grid_value = base * (2 ** (E_nearest / grain))
    return abs(grid_value - value) / value


def mean_err(consts, s, g, e):
    return np.mean([grid_error(v, s, g, e) for v in consts.values()]) * 100


sep = "=" * 72
print(sep)
print("GRAIN=26 TEST: Does the simpler iron grain work better?")
print(sep)
print()

# ─── Grid comparison table ───
grids = [
    ("L104 original",    286, 104, 1/PHI),
    ("Fe Z=26 grain",    286, 26,  1/PHI),
    ("Fe Z=26 + 1/2",    286, 26,  0.5),
    ("Fe Z=26 + 1/e",    286, 26,  1/math.e),
    ("Fe Z=26 + int",    286, 26,  1.0),
    ("Fe Z=26 + 2",      286, 26,  2.0),
    ("13 grain (F7)",    286, 13,  1/PHI),
    ("52 grain (26x2)",  286, 52,  1/PHI),
    ("Pure Fe (26,26)",  26,  26,  1/PHI),
    ("Fe (56,26)",       56,  26,  1/PHI),
]

header = f"{'Grid':22s}  {'Grain':>5s}  {'Phys':>8s}  {'Ctrl':>8s}  {'Delta':>8s}  {'Max grid err':>12s}"
print(header)
print("-" * 72)
for name, s, g, e in grids:
    p = mean_err(PHYSICAL_CONSTANTS, s, g, e)
    c = mean_err(CONTROL_CONSTANTS, s, g, e)
    max_err = (2**(1/g) - 1) / 2 * 100
    marker = " *" if name == "L104 original" else ""
    print(f"{name:22s}  {g:5d}  {p:7.4f}%  {c:7.4f}%  {c-p:+7.4f}%  {max_err:10.4f}%{marker}")

print()
print("Theoretical max grid error = (2^(1/grain) - 1) / 2")
print("  grain=104 -> +/-0.334%")
print("  grain=26  -> +/-1.350%")
print("  grain=13  -> +/-2.740%")
print()

# ─── Random comparison for grain=26 specifically ───
print("RANDOM COMPARISON for grain=26 grids")
print("-" * 60)
rng = np.random.RandomState(42)

iron26_phys = mean_err(PHYSICAL_CONSTANTS, 286, 26, 1/PHI)
iron26_ctrl = mean_err(CONTROL_CONSTANTS, 286, 26, 1/PHI)

# A. Fix grain=26, exp=1/phi, vary scaffold
rand_scaffold = []
for _ in range(10000):
    s = rng.randint(10, 501)
    rand_scaffold.append(mean_err(PHYSICAL_CONSTANTS, s, 26, 1/PHI))
pct_better_A = np.sum(np.array(rand_scaffold) < iron26_phys) / 100
print(f"A. Scaffold varied (grain=26, exp=1/phi):")
print(f"   Iron-26: {iron26_phys:.4f}%, Random mean: {np.mean(rand_scaffold):.4f}%")
print(f"   Better than {100 - pct_better_A:.1f}% of random scaffolds")
print()

# B. Vary all three params but with grain in [10, 50] range
rand_all = []
for _ in range(10000):
    s = rng.randint(10, 501)
    g = rng.randint(10, 51)
    e = rng.uniform(0.3, 3.0)
    rand_all.append(mean_err(PHYSICAL_CONSTANTS, s, g, e))

pct_better_B = np.sum(np.array(rand_all) < iron26_phys) / 100
print(f"B. All varied (scaffold 10-500, grain 10-50, exp 0.3-3.0):")
print(f"   Iron-26: {iron26_phys:.4f}%, Random mean: {np.mean(rand_all):.4f}%")
print(f"   Better than {100 - pct_better_B:.1f}% of random grids")
print()

# C. Selectivity: does (286,26,1/phi) favor physics?
sel_count = 0
iron26_delta = iron26_ctrl - iron26_phys
for _ in range(1000):
    s = rng.randint(10, 501)
    g = rng.randint(10, 51)
    e = rng.uniform(0.3, 3.0)
    pd = mean_err(CONTROL_CONSTANTS, s, g, e) - mean_err(PHYSICAL_CONSTANTS, s, g, e)
    if pd > iron26_delta:
        sel_count += 1

print(f"C. Physics selectivity:")
print(f"   Iron-26 delta (ctrl-phys): {iron26_delta:+.4f}%")
print(f"   Random grids with greater preference: {sel_count}/1000")
print(f"   Selectivity percentile: {100 - sel_count/10:.1f}%")
print()

# D. Grain=26 specific: best scaffold?
results_26 = [(s, mean_err(PHYSICAL_CONSTANTS, s, 26, 1/PHI)) for s in range(10, 501)]
results_26.sort(key=lambda x: x[1])
print(f"D. Best scaffolds for grain=26, exp=1/phi:")
for s, err in results_26[:15]:
    marker = " <-- L104" if s == 286 else ""
    print(f"   scaffold={s:4d}  mean_err={err:.4f}%{marker}")
rank_286 = next(i for i, (s, _) in enumerate(results_26) if s == 286) + 1
print(f"   286 ranks #{rank_286} out of {len(results_26)} (top {rank_286/len(results_26)*100:.1f}%)")
print()

# E. Individual constants for grain=26
print("INDIVIDUAL CONSTANTS on (286, 26, 1/phi):")
print("-" * 60)
for name, v in sorted(PHYSICAL_CONSTANTS.items(), key=lambda x: grid_error(x[1], 286, 26, 1/PHI)):
    err = grid_error(v, 286, 26, 1/PHI) * 100
    print(f"  {name:30s}  {err:.4f}%")
print(f"  Mean: {iron26_phys:.4f}%")
print()

# F. Key question: does grain=26 make the IRON constants closer?
print("IRON-SPECIFIC CONSTANTS — grain=26 vs grain=104:")
print("-" * 60)
iron_consts = {
    "Fe BCC lattice (pm)": 286.65,
    "Fe atomic radius (pm)": 126.0,
    "Fe-56 BE/A (MeV)": 8.790,
    "Fe Ka1 (keV)": 6.404,
    "He-4 BE/A (MeV)": 7.074,
}
for name, v in iron_consts.items():
    e104 = grid_error(v, 286, 104, 1/PHI) * 100
    e26 = grid_error(v, 286, 26, 1/PHI) * 100
    if abs(e26 - e104) < 1e-6:
        better = "<-- TIE"
    elif e26 < e104:
        better = "<-- 26 wins"
    else:
        better = "<-- 104 wins"
    print(f"  {name:25s}  grain=104: {e104:.4f}%  grain=26: {e26:.4f}%  {better}")
print()

# G. Normalize by theoretical max — what fraction of max error is used?
print("NORMALIZED ERROR (actual / theoretical max):")
print("-" * 60)
max104 = (2**(1/104) - 1) / 2 * 100
max26 = (2**(1/26) - 1) / 2 * 100
norm104 = mean_err(PHYSICAL_CONSTANTS, 286, 104, 1/PHI) / max104
norm26 = mean_err(PHYSICAL_CONSTANTS, 286, 26, 1/PHI) / max26
print(f"  grain=104: mean={mean_err(PHYSICAL_CONSTANTS, 286, 104, 1/PHI):.4f}%, max={max104:.4f}%, ratio={norm104:.4f}")
print(f"  grain=26:  mean={iron26_phys:.4f}%, max={max26:.4f}%, ratio={norm26:.4f}")
print(f"  A ratio of 0.5 is expected for uniformly distributed errors")
print(f"  Lower ratio = constants cluster near grid points")
if norm26 < norm104:
    print(f"  -> grain=26 has BETTER normalized fit ({norm26:.4f} vs {norm104:.4f})")
elif norm26 > norm104:
    print(f"  -> grain=104 has BETTER normalized fit ({norm104:.4f} vs {norm26:.4f})")
else:
    print(f"  -> Both grains have identical normalized fit")
print()

print(sep)
print("VERDICT")
print(sep)
print()
if pct_better_A < 5 or pct_better_B < 5:
    print(f"  (286, 26, 1/phi) IS statistically significant for physics indexing.")
    print(f"  (top {100 - pct_better_A:.1f}% vs scaffold-only, top {100 - pct_better_B:.1f}% vs all-params)")
else:
    print(f"  (286, 26, 1/phi) is NOT statistically significant vs random grids.")
    print(f"  (top {100 - pct_better_A:.1f}% vs scaffold-only, top {100 - pct_better_B:.1f}% vs all-params)")
print(f"  Normalized error ratio (grain=26): {norm26:.4f} (0.5 = expected)")
print(f"  Normalized error ratio (grain=104): {norm104:.4f}")
print(f"  Physics selectivity: {100 - sel_count/10:.1f}th percentile")
print()
