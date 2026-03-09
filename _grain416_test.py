#!/usr/bin/env python3
"""Test grain=416 — the offset constant from G(a,b,c,d) = 286^(1/phi) * 2^((8a+416-b-8c-104d)/104)."""

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
print("GRAIN=416 TEST: The offset constant from the GOD_CODE equation")
print("416 = 104 x 4 = 26 x 16 = 8 x 52")
print(sep)
print()

# ─── Compare all key grains from the equation ───
grids = [
    ("grain=13 (F7)",       286, 13,  1/PHI),
    ("grain=26 (Fe Z)",     286, 26,  1/PHI),
    ("grain=52 (26x2)",     286, 52,  1/PHI),
    ("grain=104 (L104)",    286, 104, 1/PHI),
    ("grain=208 (104x2)",   286, 208, 1/PHI),
    ("grain=416 (offset)",  286, 416, 1/PHI),
    ("grain=832 (416x2)",   286, 832, 1/PHI),
    ("grain=8 (dial step)", 286, 8,   1/PHI),
]

header = f"{'Grid':22s}  {'Grain':>5s}  {'Phys':>8s}  {'Ctrl':>8s}  {'Delta':>8s}  {'Max err':>8s}  {'Norm ratio':>10s}"
print(header)
print("-" * 80)
for name, s, g, e in grids:
    p = mean_err(PHYSICAL_CONSTANTS, s, g, e)
    c = mean_err(CONTROL_CONSTANTS, s, g, e)
    max_err = (2**(1/g) - 1) / 2 * 100
    norm = p / max_err if max_err > 0 else 0
    print(f"{name:22s}  {g:5d}  {p:7.4f}%  {c:7.4f}%  {c-p:+7.4f}%  {max_err:7.4f}%  {norm:10.4f}")

print()
print("Norm ratio: actual_mean / theoretical_max (0.5 = random expectation)")
print()

# ─── Deep dive on grain=416 ───
iron416_phys = mean_err(PHYSICAL_CONSTANTS, 286, 416, 1/PHI)
iron416_ctrl = mean_err(CONTROL_CONSTANTS, 286, 416, 1/PHI)
max416 = (2**(1/416) - 1) / 2 * 100

print(f"GRAIN=416 DETAILS")
print("-" * 60)
print(f"  Base: 286^(1/phi) = {286**(1/PHI):.6f}")
print(f"  Step: 2^(1/416) = {2**(1/416):.10f} (+{(2**(1/416)-1)*100:.4f}%/step)")
print(f"  Max grid error: +/-{max416:.4f}%")
print(f"  Physics mean: {iron416_phys:.4f}%")
print(f"  Control mean: {iron416_ctrl:.4f}%")
print(f"  Selectivity: {iron416_ctrl - iron416_phys:+.4f}%")
print(f"  Normalized ratio: {iron416_phys / max416:.4f}")
print()

# ─── Individual constants ───
print("INDIVIDUAL CONSTANTS on (286, 416, 1/phi):")
print("-" * 60)
for name, v in sorted(PHYSICAL_CONSTANTS.items(), key=lambda x: grid_error(x[1], 286, 416, 1/PHI)):
    err = grid_error(v, 286, 416, 1/PHI) * 100
    print(f"  {name:30s}  {err:.4f}%")
print(f"  Mean: {iron416_phys:.4f}%")
print()

# ─── Random comparison ───
print("RANDOM COMPARISON for grain=416")
print("-" * 60)
rng = np.random.RandomState(42)

# A. Fix grain=416, vary scaffold
rand_scaffold = []
for _ in range(10000):
    s = rng.randint(10, 501)
    rand_scaffold.append(mean_err(PHYSICAL_CONSTANTS, s, 416, 1/PHI))
rank_A = np.sum(np.array(rand_scaffold) < iron416_phys) / 100
print(f"A. Scaffold varied (grain=416, exp=1/phi):")
print(f"   Iron-416: {iron416_phys:.4f}%, Random mean: {np.mean(rand_scaffold):.4f}%")
print(f"   Percentile: {100 - rank_A:.1f}% (lower=better)")
print()

# B. Fix grain=416, scaffold=286, vary exponent
rand_exp = []
for _ in range(10000):
    e = rng.uniform(0.3, 3.0)
    rand_exp.append(mean_err(PHYSICAL_CONSTANTS, 286, 416, e))
rank_B2 = np.sum(np.array(rand_exp) < iron416_phys) / 100
print(f"B. Exponent varied (scaffold=286, grain=416):")
print(f"   1/phi: {iron416_phys:.4f}%, Random mean: {np.mean(rand_exp):.4f}%")
print(f"   Percentile: {100 - rank_B2:.1f}% (lower=better)")
print()

# C. Vary all three (grain in high range 100-800)
rand_all = []
for _ in range(10000):
    s = rng.randint(10, 501)
    g = rng.randint(100, 801)
    e = rng.uniform(0.3, 3.0)
    rand_all.append(mean_err(PHYSICAL_CONSTANTS, s, g, e))
rank_C = np.sum(np.array(rand_all) < iron416_phys) / 100
print(f"C. All varied (scaffold 10-500, grain 100-800, exp 0.3-3.0):")
print(f"   Iron-416: {iron416_phys:.4f}%, Random mean: {np.mean(rand_all):.4f}%")
print(f"   Percentile: {100 - rank_C:.1f}% (lower=better)")
print()

# D. Physics selectivity
iron416_delta = iron416_ctrl - iron416_phys
sel_count = 0
for _ in range(2000):
    s = rng.randint(10, 501)
    g = rng.randint(100, 801)
    e = rng.uniform(0.3, 3.0)
    pd = mean_err(CONTROL_CONSTANTS, s, g, e) - mean_err(PHYSICAL_CONSTANTS, s, g, e)
    if pd > iron416_delta:
        sel_count += 1
sel_pct = 100 - sel_count / 20
print(f"D. Physics selectivity:")
print(f"   Iron-416 delta (ctrl-phys): {iron416_delta:+.4f}%")
print(f"   Random grids with greater preference: {sel_count}/2000")
print(f"   Selectivity percentile: {sel_pct:.1f}%")
print()

# E. Best scaffold for grain=416
results_416 = [(s, mean_err(PHYSICAL_CONSTANTS, s, 416, 1/PHI)) for s in range(10, 501)]
results_416.sort(key=lambda x: x[1])
print(f"E. Best scaffolds for grain=416, exp=1/phi:")
for s, err in results_416[:10]:
    marker = " <-- L104" if s == 286 else ""
    print(f"   scaffold={s:4d}  mean_err={err:.4f}%{marker}")
rank_286 = next(i for i, (s, _) in enumerate(results_416) if s == 286) + 1
print(f"   286 ranks #{rank_286} out of {len(results_416)} (top {rank_286/len(results_416)*100:.1f}%)")
print()

# F. The KEY test: does 416 have better normalized fit than other grains?
print("NORMALIZED FIT COMPARISON (the real test):")
print("-" * 60)
print("If physics constants CLUSTER near grid points at grain=416")
print("more than at other grains, the norm ratio will be lower.")
print()
for g in [8, 13, 26, 52, 104, 208, 416, 832]:
    p = mean_err(PHYSICAL_CONSTANTS, 286, g, 1/PHI)
    c = mean_err(CONTROL_CONSTANTS, 286, g, 1/PHI)
    mx = (2**(1/g) - 1) / 2 * 100
    norm_p = p / mx
    norm_c = c / mx
    marker = " ***" if g == 416 else ""
    print(f"  grain={g:4d}  phys_norm={norm_p:.4f}  ctrl_norm={norm_c:.4f}  delta={norm_c-norm_p:+.4f}{marker}")

print()
print(sep)
print("VERDICT")
print(sep)
norm416 = iron416_phys / max416
print(f"  grain=416 normalized ratio: {norm416:.4f} (0.5 = random)")
print(f"  grain=104 normalized ratio: {mean_err(PHYSICAL_CONSTANTS, 286, 104, 1/PHI) / ((2**(1/104)-1)/2*100):.4f}")
print(f"  Physics selectivity: {sel_pct:.1f}th percentile")
print(f"  Scaffold 286 rank: #{rank_286}/{len(results_416)}")
print()
