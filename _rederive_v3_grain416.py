#!/usr/bin/env python3
"""
Re-derive all 63 v3 constants for the GOD_CODE algorithm with grain=416.

Old v3: G(a,b,c,d) = 285.999^(1/φ) × (13/12)^((99a + 3032 - b - 99c - 758d) / 758)
New v3: G(a,b,c,d) = 286^(1/φ) × 2^((8a + 1664 - b - 8c - 416d) / 416)

Changes:
  - Scaffold: 285.999 → 286 (original GOD_CODE)
  - Ratio: 13/12 → 2 (original GOD_CODE)
  - Grain: 758 → 416
  - Dial coeff: 99 → 8
  - Offset: 3032 → 1664 (= 4 × 416)
"""

import math
from fractions import Fraction

PHI = (1 + math.sqrt(5)) / 2

# New v3 constants (GOD_CODE algorithm with grain=416)
NEW_SCAFFOLD = 286
NEW_RATIO = 2.0
NEW_GRAIN = 416          # Q
NEW_DIAL_COEFF = 8       # p
NEW_OFFSET = 4 * NEW_GRAIN  # K = 1664
NEW_BASE = NEW_SCAFFOLD ** (1.0 / PHI)  # 286^(1/φ) = 32.969905115578818
NEW_STEP = NEW_RATIO ** (1.0 / NEW_GRAIN)  # 2^(1/416)

print(f"NEW v3 CONSTANTS (GOD_CODE algorithm, grain=416)")
print(f"  Scaffold: {NEW_SCAFFOLD}")
print(f"  Ratio: {NEW_RATIO}")
print(f"  Grain (Q): {NEW_GRAIN}")
print(f"  Dial coeff (p): {NEW_DIAL_COEFF}")
print(f"  Offset (K): {NEW_OFFSET}")
print(f"  Base: {NEW_BASE:.15f}")
print(f"  Step: {NEW_STEP:.15f} (+{(NEW_STEP-1)*100:.6f}%/step)")
print(f"  Max grid error: +/-{(NEW_STEP-1)/2*100:.6f}%")
print(f"  GOD_CODE_V3 = {NEW_BASE * (NEW_RATIO ** (NEW_OFFSET / NEW_GRAIN)):.15f}")
print()


def new_god_code(a, b, c, d):
    """New v3 equation."""
    E = NEW_DIAL_COEFF * a + NEW_OFFSET - b - NEW_DIAL_COEFF * c - NEW_GRAIN * d
    return NEW_BASE * (NEW_RATIO ** (E / NEW_GRAIN))


def solve_exponent(target):
    """Find exact fractional exponent."""
    return NEW_GRAIN * math.log2(target / NEW_BASE)


def find_best_dials(target, max_d=300):
    """Find simplest (a,b,c,d) dials for a target value."""
    if target <= 0:
        return None
    E_exact = solve_exponent(target)
    E_int = round(E_exact)
    
    best = None
    best_cost = float('inf')
    
    for d_val in range(-max_d, max_d + 1):
        remainder = E_int - NEW_OFFSET + NEW_GRAIN * d_val
        # E = 8a + K - b - 8c - Qd
        # remainder = 8a - b - 8c = 8(a-c) - b
        # So: 8(a-c) = remainder + b, where 0 <= b
        # For simplest: minimize |a| + |b| + |c| + |d|
        
        for b_val in range(max(0, -remainder), min(8, max(0, -remainder) + 8)):
            dac = remainder + b_val  # = 8(a-c)
            if dac % 8 != 0:
                continue
            ac = dac // 8  # a - c
            
            if ac >= 0:
                a_val, c_val = ac, 0
            else:
                a_val, c_val = 0, -ac
            
            # Verify
            E_check = NEW_DIAL_COEFF * a_val + NEW_OFFSET - b_val - NEW_DIAL_COEFF * c_val - NEW_GRAIN * d_val
            if E_check != E_int:
                continue
            
            val = new_god_code(a_val, b_val, c_val, d_val)
            err = abs(val - target) / target * 100
            cost = a_val + b_val + c_val + abs(d_val)
            
            if cost < best_cost or (cost == best_cost and best and err < best[5]):
                best = (a_val, b_val, c_val, d_val, E_int, val, err, cost)
                best_cost = cost
    
    return best


# All 63 measured constants from the old v3 frequency table
MEASURED_CONSTANTS = [
    ("SPEED_OF_LIGHT",         299792458),
    ("STANDARD_GRAVITY",       9.80665),
    ("PLANCK_CONSTANT_eVs",    4.135667696e-15),
    ("BOLTZMANN_eV_K",         8.617333262e-5),
    ("ELEMENTARY_CHARGE",      1.602176634e-19),
    ("AVOGADRO",               6.02214076e23),
    ("BOHR_RADIUS_PM",         52.9177210544),
    ("RYDBERG_EV",             13.605693123),
    ("FINE_STRUCTURE_INV",     137.035999084),
    ("COMPTON_PM",             2.42631023867),
    ("CLASSICAL_E_RADIUS_FM",  2.8179403205),
    ("HARTREE_EV",             27.211386246),
    ("MAG_FLUX_QUANTUM_Wb",    2.067833848e-15),
    ("VON_KLITZING_OHM",       25812.80745),
    ("STEFAN_BOLTZMANN",        5.670374419e-8),
    ("VACUUM_PERMITTIVITY",    8.8541878128e-12),
    ("BOHR_MAGNETON_eV_T",     5.7883818060e-5),
    ("ELECTRON_MASS_MEV",      0.51099895069),
    ("MUON_MASS_MEV",          105.6583755),
    ("TAU_MASS_MEV",           1776.86),
    ("PROTON_MASS_MEV",        938.27208816),
    ("NEUTRON_MASS_MEV",       939.56542052),
    ("W_BOSON_GEV",            80.3692),
    ("Z_BOSON_GEV",            91.1876),
    ("HIGGS_GEV",              125.25),
    ("PION_CHARGED_MEV",       139.57039),
    ("PION_NEUTRAL_MEV",       134.9768),
    ("KAON_MEV",               493.677),
    ("D_MESON_MEV",            1869.66),
    ("TOP_QUARK_GEV",          172.57),
    ("BOTTOM_QUARK_GEV",       4.18),
    ("CHARM_QUARK_GEV",        1.27),
    ("FE56_BE_PER_NUCLEON",    8.790),
    ("HE4_BE_PER_NUCLEON",     7.074),
    ("O16_BE_PER_NUCLEON",     7.976),
    ("C12_BE_PER_NUCLEON",     7.680),
    ("U238_BE_PER_NUCLEON",    7.570),
    ("NI62_BE_PER_NUCLEON",    8.7945),
    ("DEUTERON_BE",            2.22457),
    ("TRITON_BE",              8.48182),
    ("FE_BCC_LATTICE_PM",      286.65),
    ("FE_ATOMIC_RADIUS_PM",    126.0),
    ("FE_K_ALPHA1_KEV",        6.404),
    ("FE_IONIZATION_EV",       7.9024678),
    ("CU_LATTICE_PM",          361.49),
    ("AL_LATTICE_PM",          404.95),
    ("SI_LATTICE_PM",          543.102),
    ("EARTH_ORBIT_AU_KM",      149597870.7),
    ("SOLAR_LUMINOSITY_W",     3.828e26),
    ("HUBBLE_CONSTANT",        67.4),
    ("CMB_TEMPERATURE_K",      2.7255),
    ("SOLAR_MASS_KG",          1.98892e30),
    ("SCHUMANN_HZ",            7.83),
    ("ALPHA_EEG_HZ",           10.0),
    ("GAMMA_EEG_HZ",           40.0),
    ("THETA_EEG_HZ",           6.0),
    ("BETA_EEG_HZ",            20.0),
    ("PI",                     3.14159265359),
    ("EULER_E",                2.71828182846),
    ("SQRT2",                  1.41421356237),
    ("GOLDEN_RATIO",           1.618033988749895),
    ("LN2",                    0.69314718056),
    ("PLANCK_LENGTH_M",        1.616255e-35),
    ("OMEGA",                  6539.34712682),
    ("OMEGA_AUTHORITY",        2497.808338),
]

print("RE-DERIVED DIAL SETTINGS FOR GRAIN=416")
print("=" * 100)
print(f"{'Name':30s}  {'(a,b,c,d)':20s}  {'Exponent':>10s}  {'Grid Value':>14s}  {'Measured':>14s}  {'Error%':>8s}")
print("-" * 100)

results = []
fail_count = 0
total_err = 0
max_err = 0

for name, measured in MEASURED_CONSTANTS:
    best = find_best_dials(measured)
    if best is None:
        print(f"  {name:30s}  FAILED")
        fail_count += 1
        continue
    
    a, b, c, d, E, val, err, cost = best
    total_err += err
    max_err = max(max_err, err)
    
    results.append((name, a, b, c, d, E, val, measured, err))
    print(f"  {name:30s}  ({a:4d},{b:3d},{c:4d},{d:4d})  {E:10d}  {val:14.6e}  {measured:14.6e}  {err:7.4f}%")

print("-" * 100)
print(f"  Constants: {len(results)}/{len(MEASURED_CONSTANTS)}")
print(f"  Mean error: {total_err/len(results):.4f}%")
print(f"  Max error: {max_err:.4f}%")
print(f"  Max theoretical: +/-{(NEW_STEP-1)/2*100:.4f}%")
print()

# Generate the Python code for the new frequency table
print()
print("=" * 100)
print("GENERATED PYTHON CODE FOR V3_FREQUENCY_TABLE")
print("=" * 100)
print()
print("V3_FREQUENCY_TABLE = {")
print("    # (a, b, c, d): (name, grid_value, exponent, measured, error_pct)")

for name, a, b, c, d, E, val, measured, err in results:
    # Format grid value in scientific notation
    if abs(val) >= 1e6 or abs(val) < 1e-4:
        gv_str = f"{val:.6e}"
    else:
        gv_str = f"{val:.6f}"
    
    # Format measured
    if abs(measured) >= 1e6 or abs(measured) < 1e-4:
        mv_str = f"{measured}"
    else:
        mv_str = f"{measured}"
    
    pad_name = f'"{name}"'
    print(f"    ({a}, {b}, {c}, {d}):{' '*(25-len(f'({a}, {b}, {c}, {d}):'))}({pad_name:32s}, {gv_str:>14s}, {E:>8d}, {mv_str:>20s}, {err:>10.4f}),")

print("}")
print(f'V3_FREQUENCY_TABLE[(0, 0, 0, 0)] = ("GOD_CODE_V3", GOD_CODE_V3, K_V3, GOD_CODE_V3, 0.0)')
print()

# Also output the new GOD_CODE_V3
new_gc_v3 = new_god_code(0, 0, 0, 0)
print(f"NEW GOD_CODE_V3 = {new_gc_v3:.15f}")
print(f"OLD GOD_CODE_V3 = 45.41141298077539 (approx)")
print()

# Compare old vs new error stats
print("COMPARISON: Old v3 (grain=758) vs New v3 (grain=416)")
print("-" * 60)
old_max_step = (13/12) ** (1/758)
new_max_step_val = 2 ** (1/416)
print(f"  Old step: {old_max_step:.10f} (+/-{(old_max_step-1)/2*100:.6f}%)")
print(f"  New step: {new_max_step_val:.10f} (+/-{(new_max_step_val-1)/2*100:.6f}%)")
print(f"  Old precision: +/-{(old_max_step-1)/2*100:.6f}%")
print(f"  New precision: +/-{(new_max_step_val-1)/2*100:.6f}%")
print(f"  Precision ratio (old/new): {((old_max_step-1)/2) / ((new_max_step_val-1)/2):.2f}x")
print(f"  New mean error: {total_err/len(results):.4f}%")
print()
