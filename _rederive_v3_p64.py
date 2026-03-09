#!/usr/bin/env python3
"""Re-derive all 65 constants for GOD_CODE algorithm with grain=416, dial_coeff=64."""
import math

PHI = 1.618033988749895
X = 286
R = 2
Q = 416       # grain
P = 64        # dial_coeff (was 8, now 64)
K = 1664      # offset = 4 * Q
BASE = X ** (1.0 / PHI)
STEP = R ** (1.0 / Q)

print(f"GOD_CODE Algorithm v3 (grain={Q}, dial_coeff={P})")
print(f"G_v3(a,b,c,d) = {X}^(1/φ) × 2^(({P}a + {K} - b - {P}c - {Q}d) / {Q})")
print(f"BASE = {BASE}")
print(f"STEP = {STEP}")
print(f"b range per a-step: 0..{P-1}")
print()

CONSTANTS = {
    "SPEED_OF_LIGHT":       299792458,
    "STANDARD_GRAVITY":     9.80665,
    "PLANCK_CONSTANT_eVs":  4.135667696e-15,
    "BOLTZMANN_eV_K":       8.617333262e-5,
    "ELEMENTARY_CHARGE":    1.602176634e-19,
    "AVOGADRO":             6.02214076e23,
    "BOHR_RADIUS_PM":       52.9177210544,
    "RYDBERG_EV":           13.605693123,
    "FINE_STRUCTURE_INV":   137.035999084,
    "COMPTON_PM":           2.42631023867,
    "CLASSICAL_E_RADIUS_FM": 2.8179403205,
    "HARTREE_EV":           27.211386246,
    "MAG_FLUX_QUANTUM_Wb":  2.067833848e-15,
    "VON_KLITZING_OHM":     25812.80745,
    "STEFAN_BOLTZMANN":      5.670374419e-8,
    "VACUUM_PERMITTIVITY":   8.8541878128e-12,
    "BOHR_MAGNETON_eV_T":    5.7883818060e-5,
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
    "FE56_BE_PER_NUCLEON":   8.790,
    "HE4_BE_PER_NUCLEON":    7.074,
    "O16_BE_PER_NUCLEON":    7.976,
    "C12_BE_PER_NUCLEON":    7.680,
    "U238_BE_PER_NUCLEON":   7.570,
    "NI62_BE_PER_NUCLEON":   8.7945,
    "DEUTERON_BE":           2.22457,
    "TRITON_BE":             8.48182,
    "FE_BCC_LATTICE_PM":     286.65,
    "FE_ATOMIC_RADIUS_PM":   126.0,
    "FE_K_ALPHA1_KEV":       6.404,
    "FE_IONIZATION_EV":      7.9024678,
    "CU_LATTICE_PM":         361.49,
    "AL_LATTICE_PM":         404.95,
    "SI_LATTICE_PM":         543.102,
    "EARTH_ORBIT_AU_KM":     149597870.7,
    "SOLAR_LUMINOSITY_W":    3.828e26,
    "HUBBLE_CONSTANT":       67.4,
    "CMB_TEMPERATURE_K":     2.7255,
    "SOLAR_MASS_KG":         1.98892e30,
    "SCHUMANN_HZ":           7.83,
    "ALPHA_EEG_HZ":          10.0,
    "GAMMA_EEG_HZ":          40.0,
    "THETA_EEG_HZ":          6.0,
    "BETA_EEG_HZ":           20.0,
    "PI":                    3.14159265359,
    "EULER_E":               2.71828182846,
    "SQRT2":                 1.41421356237,
    "GOLDEN_RATIO":          1.618033988749895,
    "LN2":                   0.69314718056,
    "PLANCK_LENGTH_M":       1.616255e-35,
    "OMEGA":                 6539.34712682,
    "OMEGA_AUTHORITY":       2497.808338,
}

def solve_E(target):
    return Q * math.log(target / BASE) / math.log(R)

def find_best_dials(target, max_d=300):
    E_exact = solve_E(target)
    E_int = round(E_exact)
    best = None
    best_cost = float('inf')
    for d_val in range(-max_d, max_d + 1):
        remainder = E_int - K + Q * d_val
        b_val = (-remainder) % P
        dac = (remainder + b_val) // P
        if b_val < 0 or b_val >= P:
            continue
        if dac >= 0:
            a_val, c_val = dac, 0
        else:
            a_val, c_val = 0, -dac
        E_check = P * a_val + K - b_val - P * c_val - Q * d_val
        if E_check != E_int:
            continue
        val = BASE * (R ** (E_int / Q))
        err = abs(val - target) / target * 100
        cost = a_val + b_val + c_val + abs(d_val)
        if cost < best_cost:
            best = (a_val, b_val, c_val, d_val, E_int, val, err)
            best_cost = cost
    return best

# Derive all
results = {}
print(f"{'NAME':<28s} {'DIALS':>20s} {'E':>8s} {'ERROR%':>10s}")
print("-" * 70)
errors = []
for name, measured in CONSTANTS.items():
    r = find_best_dials(measured)
    if r is None:
        print(f"{name:<28s}  NOT FOUND")
        continue
    a, b, c, d, E, val, err = r
    results[name] = (a, b, c, d, E, val, err, measured)
    errors.append(err)
    print(f"{name:<28s} ({a:>3},{b:>3},{c:>3},{d:>4}) E={E:>7} err={err:.4f}%")

print(f"\nFound: {len(results)}/{len(CONSTANTS)}")
print(f"Mean error: {sum(errors)/len(errors):.4f}%")
print(f"Max error:  {max(errors):.4f}%")
print(f"Min error:  {min(errors):.4f}%")

# Generate Python code for frequency table
print("\n\n# === GENERATED V3_FREQUENCY_TABLE ===\n")
for name, (a, b, c, d, E, val, err, measured) in results.items():
    mfmt = f"{measured}" if measured >= 1 else f"{measured}"
    print(f"    ({a}, {b}, {c}, {d}):{' '*(max(1, 24-len(f'({a}, {b}, {c}, {d})')))}(\"{name}\",{' '*(max(1,28-len(name)))}{val:.6e},  {E},    {mfmt},{' '*(max(1,20-len(str(mfmt))))} {err:.4f}),")

# Generate named constants
print("\n\n# === GENERATED NAMED CONSTANTS ===\n")
key_names = {
    "SPEED_OF_LIGHT": "C_V3",
    "STANDARD_GRAVITY": "GRAVITY_V3",
    "BOHR_RADIUS_PM": "BOHR_V3",
    "FINE_STRUCTURE_INV": "FINE_STRUCTURE_INV_V3",
    "RYDBERG_EV": "RYDBERG_V3",
    "SCHUMANN_HZ": "SCHUMANN_V3",
    "FE_BCC_LATTICE_PM": "FE_BCC_V3",
    "FE56_BE_PER_NUCLEON": "FE56_BE_V3",
    "MUON_MASS_MEV": "MUON_V3",
    "HIGGS_GEV": "HIGGS_V3",
    "ELECTRON_MASS_MEV": "ELECTRON_MASS_V3",
    "Z_BOSON_GEV": "Z_BOSON_V3",
    "PROTON_MASS_MEV": "PROTON_V3",
    "NEUTRON_MASS_MEV": "NEUTRON_V3",
    "W_BOSON_GEV": "W_BOSON_V3",
    "TAU_MASS_MEV": "TAU_V3",
    "OMEGA": "OMEGA_V3",
    "OMEGA_AUTHORITY": "OMEGA_AUTHORITY_V3",
}
for const_name, var_name in key_names.items():
    if const_name in results:
        a, b, c, d, E, val, err, meas = results[const_name]
        print(f"{var_name} = god_code_v3({a}, {b}, {c}, {d})  # err: {err:.4f}%")

# Generate _rw_v3 calls
print("\n\n# === GENERATED _rw_v3 CALLS ===\n")
rw_meta = {
    "SPEED_OF_LIGHT":       ("speed_of_light",       "m/s",      "SI exact",        "fundamental"),
    "STANDARD_GRAVITY":     ("standard_gravity",     "m/s²",     "SI conventional", "fundamental"),
    "PLANCK_CONSTANT_eVs":  ("planck_constant_eVs",  "eV·s",     "SI exact",        "fundamental"),
    "BOLTZMANN_eV_K":       ("boltzmann_eV_K",       "eV/K",     "SI exact",        "fundamental"),
    "ELEMENTARY_CHARGE":    ("elementary_charge",     "C",        "SI exact",        "fundamental"),
    "AVOGADRO":             ("avogadro",             "mol⁻¹",   "SI exact",        "fundamental"),
    "BOHR_RADIUS_PM":       ("bohr_radius_pm",       "pm",       "CODATA 2022",     "atomic"),
    "RYDBERG_EV":           ("rydberg_eV",           "eV",       "CODATA 2022",     "atomic"),
    "FINE_STRUCTURE_INV":   ("fine_structure_inv",    "",         "CODATA 2022",     "atomic"),
    "COMPTON_PM":           ("compton_pm",           "pm",       "CODATA 2022",     "atomic"),
    "CLASSICAL_E_RADIUS_FM":("classical_e_radius_fm","fm",       "CODATA 2022",     "atomic"),
    "HARTREE_EV":           ("hartree_eV",           "eV",       "CODATA 2022",     "atomic"),
    "MAG_FLUX_QUANTUM_Wb":  ("mag_flux_quantum_Wb",  "Wb",       "CODATA 2022",     "atomic"),
    "VON_KLITZING_OHM":     ("von_klitzing_ohm",     "Ω",        "CODATA 2022",     "atomic"),
    "STEFAN_BOLTZMANN":     ("stefan_boltzmann",      "W·m⁻²·K⁻⁴","CODATA 2022",   "atomic"),
    "VACUUM_PERMITTIVITY":  ("vacuum_permittivity",  "F/m",      "CODATA 2022",     "atomic"),
    "BOHR_MAGNETON_eV_T":   ("bohr_magneton_eV_T",   "eV/T",     "CODATA 2022",     "atomic"),
    "ELECTRON_MASS_MEV":    ("electron_mass_MeV",    "MeV/c²",  "CODATA 2022",     "particle"),
    "MUON_MASS_MEV":        ("muon_mass_MeV",        "MeV/c²",  "PDG 2024",        "particle"),
    "TAU_MASS_MEV":         ("tau_mass_MeV",         "MeV/c²",  "PDG 2024",        "particle"),
    "PROTON_MASS_MEV":      ("proton_mass_MeV",      "MeV/c²",  "CODATA 2022",     "particle"),
    "NEUTRON_MASS_MEV":     ("neutron_mass_MeV",     "MeV/c²",  "CODATA 2022",     "particle"),
    "W_BOSON_GEV":          ("W_boson_GeV",          "GeV/c²",  "PDG 2024",        "particle"),
    "Z_BOSON_GEV":          ("Z_boson_GeV",          "GeV/c²",  "PDG 2024",        "particle"),
    "HIGGS_GEV":            ("higgs_GeV",            "GeV/c²",  "ATLAS/CMS 2024",  "particle"),
    "PION_CHARGED_MEV":     ("pion_charged_MeV",     "MeV/c²",  "PDG 2024",        "particle"),
    "PION_NEUTRAL_MEV":     ("pion_neutral_MeV",     "MeV/c²",  "PDG 2024",        "particle"),
    "KAON_MEV":             ("kaon_MeV",             "MeV/c²",  "PDG 2024",        "particle"),
    "D_MESON_MEV":          ("D_meson_MeV",          "MeV/c²",  "PDG 2024",        "particle"),
    "TOP_QUARK_GEV":        ("top_quark_GeV",        "GeV/c²",  "PDG 2024",        "particle"),
    "BOTTOM_QUARK_GEV":     ("bottom_quark_GeV",     "GeV/c²",  "PDG 2024",        "particle"),
    "CHARM_QUARK_GEV":      ("charm_quark_GeV",      "GeV/c²",  "PDG 2024",        "particle"),
    "FE56_BE_PER_NUCLEON":  ("fe56_be_per_nucleon",  "MeV",     "NNDC/BNL",        "nuclear"),
    "HE4_BE_PER_NUCLEON":   ("he4_be_per_nucleon",   "MeV",     "NNDC/BNL",        "nuclear"),
    "O16_BE_PER_NUCLEON":   ("o16_be_per_nucleon",   "MeV",     "NNDC/BNL",        "nuclear"),
    "C12_BE_PER_NUCLEON":   ("c12_be_per_nucleon",   "MeV",     "NNDC/BNL",        "nuclear"),
    "U238_BE_PER_NUCLEON":  ("u238_be_per_nucleon",  "MeV",     "NNDC/BNL",        "nuclear"),
    "NI62_BE_PER_NUCLEON":  ("ni62_be_per_nucleon",  "MeV",     "NNDC/BNL",        "nuclear"),
    "DEUTERON_BE":          ("deuteron_be",          "MeV",     "NNDC/BNL",        "nuclear"),
    "TRITON_BE":            ("triton_be",            "MeV",     "NNDC/BNL",        "nuclear"),
    "FE_BCC_LATTICE_PM":    ("fe_bcc_lattice_pm",    "pm",       "Kittel/CRC",      "iron"),
    "FE_ATOMIC_RADIUS_PM":  ("fe_atomic_radius_pm",  "pm",       "Slater 1964",     "iron"),
    "FE_K_ALPHA1_KEV":      ("fe_k_alpha1_keV",      "keV",      "NIST SRD 12",     "iron"),
    "FE_IONIZATION_EV":     ("fe_ionization_eV",     "eV",       "NIST ASD",        "iron"),
    "CU_LATTICE_PM":        ("cu_lattice_pm",        "pm",       "Kittel",          "crystal"),
    "AL_LATTICE_PM":        ("al_lattice_pm",        "pm",       "Kittel",          "crystal"),
    "SI_LATTICE_PM":        ("si_lattice_pm",        "pm",       "Kittel",          "crystal"),
    "EARTH_ORBIT_AU_KM":    ("earth_orbit_km",       "km",       "IAU 2012",        "astro"),
    "SOLAR_LUMINOSITY_W":   ("solar_luminosity_W",   "W",        "IAU 2015",        "astro"),
    "HUBBLE_CONSTANT":      ("hubble_constant",      "km/s/Mpc", "Planck 2018",     "astro"),
    "CMB_TEMPERATURE_K":    ("cmb_temperature_K",    "K",        "COBE/FIRAS",      "astro"),
    "SOLAR_MASS_KG":        ("solar_mass_kg",        "kg",       "IAU 2015",        "astro"),
    "SCHUMANN_HZ":          ("schumann_hz",          "Hz",       "Schumann 1952",   "resonance"),
    "ALPHA_EEG_HZ":         ("alpha_eeg_hz",         "Hz",       "Berger 1929",     "resonance"),
    "GAMMA_EEG_HZ":         ("gamma_eeg_hz",         "Hz",       "Galambos 1981",   "resonance"),
    "THETA_EEG_HZ":         ("theta_eeg_hz",         "Hz",       "Neuroscience",    "resonance"),
    "BETA_EEG_HZ":          ("beta_eeg_hz",          "Hz",       "Neuroscience",    "resonance"),
    "PI":                   ("pi",                    "",         "exact",           "math"),
    "EULER_E":              ("euler_e",               "",         "exact",           "math"),
    "SQRT2":                ("sqrt2",                 "",         "exact",           "math"),
    "GOLDEN_RATIO":         ("golden_ratio",          "",         "exact",           "math"),
    "LN2":                  ("ln2",                   "",         "exact",           "math"),
    "PLANCK_LENGTH_M":      ("planck_length_m",      "m",        "CODATA 2022",     "fundamental"),
    "OMEGA":                ("omega",                 "",         "L104 Collective Jan 6 2026", "sovereign"),
    "OMEGA_AUTHORITY":      ("omega_authority",       "",         "L104 derived: Ω/φ²",         "sovereign"),
}
for const_name, (rw_name, unit, source, domain) in rw_meta.items():
    if const_name in results:
        a, b, c, d, E, val, err, meas = results[const_name]
        mfmt = f"{meas}" if isinstance(meas, (int, float)) else str(meas)
        print(f'_rw_v3("{rw_name}",{" "*(max(1,22-len(rw_name)))}{mfmt},{" "*(max(1,20-len(str(mfmt))))}"{unit}",{" "*(max(1,12-len(unit)))}({a}, {b}, {c}, {d}),{" "*(max(1,24-len(f"({a}, {b}, {c}, {d})")))}"{source}",{" "*(max(1,20-len(source)))}"{domain}")')
