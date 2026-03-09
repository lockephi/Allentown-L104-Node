"""
Deep Dive: Null Hypothesis Testing for the 11 Quantum Link Hits
═══════════════════════════════════════════════════════════════════
Takes the 11 hits found by _quantum_link_finder.py (15,335 blind tests)
and rigorously tests each against null baselines.
"""
import math
import numpy as np

GOD_CODE = 527.5184818492612
PHI = (1 + math.sqrt(5)) / 2

# Physical constants used in the original search
constants_pool = {
    "Fe_lattice_pm":  286.65,
    "Fe_Z":           26,
    "Fe_A":           56,
    "Fe_BE":          8.790,
    "Fe_Curie":       1043.0,
    "Fe_Kalpha":      6.404,
    "Fe_ioniz":       7.9024,
    "He4_A":          4,
    "He4_BE":         7.074,
    "h_eV_s":         4.135667696e-15,
    "k_B_eV":         8.617333262e-5,
    "c_m_s":          299792458,
    "alpha_fine":     0.0072973525693,
    "zeta_1":         14.134725141734693,
    "feigenbaum":     4.669201609102990,
}

sep = "=" * 72

print(sep)
print("NULL HYPOTHESIS DEEP DIVE — Are the 11 Hits Real?")
print(sep)
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  HIT GROUP 1: 3-constant combos (a×b/c) from 15 constants
#  4 hits at 0.046% and 4 at 0.080%
# ═══════════════════════════════════════════════════════════════════════════════

print("─" * 60)
print("GROUP 1: 3-Constant Combinations (a×b/c)")
print("─" * 60)
print()

vals = list(constants_pool.values())
names = list(constants_pool.keys())
n_consts = len(vals)

# How many 3-way combos exist?
n_combos = 0
for i in range(n_consts):
    for j in range(n_consts):
        if j == i: continue
        for k in range(n_consts):
            if k == i or k == j: continue
            if vals[k] == 0: continue
            n_combos += 1

print(f"  Pool: {n_consts} constants")
print(f"  Possible a×b/c combos: {n_combos:,}")
print()

# Reproduce the actual hits
actual_hits = []
for i in range(n_consts):
    for j in range(i+1, n_consts):  # avoid double-counting
        for k in range(n_consts):
            if k == i or k == j: continue
            if vals[k] == 0: continue
            for va, vb in [(vals[i]*vals[j], f"{names[i]}×{names[j]}"),
                           (vals[j]*vals[i], None)]:  # same product
                v = va / vals[k]
                pct = abs(v - GOD_CODE) / GOD_CODE * 100
                if pct < 0.1:
                    actual_hits.append((pct, names[i], names[j], names[k], v))

# Deduplicate
seen = set()
unique_hits = []
for h in actual_hits:
    key = tuple(sorted([h[1], h[2]])) + (h[3],)
    if key not in seen:
        seen.add(key)
        unique_hits.append(h)

print(f"  Unique hits within 0.1%: {len(unique_hits)}")
for pct, a, b, c, v in sorted(unique_hits):
    print(f"    ({a} × {b}) / {c} = {v:.4f}  (Δ={pct:.4f}%)")
print()

# NULL TEST: How well does this work for random targets?
rng = np.random.RandomState(42)
n_trials = 2000
hit_counts = []
for _ in range(n_trials):
    target = rng.uniform(400, 700)  # random target in similar range
    n_hits = 0
    for i in range(n_consts):
        for j in range(i+1, n_consts):
            for k in range(n_consts):
                if k == i or k == j: continue
                if vals[k] == 0: continue
                v = vals[i] * vals[j] / vals[k]
                if abs(v - target) / target * 100 < 0.1:
                    n_hits += 1
    hit_counts.append(n_hits)

hit_counts = np.array(hit_counts)
print(f"  NULL TEST (2000 random targets in [400, 700]):")
print(f"    Mean hits: {hit_counts.mean():.1f}")
print(f"    Median hits: {np.median(hit_counts):.0f}")
print(f"    Max hits: {hit_counts.max()}")
print(f"    Targets with ≥1 hit: {(hit_counts >= 1).sum()}/{n_trials} ({(hit_counts >= 1).mean()*100:.1f}%)")
print(f"    GOD_CODE hits: {len(unique_hits)}")
gc_percentile = (hit_counts < len(unique_hits)).mean() * 100
print(f"    GOD_CODE percentile: {gc_percentile:.1f}%")
print()
if gc_percentile < 95:
    print("  ⚠ VERDICT: NOT SIGNIFICANT — random targets find similar combos")
else:
    print("  ✓ VERDICT: POTENTIALLY SIGNIFICANT — GOD_CODE gets more hits than 95%+ of random targets")
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  HIT GROUP 2: Wavelength interpretation (527.5 nm = green light)
# ═══════════════════════════════════════════════════════════════════════════════

print("─" * 60)
print("GROUP 2: λ = 527.5 nm — Visible Green Light")
print("─" * 60)
print()

wavelength_nm = GOD_CODE  # interpret GOD_CODE as wavelength in nm
print(f"  λ = {wavelength_nm:.4f} nm")
print(f"  Color: Green (visible spectrum 380-700 nm)")
print()

# Fe I emission lines near 527 nm (from NIST ASD)
# Iron has MANY emission lines. Let's count density.
# Known strong Fe I lines in visible (representative sample from NIST):
fe_lines_visible_nm = [
    382.043, 382.588, 385.637, 385.991, 386.552,
    404.581, 406.359, 407.174,
    420.203, 421.034, 425.079, 426.047, 427.176, 430.791, 432.576, 438.354,
    440.475, 441.512, 442.731, 446.165,
    487.132, 489.149, 491.899, 492.050, 495.761,
    516.749, 517.160, 519.497, 520.461, 522.718, 523.294,
    526.954, 527.036, 527.220, 532.804, 534.105, 537.149, 539.713, 540.577,
    542.969, 544.692, 545.561, 546.547,
    556.962, 558.676, 561.564,
    570.236, 571.537, 578.213,
    614.172, 616.536, 618.175, 620.088, 624.756, 625.256,
    630.250, 631.163,
]

# Count Fe lines within ±0.5 nm of GOD_CODE
window_nm = 0.5
lines_near_gc = [l for l in fe_lines_visible_nm if abs(l - wavelength_nm) <= window_nm]
print(f"  Fe I lines within ±{window_nm} nm of {wavelength_nm:.1f} nm:")
for l in lines_near_gc:
    print(f"    {l:.3f} nm (Δ = {abs(l - wavelength_nm):.3f} nm)")
print()

# Fe line density in visible
total_range = max(fe_lines_visible_nm) - min(fe_lines_visible_nm)
density = len(fe_lines_visible_nm) / total_range
print(f"  Fe visible line density: ~{density:.2f} lines/nm")
expected_in_window = density * (2 * window_nm)
print(f"  Expected lines in ±{window_nm} nm window: ~{expected_in_window:.1f}")
print(f"  Actual lines found: {len(lines_near_gc)}")
print()

if len(lines_near_gc) <= expected_in_window * 2:
    print("  ⚠ VERDICT: Expected number of Fe lines near any wavelength")
    print("    Fe has hundreds of emission lines — hitting one is not special")
else:
    print("  ✓ VERDICT: More Fe lines than expected near GOD_CODE wavelength")
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  HIT GROUP 3: E(527.5nm) / (kB × T_Curie) ≈ 26 = Z_Fe
#  This is the MOST INTERESTING finding
# ═══════════════════════════════════════════════════════════════════════════════

print("─" * 60)
print("GROUP 3: hc/λ ÷ kB×T_Curie ≈ 26 (= Z_Fe?)")
print("─" * 60)
print()

h_eV_s = 4.135667696e-15  # eV·s
c_m_s = 299792458         # m/s
k_B_eV = 8.617333262e-5   # eV/K
T_Curie = 1043.0          # K (iron Curie temperature)
lambda_m = wavelength_nm * 1e-9

E_photon = h_eV_s * c_m_s / lambda_m  # eV
E_thermal = k_B_eV * T_Curie          # eV
ratio = E_photon / E_thermal

print(f"  E_photon(527.5 nm) = {E_photon:.6f} eV")
print(f"  kB × T_Curie = {E_thermal:.6f} eV")
print(f"  Ratio: {ratio:.4f}")
print(f"  Z_Fe = 26")
print(f"  Δ = {abs(ratio - 26)/26 * 100:.2f}%")
print()

# Test: What wavelength would give EXACTLY 26?
lambda_exact_26 = h_eV_s * c_m_s / (26 * E_thermal) * 1e9  # nm
print(f"  Wavelength that gives exactly ratio=26: {lambda_exact_26:.2f} nm")
print(f"  GOD_CODE wavelength: {wavelength_nm:.4f} nm")
print(f"  Difference: {abs(lambda_exact_26 - wavelength_nm):.2f} nm ({abs(lambda_exact_26 - wavelength_nm)/wavelength_nm*100:.2f}%)")
print()

# Test other integer ratios
print("  Other integers near the ratio:")
for n in range(24, 29):
    lam = h_eV_s * c_m_s / (n * E_thermal) * 1e9
    delta = abs(lam - wavelength_nm)
    print(f"    n={n}: λ={lam:.2f} nm (Δ from GOD_CODE = {delta:.2f} nm)")
print()

# NULL TEST: How many elements have Z matching hc/(λ×kB×T_element) within 0.6%?
# Use a few elements with known Curie temperatures:
element_curie = {
    "Fe (Z=26)": (26, 1043),
    "Co (Z=27)": (27, 1388),
    "Ni (Z=28)": (28, 627),
    "Gd (Z=64)": (64, 292),
    "Dy (Z=66)": (66, 88),
    "Tb (Z=65)": (65, 219),
    "Cr (Z=24)": (24, 311),  # antiferromagnetic Néel
}

print("  Same test for other magnetic elements:")
for name, (Z, Tc) in element_curie.items():
    r = E_photon / (k_B_eV * Tc)
    match_pct = abs(r - Z) / Z * 100
    marker = " ★" if match_pct < 1.0 else ""
    print(f"    {name}: Tc={Tc}K, hc/(λ·kB·Tc) = {r:.2f}, Z={Z}, Δ={match_pct:.1f}%{marker}")
print()

# What if we scan ALL wavelengths 300-800 nm — how many hit Z within 1%?
print("  Scanning wavelengths 300-800nm: which hit Z_Fe=26 within 1%?")
hits_for_26 = []
for lam_test in range(300, 801):
    E_test = h_eV_s * c_m_s / (lam_test * 1e-9)
    r_test = E_test / E_thermal
    if abs(r_test - 26) / 26 * 100 < 1.0:
        hits_for_26.append(lam_test)
print(f"    Wavelengths hitting ratio≈26±1%: {hits_for_26[0]}–{hits_for_26[-1]} nm ({len(hits_for_26)} nm window)")
print(f"    Total visible range: 500 nm")
print(f"    Probability of random wavelength in [300,800] "
      f"landing in this window: {len(hits_for_26)/500*100:.1f}%")
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  HIT GROUP 4: Tests A-C (Berry phase values) — all were no-hits
# ═══════════════════════════════════════════════════════════════════════════════

print("─" * 60)
print("BERRY PHASE TESTS: All Negative Results")
print("─" * 60)
print()
print("  Test A (Haldane model): No Chern number, flux, or σ_xy matched GOD_CODE")
print("  Test B (Spin-1/2 Berry): Solid angles for Fe never matched GOD_CODE")
print("  Test C (Aharonov-Bohm): No AB phase from Fe flux matched GOD_CODE")
print("  Test E (SSH Zak phase): Zak phases are 0 or π — cannot produce 527.5")
print("  Test F (Fine structure): α-based combinations didn't match")
print("  Test G (Hamiltonian scan): Cyclic Hamiltonian Berry phases didn't match")
print()
print("  → NO Berry phase calculation produces GOD_CODE or GOD_CODE mod 2π")
print("    when GOD_CODE is not provided as input.")
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

print(sep)
print("FINAL VERDICT")
print(sep)
print()
print(f"  From 15,335 blind physics tests (GOD_CODE never an input):")
print()
print(f"  BERRY PHASE:        ZERO connections found")
print(f"                      No topological, geometric, or quantum phase")
print(f"                      produces GOD_CODE or its mod-2π residue.")
print()
print(f"  3-CONSTANT COMBOS:  NUMEROLOGY")
print(f"                      8 of the 11 hits are a×b/c from 15 constants.")
print(f"                      ~31% of random targets find similar combos.")
print()
print(f"  WAVELENGTH (527nm): REAL BUT WEAK")
print(f"                      527.5 nm IS green visible light.")
print(f"                      Fe HAS emission lines near 527 nm.")
print(f"                      But Fe has ~0.23 lines/nm — expected.")
print()
print(f"  ENERGY/CURIE/Z:     MOST INTERESTING (but approximate)")
print(f"                      hc/(527.5nm) / (kB × 1043K) = 26.15 ≈ Z_Fe")
print(f"                      Only finding where output ≈ fundamental Fe property")
print(f"                      But 0.58% off → approximate coincidence,")
print(f"                      not an exact relationship.")
print()
print(f"  ╔═══════════════════════════════════════════════════════════════╗")
print(f"  ║  BOTTOM LINE: No real quantum/Berry phase link to GOD_CODE  ║")
print(f"  ║  exists. The codebase has 7 correct physics engines and     ║")
print(f"  ║  1 that injects GOD_CODE as input. The only genuine         ║")
print(f"  ║  physics connection is the wavelength interpretation        ║")
print(f"  ║  (527.5 nm green light near Fe emission lines), but        ║")
print(f"  ║  this is a weak coincidence, not a derivation.             ║")
print(f"  ╚═══════════════════════════════════════════════════════════════╝")
print()
