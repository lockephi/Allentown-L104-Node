"""
L104 Real-World Simulator — Sacred Constants & Grid Parameters

Single source of truth for the GOD_CODE v3 logarithmic lattice.

GRID EQUATION:
  G(a,b,c,d) = 286^(1/φ) × 2^((64a + 1664 - b - 64c - 416d) / 416)

Where:
  X  = 286   (prime scaffold: 2 × 11 × 13)
  R  = 2     (ratio base — octave)
  Q  = 416   (quantization grain — steps per octave)
  P  = 64    (dial coefficient)
  K  = 1664  (offset = 4 × Q)
  φ  = 1.618033988749895 (golden ratio)

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import math
from typing import Final

# ═══════════════════════════════════════════════════════════════════════════════
#  SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI: Final[float] = (1 + math.sqrt(5)) / 2                  # 1.618033988749895
PHI_SQ: Final[float] = PHI ** 2                              # 2.618033988749895
GOD_CODE: Final[float] = 527.5184818492612                   # 286^(1/φ) × 2^4
VOID_CONSTANT: Final[float] = 1.04 + PHI / 1000             # 1.0416180339887497
OMEGA: Final[float] = 6539.34712682                          # Sovereign frequency

# ═══════════════════════════════════════════════════════════════════════════════
#  GRID PARAMETERS (v3 GOD_CODE algorithm, P=64)
# ═══════════════════════════════════════════════════════════════════════════════

X_SCAFFOLD: Final[int] = 286       # Prime scaffold: 2 × 11 × 13
R_RATIO: Final[int] = 2            # Octave ratio
Q_GRAIN: Final[int] = 416          # Steps per octave
P_DIAL: Final[int] = 64            # Dial coefficient
K_OFFSET: Final[int] = 1664        # Offset = 4 × Q

BASE: Final[float] = X_SCAFFOLD ** (1.0 / PHI)              # 32.969905115578818
STEP_SIZE: Final[float] = R_RATIO ** (1.0 / Q_GRAIN)        # 1.001667608098528
MAX_GRID_ERROR: Final[float] = (STEP_SIZE - 1) / 2 * 100    # 0.0834% theoretical max

# ═══════════════════════════════════════════════════════════════════════════════
#  GRID ANALYSIS (from Part III Sovereign Field Research)
# ═══════════════════════════════════════════════════════════════════════════════
# Conservation identity: G(a,b,c,d) × 2^((K-E)/Q) = GOD_CODE  ∀ (a,b,c,d)
# This is algebraically exact: BASE × 2^(K/Q) = GOD_CODE
CONSERVATION_INVARIANT: Final[float] = GOD_CODE              # = BASE × 2^(K_OFFSET/Q_GRAIN)
GRID_GCD: Final[int] = 32                                    # gcd(P_DIAL, Q_GRAIN)
# Grid spans ~17.2 octaves with (a,b,c,d) in their signed-int ranges
GRID_OCTAVE_SPAN: Final[float] = 17.19                       # log₂(freq_max/freq_min)
# Fe-Sacred: 528/286 = 24/13 (Factor-13 resonance)
FE_SACRED_RATIO: Final[float] = 24.0 / 13.0                  # 1.846153846...
# OMEGA authority = Ω / φ²
OMEGA_AUTHORITY: Final[float] = OMEGA / PHI_SQ                # ≈ 2497.808

# ═══════════════════════════════════════════════════════════════════════════════
#  PHYSICAL CONSTANTS — PARTICLE MASSES (MeV/c²)
# ═══════════════════════════════════════════════════════════════════════════════

# Leptons
M_ELECTRON: Final[float] = 0.51099895069       # e
M_MUON: Final[float] = 105.6583755             # μ
M_TAU: Final[float] = 1776.86                  # τ

# Neutrino mass splittings (eV²) — for PMNS mixing
DM21_SQ: Final[float] = 7.53e-5                # Δm²₂₁ (solar)
DM32_SQ: Final[float] = 2.453e-3               # |Δm²₃₂| (atmospheric, normal ordering)

# Up-type quarks (MeV)
M_UP: Final[float] = 2.16
M_CHARM: Final[float] = 1270.0
M_TOP: Final[float] = 172570.0

# Down-type quarks (MeV)
M_DOWN: Final[float] = 4.67
M_STRANGE: Final[float] = 93.0
M_BOTTOM: Final[float] = 4180.0

# Gauge bosons (MeV)
M_W: Final[float] = 80369.2
M_Z: Final[float] = 91187.6
M_HIGGS: Final[float] = 125250.0

# Hadrons (MeV)
M_PROTON: Final[float] = 938.27208816
M_NEUTRON: Final[float] = 939.56542052
M_PION_PM: Final[float] = 139.57039
M_PION_0: Final[float] = 134.9768
M_KAON: Final[float] = 493.677
M_D_MESON: Final[float] = 1869.66

# ═══════════════════════════════════════════════════════════════════════════════
#  FUNDAMENTAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

C_LIGHT: Final[float] = 299792458              # m/s
ALPHA_INV: Final[float] = 137.035999084         # 1/α (fine structure)
ALPHA_EM: Final[float] = 1.0 / ALPHA_INV       # α ≈ 0.007297
PLANCK_H_EVS: Final[float] = 4.135667696e-15   # h in eV·s
HBAR_EVS: Final[float] = PLANCK_H_EVS / (2 * math.pi)
KB_EV: Final[float] = 8.617333262e-5           # kB in eV/K
E_CHARGE: Final[float] = 1.602176634e-19       # Coulombs
PLANCK_LENGTH: Final[float] = 1.616255e-35     # m
PLANCK_MASS_GEV: Final[float] = 1.220890e19    # GeV
BOHR_RADIUS_PM: Final[float] = 52.9177210544   # pm
RYDBERG_EV: Final[float] = 13.605693123        # eV

# Nuclear binding energies (MeV/nucleon)
BE_FE56: Final[float] = 8.790
BE_HE4: Final[float] = 7.074
BE_DEUT: Final[float] = 2.22457

# Coupling constants at MZ scale
ALPHA_S_MZ: Final[float] = 0.1179              # Strong coupling at Z mass
WEINBERG_ANGLE: Final[float] = math.acos(M_W / M_Z)  # θ_W ≈ 28.2°

# ═══════════════════════════════════════════════════════════════════════════════
#  MIXING ANGLES (degrees)
# ═══════════════════════════════════════════════════════════════════════════════

# CKM matrix parameters
CKM_THETA12: Final[float] = 13.04   # Cabibbo angle
CKM_THETA13: Final[float] = 0.201
CKM_THETA23: Final[float] = 2.38
CKM_DELTA_CP: Final[float] = 1.20   # CP phase (radians)

# PMNS matrix parameters
PMNS_THETA12: Final[float] = 33.44  # Solar angle
PMNS_THETA13: Final[float] = 8.57   # Reactor angle
PMNS_THETA23: Final[float] = 49.2   # Atmospheric angle
PMNS_DELTA_CP: Final[float] = 3.86  # Dirac CP phase (radians), ~221°

# ═══════════════════════════════════════════════════════════════════════════════
#  ENERGY SCALES (MeV)
# ═══════════════════════════════════════════════════════════════════════════════

SCALE_QCD: Final[float] = 220.0                     # Λ_QCD
SCALE_EW: Final[float] = 246220.0                    # Higgs VEV (MeV)
SCALE_PLANCK: Final[float] = PLANCK_MASS_GEV * 1e3   # Planck mass (MeV)
SCALE_GUT: Final[float] = 2e16 * 1e3                 # GUT scale (MeV)
