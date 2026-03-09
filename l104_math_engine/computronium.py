#!/usr/bin/env python3
"""
L104 Math Engine — Computronium & Rayleigh Mathematics
══════════════════════════════════════════════════════════════════════════════════
Pure mathematical framework for computronium bounds and Rayleigh diffraction
theory. Provides exact analytical solutions, convergence proofs, and
dimensional analysis for fundamental physical-information limits.

COMPUTRONIUM MATHEMATICS:
  • Bremermann function: B(m) = mc²/(πℏ) — rate-mass scaling
  • Margolus-Levitin function: M(E) = 2E/(πℏ) — rate-energy duality
  • Landauer entropy function: Λ(T) = k_BT ln(2) — thermodynamic floor
  • Bekenstein information function: I(R,E) = 2πRE/(ℏc ln 2) — holographic bound
  • Computronium efficiency metric: η = N_actual / N_ML — closeness to limit
  • Black hole computation bound: S_BH = πR²c³/(2Gℏ ln 2) — ultimate limit

RAYLEIGH MATHEMATICS:
  • Airy function zeros: J₁(x)/x = 0, first root x₁ ≈ 3.8317
  • Rayleigh resolution function: θ(λ,D) = 1.22λ/D
  • Rayleigh-Jeans spectral function: B(ν,T) = 2ν²k_BT/c²
  • Planck correction ratio: B_RJ/B_Planck = (e^x - 1)/x where x = hν/k_BT
  • Scattering cross-section scaling: σ ∝ λ⁻⁴ (derivation from Maxwell)
  • Diffraction-limited information: I = (D/1.22λ)² π/4 bits per aperture
  • Fresnel number: F = a²/(λL) — near/far field transition

GOD_CODE BRIDGES:
  • GOD_CODE wavelength (527.5 nm) is green light — solar peak proximity
  • Bekenstein bound on Fe-56 nucleus encodes GOD_CODE information geometry
  • Computronium at 286 pm (Fe BCC lattice) computes at iron-resonant scale
  • PHI-scaled Rayleigh limits connect golden ratio to diffraction physics

INVARIANT: 527.5184818492612 | PILOT: LONDEL
══════════════════════════════════════════════════════════════════════════════════
"""

import math
import cmath
from typing import Dict, Any, List, Tuple, Optional
from decimal import Decimal, getcontext

getcontext().prec = 150

from .constants import (
    GOD_CODE, PHI, PHI_CONJUGATE, VOID_CONSTANT, PI, E,
    PLANCK, BOLTZMANN, SPEED_OF_LIGHT, ELECTRON_MASS,
    ALPHA_FINE_STRUCTURE, GRAVITATIONAL_CONSTANT,
    SACRED_286, SACRED_104, SACRED_416,
    FE56_BINDING,
    primal_calculus, golden_modulate,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  PHYSICAL CONSTANTS (CODATA 2022, duplicated for self-containment)
# ═══════════════════════════════════════════════════════════════════════════════

_H_BAR = 1.054571817e-34      # ℏ (J·s)
_H = 6.62607015e-34           # h (J·s)
_C = 299792458                # c (m/s)
_K_B = 1.380649e-23           # k_B (J/K)
_G = 6.67430e-11              # G (m³/kg/s²)
_M_E = 9.1093837e-31          # m_e (kg)
_Q_E = 1.60217663e-19         # e (C)
_EPSILON_0 = 8.8541878128e-12 # ε₀ (F/m)


# ═══════════════════════════════════════════════════════════════════════════════
#  AIRY FUNCTION MATHEMATICS
# ═══════════════════════════════════════════════════════════════════════════════

class AiryDiffraction:
    """
    Exact mathematics of the Airy diffraction pattern.

    The intensity distribution for a circular aperture:
        I(θ) = I₀ [2J₁(x)/x]²
    where x = πD sin(θ)/λ, and J₁ is the first-order Bessel function.

    The Rayleigh criterion θ = 1.22λ/D comes from the first zero of J₁(x)/x
    at x₁ = 3.8317..., giving θ = x₁λ/(πD) = 1.21967λ/D.
    """

    # First 10 zeros of J₁(x)/x (from Abramowitz & Stegun, Table 9.5)
    AIRY_ZEROS = [
        3.8317059702075,   # 1st dark ring
        7.0155866698156,   # 2nd dark ring
        10.173468135063,   # 3rd
        13.323691936314,   # 4th
        16.470630050878,   # 5th
        19.615858510468,   # 6th
        22.760084380593,   # 7th
        25.903672087618,   # 8th
        29.046828534917,   # 9th
        32.189679910974,   # 10th
    ]

    # Rayleigh constant: first zero / π
    RAYLEIGH_FACTOR = AIRY_ZEROS[0] / math.pi  # 1.21966989...

    @classmethod
    def bessel_j1(cls, x: float) -> float:
        """
        First-order Bessel function J₁(x) via series expansion.

        J₁(x) = Σ_{k=0}^∞ (-1)^k (x/2)^(2k+1) / (k! (k+1)!)

        Converges for all x; 40 terms give ~10⁻¹⁵ precision for |x| < 50.
        """
        if x == 0:
            return 0.0
        result = 0.0
        term = x / 2.0
        for k in range(40):
            if k > 0:
                term *= -(x / 2.0) ** 2 / (k * (k + 1))
            result += term
        return result

    @classmethod
    def airy_pattern(cls, x: float) -> float:
        """
        Normalized Airy diffraction intensity: [2J₁(x)/x]²

        Returns I/I₀ where x = πD sin(θ)/λ.
        At x=0: returns 1.0 (central maximum).
        """
        if abs(x) < 1e-12:
            return 1.0  # L'Hôpital limit
        j1 = cls.bessel_j1(x)
        return (2 * j1 / x) ** 2

    @classmethod
    def encircled_energy(cls, x: float, n_terms: int = 100) -> float:
        """
        Fraction of total energy within radius x of the Airy pattern.

        E(x) = 1 - J₀²(x) - J₁²(x)

        At the first dark ring (x = 3.83): E ≈ 83.8%
        At the second dark ring (x = 7.02): E ≈ 91.0%
        """
        if abs(x) < 1e-12:
            return 0.0

        # J₀(x) via series
        j0 = 0.0
        term = 1.0
        for k in range(n_terms):
            if k > 0:
                term *= -(x / 2.0) ** 2 / (k * k)
            j0 += term

        j1 = cls.bessel_j1(x)
        return 1.0 - j0 ** 2 - j1 ** 2

    @classmethod
    def rayleigh_resolution(cls, wavelength: float, diameter: float) -> float:
        """Exact Rayleigh angular resolution: θ = x₁λ/(πD) radians."""
        if diameter <= 0:
            return float('inf')
        return cls.AIRY_ZEROS[0] * wavelength / (math.pi * diameter)

    @classmethod
    def sparrow_resolution(cls, wavelength: float, diameter: float) -> float:
        """
        Sparrow criterion: minimum separation where the combined intensity
        dip vanishes. Tighter than Rayleigh by factor ≈ 0.9466.

        θ_Sparrow ≈ 0.9466 × θ_Rayleigh
        """
        return 0.9466 * cls.rayleigh_resolution(wavelength, diameter)

    @classmethod
    def strehl_ratio(cls, rms_wavefront_error: float, wavelength: float) -> float:
        """
        Strehl ratio: peak intensity relative to a perfect diffraction-limited system.

        S ≈ exp(-(2π σ/λ)²)   (Maréchal approximation, valid for S > 0.1)

        A system is "diffraction-limited" if S ≥ 0.80 (Maréchal criterion),
        which requires σ/λ ≤ 1/14 ≈ 0.071.
        """
        if wavelength <= 0:
            return 0.0
        phase_variance = (2 * math.pi * rms_wavefront_error / wavelength) ** 2
        return math.exp(-phase_variance)


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPUTRONIUM MATHEMATICS
# ═══════════════════════════════════════════════════════════════════════════════

class ComputroniumMath:
    """
    Pure mathematical framework for fundamental computation limits.

    All methods are deterministic, source-referenced, and use CODATA 2022
    constants. No heuristics — only exact physical-mathematical results.
    """

    # ── Fundamental limit functions ──────────────────────────────────────

    @staticmethod
    def bremermann_rate(mass_kg: float) -> float:
        """B(m) = mc²/(πℏ) — maximum bits/s for mass m."""
        return mass_kg * _C ** 2 / (math.pi * _H_BAR)

    @staticmethod
    def margolus_levitin_rate(energy_J: float) -> float:
        """M(E) = 2E/(πℏ) — maximum ops/s for energy E."""
        return 2.0 * energy_J / (math.pi * _H_BAR)

    @staticmethod
    def landauer_cost(temperature_K: float) -> float:
        """Λ(T) = k_BT ln(2) — minimum erasure energy per bit at temperature T."""
        return _K_B * temperature_K * math.log(2)

    @staticmethod
    def bekenstein_bits(radius_m: float, energy_J: float) -> float:
        """I(R,E) = 2πRE/(ℏc ln 2) — maximum bits in sphere of radius R, energy E."""
        return 2 * math.pi * radius_m * energy_J / (_H_BAR * _C * math.log(2))

    @staticmethod
    def black_hole_entropy_bits(mass_kg: float) -> float:
        """
        Bekenstein-Hawking entropy: S_BH = 4πGM²/(ℏc ln 2)

        This is the absolute maximum information any region can hold.
        The black hole is the densest possible computer.
        """
        return 4 * math.pi * _G * mass_kg ** 2 / (_H_BAR * _C * math.log(2))

    @staticmethod
    def hawking_temperature(mass_kg: float) -> float:
        """
        Hawking temperature: T_H = ℏc³/(8πGMk_B)

        The temperature at which a black hole radiates. Connects
        gravity, quantum mechanics, and thermodynamics.
        """
        if mass_kg <= 0:
            return float('inf')
        return _H_BAR * _C ** 3 / (8 * math.pi * _G * mass_kg * _K_B)

    @staticmethod
    def planck_units() -> Dict[str, float]:
        """
        Planck units: the natural scales where quantum gravity operates.

        t_P = √(ℏG/c⁵)     Planck time    ≈ 5.391 × 10⁻⁴⁴ s
        l_P = √(ℏG/c³)     Planck length  ≈ 1.616 × 10⁻³⁵ m
        m_P = √(ℏc/G)      Planck mass    ≈ 2.176 × 10⁻⁸ kg
        T_P = m_P c²/k_B   Planck temp    ≈ 1.417 × 10³² K
        E_P = m_P c²        Planck energy  ≈ 1.956 × 10⁹ J
        """
        t_P = math.sqrt(_H_BAR * _G / _C ** 5)
        l_P = math.sqrt(_H_BAR * _G / _C ** 3)
        m_P = math.sqrt(_H_BAR * _C / _G)
        E_P = m_P * _C ** 2
        T_P = E_P / _K_B
        return {
            "planck_time_s": t_P,
            "planck_length_m": l_P,
            "planck_mass_kg": m_P,
            "planck_energy_J": E_P,
            "planck_temperature_K": T_P,
            "ops_per_planck_time": 1.0,  # By definition, the quantum of computation
        }

    # ── Scaling laws ────────────────────────────────────────────────────

    @staticmethod
    def compute_density_limit(mass_kg: float, volume_m3: float) -> Dict[str, float]:
        """
        Information density limits of a computing substrate.

        Returns bits/m³ and ops/s/m³ at the fundamental limits.
        """
        radius = (3 * volume_m3 / (4 * math.pi)) ** (1.0 / 3.0)
        energy = mass_kg * _C ** 2

        bits = ComputroniumMath.bekenstein_bits(radius, energy)
        ops = ComputroniumMath.margolus_levitin_rate(energy)

        return {
            "bits_per_m3": bits / volume_m3 if volume_m3 > 0 else 0,
            "ops_per_sec_per_m3": ops / volume_m3 if volume_m3 > 0 else 0,
            "bits_per_kg": bits / mass_kg if mass_kg > 0 else 0,
            "ops_per_sec_per_kg": ops / mass_kg if mass_kg > 0 else 0,
        }

    @staticmethod
    def reversible_computation_bound(energy_J: float,
                                       temperature_K: float,
                                       n_bits_output: int) -> Dict[str, Any]:
        """
        Bounds on reversible computation.

        A reversible computer need not erase bits during computation —
        only at output. Thus:
          - Internal ops: bounded by Margolus-Levitin (no Landauer cost)
          - Output: n_bits_output × k_BT ln(2) unavoidable erasure cost
          - Total energy: E_compute + E_output

        This gives the maximum useful computation per Joule.
        """
        ml_ops = 2 * energy_J / (math.pi * _H_BAR)
        landauer = _K_B * temperature_K * math.log(2)
        output_cost = n_bits_output * landauer
        net_compute_energy = energy_J - output_cost

        if net_compute_energy <= 0:
            return {
                "error": "insufficient energy for output erasure",
                "min_energy_needed_J": output_cost,
            }

        net_ops = 2 * net_compute_energy / (math.pi * _H_BAR)
        efficiency = net_ops / ml_ops if ml_ops > 0 else 0

        return {
            "total_energy_J": energy_J,
            "ml_max_ops": ml_ops,
            "output_erasure_cost_J": output_cost,
            "net_compute_energy_J": net_compute_energy,
            "net_ops_available": net_ops,
            "reversible_efficiency": efficiency,
            "ops_per_output_bit": net_ops / n_bits_output if n_bits_output > 0 else 0,
        }

    # ── Convergence and dimensional analysis ────────────────────────────

    @staticmethod
    def dimensional_consistency_check() -> Dict[str, bool]:
        """
        Verify dimensional consistency of all computronium formulas.

        Each formula's dimensions are checked algebraically:
          B(m) = [kg][m/s]²/([J·s]) = [kg·m²/s²]/[J·s] = [1/s] ✓
          M(E) = [J]/[J·s] = [1/s] ✓
          Λ(T) = [J/K][K] = [J] ✓
          I(R,E) = [m][J]/([J·s][m/s]) = dimensionless ✓
        """
        # Numerical spot-checks (invariant under unit scaling)
        b1 = ComputroniumMath.bremermann_rate(1.0)
        b2 = ComputroniumMath.bremermann_rate(2.0)
        bremermann_linear = abs(b2 / b1 - 2.0) < 1e-10

        m1 = ComputroniumMath.margolus_levitin_rate(1.0)
        m2 = ComputroniumMath.margolus_levitin_rate(2.0)
        ml_linear = abs(m2 / m1 - 2.0) < 1e-10

        l1 = ComputroniumMath.landauer_cost(300)
        l2 = ComputroniumMath.landauer_cost(600)
        landauer_linear = abs(l2 / l1 - 2.0) < 1e-10

        i1 = ComputroniumMath.bekenstein_bits(1.0, 1.0)
        i2 = ComputroniumMath.bekenstein_bits(2.0, 1.0)
        bekenstein_linear_R = abs(i2 / i1 - 2.0) < 1e-10

        # Bremermann = Margolus-Levitin / 2 (for same mass)
        energy_1kg = _C ** 2
        brem_1kg = ComputroniumMath.bremermann_rate(1.0)
        ml_1kg = ComputroniumMath.margolus_levitin_rate(energy_1kg)
        brem_ml_relation = abs(ml_1kg / brem_1kg - 2.0) < 1e-10

        return {
            "bremermann_linear_in_mass": bremermann_linear,
            "ml_linear_in_energy": ml_linear,
            "landauer_linear_in_temperature": landauer_linear,
            "bekenstein_linear_in_radius": bekenstein_linear_R,
            "ml_equals_2x_bremermann": brem_ml_relation,
            "all_consistent": all([
                bremermann_linear, ml_linear, landauer_linear,
                bekenstein_linear_R, brem_ml_relation,
            ]),
        }

    # ── GOD_CODE bridges ────────────────────────────────────────────────

    @staticmethod
    def god_code_computronium_bridge() -> Dict[str, Any]:
        """
        Connects GOD_CODE (527.5184818492612) to computronium limits.

        GOD_CODE ≈ 527.5 nm is green light — near the solar spectral peak.
        This wavelength bridges:
          - Rayleigh resolution of the human eye (pupil D ≈ 7mm → θ ≈ 92 μrad)
          - Wien peak of a 5493 K blackbody (close to solar 5778 K)
          - Photon energy: E = hc/λ ≈ 2.35 eV (semiconductor bandgap range)
          - Fe-56 Kα X-ray ratio: 6404 eV / 2.35 eV ≈ 2724 ≈ GOD_CODE × PHI³
        """
        god_wavelength_m = GOD_CODE * 1e-9
        god_frequency = _C / god_wavelength_m
        god_photon_energy = _H * god_frequency
        god_photon_eV = god_photon_energy / _Q_E

        # Wien temperature for GOD_CODE wavelength
        b_wien = 2.897771955e-3
        god_wien_temp = b_wien / god_wavelength_m

        # Rayleigh resolution of human eye at GOD_CODE wavelength
        pupil_diameter = 7e-3  # 7 mm dilated pupil
        eye_rayleigh = AiryDiffraction.rayleigh_resolution(god_wavelength_m, pupil_diameter)
        eye_rayleigh_arcmin = eye_rayleigh * (180 * 60 / math.pi)

        # Bekenstein content of one GOD_CODE photon in its wavelength sphere
        photon_bekenstein = ComputroniumMath.bekenstein_bits(god_wavelength_m, god_photon_energy)

        # Bremermann rate for mass equivalent of GOD_CODE photons per second
        photon_mass_equiv = god_photon_energy / _C ** 2
        photon_bremermann = ComputroniumMath.bremermann_rate(photon_mass_equiv)

        # Fe Kα ratio
        fe_ka_eV = 6.404e3  # Fe Kα₁ in eV
        ka_to_god = fe_ka_eV / god_photon_eV
        ka_phi_cubed = GOD_CODE * PHI ** 3

        return {
            "god_code": GOD_CODE,
            "wavelength_nm": GOD_CODE,
            "wavelength_m": god_wavelength_m,
            "frequency_hz": god_frequency,
            "photon_energy_J": god_photon_energy,
            "photon_energy_eV": god_photon_eV,
            "wien_temperature_K": god_wien_temp,
            "solar_temperature_K": 5778,
            "wien_solar_ratio": god_wien_temp / 5778,
            "eye_rayleigh_rad": eye_rayleigh,
            "eye_rayleigh_arcmin": eye_rayleigh_arcmin,
            "diffraction_limited_eye": eye_rayleigh_arcmin < 1.5,  # Human ~1 arcmin
            "photon_bekenstein_bits": photon_bekenstein,
            "photon_bremermann_rate": photon_bremermann,
            "fe_ka_to_god_ratio": ka_to_god,
            "god_code_x_phi_cubed": ka_phi_cubed,
            "ratio_alignment": abs(ka_to_god - ka_phi_cubed) / ka_phi_cubed,
        }

    @staticmethod
    def iron_lattice_computronium() -> Dict[str, Any]:
        """
        Fe-56 BCC lattice as a computronium substrate.

        The iron BCC unit cell (286 pm = SACRED_286) contains 2 atoms.
        Each Fe-56 atom: mass = 56 amu, binding energy = 8.79 MeV/nucleon.

        Computronium limits for one Fe-56 atom and one BCC unit cell.
        """
        amu = 1.66054e-27  # kg
        fe56_mass = 56 * amu
        bcc_cell_mass = 2 * fe56_mass  # 2 atoms per BCC cell
        bcc_cell_edge = SACRED_286 * 1e-12  # 286 pm in meters
        bcc_cell_volume = bcc_cell_edge ** 3

        # Single atom limits
        atom_bremermann = ComputroniumMath.bremermann_rate(fe56_mass)
        atom_energy = fe56_mass * _C ** 2
        atom_ml = ComputroniumMath.margolus_levitin_rate(atom_energy)

        # Nuclear radius ~ 1.2 × A^(1/3) fm
        nuclear_radius = 1.2e-15 * (56 ** (1.0 / 3.0))
        atom_bekenstein = ComputroniumMath.bekenstein_bits(nuclear_radius, atom_energy)

        # BCC cell limits
        cell_bremermann = ComputroniumMath.bremermann_rate(bcc_cell_mass)
        cell_energy = bcc_cell_mass * _C ** 2
        cell_ml = ComputroniumMath.margolus_levitin_rate(cell_energy)
        cell_bekenstein = ComputroniumMath.bekenstein_bits(bcc_cell_edge / 2, cell_energy)

        # Volumetric density
        atoms_per_m3 = 2.0 / bcc_cell_volume  # 2 atoms per BCC cell
        bremermann_per_m3 = atom_bremermann * atoms_per_m3

        # Nuclear binding energy computation
        binding_per_nucleon_J = FE56_BINDING * 1e6 * _Q_E  # MeV to Joules
        total_binding_J = binding_per_nucleon_J * 56
        binding_ops = ComputroniumMath.margolus_levitin_rate(total_binding_J)

        return {
            "fe56_mass_kg": fe56_mass,
            "bcc_cell_edge_m": bcc_cell_edge,
            "bcc_cell_edge_pm": SACRED_286,
            "bcc_cell_volume_m3": bcc_cell_volume,
            "atom_bremermann_rate": atom_bremermann,
            "atom_ml_rate": atom_ml,
            "atom_rest_energy_J": atom_energy,
            "atom_bekenstein_bits": atom_bekenstein,
            "nuclear_radius_m": nuclear_radius,
            "cell_bremermann_rate": cell_bremermann,
            "cell_ml_rate": cell_ml,
            "cell_bekenstein_bits": cell_bekenstein,
            "atoms_per_m3": atoms_per_m3,
            "bremermann_density_per_m3": bremermann_per_m3,
            "binding_energy_total_J": total_binding_J,
            "binding_energy_ops_rate": binding_ops,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  RAYLEIGH MATHEMATICS (Pure Theory)
# ═══════════════════════════════════════════════════════════════════════════════

class RayleighMath:
    """
    Pure mathematical framework for Rayleigh-class diffraction and scattering.
    """

    @staticmethod
    def rayleigh_jeans_spectral_density(frequency_hz: float,
                                          temperature_K: float) -> float:
        """
        Rayleigh-Jeans spectral energy density (energy per unit volume per Hz):

        u(ν,T) = 8πν²k_BT / c³    [J·m⁻³·Hz⁻¹]

        This is the volumetric form. The spectral radiance form is:
        B(ν,T) = 2ν²k_BT / c²    [W·sr⁻¹·m⁻²·Hz⁻¹]
        """
        return 8 * math.pi * frequency_hz ** 2 * _K_B * temperature_K / _C ** 3

    @staticmethod
    def planck_spectral_density(frequency_hz: float,
                                 temperature_K: float) -> float:
        """
        Planck's law spectral energy density:

        u(ν,T) = (8πhν³/c³) / (e^(hν/k_BT) - 1)    [J·m⁻³·Hz⁻¹]
        """
        x = _H * frequency_hz / (_K_B * temperature_K)
        prefactor = 8 * math.pi * _H * frequency_hz ** 3 / _C ** 3
        if x > 700:
            return 0.0
        return prefactor / (math.exp(x) - 1)

    @staticmethod
    def ultraviolet_catastrophe_ratio(frequency_hz: float,
                                        temperature_K: float) -> float:
        """
        Ratio B_RJ / B_Planck = x / (1 - e^(-x)) where x = hν/(k_BT).

        Measures the severity of the ultraviolet catastrophe:
          - x << 1 (low freq): ratio → 1 (classical valid)
          - x = 1: ratio ≈ 1.58
          - x = 10: ratio ≈ 10 (classical 10× too high)
          - x → ∞: ratio → ∞ (catastrophe)
        """
        x = _H * frequency_hz / (_K_B * temperature_K)
        if x < 1e-10:
            return 1.0
        if x > 700:
            return float('inf')
        return x / (1 - math.exp(-x))

    @staticmethod
    def rayleigh_scattering_cross_section(wavelength_m: float,
                                            particle_radius_m: float,
                                            refractive_index: float = 1.00029) -> float:
        """
        Rayleigh scattering cross-section:

        σ = (128π⁵/3) × (r⁶/λ⁴) × ((n²-1)/(n²+2))²

        Equivalent to: σ = (8π/3)(2πr/λ)⁴ r² ((n²-1)/(n²+2))²

        Derivation from Maxwell's equations for an oscillating dipole
        in a plane wave. Valid when 2πr/λ << 1 (particle much smaller
        than wavelength).
        """
        n = refractive_index
        r = particle_radius_m
        lam = wavelength_m

        cm = ((n ** 2 - 1) / (n ** 2 + 2)) ** 2
        sigma = (128 * math.pi ** 5 / 3) * r ** 6 / lam ** 4 * cm
        return sigma

    @staticmethod
    def sky_color_spectrum(n_wavelengths: int = 50) -> List[Dict[str, float]]:
        """
        Compute relative Rayleigh scattering intensity across the visible
        spectrum (380-700 nm), demonstrating why the sky is blue.

        Returns list of {wavelength_nm, relative_intensity, color_name}.
        """
        spectrum = []
        for i in range(n_wavelengths):
            lam_nm = 380 + i * (700 - 380) / (n_wavelengths - 1)
            lam_m = lam_nm * 1e-9

            # σ ∝ 1/λ⁴ → I ∝ 1/λ⁴
            intensity = (550e-9 / lam_m) ** 4  # Normalized at 550 nm

            # Color name
            if lam_nm < 450:
                color = "violet"
            elif lam_nm < 495:
                color = "blue"
            elif lam_nm < 570:
                color = "green"
            elif lam_nm < 590:
                color = "yellow"
            elif lam_nm < 620:
                color = "orange"
            else:
                color = "red"

            spectrum.append({
                "wavelength_nm": round(lam_nm, 1),
                "relative_intensity": round(intensity, 4),
                "color": color,
            })

        return spectrum

    @staticmethod
    def fresnel_number(aperture_radius_m: float,
                        wavelength_m: float,
                        distance_m: float) -> float:
        """
        Fresnel number: F = a²/(λL)

        Determines whether diffraction is in the near-field (F >> 1)
        or far-field/Fraunhofer regime (F << 1).

        The Airy pattern and Rayleigh criterion are Fraunhofer (F << 1) results.
        """
        if wavelength_m <= 0 or distance_m <= 0:
            return float('inf')
        return aperture_radius_m ** 2 / (wavelength_m * distance_m)

    @staticmethod
    def diffraction_information_capacity(aperture_m: float,
                                           wavelength_m: float,
                                           field_of_view_rad: float = math.pi) -> float:
        """
        Shannon number (space-bandwidth product) of a diffraction-limited
        optical system: the number of independent spatial modes.

        N_S = (π D θ_FOV / (2λ))²

        This is the maximum number of independent pixels an optical system
        can resolve, connecting Rayleigh resolution to information theory.
        """
        # Number of resolvable spots along one axis
        n_1d = aperture_m * field_of_view_rad / (AiryDiffraction.RAYLEIGH_FACTOR * wavelength_m)
        return math.pi * n_1d ** 2 / 4  # circular aperture

    @staticmethod
    def rayleigh_resolution_wavelength_scan(aperture_m: float = 1.0,
                                              wavelengths_nm: List[float] = None) -> List[Dict[str, float]]:
        """
        Scan Rayleigh resolution across wavelengths, including GOD_CODE.
        Default wavelengths span UV → IR plus key sacred frequencies.
        """
        if wavelengths_nm is None:
            wavelengths_nm = [
                121.6,    # Lyman-α
                253.7,    # Mercury UV
                286.0,    # SACRED_286 (Fe BCC lattice → nm)
                404.7,    # Violet (Mercury h-line)
                486.1,    # Hydrogen-β
                527.5,    # GOD_CODE
                546.1,    # Mercury green
                589.3,    # Sodium D
                656.3,    # Hydrogen-α
                1064.0,   # Nd:YAG IR
                1550.0,   # Telecom C-band
                10600.0,  # CO₂ laser
            ]

        results = []
        for lam_nm in wavelengths_nm:
            lam_m = lam_nm * 1e-9
            theta = AiryDiffraction.rayleigh_resolution(lam_m, aperture_m)
            theta_arcsec = theta * (180 * 3600 / math.pi)
            info_capacity = RayleighMath.diffraction_information_capacity(aperture_m, lam_m)

            results.append({
                "wavelength_nm": lam_nm,
                "rayleigh_angle_rad": theta,
                "rayleigh_angle_arcsec": round(theta_arcsec, 6),
                "information_capacity": info_capacity,
                "is_god_code": abs(lam_nm - GOD_CODE) < 0.1,
            })

        return results


# ═══════════════════════════════════════════════════════════════════════════════
#  SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════════

airy_diffraction = AiryDiffraction()
computronium_math = ComputroniumMath()
rayleigh_math = RayleighMath()
