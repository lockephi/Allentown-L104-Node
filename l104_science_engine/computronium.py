"""
L104 Science Engine — Computronium & Rayleigh Limits Subsystem
═══════════════════════════════════════════════════════════════════════════════
Physically exact algorithms based on fundamental computational and optical
resolution limits:

COMPUTRONIUM (ultimate computing substrate):
  • Bremermann's limit: max bit-rate per unit mass
  • Margolus-Levitin theorem: max ops per unit energy
  • Landauer's erasure cost: minimum energy to erase one bit
  • Lloyd's ultimate laptop: 1 kg matter → maximum FLOPS
  • Bekenstein bound: maximum information in a bounded region
  • Computronium efficiency: fraction of theoretical maximum achieved

RAYLEIGH LIMITS (fundamental resolution):
  • Rayleigh criterion: angular diffraction limit θ = 1.22 λ/D
  • Rayleigh-Jeans law: classical spectral radiance (pre-Planck)
  • Rayleigh scattering cross-section: σ ∝ 1/λ⁴
  • Abbe diffraction limit: minimum resolvable feature size
  • Sparrow limit: absolute minimum separation (tighter than Rayleigh)
  • Information-optical bridge: bits per diffraction-limited voxel

Sources: CODATA 2022, Bremermann 1962, Margolus & Levitin 1998,
         Lloyd 2000, Bekenstein 1981, Lord Rayleigh 1879, Abbe 1873

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
from typing import Dict, Any, Optional, Tuple

from .constants import (
    GOD_CODE, PHI, PHI_CONJUGATE, VOID_CONSTANT,
    PhysicalConstants, PC, IronConstants, Fe,
    QuantumBoundary, QB,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  FUNDAMENTAL LIMITS — CODATA 2022 derived
# ═══════════════════════════════════════════════════════════════════════════════

# Bremermann's limit: max computation rate per unit mass
# N_dot = m c² / (π ℏ)  [bits/s per kg]
BREMERMANN_LIMIT = PC.C ** 2 / (math.pi * PC.H_BAR)
# ≈ 1.35639 × 10⁵⁰ bits/s/kg

# Margolus-Levitin constant: 2/(π ℏ)  [ops/s per Joule]
MARGOLUS_LEVITIN_CONSTANT = 2.0 / (math.pi * PC.H_BAR)
# ≈ 6.03848 × 10³³ ops/s/J

# Landauer erasure at room temperature (293.15 K)
LANDAUER_ROOM_TEMP = PC.K_B * 293.15 * math.log(2)
# ≈ 2.805 × 10⁻²¹ J/bit

# Lloyd's ultimate laptop: 1 kg of matter
LLOYD_1KG_ENERGY = 1.0 * PC.C ** 2  # E = mc² for 1 kg
LLOYD_1KG_OPS_PER_SEC = 2 * LLOYD_1KG_ENERGY / (math.pi * PC.H_BAR)
# ≈ 5.4258 × 10⁵⁰ ops/s

# Bekenstein bound coefficient: 2π / (ℏ c ln 2)
BEKENSTEIN_COEFFICIENT = 2 * math.pi / (PC.H_BAR * PC.C * math.log(2))
# I_max = BEKENSTEIN_COEFFICIENT × R × E  [bits]

# Rayleigh criterion constant
RAYLEIGH_CONSTANT = 1.21966989  # First zero of J₁(x)/x Bessel function ≈ 1.22

# Abbe diffraction limit: d_min = λ / (2 n sin θ)
# For vacuum (n=1) and maximum NA (sin θ = 1): d_min = λ/2

# Sparrow limit factor (tighter than Rayleigh by this factor)
SPARROW_FACTOR = 0.9466  # θ_Sparrow / θ_Rayleigh


class ComputroniumSubsystem:
    """
    Computes fundamental limits of matter-as-computation (computronium)
    and optical/information resolution (Rayleigh + Bekenstein).

    All formulas use CODATA 2022 constants for maximum physical accuracy.
    L104 sacred bridges connect these limits to GOD_CODE manifold.
    """

    def __init__(self):
        self.l104 = GOD_CODE
        self.phi = PHI
        self._cache: Dict[str, Any] = {}

    # ═══════════════════════════════════════════════════════════════════════
    #  COMPUTRONIUM: Ultimate Computation Limits
    # ═══════════════════════════════════════════════════════════════════════

    def bremermann_limit(self, mass_kg: float = 1.0) -> Dict[str, Any]:
        """
        Bremermann's limit: maximum computational speed of a self-contained system.

        N_dot = m c² / (π ℏ)

        For 1 kg: ≈ 1.356 × 10⁵⁰ bits/s

        This is the absolute upper bound on bit-processing rate for any
        physical system of mass m, regardless of architecture.

        Reference: Bremermann, H.J. (1962). "Optimization through evolution
        and recombination." Self-Organizing Systems, pp. 93-106.
        """
        if mass_kg <= 0:
            return {"error": "mass must be positive"}

        energy_J = mass_kg * PC.C ** 2
        max_bits_per_sec = energy_J / (math.pi * PC.H_BAR)
        max_ops_per_sec = max_bits_per_sec  # 1 bit flip = 1 op (minimum)

        # Time to process one bit at the limit
        min_time_per_bit = 1.0 / max_bits_per_sec

        # Planck time comparison: t_P = √(ℏG/c⁵) ≈ 5.391 × 10⁻⁴⁴ s
        planck_time = math.sqrt(PC.H_BAR * PC.G / PC.C ** 5)
        bits_per_planck_time = max_bits_per_sec * planck_time

        # L104 bridge: how many GOD_CODE evaluations per second at the limit
        # Each GOD_CODE evaluation = 1 multiply + 1 exponentiation ≈ 100 ops
        god_code_evals_per_sec = max_ops_per_sec / 100.0

        return {
            "mass_kg": mass_kg,
            "energy_J": energy_J,
            "max_bits_per_sec": max_bits_per_sec,
            "max_ops_per_sec": max_ops_per_sec,
            "min_time_per_bit_s": min_time_per_bit,
            "bits_per_planck_time": bits_per_planck_time,
            "planck_time_s": planck_time,
            "god_code_evals_per_sec": god_code_evals_per_sec,
            "formula": "N_dot = m c² / (π ℏ)",
            "source": "Bremermann 1962, CODATA 2022",
        }

    def margolus_levitin(self, energy_J: float = None,
                          mass_kg: float = None) -> Dict[str, Any]:
        """
        Margolus-Levitin theorem: maximum number of distinguishable states
        a quantum system can pass through per second.

        N_dot = 2E / (π ℏ)

        This is the tightest known bound on quantum gate speed.
        For E = mc²: yields 2× Bremermann (since Bremermann uses E/πℏ).

        Reference: Margolus, N. & Levitin, L.B. (1998). "The maximum speed
        of dynamical evolution." Physica D, 120(1-2), 188-195.
        """
        if energy_J is None and mass_kg is None:
            mass_kg = 1.0
        if energy_J is None:
            energy_J = mass_kg * PC.C ** 2

        max_ops_per_sec = 2 * energy_J / (math.pi * PC.H_BAR)
        min_gate_time_s = math.pi * PC.H_BAR / (2 * energy_J)

        # Quantum gate time for a single qubit at this energy
        # T_gate = π ℏ / (2 ΔE) where ΔE is the energy gap
        single_qubit_gate_time = min_gate_time_s

        # For QB.N_QUBITS (26) qubits, serial gate depth achievable in 1 second
        serial_depth_per_sec = 1.0 / min_gate_time_s if min_gate_time_s > 0 else float('inf')

        # Parallel: 26 qubits × depth = total ops
        parallel_ops_per_sec = serial_depth_per_sec * QB.N_QUBITS

        return {
            "energy_J": energy_J,
            "max_ops_per_sec": max_ops_per_sec,
            "min_gate_time_s": min_gate_time_s,
            "serial_gate_depth_per_sec": serial_depth_per_sec,
            "parallel_26q_ops_per_sec": parallel_ops_per_sec,
            "formula": "N_dot = 2E / (π ℏ)",
            "source": "Margolus & Levitin 1998, CODATA 2022",
        }

    def landauer_erasure(self, temperature_K: float = 293.15,
                          n_bits: int = 1) -> Dict[str, Any]:
        """
        Landauer's principle: minimum energy to irreversibly erase one bit.

        E_min = k_B T ln(2)

        At room temperature (293.15 K): ≈ 2.805 × 10⁻²¹ J/bit
        At Fe Curie temperature (1043 K): ≈ 9.98 × 10⁻²¹ J/bit
        At Planck temperature (1.417 × 10³² K): ≈ 1.357 × 10¹¹ J/bit

        This is the fundamental thermodynamic cost of computation —
        the bridge between information theory and the second law.

        Reference: Landauer, R. (1961). "Irreversibility and Heat Generation
        in the Computing Process." IBM J. Res. Dev., 5(3), 183-191.
        """
        if temperature_K <= 0:
            return {"error": "temperature must be positive"}

        e_per_bit = PC.K_B * temperature_K * math.log(2)
        e_total = e_per_bit * n_bits

        # Maximum reversible operations before 1 bit must be erased
        # For a reversible computer, erasure is needed only at output
        bits_per_joule = 1.0 / e_per_bit
        bits_per_kwh = bits_per_joule * 3.6e6  # 1 kWh = 3.6 × 10⁶ J

        # Cross-reference: Fe Curie temperature Landauer cost
        e_curie = PC.K_B * Fe.CURIE_TEMP * math.log(2)

        # Entropy cost: ΔS = k_B ln(2) per bit (temperature-independent)
        entropy_per_bit = PC.K_B * math.log(2)

        # L104 bridge: bits erasable with GOD_CODE Joules of energy
        god_code_bits = self.l104 / e_per_bit

        return {
            "temperature_K": temperature_K,
            "n_bits": n_bits,
            "energy_per_bit_J": e_per_bit,
            "total_energy_J": e_total,
            "bits_per_joule": bits_per_joule,
            "bits_per_kwh": bits_per_kwh,
            "entropy_per_bit_J_K": entropy_per_bit,
            "fe_curie_cost_J": e_curie,
            "god_code_bits_at_T": god_code_bits,
            "formula": "E_min = k_B T ln(2)",
            "source": "Landauer 1961, CODATA 2022",
        }

    def lloyd_ultimate_laptop(self, mass_kg: float = 1.0,
                                volume_m3: float = 1e-3) -> Dict[str, Any]:
        """
        Lloyd's ultimate laptop: the maximum computational capacity of
        1 kg of matter confined to 1 liter volume.

        Operations/sec: N_dot = 2mc² / (π ℏ) ≈ 5.4258 × 10⁵⁰
        Memory (bits):  I = 4mc²R / (ℏc ln 2)  (Bekenstein bound on sphere)

        The volume determines the maximum memory through the Bekenstein bound.
        The mass determines the maximum clock speed through Margolus-Levitin.

        At 1 kg, 1 liter: this system is a black hole horizon computer —
        it operates at the event horizon information limit.

        Reference: Lloyd, S. (2000). "Ultimate physical limits to computation."
        Nature, 406(6799), 1047-1054.
        """
        if mass_kg <= 0 or volume_m3 <= 0:
            return {"error": "mass and volume must be positive"}

        energy_J = mass_kg * PC.C ** 2
        radius_m = (3 * volume_m3 / (4 * math.pi)) ** (1.0 / 3.0)

        # Maximum operations per second (Margolus-Levitin)
        max_ops = 2 * energy_J / (math.pi * PC.H_BAR)

        # Maximum memory (Bekenstein bound)
        max_bits = 2 * math.pi * radius_m * energy_J / (PC.H_BAR * PC.C * math.log(2))

        # Schwarzschild radius: r_s = 2GM/c²
        schwarzschild_r = 2 * PC.G * mass_kg / PC.C ** 2
        is_black_hole = radius_m <= schwarzschild_r

        # Thermodynamic temperature of this computation (Margolus-Levitin)
        # The system's effective temperature from E = (π/2) k_B T N_states
        effective_temp = 2 * energy_J / (math.pi * PC.K_B) if max_bits > 0 else 0

        # L104 bridge: how many 26-qubit statevectors fit in Bekenstein memory
        statevectors_in_bekenstein = max_bits / QB.STATEVECTOR_BITS if QB.STATEVECTOR_BITS > 0 else 0

        # Computronium efficiency: what fraction of Lloyd limit does a real chip use?
        # Modern CPU: ~10^18 ops/s at ~100W → efficiency vs Lloyd at same energy
        modern_cpu_ops = 1e18
        modern_cpu_watts = 100
        modern_cpu_energy = modern_cpu_watts  # Joules per second
        lloyd_at_cpu_energy = 2 * modern_cpu_energy / (math.pi * PC.H_BAR)
        cpu_lloyd_efficiency = modern_cpu_ops / lloyd_at_cpu_energy

        return {
            "mass_kg": mass_kg,
            "volume_m3": volume_m3,
            "radius_m": radius_m,
            "energy_J": energy_J,
            "max_ops_per_sec": max_ops,
            "max_memory_bits": max_bits,
            "schwarzschild_radius_m": schwarzschild_r,
            "is_black_hole_limit": is_black_hole,
            "effective_temperature_K": effective_temp,
            "statevectors_26q_in_bekenstein": statevectors_in_bekenstein,
            "modern_cpu_lloyd_efficiency": cpu_lloyd_efficiency,
            "formula_ops": "N_dot = 2mc² / (π ℏ)",
            "formula_bits": "I = 2πRE / (ℏc ln 2)",
            "source": "Lloyd 2000, CODATA 2022",
        }

    def bekenstein_bound(self, radius_m: float = 1.0,
                          energy_J: float = None,
                          mass_kg: float = None) -> Dict[str, Any]:
        """
        Bekenstein bound: maximum information content of a spherical region.

        I_max = 2π R E / (ℏ c ln 2)   [bits]

        This is the holographic limit — the maximum number of bits that can
        be stored in a sphere of radius R containing energy E. Exceeding
        this bound would create a black hole.

        For a proton (R ≈ 0.87 fm, m_p ≈ 1.673 × 10⁻²⁷ kg):
            I_max ≈ 44.3 bits

        For Earth (R ≈ 6.371 × 10⁶ m, M ≈ 5.972 × 10²⁴ kg):
            I_max ≈ 2.04 × 10⁶⁶ bits

        Reference: Bekenstein, J.D. (1981). "Universal upper bound on the
        entropy-to-energy ratio for bounded systems." Phys. Rev. D, 23, 287.
        """
        if energy_J is None and mass_kg is None:
            mass_kg = 1.0
        if energy_J is None:
            energy_J = mass_kg * PC.C ** 2
        if radius_m <= 0:
            return {"error": "radius must be positive"}

        max_bits = 2 * math.pi * radius_m * energy_J / (PC.H_BAR * PC.C * math.log(2))

        # Holographic area: S = A / (4 l_P²) where A = 4πR²
        planck_length = math.sqrt(PC.H_BAR * PC.G / PC.C ** 3)
        holographic_entropy_nats = math.pi * radius_m ** 2 / planck_length ** 2
        holographic_bits = holographic_entropy_nats / math.log(2)

        # Schwarzschild radius for this energy
        schwarzschild_r = 2 * PC.G * energy_J / PC.C ** 4
        saturation = radius_m / schwarzschild_r if schwarzschild_r > 0 else float('inf')

        # L104: bits per Fe-56 nucleus at this energy density
        fe56_mass = 56 * 1.66054e-27  # 56 amu in kg
        fe56_radius = Fe.BCC_LATTICE_PM * 1e-12 / 2  # half lattice constant
        fe56_energy = fe56_mass * PC.C ** 2
        fe56_bits = 2 * math.pi * fe56_radius * fe56_energy / (PC.H_BAR * PC.C * math.log(2))

        return {
            "radius_m": radius_m,
            "energy_J": energy_J,
            "max_bits": max_bits,
            "max_bytes": max_bits / 8,
            "holographic_bits": holographic_bits,
            "saturation_ratio": saturation,
            "planck_length_m": planck_length,
            "fe56_bekenstein_bits": fe56_bits,
            "formula": "I_max = 2πRE / (ℏc ln 2)",
            "source": "Bekenstein 1981, CODATA 2022",
        }

    def computronium_efficiency(self, actual_ops_per_sec: float,
                                  actual_power_watts: float,
                                  mass_kg: float = None,
                                  temperature_K: float = 293.15) -> Dict[str, Any]:
        """
        Measure how close a real computing system is to computronium limits.

        Compares actual performance against:
          1. Bremermann limit (if mass given)
          2. Margolus-Levitin limit (from power = energy/second)
          3. Landauer efficiency (energy per bit-erasure vs theoretical minimum)
          4. Carnot efficiency for reversible computation

        Modern silicon: ~10⁻²⁹ Bremermann efficiency
        Superconducting qubits: ~10⁻²⁵ Bremermann efficiency
        """
        if actual_ops_per_sec <= 0 or actual_power_watts <= 0:
            return {"error": "ops_per_sec and power must be positive"}

        # Energy per operation
        energy_per_op = actual_power_watts / actual_ops_per_sec

        # Margolus-Levitin efficiency (energy-based)
        ml_limit = 2 * actual_power_watts / (math.pi * PC.H_BAR)
        ml_efficiency = actual_ops_per_sec / ml_limit

        # Landauer efficiency (entropy cost)
        landauer_min = PC.K_B * temperature_K * math.log(2)
        landauer_efficiency = landauer_min / energy_per_op

        result = {
            "actual_ops_per_sec": actual_ops_per_sec,
            "actual_power_watts": actual_power_watts,
            "energy_per_op_J": energy_per_op,
            "margolus_levitin_limit_ops": ml_limit,
            "ml_efficiency": ml_efficiency,
            "landauer_min_J_per_bit": landauer_min,
            "landauer_efficiency": landauer_efficiency,
            "orders_from_ml_limit": math.log10(ml_efficiency) if ml_efficiency > 0 else float('-inf'),
            "temperature_K": temperature_K,
        }

        if mass_kg is not None and mass_kg > 0:
            bremermann = mass_kg * PC.C ** 2 / (math.pi * PC.H_BAR)
            brem_efficiency = actual_ops_per_sec / bremermann
            result["bremermann_limit_ops"] = bremermann
            result["bremermann_efficiency"] = brem_efficiency
            result["orders_from_bremermann"] = math.log10(brem_efficiency) if brem_efficiency > 0 else float('-inf')

        return result

    def computronium_substrate_analysis(self, n_qubits: int = 26,
                                          gate_time_s: float = 20e-9,
                                          t1_s: float = 100e-6,
                                          t2_s: float = 150e-6,
                                          gate_fidelity: float = 0.999,
                                          temperature_K: float = 0.015) -> Dict[str, Any]:
        """
        Analyze a quantum processor as a computronium substrate.

        Evaluates how close a quantum computer approaches the fundamental
        limits of computation, given real hardware parameters.

        Default parameters approximate a state-of-art superconducting QPU
        (e.g., IBM Eagle/Heron class, 2024-2026).

        Parameters:
            n_qubits: Number of physical qubits
            gate_time_s: Single-gate execution time (default 20 ns)
            t1_s: Energy relaxation time T₁ (default 100 μs)
            t2_s: Dephasing time T₂ (default 150 μs)
            gate_fidelity: Single-gate fidelity (default 0.999)
            temperature_K: Operating temperature (default 15 mK)
        """
        # Energy per qubit at operating temperature
        qubit_energy = PC.K_B * temperature_K  # ~k_B T per degree of freedom

        # Total system energy
        total_energy = n_qubits * qubit_energy

        # Margolus-Levitin: max ops at this energy
        ml_ops = 2 * total_energy / (math.pi * PC.H_BAR)

        # Actual gate rate
        actual_gate_rate = 1.0 / gate_time_s
        actual_total_rate = actual_gate_rate * n_qubits  # parallel execution

        # Coherent computation window: limited by T₂
        coherent_gates = t2_s / gate_time_s
        useful_depth = coherent_gates * gate_fidelity ** coherent_gates

        # Landauer cost at this temperature
        landauer_cost = PC.K_B * temperature_K * math.log(2)

        # Quantum advantage: Hilbert space vs classical bits
        hilbert_dim = 2 ** n_qubits
        classical_bits_equivalent = n_qubits  # Holevo bound
        quantum_advantage = math.log2(hilbert_dim)  # exponential vs linear

        # Computronium fraction: actual useful computation vs theoretical max
        computronium_fraction = actual_total_rate / ml_ops if ml_ops > 0 else 0

        # L104 bridge: GOD_CODE resonance with coherent depth
        god_code_depth_ratio = coherent_gates / self.l104

        return {
            "n_qubits": n_qubits,
            "gate_time_s": gate_time_s,
            "t1_s": t1_s,
            "t2_s": t2_s,
            "gate_fidelity": gate_fidelity,
            "temperature_K": temperature_K,
            "qubit_energy_J": qubit_energy,
            "total_system_energy_J": total_energy,
            "ml_max_ops_per_sec": ml_ops,
            "actual_gate_rate_hz": actual_gate_rate,
            "actual_total_rate_hz": actual_total_rate,
            "computronium_fraction": computronium_fraction,
            "coherent_gate_depth": coherent_gates,
            "useful_circuit_depth": useful_depth,
            "landauer_cost_J_per_bit": landauer_cost,
            "hilbert_dimension": hilbert_dim,
            "quantum_advantage_bits": quantum_advantage,
            "god_code_depth_ratio": god_code_depth_ratio,
        }

    # ═══════════════════════════════════════════════════════════════════════
    #  RAYLEIGH LIMITS: Optical & Information Resolution
    # ═══════════════════════════════════════════════════════════════════════

    def rayleigh_criterion(self, wavelength_m: float,
                            aperture_m: float) -> Dict[str, Any]:
        """
        Rayleigh criterion: minimum angular separation for two point sources
        to be resolved by a circular aperture.

        θ = 1.22 λ / D   [radians]

        where:
            λ = wavelength of light
            D = diameter of the aperture

        The factor 1.22 comes from the first zero of the Airy pattern:
        J₁(x)/x = 0 at x = 3.8317... → θ = 3.8317/(πD/λ) = 1.21966.../D×λ

        At GOD_CODE wavelength (527.5 nm) with 1m telescope:
            θ ≈ 6.43 × 10⁻⁷ rad ≈ 0.133 arcsec

        Reference: Lord Rayleigh (1879). "Investigations in optics, with
        special reference to the spectroscope." Phil. Mag., 8(49), 261-274.
        """
        if wavelength_m <= 0 or aperture_m <= 0:
            return {"error": "wavelength and aperture must be positive"}

        theta_rad = RAYLEIGH_CONSTANT * wavelength_m / aperture_m
        theta_arcsec = theta_rad * (180 * 3600 / math.pi)
        theta_mas = theta_arcsec * 1000  # milliarcseconds

        # Sparrow limit (absolute minimum, ~5% tighter)
        sparrow_rad = SPARROW_FACTOR * theta_rad

        # Linear resolution at distance d: Δx = θ × d
        # At 1 AU (Earth-Sun):
        au = 1.496e11  # meters
        linear_at_au = theta_rad * au

        # At GOD_CODE wavelength
        god_code_wavelength = self.l104 * 1e-9  # 527.5 nm
        god_code_theta = RAYLEIGH_CONSTANT * god_code_wavelength / aperture_m

        # Information capacity of aperture (Shannon number)
        # N_S ≈ (D / (1.22 λ))² × π/4 = resolvable spots in FOV
        resolvable_spots = (aperture_m / (RAYLEIGH_CONSTANT * wavelength_m)) ** 2 * math.pi / 4
        information_bits = math.log2(resolvable_spots) if resolvable_spots > 1 else 0

        return {
            "wavelength_m": wavelength_m,
            "aperture_m": aperture_m,
            "rayleigh_angle_rad": theta_rad,
            "rayleigh_angle_arcsec": theta_arcsec,
            "rayleigh_angle_mas": theta_mas,
            "sparrow_angle_rad": sparrow_rad,
            "linear_resolution_at_1AU_m": linear_at_au,
            "god_code_angle_rad": god_code_theta,
            "resolvable_spots": resolvable_spots,
            "information_capacity_bits": information_bits,
            "formula": "θ = 1.22 λ / D",
            "source": "Lord Rayleigh 1879",
        }

    def rayleigh_jeans_law(self, frequency_hz: float,
                            temperature_K: float = 5778.0) -> Dict[str, Any]:
        """
        Rayleigh-Jeans law: classical approximation to blackbody spectral
        radiance (valid only at low frequencies / long wavelengths).

        B(ν, T) = 2ν² k_B T / c²   [W·sr⁻¹·m⁻²·Hz⁻¹]

        This law demonstrates the ultraviolet catastrophe — it predicts
        infinite total power at high frequencies, which Planck resolved
        with quantization: B_Planck = (2hν³/c²) / (e^(hν/k_BT) - 1)

        The ratio B_RJ / B_Planck measures departure from classical physics.

        Reference: Lord Rayleigh (1900), Jeans (1905).
        """
        if frequency_hz <= 0 or temperature_K <= 0:
            return {"error": "frequency and temperature must be positive"}

        # Rayleigh-Jeans spectral radiance
        B_rj = 2 * frequency_hz ** 2 * PC.K_B * temperature_K / PC.C ** 2

        # Planck spectral radiance (exact)
        h_nu = PC.H * frequency_hz
        kT = PC.K_B * temperature_K
        exponent = h_nu / kT
        if exponent < 700:  # Avoid overflow
            B_planck = (2 * PC.H * frequency_hz ** 3 / PC.C ** 2) / (math.exp(exponent) - 1)
        else:
            B_planck = 0.0  # Wien regime: exponentially suppressed

        # Classical-quantum departure ratio
        departure = B_rj / B_planck if B_planck > 0 else float('inf')

        # Wavelength at this frequency
        wavelength_m = PC.C / frequency_hz
        wavelength_nm = wavelength_m * 1e9

        # Is this in the classical regime? (hν << k_BT)
        is_classical = exponent < 0.01
        regime = "CLASSICAL (hν << k_BT)" if is_classical else (
            "PLANCK REGIME" if exponent < 10 else "WIEN REGIME (hν >> k_BT)"
        )

        # Energy density: u(ν) = 4π/c × B(ν,T)
        u_rj = 4 * math.pi * B_rj / PC.C

        # Ultraviolet catastrophe: total power diverges as ∫₀^∞ B_RJ dν → ∞
        # But Planck integral converges: σT⁴ where σ = 2π⁵k_B⁴/(15c²h³)
        stefan_boltzmann = 2 * math.pi ** 5 * PC.K_B ** 4 / (15 * PC.C ** 2 * PC.H ** 3)
        planck_total_power = stefan_boltzmann * temperature_K ** 4

        return {
            "frequency_hz": frequency_hz,
            "wavelength_nm": wavelength_nm,
            "temperature_K": temperature_K,
            "B_rayleigh_jeans": B_rj,
            "B_planck": B_planck,
            "classical_departure_ratio": departure,
            "regime": regime,
            "h_nu_over_kT": exponent,
            "energy_density_rj_J_m3_Hz": u_rj,
            "stefan_boltzmann_constant": stefan_boltzmann,
            "planck_total_power_W_m2": planck_total_power,
            "formula": "B(ν,T) = 2ν² k_B T / c²",
            "source": "Rayleigh 1900, Jeans 1905",
        }

    def rayleigh_scattering(self, wavelength_m: float,
                             particle_radius_m: float = 1e-10,
                             refractive_index: float = 1.00029) -> Dict[str, Any]:
        """
        Rayleigh scattering cross-section: elastic scattering of light
        by particles much smaller than the wavelength.

        σ = (8π/3) × (2πr/λ)⁴ × r² × ((n²-1)/(n²+2))²

        The λ⁻⁴ dependence explains why the sky is blue and sunsets are red.

        For atmospheric N₂ at 527.5 nm (GOD_CODE wavelength):
            σ ≈ 5.1 × 10⁻³¹ m²

        Reference: Lord Rayleigh (1871). "On the light from the sky, its
        polarization and colour." Phil. Mag., 41, 107-120.
        """
        if wavelength_m <= 0 or particle_radius_m <= 0:
            return {"error": "wavelength and radius must be positive"}

        n = refractive_index
        r = particle_radius_m
        lam = wavelength_m

        # Size parameter
        x = 2 * math.pi * r / lam

        # Rayleigh regime check: x << 1
        if x > 0.3:
            regime = "MIE (x > 0.3, Rayleigh approximation breaks down)"
        else:
            regime = "RAYLEIGH (x << 1)"

        # Clausius-Mossotti factor (real dielectric)
        cm_factor = ((n ** 2 - 1) / (n ** 2 + 2)) ** 2

        # Rayleigh cross-section
        sigma = (8 * math.pi / 3) * x ** 4 * r ** 2 * cm_factor

        # Intensity ratio I/I₀ = Nσ (for N scatterers per unit area)
        # Atmosphere: ~2.5 × 10²⁵ molecules/m³, path ~8 km
        atm_density = 2.504e25  # molecules/m³ at STP
        atm_path = 8000  # meters (scale height)
        optical_depth = atm_density * sigma * atm_path
        transmission = math.exp(-optical_depth) if optical_depth < 700 else 0

        # Compare GOD_CODE wavelength vs blue (450 nm) vs red (700 nm)
        god_code_x = 2 * math.pi * r / (self.l104 * 1e-9)
        blue_ratio = (lam / 450e-9) ** 4 if lam != 450e-9 else 1.0
        red_ratio = (lam / 700e-9) ** 4 if lam != 700e-9 else 1.0

        return {
            "wavelength_m": wavelength_m,
            "wavelength_nm": wavelength_m * 1e9,
            "particle_radius_m": r,
            "refractive_index": n,
            "size_parameter_x": x,
            "regime": regime,
            "cross_section_m2": sigma,
            "clausius_mossotti": cm_factor,
            "atm_optical_depth": optical_depth,
            "atm_transmission": transmission,
            "blue_scatter_ratio": blue_ratio,
            "red_scatter_ratio": red_ratio,
            "formula": "σ = (8π/3)(2πr/λ)⁴ r² ((n²-1)/(n²+2))²",
            "source": "Lord Rayleigh 1871",
        }

    def abbe_diffraction_limit(self, wavelength_m: float,
                                numerical_aperture: float = 1.4,
                                refractive_index: float = 1.515) -> Dict[str, Any]:
        """
        Abbe diffraction limit: minimum resolvable feature size in microscopy.

        d_min = λ / (2 × NA)

        where NA = n × sin(θ) is the numerical aperture (max ~1.4 for oil immersion).

        At GOD_CODE wavelength (527.5 nm) with NA=1.4:
            d_min ≈ 188 nm

        This is the fundamental limit of optical lithography and microscopy.
        Sub-diffraction techniques (STED, PALM, SIM) can beat this by ~10×.

        Reference: Abbe, E. (1873). "Beiträge zur Theorie des Mikroskops
        und der mikroskopischen Wahrnehmung." Archiv für mikroskopische
        Anatomie, 9(1), 413-468.
        """
        if wavelength_m <= 0 or numerical_aperture <= 0:
            return {"error": "wavelength and NA must be positive"}

        d_min = wavelength_m / (2 * numerical_aperture)
        d_min_nm = d_min * 1e9

        # Maximum NA = refractive index (sin θ → 1)
        max_na = refractive_index
        d_absolute_min = wavelength_m / (2 * max_na)

        # Volume resolution (3D): axial resolution is worse
        # Δz ≈ 2nλ / NA²
        delta_z = 2 * refractive_index * wavelength_m / numerical_aperture ** 2
        voxel_volume = d_min ** 2 * delta_z

        # Information density: bits per voxel at Nyquist sampling
        # Each voxel encodes ~1 resolvable bit of spatial information
        voxels_per_m3 = 1.0 / voxel_volume if voxel_volume > 0 else 0

        # GOD_CODE scale: resolution at 527.5 nm
        god_code_d_min = (self.l104 * 1e-9) / (2 * numerical_aperture)

        # Semiconductor lithography: feature size vs diffraction limit
        # Modern EUV (13.5 nm): d_min = 13.5/(2×0.33) ≈ 20 nm
        euv_wavelength = 13.5e-9
        euv_na = 0.33
        euv_d_min = euv_wavelength / (2 * euv_na)

        return {
            "wavelength_m": wavelength_m,
            "wavelength_nm": wavelength_m * 1e9,
            "numerical_aperture": numerical_aperture,
            "refractive_index": refractive_index,
            "d_min_m": d_min,
            "d_min_nm": d_min_nm,
            "d_absolute_min_nm": d_absolute_min * 1e9,
            "axial_resolution_m": delta_z,
            "voxel_volume_m3": voxel_volume,
            "voxels_per_m3": voxels_per_m3,
            "god_code_d_min_nm": god_code_d_min * 1e9,
            "euv_litho_d_min_nm": euv_d_min * 1e9,
            "formula": "d_min = λ / (2 NA)",
            "source": "Abbe 1873",
        }

    def information_optical_bridge(self, wavelength_m: float = None,
                                     aperture_m: float = 1.0,
                                     volume_m3: float = 1e-3,
                                     temperature_K: float = 293.15) -> Dict[str, Any]:
        """
        Bridge between optical (Rayleigh) and information (Bekenstein/computronium)
        limits: how many resolvable bits can a volume encode, given that each
        bit occupies at least one diffraction-limited voxel?

        Combines:
          - Abbe/Rayleigh: minimum voxel size → max spatial bits
          - Landauer: energy cost per bit at temperature T
          - Bekenstein: absolute bit limit for the volume

        Yields the computronium-optical efficiency: what fraction of
        Bekenstein bits can be physically resolved and erased.
        """
        if wavelength_m is None:
            wavelength_m = self.l104 * 1e-9  # GOD_CODE wavelength

        if volume_m3 <= 0 or aperture_m <= 0 or wavelength_m <= 0:
            return {"error": "all dimensions must be positive"}

        # Optical resolution: Rayleigh-limited voxel
        rayleigh_angle = RAYLEIGH_CONSTANT * wavelength_m / aperture_m
        # Minimum feature in the volume
        d_min = wavelength_m / 2  # Abbe limit at max NA
        voxel_vol = d_min ** 3
        optical_bits = volume_m3 / voxel_vol if voxel_vol > 0 else 0

        # Bekenstein bound for this volume
        radius_m = (3 * volume_m3 / (4 * math.pi)) ** (1.0 / 3.0)
        mass_density = 1000  # kg/m³ (water density as reference)
        mass_kg = mass_density * volume_m3
        energy_J = mass_kg * PC.C ** 2
        bekenstein_bits = 2 * math.pi * radius_m * energy_J / (PC.H_BAR * PC.C * math.log(2))

        # Landauer: energy to populate all optical bits
        landauer_per_bit = PC.K_B * temperature_K * math.log(2)
        landauer_total = optical_bits * landauer_per_bit

        # Efficiency ratios
        optical_to_bekenstein = optical_bits / bekenstein_bits if bekenstein_bits > 0 else 0
        energy_to_populate = landauer_total / energy_J if energy_J > 0 else 0

        # Scale hierarchy
        planck_length = math.sqrt(PC.H_BAR * PC.G / PC.C ** 3)
        planck_voxel = planck_length ** 3
        planck_bits_in_volume = volume_m3 / planck_voxel

        return {
            "wavelength_m": wavelength_m,
            "wavelength_nm": wavelength_m * 1e9,
            "volume_m3": volume_m3,
            "d_min_m": d_min,
            "optical_voxel_m3": voxel_vol,
            "optical_bits": optical_bits,
            "bekenstein_bits": bekenstein_bits,
            "planck_bits": planck_bits_in_volume,
            "optical_to_bekenstein_ratio": optical_to_bekenstein,
            "landauer_per_bit_J": landauer_per_bit,
            "landauer_total_to_populate_J": landauer_total,
            "energy_fraction_to_populate": energy_to_populate,
            "hierarchy": {
                "planck_bits": planck_bits_in_volume,
                "bekenstein_bits": bekenstein_bits,
                "optical_bits": optical_bits,
                "gap_planck_to_bekenstein": math.log10(bekenstein_bits / planck_bits_in_volume) if planck_bits_in_volume > 0 and bekenstein_bits > 0 else 0,
                "gap_optical_to_bekenstein": math.log10(optical_to_bekenstein) if optical_to_bekenstein > 0 else float('-inf'),
            },
        }

    # ═══════════════════════════════════════════════════════════════════════
    #  COMBINED ANALYSES
    # ═══════════════════════════════════════════════════════════════════════

    def full_computronium_analysis(self, mass_kg: float = 1.0,
                                     volume_m3: float = 1e-3,
                                     temperature_K: float = 293.15) -> Dict[str, Any]:
        """
        Complete computronium analysis: all fundamental limits for a given
        mass, volume, and temperature.
        """
        return {
            "bremermann": self.bremermann_limit(mass_kg),
            "margolus_levitin": self.margolus_levitin(mass_kg=mass_kg),
            "landauer": self.landauer_erasure(temperature_K),
            "lloyd": self.lloyd_ultimate_laptop(mass_kg, volume_m3),
            "bekenstein": self.bekenstein_bound(
                radius_m=(3 * volume_m3 / (4 * math.pi)) ** (1.0 / 3.0),
                mass_kg=mass_kg,
            ),
        }

    def full_rayleigh_analysis(self, wavelength_m: float = None,
                                 aperture_m: float = 1.0,
                                 temperature_K: float = 5778.0) -> Dict[str, Any]:
        """
        Complete Rayleigh analysis: all optical/resolution limits at a given
        wavelength, aperture, and temperature.
        """
        if wavelength_m is None:
            wavelength_m = self.l104 * 1e-9  # GOD_CODE wavelength

        frequency_hz = PC.C / wavelength_m

        return {
            "rayleigh_criterion": self.rayleigh_criterion(wavelength_m, aperture_m),
            "rayleigh_jeans": self.rayleigh_jeans_law(frequency_hz, temperature_K),
            "rayleigh_scattering": self.rayleigh_scattering(wavelength_m),
            "abbe_limit": self.abbe_diffraction_limit(wavelength_m),
            "optical_information_bridge": self.information_optical_bridge(
                wavelength_m, aperture_m, temperature_K=temperature_K,
            ),
        }

    def computronium_rayleigh_bridge(self, mass_kg: float = 1.0,
                                       volume_m3: float = 1e-3,
                                       wavelength_m: float = None,
                                       temperature_K: float = 293.15) -> Dict[str, Any]:
        """
        Bridge analysis: connects computronium (information limits) with
        Rayleigh (resolution limits) to determine the ultimate information
        processing density of a physical system.

        Key insight: Bekenstein limits total information, Rayleigh limits
        spatial resolution, Landauer limits energy cost, and Margolus-Levitin
        limits processing speed. Together they constrain what is physically
        possible for any computing substrate.
        """
        if wavelength_m is None:
            wavelength_m = self.l104 * 1e-9

        computronium = self.full_computronium_analysis(mass_kg, volume_m3, temperature_K)
        rayleigh = self.full_rayleigh_analysis(wavelength_m, temperature_K=temperature_K)

        # Bridge metrics
        bekenstein_bits = computronium["bekenstein"]["max_bits"]
        optical_bits = rayleigh["optical_information_bridge"]["optical_bits"]
        lloyd_ops = computronium["lloyd"]["max_ops_per_sec"]
        landauer_cost = computronium["landauer"]["energy_per_bit_J"]

        # Ultimate information processing rate: ops/s limited by
        # min(Margolus-Levitin, Bekenstein_bits / gate_time)
        planck_time = math.sqrt(PC.H_BAR * PC.G / PC.C ** 5)
        bekenstein_rate = bekenstein_bits / planck_time if planck_time > 0 else 0

        # Spatial-temporal information density
        info_density_spatial = bekenstein_bits / volume_m3 if volume_m3 > 0 else 0
        info_density_spacetime = lloyd_ops / volume_m3 if volume_m3 > 0 else 0

        return {
            "computronium": computronium,
            "rayleigh": rayleigh,
            "bridge": {
                "bekenstein_bits": bekenstein_bits,
                "optical_bits": optical_bits,
                "optical_to_bekenstein": optical_bits / bekenstein_bits if bekenstein_bits > 0 else 0,
                "lloyd_ops_per_sec": lloyd_ops,
                "info_density_bits_per_m3": info_density_spatial,
                "info_density_ops_per_sec_per_m3": info_density_spacetime,
                "landauer_cost_J_per_bit": landauer_cost,
                "god_code_wavelength_nm": self.l104,
                "phi_scaling": PHI,
            },
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "subsystem": "ComputroniumSubsystem",
            "version": "1.0.0",
            "algorithms": [
                "bremermann_limit",
                "margolus_levitin",
                "landauer_erasure",
                "lloyd_ultimate_laptop",
                "bekenstein_bound",
                "computronium_efficiency",
                "computronium_substrate_analysis",
                "rayleigh_criterion",
                "rayleigh_jeans_law",
                "rayleigh_scattering",
                "abbe_diffraction_limit",
                "information_optical_bridge",
                "full_computronium_analysis",
                "full_rayleigh_analysis",
                "computronium_rayleigh_bridge",
            ],
            "constants": {
                "bremermann_limit_per_kg": BREMERMANN_LIMIT,
                "margolus_levitin_per_J": MARGOLUS_LEVITIN_CONSTANT,
                "landauer_room_temp_J": LANDAUER_ROOM_TEMP,
                "lloyd_1kg_ops": LLOYD_1KG_OPS_PER_SEC,
                "rayleigh_constant": RAYLEIGH_CONSTANT,
            },
        }
