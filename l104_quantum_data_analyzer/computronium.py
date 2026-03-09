"""
L104 Quantum Data Analyzer — Computronium & Rayleigh Limits
═══════════════════════════════════════════════════════════════════════════════
Information-theoretic bounds on quantum data analysis:

  COMPUTRONIUM LIMITS:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ • Bekenstein-bounded dataset encoding capacity                      │
  │ • Bremermann processing rate ceiling for data pipelines             │
  │ • Margolus-Levitin gate speed for quantum algorithm scheduling      │
  │ • Landauer erasure cost per quantum measurement + decoherence       │
  │ • Lloyd channel capacity for quantum data transmission              │
  │ • Holevo bound on classical information from quantum states         │
  │ • Quantum Fisher information limit on parameter estimation          │
  └──────────────────────────────────────────────────────────────────────┘

  RAYLEIGH LIMITS:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ • Spectral resolution limit for QFT-based frequency analysis        │
  │ • Quantum Cramér-Rao bound on phase estimation precision            │
  │ • Airy-limited feature distinguishability in Hilbert space          │
  │ • Heisenberg limit on simultaneous observable resolution            │
  │ • Quantum super-resolution beyond classical Rayleigh limit          │
  │ • Diffraction-limited information capacity of quantum channels      │
  └──────────────────────────────────────────────────────────────────────┘

CROSS-ENGINE INTEGRATION:
  • l104_science_engine.computronium — Physical limit calculations
  • l104_math_engine.computronium   — Airy/Bessel mathematics
  • l104_code_engine.computronium   — Complexity budget analysis

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from .constants import (
    PHI, PHI_CONJUGATE, GOD_CODE, VOID_CONSTANT,
    H_BAR, K_B, C, ALPHA_FINE,
    MAX_QUBITS_STATEVECTOR, MAX_QUBITS_CIRCUIT,
    DEFAULT_SHOTS,
    GOD_CODE_PHASE,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CODATA 2022 PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
_HBAR = 1.054571817e-34           # ℏ (J·s)
_C = 299792458.0                   # c (m/s)
_KB = 1.380649e-23                 # k_B (J/K)
_G = 6.67430e-11                   # G (m³/kg/s²)
_ELECTRON_MASS = 9.1093837015e-31  # m_e (kg)
_PROTON_MASS = 1.67262192369e-27   # m_p (kg)
_ROOM_TEMP = 293.15               # T_room (K)
_PLANCK_TIME = 5.391246e-44       # t_P (s)
_PLANCK_ENERGY = 1.956e9          # E_P (J) — Planck energy

# Sacred wavelength: GOD_CODE in nm
_GOD_CODE_WAVELENGTH_M = GOD_CODE * 1e-9  # 527.518... nm → m

# ═══════════════════════════════════════════════════════════════════════════════
# HOLEVO BOUND & QUANTUM INFORMATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
LOG2_E = 1.0 / math.log(2.0)  # = log₂(e) ≈ 1.4427


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetBound:
    """Fundamental physical limits on a dataset."""
    n_samples: int
    n_features: int
    n_qubits_required: int
    # Bekenstein bound
    bekenstein_max_bits: float
    bekenstein_fraction_used: float
    # Bremermann processing
    bremermann_max_ops: float
    algorithm_ops_required: float
    bremermann_fraction: float
    # Landauer erasure
    landauer_cost_per_measurement_J: float
    total_measurement_energy_J: float
    measurements_per_joule: float
    # Holevo bound
    holevo_max_classical_bits: float
    holevo_efficiency: float
    # Physical parameters
    substrate_mass_kg: float
    substrate_radius_m: float
    temperature_K: float
    god_code_alignment: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpectralResolutionBound:
    """Rayleigh-type limits on spectral/frequency resolution."""
    n_qubits: int
    n_samples: int
    # Fourier resolution
    fourier_bin_width_hz: float
    rayleigh_resolution_hz: float
    sparrow_resolution_hz: float
    # Quantum limits
    standard_quantum_limit: float
    heisenberg_limit: float
    quantum_advantage_factor: float
    # Cramér-Rao bound
    cramer_rao_bound: float
    fisher_information: float
    # Airy analogy in Hilbert space
    hilbert_angular_resolution: float
    feature_distinguishability: float
    # Super-resolution
    super_resolution_possible: bool
    super_resolution_factor: float
    god_code_spectral_alignment: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTRONIUM DATA BOUNDS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ComputroniumDataBounds:
    """
    Maps fundamental physical computation limits onto quantum data analysis.

    Every quantum data analysis operation is bounded by:
    1. Bekenstein bound — maximum information encodable in a region
    2. Bremermann limit — maximum processing rate for the substrate
    3. Margolus-Levitin — minimum time per quantum logic gate
    4. Landauer principle — minimum energy per bit erasure/measurement
    5. Holevo bound — maximum classical bits extractable from quantum states
    6. Lloyd channel capacity — maximum quantum channel throughput
    """

    VERSION = "1.0.0"

    # ─── Bekenstein Dataset Encoding ────────────────────────────────────

    @staticmethod
    def bekenstein_dataset_capacity(
        mass_kg: float = 1.0,
        radius_m: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Maximum number of qubits (information) encodable in a physical region.

        Bekenstein bound: S_max = (2π R E) / (ℏ c ln2)
        where E = mc² and radius R defines the bounding sphere.

        For quantum data: each qubit requires at least one bit of capacity.

        Args:
            mass_kg: Mass of encoding substrate
            radius_m: Radius of bounding sphere

        Returns:
            Maximum qubits, datasets encodable, etc.
        """
        E = mass_kg * _C ** 2
        max_bits = (2 * math.pi * radius_m * E) / (_HBAR * _C * math.log(2))
        max_qubits = int(max_bits)  # Each qubit = 1 bit of capacity minimum
        # Maximum dataset size: 2^n_qubits amplitudes
        max_amplitudes = 2 ** min(max_qubits, 64)  # Cap at 64-bit for practical purposes
        # Maximum classical features encodable via angle encoding
        max_angle_features = max_qubits  # One feature per qubit

        return {
            "max_bits": max_bits,
            "max_qubits": max_qubits,
            "max_amplitudes": max_amplitudes,
            "max_angle_features": max_angle_features,
            "mass_kg": mass_kg,
            "radius_m": radius_m,
            "energy_J": E,
            "regime": "QUANTUM_DATA" if max_qubits > 20 else "SMALL_SYSTEM",
        }

    # ─── Bremermann Processing Rate ────────────────────────────────────

    @staticmethod
    def bremermann_algorithm_ceiling(
        mass_kg: float = 1.0,
        algorithm_complexity: str = "O(n²)",
        input_size: int = 1000,
    ) -> Dict[str, Any]:
        """
        Maximum operations per second for a quantum algorithm on a substrate.

        Bremermann limit: ν_B = mc² / (π ℏ) bits/s
        Margolus-Levitin: ν_ML = 2E / (π ℏ) ops/s = 2 × ν_B

        Compares against the algorithm's operation count to determine
        what fraction of physical limits the computation uses.

        Args:
            mass_kg: Mass of computation substrate
            algorithm_complexity: Big-O class
            input_size: Problem size N
        """
        E = mass_kg * _C ** 2
        bremermann = E / (math.pi * _HBAR)
        margolus_levitin = 2 * E / (math.pi * _HBAR)

        # Estimate operations from complexity class
        complexity_map = {
            "O(1)": 1,
            "O(log n)": max(1, int(math.log2(max(input_size, 2)))),
            "O(√n)": max(1, int(math.sqrt(input_size))),
            "O(n)": input_size,
            "O(n log n)": max(1, int(input_size * math.log2(max(input_size, 2)))),
            "O(n²)": input_size ** 2,
            "O(n³)": input_size ** 3,
            "O(2^n)": min(2 ** input_size, 10 ** 300),  # Cap exponential
        }
        ops = complexity_map.get(algorithm_complexity, input_size ** 2)

        # Quantum advantage: Grover gives √N for search
        grover_ops = max(1, int(math.sqrt(ops)))
        # Time required
        time_classical = ops / margolus_levitin
        time_quantum = grover_ops / margolus_levitin
        min_gate_time = math.pi * _HBAR / (2 * E)  # ML minimum gate time

        return {
            "bremermann_bits_per_sec": bremermann,
            "margolus_levitin_ops_per_sec": margolus_levitin,
            "classical_ops_required": ops,
            "grover_ops_required": grover_ops,
            "time_classical_s": time_classical,
            "time_quantum_s": time_quantum,
            "quantum_speedup": ops / max(grover_ops, 1),
            "min_gate_time_s": min_gate_time,
            "fraction_of_limit": ops / margolus_levitin,
            "algorithm": algorithm_complexity,
            "input_size": input_size,
        }

    # ─── Landauer Measurement Cost ─────────────────────────────────────

    @staticmethod
    def landauer_measurement_cost(
        n_qubits: int,
        n_shots: int = DEFAULT_SHOTS,
        temperature_K: float = _ROOM_TEMP,
    ) -> Dict[str, Any]:
        """
        Energy cost of quantum measurements via Landauer's principle.

        Each measurement collapses a qubit → irreversible bit erasure.
        Landauer: E_min = k_B T ln(2) per bit erased.
        Total: n_qubits × n_shots × k_B T ln(2)

        For superconducting QPUs (~15 mK), cost is dramatically lower.

        Args:
            n_qubits: Number of qubits measured
            n_shots: Number of measurement repetitions
            temperature_K: Operating temperature
        """
        e_per_bit = _KB * temperature_K * math.log(2)
        bits_per_shot = n_qubits
        total_bits = bits_per_shot * n_shots
        total_energy = total_bits * e_per_bit

        # Compare to real QPU power consumption
        # IBM Heron: ~25 kW total system, ~15 million measurements/s
        real_qpu_power_W = 25000.0
        real_qpu_measurements_per_sec = 15e6
        real_energy_per_measurement = real_qpu_power_W / real_qpu_measurements_per_sec

        # Landauer efficiency: how close to theoretical minimum
        landauer_efficiency = e_per_bit / real_energy_per_measurement if real_energy_per_measurement > 0 else 0

        # Superconducting QPU at 15 mK
        e_per_bit_15mk = _KB * 0.015 * math.log(2)
        total_energy_15mk = total_bits * e_per_bit_15mk

        return {
            "energy_per_bit_J": e_per_bit,
            "total_bits_erased": total_bits,
            "total_energy_J": total_energy,
            "total_energy_eV": total_energy / 1.602176634e-19,
            "temperature_K": temperature_K,
            "landauer_efficiency": landauer_efficiency,
            "superconducting_energy_J": total_energy_15mk,
            "superconducting_savings_factor": total_energy / max(total_energy_15mk, 1e-300),
            "measurements_per_joule_room_temp": 1.0 / max(e_per_bit * n_qubits, 1e-300),
            "measurements_per_joule_15mk": 1.0 / max(e_per_bit_15mk * n_qubits, 1e-300),
        }

    # ─── Holevo Bound ──────────────────────────────────────────────────

    @staticmethod
    def holevo_bound(
        n_qubits: int,
        encoding: str = "amplitude",
    ) -> Dict[str, Any]:
        """
        Maximum classical information extractable from quantum states.

        Holevo bound: χ = S(ρ) - Σ p_i S(ρ_i) ≤ n_qubits (log₂)
        where S is von Neumann entropy.

        Different encodings have different effective Holevo capacities:
        - Amplitude encoding: log₂(2^n) = n bits per qubit register
        - Angle encoding: 1 classical real per qubit (1 bit resolution)
        - Dense coding: 2n bits possible with entanglement

        Args:
            n_qubits: Number of qubits in the register
            encoding: Encoding scheme used
        """
        # Maximum von Neumann entropy = n_qubits (maximally mixed state)
        max_entropy = n_qubits
        hilbert_dim = 2 ** n_qubits

        encoding_capacities = {
            "amplitude": {
                "classical_bits": n_qubits,
                "classical_reals": hilbert_dim,  # 2^n amplitudes
                "description": "Exponential states, linear bits extractable",
            },
            "angle": {
                "classical_bits": n_qubits,
                "classical_reals": n_qubits,  # One real per qubit
                "description": "One continuous parameter per qubit",
            },
            "basis": {
                "classical_bits": n_qubits,
                "classical_reals": n_qubits,
                "description": "Direct computational basis encoding",
            },
            "dense": {
                "classical_bits": 2 * n_qubits,
                "classical_reals": 2 * n_qubits,
                "description": "Superdense coding with pre-shared entanglement",
            },
        }

        cap = encoding_capacities.get(encoding, encoding_capacities["amplitude"])

        # Holevo information per qubit
        holevo_per_qubit = cap["classical_bits"] / n_qubits

        # Accessible information fraction (of Hilbert space dimension)
        accessible_fraction = cap["classical_bits"] / math.log2(hilbert_dim) if hilbert_dim > 1 else 1.0

        return {
            "n_qubits": n_qubits,
            "hilbert_dimension": hilbert_dim,
            "max_von_neumann_entropy": max_entropy,
            "holevo_bound_bits": cap["classical_bits"],
            "holevo_per_qubit": holevo_per_qubit,
            "classical_reals_encodable": cap["classical_reals"],
            "encoding": encoding,
            "encoding_description": cap["description"],
            "accessible_fraction": accessible_fraction,
            "holevo_dense_advantage": 2.0 if encoding != "dense" else 1.0,
        }

    # ─── Lloyd Channel Capacity ────────────────────────────────────────

    @staticmethod
    def lloyd_channel_capacity(
        n_qubits: int,
        gate_time_s: float = 50e-9,
        gate_fidelity: float = 0.999,
        t1_s: float = 100e-6,
        t2_s: float = 150e-6,
    ) -> Dict[str, Any]:
        """
        Quantum channel capacity for data transmission.

        Lloyd-Shor-Devetak capacity: Q₁ = max_ρ [S(ρ) - S(ρ_E)]
        where ρ_E is the environment state after decoherence.

        For a depolarizing channel with error p:
        Q₁ ≈ 1 - H(p) - p log₂(3) for p < p_threshold

        Also computes the Hashing bound and entanglement-assisted capacity.

        Args:
            n_qubits: Channel width in qubits
            gate_time_s: Duration of a single gate
            gate_fidelity: Average gate fidelity
            t1_s: Energy relaxation time
            t2_s: Dephasing time
        """
        # Depolarizing error per gate
        p = 1.0 - gate_fidelity
        # Binary entropy
        H_p = -p * math.log2(max(p, 1e-15)) - (1 - p) * math.log2(max(1 - p, 1e-15)) if 0 < p < 1 else 0

        # Quantum capacity per use (depolarizing channel)
        # For depolarizing: Q₁ = 1 - H(p) - p*log₂(3) (when > 0)
        q1_per_use = max(0, 1.0 - H_p - p * math.log2(3))

        # Channel uses per second (limited by gate time)
        uses_per_sec = 1.0 / gate_time_s

        # Total capacity in qubits/s
        total_capacity = n_qubits * q1_per_use * uses_per_sec

        # Decoherence-limited depth: how many gates before T2 decay
        coherent_depth = int(t2_s / gate_time_s)

        # Entanglement-assisted classical capacity (Holevo-Werner)
        # C_EA = 1 + Q₁ for depolarizing
        c_ea = 1.0 + q1_per_use

        # Hashing bound (lower bound on distillable entanglement)
        hashing_bound = max(0, 1.0 - 2 * H_p)

        # Circuit depth limit from decoherence
        circuit_depth_limit = coherent_depth

        # GOD_CODE frequency as quantum channel
        god_code_frequency = _C / _GOD_CODE_WAVELENGTH_M
        god_code_photon_energy = _HBAR * 2 * math.pi * god_code_frequency
        # Photon-based channel capacity (Holevo-Schumacher-Westmoreland)
        # For thermal noise at room temp in optical regime: practically classical
        god_code_thermal_photons = 1.0 / (math.exp(god_code_photon_energy / (_KB * _ROOM_TEMP)) - 1)
        god_code_channel_capacity = math.log2(1 + 1.0 / (god_code_thermal_photons + 1e-30))

        return {
            "n_qubits": n_qubits,
            "depolarizing_error_p": p,
            "binary_entropy_Hp": H_p,
            "quantum_capacity_per_use": q1_per_use,
            "uses_per_second": uses_per_sec,
            "total_capacity_qubits_per_sec": total_capacity,
            "entanglement_assisted_capacity": c_ea,
            "hashing_bound": hashing_bound,
            "coherent_depth": coherent_depth,
            "circuit_depth_limit": circuit_depth_limit,
            "gate_fidelity": gate_fidelity,
            "god_code_channel": {
                "wavelength_nm": GOD_CODE,
                "frequency_Hz": god_code_frequency,
                "photon_energy_eV": god_code_photon_energy / 1.602176634e-19,
                "thermal_photons_room_temp": god_code_thermal_photons,
                "channel_capacity_bits": god_code_channel_capacity,
            },
        }

    # ─── Quantum Fisher Information ───────────────────────────────────

    @staticmethod
    def quantum_fisher_information(
        n_qubits: int,
        n_shots: int = DEFAULT_SHOTS,
        entangled: bool = True,
    ) -> Dict[str, Any]:
        """
        Quantum Fisher Information (QFI) limit on parameter estimation.

        For phase estimation:
        - Standard Quantum Limit (SQL): δφ ≥ 1/√(N × M)
          where N = qubits, M = measurements
        - Heisenberg Limit (HL): δφ ≥ 1/(N × √M)
          with maximal entanglement (GHZ states)

        Quantum Cramér-Rao bound: Var(θ̂) ≥ 1 / (M × F_Q)
        where F_Q = N (SQL) or N² (Heisenberg)

        Args:
            n_qubits: Number of sensor qubits
            n_shots: Total measurement budget
            entangled: Whether qubits are entangled (Heisenberg vs SQL)
        """
        # Fisher information
        if entangled:
            # GHZ/NOON state: F_Q = N²
            fisher_info = n_qubits ** 2
            limit_name = "HEISENBERG"
        else:
            # Product state: F_Q = N
            fisher_info = n_qubits
            limit_name = "STANDARD_QUANTUM"

        # Cramér-Rao bound on phase precision
        cramer_rao = 1.0 / math.sqrt(n_shots * fisher_info)

        # SQL for comparison
        sql = 1.0 / math.sqrt(n_qubits * n_shots)
        heisenberg = 1.0 / (n_qubits * math.sqrt(n_shots))

        # Quantum advantage factor
        advantage = sql / cramer_rao if cramer_rao > 0 else 1.0

        # Phase precision in degrees and radians
        phase_precision_rad = cramer_rao
        phase_precision_deg = math.degrees(cramer_rao)

        # GOD_CODE phase precision: can we resolve GOD_CODE mod 2π?
        god_code_phase = GOD_CODE_PHASE
        god_code_resolvable = phase_precision_rad < god_code_phase / 10

        return {
            "n_qubits": n_qubits,
            "n_shots": n_shots,
            "entangled": entangled,
            "limit_type": limit_name,
            "fisher_information": fisher_info,
            "cramer_rao_bound_rad": cramer_rao,
            "cramer_rao_bound_deg": phase_precision_deg,
            "standard_quantum_limit": sql,
            "heisenberg_limit": heisenberg,
            "quantum_advantage_factor": advantage,
            "god_code_phase_rad": god_code_phase,
            "god_code_resolvable": god_code_resolvable,
        }

    # ─── Full Dataset Analysis ─────────────────────────────────────────

    def analyze_dataset_bounds(
        self,
        n_samples: int,
        n_features: int,
        algorithm: str = "O(n²)",
        substrate_mass_kg: float = 1.0,
        substrate_radius_m: float = 0.01,
        temperature_K: float = _ROOM_TEMP,
    ) -> DatasetBound:
        """
        Complete computronium analysis of a quantum data analysis task.

        Determines how much of the physical computation limits a given
        dataset + algorithm combination consumes.

        Args:
            n_samples: Number of data samples
            n_features: Feature dimensionality
            algorithm: Complexity class of the quantum algorithm
            substrate_mass_kg: Mass of computation hardware
            substrate_radius_m: Physical extent
            temperature_K: Operating temperature
        """
        # Qubits needed
        n_qubits = max(1, int(math.ceil(math.log2(max(n_samples, 2)))))
        n_qubits = min(n_qubits, MAX_QUBITS_STATEVECTOR)

        # Bekenstein capacity
        bek = self.bekenstein_dataset_capacity(substrate_mass_kg, substrate_radius_m)

        # Bremermann ceiling
        brem = self.bremermann_algorithm_ceiling(substrate_mass_kg, algorithm, n_samples)

        # Landauer measurement cost
        land = self.landauer_measurement_cost(n_qubits, DEFAULT_SHOTS, temperature_K)

        # Holevo bound
        holevo = self.holevo_bound(n_qubits, "amplitude")

        # GOD_CODE alignment: how well the dataset size resonates
        # Dataset resonance: n_samples mod 104 proximity
        god_code_resonance = 1.0 - abs((n_samples % 104) - 52) / 52.0
        # Feature resonance: n_features relation to Fe(26) and PHI
        feature_resonance = 1.0 - abs(n_features - 26) / max(n_features, 26)
        god_code_alignment = (god_code_resonance + feature_resonance) / 2.0

        return DatasetBound(
            n_samples=n_samples,
            n_features=n_features,
            n_qubits_required=n_qubits,
            bekenstein_max_bits=bek["max_bits"],
            bekenstein_fraction_used=n_qubits / max(bek["max_bits"], 1),
            bremermann_max_ops=brem["margolus_levitin_ops_per_sec"],
            algorithm_ops_required=brem["classical_ops_required"],
            bremermann_fraction=brem["fraction_of_limit"],
            landauer_cost_per_measurement_J=land["energy_per_bit_J"],
            total_measurement_energy_J=land["total_energy_J"],
            measurements_per_joule=land["measurements_per_joule_room_temp"],
            holevo_max_classical_bits=holevo["holevo_bound_bits"],
            holevo_efficiency=holevo["accessible_fraction"],
            substrate_mass_kg=substrate_mass_kg,
            substrate_radius_m=substrate_radius_m,
            temperature_K=temperature_K,
            god_code_alignment=god_code_alignment,
            metadata={
                "bekenstein": bek,
                "bremermann": brem,
                "landauer": land,
                "holevo": holevo,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# RAYLEIGH SPECTRAL RESOLUTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class RayleighSpectralBounds:
    """
    Rayleigh-type resolution limits for quantum spectral analysis.

    Maps optical Rayleigh criterion concepts onto quantum data analysis:
    1. QFT spectral resolution ↔ classical Rayleigh criterion
    2. Phase estimation precision ↔ angular resolution
    3. Feature distinguishability ↔ Airy pattern overlap
    4. Heisenberg limit ↔ quantum super-resolution
    5. Quantum Cramér-Rao ↔ fundamental estimation bound
    """

    VERSION = "1.0.0"

    # Rayleigh constant: 1.21966989... (first zero of J₁(πx)/(πx))
    RAYLEIGH_CONSTANT = 1.21966989
    # Sparrow factor: when d²I/dθ² = 0 at midpoint
    SPARROW_FACTOR = 0.9466

    # ─── QFT Spectral Resolution ──────────────────────────────────────

    @staticmethod
    def qft_spectral_resolution(
        n_qubits: int,
        sampling_rate_hz: float = 1.0,
        n_samples: int = 0,
    ) -> Dict[str, Any]:
        """
        Resolution limits for QFT-based spectral analysis.

        Classical Rayleigh criterion maps to QFT as:
        - Frequency bin width: Δf = f_s / 2^n = f_s / N
        - Rayleigh resolution: 1.22 × Δf (analogy to 1.22 λ/D)
        - Sparrow resolution: 0.9466 × Δf

        Quantum advantage: QPE can achieve δf = f_s / (2^n × √M)
        with M measurements (Heisenberg scaling).

        Args:
            n_qubits: Number of qubits for QFT
            sampling_rate_hz: Data sampling rate
            n_samples: If > 0, overrides 2^n_qubits as effective N
        """
        N = n_samples if n_samples > 0 else 2 ** n_qubits

        # Classical DFT resolution
        bin_width = sampling_rate_hz / N
        rayleigh_res = RayleighSpectralBounds.RAYLEIGH_CONSTANT * bin_width
        sparrow_res = RayleighSpectralBounds.SPARROW_FACTOR * bin_width

        # Quantum QPE enhanced resolution (without extra measurements)
        # QPE with t precision qubits: δf = f_s / 2^t
        qpe_resolution = sampling_rate_hz / (2 ** n_qubits)

        # With √M Heisenberg scaling
        m = DEFAULT_SHOTS
        heisenberg_res = qpe_resolution / math.sqrt(m)

        # Number of resolvable frequencies
        n_resolvable_classical = int(N / 2)  # Nyquist
        n_resolvable_rayleigh = int(N / (2 * RayleighSpectralBounds.RAYLEIGH_CONSTANT))

        # GOD_CODE spectral alignment
        if sampling_rate_hz > 0:
            god_code_bin = GOD_CODE / sampling_rate_hz * N
            god_code_in_range = 0 <= god_code_bin <= N / 2
        else:
            god_code_bin = 0
            god_code_in_range = False

        return {
            "n_qubits": n_qubits,
            "N_points": N,
            "sampling_rate_hz": sampling_rate_hz,
            "fourier_bin_width_hz": bin_width,
            "rayleigh_resolution_hz": rayleigh_res,
            "sparrow_resolution_hz": sparrow_res,
            "qpe_resolution_hz": qpe_resolution,
            "heisenberg_resolution_hz": heisenberg_res,
            "quantum_advantage": rayleigh_res / max(heisenberg_res, 1e-30),
            "n_resolvable_classical": n_resolvable_classical,
            "n_resolvable_rayleigh": n_resolvable_rayleigh,
            "god_code_spectral_bin": god_code_bin,
            "god_code_in_range": god_code_in_range,
        }

    # ─── Phase Estimation Resolution ──────────────────────────────────

    @staticmethod
    def phase_estimation_resolution(
        precision_qubits: int,
        n_shots: int = DEFAULT_SHOTS,
    ) -> Dict[str, Any]:
        """
        Resolution of quantum phase estimation (QPE).

        Maps directly to Rayleigh criterion in phase space:
        - Phase bin width: Δφ = 2π / 2^t (analogous to λ/D)
        - Rayleigh resolution: 1.22 × Δφ / (2π) in [0,1) phase units
        - Heisenberg limit: Δφ / √M with M measurements

        Args:
            precision_qubits: Number of QPE precision qubits
            n_shots: Measurement repetitions for statistics
        """
        N = 2 ** precision_qubits
        phase_bin = 2 * math.pi / N  # radians
        phase_bin_unit = 1.0 / N  # in [0,1) phase units

        rayleigh_phase = RayleighSpectralBounds.RAYLEIGH_CONSTANT * phase_bin
        sparrow_phase = RayleighSpectralBounds.SPARROW_FACTOR * phase_bin

        # Heisenberg-limited (with measurements)
        heisenberg_phase = phase_bin / math.sqrt(n_shots)

        # Energy resolution: ΔE = ℏ × Δφ / Δt (via time-energy uncertainty)
        # For QPE with circuit depth ~ N gates, each gate ~ 50ns
        gate_time = 50e-9  # seconds
        total_time = N * gate_time
        energy_resolution = _HBAR * phase_bin / total_time  # Joules
        energy_resolution_eV = energy_resolution / 1.602176634e-19

        # GOD_CODE phase: how many bins to resolve GOD_CODE mod 2π
        god_code_phase = GOD_CODE_PHASE
        god_code_bins_needed = int(math.ceil(2 * math.pi / god_code_phase)) if god_code_phase > 0 else 1
        god_code_resolved = precision_qubits >= math.ceil(math.log2(god_code_bins_needed + 1))

        return {
            "precision_qubits": precision_qubits,
            "n_shots": n_shots,
            "phase_bin_rad": phase_bin,
            "phase_bin_unit": phase_bin_unit,
            "rayleigh_phase_rad": rayleigh_phase,
            "sparrow_phase_rad": sparrow_phase,
            "heisenberg_phase_rad": heisenberg_phase,
            "energy_resolution_eV": energy_resolution_eV,
            "total_evolution_time_s": total_time,
            "n_resolvable_phases": N,
            "god_code_phase_rad": god_code_phase,
            "god_code_resolved": god_code_resolved,
        }

    # ─── Hilbert Space Feature Resolution ─────────────────────────────

    @staticmethod
    def hilbert_feature_resolution(
        n_qubits: int,
        n_features: int,
    ) -> Dict[str, Any]:
        """
        Rayleigh-type resolution for distinguishing features in Hilbert space.

        When features are encoded as quantum states, their distinguishability
        depends on the inner product (overlap) — analogous to Airy pattern
        overlap in optics.

        Rayleigh criterion in Hilbert space:
        - Two states |ψ₁⟩, |ψ₂⟩ are "Rayleigh-resolved" when:
          |⟨ψ₁|ψ₂⟩|² ≤ 1 - 1/N (where N = 2^n_qubits)
        - Super-resolution: SWAP test can detect overlap ε ~ 1/√M

        The "aperture" is the Hilbert space dimension 2^n.
        The "wavelength" is the feature encoding angular separation.

        Args:
            n_qubits: Qubits in the register
            n_features: Number of classical features encoded
        """
        dim = 2 ** n_qubits

        # Angular separation between orthogonal basis states
        angular_separation = math.pi / (2 * dim)  # radians in Fubini-Study metric

        # Rayleigh criterion: minimum resolvable "angle" in Hilbert space
        rayleigh_angle = RayleighSpectralBounds.RAYLEIGH_CONSTANT * angular_separation

        # Feature packing: how many distinguishable features can fit
        max_orthogonal = dim  # Maximum orthogonal states
        # Rayleigh-limited: accounting for overlap tolerance
        rayleigh_limited = int(dim / RayleighSpectralBounds.RAYLEIGH_CONSTANT)

        # Feature utilization
        utilization = n_features / max_orthogonal
        rayleigh_utilization = n_features / max(rayleigh_limited, 1)

        # SWAP test discrimination
        # Minimum detectable overlap with M shots: ε_min ~ 1/√M
        swap_test_min_overlap = 1.0 / math.sqrt(DEFAULT_SHOTS)
        swap_test_resolvable_states = int(1.0 / max(swap_test_min_overlap, 1e-10))

        # Quantum tomography information
        # Full state tomography requires O(4^n) measurements
        tomography_measurements = 4 ** n_qubits
        # Shadow tomography: O(n² / ε²) for n observables
        shadow_measurements = n_features ** 2

        return {
            "n_qubits": n_qubits,
            "hilbert_dimension": dim,
            "n_features": n_features,
            "angular_separation_rad": angular_separation,
            "rayleigh_angle_rad": rayleigh_angle,
            "max_orthogonal_features": max_orthogonal,
            "rayleigh_limited_features": rayleigh_limited,
            "feature_utilization": utilization,
            "rayleigh_utilization": rayleigh_utilization,
            "swap_test_min_overlap": swap_test_min_overlap,
            "swap_test_resolvable_states": swap_test_resolvable_states,
            "tomography_measurements_full": tomography_measurements,
            "shadow_tomography_measurements": shadow_measurements,
            "regime": "UNDER_RESOLVED" if rayleigh_utilization > 1.0 else "WELL_RESOLVED",
        }

    # ─── Quantum Super-Resolution ─────────────────────────────────────

    @staticmethod
    def quantum_super_resolution(
        n_qubits: int,
        classical_resolution: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Quantum super-resolution beyond the classical Rayleigh limit.

        In quantum metrology, entangled states (NOON, GHZ) can achieve
        resolution scaling as 1/N rather than 1/√N:

        Classical: δθ = λ / (D√N)     (Rayleigh × shot noise)
        Quantum:   δθ = λ / (D×N)     (Heisenberg limit)

        Quantum illumination and NOON states provide quadratic improvement
        in angular/spectral resolution.

        Args:
            n_qubits: Number of entangled sensor qubits
            classical_resolution: Classical Rayleigh resolution (normalized)
        """
        # Standard quantum limit (product states)
        sql_resolution = classical_resolution / math.sqrt(n_qubits)

        # Heisenberg limit (NOON/GHZ states)
        heisenberg_resolution = classical_resolution / n_qubits

        # Super-resolution factor
        super_factor = classical_resolution / heisenberg_resolution

        # NOON state effective wavelength
        # NOON: |N,0⟩ + |0,N⟩ → effective wavelength λ/N
        effective_wavelength_factor = 1.0 / n_qubits

        # Quantum illumination advantage (for detection in noise)
        # Advantage scales as ~6dB per entangled pair
        qi_advantage_db = 6.0 * (n_qubits // 2)

        # De Broglie wavelength of N-photon entangled state
        # λ_dB = λ / N
        god_code_super_resolution_nm = GOD_CODE / n_qubits

        # Resolution in angular measure
        if classical_resolution > 0:
            rayleigh_arcsec = classical_resolution * 206265  # convert rad to arcsec
            heisenberg_arcsec = heisenberg_resolution * 206265
        else:
            rayleigh_arcsec = heisenberg_arcsec = 0

        return {
            "n_qubits": n_qubits,
            "classical_rayleigh": classical_resolution,
            "sql_resolution": sql_resolution,
            "heisenberg_resolution": heisenberg_resolution,
            "super_resolution_factor": super_factor,
            "effective_wavelength_factor": effective_wavelength_factor,
            "god_code_super_nm": god_code_super_resolution_nm,
            "qi_advantage_db": qi_advantage_db,
            "rayleigh_arcsec": rayleigh_arcsec,
            "heisenberg_arcsec": heisenberg_arcsec,
        }

    # ─── Full Spectral Resolution Analysis ────────────────────────────

    def analyze_spectral_bounds(
        self,
        n_qubits: int,
        n_samples: int,
        sampling_rate_hz: float = 1000.0,
        entangled: bool = True,
    ) -> SpectralResolutionBound:
        """
        Complete Rayleigh resolution analysis for quantum spectral processing.

        Args:
            n_qubits: Qubits for the QFT/QPE register
            n_samples: Number of input data samples
            sampling_rate_hz: Sampling rate of the data
            entangled: Whether quantum super-resolution is active
        """
        # QFT resolution
        qft = self.qft_spectral_resolution(n_qubits, sampling_rate_hz, n_samples)

        # Phase estimation
        qpe = self.phase_estimation_resolution(n_qubits)

        # Fisher information
        fisher = ComputroniumDataBounds.quantum_fisher_information(n_qubits, DEFAULT_SHOTS, entangled)

        # Super-resolution
        classical_res = qft["rayleigh_resolution_hz"]
        sr = self.quantum_super_resolution(n_qubits, classical_res)

        # Hilbert feature resolution
        hfr = self.hilbert_feature_resolution(n_qubits, n_samples)

        # GOD_CODE alignment
        god_freq_hz = _C / _GOD_CODE_WAVELENGTH_M
        if sampling_rate_hz > 0:
            god_code_spectral = abs(math.sin(GOD_CODE * math.pi / sampling_rate_hz))
        else:
            god_code_spectral = 0

        return SpectralResolutionBound(
            n_qubits=n_qubits,
            n_samples=n_samples,
            fourier_bin_width_hz=qft["fourier_bin_width_hz"],
            rayleigh_resolution_hz=qft["rayleigh_resolution_hz"],
            sparrow_resolution_hz=qft["sparrow_resolution_hz"],
            standard_quantum_limit=fisher["standard_quantum_limit"],
            heisenberg_limit=fisher["heisenberg_limit"],
            quantum_advantage_factor=fisher["quantum_advantage_factor"],
            cramer_rao_bound=fisher["cramer_rao_bound_rad"],
            fisher_information=fisher["fisher_information"],
            hilbert_angular_resolution=hfr["rayleigh_angle_rad"],
            feature_distinguishability=1.0 - hfr["rayleigh_utilization"],
            super_resolution_possible=entangled and n_qubits > 1,
            super_resolution_factor=sr["super_resolution_factor"],
            god_code_spectral_alignment=god_code_spectral,
            metadata={
                "qft": qft,
                "qpe": qpe,
                "fisher": fisher,
                "super_resolution": sr,
                "hilbert": hfr,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM INFORMATION BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumInformationBridge:
    """
    Bridges computronium limits with Rayleigh resolution for quantum data analysis.

    Unifies:
    - Information capacity (Bekenstein, Holevo) with resolution (Rayleigh, Heisenberg)
    - Processing limits (Bremermann, ML) with spectral analysis bounds
    - Physical energy costs (Landauer) with measurement resources
    """

    VERSION = "1.0.0"

    def __init__(self):
        self.data_bounds = ComputroniumDataBounds()
        self.spectral_bounds = RayleighSpectralBounds()

    def full_analysis(
        self,
        n_samples: int,
        n_features: int,
        algorithm: str = "O(n log n)",
        sampling_rate_hz: float = 1000.0,
        substrate_mass_kg: float = 1.0,
        temperature_K: float = _ROOM_TEMP,
    ) -> Dict[str, Any]:
        """
        Complete computronium + Rayleigh analysis of a quantum data analysis task.

        Returns unified report with all physical bounds, resolution limits,
        and GOD_CODE sacred alignment metrics.
        """
        n_qubits = max(1, min(int(math.ceil(math.log2(max(n_samples, 2)))),
                              MAX_QUBITS_STATEVECTOR))

        # Dataset bounds
        dataset = self.data_bounds.analyze_dataset_bounds(
            n_samples, n_features, algorithm,
            substrate_mass_kg, 0.01, temperature_K,
        )

        # Spectral bounds
        spectral = self.spectral_bounds.analyze_spectral_bounds(
            n_qubits, n_samples, sampling_rate_hz, entangled=True,
        )

        # Channel capacity
        channel = ComputroniumDataBounds.lloyd_channel_capacity(n_qubits)

        # Information-resolution bridge
        # Key insight: dataset capacity (bits) × spectral resolution (Hz)
        # = information rate — bounded by Bremermann
        info_rate = dataset.bekenstein_max_bits * spectral.rayleigh_resolution_hz
        bremermann_fraction = info_rate / max(dataset.bremermann_max_ops, 1)

        # Landauer-Heisenberg product: energy × precision
        # Minimum energy per resolved spectral feature
        energy_per_feature = dataset.landauer_cost_per_measurement_J / max(
            spectral.fisher_information, 1
        )

        # PHI-harmonic quality score (0-1)
        phi_quality = (
            0.3 * dataset.god_code_alignment +
            0.3 * spectral.god_code_spectral_alignment +
            0.2 * min(1.0, channel["quantum_capacity_per_use"]) +
            0.2 * min(1.0, spectral.quantum_advantage_factor / n_qubits)
        )

        return {
            "version": self.VERSION,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_qubits": n_qubits,
            "algorithm": algorithm,
            "dataset_bounds": {
                "bekenstein_max_bits": dataset.bekenstein_max_bits,
                "bekenstein_fraction": dataset.bekenstein_fraction_used,
                "bremermann_ceiling_ops": dataset.bremermann_max_ops,
                "bremermann_fraction": dataset.bremermann_fraction,
                "landauer_cost_per_measurement_J": dataset.landauer_cost_per_measurement_J,
                "holevo_bound_bits": dataset.holevo_max_classical_bits,
            },
            "spectral_bounds": {
                "rayleigh_resolution_hz": spectral.rayleigh_resolution_hz,
                "heisenberg_limit": spectral.heisenberg_limit,
                "quantum_advantage": spectral.quantum_advantage_factor,
                "super_resolution_factor": spectral.super_resolution_factor,
                "fisher_information": spectral.fisher_information,
                "cramer_rao_bound": spectral.cramer_rao_bound,
            },
            "channel": {
                "quantum_capacity_per_use": channel["quantum_capacity_per_use"],
                "total_capacity_qubits_per_sec": channel["total_capacity_qubits_per_sec"],
                "coherent_depth": channel["coherent_depth"],
            },
            "bridge": {
                "information_rate_bound": info_rate,
                "bremermann_fraction_of_rate": bremermann_fraction,
                "energy_per_resolved_feature_J": energy_per_feature,
                "phi_harmonic_quality": phi_quality,
            },
            "god_code_alignment": {
                "dataset": dataset.god_code_alignment,
                "spectral": spectral.god_code_spectral_alignment,
                "god_code_wavelength_nm": GOD_CODE,
            },
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "subsystems": {
                "data_bounds": ComputroniumDataBounds.VERSION,
                "spectral_bounds": RayleighSpectralBounds.VERSION,
            },
            "god_code": GOD_CODE,
            "phi": PHI,
            "constants": "CODATA_2022",
        }


# ─── Module-level singletons ────────────────────────────────────────────────
computronium_data_bounds = ComputroniumDataBounds()
rayleigh_spectral_bounds = RayleighSpectralBounds()
quantum_information_bridge = QuantumInformationBridge()
