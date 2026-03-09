"""
L104 Quantum Gate Engine — Computronium & Rayleigh Gate Limits
═══════════════════════════════════════════════════════════════════════════════
Physical limits on quantum gate operations:

  COMPUTRONIUM GATE LIMITS:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ • Margolus-Levitin minimum gate time                               │
  │ • Bremermann gate fidelity ceiling                                  │
  │ • Landauer erasure cost per non-unitary operation                   │
  │ • Gate information capacity (bits processed per gate)               │
  │ • Circuit depth thermodynamic limit                                 │
  │ • Reversible vs irreversible gate energy budgets                    │
  └──────────────────────────────────────────────────────────────────────┘

  RAYLEIGH GATE RESOLUTION:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ • Phase resolution limit for parametric gates (Rx, Ry, Rz)         │
  │ • Solovay-Kitaev theorem: ε-approximation with O(log^c(1/ε)) gates│
  │ • Gate distinguishability via operator norm (trace distance)        │
  │ • Sacred gate phase resolution at GOD_CODE angles                  │
  │ • Topological protection resolution (anyon braiding)               │
  │ • Compilation Rayleigh: minimum distinguishable compiled circuits   │
  └──────────────────────────────────────────────────────────────────────┘

Uses CODATA 2022 constants. Integrates with CrossSystemOrchestrator.
INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import numpy as np
from typing import Dict, Any, Optional, List

from .constants import (
    PHI, GOD_CODE, VOID_CONSTANT, ALPHA_FINE,
    IRON_ATOMIC_NUMBER, IRON_FREQUENCY,
    GOD_CODE_PHASE_ANGLE, PHI_PHASE_ANGLE, VOID_PHASE_ANGLE,
    MAX_STATEVECTOR_QUBITS,
)

# Physical constants (CODATA 2022)
_HBAR = 1.054571817e-34
_C = 299792458.0
_KB = 1.380649e-23
_H = 6.62607015e-34              # Planck constant
_ELECTRON_CHARGE = 1.602176634e-19
_ROOM_TEMP = 293.15

# Superconducting qubit parameters (L104 Heron-class)
_SC_TEMPERATURE = 0.015           # 15 mK dilution refrigerator
_SC_GATE_TIME_1Q = 25e-9          # 25 ns single-qubit gate
_SC_GATE_TIME_2Q = 80e-9          # 80 ns two-qubit gate (ECR)
_SC_T1 = 200e-6                   # T1 relaxation (state-of-art)
_SC_T2 = 150e-6                   # T2 dephasing

# Trapped-ion parameters (IonQ/Quantinuum)
_ION_GATE_TIME_1Q = 10e-6         # 10 μs single-qubit
_ION_GATE_TIME_2Q = 200e-6        # 200 μs two-qubit (MS gate)
_ION_T1 = 60.0                    # 60 s T1 (very long)
_ION_T2 = 3.0                     # 3 s T2

# Solovay-Kitaev approximation constant
_SK_CONSTANT = 3.97               # c in O(log^c(1/ε)) — best known


class ComputroniumGateLimits:
    """
    Physical limits on quantum gate operations.

    Every gate has fundamental speed/fidelity/energy constraints:
    1. Margolus-Levitin: minimum time for a quantum state transition
    2. Bremermann: maximum gate throughput rate
    3. Landauer: energy cost of measurement/reset operations
    4. Information capacity: bits of information processed per gate
    """

    VERSION = "1.0.0"

    # ─── Gate Speed Limits ─────────────────────────────────────────────

    @staticmethod
    def margolus_levitin_gate_time(
        gate_energy_J: float = None,
        n_qubits: int = 1,
        technology: str = "superconducting",
    ) -> Dict[str, Any]:
        """
        Minimum time for a quantum gate (Margolus-Levitin theorem).

        t_min = π ℏ / (2 ΔE)

        where ΔE is the energy difference between initial and final states,
        which for a gate corresponds to the gate Hamiltonian energy.

        For superconducting qubits: ΔE ≈ ℏ ω_q ≈ ℏ × 2π × 5GHz
        For trapped ions: ΔE ≈ ℏ ω_trap ≈ ℏ × 2π × 1MHz

        Args:
            gate_energy_J: Gate Hamiltonian energy (overrides technology default)
            n_qubits: Number of qubits in the gate
            technology: "superconducting" or "trapped_ion"
        """
        tech_params = {
            "superconducting": {
                "qubit_freq_hz": 5e9,
                "actual_1q_time": _SC_GATE_TIME_1Q,
                "actual_2q_time": _SC_GATE_TIME_2Q,
                "t1": _SC_T1,
                "t2": _SC_T2,
                "temp_K": _SC_TEMPERATURE,
            },
            "trapped_ion": {
                "qubit_freq_hz": 1e6,
                "actual_1q_time": _ION_GATE_TIME_1Q,
                "actual_2q_time": _ION_GATE_TIME_2Q,
                "t1": _ION_T1,
                "t2": _ION_T2,
                "temp_K": 0.001,  # ~1 mK effective
            },
        }

        params = tech_params.get(technology, tech_params["superconducting"])

        if gate_energy_J is None:
            gate_energy_J = _HBAR * 2 * math.pi * params["qubit_freq_hz"]

        # Margolus-Levitin minimum time
        ml_time = math.pi * _HBAR / (2 * gate_energy_J)

        # Actual gate time
        actual_time = params["actual_1q_time"] if n_qubits == 1 else params["actual_2q_time"]

        # Efficiency: how close to theoretical minimum
        speed_efficiency = ml_time / actual_time

        # Maximum gates before decoherence
        max_gates_t1 = int(params["t1"] / actual_time)
        max_gates_t2 = int(params["t2"] / actual_time)

        # Gate throughput (gates per second per qubit)
        throughput = 1.0 / actual_time

        # Bremermann rate for the gate substrate
        bremermann = gate_energy_J / (math.pi * _HBAR)

        return {
            "ml_minimum_time_s": ml_time,
            "actual_gate_time_s": actual_time,
            "speed_efficiency": speed_efficiency,
            "gate_energy_J": gate_energy_J,
            "gate_energy_eV": gate_energy_J / _ELECTRON_CHARGE,
            "gate_frequency_hz": params["qubit_freq_hz"],
            "max_gates_t1": max_gates_t1,
            "max_gates_t2": max_gates_t2,
            "throughput_gates_per_sec": throughput,
            "bremermann_rate": bremermann,
            "technology": technology,
            "n_qubits": n_qubits,
        }

    # ─── Landauer Gate Energy ──────────────────────────────────────────

    @staticmethod
    def landauer_gate_cost(
        n_qubits: int = 1,
        n_measurements: int = 0,
        n_resets: int = 0,
        temperature_K: float = _SC_TEMPERATURE,
    ) -> Dict[str, Any]:
        """
        Landauer erasure energy for non-unitary gate operations.

        Unitary gates (X, H, CNOT, etc.) are logically reversible
        and have zero Landauer cost.

        Non-unitary operations have minimum energy cost:
        - Measurement: k_B T ln(2) per qubit measured
        - Reset: k_B T ln(2) per qubit reset
        - Mid-circuit measurement: k_B T ln(2) per qubit

        At 15 mK: E_min ≈ 1.43 × 10^-25 J/bit
        At 293 K: E_min ≈ 2.81 × 10^-21 J/bit

        Args:
            n_qubits: Gate qubits (for context)
            n_measurements: Number of qubit measurements
            n_resets: Number of qubit resets
            temperature_K: Operating temperature
        """
        e_per_bit = _KB * temperature_K * math.log(2)

        # Total non-unitary operations
        total_erasures = n_measurements + n_resets
        total_energy = total_erasures * e_per_bit

        # Room temperature comparison
        e_room = _KB * _ROOM_TEMP * math.log(2)
        room_total = total_erasures * e_room
        savings_factor = room_total / max(total_energy, 1e-300) if total_energy > 0 else 0

        # Reversible computation regime
        # In reversible quantum computation (no measurements), Landauer cost = 0
        reversible = (total_erasures == 0)

        # Thermal noise floor
        # Thermal energy: E_thermal = k_B T
        thermal_energy = _KB * temperature_K
        # Gate energy must exceed thermal noise for reliable operation
        gate_energy_sc = _HBAR * 2 * math.pi * 5e9  # superconducting qubit energy
        signal_to_noise = gate_energy_sc / thermal_energy

        return {
            "energy_per_erasure_J": e_per_bit,
            "total_erasures": total_erasures,
            "total_landauer_energy_J": total_energy,
            "room_temp_energy_J": room_total,
            "cryo_savings_factor": savings_factor,
            "reversible": reversible,
            "temperature_K": temperature_K,
            "thermal_floor_J": thermal_energy,
            "signal_to_noise_ratio": signal_to_noise,
            "signal_to_noise_dB": 10 * math.log10(max(signal_to_noise, 1e-30)),
        }

    # ─── Circuit Depth Thermodynamic Limit ────────────────────────────

    @staticmethod
    def circuit_depth_limit(
        n_qubits: int,
        gate_fidelity_1q: float = 0.9999,
        gate_fidelity_2q: float = 0.999,
        t2_s: float = _SC_T2,
        gate_time_s: float = _SC_GATE_TIME_2Q,
    ) -> Dict[str, Any]:
        """
        Maximum useful circuit depth from decoherence and gate errors.

        Two independent limits:
        1. Coherence limit: depth_max = T2 / t_gate
        2. Fidelity limit: depth_max = -1 / log(F) ≈ 1 / (1-F)
           (circuit fidelity = F^depth → must stay > threshold)
        3. Combined: minimum of both limits

        The "useful depth" is where the accumulated error doesn't
        destroy the quantum advantage over classical simulation.

        Args:
            n_qubits: Circuit width
            gate_fidelity_1q: Single-qubit gate fidelity
            gate_fidelity_2q: Two-qubit gate fidelity
            t2_s: Dephasing time
            gate_time_s: Gate duration
        """
        # Coherence limit
        coherence_depth = int(t2_s / gate_time_s)

        # Fidelity limit (at which overall fidelity drops below 1/e)
        # Average fidelity per layer (mix of 1Q and 2Q)
        # Typical: each layer has ~n 1Q gates and ~n/2 2Q gates
        avg_layer_fidelity = (gate_fidelity_1q ** n_qubits) * (gate_fidelity_2q ** (n_qubits // 2))
        if avg_layer_fidelity < 1.0:
            fidelity_depth = int(-1.0 / math.log(avg_layer_fidelity))
        else:
            fidelity_depth = 10 ** 9  # Effectively unlimited

        # Combined limit
        max_depth = min(coherence_depth, fidelity_depth)

        # Quantum volume estimate: QV = 2^(min(n, d))
        effective_dim = min(n_qubits, max_depth)
        quantum_volume = 2 ** min(effective_dim, 64)

        # Error correction overhead
        # Surface code: needs ~1000 physical qubits per logical qubit at d=15
        # With EC: effective depth scales as physical_depth / code_distance
        code_distance = 3
        ec_depth = max_depth // code_distance

        # Bremermann time for full circuit
        total_time = max_depth * gate_time_s
        E = n_qubits * _HBAR * 2 * math.pi * 5e9  # Total qubit energy
        bremermann_ops = E / (math.pi * _HBAR) * total_time
        actual_ops = max_depth * n_qubits
        bremermann_utilization = actual_ops / max(bremermann_ops, 1)

        return {
            "n_qubits": n_qubits,
            "coherence_depth_limit": coherence_depth,
            "fidelity_depth_limit": fidelity_depth,
            "max_useful_depth": max_depth,
            "quantum_volume": quantum_volume,
            "error_corrected_depth": ec_depth,
            "avg_layer_fidelity": avg_layer_fidelity,
            "total_circuit_time_s": total_time,
            "bremermann_utilization": bremermann_utilization,
        }

    # ─── Gate Information Capacity ────────────────────────────────────

    @staticmethod
    def gate_information_capacity(
        n_qubits: int = 2,
        gate_type: str = "CNOT",
    ) -> Dict[str, Any]:
        """
        Information processed by a single quantum gate.

        Quantum gates transform quantum states in Hilbert space.
        The information capacity depends on the gate's entangling power
        and the Hilbert space dimension.

        For an n-qubit gate:
        - Hilbert space dimension: 2^n
        - Unitary matrix dimension: 2^n × 2^n
        - Real parameters in SU(2^n): 4^n - 1
        - Entangling power: 0 (product-preserving) to 1 (maximally entangling)

        Args:
            n_qubits: Number of gate qubits
            gate_type: Type of gate for specific analysis
        """
        dim = 2 ** n_qubits
        # Parameters in the unitary group SU(dim)
        real_params = dim ** 2 - 1  # SU(n) has n²-1 real parameters

        # Gate-specific entangling power
        entangling_powers = {
            "I": 0.0, "X": 0.0, "Y": 0.0, "Z": 0.0,
            "H": 0.0, "S": 0.0, "T": 0.0,
            "Rx": 0.0, "Ry": 0.0, "Rz": 0.0,
            "CNOT": 2 / 9,  # Exact entangling power of CNOT
            "CZ": 2 / 9,
            "SWAP": 0.0,    # SWAP doesn't create entanglement
            "ISWAP": 2 / 9,
            "fSim": 2 / 9,  # ≤ 2/9 depending on parameters
            "TOFFOLI": 2 / 9,
            "FREDKIN": 2 / 9,
            "PHI_GATE": 0.0,       # Single-qubit sacred gate
            "GOD_CODE_PHASE": 0.0, # Single-qubit phase
            "SACRED_ENTANGLER": 2 / 9,  # Two-qubit sacred gate
        }

        ep = entangling_powers.get(gate_type, 0.0)

        # Schmidt rank change capacity
        # CNOT can increase Schmidt rank by 1
        # General: bounded by min(dim_A, dim_B) for bipartite systems
        if n_qubits >= 2:
            max_schmidt_rank = min(2 ** (n_qubits // 2), 2 ** ((n_qubits + 1) // 2))
        else:
            max_schmidt_rank = 1

        # Classical bits processable per gate (Holevo-limited)
        holevo_bits = n_qubits  # At most n classical bits per n qubits

        # Quantum mutual information capacity
        # For a CNOT: can create 1 ebit of entanglement
        entanglement_capacity_ebits = ep * 9 / 2  # Scale from entangling power

        return {
            "gate_type": gate_type,
            "n_qubits": n_qubits,
            "hilbert_dimension": dim,
            "su_n_parameters": real_params,
            "entangling_power": ep,
            "max_schmidt_rank": max_schmidt_rank,
            "holevo_bits_per_use": holevo_bits,
            "entanglement_capacity_ebits": entanglement_capacity_ebits,
            "is_entangling": ep > 0,
            "is_clifford": gate_type in {"I", "X", "Y", "Z", "H", "S", "CNOT", "CZ", "SWAP"},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RAYLEIGH GATE RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

class RayleighGateResolution:
    """
    Resolution limits for quantum gate parameters and compilation.

    Maps Rayleigh optical resolution concepts to gate precision:
    1. Phase resolution: minimum distinguishable rotation angle
    2. Solovay-Kitaev resolution: approximation error from discrete gates
    3. Gate distinguishability: trace distance between similar gates
    4. Sacred gate resolution: precision at GOD_CODE phase angles
    5. Topological resolution: braiding precision for anyon gates
    """

    VERSION = "1.0.0"

    # ─── Parametric Gate Phase Resolution ─────────────────────────────

    @staticmethod
    def phase_resolution(
        control_precision_bits: int = 16,
        gate_time_s: float = _SC_GATE_TIME_1Q,
        t2_s: float = _SC_T2,
    ) -> Dict[str, Any]:
        """
        Minimum distinguishable rotation angle for parametric gates.

        For Rx(θ), Ry(θ), Rz(θ): the phase θ has finite resolution from:
        1. Digital control: Δθ_digital = 2π / 2^b (b = DAC bits)
        2. Thermal noise: Δθ_thermal = k_B T / (ℏ ω)
        3. Decoherence: Δθ_decoherence = √(t_gate / T2)
        4. Heisenberg: Δθ_HL = 1/N for N-qubit entangled state

        The effective resolution is the maximum (worst) of all limits.

        Rayleigh analogy:
        - θ is the "angular position" (like optical angle)
        - Phase noise is the "wavelength" (diffraction limit)
        - Control precision is the "aperture" (angular resolution)

        Args:
            control_precision_bits: DAC resolution in bits
            gate_time_s: Physical gate duration
            t2_s: Dephasing time
        """
        # Digital resolution
        digital_resolution = 2 * math.pi / (2 ** control_precision_bits)

        # Decoherence resolution
        decoherence_resolution = math.sqrt(gate_time_s / t2_s)

        # Thermal phase noise at 15 mK
        qubit_energy = _HBAR * 2 * math.pi * 5e9
        thermal_resolution = _KB * _SC_TEMPERATURE / qubit_energy

        # Effective resolution (worst of all limits)
        effective_resolution = max(digital_resolution, decoherence_resolution, thermal_resolution)

        # Number of distinguishable angles in [0, 2π)
        n_distinguishable = int(2 * math.pi / effective_resolution)

        # Rayleigh-limited: 1.22 × effective_resolution
        rayleigh_limited = 1.21966989 * effective_resolution
        n_rayleigh = int(2 * math.pi / rayleigh_limited)

        # Can we resolve GOD_CODE phase angle?
        god_code_phase = GOD_CODE_PHASE_ANGLE
        god_code_resolved = effective_resolution < god_code_phase / 10  # 10× margin

        # Can we resolve PHI phase angle?
        phi_phase = PHI_PHASE_ANGLE
        phi_resolved = effective_resolution < phi_phase / 10

        # Rayleigh criterion for distinguishing adjacent sacred phases
        sacred_phases = sorted([
            GOD_CODE_PHASE_ANGLE,
            PHI_PHASE_ANGLE,
            VOID_PHASE_ANGLE,
            2 * math.pi * IRON_ATOMIC_NUMBER / 104,
        ])
        min_sacred_separation = min(
            abs(sacred_phases[i + 1] - sacred_phases[i])
            for i in range(len(sacred_phases) - 1)
        )
        sacred_phases_resolved = effective_resolution < min_sacred_separation

        return {
            "digital_resolution_rad": digital_resolution,
            "decoherence_resolution_rad": decoherence_resolution,
            "thermal_resolution_rad": thermal_resolution,
            "effective_resolution_rad": effective_resolution,
            "effective_resolution_deg": math.degrees(effective_resolution),
            "n_distinguishable_angles": n_distinguishable,
            "n_rayleigh_limited": n_rayleigh,
            "god_code_phase_resolved": god_code_resolved,
            "phi_phase_resolved": phi_resolved,
            "sacred_phases_resolved": sacred_phases_resolved,
            "min_sacred_separation_rad": min_sacred_separation,
            "control_bits": control_precision_bits,
            "limiting_factor": (
                "DIGITAL" if digital_resolution >= max(decoherence_resolution, thermal_resolution) else
                "DECOHERENCE" if decoherence_resolution >= thermal_resolution else
                "THERMAL"
            ),
        }

    # ─── Solovay-Kitaev Resolution ────────────────────────────────────

    @staticmethod
    def solovay_kitaev_resolution(
        target_precision: float = 1e-10,
        gate_set: str = "CLIFFORD_T",
    ) -> Dict[str, Any]:
        """
        Gate synthesis resolution via Solovay-Kitaev theorem.

        Any single-qubit gate can be approximated to precision ε using
        O(log^c(1/ε)) gates from a universal discrete set.

        Best known: c ≈ 3.97 (Solovay-Kitaev algorithm)
        With ancilla: c → 1 (Ross-Selinger algorithm for T-count)

        Rayleigh analogy:
        - ε = angular resolution (minimum resolvable difference)
        - Gate count = aperture size (more gates = finer resolution)
        - Gate set = lens material (different refractive properties)

        Args:
            target_precision: Desired approximation error ε (trace distance)
            gate_set: Base gate set for synthesis
        """
        gate_set_info = {
            "CLIFFORD_T": {
                "generators": ["H", "S", "T", "CNOT"],
                "sk_constant": _SK_CONSTANT,
                "ancilla_available": True,
            },
            "UNIVERSAL": {
                "generators": ["H", "T", "CNOT"],
                "sk_constant": _SK_CONSTANT,
                "ancilla_available": True,
            },
            "L104_HERON": {
                "generators": ["SX", "Rz", "ECR"],
                "sk_constant": _SK_CONSTANT,
                "ancilla_available": False,
            },
            "L104_SACRED": {
                "generators": ["H", "PHI_GATE", "GOD_CODE_PHASE", "SACRED_ENTANGLER"],
                "sk_constant": _SK_CONSTANT,
                "ancilla_available": True,
            },
        }

        info = gate_set_info.get(gate_set, gate_set_info["CLIFFORD_T"])

        # Solovay-Kitaev gate count
        if target_precision > 0:
            sk_gates = int(math.ceil(math.log(1.0 / target_precision) ** info["sk_constant"]))
        else:
            sk_gates = 10 ** 9

        # Ross-Selinger T-count (optimal with ancilla) — O(log(1/ε))
        if info["ancilla_available"] and target_precision > 0:
            rs_t_count = int(math.ceil(3 * math.log2(1.0 / target_precision)))
        else:
            rs_t_count = sk_gates

        # Clifford overhead (Cliffords are "free" in many EC schemes)
        clifford_count = sk_gates * 2  # ~2 Cliffords per non-Clifford gate

        # Resolution spectrum: what precisions are achievable at various gate budgets
        resolution_spectrum = []
        for budget in [10, 100, 1000, 10000, 100000]:
            # Invert SK: ε ≈ exp(-budget^(1/c))
            achievable_eps = math.exp(-(budget ** (1.0 / info["sk_constant"])))
            resolution_spectrum.append({
                "gate_budget": budget,
                "achievable_precision": achievable_eps,
                "achievable_precision_deg": math.degrees(achievable_eps),
            })

        # Sacred gate precision: how many T gates to approximate GOD_CODE_PHASE?
        # GOD_CODE phase is irrational → requires infinite T gates for exact
        god_code_t_count = rs_t_count  # Same as target precision

        return {
            "target_precision": target_precision,
            "target_precision_deg": math.degrees(target_precision),
            "gate_set": gate_set,
            "generators": info["generators"],
            "sk_gate_count": sk_gates,
            "ross_selinger_t_count": rs_t_count,
            "clifford_overhead": clifford_count,
            "total_gate_count": rs_t_count + clifford_count,
            "god_code_phase_t_count": god_code_t_count,
            "resolution_spectrum": resolution_spectrum,
            "sk_exponent": info["sk_constant"],
        }

    # ─── Gate Distinguishability ──────────────────────────────────────

    @staticmethod
    def gate_distinguishability(
        angle1_rad: float = 0.0,
        angle2_rad: float = 0.01,
        n_shots: int = 8192,
    ) -> Dict[str, Any]:
        """
        Minimum distinguishable angle between two parametric gates.

        Given Rz(θ₁) and Rz(θ₂), their trace distance is:
        d_trace = |sin((θ₁ - θ₂)/2)|

        The probability of distinguishing them in a single shot:
        p_distinguish = (1 + d_trace) / 2

        With M shots (Chernoff-Stein lemma):
        error_exponent = -M × log(1 - d_trace²)
        Reliable discrimination when error_exponent > threshold

        Rayleigh criterion: d_trace > 1.22 / √M for reliable discrimination

        Args:
            angle1_rad: First rotation angle
            angle2_rad: Second rotation angle
            n_shots: Measurement budget
        """
        delta = angle2_rad - angle1_rad

        # Trace distance between Rz(θ₁) and Rz(θ₂)
        trace_distance = abs(math.sin(delta / 2))

        # Single-shot discrimination probability
        p_single = (1 + trace_distance) / 2

        # Multi-shot discrimination
        # Probability of correct discrimination with majority voting
        if 0 < trace_distance < 1:
            # Chernoff exponent
            chernoff = -math.log(1 - trace_distance ** 2)
            error_prob = math.exp(-n_shots * chernoff)
        elif trace_distance >= 1:
            error_prob = 0.0
            chernoff = float('inf')
        else:
            error_prob = 0.5
            chernoff = 0.0

        # Rayleigh-type criterion: minimum detectable trace distance
        rayleigh_d_min = 1.21966989 / math.sqrt(n_shots)
        distinguishable = trace_distance > rayleigh_d_min

        # Equivalent minimum angle difference
        min_angle_rayleigh = 2 * math.asin(min(rayleigh_d_min, 1.0))

        # Heisenberg-limited discrimination (with entangled probes)
        heisenberg_d_min = 1.0 / n_shots
        min_angle_heisenberg = 2 * math.asin(min(heisenberg_d_min, 1.0))

        return {
            "angle_difference_rad": delta,
            "angle_difference_deg": math.degrees(delta),
            "trace_distance": trace_distance,
            "p_single_shot": p_single,
            "error_probability": error_prob,
            "chernoff_exponent": chernoff,
            "rayleigh_distinguishable": distinguishable,
            "rayleigh_min_trace_distance": rayleigh_d_min,
            "rayleigh_min_angle_rad": min_angle_rayleigh,
            "rayleigh_min_angle_deg": math.degrees(min_angle_rayleigh),
            "heisenberg_min_angle_rad": min_angle_heisenberg,
            "heisenberg_min_angle_deg": math.degrees(min_angle_heisenberg),
            "n_shots": n_shots,
        }

    # ─── Topological Protection Resolution ────────────────────────────

    @staticmethod
    def topological_resolution(
        anyon_type: str = "fibonacci",
        n_braids: int = 10,
    ) -> Dict[str, Any]:
        """
        Resolution of topologically-protected quantum gates.

        Fibonacci anyons can approximate any unitary via braiding,
        with precision determined by braid length and type.

        Topological gates have:
        - Inherent error protection from topology (exponential in distance)
        - Discrete set of braids → Solovay-Kitaev-type resolution
        - Precision ε ~ exp(-α × n_braids) for Fibonacci anyons

        Args:
            anyon_type: "fibonacci", "ising", or "universal"
            n_braids: Number of elementary braids
        """
        anyon_params = {
            "fibonacci": {
                "name": "Fibonacci (τ anyons)",
                "universal": True,
                "braid_precision_rate": 0.69,  # ~ ln(φ)
                "topological_gap_fraction": 0.1,
                "phase": 4 * math.pi / 5,
            },
            "ising": {
                "name": "Ising (σ anyons)",
                "universal": False,  # Only Clifford gates
                "braid_precision_rate": 0.5,
                "topological_gap_fraction": 0.05,
                "phase": math.pi / 4,
            },
            "universal": {
                "name": "Universal anyons",
                "universal": True,
                "braid_precision_rate": 0.75,
                "topological_gap_fraction": 0.15,
                "phase": 2 * math.pi / PHI,
            },
        }

        params = anyon_params.get(anyon_type, anyon_params["fibonacci"])

        # Precision from braiding
        precision = math.exp(-params["braid_precision_rate"] * n_braids)

        # Equivalent SK gate count for same precision
        equivalent_sk_gates = int(math.ceil(
            math.log(1.0 / max(precision, 1e-300)) ** _SK_CONSTANT
        ))

        # Topological protection: error suppression exponential in gap
        # Error rate ~ exp(-Δ/T) where Δ = topological gap
        gap_fraction = params["topological_gap_fraction"]
        intrinsic_error = math.exp(-gap_fraction * n_braids)

        # Number of distinguishable gates via braiding
        n_distinguishable = int(2 * math.pi / max(precision, 1e-30))

        # Sacred gate approximation: braids needed for GOD_CODE phase
        god_code_braids = int(math.ceil(
            -math.log(1e-10) / params["braid_precision_rate"]
        ))

        return {
            "anyon_type": params["name"],
            "universal": params["universal"],
            "n_braids": n_braids,
            "precision": precision,
            "precision_deg": math.degrees(precision),
            "equivalent_sk_gates": equivalent_sk_gates,
            "intrinsic_error": intrinsic_error,
            "n_distinguishable_gates": n_distinguishable,
            "god_code_braids_needed": god_code_braids,
            "braid_phase": params["phase"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED GATE LIMITS ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class GateLimitsAnalyzer:
    """
    Unified analysis of computronium and Rayleigh limits for quantum gates.

    Combines all gate limit calculations into a comprehensive report
    for any quantum circuit.
    """

    VERSION = "1.0.0"

    def __init__(self):
        self.computronium = ComputroniumGateLimits()
        self.rayleigh = RayleighGateResolution()

    def analyze_circuit_limits(
        self,
        n_qubits: int,
        depth: int,
        n_1q_gates: int = 0,
        n_2q_gates: int = 0,
        n_measurements: int = 0,
        technology: str = "superconducting",
    ) -> Dict[str, Any]:
        """
        Full computronium + Rayleigh analysis of a quantum circuit.

        Args:
            n_qubits: Circuit width
            depth: Circuit depth (layers)
            n_1q_gates: Number of single-qubit gates
            n_2q_gates: Number of two-qubit gates
            n_measurements: Number of measurements
            technology: Hardware platform
        """
        # Gate speed limits
        speed_1q = ComputroniumGateLimits.margolus_levitin_gate_time(
            n_qubits=1, technology=technology
        )
        speed_2q = ComputroniumGateLimits.margolus_levitin_gate_time(
            n_qubits=2, technology=technology
        )

        # Landauer cost
        landauer = ComputroniumGateLimits.landauer_gate_cost(
            n_qubits=n_qubits,
            n_measurements=n_measurements,
            n_resets=0,
            temperature_K=_SC_TEMPERATURE if technology == "superconducting" else 0.001,
        )

        # Depth limit
        depth_lim = ComputroniumGateLimits.circuit_depth_limit(n_qubits)

        # Phase resolution
        phase_res = RayleighGateResolution.phase_resolution()

        # Total circuit time
        time_1q = n_1q_gates * speed_1q["actual_gate_time_s"]
        time_2q = n_2q_gates * speed_2q["actual_gate_time_s"]
        total_time = time_1q + time_2q

        # Fraction of decoherence budget used
        decoherence_fraction = total_time / speed_1q.get("t2_s", _SC_T2) if technology == "superconducting" else total_time / _ION_T2

        # Information processed
        total_gates = n_1q_gates + n_2q_gates
        info_1q = ComputroniumGateLimits.gate_information_capacity(1, "Rz")
        info_2q = ComputroniumGateLimits.gate_information_capacity(2, "CNOT")
        total_holevo_bits = (
            n_1q_gates * info_1q["holevo_bits_per_use"] +
            n_2q_gates * info_2q["holevo_bits_per_use"]
        )

        # GOD_CODE circuit alignment
        god_code_alignment = 1.0 - abs((total_gates % 104) - 52) / 52.0

        return {
            "version": self.VERSION,
            "circuit": {
                "n_qubits": n_qubits,
                "depth": depth,
                "n_1q_gates": n_1q_gates,
                "n_2q_gates": n_2q_gates,
                "n_measurements": n_measurements,
                "total_gates": total_gates,
            },
            "speed_limits": {
                "ml_1q_gate_time_s": speed_1q["ml_minimum_time_s"],
                "actual_1q_gate_time_s": speed_1q["actual_gate_time_s"],
                "ml_2q_gate_time_s": speed_2q["ml_minimum_time_s"],
                "actual_2q_gate_time_s": speed_2q["actual_gate_time_s"],
                "total_circuit_time_s": total_time,
                "decoherence_fraction": decoherence_fraction,
            },
            "depth_limits": {
                "coherence_max_depth": depth_lim["coherence_depth_limit"],
                "fidelity_max_depth": depth_lim["fidelity_depth_limit"],
                "max_useful_depth": depth_lim["max_useful_depth"],
                "depth_utilization": depth / max(depth_lim["max_useful_depth"], 1),
                "quantum_volume": depth_lim["quantum_volume"],
            },
            "energy": {
                "landauer_total_J": landauer["total_landauer_energy_J"],
                "reversible_gates": total_gates - n_measurements,
                "irreversible_ops": n_measurements,
                "signal_to_noise_dB": landauer["signal_to_noise_dB"],
            },
            "resolution": {
                "phase_resolution_rad": phase_res["effective_resolution_rad"],
                "n_distinguishable_angles": phase_res["n_distinguishable_angles"],
                "sacred_phases_resolved": phase_res["sacred_phases_resolved"],
                "limiting_factor": phase_res["limiting_factor"],
            },
            "information": {
                "total_holevo_bits": total_holevo_bits,
                "entangling_gates": n_2q_gates,
                "max_entanglement_ebits": n_2q_gates * (2 / 9),
            },
            "god_code_alignment": god_code_alignment,
            "technology": technology,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "computronium": ComputroniumGateLimits.VERSION,
            "rayleigh": RayleighGateResolution.VERSION,
            "god_code": GOD_CODE,
        }


# Singletons
computronium_gate_limits = ComputroniumGateLimits()
rayleigh_gate_resolution = RayleighGateResolution()
gate_limits_analyzer = GateLimitsAnalyzer()
