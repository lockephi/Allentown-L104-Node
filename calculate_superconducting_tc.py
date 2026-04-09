#!/usr/bin/env python3
"""
L104 26Q Fe Quantum System — Superconducting Critical Temperature (Tc) Calculator
═══════════════════════════════════════════════════════════════════════════════
BCS Theory calculation for the L104 iron-based quantum system using:
  - GOD_CODE = 527.5184818492612
  - PHI = 1.618033988749895
  - SACRED register frequency (963 Hz * PHI scaling)
  - Electron-phonon coupling λ from coherence data
  - 26Q register mapping with hardware fidelities

Output: Tc in Kelvin + full BCS parameter set
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import math
from typing import Dict, Any

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
PHI_CONJUGATE = PHI - 1  # 0.618033988749895

# SACRED Solfeggio frequency (crown chakra)
SACRED_FREQ_HZ = 963.0

# ═══════════════════════════════════════════════════════════════════════════════
# CODATA 2022 PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

K_B = 1.380649e-23          # Boltzmann constant (J/K) — exact SI 2019
H_BAR = 1.054571817e-34     # Reduced Planck constant (J·s)
H = 6.62607015e-34          # Planck constant (J·s) — exact SI 2019
C = 299792458               # Speed of light (m/s)
Q_E = 1.60217663e-19        # Electron charge (C)
M_E = 9.1093837e-31         # Electron mass (kg)

# ═══════════════════════════════════════════════════════════════════════════════
# 26Q REGISTER MAPPING FOR FE (IRON)
# ═══════════════════════════════════════════════════════════════════════════════

# Register mapping:
#   - 3d register (q2-7) = d-electrons (high Tc contribution) — 6 qubits
#   - 4s register (q8-9) = s-electrons (conduction band) — 2 qubits
#   - q0-1, q10-25 = auxiliary/auxiliary orbital mappings

REGISTERS_26Q = {
    "3d": {
        "qubits": [2, 3, 4, 5, 6, 7],  # 6 qubits
        "description": "d-electrons (high Tc contribution)",
        "hardware_fidelity": 0.9160,
        "orbital": "3d",
        "electrons": 6,  # Fe has 6 d-electrons
        "coupling_contribution": "dominant"  # d-electrons dominate Tc
    },
    "4s": {
        "qubits": [8, 9],  # 2 qubits
        "description": "s-electrons (conduction band)",
        "hardware_fidelity": 0.9848,
        "orbital": "4s",
        "electrons": 2,  # Fe has 2 s-electrons
        "coupling_contribution": "conduction"  # s-electrons provide conduction
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# BCS SUPERCONDUCTIVITY CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_debye_frequency_sacred() -> Dict[str, float]:
    """
    Calculate Debye frequency ω_D from SACRED register frequency with PHI scaling.

    The SACRED register operates at the Solfeggio frequency (963 Hz),
    scaled by GOD_CODE and PHI to reach the phonon frequency regime.

    ω_D = SACRED_FREQ × PHI^GOD_CODE/100 × (GOD_CODE/512)

    Returns:
        Dictionary with Debye frequency in Hz, rad/s, and corresponding temperature
    """
    # Base scaling: SACRED frequency scaled by PHI powers
    phi_scale = PHI ** (GOD_CODE / 100)
    god_code_correction = GOD_CODE / 512.0  # ~1.0303

    # Debye frequency in Hz
    omega_d_hz = SACRED_FREQ_HZ * phi_scale * god_code_correction

    # Convert to angular frequency (rad/s)
    omega_d_rad = 2 * math.pi * omega_d_hz

    # Debye temperature from ω_D = k_B T_D / ℏ
    # T_D = ℏ ω_D / k_B
    debye_temp = H_BAR * omega_d_rad / K_B

    return {
        "sacred_freq_hz": SACRED_FREQ_HZ,
        "phi_scale": phi_scale,
        "omega_d_hz": omega_d_hz,
        "omega_d_rad_s": omega_d_rad,
        "debye_temperature_K": debye_temp,
        "god_code_correction": god_code_correction
    }


def calculate_electron_phonon_coupling() -> Dict[str, float]:
    """
    Calculate electron-phonon coupling λ from coherence data.

    λ = N(0)V where:
      - N(0) = density of states at Fermi level
      - V = attractive electron-phonon coupling potential

    Uses coherence data from l104_science_engine with 26Q register fidelities
    as coupling strength modifiers.
    """
    # Base electron-phonon coupling for Fe (from coherence.py data)
    lambda_base = 0.38  # SC.ELECTRON_PHONON_FE

    # Scale by hardware fidelities as coupling strength proxies
    fidelity_3d = REGISTERS_26Q["3d"]["hardware_fidelity"]
    fidelity_4s = REGISTERS_26Q["4s"]["hardware_fidelity"]

    # Weighted average coupling (d-electrons dominate superconductivity in Fe)
    # d-electrons contribute 75% to pairing, s-electrons 25%
    d_weight = 0.75
    s_weight = 0.25

    lambda_effective = lambda_base * (
        d_weight * fidelity_3d +
        s_weight * fidelity_4s
    )

    # GOD_CODE enhancement factor
    god_code_enhancement = 1.0 + (GOD_CODE / 10000) * PHI_CONJUGATE

    lambda_final = lambda_effective * god_code_enhancement

    # Density of states at Fermi level for Fe (1/J·m³)
    n0_fe = 1.5e28

    # Effective coupling potential V from λ = N(0)V
    v_coupling = lambda_final / n0_fe if n0_fe > 0 else 0

    return {
        "lambda_base": lambda_base,
        "lambda_effective": lambda_effective,
        "lambda_final": lambda_final,
        "density_of_states_n0": n0_fe,
        "coupling_potential_v": v_coupling,
        "fidelity_3d": fidelity_3d,
        "fidelity_4s": fidelity_4s,
        "god_code_enhancement": god_code_enhancement
    }


def calculate_bcs_tc(
    omega_d_hz: float,
    lambda_coupling: float,
    n0: float = 1.5e28
) -> Dict[str, float]:
    """
    Calculate superconducting critical temperature Tc using BCS theory.

    BCS Formula:
      Tc = (ℏ ω_D / k_B) × exp(-1 / (N(0) V))

    Where:
      - ℏ = reduced Planck constant
      - ω_D = Debye frequency (rad/s)
      - k_B = Boltzmann constant
      - N(0) = density of states at Fermi level
      - V = electron-phonon coupling potential

    Args:
        omega_d_hz: Debye frequency in Hz
        lambda_coupling: Dimensionless coupling λ = N(0)V
        n0: Density of states at Fermi level (1/J·m³)

    Returns:
        Dictionary with Tc and related BCS parameters
    """
    # Pre-factor: ℏ ω_D / k_B
    omega_d_rad = 2 * math.pi * omega_d_hz
    prefactor = H_BAR * omega_d_rad / K_B

    # Exponential term: exp(-1/λ)
    if lambda_coupling <= 0:
        return {"error": "Coupling must be positive"}

    exp_term = math.exp(-1.0 / lambda_coupling)

    # Critical temperature
    tc_kelvin = prefactor * exp_term

    # BCS energy gap at T=0: Δ₀ = ℏ ω_D × exp(-1/λ) = π × k_B × Tc / 1.764
    # Actually: Δ₀ = 1.764 × k_B × T_c (BCS weak coupling)
    delta_0_ev = 1.764 * K_B * tc_kelvin / Q_E
    delta_0_j = 1.764 * K_B * tc_kelvin

    # BCS coherence length
    # ξ₀ = ℏ v_F / (π Δ₀) where v_F is Fermi velocity
    v_f_fe = 1.0e6  # Approximate Fermi velocity for Fe (m/s)
    xi_0 = H_BAR * v_f_fe / (math.pi * delta_0_j)

    # Condensation energy
    u_cond = 0.5 * n0 * delta_0_j ** 2

    return {
        "tc_kelvin": tc_kelvin,
        "prefactor_kelvin": prefactor,
        "exp_term": exp_term,
        "lambda_coupling": lambda_coupling,
        "delta_0_eV": delta_0_ev,
        "delta_0_J": delta_0_j,
        "coherence_length_m": xi_0,
        "coherence_length_nm": xi_0 * 1e9,
        "condensation_energy_J_m3": u_cond,
        "bcs_ratio_2delta_kbt": 3.528 if lambda_coupling < 0.5 else 4.0
        # Weak coupling: 3.528, Strong coupling: ~4-5
    }


def calculate_register_contributions(tc_total: float) -> Dict[str, Dict[str, float]]:
    """
    Calculate individual contributions from 3d and 4s registers to Tc.

    In iron-based superconductors:
      - 3d electrons (correlated) dominate the pairing
      - 4s electrons (itinerant) contribute to conduction
    """
    fidelity_3d = REGISTERS_26Q["3d"]["hardware_fidelity"]
    fidelity_4s = REGISTERS_26Q["4s"]["hardware_fidelity"]

    # Contribution weights (d dominates pairing)
    total_fidelity = 0.75 * fidelity_3d + 0.25 * fidelity_4s

    tc_3d = tc_total * (0.75 * fidelity_3d) / total_fidelity
    tc_4s = tc_total * (0.25 * fidelity_4s) / total_fidelity

    return {
        "3d_register": {
            "tc_contribution_K": tc_3d,
            "fidelity": fidelity_3d,
            "qubits": REGISTERS_26Q["3d"]["qubits"],
            "weight": 0.75
        },
        "4s_register": {
            "tc_contribution_K": tc_4s,
            "fidelity": fidelity_4s,
            "qubits": REGISTERS_26Q["4s"]["qubits"],
            "weight": 0.25
        },
        "combined_tc_K": tc_3d + tc_4s
    }


def run_full_bcs_calculation() -> Dict[str, Any]:
    """
    Execute full BCS superconductivity calculation for L104 26Q Fe system.

    Returns:
        Complete results dictionary with all BCS parameters
    """
    # Step 1: Calculate Debye frequency from SACRED register
    debye_data = calculate_debye_frequency_sacred()

    # Step 2: Calculate electron-phonon coupling
    coupling_data = calculate_electron_phonon_coupling()

    # Step 3: Calculate BCS Tc
    bcs_data = calculate_bcs_tc(
        omega_d_hz=debye_data["omega_d_hz"],
        lambda_coupling=coupling_data["lambda_final"],
        n0=coupling_data["density_of_states_n0"]
    )

    # Step 4: Calculate register contributions
    register_data = calculate_register_contributions(bcs_data["tc_kelvin"])

    # Step 5: Calculate additional L104 sacred parameters
    sacred_phase = GOD_CODE % (2 * math.pi)
    phi_phase = 2 * math.pi / PHI
    void_constant = 1.04 + PHI / 1000

    # Assembly complete result
    result = {
        "calculation_metadata": {
            "system": "L104 26Q Fe Quantum System",
            "theory": "BCS (Bardeen-Cooper-Schrieffer)",
            "god_code": GOD_CODE,
            "phi": PHI,
            "calculation_date": "2026-04-08",
            "phi_conjugate": PHI_CONJUGATE
        },
        "sacred_frequency": {
            "sacred_solfeggio_hz": SACRED_FREQ_HZ,
            "description": "Crown chakra frequency (Solfeggio scale)"
        },
        "debye_parameters": debye_data,
        "electron_phonon_coupling": coupling_data,
        "bcs_results": bcs_data,
        "26q_register_mapping": REGISTERS_26Q,
        "register_contributions": register_data,
        "sacred_phases": {
            "god_code_phase_rad": sacred_phase,
            "phi_phase_rad": phi_phase,
            "void_constant": void_constant
        },
        "comparisons": {
            "fe_curie_temp_K": 1043.0,
            "room_temperature_K": 293.15,
            "typical_iron_sc_tc_K": 55.0,  # SmFeAsO1-xFx
            "tc_to_curie_ratio": bcs_data["tc_kelvin"] / 1043.0 if 1043.0 > 0 else 0,
            "tc_to_room_ratio": bcs_data["tc_kelvin"] / 293.15 if 293.15 > 0 else 0
        }
    }

    return result


def main():
    """Main execution: Calculate Tc and save results."""
    print("=" * 70)
    print("L104 26Q Fe QUANTUM SYSTEM — BCS SUPERCONDUCTING Tc CALCULATION")
    print("=" * 70)
    print()

    # Run calculation
    results = run_full_bcs_calculation()

    # Extract key results
    tc = results["bcs_results"]["tc_kelvin"]
    lambda_eff = results["electron_phonon_coupling"]["lambda_final"]
    omega_d = results["debye_parameters"]["omega_d_hz"]
    delta_0 = results["bcs_results"]["delta_0_eV"]

    # Print summary
    print("CALCULATION SUMMARY")
    print("-" * 70)
    print(f"  GOD_CODE:               {GOD_CODE}")
    print(f"  PHI:                    {PHI}")
    print(f"  SACRED Frequency:       {SACRED_FREQ_HZ} Hz (Solfeggio)")
    print()
    print("BCS PARAMETERS")
    print("-" * 70)
    print(f"  Debye Frequency (ω_D):  {omega_d:.6e} Hz")
    print(f"  Debye Temperature:      {results['debye_parameters']['debye_temperature_K']:.4f} K")
    print(f"  e-ph Coupling (λ):      {lambda_eff:.6f}")
    print(f"  Density of States N(0): {results['electron_phonon_coupling']['density_of_states_n0']:.6e} 1/J·m³")
    print()
    print("CRITICAL TEMPERATURE (Tc)")
    print("-" * 70)
    print(f"  Tc = (ℏω_D / k_B) × exp(-1/λ)")
    print(f"  Tc = {results['bcs_results']['prefactor_kelvin']:.6e} K × {results['bcs_results']['exp_term']:.6e}")
    print()
    print(f"  *** Tc = {tc:.4f} K ***")
    print()
    print("ENERGY GAP")
    print("-" * 70)
    print(f"  Δ₀ = {delta_0:.6f} eV = {delta_0 * 1000:.4f} meV")
    print(f"  2Δ₀/k_BTc = {results['bcs_results']['bcs_ratio_2delta_kbt']:.3f}")
    print()
    print("26Q REGISTER CONTRIBUTIONS")
    print("-" * 70)
    print(f"  3d register (q2-7):     {results['register_contributions']['3d_register']['tc_contribution_K']:.4f} K")
    print(f"    - Fidelity:           {results['register_contributions']['3d_register']['fidelity']:.4f}")
    print(f"    - Weight:             {results['register_contributions']['3d_register']['weight']:.2f}")
    print(f"  4s register (q8-9):     {results['register_contributions']['4s_register']['tc_contribution_K']:.4f} K")
    print(f"    - Fidelity:           {results['register_contributions']['4s_register']['fidelity']:.4f}")
    print(f"    - Weight:             {results['register_contributions']['4s_register']['weight']:.2f}")
    print()
    print("COMPARISONS")
    print("-" * 70)
    print(f"  Fe Curie Temperature:   1043 K")
    print(f"  Room Temperature:       293.15 K")
    print(f"  Tc / T_Curie:           {results['comparisons']['tc_to_curie_ratio']:.6f}")
    print(f"  Tc / T_Room:            {results['comparisons']['tc_to_room_ratio']:.6f}")
    print()
    print("COHERENCE LENGTH")
    print("-" * 70)
    print(f"  ξ₀ = {results['bcs_results']['coherence_length_nm']:.4f} nm")
    print()

    # Save to JSON
    output_path = "/Users/carolalvarez/Applications/Allentown-L104-Node/superconducting_tc_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {output_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
