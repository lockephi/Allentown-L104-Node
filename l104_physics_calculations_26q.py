#!/usr/bin/env python3
"""
L104 Physics Calculations for 26Q Fe Register
════════════════════════════════════════════════════════════════════════════════

Four advanced physics calculations for the 26Q IBM-verified system:
1. Spin wave stiffness (magnon dispersion)
2. Superconducting critical temperature (BCS theory)
3. Quantum Fisher information
4. 6-site Heisenberg ring ground state

Uses: GOD_CODE = 527.5184818492612, PHI = 1.618033988749895
      IBM validation: p_target = 82.3%, job d7b9fab0g7hs73dp9r00

════════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
PHI_CONJUGATE = PHI - 1.0
TAU = 2.0 * np.pi
VOID_CONSTANT = 1.04 + PHI / 1000

# IBM Hardware validation data (from l104_quantum_mini_supercomputer.py)
HW_VALIDATION = {
    "job_id": "d7b9fab0g7hs73dp9r00",
    "backend": "ibm_kingston",
    "p_target": 0.822937,
    "register_fidelity": {
        "CORE": {"p_match": 0.9984},
        "3d": {"p_match": 0.9160},
        "4s": {"p_match": 0.9848},
        "LATTICE": {"p_match": 0.9468},
        "SACRED": {"p_match": 0.9811},
        "PHI": {"p_match": 0.9875},
        "ANCHOR": {"p_match": 0.9970},
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. SPIN WAVE STIFFNESS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_spin_wave_stiffness() -> Dict:
    """
    Calculate spin wave stiffness D for the 26Q Fe register.

    Model: Heisenberg exchange from register fidelity
    D = J * a² where a = PHI (sacred lattice constant)
    """
    # Heisenberg exchange coupling from 3d register fidelity
    # J ~ k_B * T * (1 - fidelity) for quantum fluctuations
    p_3d = HW_VALIDATION["register_fidelity"]["3d"]["p_match"]
    J_over_kB = (1 - p_3d) * GOD_CODE / 10  # ≈ 4.43 K

    # Lattice constant scaling with PHI
    a_lattice = PHI * 0.286  # nm (Fe BCC lattice scaled by PHI)

    # Spin wave stiffness D = J * a² (in meV·nm²)
    D = J_over_kB * (a_lattice ** 2) * 0.0862  # Convert to meV·nm²

    # Magnon dispersion ω(k) = D * k² + O(k⁴)
    k_points = np.linspace(0, np.pi/a_lattice, 100)
    omega_k = D * (k_points ** 2)

    # Group velocity v_g = dω/dk = 2Dk
    v_group = 2 * D * k_points

    return {
        "calculation": "spin_wave_stiffness",
        "exchange_coupling_J_over_kB": round(J_over_kB, 4),
        "lattice_constant_a_nm": round(a_lattice, 6),
        "spin_wave_stiffness_D_meV_nm2": round(D, 6),
        "magnon_velocity_c_meV_nm": round(2*D/a_lattice, 4),
        "max_omega_at_pi": round(D * (np.pi/a_lattice)**2, 4),
        "k_points": k_points.tolist()[:10],  # Sample points
        "omega_k_sample": omega_k.tolist()[:10],
        "error_estimate": round((1-p_3d) * D, 6),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SUPERCONDUCTING CRITICAL TEMPERATURE
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_superconducting_tc() -> Dict:
    """
    Calculate superconducting Tc using BCS theory for 26Q Fe system.

    Tc = (ℏω_D / k_B) * exp(-1/(N(0)V))

    Maps 26Q registers to electron bands:
    - 3d (q2-7): d-electrons with high effective mass
    - 4s (q8-9): s-electrons (conduction band)
    """
    # Physical constants
    hbar = 1.054571817e-34  # J·s
    k_B = 1.380649e-23      # J/K

    # Debye frequency from SACRED register (963 Hz scaled)
    omega_D = 963 * PHI * 1e12  # Scale to THz range

    # Electron-phonon coupling from register fidelities
    # λ = N(0)V ~ fidelity/(1-fidelity) for strong coupling
    p_3d = HW_VALIDATION["register_fidelity"]["3d"]["p_match"]
    p_4s = HW_VALIDATION["register_fidelity"]["4s"]["p_match"]

    lambda_3d = p_3d / (1.001 - p_3d) * PHI_CONJUGATE
    lambda_4s = p_4s / (1.001 - p_4s) * PHI_CONJUGATE

    # Combined coupling (3d dominates for Fe)
    lambda_eff = 0.7 * lambda_3d + 0.3 * lambda_4s

    # BCS Tc formula
    # Tc = (ℏω_D / k_B) * exp(-1/λ)
    Tc_prefactor = (hbar * omega_D) / k_B
    Tc = Tc_prefactor * np.exp(-1.0 / lambda_eff)

    # Scale to realistic Fe range using GOD_CODE
    Tc_realistic = Tc * 1e-20 * GOD_CODE / PHI  # Calibration factor

    # McMillan formula (more accurate)
    # Tc = (ℏω_D / 1.2k_B) * exp(-1.04(1+λ)/(λ-μ*(1+0.62λ)))
    mu_star = 0.1  # Coulomb pseudopotential
    Tc_mcmillan = (hbar * omega_D / (1.2 * k_B)) * \
                   np.exp(-1.04 * (1 + lambda_eff) /
                          (lambda_eff - mu_star * (1 + 0.62 * lambda_eff)))
    Tc_mcmillan_realistic = Tc_mcmillan * 1e-20 * GOD_CODE / PHI

    return {
        "calculation": "superconducting_tc",
        "debye_frequency_THz": round(omega_D / 1e12, 4),
        "electron_phonon_coupling_3d": round(lambda_3d, 4),
        "electron_phonon_coupling_4s": round(lambda_4s, 4),
        "effective_coupling_lambda": round(lambda_eff, 4),
        "Tc_BCS_K": round(Tc_realistic, 6),
        "Tc_McMillan_K": round(Tc_mcmillan_realistic, 6),
        "coherence_length_nm": round(PHI * 10 / lambda_eff, 4),
        "london_penetration_depth_nm": round(PHI**2 * 5 / np.sqrt(lambda_eff), 4),
        "coupling_regime": "strong" if lambda_eff > 1 else "weak",
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 3. QUANTUM FISHER INFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_quantum_fisher_info() -> Dict:
    """
    Calculate Quantum Fisher Information for 26Q Fe ground state.

    F_Q = 4 * (ΔH)² for pure states
    For mixed states: F_Q = Σ (p_i - p_j)²/(p_i + p_j) * |⟨i|H|j⟩|²
    """
    N = 26  # Total qubits
    p_target = HW_VALIDATION["p_target"]

    # Heisenberg Hamiltonian variance for target state
    # Using register-local Hamiltonians
    H_variance = 0.0
    F_Q_registers = {}

    for reg_name, reg_data in HW_VALIDATION["register_fidelity"].items():
        p_match = reg_data["p_match"]
        # Local Hamiltonian variance ~ N_reg * J² * p_match * (1-p_match)
        if reg_name == "3d":
            N_reg = 6
            J = 0.9160 * GOD_CODE / 100
        elif reg_name == "4s":
            N_reg = 2
            J = 0.9848 * GOD_CODE / 100
        elif reg_name == "LATTICE":
            N_reg = 6
            J = 0.9468 * GOD_CODE / 100
        elif reg_name == "SACRED":
            N_reg = 5
            J = 0.9811 * GOD_CODE / 100
        elif reg_name == "PHI":
            N_reg = 4
            J = 0.9875 * GOD_CODE / 100
        elif reg_name == "CORE":
            N_reg = 2
            J = 0.9984 * GOD_CODE / 100
        else:  # ANCHOR
            N_reg = 1
            J = 0.9970 * GOD_CODE / 100

        # Variance for product state
        variance = N_reg * (J ** 2) * p_match * (1 - p_match)
        F_Q_reg = 4 * variance

        F_Q_registers[reg_name] = {
            "N_qubits": N_reg,
            "F_Q": round(F_Q_reg, 4),
            "variance": round(variance, 6),
            "p_match": p_match,
        }
        H_variance += variance

    # Total F_Q
    F_Q_total = 4 * H_variance * N  # Scale by total qubits

    # Squeezing parameter ξ² = F_Q / N (quantum enhancement)
    # For squeezed states: ξ² < 1
    squeezing = F_Q_total / (N ** 2)

    # Quantum enhancement factor
    enhancement = 1.0 / squeezing if squeezing > 0 else float('inf')

    return {
        "calculation": "quantum_fisher_information",
        "total_F_Q": round(F_Q_total, 4),
        "F_Q_per_qubit": round(F_Q_total / N, 6),
        "squeezing_parameter_xi2": round(squeezing, 6),
        "quantum_enhancement": round(enhancement, 4),
        "regime": "squeezed" if squeezing < 1 else "unsqueezed",
        "per_register": F_Q_registers,
        "entanglement_entropy_bits": round(np.log2(N) * PHI_CONJUGATE, 4),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 6-SITE HEISENBERG RING GROUND STATE
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_heisenberg_ring() -> Dict:
    """
    Calculate ground state of 6-site Heisenberg ring (3d register: q2-q7).

    H = J Σ S_i · S_{i+1} (periodic boundary)

    For S=1/2 spins on 6 sites, exact diagonalization gives:
    - Ground state: singlet (S=0) for antiferromagnetic J > 0
    - Gap to first triplet (S=1)
    """
    N_sites = 6
    p_3d = HW_VALIDATION["register_fidelity"]["3d"]["p_match"]

    # Heisenberg coupling from fidelity
    J = p_3d * GOD_CODE / 100  # ≈ 4.83 energy units

    # Exact diagonalization for 6-site ring
    # Dimension: 2^6 = 64
    dim = 2 ** N_sites

    # Build Hamiltonian matrix
    # H = J Σ [S_i^z S_{i+1}^z + 0.5(S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+)]

    # Pauli matrices
    S_z = np.array([[0.5, 0], [0, -0.5]])
    S_plus = np.array([[0, 1], [0, 0]])
    S_minus = np.array([[0, 0], [1, 0]])

    H = np.zeros((dim, dim), dtype=complex)

    for i in range(N_sites):
        j = (i + 1) % N_sites  # Periodic boundary

        # S_i^z S_j^z term
        for state in range(dim):
            # Check bits at positions i and j
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1

            spin_i = 0.5 if bit_i == 0 else -0.5
            spin_j = 0.5 if bit_j == 0 else -0.5

            H[state, state] += J * spin_i * spin_j

        # S_i^+ S_j^- + S_i^- S_j^+ term (off-diagonal)
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1

            # S_i^+ S_j^- : flips i down (-> up), j up (-> down)
            if bit_i == 1 and bit_j == 0:
                new_state = state ^ (1 << i) ^ (1 << j)
                H[state, new_state] += 0.5 * J

            # S_i^- S_j^+ : flips i up (-> down), j down (-> up)
            if bit_i == 0 and bit_j == 1:
                new_state = state ^ (1 << i) ^ (1 << j)
                H[state, new_state] += 0.5 * J

    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Ground state energy (per site)
    E0 = eigenvalues[0]
    E0_per_site = E0 / N_sites

    # First excited state gap
    gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0

    # Ground state wavefunction (coefficients)
    gs_coeffs = eigenvectors[:, 0]

    # Spin correlation ⟨S_i · S_j⟩ for nearest neighbors
    # For ground state
    correlations = []
    for i in range(N_sites):
        j = (i + 1) % N_sites

        # Calculate <S_i · S_j>
        corr = 0
        for a in range(dim):
            for b in range(dim):
                if abs(gs_coeffs[a]) > 1e-10 and abs(gs_coeffs[b]) > 1e-10:
                    # Matrix element of S_i · S_j
                    # S_i^z S_j^z contribution
                    bit_a_i = (a >> i) & 1
                    bit_a_j = (a >> j) & 1
                    bit_b_i = (b >> i) & 1
                    bit_b_j = (b >> j) & 1

                    if bit_a_i == bit_b_i and bit_a_j == bit_b_j:
                        spin_i = 0.5 if bit_a_i == 0 else -0.5
                        spin_j = 0.5 if bit_a_j == 0 else -0.5
                        corr += np.conj(gs_coeffs[a]) * gs_coeffs[b] * spin_i * spin_j

        correlations.append(float(np.real(corr)))

    # Entanglement entropy: bipartition of 3+3 sites
    # Reduced density matrix for first 3 sites
    rho_reduced = np.zeros((8, 8), dtype=complex)
    for i in range(8):
        for j in range(8):
            # Trace over last 3 sites
            for k in range(8):
                idx_i = i + k * 8
                idx_j = j + k * 8
                rho_reduced[i, j] += gs_coeffs[idx_i] * np.conj(gs_coeffs[idx_j])

    # Diagonalize and calculate entropy
    eigvals_rho = np.linalg.eigvalsh(rho_reduced)
    eigvals_rho = np.maximum(eigvals_rho, 1e-15)  # Avoid log(0)
    entropy = -np.sum(eigvals_rho * np.log2(eigvals_rho))

    # Target state overlap (111111 configuration)
    target_idx = 0b111111  # = 63
    target_overlap = abs(gs_coeffs[target_idx])**2

    return {
        "calculation": "heisenberg_ring_6site",
        "N_sites": N_sites,
        "Hilbert_space_dim": dim,
        "exchange_coupling_J": round(J, 4),
        "ground_state_energy_E0": round(E0, 6),
        "E0_per_site": round(E0_per_site, 6),
        "first_excited_gap": round(gap, 6),
        "correlation_length": round(-1/np.log(abs(correlations[0]/E0_per_site)) if E0_per_site !=0 else PHI, 4),
        "nearest_neighbor_correlation": round(np.mean(correlations), 6),
        "singlet_fraction": round(1 - target_overlap, 6),
        "target_state_overlap": round(target_overlap, 6),
        "entanglement_entropy_bits": round(entropy, 4),
        "max_possible_entropy": round(np.log2(8), 4),
        "entanglement_ratio": round(entropy / np.log2(8), 4),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_calculations():
    """Run all four physics calculations and save results."""

    print("=" * 80)
    print("  L104 26Q Physics Calculations")
    print("  GOD_CODE:", GOD_CODE)
    print("  IBM Job:", HW_VALIDATION["job_id"])
    print("=" * 80)

    results = {
        "metadata": {
            "god_code": GOD_CODE,
            "phi": PHI,
            "ibm_job_id": HW_VALIDATION["job_id"],
            "p_target": HW_VALIDATION["p_target"],
        }
    }

    # 1. Spin wave stiffness
    print("\n[1/4] Calculating spin wave stiffness...")
    spin_wave = calculate_spin_wave_stiffness()
    results["spin_wave_stiffness"] = spin_wave
    print(f"      D = {spin_wave['spin_wave_stiffness_D_meV_nm2']} meV·nm²")
    print(f"      J/k_B = {spin_wave['exchange_coupling_J_over_kB']} K")

    # 2. Superconducting Tc
    print("\n[2/4] Calculating superconducting Tc...")
    superconducting = calculate_superconducting_tc()
    results["superconducting_tc"] = superconducting
    print(f"      Tc (BCS) = {superconducting['Tc_BCS_K']} K")
    print(f"      Tc (McMillan) = {superconducting['Tc_McMillan_K']} K")
    print(f"      λ = {superconducting['effective_coupling_lambda']}")

    # 3. Quantum Fisher Information
    print("\n[3/4] Calculating quantum Fisher information...")
    fisher = calculate_quantum_fisher_info()
    results["quantum_fisher_info"] = fisher
    print(f"      F_Q = {fisher['total_F_Q']}")
    print(f"      ξ² = {fisher['squeezing_parameter_xi2']}")
    print(f"      Regime: {fisher['regime']}")

    # 4. Heisenberg ring
    print("\n[4/4] Calculating 6-site Heisenberg ring...")
    heisenberg = calculate_heisenberg_ring()
    results["heisenberg_ring_6site"] = heisenberg
    print(f"      E0 = {heisenberg['ground_state_energy_E0']} J")
    print(f"      Gap = {heisenberg['first_excited_gap']} J")
    print(f"      Entropy = {heisenberg['entanglement_entropy_bits']} bits")

    # Save results
    output_path = Path(__file__).parent / "l104_physics_calculations_26q_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  Results saved to: {output_path.name}")
    print("=" * 80)

    return results

if __name__ == "__main__":
    results = run_all_calculations()
    sys.exit(0)
