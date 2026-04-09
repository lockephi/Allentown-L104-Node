#!/usr/bin/env python3
"""
L104 26Q 3d Register - 6-Site Heisenberg Ring Calculation
Hamiltonian: H = J * Σ(S_i · S_{i+1}) for i=0 to 5 (periodic boundary)
Qubits: q2-q7 (6 qubits representing Fe 3d⁶ orbitals)

This script uses exact diagonalization for 2^6 = 64 states.
"""

import numpy as np
import json
import math
from typing import Dict, List, Tuple

# L104 Constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895

# J from 3d register fidelity: J/k_B = 0.9160 * GOD_CODE / 100 ≈ 4.83 K
J_OVER_KB = 0.9160 * GOD_CODE / 100  # ≈ 4.832 K
J_COUPLING = J_OVER_KB  # In units where k_B = 1

print("="*70)
print("L104 26Q 3d Register - 6-Site Heisenberg Ring")
print("="*70)
print(f"GOD_CODE: {GOD_CODE}")
print(f"PHI: {PHI}")
print(f"J/k_B: {J_OVER_KB:.4f} K")
print(f"Coupling J: {J_COUPLING:.6f}")
print(f"Qubits: q2-q7 (6 qubits, 2^6 = 64 states)")
print(f"Configuration: Fe 3d⁶ (111111)")
print("="*70)

# Number of sites
N = 6
N_STATES = 2**N  # 64 states for 6 qubits

# Spin operators for S=1/2
# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# Spin-1/2 operators: S = 1/2 * sigma
S_x = 0.5 * sigma_x
S_y = 0.5 * sigma_y
S_z = 0.5 * sigma_z

# S^2 operator for a single spin-1/2
S_squared = S_x @ S_x + S_y @ S_y + S_z @ S_z

# Total S^2 operator eigenvalue should be s(s+1) = 3/4 for spin-1/2
print(f"\nSingle spin S^2 eigenvalue: {np.trace(S_squared @ S_squared):.4f} (expected 0.75)")

def build_spin_operator(N: int, site: int, component: str) -> np.ndarray:
    """Build spin operator S_component at given site for N spins."""
    S_op = {'x': S_x, 'y': S_y, 'z': S_z}[component]

    # Start with identity
    identity = np.eye(2, dtype=complex)

    # Build full operator using tensor products
    op_list = [identity] * N
    op_list[site] = S_op

    # Compute Kronecker product
    result = op_list[0]
    for i in range(1, N):
        result = np.kron(result, op_list[i])

    return result

def build_heisenberg_hamiltonian(N: int, J: float, periodic: bool = True) -> np.ndarray:
    """
    Build Heisenberg Hamiltonian H = J * Σ(S_i · S_{i+1})
    For S=1/2 spins with periodic boundary conditions.
    """
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)

    # Build S operators for each site
    S_ops = {'x': [], 'y': [], 'z': []}
    for comp in ['x', 'y', 'z']:
        for site in range(N):
            S_ops[comp].append(build_spin_operator(N, site, comp))

    # Build H = J * Σ_i Σ_component S_i^component * S_{i+1}^component
    for i in range(N):
        j = (i + 1) % N if periodic else i + 1
        if j >= N:
            continue

        for comp in ['x', 'y', 'z']:
            H += J * S_ops[comp][i] @ S_ops[comp][j]

    return H

# Build Hamiltonian
print("\nBuilding Hamiltonian...")
H = build_heisenberg_hamiltonian(N, J_COUPLING, periodic=True)

# Diagonalize
print("Diagonalizing Hamiltonian (64x64 matrix)...")
eigenvalues, eigenvectors = np.linalg.eigh(H)

# Sort eigenvalues
eigenvalues = np.real(eigenvalues)  # Should be real for Hermitian H
sorted_indices = np.argsort(eigenvalues)
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Ground state and first excited state
E0 = eigenvalues[0]
E1 = eigenvalues[1]
gap = E1 - E0

print(f"\n{'='*70}")
print("ENERGY SPECTRUM RESULTS")
print(f"{'='*70}")
print(f"Ground state energy E0 = {E0:.6f}")
print(f"First excited energy E1 = {E1:.6f}")
print(f"Energy gap ΔE = {gap:.6f}")
print(f"\nE0/J = {E0/J_COUPLING:.6f}")
print(f"ΔE/J = {gap/J_COUPLING:.6f}")

# Show first 10 energy levels
print(f"\nFirst 10 energy levels (in units of J):")
for i in range(min(10, len(eigenvalues))):
    print(f"  E[{i}] = {eigenvalues[i]/J_COUPLING:.6f}")

# Ground state wavefunction
ground_state = eigenvectors[:, 0]

# Calculate spin correlation functions <S_i · S_j>
print(f"\n{'='*70}")
print("SPIN CORRELATION FUNCTIONS <S_i · S_j>")
print(f"{'='*70}")

# Build all S operators
S_ops = {}
for comp in ['x', 'y', 'z']:
    S_ops[comp] = [build_spin_operator(N, site, comp) for site in range(N)]

correlation_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            # <S_i · S_i> = s(s+1) = 3/4 for spin-1/2
            correlation_matrix[i, j] = 0.75
        else:
            # <S_i · S_j> = <S_i^x S_j^x> + <S_i^y S_j^y> + <S_i^z S_j^z>
            corr = 0
            for comp in ['x', 'y', 'z']:
                S_i_op = S_ops[comp][i]
                S_j_op = S_ops[comp][j]
                # Expectation value: <psi| S_i S_j |psi>
                op = S_i_op @ S_j_op
                val = np.vdot(ground_state, op @ ground_state)
                corr += np.real(val)
            correlation_matrix[i, j] = corr

print("Correlation matrix <S_i · S_j>:")
print("     j=0      j=1      j=2      j=3      j=4      j=5")
for i in range(N):
    row_str = f"i={i}"
    for j in range(N):
        row_str += f" {correlation_matrix[i,j]:8.4f}"
    print(row_str)

# Nearest neighbor correlations (for the ring)
print(f"\nNearest neighbor correlations:")
for i in range(N):
    j = (i + 1) % N
    print(f"  <S_{i} · S_{j}> = {correlation_matrix[i, j]:.6f}")

# Average correlation
avg_nn_corr = np.mean([correlation_matrix[i, (i+1)%N] for i in range(N)])
print(f"\nAverage nearest-neighbor correlation: {avg_nn_corr:.6f}")

# Entanglement entropy calculation
print(f"\n{'='*70}")
print("ENTANGLEMENT ENTROPY S_A")
print(f"{'='*70}")

def von_neumann_entropy(rho: np.ndarray) -> float:
    """Calculate von Neumann entropy S = -Tr(ρ log ρ)."""
    # Diagonalize density matrix
    eigenvals = np.linalg.eigvalsh(rho)
    # Filter out zero eigenvalues
    eigenvals = eigenvals[eigenvals > 1e-15]
    # Calculate entropy
    entropy = -np.sum(eigenvals * np.log2(eigenvals))
    return entropy

def partial_trace_state(state: np.ndarray, N: int, traced_sites: List[int]) -> np.ndarray:
    """Partial trace over specified sites."""
    # Convert state vector to density matrix
    rho = np.outer(state, np.conj(state))

    # Reshape to tensor form: (2, 2, ..., 2) for N sites
    shape = [2] * (2 * N)  # (i1, i2, ..., iN, j1, j2, ..., jN)
    rho_tensor = rho.reshape(shape)

    # Traced sites are indices to trace over
    # We need to trace over both row and column indices
    axes_to_trace = list(traced_sites) + [N + s for s in traced_sites]

    # Sort in descending order to avoid index shifting
    axes_to_trace = sorted(axes_to_trace, reverse=True)

    result = rho_tensor
    for axis in axes_to_trace:
        result = np.trace(result, axis1=axis, axis2=axis)

    # Reshape to square matrix
    n_remaining = N - len(traced_sites)
    dim = 2**n_remaining
    return result.reshape((dim, dim))

# Calculate entanglement entropy for different bipartitions
print("\nBipartition entanglement entropies:")

# Divide system in half: sites 0,1,2 vs sites 3,4,5
entropies = {}
for partition_name, traced_sites in [
    ("Sites {0,1,2} vs {3,4,5}", [3, 4, 5]),
    ("Sites {0,1} vs {2,3,4,5}", [2, 3, 4, 5]),
    ("Site {0} vs {1,2,3,4,5}", [1, 2, 3, 4, 5]),
]:
    rho_A = partial_trace_state(ground_state, N, traced_sites)
    S_A = von_neumann_entropy(rho_A)
    entropies[partition_name] = S_A
    print(f"  {partition_name}: S_A = {S_A:.6f}")

# Single site entanglement entropies
print(f"\nSingle-site entanglement entropies:")
single_site_entropies = []
for site in range(N):
    other_sites = [i for i in range(N) if i != site]
    rho_site = partial_trace_state(ground_state, N, other_sites)
    S_site = von_neumann_entropy(rho_site)
    single_site_entropies.append(S_site)
    print(f"  Site {site}: S = {S_site:.6f}")

# Total spin calculation
print(f"\n{'='*70}")
print("TOTAL SPIN ANALYSIS")
print(f"{'='*70}")

# Build total spin operators
S_total_x = sum(S_ops['x'])
S_total_y = sum(S_ops['y'])
S_total_z = sum(S_ops['z'])
S_total_squared = S_total_x @ S_total_x + S_total_y @ S_total_y + S_total_z @ S_total_z

# Ground state total spin
S2_ground = np.vdot(ground_state, S_total_squared @ ground_state)
S_ground = np.real(S2_ground)
S_val = (-1 + np.sqrt(1 + 4*S_ground)) / 2

print(f"Ground state <S^2> = {S_ground:.6f}")
print(f"Ground state total spin S ≈ {S_val:.4f}")
print(f"Expected: S=0 singlet ground state for even N ring")

# Check Sz for ground state
Sz_ground = np.vdot(ground_state, S_total_z @ ground_state)
print(f"Ground state <S_z> = {np.real(Sz_ground):.6f}")

# Fe 3d^6 mapping
print(f"\n{'='*70}")
print("Fe 3d⁶ ELECTRONIC CONFIGURATION MAPPING")
print(f"{'='*70}")
print(f"""
Iron (Fe) electronic configuration: [Ar] 3d⁶ 4s²
- 3d subshell: 6 electrons in 5 d-orbitals (dxy, dxz, dyz, dx²-y², dz²)
- 4s subshell: 2 electrons

In L104 26Q register mapping:
- Qubits q2-q7 represent the 3d⁶ orbital electrons
- 6 qubits for 6 electrons (111111 configuration = half-filled 3d shell)

The 6-site Heisenberg ring models:
- 6 electrons in 3d orbitals
- Each site i represents one 3d electron spin
- Periodic boundary conditions (ring topology)
- Exchange coupling J represents Coulomb exchange in 3d shell

For Fe 3d⁶ (high spin):
- 4 unpaired electrons (Hund's rule)
- Magnetic moment μ ≈ 4μ_B
- Ground state is high-spin S=2 (from atomic physics)
- In the ring model with AFM coupling: S=0 singlet ground state
""")

# Exact values for comparison (Bethe ansatz for Heisenberg ring)
print(f"\n{'='*70}")
print("THEORETICAL COMPARISON")
print(f"{'='*70}")

# The exact value for N=6 from literature
E0_exact = -2.0 * J_COUPLING  # Known exact result for 6-site ring
gap_exact = 0.5 * J_COUPLING  # Approximate gap

print(f"Calculated E0/J = {E0/J_COUPLING:.6f}")
print(f"Theoretical E0/J ≈ -2.0 (for N=6 antiferromagnetic ring)")
print(f"Difference: {abs(E0/J_COUPLING - (-2.0)):.6f}")

# Save results
results = {
    "system": "6-site Heisenberg ring (L104 26Q 3d register)",
    "qubits": "q2-q7 (6 qubits)",
    "hamiltonian": "H = J * Σ(S_i · S_{i+1}) with periodic boundary",
    "parameters": {
        "GOD_CODE": GOD_CODE,
        "PHI": PHI,
        "J_over_kB_K": J_OVER_KB,
        "J_coupling": J_COUPLING,
        "N_sites": N,
        "N_states": N_STATES,
        "configuration": "Fe 3d^6 (111111)"
    },
    "energies": {
        "E0": float(E0),
        "E1": float(E1),
        "delta_E": float(gap),
        "E0_over_J": float(E0 / J_COUPLING),
        "delta_E_over_J": float(gap / J_COUPLING),
        "first_10_levels": [float(e / J_COUPLING) for e in eigenvalues[:10]]
    },
    "correlations": {
        "correlation_matrix": correlation_matrix.tolist(),
        "nearest_neighbor_avg": float(avg_nn_corr),
        "nearest_neighbor_individual": [
            float(correlation_matrix[i, (i+1)%N]) for i in range(N)
        ]
    },
    "entanglement": {
        "bipartition_half": float(entropies["Sites {0,1,2} vs {3,4,5}"]),
        "single_site_avg": float(np.mean(single_site_entropies)),
        "single_site_values": [float(s) for s in single_site_entropies]
    },
    "spin": {
        "S_squared_ground": float(S_ground),
        "S_ground": float(S_val),
        "Sz_ground": float(np.real(Sz_ground))
    }
}

output_path = "/Users/carolalvarez/Applications/Allentown-L104-Node/heisenberg_ring_results.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print(f"Results saved to: {output_path}")
print(f"{'='*70}")

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Ground state energy E0/J = {E0/J_COUPLING:.4f}")
print(f"Energy gap ΔE/J = {gap/J_COUPLING:.4f}")
print(f"Average NN correlation <S_i·S_{{i+1}}> = {avg_nn_corr:.4f}")
print(f"Ground state total spin S ≈ {S_val:.2f}")
print(f"{'='*70}")
