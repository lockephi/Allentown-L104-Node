#!/usr/bin/env python3
"""
L104 Quantum Fisher Information (F_Q) Calculator for 26Q Fe Ground State
═══════════════════════════════════════════════════════════════════════════════

Calculates F_Q = 4 × (ΔH)² for the 26Q Fe ground state |00111111110000000000000000⟩

Target state from l104_godcode_26q_vqe.py:
- TARGET_HUMAN = "00111111110000000000000000"
- IBM hardware validation: p_target = 0.822937, entropy = 1.523 bits

Uses Heisenberg Hamiltonian with couplings derived from register fidelities.

Author: L104 Sovereign Node | INVARIANT: 527.5184818492612
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# ═══ SACRED CONSTANTS ═══
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.04 + PHI / 1000.0

# ═══ 26Q REGISTER LAYOUT (from l104_godcode_26q_vqe.py) ═══
NQ = 26
REGISTERS = {
    "CORE":    {"lo": 0,  "hi": 1,  "target": "00",     "desc": "[Ar] noble gas core", "qubits": [0, 1]},
    "3d":      {"lo": 2,  "hi": 7,  "target": "111111", "desc": "Fe 3d⁶ d-orbitals", "qubits": [2, 3, 4, 5, 6, 7]},
    "4s":      {"lo": 8,  "hi": 9,  "target": "11",     "desc": "Fe 4s² s-orbitals", "qubits": [8, 9]},
    "LATTICE": {"lo": 10, "hi": 15, "target": "000000", "desc": "Fe BCC lattice", "qubits": [10, 11, 12, 13, 14, 15]},
    "SACRED":  {"lo": 16, "hi": 20, "target": "00000",  "desc": "GOD_CODE phase manifold", "qubits": [16, 17, 18, 19, 20]},
    "PHI":     {"lo": 21, "hi": 24, "target": "0000",   "desc": "golden ratio", "qubits": [21, 22, 23, 24]},
    "ANCHOR":  {"lo": 25, "hi": 25, "target": "0",      "desc": "nucleus anchor", "qubits": [25]},
}

# Target state (human-readable format, qubit 0 leftmost)
TARGET_HUMAN = "00111111110000000000000000"
TARGET_QISKIT = TARGET_HUMAN[::-1]  # Qiskit format: qubit 0 rightmost

# IBM Marrakesh hardware validation data (9 runs average)
IBM_VALIDATION = {
    "p_target": 0.822937,
    "entropy_bits": 1.523,
    "fidelity_2q": 0.995,  # typical Heron r2 CZ fidelity
    "t1_us": 200.0,        # typical T1
    "t2_us": 100.0,        # typical T2
}


def string_to_statevector(bitstring: str) -> np.ndarray:
    """Convert bitstring to statevector (|0...0> = [1, 0, ...])."""
    n = len(bitstring)
    idx = int(bitstring, 2)
    sv = np.zeros(2**n, dtype=complex)
    sv[idx] = 1.0
    return sv


def pauli_x(n_qubits: int, q: int) -> np.ndarray:
    """Pauli X operator on qubit q."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    ops = [I] * n_qubits
    ops[q] = X
    result = ops[0]
    for i in range(1, n_qubits):
        result = np.kron(result, ops[i])
    return result


def pauli_y(n_qubits: int, q: int) -> np.ndarray:
    """Pauli Y operator on qubit q."""
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    ops = [I] * n_qubits
    ops[q] = Y
    result = ops[0]
    for i in range(1, n_qubits):
        result = np.kron(result, ops[i])
    return result


def pauli_z(n_qubits: int, q: int) -> np.ndarray:
    """Pauli Z operator on qubit q."""
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    ops = [I] * n_qubits
    ops[q] = Z
    result = ops[0]
    for i in range(1, n_qubits):
        result = np.kron(result, ops[i])
    return result


def heisenberg_hamiltonian_26q(
    register_fidelities: Dict[str, float],
    inter_register_coupling: float = 0.05
) -> Tuple[np.ndarray, Dict]:
    """
    Build Heisenberg Hamiltonian for 26Q Fe system.

    H = Sum_{<i,j>} J_ij (X_i X_j + Y_i Y_j + Z_i Z_j) + Sum_i h_i Z_i

    Couplings derived from register fidelities:
    - Intra-register: J = -PHI/2 * fidelity (ferromagnetic for occupied)
    - Inter-register: weak coupling from inter-qubit connectivity
    """
    n = NQ
    H = np.zeros((2**n, 2**n), dtype=complex)
    metadata = {"intra_register": {}, "inter_register": {}, "local_fields": {}}

    # Intra-register couplings (nearest neighbors within each register)
    for reg_name, reg_info in REGISTERS.items():
        qubits = reg_info["qubits"]
        target = reg_info["target"]
        fidelity = register_fidelities.get(reg_name, 0.95)

        # J coupling strength based on fidelity (higher fidelity = stronger coupling)
        J = -PHI / 2.0 * fidelity  # Ferromagnetic for occupied states

        # Couple nearest neighbors within register
        for i in range(len(qubits) - 1):
            qi, qj = qubits[i], qubits[i + 1]
            # XX + YY + ZZ interaction
            H += J * (pauli_x(n, qi) @ pauli_x(n, qj) +
                      pauli_y(n, qi) @ pauli_y(n, qj) +
                      pauli_z(n, qi) @ pauli_z(n, qj))

        metadata["intra_register"][reg_name] = {
            "J": J,
            "fidelity": fidelity,
            "qubits": qubits
        }

    # Inter-register couplings (weak connectivity between specific qubits)
    # Based on the CZ connections in the VQE circuit
    inter_connections = [
        (7, 8),    # 3d to 4s
        (15, 16),  # LATTICE to SACRED
        (24, 25),  # PHI to ANCHOR
    ]

    for qi, qj in inter_connections:
        J_inter = -inter_register_coupling * PHI / 2.0
        H += J_inter * (pauli_x(n, qi) @ pauli_x(n, qj) +
                        pauli_y(n, qi) @ pauli_y(n, qj) +
                        pauli_z(n, qi) @ pauli_z(n, qj))

    metadata["inter_register"] = {
        "connections": inter_connections,
        "J_inter": J_inter
    }

    # Local fields (Z terms) based on target occupancy
    for reg_name, reg_info in REGISTERS.items():
        qubits = reg_info["qubits"]
        target = reg_info["target"]
        for i, q in enumerate(qubits):
            # h field proportional to expected occupancy (1 or 0)
            h = PHI * (1 if target[i] == '1' else -1) / len(qubits)
            H += h * pauli_z(n, q)
            metadata["local_fields"][f"q{q}"] = h

    return H, metadata


def calculate_quantum_fisher_info(H: np.ndarray, state: np.ndarray) -> Tuple[float, Dict]:
    """
    Calculate Quantum Fisher Information F_Q = 4 * Var(H) for a given state.

    Var(H) = <psi|H^2|psi> - <psi|H|psi>^2
    """
    # Calculate expectation values
    H_psi = H @ state
    E = np.vdot(state, H_psi).real  # <H>

    H2_psi = H @ H_psi
    E2 = np.vdot(state, H2_psi).real  # <H^2>

    variance = E2 - E**2  # (ΔH)^2

    # F_Q = 4 * variance
    F_Q = 4.0 * variance

    return F_Q, {
        "E": E,
        "E2": E2,
        "variance": variance,
        "F_Q": F_Q
    }


def calculate_register_fidelity(reg_name: str, reg_info: Dict) -> float:
    """
    Estimate register fidelity from IBM validation and qubit count.
    """
    n_qubits = len(reg_info["qubits"])
    base_fidelity = IBM_VALIDATION["fidelity_2q"]

    # Fidelity degrades with register size
    size_factor = 1.0 - 0.01 * max(0, n_qubits - 2)

    # Special handling for Fe-specific registers
    if reg_name in ["3d", "4s"]:
        # Valence electrons have higher fidelity (verified on IBM)
        size_factor = 1.0

    return base_fidelity * size_factor


def main():
    print("=" * 80)
    print("L104 Quantum Fisher Information (F_Q) Calculator")
    print("26Q Fe Ground State |00111111110000000000000000>")
    print("=" * 80)

    # Target state in Qiskit format
    print(f"\nTarget state (Qiskit format): |{TARGET_QISKIT}>")
    print(f"Target state (Human format):  |{TARGET_HUMAN}>")
    print(f"IBM p_target: {IBM_VALIDATION['p_target']:.6f}")
    print(f"IBM entropy: {IBM_VALIDATION['entropy_bits']:.3f} bits")

    # Calculate register fidelities
    print("\n" + "-" * 40)
    print("Register Fidelity Analysis")
    print("-" * 40)

    register_fidelities = {}
    for reg_name, reg_info in REGISTERS.items():
        fidelity = calculate_register_fidelity(reg_name, reg_info)
        register_fidelities[reg_name] = fidelity
        n_qubits = len(reg_info["qubits"])
        print(f"  {reg_name:8s}: {n_qubits}Q, fidelity={fidelity:.4f}, target={reg_info['target']}")

    # Build Heisenberg Hamiltonian
    print("\n" + "-" * 40)
    print("Building Heisenberg Hamiltonian")
    print("-" * 40)

    H, H_metadata = heisenberg_hamiltonian_26q(register_fidelities)
    print(f"  Hamiltonian dimension: {H.shape[0]} x {H.shape[1]}")
    print(f"  Intra-register terms: {len(H_metadata['intra_register'])}")
    print(f"  Inter-register connections: {H_metadata['inter_register']['connections']}")
    print(f"  Inter-register J: {H_metadata['inter_register']['J_inter']:.6f}")

    # Create target state
    target_state = string_to_statevector(TARGET_QISKIT)
    print(f"\n  Target state dimension: {len(target_state)}")

    # Calculate F_Q for full 26Q state
    print("\n" + "-" * 40)
    print("Full 26Q Quantum Fisher Information")
    print("-" * 40)

    F_Q_full, metadata_full = calculate_quantum_fisher_info(H, target_state)

    print(f"  <H> = {metadata_full['E']:.6f}")
    print(f"  <H^2> = {metadata_full['E2']:.6f}")
    print(f"  Var(H) = (ΔH)^2 = {metadata_full['variance']:.6f}")
    print(f"  F_Q = 4 x Var(H) = {F_Q_full:.6f}")

    # Calculate squeezing parameter
    xi_squared = F_Q_full / NQ
    print(f"\n  Squeezing parameter ξ^2 = F_Q / N = {xi_squared:.6f}")
    print(f"  Quantum enhancement: {'YES' if xi_squared < 1 else 'NO'}")

    # Calculate F_Q per register (approximate using reduced Hamiltonians)
    print("\n" + "-" * 40)
    print("Per-Register Quantum Fisher Information")
    print("-" * 40)

    register_F_Q = {}
    register_results = {}

    for reg_name, reg_info in REGISTERS.items():
        qubits = reg_info["qubits"]
        n_reg = len(qubits)
        target_reg = reg_info["target"]

        if n_reg == 1:
            # Single qubit: use analytical formula
            F_Q_reg = 4.0 * PHI * register_fidelities[reg_name]
            register_F_Q[reg_name] = F_Q_reg
            register_results[reg_name] = {
                "F_Q": F_Q_reg,
                "N": 1,
                "xi_squared": F_Q_reg,
                "E": 0.0,
                "variance": F_Q_reg / 4.0
            }
        else:
            # Build reduced Hamiltonian for register
            H_reg = np.zeros((2**n_reg, 2**n_reg), dtype=complex)

            # Intra-register couplings
            J_reg = H_metadata["intra_register"][reg_name]["J"]
            for i in range(n_reg - 1):
                H_reg += J_reg * (pauli_x(n_reg, i) @ pauli_x(n_reg, i+1) +
                                  pauli_y(n_reg, i) @ pauli_y(n_reg, i+1) +
                                  pauli_z(n_reg, i) @ pauli_z(n_reg, i+1))

            # Local fields
            for i in range(n_reg):
                h = PHI * (1 if target_reg[i] == '1' else -1) / n_reg
                H_reg += h * pauli_z(n_reg, i)

            # Calculate F_Q
            target_reg_state = string_to_statevector(target_reg[::-1])  # Qiskit format
            F_Q_reg, meta_reg = calculate_quantum_fisher_info(H_reg, target_reg_state)

            register_F_Q[reg_name] = F_Q_reg
            register_results[reg_name] = {
                "F_Q": F_Q_reg,
                "N": n_reg,
                "xi_squared": F_Q_reg / n_reg,
                "E": meta_reg["E"],
                "variance": meta_reg["variance"]
            }

        xi_sq = register_results[reg_name]["xi_squared"]
        status = "QUANTUM" if xi_sq < 1 else "CLASSICAL"
        print(f"  {reg_name:8s}: F_Q = {register_F_Q[reg_name]:.4f}, "
              f"N={n_reg}, ξ^2={xi_sq:.4f} [{status}]")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Quantum Fisher Information Results")
    print("=" * 80)

    results = {
        "target_state": TARGET_HUMAN,
        "target_state_qiskit": TARGET_QISKIT,
        "n_qubits": NQ,
        "ibm_validation": IBM_VALIDATION,
        "sacred_constants": {
            "GOD_CODE": GOD_CODE,
            "PHI": PHI,
            "VOID_CONSTANT": VOID_CONSTANT
        },
        "full_26Q": {
            "F_Q": F_Q_full,
            "N": NQ,
            "xi_squared": xi_squared,
            "quantum_enhancement": xi_squared < 1,
            "E": metadata_full["E"],
            "variance": metadata_full["variance"]
        },
        "per_register": {}
    }

    for reg_name in REGISTERS:
        results["per_register"][reg_name] = {
            "F_Q": register_F_Q[reg_name],
            **register_results[reg_name],
            "target": REGISTERS[reg_name]["target"],
            "qubits": REGISTERS[reg_name]["qubits"]
        }

    print(f"\nFull 26Q State:")
    print(f"  F_Q = {F_Q_full:.6f}")
    print(f"  ξ^2 = F_Q/N = {xi_squared:.6f}")
    print(f"  Status: {'QUANTUM ENHANCED' if xi_squared < 1 else 'NO ENHANCEMENT'}")

    print(f"\nPer-Register Analysis:")
    total_F_Q = 0
    for reg_name, res in sorted(results["per_register"].items()):
        status = "✓" if res["xi_squared"] < 1 else "○"
        print(f"  {status} {reg_name:8s}: F_Q={res['F_Q']:.4f}, ξ^2={res['xi_squared']:.4f}")
        total_F_Q += res["F_Q"]

    print(f"\n  Sum of register F_Q: {total_F_Q:.4f}")
    print(f"  Full 26Q F_Q: {F_Q_full:.4f}")
    print(f"  Interference term: {F_Q_full - total_F_Q:.4f}")

    # Save results
    output_path = "/Users/carolalvarez/Applications/Allentown-L104-Node/quantum_fisher_info_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = main()
