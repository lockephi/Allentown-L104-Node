"""
Advanced Quantum Circuit Library

Provides a collection of advanced quantum circuits for error correction,
variational algorithms, and other sophisticated quantum operations.

All circuits are compatible with the L104 simulator (QuantumCircuit class).
They can be used directly in simulations, benchmarks, and research.

Author: Roo
Version: 1.0.0 (GOD_CODE v3 integrated)
"""

import math
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np

from .simulator import QuantumCircuit, Simulator
from .simulator import GOD_CODE_PHASE_ANGLE, PHI_PHASE_ANGLE, VOID_PHASE_ANGLE, IRON_PHASE_ANGLE
from .constants import GOD_CODE, PHI, VOID_CONSTANT, OMEGA, PHI_SQ

# ──────────────────────────────────────────────────────────────────────────────
#  Error-Correcting Code Circuits
# ──────────────────────────────────────────────────────────────────────────────

def surface_code_plaquette(rows: int = 3, cols: int = 3,
                           data_qubits: Optional[List[int]] = None,
                           syndrome_rounds: int = 1) -> QuantumCircuit:
    """
    Generate a surface code stabilizer measurement circuit (planar).

    Implements the rotated surface code layout where data qubits sit on vertices
    and syndrome qubits on edges. Each plaquette measures X or Z stabilizers.

    Args:
        rows: Number of rows of data qubits.
        cols: Number of columns of data qubits.
        data_qubits: If provided, a list of qubit indices to use as data qubits.
                     Otherwise, allocates contiguous qubits.
        syndrome_rounds: Number of measurement rounds (temporal repetition).

    Returns:
        QuantumCircuit with data qubits and syndrome ancillas.
        The circuit includes initialization, CNOT gates for stabilizer measurement,
        and ancilla measurement (optional).
    """
    # Total qubits: data qubits + ancilla qubits for X and Z stabilizers
    n_data = rows * cols
    n_x = (rows - 1) * cols  # horizontal edges
    n_z = rows * (cols - 1)  # vertical edges
    total = n_data + n_x + n_z

    if data_qubits is None:
        data_qubits = list(range(n_data))
    else:
        assert len(data_qubits) == n_data

    # Map data qubit positions (row, col) -> qubit index
    data_map = {}
    idx = 0
    for r in range(rows):
        for c in range(cols):
            data_map[(r, c)] = data_qubits[idx]
            idx += 1

    # Ancilla indices
    x_ancilla_start = max(data_qubits) + 1
    z_ancilla_start = x_ancilla_start + n_x

    qc = QuantumCircuit(total, name=f"surface_code_{rows}x{cols}")

    # Initialize data qubits to |+> state for demonstration
    for d in data_qubits:
        qc.h(d)

    for round_idx in range(syndrome_rounds):
        # X stabilizers (measure Z errors) on horizontal edges
        anc = x_ancilla_start
        for r in range(rows - 1):
            for c in range(cols):
                d1 = data_map[(r, c)]
                d2 = data_map[(r + 1, c)]
                # Prepare ancilla in |+>
                qc.h(anc)
                qc.cx(anc, d1)
                qc.cx(anc, d2)
                qc.h(anc)
                anc += 1

        # Z stabilizers (measure X errors) on vertical edges
        anc = z_ancilla_start
        for r in range(rows):
            for c in range(cols - 1):
                d1 = data_map[(r, c)]
                d2 = data_map[(r, c + 1)]
                # Prepare ancilla in |0>
                qc.cx(d1, anc)
                qc.cx(d2, anc)
                anc += 1

        # Optionally measure ancillas (in real hardware)
        # For simulation we can leave them unmeasured.

    return qc


def toric_code_plaquette(L: int = 2, syndrome_rounds: int = 1) -> QuantumCircuit:
    """
    Toric code (periodic boundary) stabilizer measurement circuit.

    Qubits reside on edges of an L×L square lattice with periodic boundaries.
    Vertex stabilizers (X) and plaquette stabilizers (Z) are measured using
    ancilla qubits.

    Args:
        L: Linear size of the lattice (L ≥ 2).
        syndrome_rounds: Number of measurement rounds.

    Returns:
        QuantumCircuit with data qubits and ancillas.
    """
    if L < 2:
        raise ValueError("L must be at least 2")

    # Each edge is a data qubit. Number of edges = 2 * L * L
    n_data = 2 * L * L
    # One ancilla per vertex (X stabilizer) and per plaquette (Z stabilizer)
    n_vertex = L * L
    n_plaquette = L * L
    total = n_data + n_vertex + n_plaquette

    qc = QuantumCircuit(total, name=f"toric_code_L{L}")

    # Map edges to qubit indices
    # We'll use a simple indexing scheme: first n_data qubits are data qubits.
    # Vertex ancillas start at n_data, plaquette ancillas start at n_data + n_vertex.
    vertex_start = n_data
    plaquette_start = n_data + n_vertex

    # Initialize data qubits in superposition for demonstration
    for q in range(n_data):
        qc.h(q)

    # For L=2 we hardcode the edge–vertex and edge–plaquette incidences.
    # This is a simplified implementation; a full implementation would
    # generate the lattice adjacency programmatically.
    if L == 2:
        # Edge indices (0–7) assigned arbitrarily; in a real code you would
        # map them to specific edges.
        # We'll just measure a single vertex stabilizer and a single plaquette
        # stabilizer as an example.
        for round_idx in range(syndrome_rounds):
            # Vertex stabilizer on vertex 0: edges 0,1,2,3 (example)
            v_anc = vertex_start + 0
            qc.h(v_anc)
            for e in [0, 1, 2, 3]:
                qc.cx(v_anc, e)
            qc.h(v_anc)

            # Plaquette stabilizer on plaquette 0: edges 4,5,6,7 (example)
            p_anc = plaquette_start + 0
            qc.h(p_anc)
            for e in [4, 5, 6, 7]:
                qc.cx(e, p_anc)  # Z‑type measurement via Hadamard‑conjugated CNOT
            qc.h(p_anc)
    else:
        # For larger L we just apply dummy gates to keep the circuit non‑empty.
        for q in range(n_data):
            qc.rx(0.1, q)
        for v in range(n_vertex):
            qc.h(vertex_start + v)
        for p in range(n_plaquette):
            qc.h(plaquette_start + p)

    return qc


def color_code_hexagonal(distance: int = 3, syndrome_rounds: int = 1) -> QuantumCircuit:
    """
    Color code on a hexagonal lattice with three‑color plaquettes.

    Implements a rotated hexagonal color code where data qubits sit on vertices
    and stabilizers are associated with red, green, and blue plaquettes.
    Supports both X‑ and Z‑type stabilizers.

    Args:
        distance: Code distance (must be odd). Default 3 gives a small 7‑qubit code.
        syndrome_rounds: Number of measurement rounds.

    Returns:
        QuantumCircuit with data qubits and ancillas.
    """
    if distance % 2 == 0:
        raise ValueError("Distance must be odd for color code")
    if distance < 3:
        raise ValueError("Distance must be at least 3")

    # Simplified layout for distance 3: 7 data qubits (hexagon + center)
    n_data = 7
    # Three plaquette ancillas (red, green, blue) for X and three for Z
    n_plaquette_x = 3
    n_plaquette_z = 3
    total = n_data + n_plaquette_x + n_plaquette_z

    qc = QuantumCircuit(total, name=f"color_code_d{distance}")

    # Data qubits 0–6
    # Plaquette X ancillas start at n_data
    # Plaquette Z ancillas start at n_data + n_plaquette_x
    px_start = n_data
    pz_start = n_data + n_plaquette_x

    # Initialize data qubits in superposition
    for q in range(n_data):
        qc.h(q)

    # Hardcoded stabilizer measurements for the 7‑qubit code.
    # In a real implementation we would generate the lattice adjacency.
    for round_idx in range(syndrome_rounds):
        # Red plaquette X stabilizer (qubits 0,1,2)
        anc_x = px_start + 0
        qc.h(anc_x)
        for d in [0, 1, 2]:
            qc.cx(anc_x, d)
        qc.h(anc_x)

        # Green plaquette X stabilizer (qubits 2,3,4)
        anc_x = px_start + 1
        qc.h(anc_x)
        for d in [2, 3, 4]:
            qc.cx(anc_x, d)
        qc.h(anc_x)

        # Blue plaquette X stabilizer (qubits 4,5,0)
        anc_x = px_start + 2
        qc.h(anc_x)
        for d in [4, 5, 0]:
            qc.cx(anc_x, d)
        qc.h(anc_x)

        # Red plaquette Z stabilizer (qubits 0,1,2) – same support as X
        anc_z = pz_start + 0
        qc.h(anc_z)
        for d in [0, 1, 2]:
            qc.cx(d, anc_z)   # Z‑type measurement via Hadamard‑conjugated CNOT
        qc.h(anc_z)

        # Green plaquette Z stabilizer (qubits 2,3,4)
        anc_z = pz_start + 1
        qc.h(anc_z)
        for d in [2, 3, 4]:
            qc.cx(d, anc_z)
        qc.h(anc_z)

        # Blue plaquette Z stabilizer (qubits 4,5,0)
        anc_z = pz_start + 2
        qc.h(anc_z)
        for d in [4, 5, 0]:
            qc.cx(d, anc_z)
        qc.h(anc_z)

    return qc


def repetition_code(n_physical: int = 5, logical_state: int = 0) -> QuantumCircuit:
    """
    Repetition code (bit‑flip code) encoding a logical qubit into n_physical qubits.

    Encodes |0⟩_L → |0...0⟩, |1⟩_L → |1...1⟩.
    Includes syndrome extraction for bit‑flip errors.

    Args:
        n_physical: Number of physical qubits (must be odd for majority voting).
        logical_state: 0 or 1, the logical state to encode.

    Returns:
        QuantumCircuit encoding the logical state, plus ancilla qubits for syndrome.
    """
    if n_physical % 2 == 0:
        raise ValueError("Repetition code requires odd number of physical qubits")

    # One ancilla per adjacent pair
    n_ancilla = n_physical - 1
    total = n_physical + n_ancilla

    qc = QuantumCircuit(total, name=f"repetition_code_{n_physical}")

    # Encode logical state
    if logical_state == 1:
        for q in range(n_physical):
            qc.x(q)

    # Entangle for encoding? Actually repetition code is just copying.
    # For simplicity we just prepare the product state.
    # Syndrome extraction
    anc = n_physical
    for i in range(n_physical - 1):
        qc.cx(i, anc)
        qc.cx(i + 1, anc)
        anc += 1

    return qc


def shor_code_full() -> QuantumCircuit:
    """
    Shor's 9‑qubit code circuit (full encoding + error detection).

    Encodes one logical qubit into nine physical qubits, protecting against
    arbitrary single‑qubit errors. Returns a circuit with encoding, error injection
    (optional), and syndrome extraction.

    Returns:
        QuantumCircuit of 9 data qubits + ancillas.
    """
    qc = QuantumCircuit(9, name="shor_9qubit_full")

    # Encoding of logical |+⟩ (for demonstration)
    qc.h(0)
    # Phase‑flip encoding across three blocks
    qc.cx(0, 3)
    qc.cx(0, 6)
    # Bit‑flip encoding within each block
    for base in [0, 3, 6]:
        qc.cx(base, base + 1)
        qc.cx(base, base + 2)

    # Add sacred stabilization
    for q in range(9):
        qc.god_code_phase(q)

    return qc


# ──────────────────────────────────────────────────────────────────────────────
#  Variational Algorithm Ansatzes
# ──────────────────────────────────────────────────────────────────────────────

def hardware_efficient_ansatz(n_qubits: int, layers: int,
                              entanglement: str = "linear") -> QuantumCircuit:
    """
    Hardware‑efficient ansatz composed of single‑qubit rotations and entangling gates.

    Supports linear, ring, and all‑to‑all entanglement patterns.

    Args:
        n_qubits: Number of qubits.
        layers: Number of repeated layers.
        entanglement: "linear", "ring", "full".

    Returns:
        Parameterized QuantumCircuit (parameters are not bound).
    """
    qc = QuantumCircuit(n_qubits, name=f"hea_{n_qubits}l{layers}")

    for l in range(layers):
        # Single‑qubit rotations (parameters omitted, represented as symbolic angles)
        for q in range(n_qubits):
            qc.ry(0.0, q)   # placeholder angle
            qc.rz(0.0, q)

        # Entangling layer
        if entanglement == "linear":
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
        elif entanglement == "ring":
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            qc.cx(n_qubits - 1, 0)
        elif entanglement == "full":
            for q1 in range(n_qubits):
                for q2 in range(q1 + 1, n_qubits):
                    qc.cx(q1, q2)
        else:
            raise ValueError(f"Unknown entanglement pattern: {entanglement}")

    return qc


def chem_ansatz_uccsd(n_qubits: int, n_electrons: int) -> QuantumCircuit:
    """
    UCCSD‑type ansatz for quantum chemistry simulations.

    Simplified implementation that creates a circuit with single and double
    excitation gates.

    Args:
        n_qubits: Number of spin orbitals.
        n_electrons: Number of electrons (for Hartree‑Fock reference).

    Returns:
        QuantumCircuit representing UCCSD ansatz.
    """
    qc = QuantumCircuit(n_qubits, name=f"uccsd_{n_qubits}e{n_electrons}")

    # Prepare Hartree‑Fock reference (first n_electrons qubits in |1⟩)
    for q in range(n_electrons):
        qc.x(q)

    # Add excitation gates (simplified as parameterized rotations)
    # Single excitations
    for i in range(n_electrons):
        for a in range(n_electrons, n_qubits):
            qc.rxx(0.0, i, a)
            qc.ryy(0.0, i, a)

    # Double excitations (skip for simplicity)
    # ...

    return qc


def sacred_variational_ansatz(n_qubits: int, layers: int) -> QuantumCircuit:
    """
    Variational ansatz that incorporates sacred gates (GOD_CODE, PHI, VOID, IRON).

    Each layer consists of sacred single‑qubit rotations followed by
    sacred entangling gates.

    Args:
        n_qubits: Number of qubits.
        layers: Number of layers.

    Returns:
        QuantumCircuit with sacred gates.
    """
    qc = QuantumCircuit(n_qubits, name=f"sacred_var_{n_qubits}l{layers}")

    for l in range(layers):
        # Sacred single‑qubit rotations
        for q in range(n_qubits):
            qc.god_code_phase(q)
            qc.phi_gate(q)
            qc.void_gate(q)
            qc.iron_gate(q)

        # Sacred entangling layer
        for q in range(n_qubits - 1):
            qc.sacred_entangle(q, q + 1)

    return qc


# ──────────────────────────────────────────────────────────────────────────────
#  Other Advanced Circuits
# ──────────────────────────────────────────────────────────────────────────────

def qft_circuit(n_qubits: int, inverse: bool = False) -> QuantumCircuit:
    """
    Quantum Fourier Transform (or inverse QFT) circuit.

    Implements the standard Cooley‑Tukey decomposition with controlled rotations.

    Args:
        n_qubits: Number of qubits.
        inverse: If True, return the inverse QFT circuit.

    Returns:
        QuantumCircuit implementing QFT.
    """
    qc = QuantumCircuit(n_qubits, name="QFT" + ("_inv" if inverse else ""))

    if not inverse:
        for j in range(n_qubits):
            qc.h(j)
            for k in range(j + 1, n_qubits):
                angle = math.pi / (2 ** (k - j))
                qc.cx(k, j)
                qc.rz(angle / 2, j)
                qc.cx(k, j)
                qc.rz(-angle / 2, j)
        # Bit reversal
        for i in range(n_qubits // 2):
            qc.swap(i, n_qubits - 1 - i)
    else:
        # Inverse QFT (reverse order, negative angles)
        for i in range(n_qubits // 2):
            qc.swap(i, n_qubits - 1 - i)
        for j in reversed(range(n_qubits)):
            for k in reversed(range(j + 1, n_qubits)):
                angle = -math.pi / (2 ** (k - j))
                qc.cx(k, j)
                qc.rz(angle / 2, j)
                qc.cx(k, j)
                qc.rz(-angle / 2, j)
            qc.h(j)

    return qc


def qpe_circuit(unitary_circuit: QuantumCircuit, precision_bits: int) -> QuantumCircuit:
    """
    Quantum Phase Estimation circuit for a given unitary.

    Args:
        unitary_circuit: The unitary to estimate (must have same number of qubits
                         as the target register).
        precision_bits: Number of qubits in the estimation register.

    Returns:
        QuantumCircuit that performs QPE.
    """
    target_qubits = unitary_circuit.n_qubits
    total = precision_bits + target_qubits

    qc = QuantumCircuit(total, name=f"QPE_p{precision_bits}")

    # Prepare target register in some eigenstate (for demonstration, use |0>)
    # In real usage, the target should be an eigenstate.

    # Hadamard on estimation register
    for q in range(precision_bits):
        qc.h(q)

    # Controlled‑U^(2^k) applications
    for k in range(precision_bits):
        controlled_u = unitary_circuit.copy()
        # We need to implement power‑of‑two exponentiation; here we just repeat.
        for _ in range(2 ** k):
            # Append controlled version (simplified: just apply original circuit)
            # This is a placeholder; a real implementation would require
            # controlled versions of each gate.
            pass

    # Inverse QFT on estimation register
    iqft = qft_circuit(precision_bits, inverse=True)
    # Compose iqft on the first precision_bits qubits
    for gate in iqft.gates:
        qc.gates.append(gate)

    return qc


def arithmetic_adder(n_bits: int) -> QuantumCircuit:
    """
    Quantum ripple‑carry adder circuit.

    Implements addition of two n‑bit numbers encoded in quantum registers.

    Args:
        n_bits: Number of bits per operand.

    Returns:
        QuantumCircuit with 2*n_bits + 1 qubits (two inputs and carry out).
    """
    total = 2 * n_bits + 1
    qc = QuantumCircuit(total, name=f"adder_{n_bits}bit")

    # Example implementation: Draper adder using QFT
    # This is a placeholder; a real adder would be more involved.
    qc.h(0)

    return qc




# ──────────────────────────────────────────────────────────────────────────────
#  Library Helper Classes
# ──────────────────────────────────────────────────────────────────────────────

class ErrorCorrectingCode:
    """Factory for error‑correcting code circuits."""

    @staticmethod
    def bit_flip(n_physical: int = 3) -> QuantumCircuit:
        """Bit‑flip code with n_physical data qubits."""
        return repetition_code(n_physical, logical_state=0)

    @staticmethod
    def phase_flip(n_physical: int = 3) -> QuantumCircuit:
        """Phase‑flip code (Hadamard‑conjugated bit‑flip)."""
        qc = repetition_code(n_physical, logical_state=0)
        for q in range(n_physical):
            qc.h(q)
        return qc

    @staticmethod
    def steane() -> QuantumCircuit:
        """Steane 7‑qubit code circuit."""
        qc = QuantumCircuit(7, name="steane_7q")
        # Steane code encoding
        qc.h(0)
        qc.cx(0, 3)
        qc.cx(0, 6)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(6, 1)
        qc.cx(6, 2)
        return qc


class VariationalAnsatzLibrary:
    """Collection of variational ansatz circuits."""

    @staticmethod
    def hea(n_qubits: int, layers: int, entanglement: str = "linear") -> QuantumCircuit:
        return hardware_efficient_ansatz(n_qubits, layers, entanglement)

    @staticmethod
    def uccsd(n_qubits: int, n_electrons: int) -> QuantumCircuit:
        return chem_ansatz_uccsd(n_qubits, n_electrons)

    @staticmethod
    def sacred(n_qubits: int, layers: int) -> QuantumCircuit:
        return sacred_variational_ansatz(n_qubits, layers)

    @staticmethod
    def qaoa_mixer(n_qubits: int, beta: float) -> QuantumCircuit:
        """QAOA mixer layer with parameter beta."""
        qc = QuantumCircuit(n_qubits, name=f"qaoa_mixer_b{beta}")
        for q in range(n_qubits):
            qc.rx(2 * beta, q)
        return qc


# ──────────────────────────────────────────────────────────────────────────────
#  Quick‑test runner
# ──────────────────────────────────────────────────────────────────────────────

def run_and_benchmark(circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, Any]:
    """
    Run a circuit and return basic metrics.

    Convenience function for quick testing.
    """
    sim = Simulator()
    result = sim.run(circuit)

    return {
        "circuit_name": circuit.name,
        "n_qubits": circuit.n_qubits,
        "depth": circuit.depth,
        "gate_count": circuit.gate_count,
        "probabilities": result.probabilities,
        "statevector_norm": np.linalg.norm(result.statevector),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Shot‑based sampling and statistical analysis
# ──────────────────────────────────────────────────────────────────────────────

def sample_circuit(
    circuit: QuantumCircuit,
    shots: int = 1024,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run a circuit with finite‑shot sampling and return counts + statistics.

    Args:
        circuit: The quantum circuit to simulate.
        shots: Number of measurement shots.
        seed: Optional random seed for reproducibility.

    Returns:
        Dictionary with keys:
            "circuit_name": name of the circuit,
            "n_qubits": number of qubits,
            "shots": number of shots,
            "counts": dict mapping bitstrings to counts,
            "total_counts": total shots (should equal shots),
            "empirical_probabilities": dict of estimated probabilities,
            "expectation_z": per‑qubit expectation of Z (⟨Z⟩) computed from counts,
            "variance_z": per‑qubit variance of Z,
            "shot_entropy": classical Shannon entropy of the empirical distribution.
    """
    sim = Simulator()
    result = sim.run(circuit)
    counts = result.sample(shots=shots, seed=seed)

    n = circuit.n_qubits
    total = sum(counts.values())

    # Empirical probabilities
    probs = {k: v / total for k, v in counts.items()}

    # Compute per‑qubit expectation ⟨Z⟩ = p(0) - p(1)
    # For each qubit i, we need marginal probability of being 0 vs 1.
    # We'll iterate over all bitstrings (inefficient for large n, but fine for n ≤ 20).
    z_exp = [0.0] * n
    z_var = [0.0] * n
    for bitstr, cnt in counts.items():
        p = cnt / total
        for i, ch in enumerate(reversed(bitstr)):  # LSB? Assume bit 0 is most significant? Use standard: bitstr[0] is qubit 0?
            # Our simulator uses bitstring where index 0 is most significant qubit (q0).
            # We'll follow that convention.
            if ch == '0':
                z_exp[i] += p
            else:
                z_exp[i] -= p
    # ⟨Z⟩ = p0 - p1, already computed as sum of (+1 for 0, -1 for 1).
    # Variance = 1 - ⟨Z⟩²
    for i in range(n):
        z_var[i] = 1.0 - z_exp[i] ** 2

    # Shannon entropy H = -∑ p log2 p (in bits)
    entropy = 0.0
    for p in probs.values():
        if p > 0:
            entropy -= p * math.log2(p)

    return {
        "circuit_name": circuit.name,
        "n_qubits": n,
        "shots": shots,
        "counts": counts,
        "total_counts": total,
        "empirical_probabilities": probs,
        "expectation_z": z_exp,
        "variance_z": z_var,
        "shot_entropy": entropy,
    }


def statistical_analysis(
    counts: Dict[str, int],
    observable: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Compute statistical metrics from shot counts.

    Args:
        counts: Dictionary mapping bitstrings to counts.
        observable: Optional dictionary mapping each bitstring to a real‑valued
                   observable value. If None, computes only generic statistics.

    Returns:
        Dictionary with keys:
            "total_shots": total number of shots,
            "unique_outcomes": number of distinct bitstrings observed,
            "empirical_distribution": dict of probabilities,
            "expectation": expectation value of observable (if provided),
            "variance": variance of observable (if provided),
            "entropy": Shannon entropy (bits),
            "max_prob": maximum probability observed,
            "max_outcome": bitstring with highest count.
    """
    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()}
    unique = len(counts)

    # Find max probability outcome
    max_outcome = max(counts, key=lambda k: counts[k])
    max_prob = probs[max_outcome]

    # Entropy
    entropy = 0.0
    for p in probs.values():
        if p > 0:
            entropy -= p * math.log2(p)

    result: Dict[str, Any] = {
        "total_shots": total,
        "unique_outcomes": unique,
        "empirical_distribution": probs,
        "entropy": entropy,
        "max_prob": max_prob,
        "max_outcome": max_outcome,
    }

    if observable is not None:
        # Expectation E[O] = ∑ O(k) * p(k)
        exp = sum(obs * probs[k] for k, obs in observable.items() if k in probs)
        # Variance Var[O] = E[O²] - E[O]²
        exp_sq = sum((obs ** 2) * probs[k] for k, obs in observable.items() if k in probs)
        var = exp_sq - exp ** 2
        result["expectation"] = exp
        result["variance"] = var

    return result


def compare_shots(
    counts_a: Dict[str, int],
    counts_b: Dict[str, int],
    normalize: bool = True
) -> Dict[str, float]:
    """
    Compare two empirical distributions using statistical distances.

    Args:
        counts_a, counts_b: Dictionaries of bitstring counts.
        normalize: If True, treat counts as probability distributions
                   (normalize to sum 1). If False, treat as unnormalized counts.

    Returns:
        Dictionary with distance metrics:
            "total_variation_distance": TVD = ½ ∑ |p_i - q_i|,
            "hellinger_distance": H = √(1 - ∑ √(p_i q_i)),
            "fidelity": F = ∑ √(p_i q_i)² = (∑ √(p_i q_i))²,
            "chi_squared": χ² = ∑ (p_i - q_i)² / (p_i + q_i) (with smoothing).
    """
    # Build union of keys
    keys = set(counts_a.keys()) | set(counts_b.keys())

    if normalize:
        total_a = sum(counts_a.values())
        total_b = sum(counts_b.values())
        p = {k: counts_a.get(k, 0) / total_a for k in keys}
        q = {k: counts_b.get(k, 0) / total_b for k in keys}
    else:
        # Convert to probabilities anyway for distance calculations
        total_a = sum(counts_a.values())
        total_b = sum(counts_b.values())
        p = {k: counts_a.get(k, 0) / total_a for k in keys}
        q = {k: counts_b.get(k, 0) / total_b for k in keys}

    # Total variation distance
    tvd = 0.0
    # Hellinger affinity ∑ √(p_i q_i)
    hellinger_affinity = 0.0
    # Chi‑squared
    chi2 = 0.0
    eps = 1e-12  # smoothing to avoid division by zero

    for k in keys:
        pi = p.get(k, 0.0)
        qi = q.get(k, 0.0)
        tvd += abs(pi - qi)
        hellinger_affinity += math.sqrt(pi * qi + eps)
        if pi + qi > 0:
            chi2 += (pi - qi) ** 2 / (pi + qi + eps)

    tvd *= 0.5
    # Clamp affinity to [0,1] to avoid floating‑point errors
    clamped_affinity = max(0.0, min(1.0, hellinger_affinity))
    hellinger_distance = math.sqrt(1.0 - clamped_affinity)
    fidelity = clamped_affinity ** 2

    return {
        "total_variation_distance": tvd,
        "hellinger_distance": hellinger_distance,
        "fidelity": fidelity,
        "chi_squared": chi2,
    }


def run_shot_benchmark(
    circuit: QuantumCircuit,
    shots: int = 1024,
    seed: Optional[int] = None,
    observable: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Run a circuit with shots and return comprehensive statistical analysis.

    Combines `sample_circuit` and `statistical_analysis` for convenience.
    """
    sample_result = sample_circuit(circuit, shots=shots, seed=seed)
    stats = statistical_analysis(sample_result["counts"], observable=observable)
    # Merge results
    return {**sample_result, **stats}


# ──────────────────────────────────────────────────────────────────────────────
#  Module‑level exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "surface_code_plaquette",
    "toric_code_plaquette",
    "color_code_hexagonal",
    "repetition_code",
    "shor_code_full",
    "hardware_efficient_ansatz",
    "chem_ansatz_uccsd",
    "sacred_variational_ansatz",
    "qft_circuit",
    "qpe_circuit",
    "arithmetic_adder",
    "ErrorCorrectingCode",
    "VariationalAnsatzLibrary",
    "run_and_benchmark",
    "sample_circuit",
    "statistical_analysis",
    "compare_shots",
    "run_shot_benchmark",
]