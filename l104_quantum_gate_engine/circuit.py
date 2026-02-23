"""
===============================================================================
L104 QUANTUM GATE ENGINE — CIRCUIT REPRESENTATION
===============================================================================

A GateCircuit is an ordered sequence of gate operations on named qubits.
Supports append, compose, tensor, reverse, depth analysis, and Qiskit export.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Sequence
from dataclasses import dataclass, field

from .gates import QuantumGate, GateType, I, H, CNOT, X, Z
from .constants import GOD_CODE, PHI, VOID_CONSTANT


@dataclass
class GateOperation:
    """A single gate applied to specific qubits at a specific moment."""
    gate: QuantumGate
    qubits: Tuple[int, ...]
    classical_bits: Optional[Tuple[int, ...]] = None
    label: Optional[str] = None
    condition: Optional[Tuple[int, int]] = None  # (classical_bit, value) for conditional

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate.name,
            "qubits": list(self.qubits),
            "label": self.label,
            "gate_type": self.gate.gate_type.name,
            "parameters": self.gate.parameters,
        }


class GateCircuit:
    """
    Quantum circuit as an ordered list of gate operations.

    Features:
    - Qubit-indexed gate scheduling
    - Full unitary computation for small circuits
    - Depth and gate-count analysis
    - Qiskit QuantumCircuit export
    - Topological moment assignment for parallelism
    """

    def __init__(self, num_qubits: int, name: str = "circuit"):
        self.num_qubits = num_qubits
        self.name = name
        self.operations: List[GateOperation] = []
        self._num_classical = 0

    @property
    def num_operations(self) -> int:
        return len(self.operations)

    @property
    def depth(self) -> int:
        """Circuit depth (critical path length)."""
        if not self.operations:
            return 0
        # Track the latest moment each qubit is occupied
        qubit_moments = [0] * self.num_qubits
        for op in self.operations:
            # This gate starts after the latest involved qubit is free
            start = max(qubit_moments[q] for q in op.qubits)
            end = start + 1
            for q in op.qubits:
                qubit_moments[q] = end
        return max(qubit_moments)

    @property
    def gate_counts(self) -> Dict[str, int]:
        """Count of each gate type used."""
        counts: Dict[str, int] = {}
        for op in self.operations:
            counts[op.gate.name] = counts.get(op.gate.name, 0) + 1
        return counts

    @property
    def two_qubit_count(self) -> int:
        """Number of two-qubit gates (proxy for circuit cost)."""
        return sum(1 for op in self.operations if op.num_qubits >= 2)

    def append(self, gate: QuantumGate, qubits: Sequence[int],
               label: Optional[str] = None,
               condition: Optional[Tuple[int, int]] = None) -> 'GateCircuit':
        """Append a gate operation to the circuit."""
        qubits_tuple = tuple(qubits)
        if len(qubits_tuple) != gate.num_qubits:
            raise ValueError(
                f"Gate {gate.name} requires {gate.num_qubits} qubits, "
                f"got {len(qubits_tuple)}"
            )
        for q in qubits_tuple:
            if q < 0 or q >= self.num_qubits:
                raise ValueError(f"Qubit {q} out of range [0, {self.num_qubits})")

        self.operations.append(GateOperation(
            gate=gate, qubits=qubits_tuple, label=label, condition=condition,
        ))
        return self  # Allow chaining

    def h(self, qubit: int) -> 'GateCircuit':
        """Shorthand: append Hadamard."""
        return self.append(H, [qubit])

    def x(self, qubit: int) -> 'GateCircuit':
        """Shorthand: append Pauli-X."""
        return self.append(X, [qubit])

    def z(self, qubit: int) -> 'GateCircuit':
        """Shorthand: append Pauli-Z."""
        return self.append(Z, [qubit])

    def cx(self, control: int, target: int) -> 'GateCircuit':
        """Shorthand: append CNOT."""
        return self.append(CNOT, [control, target])

    def barrier(self) -> 'GateCircuit':
        """Insert a barrier (no-op gate for visualization/scheduling)."""
        # Barriers are identity on all qubits simultaneously — tracked as label
        self.operations.append(GateOperation(
            gate=I, qubits=(0,), label="BARRIER",
        ))
        return self

    def compose(self, other: 'GateCircuit', qubit_map: Optional[Dict[int, int]] = None) -> 'GateCircuit':
        """Append another circuit to this one, optionally remapping qubits."""
        for op in other.operations:
            if qubit_map:
                new_qubits = tuple(qubit_map.get(q, q) for q in op.qubits)
            else:
                new_qubits = op.qubits
            self.append(op.gate, new_qubits, label=op.label, condition=op.condition)
        return self

    def inverse(self) -> 'GateCircuit':
        """Return the inverse (dagger) circuit — reversed operations with adjoint gates."""
        inv = GateCircuit(self.num_qubits, name=f"{self.name}†")
        for op in reversed(self.operations):
            if op.label == "BARRIER":
                inv.barrier()
            else:
                inv.append(op.gate.dag, op.qubits)
        return inv

    def tensor(self, other: 'GateCircuit') -> 'GateCircuit':
        """Tensor product: self ⊗ other (wires placed adjacently)."""
        combined = GateCircuit(
            self.num_qubits + other.num_qubits,
            name=f"({self.name}⊗{other.name})"
        )
        for op in self.operations:
            combined.append(op.gate, op.qubits, label=op.label)
        offset = self.num_qubits
        for op in other.operations:
            new_qubits = tuple(q + offset for q in op.qubits)
            combined.append(op.gate, new_qubits, label=op.label)
        return combined

    def unitary(self) -> np.ndarray:
        """
        Compute the full unitary matrix of the circuit.
        Warning: exponential in num_qubits — use only for small circuits (≤12 qubits).
        """
        dim = 2 ** self.num_qubits
        U = np.eye(dim, dtype=complex)

        for op in self.operations:
            if op.label == "BARRIER":
                continue
            # Embed gate into full Hilbert space
            gate_full = self._embed_gate(op.gate.matrix, op.qubits)
            U = gate_full @ U

        return U

    def _embed_gate(self, gate_matrix: np.ndarray, qubits: Tuple[int, ...]) -> np.ndarray:
        """Embed a small gate matrix into the full 2^n dimensional space."""
        n = self.num_qubits
        dim = 2 ** n
        gate_dim = gate_matrix.shape[0]
        num_gate_qubits = len(qubits)

        full = np.zeros((dim, dim), dtype=complex)

        for i in range(dim):
            for j in range(dim):
                # Extract the bits corresponding to gate qubits
                # Convention: qubits[0] → MSB of gate index (big-endian)
                # This ensures cx(control, target) maps correctly:
                #   CNOT matrix has control=MSB, target=LSB
                i_gate = 0
                j_gate = 0
                match = True
                for idx, q in enumerate(qubits):
                    bit_pos = num_gate_qubits - 1 - idx
                    i_gate |= ((i >> q) & 1) << bit_pos
                    j_gate |= ((j >> q) & 1) << bit_pos

                # Non-gate qubits must match
                i_rest = i
                j_rest = j
                for q in qubits:
                    i_rest &= ~(1 << q)
                    j_rest &= ~(1 << q)
                if i_rest != j_rest:
                    continue

                full[i, j] = gate_matrix[i_gate, j_gate]

        return full

    def moment_schedule(self) -> List[List[GateOperation]]:
        """
        Schedule operations into parallel moments (layers).
        Operations in the same moment touch disjoint qubits.
        """
        moments: List[List[GateOperation]] = []
        qubit_available = [0] * self.num_qubits

        for op in self.operations:
            if op.label == "BARRIER":
                # Barrier forces all qubits to sync
                current_max = max(qubit_available) if qubit_available else 0
                qubit_available = [current_max] * self.num_qubits
                continue

            earliest = max(qubit_available[q] for q in op.qubits)
            while len(moments) <= earliest:
                moments.append([])
            moments[earliest].append(op)
            for q in op.qubits:
                qubit_available[q] = earliest + 1

        return moments

    def to_qiskit(self) -> Any:
        """
        Export to Qiskit QuantumCircuit.
        Requires qiskit to be installed.
        """
        try:
            from qiskit import QuantumCircuit as QC
        except ImportError:
            raise RuntimeError("Qiskit not installed — cannot export circuit")

        qc = QC(self.num_qubits, name=self.name)

        for op in self.operations:
            if op.label == "BARRIER":
                qc.barrier()
                continue

            g = op.gate
            qubits = list(op.qubits)

            # Map known gates to Qiskit native gates
            name_lower = g.name.lower()
            if g.name == "H":
                qc.h(qubits[0])
            elif g.name == "X":
                qc.x(qubits[0])
            elif g.name == "Y":
                qc.y(qubits[0])
            elif g.name == "Z":
                qc.z(qubits[0])
            elif g.name == "S":
                qc.s(qubits[0])
            elif g.name == "S†":
                qc.sdg(qubits[0])
            elif g.name == "T":
                qc.t(qubits[0])
            elif g.name == "T†":
                qc.tdg(qubits[0])
            elif g.name == "SX":
                qc.sx(qubits[0])
            elif g.name == "CNOT":
                qc.cx(qubits[0], qubits[1])
            elif g.name == "CZ":
                qc.cz(qubits[0], qubits[1])
            elif g.name == "SWAP":
                qc.swap(qubits[0], qubits[1])
            elif g.name == "Toffoli":
                qc.ccx(qubits[0], qubits[1], qubits[2])
            elif g.name == "Fredkin":
                qc.cswap(qubits[0], qubits[1], qubits[2])
            elif name_lower.startswith("rx("):
                qc.rx(g.parameters["theta"], qubits[0])
            elif name_lower.startswith("ry("):
                qc.ry(g.parameters["theta"], qubits[0])
            elif name_lower.startswith("rz("):
                qc.rz(g.parameters["theta"], qubits[0])
            elif name_lower.startswith("p("):
                qc.p(g.parameters["theta"], qubits[0])
            elif name_lower.startswith("u3("):
                qc.u(g.parameters["theta"], g.parameters["phi"],
                     g.parameters["lambda"], qubits[0])
            else:
                # Fallback: use unitary gate
                from qiskit.quantum_info import Operator
                qc.unitary(Operator(g.matrix), qubits, label=g.name)

        return qc

    def statistics(self) -> Dict[str, Any]:
        """Comprehensive circuit statistics."""
        moments = self.moment_schedule()
        gate_types = {}
        for op in self.operations:
            if op.label == "BARRIER":
                continue
            t = op.gate.gate_type.name
            gate_types[t] = gate_types.get(t, 0) + 1

        sacred_count = sum(1 for op in self.operations
                          if op.gate.gate_type == GateType.SACRED)
        topo_count = sum(1 for op in self.operations
                        if op.gate.gate_type == GateType.TOPOLOGICAL)

        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "num_operations": self.num_operations,
            "depth": self.depth,
            "two_qubit_count": self.two_qubit_count,
            "gate_counts": self.gate_counts,
            "gate_type_distribution": gate_types,
            "sacred_gate_count": sacred_count,
            "topological_gate_count": topo_count,
            "parallelism": self.num_operations / max(1, self.depth),
            "num_moments": len(moments),
            "god_code_aligned": sacred_count > 0,
        }

    def __repr__(self) -> str:
        return (f"GateCircuit('{self.name}', qubits={self.num_qubits}, "
                f"ops={self.num_operations}, depth={self.depth})")

    def __len__(self) -> int:
        return self.num_operations
