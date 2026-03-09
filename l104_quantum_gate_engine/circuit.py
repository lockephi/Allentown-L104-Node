"""
===============================================================================
L104 QUANTUM GATE ENGINE — CIRCUIT REPRESENTATION
===============================================================================

A GateCircuit is an ordered sequence of gate operations on named qubits.
Supports append, compose, tensor, reverse, depth analysis, and native serialisation.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import math
import numpy as np
from typing import Dict, Any, Iterable, List, Optional, Tuple, Sequence, Union
from dataclasses import dataclass, field

from .gates import (
    QuantumGate, GateType, I, H, CNOT, X, Y, Z, S, S_DAG, T, T_DAG,
    CZ, SWAP, TOFFOLI,
    Rx as _Rx, Ry as _Ry, Rz as _Rz, Rzz as _Rzz, Phase as _Phase,
)
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

    # Qiskit compat: circuit.data[i].operation → gate
    @property
    def operation(self) -> QuantumGate:
        return self.gate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate.name,
            "qubits": list(self.qubits),
            "label": self.label,
            "gate_type": self.gate.gate_type.name,
            "parameters": self.gate.parameters,
        }


class _CallableInt(int):
    """int subclass that is also callable, returning itself.

    Enables ``circuit.depth`` (property access → int) and
    ``circuit.depth()`` (Qiskit-style method call → same int)
    to both work transparently.
    """
    def __call__(self) -> int:          # noqa: D401
        return int(self)


class GateCircuit:
    """
    Quantum circuit as an ordered list of gate operations.

    Features:
    - Qubit-indexed gate scheduling
    - Full unitary computation for small circuits
    - Depth and gate-count analysis
    - Native L104 circuit serialisation
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
    def data(self):
        """Qiskit compat: circuit.data returns the list of operations."""
        return self.operations

    @property
    def depth(self) -> '_CallableInt':
        """Circuit depth (critical path length).

        Returns a ``_CallableInt`` so both ``circ.depth`` (property)
        and ``circ.depth()`` (Qiskit-style method) work identically.
        """
        if not self.operations:
            return _CallableInt(0)
        # Track the latest moment each qubit is occupied
        qubit_moments = [0] * self.num_qubits
        for op in self.operations:
            # This gate starts after the latest involved qubit is free
            start = max(qubit_moments[q] for q in op.qubits)
            end = start + 1
            for q in op.qubits:
                qubit_moments[q] = end
        return _CallableInt(max(qubit_moments))

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

    def h(self, qubit) -> 'GateCircuit':
        """Shorthand: append Hadamard. Accepts int or iterable of ints."""
        if isinstance(qubit, int):
            return self.append(H, [qubit])
        for q in qubit:
            self.append(H, [q])
        return self

    def x(self, qubit) -> 'GateCircuit':
        """Shorthand: append Pauli-X. Accepts int or iterable of ints."""
        if isinstance(qubit, int):
            return self.append(X, [qubit])
        for q in qubit:
            self.append(X, [q])
        return self

    def z(self, qubit) -> 'GateCircuit':
        """Shorthand: append Pauli-Z. Accepts int or iterable of ints."""
        if isinstance(qubit, int):
            return self.append(Z, [qubit])
        for q in qubit:
            self.append(Z, [q])
        return self

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

    # ─── Extended gate shorthands (Qiskit QuantumCircuit compat) ────────────

    def y(self, qubit) -> 'GateCircuit':
        """Shorthand: append Pauli-Y. Accepts int or iterable of ints."""
        if isinstance(qubit, int):
            return self.append(Y, [qubit])
        for q in qubit:
            self.append(Y, [q])
        return self

    def s(self, qubit) -> 'GateCircuit':
        """Shorthand: append S gate. Accepts int or iterable of ints."""
        if isinstance(qubit, int):
            return self.append(S, [qubit])
        for q in qubit:
            self.append(S, [q])
        return self

    def sdg(self, qubit) -> 'GateCircuit':
        """Shorthand: append S-dagger. Accepts int or iterable of ints."""
        if isinstance(qubit, int):
            return self.append(S_DAG, [qubit])
        for q in qubit:
            self.append(S_DAG, [q])
        return self

    def t(self, qubit) -> 'GateCircuit':
        """Shorthand: append T gate. Accepts int or iterable of ints."""
        if isinstance(qubit, int):
            return self.append(T, [qubit])
        for q in qubit:
            self.append(T, [q])
        return self

    def tdg(self, qubit) -> 'GateCircuit':
        """Shorthand: append T-dagger. Accepts int or iterable of ints."""
        if isinstance(qubit, int):
            return self.append(T_DAG, [qubit])
        for q in qubit:
            self.append(T_DAG, [q])
        return self

    def rx(self, theta: float, qubit) -> 'GateCircuit':
        """Shorthand: append Rx(θ). Qubit accepts int or iterable of ints."""
        if isinstance(qubit, int):
            return self.append(_Rx(theta), [qubit])
        for q in qubit:
            self.append(_Rx(theta), [q])
        return self

    def ry(self, theta: float, qubit) -> 'GateCircuit':
        """Shorthand: append Ry(θ). Qubit accepts int or iterable of ints."""
        if isinstance(qubit, int):
            return self.append(_Ry(theta), [qubit])
        for q in qubit:
            self.append(_Ry(theta), [q])
        return self

    def rz(self, theta: float, qubit) -> 'GateCircuit':
        """Shorthand: append Rz(θ). Qubit accepts int or iterable of ints."""
        if isinstance(qubit, int):
            return self.append(_Rz(theta), [qubit])
        for q in qubit:
            self.append(_Rz(theta), [q])
        return self

    def p(self, theta: float, qubit) -> 'GateCircuit':
        """Shorthand: append Phase gate P(θ). Qubit accepts int or iterable of ints."""
        if isinstance(qubit, int):
            return self.append(_Phase(theta), [qubit])
        for q in qubit:
            self.append(_Phase(theta), [q])
        return self

    def cry(self, theta: float, control: int, target: int) -> 'GateCircuit':
        """Shorthand: append Controlled-Ry CRY(θ)."""
        cry_gate = _Ry(theta).controlled(num_controls=1)
        return self.append(cry_gate, [control, target])

    def crx(self, theta: float, control: int, target: int) -> 'GateCircuit':
        """Shorthand: append Controlled-Rx CRX(θ)."""
        crx_gate = _Rx(theta).controlled(num_controls=1)
        return self.append(crx_gate, [control, target])

    def crz(self, theta: float, control: int, target: int) -> 'GateCircuit':
        """Shorthand: append Controlled-Rz CRZ(θ)."""
        crz_gate = _Rz(theta).controlled(num_controls=1)
        return self.append(crz_gate, [control, target])

    def cp(self, theta: float, control: int, target: int) -> 'GateCircuit':
        """Shorthand: append Controlled-Phase CP(θ)."""
        cp_gate = _Phase(theta).controlled(num_controls=1)
        return self.append(cp_gate, [control, target])

    def cz(self, control: int, target: int) -> 'GateCircuit':
        """Shorthand: append CZ."""
        return self.append(CZ, [control, target])

    def swap(self, q0: int, q1: int) -> 'GateCircuit':
        """Shorthand: append SWAP."""
        return self.append(SWAP, [q0, q1])

    def rzz(self, theta: float, q0: int, q1: int) -> 'GateCircuit':
        """Shorthand: append Rzz(θ)."""
        return self.append(_Rzz(theta), [q0, q1])

    def ccx(self, c0: int, c1: int, target: int) -> 'GateCircuit':
        """Shorthand: append Toffoli (CCX)."""
        return self.append(TOFFOLI, [c0, c1, target])

    def mcx(self, controls: List[int], target: int) -> 'GateCircuit':
        """Multi-controlled X. Toffoli for 2 controls, general for more."""
        if len(controls) == 1:
            return self.cx(controls[0], target)
        if len(controls) == 2:
            return self.ccx(controls[0], controls[1], target)
        # General: build controlled-X with n controls
        mcx_gate = X.controlled(num_controls=len(controls))
        return self.append(mcx_gate, controls + [target])

    def measure(self, qubit: int, classical_bit: int) -> 'GateCircuit':
        """Record a measurement operation (for simulation: projects state)."""
        meas_gate = QuantumGate(
            name="MEASURE", num_qubits=1,
            matrix=np.eye(2, dtype=complex),
            gate_type=GateType.MEASUREMENT,
        )
        self.operations.append(GateOperation(
            gate=meas_gate, qubits=(qubit,),
            classical_bits=(classical_bit,),
            label="MEASURE",
        ))
        if classical_bit >= self._num_classical:
            self._num_classical = classical_bit + 1
        return self

    def measure_all(self) -> 'GateCircuit':
        """Measure all qubits into classical bits 0..n-1."""
        for q in range(self.num_qubits):
            self.measure(q, q)
        return self

    @property
    def num_clbits(self) -> int:
        """Number of classical bits used."""
        return self._num_classical

    def count_ops(self) -> Dict[str, int]:
        """Count of each gate type (alias for gate_counts)."""
        return self.gate_counts

    def copy(self) -> 'GateCircuit':
        """Deep copy of the circuit."""
        new = GateCircuit(self.num_qubits, self.name)
        new._num_classical = self._num_classical
        for op in self.operations:
            new.operations.append(GateOperation(
                gate=op.gate,
                qubits=op.qubits,
                classical_bits=op.classical_bits,
                label=op.label,
                condition=op.condition,
            ))
        return new

    def remove_final_measurements(self, inplace: bool = False) -> 'GateCircuit':
        """Remove trailing measurement operations."""
        circ = self if inplace else self.copy()
        while circ.operations and (
            circ.operations[-1].label == "MEASURE"
            or circ.operations[-1].gate.gate_type == GateType.MEASUREMENT
        ):
            circ.operations.pop()
        return circ

    def compose(self, other: 'GateCircuit', qubit_map: Optional[Dict[int, int]] = None,
                inplace: bool = True, **kwargs) -> 'GateCircuit':
        """Append another circuit to this one, optionally remapping qubits.

        Args:
            other: Circuit to append.
            qubit_map: Optional remapping {source_qubit: target_qubit}.
            inplace: If True (default), modify self. If False, return a copy.
        """
        target = self if inplace else self.copy()
        for op in other.operations:
            if qubit_map:
                new_qubits = tuple(qubit_map.get(q, q) for q in op.qubits)
            else:
                new_qubits = op.qubits
            target.append(op.gate, new_qubits, label=op.label, condition=op.condition)
        return target

    def assign_parameters(self, bind_map: Dict) -> 'GateCircuit':
        """Assign numerical values to parameterized (placeholder) gates.

        Compatible with Qiskit's QuantumCircuit.assign_parameters().
        Replaces parametric rotation gates (those with theta ≈ 0.0 placeholder)
        with bound numerical values from the bind_map, matched positionally.

        Args:
            bind_map: Dict mapping Parameter/string → float value, ordered
                      by parameter index to replace placeholders sequentially.

        Returns:
            New GateCircuit with bound parameter values.
        """
        # Sort by parameter index to get ordered values
        def _sort_key(item):
            k = item[0]
            name = getattr(k, 'name', str(k))
            # Extract numeric index from names like 'θ[0]', 'γ[2]', etc.
            if '[' in name and ']' in name:
                try:
                    return int(name.split('[')[1].split(']')[0])
                except (ValueError, IndexError):
                    pass
            return hash(name)

        ordered_values = [float(v) for _, v in sorted(bind_map.items(), key=_sort_key)]

        # Gate constructors for rebuilding parametric operations
        gate_builders = {
            'RX': _Rx, 'RY': _Ry, 'RZ': _Rz, 'P': _Phase,
        }

        new = GateCircuit(self.num_qubits, self.name)
        new._num_classical = self._num_classical
        param_idx = 0

        for op in self.operations:
            # Identify placeholder parametric gates: have theta parameter ≈ 0.0
            if (op.gate.parameters
                    and 'theta' in op.gate.parameters
                    and abs(op.gate.parameters['theta']) < 1e-15
                    and param_idx < len(ordered_values)):
                base_name = op.gate.name.split('(')[0].upper()
                builder = gate_builders.get(base_name)
                if builder:
                    new_gate = builder(ordered_values[param_idx])
                    new.operations.append(GateOperation(
                        gate=new_gate, qubits=op.qubits,
                        label=op.label, condition=op.condition,
                    ))
                    param_idx += 1
                    continue

            # Non-parametric or already-bound — copy as-is
            new.operations.append(GateOperation(
                gate=op.gate, qubits=op.qubits,
                classical_bits=op.classical_bits,
                label=op.label, condition=op.condition,
            ))

        return new

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
        """Embed a small gate matrix into the full 2^n dimensional space.

        Uses numpy kron + permutation indexing instead of O(4^n) Python loops.
        Convention: qubits[0] → MSB of gate index (big-endian), so
        cx(control, target) maps CNOT control=MSB, target=LSB correctly.
        """
        n = self.num_qubits
        k = len(qubits)
        dim = 2 ** n

        # Build permutation: in the product space for kron(I_rest, gate),
        # gate occupies the lowest k bits (bit 0 = gate LSB = qubits[-1],
        # bit k-1 = gate MSB = qubits[0]). Rest qubits fill higher bits.
        rest = [q for q in range(n) if q not in qubits]
        perm_indices = list(reversed(qubits)) + rest

        # Build permutation array: perm_array[original_state] = permuted_state
        # v1.0.1: Vectorized bit manipulation instead of Python double loop.
        indices = np.arange(dim, dtype=np.intp)
        perm_array = np.zeros(dim, dtype=np.intp)
        for new_bit, old_bit in enumerate(perm_indices):
            perm_array |= ((indices >> old_bit) & 1) << new_bit

        # Gate on low bits, identity on high bits (in permuted basis)
        op_perm = np.kron(np.eye(max(1, 2 ** (n - k)), dtype=complex), gate_matrix)

        # Permute back: full[i,j] = op_perm[perm_array[i], perm_array[j]]
        return op_perm[np.ix_(perm_array, perm_array)]

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

    def to_dict_circuit(self) -> Dict[str, Any]:
        """
        Serialise the circuit to a pure L104 dictionary format.
        Framework-independent — no external quantum SDK required.

        Returns a dict with keys: name, num_qubits, operations (list of gate dicts),
        depth, num_operations.
        """
        ops = []
        for op in self.operations:
            if op.label == "BARRIER":
                ops.append({"gate": "BARRIER", "qubits": []})
                continue
            g = op.gate
            entry: Dict[str, Any] = {
                "gate": g.name,
                "qubits": list(op.qubits),
                "gate_type": g.gate_type.name,
            }
            if g.parameters:
                entry["parameters"] = g.parameters
            ops.append(entry)
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "operations": ops,
            "depth": self.depth,
            "num_operations": self.num_operations,
        }

    # Backward-compat alias (deprecated — use to_dict_circuit)
    def to_qiskit(self) -> Any:
        """DEPRECATED — Qiskit dependency removed. Returns L104 dict format instead."""
        return self.to_dict_circuit()

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

    # ═══════ MPS TENSOR NETWORK SIMULATION ═══════

    def simulate_mps(self, max_bond_dim: int = 256, shots: int = 1024,
                     sacred_mode: bool = False, **kwargs) -> 'TNSimulationResult':
        """
        Simulate this circuit using the MPS Tensor Network backend.

        Enables 25-50 qubit simulation with 100-1000x memory reduction.
        For circuits ≤ 12 qubits, results match full statevector exactly
        when bond dimension is unconstrained.

        Args:
            max_bond_dim: Maximum bond dimension χ (higher = more accurate)
            shots: Number of measurement samples
            sacred_mode: Use φ-balanced truncation with χ=104
            **kwargs: Passed to TensorNetworkSimulator.simulate()

        Returns:
            TNSimulationResult with probabilities, counts, entanglement data

        Example:
            circ = GateCircuit(25, "large_circuit")
            # ... build circuit ...
            result = circ.simulate_mps(max_bond_dim=256, shots=4096)
            print(result.probabilities)
            print(f"Memory: {result.memory_mb:.2f} MB (vs {circ.num_qubits}-qubit SV: "
                  f"{2**circ.num_qubits * 16 / 1e6:.0f} MB)")
        """
        from .tensor_network import TensorNetworkSimulator
        sim = TensorNetworkSimulator(
            max_bond_dim=max_bond_dim,
            sacred_mode=sacred_mode,
        )
        return sim.simulate(self, shots=shots, **kwargs)

    # ═══════ MEASUREMENT-FREE TRAJECTORY SIMULATION ═══════

    def simulate_trajectory(
        self,
        decoherence: str = "none",
        decoherence_rate: float = 0.01,
        weak_measurements: Optional[List[Tuple[int, int, float]]] = None,
        snapshot_every: int = 1,
        record_states: bool = True,
        seed: Optional[int] = None,
    ) -> 'TrajectoryResult':
        """
        Simulate this circuit as a measurement-free trajectory.

        Tracks state evolution gate-by-gate without projective collapse.
        Records coherence, purity, and entropy at each layer to provide
        research insights into decoherence dynamics.

        Args:
            decoherence: Model name — "none", "amplitude_damping",
                         "phase_damping", "depolarising", "sacred"
            decoherence_rate: Noise parameter per layer (0-1)
            weak_measurements: List of (layer, qubit, strength) tuples
            snapshot_every: Record detailed snapshot every N layers
            record_states: Store full statevectors in snapshots
            seed: RNG seed for reproducibility

        Returns:
            TrajectoryResult with per-layer coherence/decoherence profiles

        Example:
            circ = GateCircuit(4, "bell_decoherence")
            circ.h(0).cx(0, 1).cx(1, 2).cx(2, 3)
            result = circ.simulate_trajectory(
                decoherence="phase_damping", decoherence_rate=0.02,
                weak_measurements=[(2, 0, 0.1)],  # Weak Z on q0 at layer 2
            )
            print(f"Final purity: {result.final_purity:.6f}")
            print(f"Sacred coherence: {result.sacred_coherence}")
            print(f"Purity profile: {result.purity_profile}")
        """
        from .trajectory import TrajectorySimulator, DecoherenceModel

        model_map = {
            "none": DecoherenceModel.NONE,
            "amplitude_damping": DecoherenceModel.AMPLITUDE_DAMPING,
            "phase_damping": DecoherenceModel.PHASE_DAMPING,
            "depolarising": DecoherenceModel.DEPOLARISING,
            "depolarizing": DecoherenceModel.DEPOLARISING,
            "sacred": DecoherenceModel.SACRED,
        }
        model = model_map.get(decoherence.lower(), DecoherenceModel.NONE)

        sim = TrajectorySimulator(seed=seed)
        return sim.simulate(
            self,
            decoherence=model,
            decoherence_rate=decoherence_rate,
            weak_measurements=weak_measurements,
            snapshot_every=snapshot_every,
            record_states=record_states,
        )

    def simulate_trajectory_ensemble(
        self,
        num_trajectories: int = 100,
        decoherence: str = "phase_damping",
        decoherence_rate: float = 0.01,
        seed: Optional[int] = None,
    ) -> 'EnsembleResult':
        """
        Run Monte-Carlo trajectory ensemble for Lindblad dynamics.

        Averages N independent stochastic trajectories — converges to the
        exact open-system master equation as N → ∞.

        Args:
            num_trajectories: Number of independent trajectories
            decoherence: Model name
            decoherence_rate: Noise parameter per layer
            seed: RNG seed

        Returns:
            EnsembleResult with averaged purity/entropy profiles

        Example:
            result = circ.simulate_trajectory_ensemble(
                num_trajectories=200, decoherence="sacred", decoherence_rate=0.005
            )
            print(f"Final avg purity: {result.final_average_purity:.6f}")
            print(f"Sacred fraction: {result.sacred_coherence_fraction:.2%}")
        """
        from .trajectory import TrajectorySimulator, DecoherenceModel

        model_map = {
            "none": DecoherenceModel.NONE,
            "amplitude_damping": DecoherenceModel.AMPLITUDE_DAMPING,
            "phase_damping": DecoherenceModel.PHASE_DAMPING,
            "depolarising": DecoherenceModel.DEPOLARISING,
            "depolarizing": DecoherenceModel.DEPOLARISING,
            "sacred": DecoherenceModel.SACRED,
        }
        model = model_map.get(decoherence.lower(), DecoherenceModel.PHASE_DAMPING)

        sim = TrajectorySimulator(seed=seed)
        return sim.run_ensemble(
            self,
            num_trajectories=num_trajectories,
            decoherence=model,
            decoherence_rate=decoherence_rate,
        )

    # ═══════ ANALOG QUANTUM SIMULATION ═══════

    def analog_evolve(
        self,
        hamiltonian: Optional[Any] = None,
        t: float = 1.0,
        n_points: int = 50,
    ) -> Any:
        """
        Continuous-time Hamiltonian evolution for this circuit's qubit count.

        If no hamiltonian is provided, builds an L104 Sacred Hamiltonian.

        Args:
            hamiltonian: Hamiltonian object (or None for sacred default)
            t: Total evolution time
            n_points: Number of time samples

        Returns:
            EvolutionResult with exact states and energies

        Example:
            result = circ.analog_evolve(t=3.0, n_points=100)
            print(f"Ground energy: {result.energies[0]:.6f}")
        """
        from .analog import AnalogSimulator, HamiltonianBuilder
        sim = AnalogSimulator()
        if hamiltonian is None:
            hamiltonian = HamiltonianBuilder.sacred_hamiltonian(self.num_qubits)
        return sim.exact_evolve(hamiltonian, t=t, n_points=n_points)

    def trotterise(
        self,
        hamiltonian: Optional[Any] = None,
        t: float = 1.0,
        n_steps: int = 20,
        order: int = 2,
    ) -> 'GateCircuit':
        """
        Build a Trotterised circuit for Hamiltonian evolution.

        Args:
            hamiltonian: Hamiltonian (or None for sacred default)
            t: Total evolution time
            n_steps: Number of Trotter steps
            order: Product formula order (1, 2, or 4)

        Returns:
            New GateCircuit implementing the Trotter decomposition

        Example:
            trotter_circ = circ.trotterise(t=1.0, n_steps=10, order=2)
            print(f"Trotter gates: {trotter_circ.num_operations}")
        """
        from .analog import trotterise_to_circuit, HamiltonianBuilder, TrotterOrder
        if hamiltonian is None:
            hamiltonian = HamiltonianBuilder.sacred_hamiltonian(self.num_qubits)
        return trotterise_to_circuit(hamiltonian, t, n_steps, TrotterOrder(order))

    # ═══════ QUANTUM ML SUITE ═══════

    def build_qnn(self, depth: int = 2, ansatz: str = "hardware_efficient") -> Any:
        """
        Build a parameterised QNN circuit matching this circuit's qubit count.

        Args:
            depth: Number of variational layers
            ansatz: Ansatz type — "hardware_efficient", "strongly_entangling",
                    "sacred", "qaoa", or "data_reuploading"

        Returns:
            ParameterisedCircuit ready for training

        Example:
            qnn = circ.build_qnn(depth=3, ansatz="sacred")
            print(f"Parameters: {qnn.num_parameters}")
        """
        from .quantum_ml import AnsatzLibrary
        builders = {
            "hardware_efficient": AnsatzLibrary.hardware_efficient,
            "strongly_entangling": AnsatzLibrary.strongly_entangling,
            "sacred": AnsatzLibrary.sacred_ansatz,
            "qaoa": AnsatzLibrary.qaoa_layer,
            "data_reuploading": AnsatzLibrary.data_reuploading,
        }
        builder = builders.get(ansatz.lower(), AnsatzLibrary.hardware_efficient)
        return builder(self.num_qubits, depth=depth)

    def train_vqe(self, hamiltonian_matrix: Any = None,
                  max_iterations: int = 100, seed: Optional[int] = None) -> Any:
        """
        Run VQE on a Hamiltonian using this circuit's qubit count.

        If no Hamiltonian is provided, uses the L104 Sacred Hamiltonian.

        Args:
            hamiltonian_matrix: Hermitian matrix (None → sacred H)
            max_iterations: Optimisation steps
            seed: RNG seed

        Returns:
            VQEResult with ground energy and convergence history

        Example:
            result = circ.train_vqe(max_iterations=200)
            print(f"Ground energy: {result.ground_energy:.6f}")
        """
        from .quantum_ml import QuantumMLEngine
        qml = QuantumMLEngine()
        if hamiltonian_matrix is None:
            from .analog import HamiltonianBuilder
            H = HamiltonianBuilder.sacred_hamiltonian(self.num_qubits)
            hamiltonian_matrix = H.matrix()
        return qml.vqe(hamiltonian_matrix, max_iterations=max_iterations, seed=seed)

    # ═══════ CHAOS RESILIENCE ANALYSIS ═══════

    def is_clifford(self) -> bool:
        """Check if this circuit consists only of Clifford gates."""
        from .stabilizer_tableau import is_clifford_circuit
        return is_clifford_circuit(self)

    def simulate_stabilizer(self, shots: int = 1024, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Simulate this circuit using the Stabilizer Tableau backend.

        O(n²) per gate instead of O(2^n) — 1000x+ speedup for Clifford circuits.
        Falls back to hybrid mode if non-Clifford gates are present.

        Args:
            shots: Number of measurement samples
            seed: Optional RNG seed for reproducibility

        Returns:
            Dict with probabilities, counts, and performance metadata

        Example:
            circ = GateCircuit(100, "big_clifford")
            for i in range(100): circ.h(i)
            for i in range(99): circ.cx(i, i+1)
            result = circ.simulate_stabilizer(shots=4096)
            print(f"100-qubit Clifford in {result['execution_time_ms']:.1f}ms")
            print(f"Memory: {result['memory_bytes']} bytes (vs SV: impossible)")
        """
        from .stabilizer_tableau import (
            StabilizerTableau, HybridStabilizerSimulator, is_clifford_circuit,
        )

        if is_clifford_circuit(self):
            tab = StabilizerTableau(self.num_qubits, seed=seed)
            sim_info = tab.simulate_circuit(self)
            counts = tab.sample(shots)
            total = sum(counts.values())
            probs = {k: v / total for k, v in sorted(counts.items())}
            return {
                "probabilities": probs,
                "counts": counts,
                "shots": shots,
                "backend": "stabilizer_tableau",
                **sim_info,
            }
        else:
            hybrid = HybridStabilizerSimulator(seed=seed)
            result = hybrid.simulate(self, shots=shots)
            return {
                "probabilities": result.probabilities,
                "counts": result.counts,
                "shots": result.shots,
                "backend": result.backend_used,
                "execution_time_ms": result.execution_time_ms,
                "clifford_fraction": result.clifford_fraction,
                "memory_bytes": result.memory_bytes,
            }

    def chaos_resilience(self, noise_amplitude: float = 0.01, samples: int = 50) -> Dict[str, Any]:
        """
        Estimate circuit resilience to stochastic gate-parameter noise.

        Applies random perturbations to rotation angles and measures
        fidelity degradation. Maps directly to the chaos × conservation
        findings: circuits below the bifurcation threshold (0.35) stay
        coherent; above it, fidelity collapses.

        Only meaningful for circuits ≤ 12 qubits (needs unitary computation).

        Args:
            noise_amplitude: Max relative perturbation to rotation angles (0-1)
            samples: Number of noise trials

        Returns:
            Dict with fidelity stats, Lyapunov estimate, health verdict
        """
        if self.num_qubits > 12:
            return {
                "error": "circuit too large for unitary analysis",
                "num_qubits": self.num_qubits,
                "suggestion": "use ≤ 12 qubits for chaos resilience check",
            }
        if not self.operations:
            return {"error": "empty circuit"}

        import random

        # Ideal unitary
        U_ideal = self.unitary()
        dim = U_ideal.shape[0]

        fidelities = []
        for _ in range(samples):
            # Build noisy circuit
            noisy = GateCircuit(self.num_qubits, f"{self.name}_noisy")
            for op in self.operations:
                if op.label == "BARRIER":
                    noisy.barrier()
                    continue
                # If gate has rotation parameters, perturb them
                if op.gate.parameters and "theta" in op.gate.parameters:
                    theta = op.gate.parameters["theta"]
                    eps = noise_amplitude * (2 * random.random() - 1) * abs(theta) if theta != 0 else 0
                    noisy_theta = theta + eps
                    # Reconstruct perturbed gate
                    from .gates import Rx as _Rx_fn, Ry as _Ry_fn, Rz as _Rz_fn, make_phase as _P_fn
                    name_lower = op.gate.name.lower()
                    if name_lower.startswith("rx("):
                        g = _Rx_fn(noisy_theta)
                    elif name_lower.startswith("ry("):
                        g = _Ry_fn(noisy_theta)
                    elif name_lower.startswith("rz("):
                        g = _Rz_fn(noisy_theta)
                    elif name_lower.startswith("p("):
                        g = _P_fn(noisy_theta)
                    else:
                        g = op.gate  # Non-parametric — leave unchanged
                    noisy.append(g, op.qubits)
                else:
                    noisy.append(op.gate, op.qubits)

            U_noisy = noisy.unitary()
            # Process fidelity: |Tr(U_ideal† × U_noisy)|² / dim²
            overlap = np.abs(np.trace(U_ideal.conj().T @ U_noisy)) ** 2 / (dim ** 2)
            fidelities.append(float(overlap))

        mean_fid = sum(fidelities) / len(fidelities)
        min_fid = min(fidelities)
        max_fid = max(fidelities)
        fid_var = sum((f - mean_fid) ** 2 for f in fidelities) / len(fidelities)

        # Lyapunov estimate: how fast fidelity decays with noise
        # log(1 - mean_fidelity) / log(noise_amplitude) approximation
        lyapunov_est = 0.0
        if noise_amplitude > 0 and mean_fid < 1.0:
            lyapunov_est = math.log(max(1e-15, 1.0 - mean_fid)) / math.log(max(1e-15, noise_amplitude))

        # Map to bifurcation framework
        BIFURCATION = 0.35
        relative_noise = noise_amplitude * self.depth  # effective noise grows with depth
        below_bifurcation = relative_noise < BIFURCATION

        health = "COHERENT" if mean_fid > 0.99 else \
                 "RESILIENT" if mean_fid > 0.95 else \
                 "DEGRADED" if mean_fid > 0.80 else "DECOHERENT"

        return {
            "mean_fidelity": round(mean_fid, 8),
            "min_fidelity": round(min_fid, 8),
            "max_fidelity": round(max_fid, 8),
            "fidelity_variance": round(fid_var, 10),
            "lyapunov_estimate": round(lyapunov_est, 4),
            "noise_amplitude": noise_amplitude,
            "effective_noise": round(relative_noise, 6),
            "below_bifurcation": below_bifurcation,
            "bifurcation_threshold": BIFURCATION,
            "depth": self.depth,
            "samples": samples,
            "health": health,
        }
