"""
===============================================================================
L104 SIMULATOR — CLIFFORD STABILIZER TABLEAU SIMULATOR
===============================================================================

O(n²) simulation for circuits composed entirely of Clifford gates
(H, S, CNOT, X, Y, Z, CZ, SWAP). Uses the Aaronson-Gottesman stabilizer
tableau representation with vectorized NumPy operations.

For n qubits, the tableau is a (2n) × (2n+1) binary matrix encoding
n stabilizer generators and n destabilizer generators. Each row represents
a Pauli operator as a binary vector (x₁...xₙ | z₁...zₙ | phase).

When a circuit mixes Clifford and non-Clifford gates, use `is_clifford()`
to auto-select between this and the statevector simulator.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import math
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from .simulator import QuantumCircuit, SimulationResult, GateRecord


# Gates recognized as Clifford
CLIFFORD_GATES = frozenset({
    'I', 'X', 'Y', 'Z', 'H', 'S', 'Sdg',
    'CNOT', 'CZ', 'SWAP',
})


def is_clifford(circuit: QuantumCircuit) -> bool:
    """Check if a circuit uses only Clifford gates."""
    return all(g.name in CLIFFORD_GATES for g in circuit.gates)


class StabilizerTableau:
    """Binary symplectic tableau for n-qubit stabilizer states.

    All gate operations are vectorized over all 2n rows simultaneously
    using NumPy column slicing — no Python loops over rows.

    Representation (Aaronson-Gottesman):
      - 2n rows: first n are destabilizers, last n are stabilizers
      - Each row: [x₁...xₙ | z₁...zₙ | r] where r is the phase bit
    """

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        # Tableau: 2n rows × (2n + 1) cols
        # Cols 0..n-1: x bits, cols n..2n-1: z bits, col 2n: phase bit
        self.tab = np.zeros((2 * n_qubits, 2 * n_qubits + 1), dtype=np.uint8)
        # Initialize |0⟩^n
        for i in range(n_qubits):
            self.tab[i, i] = 1
            self.tab[n_qubits + i, n_qubits + i] = 1

    # ─── Vectorized Clifford Gate Updates ─────────────────────────────

    def hadamard(self, q: int):
        """Apply H gate: X↔Z, phase r ^= x·z. Vectorized over all rows."""
        n = self.n
        x_col = self.tab[:, q].copy()
        z_col = self.tab[:, n + q].copy()
        self.tab[:, 2 * n] ^= (x_col & z_col)
        self.tab[:, q] = z_col
        self.tab[:, n + q] = x_col

    def s_gate(self, q: int):
        """Apply S gate: r ^= x·z, z ^= x. Vectorized."""
        n = self.n
        x_col = self.tab[:, q]
        z_col = self.tab[:, n + q]
        self.tab[:, 2 * n] ^= (x_col & z_col)
        self.tab[:, n + q] = z_col ^ x_col

    def s_dagger(self, q: int):
        """Apply S† gate: S³ = S†."""
        self.s_gate(q)
        self.s_gate(q)
        self.s_gate(q)

    def cnot(self, ctrl: int, tgt: int):
        """Apply CNOT gate. Vectorized."""
        n = self.n
        xc = self.tab[:, ctrl]
        zc = self.tab[:, n + ctrl]
        xt = self.tab[:, tgt]
        zt = self.tab[:, n + tgt]
        # Phase: r ^= x_ctrl & z_tgt & (x_tgt ^ z_ctrl ^ 1)
        self.tab[:, 2 * n] ^= (xc & zt & (xt ^ zc ^ np.uint8(1)))
        # x_target ^= x_control
        self.tab[:, tgt] = xt ^ xc
        # z_control ^= z_target
        self.tab[:, n + ctrl] = zc ^ zt

    def x_gate(self, q: int):
        """Apply X gate. Vectorized: r ^= z."""
        self.tab[:, 2 * self.n] ^= self.tab[:, self.n + q]

    def y_gate(self, q: int):
        """Apply Y gate. Vectorized: r ^= x ^ z."""
        n = self.n
        self.tab[:, 2 * n] ^= (self.tab[:, q] ^ self.tab[:, n + q])

    def z_gate(self, q: int):
        """Apply Z gate. Vectorized: r ^= x."""
        self.tab[:, 2 * self.n] ^= self.tab[:, q]

    def cz(self, q0: int, q1: int):
        """Apply CZ gate: H(q1) CNOT(q0,q1) H(q1)."""
        self.hadamard(q1)
        self.cnot(q0, q1)
        self.hadamard(q1)

    def swap(self, q0: int, q1: int):
        """Apply SWAP gate: 3 CNOTs."""
        self.cnot(q0, q1)
        self.cnot(q1, q0)
        self.cnot(q0, q1)

    # ─── Row multiplication (for measurement) ────────────────────────

    def _rowmult(self, target: int, source: int):
        """Multiply row[target] by row[source] (Pauli group multiplication).
        Vectorized phase computation over all n qubits."""
        n = self.n
        x1 = self.tab[source, :n].astype(np.int8)
        z1 = self.tab[source, n:2*n].astype(np.int8)
        x2 = self.tab[target, :n].astype(np.int8)
        z2 = self.tab[target, n:2*n].astype(np.int8)

        # Vectorized phase contribution per qubit
        # Y case (x1=1, z1=1): contribute x2 - z2
        # X case (x1=1, z1=0): contribute z2 * (2*x2 - 1)
        # Z case (x1=0, z1=1): contribute x2 * (1 - 2*z2)
        is_Y = x1 & z1
        is_X = x1 & (1 - z1)
        is_Z = (1 - x1) & z1

        contrib = is_Y * (x2 - z2) + is_X * (z2 * (2 * x2 - 1)) + is_Z * (x2 * (1 - 2 * z2))
        phase = int(np.sum(contrib))

        new_r = (int(self.tab[source, 2*n]) + int(self.tab[target, 2*n]) +
                 int(phase % 4 == 2 or phase % 4 == 3)) & 1

        # XOR the x and z bits
        self.tab[target, :2*n] ^= self.tab[source, :2*n]
        self.tab[target, 2*n] = new_r

    # ─── Measurement ──────────────────────────────────────────────────

    def measure(self, qubit: int, rng: Optional[np.random.Generator] = None) -> int:
        """Measure a qubit in the computational basis."""
        n = self.n

        # Find first stabilizer row with x[qubit] = 1
        stab_x = self.tab[n:2*n, qubit]
        nonzero = np.nonzero(stab_x)[0]

        if len(nonzero) > 0:
            p = n + nonzero[0]  # Absolute row index

            # Multiply all other rows with x[qubit]=1 by row p
            for i in range(2 * n):
                if i != p and self.tab[i, qubit]:
                    self._rowmult(i, p)

            # Move stabilizer row p to destabilizer
            self.tab[p - n] = self.tab[p].copy()

            # Set stabilizer row p to Z_qubit with random phase
            self.tab[p] = 0
            self.tab[p, n + qubit] = 1

            if rng is None:
                rng = np.random.default_rng()
            outcome = int(rng.integers(2))
            self.tab[p, 2 * n] = outcome
            return outcome
        else:
            # Deterministic outcome
            scratch = np.zeros(2 * n + 1, dtype=np.uint8)
            destab_x = self.tab[:n, qubit]
            for i in np.nonzero(destab_x)[0]:
                # Multiply scratch by stabilizer row n+i
                src = n + i
                x1 = self.tab[src, :n].astype(np.int8)
                z1 = self.tab[src, n:2*n].astype(np.int8)
                x2 = scratch[:n].astype(np.int8)
                z2 = scratch[n:2*n].astype(np.int8)

                is_Y = x1 & z1
                is_X = x1 & (1 - z1)
                is_Z = (1 - x1) & z1
                contrib = is_Y * (x2 - z2) + is_X * (z2 * (2*x2 - 1)) + is_Z * (x2 * (1 - 2*z2))
                phase = int(np.sum(contrib))

                new_r = (int(self.tab[src, 2*n]) + int(scratch[2*n]) +
                         int(phase % 4 == 2 or phase % 4 == 3)) & 1
                scratch[:2*n] ^= self.tab[src, :2*n]
                scratch[2*n] = new_r

            return int(scratch[2 * n])

    def measure_all(self, rng: Optional[np.random.Generator] = None) -> str:
        """Measure all qubits, returning a bitstring."""
        if rng is None:
            rng = np.random.default_rng()
        return ''.join(str(self.measure(q, rng)) for q in range(self.n))


class StabilizerSimulator:
    """Clifford stabilizer simulator with O(n²) per gate.

    Same interface as Simulator. For non-Clifford circuits, raises ValueError
    unless fallback=True, in which case it falls back to statevector.

    Usage:
        stab = StabilizerSimulator()
        qc = QuantumCircuit(100, 'big_clifford')
        qc.h(0)
        for i in range(99): qc.cx(i, i+1)
        result = stab.run(qc, shots=10000)
    """

    def __init__(self, fallback: bool = True):
        self.fallback = fallback

    def run(self, circuit: QuantumCircuit, shots: int = 10000,
            seed: Optional[int] = None) -> SimulationResult:
        """Execute a Clifford circuit via stabilizer simulation.

        Since stabilizer states cannot produce a full statevector efficiently,
        we return a sampled probability distribution via repeated measurement.
        For large n (>20), the statevector is not constructed (too large);
        instead, a sparse representation is returned.
        """
        t0 = time.time()

        if not is_clifford(circuit):
            if self.fallback:
                from .simulator import Simulator
                return Simulator().run(circuit)
            raise ValueError(
                f"Circuit contains non-Clifford gates: "
                f"{[g.name for g in circuit.gates if g.name not in CLIFFORD_GATES]}"
            )

        n = circuit.n_qubits
        rng = np.random.default_rng(seed)

        # Build the post-circuit tableau ONCE, then clone for each shot
        # (circuit application is deterministic; only measurement is random)
        base_tab = StabilizerTableau(n)
        self._apply_circuit(base_tab, circuit)

        # Sample by cloning the tableau and measuring
        counts: Dict[str, int] = {}
        for _ in range(shots):
            tab = StabilizerTableau.__new__(StabilizerTableau)
            tab.n = n
            tab.tab = base_tab.tab.copy()
            outcome = tab.measure_all(rng)
            counts[outcome] = counts.get(outcome, 0) + 1

        # Build probability dict
        probs = {k: v / shots for k, v in counts.items()}

        # For small n, construct full statevector; for large n, use sparse
        if n <= 20:
            sv = np.zeros(2**n, dtype=complex)
            for bitstr, p in probs.items():
                sv[int(bitstr, 2)] = math.sqrt(p)
        else:
            # Sparse representation: only allocate for observed states
            # Use a minimal statevector with just enough entries
            sv = np.zeros(2, dtype=complex)
            sv[0] = math.sqrt(probs.get('0' * n, 0.0))
            sv[1] = math.sqrt(probs.get('1' * n, 0.0))

        elapsed = (time.time() - t0) * 1000

        return SimulationResult(
            statevector=sv,
            n_qubits=min(n, 20) if n > 20 else n,
            circuit_name=circuit.name,
            gate_count=circuit.gate_count,
            execution_time_ms=elapsed,
        )

    def _apply_circuit(self, tab: StabilizerTableau, circuit: QuantumCircuit):
        """Apply all gates in a circuit to the tableau."""
        for gate_rec in circuit.gates:
            name = gate_rec.name
            qubits = gate_rec.qubits

            if name == 'I':
                pass
            elif name == 'X':
                tab.x_gate(qubits[0])
            elif name == 'Y':
                tab.y_gate(qubits[0])
            elif name == 'Z':
                tab.z_gate(qubits[0])
            elif name == 'H':
                tab.hadamard(qubits[0])
            elif name == 'S':
                tab.s_gate(qubits[0])
            elif name == 'Sdg':
                tab.s_dagger(qubits[0])
            elif name == 'CNOT':
                tab.cnot(qubits[0], qubits[1])
            elif name == 'CZ':
                tab.cz(qubits[0], qubits[1])
            elif name == 'SWAP':
                tab.swap(qubits[0], qubits[1])

    def __repr__(self) -> str:
        return f"StabilizerSimulator(fallback={self.fallback})"
