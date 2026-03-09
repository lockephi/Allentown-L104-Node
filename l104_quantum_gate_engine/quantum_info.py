"""
===============================================================================
L104 QUANTUM GATE ENGINE — QUANTUM INFO (LOCAL SIMULATION)
===============================================================================

Drop-in replacements for qiskit.quantum_info classes using pure NumPy.
Eliminates the Qiskit dependency while maintaining API compatibility.

Classes:
  Statevector      — Pure quantum state |ψ⟩ as complex vector
  DensityMatrix    — Mixed state ρ as complex matrix
  Operator         — Unitary/general operator as matrix
  SparsePauliOp    — Pauli-basis observable Σ cᵢ Pᵢ
  Parameter        — Symbolic circuit parameter placeholder
  ParameterVector  — Indexed collection of Parameters

Functions:
  partial_trace    — Trace out subsystem qubits
  entropy          — Von Neumann entropy S(ρ)
  state_fidelity   — Fidelity between two quantum states
  process_fidelity — Fidelity between two quantum channels/unitaries

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

from __future__ import annotations

import math
import cmath
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ═══════════════════════════════════════════════════════════════════════════════
#  PAULI MATRICES (module-level constants)
# ═══════════════════════════════════════════════════════════════════════════════

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

_PAULI_MAP = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


# ═══════════════════════════════════════════════════════════════════════════════
#  STATEVECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class Statevector:
    """Pure quantum state represented as a complex amplitude vector.

    Compatible replacement for ``qiskit.quantum_info.Statevector``.
    """

    def __init__(self, data, num_qubits: Optional[int] = None):
        if isinstance(data, (int, np.integer)):
            n = int(data)
            self._data = np.zeros(2 ** n, dtype=complex)
            self._data[0] = 1.0
        elif isinstance(data, Statevector):
            self._data = data._data.copy()
        elif isinstance(data, np.ndarray):
            self._data = np.asarray(data, dtype=complex).ravel()
        else:
            self._data = np.asarray(data, dtype=complex).ravel()

        dim = len(self._data)
        if dim == 0 or (dim & (dim - 1)) != 0:
            raise ValueError(f"Statevector dimension {dim} is not a power of 2")
        self._num_qubits = int(math.log2(dim))

    # ── constructors ──

    @classmethod
    def from_label(cls, label: str) -> "Statevector":
        """Create a computational basis state from a bitstring label, e.g. '010'."""
        n = len(label)
        idx = int(label, 2)
        data = np.zeros(2 ** n, dtype=complex)
        data[idx] = 1.0
        return cls(data)

    @classmethod
    def from_int(cls, i: int, dims: int) -> "Statevector":
        """Computational basis state |i⟩ in a dims-dimensional space."""
        data = np.zeros(dims, dtype=complex)
        data[i] = 1.0
        return cls(data)

    @classmethod
    def from_instruction(cls, circuit) -> "Statevector":
        """Create a Statevector by evolving |0...0⟩ through a circuit.

        Compatible replacement for ``qiskit.quantum_info.Statevector.from_instruction``.

        Args:
            circuit: A GateCircuit (or any object with num_qubits and operations).

        Returns:
            Statevector after applying the circuit to the all-zeros state.
        """
        n = circuit.num_qubits
        sv = cls.from_label('0' * n)
        return sv.evolve(circuit)

    # ── properties ──

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def dim(self) -> int:
        return len(self._data)

    # ── evolution ──

    def evolve(self, op) -> "Statevector":
        """Evolve state by a unitary operator, matrix, or GateCircuit.

        Args:
            op: An Operator, numpy matrix, or GateCircuit.

        Returns:
            New Statevector after evolution.
        """
        if isinstance(op, Operator):
            new_data = op._data @ self._data
            return Statevector(new_data)

        if isinstance(op, np.ndarray):
            new_data = op @ self._data
            return Statevector(new_data)

        # GateCircuit — apply gate-by-gate via tensor contraction
        try:
            state = self._data.copy()
            n = self._num_qubits
            dim = self.dim

            for gate_op in op.operations:
                if hasattr(gate_op, 'label') and gate_op.label == "BARRIER":
                    continue
                mat = gate_op.gate.matrix
                qubits = list(gate_op.qubits)
                k = len(qubits)

                if k == 1:
                    q = qubits[0]
                    psi = state.reshape([2] * n)
                    psi = np.tensordot(mat, psi, axes=([1], [q]))
                    psi = np.moveaxis(psi, 0, q)
                    state = psi.reshape(dim)
                elif k == 2:
                    q0, q1 = qubits
                    gate_4d = mat.reshape(2, 2, 2, 2)
                    psi = state.reshape([2] * n)
                    es = _build_einsum_2q(n, q0, q1)
                    psi = np.einsum(es, gate_4d, psi)
                    state = psi.reshape(dim)
                else:
                    gate_kd = mat.reshape([2] * (2 * k))
                    psi = state.reshape([2] * n)
                    psi = _apply_general_gate(gate_kd, psi, qubits, n, k)
                    state = psi.reshape(dim)

            return Statevector(state)
        except Exception:
            raise TypeError(f"Cannot evolve Statevector with {type(op)}")

    # ── measurement ──

    def probabilities(self) -> np.ndarray:
        """Probability of each computational basis state."""
        return np.abs(self._data) ** 2

    def probabilities_dict(self) -> Dict[str, float]:
        """Probabilities as {bitstring: probability} dict (non-zero only)."""
        probs = self.probabilities()
        n = self._num_qubits
        return {
            format(i, f'0{n}b'): float(p)
            for i, p in enumerate(probs) if p > 1e-15
        }

    def sample_counts(self, shots: int = 1024, seed: Optional[int] = None) -> Dict[str, int]:
        """Sample measurement outcomes from the state.

        Args:
            shots: Number of measurement samples.
            seed: Optional RNG seed.

        Returns:
            Dict mapping bitstring to count.
        """
        rng = np.random.default_rng(seed)
        probs = self.probabilities()
        probs = probs / probs.sum()  # normalize
        indices = rng.choice(len(probs), size=shots, p=probs)
        n = self._num_qubits
        counts: Dict[str, int] = {}
        for idx in indices:
            key = format(idx, f'0{n}b')
            counts[key] = counts.get(key, 0) + 1
        return counts

    def sample_memory(self, shots: int = 1024, seed: Optional[int] = None) -> List[str]:
        """Return a list of sampled bitstrings (shot-by-shot)."""
        rng = np.random.default_rng(seed)
        probs = self.probabilities()
        probs = probs / probs.sum()
        indices = rng.choice(len(probs), size=shots, p=probs)
        n = self._num_qubits
        return [format(idx, f'0{n}b') for idx in indices]

    # ── linear algebra ──

    def expectation_value(self, operator) -> complex:
        """Compute ⟨ψ|O|ψ⟩.

        Args:
            operator: Operator, SparsePauliOp, or numpy matrix.
        """
        if isinstance(operator, SparsePauliOp):
            mat = operator.to_matrix()
        elif isinstance(operator, Operator):
            mat = operator._data
        elif isinstance(operator, np.ndarray):
            mat = operator
        else:
            raise TypeError(f"Unsupported operator type: {type(operator)}")
        return complex(self._data.conj() @ mat @ self._data)

    def inner(self, other: "Statevector") -> complex:
        """Inner product ⟨self|other⟩."""
        return complex(self._data.conj() @ other._data)

    def tensor(self, other: "Statevector") -> "Statevector":
        """Tensor product |self⟩ ⊗ |other⟩."""
        return Statevector(np.kron(self._data, other._data))

    def to_density_matrix(self) -> "DensityMatrix":
        """Convert to DensityMatrix |ψ⟩⟨ψ|."""
        return DensityMatrix(np.outer(self._data, self._data.conj()))

    def purity(self) -> float:
        """Purity of the state (always 1.0 for pure states)."""
        return 1.0

    def is_valid(self, atol: float = 1e-8) -> bool:
        """Check if normalized: ⟨ψ|ψ⟩ ≈ 1."""
        return abs(np.vdot(self._data, self._data) - 1.0) < atol

    def normalize(self) -> "Statevector":
        """Return a normalized copy."""
        norm = np.linalg.norm(self._data)
        if norm < 1e-15:
            return Statevector(self._data.copy())
        return Statevector(self._data / norm)

    def copy(self) -> "Statevector":
        return Statevector(self._data.copy())

    def __repr__(self) -> str:
        return f"Statevector({self._data}, dims=({', '.join(['2'] * self._num_qubits)},))"

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Statevector):
            return False
        return np.allclose(self._data, other._data, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
#  DENSITY MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

class DensityMatrix:
    """Mixed quantum state represented as a density matrix ρ.

    Compatible replacement for ``qiskit.quantum_info.DensityMatrix``.
    """

    def __init__(self, data):
        if isinstance(data, Statevector):
            self._data = np.outer(data._data, data._data.conj())
        elif isinstance(data, DensityMatrix):
            self._data = data._data.copy()
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                self._data = np.outer(data, data.conj())
            else:
                self._data = np.asarray(data, dtype=complex)
        else:
            self._data = np.asarray(data, dtype=complex)

        dim = self._data.shape[0]
        if dim == 0 or (dim & (dim - 1)) != 0:
            raise ValueError(f"DensityMatrix dimension {dim} is not a power of 2")
        self._num_qubits = int(math.log2(dim))

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def dim(self) -> int:
        return self._data.shape[0]

    def evolve(self, op) -> "DensityMatrix":
        """Evolve: ρ → U ρ U†.

        Args:
            op: Operator, numpy matrix, or GateCircuit.
        """
        if isinstance(op, Operator):
            U = op._data
        elif isinstance(op, np.ndarray):
            U = op
        else:
            # GateCircuit: evolve via Statevector if pure, else build unitary
            try:
                U = op.unitary()
            except Exception:
                raise TypeError(f"Cannot evolve DensityMatrix with {type(op)}")

        new_data = U @ self._data @ U.conj().T
        return DensityMatrix(new_data)

    def purity(self) -> float:
        """Tr(ρ²)."""
        return float(np.trace(self._data @ self._data).real)

    def trace(self) -> float:
        """Tr(ρ)."""
        return float(np.trace(self._data).real)

    def expectation_value(self, operator) -> complex:
        """Tr(ρ O)."""
        if isinstance(operator, SparsePauliOp):
            mat = operator.to_matrix()
        elif isinstance(operator, Operator):
            mat = operator._data
        elif isinstance(operator, np.ndarray):
            mat = operator
        else:
            raise TypeError(f"Unsupported operator type: {type(operator)}")
        return complex(np.trace(self._data @ mat))

    def to_statevector(self) -> Statevector:
        """Extract statevector (only valid for pure states)."""
        eigvals, eigvecs = np.linalg.eigh(self._data)
        idx = np.argmax(eigvals)
        return Statevector(eigvecs[:, idx])

    def is_valid(self, atol: float = 1e-8) -> bool:
        """Check trace ≈ 1 and positive semi-definite."""
        if abs(self.trace() - 1.0) > atol:
            return False
        eigvals = np.linalg.eigvalsh(self._data)
        return bool(np.all(eigvals > -atol))

    def copy(self) -> "DensityMatrix":
        return DensityMatrix(self._data.copy())

    def __repr__(self) -> str:
        return f"DensityMatrix({self._data.shape}, dims=({', '.join(['2'] * self._num_qubits)},))"


# ═══════════════════════════════════════════════════════════════════════════════
#  OPERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class Operator:
    """Quantum operator (unitary or general) as a matrix.

    Compatible replacement for ``qiskit.quantum_info.Operator``.
    """

    def __init__(self, data):
        if isinstance(data, Operator):
            self._data = data._data.copy()
        elif isinstance(data, np.ndarray):
            self._data = np.asarray(data, dtype=complex)
        else:
            # Accept GateCircuit → build unitary
            try:
                self._data = np.asarray(data.unitary(), dtype=complex)
            except Exception:
                self._data = np.asarray(data, dtype=complex)

        if self._data.ndim == 2:
            dim = self._data.shape[0]
            if dim > 0 and (dim & (dim - 1)) == 0:
                self._num_qubits = int(math.log2(dim))
            else:
                self._num_qubits = 0
        else:
            self._num_qubits = 0

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def dim(self) -> Tuple[int, int]:
        return self._data.shape

    def compose(self, other: "Operator") -> "Operator":
        """Operator composition: self · other (matrix multiplication)."""
        if isinstance(other, Operator):
            return Operator(self._data @ other._data)
        return Operator(self._data @ np.asarray(other, dtype=complex))

    def tensor(self, other: "Operator") -> "Operator":
        """Tensor product: self ⊗ other."""
        if isinstance(other, Operator):
            return Operator(np.kron(self._data, other._data))
        return Operator(np.kron(self._data, np.asarray(other, dtype=complex)))

    def adjoint(self) -> "Operator":
        """Conjugate transpose (dagger)."""
        return Operator(self._data.conj().T)

    def power(self, n: int) -> "Operator":
        """Matrix power O^n."""
        return Operator(np.linalg.matrix_power(self._data, n))

    def is_unitary(self, atol: float = 1e-10) -> bool:
        """Check U†U ≈ I."""
        product = self._data.conj().T @ self._data
        return np.allclose(product, np.eye(self._data.shape[0]), atol=atol)

    def to_matrix(self) -> np.ndarray:
        return self._data.copy()

    def __matmul__(self, other) -> "Operator":
        return self.compose(other)

    def __repr__(self) -> str:
        return f"Operator({self._data.shape})"


# ═══════════════════════════════════════════════════════════════════════════════
#  SPARSE PAULI OP
# ═══════════════════════════════════════════════════════════════════════════════

class SparsePauliOp:
    """Sparse representation of a Pauli-basis operator Σ cᵢ Pᵢ.

    Compatible replacement for ``qiskit.quantum_info.SparsePauliOp``.

    Example:
        op = SparsePauliOp(["ZZ", "XX", "YY"], [1.0, 0.5, 0.5])
        mat = op.to_matrix()
    """

    def __init__(self, labels: Union[List[str], str],
                 coeffs: Optional[Union[List[complex], np.ndarray]] = None):
        if isinstance(labels, str):
            labels = [labels]
        self._labels = list(labels)
        if coeffs is None:
            self._coeffs = np.ones(len(self._labels), dtype=complex)
        else:
            self._coeffs = np.asarray(coeffs, dtype=complex)
        if len(self._labels) == 0:
            raise ValueError("SparsePauliOp requires at least one term")
        self._num_qubits = len(self._labels[0])

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def size(self) -> int:
        return len(self._labels)

    @property
    def coeffs(self) -> np.ndarray:
        return self._coeffs

    @property
    def labels(self) -> List[str]:
        return self._labels

    def to_matrix(self) -> np.ndarray:
        """Expand to dense 2^n × 2^n matrix."""
        dim = 2 ** self._num_qubits
        mat = np.zeros((dim, dim), dtype=complex)
        for label, coeff in zip(self._labels, self._coeffs):
            term = _pauli_string_to_matrix(label)
            mat += coeff * term
        return mat

    def to_list(self) -> List[Tuple[str, complex]]:
        """Return list of (pauli_string, coefficient) tuples."""
        return list(zip(self._labels, [complex(c) for c in self._coeffs]))

    @classmethod
    def from_list(cls, terms: List[Tuple[str, complex]]) -> "SparsePauliOp":
        """Create from list of (label, coeff) tuples."""
        labels = [t[0] for t in terms]
        coeffs = [t[1] for t in terms]
        return cls(labels, coeffs)

    def adjoint(self) -> "SparsePauliOp":
        """Hermitian conjugate (Pauli strings are self-adjoint, coeffs get conjugated)."""
        return SparsePauliOp(self._labels, self._coeffs.conj())

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"SparsePauliOp({self._labels[:3]}{'...' if len(self._labels) > 3 else ''}, n={self._num_qubits})"


# ═══════════════════════════════════════════════════════════════════════════════
#  PARAMETER / PARAMETER VECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class ParameterExpression:
    """Deferred arithmetic expression on Parameters."""

    def __init__(self, expr_fn, name: str = "expr"):
        self._expr_fn = expr_fn
        self._name = name

    def bind(self, param_dict: Dict) -> float:
        return self._expr_fn(param_dict)

    def __repr__(self) -> str:
        return f"ParameterExpression({self._name})"

    # Arithmetic that produces new expressions
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return ParameterExpression(lambda d, s=self, o=other: s.bind(d) * o, f"{self._name}*{other}")
        if isinstance(other, (Parameter, ParameterExpression)):
            return ParameterExpression(lambda d, s=self, o=other: s.bind(d) * o.bind(d), f"{self._name}*{other._name}")
        return NotImplemented

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return ParameterExpression(lambda d, s=self, o=other: s.bind(d) + o, f"{self._name}+{other}")
        if isinstance(other, (Parameter, ParameterExpression)):
            return ParameterExpression(lambda d, s=self, o=other: s.bind(d) + o.bind(d), f"{self._name}+{other._name}")
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return ParameterExpression(lambda d, s=self, o=other: s.bind(d) - o, f"{self._name}-{other}")
        if isinstance(other, (Parameter, ParameterExpression)):
            return ParameterExpression(lambda d, s=self, o=other: s.bind(d) - o.bind(d), f"{self._name}-{other._name}")
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return ParameterExpression(lambda d, s=self, o=other: o - s.bind(d), f"{other}-{self._name}")
        return NotImplemented

    def __neg__(self):
        return ParameterExpression(lambda d, s=self: -s.bind(d), f"-{self._name}")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return ParameterExpression(lambda d, s=self, o=other: s.bind(d) / o, f"{self._name}/{other}")
        return NotImplemented

    def __float__(self):
        raise TypeError("Cannot convert unbound ParameterExpression to float. Call .bind() first.")


class Parameter(ParameterExpression):
    """Symbolic parameter placeholder for parameterized circuits.

    Compatible replacement for ``qiskit.circuit.Parameter``.
    """

    def __init__(self, name: str):
        self._name = name
        self._expr_fn = lambda d, n=name: d[n]

    @property
    def name(self) -> str:
        return self._name

    def bind(self, param_dict: Dict) -> float:
        if self._name in param_dict:
            return float(param_dict[self._name])
        # Try to find by Parameter object key
        for k, v in param_dict.items():
            if isinstance(k, Parameter) and k._name == self._name:
                return float(v)
            if isinstance(k, str) and k == self._name:
                return float(v)
        raise KeyError(f"Parameter '{self._name}' not found in binding dict")

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        if isinstance(other, Parameter):
            return self._name == other._name
        return NotImplemented

    def __repr__(self) -> str:
        return f"Parameter({self._name})"


class ParameterVector:
    """Indexed collection of Parameters.

    Compatible replacement for ``qiskit.circuit.ParameterVector``.

    Example:
        params = ParameterVector("theta", 5)
        for i in range(5):
            circuit.ry(params[i], i)
    """

    def __init__(self, name: str, length: int):
        self._name = name
        self._params = [Parameter(f"{name}[{i}]") for i in range(length)]

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._params)

    def __getitem__(self, idx) -> Parameter:
        return self._params[idx]

    def __iter__(self):
        return iter(self._params)

    def __repr__(self) -> str:
        return f"ParameterVector({self._name}, length={len(self._params)})"


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def partial_trace(state, trace_systems: Union[int, List[int]]) -> DensityMatrix:
    """Partial trace: trace out specified qubit subsystems.

    Args:
        state: Statevector or DensityMatrix.
        trace_systems: Qubit index or list of qubit indices to trace out.

    Returns:
        DensityMatrix of the reduced system.
    """
    if isinstance(trace_systems, int):
        trace_systems = [trace_systems]

    if isinstance(state, Statevector):
        rho = np.outer(state._data, state._data.conj())
        n = state.num_qubits
    elif isinstance(state, DensityMatrix):
        rho = state._data
        n = state.num_qubits
    else:
        rho = np.asarray(state, dtype=complex)
        dim = rho.shape[0]
        n = int(math.log2(dim))

    # Reshape ρ to tensor with 2n indices: (2,2,...,2, 2,2,...,2)
    rho_tensor = rho.reshape([2] * (2 * n))

    # Trace out each qubit in trace_systems (process from highest index down)
    keep = sorted(set(range(n)) - set(trace_systems))
    n_keep = len(keep)

    # Build trace by contracting bra and ket indices for traced qubits
    for q in sorted(trace_systems, reverse=True):
        # Trace over qubit q: contract axis q (ket) with axis q+n (bra)
        # After reshape, ket indices are 0..n-1, bra indices are n..2n-1
        current_n = rho_tensor.ndim // 2
        bra_idx = current_n + q
        rho_tensor = np.trace(rho_tensor, axis1=q, axis2=bra_idx)
        # After trace, remaining axes shift — we need to adjust
        # np.trace removes both axes and appends the trace result at the end?
        # Actually np.trace with axis1/axis2 contracts those axes

    # Reshape back to matrix
    dim_keep = 2 ** n_keep
    result = rho_tensor.reshape(dim_keep, dim_keep)
    return DensityMatrix(result)


def purity(state) -> float:
    """Purity Tr(ρ²) of a quantum state.

    Args:
        state: Statevector, DensityMatrix, or numpy array.

    Returns:
        Purity in [0, 1].  Pure states → 1.0, maximally mixed → 1/d.
    """
    if isinstance(state, Statevector):
        return 1.0
    if isinstance(state, DensityMatrix):
        return state.purity()
    arr = np.asarray(state, dtype=complex)
    if arr.ndim == 1:
        return 1.0
    return float(np.trace(arr @ arr).real)


def entropy(state, base: int = 2) -> float:
    """Von Neumann entropy S(ρ) = -Tr(ρ log ρ).

    Args:
        state: DensityMatrix or Statevector (entropy = 0 for pure states).
        base: Logarithm base (2 for bits, e for nats).

    Returns:
        Von Neumann entropy.
    """
    if isinstance(state, Statevector):
        return 0.0  # Pure states have zero entropy

    if isinstance(state, DensityMatrix):
        rho = state._data
    else:
        rho = np.asarray(state, dtype=complex)

    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]  # Filter near-zero eigenvalues

    if len(eigvals) == 0:
        return 0.0

    if base == 2:
        return float(-np.sum(eigvals * np.log2(eigvals)))
    elif base == math.e or base == 0:
        return float(-np.sum(eigvals * np.log(eigvals)))
    else:
        return float(-np.sum(eigvals * np.log(eigvals)) / math.log(base))


def state_fidelity(state1, state2) -> float:
    """Fidelity between two quantum states.

    For pure states: F = |⟨ψ₁|ψ₂⟩|²
    For mixed states: F = (Tr√(√ρ₁ ρ₂ √ρ₁))²

    Args:
        state1, state2: Statevector, DensityMatrix, or numpy array.

    Returns:
        Fidelity in [0, 1].
    """
    s1 = _to_state(state1)
    s2 = _to_state(state2)

    if isinstance(s1, Statevector) and isinstance(s2, Statevector):
        return float(abs(np.vdot(s1._data, s2._data)) ** 2)

    # At least one is a density matrix
    rho1 = s1._data if isinstance(s1, DensityMatrix) else np.outer(s1._data, s1._data.conj())
    rho2 = s2._data if isinstance(s2, DensityMatrix) else np.outer(s2._data, s2._data.conj())

    # F = (Tr √(√ρ₁ ρ₂ √ρ₁))²
    sqrt_rho1 = _matrix_sqrt(rho1)
    product = sqrt_rho1 @ rho2 @ sqrt_rho1
    sqrt_product = _matrix_sqrt(product)
    fid = float(np.trace(sqrt_product).real) ** 2
    return min(1.0, max(0.0, fid))


def process_fidelity(op1, op2=None) -> float:
    """Process fidelity between two unitary operators.

    F_proc = |Tr(U₁† U₂)|² / d²

    Args:
        op1: Operator, numpy matrix, or QuantumGate.
        op2: Second operator (default: identity).

    Returns:
        Process fidelity in [0, 1].
    """
    U1 = _to_matrix(op1)
    if op2 is None:
        U2 = np.eye(U1.shape[0], dtype=complex)
    else:
        U2 = _to_matrix(op2)

    d = U1.shape[0]
    inner = np.trace(U1.conj().T @ U2)
    return float(abs(inner) ** 2) / (d ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _pauli_string_to_matrix(label: str) -> np.ndarray:
    """Convert a Pauli string like 'XYZ' to its 2^n × 2^n matrix via Kronecker product."""
    result = np.array([[1.0]], dtype=complex)
    for ch in label:
        result = np.kron(result, _PAULI_MAP[ch.upper()])
    return result


def _matrix_sqrt(mat: np.ndarray) -> np.ndarray:
    """Matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.maximum(eigvals, 0.0)
    sqrt_vals = np.sqrt(eigvals)
    return eigvecs @ np.diag(sqrt_vals) @ eigvecs.conj().T


def _to_state(s):
    """Coerce input to Statevector or DensityMatrix."""
    if isinstance(s, (Statevector, DensityMatrix)):
        return s
    arr = np.asarray(s, dtype=complex)
    if arr.ndim == 1:
        return Statevector(arr)
    return DensityMatrix(arr)


def _to_matrix(op) -> np.ndarray:
    """Extract matrix from various operator types."""
    if isinstance(op, Operator):
        return op._data
    if isinstance(op, np.ndarray):
        return op
    if hasattr(op, 'matrix'):
        return op.matrix
    if hasattr(op, 'unitary'):
        return op.unitary()
    return np.asarray(op, dtype=complex)


def _build_einsum_2q(n: int, q0: int, q1: int) -> str:
    """Build einsum subscript string for a 2-qubit gate on n-qubit state."""
    state_in = list(range(n))
    state_out = list(range(n))
    g_o0 = n
    g_o1 = n + 1
    state_out[q0] = g_o0
    state_out[q1] = g_o1
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN'
    gate_str = ''.join(letters[i] for i in [g_o0, g_o1, q0, q1])
    in_str = ''.join(letters[i] for i in state_in)
    out_str = ''.join(letters[i] for i in state_out)
    return f"{gate_str},{in_str}->{out_str}"


def _apply_general_gate(gate_kd: np.ndarray, psi: np.ndarray,
                        qubits: list, n: int, k: int) -> np.ndarray:
    """Apply a k-qubit gate via einsum to an n-qubit state tensor."""
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN'
    state_in = list(range(n))
    state_out = list(range(n))
    gate_out_indices = []
    gate_in_indices = []
    for idx_i, q in enumerate(qubits):
        new_idx = n + idx_i
        gate_out_indices.append(new_idx)
        gate_in_indices.append(q)
        state_out[q] = new_idx
    gate_indices = gate_out_indices + gate_in_indices
    gate_str = ''.join(letters[i] for i in gate_indices)
    in_str = ''.join(letters[i] for i in state_in)
    out_str = ''.join(letters[i] for i in state_out)
    return np.einsum(f"{gate_str},{in_str}->{out_str}", gate_kd, psi)
