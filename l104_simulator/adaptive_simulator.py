"""
===============================================================================
L104 SIMULATOR — ADAPTIVE QUANTUM SIMULATION BACKEND
===============================================================================

Unified simulation layer that automatically selects the best backend based
on circuit properties:

  ┌─────────────────────────────────────────────────────────────────┐
  │                   AdaptiveSimulator                              │
  │                                                                  │
  │  Circuit Analysis:                                               │
  │    • Qubit count                                                 │
  │    • Gate types (Clifford-only?)                                 │
  │    • Entanglement structure (nearest-neighbor? all-to-all?)      │
  │    • Circuit depth & width                                       │
  │    • Available memory (CPU/GPU)                                  │
  │                                                                  │
  │  Backend Selection:                                              │
  │    ┌──────────────────────────────────────────────────────┐      │
  │    │  n ≤ 20 qubits  → Statevector (CPU, direct numpy)   │      │
  │    │  Clifford-only   → Stabilizer (O(n²), any n)        │      │
  │    │  n ≤ 26 + GPU    → GPU Chunked Statevector           │      │
  │    │  n ≤ 30 + 16GB   → CPU Chunked Statevector           │      │
  │    │  n > 30 / low χ  → MPS Tensor Network                │      │
  │    │  Fallback         → MPS with max_bond=512             │      │
  │    └──────────────────────────────────────────────────────┘      │
  └─────────────────────────────────────────────────────────────────┘

Memory budget:
  The adaptive simulator estimates memory requirements BEFORE allocating,
  preventing OOM crashes. It also reports the backend selection rationale
  for transparency.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

from __future__ import annotations

import time
import warnings
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


class SimBackend(Enum):
    """Available simulation backends."""
    STATEVECTOR_CPU = auto()       # Pure NumPy statevector (≤20Q)
    STATEVECTOR_GPU = auto()       # CuPy GPU statevector (≤26Q typical)
    CHUNKED_CPU = auto()           # Chunked statevector on CPU (≤30Q)
    CHUNKED_GPU = auto()           # Chunked statevector on GPU
    MPS = auto()                   # Matrix Product State tensor network
    STABILIZER = auto()            # Clifford stabilizer tableau (O(n²))


@dataclass
class AdaptiveSimResult:
    """Unified result from the adaptive simulator."""
    probabilities: Dict[str, float]
    statevector: Optional[np.ndarray]
    n_qubits: int
    gate_count: int
    execution_time_ms: float
    backend_used: SimBackend
    backend_rationale: str
    memory_used_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "gate_count": self.gate_count,
            "execution_time_ms": round(self.execution_time_ms, 3),
            "backend": self.backend_used.name,
            "rationale": self.backend_rationale,
            "memory_mb": round(self.memory_used_bytes / (1024**2), 2),
            "num_nonzero_states": len(self.probabilities),
            "top_states": dict(sorted(self.probabilities.items(),
                                      key=lambda x: -x[1])[:10]),
        }


class AdaptiveSimulator:
    """
    Intelligent quantum circuit simulator that auto-selects the optimal backend.

    Usage:
        sim = AdaptiveSimulator()

        # From GateCircuit (l104_quantum_gate_engine)
        result = sim.run_gate_circuit(circuit)

        # From QuantumCircuit (l104_simulator)
        result = sim.run_quantum_circuit(qc)

        # Force a specific backend
        result = sim.run_gate_circuit(circuit, force_backend=SimBackend.MPS)

        # Check what backend would be used
        info = sim.select_backend(n_qubits=26, is_clifford=False)
    """

    def __init__(self, prefer_gpu: bool = True,
                 mps_max_bond: int = 256,
                 mps_fallback_bond: int = 512,
                 max_chunk_bytes: int = 2 * 1024**3,
                 statevector_threshold: int = 22,
                 return_statevector: bool = True):
        """
        Args:
            prefer_gpu: Try GPU backends when available.
            mps_max_bond: Default MPS bond dimension.
            mps_fallback_bond: Bond dimension for MPS as fallback for very large circuits.
            max_chunk_bytes: Max bytes per chunk for chunked simulation.
            statevector_threshold: Max qubits for direct (non-chunked) statevector.
            return_statevector: If True, return the full statevector when feasible.
        """
        self.prefer_gpu = prefer_gpu
        self.mps_max_bond = mps_max_bond
        self.mps_fallback_bond = mps_fallback_bond
        self.max_chunk_bytes = max_chunk_bytes
        self.sv_threshold = statevector_threshold
        self.return_statevector = return_statevector

    def select_backend(self, n_qubits: int,
                       is_clifford: bool = False,
                       entanglement_depth: Optional[int] = None,
                       force_backend: Optional[SimBackend] = None) -> Tuple[SimBackend, str]:
        """Select the optimal backend for a given circuit profile.

        Returns (backend, rationale_string).
        """
        if force_backend is not None:
            return force_backend, f"Forced: {force_backend.name}"

        from . import gpu_backend as gpu

        # 1. Clifford-only → Stabilizer (O(n²), works for ANY n)
        if is_clifford:
            return SimBackend.STABILIZER, (
                f"Clifford-only circuit ({n_qubits}Q) → Stabilizer tableau O(n²)"
            )

        # 2. Small circuits → Direct statevector
        if n_qubits <= self.sv_threshold:
            if self.prefer_gpu and gpu.GPU_AVAILABLE:
                return SimBackend.STATEVECTOR_GPU, (
                    f"{n_qubits}Q ≤ {self.sv_threshold}Q threshold → GPU statevector "
                    f"({gpu.estimate_statevector_bytes(n_qubits) / 1024**2:.1f} MB)"
                )
            return SimBackend.STATEVECTOR_CPU, (
                f"{n_qubits}Q ≤ {self.sv_threshold}Q threshold → CPU statevector "
                f"({gpu.estimate_statevector_bytes(n_qubits) / 1024**2:.1f} MB)"
            )

        # 3. Medium circuits → Chunked statevector (fits in memory?)
        sv_bytes = gpu.estimate_statevector_bytes(n_qubits)

        if self.prefer_gpu and gpu.GPU_AVAILABLE and gpu.fits_in_memory(n_qubits, "gpu"):
            return SimBackend.CHUNKED_GPU, (
                f"{n_qubits}Q → GPU chunked statevector "
                f"({sv_bytes / 1024**2:.1f} MB fits in GPU VRAM)"
            )

        if gpu.fits_in_memory(n_qubits, "cpu"):
            return SimBackend.CHUNKED_CPU, (
                f"{n_qubits}Q → CPU chunked statevector "
                f"({sv_bytes / 1024**2:.1f} MB fits in system RAM)"
            )

        # 4. Large circuits → MPS tensor network
        return SimBackend.MPS, (
            f"{n_qubits}Q → MPS tensor network (statevector would need "
            f"{sv_bytes / 1024**3:.1f} GB, exceeds memory budget)"
        )

    def run_gate_circuit(self, circuit, force_backend: Optional[SimBackend] = None,
                         initial_state: Optional[np.ndarray] = None) -> AdaptiveSimResult:
        """Execute a GateCircuit from l104_quantum_gate_engine.

        Auto-selects backend based on circuit properties.
        """
        n = circuit.num_qubits

        # Analyze circuit for Clifford-only detection
        is_clifford = self._is_gate_circuit_clifford(circuit)

        backend, rationale = self.select_backend(
            n_qubits=n,
            is_clifford=is_clifford,
            force_backend=force_backend,
        )

        return self._dispatch_gate_circuit(circuit, backend, rationale, initial_state)

    def run_quantum_circuit(self, circuit, force_backend: Optional[SimBackend] = None,
                            initial_state: Optional[np.ndarray] = None) -> AdaptiveSimResult:
        """Execute a QuantumCircuit from l104_simulator.

        Auto-selects backend based on circuit properties.
        """
        n = circuit.n_qubits

        # Check Clifford
        is_clifford = self._is_quantum_circuit_clifford(circuit)

        backend, rationale = self.select_backend(
            n_qubits=n,
            is_clifford=is_clifford,
            force_backend=force_backend,
        )

        return self._dispatch_quantum_circuit(circuit, backend, rationale, initial_state)

    # ═══════════════════════════════════════════════════════════════════════════
    #  DISPATCH
    # ═══════════════════════════════════════════════════════════════════════════

    def _dispatch_gate_circuit(self, circuit, backend: SimBackend,
                               rationale: str,
                               initial_state: Optional[np.ndarray]) -> AdaptiveSimResult:
        """Route GateCircuit execution to the selected backend."""
        from . import gpu_backend as gpu

        t0 = time.time()
        n = circuit.num_qubits

        if backend == SimBackend.STABILIZER:
            result = self._run_gate_circuit_stabilizer(circuit)
        elif backend in (SimBackend.STATEVECTOR_CPU, SimBackend.STATEVECTOR_GPU,
                         SimBackend.CHUNKED_CPU, SimBackend.CHUNKED_GPU):
            use_gpu = backend in (SimBackend.STATEVECTOR_GPU, SimBackend.CHUNKED_GPU)
            result = self._run_gate_circuit_chunked(circuit, use_gpu, initial_state)
        elif backend == SimBackend.MPS:
            result = self._run_gate_circuit_mps(circuit, initial_state)
        else:
            # Fallback: chunked CPU
            result = self._run_gate_circuit_chunked(circuit, False, initial_state)

        elapsed = (time.time() - t0) * 1000

        return AdaptiveSimResult(
            probabilities=result.get("probabilities", {}),
            statevector=result.get("statevector"),
            n_qubits=n,
            gate_count=result.get("gate_count", 0),
            execution_time_ms=elapsed,
            backend_used=backend,
            backend_rationale=rationale,
            memory_used_bytes=result.get("memory_bytes", gpu.estimate_statevector_bytes(n)),
            metadata=result.get("metadata", {}),
        )

    def _dispatch_quantum_circuit(self, circuit, backend: SimBackend,
                                  rationale: str,
                                  initial_state: Optional[np.ndarray]) -> AdaptiveSimResult:
        """Route QuantumCircuit execution to the selected backend."""
        from . import gpu_backend as gpu

        t0 = time.time()
        n = circuit.n_qubits

        if backend == SimBackend.STABILIZER:
            result = self._run_qc_stabilizer(circuit)
        elif backend in (SimBackend.STATEVECTOR_CPU, SimBackend.STATEVECTOR_GPU,
                         SimBackend.CHUNKED_CPU, SimBackend.CHUNKED_GPU):
            use_gpu = backend in (SimBackend.STATEVECTOR_GPU, SimBackend.CHUNKED_GPU)
            result = self._run_qc_chunked(circuit, use_gpu, initial_state)
        elif backend == SimBackend.MPS:
            result = self._run_qc_mps(circuit, initial_state)
        else:
            result = self._run_qc_chunked(circuit, False, initial_state)

        elapsed = (time.time() - t0) * 1000

        return AdaptiveSimResult(
            probabilities=result.get("probabilities", {}),
            statevector=result.get("statevector"),
            n_qubits=n,
            gate_count=result.get("gate_count", circuit.gate_count),
            execution_time_ms=elapsed,
            backend_used=backend,
            backend_rationale=rationale,
            memory_used_bytes=result.get("memory_bytes", gpu.estimate_statevector_bytes(n)),
            metadata=result.get("metadata", {}),
        )

    # ═══════════════════════════════════════════════════════════════════════════
    #  BACKEND: CHUNKED STATEVECTOR (GateCircuit)
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_gate_circuit_chunked(self, circuit, use_gpu: bool,
                                  initial_state: Optional[np.ndarray]) -> Dict[str, Any]:
        """Execute GateCircuit via chunked statevector."""
        from .chunked_statevector import ChunkedStatevectorSimulator

        sim = ChunkedStatevectorSimulator(
            use_gpu=use_gpu,
            max_chunk_bytes=self.max_chunk_bytes,
            return_statevector=self.return_statevector,
        )
        result = sim.run_gate_circuit(circuit, initial_state=initial_state)

        return {
            "probabilities": result.probabilities,
            "statevector": result.statevector,
            "gate_count": result.gate_count,
            "memory_bytes": result.memory_bytes,
            "metadata": {
                "chunked": result.chunked,
                "backend_detail": result.backend,
                **result.metadata,
            },
        }

    # ═══════════════════════════════════════════════════════════════════════════
    #  BACKEND: MPS (GateCircuit)
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_gate_circuit_mps(self, circuit,
                              initial_state: Optional[np.ndarray]) -> Dict[str, Any]:
        """Execute GateCircuit via MPS tensor network."""
        from .mps_simulator import MPSSimulator

        bond = self.mps_fallback_bond if circuit.num_qubits > 40 else self.mps_max_bond
        mps_sim = MPSSimulator(max_bond=bond)
        result = mps_sim.run_gate_circuit(circuit, initial_state=initial_state)

        return {
            "probabilities": result.get("probabilities", {}),
            "statevector": result.get("statevector"),
            "gate_count": result.get("gate_count", 0),
            "memory_bytes": result.get("memory_bytes", 0),
            "metadata": {
                "max_bond_dim": result.get("max_bond_dim", 0),
                "bond_dimensions": result.get("bond_dimensions", []),
                "truncation_error": result.get("truncation_error", 0),
                "estimated_fidelity": result.get("estimated_fidelity", 1.0),
            },
        }

    # ═══════════════════════════════════════════════════════════════════════════
    #  BACKEND: STABILIZER (GateCircuit)
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_gate_circuit_stabilizer(self, circuit) -> Dict[str, Any]:
        """Execute GateCircuit via stabilizer tableau (Clifford-only)."""
        from .stabilizer import StabilizerTableau

        n = circuit.num_qubits
        tab = StabilizerTableau(n)

        gate_count = 0
        for op in circuit.operations:
            if hasattr(op, 'label') and op.label == "BARRIER":
                continue
            name = op.gate.name.upper()
            qubits = list(op.qubits)

            if name in ('I', 'IDENTITY'):
                pass
            elif name == 'X':
                tab.x_gate(qubits[0])
            elif name == 'Y':
                tab.y_gate(qubits[0])
            elif name == 'Z':
                tab.z_gate(qubits[0])
            elif name == 'H':
                tab.hadamard(qubits[0])
            elif name in ('S', 'S_GATE'):
                tab.s_gate(qubits[0])
            elif name in ('SDG', 'S_DAG'):
                tab.s_dagger(qubits[0])
            elif name in ('CNOT', 'CX'):
                tab.cnot(qubits[0], qubits[1])
            elif name == 'CZ':
                tab.cz(qubits[0], qubits[1])
            elif name == 'SWAP':
                tab.swap(qubits[0], qubits[1])
            else:
                # Unknown gate — should not reach here if is_clifford was True
                warnings.warn(f"Non-Clifford gate {name} in stabilizer path")
            gate_count += 1

        # Sample measurements
        rng = np.random.default_rng()
        shots = 10000
        counts: Dict[str, int] = {}
        for _ in range(shots):
            tab_copy = StabilizerTableau(n)
            # Re-run (stabilizer measurement is destructive)
            for op in circuit.operations:
                if hasattr(op, 'label') and op.label == "BARRIER":
                    continue
                name = op.gate.name.upper()
                qubits = list(op.qubits)
                if name in ('I', 'IDENTITY'): pass
                elif name == 'X': tab_copy.x_gate(qubits[0])
                elif name == 'Y': tab_copy.y_gate(qubits[0])
                elif name == 'Z': tab_copy.z_gate(qubits[0])
                elif name == 'H': tab_copy.hadamard(qubits[0])
                elif name in ('S', 'S_GATE'): tab_copy.s_gate(qubits[0])
                elif name in ('SDG', 'S_DAG'): tab_copy.s_dagger(qubits[0])
                elif name in ('CNOT', 'CX'): tab_copy.cnot(qubits[0], qubits[1])
                elif name == 'CZ': tab_copy.cz(qubits[0], qubits[1])
                elif name == 'SWAP': tab_copy.swap(qubits[0], qubits[1])

            outcome = tab_copy.measure_all(rng)
            counts[outcome] = counts.get(outcome, 0) + 1

        probs = {k: v / shots for k, v in counts.items()}

        return {
            "probabilities": probs,
            "statevector": None,
            "gate_count": gate_count,
            "memory_bytes": n * n * 4,  # O(n²) tableau
            "metadata": {"backend": "stabilizer", "shots": shots},
        }

    # ═══════════════════════════════════════════════════════════════════════════
    #  BACKEND: CHUNKED STATEVECTOR (QuantumCircuit)
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_qc_chunked(self, circuit, use_gpu: bool,
                        initial_state: Optional[np.ndarray]) -> Dict[str, Any]:
        """Execute QuantumCircuit via chunked statevector."""
        from .chunked_statevector import ChunkedStatevectorSimulator

        sim = ChunkedStatevectorSimulator(
            use_gpu=use_gpu,
            max_chunk_bytes=self.max_chunk_bytes,
            return_statevector=self.return_statevector,
        )
        result = sim.run_quantum_circuit(circuit, initial_state=initial_state)

        return {
            "probabilities": result.probabilities,
            "statevector": result.statevector,
            "gate_count": result.gate_count,
            "memory_bytes": result.memory_bytes,
            "metadata": {"chunked": result.chunked, "backend_detail": result.backend},
        }

    def _run_qc_mps(self, circuit,
                    initial_state: Optional[np.ndarray]) -> Dict[str, Any]:
        """Execute QuantumCircuit via MPS."""
        from .mps_simulator import MPSSimulator

        bond = self.mps_fallback_bond if circuit.n_qubits > 40 else self.mps_max_bond
        mps_sim = MPSSimulator(max_bond=bond)
        result = mps_sim.run(circuit, initial_state=initial_state)

        probs = result.probabilities
        return {
            "probabilities": probs,
            "statevector": result.statevector if self.return_statevector else None,
            "gate_count": result.gate_count,
            "memory_bytes": 0,
            "metadata": {"backend": "mps"},
        }

    def _run_qc_stabilizer(self, circuit) -> Dict[str, Any]:
        """Execute QuantumCircuit via stabilizer."""
        from .stabilizer import StabilizerSimulator
        stab = StabilizerSimulator(fallback=False)
        result = stab.run(circuit)
        return {
            "probabilities": result.probabilities,
            "statevector": result.statevector if self.return_statevector else None,
            "gate_count": result.gate_count,
            "memory_bytes": circuit.n_qubits ** 2 * 4,
            "metadata": {"backend": "stabilizer"},
        }

    # ═══════════════════════════════════════════════════════════════════════════
    #  CIRCUIT ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════

    CLIFFORD_GATE_NAMES = frozenset({
        'I', 'IDENTITY', 'X', 'Y', 'Z', 'H', 'S', 'S_GATE', 'S_DAG', 'SDG',
        'CNOT', 'CX', 'CZ', 'SWAP',
    })

    def _is_gate_circuit_clifford(self, circuit) -> bool:
        """Check if a GateCircuit uses only Clifford gates."""
        for op in circuit.operations:
            if hasattr(op, 'label') and op.label == "BARRIER":
                continue
            name = op.gate.name.upper()
            if name not in self.CLIFFORD_GATE_NAMES:
                return False
        return True

    def _is_quantum_circuit_clifford(self, circuit) -> bool:
        """Check if a QuantumCircuit uses only Clifford gates."""
        for gate_rec in circuit.gates:
            if gate_rec.name.upper() not in self.CLIFFORD_GATE_NAMES:
                return False
        return True

    # ═══════════════════════════════════════════════════════════════════════════
    #  STATUS
    # ═══════════════════════════════════════════════════════════════════════════

    def status(self) -> Dict[str, Any]:
        """Return full simulator configuration and capability report."""
        from . import gpu_backend as gpu

        return {
            "simulator": "AdaptiveSimulator",
            "version": "1.0.0",
            "backends": [b.name for b in SimBackend],
            "prefer_gpu": self.prefer_gpu,
            "gpu_available": gpu.GPU_AVAILABLE,
            "gpu_info": gpu.gpu_info(),
            "mps_max_bond": self.mps_max_bond,
            "mps_fallback_bond": self.mps_fallback_bond,
            "statevector_threshold": self.sv_threshold,
            "max_chunk_bytes_gb": round(self.max_chunk_bytes / (1024**3), 2),
            "capability": {
                "max_statevector_qubits_gpu": gpu.max_statevector_qubits() if gpu.GPU_AVAILABLE else 0,
                "max_statevector_qubits_cpu": gpu.max_statevector_qubits(
                    memory_budget_bytes=None) if not gpu.GPU_AVAILABLE else None,
                "mps_unlimited_qubits": True,
                "stabilizer_unlimited_qubits": True,
            },
        }

    def estimate_resources(self, n_qubits: int,
                           is_clifford: bool = False) -> Dict[str, Any]:
        """Estimate resources and recommend backend for an n-qubit circuit."""
        from . import gpu_backend as gpu

        backend, rationale = self.select_backend(n_qubits, is_clifford)
        sv_bytes = gpu.estimate_statevector_bytes(n_qubits)
        old_unitary = gpu.estimate_unitary_bytes(n_qubits)

        return {
            "n_qubits": n_qubits,
            "recommended_backend": backend.name,
            "rationale": rationale,
            "statevector_memory": {
                "bytes": sv_bytes,
                "mb": round(sv_bytes / 1024**2, 2),
                "gb": round(sv_bytes / 1024**3, 4),
            },
            "old_unitary_memory": {
                "bytes": old_unitary,
                "gb": round(old_unitary / 1024**3, 4),
                "note": "This is what circuit.unitary() would allocate (ELIMINATED)",
            },
            "memory_savings_factor": old_unitary / max(sv_bytes, 1),
            "mps_memory_estimate": f"~{n_qubits * self.mps_max_bond**2 * 16 / 1024**2:.1f} MB "
                                   f"(n×chi²×16B, chi={self.mps_max_bond})",
        }

    def __repr__(self) -> str:
        from . import gpu_backend as gpu
        mode = "GPU" if self.prefer_gpu and gpu.GPU_AVAILABLE else "CPU"
        return f"AdaptiveSimulator({mode}, sv≤{self.sv_threshold}Q, mps_bond={self.mps_max_bond})"
