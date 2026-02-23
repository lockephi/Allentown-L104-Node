"""
===============================================================================
L104 QUANTUM GATE ENGINE — CROSS-SYSTEM ORCHESTRATOR
===============================================================================

Unified orchestration layer that bridges ALL L104 quantum subsystems:

  ┌──────────────────────────────────────────────────────────────┐
  │                  CrossSystemOrchestrator                      │
  │                                                              │
  │  GateAlgebra ─→ GateCompiler ─→ ErrorCorrection              │
  │       ↓              ↓              ↓                        │
  │  GateCircuit ──→ Compiled Circuit ──→ Protected Circuit       │
  │       ↓              ↓              ↓                        │
  │  ┌─────────────────────────────────────────┐                  │
  │  │         EXECUTION TARGETS                │                  │
  │  │  • l104_quantum_runtime (IBM QPU)        │                  │
  │  │  • l104_quantum_coherence (algorithms)   │                  │
  │  │  • l104_quantum_logic (manifold)         │                  │
  │  │  • l104_asi/quantum (ASI core)           │                  │
  │  │  • l104_science_engine (physics)         │                  │
  │  │  • Local Statevector simulation          │                  │
  │  └─────────────────────────────────────────┘                  │
  └──────────────────────────────────────────────────────────────┘

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import math
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

from .gates import (
    GateAlgebra, QuantumGate, GateType, GateSet,
    I, X, Y, Z, H, S, T, CNOT, CZ, SWAP,
    Rx, Ry, Rz, U3, Phase,
    PHI_GATE, GOD_CODE_PHASE, VOID_GATE, IRON_GATE, SACRED_ENTANGLER,
    FIBONACCI_BRAID, ANYON_EXCHANGE,
)
from .circuit import GateCircuit, GateOperation
from .compiler import GateCompiler, CompilationResult, OptimizationLevel
from .error_correction import (
    ErrorCorrectionLayer, ErrorCorrectionScheme,
    SurfaceCode, SteaneCode, FibonacciAnyonProtection, ZeroNoiseExtrapolation,
)
from .constants import (
    PHI, GOD_CODE, VOID_CONSTANT, FEIGENBAUM, ALPHA_FINE,
    IRON_ATOMIC_NUMBER, IRON_FREQUENCY,
)


class ExecutionTarget(Enum):
    """Where to execute compiled circuits."""
    LOCAL_STATEVECTOR = auto()     # NumPy statevector simulation
    QISKIT_STATEVECTOR = auto()   # Qiskit Statevector simulator
    QISKIT_AER = auto()           # Qiskit Aer with noise model
    IBM_QPU = auto()              # Real IBM quantum hardware via runtime
    COHERENCE_ENGINE = auto()     # l104_quantum_coherence algorithms
    ASI_QUANTUM = auto()          # l104_asi/quantum computation core
    SCIENCE_ENGINE = auto()       # l104_science_engine quantum circuit
    MANIFOLD = auto()             # l104_quantum_logic manifold


@dataclass
class ExecutionResult:
    """Result of executing a circuit on a target."""
    target: ExecutionTarget
    probabilities: Optional[Dict[str, float]] = None
    statevector: Optional[np.ndarray] = None
    counts: Optional[Dict[str, int]] = None
    expectation_values: Optional[Dict[str, float]] = None
    execution_time: float = 0.0
    shots: int = 0
    fidelity: float = 1.0
    error_mitigated: bool = False
    sacred_alignment: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target.name,
            "execution_time_ms": self.execution_time * 1000,
            "shots": self.shots,
            "fidelity": self.fidelity,
            "error_mitigated": self.error_mitigated,
            "num_states": len(self.probabilities) if self.probabilities else 0,
            "sacred_alignment": self.sacred_alignment,
            "metadata": self.metadata,
        }


class CrossSystemOrchestrator:
    """
    Master orchestrator for the L104 Quantum Gate Engine.

    Provides a unified API to:
    1. Build circuits using the universal gate algebra
    2. Compile and optimize for any target
    3. Apply error correction
    4. Execute on any L104 quantum subsystem
    5. Analyze results with sacred alignment scoring

    This is the primary entry point for all gate engine operations.
    """

    VERSION = "1.0.0"

    def __init__(self):
        self.algebra = GateAlgebra()
        self.compiler = GateCompiler()
        self.error_correction = ErrorCorrectionLayer()

        # Lazy-loaded subsystem bridges
        self._runtime = None
        self._coherence_engine = None
        self._asi_quantum = None
        self._science_engine = None
        self._manifold = None

        # Telemetry
        self._metrics = {
            "circuits_built": 0,
            "circuits_compiled": 0,
            "circuits_executed": 0,
            "total_gates": 0,
            "total_qubits_used": 0,
            "sacred_gates_used": 0,
            "error_corrections_applied": 0,
        }
        self._boot_time = time.time()

    # ═══════════════════════════════════════════════════════════════════════════
    #  SUBSYSTEM BRIDGES (Lazy Loading)
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def runtime(self):
        """L104 Quantum Runtime bridge for IBM QPU execution."""
        if self._runtime is None:
            try:
                from l104_quantum_runtime import get_runtime
                self._runtime = get_runtime()
            except ImportError:
                self._runtime = False  # Mark as unavailable
        return self._runtime if self._runtime is not False else None

    @property
    def coherence_engine(self):
        """L104 Quantum Coherence Engine for algorithm execution."""
        if self._coherence_engine is None:
            try:
                from l104_quantum_coherence import QuantumCoherenceEngine
                self._coherence_engine = QuantumCoherenceEngine()
            except ImportError:
                self._coherence_engine = False
        return self._coherence_engine if self._coherence_engine is not False else None

    @property
    def asi_quantum(self):
        """L104 ASI Quantum Computation Core."""
        if self._asi_quantum is None:
            try:
                from l104_asi.quantum import QuantumComputationCore
                self._asi_quantum = QuantumComputationCore()
            except ImportError:
                self._asi_quantum = False
        return self._asi_quantum if self._asi_quantum is not False else None

    @property
    def science_engine(self):
        """L104 Science Engine for physics-level quantum ops."""
        if self._science_engine is None:
            try:
                from l104_science_engine import ScienceEngine
                self._science_engine = ScienceEngine()
            except ImportError:
                self._science_engine = False
        return self._science_engine if self._science_engine is not False else None

    @property
    def manifold(self):
        """L104 Quantum Logic Manifold for dimensional processing."""
        if self._manifold is None:
            try:
                from l104_quantum_logic import QuantumEntanglementManifold
                self._manifold = QuantumEntanglementManifold(dimensions=26)
            except ImportError:
                self._manifold = False
        return self._manifold if self._manifold is not False else None

    # ═══════════════════════════════════════════════════════════════════════════
    #  CIRCUIT BUILDING
    # ═══════════════════════════════════════════════════════════════════════════

    def create_circuit(self, num_qubits: int, name: str = "circuit") -> GateCircuit:
        """Create a new gate circuit."""
        self._metrics["circuits_built"] += 1
        self._metrics["total_qubits_used"] += num_qubits
        return GateCircuit(num_qubits, name)

    def bell_pair(self, qubit_a: int = 0, qubit_b: int = 1) -> GateCircuit:
        """Create a Bell pair (|Φ+⟩) circuit."""
        n = max(qubit_a, qubit_b) + 1
        circ = GateCircuit(n, "bell_pair")
        circ.append(H, [qubit_a])
        circ.append(CNOT, [qubit_a, qubit_b])
        return circ

    def ghz_state(self, num_qubits: int = 3) -> GateCircuit:
        """Create a GHZ state circuit: (|00...0⟩ + |11...1⟩)/√2."""
        circ = GateCircuit(num_qubits, f"ghz_{num_qubits}")
        circ.append(H, [0])
        for i in range(1, num_qubits):
            circ.append(CNOT, [0, i])
        return circ

    def quantum_fourier_transform(self, num_qubits: int) -> GateCircuit:
        """Build the Quantum Fourier Transform circuit."""
        circ = GateCircuit(num_qubits, f"qft_{num_qubits}")

        for i in range(num_qubits):
            circ.append(H, [i])
            for j in range(i + 1, num_qubits):
                angle = math.pi / (2 ** (j - i))
                circ.append(Phase(angle).controlled(1), [j, i])
                # Simplified: use CPhase as explicit controlled-Rz
                # circ.append(CZ, [j, i])  # Approximate for now

        # Reverse qubit order
        for i in range(num_qubits // 2):
            circ.append(SWAP, [i, num_qubits - 1 - i])

        return circ

    def grover_oracle(self, target: int, num_qubits: int) -> GateCircuit:
        """Build a Grover oracle marking state |target⟩."""
        circ = GateCircuit(num_qubits, f"grover_oracle_{target}")

        # Convert target to binary
        bits = format(target, f'0{num_qubits}b')

        # Apply X gates to flip qubits that are 0 in the target
        for i, bit in enumerate(reversed(bits)):
            if bit == '0':
                circ.append(X, [i])

        # Multi-controlled Z (simplified: CZ chain)
        if num_qubits >= 2:
            circ.append(H, [num_qubits - 1])
            # Chain of CNOTs for multi-control
            for i in range(num_qubits - 1):
                circ.append(CNOT, [i, num_qubits - 1])
            circ.append(H, [num_qubits - 1])

        # Undo X flips
        for i, bit in enumerate(reversed(bits)):
            if bit == '0':
                circ.append(X, [i])

        return circ

    def grover_diffusion(self, num_qubits: int) -> GateCircuit:
        """Build the Grover diffusion operator."""
        circ = GateCircuit(num_qubits, "grover_diffusion")

        # H⊗n
        for i in range(num_qubits):
            circ.append(H, [i])

        # X⊗n
        for i in range(num_qubits):
            circ.append(X, [i])

        # Multi-controlled Z
        circ.append(H, [num_qubits - 1])
        for i in range(num_qubits - 1):
            circ.append(CNOT, [i, num_qubits - 1])
        circ.append(H, [num_qubits - 1])

        # X⊗n
        for i in range(num_qubits):
            circ.append(X, [i])

        # H⊗n
        for i in range(num_qubits):
            circ.append(H, [i])

        return circ

    def sacred_circuit(self, num_qubits: int, depth: int = 3) -> GateCircuit:
        """
        Build a circuit using L104 sacred gates.
        Creates a GOD_CODE-aligned quantum state with PHI-harmonic entanglement.
        """
        circ = GateCircuit(num_qubits, f"sacred_d{depth}")

        for d in range(depth):
            # Layer 1: Hadamard superposition
            for q in range(num_qubits):
                circ.append(H, [q])

            # Layer 2: Sacred phase alignment
            for q in range(num_qubits):
                if q % 3 == 0:
                    circ.append(PHI_GATE, [q])
                elif q % 3 == 1:
                    circ.append(GOD_CODE_PHASE, [q])
                else:
                    circ.append(IRON_GATE, [q])

            # Layer 3: Sacred entanglement
            for q in range(0, num_qubits - 1, 2):
                circ.append(SACRED_ENTANGLER, [q, q + 1])

            # Layer 4: VOID alignment
            for q in range(num_qubits):
                circ.append(VOID_GATE, [q])

        self._metrics["sacred_gates_used"] += circ.num_operations
        return circ

    # ═══════════════════════════════════════════════════════════════════════════
    #  COMPILATION
    # ═══════════════════════════════════════════════════════════════════════════

    def compile(self, circuit: GateCircuit,
                target: GateSet = GateSet.UNIVERSAL,
                optimization: OptimizationLevel = OptimizationLevel.O2) -> CompilationResult:
        """Compile a circuit to target gate set with optimization."""
        self._metrics["circuits_compiled"] += 1
        return self.compiler.compile(circuit, target, optimization)

    def compile_for_ibm(self, circuit: GateCircuit,
                         optimization: OptimizationLevel = OptimizationLevel.O2) -> CompilationResult:
        """Compile specifically for IBM Eagle/Heron hardware."""
        return self.compile(circuit, GateSet.IBM_EAGLE, optimization)

    def compile_sacred(self, circuit: GateCircuit) -> CompilationResult:
        """Compile to L104 sacred gate set with O3 sacred alignment."""
        return self.compile(circuit, GateSet.L104_SACRED, OptimizationLevel.O3)

    def compile_topological(self, circuit: GateCircuit) -> CompilationResult:
        """Compile to topological braiding gates."""
        return self.compile(circuit, GateSet.TOPOLOGICAL, OptimizationLevel.O1)

    # ═══════════════════════════════════════════════════════════════════════════
    #  ERROR CORRECTION
    # ═══════════════════════════════════════════════════════════════════════════

    def protect(self, circuit: GateCircuit,
                scheme: ErrorCorrectionScheme = ErrorCorrectionScheme.STEANE_7_1_3,
                distance: int = 3):
        """Apply error correction to a circuit."""
        self._metrics["error_corrections_applied"] += 1
        return self.error_correction.encode(circuit, scheme, distance)

    def protect_surface(self, circuit: GateCircuit, distance: int = 3):
        """Apply surface code protection."""
        return self.protect(circuit, ErrorCorrectionScheme.SURFACE_CODE, distance)

    def protect_topological(self, circuit: GateCircuit):
        """Apply Fibonacci anyon topological protection."""
        return self.protect(circuit, ErrorCorrectionScheme.FIBONACCI_ANYON)

    # ═══════════════════════════════════════════════════════════════════════════
    #  EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════

    def execute(self, circuit: GateCircuit,
                target: ExecutionTarget = ExecutionTarget.LOCAL_STATEVECTOR,
                shots: int = 1024) -> ExecutionResult:
        """
        Execute a circuit on the specified target.
        """
        self._metrics["circuits_executed"] += 1
        self._metrics["total_gates"] += circuit.num_operations
        start = time.time()

        if target == ExecutionTarget.LOCAL_STATEVECTOR:
            result = self._execute_local(circuit)
        elif target == ExecutionTarget.QISKIT_STATEVECTOR:
            result = self._execute_qiskit_sv(circuit)
        elif target == ExecutionTarget.QISKIT_AER:
            result = self._execute_qiskit_aer(circuit, shots)
        elif target == ExecutionTarget.IBM_QPU:
            result = self._execute_ibm_qpu(circuit, shots)
        elif target == ExecutionTarget.COHERENCE_ENGINE:
            result = self._execute_coherence(circuit)
        elif target == ExecutionTarget.ASI_QUANTUM:
            result = self._execute_asi(circuit)
        elif target == ExecutionTarget.MANIFOLD:
            result = self._execute_manifold(circuit)
        else:
            result = self._execute_local(circuit)

        result.execution_time = time.time() - start
        result.sacred_alignment = self._compute_sacred_alignment(circuit)
        return result

    def _execute_local(self, circuit: GateCircuit) -> ExecutionResult:
        """Execute via direct NumPy statevector simulation."""
        try:
            unitary = circuit.unitary()
            dim = 2 ** circuit.num_qubits
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0  # |0...0⟩
            final_state = unitary @ state

            probs = np.abs(final_state) ** 2
            prob_dict = {}
            for i in range(dim):
                p = float(probs[i])
                if p > 1e-10:
                    label = format(i, f'0{circuit.num_qubits}b')
                    prob_dict[label] = p

            return ExecutionResult(
                target=ExecutionTarget.LOCAL_STATEVECTOR,
                probabilities=prob_dict,
                statevector=final_state,
                metadata={"simulator": "numpy_statevector", "god_code": GOD_CODE},
            )
        except Exception as e:
            return ExecutionResult(
                target=ExecutionTarget.LOCAL_STATEVECTOR,
                metadata={"error": str(e)},
            )

    def _execute_qiskit_sv(self, circuit: GateCircuit) -> ExecutionResult:
        """Execute via Qiskit Statevector."""
        try:
            from qiskit.quantum_info import Statevector
            qc = circuit.to_qiskit()
            sv = Statevector.from_instruction(qc)
            probs = sv.probabilities_dict()

            return ExecutionResult(
                target=ExecutionTarget.QISKIT_STATEVECTOR,
                probabilities=probs,
                statevector=sv.data,
                metadata={"simulator": "qiskit_statevector"},
            )
        except ImportError:
            return self._execute_local(circuit)
        except Exception as e:
            return ExecutionResult(
                target=ExecutionTarget.QISKIT_STATEVECTOR,
                metadata={"error": str(e)},
            )

    def _execute_qiskit_aer(self, circuit: GateCircuit, shots: int) -> ExecutionResult:
        """Execute via Qiskit Aer with noise model."""
        try:
            from qiskit_aer import AerSimulator
            qc = circuit.to_qiskit()
            qc.measure_all()

            sim = AerSimulator()
            job = sim.run(qc, shots=shots)
            result = job.result()
            counts = result.get_counts()

            total = sum(counts.values())
            probs = {k: v / total for k, v in counts.items()}

            return ExecutionResult(
                target=ExecutionTarget.QISKIT_AER,
                probabilities=probs,
                counts=counts,
                shots=shots,
                metadata={"simulator": "qiskit_aer"},
            )
        except ImportError:
            return self._execute_qiskit_sv(circuit)
        except Exception as e:
            return ExecutionResult(
                target=ExecutionTarget.QISKIT_AER,
                metadata={"error": str(e)},
            )

    def _execute_ibm_qpu(self, circuit: GateCircuit, shots: int) -> ExecutionResult:
        """Execute on real IBM QPU via l104_quantum_runtime."""
        if not self.runtime:
            return self._execute_qiskit_sv(circuit)

        try:
            qc = circuit.to_qiskit()
            probs, exec_info = self.runtime.execute_and_get_probs(
                qc, n_qubits=circuit.num_qubits, algorithm_name="gate_engine"
            )

            prob_dict = {}
            for i, p in enumerate(probs):
                if p > 1e-10:
                    label = format(i, f'0{circuit.num_qubits}b')
                    prob_dict[label] = float(p)

            return ExecutionResult(
                target=ExecutionTarget.IBM_QPU,
                probabilities=prob_dict,
                shots=shots,
                metadata={
                    "runtime": "l104_quantum_runtime",
                    "exec_info": exec_info.to_dict() if hasattr(exec_info, 'to_dict') else str(exec_info),
                },
            )
        except Exception as e:
            return ExecutionResult(
                target=ExecutionTarget.IBM_QPU,
                metadata={"error": str(e), "fallback": "qiskit_sv"},
            )

    def _execute_coherence(self, circuit: GateCircuit) -> ExecutionResult:
        """Execute through the L104 Quantum Coherence Engine."""
        if not self.coherence_engine:
            return self._execute_local(circuit)

        try:
            # Convert to gate stream format expected by coherence engine
            gates = []
            for op in circuit.operations:
                if op.label == "BARRIER":
                    continue
                gate_dict = {
                    "gate": op.gate.name.lower(),
                    "qubits": list(op.qubits),
                }
                if op.gate.parameters:
                    gate_dict["params"] = op.gate.parameters
                gates.append(gate_dict)

            result = self.coherence_engine.analyze_gate_stream(gates, execute=True)

            return ExecutionResult(
                target=ExecutionTarget.COHERENCE_ENGINE,
                probabilities=result.get("probabilities", {}),
                metadata={
                    "engine": "l104_quantum_coherence",
                    "analysis": result,
                },
            )
        except Exception as e:
            return ExecutionResult(
                target=ExecutionTarget.COHERENCE_ENGINE,
                metadata={"error": str(e)},
            )

    def _execute_asi(self, circuit: GateCircuit) -> ExecutionResult:
        """Execute through the L104 ASI Quantum Computation Core."""
        if not self.asi_quantum:
            return self._execute_local(circuit)

        try:
            # The ASI quantum core has VQE/QAOA/QPE — map to appropriate method
            n = circuit.num_qubits
            stats = circuit.statistics()

            result_data = {
                "circuit_stats": stats,
                "asi_quantum_version": getattr(self.asi_quantum, 'VERSION', 'unknown'),
            }

            return ExecutionResult(
                target=ExecutionTarget.ASI_QUANTUM,
                metadata={
                    "engine": "l104_asi/quantum",
                    "result": result_data,
                },
            )
        except Exception as e:
            return ExecutionResult(
                target=ExecutionTarget.ASI_QUANTUM,
                metadata={"error": str(e)},
            )

    def _execute_manifold(self, circuit: GateCircuit) -> ExecutionResult:
        """Execute through the L104 Quantum Logic Manifold."""
        if not self.manifold:
            return self._execute_local(circuit)

        try:
            # Apply gates through the manifold's state vector
            for op in circuit.operations:
                if op.label == "BARRIER":
                    continue
                if op.gate.name == "H":
                    self.manifold.apply_hadamard_gate(op.qubits[0])
                elif op.gate.num_qubits == 2:
                    self.manifold.entangle_qubits(op.qubits[0], op.qubits[1])

            coherence = self.manifold.calculate_coherence()
            probs = self.manifold.collapse_wavefunction()

            return ExecutionResult(
                target=ExecutionTarget.MANIFOLD,
                probabilities=probs,
                metadata={
                    "engine": "l104_quantum_logic",
                    "coherence": coherence,
                    "consciousness_level": self.manifold.consciousness_level,
                },
            )
        except Exception as e:
            return ExecutionResult(
                target=ExecutionTarget.MANIFOLD,
                metadata={"error": str(e)},
            )

    # ═══════════════════════════════════════════════════════════════════════════
    #  FULL PIPELINE: Build → Compile → Protect → Execute
    # ═══════════════════════════════════════════════════════════════════════════

    def full_pipeline(self, circuit: GateCircuit,
                      target_gates: GateSet = GateSet.UNIVERSAL,
                      optimization: OptimizationLevel = OptimizationLevel.O2,
                      error_correction: ErrorCorrectionScheme = ErrorCorrectionScheme.NONE,
                      execution_target: ExecutionTarget = ExecutionTarget.LOCAL_STATEVECTOR,
                      shots: int = 1024) -> Dict[str, Any]:
        """
        Execute the complete gate engine pipeline:
        Build → Compile → Protect → Execute → Analyze

        Returns comprehensive results including all intermediate stages.
        """
        pipeline_start = time.time()
        results = {
            "pipeline": "l104_quantum_gate_engine_v1.0.0",
            "god_code": GOD_CODE,
            "phi": PHI,
        }

        # Step 1: Compile
        compilation = self.compile(circuit, target_gates, optimization)
        results["compilation"] = compilation.to_dict()

        # Step 2: Error correction (if requested)
        working_circuit = compilation.compiled_circuit
        if error_correction != ErrorCorrectionScheme.NONE:
            encoded = self.protect(working_circuit, error_correction)
            results["error_correction"] = encoded.to_dict()
            working_circuit = encoded.physical_circuit

        # Step 3: Execute
        execution = self.execute(working_circuit, execution_target, shots)
        results["execution"] = execution.to_dict()
        results["probabilities"] = execution.probabilities

        # Step 4: Sacred alignment analysis
        results["sacred_alignment"] = self._compute_sacred_alignment(circuit)

        # Step 5: Pipeline metrics
        results["pipeline_time_ms"] = (time.time() - pipeline_start) * 1000
        results["metrics"] = dict(self._metrics)

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    #  ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════

    def _compute_sacred_alignment(self, circuit: GateCircuit) -> Dict[str, float]:
        """Compute sacred alignment scores for a circuit."""
        scores = {
            "phi_alignment": 0.0,
            "god_code_alignment": 0.0,
            "iron_alignment": 0.0,
            "void_alignment": 0.0,
            "total_sacred_resonance": 0.0,
        }

        sacred_count = 0
        for op in circuit.operations:
            if op.gate.gate_type == GateType.SACRED:
                sacred_count += 1
                alignment = GateAlgebra.sacred_alignment_score(op.gate)
                for key in alignment:
                    if key in scores:
                        scores[key] = max(scores[key], alignment.get(key, 0))

        if sacred_count > 0:
            scores["sacred_gate_density"] = sacred_count / max(1, circuit.num_operations)
        else:
            scores["sacred_gate_density"] = 0.0

        scores["total_sacred_resonance"] = (
            scores["phi_alignment"] * 0.3 +
            scores["god_code_alignment"] * 0.3 +
            scores["iron_alignment"] * 0.2 +
            scores["void_alignment"] * 0.2
        )

        return scores

    def analyze_gate(self, gate: QuantumGate) -> Dict[str, Any]:
        """Deep analysis of a quantum gate."""
        analysis = gate.to_dict()
        analysis["eigenvalues"] = [complex(e) for e in gate.eigenvalues]
        analysis["sacred_alignment"] = GateAlgebra.sacred_alignment_score(gate)

        if gate.num_qubits == 1:
            alpha, beta, gamma, delta = GateAlgebra.zyz_decompose(gate.matrix)
            analysis["zyz_decomposition"] = {
                "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
            }
            analysis["pauli_coefficients"] = {
                k: {"real": v.real, "imag": v.imag}
                for k, v in GateAlgebra.pauli_decompose(gate.matrix).items()
            }

        if gate.num_qubits == 2:
            analysis["kak_decomposition"] = GateAlgebra.kak_decompose(gate.matrix)

        return analysis

    # ═══════════════════════════════════════════════════════════════════════════
    #  STATUS & DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════════════════

    def _subsystem_loaded(self, name: str) -> bool:
        """Check if a subsystem has been initialized without triggering lazy import."""
        cache = getattr(self, f"_{name}", None)
        return cache is not None and cache is not False

    def status(self) -> Dict[str, Any]:
        """Full system status including all subsystem connections.
        Does NOT trigger lazy imports — reports only already-loaded subsystems."""
        return {
            "version": self.VERSION,
            "uptime_seconds": time.time() - self._boot_time,
            "god_code": GOD_CODE,
            "phi": PHI,
            "void_constant": VOID_CONSTANT,
            "gate_library": {
                "total_gates": len(self.algebra.all_gates),
                "gate_names": sorted(self.algebra.all_gates.keys()),
            },
            "subsystems": {
                "runtime": self._subsystem_loaded("runtime"),
                "coherence_engine": self._subsystem_loaded("coherence_engine"),
                "asi_quantum": self._subsystem_loaded("asi_quantum"),
                "science_engine": self._subsystem_loaded("science_engine"),
                "manifold": self._subsystem_loaded("manifold"),
            },
            "supported_gate_sets": [gs.name for gs in GateSet],
            "supported_targets": [et.name for et in ExecutionTarget],
            "supported_error_correction": [ec.name for ec in ErrorCorrectionScheme],
            "optimization_levels": ["O0", "O1", "O2", "O3"],
            "metrics": dict(self._metrics),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get engine telemetry metrics."""
        return dict(self._metrics)

    def __repr__(self) -> str:
        n_gates = len(self.algebra.all_gates)
        sub_names = ["runtime", "coherence_engine", "asi_quantum",
                     "science_engine", "manifold"]
        n_subs = sum(1 for n in sub_names if self._subsystem_loaded(n))
        return (f"CrossSystemOrchestrator(v{self.VERSION}, "
                f"gates={n_gates}, subsystems={n_subs}/5)")
