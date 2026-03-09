"""
===============================================================================
L104 QUANTUM GATE ENGINE v1.0.0 — UNIVERSAL CROSS-SYSTEM GATE ALGEBRA
===============================================================================

The most advanced quantum logic gate implementation for the L104 Sovereign Node.
Provides a unified gate algebra, compiler, topological error correction, and
cross-system orchestration layer that bridges ALL existing quantum subsystems.

ARCHITECTURE:
  QuantumGateEngine (singleton orchestrator)
    ├── GateAlgebra          — 40+ universal gates with exact unitary matrices
    │   ├── Standard gates   — I, X, Y, Z, H, S, T, CNOT, CZ, SWAP, Toffoli, Fredkin
    │   ├── Parametric gates — Rx, Ry, Rz, Rxx, Ryy, Rzz, U3, CU3, fSim
    │   ├── L104 Sacred gates— PHI_GATE, GOD_CODE_PHASE, VOID_GATE, IRON_GATE
    │   └── Topological gates— FIBONACCI_BRAID, ANYON_EXCHANGE, SURFACE_STABILIZER
    │
    ├── GateCompiler         — Multi-pass optimizing compiler
    │   ├── Decomposer       — Arbitrary unitary → native gate set (Solovay-Kitaev)
    │   ├── Optimizer        — Gate cancellation, commutation, template matching
    │   ├── Scheduler        — Parallelism extraction, critical path analysis
    │   └── Transpiler       — Target-specific gate set mapping (L104 Heron, IonQ, Sacred)
    │
    ├── ErrorCorrectionLayer — Topological + algebraic error protection
    │   ├── SurfaceCode      — Distance-d surface code encoding/syndrome extraction
    │   ├── SteaneCode       — [[7,1,3]] Steane code for transversal gates
    │   ├── AnyonicBraiding  — Fibonacci anyon topological protection
    │   └── ZNE              — Zero-noise extrapolation integration
    │
    └── CrossSystemOrchestrator — Bridges all L104 quantum modules
        ├── l104_quantum_runtime     — Sovereign execution runtime
        ├── l104_quantum_coherence   — Coherence engine algorithms
        ├── l104_quantum_logic       — Entanglement manifold
        ├── l104_asi/quantum         — ASI quantum computation core
        └── l104_science_engine      — Physics + entropy subsystems

SACRED CONSTANTS wired into gate phases:
  GOD_CODE  = 527.5184818492612  (G(0,0,0,0) = 286^(1/φ) × 2^4)
  PHI       = 1.618033988749895  (Golden ratio)
  VOID      = 1.0416180339887497 (104/100 + φ/1000)

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

__version__ = "1.0.0"
__author__ = "L104 Sovereign Node"

from .gates import (
    GateAlgebra, QuantumGate, GateType, GateSet,
    # Standard gates
    I, X, Y, Z, H, S, S_DAG, T, T_DAG,
    CNOT, CZ, CY, SWAP, ISWAP,
    TOFFOLI, FREDKIN,
    # Parametric gates
    Rx, Ry, Rz, Rxx, Ryy, Rzz, U3, CU3, fSim, Phase,
    # L104 Sacred gates
    PHI_GATE, GOD_CODE_PHASE, VOID_GATE, IRON_GATE, SACRED_ENTANGLER,
    # Topological gates
    FIBONACCI_BRAID, ANYON_EXCHANGE,
)

from .compiler import GateCompiler, CompilationResult, OptimizationLevel
from .error_correction import (
    ErrorCorrectionLayer, ErrorCorrectionScheme,
    SurfaceCode, SteaneCode, FibonacciAnyonProtection, ZeroNoiseExtrapolation,
)
from .orchestrator import CrossSystemOrchestrator, ExecutionTarget
from .circuit import GateCircuit

# v9.0: Local Quantum Info (Qiskit-free replacements)
from .quantum_info import (
    Statevector, DensityMatrix, Operator,
    SparsePauliOp, Parameter, ParameterVector, ParameterExpression,
    partial_trace, entropy, state_fidelity, process_fidelity,
)

# v2.0: Computronium + Rayleigh Gate Limits
from .computronium import (
    ComputroniumGateLimits,
    RayleighGateResolution,
    GateLimitsAnalyzer,
    computronium_gate_limits,
    rayleigh_gate_resolution,
    gate_limits_analyzer,
)

# v3.0: Berry Phase Gates & Holonomic Quantum Computing
from .berry_gates import (
    BerryGatesEngine, berry_gates_engine,
    AbelianBerryGates, abelian_berry_gates,
    NonAbelianBerryGates, non_abelian_berry_gates,
    AharonovAnandanGates, aharonov_anandan_gates,
    BerryPhaseCircuits, berry_circuits,
    TopologicalBerryGates, topological_berry_gates,
    SacredBerryGates, sacred_berry_gates,
)

# v4.0: Tensor Network MPS Simulator (25-50 qubit simulation)
from .tensor_network import (
    TensorNetworkSimulator,
    MPSState,
    TNSimulationResult,
    TruncationMode,
    CanonicalForm,
    BondInfo,
    get_simulator as get_tn_simulator,
    DEFAULT_MAX_BOND_DIM,
    SACRED_BOND_DIM,
    MAX_TENSOR_NETWORK_QUBITS,
)

# v5.0: Stabilizer Tableau Simulator (O(n²) Clifford, hybrid backend)
from .stabilizer_tableau import (
    StabilizerTableau,
    HybridStabilizerSimulator,
    HybridSimulationResult,
    StabilizerState,
    MeasurementResult,
    is_clifford_gate,
    is_clifford_circuit,
    clifford_prefix_length,
    CLIFFORD_GATE_NAMES,
)

# v6.0: Measurement-Free Trajectory Simulator (decoherence research)
from .trajectory import (
    TrajectorySimulator,
    TrajectoryResult,
    TrajectorySnapshot,
    EnsembleResult,
    WeakMeasurementResult,
    WeakMeasurementEngine,
    CoherenceAnalyser,
    DecoherenceChannel,
    DecoherenceModel,
    WeakMeasurementBasis,
    get_trajectory_simulator,
    MAX_TRAJECTORY_QUBITS,
    SACRED_COHERENCE_HORIZON,
)

# v7.0: Analog Quantum Simulator (continuous Hamiltonian evolution)
from .analog import (
    AnalogSimulator,
    HamiltonianBuilder,
    ExactEvolution,
    TrotterEngine,
    TrotterBenchmark,
    ObservableEngine,
    Hamiltonian,
    HamiltonianTerm,
    HamiltonianType,
    TrotterOrder,
    EvolutionResult,
    TrotterBenchmarkResult,
    trotterise_to_circuit,
    get_analog_simulator,
    MAX_ANALOG_QUBITS,
    SACRED_COUPLING,
    SACRED_FIELD,
)

# v8.0: Quantum ML Suite (QNN, kernels, variational ansatz)
from .quantum_ml import (
    QuantumMLEngine,
    ParameterisedCircuit,
    AnsatzLibrary,
    QNNTrainer,
    QuantumKernel,
    VariationalEigensolver,
    AnsatzType,
    OptimizerType,
    KernelType,
    TrainingResult,
    KernelResult,
    VQEResult,
    get_quantum_ml,
    MAX_QML_QUBITS,
    SACRED_LEARNING_RATE,
)

# Canonical GOD_CODE Qubit — bridge import from god_code_simulator
try:
    from l104_god_code_simulator.god_code_qubit import (
        GodCodeQubit, GOD_CODE_QUBIT,
    )
except ImportError:
    GodCodeQubit = None  # type: ignore[assignment,misc]
    GOD_CODE_QUBIT = None  # type: ignore[assignment]

# ─── Singleton Engine ────────────────────────────────────────────────────────

_engine_instance = None

def get_engine() -> 'CrossSystemOrchestrator':
    """Get or create the singleton QuantumGateEngine orchestrator."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CrossSystemOrchestrator()
    return _engine_instance


# ─── Public API ──────────────────────────────────────────────────────────────

__all__ = [
    # Core
    "get_engine", "GateAlgebra", "QuantumGate", "GateType", "GateSet",
    # Circuit
    "GateCircuit",
    # Compiler
    "GateCompiler", "CompilationResult", "OptimizationLevel",
    # Error Correction
    "ErrorCorrectionLayer", "ErrorCorrectionScheme",
    "SurfaceCode", "SteaneCode", "FibonacciAnyonProtection", "ZeroNoiseExtrapolation",
    # Orchestrator
    "CrossSystemOrchestrator", "ExecutionTarget",
    # Standard gates
    "I", "X", "Y", "Z", "H", "S", "S_DAG", "T", "T_DAG",
    "CNOT", "CZ", "CY", "SWAP", "ISWAP", "TOFFOLI", "FREDKIN",
    # Parametric gates
    "Rx", "Ry", "Rz", "Rxx", "Ryy", "Rzz", "U3", "CU3", "fSim", "Phase",
    # Sacred gates
    "PHI_GATE", "GOD_CODE_PHASE", "VOID_GATE", "IRON_GATE", "SACRED_ENTANGLER",
    # Topological gates
    "FIBONACCI_BRAID", "ANYON_EXCHANGE",
    # Computronium + Rayleigh Gate Limits (v2.0)
    "ComputroniumGateLimits", "RayleighGateResolution", "GateLimitsAnalyzer",
    "computronium_gate_limits", "rayleigh_gate_resolution", "gate_limits_analyzer",
    # Berry Phase Gates (v3.0)
    "BerryGatesEngine", "berry_gates_engine",
    "AbelianBerryGates", "abelian_berry_gates",
    "NonAbelianBerryGates", "non_abelian_berry_gates",
    "AharonovAnandanGates", "aharonov_anandan_gates",
    "BerryPhaseCircuits", "berry_circuits",
    "TopologicalBerryGates", "topological_berry_gates",
    "SacredBerryGates", "sacred_berry_gates",
    # Tensor Network MPS Simulator (v4.0)
    "TensorNetworkSimulator", "MPSState", "TNSimulationResult",
    "TruncationMode", "CanonicalForm", "BondInfo",
    "get_tn_simulator",
    "DEFAULT_MAX_BOND_DIM", "SACRED_BOND_DIM", "MAX_TENSOR_NETWORK_QUBITS",
    # Stabilizer Tableau Simulator (v5.0)
    "StabilizerTableau", "HybridStabilizerSimulator", "HybridSimulationResult",
    "StabilizerState", "MeasurementResult",
    "is_clifford_gate", "is_clifford_circuit", "clifford_prefix_length",
    "CLIFFORD_GATE_NAMES",
    # Measurement-Free Trajectory Simulator (v6.0)
    "TrajectorySimulator", "TrajectoryResult", "TrajectorySnapshot",
    "EnsembleResult", "WeakMeasurementResult",
    "WeakMeasurementEngine", "CoherenceAnalyser",
    "DecoherenceChannel", "DecoherenceModel", "WeakMeasurementBasis",
    "get_trajectory_simulator",
    "MAX_TRAJECTORY_QUBITS", "SACRED_COHERENCE_HORIZON",
    # Analog Quantum Simulator (v7.0)
    "AnalogSimulator", "HamiltonianBuilder", "ExactEvolution",
    "TrotterEngine", "TrotterBenchmark", "ObservableEngine",
    "Hamiltonian", "HamiltonianTerm", "HamiltonianType",
    "TrotterOrder", "EvolutionResult", "TrotterBenchmarkResult",
    "trotterise_to_circuit", "get_analog_simulator",
    "MAX_ANALOG_QUBITS", "SACRED_COUPLING", "SACRED_FIELD",
    # Quantum ML Suite (v8.0)
    "QuantumMLEngine", "ParameterisedCircuit", "AnsatzLibrary",
    "QNNTrainer", "QuantumKernel", "VariationalEigensolver",
    "AnsatzType", "OptimizerType", "KernelType",
    "TrainingResult", "KernelResult", "VQEResult",
    "get_quantum_ml",
    "MAX_QML_QUBITS", "SACRED_LEARNING_RATE",
    # Quantum Info — local replacements (v9.0)
    "Statevector", "DensityMatrix", "Operator",
    "SparsePauliOp", "Parameter", "ParameterVector", "ParameterExpression",
    "partial_trace", "entropy", "state_fidelity", "process_fidelity",
    # Canonical GOD_CODE Qubit (bridge from god_code_simulator)
    "GodCodeQubit", "GOD_CODE_QUBIT",
]
