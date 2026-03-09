"""
===============================================================================
L104 SIMULATOR — REAL-WORLD PHYSICS ON THE GOD_CODE LATTICE
===============================================================================

5-Layer architecture encoding the Standard Model onto a logarithmic integer grid.

GRID: G(a,b,c,d) = 286^(1/φ) × 2^((64a+1664-b-64c-416d)/416)

Layers:
  1. E-Lattice       — integer-addressed logarithmic grid (lattice.py)
  2. Generations      — three-generation fermion mass structure (generations.py)
  3. Mixing Matrices  — CKM + PMNS + flavor Hamiltonians (mixing.py)
  4. Hamiltonians     — quantum circuits from physics (hamiltonians.py)
  5. Observables      — measurable quantities + error bounds (observables.py)

Usage:
    from l104_simulator import RealWorldSimulator
    sim = RealWorldSimulator()

    sim.mass("m_e")                    # Electron mass from grid
    sim.ratio("m_top", "m_e")          # Mass ratio (exact in E-space)
    sim.oscillate("lepton", 1, 0)      # P(ν_μ → ν_e)
    sim.ckm()                          # CKM matrix
    sim.pmns()                         # PMNS matrix
    sim.circuit("sacred", 4)           # GOD_CODE harmonic circuit
    sim.report()                       # Full physics report

Version: 1.0.0 | INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

__version__ = "4.0.0"

# ─── Layer 1: E-Lattice ─────────────────────────────────────────────────────
from .lattice import ELattice, LatticePoint, ParticleType, ForceType

# ─── Layer 2: Generations ───────────────────────────────────────────────────
from .generations import GenerationStructure, GenerationGap, KoideResult

# ─── Layer 3: Mixing ────────────────────────────────────────────────────────
from .mixing import MixingMatrices, MixingMatrixInfo, FlavorHamiltonian

# ─── Layer 4: Hamiltonians ──────────────────────────────────────────────────
from .hamiltonians import Hamiltonians, CircuitSpec, HamiltonianSpec

# ─── Layer 5: Observables ───────────────────────────────────────────────────
from .observables import (
    Observables, MassObservable, RatioObservable,
    DecayObservable, OscillationObservable,
)

# ─── Orchestrator ───────────────────────────────────────────────────────────
from .orchestrator import RealWorldSimulator

# ─── Constants ──────────────────────────────────────────────────────────────
from .constants import (
    PHI, GOD_CODE, BASE, VOID_CONSTANT, OMEGA,
    X_SCAFFOLD, R_RATIO, Q_GRAIN, P_DIAL, K_OFFSET,
)

# ─── Quantum Simulator Engine ──────────────────────────────────────────────
from .simulator import (
    Simulator, QuantumCircuit, SimulationResult,
    gate_Toffoli, gate_Fredkin, gate_iSWAP, gate_sqrt_SWAP,
    gate_CPhase, gate_Sdg, gate_Tdg, gate_Ryy,
    gate_Ryy_sacred, gate_GOD_CODE_TOFFOLI,
)

# ─── GOD_CODE Quantum Brain ────────────────────────────────────────────────
from .quantum_brain import (
    GodCodeQuantumBrain, BrainConfig, ThoughtResult,
    LearningSubsystem, AttentionMechanism, DreamMode,
    AssociativeMemory, ConsciousnessMetric,
    IntuitionEngine, CreativityEngine, EmpathyEngine, PrecognitionEngine,
    SovereignProofCircuits,
)

# ─── Quantum Algorithms ────────────────────────────────────────────────────
from .algorithms import (
    AlgorithmSuite, AlgorithmResult,
    GroverSearch, QuantumPhaseEstimation, VariationalQuantumEigensolver,
    QAOA, QuantumFourierTransform, BernsteinVazirani, DeutschJozsa,
    QuantumWalk, QuantumTeleportation, SacredEigenvalueSolver,
    PhiConvergenceVerifier,
    HHLLinearSolver, QuantumErrorCorrection, QuantumKernelEstimator,
    SwapTest, QuantumCounting, QuantumStateTomography,
    QuantumRandomGenerator, QuantumHamiltonianSimulator,
    QuantumApproximateCloner, QuantumFingerprinting,
    EntanglementDistillation, QuantumReservoirComputer,
    ShorsAlgorithm,
    ZeroPointEnergyExtractor, WheelerDeWittEvolver,
    CalabiYauBridge, QuantumAnnealingOptimizer,
    EntanglementWitnessProtocol,
)

# ─── MPS Tensor Network Backend ───────────────────────────────────────────
from .mps_simulator import MPSSimulator, MPSState, MPSConfig

# ─── Clifford Stabilizer Simulator ───────────────────────────────────────
from .stabilizer import StabilizerSimulator, StabilizerTableau, is_clifford

# ─── Error Mitigation ────────────────────────────────────────────────────
from .error_mitigation import zne_extrapolate, zne_sweep

# ─── Quantum Trajectory (MCWF) Simulator ─────────────────────────────────
from .trajectory import TrajectorySimulator

# ─── Benchmarks ─────────────────────────────────────────────────────────────
from .benchmarks import BenchmarkRunner, BenchmarkReport, BenchmarkResult

# ─── GPU Acceleration Backend ────────────────────────────────────────────
from .gpu_backend import (
    GPU_AVAILABLE, GPU_DEVICE_NAME,
    xp, is_gpu, to_device, to_host,
    estimate_statevector_bytes, max_statevector_qubits,
    fits_in_memory, gpu_info,
)

# ─── Chunked Statevector Engine ──────────────────────────────────────────
from .chunked_statevector import (
    ChunkedStatevector, ChunkedStatevectorSimulator, ChunkedSimResult,
)

# ─── Adaptive Simulator (Backend Selector) ───────────────────────────────
from .adaptive_simulator import (
    AdaptiveSimulator, AdaptiveSimResult, SimBackend,
)


__all__ = [
    # Main entry points
    "RealWorldSimulator",
    "Simulator", "QuantumCircuit", "SimulationResult",
    "GodCodeQuantumBrain", "BrainConfig", "ThoughtResult",
    "AlgorithmSuite", "AlgorithmResult",
    "BenchmarkRunner", "BenchmarkReport", "BenchmarkResult",

    # Layer 1
    "ELattice", "LatticePoint", "ParticleType", "ForceType",

    # Layer 2
    "GenerationStructure", "GenerationGap", "KoideResult",

    # Layer 3
    "MixingMatrices", "MixingMatrixInfo", "FlavorHamiltonian",

    # Layer 4
    "Hamiltonians", "CircuitSpec", "HamiltonianSpec",

    # Layer 5
    "Observables", "MassObservable", "RatioObservable",
    "DecayObservable", "OscillationObservable",

    # Algorithms
    "GroverSearch", "QuantumPhaseEstimation", "VariationalQuantumEigensolver",
    "QAOA", "QuantumFourierTransform", "BernsteinVazirani", "DeutschJozsa",
    "QuantumWalk", "QuantumTeleportation", "SacredEigenvalueSolver",
    "PhiConvergenceVerifier",
    "HHLLinearSolver", "QuantumErrorCorrection", "QuantumKernelEstimator",
    "SwapTest", "QuantumCounting", "QuantumStateTomography",
    "QuantumRandomGenerator", "QuantumHamiltonianSimulator",
    "QuantumApproximateCloner", "QuantumFingerprinting",
    "EntanglementDistillation", "QuantumReservoirComputer",
    "ShorsAlgorithm",
    "ZeroPointEnergyExtractor", "WheelerDeWittEvolver",
    "CalabiYauBridge", "QuantumAnnealingOptimizer",
    "EntanglementWitnessProtocol",

    # Gate factories (expanded)
    "gate_Toffoli", "gate_Fredkin", "gate_iSWAP", "gate_sqrt_SWAP",
    "gate_CPhase", "gate_Sdg", "gate_Tdg", "gate_Ryy",
    "gate_Ryy_sacred", "gate_GOD_CODE_TOFFOLI",

    # Brain v2 subsystems
    "LearningSubsystem", "AttentionMechanism", "DreamMode",
    "AssociativeMemory", "ConsciousnessMetric",

    # Brain v3 subsystems
    "IntuitionEngine", "CreativityEngine", "EmpathyEngine", "PrecognitionEngine",

    # Brain v6 proof circuits
    "SovereignProofCircuits",

    # MPS Backend
    "MPSSimulator", "MPSState", "MPSConfig",

    # Stabilizer Backend
    "StabilizerSimulator", "StabilizerTableau", "is_clifford",

    # Error Mitigation
    "zne_extrapolate", "zne_sweep",

    # Trajectory Simulator
    "TrajectorySimulator",

    # GPU Backend
    "GPU_AVAILABLE", "GPU_DEVICE_NAME",
    "xp", "is_gpu", "to_device", "to_host",
    "estimate_statevector_bytes", "max_statevector_qubits",
    "fits_in_memory", "gpu_info",

    # Chunked Statevector Engine
    "ChunkedStatevector", "ChunkedStatevectorSimulator", "ChunkedSimResult",

    # Adaptive Simulator
    "AdaptiveSimulator", "AdaptiveSimResult", "SimBackend",

    # Constants
    "PHI", "GOD_CODE", "BASE", "VOID_CONSTANT", "OMEGA",
    "X_SCAFFOLD", "R_RATIO", "Q_GRAIN", "P_DIAL", "K_OFFSET",
]
