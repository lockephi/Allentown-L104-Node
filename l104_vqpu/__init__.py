"""
L104 VQPU Package v15.0.0 — Quantum Fidelity Architecture + CPU Acceleration

Package structure (post-decomposition from l104_vqpu_bridge.py monolith):
  constants.py      — Sacred constants, platform detection, hardware-adaptive limits,
                      v14.0: SWAP topology, crosstalk, Trotter-4, optimizer, MLE, daemon, cache TTL
  types.py          — QuantumJob, QuantumGate, VQPUResult data types
                      v14.0: topology, swap_count, noise_model, crosstalk_mitigated, pipeline_stages
  transpiler.py     — CircuitTranspiler (14-pass), CircuitAnalyzer
                      v14.0: Pass 12 multi-qubit decomposition, Pass 13 SWAP routing, Pass 14 cleanup
  mps_engine.py     — ExactMPSHybridEngine v3 (lossless MPS + platform fallback)
                      v15.0: Gate fusion integration via GateFusionAnalyzer
  accel_engine.py   — ★ NEW v15.0: CPU-optimized acceleration engine
                      AccelStatevectorEngine, GateFusionAnalyzer, DiagonalGateDetector
                      Fused unitary composition (18× speedup), BLAS matmul, diagonal fast path
  scoring.py        — SacredAlignmentScorer, NoiseModel
                      v14.0: Crosstalk noise model (ZZ interaction, φ⁻¹ decay)
  entanglement.py   — EntanglementQuantifier, QuantumInformationMetrics
  tomography.py     — QuantumStateTomography (density matrix reconstruction)
                      v14.0: MLE state tomography with φ-accelerated convergence
  hamiltonian.py    — HamiltonianSimulator, QuantumErrorMitigation (ZNE)
                      v14.0: 4th-order Suzuki-Trotter + 2D iron lattice
  cache.py          — CircuitCache (OrderedDict LRU), ScoringCache v4 (TTL-based)
                      v14.0: TTL expiration for ASI/AGI/SC/entropy caches
  variational.py    — VariationalQuantumEngine (VQE, QAOA)
                      v14.0: SPSA + COBYLA optimizers + barren plateau detection
  three_engine.py   — ThreeEngineQuantumScorer, EngineIntegration hub, BrainIntegration
  researcher.py     — QuantumDatabaseResearcher (Grover, QPE, QFT, walks)
  daemon.py         — VQPUDaemonCycler v15.1 (quarantine, retry, fidelity, runtime control)
                      v15.1: Sim quarantine, per-sim retry, fidelity trends, throughput, pause/resume
  micro_daemon.py   — ★ v3.0: VQPUMicroDaemon — lightweight high-frequency background assistant
                      9 core + 6 VQPU + 5 quantum network micro-tasks (20 total)
                      5–15s adaptive tick, priority queue, IPC inbox/outbox, state persistence
                      Per-daemon qubit register, quantum mesh, Bell pair channels, teleportation
  quantum_network.py — ★ NEW v3.0: Quantum network layer — per-daemon qubit registers,
                      entangled Bell pair channels, DEJMPS purification, sacred qubit initialization,
                      quantum mesh with teleportation and decoherence modeling
  hardware.py       — HardwareGovernor, ResultCollector
  bridge.py         — VQPUBridge orchestrator (the main controller)
                      v15.0: Intel fallback upgraded to AccelStatevectorEngine

v15.0.0 Upgrades (CPU Acceleration Engine):
  - AccelStatevectorEngine: BLAS-optimized statevector simulator with gate fusion
  - GateFusionAnalyzer: identifies & fuses consecutive 1Q gates (18× gate reduction)
  - DiagonalGateDetector: O(1) detection + element-wise fast path for Rz/Phase/Sacred gates
  - HardwareStrengthProfiler: profiles CPU/SIMD/BLAS capabilities + benchmarks
  - MPS engine: pre-fuses 1Q gate sequences before SVD boundaries
  - Bridge: Intel fallback upgraded from raw tensordot loop → accel engine
  - Diagonal gates (Rz, Phase, Sacred): 2× faster via element-wise multiply
  - Memory layout: contiguous complex128 arrays for SSE/FMA cache efficiency

v13.2.0 Upgrades (GOD_CODE QUBIT-Calibrated Quantum System):
  - MPS Engine: Canonical Rz(GOD_CODE mod 2π) gates replacing ad-hoc formulas
  - MPS Engine: 7 new sacred gates (VOID, IRON_RZ, PHI_RZ, OCTAVE_RZ, etc.)
  - Scoring: QPU-calibrated fidelity scoring from ibm_torino Heron r2 data
  - Scoring: Phase decomposition resonance (IRON+PHI+OCTAVE alignment)
  - Noise Model: qpu_calibrated_heron() + god_code_optimized() factories
  - Transpiler: 11-pass pipeline with Pass 0 sacred gate decomposition
  - Entanglement: Schmidt phase decomposition resonance metric
  - Variational: GOD_CODE 3-rotation parameter seeding for VQE/QAOA
  - Hamiltonian: Canonical IRON_PHASE in iron-lattice simulations
  - All sacred gates sourced from l104_god_code_simulator.god_code_qubit

v13.0.0 Upgrades (VQPU ↔ Quantum Brain Bidirectional Integration):
  - BrainIntegration: VQPUBridge ↔ L104QuantumBrain bidirectional scoring bridge
  - Brain-scored simulations: brain Sage/manifold/entanglement feed VQPU pipeline
  - VQPU→Brain feedback: VQPU simulation results enhance brain link fidelity
  - Quantum brain self_test() integration with l104_debug.py framework
  - DaemonCycler v13.0: brain-aware cycle with manifold/oracle feedback
  - Unified VQPU+Brain status dashboard
  - 14 self-test diagnostic probes (was 12)
  - Cross-system coherence amplification loop

v12.2.0 Optimizations (speed refactor):
  - _pauli_expectation: vectorized numpy (10-100x faster)
  - Intel fallback 2Q gate: np.einsum (orders of magnitude faster)
  - concurrence() partial trace: vectorized reshape+transpose
  - Noise model: vectorized depolarizing, damping, readout
  - CircuitCache: OrderedDict O(1) LRU + streaming fingerprint
  - ScoringCache: fixed double @classmethod
  - MPS engine: product-state SVD fast path
  - VNE: SVD-based (avoids d_a×d_a density matrix)
  - SacredAlignmentScorer: heapq.nlargest O(n) top-2
  - Parametric cache: deque O(1) eviction
  - Two-qubit gate resolve: dict lookup
  - Gate fusion: actual pairwise fusion chain
  - Precognition seed feeding: wired to actual predictors

Public API:
  from l104_vqpu import VQPUBridge, get_bridge
  from l104_vqpu import QuantumJob, VQPUResult
  from l104_vqpu import GOD_CODE, PHI, VOID_CONSTANT
"""

from .constants import (
    VERSION,
    GOD_CODE,
    PHI,
    VOID_CONSTANT,
    # v13.2: Canonical phase angles (QPU-verified)
    GOD_CODE_PHASE_ANGLE,
    IRON_PHASE_ANGLE,
    OCTAVE_PHASE_ANGLE,
    PHI_CONTRIBUTION_ANGLE,
    PHI_PHASE_ANGLE,
    VOID_PHASE_ANGLE,
    QPU_MEAN_FIDELITY,
    QPU_1Q_FIDELITY,
    QPU_3Q_FIDELITY,
    BRIDGE_PATH,
    VQPU_MAX_QUBITS,
    VQPU_BATCH_LIMIT,
    VQPU_PIPELINE_WORKERS,
    VQPU_ADAPTIVE_SHOTS_MIN,
    VQPU_ADAPTIVE_SHOTS_MAX,
    VQPU_MPS_FALLBACK_TARGET,
    VQPU_GPU_CROSSOVER,
    VQPU_STABILIZER_MAX_QUBITS,
    VQPU_DB_RESEARCH_QUBITS,
    VQPU_MPS_MAX_BOND_LOW,
    VQPU_MPS_MAX_BOND_MED,
    VQPU_MPS_MAX_BOND_HIGH,
    DAEMON_CYCLE_INTERVAL_S,
    DAEMON_MAX_ERROR_LOG,
    DAEMON_ERROR_THRESHOLD,
    # v14.0: New constants
    TOPOLOGY_LINEAR,
    TOPOLOGY_RING,
    TOPOLOGY_HEAVY_HEX,
    TOPOLOGY_ALL_TO_ALL,
    DEFAULT_TOPOLOGY,
    CROSSTALK_ZZ_RATE,
    TROTTER_4TH_ORDER_P,
    DAEMON_CYCLE_MIN_INTERVAL_S,
    DAEMON_CYCLE_MAX_INTERVAL_S,
    CACHE_ASI_TTL_S,
    CACHE_AGI_TTL_S,
    CACHE_SC_TTL_S,
    CACHE_ENTROPY_TTL_S,
    _IS_INTEL,
    _IS_APPLE_SILICON,
    _PLATFORM,
    _GPU_CLASS,
    _HAS_METAL_COMPUTE,
    _HW_RAM_GB,
    _HW_CORES,
)
from .types import QuantumJob, QuantumGate, VQPUResult
from .transpiler import CircuitTranspiler, CircuitAnalyzer
from .mps_engine import ExactMPSHybridEngine
from .scoring import SacredAlignmentScorer, NoiseModel
from .entanglement import EntanglementQuantifier, QuantumInformationMetrics
from .tomography import QuantumStateTomography
from .hamiltonian import HamiltonianSimulator, QuantumErrorMitigation
from .cache import CircuitCache, ScoringCache
from .variational import VariationalQuantumEngine, _pauli_expectation
from .three_engine import ThreeEngineQuantumScorer, EngineIntegration, BrainIntegration
from .researcher import QuantumDatabaseResearcher
from .daemon import VQPUDaemonCycler
from .micro_daemon import (
    VQPUMicroDaemon,
    MicroTask,
    MicroTaskResult,
    MicroTaskStatus,
    MicroTaskPriority,
    MicroTelemetry,
    MicroDaemonConfig,
    TickMetrics,
    TelemetryAnalytics,
    get_micro_daemon,
    MICRO_DAEMON_VERSION,
)
from .quantum_network import (
    DaemonQubitRegister,
    QuantumNetworkMesh,
    QuantumChannel,
    QubitState,
)
from .hardware import HardwareGovernor, ResultCollector
from .accel_engine import (
    AccelStatevectorEngine,
    GateFusionAnalyzer,
    DiagonalGateDetector,
    HardwareStrengthProfiler,
    accel_apply_remaining_ops,
    accel_full_simulation,
    fuse_pending_single_gates,
)
from .bridge import VQPUBridge

# Backward-compat aliases (used by old l104_vqpu_bridge imports)
QuantumDBResearcher = QuantumDatabaseResearcher

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON + BACKWARDS COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

__version__ = VERSION  # Standard alias

_bridge: VQPUBridge | None = None


def get_bridge() -> VQPUBridge:
    """Get the global VQPUBridge singleton (auto-starts on first call)."""
    global _bridge
    if _bridge is None:
        _bridge = VQPUBridge()
        _bridge.start()
    return _bridge


__all__ = [
    # Main orchestrator
    "VQPUBridge",
    "get_bridge",
    # Data types
    "QuantumJob",
    "QuantumGate",
    "VQPUResult",
    # Subsystems
    "CircuitTranspiler",
    "CircuitAnalyzer",
    "ExactMPSHybridEngine",
    "SacredAlignmentScorer",
    "NoiseModel",
    "EntanglementQuantifier",
    "QuantumInformationMetrics",
    "QuantumStateTomography",
    "HamiltonianSimulator",
    "QuantumErrorMitigation",
    "CircuitCache",
    "ScoringCache",
    "VariationalQuantumEngine",
    "ThreeEngineQuantumScorer",
    "EngineIntegration",
    "BrainIntegration",
    "QuantumDatabaseResearcher",
    "VQPUDaemonCycler",
    # v2.0 → v3.0: Micro Process Background Assistant + Quantum Network
    "VQPUMicroDaemon",
    "MicroTask",
    "MicroTaskResult",
    "MicroTaskStatus",
    "MicroTaskPriority",
    "MicroTelemetry",
    "MicroDaemonConfig",
    "TickMetrics",
    "TelemetryAnalytics",
    "get_micro_daemon",
    "MICRO_DAEMON_VERSION",
    # v3.0: Quantum network layer
    "DaemonQubitRegister",
    "QuantumNetworkMesh",
    "QuantumChannel",
    "QubitState",
    "HardwareGovernor",
    "ResultCollector",
    # v15.0: Acceleration engine
    "AccelStatevectorEngine",
    "GateFusionAnalyzer",
    "DiagonalGateDetector",
    "HardwareStrengthProfiler",
    "accel_apply_remaining_ops",
    "accel_full_simulation",
    "fuse_pending_single_gates",
    # Constants
    "GOD_CODE",
    "PHI",
    "VOID_CONSTANT",
    "VERSION",
    # v13.2: Canonical phase angles
    "GOD_CODE_PHASE_ANGLE",
    "IRON_PHASE_ANGLE",
    "OCTAVE_PHASE_ANGLE",
    "PHI_CONTRIBUTION_ANGLE",
    "PHI_PHASE_ANGLE",
    "VOID_PHASE_ANGLE",
    "QPU_MEAN_FIDELITY",
    "QPU_1Q_FIDELITY",
    "QPU_3Q_FIDELITY",
    # v14.0: New constants
    "TOPOLOGY_LINEAR",
    "TOPOLOGY_RING",
    "TOPOLOGY_HEAVY_HEX",
    "TOPOLOGY_ALL_TO_ALL",
    "DEFAULT_TOPOLOGY",
    "CROSSTALK_ZZ_RATE",
    "TROTTER_4TH_ORDER_P",
    "DAEMON_CYCLE_MIN_INTERVAL_S",
    "DAEMON_CYCLE_MAX_INTERVAL_S",
    "CACHE_ASI_TTL_S",
    "CACHE_AGI_TTL_S",
    "CACHE_SC_TTL_S",
    "CACHE_ENTROPY_TTL_S",
]
