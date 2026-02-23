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
    │   └── Transpiler       — Target-specific gate set mapping (IBM, IonQ, L104)
    │
    ├── ErrorCorrectionLayer — Topological + algebraic error protection
    │   ├── SurfaceCode      — Distance-d surface code encoding/syndrome extraction
    │   ├── SteaneCode       — [[7,1,3]] Steane code for transversal gates
    │   ├── AnyonicBraiding  — Fibonacci anyon topological protection
    │   └── ZNE              — Zero-noise extrapolation integration
    │
    └── CrossSystemOrchestrator — Bridges all L104 quantum modules
        ├── l104_quantum_runtime     — IBM QPU execution
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
]
