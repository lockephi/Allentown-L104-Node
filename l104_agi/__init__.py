"""
l104_agi — Decomposed from l104_agi_core.py (3,161 lines → package)
Phase 3B of L104 Decompression Plan.

Re-exports ALL public symbols so that:
    from l104_agi import X
works identically to the original:
    from l104_agi_core import X
"""
# Constants
from .constants import (
    AGI_CORE_VERSION, AGI_PIPELINE_EVO,
    PHI, GOD_CODE, TAU, FEIGENBAUM, ALPHA_FINE, VOID_CONSTANT,
    OMEGA, OMEGA_AUTHORITY,
    QISKIT_AVAILABLE,
)

# Circuit breaker
from .circuit_breaker import PipelineCircuitBreaker, BreakerState

# Identity Boundary — Sovereign Architectural Self-Declaration
from .identity_boundary import (
    AGIIdentityBoundary,
    L104_AGI_IS, L104_AGI_IS_NOT,
    AGI_CAPABILITY_MAP,
)

# Core + singleton + module-level functions
# v59.0: Computronium + Rayleigh Scoring Dimensions
from .computronium import (
    AGIComputroniumScoring,
    agi_computronium_scoring,
)

from .core import AGICore, agi_core, primal_calculus, resolve_non_dual_logic

# Alias for backward compatibility (2 importers use L104AGICore)
L104AGICore = AGICore

# v58.1 Benchmark Capability (delegates to l104_asi)
try:
    from l104_asi.benchmark_harness import BenchmarkHarness as _BenchmarkHarness
except ImportError:
    _BenchmarkHarness = None


__all__ = [
    # Constants
    "AGI_CORE_VERSION", "AGI_PIPELINE_EVO",
    "PHI", "GOD_CODE", "TAU", "FEIGENBAUM", "ALPHA_FINE", "VOID_CONSTANT",
    "OMEGA", "OMEGA_AUTHORITY", "QISKIT_AVAILABLE",
    # Circuit breaker
    "PipelineCircuitBreaker", "BreakerState",
    # Identity Boundary
    "AGIIdentityBoundary",
    "L104_AGI_IS", "L104_AGI_IS_NOT",
    "AGI_CAPABILITY_MAP",
    # Core
    "AGICore", "agi_core", "L104AGICore",
    # Computronium + Rayleigh (v59.0)
    "AGIComputroniumScoring", "agi_computronium_scoring",
    # Functions
    "primal_calculus", "resolve_non_dual_logic",
]
