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
    QISKIT_AVAILABLE,
)

# Circuit breaker
from .circuit_breaker import PipelineCircuitBreaker

# Core + singleton + module-level functions
from .core import AGICore, agi_core, primal_calculus, resolve_non_dual_logic

# Alias for backward compatibility (2 importers use L104AGICore)
L104AGICore = AGICore
