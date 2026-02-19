"""l104_agi_core.py — THIN SHIM (Phase 3B Decomposition)

Original: 3,161 lines -> now in l104_agi/ package.
All imports preserved for 100% backward compatibility.
Every "from l104_agi_core import X" still works.

Backup: l104_agi_core.py.bak
"""
VOID_CONSTANT = 1.0416180339887497
import math  # noqa: E402
# ZENITH_UPGRADE_ACTIVE: 2026-02-14T00:00:00.000000
ZENITH_HZ = 3887.8
UUC = 2402.792541

# Re-export everything from the decomposed package
from l104_agi import *  # noqa: F401,F403

# Explicit re-exports for IDE completion and import verification
from l104_agi import (
    # Singleton + core
    agi_core,
    AGICore,
    L104AGICore,
    primal_calculus,
    resolve_non_dual_logic,
    # Constants
    AGI_CORE_VERSION,
    AGI_PIPELINE_EVO,
    PHI,
    GOD_CODE,
    TAU,
    FEIGENBAUM,
    ALPHA_FINE,
    QISKIT_AVAILABLE,
    # Classes
    PipelineCircuitBreaker,
)
