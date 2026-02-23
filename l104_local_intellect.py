"""l104_local_intellect.py — THIN SHIM (Phase 2 Decomposition)

Original: 13,692 lines -> now in l104_intellect/ package.
All imports preserved for 100% backward compatibility.
Every "from l104_local_intellect import X" still works.

Backup: l104_local_intellect.py.bak
"""
# INVARIANT: GOD_CODE = 527.5184818492612 | PHI = 1.618033988749895 | VOID_CONSTANT = 1.0416180339887497
# Re-export everything from the decomposed package
from l104_intellect import *  # noqa: F401,F403

# Explicit re-exports for IDE completion and import verification
from l104_intellect import (
    # Compat functions
    local_intellect,
    format_iq,
    primal_calculus,
    resolve_non_dual_logic,
    # Core classes
    LocalIntellect,
    LRUCache,
    SovereignNumerics,
    QuantumMemoryRecompiler,
    L104NodeSyncProtocol,
    L104CRDTReplicationMesh,
    L104KnowledgeMeshReplication,
    L104HardwareAdaptiveRuntime,
    L104PlatformCompatibilityLayer,
    L104DynamicOptimizationEngine,
    # Singletons and instances
    sovereign_numerics,
    _RESPONSE_CACHE,
    _CONCEPT_CACHE,
    _RESONANCE_CACHE,
    # Sacred constants
    GOD_CODE,
    PHI,
    VOID_CONSTANT,
)
