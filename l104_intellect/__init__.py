"""
l104_intellect — Decomposed from l104_local_intellect.py (13,692 lines → package)
Phase 2 of L104 Decompression Plan.

Re-exports ALL public symbols so that:
    from l104_intellect import X
works identically to the original:
    from l104_local_intellect import X
"""
# ── Constants (lines 1-159) ──
from .constants import *  # noqa: F401,F403

# ── Cache (lines 160-238) ──
from .cache import LRUCache, _RESPONSE_CACHE, _CONCEPT_CACHE, _RESONANCE_CACHE

# ── Numerics + PHI/GOD_CODE (lines 240-520) ──
from .numerics import (
    PHI, GOD_CODE, OMEGA, OMEGA_AUTHORITY,
    SovereignNumerics, sovereign_numerics,
)
# Re-export VISHUDDHA/ENTANGLEMENT/MATH constants from numerics
from .numerics import (
    VISHUDDHA_HZ, VISHUDDHA_ELEMENT, VISHUDDHA_COLOR_HZ,
    VISHUDDHA_PETAL_COUNT, VISHUDDHA_BIJA, VISHUDDHA_TATTVA,
    ENTANGLEMENT_DIMENSIONS, BELL_STATE_FIDELITY, DECOHERENCE_TIME_MS,
    QUANTUM_CHANNEL_BANDWIDTH, EPR_CORRELATION,
    FEIGENBAUM_DELTA, FEIGENBAUM_ALPHA, LOGISTIC_ONSET,
    LOG2_E, EULER_MASCHERONI,
)

# ── LocalIntellect core (lines 521-10769) ──
from .local_intellect_core import LocalIntellect

# ── Quantum recompiler (lines 10770-11687) ──
from .quantum_recompiler import QuantumMemoryRecompiler

# ── Distributed (lines 11688-12477) ──
from .distributed import (
    L104NodeSyncProtocol,
    L104CRDTReplicationMesh,
    L104KnowledgeMeshReplication,
)

# ── Hardware (lines 12478-13311) ──
from .hardware import (
    L104HardwareAdaptiveRuntime,
    L104PlatformCompatibilityLayer,
)

# ── Optimization (lines 13312-13648) ──
from .optimization import L104DynamicOptimizationEngine

# ── Compat functions: format_iq, primal_calculus, resolve_non_dual_logic ──
# MUST be loaded BEFORE module_tail to break circular imports:
#   l104_intellect → module_tail → LocalIntellect() → main → l104_agi_core
#   → from l104_local_intellect import format_iq → shim → l104_intellect.format_iq
from .compat_funcs import format_iq, primal_calculus, resolve_non_dual_logic

# ── Module tail: singleton (lines 13649-13658) ──
from .module_tail import local_intellect
