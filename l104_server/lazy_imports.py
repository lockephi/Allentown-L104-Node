# L104 Server — Lazy Import System
# EVO_61: Quantum-grade deferred loading for heavy subsystems
# All subsystems load on first access, not at import time.
# This lets uvicorn start in <2s and spreads CPU load over time.

import logging
import importlib
import threading
from typing import Any, Optional

_logger = logging.getLogger(__name__)


class LazyProxy:
    """Thread-safe lazy import proxy. Loads module.attr on first attribute access.

    Usage:
        SovereignCodec = LazyProxy("l104_codec", "SovereignCodec")
        codec = SovereignCodec()  # Module loaded here, not at import time
    """
    __slots__ = ('_mod', '_attr', '_obj', '_lock', '_loaded')

    def __init__(self, module: str, attr: str = None):
        object.__setattr__(self, '_mod', module)
        object.__setattr__(self, '_attr', attr)
        object.__setattr__(self, '_obj', None)
        object.__setattr__(self, '_lock', threading.Lock())
        object.__setattr__(self, '_loaded', False)

    def _resolve(self) -> Optional[Any]:
        """Lazily resolve the import on first access."""
        if not object.__getattribute__(self, '_loaded'):
            with object.__getattribute__(self, '_lock'):
                if not object.__getattribute__(self, '_loaded'):
                    mod_name = object.__getattribute__(self, '_mod')
                    attr_name = object.__getattribute__(self, '_attr')
                    try:
                        mod = importlib.import_module(mod_name)
                        obj = getattr(mod, attr_name) if attr_name else mod
                    except Exception as e:
                        _logger.warning(f"Lazy load failed: {mod_name}.{attr_name} — {e}")
                        obj = None
                    object.__setattr__(self, '_obj', obj)
                    object.__setattr__(self, '_loaded', True)
        return object.__getattribute__(self, '_obj')

    def __getattr__(self, name: str) -> Any:
        obj = self._resolve()
        if obj is None:
            raise AttributeError(f"Module not loaded: {object.__getattribute__(self, '_mod')}")
        return getattr(obj, name)

    def __call__(self, *args, **kwargs) -> Any:
        obj = self._resolve()
        return obj(*args, **kwargs) if obj else None

    def __bool__(self) -> bool:
        return self._resolve() is not None


# ═══════════════════════════════════════════════════════════════════════════════
# CORE SUBSYSTEMS — Loaded on first use
# ═══════════════════════════════════════════════════════════════════════════════

SovereignCodec = LazyProxy("l104_codec", "SovereignCodec")
SovereignCrypt = LazyProxy("l104_security", "SovereignCrypt")
ignite_sovereign_core = LazyProxy("l104_engine", "ignite_sovereign_core")
persist_truth = LazyProxy("l104_persistence", "persist_truth")
agi_core = LazyProxy("l104_agi_core", "agi_core")
asi_core = LazyProxy("l104_asi_core", "asi_core")
google_bridge = LazyProxy("l104_google_bridge", "google_bridge")
unified_asi = LazyProxy("l104_unified_asi", "unified_asi")
asi_nexus = LazyProxy("l104_asi_nexus", "asi_nexus")
synergy_engine = LazyProxy("l104_synergy_engine", "synergy_engine")
data_matrix = LazyProxy("l104_data_matrix", "data_matrix")
evolution_engine = LazyProxy("l104_evolution_engine", "evolution_engine")


# ═══════════════════════════════════════════════════════════════════════════════
# EVO_61 PIPELINE — Lazy loaded
# ═══════════════════════════════════════════════════════════════════════════════

adaptive_learner = LazyProxy("l104_adaptive_learning", "adaptive_learner")
COGNITIVE_CORE = LazyProxy("l104_cognitive_core", "COGNITIVE_CORE")
innovation_engine = LazyProxy("l104_autonomous_innovation", "innovation_engine")
streaming_engine = LazyProxy("l104_streaming_engine", "get_streaming_engine")
ouroboros = LazyProxy("l104_thought_entropy_ouroboros", "get_thought_ouroboros")
ouroboros_duality = LazyProxy("l104_ouroboros_inverse_duality", "ouroboros_duality")
consciousness_core = LazyProxy("l104_consciousness", "l104_consciousness")
qc_module = LazyProxy("l104_quantum_consciousness", "quantum_consciousness")
sage_mode = LazyProxy("l104_sage_mode", "sage_mode")
coding_system = LazyProxy("l104_coding_system", "coding_system")
sentient_archive = LazyProxy("l104_sentient_archive", "sentient_archive")
language_engine = LazyProxy("l104_language_engine", "language_engine")
data_pipeline = LazyProxy("l104_data_pipeline", "l104_pipeline")
healing_fabric = LazyProxy("l104_self_healing_fabric", "activate_healing_fabric")
rl_engine = LazyProxy("l104_reinforcement_engine", "create_rl_engine")
neural_symbolic = LazyProxy("l104_neural_symbolic_fusion", "create_neural_symbolic_fusion")
quantum_link_builder = LazyProxy("l104_quantum_link_builder", "QuantumLinkBuilder")
GOD_CODE_HP = LazyProxy("l104_quantum_numerical_builder", "GOD_CODE_HP")


# ═══════════════════════════════════════════════════════════════════════════════
# CONSCIOUSNESS SUBSTRATE — Lazy with deferred orchestrator wiring
# ═══════════════════════════════════════════════════════════════════════════════

sage_core = LazyProxy("l104_sage_bindings", "get_sage_core")
consciousness_substrate = LazyProxy("l104_consciousness_substrate", "get_consciousness_substrate")
intricate_cognition = LazyProxy("l104_intricate_cognition", "get_intricate_cognition")
intricate_research = LazyProxy("l104_intricate_research", "get_intricate_research")
intricate_ui = LazyProxy("l104_intricate_ui", "get_intricate_ui")
intricate_learning = LazyProxy("l104_intricate_learning", "get_intricate_learning")
intricate_orchestrator = LazyProxy("l104_intricate_orchestrator", "get_intricate_orchestrator")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE LIST — For iteration/validation
# ═══════════════════════════════════════════════════════════════════════════════

ALL_PROXIES = [
    # Core
    SovereignCodec, SovereignCrypt, ignite_sovereign_core, persist_truth,
    agi_core, asi_core, google_bridge, unified_asi, asi_nexus, synergy_engine,
    data_matrix, evolution_engine,
    # EVO_61 Pipeline
    adaptive_learner, COGNITIVE_CORE, innovation_engine, streaming_engine,
    ouroboros, ouroboros_duality, consciousness_core, qc_module, sage_mode,
    coding_system, sentient_archive, language_engine, data_pipeline,
    healing_fabric, rl_engine, neural_symbolic, quantum_link_builder, GOD_CODE_HP,
    # Consciousness
    sage_core, consciousness_substrate, intricate_cognition, intricate_research,
    intricate_ui, intricate_learning, intricate_orchestrator,
]


__all__ = [
    "LazyProxy",
    # Core
    "SovereignCodec", "SovereignCrypt", "ignite_sovereign_core", "persist_truth",
    "agi_core", "asi_core", "google_bridge", "unified_asi", "asi_nexus",
    "synergy_engine", "data_matrix", "evolution_engine",
    # EVO_61 Pipeline
    "adaptive_learner", "COGNITIVE_CORE", "innovation_engine", "streaming_engine",
    "ouroboros", "ouroboros_duality", "consciousness_core", "qc_module", "sage_mode",
    "coding_system", "sentient_archive", "language_engine", "data_pipeline",
    "healing_fabric", "rl_engine", "neural_symbolic", "quantum_link_builder", "GOD_CODE_HP",
    # Consciousness Substrate
    "sage_core", "consciousness_substrate", "intricate_cognition", "intricate_research",
    "intricate_ui", "intricate_learning", "intricate_orchestrator",
    # Module list
    "ALL_PROXIES",
]