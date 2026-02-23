"""
L104 Code Engine — Centralized Lazy ASI Module Imports

Provides thread-safe, cached lazy import functions for all 9 ASI subsystem
singletons plus the code_engine singleton. Used by modules within the package
that need runtime access to external L104 systems without circular import risk.

Each getter is called at runtime (never import time) to fetch module singletons.
"""
import logging

logger = logging.getLogger("L104_CODE_ENGINE")

# ═══════════════════════════════════════════════════════════════════════════════
# LAZY ASI MODULE IMPORTS — All 9 singletons + code_engine loaded on demand
# ═══════════════════════════════════════════════════════════════════════════════

_code_engine = None
_neural_cascade = None
_evolution_engine = None
_self_optimizer = None
_innovation_engine = None
_consciousness = None
_reasoning = None
_knowledge_graph = None
_polymorph = None


def _get_code_engine():
    """Lazy import of code_engine singleton to avoid circular imports."""
    global _code_engine
    if _code_engine is None:
        try:
            from l104_code_engine import code_engine
            _code_engine = code_engine
        except ImportError:
            logger.warning("l104_code_engine singleton not available")
    return _code_engine


def _get_neural_cascade():
    """Lazy import of neural_cascade singleton."""
    global _neural_cascade
    if _neural_cascade is None:
        try:
            from l104_neural_cascade import neural_cascade
            _neural_cascade = neural_cascade
        except ImportError:
            logger.debug("l104_neural_cascade not available")
    return _neural_cascade


def _get_evolution_engine():
    """Lazy import of evolution_engine singleton."""
    global _evolution_engine
    if _evolution_engine is None:
        try:
            from l104_evolution_engine import evolution_engine
            _evolution_engine = evolution_engine
        except ImportError:
            logger.debug("l104_evolution_engine not available")
    return _evolution_engine


def _get_self_optimizer():
    """Lazy import of self_optimizer singleton."""
    global _self_optimizer
    if _self_optimizer is None:
        try:
            from l104_self_optimization import self_optimizer
            _self_optimizer = self_optimizer
        except ImportError:
            logger.debug("l104_self_optimization not available")
    return _self_optimizer


def _get_innovation_engine():
    """Lazy import of innovation_engine singleton."""
    global _innovation_engine
    if _innovation_engine is None:
        try:
            from l104_autonomous_innovation import innovation_engine
            _innovation_engine = innovation_engine
        except ImportError:
            logger.debug("l104_autonomous_innovation not available")
    return _innovation_engine


def _get_consciousness():
    """Lazy import of l104_consciousness singleton."""
    global _consciousness
    if _consciousness is None:
        try:
            from l104_consciousness import l104_consciousness
            _consciousness = l104_consciousness
        except ImportError:
            logger.debug("l104_consciousness not available")
    return _consciousness


def _get_reasoning():
    """Lazy import of l104_reasoning coordinator singleton."""
    global _reasoning
    if _reasoning is None:
        try:
            from l104_reasoning_engine import l104_reasoning
            _reasoning = l104_reasoning
        except ImportError:
            logger.debug("l104_reasoning_engine not available")
    return _reasoning


def _get_knowledge_graph():
    """Lazy import + instantiation of L104KnowledgeGraph (no module singleton)."""
    global _knowledge_graph
    if _knowledge_graph is None:
        try:
            from l104_knowledge_graph import L104KnowledgeGraph
            _knowledge_graph = L104KnowledgeGraph()
        except ImportError:
            logger.debug("l104_knowledge_graph not available")
    return _knowledge_graph


def _get_polymorph():
    """Lazy import of sovereign_polymorph singleton."""
    global _polymorph
    if _polymorph is None:
        try:
            from l104_polymorphic_core import sovereign_polymorph
            _polymorph = sovereign_polymorph
        except ImportError:
            logger.debug("l104_polymorphic_core not available")
    return _polymorph
