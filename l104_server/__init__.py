"""
L104 Server Package — Decomposed from l104_fast_server.py (22,024 lines)
EVO_61: 46 classes, 271 routes → modular package
FLAGSHIP: Dual-Layer Engine — The Duality of Nature

Import compatibility: all symbols previously available from l104_fast_server
are re-exported here. FastAPI app imported lazily (requires uvicorn/fastapi).
"""
from l104_server.learning import intellect, LearningIntellect
from l104_server.engines_quantum import SingularityConsciousnessEngine
from l104_server.engines_nexus import engine_registry, UnifiedEngineRegistry
from l104_server.constants import FAST_SERVER_VERSION, FAST_SERVER_PIPELINE_EVO

# ★ FLAGSHIP: ASI Dual-Layer Engine ★
try:
    from l104_asi.dual_layer import dual_layer_engine, DualLayerEngine, DUAL_LAYER_AVAILABLE
except ImportError:
    dual_layer_engine = None
    DualLayerEngine = None
    DUAL_LAYER_AVAILABLE = False


def get_app():
    """Lazy import of FastAPI app (requires uvicorn/fastapi installed)."""
    from l104_server.app import app
    return app


# Re-export models lazily to avoid pydantic import at package level
def _load_models():
    from l104_server.models import ChatRequest, TrainingRequest, ProviderStatus
    return ChatRequest, TrainingRequest, ProviderStatus
