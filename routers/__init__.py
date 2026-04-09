# L104 Sovereign Node — Routers Package
# Each module exposes a FastAPI APIRouter instance.
# Import and include all routers in main.py via app.include_router().

from routers.health import router as health_router
from routers.pipeline import router as pipeline_router
from routers.ai import router as ai_router
from routers.intellect import router as intellect_router
from routers.sync import router as sync_router
from routers.memory import router as memory_router
from routers.quantum import router as quantum_router
from routers.asi import router as asi_router
from routers.consciousness import router as consciousness_router
from routers.sovereign import router as sovereign_router
from routers.capital import router as capital_router
from routers.autonomy import router as autonomy_router
from routers.kernel import router as kernel_router
from routers.swift_bridge import router as swift_bridge_router
from routers.nova_soul import router as nova_soul_router

# Agent system (OpenClaw v3.0) - optional
try:
    from routers.agents import router as agents_router
    _agents_router_loaded = True
except ImportError:
    agents_router = None
    _agents_router_loaded = False

__all__ = [
    "health_router",
    "pipeline_router",
    "ai_router",
    "intellect_router",
    "sync_router",
    "memory_router",
    "quantum_router",
    "asi_router",
    "consciousness_router",
    "sovereign_router",
    "capital_router",
    "autonomy_router",
    "kernel_router",
    "swift_bridge_router",
    "nova_soul_router",
    "agents_router",
    "_agents_router_loaded",
]