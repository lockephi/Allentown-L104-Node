"""
L104 Server Routes Package

Organized by domain for maintainability:
- root_routes: Kernel, Consciousness, Research, Learning, Sovereign
- v6_core: Legacy v6 routes (chat, intellect, providers)
- v10_routes: Benchmark, NLU, Formal Logic, Deep NLU
- v14_intellect: Intellect/learning routes
- v14_quantum: Quantum memory and circuits
- v14_system: System operations
- v14_agents: Agent deployment and management
- v14_nexus: Steering, evolution, quantum networking
- v14_physics: ZPE, quantum gravity, hardware
- v14_stats: Stats, consolidate
- v16_routes: Brain sync and storage
- v26_routes: HyperMath, Hebbian, Consciousness, Solver
- v27_routes: Registry, Creative Engine
- v54_routes: Meta-cognitive, Knowledge Bridge
- v62_routes: Tri-engine integration
- v63_routes: Temporal coherence, fitness, entropy
- v64_three_engine: Three-engine integration
- v_misc: v1, v3 routes

Usage:
    from l104_server.routes import register_all_routes
    register_all_routes(app)
"""

from fastapi import APIRouter

# Route modules - import with fallbacks for incremental migration
_routes = {}

# Core routes that are always available
try:
    from l104_server.routes.v64_three_engine import router as v64_router
    _routes['v64'] = v64_router
except ImportError:
    v64_router = APIRouter()
    _routes['v64'] = v64_router

try:
    from l104_server.routes.v_misc import router as misc_router
    _routes['misc'] = misc_router
except ImportError:
    misc_router = APIRouter()
    _routes['misc'] = misc_router

# v6 routes
try:
    from l104_server.routes.v6_core import router as v6_router
    _routes['v6'] = v6_router
except ImportError:
    v6_router = APIRouter()
    _routes['v6'] = v6_router

# v10 routes
try:
    from l104_server.routes.v10_routes import router as v10_router
    _routes['v10'] = v10_router
except ImportError:
    v10_router = APIRouter()
    _routes['v10'] = v10_router

# v14 routes - extracted modules
try:
    from l104_server.routes.v14_quantum import router as v14_quantum_router
    _routes['v14_quantum'] = v14_quantum_router
except ImportError:
    v14_quantum_router = APIRouter()
    _routes['v14_quantum'] = v14_quantum_router

try:
    from l104_server.routes.v14_intellect import router as v14_intellect_router
    _routes['v14_intellect'] = v14_intellect_router
except ImportError:
    v14_intellect_router = APIRouter()
    _routes['v14_intellect'] = v14_intellect_router

try:
    from l104_server.routes.v14_system import router as v14_system_router
    _routes['v14_system'] = v14_system_router
except ImportError:
    v14_system_router = APIRouter()
    _routes['v14_system'] = v14_system_router

try:
    from l104_server.routes.v14_agents import router as v14_agents_router
    _routes['v14_agents'] = v14_agents_router
except ImportError:
    v14_agents_router = APIRouter()
    _routes['v14_agents'] = v14_agents_router

try:
    from l104_server.routes.v14_nexus import router as v14_nexus_router
    _routes['v14_nexus'] = v14_nexus_router
except ImportError:
    v14_nexus_router = APIRouter()
    _routes['v14_nexus'] = v14_nexus_router

try:
    from l104_server.routes.v14_physics import router as v14_physics_router
    _routes['v14_physics'] = v14_physics_router
except ImportError:
    v14_physics_router = APIRouter()
    _routes['v14_physics'] = v14_physics_router

try:
    from l104_server.routes.v14_stats import router as v14_stats_router
    _routes['v14_stats'] = v14_stats_router
except ImportError:
    v14_stats_router = APIRouter()
    _routes['v14_stats'] = v14_stats_router

# v16 routes
try:
    from l104_server.routes.v16_routes import router as v16_router
    _routes['v16'] = v16_router
except ImportError:
    v16_router = APIRouter()
    _routes['v16'] = v16_router

# v26 routes
try:
    from l104_server.routes.v26_routes import router as v26_router
    _routes['v26'] = v26_router
except ImportError:
    v26_router = APIRouter()
    _routes['v26'] = v26_router

# v27 routes
try:
    from l104_server.routes.v27_routes import router as v27_router
    _routes['v27'] = v27_router
except ImportError:
    v27_router = APIRouter()
    _routes['v27'] = v27_router

# v54 routes
try:
    from l104_server.routes.v54_routes import router as v54_router
    _routes['v54'] = v54_router
except ImportError:
    v54_router = APIRouter()
    _routes['v54'] = v54_router

# v62 routes
try:
    from l104_server.routes.v62_routes import router as v62_router
    _routes['v62'] = v62_router
except ImportError:
    v62_router = APIRouter()
    _routes['v62'] = v62_router

# v63 routes
try:
    from l104_server.routes.v63_routes import router as v63_router
    _routes['v63'] = v63_router
except ImportError:
    v63_router = APIRouter()
    _routes['v63'] = v63_router

# Core routes (root-level endpoints)
try:
    from l104_server.routes.core_routes import router as core_router
    _routes['core'] = core_router
except ImportError:
    core_router = APIRouter()
    _routes['core'] = core_router

# Root routes (non-versioned)
try:
    from l104_server.routes.root_routes import router as root_router
    _routes['root'] = root_router
except ImportError:
    root_router = APIRouter()
    _routes['root'] = root_router

# Core routes (favicon, landing, health, chat)
try:
    from l104_server.routes.core_routes import router as core_router
    _routes['core'] = core_router
except ImportError:
    core_router = APIRouter()
    _routes['core'] = core_router

# Placeholder routes (still in app.py)
v14_consciousness_router = APIRouter()


__all__ = [
    "register_all_routes",
    "core_router",
    "root_router",
    "v6_router",
    "v10_router",
    "v14_intellect_router",
    "v14_quantum_router",
    "v14_system_router",
    "v14_agents_router",
    "v14_nexus_router",
    "v14_physics_router",
    "v14_stats_router",
    "v14_consciousness_router",
    "v16_router",
    "v26_router",
    "v27_router",
    "v54_router",
    "v62_router",
    "v63_router",
    "v64_router",
    "misc_router",
]


def register_all_routes(app):
    """Register all route modules with the FastAPI app.

    Only registers routes that have been extracted to separate files.
    Routes still in app.py will continue to work from there.
    """
    # Order matters: register core routes first, then versioned routes
    route_order = [
        'core',      # /favicon, /landing, /health, /metrics, /chat
        'root',      # /api/kernel, /api/consciousness, etc.
        'v6',        # Legacy v6 routes
        'v10',       # Benchmark, NLU routes
        'v14_quantum',
        'v14_intellect',
        'v14_system',
        'v14_agents',
        'v14_nexus',
        'v14_physics',
        'v14_stats',
        'v16',       # Brain routes
        'v26',
        'v27',
        'v54',
        'v62',
        'v63',
        'v64',
        'misc',      # v1, v3 routes
    ]

    for key in route_order:
        if _routes.get(key):
            app.include_router(_routes[key])