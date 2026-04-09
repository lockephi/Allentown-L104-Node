"""
L104 Server — Nexus Package

Extracted from engines_nexus.py during EVO_78 refactoring.
Contains: Steering, Evolution, Health, Registry, and Integration classes.

Modules:
- steering: SteeringEngine (5-mode ASI parameter steering)
- health: NexusHealthMonitor (engine health monitoring)
- registry: UnifiedEngineRegistry, TriEngineIntegration

Usage:
    from l104_server.nexus import get_steering, get_health_monitor, get_registry
    from l104_server.nexus import SteeringEngine, NexusHealthMonitor
"""

from l104_server.nexus.steering import SteeringEngine, get_steering
from l104_server.nexus.health import NexusHealthMonitor, get_health_monitor
from l104_server.nexus.registry import (
    UnifiedEngineRegistry, get_registry,
    TriEngineIntegration, get_tri_engine,
)

# Export key classes and singletons
__all__ = [
    # Steering
    'SteeringEngine',
    'get_steering',
    # Health
    'NexusHealthMonitor',
    'get_health_monitor',
    # Registry
    'UnifiedEngineRegistry',
    'get_registry',
    'TriEngineIntegration',
    'get_tri_engine',
]
