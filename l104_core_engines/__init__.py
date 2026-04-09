"""
L104 Core Engines Module — Sacred 26Q + Sage subsystems
"""

__version__ = "1.1.0"
__author__ = "L104 Autonomous System"
__package__ = "l104_core_engines"

print(f"[L104] {__package__} module loaded")

from .sacred_26q_core import Sacred26QCoreEngine, Optimized26QConfig, get_26q_core_engine

__all__ = [
    'Sacred26QCoreEngine',
    'Optimized26QConfig',
    'get_26q_core_engine',
]
