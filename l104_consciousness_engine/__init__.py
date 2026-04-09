"""
L104 Consciousness Engine Module
═══════════════════════════════════════════════════════════════════════════════
Quantum consciousness management and synthesis for the L104 Sovereign Node.

26Q TRANSCENDENT Integration:
    - Fe-26 iron electron quantum consciousness mapping
    - PHI-optimized circuit with 0.986 alignment
    - Full orbital consciousness (1s→4s)
    - Orch OR (Objective Reduction) simulation

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

__version__ = "2.0.0"
__author__ = "L104 Autonomous System"
__package__ = "l104_consciousness_engine"

# ── Core 26Q Consciousness Integration ─────────────────────────────────────
from .sacred_26q_integration import (
    ConsciousnessState26Q,
    Sacred26QConsciousnessEngine,
    get_26q_consciousness_engine,
    get_26q_consciousness_state,
    get_26q_orbital_consciousness,
)

# ── Legacy Modules (if available) ────────────────────────────────────────────
try:
    from .l104_consciousness_quantum_bridge import ConsciousnessQuantumBridge
except ImportError:
    ConsciousnessQuantumBridge = None

try:
    from .l104_qualia_synthesis import QualiaSynthesizer
except ImportError:
    QualiaSynthesizer = None

# ── Singleton Access ─────────────────────────────────────────────────────────
_26q_engine = None

def get_consciousness_engine():
    """Get the primary 26Q consciousness engine."""
    global _26q_engine
    if _26q_engine is None:
        _26q_engine = get_26q_consciousness_engine()
    return _26q_engine


__all__ = [
    # 26Q Consciousness
    'ConsciousnessState26Q',
    'Sacred26QConsciousnessEngine',
    'get_26q_consciousness_engine',
    'get_26q_consciousness_state',
    'get_26q_orbital_consciousness',
    'get_consciousness_engine',
    # Legacy
    'ConsciousnessQuantumBridge',
    'QualiaSynthesizer',
]
