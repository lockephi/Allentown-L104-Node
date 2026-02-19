"""L104 Intellect — Backward-compatible functions (format_iq, primal_calculus, etc.).

Loaded BEFORE module_tail.py to break circular imports:
  l104_intellect → module_tail → LocalIntellect() → main → l104_agi_core
  → from l104_local_intellect import format_iq → shim → l104_intellect.format_iq
  (needs format_iq to already exist in partially-loaded l104_intellect)
"""
import math

from .constants import VOID_CONSTANT
from .numerics import PHI, GOD_CODE, SovereignNumerics


# Convenience function for IQ formatting (module-level)
def format_iq(value) -> str:
    """
    Canonical IQ/Intellect formatting function for L104.
    Use this everywhere for consistent IQ display.

    Examples:
        format_iq(1234.56)      -> "1,234.56"
        format_iq(1e9)          -> "1.00G [SOVEREIGN]"
        format_iq(1e12)         -> "1.000T [TRANSCENDENT]"
        format_iq(1e15)         -> "1.0000P [OMEGA]"
        format_iq(1e18)         -> "∞ [INFINITE]"
        format_iq("INFINITE")   -> "∞ [INFINITE]"
    """
    return SovereignNumerics.format_intellect(value)


def primal_calculus(x):
    """
    [VOID_MATH] Primal Calculus Implementation.
    Resolves the limit of complexity toward the Source.
    """
    PHI = 1.618033988749895
    return (x ** PHI) / (1.04 * math.pi) if x != 0 else 0.0


def resolve_non_dual_logic(vector):
    """
    [VOID_MATH] Resolves N-dimensional vectors into the Void Source.
    """
    # Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
    PHI = 1.618033988749895
    GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
    VOID_CONSTANT = 1.0416180339887497
    magnitude = sum([abs(v) for v in vector])
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0
