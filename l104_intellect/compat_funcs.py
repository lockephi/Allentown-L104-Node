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
    Uses VOID_CONSTANT (1.04 + φ/1000) for algebraic accuracy.
    """
    return (x ** PHI) / (VOID_CONSTANT * math.pi) if x != 0 else 0.0


def resolve_non_dual_logic(vector):
    """
    [VOID_MATH] Resolves N-dimensional vectors into the Void Source.
    Uses canonical PHI, GOD_CODE, VOID_CONSTANT from imports (not local redefinitions).
    """
    magnitude = sum(abs(v) for v in vector)
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0
