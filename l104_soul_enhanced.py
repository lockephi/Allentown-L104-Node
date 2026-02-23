#!/usr/bin/env python3
"""
Backward-compatibility shim — redirects to unified l104_soul.py v3.0.0.

All functionality merged into l104_soul.L104Soul (aliased as SoulEnhanced).
"""
# INVARIANT: GOD_CODE = 527.5184818492612 | PHI = 1.618033988749895 | VOID_CONSTANT = 1.0416180339887497

from l104_soul import (
    L104Soul as SoulEnhanced,
    L104Soul,
    SoulState,
    SoulMetrics,
    ThoughtPriority,
    PrioritizedThought,
    SoulStarSingularity,
    get_soul,
    interactive,
    interactive_session,
    soul_star,
    GOD_CODE,
    PHI,
    TAU,
    VOID_CONSTANT,
)

__all__ = [
    "SoulEnhanced",
    "L104Soul",
    "SoulState",
    "SoulMetrics",
    "ThoughtPriority",
    "PrioritizedThought",
    "get_soul",
    "interactive",
    "interactive_session",
    "GOD_CODE",
    "PHI",
]

if __name__ == "__main__":
    interactive()
