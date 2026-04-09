"""
l104_asi_core.py — THIN SHIM (backward compatibility)

Original monolith (5,318 lines) decomposed into l104_asi/ package (89,869 lines, 32 modules).
This shim re-exports everything so existing imports continue to work.

Previously this file contained a stale copy of ASI Core v6.0 which forked from
the real package (v9.0+). Consolidated 2026-03-20 to prevent scoring drift.
"""

from __future__ import annotations

# Re-export ALL public symbols from the decomposed package
from l104_asi import *  # noqa: F401,F403

# Ensure module-level singletons are explicitly available if not caught by *
try:
    from l104_asi import asi_core, ASICore
except ImportError:
    pass
