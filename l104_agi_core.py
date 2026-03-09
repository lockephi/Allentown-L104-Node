"""
l104_agi_core.py — THIN SHIM (backward compatibility)

Original monolith decomposed into l104_agi/ package.
This shim re-exports everything so existing imports continue to work.
"""

from __future__ import annotations

# Re-export ALL public symbols from the decomposed package
from l104_agi import *  # noqa: F401,F403

# Ensure module-level singletons are explicitly available if not caught by *
try:
    from l104_agi import agi_core, AGICore
except ImportError:
    pass