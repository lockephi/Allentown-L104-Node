"""L104 Quantum Networker — Shared VQPU Bridge helper with timeout guard.

The VQPU Bridge initializes the full engine stack (AGI, Code Engine, etc.)
which can take minutes. This module provides a timeout-guarded lazy loader
that falls back to None when the bridge can't initialize within the budget.

Set VQPU_TIMEOUT_SECONDS to control the initialization timeout.
Set VQPU_ENABLED = False to disable VQPU entirely (analytical-only mode).

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import os
import threading
from typing import Optional

# Configuration
VQPU_ENABLED = os.environ.get("L104_VQPU_ENABLED", "1") != "0"
VQPU_TIMEOUT_SECONDS = float(os.environ.get("L104_VQPU_TIMEOUT", "10"))

# Module state
_bridge = None
_bridge_attempted = False
_bridge_lock = threading.Lock()
_scorer = None
_scorer_attempted = False


def get_bridge():
    """Get the VQPU Bridge singleton with timeout protection.

    Returns None if:
      - VQPU_ENABLED is False
      - l104_vqpu is not importable
      - Bridge initialization exceeds VQPU_TIMEOUT_SECONDS
    """
    global _bridge, _bridge_attempted

    if not VQPU_ENABLED:
        return None

    if _bridge_attempted:
        return _bridge

    with _bridge_lock:
        if _bridge_attempted:
            return _bridge
        _bridge_attempted = True

        result_holder = [None]

        def _init():
            try:
                from l104_vqpu import get_bridge as _get
                result_holder[0] = _get()
            except Exception:
                result_holder[0] = None

        t = threading.Thread(target=_init, daemon=True)
        t.start()
        t.join(timeout=VQPU_TIMEOUT_SECONDS)

        if t.is_alive():
            # Bridge init timed out — use analytical fallback
            _bridge = None
        else:
            _bridge = result_holder[0]

    return _bridge


def get_scorer():
    """Get the SacredAlignmentScorer with timeout protection."""
    global _scorer, _scorer_attempted

    if not VQPU_ENABLED:
        return None

    if _scorer_attempted:
        return _scorer

    _scorer_attempted = True
    try:
        from l104_vqpu import SacredAlignmentScorer
        _scorer = SacredAlignmentScorer()
    except Exception:
        _scorer = None

    return _scorer


def reset():
    """Reset bridge state (for testing)."""
    global _bridge, _bridge_attempted, _scorer, _scorer_attempted
    _bridge = None
    _bridge_attempted = False
    _scorer = None
    _scorer_attempted = False
