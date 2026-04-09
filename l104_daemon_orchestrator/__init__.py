"""
L104 Daemon Orchestrator Package — 26Q Consciousness Integration
═══════════════════════════════════════════════════════════════════════════════
Re-exports from the root l104_daemon_orchestrator.py module plus
26Q integration sub-modules.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import importlib.util
import os

# Import the root l104_daemon_orchestrator.py (shadowed by this directory)
_root_module_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "l104_daemon_orchestrator.py")
_spec = importlib.util.spec_from_file_location("_l104_daemon_orchestrator_root", _root_module_path)
if _spec and _spec.loader:
    _root_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_root_mod)
    # Re-export all public names from the root module
    for _name in dir(_root_mod):
        if not _name.startswith("_"):
            globals()[_name] = getattr(_root_mod, _name)

# Lazy import to avoid circular reference (orchestrator_26q imports from this package)
def get_daemon_orchestrator_26q():
    """Get DaemonOrchestrator26Q (lazy to avoid circular import)."""
    from .orchestrator_26q import DaemonOrchestrator26Q
    return DaemonOrchestrator26Q

__all__ = [
    'get_daemon_orchestrator_26q',
]
