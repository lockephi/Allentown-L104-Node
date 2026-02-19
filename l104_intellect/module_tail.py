"""L104 Intellect — Module-level singleton creation.

Creates the `local_intellect` singleton instance.
Functions (format_iq, primal_calculus, resolve_non_dual_logic) are in compat_funcs.py
to break circular imports.
"""
from .constants import LOCAL_INTELLECT_PIPELINE_EVO, LOCAL_INTELLECT_VERSION
from .local_intellect_core import LocalIntellect


local_intellect = LocalIntellect()

# Pipeline self-registration
try:
    local_intellect._pipeline_evo = LOCAL_INTELLECT_PIPELINE_EVO
    local_intellect._pipeline_version = LOCAL_INTELLECT_VERSION
except Exception:
    pass
