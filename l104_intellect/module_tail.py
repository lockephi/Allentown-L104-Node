"""L104 Intellect — Module-level singleton creation.

Creates the `local_intellect` singleton instance LAZILY on first access.
Functions (format_iq, primal_calculus, resolve_non_dual_logic) are in compat_funcs.py
to break circular imports.

v28.0 PERFORMANCE: Lazy singleton via module-level __getattr__ to avoid
85s import penalty when only constants/format_iq are needed.
"""
from .constants import LOCAL_INTELLECT_PIPELINE_EVO, LOCAL_INTELLECT_VERSION

_local_intellect = None


def _get_local_intellect():
    """Get or create the LocalIntellect singleton (lazy initialization)."""
    global _local_intellect
    if _local_intellect is None:
        from .local_intellect_core import LocalIntellect
        _local_intellect = LocalIntellect()
        # Pipeline self-registration
        try:
            _local_intellect._pipeline_evo = LOCAL_INTELLECT_PIPELINE_EVO
            _local_intellect._pipeline_version = LOCAL_INTELLECT_VERSION
        except Exception:
            pass
    return _local_intellect


class _LazyIntellect:
    """Proxy that defers LocalIntellect creation until first attribute access."""

    def __getattr__(self, name):
        instance = _get_local_intellect()
        return getattr(instance, name)

    def __repr__(self):
        if _local_intellect is not None:
            return repr(_local_intellect)
        return "<LocalIntellect (not yet initialized — lazy)>"

    def __bool__(self):
        return True


local_intellect = _LazyIntellect()
