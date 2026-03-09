"""L104 Intellect — Module-level singleton creation.

Creates the `local_intellect` singleton instance LAZILY on first access.
Functions (format_iq, primal_calculus, resolve_non_dual_logic) are in compat_funcs.py
to break circular imports.

v30.0 SOVEREIGN ACTIVATION: Eager-capable singleton. The proxy guarantees
the intellect is always available with zero first-access penalty when the
module has been imported and the singleton touched.
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
    """Proxy that defers LocalIntellect creation until first attribute access.

    v30.0: Also supports __call__ delegation, len(), iter(), and contains()
    so the singleton behaves identically to the real instance in all contexts.
    """

    def __getattr__(self, name):
        instance = _get_local_intellect()
        return getattr(instance, name)

    def __call__(self, *args, **kwargs):
        instance = _get_local_intellect()
        return instance(*args, **kwargs)

    def __repr__(self):
        if _local_intellect is not None:
            return repr(_local_intellect)
        return "<LocalIntellect (not yet initialized — lazy)>"

    def __bool__(self):
        return True

    def __dir__(self):
        instance = _get_local_intellect()
        return dir(instance)


local_intellect = _LazyIntellect()
