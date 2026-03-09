"""
L104 ASI Knowledge Data — Structured knowledge base for language comprehension.

**Package layout** (refactored from the original monolithic file):

Sub-modules
-----------
- ``_stem``             — STEM category nodes
- ``_humanities``       — Humanities category nodes
- ``_social_sciences``  — Social-sciences category nodes
- ``_other``            — Other / miscellaneous category nodes
- ``_relations``        — Cross-subject relation tuples
- ``_registry``         — Functional query / search / summary helpers

Backward-compatible exports
---------------------------
``KNOWLEDGE_NODES`` and ``CROSS_SUBJECT_RELATIONS`` are re-exported at
package level so existing imports continue to work unchanged::

    from l104_asi.knowledge_data import KNOWLEDGE_NODES, CROSS_SUBJECT_RELATIONS

Per-category imports are also available::

    from l104_asi.knowledge_data import STEM_NODES, HUMANITIES_NODES

Registry query API::

    from l104_asi.knowledge_data import search_facts, summary
"""

from __future__ import annotations

__version__ = "2.0.0"

# ── Backward-compatible data exports ─────────────────────────────────────────
# Assembled from per-category sub-modules in canonical order.
from ._stem import STEM_NODES
from ._humanities import HUMANITIES_NODES
from ._social_sciences import SOCIAL_SCIENCES_NODES
from ._other import OTHER_NODES
from ._relations import CROSS_SUBJECT_RELATIONS

KNOWLEDGE_NODES = STEM_NODES + HUMANITIES_NODES + SOCIAL_SCIENCES_NODES + OTHER_NODES

# ── Per-category direct access ───────────────────────────────────────────────
# Allows selective imports:  from l104_asi.knowledge_data import STEM_NODES
__all__ = [
    # Full assembled list (backward compat)
    "KNOWLEDGE_NODES",
    "CROSS_SUBJECT_RELATIONS",
    # Per-category lists
    "STEM_NODES",
    "HUMANITIES_NODES",
    "SOCIAL_SCIENCES_NODES",
    "OTHER_NODES",
    # Registry helpers
    "get_all_nodes",
    "get_nodes_by_category",
    "get_nodes_by_subject",
    "get_node_by_concept",
    "get_categories",
    "get_subjects",
    "search_facts",
    "count_facts",
    "get_related_subjects",
    "summary",
    "invalidate_cache",
]

# ── Registry query API ──────────────────────────────────────────────────────
from ._registry import (
    get_all_nodes,
    get_nodes_by_category,
    get_nodes_by_subject,
    get_node_by_concept,
    get_categories,
    get_subjects,
    search_facts,
    count_facts,
    get_related_subjects,
    summary,
    invalidate_cache,
)
