"""
L104 ASI Knowledge Data — Registry & query utilities.

Provides functional helpers for filtering, searching, and aggregating
knowledge nodes.  All indexes are built lazily on first access and cached
for the lifetime of the process (call ``invalidate_cache()`` after data
changes, e.g. after a merge run).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, TypedDict


# ---------------------------------------------------------------------------
#  Result types
# ---------------------------------------------------------------------------

class FactMatch(TypedDict):
    """A single fact-search result."""
    node: Dict[str, Any]
    fact: str


class CategoryStats(TypedDict):
    """Per-category breakdown returned by ``summary()``."""
    nodes: int
    facts: int
    subjects: List[str]


# ---------------------------------------------------------------------------
# Lazy singleton — all indexes built once on first access
# ---------------------------------------------------------------------------

_CACHED_NODES: Optional[List[Dict[str, Any]]] = None
_CACHED_INDEX: Optional[Dict[str, List[Dict[str, Any]]]] = None
_CACHED_SUBJECT_INDEX: Optional[Dict[str, List[Dict[str, Any]]]] = None
_CACHED_CONCEPT_MAP: Optional[Dict[str, Dict[str, Any]]] = None


def invalidate_cache() -> None:
    """Reset all cached indexes.

    Call this after programmatically modifying the underlying data files
    (e.g. after ``_merge_mmlu_to_kb.py`` writes new sub-modules).
    """
    global _CACHED_NODES, _CACHED_INDEX, _CACHED_SUBJECT_INDEX, _CACHED_CONCEPT_MAP
    _CACHED_NODES = None
    _CACHED_INDEX = None
    _CACHED_SUBJECT_INDEX = None
    _CACHED_CONCEPT_MAP = None


def _ensure_loaded() -> List[Dict[str, Any]]:
    """Return the full KNOWLEDGE_NODES list, loading once."""
    global _CACHED_NODES
    if _CACHED_NODES is None:
        from ._stem import STEM_NODES
        from ._humanities import HUMANITIES_NODES
        from ._social_sciences import SOCIAL_SCIENCES_NODES
        from ._other import OTHER_NODES
        _CACHED_NODES = STEM_NODES + HUMANITIES_NODES + SOCIAL_SCIENCES_NODES + OTHER_NODES
    return _CACHED_NODES


# ---------------------------------------------------------------------------
#  Indexing helpers (built lazily)
# ---------------------------------------------------------------------------

def _category_index() -> Dict[str, List[Dict[str, Any]]]:
    """Index nodes by category.  Built once, cached."""
    global _CACHED_INDEX
    if _CACHED_INDEX is None:
        _CACHED_INDEX = {}
        for node in _ensure_loaded():
            _CACHED_INDEX.setdefault(node["category"], []).append(node)
    return _CACHED_INDEX


def _subject_index() -> Dict[str, List[Dict[str, Any]]]:
    """Index nodes by subject.  Built once, cached."""
    global _CACHED_SUBJECT_INDEX
    if _CACHED_SUBJECT_INDEX is None:
        _CACHED_SUBJECT_INDEX = {}
        for node in _ensure_loaded():
            _CACHED_SUBJECT_INDEX.setdefault(node["subject"], []).append(node)
    return _CACHED_SUBJECT_INDEX


def _concept_map() -> Dict[str, Dict[str, Any]]:
    """Map concept → node.  Built once, cached."""
    global _CACHED_CONCEPT_MAP
    if _CACHED_CONCEPT_MAP is None:
        _CACHED_CONCEPT_MAP = {}
        for node in _ensure_loaded():
            _CACHED_CONCEPT_MAP[node["concept"]] = node
    return _CACHED_CONCEPT_MAP


# ---------------------------------------------------------------------------
#  Public query API
# ---------------------------------------------------------------------------

def get_all_nodes() -> List[Dict[str, Any]]:
    """Return the full assembled KNOWLEDGE_NODES list."""
    return _ensure_loaded()


def get_nodes_by_category(category: str) -> List[Dict[str, Any]]:
    """Return all knowledge nodes belonging to *category*.

    >>> len(get_nodes_by_category("stem"))
    101
    """
    return _category_index().get(category, [])


def get_nodes_by_subject(subject: str) -> List[Dict[str, Any]]:
    """Return all knowledge nodes for *subject*.

    >>> len(get_nodes_by_subject("abstract_algebra")) > 0
    True
    """
    return _subject_index().get(subject, [])


def get_node_by_concept(concept: str) -> Optional[Dict[str, Any]]:
    """Lookup a single node by concept name (exact match)."""
    return _concept_map().get(concept)


def get_categories() -> List[str]:
    """Return sorted list of all category names."""
    return sorted(_category_index().keys())


def get_subjects(category: Optional[str] = None) -> List[str]:
    """Return sorted list of all subject names, optionally filtered by category."""
    if category is not None:
        return sorted({n["subject"] for n in get_nodes_by_category(category)})
    return sorted(_subject_index().keys())


def search_facts(query: str, *, category: Optional[str] = None,
                 case_sensitive: bool = False) -> List[FactMatch]:
    """Search across all facts for *query* substring.

    Returns list of :class:`FactMatch` dicts with ``node`` and ``fact`` keys.
    """
    if not case_sensitive:
        query = query.lower()
    results: List[FactMatch] = []
    nodes = get_nodes_by_category(category) if category else _ensure_loaded()
    for node in nodes:
        for fact in node.get("facts", []):
            target = fact if case_sensitive else fact.lower()
            if query in target:
                results.append({"node": node, "fact": fact})
    return results


def count_facts(category: Optional[str] = None) -> int:
    """Count total facts, optionally restricted to *category*."""
    nodes = get_nodes_by_category(category) if category else _ensure_loaded()
    return sum(len(n.get("facts", [])) for n in nodes)


def fact_count_by_category() -> Dict[str, int]:
    """Return ``{category: fact_count}`` for every category."""
    return {cat: sum(len(n.get("facts", [])) for n in nodes)
            for cat, nodes in _category_index().items()}


def get_related_subjects(subject: str) -> Set[str]:
    """Return set of subjects related to *subject* via CROSS_SUBJECT_RELATIONS.

    Relations are stored as ``"subject/concept"`` pairs — this extracts the
    subject portion for matching.  Both directions of a relation are checked.
    """
    from ._relations import CROSS_SUBJECT_RELATIONS
    related: Set[str] = set()
    for src, tgt in CROSS_SUBJECT_RELATIONS:
        src_subj = src.split("/", 1)[0] if "/" in src else src
        tgt_subj = tgt.split("/", 1)[0] if "/" in tgt else tgt
        if src_subj == subject:
            related.add(tgt_subj)
        elif tgt_subj == subject:
            related.add(src_subj)
    return related


def summary() -> Dict[str, Any]:
    """Return a compact, JSON-serializable summary of the knowledge base.

    Includes per-category breakdowns with node/fact counts and subject lists.
    """
    nodes = _ensure_loaded()
    from ._relations import CROSS_SUBJECT_RELATIONS

    cats: Dict[str, CategoryStats] = {}
    for node in nodes:
        c = node["category"]
        if c not in cats:
            cats[c] = {"nodes": 0, "facts": 0, "subjects": []}
        cats[c]["nodes"] += 1
        cats[c]["facts"] += len(node.get("facts", []))

    # Collect unique subjects per category (sorted for determinism)
    subject_sets: Dict[str, set] = {}
    for node in nodes:
        c = node["category"]
        subject_sets.setdefault(c, set()).add(node["subject"])
    for c, subjs in subject_sets.items():
        cats[c]["subjects"] = sorted(subjs)

    return {
        "total_nodes": len(nodes),
        "total_facts": count_facts(),
        "total_relations": len(CROSS_SUBJECT_RELATIONS),
        "categories": cats,
    }
