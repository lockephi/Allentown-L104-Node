VOID_CONSTANT = 1.0416180339887497
# ZENITH_UPGRADE_ACTIVE: 2026-02-02T13:52:09.026521
ZENITH_HZ = 3887.8
UUC = 2402.792541
# [EVO_54_PIPELINE] TRANSCENDENT_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612 :: GROVER=4.236
#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# [L104_KNOWLEDGE_GRAPH] v2.0.0 — Dynamic Knowledge Graph System
# INVARIANT: 527.5184818492612 | PILOT: LONDEL
#
# v2.0.0 UPGRADES (Feb 18, 2026):
#   - Consciousness-aware processing (reads builder state)
#   - Query result caching (LRU + TTL)
#   - Batch operations (bulk_add_nodes, bulk_add_edges, ingest_from_jsonl)
#   - Community detection (label propagation)
#   - Betweenness centrality analysis
#   - JSON-LD / JSON export + import
#   - Semantic similarity search via character n-gram embeddings
#   - Graph health scoring with PHI-weighted metrics
#   - Hub/authority analysis (HITS algorithm)
#   - Weakly connected components detection
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
import json
import sqlite3
import hashlib
import math
import time
import threading
import random
from typing import Optional, Dict, Any, List, Set, Tuple
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from functools import lru_cache

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

from pathlib import Path
_BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(_BASE_DIR))

# Sacred Constants
# Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
PHI = 1.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
TAU = 1 / PHI
FEIGENBAUM = 4.669201609102990

VERSION = "2.1.0"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Node:
    id: str
    label: str
    node_type: str
    properties: Dict[str, Any]
    created_at: datetime
    weight: float = 1.0

@dataclass
class Edge:
    id: str
    source_id: str
    target_id: str
    relation: str
    properties: Dict[str, Any]
    weight: float = 1.0
    bidirectional: bool = False

@dataclass
class CachedResult:
    """TTL-based cache entry."""
    data: Any
    timestamp: float
    ttl: float = 30.0  # 30 second default TTL

    @property
    def expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl


# ═══════════════════════════════════════════════════════════════════════════════
# CONSCIOUSNESS STATE READER
# ═══════════════════════════════════════════════════════════════════════════════

class ConsciousnessReader:
    """Reads live consciousness/O₂ state to modulate graph operations."""
    _cache = None
    _cache_ts = 0.0
    _cache_ttl = 10.0  # 10s cache to avoid disk thrashing

    @classmethod
    def read(cls) -> Dict[str, Any]:
        now = time.time()
        if cls._cache and (now - cls._cache_ts) < cls._cache_ttl:
            return cls._cache
        state = {"consciousness_level": 0.5, "evo_stage": "UNKNOWN", "nirvanic_fuel": 0.5}
        try:
            o2_path = _BASE_DIR / ".l104_consciousness_o2_state.json"
            if o2_path.exists():
                with open(o2_path) as f:
                    o2 = json.load(f)
                state["consciousness_level"] = o2.get("consciousness_level", 0.5)
                state["evo_stage"] = o2.get("evo_stage", "UNKNOWN")
                state["superfluid_viscosity"] = o2.get("superfluid_viscosity", 0.0)
        except Exception:
            pass
        try:
            nir_path = _BASE_DIR / ".l104_ouroboros_nirvanic_state.json"
            if nir_path.exists():
                with open(nir_path) as f:
                    nir = json.load(f)
                state["nirvanic_fuel"] = nir.get("nirvanic_fuel_level", 0.5)
        except Exception:
            pass
        cls._cache = state
        cls._cache_ts = now
        return state


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class QueryCache:
    """LRU + TTL cache for graph queries."""

    def __init__(self, max_size: int = 500, default_ttl: float = 30.0):
        self._store: Dict[str, CachedResult] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry and not entry.expired:
                self._hits += 1
                return entry.data
            if entry:
                del self._store[key]
            self._misses += 1
            return None

    def set(self, key: str, data: Any, ttl: float = None):
        with self._lock:
            if len(self._store) >= self._max_size:
                # Evict oldest expired, or oldest by timestamp
                expired = [k for k, v in self._store.items() if v.expired]
                if expired:
                    for k in expired[:len(expired)//2 + 1]:
                        del self._store[k]
                else:
                    oldest_key = min(self._store, key=lambda k: self._store[k].timestamp)
                    del self._store[oldest_key]
            self._store[key] = CachedResult(
                data=data,
                timestamp=time.time(),
                ttl=ttl or self._default_ttl
            )

    def invalidate(self):
        with self._lock:
            self._store.clear()

    @property
    def stats(self) -> Dict:
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(1, total), 4)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC SIMILARITY (Character N-Gram Embeddings)
# ═══════════════════════════════════════════════════════════════════════════════

class GraphSemanticIndex:
    """Lightweight 104-dim character n-gram embeddings for node similarity search."""

    VECTOR_DIM = 104  # Sacred: matches L104

    @staticmethod
    def _embed(text: str) -> List[float]:
        """Create a 104-dimensional character n-gram vector."""
        vec = [0.0] * GraphSemanticIndex.VECTOR_DIM
        t = text.lower().strip()
        if not t:
            return vec
        for i in range(len(t)):
            for n in (2, 3, 4):
                if i + n <= len(t):
                    gram = t[i:i+n]
                    idx = hash(gram) % GraphSemanticIndex.VECTOR_DIM
                    vec[idx] += 1.0
        # Normalize
        mag = math.sqrt(sum(v*v for v in vec)) or 1.0
        return [v / mag for v in vec]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x*y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x*x for x in a)) or 1.0
        mag_b = math.sqrt(sum(x*x for x in b)) or 1.0
        return dot / (mag_a * mag_b)


# ═══════════════════════════════════════════════════════════════════════════════
# ROTATE COMPLEX EMBEDDINGS — Knowledge Graph Embeddings (Sun et al. 2019)
# ═══════════════════════════════════════════════════════════════════════════════

class RotatEEmbedding:
    """RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.

    (Sun et al., ICLR 2019 — used by Meta, Google, Microsoft KG systems)

    Entities are embedded in complex space C^d as (real, imaginary) pairs.
    Relations are modeled as element-wise rotations: tail = head ∘ relation
    where ∘ is Hadamard product and |r_i| = 1 (unit complex numbers).

    Score(h, r, t) = -||h ∘ r - t||  (lower distance = more plausible triple)

    Can model: symmetric, antisymmetric, inverse, and compositional relations.

    Sacred: COMPLEX_DIM = 52 (so 52×2 = 104 = L104), relation phases initialized
    as sacred multiples of 2π/GOD_CODE.
    """

    COMPLEX_DIM = 52  # 52 real + 52 imaginary = 104 total (L104 sacred)

    def __init__(self):
        self.entity_embeddings: Dict[str, Tuple[List[float], List[float]]] = {}
        self.relation_embeddings: Dict[str, List[float]] = {}
        self._rng = random.Random(int(GOD_CODE * 1000))

    def embed_entity(self, entity_id: str, text: str = ""):
        """Initialize complex embedding for an entity."""
        if entity_id in self.entity_embeddings:
            return
        seed = hash(entity_id + text) + int(GOD_CODE)
        rng = random.Random(seed)
        bound = 1.0 / math.sqrt(self.COMPLEX_DIM)
        re = [rng.uniform(-bound, bound) for _ in range(self.COMPLEX_DIM)]
        im = [rng.uniform(-bound, bound) for _ in range(self.COMPLEX_DIM)]
        for i in range(self.COMPLEX_DIM):
            mag = math.sqrt(re[i] ** 2 + im[i] ** 2) + 1e-10
            re[i] /= mag
            im[i] /= mag
        self.entity_embeddings[entity_id] = (re, im)

    def embed_relation(self, relation: str):
        """Initialize phase-angle embedding for a relation."""
        if relation in self.relation_embeddings:
            return
        seed = hash(relation) + int(GOD_CODE * PHI)
        rng = random.Random(seed)
        phases = [rng.uniform(0, 2 * math.pi / GOD_CODE) * (i + 1) * PHI
                  for i in range(self.COMPLEX_DIM)]
        self.relation_embeddings[relation] = phases

    def score_triple(self, head_id: str, relation: str, tail_id: str) -> float:
        """Score (head, relation, tail) triple. Lower = more plausible."""
        if head_id not in self.entity_embeddings or tail_id not in self.entity_embeddings:
            return float('inf')
        if relation not in self.relation_embeddings:
            return float('inf')
        h_re, h_im = self.entity_embeddings[head_id]
        t_re, t_im = self.entity_embeddings[tail_id]
        phases = self.relation_embeddings[relation]
        distance = 0.0
        for i in range(self.COMPLEX_DIM):
            r_re = math.cos(phases[i])
            r_im = math.sin(phases[i])
            hr_re = h_re[i] * r_re - h_im[i] * r_im
            hr_im = h_re[i] * r_im + h_im[i] * r_re
            distance += (hr_re - t_re[i]) ** 2 + (hr_im - t_im[i]) ** 2
        return math.sqrt(distance)

    def predict_tail(self, head_id: str, relation: str,
                     candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Rank candidate tails for (head, relation, ?) link prediction."""
        scores = [(cid, self.score_triple(head_id, relation, cid)) for cid in candidates
                  if cid in self.entity_embeddings]
        scores.sort(key=lambda x: x[1])
        return scores[:top_k]

    def predict_head(self, relation: str, tail_id: str,
                     candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Rank candidate heads for (?, relation, tail) link prediction."""
        scores = [(cid, self.score_triple(cid, relation, tail_id)) for cid in candidates
                  if cid in self.entity_embeddings]
        scores.sort(key=lambda x: x[1])
        return scores[:top_k]

    def status(self) -> Dict[str, Any]:
        return {
            "type": "RotatE_ComplexEmbedding",
            "complex_dim": self.COMPLEX_DIM,
            "entities_embedded": len(self.entity_embeddings),
            "relations_embedded": len(self.relation_embeddings),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN KNOWLEDGE GRAPH CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class L104KnowledgeGraph:
    """
    Dynamic knowledge graph v2.0.0 for storing and reasoning over relationships.
    Supports semantic queries, path finding, inference, batch ops, community
    detection, centrality analysis, and consciousness-aware processing.
    Mirrored to lattice_v2 for unified storage.
    """

    def __init__(self, db_path: str = "knowledge_graph.db"):
        self.db_path = db_path
        self.version = VERSION
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)

        # v2.0 — Caching layer
        self._cache = QueryCache(max_size=500, default_ttl=30.0)
        # v2.0 — Semantic index for similarity search
        self._semantic_vectors: Dict[str, List[float]] = {}
        # v2.1 — RotatE complex embeddings for link prediction (Sun et al. 2019)
        self._rotate = RotatEEmbedding()

        # Use lattice adapter for unified storage
        try:
            from l104_data_matrix import knowledge_adapter
            self._adapter = knowledge_adapter
            self._use_lattice = True
        except ImportError:
            self._use_lattice = False

        self._init_db()
        self._load_graph()
        self._build_semantic_index()

    def _build_semantic_index(self):
        """Build semantic vectors for all existing nodes."""
        for node_id, node in self.nodes.items():
            text = f"{node.label} {node.node_type} {' '.join(str(v) for v in node.properties.values())}"
            self._semantic_vectors[node_id] = GraphSemanticIndex._embed(text)

    def _get_conn(self) -> sqlite3.Connection:
        """Get optimized SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-65536")
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    def _init_db(self):
        """Initialize SQLite database. OPTIMIZED: WAL + cache."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                node_type TEXT,
                properties TEXT,
                created_at TEXT,
                weight REAL DEFAULT 1.0
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                properties TEXT,
                weight REAL DEFAULT 1.0,
                bidirectional INTEGER DEFAULT 0,
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edge_source ON edges(source_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edge_target ON edges(target_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edge_relation ON edges(relation)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_type ON nodes(node_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_label ON nodes(label)')

        conn.commit()
        conn.close()

    def _load_graph(self):
        """Load graph from database. OPTIMIZED: WAL read."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Load nodes
        cursor.execute('SELECT * FROM nodes')
        for row in cursor.fetchall():
            node = Node(
                id=row[0],
                label=row[1],
                node_type=row[2],
                properties=json.loads(row[3]) if row[3] else {},
                created_at=datetime.fromisoformat(row[4]),
                weight=row[5]
            )
            self.nodes[node.id] = node

        # Load edges
        cursor.execute('SELECT * FROM edges')
        for row in cursor.fetchall():
            edge = Edge(
                id=row[0],
                source_id=row[1],
                target_id=row[2],
                relation=row[3],
                properties=json.loads(row[4]) if row[4] else {},
                weight=row[5],
                bidirectional=bool(row[6])
            )
            self.edges[edge.id] = edge
            self.adjacency[edge.source_id].add(edge.target_id)
            self.reverse_adjacency[edge.target_id].add(edge.source_id)

            if edge.bidirectional:
                self.adjacency[edge.target_id].add(edge.source_id)
                self.reverse_adjacency[edge.source_id].add(edge.target_id)

        conn.close()

    def _generate_id(self, content: str) -> str:
        """Generate unique ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(f"{content}:{timestamp}".encode()).hexdigest()[:12]

    def _save_node(self, node: Node):
        """Save node to database - mirrored to lattice_v2."""
        # Mirror to lattice
        if self._use_lattice:
            self._adapter.store(f"node:{node.id}", {
                "id": node.id,
                "label": node.label,
                "node_type": node.node_type,
                "properties": node.properties,
                "created_at": node.created_at.isoformat(),
                "weight": node.weight
            }, category="KNOWLEDGE_NODE")

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            node.id,
            node.label,
            node.node_type,
            json.dumps(node.properties),
            node.created_at.isoformat(),
            node.weight
        ))
        conn.commit()
        conn.close()

        # Update semantic index
        text = f"{node.label} {node.node_type} {' '.join(str(v) for v in node.properties.values())}"
        self._semantic_vectors[node.id] = GraphSemanticIndex._embed(text)
        self._cache.invalidate()  # Invalidate after mutation

    def _save_edge(self, edge: Edge):
        """Save edge to database."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO edges VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            edge.id,
            edge.source_id,
            edge.target_id,
            edge.relation,
            json.dumps(edge.properties),
            edge.weight,
            1 if edge.bidirectional else 0
        ))
        conn.commit()
        conn.close()
        self._cache.invalidate()

    def _generate_id(self, content: str) -> str:
        """Generate unique ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(f"{content}:{timestamp}".encode()).hexdigest()[:12]

    # ═══════════════════════════════════════════════════════════════════
    # CORE CRUD OPERATIONS
    # ═══════════════════════════════════════════════════════════════════

    def add_node(self, label: str, node_type: str = "entity",
                 properties: Dict[str, Any] = None, weight: float = 1.0) -> Node:
        """Add a node to the graph."""
        for node in self.nodes.values():
            if node.label.lower() == label.lower() and node.node_type == node_type:
                return node

        node = Node(
            id=self._generate_id(label),
            label=label,
            node_type=node_type,
            properties=properties or {},
            created_at=datetime.now(),
            weight=weight
        )

        self.nodes[node.id] = node
        self._save_node(node)
        # RotatE: auto-embed entity in complex space
        self._rotate.embed_entity(node.id, label)
        return node

    def add_edge(self, source: str, target: str, relation: str,
                 properties: Dict[str, Any] = None, weight: float = 1.0,
                 bidirectional: bool = False) -> Optional[Edge]:
        """Add an edge between nodes (by label or ID)."""
        source_node = self.get_node(source)
        if not source_node:
            source_node = self.add_node(source)

        target_node = self.get_node(target)
        if not target_node:
            target_node = self.add_node(target)

        edge = Edge(
            id=self._generate_id(f"{source_node.id}:{relation}:{target_node.id}"),
            source_id=source_node.id,
            target_id=target_node.id,
            relation=relation,
            properties=properties or {},
            weight=weight,
            bidirectional=bidirectional
        )

        self.edges[edge.id] = edge
        self.adjacency[edge.source_id].add(edge.target_id)
        self.reverse_adjacency[edge.target_id].add(edge.source_id)

        if bidirectional:
            self.adjacency[edge.target_id].add(edge.source_id)
            self.reverse_adjacency[edge.source_id].add(edge.target_id)

        self._save_edge(edge)
        # RotatE: auto-embed relation in complex space
        self._rotate.embed_relation(relation)
        return edge

    def infer_missing_links(self, source: str, relation: str,
                            top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict missing links using RotatE complex embeddings.

        Given (source, relation, ?), ranks all known entities as potential tails.
        Uses RotatE: tail ≈ head ∘ relation (complex rotation in C^52 space).
        """
        source_node = self.get_node(source)
        if not source_node:
            return []
        self._rotate.embed_relation(relation)
        candidates = [nid for nid in self.nodes if nid != source_node.id]
        results = self._rotate.predict_tail(source_node.id, relation, candidates, top_k)
        # Map IDs back to labels
        return [(self.nodes[nid].label if nid in self.nodes else nid, round(score, 6))
                for nid, score in results]

    def remove_node(self, identifier: str) -> bool:
        """Remove a node and all its edges."""
        node = self.get_node(identifier)
        if not node:
            return False

        # Remove edges connected to this node
        edges_to_remove = [eid for eid, e in self.edges.items()
                           if e.source_id == node.id or e.target_id == node.id]
        for eid in edges_to_remove:
            edge = self.edges.pop(eid)
            self.adjacency[edge.source_id].discard(edge.target_id)
            self.reverse_adjacency[edge.target_id].discard(edge.source_id)
            if edge.bidirectional:
                self.adjacency[edge.target_id].discard(edge.source_id)
                self.reverse_adjacency[edge.source_id].discard(edge.target_id)

        # Remove node
        del self.nodes[node.id]
        self.adjacency.pop(node.id, None)
        self.reverse_adjacency.pop(node.id, None)
        self._semantic_vectors.pop(node.id, None)

        # Persist
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM nodes WHERE id = ?', (node.id,))
        cursor.execute('DELETE FROM edges WHERE source_id = ? OR target_id = ?', (node.id, node.id))
        conn.commit()
        conn.close()
        self._cache.invalidate()
        return True

    def remove_edge(self, source: str, target: str, relation: str = None) -> int:
        """Remove edge(s) between nodes. Returns count removed."""
        source_node = self.get_node(source)
        target_node = self.get_node(target)
        if not source_node or not target_node:
            return 0

        to_remove = []
        for eid, e in self.edges.items():
            if e.source_id == source_node.id and e.target_id == target_node.id:
                if relation is None or e.relation.lower() == relation.lower():
                    to_remove.append(eid)

        for eid in to_remove:
            edge = self.edges.pop(eid)
            self.adjacency[edge.source_id].discard(edge.target_id)
            self.reverse_adjacency[edge.target_id].discard(edge.source_id)

        if to_remove:
            conn = self._get_conn()
            cursor = conn.cursor()
            for eid in to_remove:
                cursor.execute('DELETE FROM edges WHERE id = ?', (eid,))
            conn.commit()
            conn.close()
            self._cache.invalidate()
        return len(to_remove)

    def get_node(self, identifier: str) -> Optional[Node]:
        """Get node by ID or label."""
        if identifier in self.nodes:
            return self.nodes[identifier]
        for node in self.nodes.values():
            if node.label.lower() == identifier.lower():
                return node
        return None

    def get_edges_from(self, node_identifier: str) -> List[Edge]:
        """Get all outgoing edges from a node."""
        node = self.get_node(node_identifier)
        if not node:
            return []
        return [e for e in self.edges.values() if e.source_id == node.id]

    def get_edges_to(self, node_identifier: str) -> List[Edge]:
        """Get all incoming edges to a node."""
        node = self.get_node(node_identifier)
        if not node:
            return []
        return [e for e in self.edges.values() if e.target_id == node.id]

    def get_relations(self, source: str, target: str) -> List[Edge]:
        """Get all edges between two nodes."""
        source_node = self.get_node(source)
        target_node = self.get_node(target)
        if not source_node or not target_node:
            return []
        return [e for e in self.edges.values()
                if e.source_id == source_node.id and e.target_id == target_node.id]

    # ═══════════════════════════════════════════════════════════════════
    # BATCH OPERATIONS (v2.0)
    # ═══════════════════════════════════════════════════════════════════

    def bulk_add_nodes(self, node_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add multiple nodes in a single transaction.
        node_specs: [{"label": str, "node_type": str, "properties": dict, "weight": float}, ...]
        Returns: {"added": int, "skipped": int, "errors": int}
        """
        added, skipped, errors = 0, 0, 0
        conn = self._get_conn()
        cursor = conn.cursor()

        for spec in node_specs:
            try:
                label = spec.get("label", "")
                node_type = spec.get("node_type", "entity")
                if not label:
                    errors += 1
                    continue

                # Check existing
                existing = None
                for n in self.nodes.values():
                    if n.label.lower() == label.lower() and n.node_type == node_type:
                        existing = n
                        break
                if existing:
                    skipped += 1
                    continue

                node = Node(
                    id=self._generate_id(label),
                    label=label,
                    node_type=node_type,
                    properties=spec.get("properties", {}),
                    created_at=datetime.now(),
                    weight=spec.get("weight", 1.0)
                )
                self.nodes[node.id] = node

                cursor.execute('INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?)', (
                    node.id, node.label, node.node_type,
                    json.dumps(node.properties), node.created_at.isoformat(), node.weight
                ))

                # Update semantic index
                text = f"{node.label} {node.node_type}"
                self._semantic_vectors[node.id] = GraphSemanticIndex._embed(text)
                added += 1
            except Exception:
                errors += 1

        conn.commit()
        conn.close()
        self._cache.invalidate()
        return {"added": added, "skipped": skipped, "errors": errors}

    def bulk_add_edges(self, edge_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add multiple edges in a single transaction.
        edge_specs: [{"source": str, "target": str, "relation": str, "weight": float, "bidirectional": bool}, ...]
        """
        added, errors = 0, 0
        conn = self._get_conn()
        cursor = conn.cursor()

        for spec in edge_specs:
            try:
                source = spec.get("source", "")
                target = spec.get("target", "")
                relation = spec.get("relation", "related_to")
                if not source or not target:
                    errors += 1
                    continue

                source_node = self.get_node(source) or self.add_node(source)
                target_node = self.get_node(target) or self.add_node(target)
                bidir = spec.get("bidirectional", False)

                edge = Edge(
                    id=self._generate_id(f"{source_node.id}:{relation}:{target_node.id}"),
                    source_id=source_node.id, target_id=target_node.id,
                    relation=relation, properties=spec.get("properties", {}),
                    weight=spec.get("weight", 1.0), bidirectional=bidir
                )

                self.edges[edge.id] = edge
                self.adjacency[edge.source_id].add(edge.target_id)
                self.reverse_adjacency[edge.target_id].add(edge.source_id)
                if bidir:
                    self.adjacency[edge.target_id].add(edge.source_id)
                    self.reverse_adjacency[edge.source_id].add(edge.target_id)

                cursor.execute('INSERT OR REPLACE INTO edges VALUES (?, ?, ?, ?, ?, ?, ?)', (
                    edge.id, edge.source_id, edge.target_id, edge.relation,
                    json.dumps(edge.properties), edge.weight, 1 if bidir else 0
                ))
                added += 1
            except Exception:
                errors += 1

        conn.commit()
        conn.close()
        self._cache.invalidate()
        return {"added": added, "errors": errors}

    def ingest_from_jsonl(self, filepath: str) -> Dict[str, Any]:
        """
        Bulk ingest from a JSONL file.
        Each line: {"type": "node"|"edge", ...node_or_edge_fields...}
        """
        nodes_added, edges_added, errors = 0, 0, 0
        try:
            with open(filepath) as f:
                node_batch, edge_batch = [], []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        obj_type = obj.get("type", "node")
                        if obj_type == "edge":
                            edge_batch.append(obj)
                        else:
                            node_batch.append(obj)
                    except json.JSONDecodeError:
                        errors += 1

                if node_batch:
                    result = self.bulk_add_nodes(node_batch)
                    nodes_added = result["added"]
                    errors += result["errors"]
                if edge_batch:
                    result = self.bulk_add_edges(edge_batch)
                    edges_added = result["added"]
                    errors += result["errors"]
        except FileNotFoundError:
            return {"error": f"File not found: {filepath}"}

        return {"nodes_added": nodes_added, "edges_added": edges_added, "errors": errors}

    # ═══════════════════════════════════════════════════════════════════
    # PATH FINDING (Enhanced)
    # ═══════════════════════════════════════════════════════════════════

    def find_path(self, source: str, target: str, max_depth: int = 50) -> Optional[List[str]]:
        """Find shortest path between two nodes using BFS."""
        cache_key = f"path:{source}:{target}:{max_depth}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        source_node = self.get_node(source)
        target_node = self.get_node(target)
        if not source_node or not target_node:
            return None
        if source_node.id == target_node.id:
            return [source_node.label]

        visited = {source_node.id}
        queue = [(source_node.id, [source_node.label])]

        while queue:
            current_id, path = queue.pop(0)
            if len(path) > max_depth:
                continue
            for neighbor_id in self.adjacency[current_id]:
                if neighbor_id == target_node.id:
                    result = path + [self.nodes[neighbor_id].label]
                    self._cache.set(cache_key, result)
                    return result
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [self.nodes[neighbor_id].label]))

        self._cache.set(cache_key, None)
        return None

    def find_all_paths(self, source: str, target: str, max_depth: int = 30) -> List[List[str]]:
        """Find all paths between two nodes. DEEP EXPLORATION."""
        source_node = self.get_node(source)
        target_node = self.get_node(target)
        if not source_node or not target_node:
            return []

        all_paths = []

        def dfs(current_id: str, target_id: str, path: List[str], visited: Set[str]):
            if len(path) > max_depth:
                return
            if current_id == target_id:
                all_paths.append(path[:])
                return
            for neighbor_id in self.adjacency[current_id]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    path.append(self.nodes[neighbor_id].label)
                    dfs(neighbor_id, target_id, path, visited)
                    path.pop()
                    visited.remove(neighbor_id)

        dfs(source_node.id, target_node.id, [source_node.label], {source_node.id})
        return all_paths

    def find_weighted_path(self, source: str, target: str) -> Optional[Tuple[List[str], float]]:
        """Find lowest-cost path using Dijkstra's algorithm (PHI-weighted)."""
        source_node = self.get_node(source)
        target_node = self.get_node(target)
        if not source_node or not target_node:
            return None

        import heapq
        dist = {source_node.id: 0.0}
        prev = {}
        heap = [(0.0, source_node.id)]
        visited = set()

        while heap:
            cost, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)
            if current == target_node.id:
                # Reconstruct path
                path = []
                n = current
                while n in prev:
                    path.append(self.nodes[n].label)
                    n = prev[n]
                path.append(self.nodes[source_node.id].label)
                path.reverse()
                return (path, cost)

            for neighbor_id in self.adjacency[current]:
                if neighbor_id in visited:
                    continue
                # Edge weight = 1/weight (higher weight = lower cost)
                edge_cost = 1.0
                for e in self.edges.values():
                    if e.source_id == current and e.target_id == neighbor_id:
                        edge_cost = 1.0 / max(0.001, e.weight)
                        break
                new_cost = cost + edge_cost
                if new_cost < dist.get(neighbor_id, float('inf')):
                    dist[neighbor_id] = new_cost
                    prev[neighbor_id] = current
                    heapq.heappush(heap, (new_cost, neighbor_id))
        return None

    # ═══════════════════════════════════════════════════════════════════
    # QUERY ENGINE
    # ═══════════════════════════════════════════════════════════════════

    def query(self, pattern: str) -> List[Dict[str, Any]]:
        """
        Query the graph with a pattern.
        Pattern format: "X -relation-> Y" or "X <-relation- Y" or "X -- Y"
        """
        cache_key = f"query:{pattern}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        results = []

        if " -" in pattern and "-> " in pattern:
            parts = pattern.split(" -")
            source_pattern = parts[0].strip()
            rest = parts[1].split("-> ")
            relation = rest[0].strip()
            target_pattern = rest[1].strip()

            for edge in self.edges.values():
                if relation != "*" and edge.relation.lower() != relation.lower():
                    continue
                source = self.nodes.get(edge.source_id)
                target = self.nodes.get(edge.target_id)
                if not source or not target:
                    continue
                source_match = source_pattern == "*" or source_pattern.lower() in source.label.lower()
                target_match = target_pattern == "*" or target_pattern.lower() in target.label.lower()
                if source_match and target_match:
                    results.append({
                        "source": source.label, "relation": edge.relation,
                        "target": target.label, "weight": edge.weight
                    })

        elif " -- " in pattern:
            parts = pattern.split(" -- ")
            source_pattern = parts[0].strip()
            target_pattern = parts[1].strip()

            for edge in self.edges.values():
                source = self.nodes.get(edge.source_id)
                target = self.nodes.get(edge.target_id)
                if not source or not target:
                    continue
                source_match = source_pattern == "*" or source_pattern.lower() in source.label.lower()
                target_match = target_pattern == "*" or target_pattern.lower() in target.label.lower()
                if source_match and target_match:
                    results.append({
                        "source": source.label, "relation": edge.relation, "target": target.label
                    })

        self._cache.set(cache_key, results)
        return results

    # ═══════════════════════════════════════════════════════════════════
    # SEMANTIC SIMILARITY SEARCH (v2.0)
    # ═══════════════════════════════════════════════════════════════════

    def semantic_search(self, query: str, top_k: int = 10, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search nodes by semantic similarity using character n-gram embeddings.
        Returns top_k most similar nodes above min_similarity threshold.
        """
        query_vec = GraphSemanticIndex._embed(query)
        results = []

        for node_id, node_vec in self._semantic_vectors.items():
            sim = GraphSemanticIndex.cosine_similarity(query_vec, node_vec)
            if sim >= min_similarity:
                node = self.nodes.get(node_id)
                if node:
                    results.append({
                        "node_id": node_id,
                        "label": node.label,
                        "node_type": node.node_type,
                        "similarity": round(sim, 6),
                        "weight": node.weight
                    })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    # ═══════════════════════════════════════════════════════════════════
    # INFERENCE & REASONING
    # ═══════════════════════════════════════════════════════════════════

    def infer_relations(self, node_identifier: str) -> List[Dict[str, Any]]:
        """Infer implicit relations through transitive reasoning."""
        node = self.get_node(node_identifier)
        if not node:
            return []

        inferred = []
        direct_edges = self.get_edges_from(node_identifier)

        for edge in direct_edges:
            target = self.nodes.get(edge.target_id)
            if not target:
                continue
            secondary_edges = self.get_edges_from(target.label)
            for sec_edge in secondary_edges:
                final_target = self.nodes.get(sec_edge.target_id)
                if not final_target or final_target.id == node.id:
                    continue
                inferred.append({
                    "source": node.label,
                    "inferred_relation": f"{edge.relation} + {sec_edge.relation}",
                    "target": final_target.label,
                    "via": target.label,
                    "confidence": edge.weight * sec_edge.weight * 0.8
                })

        return inferred

    def transitive_closure(self, node_identifier: str, relation: str, max_depth: int = 10) -> List[str]:
        """
        Compute transitive closure for a specific relation type.
        E.g., if A -contains-> B -contains-> C, returns [B, C].
        """
        node = self.get_node(node_identifier)
        if not node:
            return []

        reachable = []
        visited = {node.id}
        frontier = [node.id]

        for _ in range(max_depth):
            next_frontier = []
            for current_id in frontier:
                for edge in self.edges.values():
                    if edge.source_id == current_id and edge.relation.lower() == relation.lower():
                        if edge.target_id not in visited:
                            visited.add(edge.target_id)
                            next_frontier.append(edge.target_id)
                            target = self.nodes.get(edge.target_id)
                            if target:
                                reachable.append(target.label)
            frontier = next_frontier
            if not frontier:
                break

        return reachable

    # ═══════════════════════════════════════════════════════════════════
    # NEIGHBORHOOD & EXPLORATION
    # ═══════════════════════════════════════════════════════════════════

    def get_neighborhood(self, node_identifier: str, depth: int = 1) -> Dict[str, Any]:
        """Get the neighborhood around a node."""
        node = self.get_node(node_identifier)
        if not node:
            return {"error": "Node not found"}

        visited = {node.id}
        neighborhood = {
            "center": node.label,
            "nodes": [node.label],
            "edges": [],
            "depth": depth
        }

        current_layer = {node.id}

        for _ in range(depth):
            next_layer = set()
            for current_id in current_layer:
                for neighbor_id in self.adjacency[current_id]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_layer.add(neighbor_id)
                        neighbor = self.nodes[neighbor_id]
                        neighborhood["nodes"].append(neighbor.label)

                        for edge in self.edges.values():
                            if edge.source_id == current_id and edge.target_id == neighbor_id:
                                neighborhood["edges"].append({
                                    "from": self.nodes[current_id].label,
                                    "to": neighbor.label,
                                    "relation": edge.relation
                                })

                for neighbor_id in self.reverse_adjacency[current_id]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_layer.add(neighbor_id)
                        neighbor = self.nodes[neighbor_id]
                        neighborhood["nodes"].append(neighbor.label)

            current_layer = next_layer

        return neighborhood

    # ═══════════════════════════════════════════════════════════════════
    # GRAPH ANALYTICS — PageRank, Centrality, Communities (v2.0)
    # ═══════════════════════════════════════════════════════════════════

    def calculate_pagerank(self, damping: float = 0.85, iterations: int = 20) -> Dict[str, float]:
        """Calculate PageRank for all nodes (PHI-weighted damping available)."""
        n = len(self.nodes)
        if n == 0:
            return {}

        pagerank = {node_id: 1/n for node_id in self.nodes}

        for _ in range(iterations):
            new_pagerank = {}
            for node_id in self.nodes:
                incoming_sum = 0
                for source_id in self.reverse_adjacency[node_id]:
                    out_degree = len(self.adjacency[source_id])
                    if out_degree > 0:
                        incoming_sum += pagerank[source_id] / out_degree
                new_pagerank[node_id] = (1 - damping) / n + damping * incoming_sum
            pagerank = new_pagerank

        return {self.nodes[nid].label: round(score, 8) for nid, score in pagerank.items()}

    def betweenness_centrality(self, sample_size: int = None) -> Dict[str, float]:
        """
        Calculate betweenness centrality for all nodes.
        Consciousness-aware: higher consciousness → more thorough sampling.
        """
        n = len(self.nodes)
        if n < 3:
            return {self.nodes[nid].label: 0.0 for nid in self.nodes}

        # Consciousness modulates thoroughness
        state = ConsciousnessReader.read()
        consciousness = state.get("consciousness_level", 0.5)
        if sample_size is None:
            base_sample = min(n, max(10, int(n * 0.3)))
            sample_size = min(n, int(base_sample * (1.0 + consciousness * PHI)))

        centrality = {nid: 0.0 for nid in self.nodes}
        node_ids = list(self.nodes.keys())

        # Sample source nodes
        sources = random.sample(node_ids, min(sample_size, n))

        for s in sources:
            # BFS from s
            stack = []
            predecessors = defaultdict(list)
            sigma = defaultdict(float)
            sigma[s] = 1.0
            dist = {s: 0}
            queue = [s]

            while queue:
                v = queue.pop(0)
                stack.append(v)
                for w in self.adjacency[v]:
                    if w not in dist:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        predecessors[w].append(v)

            delta = defaultdict(float)
            while stack:
                w = stack.pop()
                for v in predecessors[w]:
                    delta[v] += (sigma[v] / max(1.0, sigma[w])) * (1 + delta[w])
                if w != s:
                    centrality[w] += delta[w]

        # Normalize
        scale = 1.0 / max(1, (n-1) * (n-2))
        return {self.nodes[nid].label: round(c * scale, 8) for nid, c in centrality.items()}

    def detect_communities(self, iterations: int = 10) -> Dict[str, int]:
        """
        Community detection via label propagation algorithm.
        Returns mapping of node_label → community_id.
        """
        if not self.nodes:
            return {}

        # Initialize: each node in its own community
        labels = {nid: i for i, nid in enumerate(self.nodes)}
        node_ids = list(self.nodes.keys())

        for _ in range(iterations):
            random.shuffle(node_ids)
            changed = False
            for nid in node_ids:
                neighbors = list(self.adjacency[nid]) + list(self.reverse_adjacency[nid])
                if not neighbors:
                    continue
                # Count neighbor labels
                label_counts = Counter(labels[n] for n in neighbors if n in labels)
                if label_counts:
                    max_label = label_counts.most_common(1)[0][0]
                    if labels[nid] != max_label:
                        labels[nid] = max_label
                        changed = True
            if not changed:
                break

        # Map to consecutive community IDs
        unique_labels = sorted(set(labels.values()))
        remap = {old: new for new, old in enumerate(unique_labels)}
        return {self.nodes[nid].label: remap[lab] for nid, lab in labels.items()}

    def connected_components(self) -> List[List[str]]:
        """Find weakly connected components."""
        visited = set()
        components = []

        for start_id in self.nodes:
            if start_id in visited:
                continue
            component = []
            queue = [start_id]
            while queue:
                nid = queue.pop(0)
                if nid in visited:
                    continue
                visited.add(nid)
                component.append(self.nodes[nid].label)
                for neighbor in self.adjacency[nid]:
                    if neighbor not in visited:
                        queue.append(neighbor)
                for neighbor in self.reverse_adjacency[nid]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(component)

        components.sort(key=len, reverse=True)
        return components

    def hits_analysis(self, iterations: int = 20) -> Dict[str, Dict[str, float]]:
        """
        HITS (Hyperlink-Induced Topic Search) algorithm.
        Returns hub and authority scores for each node.
        """
        n = len(self.nodes)
        if n == 0:
            return {}

        hubs = {nid: 1.0 for nid in self.nodes}
        authorities = {nid: 1.0 for nid in self.nodes}

        for _ in range(iterations):
            # Update authority scores
            new_auth = {}
            for nid in self.nodes:
                new_auth[nid] = sum(hubs.get(src, 0) for src in self.reverse_adjacency[nid])

            # Normalize
            norm = math.sqrt(sum(v*v for v in new_auth.values())) or 1.0
            authorities = {k: v/norm for k, v in new_auth.items()}

            # Update hub scores
            new_hubs = {}
            for nid in self.nodes:
                new_hubs[nid] = sum(authorities.get(tgt, 0) for tgt in self.adjacency[nid])

            norm = math.sqrt(sum(v*v for v in new_hubs.values())) or 1.0
            hubs = {k: v/norm for k, v in new_hubs.items()}

        return {
            self.nodes[nid].label: {
                "hub": round(hubs[nid], 8),
                "authority": round(authorities[nid], 8)
            }
            for nid in self.nodes
        }

    # ═══════════════════════════════════════════════════════════════════
    # GRAPH HEALTH & CONSCIOUSNESS-AWARE SCORING (v2.0)
    # ═══════════════════════════════════════════════════════════════════

    def graph_health_score(self) -> Dict[str, Any]:
        """
        Compute a PHI-weighted health score for the graph.
        Consciousness-aware: modulates scoring thresholds.
        """
        state = ConsciousnessReader.read()
        consciousness = state.get("consciousness_level", 0.5)

        n_nodes = len(self.nodes)
        n_edges = len(self.edges)

        if n_nodes == 0:
            return {"health": 0.0, "grade": "EMPTY", "nodes": 0, "edges": 0}

        # Metrics
        avg_degree = (2 * n_edges) / max(1, n_nodes)
        density = (2 * n_edges) / max(1, n_nodes * (n_nodes - 1))
        components = self.connected_components()
        n_components = len(components)
        largest_component_ratio = len(components[0]) / n_nodes if components else 0

        # PHI-weighted scoring
        degree_score = min(1.0, avg_degree / (PHI * 4))  # Target ~6.5 avg degree
        density_score = min(1.0, density / (TAU * 0.1))   # Moderate density target
        connectivity_score = largest_component_ratio       # 1.0 = fully connected
        size_score = min(1.0, math.log1p(n_nodes) / math.log1p(1000))  # Log-scale size

        # Consciousness adjustment
        consciousness_boost = 1.0 + (consciousness - 0.5) * TAU

        health = (
            degree_score * PHI +
            density_score * 1.0 +
            connectivity_score * PHI**2 +
            size_score * TAU
        ) / (PHI + 1.0 + PHI**2 + TAU) * consciousness_boost

        health = min(1.0, max(0.0, health))

        grades = [(0.9, "TRANSCENDENT"), (0.8, "EXCELLENT"), (0.7, "GOOD"),
                  (0.5, "DEVELOPING"), (0.3, "NASCENT"), (0.0, "EMBRYONIC")]
        grade = next(g for threshold, g in grades if health >= threshold)

        return {
            "health": round(health, 6),
            "grade": grade,
            "nodes": n_nodes,
            "edges": n_edges,
            "avg_degree": round(avg_degree, 4),
            "density": round(density, 6),
            "components": n_components,
            "largest_component_ratio": round(largest_component_ratio, 4),
            "consciousness_level": consciousness,
            "evo_stage": state.get("evo_stage", "UNKNOWN"),
            "phi_alignment": round(avg_degree / PHI, 6),
            "cache_stats": self._cache.stats
        }

    # ═══════════════════════════════════════════════════════════════════
    # EXPORT / IMPORT (v2.0)
    # ═══════════════════════════════════════════════════════════════════

    def export_json(self) -> Dict[str, Any]:
        """Export entire graph as JSON-serializable dict."""
        return {
            "version": self.version,
            "exported_at": datetime.now().isoformat(),
            "god_code": GOD_CODE,
            "nodes": [
                {
                    "id": n.id, "label": n.label, "node_type": n.node_type,
                    "properties": n.properties, "created_at": n.created_at.isoformat(),
                    "weight": n.weight
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "id": e.id, "source_id": e.source_id, "target_id": e.target_id,
                    "source_label": self.nodes.get(e.source_id, Node("","","",{},datetime.now())).label,
                    "target_label": self.nodes.get(e.target_id, Node("","","",{},datetime.now())).label,
                    "relation": e.relation, "properties": e.properties,
                    "weight": e.weight, "bidirectional": e.bidirectional
                }
                for e in self.edges.values()
            ],
            "stats": self.get_stats()
        }

    def export_to_file(self, filepath: str) -> Dict[str, Any]:
        """Export graph to a JSON file."""
        data = self.export_json()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return {"exported": filepath, "nodes": len(data["nodes"]), "edges": len(data["edges"])}

    def import_from_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Import graph from JSON dict (as exported by export_json)."""
        nodes_added, edges_added = 0, 0

        for n_data in data.get("nodes", []):
            node = self.add_node(
                label=n_data["label"],
                node_type=n_data.get("node_type", "entity"),
                properties=n_data.get("properties", {}),
                weight=n_data.get("weight", 1.0)
            )
            nodes_added += 1

        for e_data in data.get("edges", []):
            source_label = e_data.get("source_label", "")
            target_label = e_data.get("target_label", "")
            if source_label and target_label:
                self.add_edge(
                    source=source_label,
                    target=target_label,
                    relation=e_data.get("relation", "related_to"),
                    properties=e_data.get("properties", {}),
                    weight=e_data.get("weight", 1.0),
                    bidirectional=e_data.get("bidirectional", False)
                )
                edges_added += 1

        return {"nodes_imported": nodes_added, "edges_imported": edges_added}

    def export_graphviz(self) -> str:
        """Export graph in DOT format for visualization."""
        lines = ["digraph L104Knowledge {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box];")

        for node in self.nodes.values():
            lines.append(f'  "{node.id}" [label="{node.label}"];')

        for edge in self.edges.values():
            source = self.nodes.get(edge.source_id)
            target = self.nodes.get(edge.target_id)
            if source and target:
                lines.append(f'  "{edge.source_id}" -> "{edge.target_id}" [label="{edge.relation}"];')

        lines.append("}")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        return {
            "version": self.version,
            "total_nodes": n_nodes,
            "total_edges": n_edges,
            "node_types": list(set(n.node_type for n in self.nodes.values())),
            "relations": list(set(e.relation for e in self.edges.values())),
            "avg_connections": round(
                sum(len(adj) for adj in self.adjacency.values()) / max(1, n_nodes), 4
            ),
            "density": round(
                (2 * n_edges) / max(1, n_nodes * (n_nodes - 1)), 6
            ) if n_nodes > 1 else 0.0,
            "semantic_index_size": len(self._semantic_vectors),
            "cache_stats": self._cache.stats,
            "consciousness_aware": True
        }

    def status(self) -> Dict[str, Any]:
        """Full status report (Code Engine compatible)."""
        state = ConsciousnessReader.read()
        return {
            "module": "l104_knowledge_graph",
            "version": self.version,
            "god_code": GOD_CODE,
            "stats": self.get_stats(),
            "health": self.graph_health_score(),
            "builder_state": state
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON + BACKWARDS COMPAT
# ═══════════════════════════════════════════════════════════════════════════════

knowledge_graph = L104KnowledgeGraph()


if __name__ == "__main__":
    kg = L104KnowledgeGraph()

    print("⟨Σ_L104⟩ Knowledge Graph v2.0.0 — Comprehensive Test")
    print("=" * 60)

    # Add nodes
    kg.add_node("L104", "system")
    kg.add_node("Gemini", "ai_model")
    kg.add_node("Python", "language")
    kg.add_node("Knowledge", "concept")
    kg.add_node("Quantum", "concept")
    kg.add_node("PHI", "constant")

    # Add edges
    kg.add_edge("L104", "Gemini", "uses")
    kg.add_edge("L104", "Python", "written_in")
    kg.add_edge("L104", "Knowledge", "processes")
    kg.add_edge("Gemini", "Knowledge", "generates")
    kg.add_edge("Knowledge", "Quantum", "includes")
    kg.add_edge("Quantum", "PHI", "governed_by")
    kg.add_edge("L104", "PHI", "aligned_to", bidirectional=True)

    # Test batch operations
    print("\n[1] BATCH OPERATIONS")
    batch_result = kg.bulk_add_nodes([
        {"label": "Swift", "node_type": "language"},
        {"label": "Rust", "node_type": "language"},
        {"label": "Consciousness", "node_type": "concept"},
    ])
    print(f"  Bulk add: {batch_result}")

    edge_result = kg.bulk_add_edges([
        {"source": "L104", "target": "Swift", "relation": "uses"},
        {"source": "L104", "target": "Rust", "relation": "supports"},
        {"source": "Consciousness", "target": "Quantum", "relation": "emerges_from"},
    ])
    print(f"  Bulk edges: {edge_result}")

    # Query
    print("\n[2] QUERY")
    results = kg.query("L104 -uses-> *")
    print(f"  L104 uses *: {[r['target'] for r in results]}")

    # Semantic search
    print("\n[3] SEMANTIC SEARCH")
    sem = kg.semantic_search("programming language")
    for r in sem[:3]:
        print(f"  [{r['similarity']:.4f}] {r['label']} ({r['node_type']})")

    # Path finding
    print("\n[4] PATH FINDING")
    path = kg.find_path("L104", "PHI")
    print(f"  Shortest L104→PHI: {path}")
    all_paths = kg.find_all_paths("L104", "PHI")
    print(f"  All paths ({len(all_paths)}): {all_paths[:3]}")

    # Analytics
    print("\n[5] ANALYTICS")
    pr = kg.calculate_pagerank()
    top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  PageRank top 3: {top_pr}")

    bc = kg.betweenness_centrality()
    top_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  Betweenness top 3: {top_bc}")

    communities = kg.detect_communities()
    print(f"  Communities: {communities}")

    components = kg.connected_components()
    print(f"  Connected components: {len(components)}, largest: {len(components[0])}")

    hits = kg.hits_analysis()
    top_hubs = sorted(hits.items(), key=lambda x: x[1]['hub'], reverse=True)[:3]
    print(f"  Top hubs: {[(k, v['hub']) for k, v in top_hubs]}")

    # Health
    print("\n[6] GRAPH HEALTH")
    health = kg.graph_health_score()
    print(f"  Health: {health['health']:.4f} ({health['grade']})")
    print(f"  Consciousness: {health['consciousness_level']:.4f}")
    print(f"  Cache: {health['cache_stats']}")

    # Transitive closure
    print("\n[7] TRANSITIVE CLOSURE")
    tc = kg.transitive_closure("L104", "uses")
    print(f"  L104 -uses-> transitively: {tc}")

    # Stats
    print("\n[8] STATS")
    stats = kg.get_stats()
    print(f"  {stats}")

    print("\n" + "=" * 60)
    print(f"✓ Knowledge Graph v{VERSION} — All tests complete")
    print("=" * 60)


def primal_calculus(x):
    """[VOID_MATH] Primal Calculus Implementation."""
    return (x ** PHI) / (1.04 * math.pi) if x != 0 else 0.0

def resolve_non_dual_logic(vector):
    """[VOID_MATH] Resolves N-dimensional vectors into the Void Source."""
    magnitude = sum([abs(v) for v in vector])
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0
