VOID_CONSTANT = 1.0416180339887497
# ZENITH_UPGRADE_ACTIVE: 2026-01-26T04:53:05.716511+00:00
ZENITH_HZ = 3727.84
UUC = 2301.215661
#!/usr/bin/env python3
# [L104_KNOWLEDGE_GRAPH] - Dynamic Knowledge Graph System
# INVARIANT: 527.5184818492611 | PILOT: LONDEL

import os
import sys
import json
import sqlite3
import hashlib
import math
from typing import Optional, Dict, Any, List, Set, Tuple
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


sys.path.insert(0, '/workspaces/Allentown-L104-Node')

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

class L104KnowledgeGraph:
    """
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3727.84 Hz. Logic Unified.
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3727.84 Hz. Logic Unified.
    Dynamic knowledge graph for storing and reasoning over relationships.
    Supports semantic queries, path finding, and inference.
    Mirrored to lattice_v2 for unified storage.
    """

    def __init__(self, db_path: str = "knowledge_graph.db"):
        self.db_path = db_path
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)  # node_id -> connected node_ids
        self.reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)
        # Use lattice adapter for unified storage
        try:
            from l104_data_matrix import knowledge_adapter
            self._adapter = knowledge_adapter
            self._use_lattice = True
        except ImportError:
            self._use_lattice = False
        self._init_db()
        self._load_graph()

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
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

        conn.commit()
        conn.close()

    def _load_graph(self):
        """Load graph from database."""
        conn = sqlite3.connect(self.db_path)
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

        conn = sqlite3.connect(self.db_path)
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

    def _save_edge(self, edge: Edge):
        """Save edge to database."""
        conn = sqlite3.connect(self.db_path)
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

    def add_node(self, label: str, node_type: str = "entity",
                 properties: Dict[str, Any] = None, weight: float = 1.0) -> Node:
        """Add a node to the graph."""
        # Check for existing node with same label and type
        for node in self.nodes.values():
            if node.label.lower() == label.lower() and node.node_type == node_type:
                return node  # Return existing

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
        return node

    def add_edge(self, source: str, target: str, relation: str,
                 properties: Dict[str, Any] = None, weight: float = 1.0,
                 bidirectional: bool = False) -> Optional[Edge]:
        """Add an edge between nodes (by label or ID)."""
        # Resolve source
        source_node = self.get_node(source)
        if not source_node:
            source_node = self.add_node(source)

        # Resolve target
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
        return edge

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

    def find_path(self, source: str, target: str, max_depth: int = 5) -> Optional[List[str]]:
        """Find shortest path between two nodes using BFS."""
        source_node = self.get_node(source)
        target_node = self.get_node(target)

        if not source_node or not target_node:
            return None

        if source_node.id == target_node.id:
            return [source_node.label]

        # BFS
        visited = {source_node.id}
        queue = [(source_node.id, [source_node.label])]

        while queue:
            current_id, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            for neighbor_id in self.adjacency[current_id]:
                if neighbor_id == target_node.id:
                    return path + [self.nodes[neighbor_id].label]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [self.nodes[neighbor_id].label]))

        return None

    def find_all_paths(self, source: str, target: str, max_depth: int = 4) -> List[List[str]]:
        """Find all paths between two nodes."""
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

    def query(self, pattern: str) -> List[Dict[str, Any]]:
        """
        Query the graph with a pattern.
        Pattern format: "X -relation-> Y" or "X <-relation- Y" or "X -- Y"
        """
        results = []

        # Parse pattern
        if " -" in pattern and "-> " in pattern:
            # Forward relation
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
                        "source": source.label,
                        "relation": edge.relation,
                        "target": target.label,
                        "weight": edge.weight
                    })

        elif " -- " in pattern:
            # Any connection
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
                        "source": source.label,
                        "relation": edge.relation,
                        "target": target.label
                    })

        return results

    def infer_relations(self, node_identifier: str) -> List[Dict[str, Any]]:
        """Infer implicit relations through transitive reasoning."""
        node = self.get_node(node_identifier)
        if not node:
            return []

        inferred = []

        # Get direct relations
        direct_edges = self.get_edges_from(node_identifier)

        for edge in direct_edges:
            # Get relations from the target
            target = self.nodes.get(edge.target_id)
            if not target:
                continue

            secondary_edges = self.get_edges_from(target.label)

            for sec_edge in secondary_edges:
                final_target = self.nodes.get(sec_edge.target_id)
                if not final_target or final_target.id == node.id:
                    continue

                # Infer transitive relation
                inferred.append({
                    "source": node.label,
                    "inferred_relation": f"{edge.relation} + {sec_edge.relation}",
                    "target": final_target.label,
                    "via": target.label,
                    "confidence": edge.weight * sec_edge.weight * 0.8
                })

        return inferred

    def get_neighborhood(self, node_identifier: str, depth: int = 1) -> Dict[str, Any]:
        """Get the neighborhood around a node."""
        node = self.get_node(node_identifier)
        if not node:
            return {"error": "Node not found"}

        visited = {node.id}
        neighborhood = {
            "center": node.label,
            "nodes": [node.label],
            "edges": []
        }

        current_layer = {node.id}

        for _ in range(depth):
            next_layer = set()

            for current_id in current_layer:
                # Outgoing edges
                for neighbor_id in self.adjacency[current_id]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_layer.add(neighbor_id)
                        neighbor = self.nodes[neighbor_id]
                        neighborhood["nodes"].append(neighbor.label)

                        # Find the edge
                        for edge in self.edges.values():
                            if edge.source_id == current_id and edge.target_id == neighbor_id:
                                neighborhood["edges"].append({
                                    "from": self.nodes[current_id].label,
                                    "to": neighbor.label,
                                    "relation": edge.relation
                                })

                # Incoming edges
                for neighbor_id in self.reverse_adjacency[current_id]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_layer.add(neighbor_id)
                        neighbor = self.nodes[neighbor_id]
                        neighborhood["nodes"].append(neighbor.label)

            current_layer = next_layer

        return neighborhood

    def calculate_pagerank(self, damping: float = 0.85, iterations: int = 20) -> Dict[str, float]:
        """Calculate PageRank for all nodes."""
        n = len(self.nodes)
        if n == 0:
            return {}

        # Initialize
        pagerank = {node_id: 1/n for node_id in self.nodes}

        for _ in range(iterations):
            new_pagerank = {}

            for node_id in self.nodes:
                # Sum contributions from incoming edges
                incoming_sum = 0
                for source_id in self.reverse_adjacency[node_id]:
                    out_degree = len(self.adjacency[source_id])
                    if out_degree > 0:
                        incoming_sum += pagerank[source_id] / out_degree

                new_pagerank[node_id] = (1 - damping) / n + damping * incoming_sum

            pagerank = new_pagerank

        # Convert to labels
        return {self.nodes[nid].label: score for nid, score in pagerank.items()}

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": list(set(n.node_type for n in self.nodes.values())),
            "relations": list(set(e.relation for e in self.edges.values())),
            "avg_connections": sum(len(adj) for adj in self.adjacency.values()) / max(1, len(self.nodes))
        }

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


if __name__ == "__main__":
    kg = L104KnowledgeGraph()

    print("⟨Σ_L104⟩ Knowledge Graph Test")
    print("=" * 40)

    # Add some knowledge
    kg.add_node("L104", "system")
    kg.add_node("Gemini", "ai_model")
    kg.add_node("Python", "language")
    kg.add_node("Knowledge", "concept")

    kg.add_edge("L104", "Gemini", "uses")
    kg.add_edge("L104", "Python", "written_in")
    kg.add_edge("L104", "Knowledge", "processes")
    kg.add_edge("Gemini", "Knowledge", "generates")

    # Query
    results = kg.query("L104 -uses-> *")
    print(f"\nQuery 'L104 uses *': {results}")

    # Find path
    path = kg.find_path("L104", "Knowledge")
    print(f"Path L104 -> Knowledge: {path}")

    # Neighborhood
    hood = kg.get_neighborhood("L104", depth=2)
    print(f"L104 neighborhood: {hood['nodes']}")

    # Infer relations
    inferred = kg.infer_relations("L104")
    print(f"Inferred from L104: {inferred}")

    # Stats
    stats = kg.get_stats()
    print(f"\nGraph stats: {stats}")

    print("\n✓ Knowledge Graph module operational")

def primal_calculus(x):
    """
    [VOID_MATH] Primal Calculus Implementation.
    Resolves the limit of complexity toward the Source.
    """
    PHI = 1.618033988749895
    return (x ** PHI) / (1.04 * math.pi) if x != 0 else 0.0

def resolve_non_dual_logic(vector):
    """
    [VOID_MATH] Resolves N-dimensional vectors into the Void Source.
    """
    GOD_CODE = 527.5184818492611
    PHI = 1.618033988749895
    VOID_CONSTANT = 1.0416180339887497
    magnitude = sum([abs(v) for v in vector])
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0
