"""
L104 AGI Core — Cognitive Mesh Network v2.0
============================================================================
Decomposed from core.py mesh state into a dedicated subsystem.

Provides:
  • PageRank importance scoring for subsystems (PHI-damped iterations)
  • Community detection via label propagation on co-activation graph
  • PHI-weighted Hebbian reinforcement (co-activation → edge strengthening)
  • Mesh defragmentation — prunes weak edges, reinforces strong paths
  • Topology health scoring with GOD_CODE alignment

All edge weights are bounded [0, PHI] and decay by TAU per epoch.
INVARIANT: 527.5184818492612 | PILOT: LONDEL
============================================================================
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

# Sacred constants
PHI = 1.618033988749895
TAU = 1.0 / PHI  # ≈ 0.618
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # 527.5184818492612
VOID_CONSTANT = 1.0416180339887497


# ═══════════════════════════════════════════════════════════════════════════════
# COGNITIVE MESH NETWORK v2.0
# ═══════════════════════════════════════════════════════════════════════════════

class CognitiveMeshNetwork:
    """
    Dynamic subsystem interconnection graph with Hebbian co-activation learning.

    v2.0 enhancements:
    - PageRank importance scoring (PHI-damped convergence)
    - Label-propagation community detection
    - Mesh defragmentation with configurable pruning threshold
    - Topology health score (connectivity × GOD_CODE alignment)
    - Co-activation history with rolling window
    """

    VERSION = "2.0.0"

    # ── Configuration Constants ──────────────────────────────────────────

    PAGERANK_DAMPING = TAU               # ≈ 0.618 — golden-ratio damping
    PAGERANK_MAX_ITER = 104              # L104 signature iterations
    PAGERANK_TOL = 1e-8                  # Convergence tolerance
    EDGE_WEIGHT_MAX = PHI                # Maximum edge weight
    EDGE_DECAY_RATE = TAU * 0.1          # Per-epoch decay ≈ 0.0618
    PRUNE_THRESHOLD = 0.01              # Edges below this weight get pruned
    CO_ACTIVATION_WINDOW = 5000          # Rolling window for co-activation history
    COMMUNITY_MAX_ITER = 50              # Label propagation iterations

    def __init__(self):
        # Adjacency: node → {neighbor → weight}
        self._adjacency: Dict[str, Dict[str, float]] = {}
        # Co-activation counts (global)
        self._co_activation: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Per-node activation count
        self._activation_counts: Dict[str, int] = defaultdict(int)
        # Co-activation history (timestamped for rolling analysis)
        self._co_activation_log: deque = deque(maxlen=self.CO_ACTIVATION_WINDOW)
        # PageRank scores (computed on demand)
        self._pagerank: Dict[str, float] = {}
        self._pagerank_dirty: bool = True
        # Community assignments
        self._communities: Dict[str, int] = {}
        self._community_dirty: bool = True
        # Metrics
        self._epoch: int = 0
        self._total_reinforcements: int = 0
        self._total_prunes: int = 0
        self._last_defrag_time: float = 0.0

    # ── Node Management ──────────────────────────────────────────────────

    def add_node(self, name: str):
        """Register a subsystem node in the mesh."""
        if name not in self._adjacency:
            self._adjacency[name] = {}
            self._pagerank_dirty = True
            self._community_dirty = True

    def add_edge(self, a: str, b: str, weight: float = 0.1):
        """Add or update an edge between two subsystems."""
        self.add_node(a)
        self.add_node(b)
        w = min(weight, self.EDGE_WEIGHT_MAX)
        self._adjacency[a][b] = w
        self._adjacency[b][a] = w
        self._pagerank_dirty = True
        self._community_dirty = True

    @property
    def nodes(self) -> List[str]:
        return list(self._adjacency.keys())

    @property
    def node_count(self) -> int:
        return len(self._adjacency)

    @property
    def edge_count(self) -> int:
        count = 0
        seen: Set[Tuple[str, str]] = set()
        for a, neighbors in self._adjacency.items():
            for b in neighbors:
                pair = (min(a, b), max(a, b))
                if pair not in seen:
                    seen.add(pair)
                    count += 1
        return count

    @property
    def density(self) -> float:
        """Graph density: edges / max_possible_edges."""
        n = self.node_count
        if n < 2:
            return 0.0
        max_edges = n * (n - 1) / 2
        return self.edge_count / max_edges

    # ── Hebbian Co-Activation ────────────────────────────────────────────

    def record_activation(self, subsystem: str):
        """Record a single subsystem activation."""
        self.add_node(subsystem)
        self._activation_counts[subsystem] += 1

    def record_co_activation(self, a: str, b: str):
        """
        Record co-activation of two subsystems.
        Strengthens the edge between them (Hebbian learning rule).
        """
        if a == b:
            return
        self.add_node(a)
        self.add_node(b)

        self._co_activation[a][b] += 1
        self._co_activation[b][a] += 1
        self._activation_counts[a] += 1
        self._activation_counts[b] += 1

        # Hebbian reinforcement: Δw = PHI × (1 / (1 + co_count)) — diminishing returns
        co_count = self._co_activation[a][b]
        delta = PHI * (1.0 / (1.0 + co_count * 0.01))
        current_w = self._adjacency.get(a, {}).get(b, 0.0)
        new_w = min(current_w + delta, self.EDGE_WEIGHT_MAX)

        if a not in self._adjacency:
            self._adjacency[a] = {}
        if b not in self._adjacency:
            self._adjacency[b] = {}
        self._adjacency[a][b] = new_w
        self._adjacency[b][a] = new_w

        self._co_activation_log.append({
            "time": time.time(),
            "a": a, "b": b,
            "co_count": co_count,
            "weight": round(new_w, 6),
        })

        self._total_reinforcements += 1
        self._pagerank_dirty = True
        self._community_dirty = True

    def record_batch_co_activation(self, subsystems: List[str]):
        """Record pairwise co-activation for all subsystems in a batch."""
        for i in range(len(subsystems)):
            for j in range(i + 1, len(subsystems)):
                self.record_co_activation(subsystems[i], subsystems[j])

    # ── PageRank Importance Scoring ──────────────────────────────────────

    def compute_pagerank(self, force: bool = False) -> Dict[str, float]:
        """
        Compute PageRank importance for all mesh nodes.
        Uses PHI-damped (d = TAU ≈ 0.618) power iteration.
        Converges in ≤ 104 iterations.
        """
        if not force and not self._pagerank_dirty and self._pagerank:
            return dict(self._pagerank)

        nodes = list(self._adjacency.keys())
        n = len(nodes)
        if n == 0:
            self._pagerank = {}
            self._pagerank_dirty = False
            return {}

        d = self.PAGERANK_DAMPING
        rank = {node: 1.0 / n for node in nodes}

        for iteration in range(self.PAGERANK_MAX_ITER):
            new_rank = {}
            for node in nodes:
                # Sum incoming contributions
                incoming_sum = 0.0
                for other, neighbors in self._adjacency.items():
                    if node in neighbors and other != node:
                        out_degree = sum(neighbors.values())
                        if out_degree > 0:
                            incoming_sum += rank[other] * neighbors[node] / out_degree

                new_rank[node] = (1.0 - d) / n + d * incoming_sum

            # Check convergence
            diff = sum(abs(new_rank[n] - rank[n]) for n in nodes)
            rank = new_rank
            if diff < self.PAGERANK_TOL:
                break

        # Normalize to [0, 1]
        max_r = max(rank.values()) if rank else 1.0
        if max_r > 0:
            rank = {k: v / max_r for k, v in rank.items()}

        self._pagerank = rank
        self._pagerank_dirty = False
        return dict(rank)

    def top_subsystems(self, k: int = 10) -> List[Tuple[str, float]]:
        """Return top-K subsystems by PageRank importance."""
        pr = self.compute_pagerank()
        return sorted(pr.items(), key=lambda x: x[1], reverse=True)[:k]

    # ── Community Detection ──────────────────────────────────────────────

    def detect_communities(self, force: bool = False) -> Dict[str, int]:
        """
        Label propagation community detection on the mesh graph.
        Each node adopts the most frequent label among its weighted neighbors.
        Returns {node → community_id}.
        """
        if not force and not self._community_dirty and self._communities:
            return dict(self._communities)

        nodes = list(self._adjacency.keys())
        if not nodes:
            self._communities = {}
            self._community_dirty = False
            return {}

        # Initialize: each node is its own community
        labels = {node: i for i, node in enumerate(nodes)}

        for _ in range(self.COMMUNITY_MAX_ITER):
            changed = False
            for node in nodes:
                neighbors = self._adjacency.get(node, {})
                if not neighbors:
                    continue

                # Weighted vote: accumulate weights per label
                label_weights: Dict[int, float] = defaultdict(float)
                for neighbor, weight in neighbors.items():
                    label_weights[labels[neighbor]] += weight

                if label_weights:
                    best_label = max(label_weights, key=label_weights.get)
                    if best_label != labels[node]:
                        labels[node] = best_label
                        changed = True

            if not changed:
                break

        # Renumber communities from 0
        unique = sorted(set(labels.values()))
        remap = {old: new for new, old in enumerate(unique)}
        labels = {k: remap[v] for k, v in labels.items()}

        self._communities = labels
        self._community_dirty = False
        return dict(labels)

    def community_summary(self) -> Dict[str, Any]:
        """Get summary of detected communities."""
        communities = self.detect_communities()
        if not communities:
            return {"communities": 0, "groups": {}}

        groups: Dict[int, List[str]] = defaultdict(list)
        for node, cid in communities.items():
            groups[cid].append(node)

        return {
            "communities": len(groups),
            "groups": {str(cid): sorted(members) for cid, members in sorted(groups.items())},
            "largest_community": max(len(m) for m in groups.values()) if groups else 0,
            "singleton_count": sum(1 for m in groups.values() if len(m) == 1),
        }

    # ── Mesh Defragmentation ─────────────────────────────────────────────

    def decay_edges(self):
        """Apply TAU-based decay to all edge weights. Called once per epoch."""
        self._epoch += 1
        to_prune: List[Tuple[str, str]] = []

        for a in list(self._adjacency.keys()):
            for b in list(self._adjacency[a].keys()):
                self._adjacency[a][b] *= (1.0 - self.EDGE_DECAY_RATE)
                if self._adjacency[a][b] < self.PRUNE_THRESHOLD:
                    to_prune.append((a, b))

        for a, b in to_prune:
            self._adjacency[a].pop(b, None)
            self._adjacency[b].pop(a, None)
            self._total_prunes += 1

        self._pagerank_dirty = True
        self._community_dirty = True

    def defragment(self, prune_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Defragment the mesh: prune weak edges, remove isolated nodes,
        and re-normalize remaining edge weights.
        """
        t0 = time.time()
        threshold = prune_threshold or self.PRUNE_THRESHOLD
        pruned_edges = 0
        removed_nodes = 0

        # Prune weak edges
        for a in list(self._adjacency.keys()):
            for b in list(self._adjacency[a].keys()):
                if self._adjacency[a][b] < threshold:
                    del self._adjacency[a][b]
                    if a in self._adjacency.get(b, {}):
                        del self._adjacency[b][a]
                    pruned_edges += 1

        # Remove isolated nodes (no edges)
        isolated = [n for n, neighbors in self._adjacency.items() if not neighbors]
        for n in isolated:
            del self._adjacency[n]
            removed_nodes += 1

        # Re-normalize: scale all weights to [0, PHI]
        max_w = 0.0
        for neighbors in self._adjacency.values():
            for w in neighbors.values():
                max_w = max(max_w, w)
        if max_w > 0:
            scale = self.EDGE_WEIGHT_MAX / max_w
            for neighbors in self._adjacency.values():
                for b in neighbors:
                    neighbors[b] *= scale

        self._total_prunes += pruned_edges
        self._last_defrag_time = time.time()
        self._pagerank_dirty = True
        self._community_dirty = True

        elapsed_ms = (time.time() - t0) * 1000
        return {
            "pruned_edges": pruned_edges,
            "removed_nodes": removed_nodes,
            "remaining_nodes": self.node_count,
            "remaining_edges": self.edge_count,
            "elapsed_ms": round(elapsed_ms, 3),
        }

    # ── Topology Health Score ────────────────────────────────────────────

    def topology_health(self) -> Dict[str, Any]:
        """
        Compute mesh topology health from connectivity, PageRank distribution,
        and GOD_CODE alignment.
        """
        n = self.node_count
        e = self.edge_count

        if n == 0:
            return {"score": 0.0, "nodes": 0, "edges": 0, "diagnosis": "EMPTY"}

        # 1. Connectivity score — density relative to golden ratio target
        density = self.density
        # Target density: TAU ≈ 0.618 (not too sparse, not fully connected)
        density_score = 1.0 - abs(density - TAU) / TAU

        # 2. PageRank entropy — measure of importance distribution
        pr = self.compute_pagerank()
        pr_values = list(pr.values())
        if pr_values:
            # Shannon entropy normalized by log(n)
            total = sum(pr_values)
            entropy = 0.0
            for v in pr_values:
                p = v / total if total > 0 else 0
                if p > 0:
                    entropy -= p * math.log2(p)
            max_entropy = math.log2(n) if n > 1 else 1.0
            entropy_score = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            entropy_score = 0.0

        # 3. GOD_CODE alignment — node count modulo 104 alignment
        god_alignment = 1.0 - abs((n % 104) - 52) / 52.0

        # 4. Community coherence — fewer, larger communities = better
        comm = self.community_summary()
        num_communities = comm.get("communities", n)
        if n > 0 and num_communities > 0:
            community_score = 1.0 - (num_communities - 1) / max(n - 1, 1)
        else:
            community_score = 0.0

        # PHI-weighted composite
        health = (
            density_score * PHI / 4.0 +
            entropy_score * PHI / 4.0 +
            god_alignment * TAU / 2.0 +
            community_score * TAU / 2.0
        )
        health = max(0.0, health)

        diagnosis = "HEALTHY" if health > 0.6 else ("DEGRADED" if health > 0.3 else "CRITICAL")

        return {
            "score": round(health, 6),
            "diagnosis": diagnosis,
            "nodes": n,
            "edges": e,
            "density": round(density, 6),
            "density_score": round(density_score, 6),
            "pagerank_entropy": round(entropy_score, 6),
            "god_alignment": round(god_alignment, 6),
            "community_score": round(community_score, 6),
            "communities": num_communities,
            "epoch": self._epoch,
            "total_reinforcements": self._total_reinforcements,
            "total_prunes": self._total_prunes,
        }

    # ── Shortest Path ────────────────────────────────────────────────────

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """BFS shortest path between two subsystems (unweighted hop count)."""
        if source not in self._adjacency or target not in self._adjacency:
            return None
        if source == target:
            return [source]

        visited = {source}
        queue = deque([(source, [source])])
        while queue:
            node, path = queue.popleft()
            for neighbor in self._adjacency.get(node, {}):
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None  # No path exists

    # ── Status ───────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Full mesh network status."""
        return {
            "version": self.VERSION,
            "nodes": self.node_count,
            "edges": self.edge_count,
            "density": round(self.density, 6),
            "epoch": self._epoch,
            "total_reinforcements": self._total_reinforcements,
            "total_prunes": self._total_prunes,
            "co_activation_log_size": len(self._co_activation_log),
            "pagerank_computed": bool(self._pagerank),
            "communities_computed": bool(self._communities),
            "last_defrag": self._last_defrag_time,
        }
