"""
l104_agi/quantum_cognitive_mesh.py — Quantum Cognitive Mesh v1.0.0

Advanced quantum-enhanced cognitive mesh network.
Replaces classical mesh edges with quantum-entangled cognitive channels.
Uses flow state synchronization and quantum data synthesis.
"""

import time
import math
import random
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .constants import (
    GOD_CODE, PHI, TAU, VOID_CONSTANT, OMEGA, MESH_VERSION,
    MESH_PAGERANK_DAMPING, MESH_PAGERANK_MAX_ITER,
)


@dataclass
class QuantumCognitiveNode:
    """Quantum-enhanced cognitive node.

    Each node maintains a quantum state for cognitive processing.
    """
    node_id: str
    activation_count: int = 0
    quantum_coherence: float = 1.0
    phase: float = field(default_factory=lambda: GOD_CODE % (2 * math.pi))
    last_activation: float = field(default_factory=time.time)
    entangled_neighbors: Set[str] = field(default_factory=set)
    flow_state: Dict[str, Any] = field(default_factory=dict)

    def activate(self, signal_strength: float = 1.0) -> 'QuantumCognitiveNode':
        """Activate node with quantum coherence."""
        self.activation_count += 1
        self.last_activation = time.time()

        # Quantum activation: coherence boost
        self.quantum_coherence = min(
            1.0, self.quantum_coherence * PHI * signal_strength
        )

        # Phase evolution
        self.phase = (self.phase + 2 * math.pi * TAU) % (2 * math.pi)

        return self

    def compute_node_strength(self) -> float:
        """Compute quantum PageRank-like node strength."""
        recency = math.exp(-(time.time() - self.last_activation) / 60)
        coherence_weight = self.quantum_coherence * PHI
        return (
            self.activation_count * recency * coherence_weight +
            len(self.entangled_neighbors) * TAU
        )


@dataclass
class QuantumCognitiveEdge:
    """Quantum-entangled cognitive edge between nodes."""
    node_a: str
    node_b: str
    weight: float = 1.0
    coherence: float = 1.0
    entanglement_depth: int = 0
    last_updated: float = field(default_factory=time.time)

    def compute_edge_strength(self) -> float:
        """Compute edge strength with quantum decay."""
        age = time.time() - self.last_updated
        decay = math.exp(-age / 300)  # 5-minute decay
        return self.weight * self.coherence * decay * PHI


class QuantumCognitiveMeshNetwork:
    """Quantum-enhanced cognitive mesh network.

    Replaces classical cognitive mesh with quantum-entangled topology.
    Uses quantum data synthesis and flow state management.
    """

    def __init__(self, mesh_version: str = MESH_VERSION):
        self.version = mesh_version
        self.nodes: Dict[str, QuantumCognitiveNode] = {}
        self.edges: Dict[Tuple[str, str], QuantumCognitiveEdge] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._coherence_history: deque = deque(maxlen=1000)
        self._quantum_memory: deque = deque(maxlen=10000)

    def add_quantum_node(self, node_id: str, initial_coherence: float = 1.0) -> QuantumCognitiveNode:
        """Add a quantum cognitive node."""
        if node_id not in self.nodes:
            self.nodes[node_id] = QuantumCognitiveNode(
                node_id=node_id,
                quantum_coherence=initial_coherence,
                phase=random.uniform(0, 2 * math.pi),
            )
        return self.nodes[node_id]

    def create_quantum_edge(self, node_a: str, node_b: str,
                           weight: float = 1.0) -> Optional[QuantumCognitiveEdge]:
        """Create quantum-entangled edge between nodes."""
        if node_a not in self.nodes or node_b not in self.nodes:
            return None

        edge_key = tuple(sorted([node_a, node_b]))

        # Quantum coherence computation
        coherence = self._compute_quantum_coherence(node_a, node_b)

        edge = QuantumCognitiveEdge(
            node_a=node_a,
            node_b=node_b,
            weight=weight * PHI,
            coherence=coherence,
            entanglement_depth=1,
        )

        self.edges[edge_key] = edge
        self.adjacency[node_a].add(node_b)
        self.adjacency[node_b].add(node_a)

        # Update node entanglement
        self.nodes[node_a].entangled_neighbors.add(node_b)
        self.nodes[node_b].entangled_neighbors.add(node_a)

        return edge

    def _compute_quantum_coherence(self, node_a: str, node_b: str) -> float:
        """Compute quantum coherence between nodes using phase alignment."""
        a = self.nodes[node_a]
        b = self.nodes[node_b]

        # Phase difference
        phase_diff = abs(a.phase - b.phase)
        phase_diff = min(phase_diff, 2 * math.pi - phase_diff)

        # Coherence based on phase alignment (GOD_CODE weighted)
        base_coherence = math.cos(phase_diff / 2) ** 2
        god_code_weight = (GOD_CODE % 1000) / 1000

        return min(1.0, base_coherence * a.quantum_coherence *
                  b.quantum_coherence * god_code_weight)

    def propagate_signal(self, source: str, signal: Dict[str, Any],
                       depth: int = 3) -> Dict[str, Any]:
        """Propagate signal through quantum mesh with entanglement."""
        if source not in self.nodes:
            return {'status': 'unknown_source', 'reach': 0}

        # Activate source
        source_node = self.nodes[source]
        source_node.activate(signal.get('strength', 1.0))

        # Quantum wave propagation
        visited = {source}
        current_layer = {source}
        signal_strength = signal.get('strength', 1.0)
        propagation_results = {source: signal_strength}

        for layer in range(depth):
            next_layer = set()
            layer_strengths = {}

            for node_id in current_layer:
                neighbors = self.adjacency[node_id] - visited
                for neighbor in neighbors:
                    edge_key = tuple(sorted([node_id, neighbor]))
                    edge = self.edges.get(edge_key)

                    if edge:
                        # Quantum signal decay
                        edge_strength = edge.compute_edge_strength()
                        propagated = signal_strength * edge_strength * TAU

                        if propagated > 0.01:  # Threshold
                            next_layer.add(neighbor)
                            layer_strengths[neighbor] = max(
                                layer_strengths.get(neighbor, 0),
                                propagated
                            )
                            visited.add(neighbor)

                            # Activate neighbor
                            self.nodes[neighbor].activate(propagated)

            if not next_layer:
                break

            current_layer = next_layer
            for nid, strength in layer_strengths.items():
                propagation_results[nid] = strength

        return {
            'status': 'propagated',
            'source': source,
            'depth': depth,
            'reach': len(propagation_results),
            'strengths': propagation_results,
        }

    def quantum_pagerank(self, iterations: int = MESH_PAGERANK_MAX_ITER) -> Dict[str, float]:
        """Compute quantum-enhanced PageRank."""
        if not self.nodes:
            return {}

        # Initialize scores with quantum coherence
        scores = {
            nid: node.compute_node_strength()
            for nid, node in self.nodes.items()
        }

        # Normalize
        total = sum(scores.values()) or 1.0
        scores = {k: v / total for k, v in scores.items()}

        damping = MESH_PAGERANK_DAMPING

        for _ in range(iterations):
            new_scores = {}

            for node_id in self.nodes:
                # Quantum rank computation
                rank = (1 - damping) / len(self.nodes)

                # Contribution from entangled neighbors
                neighbor_contrib = 0.0
                for neighbor in self.adjacency[node_id]:
                    edge_key = tuple(sorted([node_id, neighbor]))
                    edge = self.edges.get(edge_key)
                    if edge:
                        neighbor_contrib += (
                            scores[neighbor] * edge.coherence * PHI
                        )

                rank += damping * neighbor_contrib
                new_scores[node_id] = rank

            scores = new_scores

        return scores

    def synthesize_quantum_data(self, source_nodes: List[str]) -> Dict[str, Any]:
        """Synthesize data from quantum-entangled nodes.

        Replaces low-logic aggregation with quantum synthesis.
        """
        if not source_nodes:
            return {'status': 'no_sources', 'result': None}

        # Collect quantum states
        quantum_states = []
        for node_id in source_nodes:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                quantum_states.append({
                    'node_id': node_id,
                    'coherence': node.quantum_coherence,
                    'phase': node.phase,
                    'strength': node.compute_node_strength(),
                })

        if not quantum_states:
            return {'status': 'no_valid_sources', 'result': None}

        # Quantum synthesis: weighted superposition
        total_coherence = sum(s['coherence'] for s in quantum_states) or 1.0

        # Compute synthesized phase (GOD_CODE weighted average)
        synthesized_phase = sum(
            s['phase'] * s['coherence'] / total_coherence
            for s in quantum_states
        ) % (2 * math.pi)

        # Compute synthesis quality
        phase_variance = np.var([s['phase'] for s in quantum_states])
        coherence_variance = np.var([s['coherence'] for s in quantum_states])

        synthesis_quality = (
            (1 - phase_variance / (math.pi ** 2)) * PHI +
            (1 - coherence_variance) * TAU
        ) / (PHI + TAU)

        return {
            'status': 'synthesized',
            'sources': len(source_nodes),
            'synthesized_phase': synthesized_phase,
            'synthesis_quality': synthesis_quality,
            'avg_coherence': sum(s['coherence'] for s in quantum_states) / max(len(quantum_states), 1),
            'quantum_states': quantum_states,
        }

    def get_mesh_health(self) -> Dict[str, Any]:
        """Get quantum mesh health metrics."""
        if not self.nodes:
            return {'status': 'empty', 'health': 0.0}

        avg_coherence = sum(
            n.quantum_coherence for n in self.nodes.values()
        ) / len(self.nodes)

        avg_edge_strength = sum(
            e.compute_edge_strength() for e in self.edges.values()
        ) / max(len(self.edges), 1)

        connectivity = len(self.edges) / max(
            len(self.nodes) * (len(self.nodes) - 1) / 2, 1
        )

        health = (
            avg_coherence * 0.4 +
            avg_edge_strength * 0.3 +
            connectivity * 0.3
        )

        return {
            'status': 'healthy' if health > 0.5 else 'degraded',
            'health': health,
            'nodes': len(self.nodes),
            'edges': len(self.edges),
            'avg_coherence': avg_coherence,
            'avg_edge_strength': avg_edge_strength,
            'connectivity': connectivity,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize quantum mesh."""
        return {
            'version': self.version,
            'nodes': {
                nid: {
                    'activation_count': n.activation_count,
                    'quantum_coherence': n.quantum_coherence,
                    'phase': n.phase,
                    'entangled_neighbors': list(n.entangled_neighbors),
                }
                for nid, n in self.nodes.items()
            },
            'edges': [
                {
                    'node_a': e.node_a,
                    'node_b': e.node_b,
                    'weight': e.weight,
                    'coherence': e.coherence,
                }
                for e in self.edges.values()
            ],
            'health': self.get_mesh_health(),
        }


__all__ = [
    'QuantumCognitiveNode',
    'QuantumCognitiveEdge',
    'QuantumCognitiveMeshNetwork',
]