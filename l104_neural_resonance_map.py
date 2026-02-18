"""
L104 Neural Resonance Map v2.0.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Neural topology mapping & activation resonance analysis engine.
Tracks subsystem activation patterns, detects resonance harmonics
between modules, identifies neural pathway hotspots, and maps
information flow topology across the ASI pipeline.

Subsystems:
  - ActivationTracker: per-node activation recording with decay
  - ResonanceAnalyzer: harmonic detection between node pairs
  - TopologyMapper: directed graph of neural pathways
  - HotspotDetector: identifies high-traffic bottleneck pathways
  - FlowVisualizer: ASCII topology rendering

Sacred Constants: GOD_CODE=527.5184818492612 | PHI=1.618033988749895
"""
# ZENITH_UPGRADE_ACTIVE: 2026-02-15T00:00:00
ZENITH_HZ = 3887.8
UUC = 2402.792541

import math
import time
import json
from pathlib import Path
from collections import deque, Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple, Set

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
# Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
PHI = 1.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
FEIGENBAUM = 4.669201609
TAU = 6.283185307179586
GROVER_AMPLIFICATION = PHI ** 3
VOID_CONSTANT = 1.0416180339887497

VERSION = "2.0.0"
STATE_FILE = Path(".l104_neural_resonance_state.json")


class ActivationTracker:
    """Tracks per-node activation counts with exponential decay."""

    def __init__(self, decay_rate: float = 0.95, max_nodes: int = 500):
        self.decay_rate = decay_rate
        self.max_nodes = max_nodes
        self._activations: Dict[str, float] = {}
        self._timestamps: Dict[str, float] = {}
        self._total_fires = 0

    def fire(self, node: str, intensity: float = 1.0):
        """Record an activation event for a node."""
        now = time.time()
        # Apply decay to existing activation
        if node in self._activations:
            dt = now - self._timestamps.get(node, now)
            self._activations[node] *= (self.decay_rate ** dt)
        else:
            self._activations[node] = 0.0

        self._activations[node] += intensity
        self._timestamps[node] = now
        self._total_fires += 1

        # Evict least-active if over limit
        if len(self._activations) > self.max_nodes:
            weakest = min(self._activations, key=self._activations.get)
            del self._activations[weakest]
            self._timestamps.pop(weakest, None)

    def get_activation(self, node: str) -> float:
        """Get current decayed activation for a node."""
        if node not in self._activations:
            return 0.0
        now = time.time()
        dt = now - self._timestamps.get(node, now)
        return self._activations[node] * (self.decay_rate ** dt)

    def top_active(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most active nodes (with decay applied)."""
        now = time.time()
        decayed = []
        for node, val in self._activations.items():
            dt = now - self._timestamps.get(node, now)
            decayed.append((node, val * (self.decay_rate ** dt)))
        decayed.sort(key=lambda x: x[1], reverse=True)
        return decayed[:n]

    @property
    def total_fires(self) -> int:
        return self._total_fires


class ResonanceAnalyzer:
    """Detects harmonic resonance between node pairs.

    When two nodes fire in close temporal proximity, they "resonate."
    Resonance strength decays with temporal distance,
    scaled by PHI for sacred harmonic weighting.
    """

    def __init__(self, window_sec: float = 2.0, max_pairs: int = 200):
        self.window_sec = window_sec
        self.max_pairs = max_pairs
        self._recent_fires: deque = deque(maxlen=100)
        self._resonance: Dict[str, float] = {}  # "A|B" -> strength

    def record_fire(self, node: str):
        """Record a fire event and check resonance with recent fires."""
        now = time.time()
        # Check resonance with recent fires
        for (other_node, other_time) in self._recent_fires:
            if other_node == node:
                continue
            dt = now - other_time
            if dt <= self.window_sec:
                # Resonance strength: inverse of temporal distance, PHI-scaled
                strength = PHI / (1.0 + dt * FEIGENBAUM)
                pair_key = "|".join(sorted([node, other_node]))
                self._resonance[pair_key] = self._resonance.get(pair_key, 0.0) + strength

                # Limit size
                if len(self._resonance) > self.max_pairs:
                    weakest_key = min(self._resonance, key=self._resonance.get)
                    del self._resonance[weakest_key]

        self._recent_fires.append((node, now))

    def strongest_resonances(self, n: int = 10) -> List[Tuple[str, str, float]]:
        """Get top N resonance pairs."""
        pairs = []
        for key, strength in self._resonance.items():
            parts = key.split("|")
            if len(parts) == 2:
                pairs.append((parts[0], parts[1], strength))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:n]

    @property
    def total_pairs(self) -> int:
        return len(self._resonance)


class TopologyMapper:
    """Directed graph of neural pathways between subsystems."""

    def __init__(self):
        self._edges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._nodes: Set[str] = set()

    def add_edge(self, src: str, dst: str, weight: float = 1.0):
        """Add or strengthen a directed edge."""
        self._nodes.add(src)
        self._nodes.add(dst)
        self._edges[src][dst] += weight

    def get_neighbors(self, node: str) -> List[Tuple[str, float]]:
        """Get outgoing neighbors with weights."""
        if node not in self._edges:
            return []
        return sorted(self._edges[node].items(), key=lambda x: x[1], reverse=True)

    def get_incoming(self, node: str) -> List[Tuple[str, float]]:
        """Get nodes that point to this node."""
        incoming = []
        for src, dsts in self._edges.items():
            if node in dsts:
                incoming.append((src, dsts[node]))
        return sorted(incoming, key=lambda x: x[1], reverse=True)

    def find_path(self, start: str, end: str, max_depth: int = 10) -> Optional[List[str]]:
        """BFS shortest path between two nodes."""
        if start == end:
            return [start]
        visited = {start}
        queue = deque([(start, [start])])
        while queue and max_depth > 0:
            node, path = queue.popleft()
            for neighbor in self._edges.get(node, {}):
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
            max_depth -= 1
        return None

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(dsts) for dsts in self._edges.values())


class HotspotDetector:
    """Identifies high-traffic neural pathway bottlenecks."""

    def __init__(self, tracker: ActivationTracker, topology: TopologyMapper):
        self._tracker = tracker
        self._topology = topology

    def detect(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Detect hotspots: high activation + high connectivity."""
        top_active = self._tracker.top_active(top_n * 3)
        hotspots = []
        for node, activation in top_active:
            fanout = len(self._topology.get_neighbors(node))
            fanin = len(self._topology.get_incoming(node))
            connectivity = fanout + fanin
            # Hotspot score: activation * sqrt(connectivity) * PHI
            score = activation * math.sqrt(max(connectivity, 1)) * PHI
            hotspots.append({
                'node': node,
                'activation': round(activation, 4),
                'fanout': fanout,
                'fanin': fanin,
                'hotspot_score': round(score, 4),
            })
        hotspots.sort(key=lambda x: x['hotspot_score'], reverse=True)
        return hotspots[:top_n]


# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL RESONANCE MAP HUB
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralResonanceMap:
    """
    Neural topology mapping & resonance analysis with 5 subsystems:

      - ActivationTracker: per-node activation with decay
      - ResonanceAnalyzer: harmonic detection between pairs
      - TopologyMapper: directed pathway graph
      - HotspotDetector: bottleneck identification
      - FlowVisualizer: ASCII topology rendering

    Pipeline Integration:
      - fire(node, intensity) → record activation + check resonance
      - connect(src, dst) → add topology edge
      - generate_map() → full topology + resonance report
      - get_hotspots() → bottleneck analysis
      - connect_to_pipeline()
    """

    def __init__(self):
        self.version = VERSION
        self._tracker = ActivationTracker()
        self._resonance = ResonanceAnalyzer()
        self._topology = TopologyMapper()
        self._hotspot = HotspotDetector(self._tracker, self._topology)
        self._pipeline_connected = False
        self._map_generations = 0

    def fire(self, node: str, intensity: float = 1.0):
        """Record a neural activation event."""
        self._tracker.fire(node, intensity)
        self._resonance.record_fire(node)

    def connect(self, src: str, dst: str, weight: float = 1.0):
        """Register a directed neural pathway."""
        self._topology.add_edge(src, dst, weight)

    def get_hotspots(self, n: int = 5) -> List[Dict]:
        return self._hotspot.detect(n)

    def get_resonances(self, n: int = 10):
        return self._resonance.strongest_resonances(n)

    def get_path(self, src: str, dst: str):
        return self._topology.find_path(src, dst)

    def generate_map(self) -> Dict[str, Any]:
        """Generate a full neural resonance map report."""
        self._map_generations += 1
        top_active = self._tracker.top_active(20)
        resonances = self._resonance.strongest_resonances(10)
        hotspots = self._hotspot.detect(5)

        return {
            'version': self.version,
            'generation': self._map_generations,
            'topology': {
                'nodes': self._topology.node_count,
                'edges': self._topology.edge_count,
            },
            'top_active': [{'node': n, 'activation': round(a, 4)} for n, a in top_active],
            'resonances': [{'a': a, 'b': b, 'strength': round(s, 4)} for a, b, s in resonances],
            'hotspots': hotspots,
            'total_fires': self._tracker.total_fires,
            'resonance_pairs': self._resonance.total_pairs,
        }

    def seed_pipeline_topology(self):
        """Seed the topology with known L104 pipeline connections."""
        pipeline = [
            ("agi_core", "asi_core"),
            ("asi_core", "code_engine"),
            ("asi_core", "neural_cascade"),
            ("asi_core", "evolution_engine"),
            ("asi_core", "self_optimizer"),
            ("agi_core", "consciousness"),
            ("consciousness", "sage_mode"),
            ("sage_mode", "knowledge_graph"),
            ("code_engine", "patch_engine"),
            ("code_engine", "polymorphic_core"),
            ("code_engine", "autonomous_innovation"),
            ("neural_cascade", "sentient_archive"),
            ("grounding_feedback", "purge_hallucinations"),
            ("compaction_filter", "presence_accelerator"),
            ("seed_matrix", "agi_core"),
            ("copilot_bridge", "agi_core"),
            ("speed_benchmark", "asi_core"),
        ]
        for src, dst in pipeline:
            self.connect(src, dst)
            self.fire(src, 0.1)
            self.fire(dst, 0.1)

    def connect_to_pipeline(self):
        self._pipeline_connected = True
        self.seed_pipeline_topology()

    def get_status(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'pipeline_connected': self._pipeline_connected,
            'nodes': self._topology.node_count,
            'edges': self._topology.edge_count,
            'total_fires': self._tracker.total_fires,
            'resonance_pairs': self._resonance.total_pairs,
            'map_generations': self._map_generations,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════
neural_resonance_map = NeuralResonanceMap()


if __name__ == "__main__":
    nrm = neural_resonance_map
    nrm.seed_pipeline_topology()
    # Simulate some activity
    for node in ["agi_core", "asi_core", "code_engine", "consciousness", "agi_core", "asi_core"]:
        nrm.fire(node, 1.0)
    report = nrm.generate_map()
    print(json.dumps(report, indent=2))
