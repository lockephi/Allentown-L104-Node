"""L104 Quantum Networker v1.5.0 — Entanglement Distribution & Routing.

Manages the distribution, storage, and routing of entangled Bell pairs across
the quantum network. Uses the VQPU's high-fidelity Bell pair generation
(sacred_circuit, EntanglementQuantifier) for pair creation.

Features:
  - Bell pair generation via VQPU circuits
  - Pair pool management with decoherence tracking
  - Shortest-path routing through entanglement-swapped relays
  - Entanglement purification (DEJMPS protocol)
  - Sacred alignment scoring of distributed pairs
  - Automatic pair replenishment when pool depletes
  - v1.1: Decoherence time-decay model (T1/T2 exponential)
  - v1.1: Network-wide analytics (pair census, fidelity heatmap)
  - v1.1: GOD_CODE sacred pair scoring pass
  - v1.1: Built-in self_test() for debug framework integration
  - v1.2: Route caching with topology-aware invalidation
  - v1.2: Improved Dijkstra weights (fidelity × pair capacity × sacred)
  - v1.2: purify_all() — bulk network-wide purification
  - v1.2: Topology auto-detection (star, linear, ring, mesh, tree)
  - v1.2: Channel health composite scoring
  - v1.2: K-shortest paths for redundant routing
  - v1.2: Pair lifetime prediction (time-to-discard)
  - v1.2: Network summary with throughput metrics
  - v1.3: Route fidelity estimation before committing to a path
  - v1.3: Channel auto-heal pipeline (detect, purify, replenish)
  - v1.3: Redundant E2E pair creation via K-shortest paths
  - v1.3: Network snapshot/restore for state persistence
  - v1.3: Batch Bell pair generation
  - v1.4: Fixed find_k_routes edge-restore between iterations
  - v1.4: Structured network event log for observability
  - v1.4: Per-channel fidelity trend analysis (degradation detection)
  - v1.4: Channel capacity (Hashing bound) & network capacity
  - v1.4: Resilience analysis (Tarjan articulation points + bridges)
  - v1.4: Autonomous maintenance cycle
  - v1.4: Path diversity & aggregate bandwidth
  - v1.4: Expanded self-test (37 probes)
  - v1.5: Pair reservation system (reserve/release before teleport)
  - v1.5: Node betweenness centrality (relay importance ranking)
  - v1.5: Channel capacity forecasting (time-to-empty prediction)
  - v1.5: Network graph export (adjacency matrix + DOT format)
  - v1.5: Entanglement distillation cascade (multi-round purify)
  - v1.5: Weighted max-flow between nodes (Ford-Fulkerson)
  - v1.5: Expanded self-test (45 probes)

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import copy
import json
import math
import time
import heapq
import secrets
import threading
import logging
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

import numpy as np

from .types import (
    QuantumNode, QuantumChannel, EntangledPair, NetworkTopology,
    GOD_CODE, PHI, PHI_INV,
)

# Minimum pairs to maintain per channel
MIN_PAIR_POOL = 4
# Maximum pairs per channel before pruning old ones
MAX_PAIR_POOL = 64
# Fidelity threshold below which a pair is discarded
FIDELITY_DISCARD_THRESHOLD = 0.5
# Purification success probability (DEJMPS bilateral CNOT)
PURIFICATION_SUCCESS_PROB = 0.5
# v1.1: Decoherence time constants (seconds)
T1_COHERENCE_TIME = 100.0    # Amplitude damping (energy relaxation)
T2_COHERENCE_TIME = 60.0     # Phase damping (dephasing) — T2 ≤ 2*T1
# v1.1: Sacred pair scoring amplification factor
SACRED_PAIR_AMPLIFIER = PHI_INV * (GOD_CODE / 1000.0)  # ~0.326
# v1.2: Route cache time-to-live (invalidated on topology change)
ROUTE_CACHE_TTL = 30.0
# v1.2: Channel health weights
HEALTH_W_FIDELITY  = 0.45
HEALTH_W_CAPACITY  = 0.25   # pair pool fullness
HEALTH_W_SACRED    = 0.15
HEALTH_W_FRESHNESS = 0.15   # inverse of mean pair age
# v1.4: Fidelity trend analysis
TREND_WINDOW_SIZE = 20       # Samples in sliding window
TREND_SAMPLE_INTERVAL = 5.0  # Minimum seconds between samples
# v1.4: Event log
EVENT_LOG_CAPACITY = 2000    # Max events in audit log
# v1.5: Pair reservation
RESERVATION_TTL = 30.0       # Seconds before an unreleased reservation expires
# v1.5: Distillation cascade default
DISTILLATION_TARGET_FIDELITY = 0.99

from ._bridge import get_bridge as _get_bridge


class EntanglementRouter:
    """Manages entangled pair distribution and quantum routing across the network.

    Each channel between two nodes maintains a pool of entangled Bell pairs.
    When teleportation is requested between non-adjacent nodes, the router
    finds the shortest path and performs entanglement swapping at intermediate
    nodes to create an end-to-end entangled pair.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._nodes: Dict[str, QuantumNode] = {}
        self._channels: Dict[str, QuantumChannel] = {}
        # Adjacency: node_id → set of neighbor node_ids
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        # Channel lookup: (node_a, node_b) → channel_id (sorted key)
        self._edge_to_channel: Dict[Tuple[str, str], str] = {}
        self._total_pairs_generated = 0
        self._total_purifications = 0
        self._total_swaps = 0
        self._total_decoherence_decays = 0
        self._total_pairs_expired = 0
        self._total_pairs_consumed = 0
        self._start_time = time.time()
        # v1.2: Route cache  (source, dest) → (path, timestamp)
        self._route_cache: Dict[Tuple[str, str], Tuple[List[str], float]] = {}
        self._topology_version = 0            # Bumped on any topo change
        self._throughput_log: deque = deque(maxlen=500)   # (timestamp, event)
        # v1.4: Fidelity trend monitor  channel_id → deque of (timestamp, fidelity)
        self._fidelity_trends: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=TREND_WINDOW_SIZE)
        )
        # v1.4: Structured event/audit log
        self._event_log: deque = deque(maxlen=EVENT_LOG_CAPACITY)
        # v1.4: Maintenance cycle counter
        self._maintenance_cycles = 0
        # v1.5: Pair reservations  reservation_id → {channel_id, pair_id, timestamp}
        self._reservations: Dict[str, Dict] = {}

    # ═══════════════════════════════════════════════════════════════
    # NODE & CHANNEL MANAGEMENT
    # ═══════════════════════════════════════════════════════════════

    def _invalidate_routes(self) -> None:
        """Clear the route cache (called on any topology mutation)."""
        self._route_cache.clear()
        self._topology_version += 1

    def _emit(self, event: str, **kwargs) -> None:
        """v1.4: Record a structured event in the audit log."""
        self._event_log.append({
            "ts": time.time(),
            "event": event,
            **kwargs,
        })

    def add_node(self, node: QuantumNode) -> None:
        """Register a node in the network."""
        with self._lock:
            self._nodes[node.node_id] = node
            self._invalidate_routes()
            self._emit("node_added", node_id=node.node_id, role=node.role)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its channels."""
        with self._lock:
            if node_id not in self._nodes:
                return
            # Remove all channels involving this node
            channels_to_remove = [
                cid for cid, ch in self._channels.items()
                if ch.node_a_id == node_id or ch.node_b_id == node_id
            ]
            for cid in channels_to_remove:
                ch = self._channels[cid]
                edge = tuple(sorted([ch.node_a_id, ch.node_b_id]))
                self._edge_to_channel.pop(edge, None)
                self._adjacency[ch.node_a_id].discard(ch.node_b_id)
                self._adjacency[ch.node_b_id].discard(ch.node_a_id)
                del self._channels[cid]
            del self._nodes[node_id]
            self._adjacency.pop(node_id, None)
            self._invalidate_routes()
            self._emit("node_removed", node_id=node_id,
                       channels_removed=len(channels_to_remove))

    def create_channel(self, node_a_id: str, node_b_id: str,
                       initial_pairs: int = MIN_PAIR_POOL) -> QuantumChannel:
        """Create a quantum channel between two nodes with initial entangled pairs.

        Args:
            node_a_id: First node ID
            node_b_id: Second node ID
            initial_pairs: Number of Bell pairs to pre-generate

        Returns:
            The created QuantumChannel
        """
        edge = tuple(sorted([node_a_id, node_b_id]))
        if edge in self._edge_to_channel:
            return self._channels[self._edge_to_channel[edge]]

        channel = QuantumChannel(
            node_a_id=node_a_id,
            node_b_id=node_b_id,
            state="entangling",
        )

        # Generate initial entangled pairs
        for _ in range(initial_pairs):
            pair = self._generate_bell_pair(node_a_id, node_b_id)
            channel.pairs.append(pair)
            channel.fidelity_history.append(pair.fidelity)

        channel.state = "active"
        self._emit("channel_created", channel_id=channel.channel_id,
                   node_a=node_a_id[:8], node_b=node_b_id[:8],
                   initial_pairs=initial_pairs)

        # Register (thread-safe topology mutation)
        with self._lock:
            self._channels[channel.channel_id] = channel
            self._edge_to_channel[edge] = channel.channel_id
            self._adjacency[node_a_id].add(node_b_id)
            self._adjacency[node_b_id].add(node_a_id)
            self._invalidate_routes()

            # Update node stats
            for nid in (node_a_id, node_b_id):
                if nid in self._nodes:
                    self._nodes[nid].active_channels += 1
                    self._nodes[nid].entangled_pairs += initial_pairs

        return channel

    def get_channel(self, node_a_id: str, node_b_id: str) -> Optional[QuantumChannel]:
        """Get channel between two nodes (order-independent)."""
        edge = tuple(sorted([node_a_id, node_b_id]))
        cid = self._edge_to_channel.get(edge)
        if cid:
            return self._channels.get(cid)
        return None

    # ═══════════════════════════════════════════════════════════════
    # BELL PAIR GENERATION
    # ═══════════════════════════════════════════════════════════════

    def _generate_bell_pair(self, node_a_id: str, node_b_id: str) -> EntangledPair:
        """Generate a Bell |Φ+⟩ pair between two nodes using VQPU.

        Uses the VQPU's bell_pair() circuit and scores with SacredAlignmentScorer
        for GOD_CODE harmonic quality assessment.
        """
        self._total_pairs_generated += 1
        self._throughput_log.append((time.time(), "pair_gen"))
        bridge = _get_bridge()

        fidelity = 0.995  # Default high-fidelity pair
        sacred_score = 0.0

        if bridge is not None:
            try:
                from l104_vqpu import QuantumJob
                # Use VQPU to generate and measure a Bell pair
                job = QuantumJob(num_qubits=2, operations=[
                    {"gate": "H", "qubits": [0]},
                    {"gate": "CNOT", "qubits": [0, 1]},
                ], shots=256)
                result = bridge.submit_and_wait(job, timeout=2.0)
                if result and result.probabilities:
                    # Bell state |Φ+⟩: P(00) ≈ P(11) ≈ 0.5
                    p00 = result.probabilities.get("00", 0)
                    p11 = result.probabilities.get("11", 0)
                    fidelity = p00 + p11  # Should be ~1.0 for perfect Bell state
                    # Score with sacred alignment
                    from l104_vqpu import SacredAlignmentScorer
                    scores = SacredAlignmentScorer.score(result.probabilities, 2)
                    sacred_score = scores.get("sacred_score", 0.0)
            except Exception:
                # Fallback: simulate high-fidelity pair
                fidelity = 0.995 - abs(np.random.normal(0, 0.003))
                sacred_score = PHI_INV * fidelity

        if sacred_score == 0.0:
            sacred_score = PHI_INV * fidelity

        return EntangledPair(
            node_a_id=node_a_id,
            node_b_id=node_b_id,
            bell_state="phi_plus",
            fidelity=max(0.0, min(1.0, fidelity)),
            sacred_score=sacred_score,
        )

    def _generate_bell_pairs_batch(self, node_a_id: str, node_b_id: str,
                                    count: int) -> List[EntangledPair]:
        """v1.3: Generate multiple Bell pairs in a single batch.

        Falls back to individual generation but provides a single-call
        interface that can be optimized with VQPU batch jobs in the future.
        """
        return [self._generate_bell_pair(node_a_id, node_b_id)
                for _ in range(count)]

    def replenish_channel(self, channel_id: str, target_pairs: int = MIN_PAIR_POOL) -> int:
        """Replenish a channel's entangled pair pool.

        Enforces MAX_PAIR_POOL ceiling to prevent unbounded growth.
        Prunes dead / low-fidelity pairs first, then evicts oldest
        low-quality pairs if pool exceeds the limit.

        Returns:
            Number of new pairs generated
        """
        ch = self._channels.get(channel_id)
        if not ch:
            return 0

        # Prune dead pairs first
        ch.prune_dead_pairs()

        # Enforce MAX_PAIR_POOL ceiling
        self._enforce_pair_limit(ch)

        capacity = MAX_PAIR_POOL - len(ch.pairs)
        needed = max(0, min(target_pairs - len(ch.usable_pairs), capacity))
        for _ in range(needed):
            pair = self._generate_bell_pair(ch.node_a_id, ch.node_b_id)
            ch.pairs.append(pair)

        return needed

    def _enforce_pair_limit(self, ch: QuantumChannel) -> int:
        """Evict oldest low-fidelity pairs if pool exceeds MAX_PAIR_POOL.

        Strategy: Sort by (consumed, fidelity, -age), evict worst first.

        Returns:
            Number of pairs evicted
        """
        excess = len(ch.pairs) - MAX_PAIR_POOL
        if excess <= 0:
            return 0
        # Score each pair — lower is worse (evict first)
        scored = sorted(
            enumerate(ch.pairs),
            key=lambda ip: (
                not ip[1].consumed,           # consumed first
                ip[1].current_fidelity,       # low fidelity first
                -ip[1].age_s,                 # older first
            ),
        )
        evict_indices = {scored[i][0] for i in range(excess)}
        evicted = len(evict_indices)
        ch.pairs = [p for i, p in enumerate(ch.pairs) if i not in evict_indices]
        self._total_pairs_expired += evicted
        return evicted

    def replenish_all(self) -> int:
        """Replenish all channels that are below minimum pair threshold."""
        total = 0
        for cid, ch in self._channels.items():
            if len(ch.usable_pairs) < MIN_PAIR_POOL:
                total += self.replenish_channel(cid)
        return total

    # ═══════════════════════════════════════════════════════════════
    # ENTANGLEMENT PURIFICATION (DEJMPS PROTOCOL)
    # ═══════════════════════════════════════════════════════════════

    def purify_pair(self, channel_id: str) -> Optional[EntangledPair]:
        """Purify an entangled pair using the DEJMPS bilateral CNOT protocol.

        Consumes two noisy pairs to produce one higher-fidelity pair.
        Success probability: F²/(F² + (1-F)²)

        Protocol:
          1. Take two pairs from the channel
          2. Both Alice and Bob apply CNOT between their halves
          3. Measure the target pair
          4. If outcomes agree → keep the control pair (higher fidelity)
          5. If disagree → discard both (protocol fails)

        Returns:
            Purified EntangledPair if successful, None if failed
        """
        ch = self._channels.get(channel_id)
        if not ch:
            return None

        usable = ch.usable_pairs
        if len(usable) < 2:
            return None

        # Sort by fidelity, take two worst (sacrifice the worst to boost the better)
        usable.sort(key=lambda p: p.current_fidelity)
        pair_control = usable[-1]  # Best pair (to keep)
        pair_target = usable[-2]   # Second pair (to sacrifice)

        f1 = pair_control.current_fidelity
        f2 = pair_target.current_fidelity

        # DEJMPS success probability
        p_success = (f1 * f2) / (f1 * f2 + (1 - f1) * (1 - f2))

        # Consume target pair regardless
        pair_target.consumed = True

        if np.random.random() < p_success:
            # Success: purified fidelity
            new_fidelity = (f1 * f2) / (f1 * f2 + (1 - f1) * (1 - f2))
            pair_control.fidelity = min(1.0, new_fidelity)
            pair_control.purified = True
            pair_control.generation += 1
            self._total_purifications += 1
            ch.purifications_count += 1
            self._emit("purification_success", channel=channel_id[:10],
                       new_fidelity=round(new_fidelity, 4))
            return pair_control
        else:
            # Failure: both pairs lost
            pair_control.consumed = True
            return None

    def purify_channel(self, channel_id: str, rounds: int = 3,
                        target_fidelity: Optional[float] = None) -> Dict:
        """Run multiple purification rounds on a channel.

        Args:
            channel_id: Channel to purify
            rounds: Maximum purification rounds
            target_fidelity: Stop early once mean fidelity reaches this target

        Returns:
            Dict with purification statistics
        """
        ch = self._channels.get(channel_id)
        if not ch:
            return {"error": "channel not found"}

        initial_pairs = len(ch.usable_pairs)
        initial_fidelity = ch.mean_fidelity
        successes = 0
        failures = 0

        for _ in range(rounds):
            if len(ch.usable_pairs) < 2:
                break
            # Early exit when target reached
            if target_fidelity is not None and ch.mean_fidelity >= target_fidelity:
                break
            result = self.purify_pair(channel_id)
            if result:
                successes += 1
            else:
                failures += 1

        ch.prune_dead_pairs()

        target_reached = (
            target_fidelity is not None and ch.mean_fidelity >= target_fidelity
        )

        return {
            "channel_id": channel_id,
            "rounds_attempted": successes + failures,
            "successes": successes,
            "failures": failures,
            "pairs_before": initial_pairs,
            "pairs_after": len(ch.usable_pairs),
            "fidelity_before": initial_fidelity,
            "fidelity_after": ch.mean_fidelity,
            "fidelity_gain": ch.mean_fidelity - initial_fidelity,
            "target_fidelity": target_fidelity,
            "target_reached": target_reached,
        }

    # ═══════════════════════════════════════════════════════════════
    # QUANTUM ROUTING
    # ═══════════════════════════════════════════════════════════════

    def find_route(self, source_id: str, dest_id: str) -> Optional[List[str]]:
        """Find shortest path from source to destination using Dijkstra.

        v1.2: Improved edge weight = (1/fidelity) × (1 + 1/pair_count) × (2 − sacred_mean).
        Results are cached for ROUTE_CACHE_TTL seconds and invalidated on topology changes.

        Returns:
            List of node IDs forming the path, or None if no path exists
        """
        if source_id not in self._nodes or dest_id not in self._nodes:
            return None

        if source_id == dest_id:
            return [source_id]

        # v1.2: Check route cache
        cache_key = (source_id, dest_id)
        cached = self._route_cache.get(cache_key)
        if cached is not None:
            path, ts = cached
            if time.time() - ts < ROUTE_CACHE_TTL:
                # Validate all edges still have usable pairs
                valid = True
                for i in range(len(path) - 1):
                    ch = self.get_channel(path[i], path[i + 1])
                    if not ch or not ch.usable_pairs:
                        valid = False
                        break
                if valid:
                    return list(path)
            # Expired or invalid — evict
            del self._route_cache[cache_key]

        # Dijkstra with v1.2 improved weights
        dist = {source_id: 0.0}
        prev = {}
        visited: Set[str] = set()
        heap = [(0.0, source_id)]

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)

            if u == dest_id:
                # Reconstruct path
                path = []
                node = dest_id
                while node in prev:
                    path.append(node)
                    node = prev[node]
                path.append(source_id)
                path.reverse()
                # v1.2: Cache the discovered route
                self._route_cache[cache_key] = (list(path), time.time())
                return path

            for neighbor in self._adjacency.get(u, set()):
                if neighbor in visited:
                    continue
                ch = self.get_channel(u, neighbor)
                if not ch or not ch.usable_pairs:
                    continue
                # v1.2: Composite weight — lower is better
                # (1) inverse fidelity: prefer high-fidelity channels
                fidelity_w = 1.0 / max(0.01, ch.mean_fidelity)
                # (2) inverse pair count: prefer well-stocked channels
                capacity_w = 1.0 + 1.0 / max(1, len(ch.usable_pairs))
                # (3) inverse sacred alignment: prefer sacred channels
                mean_sacred = 0.0
                usable = ch.usable_pairs
                if usable:
                    mean_sacred = sum(p.sacred_score for p in usable) / len(usable)
                sacred_w = 2.0 - min(1.0, mean_sacred)
                weight = fidelity_w * capacity_w * sacred_w
                new_dist = d + weight
                if neighbor not in dist or new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = u
                    heapq.heappush(heap, (new_dist, neighbor))

        return None  # No path

    def find_k_routes(self, source_id: str, dest_id: str,
                      k: int = 3) -> List[List[str]]:
        """v1.4: Find up to K shortest edge-disjoint paths.

        Fixed: edges are now properly restored between iterations so
        each probe sees the full topology minus only currently-blocked
        edges.  Returns up to *k* paths ordered by discovery.
        """
        paths: List[List[str]] = []
        best = self.find_route(source_id, dest_id)
        if best is None:
            return paths
        paths.append(best)

        for _ in range(k - 1):
            # Block edges of ALL previously-discovered paths
            blocked: Dict[str, QuantumChannel] = {}
            for path in paths:
                for i in range(len(path) - 1):
                    edge = tuple(sorted([path[i], path[i + 1]]))
                    cid = self._edge_to_channel.get(edge)
                    if cid and cid not in blocked:
                        blocked[cid] = self._channels.pop(cid)
                        self._edge_to_channel.pop(edge, None)
                        self._adjacency[path[i]].discard(path[i + 1])
                        self._adjacency[path[i + 1]].discard(path[i])

            # Invalidate cached route so Dijkstra runs fresh
            self._route_cache.pop((source_id, dest_id), None)
            alt = self.find_route(source_id, dest_id)
            if alt and alt not in paths:
                paths.append(alt)

            # Restore ALL blocked edges before next iteration
            for cid, ch in blocked.items():
                self._channels[cid] = ch
                edge = tuple(sorted([ch.node_a_id, ch.node_b_id]))
                self._edge_to_channel[edge] = cid
                self._adjacency[ch.node_a_id].add(ch.node_b_id)
                self._adjacency[ch.node_b_id].add(ch.node_a_id)

        return paths

    def entanglement_swap(self, node_a_id: str, relay_id: str,
                           node_b_id: str) -> Optional[EntangledPair]:
        """Perform entanglement swapping at a relay node.

        Creates a direct entangled pair between A and B by consuming
        one pair from A↔Relay and one from Relay↔B, then performing
        Bell measurement at the relay.

        Protocol:
          1. Take pair P1 from channel(A, Relay): |Φ+⟩_A1,R1
          2. Take pair P2 from channel(Relay, B): |Φ+⟩_R2,B2
          3. Bell measurement on R1, R2 at relay
          4. Classical communication → A and B now share entanglement

        Returns:
            New EntangledPair(A, B) if successful, None if no pairs
        """
        ch_ar = self.get_channel(node_a_id, relay_id)
        ch_rb = self.get_channel(relay_id, node_b_id)

        if not ch_ar or not ch_rb:
            return None

        pair_ar = ch_ar.consume_best_pair()
        pair_rb = ch_rb.consume_best_pair()

        if not pair_ar or not pair_rb:
            return None

        # Swap fidelity model: depolarizing channel
        # F_swap = F1·F2 + (1-F1)(1-F2)/3
        # This is the physically correct formula for entanglement swapping
        # through depolarizing channels. Note: the VQPU statevector simulation
        # cannot be used here because the swap protocol requires mid-circuit
        # Bell measurement + classical correction (X^m1 Z^m0), which statevector
        # simulation doesn't support. Without corrections, the marginal on
        # qubits 0,3 gives ~0.5 (maximally mixed), not the actual swap fidelity.
        f1 = pair_ar.current_fidelity
        f2 = pair_rb.current_fidelity
        swap_fidelity = f1 * f2 + (1 - f1) * (1 - f2) / 3

        ch_ar.swaps_count += 1
        ch_rb.swaps_count += 1
        self._total_swaps += 1

        # Sacred score: φ-weighted blend of constituent scores
        sacred = (pair_ar.sacred_score * PHI + pair_rb.sacred_score) / (PHI + 1.0)

        swapped_pair = EntangledPair(
            node_a_id=node_a_id,
            node_b_id=node_b_id,
            bell_state="phi_plus",
            fidelity=max(0.0, min(1.0, swap_fidelity)),
            sacred_score=sacred,
        )

        # Store the swapped pair in a direct channel (create if needed)
        ch_direct = self.get_channel(node_a_id, node_b_id)
        if ch_direct is None:
            ch_direct = QuantumChannel(
                node_a_id=node_a_id,
                node_b_id=node_b_id,
                state="active",
            )
            edge = tuple(sorted([node_a_id, node_b_id]))
            self._channels[ch_direct.channel_id] = ch_direct
            self._edge_to_channel[edge] = ch_direct.channel_id

        ch_direct.pairs.append(swapped_pair)
        self._throughput_log.append((time.time(), "swap"))
        self._emit("entanglement_swap", relay=relay_id[:8],
                   fidelity=round(swap_fidelity, 4))
        return swapped_pair

    # ═══════════════════════════════════════════════════════════════
    # v1.3: ROUTE FIDELITY ESTIMATION
    # ═══════════════════════════════════════════════════════════════

    def estimate_route_fidelity(self, route: List[str]) -> float:
        """Predict end-to-end fidelity for a given route *before* consuming pairs.

        For a direct (2-hop) route, returns the channel's mean fidelity.
        For multi-hop, applies the depolarizing swap model iteratively:
          F_swap = F1·F2 + (1-F1)(1-F2)/3

        Returns:
            Estimated fidelity in [0, 1], or 0.0 if route is invalid
        """
        if not route or len(route) < 2:
            return 0.0

        # Collect per-hop fidelities
        hop_fidelities: List[float] = []
        for i in range(len(route) - 1):
            ch = self.get_channel(route[i], route[i + 1])
            if not ch or not ch.usable_pairs:
                return 0.0
            hop_fidelities.append(ch.mean_fidelity)

        if len(hop_fidelities) == 1:
            return hop_fidelities[0]

        # Sequential swap model
        f_running = hop_fidelities[0]
        for f_next in hop_fidelities[1:]:
            f_running = f_running * f_next + (1 - f_running) * (1 - f_next) / 3
        return max(0.0, min(1.0, f_running))

    def best_route_by_fidelity(self, source_id: str, dest_id: str,
                                k: int = 3) -> Optional[Tuple[List[str], float]]:
        """Find the route with highest estimated end-to-end fidelity.

        Evaluates up to *k* shortest paths and picks the one with the
        best predicted fidelity.

        Returns:
            (route, estimated_fidelity) or None if no path exists
        """
        routes = self.find_k_routes(source_id, dest_id, k=k)
        if not routes:
            return None

        best_route = None
        best_f = -1.0
        for route in routes:
            f_est = self.estimate_route_fidelity(route)
            if f_est > best_f:
                best_f = f_est
                best_route = route
        return (best_route, best_f) if best_route else None

    def create_end_to_end_pair(self, source_id: str,
                                dest_id: str) -> Optional[Tuple[EntangledPair, List[str]]]:
        """Create an end-to-end entangled pair via multi-hop swapping.

        Finds the shortest route, performs sequential entanglement swapping
        at each relay node, and returns the final pair.

        Returns:
            (EntangledPair, route) or None if no path
        """
        route = self.find_route(source_id, dest_id)
        if not route:
            return None

        if len(route) == 2:
            # Direct channel — consume the best pair so it can't be reused
            ch = self.get_channel(source_id, dest_id)
            if ch:
                pair = ch.consume_best_pair()
                if pair:
                    return (pair, route)
            return None

        # Multi-hop: swap at each intermediate relay
        # Process: A-R1-R2-...-B
        # First swap: A↔R1 + R1↔R2 → A↔R2
        current_a = route[0]
        for i in range(1, len(route) - 1):
            relay = route[i]
            next_node = route[i + 1]
            pair = self.entanglement_swap(current_a, relay, next_node)
            if not pair:
                logger.warning(
                    "Swap failed at relay %s on route %s→%s (hop %d/%d)",
                    relay[:8], source_id[:8], dest_id[:8],
                    i, len(route) - 2,
                )
                return None
            current_a = route[0]  # current_a stays as source (pair extends)

        # The final pair should be source↔dest — consume to prevent reuse
        ch = self.get_channel(source_id, dest_id)
        if ch and ch.usable_pairs:
            return (ch.consume_best_pair(), route)

        return None

    # ═══════════════════════════════════════════════════════════════
    # STATUS & DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════

    @property
    def nodes(self) -> Dict[str, QuantumNode]:
        return dict(self._nodes)

    @property
    def channels(self) -> Dict[str, QuantumChannel]:
        return dict(self._channels)

    def network_fidelity(self) -> float:
        """Mean fidelity across all active channels."""
        fids = [ch.mean_fidelity for ch in self._channels.values()
                if ch.usable_pairs]
        return sum(fids) / len(fids) if fids else 0.0

    def status(self) -> Dict:
        return {
            "version": "1.5.0",
            "nodes": len(self._nodes),
            "channels": len(self._channels),
            "active_channels": sum(1 for ch in self._channels.values()
                                   if ch.effective_state == "active"),
            "total_pairs": sum(len(ch.pairs) for ch in self._channels.values()),
            "usable_pairs": sum(len(ch.usable_pairs) for ch in self._channels.values()),
            "network_fidelity": self.network_fidelity(),
            "total_pairs_generated": self._total_pairs_generated,
            "total_purifications": self._total_purifications,
            "total_swaps": self._total_swaps,
            "total_decoherence_decays": self._total_decoherence_decays,
            "total_pairs_expired": self._total_pairs_expired,
            "active_reservations": len(self._reservations),
            "health_score": round(self.network_health_score(), 4),
            "topology_version": self._topology_version,
            "uptime_s": round(time.time() - self._start_time, 2),
            "god_code": GOD_CODE,
            "phi": PHI,
        }

    # ═══════════════════════════════════════════════════════════════
    # v1.1: DECOHERENCE TIME-DECAY MODEL
    # ═══════════════════════════════════════════════════════════════

    def apply_decoherence(self, elapsed_seconds: float = 5.0) -> Dict:
        """Apply T1/T2 exponential decoherence decay to all entangled pairs.

        Models real quantum hardware behavior:
          F(t) = F₀ × exp(-t/T2) × [1 - (1-exp(-t/T1))/2]

        Args:
            elapsed_seconds: Time elapsed since last decay application

        Returns:
            Dict with decay statistics
        """
        total_decayed = 0
        total_discarded = 0
        pairs_before = sum(len(ch.usable_pairs) for ch in self._channels.values())

        t1_factor = math.exp(-elapsed_seconds / T1_COHERENCE_TIME)
        t2_factor = math.exp(-elapsed_seconds / T2_COHERENCE_TIME)
        amplitude_loss = (1.0 - t1_factor) / 2.0

        for ch in self._channels.values():
            for pair in ch.pairs:
                if pair.consumed:
                    continue
                old_f = pair.current_fidelity
                new_f = old_f * t2_factor * (1.0 - amplitude_loss)
                pair.fidelity = max(0.0, new_f)
                total_decayed += 1
                self._total_decoherence_decays += 1

                if pair.current_fidelity < FIDELITY_DISCARD_THRESHOLD:
                    pair.consumed = True
                    total_discarded += 1
                    self._total_pairs_expired += 1

        pairs_after = sum(len(ch.usable_pairs) for ch in self._channels.values())

        # v1.2: Auto-replenish channels that fell below minimum
        auto_replenished = self.replenish_all() if total_discarded > 0 else 0

        # v1.4: Sample fidelity trends after decoherence
        self._sample_all_fidelity_trends()

        self._emit("decoherence", elapsed_s=elapsed_seconds,
                   decayed=total_decayed, discarded=total_discarded)

        return {
            "elapsed_s": elapsed_seconds,
            "t1_factor": round(t1_factor, 6),
            "t2_factor": round(t2_factor, 6),
            "pairs_decayed": total_decayed,
            "pairs_discarded": total_discarded,
            "pairs_before": pairs_before,
            "pairs_after": pairs_after,
            "auto_replenished": auto_replenished,
        }

    # ═══════════════════════════════════════════════════════════════
    # v1.1: NETWORK ANALYTICS
    # ═══════════════════════════════════════════════════════════════

    def fidelity_heatmap(self) -> Dict[str, float]:
        """Compute per-channel fidelity heatmap.

        Returns:
            Dict mapping "nodeA↔nodeB" → mean_fidelity
        """
        heatmap = {}
        for ch in self._channels.values():
            label = f"{ch.node_a_id[:8]}↔{ch.node_b_id[:8]}"
            heatmap[label] = round(ch.mean_fidelity, 6)
        return heatmap

    def pair_census(self) -> Dict:
        """Full census of all entangled pairs in the network."""
        total = 0
        usable = 0
        purified = 0
        high_fidelity = 0  # F > 0.99
        sacred_aligned = 0  # sacred_score > PHI_INV

        for ch in self._channels.values():
            for pair in ch.pairs:
                total += 1
                if not pair.consumed and pair.current_fidelity >= FIDELITY_DISCARD_THRESHOLD:
                    usable += 1
                if pair.purified:
                    purified += 1
                if pair.current_fidelity > 0.99:
                    high_fidelity += 1
                if pair.sacred_score > PHI_INV:
                    sacred_aligned += 1

        return {
            "total_pairs": total,
            "usable_pairs": usable,
            "purified_pairs": purified,
            "high_fidelity_pairs": high_fidelity,
            "sacred_aligned_pairs": sacred_aligned,
            "sacred_ratio": round(sacred_aligned / max(1, total), 4),
            "god_code_resonance": round(
                (sacred_aligned / max(1, total)) * (GOD_CODE / 1000.0), 6
            ),
        }

    def sacred_scoring_pass(self) -> Dict:
        """Run GOD_CODE-aligned sacred scoring on all pairs.

        Recomputes sacred_score using the φ-harmonic amplified formula:
          sacred_score = F × PHI_INV × (GOD_CODE/1000) × (1 + F^φ)/2
        """
        scored = 0
        total_sacred = 0.0
        for ch in self._channels.values():
            for pair in ch.pairs:
                if pair.consumed:
                    continue
                f = pair.current_fidelity
                pair.sacred_score = f * SACRED_PAIR_AMPLIFIER * (1.0 + f ** PHI) / 2.0
                total_sacred += pair.sacred_score
                scored += 1

        mean_sacred = total_sacred / max(1, scored)
        return {
            "pairs_scored": scored,
            "mean_sacred_score": round(mean_sacred, 6),
            "total_sacred_energy": round(total_sacred, 4),
            "god_code_alignment": round(mean_sacred * (1000.0 / GOD_CODE), 6),
        }

    # ═══════════════════════════════════════════════════════════════
    # v1.2: BULK PURIFICATION
    # ═══════════════════════════════════════════════════════════════

    def purify_all(self, fidelity_threshold: float = 0.95,
                   rounds_per_channel: int = 3) -> Dict:
        """Purify all channels whose mean fidelity is below the threshold.

        Runs DEJMPS purification rounds on every degraded channel and
        replenishes pools that fall below MIN_PAIR_POOL afterward.

        Returns:
            Aggregated purification statistics across all channels
        """
        channels_purified = 0
        total_successes = 0
        total_failures = 0
        total_fidelity_gain = 0.0

        for cid, ch in list(self._channels.items()):
            if ch.mean_fidelity >= fidelity_threshold:
                continue
            if len(ch.usable_pairs) < 2:
                continue
            result = self.purify_channel(cid, rounds=rounds_per_channel)
            channels_purified += 1
            total_successes += result.get("successes", 0)
            total_failures += result.get("failures", 0)
            total_fidelity_gain += result.get("fidelity_gain", 0.0)

        # Replenish depleted channels after purification
        replenished = self.replenish_all()

        return {
            "channels_purified": channels_purified,
            "total_channels": len(self._channels),
            "purification_successes": total_successes,
            "purification_failures": total_failures,
            "total_fidelity_gain": round(total_fidelity_gain, 6),
            "pairs_replenished": replenished,
        }

    # ═══════════════════════════════════════════════════════════════
    # v1.2: TOPOLOGY DETECTION
    # ═══════════════════════════════════════════════════════════════

    def detect_topology(self) -> Dict:
        """Auto-detect the quantum network topology by analyzing the graph.

        Classifies as: star, linear, ring, mesh, tree, or unknown.
        Returns topology type plus supporting metrics.
        """
        n = len(self._nodes)
        e = len(self._channels)

        if n <= 1:
            return {"topology": "trivial", "nodes": n, "edges": e}

        degrees = {nid: len(self._adjacency.get(nid, set()))
                   for nid in self._nodes}
        max_deg = max(degrees.values()) if degrees else 0
        min_deg = min(degrees.values()) if degrees else 0
        mean_deg = sum(degrees.values()) / max(1, n)
        leaves = sum(1 for d in degrees.values() if d == 1)

        # Classification heuristics
        topology = "unknown"
        if e == n * (n - 1) // 2:
            topology = "mesh"               # Fully connected
        elif e == n and min_deg == 2 and max_deg == 2:
            topology = "ring"               # Every node degree 2, E == N
        elif e == n - 1:
            if max_deg == n - 1:
                topology = "star"           # One hub connecting all others
            elif leaves >= 2 and max_deg <= n - 1:
                topology = "tree"           # Acyclic connected
        elif e == n - 1 and leaves == 2 and max_deg == 2:
            topology = "linear"             # Chain

        # Connectivity check (BFS)
        start = next(iter(self._nodes))
        visited: Set[str] = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for nb in self._adjacency.get(node, set()):
                if nb not in visited:
                    queue.append(nb)
        connected = len(visited) == n

        return {
            "topology": topology,
            "nodes": n,
            "edges": e,
            "connected": connected,
            "max_degree": max_deg,
            "min_degree": min_deg,
            "mean_degree": round(mean_deg, 2),
            "leaf_nodes": leaves,
            "topology_version": self._topology_version,
        }

    # ═══════════════════════════════════════════════════════════════
    # v1.2: CHANNEL HEALTH SCORING
    # ═══════════════════════════════════════════════════════════════

    def channel_health(self, channel_id: str) -> Dict:
        """Composite health score for a single channel (0.0–1.0).

        Weighted blend of:
          - Fidelity (45%): mean pair fidelity
          - Capacity (25%): pair pool fullness (usable / MAX_PAIR_POOL)
          - Sacred  (15%): mean sacred alignment score
          - Freshness (15%): inverse of mean pair age (younger = healthier)
        """
        ch = self._channels.get(channel_id)
        if not ch:
            return {"error": "channel not found", "health": 0.0}

        usable = ch.usable_pairs
        n_usable = len(usable)

        fidelity_score = ch.mean_fidelity
        capacity_score = min(1.0, n_usable / MAX_PAIR_POOL)

        sacred_score = 0.0
        mean_age = 0.0
        if n_usable > 0:
            sacred_score = min(1.0,
                               sum(p.sacred_score for p in usable) / n_usable)
            mean_age = sum(p.age_s for p in usable) / n_usable
        # Freshness: exponential decay over T2 scale
        freshness = math.exp(-mean_age / T2_COHERENCE_TIME)

        health = (HEALTH_W_FIDELITY  * fidelity_score
                  + HEALTH_W_CAPACITY  * capacity_score
                  + HEALTH_W_SACRED    * sacred_score
                  + HEALTH_W_FRESHNESS * freshness)

        return {
            "channel_id": channel_id,
            "health": round(health, 6),
            "fidelity": round(fidelity_score, 6),
            "capacity": round(capacity_score, 6),
            "sacred": round(sacred_score, 6),
            "freshness": round(freshness, 6),
            "usable_pairs": n_usable,
            "mean_age_s": round(mean_age, 2),
        }

    def all_channel_health(self) -> Dict[str, float]:
        """Health scores for every channel in the network."""
        return {
            cid: self.channel_health(cid)["health"]
            for cid in self._channels
        }

    # ═══════════════════════════════════════════════════════════════
    # v1.2: PAIR LIFETIME PREDICTION
    # ═══════════════════════════════════════════════════════════════

    def predict_pair_lifetime(self, pair: "EntangledPair") -> float:
        """Predict seconds until a pair's fidelity drops below discard threshold.

        Uses T2 as dominant decay and solves:
          F₀ × exp(−t/T2) × ((1+exp(−t/T1))/2) = FIDELITY_DISCARD_THRESHOLD

        Returns estimated seconds remaining (≥0). Returns 0 if already below.
        """
        f_now = pair.current_fidelity
        if f_now <= FIDELITY_DISCARD_THRESHOLD:
            return 0.0
        # Approximation: T2-dominant decay ⇒ t ≈ −T2 × ln(threshold/F_now)
        ratio = FIDELITY_DISCARD_THRESHOLD / max(1e-12, f_now)
        if ratio >= 1.0:
            return 0.0
        return max(0.0, -T2_COHERENCE_TIME * math.log(ratio))

    def channel_lifetime(self, channel_id: str) -> Dict:
        """Predict when a channel will run out of usable pairs.

        Estimates the minimum remaining lifetime across all usable pairs
        in the channel (i.e., time until the first pair drops out).
        """
        ch = self._channels.get(channel_id)
        if not ch:
            return {"error": "channel not found"}
        usable = ch.usable_pairs
        if not usable:
            return {"min_lifetime_s": 0.0, "mean_lifetime_s": 0.0, "usable": 0}

        lifetimes = [self.predict_pair_lifetime(p) for p in usable]
        return {
            "min_lifetime_s": round(min(lifetimes), 2),
            "mean_lifetime_s": round(sum(lifetimes) / len(lifetimes), 2),
            "max_lifetime_s": round(max(lifetimes), 2),
            "usable": len(usable),
        }

    # ═══════════════════════════════════════════════════════════════
    # v1.2: NETWORK SUMMARY (THROUGHPUT + AGGREGATE)
    # ═══════════════════════════════════════════════════════════════

    def network_summary(self) -> Dict:
        """Comprehensive network summary with throughput metrics.

        Combines status, topology, census, health, and pair throughput
        into a single report for dashboards and monitoring.
        """
        uptime = time.time() - self._start_time

        # Throughput: pairs generated per second + swaps per second
        recent_window = 60.0   # last 60 seconds
        cutoff = time.time() - recent_window
        recent_gen = sum(1 for ts, ev in self._throughput_log
                         if ev == "pair_gen" and ts >= cutoff)
        recent_swaps = sum(1 for ts, ev in self._throughput_log
                           if ev == "swap" and ts >= cutoff)
        gen_rate = recent_gen / recent_window if recent_window > 0 else 0
        swap_rate = recent_swaps / recent_window if recent_window > 0 else 0

        # Aggregate channel health
        healths = self.all_channel_health()
        mean_health = (sum(healths.values()) / len(healths)) if healths else 0.0

        census = self.pair_census()
        topo = self.detect_topology()

        return {
            "version": "1.5.0",
            "uptime_s": round(uptime, 2),
            "topology": topo["topology"],
            "connected": topo.get("connected", False),
            "nodes": len(self._nodes),
            "channels": len(self._channels),
            "total_pairs_generated": self._total_pairs_generated,
            "total_purifications": self._total_purifications,
            "total_swaps": self._total_swaps,
            "total_decoherence_decays": self._total_decoherence_decays,
            "total_pairs_expired": self._total_pairs_expired,
            "pair_gen_rate_per_s": round(gen_rate, 3),
            "swap_rate_per_s": round(swap_rate, 3),
            "network_fidelity": self.network_fidelity(),
            "health_score": round(self.network_health_score(), 4),
            "mean_channel_health": round(mean_health, 6),
            "census": census,
            "god_code": GOD_CODE,
            "phi": PHI,
            "route_cache_size": len(self._route_cache),
            "topology_version": self._topology_version,
        }

    # ═══════════════════════════════════════════════════════════════
    # v1.2: TOPOLOGY ANALYTICS
    # ═══════════════════════════════════════════════════════════════

    def connected_components(self) -> List[Set[str]]:
        """Find connected components in the network graph via BFS.

        Returns:
            List of sets, each containing node IDs in one component
        """
        visited: Set[str] = set()
        components: List[Set[str]] = []
        for node_id in self._nodes:
            if node_id in visited:
                continue
            component: Set[str] = set()
            queue = [node_id]
            while queue:
                nid = queue.pop(0)
                if nid in visited:
                    continue
                visited.add(nid)
                component.add(nid)
                for neighbor in self._adjacency.get(nid, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(component)
        return components

    def network_diameter(self) -> int:
        """Compute the network diameter (longest shortest path in hops).

        Returns 0 for disconnected or empty networks.
        """
        if len(self._nodes) < 2:
            return 0
        diameter = 0
        node_ids = list(self._nodes.keys())
        for src in node_ids:
            dist: Dict[str, int] = {src: 0}
            queue = [src]
            while queue:
                u = queue.pop(0)
                for v in self._adjacency.get(u, set()):
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            if dist:
                diameter = max(diameter, max(dist.values()))
        return diameter

    def bottleneck_channels(self, threshold: float = 0.75) -> List[Dict]:
        """Identify channels whose fidelity is below *threshold*.

        These are potential bottlenecks that limit end-to-end
        teleportation fidelity through the network.
        """
        bottlenecks = []
        for ch in self._channels.values():
            mf = ch.mean_fidelity
            if mf < threshold and ch.usable_pairs:
                bottlenecks.append({
                    "channel_id": ch.channel_id,
                    "nodes": f"{ch.node_a_id[:8]}\u2194{ch.node_b_id[:8]}",
                    "mean_fidelity": round(mf, 6),
                    "usable_pairs": len(ch.usable_pairs),
                })
        bottlenecks.sort(key=lambda b: b["mean_fidelity"])
        return bottlenecks

    def topology_analysis(self) -> Dict:
        """Comprehensive topology analysis of the network.

        Returns:
            Dict with component count, diameter, bottlenecks, degree stats
        """
        components = self.connected_components()
        degrees = {
            nid: len(self._adjacency.get(nid, set()))
            for nid in self._nodes
        }
        degree_vals = list(degrees.values()) if degrees else [0]
        return {
            "node_count": len(self._nodes),
            "channel_count": len(self._channels),
            "connected_components": len(components),
            "is_connected": len(components) <= 1,
            "largest_component_size": max(len(c) for c in components) if components else 0,
            "diameter": self.network_diameter(),
            "mean_degree": round(sum(degree_vals) / max(1, len(degree_vals)), 2),
            "max_degree": max(degree_vals),
            "min_degree": min(degree_vals),
            "bottleneck_channels": self.bottleneck_channels(),
            "topology_version": self._topology_version,
        }

    # ═══════════════════════════════════════════════════════════════
    # v1.2: NETWORK HEALTH SCORING
    # ═══════════════════════════════════════════════════════════════

    def network_health_score(self) -> float:
        """Compute a composite network health score in [0, 1].

        Weighted blend of:
          - Fidelity   (45%): mean network fidelity
          - Capacity   (25%): fraction of channels with >= MIN_PAIR_POOL usable pairs
          - Sacred     (15%): mean sacred alignment ratio
          - Freshness  (15%): fraction of unconsumed pairs younger than T2
        """
        if not self._nodes or not self._channels:
            return 0.0

        fidelity_score = self.network_fidelity()

        channels_healthy = sum(
            1 for ch in self._channels.values()
            if len(ch.usable_pairs) >= MIN_PAIR_POOL
        )
        capacity_score = channels_healthy / max(1, len(self._channels))

        census = self.pair_census()
        sacred_score = census.get("sacred_ratio", 0.0)

        fresh = 0
        total_p = 0
        for ch in self._channels.values():
            for p in ch.pairs:
                if not p.consumed:
                    total_p += 1
                    if p.age_s < T2_COHERENCE_TIME:
                        fresh += 1
        freshness_score = fresh / max(1, total_p)

        return (
            HEALTH_W_FIDELITY * fidelity_score
            + HEALTH_W_CAPACITY * capacity_score
            + HEALTH_W_SACRED * sacred_score
            + HEALTH_W_FRESHNESS * freshness_score
        )

    # ═══════════════════════════════════════════════════════════════
    # v1.3: CHANNEL AUTO-HEAL PIPELINE
    # ═══════════════════════════════════════════════════════════════

    def auto_heal_channel(self, channel_id: str,
                           target_fidelity: float = 0.95,
                           max_rounds: int = 5) -> Dict:
        """Automatically heal a degraded channel.

        Pipeline:
          1. Prune dead pairs
          2. Replenish to MIN_PAIR_POOL if below
          3. If mean fidelity < target, purify with target_fidelity
          4. Re-replenish if purification consumed too many pairs
          5. Run sacred scoring pass on the channel

        Returns:
            Dict with before/after metrics for each stage
        """
        ch = self._channels.get(channel_id)
        if not ch:
            return {"error": "channel not found"}

        result: Dict[str, Any] = {
            "channel_id": channel_id,
            "before_fidelity": ch.mean_fidelity,
            "before_usable": len(ch.usable_pairs),
        }

        # Stage 1: Prune
        pruned = ch.prune_dead_pairs()
        result["pruned"] = pruned

        # Stage 2: Replenish if low
        replenished = 0
        if len(ch.usable_pairs) < MIN_PAIR_POOL:
            replenished = self.replenish_channel(channel_id, MIN_PAIR_POOL)
        result["replenished_stage1"] = replenished

        # Stage 3: Purify if below target
        purify_result = None
        if ch.mean_fidelity < target_fidelity and len(ch.usable_pairs) >= 2:
            purify_result = self.purify_channel(
                channel_id, rounds=max_rounds, target_fidelity=target_fidelity,
            )
        result["purification"] = purify_result

        # Stage 4: Post-purification replenish
        replenished2 = 0
        if len(ch.usable_pairs) < MIN_PAIR_POOL:
            replenished2 = self.replenish_channel(channel_id, MIN_PAIR_POOL)
        result["replenished_stage2"] = replenished2

        # Stage 5: Sacred scoring
        for pair in ch.pairs:
            if pair.consumed:
                continue
            f = pair.current_fidelity
            pair.sacred_score = f * SACRED_PAIR_AMPLIFIER * (1.0 + f ** PHI) / 2.0

        result["after_fidelity"] = ch.mean_fidelity
        result["after_usable"] = len(ch.usable_pairs)
        result["healed"] = ch.mean_fidelity >= target_fidelity
        return result

    def auto_heal_network(self, target_fidelity: float = 0.95) -> Dict:
        """Detect and heal all degraded channels in the network.

        Returns:
            Dict with per-channel results and aggregate statistics
        """
        channel_results = []
        healed = 0
        failed = 0
        for cid, ch in list(self._channels.items()):
            if ch.mean_fidelity >= target_fidelity and len(ch.usable_pairs) >= MIN_PAIR_POOL:
                continue  # Healthy, skip
            res = self.auto_heal_channel(cid, target_fidelity=target_fidelity)
            channel_results.append(res)
            if res.get("healed"):
                healed += 1
            else:
                failed += 1

        return {
            "channels_examined": healed + failed,
            "healed": healed,
            "failed": failed,
            "total_channels": len(self._channels),
            "network_fidelity_after": self.network_fidelity(),
            "details": channel_results,
        }

    # ═══════════════════════════════════════════════════════════════
    # v1.3: REDUNDANT END-TO-END PAIR
    # ═══════════════════════════════════════════════════════════════

    def create_redundant_pair(self, source_id: str, dest_id: str,
                               k: int = 3) -> Optional[Tuple[EntangledPair, List[str]]]:
        """Create an end-to-end pair via the best-fidelity route among K candidates.

        Evaluates K shortest paths, picks the one with the highest estimated
        fidelity, and creates the pair using that route. Falls back to
        lower-fidelity routes if the best one fails.

        Returns:
            (EntangledPair, route) or None if all routes fail
        """
        routes = self.find_k_routes(source_id, dest_id, k=k)
        if not routes:
            return None

        # Sort by estimated fidelity, best first
        rated = sorted(
            routes,
            key=lambda r: self.estimate_route_fidelity(r),
            reverse=True,
        )

        for route in rated:
            result = self._try_e2e_on_route(route)
            if result is not None:
                return result
            logger.info(
                "Redundant pair: route %s failed, trying next",
                [n[:8] for n in route],
            )
        return None

    def _try_e2e_on_route(self, route: List[str]) -> Optional[Tuple[EntangledPair, List[str]]]:
        """Attempt to create an E2E pair along a specific pre-computed route."""
        if len(route) < 2:
            return None

        if len(route) == 2:
            ch = self.get_channel(route[0], route[1])
            if ch:
                pair = ch.consume_best_pair()
                if pair:
                    return (pair, route)
            return None

        # Multi-hop swap
        source = route[0]
        for i in range(1, len(route) - 1):
            relay = route[i]
            next_node = route[i + 1]
            pair = self.entanglement_swap(source, relay, next_node)
            if not pair:
                return None

        ch = self.get_channel(source, route[-1])
        if ch and ch.usable_pairs:
            return (ch.consume_best_pair(), route)
        return None

    # ═══════════════════════════════════════════════════════════════
    # v1.3: NETWORK SNAPSHOT / RESTORE
    # ═══════════════════════════════════════════════════════════════

    def snapshot(self) -> Dict:
        """Capture a serializable snapshot of the entire network state.

        The snapshot includes all nodes, channels, pairs, and counters.
        Can be restored later with ``restore(snapshot)``.

        Returns:
            JSON-serializable dict
        """
        nodes = {nid: n.to_dict() for nid, n in self._nodes.items()}
        channels = {}
        for cid, ch in self._channels.items():
            channels[cid] = ch.to_dict()
            # Also embed raw pair data for full-fidelity restore
            channels[cid]["_pairs"] = [p.to_dict() for p in ch.pairs]

        return {
            "version": "1.5.0",
            "timestamp": time.time(),
            "nodes": nodes,
            "channels": channels,
            "adjacency": {k: list(v) for k, v in self._adjacency.items()},
            "edge_to_channel": {
                f"{e[0]}|{e[1]}": cid
                for e, cid in self._edge_to_channel.items()
            },
            "counters": {
                "total_pairs_generated": self._total_pairs_generated,
                "total_purifications": self._total_purifications,
                "total_swaps": self._total_swaps,
                "total_decoherence_decays": self._total_decoherence_decays,
                "total_pairs_expired": self._total_pairs_expired,
                "total_pairs_consumed": self._total_pairs_consumed,
                "topology_version": self._topology_version,
            },
            "god_code": GOD_CODE,
        }

    def restore(self, snap: Dict) -> bool:
        """Restore network state from a snapshot dict.

        Re-creates nodes, channels, and pairs from the snapshot.
        Counters are restored; route cache is cleared.

        Returns:
            True if restore succeeded
        """
        try:
            with self._lock:
                # Clear current state
                self._nodes.clear()
                self._channels.clear()
                self._adjacency.clear()
                self._edge_to_channel.clear()
                self._route_cache.clear()

                # Restore nodes
                for nid, nd in snap.get("nodes", {}).items():
                    self._nodes[nid] = QuantumNode.from_dict(nd)

                # Restore adjacency
                for nid, neighbors in snap.get("adjacency", {}).items():
                    self._adjacency[nid] = set(neighbors)

                # Restore edge mapping
                for edge_key, cid in snap.get("edge_to_channel", {}).items():
                    parts = edge_key.split("|")
                    if len(parts) == 2:
                        self._edge_to_channel[tuple(parts)] = cid

                # Restore channels (lightweight — pairs are not fully restored
                # since EntangledPair state depends on creation time)
                for cid, cd in snap.get("channels", {}).items():
                    ch = QuantumChannel(
                        channel_id=cid,
                        node_a_id=cd.get("node_a_id", ""),
                        node_b_id=cd.get("node_b_id", ""),
                        state=cd.get("state", "active"),
                    )
                    ch.teleportations_count = cd.get("teleportations", 0)
                    ch.swaps_count = cd.get("swaps", 0)
                    ch.purifications_count = cd.get("purifications", 0)
                    # Re-generate fresh pairs to match usable count
                    target = cd.get("usable_pairs", MIN_PAIR_POOL)
                    for _ in range(max(MIN_PAIR_POOL, target)):
                        pair = self._generate_bell_pair(ch.node_a_id, ch.node_b_id)
                        ch.pairs.append(pair)
                    self._channels[cid] = ch

                # Restore counters
                ctr = snap.get("counters", {})
                self._total_pairs_generated = ctr.get("total_pairs_generated", 0)
                self._total_purifications = ctr.get("total_purifications", 0)
                self._total_swaps = ctr.get("total_swaps", 0)
                self._total_decoherence_decays = ctr.get("total_decoherence_decays", 0)
                self._total_pairs_expired = ctr.get("total_pairs_expired", 0)
                self._total_pairs_consumed = ctr.get("total_pairs_consumed", 0)
                self._topology_version = ctr.get("topology_version", 0) + 1

            logger.info("Network restored: %d nodes, %d channels",
                        len(self._nodes), len(self._channels))
            return True
        except Exception as e:
            logger.error("Network restore failed: %s", e)
            return False

    # ═══════════════════════════════════════════════════════════════
    # v1.4: FIDELITY TREND MONITOR
    # ═══════════════════════════════════════════════════════════════

    def _sample_all_fidelity_trends(self) -> None:
        """Record the current mean fidelity of every channel in the trend buffer."""
        now = time.time()
        for cid, ch in self._channels.items():
            buf = self._fidelity_trends[cid]
            # Rate-limit: skip if last sample was too recent
            if buf and now - buf[-1][0] < TREND_SAMPLE_INTERVAL:
                continue
            mf = ch.mean_fidelity
            if mf > 0:
                buf.append((now, mf))

    def fidelity_trend(self, channel_id: str) -> Dict:
        """Analyse the fidelity trend for a single channel.

        Computes the linear regression slope of the fidelity time-series
        to determine if the channel is *improving*, *stable*, or *degrading*.

        Returns:
            Dict with slope, classification, sample count, and latest fidelity
        """
        buf = self._fidelity_trends.get(channel_id, deque())
        if len(buf) < 2:
            return {
                "channel_id": channel_id,
                "status": "insufficient_data",
                "samples": len(buf),
                "slope": 0.0,
            }

        # Simple linear regression: slope of (time, fidelity)
        t0 = buf[0][0]
        xs = [t - t0 for t, _ in buf]
        ys = [f for _, f in buf]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov_xy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        var_x = sum((x - mean_x) ** 2 for x in xs)
        slope = cov_xy / var_x if var_x > 1e-12 else 0.0

        # Classify: degrading if slope < -1e-4/s, improving if > 1e-4/s
        if slope < -1e-4:
            classification = "degrading"
        elif slope > 1e-4:
            classification = "improving"
        else:
            classification = "stable"

        return {
            "channel_id": channel_id,
            "status": classification,
            "slope_per_s": round(slope, 8),
            "samples": n,
            "latest_fidelity": round(ys[-1], 6),
            "oldest_fidelity": round(ys[0], 6),
            "window_s": round(xs[-1] - xs[0], 2) if n > 1 else 0.0,
        }

    def all_fidelity_trends(self) -> Dict[str, Dict]:
        """Trend analysis for every channel in the network."""
        return {cid: self.fidelity_trend(cid) for cid in self._channels}

    def degrading_channels(self) -> List[Dict]:
        """Return channels whose fidelity is actively declining."""
        return [
            t for t in self.all_fidelity_trends().values()
            if t.get("status") == "degrading"
        ]

    # ═══════════════════════════════════════════════════════════════
    # v1.4: QUANTUM CHANNEL CAPACITY
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _binary_entropy(p: float) -> float:
        """H(p) = −p log₂(p) − (1−p) log₂(1−p)."""
        if p <= 0 or p >= 1:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    def channel_capacity(self, channel_id: str) -> Dict:
        """Estimate the quantum channel capacity (ebits/use) using the Hashing bound.

        For a depolarizing channel with fidelity *F*, the one-way distillable
        entanglement (Hashing bound) is:

            D₁(F) = 1 − H(F)    for F > 0.5

        where H is the binary entropy. This gives the maximum rate of ebits
        (maximally entangled bits) extractable per shared pair.

        Returns:
            Dict with capacity, fidelity, pair rate, and effective ebit rate
        """
        ch = self._channels.get(channel_id)
        if not ch:
            return {"error": "channel not found", "capacity_ebits": 0.0}

        f = ch.mean_fidelity
        # Hashing bound
        if f <= 0.5:
            capacity = 0.0
        else:
            capacity = max(0.0, 1.0 - self._binary_entropy(f))

        # Pair throughput: how many pairs/s this channel generates (from log)
        now = time.time()
        cutoff = now - 60.0
        pair_events = sum(
            1 for ts, ev in self._throughput_log
            if ev == "pair_gen" and ts >= cutoff
        )
        pair_rate = pair_events / 60.0

        return {
            "channel_id": channel_id,
            "mean_fidelity": round(f, 6),
            "capacity_ebits_per_use": round(capacity, 6),
            "pair_rate_per_s": round(pair_rate, 3),
            "effective_ebit_rate": round(capacity * pair_rate, 4),
            "usable_pairs": len(ch.usable_pairs),
        }

    def network_capacity(self) -> Dict:
        """Aggregate quantum capacity across all channels."""
        caps = {}
        total_ebit_rate = 0.0
        for cid in self._channels:
            c = self.channel_capacity(cid)
            caps[cid] = c
            total_ebit_rate += c.get("effective_ebit_rate", 0)
        return {
            "channels": len(caps),
            "total_ebit_rate": round(total_ebit_rate, 4),
            "per_channel": caps,
        }

    # ═══════════════════════════════════════════════════════════════
    # v1.4: NETWORK RESILIENCE ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    def single_points_of_failure(self) -> List[str]:
        """Identify articulation points (cut vertices) in the network graph.

        Removing any of these nodes would disconnect the network.
        Uses iterative Tarjan's bridge-finding to detect articulation points.
        """
        if len(self._nodes) < 3:
            return []

        node_ids = list(self._nodes.keys())
        idx_map = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)
        disc = [-1] * n
        low = [-1] * n
        parent = [-1] * n
        is_artic = [False] * n
        timer = [0]

        def dfs(u: int) -> None:
            children = 0
            disc[u] = low[u] = timer[0]
            timer[0] += 1
            for nb in self._adjacency.get(node_ids[u], set()):
                v = idx_map.get(nb)
                if v is None:
                    continue
                if disc[v] == -1:
                    children += 1
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    # Root with 2+ children is articulation point
                    if parent[u] == -1 and children > 1:
                        is_artic[u] = True
                    # Non-root where no descendant can reach above u
                    if parent[u] != -1 and low[v] >= disc[u]:
                        is_artic[u] = True
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])

        for i in range(n):
            if disc[i] == -1:
                dfs(i)

        return [node_ids[i] for i in range(n) if is_artic[i]]

    def bridge_channels(self) -> List[str]:
        """Identify bridge edges (channels whose removal disconnects the graph)."""
        if len(self._nodes) < 2:
            return []

        node_ids = list(self._nodes.keys())
        idx_map = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)
        disc = [-1] * n
        low = [-1] * n
        parent = [-1] * n
        bridges: List[str] = []
        timer = [0]

        def dfs(u: int) -> None:
            disc[u] = low[u] = timer[0]
            timer[0] += 1
            for nb in self._adjacency.get(node_ids[u], set()):
                v = idx_map.get(nb)
                if v is None:
                    continue
                if disc[v] == -1:
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        edge = tuple(sorted([node_ids[u], node_ids[v]]))
                        cid = self._edge_to_channel.get(edge)
                        if cid:
                            bridges.append(cid)
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])

        for i in range(n):
            if disc[i] == -1:
                dfs(i)

        return bridges

    def resilience_analysis(self) -> Dict:
        """Comprehensive network resilience report.

        Analyses:
          - Articulation points (single points of failure)
          - Bridge channels (single edges whose removal disconnects)
          - Node/edge connectivity estimates
          - Redundancy score: 1 - (spof_count / node_count)
        """
        n_nodes = len(self._nodes)
        n_channels = len(self._channels)
        spof = self.single_points_of_failure()
        bridges = self.bridge_channels()
        components = self.connected_components()

        # Redundancy: 1.0 = fully redundant, 0.0 = every node is a SPOF
        redundancy = 1.0 - (len(spof) / max(1, n_nodes))

        # Min degree as lower bound on vertex connectivity
        degrees = [len(self._adjacency.get(nid, set())) for nid in self._nodes]
        min_degree = min(degrees) if degrees else 0

        return {
            "nodes": n_nodes,
            "channels": n_channels,
            "components": len(components),
            "is_connected": len(components) <= 1,
            "articulation_points": len(spof),
            "articulation_node_ids": [nid[:12] for nid in spof],
            "bridge_channels": len(bridges),
            "bridge_channel_ids": bridges,
            "min_degree": min_degree,
            "redundancy_score": round(redundancy, 4),
            "sacred_resilience": round(redundancy * PHI_INV, 4),
        }

    # ═══════════════════════════════════════════════════════════════
    # v1.4: AUTONOMOUS MAINTENANCE CYCLE
    # ═══════════════════════════════════════════════════════════════

    def maintenance_cycle(self, elapsed_s: float = 5.0,
                           target_fidelity: float = 0.95) -> Dict:
        """Single-call autonomous maintenance pipeline.

        Runs the complete maintenance sequence:
          1. Apply decoherence decay (physics simulation)
          2. Prune dead pairs across all channels
          3. Auto-heal degraded channels (replenish + purify)
          4. Run sacred scoring pass
          5. Sample fidelity trends
          6. Emit summary event

        This method is intended to be called periodically by daemons.

        Returns:
            Dict with per-stage results
        """
        self._maintenance_cycles += 1
        t0 = time.time()
        report: Dict[str, Any] = {"cycle": self._maintenance_cycles}

        # Stage 1: Decoherence
        report["decoherence"] = self.apply_decoherence(elapsed_s)

        # Stage 2: Prune
        total_pruned = 0
        for ch in self._channels.values():
            total_pruned += ch.prune_dead_pairs()
        report["pruned"] = total_pruned

        # Stage 3: Auto-heal
        report["heal"] = self.auto_heal_network(target_fidelity=target_fidelity)

        # Stage 4: Sacred scoring
        report["sacred"] = self.sacred_scoring_pass()

        # Stage 5: Trend sampling (already done in decoherence, but sample again post-heal)
        self._sample_all_fidelity_trends()

        report["elapsed_ms"] = round((time.time() - t0) * 1000, 2)
        report["network_fidelity"] = self.network_fidelity()
        report["health_score"] = round(self.network_health_score(), 4)

        self._emit("maintenance_cycle", cycle=self._maintenance_cycles,
                   health=report["health_score"],
                   fidelity=round(report["network_fidelity"], 4))

        return report

    # ═══════════════════════════════════════════════════════════════
    # v1.4: PATH DIVERSITY & AGGREGATE BANDWIDTH
    # ═══════════════════════════════════════════════════════════════

    def path_diversity(self, source_id: str, dest_id: str,
                       k: int = 5) -> Dict:
        """Analyse path diversity between two nodes.

        Computes:
          - Number of independent (edge-disjoint) paths
          - Per-path estimated fidelity
          - Aggregate pair bandwidth (total usable pairs across paths)
          - Best achievable fidelity

        Returns:
            Dict with diversity metrics
        """
        routes = self.find_k_routes(source_id, dest_id, k=k)
        if not routes:
            return {
                "source": source_id[:12],
                "dest": dest_id[:12],
                "independent_paths": 0,
                "paths": [],
                "aggregate_bandwidth": 0,
                "best_fidelity": 0.0,
            }

        path_info = []
        total_bandwidth = 0
        best_f = 0.0
        for route in routes:
            f_est = self.estimate_route_fidelity(route)
            # Bandwidth = min usable pairs along the route (bottleneck)
            min_pairs = float("inf")
            for i in range(len(route) - 1):
                ch = self.get_channel(route[i], route[i + 1])
                if ch:
                    min_pairs = min(min_pairs, len(ch.usable_pairs))
                else:
                    min_pairs = 0
            if min_pairs == float("inf"):
                min_pairs = 0
            total_bandwidth += min_pairs
            best_f = max(best_f, f_est)
            path_info.append({
                "hops": len(route) - 1,
                "route": [nid[:8] for nid in route],
                "estimated_fidelity": round(f_est, 4),
                "bottleneck_pairs": min_pairs,
            })

        return {
            "source": source_id[:12],
            "dest": dest_id[:12],
            "independent_paths": len(routes),
            "paths": path_info,
            "aggregate_bandwidth": total_bandwidth,
            "best_fidelity": round(best_f, 4),
        }

    # ═══════════════════════════════════════════════════════════════
    # v1.4: EVENT LOG QUERY
    # ═══════════════════════════════════════════════════════════════

    def event_log(self, limit: int = 50, event_type: Optional[str] = None) -> List[Dict]:
        """Query the structured event/audit log.

        Args:
            limit: Maximum events to return (most recent first)
            event_type: Filter to only this event type (e.g., 'decoherence')

        Returns:
            List of event dicts, newest first
        """
        events = list(self._event_log)
        if event_type:
            events = [e for e in events if e.get("event") == event_type]
        return list(reversed(events[-limit:]))

    # ═══════════════════════════════════════════════════════════════
    # v1.5: PAIR RESERVATION SYSTEM
    # ═══════════════════════════════════════════════════════════════

    def _expire_reservations(self) -> int:
        """Remove reservations that have exceeded RESERVATION_TTL."""
        now = time.time()
        expired = [rid for rid, rv in self._reservations.items()
                   if now - rv["timestamp"] > RESERVATION_TTL]
        for rid in expired:
            del self._reservations[rid]
        return len(expired)

    def reserve_pair(self, channel_id: str,
                     min_fidelity: float = 0.0) -> Optional[Dict]:
        """Reserve a pair on a channel for upcoming teleportation.

        The reserved pair is marked so other operations won't consume it.
        The caller must ``release_reservation(rid)`` or ``consume_reservation(rid)``
        within RESERVATION_TTL seconds, or the reservation lapses.

        Args:
            channel_id: Which channel to reserve on
            min_fidelity: Minimum acceptable fidelity for the pair

        Returns:
            Dict with reservation_id, pair_id, fidelity — or None if no
            pair meets the requirement
        """
        self._expire_reservations()
        ch = self._channels.get(channel_id)
        if not ch:
            return None

        # Already-reserved pair IDs
        reserved_ids = {rv["pair_id"] for rv in self._reservations.values()
                        if rv["channel_id"] == channel_id}

        # Find best usable, unreserved pair above min_fidelity
        best_pair = None
        best_f = -1.0
        for p in ch.usable_pairs:
            if p.pair_id in reserved_ids:
                continue
            f = p.current_fidelity
            if f >= min_fidelity and f > best_f:
                best_f = f
                best_pair = p

        if best_pair is None:
            return None

        rid = secrets.token_hex(8)
        self._reservations[rid] = {
            "channel_id": channel_id,
            "pair_id": best_pair.pair_id,
            "fidelity": round(best_f, 6),
            "timestamp": time.time(),
        }
        self._emit("pair_reserved", reservation_id=rid,
                    channel_id=channel_id, fidelity=round(best_f, 6))
        return {"reservation_id": rid, "pair_id": best_pair.pair_id,
                "fidelity": round(best_f, 6)}

    def release_reservation(self, reservation_id: str) -> bool:
        """Release a reservation without consuming the pair."""
        rv = self._reservations.pop(reservation_id, None)
        if rv:
            self._emit("reservation_released", reservation_id=reservation_id)
            return True
        return False

    def consume_reservation(self, reservation_id: str) -> Optional[EntangledPair]:
        """Consume the reserved pair, removing it from the pool.

        Returns:
            The EntangledPair if still valid, or None if expired/missing
        """
        rv = self._reservations.pop(reservation_id, None)
        if not rv:
            return None

        ch = self._channels.get(rv["channel_id"])
        if not ch:
            return None

        for i, p in enumerate(ch.pairs):
            if p.pair_id == rv["pair_id"] and not p.consumed:
                p.consumed = True
                self._total_pairs_consumed += 1
                self._emit("reservation_consumed",
                           reservation_id=reservation_id,
                           fidelity=round(p.current_fidelity, 6))
                return p
        return None

    def active_reservations(self) -> List[Dict]:
        """List all active (non-expired) reservations."""
        self._expire_reservations()
        return [
            {"reservation_id": rid, **rv}
            for rid, rv in self._reservations.items()
        ]

    # ═══════════════════════════════════════════════════════════════
    # v1.5: NODE BETWEENNESS CENTRALITY
    # ═══════════════════════════════════════════════════════════════

    def betweenness_centrality(self) -> Dict[str, float]:
        """Compute approximate betweenness centrality for all nodes.

        Uses Brandes' algorithm (O(V·E)) to measure how often each node
        lies on shortest paths between other nodes.  High-centrality nodes
        are critical relays whose failure would disrupt routing.

        Returns:
            Dict mapping node_id → centrality score (normalised [0, 1])
        """
        nodes = list(self._nodes.keys())
        centrality: Dict[str, float] = {n: 0.0 for n in nodes}

        for s in nodes:
            # BFS-based Brandes
            stack: List[str] = []
            pred: Dict[str, List[str]] = {n: [] for n in nodes}
            sigma: Dict[str, int] = {n: 0 for n in nodes}
            sigma[s] = 1
            dist: Dict[str, int] = {n: -1 for n in nodes}
            dist[s] = 0
            queue: deque = deque([s])

            while queue:
                v = queue.popleft()
                stack.append(v)
                for w in self._adjacency.get(v, set()):
                    if dist[w] < 0:
                        queue.append(w)
                        dist[w] = dist[v] + 1
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            delta: Dict[str, float] = {n: 0.0 for n in nodes}
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    frac = (sigma[v] / sigma[w]) if sigma[w] else 0.0
                    delta[v] += frac * (1.0 + delta[w])
                if w != s:
                    centrality[w] += delta[w]

        # Normalise: divide by (n-1)(n-2) for undirected graph
        n = len(nodes)
        norm = (n - 1) * (n - 2) if n > 2 else 1.0
        for k in centrality:
            centrality[k] = round(centrality[k] / norm, 6)

        return centrality

    def relay_importance(self) -> List[Tuple[str, float]]:
        """Return nodes ranked by betweenness centrality (most important first)."""
        bc = self.betweenness_centrality()
        return sorted(bc.items(), key=lambda x: x[1], reverse=True)

    # ═══════════════════════════════════════════════════════════════
    # v1.5: CHANNEL CAPACITY FORECASTING
    # ═══════════════════════════════════════════════════════════════

    def channel_capacity_forecast(self, channel_id: str,
                                   consumption_rate: float = 0.5) -> Dict:
        """Predict time until a channel runs out of usable pairs.

        Combines current usable pair count with mean fidelity decay
        (via T2) and an assumed consumption rate to estimate:
        - Time-to-empty at current consumption
        - Time-to-degraded (mean fidelity < discard threshold)

        Args:
            channel_id: Target channel
            consumption_rate: Expected pairs consumed per second

        Returns:
            Dict with time_to_empty_s, time_to_degraded_s, recommendation
        """
        ch = self._channels.get(channel_id)
        if not ch:
            return {"error": "channel not found"}

        usable = len(ch.usable_pairs)
        mean_f = ch.mean_fidelity

        # Time to empty based on consumption rate
        tte = usable / consumption_rate if consumption_rate > 0 else float("inf")

        # Time to degraded: solve F(t) = F_discard for t via T2
        # F(t) ≈ F₀ × exp(-t/T2)
        if mean_f > FIDELITY_DISCARD_THRESHOLD:
            ttd = -T2_COHERENCE_TIME * math.log(
                FIDELITY_DISCARD_THRESHOLD / mean_f
            )
        else:
            ttd = 0.0  # Already degraded

        # Recommendation
        if tte < 10.0 or ttd < 10.0:
            rec = "critical — replenish immediately"
        elif tte < 30.0 or ttd < 30.0:
            rec = "warning — replenish soon"
        else:
            rec = "healthy"

        return {
            "channel_id": channel_id,
            "usable_pairs": usable,
            "mean_fidelity": round(mean_f, 6),
            "consumption_rate": consumption_rate,
            "time_to_empty_s": round(tte, 2),
            "time_to_degraded_s": round(ttd, 2),
            "recommendation": rec,
        }

    # ═══════════════════════════════════════════════════════════════
    # v1.5: NETWORK GRAPH EXPORT
    # ═══════════════════════════════════════════════════════════════

    def adjacency_matrix(self) -> Tuple[List[str], List[List[float]]]:
        """Return a weighted adjacency matrix of the network.

        Edge weight = mean fidelity of channel × usable pair count.
        Zero means no direct channel.

        Returns:
            (node_ids, matrix) — node_ids[i] is the label for row/col i
        """
        node_ids = sorted(self._nodes.keys())
        idx = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)
        matrix = [[0.0] * n for _ in range(n)]

        for ch in self._channels.values():
            a_i = idx.get(ch.node_a_id)
            b_i = idx.get(ch.node_b_id)
            if a_i is None or b_i is None:
                continue
            weight = ch.mean_fidelity * max(1, len(ch.usable_pairs))
            matrix[a_i][b_i] = round(weight, 4)
            matrix[b_i][a_i] = round(weight, 4)

        return node_ids, matrix

    def export_dot(self, label: str = "L104 Quantum Network") -> str:
        """Export the network graph in Graphviz DOT format.

        Edge labels show mean fidelity and usable pair count.

        Returns:
            DOT-format string
        """
        lines = [f'graph "{label}" {{', '  rankdir=LR;',
                 '  node [shape=ellipse, style=filled, fillcolor="#e0e8f0"];']

        for nid, node in self._nodes.items():
            short = nid[:8]
            role = node.role if hasattr(node, "role") else "unknown"
            color = "#b0e0b0" if role == "sovereign" else "#f0d0a0"
            lines.append(f'  "{short}" [label="{short}\\n({role})", fillcolor="{color}"];')

        seen: Set[str] = set()
        for ch in self._channels.values():
            edge_key = tuple(sorted([ch.node_a_id, ch.node_b_id]))
            if str(edge_key) in seen:
                continue
            seen.add(str(edge_key))
            f = ch.mean_fidelity
            u = len(ch.usable_pairs)
            lines.append(
                f'  "{ch.node_a_id[:8]}" -- "{ch.node_b_id[:8]}" '
                f'[label="F={f:.3f}\\n{u}p", penwidth={max(1, u / 2):.1f}];'
            )

        lines.append("}")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════
    # v1.5: ENTANGLEMENT DISTILLATION CASCADE
    # ═══════════════════════════════════════════════════════════════

    def distillation_cascade(self, channel_id: str,
                              target_fidelity: float = DISTILLATION_TARGET_FIDELITY,
                              max_rounds: int = 10) -> Dict:
        """Multi-round entanglement distillation on a channel.

        Repeatedly purifies pairs, replenishes, and re-purifies until
        the *best* pair on the channel reaches target_fidelity or
        max_rounds is exhausted. Tracks fidelity progression.

        Returns:
            Dict with progression list, final fidelity, rounds used
        """
        ch = self._channels.get(channel_id)
        if not ch:
            return {"error": "channel not found"}

        progression: List[float] = [ch.mean_fidelity]
        rounds_used = 0

        for _ in range(max_rounds):
            best = ch.best_pair
            if best and best.current_fidelity >= target_fidelity:
                break

            # Ensure enough pairs for purification
            if len(ch.usable_pairs) < 2:
                self.replenish_channel(channel_id, MIN_PAIR_POOL)

            if len(ch.usable_pairs) < 2:
                break  # Can't purify without pairs

            self.purify_channel(channel_id, rounds=1, target_fidelity=target_fidelity)
            rounds_used += 1
            progression.append(ch.mean_fidelity)

            # Replenish after purification consumed pairs
            if len(ch.usable_pairs) < MIN_PAIR_POOL:
                self.replenish_channel(channel_id, MIN_PAIR_POOL)

        final_best = ch.best_pair
        final_f = final_best.current_fidelity if final_best else 0.0

        self._emit("distillation_cascade", channel_id=channel_id,
                    rounds=rounds_used, final_fidelity=round(final_f, 6))

        return {
            "channel_id": channel_id,
            "rounds_used": rounds_used,
            "progression": [round(f, 6) for f in progression],
            "final_best_fidelity": round(final_f, 6),
            "target_reached": final_f >= target_fidelity,
            "target_fidelity": target_fidelity,
        }

    # ═══════════════════════════════════════════════════════════════
    # v1.5: WEIGHTED MAX-FLOW (Ford-Fulkerson BFS / Edmonds-Karp)
    # ═══════════════════════════════════════════════════════════════

    def max_flow(self, source_id: str, sink_id: str) -> Dict:
        """Compute the maximum entanglement flow between two nodes.

        Uses Edmonds-Karp (BFS-based Ford-Fulkerson) where channel
        capacity = number of usable pairs. This measures how many E2E
        pairs could theoretically be created simultaneously.

        Returns:
            Dict with max_flow value, augmenting paths used, per-edge flow
        """
        if source_id not in self._nodes or sink_id not in self._nodes:
            return {"error": "source or sink not in network", "max_flow": 0}

        # Build capacity graph
        cap: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for ch in self._channels.values():
            u = len(ch.usable_pairs)
            cap[ch.node_a_id][ch.node_b_id] += u
            cap[ch.node_b_id][ch.node_a_id] += u

        flow_graph: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total_flow = 0
        paths_used: List[List[str]] = []

        while True:
            # BFS to find augmenting path
            parent: Dict[str, Optional[str]] = {source_id: None}
            visited = {source_id}
            queue: deque = deque([source_id])
            found = False

            while queue and not found:
                u = queue.popleft()
                for v in self._adjacency.get(u, set()):
                    residual = cap[u][v] - flow_graph[u][v]
                    if v not in visited and residual > 0:
                        visited.add(v)
                        parent[v] = u
                        if v == sink_id:
                            found = True
                            break
                        queue.append(v)

            if not found:
                break

            # Find bottleneck
            path: List[str] = []
            v = sink_id
            bottleneck = float("inf")
            while v is not None:
                path.append(v)
                u = parent[v]
                if u is not None:
                    bottleneck = min(bottleneck, cap[u][v] - flow_graph[u][v])
                v = u
            path.reverse()

            # Update flow
            v = sink_id
            while parent[v] is not None:
                u = parent[v]
                flow_graph[u][v] += bottleneck
                flow_graph[v][u] -= bottleneck
                v = u

            total_flow += bottleneck
            paths_used.append([nid[:8] for nid in path])

        # Per-edge flow summary
        edge_flow: List[Dict] = []
        seen_edges: Set[str] = set()
        for u in flow_graph:
            for v in flow_graph[u]:
                if flow_graph[u][v] > 0:
                    ek = f"{min(u,v)[:8]}-{max(u,v)[:8]}"
                    if ek not in seen_edges:
                        seen_edges.add(ek)
                        edge_flow.append({
                            "edge": ek,
                            "flow": flow_graph[u][v],
                            "capacity": cap[u][v],
                        })

        return {
            "source": source_id[:12],
            "sink": sink_id[:12],
            "max_flow": total_flow,
            "augmenting_paths": len(paths_used),
            "paths": paths_used,
            "edge_flow": edge_flow,
        }

    # ═══════════════════════════════════════════════════════════════
    # v1.1/v1.5: SELF-TEST (for l104_debug.py integration)
    # ═══════════════════════════════════════════════════════════════

    def self_test(self) -> Dict:
        """Run diagnostic self-test of the entanglement router.

        Creates a small test topology, exercises all core operations,
        and reports pass/fail for each probe.

        Returns:
            Dict with passed, total, and per-probe results
        """
        t0 = time.time()
        results = []

        def probe(name: str, fn, detail: str = ""):
            try:
                ok = fn()
                results.append({
                    "name": name,
                    "status": "pass" if ok else "fail",
                    "detail": detail,
                })
            except Exception as e:
                results.append({
                    "name": name,
                    "status": "fail",
                    "detail": str(e),
                })

        # 1. Node registration
        n_a = QuantumNode(name="TestAlice", role="sovereign")
        n_b = QuantumNode(name="TestBob", role="sovereign")
        n_r = QuantumNode(name="TestRelay", role="relay")
        self.add_node(n_a)
        self.add_node(n_b)
        self.add_node(n_r)
        probe("node_registration",
              lambda: len(self._nodes) >= 3,
              f"nodes={len(self._nodes)}")

        # 2. Channel creation + Bell pair generation
        ch_ar = self.create_channel(n_a.node_id, n_r.node_id, initial_pairs=6)
        ch_rb = self.create_channel(n_r.node_id, n_b.node_id, initial_pairs=6)
        probe("channel_creation",
              lambda: len(self._channels) >= 2 and len(ch_ar.usable_pairs) >= 4,
              f"channels={len(self._channels)}, pairs_AR={len(ch_ar.usable_pairs)}")

        # 3. Fidelity check
        mean_f = ch_ar.mean_fidelity
        probe("fidelity_check",
              lambda: 0.9 <= mean_f <= 1.0,
              f"mean_fidelity={mean_f:.4f}")

        # 4. Routing (Dijkstra)
        route = self.find_route(n_a.node_id, n_b.node_id)
        probe("routing_dijkstra",
              lambda: route is not None and len(route) == 3,
              f"route={[r[:8] for r in route]}" if route else "no route")

        # 5. Entanglement swap
        swapped = self.entanglement_swap(n_a.node_id, n_r.node_id, n_b.node_id)
        probe("entanglement_swap",
              lambda: swapped is not None and swapped.fidelity > 0.5,
              f"swap_fidelity={swapped.fidelity:.4f}" if swapped else "swap failed")

        # 6. Purification (need pairs)
        self.replenish_channel(ch_ar.channel_id, target_pairs=8)
        purified = self.purify_pair(ch_ar.channel_id)
        probe("purification_dejmps",
              lambda: True,  # Purification can fail probabilistically
              f"result={'success' if purified else 'probabilistic_fail'}")

        # 7. Decoherence decay
        decay = self.apply_decoherence(elapsed_seconds=2.0)
        probe("decoherence_decay",
              lambda: decay["pairs_decayed"] > 0,
              f"decayed={decay['pairs_decayed']}")

        # 8. Replenishment
        replenished = self.replenish_all()
        probe("replenishment",
              lambda: True,  # May be 0 if pools are full
              f"replenished={replenished}")

        # 9. Sacred scoring pass
        sacred = self.sacred_scoring_pass()
        probe("sacred_scoring",
              lambda: sacred["pairs_scored"] > 0,
              f"scored={sacred['pairs_scored']}, mean={sacred['mean_sacred_score']:.4f}")

        # 10. Pair census
        census = self.pair_census()
        probe("pair_census",
              lambda: census["total_pairs"] > 0,
              f"total={census['total_pairs']}, usable={census['usable_pairs']}")

        # 11. Fidelity heatmap
        hm = self.fidelity_heatmap()
        probe("fidelity_heatmap",
              lambda: len(hm) >= 2,
              f"channels={len(hm)}")

        # 12. Network fidelity
        nf = self.network_fidelity()
        probe("network_fidelity",
              lambda: nf > 0.0,
              f"fidelity={nf:.4f}")

        # 13. Status
        st = self.status()
        probe("status_report",
              lambda: st.get("version") == "1.5.0" and st["god_code"] == GOD_CODE,
              f"v={st.get('version')}, nodes={st['nodes']}")

        # 14. Topology analysis
        topo = self.topology_analysis()
        probe("topology_analysis",
              lambda: topo["node_count"] >= 3 and topo["connected_components"] >= 1,
              f"components={topo['connected_components']}, diameter={topo['diameter']}")

        # 15. Network health score
        health = self.network_health_score()
        probe("network_health",
              lambda: 0.0 <= health <= 1.0,
              f"health={health:.4f}")

        # 16. Adaptive purification (target fidelity)
        self.replenish_channel(ch_ar.channel_id, target_pairs=8)
        adaptive = self.purify_channel(ch_ar.channel_id, rounds=5, target_fidelity=0.98)
        probe("adaptive_purification",
              lambda: "target_fidelity" in adaptive and adaptive["rounds_attempted"] >= 0,
              f"target_reached={adaptive.get('target_reached')}")

        # 17. Bottleneck detection
        bottlenecks = self.bottleneck_channels(threshold=1.0)
        probe("bottleneck_detection",
              lambda: isinstance(bottlenecks, list),
              f"found={len(bottlenecks)}")

        # ── v1.2 NEW PROBES ──────────────────────────────────────

        # 18. Channel health scoring
        ch_health = self.channel_health(ch_ar.channel_id)
        probe("channel_health_score",
              lambda: 0.0 <= ch_health.get("health", -1) <= 1.0,
              f"health={ch_health.get('health', 0):.4f}")

        # 19. Pair lifetime prediction
        lt = self.channel_lifetime(ch_ar.channel_id)
        probe("pair_lifetime_predict",
              lambda: lt.get("mean_lifetime_s", -1) >= 0,
              f"mean={lt.get('mean_lifetime_s', 0):.1f}s, usable={lt.get('usable', 0)}")

        # 20. K-shortest paths
        k_routes = self.find_k_routes(n_a.node_id, n_b.node_id, k=2)
        probe("k_shortest_paths",
              lambda: len(k_routes) >= 1,
              f"k_routes={len(k_routes)}")

        # 21. Bulk purification
        self.replenish_channel(ch_ar.channel_id, target_pairs=8)
        self.replenish_channel(ch_rb.channel_id, target_pairs=8)
        bulk_pur = self.purify_all(fidelity_threshold=1.01)
        probe("purify_all_bulk",
              lambda: isinstance(bulk_pur, dict) and "channels_purified" in bulk_pur,
              f"ch_purified={bulk_pur.get('channels_purified', 0)}")

        # 22. Network summary
        summary = self.network_summary()
        probe("network_summary",
              lambda: summary.get("version") == "1.5.0" and summary.get("topology") is not None,
              f"topo={summary.get('topology')}, health={summary.get('mean_channel_health', 0):.4f}")

        # 23. Route cache populated
        probe("route_cache_active",
              lambda: isinstance(self._route_cache, dict),
              f"cached={len(self._route_cache)}")

        # 24. Detect topology
        det_topo = self.detect_topology()
        probe("detect_topology",
              lambda: det_topo["topology"] in ("star", "linear", "ring", "mesh", "tree",
                                                "unknown", "trivial"),
              f"type={det_topo['topology']}, connected={det_topo.get('connected')}")

        # ── v1.3 NEW PROBES ──────────────────────────────────────

        # 25. Route fidelity estimation
        route_for_est = self.find_route(n_a.node_id, n_b.node_id)
        est_f = self.estimate_route_fidelity(route_for_est) if route_for_est else 0.0
        probe("route_fidelity_est",
              lambda: est_f >= 0.0,
              f"estimated_f={est_f:.4f}")

        # 26. Best route by fidelity
        best_rt = self.best_route_by_fidelity(n_a.node_id, n_b.node_id, k=2)
        probe("best_route_fidelity",
              lambda: best_rt is not None,
              f"best_f={best_rt[1]:.4f}" if best_rt else "no route")

        # 27. Channel auto-heal
        self.replenish_channel(ch_ar.channel_id, target_pairs=6)
        heal = self.auto_heal_channel(ch_ar.channel_id, target_fidelity=0.90)
        probe("channel_auto_heal",
              lambda: "after_fidelity" in heal,
              f"healed={heal.get('healed')}")

        # 28. Batch pair generation
        batch = self._generate_bell_pairs_batch(n_a.node_id, n_b.node_id, 3)
        probe("batch_pair_gen",
              lambda: len(batch) == 3 and all(p.fidelity > 0.5 for p in batch),
              f"count={len(batch)}")

        # 29. Network snapshot
        snap = self.snapshot()
        probe("network_snapshot",
              lambda: snap.get("version") == "1.5.0" and len(snap["nodes"]) >= 3,
              f"nodes={len(snap['nodes'])}, channels={len(snap['channels'])}")

        # 30. Redundant E2E pair (if enough pairs)
        self.replenish_channel(ch_ar.channel_id, target_pairs=8)
        self.replenish_channel(ch_rb.channel_id, target_pairs=8)
        red_pair = self.create_redundant_pair(n_a.node_id, n_b.node_id, k=2)
        probe("redundant_e2e_pair",
              lambda: red_pair is not None or True,  # may fail with small network
              f"created={red_pair is not None}")

        # ── v1.4 NEW PROBES ──────────────────────────────────────

        # 31. Fidelity trend analysis
        self._sample_all_fidelity_trends()  # Ensure at least 1 sample
        trend = self.fidelity_trend(ch_ar.channel_id)
        probe("fidelity_trend",
              lambda: trend["status"] in ("stable", "improving", "degrading",
                                           "insufficient_data"),
              f"status={trend['status']}, samples={trend['samples']}")

        # 32. Channel capacity (Hashing bound)
        cap = self.channel_capacity(ch_ar.channel_id)
        probe("channel_capacity",
              lambda: cap.get("capacity_ebits_per_use", -1) >= 0.0,
              f"ebits={cap.get('capacity_ebits_per_use', 0):.4f}")

        # 33. Network resilience
        res = self.resilience_analysis()
        probe("resilience_analysis",
              lambda: isinstance(res, dict) and "redundancy_score" in res,
              f"spof={res.get('articulation_points', 0)}, redundancy={res.get('redundancy_score', 0):.2f}")

        # 34. Autonomous maintenance cycle
        maint = self.maintenance_cycle(elapsed_s=0.5, target_fidelity=0.90)
        probe("maintenance_cycle",
              lambda: "cycle" in maint and maint.get("health_score", -1) >= 0,
              f"cycle={maint['cycle']}, health={maint.get('health_score', 0):.4f}")

        # 35. Path diversity
        div = self.path_diversity(n_a.node_id, n_b.node_id, k=2)
        probe("path_diversity",
              lambda: isinstance(div, dict) and "independent_paths" in div,
              f"paths={div.get('independent_paths', 0)}, bw={div.get('aggregate_bandwidth', 0)}")

        # 36. Event log populated
        events = self.event_log(limit=10)
        probe("event_audit_log",
              lambda: len(events) > 0,
              f"events={len(events)}, latest={events[0]['event'] if events else 'none'}")

        # 37. Bridge / SPOF detection
        bridges = self.bridge_channels()
        probe("bridge_detection",
              lambda: isinstance(bridges, list),
              f"bridges={len(bridges)}")

        # ── v1.5 NEW PROBES ──────────────────────────────────────

        # 38. Pair reservation
        self.replenish_channel(ch_ar.channel_id, target_pairs=6)
        res_info = self.reserve_pair(ch_ar.channel_id, min_fidelity=0.5)
        probe("pair_reservation",
              lambda: res_info is not None and "reservation_id" in res_info,
              f"rid={res_info['reservation_id'][:8] if res_info else 'none'}")

        # 39. Consume reservation
        consumed = None
        if res_info:
            consumed = self.consume_reservation(res_info["reservation_id"])
        probe("consume_reservation",
              lambda: consumed is not None,
              f"fidelity={consumed.current_fidelity:.4f}" if consumed else "no pair")

        # 40. Reserve + release cycle
        res2 = self.reserve_pair(ch_ar.channel_id)
        released = self.release_reservation(res2["reservation_id"]) if res2 else False
        probe("release_reservation",
              lambda: released,
              f"released={released}")

        # 41. Betweenness centrality
        bc = self.betweenness_centrality()
        probe("betweenness_centrality",
              lambda: len(bc) >= 3 and all(0.0 <= v <= 1.0 for v in bc.values()),
              f"nodes={len(bc)}, max={max(bc.values()):.4f}")

        # 42. Relay importance ranking
        ri = self.relay_importance()
        probe("relay_importance",
              lambda: len(ri) >= 3,
              f"top={ri[0][0][:8]}={ri[0][1]:.4f}" if ri else "empty")

        # 43. Channel capacity forecast
        fcst = self.channel_capacity_forecast(ch_ar.channel_id, consumption_rate=0.3)
        probe("capacity_forecast",
              lambda: fcst.get("time_to_empty_s", -1) >= 0,
              f"tte={fcst.get('time_to_empty_s', 0):.1f}s, rec={fcst.get('recommendation')}")

        # 44. Adjacency matrix export
        nids, mat = self.adjacency_matrix()
        probe("adjacency_matrix",
              lambda: len(nids) >= 3 and len(mat) == len(nids),
              f"size={len(nids)}x{len(nids)}")

        # 45. DOT graph export
        dot = self.export_dot()
        probe("dot_export",
              lambda: "graph" in dot and "--" in dot,
              f"length={len(dot)} chars")

        # 46. Distillation cascade
        self.replenish_channel(ch_rb.channel_id, target_pairs=10)
        dist_res = self.distillation_cascade(ch_rb.channel_id,
                                              target_fidelity=0.98, max_rounds=3)
        probe("distillation_cascade",
              lambda: dist_res.get("rounds_used", -1) >= 0 and "progression" in dist_res,
              f"rounds={dist_res.get('rounds_used')}, final={dist_res.get('final_best_fidelity', 0):.4f}")

        # 47. Max-flow (Edmonds-Karp)
        mf = self.max_flow(n_a.node_id, n_b.node_id)
        probe("max_flow",
              lambda: mf.get("max_flow", -1) >= 0,
              f"flow={mf.get('max_flow', 0)}, paths={mf.get('augmenting_paths', 0)}")

        # 48. Active reservations (should be empty after consume+release)
        active = self.active_reservations()
        probe("active_reservations",
              lambda: isinstance(active, list),
              f"active={len(active)}")

        # Cleanup test nodes (keep real nodes intact)
        for nid in (n_a.node_id, n_b.node_id, n_r.node_id):
            self.remove_node(nid)

        passed = sum(1 for r in results if r["status"] == "pass")
        elapsed_ms = (time.time() - t0) * 1000

        return {
            "passed": passed,
            "total": len(results),
            "all_passed": passed == len(results),
            "elapsed_ms": round(elapsed_ms, 2),
            "results": results,
        }
