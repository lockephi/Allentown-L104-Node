#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
 L104 QUANTUM NETWORK SPEED BENCHMARK v1.0
 GOD_CODE=527.5184818492612 | PHI=1.618033988749895
═══════════════════════════════════════════════════════════════════
 Comprehensive speed benchmark for the Quantum Networker v1.4.0:

  1.  Bell Pair Generation — throughput for pair creation
  2.  Channel Create — node-to-node channel establishment
  3.  Dijkstra Routing — shortest-path latency
  4.  K-Shortest Routing — multi-path finding
  5.  Entanglement Swap — Bell measurement relay latency
  6.  BB84 QKD — key exchange throughput
  7.  E91 QKD — entangled key exchange throughput
  8.  Teleportation — score/phase/state/bitstring transfer
  9.  Multi-Hop Teleportation — relay chain transfer
  10. Repeater Chain — end-to-end chain establishment
  11. Purification (DEJMPS) — fidelity distillation speed
  12. Decoherence & Replenishment — maintenance cycle speed
  13. Sacred Scoring Pass — GOD_CODE pair scoring
  14. Network Analytics — topology, health, census, heatmap
  15. Scaling Analysis — node/channel count vs. latency
  16. Self-Test Diagnostic — full self_test() speed

 Reports per-subsystem latencies, throughput, and scaling.

 INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════
"""

import math
import time
import statistics
import json
import sys
import os

# ── Constants ────────────────────────────────────────────────────────────────
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497

# ── Imports ──────────────────────────────────────────────────────────────────
from l104_quantum_networker import (
    QuantumNetworker,
    QuantumKeyDistribution,
    EntanglementRouter,
    QuantumTeleporter,
    QuantumRepeaterChain,
    FidelityMonitor,
    QuantumNode,
    QuantumChannel,
    EntangledPair,
    QKDKey,
    TeleportResult,
    NetworkStatus,
)


# ═══════════════════════════════════════════════════════════════════
# UTILITY — Timing & Reporting
# ═══════════════════════════════════════════════════════════════════

class BenchTimer:
    """High-resolution benchmark timer with statistics."""

    def __init__(self, name: str):
        self.name = name
        self.times: list[float] = []
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        elapsed_ns = time.perf_counter_ns() - self._start
        self.times.append(elapsed_ns / 1_000_000)  # ms

    def record(self, elapsed_ms: float):
        self.times.append(elapsed_ms)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times) if self.times else 0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times) if self.times else 0

    @property
    def min_ms(self) -> float:
        return min(self.times) if self.times else 0

    @property
    def max_ms(self) -> float:
        return max(self.times) if self.times else 0

    @property
    def stdev_ms(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0

    @property
    def throughput_hz(self) -> float:
        """Operations per second."""
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else float('inf')

    def summary(self) -> dict:
        return {
            "name": self.name,
            "iterations": len(self.times),
            "mean_ms": round(self.mean_ms, 4),
            "median_ms": round(self.median_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "stdev_ms": round(self.stdev_ms, 4),
            "throughput_hz": round(self.throughput_hz, 2),
        }


def _header(title: str):
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print(f"{'─' * 65}")


def _result(label: str, timer: BenchTimer, extra: str = ""):
    s = timer.summary()
    line = f"  {label:<40s} {s['mean_ms']:>9.3f} ms  ({s['throughput_hz']:>10.1f} ops/s)"
    if extra:
        line += f"  {extra}"
    print(line)


def _fresh_network(name: str = "L104-Bench") -> QuantumNetworker:
    """Create a fresh QuantumNetworker for benchmarking."""
    return QuantumNetworker(node_name=name, simulation_mode=True)


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 1: Bell Pair Generation
# ═══════════════════════════════════════════════════════════════════

def bench_bell_pair_generation():
    _header("BENCHMARK 1: Bell Pair Generation — Throughput")
    results = {}
    net = _fresh_network()
    alice = net.add_node("Alice", role="sovereign")
    bob = net.add_node("Bob", role="sovereign")

    # 1a. Single pair generation via channel creation
    timer_single = BenchTimer("single_pair")
    N = 50
    for _ in range(N):
        n = _fresh_network()
        a = n.add_node("A", role="sovereign")
        b = n.add_node("B", role="sovereign")
        with timer_single:
            n.connect(a.node_id, b.node_id, pairs=1)
    _result("Single pair (incl. channel)", timer_single)
    results["single_pair_ms"] = timer_single.mean_ms

    # 1b. Batch pair generation (8 pairs)
    timer_batch8 = BenchTimer("batch_8_pairs")
    for _ in range(N):
        n = _fresh_network()
        a = n.add_node("A", role="sovereign")
        b = n.add_node("B", role="sovereign")
        with timer_batch8:
            n.connect(a.node_id, b.node_id, pairs=8)
    _result("8-pair batch (incl. channel)", timer_batch8)
    results["batch_8_pairs_ms"] = timer_batch8.mean_ms

    # 1c. Large batch (32 pairs)
    timer_batch32 = BenchTimer("batch_32_pairs")
    for _ in range(20):
        n = _fresh_network()
        a = n.add_node("A", role="sovereign")
        b = n.add_node("B", role="sovereign")
        with timer_batch32:
            n.connect(a.node_id, b.node_id, pairs=32)
    _result("32-pair batch (incl. channel)", timer_batch32)
    results["batch_32_pairs_ms"] = timer_batch32.mean_ms

    # 1d. Replenishment throughput
    timer_replenish = BenchTimer("replenish")
    ch = net.connect(alice.node_id, bob.node_id, pairs=4)
    for _ in range(N):
        with timer_replenish:
            net.router.replenish_channel(ch.channel_id, target_pairs=12)
    _result("Replenish to 12 pairs", timer_replenish)
    results["replenish_ms"] = timer_replenish.mean_ms

    results["throughput_pairs_per_sec"] = round(8000.0 / timer_batch8.mean_ms, 1) if timer_batch8.mean_ms > 0 else 0
    print(f"\n  ★ Pair throughput: {results['throughput_pairs_per_sec']:.0f} pairs/sec (8-batch)")
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 2: Channel Creation
# ═══════════════════════════════════════════════════════════════════

def bench_channel_creation():
    _header("BENCHMARK 2: Channel Creation — Latency")
    results = {}
    N = 40

    timer_create = BenchTimer("channel_create")
    for i in range(N):
        n = _fresh_network()
        a = n.add_node(f"A{i}", role="sovereign")
        b = n.add_node(f"B{i}", role="sovereign")
        with timer_create:
            n.connect(a.node_id, b.node_id, pairs=6)
    _result("Channel create (6 pairs)", timer_create)
    results["channel_create_ms"] = timer_create.mean_ms

    # Many channels in same network
    net = _fresh_network()
    nodes = [net.add_node(f"N{i}", role="sovereign") for i in range(10)]
    timer_multi = BenchTimer("multi_channel")
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            with timer_multi:
                net.connect(nodes[i].node_id, nodes[j].node_id, pairs=4)
    _result("Channel in 10-node mesh (4 pairs)", timer_multi)
    results["mesh_channel_ms"] = timer_multi.mean_ms
    results["mesh_channels_created"] = len(timer_multi.times)
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 3: Dijkstra Routing
# ═══════════════════════════════════════════════════════════════════

def bench_routing_dijkstra():
    _header("BENCHMARK 3: Dijkstra Routing — Shortest Path")
    results = {}

    # Build a chain topology: N0 — N1 — N2 — ... — N9
    net = _fresh_network()
    chain = [net.add_node(f"Chain-{i}", role="sovereign") for i in range(10)]
    for i in range(9):
        net.connect(chain[i].node_id, chain[i + 1].node_id, pairs=6)

    # 3a. Short route (2 hops)
    timer_short = BenchTimer("route_2hop")
    for _ in range(200):
        net.router._route_cache.clear()
        with timer_short:
            net.router.find_route(chain[0].node_id, chain[2].node_id)
    _result("2-hop route (no cache)", timer_short)
    results["route_2hop_ms"] = timer_short.mean_ms

    # 3b. Long route (9 hops)
    timer_long = BenchTimer("route_9hop")
    for _ in range(200):
        net.router._route_cache.clear()
        with timer_long:
            net.router.find_route(chain[0].node_id, chain[9].node_id)
    _result("9-hop route (no cache)", timer_long)
    results["route_9hop_ms"] = timer_long.mean_ms

    # 3c. Cached route (should be near-zero)
    net.router._route_cache.clear()
    net.router.find_route(chain[0].node_id, chain[9].node_id)
    timer_cached = BenchTimer("route_cached")
    for _ in range(500):
        with timer_cached:
            net.router.find_route(chain[0].node_id, chain[9].node_id)
    _result("9-hop cached route", timer_cached)
    results["route_cached_ms"] = timer_cached.mean_ms

    # 3d. Mesh routing (10 nodes fully connected)
    net2 = _fresh_network()
    mesh = [net2.add_node(f"Mesh-{i}", role="sovereign") for i in range(10)]
    for i in range(10):
        for j in range(i + 1, 10):
            net2.connect(mesh[i].node_id, mesh[j].node_id, pairs=4)

    timer_mesh = BenchTimer("mesh_route")
    for _ in range(200):
        net2.router._route_cache.clear()
        with timer_mesh:
            net2.router.find_route(mesh[0].node_id, mesh[9].node_id)
    _result("10-node mesh route (no cache)", timer_mesh)
    results["mesh_route_ms"] = timer_mesh.mean_ms
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 4: K-Shortest Routes
# ═══════════════════════════════════════════════════════════════════

def bench_k_shortest():
    _header("BENCHMARK 4: K-Shortest Routes — Multi-Path Finding")
    results = {}

    net = _fresh_network()
    nodes = [net.add_node(f"K-{i}", role="sovereign") for i in range(8)]
    # Create redundant paths
    for i in range(7):
        net.connect(nodes[i].node_id, nodes[i + 1].node_id, pairs=6)
    # Cross-links for alternative paths
    net.connect(nodes[0].node_id, nodes[3].node_id, pairs=4)
    net.connect(nodes[0].node_id, nodes[5].node_id, pairs=4)
    net.connect(nodes[2].node_id, nodes[6].node_id, pairs=4)
    net.connect(nodes[4].node_id, nodes[7].node_id, pairs=4)

    timer_k2 = BenchTimer("k2_routes")
    for _ in range(100):
        with timer_k2:
            net.router.find_k_routes(nodes[0].node_id, nodes[7].node_id, k=2)
    _result("K=2 routes (8 nodes)", timer_k2)
    results["k2_routes_ms"] = timer_k2.mean_ms

    timer_k3 = BenchTimer("k3_routes")
    for _ in range(100):
        with timer_k3:
            net.router.find_k_routes(nodes[0].node_id, nodes[7].node_id, k=3)
    _result("K=3 routes (8 nodes)", timer_k3)
    results["k3_routes_ms"] = timer_k3.mean_ms
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 5: Entanglement Swap
# ═══════════════════════════════════════════════════════════════════

def bench_entanglement_swap():
    _header("BENCHMARK 5: Entanglement Swap — Relay Latency")
    results = {}
    N = 50

    timer_swap = BenchTimer("swap")
    for _ in range(N):
        net = _fresh_network()
        a = net.add_node("A", role="sovereign")
        r = net.add_node("R", role="relay")
        b = net.add_node("B", role="sovereign")
        net.connect(a.node_id, r.node_id, pairs=8)
        net.connect(r.node_id, b.node_id, pairs=8)
        with timer_swap:
            net.router.entanglement_swap(a.node_id, r.node_id, b.node_id)
    _result("Single entanglement swap", timer_swap)
    results["swap_ms"] = timer_swap.mean_ms
    results["swap_throughput_hz"] = timer_swap.throughput_hz
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 6: BB84 QKD
# ═══════════════════════════════════════════════════════════════════

def bench_bb84_qkd():
    _header("BENCHMARK 6: BB84 QKD — Key Exchange Throughput")
    results = {}
    N = 30

    # 6a. 256-bit key
    timer_256 = BenchTimer("bb84_256")
    for _ in range(N):
        net = _fresh_network()
        a = net.add_node("A", role="sovereign")
        b = net.add_node("B", role="sovereign")
        net.connect(a.node_id, b.node_id, pairs=8)
        with timer_256:
            net.establish_qkd(a.node_id, b.node_id, "bb84", 256)
    _result("BB84 256-bit key", timer_256)
    results["bb84_256_ms"] = timer_256.mean_ms

    # 6b. 512-bit key
    timer_512 = BenchTimer("bb84_512")
    for _ in range(N):
        net = _fresh_network()
        a = net.add_node("A", role="sovereign")
        b = net.add_node("B", role="sovereign")
        net.connect(a.node_id, b.node_id, pairs=8)
        with timer_512:
            net.establish_qkd(a.node_id, b.node_id, "bb84", 512)
    _result("BB84 512-bit key", timer_512)
    results["bb84_512_ms"] = timer_512.mean_ms

    # 6c. 1024-bit key
    timer_1024 = BenchTimer("bb84_1024")
    for _ in range(20):
        net = _fresh_network()
        a = net.add_node("A", role="sovereign")
        b = net.add_node("B", role="sovereign")
        net.connect(a.node_id, b.node_id, pairs=8)
        with timer_1024:
            net.establish_qkd(a.node_id, b.node_id, "bb84", 1024)
    _result("BB84 1024-bit key", timer_1024)
    results["bb84_1024_ms"] = timer_1024.mean_ms

    results["bb84_bits_per_sec"] = round(256000.0 / timer_256.mean_ms, 1) if timer_256.mean_ms > 0 else 0
    print(f"\n  ★ BB84 throughput: {results['bb84_bits_per_sec']:.0f} bits/sec (256-bit)")
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 7: E91 QKD
# ═══════════════════════════════════════════════════════════════════

def bench_e91_qkd():
    _header("BENCHMARK 7: E91 QKD — Entangled Key Exchange")
    results = {}
    N = 30

    timer_e91 = BenchTimer("e91_256")
    for _ in range(N):
        net = _fresh_network()
        a = net.add_node("A", role="sovereign")
        b = net.add_node("B", role="sovereign")
        net.connect(a.node_id, b.node_id, pairs=8)
        with timer_e91:
            net.establish_qkd(a.node_id, b.node_id, "e91", 256)
    _result("E91 256-bit key", timer_e91)
    results["e91_256_ms"] = timer_e91.mean_ms
    results["e91_bits_per_sec"] = round(256000.0 / timer_e91.mean_ms, 1) if timer_e91.mean_ms > 0 else 0
    print(f"\n  ★ E91 throughput: {results['e91_bits_per_sec']:.0f} bits/sec (256-bit)")
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 8: Teleportation
# ═══════════════════════════════════════════════════════════════════

def bench_teleportation():
    _header("BENCHMARK 8: Quantum Teleportation — Transfer Latency")
    results = {}
    N = 40

    # 8a. Score teleportation
    timer_score = BenchTimer("teleport_score")
    for _ in range(N):
        net = _fresh_network()
        a = net.add_node("A", role="sovereign")
        b = net.add_node("B", role="sovereign")
        net.connect(a.node_id, b.node_id, pairs=8)
        with timer_score:
            net.teleport_score(a.node_id, b.node_id, score=0.618)
    _result("Score teleportation", timer_score)
    results["teleport_score_ms"] = timer_score.mean_ms

    # 8b. Phase teleportation
    timer_phase = BenchTimer("teleport_phase")
    for _ in range(N):
        net = _fresh_network()
        a = net.add_node("A", role="sovereign")
        b = net.add_node("B", role="sovereign")
        net.connect(a.node_id, b.node_id, pairs=8)
        with timer_phase:
            net.teleport_phase(a.node_id, b.node_id, phase=math.pi / PHI)
    _result("Phase teleportation", timer_phase)
    results["teleport_phase_ms"] = timer_phase.mean_ms

    # 8c. State teleportation (|ψ⟩ = 0.6|0⟩ + 0.8i|1⟩)
    timer_state = BenchTimer("teleport_state")
    alpha, beta = complex(0.6, 0), complex(0, 0.8)
    for _ in range(N):
        net = _fresh_network()
        a = net.add_node("A", role="sovereign")
        b = net.add_node("B", role="sovereign")
        net.connect(a.node_id, b.node_id, pairs=8)
        with timer_state:
            net.teleport_state(a.node_id, b.node_id, state_vector=[alpha, beta])
    _result("State teleportation", timer_state)
    results["teleport_state_ms"] = timer_state.mean_ms

    # 8d. Bitstring teleportation
    timer_bits = BenchTimer("teleport_bitstring")
    for _ in range(N):
        net = _fresh_network()
        a = net.add_node("A", role="sovereign")
        b = net.add_node("B", role="sovereign")
        net.connect(a.node_id, b.node_id, pairs=8)
        with timer_bits:
            net.teleport_bitstring(a.node_id, b.node_id, bitstring="10110100")
    _result("Bitstring teleportation (8-bit)", timer_bits)
    results["teleport_bitstring_ms"] = timer_bits.mean_ms

    # 8e. GOD_CODE sacred score teleport
    timer_gc = BenchTimer("teleport_godcode")
    gc_frac = GOD_CODE % 1.0
    for _ in range(N):
        net = _fresh_network()
        a = net.add_node("A", role="sovereign")
        b = net.add_node("B", role="sovereign")
        net.connect(a.node_id, b.node_id, pairs=8)
        with timer_gc:
            net.teleport_score(a.node_id, b.node_id, score=gc_frac)
    _result("GOD_CODE score teleport", timer_gc)
    results["teleport_godcode_ms"] = timer_gc.mean_ms

    results["teleport_throughput_hz"] = timer_score.throughput_hz
    print(f"\n  ★ Teleport throughput: {results['teleport_throughput_hz']:.0f} teleports/sec")
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 9: Multi-Hop Teleportation
# ═══════════════════════════════════════════════════════════════════

def bench_multihop_teleport():
    _header("BENCHMARK 9: Multi-Hop Teleportation — Relay Chain")
    results = {}
    N = 30

    # 9a. 2-hop teleport (A — R — B)
    timer_2hop = BenchTimer("multihop_2")
    for _ in range(N):
        net = _fresh_network()
        a = net.add_node("A", role="sovereign")
        r = net.add_node("R", role="relay")
        b = net.add_node("B", role="sovereign")
        net.connect(a.node_id, r.node_id, pairs=8)
        net.connect(r.node_id, b.node_id, pairs=8)
        with timer_2hop:
            net.teleport_score(a.node_id, b.node_id, score=0.75)
    _result("2-hop teleport (A→R→B)", timer_2hop)
    results["multihop_2_ms"] = timer_2hop.mean_ms

    # 9b. 3-hop teleport (A — R1 — R2 — B)
    timer_3hop = BenchTimer("multihop_3")
    for _ in range(N):
        net = _fresh_network()
        a = net.add_node("A", role="sovereign")
        r1 = net.add_node("R1", role="relay")
        r2 = net.add_node("R2", role="relay")
        b = net.add_node("B", role="sovereign")
        net.connect(a.node_id, r1.node_id, pairs=8)
        net.connect(r1.node_id, r2.node_id, pairs=8)
        net.connect(r2.node_id, b.node_id, pairs=8)
        with timer_3hop:
            net.teleport_score(a.node_id, b.node_id, score=0.75)
    _result("3-hop teleport (A→R1→R2→B)", timer_3hop)
    results["multihop_3_ms"] = timer_3hop.mean_ms

    # 9c. 5-hop teleport
    timer_5hop = BenchTimer("multihop_5")
    for _ in range(20):
        net = _fresh_network()
        nodes = [net.add_node(f"N{i}", role="relay" if 0 < i < 5 else "sovereign") for i in range(6)]
        for i in range(5):
            net.connect(nodes[i].node_id, nodes[i + 1].node_id, pairs=8)
        with timer_5hop:
            net.teleport_score(nodes[0].node_id, nodes[5].node_id, score=0.5)
    _result("5-hop teleport", timer_5hop)
    results["multihop_5_ms"] = timer_5hop.mean_ms
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 10: Repeater Chain
# ═══════════════════════════════════════════════════════════════════

def bench_repeater_chain():
    _header("BENCHMARK 10: Repeater Chain — End-to-End Establishment")
    results = {}
    N = 25

    # 10a. 3-segment chain
    timer_3seg = BenchTimer("chain_3seg")
    for _ in range(N):
        net = _fresh_network()
        nodes = [net.add_node(f"C{i}", role="relay" if 0 < i < 3 else "sovereign") for i in range(4)]
        for i in range(3):
            net.connect(nodes[i].node_id, nodes[i + 1].node_id, pairs=8)
        with timer_3seg:
            net.repeater.establish_chain(
                [n.node_id for n in nodes],
                pairs_per_segment=6,
            )
    _result("3-segment repeater chain", timer_3seg)
    results["chain_3seg_ms"] = timer_3seg.mean_ms

    # 10b. 5-segment chain
    timer_5seg = BenchTimer("chain_5seg")
    for _ in range(20):
        net = _fresh_network()
        nodes = [net.add_node(f"C{i}", role="relay" if 0 < i < 5 else "sovereign") for i in range(6)]
        for i in range(5):
            net.connect(nodes[i].node_id, nodes[i + 1].node_id, pairs=8)
        with timer_5seg:
            net.repeater.establish_chain(
                [n.node_id for n in nodes],
                pairs_per_segment=6,
            )
    _result("5-segment repeater chain", timer_5seg)
    results["chain_5seg_ms"] = timer_5seg.mean_ms

    # 10c. Fidelity estimation (no network needed)
    timer_est = BenchTimer("fidelity_estimate")
    net_tmp = _fresh_network()
    for _ in range(500):
        with timer_est:
            net_tmp.repeater.estimate_chain_fidelity(5, segment_fidelity=0.995)
    _result("Fidelity estimation (5 seg)", timer_est)
    results["fidelity_estimate_ms"] = timer_est.mean_ms
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 11: DEJMPS Purification
# ═══════════════════════════════════════════════════════════════════

def bench_purification():
    _header("BENCHMARK 11: DEJMPS Purification — Fidelity Distillation")
    results = {}
    N = 30

    # 11a. Single-round purification
    timer_1r = BenchTimer("purify_1round")
    for _ in range(N):
        net = _fresh_network()
        a = net.add_node("A", role="sovereign")
        b = net.add_node("B", role="sovereign")
        ch = net.connect(a.node_id, b.node_id, pairs=16)
        with timer_1r:
            net.purify(a.node_id, b.node_id, rounds=1)
    _result("1-round purification", timer_1r)
    results["purify_1round_ms"] = timer_1r.mean_ms

    # 11b. 3-round purification
    timer_3r = BenchTimer("purify_3rounds")
    for _ in range(N):
        net = _fresh_network()
        a = net.add_node("A", role="sovereign")
        b = net.add_node("B", role="sovereign")
        ch = net.connect(a.node_id, b.node_id, pairs=16)
        with timer_3r:
            net.purify(a.node_id, b.node_id, rounds=3)
    _result("3-round purification", timer_3r)
    results["purify_3rounds_ms"] = timer_3r.mean_ms

    # 11c. 5-round purification
    timer_5r = BenchTimer("purify_5rounds")
    for _ in range(20):
        net = _fresh_network()
        a = net.add_node("A", role="sovereign")
        b = net.add_node("B", role="sovereign")
        ch = net.connect(a.node_id, b.node_id, pairs=24)
        with timer_5r:
            net.purify(a.node_id, b.node_id, rounds=5)
    _result("5-round purification", timer_5r)
    results["purify_5rounds_ms"] = timer_5r.mean_ms
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 12: Decoherence & Replenishment
# ═══════════════════════════════════════════════════════════════════

def bench_decoherence_replenish():
    _header("BENCHMARK 12: Decoherence & Replenishment — Maintenance")
    results = {}

    net = _fresh_network()
    nodes = [net.add_node(f"D{i}", role="sovereign") for i in range(6)]
    for i in range(5):
        net.connect(nodes[i].node_id, nodes[i + 1].node_id, pairs=8)
    # Cross links
    net.connect(nodes[0].node_id, nodes[3].node_id, pairs=6)
    net.connect(nodes[1].node_id, nodes[4].node_id, pairs=6)

    # 12a. Decoherence pass
    timer_decohere = BenchTimer("decoherence")
    for _ in range(100):
        with timer_decohere:
            net.router.apply_decoherence(elapsed_seconds=1.0)
    _result("Decoherence pass (7 channels)", timer_decohere)
    results["decoherence_ms"] = timer_decohere.mean_ms

    # 12b. Replenish all
    timer_replenish_all = BenchTimer("replenish_all")
    for _ in range(50):
        with timer_replenish_all:
            net.router.replenish_all()
    _result("Replenish all channels", timer_replenish_all)
    results["replenish_all_ms"] = timer_replenish_all.mean_ms
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 13: Sacred Scoring Pass
# ═══════════════════════════════════════════════════════════════════

def bench_sacred_scoring():
    _header("BENCHMARK 13: Sacred Scoring Pass — GOD_CODE Alignment")
    results = {}

    net = _fresh_network()
    nodes = [net.add_node(f"S{i}", role="sovereign") for i in range(6)]
    for i in range(5):
        net.connect(nodes[i].node_id, nodes[i + 1].node_id, pairs=8)
    net.connect(nodes[0].node_id, nodes[5].node_id, pairs=6)

    timer_sacred = BenchTimer("sacred_scoring")
    for _ in range(100):
        with timer_sacred:
            net.router.sacred_scoring_pass()
    _result("Sacred scoring pass (6 channels)", timer_sacred)
    results["sacred_scoring_ms"] = timer_sacred.mean_ms
    results["sacred_throughput_hz"] = timer_sacred.throughput_hz
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 14: Network Analytics
# ═══════════════════════════════════════════════════════════════════

def bench_network_analytics():
    _header("BENCHMARK 14: Network Analytics — Topology & Health")
    results = {}

    # Build a medium network (8 nodes, 12 channels)
    net = _fresh_network()
    nodes = [net.add_node(f"A{i}", role="sovereign") for i in range(8)]
    # Chain
    for i in range(7):
        net.connect(nodes[i].node_id, nodes[i + 1].node_id, pairs=6)
    # Cross-links
    net.connect(nodes[0].node_id, nodes[4].node_id, pairs=4)
    net.connect(nodes[1].node_id, nodes[5].node_id, pairs=4)
    net.connect(nodes[2].node_id, nodes[6].node_id, pairs=4)
    net.connect(nodes[3].node_id, nodes[7].node_id, pairs=4)
    net.connect(nodes[0].node_id, nodes[7].node_id, pairs=4)

    N = 100

    # 14a. Pair census
    timer_census = BenchTimer("pair_census")
    for _ in range(N):
        with timer_census:
            net.router.pair_census()
    _result("Pair census", timer_census)
    results["pair_census_ms"] = timer_census.mean_ms

    # 14b. Fidelity heatmap
    timer_heatmap = BenchTimer("fidelity_heatmap")
    for _ in range(N):
        with timer_heatmap:
            net.router.fidelity_heatmap()
    _result("Fidelity heatmap", timer_heatmap)
    results["fidelity_heatmap_ms"] = timer_heatmap.mean_ms

    # 14c. Topology detection
    timer_topo = BenchTimer("detect_topology")
    for _ in range(N):
        with timer_topo:
            net.router.detect_topology()
    _result("Topology detection", timer_topo)
    results["detect_topology_ms"] = timer_topo.mean_ms

    # 14d. Topology analysis
    timer_topo_analysis = BenchTimer("topology_analysis")
    for _ in range(N):
        with timer_topo_analysis:
            net.router.topology_analysis()
    _result("Topology analysis", timer_topo_analysis)
    results["topology_analysis_ms"] = timer_topo_analysis.mean_ms

    # 14e. Network health score
    timer_health = BenchTimer("network_health")
    for _ in range(N):
        with timer_health:
            net.router.network_health_score()
    _result("Network health score", timer_health)
    results["network_health_ms"] = timer_health.mean_ms

    # 14f. Network summary
    timer_summary = BenchTimer("network_summary")
    for _ in range(N):
        with timer_summary:
            net.router.network_summary()
    _result("Network summary", timer_summary)
    results["network_summary_ms"] = timer_summary.mean_ms

    # 14g. Bottleneck detection
    timer_bottleneck = BenchTimer("bottleneck")
    for _ in range(N):
        with timer_bottleneck:
            net.router.bottleneck_channels()
    _result("Bottleneck detection", timer_bottleneck)
    results["bottleneck_ms"] = timer_bottleneck.mean_ms

    # 14h. Fidelity scan + auto-heal
    timer_scan = BenchTimer("fidelity_scan")
    for _ in range(50):
        with timer_scan:
            net.scan_fidelity(auto_heal=True)
    _result("Fidelity scan + auto-heal", timer_scan)
    results["fidelity_scan_ms"] = timer_scan.mean_ms

    # 14i. Network status
    timer_status = BenchTimer("network_status")
    for _ in range(N):
        with timer_status:
            net.network_status()
    _result("Network status report", timer_status)
    results["network_status_ms"] = timer_status.mean_ms

    # 14j. Full status
    timer_full = BenchTimer("full_status")
    for _ in range(50):
        with timer_full:
            net.status()
    _result("Full status report", timer_full)
    results["full_status_ms"] = timer_full.mean_ms

    # 14k. Snapshot
    timer_snap = BenchTimer("snapshot")
    for _ in range(50):
        with timer_snap:
            net.router.snapshot()
    _result("Full network snapshot", timer_snap)
    results["snapshot_ms"] = timer_snap.mean_ms
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 15: Scaling Analysis
# ═══════════════════════════════════════════════════════════════════

def bench_scaling():
    _header("BENCHMARK 15: Scaling Analysis — Node Count vs. Latency")
    results = {}

    sizes = [5, 10, 20, 40]
    route_times = {}
    sacred_times = {}
    status_times = {}

    for n_nodes in sizes:
        net = _fresh_network()
        nodes = [net.add_node(f"Scale-{i}", role="sovereign") for i in range(n_nodes)]
        # Build a chain + a few cross-links
        for i in range(n_nodes - 1):
            net.connect(nodes[i].node_id, nodes[i + 1].node_id, pairs=4)
        # Cross-links every 3rd node
        for i in range(0, n_nodes - 3, 3):
            net.connect(nodes[i].node_id, nodes[i + 3].node_id, pairs=3)

        # Route timing
        timer_r = BenchTimer(f"route_{n_nodes}")
        for _ in range(50):
            net.router._route_cache.clear()
            with timer_r:
                net.router.find_route(nodes[0].node_id, nodes[-1].node_id)
        route_times[n_nodes] = timer_r.mean_ms

        # Sacred scoring
        timer_s = BenchTimer(f"sacred_{n_nodes}")
        for _ in range(50):
            with timer_s:
                net.router.sacred_scoring_pass()
        sacred_times[n_nodes] = timer_s.mean_ms

        # Status
        timer_st = BenchTimer(f"status_{n_nodes}")
        for _ in range(50):
            with timer_st:
                net.status()
        status_times[n_nodes] = timer_st.mean_ms

    print(f"\n  {'Nodes':<8s} {'Route ms':>10s} {'Sacred ms':>10s} {'Status ms':>10s}")
    print(f"  {'─' * 40}")
    for n in sizes:
        print(f"  {n:<8d} {route_times[n]:>10.3f} {sacred_times[n]:>10.3f} {status_times[n]:>10.3f}")

    results["scaling_route"] = {str(k): round(v, 4) for k, v in route_times.items()}
    results["scaling_sacred"] = {str(k): round(v, 4) for k, v in sacred_times.items()}
    results["scaling_status"] = {str(k): round(v, 4) for k, v in status_times.items()}
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 16: Self-Test Diagnostic Speed
# ═══════════════════════════════════════════════════════════════════

def bench_self_test():
    _header("BENCHMARK 16: Self-Test Diagnostic — Full Probe")
    results = {}

    timer_self = BenchTimer("self_test")
    for _ in range(10):
        net = _fresh_network()
        with timer_self:
            net.self_test()
    _result("Full self-test diagnostic", timer_self)
    results["self_test_ms"] = timer_self.mean_ms
    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("═" * 65)
    print("  L104 QUANTUM NETWORK SPEED BENCHMARK v1.0")
    print(f"  GOD_CODE={GOD_CODE} | PHI={PHI}")
    print(f"  VOID_CONSTANT={VOID_CONSTANT}")
    print("═" * 65)

    t_total_start = time.time()
    all_results = {}
    compare_mode = "--compare" in sys.argv

    benchmarks = [
        ("bell_pair_gen", bench_bell_pair_generation),
        ("channel_creation", bench_channel_creation),
        ("routing_dijkstra", bench_routing_dijkstra),
        ("k_shortest", bench_k_shortest),
        ("entanglement_swap", bench_entanglement_swap),
        ("bb84_qkd", bench_bb84_qkd),
        ("e91_qkd", bench_e91_qkd),
        ("teleportation", bench_teleportation),
        ("multihop_teleport", bench_multihop_teleport),
        ("repeater_chain", bench_repeater_chain),
        ("purification", bench_purification),
        ("decoherence_replenish", bench_decoherence_replenish),
        ("sacred_scoring", bench_sacred_scoring),
        ("network_analytics", bench_network_analytics),
        ("scaling", bench_scaling),
        ("self_test", bench_self_test),
    ]

    for key, fn in benchmarks:
        try:
            all_results[key] = fn()
        except Exception as e:
            print(f"\n  ✗ BENCHMARK FAILED: {key} — {e}")
            import traceback
            traceback.print_exc()
            all_results[key] = {"error": str(e)}

    total_elapsed = time.time() - t_total_start

    # ─── Summary ─────────────────────────────────────────────────
    _header("SUMMARY — Key Metrics")

    metrics = [
        ("Bell pair (8 batch)", all_results.get("bell_pair_gen", {}).get("batch_8_pairs_ms")),
        ("Channel create", all_results.get("channel_creation", {}).get("channel_create_ms")),
        ("Dijkstra 9-hop", all_results.get("routing_dijkstra", {}).get("route_9hop_ms")),
        ("Dijkstra cached", all_results.get("routing_dijkstra", {}).get("route_cached_ms")),
        ("Entanglement swap", all_results.get("entanglement_swap", {}).get("swap_ms")),
        ("BB84 QKD 256-bit", all_results.get("bb84_qkd", {}).get("bb84_256_ms")),
        ("E91 QKD 256-bit", all_results.get("e91_qkd", {}).get("e91_256_ms")),
        ("Score teleport", all_results.get("teleportation", {}).get("teleport_score_ms")),
        ("3-hop teleport", all_results.get("multihop_teleport", {}).get("multihop_3_ms")),
        ("Repeater chain (3 seg)", all_results.get("repeater_chain", {}).get("chain_3seg_ms")),
        ("Purification (3 rounds)", all_results.get("purification", {}).get("purify_3rounds_ms")),
        ("Sacred scoring pass", all_results.get("sacred_scoring", {}).get("sacred_scoring_ms")),
        ("Network health", all_results.get("network_analytics", {}).get("network_health_ms")),
        ("Full status report", all_results.get("network_analytics", {}).get("full_status_ms")),
        ("Self-test diagnostic", all_results.get("self_test", {}).get("self_test_ms")),
    ]

    print(f"\n  {'Operation':<35s} {'Latency':>10s}")
    print(f"  {'─' * 47}")
    for label, val in metrics:
        if val is not None:
            print(f"  {label:<35s} {val:>9.3f} ms")
        else:
            print(f"  {label:<35s}       N/A")

    # ─── Save results ────────────────────────────────────────────
    all_results["_meta"] = {
        "benchmark": "L104 Quantum Network Speed Benchmark",
        "version": "1.0",
        "total_elapsed_sec": round(total_elapsed, 3),
        "god_code": GOD_CODE,
        "phi": PHI,
        "void_constant": VOID_CONSTANT,
    }

    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "_bench_quantum_network_speed_results.json")
    prev_path = report_path.replace("_results.json", "_prev_results.json")

    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {os.path.basename(report_path)}")

    # ─── Regression comparison ───────────────────────────────────
    if compare_mode:
        _header("REGRESSION ANALYSIS")
        if not os.path.exists(prev_path):
            print(f"  No previous baseline found at {os.path.basename(prev_path)}")
            print(f"  Tip: Copy current results to {os.path.basename(prev_path)} as baseline.")
        else:
            with open(prev_path) as f:
                prev = json.load(f)

            regression_count = 0
            improvement_count = 0
            compare_keys = [
                ("bell_pair_gen", "batch_8_pairs_ms", 1.15),
                ("channel_creation", "channel_create_ms", 1.15),
                ("routing_dijkstra", "route_9hop_ms", 1.20),
                ("entanglement_swap", "swap_ms", 1.15),
                ("bb84_qkd", "bb84_256_ms", 1.15),
                ("e91_qkd", "e91_256_ms", 1.15),
                ("teleportation", "teleport_score_ms", 1.15),
                ("multihop_teleport", "multihop_3_ms", 1.20),
                ("purification", "purify_3rounds_ms", 1.15),
                ("sacred_scoring", "sacred_scoring_ms", 1.20),
                ("network_analytics", "full_status_ms", 1.20),
                ("self_test", "self_test_ms", 1.20),
            ]

            for section, key, threshold in compare_keys:
                old_val = prev.get(section, {}).get(key)
                new_val = all_results.get(section, {}).get(key)
                if old_val is None or new_val is None:
                    continue

                if new_val > old_val * threshold:
                    print(f"  ⚠ REGRESSION: {section}.{key}: {old_val:.4f} → {new_val:.4f} "
                          f"({(new_val / old_val - 1) * 100:+.1f}%)")
                    regression_count += 1
                elif new_val < old_val * 0.95:
                    print(f"  ✓ IMPROVED:   {section}.{key}: {old_val:.4f} → {new_val:.4f} "
                          f"({(new_val / old_val - 1) * 100:+.1f}%)")
                    improvement_count += 1

            if regression_count == 0:
                print(f"\n  ✓ No regressions detected ({improvement_count} improvements)")
            else:
                print(f"\n  ⚠ {regression_count} regression(s), {improvement_count} improvement(s)")

    print()
    print("═" * 65)
    print(f"  QUANTUM NETWORK SPEED BENCHMARK v1.0 COMPLETE — {total_elapsed:.2f}s")
    print(f"  INVARIANT: {GOD_CODE} | PILOT: LONDEL")
    print("═" * 65)

    return all_results


if __name__ == "__main__":
    main()
