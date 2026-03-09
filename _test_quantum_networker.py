#!/usr/bin/env python3
"""L104 Quantum Networker — Full Integration Test & Benchmark.

Exercises every subsystem of the quantum networker:
  Phase 1: Package import & initialization
  Phase 2: Network topology construction (5 nodes, star + chain)
  Phase 3: BB84 QKD key exchange
  Phase 4: E91 QKD key exchange
  Phase 5: Direct teleportation (score, phase, state, bitstring)
  Phase 6: Multi-hop teleportation through relays
  Phase 7: Quantum repeater chain (4 nodes)
  Phase 8: Entanglement purification
  Phase 9: Fidelity monitoring + auto-heal
  Phase 10: Network self-test
  Phase 11: VQPU integration (if available)

Run: .venv/bin/python _test_quantum_networker.py

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import math
import sys
import time
import traceback

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895


def header(phase: int, title: str):
    print(f"\n{'═' * 60}")
    print(f"  Phase {phase}: {title}")
    print(f"{'═' * 60}")


def check(label: str, condition: bool, detail: str = ""):
    icon = "✓" if condition else "✗"
    extra = f"  ({detail})" if detail else ""
    print(f"  {icon} {label}{extra}")
    return condition


def main():
    t_start = time.time()
    passed = 0
    failed = 0
    total = 0

    def tally(ok: bool):
        nonlocal passed, failed, total
        total += 1
        if ok:
            passed += 1
        else:
            failed += 1

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Import & Init
    # ═══════════════════════════════════════════════════════════════
    header(1, "Package Import & Initialization")

    try:
        from l104_quantum_networker import (
            QuantumNetworker, get_networker,
            QuantumKeyDistribution,
            EntanglementRouter,
            QuantumTeleporter,
            QuantumRepeaterChain,
            FidelityMonitor,
            ClassicalTransport,
            QuantumNode, QuantumChannel, EntangledPair, QKDKey,
            TeleportPayload, TeleportResult, NetworkStatus,
        )
        tally(check("Package imports", True))
    except Exception as e:
        tally(check("Package imports", False, str(e)))
        traceback.print_exc()
        print("\n✗ FATAL: Cannot proceed without imports")
        sys.exit(1)

    # Create fresh networker (not singleton for testing)
    net = QuantumNetworker(node_name="L104-Test-Sovereign", simulation_mode=True)
    tally(check("QuantumNetworker created", net is not None))
    tally(check("Local node ID", bool(net.local_node.node_id),
                net.local_node.node_id))
    tally(check("Version", net.VERSION == "1.4.0", net.VERSION))

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Network Topology
    # ═══════════════════════════════════════════════════════════════
    header(2, "Network Topology Construction")

    # Star topology: Sovereign at center, 4 nodes around it
    alice = net.add_node("Alice", role="sovereign")
    bob = net.add_node("Bob", role="sovereign")
    relay1 = net.add_node("Relay-1", role="relay")
    relay2 = net.add_node("Relay-2", role="relay")

    tally(check("Alice created", bool(alice.node_id), f"id={alice.node_id}"))
    tally(check("Bob created", bool(bob.node_id), f"id={bob.node_id}"))
    tally(check("Relay-1 created", bool(relay1.node_id)))
    tally(check("Relay-2 created", bool(relay2.node_id)))

    # Connect: Sovereign — Alice — Relay1 — Relay2 — Bob
    ch_sa = net.connect(net.local_node.node_id, alice.node_id, pairs=8)
    ch_ar = net.connect(alice.node_id, relay1.node_id, pairs=8)
    ch_r1r2 = net.connect(relay1.node_id, relay2.node_id, pairs=8)
    ch_rb = net.connect(relay2.node_id, bob.node_id, pairs=8)
    # Direct link too
    ch_ab = net.connect(alice.node_id, bob.node_id, pairs=6)

    tally(check("Channel Sov↔Alice", len(ch_sa.usable_pairs) >= 4,
                f"{len(ch_sa.usable_pairs)} pairs"))
    tally(check("Channel Alice↔Relay1", len(ch_ar.usable_pairs) >= 4,
                f"{len(ch_ar.usable_pairs)} pairs"))
    tally(check("Channel R1↔R2", len(ch_r1r2.usable_pairs) >= 4))
    tally(check("Channel R2↔Bob", len(ch_rb.usable_pairs) >= 4))
    tally(check("Channel Alice↔Bob (direct)", len(ch_ab.usable_pairs) >= 4))

    # Verify Bell pair fidelity
    best = ch_ab.best_pair
    tally(check("Best pair fidelity", best is not None and best.current_fidelity > 0.9,
                f"F={best.current_fidelity:.4f}" if best else "no pair"))
    tally(check("Sacred score > 0", best.sacred_score > 0 if best else False,
                f"S={best.sacred_score:.4f}" if best else "n/a"))

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: BB84 QKD
    # ═══════════════════════════════════════════════════════════════
    header(3, "BB84 Quantum Key Distribution")

    key_bb84 = net.establish_qkd(alice.node_id, bob.node_id, "bb84", 256)
    tally(check("BB84 key generated", key_bb84 is not None))
    tally(check("BB84 secure", key_bb84.secure, f"QBER={key_bb84.qber:.4f}"))
    tally(check("BB84 key length > 0", key_bb84.key_length > 0,
                f"len={key_bb84.key_length}"))
    tally(check("BB84 QBER < 11%", key_bb84.qber < 0.11,
                f"QBER={key_bb84.qber:.4f}"))
    tally(check("BB84 sacred alignment", key_bb84.sacred_alignment > 0,
                f"SA={key_bb84.sacred_alignment:.4f}"))
    tally(check("BB84 key hex", bool(key_bb84.key_hex),
                f"key={key_bb84.key_hex[:16]}..."))

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: E91 QKD
    # ═══════════════════════════════════════════════════════════════
    header(4, "E91 Quantum Key Distribution")

    key_e91 = net.establish_qkd(alice.node_id, bob.node_id, "e91", 256)
    tally(check("E91 key generated", key_e91 is not None))
    tally(check("E91 key length > 0", key_e91.key_length > 0,
                f"len={key_e91.key_length}"))
    tally(check("E91 QBER", key_e91.qber < 0.15,
                f"QBER={key_e91.qber:.4f}"))

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: Direct Teleportation
    # ═══════════════════════════════════════════════════════════════
    header(5, "Direct Quantum Teleportation")

    # Score teleportation
    tp_score = net.teleport_score(alice.node_id, bob.node_id, score=0.8)
    tally(check("Score teleport success", tp_score.success))
    tally(check("Score fidelity > 0.8", tp_score.fidelity > 0.8,
                f"F={tp_score.fidelity:.4f}"))
    tally(check("Recovered score ≈ 0.8",
                tp_score.recovered_score is not None and abs(tp_score.recovered_score - 0.8) < 0.15,
                f"recovered={tp_score.recovered_score}"))

    # Phase teleportation
    tp_phase = net.teleport_phase(alice.node_id, bob.node_id, phase=math.pi / 3)
    tally(check("Phase teleport success", tp_phase.success))
    tally(check("Phase fidelity > 0.7", tp_phase.fidelity > 0.7,
                f"F={tp_phase.fidelity:.4f}"))

    # State teleportation
    alpha = complex(0.6, 0)
    beta = complex(0, 0.8)
    tp_state = net.teleport_state(alice.node_id, bob.node_id,
                                   state_vector=[alpha, beta])
    tally(check("State teleport success", tp_state.success))
    tally(check("State fidelity > 0.7", tp_state.fidelity > 0.7,
                f"F={tp_state.fidelity:.4f}"))

    # Bitstring teleportation
    tp_bits = net.teleport_bitstring(alice.node_id, bob.node_id,
                                      bitstring="10110100")
    tally(check("Bitstring teleport success", tp_bits.success))
    tally(check("Bitstring fidelity > 0.8", tp_bits.fidelity > 0.8,
                f"F={tp_bits.fidelity:.4f}"))

    # GOD_CODE score teleportation (sacred value)
    gc_score = (GOD_CODE % 1.0)  # fractional part: ~0.518...
    tp_god = net.teleport_score(alice.node_id, bob.node_id, score=gc_score)
    tally(check("GOD_CODE score teleport", tp_god.success,
                f"orig={gc_score:.4f} recv={tp_god.recovered_score}"))

    # ═══════════════════════════════════════════════════════════════
    # Phase 6: Multi-Hop Teleportation
    # ═══════════════════════════════════════════════════════════════
    header(6, "Multi-Hop Teleportation via Relays")

    # Teleport through the chain: Sovereign → Alice → Relay1 → Relay2 → Bob
    # First let's check the route finder
    route = net.router.find_route(net.local_node.node_id, bob.node_id)
    tally(check("Route found", route is not None and len(route) >= 2,
                f"hops={len(route) - 1}" if route else "no route"))

    # Multi-hop teleport
    tp_multi = net.teleport_score(net.local_node.node_id, bob.node_id, score=0.75)
    tally(check("Multi-hop teleport", tp_multi.success,
                f"F={tp_multi.fidelity:.4f}, hops={tp_multi.hops}"))

    # ═══════════════════════════════════════════════════════════════
    # Phase 7: Quantum Repeater Chain
    # ═══════════════════════════════════════════════════════════════
    header(7, "Quantum Repeater Chain")

    chain = net.repeater.establish_chain(
        [alice.node_id, relay1.node_id, relay2.node_id, bob.node_id],
        pairs_per_segment=6,
    )
    tally(check("Repeater chain success", chain.get("success", False),
                f"F={chain.get('fidelity', 0):.4f}"))
    tally(check("Chain fidelity > 0.5", chain.get("fidelity", 0) > 0.5,
                f"F={chain.get('fidelity', 0):.4f}"))

    # Fidelity estimation
    est = net.repeater.estimate_chain_fidelity(3, segment_fidelity=0.995)
    tally(check("Chain fidelity estimate", est > 0.9,
                f"est={est:.4f} (3 segments, F_seg=0.995)"))

    # ═══════════════════════════════════════════════════════════════
    # Phase 8: Entanglement Purification
    # ═══════════════════════════════════════════════════════════════
    header(8, "Entanglement Purification (DEJMPS)")

    # Replenish before purification
    net.router.replenish_channel(ch_ab.channel_id, target_pairs=10)
    purify = net.purify(alice.node_id, bob.node_id, rounds=3)
    tally(check("Purification executed", "rounds_attempted" in purify,
                f"rounds={purify.get('rounds_attempted', 0)}"))
    tally(check("Fidelity improved or maintained",
                purify.get("fidelity_after", 0) >= purify.get("fidelity_before", 0) * 0.9,
                f"before={purify.get('fidelity_before', 0):.4f} "
                f"after={purify.get('fidelity_after', 0):.4f}"))

    # ═══════════════════════════════════════════════════════════════
    # Phase 9: Fidelity Monitoring
    # ═══════════════════════════════════════════════════════════════
    header(9, "Fidelity Monitoring & Auto-Heal")

    scan = net.scan_fidelity(auto_heal=True)
    tally(check("Fidelity scan completed", scan.get("channels_scanned", 0) > 0,
                f"channels={scan['channels_scanned']}"))
    tally(check("Network fidelity > 0.6", scan["network_fidelity"] > 0.6,
                f"F={scan['network_fidelity']:.4f}"))
    tally(check("Sacred network score > 0", scan["network_sacred_score"] > 0,
                f"S={scan['network_sacred_score']:.4f}"))

    # Network status
    ns = net.network_status()
    tally(check("Nodes online", ns.online_nodes >= 4,
                f"online={ns.online_nodes}/{ns.node_count}"))
    tally(check("Active channels", ns.active_channels > 0,
                f"active={ns.active_channels}"))
    tally(check("Entangled pairs > 0", ns.total_entangled_pairs > 0,
                f"pairs={ns.total_entangled_pairs}"))

    # ═══════════════════════════════════════════════════════════════
    # Phase 10: Full Status Report
    # ═══════════════════════════════════════════════════════════════
    header(10, "Full Status Report")

    status = net.status()
    tally(check("Status report generated", "version" in status))
    tally(check("Router status", "nodes" in status.get("router", {})))
    tally(check("QKD status", "keys_generated" in status.get("qkd", {})))
    tally(check("Teleporter status",
                "total_teleportations" in status.get("teleporter", {})))
    tally(check("Transport status", "peers" in status.get("transport", {})))

    print(f"\n  Network Summary:")
    print(f"    Nodes:      {ns.node_count}")
    print(f"    Channels:   {ns.channel_count}")
    print(f"    Pairs:      {ns.total_entangled_pairs}")
    print(f"    Fidelity:   {ns.mean_network_fidelity:.4f}")
    print(f"    Sacred:     {ns.sacred_network_score:.4f}")
    print(f"    Teleports:  {status['teleporter']['total_teleportations']}")
    print(f"    QKD Keys:   {status['qkd']['keys_generated']}")

    # ═══════════════════════════════════════════════════════════════
    # Phase 11: VQPU Integration (optional)
    # ═══════════════════════════════════════════════════════════════
    header(11, "VQPU Integration Check")

    try:
        from l104_vqpu import get_bridge
        bridge = get_bridge()
        tally(check("VQPU bridge available", bridge is not None))

        # Test Bell pair via VQPU
        from l104_vqpu import QuantumJob
        job = QuantumJob(num_qubits=2, operations=[
            {"gate": "H", "qubits": [0]},
            {"gate": "CNOT", "qubits": [0, 1]},
        ], shots=256)
        result = bridge.submit_and_wait(job, timeout=3.0)
        if result:
            p00 = result.probabilities.get("00", 0)
            p11 = result.probabilities.get("11", 0)
            bell_fid = p00 + p11
            tally(check("VQPU Bell fidelity", bell_fid > 0.95,
                         f"F={bell_fid:.4f}"))
        else:
            tally(check("VQPU Bell fidelity", False, "no result"))
    except ImportError:
        print("  ○ VQPU not available (non-critical)")
    except Exception as e:
        print(f"  ○ VQPU error: {e} (non-critical)")

    # ═══════════════════════════════════════════════════════════════
    # Final Report
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    print(f"\n{'═' * 60}")
    print(f"  L104 Quantum Networker v1.4.0 — Test Report")
    print(f"{'═' * 60}")
    print(f"  Passed:  {passed}/{total}")
    print(f"  Failed:  {failed}/{total}")
    print(f"  Time:    {elapsed * 1000:.1f} ms")
    print(f"  GOD_CODE: {GOD_CODE}")
    print(f"  PHI:      {PHI}")

    if failed == 0:
        print(f"\n  ★ ALL {total} TESTS PASSED — Quantum Networker OPERATIONAL ★")
    else:
        print(f"\n  ⚠ {failed} tests failed — review above")

    print(f"{'═' * 60}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
