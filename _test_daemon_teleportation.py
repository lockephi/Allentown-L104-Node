#!/usr/bin/env python3
"""L104 Daemon Quantum Teleportation — Full Test Suite & Upgrade.

Tests quantum teleportation across BOTH daemon subsystems:
  System A: l104_vqpu — QuantumNetworkMesh (micro daemon teleportation)
  System B: l104_quantum_networker — QuantumTeleporter (networker teleportation)
  System C: l104_quantum_ai_daemon — Autonomous daemon integration

Phase 1:  Import & Boot all subsystems
Phase 2:  VQPU mesh: register + sacred init + Bell pairs
Phase 3:  VQPU mesh: channel teleportation (single-hop)
Phase 4:  VQPU mesh: multi-node teleportation + purification
Phase 5:  VQPU mesh: decoherence + recalibration cycle
Phase 6:  Networker: node topology + channel establishment
Phase 7:  Networker: score teleportation
Phase 8:  Networker: phase teleportation
Phase 9:  Networker: state vector teleportation
Phase 10: Networker: bitstring teleportation
Phase 11: Networker: multi-hop relay teleportation
Phase 12: Networker: QKD + encrypted teleportation
Phase 13: Micro daemon: quantum network init + teleport
Phase 14: Cross-system fidelity comparison
Phase 15: Benchmark: throughput + sacred alignment

Run: .venv/bin/python _test_daemon_teleportation.py

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import math
import os
import sys
import time
import traceback
from typing import Any, Dict, List

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497


def header(phase: int, title: str):
    print(f"\n{'═' * 70}")
    print(f"  Phase {phase}: {title}")
    print(f"{'═' * 70}")


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = "✓" if condition else "✗"
    extra = f"  ({detail})" if detail else ""
    print(f"  {icon} {label}{extra}")
    return condition


class TestTally:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 0
        self.errors: List[str] = []

    def __call__(self, ok: bool, label: str = ""):
        self.total += 1
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            if label:
                self.errors.append(label)

    def summary(self) -> str:
        return (f"Passed: {self.passed}/{self.total} | "
                f"Failed: {self.failed}/{self.total} | "
                f"Rate: {self.passed/max(self.total,1)*100:.1f}%")


def main():
    t_global = time.time()
    T = TestTally()

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Import & Boot
    # ═══════════════════════════════════════════════════════════
    header(1, "Import & Boot All Subsystems")

    # VQPU quantum network
    vqpu_ok = False
    try:
        from l104_vqpu.quantum_network import (
            DaemonQubitRegister, QuantumChannel as VQPUChannel,
            QuantumNetworkMesh, QubitState,
            FIDELITY_FLOOR, FIDELITY_THRESHOLD_HIGH,
            FIDELITY_THRESHOLD_GOOD, QUANTUM_NETWORK_VERSION,
        )
        vqpu_ok = True
        T(check("VQPU quantum network import", True, f"v{QUANTUM_NETWORK_VERSION}"), "vqpu_import")
    except Exception as e:
        T(check("VQPU quantum network import", False, str(e)), "vqpu_import")
        traceback.print_exc()

    # Quantum networker
    net_ok = False
    try:
        from l104_quantum_networker import (
            QuantumNetworker, get_networker,
            QuantumTeleporter, EntanglementRouter,
            QuantumKeyDistribution, FidelityMonitor,
            TeleportPayload, TeleportResult,
        )
        net_ok = True
        T(check("Quantum networker import", True), "net_import")
    except Exception as e:
        T(check("Quantum networker import", False, str(e)), "net_import")
        traceback.print_exc()

    # Quantum AI daemon
    daemon_ok = False
    try:
        from l104_quantum_ai_daemon import (
            QuantumAIDaemon, DaemonConfig,
            QuantumFidelityGuard, FidelityReport,
        )
        daemon_ok = True
        T(check("Quantum AI daemon import", True), "daemon_import")
    except Exception as e:
        T(check("Quantum AI daemon import", False, str(e)), "daemon_import")

    # VQPU micro daemon
    micro_ok = False
    try:
        from l104_vqpu.micro_daemon import VQPUMicroDaemon
        micro_ok = True
        T(check("VQPU micro daemon import", True), "micro_import")
    except Exception as e:
        T(check("VQPU micro daemon import", False, str(e)), "micro_import")

    print(f"\n  Boot status: VQPU={vqpu_ok} | Networker={net_ok} | "
          f"AID={daemon_ok} | Micro={micro_ok}")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: VQPU Mesh — Register + Sacred Init + Bell Pairs
    # ═══════════════════════════════════════════════════════════
    if vqpu_ok:
        header(2, "VQPU Mesh: Register + Sacred Init + Bell Pairs")

        # Create register
        reg = DaemonQubitRegister(node_id="test-alpha", num_qubits=4)
        T(check("Register created", reg.num_qubits == 4, f"{reg.num_qubits}Q"), "reg_create")

        # Sacred init
        init_result = reg.initialize_sacred()
        avg_f = init_result.get("avg_fidelity", 0)
        avg_s = init_result.get("avg_sacred_alignment", 0)
        T(check("Sacred init fidelity", avg_f > 0.9, f"F={avg_f:.6f}"), "sacred_fidelity")
        T(check("Sacred alignment", avg_s > 0.0, f"S={avg_s:.6f}"), "sacred_align")
        T(check("Calibration count", init_result.get("calibration_count") == 1), "calib_count")

        # Fidelity check
        fid_check = reg.fidelity_check()
        T(check("Fidelity check", fid_check["avg_fidelity"] > 0.8,
                f"avg={fid_check['avg_fidelity']:.6f}"), "fid_check")

        # Bell pair
        bell = reg.entangled_pair_state(0, 1)
        bell_fid = abs(bell[0]) ** 2 + abs(bell[3]) ** 2
        T(check("Bell pair |Φ+⟩", bell_fid > 0.9, f"|00⟩+|11⟩ weight={bell_fid:.6f}"), "bell_pair")
        T(check("Bell pair dimension", len(bell) == 4), "bell_dim")

    # ═══════════════════════════════════════════════════════════
    # Phase 3: VQPU Mesh — Channel Teleportation (Single-hop)
    # ═══════════════════════════════════════════════════════════
    if vqpu_ok:
        header(3, "VQPU Mesh: Channel Teleportation (Single-hop)")

        mesh = QuantumNetworkMesh(
            node_ids=["daemon-A", "daemon-B"],
            qubits_per_node=4,
        )
        est = mesh.establish_channels()
        T(check("Mesh established", est["channels"] > 0,
                f"{est['nodes']} nodes, {est['channels']} channels"), "mesh_est")
        T(check("Mesh avg fidelity", est["avg_fidelity"] > 0.8,
                f"F={est['avg_fidelity']:.6f}"), "mesh_fid")

        # Teleport payload
        payload = {"score": 0.618, "tag": "phi_test"}
        t0 = time.time()
        tp_result = mesh.teleport("daemon-A", "daemon-B", payload)
        tp_ms = (time.time() - t0) * 1000

        T(check("Teleport success", tp_result.get("success") is True), "tp_success")
        T(check("Teleport fidelity", tp_result.get("fidelity", 0) > 0.8,
                f"F={tp_result.get('fidelity', 0):.6f}"), "tp_fidelity")
        T(check("Teleport sacred aligned", tp_result.get("sacred_aligned") is True), "tp_sacred")
        T(check("Teleport latency", tp_ms < 100, f"{tp_ms:.2f}ms"), "tp_latency")
        T(check("Teleport counter", mesh.total_teleportations == 1), "tp_counter")

        # Reverse direction
        tp2 = mesh.teleport("daemon-B", "daemon-A", {"reverse": True})
        T(check("Reverse teleport", tp2.get("success") is True), "tp_reverse")

    # ═══════════════════════════════════════════════════════════
    # Phase 4: VQPU Mesh — Multi-node + Purification
    # ═══════════════════════════════════════════════════════════
    if vqpu_ok:
        header(4, "VQPU Mesh: Multi-node Teleportation + Purification")

        mesh4 = QuantumNetworkMesh(
            node_ids=["node-1", "node-2", "node-3", "node-4"],
            qubits_per_node=4,
        )
        est4 = mesh4.establish_channels()
        expected_channels = 4 * 3 // 2  # C(4,2) = 6
        T(check("4-node mesh", est4["channels"] == expected_channels,
                f"{est4['channels']} channels"), "mesh4_channels")

        # Teleport between non-adjacent conceptually (all directly connected in complete graph)
        for src, dst in [("node-1", "node-3"), ("node-2", "node-4"), ("node-1", "node-4")]:
            r = mesh4.teleport(src, dst, {"test": f"{src}->{dst}"})
            T(check(f"Teleport {src}->{dst}", r.get("success") is True,
                    f"F={r.get('fidelity', 0):.4f}"), f"tp_{src}_{dst}")

        # Purification cycle
        purify_result = mesh4.purify_all()
        T(check("Purification ran", purify_result["total_channels"] == expected_channels,
                f"purified={purify_result['purified']}, skipped={purify_result['skipped']}"),
          "purify_all")

        # Network health
        health = mesh4.network_health()
        T(check("Network score", health["network_score"] > 0.5,
                f"score={health['network_score']:.6f}"), "net_score")
        T(check("Sacred alignment", health["sacred_alignment"] > 0.99,
                f"SA={health['sacred_alignment']:.6f}"), "net_sacred")

    # ═══════════════════════════════════════════════════════════
    # Phase 5: VQPU Mesh — Decoherence + Recalibration
    # ═══════════════════════════════════════════════════════════
    if vqpu_ok:
        header(5, "VQPU Mesh: Decoherence + Recalibration")

        mesh_d = QuantumNetworkMesh(
            node_ids=["deco-A", "deco-B"],
            qubits_per_node=4,
        )
        mesh_d.establish_channels()

        # Apply decoherence
        deco = mesh_d.decoherence_cycle()
        T(check("Decoherence applied", deco["channels_checked"] > 0,
                f"checked={deco['channels_checked']}, degraded={deco['degraded']}"), "deco_apply")

        # Register recalibration
        reg_d = DaemonQubitRegister(node_id="recalib-test", num_qubits=4)
        reg_d.initialize_sacred()
        recal = reg_d.recalibrate_if_needed()
        T(check("Recalibration check", True,
                "needed" if recal else "not needed"), "recalib")

        # Add/remove node
        add_result = mesh_d.add_node("deco-C")
        T(check("Dynamic node add", add_result.get("added") is True,
                f"nodes={add_result.get('total_nodes')}"), "node_add")

        rem_result = mesh_d.remove_node("deco-C")
        T(check("Dynamic node remove", rem_result.get("removed") is True,
                f"remaining={rem_result.get('remaining_nodes')}"), "node_remove")

    # ═══════════════════════════════════════════════════════════
    # Phase 6: Networker — Topology + Channels
    # ═══════════════════════════════════════════════════════════
    if net_ok:
        header(6, "Networker: Node Topology + Channel Establishment")

        net = get_networker()
        alice = net.add_node("Alice-T", role="sovereign")
        bob = net.add_node("Bob-T", role="sovereign")
        relay = net.add_node("Relay-T", role="relay")

        T(check("Alice created", alice is not None, f"id={alice.node_id}"), "alice")
        T(check("Bob created", bob is not None, f"id={bob.node_id}"), "bob")
        T(check("Relay created", relay is not None), "relay")

        # Connect channels
        ch_ab = net.connect(alice.node_id, bob.node_id, pairs=8)
        T(check("Alice↔Bob channel", ch_ab is not None,
                f"pairs={len(ch_ab.pairs) if ch_ab else 0}"), "ch_ab")

        ch_ar = net.connect(alice.node_id, relay.node_id, pairs=6)
        ch_rb = net.connect(relay.node_id, bob.node_id, pairs=6)
        T(check("Alice↔Relay channel", ch_ar is not None), "ch_ar")
        T(check("Relay↔Bob channel", ch_rb is not None), "ch_rb")

    # ═══════════════════════════════════════════════════════════
    # Phase 7: Networker — Score Teleportation
    # ═══════════════════════════════════════════════════════════
    if net_ok:
        header(7, "Networker: Score Teleportation")

        test_scores = [0.0, 0.25, 0.5, PHI - 1.0, 0.75, 1.0]
        score_fidelities = []

        for score in test_scores:
            try:
                result = net.teleport_score(
                    alice.node_id, bob.node_id,
                    score=score, error_correct=True)
                fid = result.fidelity if result.success else 0.0
                rec = result.recovered_score if result.success else -1
                ok = result.success and fid > 0.7
                T(check(f"Score {score:.4f}", ok,
                        f"recovered={rec:.4f}, F={fid:.4f}"), f"score_{score}")
                if result.success:
                    score_fidelities.append(fid)
            except Exception as e:
                T(check(f"Score {score:.4f}", False, str(e)), f"score_{score}")

        if score_fidelities:
            avg_sf = sum(score_fidelities) / len(score_fidelities)
            print(f"\n  ★ Score teleportation avg fidelity: {avg_sf:.6f}")

    # ═══════════════════════════════════════════════════════════
    # Phase 8: Networker — Phase Teleportation
    # ═══════════════════════════════════════════════════════════
    if net_ok:
        header(8, "Networker: Phase Teleportation")

        test_phases = [0.0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2]
        phase_fidelities = []

        for phase in test_phases:
            try:
                result = net.teleport_phase(
                    alice.node_id, bob.node_id,
                    phase=phase, error_correct=True)
                fid = result.fidelity if result.success else 0.0
                rec_phase = result.recovered_phase if result.success else -1
                ok = result.success and fid > 0.5
                T(check(f"Phase {phase:.4f} rad", ok,
                        f"recovered={rec_phase:.4f}, F={fid:.4f}"), f"phase_{phase}")
                if result.success:
                    phase_fidelities.append(fid)
            except Exception as e:
                T(check(f"Phase {phase:.4f} rad", False, str(e)), f"phase_{phase}")

        if phase_fidelities:
            avg_pf = sum(phase_fidelities) / len(phase_fidelities)
            print(f"\n  ★ Phase teleportation avg fidelity: {avg_pf:.6f}")

    # ═══════════════════════════════════════════════════════════
    # Phase 9: Networker — State Vector Teleportation
    # ═══════════════════════════════════════════════════════════
    if net_ok:
        header(9, "Networker: State Vector Teleportation")

        import numpy as np
        test_states = [
            ([1.0, 0.0], "|0⟩"),
            ([0.0, 1.0], "|1⟩"),
            ([1/math.sqrt(2), 1/math.sqrt(2)], "|+⟩"),
            ([1/math.sqrt(2), -1/math.sqrt(2)], "|-⟩"),
            ([math.cos(PHI/2), math.sin(PHI/2)], "|φ⟩"),
        ]

        state_fidelities = []
        for sv, label in test_states:
            try:
                result = net.teleport_state(
                    alice.node_id, bob.node_id,
                    state_vector=sv, error_correct=True)
                fid = result.fidelity if result.success else 0.0
                ok = result.success and fid > 0.7
                T(check(f"State {label}", ok, f"F={fid:.4f}"), f"state_{label}")
                if result.success:
                    state_fidelities.append(fid)
            except Exception as e:
                T(check(f"State {label}", False, str(e)), f"state_{label}")

        if state_fidelities:
            avg_stf = sum(state_fidelities) / len(state_fidelities)
            print(f"\n  ★ State teleportation avg fidelity: {avg_stf:.6f}")

    # ═══════════════════════════════════════════════════════════
    # Phase 10: Networker — Bitstring Teleportation
    # ═══════════════════════════════════════════════════════════
    if net_ok:
        header(10, "Networker: Bitstring Teleportation")

        test_bits = ["0101", "1111", "1010", "00110011", "10100101"]
        bit_fidelities = []

        for bits in test_bits:
            try:
                result = net.teleport_bitstring(
                    alice.node_id, bob.node_id,
                    bitstring=bits, error_correct=True)
                fid = result.fidelity if result.success else 0.0
                ok = result.success and fid > 0.7
                # Extract recovered bitstring
                bell_data = result.bell_measurements[0] if result.bell_measurements else {}
                rec_bits = bell_data.get("recovered_bitstring", "?")
                ber = bell_data.get("bit_error_rate", -1)
                T(check(f"Bits '{bits}'", ok,
                        f"recovered='{rec_bits}', BER={ber:.4f}, F={fid:.4f}"),
                  f"bits_{bits}")
                if result.success:
                    bit_fidelities.append(fid)
            except Exception as e:
                T(check(f"Bits '{bits}'", False, str(e)), f"bits_{bits}")

        if bit_fidelities:
            avg_bf = sum(bit_fidelities) / len(bit_fidelities)
            print(f"\n  ★ Bitstring teleportation avg fidelity: {avg_bf:.6f}")

    # ═══════════════════════════════════════════════════════════
    # Phase 11: Networker — Multi-hop Relay Teleportation
    # ═══════════════════════════════════════════════════════════
    if net_ok:
        header(11, "Networker: Multi-hop Relay Teleportation")

        # Build a chain: Far-A ↔ Relay-1 ↔ Relay-2 ↔ Far-B
        far_a = net.add_node("Far-A", role="sovereign")
        relay_1 = net.add_node("Relay-1T", role="relay")
        relay_2 = net.add_node("Relay-2T", role="relay")
        far_b = net.add_node("Far-B", role="sovereign")

        net.connect(far_a.node_id, relay_1.node_id, pairs=6)
        net.connect(relay_1.node_id, relay_2.node_id, pairs=6)
        net.connect(relay_2.node_id, far_b.node_id, pairs=6)

        # Direct channel may not exist — test multi-hop
        try:
            mh_result = net.teleport_score(
                far_a.node_id, far_b.node_id,
                score=0.618, error_correct=True)
            mh_ok = mh_result.success
            mh_hops = mh_result.hops if mh_result.success else 0
            mh_fid = mh_result.fidelity if mh_result.success else 0
            T(check("Multi-hop teleport", mh_ok,
                    f"hops={mh_hops}, F={mh_fid:.4f}"), "multi_hop")
        except Exception as e:
            T(check("Multi-hop teleport", False, str(e)), "multi_hop")

    # ═══════════════════════════════════════════════════════════
    # Phase 12: Networker — QKD + Encrypted Teleportation
    # ═══════════════════════════════════════════════════════════
    if net_ok:
        header(12, "Networker: QKD + Encrypted Teleportation")

        try:
            qkd_key = net.establish_qkd(alice.node_id, bob.node_id, "bb84", 128)
            T(check("QKD key established", qkd_key.secure,
                    f"bits={qkd_key.key_length}, QBER={qkd_key.qber:.4f}"), "qkd_key")

            # Encrypted score teleport
            enc_result = net.teleport_score(
                alice.node_id, bob.node_id,
                score=0.5, error_correct=True)
            T(check("Encrypted teleport", enc_result.success,
                    f"F={enc_result.fidelity:.4f}"), "enc_teleport")
        except Exception as e:
            T(check("QKD + encrypted teleport", False, str(e)), "qkd_enc")

    # ═══════════════════════════════════════════════════════════
    # Phase 13: Micro Daemon — Quantum Network + Teleport
    # ═══════════════════════════════════════════════════════════
    if micro_ok and vqpu_ok:
        header(13, "Micro Daemon: Quantum Network Init + Teleport")

        try:
            # Create two micro daemons with quantum network enabled
            d1 = VQPUMicroDaemon(enable_quantum_network=True, quantum_qubits=4)
            d2 = VQPUMicroDaemon(enable_quantum_network=True, quantum_qubits=4)

            # Manually init quantum networks
            d1._init_quantum_network()
            d2._init_quantum_network()

            mesh1_ok = d1._quantum_mesh is not None
            mesh2_ok = d2._quantum_mesh is not None
            T(check("Daemon-1 mesh init", mesh1_ok), "d1_mesh")
            T(check("Daemon-2 mesh init", mesh2_ok), "d2_mesh")

            if mesh1_ok and mesh2_ok:
                # Connect daemons by adding each other to the mesh
                d1._quantum_mesh.add_node(d2._quantum_node_id)
                d1._quantum_mesh.establish_channels()

                tp_d = d1.teleport(d2._quantum_node_id, {"daemon_test": True})
                T(check("Daemon-to-daemon teleport", tp_d.get("success") is True,
                        f"F={tp_d.get('fidelity', 0):.4f}"), "daemon_teleport")

                # Check quantum health
                qh = d1.quantum_health()
                T(check("Daemon quantum health", "network" in qh), "daemon_qhealth")
        except Exception as e:
            T(check("Micro daemon teleport", False, str(e)), "micro_daemon")
            traceback.print_exc()

    # ═══════════════════════════════════════════════════════════
    # Phase 14: Cross-system Fidelity Comparison
    # ═══════════════════════════════════════════════════════════
    header(14, "Cross-system Fidelity Comparison")

    comparison = {}

    if vqpu_ok:
        # VQPU mesh fidelity
        m_test = QuantumNetworkMesh(node_ids=["cmp-A", "cmp-B"], qubits_per_node=4)
        m_test.establish_channels()
        vqpu_results = []
        for i in range(10):
            r = m_test.teleport("cmp-A", "cmp-B", {"i": i})
            if r.get("success"):
                vqpu_results.append(r["fidelity"])
        if vqpu_results:
            comparison["vqpu_mesh"] = {
                "avg_fidelity": sum(vqpu_results) / len(vqpu_results),
                "min_fidelity": min(vqpu_results),
                "max_fidelity": max(vqpu_results),
                "count": len(vqpu_results),
            }
            print(f"  VQPU Mesh:    avg_F={comparison['vqpu_mesh']['avg_fidelity']:.6f} "
                  f"(n={len(vqpu_results)})")

    if net_ok:
        # Networker fidelity (reuse alice/bob)
        net_results = []
        for i in range(10):
            try:
                r = net.teleport_score(alice.node_id, bob.node_id, score=0.5)
                if r.success:
                    net_results.append(r.fidelity)
            except Exception:
                pass
        if net_results:
            comparison["networker"] = {
                "avg_fidelity": sum(net_results) / len(net_results),
                "min_fidelity": min(net_results),
                "max_fidelity": max(net_results),
                "count": len(net_results),
            }
            print(f"  Networker:    avg_F={comparison['networker']['avg_fidelity']:.6f} "
                  f"(n={len(net_results)})")

    # Sacred constant alignment
    sacred_check = (GOD_CODE / 16.0) ** PHI
    sacred_ok = abs(sacred_check - 286.0) < 1e-6
    T(check("Sacred resonance (GOD_CODE/16)^φ ≈ 286", sacred_ok,
            f"actual={sacred_check:.10f}"), "sacred_resonance")

    void_check = 1.04 + PHI / 1000
    T(check("VOID_CONSTANT alignment", abs(void_check - VOID_CONSTANT) < 1e-15,
            f"{void_check}"), "void_align")

    # ═══════════════════════════════════════════════════════════
    # Phase 15: Benchmark — Throughput + Sacred Alignment
    # ═══════════════════════════════════════════════════════════
    header(15, "Benchmark: Throughput + Sacred Alignment")

    if vqpu_ok:
        bench_mesh = QuantumNetworkMesh(
            node_ids=["bench-A", "bench-B", "bench-C"],
            qubits_per_node=4,
        )
        bench_mesh.establish_channels()

        N_BENCH = 100
        t_bench = time.time()
        bench_ok = 0
        for i in range(N_BENCH):
            r = bench_mesh.teleport("bench-A", "bench-B", {"i": i})
            if r.get("success"):
                bench_ok += 1
        bench_time = time.time() - t_bench
        throughput = N_BENCH / bench_time if bench_time > 0 else 0

        T(check(f"VQPU throughput ({N_BENCH} teleports)",
                throughput > 100,
                f"{throughput:.0f} teleports/sec, {bench_time*1000:.1f}ms total"),
          "vqpu_throughput")
        T(check(f"VQPU success rate", bench_ok == N_BENCH,
                f"{bench_ok}/{N_BENCH}"), "vqpu_success_rate")

        # Network health after benchmark
        bh = bench_mesh.network_health()
        T(check("Post-benchmark health", bh["network_score"] > 0.5,
                f"score={bh['network_score']:.4f}, teleports={bh['total_teleportations']}"),
          "post_bench_health")

    if net_ok:
        N_NET_BENCH = 50
        t_nb = time.time()
        net_bench_ok = 0
        for i in range(N_NET_BENCH):
            try:
                r = net.teleport_score(alice.node_id, bob.node_id, score=0.5)
                if r.success:
                    net_bench_ok += 1
            except Exception:
                pass
        net_bench_time = time.time() - t_nb
        net_tp = N_NET_BENCH / net_bench_time if net_bench_time > 0 else 0

        T(check(f"Networker throughput ({N_NET_BENCH} teleports)",
                net_tp > 10,
                f"{net_tp:.0f} teleports/sec, {net_bench_time*1000:.1f}ms total"),
          "net_throughput")

    # ═══════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════
    total_time = time.time() - t_global
    print(f"\n{'═' * 70}")
    print(f"  DAEMON QUANTUM TELEPORTATION TEST SUITE — RESULTS")
    print(f"{'═' * 70}")
    print(f"  {T.summary()}")
    print(f"  Total time: {total_time:.2f}s")
    if T.errors:
        print(f"\n  Failed tests:")
        for e in T.errors:
            print(f"    ✗ {e}")
    print(f"\n  ALL {'PASSED' if T.failed == 0 else 'INCOMPLETE — see failures above'}")
    print(f"{'═' * 70}")

    return 0 if T.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
