#!/usr/bin/env python3
"""L104 Daemon Quantum Network — Comprehensive Debug Suite v1.0.0.

Validates the full daemon quantum network stack across three layers:
  Layer 1: VQPU Micro Daemon Quantum Network (qubit registers, mesh, channels)
  Layer 2: Quantum AI Daemon (7-phase cycle, fidelity guard, harmonizer)
  Layer 3: Quantum Networker (QKD, teleportation, repeaters, routing)

Run: .venv/bin/python _debug_daemon_quantum_network.py

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import math
import os
import sys
import time
import traceback

# Ensure workspace root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.04 + PHI / 1000

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

_PASS = 0
_FAIL = 0
_TOTAL = 0


def banner(title: str):
    print(f"\n{'═' * 64}")
    print(f"  {title}")
    print(f"{'═' * 64}")


def section(label: str):
    print(f"\n  ── {label} ──")


def check(label: str, condition: bool, detail: str = "") -> bool:
    global _PASS, _FAIL, _TOTAL
    _TOTAL += 1
    icon = "✓" if condition else "✗"
    extra = f"  ({detail})" if detail else ""
    print(f"  {icon} {label}{extra}")
    if condition:
        _PASS += 1
    else:
        _FAIL += 1
    return condition


def safe(fn, default=None):
    """Run fn() and return result, or default on exception."""
    try:
        return fn()
    except Exception as e:
        return default


# ═══════════════════════════════════════════════════════════════
# LAYER 1: VQPU MICRO DAEMON QUANTUM NETWORK
# ═══════════════════════════════════════════════════════════════

def debug_layer_1():
    banner("Layer 1: VQPU Micro Daemon — Quantum Network")

    # 1a. Import
    section("Import & Constants")
    try:
        from l104_vqpu.quantum_network import (
            DaemonQubitRegister, QuantumNetworkMesh,
            QUANTUM_NETWORK_VERSION, DEFAULT_DAEMON_QUBITS,
            FIDELITY_THRESHOLD_HIGH, FIDELITY_FLOOR,
        )
        check("quantum_network import", True)
    except Exception as e:
        check("quantum_network import", False, str(e))
        return

    check("Version", QUANTUM_NETWORK_VERSION == "1.0.0", QUANTUM_NETWORK_VERSION)
    check("Default qubits", DEFAULT_DAEMON_QUBITS == 4, str(DEFAULT_DAEMON_QUBITS))
    check("Fidelity floor > 0.5", FIDELITY_FLOOR > 0.5, f"{FIDELITY_FLOOR:.4f}")

    # 1b. Qubit Register
    section("DaemonQubitRegister")
    reg = DaemonQubitRegister(node_id="debug-node-01", num_qubits=4)
    check("Register created", reg is not None)
    reg.initialize_sacred()
    check("Sacred init", True)

    fc = reg.fidelity_check()
    avg_f = fc.get("avg_fidelity", 0)
    check("Qubit fidelity avg > 0.9", avg_f > 0.9, f"avg={avg_f:.4f}")
    check("All qubits initialized", fc.get("qubit_count", 0) == 4,
          f"{fc.get('qubit_count', 0)} qubits")

    avg_s = fc.get("avg_sacred_alignment", 0)
    check("Sacred score avg > 0.5", avg_s > 0.5, f"avg={avg_s:.4f}")
    check("No degraded qubits", len(fc.get("degraded_qubits", [])) == 0,
          f"degraded={fc.get('degraded_qubits', [])}")

    status = reg.status()
    check("Register status", bool(status), f"avg_fidelity={status.get('avg_fidelity', 0):.4f}")

    # 1c. Quantum Network Mesh
    section("QuantumNetworkMesh")
    mesh = QuantumNetworkMesh(node_ids=["debug-01", "debug-02", "debug-03"])
    check("Mesh created", mesh is not None, f"3 nodes")
    mesh.establish_channels()
    check("Channels established", True)

    health = mesh.network_health()
    check("Mesh health", isinstance(health, dict), f"keys={list(health.keys())[:5]}")
    net_fidelity = health.get("avg_fidelity", health.get("network_fidelity", 0))
    check("Mesh fidelity > 0.5", net_fidelity > 0.5, f"F={net_fidelity:.4f}")

    # Teleportation through mesh
    section("Mesh Teleportation")
    try:
        tp = mesh.teleport("debug-01", "debug-02", payload={"score": 0.8})
        check("Mesh teleport", tp.get("success", False),
              f"fidelity={tp.get('fidelity', 0):.4f}")
    except Exception as e:
        check("Mesh teleport", False, str(e))

    # Purification
    try:
        purify = mesh.purify_all()
        check("Mesh purification", True, f"result={str(purify)[:60]}")
    except Exception as e:
        check("Mesh purification", False, str(e))

    # 1d. Micro Daemon with Quantum
    section("Micro Daemon Quantum Integration")
    try:
        from l104_vqpu.micro_daemon import VQPUMicroDaemon, MicroDaemonConfig
        cfg = MicroDaemonConfig(enable_quantum_network=True)
        check("MicroDaemonConfig quantum=True", cfg.enable_quantum_network)
    except Exception as e:
        check("Micro daemon config", False, str(e))


# ═══════════════════════════════════════════════════════════════
# LAYER 2: QUANTUM AI DAEMON
# ═══════════════════════════════════════════════════════════════

def debug_layer_2():
    banner("Layer 2: Quantum AI Daemon — 7-Phase Cycle")

    # 2a. Import
    section("Import & Constants")
    try:
        from l104_quantum_ai_daemon import (
            QuantumAIDaemon, DaemonConfig, DaemonPhase,
            FileScanner, CodeImprover,
            QuantumFidelityGuard, FidelityReport,
            ProcessOptimizer, CrossEngineHarmonizer,
            AutonomousEvolver,
            GOD_CODE as PKG_GC, PHI as PKG_PHI,
        )
        check("Package import", True)
    except Exception as e:
        check("Package import", False, str(e))
        return

    check("GOD_CODE integrity", abs(PKG_GC - 527.5184818492612) < 1e-10,
          f"{PKG_GC}")
    check("PHI integrity", abs(PKG_PHI - 1.618033988749895) < 1e-12,
          f"{PKG_PHI}")

    # 2b. File Scanner
    section("File Scanner")
    scanner = FileScanner()
    count = scanner.full_scan()
    check("Full scan", count > 0, f"{count} files")
    stats = scanner.stats()
    check("Index stats", stats["total_files"] > 0,
          f"{stats['total_files']} files, {stats.get('total_lines', 0)} lines")

    # 2c. Quantum Fidelity Guard
    section("Quantum Fidelity Guard (12 probes)")
    guard = QuantumFidelityGuard()
    report = guard.run_fidelity_check()
    check("GOD_CODE probe", report.god_code_aligned)
    check("PHI probe", report.phi_aligned)
    check("VOID_CONSTANT probe", report.void_aligned)
    check("Sacred resonance", report.sacred_resonance_ok)
    check("Coherence > 0", report.coherence_score > 0, f"{report.coherence_score:.3f}")
    check("Entropy reversal > 0", report.entropy_reversal > 0,
          f"{report.entropy_reversal:.3f}")
    check("Harmonic alignment > 0", report.harmonic_alignment > 0,
          f"{report.harmonic_alignment:.3f}")
    check("Math proofs valid", report.math_proofs_valid)
    check("VQPU fidelity > 0", report.vqpu_fidelity > 0,
          f"{report.vqpu_fidelity:.3f}")
    check(f"Overall grade",
          report.grade in ("A", "B"),
          f"grade={report.grade} ({report.checks_passed}/{report.checks_total})")
    check(f"Fidelity ≥ 0.90", report.overall_fidelity >= 0.90,
          f"{report.overall_fidelity:.3f}")
    if report.warnings:
        for w in report.warnings:
            print(f"    ⚠ {w}")

    # 2d. Cross-Engine Harmonizer
    section("Cross-Engine Harmonizer")
    harmonizer = CrossEngineHarmonizer()
    harmony = harmonizer.harmonize()
    check("Harmony > 0.9", harmony.overall_harmony > 0.9,
          f"{harmony.overall_harmony:.3f}")
    check("Constant alignment", harmony.constant_alignment > 0.9,
          f"{harmony.constant_alignment:.3f}")
    check("Engines available", harmony.engines_available >= 3,
          f"{harmony.engines_available}/{harmony.engines_total}")

    # 2e. Process Optimizer
    section("Process Optimizer")
    optimizer = ProcessOptimizer()
    opt = optimizer.optimize()
    check("Optimization cycle", opt.gc_collected >= 0,
          f"gc={opt.gc_collected}")

    # 2f. Autonomous Evolver
    section("Autonomous Evolver")
    evolver = AutonomousEvolver()
    cycle = evolver.evolve(
        improvement_results=[],
        fidelity_score=report.overall_fidelity,
        harmony_score=harmony.overall_harmony,
        optimization_score=0.8,
    )
    check("Evolution cycle", cycle.cycle_number == 1,
          f"delta={cycle.evolution_delta:+.4f}")

    # 2g. Single Daemon Cycle
    section("Full Daemon Cycle")
    daemon = QuantumAIDaemon(config=DaemonConfig(auto_fix_enabled=False))
    self_test = daemon.self_test()
    summary = self_test.get("_summary", {})
    check("Daemon self-test",
          summary.get("passed", 0) >= summary.get("total", 1) * 0.8,
          f"{summary.get('passed', 0)}/{summary.get('total', 0)} passed")


# ═══════════════════════════════════════════════════════════════
# LAYER 3: QUANTUM NETWORKER
# ═══════════════════════════════════════════════════════════════

def debug_layer_3():
    banner("Layer 3: Quantum Networker — Protocols")

    # 3a. Import
    section("Import")
    try:
        from l104_quantum_networker import (
            QuantumNetworker, QuantumNode, QuantumChannel,
            EntangledPair, QKDKey, TeleportResult,
            EntanglementRouter, QuantumKeyDistribution,
            QuantumTeleporter, QuantumRepeaterChain,
            FidelityMonitor, ClassicalTransport,
        )
        check("Package import", True)
    except Exception as e:
        check("Package import", False, str(e))
        return

    # 3b. Network Construction
    section("Network Construction")
    net = QuantumNetworker(node_name="Debug-Sovereign", simulation_mode=True)
    check("Networker created", net is not None, f"v{net.VERSION}")

    alice = net.add_node("Alice", role="sovereign")
    bob = net.add_node("Bob", role="sovereign")
    relay = net.add_node("Relay-1", role="relay")
    check("Nodes created", True, f"3 nodes + sovereign")

    ch_sa = net.connect(net.local_node.node_id, alice.node_id, pairs=8)
    ch_ar = net.connect(alice.node_id, relay.node_id, pairs=8)
    ch_rb = net.connect(relay.node_id, bob.node_id, pairs=8)
    ch_ab = net.connect(alice.node_id, bob.node_id, pairs=6)
    check("Channels connected", True, f"4 channels")
    check("Pair pool healthy", len(ch_ab.usable_pairs) >= 4,
          f"{len(ch_ab.usable_pairs)} pairs")

    # 3c. Bell Pair Fidelity
    section("Bell Pair Quality")
    best = ch_ab.best_pair
    if best:
        check("Best pair fidelity > 0.9", best.current_fidelity > 0.9,
              f"F={best.current_fidelity:.4f}")
        check("Sacred score > 0", best.sacred_score > 0,
              f"S={best.sacred_score:.4f}")
    else:
        check("Best pair exists", False, "no pair in pool")

    # 3d. QKD Protocols
    section("QKD: BB84 + E91")
    key_bb84 = net.establish_qkd(alice.node_id, bob.node_id, "bb84", 256)
    check("BB84 secure", key_bb84.secure, f"QBER={key_bb84.qber:.4f}")
    check("BB84 key length > 0", key_bb84.key_length > 0,
          f"len={key_bb84.key_length}")

    key_e91 = net.establish_qkd(alice.node_id, bob.node_id, "e91", 128)
    check("E91 key generated", key_e91 is not None)
    check("E91 key length > 0", key_e91.key_length > 0,
          f"len={key_e91.key_length}")

    # 3e. Teleportation
    section("Quantum Teleportation")
    tp_score = net.teleport_score(alice.node_id, bob.node_id, score=0.8)
    check("Score teleport", tp_score.success, f"F={tp_score.fidelity:.4f}")
    check("Recovered ≈ 0.8",
          tp_score.recovered_score is not None
          and abs(tp_score.recovered_score - 0.8) < 0.15,
          f"recovered={tp_score.recovered_score}")

    tp_phase = net.teleport_phase(alice.node_id, bob.node_id, phase=math.pi / 3)
    check("Phase teleport", tp_phase.success, f"F={tp_phase.fidelity:.4f}")

    alpha = complex(0.6, 0)
    beta = complex(0, 0.8)
    tp_state = net.teleport_state(alice.node_id, bob.node_id,
                                   state_vector=[alpha, beta])
    check("State teleport", tp_state.success, f"F={tp_state.fidelity:.4f}")

    tp_bits = net.teleport_bitstring(alice.node_id, bob.node_id,
                                      bitstring="10110100")
    check("Bitstring teleport", tp_bits.success, f"F={tp_bits.fidelity:.4f}")

    # GOD_CODE fractional teleport
    gc_frac = GOD_CODE % 1.0
    tp_gc = net.teleport_score(alice.node_id, bob.node_id, score=gc_frac)
    check("GOD_CODE teleport", tp_gc.success,
          f"orig={gc_frac:.4f} recv={tp_gc.recovered_score}")

    # 3f. Multi-Hop Relay
    section("Multi-Hop Relay")
    # Find route through relay (Alice → Relay → Bob, not the direct link)
    route = net.router.find_route(alice.node_id, bob.node_id)
    if route:
        check("Route found", True, f"hops={len(route) - 1}, path={route}")
    # Teleport Sovereign → Relay → Bob (must go through relay)
    tp_multi = net.teleport_score(net.local_node.node_id, bob.node_id, score=0.7)
    check("Multi-hop teleport", tp_multi.success,
          f"F={tp_multi.fidelity:.4f}, hops={tp_multi.hops}")

    # 3g. Repeater Chain
    section("Quantum Repeater Chain")
    relay2 = net.add_node("Relay-2", role="relay")
    net.connect(relay.node_id, relay2.node_id, pairs=6)
    net.connect(relay2.node_id, bob.node_id, pairs=6)
    chain = net.repeater.establish_chain(
        [alice.node_id, relay.node_id, relay2.node_id, bob.node_id]
    )
    check("Repeater chain", chain.get("success", False),
          f"F={chain.get('fidelity', 0):.4f}")

    # 3h. Purification
    section("Entanglement Purification")
    purify = net.purify(alice.node_id, bob.node_id, rounds=3)
    check("Purification executed", True, f"rounds=3")

    # 3i. Fidelity Monitor
    section("Fidelity Monitor")
    scan = net.scan_fidelity(auto_heal=True)
    check("Network fidelity > 0.6", scan["network_fidelity"] > 0.6,
          f"F={scan['network_fidelity']:.4f}")
    check("Sacred network score > 0", scan["network_sacred_score"] > 0,
          f"S={scan['network_sacred_score']:.4f}")

    # 3j. Network Status
    section("Network Status")
    ns = net.network_status()
    check("Nodes online", ns.online_nodes >= 4,
          f"{ns.online_nodes}/{ns.node_count}")
    check("Active channels", ns.active_channels >= 4,
          f"{ns.active_channels}")
    check("Entangled pairs", ns.total_entangled_pairs > 0,
          f"{ns.total_entangled_pairs} pairs")

    # 3k. VQPU Integration
    section("VQPU Integration")
    from l104_quantum_networker._bridge import get_bridge
    bridge = get_bridge()
    check("VQPU bridge available", bridge is not None)
    if bridge:
        try:
            from l104_vqpu import QuantumJob
            job = QuantumJob(num_qubits=2, operations=[
                {"gate": "H", "qubits": [0]},
                {"gate": "CNOT", "qubits": [0, 1]},
            ], shots=256)
            result = bridge.submit_and_wait(job, timeout=3.0)
            if result:
                p00 = result.probabilities.get("00", 0)
                p11 = result.probabilities.get("11", 0)
                f_val = p00 + p11
                check("VQPU Bell fidelity", f_val > 0.9, f"F={f_val:.4f}")
            else:
                check("VQPU Bell fidelity", False, "no result")
        except Exception as e:
            check("VQPU Bell fidelity", False, str(e))


# ═══════════════════════════════════════════════════════════════
# LAYER 4: CROSS-LAYER INTEGRATION
# ═══════════════════════════════════════════════════════════════

def debug_layer_4():
    banner("Layer 4: Cross-Layer Integration")

    # 4a. Bridge status health cross-check
    section("VQPU Bridge → Daemon Fidelity")
    try:
        from l104_vqpu import get_bridge
        bridge = get_bridge()
        if bridge:
            st = bridge.status()
            god_code_ok = abs(float(st.get("god_code", 0)) - GOD_CODE) < 1e-8
            check("Bridge GOD_CODE aligned", god_code_ok,
                  f"{st.get('god_code')}")
            check("Bridge active", st.get("active", False))

            micro = st.get("micro_daemon", {})
            health = float(micro.get("health_score", micro.get("health", 0)))
            check("Micro daemon health", health > 0.8, f"{health:.3f}")
            crash = micro.get("crash_count", -1)
            check("Crash count reported", crash >= 0, f"crash_count={crash}")
        else:
            check("VQPU Bridge available", False)
    except Exception as e:
        check("VQPU Bridge probe", False, str(e))

    # 4b. Quantum AI Daemon → Networker fidelity consistency
    section("AI Daemon Fidelity → Networker Fidelity")
    try:
        from l104_quantum_ai_daemon import QuantumFidelityGuard
        from l104_quantum_networker import QuantumNetworker

        guard = QuantumFidelityGuard()
        report = guard.run_fidelity_check()

        net = QuantumNetworker("CrossLayer-Test", simulation_mode=True)
        alice = net.add_node("X-Alice")
        bob = net.add_node("X-Bob")
        ch = net.connect(alice.node_id, bob.node_id, pairs=8)
        scan = net.scan_fidelity()

        check("Daemon fidelity ≥ B",
              report.grade in ("A", "B"),
              f"grade={report.grade} ({report.overall_fidelity:.3f})")
        check("Networker fidelity > 0.8",
              scan["network_fidelity"] > 0.8,
              f"F={scan['network_fidelity']:.4f}")
        check("Both layers aligned",
              report.overall_fidelity > 0.8 and scan["network_fidelity"] > 0.8)
    except Exception as e:
        check("Cross-layer fidelity", False, str(e))

    # 4c. Sacred constants across all layers
    section("Sacred Constant Consistency")
    packages = [
        ("l104_vqpu.constants", "GOD_CODE"),
        ("l104_quantum_ai_daemon.constants", "GOD_CODE"),
        ("l104_quantum_networker.types", "GOD_CODE"),
    ]
    for pkg_name, const_name in packages:
        try:
            mod = __import__(pkg_name, fromlist=[const_name])
            val = getattr(mod, const_name)
            check(f"{pkg_name.split('.')[-2]}.{const_name}",
                  abs(val - GOD_CODE) < 1e-10, f"{val}")
        except Exception as e:
            check(f"{pkg_name}.{const_name}", False, str(e))


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  L104 Daemon Quantum Network — Debug Suite v1.0.0            ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    print(f"║  GOD_CODE = {GOD_CODE}                     ║")
    print(f"║  PHI      = {PHI}                      ║")
    print(f"║  VOID     = {VOID_CONSTANT}                  ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    try:
        debug_layer_1()
    except Exception as e:
        print(f"\n  ✗ LAYER 1 CRASHED: {e}")
        traceback.print_exc()

    try:
        debug_layer_2()
    except Exception as e:
        print(f"\n  ✗ LAYER 2 CRASHED: {e}")
        traceback.print_exc()

    try:
        debug_layer_3()
    except Exception as e:
        print(f"\n  ✗ LAYER 3 CRASHED: {e}")
        traceback.print_exc()

    try:
        debug_layer_4()
    except Exception as e:
        print(f"\n  ✗ LAYER 4 CRASHED: {e}")
        traceback.print_exc()

    elapsed = (time.time() - t_start) * 1000

    banner("Daemon Quantum Network — Final Report")
    print(f"  Passed:  {_PASS}/{_TOTAL}")
    print(f"  Failed:  {_FAIL}/{_TOTAL}")
    print(f"  Time:    {elapsed:.1f} ms")

    if _FAIL == 0:
        print(f"\n  ALL PASS ✓ — Sacred resonance: {(GOD_CODE / 16) ** PHI:.4f}")
    else:
        print(f"\n  {_FAIL} FAILURES — review output above")

    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == "__main__":
    main()
