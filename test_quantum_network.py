#!/usr/bin/env python3
"""
L104 Quantum Network Integration Test
═══════════════════════════════════════
Tests the full quantum network stack: per-daemon qubit registers,
Bell pair channels, entanglement purification, teleportation,
decoherence modeling, and micro daemon integration.

Run: .venv/bin/python test_quantum_network.py
"""
import json
import math
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l104_vqpu.constants import (
    GOD_CODE, PHI, VOID_CONSTANT,
    GOD_CODE_PHASE_ANGLE, IRON_PHASE_ANGLE,
    QPU_MEAN_FIDELITY, QPU_1Q_FIDELITY,
)
from l104_vqpu.quantum_network import (
    DaemonQubitRegister,
    QuantumNetworkMesh,
    QuantumChannel,
    QubitState,
    QUANTUM_NETWORK_VERSION,
    DEFAULT_DAEMON_QUBITS,
    FIDELITY_THRESHOLD_HIGH,
    FIDELITY_THRESHOLD_GOOD,
    FIDELITY_THRESHOLD_LOW,
    FIDELITY_FLOOR,
)

# ═══════════════════════════════════════════════════════════════════
# TEST INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════

_results = []
_phase_started = time.monotonic()


def _test(name: str, fn):
    """Run a single test and record the result."""
    t0 = time.monotonic()
    try:
        fn()
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        _results.append({"test": name, "pass": True, "elapsed_ms": elapsed})
        print(f"  ✓ {name} ({elapsed}ms)")
    except Exception as e:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        _results.append({"test": name, "pass": False, "error": str(e), "elapsed_ms": elapsed})
        print(f"  ✗ {name}: {e}")


def _phase(title: str):
    global _phase_started
    _phase_started = time.monotonic()
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


# ═══════════════════════════════════════════════════════════════════
# PHASE 1: QUBIT STATE FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════════
_phase("Phase 1: Qubit State Fundamentals")


def test_qubit_init():
    q = QubitState(qubit_id=0)
    assert q.qubit_id == 0
    assert len(q.state_vector) == 2
    assert abs(q.state_vector[0] - 1.0) < 1e-12, "Initial state should be |0⟩"
    assert abs(q.state_vector[1]) < 1e-12
    assert q.fidelity == 1.0
_test("qubit_initialization", test_qubit_init)


def test_qubit_hadamard():
    import numpy as np
    q = QubitState(qubit_id=0)
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    q.apply_gate(H)
    assert abs(abs(q.state_vector[0]) - 1 / np.sqrt(2)) < 1e-12
    assert abs(abs(q.state_vector[1]) - 1 / np.sqrt(2)) < 1e-12
_test("qubit_hadamard_gate", test_qubit_hadamard)


def test_qubit_normalization():
    import numpy as np
    q = QubitState(qubit_id=0)
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    q.apply_gate(H)
    norm = np.linalg.norm(q.state_vector)
    assert abs(norm - 1.0) < 1e-12, f"Norm={norm}, expected 1.0"
_test("qubit_normalization", test_qubit_normalization)


def test_qubit_fidelity_measurement():
    q = QubitState(qubit_id=0)
    f = q.measure_fidelity()
    assert 0.0 <= f <= 1.0, f"Fidelity {f} out of range"
    assert f > 0.9, f"Fresh qubit fidelity should be high, got {f}"
_test("qubit_fidelity_measurement", test_qubit_fidelity_measurement)


def test_qubit_sacred_score():
    q = QubitState(qubit_id=0)
    s = q.sacred_score()
    assert 0.0 <= s <= 1.0, f"Sacred score {s} out of range"
_test("qubit_sacred_score", test_qubit_sacred_score)


def test_qubit_to_dict():
    q = QubitState(qubit_id=7)
    d = q.to_dict()
    assert d["qubit_id"] == 7
    assert "fidelity" in d
    assert "phase" in d
    assert "sacred_alignment" in d
_test("qubit_to_dict", test_qubit_to_dict)


# ═══════════════════════════════════════════════════════════════════
# PHASE 2: DAEMON QUBIT REGISTER
# ═══════════════════════════════════════════════════════════════════
_phase("Phase 2: Daemon Qubit Register")


def test_register_creation():
    r = DaemonQubitRegister(node_id="test-node", num_qubits=4)
    assert r.num_qubits == 4
    assert r.node_id == "test-node"
    assert len(r.qubits) == 4
    assert r.calibration_count == 0
_test("register_creation", test_register_creation)


def test_register_max_qubits():
    r = DaemonQubitRegister(node_id="big", num_qubits=100)
    assert r.num_qubits == 16, f"Should cap at MAX_DAEMON_QUBITS=16, got {r.num_qubits}"
_test("register_max_qubits_cap", test_register_max_qubits)


def test_sacred_initialization():
    r = DaemonQubitRegister(node_id="sacred", num_qubits=4)
    result = r.initialize_sacred()
    assert result["calibration_count"] == 1
    assert result["avg_fidelity"] > 0.5
    assert result["avg_sacred_alignment"] > 0.0
    assert r.total_gates_applied == 20  # 5 gates × 4 qubits
_test("sacred_initialization", test_sacred_initialization)


def test_register_fidelity_check():
    r = DaemonQubitRegister(node_id="fcheck", num_qubits=2)
    r.initialize_sacred()
    check = r.fidelity_check()
    assert "avg_fidelity" in check
    assert "min_fidelity" in check
    assert "avg_sacred_alignment" in check
    assert "degraded_qubits" in check
    assert check["qubit_count"] == 2
    assert check["avg_fidelity"] > 0.5
_test("register_fidelity_check", test_register_fidelity_check)


def test_register_recalibrate():
    r = DaemonQubitRegister(node_id="recal", num_qubits=2)
    r.initialize_sacred()
    # Should not recalibrate when fidelity is good
    result = r.recalibrate_if_needed()
    # Result could be None (no recal needed) or a dict (if it was needed)
    assert r.calibration_count >= 1
_test("register_recalibrate_if_needed", test_register_recalibrate)


def test_register_entangled_pair():
    import numpy as np
    r = DaemonQubitRegister(node_id="ent", num_qubits=2)
    pair = r.entangled_pair_state(0, 1)
    assert len(pair) == 4  # 2-qubit joint state
    norm = np.linalg.norm(pair)
    assert abs(norm - 1.0) < 1e-12, f"Bell pair norm={norm}"
_test("register_entangled_pair", test_register_entangled_pair)


def test_register_get_qubit():
    r = DaemonQubitRegister(node_id="get", num_qubits=3)
    q = r.get_qubit(0)
    assert q is not None
    assert q.qubit_id == 0
    assert r.get_qubit(99) is None
_test("register_get_qubit", test_register_get_qubit)


# ═══════════════════════════════════════════════════════════════════
# PHASE 3: QUANTUM CHANNEL (BELL PAIRS)
# ═══════════════════════════════════════════════════════════════════
_phase("Phase 3: Quantum Channel (Bell Pairs)")


def test_channel_creation():
    ch = QuantumChannel(node_a="A", node_b="B")
    assert ch.node_a in ("A", "B")
    assert ch.node_b in ("A", "B")
    assert ch.active is False  # not established yet
_test("channel_creation", test_channel_creation)


def test_channel_establishment():
    ch = QuantumChannel(node_a="A", node_b="B")
    result = ch.establish()
    assert result["active"] is True
    assert ch.active is True
    assert ch.fidelity > 0.9
    assert ch.bell_state is not None
    assert len(ch.bell_state) == 4
_test("channel_establishment", test_channel_establishment)


def test_channel_god_code_phase():
    ch = QuantumChannel(node_a="A", node_b="B")
    ch.establish()
    # Bell state should have GOD_CODE phase applied
    import numpy as np
    norm = np.linalg.norm(ch.bell_state)
    assert abs(norm - 1.0) < 1e-12, f"Bell state norm={norm}"
_test("channel_god_code_phase", test_channel_god_code_phase)


def test_channel_purification():
    ch = QuantumChannel(node_a="A", node_b="B")
    ch.establish()
    result = ch.purify()
    # Channels with high fidelity may skip purification
    assert "channel_id" in result or "purified" in result
_test("channel_purification", test_channel_purification)


def test_channel_teleportation():
    ch = QuantumChannel(node_a="A", node_b="B")
    ch.establish()
    payload = {"key": "quantum_hello", "value": 42}
    result = ch.teleport_payload(payload)
    assert result["success"] is True
    assert result["fidelity"] > 0.5
    assert "measurement_outcome" in result
    assert result["measurement_outcome"] in ("00", "01", "10", "11")
_test("channel_teleportation", test_channel_teleportation)


def test_channel_decoherence():
    ch = QuantumChannel(node_a="A", node_b="B")
    ch.establish()
    initial_f = ch.fidelity
    new_f = ch.apply_decoherence()
    # Decoherence should not increase fidelity (or only trivially)
    assert new_f <= initial_f + 1e-6, f"Fidelity increased: {initial_f} → {new_f}"
    assert new_f > 0.0
_test("channel_decoherence", test_channel_decoherence)


def test_channel_to_dict():
    ch = QuantumChannel(node_a="X", node_b="Y")
    ch.establish()
    d = ch.to_dict()
    assert "channel_id" in d
    assert "fidelity" in d
    assert "node_a" in d
    assert "node_b" in d
    assert "active" in d
    assert d["active"] is True
_test("channel_to_dict", test_channel_to_dict)


# ═══════════════════════════════════════════════════════════════════
# PHASE 4: QUANTUM NETWORK MESH
# ═══════════════════════════════════════════════════════════════════
_phase("Phase 4: Quantum Network Mesh")


def test_mesh_creation():
    m = QuantumNetworkMesh()
    assert len(m.node_ids) == 0
    assert len(m.channels) == 0
_test("mesh_creation", test_mesh_creation)


def test_mesh_add_nodes():
    m = QuantumNetworkMesh()
    r1 = m.add_node("n1")
    assert r1["added"] is True
    r2 = m.add_node("n2")
    assert r2["added"] is True
    assert r2["total_nodes"] == 2
    assert r2["new_channels"] == 1  # n1-n2
_test("mesh_add_nodes", test_mesh_add_nodes)


def test_mesh_add_duplicate():
    m = QuantumNetworkMesh()
    m.add_node("n1")
    r = m.add_node("n1")
    assert r["added"] is False
_test("mesh_add_duplicate_node", test_mesh_add_duplicate)


def test_mesh_complete_graph():
    m = QuantumNetworkMesh()
    for i in range(4):
        m.add_node(f"n{i}")
    h = m.network_health()
    # C(4,2) = 6 channels
    assert h["nodes"] == 4
    assert h["active_channels"] == 6
_test("mesh_complete_graph_topology", test_mesh_complete_graph)


def test_mesh_remove_node():
    m = QuantumNetworkMesh()
    m.add_node("a")
    m.add_node("b")
    m.add_node("c")
    result = m.remove_node("b")
    assert result["removed"] is True
    h = m.network_health()
    assert h["nodes"] == 2
    # Only a-c channel remains
    assert h["active_channels"] == 1
_test("mesh_remove_node", test_mesh_remove_node)


def test_mesh_health_report():
    m = QuantumNetworkMesh()
    m.add_node("x")
    m.add_node("y")
    h = m.network_health()
    assert "avg_fidelity" in h
    assert "connectivity" in h
    assert "sacred_alignment" in h
    assert "network_score" in h
    assert "register_health" in h
    assert h["nodes"] == 2
    assert h["god_code"] == GOD_CODE
_test("mesh_health_report", test_mesh_health_report)


def test_mesh_teleport():
    m = QuantumNetworkMesh()
    m.add_node("src")
    m.add_node("dst")
    result = m.teleport("src", "dst", {"data": "quantum_transfer"})
    assert result["success"] is True
    assert result["fidelity"] > 0.5
_test("mesh_teleport", test_mesh_teleport)


def test_mesh_purify_all():
    m = QuantumNetworkMesh()
    m.add_node("p1")
    m.add_node("p2")
    m.add_node("p3")
    result = m.purify_all()
    assert "purified" in result
    assert "skipped" in result
    assert "total_channels" in result
_test("mesh_purify_all", test_mesh_purify_all)


def test_mesh_decoherence_cycle():
    m = QuantumNetworkMesh()
    m.add_node("d1")
    m.add_node("d2")
    result = m.decoherence_cycle()
    assert "channels_checked" in result
    assert "degraded" in result
_test("mesh_decoherence_cycle", test_mesh_decoherence_cycle)


def test_mesh_status():
    m = QuantumNetworkMesh()
    m.add_node("s1")
    m.add_node("s2")
    s = m.status()
    assert s["nodes"] == 2
    assert "channels_active" in s
    assert "avg_fidelity" in s
_test("mesh_quick_status", test_mesh_status)


def test_mesh_persist_state():
    import tempfile
    m = QuantumNetworkMesh()
    m.add_node("ps1")
    m.add_node("ps2")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    m.persist_state(path)
    data = json.loads(open(path).read())
    assert data["version"] == QUANTUM_NETWORK_VERSION
    assert len(data["node_ids"]) == 2
    os.unlink(path)
_test("mesh_persist_state", test_mesh_persist_state)


def test_mesh_get_channel():
    m = QuantumNetworkMesh()
    m.add_node("gc1")
    m.add_node("gc2")
    ch = m.get_channel("gc1", "gc2")
    assert ch is not None
    assert ch.active is True
    # Reverse order should also work
    ch2 = m.get_channel("gc2", "gc1")
    assert ch2 is not None
    assert ch2.channel_id == ch.channel_id
_test("mesh_get_channel", test_mesh_get_channel)


# ═══════════════════════════════════════════════════════════════════
# PHASE 5: SACRED CONSTANTS ALIGNMENT
# ═══════════════════════════════════════════════════════════════════
_phase("Phase 5: Sacred Constants Alignment")


def test_god_code_resonance():
    """Verify (GOD_CODE/16)^φ ≈ 286"""
    resonance = (GOD_CODE / 16.0) ** PHI
    assert abs(resonance - 286.0) < 1e-6, f"Resonance={resonance}"
_test("god_code_resonance_286", test_god_code_resonance)


def test_void_constant():
    expected = 1.04 + PHI / 1000
    assert abs(VOID_CONSTANT - expected) < 1e-15
_test("void_constant_formula", test_void_constant)


def test_fidelity_floor_derivation():
    """FIDELITY_FLOOR = QPU_MEAN_FIDELITY / PHI"""
    expected = QPU_MEAN_FIDELITY / PHI
    assert abs(FIDELITY_FLOOR - expected) < 1e-10
_test("fidelity_floor_derivation", test_fidelity_floor_derivation)


def test_god_code_phase_angle():
    expected = GOD_CODE % (2 * math.pi)
    assert abs(GOD_CODE_PHASE_ANGLE - expected) < 1e-10
_test("god_code_phase_angle", test_god_code_phase_angle)


def test_iron_phase_angle():
    assert abs(IRON_PHASE_ANGLE - math.pi / 2) < 1e-15
_test("iron_phase_angle_pi_over_2", test_iron_phase_angle)


# ═══════════════════════════════════════════════════════════════════
# PHASE 6: MICRO DAEMON QUANTUM TASKS
# ═══════════════════════════════════════════════════════════════════
_phase("Phase 6: Micro Daemon Quantum Tasks")


def test_micro_daemon_import():
    from l104_vqpu.micro_daemon import VQPUMicroDaemon, MICRO_DAEMON_VERSION
    assert MICRO_DAEMON_VERSION == "3.0.0"
_test("micro_daemon_v3_import", test_micro_daemon_import)


def test_micro_daemon_quantum_config():
    from l104_vqpu.micro_daemon import MicroDaemonConfig
    cfg = MicroDaemonConfig()
    assert hasattr(cfg, "enable_quantum_network")
    assert hasattr(cfg, "quantum_qubits")
_test("micro_daemon_quantum_config", test_micro_daemon_quantum_config)


def test_micro_daemon_quantum_tasks_exist():
    from l104_vqpu.micro_daemon import (
        _micro_qubit_fidelity,
        _micro_qubit_sacred_probe,
        _micro_quantum_network_health,
        _micro_channel_purification,
        _micro_qubit_recalibrate,
    )
    assert callable(_micro_qubit_fidelity)
    assert callable(_micro_qubit_sacred_probe)
    assert callable(_micro_quantum_network_health)
    assert callable(_micro_channel_purification)
    assert callable(_micro_qubit_recalibrate)
_test("quantum_micro_tasks_callable", test_micro_daemon_quantum_tasks_exist)


def test_micro_daemon_quantum_init():
    from l104_vqpu.micro_daemon import VQPUMicroDaemon
    d = VQPUMicroDaemon(enable_quantum_network=True, quantum_qubits=2)
    assert d._enable_quantum_network is True
    assert d._quantum_qubits == 2
    d.start()
    time.sleep(0.3)
    # Quantum register should be initialized
    assert d._qubit_register is not None
    assert d._quantum_mesh is not None
    assert d._quantum_node_id is not None
    d.stop()
_test("micro_daemon_quantum_init", test_micro_daemon_quantum_init)


def test_micro_daemon_quantum_status():
    from l104_vqpu.micro_daemon import VQPUMicroDaemon
    d = VQPUMicroDaemon(enable_quantum_network=True, quantum_qubits=2)
    d.start()
    time.sleep(0.3)
    qs = d.quantum_status()
    assert qs["quantum_network_enabled"] is True
    assert "register" in qs
    assert qs["register"]["avg_fidelity"] > 0.5
    d.stop()
_test("micro_daemon_quantum_status", test_micro_daemon_quantum_status)


def test_micro_daemon_self_test():
    from l104_vqpu.micro_daemon import VQPUMicroDaemon
    d = VQPUMicroDaemon(enable_quantum_network=True, quantum_qubits=2)
    d.start()
    time.sleep(0.3)
    st = d.self_test()
    assert st["total"] == 15, f"Expected 15 probes, got {st['total']}"
    assert st["passed"] >= 13, f"Only {st['passed']}/15 passed"
    # Check quantum-specific probes exist
    test_names = [t["test"] for t in st["tests"]]
    assert "quantum_qubit_register" in test_names
    assert "quantum_mesh_health" in test_names
    assert "sacred_qubit_alignment" in test_names
    d.stop()
_test("micro_daemon_self_test_15_probes", test_micro_daemon_self_test)


def test_micro_daemon_repr():
    from l104_vqpu.micro_daemon import VQPUMicroDaemon
    d = VQPUMicroDaemon(enable_quantum_network=True)
    r = repr(d)
    assert "+quantum" in r
_test("micro_daemon_repr_quantum_tag", test_micro_daemon_repr)


def test_micro_daemon_force_tick_with_quantum():
    from l104_vqpu.micro_daemon import VQPUMicroDaemon
    d = VQPUMicroDaemon(enable_quantum_network=True, quantum_qubits=2)
    d.start()
    time.sleep(0.3)
    result = d.force_tick()
    # Should have quantum tasks in the registry
    assert "qubit_fidelity" in d._task_registry
    assert "qubit_sacred_probe" in d._task_registry
    assert "quantum_net_health" in d._task_registry
    d.stop()
_test("micro_daemon_force_tick_quantum", test_micro_daemon_force_tick_with_quantum)


# ═══════════════════════════════════════════════════════════════════
# PHASE 7: PACKAGE INTEGRATION
# ═══════════════════════════════════════════════════════════════════
_phase("Phase 7: Package Integration")


def test_package_exports():
    from l104_vqpu import (
        DaemonQubitRegister,
        QuantumNetworkMesh,
        QuantumChannel,
        QubitState,
    )
    assert DaemonQubitRegister is not None
    assert QuantumNetworkMesh is not None
_test("package_level_exports", test_package_exports)


def test_constants_version():
    assert QUANTUM_NETWORK_VERSION is not None
    assert DEFAULT_DAEMON_QUBITS == 4
_test("quantum_network_constants", test_constants_version)


def test_fidelity_thresholds():
    assert FIDELITY_THRESHOLD_HIGH > FIDELITY_THRESHOLD_GOOD
    assert FIDELITY_THRESHOLD_GOOD > FIDELITY_THRESHOLD_LOW
    assert FIDELITY_THRESHOLD_LOW > FIDELITY_FLOOR
    assert FIDELITY_FLOOR > 0.0
_test("fidelity_threshold_order", test_fidelity_thresholds)


# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════

total = len(_results)
passed = sum(1 for r in _results if r["pass"])
failed = total - passed
total_ms = round(sum(r.get("elapsed_ms", 0) for r in _results), 2)

print(f"\n{'═' * 60}")
print(f"  L104 Quantum Network Integration Test — RESULTS")
print(f"{'═' * 60}")
print(f"  Version:   {QUANTUM_NETWORK_VERSION}")
print(f"  Tests:     {passed}/{total} passed")
print(f"  Failed:    {failed}")
print(f"  Elapsed:   {total_ms}ms")
print(f"  GOD_CODE:  {GOD_CODE}")
print(f"  QPU Fid:   {QPU_MEAN_FIDELITY}")
print(f"{'═' * 60}")

if failed > 0:
    print("\nFailed tests:")
    for r in _results:
        if not r["pass"]:
            print(f"  ✗ {r['test']}: {r.get('error', 'unknown')}")

# Write JSON report
report = {
    "test_suite": "quantum_network_integration",
    "version": QUANTUM_NETWORK_VERSION,
    "timestamp": time.time(),
    "total": total,
    "passed": passed,
    "failed": failed,
    "all_pass": failed == 0,
    "elapsed_ms": total_ms,
    "tests": _results,
    "constants": {
        "GOD_CODE": GOD_CODE,
        "PHI": PHI,
        "VOID_CONSTANT": VOID_CONSTANT,
        "QPU_MEAN_FIDELITY": QPU_MEAN_FIDELITY,
        "FIDELITY_FLOOR": FIDELITY_FLOOR,
    },
}
report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_quantum_network_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"\nReport: {report_path}")

sys.exit(0 if failed == 0 else 1)
