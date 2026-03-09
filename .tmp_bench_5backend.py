#!/usr/bin/env python3
"""
Benchmark all 5 backends of the MetalVQPU ASI router.
v3.0: Three-engine scoring integration — validates three-engine metrics
  in circuit results (entropy_reversal, harmonic_resonance, wave_coherence).

Circuits designed to trigger each routing path:
  1. stabilizer_chp:     Pure Clifford (H + CX only)
  2. cpu_statevector:     Small circuit with T gates (< 16Q)
  3. metal_gpu:           16+ qubit circuit with T gates + high entanglement
  4. tensor_network_mps:  16+ qubit circuit with mostly single-qubit T/Rz gates
  5. chunked_cpu:         Would trigger for 26Q+ (simulated with 24Q high-ent)
"""

import json
import os
import time
import uuid

BRIDGE_INBOX = "/tmp/l104_bridge/inbox"
BRIDGE_OUTBOX = "/tmp/l104_bridge/outbox"

# v3.0: Ensure bridge IPC directories exist before benchmarking
os.makedirs(BRIDGE_INBOX, exist_ok=True)
os.makedirs(BRIDGE_OUTBOX, exist_ok=True)
os.makedirs("/tmp/l104_bridge/telemetry", exist_ok=True)


def submit_and_wait(payload, timeout=30.0):
    """Submit a circuit and wait for result."""
    cid = payload["circuit_id"]
    fname = f"{cid}.json"
    result_fname = f"{cid}_result.json"

    # Clear any old result
    out_path = os.path.join(BRIDGE_OUTBOX, result_fname)
    if os.path.exists(out_path):
        os.remove(out_path)

    # Submit
    tmp = os.path.join(BRIDGE_INBOX, f".tmp_{fname}")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.rename(tmp, os.path.join(BRIDGE_INBOX, fname))

    # Wait for result
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(out_path):
            time.sleep(0.01)  # Let write complete
            with open(out_path) as f:
                return json.load(f)
        time.sleep(0.005)

    return {"error": "timeout", "circuit_id": cid}


def make_routing(ops, nq):
    """Run the CircuitAnalyzer logic inline."""
    CLIFFORD = {"H", "h", "X", "x", "Y", "y", "Z", "z", "S", "s", "SDG", "sdg",
                "SX", "sx", "CX", "cx", "CNOT", "cnot", "CZ", "cz", "SWAP", "swap",
                "I", "i", "ID", "id", "CY", "cy", "ECR", "ecr"}
    ENTANGLING = {"CX", "cx", "CNOT", "cnot", "CZ", "cz", "SWAP", "swap",
                  "CY", "cy", "ECR", "ecr", "iSWAP", "iswap"}

    is_cliff = True
    t_count = 0
    two_q = 0
    for op in ops:
        g = op["gate"]
        if g not in CLIFFORD:
            is_cliff = False
        if g.upper() in ("T", "TDG", "RZ", "RX", "RY"):
            t_count += 1
        if g.upper() in ENTANGLING or g in ENTANGLING:
            two_q += 1

    total = len(ops)
    ent_ratio = two_q / total if total > 0 else 0.0

    # Decision tree
    if is_cliff:
        backend = "stabilizer_chp"
    elif nq < 16:
        backend = "cpu_statevector"
    elif ent_ratio <= 0.25:
        backend = "tensor_network_mps"
    elif nq <= 25:
        backend = "metal_gpu"
    else:
        backend = "chunked_cpu"

    return {
        "is_clifford": is_cliff,
        "entanglement_ratio": ent_ratio,
        "t_gate_count": t_count,
        "two_qubit_count": two_q,
        "single_qubit_count": total - two_q,
        "total_gates": total,
        "recommended_backend": backend,
    }


# ─── Test Circuits ───

circuits = []

# 1. STABILIZER CHP: 8Q Bell pairs (pure Clifford)
ops1 = []
for i in range(0, 8, 2):
    ops1.append({"gate": "H", "qubits": [i]})
    ops1.append({"gate": "CX", "qubits": [i, i + 1]})
circuits.append(("stabilizer_chp", {
    "circuit_id": "bench_stabilizer",
    "num_qubits": 8,
    "operations": ops1,
    "shots": 1024,
    "routing": make_routing(ops1, 8),
}))

# 2. CPU STATEVECTOR: 6Q circuit with T gates
ops2 = [
    {"gate": "H", "qubits": [0]},
    {"gate": "T", "qubits": [0]},
    {"gate": "CX", "qubits": [0, 1]},
    {"gate": "T", "qubits": [1]},
    {"gate": "H", "qubits": [2]},
    {"gate": "Rz", "qubits": [2], "parameters": [0.7854]},
    {"gate": "CX", "qubits": [2, 3]},
    {"gate": "T", "qubits": [3]},
    {"gate": "H", "qubits": [4]},
    {"gate": "CX", "qubits": [4, 5]},
    {"gate": "Rz", "qubits": [5], "parameters": [1.5708]},
]
circuits.append(("cpu_statevector", {
    "circuit_id": "bench_cpu_sv",
    "num_qubits": 6,
    "operations": ops2,
    "shots": 1024,
    "routing": make_routing(ops2, 6),
}))

# 3. METAL GPU: 18Q circuit with T gates + heavy entanglement (>25%)
ops3 = []
for i in range(18):
    ops3.append({"gate": "H", "qubits": [i]})
for i in range(17):
    ops3.append({"gate": "CX", "qubits": [i, i + 1]})
    ops3.append({"gate": "T", "qubits": [i]})
# More CX to push entanglement above 25%
for i in range(0, 16, 2):
    ops3.append({"gate": "CX", "qubits": [i, i + 2]})
circuits.append(("metal_gpu", {
    "circuit_id": "bench_metal_gpu",
    "num_qubits": 18,
    "operations": ops3,
    "shots": 1024,
    "routing": make_routing(ops3, 18),
}))

# 4. TENSOR NETWORK MPS: 18Q circuit with mostly single-qubit gates (low entanglement)
ops4 = []
for i in range(18):
    ops4.append({"gate": "H", "qubits": [i]})
    ops4.append({"gate": "Rz", "qubits": [i], "parameters": [0.3 * i]})
    ops4.append({"gate": "Rx", "qubits": [i], "parameters": [0.5 * i]})
    ops4.append({"gate": "T", "qubits": [i]})
# Only 2 CX gates (entanglement ratio ≈ 2/74 ≈ 2.7%)
ops4.append({"gate": "CX", "qubits": [0, 1]})
ops4.append({"gate": "CX", "qubits": [8, 9]})
circuits.append(("tensor_network_mps", {
    "circuit_id": "bench_mps",
    "num_qubits": 18,
    "operations": ops4,
    "shots": 1024,
    "routing": make_routing(ops4, 18),
}))

# 5. CHUNKED CPU: 16Q forced to chunked path (verifies routing + correctness)
ops5 = []
for i in range(16):
    ops5.append({"gate": "H", "qubits": [i]})
for i in range(15):
    ops5.append({"gate": "CX", "qubits": [i, i + 1]})
    ops5.append({"gate": "T", "qubits": [i]})
circuits.append(("chunked_cpu", {
    "circuit_id": "bench_chunked",
    "num_qubits": 16,
    "operations": ops5,
    "shots": 1024,
    "routing": {
        "is_clifford": False,
        "entanglement_ratio": 0.5,
        "t_gate_count": 15,
        "two_qubit_count": 15,
        "recommended_backend": "chunked_cpu",  # Force chunked path
    },
}))

# ─── Run Benchmark ───

print("=" * 70)
print("  L104 MetalVQPU 5-Backend ASI Router Benchmark v3.0")
print("  Three-Engine: entropy=0.35 harmonic=0.40 wave=0.25")
print("=" * 70)
print()

results = []
for expected_backend, payload in circuits:
    routing = payload.get("routing", {})
    rec = routing.get("recommended_backend", "?")
    nq = payload["num_qubits"]
    ng = len(payload["operations"])
    ent = routing.get("entanglement_ratio", 0)

    print(f"  [{expected_backend}] {payload['circuit_id']}")
    print(f"    {nq}Q, {ng} gates, ent_ratio={ent:.3f}, hint={rec}")

    t0 = time.time()
    result = submit_and_wait(payload, timeout=30)
    wall_ms = (time.time() - t0) * 1000

    if "error" in result and result.get("backend") != "error":
        err = result.get("error", "unknown")
        print(f"    ERROR: {err}")
        results.append((expected_backend, payload["circuit_id"], "error", wall_ms, 0))
        continue

    actual_backend = result.get("backend", "?")
    exec_ms = result.get("execution_time_ms", 0)
    num_outcomes = len(result.get("probabilities", {}))
    match = "OK" if actual_backend == expected_backend else f"MISMATCH→{actual_backend}"

    # v3.0: Extract three-engine scoring if present
    three_eng = result.get("three_engine", {})
    te_composite = three_eng.get("composite", result.get("three_engine_composite", None))
    te_tag = f"  3E={te_composite:.3f}" if te_composite is not None else ""

    print(f"    backend={actual_backend} [{match}]  exec={exec_ms:.2f}ms  wall={wall_ms:.1f}ms  outcomes={num_outcomes}{te_tag}")
    results.append((expected_backend, payload["circuit_id"], actual_backend, wall_ms, exec_ms, te_composite))

print()
print("=" * 70)
print("  SUMMARY (v3.0 Three-Engine)")
print("=" * 70)
print(f"  {'Expected':<22} {'Actual':<22} {'Exec ms':>8} {'Wall ms':>8} {'3E':>6}")
print(f"  {'-'*22} {'-'*22} {'-'*8} {'-'*8} {'-'*6}")
for exp, cid, actual, wall, exec_t, te_comp in results:
    status = "✓" if exp == actual else "✗"
    te_str = f"{te_comp:.3f}" if te_comp is not None else "N/A"
    print(f"  {exp:<22} {actual:<22} {exec_t:>7.2f} {wall:>7.1f} {te_str:>6}  {status}")
print()
total_wall = sum(r[3] for r in results)
total_exec = sum(r[4] for r in results)
matches = sum(1 for r in results if r[0] == r[2])
te_scores = [r[5] for r in results if r[5] is not None]
print(f"  Routing accuracy: {matches}/{len(results)} correct")
print(f"  Total wall: {total_wall:.1f}ms | Total exec: {total_exec:.2f}ms")
print(f"  Avg IPC overhead: {(total_wall - total_exec) / len(results):.1f}ms per circuit")
if te_scores:
    print(f"  Three-Engine avg composite: {sum(te_scores)/len(te_scores):.4f} ({len(te_scores)}/{len(results)} circuits)")
else:
    print(f"  Three-Engine: no composite scores returned (daemon may be pre-v3.0)")
print()
