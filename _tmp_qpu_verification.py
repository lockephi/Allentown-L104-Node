#!/usr/bin/env python3
"""
L104 Quick Real QPU Verification via IBM Quantum
═══════════════════════════════════════════════════
Runs GOD_CODE circuits on real IBM hardware (ibm_torino / least-busy backend).
Compares live QPU results against embedded QPU verification data.

Temp file — safe to delete after use.
"""

import os, sys, time, json
from dotenv import load_dotenv
load_dotenv()

# ─── Step 1: Connect to IBM Quantum ─────────────────────────────────
print("=" * 64)
print("  L104 REAL QPU VERIFICATION — IBM Quantum")
print("=" * 64)

token = os.environ.get("IBMQ_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN")
if not token:
    print("\n  FATAL: No IBM Quantum token. Set IBMQ_TOKEN in .env")
    sys.exit(1)

channel = os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_cloud")
instance = os.environ.get("IBM_QUANTUM_INSTANCE") or None

print(f"\n  [1/6] Connecting to IBM Quantum (channel={channel})...")
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Batch
    from qiskit import QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    svc_kwargs = {"channel": channel, "token": token}
    if instance:
        svc_kwargs["instance"] = instance
    service = QiskitRuntimeService(**svc_kwargs)
    print(f"        OK — authenticated (channel={channel})")
except Exception as e:
    print(f"        FAIL: {e}")
    sys.exit(1)

# ─── Step 2: Select backend ─────────────────────────────────────────
print(f"\n  [2/6] Selecting backend...")
try:
    backends = service.backends(min_num_qubits=2, operational=True)
    backend_names = [b.name for b in backends]
    print(f"        Available: {backend_names}")

    # Prefer ibm_torino (Heron r2, our verified backend), else least_busy
    preferred = ["ibm_torino", "ibm_brisbane", "ibm_osaka", "ibm_kyoto", "ibm_sherbrooke"]
    backend = None
    for pref in preferred:
        if pref in backend_names:
            backend = service.backend(pref)
            break
    if backend is None:
        backend = service.least_busy(min_num_qubits=2, operational=True)

    print(f"        Selected: {backend.name} ({backend.num_qubits}Q)")
except Exception as e:
    print(f"        FAIL: {e}")
    sys.exit(1)

# ─── Step 3: Build verification circuits ────────────────────────────
print(f"\n  [3/6] Building verification circuits...")

import numpy as np
from l104_god_code_simulator.constants import GOD_CODE, PHI

circuits = {}

# Circuit 1: Bell State (2Q) — deterministic entanglement check
qc1 = QuantumCircuit(2, name="bell_state")
qc1.h(0)
qc1.cx(0, 1)
qc1.measure_all()
circuits["bell_state"] = {
    "qc": qc1,
    "ideal": {"00": 0.5, "11": 0.5},
    "desc": "Bell state |Φ+⟩ — expect 50/50 on |00⟩,|11⟩",
}

# Circuit 2: 1Q GOD_CODE Phase Gate — sacred alignment
theta_god = 2.0 * np.pi * (GOD_CODE / 1000.0)  # GOD_CODE-derived rotation
qc2 = QuantumCircuit(1, name="1q_god_code")
qc2.h(0)
qc2.rz(theta_god, 0)
qc2.h(0)
qc2.measure_all()
p1_ideal = np.sin(theta_god / 2) ** 2
circuits["1q_god_code"] = {
    "qc": qc2,
    "ideal": {"0": 1.0 - p1_ideal, "1": p1_ideal},
    "desc": f"GOD_CODE phase gate θ={theta_god:.4f} rad",
}

# Circuit 3: GHZ State (3Q) — multipartite entanglement
qc3 = QuantumCircuit(3, name="ghz_3q")
qc3.h(0)
qc3.cx(0, 1)
qc3.cx(1, 2)
qc3.measure_all()
circuits["ghz_3q"] = {
    "qc": qc3,
    "ideal": {"000": 0.5, "111": 0.5},
    "desc": "GHZ |000⟩+|111⟩ — tripartite entanglement",
}

# Circuit 4: PHI-rotation interference (1Q)
theta_phi = 2.0 * np.pi / PHI
qc4 = QuantumCircuit(1, name="phi_interference")
qc4.h(0)
qc4.rz(theta_phi, 0)
qc4.h(0)
qc4.measure_all()
p1_phi = np.sin(theta_phi / 2) ** 2
circuits["phi_interference"] = {
    "qc": qc4,
    "ideal": {"0": 1.0 - p1_phi, "1": p1_phi},
    "desc": f"PHI interference θ=2π/φ={theta_phi:.4f} rad",
}

print(f"        Built {len(circuits)} circuits")
for name, c in circuits.items():
    print(f"          - {name}: {c['desc']}")

# ─── Step 4: Transpile for hardware ─────────────────────────────────
print(f"\n  [4/6] Transpiling for {backend.name}...")
pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
transpiled = []
circuit_names = []
for name, c in circuits.items():
    t = pm.run(c["qc"])
    transpiled.append(t)
    circuit_names.append(name)
    print(f"          {name}: depth={t.depth()}, gates={t.count_ops()}")

# ─── Step 5: Execute on real QPU ─────────────────────────────────────
SHOTS = 4096
print(f"\n  [5/6] Submitting {len(transpiled)} circuits to {backend.name} ({SHOTS} shots each)...")
print(f"        This may take 1-5 minutes depending on queue...")

t0 = time.time()
try:
    # Use Batch mode for efficient multi-circuit execution
    with Batch(backend=backend) as batch:
        sampler = Sampler(mode=batch)
        job = sampler.run(transpiled, shots=SHOTS)
        job_id = job.job_id()
        print(f"        Job ID: {job_id}")
        print(f"        Waiting for results...")
        result = job.result()
    elapsed = time.time() - t0
    print(f"        Completed in {elapsed:.1f}s")
except Exception as e:
    print(f"        EXECUTION FAILED: {e}")
    print(f"        (Check IBM Quantum dashboard for queue status)")
    sys.exit(1)

# ─── Step 6: Analyze results ─────────────────────────────────────────
print(f"\n  [6/6] Analyzing QPU results...")
print("-" * 64)

results_data = {}
all_fidelities = []

for i, name in enumerate(circuit_names):
    pub_result = result[i]
    counts = pub_result.data.meas.get_counts()
    total = sum(counts.values())
    probs = {k: v / total for k, v in sorted(counts.items(), key=lambda x: -x[1])}
    ideal = circuits[name]["ideal"]

    # Bhattacharyya fidelity
    all_states = sorted(set(list(probs.keys()) + list(ideal.keys())))
    bc = sum(np.sqrt(probs.get(s, 0) * ideal.get(s, 0)) for s in all_states)
    fidelity = bc ** 2

    all_fidelities.append(fidelity)
    status = "PASS" if fidelity > 0.85 else "WARN" if fidelity > 0.70 else "FAIL"

    results_data[name] = {
        "counts": counts,
        "probabilities": dict(list(probs.items())[:6]),
        "fidelity": fidelity,
        "status": status,
    }

    print(f"\n  {status}  {name}")
    print(f"        {circuits[name]['desc']}")
    print(f"        Fidelity: {fidelity:.6f}")
    top = list(probs.items())[:4]
    print(f"        Top states: {' | '.join(f'{s}:{p:.4f}' for s,p in top)}")
    ideal_top = sorted(ideal.items(), key=lambda x: -x[1])[:4]
    print(f"        Ideal:      {' | '.join(f'{s}:{p:.4f}' for s,p in ideal_top)}")

# ─── Summary ─────────────────────────────────────────────────────────
mean_fid = np.mean(all_fidelities)
print("\n" + "=" * 64)
print(f"  RESULTS SUMMARY")
print(f"  Backend:        {backend.name}")
print(f"  Job ID:         {job_id}")
print(f"  Shots:          {SHOTS}")
print(f"  Circuits:       {len(circuits)}")
print(f"  Mean fidelity:  {mean_fid:.6f}")
print(f"  Elapsed:        {elapsed:.1f}s")
all_pass = all(r["status"] != "FAIL" for r in results_data.values())
print(f"  Verdict:        {'ALL PASS' if all_pass else 'HAS FAILURES'}")

# Compare with embedded QPU data
from l104_god_code_simulator.qpu_verification import get_qpu_verification_data
embedded = get_qpu_verification_data()
print(f"\n  Reference QPU data (ibm_torino, 2026-03-04):")
print(f"    Embedded mean fidelity: {embedded['mean_qpu_fidelity']:.6f}")
print(f"    Live mean fidelity:     {mean_fid:.6f}")
delta = abs(mean_fid - embedded["mean_qpu_fidelity"])
print(f"    Delta:                  {delta:.6f}")
print("=" * 64)

# Save results
out = {
    "backend": backend.name,
    "job_id": job_id,
    "shots": SHOTS,
    "elapsed_s": round(elapsed, 2),
    "mean_fidelity": round(mean_fid, 8),
    "circuits": results_data,
    "reference_mean_fidelity": embedded["mean_qpu_fidelity"],
}
outpath = "_tmp_qpu_verification_result.json"
with open(outpath, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n  Results saved to {outpath}")
