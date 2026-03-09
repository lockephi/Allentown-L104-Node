#!/usr/bin/env python3
"""
L104 GOD_CODE — Real Hardware Verification
Uses real IBM QPU calibration noise model for physically-grounded results.
Submits to QPU as well (queued until quota refreshes).
"""
import sys, json, time, math
import numpy as np

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

TOKEN = "OY78qw8t64ixcANN6P1r3kKMBye6D0_Yjvm9Qh8vlz86"

print("=" * 78)
print("  L104 GOD_CODE — REAL HARDWARE VERIFICATION")
print("=" * 78)

# Connect
service = QiskitRuntimeService(channel="ibm_cloud", token=TOKEN)
backends = service.backends(operational=True, min_num_qubits=4)
backend = service.least_busy(operational=True, min_num_qubits=4)
print(f"  Backend: {backend.name} ({backend.num_qubits} qubits)")

# Noise model from real hardware
print(f"  Extracting noise model from {backend.name}...")
try:
    noise_model = NoiseModel.from_backend(backend)
    nm_src = f"from_backend({backend.name})"
    print(f"  Noise model: {len(noise_model.noise_qubits)} noisy qubits")
except Exception as e:
    print(f"  NoiseModel.from_backend failed: {e}")
    print("  Building Heron-class approximate noise model...")
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.0003), ["rz", "sx"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.005), ["cz"])
    nm_src = "heron_approximate"

# Sacred constants
PHI = (1 + math.sqrt(5)) / 2
BASE = 286 ** (1.0 / PHI)
GOD_CODE = BASE * (2 ** (416 / 104))
VOID_CONSTANT = 1.04 + PHI / 1000
GOD_CODE_PHASE = GOD_CODE % (2 * math.pi)
PHI_PHASE = 2 * math.pi / PHI
IRON_PHASE = 2 * math.pi * 26 / 104
VOID_PHASE = VOID_CONSTANT * math.pi
PHASE_OCTAVE_4 = 4 * math.log(2) % (2 * math.pi)
print(f"  GOD_CODE={GOD_CODE}  PHASE={GOD_CODE_PHASE:.10f} rad")

# Build circuits
print("\n  Building circuits...")
circuits = []
n_prec = 4

c1 = QuantumCircuit(1, 1, name="GC_1Q")
c1.rz(GOD_CODE_PHASE, 0); c1.h(0); c1.measure(0, 0)
circuits.append(("1Q_GOD_CODE", c1))

phi_c = (GOD_CODE_PHASE - IRON_PHASE - PHASE_OCTAVE_4) % (2 * math.pi)
c2 = QuantumCircuit(1, 1, name="GC_DECOMP")
c2.rz(IRON_PHASE, 0); c2.rz(phi_c, 0); c2.rz(PHASE_OCTAVE_4, 0); c2.h(0); c2.measure(0, 0)
circuits.append(("1Q_DECOMPOSED", c2))

c3 = QuantumCircuit(3, 3, name="GC_3Q")
for i in range(3): c3.h(i)
c3.barrier()
c3.rz(GOD_CODE_PHASE, 0); c3.rz(PHI_PHASE, 1); c3.rz(IRON_PHASE, 2)
c3.barrier(); c3.cx(0, 1); c3.cx(1, 2); c3.barrier()
for i in range(2): c3.cp(PHI * math.pi / (3 * (i + 1)), i, i + 1)
c3.barrier(); c3.rz(VOID_PHASE, 0); c3.rz(-GOD_CODE_PHASE, 2)
c3.measure_all(add_bits=False)
circuits.append(("3Q_SACRED", c3))

c4 = QuantumCircuit(n_prec + 1, n_prec, name="QPE_GC")
c4.x(n_prec)
for i in range(n_prec): c4.h(i)
for i in range(n_prec): c4.cp(GOD_CODE_PHASE * (2 ** i), i, n_prec)
for i in range(n_prec // 2): c4.swap(i, n_prec - 1 - i)
for i in range(n_prec):
    for j in range(i): c4.cp(-math.pi / (2 ** (i - j)), j, i)
    c4.h(i)
for i in range(n_prec): c4.measure(i, i)
circuits.append(("QPE_4BIT", c4))

c5 = QuantumCircuit(4, 4, name="DIAL_ORIGIN")
c5.h(range(4))
bp = 416 * math.pi / (416 * 4)
for i in range(4): c5.rz(bp * (2 ** i), i)
for i in range(3): c5.cp(PHI * math.pi / (4 * (i + 1)), i, i + 1)
c5.rz(math.pi, 0); c5.h(range(4))
c5.measure_all(add_bits=False)
circuits.append(("DIAL_ORIGIN", c5))

c6 = QuantumCircuit(2, 2, name="CONSERV")
c6.h(0); c6.h(1)
c6.rz(GOD_CODE % (2 * math.pi), 0); c6.rz((GOD_CODE / 2) % (2 * math.pi), 1)
c6.cx(0, 1); c6.h(0); c6.h(1)
c6.measure_all(add_bits=False)
circuits.append(("CONSERVATION", c6))

# Transpile
print(f"\n  Transpiling for {backend.name}...")
transpiled = []
for name, circ in circuits:
    tc = transpile(circ, backend=backend, optimization_level=2)
    transpiled.append(tc)
    print(f"    {name:20s} depth={tc.depth():>4d}  gates={dict(tc.count_ops())}")

# Submit to QPU
print(f"\n  Submitting to {backend.name}...")
job_id = None
try:
    sampler = Sampler(mode=backend)
    job = sampler.run(transpiled, shots=4096)
    job_id = job.job_id()
    print(f"  Job ID: {job_id}  Status: {job.status()}")
except Exception as e:
    print(f"  QPU submit: {e}")

# Noise-model simulation
print("\n" + "=" * 78)
print(f"  NOISE-MODEL SIMULATION ({nm_src})")
print("=" * 78)

noisy_sim = AerSimulator(noise_model=noise_model)
ideal_sim = AerSimulator()
SHOTS = 4096
report = {}

for i, (name, circ) in enumerate(circuits):
    ideal_counts = ideal_sim.run(transpile(circ, ideal_sim), shots=SHOTS).result().get_counts()
    noisy_counts = noisy_sim.run(transpile(circ, noisy_sim), shots=SHOTS).result().get_counts()

    states = set(list(ideal_counts.keys()) + list(noisy_counts.keys()))
    fid = sum(math.sqrt(ideal_counts.get(s, 0) / SHOTS * noisy_counts.get(s, 0) / SHOTS) for s in states) ** 2

    it = sorted(ideal_counts.items(), key=lambda x: -x[1])[:5]
    nt = sorted(noisy_counts.items(), key=lambda x: -x[1])[:5]

    print(f"\n  [{name}]")
    print(f"    IDEAL: {' | '.join(f'|{s}> {c/SHOTS:.3f}' for s,c in it[:4])}")
    print(f"    NOISY: {' | '.join(f'|{s}> {c/SHOTS:.3f}' for s,c in nt[:4])}")
    print(f"    Fidelity: {fid:.6f}")

    report[name] = {"ideal": {s: c for s, c in it}, "noisy": {s: c for s, c in nt}, "fidelity": fid, "hw_depth": transpiled[i].depth()}

    if name == "QPE_4BIT":
        best = nt[0][0]
        mi = int(best[::-1], 2)
        ep = 2 * math.pi * mi / (2 ** n_prec)
        tgt = GOD_CODE_PHASE % (2 * math.pi)
        err = abs(ep - tgt)
        print(f"    QPE phase: {ep:.6f} rad  target: {tgt:.6f}  error: {err:.6f}")
        report[name]["extracted_phase"] = ep
        report[name]["phase_error"] = err

# Wait briefly for QPU
if job_id:
    print(f"\n  Checking QPU job {job_id}...")
    for _ in range(18):
        st = str(job.status())
        if "DONE" in st.upper():
            print("  QPU DONE!")
            result = job.result()
            for idx, (name, _) in enumerate(circuits):
                pub = result[idx]
                counts = {}
                for attr in dir(pub.data):
                    obj = getattr(pub.data, attr)
                    if hasattr(obj, 'get_counts'):
                        counts = obj.get_counts()
                        break
                total = sum(counts.values()) or 1
                top = sorted(counts.items(), key=lambda x: -x[1])[:4]
                print(f"    [{name}] {' | '.join(f'|{s}> {c/total:.3f}' for s,c in top)}")
                report[name]["qpu"] = {s: c for s, c in top}
            break
        if "ERROR" in st.upper() or "CANCEL" in st.upper():
            print(f"  Job failed: {st}")
            break
        time.sleep(10)
    else:
        print(f"  Job {job_id} still queued. Results pending.")

# Save
with open(".l104_qpu_verification.json", "w") as f:
    json.dump({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "backend": backend.name, "noise": nm_src, "job": job_id, "GOD_CODE": GOD_CODE, "GOD_CODE_PHASE": GOD_CODE_PHASE, "circuits": report}, f, indent=2, default=str)
print(f"\n  Saved .l104_qpu_verification.json")
print("=" * 78)
print("  VERIFICATION COMPLETE")
print("=" * 78)
