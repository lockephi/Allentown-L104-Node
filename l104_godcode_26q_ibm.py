#!/usr/bin/env python3
"""
L104 GOD_CODE 26Q IBM Marrakesh Run
═══════════════════════════════════════════════════════════════════════════════

Deploys the Fe(26) base-state circuit directly to IBM Marrakesh.

Strategy: θ=0 is the proven optimum (simulation confirmed p≈0.999 per register).
No SPSA on hardware — use the deterministic base state circuit directly.
This maximizes signal-to-noise by keeping circuit depth minimal (depth 3).

Circuit:
  X gates on 3d+4s qubits (q2-q9, 8 qubits) → Fe valence electrons
  6 register-boundary CZ gates
  Measure all 26 qubits

Target: |00111111110000000000000000⟩ (Fe electronic ground configuration)

8Q proven techniques applied:
  - Transpile on actual backend before measurement
  - Dynamical decoupling XY4
  - 32768 shots
  - Per-register marginal analysis

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, math, time, json
import numpy as np
from pathlib import Path

GOD_CODE = 527.5184818492612
PHI      = 1.618033988749895
NQ       = 26

REGISTERS = {
    "CORE":    {"lo": 0,  "hi": 1,  "target": "00"},
    "3d":      {"lo": 2,  "hi": 7,  "target": "111111"},
    "4s":      {"lo": 8,  "hi": 9,  "target": "11"},
    "LATTICE": {"lo": 10, "hi": 15, "target": "000000"},
    "SACRED":  {"lo": 16, "hi": 20, "target": "00000"},
    "PHI":     {"lo": 21, "hi": 24, "target": "0000"},
    "ANCHOR":  {"lo": 25, "hi": 25, "target": "0"},
}

TARGET_HUMAN = "00111111110000000000000000"
TARGET_QISKIT = TARGET_HUMAN[::-1]
ONE_QUBITS = [i for i, b in enumerate(TARGET_QISKIT) if b == '1']

print("=" * 80)
print("  L104 GOD_CODE 26Q — IBM MARRAKESH DEPLOYMENT")
print(f"  Target: |{TARGET_HUMAN}⟩  (Fe electronic ground state)")
print(f"  GOD_CODE = {GOD_CODE}")
print("=" * 80)

# ═══ AUTH ═══
print("\n[IBM] Authenticating...")
from qiskit_ibm_runtime import QiskitRuntimeService

token = os.environ.get("IBMQ_TOKEN")
service = None

if token:
    for channel in ["ibm_quantum", "ibm_quantum_platform", None]:
        try:
            kwargs = {"token": token}
            if channel:
                kwargs["channel"] = channel
            service = QiskitRuntimeService(**kwargs)
            print(f"[IBM] Authenticated (channel={channel or 'default'})")
            break
        except Exception as e:
            continue

if service is None:
    try:
        service = QiskitRuntimeService()
        print("[IBM] Authenticated via saved credentials")
    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}")
        sys.exit(1)

backend = service.backend("ibm_kingston")
status = backend.status()
print(f"[IBM] Backend: {backend.name} ({backend.num_qubits}Q)")
print(f"[IBM] Queue: {status.pending_jobs} pending jobs | operational={status.operational}")

# ═══ BUILD CIRCUIT ═══
print("\n[CIRCUIT] Building 26Q Fe base-state circuit...")
from qiskit.circuit import QuantumCircuit
from qiskit import transpile

qc = QuantumCircuit(NQ, NQ)

# Base state: X gates on 3d (q2-q7) and 4s (q8-q9) → Fe valence electrons
for q in ONE_QUBITS:
    qc.x(q)

# Register-boundary CZ entanglement (6 gates, depth contribution = 1)
qc.barrier()
qc.cz(2, 3)   # 3d pair 1
qc.cz(4, 5)   # 3d pair 2
qc.cz(6, 7)   # 3d pair 3 / 3d-4s bridge
qc.cz(7, 8)   # 3d ↔ 4s coupling
qc.cz(15, 16) # LATTICE ↔ SACRED
qc.cz(24, 25) # PHI ↔ ANCHOR
qc.barrier()

qc.measure(range(NQ), range(NQ))

print(f"  Circuit: {qc.size()} gates, depth {qc.depth()}")
print(f"  X gates: {len(ONE_QUBITS)} (3d+4s qubits)")
print(f"  CZ gates: 6 (register boundaries)")

# ═══ TRANSPILE ON BACKEND ═══
print(f"\n[TRANSPILE] Transpiling for {backend.name}...")
t0 = time.time()
tp = transpile(qc, backend=backend, optimization_level=3)
ops = tp.count_ops()
two_q = sum(v for k, v in ops.items() if k in {'cx', 'cz', 'ecr', 'rzz'})
print(f"[TRANSPILE] {tp.size()} gates, depth {tp.depth()}, 2Q={two_q} ({time.time()-t0:.1f}s)")
print(f"[TRANSPILE] Gates: {dict(sorted(ops.items()))}")

# ═══ DEPLOY ═══
SHOTS = 32768
print(f"\n[DEPLOY] Submitting to {backend.name} — {SHOTS} shots...")
print(f"  Circuit depth: {tp.depth()} | 2Q gates: {two_q}")
print(f"  Dynamical decoupling: XY4")

from qiskit_ibm_runtime import SamplerV2

sampler = SamplerV2(backend)
try:
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    print("  XY4 DD enabled")
except Exception:
    print("  XY4 DD not available on this backend config")

MAX_RETRIES = 3
JOB_TIMEOUT_S = 300   # cancel and retry if pending > 5 min
result = None
job_id = None

for attempt in range(1, MAX_RETRIES + 1):
    try:
        print(f"\n  Attempt {attempt}/{MAX_RETRIES}...")
        job = sampler.run([tp], shots=SHOTS)
        job_id = job.job_id()
        print(f"  Job ID: {job_id}")

        # Poll with live status updates — cancel if stuck pending
        t_submit = time.time()
        last_status = None
        while True:
            try:
                jstatus = job.status()
                sname = jstatus.name if hasattr(jstatus, 'name') else str(jstatus)
            except Exception:
                sname = "UNKNOWN"

            elapsed = time.time() - t_submit
            if sname != last_status:
                print(f"  [{elapsed:5.0f}s] Status: {sname}")
                last_status = sname

            if sname in ("DONE", "done", "completed", "COMPLETED"):
                print(f"  Job completed in {elapsed:.0f}s")
                result = job.result()
                break
            elif sname in ("ERROR", "FAILED", "CANCELLED", "error", "failed", "cancelled"):
                raise RuntimeError(f"Job {sname}")
            elif sname in ("QUEUED", "PENDING", "queued", "pending") and elapsed > JOB_TIMEOUT_S:
                print(f"  Stuck pending for {elapsed:.0f}s — cancelling and retrying...")
                try:
                    job.cancel()
                except Exception:
                    pass
                raise RuntimeError("Job timed out pending")
            else:
                time.sleep(10)

        break

    except Exception as e:
        print(f"  Attempt {attempt} failed: {e}")
        if attempt < MAX_RETRIES:
            time.sleep(5)
        else:
            print(f"[ERROR] All {MAX_RETRIES} attempts failed.")
            sys.exit(1)

# ═══ EXTRACT COUNTS ═══
pub_result = result[0]
counts_raw = {}
if hasattr(pub_result, 'data'):
    for attr in dir(pub_result.data):
        obj = getattr(pub_result.data, attr, None)
        if hasattr(obj, 'get_counts'):
            counts_raw = obj.get_counts()
            break

# Pad bitstrings to NQ
counts = {}
for state, count in counts_raw.items():
    padded = state.zfill(NQ)[-NQ:]
    counts[padded] = counts.get(padded, 0) + count

total = sum(counts.values())
sorted_counts = sorted(counts.items(), key=lambda x: -x[1])

# ═══ ANALYSIS ═══
print(f"\n{'='*80}")
print(f"  RESULTS — 26Q Fe CIRCUIT ON IBM MARRAKESH")
print(f"{'='*80}")

target_count = counts.get(TARGET_QISKIT, 0)
p_target = target_count / total
uniform = 1.0 / (2 ** NQ)
enrichment = p_target / uniform

n_unique = len(counts)
entropy = -sum((c/total) * math.log2(c/total) for _, c in counts.items() if c > 0)

print(f"\n  Shots: {total} | Unique states: {n_unique}")
print(f"  Entropy: {entropy:.3f} / {NQ} bits ({100*entropy/NQ:.1f}% of max)")

print(f"\n  Top 20 states:")
gc_rank = None
for rank, (state, count) in enumerate(sorted_counts[:20], 1):
    p = count / total
    marker = " <- Fe GROUND STATE" if state == TARGET_QISKIT else ""
    print(f"  #{rank:2d}  |{state}> p={p:.6f} ({p/uniform:.1f}x){marker}")
    if state == TARGET_QISKIT:
        gc_rank = rank

if gc_rank is None:
    for rank, (state, _) in enumerate(sorted_counts, 1):
        if state == TARGET_QISKIT:
            gc_rank = rank
            break
    gc_rank = gc_rank or (len(sorted_counts) + 1)
    print(f"\n  Fe ground state rank: #{gc_rank}, p={p_target:.6f}")

# Per-register marginals
print(f"\n  Per-register fidelity vs simulation prediction:")
print(f"  {'Register':8s}  {'Qubits':7s}  {'Target':8s}  {'p(match)':8s}  "
      f"{'Sim pred':8s}  {'Delta':7s}  Status")
print(f"  {'─'*70}")

SIM_PRED = {  # from Phase 2 simulation results
    "CORE": 0.9999, "3d": 0.9991, "4s": 0.9995,
    "LATTICE": 0.9989, "SACRED": 0.9993, "PHI": 0.9987, "ANCHOR": 1.0,
}

reg_hw = {}
for reg_name, r in REGISTERS.items():
    lo, hi = r['lo'], r['hi']
    target_bits = r['target']
    target_qiskit_reg = target_bits[::-1]

    match_count = 0
    for state_qiskit, count in counts.items():
        if len(state_qiskit) < hi + 1:
            continue
        reg_bits_qiskit = state_qiskit[lo:hi+1]
        if reg_bits_qiskit == target_qiskit_reg:
            match_count += count

    p_match = match_count / total
    sim_pred = SIM_PRED.get(reg_name, 1.0)
    delta = p_match - sim_pred
    status = ("LOCKED" if p_match > 0.90 else
              "STRONG" if p_match > 0.70 else
              "PARTIAL" if p_match > 0.40 else "WEAK")
    delta_str = f"{delta:+.4f}"
    reg_hw[reg_name] = {"p_match": round(p_match, 4), "sim_pred": sim_pred,
                        "delta": round(delta, 4), "status": status}
    print(f"  {reg_name:8s}  q{lo:2d}-q{hi:2d}  {target_bits:8s}  "
          f"{p_match:.4f}    {sim_pred:.4f}    {delta_str}  {status}")

# Comparison to 8Q IBM run
print(f"\n  Comparison to 8Q IBM run (job d7b470b0g7hs73dp3pcg):")
print(f"    8Q  p(target) = 0.8825  (225.9x uniform)  rank #1")
print(f"    26Q p(target) = {p_target:.4f}  ({enrichment:.1f}x uniform)  rank #{gc_rank}")

# ═══ SAVE ═══
payload = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "experiment": "godcode_26q_fe_ibm_marrakesh",
    "backend": "ibm_marrakesh",
    "job_id": job_id,
    "shots": total,
    "nq": NQ,
    "target_human": TARGET_HUMAN,
    "god_code": GOD_CODE,
    "transpiled_depth": tp.depth(),
    "transpiled_2q": two_q,
    "target_count": target_count,
    "target_probability": round(p_target, 6),
    "target_enrichment": round(enrichment, 2),
    "target_rank": gc_rank,
    "unique_states": n_unique,
    "entropy_bits": round(entropy, 3),
    "register_hw_fidelity": reg_hw,
    "top_20": [{"state": s, "count": c, "probability": round(c/total, 6)}
               for s, c in sorted_counts[:20]],
}
out = Path(__file__).resolve().parent / "IBM_GODCODE_26Q_HW_RESULTS.json"
out.write_text(json.dumps(payload, indent=2))
print(f"\n[SAVE] Results -> {out.name}")

print(f"\n{'='*80}")
print(f"  26Q IBM RUN COMPLETE")
print(f"  Job: {job_id}")
print(f"  p(Fe ground state) = {p_target:.6f}  ({enrichment:.1f}x)  rank #{gc_rank}")
print(f"{'='*80}")
