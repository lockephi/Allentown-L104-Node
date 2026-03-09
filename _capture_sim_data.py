#!/usr/bin/env python3
"""Capture simulation data from God Code Simulator + daemon telemetry."""
import json, time, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

t0 = time.time()

# ── 1. God Code Simulator data ──────────────────────────────────────────
from l104_god_code_simulator import god_code_simulator

report = god_code_simulator.run_all()
print(f"23 simulations in {time.time()-t0:.2f}s")

for cat, results in report.items():
    if isinstance(results, dict):
        for name, r in results.items():
            if hasattr(r, "fidelity"):
                gc = getattr(r, "god_code_alignment", "N/A")
                print(f"  {cat}/{name}: fidelity={r.fidelity:.6f} gc_alignment={gc}")

# ── 2. Parametric sweeps ────────────────────────────────────────────────
print("\n=== Parametric Sweeps ===")
sweep = god_code_simulator.parametric_sweep("dial_a", start=0, stop=8)
print(f"Dial sweep: {len(sweep)} points")

noise_sweep = god_code_simulator.parametric_sweep("noise")
print(f"Noise sweep: {len(noise_sweep)} points")

# ── 3. Adaptive optimization ────────────────────────────────────────────
print("\n=== Adaptive Optimization ===")
opt = god_code_simulator.adaptive_optimize(target_fidelity=0.999, nq=5, depth=4)
print(f"Best fidelity: {opt.get('best_fidelity', '?')}")
print(f"Best strategy: {opt.get('best_strategy', '?')}")

nr = god_code_simulator.optimize_noise_resilience(nq=4, noise_level=0.02)
print(f"Noise resilience: best={nr.get('best_strategy', '?')}, f={nr.get('best_fidelity', '?')}")

# ── 4. Daemon bridge data ──────────────────────────────────────────────
print("\n=== Swift Daemon Bridge ===")
bridge_result_path = "/tmp/l104_bridge/outbox"
if os.path.isdir(bridge_result_path):
    results = os.listdir(bridge_result_path)
    print(f"Daemon outbox: {len(results)} results")
    for rf in sorted(results)[:3]:
        fp = os.path.join(bridge_result_path, rf)
        with open(fp) as f:
            data = json.load(f)
        print(f"  {rf}:")
        print(f"    backend={data.get('backend')}")
        print(f"    execution_time_ms={data.get('execution_time_ms', '?')}")
        print(f"    sacred_score={data.get('meta_sacred_alignment_sacred_score', '?')}")
        print(f"    three_engine={data.get('three_engine', {})}")

# ── 5. Latest telemetry ────────────────────────────────────────────────
telem_path = "/tmp/l104_bridge/telemetry"
if os.path.isdir(telem_path):
    sessions = sorted(os.listdir(telem_path))
    if sessions:
        latest = os.path.join(telem_path, sessions[-1])
        with open(latest) as f:
            t = json.load(f)
        print(f"\nLatest telemetry ({sessions[-1]}):")
        print(f"  throughput: {t.get('throughput_jobs_per_sec', '?')} jobs/sec")
        print(f"  success_rate: {t.get('jobs_success_rate', '?')}")
        cap = t.get("v6_metrics", {})
        print(f"  max_qubits: {cap.get('capacity_max_qubits', '?')}")
        print(f"  batch_limit: {cap.get('capacity_batch_limit', '?')}")

print(f"\nTotal time: {time.time()-t0:.2f}s")
