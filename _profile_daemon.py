#!/usr/bin/env python3
"""Profile VQPU Daemon cycle speed."""
import time

# Profile daemon cycle
t0 = time.perf_counter()
from l104_vqpu.daemon import VQPUDaemonCycler
t1 = time.perf_counter()
print(f"Import daemon: {(t1-t0)*1000:.1f}ms")

d = VQPUDaemonCycler(interval=999)
t2 = time.perf_counter()
print(f"Init daemon:   {(t2-t1)*1000:.1f}ms")

# First cycle (cold)
result = d.run_cycle_now()
t3 = time.perf_counter()
print(f"First cycle:   {(t3-t2)*1000:.1f}ms")
print(f"  sims passed: {result.get('passed','?')}/{result.get('total','?')}")
print(f"  cycle_ms:    {result.get('elapsed_ms','?')}")

# Second cycle (warm - cached imports)
result2 = d.run_cycle_now()
t4 = time.perf_counter()
print(f"Second cycle:  {(t4-t3)*1000:.1f}ms")
print(f"  sims passed: {result2.get('passed','?')}/{result2.get('total','?')}")
print(f"  cycle_ms:    {result2.get('elapsed_ms','?')}")

# Third cycle
result3 = d.run_cycle_now()
t5 = time.perf_counter()
print(f"Third cycle:   {(t5-t4)*1000:.1f}ms")
print(f"  cycle_ms:    {result3.get('elapsed_ms','?')}")

# Per-sim breakdown from last cycle
print()
print("Per-sim breakdown (cycle 3):")
total_sim_ms = 0
for r in result3.get("results", []):
    name = r.get("name", "?")
    ms = r.get("elapsed_ms", 0)
    passed = r.get("passed", False)
    total_sim_ms += ms
    status = "PASS" if passed else "FAIL"
    err = f"  ERR: {r.get('error','')}" if not passed else ""
    print(f"  {name:45s} {ms:8.2f}ms  {status}{err}")

print(f"\n  Sum of sim times: {total_sim_ms:.1f}ms")
print(f"  Cycle wall time: {result3.get('elapsed_ms','?')}ms")
print(f"  Overhead:         {float(result3.get('elapsed_ms',0)) - total_sim_ms:.1f}ms")

# Profile harvest overhead separately
print("\n--- Harvest overhead (warm) ---")
t6 = time.perf_counter()
brain = d._harvest_brain_intelligence()
t7 = time.perf_counter()
print(f"  harvest_brain:     {(t7-t6)*1000:.2f}ms")

t7b = time.perf_counter()
evo = d._harvest_evolution_scores()
t8 = time.perf_counter()
print(f"  harvest_evolution: {(t8-t7b)*1000:.2f}ms")

t8b = time.perf_counter()
sub = d._harvest_subconscious()
t9 = time.perf_counter()
print(f"  harvest_subconsc:  {(t9-t8b)*1000:.2f}ms")

# Profile feed overhead
print("\n--- Feed overhead (per result) ---")
from l104_god_code_simulator.simulations.vqpu_findings import VQPU_FINDINGS_SIMULATIONS
entry = VQPU_FINDINGS_SIMULATIONS[0]
res = entry[1]()
t10 = time.perf_counter()
for _ in range(100):
    d._feed_coherence(res)
    d._feed_entropy(res)
t11 = time.perf_counter()
print(f"  100x feed (coh+ent): {(t11-t10)*1000:.2f}ms ({(t11-t10)*10:.3f}ms/pair)")

# Profile just the sims (sequential, no overhead)
print("\n--- Raw sequential sim timing ---")
t12 = time.perf_counter()
for entry in VQPU_FINDINGS_SIMULATIONS:
    entry[1]()
t13 = time.perf_counter()
print(f"  All {len(VQPU_FINDINGS_SIMULATIONS)} sims sequential: {(t13-t12)*1000:.1f}ms")
