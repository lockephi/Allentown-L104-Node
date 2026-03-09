"""Benchmark VQPU Daemon v13.4 speed improvements."""
import time
import sys

# Clear any cached modules to simulate fresh start
for mod in list(sys.modules):
    if 'vqpu' in mod or 'god_code_simulator' in mod:
        del sys.modules[mod]

# Force reload of the daemon's cache
import l104_vqpu.daemon as dm
dm._cached_findings_sims = None

from l104_vqpu.daemon import VQPUDaemonCycler

cycler = VQPUDaemonCycler(interval=999)

# Run 3 cycles to measure cold + warm performance
results = []
for i in range(3):
    t0 = time.monotonic()
    r = cycler._run_findings_cycle()
    elapsed = (time.monotonic() - t0) * 1000
    sims = r.get('total', 0)
    passed = r.get('passed', 0)
    results.append(elapsed)
    print(f"Cycle {i+1}: {elapsed:,.1f}ms  ({sims} sims, {passed} passed)")
    # Show individual sim timings on first cycle
    if i == 0 and 'results' in r:
        for sr in sorted(r['results'], key=lambda x: x.get('elapsed_ms', 0), reverse=True):
            ms = sr.get('elapsed_ms', 0)
            nm = sr.get('name', '?')
            ok = 'PASS' if sr.get('passed') else 'FAIL'
            print(f"  {ok} {nm}: {ms:.1f}ms")

print()
print(f"Cold cycle:  {results[0]:,.1f}ms")
print(f"Warm cycles: {results[1]:,.1f}ms, {results[2]:,.1f}ms")
print(f"Avg warm:    {(results[1]+results[2])/2:,.1f}ms")

# Compare with baseline (measured earlier)
baseline_warm = 4726.0  # v13.3 warm cycle (all 11 sims + ThreadPoolExecutor)
avg_warm = (results[1] + results[2]) / 2
speedup = baseline_warm / avg_warm if avg_warm > 0 else 0
print(f"\nBaseline (v13.3 warm): {baseline_warm:,.0f}ms")
print(f"New (v13.4 warm):      {avg_warm:,.1f}ms")
print(f"Speedup:               {speedup:.1f}x")
