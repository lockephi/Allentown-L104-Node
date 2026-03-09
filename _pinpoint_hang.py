#!/usr/bin/env python3
"""Pinpoint which line in get_status() hangs."""
import sys, os, time

class Suppress:
    def __enter__(self):
        self._o, self._e = sys.stdout, sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    def __exit__(self, *a):
        sys.stdout.close(); sys.stderr.close()
        sys.stdout, sys.stderr = self._o, self._e

LOG = open("_pinpoint_results.txt", "w")
def log(msg):
    LOG.write(msg + "\n"); LOG.flush()
    print(msg)

with Suppress():
    from l104_agi import agi_core

log("=== Pinpointing get_status() hang ===\n")

# Test each line in get_status individually
log("1. verify_survivor_algorithm...")
t = time.time()
try:
    from l104_persistence import verify_survivor_algorithm
    r = verify_survivor_algorithm()
    log(f"   PASS ({time.time()-t:.2f}s) -> {r}")
except Exception as e:
    log(f"   FAIL ({time.time()-t:.2f}s) -> {e}")

log("2. _autonomous_agi (cached ref)...")
t = time.time()
log(f"   Value: {agi_core._autonomous_agi} ({time.time()-t:.2f}s)")

log("3. _research_engine (cached ref)...")
t = time.time()
log(f"   Value: {agi_core._research_engine} ({time.time()-t:.2f}s)")

log("4. _read_consciousness_state()...")
t = time.time()
try:
    c = agi_core._read_consciousness_state()
    log(f"   PASS ({time.time()-t:.2f}s) -> keys={list(c.keys())}")
except Exception as e:
    log(f"   FAIL ({time.time()-t:.2f}s) -> {e}")

log("5. _dual_layer_engine (cached ref)...")
t = time.time()
log(f"   Value: {agi_core._dual_layer_engine} ({time.time()-t:.2f}s)")

log("6. _safe_evolution_stage()...")
t = time.time()
try:
    with Suppress():
        s = agi_core._safe_evolution_stage()
    log(f"   PASS ({time.time()-t:.2f}s) -> {s}")
except Exception as e:
    log(f"   FAIL ({time.time()-t:.2f}s) -> {e}")

log("7. truth resonance...")
t = time.time()
try:
    r = agi_core.truth['meta']['resonance']
    log(f"   PASS ({time.time()-t:.2f}s) -> {r}")
except Exception as e:
    log(f"   FAIL ({time.time()-t:.2f}s) -> {e}")

log("8. lattice_scalar...")
t = time.time()
try:
    r = agi_core.lattice_scalar
    log(f"   PASS ({time.time()-t:.2f}s) -> {r}")
except Exception as e:
    log(f"   FAIL ({time.time()-t:.2f}s) -> {e}")

log("9. identity_boundary.get_status()...")
t = time.time()
try:
    r = agi_core.identity_boundary.get_status()
    log(f"   PASS ({time.time()-t:.2f}s) -> keys={list(r.keys())[:5]}")
except Exception as e:
    log(f"   FAIL ({time.time()-t:.2f}s) -> {e}")

log("10. kernel_status()...")
t = time.time()
try:
    with Suppress():
        r = agi_core.kernel_status()
    log(f"   PASS ({time.time()-t:.2f}s)")
except Exception as e:
    log(f"   FAIL ({time.time()-t:.2f}s) -> {e}")

log("\n=== Now testing self_heal components ===\n")

log("11. asi_self_heal.proactive_scan()...")
t = time.time()
try:
    from l104_agi.constants import asi_self_heal
    with Suppress():
        r = asi_self_heal.proactive_scan()
    log(f"   PASS ({time.time()-t:.2f}s) -> status={r.get('status')}")
except Exception as e:
    log(f"   FAIL ({time.time()-t:.2f}s) -> {e}")

log("12. from l104_self_heal_master import main...")
t = time.time()
try:
    from l104_self_heal_master import main as run_master_heal
    log(f"   PASS import ({time.time()-t:.2f}s)")
except Exception as e:
    log(f"   FAIL ({time.time()-t:.2f}s) -> {e}")

log("\n=== Now testing self_evolve_codebase components ===\n")

log("13. from l104_derivation import DerivationEngine...")
t = time.time()
try:
    from l104_derivation import DerivationEngine
    log(f"   PASS import ({time.time()-t:.2f}s)")
except Exception as e:
    log(f"   FAIL ({time.time()-t:.2f}s) -> {e}")

log("14. DerivationEngine.derive_and_execute()...")
t = time.time()
try:
    with Suppress():
        r = DerivationEngine.derive_and_execute("ANALYZE_CORE_BOTTLENECKS")
    log(f"   PASS ({time.time()-t:.2f}s)")
except Exception as e:
    log(f"   FAIL ({time.time()-t:.2f}s) -> {e}")

log("15. get_autonomous_agi() (lazy import)...")
t = time.time()
try:
    with Suppress():
        r = agi_core.get_autonomous_agi()
    log(f"   PASS ({time.time()-t:.2f}s) -> {r is not None}")
except Exception as e:
    log(f"   FAIL ({time.time()-t:.2f}s) -> {e}")

log("\n=== Done ===")
LOG.close()
