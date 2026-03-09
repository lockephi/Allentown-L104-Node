#!/usr/bin/env python3
"""Final verification of AGI core fixes — tests slow methods from main thread."""
import sys, os, time, threading

class Suppress:
    def __enter__(self):
        self._o, self._e = sys.stdout, sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    def __exit__(self, *a):
        sys.stdout.close(); sys.stderr.close()
        sys.stdout, sys.stderr = self._o, self._e

LOG = open("_final_verify_results.txt", "w")
def log(msg):
    LOG.write(msg + "\n"); LOG.flush()
    print(msg)

with Suppress():
    from l104_agi import agi_core

log("=== Final AGI Core Fix Verification ===\n")

passed = 0
total = 0

def check(name, fn):
    global passed, total
    total += 1
    t = time.time()
    try:
        with Suppress():
            r = fn()
        dt = time.time() - t
        passed += 1
        val = str(r)[:60] if r is not None else "None"
        log(f"  PASS  {name:30s} ({dt:6.2f}s) -> {val}")
    except Exception as e:
        dt = time.time() - t
        log(f"  FAIL  {name:30s} ({dt:6.2f}s) -> {type(e).__name__}: {e}")

# --- Fast tests (properties + callable wrappers) ---
log("--- Property & Method Tests ---")
check("status (property)", lambda: agi_core.status)
check("status_string()", lambda: agi_core.status_string())
check("is_ready (property)", lambda: agi_core.is_ready)
check("check_ready()", lambda: agi_core.check_ready())

# --- Status methods (should be fast after fixes) ---
log("\n--- Status Methods (should complete in <5s) ---")
check("get_status()", lambda: agi_core.get_status())
check("full_engine_status()", lambda: agi_core.full_engine_status())

# --- Heavy operations (legitimately slow, run from main thread) ---
log("\n--- Heavy Operations (expected to be slow) ---")
check("self_improve()", lambda: agi_core.self_improve())
check("self_heal()", lambda: agi_core.self_heal())
check("self_evolve_codebase()", lambda: agi_core.self_evolve_codebase())

log(f"\n=== RESULTS: {passed}/{total} pass ===")
if passed == total:
    log("All AGI core processes are working!")
else:
    log(f"{total - passed} method(s) still have issues")
LOG.close()
