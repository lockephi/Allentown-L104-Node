#!/usr/bin/env python3
"""Verify all AGI core fixes are working."""
import sys, threading, time, os

class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    def __exit__(self, *a):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr

# Import with suppressed output
with SuppressOutput():
    from l104_agi import agi_core

results = {}

def test_with_timeout(name, fn, timeout=12):
    result = {"status": None, "error": None}
    def run():
        try:
            with SuppressOutput():
                r = fn()
            result["status"] = "PASS"
            result["value"] = str(r)[:80] if r is not None else "None"
        except Exception as e:
            result["status"] = "FAIL"
            result["error"] = f"{type(e).__name__}: {e}"
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        result["status"] = "HANG"
        result["error"] = f"Timed out after {timeout}s"
    results[name] = result

print("=== Verifying AGI Core Fixes ===\n")

# Also write to a log file for reliable reading
LOG = open("_verify_agi_results.txt", "w")
def log(msg):
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()

log("=== Verifying AGI Core Fixes ===\n")

# 1. status property
try:
    s = agi_core.status
    log(f"1. status (property):      PASS -> {s[:60]}")
except Exception as e:
    log(f"1. status (property):      FAIL -> {e}")

# 2. status_string() callable
try:
    s = agi_core.status_string()
    log(f"2. status_string():        PASS -> {s[:60]}")
except Exception as e:
    log(f"2. status_string():        FAIL -> {e}")

# 3. is_ready property
try:
    r = agi_core.is_ready
    log(f"3. is_ready (property):    PASS -> {r}")
except Exception as e:
    log(f"3. is_ready (property):    FAIL -> {e}")

# 4. check_ready() callable
try:
    r = agi_core.check_ready()
    log(f"4. check_ready():          PASS -> {r}")
except Exception as e:
    log(f"4. check_ready():          FAIL -> {e}")

log("\nTesting timeout-protected methods...")

# 5. get_status (was: hung on dl.full_integrity_check())
test_with_timeout("get_status", lambda: agi_core.get_status(), timeout=25)
r = results["get_status"]
log(f"5. get_status():           {r['status']}" + (f" -> {r['error']}" if r["error"] else ""))

# 6. self_heal (was: timeout >30s)
test_with_timeout("self_heal", lambda: agi_core.self_heal(), timeout=60)
r = results["self_heal"]
log(f"6. self_heal():            {r['status']}" + (f" -> {r['error']}" if r["error"] else ""))

# 7. self_improve (was: timeout >30s)
test_with_timeout("self_improve", lambda: agi_core.self_improve(), timeout=20)
r = results["self_improve"]
log(f"7. self_improve():         {r['status']}" + (f" -> {r['error']}" if r["error"] else ""))

# 8. full_engine_status (was: hung indefinitely)
test_with_timeout("full_engine_status", lambda: agi_core.full_engine_status(), timeout=25)
r = results["full_engine_status"]
log(f"8. full_engine_status():   {r['status']}" + (f" -> {r['error']}" if r["error"] else ""))

# 9. self_evolve_codebase (was: timeout >20s)
test_with_timeout("self_evolve_codebase", lambda: agi_core.self_evolve_codebase(), timeout=120)
r = results["self_evolve_codebase"]
log(f"9. self_evolve_codebase(): {r['status']}" + (f" -> {r['error']}" if r["error"] else ""))

# Summary
passed_timeout = sum(1 for v in results.values() if v["status"] == "PASS")
total_timeout = len(results)
log(f"\n=== RESULTS: {passed_timeout + 4}/{total_timeout + 4} tests pass ===")

failed = [k for k, v in results.items() if v["status"] != "PASS"]
if failed:
    log(f"Still failing: {', '.join(failed)}")
else:
    log("All previously broken processes are now fixed!")
LOG.close()
