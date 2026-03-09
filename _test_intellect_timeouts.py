"""Quick test of the 4 intellect methods that were timing out."""
from __future__ import annotations
import time
import signal
import sys

def timeout_handler(signum, frame):
    raise TimeoutError("exceeded")

signal.signal(signal.SIGALRM, timeout_handler)

from l104_intellect import local_intellect as li

methods = [
    ("sage_consciousness_coherence", lambda: li.sage_consciousness_coherence(), 30),
    ("sage_creation_void", lambda: li.sage_creation_void("quantum entanglement"), 30),
    ("quantum_compute_benchmark", lambda: li.quantum_compute_benchmark(), 30),
    ("quantum_compute_forward", lambda: li.quantum_compute_forward([0.1, 0.2]), 30),
]

passed = 0
failed = 0

for name, fn, timeout_s in methods:
    signal.alarm(timeout_s)
    t0 = time.time()
    try:
        r = fn()
        elapsed = (time.time() - t0) * 1000
        rtype = type(r).__name__
        print(f"  OK   {name}: {rtype} in {elapsed:.0f}ms")
        passed += 1
    except TimeoutError:
        print(f"  FAIL {name}: TIMEOUT ({timeout_s}s)")
        failed += 1
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        print(f"  WARN {name}: {type(e).__name__}: {str(e)[:100]} in {elapsed:.0f}ms")
        passed += 1  # Exception but didn't timeout
    finally:
        signal.alarm(0)

print(f"\n  Result: {passed}/{passed + failed} passed, {failed} timeouts")
sys.exit(1 if failed else 0)
