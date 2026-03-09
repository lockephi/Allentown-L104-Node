#!/usr/bin/env python3
"""Quick test: bridge auto-wiring in micro daemon v2.5."""
import time, sys

from l104_vqpu.micro_daemon import VQPUMicroDaemon

d = VQPUMicroDaemon()
print(f"Before start: bridge={d._bridge}")
d.start()
time.sleep(2)

print(f"After start:  bridge={d._bridge}")
print(f"Bridge active: {getattr(d._bridge, '_active', None)}")

st = d.self_test()
for t in st["tests"]:
    tag = "OK" if t["pass"] else "FAIL"
    detail = t.get("detail", t.get("error", ""))
    print(f"  [{tag}] {t['test']}: {detail}")

print(f"\nPassed: {st['passed']}/{st['total']}")

# Specifically check bridge_wiring
bw = next((t for t in st["tests"] if t["test"] == "bridge_wiring"), None)
if bw and bw["pass"]:
    print("\n=== BRIDGE WIRING: CONNECTED ===")
else:
    print(f"\n=== BRIDGE WIRING: FAILED — {bw} ===")

d.stop()
sys.exit(0 if st["all_pass"] else 1)
