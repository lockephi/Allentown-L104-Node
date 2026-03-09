import json

import sys
report_file = sys.argv[1] if len(sys.argv) > 1 else "l104_debug_report_v2.json"
with open(report_file) as f:
    d = json.load(f)

s = d["summary"]
print(f"TOTAL: {s['passed']}/{s['total']} passed, {s['failed']} failed, {s['warnings']} warnings")
print(f"Time: {d.get('total_time_seconds', 0):.1f}s")
print()

print("BY ENGINE:")
for eng, stats in d.get("by_engine", {}).items():
    p = stats.get("passed", 0)
    fl = stats.get("failed", 0)
    w = stats.get("warnings", 0)
    t = p + fl + w
    icon = "OK" if fl == 0 else "!!"
    extra = ""
    if fl > 0 or w > 0:
        extra = f"  ({fl} fail, {w} warn)"
    print(f"  [{icon}] {eng:20s}  {p}/{t} passed{extra}")

print()
failures = [r for r in d.get("results", []) if r.get("status") == "FAIL"]
print(f"FAILURES ({len(failures)}):")
for r in failures:
    detail = str(r.get("detail", ""))[:120]
    print(f"  X  {r.get('test', '?')} -- {detail}")

print()
warnings = [r for r in d.get("results", []) if r.get("status") == "WARN"]
print(f"WARNINGS ({len(warnings)}):")
for r in warnings:
    detail = str(r.get("detail", ""))[:120]
    print(f"  !  {r.get('test', '?')} -- {detail}")
