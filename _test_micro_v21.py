#!/usr/bin/env python3
"""Quick validation of micro daemon v2.4 improvements."""
import time
import sys

print("=== VQPU Micro Daemon v2.4 Validation ===")

# 1. Imports
from l104_vqpu.micro_daemon import (
    VQPUMicroDaemon, MicroTaskPriority, TickMetrics, MICRO_DAEMON_VERSION,
    MicroTask, MicroTaskResult, MicroTaskStatus, MicroTelemetry, MicroDaemonConfig,
    TelemetryAnalytics, MICRO_TASK_TIMEOUT_S,
)
print(f"[OK] All imports successful (including v2.5: TelemetryAnalytics, MICRO_TASK_TIMEOUT_S)")
print(f"     VERSION: {MICRO_DAEMON_VERSION}")
assert MICRO_DAEMON_VERSION == "4.0.0", f"Expected 4.0.0, got {MICRO_DAEMON_VERSION}"

# 2. Package-level imports
from l104_vqpu import MicroTaskPriority as MTP2, TickMetrics as TM2, TelemetryAnalytics as TA2
assert MTP2 is MicroTaskPriority, "Package import mismatch"
assert TM2 is TickMetrics, "Package import mismatch"
assert TA2 is TelemetryAnalytics, "Package import mismatch"
print(f"[OK] Package-level imports verified")

# 3. Instantiate + start
d = VQPUMicroDaemon()
d.start()
time.sleep(0.5)
print(f"[OK] Daemon started, active={d._active}")

# 4. Self-test (12 probes)
st = d.self_test()
print(f"\n[SELF-TEST] {st['passed']}/{st['total']} passed in {st['elapsed_ms']}ms")
for t in st["tests"]:
    tag = "OK" if t["pass"] else "FAIL"
    detail = t.get("detail", t.get("error", ""))
    print(f"  [{tag}] {t['test']}: {detail}")

# 5. Status with v2.3/v2.4 fields
s = d.status()
print(f"\n[STATUS]")
print(f"  tick={s['tick']}, crash_count={s['crash_count']}")
print(f"  health={s['health_score']}, pass_rate={s['pass_rate']}")
print(f"  throttled_tasks={s['throttled_tasks']}")
print(f"  fail_streaks={s['fail_streaks']}")

# 6. task_stats()
ts = d.task_stats()
print(f"\n[TASK STATS] {len(ts)} task types:")
for name, stats in list(ts.items())[:5]:
    print(f"  {name}: {stats['count']}x mean={stats['mean_ms']}ms max={stats['max_ms']}ms")

# 7. v2.3: analytics()
report = d.analytics()
grade = report.get("grade", {}).get("grade", "?")
trend = report.get("trend", {}).get("direction", "?")
print(f"\n[ANALYTICS] grade={grade}, trend={trend}, anomalies={len(report.get('anomalies', []))}")
assert grade in ("A", "B", "C", "D", "F", "?"), f"Bad grade: {grade}"

# 8. v2.3: throttled_tasks()
throttled = d.throttled_tasks()
print(f"[THROTTLED] {len(throttled)} tasks throttled")

# 9. dump_metrics()
path = d.dump_metrics()
print(f"\n[DUMP] Metrics written to {path}")

# 10. reset_stats()
before = d.reset_stats()
print(f"\n[RESET] Before: tick={before['tick']}, tasks={before['total_tasks_run']}")
assert d._tick == 0, "Tick not reset"
assert d._total_tasks_run == 0, "Tasks not reset"
assert d._health_score == 1.0, "Health not reset"
print(f"[OK] Stats reset verified")

# 11. v2.4: Task timeout constant
print(f"\n[TIMEOUT] MICRO_TASK_TIMEOUT_S = {MICRO_TASK_TIMEOUT_S}")
assert MICRO_TASK_TIMEOUT_S == 5.0, f"Expected 5.0, got {MICRO_TASK_TIMEOUT_S}"
print(f"[OK] Task timeout constant verified")

# 12. Score check resonance
from l104_vqpu.micro_daemon import _micro_score_check
sc = _micro_score_check({"tick": 0})
print(f"\n[SCORE CHECK] resonance_286={sc['resonance_286']}, sacred_pass={sc['sacred_pass']}")
assert sc["sacred_pass"], f"Sacred resonance FAILED: alignment_error={sc['alignment_error']}"
print(f"[OK] Sacred resonance formula: (GOD_CODE/16)^phi ≈ 286")

# 13. Stop
d.stop()
print(f"\n[OK] Daemon stopped cleanly")

# Summary
if st["all_pass"]:
    print(f"\n=== ALL {st['total']} PROBES PASSED ===")
    sys.exit(0)
else:
    failed = [t["test"] for t in st["tests"] if not t["pass"]]
    print(f"\n=== {len(failed)} PROBES FAILED: {failed} ===")
    sys.exit(1)
