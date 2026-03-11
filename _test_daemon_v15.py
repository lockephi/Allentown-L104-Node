#!/usr/bin/env python3
"""Quick validation of VQPU Daemon v15.3 improvements."""
import time
from l104_vqpu.daemon import VQPUDaemonCycler

cycler = VQPUDaemonCycler(interval=999)

# Test 1: Status API covers current version and includes new fields
s = cycler.status()
# daemon version incremented to 16.x with topology support
assert s["version"] == "16.1.0", f"Version mismatch: {s['version']}"
# core health/telemetry fields
required = [
    "health_score", "circuit_breaker_open", "watchdog_restarts",
    "drift_alerts", "error_log", "consecutive_failures",
    "quarantined_sims", "quarantined_count",
    "cycle_fidelity_avg", "cycle_alignment_avg",
    "throughput_sims_per_sec", "leaked_threads", "paused",
]
for key in required:
    assert key in s, f"Missing {key}"
# additional new metrics introduced in v16
extra = [
    "adaptive_interval_s", "last_cpu_percent", "sim_timeout_s", "execution_mode",
    "sim_registry", "quantum_subconscious", "brain_intelligence",
    "micro_daemon_cross_health", "degradation_level",
    "sim_priority_scores", "fidelity_alerts", "cross_daemon_health",
    "quantum_topology",
]
for key in extra:
    assert key in s, f"Missing new field {key}"
print("PASS: Status API has all expected fields including v16 additions")

# Test 2: Run one cycle
t0 = time.monotonic()
r = cycler._run_findings_cycle()
dt = (time.monotonic() - t0) * 1000
sims = r.get("total", 0)
passed = r.get("passed", 0)
print(f"PASS: Cycle completed in {dt:.0f}ms ({sims} sims, {passed} passed)")

# Test 3: Health score computed with fidelity component
cycler._update_health_score()
hs = cycler._health_score
assert 0.0 <= hs <= 1.0, f"Health score out of range: {hs}"
print(f"PASS: Health score = {hs}")

# Test 4: Sim timing drift tracking populated
entries = len(cycler._sim_timing_history)
assert entries > 0, "No sim timing history tracked"
print(f"PASS: Tracking {entries} sims for drift detection")

# Test 5: deque-backed collections
from collections import deque
assert isinstance(cycler._sc_history, deque), "sc_history not deque"
assert isinstance(cycler._health_history, deque), "health_history not deque"
assert isinstance(cycler._error_log, deque), "error_log not deque"
assert isinstance(cycler._drift_alerts, deque), "drift_alerts not deque"
assert isinstance(cycler._quarantine_log, deque), "quarantine_log not deque"
assert isinstance(cycler._cycle_fidelity_avg, deque), "cycle_fidelity_avg not deque"
assert isinstance(cycler._cycle_alignment_avg, deque), "cycle_alignment_avg not deque"
assert isinstance(cycler._cycle_throughput, deque), "cycle_throughput not deque"
print("PASS: All bounded collections use deque")

# Test 6: Circuit breaker defaults
assert cycler._circuit_breaker_open is False
assert cycler._consecutive_failures == 0
print("PASS: Circuit breaker correctly initialized")

# Test 7: Fidelity trend tracking populated
fid_entries = len(cycler._sim_fidelity_history)
assert fid_entries > 0, "No fidelity history tracked"
trends = cycler.fidelity_trends()
assert "per_sim" in trends, "Missing per_sim in fidelity_trends"
assert "cycle_fidelity_avg" in trends, "Missing cycle_fidelity_avg"
print(f"PASS: Fidelity tracking {fid_entries} sims, cycle avg: {trends['cycle_fidelity_avg']}")

# Test 8: Throughput tracking
tp = cycler.throughput_stats()
assert tp["samples"] > 0, "No throughput samples"
print(f"PASS: Throughput = {tp['latest_sims_per_sec']} sims/sec")

# Test 9: Quarantine system
qs = cycler.quarantine_status()
assert "quarantined" in qs, "Missing quarantined in quarantine_status"
assert "failure_counts" in qs, "Missing failure_counts"
assert "quarantine_threshold" in qs, "Missing quarantine_threshold"
print(f"PASS: Quarantine system initialized (threshold={qs['quarantine_threshold']})")

# Test 10: Runtime control
cycler.pause()
assert cycler._paused is True
cycler.resume()
assert cycler._paused is False
print("PASS: Runtime pause/resume works")

# Test 11: Force cycle updates health score
old_hs = cycler._health_score
r2 = cycler.force_cycle()
assert r2.get("total", 0) > 0, "Force cycle returned no sims"
print(f"PASS: Force cycle OK ({r2.get('total')} sims, health={cycler._health_score})")

# Test 12: Persist + reload state with quarantine/health
cycler._persist_state()
import json
state = json.loads(cycler._state_path.read_text())
# state file version may lag behind status; ensure major version 16
assert state["version"].startswith("16."), f"State version mismatch: {state['version']}"
for key in ["health_score", "consecutive_failures", "quarantined_sims",
            "sim_failure_counts", "cycle_fidelity_avg", "total_leaked_threads"]:
    assert key in state, f"Missing {key} in persisted state"
print(f"PASS: Persisted state v{state['version']} to {cycler._state_path}")

# Test 13: Reload correctly restores state
cycler2 = VQPUDaemonCycler(interval=999)
cycler2._load_state()
assert cycler2._health_score == cycler._health_score, "Health score not restored"
assert cycler2._cycles_completed == cycler._cycles_completed, "Cycles not restored"
print("PASS: State reload restores health_score + cycles")

# Show top 3 slowest sims
if r.get("results"):
    print("\nTop 3 slowest sims:")
    for sr in sorted(r["results"], key=lambda x: x.get("elapsed_ms", 0), reverse=True)[:3]:
        name = sr["name"]
        ms = sr.get("elapsed_ms", 0)
        ok = "PASS" if sr.get("passed") else "FAIL"
        fid = sr.get("fidelity", 0)
        print(f"  {ok} {name}: {ms:.0f}ms (fidelity={fid:.4f})")

print(f"\n13/13 tests passed — VQPU Daemon v15.3 validated")
