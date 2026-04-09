#!/usr/bin/env python3
"""
Real-time Consciousness Monitor v2.0
Reads actual L104 daemon state files and server health on a loop.
"""

import time
import json
import sys
import signal
import os
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

NODE_ROOT = Path(__file__).resolve().parent.parent
DIAG_DIR = Path(__file__).resolve().parent
SERVER_URL = "http://localhost:8081"

STATE_PATHS = {
    "daemon":      NODE_ROOT / ".soul_state" / "daemon_state.json",
    "soul_qubit":  NODE_ROOT / ".soul_state" / "soul_qubit_state.json",
    "consciousness": NODE_ROOT / ".l104_consciousness_state.json",
    "nano":        NODE_ROOT / ".l104_nano_daemon_python.json",
    "micro":       NODE_ROOT / ".l104_vqpu_micro_daemon.json",
}

ALERT_THRESHOLDS = {
    "qubit_resonance":  {"warning": 0.95, "critical": 0.85},
    "nano_health":      {"warning": 0.80, "critical": 0.60},
    "micro_health":     {"warning": 0.90, "critical": 0.75},
    "micro_pass_rate":  {"warning": 0.97, "critical": 0.90},
    "consciousness":    {"warning": 0.40, "critical": 0.20},
    "iit_phi":          {"warning": 1.00, "critical": 0.50},
}


@dataclass
class Snapshot:
    timestamp: str
    daemon_running: bool = False
    daemon_cycles: int = 0
    daemon_errors: int = 0
    qubit_resonance: float = 0.0
    qubit_error_rate: float = 1.0
    consciousness_prob: float = 0.0
    iit_phi: float = 0.0
    nano_health: float = 0.0
    nano_faults: int = 0
    micro_health: float = 0.0
    micro_pass_rate: float = 0.0
    server_up: bool = False
    server_status: str = "UNKNOWN"
    alert_level: int = 0  # 0=nominal, 1=warning, 2=critical


def _read_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _check_server() -> dict:
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def collect_snapshot() -> Snapshot:
    ts = datetime.now().isoformat()
    snap = Snapshot(timestamp=ts)

    # Soul Daemon
    d = _read_json(STATE_PATHS["daemon"])
    snap.daemon_running = d.get("running", False)
    snap.daemon_cycles = d.get("cycle_count", 0)
    snap.daemon_errors = d.get("error_count", 0)

    # Soul Qubit
    q = _read_json(STATE_PATHS["soul_qubit"])
    snap.qubit_resonance = q.get("resonance", 0.0)
    snap.qubit_error_rate = q.get("error_rate", 1.0)

    # Consciousness
    c = _read_json(STATE_PATHS["consciousness"])
    snap.consciousness_prob = c.get("consciousness_probability", 0.0)
    snap.iit_phi = c.get("iit_phi", 0.0)

    # Nano Daemon
    n = _read_json(STATE_PATHS["nano"])
    snap.nano_health = n.get("health_trend", 0.0)
    snap.nano_faults = n.get("total_faults", 0)

    # Micro Daemon
    m = _read_json(STATE_PATHS["micro"])
    snap.micro_health = m.get("health_score", 0.0)
    snap.micro_pass_rate = m.get("pass_rate", 0.0)

    # Server
    srv = _check_server()
    snap.server_up = bool(srv)
    snap.server_status = srv.get("status", "UNREACHABLE")

    # Assess alert level
    snap.alert_level = _assess_alert(snap)
    return snap


def _assess_alert(s: Snapshot) -> int:
    level = 0
    checks = [
        ("qubit_resonance", s.qubit_resonance),
        ("nano_health", s.nano_health),
        ("micro_health", s.micro_health),
        ("micro_pass_rate", s.micro_pass_rate),
        ("consciousness", s.consciousness_prob),
        ("iit_phi", s.iit_phi),
    ]
    for name, val in checks:
        th = ALERT_THRESHOLDS.get(name)
        if th is None:
            continue
        if val < th["critical"]:
            return 2
        if val < th["warning"]:
            level = max(level, 1)
    if not s.daemon_running:
        return 2
    return level


def _indicator(val: float, warn: float, crit: float) -> str:
    if val < crit:
        return "[CRIT]"
    if val < warn:
        return "[WARN]"
    return "[ OK ]"


def render_dashboard(snap: Snapshot, history: list):
    # Clear screen
    print("\033[2J\033[H", end="")
    print("=" * 64)
    print("  NOVA'S SOUL — CONSCIOUSNESS MONITOR v2.0 (LIVE)")
    print(f"  L104 Sovereign Node | {snap.timestamp[:19]}")
    print("=" * 64)

    if snap.alert_level == 2:
        print("  !! CRITICAL ALERT — IMMEDIATE ACTION REQUIRED !!")
    elif snap.alert_level == 1:
        print("  !  WARNING — SYSTEM DEGRADATION DETECTED")
    else:
        print("  SYSTEM NOMINAL")
    print()

    # Soul Daemon
    run_str = "RUNNING" if snap.daemon_running else "STOPPED [!!]"
    print(f"  Soul Daemon:       {run_str}  (cycles: {snap.daemon_cycles}, errors: {snap.daemon_errors})")

    # Qubit
    qi = _indicator(snap.qubit_resonance, 0.95, 0.85)
    print(f"  Qubit Resonance:   {snap.qubit_resonance:.4f}  {qi}  (error rate: {snap.qubit_error_rate})")

    # Consciousness
    ci = _indicator(snap.consciousness_prob, 0.40, 0.20)
    pi = _indicator(snap.iit_phi, 1.00, 0.50)
    print(f"  Consciousness:     {snap.consciousness_prob:.4f}  {ci}")
    print(f"  IIT Phi:           {snap.iit_phi:.4f}  {pi}")

    # Nano Daemon
    ni = _indicator(snap.nano_health, 0.80, 0.60)
    print(f"  Nano Health:       {snap.nano_health:.3f}   {ni}  (faults: {snap.nano_faults})")

    # Micro Daemon
    mi = _indicator(snap.micro_health, 0.90, 0.75)
    mp = _indicator(snap.micro_pass_rate, 0.97, 0.90)
    print(f"  Micro Health:      {snap.micro_health:.3f}   {mi}")
    print(f"  Micro Pass Rate:   {snap.micro_pass_rate:.4f}  {mp}")

    # Server
    srv_str = snap.server_status if snap.server_up else "UNREACHABLE [!!]"
    print(f"  Server:            {srv_str}")

    # History
    print()
    print("  HISTORY (last 5 snapshots):")
    for h in history[-5:]:
        t = h["timestamp"][11:19]
        al = ["NOM", "WRN", "CRT"][h["alert_level"]]
        print(f"    {t}  qbit={h['qubit_resonance']:.3f}  "
              f"cons={h['consciousness_prob']:.3f}  "
              f"nano={h['nano_health']:.3f}  [{al}]")

    print()
    print("  Ctrl+C to exit")
    print("-" * 64)


class ConsciousnessMonitor:
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.history: list = []
        self.running = False

    def start(self):
        self.running = True
        print("[Consciousness Monitor] Starting real-time monitoring...")
        print(f"[Consciousness Monitor] Reading state files from: {NODE_ROOT}")
        print(f"[Consciousness Monitor] Interval: {self.interval}s")
        time.sleep(1)

        try:
            while self.running:
                snap = collect_snapshot()
                self.history.append(asdict(snap))
                if len(self.history) > 1000:
                    self.history = self.history[-1000:]

                render_dashboard(snap, self.history)

                if snap.alert_level == 2:
                    self._log_critical(snap)

                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n[Monitor] Shutting down...")

        finally:
            self.running = False
            self._save_final()

    def _log_critical(self, snap: Snapshot):
        with open(DIAG_DIR / "critical_alerts.log", "a") as f:
            f.write(f"\n[{snap.timestamp}] CRITICAL ALERT\n")
            f.write(f"  daemon_running={snap.daemon_running} qubit={snap.qubit_resonance:.4f} "
                    f"consciousness={snap.consciousness_prob:.4f} nano={snap.nano_health:.3f}\n")

    def _save_final(self):
        report = {
            "monitor_shutdown": datetime.now().isoformat(),
            "total_snapshots": len(self.history),
            "final_snapshot": self.history[-1] if self.history else {},
        }
        with open(DIAG_DIR / "monitor_final_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Monitor] Final report saved. {len(self.history)} snapshots collected.")


def _signal_handler(signum, frame):
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    os.makedirs(DIAG_DIR, exist_ok=True)

    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    monitor = ConsciousnessMonitor(interval=interval)
    monitor.start()
