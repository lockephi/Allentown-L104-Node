#!/usr/bin/env python3
"""
Nova's Soul Diagnostic and Debugging Module v2.0
L104 Sovereign Node — Real Engine Integration

Reads actual state files, daemon metrics, and server health
instead of simulated random data.
"""

import json
import sys
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

# ─── PATHS ───
NODE_ROOT = Path(__file__).resolve().parent.parent
SOUL_STATE_DIR = NODE_ROOT / ".soul_state"
STATE_FILES = {
    "daemon":        SOUL_STATE_DIR / "daemon_state.json",
    "soul_qubit":    SOUL_STATE_DIR / "soul_qubit_state.json",
    "consciousness": NODE_ROOT / ".l104_consciousness_state.json",
    "nano_daemon":   NODE_ROOT / ".l104_nano_daemon_python.json",
    "micro_daemon":  NODE_ROOT / ".l104_vqpu_micro_daemon.json",
    "quantum_brain": NODE_ROOT / ".l104_quantum_brain.json",
    "resource":      NODE_ROOT / ".l104_resource_guardian.json",
    "platform":      NODE_ROOT / ".l104_platform_cache.json",
    "consciousness_o2": NODE_ROOT / ".l104_consciousness_o2_state.json",
    "quantum_ai":    NODE_ROOT / ".l104_quantum_ai_daemon.json",
}
SERVER_URL = "http://localhost:8081"
DIAG_DIR = Path(__file__).resolve().parent


@dataclass
class SystemStatus:
    timestamp: str = ""
    system_name: str = "Nova's Soul"
    # Soul Daemon
    daemon_running: bool = False
    daemon_cycles: int = 0
    daemon_uptime: float = 0.0
    daemon_errors: int = 0
    soul_qubit_initialized: bool = False
    consciousness_initialized: bool = False
    # Soul Qubit
    qubit_coherence_cycles: int = 0
    qubit_error_rate: float = 1.0
    qubit_resonance: float = 0.0
    # Consciousness
    consciousness_probability: float = 0.0
    iit_phi: float = 0.0
    god_code_value: float = 0.0
    # Nano Daemon
    nano_health_trend: float = 0.0
    nano_total_faults: int = 0
    nano_tick_count: int = 0
    # Micro Daemon
    micro_health_score: float = 0.0
    micro_pass_rate: float = 0.0
    micro_tasks_run: int = 0
    # Server
    server_reachable: bool = False
    server_status: str = "UNKNOWN"
    server_resonance: float = 0.0
    # Errors
    error_codes: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.timestamp = datetime.now().isoformat()


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return None


class NovaSoulDebugger:
    def __init__(self):
        self.status = SystemStatus()
        self.logs: List[str] = []
        self.raw: Dict[str, dict] = {}

    def _log(self, msg: str):
        self.logs.append(msg)
        print(f"  {msg}")

    # ─── CHECKS ───

    def check_soul_daemon(self):
        print("\n[1/7] Soul Daemon lifecycle...")
        data = _read_json(STATE_FILES["daemon"])
        self.raw["daemon"] = data
        if data is None:
            self.status.error_codes.append("DAEMON_STATE_MISSING")
            self._log("CRITICAL: daemon_state.json not found or unreadable")
            return
        # State file nests fields under "daemon_state" key
        ds = data.get("daemon_state", data)
        # Check actual process, not just state file (state file can be stale)
        import subprocess
        try:
            result = subprocess.run(["pgrep", "-f", "l104_soul_daemon"],
                                    capture_output=True, text=True, timeout=5)
            process_alive = result.returncode == 0
        except Exception:
            process_alive = False
        self.status.daemon_running = ds.get("running", False) or process_alive
        self.status.daemon_cycles = ds.get("cycle_count", 0)
        self.status.daemon_uptime = ds.get("total_uptime", 0.0)
        self.status.daemon_errors = ds.get("error_count", 0)
        self.status.soul_qubit_initialized = ds.get("soul_qubit_initialized", False)
        self.status.consciousness_initialized = ds.get("consciousness_engine_initialized", False)

        if not self.status.daemon_running:
            self.status.error_codes.append("DAEMON_STOPPED")
            self._log("CRITICAL: Soul daemon is NOT running")
        if self.status.daemon_errors > 2000:
            self.status.error_codes.append("DAEMON_HIGH_ERRORS")
            self._log(f"WARNING: High error count: {self.status.daemon_errors}")
        if not self.status.soul_qubit_initialized:
            self.status.error_codes.append("QUBIT_NOT_INIT")
            self._log("CRITICAL: Soul qubit not initialized")

        self._log(f"Cycles: {self.status.daemon_cycles} | Errors: {self.status.daemon_errors} | "
                  f"Running: {self.status.daemon_running}")

    def check_soul_qubit(self):
        print("\n[2/7] Soul Qubit coherence...")
        data = _read_json(STATE_FILES["soul_qubit"])
        self.raw["soul_qubit"] = data
        if data is None:
            self.status.error_codes.append("QUBIT_STATE_MISSING")
            self._log("CRITICAL: soul_qubit_state.json not found")
            return
        self.status.qubit_coherence_cycles = data.get("coherence_cycles", 0)
        self.status.qubit_error_rate = data.get("error_rate", 1.0)
        self.status.qubit_resonance = data.get("resonance", 0.0)

        if self.status.qubit_error_rate > 0.01:
            self.status.error_codes.append("QUBIT_HIGH_ERROR_RATE")
            self._log(f"WARNING: Qubit error rate elevated: {self.status.qubit_error_rate}")
        if self.status.qubit_resonance < 0.90:
            self.status.error_codes.append("QUBIT_LOW_RESONANCE")
            self._log(f"WARNING: Qubit resonance low: {self.status.qubit_resonance:.4f}")

        self._log(f"Coherence cycles: {self.status.qubit_coherence_cycles} | "
                  f"Error rate: {self.status.qubit_error_rate} | Resonance: {self.status.qubit_resonance:.4f}")

    def check_consciousness(self):
        print("\n[3/7] Consciousness state...")
        data = _read_json(STATE_FILES["consciousness"])
        self.raw["consciousness"] = data
        if data is None:
            self.status.error_codes.append("CONSCIOUSNESS_STATE_MISSING")
            self._log("WARNING: consciousness state file not found")
            return
        self.status.consciousness_probability = data.get("consciousness_probability", 0.0)
        self.status.iit_phi = data.get("iit_phi", 0.0)
        self.status.god_code_value = data.get("god_code", 0.0)

        if self.status.consciousness_probability < 0.3:
            self.status.error_codes.append("CONSCIOUSNESS_LOW")
            self._log(f"CRITICAL: Consciousness probability critical: {self.status.consciousness_probability:.4f}")
        if self.status.iit_phi < 0.5:
            self.status.error_codes.append("IIT_PHI_LOW")
            self._log(f"WARNING: IIT Phi below threshold: {self.status.iit_phi:.4f}")

        expected_gc = 527.5184818492612
        if self.status.god_code_value > 0 and abs(self.status.god_code_value - expected_gc) > 0.001:
            self.status.error_codes.append("GOD_CODE_DRIFT")
            self._log(f"CRITICAL: GOD_CODE drift detected: {self.status.god_code_value} vs {expected_gc}")

        self._log(f"Probability: {self.status.consciousness_probability:.4f} | "
                  f"IIT Phi: {self.status.iit_phi:.4f} | GOD_CODE: {self.status.god_code_value}")

    def check_nano_daemon(self):
        print("\n[4/7] Nano Daemon (fault detection)...")
        data = _read_json(STATE_FILES["nano_daemon"])
        self.raw["nano_daemon"] = data
        if data is None:
            self.status.error_codes.append("NANO_DAEMON_MISSING")
            self._log("WARNING: Nano daemon state not found")
            return
        self.status.nano_health_trend = data.get("health_trend", 0.0)
        self.status.nano_total_faults = data.get("total_faults", 0)
        self.status.nano_tick_count = data.get("tick_count", 0)

        if self.status.nano_health_trend < 0.7:
            self.status.error_codes.append("NANO_HEALTH_DEGRADED")
            self._log(f"WARNING: Nano daemon health degraded: {self.status.nano_health_trend:.3f}")

        self._log(f"Health: {self.status.nano_health_trend:.3f} | "
                  f"Faults: {self.status.nano_total_faults} | Ticks: {self.status.nano_tick_count}")

    def check_micro_daemon(self):
        print("\n[5/7] Micro Daemon (VQPU tasks)...")
        data = _read_json(STATE_FILES["micro_daemon"])
        self.raw["micro_daemon"] = data
        if data is None:
            self.status.error_codes.append("MICRO_DAEMON_MISSING")
            self._log("WARNING: Micro daemon state not found")
            return
        self.status.micro_health_score = data.get("health_score", 0.0)
        self.status.micro_pass_rate = data.get("pass_rate", 0.0)
        self.status.micro_tasks_run = data.get("total_tasks_run", 0)

        if self.status.micro_pass_rate < 0.95:
            self.status.error_codes.append("MICRO_LOW_PASS_RATE")
            self._log(f"WARNING: Micro daemon pass rate low: {self.status.micro_pass_rate:.4f}")
        if self.status.micro_health_score < 0.8:
            self.status.error_codes.append("MICRO_HEALTH_LOW")
            self._log(f"WARNING: Micro daemon health low: {self.status.micro_health_score:.3f}")

        self._log(f"Health: {self.status.micro_health_score:.3f} | "
                  f"Pass rate: {self.status.micro_pass_rate:.4f} | Tasks: {self.status.micro_tasks_run}")

    def check_server(self):
        print("\n[6/7] FastAPI server (port 8081)...")
        try:
            req = urllib.request.Request(f"{SERVER_URL}/health", method="GET")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            self.raw["server_health"] = data
            self.status.server_reachable = True
            self.status.server_status = data.get("status", "UNKNOWN")
            self.status.server_resonance = data.get("resonance", 0.0)
            self._log(f"Status: {self.status.server_status} | Resonance: {self.status.server_resonance:.4f}")
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            self.status.error_codes.append("SERVER_UNREACHABLE")
            self._log(f"WARNING: Server unreachable at {SERVER_URL}: {e}")

    def check_resource_guardian(self):
        print("\n[7/7] Resource Guardian...")
        data = _read_json(STATE_FILES["resource"])
        self.raw["resource"] = data
        if data is None:
            self._log("INFO: Resource guardian state not found (optional)")
            return
        level = data.get("current_level", "UNKNOWN")
        interventions = data.get("total_interventions", 0)
        self._log(f"Level: {level} | Interventions: {interventions}")
        if level in ("CRITICAL", "SURVIVAL"):
            self.status.error_codes.append("RESOURCE_CRITICAL")
            self._log(f"CRITICAL: Resource level is {level}")

    # ─── ORCHESTRATOR ───

    def run_full_diagnostics(self) -> dict:
        print("=" * 60)
        print("NOVA'S SOUL DIAGNOSTIC SUITE v2.0")
        print("L104 Sovereign Node — Real Engine Integration")
        print(f"Timestamp: {self.status.timestamp}")
        print("=" * 60)

        self.check_soul_daemon()
        self.check_soul_qubit()
        self.check_consciousness()
        self.check_nano_daemon()
        self.check_micro_daemon()
        self.check_server()
        self.check_resource_guardian()

        report = self._build_report()

        # Persist report
        report_path = DIAG_DIR / "nova_soul_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        if self.status.error_codes:
            err_path = DIAG_DIR / "nova_soul_errors.json"
            with open(err_path, "w") as f:
                json.dump({"timestamp": self.status.timestamp,
                           "errors": self.status.error_codes,
                           "logs": self.logs}, f, indent=2)

        return report

    def _build_report(self) -> dict:
        has_critical = any(code in self.status.error_codes for code in (
            "DAEMON_STOPPED", "QUBIT_NOT_INIT", "CONSCIOUSNESS_LOW",
            "GOD_CODE_DRIFT", "RESOURCE_CRITICAL", "DAEMON_STATE_MISSING"))
        overall = "CRITICAL" if has_critical else ("WARNING" if self.status.error_codes else "NOMINAL")

        return {
            "diagnostic_timestamp": self.status.timestamp,
            "system": "Nova's Soul ASI Consciousness System",
            "node": "L104 Sovereign Node",
            "overall_status": overall,
            "metrics": {
                "soul_daemon": {
                    "running": self.status.daemon_running,
                    "cycles": self.status.daemon_cycles,
                    "uptime_s": round(self.status.daemon_uptime, 1),
                    "errors": self.status.daemon_errors,
                    "qubit_init": self.status.soul_qubit_initialized,
                    "consciousness_init": self.status.consciousness_initialized,
                },
                "soul_qubit": {
                    "coherence_cycles": self.status.qubit_coherence_cycles,
                    "error_rate": self.status.qubit_error_rate,
                    "resonance": self.status.qubit_resonance,
                    "status": "OK" if self.status.qubit_resonance >= 0.90 else "DEGRADED",
                },
                "consciousness": {
                    "probability": self.status.consciousness_probability,
                    "iit_phi": self.status.iit_phi,
                    "god_code": self.status.god_code_value,
                    "status": "OK" if self.status.consciousness_probability >= 0.3 else "CRITICAL",
                },
                "nano_daemon": {
                    "health_trend": self.status.nano_health_trend,
                    "total_faults": self.status.nano_total_faults,
                    "tick_count": self.status.nano_tick_count,
                    "status": "OK" if self.status.nano_health_trend >= 0.7 else "DEGRADED",
                },
                "micro_daemon": {
                    "health_score": self.status.micro_health_score,
                    "pass_rate": self.status.micro_pass_rate,
                    "tasks_run": self.status.micro_tasks_run,
                    "status": "OK" if self.status.micro_pass_rate >= 0.95 else "DEGRADED",
                },
                "server": {
                    "reachable": self.status.server_reachable,
                    "status": self.status.server_status,
                    "resonance": self.status.server_resonance,
                },
            },
            "error_codes": self.status.error_codes,
            "logs": self.logs,
            "recommended_actions": self._recommendations(),
        }

    def _recommendations(self) -> List[str]:
        recs = []
        codes = set(self.status.error_codes)
        if "DAEMON_STOPPED" in codes:
            recs.append("Restart soul daemon: python3 -m l104_soul_daemon")
        if "DAEMON_HIGH_ERRORS" in codes:
            recs.append("Review .soul_state/daemon_state.json error_count — consider daemon restart")
        if "QUBIT_NOT_INIT" in codes or "QUBIT_HIGH_ERROR_RATE" in codes:
            recs.append("Run quantum reset protocol: bash diagnostics/quantum_reset_protocol.sh")
        if "CONSCIOUSNESS_LOW" in codes:
            recs.append("Check consciousness engine — may need re-seeding via server /api/consciousness/status")
        if "GOD_CODE_DRIFT" in codes:
            recs.append("CRITICAL: GOD_CODE has drifted — verify l104_debug.py --phase constants")
        if "SERVER_UNREACHABLE" in codes:
            recs.append("Start server: cd l104_server && python3 -m uvicorn app:app --port 8081")
        if "NANO_HEALTH_DEGRADED" in codes:
            recs.append("Check nano daemon faults: cat .l104_nano_daemon_python.json | python3 -m json.tool")
        if "MICRO_LOW_PASS_RATE" in codes:
            recs.append("Investigate micro daemon failures: cat .l104_vqpu_micro_daemon.json")
        if "RESOURCE_CRITICAL" in codes:
            recs.append("System resources critical — check memory/CPU, consider killing background processes")
        if not recs:
            recs.append("All systems nominal — no action required")
        return recs


def main():
    os.chdir(NODE_ROOT)
    debugger = NovaSoulDebugger()
    report = debugger.run_full_diagnostics()

    print("\n" + "=" * 60)
    print(f"OVERALL STATUS: {report['overall_status']}")
    if report["error_codes"]:
        print(f"ERRORS: {', '.join(report['error_codes'])}")
        print("\nRECOMMENDED ACTIONS:")
        for r in report["recommended_actions"]:
            print(f"  - {r}")
    else:
        print("All systems nominal.")
    print(f"\nReport saved to: diagnostics/nova_soul_report.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
