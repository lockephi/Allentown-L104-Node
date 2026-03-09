"""L104 Tri-Nano Daemon Orchestrator v2.0.0

Orchestrates the three nano-level fault detection daemons:
  1. Swift Nano Daemon  — Mach kernel introspection, Accelerate, Metal  (10 probes)
  2. C Nano Daemon      — IEEE 754 bit-level, FPU, NDE integrity       (8 probes)
  3. Python AI Daemon   — AI anomaly detection, trend prediction       (12 probes)

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │                   TRI-NANO ORCHESTRATOR                         │
  │                  l104_nano_triad.py v2.0                        │
  ├────────────────┬──────────────────┬────────────────────────────┤
  │  Swift Nano    │   C Nano         │  Python AI Nano            │
  │  NanoDaemon    │   l104_nano_     │  nano_daemon.py            │
  │  .swift        │   daemon.c       │  (12 probes + 3 AI)       │
  │  10 probes     │   8 probes       │                            │
  │  Mach/Metal    │   IEEE 754/FPU   │  Isolation Forest          │
  │  Accelerate    │   NDE Pipeline   │  Trend Prediction          │
  │  SecRandom     │   Entropy pool   │  Cross-Correlator          │
  └────────┬───────┴────────┬─────────┴──────────┬─────────────────┘
           │                │                     │
           └────────────────┼─────────────────────┘
                            │
                 /tmp/l104_bridge/nano/
                 ├── swift_outbox/   (Swift reports)
                 ├── c_outbox/       (C reports)
                 ├── python_outbox/  (Python reports)
                 ├── swift_heartbeat
                 ├── c_heartbeat
                 ├── python_heartbeat
                 └── triad_report.json  (unified)

Usage:
  python l104_nano_triad.py                    # Launch all three daemons
  python l104_nano_triad.py --self-test        # Self-test all three
  python l104_nano_triad.py --status           # Read unified health
  python l104_nano_triad.py --build            # Build C + Swift daemons
  python l104_nano_triad.py --install          # Install launchd plists

GOD_CODE=527.5184818492612 | PHI=1.618033988749895 | PILOT: LONDEL
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

VERSION = "2.0.0"
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.04 + PHI / 1000.0

# Paths
L104_ROOT = os.environ.get("L104_ROOT",
    os.path.dirname(os.path.abspath(__file__)))
NANO_BASE = "/tmp/l104_bridge/nano"
TRIAD_REPORT = f"{NANO_BASE}/triad_report.json"
TRIAD_PID = f"{NANO_BASE}/triad.pid"

# Daemon locations
C_DAEMON_BIN = os.path.join(L104_ROOT, "l104_core_c", "build", "l104_nano_daemon")
SWIFT_DAEMON_BIN = os.path.join(L104_ROOT, "L104SwiftApp", ".build", "release", "L104NanoDaemon")
PYTHON_DAEMON_MODULE = "l104_vqpu.nano_daemon"
VENV_PYTHON = os.path.join(L104_ROOT, ".venv", "bin", "python")

# v2.0: Auto-recovery
MAX_AUTO_RESTARTS = 3
RESTART_COOLDOWN_S = 30.0

# v2.0: Cross-daemon state files for unified dashboard
DAEMON_STATE_FILES = {
    "vqpu_cycler": ".l104_vqpu_daemon_state.json",
    "micro_daemon": ".l104_vqpu_micro_daemon.json",
    "guardian": ".l104_resource_guardian.json",
    "quantum_ai": ".l104_quantum_ai_daemon.json",
}

# Heartbeat paths
HEARTBEATS = {
    "swift": f"{NANO_BASE}/swift_heartbeat",
    "c": f"{NANO_BASE}/c_heartbeat",
    "python": f"{NANO_BASE}/python_heartbeat",
}

# PID files
PIDS = {
    "swift": f"{NANO_BASE}/swift_nano.pid",
    "c": f"{NANO_BASE}/c_nano.pid",
    "python": f"{NANO_BASE}/python_nano.pid",
}

# Outboxes
OUTBOXES = {
    "swift": f"{NANO_BASE}/swift_outbox",
    "c": f"{NANO_BASE}/c_outbox",
    "python": f"{NANO_BASE}/python_outbox",
}


# ═══════════════════════════════════════════════════════════════════════
# DAEMON HEALTH READER
# ═══════════════════════════════════════════════════════════════════════

# Map daemon names → their binary paths (None = no binary needed)
_DAEMON_BINARIES = {
    "swift": SWIFT_DAEMON_BIN,
    "c": C_DAEMON_BIN,
    "python": None,  # Runs in-process
}


def _check_heartbeat(name: str) -> dict:
    """Check a daemon's heartbeat file for liveness."""
    path = HEARTBEATS.get(name)
    if not path or not os.path.exists(path):
        # If binary doesn't exist, report "not built" rather than generic "no heartbeat"
        binary = _DAEMON_BINARIES.get(name)
        if binary and not os.path.exists(binary):
            return {"alive": False, "age": -1, "reason": f"binary not built ({os.path.basename(binary)})"}
        return {"alive": False, "age": -1, "reason": "no heartbeat file"}

    try:
        # Ignore test-only heartbeats (written by --self-test, not a live daemon)
        try:
            hb_data = json.loads(Path(path).read_text())
            if isinstance(hb_data, dict) and hb_data.get("test"):
                binary = _DAEMON_BINARIES.get(name)
                if binary and not os.path.exists(binary):
                    return {"alive": False, "age": -1, "reason": f"binary not built ({os.path.basename(binary)})"}
                return {"alive": False, "age": -1, "reason": "heartbeat is from self-test only"}
        except (json.JSONDecodeError, ValueError):
            pass  # Plain-text heartbeat — check normally

        age = time.time() - os.path.getmtime(path)
        alive = age < 30.0  # Stale if >30s
        return {"alive": alive, "age": round(age, 1),
                "reason": "ok" if alive else f"stale ({age:.0f}s)"}
    except OSError as e:
        return {"alive": False, "age": -1, "reason": str(e)}


def _check_pid(name: str) -> dict:
    """Check if a daemon's PID is running."""
    path = PIDS.get(name)
    if not path or not os.path.exists(path):
        return {"running": False, "pid": None}

    try:
        pid = int(Path(path).read_text().strip())
        os.kill(pid, 0)  # Check if alive
        return {"running": True, "pid": pid}
    except (ValueError, ProcessLookupError, PermissionError):
        return {"running": False, "pid": None}


def _read_latest_report(name: str) -> dict | None:
    """Read the latest tick report from a daemon's outbox."""
    outbox = OUTBOXES.get(name)
    if not outbox or not os.path.isdir(outbox):
        return None

    try:
        files = sorted(Path(outbox).glob("tick_*.json"))
        if files:
            return json.loads(files[-1].read_text())
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════
# BUILD
# ═══════════════════════════════════════════════════════════════════════

def build_c_daemon() -> bool:
    """Build the C nano daemon."""
    c_dir = os.path.join(L104_ROOT, "l104_core_c")
    print("[Triad] Building C nano daemon...")
    try:
        result = subprocess.run(
            ["make", "nano-daemon"],
            cwd=c_dir, capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print("  ✓ C nano daemon built successfully")
            return True
        else:
            print(f"  ✗ C build failed: {result.stderr[:200]}")
            # Fallback: direct compile
            print("  → Attempting direct compile...")
            result2 = subprocess.run(
                ["clang", "-O3", "-march=native", "-o",
                 os.path.join(c_dir, "build", "l104_nano_daemon"),
                 os.path.join(c_dir, "l104_nano_daemon.c"),
                 os.path.join(c_dir, "l104_sage_core.c"),
                 "-lm", "-lpthread"],
                cwd=c_dir, capture_output=True, text=True, timeout=60
            )
            os.makedirs(os.path.join(c_dir, "build"), exist_ok=True)
            if result2.returncode == 0:
                print("  ✓ C nano daemon built via direct compile")
                return True
            print(f"  ✗ Direct compile failed: {result2.stderr[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ Build error: {e}")
        return False


def build_swift_daemon() -> bool:
    """Build the Swift nano daemon."""
    swift_dir = os.path.join(L104_ROOT, "L104SwiftApp")
    print("[Triad] Building Swift nano daemon...")
    try:
        result = subprocess.run(
            ["swift", "build", "-c", "release", "--product", "L104NanoDaemon"],
            cwd=swift_dir, capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print("  ✓ Swift nano daemon built successfully")
            return True
        else:
            print(f"  ✗ Swift build failed: {result.stderr[:300]}")
            return False
    except Exception as e:
        print(f"  ✗ Build error: {e}")
        return False


def build_all() -> dict:
    """Build both C and Swift nano daemons."""
    results = {}
    results["c"] = build_c_daemon()
    results["swift"] = build_swift_daemon()
    results["python"] = True  # No build needed
    return results


# ═══════════════════════════════════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════════════════════════════════

class TriNanoDaemon:
    """Orchestrates all three nano daemons."""

    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self.running = False
        self._python_daemon = None  # In-process Python daemon

        # v2.0: Auto-recovery tracking
        self._restart_counts: dict[str, int] = {"swift": 0, "c": 0, "python": 0}
        self._last_restart_ts: dict[str, float] = {"swift": 0, "c": 0, "python": 0}

        # v2.0: Performance profiling
        self._launch_times: dict[str, float] = {}  # daemon → launch_time_ms

        # v2.0: Unified health
        self._unified_health = 1.0

    def launch_c(self, tick_ms: int = 3000) -> subprocess.Popen | None:
        """Launch C nano daemon as subprocess."""
        if not os.path.exists(C_DAEMON_BIN):
            print(f"[Triad] C daemon not found at {C_DAEMON_BIN} — build with --build first")
            return None

        proc = subprocess.Popen(
            [C_DAEMON_BIN, "--tick", str(tick_ms)],
            cwd=L104_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.processes["c"] = proc
        print(f"[Triad] C nano daemon launched (PID={proc.pid}, tick={tick_ms}ms)")
        return proc

    def launch_swift(self, tick_s: float = 3.0) -> subprocess.Popen | None:
        """Launch Swift nano daemon as subprocess."""
        if not os.path.exists(SWIFT_DAEMON_BIN):
            print(f"[Triad] Swift daemon not found at {SWIFT_DAEMON_BIN} — build with --build first")
            return None

        proc = subprocess.Popen(
            [SWIFT_DAEMON_BIN, "--tick", str(tick_s)],
            cwd=L104_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.processes["swift"] = proc
        print(f"[Triad] Swift nano daemon launched (PID={proc.pid}, tick={tick_s}s)")
        return proc

    def launch_python(self, tick_s: float = 3.0) -> bool:
        """Launch Python nano daemon in a background thread."""
        try:
            from l104_vqpu.nano_daemon import NanoDaemon
            self._python_daemon = NanoDaemon(tick_interval=tick_s, verbose=False)

            def _run():
                self._python_daemon.run()

            thread = threading.Thread(target=_run, daemon=True, name="nano-python")
            thread.start()
            self.processes["python_thread"] = thread
            print(f"[Triad] Python AI nano daemon launched (thread, tick={tick_s}s)")
            return True
        except Exception as e:
            print(f"[Triad] Python daemon launch failed: {e}")
            # Fallback: subprocess
            proc = subprocess.Popen(
                [VENV_PYTHON, "-m", PYTHON_DAEMON_MODULE, "--tick", str(tick_s)],
                cwd=L104_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.processes["python"] = proc
            print(f"[Triad] Python AI nano daemon launched (subprocess PID={proc.pid})")
            return True

    def launch_all(self, tick_s: float = 3.0):
        """Launch all three nano daemons."""
        os.makedirs(NANO_BASE, exist_ok=True)
        for outbox in OUTBOXES.values():
            os.makedirs(outbox, exist_ok=True)

        # Write triad PID
        with open(TRIAD_PID, "w") as fp:
            fp.write(f"{os.getpid()}\n")

        self.running = True
        self.launch_c(tick_ms=int(tick_s * 1000))
        self.launch_swift(tick_s=tick_s)
        self.launch_python(tick_s=tick_s)

    def status(self) -> dict:
        """Unified status of all three daemons."""
        result = {
            "orchestrator": "l104_nano_triad",
            "version": VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "daemons": {},
        }

        for name in ["swift", "c", "python"]:
            hb = _check_heartbeat(name)
            pid_info = _check_pid(name)
            report = _read_latest_report(name)

            result["daemons"][name] = {
                "heartbeat": hb,
                "pid": pid_info,
                "health": report.get("health", None) if report else None,
                "fault_count": report.get("fault_count", report.get("faults", None)) if report else None,
                "tick": report.get("tick", None) if report else None,
            }

        # Unified health: weighted average of all three
        healths = [d["health"] for d in result["daemons"].values() if d["health"] is not None]
        result["unified_health"] = sum(healths) / len(healths) if healths else None
        result["all_alive"] = all(d["heartbeat"]["alive"] for d in result["daemons"].values())

        # v2.0: Include auto-recovery stats and dashboard health
        result["auto_recovery"] = {
            "restart_counts": dict(self._restart_counts),
            "max_restarts": MAX_AUTO_RESTARTS,
        }
        result["launch_times_ms"] = dict(self._launch_times)
        result["dashboard_health"] = self._unified_health

        return result

    def auto_recover(self) -> dict:
        """v2.0: Detect dead nano daemons and auto-restart them."""
        results = {}
        now = time.time()
        for name in ["swift", "c", "python"]:
            heartbeat_path = HEARTBEATS.get(name, "")
            alive = False
            if heartbeat_path and os.path.exists(heartbeat_path):
                try:
                    age = now - os.path.getmtime(heartbeat_path)
                    alive = age < 15.0  # Heartbeat within 15 seconds
                except Exception:
                    pass

            if not alive:
                if (self._restart_counts[name] < MAX_AUTO_RESTARTS and
                        (now - self._last_restart_ts[name]) > RESTART_COOLDOWN_S):
                    self._restart_counts[name] += 1
                    self._last_restart_ts[name] = now
                    try:
                        self._restart_daemon(name)
                        results[name] = {"action": "restarted", "attempt": self._restart_counts[name]}
                    except Exception as e:
                        results[name] = {"action": "restart_failed", "error": str(e)[:60]}
                else:
                    results[name] = {"action": "skip", "reason": "max_restarts_or_cooldown"}
            else:
                results[name] = {"action": "alive"}

        return results

    def _restart_daemon(self, name: str):
        """v2.0: Restart a specific nano daemon."""
        if name == "python":
            subprocess.Popen(
                [VENV_PYTHON, "-m", PYTHON_DAEMON_MODULE],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
        elif name == "c" and os.path.exists(C_DAEMON_BIN):
            subprocess.Popen(
                [C_DAEMON_BIN],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
        elif name == "swift" and os.path.exists(SWIFT_DAEMON_BIN):
            subprocess.Popen(
                [SWIFT_DAEMON_BIN],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)

    def unified_health_dashboard(self) -> dict:
        """v2.0: Unified health from all nano daemons + other L104 daemons."""
        dashboard = {"nano_daemons": {}, "system_daemons": {}}

        # Nano daemon health from heartbeats
        now = time.time()
        for name in ["swift", "c", "python"]:
            hb_path = HEARTBEATS.get(name, "")
            if hb_path and os.path.exists(hb_path):
                try:
                    age = now - os.path.getmtime(hb_path)
                    dashboard["nano_daemons"][name] = {
                        "alive": age < 15.0,
                        "heartbeat_age_s": round(age, 1),
                        "restart_count": self._restart_counts.get(name, 0),
                    }
                except Exception:
                    dashboard["nano_daemons"][name] = {"alive": False}
            else:
                dashboard["nano_daemons"][name] = {"alive": False}

        # System daemon health from state files
        for daemon_name, state_file in DAEMON_STATE_FILES.items():
            try:
                path = os.path.join(L104_ROOT, state_file)
                if os.path.exists(path):
                    with open(path) as f:
                        data = json.load(f)
                    dashboard["system_daemons"][daemon_name] = {
                        "available": True,
                        "health_score": data.get("health_score", data.get("health_trend", 0)),
                        "version": data.get("version", "?"),
                    }
                else:
                    dashboard["system_daemons"][daemon_name] = {"available": False}
            except Exception:
                dashboard["system_daemons"][daemon_name] = {"available": False}

        # Compute unified health
        all_scores = []
        for nd in dashboard["nano_daemons"].values():
            all_scores.append(1.0 if nd.get("alive") else 0.0)
        for sd in dashboard["system_daemons"].values():
            if sd.get("available"):
                h = sd.get("health_score", 0)
                all_scores.append(h if isinstance(h, (int, float)) else 0.0)

        self._unified_health = round(
            sum(all_scores) / max(len(all_scores), 1), 4) if all_scores else 0.0
        dashboard["unified_health"] = self._unified_health
        dashboard["version"] = "2.0.0"

        return dashboard

    def write_triad_report(self):
        """Write unified triad report to IPC."""
        report = self.status()
        try:
            with open(TRIAD_REPORT, "w") as fp:
                json.dump(report, fp, indent=2)
        except Exception:
            pass

    def shutdown(self):
        """Shutdown all daemons."""
        self.running = False
        print("[Triad] Shutting down all nano daemons...")

        if self._python_daemon:
            self._python_daemon.running = False

        for name, proc in self.processes.items():
            if isinstance(proc, subprocess.Popen):
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                print(f"  ✓ {name} daemon stopped")

        try:
            os.unlink(TRIAD_PID)
        except OSError:
            pass

        print("[Triad] All nano daemons stopped")

    def monitor_loop(self, interval: float = 10.0):
        """Monitor all three daemons, restart if needed, write triad reports."""
        print(f"[Triad] Monitor loop started (interval={interval}s)")

        while self.running:
            try:
                self.write_triad_report()
                status = self.status()

                # Check for dead daemons and restart
                for name in ["c", "swift"]:
                    hb = status["daemons"][name]["heartbeat"]
                    if not hb["alive"] and name in self.processes:
                        proc = self.processes[name]
                        if isinstance(proc, subprocess.Popen) and proc.poll() is not None:
                            print(f"[Triad] {name} daemon died (exit={proc.returncode}), restarting...")
                            if name == "c":
                                self.launch_c()
                            elif name == "swift":
                                self.launch_swift()

                # Log unified health
                uh = status.get("unified_health")
                alive = status.get("all_alive")
                if uh is not None:
                    print(f"[Triad] unified_health={uh:.4f} all_alive={alive}")

                time.sleep(interval)
            except KeyboardInterrupt:
                break

        self.shutdown()


# ═══════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════

def self_test_all() -> int:
    """Run self-tests on all three nano daemons."""
    failures = 0

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  L104 TRI-NANO DAEMON — Unified Self-Test                      ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    # 1. Python AI nano daemon
    print("━━━ Python AI Nano Daemon ━━━")
    try:
        from l104_vqpu.nano_daemon import NanoDaemon
        daemon = NanoDaemon()
        py_failures = daemon.self_test()
        failures += py_failures
    except Exception as e:
        print(f"  FAIL: Python nano daemon import error: {e}")
        failures += 1

    print()

    # 2. C nano daemon
    print("━━━ C Nano Daemon ━━━")
    if os.path.exists(C_DAEMON_BIN):
        try:
            result = subprocess.run(
                [C_DAEMON_BIN, "--self-test"],
                capture_output=True, text=True, timeout=30
            )
            print(result.stdout)
            if result.returncode != 0:
                failures += 1
                print("  FAIL: C nano daemon self-test failed")
        except Exception as e:
            print(f"  SKIP: C daemon not runnable: {e}")
    else:
        print(f"  SKIP: C daemon not built (run with --build first)")

    print()

    # 3. Swift nano daemon
    print("━━━ Swift Nano Daemon ━━━")
    if os.path.exists(SWIFT_DAEMON_BIN):
        try:
            result = subprocess.run(
                [SWIFT_DAEMON_BIN, "--self-test"],
                capture_output=True, text=True, timeout=30
            )
            print(result.stdout)
            if result.returncode != 0:
                failures += 1
                print("  FAIL: Swift nano daemon self-test failed")
        except Exception as e:
            print(f"  SKIP: Swift daemon not runnable: {e}")
    else:
        print(f"  SKIP: Swift daemon not built (run with --build first)")

    print()
    print(f"[Triad] Unified self-test: {failures} total failures")
    return failures


# ═══════════════════════════════════════════════════════════════════════
# INSTALL LAUNCHD PLISTS
# ═══════════════════════════════════════════════════════════════════════

def install_plists():
    """Install launchd plist files for all three nano daemons."""
    config_dir = os.path.join(L104_ROOT, "config")
    launch_agents = os.path.expanduser("~/Library/LaunchAgents")
    os.makedirs(launch_agents, exist_ok=True)

    plists = [
        "com.l104.nano-daemon-c.plist",
        "com.l104.nano-daemon-swift.plist",
        "com.l104.nano-daemon-python.plist",
    ]

    for plist in plists:
        src = os.path.join(config_dir, plist)
        dst = os.path.join(launch_agents, plist)

        if not os.path.exists(src):
            print(f"  ✗ {plist} not found in config/")
            continue

        subprocess.run(["cp", src, dst])
        print(f"  ✓ Installed {plist}")

        label = plist.replace(".plist", "")
        subprocess.run(["launchctl", "load", dst])
        print(f"  ✓ Loaded {label}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="L104 Tri-Nano Daemon Orchestrator")
    parser.add_argument("--self-test", action="store_true", help="Self-test all three daemons")
    parser.add_argument("--status", action="store_true", help="Show unified daemon status")
    parser.add_argument("--build", action="store_true", help="Build C + Swift daemons")
    parser.add_argument("--install", action="store_true", help="Install launchd plists")
    parser.add_argument("--tick", type=float, default=3.0, help="Tick interval (seconds)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  L104 TRI-NANO DAEMON ORCHESTRATOR v{VERSION}                      ║")
    print("║  Swift + C + Python AI — Atomized Fault Detection              ║")
    print("║  GOD_CODE=527.5184818492612 | PHI=1.618033988749895            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    if args.build:
        results = build_all()
        print(f"\nBuild results: {results}")
        sys.exit(0 if all(results.values()) else 1)

    if args.self_test:
        failures = self_test_all()
        sys.exit(1 if failures > 0 else 0)

    if args.status:
        triad = TriNanoDaemon()
        status = triad.status()
        print(json.dumps(status, indent=2))
        sys.exit(0)

    if args.install:
        install_plists()
        sys.exit(0)

    # Default: launch all three and monitor
    triad = TriNanoDaemon()

    def _shutdown(signum, frame):
        triad.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    triad.launch_all(tick_s=args.tick)
    triad.monitor_loop(interval=10.0)


if __name__ == "__main__":
    main()
