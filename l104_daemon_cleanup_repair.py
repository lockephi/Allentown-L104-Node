#!/usr/bin/env python3
"""L104 Daemon Cleanup and Repair Utility.

Diagnoses and repairs common daemon state issues:
  - Stale PID files from crashed daemons
  - Orphaned state files
  - Corrupted crash count tracking
  - Quantum mesh state corruption
  - IPC inbox/outbox accumulation

Usage:
    python l104_daemon_cleanup_repair.py --status     # Check daemon health
    python l104_daemon_cleanup_repair.py --clean      # Clean stale files
    python l104_daemon_cleanup_repair.py --reset      # Full reset (careful!)
    python l104_daemon_cleanup_repair.py --repair     # Auto-repair issues
    python l104_daemon_cleanup_repair.py --full       # Full diagnostic + cleanup
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess


# ════════════════════════════════════════════════════════════════════════════════
# DAEMON PATHS
# ════════════════════════════════════════════════════════════════════════════════

MICRO_BRIDGE_PATH = Path("/tmp/l104_bridge/micro")
VQPU_STATE_FILE = ".l104_vqpu_daemon_state.json"
VQPU_MICRO_STATE_FILE = ".l104_vqpu_micro_daemon.json"
VQPU_MICRO_PID_FILE = MICRO_BRIDGE_PATH / "micro_daemon.pid"
VQPU_HEARTBEAT_FILE = MICRO_BRIDGE_PATH / "heartbeat"
QUANTUM_AI_DAEMON_STATE = ".l104_quantum_ai_daemon.json"
QUANTUM_MESH_STATE = '.l104_quantum_mesh_state.json'


# ════════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

class DaemonDiagnostics:
    """Comprehensive daemon state diagnostics."""

    def __init__(self, root_path: Path = None):
        self.root = Path(root_path or os.getcwd())
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.stats: Dict = {}

    def check_vqpu_micro_daemon(self) -> Dict:
        """Check VQPU Micro Daemon state."""
        result = {
            "name": "VQPU Micro Daemon",
            "healthy": True,
            "issues": [],
            "files": {}
        }

        # Check PID file
        pid_file = VQPU_MICRO_PID_FILE
        if pid_file.exists():
            try:
                old_pid = int(pid_file.read_text().strip())
                result["files"]["pid_file"] = {
                    "exists": True,
                    "path": str(pid_file),
                    "content_pid": old_pid,
                    "current_pid": os.getpid(),
                    "same_process": old_pid == os.getpid(),
                }

                # Check if that PID is alive
                try:
                    os.kill(old_pid, 0)
                    result["files"]["pid_file"]["process_alive"] = True
                    result["healthy"] = False
                    result["issues"].append(
                        f"PID file exists and references alive process {old_pid}")
                except (ProcessLookupError, OSError):
                    result["files"]["pid_file"]["process_alive"] = False
                    result["issues"].append(
                        f"PID file stale (process {old_pid} is dead)")
            except Exception as e:
                result["issues"].append(f"PID file corrupt: {e}")
                result["healthy"] = False
        else:
            result["files"]["pid_file"] = {"exists": False, "path": str(pid_file)}

        # Check state file
        state_file = self.root / VQPU_MICRO_STATE_FILE
        if state_file.exists():
            try:
                state_data = json.loads(state_file.read_text())
                result["files"]["state"] = {
                    "exists": True,
                    "size_bytes": state_file.stat().st_size,
                    "tick": state_data.get("tick", 0),
                    "crash_count": state_data.get("crash_count", 0),
                    "health_score": round(state_data.get("health_score", 0.0), 3),
                }
                if state_data.get("crash_count", 0) > 3:
                    result["issues"].append(
                        f"High crash count: {state_data['crash_count']}")
                    result["healthy"] = False
            except Exception as e:
                result["issues"].append(f"State file corrupt: {e}")
                result["healthy"] = False
        else:
            result["files"]["state"] = {"exists": False}

        # Check heartbeat
        if VQPU_HEARTBEAT_FILE.exists():
            try:
                mtime = VQPU_HEARTBEAT_FILE.stat().st_mtime
                age_s = time.time() - mtime
                result["files"]["heartbeat"] = {
                    "exists": True,
                    "age_seconds": round(age_s, 1),
                    "fresh": age_s < 60,
                }
                if age_s > 300:
                    self.warnings.append(
                        f"Micro daemon heartbeat stale ({age_s:.0f}s old)")
            except Exception:
                pass

        # Check IPC
        inbox = MICRO_BRIDGE_PATH / "inbox"
        outbox = MICRO_BRIDGE_PATH / "outbox"
        if inbox.exists():
            files = list(inbox.glob("*.json"))
            result["files"]["ipc_inbox"] = {
                "exists": True,
                "pending_jobs": len(files),
            }
            if len(files) > 50:
                result["issues"].append(
                    f"Large IPC inbox backlog: {len(files)} pending jobs")
                result["healthy"] = False

        if outbox.exists():
            files = list(outbox.glob("*.json"))
            result["files"]["ipc_outbox"] = {
                "exists": True,
                "accumulated_results": len(files),
            }
            if len(files) > 100:
                self.warnings.append(
                    f"Large IPC outbox: {len(files)} accumulated results")

        return result

    def check_quantum_ai_daemon(self) -> Dict:
        """Check Quantum AI Daemon state."""
        result = {
            "name": "Quantum AI Daemon",
            "healthy": True,
            "issues": [],
            "files": {}
        }

        state_file = self.root / QUANTUM_AI_DAEMON_STATE
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
                result["files"]["state"] = {
                    "exists": True,
                    "size_bytes": state_file.stat().st_size,
                    "active": state.get("active", False),
                    "phase": state.get("phase", "unknown"),
                }
            except Exception as e:
                result["issues"].append(f"State file corrupt: {e}")
                result["healthy"] = False
        else:
            result["files"]["state"] = {"exists": False}

        return result

    def check_quantum_mesh(self) -> Dict:
        """Check quantum mesh network state."""
        result = {
            "name": "Quantum Mesh Network",
            "healthy": True,
            "issues": [],
            "files": {}
        }

        state_file = self.root / QUANTUM_MESH_STATE
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
                result["files"]["state"] = {
                    "exists": True,
                    "size_bytes": state_file.stat().st_size,
                    "nodes": state.get("nodes", 0),
                    "channels": state.get("channels", 0),
                    "avg_fidelity": round(state.get("avg_fidelity", 0.0), 4),
                }
            except Exception as e:
                result["issues"].append(f"State file corrupt: {e}")
                result["healthy"] = False
        else:
            result["files"]["state"] = {"exists": False}

        return result

    def run_full_diagnostic(self) -> Dict:
        """Run complete diagnostic on all daemons."""
        report = {
            "timestamp": time.time(),
            "host_pid": os.getpid(),
            "daemons": {}
        }

        # Check VQPU Micro Daemon
        report["daemons"]["vqpu_micro"] = self.check_vqpu_micro_daemon()

        # Check Quantum AI Daemon
        report["daemons"]["quantum_ai"] = self.check_quantum_ai_daemon()

        # Check Quantum Mesh
        report["daemons"]["quantum_mesh"] = self.check_quantum_mesh()

        # Summary
        all_healthy = all(d.get("healthy", False)
                         for d in report["daemons"].values())
        report["summary"] = {
            "total_daemons": len(report["daemons"]),
            "healthy_count": sum(1 for d in report["daemons"].values()
                                if d.get("healthy", False)),
            "all_healthy": all_healthy,
            "issues": self.issues,
            "warnings": self.warnings,
        }

        return report


# ════════════════════════════════════════════════════════════════════════════════
# REPAIR FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

class DaemonRepair:
    """Automated daemon repair utilities."""

    def __init__(self, root_path: Path = None, dry_run: bool = False):
        self.root = Path(root_path or os.getcwd())
        self.dry_run = dry_run
        self.actions_taken: List[str] = []

    def clean_stale_pid_files(self) -> Dict:
        """Remove stale PID files."""
        result = {"cleaned": 0, "failed": 0, "errors": []}

        pid_file = VQPU_MICRO_PID_FILE
        if pid_file.exists():
            try:
                old_pid = int(pid_file.read_text().strip())
                # Check if process is alive
                try:
                    os.kill(old_pid, 0)
                    # Process still alive — don't delete
                    return {"note": "PID refers to live process", **result}
                except ProcessLookupError:
                    # Process is dead — safe to remove
                    if not self.dry_run:
                        pid_file.unlink()
                    result["cleaned"] += 1
                    self.actions_taken.append(f"Cleaned stale PID file: {pid_file}")
            except Exception as e:
                result["errors"].append(str(e))
                result["failed"] += 1

        return result

    def reset_crash_count(self) -> Dict:
        """Reset crash count to 0."""
        result = {"reset": False, "error": None}

        state_file = self.root / VQPU_MICRO_STATE_FILE
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
                old_count = state.get("crash_count", 0)
                state["crash_count"] = 0
                if not self.dry_run:
                    state_file.write_text(json.dumps(state, indent=2))
                result["reset"] = True
                result["previous_count"] = old_count
                self.actions_taken.append(
                    f"Reset crash count from {old_count} to 0")
            except Exception as e:
                result["error"] = str(e)

        return result

    def clean_ipc_queue(self, max_age_seconds: int = 3600) -> Dict:
        """Clean old IPC job files."""
        result = {"inbox_cleaned": 0, "outbox_cleaned": 0, "errors": []}

        current_time = time.time()

        # Clean inbox
        inbox = MICRO_BRIDGE_PATH / "inbox"
        if inbox.exists():
            for f in inbox.glob("*.json"):
                age = current_time - f.stat().st_mtime
                if age > max_age_seconds:
                    try:
                        if not self.dry_run:
                            f.unlink()
                        result["inbox_cleaned"] += 1
                    except Exception as e:
                        result["errors"].append(str(e))

        # Clean outbox
        outbox = MICRO_BRIDGE_PATH / "outbox"
        if outbox.exists():
            for f in outbox.glob("*.json"):
                age = current_time - f.stat().st_mtime
                if age > max_age_seconds * 2:  # Keep outbox longer
                    try:
                        if not self.dry_run:
                            f.unlink()
                        result["outbox_cleaned"] += 1
                    except Exception as e:
                        result["errors"].append(str(e))

        if result["inbox_cleaned"] > 0 or result["outbox_cleaned"] > 0:
            self.actions_taken.append(
                f"Cleaned {result['inbox_cleaned']} inbox + "
                f"{result['outbox_cleaned']} outbox files")

        return result

    def full_repair(self) -> Dict:
        """Run all repair operations."""
        return {
            "pid_cleanup": self.clean_stale_pid_files(),
            "crash_count_reset": self.reset_crash_count(),
            "ipc_cleanup": self.clean_ipc_queue(),
            "actions": self.actions_taken,
        }


# ════════════════════════════════════════════════════════════════════════════════
# MAIN CLI
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="L104 Daemon Cleanup & Repair Utility")
    parser.add_argument("--status", action="store_true",
                       help="Check daemon health")
    parser.add_argument("--clean", action="store_true",
                       help="Clean stale PID and IPC files")
    parser.add_argument("--reset", action="store_true",
                       help="Reset crash counts to 0 (careful!)")
    parser.add_argument("--repair", action="store_true",
                       help="Auto-repair detected issues")
    parser.add_argument("--full", action="store_true",
                       help="Full diagnostic + auto-repair")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done, don't actually do it")
    parser.add_argument("--root", type=Path, default=None,
                       help="L104 root directory (default: cwd)")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON")

    args = parser.parse_args()

    # Run diagnostics
    diag = DaemonDiagnostics(args.root)
    diagnostic = diag.run_full_diagnostic()

    if args.status or args.full or (not any([args.clean, args.reset, args.repair])):
        # Print diagnostic report
        if args.json:
            print(json.dumps(diagnostic, indent=2, default=str))
        else:
            _print_diagnostic_report(diagnostic)

    if args.clean or args.repair or args.full:
        # Run repairs
        repair = DaemonRepair(args.root, dry_run=args.dry_run)
        repair_result = repair.full_repair() if args.full else {
            "pid_cleanup": repair.clean_stale_pid_files() if args.clean else None,
            "crash_reset": repair.reset_crash_count() if args.reset else None,
        }

        if args.json:
            print(json.dumps(repair_result, indent=2, default=str))
        else:
            _print_repair_report(repair_result, args.dry_run)

    return 0


def _print_diagnostic_report(report: Dict):
    """Pretty-print diagnostic report."""
    print("\n" + "=" * 80)
    print("L104 DAEMON DIAGNOSTIC REPORT")
    print("=" * 80)

    for daemon_name, daemon_info in report.get("daemons", {}).items():
        health = "✅ HEALTHY" if daemon_info.get("healthy") else "❌ ISSUES"
        print(f"\n{daemon_info['name']:30} {health}")

        if daemon_info.get("issues"):
            print("  Issues:")
            for issue in daemon_info["issues"]:
                print(f"    ⚠️  {issue}")

        for file_name, file_info in daemon_info.get("files", {}).items():
            if file_info.get("exists"):
                print(f"  ✓ {file_name}: {file_info}")

    summary = report.get("summary", {})
    print("\n" + "─" * 80)
    print(f"Status: {summary.get('healthy_count')}/{summary.get('total_daemons')} healthy")
    if summary.get("warnings"):
        print("\nWarnings:")
        for w in summary.get("warnings", []):
            print(f"  ⚠️  {w}")


def _print_repair_report(report: Dict, dry_run: bool):
    """Pretty-print repair report."""
    print("\n" + "=" * 80)
    if dry_run:
        print("L104 DAEMON REPAIR (DRY RUN — no changes made)")
    else:
        print("L104 DAEMON REPAIR COMPLETED")
    print("=" * 80)

    for op_name, op_result in report.items():
        if op_result is None:
            continue
        print(f"\n{op_name}:")
        if isinstance(op_result, dict):
            for key, val in op_result.items():
                print(f"  {key}: {val}")
        else:
            print(f"  {op_result}")


if __name__ == "__main__":
    sys.exit(main())
