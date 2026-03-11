#!/usr/bin/env python3
"""Emergency daemon rescue script - kills stale daemons and repairs state."""

import os
import json
import signal
import time
from pathlib import Path

MICRO_PID_FILE = Path("/tmp/l104_bridge/micro/micro_daemon.pid")
VQPU_MICRO_STATE = Path(".l104_vqpu_micro_daemon.json")

def main():
    print("=" * 80)
    print("L104 DAEMON EMERGENCY RESCUE")
    print("=" * 80)

    # Read PID file
    if not MICRO_PID_FILE.exists():
        print("✅ No stale PID file found")
        return

    try:
        old_pid = int(MICRO_PID_FILE.read_text().strip())
        print(f"\n🔍 Found daemon PID: {old_pid}")

        # Check if it's alive
        try:
            os.kill(old_pid, 0)
            print(f"   Status: ALIVE (signal 0 check passed)")

            # Try to kill it gracefully first
            print(f"\n⚠️  Killing stale daemon process {old_pid}...")
            try:
                os.kill(old_pid, signal.SIGTERM)
                time.sleep(2)
                # Check if it died
                try:
                    os.kill(old_pid, 0)
                    print(f"   SIGTERM didn't work, sending SIGKILL...")
                    os.kill(old_pid, signal.SIGKILL)
                    time.sleep(1)
                except ProcessLookupError:
                    print(f"   ✅ Process terminated cleanly")
            except Exception as e:
                print(f"   ⚠️  Error: {e}")

        except ProcessLookupError:
            print(f"   Status: DEAD (no process with this PID)")

    except ValueError as e:
        print(f"⚠️  Invalid PID in file: {e}")

    # Clean PID file
    print(f"\n🧹 Cleaning PID file: {MICRO_PID_FILE}")
    try:
        MICRO_PID_FILE.unlink()
        print("   ✅ Cleaned")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")

    # Reset crash count
    print(f"\n🔄 Resetting crash count...")
    if VQPU_MICRO_STATE.exists():
        try:
            state = json.loads(VQPU_MICRO_STATE.read_text())
            old_count = state.get("crash_count", 0)
            state["crash_count"] = 0
            VQPU_MICRO_STATE.write_text(json.dumps(state, indent=2))
            print(f"   ✅ Reset from {old_count} → 0")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")

    print("\n" + "=" * 80)
    print("✅ DAEMON RESCUE COMPLETE")
    print("   The daemon can now restart cleanly.")
    print("=" * 80)

if __name__ == "__main__":
    main()
