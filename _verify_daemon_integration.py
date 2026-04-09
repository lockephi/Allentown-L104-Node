#!/usr/bin/env python3
"""
Quick Verification: Daemon Integration
═════════════════════════════════════════════════════════════════════════════

Verifies that all three daemons have orchestrator integration code in place.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def check_file_for_string(file_path, search_strings, name):
    """Check if file contains all search strings."""
    try:
        content = Path(file_path).read_text()
        found = {}
        for search_str in search_strings:
            found[search_str] = search_str in content

        all_found = all(found.values())
        status = "✓" if all_found else "✗"
        print(f"{status} {name}")

        if not all_found:
            for search_str, result in found.items():
                mark = "✓" if result else "✗"
                print(f"  {mark} {search_str}")

        return all_found
    except Exception as e:
        print(f"✗ {name} - Error: {e}")
        return False

def main():
    print("=" * 70)
    print("DAEMON ORCHESTRATOR INTEGRATION VERIFICATION")
    print("=" * 70)
    print()

    checks = {
        "VQPU Daemon": {
            "file": "l104_vqpu/daemon.py",
            "strings": [
                "from l104_daemon_adapter import DaemonAdapter",
                "self._adapter = None",
                "self._orchestrator = None",
                "def set_orchestrator",
                "self._adapter.on_cycle_start()",
                "self._adapter.on_cycle_end(",
                "self._adapter.emit_fidelity_alert(",
                "self._adapter.emit_error(",
            ]
        },
        "QuantumAI Daemon": {
            "file": "l104_quantum_ai_daemon/daemon.py",
            "strings": [
                "from l104_daemon_adapter import DaemonAdapter",
                "self._adapter = None",
                "self._orchestrator = None",
                "def set_orchestrator",
                "self._adapter.on_cycle_start()",
                "self._adapter.on_cycle_end(",
                "self._adapter.emit_fidelity_alert(",
                "self._adapter.emit_error(",
            ]
        },
        "Soul Daemon": {
            "file": "l104_soul_daemon/daemon.py",
            "strings": [
                "from l104_daemon_adapter import DaemonAdapter",
                "self.adapter = None",
                "self.orchestrator = None",
                "def set_orchestrator",
                "self.adapter.on_cycle_start()",
                "self.adapter.on_cycle_end(",
                "self.adapter.emit_fidelity_alert(",
                "self.adapter.emit_error(",
            ]
        },
    }

    results = {}
    for daemon_name, check_info in checks.items():
        print(f"\n{daemon_name}:")
        print("-" * 70)
        results[daemon_name] = check_file_for_string(
            check_info["file"],
            check_info["strings"],
            daemon_name
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for daemon_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {daemon_name}")

    print()
    print(f"Result: {passed}/{total} daemons fully integrated")

    if passed == total:
        print("\n✓ All daemon integrations verified!")
        return 0
    else:
        print(f"\n✗ {total - passed} daemon(s) need integration")
        return 1

if __name__ == "__main__":
    sys.exit(main())
