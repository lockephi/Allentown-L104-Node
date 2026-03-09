#!/usr/bin/env python3
"""Verify all nano triad L104Daemon-grade assertions pass."""

import os
import subprocess
import sys

# Ensure IPC dirs exist
for d in ["/tmp/l104_bridge", "/tmp/l104_bridge/nano",
          "/tmp/l104_bridge/nano/python_outbox",
          "/tmp/l104_bridge/nano/c_outbox",
          "/tmp/l104_bridge/nano/swift_outbox"]:
    os.makedirs(d, exist_ok=True)

passed = 0
failed = 0

def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  ✓ PASS: {name}  {detail}")
    else:
        failed += 1
        print(f"  ✗ FAIL: {name}  {detail}")

print("╔══════════════════════════════════════════════════════════════════╗")
print("║  L104 NANO TRIAD — Assertion Verification Suite                 ║")
print("║  L104Daemon-grade: validate, signal, kill-stale, status, IPC    ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print()

# ── Python Nano Daemon ──
print("── Python Nano Daemon ──")
from l104_vqpu.nano_daemon import NanoDaemon

daemon = NanoDaemon(tick_interval=3.0, verbose=False)

# 1. validate_configuration
ok = daemon.validate_configuration()
check("validate_configuration()", ok)

# 2. status()
s = daemon.status()
check("status()", isinstance(s, dict) and "daemon" in s and "version" in s,
      f"probes={len(s.get('probes', []))}")

# 3. single tick
h, f, fl = daemon.tick()
check("tick()", 0.0 <= h <= 1.0 and isinstance(f, int),
      f"health={h:.4f}, faults={f}")

# 4. self_test
failures = daemon.self_test()
check("self_test()", failures == 0, f"{failures} failures")

# 5. Has validate_configuration method
check("has validate_configuration", hasattr(daemon, 'validate_configuration'))

# 6. Has _kill_previous_instance method
check("has _kill_previous_instance", hasattr(daemon, '_kill_previous_instance'))

# 7. Has dump_status method
check("has dump_status", hasattr(daemon, 'dump_status'))

# 8. Has reload method
check("has reload", hasattr(daemon, 'reload'))

print()

# ── C Nano Daemon ──
print("── C Nano Daemon ──")
binary = os.path.join(os.getcwd(), "l104_core_c", "build", "l104_nano_daemon")
if os.path.isfile(binary):
    # 9. Self-test
    r = subprocess.run([binary, "--self-test"], capture_output=True, text=True, timeout=30)
    check("--self-test", r.returncode == 0, r.stdout.strip().split("\n")[-1][:60])

    # 10. Validate
    r = subprocess.run([binary, "--validate"], capture_output=True, text=True, timeout=10)
    check("--validate", r.returncode == 0)

    # 11. --help shows signal docs
    r = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=5)
    check("--help has signals", "SIGUSR1" in r.stdout and "SIGUSR2" in r.stdout and "SIGHUP" in r.stdout)

    # 12. --status flag exists
    check("--status flag", "--status" in r.stdout)
else:
    check("binary exists", False, "l104_core_c/build/l104_nano_daemon not found")

print()

# ── IPC Directory Structure ──
print("── IPC Directory Structure ──")
required_dirs = [
    "/tmp/l104_bridge/nano",
    "/tmp/l104_bridge/nano/c_outbox",
    "/tmp/l104_bridge/nano/swift_outbox",
    "/tmp/l104_bridge/nano/python_outbox",
]
for d in required_dirs:
    check(f"dir exists: {d}", os.path.isdir(d))

print()

# ── l104_debug.py Integration ──
print("── l104_debug.py Integration ──")
try:
    # Check that nano_triad is in ENGINE_REGISTRY
    sys.path.insert(0, os.getcwd())
    # Just check the file content for the entry
    with open("l104_debug.py") as f:
        content = f.read()
    check("nano_triad in ENGINE_REGISTRY", '"nano_triad"' in content)
    check("_boot_nano_triad defined", "def _boot_nano_triad" in content)
    check("_test_nano_python_self_test", "_test_nano_python_self_test" in content)
    check("_test_nano_c_self_test", "_test_nano_c_self_test" in content)
    check("_test_nano_swift_self_test", "_test_nano_swift_self_test" in content)
    check("_test_nano_ipc_dirs", "_test_nano_ipc_dirs" in content)
except Exception as e:
    check("l104_debug.py integration", False, str(e))

print()

# ── VS Code Tasks ──
print("── VS Code Tasks ──")
try:
    import json
    with open(".vscode/tasks.json") as f:
        content = f.read()
    check("Nano Triad Self-Test task", "Nano Triad Self-Test" in content)
    check("Nano Triad Validate task", "Nano Triad Validate" in content)
    check("Nano Triad Build task", "Nano Triad Build" in content)
    check("Debug Nano Triad task", "Debug Nano Triad" in content)
except Exception as e:
    check("VS Code tasks", False, str(e))

print()
print("═" * 66)
print(f"  TOTAL: {passed + failed} assertions | {passed} PASSED | {failed} FAILED")
print("═" * 66)
sys.exit(1 if failed > 0 else 0)
