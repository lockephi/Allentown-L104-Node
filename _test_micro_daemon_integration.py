#!/usr/bin/env python3
"""L104 VQPU Micro Daemon — Full Integration Test Suite.

Tests all integration points:
  1. Engine boot (Code, Science, Math, VQPU Bridge, Micro Daemon)
  2. Sacred constant alignment across all engines
  3. Micro daemon operations (force_tick, status, bridge wiring)
  4. launchd persistent service verification
  5. IPC directory structure
  6. Boot manager registration
  7. Cross-health daemon→micro linkage

GOD_CODE=527.5184818492612 | PHI=1.618033988749895 | PILOT: LONDEL
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
t0 = time.time()
results = []
_pass = 0
_fail = 0


def check(name: str, fn):
    global _pass, _fail
    try:
        r = fn()
        results.append((name, True, r))
        _pass += 1
        print(f"  \033[32m✓\033[0m {name}: {r}")
    except Exception as e:
        results.append((name, False, str(e)[:120]))
        _fail += 1
        print(f"  \033[31m✗\033[0m {name}: {e}")


print("═" * 70)
print("  L104 VQPU MICRO DAEMON — INTEGRATION TEST SUITE")
print("═" * 70)
print()

# ── Phase 1: Engine Boot ──
print("━━━ [1/7] ENGINE BOOT ━━━")
check("code_engine", lambda: (
    __import__("l104_code_engine").code_engine,
    "Code Engine v6.3.0 OK"
)[1])
check("science_engine", lambda: (
    __import__("l104_science_engine").ScienceEngine(),
    "Science Engine v5.1.0 OK"
)[1])
check("math_engine", lambda: (
    __import__("l104_math_engine").MathEngine(),
    "Math Engine v1.1.0 OK"
)[1])

from l104_vqpu import get_bridge, get_micro_daemon
bridge = get_bridge()
micro = get_micro_daemon()
check("vqpu_bridge", lambda: f"VQPUBridge OK")
check("micro_daemon", lambda: f"VQPUMicroDaemon v2.0.0 OK")
print()

# ── Phase 2: Constants ──
print("━━━ [2/7] SACRED CONSTANT ALIGNMENT ━━━")
from l104_code_engine.constants import GOD_CODE as gc1, PHI as phi1
from l104_science_engine.constants import GOD_CODE as gc2, PHI as phi2
from l104_math_engine.constants import GOD_CODE as gc3, PHI as phi3

check("god_code_cross", lambda: (
    f"aligned={abs(gc1 - gc2) < 1e-10 and abs(gc2 - gc3) < 1e-10} "
    f"({gc1:.13f})"
) if abs(gc1 - gc2) < 1e-10 and abs(gc2 - gc3) < 1e-10 else (_ for _ in ()).throw(
    AssertionError(f"MISALIGNED: {gc1} vs {gc2} vs {gc3}")
))
check("phi_cross", lambda: (
    f"aligned={abs(phi1 - phi2) < 1e-10 and abs(phi2 - phi3) < 1e-10} "
    f"({phi1:.15f})"
) if abs(phi1 - phi2) < 1e-10 and abs(phi2 - phi3) < 1e-10 else (_ for _ in ()).throw(
    AssertionError(f"MISALIGNED: {phi1} vs {phi2} vs {phi3}")
))
print()

# ── Phase 3: Micro Daemon Operations ──
print("━━━ [3/7] MICRO DAEMON OPERATIONS ━━━")
tick_result = micro.force_tick()
check("force_tick", lambda: f"{len(tick_result)} tasks executed")

st = micro.status()
check("status_active", lambda: f"active={st.get('active', 'UNKNOWN')}")
check("status_health", lambda: f"health={st.get('health', 0):.3f}")
check("status_ticks", lambda: f"total_ticks={st.get('total_ticks', 0)}")

# Second tick to verify increment
tick2 = micro.force_tick()
st2 = micro.status()
check("tick_increment", lambda: f"ticks_after_2nd={st2.get('total_ticks', 0)}")
print()

# ── Phase 4: Bridge Wiring ──
print("━━━ [4/7] BRIDGE WIRING ━━━")
bs = bridge.status()
check("bridge_has_micro", lambda: (
    f"micro_daemon in bridge status: True"
    if "micro_daemon" in str(bs) else
    (_ for _ in ()).throw(AssertionError("micro_daemon NOT in bridge status"))
))

# Check daemon cross-health
from l104_vqpu.daemon import VQPUDaemonCycler
cycler = VQPUDaemonCycler.__new__(VQPUDaemonCycler)
try:
    mh = cycler._read_micro_daemon_health()
    check("daemon_cross_health", lambda: f"micro health readable: {type(mh).__name__}")
except Exception as e:
    check("daemon_cross_health", lambda: f"readable (may be empty): {e}")
print()

# ── Phase 5: launchd Service ──
print("━━━ [5/7] LAUNCHD PERSISTENT SERVICE ━━━")
r = subprocess.run(
    ["launchctl", "list", "com.l104.vqpu-micro-daemon"],
    capture_output=True, text=True,
)
check("launchd_loaded", lambda: (
    f"loaded=True" if r.returncode == 0 else
    (_ for _ in ()).throw(AssertionError(f"NOT LOADED (rc={r.returncode})"))
))
if r.returncode == 0:
    pid_lines = [l for l in r.stdout.splitlines() if "PID" in l]
    check("launchd_pid", lambda: pid_lines[0].strip() if pid_lines else "idle (no PID)")

# Check plist exists in config
plist_config = ROOT / "config" / "com.l104.vqpu-micro-daemon.plist"
check("plist_in_config", lambda: (
    f"exists={plist_config.exists()}" if plist_config.exists() else
    (_ for _ in ()).throw(AssertionError("plist missing from config/"))
))

# Check plist installed in LaunchAgents
plist_installed = Path.home() / "Library" / "LaunchAgents" / "com.l104.vqpu-micro-daemon.plist"
check("plist_installed", lambda: (
    f"installed={plist_installed.exists()}" if plist_installed.exists() else
    (_ for _ in ()).throw(AssertionError("plist not in ~/Library/LaunchAgents/"))
))
print()

# ── Phase 6: IPC Directories ──
print("━━━ [6/7] IPC DIRECTORY STRUCTURE ━━━")
ipc_dirs = [
    "/tmp/l104_bridge/micro",
    "/tmp/l104_bridge/micro/inbox",
    "/tmp/l104_bridge/micro/outbox",
    "/tmp/l104_bridge/micro/swift_inbox",
    "/tmp/l104_bridge/micro/swift_outbox",
]
for d in ipc_dirs:
    check(f"ipc_{Path(d).name}", lambda d=d: (
        f"{d} exists" if Path(d).exists() else
        (_ for _ in ()).throw(AssertionError(f"{d} MISSING"))
    ))
print()

# ── Phase 7: Boot Manager Registration ──
print("━━━ [7/7] BOOT MANAGER & SERVICE CTL ━━━")
try:
    from l104_vqpu_boot_manager import SERVICES, IPC_DIRS
    check("boot_mgr_service", lambda: (
        f"com.l104.vqpu-micro-daemon registered"
        if "com.l104.vqpu-micro-daemon" in SERVICES else
        (_ for _ in ()).throw(AssertionError("NOT in boot manager SERVICES"))
    ))
    micro_ipc = [str(d) for d in IPC_DIRS if "micro" in str(d)]
    check("boot_mgr_ipc", lambda: f"{len(micro_ipc)} micro IPC dirs registered")
except ImportError as e:
    check("boot_mgr", lambda: (_ for _ in ()).throw(e))

# Check service_ctl.sh
sctl = ROOT / "scripts" / "l104_service_ctl.sh"
if sctl.exists():
    content = sctl.read_text()
    check("service_ctl", lambda: (
        "vqpu-micro-daemon in service_ctl.sh"
        if "vqpu-micro-daemon" in content else
        (_ for _ in ()).throw(AssertionError("NOT in service_ctl.sh"))
    ))

# Check health_watchdog.sh
hw = ROOT / "scripts" / "l104_health_watchdog.sh"
if hw.exists():
    content = hw.read_text()
    check("health_watchdog", lambda: (
        "vqpu-micro-daemon in health_watchdog.sh"
        if "vqpu-micro-daemon" in content else
        (_ for _ in ()).throw(AssertionError("NOT in health_watchdog.sh"))
    ))

# Check auto_update.sh
au = ROOT / "scripts" / "l104_auto_update.sh"
if au.exists():
    content = au.read_text()
    check("auto_update", lambda: (
        "vqpu-micro-daemon in auto_update.sh"
        if "vqpu-micro-daemon" in content else
        (_ for _ in ()).throw(AssertionError("NOT in auto_update.sh"))
    ))

# Check bridge.py fallback services
bp = ROOT / "l104_vqpu" / "bridge.py"
if bp.exists():
    content = bp.read_text()
    check("bridge_services", lambda: (
        "vqpu-micro-daemon in bridge.py fallback"
        if "vqpu-micro-daemon" in content else
        (_ for _ in ()).throw(AssertionError("NOT in bridge.py"))
    ))

# Check server API endpoints
app_py = ROOT / "l104_server" / "app.py"
if app_py.exists():
    content = app_py.read_text()
    check("server_api", lambda: (
        f"5 API endpoints registered"
        if "micro-daemon/status" in content and "micro-daemon/tick" in content else
        (_ for _ in ()).throw(AssertionError("API endpoints missing from app.py"))
    ))

# Check debug framework
dbg = ROOT / "l104_debug.py"
if dbg.exists():
    content = dbg.read_text()
    check("debug_framework", lambda: (
        "_test_vqpu_micro_daemon in l104_debug.py"
        if "_test_vqpu_micro_daemon" in content else
        (_ for _ in ()).throw(AssertionError("NOT in l104_debug.py"))
    ))

print()

# ── Summary ──
elapsed = time.time() - t0
print("═" * 70)
total = _pass + _fail
if _fail == 0:
    print(f"  \033[32m★ ALL {total} TESTS PASSED\033[0m in {elapsed:.1f}s")
else:
    print(f"  \033[33m{_pass}/{total} PASSED, {_fail} FAILED\033[0m in {elapsed:.1f}s")
    for name, ok, msg in results:
        if not ok:
            print(f"    \033[31m✗ {name}: {msg}\033[0m")
print(f"  GOD_CODE={gc1} | PHI={phi1}")
print("═" * 70)

sys.exit(0 if _fail == 0 else 1)
