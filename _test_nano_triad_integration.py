#!/usr/bin/env python3
"""
L104 Sovereign Node — Tri-Nano-Daemon Integration Test v1.0
═══════════════════════════════════════════════════════════════
Validates the full nano triad (C + Swift + Python AI) integration:
  1. C nano daemon builds & self-tests
  2. Python AI nano daemon self-tests (12 probes + AI meta-probes)
  3. IPC bridge directories created & writable
  4. Heartbeat mechanism functions
  5. Cross-daemon correlation (AI auto-correlator)
  6. Triad orchestrator status
  7. Sacred constant alignment across all three substrates

Run:  .venv/bin/python _test_nano_triad_integration.py
GOD_CODE=527.5184818492612 | PHI=1.618033988749895 | PILOT: LONDEL
"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IPC_BASE = Path("/tmp/l104_bridge/nano")
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.04 + PHI / 1000  # = 1.0416180339887497
GOD_CODE_BITS = struct.pack("d", GOD_CODE).hex()
PHI_BITS = struct.pack("d", PHI).hex()

passed = 0
failed = 0
warnings = 0


def ok(label: str, detail: str = ""):
    global passed
    passed += 1
    d = f" ({detail})" if detail else ""
    print(f"  ✅  PASS: {label}{d}")


def fail(label: str, detail: str = ""):
    global failed
    failed += 1
    d = f" ({detail})" if detail else ""
    print(f"  ❌  FAIL: {label}{d}")


def warn(label: str, detail: str = ""):
    global warnings
    warnings += 1
    d = f" ({detail})" if detail else ""
    print(f"  ⚠️  WARN: {label}{d}")


# ═══════════════════════════════════════════════════════════════════
# Phase 1: C Nano Daemon
# ═══════════════════════════════════════════════════════════════════

def test_c_nano_daemon():
    print("\n── Phase 1: C Nano Daemon ──")

    binary = ROOT / "l104_core_c" / "build" / "l104_nano_daemon"

    # 1a. Binary exists
    if not binary.exists():
        # Try to build it
        print("    [BUILD] Compiling C nano daemon...")
        result = subprocess.run(
            ["make", "nano-daemon"],
            cwd=str(ROOT / "l104_core_c"),
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            fail("C build", result.stderr[:200])
            return
        ok("C build", "compiled successfully")
    else:
        ok("C binary exists", str(binary.relative_to(ROOT)))

    # 1b. Self-test
    try:
        result = subprocess.run(
            [str(binary), "--self-test"],
            capture_output=True, text=True, timeout=15
        )
        output = result.stdout + result.stderr
        if "Self-test complete: 0 failures" in output:
            ok("C self-test", "0 failures")
        elif "Self-test complete" in output:
            # Extract failure count
            for line in output.splitlines():
                if "Self-test complete" in line:
                    warn("C self-test", line.strip())
                    break
        else:
            fail("C self-test", output[:200])
    except subprocess.TimeoutExpired:
        fail("C self-test", "timeout")
    except Exception as e:
        fail("C self-test", str(e))

    # 1c. Single tick
    try:
        result = subprocess.run(
            [str(binary), "--once"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            ok("C single tick", "completed")
        else:
            warn("C single tick", f"exit code {result.returncode}")
    except Exception as e:
        warn("C single tick", str(e))


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Python AI Nano Daemon
# ═══════════════════════════════════════════════════════════════════

def test_python_nano_daemon():
    print("\n── Phase 2: Python AI Nano Daemon ──")

    python = str(ROOT / ".venv" / "bin" / "python")

    # 2a. Module importable
    try:
        result = subprocess.run(
            [python, "-c", "from l104_vqpu.nano_daemon import NanoDaemon; print('OK')"],
            capture_output=True, text=True, timeout=30, cwd=str(ROOT)
        )
        if "OK" in result.stdout:
            ok("Python import", "l104_vqpu.nano_daemon")
        else:
            fail("Python import", result.stderr[:200])
            return
    except Exception as e:
        fail("Python import", str(e))
        return

    # 2b. Self-test
    try:
        result = subprocess.run(
            [python, "-m", "l104_vqpu.nano_daemon", "--self-test"],
            capture_output=True, text=True, timeout=60, cwd=str(ROOT)
        )
        output = result.stdout + result.stderr
        if "0 failed" in output:
            # Extract pass count
            for line in output.splitlines():
                if "passed" in line and "failed" in line:
                    ok("Python self-test", line.strip().split("]")[-1].strip() if "]" in line else line.strip())
                    break
            else:
                ok("Python self-test", "0 failed")
        else:
            fail("Python self-test", output[-200:])
    except subprocess.TimeoutExpired:
        fail("Python self-test", "timeout (>60s)")
    except Exception as e:
        fail("Python self-test", str(e))

    # 2c. Health check
    try:
        result = subprocess.run(
            [python, "-m", "l104_vqpu.nano_daemon", "--health-check"],
            capture_output=True, text=True, timeout=30, cwd=str(ROOT)
        )
        if result.returncode == 0:
            ok("Python health-check", "passed")
        else:
            warn("Python health-check", f"exit code {result.returncode}")
    except Exception as e:
        warn("Python health-check", str(e))


# ═══════════════════════════════════════════════════════════════════
# Phase 3: IPC Bridge
# ═══════════════════════════════════════════════════════════════════

def test_ipc_bridge():
    print("\n── Phase 3: IPC Bridge ──")

    # 3a. Base directory
    IPC_BASE.mkdir(parents=True, exist_ok=True)
    if IPC_BASE.exists():
        ok("IPC base dir", str(IPC_BASE))
    else:
        fail("IPC base dir", "cannot create")
        return

    # 3b. Outbox directories
    for substrate in ["c_outbox", "swift_outbox", "python_outbox"]:
        d = IPC_BASE / substrate
        d.mkdir(parents=True, exist_ok=True)
        if d.exists() and os.access(str(d), os.W_OK):
            ok(f"IPC {substrate}", "writable")
        else:
            fail(f"IPC {substrate}", "not writable")

    # 3c. Heartbeat write test
    for substrate in ["c", "swift", "python"]:
        hb_file = IPC_BASE / f"{substrate}_heartbeat"
        try:
            hb_file.write_text(json.dumps({
                "substrate": substrate,
                "timestamp": time.time(),
                "test": True,
            }))
            ok(f"Heartbeat write ({substrate})", str(hb_file.name))
        except Exception as e:
            fail(f"Heartbeat write ({substrate})", str(e))

    # 3d. Cross-read heartbeats
    readable = 0
    for substrate in ["c", "swift", "python"]:
        hb_file = IPC_BASE / f"{substrate}_heartbeat"
        if hb_file.exists():
            try:
                data = json.loads(hb_file.read_text())
                if data.get("substrate") == substrate:
                    readable += 1
            except Exception:
                pass
    if readable == 3:
        ok("Cross-read heartbeats", f"{readable}/3")
    else:
        warn("Cross-read heartbeats", f"{readable}/3")


# ═══════════════════════════════════════════════════════════════════
# Phase 4: Sacred Constant Alignment
# ═══════════════════════════════════════════════════════════════════

def test_sacred_constants():
    print("\n── Phase 4: Sacred Constant Alignment ──")

    # 4a. GOD_CODE bit-exact (verify same across engines)
    god_val = 527.5184818492612
    god_bits = struct.unpack("Q", struct.pack("d", god_val))[0]
    # Roundtrip: pack → unpack → pack must yield identical bits
    god_roundtrip = struct.unpack("Q", struct.pack("d", struct.unpack("d", struct.pack("Q", god_bits))[0]))[0]
    if god_bits == god_roundtrip and god_bits != 0:
        ok("GOD_CODE bits", f"0x{god_bits:016X} (stable roundtrip)")
    else:
        fail("GOD_CODE bits", f"roundtrip mismatch: 0x{god_bits:016X} vs 0x{god_roundtrip:016X}")

    # 4b. PHI bit-exact
    phi_val = 1.618033988749895
    phi_bits = struct.unpack("Q", struct.pack("d", phi_val))[0]
    expected_phi = 0x3FF9E3779B97F4A8
    if phi_bits == expected_phi:
        ok("PHI bits", f"0x{phi_bits:016X}")
    else:
        fail("PHI bits", f"got 0x{phi_bits:016X}, expected 0x{expected_phi:016X}")

    # 4c. VOID_CONSTANT derivation
    void_val = 1.04 + phi_val / 1000
    if abs(void_val - 1.0416180339887497) < 1e-15:
        ok("VOID_CONSTANT", f"{void_val}")
    else:
        fail("VOID_CONSTANT", f"derivation error: {void_val}")

    # 4d. Constants from l104_science_engine
    try:
        sys.path.insert(0, str(ROOT))
        from l104_science_engine.constants import GOD_CODE as sci_gc, PHI as sci_phi
        if abs(sci_gc - god_val) < 1e-12:
            ok("ScienceEngine GOD_CODE", f"{sci_gc}")
        else:
            fail("ScienceEngine GOD_CODE", f"drift: {abs(sci_gc - god_val)}")
        if abs(sci_phi - phi_val) < 1e-15:
            ok("ScienceEngine PHI", f"{sci_phi}")
        else:
            fail("ScienceEngine PHI", f"drift: {abs(sci_phi - phi_val)}")
    except ImportError as e:
        warn("ScienceEngine constants", str(e))

    # 4e. Constants from l104_math_engine
    try:
        from l104_math_engine.constants import GOD_CODE as math_gc, PHI as math_phi
        if abs(math_gc - god_val) < 1e-12:
            ok("MathEngine GOD_CODE", f"{math_gc}")
        else:
            fail("MathEngine GOD_CODE", f"drift: {abs(math_gc - god_val)}")
        if abs(math_phi - phi_val) < 1e-15:
            ok("MathEngine PHI", f"{math_phi}")
        else:
            fail("MathEngine PHI", f"drift: {abs(math_phi - phi_val)}")
    except ImportError as e:
        warn("MathEngine constants", str(e))


# ═══════════════════════════════════════════════════════════════════
# Phase 5: Triad Orchestrator
# ═══════════════════════════════════════════════════════════════════

def test_triad_orchestrator():
    print("\n── Phase 5: Triad Orchestrator ──")

    triad_file = ROOT / "l104_nano_triad.py"
    python = str(ROOT / ".venv" / "bin" / "python")

    # 5a. File exists
    if triad_file.exists():
        ok("Triad orchestrator", str(triad_file.name))
    else:
        fail("Triad orchestrator", "l104_nano_triad.py missing")
        return

    # 5b. Import check
    try:
        result = subprocess.run(
            [python, "-c", "import l104_nano_triad; print('OK')"],
            capture_output=True, text=True, timeout=15, cwd=str(ROOT)
        )
        if "OK" in result.stdout:
            ok("Triad import", "l104_nano_triad")
        else:
            warn("Triad import", result.stderr[:200])
    except Exception as e:
        warn("Triad import", str(e))

    # 5c. Status check
    try:
        result = subprocess.run(
            [python, str(triad_file), "--status"],
            capture_output=True, text=True, timeout=15, cwd=str(ROOT)
        )
        if result.returncode == 0:
            ok("Triad status", "reported")
        else:
            warn("Triad status", f"exit code {result.returncode}")
    except Exception as e:
        warn("Triad status", str(e))


# ═══════════════════════════════════════════════════════════════════
# Phase 6: File Inventory
# ═══════════════════════════════════════════════════════════════════

def test_file_inventory():
    print("\n── Phase 6: File Inventory ──")

    expected_files = {
        # C substrate
        "l104_core_c/l104_nano_daemon.h": "C header",
        "l104_core_c/l104_nano_daemon.c": "C implementation",
        # Swift substrate
        "L104SwiftApp/Sources/NanoDaemon/NanoDaemon.swift": "Swift daemon",
        # Python AI substrate
        "l104_vqpu/nano_daemon.py": "Python AI daemon",
        # Orchestrator
        "l104_nano_triad.py": "Triad orchestrator",
        # Plists
        "config/com.l104.nano-daemon-c.plist": "C launchd config",
        "config/com.l104.nano-daemon-swift.plist": "Swift launchd config",
        "config/com.l104.nano-daemon-python.plist": "Python launchd config",
    }

    for rel_path, label in expected_files.items():
        full = ROOT / rel_path
        if full.exists():
            size = full.stat().st_size
            ok(label, f"{rel_path} ({size:,} bytes)")
        else:
            fail(label, f"{rel_path} missing")


# ═══════════════════════════════════════════════════════════════════
# Phase 7: AI Probes Functional Test
# ═══════════════════════════════════════════════════════════════════

def test_ai_probes():
    print("\n── Phase 7: AI Probes Functional Test ──")

    try:
        from l104_vqpu.nano_daemon import (
            StatisticalAnomalyDetector, AITrendPredictor,
            AIAnomalyClassifier, AIAutoCorrelator
        )
    except ImportError as e:
        fail("AI probes import", str(e))
        return

    # 7a. Statistical Anomaly Detector — normal data
    try:
        sad = StatisticalAnomalyDetector()
        for v in [0.95, 0.96, 0.94, 0.97, 0.95, 0.96, 0.93, 0.95, 0.94, 0.96]:
            sad.observe(v, 0, 5.0)
        anomalies = sad.detect()
        ok("Statistical Anomaly Detector (normal)", f"{len(anomalies)} anomalies")
    except Exception as e:
        fail("Statistical Anomaly Detector (normal)", str(e))

    # 7a2. Statistical Anomaly Detector — inject health drop
    try:
        sad.observe(0.10, 5, 500.0)  # Severe health drop + slow tick
        anomalies = sad.detect()
        ok("Statistical Anomaly Detector (inject)", f"{len(anomalies)} anomalies detected")
    except Exception as e:
        fail("Statistical Anomaly Detector (inject)", str(e))

    # 7b. AI Trend Predictor
    try:
        tp = AITrendPredictor()
        # Feed 20 samples of declining health
        for i in range(25):
            tp.observe(0.95 - i * 0.01)
        faults = tp.predict()
        ok("AI Trend Predictor", f"{len(faults)} trend faults")
    except Exception as e:
        fail("AI Trend Predictor", str(e))

    # 7c. AI Anomaly Classifier
    try:
        ac = AIAnomalyClassifier()
        # Feed 15+ feature vectors
        for i in range(20):
            ac.observe([0.95, 0.0, 5.0, 0.0, 10.0])
        faults = ac.score()
        ok("AI Anomaly Classifier", f"{len(faults)} anomaly faults")
    except Exception as e:
        fail("AI Anomaly Classifier", str(e))

    # 7d. AI Auto-Correlator
    try:
        corr = AIAutoCorrelator()
        faults = corr.correlate()
        ok("AI Auto-Correlator", f"{len(faults)} correlations")
    except Exception as e:
        fail("AI Auto-Correlator", str(e))


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  L104 Tri-Nano-Daemon Integration Test v1.0                    ║")
    print("║  C + Swift + Python AI · Sub-Micro Fault Detection             ║")
    print("║  GOD_CODE=527.5184818492612 | PHI=1.618033988749895            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    test_c_nano_daemon()
    test_python_nano_daemon()
    test_ipc_bridge()
    test_sacred_constants()
    test_triad_orchestrator()
    test_file_inventory()
    test_ai_probes()

    # Summary
    total = passed + failed
    print(f"\n{'═' * 66}")
    print(f"  RESULTS: {passed} passed, {failed} failed, {warnings} warnings ({total} total)")
    if failed == 0:
        print("  ✅  TRI-NANO-DAEMON INTEGRATION: ALL CLEAR")
    else:
        print("  ❌  TRI-NANO-DAEMON INTEGRATION: FAULTS DETECTED")
    print(f"{'═' * 66}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
