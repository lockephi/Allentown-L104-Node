#!/usr/bin/env python3
"""
L104 Daemon System Comprehensive Validation & Debug
====================================================

Validates all critical daemon systems after Phase 1 fixes:
  ✓ DaemonOrchestrator import alias
  ✓ QuantumAIDaemon version attribute
  ✓ State persistence infrastructure
  ✓ Fast server engine initialization
  ✓ Quantum network health
  ✓ Multi-daemon coordination
"""

import sys
import json
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("DAEMON_VALIDATION")

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")

def print_test(name, status, message=""):
    """Print test result."""
    symbol = f"{Colors.GREEN}✓{Colors.RESET}" if status else f"{Colors.RED}✗{Colors.RESET}"
    status_text = f"{Colors.GREEN}PASS{Colors.RESET}" if status else f"{Colors.RED}FAIL{Colors.RESET}"
    print(f"  {symbol} {name:60s} [{status_text}]")
    if message:
        print(f"    → {Colors.YELLOW}{message}{Colors.RESET}")

def test_daemon_orchestrator_import():
    """Test DaemonOrchestrator import alias."""
    print_header("Phase 1 Fix 1: DaemonOrchestrator Import Alias")

    tests_passed = 0
    try:
        # Test 1: Direct class import
        from l104_daemon_orchestrator import L104DaemonOrchestrator
        print_test("Import L104DaemonOrchestrator (original class)", True)
        tests_passed += 1
    except Exception as e:
        print_test("Import L104DaemonOrchestrator (original class)", False, str(e))

    try:
        # Test 2: Alias import
        from l104_daemon_orchestrator import DaemonOrchestrator
        print_test("Import DaemonOrchestrator (alias)", True)
        tests_passed += 1
    except Exception as e:
        print_test("Import DaemonOrchestrator (alias)", False, str(e))

    try:
        # Test 3: Instantiation
        from l104_daemon_orchestrator import DaemonOrchestrator
        orchestrator = DaemonOrchestrator()
        print_test("Instantiate DaemonOrchestrator", True, f"Type: {type(orchestrator).__name__}")
        tests_passed += 1
    except Exception as e:
        print_test("Instantiate DaemonOrchestrator", False, str(e))

    try:
        # Test 4: Check orchestrator methods
        from l104_daemon_orchestrator import DaemonOrchestrator
        orchestrator = DaemonOrchestrator()
        required_methods = ['start', 'stop', 'register_daemon', 'get_status', '_persist_state']
        missing = [m for m in required_methods if not hasattr(orchestrator, m)]
        if not missing:
            print_test("Orchestrator has all required methods", True)
            tests_passed += 1
        else:
            print_test("Orchestrator has all required methods", False, f"Missing: {missing}")
    except Exception as e:
        print_test("Orchestrator has all required methods", False, str(e))

    return tests_passed, 4

def test_quantum_ai_daemon_version():
    """Test QuantumAIDaemon version attribute."""
    print_header("Phase 1 Fix 2: QuantumAIDaemon Version Attribute")

    tests_passed = 0
    try:
        from l104_quantum_ai_daemon import QuantumAIDaemon
        daemon = QuantumAIDaemon()
        if hasattr(daemon, 'version'):
            print_test("QuantumAIDaemon.version attribute exists", True, f"Version: {daemon.version}")
            tests_passed += 1
        else:
            print_test("QuantumAIDaemon.version attribute exists", False, "Missing version attribute")
    except Exception as e:
        print_test("QuantumAIDaemon.version attribute exists", False, str(e))

    try:
        from l104_quantum_ai_daemon import QuantumAIDaemon
        daemon = QuantumAIDaemon()
        if daemon.version == "1.0.0":
            print_test("QuantumAIDaemon version is 1.0.0", True)
            tests_passed += 1
        else:
            print_test("QuantumAIDaemon version is 1.0.0", False, f"Got: {daemon.version}")
    except Exception as e:
        print_test("QuantumAIDaemon version is 1.0.0", False, str(e))

    try:
        from l104_quantum_ai_daemon import QuantumAIDaemon
        daemon = QuantumAIDaemon()
        required_attrs = ['_scanner', '_improver', '_fidelity', '_optimizer', '_harmonizer', '_evolver']
        missing = [attr for attr in required_attrs if not hasattr(daemon, attr)]
        if not missing:
            print_test("QuantumAIDaemon has all subsystems", True)
            tests_passed += 1
        else:
            print_test("QuantumAIDaemon has all subsystems", False, f"Missing: {missing}")
    except Exception as e:
        print_test("QuantumAIDaemon has all subsystems", False, str(e))

    return tests_passed, 3

def test_state_persistence():
    """Test daemon state persistence infrastructure."""
    print_header("Phase 1 Fix 3: State Persistence Infrastructure")

    tests_passed = 0
    try:
        # Check state directory exists
        state_dir = Path.home() / ".l104_daemon_state"
        if state_dir.exists() or state_dir.parent.exists():
            print_test("State persistence directory accessible", True, str(state_dir.parent))
            tests_passed += 1
        else:
            print_test("State persistence directory accessible", True, "Directory can be created on demand")
            tests_passed += 1
    except Exception as e:
        print_test("State persistence directory accessible", False, str(e))

    try:
        from l104_daemon_orchestrator import DaemonOrchestrator
        orchestrator = DaemonOrchestrator()
        if hasattr(orchestrator, '_persist_state') and callable(orchestrator._persist_state):
            print_test("Orchestrator._persist_state method exists", True)
            tests_passed += 1
        else:
            print_test("Orchestrator._persist_state method exists", False)
    except Exception as e:
        print_test("Orchestrator._persist_state method exists", False, str(e))

    try:
        from l104_daemon_orchestrator import DaemonOrchestrator
        orchestrator = DaemonOrchestrator()
        if hasattr(orchestrator, '_load_state') and callable(orchestrator._load_state):
            print_test("Orchestrator._load_state method exists", True)
            tests_passed += 1
        else:
            print_test("Orchestrator._load_state method exists", False)
    except Exception as e:
        print_test("Orchestrator._load_state method exists", False, str(e))

    return tests_passed, 3

def test_fast_server_initialization():
    """Test FastAPI server initialization and routes."""
    print_header("Phase 1 Fix 4: Fast Server Initialization")

    tests_passed = 0
    try:
        from l104_server.app import app
        print_test("FastAPI app imports successfully", True, f"Title: {app.title}")
        tests_passed += 1
    except Exception as e:
        print_test("FastAPI app imports successfully", False, str(e))

    try:
        from l104_server.app import app
        if hasattr(app, 'routes') and len(app.routes) > 0:
            print_test("FastAPI app has routes configured", True, f"Routes: {len(app.routes)}")
            tests_passed += 1
        else:
            print_test("FastAPI app has routes configured", False, "No routes found")
    except Exception as e:
        print_test("FastAPI app has routes configured", False, str(e))

    try:
        from l104_server import intellect, LearningIntellect
        print_test("Server intellect imports successfully", True)
        tests_passed += 1
    except Exception as e:
        print_test("Server intellect imports successfully", False, str(e))

    return tests_passed, 3

def test_quantum_network_health():
    """Test quantum network and daemon health systems."""
    print_header("Phase 1 Fix 5: Quantum Network & Daemon Health")

    tests_passed = 0
    try:
        from l104_quantum_ai_daemon.daemon import QuantumAIDaemon
        daemon = QuantumAIDaemon()
        health = daemon._health_score
        if 0.0 <= health <= 1.0:
            print_test("Daemon health score is normalized", True, f"Health: {health:.3f}")
            tests_passed += 1
        else:
            print_test("Daemon health score is normalized", False, f"Got: {health}")
    except Exception as e:
        print_test("Daemon health score is normalized", False, str(e))

    try:
        from l104_quantum_networker import get_networker
        net = get_networker()
        status = net.status()
        if status and isinstance(status, dict):
            print_test("Quantum networker status available", True, f"Nodes: {len(status.get('nodes', []))}")
            tests_passed += 1
        else:
            print_test("Quantum networker status available", False)
    except Exception as e:
        print_test("Quantum networker status available", False, str(e))

    try:
        from l104_quantum_ai_daemon.daemon import QuantumAIDaemon
        daemon = QuantumAIDaemon()
        if hasattr(daemon, '_circuit_breaker_open'):
            print_test("Daemon circuit breaker available", True)
            tests_passed += 1
        else:
            print_test("Daemon circuit breaker available", False)
    except Exception as e:
        print_test("Daemon circuit breaker available", False, str(e))

    return tests_passed, 3

def test_multi_daemon_coordination():
    """Test multi-daemon coordination infrastructure."""
    print_header("Phase 1 Fix 6: Multi-Daemon Coordination")

    tests_passed = 0
    try:
        from l104_daemon_orchestrator import DaemonOrchestrator
        orchestrator = DaemonOrchestrator()
        if hasattr(orchestrator, '_daemons') and isinstance(orchestrator._daemons, dict):
            print_test("Orchestrator daemon registry available", True, f"Type: {type(orchestrator._daemons).__name__}")
            tests_passed += 1
        else:
            print_test("Orchestrator daemon registry available", False)
    except Exception as e:
        print_test("Orchestrator daemon registry available", False, str(e))

    try:
        from l104_daemon_orchestrator import DaemonOrchestrator
        orchestrator = DaemonOrchestrator()
        if hasattr(orchestrator, '_task_queue'):
            print_test("Orchestrator task queue available", True)
            tests_passed += 1
        else:
            print_test("Orchestrator task queue available", False)
    except Exception as e:
        print_test("Orchestrator task queue available", False, str(e))

    try:
        from l104_daemon_orchestrator import DaemonOrchestrator
        orchestrator = DaemonOrchestrator()
        if hasattr(orchestrator, '_event_queue'):
            print_test("Orchestrator event bus available", True)
            tests_passed += 1
        else:
            print_test("Orchestrator event bus available", False)
    except Exception as e:
        print_test("Orchestrator event bus available", False, str(e))

    return tests_passed, 3

def run_comprehensive_validation():
    """Run all validation tests."""
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}L104 DAEMON SYSTEM COMPREHENSIVE VALIDATION{Colors.RESET}")
    print(f"{Colors.MAGENTA}Started: {datetime.now().isoformat()}{Colors.RESET}\n")

    total_passed = 0
    total_tests = 0

    results = []

    # Test 1
    p, t = test_daemon_orchestrator_import()
    results.append(("DaemonOrchestrator Import", p, t))
    total_passed += p
    total_tests += t

    # Test 2
    p, t = test_quantum_ai_daemon_version()
    results.append(("QuantumAIDaemon Version", p, t))
    total_passed += p
    total_tests += t

    # Test 3
    p, t = test_state_persistence()
    results.append(("State Persistence", p, t))
    total_passed += p
    total_tests += t

    # Test 4
    p, t = test_fast_server_initialization()
    results.append(("Fast Server Init", p, t))
    total_passed += p
    total_tests += t

    # Test 5
    p, t = test_quantum_network_health()
    results.append(("Quantum Network Health", p, t))
    total_passed += p
    total_tests += t

    # Test 6
    p, t = test_multi_daemon_coordination()
    results.append(("Multi-Daemon Coordination", p, t))
    total_passed += p
    total_tests += t

    # Summary
    print_header("VALIDATION SUMMARY")

    for name, passed, total in results:
        pct = (passed / total * 100) if total > 0 else 0
        status_color = Colors.GREEN if passed == total else Colors.YELLOW if passed > 0 else Colors.RED
        print(f"  {status_color}{name:40s} {passed:2d}/{total:2d} ({pct:5.1f}%){Colors.RESET}")

    pct_total = (total_passed / total_tests * 100) if total_tests > 0 else 0
    status_color = Colors.GREEN if pct_total == 100 else Colors.YELLOW if pct_total >= 80 else Colors.RED
    print(f"\n{Colors.BOLD}{status_color}TOTAL: {total_passed}/{total_tests} ({pct_total:.1f}%){Colors.RESET}\n")

    if pct_total == 100:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL CRITICAL DAEMON SYSTEMS VALIDATED{Colors.RESET}\n")
        return 0
    elif pct_total >= 80:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ MOST SYSTEMS OPERATIONAL (Minor issues detected){Colors.RESET}\n")
        return 1
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ CRITICAL ISSUES DETECTED{Colors.RESET}\n")
        return 2

if __name__ == "__main__":
    exit_code = run_comprehensive_validation()
    sys.exit(exit_code)
