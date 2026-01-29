#!/usr/bin/env python3
# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  L104 SOVEREIGN NODE - UNIFIED TEST RUNNER                                    ║
# ║  INVARIANT: 527.5184818492612 | PILOT: LONDEL                                 ║
# ║  THE ARMY MARCHES WITH VALIDATED TRUTH                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

"""
Unified Test Runner for L104 Sovereign Node.

This script discovers and runs all test suites, generating a comprehensive
validation report that proves the mathematical, physical, and engineering
integrity of the L104 system.

Usage:
    python tests/run_all_tests.py          # Run all tests
    python tests/run_all_tests.py -v       # Verbose output
    python tests/run_all_tests.py --json   # Output JSON report
"""

import os
import sys
import math
import json
import unittest
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTS - THE INVARIANTS
# ════════════════════════════════════════════════════════════════════════════════

GOD_CODE = 527.5184818492612
PHI = (1 + math.sqrt(5)) / 2
TAU = 1 / PHI
REAL_GROUNDING = GOD_CODE / (2 ** 1.25)
FRAME_LOCK = 416 / 286

# ════════════════════════════════════════════════════════════════════════════════
# SOVEREIGN RESONANCE VERIFICATION
# ════════════════════════════════════════════════════════════════════════════════

def verify_sovereign_resonance() -> Dict[str, Any]:
    """
    Pre-flight check: Verify the core mathematical invariants
    before running any tests.
    """
    results = {
        "verified": True,
        "invariants": {},
        "timestamp": datetime.now().isoformat()
    }

    # Verify GOD_CODE formula
    computed = (286 ** (1/PHI)) * 16
    god_code_ok = abs(computed - GOD_CODE) < 1e-10
    results["invariants"]["GOD_CODE"] = {
        "expected": GOD_CODE,
        "computed": computed,
        "verified": god_code_ok,
        "formula": "286^(1/φ) × 16"
    }
    results["verified"] &= god_code_ok

    # Verify PHI properties
    phi_identity = PHI ** 2 - PHI - 1
    phi_ok = abs(phi_identity) < 1e-14
    results["invariants"]["PHI_IDENTITY"] = {
        "expected": 0.0,
        "computed": phi_identity,
        "verified": phi_ok,
        "formula": "φ² - φ - 1 = 0"
    }
    results["verified"] &= phi_ok

    # Verify TAU fusion rule
    tau_fusion = TAU ** 2 + TAU - 1
    tau_ok = abs(tau_fusion) < 1e-14
    results["invariants"]["TAU_FUSION"] = {
        "expected": 0.0,
        "computed": tau_fusion,
        "verified": tau_ok,
        "formula": "τ² + τ = 1"
    }
    results["verified"] &= tau_ok

    # Verify exponent reduction
    exponent = 416 / 104
    exponent_ok = abs(exponent - 4.0) < 1e-15
    results["invariants"]["EXPONENT_REDUCTION"] = {
        "expected": 4.0,
        "computed": exponent,
        "verified": exponent_ok,
        "formula": "(2^(1/104))^416 = 2^4 = 16"
    }
    results["verified"] &= exponent_ok

    return results


# ════════════════════════════════════════════════════════════════════════════════
# TEST DISCOVERY AND EXECUTION
# ════════════════════════════════════════════════════════════════════════════════

class L104TestResult(unittest.TestResult):
    """
    Custom test result class that tracks detailed information
    about each test run.
    """

    def __init__(self):
        super().__init__()
        self.test_details: List[Dict[str, Any]] = []
        self.start_time = None
        self.current_test_start = None

    def startTest(self, test):
        super().startTest(test)
        self.current_test_start = time.time()

    def stopTest(self, test):
        elapsed = time.time() - self.current_test_start
        detail = {
            "name": str(test),
            "module": test.__class__.__module__,
            "class": test.__class__.__name__,
            "method": test._testMethodName,
            "elapsed_ms": round(elapsed * 1000, 2),
            "status": "passed"
        }

        # Check if this test failed or errored
        for failed_test, _ in self.failures:
            if failed_test == test:
                detail["status"] = "failed"
                break

        for errored_test, _ in self.errors:
            if errored_test == test:
                detail["status"] = "error"
                break

        for skipped_test, _ in self.skipped:
            if skipped_test == test:
                detail["status"] = "skipped"
                break

        self.test_details.append(detail)
        super().stopTest(test)


def discover_tests(test_dir: Path) -> unittest.TestSuite:
    """
    Discover all test modules in the test directory.
    """
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(test_dir),
        pattern='test_*.py',
        top_level_dir=str(PROJECT_ROOT)
    )
    return suite


def run_tests(verbosity: int = 2) -> L104TestResult:
    """
    Run all discovered tests and return the result.
    """
    test_dir = Path(__file__).parent
    suite = discover_tests(test_dir)

    result = L104TestResult()
    result.start_time = time.time()

    # Run with or without text output
    if verbosity > 0:
        runner = unittest.TextTestRunner(
            verbosity=verbosity,
            resultclass=lambda: result
        )
        # We need to re-run with our custom result
        suite.run(result)
    else:
        suite.run(result)

    result.elapsed_total = time.time() - result.start_time
    return result


# ════════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ════════════════════════════════════════════════════════════════════════════════

def generate_report(result: L104TestResult, resonance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a comprehensive JSON report of the test run.
    """
    passed = sum(1 for d in result.test_details if d["status"] == "passed")
    failed = sum(1 for d in result.test_details if d["status"] == "failed")
    errors = sum(1 for d in result.test_details if d["status"] == "error")
    skipped = sum(1 for d in result.test_details if d["status"] == "skipped")
    total = len(result.test_details)

    # Calculate pass rate with resonance adjustment
    pass_rate = passed / total if total > 0 else 0.0
    resonance_factor = pass_rate * (1 / PHI) if resonance["verified"] else 0.0

    report = {
        "L104_SOVEREIGN_VALIDATION_REPORT": {
            "timestamp": datetime.now().isoformat(),
            "god_code": GOD_CODE,
            "pilot": "LONDEL",
            "invariant_status": "LOCKED" if resonance["verified"] else "BREACH"
        },
        "resonance_verification": resonance,
        "test_summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "pass_rate": round(pass_rate * 100, 2),
            "elapsed_seconds": round(result.elapsed_total, 3)
        },
        "test_details": result.test_details,
        "failure_details": [
            {
                "test": str(test),
                "traceback": traceback
            }
            for test, traceback in result.failures
        ],
        "error_details": [
            {
                "test": str(test),
                "traceback": traceback
            }
            for test, traceback in result.errors
        ],
        "validation_status": {
            "mathematical_foundation": "VERIFIED" if resonance["verified"] else "INVALID",
            "all_tests_passed": passed == total,
            "sovereign_lock": pass_rate >= 0.95 and resonance["verified"]
        }
    }

    return report


def print_banner():
    """Print the L104 test runner banner."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     ██╗      ██╗ ██████╗ ██╗  ██╗    ████████╗███████╗███████╗████████╗       ║
║     ██║     ███║██╔═══██╗██║  ██║    ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝       ║
║     ██║     ╚██║██║   ██║███████║       ██║   █████╗  ███████╗   ██║          ║
║     ██║      ██║██║   ██║╚════██║       ██║   ██╔══╝  ╚════██║   ██║          ║
║     ███████╗ ██║╚██████╔╝     ██║       ██║   ███████╗███████║   ██║          ║
║     ╚══════╝ ╚═╝ ╚═════╝      ╚═╝       ╚═╝   ╚══════╝╚══════╝   ╚═╝          ║
║                                                                               ║
║                    SOVEREIGN VALIDATION FRAMEWORK                             ║
║                    GOD_CODE = 527.5184818492612                               ║
║                    PILOT: LONDEL                                              ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")


def print_resonance_status(resonance: Dict[str, Any]):
    """Print the resonance verification status."""
    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                     SOVEREIGN RESONANCE VERIFICATION                        │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")

    for name, data in resonance["invariants"].items():
        status = "✓" if data["verified"] else "✗"
        print(f"│  {status} {name:<20} {data['formula']:<40} │")

    overall = "LOCKED ✓" if resonance["verified"] else "BREACH ✗"
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print(f"│                     OVERALL STATUS: {overall:<15}                       │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")


def print_summary(report: Dict[str, Any]):
    """Print the test summary."""
    summary = report["test_summary"]
    validation = report["validation_status"]

    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                           TEST RESULTS SUMMARY                              │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print(f"│  Total Tests:     {summary['total']:<10}                                          │")
    print(f"│  Passed:          {summary['passed']:<10} ✓                                        │")
    print(f"│  Failed:          {summary['failed']:<10} {'✗' if summary['failed'] > 0 else ' '}                                        │")
    print(f"│  Errors:          {summary['errors']:<10} {'✗' if summary['errors'] > 0 else ' '}                                        │")
    print(f"│  Skipped:         {summary['skipped']:<10}                                          │")
    print(f"│  Pass Rate:       {summary['pass_rate']}%                                              │")
    print(f"│  Elapsed:         {summary['elapsed_seconds']}s                                            │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")

    lock_status = "SOVEREIGN LOCK ENGAGED ✓" if validation["sovereign_lock"] else "SOVEREIGN LOCK BREACH ✗"
    print(f"│                      {lock_status:<30}                     │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")

    if summary["failed"] > 0 or summary["errors"] > 0:
        print("\n⚠ FAILURES AND ERRORS:")
        for failure in report["failure_details"]:
            print(f"\n  FAILED: {failure['test']}")
            for line in failure['traceback'].split('\n')[:5]:
                print(f"    {line}")
        for error in report["error_details"]:
            print(f"\n  ERROR: {error['test']}")
            for line in error['traceback'].split('\n')[:5]:
                print(f"    {line}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="L104 Sovereign Node Test Runner")
    parser.add_argument("-v", "--verbose", action="count", default=1,
                        help="Increase verbosity (can be used multiple times)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Minimal output")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON report to stdout")
    parser.add_argument("--report", type=str,
                        help="Path to save JSON report")

    args = parser.parse_args()

    verbosity = 0 if args.quiet else args.verbose

    if not args.json:
        print_banner()

    # Pre-flight: Verify resonance
    resonance = verify_sovereign_resonance()

    if not args.json:
        print_resonance_status(resonance)
        print("\n" + "═" * 79)
        print("                           RUNNING TEST SUITES")
        print("═" * 79 + "\n")

    # Run all tests
    result = run_tests(verbosity=verbosity)

    # Generate report
    report = generate_report(result, resonance)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_summary(report)

    if args.report:
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Report saved to: {args.report}")

    # Exit code based on test success
    if report["test_summary"]["failed"] > 0 or report["test_summary"]["errors"] > 0:
        sys.exit(1)
    elif not resonance["verified"]:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
