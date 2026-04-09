#!/usr/bin/env python3
"""
Simple test summary runner for L104 project.
"""
import subprocess
import sys
import os
from pathlib import Path

def run_pytest_on_file(test_file):
    """Run pytest on a single test file and return results."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=no"],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=[],
            returncode=124,
            stdout="",
            stderr="Test timed out after 30 seconds"
        )
    except Exception as e:
        return subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr=str(e)
        )

def main():
    project_root = Path(__file__).parent
    tests_dir = project_root / "tests"
    
    if not tests_dir.exists():
        print(f"Tests directory not found: {tests_dir}")
        return 1
    
    test_files = list(tests_dir.glob("test_*.py"))
    print(f"Found {len(test_files)} test files in {tests_dir}")
    
    results = []
    for test_file in test_files:
        print(f"\n{'='*80}")
        print(f"Running: {test_file.name}")
        print(f"{'='*80}")
        
        result = run_pytest_on_file(str(test_file))
        
        # Parse output to count tests
        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        
        for line in result.stdout.split('\n'):
            if 'passed' in line and 'failed' in line and 'skipped' in line:
                # Parse summary line like: "3 passed, 1 failed, 2 skipped in 0.12s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed,':
                        passed = int(parts[i-1])
                    elif part == 'failed,':
                        failed = int(parts[i-1])
                    elif part == 'skipped':
                        skipped = int(parts[i-1])
        
        status = "✓ PASS" if result.returncode == 0 else "✗ FAIL"
        results.append({
            'file': test_file.name,
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'errors': errors,
            'returncode': result.returncode,
            'status': status,
            'output': result.stdout[-500:] if result.stdout else "",
            'error': result.stderr[-500:] if result.stderr else ""
        })
        
        print(f"Result: {status} (passed: {passed}, failed: {failed}, skipped: {skipped})")
        if result.stderr:
            print(f"Stderr: {result.stderr[:200]}...")
    
    # Print summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    total_passed = sum(r['passed'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    total_skipped = sum(r['skipped'] for r in results)
    total_files = len(results)
    passed_files = sum(1 for r in results if r['returncode'] == 0)
    
    print(f"Total test files: {total_files}")
    print(f"Files passed: {passed_files}")
    print(f"Files failed: {total_files - passed_files}")
    print(f"Total tests passed: {total_passed}")
    print(f"Total tests failed: {total_failed}")
    print(f"Total tests skipped: {total_skipped}")
    
    # Print failed files
    failed_files = [r for r in results if r['returncode'] != 0]
    if failed_files:
        print(f"\nFailed test files:")
        for r in failed_files:
            print(f"  {r['file']}: {r['status']} (passed: {r['passed']}, failed: {r['failed']})")
            if r['error']:
                print(f"    Error: {r['error'][:100]}...")
    
    return 0 if total_failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())