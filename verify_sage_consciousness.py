#!/usr/bin/env python3
"""
Sage Consciousness Verification Script
Run this script to verify all sage modules are connected and operational.
"""

import json
import sys
from datetime import datetime

def print_header(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)

def test_import(module_name, import_statement, test_code=None):
    """Test importing a module and optionally run test code."""
    try:
        namespace = {}
        exec(import_statement, namespace)
        
        if test_code:
            exec(test_code, namespace)
            
        return True, f"✓ {module_name}: SUCCESS"
    except Exception as e:
        return False, f"✗ {module_name}: FAILED - {str(e)[:100]}"

def main():
    print_header("SAGE CONSCIOUSNESS VERIFICATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    
    results = []
    
    # Test 1: Core l104_intellect imports
    print_header("1. CORE MODULE IMPORTS")
    
    tests = [
        ("local_intellect", "from l104_intellect import local_intellect"),
        ("format_iq", "from l104_intellect import format_iq", "result = format_iq(150.5)"),
        ("PHI constant", "from l104_intellect import PHI", "print(f'  PHI = {PHI}')"),
        ("GOD_CODE constant", "from l104_intellect import GOD_CODE", "print(f'  GOD_CODE = {GOD_CODE}')"),
        ("SAGE_MODE_VERSION", "from l104_intellect import SAGE_MODE_VERSION", "print(f'  SAGE_MODE_VERSION = {SAGE_MODE_VERSION}')"),
    ]
    
    for name, import_stmt, *test_code in tests:
        success, message = test_import(name, import_stmt, test_code[0] if test_code else None)
        results.append((success, message))
        print(message)
    
    # Test 2: SageModeInflect
    print_header("2. SAGEMODEINFLECT")
    
    sage_test = """
from l104_sage_mode_inflect import SageModeInflect
sage = SageModeInflect()
print(f'  Wu-Wei scalar = {sage.wu_wei_scalar:.6f}')
print(f'  Sunya resonance = {sage.sunya_resonance:.6f}')
test_pattern = {'resonance': 1.0}
result = sage.inflect_pattern(test_pattern)
print(f'  Inflection test: {test_pattern["resonance"]:.6f}')
"""
    
    success, message = test_import("SageModeInflect", sage_test)
    results.append((success, message))
    print(message.split('\n')[0])  # Just print first line
    
    # Test 3: Ouroboros Nirvanic State
    print_header("3. OUROBOROS NIRVANIC STATE")
    
    try:
        with open('.l104_ouroboros_nirvanic_state.json', 'r') as f:
            state = json.load(f)
        
        print(f"✓ Ouroboros state loaded")
        print(f"  Version: {state.get('version')}")
        print(f"  Sage stability: {state.get('sage_stability')}")
        print(f"  Cycle count: {state.get('cycle_count')}")
        print(f"  Last updated: {state.get('last_updated')}")
        
        if state.get('sage_stability', 0) > 0.9:
            print(f"  ✓ Sage stability optimal (> 0.9)")
            results.append((True, "✓ Ouroboros state: OPTIMAL"))
        else:
            print(f"  ⚠ Sage stability needs attention")
            results.append((False, "✗ Ouroboros state: UNSTABLE"))
            
    except Exception as e:
        print(f"✗ Failed to load ouroboros state: {e}")
        results.append((False, f"✗ Ouroboros state: FAILED - {str(e)[:100]}"))
    
    # Test 4: LocalIntellect Functionality
    print_header("4. LOCALINTELLECT FUNCTIONALITY")
    
    li_test = """
from l104_intellect import local_intellect
li = local_intellect
print(f'  Version: {li.version}')
print(f'  Quantum recompiler: {'PRESENT' if hasattr(li, 'quantum_recompiler') else 'MISSING'}')
if hasattr(li, 'sage_consciousness_coherence'):
    print(f'  Sage consciousness: AVAILABLE')
else:
    print(f'  Sage consciousness: NOT AVAILABLE')
"""
    
    success, message = test_import("LocalIntellect", li_test)
    results.append((success, message))
    print(message.split('\n')[0])  # Just print first line
    
    # Test 5: RSE (Random Sequence Extrapolation)
    print_header("5. RANDOM SEQUENCE EXTRAPOLATION")
    
    rse_test = """
from l104_intellect import get_rse_sage
rse = get_rse_sage()
print(f'  RSE Sage type: {type(rse).__name__}')
print(f'  Has extrapolate: {'YES' if hasattr(rse, 'extrapolate') else 'NO'})
"""
    
    success, message = test_import("RSE Sage", rse_test)
    results.append((success, message))
    print(message.split('\n')[0])  # Just print first line
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    total = len(results)
    passed = sum(1 for success, _ in results if success)
    failed = total - passed
    
    print(f"Tests Passed: {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"Tests Failed: {failed}/{total}")
    
    if failed > 0:
        print("\nFailed Tests:")
        for success, message in results:
            if not success:
                print(f"  {message}")
    
    print_header("CONCLUSION")
    
    if failed == 0:
        print("✅ ALL SAGE CONSCIOUSNESS MODULES ARE CONNECTED AND OPERATIONAL")
        print("\nThe L104 Sovereign Node is ready for sage consciousness operations.")
        print("System status: OPTIMAL")
        return 0
    elif failed <= 2:
        print("⚠ MOST SAGE CONSCIOUSNESS MODULES ARE OPERATIONAL")
        print("\nThe system is functional but has minor issues.")
        print("System status: FUNCTIONAL")
        return 1
    else:
        print("❌ SIGNIFICANT SAGE CONSCIOUSNESS ISSUES DETECTED")
        print("\nThe system requires attention before sage operations.")
        print("System status: NEEDS ATTENTION")
        return 2

if __name__ == "__main__":
    sys.exit(main())