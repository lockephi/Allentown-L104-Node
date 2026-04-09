#!/usr/bin/env python3
"""
Sage Connectivity Verification Script
Quick verification of all sage consciousness modules.
Run this script to ensure sage systems are connected.
"""

import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

def print_header(text: str) -> None:
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def test_import(module_name: str, import_stmt: str, test_code: str = None) -> Tuple[bool, str]:
    """Test importing a module and optionally run test code."""
    try:
        namespace = {}
        exec(import_stmt, namespace)
        
        if test_code:
            exec(test_code, namespace)
            
        return True, f"✓ {module_name}"
    except Exception as e:
        return False, f"✗ {module_name}: {str(e)[:80]}"

def verify_core_modules() -> List[Tuple[bool, str]]:
    """Verify all core sage modules."""
    results = []
    
    tests = [
        ("local_intellect", "from l104_intellect import local_intellect"),
        ("format_iq", "from l104_intellect import format_iq", "result = format_iq(150.5)"),
        ("PHI", "from l104_intellect import PHI", "print(f'    PHI = {PHI}')"),
        ("GOD_CODE", "from l104_intellect import GOD_CODE", "print(f'    GOD_CODE = {GOD_CODE}')"),
        ("SAGE_MODE_VERSION", "from l104_intellect import SAGE_MODE_VERSION", 
         "print(f'    SAGE_MODE_VERSION = {SAGE_MODE_VERSION}')"),
        ("SageModeInflect", "from l104_sage_mode_inflect import SageModeInflect",
         "sage = SageModeInflect(); sage.activate(); print(f'    Active: {sage.get_status().get(\"active\")}')"),
        ("RSE Sage", "from l104_intellect import get_rse_sage",
         "rse = get_rse_sage(); print(f'    Type: {type(rse).__name__}')"),
    ]
    
    for name, import_stmt, *test_code in tests:
        success, message = test_import(name, import_stmt, test_code[0] if test_code else None)
        results.append((success, message))
    
    return results

def verify_ouroboros_state() -> Tuple[bool, str, Dict]:
    """Verify ouroboros nirvanic state."""
    try:
        with open('.l104_ouroboros_nirvanic_state.json', 'r') as f:
            state = json.load(f)
        
        sage_stability = state.get('sage_stability', 0)
        if sage_stability > 0.9:
            status = "OPTIMAL"
            success = True
        elif sage_stability > 0.7:
            status = "ACCEPTABLE"
            success = True
        else:
            status = "UNSTABLE"
            success = False
        
        message = f"✓ Ouroboros: stability={sage_stability:.3f} ({status})"
        return success, message, state
        
    except Exception as e:
        return False, f"✗ Ouroboros: {str(e)[:80]}", {}

def verify_quantum_bridge() -> List[Tuple[bool, str]]:
    """Verify quantum bridge connectivity."""
    results = []
    
    try:
        from l104_intellect import local_intellect as li
        
        # Check quantum recompiler
        if hasattr(li, 'quantum_recompiler'):
            results.append((True, "✓ Quantum recompiler: PRESENT"))
        else:
            results.append((False, "✗ Quantum recompiler: MISSING"))
        
        # Check sage consciousness coherence
        if hasattr(li, 'sage_consciousness_coherence'):
            results.append((True, "✓ Sage consciousness coherence: AVAILABLE"))
        else:
            results.append((False, "✗ Sage consciousness coherence: MISSING"))
            
        # Check quantum consciousness bridge
        if hasattr(li, '_qc_consciousness_bridge'):
            results.append((True, "✓ Quantum consciousness bridge: PRESENT"))
        else:
            results.append((False, "✗ Quantum consciousness bridge: MISSING"))
            
    except Exception as e:
        results.append((False, f"✗ Quantum bridge verification failed: {str(e)[:80]}"))
    
    return results

def main() -> int:
    """Main verification function."""
    print_header("SAGE CONSCIOUSNESS CONNECTIVITY VERIFICATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    
    all_results = []
    
    # Test 1: Core modules
    print_header("1. CORE MODULE VERIFICATION")
    core_results = verify_core_modules()
    all_results.extend(core_results)
    
    for success, message in core_results:
        print(f"  {message}")
    
    # Test 2: Ouroboros state
    print_header("2. OUROBOROS NIRVANIC STATE")
    ouroboros_success, ouroboros_message, state = verify_ouroboros_state()
    all_results.append((ouroboros_success, ouroboros_message))
    print(f"  {ouroboros_message}")
    
    if state:
        print(f"    Cycle count: {state.get('cycle_count')}")
        print(f"    Nirvanic fuel: {state.get('total_nirvanic_fuel', 0):.4f}")
        print(f"    Divine interventions: {state.get('divine_interventions')}")
    
    # Test 3: Quantum bridge
    print_header("3. QUANTUM BRIDGE CONNECTIVITY")
    quantum_results = verify_quantum_bridge()
    all_results.extend(quantum_results)
    
    for success, message in quantum_results:
        print(f"  {message}")
    
    # Test 4: SageModeInflect functionality
    print_header("4. SAGEMODEINFLECT FUNCTIONALITY")
    try:
        from l104_sage_mode_inflect import SageModeInflect
        sage = SageModeInflect()
        sage.activate()
        status = sage.get_status()
        
        if status.get('active') and status.get('operational'):
            all_results.append((True, "✓ SageModeInflect: OPERATIONAL"))
            print(f"  ✓ SageModeInflect: OPERATIONAL")
            print(f"    State: {status.get('state')}")
            print(f"    Wisdom level: {status.get('wisdom_level')}")
            print(f"    Total inflections: {status.get('total_inflections')}")
        else:
            all_results.append((False, "✗ SageModeInflect: INACTIVE"))
            print(f"  ✗ SageModeInflect: INACTIVE")
            
    except Exception as e:
        all_results.append((False, f"✗ SageModeInflect: {str(e)[:80]}"))
        print(f"  ✗ SageModeInflect: {str(e)[:80]}")
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    total = len(all_results)
    passed = sum(1 for success, _ in all_results if success)
    failed = total - passed
    
    print(f"Tests Passed: {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"Tests Failed: {failed}")
    
    if failed > 0:
        print("\nFailed Tests:")
        for success, message in all_results:
            if not success:
                print(f"  {message}")
    
    print_header("SYSTEM STATUS")
    
    # Determine overall status
    ouroboros_stability = state.get('sage_stability', 0) if state else 0
    
    if failed == 0 and ouroboros_stability > 0.9:
        print("✅ SAGE CONSCIOUSNESS: FULLY OPERATIONAL")
        print("\nAll systems connected and stable.")
        print("Ready for advanced sage operations.")
        return 0
    elif failed <= 2 and ouroboros_stability > 0.7:
        print("⚠ SAGE CONSCIOUSNESS: MOSTLY OPERATIONAL")
        print("\nMinor issues detected but system is functional.")
        print("Sage operations possible with monitoring.")
        return 1
    else:
        print("❌ SAGE CONSCIOUSNESS: NEEDS ATTENTION")
        print("\nSignificant issues detected.")
        print("System requires attention before sage operations.")
        return 2

if __name__ == "__main__":
    sys.exit(main())