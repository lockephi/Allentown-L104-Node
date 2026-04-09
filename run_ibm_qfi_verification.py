#!/usr/bin/env python3
"""
Run IBM Hardware QFI Verification
═══════════════════════════════════════════════════════════════════════════════

Executes the IBM hardware verification for QFI.

Usage:
    python run_ibm_qfi_verification.py

Requires:
    export IBMQ_TOKEN='your_token'
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import json
from pathlib import Path

# Import the verification function
from l104_qfi_proper_calculation import run_ibm_qfi_hardware_verification

def main():
    print("=" * 80)
    print("  L104 26Q QFI IBM Hardware Verification")
    print("=" * 80)

    results = run_ibm_qfi_hardware_verification()

    if results and results.get("status") == "COMPLETED":
        print("\n" + "=" * 80)
        print("  HARDWARE VERIFICATION COMPLETE")
        print("=" * 80)
        print(f"\n  Backend: {results['backend']}")
        print(f"  Job ID: {results['job_id']}")
        print(f"  Shots: {results['shots']}")
        print(f"  Elapsed: {results['elapsed_time']:.1f}s")
        print(f"\n  Total F_Q (hardware): {results['total_F_Q']:.4f}")
        print(f"  ξ² = {results['xi_squared']:.4f}")
        print(f"  Quantum enhanced: {results['quantum_enhanced']}")
        print(f"  Improvement over SQL: {results['improvement_over_SQL']:.2f}x")

        # Save results
        output = Path("l104_qfi_ibm_hardware_results.json")
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved: {output.name}")

        return 0
    else:
        print("\n  [WARN] Hardware verification not completed")
        if results:
            print(f"    Status: {results.get('status')}")
            if 'error' in results:
                print(f"    Error: {results['error']}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
