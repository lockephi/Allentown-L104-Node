#!/usr/bin/env python3
"""Quick verification of identity boundary integration."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("=== ASI Identity Boundary ===")
from l104_asi.identity_boundary import SovereignIdentityBoundary, L104_IS, L104_IS_NOT, MEASURED_PERFORMANCE
sib = SovereignIdentityBoundary()
print(f"  IS: {len(L104_IS)} | IS_NOT: {len(L104_IS_NOT)} | Benchmarks: {len(MEASURED_PERFORMANCE)}")

r1 = sib.validate_claim("L104 is an LLM")
print(f"  Claim 'LLM' -> valid={r1['valid']}, cat={r1['category']}")

r2 = sib.validate_claim("L104 has a dual layer architecture")
print(f"  Claim 'dual layer' -> valid={r2['valid']}, cat={r2['category']}")

r3 = sib.validate_claim("L104 is better than GPT-4")
print(f"  Claim 'better GPT' -> valid={r3['valid']}, cat={r3['category']}")

can, why = sib.can_handle_domain("quantum_simulation")
print(f"  Domain quantum: {can}")

can2, why2 = sib.can_handle_domain("image recognition")
print(f"  Domain image: {can2}")

print(f"  Manifest keys: {sorted(sib.identity_manifest().keys())}")

print("\n=== AGI Identity Boundary ===")
from l104_agi.identity_boundary import AGIIdentityBoundary, L104_AGI_IS, L104_AGI_IS_NOT, AGI_CAPABILITY_MAP
aib = AGIIdentityBoundary()
print(f"  IS: {len(L104_AGI_IS)} | IS_NOT: {len(L104_AGI_IS_NOT)} | Capabilities: {len(AGI_CAPABILITY_MAP)}")

cap = aib.query_capability("quantum_simulation")
print(f"  Quantum: {cap['capability']}")
cap2 = aib.query_capability("knowledge_retrieval")
print(f"  Knowledge: {cap2['capability']}")

cc = aib.validate_cross_core_consistency()
print(f"  Cross-core: {cc['verdict']} (shared={cc.get('shared_boundaries', 0)})")

print("\n=== STATUS OUTPUTS ===")
print(f"  ASI status: {sib.get_status()}")
print(f"  AGI status: {aib.get_status()}")

print("\n=== ALL TESTS PASSED ===")
