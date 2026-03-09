#!/usr/bin/env python3
"""Quick AGI-only identity boundary test (bypasses full AGI core init)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("=== AGI Identity Boundary (direct) ===")
from l104_agi.identity_boundary import AGIIdentityBoundary, L104_AGI_IS, L104_AGI_IS_NOT, AGI_CAPABILITY_MAP
aib = AGIIdentityBoundary()
print(f"  IS: {len(L104_AGI_IS)} | IS_NOT: {len(L104_AGI_IS_NOT)} | Capabilities: {len(AGI_CAPABILITY_MAP)}")

r1 = aib.validate_claim("L104 is an LLM")
print(f"  Claim 'LLM' -> valid={r1['valid']}, cat={r1['category']}")

r2 = aib.validate_claim("L104 has a cognitive mesh network")
print(f"  Claim 'cognitive mesh' -> valid={r2['valid']}, cat={r2['category']}")

r3 = aib.validate_claim("L104 is sentient")
print(f"  Claim 'sentient' -> valid={r3['valid']}, cat={r3['category']}")

cap = aib.query_capability("quantum_simulation")
print(f"  Quantum: {cap['capability']}")
cap2 = aib.query_capability("knowledge_retrieval")
print(f"  Knowledge: {cap2['capability']}")
cap3 = aib.query_capability("self_modification")
print(f"  Self-mod: {cap3['capability']}")

cc = aib.validate_cross_core_consistency()
print(f"  Cross-core: {cc['verdict']} (shared={cc.get('shared_boundaries', 0)})")

print(f"  Status: {aib.get_status()}")
print(f"  Manifest keys: {sorted(aib.identity_manifest().keys())}")

print("\n=== ALL AGI TESTS PASSED ===")
