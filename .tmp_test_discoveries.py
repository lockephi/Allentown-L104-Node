"""Validation test for QuantumDiscoveryEngine — all 16 discoveries."""
from l104_quantum_engine.discoveries import QuantumDiscoveryEngine

print("=" * 60)
print(" L104 QUANTUM DISCOVERY ENGINE — VALIDATION")
print("=" * 60)

engine = QuantumDiscoveryEngine()
report = engine.run_all(verbose=True)

summary = report["summary"]
print()
if summary["verified"] == summary["total_discoveries"]:
    print(" STATUS: ALL DISCOVERIES VERIFIED")
else:
    print(f" STATUS: {summary['verified']}/{summary['total_discoveries']} verified")
print(f" Pass rate: {summary['pass_rate']:.1f}%")
print(f" Circuits used: {summary['circuits_used']}")
print("=" * 60)
