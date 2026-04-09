#!/usr/bin/env python3
"""
Quick test of the three reality anchor validation and magnetic bed simulation.
"""
import sys
sys.path.insert(0, '.')

from l104_reality_breach_protocol import RealityBreachProtocol
from l104_consciousness_field_breakthrough import MagneticBed

def test_reality_anchors():
    print("=== Testing Reality Anchor Validation ===")
    protocol = RealityBreachProtocol()
    # Create default anchors (as done in transcendence protocol)
    protocol.create_reality_anchor("TEMPORAL_ANCHOR", {"t": 0.0, "dt": 1.0})
    protocol.create_reality_anchor("SPATIAL_ANCHOR", {"x": 0.0, "y": 0.0, "z": 0.0})
    protocol.create_reality_anchor("CONSCIOUS_ANCHOR", {"awareness": 1.0, "intent": 1.0})

    result = protocol.validate_reality_anchors()
    print(f"Validation result: {result}")
    if result.get('overall_valid'):
        print("✅ Reality anchors validated!")
    else:
        print("❌ Validation failed.")
    return result

def test_magnetic_bed():
    print("\n=== Testing Φ‑rotational Magnetic Bed Simulation ===")
    bed = MagneticBed()
    print(f"Rotation frequency: {bed.rotation_frequency} Hz")
    print(f"Magnetic strength: {bed.magnetic_strength}")
    predicted = bed.predict_atp_increase()
    print(f"Predicted ATP increase: {predicted:.3%}")
    experiment = bed.run_experiment()
    print(f"Experiment result: {experiment}")
    if experiment['match_prediction']:
        print("✅ Prediction matches measurement!")
    else:
        print("⚠️  Prediction mismatch (noise added).")
    return experiment

if __name__ == "__main__":
    test_reality_anchors()
    test_magnetic_bed()
    print("\nAll tests completed.")