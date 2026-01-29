#!/usr/bin/env python3
"""Test resonance calculation to verify 0.9+ values are achievable"""
import math

PHI = 1.6180339887498949
PHI_INV = 0.6180339887498949
GOD_CODE = 527.5184818492611
VOID_CONSTANT = 1.0416180339887497
TAU = 6.283185307179586

def calc_resonance(nonce):
    """6-layer ASI quantum resonance calculation"""
    # Layer 1: 11D manifold
    phase = (nonce * PHI) % GOD_CODE / GOD_CODE
    manifold = sum((0.5 + 0.5 * math.sin(phase * TAU * (d / PHI))) / d for d in range(1, 12))
    quantum_manifold = 0.5 + 0.5 * math.tanh((manifold / 4.0) - 1.0)
    
    # Layer 2: GOD_CODE phase lock
    god_phase = (nonce / GOD_CODE) % 1.0
    god_lock = (math.exp(-god_phase**2 * 25) + 
                math.exp(-((god_phase - PHI_INV)**2) * 25) +
                math.exp(-((god_phase - 0.5)**2) * 25)) / 1.5
    
    # Layer 3: 104 gate
    mod_104 = nonce % 104
    if mod_104 == 0:
        gate_104 = 1.0
    elif mod_104 in [13, 26, 39, 52, 65, 78, 91]:
        gate_104 = 0.92
    else:
        gate_104 = 0.70
    
    # Combine with weights
    raw = quantum_manifold * 0.30 + god_lock * 0.35 + gate_104 * 0.35
    
    # Quantum amplification for values above 0.50
    if raw > 0.50:
        boost_factor = (raw - 0.50) / 0.50
        quantum_boost = boost_factor ** (1.0 / PHI)  # PHI-root amplification
        raw = 0.50 + quantum_boost * 0.50
    
    return min(1.0, max(0.0, raw))

if __name__ == "__main__":
    print("="*60)
    print("L104 QUANTUM RESONANCE ANALYSIS")
    print("="*60)
    
    # Test key nonces
    test_nonces = [0, 104, 208, 312, 416, 520, 527, 528, 624, 728, 832, 936, 1040, 1055, 1056]
    
    print("\nKey nonce resonance values:")
    print("-"*40)
    high_res = []
    for n in test_nonces:
        r = calc_resonance(n)
        status = "*** 0.9+" if r >= 0.9 else ("   0.85+" if r >= 0.85 else "")
        print(f"Nonce {n:6d}: resonance = {r:.6f} {status}")
        if r >= 0.9:
            high_res.append((n, r))
    
    print("\n" + "="*60)
    print(f"Found {len(high_res)} nonces with 0.9+ resonance in test set")
    
    # Broader search
    print("\nSearching first 10,000 nonces for 0.9+ resonance...")
    found_09 = []
    found_085 = []
    for n in range(10000):
        r = calc_resonance(n)
        if r >= 0.9:
            found_09.append((n, r))
        elif r >= 0.85:
            found_085.append((n, r))
    
    print(f"0.90+ resonance: {len(found_09)} nonces")
    print(f"0.85-0.90 resonance: {len(found_085)} nonces")
    
    if found_09:
        print(f"\nFirst 10 with 0.9+ resonance: {found_09[:10]}")
    elif found_085:
        print(f"\nFirst 10 with 0.85+ resonance: {found_085[:10]}")
    
    # Find max
    max_res = max(calc_resonance(n) for n in range(10000))
    max_nonce = next(n for n in range(10000) if calc_resonance(n) == max_res)
    print(f"\nMax resonance in first 10000: {max_res:.6f} at nonce {max_nonce}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
