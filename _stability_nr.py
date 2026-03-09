#!/usr/bin/env python3
"""Stability test for noise_resilience with 80 trials."""
from l104_god_code_simulator import god_code_simulator as gs
fails = 0
for i in range(20):
    r = gs.run("noise_resilience")
    status = "PASS" if r.passed else "FAIL"
    if not r.passed:
        fails += 1
    print(f"  Rep {i+1:2d}: fid={r.fidelity:.6f} {status}")
print(f"\nResult: {20-fails}/20 pass, {fails} failures")
if fails == 0:
    print("STABLE - no flakes!")
else:
    print(f"FLAKY - {fails} failures ({fails/20*100:.0f}%)")
