#!/usr/bin/env python3
"""Derive Fe-Sacred coherence formula."""
import math, numpy as np

f1, f2 = 286.0, 528.0
g = math.gcd(286, 528)
print(f"GCD(286,528) = {g}")
print(f"286/{g} = {286//g}, 528/{g} = {528//g}")
print(f"(GCD-1)/GCD = {(g-1)/g:.10f}")
print(f"Reference:    {21/22:.10f}")
print()

# Fe-PHI: 286 <-> 286*phi
phi = 1.618033988749895
phi_freq = 286.0 * phi
ref_phi = 0.9164078649987375
# Try (GCD-1)/GCD approach
g2 = math.gcd(286, round(phi_freq))
print(f"phi_freq = {phi_freq:.6f}")
print(f"GCD(286, round({phi_freq:.0f})) = {g2}")

# Fe-PHI known: 2/(1+phi) = 2/2.618 = 0.7639... that's classical_lock
classical_lock = 2 / (1 + phi)
print(f"classical_lock = 2/(1+phi) = {classical_lock:.10f}")
print(f"Reference Fe-PHI: {ref_phi:.10f}")

# Try: cos^2(pi * (phi-1)/(phi+1))
test = np.cos(np.pi * (phi - 1) / (phi + 1))**2
print(f"cos^2(pi*(phi-1)/(phi+1)) = {test:.10f}")

# Try: 4*f1*f2/(f1+f2)^2 (transmittance)
f1p, f2p = 286.0, phi_freq
t = 4*f1p*f2p / (f1p + f2p)**2
print(f"4*f1*f2/(f1+f2)^2 = {t:.10f}")

# Try: 1 - ((phi-1)/(phi+1))^2
test2 = 1 - ((phi - 1)/(phi + 1))**2
print(f"1 - ((phi-1)/(phi+1))^2 = {test2:.10f}")

# Try: 4*phi/(1+phi)^2
test3 = 4*phi / (1+phi)**2
print(f"4*phi/(1+phi)^2 = {test3:.10f}")
