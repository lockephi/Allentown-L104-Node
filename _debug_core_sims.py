#!/usr/bin/env python3
"""Debug the 2 failing core simulations."""
import math
from l104_god_code_simulator.constants import GOD_CODE
from l104_god_code_simulator.quantum_primitives import god_code_dial, god_code_fn

# === Bug 1: dial_sweep_a ===
# Conservation law: G(a,0,0,0) = BASE * 2^((8a+416)/104)
# So G(a) / G(0) = 2^(8a/104)
# Therefore: G(a) * 2^(-8a/104) = G(0) = GOD_CODE  (conservation)
# The BUGGY test uses: g * 2^(+8a/104) which GROWS instead of conserving

print("=== DIAL_SWEEP_A DEBUG ===")
for a in range(4):
    g = god_code_dial(a=a)
    x = 8 * a
    product_buggy = g * (2.0 ** (x / 104.0))
    product_fixed = g * (2.0 ** (-x / 104.0))
    print(f"  a={a}: G={g:.6f}  buggy_product={product_buggy:.6f}  "
          f"fixed_product={product_fixed:.6f}  GOD_CODE={GOD_CODE:.6f}")

# === Bug 2: 104_tet_spectrum ===
# GOD_CODE = 286^(1/phi) * 2^(416/104) = 286^(1/phi) * 16
# Step in 104-TET from base=286: step = 104 * log2(GOD_CODE / 286)
# This equals: 104 * [(1/phi - 1)*log2(286) + 4]
# 1/phi - 1 = -0.38197... => step is NOT an integer
# The remainder is ~0.147 which is a sacred non-integer offset

god_code_step = 104 * math.log2(GOD_CODE / 286.0)
remainder = abs(god_code_step - round(god_code_step))
print(f"\n=== 104_TET_SPECTRUM DEBUG ===")
print(f"  GOD_CODE = {GOD_CODE:.10f}")
print(f"  GOD_CODE / 286 = {GOD_CODE / 286:.10f}")
print(f"  god_code_step = {god_code_step:.10f}")
print(f"  round(step) = {round(god_code_step)}")
print(f"  remainder = {remainder:.10f}")
print(f"  floor(step) = {math.floor(god_code_step)}")
print(f"  frac(step) = {god_code_step - math.floor(god_code_step):.10f}")

# The 416/104 = 4 exactly contributes step = 104*4 = 416
# But 286^(1/phi) contribution: 104*(1/phi - 1)*log2(286)
phi_step = 104 * (1/1.618033988749895 - 1) * math.log2(286)
octave_step = 104 * 4  # = 416
print(f"\n  phi contribution steps = {phi_step:.10f}")
print(f"  octave contribution steps = {octave_step}")
print(f"  total = {phi_step + octave_step:.10f}")
print(f"  The non-integer part ({remainder:.6f}) comes from phi's irrationality.")
print(f"  This is by DESIGN — GOD_CODE transcends the 104-TET lattice.")

# What the test SHOULD check: GOD_CODE is within the 104-TET lattice bounds
# and its fractional step position encodes the golden ratio
frac = god_code_step - math.floor(god_code_step)
phi_frac = 1.0 / 1.618033988749895  # 0.618...
print(f"\n  Fractional position in step = {frac:.10f}")
print(f"  Nearest sacred fractions:")
print(f"    1/phi   = {phi_frac:.10f}")
print(f"    phi - 1 = {phi_frac:.10f}")
print(f"    frac vs 1-1/phi = {abs(frac - (1-phi_frac)):.10f}")
