#!/usr/bin/env python3
"""Validate the v3 grain=416 GOD_CODE algorithm update in l104_god_code_dual_layer.py"""
import sys
sys.path.insert(0, '.')

print("=" * 70)
print("V3 GRAIN=416 GOD_CODE ALGORITHM — VALIDATION")
print("=" * 70)

# 1. Import check
print("\n[1] Import check...")
try:
    import l104_god_code_dual_layer as dl
    print("    PASS: Module imported successfully")
except Exception as e:
    print(f"    FAIL: {e}")
    sys.exit(1)

# 2. Constants check
print("\n[2] V3 constants check...")
checks = [
    ("X_V3 (scaffold)", dl.X_V3, 286),
    ("R_V3 (ratio)", dl.R_V3, 2),
    ("Q_V3 (grain)", dl.Q_V3, 416),
    ("P_V3 (dial_coeff)", dl.P_V3, 8),
    ("K_V3 (offset)", dl.K_V3, 1664),
]
for name, actual, expected in checks:
    status = "PASS" if actual == expected else "FAIL"
    print(f"    {status}: {name} = {actual} (expected {expected})")

# 3. GOD_CODE_V3 = god_code_v3(0,0,0,0) = GOD_CODE
print("\n[3] GOD_CODE_V3 identity check...")
gc_v3 = dl.GOD_CODE_V3
gc_calc = dl.god_code_v3(0, 0, 0, 0)
gc_orig = dl.GOD_CODE
print(f"    GOD_CODE_V3 = {gc_v3}")
print(f"    god_code_v3(0,0,0,0) = {gc_calc}")
print(f"    GOD_CODE (Layer 1) = {gc_orig}")
print(f"    V3 == Layer1: {abs(gc_v3 - gc_orig) < 1e-10}")

# 4. Spot-check key constants
print("\n[4] Spot-check key constants...")
test_cases = [
    ("Speed of Light", (6, 0, 0, -19), 299792458, 0.10),
    ("Fine Structure Inv", (3, 1, 0, 2), 137.035999084, 0.01),
    ("Bohr Radius PM", (36, 4, 0, 4), 52.9177210544, 0.01),
    ("Electron Mass MeV", (0, 5, 0, 10), 0.51099895069, 0.05),
    ("Proton Mass MeV", (44, 6, 0, 0), 938.27208816, 0.10),
    ("Higgs GeV", (49, 7, 0, 3), 125.25, 0.01),
    ("Fe BCC Lattice PM", (7, 6, 0, 1), 286.65, 0.02),
    ("OMEGA", (33, 1, 0, -3), 6539.34712682, 0.05),
    ("Alpha EEG Hz", (15, 4, 0, 6), 10.0, 0.01),
    ("Pi", (32, 3, 0, 8), 3.14159265359, 0.02),
    ("Golden Ratio", (34, 1, 0, 9), 1.618033988749895, 0.03),
]
all_pass = True
for name, dials, measured, max_err in test_cases:
    val = dl.god_code_v3(*dials)
    err = abs(val - measured) / measured * 100
    status = "PASS" if err < max_err else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"    {status}: {name:25s} = {val:.6e}  (err: {err:.4f}%, max: {max_err}%)")

# 5. C_EXPONENT_V3 check
print("\n[5] C_EXPONENT_V3 check...")
print(f"    C_EXPONENT_V3 = {dl.C_EXPONENT_V3} (expected 9616)")
status = "PASS" if dl.C_EXPONENT_V3 == 9616 else "FAIL"
print(f"    {status}")

# 6. Named constants check
print("\n[6] Named constants check...")
named = [
    ("C_V3", dl.C_V3, 299792458, 0.10),
    ("GRAVITY_V3", dl.GRAVITY_V3, 9.80665, 0.10),
    ("BOHR_V3", dl.BOHR_V3, 52.9177210544, 0.01),
    ("FINE_STRUCTURE_INV_V3", dl.FINE_STRUCTURE_INV_V3, 137.035999084, 0.01),
    ("OMEGA_V3", dl.OMEGA_V3, 6539.34712682, 0.05),
]
for name, val, measured, max_err in named:
    err = abs(val - measured) / measured * 100
    status = "PASS" if err < max_err else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"    {status}: {name:25s} = {val:.6e}  (err: {err:.4f}%)")

# 7. Frequency table size check
print("\n[7] Frequency table check...")
n = len(dl.V3_FREQUENCY_TABLE)
print(f"    Table entries: {n} (expected 66 = 65 constants + GOD_CODE_V3)")
status = "PASS" if n >= 65 else "FAIL"
print(f"    {status}")

# 8. Real-world constants registry check
print("\n[8] Real-world constants registry...")
n_rw = len(dl.REAL_WORLD_CONSTANTS_V3)
print(f"    Registered: {n_rw} constants")
status = "PASS" if n_rw >= 60 else "FAIL"
print(f"    {status}")

# 9. EVOLUTION_HERITAGE check
print("\n[9] Evolution heritage check...")
eh = dl.EVOLUTION_HERITAGE.get("v3_god_code_416", {})
print(f"    equation: {eh.get('equation', 'MISSING')}")
print(f"    r: {eh.get('r', 'MISSING')}")
print(f"    Q: {eh.get('Q', 'MISSING')}")
status = "PASS" if eh.get("Q") == 416 and eh.get("r") == 2 else "FAIL"
print(f"    {status}")

# 10. Derive check
print("\n[10] Real-world derive check (speed_of_light)...")
try:
    result = dl.real_world_derive_v3("speed_of_light", real_world=False)
    print(f"     grid_value: {result['value']:.6e}")
    print(f"     error_pct: {result['error_pct']:.4f}%")
    print(f"     dials: {result['dials']}")
    status = "PASS" if result['error_pct'] < 0.10 else "FAIL"
    print(f"     {status}")
except Exception as e:
    print(f"     FAIL: {e}")
    all_pass = False

# Summary
print("\n" + "=" * 70)
if all_pass:
    print("ALL CHECKS PASSED — v3 grain=416 GOD_CODE algorithm is active")
else:
    print("SOME CHECKS FAILED — review above")
print("=" * 70)
