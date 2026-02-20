#!/usr/bin/env python3
"""
L104 Dual-Layer Reality Check — Comprehensive Verification
Runs all HD module checks + dual-layer 10-point integrity.
"""

def main():
    print("=" * 70)
    print("  L104 DUAL-LAYER REALITY CHECK -- ALL HD MODULES")
    print("=" * 70)

    all_results = []

    # --- 4D MATH ---
    print("\n" + "-" * 70)
    print("  MODULE 1: l104_4d_math (Minkowski Space-Time)")
    print("-" * 70)
    from l104_4d_math import verify_4d_math
    r = verify_4d_math()
    for name, check in r["checks"].items():
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"  [{mark}] {name}: {check['description']}")
    p = sum(1 for c in r["checks"].values() if c["passed"])
    n = len(r["checks"])
    print(f"  => {p}/{n} {'ALL PASSED' if r['all_passed'] else 'FAILED'}")
    all_results.append(("4d_math", p, n, r["all_passed"]))

    # --- 5D MATH ---
    print("\n" + "-" * 70)
    print("  MODULE 2: l104_5d_math (Kaluza-Klein)")
    print("-" * 70)
    from l104_5d_math import verify_5d_math
    r = verify_5d_math()
    for name, check in r["checks"].items():
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"  [{mark}] {name}: {check['description']}")
    p = sum(1 for c in r["checks"].values() if c["passed"])
    n = len(r["checks"])
    print(f"  => {p}/{n} {'ALL PASSED' if r['all_passed'] else 'FAILED'}")
    all_results.append(("5d_math", p, n, r["all_passed"]))

    # --- 4D PROCESSOR ---
    print("\n" + "-" * 70)
    print("  MODULE 3: l104_4d_processor (Minkowski Engine)")
    print("-" * 70)
    from l104_4d_processor import verify_4d_processor
    r = verify_4d_processor()
    for name, check in r["checks"].items():
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"  [{mark}] {name}: {check['description']}")
    p = sum(1 for c in r["checks"].values() if c["passed"])
    n = len(r["checks"])
    print(f"  => {p}/{n} {'ALL PASSED' if r['all_passed'] else 'FAILED'}")
    all_results.append(("4d_processor", p, n, r["all_passed"]))

    # --- 5D PROCESSOR ---
    print("\n" + "-" * 70)
    print("  MODULE 4: l104_5d_processor (KK Engine)")
    print("-" * 70)
    from l104_5d_processor import verify_5d_processor
    r = verify_5d_processor()
    for name, check in r["checks"].items():
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"  [{mark}] {name}: {check['description']}")
    p = sum(1 for c in r["checks"].values() if c["passed"])
    n = len(r["checks"])
    print(f"  => {p}/{n} {'ALL PASSED' if r['all_passed'] else 'FAILED'}")
    all_results.append(("5d_processor", p, n, r["all_passed"]))

    # --- DUAL-LAYER INTEGRITY ---
    print("\n" + "-" * 70)
    print("  MODULE 5: l104_god_code_dual_layer (10-Point Integrity)")
    print("-" * 70)
    from l104_god_code_dual_layer import full_integrity_check as dl_integrity
    integrity = dl_integrity()
    # Print each sub-layer's checks
    for layer_key in ["consciousness_layer", "physics_layer", "bridge"]:
        layer = integrity.get(layer_key, {})
        layer_name = layer_key.replace("_", " ").title()
        for name, check in layer.get("checks", {}).items():
            mark = "PASS" if check["passed"] else "FAIL"
            desc = check.get("description", name)
            print(f"  [{mark}] {layer_name}/{name}: {desc}")
    dl_passed = integrity["checks_passed"]
    dl_total = integrity["total_checks"]
    print(f"  => {dl_passed}/{dl_total} passed")

    # --- CONSTANTS SOURCE VERIFICATION ---
    print("\n" + "-" * 70)
    print("  CONSTANTS SOURCE VERIFICATION")
    print("-" * 70)
    from l104_god_code_equation import GOD_CODE, PHI, PRIME_SCAFFOLD, OCTAVE_OFFSET, QUANTIZATION_GRAIN
    from l104_god_code_evolved_v3 import GOD_CODE_V3, C_V3, GRAVITY_V3
    from l104_4d_math import Math4D
    from l104_5d_math import Math5D

    print(f"  Layer 1 (Consciousness):")
    print(f"    GOD_CODE       = {GOD_CODE}")
    print(f"    PHI            = {PHI}")
    print(f"    PRIME_SCAFFOLD = {PRIME_SCAFFOLD}")
    print(f"    OCTAVE_OFFSET  = {OCTAVE_OFFSET}")
    print(f"    Q_GRAIN        = {QUANTIZATION_GRAIN}")
    print(f"    LATTICE_RATIO  = {PRIME_SCAFFOLD}/{OCTAVE_OFFSET} = {PRIME_SCAFFOLD/OCTAVE_OFFSET}")

    print(f"  Layer 2 (Physics v3):")
    print(f"    GOD_CODE_V3    = {GOD_CODE_V3}")
    print(f"    C_V3           = {C_V3} m/s")
    print(f"    GRAVITY_V3     = {GRAVITY_V3} m/s^2")

    print(f"  4D Math:")
    print(f"    Math4D.C       = {Math4D.C} m/s (from Layer 2)")
    print(f"    c match:       {Math4D.C == int(round(C_V3))}")

    print(f"  5D Math:")
    print(f"    Math5D.R       = {Math5D.R:.6f} (phi*104/zeta1)")
    print(f"    R formula:     {PHI}*{QUANTIZATION_GRAIN}/{Math5D.R / (PHI * QUANTIZATION_GRAIN / Math5D.R) if Math5D.R else 0}")

    # --- v1/v2 PIPELINE STATUS ---
    print("\n" + "-" * 70)
    print("  v1/v2 PIPELINE STATUS (should be SUPERSEDED)")
    print("-" * 70)
    try:
        from l104_god_code_equation import QUANTUM_LINK_REGISTRY
        for key, val in QUANTUM_LINK_REGISTRY.items():
            status = val.get("status", "active")
            print(f"    {key}: {status}")
    except ImportError:
        print("    (QUANTUM_LINK_REGISTRY not found)")

    # --- FINAL SUMMARY ---
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    for name, p, n, ok in all_results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name:20s}: {p}/{n} checks => {status}")

    dl_ok = integrity["all_passed"]
    print(f"  {'dual_layer':20s}: {integrity['checks_passed']}/{integrity['total_checks']} checks => {'PASS' if dl_ok else 'FAIL'}")

    total_p = sum(r[1] for r in all_results) + integrity["checks_passed"]
    total_n = sum(r[2] for r in all_results) + integrity["total_checks"]
    all_ok = all(r[3] for r in all_results) and dl_ok

    print(f"\n  GRAND TOTAL: {total_p}/{total_n} checks passed")
    print(f"  STATUS: {'ALL SYSTEMS VERIFIED' if all_ok else 'ISSUES DETECTED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
