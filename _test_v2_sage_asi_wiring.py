#!/usr/bin/env python3
"""Test Professor Mode V2 wiring into Sage Mode, Sage Core, and ASI Pipeline."""

import sys
import os
import traceback

# Fix pre-existing path issue in sage_core's CodeSelfModifier
os.makedirs("/workspaces/Allentown-L104-Node/.l104_backups", exist_ok=True) if os.path.exists("/workspaces") else None
os.makedirs(os.path.join(os.path.dirname(__file__), ".l104_backups"), exist_ok=True)

PASS = 0
FAIL = 0

def test(name, fn):
    global PASS, FAIL
    try:
        result = fn()
        if result:
            print(f"  ✓ {name}")
            PASS += 1
        else:
            print(f"  ✗ {name} — returned False")
            FAIL += 1
    except Exception as e:
        print(f"  ✗ {name} — {e}")
        traceback.print_exc()
        FAIL += 1


print("=" * 70)
print("  PROFESSOR V2 → SAGE MODE / SAGE CORE / ASI PIPELINE WIRING TEST")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# 1. SAGE MODE WIRING
# ═══════════════════════════════════════════════════════════════
print("\n[1/3] SAGE MODE WIRING")

def test_sage_mode_import():
    from l104_sage_mode import sage_mode, PROFESSOR_V2_AVAILABLE
    return PROFESSOR_V2_AVAILABLE

def test_sage_mode_v2_subsystems():
    from l104_sage_mode import sage_mode
    return (
        sage_mode._v2_available and
        sage_mode._v2_hilbert is not None and
        sage_mode._v2_coding is not None and
        sage_mode._v2_magic is not None and
        sage_mode._v2_research is not None and
        sage_mode._v2_research_team is not None and
        sage_mode._v2_intellect is not None and
        sage_mode._v2_crystallizer is not None and
        sage_mode._v2_evaluator is not None
    )

def test_sage_mode_v2_status():
    from l104_sage_mode import sage_mode
    status = sage_mode.get_v2_status()
    return status["v2_available"] and status["hilbert"] and status["coding"] and status["magic"]

def test_sage_mode_v2_research():
    from l104_sage_mode import sage_mode
    result = sage_mode.v2_research_invention("QUANTUM_VOID_THEORY", depth=2)
    return "error" not in result and "research" in result

def test_sage_mode_v2_coding():
    from l104_sage_mode import sage_mode
    result = sage_mode.v2_coding_invention("recursion", paradigm="functional")
    return "error" not in result and "teaching" in result

def test_sage_mode_v2_magic():
    from l104_sage_mode import sage_mode
    result = sage_mode.v2_magic_derivation("golden_ratio", depth=3)
    return "error" not in result and "derivation" in result

def test_sage_mode_v2_intellect():
    from l104_sage_mode import sage_mode
    result = sage_mode.v2_unlimited_intellect_solve("consciousness")
    return "error" not in result and "solution" in result

test("Import + PROFESSOR_V2_AVAILABLE", test_sage_mode_import)
test("V2 subsystems initialized", test_sage_mode_v2_subsystems)
test("get_v2_status()", test_sage_mode_v2_status)
test("v2_research_invention()", test_sage_mode_v2_research)
test("v2_coding_invention()", test_sage_mode_v2_coding)
test("v2_magic_derivation()", test_sage_mode_v2_magic)
test("v2_unlimited_intellect_solve()", test_sage_mode_v2_intellect)


# ═══════════════════════════════════════════════════════════════
# 2. SAGE CORE WIRING
# ═══════════════════════════════════════════════════════════════
print("\n[2/3] SAGE CORE WIRING")

def test_sage_core_import():
    from l104_sage_core import sage_core, PROFESSOR_V2_AVAILABLE
    return PROFESSOR_V2_AVAILABLE

def test_sage_core_v2_subsystems():
    from l104_sage_core import sage_core
    return (
        sage_core._v2_available and
        sage_core._v2_hilbert is not None and
        sage_core._v2_coding is not None and
        sage_core._v2_magic is not None and
        sage_core._v2_research is not None and
        sage_core._v2_intellect is not None
    )

def test_sage_core_status():
    from l104_sage_core import sage_core
    status = sage_core.get_status()
    return "professor_v2" in status and status["professor_v2"]["available"]

def test_sage_core_v2_research():
    from l104_sage_core import sage_core
    result = sage_core.v2_research("SAGE_WISDOM_MATH", depth=2)
    return "error" not in result and "research" in result

def test_sage_core_v2_coding():
    from l104_sage_core import sage_core
    result = sage_core.v2_coding_mastery("polymorphism")
    return "error" not in result and "teaching" in result

def test_sage_core_v2_magic():
    from l104_sage_core import sage_core
    result = sage_core.v2_magic_derivation("fibonacci_spiral", depth=3)
    return "error" not in result and "derivation" in result

def test_sage_core_v2_solve():
    from l104_sage_core import sage_core
    result = sage_core.v2_unlimited_solve("recursion")
    return "error" not in result and "solution" in result

test("Import + PROFESSOR_V2_AVAILABLE", test_sage_core_import)
test("V2 subsystems initialized", test_sage_core_v2_subsystems)
test("get_status() includes V2", test_sage_core_status)
test("v2_research()", test_sage_core_v2_research)
test("v2_coding_mastery()", test_sage_core_v2_coding)
test("v2_magic_derivation()", test_sage_core_v2_magic)
test("v2_unlimited_solve()", test_sage_core_v2_solve)


# ═══════════════════════════════════════════════════════════════
# 3. ASI PIPELINE WIRING
# ═══════════════════════════════════════════════════════════════
print("\n[3/3] ASI PIPELINE WIRING")

def test_asi_import():
    from l104_asi_core import asi_core, PROFESSOR_V2_AVAILABLE
    return PROFESSOR_V2_AVAILABLE

def test_asi_v2_slot():
    from l104_asi_core import asi_core
    return hasattr(asi_core, '_professor_v2')

def test_asi_v2_metrics():
    from l104_asi_core import asi_core
    m = asi_core._pipeline_metrics
    return all(k in m for k in [
        "v2_research_cycles", "v2_coding_mastery",
        "v2_magic_derivations", "v2_hilbert_validations"
    ])

def test_asi_connect_pipeline():
    from l104_asi_core import asi_core
    result = asi_core.connect_pipeline()
    connected = result.get("connected", [])
    return "professor_v2" in connected

def test_asi_v2_connected():
    from l104_asi_core import asi_core
    return asi_core._professor_v2 is not None and isinstance(asi_core._professor_v2, dict)

def test_asi_status_shows_v2():
    from l104_asi_core import asi_core
    status = asi_core.get_status()
    return "professor_v2" in status.get("subsystems", {})

def test_asi_pipeline_research():
    from l104_asi_core import asi_core
    result = asi_core.pipeline_professor_research("ASI_CONSCIOUSNESS", depth=2)
    return result.get("status") == "completed"

def test_asi_pipeline_coding():
    from l104_asi_core import asi_core
    result = asi_core.pipeline_coding_mastery("abstract_factory")
    return result.get("status") == "mastered"

def test_asi_pipeline_magic():
    from l104_asi_core import asi_core
    result = asi_core.pipeline_magic_derivation("phi_spiral", depth=3)
    return result.get("status") == "derived"

def test_asi_pipeline_hilbert():
    from l104_asi_core import asi_core
    result = asi_core.pipeline_hilbert_validate("test_concept")
    return result.get("status") == "validated"

test("Import + PROFESSOR_V2_AVAILABLE", test_asi_import)
test("_professor_v2 slot exists", test_asi_v2_slot)
test("V2 metrics in _pipeline_metrics", test_asi_v2_metrics)
test("connect_pipeline() includes professor_v2", test_asi_connect_pipeline)
test("_professor_v2 is connected dict", test_asi_v2_connected)
test("get_status() shows professor_v2", test_asi_status_shows_v2)
test("pipeline_professor_research()", test_asi_pipeline_research)
test("pipeline_coding_mastery()", test_asi_pipeline_coding)
test("pipeline_magic_derivation()", test_asi_pipeline_magic)
test("pipeline_hilbert_validate()", test_asi_pipeline_hilbert)


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
total = PASS + FAIL
print(f"  RESULTS: {PASS}/{total} PASSED  |  {FAIL} FAILED")
if FAIL == 0:
    print("  STATUS: ALL TESTS PASSED ✓")
else:
    print(f"  STATUS: {FAIL} TESTS FAILED ✗")
print("=" * 70)

sys.exit(0 if FAIL == 0 else 1)
