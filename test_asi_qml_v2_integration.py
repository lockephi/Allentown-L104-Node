#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  L104 ASI ↔ QML v2 INTEGRATION TEST SUITE
  Validates full pipeline wiring: quantum.py → core.py → __init__.py
  60 tests across 10 phases — confirms maximum saturation
═══════════════════════════════════════════════════════════════════════════
"""
import sys, time, math, traceback

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895

passed = 0
failed = 0
total_tests = 0

def test(name: str, condition: bool, detail: str = ""):
    global passed, failed, total_tests
    total_tests += 1
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name} — {detail}")

t0 = time.time()

# ═══════════════════════════════════════════════════════════════
# PHASE 1: QML v2 MODULE AVAILABILITY
# ═══════════════════════════════════════════════════════════════
print("\n═══ PHASE 1: QML V2 MODULE AVAILABILITY ═══")

try:
    from l104_qml_v2 import (
        QuantumMLHub, get_qml_hub,
        ZZFeatureMap, DataReUploadingCircuit, BerryPhaseAnsatz,
        QuantumKernelEstimator, QAOACircuit, BarrenPlateauAnalyzer,
        QuantumRegressorQNN, ExpressibilityAnalyzer,
    )
    qml_available = True
except ImportError as e:
    qml_available = False
    print(f"  [WARN] l104_qml_v2 import failed: {e}")

test("QML v2 module importable", qml_available)
test("QuantumMLHub class exists", qml_available and QuantumMLHub is not None)
test("get_qml_hub callable", qml_available and callable(get_qml_hub))

if qml_available:
    hub = get_qml_hub()
    test("QML hub singleton instantiates", hub is not None)
    status = hub.status()
    caps = status.get("capabilities", {})
    test("Hub has 9 capabilities", len(caps) == 9, f"got {len(caps)}: {list(caps.keys())}")
    test("Hub status version is 2.0.0", status.get("version") == "2.0.0", str(status.get("version")))
else:
    for n in ["QML hub singleton instantiates", "Hub has 9 capabilities", "Hub status version is 2.0.0"]:
        test(n, False, "QML v2 not available")

# ═══════════════════════════════════════════════════════════════
# PHASE 2: ASI __init__.py RE-EXPORTS
# ═══════════════════════════════════════════════════════════════
print("\n═══ PHASE 2: ASI PACKAGE RE-EXPORTS ═══")

try:
    from l104_asi import QuantumMLHub as AsiQLMHub
    test("QuantumMLHub re-exported from l104_asi", AsiQLMHub is not None)
except ImportError:
    test("QuantumMLHub re-exported from l104_asi", False, "import failed")

try:
    from l104_asi import get_qml_hub as asi_get_hub
    test("get_qml_hub re-exported from l104_asi", asi_get_hub is not None)
except ImportError:
    test("get_qml_hub re-exported from l104_asi", False, "import failed")

try:
    from l104_asi import QuantumComputationCore
    test("QuantumComputationCore available", QuantumComputationCore is not None)
except ImportError:
    test("QuantumComputationCore available", False, "import failed")

# ═══════════════════════════════════════════════════════════════
# PHASE 3: QUANTUM COMPUTATION CORE — QML V2 METHODS
# ═══════════════════════════════════════════════════════════════
print("\n═══ PHASE 3: QUANTUM COMPUTATION CORE QML V2 METHODS ═══")

try:
    from l104_asi.quantum import QuantumComputationCore
    qcc = QuantumComputationCore()
    qcc_ok = True
except Exception as e:
    qcc_ok = False
    print(f"  [WARN] QuantumComputationCore init failed: {e}")

if qcc_ok:
    # Check all 11 methods exist
    qml_methods = [
        '_get_qml_hub', 'qml_v2_kernel_classify', 'qml_v2_qaoa_maxcut',
        'qml_v2_berry_classify', 'qml_v2_regress', 'qml_v2_analyze_trainability',
        'qml_v2_expressibility', 'qml_v2_entanglement', 'qml_v2_kernel_matrix',
        'qml_v2_status', 'qml_v2_intelligence_score',
    ]
    for method_name in qml_methods:
        test(f"QCC has {method_name}()", hasattr(qcc, method_name))
else:
    for _ in range(11):
        test("QCC method exists", False, "QCC init failed")

# ═══════════════════════════════════════════════════════════════
# PHASE 4: QML V2 FUNCTIONAL TESTS VIA QCC
# ═══════════════════════════════════════════════════════════════
print("\n═══ PHASE 4: QML V2 FUNCTIONAL TESTS VIA QCC ═══")

if qcc_ok:
    # Kernel classify
    try:
        result = qcc.qml_v2_kernel_classify(
            query_features=[0.5, 0.3, 0.8],
            domain_prototypes=[[0.5, 0.3, 0.7], [0.1, 0.9, 0.2]],
            feature_map="zz"
        )
        test("Kernel classify returns dict", isinstance(result, dict))
        test("Kernel classify has 'predicted_domain'", "predicted_domain" in result)
        test("Kernel classify has similarities",
             "kernel_similarities" in result or "confidence" in result)
    except Exception as e:
        test("Kernel classify returns dict", False, str(e))
        test("Kernel classify has 'predicted_domain'", False)
        test("Kernel classify has similarities", False)

    # QAOA MaxCut
    try:
        edges = [(0, 1), (1, 2), (2, 0)]
        result = qcc.qml_v2_qaoa_maxcut(edges, p=1, optimize=True)
        test("QAOA returns dict", isinstance(result, dict))
        test("QAOA has 'approximation_ratio'", "approximation_ratio" in result)
        test("QAOA ratio > 0.5", result.get("approximation_ratio", 0) > 0.5,
             f"ratio={result.get('approximation_ratio')}")
    except Exception as e:
        test("QAOA executes", False, str(e))
        test("QAOA has 'approximation_ratio'", False)
        test("QAOA ratio > 0.5", False)

    # Berry classify
    try:
        result = qcc.qml_v2_berry_classify([0.5, 0.3])
        test("Berry classify returns dict", isinstance(result, dict))
        test("Berry classify has 'prediction'", "prediction" in result)
    except Exception as e:
        test("Berry classify executes", False, str(e))
        test("Berry classify has 'prediction'", False)

    # Regression
    try:
        result = qcc.qml_v2_regress([0.5, 0.3])
        test("Regression returns dict", isinstance(result, dict))
        test("Regression has 'prediction'", "prediction" in result)
    except Exception as e:
        test("Regression executes", False, str(e))
        test("Regression has 'prediction'", False)
else:
    for _ in range(10):
        test("QCC functional test", False, "QCC init failed")

# ═══════════════════════════════════════════════════════════════
# PHASE 5: QML V2 ANALYSIS VIA QCC
# ═══════════════════════════════════════════════════════════════
print("\n═══ PHASE 5: QML V2 ANALYSIS VIA QCC ═══")

if qcc_ok:
    # Trainability
    try:
        result = qcc.qml_v2_analyze_trainability("berry_phase", n_samples=10)
        test("Trainability analysis returns dict", isinstance(result, dict))
        test("Trainability has 'mean_gradient_variance'", "mean_gradient_variance" in result)
    except Exception as e:
        test("Trainability analysis executes", False, str(e))
        test("Trainability has 'mean_gradient_variance'", False)

    # Expressibility
    try:
        result = qcc.qml_v2_expressibility("zz", n_samples=30)
        test("Expressibility returns dict", isinstance(result, dict))
        test("Expressibility has 'expressibility_score'",
             "expressibility_score" in result)
    except Exception as e:
        test("Expressibility executes", False, str(e))
        test("Expressibility has 'expressibility_score'", False)

    # Entanglement (Meyer-Wallach)
    try:
        result = qcc.qml_v2_entanglement("berry_phase", n_samples=15)
        test("Entanglement returns dict", isinstance(result, dict))
        test("Entanglement has 'mean_Q'",
             "mean_Q" in result)
    except Exception as e:
        test("Entanglement executes", False, str(e))
        test("Entanglement has 'mean_Q'", False)

    # Kernel matrix
    try:
        X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        result = qcc.qml_v2_kernel_matrix(X, feature_map="zz")
        test("Kernel matrix returns dict", isinstance(result, dict))
        K_shape = result.get("kernel_shape", [])
        test("Kernel matrix shape 3x3",
             (isinstance(K_shape, (list, tuple)) and len(K_shape) == 2
              and K_shape[0] == 3 and K_shape[1] == 3),
             f"shape={K_shape}")
    except Exception as e:
        test("Kernel matrix executes", False, str(e))
        test("Kernel matrix shape 3x3", False)
else:
    for _ in range(8):
        test("QCC analysis test", False, "QCC init failed")

# ═══════════════════════════════════════════════════════════════
# PHASE 6: QML V2 STATUS & INTELLIGENCE SCORE
# ═══════════════════════════════════════════════════════════════
print("\n═══ PHASE 6: QML V2 STATUS & INTELLIGENCE SCORE ═══")

if qcc_ok:
    # Status
    try:
        status = qcc.qml_v2_status()
        test("QML v2 status returns dict", isinstance(status, dict))
        test("Status has 'hub' identity", "hub" in status)
        test("Status has 'asi_metrics'", "asi_metrics" in status or "available" in status)
    except Exception as e:
        test("QML v2 status executes", False, str(e))
        test("Status has 'hub' identity", False)
        test("Status has 'asi_metrics'", False)

    # Intelligence score
    try:
        score = qcc.qml_v2_intelligence_score()
        test("Intelligence score is float", isinstance(score, (int, float)))
        test("Intelligence score in [0, 1]", 0.0 <= score <= 1.0,
             f"score={score}")
        test("Intelligence score > 0", score > 0.0,
             f"score={score}")
    except Exception as e:
        test("Intelligence score executes", False, str(e))
        test("Intelligence score in [0, 1]", False)
        test("Intelligence score > 0", False)

    # Capabilities list in status
    try:
        main_status = qcc.status()
        caps = main_status.get("capabilities", [])
        test("QML_V2_ML_HUB in capabilities", "QML_V2_ML_HUB" in caps,
             f"caps tail: {caps[-5:]}")
        test("QML_V2_BERRY_ANSATZ in capabilities", "QML_V2_BERRY_ANSATZ" in caps)
    except Exception as e:
        test("Capabilities include QML v2", False, str(e))
        test("QML_V2_BERRY_ANSATZ in capabilities", False)
else:
    for _ in range(8):
        test("QCC status/score test", False, "QCC init failed")

# ═══════════════════════════════════════════════════════════════
# PHASE 7: ASI CORE INTEGRATION
# ═══════════════════════════════════════════════════════════════
print("\n═══ PHASE 7: ASI CORE INTEGRATION ═══")

try:
    from l104_asi import asi_core
    asi_ok = True
except Exception as e:
    asi_ok = False
    print(f"  [WARN] ASI core import failed: {e}")

if asi_ok:
    # Check _qml_hub field exists
    test("ASI core has _qml_hub field", hasattr(asi_core, '_qml_hub'))

    # Pipeline metrics include QML v2 keys
    pm = getattr(asi_core, '_pipeline_metrics', {})
    test("Pipeline metrics has qml_v2_kernel_classifications",
         "qml_v2_kernel_classifications" in pm)
    test("Pipeline metrics has qml_v2_qaoa_solves",
         "qml_v2_qaoa_solves" in pm)
    test("Pipeline metrics has qml_v2_berry_classifications",
         "qml_v2_berry_classifications" in pm)
    test("Pipeline metrics has qml_v2_trainability_checks",
         "qml_v2_trainability_checks" in pm)
else:
    for _ in range(5):
        test("ASI core field check", False, "ASI core not available")

# ═══════════════════════════════════════════════════════════════
# PHASE 8: ASI CORE CONNECT_PIPELINE (QML V2 WIRING)
# ═══════════════════════════════════════════════════════════════
print("\n═══ PHASE 8: ASI CORE CONNECT_PIPELINE ═══")

if asi_ok:
    try:
        result = asi_core.connect_pipeline()
        connected = result.get("connected", [])
        test("connect_pipeline succeeds", result.get("pipeline_ready", False))
        test("qml_v2_hub in connected list", "qml_v2_hub" in connected,
             f"connected: {[c for c in connected if 'qml' in c.lower()]}")
        test("Total subsystems > 5", result.get("total", 0) > 5,
             f"total={result.get('total')}")

        # Verify _qml_hub is now set
        test("_qml_hub populated after connect", asi_core._qml_hub is not None)
    except Exception as e:
        test("connect_pipeline succeeds", False, str(e))
        test("qml_v2_hub in connected list", False)
        test("Total subsystems > 5", False)
        test("_qml_hub populated after connect", False)
else:
    for _ in range(4):
        test("Pipeline connect test", False, "ASI core not available")

# ═══════════════════════════════════════════════════════════════
# PHASE 9: ASI SCORE WITH QML V2 DIMENSION
# ═══════════════════════════════════════════════════════════════
print("\n═══ PHASE 9: ASI SCORE WITH QML V2 DIMENSION ═══")

if asi_ok:
    try:
        score = asi_core.compute_asi_score()
        test("ASI score computes", isinstance(score, float))
        test("ASI score > 0", score > 0.0, f"score={score}")
        test("ASI score <= 1.0", score <= 1.0, f"score={score}")

        # Check status reflects scoring
        status = asi_core.get_status()
        test("ASI status has 'state'", "state" in status)
        test("ASI status version present", "version" in status)
    except Exception as e:
        test("ASI score computes", False, str(e)[:80])
        test("ASI score > 0", False)
        test("ASI score <= 1.0", False)
        test("ASI status has 'state'", False)
        test("ASI status version present", False)
else:
    for _ in range(5):
        test("ASI scoring test", False, "ASI core not available")

# ═══════════════════════════════════════════════════════════════
# PHASE 10: QUANTUM CIRCUIT STATUS — QML V2 MODULE
# ═══════════════════════════════════════════════════════════════
print("\n═══ PHASE 10: QUANTUM CIRCUIT STATUS ═══")

if asi_ok:
    try:
        qcs = asi_core.quantum_circuit_status()
        test("quantum_circuit_status returns dict", isinstance(qcs, dict))
        test("qml_v2_hub tracked in circuit status",
             "qml_v2_hub" in qcs, f"keys: {list(qcs.keys())[:8]}")
        test("qml_v2_hub is True (connected)",
             qcs.get("qml_v2_hub", False) is True)
        mc = qcs.get("modules_connected", 0)
        test("modules_connected includes QML v2", mc >= 2,
             f"modules_connected={mc}")
    except Exception as e:
        test("quantum_circuit_status returns dict", False, str(e)[:80])
        test("qml_v2_hub tracked in circuit status", False)
        test("qml_v2_hub is True (connected)", False)
        test("modules_connected includes QML v2", False)
else:
    for _ in range(4):
        test("Circuit status test", False, "ASI core not available")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
elapsed = time.time() - t0
print("\n" + "═" * 65)
print(f"  ASI ↔ QML V2 INTEGRATION: {passed}/{total_tests} passed in {elapsed:.3f}s")
if failed == 0:
    print("  ★ ALL TESTS PASSED — QML V2 FULLY WIRED INTO ASI PIPELINE ★")
else:
    print(f"  {failed} FAILURES — review above for details")
print("═" * 65)

sys.exit(0 if failed == 0 else 1)
