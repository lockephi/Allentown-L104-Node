#!/usr/bin/env python3
"""
L104 ALL-ENGINE QUANTUM CIRCUIT INTEGRATION — E2E Verification
Tests quantum circuit wiring across all 7 engine packages:
  Phase 1: Science Engine (5 modules)
  Phase 2: Math Engine (6 modules)
  Phase 3: Code Engine (6 modules)
  Phase 4: ASI Core (11 modules — expanded fleet)
  Phase 5: AGI Core (10 modules — expanded fleet)
  Phase 6: Intellect (7 modules)
  Phase 7: Server routes (11 circuit endpoints)
"""
import os, sys, time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Clear ALL IBM tokens to force simulator mode and prevent QPU hang
os.environ.pop("IBMQ_TOKEN", None)
os.environ.pop("IBM_QUANTUM_TOKEN", None)
os.environ.pop("IBM_QUANTUM_CHANNEL", None)
os.environ["IBMQ_TOKEN"] = ""
os.environ["IBM_QUANTUM_TOKEN"] = ""

# Patch quantum runtime BEFORE any engine imports to prevent QPU hang
# The runtime auto-connects in __init__, but with blank tokens it won't connect
try:
    from l104_quantum_runtime import get_runtime
    rt = get_runtime()
    rt._connected = False
    print("[PATCH] quantum_runtime._connected = False (safe simulator mode)")
except Exception:
    print("[PATCH] quantum_runtime not loaded — proceeding in offline mode")

passed = 0
failed = 0
errors = []

def check(name, condition, detail=""):
    global passed, failed, errors
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        errors.append(f"{name}: {detail}")
        print(f"  ❌ {name} — {detail}")

# ═══════════════════════════════════════════════
# PHASE 1: SCIENCE ENGINE
# ═══════════════════════════════════════════════
print("\n" + "═" * 60)
print("PHASE 1: SCIENCE ENGINE — Quantum Circuit Integration")
print("═" * 60)
try:
    from l104_science_engine import ScienceEngine
    se = ScienceEngine()
    check("ScienceEngine import", True)

    # Check quantum methods exist
    check("SE quantum_grover_search exists", hasattr(se, 'quantum_grover_search'))
    check("SE quantum_vqe_optimize exists", hasattr(se, 'quantum_vqe_optimize'))
    check("SE quantum_shor_factor exists", hasattr(se, 'quantum_shor_factor'))
    check("SE quantum_topological_compute exists", hasattr(se, 'quantum_topological_compute'))
    check("SE quantum_25q_build exists", hasattr(se, 'quantum_25q_build'))
    check("SE quantum_gravity_erepr exists", hasattr(se, 'quantum_gravity_erepr'))
    check("SE quantum_consciousness_phi exists", hasattr(se, 'quantum_consciousness_phi'))
    check("SE quantum_circuit_status exists", hasattr(se, 'quantum_circuit_status'))

    # Run status
    status = se.quantum_circuit_status()
    check("SE status returns dict", isinstance(status, dict), str(type(status)))
    check("SE status has version", 'version' in status, str(status.keys()))
except Exception as e:
    check("ScienceEngine import", False, str(e))

# ═══════════════════════════════════════════════
# PHASE 2: MATH ENGINE
# ═══════════════════════════════════════════════
print("\n" + "═" * 60)
print("PHASE 2: MATH ENGINE — Quantum Circuit Integration")
print("═" * 60)
try:
    from l104_math_engine import MathEngine
    me = MathEngine()
    check("MathEngine import", True)

    check("ME quantum_vqe_optimize exists", hasattr(me, 'quantum_vqe_optimize'))
    check("ME quantum_grover_search exists", hasattr(me, 'quantum_grover_search'))
    check("ME quantum_qaoa_optimize exists", hasattr(me, 'quantum_qaoa_optimize'))
    check("ME quantum_topological_compute exists", hasattr(me, 'quantum_topological_compute'))
    check("ME quantum_shor_factor exists", hasattr(me, 'quantum_shor_factor'))
    check("ME quantum_25q_build exists", hasattr(me, 'quantum_25q_build'))
    check("ME quantum_gravity_holographic exists", hasattr(me, 'quantum_gravity_holographic'))
    check("ME quantum_knot_invariant exists", hasattr(me, 'quantum_knot_invariant'))
    check("ME quantum_circuit_status exists", hasattr(me, 'quantum_circuit_status'))

    status = me.quantum_circuit_status()
    check("ME status returns dict", isinstance(status, dict), str(type(status)))
except Exception as e:
    check("MathEngine import", False, str(e))

# ═══════════════════════════════════════════════
# PHASE 3: CODE ENGINE
# ═══════════════════════════════════════════════
print("\n" + "═" * 60)
print("PHASE 3: CODE ENGINE — Quantum Circuit Integration")
print("═" * 60)
try:
    from l104_code_engine import code_engine
    check("CodeEngine import", True)

    check("CE quantum_coherence_grover exists", hasattr(code_engine, 'quantum_coherence_grover'))
    check("CE quantum_coherence_vqe exists", hasattr(code_engine, 'quantum_coherence_vqe'))
    check("CE quantum_coherence_shor exists", hasattr(code_engine, 'quantum_coherence_shor'))
    check("CE quantum_25q_build exists", hasattr(code_engine, 'quantum_25q_build'))
    check("CE quantum_ai_transformer exists", hasattr(code_engine, 'quantum_ai_transformer'))
    check("CE quantum_causal_reason exists", hasattr(code_engine, 'quantum_causal_reason'))
    check("CE quantum_grover_nerve exists", hasattr(code_engine, 'quantum_grover_nerve'))
    check("CE quantum_full_circuit_status exists", hasattr(code_engine, 'quantum_full_circuit_status'))

    status = code_engine.quantum_full_circuit_status()
    check("CE status returns dict", isinstance(status, dict), str(type(status)))
    check("CE status has version", 'version' in status, str(status.keys()))
except Exception as e:
    check("CodeEngine import", False, str(e))

# ═══════════════════════════════════════════════
# PHASE 4: ASI CORE (expanded fleet)
# ═══════════════════════════════════════════════
print("\n" + "═" * 60)
print("PHASE 4: ASI CORE — Expanded Quantum Circuit Fleet")
print("═" * 60)
try:
    from l104_asi import asi_core
    check("ASI Core import", True)

    # Original v10.0 methods
    check("ASI quantum_grover_search exists", hasattr(asi_core, 'quantum_grover_search'))
    check("ASI quantum_25q_execute exists", hasattr(asi_core, 'quantum_25q_execute'))
    check("ASI quantum_shor_factor exists", hasattr(asi_core, 'quantum_shor_factor'))
    check("ASI quantum_topological_compute exists", hasattr(asi_core, 'quantum_topological_compute'))

    # Expanded v10.1 methods
    check("ASI get_gravity_engine exists", hasattr(asi_core, 'get_gravity_engine'))
    check("ASI get_consciousness_calc exists", hasattr(asi_core, 'get_consciousness_calc'))
    check("ASI get_ai_architectures exists", hasattr(asi_core, 'get_ai_architectures'))
    check("ASI get_mining_engine exists", hasattr(asi_core, 'get_mining_engine'))
    check("ASI get_data_storage exists", hasattr(asi_core, 'get_data_storage'))
    check("ASI get_reasoning_engine exists", hasattr(asi_core, 'get_reasoning_engine'))
    check("ASI quantum_gravity_erepr exists", hasattr(asi_core, 'quantum_gravity_erepr'))
    check("ASI quantum_consciousness_phi exists", hasattr(asi_core, 'quantum_consciousness_phi'))
    check("ASI quantum_ai_transformer exists", hasattr(asi_core, 'quantum_ai_transformer'))
    check("ASI quantum_causal_reason exists", hasattr(asi_core, 'quantum_causal_reason'))
    check("ASI quantum_mining_solve exists", hasattr(asi_core, 'quantum_mining_solve'))
    check("ASI quantum_store_state exists", hasattr(asi_core, 'quantum_store_state'))
    check("ASI quantum_reason exists", hasattr(asi_core, 'quantum_reason'))

    status = asi_core.quantum_circuit_status()
    check("ASI status returns dict", isinstance(status, dict))
    check("ASI status version 10.1", status.get('version', '').startswith('10.1'), status.get('version'))
except Exception as e:
    check("ASI Core import", False, str(e))

# ═══════════════════════════════════════════════
# PHASE 5: AGI CORE (expanded fleet)
# ═══════════════════════════════════════════════
print("\n" + "═" * 60)
print("PHASE 5: AGI CORE — Expanded Quantum Circuit Fleet")
print("═" * 60)
try:
    from l104_agi import agi_core
    check("AGI Core import", True)

    # Original v58.2 methods
    check("AGI quantum_grover_search exists", hasattr(agi_core, 'quantum_grover_search'))
    check("AGI quantum_25q_execute exists", hasattr(agi_core, 'quantum_25q_execute'))
    check("AGI quantum_shor_factor exists", hasattr(agi_core, 'quantum_shor_factor'))
    check("AGI quantum_topological_compute exists", hasattr(agi_core, 'quantum_topological_compute'))
    check("AGI quantum_grover_nerve_search exists", hasattr(agi_core, 'quantum_grover_nerve_search'))

    # Expanded v58.3 methods
    check("AGI get_gravity_engine exists", hasattr(agi_core, 'get_gravity_engine'))
    check("AGI get_consciousness_calc exists", hasattr(agi_core, 'get_consciousness_calc'))
    check("AGI get_ai_architectures exists", hasattr(agi_core, 'get_ai_architectures'))
    check("AGI get_mining_engine exists", hasattr(agi_core, 'get_mining_engine'))
    check("AGI get_data_storage exists", hasattr(agi_core, 'get_data_storage'))
    check("AGI get_reasoning_engine exists", hasattr(agi_core, 'get_reasoning_engine'))
    check("AGI quantum_gravity_erepr exists", hasattr(agi_core, 'quantum_gravity_erepr'))
    check("AGI quantum_consciousness_phi exists", hasattr(agi_core, 'quantum_consciousness_phi'))
    check("AGI quantum_ai_transformer exists", hasattr(agi_core, 'quantum_ai_transformer'))
    check("AGI quantum_mining_solve exists", hasattr(agi_core, 'quantum_mining_solve'))
    check("AGI quantum_reason exists", hasattr(agi_core, 'quantum_reason'))

    status = agi_core.quantum_circuit_status()
    check("AGI status returns dict", isinstance(status, dict))
    check("AGI status version 58.3", status.get('version', '').startswith('58.3'), status.get('version'))

    # Test eager connect
    connect_result = agi_core.quantum_connect_all_circuits()
    check("AGI connect_all returns dict", isinstance(connect_result, dict))
    check("AGI connect_all has total", 'total_connected' in connect_result, str(connect_result.keys()))
except Exception as e:
    check("AGI Core import", False, str(e))

# ═══════════════════════════════════════════════
# PHASE 6: INTELLECT
# ═══════════════════════════════════════════════
print("\n" + "═" * 60)
print("PHASE 6: INTELLECT — Quantum Circuit Integration")
print("═" * 60)
try:
    from l104_intellect import local_intellect
    check("Intellect import", True)

    check("INT quantum_grover_search exists", hasattr(local_intellect, 'quantum_grover_search'))
    check("INT quantum_25q_build exists", hasattr(local_intellect, 'quantum_25q_build'))
    check("INT quantum_gravity_erepr exists", hasattr(local_intellect, 'quantum_gravity_erepr'))
    check("INT quantum_consciousness_phi exists", hasattr(local_intellect, 'quantum_consciousness_phi'))
    check("INT quantum_reason exists", hasattr(local_intellect, 'quantum_reason'))
    check("INT quantum_circuit_status exists", hasattr(local_intellect, 'quantum_circuit_status'))

    status = local_intellect.quantum_circuit_status()
    check("INT status returns dict", isinstance(status, dict))
    check("INT status has version", 'version' in status, str(status.keys()))
except Exception as e:
    check("Intellect import", False, str(e))

# ═══════════════════════════════════════════════
# PHASE 7: LIVE QUANTUM EXECUTION (via CoherenceEngine)
# ═══════════════════════════════════════════════
print("\n" + "═" * 60)
print("PHASE 7: LIVE QUANTUM EXECUTION")
print("═" * 60)
try:
    from l104_quantum_coherence import QuantumCoherenceEngine
    qce = QuantumCoherenceEngine()
    check("QuantumCoherenceEngine import", True)

    # Grover search
    result = qce.grover_search(target_index=3, search_space_qubits=3)
    check("Grover search returns dict", isinstance(result, dict))
    check("Grover search has quantum key", 'quantum' in result or 'found_target' in result
          or 'algorithm' in result,
          str(list(result.keys())[:5]))

    # Status
    status = qce.get_status()
    check("CoherenceEngine status", isinstance(status, dict))
except Exception as e:
    check("QuantumCoherenceEngine import", False, str(e))

# ═══════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════
print("\n" + "═" * 60)
total = passed + failed
print(f"ALL-ENGINE QUANTUM CIRCUIT INTEGRATION: {passed}/{total} passed")
if errors:
    print(f"\nFailed ({len(errors)}):")
    for e in errors:
        print(f"  • {e}")
print("═" * 60)

sys.exit(0 if failed == 0 else 1)
