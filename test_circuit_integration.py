#!/usr/bin/env python3
"""
v10.0 / v58.2 Full Circuit Integration — End-to-End Verification
Tests that ALL quantum circuit modules are wired into ASI and AGI cores.
"""
import os, sys, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Force simulator mode — avoid real QPU calls during integration testing
_saved_token = os.environ.pop("IBMQ_TOKEN", None)
_saved_token2 = os.environ.pop("IBM_QUANTUM_TOKEN", None)

PASS = 0
FAIL = 0

def check(label, condition):
    global PASS, FAIL
    status = "✅ PASS" if condition else "❌ FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {status}  {label}")

print("=" * 72)
print("  FULL CIRCUIT INTEGRATION — E2E VERIFICATION")
print("=" * 72)

# ── Phase 1: ASI Core ──────────────────────────────────────────────────
print("\n── Phase 1: ASI Core (l104_asi/core.py) ──")
from l104_asi import asi_core
check("ASI Core imported", asi_core is not None)
check("ASI has _coherence_engine attr", hasattr(asi_core, '_coherence_engine'))
check("ASI has _builder_25q attr", hasattr(asi_core, '_builder_25q'))
check("ASI has _grover_nerve attr", hasattr(asi_core, '_grover_nerve'))
check("ASI has _quantum_computation_pipeline attr", hasattr(asi_core, '_quantum_computation_pipeline'))
check("ASI has quantum_circuit_status()", hasattr(asi_core, 'quantum_circuit_status'))
check("ASI has quantum_grover_search()", hasattr(asi_core, 'quantum_grover_search'))
check("ASI has quantum_25q_execute()", hasattr(asi_core, 'quantum_25q_execute'))
check("ASI has quantum_shor_factor()", hasattr(asi_core, 'quantum_shor_factor'))
check("ASI has quantum_topological_compute()", hasattr(asi_core, 'quantum_topological_compute'))

# Lazy connect
print("\n  Lazy-loading quantum modules into ASI core...")
ce = asi_core.get_coherence_engine()
check("ASI CoherenceEngine loaded", ce is not None)
b25 = asi_core.get_builder_25q()
check("ASI 25Q Builder loaded", b25 is not None)
gn = asi_core.get_grover_nerve()
check("ASI GroverNerve loaded", gn is not None)

# Status check
status = asi_core.quantum_circuit_status()
check("ASI quantum_circuit_status() returns dict", isinstance(status, dict))
check("ASI reports version", 'version' in status)
print(f"    → modules_connected = {status.get('modules_connected', 0)}")

# ── Phase 2: ASI QuantumComputationCore ────────────────────────────────
print("\n── Phase 2: ASI QuantumComputationCore (l104_asi/quantum.py) ──")
from l104_asi.quantum import QuantumComputationCore
qcc = QuantumComputationCore()
check("QuantumComputationCore instantiated", qcc is not None)
check("QCC has coherence_grover_search()", hasattr(qcc, 'coherence_grover_search'))
check("QCC has build_25q_full_circuit()", hasattr(qcc, 'build_25q_full_circuit'))
check("QCC has build_25q_grover()", hasattr(qcc, 'build_25q_grover'))
check("QCC has build_25q_qft()", hasattr(qcc, 'build_25q_qft'))
check("QCC has build_25q_vqe()", hasattr(qcc, 'build_25q_vqe'))
check("QCC has build_25q_topological_braiding()", hasattr(qcc, 'build_25q_topological_braiding'))
check("QCC has build_25q_iron_simulator()", hasattr(qcc, 'build_25q_iron_simulator'))
check("QCC has execute_25q_circuit()", hasattr(qcc, 'execute_25q_circuit'))
check("QCC has grover_nerve_search()", hasattr(qcc, 'grover_nerve_search'))
check("QCC has full_circuit_status()", hasattr(qcc, 'full_circuit_status'))
fcs = qcc.full_circuit_status()
check("QCC full_circuit_status() returns dict", isinstance(fcs, dict))
print(f"    → circuit modules = {fcs}")

# ── Phase 3: AGI Core ──────────────────────────────────────────────────
print("\n── Phase 3: AGI Core (l104_agi/core.py) ──")

# Patch quantum_runtime to avoid real QPU calls during AGI init
import l104_quantum_runtime as _qrt
_rt = getattr(_qrt, '_runtime_singleton', None)
if _rt is None and hasattr(_qrt, 'get_runtime'):
    try:
        _rt = _qrt.get_runtime()
    except Exception:
        pass
if _rt and hasattr(_rt, '_connected'):
    _saved_connected = _rt._connected
    _rt._connected = False
    print("  [patched quantum_runtime to simulator mode for safe AGI init]")

from l104_agi import agi_core

# Restore runtime
if _rt and hasattr(_rt, '_connected'):
    _rt._connected = _saved_connected
check("AGI Core imported", agi_core is not None)
check("AGI has _coherence_engine attr", hasattr(agi_core, '_coherence_engine'))
check("AGI has _builder_25q attr", hasattr(agi_core, '_builder_25q'))
check("AGI has _grover_nerve attr", hasattr(agi_core, '_grover_nerve'))
check("AGI has _quantum_computation_pipeline attr", hasattr(agi_core, '_quantum_computation_pipeline'))
check("AGI has quantum_circuit_status()", hasattr(agi_core, 'quantum_circuit_status'))
check("AGI has quantum_connect_all_circuits()", hasattr(agi_core, 'quantum_connect_all_circuits'))
check("AGI has quantum_grover_search()", hasattr(agi_core, 'quantum_grover_search'))
check("AGI has quantum_25q_execute()", hasattr(agi_core, 'quantum_25q_execute'))
check("AGI has quantum_shor_factor()", hasattr(agi_core, 'quantum_shor_factor'))
check("AGI has quantum_topological_compute()", hasattr(agi_core, 'quantum_topological_compute'))
check("AGI has quantum_grover_nerve_search()", hasattr(agi_core, 'quantum_grover_nerve_search'))

# Eager connect all
print("\n  Eager-connecting all quantum modules into AGI core...")
conn = agi_core.quantum_connect_all_circuits()
check("AGI quantum_connect_all_circuits() returns dict", isinstance(conn, dict))
check("AGI CoherenceEngine connected", conn.get('coherence_engine', False))
check("AGI 25Q Builder connected", conn.get('builder_25q', False))
check("AGI GroverNerve connected", conn.get('grover_nerve', False))
print(f"    → total connected = {conn.get('total_connected', 0)}")

# AGI status
agi_status = agi_core.quantum_circuit_status()
check("AGI quantum_circuit_status() returns dict", isinstance(agi_status, dict))
print(f"    → version = {agi_status.get('version')}")
print(f"    → modules_connected = {agi_status.get('modules_connected', 0)}")

# ── Phase 4: Standalone Modules Exist ──────────────────────────────────
print("\n── Phase 4: Standalone Module Imports ──")
try:
    from l104_quantum_coherence import QuantumCoherenceEngine
    check("l104_quantum_coherence importable", True)
except Exception:
    check("l104_quantum_coherence importable", False)
try:
    from l104_25q_engine_builder import L104_25Q_CircuitBuilder
    check("l104_25q_engine_builder importable", True)
except Exception:
    check("l104_25q_engine_builder importable", False)
try:
    from l104_grover_nerve_link import get_grover_nerve
    check("l104_grover_nerve_link importable", True)
except Exception:
    check("l104_grover_nerve_link importable", False)
try:
    from l104_quantum_computation_pipeline import QuantumNeuralNetwork, VariationalQuantumClassifier
    check("l104_quantum_computation_pipeline importable", True)
except Exception:
    check("l104_quantum_computation_pipeline importable", False)
try:
    from l104_quantum_runtime import get_runtime
    check("l104_quantum_runtime importable", True)
except Exception:
    check("l104_quantum_runtime importable", False)
try:
    from l104_qiskit_utils import L104QiskitUtils
    check("l104_qiskit_utils importable", True)
except Exception:
    check("l104_qiskit_utils importable", False)

# ── Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
total = PASS + FAIL
print(f"  RESULT: {PASS}/{total} passed  ({FAIL} failed)")
if FAIL == 0:
    print("  ✅ ALL QUANTUM CIRCUITS CONNECTED TO ASI + AGI CORES")
else:
    print(f"  ⚠️  {FAIL} check(s) need attention")
print("=" * 72)
