#!/usr/bin/env python3
"""Verify full quantum circuit fleet wiring across all 6 engines."""
import os, sys
os.environ['IBMQ_TOKEN'] = ''
os.environ['IBM_QUANTUM_TOKEN'] = ''
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

results = {}
total = 0
passed = 0

def check(name, fn):
    global total, passed
    total += 1
    try:
        r = fn()
        passed += 1
        results[name] = 'PASS'
        return r
    except Exception as e:
        results[name] = f'FAIL: {e}'
        return None

# 1. ASI CORE v10.2
print('=== ASI CORE ===')
from l104_asi import asi_core
check('ASI import', lambda: asi_core)
s = check('ASI quantum_circuit_status', lambda: asi_core.quantum_circuit_status())
if s: print(f'  version={s["version"]}, modules={s.get("modules_connected","?")}')
check('ASI get_quantum_runtime', lambda: asi_core.get_quantum_runtime)
check('ASI get_quantum_accelerator', lambda: asi_core.get_quantum_accelerator)
check('ASI get_quantum_inspired', lambda: asi_core.get_quantum_inspired)
check('ASI get_consciousness_bridge', lambda: asi_core.get_consciousness_bridge)
check('ASI get_numerical_builder', lambda: asi_core.get_numerical_builder)
check('ASI get_quantum_magic', lambda: asi_core.get_quantum_magic)
check('ASI quantum_accelerator_compute', lambda: asi_core.quantum_accelerator_compute())
check('ASI quantum_inspired_optimize', lambda: asi_core.quantum_inspired_optimize())
check('ASI quantum_consciousness_bridge_decide', lambda: asi_core.quantum_consciousness_bridge_decide())
check('ASI quantum_numerical_compute', lambda: asi_core.quantum_numerical_compute())
check('ASI quantum_magic_infer', lambda: asi_core.quantum_magic_infer())

# 2. AGI CORE v58.5
print('=== AGI CORE ===')
from l104_agi import agi_core
check('AGI import', lambda: agi_core)
s = check('AGI quantum_circuit_status', lambda: agi_core.quantum_circuit_status())
if s: print(f'  version={s["version"]}, modules={s.get("modules_connected","?")}')
check('AGI get_quantum_accelerator', lambda: agi_core.get_quantum_accelerator)
check('AGI get_quantum_inspired', lambda: agi_core.get_quantum_inspired)
check('AGI get_consciousness_bridge', lambda: agi_core.get_consciousness_bridge)
check('AGI get_numerical_builder', lambda: agi_core.get_numerical_builder)
check('AGI get_quantum_magic', lambda: agi_core.get_quantum_magic)
check('AGI quantum_accelerator_compute', lambda: agi_core.quantum_accelerator_compute())
check('AGI quantum_inspired_optimize', lambda: agi_core.quantum_inspired_optimize())
check('AGI quantum_consciousness_bridge_decide', lambda: agi_core.quantum_consciousness_bridge_decide())

# 3. SCIENCE ENGINE v4.2
print('=== SCIENCE ENGINE ===')
from l104_science_engine import ScienceEngine
se = ScienceEngine()
check('SCI import', lambda: se)
s = check('SCI quantum_circuit_status', lambda: se.quantum_circuit_status())
if s: print(f'  version={s["version"]}, modules={s.get("modules_connected","?")}')
check('SCI quantum_accelerator_entangle', lambda: se.quantum_accelerator_entangle())
check('SCI quantum_inspired_optimize', lambda: se.quantum_inspired_optimize())
check('SCI quantum_reason', lambda: se.quantum_reason())
check('SCI quantum_grover_nerve_search', lambda: se.quantum_grover_nerve_search())

# 4. MATH ENGINE v1.2
print('=== MATH ENGINE ===')
from l104_math_engine import MathEngine
me = MathEngine()
check('MATH import', lambda: me)
s = check('MATH quantum_circuit_status', lambda: me.quantum_circuit_status())
if s: print(f'  version={s["version"]}, modules={s.get("modules_connected","?")}')
check('MATH quantum_accelerator_compute', lambda: me.quantum_accelerator_compute())
check('MATH quantum_inspired_anneal', lambda: me.quantum_inspired_anneal())
check('MATH quantum_reason', lambda: me.quantum_reason())
check('MATH quantum_consciousness_phi', lambda: me.quantum_consciousness_phi())
check('MATH quantum_grover_nerve_search', lambda: me.quantum_grover_nerve_search())

# 5. CODE ENGINE v6.4
print('=== CODE ENGINE ===')
from l104_code_engine import code_engine
check('CODE import', lambda: code_engine)
s = check('CODE quantum_full_circuit_status', lambda: code_engine.quantum_full_circuit_status())
if s: print(f'  version={s["version"]}, modules={s.get("modules_connected","?")}')
check('CODE quantum_accelerator_entangle', lambda: code_engine.quantum_accelerator_entangle())
check('CODE quantum_inspired_optimize', lambda: code_engine.quantum_inspired_optimize())
check('CODE quantum_reason', lambda: code_engine.quantum_reason())
check('CODE quantum_gravity_compute', lambda: code_engine.quantum_gravity_compute())
check('CODE quantum_consciousness_phi', lambda: code_engine.quantum_consciousness_phi())
check('CODE quantum_numerical_compute', lambda: code_engine.quantum_numerical_compute())

# 6. INTELLECT v27.2
print('=== INTELLECT ===')
from l104_intellect import local_intellect
check('INT import', lambda: local_intellect)
s = check('INT quantum_circuit_status', lambda: local_intellect.quantum_circuit_status())
if s: print(f'  version={s.get("version","?")}, modules={s.get("modules_connected","?")}')
check('INT get_quantum_accelerator', lambda: local_intellect.get_quantum_accelerator)
check('INT get_quantum_inspired', lambda: local_intellect.get_quantum_inspired)
check('INT get_quantum_numerical', lambda: local_intellect.get_quantum_numerical)
check('INT get_quantum_magic', lambda: local_intellect.get_quantum_magic)
check('INT get_quantum_runtime', lambda: local_intellect.get_quantum_runtime)
check('INT quantum_accelerator_compute', lambda: local_intellect.quantum_accelerator_compute())
check('INT quantum_inspired_optimize', lambda: local_intellect.quantum_inspired_optimize())
check('INT quantum_numerical_compute', lambda: local_intellect.quantum_numerical_compute())
check('INT quantum_magic_infer', lambda: local_intellect.quantum_magic_infer())

# SUMMARY
print(f'\n=== RESULTS: {passed}/{total} PASSED ===')
fails = {k: v for k, v in results.items() if 'FAIL' in str(v)}
if fails:
    print('FAILURES:')
    for k, v in fails.items():
        print(f'  {k}: {v}')
else:
    print('ALL CHECKS PASSED')
