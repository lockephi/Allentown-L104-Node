"""L104 Phase 2: Full Engine Wiring Verification"""
import json

print('=' * 60)
print('  L104 PHASE 2: FULL ENGINE WIRING VERIFICATION')
print('=' * 60)

passed = 0
failed = 0

# TEST 1: LocalIntellect expanded KB
print('\n> TEST 1: LocalIntellect KB expansion')
from l104_intellect import local_intellect
local_intellect._ensure_quantum_origin_sage()
status = local_intellect.quantum_origin_sage_status()
nkf = status.get('native_kernel_fleet', {})
print(f'  c_kernel={nkf.get("c_kernel")}')
print(f'  asm_kernel={nkf.get("asm_kernel")}')
print(f'  cuda_kernel={nkf.get("cuda_kernel")}')
print(f'  rust_kernel={nkf.get("rust_kernel")}')
kb_injected = nkf.get("kb_entries_injected", 0)
print(f'  kb_entries_injected={kb_injected}')
engine_cats = set(
    e.get('category', '')
    for e in local_intellect.training_data
    if e.get('source', '').endswith('_kb_training')
)
print(f'  engine_kb_categories={sorted(engine_cats)}')
if kb_injected > 14:
    print('  PASSED')
    passed += 1
else:
    print('  FAILED')
    failed += 1

# TEST 2: Quantum Engine brain.py
print('\n> TEST 2: Quantum Engine wiring')
from l104_quantum_engine.brain import L104QuantumBrain
brain = L104QuantumBrain()
li = brain._get_local_intellect()
print(f'  local_intellect={li is not None}')
orch = brain._get_sage_orchestrator()
print(f'  sage_orchestrator={orch is not None}')
brain._feed_intellect_kb()
print(f'  intellect_kb_fed={brain._intellect_kb_fed}')
ks = brain.kernel_status()
print(f'  kernel_available={ks.get("available", False)}')
if li is not None:
    print('  PASSED')
    passed += 1
else:
    print('  FAILED')
    failed += 1

# TEST 3: Quantum Gate Engine
print('\n> TEST 3: Quantum Gate Engine wiring')
from l104_quantum_gate_engine import get_engine
engine = get_engine()
li2 = engine.local_intellect
print(f'  local_intellect={li2 is not None}')
orch2 = engine.sage_orchestrator
print(f'  sage_orchestrator={orch2 is not None}')
engine.feed_intellect_kb()
print(f'  intellect_kb_fed={engine._intellect_kb_fed}')
ks2 = engine.kernel_status()
print(f'  kernel_available={ks2.get("available", False)}')
if li2 is not None:
    print('  PASSED')
    passed += 1
else:
    print('  FAILED')
    failed += 1

# TEST 4: Dual-Layer Engine
print('\n> TEST 4: Dual-Layer Engine wiring')
from l104_asi.dual_layer import DualLayerEngine
dle = DualLayerEngine()
orch3 = dle._get_sage_orchestrator()
print(f'  sage_orchestrator={orch3 is not None}')
li3 = dle._get_local_intellect()
print(f'  local_intellect={li3 is not None}')
dle._feed_intellect_kb()
print(f'  intellect_kb_fed={dle._intellect_kb_fed}')
ks3 = dle.kernel_status()
print(f'  kernel_available={ks3.get("available", False)}')
integrity = dle.full_integrity_check(force=True)
engine_label = integrity.get('engine', '')
print(f'  integrity_engine={engine_label[:65]}')
has_kernel = 'kernel bridge' in engine_label or 'GOD_CODE' in engine_label
if has_kernel:
    print('  PASSED')
    passed += 1
else:
    print('  FAILED')
    failed += 1

# TEST 5: ASI Core KB write-back
print('\n> TEST 5: ASI Core KB write-back')
from l104_asi import asi_core
wb = asi_core.intellect_write_back()
print(f'  entries_written={wb.get("entries_written", 0)}')
print(f'  total_training_data={wb.get("total_training_data", 0)}')
if wb.get('entries_written', 0) > 0:
    print('  PASSED')
    passed += 1
else:
    print('  FAILED')
    failed += 1

# TEST 6: AGI constants import fix
print('\n> TEST 6: AGI constants import fix')
try:
    from l104_agi.constants import format_iq
    result = format_iq(527.5184818492612)
    print(f'  format_iq(GOD_CODE)={result}')
    print('  PASSED')
    passed += 1
except Exception as e:
    print(f'  FAILED: {e}')
    failed += 1

# SUMMARY
print('\n' + '=' * 60)
print('  PHASE 2 WIRING SUMMARY')
print('=' * 60)
sources = {}
for e in local_intellect.training_data:
    src = e.get('source', 'unknown')
    sources[src] = sources.get(src, 0) + 1
kb_sources = {k: v for k, v in sources.items() if 'kb' in k.lower()}
print(f'  Total KB training entries from wiring: {sum(kb_sources.values())}')
for src, count in sorted(kb_sources.items()):
    print(f'    {src}: {count}')
print(f'  Tests: {passed}/{passed + failed} passed')
print('=' * 60)
