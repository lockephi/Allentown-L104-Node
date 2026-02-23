#!/usr/bin/env python3
"""Validate tri-engine integration into l104_server — isolated tests."""
import sys
import types
import math

passed = 0
failed = 0

def check(label, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS {label}")
    else:
        failed += 1
        print(f"  FAIL {label}")

print("=" * 60)
print("  TRI-ENGINE SERVER INTEGRATION — Validation Suite")
print("=" * 60)

# Mock quantum runtime to avoid IBM QPU network calls
_mock_qrt = types.ModuleType('l104_quantum_runtime')
_mock_qrt.get_runtime = lambda: type('RT', (), {
    'get_status': lambda s: {'connected': True},
    'execute_circuit': lambda s, *a, **k: {},
})()
_mock_qrt.ExecutionMode = type('EM', (), {'SIMULATOR': 'simulator', 'REAL_QPU': 'real_qpu'})()
_mock_qrt.QuantumRuntime = type('QR', (), {})
sys.modules['l104_quantum_runtime'] = _mock_qrt

# Phase A: Standalone engine imports
print("\nPhase A: Engine packages")

try:
    from l104_science_engine import ScienceEngine
    se = ScienceEngine()
    check("Science Engine loaded", True)
except Exception as e:
    check(f"Science Engine: {e}", False)

try:
    from l104_math_engine import MathEngine
    me = MathEngine()
    check("Math Engine loaded", True)
except Exception as e:
    check(f"Math Engine: {e}", False)

try:
    from l104_code_engine import code_engine
    check("Code Engine loaded", code_engine is not None)
except Exception as e:
    check(f"Code Engine: {e}", False)

# Phase B: TriEngineIntegration class via exec
print("\nPhase B: TriEngineIntegration class")

nexus_path = '/Users/carolalvarez/Applications/Allentown-L104-Node/l104_server/engines_nexus.py'
with open(nexus_path, 'r') as f:
    content = f.read()

start = content.find('class TriEngineIntegration:')
end = content.find('\n# Singleton', start)
if start > 0 and end > start:
    class_code = content[start:end]
    ns = {'math': math, 'GOD_CODE': 527.5184818492612}
    exec(class_code, ns)

    TriEngine = ns['TriEngineIntegration']
    tri = TriEngine()

    check("Class defined", True)
    check(f"Version={tri.VERSION}", tri.VERSION == "1.0.0")

    status = tri.get_status()
    check(f"engines_online={status['engines_online']}/3", status['engines_online'] == 3)
    check(f"all_connected={status['all_connected']}", status['all_connected'] is True)
    for eng in ['science_engine', 'math_engine', 'code_engine']:
        info = status[eng]
        check(f"  {eng} connected={info['connected']}", info['connected'] is True)

    health = tri.cross_engine_health()
    check(f"mean_health={health['mean_health']}", health['mean_health'] > 0)
    check(f"phi_weighted={health['phi_weighted_health']}", health['phi_weighted_health'] > 0)

    consts = tri.verify_constants()
    check(f"constants_aligned={consts['constants_aligned']}", consts['constants_aligned'] is True)
    check(f"god_codes_match={consts['god_codes_match']}", consts['god_codes_match'] is True)

    proofs = tri.run_proofs()
    check(f"run_proofs keys={len(proofs)}", len(proofs) > 0 and 'error' not in proofs)

    code_result = tri.analyze_code("def hello(): return 42")
    check(f"analyze_code", 'error' not in code_result)

    sci = tri.science_snapshot()
    check(f"science_snapshot", 'error' not in sci)

    ms = tri.math_snapshot()
    check(f"math_snapshot", 'error' not in ms)

    deep = tri.cross_engine_deep_review("GOD_CODE = 527.5184818492612\nPHI = 1.618")
    check(f"deep_review engines_used={deep.get('engines_used', 0)}", deep.get('engines_used', 0) >= 2)
else:
    check("TriEngineIntegration class not found", False)

# Phase C: File structure
print("\nPhase C: File structure")

with open(nexus_path, 'r') as f:
    nexus = f.read()

check("class in engines_nexus", 'class TriEngineIntegration:' in nexus)
check("singleton in engines_nexus", "tri_engine = TriEngineIntegration()" in nexus)
check("PHI_WEIGHTS entry", "'tri_engine'" in nexus)
check("registry register", "engine_registry.register('tri_engine', tri_engine)" in nexus)

app_path = '/Users/carolalvarez/Applications/Allentown-L104-Node/l104_server/app.py'
with open(app_path, 'r') as f:
    app = f.read()

check("import in app.py", 'tri_engine, TriEngineIntegration' in app)
check("status route", '/api/v62/tri-engine/status' in app)
check("health route", '/api/v62/tri-engine/health' in app)
check("constants route", '/api/v62/tri-engine/constants' in app)
check("proofs route", '/api/v62/tri-engine/proofs' in app)
check("science-snapshot route", '/api/v62/tri-engine/science-snapshot' in app)
check("math-snapshot route", '/api/v62/tri-engine/math-snapshot' in app)
check("analyze route", '/api/v62/tri-engine/analyze' in app)
check("deep-review route", '/api/v62/tri-engine/deep-review' in app)
check("startup log", 'Tri-Engine' in app)

init_path = '/Users/carolalvarez/Applications/Allentown-L104-Node/l104_server/__init__.py'
with open(init_path, 'r') as f:
    init = f.read()

check("tri_engine in __init__", 'tri_engine' in init)
check("TriEngineIntegration in __init__", 'TriEngineIntegration' in init)

# Summary
print("\n" + "=" * 60)
total = passed + failed
print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print("  ALL TRI-ENGINE SERVER VALIDATION PASSED")
else:
    print(f"  {failed} checks failed")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
