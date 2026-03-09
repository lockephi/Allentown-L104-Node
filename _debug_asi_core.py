"""Debug script for l104_asi/core.py — tests all major ASI core methods."""
import json
import traceback
import sys

def test(name, fn):
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")
    try:
        result = fn()
        print(f"  PASS")
        return result
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return None

print("Importing l104_asi...")
from l104_asi import asi_core, ASICore

# Test 1: Basic status
def t1():
    status = asi_core.get_status()
    print(f"  version: {status.get('version')}")
    print(f"  pipeline_connected: {status.get('pipeline_connected')}")
    print(f"  subsystems: {status.get('subsystems_connected', '?')}")
    print(f"  status: {status.get('status')}")
    return status
test("get_status()", t1)

# Test 2: connect_pipeline
def t2():
    result = asi_core.connect_pipeline()
    print(f"  result type: {type(result).__name__}")
    if isinstance(result, dict):
        print(f"  connected: {result.get('connected', '?')}")
        print(f"  subsystems_connected: {result.get('subsystems_connected', '?')}")
    return result
test("connect_pipeline()", t2)

# Test 3: compute_asi_score
def t3():
    score = asi_core.compute_asi_score()
    print(f"  score type: {type(score).__name__}")
    if isinstance(score, dict):
        print(f"  total: {score.get('total', score.get('asi_score', '?'))}")
        print(f"  dimensions: {score.get('dimensions', score.get('dimension_count', '?'))}")
        # Show first few dimensions
        dims = score.get('dimension_scores', score.get('breakdown', {}))
        if isinstance(dims, dict):
            for k, v in list(dims.items())[:5]:
                print(f"    {k}: {v}")
            if len(dims) > 5:
                print(f"    ... ({len(dims)} total dimensions)")
    elif isinstance(score, (int, float)):
        print(f"  score: {score}")
    return score
test("compute_asi_score()", t3)

# Test 4: three_engine_status
def t4():
    te = asi_core.three_engine_status()
    print(f"  result: {json.dumps(te, indent=2, default=str)[:500]}")
    return te
test("three_engine_status()", t4)

# Test 5: three_engine scoring
def t5():
    for method in ['three_engine_entropy_score', 'three_engine_harmonic_score', 'three_engine_wave_coherence_score']:
        try:
            val = getattr(asi_core, method)()
            print(f"  {method}: {val}")
        except Exception as e:
            print(f"  {method} FAIL: {e}")
            traceback.print_exc()
test("three_engine_scoring", t5)

# Test 6: evolution_stage / evolution_index
def t6():
    print(f"  evolution_stage: {asi_core.evolution_stage}")
    print(f"  evolution_index: {asi_core.evolution_index}")
test("evolution properties", t6)

# Test 7: formal_logic_score / deep_nlu_score
def t7():
    print(f"  formal_logic_score: {asi_core.formal_logic_score}")
    print(f"  deep_nlu_score: {asi_core.deep_nlu_score}")
test("formal_logic_score + deep_nlu_score", t7)

# Test 8: kb_reconstruction_fidelity_score
def t8():
    score = asi_core.kb_reconstruction_fidelity_score
    print(f"  kb_reconstruction_fidelity_score: {score}")
    return score
test("kb_reconstruction_fidelity_score", t8)

# Test 9: analyze_argument
def t9():
    result = asi_core.analyze_argument(["All men are mortal", "Socrates is a man"], "Socrates is mortal")
    print(f"  result type: {type(result).__name__}")
    print(f"  result: {json.dumps(result, indent=2, default=str)[:300]}")
    return result
test("analyze_argument()", t9)

# Test 10: detect_fallacies
def t10():
    result = asi_core.detect_fallacies("You should believe me because I said so")
    print(f"  result type: {type(result).__name__}")
    print(f"  result: {result[:3] if isinstance(result, list) else result}")
    return result
test("detect_fallacies()", t10)

# Test 11: deep_understand
def t11():
    result = asi_core.deep_understand("The quantum computer processed data faster than classical machines")
    print(f"  result type: {type(result).__name__}")
    if isinstance(result, dict):
        for k, v in list(result.items())[:5]:
            print(f"    {k}: {str(v)[:100]}")
    return result
test("deep_understand()", t11)

# Test 12: pipeline_solve
def t12():
    result = asi_core.pipeline_solve("What is 2+2?")
    print(f"  result type: {type(result).__name__}")
    if isinstance(result, dict):
        keys = list(result.keys())[:10]
        print(f"  keys: {keys}")
        sol = result.get('solution', result.get('answer', ''))
        print(f"  solution: {str(sol)[:200]}")
    return result
test("pipeline_solve('What is 2+2?')", t12)

# Test 13: run_full_assessment
def t13():
    result = asi_core.run_full_assessment()
    print(f"  result type: {type(result).__name__}")
    if isinstance(result, dict):
        for k, v in list(result.items())[:8]:
            print(f"    {k}: {str(v)[:100]}")
    return result
test("run_full_assessment()", t13)

# Test 14: dual layer
def t14():
    if asi_core.dual_layer_available:
        dl = asi_core.dual_layer
        print(f"  dual_layer type: {type(dl).__name__}")
        status = dl.status() if hasattr(dl, 'status') else dl.get_status() if hasattr(dl, 'get_status') else "no status method"
        print(f"  status: {str(status)[:300]}")
    else:
        print("  dual_layer NOT AVAILABLE")
test("dual_layer", t14)

# Test 15: intellect channels
def t15():
    for method in ['intellect_think', 'intellect_search']:
        try:
            if method == 'intellect_think':
                val = asi_core.intellect_think("test thought")
            else:
                val = asi_core.intellect_search("test query")
            print(f"  {method}: {str(val)[:200]}")
        except Exception as e:
            print(f"  {method} FAIL: {e}")
            traceback.print_exc()
test("intellect channels", t15)

# Test 16: quantum computation
def t16():
    if asi_core._quantum_computation is None:
        # Try lazy load
        qc = asi_core._quantum_computation
        print(f"  quantum_computation: not yet loaded (None)")
    else:
        print(f"  quantum_computation: {type(asi_core._quantum_computation).__name__}")
    # Check QuantumComputationCore directly
    from l104_asi.quantum import QuantumComputationCore
    qcc = QuantumComputationCore()
    qs = qcc.status()
    print(f"  QuantumComputationCore status: {json.dumps(qs, indent=2, default=str)[:300]}")
test("quantum computation", t16)

print(f"\n{'='*60}")
print("  ALL TESTS COMPLETE")
print(f"{'='*60}")
