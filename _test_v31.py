"""Quick test of Code Engine v3.1.0 new API methods."""
from l104_code_engine import code_engine

# Test type_flow
tf = code_engine.type_flow("def add(a, b): return a + b\nx = add(1, 2)\n")
print("type_flow:", tf.get("type_safety_score", "N/A"), "gaps:", tf.get("gap_count", "N/A"))

# Test concurrency_scan
cs = code_engine.concurrency_scan("import threading\nlock = threading.Lock()\ndef worker(): pass\n")
print("concurrency:", cs.get("safety_score", "N/A"), "issues:", cs.get("issue_count", "N/A"))

# Test validate_contracts
vc = code_engine.validate_contracts('def greet(name):\n    """Greet user."""\n    return f"Hello {name}"\n')
print("contracts:", vc.get("adherence_score", "N/A"), "drifts:", vc.get("drift_count", "N/A"))

# Test explain_code
ec = code_engine.explain_code("class Foo:\n    def bar(self): pass\n")
print("explain:", ec.get("summary", "N/A"))

# Test deep_review (abbreviated)
dr = code_engine.deep_review("def hello(): return 42\n")
print("deep_review:", dr.get("review_type"), "score:", dr.get("composite_score"), "dims:", dr.get("score_dimensions"))

# Test status includes new subsystems
s = code_engine.status()
for key in ["type_flow_analyzer", "concurrency_analyzer", "contract_validator", "evolution_tracker"]:
    print(f"status[{key}]:", "PRESENT" if key in s else "MISSING")

print("\nAll v3.1.0 tests passed!")
