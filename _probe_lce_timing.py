#!/usr/bin/env python3
"""Timing probe: find which engine loader blocks during LCE initialization."""
import sys, time
sys.path.insert(0, ".")

def timed_load(name, loader):
    t0 = time.time()
    sys.stdout.write(f"  Loading {name:25s} ... ")
    sys.stdout.flush()
    try:
        obj = loader()
        dt = (time.time() - t0) * 1000
        tname = type(obj).__name__ if obj is not None else "None"
        print(f"OK ({dt:.0f}ms) -> {tname}")
        return obj
    except Exception as e:
        dt = (time.time() - t0) * 1000
        print(f"FAIL ({dt:.0f}ms): {e}")
        return None

print("=" * 60)
print("  ENGINE LOADER TIMING PROBE")
print("=" * 60)

timed_load("ScienceEngine", lambda: __import__("l104_science_engine").ScienceEngine())
timed_load("MathEngine", lambda: __import__("l104_math_engine").MathEngine())
timed_load("code_engine", lambda: __import__("l104_code_engine").code_engine)
timed_load("quantum_gate", lambda: __import__("l104_quantum_gate_engine").get_engine())
timed_load("QuantumMathCore", lambda: __import__("l104_quantum_engine").QuantumMathCore)

print("\n  --- ASI sub-modules ---")
timed_load("dual_layer", lambda: getattr(__import__("l104_asi.dual_layer", fromlist=["dual_layer_engine"]), "dual_layer_engine"))
timed_load("FormalLogicEngine", lambda: __import__("l104_asi.formal_logic", fromlist=["FormalLogicEngine"]).FormalLogicEngine())
timed_load("DeepNLUEngine", lambda: __import__("l104_asi.deep_nlu", fromlist=["DeepNLUEngine"]).DeepNLUEngine())

print("\n  --- Intellect ---")
timed_load("local_intellect", lambda: __import__("l104_intellect", fromlist=["local_intellect"]).local_intellect)

print("\n  --- KB + LSA ---")
t0 = time.time()
sys.stdout.write("  Loading KB + initialize()    ... ")
sys.stdout.flush()
try:
    from l104_asi.language_comprehension import MMLUKnowledgeBase
    kb = MMLUKnowledgeBase()
    kb.initialize()
    dt = (time.time() - t0) * 1000
    st = kb.get_status()
    print(f"OK ({dt:.0f}ms) nodes={st.get('total_nodes')}, facts={st.get('total_facts')}")
except Exception as e:
    dt = (time.time() - t0) * 1000
    print(f"FAIL ({dt:.0f}ms): {e}")

# LSA fitting test
t0 = time.time()
sys.stdout.write("  LSA fit on KB facts          ... ")
sys.stdout.flush()
try:
    from l104_asi.language_comprehension import LatentSemanticAnalyzer
    lsa = LatentSemanticAnalyzer(n_components=50)
    all_facts = []
    for node in kb.nodes.values():
        all_facts.append(node.definition)
        all_facts.extend(node.facts)
    lsa.fit(all_facts)
    dt = (time.time() - t0) * 1000
    print(f"OK ({dt:.0f}ms) docs={len(all_facts)}, fitted={lsa._fitted}")
except Exception as e:
    dt = (time.time() - t0) * 1000
    print(f"FAIL ({dt:.0f}ms): {e}")

print("\n  --- Full LCE init ---")
t0 = time.time()
sys.stdout.write("  LanguageComprehensionEngine  ... ")
sys.stdout.flush()
try:
    from l104_asi.language_comprehension import LanguageComprehensionEngine
    lce = LanguageComprehensionEngine()
    lce.initialize()
    dt = (time.time() - t0) * 1000
    print(f"OK ({dt:.0f}ms)")
except Exception as e:
    dt = (time.time() - t0) * 1000
    print(f"FAIL ({dt:.0f}ms): {e}")
    import traceback; traceback.print_exc()

print("\nDONE")
