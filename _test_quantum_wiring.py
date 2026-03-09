#!/usr/bin/env python3
"""Quick test: verify quantum engine connections across all 4 scoring engines."""
import sys

def main():
    print("=== Testing Quantum Engine Connections ===")
    ok = True

    # Test quantum gate engine
    try:
        from l104_quantum_gate_engine import get_engine, ExecutionTarget
        engine = get_engine()
        status = engine.status()
        gates = status.get("gate_library", {}).get("total_gates", 0)
        print(f"  [OK] Quantum Gate Engine v{status.get('version','?')} - {gates} gates")
        bell = engine.bell_pair()
        result = engine.execute(bell, ExecutionTarget.LOCAL_STATEVECTOR)
        print(f"       Bell pair probs: {result.probabilities}")
        print(f"       Sacred alignment: {result.sacred_alignment}")
    except Exception as e:
        print(f"  [FAIL] Quantum Gate Engine: {e}")
        ok = False

    # Test QuantumMathCore
    try:
        from l104_quantum_engine import QuantumMathCore as QMC
        import math
        state = [complex(0.5), complex(0.5), complex(0.5), complex(0.5)]
        amplified = QMC.grover_operator(state, [2], 1)
        probs = [abs(a)**2 for a in amplified]
        print(f"  [OK] QuantumMathCore - Grover: {[round(p,4) for p in probs]}")
        tp = QMC.tunnel_probability(1.0, 0.5, 1.0)
        print(f"       Tunnel probability: {tp:.6f}")
    except Exception as e:
        print(f"  [FAIL] QuantumMathCore: {e}")
        ok = False

    print()
    print("=== Testing Scoring Engine Quantum Wiring ===")

    # MMLU
    try:
        from l104_asi.language_comprehension import LanguageComprehensionEngine
        lce = LanguageComprehensionEngine()
        lce.initialize()
        es = lce.get_status().get("engine_support", {})
        qg = es.get("quantum_gate_engine", False)
        qm = es.get("quantum_math_core", False)
        print(f"  [{'OK' if qg and qm else 'WARN'}] MMLU  - gate={qg}, math_core={qm}")
        if not (qg and qm):
            ok = False
    except Exception as e:
        print(f"  [FAIL] MMLU: {e}")
        ok = False

    # ARC
    try:
        from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
        cre = CommonsenseReasoningEngine()
        cre.initialize()
        es = cre.get_status().get("engine_support", {})
        qg = es.get("quantum_gate_engine", False)
        qm = es.get("quantum_math_core", False)
        print(f"  [{'OK' if qg and qm else 'WARN'}] ARC   - gate={qg}, math_core={qm}")
        if not (qg and qm):
            ok = False
    except Exception as e:
        print(f"  [FAIL] ARC: {e}")
        ok = False

    # MATH
    try:
        from l104_asi.symbolic_math_solver import SymbolicMathSolver
        sms = SymbolicMathSolver()
        es = sms.get_status().get("engine_support", {})
        qg = es.get("quantum_gate_engine", False)
        qm = es.get("quantum_math_core", False)
        print(f"  [{'OK' if qg and qm else 'WARN'}] MATH  - gate={qg}, math_core={qm}")
        if not (qg and qm):
            ok = False
    except Exception as e:
        print(f"  [FAIL] MATH: {e}")
        ok = False

    # HumanEval
    try:
        from l104_asi.code_generation import CodeGenerationEngine
        cge = CodeGenerationEngine()
        es = cge.get_status().get("engine_support", {})
        qg = es.get("quantum_gate_engine", False)
        qm = es.get("quantum_math_core", False)
        print(f"  [{'OK' if qg and qm else 'WARN'}] HEval - gate={qg}, math_core={qm}")
        if not (qg and qm):
            ok = False
    except Exception as e:
        print(f"  [FAIL] HumanEval: {e}")
        ok = False

    print()
    if ok:
        print("=== ALL QUANTUM LINKS ACTIVE ===")
    else:
        print("=== SOME LINKS MISSING (non-fatal) ===")

    # Quick functional test: MMLU with quantum amplification
    print()
    print("=== Functional Test: MMLU Answer with Quantum Amplification ===")
    try:
        lce2 = LanguageComprehensionEngine()
        lce2.initialize()
        result = lce2.answer_mcq(
            "What is the phenomenon where two quantum particles are correlated?",
            ["Superposition", "Entanglement", "Tunnelling", "Diffraction"],
            subject="quantum_physics"
        )
        print(f"  Answer: {result['answer']} ({result['answer_text']})")
        print(f"  Confidence: {result['confidence']}")
    except Exception as e:
        print(f"  [FAIL] Functional test: {e}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
