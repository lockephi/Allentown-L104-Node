#!/usr/bin/env python3
"""Validation test for ASI v12.0 Deep Upgrade Wave."""
import sys

def main():
    passed = 0
    total = 7

    print("=== Phase 1: Package Import ===")
    try:
        import l104_asi
        print("  OK: l104_asi imported")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    print("\n=== Phase 2: New Class Exports ===")
    try:
        from l104_asi import MCTSReasoner, ReflectionRefinementLoop
        from l104_asi import PipelineReplayBuffer, PipelineOrchestrator
        print(f"  MCTSReasoner: {MCTSReasoner.__name__}")
        print(f"  ReflectionRefinementLoop: {ReflectionRefinementLoop.__name__}")
        print(f"  PipelineReplayBuffer: {PipelineReplayBuffer.__name__}")
        print(f"  PipelineOrchestrator: {PipelineOrchestrator.__name__}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")

    print("\n=== Phase 3: quantum.py v12.0 ===")
    try:
        from l104_asi import QuantumComputationCore
        qc = QuantumComputationCore()
        s = qc.status()
        print(f"  version: {s['version']}")
        v12_caps = [c for c in s['capabilities']
                    if c in ('ENTANGLEMENT_FIDELITY_BENCH', 'QUANTUM_GRADIENT',
                             'CROSS_ENTROPY_BENCHMARK', 'QUANTUM_VOLUME_EST')]
        print(f"  v12 capabilities: {v12_caps}")
        assert s['version'] == '12.0.0', f"Expected 12.0.0, got {s['version']}"
        assert len(v12_caps) == 4, f"Expected 4 v12 caps, got {len(v12_caps)}"
        assert hasattr(qc, 'entanglement_fidelity_benchmark')
        assert hasattr(qc, 'quantum_gradient_estimation')
        assert hasattr(qc, 'cross_entropy_benchmark')
        assert hasattr(qc, 'quantum_volume_estimation')
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")

    print("\n=== Phase 4: self_mod.py v7.0 ===")
    try:
        from l104_asi import SelfModificationEngine
        sm = SelfModificationEngine()
        methods = ['get_lineage_graph', 'grover_amplified_transform_select',
                   'quantum_tunnel_escape', 'multi_file_evolve', 'predict_complexity']
        for m in methods:
            assert hasattr(sm, m), f"Missing method: {m}"
            print(f"  {m}: OK")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")

    print("\n=== Phase 5: dual_layer.py v4.0 ===")
    try:
        from l104_asi import dual_layer_engine
        print(f"  VERSION: {dual_layer_engine.VERSION}")
        assert dual_layer_engine.VERSION == '4.0.0'
        methods = ['duality_coherence', 'cross_layer_resonance_scan', 'duality_collapse_statistics']
        for m in methods:
            assert hasattr(dual_layer_engine, m), f"Missing: {m}"
            print(f"  {m}: OK")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")

    print("\n=== Phase 6: pipeline.py v7.0 ===")
    try:
        from l104_asi import SolutionChannel
        ch = SolutionChannel('test', 'test')
        assert hasattr(ch, 'enqueue'), "Missing enqueue"
        assert hasattr(ch, 'dequeue'), "Missing dequeue"
        assert hasattr(ch, 'get_health'), "Missing get_health"
        assert hasattr(ch, '_check_circuit_breaker'), "Missing circuit breaker"
        assert ch._cb_state == 'CLOSED', f"Expected CLOSED, got {ch._cb_state}"
        # Test priority queue
        ch.enqueue({'q': 'low'}, priority=10)
        ch.enqueue({'q': 'high'}, priority=1)
        high = ch.dequeue()
        assert high['q'] == 'high', f"Expected 'high', got {high}"
        print("  Priority queue: OK")
        print(f"  Circuit breaker: {ch._cb_state}")
        # Test open circuit breaker
        ch.open_circuit_breaker()
        assert ch._cb_state == 'OPEN', f"Expected OPEN, got {ch._cb_state}"
        health = ch.get_health()
        assert health['circuit_breaker'] == 'OPEN'
        result = ch.solve({'q': 'should fail'})
        assert result['error'] == 'Circuit breaker OPEN'
        print(f"  Circuit breaker opened: {ch._cb_state}")
        # Reset and verify
        ch.close_circuit_breaker()
        assert ch._cb_state == 'CLOSED'
        print(f"  Circuit breaker reset: {ch._cb_state}")
        print("  get_health: OK")
        # Test orchestrator
        orch = PipelineOrchestrator()
        stat = orch.get_status()
        assert 'hub_stats' in stat
        assert 'router_status' in stat
        assert 'telemetry' in stat
        assert 'replay_buffer' in stat
        print(f"  Orchestrator status keys: {list(stat.keys())}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")

    print("\n=== Phase 7: Functional Tests ===")
    try:
        # MCTS
        mcts = MCTSReasoner()
        result = mcts.search('What is 2+2?', lambda p: {'solution': '4', 'confidence': 0.9})
        print(f"  MCTS: confidence={result['confidence']:.4f}, solution={bool(result['solution'])}")

        # Reflection
        ref = ReflectionRefinementLoop()
        rr = ref.reflect_and_refine('Explain gravity', lambda p: {'solution': 'Gravity is...', 'confidence': 0.7})
        print(f"  Reflection: converged={rr['converged']}, reflections={rr['reflections']}")

        # Quantum gradient
        grad = qc.quantum_gradient_estimation(observable_dim=3, parameter_count=4)
        print(f"  Gradient: norm={grad['gradient_norm']}, converged={grad['converged']}")

        # Predict complexity
        src = "def hello():\n    if True:\n        print('hi')\n    for i in range(10):\n        pass\n"
        cx = sm.predict_complexity(src)
        print(f"  Complexity: cyclomatic={cx['cyclomatic_complexity']}, MI={cx['maintainability_index']}")

        # XEB
        xeb = qc.cross_entropy_benchmark(n_qubits=3, depth=4, n_samples=100)
        print(f"  XEB: fidelity={xeb['xeb_fidelity']}, heavy={xeb['heavy_output_fraction']}")

        # Duality coherence
        dc = dual_layer_engine.duality_coherence(n_samples=5)
        print(f"  Duality coherence: correlation={dc['correlation']}, resonance={dc['resonance_strength']}")

        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 50}")
    print(f"VALIDATION: {passed}/{total} PHASES PASSED")
    if passed == total:
        print("=== ALL PHASES PASSED ===")
    else:
        print(f"=== {total - passed} PHASE(S) FAILED ===")
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
