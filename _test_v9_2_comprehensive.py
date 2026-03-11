#!/usr/bin/env python3
"""
Comprehensive v9.2.0 Integration Test
=====================================
Tests all v9.2.0 components for proper functionality
"""

import sys
sys.path.insert(0, '/Users/carolalvarez/Applications/Allentown-L104-Node')

def test_imports():
    """Test all v9.2.0 imports from l104_asi."""
    try:
        from l104_asi import (
            # v9.2.0 Optimization Engine
            AdaptiveLatencyTargeter,
            MemoryBudgetOptimizer,
            CascadingOptimizationController,
            PipelineOptimizationOrchestrator,

            # v9.2.0 Activation Sequencer
            AdaptiveActivationSequencer,
            ActivationPhase,
            ActivationMetrics,
            SequencingMode,

            # v9.2.0 Constants
            TARGET_LATENCY_LCE_MS,
            TARGET_LATENCY_QE_MS,
            TARGET_LATENCY_SM_MS,
            TARGET_LATENCY_COHERENCE_MS,
            TARGET_LATENCY_ACTIVATION_MS,
            TIMEOUT_NORMAL_MULTIPLIER,
            TIMEOUT_DEGRADED_MULTIPLIER,
            TIMEOUT_CRITICAL_MULTIPLIER,
            MAX_MEMORY_PERCENT_ASI,
            MEMORY_SAFETY_MARGIN_PCT,
            ADAPTIVE_MEMORY_THRESHOLD_PCT,
            CASCADE_MAX_RETRY_ADAPTIVE,
            CASCADE_CONFIDENCE_THRESHOLD,
            CASCADE_FAILURE_THRESHOLD,
            ACTIVATION_STEPS_V9_2,
            ACTIVATION_WARMUP_SAMPLES,
            ACTIVATION_COOLDOWN_ITERATIONS,
            SCORE_AGGREGATION_MODE,
            SCORE_HARMONIC_WEIGHT,
            SCORE_OUTLIER_DETECTION,
            SCORE_OUTLIER_SIGMA,
        )
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_constants():
    """Verify all constants have correct values."""
    from l104_asi import (
        TARGET_LATENCY_LCE_MS,
        TARGET_LATENCY_QE_MS,
        TIMEOUT_NORMAL_MULTIPLIER,
        ACTIVATION_STEPS_V9_2,
        SCORE_HARMONIC_WEIGHT,
        SCORE_OUTLIER_SIGMA,
    )

    checks = [
        (TARGET_LATENCY_LCE_MS == 50.0, f"LCE target: {TARGET_LATENCY_LCE_MS}"),
        (TARGET_LATENCY_QE_MS == 30.0, f"QE target: {TARGET_LATENCY_QE_MS}"),
        (TIMEOUT_NORMAL_MULTIPLIER > 4.0, f"Timeout multiplier: {TIMEOUT_NORMAL_MULTIPLIER:.3f}"),
        (ACTIVATION_STEPS_V9_2 == 25, f"Activation steps: {ACTIVATION_STEPS_V9_2}"),
        (0.6 < SCORE_HARMONIC_WEIGHT < 0.65, f"Harmonic weight: {SCORE_HARMONIC_WEIGHT}"),
        (SCORE_OUTLIER_SIGMA == 2.0, f"Outlier sigma: {SCORE_OUTLIER_SIGMA}"),
    ]

    all_pass = True
    for passed, desc in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {desc}")
        all_pass = all_pass and passed

    return all_pass


def test_orchestrator():
    """Test optimization orchestrator."""
    from l104_asi.optimization_engine import (
        get_orchestrator,
        PipelineOptimizationOrchestrator,
        StageType,
        LatencyInfo
    )

    orch = get_orchestrator()

    checks = [
        (isinstance(orch, PipelineOptimizationOrchestrator), "Instance type correct"),
        (orch.optimization_enabled, "Optimization enabled"),
        (hasattr(orch, 'latency_targeter'), "Has latency_targeter"),
        (hasattr(orch, 'memory_optimizer'), "Has memory_optimizer"),
        (hasattr(orch, 'cascade_controller'), "Has cascade_controller"),
    ]

    # Test methods
    timeout = orch.get_stage_timeout(StageType.SYMBOLIC_MATH)
    checks.append((timeout > 0, f"Get timeout: {timeout:.1f}ms"))

    status = orch.get_optimization_status()
    checks.append(('latency' in status and 'memory' in status, "Status dict complete"))

    all_pass = True
    for passed, desc in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {desc}")
        all_pass = all_pass and passed

    return all_pass


def test_sequencer():
    """Test activation sequencer."""
    from l104_asi.adaptive_activation_sequencer import (
        get_sequencer,
        AdaptiveActivationSequencer,
        ActivationPhase,
        SequencingMode,
    )

    seq = get_sequencer()

    checks = [
        (isinstance(seq, AdaptiveActivationSequencer), "Instance type correct"),
        (seq.current_mode == SequencingMode.ADAPTIVE, "Initial mode is ADAPTIVE"),
        (seq.warmup_samples_remaining == 3, f"Warmup samples: {seq.warmup_samples_remaining}"),
        (seq.num_phases == 25, f"Phase count: {seq.num_phases}"),
    ]

    # Test methods
    order = seq.get_optimal_sequence()
    checks.append((len(order) > 0, f"Optimal sequence: {len(order)} phases"))

    should_warmup = seq.should_perform_warmup()
    checks.append((should_warmup, "Should perform warmup"))

    status = seq.get_activation_status()
    checks.append(('current_mode' in status and 'phases' in status, "Status dict complete"))

    all_pass = True
    for passed, desc in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {desc}")
        all_pass = all_pass and passed

    return all_pass


def test_metrics():
    """Test activation metrics."""
    from l104_asi.adaptive_activation_sequencer import (
        get_metrics,
        ActivationMetrics,
    )

    metr = get_metrics()

    checks = [
        (isinstance(metr, ActivationMetrics), "Instance type correct"),
        (metr.total_activations == 0, "Initial activations is 0"),
        (metr.get_success_rate() == 0.0, "Initial success rate is 0%"),
    ]

    # Simulate activations
    metr.record_activation(True, 0.05, 0.95)
    metr.record_activation(True, 0.04, 0.92)
    metr.record_activation(False, 0.10, 0.50)

    checks.extend([
        (metr.total_activations == 3, f"Total activations: {metr.total_activations}"),
        (metr.successful_activations == 2, f"Successful: {metr.successful_activations}"),
        (abs(metr.get_success_rate() - 2/3) < 0.01, f"Success rate: {metr.get_success_rate():.1%}"),
    ])

    all_pass = True
    for passed, desc in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {desc}")
        all_pass = all_pass and passed

    return all_pass


def main():
    """Run all tests."""
    print("=" * 70)
    print("ASI v9.2.0 — Comprehensive Integration Test")
    print("=" * 70)

    results = []

    print("\n1. Testing Imports...")
    results.append(("Imports", test_imports()))

    print("\n2. Testing Constants...")
    results.append(("Constants", test_constants()))

    print("\n3. Testing Optimization Orchestrator...")
    results.append(("Orchestrator", test_orchestrator()))

    print("\n4. Testing Activation Sequencer...")
    results.append(("Sequencer", test_sequencer()))

    print("\n5. Testing Activation Metrics...")
    results.append(("Metrics", test_metrics()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {name}")
        all_passed = all_passed and passed

    print("=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED — v9.2.0 Ready for Integration!")
        return 0
    else:
        print("❌ SOME TESTS FAILED — Review issues above")
        return 1


if __name__ == "__main__":
    exit(main())
