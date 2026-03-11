#!/usr/bin/env python3
"""Quick test of v9.2.0 modules."""

import sys
sys.path.insert(0, '/Users/carolalvarez/Applications/Allentown-L104-Node')

# Test optimization_engine imports
try:
    from l104_asi.optimization_engine import (
        AdaptiveLatencyTargeter,
        MemoryBudgetOptimizer,
        CascadingOptimizationController,
        PipelineOptimizationOrchestrator,
        get_orchestrator,
        StageType,
        LatencyInfo
    )
    print('✓ optimization_engine imports successful')
except Exception as e:
    print(f'✗ optimization_engine import failed: {e}')
    sys.exit(1)

# Test adaptive_activation_sequencer imports
try:
    from l104_asi.adaptive_activation_sequencer import (
        AdaptiveActivationSequencer,
        ActivationPhase,
        ActivationMetrics,
        SequencingMode,
        get_sequencer,
        get_metrics
    )
    print('✓ adaptive_activation_sequencer imports successful')
except Exception as e:
    print(f'✗ adaptive_activation_sequencer import failed: {e}')
    sys.exit(1)

# Test initialization
try:
    orch = get_orchestrator()
    print(f'✓ Orchestrator initialized')

    seq = get_sequencer()
    print(f'✓ Sequencer initialized')
    print(f'  - Mode: {seq.current_mode}')
    print(f'  - Warmup remaining: {seq.warmup_samples_remaining}')

    metr = get_metrics()
    print(f'✓ Metrics initialized')

    print('\n✅ All v9.2.0 modules loaded and initialized successfully!')
except Exception as e:
    print(f'✗ Initialization failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
