#!/usr/bin/env python3
"""Quick diagnostic for Phase 4 failures."""
import sys

print("=== 4a. kernel_optimizer ===")
try:
    from l104_kernel_optimizer import run_optimization as kernel_run_optimization
    k = kernel_run_optimization()
    print(f"  OK: {list(k.keys())[:5]}")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n=== 4d. ComputroniumProcessUpgrader ===")
try:
    from l104_computronium_process_upgrader import ComputroniumProcessUpgrader
    cpu = ComputroniumProcessUpgrader()
    if hasattr(cpu, 'run_transfusion'):
        cpu_report = cpu.run_transfusion()
        print(f"  OK run_transfusion: {list(cpu_report.keys())[:5]}")
    else:
        print(f"  MISSING run_transfusion. Methods: {[m for m in dir(cpu) if not m.startswith('_')]}")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n=== 4e. L104 Optimizer ===")
try:
    from l104_optimizer import get_optimizer
    main_optimizer = get_optimizer()
    if main_optimizer is None:
        print("  FAIL: get_optimizer() returned None")
    else:
        main_optimizer.start()
        stats = main_optimizer.get_statistics()
        print(f"  OK: {list(stats.keys())[:5]}")
        main_optimizer.stop()
except Exception as e:
    print(f"  FAIL: {e}")

print("\n=== 4f. DataSpaceOptimizer ===")
try:
    from l104_data_space_optimizer import DataSpaceOptimizer
    ds = DataSpaceOptimizer()
    if hasattr(ds, 'scan_directory'):
        ds.scan_directory()
        print(f"  OK: total_files={ds.stats.get('total_files', 'N/A')}")
    else:
        print(f"  MISSING scan_directory. Methods: {[m for m in dir(ds) if not m.startswith('_')]}")
except Exception as e:
    print(f"  FAIL: {e}")
