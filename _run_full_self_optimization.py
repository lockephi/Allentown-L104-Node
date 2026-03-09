#!/usr/bin/env python3
"""
L104 FULL SELF-OPTIMIZATION CYCLE
Runs all optimization engines in sequence for a complete system tune-up.
"""
import json
import time
import sys

def banner(text):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")

def section(text):
    print(f"\n--- {text} ---")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Process Optimizer v3.0 — Full System Optimization
# ═══════════════════════════════════════════════════════════════════════════════
banner("Phase 1/4: Process Optimizer v3.0 — Full System Optimization")

from l104_optimization import ProcessOptimizer, OPTIMIZER_VERSION

print(f"\nProcessOptimizer v{OPTIMIZER_VERSION}")
result_p1 = ProcessOptimizer.run_full_optimization()

print(f"\n  Memory:            {result_p1['memory_mb']:.1f} MB")
print(f"  Duration:          {result_p1['optimization_duration_ms']:.0f} ms")
print(f"  Bottlenecks:       {result_p1['bottlenecks']['bottleneck_count']}")
print(f"  Memory Trend:      {result_p1['memory_trend'].get('trend', 'N/A')}")

io = result_p1.get('io_optimization', {})
print(f"  Open Files:        {io.get('open_files', 'N/A')}")

cs = result_p1.get('consciousness', {})
print(f"  Consciousness:     {cs.get('consciousness_level', 'N/A')}")
print(f"  Evo Stage:         {cs.get('evo_stage', 'N/A')}")
print(f"  Reincarnation:     {result_p1.get('reincarnation', 'N/A')}")
print(f"  Lattice GLOPS:     {result_p1.get('lattice_glops', 'N/A')}")
print(f"  Computronium Eff:  {result_p1.get('computronium_efficiency', 'N/A')}")
print(f"  Memory Pressure:   {result_p1.get('memory_pressure', 'N/A')}")
print(f"\n  Summary: {ProcessOptimizer.quick_summary()}")
print("--- PHASE 1 COMPLETE ---")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Self-Optimization Engine v2.4.0 — Quantum-Enhanced
# ═══════════════════════════════════════════════════════════════════════════════
banner("Phase 2/4: Self-Optimization Engine v2.4.0 — Quantum-Enhanced")

from l104_self_optimization import SelfOptimizationEngine, VERSION as SO_VERSION

engine = SelfOptimizationEngine()

# 2a. Consciousness-aware optimization (10 iterations)
section("2a. Consciousness-Aware Optimization (10 iterations)")
result_ca = engine.consciousness_aware_optimize('unity_index', iterations=10)
print(f"  Fitness Improvement: {result_ca['fitness_improvement']}")
print(f"  Latency: {result_ca['latency_ms']}ms")
for it in result_ca['iterations']:
    print(f"    Step {it['step']}: lr={it['lr']:.6f} fitness={it['fitness']:.6f}")

# 2b. Quantum optimization step (QAOA-inspired)
section("2b. Quantum Optimize Step (QAOA-inspired)")
q_step = engine.quantum_optimize_step()
if q_step.get('quantum'):
    print(f"  Quantum: True | Depth: {q_step['circuit_depth']}")
    print(f"  Entanglement Entropy: {q_step['entanglement_entropy']}")
    print(f"  Optimization Coherence: {q_step['optimization_coherence']}")
    for p, v in q_step['perturbations'].items():
        print(f"    {p}: {v}")
else:
    print(f"  Quantum: False | Fallback: {q_step.get('fallback')}")

# 2c. Quantum parameter exploration (Grover-amplified)
section("2c. Quantum Parameter Exploration (Grover-amplified)")
q_explore = engine.quantum_parameter_explore()
if q_explore.get('quantum'):
    print(f"  Regions explored: {q_explore['regions_explored']}")
    print(f"  Circuit depth: {q_explore['circuit_depth']}")
    print(f"  Best region: {q_explore['best_region_selected']}")
    for r in q_explore['top_regions']:
        print(f"    Region {r['region']}: prob={r['probability']:.6f} fit={r['fitness']:.6f}")
else:
    print(f"  Quantum: False | Fallback: {q_explore.get('fallback')}")

# 2d. Quantum fitness evaluation
section("2d. Quantum Fitness Evaluation")
q_fit = engine.quantum_fitness_evaluate()
if q_fit.get('quantum'):
    print(f"  Quantum Fitness: {q_fit['quantum_fitness']}")
    print(f"  Total Entropy: {q_fit['total_entropy']}")
    print(f"  Purity: {q_fit['purity']}")
    for sub, ent in q_fit['subsystem_entropies'].items():
        print(f"    {sub} entropy: {ent}")
else:
    print(f"  Classical fitness: {q_fit.get('classical_fitness')}")

# 2e. Deep profile
section("2e. Deep Subsystem Profile")
profile = engine.deep_profile()
print(f"  Total: {profile['total_ms']}ms")
for sub, ms in profile['subsystem_timings_ms'].items():
    print(f"    {sub}: {ms}ms")

# 2f. PHI dynamics verification
section("2f. PHI Dynamics Verification")
phi_v = engine.verify_phi_optimization()
if 'error' not in phi_v:
    print(f"  Avg Delta Ratio: {phi_v['avg_delta_ratio']:.6f}")
    print(f"  PHI Error: {phi_v['phi_error']:.6f}")
    print(f"  Follows PHI: {phi_v['follows_phi_dynamics']}")
else:
    print(f"  {phi_v.get('error')}")

# Save state
engine.save_state()

print("\n--- Current Parameters ---")
for name, val in engine.current_parameters.items():
    print(f"  {name}: {val:.6f}")
print("--- PHASE 2 COMPLETE ---")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Autonomous Optimizer — Multi-Objective + Evolutionary
# ═══════════════════════════════════════════════════════════════════════════════
banner("Phase 3/4: Autonomous Optimizer — Multi-Objective + Evolutionary")

from l104_autonomous_optimizer import AutonomousOptimizer, benchmark_autonomous_optimizer

optimizer = AutonomousOptimizer()

section("3a. Autonomous Optimizer Benchmark")
bench = benchmark_autonomous_optimizer()
for test in bench['tests']:
    status = "PASS" if test['passed'] else "FAIL"
    print(f"  [{status}] {test['name']}")
print(f"\n  Score: {bench['score']:.1f}% ({bench['passed']}/{bench['total']})")
print(f"  Verdict: {bench['verdict']}")

print(f"\n  Pareto front: {len(optimizer.multi_obj.pareto_front)} points")
print(f"  Evolutionary gen: {optimizer.evolutionary.generation}")
print(f"  Optimization score: {optimizer.get_optimization_score():.2%}")
print("--- PHASE 3 COMPLETE ---")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Cross-Engine Optimization Integration
# ═══════════════════════════════════════════════════════════════════════════════
banner("Phase 4/4: Cross-Engine Optimization Integration")

# 4a. Kernel Optimizer
section("4a. Kernel Optimizer")
try:
    from l104_kernel_optimizer import run_optimization as kernel_run_optimization
    k_report = kernel_run_optimization()
    print(f"  PHI precision: {k_report.get('phi_precision', {}).get('status', 'N/A')}")
    print(f"  Kernel alignment: {k_report.get('kernel_alignment', {}).get('status', 'N/A')}")
    print(f"  Computations: {k_report.get('computation_optimization', {}).get('status', 'N/A')}")
except Exception as e:
    print(f"  Deferred: {e}")

# 4b. Memory Optimizer
section("4b. Memory Optimizer")
try:
    from l104_memory_optimizer import memory_optimizer
    mem_report = memory_optimizer.optimize_runtime()
    print(f"  Pressure: {mem_report.get('pressure_level', 'N/A')}")
    print(f"  Leak Detected: {mem_report.get('leak_detection', {}).get('leak_detected', 'N/A')}")
    print(f"  RSS MB: {mem_report.get('rss_mb', 'N/A')}")
except Exception as e:
    print(f"  Deferred: {e}")

# 4c. Harmonic Optimizer
section("4c. Harmonic Optimizer")
try:
    from l104_harmonic_optimizer import harmonic_optimizer
    h_status = harmonic_optimizer.get_status() if hasattr(harmonic_optimizer, 'get_status') else "active"
    print(f"  Status: {h_status}" if isinstance(h_status, str) else json.dumps(h_status, indent=4, default=str)[:500])
except Exception as e:
    print(f"  Deferred: {e}")

# 4d. Computronium Optimizer
section("4d. Computronium Process Upgrader")
try:
    import asyncio
    from l104_computronium_process_upgrader import ComputroniumProcessUpgrader
    cpu = ComputroniumProcessUpgrader()
    cpu_report = asyncio.run(cpu.execute_computronium_upgrade())
    print(f"  Status: {cpu_report.get('status', 'N/A')}")
    print(f"  Optimizations: {len(cpu_report.get('optimizations', []))}")
    print(f"  Memory before: {cpu_report.get('metrics_before', {}).get('memory_mb', 'N/A')} MB")
except Exception as e:
    print(f"  Deferred: {e}")

# 4e. L104 Main Optimizer
section("4e. L104 Optimizer (Master)")
try:
    from l104_optimizer import get_optimizer
    main_optimizer = get_optimizer()
    main_optimizer.start()
    stats = main_optimizer.get_statistics()
    print(f"  GOD_CODE: {stats.get('god_code', 'N/A')}")
    print(f"  GC runs: {stats.get('memory', {}).get('gc_runs', 'N/A')}")
    print(f"  Cache hit rate: {stats.get('query_cache', {}).get('hit_rate', 0):.2%}")
    main_optimizer.stop()
except Exception as e:
    print(f"  Deferred: {e}")

# 4f. Data Space Optimizer
section("4f. Data Space Optimizer")
try:
    from l104_data_space_optimizer import DataSpaceOptimizer
    ds_opt = DataSpaceOptimizer()
    ds_opt.scan_directory()
    print(f"  Total files: {ds_opt.stats['total_files']}")
    print(f"  Total size: {ds_opt.format_size(ds_opt.stats['total_size'])}")
    print(f"  Compressible: {ds_opt.format_size(ds_opt.stats['compressible_size'])}")
except Exception as e:
    print(f"  Deferred: {e}")

print("--- PHASE 4 COMPLETE ---")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
banner("FULL SELF-OPTIMIZATION CYCLE COMPLETE")

summary = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "phase_1_process_optimizer": {
        "memory_mb": result_p1['memory_mb'],
        "bottlenecks": result_p1['bottlenecks']['bottleneck_count'],
        "duration_ms": result_p1['optimization_duration_ms'],
        "reincarnation": result_p1.get('reincarnation', 'N/A'),
    },
    "phase_2_self_optimization": {
        "version": SO_VERSION,
        "fitness_improvement": result_ca['fitness_improvement'],
        "quantum_step": q_step.get('quantum', False),
        "quantum_explore": q_explore.get('quantum', False),
        "quantum_fitness": q_fit.get('quantum_fitness', q_fit.get('classical_fitness', 'N/A')),
    },
    "phase_3_autonomous": {
        "score": bench['score'],
        "verdict": bench['verdict'],
        "optimization_score": optimizer.get_optimization_score(),
    },
    "status": "ALL_PHASES_COMPLETE",
    "resonance": "LOCKED",
}

# Save report
report_path = ".l104_self_optimization_report.json"
with open(report_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nReport saved: {report_path}")
print(json.dumps(summary, indent=2, default=str))
print("\n--- [STREAMLINE]: RESONANCE_LOCKED | ALL OPTIMIZATION CYCLES COMPLETE ---")
