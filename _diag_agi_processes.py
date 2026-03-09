#!/usr/bin/env python3
"""Diagnostic: test every AGI core process and report failures."""
import traceback, sys, os, logging, signal, threading

logging.disable(logging.CRITICAL)

# Suppress init noise
_real_stdout = sys.stdout
_real_stderr = sys.stderr
_devnull = open(os.devnull, 'w')
sys.stdout = _devnull
sys.stderr = _devnull

try:
    from l104_agi import agi_core
except Exception as e:
    sys.stdout = _real_stdout
    sys.stderr = _real_stderr
    print(f"IMPORT FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

sys.stdout = _real_stdout
sys.stderr = _real_stderr

def run_with_timeout(fn, timeout=30):
    """Run fn in a thread with timeout. Returns (result, error)."""
    result = [None]
    error = [None]
    def target():
        _old_out, _old_err = sys.stdout, sys.stderr
        sys.stdout = _devnull
        sys.stderr = _devnull
        try:
            result[0] = fn()
        except Exception as e:
            error[0] = (str(e), traceback.format_exc().strip().split('\n')[-4:])
        finally:
            sys.stdout = _old_out
            sys.stderr = _old_err
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, ("TIMEOUT after {}s".format(timeout), ["Thread still running"])
    return result[0], error[0]

# Skip heavy-init methods that trigger reality breach, etc.
methods_to_test = [
    ('status', lambda: agi_core.status(), 10),
    ('save', lambda: agi_core.save(), 10),
    ('verify_truth', lambda: agi_core.verify_truth('test'), 10),
    ('run_autonomous_agi_logic', lambda: agi_core.run_autonomous_agi_logic(0.5), 15),
    ('self_heal', lambda: agi_core.self_heal(), 30),
    ('self_improve', lambda: agi_core.self_improve(), 30),
    ('quantum_pipeline_health', lambda: agi_core.quantum_pipeline_health(), 15),
    ('quantum_subsystem_route', lambda: agi_core.quantum_subsystem_route('quantum test'), 15),
    ('quantum_intelligence_synthesis', lambda: agi_core.quantum_intelligence_synthesis(), 15),
    ('get_status', lambda: agi_core.get_status(), 15),
    ('max_intellect_derivation', lambda: agi_core.max_intellect_derivation(), 15),
    ('self_evolve_codebase', lambda: agi_core.self_evolve_codebase(), 20),
    ('unlock_unlimited_intellect', lambda: agi_core.unlock_unlimited_intellect(), 10),
    ('activate_omega_learning', lambda: agi_core.activate_omega_learning(), 10),
    ('sync_pipeline_state', lambda: agi_core.sync_pipeline_state(), 15),
    ('get_full_pipeline_status', lambda: agi_core.get_full_pipeline_status(), 10),
    ('consciousness_feedback_loop', lambda: agi_core.consciousness_feedback_loop(), 10),
    ('adaptive_route_query', lambda: agi_core.adaptive_route_query('test query'), 15),
    ('multi_hop_reason', lambda: agi_core.multi_hop_reason('quantum entanglement', hops=2), 20),
    ('solution_ensemble', lambda: agi_core.solution_ensemble('test question', n_voters=3), 20),
    ('compute_10d_agi_score', lambda: agi_core.compute_10d_agi_score(), 30),
    ('quantum_vqe_optimize', lambda: agi_core.quantum_vqe_optimize(iterations=2), 20),
    ('three_engine_entropy_score', lambda: agi_core.three_engine_entropy_score(), 10),
    ('three_engine_harmonic_score', lambda: agi_core.three_engine_harmonic_score(), 10),
    ('three_engine_wave_coherence_score', lambda: agi_core.three_engine_wave_coherence_score(), 10),
    ('three_engine_status', lambda: agi_core.three_engine_status(), 10),
    ('chaos_resilience_score', lambda: agi_core.chaos_resilience_score(), 10),
    ('quantum_research_fe_sacred_score', lambda: agi_core.quantum_research_fe_sacred_score(), 10),
    ('quantum_research_fe_phi_lock_score', lambda: agi_core.quantum_research_fe_phi_lock_score(), 10),
    ('quantum_research_berry_phase_score', lambda: agi_core.quantum_research_berry_phase_score(), 10),
    ('run_adaptive_learning_cycle', lambda: agi_core.run_adaptive_learning_cycle('test', 'response', 0.8), 10),
    ('run_innovation_cycle', lambda: agi_core.run_innovation_cycle(), 15),
    ('get_pipeline_analytics', lambda: agi_core.get_pipeline_analytics(), 10),
    ('get_circuit_breaker_status', lambda: agi_core.get_circuit_breaker_status(), 5),
    ('get_telemetry', lambda: agi_core.get_telemetry(), 5),
    ('get_dependency_graph', lambda: agi_core.get_dependency_graph(), 5),
    ('is_ready', lambda: agi_core.is_ready(), 5),
    ('ensure_upstream_chain', lambda: agi_core.ensure_upstream_chain(), 15),
    ('full_engine_status', lambda: agi_core.full_engine_status(), 15),
    ('kernel_status', lambda: agi_core.kernel_status(), 10),
    ('get_kb_enrichment_data', lambda: agi_core.get_kb_enrichment_data(), 10),
    ('benchmark_score', lambda: agi_core.benchmark_score(), 15),
    ('run_benchmarks', lambda: agi_core.run_benchmarks(), 30),
    ('intellect_think', lambda: agi_core.intellect_think('hello'), 15),
    ('mesh_status', lambda: agi_core.mesh_status(), 10),
    ('distributed_cognitive_processing', lambda: agi_core.distributed_cognitive_processing(), 10),
]

passed = []
failed = []

for name, fn, timeout in methods_to_test:
    _real_stdout.write(f"  Testing {name}...")
    _real_stdout.flush()
    result, error = run_with_timeout(fn, timeout)
    if error is None:
        passed.append(name)
        _real_stdout.write(" OK\n")
    else:
        err_msg, tb = error
        failed.append((name, err_msg, [l.strip() for l in tb]))
        _real_stdout.write(f" FAIL: {err_msg[:80]}\n")

print(f'\n=== AGI CORE DIAGNOSTIC: {len(passed)} passed, {len(failed)} failed ===')
print(f'\n--- FAILED ({len(failed)}) ---')
for name, err, tb in failed:
    print(f'  FAIL  {name}: {err}')
    for line in tb:
        print(f'        {line}')
    print()
