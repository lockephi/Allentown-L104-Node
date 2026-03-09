#!/usr/bin/env python3
"""
L104 God Code Simulator — Integration Test
═══════════════════════════════════════════════════════════════════════════════
Tests the unified GodCodeSimulator package and its wiring into:
  1. Standalone simulator API (23 simulations, sweeps, optimizer, feedback)
  2. ScienceEngine integration (coherence, entropy, feedback loop)
  3. MathEngine integration (god code verification, sweeps)
  4. ASI pipeline integration (connect_pipeline, pipeline_solve)

Run: .venv/bin/python test_god_code_simulator.py
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import time
import traceback

# ── Test infrastructure ──
class TestTracker:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name: str, condition: bool, detail: str = ""):
        if condition:
            self.passed += 1
            print(f"  ✓ {name}" + (f" — {detail}" if detail else ""))
        else:
            self.failed += 1
            self.errors.append(name)
            print(f"  ✗ {name}" + (f" — {detail}" if detail else ""))

    @property
    def total(self):
        return self.passed + self.failed

    def report(self):
        print(f"\n{'='*70}")
        print(f"  RESULTS: {self.passed}/{self.total} passed, {self.failed} failed")
        if self.errors:
            print(f"  FAILURES: {', '.join(self.errors)}")
        print(f"{'='*70}")
        return self.failed == 0


tracker = TestTracker()
t_start = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Standalone Simulator Package
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  PHASE 1: GOD CODE SIMULATOR — STANDALONE PACKAGE")
print("═" * 70)

try:
    from l104_god_code_simulator import (
        GodCodeSimulator, SimulationResult, SimulationCatalog,
        ParametricSweepEngine, AdaptiveOptimizer, FeedbackLoopEngine,
        god_code_simulator,
    )
    tracker.check("import_package", True, "All classes imported")
except Exception as e:
    tracker.check("import_package", False, str(e))
    print("FATAL: Cannot continue without package import")
    sys.exit(1)

# Version + status
status = god_code_simulator.get_status()
tracker.check("version", status["version"] == "2.0.0", f"v{status['version']}")
tracker.check("catalog_count", status["simulations_registered"] >= 20,
              f"{status['simulations_registered']} simulations registered")
tracker.check("categories", len(status["categories"]) == 4,
              f"categories: {status['categories']}")

# Run single simulation
result = god_code_simulator.run("conservation_proof")
tracker.check("run_conservation", result.passed, result.summary())
tracker.check("result_type", isinstance(result, SimulationResult), "SimulationResult")

# Run by category
quantum_results = god_code_simulator.run_category("quantum")
tracker.check("run_quantum_category", len(quantum_results) >= 5,
              f"{len(quantum_results)} quantum simulations")
tracker.check("quantum_all_pass", all(r.passed for r in quantum_results),
              f"{sum(1 for r in quantum_results if r.passed)}/{len(quantum_results)} passed")

# Run all
report = god_code_simulator.run_all()
tracker.check("run_all", report["total"] >= 20, f"{report['total']} total simulations")
tracker.check("pass_rate", report["pass_rate"] > 0.8,
              f"pass rate = {report['pass_rate']:.1%}")

# Coherence payload format
result = god_code_simulator.run("entanglement_entropy")
payload = result.to_coherence_payload()
tracker.check("coherence_payload", "total_fidelity" in payload and "probabilities" in payload,
              f"keys: {list(payload.keys())}")

# Entropy input
entropy_val = result.to_entropy_input()
tracker.check("entropy_input", isinstance(entropy_val, float) and entropy_val >= 0,
              f"entropy={entropy_val:.4f}")

# ASI scoring
asi_score = result.to_asi_scoring()
tracker.check("asi_scoring", "fidelity" in asi_score and "sacred_alignment" in asi_score,
              f"keys: {list(asi_score.keys())}")

# Math verification
math_ver = result.to_math_verification()
tracker.check("math_verification", "god_code_measured" in math_ver,
              f"god_code_measured={math_ver.get('god_code_measured', 0):.4f}")

# Parametric sweep
sweep = god_code_simulator.parametric_sweep("dial_a", start=0, stop=4)
sweep_pass = sum(1 for s in sweep if s["passed"])
tracker.check("dial_sweep", len(sweep) >= 4 and sweep_pass == len(sweep),
              f"{len(sweep)} points, {sweep_pass}/{len(sweep)} conserved")

depth_sweep = god_code_simulator.parametric_sweep("depth")
tracker.check("depth_sweep", len(depth_sweep) >= 4,
              f"{len(depth_sweep)} depth points")

# Adaptive optimizer
opt_result = god_code_simulator.adaptive_optimize(target_fidelity=0.5, nq=3, depth=2)
tracker.check("adaptive_optimize", "best_fidelity" in opt_result,
              f"best_fidelity={opt_result['best_fidelity']:.4f}")

noise_result = god_code_simulator.optimize_noise_resilience(nq=2, noise_level=0.1)
tracker.check("noise_resilience", "best_strategy" in noise_result,
              f"best_strategy={noise_result['best_strategy']}")

# Feedback loop (standalone, no live engines)
fb_result = god_code_simulator.run_feedback_loop(iterations=3)
tracker.check("feedback_loop_standalone", fb_result["iterations"] >= 3,
              f"iterations={fb_result['iterations']}, "
              f"avg_coherence={fb_result['avg_coherence']:.4f}")

# Engine integration convenience methods
coh_payload = god_code_simulator.simulate_for_coherence("entanglement_entropy")
tracker.check("simulate_for_coherence", "total_fidelity" in coh_payload)

ent_val = god_code_simulator.simulate_for_entropy("decoherence_model")
tracker.check("simulate_for_entropy", isinstance(ent_val, float))

asi_payload = god_code_simulator.simulate_for_asi("sacred_cascade")
tracker.check("simulate_for_asi", "simulation_name" in asi_payload)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Science Engine Integration
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  PHASE 2: SCIENCE ENGINE INTEGRATION")
print("═" * 70)

try:
    from l104_science_engine import science_engine, ScienceEngine
    tracker.check("se_import", True)

    # Check god_code_sim attribute
    tracker.check("se_god_code_sim_attr", hasattr(science_engine, 'god_code_sim'),
                  f"god_code_sim={'connected' if science_engine.god_code_sim else 'None'}")

    # Run simulation via ScienceEngine
    sim_result = science_engine.run_god_code_simulation("conservation_proof")
    tracker.check("se_run_sim", "passed" in sim_result or "error" not in sim_result,
                  str(sim_result.get("passed", sim_result.get("error", "?"))))

    # Check feedback loop method
    tracker.check("se_feedback_method", hasattr(science_engine, 'run_god_code_feedback_loop'))

    # Check god_code_simulation_status
    sim_status = science_engine.god_code_simulation_status()
    tracker.check("se_sim_status", sim_status.get("available", False),
                  f"sims={sim_status.get('simulations_registered', '?')}")

    # Check it appears in full status
    full_status = science_engine.get_full_status()
    tracker.check("se_full_status", "god_code_simulator" in full_status,
                  f"domains: {len(full_status.get('active_domains', []))}")

    # Verify GOD_CODE_SIMULATION domain
    tracker.check("se_domain_added", "GOD_CODE_SIMULATION" in full_status.get("active_domains", []))

except Exception as e:
    tracker.check("se_integration", False, f"{e}\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Math Engine Integration
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  PHASE 3: MATH ENGINE INTEGRATION")
print("═" * 70)

try:
    from l104_math_engine import math_engine, MathEngine
    tracker.check("me_import", True)

    # Run god code simulation via Math Engine
    sim_result = math_engine.run_god_code_simulation("conservation_proof")
    tracker.check("me_run_sim", "passed" in sim_result,
                  f"passed={sim_result.get('passed')}, conservation={sim_result.get('local_conservation')}")

    # Run parametric sweep via Math Engine
    sweep_result = math_engine.simulate_god_code_sweep("a", 0, 3)
    tracker.check("me_sweep", len(sweep_result) >= 4,
                  f"{len(sweep_result)} points")

    # Run all simulations
    all_result = math_engine.simulate_all_god_code()
    tracker.check("me_run_all", "total" in all_result,
                  f"total={all_result.get('total')}, "
                  f"passed={all_result.get('passed')}")

    # Check simulator in status
    me_status = math_engine.status()
    tracker.check("me_status", "god_code_simulator" in me_status,
                  f"simulator available: {me_status.get('god_code_simulator', {}).get('simulations_registered', '?')}")

except Exception as e:
    tracker.check("me_integration", False, f"{e}\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: ASI Pipeline Integration
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  PHASE 4: ASI PIPELINE INTEGRATION")
print("═" * 70)

try:
    from l104_asi import asi_core
    tracker.check("asi_import", True)

    # Check _god_code_simulator attribute initialized
    tracker.check("asi_sim_attr", hasattr(asi_core, '_god_code_simulator'),
                  f"_god_code_simulator={'set' if asi_core._god_code_simulator else 'None (pre-connect)'}")

    # Connect pipeline (this wires the simulator)
    connect_result = asi_core.connect_pipeline()
    connected = connect_result.get("connected", [])
    tracker.check("asi_connect_pipeline",
                  "god_code_simulator" in connected,
                  f"god_code_simulator in {len(connected)} connected subsystems")

    # Verify simulator is now set
    tracker.check("asi_sim_connected", asi_core._god_code_simulator is not None,
                  "Simulator connected post-pipeline")

    # Test pipeline_solve with a sacred query
    solve_result = asi_core.pipeline_solve("What is the entanglement entropy of GOD_CODE?")
    tracker.check("asi_pipeline_solve", isinstance(solve_result, dict),
                  f"keys: {list(solve_result.keys())[:5]}")

    # Check god_code_simulation appears in solve result
    has_sim_result = "god_code_simulation" in solve_result
    tracker.check("asi_sim_in_solve", has_sim_result,
                  f"god_code_simulation={'present' if has_sim_result else 'absent'}")

except Exception as e:
    tracker.check("asi_integration", False, f"{e}\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Cross-Engine Feedback Loop (Science + Math + Simulator)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  PHASE 5: CROSS-ENGINE FEEDBACK LOOP")
print("═" * 70)

try:
    from l104_god_code_simulator.simulator import PHI, GOD_CODE

    # Create fresh simulator with live engine connections
    sim = GodCodeSimulator()
    try:
        from l104_science_engine import science_engine as se
        sim.connect_engines(coherence=se.coherence, entropy=se.entropy)
        # Initialize coherence field for real feedback
        se.coherence.initialize([1.0, 0.5, 0.3, PHI, GOD_CODE / 1000.0])
    except ImportError:
        pass
    try:
        from l104_math_engine import math_engine as me
        sim.feedback.connect_math(me)
    except ImportError:
        pass

    # Run feedback loop with live engines
    fb = sim.run_feedback_loop(iterations=4)
    tracker.check("cross_feedback_loop", fb["iterations"] >= 3,
                  f"iterations={fb['iterations']}, converging={fb['converging']}")
    tracker.check("cross_engines_connected",
                  fb["engines_connected"].get("coherence", False) or
                  fb["engines_connected"].get("entropy", False),
                  f"engines={fb['engines_connected']}")

    # Batch simulate for coherence feedback loop
    batch = sim.batch_simulate_for_coherence(n=3)
    tracker.check("batch_coherence", len(batch) >= 3,
                  f"{len(batch)} payloads generated")

    # Run real coherence feedback loop if initialized
    try:
        if se.coherence.coherence_field:
            fb_real = se.coherence.run_feedback_loop(batch, evolve_steps=3)
            tracker.check("real_coherence_feedback",
                          fb_real.get("iterations", 0) >= 2,
                          f"iterations={fb_real.get('iterations')}, "
                          f"converging={fb_real.get('converging')}")
        else:
            tracker.check("real_coherence_feedback", True, "skipped (no coherence field)")
    except Exception as e:
        tracker.check("real_coherence_feedback", False, str(e))

except Exception as e:
    tracker.check("cross_engine_feedback", False, f"{e}\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

elapsed = time.time() - t_start
print(f"\n  Total time: {elapsed:.2f}s")
ok = tracker.report()
sys.exit(0 if ok else 1)
