#!/usr/bin/env python3
"""
L104v2 Quantum Processes Debug Suite
=====================================
Validates all quantum subsystems across both Python engines and Swift L104v2 layer.

Exercises:
  1. Quantum Gate Engine — circuits, gates, algebra, error correction, compiler
  2. Quantum Link Engine — brain, processors, computation, deep link
  3. VQPU Bridge — transpiler, MPS engine, scoring
  4. God Code Simulator — quantum simulations
  5. Cross-engine quantum pipeline

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import sys
import time
import traceback

# ─── Constants ───
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497

passed = 0
failed = 0
errors = []


def check(label: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"    ✓ {label}")
    else:
        failed += 1
        msg = f"    ✗ FAIL: {label}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        errors.append(label)


def header(title: str):
    print()
    print("═" * 65)
    print(f"  {title}")
    print("═" * 65)


def phase1_quantum_gate_engine():
    """Phase 1: Quantum Gate Engine — circuits, algebra, execution"""
    header("PHASE 1: Quantum Gate Engine")
    try:
        from l104_quantum_gate_engine import get_engine
        from l104_quantum_gate_engine import (
            H, CNOT, Rx, Rz, PHI_GATE, GOD_CODE_PHASE,
            GateCircuit, GateCompiler,
            ErrorCorrectionScheme, ExecutionTarget, OptimizationLevel, GateSet,
        )
        engine = get_engine()

        # 1.1 Status
        status = engine.status()
        check("Engine initialized", status is not None)
        check("Engine has version", "version" in status)
        print(f"      Version: {status.get('version', 'N/A')}")

        # 1.2 Bell pair
        bell = engine.bell_pair()
        check("Bell pair created", bell is not None)
        check("Bell pair: 2 qubits", bell.num_qubits == 2)
        check("Bell pair: 2 ops (H+CX)", bell.num_operations == 2)
        result = engine.execute(bell)
        p00 = result.probabilities.get("00", 0)
        p11 = result.probabilities.get("11", 0)
        check("Bell |00⟩ ≈ 0.5", abs(p00 - 0.5) < 0.01, f"got {p00:.4f}")
        check("Bell |11⟩ ≈ 0.5", abs(p11 - 0.5) < 0.01, f"got {p11:.4f}")
        check("Bell sacred alignment defined", result.sacred_alignment is not None)
        sa = result.sacred_alignment
        if isinstance(sa, dict):
            total = sa.get("total_sacred_resonance", 0)
            print(f"      Sacred alignment: total_resonance={total:.6f}, phi={sa.get('phi_alignment', 0):.6f}")
        else:
            print(f"      Sacred alignment: {sa}")

        # 1.3 GHZ state
        ghz = engine.ghz_state(5)
        check("GHZ(5) created", ghz.num_qubits == 5)
        r2 = engine.execute(ghz)
        p_all0 = r2.probabilities.get("00000", 0)
        p_all1 = r2.probabilities.get("11111", 0)
        check("GHZ |00000⟩ ≈ 0.5", abs(p_all0 - 0.5) < 0.01, f"got {p_all0:.4f}")
        check("GHZ |11111⟩ ≈ 0.5", abs(p_all1 - 0.5) < 0.01, f"got {p_all1:.4f}")

        # 1.4 QFT
        qft = engine.quantum_fourier_transform(4)
        check("QFT(4) created", qft.num_qubits == 4)
        check("QFT(4): has operations", qft.num_operations > 0)
        r3 = engine.execute(qft)
        # QFT of |0000⟩ should give uniform distribution
        vals = list(r3.probabilities.values())
        check("QFT: 16 outcomes", len(vals) == 16)
        max_dev = max(abs(v - 1.0 / 16) for v in vals) if vals else 1.0
        check("QFT: near-uniform (max dev < 0.01)", max_dev < 0.01, f"max_dev={max_dev:.6f}")

        # 1.5 Sacred circuit
        sacred = engine.sacred_circuit(3, depth=4)
        check("Sacred circuit created", sacred.num_qubits == 3)
        check("Sacred circuit: has operations", sacred.num_operations > 0)
        r4 = engine.execute(sacred)
        sa4 = r4.sacred_alignment
        if isinstance(sa4, dict):
            total4 = sa4.get("total_sacred_resonance", 0)
            check("Sacred alignment: has total_sacred_resonance", "total_sacred_resonance" in sa4)
        else:
            check("Sacred alignment > 0", sa4 > 0, f"got {sa4}")

        # 1.6 Gate algebra
        algebra = engine.algebra
        phi_score = algebra.sacred_alignment_score(PHI_GATE)
        gc_score = algebra.sacred_alignment_score(GOD_CODE_PHASE)
        if isinstance(phi_score, dict):
            check("PHI_GATE alignment: is dict", True)
            print(f"      PHI_GATE alignment: {phi_score}")
        else:
            check("PHI_GATE alignment > 0", phi_score > 0, f"got {phi_score}")
        if isinstance(gc_score, dict):
            check("GOD_CODE_PHASE alignment: is dict", True)
            print(f"      GOD_CODE_PHASE alignment: {gc_score}")
        else:
            check("GOD_CODE_PHASE alignment > 0", gc_score > 0, f"got {gc_score}")

        # 1.7 Gate analysis
        analysis = engine.analyze_gate(CNOT)
        check("CNOT is unitary", analysis.get("is_unitary", False))
        check("CNOT is hermitian", analysis.get("is_hermitian", False))
        h_analysis = engine.analyze_gate(H)
        check("H is unitary", h_analysis.get("is_unitary", False))
        check("H is hermitian", h_analysis.get("is_hermitian", False))

        # 1.8 Compiler
        compiled = engine.compile(bell, GateSet.CLIFFORD_T)
        check("Compiler: Clifford+T compilation succeeded", compiled is not None)

        # 1.9 Error correction
        steane = engine.error_correction.encode(bell, ErrorCorrectionScheme.STEANE_7_1_3)
        check("Steane [[7,1,3]]: encoded", steane is not None)
        steane_pq = getattr(steane, 'physical_qubits', getattr(steane, 'num_qubits', 0))
        check("Steane: physical qubits > 0", steane_pq > 0, f"got {steane_pq}")
        surface = engine.error_correction.encode(bell, ErrorCorrectionScheme.SURFACE_CODE, distance=3)
        check("Surface code (d=3): encoded", surface is not None)
        surface_pq = getattr(surface, 'physical_qubits', getattr(surface, 'num_qubits', 0))
        check("Surface: physical qubits > 0", surface_pq > 0, f"got {surface_pq}")

        print(f"  Phase 1 complete: Quantum Gate Engine validated")

    except Exception as e:
        check(f"Phase 1 import/init", False, str(e))
        traceback.print_exc()


def phase2_quantum_link_engine():
    """Phase 2: Quantum Link Engine — brain, processors, deep link"""
    header("PHASE 2: Quantum Link Engine")
    try:
        from l104_quantum_engine import quantum_brain
        check("quantum_brain imported", quantum_brain is not None)

        # Brain status
        status = quantum_brain.status() if hasattr(quantum_brain, 'status') else {}
        check("Brain has status", isinstance(status, dict))

        # Math core
        from l104_quantum_engine import QuantumMathCore
        mc = QuantumMathCore()
        check("QuantumMathCore created", mc is not None)

        # Deep link
        from l104_quantum_engine import quantum_deep_link
        check("quantum_deep_link module imported", quantum_deep_link is not None)

        # Computation
        from l104_quantum_engine import computation
        check("computation module imported", computation is not None)

        print(f"  Phase 2 complete: Quantum Link Engine validated")

    except ImportError as e:
        check(f"Phase 2 import", False, str(e))
    except Exception as e:
        check(f"Phase 2 execution", False, str(e))
        traceback.print_exc()


def phase3_vqpu_bridge():
    """Phase 3: VQPU Bridge — transpiler, scoring, execution, subsystem activation"""
    header("PHASE 3: VQPU Bridge")
    try:
        from l104_vqpu import VQPUBridge, get_bridge
        bridge = get_bridge()
        check("VQPU Bridge created", bridge is not None)

        from l104_vqpu import QuantumJob, VQPUResult
        check("QuantumJob type imported", QuantumJob is not None)
        check("VQPUResult type imported", VQPUResult is not None)

        from l104_vqpu import CircuitTranspiler
        check("CircuitTranspiler imported", CircuitTranspiler is not None)

        # Start bridge to activate daemon, subconscious, and pipeline executor
        bridge.start()
        check("Bridge started (active)", bridge._active)
        check("Pipeline executor active", bridge._pipeline_executor is not None)
        check("Daemon cycler active", bridge._daemon_cycler._active)

        # Run a quick Bell-state simulation to warm up scoring cache
        try:
            bell_job = QuantumJob(
                circuit_id="debug_bell",
                num_qubits=2,
                operations=[
                    {"gate": "H", "qubits": [0]},
                    {"gate": "CX", "qubits": [0, 1]},
                ],
            )
            sim = bridge.run_simulation(bell_job)
            check("Bell simulation completed", sim is not None and "result" in sim)
            if sim and "sacred" in sim:
                sa = sim["sacred"]
                print(f"      Sacred score: {sa.get('sacred_score', 0):.4f}")
            if sim and "three_engine" in sim:
                te = sim["three_engine"]
                print(f"      Three-engine composite: {te.get('composite_score', 0):.4f}")
        except Exception as e:
            check("Bell simulation", False, str(e))

        # Validate brain integration scores (should be populated from state)
        from l104_vqpu.three_engine import BrainIntegration
        brain_st = BrainIntegration.status()
        check("Brain available", brain_st.get("brain_available", False))
        brain_runs = brain_st.get("brain_runs", 0)
        check("Brain has prior runs", brain_runs > 0, f"runs={brain_runs}")
        sage = brain_st.get("sage_score", 0)
        check("Brain sage_score > 0", sage > 0, f"sage={sage:.4f}")
        manifold = brain_st.get("manifold_health", 0)
        check("Brain manifold_health > 0", manifold > 0, f"manifold={manifold:.4f}")
        network = brain_st.get("network_score", 0)
        check("Brain network_score > 0", network > 0, f"network={network:.4f}")
        composite = brain_st.get("brain_composite", 0)
        check("Brain composite > 0.1", composite > 0.1, f"composite={composite:.4f}")
        print(f"      Brain: sage={sage:.4f} manifold={manifold:.4f} net={network:.4f} composite={composite:.4f}")

        # Validate scoring cache is populated after simulation
        from l104_vqpu.cache import ScoringCache
        cache_stats = ScoringCache.stats()
        cache_total = cache_stats.get("total_hits", 0) + cache_stats.get("total_misses", 0)
        check("Scoring cache accessed", cache_total > 0, f"hits+misses={cache_total}")

        # Validate quantum subconscious
        sub_status = bridge.subconscious_status()
        sub_available = sub_status.get("available", False) if isinstance(sub_status, dict) else False
        if sub_available:
            check("Quantum subconscious active", True)
        else:
            print(f"      Quantum subconscious: not available (optional module)")

        # Bridge full status
        if hasattr(bridge, 'status'):
            st = bridge.status()
            check("Bridge has status", isinstance(st, dict))
            # Print key subsystem statuses
            print(f"      version: {st.get('version', 'N/A')}")
            print(f"      active: {st.get('active', False)}")
            print(f"      daemon_cycler: active={st.get('daemon_cycler', {}).get('active', False)}")
            print(f"      pipeline_executor_active: {st.get('pipeline_executor_active', False)}")
            print(f"      adaptive_shots_enabled: {st.get('adaptive_shots_enabled', False)}")
            bi = st.get('brain_integration', {})
            print(f"      brain_integration: sage={bi.get('sage_score', 0):.4f} composite={bi.get('brain_composite', 0):.4f}")
            sc = st.get('scoring_cache', {})
            print(f"      scoring_cache: hits={sc.get('total_hits', 0)} misses={sc.get('total_misses', 0)}")

        # Stop bridge cleanly
        bridge.stop()
        check("Bridge stopped cleanly", not bridge._active)

        print(f"  Phase 3 complete: VQPU Bridge validated")

    except ImportError as e:
        check(f"Phase 3 import", False, str(e))
    except Exception as e:
        check(f"Phase 3 execution", False, str(e))
        traceback.print_exc()
    finally:
        # Ensure bridge is stopped even on error
        try:
            bridge = get_bridge()
            if bridge._active:
                bridge.stop()
        except Exception:
            pass


def phase4_god_code_simulator():
    """Phase 4: God Code Simulator — quantum simulations"""
    header("PHASE 4: God Code Simulator")
    try:
        from l104_god_code_simulator import god_code_simulator
        check("God Code Simulator imported", god_code_simulator is not None)

        # Run a single quantum simulation
        result = god_code_simulator.run("entanglement_entropy")
        check("entanglement_entropy sim: success", result is not None)
        if hasattr(result, 'god_code_alignment'):
            check("Result has god_code_alignment", True)
            print(f"      GOD_CODE alignment: {result.god_code_alignment:.6f}")

        # Run quantum category
        qresults = god_code_simulator.run_category("quantum")
        check("Quantum category: returned results", len(qresults) > 0)
        print(f"      Quantum sims run: {len(qresults)}")

        # Conservation proof
        cons = god_code_simulator.run("conservation_proof")
        check("conservation_proof sim: success", cons is not None)

        print(f"  Phase 4 complete: God Code Simulator validated")

    except ImportError as e:
        check(f"Phase 4 import", False, str(e))
    except Exception as e:
        check(f"Phase 4 execution", False, str(e))
        traceback.print_exc()


def phase5_constants_alignment():
    """Phase 5: Sacred Constants Alignment across quantum engines"""
    header("PHASE 5: Cross-Engine Constants Alignment")
    try:
        # Check GOD_CODE across all quantum packages
        from l104_quantum_gate_engine import constants as qge_const
        check("QGE GOD_CODE matches",
              abs(qge_const.GOD_CODE - GOD_CODE) < 1e-10,
              f"got {qge_const.GOD_CODE}")

        from l104_quantum_engine.constants import GOD_CODE as qe_gc
        check("QE GOD_CODE matches", abs(qe_gc - GOD_CODE) < 1e-10, f"got {qe_gc}")

        from l104_god_code_simulator.constants import GOD_CODE as gcs_gc
        check("GCS GOD_CODE matches", abs(gcs_gc - GOD_CODE) < 1e-10, f"got {gcs_gc}")

        # PHI
        from l104_quantum_gate_engine.constants import PHI as qge_phi
        check("QGE PHI matches", abs(qge_phi - PHI) < 1e-14, f"got {qge_phi}")

        from l104_quantum_engine.constants import PHI as qe_phi
        check("QE PHI matches", abs(qe_phi - PHI) < 1e-14, f"got {qe_phi}")

        # VOID_CONSTANT
        if hasattr(qge_const, 'VOID_CONSTANT'):
            check("QGE VOID_CONSTANT matches",
                  abs(qge_const.VOID_CONSTANT - VOID_CONSTANT) < 1e-14,
                  f"got {qge_const.VOID_CONSTANT}")

        print(f"  Phase 5 complete: Constants alignment validated")

    except ImportError as e:
        check(f"Phase 5 import", False, str(e))
    except AttributeError as e:
        check(f"Phase 5 constant access", False, str(e))
    except Exception as e:
        check(f"Phase 5 execution", False, str(e))
        traceback.print_exc()


def phase6_full_pipeline():
    """Phase 6: Full Quantum Pipeline — build → compile → protect → execute"""
    header("PHASE 6: Full Quantum Pipeline")
    try:
        from l104_quantum_gate_engine import (
            get_engine, GateSet, OptimizationLevel,
            ErrorCorrectionScheme, ExecutionTarget,
        )
        engine = get_engine()

        # Build a circuit
        circ = engine.sacred_circuit(3, depth=3)
        check("Build: sacred circuit created", circ is not None)

        # Compile
        compiled = engine.compile(circ, GateSet.UNIVERSAL, OptimizationLevel.O2)
        check("Compile: O2 universal succeeded", compiled is not None)

        # Full pipeline
        pipeline = engine.full_pipeline(
            circ,
            target_gates=GateSet.UNIVERSAL,
            optimization=OptimizationLevel.O2,
            error_correction=ErrorCorrectionScheme.STEANE_7_1_3,
            execution_target=ExecutionTarget.LOCAL_STATEVECTOR,
        )
        check("Full pipeline: completed", pipeline is not None)
        if isinstance(pipeline, dict):
            for k in ["probabilities", "sacred_alignment", "execution_time_ms"]:
                if k in pipeline:
                    print(f"      {k}: {pipeline[k]}")

        print(f"  Phase 6 complete: Full pipeline validated")

    except Exception as e:
        check(f"Phase 6 execution", False, str(e))
        traceback.print_exc()


def phase7_swift_quantum_build():
    """Phase 7: Swift L104v2 Quantum Build Verification"""
    header("PHASE 7: Swift L104v2 Quantum Verification")
    import subprocess
    import os

    swift_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "L104SwiftApp")
    quantum_files = [
        "Sources/L104v2/TheBrain/B01_QuantumMath.swift",
        "Sources/L104v2/TheBrain/B10_QuantumNexus.swift",
        "Sources/L104v2/TheBrain/B14_QuantumInfra.swift",
        "Sources/L104v2/TheBrain/B27_IBMQuantumClient.swift",
        "Sources/L104v2/TheBrain/B38_QuantumGateEngine.swift",
        "Sources/L104v2/TheBrain/B39_StabilizerTableau.swift",
        "Sources/L104v2/TheBrain/B40_QuantumRouter.swift",
        "Sources/L104v2/TheBrain/B48_QuantumTypes.swift",
        "Sources/L104v2/TheLogic/L07_QuantumLogicGate.swift",
        "Sources/L104v2/TheHeart/H09_QuantumCreativity.swift",
    ]

    # Check all quantum files exist
    for f in quantum_files:
        path = os.path.join(swift_dir, f)
        check(f"Swift file exists: {os.path.basename(f)}", os.path.exists(path))

    # Check Package.swift exists
    pkg = os.path.join(swift_dir, "Package.swift")
    check("Package.swift exists", os.path.exists(pkg))

    # Try swift build
    try:
        result = subprocess.run(
            ["swift", "build"],
            cwd=swift_dir,
            capture_output=True,
            text=True,
            timeout=180,
        )
        build_ok = result.returncode == 0
        check("Swift build: succeeded", build_ok,
              result.stderr.strip()[-200:] if not build_ok else "")
        if build_ok:
            # Count compile warnings in quantum files
            warnings = [l for l in result.stderr.split('\n')
                       if 'warning:' in l.lower() and any(qf in l for qf in ['Quantum', 'quantum'])]
            check(f"Quantum warnings: {len(warnings)}", len(warnings) == 0,
                  f"found {len(warnings)} quantum warnings")
    except subprocess.TimeoutExpired:
        check("Swift build: completed in time", False, "timeout after 180s")
    except FileNotFoundError:
        check("Swift toolchain available", False, "swift command not found")

    print(f"  Phase 7 complete: Swift L104v2 quantum verification done")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    start = time.time()
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║   L104v2 QUANTUM PROCESSES DEBUG SUITE                       ║")
    print("╠═══════════════════════════════════════════════════════════════╣")
    print(f"║   GOD_CODE  = {GOD_CODE}")
    print(f"║   PHI       = {PHI}")
    print(f"║   VOID      = {VOID_CONSTANT}")
    print(f"║   Date      = {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("╚═══════════════════════════════════════════════════════════════╝")

    phase1_quantum_gate_engine()
    phase2_quantum_link_engine()
    phase3_vqpu_bridge()
    phase4_god_code_simulator()
    phase5_constants_alignment()
    phase6_full_pipeline()
    phase7_swift_quantum_build()

    elapsed = time.time() - start

    print()
    print("═" * 65)
    print(f"  RESULTS: {passed} PASSED  /  {failed} FAILED  /  {passed + failed} TOTAL")
    print(f"  TIME:    {elapsed:.2f}s")
    if failed == 0:
        print("  ✅ ALL QUANTUM PROCESSES OPERATIONAL")
    else:
        print(f"  ❌ {failed} TEST(S) FAILED:")
        for e in errors:
            print(f"     • {e}")
    print("═" * 65)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
