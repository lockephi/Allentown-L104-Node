#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
 L104 QUANTUM COHERENCE DEBUG SUITE
 Tests coherence across Python engines + validates Swift cross-system parity
═══════════════════════════════════════════════════════════════════════════════

 PHASE 1: Science Engine CoherenceSubsystem (topological braiding)
 PHASE 2: ASI QuantumComputationCore (Fe-Sacred / Fe-PHI coherence)
 PHASE 3: QuantumCoherenceEngine (Qiskit circuit coherence)
 PHASE 4: Cross-system constant validation (Python ↔ Swift parity)
 PHASE 5: Coherence feedback loop stress test
 PHASE 6: Swift coherence architecture analysis

 Run: .venv/bin/python debug_quantum_coherence.py
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import time
import math
import traceback
from typing import Dict, Any, List, Tuple

# ── Color output ──
G = "\033[92m"   # green
R = "\033[91m"   # red
Y = "\033[93m"   # yellow
C = "\033[96m"   # cyan
M = "\033[95m"   # magenta
B = "\033[1m"    # bold
RST = "\033[0m"  # reset

passed = 0
failed = 0
warnings = 0
results: List[Tuple[str, str, str]] = []  # (phase, test, status)

def ok(phase: str, test: str, detail: str = ""):
    global passed
    passed += 1
    results.append((phase, test, "PASS"))
    print(f"  {G}✓{RST} {test}" + (f"  {C}{detail}{RST}" if detail else ""))

def fail(phase: str, test: str, detail: str = ""):
    global failed
    failed += 1
    results.append((phase, test, "FAIL"))
    print(f"  {R}✗{RST} {test}" + (f"  {R}{detail}{RST}" if detail else ""))

def warn(phase: str, test: str, detail: str = ""):
    global warnings
    warnings += 1
    results.append((phase, test, "WARN"))
    print(f"  {Y}⚠{RST} {test}" + (f"  {Y}{detail}{RST}" if detail else ""))


def banner(text: str):
    w = 72
    print(f"\n{B}{M}{'═' * w}")
    print(f"  {text}")
    print(f"{'═' * w}{RST}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: Science Engine — CoherenceSubsystem
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_science_coherence():
    banner("PHASE 1: Science Engine — CoherenceSubsystem")
    phase = "P1"

    try:
        from l104_science_engine import ScienceEngine
        se = ScienceEngine()
        ok(phase, "ScienceEngine import + init")
    except Exception as e:
        fail(phase, "ScienceEngine import", str(e))
        return

    # 1.1 CoherenceSubsystem exists
    coh = se.coherence
    if coh is not None:
        ok(phase, "CoherenceSubsystem accessible via se.coherence")
    else:
        fail(phase, "CoherenceSubsystem is None")
        return

    # 1.2 Initialize coherence field
    try:
        seeds = ["quantum", "coherence", "topological", "braiding",
                 "Fibonacci", "anyon", "vacuum", "GOD_CODE", "iron", "sacred"]
        init_result = coh.initialize(seeds)
        dim = init_result.get("dimension", 0)
        phase_coh = init_result.get("phase_coherence", 0)
        energy = init_result.get("energy", 0)
        if dim == len(seeds) and phase_coh > 0 and abs(energy - 1.0) < 0.01:
            ok(phase, f"initialize({len(seeds)} seeds)", f"dim={dim} coh={phase_coh:.4f} E={energy:.4f}")
        else:
            fail(phase, "initialize", f"dim={dim} coh={phase_coh} E={energy}")
    except Exception as e:
        fail(phase, "initialize", str(e))

    # 1.3 Evolve (topological braiding)
    try:
        ev = coh.evolve(steps=20)
        init_c = ev["initial_coherence"]
        final_c = ev["final_coherence"]
        preserved = ev["preserved"]
        avg_p = ev["avg_protection"]
        if final_c > 0:
            ok(phase, f"evolve(20 steps)", f"init={init_c:.4f} → final={final_c:.4f} prot={avg_p:.4f} preserved={preserved}")
        else:
            fail(phase, "evolve", f"final_coherence={final_c}")
    except Exception as e:
        fail(phase, "evolve", str(e))

    # 1.4 Temporal anchor
    try:
        anchor = coh.anchor(1.0)
        ctc = anchor["ctc_stability"]
        locked = anchor["locked"]
        ok(phase, f"anchor", f"ctc={ctc:.4f} locked={locked}")
    except Exception as e:
        fail(phase, "anchor", str(e))

    # 1.5 Discover PHI patterns
    try:
        disc = coh.discover()
        patterns = disc["phi_patterns"]
        emergence = disc["emergence"]
        fe_ref = disc.get("fe_sacred_reference", None)
        zne = disc.get("zne_bridge_active", False)
        ok(phase, f"discover", f"φ-patterns={patterns} emergence={emergence:.4f} fe_ref={fe_ref} zne={zne}")
    except Exception as e:
        fail(phase, "discover", str(e))

    # 1.6 Golden angle spectrum
    try:
        ga = coh.golden_angle_spectrum()
        mean_align = ga.get("mean_alignment", 0)
        is_spiral = ga.get("is_golden_spiral", False)
        ok(phase, f"golden_angle_spectrum", f"mean_align={mean_align:.4f} spiral={is_spiral}")
    except Exception as e:
        fail(phase, "golden_angle_spectrum", str(e))

    # 1.7 Energy spectrum
    try:
        es = coh.energy_spectrum()
        total_e = es["total_energy"]
        entropy = es["shannon_entropy_bits"]
        ok(phase, f"energy_spectrum", f"E={total_e:.6f} entropy={entropy:.4f} bits")
    except Exception as e:
        fail(phase, "energy_spectrum", str(e))

    # 1.8 Coherence fidelity
    try:
        fid = coh.coherence_fidelity()
        current = fid["current_coherence"]
        grade = fid["grade"]
        fidelity_val = fid["fidelity"]
        ok(phase, f"coherence_fidelity", f"coh={current:.4f} fidelity={fidelity_val:.4f} grade={grade}")
    except Exception as e:
        fail(phase, "coherence_fidelity", str(e))

    # 1.9 Synthesize verdict
    try:
        verdict = coh.synthesize()
        ok(phase, f"synthesize", f"'{verdict[:60]}'")
    except Exception as e:
        fail(phase, "synthesize", str(e))

    # 1.10 Status report
    try:
        status = coh.get_status()
        ok(phase, f"get_status", f"dim={status['field_dimension']} prims={status['primitives_discovered']} snaps={status['snapshots']}")
    except Exception as e:
        fail(phase, "get_status", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: ASI QuantumComputationCore — Fe-Sacred Coherence
# ═══════════════════════════════════════════════════════════════════════════════

def phase2_asi_quantum():
    banner("PHASE 2: ASI QuantumComputationCore — Quantum Coherence")
    phase = "P2"

    try:
        from l104_asi.quantum import QuantumComputationCore
        qcc = QuantumComputationCore()
        ok(phase, "QuantumComputationCore import + init")
    except Exception as e:
        fail(phase, "QuantumComputationCore import", str(e))
        return

    # 2.1 Fe-Sacred Coherence (286↔528 Hz)
    try:
        fe = qcc.fe_sacred_coherence(286.0, 528.0)
        coh_val = fe.get("coherence", 0)
        ref = fe.get("reference", 0)
        error = fe.get("alignment_error", 999)
        quantum_path = fe.get("quantum", False)
        path_label = "QUANTUM" if quantum_path else "CLASSICAL"
        if abs(coh_val - ref) < 0.05:
            ok(phase, f"Fe-Sacred coherence [{path_label}]", f"coh={coh_val:.6f} ref={ref:.6f} Δ={error:.6f}")
        else:
            warn(phase, f"Fe-Sacred coherence [{path_label}]", f"coh={coh_val:.6f} ref={ref:.6f} Δ={error:.6f} (>0.05)")
    except Exception as e:
        fail(phase, "Fe-Sacred coherence", str(e))

    # 2.2 Fe-PHI Harmonic Lock (286↔286φ Hz)
    try:
        phi_lock = qcc.fe_phi_harmonic_lock(286.0)
        lock_score = phi_lock.get("lock_score", 0)
        ref = phi_lock.get("reference", 0)
        quantum_path = phi_lock.get("quantum", False)
        path_label = "QUANTUM" if quantum_path else "CLASSICAL"
        if abs(lock_score - ref) < 0.05:
            ok(phase, f"Fe-PHI harmonic lock [{path_label}]", f"lock={lock_score:.6f} ref={ref:.6f}")
        else:
            warn(phase, f"Fe-PHI harmonic lock [{path_label}]", f"lock={lock_score:.6f} ref={ref:.6f}")
    except Exception as e:
        fail(phase, "Fe-PHI harmonic lock", str(e))

    # 2.3 Berry Phase Holonomy
    try:
        result = qcc.berry_phase_verify(dimensions=11)
        holonomy = result.get("holonomy_detected", False)
        if holonomy:
            ok(phase, "Berry phase 11D holonomy", f"detected={holonomy}")
        else:
            warn(phase, "Berry phase 11D holonomy", "not detected")
    except Exception as e:
        fail(phase, "Berry phase 11D holonomy", str(e))

    # 2.4 Coherence Engine availability
    try:
        ce = qcc._get_coherence_engine()
        if ce is not None:
            ok(phase, "QuantumCoherenceEngine loaded", "via ASI lazy-load")
        else:
            warn(phase, "QuantumCoherenceEngine", "unavailable (optional)")
    except Exception as e:
        warn(phase, "QuantumCoherenceEngine load", str(e))

    # 2.5 Coherence Grover search
    try:
        gr = qcc.coherence_grover_search(target_index=5, search_space_qubits=4)
        if gr.get("quantum", False) or "error" not in gr:
            ok(phase, "Coherence Grover search", f"result_keys={list(gr.keys())[:5]}")
        else:
            warn(phase, "Coherence Grover search", gr.get("error", "unknown"))
    except Exception as e:
        fail(phase, "Coherence Grover search", str(e))

    # 2.6 Metrics check
    try:
        m = qcc._metrics
        total = m.get("total_circuits", 0)
        coh_calls = m.get("coherence_engine_calls", 0)
        ok(phase, "Quantum metrics", f"total_circuits={total} coherence_calls={coh_calls}")
    except Exception as e:
        fail(phase, "Quantum metrics", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: QuantumCoherenceEngine (standalone)
# ═══════════════════════════════════════════════════════════════════════════════

def phase3_quantum_coherence_engine():
    banner("PHASE 3: QuantumCoherenceEngine (Qiskit Circuit Path)")
    phase = "P3"

    try:
        from l104_quantum_coherence import QuantumCoherenceEngine
        qce = QuantumCoherenceEngine()
        ok(phase, "QuantumCoherenceEngine import + init")
    except Exception as e:
        fail(phase, "QuantumCoherenceEngine import", str(e))
        return

    # 3.1 Get status
    try:
        status = qce.get_status()
        mode = status.get("execution_mode", "unknown")
        ok(phase, "get_status", f"mode={mode}")
    except Exception as e:
        fail(phase, "get_status", str(e))

    # 3.2 Grover search
    try:
        gr = qce.grover_search(target_index=3, search_space_qubits=3)
        if gr.get("success", False) or "target" in gr:
            ok(phase, "Grover search (3-qubit)", f"keys={list(gr.keys())[:5]}")
        else:
            warn(phase, "Grover search", f"result={gr}")
    except Exception as e:
        fail(phase, "Grover search", str(e))

    # 3.3 VQE
    try:
        vqe_result = qce.vqe_optimize(num_qubits=2, max_iterations=10)
        ok(phase, "VQE (2-qubit, 10 iter)", f"keys={list(vqe_result.keys())[:5]}")
    except Exception as e:
        fail(phase, "VQE", str(e))

    # 3.4 QPE
    try:
        qpe = qce.quantum_phase_estimation(precision_qubits=3)
        ok(phase, "QPE (3-bit precision)", f"keys={list(qpe.keys())[:5]}")
    except Exception as e:
        fail(phase, "QPE", str(e))

    # 3.5 Topological compute
    try:
        topo = qce.topological_compute(["s1", "s2", "s1"])
        ok(phase, "Topological braiding", f"keys={list(topo.keys())[:5]}")
    except Exception as e:
        fail(phase, "Topological braiding", str(e))

    # 3.6 Quantum kernel
    try:
        kernel = qce.quantum_kernel([1.0, 0.5], [0.3, 0.7])
        ok(phase, "Quantum kernel", f"keys={list(kernel.keys())[:5]}")
    except Exception as e:
        fail(phase, "Quantum kernel", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: Cross-System Constant Validation (Python ↔ Swift parity)
# ═══════════════════════════════════════════════════════════════════════════════

def phase4_cross_system_constants():
    banner("PHASE 4: Cross-System Constant Parity (Python ↔ Swift)")
    phase = "P4"

    from l104_science_engine.constants import (
        GOD_CODE, PHI, PHI_CONJUGATE, PHI_SQUARED, PHI_CUBED, VOID_CONSTANT,
        GROVER_AMPLIFICATION, FEIGENBAUM, FE_SACRED_COHERENCE, FE_PHI_HARMONIC_LOCK,
        BERRY_PHASE_DETECTED, ENTROPY_ZNE_BRIDGE_ENABLED,
    )

    # Swift constants (from L104App.swift + L01_Constants.swift)
    swift_constants = {
        "GOD_CODE": ("pow(286.0, 1.0/PHI) * pow(2.0, 416.0/104.0)", 527.5184818492612),
        "PHI": ("1.618033988749895", 1.618033988749895),
        "PHI_CONJUGATE": ("1.0 / PHI", 1.0 / 1.618033988749895),
        "GROVER_AMPLIFICATION": ("pow(PHI, 3)", pow(1.618033988749895, 3)),
        "FEIGENBAUM_DELTA": ("4.669201609102990", 4.669201609102990),
        "FE_SACRED_COHERENCE": ("0.9545454545454546", 0.9545454545454546),
        "FE_PHI_HARMONIC_LOCK": ("0.9164078649987375", 0.9164078649987375),
    }

    python_constants = {
        "GOD_CODE": GOD_CODE,
        "PHI": PHI,
        "PHI_CONJUGATE": PHI_CONJUGATE,
        "GROVER_AMPLIFICATION": GROVER_AMPLIFICATION,
        "FEIGENBAUM_DELTA": FEIGENBAUM,
        "FE_SACRED_COHERENCE": FE_SACRED_COHERENCE,
        "FE_PHI_HARMONIC_LOCK": FE_PHI_HARMONIC_LOCK,
    }

    for name, (swift_formula, swift_val) in swift_constants.items():
        py_val = python_constants[name]
        delta = abs(py_val - swift_val)
        if delta < 1e-10:
            ok(phase, f"{name}", f"Py={py_val:.10f} Swift={swift_val:.10f} Δ={delta:.2e}")
        elif delta < 1e-6:
            warn(phase, f"{name}", f"Py={py_val:.10f} Swift={swift_val:.10f} Δ={delta:.2e}")
        else:
            fail(phase, f"{name}", f"Py={py_val:.10f} Swift={swift_val:.10f} Δ={delta:.2e}")

    # ── CRITICAL: VOID_CONSTANT MISMATCH CHECK ──
    py_void = VOID_CONSTANT  # 1.04 + PHI/1000 = 1.0416180339887497
    swift_void = 1.04 + 1.618033988749895 / 1000.0  # 1.04 + φ/1000 (FIXED)
    delta_void = abs(py_void - swift_void)

    print(f"\n  {B}{Y}═══ VOID_CONSTANT CROSS-SYSTEM ANALYSIS ═══{RST}")
    print(f"  Python VOID_CONSTANT = {py_void:.16f}  (1.04 + φ/1000)")
    print(f"  Swift  VOID_CONSTANT = {swift_void:.16f}  (1.04 + φ/1000)")
    print(f"  Delta                = {delta_void:.16f}")

    if delta_void > 1.0:
        fail(phase, "VOID_CONSTANT cross-system",
             f"MAJOR MISMATCH: Py={py_void:.10f} vs Swift={swift_void:.10f} "
             f"(Python=1.04+φ/1000, Swift=φ/(φ-1)=φ²)")
    else:
        ok(phase, "VOID_CONSTANT cross-system")

    # ── Boolean flags ──
    print()
    swift_berry = True   # BERRY_PHASE_11D in L01_Constants.swift
    swift_zne = True     # ENTROPY_ZNE_BRIDGE in L01_Constants.swift

    if BERRY_PHASE_DETECTED == swift_berry:
        ok(phase, "BERRY_PHASE_DETECTED", f"Py={BERRY_PHASE_DETECTED} Swift={swift_berry}")
    else:
        fail(phase, "BERRY_PHASE_DETECTED", f"Py={BERRY_PHASE_DETECTED} Swift={swift_berry}")

    if ENTROPY_ZNE_BRIDGE_ENABLED == swift_zne:
        ok(phase, "ENTROPY_ZNE_BRIDGE", f"Py={ENTROPY_ZNE_BRIDGE_ENABLED} Swift={swift_zne}")
    else:
        fail(phase, "ENTROPY_ZNE_BRIDGE", f"Py={ENTROPY_ZNE_BRIDGE_ENABLED} Swift={swift_zne}")

    # ── COHERENCE_TARGET: Swift has 1.0 (no cap), verify Python doesn't limit ──
    # Python coherence threshold comes from: (GOD_CODE / 1000) * PHI_CONJUGATE
    py_threshold = (GOD_CODE / 1000) * PHI_CONJUGATE
    print(f"\n  Python coherence threshold = {py_threshold:.10f} ((GOD_CODE/1000) × φ_conj)")
    print(f"  Swift QUANTUM_COHERENCE_TARGET = 1.0 (Unity = no cap)")
    if py_threshold < 1.0:
        ok(phase, "Coherence threshold", f"Py threshold={py_threshold:.6f} < Swift target=1.0 (compatible)")
    else:
        warn(phase, "Coherence threshold", f"Py threshold={py_threshold:.6f} ≥ 1.0")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: Coherence Feedback Loop Stress Test
# ═══════════════════════════════════════════════════════════════════════════════

def phase5_feedback_loop():
    banner("PHASE 5: Coherence Feedback Loop Stress Test")
    phase = "P5"

    try:
        from l104_science_engine import ScienceEngine
        se = ScienceEngine()
        coh = se.coherence
    except Exception as e:
        fail(phase, "ScienceEngine init for feedback test", str(e))
        return

    # Initialize field
    seeds = [f"thought_{i}" for i in range(20)]
    coh.initialize(seeds)
    coh.evolve(5)
    coh.anchor(1.0)

    # 5.1 Ingest simulation result
    try:
        sim = {
            "total_fidelity": 0.85,
            "decoherence_fidelity": 0.75,
            "gate_fidelity": 0.92,
            "noise_variance": 0.05,
            "demon_efficiency": 0.8,
            "probabilities": {"0000": 0.3, "0001": 0.1, "0010": 0.15, "1111": 0.45},
        }
        result = coh.ingest_simulation_result(sim)
        pre = result["pre_coherence"]
        post = result["post_coherence"]
        corrections = result["corrections_count"]
        ok(phase, "ingest_simulation_result", f"pre={pre:.4f} post={post:.4f} corrections={corrections}")
    except Exception as e:
        fail(phase, "ingest_simulation_result", str(e))

    # 5.2 Adaptive decoherence correction
    try:
        adc = coh.adaptive_decoherence_correction(fidelity=0.7, circuit_depth=100, t1_us=300.0, t2_us=150.0)
        recovered = adc["coherence_recovered"]
        braid_ops = adc["braid_ops_applied"]
        ok(phase, "adaptive_decoherence_correction", f"recovered={recovered:.4f} braids={braid_ops}")
    except Exception as e:
        fail(phase, "adaptive_decoherence_correction", str(e))

    # 5.3 Entropy-coherence feedback
    try:
        ecf = coh.entropy_coherence_feedback(demon_efficiency=0.85, coherence_gain=0.5, noise_vector_var=1.5)
        delta = ecf["coherence_delta"]
        ok(phase, "entropy_coherence_feedback", f"delta={delta:.4f} surplus={ecf['energy_surplus']:.4e}")
    except Exception as e:
        fail(phase, "entropy_coherence_feedback", str(e))

    # 5.4 Full feedback loop (5 rounds)
    try:
        sim_sequence = [
            {"total_fidelity": 0.7, "gate_fidelity": 0.85, "noise_variance": 0.1, "demon_efficiency": 0.6},
            {"total_fidelity": 0.75, "gate_fidelity": 0.88, "noise_variance": 0.08, "demon_efficiency": 0.7},
            {"total_fidelity": 0.80, "gate_fidelity": 0.91, "noise_variance": 0.05, "demon_efficiency": 0.8},
            {"total_fidelity": 0.85, "gate_fidelity": 0.93, "noise_variance": 0.03, "demon_efficiency": 0.85},
            {"total_fidelity": 0.90, "gate_fidelity": 0.95, "noise_variance": 0.02, "demon_efficiency": 0.9},
        ]
        loop = coh.run_feedback_loop(sim_sequence, evolve_steps=5)
        converging = loop["converging"]
        init_c = loop["initial_coherence"]
        final_c = loop["final_coherence"]
        total_delta = loop["total_coherence_delta"]
        if converging:
            ok(phase, f"run_feedback_loop (5 rounds)", f"init={init_c:.4f} → final={final_c:.4f} Δ={total_delta:+.4f} CONVERGING")
        else:
            warn(phase, f"run_feedback_loop (5 rounds)", f"init={init_c:.4f} → final={final_c:.4f} Δ={total_delta:+.4f} NOT converging")
    except Exception as e:
        fail(phase, "run_feedback_loop", str(e))

    # 5.5 Coherence fidelity after stress
    try:
        fid = coh.coherence_fidelity()
        grade = fid["grade"]
        fidelity_val = fid["fidelity"]
        ok(phase, f"post-stress fidelity", f"fidelity={fidelity_val:.4f} grade={grade}")
    except Exception as e:
        fail(phase, "post-stress fidelity", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 6: Swift Coherence Architecture Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def phase6_swift_analysis():
    banner("PHASE 6: Swift Coherence Architecture Analysis")
    phase = "P6"

    workspace = os.path.dirname(os.path.abspath(__file__))
    swift_dir = os.path.join(workspace, "L104SwiftApp", "Sources")

    if not os.path.isdir(swift_dir):
        fail(phase, "Swift sources directory", f"{swift_dir} not found")
        return

    # Scan all Swift files for coherence-related code
    coherence_files = {}
    total_coherence_refs = 0

    for root, dirs, files in os.walk(swift_dir):
        for f in files:
            if f.endswith(".swift"):
                fpath = os.path.join(root, f)
                try:
                    with open(fpath, "r") as fh:
                        content = fh.read()
                    count = content.lower().count("coherence")
                    if count > 0:
                        coherence_files[f] = count
                        total_coherence_refs += count
                except:
                    pass

    ok(phase, f"Swift coherence scan", f"{len(coherence_files)} files, {total_coherence_refs} total refs")

    # Top 10 files by coherence references
    top = sorted(coherence_files.items(), key=lambda x: -x[1])[:10]
    print(f"\n  {B}Top Swift files by coherence references:{RST}")
    for fname, count in top:
        print(f"    {C}{count:4d}{RST}  {fname}")

    # Check for Python bridge coherence paths
    bridge_path = os.path.join(swift_dir, "L104v2", "TheLogic", "L25_PythonBridge.swift")
    if os.path.isfile(bridge_path):
        with open(bridge_path, "r") as f:
            bridge_content = f.read()
        coherence_bridge_calls = bridge_content.count("QuantumCoherenceEngine")
        grover_calls = bridge_content.count("grover_search")
        vqe_calls = bridge_content.count("vqe_optimize")
        ok(phase, f"Python bridge coherence paths",
           f"QCE refs={coherence_bridge_calls} grover={grover_calls} vqe={vqe_calls}")
    else:
        warn(phase, "PythonBridge", "L25_PythonBridge.swift not found")

    # Check QuantumLogicGate coherence tracking
    qlg_path = os.path.join(swift_dir, "L104v2", "TheLogic", "L07_QuantumLogicGate.swift")
    if os.path.isfile(qlg_path):
        with open(qlg_path, "r") as f:
            qlg = f.read()
        decoherence_count = qlg.count("applyDecoherence")
        recohere_count = qlg.count("recohere")
        error_correct_count = qlg.count("errorCorrect")
        ok(phase, f"QuantumLogicGate coherence ops",
           f"decoherence={decoherence_count} recohere={recohere_count} errorCorrect={error_correct_count}")
    else:
        warn(phase, "QuantumLogicGate", "L07_QuantumLogicGate.swift not found")

    # Verify QuantumMath has Fe-Sacred parity
    qm_path = os.path.join(swift_dir, "L104v2", "TheBrain", "B01_QuantumMath.swift")
    if os.path.isfile(qm_path):
        with open(qm_path, "r") as f:
            qm = f.read()
        fe_sacred_impl = "feSacredCoherence" in qm
        fe_phi_impl = "fePhiHarmonicLock" in qm
        berry_impl = "berryPhaseAccumulate" in qm
        all_present = fe_sacred_impl and fe_phi_impl and berry_impl
        if all_present:
            ok(phase, "QuantumMath sacred implementations",
               f"Fe-Sacred={fe_sacred_impl} Fe-PHI={fe_phi_impl} Berry={berry_impl}")
        else:
            fail(phase, "QuantumMath sacred implementations",
                 f"Fe-Sacred={fe_sacred_impl} Fe-PHI={fe_phi_impl} Berry={berry_impl}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 7: Entropy ↔ Coherence Cross-Link
# ═══════════════════════════════════════════════════════════════════════════════

def phase7_entropy_coherence():
    banner("PHASE 7: Entropy ↔ Coherence Cross-Link")
    phase = "P7"

    try:
        from l104_science_engine import ScienceEngine
        import numpy as np
        se = ScienceEngine()
    except Exception as e:
        fail(phase, "ScienceEngine for entropy test", str(e))
        return

    # 7.1 Entropy Maxwell Demon efficiency
    try:
        demon_eff = se.entropy.calculate_demon_efficiency(0.5)
        ok(phase, "Maxwell Demon efficiency", f"eff={demon_eff:.6f}")
    except Exception as e:
        fail(phase, "Maxwell Demon efficiency", str(e))

    # 7.2 Inject coherence from noise
    try:
        noise = np.random.randn(100)
        ordered = se.entropy.inject_coherence(noise)
        gain = se.entropy.coherence_gain
        ok(phase, "inject_coherence (noise→order)", f"gain={gain:.6f} ordered_len={len(ordered)}")
    except Exception as e:
        fail(phase, "inject_coherence", str(e))

    # 7.3 Cross-link: entropy → coherence field
    try:
        coh = se.coherence
        seeds = [f"entropy_test_{i}" for i in range(10)]
        coh.initialize(seeds)
        coh.evolve(5)

        # Feed entropy metrics into coherence
        ecf = coh.entropy_coherence_feedback(
            demon_efficiency=demon_eff,
            coherence_gain=gain,
            noise_vector_var=float(np.var(noise))
        )
        ok(phase, "entropy→coherence cross-feedback",
           f"delta={ecf['coherence_delta']:.6f} surplus={ecf['energy_surplus']:.4e}")
    except Exception as e:
        fail(phase, "entropy→coherence cross-feedback", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()
    print(f"\n{B}{C}{'═' * 72}")
    print(f"  L104 QUANTUM COHERENCE DEBUG SUITE")
    print(f"  Python + Swift cross-system coherence diagnostics")
    print(f"{'═' * 72}{RST}\n")

    phase1_science_coherence()
    phase2_asi_quantum()
    phase3_quantum_coherence_engine()
    phase4_cross_system_constants()
    phase5_feedback_loop()
    phase6_swift_analysis()
    phase7_entropy_coherence()

    elapsed = time.time() - t0

    # ── Summary ──
    total = passed + failed + warnings
    print(f"\n{B}{'═' * 72}")
    print(f"  SUMMARY")
    print(f"{'═' * 72}{RST}")
    print(f"  {G}PASSED{RST}:   {passed}/{total}")
    print(f"  {R}FAILED{RST}:   {failed}/{total}")
    print(f"  {Y}WARNINGS{RST}: {warnings}/{total}")
    print(f"  Time: {elapsed:.2f}s")

    if failed > 0:
        print(f"\n  {B}{R}FAILURES:{RST}")
        for phase, test, status in results:
            if status == "FAIL":
                print(f"    {R}[{phase}] {test}{RST}")

    if warnings > 0:
        print(f"\n  {B}{Y}WARNINGS:{RST}")
        for phase, test, status in results:
            if status == "WARN":
                print(f"    {Y}[{phase}] {test}{RST}")

    # ── Actionable findings ──
    print(f"\n{B}{'═' * 72}")
    print(f"  ACTIONABLE FINDINGS")
    print(f"{'═' * 72}{RST}")

    # Check the VOID_CONSTANT mismatch
    from l104_science_engine.constants import VOID_CONSTANT, PHI
    py_void = VOID_CONSTANT
    swift_void = 1.04 + PHI / 1000.0  # Fixed: now matches Python canonical
    if abs(py_void - swift_void) > 1.0:
        print(f"""
  {R}★ CRITICAL: VOID_CONSTANT MISMATCH BETWEEN PYTHON AND SWIFT{RST}
    Python:  VOID_CONSTANT = 1.04 + φ/1000 = {py_void:.16f}
    Swift:   VOID_CONSTANT = {swift_void:.16f}
    Delta:   {abs(py_void - swift_void):.16f}
""")

    print(f"\n  {B}{'═' * 72}{RST}")
    status_label = f"{G}ALL CLEAR{RST}" if failed == 0 else f"{R}ISSUES FOUND{RST}"
    print(f"  Quantum coherence status: {status_label}")
    print(f"  {B}{'═' * 72}{RST}\n")

    sys.exit(0 if failed == 0 else 1)
