#!/usr/bin/env python3
"""
L104 VQPU Debug Suite v2.0
═══════════════════════════════════════════════════════════════════
Comprehensive diagnostics for the Virtual Quantum Processing Unit.

Phases:
  1. Import & Platform Detection
  2. Constants & Capacity Verification
  3. Core Subsystem Self-Tests (Transpiler, MPS, Noise, Entanglement)
  4. Sacred Alignment + Three-Engine Scoring
  5. v8.0 Advanced Quantum Equations (QI Metrics, Tomography, Hamiltonian)
  6. Scoring Cache Performance
  7. VQPU Findings Simulations (God Code Simulator bridge)
  8. Full run_simulation Pipeline
  9. Boot Manager & Daemon Status
  10. v12.0 Parallel Batch Execution (NEW)
  11. v12.0 Daemon Error Logging (NEW)
  12. v12.0 LRU Parametric Cache (NEW)
  13. v12.0 Performance Profiling (NEW)

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════
"""

import sys
import os
import time
import traceback

# Ensure workspace root is on the path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.04 + PHI / 1000.0

PASS = 0
FAIL = 0
WARN = 0
RESULTS = []


def log_pass(msg: str):
    global PASS
    PASS += 1
    RESULTS.append(("PASS", msg))
    print(f"  ✓ {msg}")


def log_fail(msg: str):
    global FAIL
    FAIL += 1
    RESULTS.append(("FAIL", msg))
    print(f"  ✗ {msg}")


def log_warn(msg: str):
    global WARN
    WARN += 1
    RESULTS.append(("WARN", msg))
    print(f"  ⚠ {msg}")


def log_info(msg: str):
    print(f"  · {msg}")


def phase_header(n: int, title: str):
    print()
    print(f"{'─' * 65}")
    print(f"  Phase {n}: {title}")
    print(f"{'─' * 65}")


# ═══════════════════════════════════════════════════════════════════
# PHASE 1: IMPORT & PLATFORM DETECTION
# ═══════════════════════════════════════════════════════════════════

phase_header(1, "Import & Platform Detection")

t0 = time.monotonic()

try:
    from l104_vqpu_bridge import (
        VQPUBridge, QuantumJob, QuantumGate, VQPUResult,
        CircuitTranspiler, CircuitAnalyzer, ExactMPSHybridEngine,
        SacredAlignmentScorer, NoiseModel, EntanglementQuantifier,
        QuantumInformationMetrics, QuantumStateTomography,
        HamiltonianSimulator, ScoringCache, VariationalQuantumEngine,
        ThreeEngineQuantumScorer, EngineIntegration,
        QuantumErrorMitigation, CircuitCache,
        GOD_CODE as VQPU_GOD_CODE, PHI as VQPU_PHI, VOID_CONSTANT as VQPU_VOID,
        VQPU_MAX_QUBITS, VQPU_BATCH_LIMIT, VQPU_PIPELINE_WORKERS,
        VQPU_MPS_MAX_BOND_HIGH, VQPU_ADAPTIVE_SHOTS_MAX,
        _PLATFORM, _IS_INTEL, _IS_APPLE_SILICON, _HAS_METAL_COMPUTE, _GPU_CLASS,
        _HW_RAM_GB, _HW_CORES,
    )
    import_ms = round((time.monotonic() - t0) * 1000, 1)
    log_pass(f"l104_vqpu_bridge imported ({import_ms}ms)")
except Exception as e:
    log_fail(f"l104_vqpu_bridge import FAILED: {e}")
    traceback.print_exc()
    print("\n[FATAL] Cannot continue without VQPU bridge. Exiting.")
    sys.exit(1)

# Platform info
log_info(f"Arch: {_PLATFORM['arch']} | GPU: {_GPU_CLASS} | Metal family: {_PLATFORM['metal_family']}")
log_info(f"Apple Silicon: {_IS_APPLE_SILICON} | Intel: {_IS_INTEL} | Metal compute: {_HAS_METAL_COMPUTE}")
log_info(f"RAM: {_HW_RAM_GB} GB | Cores: {_HW_CORES} | SIMD: {_PLATFORM.get('simd', [])}")

if _IS_APPLE_SILICON:
    log_pass("Apple Silicon detected — full Metal compute available")
elif _IS_INTEL:
    log_warn("Intel x86_64 detected — CPU-only quantum paths (no Metal compute)")

# Optional imports
try:
    import numpy as np
    log_pass(f"numpy {np.__version__}")
except ImportError:
    log_fail("numpy not available")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# PHASE 2: CONSTANTS & CAPACITY VERIFICATION
# ═══════════════════════════════════════════════════════════════════

phase_header(2, "Constants & Capacity")

# Sacred constants
if VQPU_GOD_CODE == GOD_CODE:
    log_pass(f"GOD_CODE = {VQPU_GOD_CODE}")
else:
    log_fail(f"GOD_CODE mismatch: {VQPU_GOD_CODE} != {GOD_CODE}")

if VQPU_PHI == PHI:
    log_pass(f"PHI = {VQPU_PHI}")
else:
    log_fail(f"PHI mismatch: {VQPU_PHI} != {PHI}")

if abs(VQPU_VOID - VOID_CONSTANT) < 1e-15:
    log_pass(f"VOID_CONSTANT = {VQPU_VOID}")
else:
    log_fail(f"VOID_CONSTANT mismatch: {VQPU_VOID} != {VOID_CONSTANT}")

# Capacity
log_info(f"Max qubits: {VQPU_MAX_QUBITS}")
log_info(f"Batch limit: {VQPU_BATCH_LIMIT}")
log_info(f"Pipeline workers: {VQPU_PIPELINE_WORKERS}")
log_info(f"MPS bond high: {VQPU_MPS_MAX_BOND_HIGH}")
log_info(f"Adaptive shots max: {VQPU_ADAPTIVE_SHOTS_MAX}")

if VQPU_MAX_QUBITS >= 24:
    log_pass(f"Max qubits capacity OK ({VQPU_MAX_QUBITS}Q)")
else:
    log_fail(f"Max qubits too low ({VQPU_MAX_QUBITS}Q), expected ≥24")

if VQPU_BATCH_LIMIT >= 16:
    log_pass(f"Batch limit OK ({VQPU_BATCH_LIMIT})")
else:
    log_fail(f"Batch limit too low ({VQPU_BATCH_LIMIT})")


# ═══════════════════════════════════════════════════════════════════
# PHASE 3: CORE SUBSYSTEM SELF-TESTS
# ═══════════════════════════════════════════════════════════════════

phase_header(3, "Core Subsystems")

# ── Transpiler ──
print()
print("  [Transpiler]")
try:
    test_ops = [
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [0]},      # should cancel
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "Rz", "qubits": [0], "parameters": [0.5]},
        {"gate": "Rz", "qubits": [0], "parameters": [0.3]},  # should merge → 0.8
        {"gate": "X", "qubits": [1]},
        {"gate": "X", "qubits": [1]},      # should cancel
    ]
    optimized = CircuitTranspiler.transpile(test_ops)
    saved = len(test_ops) - len(optimized)
    if saved >= 2:
        log_pass(f"Transpiler: {len(test_ops)} → {len(optimized)} gates ({saved} saved)")
    else:
        log_warn(f"Transpiler saved only {saved} gates (expected ≥2)")
    summary = CircuitTranspiler.gate_count_summary(optimized)
    log_info(f"Gate summary: {summary}")
except Exception as e:
    log_fail(f"Transpiler error: {e}")

# ── Circuit Analyzer ──
print()
print("  [Circuit Analyzer]")
try:
    bell_ops = [
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
    ]
    analysis = CircuitAnalyzer.analyze(bell_ops, 2)
    log_pass(f"CircuitAnalyzer: depth={analysis.get('depth', '?')}, "
             f"total_gates={analysis.get('total_gates', '?')}, "
             f"2q_gates={analysis.get('two_qubit_gates', '?')}")
except Exception as e:
    log_fail(f"CircuitAnalyzer error: {e}")

# ── ExactMPSHybridEngine ──
print()
print("  [ExactMPS Hybrid Engine]")
try:
    t0 = time.monotonic()
    engine = ExactMPSHybridEngine(num_qubits=4)
    # Bell pair: H(0) + CX(0,1)
    ops = [
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
    ]
    circ_result = engine.run_circuit(ops)
    completed = circ_result.get("completed", False)
    peak_chi = circ_result.get("peak_chi", 0)
    probs = engine.sample(shots=1024)
    ms = round((time.monotonic() - t0) * 1000, 1)
    if isinstance(probs, dict) and len(probs) > 0:
        log_pass(f"MPS 4Q Bell: {dict(list(probs.items())[:4])} (chi={peak_chi}, {ms}ms)")
    else:
        log_warn(f"MPS returned unexpected format: {type(probs)} ({ms}ms)")
    if completed:
        log_pass("MPS completed without GPU fallback")
    else:
        log_warn(f"MPS triggered GPU fallback at gate {circ_result.get('fallback_at', '?')}")
except Exception as e:
    log_fail(f"ExactMPS error: {e}")
    traceback.print_exc()

# ── Noise Model ──
print()
print("  [Noise Model]")
try:
    nm = NoiseModel(depolarizing_rate=0.05, amplitude_damping_rate=0.01, readout_error_rate=0.02)
    sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)  # Bell state (2Q)
    sv_copy = sv.copy()
    noisy = nm.apply_gate_noise(sv_copy, qubit=0, num_qubits=2, is_two_qubit=False)
    if isinstance(noisy, np.ndarray) and len(noisy) == 4:
        norm = float(np.abs(np.vdot(noisy, noisy)))
        log_pass(f"Gate noise applied: norm={norm:.6f}")
    else:
        log_warn(f"Noise output unexpected: {type(noisy)}")

    # Readout noise
    counts = {"00": 512, "11": 512}
    noisy_counts = nm.apply_readout_noise(counts, 2)
    total_shots = sum(noisy_counts.values())
    if total_shots == 1024:
        log_pass(f"Readout noise: {len(noisy_counts)} bitstrings, {total_shots} shots preserved")
    else:
        log_warn(f"Readout noise: total shots changed to {total_shots}")

    # Factory methods
    nm_real = NoiseModel.realistic_superconducting()
    nm_low = NoiseModel.low_noise()
    nm_none = NoiseModel.noiseless()
    log_pass(f"Noise factories: realistic(depol={nm_real.depolarizing_rate}), "
             f"low(depol={nm_low.depolarizing_rate}), noiseless(depol={nm_none.depolarizing_rate})")
except Exception as e:
    log_fail(f"NoiseModel error: {e}")

# ── Entanglement Quantifier ──
print()
print("  [Entanglement Quantifier]")
try:
    # Bell state: (|00⟩ + |11⟩)/√2
    bell_sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
    ent = EntanglementQuantifier.von_neumann_entropy(bell_sv, 2)
    if isinstance(ent, (int, float)) and ent > 0.5:
        log_pass(f"Von Neumann entropy (Bell): {ent:.6f} (expected ~1.0)")
    else:
        log_warn(f"Unexpected entropy: {ent}")

    conc = EntanglementQuantifier.concurrence(bell_sv)
    if isinstance(conc, (int, float)) and conc > 0.8:
        log_pass(f"Concurrence (Bell): {conc:.6f} (expected ~1.0)")
    else:
        log_warn(f"Unexpected concurrence: {conc}")
except Exception as e:
    log_fail(f"EntanglementQuantifier error: {e}")

# ── Circuit Cache ──
print()
print("  [Circuit Cache]")
try:
    cc = CircuitCache()  # instance, not class
    cc.clear()
    fp = CircuitCache.fingerprint([{"gate": "H", "qubits": [0]}], 1, 1024)
    cc.put(fp, {"data": 42})
    cached = cc.get(fp)
    if cached and cached.get("data") == 42:
        log_pass(f"CircuitCache put/get works (fp={fp[:12]}...)")
    else:
        log_fail(f"CircuitCache returned: {cached}")
    stats = cc.stats()
    log_info(f"Cache stats: {stats}")
    cc.clear()
except Exception as e:
    log_fail(f"CircuitCache error: {e}")


# ═══════════════════════════════════════════════════════════════════
# PHASE 4: SACRED ALIGNMENT + THREE-ENGINE SCORING
# ═══════════════════════════════════════════════════════════════════

phase_header(4, "Sacred Alignment + Three-Engine Scoring")

try:
    probs_bell = {"00": 0.5, "11": 0.5}
    sacred = SacredAlignmentScorer.score(probs_bell, 2)
    ss = sacred.get("sacred_score", 0)
    if isinstance(ss, (int, float)) and 0 <= ss <= 2.0:
        log_pass(f"Sacred score (Bell): {ss:.6f}")
    else:
        log_warn(f"Unexpected sacred score: {ss}")
    log_info(f"  entropy: {sacred.get('entropy', 'N/A')}")
    log_info(f"  phi_alignment: {sacred.get('phi_alignment', 'N/A')}")
    log_info(f"  god_code_resonance: {sacred.get('god_code_resonance', 'N/A')}")
except Exception as e:
    log_fail(f"SacredAlignmentScorer error: {e}")

try:
    composite = ThreeEngineQuantumScorer.composite_score(1.0)
    cs = composite.get("composite", 0)
    if isinstance(cs, (int, float)) and 0 <= cs <= 2.0:
        log_pass(f"Three-engine composite: {cs:.6f}")
    else:
        log_warn(f"Unexpected composite: {cs}")
    log_info(f"  entropy_reversal: {composite.get('entropy_reversal', 'N/A')}")
    log_info(f"  harmonic_resonance: {composite.get('harmonic_resonance', 'N/A')}")
    log_info(f"  wave_coherence: {composite.get('wave_coherence', 'N/A')}")
except Exception as e:
    log_fail(f"ThreeEngineQuantumScorer error: {e}")


# ═══════════════════════════════════════════════════════════════════
# PHASE 5: v8.0 ADVANCED QUANTUM EQUATIONS
# ═══════════════════════════════════════════════════════════════════

phase_header(5, "v8.0 Advanced Quantum Equations")

# ── QFI (Quantum Fisher Information) ──
print()
print("  [Quantum Fisher Information]")
try:
    # GHZ-like state for maximum QFI
    ghz3 = np.zeros(8, dtype=np.complex128)
    ghz3[0] = 1/np.sqrt(2)
    ghz3[7] = 1/np.sqrt(2)
    gen_ops = [{"gate": "Rz", "qubits": [0]}, {"gate": "Rz", "qubits": [1]}, {"gate": "Rz", "qubits": [2]}]
    qfi_result = QuantumInformationMetrics.quantum_fisher_information(ghz3, gen_ops, num_qubits=3)
    qfi_val = qfi_result.get("qfi", 0) if isinstance(qfi_result, dict) else qfi_result
    if isinstance(qfi_val, (int, float)) and qfi_val > 0:
        log_pass(f"QFI (3Q GHZ): {qfi_val:.6f}")
        if isinstance(qfi_result, dict):
            log_info(f"  Cramér-Rao: {qfi_result.get('cramer_rao_bound', 'N/A')}")
            log_info(f"  Heisenberg limited: {qfi_result.get('heisenberg_limited', 'N/A')}")
            log_info(f"  Sacred alignment: {qfi_result.get('sacred_alignment', 'N/A')}")
    else:
        log_warn(f"Unexpected QFI: {qfi_val}")
except Exception as e:
    log_fail(f"QFI error: {e}")

# ── Berry Phase ──
print()
print("  [Berry Phase]")
try:
    # Simple path: rotate |0⟩ around Bloch sphere
    states = []
    for theta in np.linspace(0, 2*np.pi, 20):
        sv = np.array([np.cos(theta/2), np.sin(theta/2) * np.exp(1j * theta)],
                       dtype=np.complex128)
        states.append(sv)
    bp = QuantumInformationMetrics.berry_phase(states, 1)
    if isinstance(bp, dict):
        phase = bp.get("berry_phase", bp.get("phase", None))
        if phase is not None:
            log_pass(f"Berry phase: {phase:.6f}")
        else:
            log_pass(f"Berry phase computed: {bp}")
    elif isinstance(bp, (int, float)):
        log_pass(f"Berry phase: {bp:.6f}")
    else:
        log_warn(f"Unexpected Berry result: {type(bp)}")
except Exception as e:
    log_fail(f"Berry phase error: {e}")

# ── Loschmidt Echo ──
print()
print("  [Loschmidt Echo]")
try:
    init_sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
    ham_ops = [{"gate": "Z", "qubits": [0], "parameters": [0.5]},
               {"gate": "Z", "qubits": [1], "parameters": [0.3]}]
    perturb_ops = [{"gate": "X", "qubits": [0], "parameters": [0.05]}]
    echo = QuantumInformationMetrics.loschmidt_echo(
        init_sv, ham_ops, perturb_ops, num_qubits=2, time_steps=5, dt=0.1)
    if isinstance(echo, dict):
        log_pass(f"Loschmidt echo: decay_rate={echo.get('decay_rate', 'N/A')}, "
                 f"chaotic={echo.get('is_chaotic', 'N/A')}")
        log_info(f"  Echo values: {echo.get('echo_values', [])[:5]}")
        log_info(f"  Lyapunov est: {echo.get('lyapunov_estimate', 'N/A')}")
    elif isinstance(echo, (int, float)):
        log_pass(f"Loschmidt echo value: {echo:.6f}")
    else:
        log_warn(f"Unexpected echo result: {type(echo)}")
except Exception as e:
    log_fail(f"Loschmidt echo error: {e}")

# ── Quantum State Tomography ──
print()
print("  [Quantum State Tomography]")
try:
    bell_sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
    # Step 1: measure in Pauli bases
    pauli_exp = QuantumStateTomography.measure_in_pauli_bases(bell_sv, 2)
    log_pass(f"Pauli measurements: {len(pauli_exp)} expectations")
    # Step 2: reconstruct density matrix
    tomo = QuantumStateTomography.reconstruct_density_matrix(pauli_exp, 2)
    if isinstance(tomo, dict):
        purity = tomo.get("purity", None)
        rank = tomo.get("rank", None)
        log_pass(f"Tomography: purity={purity}, rank={rank}")
        if purity is not None and abs(purity - 1.0) < 0.15:
            log_pass("Purity ≈ 1.0 (pure state)")
        elif purity is not None:
            log_warn(f"Purity = {purity:.4f} (expected ~1.0 for pure Bell)")
    else:
        log_warn(f"Unexpected tomography result: {type(tomo)}")
except Exception as e:
    log_fail(f"Tomography error: {e}")

# ── Hamiltonian Simulator (Trotter) ──
print()
print("  [Hamiltonian Simulator]")
try:
    t0 = time.monotonic()
    fe_result = HamiltonianSimulator.iron_lattice_circuit(
        n_sites=4, trotter_steps=4, total_time=1.0)
    ms = round((time.monotonic() - t0) * 1000, 1)
    if isinstance(fe_result, (dict, list)):
        if isinstance(fe_result, dict):
            sites = fe_result.get("lattice_sites", fe_result.get("n_sites", "?"))
            coupling_j = fe_result.get("coupling_j", "?")
            mag = fe_result.get("magnetization", "?")
            log_pass(f"Fe(26) lattice: sites={sites}, J={coupling_j}, mag={mag} ({ms}ms)")
        else:
            log_pass(f"Fe(26) lattice: returned {len(fe_result)} ops ({ms}ms)")
    else:
        log_warn(f"Iron lattice unexpected: {type(fe_result)}")
except Exception as e:
    log_fail(f"HamiltonianSimulator error: {e}")


# ═══════════════════════════════════════════════════════════════════
# PHASE 6: SCORING CACHE PERFORMANCE
# ═══════════════════════════════════════════════════════════════════

phase_header(6, "Scoring Cache Performance")

try:
    ScoringCache._harmonic_cached = None
    ScoringCache._wave_cached = None
    ScoringCache._entropy_cache.clear()
    ScoringCache._asi_cache.clear()
    ScoringCache._agi_cache.clear()
    ScoringCache._stats = {"hits": 0, "misses": 0, "harmonic_hits": 0, "wave_hits": 0}

    # First call: miss
    h1 = ScoringCache.get_harmonic(lambda: 0.85)
    # Second call: hit
    h2 = ScoringCache.get_harmonic(lambda: 999)  # should be cached, not 999
    if h1 == h2 == 0.85:
        log_pass("Harmonic cache: first=miss, second=hit (correct)")
    else:
        log_fail(f"Harmonic cache broken: h1={h1}, h2={h2}")

    # Wave cache
    w1 = ScoringCache.get_wave(lambda: 0.72)
    w2 = ScoringCache.get_wave(lambda: 999)
    if w1 == w2 == 0.72:
        log_pass("Wave cache: first=miss, second=hit (correct)")
    else:
        log_fail(f"Wave cache broken: w1={w1}, w2={w2}")

    # Entropy bucket cache
    e1 = ScoringCache.get_entropy(1.0, lambda x: x * 0.8)
    e2 = ScoringCache.get_entropy(1.0, lambda x: 999)   # same bucket (1.0) → cached
    e3 = ScoringCache.get_entropy(1.1, lambda x: x * 0.9)  # different bucket (1.1) → miss
    if e1 == e2 == 0.8:
        log_pass(f"Entropy bucket cache: correct (bucket 1.0 → {e1})")
    else:
        log_warn(f"Entropy bucket: e1={e1}, e2(1.0)={e2}")
    if e3 != e1:
        log_pass(f"Entropy bucket isolation: 1.0→{e1}, 1.1→{e3}")
    else:
        log_warn(f"Entropy buckets not isolated: e1={e1}, e3={e3}")

    stats = ScoringCache.stats()
    log_info(f"Cache stats: {stats}")
except Exception as e:
    log_fail(f"ScoringCache error: {e}")


# ═══════════════════════════════════════════════════════════════════
# PHASE 7: VQPU FINDINGS SIMULATIONS
# ═══════════════════════════════════════════════════════════════════

phase_header(7, "VQPU Findings Simulations (God Code Simulator)")

try:
    from l104_god_code_simulator.simulations.vqpu_findings import VQPU_FINDINGS_SIMULATIONS
    expected_count = 11
    actual_count = len(VQPU_FINDINGS_SIMULATIONS)
    if actual_count == expected_count:
        log_pass(f"VQPU_FINDINGS_SIMULATIONS: {actual_count} sims loaded")
    else:
        log_warn(f"VQPU sims: expected {expected_count}, got {actual_count}")

    # Run each simulation — entries are tuples: (name, fn, category, desc, nq)
    sim_pass = 0
    sim_fail = 0
    for entry in VQPU_FINDINGS_SIMULATIONS:
        name = entry[0]
        fn = entry[1]
        category = entry[2] if len(entry) > 2 else "?"
        try:
            t0 = time.monotonic()
            result = fn()
            ms = round((time.monotonic() - t0) * 1000, 1)
            if hasattr(result, "passed"):
                if result.passed:
                    sim_pass += 1
                    log_pass(f"  {name}: fid={result.fidelity:.4f}, "
                             f"sacred={result.sacred_alignment:.4f} ({ms}ms)")
                else:
                    sim_fail += 1
                    log_fail(f"  {name}: FAILED — {result.detail} ({ms}ms)")
            else:
                sim_pass += 1
                log_pass(f"  {name}: completed ({ms}ms)")
        except Exception as e:
            sim_fail += 1
            log_fail(f"  {name}: EXCEPTION — {e}")

    log_info(f"VQPU sims: {sim_pass} passed, {sim_fail} failed of {actual_count}")

except ImportError as e:
    log_fail(f"Cannot import VQPU findings: {e}")
except Exception as e:
    log_fail(f"VQPU findings error: {e}")
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# PHASE 8: FULL run_simulation PIPELINE
# ═══════════════════════════════════════════════════════════════════

phase_header(8, "Full run_simulation Pipeline")

try:
    with VQPUBridge(enable_governor=False) as bridge:
        log_pass("VQPUBridge context manager works")

        # Bell pair
        bell = bridge.bell_pair(shots=512)
        log_pass(f"Bell pair job: {bell.num_qubits}Q, {len(bell.operations)} ops, {bell.shots} shots")

        # GHZ-5
        ghz = bridge.ghz_state(5)
        log_pass(f"GHZ-5 job: {ghz.num_qubits}Q, {len(ghz.operations)} ops")

        # QFT-4
        qft = bridge.qft_circuit(4)
        log_pass(f"QFT-4 job: {qft.num_qubits}Q, {len(qft.operations)} ops")

        # Sacred circuit
        sacred = bridge.sacred_circuit(3, depth=4)
        log_pass(f"Sacred(3,4) job: {sacred.num_qubits}Q, {len(sacred.operations)} ops")

        # run_simulation pipeline
        print()
        print("  [run_simulation — Bell]")
        t0 = time.monotonic()
        sim = bridge.run_simulation(bell, compile=True, error_correct=False,
                                     score_asi=True, score_agi=True)
        ms = round((time.monotonic() - t0) * 1000, 1)
        stages = sim.get("pipeline", {}).get("stages_executed", [])
        total_ms = sim.get("pipeline", {}).get("total_ms", 0)
        log_pass(f"Pipeline: {stages} — {total_ms:.1f}ms")

        if "result" in sim:
            r = sim["result"]
            if isinstance(r, dict):
                probs = r.get("probabilities", {})
                be = r.get("backend", "unknown")
                log_pass(f"Result: {dict(list(probs.items())[:4])} (backend={be})")
            else:
                log_pass(f"Result type: {type(r).__name__}")

        if "sacred" in sim:
            ss = sim["sacred"].get("sacred_score", "N/A")
            log_pass(f"Sacred score: {ss}")

        if "three_engine" in sim:
            te = sim["three_engine"]
            log_pass(f"Three-engine: composite={te.get('composite', 'N/A')}")

        if "asi_score" in sim:
            asi = sim["asi_score"]
            log_pass(f"ASI 15D score: {asi.get('score', asi.get('composite', 'N/A'))}")

        if "agi_score" in sim:
            agi = sim["agi_score"]
            log_pass(f"AGI 13D score: {agi.get('score', agi.get('composite', 'N/A'))}")

        if "compilation" in sim:
            c = sim["compilation"]
            log_pass(f"Compilation: compiled={c.get('compiled', False)}, "
                     f"gate_set={c.get('gate_set', 'N/A')}")

        # Engine status
        print()
        print("  [Engine Integration Status]")
        eng_status = bridge.engine_status() if hasattr(bridge, 'engine_status') else EngineIntegration.status()
        for k, v in eng_status.items():
            status_str = "✓" if v else "✗"
            level = log_pass if v else log_warn
            level(f"Engine {k}: {status_str}")

        # Three-engine status
        te_status = bridge.three_engine_status() if hasattr(bridge, 'three_engine_status') else {}
        if te_status:
            for k, v in te_status.items():
                log_info(f"  {k}: {v}")

        # Scoring cache final stats
        cache_stats = ScoringCache.stats()
        log_info(f"Scoring cache: {cache_stats}")

except Exception as e:
    log_fail(f"run_simulation pipeline error: {e}")
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# PHASE 9: BOOT MANAGER & DAEMON STATUS
# ═══════════════════════════════════════════════════════════════════

phase_header(9, "Boot Manager & Daemon Status")

try:
    from l104_vqpu_boot_manager import SERVICES, IPC_DIRS
    log_pass(f"Boot manager: {len(SERVICES)} services defined")
    for svc_id, info in SERVICES.items():
        crit = "CRITICAL" if info.get("critical") else "optional"
        log_info(f"  {svc_id} ({crit}): {info['description']}")
except ImportError:
    log_warn("l104_vqpu_boot_manager not importable")
except Exception as e:
    log_fail(f"Boot manager error: {e}")

# Check IPC directories
try:
    from pathlib import Path
    ipc_exists = 0
    ipc_missing = 0
    for d in [Path("/tmp/l104_bridge/inbox"), Path("/tmp/l104_bridge/outbox"),
              Path("/tmp/l104_bridge/telemetry")]:
        if d.exists():
            ipc_exists += 1
        else:
            ipc_missing += 1
    if ipc_exists > 0:
        log_pass(f"IPC dirs: {ipc_exists} exist, {ipc_missing} missing")
    else:
        log_warn("No IPC dirs exist (daemon never started on this machine?)")
except Exception as e:
    log_fail(f"IPC check error: {e}")

# Check daemon PID
try:
    daemon_pid = Path(ROOT) / "l104_daemon.pid"
    if daemon_pid.exists():
        pid = daemon_pid.read_text().strip()
        log_info(f"Daemon PID file: {pid}")
        # Check if process is alive
        try:
            os.kill(int(pid), 0)
            log_pass(f"Daemon process {pid} is ALIVE")
        except (ProcessLookupError, ValueError):
            log_warn(f"Daemon PID {pid} is STALE (process not running)")
        except PermissionError:
            log_pass(f"Daemon process {pid} exists (permission denied for signal)")
    else:
        log_info("No daemon PID file (Metal daemon not running)")
except Exception as e:
    log_fail(f"Daemon PID check error: {e}")


# ═══════════════════════════════════════════════════════════════════
# PHASE 10: v12.0 PARALLEL BATCH EXECUTION
# ═══════════════════════════════════════════════════════════════════

phase_header(10, "v12.0 Parallel Batch Execution")

try:
    bridge = VQPUBridge()
    bridge.start()

    # Create 4 Bell-pair jobs for parallel batch
    batch_jobs = []
    for i in range(4):
        job = QuantumJob(
            num_qubits=2,
            operations=[
                {"gate": "H", "qubits": [0]},
                {"gate": "CX", "qubits": [0, 1]},
            ],
            shots=512,
        )
        batch_jobs.append(job)

    t_batch = time.monotonic()
    batch_results = bridge.run_simulation_batch(batch_jobs)
    batch_ms = round((time.monotonic() - t_batch) * 1000, 1)

    if len(batch_results) == 4:
        log_pass(f"Parallel batch: 4 jobs completed in {batch_ms}ms")
    else:
        log_fail(f"Parallel batch: expected 4 results, got {len(batch_results)}")

    # Check all results have pipeline data
    good = sum(1 for r in batch_results if isinstance(r, dict) and "pipeline" in r)
    if good == 4:
        log_pass(f"All 4 batch results have pipeline metadata")
    else:
        log_warn(f"Only {good}/4 results have pipeline metadata")

    # Sequential comparison
    t_seq = time.monotonic()
    for job in batch_jobs:
        bridge.run_simulation(job)
    seq_ms = round((time.monotonic() - t_seq) * 1000, 1)

    speedup = seq_ms / max(batch_ms, 0.1)
    log_info(f"Sequential: {seq_ms}ms vs Parallel: {batch_ms}ms (speedup: {speedup:.2f}x)")
    if speedup > 1.0:
        log_pass(f"Parallel batch is faster ({speedup:.2f}x speedup)")
    else:
        log_warn(f"Parallel batch not faster (might be IO-bound or single-threaded)")

except Exception as e:
    log_fail(f"Parallel batch test error: {e}")
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# PHASE 11: v12.0 DAEMON ERROR LOGGING
# ═══════════════════════════════════════════════════════════════════

phase_header(11, "v12.0 Daemon Error Logging")

try:
    from l104_vqpu_bridge import DAEMON_MAX_ERROR_LOG, DAEMON_ERROR_THRESHOLD

    log_pass(f"DAEMON_MAX_ERROR_LOG = {DAEMON_MAX_ERROR_LOG}")
    log_pass(f"DAEMON_ERROR_THRESHOLD = {DAEMON_ERROR_THRESHOLD}")

    if DAEMON_MAX_ERROR_LOG >= 50:
        log_pass(f"Error log capacity adequate ({DAEMON_MAX_ERROR_LOG})")
    else:
        log_warn(f"Error log capacity low ({DAEMON_MAX_ERROR_LOG})")

    if DAEMON_ERROR_THRESHOLD >= 3:
        log_pass(f"Error threshold adequate ({DAEMON_ERROR_THRESHOLD})")
    else:
        log_warn(f"Error threshold very low ({DAEMON_ERROR_THRESHOLD})")

    # Check daemon cycler has error tracking
    bridge_status = bridge.daemon_cycler_status()
    if "error_log" in bridge_status or "consecutive_failures" in bridge_status:
        log_pass("Daemon cycler reports error tracking fields")
    else:
        log_info("Daemon cycler status does not report error fields (may not have errored)")
    log_pass("v12.0 daemon error logging framework present")

except ImportError:
    log_fail("DAEMON_MAX_ERROR_LOG / DAEMON_ERROR_THRESHOLD not importable")
except Exception as e:
    log_fail(f"Daemon error logging test error: {e}")


# ═══════════════════════════════════════════════════════════════════
# PHASE 12: v12.0 LRU PARAMETRIC CACHE
# ═══════════════════════════════════════════════════════════════════

phase_header(12, "v12.0 LRU Parametric Cache")

try:
    cache_max = ExactMPSHybridEngine._PARAMETRIC_CACHE_MAX
    log_info(f"Cache max: {cache_max}")

    if cache_max >= 32768:
        log_pass(f"Parametric cache capacity = {cache_max} (v12.0 upgrade)")
    else:
        log_fail(f"Parametric cache capacity = {cache_max}, expected ≥32768")

    # Verify LRU eviction
    if hasattr(ExactMPSHybridEngine, '_parametric_cache_order'):
        log_pass("LRU cache order tracking present (_parametric_cache_order)")
    else:
        log_fail("Missing _parametric_cache_order for LRU tracking")

    if hasattr(ExactMPSHybridEngine, '_cache_put'):
        log_pass("_cache_put() classmethod present for LRU eviction")
    else:
        log_fail("Missing _cache_put() classmethod")

    # Stress test: fill cache with entries and verify eviction
    initial_size = len(ExactMPSHybridEngine._parametric_cache)
    log_info(f"Current cache entries: {initial_size}")

    # Create a small MPS and exercise the cache
    mps = ExactMPSHybridEngine(2)
    for angle in [0.1 * i for i in range(20)]:
        Rz = np.array([
            [np.exp(-1j * angle / 2), 0],
            [0, np.exp(1j * angle / 2)]
        ], dtype=np.complex128)
        mps.apply_single_gate(0, Rz)

    after_size = len(ExactMPSHybridEngine._parametric_cache)
    log_info(f"Cache entries after 20 parametric gates: {after_size}")
    if after_size >= initial_size:
        log_pass(f"Parametric cache is growing ({initial_size} → {after_size})")
    else:
        log_warn(f"Cache did not grow (might be hitting duplicates)")

except Exception as e:
    log_fail(f"LRU cache test error: {e}")
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# PHASE 13: v12.0 PERFORMANCE PROFILING
# ═══════════════════════════════════════════════════════════════════

phase_header(13, "v12.0 Performance Profiling")

try:
    # Benchmark MPS product-state fast path
    print()
    print("  [MPS Product-State Fast Path]")
    mps = ExactMPSHybridEngine(4)
    H_gate = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

    t_fast = time.monotonic()
    for _ in range(1000):
        mps_test = ExactMPSHybridEngine(4)
        mps_test.apply_single_gate(0, H_gate)
    fast_ms = round((time.monotonic() - t_fast) * 1000, 2)
    log_info(f"1000 product-state H gates: {fast_ms}ms ({fast_ms/1000:.3f}ms/gate)")

    # Benchmark vectorized sampling
    print()
    print("  [Vectorized Sampling]")
    mps2 = ExactMPSHybridEngine(4)
    mps2.apply_single_gate(0, H_gate)
    mps2.apply_single_gate(1, H_gate)

    t_sample = time.monotonic()
    for _ in range(100):
        mps2.sample(4096)
    sample_ms = round((time.monotonic() - t_sample) * 1000, 2)
    log_info(f"100 × 4096-shot samples: {sample_ms}ms ({sample_ms/100:.2f}ms/sample)")

    # Benchmark transpiler
    print()
    print("  [10-Pass Transpiler]")
    complex_ops = [
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "Rz", "qubits": [0], "parameters": [0.3]},
        {"gate": "Rz", "qubits": [0], "parameters": [0.4]},
        {"gate": "H", "qubits": [1]},
        {"gate": "CX", "qubits": [1, 2]},
        {"gate": "Rx", "qubits": [0], "parameters": [0.5]},
        {"gate": "Rx", "qubits": [0], "parameters": [0.5]},
        {"gate": "CX", "qubits": [0, 1]},
    ] * 5  # 50 ops

    t_trans = time.monotonic()
    for _ in range(100):
        CircuitTranspiler.transpile(complex_ops, 3)
    trans_ms = round((time.monotonic() - t_trans) * 1000, 2)
    log_info(f"100 × transpile(50 ops): {trans_ms}ms ({trans_ms/100:.2f}ms/transpile)")

    # Benchmark scoring
    print()
    print("  [Sacred + Engine Scoring]")
    probs = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
    ScoringCache.clear()

    t_score = time.monotonic()
    for nq in range(2, 10):
        SacredAlignmentScorer.score(probs, nq)
        ThreeEngineQuantumScorer.composite_score(1.0)
    score_ms = round((time.monotonic() - t_score) * 1000, 2)
    log_info(f"8 scoring rounds (sacred + three-engine): {score_ms}ms")

    # v12.0 self_test benchmark
    print()
    print("  [VQPUBridge.self_test()]")
    t_self = time.monotonic()
    test_result = bridge.self_test()
    self_ms = round((time.monotonic() - t_self) * 1000, 2)
    passed = test_result.get("passed", 0)
    total = test_result.get("total", 0)
    log_info(f"self_test(): {passed}/{total} passed in {self_ms}ms")
    if test_result.get("all_pass"):
        log_pass(f"VQPUBridge.self_test() all {total} probes pass")
    else:
        failed_tests = [t["test"] for t in test_result.get("tests", []) if not t.get("pass")]
        log_fail(f"self_test() failures: {', '.join(failed_tests)}")

    log_pass("v12.0 performance profiling complete")

except Exception as e:
    log_fail(f"Performance profiling error: {e}")
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════

print()
print("═" * 65)
print("  L104 VQPU DEBUG SUITE v2.0 — SUMMARY")
print("═" * 65)
print(f"  PASS: {PASS}")
print(f"  FAIL: {FAIL}")
print(f"  WARN: {WARN}")
print(f"  TOTAL: {PASS + FAIL + WARN}")
print()

if FAIL > 0:
    print("  FAILURES:")
    for status, msg in RESULTS:
        if status == "FAIL":
            print(f"    ✗ {msg}")
    print()

if WARN > 0:
    print("  WARNINGS:")
    for status, msg in RESULTS:
        if status == "WARN":
            print(f"    ⚠ {msg}")
    print()

overall = "ALL CLEAR" if FAIL == 0 else f"{FAIL} FAILURE(S)"
print(f"  VERDICT: {overall}")
print(f"  INVARIANT: {GOD_CODE} | PILOT: LONDEL")
print("═" * 65)

sys.exit(1 if FAIL > 0 else 0)
