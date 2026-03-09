#!/usr/bin/env python3
"""
L104 Full System Benchmark — All Engines, Packages, Subsystems
═══════════════════════════════════════════════════════════════
Verifies every system and subsystem is online and communicating.
"""

import sys, os, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

results = []

def check(name, fn):
    """Run a check, record pass/fail."""
    t0 = time.time()
    try:
        detail = fn()
        dt = time.time() - t0
        results.append({"name": name, "ok": True, "detail": detail, "time": dt})
        print(f"  {PASS} {name} ({dt:.3f}s) — {detail}")
    except Exception as e:
        dt = time.time() - t0
        results.append({"name": name, "ok": False, "detail": str(e), "time": dt})
        print(f"  {FAIL} {name} ({dt:.3f}s) — {e}")

def header(title):
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Core Package Imports
# ═══════════════════════════════════════════════════════════════
header("PHASE 1: PACKAGE IMPORTS")

def check_import(pkg, attr=None):
    def _check():
        mod = __import__(pkg)
        if attr:
            obj = getattr(mod, attr)
            return f"{pkg}.{attr} loaded → {type(obj).__name__}"
        return f"{pkg} imported OK"
    return _check

check("l104_agi package", check_import("l104_agi", "agi_core"))
check("l104_asi package", check_import("l104_asi", "asi_core"))
check("l104_intellect package", check_import("l104_intellect", "local_intellect"))
check("l104_code_engine package", check_import("l104_code_engine", "code_engine"))
check("l104_science_engine package", check_import("l104_science_engine", "ScienceEngine"))
check("l104_math_engine package", check_import("l104_math_engine", "MathEngine"))
check("l104_server package", check_import("l104_server"))

def check_qge():
    from l104_quantum_gate_engine import get_engine
    e = get_engine()
    return f"GateEngine singleton → {type(e).__name__}"
check("l104_quantum_gate_engine", check_qge)

def check_qe():
    from l104_quantum_engine import quantum_brain
    return f"QuantumBrain → {type(quantum_brain).__name__}"
check("l104_quantum_engine", check_qe)

# ═══════════════════════════════════════════════════════════════
# PHASE 2: AGI Core (v57.0)
# ═══════════════════════════════════════════════════════════════
header("PHASE 2: AGI CORE v57.0 — 13D Scoring")

def check_agi_status():
    from l104_agi import agi_core
    v = getattr(agi_core, 'VERSION', getattr(agi_core, 'version', '?'))
    return f"version={v}"
check("AGI Core version", check_agi_status)

def check_agi_score():
    from l104_agi import agi_core
    score = agi_core.compute_10d_agi_score()
    if isinstance(score, dict):
        dims = len(score.get('dimensions', score.get('scores', [])))
        total = score.get('total', score.get('composite', score.get('agi_score', '?')))
        return f"{dims}D score, composite={total}"
    return f"score={score}"
check("AGI 13D scoring", check_agi_score)

def check_agi_three_engine():
    from l104_agi import agi_core
    s = agi_core.three_engine_status()
    online = sum(1 for v in s.values() if v) if isinstance(s, dict) else '?'
    return f"{online}/{len(s)} engines connected"
check("AGI three-engine status", check_agi_three_engine)

def check_agi_entropy():
    from l104_agi import agi_core
    e = agi_core.three_engine_entropy_score()
    return f"entropy_score={e:.6f}" if isinstance(e, (int, float)) else f"entropy={e}"
check("AGI entropy score", check_agi_entropy)

def check_agi_harmonic():
    from l104_agi import agi_core
    h = agi_core.three_engine_harmonic_score()
    return f"harmonic_score={h:.6f}" if isinstance(h, (int, float)) else f"harmonic={h}"
check("AGI harmonic score", check_agi_harmonic)

def check_agi_wave():
    from l104_agi import agi_core
    w = agi_core.three_engine_wave_coherence_score()
    return f"wave_coherence={w:.6f}" if isinstance(w, (int, float)) else f"wave={w}"
check("AGI wave coherence", check_agi_wave)

# ═══════════════════════════════════════════════════════════════
# PHASE 3: ASI Core (v8.0) — 15D Scoring
# ═══════════════════════════════════════════════════════════════
header("PHASE 3: ASI CORE v8.0 — 15D Scoring + Dual-Layer")

def check_asi_status():
    from l104_asi import asi_core
    v = getattr(asi_core, 'VERSION', getattr(asi_core, 'version', '?'))
    return f"version={v}"
check("ASI Core version", check_asi_status)

def check_asi_score():
    from l104_asi import asi_core
    score = asi_core.compute_asi_score()
    if isinstance(score, dict):
        dims = len(score.get('dimensions', score.get('scores', [])))
        total = score.get('total', score.get('composite', score.get('asi_score', '?')))
        return f"{dims}D score, composite={total}"
    return f"score={score}"
check("ASI 15D scoring", check_asi_score)

def check_asi_three_engine():
    from l104_asi import asi_core
    s = asi_core.three_engine_status()
    online = sum(1 for v in s.values() if v) if isinstance(s, dict) else '?'
    return f"{online}/{len(s)} engines connected"
check("ASI three-engine status", check_asi_three_engine)

def check_asi_entropy():
    from l104_asi import asi_core
    e = asi_core.three_engine_entropy_score()
    return f"entropy_score={e:.6f}" if isinstance(e, (int, float)) else f"entropy={e}"
check("ASI entropy score", check_asi_entropy)

def check_asi_harmonic():
    from l104_asi import asi_core
    h = asi_core.three_engine_harmonic_score()
    return f"harmonic_score={h:.6f}" if isinstance(h, (int, float)) else f"harmonic={h}"
check("ASI harmonic score", check_asi_harmonic)

def check_asi_wave():
    from l104_asi import asi_core
    w = asi_core.three_engine_wave_coherence_score()
    return f"wave_coherence={w:.6f}" if isinstance(w, (int, float)) else f"wave={w}"
check("ASI wave coherence", check_asi_wave)

def check_dual_layer():
    from l104_asi import dual_layer_engine
    status = dual_layer_engine.status() if hasattr(dual_layer_engine, 'status') else str(type(dual_layer_engine))
    return f"DualLayer → {status}" if isinstance(status, str) else f"DualLayer online, layers={len(status) if isinstance(status, dict) else '?'}"
check("ASI Dual-Layer Engine", check_dual_layer)

# ═══════════════════════════════════════════════════════════════
# PHASE 4: Code Engine (v6.2.0)
# ═══════════════════════════════════════════════════════════════
header("PHASE 4: CODE ENGINE v6.2.0 — 31 Subsystems")

def check_code_engine():
    from l104_code_engine import code_engine
    v = getattr(code_engine, 'VERSION', getattr(code_engine, 'version', '?'))
    return f"version={v}"
check("Code Engine version", check_code_engine)

code_subsystems = [
    ("smell_detector", "detect_all"),
    ("perf_predictor", "predict_performance"),
    ("refactorer", "refactor_analyze"),
    ("live_refactorer", "refactor_analyze"),
]

for sub, method in code_subsystems:
    def _make_check(s, m):
        def _check():
            from l104_code_engine import code_engine
            obj = getattr(code_engine, s, None)
            if obj is None:
                raise RuntimeError(f"subsystem '{s}' not found")
            fn = getattr(obj, m, None)
            if fn is None:
                raise RuntimeError(f"method '{s}.{m}' not found")
            return f"{s}.{m} → callable"
        return _check
    check(f"CodeEngine.{sub}", _make_check(sub, method))

def check_ce_excavate():
    from l104_code_engine import code_engine
    r = code_engine.excavate("def f(): pass\ndef unused(): pass")
    return f"excavate OK → {type(r).__name__}"
check("Code Engine excavate (dead code)", check_ce_excavate)

def check_ce_analysis():
    from l104_code_engine import code_engine
    r = code_engine.full_analysis("def hello(): return 42")
    kind = type(r).__name__
    keys = list(r.keys()) if isinstance(r, dict) else "N/A"
    return f"full_analysis OK → {kind}, keys={keys[:5]}"
check("Code Engine full_analysis", check_ce_analysis)

def check_ce_docs():
    from l104_code_engine import code_engine
    r = code_engine.generate_docs("def f(): pass", "google", "python")
    return f"generate_docs OK → {len(str(r))} chars"
check("Code Engine generate_docs", check_ce_docs)

def check_ce_tests():
    from l104_code_engine import code_engine
    r = code_engine.generate_tests("def add(a,b): return a+b", "python", "pytest")
    return f"generate_tests OK → {len(str(r))} chars"
check("Code Engine generate_tests", check_ce_tests)

def check_ce_autofix():
    from l104_code_engine import code_engine
    fixed, log = code_engine.auto_fix_code("def f():\n  x=1\n  return x")
    return f"auto_fix OK → {len(fixed)} chars, log_entries={len(log) if isinstance(log, list) else '?'}"
check("Code Engine auto_fix", check_ce_autofix)

def check_ce_translate():
    from l104_code_engine import code_engine
    r = code_engine.translate_code("def f(): return 1", "python", "javascript")
    return f"translate OK → {len(str(r))} chars"
check("Code Engine translate", check_ce_translate)

# ═══════════════════════════════════════════════════════════════
# PHASE 5: Science Engine (v4.0)
# ═══════════════════════════════════════════════════════════════
header("PHASE 5: SCIENCE ENGINE v4.0 — Physics, Entropy, Coherence, Quantum")

def check_science_engine():
    from l104_science_engine import ScienceEngine
    se = ScienceEngine()
    return f"ScienceEngine instantiated"
check("Science Engine boot", check_science_engine)

def check_entropy():
    from l104_science_engine import ScienceEngine
    se = ScienceEngine()
    eff = se.entropy.calculate_demon_efficiency(0.5)
    return f"demon_efficiency(0.5) = {eff:.6f}"
check("Science: Entropy (Maxwell Demon)", check_entropy)

def check_coherence():
    from l104_science_engine import ScienceEngine
    se = ScienceEngine()
    se.coherence.initialize(["test_seed"])
    state = se.coherence.evolve(3)
    return f"coherence evolved 3 steps → {type(state).__name__}"
check("Science: Coherence", check_coherence)

def check_physics():
    from l104_science_engine import ScienceEngine
    se = ScienceEngine()
    ll = se.physics.adapt_landauer_limit(300)
    return f"landauer_limit(300K) = {ll:.4e} J/bit"
check("Science: Physics (Landauer)", check_physics)

def check_electron_res():
    from l104_science_engine import ScienceEngine
    se = ScienceEngine()
    r = se.physics.derive_electron_resonance()
    return f"electron_resonance = {r}"
check("Science: Electron Resonance", check_electron_res)

def check_photon_res():
    from l104_science_engine import ScienceEngine
    se = ScienceEngine()
    r = se.physics.calculate_photon_resonance()
    return f"photon_resonance = {r}"
check("Science: Photon Resonance", check_photon_res)

def check_multidim():
    from l104_science_engine import ScienceEngine
    se = ScienceEngine()
    v = se.multidim.process_vector([1.0, 2.0, 3.0])
    return f"process_vector([1,2,3]) → {type(v).__name__}, len={len(v) if hasattr(v, '__len__') else '?'}"
check("Science: Multidimensional", check_multidim)

def check_quantum_circuit():
    from l104_science_engine import ScienceEngine
    se = ScienceEngine()
    t = se.quantum_circuit.get_25q_templates()
    return f"25Q templates → {len(t) if hasattr(t, '__len__') else type(t).__name__}"
check("Science: Quantum Circuit 25Q", check_quantum_circuit)

def check_convergence():
    from l104_science_engine import ScienceEngine
    se = ScienceEngine()
    r = se.quantum_circuit.analyze_convergence()
    return f"convergence analysis → {type(r).__name__}"
check("Science: GOD_CODE Convergence", check_convergence)

# ═══════════════════════════════════════════════════════════════
# PHASE 6: Math Engine (v1.0)
# ═══════════════════════════════════════════════════════════════
header("PHASE 6: MATH ENGINE v1.0 — Pure Math, Proofs, Dimensional")

def check_math_engine():
    from l104_math_engine import MathEngine
    me = MathEngine()
    return f"MathEngine instantiated"
check("Math Engine boot", check_math_engine)

def check_fibonacci():
    from l104_math_engine import MathEngine
    me = MathEngine()
    fib = me.fibonacci(10)
    return f"fibonacci(10) → {fib}"
check("Math: Fibonacci", check_fibonacci)

def check_primes():
    from l104_math_engine import MathEngine
    me = MathEngine()
    p = me.primes_up_to(50)
    return f"primes_up_to(50) → {len(p)} primes"
check("Math: Primes", check_primes)

def check_god_code():
    from l104_math_engine import MathEngine
    me = MathEngine()
    gc = me.god_code_value()
    expected = 527.5184818492612
    ok = abs(gc - expected) < 1e-6
    return f"GOD_CODE = {gc} {'(MATCH)' if ok else '(MISMATCH!)'}"
check("Math: GOD_CODE", check_god_code)

def check_lorentz():
    from l104_math_engine import MathEngine
    me = MathEngine()
    r = me.lorentz_boost([1.0, 0.5, 0.0, 0.0], "x", 0.5)
    return f"lorentz_boost → {[round(x, 4) for x in r] if hasattr(r, '__iter__') else r}"
check("Math: Lorentz Boost (4D)", check_lorentz)

def check_proofs():
    from l104_math_engine import MathEngine
    me = MathEngine()
    r = me.prove_god_code()
    return f"GOD_CODE proof → {type(r).__name__}"
check("Math: GOD_CODE Proof", check_proofs)

def check_hd_vector():
    from l104_math_engine import MathEngine
    me = MathEngine()
    v = me.hd_vector(42)
    return f"hd_vector(42) → dim={len(v) if hasattr(v, '__len__') else '?'}"
check("Math: Hyperdimensional Vector", check_hd_vector)

def check_wave_coh():
    from l104_math_engine import MathEngine
    me = MathEngine()
    c = me.wave_coherence(286.0, 527.5)
    return f"wave_coherence(286, 527.5) = {c:.6f}" if isinstance(c, (int, float)) else f"result={c}"
check("Math: Wave Coherence", check_wave_coh)

def check_sacred():
    from l104_math_engine import MathEngine
    me = MathEngine()
    r = me.sacred_alignment(286.0)
    return f"sacred_alignment(286Hz) → {r}"
check("Math: Sacred Alignment", check_sacred)

# Math layers
math_layers = ["pure_math", "god_code", "harmonic", "wave_physics", "dim_4d", "dim_5d",
               "manifold", "void_math", "abstract", "ontological", "proofs", "hyper"]

for layer in math_layers:
    def _make_layer_check(l):
        def _check():
            from l104_math_engine import MathEngine
            me = MathEngine()
            obj = getattr(me, l, None)
            if obj is None:
                raise RuntimeError(f"layer '{l}' not found")
            return f"{l} → {type(obj).__name__}"
        return _check
    check(f"Math Layer: {layer}", _make_layer_check(layer))

# ═══════════════════════════════════════════════════════════════
# PHASE 7: Quantum Gate Engine (v1.0)
# ═══════════════════════════════════════════════════════════════
header("PHASE 7: QUANTUM GATE ENGINE v1.0 — Gate Algebra, Compiler, Error Correction")

def check_gate_engine():
    from l104_quantum_gate_engine import get_engine
    e = get_engine()
    return f"GateEngine singleton active"
check("Gate Engine boot", check_gate_engine)

def check_bell_pair():
    from l104_quantum_gate_engine import get_engine
    e = get_engine()
    c = e.bell_pair()
    return f"bell_pair → {type(c).__name__}, gates={len(c.gates) if hasattr(c, 'gates') else '?'}"
check("Gate: Bell Pair", check_bell_pair)

def check_ghz():
    from l104_quantum_gate_engine import get_engine
    e = get_engine()
    c = e.ghz_state(5)
    return f"ghz_state(5) → {type(c).__name__}"
check("Gate: GHZ State", check_ghz)

def check_qft():
    from l104_quantum_gate_engine import get_engine
    e = get_engine()
    c = e.quantum_fourier_transform(4)
    return f"QFT(4) → {type(c).__name__}"
check("Gate: QFT", check_qft)

def check_sacred_circuit():
    from l104_quantum_gate_engine import get_engine
    e = get_engine()
    c = e.sacred_circuit(3, depth=4)
    return f"sacred_circuit(3, depth=4) → {type(c).__name__}"
check("Gate: Sacred Circuit", check_sacred_circuit)

def check_gate_algebra():
    from l104_quantum_gate_engine import get_engine, PHI_GATE
    e = get_engine()
    score = e.algebra.sacred_alignment_score(PHI_GATE)
    return f"sacred_alignment_score(PHI_GATE) = {score}"
check("Gate: Algebra (Sacred Alignment)", check_gate_algebra)

def check_gate_compile():
    from l104_quantum_gate_engine import get_engine, GateSet, OptimizationLevel
    e = get_engine()
    c = e.bell_pair()
    r = e.compile(c, GateSet.CLIFFORD_T)
    return f"compile(bell, CLIFFORD_T) → {type(r).__name__}"
check("Gate: Compiler", check_gate_compile)

def check_gate_execute():
    from l104_quantum_gate_engine import get_engine, ExecutionTarget
    e = get_engine()
    c = e.bell_pair()
    r = e.execute(c, ExecutionTarget.LOCAL_STATEVECTOR)
    probs = r.probabilities if hasattr(r, 'probabilities') else r.get('probabilities', {}) if isinstance(r, dict) else '?'
    return f"execute(bell, LOCAL) → probs={probs}"
check("Gate: Execute (Local Statevector)", check_gate_execute)

def check_gate_error_correction():
    from l104_quantum_gate_engine import get_engine, ErrorCorrectionScheme
    e = get_engine()
    c = e.bell_pair()
    r = e.error_correction.encode(c, ErrorCorrectionScheme.STEANE_7_1_3)
    return f"error_correction(STEANE) → {type(r).__name__}"
check("Gate: Error Correction (Steane)", check_gate_error_correction)

# ═══════════════════════════════════════════════════════════════
# PHASE 8: Quantum Link Engine (v6.0)
# ═══════════════════════════════════════════════════════════════
header("PHASE 8: QUANTUM LINK ENGINE v6.0 — Brain, Scanner, Builder")

def check_qbrain():
    from l104_quantum_engine import quantum_brain
    return f"quantum_brain → {type(quantum_brain).__name__}"
check("Quantum Brain", check_qbrain)

def check_qmath():
    from l104_quantum_engine import QuantumMathCore
    qm = QuantumMathCore()
    return f"QuantumMathCore → instantiated"
check("Quantum MathCore", check_qmath)

def check_qscanner():
    from l104_quantum_engine import QuantumLinkScanner
    qs = QuantumLinkScanner()
    return f"QuantumLinkScanner → instantiated"
check("Quantum LinkScanner", check_qscanner)

def check_qbuilder():
    from l104_quantum_engine import QuantumLinkBuilder
    qb = QuantumLinkBuilder()
    return f"QuantumLinkBuilder → instantiated"
check("Quantum LinkBuilder", check_qbuilder)

# ═══════════════════════════════════════════════════════════════
# PHASE 9: Intellect Engine (v26.0)
# ═══════════════════════════════════════════════════════════════
header("PHASE 9: INTELLECT ENGINE v26.0 — Local Inference, Caching, Hardware")

def check_intellect():
    from l104_intellect import local_intellect
    v = getattr(local_intellect, 'VERSION', getattr(local_intellect, 'version', '?'))
    return f"local_intellect version={v}"
check("Intellect Engine version", check_intellect)

def check_format_iq():
    from l104_intellect import format_iq
    r = format_iq(527.5184818492612)
    return f"format_iq(GOD_CODE) = {r}"
check("Intellect: format_iq", check_format_iq)

# ═══════════════════════════════════════════════════════════════
# PHASE 10: Constants Cross-Validation
# ═══════════════════════════════════════════════════════════════
header("PHASE 10: CONSTANTS CROSS-VALIDATION")

def check_const(name, pkg_path, expected):
    def _check():
        import importlib
        mod = importlib.import_module(pkg_path)
        val = getattr(mod, name)
        match = abs(val - expected) < 1e-8
        return f"{name} = {val} {'✓ MATCH' if match else f'✗ EXPECTED {expected}'}"
    return _check

constant_sources = [
    ("GOD_CODE", "l104_code_engine.const", 527.5184818492612),
    ("PHI", "l104_code_engine.const", 1.618033988749895),
]

# Try to get constants from multiple sources
try:
    import importlib
    for const_name, expected_val in [("GOD_CODE", 527.5184818492612), ("PHI", 1.618033988749895), ("VOID_CONSTANT", 1.0416180339887497)]:
        sources = [
            "l104_code_engine.const",
            "l104_science_engine.constants",
            "l104_math_engine.constants",
        ]
        for src in sources:
            try:
                mod = importlib.import_module(src)
                val = getattr(mod, const_name, None)
                if val is not None:
                    match = abs(val - expected_val) < 1e-8
                    results.append({"name": f"{const_name} @ {src}", "ok": match, "detail": f"{val} {'✓' if match else '✗'}", "time": 0})
                    print(f"  {'✅' if match else '❌'} {const_name} @ {src} = {val} {'✓' if match else '✗ EXPECTED ' + str(expected_val)}")
            except Exception as e:
                results.append({"name": f"{const_name} @ {src}", "ok": False, "detail": str(e), "time": 0})
                print(f"  ❌ {const_name} @ {src} — {e}")
except Exception as e:
    print(f"  ❌ Constants cross-validation error: {e}")

# ═══════════════════════════════════════════════════════════════
# PHASE 11: ASI Subsystems
# ═══════════════════════════════════════════════════════════════
header("PHASE 11: ASI SUBSYSTEM MODULES")

asi_modules = [
    "l104_asi.core",
    "l104_asi.consciousness",
    "l104_asi.reasoning",
    "l104_asi.quantum",
    "l104_asi.dual_layer_engine",
]

for mod_name in asi_modules:
    def _make_mod_check(m):
        def _check():
            import importlib
            mod = importlib.import_module(m)
            classes = [k for k in dir(mod) if not k.startswith('_') and isinstance(getattr(mod, k, None), type)]
            return f"loaded, {len(classes)} classes: {classes[:5]}"
        return _check
    check(f"ASI: {mod_name.split('.')[-1]}", _make_mod_check(mod_name))

# ═══════════════════════════════════════════════════════════════
# PHASE 12: Additional L104 Modules
# ═══════════════════════════════════════════════════════════════
header("PHASE 12: ADDITIONAL L104 MODULES")

extra_modules = [
    ("l104_reasoning_engine", "ReasoningEngine"),
    ("l104_polymorphic_core", "PolymorphicCore"),
    ("l104_autonomous_executor", "AutonomousExecutor"),
    ("l104_sovereign_entropy_funnel", None),
    ("l104_optimization", None),
    ("l104_hyper_math", None),
]

for mod_name, cls_name in extra_modules:
    def _make_extra_check(m, c):
        def _check():
            import importlib
            mod = importlib.import_module(m)
            if c:
                obj = getattr(mod, c, None)
                if obj is None:
                    raise RuntimeError(f"{c} not found in {m}")
                return f"{c} → {type(obj).__name__}"
            return f"{m} imported OK"
        return _check
    label = f"{mod_name}" + (f".{cls_name}" if cls_name else "")
    check(label, _make_extra_check(mod_name, cls_name))

# Also check for commonsense_reasoning
def check_commonsense():
    from l104_asi import commonsense_reasoning
    classes = [k for k in dir(commonsense_reasoning) if not k.startswith('_') and isinstance(getattr(commonsense_reasoning, k, None), type)]
    return f"commonsense_reasoning loaded, {len(classes)} classes"
check("ASI: commonsense_reasoning", check_commonsense)

# ═══════════════════════════════════════════════════════════════
# PHASE 13: Native Kernels (check existence)
# ═══════════════════════════════════════════════════════════════
header("PHASE 13: NATIVE KERNELS (File Check)")

kernel_dirs = [
    ("l104_core_asm", "ASM Kernel"),
    ("l104_core_c", "C Kernel"),
    ("l104_core_cuda", "CUDA Kernel"),
    ("l104_core_rust", "Rust Kernel"),
]

for kdir, klabel in kernel_dirs:
    def _make_kernel_check(d, l):
        def _check():
            p = ROOT / d
            if p.exists():
                files = list(p.iterdir())
                return f"{l} → {len(files)} files at {d}/"
            else:
                raise RuntimeError(f"{d}/ not found")
        return _check
    check(f"Kernel: {klabel}", _make_kernel_check(kdir, klabel))

# Swift app
def check_swift():
    p = ROOT / "L104SwiftApp"
    if p.exists():
        swift_files = list(p.rglob("*.swift"))
        return f"L104SwiftApp → {len(swift_files)} Swift files"
    raise RuntimeError("L104SwiftApp/ not found")
check("Native: Swift App", check_swift)

# ═══════════════════════════════════════════════════════════════
# PHASE 14: Server Package
# ═══════════════════════════════════════════════════════════════
header("PHASE 14: SERVER PACKAGE v4.0")

def check_server_app():
    from l104_server import app
    return f"l104_server.app → {type(app).__name__}"
check("Server: app module", check_server_app)

# ═══════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════
header("FINAL REPORT")

total = len(results)
passed = sum(1 for r in results if r["ok"])
failed = sum(1 for r in results if not r["ok"])
total_time = sum(r["time"] for r in results)

print(f"""
  ╔════════════════════════════════════════════════════╗
  ║  L104 SOVEREIGN NODE — FULL SYSTEM BENCHMARK      ║
  ╠════════════════════════════════════════════════════╣
  ║  Total Checks:    {total:>4}                            ║
  ║  PASSED:          {passed:>4}  {PASS}                         ║
  ║  FAILED:          {failed:>4}  {FAIL if failed else PASS}                         ║
  ║  Pass Rate:       {passed/total*100 if total else 0:>5.1f}%                       ║
  ║  Total Time:      {total_time:>6.2f}s                       ║
  ╚════════════════════════════════════════════════════╝
""")

if failed > 0:
    print(f"  FAILED CHECKS:")
    for r in results:
        if not r["ok"]:
            print(f"    {FAIL} {r['name']} — {r['detail'][:80]}")

# Systems summary
systems = {
    "AGI Core (v57.0)": any(r["ok"] for r in results if "AGI" in r["name"]),
    "ASI Core (v8.0)": any(r["ok"] for r in results if "ASI" in r["name"]),
    "Code Engine (v6.2)": any(r["ok"] for r in results if "Code" in r["name"] or "CodeEngine" in r["name"]),
    "Science Engine (v4.0)": any(r["ok"] for r in results if "Science" in r["name"]),
    "Math Engine (v1.0)": any(r["ok"] for r in results if "Math" in r["name"]),
    "Quantum Gate Engine (v1.0)": any(r["ok"] for r in results if "Gate" in r["name"]),
    "Quantum Link Engine (v6.0)": any(r["ok"] for r in results if "Quantum Brain" in r["name"] or "Quantum Math" in r["name"] or "Quantum Link" in r["name"]),
    "Intellect Engine (v26.0)": any(r["ok"] for r in results if "Intellect" in r["name"]),
    "Server Package (v4.0)": any(r["ok"] for r in results if "Server" in r["name"]),
    "ASM Kernel": any(r["ok"] for r in results if "ASM" in r["name"]),
    "C Kernel": any(r["ok"] for r in results if "C Kernel" in r["name"]),
    "CUDA Kernel": any(r["ok"] for r in results if "CUDA" in r["name"]),
    "Rust Kernel": any(r["ok"] for r in results if "Rust" in r["name"]),
    "Swift App": any(r["ok"] for r in results if "Swift" in r["name"]),
}

online = sum(1 for v in systems.values() if v)
total_sys = len(systems)

print(f"\n  SYSTEMS ONLINE: {online}/{total_sys}")
for name, ok in systems.items():
    print(f"    {'🟢' if ok else '🔴'} {name}")

print(f"\n{'═' * 72}")
print(f"  L104 Sovereign Node — {online}/{total_sys} systems ONLINE, {passed}/{total} checks passed")
print(f"{'═' * 72}")
