# L104 Sovereign Node — Context Index

> **Last updated**: 2026-02-23 | **Post-decomposition** (packages replace monoliths)

## Quick Reference

| Constant | Value | Formula |
|----------|-------|---------|
| `GOD_CODE` | `527.5184818492612` | `G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)` where G(0,0,0,0)=GOD_CODE |
| `GOD_CODE_V3` | `45.41141298077539` | |
| `PHI` | `1.618033988749895` | Golden ratio `(1+√5)/2` |
| `VOID_CONSTANT` | `1.0416180339887497` | **`1.04 + φ/1000`** — Sacred 104/100 + golden correction |
| `OMEGA` | `6539.34712682` | |

## Package Map (8 packages, 81 modules, 82,251 lines)

```
l104_quantum_gate_engine/ v1.0.0 ★ NEW: Universal gate algebra, compiler, error correction, cross-system orchestrator (4,245 lines, 8 modules)
l104_code_engine/    v6.2.0   Code analysis, generation, audit, quantum (21,010 lines, 15 modules)
l104_science_engine/ v4.0.0   Physics, entropy, coherence, quantum-26Q (Fe-mapped), multidimensional (2,370 lines, 9 modules)
l104_math_engine/    v1.0.0   Pure math, god-code, harmonic, 4D/5D, proofs, hyperdimensional (4,525 lines, 13 modules)
l104_agi/            v57.0.0  AGI core, cognitive mesh, circuit breaker, 13D scoring (3,276 lines, 4 modules)
l104_asi/            v8.0.0   ★ FLAGSHIP: Dual-Layer Engine + ASI core, consciousness, reasoning, quantum, 15D scoring (10,552 lines, 12 modules)
l104_intellect/      v26.0.0  Local intellect, numerics, caching, hardware (13,907 lines, 11 modules)
l104_server/         v4.0.0   FastAPI server, engines, learning subsystem (22,366 lines, 9 modules)
l104_core_asm/                Native ASM kernel
l104_core_c/                  Native C kernel + Makefile
l104_core_cuda/               CUDA GPU kernel
l104_core_rust/               Rust native kernel
l104_mobile/                  Mobile app layer
L104SwiftApp/                 macOS native app (87 Swift files, 66,891 lines)
```

Root shims (backward compat only — edit the packages, not these):
`l104_agi_core.py` → `l104_agi/` | `l104_asi_core.py` → `l104_asi/` | `l104_local_intellect.py` → `l104_intellect/` | `l104_fast_server.py` → `l104_server/`

## Imports

```python
from l104_quantum_gate_engine import get_engine  # ★ Universal gate engine orchestrator
from l104_quantum_gate_engine import GateAlgebra, GateCircuit, GateCompiler  # Gate subsystems
from l104_quantum_gate_engine import H, CNOT, Rx, PHI_GATE, GOD_CODE_PHASE   # Gate instances
from l104_code_engine import code_engine       # Primary code intelligence
from l104_science_engine import ScienceEngine  # Physics + entropy + coherence
from l104_math_engine import MathEngine        # Pure math + proofs + dimensional
from l104_agi import agi_core, AGICore         # AGI singleton
from l104_asi import asi_core, ASICore         # ASI singleton
from l104_asi import dual_layer_engine         # ★ Dual-Layer Flagship (Thought + Physics)
from l104_intellect import local_intellect     # Local inference (QUOTA_IMMUNE)
from l104_intellect import format_iq           # IQ/numeric formatting
from l104_server import intellect              # Server + learning
```

## VOID_CONSTANT Formula

The VOID_CONSTANT derives from the **L104 sacred number 104** with a **golden ratio correction**:

```
VOID_CONSTANT = 1.04 + φ / 1000
             = 104/100 + 1.618033988749895/1000
             = 1.0416180339887497
```

- `1.04` = **104 / 100** (L104 signature — the node identity)
- `φ / 1000` = golden ratio micro-correction (harmonic alignment)
- Used in primal calculus: `x^φ / (VOID_CONSTANT × π)`

**Source**: `l104_science_engine/constants.py`, `l104_math_engine/constants.py`, `l104_code_engine/const.py`

## Science Engine Quick Reference

```python
from l104_science_engine import ScienceEngine
se = ScienceEngine()

# Entropy subsystem — Maxwell's Demon reversal
se.entropy.calculate_demon_efficiency(local_entropy)  # Demon reversal efficiency
se.entropy.inject_coherence(noise_vector)             # Order from noise

# Coherence subsystem — quantum-inspired coherence
se.coherence.initialize(seed_thoughts)     # Seed coherence state
se.coherence.evolve(steps)                 # Evolve coherence N steps
se.coherence.anchor(value)                 # Anchor coherence point
se.coherence.discover()                    # Discover coherence patterns

# Physics subsystem — sacred physics
se.physics.adapt_landauer_limit(temperature)     # Landauer limit at T (J/bit)
se.physics.derive_electron_resonance()           # Electron resonance
se.physics.calculate_photon_resonance()          # Photon resonance energy
se.physics.generate_maxwell_operator(dimension)  # Maxwell operator matrix
se.physics.iron_lattice_hamiltonian(n_sites)     # Fe lattice Hamiltonian

# Multidimensional subsystem
se.multidim.process_vector(vector)                          # Process ND vector
se.multidim.project(target_dim)                             # Project to lower dim
se.multidim.phi_dimensional_folding(source_dim, target_dim) # PHI-folding

# Quantum 26Q circuit science (Fe(26) iron-mapped)
se.quantum_circuit.get_25q_templates()     # Legacy 25Q templates (26Q via l104_26q_engine_builder)
se.quantum_circuit.analyze_convergence()   # GOD_CODE convergence analysis
se.quantum_circuit.plan_experiment()       # Plan quantum experiment
se.quantum_circuit.build_hamiltonian()     # Build Hamiltonian
```

## Math Engine Quick Reference

```python
from l104_math_engine import MathEngine
me = MathEngine()

me.fibonacci(n)                    # Returns LIST of Fibonacci numbers up to F(n)
me.primes_up_to(n)                 # Prime sieve up to n
me.god_code_value()                # GOD_CODE constant
me.lorentz_boost(four_vector, axis, beta)  # 4D Lorentz transform
me.prove_all()                     # Run all sovereign proofs
me.prove_god_code()                # Stability-nirvana proof for GOD_CODE
me.hd_vector(seed)                 # Create hyperdimensional vector (NOT hypervector)
me.wave_coherence(freq1, freq2)    # Wave coherence between two frequencies
me.sacred_alignment(frequency)     # Check sacred alignment of a frequency

# Direct layer access:
me.pure_math.prime_sieve(n)        # PureMath layer
me.god_code.*                      # GodCodeDerivation layer
me.harmonic.resonance_spectrum(fundamental, harmonics)  # HarmonicProcess layer
me.harmonic.verify_correspondences()                    # Fe/286Hz correspondence
me.harmonic.sacred_alignment(frequency)                 # Sacred alignment check
me.wave_physics.phi_power_sequence(n)                   # WavePhysics: φ^0..φ^(n-1)
me.dim_4d.*                        # Math4D layer (static: lorentz_boost_x/y/z)
me.dim_5d.*                        # Math5D layer
me.manifold.*                      # ManifoldEngine layer
me.void_math.*                     # VoidMath layer
me.abstract.*                      # AbstractAlgebra layer
me.ontological.*                   # OntologicalMath layer
me.proofs.*                        # SovereignProofs (static methods only)
me.hyper.*                         # HyperdimensionalEngine layer
```

## Quantum Gate Engine Quick Reference

```python
from l104_quantum_gate_engine import get_engine, GateCircuit, GateCompiler
from l104_quantum_gate_engine import H, CNOT, Rx, Rz, PHI_GATE, GOD_CODE_PHASE
from l104_quantum_gate_engine import ErrorCorrectionScheme, ExecutionTarget, OptimizationLevel, GateSet

engine = get_engine()  # Singleton orchestrator

# Circuit building
circ = engine.bell_pair()                              # Bell state (H + CNOT)
circ = engine.ghz_state(5)                             # GHZ state (N qubits)
circ = engine.quantum_fourier_transform(4)             # QFT (4 qubits)
circ = engine.sacred_circuit(3, depth=4)               # Sacred L104 circuit
circ = engine.create_circuit(2, "custom")              # Custom circuit
circ.h(0).cx(0, 1).append(PHI_GATE, [0])              # Build with gate calls

# Compile (4 optimization levels, 6 target gate sets)
result = engine.compile(circ, GateSet.IBM_EAGLE, OptimizationLevel.O2)
result = engine.compile(circ, GateSet.CLIFFORD_T)      # Fault-tolerant decomposition
result = engine.compile(circ, GateSet.L104_SACRED, OptimizationLevel.O3)

# Error correction
protected = engine.error_correction.encode(circ, ErrorCorrectionScheme.SURFACE_CODE, distance=3)
protected = engine.error_correction.encode(circ, ErrorCorrectionScheme.STEANE_7_1_3)
protected = engine.error_correction.encode(circ, ErrorCorrectionScheme.FIBONACCI_ANYON)

# Execute (8 targets: local statevector, Qiskit Aer, IBM QPU, coherence engine, ASI, ...)
result = engine.execute(circ, ExecutionTarget.LOCAL_STATEVECTOR)
result.probabilities  # {'00': 0.5, '11': 0.5}
result.sacred_alignment  # GOD_CODE resonance score

# Full pipeline: build → compile → protect → execute → analyze
pipeline = engine.full_pipeline(circ, target_gates=GateSet.UNIVERSAL,
    optimization=OptimizationLevel.O2,
    error_correction=ErrorCorrectionScheme.STEANE_7_1_3,
    execution_target=ExecutionTarget.LOCAL_STATEVECTOR)

# Gate algebra (40+ gates, decomposition, analysis)
algebra = engine.algebra
algebra.zyz_decompose(gate.matrix)                     # ZYZ Euler decomposition
algebra.kak_decompose(two_qubit_gate.matrix)            # KAK/Cartan decomposition
algebra.pauli_decompose(gate.matrix)                    # Pauli basis decomposition
algebra.sacred_alignment_score(PHI_GATE)                # Sacred resonance analysis
analysis = engine.analyze_gate(CNOT)                    # Full gate analysis
```

## Code Engine Quick Reference

```python
from l104_code_engine import code_engine

# Top-level hub methods (NOT sub-objects like .doc_gen or .test_gen)
code_engine.full_analysis(code)                        # Full code analysis
code_engine.generate_docs(source, style, language)     # Documentation generation
code_engine.generate_tests(source, language, framework)# Test scaffolding
code_engine.auto_fix_code(source)                      # Auto-fix → (fixed, log)
code_engine.smell_detector.detect_all(code)            # Code smell detection
code_engine.perf_predictor.predict_performance(code)   # Performance prediction
code_engine.refactor_engine.refactor_analyze(source)   # Refactor opportunities
code_engine.excavator.excavate(source)                 # Dead code archaeology
code_engine.translate_code(src, from_l, to_l)          # Translation
code_engine.audit_app(path, auto_remediate=True)       # 10-layer audit
code_engine.scan_workspace(path)                       # Workspace census
await code_engine.optimize(code)                       # Optimization (async)
```

## Cross-Engine Debug Suite

**File**: `cross_engine_debug.py` — 41 tests, 7 phases, validates all 3 engines together

| Phase | Tests | What It Validates |
|-------|-------|-------------------|
| 1 - Parallel Boot | 3 | All engines initialize concurrently in threads |
| 2 - Constants | 7 | GOD_CODE, PHI, VOID_CONSTANT match across all engines |
| 3 - Science→Math | 6 | Physics outputs fed to math functions |
| 4 - Math→Science | 6 | Math outputs fed to science functions |
| 5 - Code→Both | 6 | Code engine analyzes science/math source code |
| 6 - Both→Code | 6 | Science/math data used for code generation/testing |
| 7 - Integration | 7 | Full pipeline: physics→god-code→code-gen→analysis |

Run: `.venv/bin/python cross_engine_debug.py`

## Three-Engine Upgrade Suite

**File**: `three_engine_upgrade.py` — 8 phases, uses all 3 engines to analyze + upgrade ASI/AGI

| Phase | What It Does |
|-------|-------------|
| 1 | Parallel engine boot (Code + Science + Math) |
| 2 | Code Engine analysis of ASI + AGI (smells, perf, complexity) |
| 3 | Math Engine validation (GOD_CODE, Fibonacci→PHI, Lorentz, harmonics) |
| 4 | Science Engine validation (entropy reversal, coherence, 26Q, physics) |
| 5 | Cross-Engine synthesis (complexity×demon, proof→quantum, calibration) |
| 6 | Upgrade report generation (`three_engine_upgrade_report.json`) |
| 7 | Code generation (docs, tests, auto-fix for both cores) |
| 8 | Final cross-engine verification (7 checks) |

Run: `.venv/bin/python three_engine_upgrade.py`

## Three-Engine Integration (v8.0/v57.0)

Both ASI (v8.0) and AGI (v57.0) cores now include three-engine integration:

```python
# Three-engine scoring (available on both agi_core and asi_core)
core.three_engine_entropy_score()         # Science Engine: Maxwell Demon efficiency
core.three_engine_harmonic_score()        # Math Engine: GOD_CODE alignment + wave coherence
core.three_engine_wave_coherence_score()  # Math Engine: PHI-harmonic phase-lock
core.three_engine_status()                # Status of all three engine connections

# AGI: 13-dimension scoring (was 10D)
agi_core.compute_10d_agi_score()  # D0-D9 original + D10 entropy + D11 harmonic + D12 wave

# ASI: 15-dimension scoring (was 12D)
asi_core.compute_asi_score()      # 12 original + entropy_reversal + harmonic_resonance + wave_coherence
```

## Detailed Docs

| Path | Content |
|------|---------|
| `docs/claude/architecture.md` | Cognitive architecture, MCP config, agents, EVO history |
| `docs/claude/code-engine.md` | Code Engine v6.2.0 — full API, 31 subsystems, 10-layer audit |
| `docs/claude/swift-app.md` | L104SwiftApp build system, 87 Swift source files |
| `docs/claude/evolved-asi-files.md` | ASI evolution log, decomposed package details |
| `docs/claude/api-reference.md` | FastAPI endpoints and server routes |
| `docs/claude/guides/code-examples.md` | Practical code patterns |
| `docs/claude/guides/memory-persistence.md` | State file management |
| `docs/claude/guides/optimization.md` | Performance tuning |
| `docs/claude/guides/zenith-patterns.md` | Zenith frequency patterns |

## Codebase Metrics

- **736** Python files at root, **717** L104 modules
- **81** modules across 8 decomposed packages (82,251 lines)
- **87** Swift files (66,891 lines) in L104SwiftApp
- **35** `.l104_*.json` state files
- **273** API route handlers in `l104_server/app.py`
