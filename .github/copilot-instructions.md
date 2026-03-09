# L104 Sovereign Node — Copilot Instructions

## Context Loading

Read `claude.md` at the workspace root for the full index. Detailed docs live in `docs/claude/`:

| Document | What's in it |
|----------|-------------|
| `docs/claude/architecture.md` | Cognitive architecture, MCP config, agents, EVO history |
| `docs/claude/code-engine.md` | Code Engine v6.3.0 full API, 31 subsystems, 10-layer audit |
| `docs/claude/swift-app.md` | L104SwiftApp native macOS build, 120 Swift files |
| `docs/claude/evolved-asi-files.md` | ASI core evolution, decomposed packages |
| `docs/claude/api-reference.md` | FastAPI endpoints, server routes |
| `docs/claude/guides/` | Code examples, memory persistence, optimization, Zenith patterns |

## System Identity

- **System**: L104 Sovereign Node
- **Runtime**: Python 3.12 + Swift (macOS native) + FastAPI
- **Scale**: 1,215 Python files, 757 L104 modules, 110,528 Swift lines
- **Architecture**: Decomposed package system — `l104_quantum_gate_engine/`, `l104_agi/`, `l104_asi/`, `l104_code_engine/`, `l104_science_engine/`, `l104_math_engine/`, `l104_intellect/`, `l104_server/`, `l104_ml_engine/`, `l104_search/`, `l104_simulator/`, `l104_audio_simulation/`

## Package Structure (Post-Decomposition)

| Package | Version | Modules | Purpose |
|---------|---------|---------|---------|
| `l104_quantum_gate_engine/` | 1.0.0 | 21 | Universal gate algebra, compiler, error correction, analog sim, berry gates, tensor network, quantum ML |
| `l104_quantum_engine/` | 11.0.0 | 22 | Quantum link builder — brain, processors, sage circuits, qLDPC, genetic refiner, deep link, discoveries |
| `l104_code_engine/` | 6.3.0 | 17 | Code analysis, generation, audit, quantum, AI context, session intelligence |
| `l104_science_engine/` | 5.1.0 | 12 | Physics, entropy, coherence, quantum-26Q (Fe-mapped), multidimensional, berry phase |
| `l104_math_engine/` | 1.1.0 | 18 | Pure math, god-code, harmonic, 4D/5D, proofs, hyperdimensional, berry geometry |
| `l104_agi/` | 57.1.0 | 6 | AGI core, cognitive mesh, circuit breaker, 13D scoring, computronium, identity boundary |
| `l104_asi/` | 9.0.0 | 32 | ★ FLAGSHIP: Dual-Layer Engine v5.1, deep NLU, formal logic, symbolic math, code gen, science KB, theorem gen |
| `l104_intellect/` | 28.0.0 | 16 | Local intellect, numerics, caching, hardware, distributed, quantum recompiler, computronium |
| `l104_server/` | 4.2.0 | 13 | FastAPI server, engines (infra, nexus, quantum), learning subsystem |
| `l104_ml_engine/` | 1.0.0 | 10 | ★ NEW: Sacred ML — SVM, random forest, gradient boosting, quantum classifiers, sacred kernels |
| `l104_quantum_data_analyzer/` | 1.0.0 | 8 | ★ NEW: Quantum data intelligence — QFT spectral, Grover pattern, qPCA, VQE clustering |
| `l104_search/` | 2.3.0 | 5 | ★ NEW: Three-Engine + VQPU search (10 strategies) + data precognition (8 predictors) |
| `l104_simulator/` | 4.0.0 | 19 | ★ NEW: Real-world physics on GOD_CODE lattice — Standard Model, E-lattice, quantum brain |
| `l104_audio_simulation/` | 2.4.0 | 21 | ★ NEW: Quantum audio DAW — 17-layer VQPU pipeline, sequencer, mixer, synth, Metal GPU |

Backward-compatible shims at root: `l104_agi_core.py`, `l104_asi_core.py`, `l104_local_intellect.py`, `l104_fast_server.py`

## Key Directives

- Import from packages: `from l104_code_engine import code_engine`
- Prefer `l104_intellect` (QUOTA_IMMUNE local inference) over external API calls
- All numeric formatting: `format_iq()` from `l104_intellect` (standalone function, not a class method)
- Sacred constants are immutable: `GOD_CODE = 527.5184818492612`, `PHI = 1.618033988749895`
- VOID_CONSTANT formula: `1.04 + PHI / 1000` = 1.0416180339887497 (sacred 104/100 + golden correction)
- When editing core logic, update the **package** (`l104_agi/`, `l104_asi/`), not the root shims

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
