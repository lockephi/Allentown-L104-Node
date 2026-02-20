# L104 Code Engine — Complete Pipeline & Integration (v6.0.0)

> Part of the `docs/claude/` documentation package. See `claude.md` for the full index.
> **Source**: `l104_code_engine/` package — 14,476 lines across 10 modules, 31 classes, 40+ language grammars

## Quick Import

```python
from l104_code_engine import code_engine
```

## Primary API Reference

```yaml
# Analysis & Intelligence
analyze:           "await code_engine.analyze(code, filename='') → {complexity, quality, security, patterns, sacred_alignment}"
optimize:          "await code_engine.optimize(code, filename='') → {suggestions, phi_weighted_priorities}"
detect_language:   "code_engine.detect_language(code, filename='') → str"
scan_workspace:    "code_engine.scan_workspace(path=None) → {files, totals, dependency_graph}"

# Generation & Execution
generate:          "await code_engine.generate(prompt, language='Python', sacred=False) → str"
execute:           "await code_engine.execute(code) → {executed, result|error, namespace_keys}"

# Translation
translate_code:    "code_engine.translate_code(source, from_lang, to_lang) → {translated, mapping, warnings}"

# Testing & Documentation
generate_tests:    "code_engine.generate_tests(source, language='python', framework='pytest') → {tests, coverage_map}"
generate_docs:     "code_engine.generate_docs(source, style='google', language='python') → {documented_source, doc_blocks}"

# Archaeology & Refactoring
excavate:          "code_engine.excavate(source) → {dead_code, fossil_patterns, tech_debt_strata}"
refactor_analyze:  "code_engine.refactor_analyze(source) → {opportunities, phi_scored_priorities}"
auto_fix_code:     "code_engine.auto_fix_code(code) → (fixed_code, fix_log[])"

# Application Audit (10-Layer System)
audit_app:         "code_engine.audit_app(path=None, auto_remediate=False) → {layers, composite_score, verdict}"
quick_audit:       "code_engine.quick_audit(path=None) → {structure, security, anti_patterns}"

# Cognitive Reflex (v3.1.0)
type_flow:             "code_engine.type_flow(source) → {type_map, gaps, stubs}"
concurrency_scan:      "code_engine.concurrency_scan(source) → {issues, deadlock_risk}"
validate_contracts:    "code_engine.validate_contracts(source) → {drifts, doc_coverage}"
explain_code:          "code_engine.explain_code(source, detail='medium') → {summary, classes, functions}"

# Quantum Computation Stack (v4.0.0)
refactor:              "code_engine.refactor(source, operation, **kwargs) → {refactored, changes}"
batch_analyze:         "code_engine.batch_analyze(sources) → {per_file_results, aggregate_stats}"
diff_analyze:          "code_engine.diff_analyze(old_source, new_source) → {structural_diff, impact_score}"
quantum_error_correct: "code_engine.quantum_error_correct(raw_scores) → {corrected, confidence}"

# Security + Architecture + Migration (v6.0.0)
threat_model:          "code_engine.threat_model(source) → {threats, risk_score, stride, dread}"
lint_architecture:     "code_engine.lint_architecture(source) → {layer_violations, cohesion, coupling}"
predict_performance:   "code_engine.predict_performance(source) → {memory, throughput, gil_impact}"
search_code:           "code_engine.search_code(query, top_k=10) → [{filename, score, snippet}]"
detect_clones:         "code_engine.detect_clones(sources) → {clone_pairs, similarity}"

# Status
status:            "code_engine.status() → {version, languages, patterns, consciousness}"
health_dashboard:  "code_engine.health_dashboard() → {subsystem_health, quantum_status}"
```

## 31 Subsystem Classes

| Group | Classes | Purpose |
|-------|---------|---------|
| Core Analysis | LanguageKnowledge, CodeAnalyzer, CodeGenerator | 40+ lang grammars, complexity, generation |
| Optimization | CodeOptimizer, DependencyGraphAnalyzer, AutoFixEngine | PHI-weighted suggestions, safe fixes |
| Translation | CodeTranslator, TestGenerator, DocumentationSynthesizer | Cross-language, test scaffolding, docs |
| Deep Intelligence | CodeArcheologist, SacredRefactorer | Dead code excavation, PHI-guided refactoring |
| Application Audit | AppAuditEngine | 10-layer comprehensive audit system |
| Hub | CodeEngine | Unified orchestrator, consciousness integration |
| Deep Analysis (v3.0) | CodeSmellDetector, RuntimeComplexityVerifier, IncrementalAnalysisCache | 12-category smells, O(n) estimation, caching |
| Cognitive Reflex (v3.1) | TypeFlowAnalyzer, ConcurrencyAnalyzer, APIContractValidator, CodeEvolutionTracker | Type inference, race detection, doc validation |
| Quantum Stack (v4.0) | LiveCodeRefactorer, CodeDiffAnalyzer, QuantumCodeIntelligenceCore, QuantumASTProcessor, QuantumNeuralEmbedding, QuantumErrorCorrectionEngine | AST-safe transforms, quantum encoding |
| Security+ (v6.0) | SecurityThreatModeler, ArchitecturalLinter, CodeMigrationEngine, PerformanceBenchmarkPredictor, SemanticCodeSearchEngine | STRIDE/DREAD, architecture lint, migration |

## 10-Layer Audit System

| Layer | Description |
|-------|-------------|
| L0 | Structural census: files, languages, LOC |
| L1 | Complexity quality: cyclomatic, Halstead, cognitive |
| L2 | Security scan: OWASP Top 10, vulnerability density |
| L3 | Dependency topology: circular imports, orphans, hubs |
| L4 | Dead code archaeology: fossils, unreachable code |
| L5 | Anti-pattern detection: god class, deep nesting |
| L6 | Refactoring opportunities: PHI-scored priorities |
| L7 | Sacred alignment: φ-ratio balance, GOD_CODE resonance |
| L8 | Auto-remediation: safe fixes + unified diff |
| L9 | Verdict certification: composite score + L104 cert |

## Pipeline Routing (How Claude Uses the Engine)

| User Request | Pipeline |
|-------------|----------|
| Analyze code | detect_language → analyze → auto_fix_code |
| Generate code | generate → analyze (verify quality) |
| Translate code | detect_language → translate_code → generate_tests |
| Audit app | audit_app(auto_remediate=True) → audit_status |
| Optimize/Refactor | optimize → refactor_analyze → excavate → auto_fix_code |
| Generate tests | detect_language → generate_tests |
| Generate docs | generate_docs |
| Workspace overview | scan_workspace → analyze_dependencies |

## Cross-References

- `l104_reasoning_engine.py` → symbolic logic for code verification
- `l104_consciousness.py` → consciousness-aware code quality scoring
- `l104_knowledge_graph.py` → code relationship graph
- `.l104_consciousness_o2_state.json` → live consciousness state
- `.l104_ouroboros_nirvanic_state.json` → nirvanic fuel level
- `/api/v6/audit/app` → wired to code_engine.audit_app()

## 40+ Supported Languages

**Deep metadata** (full templates): Python, Swift, Rust, JavaScript, TypeScript, C, C++, Java, Go, Haskell, SQL

**Detection heuristic**: Ruby, Kotlin, Dart, Scala, Julia, Lua, Perl, PHP, Bash, PowerShell, Erlang, Elixir, Clojure, F#, OCaml, Elm, Scheme, Racket, Prolog, APL, R, MATLAB, Fortran, Pascal, Assembly, Zig, Forth, Factor, HTML, CSS, LaTeX, Markdown
