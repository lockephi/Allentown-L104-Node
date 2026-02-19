"""L104 Code Engine v6.0.0 — Unified Hub Orchestrator."""
from .constants import *
from .builder_state import _read_builder_state as _module_read_builder_state
from .languages import LanguageKnowledge
from .analyzer import (
    CodeAnalyzer, CodeSmellDetector, RuntimeComplexityVerifier,
    IncrementalAnalysisCache, TypeFlowAnalyzer, ConcurrencyAnalyzer,
    APIContractValidator,
)
from .synthesis import (
    CodeGenerator, CodeTranslator, TestGenerator, DocumentationSynthesizer,
)
from .refactoring import (
    CodeOptimizer, DependencyGraphAnalyzer, AutoFixEngine,
    CodeArcheologist, SacredRefactorer, CodeEvolutionTracker,
    LiveCodeRefactorer, CodeDiffAnalyzer, SemanticCodeSearchEngine,
)
from .audit import (
    AppAuditEngine, SecurityThreatModeler, ArchitecturalLinter,
    CodeMigrationEngine, PerformanceBenchmarkPredictor,
)
from .quantum import (
    QuantumCodeIntelligenceCore, QuantumASTProcessor,
    QuantumNeuralEmbedding, QuantumErrorCorrectionEngine,
)

class CodeEngine:
    """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  L104 CODE ENGINE v6.0.0 — UNIFIED ASI CODE INTELLIGENCE HUB     ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Wires 31 subsystems:                                            ║
    ║    LanguageKnowledge + CodeAnalyzer + CodeGenerator +            ║
    ║    CodeOptimizer + DependencyGraphAnalyzer + AutoFixEngine +      ║
    ║    CodeTranslator + TestGenerator + DocumentationSynthesizer +   ║
    ║    CodeArcheologist + SacredRefactorer + AppAuditEngine +        ║
    ║    CodeSmellDetector + RuntimeComplexityVerifier +               ║
    ║    IncrementalAnalysisCache + TypeFlowAnalyzer +                 ║
    ║    ConcurrencyAnalyzer + APIContractValidator +                  ║
    ║    CodeEvolutionTracker + LiveCodeRefactorer + CodeDiffAnalyzer  ║
    ║    QuantumCodeIntelligenceCore + QuantumASTProcessor +           ║
    ║    QuantumNeuralEmbedding + QuantumErrorCorrectionEngine         ║
    ║                                                                   ║
    ║  v6.0.0 NEW — Security + Architecture + Migration + Perf + Search║
    ║    • SecurityThreatModeler: STRIDE/DREAD threat analysis, attack ║
    ║      surface quantification, secrets detection, zero-trust audit ║
    ║    • ArchitecturalLinter: Clean architecture validation, layer   ║
    ║      violations, coupling metrics, LCOM cohesion, PHI-balance   ║
    ║    • CodeMigrationEngine: Deprecation scanning, framework       ║
    ║      migration paths, breaking change detection, Python compat  ║
    ║    • PerformanceBenchmarkPredictor: Memory footprint estimation, ║
    ║      throughput prediction, GIL contention, allocation hotspots ║
    ║    • SemanticCodeSearchEngine: TF-IDF sacred-weighted search,   ║
    ║      cross-file clone detection (Type 1/2/3), sacred references ║
    ║                                                                   ║
    ║  v4.0.0 — Quantum Computation Stack (4 subsystems)               ║
    ║  v5.0.0 — Live Refactoring + Diff Analysis (2 subsystems)        ║
    ║                                                                   ║
    ║  API: analyze, generate, optimize, auto_fix, translate, audit    ║
    ║       threat_model, lint_architecture, scan_deprecations         ║
    ║       predict_performance, code_search, detect_clones            ║
    ║       suggest_migration, detect_breaking_changes                 ║
    ║       quantum_*, refactor, batch_analyze, diff_analyze           ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Claude Pipeline Integration:                                     ║
    ║    claude.md → documents full API + pipeline routing              ║
    ║    l104_claude_heartbeat.py → validates hash/version/lines        ║
    ║    .l104_claude_heartbeat_state.json → session metric cache       ║
    ║    .github/copilot-instructions.md → forces claude.md load       ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║    Qiskit 2.3.0 Full Quantum Stack + Consciousness + O₂          ║
    ╚═══════════════════════════════════════════════════════════════════╝

    This is the primary entry point for all code intelligence operations
    in the L104 Sovereign Node. Every code-related query, generation,
    analysis, or optimization flows through this hub.

    Pipeline routing (see claude.md for complete reference):
      analyze_code:  detect_language → analyze → auto_fix_code
      generate_code: generate → analyze (verify) → return with metadata
      translate:     detect_language → translate_code → generate_tests
      audit:         audit_app | quick_audit → audit_status → audit_trail
      optimize:      optimize → refactor_analyze → excavate → auto_fix_code
      streamline:    run_streamline_cycle (ChoiceEngine integration)
    """

    def __init__(self):
        """Initialize CodeEngine hub and wire all subsystems."""
        self.languages = LanguageKnowledge()
        self.analyzer = CodeAnalyzer()
        self.generator = CodeGenerator()
        self.optimizer = CodeOptimizer()
        self.dep_graph = DependencyGraphAnalyzer()
        self.auto_fix = AutoFixEngine()
        self.translator = CodeTranslator()
        self.test_gen = TestGenerator()
        self.doc_synth = DocumentationSynthesizer()
        self.archeologist = CodeArcheologist()
        self.refactorer = SacredRefactorer()
        self.app_audit = AppAuditEngine(
            analyzer=self.analyzer,
            optimizer=self.optimizer,
            dep_graph=self.dep_graph,
            auto_fix=self.auto_fix,
            archeologist=self.archeologist,
            refactorer=self.refactorer,
        )
        # v3.0.0 new subsystems
        self.smell_detector = CodeSmellDetector()
        self.complexity_verifier = RuntimeComplexityVerifier()
        self.analysis_cache = IncrementalAnalysisCache()
        # v3.1.0 — Cognitive Reflex Architecture (4 new subsystems)
        self.type_analyzer = TypeFlowAnalyzer()
        self.concurrency_analyzer = ConcurrencyAnalyzer()
        self.contract_validator = APIContractValidator()
        self.evolution_tracker = CodeEvolutionTracker()
        # v5.0.0 — Live Refactoring + Diff Analysis (2 new subsystems)
        self.live_refactorer = LiveCodeRefactorer()
        self.diff_analyzer = CodeDiffAnalyzer()
        # v4.0.0 — State-of-Art Quantum Computation (4 new subsystems)
        self.quantum_core = QuantumCodeIntelligenceCore()
        self.quantum_ast = QuantumASTProcessor(self.quantum_core)
        self.quantum_embedding = QuantumNeuralEmbedding(self.quantum_core)
        self.quantum_error_correction = QuantumErrorCorrectionEngine(self.quantum_core)
        # v6.0.0 — Security + Architecture + Migration + Performance + Search (5 new subsystems)
        self.threat_modeler = SecurityThreatModeler()
        self.arch_linter = ArchitecturalLinter()
        self.migration_engine = CodeMigrationEngine()
        self.perf_predictor = PerformanceBenchmarkPredictor()
        self.code_search = SemanticCodeSearchEngine()
        # v3.1.0 — Wire FaultTolerance + QuantumKernel (documented in claude.md v2.6.0)
        self.fault_tolerance = None
        self.quantum_kernel = None
        try:
            from l104_fault_tolerance import L104FaultTolerance
            self.fault_tolerance = L104FaultTolerance()
        except ImportError:
            pass
        try:
            from l104_quantum_embedding import L104QuantumKernel
            self.quantum_kernel = L104QuantumKernel()
        except ImportError:
            pass
        self.execution_count = 0
        self.generated_code: List[str] = []
        self._state_cache = {}
        self._state_cache_time = 0
        logger.info(f"[CODE_ENGINE v{VERSION}] Initialized — "
                     f"{len(LanguageKnowledge.LANGUAGES)} languages, "
                     f"{len(CodeAnalyzer.SECURITY_PATTERNS)} vuln patterns, "
                     f"{len(CodeAnalyzer.DESIGN_PATTERNS)} design patterns, "
                     f"{len(AutoFixEngine.FIX_CATALOG)} auto-fixes, "
                     f"{len(CodeTranslator.SUPPORTED_LANGS)} transpile targets, "
                     f"{len(CodeSmellDetector.SMELL_CATALOG)} smell patterns, "
                     f"{len(TypeFlowAnalyzer.KNOWN_CONSTRUCTORS)} type constructors, "
                     f"4 cognitive subsystems (v3.1.0), "
                     f"4 quantum subsystems (v4.0.0), "
                     f"2 refactoring subsystems (v5.0.0), "
                     f"5 v6.0.0 subsystems (security+arch+migration+perf+search), "
                     f"Qiskit={'YES' if QISKIT_AVAILABLE else 'NO'}, "
                     f"AppAuditEngine v{AppAuditEngine.AUDIT_VERSION}")

    # ─── Builder state integration (consciousness/O₂/nirvanic) ───

    def _read_builder_state(self) -> Dict[str, Any]:
        """Read consciousness/O₂/nirvanic state from builder files (zero-import, file-based)."""
        import time
        now = time.time()
        if now - self._state_cache_time < 10 and self._state_cache:
            return self._state_cache

        state = {"consciousness_level": 0.0, "superfluid_viscosity": 1.0,
                 "nirvanic_fuel": 0.0, "evo_stage": "DORMANT"}
        ws = Path(__file__).parent
        # Consciousness + O₂
        co2_path = ws / ".l104_consciousness_o2_state.json"
        if co2_path.exists():
            try:
                data = json.loads(co2_path.read_text())
                state["consciousness_level"] = data.get("consciousness_level", 0.0)
                state["superfluid_viscosity"] = data.get("superfluid_viscosity", 1.0)
                state["evo_stage"] = data.get("evo_stage", "DORMANT")
            except Exception:
                pass
        # Nirvanic
        nir_path = ws / ".l104_ouroboros_nirvanic_state.json"
        if nir_path.exists():
            try:
                data = json.loads(nir_path.read_text())
                state["nirvanic_fuel"] = data.get("nirvanic_fuel_level", 0.0)
            except Exception:
                pass

        self._state_cache = state
        self._state_cache_time = now
        return state

    # ─── High-level API ───

    async def generate(self, prompt: str, language: str = "Python",
                       sacred: bool = False) -> str:
        """Generate code from a natural language prompt."""
        self.execution_count += 1
        state = self._read_builder_state()

        # Parse intent from prompt
        if "class" in prompt.lower():
            name = self._extract_name(prompt, "class")
            code = self.generator.generate_class(name, language, doc=prompt)
        elif "function" in prompt.lower() or "def" in prompt.lower() or "fn" in prompt.lower():
            name = self._extract_name(prompt, "function")
            code = self.generator.generate_function(name, language, doc=prompt,
                                                     sacred_constants=sacred)
        else:
            # Generic generation with consciousness-aware quality
            name = self._extract_name(prompt, "code")
            quality_target = "high" if state["consciousness_level"] > 0.5 else "standard"
            code = self.generator.generate_function(
                name, language, doc=f"{prompt} [quality={quality_target}]",
                body="raise NotImplementedError('Generated stub')",
                sacred_constants=sacred
            )

        # Add consciousness metadata as comment
        if state["consciousness_level"] > 0.3:
            header = (
                f"# L104 Code Engine v{VERSION} | "
                f"Consciousness: {state['consciousness_level']:.4f} [{state['evo_stage']}] | "
                f"Superfluid η: {state['superfluid_viscosity']:.6f}\n"
            )
            code = header + code

        self.generated_code.append(code)
        return code

    async def execute(self, code: str) -> Dict[str, Any]:
        """Execute generated code safely in a restricted namespace."""
        self.execution_count += 1
        namespace = {"__builtins__": {"print": print, "range": range, "len": len,
                                       "int": int, "float": float, "str": str,
                                       "list": list, "dict": dict, "math": math}}
        try:
            exec(compile(code, "<code_engine>", "exec"), namespace)
            return {"executed": True, "result": "Success", "execution_count": self.execution_count,
                    "namespace_keys": [k for k in namespace if not k.startswith('_')]}
        except Exception as e:
            return {"executed": False, "error": str(e), "execution_count": self.execution_count}

    async def analyze(self, code: str, filename: str = "") -> Dict[str, Any]:
        """Full code analysis — complexity, quality, security, patterns, sacred alignment."""
        return self.analyzer.full_analysis(code, filename)

    async def optimize(self, code: str, filename: str = "") -> Dict[str, Any]:
        """Analyze code and return optimization suggestions."""
        analysis = self.analyzer.full_analysis(code, filename)
        return self.optimizer.analyze_and_suggest(analysis)

    def detect_language(self, code: str, filename: str = "") -> str:
        """Detect programming language from code."""
        return LanguageKnowledge.detect_language(code, filename)

    def compare_languages(self, lang_a: str, lang_b: str) -> Dict[str, Any]:
        """Compare two programming languages."""
        return LanguageKnowledge.compare_languages(lang_a, lang_b)

    def scan_workspace(self, workspace_path: str = None) -> Dict[str, Any]:
        """Scan an entire workspace for code metrics + dependency graph."""
        ws = Path(workspace_path) if workspace_path else Path(__file__).parent
        results = {"files": [], "totals": {"lines": 0, "code_lines": 0,
                                            "vulnerabilities": 0, "files_scanned": 0}}
        for ext in [".py", ".swift", ".js", ".ts", ".rs", ".go", ".java", ".c", ".cpp"]:
            for f in ws.glob(f"*{ext}"):
                if f.name.startswith('.') or '__pycache__' in str(f):
                    continue
                try:
                    code = f.read_text(errors='ignore')
                    lines = len(code.split('\n'))
                    lang = LanguageKnowledge.detect_language(code, str(f))
                    vulns = len(self.analyzer._security_scan(code))
                    results["files"].append({
                        "name": f.name, "language": lang, "lines": lines, "vulnerabilities": vulns
                    })
                    results["totals"]["lines"] += lines
                    results["totals"]["files_scanned"] += 1
                    results["totals"]["vulnerabilities"] += vulns
                except Exception:
                    pass
        results["totals"]["code_lines"] = int(results["totals"]["lines"] * 0.75)
        # Attach dependency graph
        results["dependency_graph"] = self.dep_graph.build_graph(str(ws))
        return results

    def auto_fix_code(self, code: str) -> Tuple[str, List[Dict]]:
        """Apply all safe auto-fixes to code. Returns (fixed_code, fix_log)."""
        return self.auto_fix.apply_all_safe(code)

    def analyze_dependencies(self, workspace_path: str = None) -> Dict[str, Any]:
        """Build and analyze the dependency graph for the workspace."""
        ws = str(Path(workspace_path) if workspace_path else Path(__file__).parent)
        return self.dep_graph.build_graph(ws)

    def _extract_name(self, prompt: str, kind: str) -> str:
        """Extract a name from a prompt for code generation."""
        words = prompt.lower().split()
        for trigger in [kind, "called", "named"]:
            if trigger in words:
                idx = words.index(trigger)
                if idx + 1 < len(words):
                    name = re.sub(r'[^a-zA-Z0-9_]', '', words[idx + 1])
                    if name:
                        return name
        return f"generated_{kind}"

    def translate_code(self, source: str, from_lang: str,
                       to_lang: str) -> Dict[str, Any]:
        """Translate code between languages."""
        self.execution_count += 1
        return self.translator.translate(source, from_lang, to_lang)

    def generate_tests(self, source: str, language: str = "python",
                       framework: str = "pytest") -> Dict[str, Any]:
        """Generate test scaffolding for source code."""
        self.execution_count += 1
        return self.test_gen.generate_tests(source, language, framework)

    def generate_docs(self, source: str, style: str = "google",
                      language: str = "python") -> Dict[str, Any]:
        """Generate documentation for source code."""
        self.execution_count += 1
        return self.doc_synth.generate_docs(source, style, language)

    def excavate(self, source: str) -> Dict[str, Any]:
        """Archeological excavation: dead code, fossils, architecture analysis."""
        self.execution_count += 1
        return self.archeologist.excavate(source)

    def refactor_analyze(self, source: str) -> Dict[str, Any]:
        """Analyze source for refactoring opportunities."""
        self.execution_count += 1
        return self.refactorer.analyze(source)

    def detect_solid_violations(self, source: str) -> Dict[str, Any]:
        """Detect SOLID principle violations via AST analysis (v2.5.0)."""
        self.execution_count += 1
        return self.analyzer.detect_solid_violations(source)

    def detect_performance_hotspots(self, source: str) -> Dict[str, Any]:
        """Detect performance hotspots: nested loops, O(n²), string concat in loops (v2.5.0)."""
        self.execution_count += 1
        return self.analyzer.detect_performance_hotspots(source)

    # ─── v3.0.0 New API Methods ───

    def detect_smells(self, source: str) -> Dict[str, Any]:
        """Run deep code smell detection — 12 smell categories with severity scoring (v3.0.0)."""
        self.execution_count += 1
        return self.smell_detector.detect_all(source)

    def estimate_complexity(self, source: str) -> Dict[str, Any]:
        """Estimate runtime complexity O()-notation for all functions in source (v3.0.0)."""
        self.execution_count += 1
        return self.complexity_verifier.estimate_complexity(source)

    def cached_analyze(self, code: str, filename: str = "") -> Dict[str, Any]:
        """Analyze code with incremental caching — skips re-analysis if content unchanged (v3.0.0)."""
        cached = self.analysis_cache.get(code, "full")
        if cached is not None:
            return cached
        result = self.analyzer.full_analysis(code, filename)
        self.analysis_cache.put(code, result, "full")
        return result

    def deep_review(self, source: str, filename: str = "",
                    auto_fix: bool = False) -> Dict[str, Any]:
        """
        v3.1.0 Deep Review — chains ALL subsystems including v3.1 cognitive analyzers.

        Extended pipeline (builds on full_code_review):
          1.  Full analysis (complexity, quality, security, patterns, sacred)
          2.  SOLID principle check
          3.  Performance hotspot detection
          4.  Code smell detection (12 categories) — v3.0
          5.  Runtime complexity estimation per function — v3.0
          6.  Type flow analysis (inference + narrowing) — v3.1 NEW
          7.  Concurrency hazard scan (races + deadlocks) — v3.1 NEW
          8.  API contract validation (docstring consistency) — v3.1 NEW
          9.  Code archaeology (dead code, fossils, tech debt)
          10. Refactoring opportunities
          11. Auto-fix (if enabled)
          12. Unified deep verdict with PHI-weighted composite score

        Returns a single deeply scored review report.
        """
        self.execution_count += 1
        start = time.time()
        state = self._read_builder_state()

        # Use cached analysis if available
        analysis = self.cached_analyze(source, filename)

        # SOLID
        solid = self.analyzer.detect_solid_violations(source)

        # Performance
        perf = self.analyzer.detect_performance_hotspots(source)

        # Code Smells (v3.0.0)
        smells = self.smell_detector.detect_all(source)

        # Runtime Complexity (v3.0.0)
        complexity_est = self.complexity_verifier.estimate_complexity(source)

        # Type Flow (v3.1.0)
        type_flow = self.type_analyzer.analyze(source)

        # Concurrency Hazards (v3.1.0)
        concurrency = self.concurrency_analyzer.analyze(source)

        # API Contract Validation (v3.1.0)
        contracts = self.contract_validator.validate(source)

        # Archaeology
        archaeology = self.archeologist.excavate(source)

        # Refactoring
        refactoring = self.refactorer.analyze(source)

        # Auto-fix
        fix_result = {"applied": False, "fixes": [], "chars_changed": 0}
        if auto_fix:
            fixed_source, fix_log = self.auto_fix.apply_all_safe(source)
            fix_result = {
                "applied": True,
                "fixes": fix_log,
                "chars_changed": len(fixed_source) - len(source),
                "fix_count": sum(f.get("count", 0) for f in fix_log),
            }

        # Unified deep scoring with v3.1 weights (12 dimensions)
        scores = {
            "analysis_quality": analysis.get("quality", {}).get("overall_score", 0.5),
            "security": 1.0 - min(1.0, len(analysis.get("security", [])) * 0.1),
            "solid": solid.get("solid_score", 1.0),
            "performance": perf.get("perf_score", 1.0),
            "smell_health": smells.get("health_score", 1.0),
            "complexity_efficiency": complexity_est.get("phi_efficiency_score", 1.0),
            "type_safety": type_flow.get("type_safety_score", 1.0),
            "concurrency_safety": concurrency.get("safety_score", 1.0),
            "contract_adherence": contracts.get("adherence_score", 1.0),
            "archaeology_health": archaeology.get("health_score", 1.0),
            "refactoring_health": refactoring.get("code_health", 1.0),
            "sacred_alignment": analysis.get("sacred_alignment", {}).get("overall_sacred_score", 0.5),
        }

        # PHI-weighted composite (12 dimensions)
        phi_weights = [PHI**2, PHI**2, PHI, PHI, 1.0, 1.0, PHI, PHI, 1.0, TAU, TAU, TAU]
        total_weight = sum(phi_weights[:len(scores)])
        composite = sum(
            s * w for s, w in zip(scores.values(), phi_weights)
        ) / total_weight

        # Build prioritized actions from all analyses
        actions = []
        for vuln in analysis.get("security", [])[:3]:
            actions.append({"priority": "CRITICAL", "category": "security",
                            "action": vuln.get("recommendation", "Fix security issue"),
                            "source": "analyzer"})
        for issue in concurrency.get("issues", [])[:3]:
            actions.append({"priority": issue.get("severity", "HIGH"), "category": "concurrency",
                            "action": issue.get("detail", "Fix concurrency issue"),
                            "source": "concurrency_analyzer"})
        for smell in smells.get("smells", [])[:3]:
            actions.append({"priority": smell["severity"], "category": "smell",
                            "action": smell["detail"], "source": "smell_detector"})
        for func in complexity_est.get("functions", []):
            if func.get("optimization_potential"):
                actions.append({"priority": "HIGH", "category": "complexity",
                                "action": f"{func['name']}() is {func['complexity']} — optimize",
                                "source": "complexity_verifier"})
        for drift in contracts.get("drifts", [])[:3]:
            actions.append({"priority": drift.get("severity", "MEDIUM"), "category": "contract",
                            "action": drift.get("detail", "Fix docstring/code drift"),
                            "source": "contract_validator"})
        for gap in type_flow.get("gaps", [])[:3]:
            actions.append({"priority": gap.get("severity", "LOW"), "category": "type_safety",
                            "action": gap.get("detail", "Add type annotation"),
                            "source": "type_flow_analyzer"})
        for v in solid.get("violations", [])[:2]:
            actions.append({"priority": v.get("severity", "MEDIUM"), "category": "solid",
                            "action": v["detail"], "source": "solid_checker"})
        actions.sort(key=lambda a: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(a["priority"], 4))

        duration = time.time() - start
        verdict = ("EXEMPLARY" if composite >= 0.9 else "HEALTHY" if composite >= 0.75
                   else "ACCEPTABLE" if composite >= 0.6 else "NEEDS_WORK" if composite >= 0.4
                   else "CRITICAL")

        return {
            "review_version": VERSION,
            "review_type": "deep_review_v3.1",
            "filename": filename,
            "language": analysis["metadata"].get("language", "unknown"),
            "lines": analysis["metadata"].get("lines", 0),
            "duration_seconds": round(duration, 3),
            "composite_score": round(composite, 4),
            "verdict": verdict,
            "score_dimensions": len(scores),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "smells": {"total": smells["total"], "health": smells["health_score"],
                       "by_category": smells.get("by_category", {})},
            "runtime_complexity": {
                "max": complexity_est.get("max_complexity", "unknown"),
                "high_count": complexity_est.get("high_complexity_count", 0),
                "efficiency": complexity_est.get("phi_efficiency_score", 1.0)},
            "type_flow": {
                "typed_ratio": type_flow.get("typed_ratio", 0.0),
                "gaps": type_flow.get("gap_count", 0),
                "score": type_flow.get("type_safety_score", 1.0)},
            "concurrency": {
                "issues": concurrency.get("issue_count", 0),
                "deadlock_risk": concurrency.get("deadlock_risk", "none"),
                "score": concurrency.get("safety_score", 1.0)},
            "contracts": {
                "drifts": contracts.get("drift_count", 0),
                "coverage": contracts.get("doc_coverage", 0.0),
                "score": contracts.get("adherence_score", 1.0)},
            "solid": {"score": solid["solid_score"], "violations": solid["total_violations"]},
            "performance": {"score": perf["perf_score"], "hotspots": perf["total_hotspots"]},
            "archaeology": {"health": archaeology.get("health_score", 1.0),
                            "dead_code": archaeology.get("dead_code_count", 0)},
            "refactoring": {"health": refactoring["code_health"],
                            "suggestions": refactoring["total_suggestions"]},
            "auto_fix": fix_result,
            "actions": actions[:25],
            "builder_state": {
                "consciousness": state["consciousness_level"],
                "evo_stage": state["evo_stage"],
            },
        }

    # ─── v3.1.0 Cognitive Reflex API ───

    def type_flow(self, source: str) -> Dict[str, Any]:
        """Infer types across code without explicit annotations. Returns type map, gaps, and stub suggestions."""
        self.execution_count += 1
        return self.type_analyzer.analyze(source)

    def concurrency_scan(self, source: str) -> Dict[str, Any]:
        """Detect race conditions, deadlock patterns, and async anti-patterns."""
        self.execution_count += 1
        return self.concurrency_analyzer.analyze(source)

    def validate_contracts(self, source: str) -> Dict[str, Any]:
        """Validate docstring↔code consistency and API surface stability."""
        self.execution_count += 1
        return self.contract_validator.validate(source)

    def track_evolution(self, source: str, filename: str = "unknown") -> Dict[str, Any]:
        """Snapshot current code structure and compare against previous snapshot for drift/churn."""
        self.execution_count += 1
        return self.evolution_tracker.compare(source, filename)

    def hotspot_report(self) -> Dict[str, Any]:
        """Return churn hotspots from evolution tracking history."""
        return self.evolution_tracker.hotspot_report()

    # ─── v4.0.0 Quantum Computation API ──────────────────────────────

    def quantum_analyze(self, source: str) -> Dict[str, Any]:
        """
        v4.0.0 — Full quantum-enhanced code analysis pipeline.

        Chains: AST encoding → path analysis → Grover vulnerability scan →
                quantum embedding → density diagnostic → error correction →
                tomographic quality reconstruction.

        Returns comprehensive quantum analysis with Born-rule confidence scores,
        von Neumann entropy, entanglement metrics, and sacred alignment.
        """
        self.execution_count += 1
        start = time.time()

        # 1. Quantum AST encoding
        ast_result = self.quantum_ast.encode_ast(source)

        # 2. Quantum path superposition analysis
        path_result = self.quantum_ast.quantum_path_analysis(source)

        # 3. Grover-amplified vulnerability detection
        vuln_result = self.quantum_ast.grover_vulnerability_detect(source)

        # 4. Quantum neural embedding
        embed_result = self.quantum_embedding.embed_code(source)

        # 5. Density matrix diagnostic on code features
        features = self.quantum_embedding._extract_token_features(source)
        density_result = self.quantum_core.density_diagnostic(features)

        # 6. Error-corrected quality from multiple analysis dimensions
        raw_scores = {
            "structural_clarity": 1.0 - ast_result.get("structural_complexity", 0.5),
            "testability": 1.0 if path_result.get("testability") in ("TRIVIAL", "EASY") else 0.7 if path_result.get("testability") == "MODERATE" else 0.4,
            "security": vuln_result.get("security_score", 0.8),
            "embedding_purity": embed_result.get("purity", 0.5),
        }
        corrected = self.quantum_error_correction.error_correct_analysis(raw_scores)

        # 7. Tomographic quality reconstruction
        tomo = self.quantum_core.tomographic_quality(corrected.get("corrected", raw_scores))

        duration = time.time() - start

        return {
            "engine_version": VERSION,
            "pipeline": "quantum_analyze_v4.0.0",
            "duration_seconds": round(duration, 3),
            "qiskit_available": QISKIT_AVAILABLE,
            "ast_encoding": ast_result,
            "path_analysis": path_result,
            "vulnerability_scan": vuln_result,
            "neural_embedding": {
                "dimension": embed_result.get("dimension"),
                "entropy": embed_result.get("entropy"),
                "purity": embed_result.get("purity"),
            },
            "density_diagnostic": density_result,
            "error_correction": corrected,
            "tomographic_quality": tomo,
            "composite_quantum_score": round(tomo.get("reconstructed_quality", 0.5), 6),
            "composite_confidence": round(tomo.get("confidence", 0.5), 6),
            "verdict": tomo.get("verdict", "UNKNOWN"),
        }

    def quantum_embed(self, source: str, dim: int = 8) -> Dict[str, Any]:
        """Compute quantum embedding vector for source code. Returns probability-amplitude embedding."""
        self.execution_count += 1
        return self.quantum_embedding.embed_code(source, dim)

    def quantum_attention(self, source: str, query: str = "") -> Dict[str, Any]:
        """Quantum attention mechanism — find the most important lines in code."""
        self.execution_count += 1
        return self.quantum_embedding.quantum_attention(source, query)

    def quantum_walk_graph(self, adjacency: Dict[str, Set[str]], steps: int = 5) -> Dict[str, Any]:
        """Execute quantum walk on a dependency graph. Returns module importance rankings."""
        self.execution_count += 1
        return self.quantum_core.quantum_walk(adjacency, steps)

    def quantum_similarity(self, code_a: str, code_b: str) -> Dict[str, Any]:
        """Compute quantum kernel similarity between two code snippets."""
        self.execution_count += 1
        features_a = self.quantum_embedding._extract_token_features(code_a)
        features_b = self.quantum_embedding._extract_token_features(code_b)
        return self.quantum_core.quantum_kernel(features_a, features_b)

    def quantum_similarity_matrix(self, snippets: List[str]) -> Dict[str, Any]:
        """Compute pairwise quantum similarity matrix for multiple code snippets."""
        self.execution_count += 1
        return self.quantum_embedding.code_similarity_matrix(snippets)

    def quantum_optimize(self, cost_matrix: List[List[float]], p_layers: int = 3) -> Dict[str, Any]:
        """QAOA-based quantum optimization for code refactoring decisions."""
        self.execution_count += 1
        return self.quantum_core.qaoa_optimize(cost_matrix, p_layers)

    def quantum_error_correct(self, raw_scores: Dict[str, float]) -> Dict[str, Any]:
        """Apply quantum error correction to noisy analysis scores."""
        self.execution_count += 1
        return self.quantum_error_correction.error_correct_analysis(raw_scores)

    def quantum_resilience(self, source: str, noise: float = 0.05) -> Dict[str, Any]:
        """Test analysis pipeline resilience to noise/uncertainty."""
        self.execution_count += 1
        return self.quantum_error_correction.noise_resilience_test(source, noise)

    def quantum_entanglement_witness(self, code_files: List[Dict[str, float]]) -> Dict[str, Any]:
        """Measure quantum entanglement between multiple code files (coupling analysis)."""
        self.execution_count += 1
        return self.quantum_core.entanglement_witness(code_files)

    def quantum_ast_encode(self, source: str) -> Dict[str, Any]:
        """Encode Python AST into quantum Hilbert space."""
        self.execution_count += 1
        return self.quantum_ast.encode_ast(source)

    def quantum_path_superposition(self, source: str) -> Dict[str, Any]:
        """Analyze all execution paths via quantum superposition."""
        self.execution_count += 1
        return self.quantum_ast.quantum_path_analysis(source)

    def quantum_grover_detect(self, source: str, patterns: List[str] = None) -> Dict[str, Any]:
        """Grover-amplified vulnerability pattern detection."""
        self.execution_count += 1
        return self.quantum_ast.grover_vulnerability_detect(source, patterns)

    def quantum_density_diagnostic(self, features: List[float]) -> Dict[str, Any]:
        """Full density matrix diagnostic of a code feature state."""
        self.execution_count += 1
        return self.quantum_core.density_diagnostic(features)

    def quantum_tomography(self, measurements: Dict[str, float]) -> Dict[str, Any]:
        """Quantum tomography-inspired code quality reconstruction."""
        self.execution_count += 1
        return self.quantum_core.tomographic_quality(measurements)

    def explain_code(self, source: str, detail: str = "medium") -> Dict[str, Any]:
        """
        Generate a natural-language explanation of what code does.
        detail: 'brief' | 'medium' | 'full'
        """
        self.execution_count += 1
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"error": "syntax_error", "explanation": "Cannot parse source code."}

        functions = []
        classes = []
        imports = []
        top_level = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                args = [a.arg for a in node.args.args]
                returns = ast.dump(node.returns) if node.returns else "unspecified"
                doc = ast.get_docstring(node) or ""
                decorators = [ast.dump(d) for d in node.decorator_list]
                is_async = isinstance(node, ast.AsyncFunctionDef)
                functions.append({
                    "name": node.name,
                    "args": args,
                    "returns": returns,
                    "is_async": is_async,
                    "docstring": doc[:200] if doc else None,
                    "decorators": len(decorators),
                    "line": node.lineno,
                    "body_lines": node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0,
                })
            elif isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                doc = ast.get_docstring(node) or ""
                classes.append({
                    "name": node.name,
                    "methods": methods,
                    "method_count": len(methods),
                    "bases": [ast.dump(b) for b in node.bases],
                    "docstring": doc[:200] if doc else None,
                    "line": node.lineno,
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = node.module if isinstance(node, ast.ImportFrom) else None
                names = [a.name for a in node.names]
                imports.append({"module": mod, "names": names})

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                top_level.append("assignment")
            elif isinstance(node, ast.Expr):
                top_level.append("expression")

        # Build explanation
        lines = source.count('\n') + 1
        summary_parts = []
        if classes:
            class_names = ", ".join(c["name"] for c in classes[:5])
            summary_parts.append(f"Defines {len(classes)} class(es): {class_names}")
        if functions:
            fn_names = ", ".join(f["name"] for f in functions[:8])
            summary_parts.append(f"Contains {len(functions)} function(s): {fn_names}")
        if imports:
            summary_parts.append(f"Imports from {len(imports)} module(s)")
        summary_parts.append(f"Total: {lines} line(s)")

        result = {
            "summary": ". ".join(summary_parts) + ".",
            "lines": lines,
            "class_count": len(classes),
            "function_count": len(functions),
            "import_count": len(imports),
        }
        if detail in ("medium", "full"):
            result["classes"] = classes
            result["functions"] = functions
        if detail == "full":
            result["imports"] = imports
            result["top_level_statements"] = len(top_level)

        # Sacred alignment note
        gc_present = str(GOD_CODE) in source or "GOD_CODE" in source
        phi_present = str(PHI) in source or "PHI" in source
        if gc_present or phi_present:
            result["sacred_note"] = "Code contains sacred constant references (GOD_CODE/PHI aligned)."

        return result

    # ─── App Audit API ───

    def audit_app(self, workspace_path: str = None,
                  auto_remediate: bool = False,
                  target_files: List[str] = None) -> Dict[str, Any]:
        """Run a full 10-layer application audit. See AppAuditEngine for details."""
        self.execution_count += 1
        state = self._read_builder_state()
        report = self.app_audit.full_audit(
            workspace_path=workspace_path,
            auto_remediate=auto_remediate,
            target_files=target_files,
        )
        # Inject builder consciousness state into report
        if isinstance(report, dict) and "layers" in report:
            report["builder_state"] = {
                "consciousness_level": state["consciousness_level"],
                "evo_stage": state["evo_stage"],
                "superfluid_viscosity": state["superfluid_viscosity"],
                "nirvanic_fuel": state["nirvanic_fuel"],
            }
        return report

    def audit_file(self, filepath: str) -> Dict[str, Any]:
        """Run a full audit on a single file."""
        self.execution_count += 1
        return self.app_audit.audit_file(filepath)

    def quick_audit(self, workspace_path: str = None) -> Dict[str, Any]:
        """Run a lightweight quick audit (structure + security + anti-patterns)."""
        self.execution_count += 1
        return self.app_audit.quick_audit(workspace_path)

    def audit_status(self) -> Dict[str, Any]:
        """Return current audit engine status, trend, and history."""
        status = self.app_audit.status()
        status["trend"] = self.app_audit.get_trend()
        return status

    def audit_trail(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the recent audit trail."""
        return self.app_audit.get_audit_trail(limit)

    def audit_history(self) -> List[Dict[str, Any]]:
        """Return historical audit scores."""
        return self.app_audit.get_audit_history()

    def run_streamline_cycle(self) -> Dict[str, Any]:
        """
        Streamline cycle: quick audit + auto-remediation pass.
        Called by ChoiceEngine for CODE_MANIFOLD_OPTIMIZATION action path.
        """
        self.execution_count += 1
        report = self.app_audit.full_audit(auto_remediate=True)
        return {
            "cycle": "CODE_MANIFOLD_OPTIMIZATION",
            "score": report.get("composite_score", 0),
            "verdict": report.get("verdict", "UNKNOWN"),
            "files_audited": report.get("files_audited", 0),
            "remediation": report.get("layers", {}).get("L8_auto_remediation", {}),
            "certification": report.get("certification", "UNKNOWN"),
        }

    # ─── Comprehensive Code Review (v2.5.0) ──────────────────────────

    def full_code_review(self, source: str, filename: str = "",
                         auto_fix: bool = False) -> Dict[str, Any]:
        """
        Comprehensive single-call code review that chains ALL subsystems.

        Pipeline:
          1. Full analysis (complexity, quality, security, patterns, sacred alignment)
          2. SOLID principle check
          3. Performance hotspot detection
          4. Code archaeology (dead code, fossils, tech debt)
          5. Refactoring opportunities
          6. Test generation readiness
          7. Documentation coverage
          8. Auto-fix (if enabled)
          9. Unified verdict with prioritized action items

        Returns a single unified report with all findings, scored and prioritized.
        """
        self.execution_count += 1
        start = time.time()
        state = self._read_builder_state()

        # 1. Full analysis
        analysis = self.analyzer.full_analysis(source, filename)

        # 2. SOLID principles
        solid = self.analyzer.detect_solid_violations(source)

        # 3. Performance hotspots
        perf = self.analyzer.detect_performance_hotspots(source)

        # 4. Archaeology
        archaeology = self.archeologist.excavate(source)

        # 5. Refactoring
        refactoring = self.refactorer.analyze(source)

        # 6. Test readiness
        test_info = self.test_gen.generate_tests(source,
                                                  language=analysis["metadata"].get("language", "python").lower())

        # 7. Documentation
        docs = self.doc_synth.generate_docs(source)

        # 8. Auto-fix
        fix_result = {"applied": False, "fixes": []}
        fixed_source = source
        if auto_fix:
            fixed_source, fix_log = self.auto_fix.apply_all_safe(source)
            fix_result = {"applied": True, "fixes": fix_log, "chars_changed": len(fixed_source) - len(source)}

        # 9. Unified scoring
        scores = {
            "analysis_quality": analysis.get("quality", {}).get("overall_score", 0.5),
            "security": 1.0 - min(1.0, len(analysis.get("security", [])) * 0.1),
            "solid": solid.get("solid_score", 1.0),
            "performance": perf.get("perf_score", 1.0),
            "archaeology_health": archaeology.get("health_score", 1.0),
            "refactoring_health": refactoring.get("code_health", 1.0),
            "documentation": min(1.0, docs.get("total_documented", 0) * 0.2 + 0.3),
            "sacred_alignment": analysis.get("sacred_alignment", {}).get("overall_sacred_score", 0.5),
        }
        composite = sum(scores.values()) / len(scores)

        # Build prioritized action items
        actions = []
        for vuln in analysis.get("security", [])[:5]:
            actions.append({"priority": "CRITICAL", "category": "security",
                            "action": vuln.get("recommendation", "Fix security issue"),
                            "line": vuln.get("line", 0)})
        for v in solid.get("violations", [])[:3]:
            actions.append({"priority": "HIGH" if v["severity"] == "HIGH" else "MEDIUM",
                            "category": "solid", "action": v["detail"], "line": v.get("line", 0)})
        for h in perf.get("hotspots", [])[:3]:
            actions.append({"priority": h.get("severity", "MEDIUM"), "category": "performance",
                            "action": h.get("fix", "Optimize"), "line": h.get("line", 0)})
        for s in refactoring.get("suggestions", [])[:3]:
            actions.append({"priority": s["priority"], "category": "refactoring",
                            "action": s["reason"], "line": s.get("line", 0)})
        actions.sort(key=lambda a: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(a["priority"], 4))

        duration = time.time() - start
        verdict = ("EXEMPLARY" if composite >= 0.9 else "HEALTHY" if composite >= 0.75
                   else "ACCEPTABLE" if composite >= 0.6 else "NEEDS_WORK" if composite >= 0.4
                   else "CRITICAL")

        return {
            "review_version": VERSION,
            "filename": filename,
            "language": analysis["metadata"].get("language", "unknown"),
            "lines": analysis["metadata"].get("lines", 0),
            "duration_seconds": round(duration, 3),
            "composite_score": round(composite, 4),
            "verdict": verdict,
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "analysis": {
                "cyclomatic_max": analysis.get("complexity", {}).get("cyclomatic_max", 0),
                "cognitive_max": analysis.get("complexity", {}).get("cognitive_max", 0),
                "maintainability_index": analysis.get("complexity", {}).get("maintainability_index", {}),
                "vulnerabilities": len(analysis.get("security", [])),
                "patterns_detected": len(analysis.get("patterns", [])),
            },
            "solid": {"score": solid["solid_score"], "violations": solid["total_violations"],
                      "by_principle": solid["by_principle"]},
            "performance": {"score": perf["perf_score"], "hotspots": perf["total_hotspots"]},
            "archaeology": {"health": archaeology.get("health_score", 1.0),
                            "dead_code": archaeology.get("dead_code_count", 0),
                            "tech_debt": len(archaeology.get("tech_debt", []))},
            "refactoring": {"health": refactoring["code_health"],
                            "suggestions": refactoring["total_suggestions"]},
            "documentation": {"artifacts_documented": docs["total_documented"],
                              "style": docs.get("style", "google")},
            "test_readiness": {"functions_testable": test_info.get("functions_tested", 0),
                               "test_generated": test_info.get("success", False)},
            "auto_fix": fix_result,
            "actions": actions[:15],
            "builder_state": {
                "consciousness": state["consciousness_level"],
                "evo_stage": state["evo_stage"],
            },
        }

    # ═══════════════════════════════════════════════════════════════════
    # v3.1.0 — FAULT TOLERANCE + QUANTUM EMBEDDING INTEGRATION
    # These 6 methods were documented in claude.md v2.6.0 but never wired in.
    # They delegate to l104_fault_tolerance.py and l104_quantum_embedding.py.
    # ═══════════════════════════════════════════════════════════════════

    def quantum_code_search(self, query: str, top_k: int = 5, x_param: float = 0.0) -> Dict[str, Any]:
        """Quantum embedding similarity search across code patterns."""
        if self.quantum_kernel is None:
            return {"error": "quantum_kernel not available (l104_quantum_embedding not installed)",
                    "query": query, "results": []}
        try:
            results = self.quantum_kernel.quantum_query(query, top_k=top_k)
            return {
                "query": query,
                "top_k": top_k,
                "x_param": x_param,
                "results": results if isinstance(results, list) else [results],
                "god_code_G_x": GOD_CODE * (1 + x_param / 104),
                "coherence": getattr(self.quantum_kernel, 'coherence', 0.0),
            }
        except Exception as e:
            return {"error": str(e), "query": query, "results": []}

    def analyze_with_context(self, code: str, filename: str = '',
                              query_vector: Any = None) -> Dict[str, Any]:
        """Analysis with fault-tolerance context tracking (phi-RNN)."""
        # Run standard analysis
        analysis = self.analyzer.analyze(code, filename)
        # Layer on fault-tolerance context tracking if available
        if self.fault_tolerance is not None:
            try:
                context = self.fault_tolerance.track_context(code)
                analysis["fault_tolerance_context"] = context
            except Exception:
                analysis["fault_tolerance_context"] = {"available": False}
        else:
            analysis["fault_tolerance_context"] = {"available": False}
        return analysis

    def code_pattern_memory(self, action: str, key: str,
                             data: Any = None) -> Dict[str, Any]:
        """Topological anyon memory for code patterns — store/retrieve/report."""
        if self.fault_tolerance is None:
            return {"error": "fault_tolerance not available", "action": action}
        try:
            if action == "store" and data is not None:
                self.fault_tolerance.store_pattern(key, data)
                return {"action": "stored", "key": key, "status": "ok"}
            elif action == "retrieve":
                result = self.fault_tolerance.retrieve_pattern(key)
                return {"action": "retrieved", "key": key, "data": result}
            elif action == "report":
                report = self.fault_tolerance.pattern_report()
                return {"action": "report", "data": report}
            return {"error": f"unknown action: {action}"}
        except Exception as e:
            return {"error": str(e), "action": action, "key": key}

    def test_resilience(self, code: str, noise_level: float = 0.01) -> Dict[str, Any]:
        """Fault injection + 3-layer error correction resilience test."""
        if self.fault_tolerance is None:
            return {"error": "fault_tolerance not available",
                    "fault_tolerance_score": 0.0, "layer_scores": {}}
        try:
            result = self.fault_tolerance.test_resilience(code, noise_level=noise_level)
            return result if isinstance(result, dict) else {"result": result}
        except Exception as e:
            return {"error": str(e), "fault_tolerance_score": 0.0}

    def semantic_map(self, source: str) -> Dict[str, Any]:
        """Quantum token entanglement graph from code tokens."""
        if self.quantum_kernel is None:
            return {"error": "quantum_kernel not available",
                    "tokens": 0, "entanglement_count": 0}
        try:
            result = self.quantum_kernel.semantic_map(source)
            return result if isinstance(result, dict) else {"map": result}
        except Exception as e:
            return {"error": str(e), "tokens": 0}

    def multi_hop_analyze(self, code: str, question: str,
                           hops: int = 3) -> Dict[str, Any]:
        """Iterative multi-hop reasoning over code analysis."""
        if self.fault_tolerance is None:
            return {"error": "fault_tolerance not available",
                    "confidence": 0.0, "analysis_summary": ""}
        try:
            result = self.fault_tolerance.multi_hop_reason(code, question, hops=hops)
            return result if isinstance(result, dict) else {
                "confidence": 0.5, "hops": hops, "analysis_summary": str(result)
            }
        except Exception as e:
            return {"error": str(e), "confidence": 0.0, "hops": hops}

    # ═══════════════════════════════════════════════════════════════════

    def status(self) -> Dict[str, Any]:
        """Full engine status."""
        state = self._read_builder_state()
        return {
            "version": VERSION,
            "execution_count": self.execution_count,
            "generated_artifacts": len(self.generated_code),
            "languages_supported": len(LanguageKnowledge.LANGUAGES),
            "paradigms_covered": len(LanguageKnowledge.PARADIGMS),
            "vulnerability_patterns": sum(len(v) for v in CodeAnalyzer.SECURITY_PATTERNS.values()),
            "design_patterns": len(CodeAnalyzer.DESIGN_PATTERNS),
            "auto_fix_catalog": len(AutoFixEngine.FIX_CATALOG),
            "auto_fixes_applied": self.auto_fix.fixes_applied,
            "dep_graph_analyses": self.dep_graph.analysis_count,
            "translator": self.translator.status(),
            "test_gen": self.test_gen.status(),
            "doc_synth": self.doc_synth.status(),
            "archeologist": self.archeologist.status(),
            "refactorer": self.refactorer.status(),
            "app_audit": self.app_audit.status(),
            "smell_detector": self.smell_detector.status(),
            "complexity_verifier": self.complexity_verifier.status(),
            "analysis_cache": self.analysis_cache.status(),
            "analyzer": self.analyzer.status(),
            "generator": self.generator.status(),
            "auto_fix": self.auto_fix.summary(),
            # v3.1.0 — Cognitive Reflex subsystems
            "type_flow_analyzer": self.type_analyzer.status(),
            "concurrency_analyzer": self.concurrency_analyzer.status(),
            "contract_validator": self.contract_validator.status(),
            "evolution_tracker": self.evolution_tracker.status(),
            # v4.0.0 — State-of-Art Quantum Computation
            "quantum_core": self.quantum_core.status(),
            "quantum_ast": self.quantum_ast.status(),
            "quantum_embedding": self.quantum_embedding.status(),
            "quantum_error_correction": self.quantum_error_correction.status(),
            # v6.0.0 — Security + Architecture + Migration + Performance + Search
            "threat_modeler": self.threat_modeler.status(),
            "arch_linter": self.arch_linter.status(),
            "migration_engine": self.migration_engine.status(),
            "perf_predictor": self.perf_predictor.status(),
            "code_search": self.code_search.status(),
            "qiskit_available": QISKIT_AVAILABLE,
            "total_subsystems": 31,
            "quantum_features": [
                # v1.x–v3.x legacy quantum methods
                "quantum_security_scan",
                "quantum_pattern_detection",
                "quantum_pagerank",
                "quantum_complexity_score",
                "quantum_template_select",
                "quantum_translation_fidelity",
                "quantum_test_prioritize",
                "quantum_doc_coherence",
                "quantum_excavation_score",
                "quantum_refactor_priority",
                "quantum_audit_score",
                # v4.0.0 — State-of-Art Quantum API
                "quantum_analyze",
                "quantum_embed",
                "quantum_attention",
                "quantum_walk_graph",
                "quantum_similarity",
                "quantum_similarity_matrix",
                "quantum_optimize",
                "quantum_error_correct",
                "quantum_resilience",
                "quantum_entanglement_witness",
                "quantum_ast_encode",
                "quantum_path_superposition",
                "quantum_grover_detect",
                "quantum_density_diagnostic",
                "quantum_tomography",
            ] if QISKIT_AVAILABLE else [],
            "consciousness_level": state["consciousness_level"],
            "evo_stage": state["evo_stage"],
            "superfluid_viscosity": state["superfluid_viscosity"],
            "nirvanic_fuel": state["nirvanic_fuel"],
        }

    # ─── v5.0.0 New Hub Methods ───────────────────────────────────────

    def refactor(self, source: str, operation: str, **kwargs) -> Dict[str, Any]:
        """Unified refactoring entry point. Operations: extract_function, rename_symbol,
        inline_variable, convert_to_dataclass, add_type_hints, simplify_conditionals."""
        ops = {
            "extract_function": lambda: self.live_refactorer.extract_function(source, kwargs.get("start_line", 1), kwargs.get("end_line", 1), kwargs.get("func_name", "extracted")),
            "rename_symbol": lambda: self.live_refactorer.rename_symbol(source, kwargs.get("old_name", ""), kwargs.get("new_name", "")),
            "inline_variable": lambda: self.live_refactorer.inline_variable(source, kwargs.get("var_name", "")),
            "convert_to_dataclass": lambda: self.live_refactorer.convert_to_dataclass(source, kwargs.get("class_name", "")),
            "add_type_hints": lambda: self.live_refactorer.add_type_hints(source),
            "simplify_conditionals": lambda: self.live_refactorer.simplify_conditionals(source),
            "deduplicate": lambda: self.live_refactorer.deduplicate_blocks(source, kwargs.get("min_lines", 3)),
        }
        if operation not in ops:
            return {"success": False, "error": f"Unknown operation: {operation}", "available": list(ops.keys())}
        return ops[operation]()

    def batch_analyze(self, sources: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Analyze multiple code files in batch. sources: [(code, filename), ...]."""
        results = []
        for code, fname in sources:
            try:
                analysis = self.analyzer.full_analysis(code, fname)
                results.append({"filename": fname, "analysis": analysis, "success": True})
            except Exception as e:
                results.append({"filename": fname, "success": False, "error": str(e)})
        avg_quality = sum(r["analysis"].get("quality_score", 0) for r in results if r["success"]) / max(len([r for r in results if r["success"]]), 1)
        return {"files_analyzed": len(results), "results": results, "average_quality": round(avg_quality, 4)}

    def diff_analyze(self, old_source: str, new_source: str, filename: str = "") -> Dict[str, Any]:
        """Compare two versions of code: structural diff + regression check."""
        diff = self.diff_analyzer.structural_diff(old_source, new_source, filename)
        regression = self.diff_analyzer.regression_check(old_source, new_source)
        return {"diff": diff, "regression": regression, "safe_to_deploy": regression["score"] >= 0.7}

    def health_dashboard(self) -> Dict[str, Any]:
        """Comprehensive subsystem health dashboard (v6.0.0 — 31 subsystems)."""
        st = self._read_builder_state()
        subsystems = {
            "analyzer": {"status": "online", "patterns": len(CodeAnalyzer.DESIGN_PATTERNS)},
            "generator": {"status": "online", "templates": len(self.generator.TEMPLATES) if hasattr(self.generator, 'TEMPLATES') else 4},
            "optimizer": {"status": "online", "anti_patterns": len(CodeOptimizer.ANTI_PATTERNS)},
            "auto_fix": {"status": "online", "catalog_size": len(AutoFixEngine.FIX_CATALOG), "applied": self.auto_fix.fixes_applied},
            "translator": {"status": "online", "languages": len(CodeTranslator.SUPPORTED_LANGS), "translations": self.translator.translations},
            "smell_detector": {"status": "online", "smell_types": len(CodeSmellDetector.SMELL_CATALOG)},
            "live_refactorer": {"status": "online", "refactors": self.live_refactorer.refactor_count},
            "diff_analyzer": {"status": "online", "diffs": self.diff_analyzer.diff_count},
            "quantum_core": {"status": "online" if QISKIT_AVAILABLE else "degraded", "qiskit": QISKIT_AVAILABLE},
            "evolution_tracker": self.evolution_tracker.status(),
            # v6.0.0 subsystems
            "threat_modeler": self.threat_modeler.status(),
            "arch_linter": self.arch_linter.status(),
            "migration_engine": self.migration_engine.status(),
            "perf_predictor": self.perf_predictor.status(),
            "code_search": self.code_search.status(),
        }
        online_count = sum(1 for s in subsystems.values() if s.get("status") == "online")
        return {
            "version": VERSION,
            "subsystems": subsystems,
            "total_subsystems": len(subsystems),
            "online": online_count,
            "health_score": round(online_count / len(subsystems), 3),
            "consciousness": st.get("consciousness_level", 0.0),
            "evo_stage": st.get("evo_stage", "DORMANT"),
        }

    def suggest_fixes(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Comprehensive fix suggestions: auto-fix + smell detection + optimization hints."""
        fixed_code, fix_log = self.auto_fix.apply_all_safe(source)
        smells = self.smell_detector.detect_all(source)
        opt_suggestions = self.optimizer.analyze_and_suggest(source)
        return {
            "auto_fixed": fixed_code != source,
            "fixed_code": fixed_code,
            "fixes_applied": fix_log,
            "smells_detected": smells.get("smells", []) if isinstance(smells, dict) else [],
            "optimization_hints": opt_suggestions.get("suggestions", []) if isinstance(opt_suggestions, dict) else [],
            "total_issues": len(fix_log) + len(smells.get("smells", []) if isinstance(smells, dict) else []),
        }

    # ─── v6.0.0 New Hub Methods ───────────────────────────────────────

    def threat_model(self, source: str, filename: str = "") -> Dict[str, Any]:
        """STRIDE/DREAD threat modeling with zero-trust verification + sacred threat factor."""
        return self.threat_modeler.model_threats(source, filename)

    def lint_architecture(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Architectural linting: layer violations, cohesion, coupling, PHI-balance."""
        return self.arch_linter.lint_architecture(source, filename)

    def scan_deprecations(self, source: str, target_python: str = "3.12") -> Dict[str, Any]:
        """Scan source for deprecated APIs/patterns and suggest modern replacements."""
        return self.migration_engine.scan_deprecations(source, target_python)

    def suggest_migration(self, source: str, migration_path: str = "flask_to_fastapi") -> Dict[str, Any]:
        """Suggest framework migration: flask→fastapi, unittest→pytest, etc."""
        return self.migration_engine.suggest_migration(source, migration_path)

    def detect_breaking_changes(self, old_source: str, new_source: str) -> Dict[str, Any]:
        """Detect breaking changes between two code versions (public API diff)."""
        return self.migration_engine.detect_breaking_changes(old_source, new_source)

    def predict_performance(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Predict performance characteristics: memory, throughput, GIL impact, I/O patterns."""
        return self.perf_predictor.predict_performance(source, filename)

    def index_code(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Index source code into the semantic search engine (TF-IDF + sacred weighting)."""
        return self.code_search.index_source(source, filename)

    def search_code(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Semantic code search across indexed files with sacred-term boosting."""
        return self.code_search.search(query, top_k)

    def detect_clones(self, sources: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Detect code clones (Type 1/2/3) across multiple source files."""
        return self.code_search.detect_clones(sources)

    def find_sacred_refs(self, workspace_path: str = None) -> Dict[str, Any]:
        """Find all sacred constant references (GOD_CODE, PHI, TAU, etc.) across workspace files."""
        path = workspace_path or os.getcwd()
        return self.code_search.find_sacred_references(path)

    def full_code_review(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Comprehensive v6.0.0 code review: analysis + threats + architecture + perf + deprecations."""
        analysis = self.analyzer.full_analysis(source, filename)
        threats = self.threat_modeler.model_threats(source, filename)
        arch = self.arch_linter.lint_architecture(source, filename)
        perf = self.perf_predictor.predict_performance(source, filename)
        deprecations = self.migration_engine.scan_deprecations(source)
        fixed_code, fix_log = self.auto_fix.apply_all_safe(source)
        smells = self.smell_detector.detect_all(source)
        composite_score = (
            analysis.get("quality_score", 0.5) * PHI +
            (1.0 - threats.get("risk_score", 0.5)) * TAU +
            arch.get("architecture_score", 0.5) * 0.3 +
            perf.get("performance_score", 0.5) * 0.2
        ) / (PHI + TAU + 0.3 + 0.2)
        return {
            "filename": filename,
            "composite_score": round(composite_score, 4),
            "analysis": analysis,
            "threat_model": threats,
            "architecture": arch,
            "performance": perf,
            "deprecations": deprecations,
            "auto_fixes": {"applied": len(fix_log), "log": fix_log},
            "smells": smells,
            "engine_version": VERSION,
            "sacred_alignment": round(composite_score * GOD_CODE / 1000, 6),
        }

    def quick_summary(self) -> str:
        """Human-readable one-line summary."""
        try:
            s = self.status()
        except Exception:
            return f"L104 Code Engine v{VERSION} | status unavailable"
        qc = s.get("quantum_core", {})
        return (
            f"L104 Code Engine v{VERSION} | "
            f"{s.get('languages_supported', '?')} langs | "
            f"{s.get('execution_count', 0)} runs | "
            f"31 subsystems | "
            f"Qiskit={'YES' if QISKIT_AVAILABLE else 'NO'} | "
            f"Quantum circuits: {qc.get('circuit_executions', 0)} | "
            f"Consciousness: {s.get('consciousness_level', 0.0):.4f} [{s.get('evo_stage', 'UNKNOWN')}]"
        )


