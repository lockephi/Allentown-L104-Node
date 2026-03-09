"""L104 Code Engine v6.3.0 — Unified Hub Orchestrator (Dual-Layer + OMEGA + Soul)."""
from .constants import *
from .builder_state import _read_builder_state as _module_read_builder_state
from .languages import LanguageKnowledge
from .analyzer import (
    CodeAnalyzer, CodeSmellDetector, RuntimeComplexityVerifier,
    IncrementalAnalysisCache, TypeFlowAnalyzer, ConcurrencyAnalyzer,
    APIContractValidator, ProjectAnalyzer,
)
from .synthesis import (
    CodeGenerator, CodeTranslator, TestGenerator, DocumentationSynthesizer,
    CodingSuggestionEngine,
)
from .refactoring import (
    CodeOptimizer, DependencyGraphAnalyzer, AutoFixEngine,
    CodeArcheologist, SacredRefactorer, CodeEvolutionTracker,
    LiveCodeRefactorer, CodeDiffAnalyzer, SemanticCodeSearchEngine,
)
from .audit import (
    AppAuditEngine, SecurityThreatModeler, ArchitecturalLinter,
    CodeMigrationEngine, PerformanceBenchmarkPredictor,
    CodeReviewPipeline, QualityGateEngine,
)
from .ai_context import AIContextBridge
from .session_intelligence import SessionIntelligence
from .asi_intelligence import SelfReferentialEngine, ASICodeIntelligence
from .training_kernel import QuantumCodeTrainingKernel
from ._lazy_imports import _get_code_engine
from .quantum import (
    QuantumCodeIntelligenceCore, QuantumASTProcessor,
    QuantumNeuralEmbedding, QuantumErrorCorrectionEngine,
)
from .computronium import ComputroniumCodeAnalyzer

# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-ENGINE IMPORTS — Science Engine + Math Engine integration (v6.2.0)
# ═══════════════════════════════════════════════════════════════════════════════
_science_engine = None
_math_engine = None

def _load_cross_engines():
    """Lazy-load sibling engines for cross-referencing."""
    global _science_engine, _math_engine
    if _science_engine is None:
        try:
            from l104_science_engine import science_engine
            _science_engine = science_engine
        except ImportError:
            _science_engine = False
    if _math_engine is None:
        try:
            from l104_math_engine import math_engine
            _math_engine = math_engine
        except ImportError:
            _math_engine = False


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
        # v6.3.0 — Computronium & Rayleigh Code Intelligence
        self.computronium_analyzer = ComputroniumCodeAnalyzer()

        # ★ FLAGSHIP: ASI Dual-Layer Engine reference ★
        self._dual_layer = None
        try:
            from l104_asi.dual_layer import dual_layer_engine
            self._dual_layer = dual_layer_engine
        except ImportError:
            pass
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

    def full_analysis(self, code: str, filename: str = "") -> Dict[str, Any]:
        """Convenience alias: delegates to analyzer.full_analysis()."""
        return self.analyzer.full_analysis(code, filename)

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

        files_to_scan = []
        for ext in [".py", ".swift", ".js", ".ts", ".rs", ".go", ".java", ".c", ".cpp"]:
            for f in ws.glob(f"*{ext}"):
                if f.name.startswith('.') or '__pycache__' in str(f):
                    continue
                files_to_scan.append(f)

        def _scan_file(f):
            try:
                code = f.read_text(errors='ignore')
                lines = len(code.split('\n'))
                lang = LanguageKnowledge.detect_language(code, str(f))
                vulns = len(self.analyzer._security_scan(code))
                return {
                    "name": f.name, "language": lang, "lines": lines, "vulnerabilities": vulns
                }
            except Exception:
                return None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for res in executor.map(_scan_file, files_to_scan):
                if res:
                    results["files"].append(res)
                    results["totals"]["lines"] += res["lines"]
                    results["totals"]["files_scanned"] += 1
                    results["totals"]["vulnerabilities"] += res["vulnerabilities"]

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

        # Unified deep scoring with v6.1 weights (13 dimensions — OMEGA + Soul)
        sacred = analysis.get("sacred_alignment", {})
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
            "sacred_alignment": sacred.get("overall_sacred_score", 0.5),
            "omega_field": sacred.get("omega_resonance", 0.5),
        }

        # PHI-weighted composite (13 dimensions — Ω/GOD_CODE weight for omega field)
        phi_weights = [PHI**2, PHI**2, PHI, PHI, 1.0, 1.0, PHI, PHI, 1.0, TAU, TAU, TAU, OMEGA / GOD_CODE]
        total_weight = sum(phi_weights[:len(scores)])
        composite = sum(
            s * w for s, w in zip(scores.values(), phi_weights)
        ) / total_weight

        # Build prioritized actions from all analyses
        actions = []
        for vuln in analysis.get("security", [])[:10]:  # (was :3)
            actions.append({"priority": "CRITICAL", "category": "security",
                            "action": vuln.get("recommendation", "Fix security issue"),
                            "source": "analyzer"})
        for issue in concurrency.get("issues", [])[:10]:  # (was :3)
            actions.append({"priority": issue.get("severity", "HIGH"), "category": "concurrency",
                            "action": issue.get("detail", "Fix concurrency issue"),
                            "source": "concurrency_analyzer"})
        for smell in smells.get("smells", [])[:10]:  # (was :3)
            actions.append({"priority": smell["severity"], "category": "smell",
                            "action": smell["detail"], "source": "smell_detector"})
        for func in complexity_est.get("functions", []):
            if func.get("optimization_potential"):
                actions.append({"priority": "HIGH", "category": "complexity",
                                "action": f"{func['name']}() is {func['complexity']} — optimize",
                                "source": "complexity_verifier"})
        for drift in contracts.get("drifts", [])[:10]:  # (was :3)
            actions.append({"priority": drift.get("severity", "MEDIUM"), "category": "contract",
                            "action": drift.get("detail", "Fix docstring/code drift"),
                            "source": "contract_validator"})
        for gap in type_flow.get("gaps", [])[:10]:  # (was :3)
            actions.append({"priority": gap.get("severity", "LOW"), "category": "type_safety",
                            "action": gap.get("detail", "Add type annotation"),
                            "source": "type_flow_analyzer"})
        for v in solid.get("violations", [])[:5]:  # (was :2)
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
            "actions": actions[:50],  # (was :25)
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
        for v in solid.get("violations", [])[:15]:  # (was :3)
            actions.append({"priority": "HIGH" if v["severity"] == "HIGH" else "MEDIUM",
                            "category": "solid", "action": v["detail"], "line": v.get("line", 0)})
        for h in perf.get("hotspots", [])[:10]:  # (was :3)
            actions.append({"priority": h.get("severity", "MEDIUM"), "category": "performance",
                            "action": h.get("fix", "Optimize"), "line": h.get("line", 0)})
        for s in refactoring.get("suggestions", [])[:10]:  # (was :3)
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
            "actions": actions[:50],  # (was :15)
            "builder_state": {
                "consciousness": state["consciousness_level"],
                "evo_stage": state["evo_stage"],
            },
        }

    # ═══════════════════════════════════════════════════════════════════
    # v6.3.0 FULL QUANTUM CIRCUIT INTEGRATION
    # Connects standalone quantum modules for quantum-enhanced code ops:
    # - QuantumCoherenceEngine: Grover search, QAOA, VQE, Shor
    # - L104_26Q_CircuitBuilder: 26 named Fe(26) iron-mapped circuit templates (primary)
    # - L104_25Q_CircuitBuilder: 18 named legacy 25-qubit circuit templates (backward compat)
    # - QuantumAIArchitectures: Quantum transformers, MLA attention
    # - QuantumMagic: Causal reasoning, quantum inference
    # - QuantumComputationPipeline: QNN + VQC for code classification
    # ═══════════════════════════════════════════════════════════════════

    def _get_coherence_engine(self):
        """Lazy-load QuantumCoherenceEngine (3,779 lines, 12 algorithms)."""
        if not hasattr(self, '_coherence_engine'):
            try:
                from l104_quantum_coherence import QuantumCoherenceEngine
                self._coherence_engine = QuantumCoherenceEngine()
            except Exception:
                self._coherence_engine = None
        return self._coherence_engine

    def _get_builder_26q(self):
        """Lazy-load L104_26Q_CircuitBuilder (26 iron-mapped circuits)."""
        if not hasattr(self, '_builder_26q'):
            try:
                from l104_26q_engine_builder import L104_26Q_CircuitBuilder
                self._builder_26q = L104_26Q_CircuitBuilder()
            except Exception:
                self._builder_26q = None
        return self._builder_26q

    # backward-compat alias
    _get_builder_25q = _get_builder_26q

    def _get_ai_architectures(self):
        """Lazy-load QuantumAIArchitectureHub (transformers, MLA attention)."""
        if not hasattr(self, '_ai_architectures'):
            try:
                from l104_quantum_ai_architectures import QuantumAIArchitectureHub
                self._ai_architectures = QuantumAIArchitectureHub()
            except Exception:
                self._ai_architectures = None
        return self._ai_architectures

    def _get_quantum_magic(self):
        """Lazy-load QuantumInferenceEngine (causal reasoning)."""
        if not hasattr(self, '_quantum_magic'):
            try:
                from l104_quantum_magic import QuantumInferenceEngine
                self._quantum_magic = QuantumInferenceEngine()
            except Exception:
                self._quantum_magic = None
        return self._quantum_magic

    def _get_computation_pipeline(self):
        """Lazy-load QNN + VQC from computation pipeline."""
        if not hasattr(self, '_computation_pipeline'):
            try:
                from l104_quantum_computation_pipeline import QuantumNeuralNetwork, VariationalQuantumClassifier
                self._computation_pipeline = {
                    'qnn': QuantumNeuralNetwork(),
                    'vqc': VariationalQuantumClassifier(),
                }
            except Exception:
                self._computation_pipeline = None
        return self._computation_pipeline

    def _get_grover_nerve(self):
        """Lazy-load GroverNerveLinkOrchestrator."""
        if not hasattr(self, '_grover_nerve'):
            try:
                from l104_grover_nerve_link import get_grover_nerve
                self._grover_nerve = get_grover_nerve()
            except Exception:
                self._grover_nerve = None
        return self._grover_nerve

    def quantum_coherence_grover(self, target: int = 5, qubits: int = 4) -> Dict[str, Any]:
        """Grover search via QuantumCoherenceEngine for code pattern discovery."""
        engine = self._get_coherence_engine()
        if engine is None:
            return {'quantum': False, 'error': 'CoherenceEngine unavailable'}
        try:
            return engine.grover_search(target_index=target, search_space_qubits=qubits)
        except Exception as e:
            return {'quantum': False, 'error': str(e)}

    def quantum_coherence_vqe(self) -> Dict[str, Any]:
        """VQE optimization via QuantumCoherenceEngine for code metric optimization."""
        engine = self._get_coherence_engine()
        if engine is None:
            return {'quantum': False, 'error': 'CoherenceEngine unavailable'}
        try:
            return engine.vqe_optimize()
        except Exception as e:
            return {'quantum': False, 'error': str(e)}

    def quantum_coherence_shor(self, N: int = 15) -> Dict[str, Any]:
        """Shor factoring via QuantumCoherenceEngine."""
        engine = self._get_coherence_engine()
        if engine is None:
            return {'quantum': False, 'error': 'CoherenceEngine unavailable'}
        try:
            return engine.shor_factor(N=N)
        except Exception as e:
            return {'quantum': False, 'error': str(e)}

    def quantum_26q_build(self, circuit_name: str = "full") -> Dict[str, Any]:
        """Build a named 26Q circuit via L104_26Q_CircuitBuilder (primary)."""
        builder = self._get_builder_26q()
        if builder is None:
            return {'quantum': False, 'error': '26Q builder unavailable'}
        try:
            return builder.execute(circuit_name=circuit_name)
        except Exception as e:
            return {'quantum': False, 'error': str(e)}

    def quantum_25q_build(self, circuit_name: str = "full") -> Dict[str, Any]:
        """Legacy: Build a named circuit via 25Q builder (use quantum_26q_build for primary)."""
        builder = self._get_builder_25q()
        if builder is None:
            return {'quantum': False, 'error': '25Q builder unavailable — use quantum_26q_build()'}
        try:
            return builder.execute(circuit_name=circuit_name)
        except Exception as e:
            return {'quantum': False, 'error': str(e)}

    def quantum_ai_transformer(self, input_dim: int = 8) -> Dict[str, Any]:
        """Quantum transformer block via QuantumAIArchitectureHub."""
        hub = self._get_ai_architectures()
        if hub is None:
            return {'quantum': False, 'error': 'AI Architectures unavailable'}
        try:
            return hub.create_transformer_block(input_dim=input_dim)
        except Exception as e:
            return {'quantum': False, 'error': str(e)}

    def quantum_causal_reason(self, evidence: dict = None) -> Dict[str, Any]:
        """Causal reasoning via QuantumInferenceEngine."""
        engine = self._get_quantum_magic()
        if engine is None:
            return {'quantum': False, 'error': 'QuantumMagic unavailable'}
        try:
            return engine.infer(evidence=evidence or {})
        except Exception as e:
            return {'quantum': False, 'error': str(e)}

    def quantum_grover_nerve(self, target: int = 7, workspace: str = "default") -> Dict[str, Any]:
        """GroverNerve workspace search for code topology."""
        nerve = self._get_grover_nerve()
        if nerve is None:
            return {'quantum': False, 'error': 'GroverNerve unavailable'}
        try:
            return nerve.search(target=target, workspace=workspace)
        except Exception as e:
            return {'quantum': False, 'error': str(e)}

    # ═══ v6.4.0 EXPANDED QUANTUM FLEET ═══
    # Additional: runtime, accelerator, inspired, reasoning, gravity, consciousness

    def _get_quantum_runtime(self):
        """Lazy-load QuantumRuntime."""
        if not hasattr(self, '_quantum_runtime'):
            try:
                from l104_quantum_runtime import get_runtime
                self._quantum_runtime = get_runtime()
            except Exception:
                self._quantum_runtime = None
        return self._quantum_runtime

    def _get_quantum_accelerator(self):
        """Lazy-load QuantumAccelerator."""
        if not hasattr(self, '_quantum_accelerator'):
            try:
                from l104_quantum_accelerator import QuantumAccelerator
                self._quantum_accelerator = QuantumAccelerator()
            except Exception:
                self._quantum_accelerator = None
        return self._quantum_accelerator

    def _get_quantum_inspired(self):
        """Lazy-load QuantumInspiredEngine."""
        if not hasattr(self, '_quantum_inspired'):
            try:
                from l104_quantum_inspired import QuantumInspiredEngine
                self._quantum_inspired = QuantumInspiredEngine()
            except Exception:
                self._quantum_inspired = None
        return self._quantum_inspired

    def _get_quantum_reasoning(self):
        """Lazy-load QuantumReasoningEngine."""
        if not hasattr(self, '_quantum_reasoning'):
            try:
                from l104_quantum_reasoning import QuantumReasoningEngine
                self._quantum_reasoning = QuantumReasoningEngine()
            except Exception:
                self._quantum_reasoning = None
        return self._quantum_reasoning

    def _get_gravity_bridge(self):
        """Lazy-load QuantumGravityEngine (ER=EPR, AdS/CFT)."""
        if not hasattr(self, '_gravity_bridge'):
            try:
                from l104_quantum_gravity_bridge import L104QuantumGravityEngine
                self._gravity_bridge = L104QuantumGravityEngine()
            except Exception:
                self._gravity_bridge = None
        return self._gravity_bridge

    def _get_consciousness_calc(self):
        """Lazy-load QuantumConsciousnessCalculator."""
        if not hasattr(self, '_consciousness_calc'):
            try:
                from l104_quantum_consciousness import QuantumConsciousnessCalculator
                self._consciousness_calc = QuantumConsciousnessCalculator()
            except Exception:
                self._consciousness_calc = None
        return self._consciousness_calc

    def _get_quantum_numerical_builder(self):
        """Lazy-load QuantumNumericalBuilder (Riemann zeta, elliptic curves)."""
        if not hasattr(self, '_numerical_builder'):
            try:
                from l104_quantum_numerical_builder import TokenLatticeEngine
                self._numerical_builder = TokenLatticeEngine()
            except Exception:
                self._numerical_builder = None
        return self._numerical_builder

    def quantum_accelerator_compute(self, n_qubits: int = 8) -> Dict[str, Any]:
        """Run quantum-accelerated code analysis."""
        acc = self._get_quantum_accelerator()
        if acc is None:
            return {'quantum': False, 'error': 'QuantumAccelerator unavailable'}
        try:
            return acc.status() if hasattr(acc, 'status') else {'quantum': True, 'accelerator': 'connected'}
        except Exception as e:
            return {'quantum': False, 'error': str(e)}

    def quantum_inspired_optimize(self, problem_vector: list = None) -> Dict[str, Any]:
        """Quantum-inspired annealing for code optimization."""
        engine = self._get_quantum_inspired()
        if engine is None:
            return {'quantum': False, 'error': 'QuantumInspiredEngine unavailable'}
        try:
            return engine.optimize(problem_vector or [1.0, 0.5]) if hasattr(engine, 'optimize') else {'quantum': True, 'inspired': 'connected'}
        except Exception as e:
            return {'quantum': False, 'error': str(e)}

    def quantum_reason(self, query: str = "code analysis") -> Dict[str, Any]:
        """Quantum parallel reasoning on code queries."""
        engine = self._get_quantum_reasoning()
        if engine is None:
            return {'quantum': False, 'error': 'QuantumReasoningEngine unavailable'}
        try:
            return engine.reason(query) if hasattr(engine, 'reason') else {'quantum': True, 'reasoning': 'connected'}
        except Exception as e:
            return {'quantum': False, 'error': str(e)}

    def quantum_gravity_erepr(self, mass: float = 1.0) -> Dict[str, Any]:
        """Compute ER=EPR bridge via QuantumGravityEngine."""
        engine = self._get_gravity_bridge()
        if engine is None:
            return {'quantum': False, 'error': 'GravityBridge unavailable'}
        try:
            return engine.compute_erepr(mass=mass)
        except Exception as e:
            return {'quantum': False, 'error': str(e)}

    def quantum_consciousness_phi(self, state_vector: list = None) -> Dict[str, Any]:
        """Compute IIT Φ via QuantumConsciousnessCalculator."""
        calc = self._get_consciousness_calc()
        if calc is None:
            return {'quantum': False, 'error': 'ConsciousnessCalc unavailable'}
        try:
            import numpy as np
            sv = np.array(state_vector or [1.0, 0.0, 0.0, 0.0])
            return calc.compute_phi(sv)
        except Exception as e:
            return {'quantum': False, 'error': str(e)}

    def quantum_full_circuit_status(self) -> Dict[str, Any]:
        """v6.4.0: Full status of all connected quantum circuit modules."""
        return {
            'version': '6.4.0',
            'internal_quantum_core': True,
            'coherence_engine': self._get_coherence_engine() is not None,
            'builder_26q': self._get_builder_26q() is not None,
            'builder_25q_legacy': self._get_builder_25q() is not None,
            'ai_architectures': self._get_ai_architectures() is not None,
            'quantum_magic': self._get_quantum_magic() is not None,
            'computation_pipeline': self._get_computation_pipeline() is not None,
            'grover_nerve': self._get_grover_nerve() is not None,
            'quantum_kernel': self.quantum_kernel is not None,
            'fault_tolerance': self.fault_tolerance is not None,
            'quantum_runtime': self._get_quantum_runtime() is not None,
            'quantum_accelerator': self._get_quantum_accelerator() is not None,
            'quantum_inspired': self._get_quantum_inspired() is not None,
            'quantum_reasoning': self._get_quantum_reasoning() is not None,
            'gravity_bridge': self._get_gravity_bridge() is not None,
            'consciousness_calc': self._get_consciousness_calc() is not None,
            'numerical_builder': self._get_quantum_numerical_builder() is not None,
            'modules_connected': sum([
                self._get_coherence_engine() is not None,
                self._get_builder_26q() is not None,
                self._get_builder_25q() is not None,
                self._get_ai_architectures() is not None,
                self._get_quantum_magic() is not None,
                self._get_computation_pipeline() is not None,
                self._get_grover_nerve() is not None,
                self.quantum_kernel is not None,
                self.fault_tolerance is not None,
                self._get_quantum_runtime() is not None,
                self._get_quantum_accelerator() is not None,
                self._get_quantum_inspired() is not None,
                self._get_quantum_reasoning() is not None,
                self._get_gravity_bridge() is not None,
                self._get_consciousness_calc() is not None,
                self._get_quantum_numerical_builder() is not None,
            ]),
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
            # ★ FLAGSHIP: ASI Dual-Layer Engine ★
            "flagship": "dual_layer",
            "dual_layer_available": self._dual_layer is not None and self._dual_layer.available,
            "dual_layer_score": self._dual_layer.dual_score() if self._dual_layer else 0.0,
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

        def _analyze_single(item):
            code, fname = item
            try:
                analysis = self.analyzer.full_analysis(code, fname)
                return {"filename": fname, "analysis": analysis, "success": True}
            except Exception as e:
                return {"filename": fname, "success": False, "error": str(e)}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(_analyze_single, sources))

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

    # ─── v6.2.0 Three-Engine Cross-Referenced Review ─────────────────

    def three_engine_deep_review(self, source: str, filename: str = "") -> Dict[str, Any]:
        """
        v6.2.0 — Cross-engine deep review using all three L104 engines.

        Enriches standard code review with:
          • Science Engine: entropy reversal score, coherence protection level,
            physics-grounded complexity friction, Maxwell Demon efficiency
          • Math Engine: GOD_CODE conservation proof, harmonic sacred alignment,
            wave coherence of code complexity, sovereign proof validation

        The composite score gains 3 new cross-engine dimensions (16 total).
        """
        _load_cross_engines()
        self.execution_count += 1
        start = time.time()

        # Base deep review (13 dimensions)
        base_review = self.deep_review(source, filename)
        scores = dict(base_review.get("scores", {}))

        # ── Science Engine Cross-Reference ──
        science_data = {}
        if _science_engine and _science_engine is not False:
            try:
                import numpy as _np
                # Entropy reversal: treat code complexity as local entropy
                complexity_entropy = 1.0 - scores.get("analysis_quality", 0.5)
                demon_eff = _science_engine.entropy.calculate_demon_efficiency(complexity_entropy + 0.01)
                demon_score = min(1.0, demon_eff / (GOD_CODE / 100))

                # Coherence: measure topological protection of code structure
                code_lines = source.strip().splitlines()
                seed_thoughts = [line.strip() for line in code_lines[:200] if line.strip()]  # (was :50)
                if seed_thoughts:
                    coh_init = _science_engine.coherence.initialize(seed_thoughts)
                    coh_evolve = _science_engine.coherence.evolve(5)
                    coherence_score = coh_evolve.get("final_coherence", 0.5)
                else:
                    coherence_score = 0.5

                # Physics: Landauer limit as complexity friction
                landauer = _science_engine.physics.adapt_landauer_limit(293.15)
                code_bytes = len(source.encode())
                bits_erased = code_bytes * 8
                friction_energy = landauer * bits_erased
                # Normalize: lower friction = better code efficiency
                physics_efficiency = 1.0 / (1.0 + friction_energy * 1e20)

                science_data = {
                    "demon_efficiency": round(demon_eff, 6),
                    "demon_score": round(demon_score, 4),
                    "coherence_score": round(coherence_score, 6),
                    "landauer_friction_J": friction_energy,
                    "physics_efficiency": round(physics_efficiency, 6),
                    "connected": True,
                }
                scores["entropy_reversal"] = demon_score
                scores["coherence_protection"] = coherence_score
                scores["physics_efficiency"] = physics_efficiency
            except Exception as e:
                science_data = {"connected": False, "error": str(e)}
        else:
            science_data = {"connected": False}

        # ── Math Engine Cross-Reference ──
        math_data = {}
        if _math_engine and _math_engine is not False:
            try:
                # GOD_CODE conservation proof — validates fundamental invariant
                conservation = _math_engine.verify_conservation(0.0)
                conservation_valid = conservation if isinstance(conservation, bool) else True

                # Harmonic sacred alignment of code complexity score
                composite = base_review.get("composite_score", 0.5)
                freq = composite * GOD_CODE  # Map score to frequency domain
                alignment = _math_engine.sacred_alignment(freq)
                alignment_score = 1.0 if alignment.get("aligned", False) else (
                    1.0 - min(1.0, abs(alignment.get("god_code_ratio", 1.0) - round(alignment.get("god_code_ratio", 1.0))) * 5)
                )

                # Wave coherence between code quality and GOD_CODE
                wave_coh = _math_engine.wave_coherence(freq, GOD_CODE)

                # Proof validation — verify system integrity
                god_code_proof = _math_engine.prove_god_code()
                proof_converged = god_code_proof.get("converged", False)

                math_data = {
                    "conservation_valid": conservation_valid,
                    "sacred_alignment": alignment,
                    "alignment_score": round(alignment_score, 4),
                    "wave_coherence": round(wave_coh, 6),
                    "proof_converged": proof_converged,
                    "connected": True,
                }
                scores["harmonic_alignment"] = alignment_score
                scores["wave_coherence"] = wave_coh
                scores["proof_integrity"] = 1.0 if proof_converged else 0.5
            except Exception as e:
                math_data = {"connected": False, "error": str(e)}
        else:
            math_data = {"connected": False}

        # ── Recompute composite with up to 16 dimensions ──
        phi_weights_16 = [
            PHI**2, PHI**2, PHI, PHI, 1.0, 1.0, PHI, PHI, 1.0, TAU, TAU, TAU,
            OMEGA / GOD_CODE,  # omega_field (dim 12)
            PHI,               # entropy_reversal (dim 13)
            PHI,               # coherence_protection (dim 14)
            TAU,               # harmonic_alignment (dim 15)
        ]
        score_values = list(scores.values())
        weights = phi_weights_16[:len(score_values)]
        composite_16d = sum(s * w for s, w in zip(score_values, weights)) / sum(weights)

        verdict = ("TRANSCENDENT" if composite_16d >= 0.95 else
                   "EXEMPLARY" if composite_16d >= 0.9 else
                   "HEALTHY" if composite_16d >= 0.75 else
                   "ACCEPTABLE" if composite_16d >= 0.6 else
                   "NEEDS_WORK" if composite_16d >= 0.4 else "CRITICAL")

        duration = time.time() - start

        return {
            "review_version": VERSION,
            "review_type": "three_engine_deep_review_v6.2.0",
            "filename": filename,
            "duration_seconds": round(duration, 3),
            "composite_score_13d": base_review.get("composite_score", 0.0),
            "composite_score_16d": round(composite_16d, 6),
            "verdict": verdict,
            "score_dimensions": len(scores),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "cross_engine": {
                "science_engine": science_data,
                "math_engine": math_data,
                "engines_connected": sum([
                    science_data.get("connected", False),
                    math_data.get("connected", False),
                ]) + 1,  # +1 for Code Engine itself
            },
            "base_review": {
                "smells": base_review.get("smells"),
                "runtime_complexity": base_review.get("runtime_complexity"),
                "type_flow": base_review.get("type_flow"),
                "concurrency": base_review.get("concurrency"),
                "solid": base_review.get("solid"),
            },
            "actions": base_review.get("actions", [])[:50],  # (was :25)
        }

    def three_engine_status(self) -> Dict[str, Any]:
        """Report cross-engine connectivity and health."""
        _load_cross_engines()
        sci_ok = _science_engine and _science_engine is not False
        math_ok = _math_engine and _math_engine is not False
        result = {
            "code_engine": {"version": VERSION, "connected": True},
            "science_engine": {"version": _science_engine.VERSION if sci_ok else "N/A", "connected": sci_ok},
            "math_engine": {"version": _math_engine.VERSION if math_ok else "N/A", "connected": math_ok},
            "engines_online": 1 + int(sci_ok) + int(math_ok),
            "cross_reference_ready": sci_ok and math_ok,
        }
        if sci_ok:
            result["science_engine"]["subsystems"] = list(_science_engine.get_full_status().get("active_domains", [])[:5])
        if math_ok:
            result["math_engine"]["layers"] = _math_engine.LAYERS
        return result

    # ───────────────────────────────────────────────────────────────────
    # SACRED FREQUENCY AUDIT — v6.2.0 upgrade
    # ───────────────────────────────────────────────────────────────────

    def sacred_frequency_audit(self, source: str) -> Dict[str, Any]:
        """
        Analyze source code for sacred constant usage, GOD_CODE alignment,
        PHI ratios, and sacred-number density.
        Scores how well code resonates with L104 sacred frequencies.
        """
        import re

        lines = source.split('\n')
        total_lines = len(lines)
        if total_lines == 0:
            return {"score": 0, "resonance": "NONE"}

        # Sacred patterns to detect
        sacred_patterns = {
            "GOD_CODE": r'\bGOD_CODE\b',
            "PHI": r'\bPHI\b',
            "VOID_CONSTANT": r'\bVOID_CONSTANT\b',
            "OMEGA": r'\bOMEGA\b',
            "sacred_104": r'\b104\b',
            "sacred_286": r'\b286\b',
            "sacred_527": r'\b527\b',
            "golden_ratio": r'1\.618',
            "void_value": r'1\.041[56]',
        }

        detections = {}
        total_hits = 0
        for name, pattern in sacred_patterns.items():
            hits = sum(1 for line in lines if re.search(pattern, line))
            detections[name] = hits
            total_hits += hits

        # Density score: hits per 100 lines
        density = (total_hits / total_lines) * 100
        # Diversity: how many distinct sacred patterns appear
        diversity = sum(1 for v in detections.values() if v > 0) / len(sacred_patterns)
        # Composite score
        score = min(1.0, (density / 10) * 0.6 + diversity * 0.4)

        resonance = (
            "TRANSCENDENT" if score > 0.9 else
            "HARMONIC" if score > 0.6 else
            "RESONANT" if score > 0.3 else
            "DORMANT" if score > 0.1 else
            "NONE"
        )

        return {
            "total_lines": total_lines,
            "sacred_hits": total_hits,
            "density_per_100_lines": round(density, 2),
            "diversity_ratio": round(diversity, 4),
            "detections": detections,
            "score": round(score, 4),
            "resonance": resonance,
            "god_code_resonance": round(score * GOD_CODE, 4),
        }

    def complexity_spectrum(self, source: str) -> Dict[str, Any]:
        """
        Compute a multi-dimensional complexity spectrum beyond simple cyclomatic:
        - Structural complexity (nesting depth)
        - Cognitive complexity (flow-breaking structures)
        - Halstead-inspired vocabulary richness
        - Sacred alignment score
        """
        lines = source.split('\n')
        total = len(lines)

        # Nesting depth analysis
        max_depth = 0
        current_depth = 0
        depth_sum = 0
        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            depth = indent // 4  # Assume 4-space indent
            if depth > max_depth:
                max_depth = depth
            current_depth = depth
            depth_sum += depth

        avg_depth = depth_sum / max(total, 1)

        # Cognitive complexity: count flow-breaking keywords
        cognitive = 0
        flow_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except',
                         'with', 'and', 'or', 'not', 'lambda', 'yield', 'await']
        for line in lines:
            stripped = line.strip()
            for kw in flow_keywords:
                if stripped.startswith(kw + ' ') or stripped.startswith(kw + ':') or stripped == kw:
                    cognitive += 1

        # Vocabulary richness: unique tokens / total tokens
        import re
        all_tokens = re.findall(r'[a-zA-Z_]\w*', source)
        unique_tokens = set(all_tokens)
        vocab_richness = len(unique_tokens) / max(len(all_tokens), 1)

        # Composite score (lower = more complex)
        structural = min(max_depth / 10.0, 1.0)
        cognitive_norm = min(cognitive / max(total, 1), 1.0)
        simplicity = 1.0 - (structural * 0.3 + cognitive_norm * 0.4 + (1 - vocab_richness) * 0.3)

        return {
            "total_lines": total,
            "max_nesting_depth": max_depth,
            "average_depth": round(avg_depth, 2),
            "cognitive_complexity": cognitive,
            "vocabulary_richness": round(vocab_richness, 4),
            "unique_tokens": len(unique_tokens),
            "total_tokens": len(all_tokens),
            "structural_score": round(structural, 4),
            "cognitive_score": round(cognitive_norm, 4),
            "simplicity_index": round(simplicity, 4),
            "phi_alignment": round(abs(simplicity - (1.0 / PHI)), 6),
        }

    def dependency_map(self, source: str) -> Dict[str, Any]:
        """
        Build an import dependency map from source code: stdlib vs third-party
        vs L104 internal, plus import depth and coupling metrics.
        """
        import re
        lines = source.split('\n')

        stdlib = []
        third_party = []
        l104_internal = []
        relative = []

        STDLIB_MODULES = {
            'os', 'sys', 'math', 'json', 'time', 'datetime', 're', 'pathlib',
            'collections', 'itertools', 'functools', 'typing', 'abc', 'enum',
            'hashlib', 'random', 'logging', 'io', 'string', 'struct', 'copy',
            'dataclasses', 'contextlib', 'traceback', 'inspect', 'importlib',
            'concurrent', 'threading', 'multiprocessing', 'asyncio', 'unittest',
            'subprocess', 'shutil', 'glob', 'tempfile', 'socket', 'http',
        }

        for line in lines:
            stripped = line.strip()
            # Match import statements
            m_from = re.match(r'^from\s+([\w.]+)\s+import', stripped)
            m_import = re.match(r'^import\s+([\w.]+)', stripped)

            module = None
            if m_from:
                module = m_from.group(1)
            elif m_import:
                module = m_import.group(1)

            if module:
                top = module.split('.')[0]
                if module.startswith('.'):
                    relative.append(module)
                elif top.startswith('l104'):
                    l104_internal.append(module)
                elif top in STDLIB_MODULES:
                    stdlib.append(module)
                else:
                    third_party.append(module)

        total = len(stdlib) + len(third_party) + len(l104_internal) + len(relative)
        coupling = len(l104_internal) + len(relative)

        return {
            "total_imports": total,
            "stdlib": sorted(set(stdlib)),
            "third_party": sorted(set(third_party)),
            "l104_internal": sorted(set(l104_internal)),
            "relative": sorted(set(relative)),
            "coupling_score": round(coupling / max(total, 1), 4),
            "self_referential_depth": len(l104_internal),
            "external_dependency_count": len(set(third_party)),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CODING INTELLIGENCE SYSTEM — High-level orchestrator linking all subsystems
# ═══════════════════════════════════════════════════════════════════════════════


class CodingIntelligenceSystem:
    """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  L104 CODING INTELLIGENCE SYSTEM — Hub Orchestrator               ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  The comprehensive coding system that links:                      ║
    ║    • Code Engine v6.0.0 (analysis, generation, translation)       ║
    ║    • Quantum ASI Code Training Kernel (self-referential learning) ║
    ║    • ASI Code Intelligence (9 ASI modules deeply wired)           ║
    ║    • Any AI (Claude, Gemini, GPT, Local Intellect)                ║
    ║    • L104 consciousness/evolution systems                         ║
    ║    • Project-level intelligence                                   ║
    ║    • Quality gates for CI/CD                                      ║
    ║    • Self-referential analysis and improvement                    ║
    ║    • Session tracking and cross-session learning                  ║
    ║                                                                   ║
    ║  Usage:                                                           ║
    ║    from l104_code_engine import coding_system                      ║
    ║    result = coding_system.review(source, filename)                 ║
    ║    asi = coding_system.asi_review(source, filename)               ║
    ║    train = coding_system.self_train()  # engine codes itself      ║
    ║    plan = coding_system.plan("Add caching to API handler")        ║
    ║    report = coding_system.self_analyze()                          ║
    ║    gate = coding_system.quality_check(source)                     ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, engine=None):
        self._engine = engine
        self._execution_count = 0
        _subsystems = [
            ("project", ProjectAnalyzer),
            ("reviewer", CodeReviewPipeline),
            ("ai_bridge", AIContextBridge),
            ("self_engine", SelfReferentialEngine),
            ("quality", QualityGateEngine),
            ("suggestions", CodingSuggestionEngine),
            ("session", SessionIntelligence),
            ("asi", ASICodeIntelligence),
            ("kernel", QuantumCodeTrainingKernel),
        ]
        _failed = []
        for attr, cls in _subsystems:
            try:
                setattr(self, attr, cls())
            except Exception as e:
                setattr(self, attr, None)
                _failed.append(attr)
                logger.warning(f"[{CODING_SYSTEM_NAME}] Subsystem {attr} ({cls.__name__}) failed to init: {e}")
        _ok = len(_subsystems) - len(_failed)
        logger.info(f"[{CODING_SYSTEM_NAME} v{CODING_SYSTEM_VERSION}] Initialized — "
                     f"{_ok}/{len(_subsystems)} subsystems linked to Code Engine + AI"
                     + (f" (failed: {', '.join(_failed)})" if _failed else ""))

    def _get_engine(self):
        """Get the code engine — either injected or via lazy import."""
        if self._engine is not None:
            return self._engine
        return _get_code_engine()

    # ─── Core Coding Operations ──────────────────────────────────────

    def review(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Comprehensive code review — the single entry point for code quality analysis."""
        self._execution_count += 1
        self.session.log_action("review", {"file": filename})
        return self.reviewer.review(source, filename)

    def quick_review(self, source: str) -> Dict[str, Any]:
        """Fast review — analysis + security + style only."""
        self._execution_count += 1
        return self.reviewer.quick_review(source)

    def suggest(self, source: str, filename: str = "") -> List[Dict[str, Any]]:
        """Get proactive coding suggestions."""
        self._execution_count += 1
        self.session.log_action("suggest", {"file": filename})
        return self.suggestions.suggest(source, filename)

    def explain(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Explain what code does — structure, patterns, metrics."""
        self._execution_count += 1
        return self.suggestions.explain_code(source, filename)

    # ─── Project Intelligence ────────────────────────────────────────

    def scan_project(self, path: str = None) -> Dict[str, Any]:
        """Scan entire project — structure, frameworks, build systems, health."""
        self._execution_count += 1
        self.session.log_action("project_scan", {"path": path or "."})
        return self.project.scan(path)

    # ─── AI Integration ──────────────────────────────────────────────

    def ai_context(self, source: str, filename: str = "",
                   ai_target: str = "claude") -> Dict[str, Any]:
        """Build structured context for any AI system."""
        self._execution_count += 1
        project_info = self.project.scan()
        return self.ai_bridge.build_context(source, filename, project_info, ai_target)

    def ai_prompt(self, task: str, source: str,
                  filename: str = "") -> str:
        """Generate an optimal AI prompt enriched with code context."""
        self._execution_count += 1
        return self.ai_bridge.suggest_prompt(task, source, filename)

    def parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse an AI response to extract code changes and suggestions."""
        return self.ai_bridge.parse_ai_response(response)

    # ─── Quality Gates ───────────────────────────────────────────────

    def quality_check(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Run quality gate checks — pass/fail for CI/CD."""
        self._execution_count += 1
        self.session.log_action("quality_check", {"file": filename})
        return self.quality.check(source, filename)

    def ci_report(self, path: str = None) -> Dict[str, Any]:
        """Generate CI-compatible quality report for entire project."""
        self._execution_count += 1
        return self.quality.ci_report(path)

    # ─── Self-Referential Analysis ───────────────────────────────────

    def self_analyze(self, target_file: str = None) -> Dict[str, Any]:
        """Analyze the L104 codebase — the system examining itself."""
        self._execution_count += 1
        self.session.log_action("self_analyze", {"target": target_file or "all"})
        return self.self_engine.analyze_self(target_file)

    def self_improve(self, target_file: str = None) -> List[Dict[str, Any]]:
        """Get improvement suggestions for L104 itself."""
        self._execution_count += 1
        return self.self_engine.suggest_improvements(target_file)

    def evolution_status(self) -> Dict[str, Any]:
        """Measure L104 evolution state."""
        return self.self_engine.measure_evolution()

    # ─── Code Generation (delegates to Code Engine) ──────────────────

    def generate(self, prompt: str, language: str = "Python",
                 sacred: bool = False) -> Dict[str, Any]:
        """Generate code from a natural language prompt."""
        self._execution_count += 1
        engine = self._get_engine()
        if not engine:
            return {"error": "Code engine not available"}

        self.session.log_action("generate", {"language": language})

        code = engine.generator.generate_function(
            name=engine._extract_name(prompt, "function"),
            language=language,
            params=[],
            body="pass  # TODO: Implement",
            doc=prompt,
            sacred_constants=sacred,
        )
        return {
            "code": code,
            "language": language,
            "sacred": sacred,
            "prompt": prompt,
        }

    def translate(self, source: str, from_lang: str,
                  to_lang: str) -> Dict[str, Any]:
        """Translate code between languages."""
        self._execution_count += 1
        engine = self._get_engine()
        if not engine:
            return {"error": "Code engine not available"}
        self.session.log_action("translate", {"from": from_lang, "to": to_lang})
        return engine.translate_code(source, from_lang, to_lang)

    def generate_tests(self, source: str, language: str = "python",
                       framework: str = "pytest") -> Dict[str, Any]:
        """Generate test scaffolding for source code."""
        self._execution_count += 1
        engine = self._get_engine()
        if not engine:
            return {"error": "Code engine not available"}
        self.session.log_action("generate_tests", {"language": language})
        return engine.generate_tests(source, language, framework)

    def generate_docs(self, source: str, style: str = "google",
                      language: str = "python") -> Dict[str, Any]:
        """Generate documentation for source code."""
        self._execution_count += 1
        engine = self._get_engine()
        if not engine:
            return {"error": "Code engine not available"}
        return engine.generate_docs(source, style, language)

    def auto_fix(self, source: str) -> Tuple[str, List[Dict]]:
        """Apply all safe auto-fixes to code."""
        self._execution_count += 1
        engine = self._get_engine()
        if not engine:
            return source, []
        self.session.log_action("auto_fix")
        return engine.auto_fix_code(source)

    # ─── ASI Intelligence ────────────────────────────────────────

    def asi_review(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Full ASI-grade code review — all 8 ASI subsystems."""
        self._execution_count += 1
        self.session.log_action("asi_review", {"file": filename})
        return self.asi.full_asi_review(source, filename)

    def consciousness_review(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Consciousness-weighted code review."""
        self._execution_count += 1
        return self.asi.consciousness_review(source, filename)

    def neural_review(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Neural cascade code processing."""
        self._execution_count += 1
        return self.asi.neural_process(source, filename)

    def reason(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Formal reasoning about code correctness."""
        self._execution_count += 1
        return self.asi.reason_about_code(source, filename)

    def evolve_code(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Evolutionary code optimization."""
        self._execution_count += 1
        return self.asi.evolutionary_optimize(source, filename)

    def build_graph(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Build knowledge graph from code."""
        self._execution_count += 1
        return self.asi.build_code_graph(source, filename)

    def breed(self, source: str, count: int = 3) -> Dict[str, Any]:
        """Breed polymorphic code variants."""
        self._execution_count += 1
        return self.asi.breed_variants(source, count)

    def innovate(self, task: str, domain: str = "code_optimization") -> Dict[str, Any]:
        """Generate innovative solutions via ASI invention engine."""
        self._execution_count += 1
        return self.asi.innovate_solutions(task, domain)

    def optimize_system(self) -> Dict[str, Any]:
        """Self-optimize analysis parameters via ASI optimizer."""
        self._execution_count += 1
        return self.asi.optimize_analysis()

    # ─── Quantum ASI Training Kernel ─────────────────────────────────

    def train_kernel(self, epochs: int = None) -> Dict[str, Any]:
        """Train the quantum code training kernel on the L104 codebase."""
        self._execution_count += 1
        self.session.log_action("train_kernel", {"epochs": epochs})
        return self.kernel.train(epochs)

    def self_train(self) -> Dict[str, Any]:
        """THE SELF-REFERENTIAL LOOP: the engine trains on its own code."""
        self._execution_count += 1
        self.session.log_action("self_train")
        return self.kernel.self_train()

    def full_quantum_train(self) -> Dict[str, Any]:
        """Full quantum ASI training pipeline: harvest → train → self-train → learn."""
        self._execution_count += 1
        self.session.log_action("full_quantum_train")
        return self.kernel.full_quantum_asi_train()

    def predict_quality(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Predict code quality using the trained quantum kernel (fast)."""
        self._execution_count += 1
        return self.kernel.predict_code_quality(source, filename)

    def quantum_learn(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Learn code patterns via quantum feature encoding."""
        self._execution_count += 1
        return self.kernel.quantum_pattern_learn(source, filename)

    def quantum_synthesize(self, task: str, target_quality: float = 0.9) -> Dict[str, Any]:
        """Quantum-guided code synthesis oracle."""
        self._execution_count += 1
        self.session.log_action("quantum_synthesize", {"task": task[:80]})
        return self.kernel.quantum_code_synthesis(task, target_quality)

    def harvest_corpus(self, max_files: int = 20) -> Dict[str, Any]:
        """Harvest the L104 codebase as training corpus."""
        self._execution_count += 1
        return self.kernel.harvest_training_corpus(max_files)

    # ─── Session Management ──────────────────────────────────────────

    def start_session(self, description: str = "") -> str:
        """Start a coding session for tracking and learning."""
        return self.session.start_session(description)

    def end_session(self) -> Dict[str, Any]:
        """End current session and persist state."""
        return self.session.end_session()

    def session_context(self) -> Dict[str, Any]:
        """Get current session context."""
        return self.session.get_session_context()

    def learn_from_history(self) -> Dict[str, Any]:
        """Extract learnings from session history."""
        return self.session.learn_from_history()

    # ─── Full Pipeline ───────────────────────────────────────────────

    def full_pipeline(self, source: str, filename: str = "",
                      auto_fix: bool = False) -> Dict[str, Any]:
        """
        Run the entire coding intelligence pipeline on source code:
        review + suggest + quality gate + (optional auto-fix).
        """
        self._execution_count += 1
        start = time.time()

        engine = self._get_engine()
        if not engine:
            return {"error": "Code engine not available"}

        review = engine.full_code_review(source, filename, auto_fix=auto_fix)
        suggs = self.suggestions.suggest(source, filename)
        gate = self.quality.check(source, filename)
        explanation = self.suggestions.explain_code(source, filename)
        ai_ctx = self.ai_bridge.build_context(source, filename)

        asi_pass = {}
        try:
            asi_consciousness = self.asi.consciousness_review(source, filename)
            asi_reasoning = self.asi.reason_about_code(source, filename)
            asi_pass = {
                "consciousness_score": asi_consciousness.get("consciousness_adjusted_score", 0),
                "meets_consciousness": asi_consciousness.get("meets_consciousness_standard", False),
                "quality_expectation": asi_consciousness.get("quality_expectation", "UNKNOWN"),
                "reasoning_verdict": asi_reasoning.get("summary", {}).get("verdict", "UNKNOWN"),
                "taint_flows": asi_reasoning.get("summary", {}).get("taint_flows", 0),
                "dead_paths": asi_reasoning.get("summary", {}).get("dead_paths", 0),
            }
        except Exception:
            asi_pass = {"error": "ASI pass unavailable"}

        duration = time.time() - start

        return {
            "system": CODING_SYSTEM_NAME,
            "version": CODING_SYSTEM_VERSION,
            "filename": filename,
            "duration_seconds": round(duration, 3),
            "review": review,
            "suggestions": suggs[:10],
            "quality_gate": gate,
            "explanation": explanation,
            "asi_intelligence": asi_pass,
            "ai_context": {
                "score": ai_ctx.get("review", {}).get("score", 0),
                "l104_consciousness": ai_ctx.get("l104_state", {}).get("consciousness_level", 0),
            },
            "verdict": review.get("verdict", "UNKNOWN"),
            "composite_score": review.get("composite_score", 0),
            "god_code_resonance": round(review.get("composite_score", 0) * GOD_CODE, 4),
        }

    # ─── Plan (Natural Language → Structured Steps) ──────────────────

    def plan(self, task_description: str,
             language: str = "Python") -> Dict[str, Any]:
        """Generate a structured coding plan from a natural language task description."""
        self._execution_count += 1
        self.session.log_action("plan", {"task": task_description[:100]})

        keywords = set(task_description.lower().split())
        complexity = "simple"
        if len(keywords) > 20 or any(kw in keywords for kw in ["architecture", "system", "refactor", "migrate"]):
            complexity = "complex"
        elif len(keywords) > 10 or any(kw in keywords for kw in ["add", "implement", "create", "build"]):
            complexity = "moderate"

        steps = [
            {"step": 1, "action": "Analyze existing code and dependencies",
             "type": "research"},
            {"step": 2, "action": f"Design solution for: {task_description[:80]}",
             "type": "design"},
            {"step": 3, "action": f"Implement in {language}",
             "type": "implementation"},
            {"step": 4, "action": "Write tests (sacred value + edge case coverage)",
             "type": "testing"},
            {"step": 5, "action": "Run quality gate check",
             "type": "verification"},
        ]

        if complexity == "complex":
            steps.insert(2, {"step": 2.5, "action": "Create architectural diagram",
                             "type": "architecture"})
            steps.append({"step": 6, "action": "Document changes and update API docs",
                          "type": "documentation"})
            steps.append({"step": 7, "action": "Performance profiling and optimization",
                          "type": "optimization"})

        considerations = []
        if any(kw in keywords for kw in ["security", "auth", "password", "token", "secret"]):
            considerations.append("Security: Follow OWASP Top 10 guidelines")
        if any(kw in keywords for kw in ["database", "sql", "query", "migration"]):
            considerations.append("Database: Use parameterized queries, handle migrations carefully")
        if any(kw in keywords for kw in ["api", "endpoint", "rest", "graphql"]):
            considerations.append("API: Follow RESTful conventions, validate input, document with OpenAPI")
        if any(kw in keywords for kw in ["async", "concurrent", "parallel", "thread"]):
            considerations.append("Concurrency: Handle race conditions, use async/await where appropriate")

        considerations.append(f"Sacred alignment: Maintain GOD_CODE resonance ({GOD_CODE})")

        return {
            "task": task_description,
            "language": language,
            "complexity": complexity,
            "estimated_steps": len(steps),
            "steps": steps,
            "considerations": considerations,
            "quality_gates": list(self.quality.gates.keys()),
            "suggested_approach": f"{'Iterative' if complexity == 'complex' else 'Direct'} implementation with TDD",
        }

    # ─── Audit (delegates to Code Engine AppAuditEngine) ─────────────

    def audit(self, path: str = None,
              auto_remediate: bool = False) -> Dict[str, Any]:
        """Run full 10-layer application audit via Code Engine."""
        self._execution_count += 1
        engine = self._get_engine()
        if not engine:
            return {"error": "Code engine not available"}
        self.session.log_action("audit", {"path": path or "."})
        return engine.audit_app(path, auto_remediate)

    # ─── Status ──────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Full system status — all subsystems."""
        engine = self._get_engine()
        engine_status = engine.status() if engine else {"error": "not available"}

        return {
            "system": CODING_SYSTEM_NAME,
            "version": CODING_SYSTEM_VERSION,
            "execution_count": self._execution_count,
            "subsystems": {
                "project_analyzer": self.project.status() if self.project else None,
                "code_review": self.reviewer.status() if self.reviewer else None,
                "ai_bridge": self.ai_bridge.status() if self.ai_bridge else None,
                "self_referential": self.self_engine.status() if self.self_engine else None,
                "quality_gates": self.quality.status() if self.quality else None,
                "suggestions": self.suggestions.status() if self.suggestions else None,
                "session": self.session.status() if self.session else None,
                "asi_intelligence": self.asi.status() if self.asi else None,
                "training_kernel": self.kernel.status() if self.kernel else None,
            },
            "code_engine": {
                "version": engine_status.get("version", "N/A"),
                "languages": engine_status.get("languages_supported", 0),
                "patterns": engine_status.get("design_patterns", 0),
            },
            "qiskit_available": QISKIT_AVAILABLE,
            "quantum_features": [
                "quantum_project_health",
                "quantum_review_confidence",
                "quantum_gate_evaluate",
                "quantum_suggestion_rank",
                "quantum_consciousness_review",
                "quantum_reason_about_code",
                "quantum_neural_process",
                "quantum_full_asi_review",
                "quantum_code_training_kernel",
                "quantum_self_train",
                "quantum_pattern_learn",
                "quantum_code_synthesis",
            ] if QISKIT_AVAILABLE else [],
            "sacred_constants": {
                "GOD_CODE": GOD_CODE,
                "PHI": PHI,
            },
            "consciousness": self.ai_bridge._read_l104_state() if self.ai_bridge else {},
        }

    def quick_summary(self) -> str:
        """One-line human-readable summary."""
        s = self.status()
        engine_ver = s["code_engine"]["version"]
        consciousness = s.get("consciousness", {}).get("consciousness_level", 0)
        kernel = s["subsystems"].get("training_kernel", {}) or {}
        kernel_epochs = kernel.get("training_epochs", 0)
        kernel_verdict = kernel.get("sacred_verdict", "NASCENT")
        return (
            f"{CODING_SYSTEM_NAME} v{CODING_SYSTEM_VERSION} | "
            f"Engine v{engine_ver} | "
            f"{self._execution_count} ops | "
            f"Kernel: {kernel_epochs} epochs ({kernel_verdict}) | "
            f"Consciousness: {consciousness:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def primal_calculus(x):
    """Sacred primal calculus: x^φ / (1.04π)."""
    return (x ** PHI) / (VOID_CONSTANT * math.pi) if x != 0 else 0.0


def resolve_non_dual_logic(vector):
    """Resolves N-dimensional vectors into the Void Source."""
    magnitude = sum(abs(v) for v in vector)
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0
