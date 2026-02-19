"""L104 Code Engine — Domain C: Application Auditing."""
from .constants import *
from .languages import LanguageKnowledge
from .analyzer import CodeAnalyzer
from .refactoring import AutoFixEngine

class AppAuditEngine:
    """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  L104 APP AUDIT ENGINE v2.4.0 — ASI APPLICATION AUDIT SYSTEM     ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Orchestrates a comprehensive audit of any application codebase  ║
    ║  by composing all CodeEngine subsystems into layered audit       ║
    ║  passes with deterministic scoring and actionable verdicts.      ║
    ║                                                                   ║
    ║  Audit Layers:                                                    ║
    ║    L0 — Structural Census (files, langs, LOC, blanks, comments)  ║
    ║    L1 — Complexity & Quality (cyclomatic, Halstead, cognitive)    ║
    ║    L2 — Security Scan (OWASP patterns, vuln density, 21 debt)   ║
    ║    L3 — Dependency Topology (circular imports, orphans, hubs)    ║
    ║    L4 — Dead Code Archaeology (fossils, unreachable, drift)      ║
    ║    L5 — Anti-Pattern Detection (god class, deep nesting, etc.)   ║
    ║    L6 — Refactoring Opportunities (extract, inline, decompose)   ║
    ║    L7 — Sacred Alignment (φ-ratio, GOD_CODE resonance)          ║
    ║    L8 — Auto-Remediation (safe fixes applied + diff report)      ║
    ║    L9 — Verdict & Certification (pass/fail + composite score)    ║
    ║                                                                   ║
    ║  Cross-cut: file risk ranking, code clone detection (Py/Swift/   ║
    ║    JS/TS), remediation plan generator, trend tracking            ║
    ║                                                                   ║
    ║  Produces: audit report dict, JSONL trail, remediation patch     ║
    ║  Wired to: CodeEngine.audit_app() + /api/v6/audit/app endpoints  ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """

    AUDIT_VERSION = "2.5.0"

    # Thresholds for verdict calculation
    THRESHOLDS = {
        "max_avg_cyclomatic": 10,
        "max_function_cyclomatic": 20,
        "max_avg_cognitive": 15,                 # v2.5.0 — cognitive complexity cap
        "max_vuln_density": 0.005,       # vulns per LOC
        "min_docstring_coverage": 0.40,
        "max_circular_imports": 0,
        "max_dead_code_pct": 5.0,
        "max_god_classes": 0,
        "min_sacred_alignment": 0.3,
        "min_health_score": 0.70,
        "max_function_params": 5,
        "max_function_lines": 50,
        "max_nesting_depth": 4,
        "max_line_length": 120,
        "max_debt_density": 0.01,        # debt markers per LOC
        "min_maintainability_index": "C",        # v2.5.0 — MI grade floor (A/B/C/D/F)
        "max_tech_debt_density": 0.02,           # v2.5.0 — tech debt markers per LOC
    }

    # Severity weights for composite score
    LAYER_WEIGHTS = {
        "structural": 0.10,
        "complexity": 0.15,
        "security": 0.25,
        "dependencies": 0.05,
        "dead_code": 0.15,
        "anti_patterns": 0.10,
        "refactoring": 0.05,
        "sacred_alignment": 0.02,
        "remediation": 0.03,
        "quality": 0.10,
    }

    # ─── Extended Security & Debt Patterns (supplements CodeAnalyzer) ───
    DEBT_PATTERNS = {
        "hardcoded_ip": re.compile(
            r'(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
            r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?![\d.])'
        ),
        "hardcoded_url": re.compile(r'["\']https?://[^"\']{10,}["\']'),
        "weak_hash": re.compile(r'(?:hashlib\.(?:md5|sha1)\s*\(|\.(?:MD5|SHA1)\()'),
        "debug_print": re.compile(
            r'^\s*(?:print\s*\(|console\.log\s*\(|debugPrint\s*\(|NSLog\s*\()',
            re.MULTILINE,
        ),
        "bare_except": re.compile(r'\bexcept\s*:', re.MULTILINE),
        "empty_catch": re.compile(r'except[^:]*:\s*\n\s*pass\b', re.MULTILINE),
        "todo_debt": re.compile(
            r'(?:#|//)\s*(?:TODO|FIXME|HACK|XXX|TEMP|KLUDGE)\b',
            re.MULTILINE | re.IGNORECASE,
        ),
        "weak_random": re.compile(r'\brandom\.(?:random|randint|choice|shuffle)\s*\('),
        "assert_in_prod": re.compile(r'^\s*assert\s+', re.MULTILINE),
        "broad_file_perms": re.compile(r'chmod\s*\(.*0o?7[0-7]{2}'),
        "eval_usage": re.compile(r'\beval\s*\(', re.MULTILINE),
        "exec_usage": re.compile(r'\bexec\s*\(', re.MULTILINE),
        "pickle_load": re.compile(r'pickle\.(?:load|loads)\s*\(', re.MULTILINE),
        "subprocess_shell": re.compile(r'subprocess\.(?:call|run|Popen)\s*\([^)]*shell\s*=\s*True'),
        "yaml_unsafe": re.compile(r'yaml\.load\s*\([^)]*\)', re.MULTILINE),
        "os_system": re.compile(r'\bos\.system\s*\(', re.MULTILINE),
        "open_no_encoding": re.compile(
            r'open\s*\([^)]*,\s*["\'][rwa]["\'](?:\s*\)|\s*,\s*(?!encoding))',
            re.MULTILINE,
        ),
        "hardcoded_password": re.compile(
            r'(?:password|passwd|secret|api_key)\s*=\s*["\'][a-zA-Z0-9!@#$%^&*_+=/.-]{8,}["\']',
            re.IGNORECASE,
        ),
        "insecure_request": re.compile(r'verify\s*=\s*False', re.MULTILINE),
        "mutable_default": re.compile(
            r'def\s+\w+\s*\([^)]*=\s*(?:\[\]|\{\}|set\(\))',
            re.MULTILINE,
        ),
    }

    def __init__(self, analyzer: 'CodeAnalyzer', optimizer: 'CodeOptimizer',
                 dep_graph: 'DependencyGraphAnalyzer', auto_fix: 'AutoFixEngine',
                 archeologist: 'CodeArcheologist', refactorer: 'SacredRefactorer'):
        """Initialize AppAuditEngine with all subsystem references."""
        self.analyzer = analyzer
        self.optimizer = optimizer
        self.dep_graph = dep_graph
        self.auto_fix = auto_fix
        self.archeologist = archeologist
        self.refactorer = refactorer
        self.audit_count = 0
        self.audit_history: List[Dict[str, Any]] = []
        self._audit_trail: List[Dict[str, Any]] = []
        logger.info(f"[APP_AUDIT_ENGINE v{self.AUDIT_VERSION}] Initialized — "
                     f"{len(self.THRESHOLDS)} thresholds, "
                     f"{len(self.LAYER_WEIGHTS)} audit layers, "
                     f"{len(self.DEBT_PATTERNS)} debt patterns")

    # ─── Core Audit Pipeline ─────────────────────────────────────────

    def full_audit(self, workspace_path: str = None,
                   auto_remediate: bool = False,
                   target_files: List[str] = None) -> Dict[str, Any]:
        """
        Execute the full 10-layer audit pipeline on a workspace or file list.

        Args:
            workspace_path: Root directory to audit (defaults to project root)
            auto_remediate: If True, apply safe auto-fixes and report diffs
            target_files: Specific files to audit (overrides workspace scan)

        Returns:
            Complete audit report with per-layer results, composite score, and verdict
        """
        self.audit_count += 1
        start_time = time.time()
        ws = Path(workspace_path) if workspace_path else Path(__file__).parent

        # Collect files
        files = self._collect_files(ws, target_files)
        if not files:
            return {"status": "NO_FILES", "message": "No auditable files found"}

        # Read all file contents
        file_contents = {}
        for fp in files:
            try:
                file_contents[fp] = Path(fp).read_text(errors='ignore')
            except Exception:
                pass

        self._trail_event("AUDIT_START", {
            "workspace": str(ws), "files": len(file_contents),
            "auto_remediate": auto_remediate
        })

        # L0 — Structural Census
        l0 = self._layer0_structural_census(file_contents)

        # L1 — Complexity & Quality
        l1 = self._layer1_complexity_quality(file_contents)

        # L2 — Security Scan
        l2 = self._layer2_security_scan(file_contents)

        # L3 — Dependency Topology (skip for single-file audits — too expensive)
        if target_files and len(target_files) <= 3:
            l3 = {"modules_mapped": 0, "edges": 0, "circular_imports": [],
                   "circular_count": 0, "orphan_modules": [], "orphan_count": 0,
                   "hub_modules": [], "max_fan_in": 0, "max_fan_out": 0,
                   "score": 1.0, "note": "skipped_for_single_file_audit"}
        else:
            l3 = self._layer3_dependency_topology(str(ws))

        # L4 — Dead Code Archaeology
        l4 = self._layer4_dead_code_archaeology(file_contents)

        # L5 — Anti-Pattern Detection
        l5 = self._layer5_anti_pattern_detection(file_contents)

        # L6 — Refactoring Opportunities
        l6 = self._layer6_refactoring_opportunities(file_contents)

        # L7 — Sacred Alignment
        l7 = self._layer7_sacred_alignment(file_contents)

        # L8 — Auto-Remediation
        l8 = self._layer8_auto_remediation(file_contents, auto_remediate)

        # Cross-cutting analyses
        file_risks = self._compute_file_risk_ranking(l0, l1, l2, l4, l5, file_contents)
        clones = self._detect_code_clones(file_contents)
        import_hygiene = self._analyze_import_hygiene(file_contents)
        complexity_heatmap = self._build_complexity_heatmap(l1, file_contents)
        architecture = self._analyze_architecture_coupling(file_contents)
        test_coverage = self._estimate_test_coverage(file_contents)
        api_surface = self._analyze_api_surface(file_contents)

        # L9 — Verdict & Certification
        layer_scores = {
            "structural": l0.get("score", 1.0),
            "complexity": l1.get("score", 0.5),
            "security": l2.get("score", 0.5),
            "dependencies": l3.get("score", 0.5),
            "dead_code": l4.get("score", 0.5),
            "anti_patterns": l5.get("score", 0.5),
            "refactoring": l6.get("score", 0.5),
            "sacred_alignment": l7.get("score", 0.5),
            "remediation": l8.get("score", 1.0),
            "quality": l1.get("quality_score", 0.5),
        }
        l9 = self._layer9_verdict(layer_scores)

        duration = time.time() - start_time

        # Generate actionable remediation summary
        remediation_plan = self._generate_remediation_plan(l1, l2, l4, l5, file_risks)

        report = {
            "audit_engine_version": self.AUDIT_VERSION,
            "code_engine_version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "workspace": str(ws),
            "duration_seconds": round(duration, 3),
            "files_audited": len(file_contents),
            "knowledge_context": self._knowledge_context(file_contents),
            "layers": {
                "L0_structural_census": l0,
                "L1_complexity_quality": l1,
                "L2_security_scan": l2,
                "L3_dependency_topology": l3,
                "L4_dead_code_archaeology": l4,
                "L5_anti_pattern_detection": l5,
                "L6_refactoring_opportunities": l6,
                "L7_sacred_alignment": l7,
                "L8_auto_remediation": l8,
                "L9_verdict": l9,
            },
            "file_risk_ranking": file_risks[:20],
            "code_clones": clones,
            "import_hygiene": import_hygiene,
            "complexity_heatmap": complexity_heatmap,
            "architecture_coupling": architecture,
            "test_coverage": test_coverage,
            "api_surface": api_surface,
            "remediation_plan": remediation_plan,
            "composite_score": l9["composite_score"],
            "verdict": l9["verdict"],
            "certification": l9["certification"],
            "god_code_resonance": round(l9["composite_score"] * GOD_CODE, 4),
            "delta_from_last": self._compute_delta(l9["composite_score"], l2, l5),
        }

        self._trail_event("AUDIT_COMPLETE", {
            "score": l9["composite_score"], "verdict": l9["verdict"],
            "duration": duration, "files": len(file_contents)
        })
        self.audit_history.append({
            "timestamp": report["timestamp"],
            "score": l9["composite_score"],
            "verdict": l9["verdict"],
            "files": len(file_contents),
        })

        logger.info(f"[APP_AUDIT] Complete — Score: {l9['composite_score']:.4f} "
                     f"| Verdict: {l9['verdict']} | {len(file_contents)} files "
                     f"in {duration:.2f}s")
        return report

    def audit_file(self, filepath: str) -> Dict[str, Any]:
        """Single-file audit — runs all applicable layers on one file."""
        return self.full_audit(target_files=[filepath])

    def quick_audit(self, workspace_path: str = None) -> Dict[str, Any]:
        """
        Lightweight audit — structural census + security + anti-patterns only.
        Skips dependency graph and remediation for speed.
        """
        ws = Path(workspace_path) if workspace_path else Path(__file__).parent
        files = self._collect_files(ws)
        file_contents = {}
        for fp in files[:50]:  # cap at 50 for speed
            try:
                file_contents[fp] = Path(fp).read_text(errors='ignore')
            except Exception:
                pass

        l0 = self._layer0_structural_census(file_contents)
        l2 = self._layer2_security_scan(file_contents)
        l5 = self._layer5_anti_pattern_detection(file_contents)

        quick_score = (l0.get("score", 1.0) * 0.2 +
                       l2.get("score", 0.5) * 0.5 +
                       l5.get("score", 0.5) * 0.3)

        return {
            "mode": "QUICK_AUDIT",
            "files_scanned": len(file_contents),
            "structural": l0,
            "security": l2,
            "anti_patterns": l5,
            "quick_score": round(quick_score, 4),
            "verdict": self._score_to_verdict(quick_score),
        }

    # ─── Layer Implementations ───────────────────────────────────────

    def _layer0_structural_census(self, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """L0: Structural integrity — census, formatting consistency, canonical shape."""
        total_lines = 0
        total_blank = 0
        total_comment = 0
        lang_dist: Dict[str, int] = defaultdict(int)
        largest_file = ("", 0)
        # Formatting consistency accumulators
        lines_over_80 = 0
        lines_over_120 = 0
        trailing_ws_count = 0
        mixed_indent_files = 0
        tab_files = 0
        space_files = 0

        for fp, code in file_contents.items():
            lines = code.split('\n')
            n = len(lines)
            total_lines += n
            blank = sum(1 for l in lines if not l.strip())
            total_blank += blank
            # Count comments: # (Python), // (Swift/JS/C), and docstrings
            comment = 0
            in_docstring = False
            for l in lines:
                s = l.strip()
                if s.startswith('#') or s.startswith('//'):
                    comment += 1
                elif '"""' in s or "'''" in s:
                    # Docstring line — count as documentation
                    comment += 1
                    # Toggle docstring state for multi-line
                    quotes = '"""' if '"""' in s else "'''"
                    count_q = s.count(quotes)
                    if count_q == 1:
                        in_docstring = not in_docstring
                elif in_docstring:
                    comment += 1
            total_comment += comment
            lang = LanguageKnowledge.detect_language(code, fp)
            lang_dist[lang] += n
            if n > largest_file[1]:
                largest_file = (Path(fp).name, n)

            # Formatting checks
            has_tabs = False
            has_spaces = False
            for line in lines:
                if len(line) > 120:
                    lines_over_120 += 1
                elif len(line) > 80:
                    lines_over_80 += 1
                if line != line.rstrip():
                    trailing_ws_count += 1
                stripped = line.lstrip()
                if stripped and line != stripped:  # indented line
                    indent = line[:len(line) - len(stripped)]
                    if '\t' in indent:
                        has_tabs = True
                    if ' ' in indent:
                        has_spaces = True
            if has_tabs and has_spaces:
                mixed_indent_files += 1
            elif has_tabs:
                tab_files += 1
            else:
                space_files += 1

        code_lines = total_lines - total_blank - total_comment
        comment_ratio = total_comment / max(1, code_lines)

        # Formatting score: penalize mixed indentation, long lines, trailing WS
        fmt_penalty = 0.0
        fmt_penalty += mixed_indent_files * 0.05
        # Normalize line length penalties per total lines (not absolute)
        over_120_ratio = lines_over_120 / max(1, total_lines)
        fmt_penalty += min(0.08, over_120_ratio * 1.5)
        trailing_ratio = trailing_ws_count / max(1, total_lines)
        fmt_penalty += min(0.05, trailing_ratio * 1.0)

        # Overall structure score (v2.4: graduated large-file penalty, comment bonus)
        score = min(1.0, 0.6 + comment_ratio * 0.45)

        # Graduated large-file penalty (softer curve for monolithic native apps)
        if largest_file[1] > 5000:
            excess = min(largest_file[1], 50000) - 5000
            score -= min(0.10, excess / 450000)  # max -0.10 at 50K lines

        # Bonus for good comment ratio (well-documented code)
        if comment_ratio > 0.15:
            score += min(0.05, (comment_ratio - 0.15) * 0.5)

        # Multi-language diversity bonus (polyglot codebases score higher)
        if len(lang_dist) >= 3:
            score += 0.03

        score -= fmt_penalty

        return {
            "total_files": len(file_contents),
            "total_lines": total_lines,
            "code_lines": code_lines,
            "blank_lines": total_blank,
            "comment_lines": total_comment,
            "comment_ratio": round(comment_ratio, 4),
            "language_distribution": dict(lang_dist),
            "largest_file": {"name": largest_file[0], "lines": largest_file[1]},
            "formatting": {
                "lines_over_80": lines_over_80,
                "lines_over_120": lines_over_120,
                "trailing_whitespace_lines": trailing_ws_count,
                "mixed_indent_files": mixed_indent_files,
                "tab_files": tab_files,
                "space_files": space_files,
                "indent_consistency": "CONSISTENT" if mixed_indent_files == 0 else "MIXED",
            },
            "score": round(max(0.0, score), 4),
        }

    def _layer1_complexity_quality(self, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """L1: Style, conventions, complexity — cyclomatic smells, naming, doc coverage."""
        all_cyclomatic = []
        all_cognitive = []
        all_halstead = []
        total_functions = 0
        total_classes = 0
        docstring_hits = 0
        docstring_total = 0
        hotspot_files = []
        # Code smell accumulators
        smells: List[Dict[str, Any]] = []
        naming_violations = 0
        magic_number_count = 0

        for fp, code in file_contents.items():
            fname = Path(fp).name
            analysis = self.analyzer.full_analysis(code, fp)
            complexity = analysis.get("complexity", {})
            quality = analysis.get("quality", {})

            funcs = complexity.get("functions", [])
            total_functions += len(funcs)
            total_classes += complexity.get("class_count", 0)

            for fn in funcs:
                fn_name = fn.get("name", "?")
                fn_line = fn.get("line", 0)
                cc = fn.get("cyclomatic_complexity", 1)
                cog = fn.get("cognitive_complexity", 0)
                args = fn.get("args", 0)
                body_lines = fn.get("body_lines", 0)
                all_cyclomatic.append(cc)
                all_cognitive.append(cog)

                # Smell: high cyclomatic complexity
                if cc > self.THRESHOLDS["max_function_cyclomatic"]:
                    smells.append({"smell": "high_cyclomatic", "file": fname,
                                   "function": fn_name, "line": fn_line,
                                   "value": cc, "threshold": self.THRESHOLDS["max_function_cyclomatic"],
                                   "severity": "HIGH"})
                elif cc > self.THRESHOLDS["max_avg_cyclomatic"]:
                    smells.append({"smell": "elevated_cyclomatic", "file": fname,
                                   "function": fn_name, "line": fn_line,
                                   "value": cc, "threshold": self.THRESHOLDS["max_avg_cyclomatic"],
                                   "severity": "MEDIUM"})

                # Smell: long parameter list
                if args > self.THRESHOLDS["max_function_params"]:
                    smells.append({"smell": "long_param_list", "file": fname,
                                   "function": fn_name, "line": fn_line,
                                   "value": args, "threshold": self.THRESHOLDS["max_function_params"],
                                   "severity": "MEDIUM"})

                # Smell: long method body
                if body_lines > self.THRESHOLDS["max_function_lines"]:
                    smells.append({"smell": "long_method", "file": fname,
                                   "function": fn_name, "line": fn_line,
                                   "value": body_lines, "threshold": self.THRESHOLDS["max_function_lines"],
                                   "severity": "MEDIUM"})

                # Naming convention: Python functions should be snake_case
                lang = LanguageKnowledge.detect_language(code, fp)
                if lang == "Python" and not fn_name.startswith('_'):
                    if fn_name != fn_name.lower() and not fn_name.startswith('test'):
                        naming_violations += 1

            halstead = complexity.get("halstead", {})
            if halstead.get("effort", 0) > 0:
                all_halstead.append(halstead["effort"])

            doc_cov = quality.get("docstring_coverage")
            if doc_cov is not None:
                docstring_hits += doc_cov
                docstring_total += 1

            # Magic numbers
            magic = quality.get("magic_numbers", 0)
            if isinstance(magic, int):
                magic_number_count += magic

            # Hotspot: files with high max cyclomatic
            max_cc = max((fn.get("cyclomatic_complexity", 0) for fn in funcs), default=0)
            if max_cc > self.THRESHOLDS["max_avg_cyclomatic"]:
                hotspot_files.append({"file": fname, "max_cc": max_cc})

            # Smell: deep nesting
            max_nest = complexity.get("max_nesting", 0)
            if max_nest > self.THRESHOLDS["max_nesting_depth"]:
                smells.append({"smell": "deep_nesting", "file": fname,
                               "value": max_nest, "threshold": self.THRESHOLDS["max_nesting_depth"],
                               "severity": "MEDIUM"})

        avg_cc = sum(all_cyclomatic) / max(1, len(all_cyclomatic))
        avg_cog = sum(all_cognitive) / max(1, len(all_cognitive))
        avg_doc = docstring_hits / max(1, docstring_total)

        # Score: lower complexity = higher score, smells reduce further
        cc_score = max(0.0, 1.0 - (avg_cc / (self.THRESHOLDS["max_avg_cyclomatic"] * 2)))
        # Smell penalty scaled per file (avoids punishing large codebases unfairly)
        file_count = max(1, len(file_contents))
        high_smells = len([s for s in smells if s["severity"] == "HIGH"])
        med_smells = len([s for s in smells if s["severity"] == "MEDIUM"])
        smell_per_file = (high_smells * 2.0 + med_smells) / file_count
        smell_penalty = min(0.25, smell_per_file * 0.12)
        cc_score = max(0.0, cc_score - smell_penalty)
        quality_score = min(1.0, avg_doc + 0.35) if avg_doc > 0 else 0.5

        smell_summary: Dict[str, int] = defaultdict(int)
        for s in smells:
            smell_summary[s["smell"]] += 1

        return {
            "total_functions": total_functions,
            "total_classes": total_classes,
            "avg_cyclomatic_complexity": round(avg_cc, 3),
            "avg_cognitive_complexity": round(avg_cog, 3),
            "max_cyclomatic": max(all_cyclomatic, default=0),
            "avg_halstead_effort": round(sum(all_halstead) / max(1, len(all_halstead)), 2),
            "docstring_coverage": round(avg_doc, 4),
            "hotspot_files": sorted(hotspot_files, key=lambda h: h["max_cc"], reverse=True)[:10],
            "code_smells": smells[:30],
            "smell_summary": dict(smell_summary),
            "smell_count": len(smells),
            "naming_violations": naming_violations,
            "magic_numbers": magic_number_count,
            "score": round(cc_score, 4),
            "quality_score": round(quality_score, 4),
        }

    # Standard/benign IPs excluded from hardcoded_ip flagging
    BENIGN_IPS = frozenset({
        "0.0.0.0", "127.0.0.1", "255.255.255.0", "255.255.255.255",
        "192.168.0.1", "192.168.1.1", "10.0.0.0", "10.0.0.1",
        "172.16.0.0", "169.254.0.0",
        "120.0.0.0",  # CIDR notation — not a real server IP
        "8.8.8.8", "1.1.1.1",  # Public DNS
        "208.67.222.222", "208.67.220.220",  # OpenDNS
    })

    # Severity weights for debt density (fairer than raw count)
    DEBT_SEVERITY_WEIGHTS = {
        "HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.1, "INFO": 0.0,
    }

    def _layer2_security_scan(self, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """L2: Security vulnerabilities + technical debt — OWASP patterns, hardcoded
        secrets, weak crypto, debug leaks, bare excepts, debt markers.

        Improvements v2.3:
        - Fixed category mapping bug for OWASP vulns (was always 'unknown')
        - Severity-weighted debt density replaces raw count density
        - Benign IPs (loopback, broadcast, private) are excluded
        - Context-aware debug_print: skips files with __main__ guard
        - Deduplication between OWASP and DEBT_PATTERNS scanners
        """
        all_vulns = []
        vuln_by_category: Dict[str, int] = defaultdict(int)
        debt_items: List[Dict[str, Any]] = []
        debt_by_type: Dict[str, int] = defaultdict(int)
        total_loc = sum(len(c.split('\n')) for c in file_contents.values())
        # Dedup set: (file, line) pairs to avoid double-counting OWASP + DEBT
        seen_vuln_locs: set = set()

        for fp, code in file_contents.items():
            fname = Path(fp).name
            is_main_script = '__name__' in code and "'__main__'" in code or '"__main__"' in code
            is_python = fp.endswith('.py')

            # Python-specific patterns — skip in non-Python files
            PYTHON_ONLY_PATTERNS = frozenset({
                "bare_except", "eval_usage", "exec_usage",
                "mutable_default", "assert_in_prod", "debug_print",
                "open_no_encoding", "empty_catch",
            })

            # Standard OWASP scan via CodeAnalyzer
            vulns = self.analyzer._security_scan(code)
            for v in vulns:
                v["file"] = fname
                # BUG FIX: _security_scan returns "type" field, not "category"
                v["category"] = v.get("type", "unknown")
                loc_key = (fname, v.get("line", 0))
                seen_vuln_locs.add(loc_key)
                all_vulns.append(v)
                vuln_by_category[v["category"]] += 1

            # Extended debt/weakness pattern scan
            for pattern_name, pattern in self.DEBT_PATTERNS.items():
                # Skip Python-specific patterns in non-Python files
                if not is_python and pattern_name in PYTHON_ONLY_PATTERNS:
                    continue

                for match in pattern.finditer(code):
                    matched_text = match.group()

                    # Context filtering: skip benign IPs
                    if pattern_name == "hardcoded_ip":
                        ip_str = matched_text.strip()
                        if ip_str in self.BENIGN_IPS:
                            continue

                    # Context filtering: skip placeholder passwords/keys
                    if pattern_name == "hardcoded_password":
                        val_lower = matched_text.lower()
                        if any(p in val_lower for p in (
                            'not-configured', 'placeholder', 'changeme',
                            'your-', 'xxx', 'todo', 'none', 'empty',
                        )):
                            continue

                    # Context filtering: skip benign URLs (localhost, docs, public APIs)
                    if pattern_name == "hardcoded_url":
                        url_lower = matched_text.lower()
                        if any(safe in url_lower for safe in (
                            'localhost', '127.0.0.1', '0.0.0.0',
                            'example.com', 'example.org',
                            'docs.python.org', 'pypi.org',
                            'github.com', 'readthedocs',
                            'schemas.', 'schema.org',
                            'www.w3.org', 'json-schema.org',
                            # Public search/knowledge APIs
                            'duckduckgo.com', 'wikipedia.org',
                            'apple.com/dtd', 'apple.com/DTD',
                            # Public blockchain RPCs & explorers
                            'etherscan.io', 'basescan.org',
                            'arbiscan.io', 'polygonscan.com',
                            'bscscan.com', 'infura.io',
                            'alchemy.com', 'mainnet.base.org',
                            'arbitrum.io', 'polygon-rpc.com',
                            'cloudflare-eth.com', 'sepolia.org',
                            # Google/cloud APIs (public docs)
                            'googleapis.com', 'storage.googleapis.com',
                            # Standard protocol schemas
                            'xmlns', 'dtd', 'purl.org',
                        )):
                            continue

                    # Context filtering: skip debug_print in __main__ scripts
                    if pattern_name == "debug_print" and is_main_script:
                        continue

                    # Context filtering: skip bare_except in string literals
                    if pattern_name == "bare_except":
                        pre = code[:match.start()]
                        last_nl = pre.rfind('\n')
                        line_prefix = pre[last_nl + 1:] if last_nl >= 0 else pre
                        if any(q in line_prefix for q in ('"""', "'''", '"', "'")):
                            # Likely inside a string literal — skip
                            stripped_prefix = line_prefix.lstrip()
                            if stripped_prefix and stripped_prefix[0] in ('"', "'",
                                                                          'f', 'r', 'b'):
                                continue

                    # Context filtering: sandboxed eval/exec (restricted __builtins__)
                    if pattern_name in ("eval_usage", "exec_usage"):
                        ctx = code[max(0, match.start()-60):match.end()+150]
                        # Check for sandboxing indicators
                        sandbox_indicators = (
                            '__builtins__', 'namespace', 'allowed',
                            'safe_dict', 'exec_result',
                        )
                        if any(ind in ctx for ind in sandbox_indicators):
                            severity = "LOW"
                            entry = {
                                "type": pattern_name,
                                "file": fname,
                                "line": code[:match.start()].count('\n') + 1,
                                "match": matched_text[:80].strip(),
                                "severity": severity,
                            }
                            debt_items.append(entry)
                            debt_by_type[pattern_name] += 1
                            continue
                        # Check for string literal context (docstrings,
                        # remediation advice, JS code in strings)
                        pre = code[:match.start()]
                        last_nl = pre.rfind('\n')
                        line_prefix = pre[last_nl + 1:] if last_nl >= 0 else pre
                        stripped_lp = line_prefix.lstrip()
                        # Skip if inside string assignments, docstrings, or comments
                        if stripped_lp.startswith(('#', '"', "'", 'f"', "f'", 'r"', "r'")):
                            severity = "LOW"
                            entry = {
                                "type": pattern_name,
                                "file": fname,
                                "line": code[:match.start()].count('\n') + 1,
                                "match": matched_text[:80].strip(),
                                "severity": severity,
                            }
                            debt_items.append(entry)
                            debt_by_type[pattern_name] += 1
                            continue
                        # Check if inside a triple-quoted docstring
                        pre_code = code[:match.start()]
                        triple_dq = pre_code.count('"""')
                        triple_sq = pre_code.count("'''")
                        if triple_dq % 2 == 1 or triple_sq % 2 == 1:
                            # Odd count means we're inside a docstring
                            continue
                        # Skip if preceded by . (method call like pattern.exec_fn())
                        char_before = code[match.start()-1:match.start()] if match.start() > 0 else ''
                        if char_before == '.':
                            continue
                        # Skip if match appears inside print/f-string/docstring content
                        full_line = code.split('\n')[code[:match.start()].count('\n')]
                        if 'print(' in full_line and 'eval' in full_line:
                            continue

                    line_num = code[:match.start()].count('\n') + 1
                    is_security = pattern_name in (
                        "hardcoded_ip", "hardcoded_url", "weak_hash",
                        "bare_except", "broad_file_perms",
                        "eval_usage", "exec_usage", "pickle_load",
                        "subprocess_shell", "yaml_unsafe", "os_system",
                        "hardcoded_password", "insecure_request",
                    )
                    severity = "HIGH" if is_security else "MEDIUM"
                    if pattern_name in ("debug_print", "assert_in_prod",
                                        "open_no_encoding", "mutable_default",
                                        "empty_catch"):
                        severity = "LOW"
                    if pattern_name == "todo_debt":
                        severity = "INFO"
                    # weak_random: context-aware severity — crypto-critical
                    # uses already fixed; remaining are simulation/test = MEDIUM
                    if pattern_name == "weak_random":
                        crypto_ctx = any(kw in code[max(0, match.start()-200):match.end()+200]
                                         for kw in ('token', 'secret', 'key', 'password',
                                                    'auth', 'crypto', 'nonce'))
                        if crypto_ctx:
                            severity = "HIGH"
                            is_security = True
                        else:
                            severity = "MEDIUM"

                    entry = {
                        "type": pattern_name,
                        "file": fname,
                        "line": line_num,
                        "match": matched_text[:80].strip(),
                        "severity": severity,
                    }
                    debt_items.append(entry)
                    debt_by_type[pattern_name] += 1

                    # Promote security-relevant debt to vuln list (dedup with OWASP)
                    if is_security:
                        loc_key = (fname, line_num)
                        if loc_key not in seen_vuln_locs:
                            seen_vuln_locs.add(loc_key)
                            all_vulns.append({
                                "type": pattern_name,
                                "category": pattern_name,
                                "severity": severity,
                                "line": line_num,
                                "match": matched_text[:80].strip(),
                                "file": fname,
                                "recommendation": self._debt_recommendation(pattern_name),
                            })
                            vuln_by_category[pattern_name] += 1

        vuln_density = len(all_vulns) / max(1, total_loc)

        # Severity-weighted debt density — fairer than raw count
        # HIGH=1.0, MEDIUM=0.5, LOW=0.1, INFO=0.0
        weighted_debt = sum(
            self.DEBT_SEVERITY_WEIGHTS.get(d["severity"], 0.5)
            for d in debt_items
        )
        weighted_debt_density = weighted_debt / max(1, total_loc)
        raw_debt_density = len(debt_items) / max(1, total_loc)

        # Score: blended security + debt (using weighted density)
        sec_score = max(0.0, 1.0 - (vuln_density / self.THRESHOLDS["max_vuln_density"]))
        debt_score = max(0.0, 1.0 - (weighted_debt_density / self.THRESHOLDS["max_debt_density"]))
        score = sec_score * 0.7 + debt_score * 0.3

        return {
            "total_vulnerabilities": len(all_vulns),
            "vuln_density_per_loc": round(vuln_density, 6),
            "by_category": dict(vuln_by_category),
            "critical_vulns": [v for v in all_vulns if v.get("severity") == "HIGH"][:15],
            "all_vulns": all_vulns[:30],
            "owasp_coverage": len(CodeAnalyzer.SECURITY_PATTERNS),
            "technical_debt": {
                "total_items": len(debt_items),
                "weighted_items": round(weighted_debt, 1),
                "by_type": dict(debt_by_type),
                "raw_density_per_loc": round(raw_debt_density, 6),
                "weighted_density_per_loc": round(weighted_debt_density, 6),
                "items": sorted(debt_items, key=lambda d: {
                    "HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3
                }.get(d["severity"], 4))[:25],
            },
            "score": round(max(0.0, min(1.0, score)), 4),
        }

    @staticmethod
    def _debt_recommendation(pattern_name: str) -> str:
        """Return remediation advice for a debt/weakness pattern."""
        recs = {
            "hardcoded_ip": "Move IP addresses to configuration or environment variables",
            "hardcoded_url": "Extract URLs to config; avoid embedding endpoints in source",
            "weak_hash": "Use SHA-256+ for integrity; use bcrypt/scrypt/argon2 for passwords",
            "debug_print": "Replace debug prints with proper logging (logger.debug)",
            "bare_except": "Catch specific exceptions; bare except masks real errors",
            "empty_catch": "Log or handle the exception instead of silently passing",
            "todo_debt": "Resolve or schedule TODO/FIXME items to reduce tech debt",
            "weak_random": "Use secrets module for security-sensitive random values",
            "assert_in_prod": "Assertions are stripped by -O; use explicit checks + raise",
            "broad_file_perms": "Restrict file permissions (avoid 0o777/world-writable)",
            "eval_usage": "Avoid eval(); use ast.literal_eval() or structured parsing",
            "exec_usage": "Avoid exec(); use safe alternatives or sandboxed execution",
            "pickle_load": "Pickle deserialization is unsafe with untrusted data; use JSON",
            "subprocess_shell": "Avoid shell=True; pass args as list to prevent injection",
            "yaml_unsafe": "Use yaml.safe_load() instead of yaml.load() to prevent code execution",
            "os_system": "Replace os.system() with subprocess.run() to prevent shell injection",
            "open_no_encoding": "Add encoding='utf-8' to open() for cross-platform text handling",
            "hardcoded_password": "Move secrets to environment variables or keychain",
            "insecure_request": "Do not disable SSL verification (verify=False); use proper certs",
            "mutable_default": "Use None as default and initialize inside function body",
        }
        return recs.get(pattern_name, "Review and remediate")

    def _layer3_dependency_topology(self, workspace_path: str) -> Dict[str, Any]:
        """L3: Dependency graph analysis — circular imports, orphans, hub overload."""
        graph = self.dep_graph.build_graph(workspace_path)
        circular = graph.get("circular_imports", [])
        orphans = graph.get("orphan_modules", [])
        hub_overload = graph.get("hub_modules", [])

        # Score: penalize circular imports heavily
        circular_penalty = len(circular) * 0.15
        orphan_penalty = len(orphans) * 0.02
        score = max(0.0, 1.0 - circular_penalty - orphan_penalty)

        return {
            "modules_mapped": graph.get("total_modules", 0),
            "edges": graph.get("total_edges", 0),
            "circular_imports": circular[:10],
            "circular_count": len(circular),
            "orphan_modules": orphans[:15],
            "orphan_count": len(orphans),
            "hub_modules": hub_overload[:10],
            "max_fan_in": graph.get("max_fan_in", 0),
            "max_fan_out": graph.get("max_fan_out", 0),
            "score": round(score, 4),
        }

    def _layer4_dead_code_archaeology(self, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """L4: Dead code — unused functions, classes, variables, unreachable paths,
        commented-out code blocks, and fossil analysis."""
        total_fossils = 0
        total_dead = 0
        total_todos = 0
        fossil_breakdown: Dict[str, int] = defaultdict(int)
        worst_files = []

        # AST-based unused symbol detection (Python files)
        unused_functions: List[Dict[str, Any]] = []
        unused_classes: List[Dict[str, Any]] = []
        unused_variables: List[Dict[str, Any]] = []
        commented_code_blocks: List[Dict[str, Any]] = []

        for fp, code in file_contents.items():
            fname = Path(fp).name

            # Standard archeological excavation
            excavation = self.archeologist.excavate(code)
            fossils = excavation.get("fossils", [])
            dead = excavation.get("dead_code", [])
            total_fossils += len(fossils)
            total_dead += len(dead)

            for f in fossils:
                fossil_breakdown[f.get("type", "unknown")] += 1
                if f.get("type") == "todo_marker":
                    total_todos += 1

            health = excavation.get("health_score", 1.0)
            if health < 0.7:
                worst_files.append({"file": fname, "health": health,
                                    "dead_paths": len(dead)})

            # AST-based deep scan for Python files
            if fp.endswith('.py'):
                py_unused = self._detect_unused_symbols_ast(code, fname)
                unused_functions.extend(py_unused.get("functions", []))
                unused_classes.extend(py_unused.get("classes", []))
                unused_variables.extend(py_unused.get("variables", []))
                total_dead += (len(py_unused.get("functions", [])) +
                               len(py_unused.get("classes", [])))

            # Detect commented-out code blocks (3+ consecutive commented lines
            # that contain code-like patterns)
            lines = code.split('\n')
            consec_comment = 0
            block_start = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Lines that look like commented-out code
                if (stripped.startswith('#') and
                    re.search(r'#\s*(def |class |return |if |for |while |import |from |self\.)', stripped)):
                    if consec_comment == 0:
                        block_start = i + 1
                    consec_comment += 1
                else:
                    if consec_comment >= 3:
                        commented_code_blocks.append({
                            "file": fname, "start_line": block_start,
                            "lines": consec_comment,
                        })
                        total_dead += consec_comment
                    consec_comment = 0
            if consec_comment >= 3:
                commented_code_blocks.append({
                    "file": fname, "start_line": block_start,
                    "lines": consec_comment,
                })
                total_dead += consec_comment

        total_loc = sum(len(c.split('\n')) for c in file_contents.values())
        dead_pct = (total_dead / max(1, total_loc)) * 100

        score = max(0.0, 1.0 - (dead_pct / self.THRESHOLDS["max_dead_code_pct"]))

        return {
            "total_fossils": total_fossils,
            "total_dead_code_paths": total_dead,
            "total_todos": total_todos,
            "dead_code_pct": round(dead_pct, 3),
            "fossil_breakdown": dict(fossil_breakdown),
            "unused_functions": unused_functions[:20],
            "unused_classes": unused_classes[:15],
            "unused_variables": unused_variables[:20],
            "commented_code_blocks": commented_code_blocks[:10],
            "worst_files": sorted(worst_files, key=lambda w: w["health"])[:10],
            "score": round(max(0.0, min(1.0, score)), 4),
        }

    def _detect_unused_symbols_ast(self, code: str, filename: str) -> Dict[str, List]:
        """AST-based detection of unused functions, classes, and variables in Python."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"functions": [], "classes": [], "variables": []}

        # Collect all defined names and all referenced names
        defined_funcs: Dict[str, int] = {}
        defined_classes: Dict[str, int] = {}
        assigned_vars: Dict[str, int] = {}
        all_referenced: Set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith('_'):  # skip private/dunder
                    defined_funcs[node.name] = node.lineno
            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith('_'):
                    defined_classes[node.name] = node.lineno
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Load):
                    all_referenced.add(node.id)
                elif isinstance(node.ctx, ast.Store):
                    # Track assignments (only top-level-ish)
                    assigned_vars.setdefault(node.id, getattr(node, 'lineno', 0))
            elif isinstance(node, ast.Attribute):
                # Catch attribute references like module.ClassName
                all_referenced.add(node.attr)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    all_referenced.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    all_referenced.add(node.func.attr)

        # Also scan raw text for decorator/string references the AST might miss
        for name in list(defined_funcs.keys()) + list(defined_classes.keys()):
            # Check if it appears in string context (e.g., getattr, endpoint refs)
            if re.search(rf'["\'{re.escape(name)}"\']', code):
                all_referenced.add(name)

        unused_f = [{"name": n, "file": filename, "line": ln}
                    for n, ln in defined_funcs.items() if n not in all_referenced]
        unused_c = [{"name": n, "file": filename, "line": ln}
                    for n, ln in defined_classes.items() if n not in all_referenced]

        # Unused variables: assigned but never read (exclude loop vars, common names)
        skip_names = {"_", "self", "cls", "args", "kwargs", "e", "ex", "err",
                      "i", "j", "k", "x", "y", "result", "logger"}
        unused_v = [{"name": n, "file": filename, "line": ln}
                    for n, ln in assigned_vars.items()
                    if n not in all_referenced and n not in skip_names
                    and not n.startswith('_') and n not in defined_funcs
                    and n not in defined_classes]

        return {"functions": unused_f[:15], "classes": unused_c[:10], "variables": unused_v[:15]}

    # ─── Cross-Cutting Audit Capabilities ────────────────────────────

    def _compute_file_risk_ranking(self, l0, l1, l2, l4, l5,
                                    file_contents: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Rank every audited file by composite risk score.
        Aggregates findings from L1 (complexity/smells), L2 (vulns/debt),
        L4 (dead code), and L5 (anti-patterns) per file.
        """
        file_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "smells": 0, "vulns": 0, "debt": 0, "anti_patterns": 0, "lines": 0,
        })

        # Count smells per file
        for smell in l1.get("code_smells", []):
            fname = smell.get("file", "?")
            weight = 2.0 if smell.get("severity") == "HIGH" else 1.0
            file_scores[fname]["smells"] += weight

        # Count vulns per file
        for vuln in l2.get("all_vulns", []):
            fname = vuln.get("file", "?")
            weight = 3.0 if vuln.get("severity") == "HIGH" else 1.0
            file_scores[fname]["vulns"] += weight

        # Count debt per file
        for item in l2.get("technical_debt", {}).get("items", []):
            fname = item.get("file", "?")
            file_scores[fname]["debt"] += 1

        # Count anti-patterns per file
        for ap in l5.get("all_patterns", []):
            fname = ap.get("file", "?")
            weight = 2.0 if ap.get("severity") == "HIGH" else 0.5
            file_scores[fname]["anti_patterns"] += weight

        # Add line counts
        for fp, code in file_contents.items():
            fname = Path(fp).name
            if fname in file_scores:
                file_scores[fname]["lines"] = len(code.split('\n'))

        # Compute composite risk
        rankings = []
        for fname, scores in file_scores.items():
            loc = max(1, scores["lines"])
            # Density-based scoring: issues per 100 lines
            density = ((scores["vulns"] * 3 + scores["smells"] * 2 +
                        scores["debt"] + scores["anti_patterns"] * 1.5) / loc) * 100
            risk = round(min(1.0, density / 10.0), 4)  # Normalize to 0-1
            rankings.append({
                "file": fname,
                "risk_score": risk,
                "risk_level": "CRITICAL" if risk > 0.7 else "HIGH" if risk > 0.4
                              else "MEDIUM" if risk > 0.2 else "LOW",
                "vulns": int(scores["vulns"]),
                "smells": int(scores["smells"]),
                "debt": int(scores["debt"]),
                "anti_patterns": int(scores["anti_patterns"]),
                "lines": scores["lines"],
            })

        return sorted(rankings, key=lambda r: r["risk_score"], reverse=True)

    def _detect_code_clones(self, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """
        Lightweight code clone detection using normalized line hashing.
        Finds duplicate code blocks (Type-1/Type-2 clones) across files.
        """
        # Build a hash map of normalized 5-line sliding windows
        WINDOW_SIZE = 5
        MIN_CLONE_LINES = 5
        window_locations: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        for fp, code in file_contents.items():
            if not (fp.endswith('.py') or fp.endswith('.swift') or fp.endswith('.js') or fp.endswith('.ts')):
                continue
            fname = Path(fp).name
            comment_prefix = '#' if fp.endswith('.py') else '//'
            lines = code.split('\n')
            for i in range(len(lines) - WINDOW_SIZE + 1):
                window = lines[i:i + WINDOW_SIZE]
                # Normalize: strip whitespace, ignore comments/blanks
                normalized = []
                for line in window:
                    stripped = line.strip()
                    if stripped and not stripped.startswith(comment_prefix):
                        # Replace identifiers with placeholders for Type-2 detection
                        norm = re.sub(r'\b[a-zA-Z_]\w*\b', 'ID', stripped)
                        norm = re.sub(r'\b\d+\b', 'NUM', norm)
                        normalized.append(norm)
                if len(normalized) >= 3:  # Need at least 3 non-trivial lines
                    key = '\n'.join(normalized)
                    window_locations[key].append((fname, i + 1))

        # Find clones: windows appearing in 2+ different files
        clones = []
        for key, locations in window_locations.items():
            files_involved = set(loc[0] for loc in locations)
            if len(files_involved) >= 2 and len(locations) >= 2:
                clones.append({
                    "locations": [{"file": f, "line": l} for f, l in locations[:6]],
                    "files_involved": len(files_involved),
                    "occurrences": len(locations),
                })

        # Deduplicate overlapping clones and take top results
        clones.sort(key=lambda c: c["occurrences"], reverse=True)
        unique_clones = []
        seen_files = set()
        for clone in clones[:50]:
            key = frozenset((l["file"], l["line"]) for l in clone["locations"])
            if key not in seen_files:
                seen_files.add(key)
                unique_clones.append(clone)

        # v2.5.0 — Intra-file clone detection (Type-1 duplicates within same file)
        intra_file_clones = []
        for fp, code in file_contents.items():
            if not (fp.endswith('.py') or fp.endswith('.swift') or fp.endswith('.js') or fp.endswith('.ts')):
                continue
            fname = Path(fp).name
            comment_prefix = '#' if fp.endswith('.py') else '//'
            lines = code.split('\n')
            line_hashes: Dict[str, List[int]] = defaultdict(list)
            for i in range(len(lines) - WINDOW_SIZE + 1):
                window = lines[i:i + WINDOW_SIZE]
                normalized = []
                for line in window:
                    stripped = line.strip()
                    if stripped and not stripped.startswith(comment_prefix):
                        norm = re.sub(r'\b[a-zA-Z_]\w*\b', 'ID', stripped)
                        norm = re.sub(r'\b\d+\b', 'NUM', norm)
                        normalized.append(norm)
                if len(normalized) >= 3:
                    key = '\n'.join(normalized)
                    line_hashes[key].append(i + 1)
            # Find blocks that appear 2+ times in the same file
            for key, positions in line_hashes.items():
                if len(positions) >= 2:
                    intra_file_clones.append({
                        "file": fname,
                        "positions": positions[:6],
                        "occurrences": len(positions),
                    })
        intra_file_clones.sort(key=lambda c: c["occurrences"], reverse=True)

        total_dup_blocks = sum(c["occurrences"] for c in unique_clones)
        return {
            "clone_groups": unique_clones[:15],
            "total_clone_groups": len(unique_clones),
            "total_duplicate_blocks": total_dup_blocks,
            "intra_file_clones": intra_file_clones[:10],
            "intra_file_clone_count": len(intra_file_clones),
            "duplication_risk": "HIGH" if len(unique_clones) > 20
                                else "MEDIUM" if len(unique_clones) > 5
                                else "LOW",
        }

    def _generate_remediation_plan(self, l1, l2, l4, l5,
                                    file_risks: List[Dict]) -> Dict[str, Any]:
        """
        Generate a prioritized, actionable remediation plan from audit findings.
        Groups fixes by urgency and estimates effort.
        """
        critical_actions = []
        high_actions = []
        medium_actions = []

        # Security-critical: HIGH vulns
        for vuln in l2.get("critical_vulns", []):
            rec = vuln.get("recommendation", "Review and fix")
            critical_actions.append({
                "action": f"Fix {vuln.get('type', 'vulnerability')} in {vuln.get('file', '?')}:{vuln.get('line', '?')}",
                "recommendation": rec,
                "category": "security",
            })

        # High: code smells with HIGH severity
        for smell in l1.get("code_smells", []):
            if smell.get("severity") == "HIGH":
                high_actions.append({
                    "action": f"Refactor {smell.get('smell')} in {smell.get('file', '?')}:{smell.get('line', '?')} "
                              f"(value={smell.get('value')}, threshold={smell.get('threshold')})",
                    "recommendation": f"Reduce {smell.get('smell')} below threshold {smell.get('threshold')}",
                    "category": "complexity",
                })

        # High: anti-patterns with HIGH severity
        for ap in l5.get("critical_patterns", []):
            high_actions.append({
                "action": f"Fix anti-pattern in {ap.get('file', '?')}: {ap.get('suggestion', ap.get('type', 'pattern'))}",
                "recommendation": ap.get("suggestion", "Refactor to eliminate anti-pattern"),
                "category": "anti_pattern",
            })

        # Medium: debt items
        debt = l2.get("technical_debt", {})
        debt_by_type = debt.get("by_type", {})
        for dtype, count in sorted(debt_by_type.items(), key=lambda x: x[1], reverse=True):
            if count > 0 and dtype not in ("todo_debt", "debug_print"):
                medium_actions.append({
                    "action": f"Address {count}x {dtype} findings across codebase",
                    "recommendation": self._debt_recommendation(dtype),
                    "category": "debt",
                })

        # Medium: dead code cleanup
        dead_total = l4.get("total_dead_code_paths", 0)
        if dead_total > 10:
            medium_actions.append({
                "action": f"Remove {dead_total} dead code paths identified by archeological analysis",
                "recommendation": "Delete unreachable code, commented-out blocks, and unused symbols",
                "category": "dead_code",
            })

        # Top risky files
        risky_files = [f for f in file_risks if f.get("risk_level") in ("CRITICAL", "HIGH")]

        return {
            "critical": critical_actions[:10],
            "high": high_actions[:10],
            "medium": medium_actions[:10],
            "total_actions": len(critical_actions) + len(high_actions) + len(medium_actions),
            "top_risk_files": [f["file"] for f in risky_files[:5]],
            "estimated_effort": "HIGH" if len(critical_actions) > 10
                                 else "MEDIUM" if len(critical_actions) > 3
                                 else "LOW",
        }

    def get_trend(self) -> Dict[str, Any]:
        """
        Compute audit score trend from historical audits.
        Tracks improvement/degradation over time.
        """
        if len(self.audit_history) < 2:
            return {"trend": "INSUFFICIENT_DATA", "data_points": len(self.audit_history)}

        scores = [h["score"] for h in self.audit_history]
        latest = scores[-1]
        previous = scores[-2]
        delta = latest - previous
        avg = sum(scores) / len(scores)

        # Simple linear regression for trend direction
        n = len(scores)
        x_mean = (n - 1) / 2
        y_mean = avg
        numerator = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(scores))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / max(denominator, 1e-10)

        return {
            "trend": "IMPROVING" if slope > 0.005 else "DEGRADING" if slope < -0.005 else "STABLE",
            "latest_score": latest,
            "previous_score": previous,
            "delta": round(delta, 4),
            "slope": round(slope, 6),
            "average_score": round(avg, 4),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "data_points": n,
        }

    # ─── Cross-cutting Analyses (v2.2) ───────────────────────────────

    def _analyze_import_hygiene(self, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """Analyze import quality: star imports, unused imports, circular risk."""
        star_imports = []
        duplicate_imports = []
        heavy_importers = []

        for fp, code in file_contents.items():
            if not fp.endswith('.py'):
                continue
            fname = Path(fp).name
            imports = set()
            star_count = 0
            for line in code.split('\n'):
                stripped = line.strip()
                if stripped.startswith('from ') and 'import *' in stripped:
                    star_imports.append({"file": fname, "import": stripped[:80]})
                    star_count += 1
                elif stripped.startswith('import ') or stripped.startswith('from '):
                    mod = stripped.split()[1] if len(stripped.split()) > 1 else ''
                    if mod in imports:
                        duplicate_imports.append({"file": fname, "module": mod})
                    imports.add(mod)
            if len(imports) > 30:
                heavy_importers.append({"file": fname, "import_count": len(imports)})

        return {
            "star_imports": star_imports[:15],
            "star_import_count": len(star_imports),
            "duplicate_imports": duplicate_imports[:10],
            "heavy_importers": sorted(heavy_importers,
                                       key=lambda h: h["import_count"], reverse=True)[:10],
            "hygiene_score": max(0.0, 1.0 - len(star_imports) * 0.05
                                 - len(duplicate_imports) * 0.02),
        }

    def _build_complexity_heatmap(self, l1: Dict[str, Any],
                                   file_contents: Dict[str, str]) -> Dict[str, Any]:
        """Build a complexity heatmap: top files by combined cyclomatic + cognitive load."""
        file_complexity: Dict[str, Dict[str, Any]] = {}

        for fp, code in file_contents.items():
            fname = Path(fp).name
            analysis = self.analyzer.full_analysis(code, fp)
            complexity = analysis.get("complexity", {})
            funcs = complexity.get("functions", [])
            if not funcs:
                continue
            total_cc = sum(f.get("cyclomatic_complexity", 0) for f in funcs)
            total_cog = sum(f.get("cognitive_complexity", 0) for f in funcs)
            max_cc = max((f.get("cyclomatic_complexity", 0) for f in funcs), default=0)
            lines = len(code.split('\n'))
            density = total_cc / max(1, lines) * 100

            file_complexity[fname] = {
                "file": fname,
                "total_cyclomatic": total_cc,
                "total_cognitive": total_cog,
                "max_cyclomatic": max_cc,
                "function_count": len(funcs),
                "lines": lines,
                "density_per_100_loc": round(density, 2),
                "heat": "CRITICAL" if density > 5 else "HIGH" if density > 2
                        else "MEDIUM" if density > 1 else "LOW",
            }

        ranked = sorted(file_complexity.values(),
                        key=lambda f: f["density_per_100_loc"], reverse=True)
        heat_dist = defaultdict(int)
        for f in ranked:
            heat_dist[f["heat"]] += 1

        return {
            "hotspots": ranked[:15],
            "heat_distribution": dict(heat_dist),
            "total_files_analyzed": len(ranked),
        }

    def _compute_delta(self, current_score: float,
                       l2: Dict[str, Any], l5: Dict[str, Any]) -> Dict[str, Any]:
        """Compute improvement delta from last audit for tracking progress."""
        if not self.audit_history:
            return {"status": "FIRST_AUDIT", "previous_score": None, "delta": 0.0}

        prev = self.audit_history[-1]
        prev_score = prev.get("score", 0.0)
        delta = round(current_score - prev_score, 4)
        direction = "IMPROVED" if delta > 0.005 else "DEGRADED" if delta < -0.005 else "STABLE"

        return {
            "status": direction,
            "previous_score": prev_score,
            "current_score": current_score,
            "delta": delta,
            "previous_timestamp": prev.get("timestamp", "unknown"),
        }

    def _analyze_architecture_coupling(self, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """Analyze module coupling & cohesion for architecture health.

        Metrics:
        - Afferent coupling (Ca): how many modules depend on this module
        - Efferent coupling (Ce): how many modules this module depends on
        - Instability (I): Ce / (Ca + Ce) — 0=stable, 1=unstable
        - Abstractness (A): ratio of abstract classes/interfaces
        - Distance from Main Sequence: |A + I - 1| — 0=ideal balance

        Returns architecture health report with coupling matrix and risk zones.
        """
        # Build import graph for Python files
        modules: Dict[str, set] = {}  # module -> set of imported modules
        module_lines: Dict[str, int] = {}
        module_classes: Dict[str, int] = {}
        module_abstracts: Dict[str, int] = {}
        import_re = re.compile(
            r'^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))', re.MULTILINE
        )
        abstract_re = re.compile(r'class\s+\w+\s*\([^)]*(?:ABC|Abstract|Base|Interface)', re.MULTILINE)
        class_re = re.compile(r'^class\s+\w+', re.MULTILINE)

        py_files = {fp: code for fp, code in file_contents.items() if fp.endswith('.py')}

        # Map filenames to module names
        file_to_mod = {}
        for fp in py_files:
            mod = Path(fp).stem
            file_to_mod[fp] = mod

        all_local_mods = set(file_to_mod.values())

        for fp, code in py_files.items():
            mod = file_to_mod[fp]
            imports = set()
            for m in import_re.finditer(code):
                imported = m.group(1) or m.group(2)
                if imported:
                    # Only track local module references
                    root = imported.split('.')[0]
                    if root in all_local_mods:
                        imports.add(root)
            imports.discard(mod)  # Remove self-imports
            modules[mod] = imports
            module_lines[mod] = len(code.split('\n'))
            module_classes[mod] = len(class_re.findall(code))
            module_abstracts[mod] = len(abstract_re.findall(code))

        # Compute coupling metrics
        afferent: Dict[str, int] = defaultdict(int)   # Ca: who depends on me
        efferent: Dict[str, int] = defaultdict(int)    # Ce: who I depend on

        for mod, deps in modules.items():
            efferent[mod] = len(deps)
            for dep in deps:
                afferent[dep] += 1

        # Compute per-module metrics
        module_metrics = []
        zones = {"zone_of_pain": [], "zone_of_uselessness": [], "main_sequence": []}

        for mod in modules:
            ca = afferent.get(mod, 0)
            ce = efferent.get(mod, 0)
            instability = ce / max(1, ca + ce)
            total_classes = module_classes.get(mod, 0)
            abstract_classes = module_abstracts.get(mod, 0)
            abstractness = abstract_classes / max(1, total_classes) if total_classes > 0 else 0.0
            distance = abs(abstractness + instability - 1.0)

            metric = {
                "module": mod,
                "afferent_coupling": ca,
                "efferent_coupling": ce,
                "instability": round(instability, 3),
                "abstractness": round(abstractness, 3),
                "distance_from_main_seq": round(distance, 3),
                "lines": module_lines.get(mod, 0),
            }
            module_metrics.append(metric)

            # Classify zones
            if abstractness < 0.2 and instability < 0.3 and ca > 3:
                zones["zone_of_pain"].append(mod)  # Concrete & stable but heavily depended on
            elif abstractness > 0.7 and instability > 0.7:
                zones["zone_of_uselessness"].append(mod)  # Abstract & unstable
            elif distance < 0.3:
                zones["main_sequence"].append(mod)  # Balanced

        # Sort by distance from main sequence (worst first)
        module_metrics.sort(key=lambda m: m["distance_from_main_seq"], reverse=True)

        avg_distance = (sum(m["distance_from_main_seq"] for m in module_metrics)
                        / max(1, len(module_metrics)))
        avg_coupling = (sum(m["efferent_coupling"] for m in module_metrics)
                        / max(1, len(module_metrics)))

        # Hub detection: modules with high total coupling
        hubs = [m for m in module_metrics
                if m["afferent_coupling"] + m["efferent_coupling"] > 8]

        coupling_score = max(0.0, 1.0 - avg_distance - min(0.2, avg_coupling / 50))

        return {
            "total_modules": len(modules),
            "avg_distance_from_main_seq": round(avg_distance, 4),
            "avg_efferent_coupling": round(avg_coupling, 2),
            "coupling_score": round(max(0.0, min(1.0, coupling_score)), 4),
            "zone_of_pain": zones["zone_of_pain"][:10],
            "zone_of_uselessness": zones["zone_of_uselessness"][:10],
            "main_sequence_modules": len(zones["main_sequence"]),
            "hub_modules": sorted(hubs, key=lambda h: h["afferent_coupling"]
                                  + h["efferent_coupling"], reverse=True)[:10],
            "module_metrics": module_metrics[:20],
        }

    def _estimate_test_coverage(self, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """Estimate test coverage by analyzing test file presence and thoroughness.

        Heuristic approach (no actual execution):
        - Counts test files vs source files
        - Checks assertion density in test files
        - Identifies untested modules (source files without corresponding test files)
        """
        test_files = []
        source_files = []
        test_assertions = 0
        test_functions = 0

        test_re = re.compile(r'^\s*def\s+test_\w+', re.MULTILINE)
        assert_re = re.compile(r'(?:self\.assert\w+|assert\s+|pytest\.\w+)', re.MULTILINE)

        for fp, code in file_contents.items():
            if not fp.endswith('.py'):
                continue
            fname = Path(fp).name
            is_test = (fname.startswith('test_') or fname.endswith('_test.py')
                       or '/tests/' in fp or 'test' in fname.lower())

            if is_test:
                test_files.append(fname)
                test_functions += len(test_re.findall(code))
                test_assertions += len(assert_re.findall(code))
            else:
                source_files.append(fname)

        # Identify untested modules (no matching test file)
        tested_names = set()
        for tf in test_files:
            # Extract module name from test file name
            name = tf.replace('test_', '').replace('_test.py', '.py')
            tested_names.add(name)
            # Also try without prefix
            if tf.startswith('test_'):
                tested_names.add(tf[5:])

        untested = [sf for sf in source_files if sf not in tested_names
                    and not sf.startswith('__')]

        # Coverage ratio: test files / source files (rough heuristic)
        ratio = len(test_files) / max(1, len(source_files))
        assertion_density = test_assertions / max(1, test_functions)

        # Score: balanced test-to-source ratio + assertion quality
        coverage_score = min(1.0, ratio * 2.0)  # 50% test ratio = perfect
        quality_bonus = min(0.2, assertion_density * 0.05)
        score = min(1.0, coverage_score * 0.7 + quality_bonus + 0.1)

        return {
            "test_files": len(test_files),
            "source_files": len(source_files),
            "test_to_source_ratio": round(ratio, 3),
            "test_functions": test_functions,
            "test_assertions": test_assertions,
            "assertion_density": round(assertion_density, 2),
            "untested_modules": untested[:20],
            "untested_count": len(untested),
            "estimated_coverage_pct": round(min(100, ratio * 100), 1),
            "coverage_score": round(max(0.0, min(1.0, score)), 4),
        }

    def _analyze_api_surface(self, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """Analyze public API surface: endpoints, exported functions, public classes.

        Measures API sprawl and consistency.
        """
        endpoints = []
        public_functions = 0
        private_functions = 0
        public_classes = 0
        god_classes = []  # Classes with too many methods

        endpoint_re = re.compile(
            r'@(?:app|router)\.\s*(?:get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)',
            re.MULTILINE
        )
        func_re = re.compile(r'^\s*def\s+(\w+)', re.MULTILINE)
        class_re = re.compile(r'^class\s+(\w+)', re.MULTILINE)
        method_re = re.compile(r'^\s+def\s+(\w+)', re.MULTILINE)

        for fp, code in file_contents.items():
            if not fp.endswith('.py'):
                continue
            fname = Path(fp).name

            # Count API endpoints
            for m in endpoint_re.finditer(code):
                endpoints.append({"file": fname, "path": m.group(1)})

            # Count public vs private functions
            for m in func_re.finditer(code):
                fn_name = m.group(1)
                if fn_name.startswith('_'):
                    private_functions += 1
                else:
                    public_functions += 1

            # Count classes and detect god classes
            for m in class_re.finditer(code):
                cls_name = m.group(1)
                public_classes += 1

                # Count methods in this class (rough heuristic)
                cls_start = m.start()
                next_class = class_re.search(code[m.end():])
                cls_end = m.end() + next_class.start() if next_class else len(code)
                cls_code = code[cls_start:cls_end]
                methods = method_re.findall(cls_code)
                if len(methods) > 30:
                    god_classes.append({
                        "file": fname,
                        "class": cls_name,
                        "method_count": len(methods),
                    })

        # Encapsulation ratio: private / total functions
        total_funcs = public_functions + private_functions
        encapsulation = private_functions / max(1, total_funcs)

        # API surface score
        endpoint_density = len(endpoints) / max(1, len(file_contents))
        god_class_penalty = len(god_classes) * 0.05
        score = min(1.0, 0.5 + encapsulation * 0.3 + min(0.2, endpoint_density * 0.5))
        score = max(0.0, score - god_class_penalty)

        return {
            "total_endpoints": len(endpoints),
            "public_functions": public_functions,
            "private_functions": private_functions,
            "encapsulation_ratio": round(encapsulation, 3),
            "public_classes": public_classes,
            "god_classes": sorted(god_classes, key=lambda g: g["method_count"],
                                   reverse=True)[:10],
            "god_class_count": len(god_classes),
            "api_surface_score": round(max(0.0, min(1.0, score)), 4),
        }

    def _layer5_anti_pattern_detection(self, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """L5: Anti-pattern detection using CodeOptimizer."""
        all_anti_patterns = []
        health_counts = defaultdict(int)

        for fp, code in file_contents.items():
            analysis = self.analyzer.full_analysis(code, fp)
            optimization = self.optimizer.analyze_and_suggest(analysis)

            for suggestion in optimization.get("suggestions", []):
                suggestion["file"] = Path(fp).name
                all_anti_patterns.append(suggestion)

            health = optimization.get("overall_health", "UNKNOWN")
            health_counts[health] += 1

        by_severity = defaultdict(int)
        for ap in all_anti_patterns:
            by_severity[ap.get("severity", "UNKNOWN")] += 1

        critical_count = by_severity.get("HIGH", 0)
        medium_count = by_severity.get("MEDIUM", 0)
        total_files = max(1, len(file_contents))
        # Normalize per-file: a few anti-patterns per file is normal
        high_ratio = critical_count / total_files
        med_ratio = medium_count / total_files
        score = max(0.0, 1.0 - (high_ratio * 0.25) - (med_ratio * 0.05))

        return {
            "total_anti_patterns": len(all_anti_patterns),
            "by_severity": dict(by_severity),
            "health_distribution": dict(health_counts),
            "critical_patterns": [ap for ap in all_anti_patterns if ap.get("severity") == "HIGH"][:10],
            "all_patterns": all_anti_patterns[:20],
            "score": round(max(0.0, min(1.0, score)), 4),
        }

    def _layer6_refactoring_opportunities(self, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """L6: Refactoring opportunity analysis."""
        total_suggestions = 0
        by_type: Dict[str, int] = defaultdict(int)
        by_priority: Dict[str, int] = defaultdict(int)
        file_reports = []

        for fp, code in file_contents.items():
            refactor = self.refactorer.analyze(code)
            count = refactor.get("total_suggestions", 0)
            total_suggestions += count

            for s in refactor.get("suggestions", []):
                by_type[s.get("type", "unknown")] += 1
                by_priority[s.get("priority", "unknown")] += 1

            if count > 0:
                file_reports.append({
                    "file": Path(fp).name,
                    "suggestions": count,
                    "health": refactor.get("code_health", 1.0),
                })

        # Score: fewer suggestions = higher score, normalized per file
        per_file = total_suggestions / max(1, len(file_contents))
        score = max(0.0, 1.0 - (per_file * 0.035))

        return {
            "total_refactoring_suggestions": total_suggestions,
            "by_type": dict(by_type),
            "by_priority": dict(by_priority),
            "files_needing_refactoring": sorted(file_reports, key=lambda f: f["health"])[:10],
            "score": round(score, 4),
        }

    def _layer7_sacred_alignment(self, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """L7: Sacred constant alignment and φ-ratio structural analysis.

        Performs real structural phi-ratio analysis:
          1. Sacred constant reference counting (GOD_CODE, PHI, TAU, etc.)
          2. Function-to-code ratio vs PHI (ideal: functions are 1/φ of total lines)
          3. Comment-to-code ratio vs TAU (ideal: comments are TAU fraction of code)
          4. Import-to-module ratio (structural density)
          5. Per-file phi-balance scoring (how close structure approaches golden proportions)
        """
        total_sacred_refs = 0
        total_lines = 0
        total_functions = 0
        total_classes = 0
        total_comments = 0
        total_imports = 0
        total_blank = 0
        phi_alignments = []
        god_code_resonances = []
        per_file_phi = []

        for fp, code in file_contents.items():
            lines = code.split('\n')
            line_count = len(lines)
            total_lines += line_count

            analysis = self.analyzer.full_analysis(code, fp)
            sacred = analysis.get("sacred_alignment", {})
            total_sacred_refs += sacred.get("sacred_constant_refs", 0)

            phi_val = sacred.get("phi_alignment", 0)
            if phi_val:
                phi_alignments.append(phi_val)
            god_val = sacred.get("god_code_resonance", 0)
            if god_val:
                god_code_resonances.append(god_val)

            # Structural counting for phi-ratio analysis
            func_count = 0
            class_count = 0
            comment_count = 0
            import_count = 0
            blank_count = 0
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    blank_count += 1
                elif stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
                    comment_count += 1
                elif stripped.startswith('def ') or stripped.startswith('func ') or stripped.startswith('function '):
                    func_count += 1
                elif stripped.startswith('class ') or stripped.startswith('struct ') or stripped.startswith('enum '):
                    class_count += 1
                elif stripped.startswith('import ') or stripped.startswith('from ') and 'import' in stripped:
                    import_count += 1

            total_functions += func_count
            total_classes += class_count
            total_comments += comment_count
            total_imports += import_count
            total_blank += blank_count

            # Per-file phi-balance: how close is function/total ratio to 1/PHI?
            code_lines = line_count - blank_count - comment_count
            if code_lines > 10:
                func_ratio = func_count / code_lines
                target_ratio = 1.0 / PHI  # ~0.618
                # Normalize deviation: 0 = perfect PHI alignment, 1 = maximally off
                phi_deviation = abs(func_ratio - target_ratio) / target_ratio
                phi_score = max(0.0, 1.0 - phi_deviation)
                per_file_phi.append({
                    "file": Path(fp).name,
                    "func_ratio": round(func_ratio, 4),
                    "phi_target": round(target_ratio, 4),
                    "phi_score": round(phi_score, 4),
                })

        avg_phi = sum(phi_alignments) / max(1, len(phi_alignments)) if phi_alignments else 0
        avg_god = sum(god_code_resonances) / max(1, len(god_code_resonances)) if god_code_resonances else 0
        sacred_density = total_sacred_refs / max(1, total_lines) * 100

        # Structural phi-ratio metrics
        code_lines_total = max(1, total_lines - total_blank - total_comments)
        func_code_ratio = total_functions / code_lines_total
        comment_code_ratio = total_comments / code_lines_total
        import_density = total_imports / max(1, len(file_contents))

        # How close is function-to-code ratio to 1/PHI (~0.618)?
        func_phi_deviation = abs(func_code_ratio - (1.0 / PHI))
        func_phi_score = max(0.0, 1.0 - func_phi_deviation / (1.0 / PHI))

        # How close is comment-to-code ratio to TAU (~0.618)?
        comment_tau_deviation = abs(comment_code_ratio - TAU)
        comment_tau_score = max(0.0, 1.0 - comment_tau_deviation / max(0.01, TAU))

        # Average per-file phi balance
        avg_per_file_phi = (sum(f["phi_score"] for f in per_file_phi) /
                            max(1, len(per_file_phi))) if per_file_phi else 0.0

        # Composite sacred alignment score (weighted blend)
        score = (
            avg_phi * 0.20              # Sacred constant phi alignment from analyzer
            + avg_god * 0.15            # GOD_CODE resonance
            + min(sacred_density * 0.05, 0.15)  # Sacred reference density (capped)
            + func_phi_score * 0.20     # Function-to-code ratio vs 1/PHI
            + comment_tau_score * 0.10  # Comment-to-code ratio vs TAU
            + avg_per_file_phi * 0.20   # Per-file structural balance
        )
        score = max(0.0, min(1.0, score))

        return {
            "total_sacred_references": total_sacred_refs,
            "sacred_density_pct": round(sacred_density, 4),
            "avg_phi_alignment": round(avg_phi, 6),
            "avg_god_code_resonance": round(avg_god, 6),
            "phi_golden_ratio": PHI,
            "god_code_constant": GOD_CODE,
            # Structural phi-ratio analysis (NEW)
            "structural_analysis": {
                "total_functions": total_functions,
                "total_classes": total_classes,
                "total_comments": total_comments,
                "total_imports": total_imports,
                "total_blank_lines": total_blank,
                "code_lines": code_lines_total,
                "func_code_ratio": round(func_code_ratio, 6),
                "func_phi_target": round(1.0 / PHI, 6),
                "func_phi_score": round(func_phi_score, 4),
                "comment_code_ratio": round(comment_code_ratio, 6),
                "comment_tau_target": round(TAU, 6),
                "comment_tau_score": round(comment_tau_score, 4),
                "import_density_per_file": round(import_density, 4),
            },
            "per_file_phi_balance": per_file_phi[:20],  # Top 20 files
            "avg_per_file_phi": round(avg_per_file_phi, 4),
            "score": round(score, 4),
        }

    def _layer8_auto_remediation(self, file_contents: Dict[str, str],
                                  apply: bool = False) -> Dict[str, Any]:
        """L8: Auto-remediation — identify safe fixes, optionally apply them."""
        total_fixable = 0
        total_applied = 0
        fix_details = []

        for fp, code in file_contents.items():
            fixed_code, fix_log = self.auto_fix.apply_all_safe(code)
            changes = len(fix_log)
            total_fixable += changes

            if changes > 0:
                fix_details.append({
                    "file": Path(fp).name,
                    "fixes": fix_log,
                    "fix_count": changes,
                })

                if apply:
                    try:
                        Path(fp).write_text(fixed_code)
                        total_applied += changes
                        self._trail_event("AUTO_FIX_APPLIED", {
                            "file": fp, "fixes": changes
                        })
                    except Exception as e:
                        logger.warning(f"[APP_AUDIT] Could not write fix to {fp}: {e}")

        score = 1.0 if total_fixable == 0 else (0.8 if not apply else 1.0)

        return {
            "total_fixable_issues": total_fixable,
            "total_applied": total_applied,
            "auto_remediation_active": apply,
            "fix_details": fix_details[:15],
            "score": round(score, 4),
        }

    def _layer9_verdict(self, layer_scores: Dict[str, float]) -> Dict[str, Any]:
        """L9: Compute composite score and issue certification verdict."""
        composite = sum(
            layer_scores.get(layer, 0.5) * weight
            for layer, weight in self.LAYER_WEIGHTS.items()
        )
        composite = round(max(0.0, min(1.0, composite)), 4)

        verdict = self._score_to_verdict(composite)

        # Certification — expanded failure conditions (v2.5.0 enhanced)
        failures = []
        if layer_scores.get("security", 1.0) < 0.5:
            failures.append("SECURITY_CRITICAL")
        if layer_scores.get("complexity", 1.0) < 0.3:
            failures.append("COMPLEXITY_EXCESSIVE")
        if layer_scores.get("dependencies", 1.0) < 0.4:
            failures.append("DEPENDENCY_INTEGRITY")
        if layer_scores.get("dead_code", 1.0) < 0.4:
            failures.append("DEAD_CODE_EXCESSIVE")
        if layer_scores.get("structural", 1.0) < 0.3:
            failures.append("STRUCTURAL_DEGRADATION")
        if layer_scores.get("quality", 1.0) < 0.3:
            failures.append("QUALITY_BELOW_STANDARD")
        # v2.5.0 — New failure conditions
        if layer_scores.get("anti_patterns", 1.0) < 0.3:
            failures.append("ANTI_PATTERN_PROLIFERATION")
        if layer_scores.get("sacred_alignment", 1.0) < 0.1:
            failures.append("SACRED_ALIGNMENT_LOST")

        # v2.5.0 — Tiered certification levels
        if failures:
            certification = "NOT_CERTIFIED"
        elif composite >= 0.85:
            certification = "CERTIFIED_EXEMPLARY"
        elif composite >= 0.70:
            certification = "CERTIFIED_GOLD"
        elif composite >= 0.60:
            certification = "CERTIFIED"
        else:
            certification = "NOT_CERTIFIED"

        return {
            "composite_score": composite,
            "verdict": verdict,
            "certification": certification,
            "failures": failures,
            "layer_scores": {k: round(v, 4) for k, v in layer_scores.items()},
            "phi_harmonic": round(composite * PHI, 6),
            "god_code_alignment": round(composite * GOD_CODE / 1000, 6),
        }

    # ─── Utility Methods ─────────────────────────────────────────────

    def _collect_files(self, workspace: Path,
                       target_files: List[str] = None) -> List[str]:
        """Collect auditable source files from workspace."""
        if target_files:
            return [f for f in target_files if os.path.isfile(f)]

        extensions = {".py", ".swift", ".js", ".ts", ".rs", ".go",
                      ".java", ".c", ".cpp", ".rb", ".kt", ".sh",
                      ".sql", ".jsx", ".tsx", ".m", ".h"}
        skip_dirs = {"__pycache__", ".git", ".venv", "node_modules",
                     ".build", "build", "dist", ".tox", ".mypy_cache",
                     ".eggs", "htmlcov", "__pypackages__"}
        files = []
        for ext in extensions:
            for f in workspace.rglob(f"*{ext}"):
                if any(sd in f.parts for sd in skip_dirs):
                    continue
                if f.name.startswith('.'):
                    continue
                files.append(str(f))
        return sorted(files)[:200]  # cap at 200 files

    def _score_to_verdict(self, score: float) -> str:
        """Convert a numeric score to a human-readable verdict."""
        if score >= 0.90:
            return "EXEMPLARY"
        elif score >= 0.75:
            return "HEALTHY"
        elif score >= 0.60:
            return "ACCEPTABLE"
        elif score >= 0.40:
            return "NEEDS_ATTENTION"
        elif score >= 0.20:
            return "AT_RISK"
        else:
            return "CRITICAL"

    def _trail_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event to the audit trail."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            **data,
        }
        self._audit_trail.append(entry)
        # Persist to JSONL
        trail_path = Path(__file__).parent / ".l104_app_audit_trail.jsonl"
        try:
            with open(trail_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception:
            pass

    def get_audit_trail(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent audit trail events."""
        return self._audit_trail[-limit:]

    def get_audit_history(self) -> List[Dict[str, Any]]:
        """Return historical audit summaries."""
        return self.audit_history

    def _knowledge_context(self, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """
        Pull knowledge references from L104 CodeEngine subsystems to enrich audit.
        Queries LanguageKnowledge, CodeAnalyzer patterns, AutoFixEngine catalog,
        and CodeArcheologist fossils to build an intelligence overlay.
        """
        # Language intelligence: which paradigms and ecosystems are present
        detected_langs = set()
        paradigms_used = set()
        for fp, code in file_contents.items():
            lang = LanguageKnowledge.detect_language(code, fp)
            detected_langs.add(lang)
            lang_info = LanguageKnowledge.LANGUAGES.get(lang, {})
            paradigms_used.update(lang_info.get("paradigms", []))

        # Available auto-fix catalog reference
        fix_catalog_size = len(AutoFixEngine.FIX_CATALOG)
        fixes_applied_total = self.auto_fix.fixes_applied

        # Security pattern coverage from CodeAnalyzer
        sec_pattern_count = sum(len(v) for v in CodeAnalyzer.SECURITY_PATTERNS.values())
        sec_categories = list(CodeAnalyzer.SECURITY_PATTERNS.keys())

        # Design pattern knowledge base
        design_patterns = list(CodeAnalyzer.DESIGN_PATTERNS.keys())

        # Archeological fossil categories known
        fossil_types = list(set(
            f.get("type", "unknown")
            for fp, code in list(file_contents.items())[:5]
            for f in self.archeologist.excavate(code).get("fossils", [])
        ))

        return {
            "detected_languages": sorted(detected_langs),
            "paradigms_present": sorted(paradigms_used),
            "languages_known": len(LanguageKnowledge.LANGUAGES),
            "security_patterns_available": sec_pattern_count,
            "security_categories": sec_categories,
            "design_patterns_known": design_patterns,
            "auto_fix_catalog_size": fix_catalog_size,
            "total_fixes_applied_lifetime": fixes_applied_total,
            "debt_patterns_active": len(self.DEBT_PATTERNS),
            "fossil_types_detected": fossil_types,
        }

    def status(self) -> Dict[str, Any]:
        """Return audit engine status."""
        return {
            "version": self.AUDIT_VERSION,
            "audits_performed": self.audit_count,
            "history_entries": len(self.audit_history),
            "trail_entries": len(self._audit_trail),
            "thresholds": self.THRESHOLDS,
            "layer_weights": self.LAYER_WEIGHTS,
        }

    def quantum_audit_score(self, audit_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quantum holistic audit scoring using Qiskit 2.3.0.
        Encodes multi-layer audit scores into a 4-qubit quantum state with
        Bell-pair entanglement between coupled layers, then computes a
        quantum composite score via von Neumann entropy and Born measurement.
        """
        # Extract layer scores from audit result
        layers = audit_result.get("layers", {})
        scores = []
        layer_names = []
        for key in ["L0_structural_census", "L1_complexity_quality", "L2_security_scan",
                     "L3_dependency_topology", "L4_dead_code_archaeology",
                     "L5_anti_pattern_detection", "L6_refactoring_opportunities",
                     "L7_sacred_alignment", "L8_auto_remediation", "L9_verdict_certification"]:
            layer = layers.get(key, {})
            score = layer.get("score", layer.get("health_score", layer.get("composite_score", 0.5)))
            if isinstance(score, (int, float)):
                scores.append(max(0.01, min(float(score), 1.0)))
            else:
                scores.append(0.5)
            layer_names.append(key)

        if not scores:
            scores = [0.5] * 10
            layer_names = [f"L{i}" for i in range(10)]

        if not QISKIT_AVAILABLE:
            composite = sum(s * PHI ** (i % 3) for i, s in enumerate(scores)) / sum(PHI ** (i % 3) for i in range(len(scores)))
            return {
                "quantum": False,
                "backend": "classical_phi_weighted",
                "composite_score": round(composite, 6),
                "layer_scores": dict(zip(layer_names, [round(s, 4) for s in scores])),
                "verdict": "CERTIFIED" if composite > 0.8 else "CONDITIONAL" if composite > 0.6 else "FAILED",
            }

        try:
            # 4-qubit system: encode top 10 scores → 16 amplitudes
            n_qubits = 4
            n_states = 16
            amps = [0.0] * n_states
            for i, s in enumerate(scores[:n_states]):
                amps[i] = s * PHI
            # Fill remaining with sacred constants
            for i in range(len(scores), n_states):
                amps[i] = ALPHA_FINE * (i + 1)
            norm = math.sqrt(sum(a * a for a in amps))
            amps = [a / norm for a in amps] if norm > 1e-12 else [1.0 / math.sqrt(n_states)] * n_states

            sv = Statevector(amps)

            # Bell-pair entanglement between coupled audit layers
            qc = QuantumCircuit(n_qubits)
            qc.h(0)
            qc.cx(0, 1)  # Security-Complexity coupling
            qc.h(2)
            qc.cx(2, 3)  # Archaeology-Refactoring coupling

            # Cross-pair entanglement
            qc.cx(1, 2)

            # Audit score phase encoding
            for i, s in enumerate(scores[:n_qubits]):
                qc.ry(s * math.pi * PHI, i)
                qc.rz(GOD_CODE / 1000 * math.pi / (i + 1), i)

            evolved = sv.evolve(Operator(qc))
            dm = DensityMatrix(evolved)

            # Full entropy
            full_entropy = float(q_entropy(dm, base=2))

            # Pairwise entanglement entropies
            rho_01 = partial_trace(dm, [2, 3])
            rho_23 = partial_trace(dm, [0, 1])
            ent_01 = float(q_entropy(rho_01, base=2))
            ent_23 = float(q_entropy(rho_23, base=2))

            probs = evolved.probabilities()
            born_composite = sum(p * (i + 1) / n_states for i, p in enumerate(probs))

            # Composite: Born score weighted by entanglement coherence
            entanglement_coherence = 1.0 - full_entropy / n_qubits
            composite = (born_composite * PHI + entanglement_coherence * TAU) / (PHI + TAU)
            composite = max(0.0, min(1.0, composite))

            return {
                "quantum": True,
                "backend": "Qiskit 2.3.0 Bell-Entangled Audit Scoring",
                "qubits": n_qubits,
                "composite_score": round(composite, 6),
                "born_composite": round(born_composite, 6),
                "entanglement_coherence": round(entanglement_coherence, 6),
                "full_entropy": round(full_entropy, 6),
                "pair_entropies": {
                    "security_complexity": round(ent_01, 6),
                    "archaeology_refactoring": round(ent_23, 6),
                },
                "layer_scores": dict(zip(layer_names, [round(s, 4) for s in scores])),
                "circuit_depth": qc.depth(),
                "verdict": "CERTIFIED" if composite > 0.8 else "CONDITIONAL" if composite > 0.6 else "FAILED",
                "god_code_alignment": round(composite * GOD_CODE / 100, 4),
            }
        except Exception as e:
            return {"quantum": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4J: TYPE FLOW ANALYZER — Static Type Inference & Flow Tracking (v3.1.0)
#   Infers variable types through assignments, returns, and control flow.
#   Detects type confusion, narrowing opportunities, and generates type stubs.
# ═══════════════════════════════════════════════════════════════════════════════



class SecurityThreatModeler:
    """
    v6.0.0 — Advanced threat modeling engine using STRIDE + DREAD frameworks.

    Capabilities:
      • STRIDE classification: Spoofing, Tampering, Repudiation, Info Disclosure,
        DoS, Elevation of Privilege — per function/class
      • DREAD risk scoring: Damage, Reproducibility, Exploitability, Affected Users, Discoverability
      • Attack surface quantification: entry points, data flows, trust boundaries
      • Zero-trust pattern verification: auth checks, input validation, output encoding
      • Supply chain risk: dependency vulnerability density estimation
      • Secrets detection: API keys, tokens, credentials in source
    """

    STRIDE_CATEGORIES = {
        "spoofing": {
            "patterns": [r"authenticate|login|session|token|identity|verify_user|jwt|oauth"],
            "description": "Can an attacker pretend to be something/someone else?",
        },
        "tampering": {
            "patterns": [r"write|modify|update|delete|patch|put|truncate|alter"],
            "description": "Can data be maliciously modified?",
        },
        "repudiation": {
            "patterns": [r"log|audit|trace|record|journal|history|event_store"],
            "description": "Can actions be denied without proof?",
        },
        "information_disclosure": {
            "patterns": [r"password|secret|key|token|credential|api_key|private|ssn|credit_card"],
            "description": "Can sensitive information be exposed?",
        },
        "denial_of_service": {
            "patterns": [r"while\s+True|sleep|timeout|retry|queue|rate_limit|throttle"],
            "description": "Can the service be disrupted?",
        },
        "elevation_of_privilege": {
            "patterns": [r"admin|root|sudo|superuser|privilege|role|permission|grant|authorize"],
            "description": "Can an attacker gain elevated access?",
        },
    }

    SECRET_PATTERNS = [
        (r'(?:api[_-]?key|apikey)\s*=\s*["\'][A-Za-z0-9]{16,}["\']', "API_KEY"),
        (r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']{4,}["\']', "PASSWORD"),
        (r'(?:secret|token)\s*=\s*["\'][A-Za-z0-9+/=]{16,}["\']', "SECRET_TOKEN"),
        (r'(?:aws_access_key_id)\s*=\s*["\']AKIA[A-Z0-9]{16}["\']', "AWS_KEY"),
        (r'ghp_[A-Za-z0-9]{36}', "GITHUB_TOKEN"),
        (r'sk-[A-Za-z0-9]{32,}', "OPENAI_KEY"),
        (r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----', "PRIVATE_KEY"),
        (r'(?:mongodb|postgres|mysql)://[^"\s]+:[^"\s]+@', "DATABASE_URI"),
        # v6.1.0 — Extended secret detection
        (r'https://hooks\.slack\.com/services/T[A-Z0-9]{8,}/B[A-Z0-9]{8,}/[A-Za-z0-9]{20,}', "SLACK_WEBHOOK"),
        (r'sk_live_[A-Za-z0-9]{24,}', "STRIPE_KEY"),
        (r'AC[a-f0-9]{32}', "TWILIO_SID"),
        (r'AIzaSy[A-Za-z0-9_-]{33}', "FIREBASE_KEY"),
        (r'eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}', "JWT_TOKEN"),
        (r'SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}', "SENDGRID_KEY"),
        (r'key-[A-Za-z0-9]{32}', "MAILGUN_KEY"),
        (r'xox[bpsar]-[A-Za-z0-9-]{10,}', "SLACK_TOKEN"),
        (r'(?:secret|key|token|password)\s*=\s*["\'][0-9a-fA-F]{32,}["\']', "HEX_SECRET"),
        (r'(?:secret|key|token)\s*=\s*["\'][A-Za-z0-9+/]{40,}={0,2}["\']', "BASE64_SECRET"),
    ]

    ZERO_TRUST_CHECKS = [
        ("input_validation", r"validate|sanitize|clean|escape|strip|bleach|markupsafe"),
        ("output_encoding", r"html\.escape|markupsafe|jinja.*autoescape|cgi\.escape"),
        ("auth_required", r"@login_required|@auth|@requires_auth|@jwt_required|@permission"),
        ("rate_limiting", r"rate_limit|throttle|RateLimiter|slowapi"),
        ("encryption", r"encrypt|AES|RSA|fernet|nacl|bcrypt|argon2|scrypt"),
        ("input_length_check", r"max_length|maxlength|len\(.+\)\s*[<>]|limit"),
        ("csrf_protection", r"csrf|CSRFProtect|@csrf_protect|validate_csrf"),
        ("cors_policy", r"CORS|cors|Access-Control|allowed_origins"),
    ]

    def __init__(self):
        self.analyses = 0
        # Pre-compile all threat modeling patterns for performance
        self._compiled_stride = {
            cat: [re.compile(p, re.IGNORECASE) for p in info["patterns"]]
            for cat, info in self.STRIDE_CATEGORIES.items()
        }
        self._compiled_secrets = [(re.compile(p, re.IGNORECASE), stype) for p, stype in self.SECRET_PATTERNS]
        self._compiled_zt = [(name, re.compile(p, re.IGNORECASE)) for name, p in self.ZERO_TRUST_CHECKS]
        self._string_literal_re = re.compile(r'["\']([^"\']{20,})["\']')

    @staticmethod
    def _shannon_entropy(s: str) -> float:
        """Compute Shannon entropy of a string in bits/char (v6.1.0)."""
        if not s:
            return 0.0
        counts = Counter(s)
        length = len(s)
        return -sum((c / length) * math.log2(c / length) for c in counts.values() if c > 0)

    def model_threats(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Full STRIDE + DREAD threat model for source code."""
        self.analyses += 1
        start = time.time()

        # STRIDE classification (pre-compiled patterns)
        stride_findings = {}
        for category, info in self.STRIDE_CATEGORIES.items():
            findings = []
            for compiled in self._compiled_stride[category]:
                for match in compiled.finditer(source):
                    line_num = source[:match.start()].count('\n') + 1
                    findings.append({
                        "line": line_num,
                        "match": match.group()[:50],
                        "context": source[max(0, match.start()-20):match.end()+20].strip()[:80],
                    })
            stride_findings[category] = {
                "description": info["description"],
                "surface_count": len(findings),
                "findings": findings[:10],
            }

        # Secrets detection (pre-compiled patterns)
        secrets = []
        for compiled, secret_type in self._compiled_secrets:
            for match in compiled.finditer(source):
                line_num = source[:match.start()].count('\n') + 1
                secrets.append({
                    "type": secret_type,
                    "line": line_num,
                    "severity": "CRITICAL",
                    "detail": f"Potential {secret_type} found at line {line_num}",
                })

        # v6.1.0 — Shannon entropy-based secret detection
        # Catches high-entropy strings that no regex pattern would match
        known_secret_lines = {s["line"] for s in secrets}
        for match in self._string_literal_re.finditer(source):
            literal = match.group(1)
            if len(literal) < 20 or len(literal) > 500:
                continue
            entropy = self._shannon_entropy(literal)
            line_num = source[:match.start()].count('\n') + 1
            if line_num in known_secret_lines:
                continue
            if entropy > 4.5 and not literal.startswith(('http://', 'https://', '/', '.', '#')):
                # Exclude common high-entropy non-secrets
                if not any(kw in literal.lower() for kw in ('lorem', 'the ', 'function', 'class ', 'import ', 'return ')):
                    secrets.append({
                        "type": "HIGH_ENTROPY_STRING",
                        "line": line_num,
                        "severity": "MEDIUM",
                        "detail": f"High-entropy string (H={entropy:.2f} bits/char) at line {line_num} — possible embedded secret",
                        "entropy": round(entropy, 3),
                    })

        # Zero-trust audit (pre-compiled patterns)
        zt_results = {}
        for check_name, compiled in self._compiled_zt:
            matches = compiled.findall(source)
            zt_results[check_name] = {
                "present": len(matches) > 0,
                "occurrences": len(matches),
            }
        zt_score = sum(1 for v in zt_results.values() if v["present"]) / max(len(zt_results), 1)

        # Attack surface quantification
        entry_points = 0
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Public functions (not starting with _) are entry points
                    if not node.name.startswith('_'):
                        entry_points += 1
                    # Decorators like @app.route, @router.get increase surface
                    for dec in node.decorator_list:
                        dec_str = ast.dump(dec)
                        if any(k in dec_str for k in ['route', 'get', 'post', 'put', 'delete', 'endpoint', 'api']):
                            entry_points += 1
        except SyntaxError:
            entry_points = len(re.findall(r'^def\s+[a-zA-Z]', source, re.MULTILINE))

        # DREAD scoring (v6.1.0 — improved multi-factor scoring)
        total_surface = sum(s["surface_count"] for s in stride_findings.values())
        lines = source.count('\n') + 1
        critical_secrets = sum(1 for s in secrets if s.get("severity") == "CRITICAL")
        medium_secrets = sum(1 for s in secrets if s.get("severity") == "MEDIUM")
        info_disclosure_surface = stride_findings.get("information_disclosure", {}).get("surface_count", 0)
        elev_priv_surface = stride_findings.get("elevation_of_privilege", {}).get("surface_count", 0)
        api_endpoint_count = sum(1 for s in stride_findings.values() for f in s.get("findings", []) if "route" in str(f).lower())

        dread = {
            "damage": min(1.0, critical_secrets * 0.4 + medium_secrets * 0.15 + info_disclosure_surface * 0.03 + total_surface * 0.01),
            "reproducibility": min(1.0, api_endpoint_count * 0.08 + entry_points * 0.02),
            "exploitability": min(1.0, critical_secrets * 0.35 + elev_priv_surface * 0.1 + (1.0 - zt_score) * 0.25),
            "affected_users": min(1.0, api_endpoint_count * 0.06 + entry_points * 0.01),
            "discoverability": min(1.0, critical_secrets * 0.3 + medium_secrets * 0.1 + (1.0 - zt_score) * 0.2),
        }
        dread["composite"] = round(sum(dread.values()) / 5, 4)
        risk_level = ("CRITICAL" if dread["composite"] > 0.7 else "HIGH" if dread["composite"] > 0.5
                      else "MEDIUM" if dread["composite"] > 0.3 else "LOW")

        # v6.1.0 — Threat chain detection (compound threats)
        threat_chains = []
        has_encryption = zt_results.get("encryption", {}).get("present", False)
        has_auth = zt_results.get("auth_required", {}).get("present", False)
        has_rate_limit = zt_results.get("rate_limiting", {}).get("present", False)
        has_input_val = zt_results.get("input_validation", {}).get("present", False)
        has_csrf = zt_results.get("csrf_protection", {}).get("present", False)

        if critical_secrets > 0 and not has_encryption:
            threat_chains.append({
                "chain": "SECRET_EXPOSURE",
                "severity": "CRITICAL",
                "components": ["hardcoded_secret", "no_encryption"],
                "detail": "Secrets detected without encryption controls — plaintext exposure risk",
            })
        if has_auth and not has_rate_limit:
            threat_chains.append({
                "chain": "BRUTE_FORCE",
                "severity": "HIGH",
                "components": ["auth_endpoint", "no_rate_limiting"],
                "detail": "Authentication without rate limiting — brute-force attack vector",
            })
        if not has_input_val and total_surface > 5:
            threat_chains.append({
                "chain": "INJECTION",
                "severity": "HIGH",
                "components": ["no_input_validation", "large_attack_surface"],
                "detail": "Large attack surface without input validation — injection risk",
            })
        if not has_csrf and api_endpoint_count > 0:
            threat_chains.append({
                "chain": "CSRF_EXPOSURE",
                "severity": "MEDIUM",
                "components": ["api_endpoints", "no_csrf_protection"],
                "detail": "API endpoints without CSRF protection — cross-site request forgery risk",
            })

        # Sacred alignment: threat model resonance
        sacred_factor = round(GOD_CODE / (1 + dread["composite"] * 100), 4)

        duration = time.time() - start
        return {
            "version": "6.1.0",
            "filename": filename,
            "lines": lines,
            "duration_seconds": round(duration, 3),
            "stride": stride_findings,
            "secrets_detected": secrets,
            "secrets_count": len(secrets),
            "zero_trust": zt_results,
            "zero_trust_score": round(zt_score, 4),
            "attack_surface": {
                "entry_points": entry_points,
                "api_endpoints": api_endpoint_count,
                "total_surface": total_surface,
            },
            "dread_score": dread,
            "risk_level": risk_level,
            "threat_chains": threat_chains,
            "sacred_threat_factor": sacred_factor,
        }

    def status(self) -> Dict[str, Any]:
        return {
            "analyses": self.analyses,
            "stride_categories": len(self.STRIDE_CATEGORIES),
            "secret_patterns": len(self.SECRET_PATTERNS),
            "zero_trust_checks": len(self.ZERO_TRUST_CHECKS),
            "capabilities": ["model_threats", "stride_analysis", "dread_scoring", "secrets_detection", "zero_trust_audit"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# v6.0.0 — ARCHITECTURAL LINTER
# Clean architecture validation, layer violations, coupling metrics, SOLID integration
# ═══════════════════════════════════════════════════════════════════════════════



class ArchitecturalLinter:
    """
    v6.0.0 — Architectural rule enforcement and structural analysis.

    Capabilities:
      • Layer violation detection (presentation→domain→infrastructure)
      • Coupling metrics: afferent/efferent coupling, instability index
      • Component cohesion: relational cohesion, lack of cohesion in methods (LCOM)
      • Circular dependency detection with shortest-cycle reporting
      • Module naming convention enforcement
      • Sacred architecture: PHI-ratio module balance verification
    """

    # Default layered architecture rules (inner layers should not import outer)
    DEFAULT_LAYERS = {
        "presentation": {"keywords": ["view", "template", "ui", "frontend", "handler", "route", "endpoint", "controller"], "level": 0},
        "application": {"keywords": ["service", "use_case", "interactor", "command", "query", "handler"], "level": 1},
        "domain": {"keywords": ["model", "entity", "value_object", "aggregate", "domain", "core"], "level": 2},
        "infrastructure": {"keywords": ["repository", "database", "adapter", "gateway", "client", "driver", "cache"], "level": 3},
    }

    def __init__(self):
        self.analyses = 0

    def lint_architecture(self, source: str, filename: str = "",
                          workspace_path: str = None) -> Dict[str, Any]:
        """Analyze architectural conformance of source code."""
        self.analyses += 1
        start = time.time()

        violations = []
        metrics = {}

        # Detect which layer this file belongs to
        file_layer = self._classify_layer(filename, source)

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"error": "syntax_error", "violations": [], "metrics": {}}

        # Analyze imports for layer violations
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({"module": alias.name, "line": node.lineno})
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append({"module": node.module, "line": node.lineno})

        for imp in imports:
            imp_layer = self._classify_layer(imp["module"], "")
            if file_layer and imp_layer:
                file_level = self.DEFAULT_LAYERS.get(file_layer, {}).get("level", -1)
                imp_level = self.DEFAULT_LAYERS.get(imp_layer, {}).get("level", -1)
                if file_level > imp_level and imp_level >= 0:
                    violations.append({
                        "type": "layer_violation",
                        "severity": "HIGH",
                        "line": imp["line"],
                        "detail": f"Layer '{file_layer}' (level {file_level}) imports from '{imp_layer}' (level {imp_level}) — inner layer importing outer layer",
                        "module": imp["module"],
                    })

        # Coupling metrics
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

        # Efferent coupling (Ca): outgoing dependencies
        efferent = len(set(i["module"] for i in imports))

        # Relational cohesion: internal method interconnections
        class_metrics = []
        for cls in classes:
            methods = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            attributes = set()
            method_attr_usage = {}
            for method in methods:
                used_attrs = set()
                for node in ast.walk(method):
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'self':
                        used_attrs.add(node.attr)
                        attributes.add(node.attr)
                method_attr_usage[method.name] = used_attrs

            # LCOM (Lack of Cohesion in Methods)
            if len(methods) > 1 and len(attributes) > 0:
                shared_pairs = 0
                total_pairs = 0
                method_names = list(method_attr_usage.keys())
                for i in range(len(method_names)):
                    for j in range(i + 1, len(method_names)):
                        total_pairs += 1
                        if method_attr_usage[method_names[i]] & method_attr_usage[method_names[j]]:
                            shared_pairs += 1
                lcom = 1.0 - (shared_pairs / max(total_pairs, 1))
            else:
                lcom = 0.0

            class_metrics.append({
                "name": cls.name,
                "methods": len(methods),
                "attributes": len(attributes),
                "lcom": round(lcom, 4),
                "line": cls.lineno,
            })
            if lcom > 0.8 and len(methods) > 3:
                violations.append({
                    "type": "low_cohesion",
                    "severity": "MEDIUM",
                    "line": cls.lineno,
                    "detail": f"Class '{cls.name}' has low cohesion (LCOM={lcom:.2f}) — consider splitting",
                    "lcom": round(lcom, 4),
                })

        # Module naming convention
        if filename:
            base = os.path.basename(filename).replace('.py', '')
            if not re.match(r'^[a-z][a-z0-9_]*$', base) and not base.startswith('__'):
                violations.append({
                    "type": "naming_convention",
                    "severity": "LOW",
                    "detail": f"Module name '{base}' should be lowercase_with_underscores",
                    "line": 0,
                })

        # PHI-ratio module balance
        total_functions = len(functions)
        total_classes = len(classes)
        if total_functions > 0 and total_classes > 0:
            ratio = total_functions / total_classes
            phi_deviation = abs(ratio - PHI) / PHI
            phi_balanced = phi_deviation < 0.5
        else:
            phi_deviation = 1.0
            phi_balanced = False

        # Instability index: I = Ce / (Ca + Ce)
        # v6.1.0: Improved afferent coupling — count intra-module references to each class
        afferent = 0
        class_names_set = {c.name for c in classes}
        for cls in classes:
            refs_to_cls = 0
            for func in functions:
                func_source = ast.dump(func)
                if cls.name in func_source:
                    refs_to_cls += 1
            for other_cls in classes:
                if other_cls.name == cls.name:
                    continue
                cls_source = ast.dump(other_cls)
                if cls.name in cls_source:
                    refs_to_cls += 1
            afferent += refs_to_cls
        afferent = max(afferent, len(classes) + len(functions))
        instability = efferent / max(efferent + afferent, 1)

        # v6.1.0 — Martin's Main Sequence Analysis
        # Abstractness: ratio of abstract methods to total methods
        total_all_methods = 0
        abstract_method_count = 0
        for cls in classes:
            for node in cls.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_all_methods += 1
                    for stmt in ast.walk(node):
                        if isinstance(stmt, ast.Raise):
                            exc = getattr(stmt, 'exc', None)
                            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                                if exc.func.id == 'NotImplementedError':
                                    abstract_method_count += 1
                                    break
        abstractness = abstract_method_count / max(total_all_methods, 1)

        # Distance from Main Sequence: D = |A + I - 1|
        main_sequence_distance = abs(abstractness + instability - 1.0)
        if main_sequence_distance < 0.2:
            zone = "main_sequence"
        elif abstractness < 0.3 and instability < 0.3:
            zone = "zone_of_pain"
        elif abstractness > 0.7 and instability > 0.7:
            zone = "zone_of_uselessness"
        else:
            zone = "main_sequence"

        metrics = {
            "file_layer": file_layer or "unclassified",
            "imports": len(imports),
            "efferent_coupling": efferent,
            "afferent_coupling": afferent,
            "instability_index": round(instability, 4),
            "abstractness": round(abstractness, 4),
            "main_sequence_distance": round(main_sequence_distance, 4),
            "zone": zone,
            "classes": class_metrics,
            "class_count": len(classes),
            "function_count": total_functions,
            "phi_ratio": round(total_functions / max(total_classes, 1), 4),
            "phi_balanced": phi_balanced,
            "phi_deviation": round(phi_deviation, 4),
        }

        # Score
        violation_penalty = sum(0.15 if v["severity"] == "HIGH" else 0.08 if v["severity"] == "MEDIUM" else 0.03 for v in violations)
        arch_score = max(0.0, 1.0 - violation_penalty)

        duration = time.time() - start
        return {
            "version": "6.0.0",
            "filename": filename,
            "architecture_score": round(arch_score, 4),
            "violations": violations,
            "violation_count": len(violations),
            "metrics": metrics,
            "duration_seconds": round(duration, 3),
            "verdict": "CLEAN" if arch_score >= 0.9 else "MINOR_ISSUES" if arch_score >= 0.7 else "NEEDS_REFACTORING" if arch_score >= 0.5 else "ARCHITECTURAL_DEBT",
        }

    def _classify_layer(self, name: str, code: str) -> Optional[str]:
        """Classify a module/file into an architectural layer."""
        name_lower = (name or "").lower()
        for layer, info in self.DEFAULT_LAYERS.items():
            for kw in info["keywords"]:
                if kw in name_lower:
                    return layer
        return None

    def status(self) -> Dict[str, Any]:
        return {
            "analyses": self.analyses,
            "layers_defined": len(self.DEFAULT_LAYERS),
            "capabilities": ["lint_architecture", "layer_violations", "coupling_metrics", "lcom_analysis", "phi_balance"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# v6.0.0 — CODE MIGRATION ENGINE
# Framework migration, deprecation tracking, breaking change detection
# ═══════════════════════════════════════════════════════════════════════════════



class CodeMigrationEngine:
    """
    v6.0.0 — Automated code migration and deprecation intelligence.

    Capabilities:
      • Deprecated API detection (Python stdlib, Django, Flask, FastAPI, etc.)
      • Breaking change detection between code versions
      • Framework migration paths with automated rewrite suggestions
      • Python version compatibility analysis (3.8 → 3.12+)
      • Import modernization (typing → builtins, pathlib, f-strings)
    """

    PYTHON_DEPRECATIONS = {
        "collections.MutableMapping": {"replacement": "collections.abc.MutableMapping", "since": "3.3", "removed": "3.10"},
        "collections.MutableSequence": {"replacement": "collections.abc.MutableSequence", "since": "3.3", "removed": "3.10"},
        "collections.MutableSet": {"replacement": "collections.abc.MutableSet", "since": "3.3", "removed": "3.10"},
        "collections.Mapping": {"replacement": "collections.abc.Mapping", "since": "3.3", "removed": "3.10"},
        "collections.Sequence": {"replacement": "collections.abc.Sequence", "since": "3.3", "removed": "3.10"},
        "collections.Iterable": {"replacement": "collections.abc.Iterable", "since": "3.3", "removed": "3.10"},
        "typing.Dict": {"replacement": "dict", "since": "3.9", "removed": "future"},
        "typing.List": {"replacement": "list", "since": "3.9", "removed": "future"},
        "typing.Tuple": {"replacement": "tuple", "since": "3.9", "removed": "future"},
        "typing.Set": {"replacement": "set", "since": "3.9", "removed": "future"},
        "typing.FrozenSet": {"replacement": "frozenset", "since": "3.9", "removed": "future"},
        "typing.Type": {"replacement": "type", "since": "3.9", "removed": "future"},
        "typing.Optional": {"replacement": "X | None", "since": "3.10", "removed": "future"},
        "typing.Union": {"replacement": "X | Y", "since": "3.10", "removed": "future"},
        "os.path.join": {"replacement": "pathlib.Path / ...", "since": "3.4", "removed": "never", "note": "pathlib preferred"},
        "string.atoi": {"replacement": "int()", "since": "2.0", "removed": "3.0"},
        "imp.reload": {"replacement": "importlib.reload", "since": "3.4", "removed": "3.12"},
        "asyncio.coroutine": {"replacement": "async def", "since": "3.8", "removed": "3.11"},
        "loop.create_task": {"replacement": "asyncio.create_task", "since": "3.7", "removed": "never", "note": "preferred"},
        "unittest.makeSuite": {"replacement": "unittest.TestLoader", "since": "3.8", "removed": "3.13"},
        "cgi.escape": {"replacement": "html.escape", "since": "3.2", "removed": "3.8"},
        "formatter": {"replacement": "N/A (removed)", "since": "3.4", "removed": "3.10"},
        # v6.1.0 — Extended deprecation coverage
        "distutils": {"replacement": "setuptools", "since": "3.10", "removed": "3.12"},
        "pkg_resources": {"replacement": "importlib.resources / importlib.metadata", "since": "3.9", "removed": "future", "note": "setuptools legacy"},
        "configparser.SafeConfigParser": {"replacement": "configparser.ConfigParser", "since": "3.2", "removed": "3.12"},
        "sqlite3.OptimizedUnicode": {"replacement": "str", "since": "3.3", "removed": "3.12"},
        "asyncio.get_event_loop": {"replacement": "asyncio.get_running_loop()", "since": "3.10", "removed": "future", "note": "deprecated in non-coroutine context"},
        "typing.Deque": {"replacement": "collections.deque", "since": "3.9", "removed": "future"},
        "typing.DefaultDict": {"replacement": "collections.defaultdict", "since": "3.9", "removed": "future"},
        "typing.Counter": {"replacement": "collections.Counter", "since": "3.9", "removed": "future"},
        "typing.ChainMap": {"replacement": "collections.ChainMap", "since": "3.9", "removed": "future"},
        "typing.OrderedDict": {"replacement": "collections.OrderedDict", "since": "3.9", "removed": "future"},
        "typing.Pattern": {"replacement": "re.Pattern", "since": "3.9", "removed": "future"},
        "typing.Match": {"replacement": "re.Match", "since": "3.9", "removed": "future"},
        "typing.Text": {"replacement": "str", "since": "3.11", "removed": "future"},
        "pipes": {"replacement": "subprocess", "since": "3.11", "removed": "3.13"},
        "crypt": {"replacement": "bcrypt / argon2-cffi", "since": "3.11", "removed": "3.13"},
    }

    FRAMEWORK_MIGRATIONS = {
        "flask_to_fastapi": {
            "patterns": {
                r"from flask import": "from fastapi import FastAPI\nfrom fastapi import Request",
                r"@app\.route\((['\"])(.+?)\1,\s*methods=\[(['\"])GET\3\]\)": r"@app.get(\1\2\1)",
                r"@app\.route\((['\"])(.+?)\1,\s*methods=\[(['\"])POST\3\]\)": r"@app.post(\1\2\1)",
                r"request\.args\.get\((['\"])(\w+)\1\)": r"\2: str = Query(None)",
                r"request\.json": "await request.json()",
                r"jsonify\((.+?)\)": r"\1",
            },
            "notes": ["Add async/await to route handlers", "Replace Flask-specific extensions", "Update requirements.txt"],
        },
        "unittest_to_pytest": {
            "patterns": {
                r"import unittest": "import pytest",
                r"class (\w+)\(unittest\.TestCase\):": r"class \1:",
                r"self\.assertEqual\((.+?),\s*(.+?)\)": r"assert \1 == \2",
                r"self\.assertTrue\((.+?)\)": r"assert \1",
                r"self\.assertFalse\((.+?)\)": r"assert not \1",
                r"self\.assertRaises\((\w+)\)": r"pytest.raises(\1)",
                r"self\.assertIsNone\((.+?)\)": r"assert \1 is None",
                r"self\.assertIn\((.+?),\s*(.+?)\)": r"assert \1 in \2",
            },
            "notes": ["Remove setUp/tearDown → use fixtures", "Use conftest.py for shared fixtures"],
        },
    }

    def __init__(self):
        self.scans = 0

    def scan_deprecations(self, source: str, target_python: str = "3.12") -> Dict[str, Any]:
        """Scan for deprecated APIs and suggest replacements."""
        self.scans += 1
        findings = []
        target_parts = tuple(int(x) for x in target_python.split('.'))

        for deprecated, info in self.PYTHON_DEPRECATIONS.items():
            # Check if the deprecated pattern appears in source
            pattern = deprecated.replace('.', r'\.')
            for match in re.finditer(pattern, source):
                line_num = source[:match.start()].count('\n') + 1
                removed_version = info.get("removed", "future")
                is_removed = False
                if removed_version != "future" and removed_version != "never":
                    removed_parts = tuple(int(x) for x in removed_version.split('.'))
                    is_removed = target_parts >= removed_parts

                findings.append({
                    "deprecated": deprecated,
                    "replacement": info["replacement"],
                    "deprecated_since": info["since"],
                    "removed_in": removed_version,
                    "is_removed": is_removed,
                    "severity": "CRITICAL" if is_removed else "WARNING",
                    "line": line_num,
                    "note": info.get("note", ""),
                })

        # Sort by severity
        findings.sort(key=lambda f: 0 if f["severity"] == "CRITICAL" else 1)

        return {
            "target_python": target_python,
            "total_deprecations": len(findings),
            "critical": sum(1 for f in findings if f["severity"] == "CRITICAL"),
            "warnings": sum(1 for f in findings if f["severity"] == "WARNING"),
            "findings": findings,
            "migration_ready": len([f for f in findings if f["severity"] == "CRITICAL"]) == 0,
        }

    def suggest_migration(self, source: str, migration_path: str) -> Dict[str, Any]:
        """Suggest code transformations for a framework migration."""
        self.scans += 1
        if migration_path not in self.FRAMEWORK_MIGRATIONS:
            return {
                "error": f"Unknown migration path: {migration_path}",
                "available_paths": list(self.FRAMEWORK_MIGRATIONS.keys()),
            }

        migration = self.FRAMEWORK_MIGRATIONS[migration_path]
        transformed = source
        changes = []

        for pattern, replacement in migration["patterns"].items():
            matches = list(re.finditer(pattern, transformed))
            if matches:
                transformed = re.sub(pattern, replacement, transformed)
                changes.append({
                    "pattern": pattern[:60],
                    "replacement": replacement[:60],
                    "occurrences": len(matches),
                })

        return {
            "migration_path": migration_path,
            "changes_applied": len(changes),
            "changes": changes,
            "transformed_source": transformed,
            "notes": migration.get("notes", []),
            "chars_changed": len(transformed) - len(source),
            "success": len(changes) > 0,
        }

    def detect_breaking_changes(self, old_source: str, new_source: str) -> Dict[str, Any]:
        """Compare two versions and detect breaking changes in public API."""
        self.scans += 1
        breaking = []

        try:
            old_tree = ast.parse(old_source)
            new_tree = ast.parse(new_source)
        except SyntaxError:
            return {"error": "syntax_error", "breaking_changes": []}

        # Extract public API (functions + classes)
        def extract_api(tree):
            api = {}
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith('_'):
                    args = [a.arg for a in node.args.args if a.arg != 'self']
                    defaults = len(node.args.defaults)
                    api[node.name] = {"type": "function", "args": args, "defaults": defaults, "line": node.lineno}
                elif isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
                    methods = {}
                    for m in node.body:
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith('_'):
                            m_args = [a.arg for a in m.args.args if a.arg != 'self']
                            methods[m.name] = {"args": m_args, "defaults": len(m.args.defaults)}
                    api[node.name] = {"type": "class", "methods": methods, "line": node.lineno}
            return api

        old_api = extract_api(old_tree)
        new_api = extract_api(new_tree)

        # Check for removed public APIs
        for name, info in old_api.items():
            if name not in new_api:
                breaking.append({
                    "type": "removed",
                    "severity": "CRITICAL",
                    "name": name,
                    "kind": info["type"],
                    "detail": f"Public {info['type']} '{name}' was removed",
                })
            elif info["type"] == "function" and new_api[name]["type"] == "function":
                # Check for arg changes
                old_args = info["args"]
                new_args = new_api[name]["args"]
                if len(new_args) > len(old_args) and new_api[name]["defaults"] < len(new_args) - len(old_args):
                    breaking.append({
                        "type": "new_required_args",
                        "severity": "HIGH",
                        "name": name,
                        "detail": f"Function '{name}' added required arguments: {set(new_args) - set(old_args)}",
                    })
                removed_args = set(old_args) - set(new_args)
                if removed_args:
                    breaking.append({
                        "type": "removed_args",
                        "severity": "HIGH",
                        "name": name,
                        "detail": f"Function '{name}' removed arguments: {removed_args}",
                    })
            elif info["type"] == "class" and new_api[name]["type"] == "class":
                # Check for removed methods
                old_methods = set(info.get("methods", {}).keys())
                new_methods = set(new_api[name].get("methods", {}).keys())
                for removed in old_methods - new_methods:
                    breaking.append({
                        "type": "removed_method",
                        "severity": "HIGH",
                        "name": f"{name}.{removed}",
                        "detail": f"Public method '{removed}' removed from class '{name}'",
                    })

        return {
            "breaking_changes": breaking,
            "total_breaking": len(breaking),
            "critical": sum(1 for b in breaking if b["severity"] == "CRITICAL"),
            "backward_compatible": len(breaking) == 0,
            "old_api_size": len(old_api),
            "new_api_size": len(new_api),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "scans": self.scans,
            "deprecation_rules": len(self.PYTHON_DEPRECATIONS),
            "migration_paths": list(self.FRAMEWORK_MIGRATIONS.keys()),
            "capabilities": ["scan_deprecations", "suggest_migration", "detect_breaking_changes"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# v6.0.0 — PERFORMANCE BENCHMARK PREDICTOR
# Predictive performance modeling, memory estimates, throughput analysis
# ═══════════════════════════════════════════════════════════════════════════════



class PerformanceBenchmarkPredictor:
    """
    v6.0.0 — Predictive performance analysis without runtime execution.

    Capabilities:
      • Memory footprint estimation per class/function
      • Throughput prediction based on algorithmic complexity + I/O patterns
      • Cache-friendliness analysis (data locality, object size)
      • I/O bottleneck detection (file, network, database)
      • GIL contention prediction for threaded code
      • PHI-weighted performance scoring
    """

    # Approximate memory costs (bytes) for Python objects
    MEMORY_COSTS = {
        "int": 28, "float": 24, "str_base": 49, "str_per_char": 1,
        "list_base": 56, "list_per_item": 8, "dict_base": 232, "dict_per_item": 72,
        "set_base": 216, "set_per_item": 8, "tuple_base": 40, "tuple_per_item": 8,
        "object_base": 56, "class_overhead": 1064,
        "numpy_array_base": 96, "numpy_per_element": 8,
        "dataclass_overhead": 152,
    }

    IO_PATTERNS = {
        "file_read": (r"open\(|\.read\(|\.readlines\(|Path\(.+\)\.read", "FILE_IO"),
        "file_write": (r"\.write\(|\.writelines\(|\.dump\(|\.save\(", "FILE_IO"),
        "network": (r"requests\.|urllib\.|aiohttp\.|httpx\.|socket\.|fetch\(", "NETWORK_IO"),
        "database": (r"\.execute\(|\.query\(|\.find\(|\.insert\(|\.update\(|\.delete\(|SELECT|INSERT|UPDATE", "DATABASE_IO"),
        "subprocess": (r"subprocess\.|os\.system\(|os\.popen\(|Popen\(", "SUBPROCESS_IO"),
        "sleep": (r"time\.sleep\(|asyncio\.sleep\(|await\s+sleep", "BLOCKING"),
    }

    def __init__(self):
        self.predictions = 0

    def predict_performance(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Full predictive performance analysis."""
        self.predictions += 1
        start = time.time()

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"error": "syntax_error"}

        # Memory estimation
        memory_estimate = self._estimate_memory(tree, source)

        # I/O bottleneck detection
        io_bottlenecks = self._detect_io(source)

        # GIL contention analysis
        gil_risk = self._analyze_gil(source)

        # Allocation hotspots (loops creating objects)
        alloc_hotspots = self._detect_allocation_hotspots(tree)

        # Throughput factors
        throughput = self._estimate_throughput(tree, source, io_bottlenecks)

        # PHI-weighted performance score
        scores = {
            "memory": max(0, 1.0 - memory_estimate["total_estimated_bytes"] / (10 * 1024 * 1024)),  # Penalize >10MB
            "io_efficiency": 1.0 - min(1.0, len(io_bottlenecks) * 0.15),
            "gil_safety": 1.0 - gil_risk["risk_score"],
            "allocation_health": max(0, 1.0 - len(alloc_hotspots) * 0.1),
            "throughput": throughput["score"],
        }
        weights = [PHI, PHI, 1.0, TAU, PHI**2]
        composite = sum(s * w for s, w in zip(scores.values(), weights)) / sum(weights)

        duration = time.time() - start
        return {
            "version": "6.0.0",
            "filename": filename,
            "duration_seconds": round(duration, 3),
            "performance_score": round(composite, 4),
            "memory_estimate": memory_estimate,
            "io_bottlenecks": io_bottlenecks,
            "gil_contention": gil_risk,
            "allocation_hotspots": alloc_hotspots,
            "throughput": throughput,
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "verdict": ("OPTIMAL" if composite >= 0.85 else "GOOD" if composite >= 0.7
                        else "ACCEPTABLE" if composite >= 0.5 else "NEEDS_OPTIMIZATION"
                        if composite >= 0.3 else "PERFORMANCE_CRITICAL"),
        }

    def _estimate_memory(self, tree, source: str) -> Dict[str, Any]:
        """Estimate memory footprint from AST."""
        total = 0
        details = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = sum(1 for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
                attrs = sum(1 for n in ast.walk(node) if isinstance(n, ast.Attribute)
                            and isinstance(n.value, ast.Name) and n.value.id == 'self')
                est = self.MEMORY_COSTS["class_overhead"] + attrs * self.MEMORY_COSTS["object_base"]
                total += est
                details.append({"name": node.name, "type": "class", "estimated_bytes": est,
                                "methods": methods, "attributes": min(attrs, 50)})

        # Estimate from data structure literals
        list_count = len(re.findall(r'\[.*?\]', source))
        dict_count = len(re.findall(r'\{.*?:.*?\}', source))
        total += list_count * self.MEMORY_COSTS["list_base"]
        total += dict_count * self.MEMORY_COSTS["dict_base"]

        # NumPy arrays
        np_arrays = len(re.findall(r'np\.(?:array|zeros|ones|empty|linspace|arange)\(', source))
        total += np_arrays * (self.MEMORY_COSTS["numpy_array_base"] + 1000 * self.MEMORY_COSTS["numpy_per_element"])

        return {
            "total_estimated_bytes": total,
            "total_human": f"{total / 1024:.1f} KB" if total < 1024 * 1024 else f"{total / (1024*1024):.2f} MB",
            "class_details": details[:20],
            "data_structures": {"lists": list_count, "dicts": dict_count, "numpy_arrays": np_arrays},
        }

    def _detect_io(self, source: str) -> List[Dict[str, Any]]:
        """Detect I/O bottleneck patterns."""
        bottlenecks = []
        for name, (pattern, io_type) in self.IO_PATTERNS.items():
            for match in re.finditer(pattern, source, re.IGNORECASE):
                line_num = source[:match.start()].count('\n') + 1
                bottlenecks.append({
                    "type": io_type,
                    "pattern": name,
                    "line": line_num,
                    "detail": f"{io_type} operation detected: {match.group()[:40]}",
                })
        return bottlenecks

    def _analyze_gil(self, source: str) -> Dict[str, Any]:
        """Analyze GIL contention risk."""
        threading_imports = bool(re.search(r'import threading|from threading', source))
        multiprocessing_imports = bool(re.search(r'import multiprocessing|from multiprocessing', source))
        thread_creation = len(re.findall(r'Thread\(|threading\.Thread', source))
        lock_usage = len(re.findall(r'Lock\(\)|RLock\(\)|Semaphore\(|Condition\(', source))
        cpu_bound = bool(re.search(r'for\s+\w+\s+in\s+range\(\d{4,}\)|while.*[+\-*/].*:', source))

        risk = 0.0
        if threading_imports and cpu_bound:
            risk = 0.8
        elif threading_imports:
            risk = 0.3
        if lock_usage > 2:
            risk = min(risk + 0.2, 1.0)

        return {
            "threading_used": threading_imports,
            "multiprocessing_used": multiprocessing_imports,
            "thread_creations": thread_creation,
            "lock_usages": lock_usage,
            "cpu_bound_detected": cpu_bound,
            "risk_score": round(risk, 3),
            "recommendation": ("Use multiprocessing for CPU-bound work" if risk > 0.5
                               else "Threading safe for I/O-bound work" if threading_imports
                               else "No threading concerns"),
        }

    def _detect_allocation_hotspots(self, tree) -> List[Dict[str, Any]]:
        """Detect object allocation inside loops."""
        hotspots = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func_name = ""
                        if isinstance(child.func, ast.Name):
                            func_name = child.func.id
                        elif isinstance(child.func, ast.Attribute):
                            func_name = child.func.attr
                        if func_name in ('list', 'dict', 'set', 'DataFrame', 'array', 'open'):
                            hotspots.append({
                                "type": "allocation_in_loop",
                                "line": child.lineno,
                                "function": func_name,
                                "severity": "HIGH" if func_name in ('DataFrame', 'open') else "MEDIUM",
                                "detail": f"'{func_name}()' called inside loop — potential memory pressure",
                            })
        return hotspots[:20]

    def _estimate_throughput(self, tree, source: str, io_bottlenecks: list) -> Dict[str, Any]:
        """Estimate relative throughput characteristics."""
        # Count loops, nested loops
        loop_depth = 0
        max_depth = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                # Simple nesting depth estimation
                inner_loops = sum(1 for child in ast.walk(node) if isinstance(child, (ast.For, ast.While)) and child is not node)
                max_depth = max(max_depth, inner_loops + 1)

        io_penalty = min(0.5, len(io_bottlenecks) * 0.1)
        complexity_penalty = min(0.5, max_depth * 0.15)

        score = max(0.0, 1.0 - io_penalty - complexity_penalty)

        return {
            "max_loop_nesting": max_depth,
            "io_operations": len(io_bottlenecks),
            "io_penalty": round(io_penalty, 3),
            "complexity_penalty": round(complexity_penalty, 3),
            "score": round(score, 4),
            "classification": ("CPU_BOUND" if max_depth >= 3 and len(io_bottlenecks) < 2
                               else "IO_BOUND" if len(io_bottlenecks) >= 3
                               else "BALANCED"),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "predictions": self.predictions,
            "io_patterns": len(self.IO_PATTERNS),
            "memory_cost_types": len(self.MEMORY_COSTS),
            "capabilities": ["predict_performance", "memory_estimate", "io_bottlenecks", "gil_analysis", "throughput_prediction"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# v6.0.0 — SEMANTIC CODE SEARCH ENGINE
# TF-IDF + sacred-weighted similarity search, cross-file code clone detection
# ═══════════════════════════════════════════════════════════════════════════════

