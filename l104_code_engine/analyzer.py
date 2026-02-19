"""L104 Code Engine — Domain A: Analysis & Formatting."""
from .constants import *
from .languages import LanguageKnowledge

class CodeAnalyzer:
    """Deep code analysis using Python's ast module + custom metrics.
    Computes cyclomatic complexity, Halstead metrics, cognitive complexity,
    dead code detection, security vulnerability patterns, and sacred-constant
    alignment scoring."""

    # OWASP-derived security patterns
    SECURITY_PATTERNS = {
        # ── OWASP Top 10 2025 + CWE Top 25 2024 Coverage ──
        # A05:2025 Injection / CWE-89 SQL Injection
        "sql_injection": [
            r"execute\s*\(\s*[\"'].*%s.*[\"']\s*%",
            r"f[\"'].*SELECT.*{.*}.*FROM",
            r"\.format\(.*\).*(?:SELECT|INSERT|DELETE|UPDATE)",
            r"cursor\.execute\s*\(\s*[\"'].*\+",
            r"raw\s*\(\s*[\"'].*%",
        ],
        # A05:2025 Injection / CWE-78 OS Command Injection
        "command_injection": [
            r"os\.system\s*\(",
            r"subprocess\.call\s*\([^,]*shell\s*=\s*True",
            r"(?<!ast\.literal_)eval\s*\([^)]*\+",         # eval with string concat = dangerous
            r"exec\s*\(\s*(?:f[\"']|[\"'].*\+|.*\.format)",  # exec with format strings
            r"subprocess\.Popen\s*\([^)]*shell\s*=\s*True",
            r"os\.popen\s*\(",
            r"commands\.getoutput\s*\(",
        ],
        # A01:2025 Broken Access Control / CWE-22 Path Traversal
        "path_traversal": [
            r"open\s*\(.*\+.*\)",
            r"os\.path\.join\s*\(.*request",
            r"\.\.\/",                                       # directory traversal literals
            r"sendFile\s*\(.*req\.",
            r"os\.path\.join\s*\(.*input\s*\(",
        ],
        # A04:2025 Cryptographic Failures / CWE-798 Hardcoded Credentials
        "hardcoded_secrets": [
            r"(?:password|secret|api_key|token|private_key)\s*=\s*[\"'][a-zA-Z0-9+/=_-]{16,}[\"']",
            r"(?:AWS|AZURE|GCP|GITHUB|SLACK)_(?:SECRET|KEY|TOKEN)\s*=\s*[\"'][^\"']+[\"']",
            r"(?:BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY)",
            r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}",  # GitHub tokens
        ],
        # A08:2025 Software/Data Integrity / CWE-502 Insecure Deserialization
        "insecure_deserialization": [
            r"pickle\.loads?\s*\(",
            r"yaml\.load\s*\([^,]*\)(?!\s*,\s*Loader)",
            r"marshal\.loads?\s*\(",
            r"shelve\.open\s*\(",
            r"jsonpickle\.decode\s*\(",
        ],
        # CWE-79 Cross-Site Scripting
        "xss_potential": [
            r"innerHTML\s*=",
            r"document\.write\s*\(",
            r"\.html\s*\(.*\+",
            r"dangerouslySetInnerHTML",
            r"v-html\s*=",
            r"\|safe\b",                                     # Jinja2 safe filter
        ],
        # CWE-352 Cross-Site Request Forgery (NEW)
        "csrf_vulnerability": [
            r"@csrf_exempt",
            r"csrf_protect\s*=\s*False",
            r"WTF_CSRF_ENABLED\s*=\s*False",
        ],
        # A02:2025 Security Misconfiguration (NEW)
        "security_misconfiguration": [
            r"DEBUG\s*=\s*True",
            r"ALLOWED_HOSTS\s*=\s*\[\s*[\"']\*[\"']",
            r"verify\s*=\s*False",                           # SSL verification disabled
            r"(?:CORS_ALLOW_ALL|CORS_ORIGIN_ALLOW_ALL)\s*=\s*True",
            r"app\.run\s*\([^)]*debug\s*=\s*True",
        ],
        # CWE-918 Server-Side Request Forgery (NEW)
        "ssrf_potential": [
            r"requests\.(?:get|post|put|delete|head)\s*\(.*(?:request\.|input\(|argv)",
            r"urllib\.request\.urlopen\s*\(.*(?:request\.|input\()",
            r"urlopen\s*\(.*\+",
        ],
        # CWE-434 Unrestricted File Upload (NEW)
        "unrestricted_upload": [
            r"\.save\s*\(.*filename\)",
            r"request\.files\[",
            r"file\.save\s*\(.*os\.path\.join",
        ],
        # CWE-94 Code Injection (NEW)
        "code_injection": [
            r"compile\s*\(.*[\"']exec[\"']",
            r"__import__\s*\(.*input",
            r"importlib\.import_module\s*\(.*request",
            r"getattr\s*\(.*request\.",
        ],
        # CWE-287/CWE-306 Authentication Failures (NEW)
        "authentication_failure": [
            r"authenticate\s*=\s*False",
            r"login_required\s*=\s*False",
            r"@no_auth",
            r"AllowAny",
        ],
        # CWE-400 Uncontrolled Resource Consumption (NEW)
        "resource_exhaustion": [
            r"while\s+True\s*:",
            r"re\.compile\s*\([\"'].*(?:\.\*){3,}",          # ReDoS patterns
            r"\.read\s*\(\s*\)",                              # unbounded read
        ],
        # A09:2025 Security Logging Failures (NEW)
        "logging_sensitive_data": [
            r"(?:logger|logging|log)\.(?:info|debug|warning|error)\s*\(.*(?:password|secret|token|api_key)",
            r"print\s*\(.*(?:password|secret|token|api_key)",
        ],
        # A03:2025 Software Supply Chain (NEW)
        "supply_chain_risk": [
            r"pip\s+install\s+--index-url\s+http://",       # insecure package source
            r"curl\s+.*\|\s*(?:bash|sh|python)",             # pipe to shell
            r"wget\s+.*\|\s*(?:bash|sh)",
        ],
    }

    # Algorithm complexity patterns
    COMPLEXITY_PATTERNS = {
        "O(1)": ["hash_lookup", "array_index", "constant_return"],
        "O(log n)": ["binary_search", "tree_traversal_balanced", "divide_conquer"],
        "O(n)": ["single_loop", "linear_scan", "list_comprehension"],
        "O(n log n)": ["merge_sort", "heap_sort", "sorted_builtin"],
        "O(n²)": ["nested_loops", "bubble_sort", "selection_sort"],
        "O(n³)": ["triple_nested", "matrix_multiply_naive"],
        "O(2^n)": ["recursive_fibonacci", "power_set", "backtracking"],
        "O(n!)": ["permutations", "tsp_brute_force"],
    }

    # Design pattern indicators — 25 GOF + Modern patterns (was 10)
    DESIGN_PATTERNS = {
        # ── Creational Patterns ──
        "singleton": [r"_instance\s*=\s*None", r"__new__\s*\(", r"@classmethod.*instance"],
        "factory": [r"def\s+create_\w+", r"class\s+\w+Factory"],
        "abstract_factory": [r"class\s+Abstract\w+Factory", r"def\s+create_\w+\(self\).*->.*ABC"],
        "builder": [r"\.set_\w+\(", r"\.build\(\)", r"class\s+\w+Builder"],
        "prototype": [r"def\s+clone\(self\)", r"copy\.deepcopy\(self\)", r"import\s+copy"],
        # ── Structural Patterns ──
        "adapter": [r"class\s+\w+Adapter", r"def\s+adapt\("],
        "bridge": [r"class\s+\w+Bridge", r"self\._implementor", r"def\s+set_implementor\("],
        "composite": [r"self\._children", r"def\s+add\(self.*child\)", r"class\s+\w+Composite"],
        "decorator": [r"def\s+\w+\(func\)", r"@functools\.wraps", r"wrapper\("],
        "facade": [r"class\s+\w+Facade", r"class\s+\w+Gateway", r"def\s+\w+_simplified\("],
        "flyweight": [r"_cache\s*=\s*\{\}", r"class\s+\w+Pool", r"def\s+get_instance\(.*key"],
        "proxy": [r"class\s+\w+Proxy", r"self\._real_\w+", r"def\s+__getattr__\(self"],
        # ── Behavioral Patterns ──
        "observer": [r"\.subscribe\(", r"\.notify\(", r"listeners", r"callbacks"],
        "strategy": [r"class\s+\w+Strategy", r"\.set_strategy\(", r"\.execute\("],
        "iterator": [r"def\s+__iter__\(", r"def\s+__next__\(", r"yield\s+"],
        "command": [r"def\s+execute\(self\)", r"class\s+\w+Command"],
        "template_method": [r"def\s+_\w+\(self\)", r"raise\s+NotImplementedError"],
        "chain_of_responsibility": [r"self\._next_handler", r"def\s+set_next\(", r"class\s+\w+Handler"],
        "mediator": [r"class\s+\w+Mediator", r"self\._colleagues", r"def\s+notify\(self.*sender"],
        "memento": [r"def\s+save_state\(", r"def\s+restore_state\(", r"class\s+\w+Memento"],
        "state": [r"class\s+\w+State", r"self\._state\s*=", r"def\s+transition_to\("],
        "visitor": [r"def\s+accept\(self.*visitor\)", r"def\s+visit_\w+\(", r"class\s+\w+Visitor"],
        "interpreter": [r"def\s+interpret\(self", r"class\s+\w+Expression", r"class\s+\w+Parser"],
        # ── Modern / Architectural Patterns ──
        "mvc": [r"class\s+\w+Controller", r"class\s+\w+View", r"class\s+\w+Model"],
        "dependency_injection": [r"def\s+__init__\(self.*service", r"@inject", r"class\s+\w+Container"],
    }

    def __init__(self):
        """Initialize CodeAnalyzer with counters, pattern tracking, and pre-compiled regex."""
        self.analysis_count = 0
        self.total_lines_analyzed = 0
        self.vulnerability_count = 0
        self.pattern_detections = Counter()
        self._analysis_cache: Dict[str, Dict] = {}
        # Pre-compile security & design pattern regex (avoids re-compilation per scan)
        self._compiled_security = {
            vtype: [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in patterns]
            for vtype, patterns in self.SECURITY_PATTERNS.items()
        }
        self._compiled_design = {
            pname: [re.compile(ind, re.MULTILINE) for ind in indicators]
            for pname, indicators in self.DESIGN_PATTERNS.items()
        }

    def full_analysis(self, code: str, filename: str = "") -> Dict[str, Any]:
        """Run comprehensive analysis on code: AST, complexity, quality, security, patterns."""
        # Content-hash cache: skip re-analysis for identical code
        _code_hash = hashlib.md5(code.encode()).hexdigest()
        if _code_hash in self._analysis_cache:
            self.analysis_count += 1
            _cached = self._analysis_cache[_code_hash].copy()
            _cached["metadata"] = _cached["metadata"].copy()
            _cached["metadata"]["filename"] = filename
            _cached["metadata"]["timestamp"] = datetime.now().isoformat()
            return _cached
        self.analysis_count += 1
        lines = code.split('\n')
        self.total_lines_analyzed += len(lines)

        result = {
            "metadata": {
                "filename": filename,
                "language": LanguageKnowledge.detect_language(code, filename),
                "lines": len(lines),
                "blank_lines": sum(1 for l in lines if not l.strip()),
                "comment_lines": sum(1 for l in lines if l.strip().startswith('#') or l.strip().startswith('//')),
                "code_lines": 0,
                "characters": len(code),
                "timestamp": datetime.now().isoformat(),
                "engine_version": VERSION,
            },
            "complexity": {},
            "quality": {},
            "security": [],
            "patterns": [],
            "sacred_alignment": {},
        }
        result["metadata"]["code_lines"] = (
            result["metadata"]["lines"] - result["metadata"]["blank_lines"] - result["metadata"]["comment_lines"]
        )

        # Python-specific deep analysis via AST (shared parse tree)
        lang = result["metadata"]["language"]
        if lang == "Python":
            try:
                _tree = ast.parse(code)
            except SyntaxError:
                _tree = None
            result["complexity"] = self._python_complexity(code, _tree=_tree)
            result["quality"] = self._python_quality(code, lines, _tree=_tree)
        else:
            result["complexity"] = self._generic_complexity(code, lines)
            result["quality"] = self._generic_quality(code, lines)

        # Security scan (language-agnostic patterns)
        result["security"] = self._security_scan(code)
        self.vulnerability_count += len(result["security"])

        # Design pattern detection
        result["patterns"] = self._detect_patterns(code)

        # Sacred constant alignment
        result["sacred_alignment"] = self._sacred_alignment(code, result)

        # Cache result (max 64 entries, evict oldest on overflow)
        if len(self._analysis_cache) >= 64:
            _oldest = next(iter(self._analysis_cache))
            del self._analysis_cache[_oldest]
        self._analysis_cache[_code_hash] = result

        return result

    def _python_complexity(self, code: str, _tree: ast.AST = None) -> Dict[str, Any]:
        """Python-specific complexity analysis using ast module."""
        try:
            tree = _tree if _tree is not None else ast.parse(code)
        except SyntaxError as e:
            return {"error": f"SyntaxError: {e}", "cyclomatic": -1}

        functions = []
        classes = []
        imports = []
        global_vars = []
        decorators_used = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cc = self._cyclomatic_complexity(node)
                cognitive = self._cognitive_complexity(node)
                functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "args": len(node.args.args),
                    "cyclomatic_complexity": cc,
                    "cognitive_complexity": cognitive,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "has_docstring": (
                        isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    ) if node.body else False,
                    "decorators": [self._decorator_name(d) for d in node.decorator_list],
                    "nested_depth": self._max_nesting_depth(node),
                    "body_lines": len(node.body),
                })
                for d in node.decorator_list:
                    decorators_used.add(self._decorator_name(d))
            elif isinstance(node, ast.ClassDef):
                methods = [n for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                classes.append({
                    "name": node.name,
                    "line": node.lineno,
                    "methods": len(methods),
                    "bases": [self._node_name(b) for b in node.bases],
                    "has_docstring": (
                        isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    ) if node.body else False,
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                else:
                    imports.append(f"{node.module}.{','.join(a.name for a in node.names)}")
            elif isinstance(node, ast.Assign) and isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        global_vars.append(target.id)

        # Halstead metrics
        halstead = self._halstead_metrics(code)

        total_cc = sum(f["cyclomatic_complexity"] for f in functions) if functions else 0
        avg_cc = total_cc / len(functions) if functions else 0

        # Maintainability Index (Radon/SEI/VS derivative)
        mi = self._maintainability_index(halstead, total_cc, code)

        return {
            "cyclomatic_total": total_cc,
            "cyclomatic_average": round(avg_cc, 2),
            "cyclomatic_max": max((f["cyclomatic_complexity"] for f in functions), default=0),
            "cognitive_max": max((f["cognitive_complexity"] for f in functions), default=0),
            "maintainability_index": mi,
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "global_variables": global_vars,
            "decorators_used": list(decorators_used),
            "halstead": halstead,
            "max_nesting": max((f["nested_depth"] for f in functions), default=0),
            "function_count": len(functions),
            "class_count": len(classes),
            "import_count": len(imports),
        }

    def _cyclomatic_complexity(self, node: ast.AST) -> int:
        """Compute McCabe cyclomatic complexity for an AST node."""
        complexity = 1  # Base path
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.Assert, ast.With)):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
                if child.ifs:
                    complexity += len(child.ifs)
        return complexity

    def _cognitive_complexity(self, node: ast.AST, nesting: int = 0) -> int:
        """Compute cognitive complexity (SonarSource-derived).

        Rules (per Sonar Cognitive Complexity white paper):
        1. +1 for each break in linear flow (if, for, while, try, catch)
        2. +1 per nesting level for nested flow-break structures
        3. +1 for each boolean operator sequence change (and/or mixed)
        4. +1 for else/elif (a branch the reader must track)
        5. +1 for break/continue/goto (jump-to labels)
        6. No increment for the method itself or switch cases
        7. Recursion increments +1 (detected call to own name)
        """
        score = 0
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                score += 1 + nesting  # +1 structural, +nesting penalty
                # Check for else/elif branches
                if isinstance(child, ast.If) and child.orelse:
                    if child.orelse and isinstance(child.orelse[0], ast.If):
                        score += 1  # elif — additional branch
                    else:
                        score += 1  # else — additional branch
                score += self._cognitive_complexity(child, nesting + 1)
            elif isinstance(child, ast.BoolOp):
                # +1 per sequence of mixed boolean operators
                score += 1
                # Additional +1 for each operator beyond the first in a compound expression
                if len(child.values) > 2:
                    score += len(child.values) - 2
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Nested function increments nesting but not score
                score += self._cognitive_complexity(child, nesting + 1)
            elif isinstance(child, ast.Try):
                score += 1 + nesting
                score += self._cognitive_complexity(child, nesting + 1)
            elif isinstance(child, ast.ExceptHandler):
                score += 1 + nesting  # catch block — flow break
                score += self._cognitive_complexity(child, nesting + 1)
            elif isinstance(child, (ast.Break, ast.Continue)):
                score += 1  # jump-to label
            elif isinstance(child, ast.IfExp):  # ternary operator
                score += 1 + nesting
                score += self._cognitive_complexity(child, nesting)
            else:
                score += self._cognitive_complexity(child, nesting)
        return score

    def _max_nesting_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Compute maximum nesting depth in an AST subtree."""
        max_d = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                max_d = max(max_d, self._max_nesting_depth(child, depth + 1))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                max_d = max(max_d, self._max_nesting_depth(child, depth + 1))
            else:
                max_d = max(max_d, self._max_nesting_depth(child, depth))
        return max_d

    def _halstead_metrics(self, code: str) -> Dict[str, float]:
        """Compute Halstead complexity metrics from token stream."""
        operators = set()
        operands = set()
        total_operators = 0
        total_operands = 0
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
            for tok in tokens:
                if tok.type == tokenize.OP:
                    operators.add(tok.string)
                    total_operators += 1
                elif tok.type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING):
                    operands.add(tok.string)
                    total_operands += 1
        except (tokenize.TokenError, IndentationError, SyntaxError):
            pass

        n1 = len(operators)  # Unique operators
        n2 = len(operands)   # Unique operands
        N1 = total_operators  # Total operators
        N2 = total_operands   # Total operands
        N = N1 + N2           # Program length
        n = n1 + n2           # Vocabulary

        volume = N * math.log2(n) if n > 0 else 0
        difficulty = (n1 / 2.0) * (N2 / max(1, n2)) if n2 > 0 else 0
        effort = volume * difficulty
        time_to_program = effort / 18.0  # Halstead's constant
        bugs_estimate = volume / 3000.0  # Halstead's bug prediction

        return {
            "vocabulary": n,
            "length": N,
            "volume": round(volume, 2),
            "difficulty": round(difficulty, 2),
            "effort": round(effort, 2),
            "time_estimate_seconds": round(time_to_program, 2),
            "bugs_estimate": round(bugs_estimate, 4),
            "unique_operators": n1,
            "unique_operands": n2,
        }

    def _maintainability_index(self, halstead: Dict, cyclomatic_total: int, code: str) -> Dict[str, Any]:
        """
        Compute Maintainability Index using Radon/SEI/VS derivative formula.
        MI = max[0, 100 * (171 - 5.2*ln(V) - 0.23*G - 16.2*ln(L) + 50*sin(sqrt(2.4*C))) / 171]
        Where V=Halstead Volume, G=Cyclomatic Complexity, L=SLOC, C=comment ratio (radians).
        """
        V = max(1, halstead.get("volume", 1))
        G = max(0, cyclomatic_total)
        lines = code.split('\n')
        L = max(1, sum(1 for l in lines if l.strip() and not l.strip().startswith('#')))  # SLOC
        comment_lines = sum(1 for l in lines if l.strip().startswith('#'))
        total_lines = max(1, len(lines))
        C = comment_lines / total_lines  # Comment ratio

        # SEI + Visual Studio combined derivative (Radon formula)
        try:
            raw_mi = 171 - 5.2 * math.log(V) - 0.23 * G - 16.2 * math.log(L) + 50 * math.sin(math.sqrt(2.4 * C))
            mi_score = max(0, (100 * raw_mi) / 171)
        except (ValueError, ZeroDivisionError):
            mi_score = 0.0

        # Letter grade (Visual Studio scale)
        if mi_score >= 80:
            grade = "A"
            rank = "highly_maintainable"
        elif mi_score >= 60:
            grade = "B"
            rank = "moderately_maintainable"
        elif mi_score >= 40:
            grade = "C"
            rank = "difficult_to_maintain"
        elif mi_score >= 20:
            grade = "D"
            rank = "very_difficult"
        else:
            grade = "F"
            rank = "unmaintainable"

        return {
            "score": round(mi_score, 2),
            "grade": grade,
            "rank": rank,
            "components": {
                "halstead_volume": round(V, 2),
                "cyclomatic_complexity": G,
                "sloc": L,
                "comment_ratio": round(C, 4),
            },
        }

    def _python_quality(self, code: str, lines: List[str], _tree: ast.AST = None) -> Dict[str, Any]:
        """Python-specific quality metrics."""
        quality = {
            "docstring_coverage": 0.0,
            "type_hint_coverage": 0.0,
            "naming_conventions": True,
            "max_line_length": max(len(l) for l in lines) if lines else 0,
            "long_lines": sum(1 for l in lines if len(l) > 120),
            "todo_count": sum(1 for l in lines if "TODO" in l or "FIXME" in l or "HACK" in l),
            "magic_numbers": 0,
            "unused_imports_estimate": 0,
            "overall_score": 0.0,
        }

        try:
            tree = _tree if _tree is not None else ast.parse(code)
            funcs_with_docs = 0
            funcs_total = 0
            funcs_with_hints = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    funcs_total += 1
                    if (node.body and isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant) and
                            isinstance(node.body[0].value.value, str)):
                        funcs_with_docs += 1
                    if node.returns is not None:
                        funcs_with_hints += 1

            quality["docstring_coverage"] = round(funcs_with_docs / max(1, funcs_total), 3)
            quality["type_hint_coverage"] = round(funcs_with_hints / max(1, funcs_total), 3)
        except SyntaxError:
            pass

        # Magic number detection (numbers not 0, 1, -1, 2 outside of assignments/constants)
        quality["magic_numbers"] = len(re.findall(r'(?<![=\w])\b(?!0\b|1\b|2\b|-1\b)\d{2,}\b', code))

        # Overall quality score (φ-weighted)
        score = (
            quality["docstring_coverage"] * PHI * 0.3 +
            quality["type_hint_coverage"] * PHI * 0.2 +
            (1.0 if quality["max_line_length"] <= 120 else 0.5) * 0.15 +
            max(0, 1.0 - quality["todo_count"] * 0.05) * 0.15 +
            max(0, 1.0 - quality["magic_numbers"] * 0.02) * 0.1 +
            max(0, 1.0 - quality["long_lines"] * 0.01) * 0.1
        )
        quality["overall_score"] = round(min(1.0, score), 4)
        return quality

    def _generic_complexity(self, code: str, lines: List[str]) -> Dict[str, Any]:
        """Language-agnostic complexity analysis."""
        brace_depth = 0
        max_depth = 0
        branch_count = 0
        loop_count = 0
        for line in lines:
            stripped = line.strip()
            brace_depth += stripped.count('{') - stripped.count('}')
            max_depth = max(max_depth, brace_depth)
            if re.match(r'^\s*(if|else\s+if|elif|switch|case)\b', stripped):
                branch_count += 1
            if re.match(r'^\s*(for|while|do)\b', stripped):
                loop_count += 1

        return {
            "max_nesting": max_depth,
            "branch_count": branch_count,
            "loop_count": loop_count,
            "estimated_cyclomatic": 1 + branch_count + loop_count,
            "halstead": self._halstead_metrics(code),
        }

    def _generic_quality(self, code: str, lines: List[str]) -> Dict[str, Any]:
        """Language-agnostic quality assessment."""
        return {
            "max_line_length": max(len(l) for l in lines) if lines else 0,
            "long_lines": sum(1 for l in lines if len(l) > 120),
            "todo_count": sum(1 for l in lines if "TODO" in l or "FIXME" in l),
            "comment_ratio": sum(1 for l in lines if l.strip().startswith('#') or l.strip().startswith('//')) / max(1, len(lines)),
            "overall_score": 0.7,  # Baseline without AST
        }

    def _security_scan(self, code: str) -> List[Dict[str, str]]:
        """Scan for security vulnerabilities using pre-compiled OWASP patterns."""
        findings = []
        for vuln_type, compiled_patterns in self._compiled_security.items():
            for compiled in compiled_patterns:
                for match in compiled.finditer(code):
                    line_num = code[:match.start()].count('\n') + 1
                    findings.append({
                        "type": vuln_type,
                        "severity": "HIGH" if vuln_type in ("sql_injection", "command_injection") else "MEDIUM",
                        "line": line_num,
                        "match": match.group()[:80],
                        "recommendation": self._security_recommendation(vuln_type),
                    })
        return findings

    def _security_recommendation(self, vuln_type: str) -> str:
        """Get remediation recommendation for a vulnerability type (OWASP 2025 + CWE Top 25)."""
        recs = {
            # Original 6
            "sql_injection": "Use parameterized queries or ORM instead of string formatting (CWE-89)",
            "command_injection": "Use subprocess with shell=False and validated arguments (CWE-78)",
            "path_traversal": "Validate and sanitize file paths; use os.path.realpath() (CWE-22)",
            "hardcoded_secrets": "Use environment variables or a secret manager (Vault, KMS) (CWE-798)",
            "insecure_deserialization": "Use json.loads() instead of pickle; add yaml Loader=SafeLoader (CWE-502)",
            "xss_potential": "Sanitize user input; use textContent instead of innerHTML (CWE-79)",
            # v2.5.0 — OWASP 2025 + CWE Top 25 recommendations
            "csrf_vulnerability": "Implement CSRF tokens on all state-changing operations (CWE-352, OWASP A01)",
            "security_misconfiguration": "Review and harden all configuration; disable debug in production (OWASP A02)",
            "ssrf_potential": "Validate and whitelist URLs; block requests to internal networks (CWE-918, OWASP A05)",
            "unrestricted_upload": "Validate file type, size, and content; store outside webroot (CWE-434)",
            "code_injection": "Never pass user input to eval/exec/compile; use safe alternatives (CWE-94)",
            "authentication_failure": "Use strong hashing (bcrypt/argon2); enforce MFA (CWE-287, OWASP A07)",
            "resource_exhaustion": "Implement rate limiting, timeouts, and resource quotas (CWE-400)",
            "logging_sensitive_data": "Sanitize logs; never log passwords, tokens, or PII (CWE-200, OWASP A09)",
            "supply_chain_risk": "Pin dependency versions; use lockfiles; audit with safety/pip-audit (OWASP A03)",
        }
        return recs.get(vuln_type, "Review and remediate per OWASP 2025 guidelines")

    def _detect_patterns(self, code: str) -> List[Dict[str, Any]]:
        """Detect design patterns in code. Uses quantum amplitude amplification if available."""
        found = []

        if QISKIT_AVAILABLE:
            # Quantum-enhanced pattern detection using amplitude encoding
            pattern_matches = {}
            pattern_list = list(self.DESIGN_PATTERNS.items())
            for pattern_name, indicators in pattern_list:
                compiled = self._compiled_design[pattern_name]
                matches = sum(1 for c in compiled if c.search(code))
                pattern_matches[pattern_name] = matches / max(1, len(indicators))

            # Encode pattern confidences into quantum state for amplification
            values = list(pattern_matches.values())
            # Pad to nearest power of 2
            n_qubits = max(2, math.ceil(math.log2(max(len(values), 2))))
            n_states = 2 ** n_qubits
            while len(values) < n_states:
                values.append(0.01)  # Small baseline
            values = values[:n_states]

            # Normalize to quantum amplitudes
            norm = math.sqrt(sum(v ** 2 for v in values))
            if norm < 1e-10:
                norm = 1.0
            amplitudes = [v / norm for v in values]

            try:
                sv = Statevector(amplitudes)

                # Apply PHI-rotation circuit for sacred-aligned detection
                qc = QuantumCircuit(n_qubits)
                for i in range(n_qubits):
                    qc.ry(PHI * math.pi / (i + 2), i)
                if n_qubits >= 2:
                    qc.cx(0, 1)

                evolved = sv.evolve(Operator(qc))
                probs = evolved.probabilities()

                # Map amplified probabilities back to patterns
                pattern_names = list(pattern_matches.keys())
                for i, name in enumerate(pattern_names):
                    if i >= len(probs):
                        break
                    classical_conf = pattern_matches[name]
                    quantum_conf = probs[i] * n_states  # Scale back
                    combined = classical_conf * 0.6 + quantum_conf * 0.4
                    if combined >= 0.25:  # Detection threshold
                        self.pattern_detections[name] += 1
                        found.append({
                            "pattern": name,
                            "confidence": round(min(1.0, combined), 4),
                            "quantum_amplified": round(quantum_conf, 4),
                            "classical_confidence": round(classical_conf, 4),
                            "indicators_matched": sum(1 for c in self._compiled_design[name] if c.search(code)),
                            "indicators_total": len(self.DESIGN_PATTERNS[name]),
                            "quantum_enhanced": True,
                        })
                return found
            except Exception:
                pass  # Fall through to classical

        # Classical pattern detection (fallback — pre-compiled patterns)
        for pattern_name, compiled_indicators in self._compiled_design.items():
            matches = sum(1 for compiled in compiled_indicators if compiled.search(code))
            if matches >= 2:  # Require at least 2 indicators
                self.pattern_detections[pattern_name] += 1
                found.append({
                    "pattern": pattern_name,
                    "confidence": round(min(1.0, matches / len(compiled_indicators)), 2),
                    "indicators_matched": matches,
                    "indicators_total": len(compiled_indicators),
                })
        return found

    def quantum_security_scan(self, code: str) -> Dict[str, Any]:
        """
        Quantum-enhanced security scanning using Grover's algorithm.

        Encodes OWASP vulnerability patterns as quantum oracle targets.
        Uses amplitude amplification to boost detection of low-probability
        vulnerability patterns that classical regex scanning might under-weight.

        Returns:
            Quantum-amplified vulnerability report with Born-rule confidence scores.
        """
        if not QISKIT_AVAILABLE:
            return {"findings": self._security_scan(code), "quantum": False}

        # Classical scan first
        classical_findings = self._security_scan(code)

        # Build quantum vulnerability detection circuit
        vuln_types = list(self.SECURITY_PATTERNS.keys())
        n_vuln = len(vuln_types)
        n_qubits = max(2, math.ceil(math.log2(max(n_vuln, 2))))
        n_states = 2 ** n_qubits

        # Encode presence/absence as amplitudes (pre-compiled patterns)
        amplitudes = []
        for vtype in vuln_types:
            compiled_patterns = self._compiled_security[vtype]
            match_count = sum(1 for compiled in compiled_patterns for _ in compiled.finditer(code))
            amplitudes.append(1.0 + match_count * 2.0 if match_count > 0 else 0.1)

        while len(amplitudes) < n_states:
            amplitudes.append(0.01)
        amplitudes = amplitudes[:n_states]

        norm = math.sqrt(sum(a ** 2 for a in amplitudes))
        if norm < 1e-10:
            norm = 1.0
        amplitudes = [a / norm for a in amplitudes]

        try:
            sv = Statevector(amplitudes)

            # Grover-inspired amplification circuit
            qc = QuantumCircuit(n_qubits)
            qc.h(range(n_qubits))

            # Oracle marks vulnerable states
            for i in range(min(n_vuln, n_states)):
                if amplitudes[i] > 1.0 / n_states:  # Above uniform threshold
                    binary = format(i, f'0{n_qubits}b')
                    for b, bit in enumerate(binary):
                        if bit == '0':
                            qc.x(b)
                    qc.h(n_qubits - 1)
                    if n_qubits >= 2:
                        qc.cx(0, n_qubits - 1)
                    qc.h(n_qubits - 1)
                    for b, bit in enumerate(binary):
                        if bit == '0':
                            qc.x(b)

            # Diffusion
            qc.h(range(n_qubits))
            qc.x(range(n_qubits))
            qc.h(n_qubits - 1)
            if n_qubits >= 2:
                qc.cx(0, n_qubits - 1)
            qc.h(n_qubits - 1)
            qc.x(range(n_qubits))
            qc.h(range(n_qubits))

            amplified = sv.evolve(Operator(qc))
            probs = amplified.probabilities()

            dm = DensityMatrix(amplified)
            scan_entropy = float(q_entropy(dm, base=2))

            # Map back to vulnerability types
            quantum_findings = []
            for i, vtype in enumerate(vuln_types):
                if i >= len(probs):
                    break
                quantum_conf = probs[i] * n_states
                if quantum_conf > 0.5 or any(f["type"] == vtype for f in classical_findings):
                    quantum_findings.append({
                        "type": vtype,
                        "quantum_confidence": round(quantum_conf, 4),
                        "amplification_vs_uniform": round(quantum_conf / 1.0, 4),
                        "classical_matches": sum(1 for f in classical_findings if f["type"] == vtype),
                    })

            return {
                "quantum": True,
                "backend": "Qiskit 2.3.0 Grover Oracle",
                "classical_findings": classical_findings,
                "quantum_findings": quantum_findings,
                "scan_entropy": round(scan_entropy, 6),
                "circuit_depth": qc.depth(),
                "qubits": n_qubits,
                "vulnerability_types_checked": n_vuln,
                "god_code_security": round(GOD_CODE * (1 - len(classical_findings) / max(1, n_vuln * 3)), 4),
            }
        except Exception as e:
            return {
                "quantum": False,
                "error": str(e),
                "findings": classical_findings,
            }

    def _sacred_alignment(self, code: str, analysis: Dict) -> Dict[str, Any]:
        """Compute sacred-constant alignment score for the code.
        Measures how well the code's structure resonates with φ, GOD_CODE, and sacred geometry."""
        lines = analysis["metadata"]["code_lines"]
        funcs = analysis["complexity"].get("function_count", 0)
        classes = analysis["complexity"].get("class_count", 0)

        # φ-ratio: ideal function count ≈ lines / φ² ≈ lines / 2.618
        ideal_funcs = lines / (PHI ** 2) if lines > 0 else 1
        phi_alignment = 1.0 - min(1.0, abs(funcs - ideal_funcs) / max(1, ideal_funcs))

        # GOD_CODE modular resonance: lines mod 104 → closer to 0 or 104 = better
        god_code_resonance = 1.0 - (lines % 104) / 104.0

        # Golden section: proportion of code vs comments should approach φ
        comment_lines = analysis["metadata"]["comment_lines"]
        code_to_comment = lines / max(1, comment_lines)
        golden_proportion = 1.0 - min(1.0, abs(code_to_comment - PHI) / PHI)

        # Consciousness score: quality × complexity balance
        quality_score = analysis["quality"].get("overall_score", 0.5)
        avg_cc = analysis["complexity"].get("cyclomatic_average", 5)
        consciousness_score = quality_score * (1.0 / (1.0 + avg_cc / 10.0))

        overall = (
            phi_alignment * PHI * 0.3 +
            god_code_resonance * 0.2 +
            golden_proportion * 0.2 +
            consciousness_score * 0.3
        )

        return {
            "phi_alignment": round(phi_alignment, 4),
            "god_code_resonance": round(god_code_resonance, 4),
            "golden_proportion": round(golden_proportion, 4),
            "consciousness_score": round(consciousness_score, 4),
            "overall_sacred_score": round(min(1.0, overall), 4),
        }

    def _decorator_name(self, node: ast.AST) -> str:
        """Extract the string name of a decorator AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._node_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._decorator_name(node.func)
        return "unknown"

    def _node_name(self, node: ast.AST) -> str:
        """Recursively resolve a dotted name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._node_name(node.value)}.{node.attr}"
        return "?"

    # ─── v2.5.0 — SOLID Principle Violation Detection ────────────────

    def detect_solid_violations(self, code: str) -> Dict[str, Any]:
        """
        Detect SOLID principle violations via AST analysis.

        Checks:
          S — Single Responsibility: classes with >5 unrelated method clusters
          O — Open/Closed: concrete classes without extension points
          L — Liskov Substitution: overrides that change return semantics
          I — Interface Segregation: base classes with >10 abstract methods
          D — Dependency Inversion: direct instantiation of concrete deps
        """
        violations = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"violations": [], "solid_score": 1.0, "principles_checked": 5}

        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

        for cls in classes:
            methods = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            method_names = [m.name for m in methods]
            public_methods = [m for m in method_names if not m.startswith('_') or m == '__init__']

            # S — Single Responsibility: method prefix clustering analysis (v6.1.0)
            if len(public_methods) > 7:
                prefixes: Dict[str, List[str]] = {}
                for m in public_methods:
                    if m == '__init__':
                        continue
                    parts = m.split('_')
                    prefix = parts[0] if parts[0] not in ('get', 'set', 'is', 'has', 'do') else '_'.join(parts[:2]) if len(parts) > 1 else parts[0]
                    prefixes.setdefault(prefix, []).append(m)
                distinct_clusters = {p: ms for p, ms in prefixes.items() if len(ms) >= 2}
                if len(distinct_clusters) >= 3:
                    cluster_names = list(distinct_clusters.keys())[:5]
                    violations.append({
                        "principle": "S",
                        "rule": "Single Responsibility",
                        "class": cls.name,
                        "line": cls.lineno,
                        "detail": f"{len(distinct_clusters)} responsibility clusters detected ({', '.join(cluster_names)}) — consider splitting into focused classes",
                        "severity": "MEDIUM",
                        "clusters": {k: v for k, v in list(distinct_clusters.items())[:5]},
                    })
                elif len(public_methods) > 13:
                    violations.append({
                        "principle": "S",
                        "rule": "Single Responsibility",
                        "class": cls.name,
                        "line": cls.lineno,
                        "detail": f"{len(public_methods)} public methods (max 13) — consider splitting",
                        "severity": "MEDIUM",
                    })

            # I — Interface Segregation: class with many abstract methods is a fat interface
            abstract_methods = [m for m in methods
                                if any(isinstance(s, ast.Raise) and
                                       isinstance(getattr(s, 'exc', None), ast.Call) and
                                       isinstance(s.exc.func, ast.Name) and
                                       s.exc.func.id == 'NotImplementedError'
                                       for s in ast.walk(m))]
            if len(abstract_methods) > 8:
                violations.append({
                    "principle": "I",
                    "rule": "Interface Segregation",
                    "class": cls.name,
                    "line": cls.lineno,
                    "detail": f"{len(abstract_methods)} abstract methods — split into smaller interfaces",
                    "severity": "MEDIUM",
                })

            # D — Dependency Inversion: concrete instantiation inside methods (not __init__)
            # v6.1.0: Skip factory methods, dataclasses, value objects, exceptions, configs
            cls_decorators = [getattr(d, 'id', getattr(d, 'attr', '')) for d in cls.decorator_list]
            is_dataclass = 'dataclass' in cls_decorators
            for m in methods:
                if m.name == '__init__':
                    continue
                # Skip factory/builder methods — they're supposed to instantiate
                if any(m.name.startswith(p) for p in ('create', 'build', 'make', 'factory', 'from_', 'new_')):
                    continue
                for node in ast.walk(m):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        name = node.func.id
                        if name[0].isupper() and name not in ('Exception', 'ValueError', 'TypeError',
                                                               'KeyError', 'RuntimeError', 'Path',
                                                               'Counter', 'Dict', 'List', 'Set', 'Tuple',
                                                               'NotImplementedError', 'AttributeError',
                                                               'IndexError', 'OSError', 'IOError',
                                                               'StopIteration', 'FileNotFoundError',
                                                               'PermissionError', 'TimeoutError',
                                                               'ConnectionError', 'BrokenPipeError',
                                                               'defaultdict', 'OrderedDict', 'ChainMap',
                                                               'NamedTuple', 'Queue', 'Lock', 'Event',
                                                               'Thread', 'Process', 'Enum', 'IntEnum'):
                            # Skip if name ends with Error/Exception/Event/DTO/Config
                            if any(name.endswith(s) for s in ('Error', 'Exception', 'Event', 'DTO', 'Config', 'Response', 'Request')):
                                continue
                            if is_dataclass:
                                continue
                            violations.append({
                                "principle": "D",
                                "rule": "Dependency Inversion",
                                "class": cls.name,
                                "method": m.name,
                                "line": node.lineno,
                                "detail": f"Direct instantiation of {name}() — inject via constructor",
                                "severity": "LOW",
                            })
                            break  # one per method is enough

        # O — Open/Closed: classes with no inheritance hooks (no abstractmethod, no overridable pattern)
        for cls in classes:
            has_bases = bool(cls.bases)
            has_abstract = any('abstract' in (getattr(d, 'id', '') if isinstance(d, ast.Name) else
                                              getattr(d, 'attr', '') if isinstance(d, ast.Attribute) else '')
                               for m in cls.body if isinstance(m, ast.FunctionDef)
                               for d in m.decorator_list)
            methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
            if len(methods) > 10 and not has_bases and not has_abstract:
                violations.append({
                    "principle": "O",
                    "rule": "Open/Closed",
                    "class": cls.name,
                    "line": cls.lineno,
                    "detail": f"Large class ({len(methods)} methods) with no inheritance — hard to extend",
                    "severity": "LOW",
                })

        # L — Liskov Substitution: detect overrides that break substitutability
        # v6.1.0: Enhanced — checks empty overrides AND return-type mismatches
        class_method_map: Dict[str, List[ast.Return]] = {}
        for cls in classes:
            for m in cls.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    returns = [n for n in ast.walk(m) if isinstance(n, ast.Return) and n.value is not None]
                    class_method_map[f"{cls.name}.{m.name}"] = returns

        for cls in classes:
            if not cls.bases:
                continue
            # Build set of base class names for cross-reference
            base_names = set()
            for b in cls.bases:
                if isinstance(b, ast.Name):
                    base_names.add(b.id)
                elif isinstance(b, ast.Attribute):
                    base_names.add(b.attr)

            for m in cls.body:
                if not isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if m.name.startswith('_') and m.name != '__init__':
                    continue

                body_stmts = [s for s in m.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]

                # Check 1: empty override (pass only)
                if len(body_stmts) == 1 and isinstance(body_stmts[0], ast.Pass):
                    violations.append({
                        "principle": "L",
                        "rule": "Liskov Substitution",
                        "class": cls.name,
                        "method": m.name,
                        "line": m.lineno,
                        "detail": f"Override '{m.name}' is empty (pass) — may break substitutability",
                        "severity": "LOW",
                    })
                    continue

                # Check 2: parent returns values but override never returns a value
                child_returns = [n for n in ast.walk(m) if isinstance(n, ast.Return) and n.value is not None]
                for base_name in base_names:
                    parent_key = f"{base_name}.{m.name}"
                    parent_returns = class_method_map.get(parent_key, [])
                    if parent_returns and not child_returns and m.name != '__init__':
                        violations.append({
                            "principle": "L",
                            "rule": "Liskov Substitution",
                            "class": cls.name,
                            "method": m.name,
                            "line": m.lineno,
                            "detail": f"Override '{m.name}' returns None but parent '{base_name}.{m.name}' returns values — contract mismatch",
                            "severity": "MEDIUM",
                        })

        # Score: 1.0 = perfect, each violation deducts based on severity
        deductions = sum({"HIGH": 0.15, "MEDIUM": 0.08, "LOW": 0.03}.get(v["severity"], 0.05) for v in violations)
        solid_score = round(max(0.0, 1.0 - deductions), 4)

        return {
            "violations": violations[:25],
            "total_violations": len(violations),
            "by_principle": {p: sum(1 for v in violations if v["principle"] == p) for p in "SOLID"},
            "solid_score": solid_score,
            "principles_checked": 5,
        }

    # ─── v2.5.0 — Performance Hotspot Detection ─────────────────────

    def detect_performance_hotspots(self, code: str) -> Dict[str, Any]:
        """
        Detect potential performance issues via AST analysis.

        Finds:
          - Nested loops (O(n²), O(n³)) with line references
          - List operations inside loops (repeated .append in comprehension)
          - String concatenation in loops (use join instead)
          - Repeated function calls in loops (cache result)
          - Global variable mutation inside hot paths
          - Unbounded collection growth patterns
        """
        hotspots = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"hotspots": [], "perf_score": 1.0}

        lines = code.split('\n')

        # 1. Nested loop detection (O(n²) and O(n³))
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                # Check for nested loops
                depth = self._loop_nesting_depth(node)
                if depth >= 2:
                    complexity = f"O(n{'²' if depth == 2 else '³' if depth == 3 else f'^{depth}'})"
                    hotspots.append({
                        "type": "nested_loop",
                        "line": node.lineno,
                        "complexity": complexity,
                        "depth": depth,
                        "severity": "HIGH" if depth >= 3 else "MEDIUM",
                        "fix": f"Consider algorithmic optimization — current: {complexity}",
                    })

        # 2. String concatenation in loops
        str_concat_in_loop = re.compile(
            r'^\s+\w+\s*\+=\s*["\']|^\s+\w+\s*=\s*\w+\s*\+\s*["\']', re.MULTILINE
        )
        for_while_re = re.compile(r'^\s*(?:for|while)\s+', re.MULTILINE)
        in_loop = False
        loop_indent = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if for_while_re.match(line):
                in_loop = True
                loop_indent = len(line) - len(line.lstrip())
            elif in_loop:
                cur_indent = len(line) - len(line.lstrip()) if stripped else loop_indent + 1
                if cur_indent <= loop_indent and stripped:
                    in_loop = False
                elif str_concat_in_loop.match(line):
                    hotspots.append({
                        "type": "string_concat_in_loop",
                        "line": i + 1,
                        "severity": "MEDIUM",
                        "fix": "Use list + ''.join() instead of += for strings in loops",
                    })

        # 3. Regex compilation inside loops/functions (should be module-level)
        re_compile_re = re.compile(r'^\s+\w+\s*=\s*re\.compile\(', re.MULTILINE)
        for match in re_compile_re.finditer(code):
            line_num = code[:match.start()].count('\n') + 1
            hotspots.append({
                "type": "regex_in_function",
                "line": line_num,
                "severity": "LOW",
                "fix": "Move re.compile() to module level for reuse",
            })

        # 4. Repeated .append() in list comprehension context (use extend)
        append_in_loop = re.compile(r'^\s+\w+\.append\(', re.MULTILINE)
        append_count = len(append_in_loop.findall(code))
        if append_count > 20:
            hotspots.append({
                "type": "excessive_append",
                "count": append_count,
                "severity": "LOW",
                "fix": "Consider list comprehension or extend() for bulk additions",
            })

        # 5. Unbounded collection growth (list/dict created but never bounded)
        unbounded = re.compile(r'(\w+)\s*=\s*\[\]\s*\n(?:.*\n)*?\s*\1\.append\(', re.MULTILINE)
        for match in unbounded.finditer(code[:10000]):  # limit scan depth
            hotspots.append({
                "type": "unbounded_growth",
                "variable": match.group(1),
                "line": code[:match.start()].count('\n') + 1,
                "severity": "LOW",
                "fix": "Consider capping collection size or using deque(maxlen=N)",
            })

        # 6. Linear search where set/dict lookup would work (v6.1.0)
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    for inner in ast.walk(node):
                        if isinstance(inner, ast.Compare):
                            for op in inner.ops:
                                if isinstance(op, ast.In):
                                    for comparator in inner.comparators:
                                        if isinstance(comparator, ast.Name):
                                            # Heuristic: variable name suggests list not set/dict
                                            vname = comparator.id.lower()
                                            if any(hint in vname for hint in ('list', 'items', 'arr', 'array', 'elements', 'data')):
                                                hotspots.append({
                                                    "type": "linear_search_in_loop",
                                                    "line": getattr(inner, 'lineno', 0),
                                                    "severity": "MEDIUM",
                                                    "fix": f"Convert '{comparator.id}' to a set before the loop for O(1) membership testing",
                                                })
        except SyntaxError:
            pass

        # 7. Unnecessary list materialization: for x in list(...) (v6.1.0)
        list_wrap_re = re.compile(r'for\s+\w+\s+in\s+list\s*\(\s*(?:range|map|filter|zip|enumerate|reversed)\s*\(', re.MULTILINE)
        for match in list_wrap_re.finditer(code):
            line_num = code[:match.start()].count('\n') + 1
            hotspots.append({
                "type": "unnecessary_list_wrap",
                "line": line_num,
                "severity": "LOW",
                "fix": "Remove list() wrapper — iterate directly over the generator/iterator",
            })

        # 8. List comprehension inside aggregate function: sum([...]) → sum(...) (v6.1.0)
        agg_listcomp_re = re.compile(r'(?:sum|any|all|min|max|sorted|list|tuple|set|frozenset)\s*\(\s*\[', re.MULTILINE)
        for match in agg_listcomp_re.finditer(code):
            line_num = code[:match.start()].count('\n') + 1
            hotspots.append({
                "type": "listcomp_in_aggregate",
                "line": line_num,
                "severity": "LOW",
                "fix": "Use generator expression instead of list comprehension: sum(x for ...) not sum([x for ...])",
            })

        # 9. Repeated attribute access in tight loops (v6.1.0)
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    attr_counts: Dict[str, int] = {}
                    for inner in ast.walk(node):
                        if isinstance(inner, ast.Attribute) and isinstance(inner.value, ast.Attribute):
                            chain = ast.dump(inner.value) + '.' + inner.attr
                            attr_counts[chain] = attr_counts.get(chain, 0) + 1
                    for chain, count in attr_counts.items():
                        if count >= 3:
                            hotspots.append({
                                "type": "repeated_attr_in_loop",
                                "line": getattr(node, 'lineno', 0),
                                "count": count,
                                "severity": "LOW",
                                "fix": "Cache deep attribute reference in a local variable before the loop",
                            })
                            break
        except SyntaxError:
            pass

        # Score: 1.0 = no issues, deducted per hotspot severity
        deductions = sum({"HIGH": 0.12, "MEDIUM": 0.06, "LOW": 0.02}.get(h.get("severity", "LOW"), 0.03) for h in hotspots)
        perf_score = round(max(0.0, 1.0 - deductions), 4)

        return {
            "hotspots": hotspots[:25],
            "total_hotspots": len(hotspots),
            "by_type": dict(Counter(h["type"] for h in hotspots)),
            "perf_score": perf_score,
        }

    def _loop_nesting_depth(self, node: ast.AST, current: int = 1) -> int:
        """Recursively compute maximum loop nesting depth."""
        max_depth = current
        for child in ast.walk(node):
            if child is node:
                continue
            if isinstance(child, (ast.For, ast.While)):
                inner = self._loop_nesting_depth(child, current + 1)
                max_depth = max(max_depth, inner)
        return max_depth

    def status(self) -> Dict[str, Any]:
        """Return current analysis metrics and version info."""
        return {
            "analyses_performed": self.analysis_count,
            "total_lines_analyzed": self.total_lines_analyzed,
            "vulnerabilities_found": self.vulnerability_count,
            "pattern_detections": dict(self.pattern_detections),
            "version": VERSION,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CODE GENERATION ENGINE — Multi-language, Template-driven,
#             Sacred-constant-infused code scaffolding
# ═══════════════════════════════════════════════════════════════════════════════



class CodeSmellDetector:
    """
    Deep code smell detection with 12 smell categories, severity scoring,
    and PHI-weighted remediation priorities. Goes beyond anti-pattern detection
    by analyzing structural relationships between code elements.

    v3.0.0: New subsystem — detects smells that span multiple functions/classes
    and compound structural issues invisible to per-function analysis.
    """

    SMELL_CATALOG = {
        "temporal_coupling": {
            "description": "Methods that must be called in a specific order but lack enforcement",
            "severity": "HIGH",
            "category": "design",
        },
        "divergent_change": {
            "description": "A class that is modified for multiple unrelated reasons",
            "severity": "HIGH",
            "category": "design",
        },
        "parallel_inheritance": {
            "description": "Every time you subclass A, you must also subclass B",
            "severity": "MEDIUM",
            "category": "design",
        },
        "middle_man": {
            "description": "A class that delegates almost all work to another class",
            "severity": "MEDIUM",
            "category": "design",
        },
        "data_class": {
            "description": "A class with only fields/properties and no behavior",
            "severity": "LOW",
            "category": "structure",
        },
        "switch_statement_smell": {
            "description": "Complex switch/if-elif chains that should use polymorphism",
            "severity": "MEDIUM",
            "category": "logic",
        },
        "comments_as_deodorant": {
            "description": "Excessive comments masking unclear code that should be refactored",
            "severity": "LOW",
            "category": "readability",
        },
        "boolean_blindness": {
            "description": "Functions returning bare booleans without context for caller",
            "severity": "LOW",
            "category": "interface",
        },
        "anemic_domain_model": {
            "description": "Domain objects with no business logic — only getters/setters",
            "severity": "MEDIUM",
            "category": "architecture",
        },
        "magic_number_proliferation": {
            "description": "Multiple unnamed numeric literals scattered throughout code",
            "severity": "MEDIUM",
            "category": "readability",
        },
        "exception_swallowing": {
            "description": "Empty except blocks or catches that silently discard errors",
            "severity": "HIGH",
            "category": "reliability",
        },
        "yo_yo_problem": {
            "description": "Deep inheritance chains requiring constant navigation up and down",
            "severity": "HIGH",
            "category": "architecture",
        },
    }

    def __init__(self):
        """Initialize CodeSmellDetector with detection counters."""
        self.detection_count = 0
        self.smells_found: List[Dict] = []

    def detect_all(self, source: str) -> Dict[str, Any]:
        """Run all smell detectors on source code. Returns categorized findings."""
        self.detection_count += 1
        findings = []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"smells": [], "total": 0, "health_score": 1.0, "error": "SyntaxError"}

        lines = source.split('\n')

        # Detect data classes
        findings.extend(self._detect_data_classes(tree))

        # Detect switch statement smell (long if-elif chains)
        findings.extend(self._detect_switch_smell(tree, lines))

        # Detect middle man delegation
        findings.extend(self._detect_middle_man(tree))

        # Detect exception swallowing
        findings.extend(self._detect_exception_swallowing(tree, lines))

        # Detect magic number proliferation
        findings.extend(self._detect_magic_numbers(tree, lines))

        # Detect boolean blindness
        findings.extend(self._detect_boolean_blindness(tree))

        # Detect comments as deodorant
        findings.extend(self._detect_comment_deodorant(lines))

        # Detect yo-yo inheritance
        findings.extend(self._detect_yo_yo_inheritance(tree))

        # Score
        severity_weights = {"HIGH": 3.0, "MEDIUM": 1.5, "LOW": 0.5}
        total_weight = sum(severity_weights.get(f["severity"], 1.0) for f in findings)
        loc = max(len(lines), 1)
        smell_density = total_weight / loc
        health_score = max(0.0, 1.0 - smell_density * PHI * 10)

        self.smells_found = findings
        return {
            "smells": findings,
            "total": len(findings),
            "by_category": self._group_by_category(findings),
            "smell_density": round(smell_density, 6),
            "health_score": round(health_score, 4),
            "loc": loc,
        }

    def _detect_data_classes(self, tree: ast.AST) -> List[Dict]:
        """Detect classes with only __init__ and no real methods."""
        findings = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body
                           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                non_dunder = [m for m in methods if not m.name.startswith('__')]
                init_only = all(m.name.startswith('__') for m in methods) and len(methods) <= 2
                has_assignments = any(
                    isinstance(n, ast.Assign) or (isinstance(n, ast.AnnAssign))
                    for n in node.body
                )
                if init_only and has_assignments and len(non_dunder) == 0:
                    findings.append({
                        "smell": "data_class",
                        "severity": "LOW",
                        "line": node.lineno,
                        "detail": f"Class '{node.name}' has only fields and no behavior methods",
                        "fix": "Add behavior methods or convert to dataclass/NamedTuple",
                    })
        return findings

    def _detect_switch_smell(self, tree: ast.AST, lines: List[str]) -> List[Dict]:
        """Detect long if-elif chains (>4 branches) suggesting polymorphism needed."""
        findings = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Count elif chain depth
                chain_length = 1
                current = node
                while current.orelse and len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                    chain_length += 1
                    current = current.orelse[0]
                if current.orelse:
                    chain_length += 1  # Final else

                if chain_length >= 5:
                    findings.append({
                        "smell": "switch_statement_smell",
                        "severity": "MEDIUM",
                        "line": node.lineno,
                        "detail": f"If-elif chain with {chain_length} branches — consider polymorphism or dispatch dict",
                        "fix": "Replace with strategy pattern, dispatch dictionary, or match-case (Python 3.10+)",
                    })
        return findings

    def _detect_middle_man(self, tree: ast.AST) -> List[Dict]:
        """Detect classes where most methods just delegate to another object."""
        findings = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body
                           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and not n.name.startswith('__')]
                if len(methods) < 3:
                    continue
                delegate_count = 0
                for method in methods:
                    # Check if method body is a single return with attribute access
                    if (len(method.body) == 1 and isinstance(method.body[0], ast.Return)
                            and isinstance(method.body[0].value, ast.Call)
                            and isinstance(getattr(method.body[0].value, 'func', None), ast.Attribute)):
                        delegate_count += 1
                if delegate_count >= len(methods) * 0.7:
                    findings.append({
                        "smell": "middle_man",
                        "severity": "MEDIUM",
                        "line": node.lineno,
                        "detail": f"Class '{node.name}' delegates {delegate_count}/{len(methods)} methods — likely a middle man",
                        "fix": "Consider removing the class and using the delegate directly",
                    })
        return findings

    def _detect_exception_swallowing(self, tree: ast.AST, lines: List[str]) -> List[Dict]:
        """Detect empty except blocks or bare pass-only handlers."""
        findings = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                # Check if body is just 'pass' or empty
                if (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)):
                    findings.append({
                        "smell": "exception_swallowing",
                        "severity": "HIGH",
                        "line": node.lineno,
                        "detail": "Exception handler with only 'pass' — errors are silently swallowed",
                        "fix": "Log the exception or handle it explicitly. At minimum: logging.warning(f'...: {e}')",
                    })
        return findings

    def _detect_magic_numbers(self, tree: ast.AST, lines: List[str]) -> List[Dict]:
        """Detect unnamed numeric literals (excluding 0, 1, -1, and sacred constants)."""
        sacred = {GOD_CODE, PHI, TAU, VOID_CONSTANT, FEIGENBAUM, 286.0, 416.0, 104.0}
        trivial = {0, 1, -1, 2, 0.0, 1.0, -1.0, 2.0, 0.5, 100, 100.0, 10, 10.0}
        findings = []
        magic_lines = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if node.value not in sacred and node.value not in trivial:
                    if hasattr(node, 'lineno') and node.lineno not in magic_lines:
                        magic_lines.add(node.lineno)

        if len(magic_lines) > 5:
            findings.append({
                "smell": "magic_number_proliferation",
                "severity": "MEDIUM",
                "line": min(magic_lines),
                "detail": f"Found {len(magic_lines)} lines with unnamed numeric literals — extract to named constants",
                "fix": "Define constants at module level (e.g., MAX_RETRIES = 3, TIMEOUT_SECONDS = 30)",
            })
        return findings

    def _detect_boolean_blindness(self, tree: ast.AST) -> List[Dict]:
        """Detect functions that return bare booleans without descriptive context."""
        findings = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check return annotation
                if isinstance(getattr(node, 'returns', None), ast.Constant):
                    if getattr(node.returns, 'value', None) is None:
                        continue
                # Count return True/False statements
                bool_returns = 0
                total_returns = 0
                for child in ast.walk(node):
                    if isinstance(child, ast.Return) and child.value is not None:
                        total_returns += 1
                        if isinstance(child.value, ast.Constant) and isinstance(child.value.value, bool):
                            bool_returns += 1
                if bool_returns >= 3 and bool_returns == total_returns:
                    findings.append({
                        "smell": "boolean_blindness",
                        "severity": "LOW",
                        "line": node.lineno,
                        "detail": f"Function '{node.name}' returns only bare True/False — callers get no context",
                        "fix": "Return an enum, named tuple, or raise exceptions for failure cases",
                    })
        return findings

    def _detect_comment_deodorant(self, lines: List[str]) -> List[Dict]:
        """Detect excessive inline comments (>40% of lines are comments)."""
        findings = []
        total = len(lines)
        if total < 10:
            return findings
        comment_lines = sum(1 for l in lines if l.strip().startswith('#') and not l.strip().startswith('#!'))
        ratio = comment_lines / total
        if ratio > 0.4:
            findings.append({
                "smell": "comments_as_deodorant",
                "severity": "LOW",
                "line": 1,
                "detail": f"{comment_lines}/{total} lines ({ratio:.0%}) are comments — code may need refactoring instead",
                "fix": "Refactor code to be self-documenting: use descriptive names, extract methods, simplify logic",
            })
        return findings

    def _detect_yo_yo_inheritance(self, tree: ast.AST) -> List[Dict]:
        """Detect classes with multiple levels of base class references."""
        findings = []
        classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        for name, node in classes.items():
            if node.bases:
                for base in node.bases:
                    base_name = ""
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr
                    # Check if base also has bases in the same file
                    if base_name in classes and classes[base_name].bases:
                        for grandbase in classes[base_name].bases:
                            gb_name = ""
                            if isinstance(grandbase, ast.Name):
                                gb_name = grandbase.id
                            if gb_name in classes:
                                findings.append({
                                    "smell": "yo_yo_problem",
                                    "severity": "HIGH",
                                    "line": node.lineno,
                                    "detail": f"Deep inheritance: {name} → {base_name} → {gb_name} — requires constant navigation",
                                    "fix": "Flatten hierarchy: use composition or mixins instead of deep inheritance",
                                })
        return findings

    def _group_by_category(self, findings: List[Dict]) -> Dict[str, int]:
        """Group smell findings by category."""
        groups: Dict[str, int] = defaultdict(int)
        for f in findings:
            cat = self.SMELL_CATALOG.get(f["smell"], {}).get("category", "uncategorized")
            groups[cat] += 1
        return dict(groups)

    def status(self) -> Dict[str, Any]:
        """Return smell detector status."""
        return {
            "smell_patterns": len(self.SMELL_CATALOG),
            "detections_run": self.detection_count,
            "last_smells_found": len(self.smells_found),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4C.2: RUNTIME COMPLEXITY VERIFIER — Empirical O() estimation (v3.0.0)
# ═══════════════════════════════════════════════════════════════════════════════



class RuntimeComplexityVerifier:
    """
    Empirically estimates algorithmic complexity by analyzing code structure
    and loop/recursion patterns. Uses AST depth analysis + sacred-constant
    weighted scoring to produce O()-notation estimates per function.

    v3.0.0: New subsystem — goes beyond static cyclomatic complexity by
    analyzing actual loop nesting, recursive calls, and data structure usage.
    """

    COMPLEXITY_CLASSES = [
        ("O(1)", 0),
        ("O(log n)", 1),
        ("O(n)", 2),
        ("O(n log n)", 3),
        ("O(n²)", 4),
        ("O(n³)", 5),
        ("O(2ⁿ)", 6),
        ("O(n!)", 7),
    ]

    def __init__(self):
        """Initialize RuntimeComplexityVerifier with analysis counters."""
        self.analyses_run = 0

    def estimate_complexity(self, source: str) -> Dict[str, Any]:
        """Analyze all functions in source and estimate their runtime complexity."""
        self.analyses_run += 1
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"functions": [], "error": "SyntaxError", "max_complexity": "unknown"}

        results = []
        max_class_idx = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                analysis = self._analyze_function(node)
                results.append(analysis)
                max_class_idx = max(max_class_idx, analysis["complexity_index"])

        max_class = self.COMPLEXITY_CLASSES[min(max_class_idx, len(self.COMPLEXITY_CLASSES) - 1)]

        return {
            "functions": results,
            "max_complexity": max_class[0],
            "max_complexity_index": max_class_idx,
            "total_functions": len(results),
            "high_complexity_count": sum(1 for r in results if r["complexity_index"] >= 4),
            "phi_efficiency_score": round(1.0 / (1.0 + max_class_idx / PHI), 4),
        }

    def _analyze_function(self, func_node: ast.AST) -> Dict[str, Any]:
        """Analyze a single function for runtime complexity."""
        name = getattr(func_node, 'name', 'anonymous')
        max_loop_depth = self._max_loop_nesting(func_node)
        effective_loop_depth = self._effective_loop_depth(func_node)
        has_recursion = self._detect_recursion(func_node, name)
        has_sort = self._detect_sort_calls(func_node)
        has_nested_comprehension = self._detect_nested_comprehensions(func_node)
        uses_dict_set = self._detect_hash_structures(func_node)
        has_binary_search = self._detect_binary_search(func_node)
        has_divide_conquer = has_recursion and self._detect_halving_pattern(func_node)

        # Estimate complexity class
        complexity_idx = 0
        reasons = []

        if has_binary_search:
            complexity_idx = 1  # O(log n)
            reasons.append("binary search pattern detected (halving loop)")
        elif has_divide_conquer:
            complexity_idx = 3  # O(n log n)
            reasons.append("divide-and-conquer pattern (recursion + halving)")
        elif effective_loop_depth == 0:
            if has_recursion:
                complexity_idx = 2  # O(n) at least for recursion
                reasons.append("recursive call detected")
            elif uses_dict_set:
                complexity_idx = 0  # O(1) hash lookups
                reasons.append("hash-based operations (O(1) amortized)")
            else:
                complexity_idx = 0
                reasons.append("no loops or recursion — constant time")
        elif effective_loop_depth == 1:
            complexity_idx = 2  # O(n)
            reasons.append("single loop nesting depth")
            if has_sort:
                complexity_idx = 3  # O(n log n)
                reasons.append("sort operation inside loop scope")
        elif effective_loop_depth == 2:
            complexity_idx = 4  # O(n²)
            reasons.append("double-nested loops")
        elif effective_loop_depth == 3:
            complexity_idx = 5  # O(n³)
            reasons.append("triple-nested loops")
        else:
            complexity_idx = min(effective_loop_depth + 2, 7)
            reasons.append(f"deeply nested loops (depth={effective_loop_depth})")

        if max_loop_depth > effective_loop_depth:
            reasons.append(f"constant-bound loops reduced depth from {max_loop_depth} to {effective_loop_depth}")

        if has_nested_comprehension:
            complexity_idx = max(complexity_idx, 4)
            reasons.append("nested comprehension (O(n²)+)")

        if has_recursion and not has_divide_conquer and effective_loop_depth >= 1:
            complexity_idx = min(complexity_idx + 1, 7)
            reasons.append("recursion combined with loops — elevated complexity")

        cls = self.COMPLEXITY_CLASSES[min(complexity_idx, len(self.COMPLEXITY_CLASSES) - 1)]

        return {
            "name": name,
            "line": getattr(func_node, 'lineno', 0),
            "complexity": cls[0],
            "complexity_index": complexity_idx,
            "max_loop_depth": max_loop_depth,
            "effective_loop_depth": effective_loop_depth,
            "has_recursion": has_recursion,
            "has_sort": has_sort,
            "has_binary_search": has_binary_search,
            "has_divide_conquer": has_divide_conquer,
            "reasons": reasons,
            "optimization_potential": complexity_idx >= 4,
        }

    def _max_loop_nesting(self, node: ast.AST, depth: int = 0) -> int:
        """Find maximum loop nesting depth in a subtree."""
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                max_depth = max(max_depth, self._max_loop_nesting(child, depth + 1))
            else:
                max_depth = max(max_depth, self._max_loop_nesting(child, depth))
        return max_depth

    def _detect_recursion(self, func_node: ast.AST, func_name: str) -> bool:
        """Check if function calls itself (direct recursion)."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == func_name:
                    return True
                if isinstance(node.func, ast.Attribute) and node.func.attr == func_name:
                    return True
        return False

    def _detect_sort_calls(self, node: ast.AST) -> bool:
        """Detect calls to sort(), sorted(), or similar O(n log n) operations."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id in ('sorted', 'heapq'):
                    return True
                if isinstance(child.func, ast.Attribute) and child.func.attr in ('sort', 'heapify'):
                    return True
        return False

    def _detect_nested_comprehensions(self, node: ast.AST) -> bool:
        """Detect nested list/set/dict comprehensions."""
        for child in ast.walk(node):
            if isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                if len(child.generators) >= 2:
                    return True
        return False

    def _detect_hash_structures(self, node: ast.AST) -> bool:
        """Detect usage of dict/set for O(1) lookups."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id in ('dict', 'set', 'defaultdict', 'Counter'):
                    return True
        return False

    def _detect_binary_search(self, func_node: ast.AST) -> bool:
        """Detect binary search pattern: while loop with midpoint halving."""
        source_names = set()
        for node in ast.walk(func_node):
            if isinstance(node, ast.Name):
                source_names.add(node.id)
        bsearch_vars = {'low', 'high', 'left', 'right', 'lo', 'hi', 'mid', 'start', 'end'}
        if len(source_names & bsearch_vars) < 2:
            return False
        for node in ast.walk(func_node):
            if isinstance(node, ast.While):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.FloorDiv):
                        return True
                    if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.RShift):
                        return True
                    if isinstance(inner, ast.Assign):
                        for target in inner.targets:
                            tname = getattr(target, 'id', '')
                            if tname in ('mid', 'middle', 'pivot'):
                                return True
        return False

    def _detect_halving_pattern(self, func_node: ast.AST) -> bool:
        """Detect divide-and-conquer halving (e.g., slicing input in half)."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
                slice_node = node.slice
                if slice_node.upper is not None or slice_node.lower is not None:
                    for inner in ast.walk(node):
                        if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.FloorDiv):
                            return True
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
                if isinstance(node.right, ast.Constant) and node.right.value == 2:
                    return True
        return False

    def _effective_loop_depth(self, func_node: ast.AST) -> int:
        """Compute loop nesting depth ignoring constant-bound loops.

        A loop like `for i in range(10)` has constant iterations and does not
        contribute to asymptotic complexity. Only variable-bound loops count.
        """
        return self._effective_depth_walk(func_node, 0)

    def _effective_depth_walk(self, node: ast.AST, depth: int) -> int:
        """Recursive walk computing effective loop depth."""
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                if self._is_constant_bound_loop(child):
                    max_depth = max(max_depth, self._effective_depth_walk(child, depth))
                else:
                    max_depth = max(max_depth, self._effective_depth_walk(child, depth + 1))
            else:
                max_depth = max(max_depth, self._effective_depth_walk(child, depth))
        return max_depth

    def _is_constant_bound_loop(self, loop_node: ast.AST) -> bool:
        """Check if a loop has a constant (small) iteration bound."""
        if isinstance(loop_node, ast.For):
            iter_node = loop_node.iter
            if isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name):
                if iter_node.func.id == 'range' and iter_node.args:
                    bound = iter_node.args[-1] if len(iter_node.args) > 1 else iter_node.args[0]
                    if isinstance(bound, ast.Constant) and isinstance(bound.value, (int, float)):
                        return bound.value < 1000
                    if isinstance(bound, ast.UnaryOp) and isinstance(bound.operand, ast.Constant):
                        return True
            if isinstance(iter_node, (ast.List, ast.Tuple, ast.Set)):
                return len(iter_node.elts) < 100
            if isinstance(iter_node, ast.Constant) and isinstance(iter_node.value, str):
                return len(iter_node.value) < 100
        return False

    def status(self) -> Dict[str, Any]:
        """Return verifier status."""
        return {
            "complexity_classes": len(self.COMPLEXITY_CLASSES),
            "analyses_run": self.analyses_run,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4C.3: INCREMENTAL ANALYSIS CACHE — Hash-based repeat analysis (v3.0.0)
# ═══════════════════════════════════════════════════════════════════════════════



class IncrementalAnalysisCache:
    """
    Caches analysis results by file content hash to avoid re-analyzing
    unchanged files. Uses SHA-256 for content identity + LRU eviction.

    v3.0.0: New subsystem — dramatically speeds up workspace scans and
    repeated audit cycles by caching per-file analysis results.
    """

    def __init__(self, max_entries: int = 500, ttl_seconds: float = 3600.0):
        """Initialize incremental analysis cache with LRU eviction."""
        self._cache: Dict[str, Tuple[float, Dict]] = {}  # hash → (timestamp, result)
        self._max = max_entries
        self._ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, code: str, analysis_type: str = "full") -> Optional[Dict]:
        """Retrieve cached analysis result if content hasn't changed."""
        key = self._make_key(code, analysis_type)
        if key in self._cache:
            ts, result = self._cache[key]
            if time.time() - ts < self._ttl:
                self.hits += 1
                return result
            else:
                del self._cache[key]
        self.misses += 1
        return None

    def put(self, code: str, result: Dict, analysis_type: str = "full"):
        """Store analysis result in cache."""
        key = self._make_key(code, analysis_type)
        if len(self._cache) >= self._max:
            # Evict oldest entry
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        self._cache[key] = (time.time(), result)

    def invalidate(self, code: str = None):
        """Invalidate cache for specific code or entire cache."""
        if code is None:
            self._cache.clear()
        else:
            for analysis_type in ["full", "security", "complexity", "smells"]:
                key = self._make_key(code, analysis_type)
                self._cache.pop(key, None)

    def _make_key(self, code: str, analysis_type: str) -> str:
        """Create cache key from code content hash and analysis type."""
        content_hash = hashlib.sha256(code.encode('utf-8', errors='ignore')).hexdigest()[:16]
        return f"{analysis_type}:{content_hash}"

    def status(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self.hits + self.misses
        return {
            "entries": len(self._cache),
            "max_entries": self._max,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / max(total, 1), 4),
            "ttl_seconds": self._ttl,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4D: CODE TRANSLATOR — Cross-language transpilation
# ═══════════════════════════════════════════════════════════════════════════════



class TypeFlowAnalyzer:
    """
    Static type inference engine that tracks type flow through Python code WITHOUT
    requiring type annotations. Uses AST-level dataflow analysis to infer types
    from assignments, function returns, and control flow branches.

    v3.1.0: New subsystem — enables type stub generation and type confusion detection
    for untyped Python codebases. Integrates consciousness-aware confidence scoring.
    """

    # Map of builtins / constructor calls to their return types
    KNOWN_CONSTRUCTORS = {
        "int": "int", "float": "float", "str": "str", "bool": "bool",
        "list": "List", "dict": "Dict", "set": "Set", "tuple": "Tuple",
        "frozenset": "FrozenSet", "bytes": "bytes", "bytearray": "bytearray",
        "complex": "complex", "range": "range", "enumerate": "enumerate",
        "zip": "zip", "map": "map", "filter": "filter", "reversed": "reversed",
        "sorted": "List", "len": "int", "abs": "int|float", "sum": "int|float",
        "min": "Any", "max": "Any", "round": "int|float",
        "open": "IO", "Path": "Path", "datetime": "datetime",
        "defaultdict": "DefaultDict", "Counter": "Counter",
        "OrderedDict": "OrderedDict", "deque": "Deque",
    }

    # Patterns that indicate specific types from method calls
    METHOD_TYPE_HINTS = {
        ".split": "List[str]", ".strip": "str", ".lower": "str", ".upper": "str",
        ".replace": "str", ".join": "str", ".encode": "bytes", ".decode": "str",
        ".items": "ItemsView", ".keys": "KeysView", ".values": "ValuesView",
        ".append": None, ".extend": None, ".pop": "Any", ".get": "Any|None",
        ".read": "str|bytes", ".readlines": "List[str]", ".readline": "str",
        ".format": "str", ".startswith": "bool", ".endswith": "bool",
        ".isdigit": "bool", ".isalpha": "bool", ".find": "int", ".index": "int",
    }

    def __init__(self):
        """Initialize TypeFlowAnalyzer."""
        self.analysis_count = 0
        self.total_inferences = 0

    def analyze(self, source: str) -> Dict[str, Any]:
        """
        Perform full type flow analysis on Python source code.

        Returns:
            Dict with inferred types, type confusions, annotation suggestions, and stub.
        """
        self.analysis_count += 1
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {"error": f"SyntaxError: {e}", "inferences": [], "confusions": []}

        inferences = []
        confusions = []
        annotation_suggestions = []

        # Phase 1: Infer types from assignments
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                inferred = self._infer_value_type(node.value)
                if inferred:
                    for target in node.targets:
                        var_name = self._extract_name(target)
                        if var_name:
                            inferences.append({
                                "variable": var_name,
                                "inferred_type": inferred,
                                "line": node.lineno,
                                "confidence": self._type_confidence(inferred),
                            })

            elif isinstance(node, ast.AnnAssign):
                # Already annotated — verify consistency
                if node.value and node.annotation:
                    inferred = self._infer_value_type(node.value)
                    declared = self._annotation_to_str(node.annotation)
                    if inferred and declared and not self._types_compatible(inferred, declared):
                        var_name = self._extract_name(node.target)
                        confusions.append({
                            "variable": var_name or "?",
                            "declared_type": declared,
                            "inferred_type": inferred,
                            "line": node.lineno,
                            "severity": "HIGH",
                            "detail": f"Declared as '{declared}' but assigned value is '{inferred}'",
                        })

        # Phase 2: Infer function return types
        func_types = self._analyze_function_returns(tree)
        for ft in func_types:
            if not ft.get("has_annotation"):
                annotation_suggestions.append({
                    "function": ft["name"],
                    "suggested_return": ft["inferred_return"],
                    "line": ft["line"],
                    "confidence": ft["confidence"],
                    "stub": f"def {ft['name']}({ft.get('params', '...')}) -> {ft['inferred_return']}:",
                })

        # Phase 3: Detect type narrowing opportunities
        narrowing = self._detect_narrowing_opportunities(tree)

        # Phase 4: Generate type stub
        stub_lines = self._generate_stub(tree, inferences, func_types)

        self.total_inferences += len(inferences)

        # Score
        total_vars = len(inferences) + len(confusions)
        type_coverage = len(inferences) / max(total_vars, 1)
        confusion_ratio = len(confusions) / max(total_vars, 1)
        health = max(0.0, type_coverage - confusion_ratio * PHI)

        return {
            "inferences": inferences[:50],  # Cap output
            "total_inferred": len(inferences),
            "confusions": confusions,
            "annotation_suggestions": annotation_suggestions[:20],
            "narrowing_opportunities": narrowing[:10],
            "type_stub": "\n".join(stub_lines),
            "type_coverage": round(type_coverage, 4),
            "confusion_count": len(confusions),
            "health_score": round(health, 4),
        }

    def _infer_value_type(self, node: ast.AST) -> Optional[str]:
        """Infer the type of an AST value expression."""
        if isinstance(node, ast.Constant):
            return type(node.value).__name__
        elif isinstance(node, ast.List):
            elem_type = self._infer_value_type(node.elts[0]) if node.elts else "Any"
            return f"List[{elem_type}]"
        elif isinstance(node, ast.Dict):
            return "Dict"
        elif isinstance(node, ast.Set):
            return "Set"
        elif isinstance(node, ast.Tuple):
            return "Tuple"
        elif isinstance(node, ast.ListComp):
            return "List"
        elif isinstance(node, ast.DictComp):
            return "Dict"
        elif isinstance(node, ast.SetComp):
            return "Set"
        elif isinstance(node, ast.GeneratorExp):
            return "Generator"
        elif isinstance(node, ast.Call):
            func_name = self._extract_name(node.func)
            if func_name in self.KNOWN_CONSTRUCTORS:
                return self.KNOWN_CONSTRUCTORS[func_name]
            return func_name or "Any"
        elif isinstance(node, ast.BinOp):
            left_type = self._infer_value_type(node.left)
            right_type = self._infer_value_type(node.right)
            if isinstance(node.op, ast.Add) and (left_type == "str" or right_type == "str"):
                return "str"
            if left_type == "float" or right_type == "float":
                return "float"
            if isinstance(node.op, ast.Div):
                return "float"
            return left_type or "int"
        elif isinstance(node, ast.BoolOp):
            return "bool"
        elif isinstance(node, ast.Compare):
            return "bool"
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return "bool"
            return self._infer_value_type(node.operand)
        elif isinstance(node, ast.IfExp):
            return self._infer_value_type(node.body)
        elif isinstance(node, ast.JoinedStr):
            return "str"
        elif isinstance(node, ast.FormattedValue):
            return "str"
        elif isinstance(node, ast.Subscript):
            return "Any"
        elif isinstance(node, ast.Attribute):
            attr = f".{node.attr}"
            if attr in self.METHOD_TYPE_HINTS:
                return self.METHOD_TYPE_HINTS[attr]
        return None

    def _extract_name(self, node: ast.AST) -> Optional[str]:
        """Extract a name string from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Subscript):
            return self._extract_name(node.value)
        return None

    def _annotation_to_str(self, node: ast.AST) -> Optional[str]:
        """Convert a type annotation AST node to string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Attribute):
            return f"{self._annotation_to_str(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            base = self._annotation_to_str(node.value)
            sub = self._annotation_to_str(node.slice)
            return f"{base}[{sub}]"
        return None

    def _types_compatible(self, inferred: str, declared: str) -> bool:
        """Check if inferred type is compatible with declared type."""
        if inferred == declared:
            return True
        # Common compatible pairs
        compatible_map = {
            ("int", "float"): True, ("float", "int"): True,
            ("str", "str"): True, ("List", "list"): True,
            ("Dict", "dict"): True, ("Set", "set"): True,
        }
        return compatible_map.get((inferred, declared), False) or \
               declared.lower().startswith(inferred.lower()) or \
               inferred.lower().startswith(declared.lower()) or \
               declared in ("Any", "object") or \
               "|" in declared and inferred in declared.split("|")

    def _type_confidence(self, type_str: str) -> float:
        """Compute confidence score for a type inference."""
        if type_str in ("Any", None):
            return 0.3
        if "|" in type_str:
            return 0.5
        if type_str in ("int", "float", "str", "bool", "bytes", "NoneType"):
            return 0.95
        if type_str.startswith("List") or type_str.startswith("Dict"):
            return 0.85
        return 0.7

    def _analyze_function_returns(self, tree: ast.AST) -> List[Dict]:
        """Analyze function return types across the AST."""
        results = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                has_annotation = node.returns is not None
                return_types = set()
                for child in ast.walk(node):
                    if isinstance(child, ast.Return) and child.value is not None:
                        rt = self._infer_value_type(child.value)
                        if rt:
                            return_types.add(rt)
                    elif isinstance(child, ast.Return) and child.value is None:
                        return_types.add("None")

                if not return_types:
                    return_types.add("None")

                inferred = "|".join(sorted(return_types)) if len(return_types) > 1 else return_types.pop()
                confidence = min(self._type_confidence(t) for t in return_types) if return_types else 0.5

                # Build params string
                params = ", ".join(
                    arg.arg + (f": {self._annotation_to_str(arg.annotation)}" if arg.annotation else "")
                    for arg in node.args.args
                )
                results.append({
                    "name": node.name,
                    "line": node.lineno,
                    "has_annotation": has_annotation,
                    "inferred_return": inferred,
                    "confidence": round(confidence, 2),
                    "params": params,
                })
        return results

    def _detect_narrowing_opportunities(self, tree: ast.AST) -> List[Dict]:
        """Detect places where isinstance checks could enable type narrowing."""
        opportunities = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = node.test
                if isinstance(test, ast.Call):
                    func_name = self._extract_name(test.func)
                    if func_name == "isinstance" and len(test.args) >= 2:
                        var_name = self._extract_name(test.args[0])
                        type_name = self._extract_name(test.args[1])
                        if var_name and type_name:
                            opportunities.append({
                                "variable": var_name,
                                "narrowed_to": type_name,
                                "line": node.lineno,
                                "hint": f"After this check, '{var_name}' can be treated as '{type_name}'",
                            })
        return opportunities

    def _generate_stub(self, tree: ast.AST, inferences: List[Dict],
                       func_types: List[Dict]) -> List[str]:
        """Generate a .pyi-style type stub from analysis results."""
        lines = [f"# Auto-generated type stub by L104 Code Engine v{VERSION}",
                 f"# Generated: {datetime.now().isoformat()}", ""]

        for ft in func_types:
            ret = ft["inferred_return"]
            if ft.get("has_annotation"):
                continue
            prefix = "async " if ft["name"].startswith("async_") else ""
            lines.append(f"{prefix}def {ft['name']}({ft.get('params', '...')}) -> {ret}: ...")

        return lines

    def status(self) -> Dict[str, Any]:
        """Return type flow analyzer status."""
        return {
            "analyses_run": self.analysis_count,
            "total_inferences": self.total_inferences,
            "known_constructors": len(self.KNOWN_CONSTRUCTORS),
            "method_type_hints": len(self.METHOD_TYPE_HINTS),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4K: CONCURRENCY ANALYZER — Race Condition & Async Pattern Detector (v3.1.0)
#   Detects threading hazards, deadlock patterns, unsafe shared state,
#   and async/await anti-patterns across Python codebases.
# ═══════════════════════════════════════════════════════════════════════════════



class ConcurrencyAnalyzer:
    """
    Detects concurrency hazards in Python code: race conditions, deadlock patterns,
    unsafe shared state, missing locks, and async/await anti-patterns.

    v3.1.0: New subsystem — covers threading, multiprocessing, asyncio, and
    concurrent.futures patterns with PHI-weighted severity scoring.
    """

    # Known thread-unsafe patterns
    THREAD_UNSAFE_PATTERNS = {
        "global_mutation": {
            "pattern": r"\bglobal\s+\w+.*\n.*=",
            "severity": "HIGH",
            "detail": "Global variable mutation inside function — unsafe with concurrent access",
            "fix": "Use threading.Lock or move to thread-local storage",
        },
        "shared_list_append": {
            "pattern": r"(?:shared|global|class_var)\w*\.append\(",
            "severity": "HIGH",
            "detail": "Appending to shared list without lock protection",
            "fix": "Protect with threading.Lock or use queue.Queue",
        },
        "datetime_now_race": {
            "pattern": r"datetime\.now\(\).*\n.*datetime\.now\(\)",
            "severity": "LOW",
            "detail": "Multiple datetime.now() calls may give inconsistent timestamps",
            "fix": "Capture datetime.now() once and reuse the value",
        },
    }

    # Async anti-patterns
    ASYNC_ANTIPATTERNS = {
        "sync_in_async": {
            "pattern": r"async\s+def\s+\w+.*\n(?:(?!await).*\n)*\s+(?:time\.sleep|requests\.\w+|open\()",
            "severity": "HIGH",
            "detail": "Blocking synchronous call inside async function",
            "fix": "Use asyncio.sleep(), aiohttp, or aiofiles for async I/O",
        },
        "missing_await": {
            "pattern": r"(?:async\s+def.*\n(?:.*\n)*?)\s+\w+\.\w+\(.*\)(?!\s*\n\s*await)",
            "severity": "MEDIUM",
            "detail": "Coroutine call without await — result is a coroutine object, not the value",
            "fix": "Add 'await' before coroutine calls",
        },
        "bare_create_task": {
            "pattern": r"asyncio\.create_task\([^)]+\)(?!\s*\n\s*(?:await|tasks|_))",
            "severity": "LOW",
            "detail": "create_task() result not stored — task may be garbage collected",
            "fix": "Store task reference: task = asyncio.create_task(...)",
        },
    }

    DEADLOCK_INDICATORS = [
        "lock.acquire", "rlock.acquire", "semaphore.acquire",
        "Lock()", "RLock()", "Condition()",
    ]

    def __init__(self):
        """Initialize ConcurrencyAnalyzer."""
        self.analysis_count = 0
        self.total_hazards = 0

    def analyze(self, source: str) -> Dict[str, Any]:
        """
        Full concurrency analysis on Python source code.

        Returns hazards, async issues, deadlock risks, and safety recommendations.
        """
        self.analysis_count += 1
        lines = source.split('\n')
        hazards = []
        async_issues = []
        deadlock_risks = []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"error": "SyntaxError", "hazards": [], "async_issues": []}

        # Phase 1: Detect thread-unsafe patterns via regex
        for name, pat_info in self.THREAD_UNSAFE_PATTERNS.items():
            for match in re.finditer(pat_info["pattern"], source, re.MULTILINE):
                line_no = source[:match.start()].count('\n') + 1
                hazards.append({
                    "type": name,
                    "severity": pat_info["severity"],
                    "line": line_no,
                    "detail": pat_info["detail"],
                    "fix": pat_info["fix"],
                })

        # Phase 2: AST-level concurrency analysis
        hazards.extend(self._detect_shared_state_mutation(tree))
        hazards.extend(self._detect_unprotected_counter(tree, lines))

        # Phase 3: Async anti-pattern detection
        async_issues.extend(self._detect_sync_in_async(tree, lines))
        async_issues.extend(self._detect_missing_await(tree))

        # Phase 4: Deadlock risk assessment
        deadlock_risks = self._assess_deadlock_risk(source, tree)

        # Phase 5: Thread pool sizing check
        pool_issues = self._check_pool_sizing(source)

        self.total_hazards += len(hazards) + len(async_issues)

        # Score
        severity_weights = {"HIGH": 3.0, "MEDIUM": 1.5, "LOW": 0.5}
        total_weight = sum(severity_weights.get(h.get("severity", "MEDIUM"), 1.0)
                          for h in hazards + async_issues + deadlock_risks)
        loc = max(len(lines), 1)
        hazard_density = total_weight / loc
        safety_score = max(0.0, 1.0 - hazard_density * PHI * 5)

        return {
            "hazards": hazards,
            "async_issues": async_issues,
            "deadlock_risks": deadlock_risks,
            "pool_issues": pool_issues,
            "total_hazards": len(hazards),
            "total_async_issues": len(async_issues),
            "total_deadlock_risks": len(deadlock_risks),
            "hazard_density": round(hazard_density, 6),
            "safety_score": round(safety_score, 4),
            "uses_threading": "threading" in source or "Thread(" in source,
            "uses_asyncio": "asyncio" in source or "async def" in source,
            "uses_multiprocessing": "multiprocessing" in source,
        }

    def _detect_shared_state_mutation(self, tree: ast.AST) -> List[Dict]:
        """Detect potential shared state mutations in class methods."""
        findings = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Find class-level attributes
                class_attrs = set()
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            name = self._extract_name(target)
                            if name:
                                class_attrs.add(name)

                # Check for mutations in methods without lock
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        has_lock = any(
                            "lock" in ast.dump(child).lower()
                            for child in ast.walk(item)
                            if isinstance(child, ast.Attribute)
                        )
                        if not has_lock:
                            for child in ast.walk(item):
                                if isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Store):
                                    if isinstance(child.value, ast.Name) and child.value.id == "self":
                                        if child.attr in class_attrs:
                                            findings.append({
                                                "type": "unprotected_class_state",
                                                "severity": "MEDIUM",
                                                "line": child.lineno,
                                                "detail": f"self.{child.attr} modified in {item.name}() without lock — may race",
                                                "fix": f"Protect self.{child.attr} access with self._lock",
                                            })
        return findings

    def _detect_unprotected_counter(self, tree: ast.AST, lines: List[str]) -> List[Dict]:
        """Detect +=/-= operations that may be non-atomic."""
        findings = []
        for node in ast.walk(tree):
            if isinstance(node, ast.AugAssign) and isinstance(node.op, (ast.Add, ast.Sub)):
                if isinstance(node.target, ast.Attribute):
                    if isinstance(node.target.value, ast.Name) and node.target.value.id == "self":
                        # Check context — inside a method of a class
                        findings.append({
                            "type": "non_atomic_counter",
                            "severity": "MEDIUM",
                            "line": node.lineno,
                            "detail": f"self.{node.target.attr} += is not atomic — may race under concurrency",
                            "fix": "Use threading.Lock, or atomic operations (e.g., itertools.count)",
                        })
        return findings

    def _detect_sync_in_async(self, tree: ast.AST, lines: List[str]) -> List[Dict]:
        """Detect synchronous blocking calls inside async functions."""
        findings = []
        blocking_calls = {
            "time.sleep": "asyncio.sleep",
            "requests.get": "aiohttp",
            "requests.post": "aiohttp",
            "requests.put": "aiohttp",
            "open": "aiofiles.open",
            "subprocess.run": "asyncio.create_subprocess_exec",
            "subprocess.call": "asyncio.create_subprocess_exec",
            "os.system": "asyncio.create_subprocess_shell",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        call_name = ""
                        if isinstance(child.func, ast.Attribute):
                            if isinstance(child.func.value, ast.Name):
                                call_name = f"{child.func.value.id}.{child.func.attr}"
                        elif isinstance(child.func, ast.Name):
                            call_name = child.func.id
                        if call_name in blocking_calls:
                            findings.append({
                                "type": "sync_in_async",
                                "severity": "HIGH",
                                "line": child.lineno,
                                "detail": f"Blocking call '{call_name}' inside async def {node.name}()",
                                "fix": f"Replace with '{blocking_calls[call_name]}' for non-blocking I/O",
                                "async_function": node.name,
                            })
        return findings

    def _detect_missing_await(self, tree: ast.AST) -> List[Dict]:
        """Detect coroutine calls that are missing await."""
        findings = []
        # Collect known async function names
        async_funcs = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                async_funcs.add(node.name)

        # Check for calls to known async functions without await
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Expr) and isinstance(child.value, ast.Call):
                        call_name = self._extract_call_name(child.value)
                        if call_name in async_funcs:
                            findings.append({
                                "type": "missing_await",
                                "severity": "HIGH",
                                "line": child.lineno,
                                "detail": f"Coroutine '{call_name}()' called without 'await' — result is discarded",
                                "fix": f"Add 'await': await {call_name}(...)",
                            })
        return findings

    def _assess_deadlock_risk(self, source: str, tree: ast.AST) -> List[Dict]:
        """Assess deadlock risk based on lock usage patterns."""
        risks = []
        lock_names = set()

        # Find all lock declarations
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call):
                    call_name = self._extract_call_name(node.value)
                    if call_name and "lock" in call_name.lower():
                        for target in node.targets:
                            name = self._extract_name(target)
                            if name:
                                lock_names.add(name)

        # Check for nested lock acquisition (deadlock risk)
        if len(lock_names) >= 2:
            risks.append({
                "type": "multi_lock_deadlock_risk",
                "severity": "HIGH",
                "line": 1,
                "detail": f"Multiple locks declared ({', '.join(sorted(lock_names)[:5])}) — nested acquisition may deadlock",
                "fix": "Always acquire locks in a consistent global order, or use a single coarse-grained lock",
            })

        return risks

    def _check_pool_sizing(self, source: str) -> List[Dict]:
        """Check thread/process pool sizing."""
        issues = []
        # Detect hardcoded large pool sizes
        for match in re.finditer(r'(?:ThreadPoolExecutor|ProcessPoolExecutor)\s*\(\s*(?:max_workers\s*=\s*)?(\d+)', source):
            size = int(match.group(1))
            line_no = source[:match.start()].count('\n') + 1
            if size > 100:
                issues.append({
                    "type": "oversized_pool",
                    "severity": "MEDIUM",
                    "line": line_no,
                    "detail": f"Pool size {size} is unusually large — may exhaust system resources",
                    "fix": f"Use os.cpu_count() or a smaller fixed size (recommended: {min(32, size)})",
                })
            elif size < 2:
                issues.append({
                    "type": "undersized_pool",
                    "severity": "LOW",
                    "line": line_no,
                    "detail": f"Pool size {size} provides no parallelism benefit",
                    "fix": "Use at least 2 workers for parallelism, or omit pool entirely",
                })
        return issues

    def _extract_name(self, node: ast.AST) -> Optional[str]:
        """Extract name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _extract_call_name(self, node: ast.Call) -> Optional[str]:
        """Extract the function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def status(self) -> Dict[str, Any]:
        """Return concurrency analyzer status."""
        return {
            "analyses_run": self.analysis_count,
            "total_hazards_detected": self.total_hazards,
            "thread_patterns": len(self.THREAD_UNSAFE_PATTERNS),
            "async_patterns": len(self.ASYNC_ANTIPATTERNS),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4L: API CONTRACT VALIDATOR — Signature & Contract Verification (v3.1.0)
#   Validates function signatures, docstring-code consistency, return contract
#   adherence, and public API surface stability detection.
# ═══════════════════════════════════════════════════════════════════════════════



class APIContractValidator:
    """
    Validates the 'contract' between a function's signature, docstring, and actual
    behavior. Detects lie-by-docstring, mismatched parameters, undocumented exceptions,
    and public API surface drift.

    v3.1.0: New subsystem — bridges documentation and code analysis to ensure
    that documented behavior matches actual implementation.
    """

    def __init__(self):
        """Initialize APIContractValidator."""
        self.validations_run = 0
        self.violations_found = 0

    def validate(self, source: str) -> Dict[str, Any]:
        """
        Validate all function/method contracts in source code.

        Checks:
          1. Docstring-parameter mismatch
          2. Undocumented exceptions (raises not in docstring)
          3. Return type contract violations
          4. Missing docstrings on public functions
          5. Deprecated but still-called functions
        """
        self.validations_run += 1
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"error": "SyntaxError", "violations": [], "api_surface": []}

        violations = []
        api_surface = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_public = not node.name.startswith('_')
                docstring = ast.get_docstring(node)

                if is_public:
                    api_surface.append({
                        "name": node.name,
                        "line": node.lineno,
                        "params": [arg.arg for arg in node.args.args if arg.arg != "self"],
                        "has_docstring": docstring is not None,
                        "has_return_annotation": node.returns is not None,
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                    })

                # Check 1: Missing docstring on public function
                if is_public and not docstring:
                    violations.append({
                        "type": "missing_docstring",
                        "function": node.name,
                        "line": node.lineno,
                        "severity": "MEDIUM",
                        "detail": f"Public function '{node.name}' lacks a docstring",
                        "fix": f"Add docstring: def {node.name}(...):\n    \"\"\"Description.\"\"\"",
                    })

                if docstring:
                    # Check 2: Param mismatch
                    violations.extend(self._check_param_mismatch(node, docstring))

                    # Check 3: Undocumented exceptions
                    violations.extend(self._check_undocumented_raises(node, docstring))

                    # Check 4: Documented params not in signature
                    violations.extend(self._check_phantom_params(node, docstring))

        # Check 5: Exported names analysis
        export_info = self._analyze_exports(tree)

        self.violations_found += len(violations)

        # Contract health score
        total_funcs = len(api_surface)
        violation_ratio = len(violations) / max(total_funcs, 1)
        docstring_coverage = sum(1 for f in api_surface if f["has_docstring"]) / max(total_funcs, 1)
        annotation_coverage = sum(1 for f in api_surface if f["has_return_annotation"]) / max(total_funcs, 1)
        contract_health = (docstring_coverage * PHI + annotation_coverage + (1.0 - violation_ratio)) / (PHI + 2)

        return {
            "violations": violations,
            "total_violations": len(violations),
            "api_surface": api_surface,
            "api_surface_count": len(api_surface),
            "docstring_coverage": round(docstring_coverage, 4),
            "annotation_coverage": round(annotation_coverage, 4),
            "contract_health": round(contract_health, 4),
            "exports": export_info,
        }

    def _check_param_mismatch(self, node: ast.FunctionDef, docstring: str) -> List[Dict]:
        """Check if function params match docstring-documented params."""
        violations = []
        sig_params = {arg.arg for arg in node.args.args if arg.arg not in ("self", "cls")}
        # Also include *args and **kwargs
        if node.args.vararg:
            sig_params.add(node.args.vararg.arg)
        if node.args.kwarg:
            sig_params.add(node.args.kwarg.arg)

        # Parse docstring for parameter mentions (Google/NumPy/Sphinx styles)
        doc_params = set()
        # Google style: "    param_name (type): description" or "    param_name: description"
        for match in re.finditer(r'^\s+(\w+)\s*(?:\([^)]*\))?\s*:', docstring, re.MULTILINE):
            word = match.group(1)
            if word.lower() not in ("returns", "return", "raises", "raise", "yields", "yield",
                                     "note", "notes", "example", "examples", "args", "kwargs",
                                     "attributes", "todo", "see", "references", "warning"):
                doc_params.add(word)
        # Sphinx style: ":param param_name:"
        for match in re.finditer(r':param\s+(?:\w+\s+)?(\w+):', docstring):
            doc_params.add(match.group(1))

        # Find params in signature but not documented
        undocumented = sig_params - doc_params
        for p in undocumented:
            if len(sig_params) > 1:  # Skip single-param functions
                violations.append({
                    "type": "undocumented_param",
                    "function": node.name,
                    "line": node.lineno,
                    "severity": "LOW",
                    "detail": f"Parameter '{p}' in signature but not in docstring",
                    "fix": f"Document parameter '{p}' in the docstring",
                })

        # Find params in docstring but not in signature (phantom params)
        phantom = doc_params - sig_params
        for p in phantom:
            violations.append({
                "type": "phantom_param",
                "function": node.name,
                "line": node.lineno,
                "severity": "MEDIUM",
                "detail": f"Docstring mentions '{p}' but it's not in the function signature",
                "fix": f"Remove '{p}' from docstring or add it to the signature",
            })

        return violations

    def _check_undocumented_raises(self, node: ast.FunctionDef, docstring: str) -> List[Dict]:
        """Detect raised exceptions not mentioned in docstring."""
        violations = []
        raised_exceptions = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Raise) and child.exc:
                if isinstance(child.exc, ast.Call):
                    exc_name = None
                    if isinstance(child.exc.func, ast.Name):
                        exc_name = child.exc.func.id
                    elif isinstance(child.exc.func, ast.Attribute):
                        exc_name = child.exc.func.attr
                    if exc_name:
                        raised_exceptions.add(exc_name)
                elif isinstance(child.exc, ast.Name):
                    raised_exceptions.add(child.exc.id)

        # Check docstring for Raises section
        documented_raises = set()
        for match in re.finditer(r'(?:Raises|:raises?\s+)(\w+)', docstring):
            documented_raises.add(match.group(1))

        undocumented = raised_exceptions - documented_raises
        for exc in undocumented:
            violations.append({
                "type": "undocumented_exception",
                "function": node.name,
                "line": node.lineno,
                "severity": "MEDIUM",
                "detail": f"Function raises {exc} but docstring doesn't document it",
                "fix": f"Add 'Raises:\n    {exc}: description' to docstring",
            })

        return violations

    def _check_phantom_params(self, node: ast.FunctionDef, docstring: str) -> List[Dict]:
        """Already covered in _check_param_mismatch — returns empty for hook."""
        return []

    def _analyze_exports(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze __all__ and module-level public names."""
        all_names = None
        public_names = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                            all_names = [
                                elt.value if isinstance(elt, ast.Constant) else "?"
                                for elt in node.value.elts
                            ]
                        elif not target.id.startswith('_'):
                            public_names.append(target.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith('_'):
                    public_names.append(node.name)
            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith('_'):
                    public_names.append(node.name)

        return {
            "has_all": all_names is not None,
            "all_names": all_names,
            "public_names": public_names[:50],
            "public_count": len(public_names),
        }

    def status(self) -> Dict[str, Any]:
        """Return API contract validator status."""
        return {
            "validations_run": self.validations_run,
            "total_violations": self.violations_found,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4M: CODE EVOLUTION TRACKER — Change Frequency & Stability Analysis (v3.1.0)
#   Measures code stability by tracking function-level change frequency,
#   identifying hotspot churn, and computing stability metrics.
# ═══════════════════════════════════════════════════════════════════════════════

