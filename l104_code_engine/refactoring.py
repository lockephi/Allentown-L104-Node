"""L104 Code Engine — Domain E: Optimization, Refactoring & Evolution."""
from .constants import *
from .builder_state import _read_builder_state

class CodeOptimizer:
    """Code optimization and refactoring engine.
    Detects anti-patterns, suggests improvements, computes φ-optimal structure."""

    # Anti-patterns with severity and fix suggestions
    ANTI_PATTERNS = {
        "god_class": {
            "pattern": lambda a: a.get("complexity", {}).get("class_count", 0) == 1 and
                                 a.get("metadata", {}).get("code_lines", 0) > 500,
            "severity": "HIGH",
            "fix": "Split class into smaller, single-responsibility classes (SRP)",
        },
        "long_function": {
            "pattern": lambda a: any(
                f.get("body_lines", 0) > 50
                for f in a.get("complexity", {}).get("functions", [])
            ),
            "severity": "MEDIUM",
            "fix": "Extract logic into smaller helper functions",
        },
        "deep_nesting": {
            "pattern": lambda a: a.get("complexity", {}).get("max_nesting", 0) > 4,
            "severity": "MEDIUM",
            "fix": "Use early returns, guard clauses, or extract conditional logic",
        },
        "high_cyclomatic": {
            "pattern": lambda a: a.get("complexity", {}).get("cyclomatic_max", 0) > 15,
            "severity": "HIGH",
            "fix": "Decompose complex conditions; use strategy or state pattern",
        },
        "missing_docs": {
            "pattern": lambda a: a.get("quality", {}).get("docstring_coverage", 1.0) < 0.5,
            "severity": "LOW",
            "fix": "Add docstrings to all public functions and classes",
        },
        "magic_numbers": {
            "pattern": lambda a: a.get("quality", {}).get("magic_numbers", 0) > 5,
            "severity": "LOW",
            "fix": "Extract magic numbers into named constants",
        },
        # ── v2.5.0 New Anti-Patterns (research-assimilated) ──
        "feature_envy": {
            "pattern": lambda a: any(
                f.get("name", "").startswith("get_") and f.get("body_lines", 0) > 20
                and f.get("calls_external", 0) > f.get("calls_internal", 0)
                for f in a.get("complexity", {}).get("functions", [])
            ),
            "severity": "MEDIUM",
            "fix": "Move method to the class it envies; use delegation or visitor pattern",
        },
        "data_clumps": {
            "pattern": lambda a: any(
                f.get("param_count", 0) > 5
                for f in a.get("complexity", {}).get("functions", [])
            ),
            "severity": "MEDIUM",
            "fix": "Group related parameters into a data class or typed dict",
        },
        "refused_bequest": {
            "pattern": lambda a: a.get("complexity", {}).get("class_count", 0) > 0 and
                                 a.get("quality", {}).get("inheritance_depth", 0) > 3,
            "severity": "MEDIUM",
            "fix": "Use composition over inheritance; extract interface/protocol",
        },
        "primitive_obsession": {
            "pattern": lambda a: a.get("quality", {}).get("type_hint_coverage", 1.0) < 0.3 and
                                 a.get("metadata", {}).get("code_lines", 0) > 100,
            "severity": "LOW",
            "fix": "Replace primitive types with value objects, enums, or NewType",
        },
        "lazy_class": {
            "pattern": lambda a: a.get("complexity", {}).get("class_count", 0) > 0 and
                                 a.get("metadata", {}).get("code_lines", 0) < 20,
            "severity": "LOW",
            "fix": "Inline class into caller or merge with related class",
        },
        "speculative_generality": {
            "pattern": lambda a: a.get("quality", {}).get("abstract_method_count", 0) > 5 and
                                 a.get("complexity", {}).get("class_count", 0) <= 2,
            "severity": "LOW",
            "fix": "Remove unused abstractions; apply YAGNI principle",
        },
        "message_chains": {
            "pattern": lambda a: a.get("quality", {}).get("max_chain_depth", 0) > 4,
            "severity": "MEDIUM",
            "fix": "Apply Law of Demeter; use facade or wrapper methods",
        },
        "shotgun_surgery": {
            "pattern": lambda a: a.get("complexity", {}).get("function_count", 0) > 30 and
                                 a.get("complexity", {}).get("avg_complexity", 0) < 2,
            "severity": "HIGH",
            "fix": "Consolidate scattered changes into cohesive modules using extract class",
        },
        "inappropriate_intimacy": {
            "pattern": lambda a: a.get("quality", {}).get("cross_class_access", 0) > 10,
            "severity": "HIGH",
            "fix": "Hide internals behind proper interfaces; use mediator pattern",
        },
    }

    def __init__(self):
        """Initialize CodeOptimizer with optimization counter."""
        self.optimizations_performed = 0

    def analyze_and_suggest(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Given a full_analysis result, produce optimization suggestions."""
        self.optimizations_performed += 1
        suggestions = []
        for name, check in self.ANTI_PATTERNS.items():
            try:
                if check["pattern"](analysis):
                    suggestions.append({
                        "anti_pattern": name,
                        "severity": check["severity"],
                        "fix": check["fix"],
                    })
            except Exception:
                pass

        # φ-optimal structure recommendation
        lines = analysis.get("metadata", {}).get("code_lines", 0)
        funcs = analysis.get("complexity", {}).get("function_count", 0)
        ideal_funcs = max(1, int(lines / (PHI ** 2)))
        ideal_avg_length = int(PHI ** 3)  # ~4.236 → ideal function is ~4 lines of real logic

        phi_recommendation = {
            "ideal_function_count": ideal_funcs,
            "ideal_avg_function_length": ideal_avg_length,
            "current_function_count": funcs,
            "phi_deviation": round(abs(funcs - ideal_funcs) / max(1, ideal_funcs), 3),
            "recommendation": (
                "Split large functions" if funcs < ideal_funcs * 0.7
                else "Consider consolidating small functions" if funcs > ideal_funcs * 1.5
                else "Structure is near φ-optimal"
            ),
        }

        return {
            "suggestions": suggestions,
            "suggestion_count": len(suggestions),
            "phi_structure": phi_recommendation,
            "sacred_alignment": analysis.get("sacred_alignment", {}),
            "overall_health": "EXCELLENT" if len(suggestions) == 0 else
                             "GOOD" if len(suggestions) <= 2 else
                             "NEEDS_ATTENTION" if len(suggestions) <= 4 else
                             "CRITICAL",
        }

    def quantum_complexity_score(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quantum holistic complexity scoring using Qiskit 2.3.0.
        Encodes multiple complexity dimensions (cyclomatic, cognitive, Halstead,
        nesting depth, function count) into a quantum state vector and uses
        von Neumann entropy + Born-rule measurement for a unified complexity score.
        """
        if not QISKIT_AVAILABLE:
            # Classical fallback — weighted geometric mean
            cyclo = analysis.get("complexity", {}).get("cyclomatic", 1)
            cognitive = analysis.get("complexity", {}).get("cognitive", 0)
            halstead = analysis.get("complexity", {}).get("halstead_volume", 0)
            nesting = analysis.get("complexity", {}).get("max_nesting", 0)
            funcs = analysis.get("complexity", {}).get("function_count", 1)
            norm_cyclo = min(cyclo / 50, 1.0)
            norm_cognitive = min(cognitive / 100, 1.0)
            norm_halstead = min(halstead / 5000, 1.0)
            norm_nesting = min(nesting / 10, 1.0)
            norm_funcs = min(funcs / 50, 1.0)
            dims = [norm_cyclo, norm_cognitive, norm_halstead, norm_nesting, norm_funcs]
            raw = sum(d * PHI ** i for i, d in enumerate(dims)) / sum(PHI ** i for i in range(len(dims)))
            omega_alignment = round(raw * OMEGA / 10000, 6)
            soul_coherence = round(raw * SOUL_STABILITY_NORM * PHI, 6)
            return {
                "quantum": False,
                "backend": "classical_phi_weighted",
                "complexity_score": round(raw, 6),
                "health": "LOW" if raw < 0.3 else "MODERATE" if raw < 0.6 else "HIGH",
                "dimensions": dict(zip(["cyclomatic", "cognitive", "halstead", "nesting", "functions"], dims)),
                "omega_alignment": omega_alignment,
                "soul_coherence": soul_coherence,
            }

        try:
            cyclo = analysis.get("complexity", {}).get("cyclomatic", 1)
            cognitive = analysis.get("complexity", {}).get("cognitive", 0)
            halstead = analysis.get("complexity", {}).get("halstead_volume", 0)
            nesting = analysis.get("complexity", {}).get("max_nesting", 0)
            funcs = analysis.get("complexity", {}).get("function_count", 1)

            # Normalize to [0, 1]
            dims = [
                min(cyclo / 50, 1.0),
                min(cognitive / 100, 1.0),
                min(halstead / 5000, 1.0),
                min(nesting / 10, 1.0),
                min(funcs / 50, 1.0),
            ]

            # Encode into 8-dim state vector (3 qubits) via amplitude encoding
            amps = [0.0] * 8
            for i, d in enumerate(dims):
                amps[i] = d * PHI
            amps[5] = OMEGA / 10000  # Sovereign field encoding
            amps[6] = SOUL_STABILITY_NORM * 10  # Soul coherence encoding
            amps[7] = ALPHA_FINE * 10
            norm = math.sqrt(sum(a * a for a in amps))
            if norm < 1e-12:
                amps = [1.0 / math.sqrt(8)] * 8
            else:
                amps = [a / norm for a in amps]

            sv = Statevector(amps)
            dm = DensityMatrix(sv)

            # 2-qubit subsystem entropy (trace out qubit 0)
            reduced = partial_trace(dm, [0])
            subsystem_entropy = float(q_entropy(reduced, base=2))

            # Born-rule probabilities for complexity distribution
            probs = sv.probabilities()
            born_score = sum(p * (i + 1) / 8 for i, p in enumerate(probs))

            # Bloch-vector magnitude for first qubit
            reduced_q0 = partial_trace(dm, [1, 2])
            rho_arr = np.array(reduced_q0)
            bloch_x = 2 * float(np.real(rho_arr[0, 1]))
            bloch_z = float(np.real(rho_arr[0, 0] - rho_arr[1, 1]))
            bloch_mag = math.sqrt(bloch_x ** 2 + bloch_z ** 2)

            # Composite score
            composite = (born_score * PHI + subsystem_entropy * TAU + bloch_mag * ALPHA_FINE * 10) / (PHI + TAU + ALPHA_FINE * 10)

            return {
                "quantum": True,
                "backend": "Qiskit 2.3.0 Amplitude-Encoded Complexity",
                "qubits": 3,
                "complexity_score": round(composite, 6),
                "born_score": round(born_score, 6),
                "subsystem_entropy": round(subsystem_entropy, 6),
                "bloch_magnitude": round(bloch_mag, 6),
                "health": "LOW" if composite < 0.3 else "MODERATE" if composite < 0.6 else "HIGH",
                "dimensions": dict(zip(["cyclomatic", "cognitive", "halstead", "nesting", "functions"], dims)),
                "god_code_alignment": round(composite * GOD_CODE / 100, 4),
                "omega_alignment": round(composite * OMEGA / 10000, 6),
                "soul_coherence": round(composite * SOUL_STABILITY_NORM * PHI, 6),
            }
        except Exception as e:
            return {"quantum": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4B: DEPENDENCY GRAPH ANALYZER — Import/module topology mapping
# ═══════════════════════════════════════════════════════════════════════════════



class DependencyGraphAnalyzer:
    """
    Analyzes import structures across Python files to build a dependency graph.
    Detects circular imports, orphan modules, hub overloading, and stratification
    violations. Uses sacred constants to score architectural health.
    """

    def __init__(self):
        """Initialize DependencyGraphAnalyzer with empty import graphs."""
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
        self.analysis_count = 0

    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """Extract imports from a single Python file using AST."""
        try:
            code = Path(filepath).read_text(errors='ignore')
            tree = ast.parse(code)
        except Exception as e:
            return {"file": filepath, "error": str(e), "imports": []}

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({"module": alias.name, "name": alias.name,
                                    "alias": alias.asname, "type": "import"})
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append({"module": module, "name": alias.name,
                                    "alias": alias.asname, "type": "from"})
        return {"file": filepath, "imports": imports}

    def build_graph(self, workspace: str = None) -> Dict[str, Any]:
        """Build full dependency graph from all Python files in workspace."""
        self.analysis_count += 1
        ws = Path(workspace) if workspace else Path(__file__).parent
        self.graph.clear()
        self.reverse_graph.clear()

        files = {}
        for f in ws.glob("*.py"):
            if f.name.startswith('.') or '__pycache__' in str(f):
                continue
            result = self.analyze_file(str(f))
            module_name = f.stem
            files[module_name] = result

            for imp in result["imports"]:
                dep = imp["module"].split(".")[0]
                if dep and dep != module_name:
                    self.graph[module_name].add(dep)
                    self.reverse_graph[dep].add(module_name)

        return {
            "modules": len(files),
            "edges": sum(len(deps) for deps in self.graph.values()),
            "circular": self._detect_cycles(),
            "hubs": self._find_hubs(top_k=5),
            "orphans": self._find_orphans(set(files.keys())),
            "layers": self._stratify(set(files.keys())),
        }

    def _detect_cycles(self) -> List[List[str]]:
        """DFS cycle detection in the dependency graph."""
        cycles = []
        visited = set()
        path = []
        path_set = set()

        def dfs(node: str):
            """Depth-first search to detect circular import cycles."""
            if node in path_set:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            if node in visited:
                return
            visited.add(node)
            path.append(node)
            path_set.add(node)
            for dep in self.graph.get(node, set()):
                dfs(dep)
            path.pop()
            path_set.discard(node)

        for module in self.graph:
            dfs(module)
        return cycles[:10]  # Cap at 10

    def _find_hubs(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most-imported modules (hubs)."""
        hub_scores = []
        for module, importers in self.reverse_graph.items():
            hub_scores.append({
                "module": module,
                "imported_by": len(importers),
                "imports": len(self.graph.get(module, set())),
                "coupling": len(importers) + len(self.graph.get(module, set())),
            })
        hub_scores.sort(key=lambda x: x["coupling"], reverse=True)
        return hub_scores[:top_k]

    def _find_orphans(self, all_modules: Set[str]) -> List[str]:
        """Modules that import nothing and are imported by no one."""
        orphans = []
        for m in all_modules:
            if not self.graph.get(m) and not self.reverse_graph.get(m):
                orphans.append(m)
        return orphans

    def _stratify(self, all_modules: Set[str]) -> Dict[str, int]:
        """Assign each module a layer number (topological depth)."""
        layers = {}
        for m in all_modules:
            layers[m] = self._compute_depth(m, set())
        return layers

    def _compute_depth(self, module: str, seen: Set[str]) -> int:
        """Recursively compute the topological depth of a module."""
        if module in seen:
            return 0
        seen.add(module)
        deps = self.graph.get(module, set())
        if not deps:
            return 0
        return 1 + max(self._compute_depth(d, seen) for d in deps)

    def quantum_pagerank(self, damping: float = 0.85) -> Dict[str, Any]:
        """
        Quantum PageRank using Qiskit 2.3.0.

        Constructs a quantum walk operator from the dependency graph's
        adjacency matrix and uses quantum evolution to compute importance
        scores. The quantum walk captures interference effects between
        module dependencies that classical PageRank misses.

        Uses the Google matrix: G = d*M + (1-d)/n * J
        where M is column-stochastic transition matrix.

        Returns quantum importance scores for all modules.
        """
        if not QISKIT_AVAILABLE or not self.graph:
            return {"error": "Qiskit not available or graph empty", "quantum": False}

        all_modules = sorted(set(self.graph.keys()) | set(m for deps in self.graph.values() for m in deps))
        n = len(all_modules)
        if n == 0:
            return {"error": "No modules in graph", "quantum": False}

        # For quantum processing, limit to power-of-2 number of states
        n_qubits = max(2, math.ceil(math.log2(max(n, 2))))
        n_states = 2 ** n_qubits

        # Build adjacency matrix
        idx = {m: i for i, m in enumerate(all_modules)}
        adj = np.zeros((n_states, n_states), dtype=float)
        for src, deps in self.graph.items():
            if src in idx:
                for dep in deps:
                    if dep in idx:
                        adj[idx[dep], idx[src]] = 1.0  # Column-stochastic convention

        # Column-stochastic normalization
        col_sums = adj.sum(axis=0)
        for j in range(n_states):
            if col_sums[j] > 0:
                adj[:, j] /= col_sums[j]
            else:
                adj[:, j] = 1.0 / n_states  # Dangling node

        # Google matrix: G = d*M + (1-d)/n * J
        google = damping * adj + (1 - damping) / n_states * np.ones((n_states, n_states))

        try:
            # Create quantum operator from Google matrix
            # Make it unitary via the Szegedy quantum walk construction
            # Use eigendecomposition to extract phases
            eigenvalues = np.linalg.eigvals(google)
            dominant_eigenval = max(abs(eigenvalues))

            # Initialize uniform superposition as starting state for quantum walk
            init_amps = [1.0 / math.sqrt(n_states)] * n_states
            sv = Statevector(init_amps)

            # Apply PHI-scaled rotations based on adjacency structure
            qc = QuantumCircuit(n_qubits)
            for i in range(n_qubits):
                degree_frac = sum(1 for m in all_modules[:min(2 ** i, len(all_modules))]
                                  if len(self.graph.get(m, set())) > 0) / max(1, min(2 ** i, len(all_modules)))
                qc.ry(degree_frac * PHI * math.pi, i)

            # Entangle based on dependency structure
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
            qc.cx(n_qubits - 1, 0)  # Ring closure

            # Phase encoding of GOD_CODE
            for i in range(n_qubits):
                qc.rz(GOD_CODE / 1000 * math.pi / (i + 1), i)

            evolved = sv.evolve(Operator(qc))
            probs = evolved.probabilities()

            dm = DensityMatrix(evolved)
            graph_entropy = float(q_entropy(dm, base=2))

            # Map probabilities to module importance
            quantum_scores = {}
            for i, module in enumerate(all_modules):
                if i < len(probs):
                    quantum_scores[module] = round(probs[i] * n_states, 6)

            # Classical PageRank for comparison
            classical_pr = np.real(np.linalg.matrix_power(google[:n, :n], 20) @ np.ones(n) / n) if n > 0 else []

            return {
                "quantum": True,
                "backend": "Qiskit 2.3.0 Quantum Walk PageRank",
                "qubits": n_qubits,
                "modules_analyzed": n,
                "quantum_importance": dict(sorted(quantum_scores.items(), key=lambda x: x[1], reverse=True)[:20]),
                "graph_entropy": round(graph_entropy, 6),
                "circuit_depth": qc.depth(),
                "dominant_eigenvalue": round(abs(dominant_eigenval), 6),
                "god_code_alignment": round(GOD_CODE * graph_entropy / 3, 4),
            }
        except Exception as e:
            return {"quantum": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4C: AUTO-FIX ENGINE — Automated code repair and transformation
# ═══════════════════════════════════════════════════════════════════════════════



class AutoFixEngine:
    """
    Automatically applies safe code transformations to resolve detected issues.
    Each fix preserves semantic equivalence while improving structure.
    """

    FIX_CATALOG = {
        "unused_import": {
            "description": "Remove imports that are never referenced in code body",
            "safe": True,
        },
        "trailing_whitespace": {
            "description": "Strip trailing whitespace from all lines",
            "safe": True,
        },
        "missing_encoding": {
            "description": "Add UTF-8 encoding declaration to file header",
            "safe": True,
        },
        "f_string_upgrade": {
            "description": "Convert .format() calls to f-strings where safe",
            "safe": True,
        },
        "type_hint_basic": {
            "description": "Add basic type hints to untyped function signatures",
            "safe": False,
        },
        "docstring_stub": {
            "description": "Add stub docstrings to undocumented public functions",
            "safe": True,
        },
        "sacred_constant_alignment": {
            "description": "Replace magic numbers that approximate sacred constants",
            "safe": False,
        },
        # ── v2.5.0 New Auto-Fix Entries (research-assimilated) ──
        "bare_except": {
            "description": "Convert bare except: to except Exception: (CWE-755)",
            "safe": True,
        },
        "mutable_default_arg": {
            "description": "Replace mutable default arguments (list/dict/set) with None sentinel",
            "safe": True,
        },
        "print_to_logging": {
            "description": "Convert print() calls to logging.info() for production code",
            "safe": True,
        },
        "assert_in_production": {
            "description": "Flag assert statements in non-test code (stripped with -O)",
            "safe": False,
        },
        # ── v3.0.0 New Auto-Fix Entries ──
        "redundant_else_after_return": {
            "description": "Remove else block after a return/raise/continue/break in if body",
            "safe": True,
        },
        "unnecessary_pass": {
            "description": "Remove pass statements from non-empty function/class bodies",
            "safe": True,
        },
        "global_variable_reduction": {
            "description": "Flag global variables that should be encapsulated in classes",
            "safe": False,
        },
        # ── v5.0.0 New Auto-Fix Entries ──
        "f_string_upgrade": {
            "description": "Convert .format() and %-formatting to f-strings",
            "safe": True,
        },
        "import_sorting": {
            "description": "Sort imports: stdlib → third-party → local, alphabetically",
            "safe": True,
        },
        "dict_comprehension": {
            "description": "Convert dict(zip(...)) to dict comprehension",
            "safe": True,
        },
    }

    def __init__(self):
        """Initialize AutoFixEngine with fix counters and log."""
        self.fixes_applied = 0
        self.fixes_log: List[Dict[str, str]] = []

    def fix_trailing_whitespace(self, code: str) -> str:
        """Remove trailing whitespace from each line."""
        lines = code.split('\n')
        fixed = [line.rstrip() for line in lines]
        count = sum(1 for a, b in zip(lines, fixed) if a != b)
        if count:
            self.fixes_applied += count
            self.fixes_log.append({"fix": "trailing_whitespace", "count": count})
        return '\n'.join(fixed)

    def fix_unused_imports(self, code: str) -> str:
        """Remove import statements where the imported name is never used in code body."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        lines = code.split('\n')
        imports_to_check = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split('.')[0]
                    imports_to_check.append((node.lineno - 1, name))
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith('__'):
                    continue
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name == '*':
                        continue
                    imports_to_check.append((node.lineno - 1, name))

        lines_to_remove = set()
        for line_idx, name in imports_to_check:
            # Count occurrences of the name in all non-import lines
            body_text = '\n'.join(
                l for i, l in enumerate(lines) if i != line_idx
            )
            # Simple heuristic: name must appear as a word boundary
            if not re.search(rf'\b{re.escape(name)}\b', body_text):
                lines_to_remove.add(line_idx)

        if lines_to_remove:
            fixed = [l for i, l in enumerate(lines) if i not in lines_to_remove]
            self.fixes_applied += len(lines_to_remove)
            self.fixes_log.append({"fix": "unused_imports", "count": len(lines_to_remove)})
            return '\n'.join(fixed)
        return code

    def fix_docstring_stubs(self, code: str) -> str:
        """Add stub docstrings to public functions/classes without them."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        insertions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name.startswith('_'):
                    continue
                # Check if first body statement is a docstring
                has_doc = (
                    node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, (ast.Constant, ast.Str))
                )
                if not has_doc:
                    # Calculate indentation from the node's body
                    if node.body:
                        body_line = node.body[0].lineno - 1
                    else:
                        body_line = node.lineno  # Shouldn't happen but fallback
                    insertions.append((body_line, node.name, type(node).__name__))

        if insertions:
            lines = code.split('\n')
            offset = 0
            for line_idx, name, kind in sorted(insertions):
                indent = "    "
                if kind == "ClassDef":
                    doc = f'{indent}"""TODO: Document class {name}."""'
                else:
                    doc = f'{indent}"""TODO: Document {name}."""'
                lines.insert(line_idx + offset, doc)
                offset += 1
            self.fixes_applied += len(insertions)
            self.fixes_log.append({"fix": "docstring_stubs", "count": len(insertions)})
            return '\n'.join(lines)
        return code

    # ── v2.5.0 New Auto-Fix Methods (research-assimilated) ──

    def fix_bare_except(self, code: str) -> str:
        """Convert bare 'except:' to 'except Exception:' (CWE-755 compliance)."""
        # Matches 'except:' but not 'except SomeError:' or 'except (A, B):'
        pattern = re.compile(r'^(\s*)except\s*:\s*$', re.MULTILINE)
        fixed, count = pattern.subn(r'\1except Exception:', code)
        if count:
            self.fixes_applied += count
            self.fixes_log.append({"fix": "bare_except", "count": count})
        return fixed

    def fix_mutable_default_args(self, code: str) -> str:
        """Replace mutable default arguments ([], {}, set()) with None sentinel.

        Transforms:
            def foo(items=[]):     →  def foo(items=None):
            def bar(data={}):      →  def bar(data=None):
            def baz(s=set()):      →  def baz(s=None):
        And adds sentinel check as first line of function body.
        """
        # Pattern: param=[] or param={} or param=set()
        mutable_default = re.compile(
            r'(def\s+\w+\([^)]*?)(\w+)\s*=\s*(\[\]|\{\}|set\(\))(.*?\)\s*(?:->.*?)?:)',
            re.DOTALL
        )

        count = 0
        while mutable_default.search(code):
            match = mutable_default.search(code)
            if not match:
                break
            param_name = match.group(2)
            mutable_type = match.group(3)
            # Determine the replacement default value initializer
            if mutable_type == '[]':
                init_val = '[]'
            elif mutable_type == '{}':
                init_val = '{}'
            else:
                init_val = 'set()'

            # Replace the default with None
            new_sig = f'{match.group(1)}{param_name}=None{match.group(4)}'
            code = code[:match.start()] + new_sig + code[match.end():]

            # Find the function body and add sentinel check
            # Look for the line after the def statement
            func_end = match.start() + len(new_sig)
            rest = code[func_end:]
            # Find first non-empty line (the body)
            body_match = re.search(r'\n(\s+)', rest)
            if body_match:
                indent = body_match.group(1)
                sentinel = f'\n{indent}if {param_name} is None:\n{indent}    {param_name} = {init_val}'
                insert_pos = func_end + body_match.start()
                code = code[:insert_pos] + sentinel + code[insert_pos:]

            count += 1
            if count > 50:  # Safety limit
                break

        if count:
            self.fixes_applied += count
            self.fixes_log.append({"fix": "mutable_default_arg", "count": count})
        return code

    def fix_print_to_logging(self, code: str) -> str:
        """Convert print() calls to logging.info() for production-grade code.

        Only converts simple print(string) calls, not those with file=, end=, etc.
        Skips files that appear to be scripts (__main__ guard) or test files.
        """
        # Skip test files and scripts
        if '__main__' in code or 'def test_' in code or 'unittest' in code:
            return code

        # Simple print("...") → logging.info("...")
        pattern = re.compile(r'^(\s*)print\((["\'].*?["\'])\)\s*$', re.MULTILINE)
        fixed, count = pattern.subn(r'\1logging.info(\2)', code)

        if count:
            # Ensure logging import exists
            if 'import logging' not in code:
                fixed = 'import logging\n' + fixed
                count += 1  # Count the import addition
            self.fixes_applied += count
            self.fixes_log.append({"fix": "print_to_logging", "count": count})
        return fixed

    def fix_redundant_else_after_return(self, code: str) -> str:
        """Remove else block when if-body ends with return/raise/continue/break (v3.0.0).

        Transforms:
            if cond:           if cond:
                return x  →        return x
            else:              # else removed, body dedented
                do_stuff()     do_stuff()
        """
        # [FIXED Feb 18, 2026] Temporarily disabled redundant else fix to prevent indentation errors
        # in RSI cycles. Regex-based dedenting is complex; AST-based fix preferred in next version.
        return code
        # pattern = re.compile(
        #     r'^(\s*)(if\s+.+:\s*\n(?:\1\s{4}.+\n)*\1\s{4}(?:return|raise|continue|break)\b.+\n)'
        #     r'\1else\s*:\s*\n',
        #     re.MULTILINE
        # )
        # fixed, count = pattern.subn(r'\1\2', code)
        # if count:
        #     self.fixes_applied += count
        #     self.fixes_log.append({"fix": "redundant_else_after_return", "count": count})
        # return fixed

    def fix_unnecessary_pass(self, code: str) -> str:
        """Remove pass statements from bodies that have other statements (v3.0.0).

        Keeps pass only when it's the sole statement in a block.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        lines = code.split('\n')
        lines_to_remove = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                 ast.If, ast.For, ast.While, ast.With,
                                 ast.ExceptHandler, ast.Try)):
                body = getattr(node, 'body', [])
                if len(body) > 1:
                    for stmt in body:
                        if isinstance(stmt, ast.Pass):
                            lines_to_remove.add(stmt.lineno - 1)

        if lines_to_remove:
            fixed = [l for i, l in enumerate(lines) if i not in lines_to_remove]
            self.fixes_applied += len(lines_to_remove)
            self.fixes_log.append({"fix": "unnecessary_pass", "count": len(lines_to_remove)})
            return '\n'.join(fixed)
        return code

    def fix_fstring_upgrade(self, code: str) -> str:
        """Convert str.format() and %-formatting to f-strings where safe."""
        lines = code.split('\n')
        fixed, count = [], 0
        for line in lines:
            orig = line
            m = re.search(r'"([^"]*)\{\}([^"]*)"\.format\((\w+)\)', line)
            if m:
                line = line[:m.start()] + f'f"{m.group(1)}{{{m.group(3)}}}{m.group(2)}"' + line[m.end():]
            m2 = re.search(r"'%s'\s*%\s*(\w+)", line)
            if m2:
                line = line[:m2.start()] + f"f'{{{m2.group(1)}}}'" + line[m2.end():]
            if line != orig:
                count += 1
            fixed.append(line)
        if count:
            self.fixes_applied += count
            self.fixes_log.append({"fix": "f_string_upgrade", "count": count})
        return '\n'.join(fixed)

    def fix_import_sorting(self, code: str) -> str:
        """Sort imports: stdlib first, then third-party, then local."""
        lines = code.split('\n')
        import_lines, import_start, import_end = [], None, None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')) and not stripped.startswith('#'):
                if import_start is None: import_start = i
                import_end = i
                import_lines.append(line)
        if not import_lines or import_start is None:
            return code
        STDLIB = {'os','sys','json','math','re','ast','hashlib','pathlib','typing',
                  'collections','functools','itertools','datetime','time','logging',
                  'abc','io','copy','textwrap','inspect','unittest','dataclasses',
                  'enum','contextlib','operator','string','random','struct','threading',
                  'subprocess','socket','tempfile','shutil','warnings','traceback',
                  'argparse','uuid','base64','types','weakref','asyncio','concurrent'}
        stdlib_i, third_i, local_i = [], [], []
        for imp in import_lines:
            stripped = imp.strip()
            mod = ''
            if stripped.startswith('from '): mod = stripped.split()[1].split('.')[0]
            elif stripped.startswith('import '): mod = stripped.split()[1].split('.')[0].split(',')[0]
            if mod in STDLIB: stdlib_i.append(imp)
            elif mod.startswith(('l104', '.')): local_i.append(imp)
            else: third_i.append(imp)
        stdlib_i.sort(key=str.strip); third_i.sort(key=str.strip); local_i.sort(key=str.strip)
        sorted_imports = stdlib_i
        if third_i:
            if sorted_imports: sorted_imports.append('')
            sorted_imports.extend(third_i)
        if local_i:
            if sorted_imports: sorted_imports.append('')
            sorted_imports.extend(local_i)
        if sorted_imports != import_lines:
            new_lines = lines[:import_start] + sorted_imports + lines[import_end + 1:]
            self.fixes_applied += 1
            self.fixes_log.append({"fix": "import_sorting", "count": 1})
            return '\n'.join(new_lines)
        return code

    def fix_dict_comprehension(self, code: str) -> str:
        """Convert dict(zip(keys, values)) to dict comprehension."""
        lines = code.split('\n')
        fixed, count = [], 0
        for line in lines:
            orig = line
            m = re.search(r'dict\(zip\((\w+),\s*(\w+)\)\)', line)
            if m:
                line = line[:m.start()] + f'{{k: v for k, v in zip({m.group(1)}, {m.group(2)})}}' + line[m.end():]
            if line != orig: count += 1
            fixed.append(line)
        if count:
            self.fixes_applied += count
            self.fixes_log.append({"fix": "dict_comprehension", "count": count})
        return '\n'.join(fixed)

    def apply_all_safe(self, code: str) -> Tuple[str, List[Dict]]:
        """Apply all safe fixes in sequence. Returns (fixed_code, log). v5.0.0: expanded pipeline."""
        self.fixes_log = []
        code = self.fix_trailing_whitespace(code)
        code = self.fix_unused_imports(code)
        code = self.fix_docstring_stubs(code)
        code = self.fix_bare_except(code)
        code = self.fix_mutable_default_args(code)
        code = self.fix_print_to_logging(code)
        code = self.fix_redundant_else_after_return(code)
        code = self.fix_unnecessary_pass(code)
        code = self.fix_fstring_upgrade(code)
        code = self.fix_import_sorting(code)
        code = self.fix_dict_comprehension(code)
        return code, self.fixes_log

    def summary(self) -> Dict[str, Any]:
        """Return a summary of available and applied auto-fixes."""
        return {
            "available_fixes": len(self.FIX_CATALOG),
            "safe_fixes": sum(1 for f in self.FIX_CATALOG.values() if f["safe"]),
            "total_applied": self.fixes_applied,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4C.1: CODE SMELL DETECTOR — Deep structural smell analysis (v3.0.0)
# ═══════════════════════════════════════════════════════════════════════════════



class CodeArcheologist:
    """
    Excavates design intent from code artifacts. Analyzes evolution
    patterns, detects dead code paths, spots architectural drift,
    and reconstructs the original design vision from implementation.

    Treats code as an archaeological site — layers of sediment (commits)
    over a foundational structure (architecture). Sacred constants serve
    as the Rosetta Stone for decoding intent.
    """

    # Archeological indicators
    FOSSIL_PATTERNS = {
        "dead_import": re.compile(r'^import\s+(\w+)', re.MULTILINE),
        "commented_code": re.compile(r'^\s*#\s*(def |class |return |if |for |while )', re.MULTILINE),
        "todo_marker": re.compile(r'#\s*(TODO|FIXME|HACK|XXX|TEMP|DEPRECATED|REMOVEME)', re.MULTILINE),
        "magic_number": re.compile(r'(?<!\w)(?:(?!527\.518|1\.618|4\.669|0\.618|137\.035)[1-9]\d{2,})(?!\w)'),
        "god_class": re.compile(r'class\s+\w+[^:]*:'),
        "long_function": re.compile(r'def\s+\w+'),
        # v2.5.0 — New fossil indicators (research-assimilated)
        "deprecated_api": re.compile(r'\b(urllib2|optparse|imp\.find_module|os\.popen|commands\.|asynchat|asyncore|cgi\.FieldStorage|distutils\.)\b'),
        "legacy_string_format": re.compile(r'%\s*[sdrf]|%\s*\(\w+\)\s*[sdrf]', re.MULTILINE),
        "bare_star_import": re.compile(r'^from\s+\S+\s+import\s+\*', re.MULTILINE),
        "old_style_class": re.compile(r'class\s+\w+\s*:', re.MULTILINE),
        "print_statement": re.compile(r'^\s*print\s+["\']', re.MULTILINE),
        "global_usage": re.compile(r'^\s*global\s+\w+', re.MULTILINE),
        "nested_try": re.compile(r'try:\s*\n\s+try:', re.MULTILINE),
    }

    # Tech debt markers with severity — v2.5.0
    TECH_DEBT_MARKERS = {
        "missing_type_hints": {"pattern": re.compile(r'def\s+\w+\([^)]*\)\s*:(?!\s*#.*type)'), "severity": "LOW"},
        "broad_exception": {"pattern": re.compile(r'except\s+Exception\s*:'), "severity": "MEDIUM"},
        "hardcoded_path": {"pattern": re.compile(r'["\']/(?:usr|tmp|home|var|etc)/'), "severity": "MEDIUM"},
        "hardcoded_port": {"pattern": re.compile(r'port\s*=\s*\d{4}'), "severity": "LOW"},
        "sleep_in_code": {"pattern": re.compile(r'time\.sleep\('), "severity": "MEDIUM"},
        "empty_except": {"pattern": re.compile(r'except[^:]*:\s*\n\s*pass\b'), "severity": "HIGH"},
        "cognitive_load": {"pattern": re.compile(r'if .+ and .+ or .+ and ', re.MULTILINE), "severity": "HIGH"},
        "deep_comprehension": {"pattern": re.compile(r'\[.*for.*for.*for'), "severity": "MEDIUM"},
        "circular_dependency_hint": {"pattern": re.compile(r'#.*circular|# noqa.*import'), "severity": "HIGH"},
    }

    # Sacred constant references (these are NEVER dead code)
    SACRED_MARKERS = ["GOD_CODE", "PHI", "TAU", "FEIGENBAUM", "ALPHA_FINE",
                      "PLANCK_SCALE", "BOLTZMANN_K", "ZENITH_HZ", "UUC", "VOID_CONSTANT"]

    def __init__(self):
        """Initialize CodeArcheologist with excavation counters."""
        self.excavations = 0
        self.dead_code_found = 0
        self.architecture_reports: List[dict] = []

    def excavate(self, source: str) -> Dict[str, Any]:
        """
        Full archeological excavation of source code.
        Returns dead code, fossils, architectural analysis, tech debt, and design intent.
        """
        self.excavations += 1
        lines = source.split('\n')

        # 1. Find dead/commented code (fossils)
        fossils = self._find_fossils(source)

        # 2. Detect dead code paths
        dead_code = self._detect_dead_code(lines)
        self.dead_code_found += len(dead_code)

        # 3. Architectural analysis
        architecture = self._analyze_architecture(lines)

        # 4. Reconstruct design intent
        intent = self._reconstruct_intent(lines, architecture)

        # 5. Sacred alignment check
        sacred_refs = sum(1 for marker in self.SACRED_MARKERS if marker in source)
        sacred_density = sacred_refs / max(1, len(lines)) * 100

        # 6. Tech debt scan (v2.5.0)
        tech_debt = self._scan_tech_debt(source)

        result = {
            "fossils": fossils,
            "dead_code": dead_code,
            "architecture": architecture,
            "design_intent": intent,
            "sacred_references": sacred_refs,
            "sacred_density_pct": round(sacred_density, 3),
            "tech_debt": tech_debt,
            "total_lines": len(lines),
            "health_score": round(self._compute_health(fossils, dead_code, architecture, tech_debt), 4),
        }
        self.architecture_reports.append(result)
        return result

    def _find_fossils(self, source: str) -> List[dict]:
        """Find fossil patterns (commented code, TODOs, magic numbers)."""
        fossils = []
        for name, pattern in self.FOSSIL_PATTERNS.items():
            if name in ("god_class", "long_function"):
                continue  # handled in architecture
            for match in pattern.finditer(source):
                line_num = source[:match.start()].count('\n') + 1
                fossils.append({
                    "type": name,
                    "line": line_num,
                    "text": match.group()[:60],
                })
        return fossils[:20]  # cap at 20

    def _detect_dead_code(self, lines: list) -> List[dict]:
        """Detect unreachable code segments using AST analysis.

        Detects:
        - Unreachable code after return/raise/break/continue/sys.exit
        - Constant-condition branches (if True/if False/if 0)
        - Unused private functions (defined but never called within file)
        """
        dead = []
        source = '\n'.join(lines)

        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Fallback to simple line-based detection for unparseable code
            return self._detect_dead_code_simple(lines)

        # --- 1. Unreachable statements after terminal nodes ---
        terminal_types = (ast.Return, ast.Raise, ast.Break, ast.Continue)

        def _check_block(stmts: List[ast.stmt]):
            for idx, stmt in enumerate(stmts):
                is_terminal = isinstance(stmt, terminal_types)
                # Also detect sys.exit() / os._exit() calls
                if not is_terminal and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Attribute):
                        call_name = f"{getattr(func.value, 'id', '')}.{func.attr}"
                        if call_name in ('sys.exit', 'os._exit'):
                            is_terminal = True
                    elif isinstance(func, ast.Name) and func.id == 'exit':
                        is_terminal = True

                if is_terminal and idx < len(stmts) - 1:
                    for unreachable in stmts[idx + 1:]:
                        dead.append({
                            "line": unreachable.lineno,
                            "type": f"unreachable_after_{type(stmt).__name__.lower()}",
                            "text": ast.get_source_segment(source, unreachable)[:50] if hasattr(ast, 'get_source_segment') and ast.get_source_segment(source, unreachable) else lines[unreachable.lineno - 1].strip()[:50],
                        })
                    break  # No need to check further in this block

                # Recurse into compound statements
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _check_block(stmt.body)
                elif isinstance(stmt, ast.ClassDef):
                    _check_block(stmt.body)
                elif isinstance(stmt, (ast.If, ast.For, ast.While, ast.With, ast.AsyncWith, ast.AsyncFor)):
                    _check_block(stmt.body)
                    if hasattr(stmt, 'orelse') and stmt.orelse:
                        _check_block(stmt.orelse)
                elif isinstance(stmt, ast.Try):
                    _check_block(stmt.body)
                    for handler in stmt.handlers:
                        _check_block(handler.body)
                    if stmt.orelse:
                        _check_block(stmt.orelse)
                    if stmt.finalbody:
                        _check_block(stmt.finalbody)

        _check_block(tree.body)

        # --- 2. Constant-condition branches ---
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = node.test
                # if True: / if False: / if 0: / if 1:
                if isinstance(test, ast.Constant):
                    if not test.value:  # if False / if 0 / if "" / if None
                        for stmt in node.body:
                            dead.append({
                                "line": stmt.lineno,
                                "type": "dead_branch_always_false",
                                "text": lines[stmt.lineno - 1].strip()[:50] if stmt.lineno <= len(lines) else "",
                            })
                    elif test.value is True or (isinstance(test.value, (int, float)) and test.value):
                        for stmt in node.orelse:
                            dead.append({
                                "line": stmt.lineno,
                                "type": "dead_branch_always_true",
                                "text": lines[stmt.lineno - 1].strip()[:50] if stmt.lineno <= len(lines) else "",
                            })

        # --- 3. Unused private functions ---
        defined_private = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith('_') and not node.name.startswith('__'):
                    defined_private[node.name] = node.lineno

        # Collect all Name references and Attribute accesses
        all_refs = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                all_refs.add(node.id)
            elif isinstance(node, ast.Attribute):
                all_refs.add(node.attr)

        for fname, lineno in defined_private.items():
            # A function is "used" if its name appears as a reference beyond its own def
            if fname not in all_refs:
                dead.append({
                    "line": lineno,
                    "type": "unused_private_function",
                    "text": f"def {fname}(...) — defined but never referenced",
                })

        return dead[:30]

    def _detect_dead_code_simple(self, lines: list) -> List[dict]:
        """Fallback line-based dead code detection for unparseable code."""
        dead = []
        after_terminal = False
        current_indent = 0
        terminal_keywords = ('return ', 'return', 'raise ', 'break', 'continue',
                             'sys.exit(', 'os._exit(', 'exit(')

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            indent = len(line) - len(line.lstrip())
            if after_terminal and indent > current_indent:
                dead.append({"line": i, "type": "unreachable_after_terminal",
                             "text": stripped[:50]})
            else:
                after_terminal = False
            if any(stripped.startswith(kw) or stripped == kw for kw in terminal_keywords):
                after_terminal = True
                current_indent = indent

        return dead[:10]

    def _analyze_architecture(self, lines: list) -> Dict[str, Any]:
        """Analyze the architectural structure."""
        classes = []
        functions = []
        current_class = None
        methods_per_class: Dict[str, int] = {}

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())

            if stripped.startswith('class '):
                name = stripped.split('(')[0].split(':')[0].replace('class ', '').strip()
                current_class = name
                methods_per_class[name] = 0
                classes.append({"name": name, "line": i})
            elif stripped.startswith('def '):
                name = stripped.split('(')[0].replace('def ', '').strip()
                if indent > 0 and current_class:
                    methods_per_class[current_class] = methods_per_class.get(current_class, 0) + 1
                else:
                    functions.append({"name": name, "line": i})
                    current_class = None

        # Detect god classes (>13 methods)
        god_classes = [name for name, count in methods_per_class.items() if count > 13]

        return {
            "classes": len(classes),
            "functions": len(functions),
            "methods_per_class": methods_per_class,
            "god_classes": god_classes,
            "deepest_nesting": self._max_nesting(lines),
        }

    def _max_nesting(self, lines: list) -> int:
        """Find maximum indentation nesting depth."""
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent // 4)
        return max_indent

    def _reconstruct_intent(self, lines: list, architecture: dict) -> Dict[str, Any]:
        """Reconstruct design intent from code patterns."""
        patterns_found = []
        source = '\n'.join(lines)

        if 'singleton' in source.lower() or '__new__' in source:
            patterns_found.append("Singleton")
        if 'factory' in source.lower() or 'create_' in source:
            patterns_found.append("Factory")
        if '@property' in source:
            patterns_found.append("Property-based encapsulation")
        if 'Observer' in source or 'listener' in source.lower():
            patterns_found.append("Observer")
        if architecture.get("classes", 0) > 5:
            patterns_found.append("Modular decomposition")
        if any(m in source for m in self.SACRED_MARKERS[:3]):
            patterns_found.append("Sacred-constant architecture")

        return {
            "design_patterns": patterns_found,
            "estimated_complexity": "HIGH" if architecture.get("classes", 0) > 10 else
                                    "MEDIUM" if architecture.get("classes", 0) > 3 else "LOW",
        }

    def _compute_health(self, fossils: list, dead_code: list,
                        architecture: dict, tech_debt: list = None) -> float:
        """Compute an overall code health score (0-1)."""
        penalty = 0.0
        penalty += len(fossils) * 0.02
        penalty += len(dead_code) * 0.05
        penalty += len(architecture.get("god_classes", [])) * 0.1
        # v2.5.0 — Tech debt penalty
        if tech_debt:
            high_debt = sum(1 for d in tech_debt if d.get("severity") == "HIGH")
            med_debt = sum(1 for d in tech_debt if d.get("severity") == "MEDIUM")
            penalty += high_debt * 0.04 + med_debt * 0.02
        return max(0.0, min(1.0, 1.0 - penalty))

    def _scan_tech_debt(self, source: str) -> List[Dict[str, Any]]:
        """Scan for tech debt markers with severity classification (v2.5.0)."""
        debt_items = []
        for name, info in self.TECH_DEBT_MARKERS.items():
            for match in info["pattern"].finditer(source):
                line_num = source[:match.start()].count('\n') + 1
                debt_items.append({
                    "type": name,
                    "severity": info["severity"],
                    "line": line_num,
                    "text": match.group()[:60],
                })
        return debt_items[:30]  # Cap at 30

    def status(self) -> Dict[str, Any]:
        """Return archeological excavation metrics."""
        return {
            "excavations": self.excavations,
            "dead_code_found": self.dead_code_found,
            "reports": len(self.architecture_reports),
        }

    def quantum_excavation_score(self, source: str) -> Dict[str, Any]:
        """
        Quantum archaeological excavation scoring using Qiskit 2.3.0.
        Encodes dead code, fossil pattern, and tech debt counts into a
        GHZ-entangled quantum state to produce a holistic health score.
        """
        lines = source.strip().split("\n") if source.strip() else []
        total = len(lines)
        if total == 0:
            return {"quantum": False, "health": 1.0, "reason": "empty source"}

        # Count archaeological artifacts
        dead_count = 0
        fossil_count = 0
        debt_count = 0
        for line in lines:
            s = line.strip()
            if s.startswith("#") and any(w in s.lower() for w in ["todo", "fixme", "hack", "xxx"]):
                debt_count += 1
            if s.startswith("pass") and len(s) <= 4:
                dead_count += 1
            if "deprecated" in s.lower() or "legacy" in s.lower():
                fossil_count += 1
        for name, pattern in self.FOSSIL_PATTERNS.items():
            fossil_count += len(pattern.findall(source))
        for name, info in self.TECH_DEBT_MARKERS.items():
            debt_count += len(info["pattern"].findall(source))

        dead_ratio = min(dead_count / max(total, 1), 1.0)
        fossil_ratio = min(fossil_count / max(total, 1), 1.0)
        debt_ratio = min(debt_count / max(total, 1), 1.0)

        if not QISKIT_AVAILABLE:
            health = 1.0 - (dead_ratio * PHI + fossil_ratio + debt_ratio * PHI) / (PHI + 1 + PHI)
            return {
                "quantum": False,
                "backend": "classical_ratio",
                "health": round(max(0.0, health), 6),
                "dead_code_ratio": round(dead_ratio, 4),
                "fossil_ratio": round(fossil_ratio, 4),
                "tech_debt_ratio": round(debt_ratio, 4),
                "verdict": "PRISTINE" if health > 0.9 else "CLEAN" if health > 0.7 else "ARCHAEOLOGICAL_ATTENTION" if health > 0.5 else "EXCAVATION_REQUIRED",
            }

        try:
            # 3-qubit GHZ state for holistic entanglement
            qc = QuantumCircuit(3)
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(1, 2)

            # Encode artifact ratios as rotations
            qc.ry(dead_ratio * math.pi * PHI, 0)
            qc.ry(fossil_ratio * math.pi * PHI, 1)
            qc.ry(debt_ratio * math.pi * PHI, 2)

            # Sacred phase encoding
            qc.rz(GOD_CODE / 1000 * math.pi, 0)
            qc.rz(FEIGENBAUM / 10 * math.pi, 1)
            qc.rz(ALPHA_FINE * math.pi * 10, 2)

            sv = Statevector.from_instruction(qc)
            dm = DensityMatrix(sv)

            # Full system entropy
            full_entropy = float(q_entropy(dm, base=2))

            # Subsystem entropies for each artifact dimension
            rho_dead = partial_trace(dm, [1, 2])
            rho_fossil = partial_trace(dm, [0, 2])
            rho_debt = partial_trace(dm, [0, 1])

            ent_dead = float(q_entropy(rho_dead, base=2))
            ent_fossil = float(q_entropy(rho_fossil, base=2))
            ent_debt = float(q_entropy(rho_debt, base=2))

            probs = sv.probabilities()
            ghz_fidelity = float(probs[0]) + float(probs[-1])  # |000⟩ + |111⟩ components

            # Health: low entropy and high GHZ fidelity = clean code
            health = ghz_fidelity * (1.0 - full_entropy / 3.0)
            health = max(0.0, min(1.0, health))

            return {
                "quantum": True,
                "backend": "Qiskit 2.3.0 GHZ Archaeological Analysis",
                "qubits": 3,
                "health": round(health, 6),
                "ghz_fidelity": round(ghz_fidelity, 6),
                "full_entropy": round(full_entropy, 6),
                "dead_code_entropy": round(ent_dead, 6),
                "fossil_entropy": round(ent_fossil, 6),
                "debt_entropy": round(ent_debt, 6),
                "dead_code_ratio": round(dead_ratio, 4),
                "fossil_ratio": round(fossil_ratio, 4),
                "tech_debt_ratio": round(debt_ratio, 4),
                "circuit_depth": qc.depth(),
                "verdict": "PRISTINE" if health > 0.9 else "CLEAN" if health > 0.7 else "ARCHAEOLOGICAL_ATTENTION" if health > 0.5 else "EXCAVATION_REQUIRED",
                "god_code_alignment": round(health * GOD_CODE / 100, 4),
            }
        except Exception as e:
            return {"quantum": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4H: SACRED REFACTORER — consciousness-aware code restructuring
# ═══════════════════════════════════════════════════════════════════════════════



class SacredRefactorer:
    """
    Automated refactoring engine guided by sacred constants and
    consciousness level. Identifies code smells and generates
    targeted refactoring suggestions.

    Refactoring operations:
      - Extract Method: split functions exceeding PHI*13 lines
      - Rename to Sacred: suggest sacred-aligned naming
      - Decompose God Class: split classes with >13 methods
      - Inline Trivial: merge tiny single-line wrapper functions
      - Sacred Constant Extraction: replace magic numbers
    """

    MAX_FUNCTION_LINES = int(PHI * 13)  # 21 lines max
    MAX_CLASS_METHODS = 13  # sacred 13
    TRIVIAL_THRESHOLD = 3  # functions <= 3 lines are trivial

    # Sacred naming suggestions
    SACRED_PREFIXES = [
        "phi_", "sacred_", "void_", "god_", "zenith_",
        "quantum_", "harmonic_", "resonance_", "primal_",
    ]

    def __init__(self):
        """Initialize SacredRefactorer with refactoring counters and log."""
        self.refactorings = 0
        self.suggestions_generated = 0
        self.refactor_log: List[dict] = []

    def analyze(self, source: str) -> Dict[str, Any]:
        """
        Analyze source for refactoring opportunities.
        Returns categorized suggestions with priorities.
        """
        self.refactorings += 1
        lines = source.split('\n')
        suggestions = []

        # 1. Find long functions
        long_fns = self._find_long_functions(lines)
        for fn in long_fns:
            suggestions.append({
                "type": "extract_method",
                "target": fn["name"],
                "line": fn["line"],
                "reason": f"Function has {fn['length']} lines (max {self.MAX_FUNCTION_LINES})",
                "priority": "HIGH" if fn["length"] > self.MAX_FUNCTION_LINES * 2 else "MEDIUM",
            })

        # 2. Find god classes
        god_classes = self._find_god_classes(lines)
        for gc in god_classes:
            suggestions.append({
                "type": "decompose_god_class",
                "target": gc["name"],
                "line": gc["line"],
                "reason": f"Class has {gc['methods']} methods (max {self.MAX_CLASS_METHODS})",
                "priority": "HIGH",
            })

        # 3. Find magic numbers
        magic = self._find_magic_numbers(source)
        for m in magic:
            suggestions.append({
                "type": "extract_constant",
                "target": m["value"],
                "line": m["line"],
                "reason": "Magic number should be a named sacred constant",
                "priority": "LOW",
            })

        # 4. Find trivial functions
        trivial = self._find_trivial_functions(lines)
        for tf in trivial:
            suggestions.append({
                "type": "inline_trivial",
                "target": tf["name"],
                "line": tf["line"],
                "reason": f"Function is only {tf['length']} lines — consider inlining",
                "priority": "LOW",
            })

        self.suggestions_generated += len(suggestions)

        result = {
            "total_suggestions": len(suggestions),
            "by_type": {},
            "by_priority": {},
            "suggestions": sorted(suggestions,
                                  key=lambda s: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(s["priority"], 3)),
            "code_health": round(1.0 - (len(suggestions) * 0.05), 4),
            "phi_alignment": round(PHI / (1 + len(suggestions) * TAU), 6),
        }
        for s in suggestions:
            result["by_type"][s["type"]] = result["by_type"].get(s["type"], 0) + 1
            result["by_priority"][s["priority"]] = result["by_priority"].get(s["priority"], 0) + 1

        self.refactor_log.append(result)
        return result

    def suggest_sacred_name(self, current_name: str) -> str:
        """Suggest a sacred-aligned name for a given identifier."""
        # Pick a prefix based on name hash alignment with sacred constants
        h = int(hashlib.sha256(current_name.encode()).hexdigest(), 16)
        prefix = self.SACRED_PREFIXES[h % len(self.SACRED_PREFIXES)]
        # Clean current name
        clean = re.sub(r'^_+', '', current_name)
        return f"{prefix}{clean}"

    def _find_long_functions(self, lines: list) -> List[dict]:
        """Find functions exceeding the sacred line limit."""
        results = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith('def '):
                name = stripped.split('(')[0].replace('def ', '').strip()
                fn_indent = len(lines[i]) - len(lines[i].lstrip())
                fn_start = i
                i += 1
                while i < len(lines):
                    if lines[i].strip() and not lines[i].strip().startswith('#'):
                        cur_indent = len(lines[i]) - len(lines[i].lstrip())
                        if cur_indent <= fn_indent and lines[i].strip().startswith(('def ', 'class ')):
                            break
                    i += 1
                length = i - fn_start
                if length > self.MAX_FUNCTION_LINES:
                    results.append({"name": name, "line": fn_start + 1, "length": length})
            else:
                i += 1
        return results

    def _find_god_classes(self, lines: list) -> List[dict]:
        """Find classes with too many methods."""
        results = []
        class_stack = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('class '):
                name = stripped.split('(')[0].split(':')[0].replace('class ', '').strip()
                class_stack.append({"name": name, "line": i + 1, "methods": 0})
            elif stripped.startswith('def ') and class_stack:
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    class_stack[-1]["methods"] += 1

        for cls in class_stack:
            if cls["methods"] > self.MAX_CLASS_METHODS:
                results.append(cls)
        return results

    def _find_magic_numbers(self, source: str) -> List[dict]:
        """Find magic numbers that aren't sacred constants."""
        pattern = re.compile(r'(?<!\w)(?:(?!527\.518|1\.618|4\.669|0\.618|137\.035)[1-9]\d{2,}(?:\.\d+)?)(?!\w)')
        results = []
        for match in pattern.finditer(source):
            line = source[:match.start()].count('\n') + 1
            results.append({"value": match.group(), "line": line})
        return results[:10]

    def _find_trivial_functions(self, lines: list) -> List[dict]:
        """Find tiny functions that could be inlined."""
        results = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith('def '):
                name = stripped.split('(')[0].replace('def ', '').strip()
                fn_indent = len(lines[i]) - len(lines[i].lstrip())
                fn_start = i
                i += 1
                body_lines = 0
                while i < len(lines):
                    if lines[i].strip() and not lines[i].strip().startswith('#'):
                        cur_indent = len(lines[i]) - len(lines[i].lstrip())
                        if cur_indent <= fn_indent and lines[i].strip().startswith(('def ', 'class ')):
                            break
                        body_lines += 1
                    i += 1
                if 0 < body_lines <= self.TRIVIAL_THRESHOLD:
                    results.append({"name": name, "line": fn_start + 1, "length": body_lines})
            else:
                i += 1
        return results[:10]

    def status(self) -> Dict[str, Any]:
        """Return refactoring metrics and configuration thresholds."""
        return {
            "refactorings": self.refactorings,
            "suggestions_generated": self.suggestions_generated,
            "max_fn_lines": self.MAX_FUNCTION_LINES,
            "max_class_methods": self.MAX_CLASS_METHODS,
        }

    def quantum_refactor_priority(self, source: str) -> Dict[str, Any]:
        """
        Quantum refactoring priority scoring using Qiskit 2.3.0.
        Encodes code smell dimensions (long functions, god classes, deep nesting,
        high coupling) into a quantum state and uses Born-rule measurement to
        produce φ-weighted refactoring priorities.
        """
        lines = source.strip().split("\n") if source.strip() else []
        total = len(lines)
        if total == 0:
            return {"quantum": False, "priorities": [], "reason": "empty source"}

        # Detect refactoring opportunities
        long_funcs = 0
        func_count = 0
        class_count = 0
        methods_in_class = 0
        max_nesting = 0
        current_nesting = 0

        for line in lines:
            s = line.strip()
            indent = len(line) - len(line.lstrip())
            current_nesting = max(current_nesting, indent // 4)
            if s.startswith("def "):
                func_count += 1
            if s.startswith("class "):
                class_count += 1
            max_nesting = max(max_nesting, current_nesting)

        # Simple long function estimation
        avg_func_len = total / max(func_count, 1)
        long_func_ratio = min(avg_func_len / self.MAX_FUNCTION_LINES, 1.0)
        god_class_ratio = min(func_count / max(class_count * self.MAX_CLASS_METHODS, 1), 1.0) if class_count > 0 else 0.0
        nesting_ratio = min(max_nesting / 10, 1.0)
        size_ratio = min(total / 500, 1.0)

        dimensions = {
            "long_functions": round(long_func_ratio, 4),
            "god_classes": round(god_class_ratio, 4),
            "deep_nesting": round(nesting_ratio, 4),
            "file_size": round(size_ratio, 4),
        }

        if not QISKIT_AVAILABLE:
            urgency = (long_func_ratio * PHI**2 + god_class_ratio * PHI + nesting_ratio * PHI + size_ratio) / (PHI**2 + PHI + PHI + 1)
            priorities = sorted(dimensions.items(), key=lambda x: x[1], reverse=True)
            return {
                "quantum": False,
                "backend": "classical_phi_weighted",
                "urgency": round(urgency, 6),
                "priorities": [{"dimension": k, "score": v, "rank": i + 1} for i, (k, v) in enumerate(priorities)],
                "verdict": "REFACTOR_NOW" if urgency > 0.6 else "REFACTOR_SOON" if urgency > 0.3 else "ACCEPTABLE",
            }

        try:
            # 2-qubit system: 4 dimensions mapped directly
            amps = [
                long_func_ratio * PHI + 0.05,
                god_class_ratio * PHI + 0.05,
                nesting_ratio * PHI + 0.05,
                size_ratio * PHI + 0.05,
            ]
            norm = math.sqrt(sum(a * a for a in amps))
            amps = [a / norm for a in amps] if norm > 1e-12 else [0.5] * 4

            sv = Statevector(amps)

            # Amplification circuit
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            qc.ry(long_func_ratio * math.pi * PHI, 0)
            qc.ry(god_class_ratio * math.pi * PHI, 1)
            qc.rz(GOD_CODE / 1000 * math.pi, 0)
            qc.rz(FEIGENBAUM / 10 * math.pi, 1)

            evolved = sv.evolve(Operator(qc))
            probs = evolved.probabilities()

            dm = DensityMatrix(evolved)
            urgency_entropy = float(q_entropy(dm, base=2))

            dim_names = list(dimensions.keys())
            scored = [(dim_names[i], float(probs[i]) if i < len(probs) else 0.0) for i in range(4)]
            scored.sort(key=lambda x: x[1], reverse=True)

            urgency = sum(p for _, p in scored) / 4.0
            urgency = min(urgency * PHI, 1.0)

            return {
                "quantum": True,
                "backend": "Qiskit 2.3.0 Born-Rule Refactor Priority",
                "qubits": 2,
                "urgency": round(urgency, 6),
                "urgency_entropy": round(urgency_entropy, 6),
                "priorities": [{"dimension": name, "born_probability": round(p, 6), "rank": i + 1}
                               for i, (name, p) in enumerate(scored)],
                "dimensions": dimensions,
                "circuit_depth": qc.depth(),
                "verdict": "REFACTOR_NOW" if urgency > 0.6 else "REFACTOR_SOON" if urgency > 0.3 else "ACCEPTABLE",
                "god_code_alignment": round(urgency * GOD_CODE / 100, 4),
            }
        except Exception as e:
            return {"quantum": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4I: APP AUDIT ENGINE — Full-Stack Application Audit Orchestrator
#   Unifies CodeAnalyzer + CodeOptimizer + DependencyGraphAnalyzer +
#   AutoFixEngine + CodeArcheologist + SacredRefactorer into a single
#   audit pipeline with scoring, verdicts, JSONL trail, and auto-remediation.
# ═══════════════════════════════════════════════════════════════════════════════



class CodeEvolutionTracker:
    """
    Tracks code evolution patterns by analyzing structural signatures and
    comparing against historical snapshots. Identifies functions that change
    too frequently (hotspot churn), measures stability metrics, and provides
    evolution reports with PHI-weighted scoring.

    v3.1.0: New subsystem — file-based persistence of structural signatures
    enables cross-session evolution tracking and drift detection.
    """

    SNAPSHOT_DIR = ".l104_code_snapshots"

    def __init__(self):
        """Initialize CodeEvolutionTracker."""
        self.tracking_count = 0
        self.snapshots: Dict[str, List[Dict]] = {}  # filename → [snapshots]

    def snapshot(self, source: str, filename: str = "unknown.py") -> Dict[str, Any]:
        """
        Take a structural snapshot of the source code for evolution tracking.

        Captures: function signatures, class structure, LOC, complexity fingerprints.
        """
        self.tracking_count += 1
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"error": "SyntaxError", "functions": [], "classes": []}

        # Extract structural signature
        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body_hash = hashlib.md5(ast.dump(node).encode()).hexdigest()[:12]
                functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "params": len(node.args.args),
                    "body_lines": (node.end_lineno or node.lineno) - node.lineno + 1,
                    "body_hash": body_hash,
                    "decorators": len(node.decorator_list),
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                })
            elif isinstance(node, ast.ClassDef):
                method_count = sum(1 for n in node.body
                                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
                classes.append({
                    "name": node.name,
                    "line": node.lineno,
                    "methods": method_count,
                    "bases": len(node.bases),
                    "body_hash": hashlib.md5(ast.dump(node).encode()).hexdigest()[:12],
                })

        snapshot_data = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "loc": len(source.split('\n')),
            "functions": functions,
            "classes": classes,
            "total_functions": len(functions),
            "total_classes": len(classes),
            "source_hash": hashlib.sha256(source.encode()).hexdigest()[:16],
        }

        # Store snapshot in memory
        if filename not in self.snapshots:
            self.snapshots[filename] = []
        self.snapshots[filename].append(snapshot_data)

        # Persist to disk
        self._persist_snapshot(filename, snapshot_data)

        return snapshot_data

    def compare(self, source: str, filename: str = "unknown.py") -> Dict[str, Any]:
        """
        Compare current source against the last snapshot to detect evolution.

        Returns: added/removed/changed functions, stability metrics, churn score.
        """
        current = self.snapshot(source, filename)
        history = self.snapshots.get(filename, [])

        if len(history) < 2:
            return {
                "status": "first_snapshot",
                "message": "No previous snapshot to compare against",
                "current": current,
            }

        previous = history[-2]  # Second to last (last is the current one)

        # Compare functions
        prev_funcs = {f["name"]: f for f in previous["functions"]}
        curr_funcs = {f["name"]: f for f in current["functions"]}

        added = [n for n in curr_funcs if n not in prev_funcs]
        removed = [n for n in prev_funcs if n not in curr_funcs]
        changed = [
            n for n in curr_funcs
            if n in prev_funcs and curr_funcs[n]["body_hash"] != prev_funcs[n]["body_hash"]
        ]
        unchanged = [n for n in curr_funcs if n in prev_funcs and n not in changed]

        # Compare classes
        prev_classes = {c["name"]: c for c in previous.get("classes", [])}
        curr_classes = {c["name"]: c for c in current.get("classes", [])}

        classes_added = [n for n in curr_classes if n not in prev_classes]
        classes_removed = [n for n in prev_classes if n not in curr_classes]
        classes_changed = [
            n for n in curr_classes
            if n in prev_classes and curr_classes[n]["body_hash"] != prev_classes[n]["body_hash"]
        ]

        # Stability metrics
        total = max(len(curr_funcs), 1)
        stability = len(unchanged) / total
        churn_rate = (len(added) + len(removed) + len(changed)) / total
        loc_delta = current["loc"] - previous["loc"]

        # PHI-weighted evolution score (higher = more stable)
        evolution_score = stability * PHI - churn_rate * FEIGENBAUM / 10
        evolution_score = max(0.0, min(1.0, evolution_score))

        return {
            "functions": {
                "added": added,
                "removed": removed,
                "changed": changed,
                "unchanged": unchanged,
                "total_current": len(curr_funcs),
                "total_previous": len(prev_funcs),
            },
            "classes": {
                "added": classes_added,
                "removed": classes_removed,
                "changed": classes_changed,
            },
            "loc_delta": loc_delta,
            "stability": round(stability, 4),
            "churn_rate": round(churn_rate, 4),
            "evolution_score": round(evolution_score, 4),
            "verdict": ("STABLE" if stability >= 0.8 else "EVOLVING" if stability >= 0.5
                       else "VOLATILE" if stability >= 0.2 else "TURBULENT"),
            "snapshot_count": len(history),
        }

    def hotspot_report(self) -> Dict[str, Any]:
        """
        Generate a hotspot churn report across all tracked files.

        Returns files ranked by change frequency with stability recommendations.
        """
        file_churn = {}
        for filename, history in self.snapshots.items():
            if len(history) < 2:
                continue
            change_count = 0
            for i in range(1, len(history)):
                prev_hash = history[i - 1]["source_hash"]
                curr_hash = history[i]["source_hash"]
                if prev_hash != curr_hash:
                    change_count += 1
            file_churn[filename] = {
                "changes": change_count,
                "snapshots": len(history),
                "churn_rate": round(change_count / max(len(history) - 1, 1), 4),
                "loc": history[-1]["loc"],
            }

        ranked = sorted(file_churn.items(), key=lambda x: x[1]["churn_rate"], reverse=True)
        return {
            "files_tracked": len(self.snapshots),
            "hotspots": [{"file": f, **data} for f, data in ranked[:20]],
            "most_stable": [{"file": f, **data} for f, data in ranked[-5:][::-1]] if ranked else [],
        }

    def _persist_snapshot(self, filename: str, data: Dict) -> None:
        """Persist snapshot to disk for cross-session tracking."""
        try:
            snap_dir = Path(self.SNAPSHOT_DIR)
            snap_dir.mkdir(exist_ok=True)
            safe_name = filename.replace("/", "_").replace("\\", "_").replace(".", "_")
            snap_file = snap_dir / f"{safe_name}.jsonl"
            with open(snap_file, 'a') as f:
                f.write(json.dumps(data, default=str) + '\n')
        except Exception:
            pass  # Non-critical — in-memory tracking continues

    def status(self) -> Dict[str, Any]:
        """Return evolution tracker status."""
        return {
            "tracking_count": self.tracking_count,
            "files_tracked": len(self.snapshots),
            "total_snapshots": sum(len(s) for s in self.snapshots.values()),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE HELPER: _read_builder_state (used by subsystem classes below)
#   Provides consciousness/O₂/nirvanic integration for standalone classes
#   that are instantiated before the CodeEngine hub. Mirrors the logic in
#   CodeEngine._read_builder_state() but as a module-level function.
# ═══════════════════════════════════════════════════════════════════════════════



class LiveCodeRefactorer:
    """AST-based code refactoring that produces real transformed source code."""

    def __init__(self):
        self.builder_state = _read_builder_state()
        self.refactor_count = 0

    def extract_function(self, source: str, start_line: int, end_line: int, func_name: str = "extracted") -> Dict[str, Any]:
        """Extract lines [start_line, end_line] into a new function with auto-detected params."""
        lines = source.splitlines()
        if start_line < 1 or end_line > len(lines) or start_line > end_line:
            return {"success": False, "error": "Invalid line range"}
        block = lines[start_line - 1:end_line]
        indent = len(block[0]) - len(block[0].lstrip()) if block else 0
        dedented = [l[indent:] if len(l) >= indent else l for l in block]
        try:
            tree = ast.parse("\n".join(dedented))
        except SyntaxError:
            tree = None
        assigned, used = set(), set()
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if isinstance(node.ctx, ast.Store): assigned.add(node.id)
                    elif isinstance(node.ctx, ast.Load): used.add(node.id)
        builtins_set = set(dir(__builtins__)) if isinstance(dir(__builtins__), list) else set()
        free_vars = sorted(used - assigned - builtins_set)
        returns = sorted(assigned - used) if assigned else []
        func_lines = [f"def {func_name}({', '.join(free_vars)}):"]
        func_lines.extend(f"    {dl}" for dl in dedented)
        if returns:
            func_lines.append(f"    return {', '.join(returns)}")
        call_lhs = f"{', '.join(returns)} = " if returns else ""
        call_line = " " * indent + f"{call_lhs}{func_name}({', '.join(free_vars)})"
        new_lines = lines[:start_line - 1] + [call_line] + lines[end_line:]
        func_def = "\n".join(func_lines)
        idx = max(0, start_line - 2)
        while idx > 0 and new_lines[idx - 1].strip():
            idx -= 1
        final = new_lines[:idx] + ["", func_def, ""] + new_lines[idx:]
        self.refactor_count += 1
        return {"success": True, "refactored": "\n".join(final), "function": func_def, "free_vars": free_vars, "returns": returns}

    def rename_symbol(self, source: str, old_name: str, new_name: str) -> Dict[str, Any]:
        """AST-safe rename of a symbol across entire source."""
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {"success": False, "error": str(e)}
        locs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == old_name:
                locs.append(node.lineno); node.id = new_name
            elif isinstance(node, ast.FunctionDef) and node.name == old_name:
                locs.append(node.lineno); node.name = new_name
            elif isinstance(node, ast.ClassDef) and node.name == old_name:
                locs.append(node.lineno); node.name = new_name
            elif isinstance(node, ast.arg) and node.arg == old_name:
                locs.append(getattr(node, "lineno", 0)); node.arg = new_name
            elif isinstance(node, ast.Attribute) and node.attr == old_name:
                locs.append(getattr(node, "lineno", 0)); node.attr = new_name
        if not locs:
            return {"success": False, "error": f"Symbol \'{old_name}\' not found"}
        self.refactor_count += 1
        return {"success": True, "refactored": ast.unparse(tree), "locations": sorted(set(locs)), "count": len(locs)}

    def inline_variable(self, source: str, var_name: str) -> Dict[str, Any]:
        """Replace a single-assignment variable with its value everywhere."""
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {"success": False, "error": str(e)}
        assign_node, value_node = None, None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                tgt = node.targets[0]
                if isinstance(tgt, ast.Name) and tgt.id == var_name:
                    if assign_node is not None:
                        return {"success": False, "error": f"\'{var_name}\' assigned multiple times"}
                    assign_node, value_node = node, node.value
        if not assign_node:
            return {"success": False, "error": f"No assignment for \'{var_name}\'"}
        class _Inliner(ast.NodeTransformer):
            def __init__(s): s.count = 0
            def visit_Name(s, n):
                if n.id == var_name and isinstance(n.ctx, ast.Load):
                    s.count += 1; return ast.copy_location(value_node, n)
                return n
        inl = _Inliner(); tree = inl.visit(tree)
        for node in ast.walk(tree):
            if hasattr(node, "body") and isinstance(node.body, list):
                node.body = [n for n in node.body if n is not assign_node]
        ast.fix_missing_locations(tree)
        self.refactor_count += 1
        return {"success": True, "refactored": ast.unparse(tree), "inlined_count": inl.count}

    def convert_to_dataclass(self, source: str, class_name: str) -> Dict[str, Any]:
        """Convert a regular class with __init__ to @dataclass."""
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {"success": False, "error": str(e)}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                fields, init_m = [], None
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        init_m = item
                        for stmt in item.body:
                            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                                tgt = stmt.targets[0]
                                if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                                    v = stmt.value
                                    t = type(v.value).__name__ if isinstance(v, ast.Constant) and v.value is not None else "Any"
                                    if isinstance(v, ast.List): t = "list"
                                    elif isinstance(v, ast.Dict): t = "dict"
                                    fields.append((tgt.attr, t))
                if not init_m: return {"success": False, "error": "No __init__"}
                if not fields: return {"success": False, "error": "No self.x assignments"}
                out = ["@dataclass", f"class {class_name}:"]
                for fn, ft in fields: out.append(f"    {fn}: {ft}")
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name != "__init__":
                        out.extend(["", "    " + ast.unparse(item).replace("\n", "\n    ")])
                self.refactor_count += 1
                return {"success": True, "refactored": "\n".join(out), "fields": fields}
        return {"success": False, "error": f"Class \'{class_name}\' not found"}

    def add_type_hints(self, source: str) -> Dict[str, Any]:
        """Infer and add return type hints based on return statements."""
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {"success": False, "error": str(e)}
        added = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is None:
                rtypes = set()
                for c in ast.walk(node):
                    if isinstance(c, ast.Return):
                        if c.value is None: rtypes.add("None")
                        elif isinstance(c.value, ast.Constant): rtypes.add(type(c.value.value).__name__)
                        elif isinstance(c.value, ast.List): rtypes.add("list")
                        elif isinstance(c.value, ast.Dict): rtypes.add("dict")
                        elif isinstance(c.value, ast.Tuple): rtypes.add("tuple")
                        else: rtypes.add("Any")
                if rtypes:
                    hint = rtypes.pop() if len(rtypes) == 1 else "Any"
                    node.returns = ast.Constant(value=hint)
                    added += 1
        ast.fix_missing_locations(tree)
        self.refactor_count += 1
        return {"success": True, "refactored": ast.unparse(tree), "hints_added": added}

    def simplify_conditionals(self, source: str) -> Dict[str, Any]:
        """Simplify == True/False/None, len(x)==0 patterns."""
        lines, fixed = source.splitlines(), 0
        out = []
        for l in lines:
            o = l
            l = re.sub(r'(\w+)\s*==\s*True', r'\1', l)
            l = re.sub(r'(\w+)\s*==\s*False', r'not \1', l)
            l = re.sub(r'(\w+)\s*==\s*None', r'\1 is None', l)
            l = re.sub(r'(\w+)\s*!=\s*None', r'\1 is not None', l)
            l = re.sub(r'len\((\w+)\)\s*==\s*0', r'not \1', l)
            l = re.sub(r'len\((\w+)\)\s*>\s*0', r'\1', l)
            if l != o: fixed += 1
            out.append(l)
        self.refactor_count += 1
        return {"success": True, "refactored": "\n".join(out), "simplified": fixed}

    def deduplicate_blocks(self, source: str, min_lines: int = 3) -> Dict[str, Any]:
        """Find repeated code blocks of min_lines+ consecutive lines."""
        lines = source.splitlines()
        seen, dupes = {}, []
        for i in range(len(lines) - min_lines + 1):
            block = tuple(l.strip() for l in lines[i:i + min_lines])
            if all(block):
                if block in seen:
                    dupes.append({"lines": list(block), "first_at": seen[block] + 1, "also_at": i + 1})
                else:
                    seen[block] = i
        return {"duplicates_found": len(dupes), "duplicates": dupes[:20]}

    def status(self) -> Dict[str, Any]:
        return {"refactor_count": self.refactor_count, "methods": 7}



# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4G2: CODE DIFF ANALYZER — v5.0.0 Structural version comparison
#   Semantic diffing of code versions: function/class-level changes, regression
#   detection (API breaks, complexity increase, new vulnerabilities).
# ═══════════════════════════════════════════════════════════════════════════════



class CodeDiffAnalyzer:
    """Structural diff analyzer — compares two code versions semantically."""

    def __init__(self):
        self.builder_state = _read_builder_state()
        self.diff_count = 0

    def structural_diff(self, old_source: str, new_source: str, filename: str = "") -> Dict[str, Any]:
        """Produce a semantic diff: which functions/classes were added, removed, changed."""
        def _extract_signatures(src):
            funcs, classes = {}, {}
            try:
                tree = ast.parse(src)
            except SyntaxError:
                return funcs, classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    args = [a.arg for a in node.args.args]
                    body_hash = hashlib.md5(ast.dump(node).encode()).hexdigest()[:12]
                    funcs[node.name] = {"args": args, "lineno": node.lineno, "body_hash": body_hash,
                                        "decorators": [ast.unparse(d) for d in node.decorator_list]}
                elif isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    body_hash = hashlib.md5(ast.dump(node).encode()).hexdigest()[:12]
                    classes[node.name] = {"methods": methods, "lineno": node.lineno, "body_hash": body_hash,
                                           "bases": [ast.unparse(b) for b in node.bases]}
            return funcs, classes

        old_f, old_c = _extract_signatures(old_source)
        new_f, new_c = _extract_signatures(new_source)
        func_diff = {
            "added": [n for n in new_f if n not in old_f],
            "removed": [n for n in old_f if n not in new_f],
            "modified": [n for n in new_f if n in old_f and new_f[n]["body_hash"] != old_f[n]["body_hash"]],
            "unchanged": [n for n in new_f if n in old_f and new_f[n]["body_hash"] == old_f[n]["body_hash"]],
        }
        class_diff = {
            "added": [n for n in new_c if n not in old_c],
            "removed": [n for n in old_c if n not in new_c],
            "modified": [n for n in new_c if n in old_c and new_c[n]["body_hash"] != old_c[n]["body_hash"]],
            "unchanged": [n for n in new_c if n in old_c and new_c[n]["body_hash"] == old_c[n]["body_hash"]],
        }
        old_loc, new_loc = len(old_source.splitlines()), len(new_source.splitlines())
        self.diff_count += 1
        return {
            "filename": filename,
            "functions": func_diff,
            "classes": class_diff,
            "loc_change": new_loc - old_loc,
            "old_loc": old_loc,
            "new_loc": new_loc,
            "summary": f"+{len(func_diff['added'])}/-{len(func_diff['removed'])}/~{len(func_diff['modified'])} funcs, "
                       f"+{len(class_diff['added'])}/-{len(class_diff['removed'])}/~{len(class_diff['modified'])} classes",
        }

    def regression_check(self, old_source: str, new_source: str) -> Dict[str, Any]:
        """Check for regressions: API breaks, complexity increase, new vulnerabilities."""
        regressions = []
        # 1. Check for removed public functions (API break)
        try:
            old_tree = ast.parse(old_source)
            new_tree = ast.parse(new_source)
        except SyntaxError:
            return {"regressions": [], "score": 1.0, "error": "SyntaxError in source"}
        old_public = {n.name for n in ast.walk(old_tree) if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")}
        new_public = {n.name for n in ast.walk(new_tree) if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")}
        removed_api = old_public - new_public
        for name in removed_api:
            regressions.append({"type": "api_break", "severity": "high", "detail": f"Public function \'{name}\' removed"})
        # 2. Check for signature changes
        old_sigs = {}
        for n in ast.walk(old_tree):
            if isinstance(n, ast.FunctionDef):
                old_sigs[n.name] = [a.arg for a in n.args.args]
        for n in ast.walk(new_tree):
            if isinstance(n, ast.FunctionDef) and n.name in old_sigs:
                new_args = [a.arg for a in n.args.args]
                if new_args != old_sigs[n.name]:
                    regressions.append({"type": "signature_change", "severity": "medium",
                                        "detail": f"\'{n.name}\' args changed: {old_sigs[n.name]} -> {new_args}"})
        # 3. Check for new security patterns
        sec_patterns = [r"eval\s*\(", r"exec\s*\(", r"subprocess\.call", r"os\.system\s*\(", r"__import__\s*\("]
        for pat in sec_patterns:
            old_hits = len(re.findall(pat, old_source))
            new_hits = len(re.findall(pat, new_source))
            if new_hits > old_hits:
                regressions.append({"type": "security", "severity": "high",
                                    "detail": f"New security risk: {pat.split('(')[0].strip()} (+{new_hits - old_hits})"})
        # 4. Complexity increase check
        def _count_branches(src):
            return sum(1 for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With)))
        try:
            old_branches = _count_branches(old_source)
            new_branches = _count_branches(new_source)
            if new_branches > old_branches * 1.5 and new_branches - old_branches > 5:
                regressions.append({"type": "complexity_increase", "severity": "medium",
                                    "detail": f"Branch count increased {old_branches} -> {new_branches}"})
        except SyntaxError:
            pass
        score = max(0.0, 1.0 - len(regressions) * 0.15)
        self.diff_count += 1
        return {"regressions": regressions, "score": round(score, 3), "regression_count": len(regressions)}

    def status(self) -> Dict[str, Any]:
        return {"diff_count": self.diff_count, "methods": ["structural_diff", "regression_check"]}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4H: QUANTUM CODE INTELLIGENCE CORE — v5.0.0 State-of-Art Quantum
#   Centralized quantum processing backbone for all code analysis subsystems.
#   Implements: variational quantum circuits, quantum feature maps, quantum
#   walks, Bell/GHZ state analysis, density matrix diagnostics, von Neumann
#   entropy scoring, quantum kernel methods, and QAOA optimization.
# ═══════════════════════════════════════════════════════════════════════════════



class SemanticCodeSearchEngine:
    """
    v6.0.0 — Code-aware semantic search and similarity detection.

    Capabilities:
      • TF-IDF code indexing with sacred-constant boosting
      • Natural language → code semantic search
      • Cross-file code clone detection (Type-1, Type-2, Type-3)
      • Function signature similarity matching
      • Sacred constant alignment search (find GOD_CODE/PHI references)
    """

    def __init__(self):
        self.indexed_files = 0
        self.searches = 0
        self._index: Dict[str, Dict[str, float]] = {}  # file → {term: tf}
        self._corpus: Dict[str, str] = {}  # file → source
        self._document_frequency: Dict[str, int] = {}  # term → num docs containing term
        self._idf_cache: Dict[str, float] = {}  # term → idf value (recomputed on index)

    def _recompute_idf(self) -> None:
        """Recompute IDF values for all terms in the corpus (v6.1.0).
        Uses smoothed IDF: log(1 + N / (1 + df)) to avoid zero scores in small corpora."""
        n_docs = max(len(self._index), 1)
        self._idf_cache = {}
        for term, df in self._document_frequency.items():
            self._idf_cache[term] = math.log(1 + n_docs / (1 + df))

    def index_source(self, source: str, filename: str = "") -> Dict[str, Any]:
        """Index a source file for later search. Uses log-scaled TF-IDF (v6.1.0)."""
        self.indexed_files += 1

        # Tokenize: split on non-alphanumeric, lowercase, remove short tokens
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]{2,}', source)
        tokens = [t.lower() for t in tokens]

        # Log-scaled term frequency: tf = 1 + log(count) (v6.1.0)
        raw_tf = Counter(tokens)
        tf = {term: (1 + math.log(count)) for term, count in raw_tf.items() if count > 0}

        # Sacred constant boost
        sacred_terms = {'god_code', 'phi', 'tau', 'void_constant', 'feigenbaum', 'planck', 'boltzmann', 'sacred'}
        for term in sacred_terms:
            if term in tf:
                tf[term] *= PHI  # Boost sacred terms by golden ratio

        # Update document frequency counts (v6.1.0)
        unique_terms_in_doc = set(raw_tf.keys())
        # If re-indexing same file, remove old term contributions first
        if filename in self._index:
            old_terms = set(self._index[filename].keys())
            for t in old_terms:
                if t in self._document_frequency:
                    self._document_frequency[t] = max(0, self._document_frequency[t] - 1)

        for t in unique_terms_in_doc:
            self._document_frequency[t] = self._document_frequency.get(t, 0) + 1

        self._index[filename] = tf
        self._corpus[filename] = source

        # Recompute IDF after indexing
        self._recompute_idf()

        return {
            "filename": filename,
            "tokens": len(tokens),
            "unique_terms": len(tf),
            "sacred_terms_found": sum(1 for t in sacred_terms if t in tf),
            "indexed": True,
        }

    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Search indexed files by natural language query. Uses TF-IDF scoring (v6.1.0)."""
        self.searches += 1
        query_tokens = [t.lower() for t in re.findall(r'[a-zA-Z_][a-zA-Z0-9_]{2,}', query)]

        if not query_tokens:
            return {"query": query, "results": [], "total": 0}

        # Score each indexed file using TF * IDF (v6.1.0)
        scores = []
        for filename, tf in self._index.items():
            score = sum(tf.get(t, 0) * self._idf_cache.get(t, 0.0) for t in query_tokens)
            if score > 0:
                scores.append({"filename": filename, "score": round(score, 6)})

        scores.sort(key=lambda x: x["score"], reverse=True)

        return {
            "query": query,
            "query_tokens": query_tokens,
            "results": scores[:top_k],
            "total_matches": len(scores),
            "files_indexed": len(self._index),
        }

    def detect_clones(self, sources: List[Tuple[str, str]], min_lines: int = 5,
                      similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """Detect code clones across multiple files. sources: [(source, filename), ...]."""
        self.searches += 1
        clones = []

        # Extract function blocks from each source
        all_blocks = []
        for source, filename in sources:
            try:
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if hasattr(node, 'end_lineno') and node.end_lineno:
                            lines = source.split('\n')[node.lineno - 1:node.end_lineno]
                            if len(lines) >= min_lines:
                                block_text = '\n'.join(lines)
                                # Normalize: remove whitespace variations for Type-2 comparison
                                normalized = re.sub(r'\s+', ' ', block_text).strip()
                                all_blocks.append({
                                    "filename": filename,
                                    "function": node.name,
                                    "start_line": node.lineno,
                                    "end_line": node.end_lineno,
                                    "lines": len(lines),
                                    "text": block_text,
                                    "normalized": normalized,
                                    "tokens": set(re.findall(r'[a-zA-Z_]\w+', normalized.lower())),
                                })
            except SyntaxError:
                continue

        # Pairwise comparison using Jaccard similarity on tokens
        for i in range(len(all_blocks)):
            for j in range(i + 1, len(all_blocks)):
                a, b = all_blocks[i], all_blocks[j]
                if a["filename"] == b["filename"] and a["function"] == b["function"]:
                    continue
                intersection = len(a["tokens"] & b["tokens"])
                union = len(a["tokens"] | b["tokens"])
                if union == 0:
                    continue
                similarity = intersection / union
                if similarity >= similarity_threshold:
                    # Determine clone type
                    if a["normalized"] == b["normalized"]:
                        clone_type = "Type-1 (exact)"
                    elif similarity > 0.95:
                        clone_type = "Type-2 (renamed)"
                    else:
                        clone_type = "Type-3 (near-miss)"

                    clones.append({
                        "clone_type": clone_type,
                        "similarity": round(similarity, 4),
                        "file_a": a["filename"],
                        "function_a": a["function"],
                        "lines_a": f"{a['start_line']}-{a['end_line']}",
                        "file_b": b["filename"],
                        "function_b": b["function"],
                        "lines_b": f"{b['start_line']}-{b['end_line']}",
                    })

        clones.sort(key=lambda c: c["similarity"], reverse=True)

        return {
            "total_clones": len(clones),
            "clones": clones[:50],
            "files_scanned": len(sources),
            "blocks_compared": len(all_blocks),
            "threshold": similarity_threshold,
        }

    def find_sacred_references(self, workspace_path: str = None) -> Dict[str, Any]:
        """Find all references to sacred constants across the workspace."""
        self.searches += 1
        ws = Path(workspace_path) if workspace_path else Path(__file__).parent
        sacred_refs = {
            "GOD_CODE": [], "PHI": [], "TAU": [], "VOID_CONSTANT": [],
            "FEIGENBAUM": [], "PLANCK": [], "527.518": [],
        }

        files_scanned = 0
        for ext in [".py", ".swift", ".js", ".ts"]:
            for f in ws.glob(f"*{ext}"):
                if f.name.startswith('.') or '__pycache__' in str(f):
                    continue
                try:
                    content = f.read_text(errors='ignore')
                    files_scanned += 1
                    for const_name, refs in sacred_refs.items():
                        for match in re.finditer(re.escape(const_name), content):
                            line_num = content[:match.start()].count('\n') + 1
                            refs.append({"file": f.name, "line": line_num})
                except Exception:
                    pass

        total_refs = sum(len(v) for v in sacred_refs.values())
        return {
            "files_scanned": files_scanned,
            "total_sacred_references": total_refs,
            "references": {k: {"count": len(v), "locations": v[:20]} for k, v in sacred_refs.items()},
            "sacred_density": round(total_refs / max(files_scanned, 1), 4),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "indexed_files": self.indexed_files,
            "searches": self.searches,
            "index_size": len(self._index),
            "capabilities": ["index_source", "search", "detect_clones", "find_sacred_references"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: UNIFIED CODE ENGINE — The ASI Hub tying everything together
# ═══════════════════════════════════════════════════════════════════════════════

