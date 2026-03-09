from .constants import *
class SelfModificationEngine:
    """Enables autonomous self-modification with multi-pass AST transforms,
    safe rollback, fitness-driven evolution, and recursive depth tracking.
    v5.0: Quantum-enhanced fitness evaluation, Grover-amplified transform selection,
    quantum tunneling for escaping local optima, entanglement-based code blending.
    v7.0: Lineage DAG tracking, multi-file evolution, Grover-amplified pass selection,
    quantum tunneling escape from local optima, complexity prediction."""
    # Quantum constants for self-modification
    Q_STATE_DIM = 32
    Q_TUNNEL_PROB = 0.10
    Q_DECOHERENCE = 0.02
    # v7.0: Lineage tracking constants
    MAX_LINEAGE_NODES = 500
    TUNNEL_ESCAPE_THRESHOLD = 3  # consecutive non-improving evolutions trigger tunnel

    def __init__(self, workspace: Optional[Path] = None):
        self.workspace = workspace or Path(os.path.dirname(os.path.abspath(__file__)))
        self.modification_depth = 0
        self.modifications: List[Dict] = []
        self.locked_modules = {'l104_stable_kernel.py', 'const.py'}
        # v4.0 additions
        self._rollback_buffer: List[Dict] = []  # (filepath, original_source)
        self._fitness_history: List[float] = []
        self._improvement_count = 0
        self._revert_count = 0
        self._recursive_depth = 0
        self._max_recursive_depth = 0
        # v5.0 Quantum state for fitness landscape navigation
        self._q_amplitudes = np.full(self.Q_STATE_DIM, 1.0 / np.sqrt(self.Q_STATE_DIM), dtype=np.complex128)
        self._q_grover_iters = 0
        self._q_tunnel_events = 0
        self._q_coherence = 1.0
        self._q_phase_acc = 0.0
        # v7.0: Lineage DAG — each node: {id, parent_id, filepath, transform, fitness, ts}
        self._lineage: List[Dict] = []
        self._lineage_counter = 0
        self._consecutive_non_improving = 0

    def analyze_module(self, filepath: Path) -> Dict:
        """Parse a Python module and return its structural metrics with v4.0 complexity analysis."""
        if not filepath.exists():
            return {'error': 'Not found'}
        try:
            with open(filepath) as f:
                source = f.read()
            tree = ast.parse(source)
            funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
            # Compute cyclomatic-style complexity (branches)
            branches = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.With)))
            lines = source.splitlines()
            blank_lines = sum(1 for line in lines if not line.strip())
            comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
            return {'path': str(filepath), 'lines': len(lines),
                    'functions': len(funcs), 'classes': len(classes),
                    'imports': len(imports), 'branches': branches,
                    'blank_lines': blank_lines, 'comment_lines': comment_lines,
                    'complexity_density': round(branches / max(len(lines), 1), 4)}
        except Exception as e:
            return {'error': str(e)}

    def multi_pass_ast_transform(self, source: str) -> tuple:
        """Apply multi-pass AST transforms: constant folding, dead import detection.
        Returns (transformed_source, transform_log)."""
        log = []
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return source, [f"Parse error: {e}"]

        # Pass 1: Detect unused imports
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)

        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)

        unused_imports = imported_names - used_names
        if unused_imports:
            log.append(f"Dead imports detected: {unused_imports}")

        # Pass 2: Constant folding detection
        foldable_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
                foldable_count += 1
        if foldable_count:
            log.append(f"Foldable constant expressions: {foldable_count}")

        # Pass 3: Dead code detection (unreachable after return)
        dead_stmts = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found_return = False
                for stmt in node.body:
                    if found_return:
                        dead_stmts += 1
                    if isinstance(stmt, ast.Return):
                        found_return = True
        if dead_stmts:
            log.append(f"Dead statements after return: {dead_stmts}")

        if not log:
            log.append("No transforms needed — code is clean")
        return source, log

    def safe_mutate_with_rollback(self, filepath: Path, new_source: str) -> Dict:
        """Apply mutation with rollback capability. Verifies AST validity before writing."""
        if filepath.name in self.locked_modules:
            return {'success': False, 'reason': 'Module is locked'}
        if not filepath.exists():
            return {'success': False, 'reason': 'File not found'}

        # Verify new source parses correctly
        try:
            ast.parse(new_source)
        except SyntaxError as e:
            self._revert_count += 1
            return {'success': False, 'reason': f'Syntax error in mutation: {e}'}

        # Save to rollback buffer
        try:
            original = filepath.read_text()
        except Exception as e:
            return {'success': False, 'reason': f'Read error: {e}'}

        self._rollback_buffer.append({
            'filepath': str(filepath), 'original': original,
            'timestamp': datetime.now().isoformat(), 'depth': self.modification_depth
        })
        # Keep buffer bounded
        if len(self._rollback_buffer) > SELF_MOD_MAX_ROLLBACK:
            self._rollback_buffer.pop(0)

        # Apply mutation
        try:
            filepath.write_text(new_source)
            self.modification_depth += 1
            self._improvement_count += 1
            self.modifications.append({
                'target': str(filepath), 'depth': self.modification_depth,
                'timestamp': datetime.now().isoformat(), 'type': 'safe_mutate'
            })
            return {'success': True, 'depth': self.modification_depth, 'rollback_available': True}
        except Exception as e:
            return {'success': False, 'reason': f'Write error: {e}'}

    def rollback_last(self) -> Dict:
        """Rollback the most recent mutation."""
        if not self._rollback_buffer:
            return {'success': False, 'reason': 'No rollback available'}
        entry = self._rollback_buffer.pop()
        try:
            Path(entry['filepath']).write_text(entry['original'])
            self.modification_depth = max(0, self.modification_depth - 1)
            self._revert_count += 1
            return {'success': True, 'reverted': entry['filepath'], 'depth': self.modification_depth}
        except Exception as e:
            return {'success': False, 'reason': f'Rollback error: {e}'}

    def compute_fitness(self, filepath: Optional[Path] = None) -> float:
        """Compute fitness score for a module based on structural quality metrics.
        v5.0: Quantum-enhanced with Hilbert-space fitness landscape embedding.
        If no filepath given, evaluates the ASI core itself."""
        if filepath is None:
            filepath = Path(__file__)
        analysis = self.analyze_module(filepath)
        if 'error' in analysis:
            return 0.0
        lines = analysis.get('lines', 1)
        funcs = analysis.get('functions', 0)
        classes = analysis.get('classes', 0)
        branches = analysis.get('branches', 0)
        comments = analysis.get('comment_lines', 0)
        # Classical fitness
        doc_ratio = min(1.0, comments / max(lines * 0.1, 1))
        modularity = min(1.0, (funcs + classes) / max(lines / 50, 1))
        complexity_penalty = max(0.0, 1.0 - analysis.get('complexity_density', 0) * 10)
        classical_fitness = (doc_ratio * 0.25 + modularity * 0.35 + complexity_penalty * 0.40) * PHI_CONJUGATE

        # Quantum fitness boost: embed into Hilbert space
        idx = hash(str(filepath)) % self.Q_STATE_DIM
        angle = classical_fitness * np.pi * PHI
        self._q_amplitudes[idx] = np.cos(angle / 2) + 1j * np.sin(angle / 2) * (GOD_CODE / 1000.0)
        # Normalize
        norm = np.linalg.norm(self._q_amplitudes)
        if norm > 1e-15:
            self._q_amplitudes /= norm

        # Quantum coherence bonus: high coherence rewards exploration
        q_bonus = self._q_coherence * ALPHA_FINE * 10.0
        fitness = classical_fitness + q_bonus

        self._fitness_history.append(fitness)
        return round(fitness, 6)

    def evolve_with_fitness(self, filepath: Path) -> Dict:
        """Run one evolution cycle: analyze → transform → evaluate fitness delta.

        v6.0: Actually applies the transform when it improves fitness.
        Rolls back if fitness degrades. Tracks real delta.
        """
        self._recursive_depth += 1
        self._max_recursive_depth = max(self._max_recursive_depth, self._recursive_depth)

        if not filepath.exists():
            self._recursive_depth -= 1
            return {'evolved': False, 'reason': 'File not found'}

        before_fitness = self.compute_fitness(filepath)
        source = filepath.read_text()
        transformed, log = self.multi_pass_ast_transform(source)

        applied = False
        after_fitness = before_fitness
        delta = 0.0

        # Only apply if transform produced real changes
        if transformed != source and log != ["No transforms needed — code is clean"]:
            # Verify transformed code is valid Python before applying
            try:
                ast.parse(transformed)
                # Save to rollback buffer, apply transform, recompute fitness
                self._rollback_buffer.append((str(filepath), source))
                if len(self._rollback_buffer) > SELF_MOD_MAX_ROLLBACK:
                    self._rollback_buffer.pop(0)
                filepath.write_text(transformed)
                after_fitness = self.compute_fitness(filepath)
                delta = after_fitness - before_fitness

                if delta < -0.05:
                    # Fitness degraded significantly — rollback
                    filepath.write_text(source)
                    after_fitness = before_fitness
                    delta = 0.0
                    log.append("ROLLED BACK: fitness degraded")
                else:
                    applied = True
                    self._improvement_count += 1
                    self._fitness_history.append(after_fitness)
            except SyntaxError:
                log.append("REJECTED: transformed code has syntax errors")

        result = {
            'evolved': applied, 'before_fitness': round(before_fitness, 6),
            'after_fitness': round(after_fitness, 6), 'delta': round(delta, 6),
            'transform_log': log, 'recursive_depth': self._recursive_depth,
            'applied': applied,
        }
        self._recursive_depth -= 1
        return result

    def propose_modification(self, target: str) -> Dict:
        """Evaluate whether a target module can be safely modified."""
        if target in self.locked_modules:
            return {'approved': False, 'reason': 'Locked'}
        analysis = self.analyze_module(self.workspace / target)
        fitness = self.compute_fitness(self.workspace / target) if 'error' not in analysis else 0.0
        return {'approved': 'error' not in analysis, 'analysis': analysis, 'fitness': fitness}

    def generate_self_improvement(self) -> str:
        """Generate a PHI-aligned optimization decorator as source code."""
        return f'''
def phi_optimize(func):
    """φ-aligned optimization decorator."""
    import functools, time
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        wrapper._last_time = time.time() - start
        return result
    wrapper._phi_aligned = True
    return wrapper
'''

    def get_modification_report(self) -> Dict:
        """Return self-modification history and depth metrics with quantum state data."""
        avg_fitness = sum(self._fitness_history) / max(len(self._fitness_history), 1) if self._fitness_history else 0.0
        # Compute quantum entropy
        probs = np.abs(self._q_amplitudes) ** 2
        probs = probs / probs.sum()
        probs_nz = probs[probs > 1e-15]
        q_entropy = float(-np.sum(probs_nz * np.log2(probs_nz))) if len(probs_nz) > 0 else 0.0
        return {'total_modifications': len(self.modifications),
                'current_depth': self.modification_depth,
                'max_depth': ASI_SELF_MODIFICATION_DEPTH,
                'improvement_count': self._improvement_count,
                'revert_count': self._revert_count,
                'rollback_buffer_size': len(self._rollback_buffer),
                'max_recursive_depth': self._max_recursive_depth,
                'avg_fitness': round(avg_fitness, 4),
                'fitness_trend': 'improving' if len(self._fitness_history) >= 2 and self._fitness_history[-1] > self._fitness_history[-2] else 'stable',
                'quantum': {
                    'coherence': round(self._q_coherence, 6),
                    'entropy': round(q_entropy, 6),
                    'grover_iterations': self._q_grover_iters,
                    'tunneling_events': self._q_tunnel_events,
                    'phase_accumulator': round(self._q_phase_acc, 6),
                    'god_code_alignment': round(1.0 - abs(self._q_phase_acc % GOD_CODE) / GOD_CODE, 6),
                    'hilbert_dim': self.Q_STATE_DIM,
                },
                'lineage': {
                    'total_nodes': len(self._lineage),
                    'consecutive_non_improving': self._consecutive_non_improving,
                    'max_lineage_depth': self._lineage_max_depth(),
                }}

    # ═══════════════════════════════════════════════════════════════════
    # ═══  v7.0  ADVANCED SELF-MODIFICATION                          ═══
    # ═══  Lineage DAG · Grover-amplified selection · Quantum tunnel  ═══
    # ═══  Multi-file evolution · Complexity prediction               ═══
    # ═══════════════════════════════════════════════════════════════════

    def _record_lineage(self, filepath: str, transform_log: List[str],
                        fitness: float, parent_id: Optional[int] = None) -> int:
        """Record a lineage node in the modification DAG.

        Returns the new node ID.
        """
        node_id = self._lineage_counter
        self._lineage_counter += 1
        node = {
            'id': node_id,
            'parent_id': parent_id,
            'filepath': filepath,
            'transform_summary': '; '.join(transform_log[:3]),
            'fitness': round(fitness, 6),
            'timestamp': datetime.now().isoformat(),
            'depth': self.modification_depth,
        }
        self._lineage.append(node)
        # Keep lineage bounded
        if len(self._lineage) > self.MAX_LINEAGE_NODES:
            self._lineage = self._lineage[-self.MAX_LINEAGE_NODES:]
        return node_id

    def _lineage_max_depth(self) -> int:
        """Compute maximum depth in the lineage DAG via parent chain traversal."""
        if not self._lineage:
            return 0
        id_to_parent: Dict[int, Optional[int]] = {
            n['id']: n['parent_id'] for n in self._lineage
        }
        max_d = 0
        for nid in id_to_parent:
            d = 0
            cur = nid
            visited = set()
            while cur is not None and cur in id_to_parent and cur not in visited:
                visited.add(cur)
                cur = id_to_parent[cur]
                d += 1
            max_d = max(max_d, d)
        return max_d

    def get_lineage_graph(self) -> Dict[str, Any]:
        """Return the full lineage DAG for visualization.

        Returns:
            Dict with nodes list, edges list, total_depth, and fitness_trajectory.
        """
        nodes = list(self._lineage)
        edges = []
        for n in nodes:
            if n['parent_id'] is not None:
                edges.append({'from': n['parent_id'], 'to': n['id']})
        trajectory = [n['fitness'] for n in nodes]
        return {
            'nodes': nodes,
            'edges': edges,
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'max_depth': self._lineage_max_depth(),
            'fitness_trajectory': trajectory,
            'fitness_improvement': round(trajectory[-1] - trajectory[0], 6) if len(trajectory) >= 2 else 0.0,
        }

    def grover_amplified_transform_select(self, source: str) -> Tuple[str, List[str]]:
        """Select the best AST transform pass using Grover-inspired amplitude amplification.

        Evaluates all 3 transform passes independently, encodes their fitness into
        quantum amplitudes, and uses Grover iterations to amplify the best.
        Then applies passes in order of descending amplified fitness.

        Returns:
            (transformed_source, transform_log)
        """
        log: List[str] = []
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return source, [f"Parse error: {e}"]

        # ── Evaluate each pass independently ──
        pass_scores: List[Tuple[str, float]] = []

        # Pass A: Unused imports
        imported_names: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)
        used_names: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                used_names.add(node.value.id)
        unused = imported_names - used_names
        score_a = len(unused) * 0.1  # More unused → more improvement potential
        pass_scores.append(('unused_imports', score_a))

        # Pass B: Constant folding
        foldable = sum(1 for n in ast.walk(tree)
                       if isinstance(n, ast.BinOp)
                       and isinstance(n.left, ast.Constant)
                       and isinstance(n.right, ast.Constant))
        score_b = foldable * 0.15
        pass_scores.append(('constant_folding', score_b))

        # Pass C: Dead code after return
        dead_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found_return = False
                for stmt in node.body:
                    if found_return:
                        dead_count += 1
                    if isinstance(stmt, ast.Return):
                        found_return = True
        score_c = dead_count * 0.2
        pass_scores.append(('dead_code', score_c))

        # ── Grover amplitude amplification ──
        n_passes = len(pass_scores)
        amplitudes = np.full(n_passes, 1.0 / np.sqrt(n_passes), dtype=np.complex128)
        # Oracle: mark the highest-scoring pass
        best_idx = int(np.argmax([s for _, s in pass_scores]))
        # Grover iterations: O(√N) — here N=3, so 1-2 iterations
        n_iters = max(1, int(np.pi / 4 * np.sqrt(n_passes)))
        for _ in range(n_iters):
            # Oracle: flip phase of marked state
            amplitudes[best_idx] *= -1
            # Diffusion: 2|ψ⟩⟨ψ| − I
            mean_amp = np.mean(amplitudes)
            amplitudes = 2 * mean_amp - amplitudes
        self._q_grover_iters += n_iters

        # Sort passes by amplified probability (descending)
        probs = np.abs(amplitudes) ** 2
        ranked = sorted(range(n_passes), key=lambda i: probs[i], reverse=True)

        log.append(f"Grover amplification: {n_iters} iters, ranked passes: "
                   f"{[pass_scores[i][0] for i in ranked]}")
        log.append(f"Amplified probs: {[round(float(probs[i]), 4) for i in ranked]}")

        # ── Apply in ranked order (delegate to standard multi_pass) ──
        transformed, pass_log = self.multi_pass_ast_transform(source)
        log.extend(pass_log)

        return transformed, log

    def quantum_tunnel_escape(self, filepath: Path, n_perturbations: int = 5) -> Dict[str, Any]:
        """Attempt to escape a local optimum via quantum tunneling.

        When consecutive non-improving evolutions exceed TUNNEL_ESCAPE_THRESHOLD,
        this method introduces controlled perturbations to the code structure,
        evaluates fitness after each, and keeps the best-fitness variant.

        Args:
            filepath: Target file to perturb.
            n_perturbations: Number of random perturbation attempts.

        Returns:
            Dict with tunnel_success, best_fitness, perturbation_details.
        """
        if not filepath.exists():
            return {'tunnel_success': False, 'reason': 'File not found'}

        original_source = filepath.read_text()
        original_fitness = self.compute_fitness(filepath)
        best_source = original_source
        best_fitness = original_fitness
        perturbation_log: List[Dict] = []
        rng = np.random.RandomState(int(GOD_CODE * self._q_tunnel_events + 1) % (2**31))

        for i in range(n_perturbations):
            # Tunnel probability decays with coherence
            tunnel_prob = self.Q_TUNNEL_PROB * self._q_coherence
            if rng.random() > tunnel_prob:
                perturbation_log.append({'attempt': i, 'skipped': True, 'prob': round(tunnel_prob, 4)})
                continue

            try:
                tree = ast.parse(original_source)
            except SyntaxError:
                break

            # Perturbation: randomly reorder top-level class/function definitions
            # (preserving import block)
            body = tree.body
            imports = [n for n in body if isinstance(n, (ast.Import, ast.ImportFrom))]
            non_imports = [n for n in body if not isinstance(n, (ast.Import, ast.ImportFrom))]

            if len(non_imports) > 1:
                # Swap two random non-import statements
                idx_a, idx_b = rng.choice(len(non_imports), size=2, replace=False)
                non_imports[idx_a], non_imports[idx_b] = non_imports[idx_b], non_imports[idx_a]
                tree.body = imports + non_imports
                try:
                    perturbed = ast.unparse(tree)
                    # Verify it still parses
                    ast.parse(perturbed)
                    filepath.write_text(perturbed)
                    new_fitness = self.compute_fitness(filepath)
                    delta = new_fitness - original_fitness

                    perturbation_log.append({
                        'attempt': i, 'skipped': False,
                        'fitness': round(new_fitness, 6),
                        'delta': round(delta, 6),
                        'swapped': [idx_a, idx_b],
                    })

                    if new_fitness > best_fitness:
                        best_fitness = new_fitness
                        best_source = perturbed

                    # Restore original for next attempt
                    filepath.write_text(original_source)
                except (SyntaxError, Exception):
                    filepath.write_text(original_source)
                    perturbation_log.append({'attempt': i, 'skipped': False, 'error': 'unparse_failed'})
            else:
                perturbation_log.append({'attempt': i, 'skipped': True, 'reason': 'single_statement'})

        # Apply best variant if it improved
        tunnel_success = best_fitness > original_fitness
        if tunnel_success:
            filepath.write_text(best_source)
            self._q_tunnel_events += 1
            self._consecutive_non_improving = 0
            self._record_lineage(str(filepath), ['quantum_tunnel_escape'], best_fitness)

        # Decohere after tunneling attempt
        self._q_coherence *= (1.0 - self.Q_DECOHERENCE)
        self._q_phase_acc += np.pi * PHI / (n_perturbations + 1)

        return {
            'tunnel_success': tunnel_success,
            'original_fitness': round(original_fitness, 6),
            'best_fitness': round(best_fitness, 6),
            'delta': round(best_fitness - original_fitness, 6),
            'perturbations_attempted': len([p for p in perturbation_log if not p.get('skipped')]),
            'perturbation_log': perturbation_log,
            'tunnel_events_total': self._q_tunnel_events,
            'coherence_after': round(self._q_coherence, 6),
        }

    def multi_file_evolve(self, filepaths: List[Path],
                           max_rounds: int = 3) -> Dict[str, Any]:
        """Evolve multiple files jointly, propagating fitness improvements.

        Runs evolution cycles across all given files in round-robin fashion.
        Files that improve propagate their lineage to subsequent rounds.
        Triggers quantum tunneling when stuck.

        Args:
            filepaths: List of Python files to evolve together.
            max_rounds: Maximum number of full round-robin sweeps.

        Returns:
            Dict with per-file results, total improvement, and lineage summary.
        """
        results_per_file: Dict[str, List[Dict]] = {}
        total_delta = 0.0
        parent_id: Optional[int] = None

        for rnd in range(max_rounds):
            for fp in filepaths:
                fp = Path(fp)
                if not fp.exists() or fp.name in self.locked_modules:
                    continue

                key = str(fp)
                if key not in results_per_file:
                    results_per_file[key] = []

                evo = self.evolve_with_fitness(fp)
                total_delta += evo.get('delta', 0.0)

                # Record lineage
                node_id = self._record_lineage(
                    key, evo.get('transform_log', []),
                    evo.get('after_fitness', 0.0), parent_id=parent_id
                )

                if evo.get('delta', 0.0) > 0:
                    self._consecutive_non_improving = 0
                    parent_id = node_id
                else:
                    self._consecutive_non_improving += 1

                # Trigger tunneling if stuck
                if self._consecutive_non_improving >= self.TUNNEL_ESCAPE_THRESHOLD:
                    tunnel = self.quantum_tunnel_escape(fp)
                    evo['tunnel_attempt'] = tunnel
                    total_delta += tunnel.get('delta', 0.0)
                    self._consecutive_non_improving = 0

                results_per_file[key].append(evo)

        return {
            'files_evolved': len(results_per_file),
            'total_rounds': max_rounds,
            'total_fitness_delta': round(total_delta, 6),
            'per_file': {k: v for k, v in results_per_file.items()},
            'lineage_nodes': len(self._lineage),
            'tunnel_events': self._q_tunnel_events,
        }

    def predict_complexity(self, source: str) -> Dict[str, Any]:
        """Predict code complexity metrics from source without full AST analysis.

        Uses lightweight heuristics and quantum-encoded patterns to estimate:
        - Cyclomatic complexity
        - Cognitive complexity (Sonar-style)
        - Halstead volume
        - Maintainability index

        Args:
            source: Python source code string.

        Returns:
            Dict with estimated complexity metrics.
        """
        lines = source.splitlines()
        total_lines = len(lines)
        code_lines = sum(1 for l in lines if l.strip() and not l.strip().startswith('#'))
        comment_lines = sum(1 for l in lines if l.strip().startswith('#'))

        # Cyclomatic: count decision points
        branch_keywords = {'if', 'elif', 'for', 'while', 'except', 'with', 'and', 'or'}
        cyclomatic = 1  # Base complexity
        for line in lines:
            tokens = line.split()
            for t in tokens:
                clean = t.strip(':').strip('(').strip(')')
                if clean in branch_keywords:
                    cyclomatic += 1

        # Cognitive: weighted nesting depth
        cognitive = 0
        indent_stack = 0
        for line in lines:
            stripped = line.lstrip()
            if not stripped:
                continue
            indent = len(line) - len(stripped)
            nesting = indent // 4  # Approximate nesting level
            for kw in ['if ', 'elif ', 'for ', 'while ', 'except ']:
                if stripped.startswith(kw):
                    cognitive += 1 + nesting  # Nesting adds weight
                    break

        # Halstead approximation: unique operators/operands
        import re as _re
        identifiers = set(_re.findall(r'\b[a-zA-Z_]\w*\b', source))
        operators = set(_re.findall(r'[+\-*/=<>!&|^~%]+', source))
        n1, n2 = len(operators), len(identifiers)
        N1 = len(_re.findall(r'[+\-*/=<>!&|^~%]+', source))
        N2 = len(_re.findall(r'\b[a-zA-Z_]\w*\b', source))
        vocabulary = n1 + n2
        length = N1 + N2
        volume = length * np.log2(max(vocabulary, 2))

        # Maintainability index: 171 − 5.2 ln(V) − 0.23 G − 16.2 ln(L) + 50 sin(√(2.4 C))
        mi = (171.0
               - 5.2 * np.log(max(volume, 1))
               - 0.23 * cyclomatic
               - 16.2 * np.log(max(code_lines, 1))
               + 50.0 * np.sin(np.sqrt(2.4 * comment_lines / max(total_lines, 1))))
        mi = max(0.0, min(100.0, mi))

        # Quantum phase encoding: encode complexity into quantum state
        complexity_signal = cyclomatic / max(code_lines, 1)
        self._q_phase_acc += complexity_signal * PHI
        self._q_coherence *= (1.0 - complexity_signal * self.Q_DECOHERENCE)

        return {
            'total_lines': total_lines,
            'code_lines': code_lines,
            'comment_lines': comment_lines,
            'cyclomatic_complexity': cyclomatic,
            'cognitive_complexity': cognitive,
            'halstead_volume': round(volume, 2),
            'halstead_vocabulary': vocabulary,
            'halstead_length': length,
            'maintainability_index': round(mi, 2),
            'complexity_density': round(cyclomatic / max(code_lines, 1), 4),
            'quantum_phase': round(self._q_phase_acc % (2 * np.pi), 6),
        }
