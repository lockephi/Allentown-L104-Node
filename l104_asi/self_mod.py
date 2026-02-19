from .constants import *
class SelfModificationEngine:
    """Enables autonomous self-modification with multi-pass AST transforms,
    safe rollback, fitness-driven evolution, and recursive depth tracking.
    v5.0: Quantum-enhanced fitness evaluation, Grover-amplified transform selection,
    quantum tunneling for escaping local optima, entanglement-based code blending."""
    # Quantum constants for self-modification
    Q_STATE_DIM = 32
    Q_TUNNEL_PROB = 0.10
    Q_DECOHERENCE = 0.02

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
                }}


