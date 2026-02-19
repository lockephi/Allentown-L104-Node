from .constants import *
class SolutionChannel:
    """Direct channel to solutions."""
    def __init__(self, name: str, domain: str):
        self.name = name
        self.domain = domain
        self.solvers: List[Callable] = []
        self.cache: Dict[str, Any] = {}
        self.latency_ms = 0.0
        self.invocations = 0
        self.success_rate = 0.0

    def add_solver(self, solver: Callable):
        self.solvers.append(solver)

    def solve(self, problem: Dict) -> Dict:
        start = time.time()
        self.invocations += 1
        h = hashlib.sha256(str(problem).encode()).hexdigest()
        if h in self.cache:
            self.latency_ms = (time.time() - start) * 1000
            return {'solution': self.cache[h], 'cached': True}
        for solver in self.solvers:
            try:
                sol = solver(problem)
                if sol is not None:
                    self.cache[h] = sol
                    self.latency_ms = (time.time() - start) * 1000
                    self.success_rate = (self.success_rate * (self.invocations-1) + 1) / self.invocations
                    return {'solution': sol, 'cached': False}
            except Exception:
                continue
        self.latency_ms = (time.time() - start) * 1000
        return {'solution': None, 'error': 'No solver succeeded'}


class DirectSolutionHub:
    """Hub for direct solution channels."""
    def __init__(self):
        self.channels: Dict[str, SolutionChannel] = {}
        self._init_channels()

    def _init_channels(self):
        # Math channel
        math = SolutionChannel('mathematics', 'mathematics')
        math.add_solver(self._solve_arithmetic)
        math.add_solver(self._solve_sacred)
        self.channels['mathematics'] = math

        # Knowledge channel
        knowledge = SolutionChannel('knowledge', 'knowledge')
        knowledge.add_solver(self._solve_knowledge)
        self.channels['knowledge'] = knowledge

        # Code channel
        code = SolutionChannel('code', 'computer_science')
        code.add_solver(self._solve_code)
        self.channels['code'] = code

    def _solve_arithmetic(self, p: Dict) -> Any:
        expr = p.get('expression', '')
        if expr and all(c in '0123456789+-*/.() ' for c in expr):
            try:
                return eval(expr, {"__builtins__": {}}, {})
            except Exception:
                pass
        return None

    def _solve_sacred(self, p: Dict) -> Any:
        q = p.get('query', '').lower()
        answers = {'god_code': GOD_CODE, 'phi': PHI, 'tau': TAU, 'golden': PHI,
                   'void': VOID_CONSTANT, 'omega': OMEGA_AUTHORITY}
        for k, v in answers.items():
            if k in q:
                return v
        return None

    def _solve_knowledge(self, p: Dict) -> Any:
        q = p.get('query', '').lower()
        kb = {
            'l104': f'Sovereign intelligence kernel with GOD_CODE={GOD_CODE}',
            'consciousness': 'Emergent property of complex information processing',
            'fibonacci': f'Sequence converging to PHI={PHI}'
        }
        for k, v in kb.items():
            if k in q:
                return v
        return None

    def _solve_code(self, p: Dict) -> Any:
        task = p.get('task', '').lower()
        if 'fibonacci' in task:
            return 'def fib(n): return n if n<=1 else fib(n-1)+fib(n-2)'
        if 'phi' in task:
            return f'PHI = {PHI}'
        return None

    def route_problem(self, p: Dict) -> str:
        q = str(p).lower()
        if any(x in q for x in ['god_code', 'phi', 'tau', 'calculate', '+', '-', '*']):
            return 'mathematics'
        if any(x in q for x in ['code', 'function', 'program']):
            return 'code'
        return 'knowledge'

    def solve(self, problem: Dict) -> Dict:
        channel_name = self.route_problem(problem)
        channel = self.channels.get(channel_name)
        if not channel:
            return {'error': 'No channel'}
        result = channel.solve(problem)
        result['channel'] = channel_name
        result['latency_ms'] = channel.latency_ms
        return result

    def get_channel_stats(self) -> Dict:
        return {n: {'invocations': c.invocations, 'success_rate': c.success_rate}
                for n, c in self.channels.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# v5.0 SOVEREIGN INTELLIGENCE PIPELINE ENGINES
# ═══════════════════════════════════════════════════════════════════════════════


class PipelineTelemetry:
    """Per-subsystem latency, success rate, and throughput tracking with EMA smoothing.

    v5.0: Tracks every subsystem invocation, computes exponential moving averages
    for latency, maintains per-subsystem success/failure counts, detects anomalies.
    """
    def __init__(self, ema_alpha: float = TELEMETRY_EMA_ALPHA):
        self.ema_alpha = ema_alpha
        self._subsystem_stats: Dict[str, Dict[str, Any]] = {}
        self._global_ops = 0
        self._global_errors = 0
        self._start_time = time.time()

    def record(self, subsystem: str, latency_ms: float, success: bool, metadata: Optional[Dict] = None):
        """Record a subsystem invocation with latency and success/failure."""
        if subsystem not in self._subsystem_stats:
            self._subsystem_stats[subsystem] = {
                'invocations': 0, 'successes': 0, 'failures': 0,
                'ema_latency_ms': latency_ms, 'peak_latency_ms': latency_ms,
                'total_latency_ms': 0.0, 'last_invocation': None,
                'error_streak': 0, 'best_latency_ms': latency_ms,
            }

        stats = self._subsystem_stats[subsystem]
        stats['invocations'] += 1
        stats['total_latency_ms'] += latency_ms
        stats['last_invocation'] = time.time()

        # EMA latency
        stats['ema_latency_ms'] = (
            self.ema_alpha * latency_ms + (1 - self.ema_alpha) * stats['ema_latency_ms']
        )
        stats['peak_latency_ms'] = max(stats['peak_latency_ms'], latency_ms)
        stats['best_latency_ms'] = min(stats['best_latency_ms'], latency_ms)

        if success:
            stats['successes'] += 1
            stats['error_streak'] = 0
        else:
            stats['failures'] += 1
            stats['error_streak'] += 1
            self._global_errors += 1

        self._global_ops += 1

    def get_subsystem_stats(self, subsystem: str) -> Dict:
        """Get statistics for a single subsystem."""
        stats = self._subsystem_stats.get(subsystem)
        if not stats:
            return {'subsystem': subsystem, 'status': 'NO_DATA'}
        invocations = stats['invocations']
        return {
            'subsystem': subsystem,
            'invocations': invocations,
            'success_rate': round(stats['successes'] / max(invocations, 1), 4),
            'ema_latency_ms': round(stats['ema_latency_ms'], 3),
            'avg_latency_ms': round(stats['total_latency_ms'] / max(invocations, 1), 3),
            'peak_latency_ms': round(stats['peak_latency_ms'], 3),
            'best_latency_ms': round(stats['best_latency_ms'], 3),
            'error_streak': stats['error_streak'],
            'health': 'CRITICAL' if stats['error_streak'] >= 5 else
                      'DEGRADED' if stats['error_streak'] >= 2 else 'HEALTHY',
        }

    def get_dashboard(self) -> Dict:
        """Full telemetry dashboard across all subsystems."""
        uptime = time.time() - self._start_time
        subsystem_reports = {
            name: self.get_subsystem_stats(name)
            for name in self._subsystem_stats
        }
        healthy = sum(1 for r in subsystem_reports.values() if r.get('health') == 'HEALTHY')
        degraded = sum(1 for r in subsystem_reports.values() if r.get('health') == 'DEGRADED')
        critical = sum(1 for r in subsystem_reports.values() if r.get('health') == 'CRITICAL')
        total = len(subsystem_reports)
        return {
            'global_ops': self._global_ops,
            'global_errors': self._global_errors,
            'global_success_rate': round(1.0 - self._global_errors / max(self._global_ops, 1), 4),
            'uptime_s': round(uptime, 2),
            'throughput_ops_per_s': round(self._global_ops / max(uptime, 0.001), 2),
            'subsystems_tracked': total,
            'healthy': healthy, 'degraded': degraded, 'critical': critical,
            'pipeline_health': round(healthy / max(total, 1), 4),
            'subsystems': subsystem_reports,
        }

    def detect_anomalies(self, sigma_threshold: float = HEALTH_ANOMALY_SIGMA) -> List[Dict]:
        """Detect subsystems with anomalous latency (> sigma_threshold standard deviations)."""
        if len(self._subsystem_stats) < 2:
            return []
        latencies = [s['ema_latency_ms'] for s in self._subsystem_stats.values()]
        mean_lat = sum(latencies) / len(latencies)
        variance = sum((l - mean_lat) ** 2 for l in latencies) / len(latencies)
        std_dev = math.sqrt(variance) if variance > 0 else 1.0
        anomalies = []
        for name, stats in self._subsystem_stats.items():
            z_score = (stats['ema_latency_ms'] - mean_lat) / max(std_dev, 1e-6)
            if abs(z_score) > sigma_threshold:
                anomalies.append({
                    'subsystem': name, 'z_score': round(z_score, 3),
                    'ema_latency_ms': round(stats['ema_latency_ms'], 3),
                    'type': 'SLOW' if z_score > 0 else 'UNUSUALLY_FAST',
                })
        return anomalies


class SoftmaxGatingRouter:
    """Mixture of Experts (MoE) gating network — DeepSeek-V3 style (Dec 2024).

    Routes queries to subsystems using learned softmax gating with top-K selection.
    g(x) = Softmax(W_gate × embed(query)), selects top-K experts.

    Key innovations from DeepSeek-V3 (256 experts, 671B params):
    - Auxiliary-loss-free load balancing via per-expert bias
    - Shared expert always active (here: 'direct_solution' always included)
    - Bias adjusted outside gradient to avoid distorting training objective

    Sacred: GOD_CODE-seeded weights, PHI-weighted balance coefficient,
    embed_dim = 64, top_k = int(PHI * 2) = 3.
    """

    def __init__(self, num_experts: int = 16, embed_dim: int = 64, top_k: int = None):
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.top_k = top_k or max(1, int(PHI * 2))  # 3
        rng = random.Random(int(GOD_CODE * 1000 + 314))
        bound = 1.0 / math.sqrt(embed_dim)
        self.W_gate = [[rng.uniform(-bound, bound) for _ in range(embed_dim)]
                       for _ in range(num_experts)]
        # DeepSeek-V3 load balancing bias (adjusted outside gradient)
        self.expert_bias = [0.0] * num_experts
        self.expert_load: Dict[int, int] = {i: 0 for i in range(num_experts)}
        self.expert_names: Dict[int, str] = {}
        self.name_to_id: Dict[str, int] = {}
        self.balance_gamma = TAU / 100.0  # bias step size ~0.00618
        self.route_count = 0

    def register_expert(self, expert_id: int, name: str):
        self.expert_names[expert_id] = name
        self.name_to_id[name] = expert_id

    def _embed_query(self, query: str) -> List[float]:
        """Character n-gram embedding to embed_dim."""
        vec = [0.0] * self.embed_dim
        q = query.lower()
        for i in range(len(q)):
            for n in (2, 3, 4):
                if i + n <= len(q):
                    gram = q[i:i + n]
                    idx = hash(gram) % self.embed_dim
                    vec[idx] += 1.0
        mag = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / mag for v in vec]

    def gate(self, query: str) -> List[Tuple[str, float]]:
        """Compute MoE gating scores, return top-K (expert_name, weight) pairs."""
        self.route_count += 1
        x = self._embed_query(query)
        total_load = sum(self.expert_load.values()) + 1

        # Logits = W_gate @ x + load-balancing bias
        logits = []
        for i in range(self.num_experts):
            score = sum(self.W_gate[i][j] * x[j] for j in range(self.embed_dim))
            # DeepSeek-V3: bias penalizes overloaded experts
            load_frac = self.expert_load.get(i, 0) / total_load
            logits.append(score + self.expert_bias[i] - load_frac * TAU)

        # Softmax
        max_l = max(logits) if logits else 0
        exp_l = [math.exp(min(l - max_l, 20)) for l in logits]
        total = sum(exp_l) + 1e-10
        probs = [e / total for e in exp_l]

        # Top-K selection
        indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        top_k = indexed[:self.top_k]
        sel_total = sum(w for _, w in top_k) + 1e-10
        result = []
        for idx, w in top_k:
            name = self.expert_names.get(idx, f"expert_{idx}")
            result.append((name, w / sel_total))
            self.expert_load[idx] = self.expert_load.get(idx, 0) + 1

        # Periodically update bias for load balancing (DeepSeek-V3 style)
        if self.route_count % 20 == 0:
            self._update_balance_bias()

        return result

    def _update_balance_bias(self):
        """DeepSeek-V3: adjust bias to balance load without affecting training gradient."""
        if not self.expert_load:
            return
        total = sum(self.expert_load.values()) + 1
        target = total / max(self.num_experts, 1)
        for i in range(self.num_experts):
            load = self.expert_load.get(i, 0)
            if load > target * 1.2:
                self.expert_bias[i] -= self.balance_gamma
            elif load < target * 0.8:
                self.expert_bias[i] += self.balance_gamma

    def feedback(self, expert_name: str, success: bool):
        """Reinforce or weaken expert gate weights based on outcome."""
        eid = self.name_to_id.get(expert_name)
        if eid is None:
            return
        lr = ALPHA_FINE * PHI  # ~0.0118
        delta = lr if success else -lr * TAU
        for j in range(self.embed_dim):
            self.W_gate[eid][j] += delta * random.gauss(0, 0.01)

    def get_status(self) -> Dict:
        return {
            'type': 'SoftmaxGatingRouter_MoE_DeepSeekV3',
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'routes_computed': self.route_count,
            'expert_load': dict(sorted(self.expert_load.items(), key=lambda x: x[1], reverse=True)[:5]),
        }


class AdaptivePipelineRouter:
    """TF-IDF subsystem routing with MoE gating (DeepSeek-V3) + reinforcement learning.

    v6.0: Routes queries to subsystems using TF-IDF keyword scoring. Term frequency
    is counted per-query, inverse document frequency penalizes keywords that appear
    across many subsystems. Affinity weights are updated via success/failure feedback
    with PHI-weighted learning rate and TAU-decay for stale associations. Tracks
    per-keyword success rates to inform future routing decisions.
    """
    def __init__(self):
        self._subsystem_keywords: Dict[str, List[str]] = {
            'computronium': ['density', 'compute', 'entropy', 'dimension', 'cascade', 'compress', 'lattice'],
            'manifold_resolver': ['space', 'dimension', 'topology', 'manifold', 'geometry', 'embed'],
            'shadow_gate': ['adversarial', 'stress', 'test', 'attack', 'robust', 'vulnerability'],
            'non_dual_logic': ['paradox', 'contradiction', 'truth', 'logic', 'both', 'neither'],
            'recursive_inventor': ['invent', 'create', 'novel', 'idea', 'innovate', 'design'],
            'transcendent_solver': ['transcend', 'meta', 'consciousness', 'awareness', 'wisdom'],
            'almighty_asi': ['omniscient', 'pattern', 'universal', 'absolute', 'complete'],
            'hyper_asi': ['hyper', 'unified', 'activation', 'combine', 'integrate'],
            'processing_engine': ['process', 'analyze', 'ensemble', 'multi', 'cognitive'],
            'erasi_engine': ['entropy', 'reversal', 'erasi', 'thermodynamic', 'order'],
            'sage_core': ['sage', 'wisdom', 'philosophy', 'deep', 'meaning'],
            'asi_nexus': ['swarm', 'multi-agent', 'coordinate', 'nexus', 'collective'],
            'asi_research': ['research', 'investigate', 'study', 'explore', 'discover'],
            'asi_language': ['language', 'linguistic', 'grammar', 'semantic', 'speech'],
            'asi_harness': ['code', 'analyze', 'optimize', 'refactor', 'engineering'],
            'direct_solution': ['solve', 'calculate', 'compute', 'answer', 'math', 'phi', 'god_code'],
        }
        # Affinity weights: keyword → subsystem → weight (learned over time)
        self._affinity_matrix: Dict[str, Dict[str, float]] = {}
        for subsystem, keywords in self._subsystem_keywords.items():
            self._affinity_matrix[subsystem] = {kw: 1.0 for kw in keywords}
        # Compute IDF: log(N_subsystems / count_of_subsystems_containing_keyword)
        self._idf: Dict[str, float] = self._compute_idf()
        # Per-keyword success tracking for reinforcement
        self._keyword_stats: Dict[str, Dict[str, int]] = {}  # kw → {successes, attempts}
        self._route_count = 0
        self._feedback_count = 0
        self._learning_rate = PHI / 10.0  # ≈0.1618
        # MoE Gating Router (DeepSeek-V3 style) — learned softmax routing
        self._moe_router = SoftmaxGatingRouter(
            num_experts=len(self._subsystem_keywords),
            embed_dim=64, top_k=3
        )
        for i, name in enumerate(self._subsystem_keywords.keys()):
            self._moe_router.register_expert(i, name)
        self._moe_warmup = 50  # Use TF-IDF for first N routes, then switch to MoE

    def _compute_idf(self) -> Dict[str, float]:
        """Compute inverse document frequency for each keyword across subsystems."""
        n_subsystems = len(self._subsystem_keywords)
        keyword_doc_count: Dict[str, int] = {}
        for keywords in self._subsystem_keywords.values():
            for kw in keywords:
                keyword_doc_count[kw] = keyword_doc_count.get(kw, 0) + 1
        return {
            kw: math.log((n_subsystems + 1) / (count + 1)) + 1.0
            for kw, count in keyword_doc_count.items()
        }

    def _tokenize(self, text: str) -> Dict[str, int]:
        """Tokenize query into word-frequency map (term frequency)."""
        tokens = re.findall(r'[a-z_]+', text.lower())
        tf: Dict[str, int] = {}
        for token in tokens:
            if len(token) > 2:
                tf[token] = tf.get(token, 0) + 1
        return tf

    def route(self, query: str) -> List[Tuple[str, float]]:
        """Route a query using MoE gating (DeepSeek-V3) with TF-IDF fallback."""
        # After warmup, prefer MoE learned routing over TF-IDF keyword matching
        if self._route_count >= self._moe_warmup:
            moe_result = self._moe_router.gate(query)
            if moe_result and moe_result[0][1] > 0.1:
                self._route_count += 1
                return moe_result

        # TF-IDF fallback (or during warmup phase)
        query_tf = self._tokenize(query)
        scores: Dict[str, float] = {}

        for subsystem, affinities in self._affinity_matrix.items():
            score = 0.0
            for keyword, affinity_weight in affinities.items():
                # Check both exact token match and substring containment
                tf = query_tf.get(keyword, 0)
                if tf == 0 and keyword in query.lower():
                    tf = 1  # substring match gets base TF of 1
                if tf > 0:
                    idf = self._idf.get(keyword, 1.0)
                    # TF-IDF × learned affinity × keyword success rate
                    kw_stats = self._keyword_stats.get(keyword)
                    success_boost = 1.0
                    if kw_stats and kw_stats.get('attempts', 0) >= 3:
                        success_boost = 0.5 + (kw_stats['successes'] / kw_stats['attempts'])
                    score += (1 + math.log(1 + tf)) * idf * affinity_weight * success_boost
            scores[subsystem] = score

        self._route_count += 1
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(name, round(score, 4)) for name, score in ranked if score > 0]

    def feedback(self, subsystem: str, keywords: List[str], success: bool, confidence: float = 0.8):
        """Update affinity matrix and keyword stats from solution outcome."""
        if subsystem not in self._affinity_matrix:
            self._affinity_matrix[subsystem] = {}

        # Reinforcement learning update: scale by confidence and learning rate
        lr = self._learning_rate * confidence
        delta = lr if success else -lr * TAU  # Asymmetric: penalize less than reward

        for kw in keywords:
            # Update affinity weight
            current = self._affinity_matrix[subsystem].get(kw, 0.5)
            self._affinity_matrix[subsystem][kw] = max(0.01, min(5.0, current + delta))
            # Track keyword success rate
            if kw not in self._keyword_stats:
                self._keyword_stats[kw] = {'successes': 0, 'attempts': 0}
            self._keyword_stats[kw]['attempts'] += 1
            if success:
                self._keyword_stats[kw]['successes'] += 1

        # PHI-decay: slightly reduce all affinities for this subsystem to prevent overfitting
        decay = 1.0 - (TAU / 100.0)  # ~0.9938 per feedback cycle
        for kw in self._affinity_matrix[subsystem]:
            if kw not in keywords:
                self._affinity_matrix[subsystem][kw] *= decay

        self._feedback_count += 1
        # Propagate feedback to MoE router
        self._moe_router.feedback(subsystem, success)
        # Recompute IDF periodically as new keywords may be added
        if self._feedback_count % 50 == 0:
            self._idf = self._compute_idf()

    def get_status(self) -> Dict:
        top_keywords = sorted(
            self._keyword_stats.items(),
            key=lambda x: x[1].get('successes', 0), reverse=True
        )[:10] if self._keyword_stats else []
        return {
            'subsystems_tracked': len(self._affinity_matrix),
            'total_keywords': sum(len(v) for v in self._affinity_matrix.values()),
            'routes_computed': self._route_count,
            'feedback_updates': self._feedback_count,
            'tracked_keywords': len(self._keyword_stats),
            'top_keywords': [(kw, s['successes'], s['attempts']) for kw, s in top_keywords],
        }


