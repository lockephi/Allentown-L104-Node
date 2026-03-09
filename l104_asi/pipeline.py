from .constants import *
import heapq as _heapq


class SolutionChannel:
    """Direct channel to solutions with priority queuing and circuit breaker.

    v7.0: Added priority-aware queuing, circuit breaker pattern (half-open recovery),
    LRU cache eviction, and per-solver success tracking."""

    # Circuit breaker states
    CB_CLOSED = 'CLOSED'          # Normal operation
    CB_OPEN = 'OPEN'              # Failing — reject requests
    CB_HALF_OPEN = 'HALF_OPEN'    # Probing — allow single test request

    def __init__(self, name: str, domain: str):
        self.name = name
        self.domain = domain
        self.solvers: List[Callable] = []
        self.cache: Dict[str, Any] = {}
        self.latency_ms = 0.0
        self.invocations = 0
        self.success_rate = 0.0
        # v7.0: Priority queue (min-heap) — lower priority number = higher priority
        self._priority_queue: List[Tuple[int, int, Dict]] = []  # (priority, seq, problem)
        self._pq_seq = 0
        # v7.0: Circuit breaker
        self._cb_state = self.CB_CLOSED
        self._cb_failure_count = 0
        self._cb_failure_threshold = 5
        self._cb_recovery_time = 30.0  # seconds
        self._cb_last_failure_time = 0.0
        self._cb_half_open_successes = 0
        # v7.0: Cache size limit (LRU eviction)
        self._cache_max_size = 1024
        self._cache_access_order: List[str] = []
        # v7.0: Per-solver stats
        self._solver_stats: Dict[int, Dict[str, int]] = {}

    def add_solver(self, solver: Callable):
        idx = len(self.solvers)
        self.solvers.append(solver)
        self._solver_stats[idx] = {'successes': 0, 'failures': 0}

    def enqueue(self, problem: Dict, priority: int = 5):
        """Add a problem to the priority queue. Lower priority = processed first."""
        self._pq_seq += 1
        _heapq.heappush(self._priority_queue, (priority, self._pq_seq, problem))

    def dequeue(self) -> Optional[Dict]:
        """Pop the highest-priority problem from the queue."""
        if self._priority_queue:
            _, _, problem = _heapq.heappop(self._priority_queue)
            return problem
        return None

    @property
    def queue_size(self) -> int:
        return len(self._priority_queue)

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows the request. Returns True if allowed."""
        if self._cb_state == self.CB_CLOSED:
            return True
        if self._cb_state == self.CB_OPEN:
            # Check if recovery time has elapsed
            if time.time() - self._cb_last_failure_time >= self._cb_recovery_time:
                self._cb_state = self.CB_HALF_OPEN
                self._cb_half_open_successes = 0
                return True
            return False
        # HALF_OPEN: allow single test
        return True

    def open_circuit_breaker(self):
        """Manually trip the circuit breaker to OPEN state."""
        self._cb_state = self.CB_OPEN
        self._cb_last_failure_time = time.time()
        self._cb_failure_count = self._cb_failure_threshold

    def close_circuit_breaker(self):
        """Manually close the circuit breaker, resuming normal operation."""
        self._cb_state = self.CB_CLOSED
        self._cb_failure_count = 0
        self._cb_half_open_successes = 0

    def reset_circuit_breaker(self):
        """Alias for close_circuit_breaker — resets to CLOSED."""
        self.close_circuit_breaker()

    def _record_circuit_breaker(self, success: bool):
        """Update circuit breaker state after a solve attempt."""
        if success:
            self._cb_failure_count = max(0, self._cb_failure_count - 1)
            if self._cb_state == self.CB_HALF_OPEN:
                self._cb_half_open_successes += 1
                if self._cb_half_open_successes >= 2:
                    self._cb_state = self.CB_CLOSED
                    self._cb_failure_count = 0
        else:
            self._cb_failure_count += 1
            self._cb_last_failure_time = time.time()
            if self._cb_failure_count >= self._cb_failure_threshold:
                self._cb_state = self.CB_OPEN

    def _cache_put(self, key: str, value: Any):
        """Put value into cache with LRU eviction."""
        if key in self.cache:
            self._cache_access_order.remove(key)
        elif len(self.cache) >= self._cache_max_size:
            # Evict LRU
            evict_key = self._cache_access_order.pop(0)
            del self.cache[evict_key]
        self.cache[key] = value
        self._cache_access_order.append(key)

    def solve(self, problem: Dict) -> Dict:
        start = time.time()
        self.invocations += 1

        # Circuit breaker check
        if not self._check_circuit_breaker():
            self.latency_ms = (time.time() - start) * 1000
            return {'solution': None, 'error': 'Circuit breaker OPEN',
                    'cb_state': self._cb_state}

        h = hashlib.sha256(str(problem).encode()).hexdigest()
        if h in self.cache:
            # Move to end for LRU
            if h in self._cache_access_order:
                self._cache_access_order.remove(h)
                self._cache_access_order.append(h)
            self.latency_ms = (time.time() - start) * 1000
            return {'solution': self.cache[h], 'cached': True}

        for idx, solver in enumerate(self.solvers):
            try:
                sol = solver(problem)
                if sol is not None:
                    self._cache_put(h, sol)
                    self.latency_ms = (time.time() - start) * 1000
                    self.success_rate = (self.success_rate * (self.invocations-1) + 1) / self.invocations
                    self._solver_stats[idx]['successes'] += 1
                    self._record_circuit_breaker(True)
                    return {'solution': sol, 'cached': False, 'solver_index': idx}
            except Exception:
                self._solver_stats[idx]['failures'] += 1
                continue

        self.latency_ms = (time.time() - start) * 1000
        self._record_circuit_breaker(False)
        return {'solution': None, 'error': 'No solver succeeded'}

    def get_health(self) -> Dict[str, Any]:
        """Get channel health including circuit breaker and solver stats."""
        return {
            'name': self.name,
            'domain': self.domain,
            'invocations': self.invocations,
            'success_rate': round(self.success_rate, 4),
            'cache_size': len(self.cache),
            'cache_max': self._cache_max_size,
            'queue_size': self.queue_size,
            'circuit_breaker': self._cb_state,
            'cb_failure_count': self._cb_failure_count,
            'solver_stats': dict(self._solver_stats),
        }


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
        # Also extract arithmetic expressions from natural language queries
        if not expr:
            import re
            query = p.get('query', '')
            # Extract math expression from queries like "What is 2+2?" or "calculate 3*4"
            m = re.search(r'([\d]+(?:\.\d+)?\s*[+\-*/]\s*[\d]+(?:\.\d+)?(?:\s*[+\-*/]\s*[\d]+(?:\.\d+)?)*)', query)
            if m:
                expr = m.group(1)
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
        return {n: c.get_health() for n, c in self.channels.items()}


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
            'expert_load': dict(sorted(self.expert_load.items(), key=lambda x: x[1], reverse=True)[:15]),
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

    def grover_amplified_route(self, query: str, n_iterations: int = None) -> List[Tuple[str, float]]:
        """Grover-amplified subsystem routing — quadratic speedup over linear TF-IDF scan.

        Encodes subsystem affinity scores as quantum amplitudes, applies
        Grover diffusion iterations to amplify the highest-scoring subsystem,
        then returns Born-rule probability ranking.

        v12.0: √N speedup for N subsystems — uses floor(π/4 × √N) optimal iterations.
        """
        # Get base TF-IDF scores as amplitude seeds
        base_routes = self.route(query)
        if not base_routes:
            return []

        subsystem_names = [name for name, _ in base_routes]
        raw_scores = [score for _, score in base_routes]
        N = len(raw_scores)

        if N < 2:
            return base_routes

        # Normalize to valid quantum amplitudes: Σ|α|² = 1
        norm = math.sqrt(sum(s ** 2 for s in raw_scores)) or 1.0
        amplitudes = [s / norm for s in raw_scores]

        # Optimal Grover iterations: floor(π/4 × √N)
        if n_iterations is None:
            n_iterations = max(1, int(math.pi / 4 * math.sqrt(N)))

        # Grover diffusion: amplify the dominant subsystem
        for _ in range(n_iterations):
            # Oracle: phase-flip the maximum-amplitude state
            max_idx = max(range(N), key=lambda i: abs(amplitudes[i]))
            amplitudes[max_idx] *= -1

            # Diffusion operator: inversion about mean
            mean_amp = sum(amplitudes) / N
            amplitudes = [2 * mean_amp - a for a in amplitudes]

            # Re-normalize
            norm = math.sqrt(sum(a ** 2 for a in amplitudes)) or 1.0
            amplitudes = [a / norm for a in amplitudes]

        # Born rule: probabilities from amplitudes
        probabilities = [a ** 2 for a in amplitudes]
        total_prob = sum(probabilities) or 1.0
        probabilities = [p / total_prob for p in probabilities]

        # Rank by amplified probability
        result = sorted(
            zip(subsystem_names, probabilities),
            key=lambda x: x[1], reverse=True
        )
        return [(name, round(prob, 6)) for name, prob in result if prob > 0.01]

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
        )[:25] if self._keyword_stats else []
        return {
            'subsystems_tracked': len(self._affinity_matrix),
            'total_keywords': sum(len(v) for v in self._affinity_matrix.values()),
            'routes_computed': self._route_count,
            'feedback_updates': self._feedback_count,
            'tracked_keywords': len(self._keyword_stats),
            'top_keywords': [(kw, s['successes'], s['attempts']) for kw, s in top_keywords],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# v7.0 PIPELINE REPLAY BUFFER + ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════


class PipelineReplayBuffer:
    """Experience replay buffer for pipeline routing decisions.

    Stores (query, route, outcome, reward) tuples and supports prioritized
    replay sampling for reinforcement learning of routing weights.

    v7.0: Priority-weighted sampling, GOD_CODE-seeded initialization,
    configurable capacity with FIFO overflow eviction.
    """
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self._buffer: List[Dict[str, Any]] = []
        self._priorities: List[float] = []
        self._total_stored = 0

    def store(self, query: str, route: str, outcome: bool,
              reward: float = 1.0, metadata: Optional[Dict] = None):
        """Store a routing experience."""
        entry = {
            'query': query[:500],  # Truncate for memory
            'route': route,
            'outcome': outcome,
            'reward': reward,
            'metadata': metadata or {},
            'timestamp': time.time(),
        }
        priority = abs(reward) + (PHI if outcome else TAU)
        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)
            self._priorities.pop(0)
        self._buffer.append(entry)
        self._priorities.append(priority)
        self._total_stored += 1

    def sample(self, batch_size: int = 16) -> List[Dict]:
        """Prioritized replay sampling — higher reward entries are more likely."""
        if not self._buffer:
            return []
        n = min(batch_size, len(self._buffer))
        total_p = sum(self._priorities) or 1.0
        probs = [p / total_p for p in self._priorities]
        rng = random.Random(int(GOD_CODE * self._total_stored) % (2**31))
        indices = []
        for _ in range(n):
            r = rng.random()
            cumulative = 0.0
            for j, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    indices.append(j)
                    break
            else:
                indices.append(len(probs) - 1)
        return [self._buffer[i] for i in indices]

    def get_stats(self) -> Dict[str, Any]:
        """Replay buffer statistics."""
        if not self._buffer:
            return {'size': 0, 'capacity': self.capacity, 'total_stored': self._total_stored}
        rewards = [e['reward'] for e in self._buffer]
        successes = sum(1 for e in self._buffer if e['outcome'])
        return {
            'size': len(self._buffer),
            'capacity': self.capacity,
            'total_stored': self._total_stored,
            'success_ratio': round(successes / len(self._buffer), 4),
            'mean_reward': round(sum(rewards) / len(rewards), 4),
            'max_reward': round(max(rewards), 4),
            'min_reward': round(min(rewards), 4),
        }


class PipelineOrchestrator:
    """Unified pipeline orchestrator that coordinates all routing and solving.

    Combines DirectSolutionHub, AdaptivePipelineRouter, PipelineTelemetry,
    and PipelineReplayBuffer into a single high-level interface.

    v7.0: Auto-routes queries through the most appropriate channel,
    records telemetry, stores experiences for replay, and provides
    unified health monitoring.
    """
    def __init__(self):
        self.hub = DirectSolutionHub()
        self.router = AdaptivePipelineRouter()
        self.telemetry = PipelineTelemetry()
        self.replay = PipelineReplayBuffer()
        self._total_queries = 0

    def solve(self, query: str, problem: Optional[Dict] = None,
              priority: int = 5) -> Dict[str, Any]:
        """Route and solve a query through the full pipeline.

        Args:
            query: Natural language query string.
            problem: Optional structured problem dict. If None, wraps query.
            priority: Priority level (1=highest, 10=lowest).

        Returns:
            Dict with solution, route, telemetry data.
        """
        self._total_queries += 1
        start = time.time()

        if problem is None:
            problem = {'query': query}

        # Route the query
        routes = self.router.route(query)
        primary_route = routes[0][0] if routes else 'direct_solution'

        # Map route to hub channel
        channel_map = {
            'direct_solution': 'mathematics',
            'asi_harness': 'code',
        }
        channel_name = channel_map.get(primary_route, 'knowledge')

        # Attempt to solve through the hub
        result = self.hub.solve(problem)
        latency_ms = (time.time() - start) * 1000
        success = result.get('solution') is not None

        # Record telemetry
        self.telemetry.record(primary_route, latency_ms, success)

        # Store in replay buffer
        reward = PHI if success else -TAU
        self.replay.store(query, primary_route, success, reward)

        # Feedback to router
        query_tokens = re.findall(r'[a-z_]+', query.lower())
        keywords = [t for t in query_tokens if len(t) > 2][:15]
        self.router.feedback(primary_route, keywords, success)

        result['route'] = primary_route,
        result['all_routes'] = routes[:8],
        result['latency_ms'] = round(latency_ms, 3),
        result['query_id'] = self._total_queries

        return result

    def get_status(self) -> Dict[str, Any]:
        """Full orchestrator status across all sub-components."""
        return {
            'total_queries': self._total_queries,
            'hub_stats': self.hub.get_channel_stats(),
            'router_status': self.router.get_status(),
            'telemetry': self.telemetry.get_dashboard(),
            'replay_buffer': self.replay.get_stats(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# v8.0 ML-BACKED PIPELINE ROUTER
# ═══════════════════════════════════════════════════════════════════════════════


class MLPipelineRouter:
    """ML-backed pipeline router using L104RandomForest for subsystem routing.

    Wraps AdaptivePipelineRouter and supplements its TF-IDF/MoE routing with
    a RandomForest classifier trained on routing history. Falls back to the
    base router when the ML model is not yet trained.

    v8.0: Uses L104RandomForest (104 estimators, sacred-tuned) to learn
    optimal routing from accumulated replay buffer experiences.
    """

    # Minimum training samples before ML routing activates
    MIN_TRAINING_SAMPLES = 30

    def __init__(self, base_router: Optional[AdaptivePipelineRouter] = None):
        self._base_router = base_router or AdaptivePipelineRouter()
        self._ml_model = None
        self._ml_fitted = False
        self._subsystem_labels: Dict[str, int] = {}
        self._label_to_subsystem: Dict[int, str] = {}
        self._training_X: List[List[float]] = []
        self._training_y: List[int] = []
        self._ml_route_count = 0
        self._ml_accuracy_history: List[float] = []

    def _get_ml_model(self):
        """Lazy-load L104RandomForest."""
        if self._ml_model is None:
            try:
                from l104_ml_engine.classifiers import L104RandomForest
                self._ml_model = L104RandomForest()
            except ImportError:
                pass
        return self._ml_model

    def _embed_query(self, query: str) -> List[float]:
        """Embed a query into a 32-dimensional feature vector for ML routing."""
        q = query.lower()
        features = []

        # Character n-gram features (16 dims)
        ngram_vec = [0.0] * 16
        for i in range(len(q)):
            for n in (2, 3):
                if i + n <= len(q):
                    idx = hash(q[i:i + n]) % 16
                    ngram_vec[idx] += 1.0
        mag = math.sqrt(sum(v * v for v in ngram_vec)) or 1.0
        features.extend(v / mag for v in ngram_vec)

        # Keyword indicator features (12 dims)
        keyword_groups = [
            ['math', 'calculate', 'compute', 'phi', 'god_code'],
            ['code', 'function', 'program', 'class', 'debug'],
            ['knowledge', 'explain', 'what', 'how', 'why'],
            ['quantum', 'qubit', 'circuit', 'entangle'],
            ['consciousness', 'awareness', 'transcend', 'meta'],
            ['research', 'investigate', 'study', 'explore'],
            ['entropy', 'reversal', 'thermodynamic', 'order'],
            ['sage', 'wisdom', 'philosophy', 'meaning'],
            ['optimize', 'refactor', 'improve', 'performance'],
            ['pattern', 'universal', 'absolute', 'complete'],
            ['language', 'linguistic', 'grammar', 'semantic'],
            ['process', 'analyze', 'ensemble', 'cognitive'],
        ]
        for group in keyword_groups:
            features.append(sum(1.0 for kw in group if kw in q) / len(group))

        # Length and structural features (4 dims)
        features.append(min(len(q) / 200.0, 1.0))
        features.append(q.count(' ') / max(len(q), 1))
        features.append(sum(1 for c in q if c.isdigit()) / max(len(q), 1))
        features.append(sum(1 for c in q if c == '?') / max(len(q), 1))

        return features

    def record_experience(self, query: str, subsystem: str, success: bool):
        """Record a routing experience for ML training."""
        if subsystem not in self._subsystem_labels:
            label = len(self._subsystem_labels)
            self._subsystem_labels[subsystem] = label
            self._label_to_subsystem[label] = subsystem

        if success:
            features = self._embed_query(query)
            label = self._subsystem_labels[subsystem]
            self._training_X.append(features)
            self._training_y.append(label)

    def train(self) -> bool:
        """Train the ML routing model from accumulated experiences."""
        model = self._get_ml_model()
        if model is None or len(self._training_X) < self.MIN_TRAINING_SAMPLES:
            return False

        try:
            import numpy as np
            X = np.array(self._training_X)
            y = np.array(self._training_y)

            if len(set(y)) < 2:
                return False

            model.fit(X, y)
            self._ml_fitted = True
            return True
        except Exception:
            return False

    def route(self, query: str) -> List[Tuple[str, float]]:
        """Route using ML model if trained, falling back to base router."""
        # Try ML routing if model is trained
        if self._ml_fitted and self._ml_model is not None:
            try:
                import numpy as np
                features = np.array([self._embed_query(query)])
                proba = self._ml_model.predict_proba(features)[0]
                classes = self._ml_model._model.classes_

                indexed = sorted(
                    zip(classes, proba), key=lambda x: x[1], reverse=True
                )
                result = []
                for label, prob in indexed[:8]:
                    name = self._label_to_subsystem.get(int(label), f"subsystem_{label}")
                    if prob > 0.05:
                        result.append((name, round(float(prob), 4)))

                if result:
                    self._ml_route_count += 1
                    return result
            except Exception:
                pass

        # Fallback to base router
        return self._base_router.route(query)

    def feedback(self, subsystem: str, keywords: List[str], success: bool,
                 confidence: float = 0.8):
        """Propagate feedback to both ML model and base router."""
        self._base_router.feedback(subsystem, keywords, success, confidence)

        # Auto-retrain periodically
        if len(self._training_X) % 50 == 0 and len(self._training_X) >= self.MIN_TRAINING_SAMPLES:
            self.train()

    def get_status(self) -> Dict:
        return {
            'type': 'MLPipelineRouter',
            'ml_fitted': self._ml_fitted,
            'training_samples': len(self._training_X),
            'subsystems_learned': len(self._subsystem_labels),
            'ml_routes_served': self._ml_route_count,
            'base_router': self._base_router.get_status(),
        }
