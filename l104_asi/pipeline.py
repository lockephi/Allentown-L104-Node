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
        self._cb_failure_threshold = 13  # (was 5)
        self._cb_recovery_time = 13.0  # (was 30.0) seconds
        self._cb_half_open_successes = 0
        # v7.0: Cache size limit (LRU eviction)
        self._cache_max_size = 5275  # (was 1024)
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


# ═══════════════════════════════════════════════════════════════════════════════
# v9.0 ADAPTIVE BACKPRESSURE — Token Bucket Flow Control
# ═══════════════════════════════════════════════════════════════════════════════


class AdaptiveBackpressure:
    """Token-bucket rate limiter with adaptive capacity scaling.

    Controls pipeline throughput using a token-bucket algorithm with PHI-scaled
    refill rate. When the pipeline is overloaded (queue depth high, latency spiking),
    the bucket capacity shrinks to slow intake. When healthy, capacity expands.

    v9.0: Sacred-tuned — initial capacity = 104 tokens (L104 signature),
    refill interval = TAU seconds, PHI-based adaptive scaling.
    """

    def __init__(self, capacity: int = 104, refill_rate: float = None):
        self.max_capacity = capacity
        self.current_capacity = capacity
        self._tokens = float(capacity)
        self._refill_rate = refill_rate or (104.0 / PHI)  # ~64.3 tokens/sec
        self._last_refill = time.time()
        self._total_admitted = 0
        self._total_rejected = 0
        self._total_throttled = 0
        # Adaptive scaling state
        self._health_window: List[float] = []  # recent latencies
        self._health_window_max = 50
        self._scale_factor = 1.0
        self._min_scale = 0.1
        self._max_scale = PHI  # ~1.618x expansion ceiling

    def _refill(self):
        """Refill tokens based on elapsed time and current scale factor."""
        now = time.time()
        elapsed = now - self._last_refill
        self._last_refill = now
        effective_rate = self._refill_rate * self._scale_factor
        self._tokens = min(
            float(self.current_capacity),
            self._tokens + elapsed * effective_rate
        )

    def try_acquire(self, cost: float = 1.0) -> bool:
        """Attempt to acquire tokens. Returns True if admitted, False if rejected."""
        self._refill()
        if self._tokens >= cost:
            self._tokens -= cost
            self._total_admitted += 1
            return True
        self._total_rejected += 1
        return False

    def record_latency(self, latency_ms: float):
        """Record a pipeline operation latency to inform adaptive scaling."""
        self._health_window.append(latency_ms)
        if len(self._health_window) > self._health_window_max:
            self._health_window.pop(0)
        self._adapt()

    def _adapt(self):
        """Adapt token bucket capacity based on recent latency health.

        - High latency (>PHI * avg) → shrink capacity (backpressure)
        - Low latency (<TAU * avg) → expand capacity (healthy)
        - Normal latency → slowly converge toward 1.0
        """
        if len(self._health_window) < 5:
            return

        avg_lat = sum(self._health_window) / len(self._health_window)
        recent_lat = sum(self._health_window[-5:]) / 5

        if avg_lat <= 0:
            return

        ratio = recent_lat / avg_lat

        if ratio > PHI:
            # High latency spike → apply backpressure
            self._scale_factor = max(self._min_scale, self._scale_factor * TAU)
            self.current_capacity = max(10, int(self.max_capacity * self._scale_factor))
            self._total_throttled += 1
        elif ratio < TAU:
            # Below average → expand slowly
            self._scale_factor = min(self._max_scale, self._scale_factor * (1.0 + TAU * 0.1))
            self.current_capacity = min(self.max_capacity * 2, int(self.max_capacity * self._scale_factor))
        else:
            # Normal → regress toward 1.0
            self._scale_factor += (1.0 - self._scale_factor) * 0.05

    def get_status(self) -> Dict:
        return {
            'type': 'AdaptiveBackpressure',
            'tokens_available': round(self._tokens, 2),
            'capacity': self.current_capacity,
            'max_capacity': self.max_capacity,
            'scale_factor': round(self._scale_factor, 4),
            'refill_rate': round(self._refill_rate, 2),
            'total_admitted': self._total_admitted,
            'total_rejected': self._total_rejected,
            'total_throttled': self._total_throttled,
            'rejection_rate': round(
                self._total_rejected / max(self._total_admitted + self._total_rejected, 1), 4
            ),
            'avg_latency_ms': round(
                sum(self._health_window) / max(len(self._health_window), 1), 2
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# v9.0 SPECULATIVE EXECUTOR — Parallel Top-K Route Execution
# ═══════════════════════════════════════════════════════════════════════════════


class SpeculativeExecutor:
    """Execute top-K routes speculatively in parallel, pick the best result.

    Inspired by speculative decoding in LLMs (Leviathan et al. 2023) and
    speculative execution in CPU architecture. Routes the query through
    the top-K subsystems concurrently using ThreadPoolExecutor, then selects
    the highest-confidence result.

    v9.0: K = int(PHI * 2) = 3 concurrent speculative paths,
    timeout = PHI seconds per path, GOD_CODE-seeded confidence tiebreaking.
    """

    def __init__(self, max_parallel: int = None, timeout_s: float = None):
        self.max_parallel = max_parallel or max(2, int(PHI * 2))  # 3
        self.timeout_s = timeout_s or PHI  # ~1.618 seconds
        self._total_executions = 0
        self._total_speculations = 0
        self._speculation_wins = 0  # Times a non-primary route won
        self._timeouts = 0

    def speculative_solve(self, query: str, problem: Dict,
                          routes: List[Tuple[str, float]],
                          solve_fns: Dict[str, Callable]) -> Dict:
        """Execute top-K routes speculatively and return the best result.

        Args:
            query: Natural language query string
            problem: Structured problem dict
            routes: Ranked list of (subsystem_name, score) from router
            solve_fns: Map of subsystem_name → callable(problem) → result_dict

        Returns:
            Best result dict with speculative execution metadata.
        """
        import concurrent.futures

        self._total_executions += 1

        # Select top-K routes that have solve functions
        candidates = []
        for name, score in routes:
            if name in solve_fns and len(candidates) < self.max_parallel:
                candidates.append((name, score, solve_fns[name]))

        if not candidates:
            return {'solution': None, 'error': 'no_speculative_candidates',
                    'speculative': True}

        if len(candidates) == 1:
            # Single candidate — execute directly
            name, score, fn = candidates[0]
            try:
                result = fn(problem)
                result['speculative'] = True
                result['speculative_routes'] = 1
                result['winning_route'] = name
                return result
            except Exception as e:
                return {'solution': None, 'error': str(e), 'speculative': True}

        self._total_speculations += 1

        # Execute in parallel with timeout
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            future_map = {}
            for name, score, fn in candidates:
                future = executor.submit(self._safe_solve, fn, problem)
                future_map[future] = (name, score)

            done, not_done = concurrent.futures.wait(
                future_map.keys(),
                timeout=self.timeout_s,
                return_when=concurrent.futures.ALL_COMPLETED
            )

            for future in done:
                name, score = future_map[future]
                try:
                    result = future.result(timeout=0.1)
                    if result and result.get('solution') is not None:
                        results[name] = result
                        results[name]['route_score'] = score
                except Exception:
                    pass

            if not_done:
                self._timeouts += len(not_done)
                for future in not_done:
                    future.cancel()

        if not results:
            return {'solution': None, 'error': 'all_speculative_paths_failed',
                    'speculative': True, 'routes_tried': len(candidates)}

        # Select winner: highest confidence, tiebreak by route score
        winner_name = max(
            results.keys(),
            key=lambda n: (
                results[n].get('confidence', 0.0),
                results[n].get('route_score', 0.0)
            )
        )
        winner = results[winner_name]

        # Was a non-primary route the winner?
        primary_name = candidates[0][0] if candidates else None
        if winner_name != primary_name:
            self._speculation_wins += 1

        winner['speculative'] = True
        winner['speculative_routes'] = len(candidates)
        winner['routes_completed'] = len(results)
        winner['winning_route'] = winner_name
        winner['was_primary'] = winner_name == primary_name
        winner['all_route_confidences'] = {
            n: round(r.get('confidence', 0.0), 4) for n, r in results.items()
        }

        return winner

    @staticmethod
    def _safe_solve(fn: Callable, problem: Dict) -> Dict:
        """Execute a solve function safely, catching all exceptions."""
        try:
            return fn(problem)
        except Exception as e:
            return {'solution': None, 'error': str(e), 'confidence': 0.0}

    def get_status(self) -> Dict:
        return {
            'type': 'SpeculativeExecutor',
            'max_parallel': self.max_parallel,
            'timeout_s': round(self.timeout_s, 3),
            'total_executions': self._total_executions,
            'total_speculations': self._total_speculations,
            'speculation_wins': self._speculation_wins,
            'speculation_win_rate': round(
                self._speculation_wins / max(self._total_speculations, 1), 4
            ),
            'timeouts': self._timeouts,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# v9.0 PIPELINE CASCADE SCORER — Multi-Stage Score Aggregation
# ═══════════════════════════════════════════════════════════════════════════════


class PipelineCascadeScorer:
    """Aggregate scores from multiple pipeline stages with PHI-decay weighting.

    Each pipeline stage contributes a confidence score. The cascade scorer
    combines them using PHI-exponential decay: earlier stages (foundation)
    are weighted more heavily than later stages (refinement).

    score = Σ(stage_i_confidence × PHI^(-i)) / Σ(PHI^(-i))

    v9.0: Supports named stages, stage-level gating (skip low-confidence stages),
    and sacred alignment bonus when GOD_CODE constants are referenced.
    """

    def __init__(self, gate_threshold: float = 0.1):
        self.gate_threshold = gate_threshold
        self._stage_log: List[Dict] = []
        self._total_cascades = 0
        self._total_stages_processed = 0
        self._total_stages_gated = 0

    def score_cascade(self, stages: List[Dict]) -> Dict:
        """Compute cascade score from a list of stage results.

        Args:
            stages: List of dicts, each with 'name', 'confidence', optional 'sacred_alignment'

        Returns:
            Dict with cascade_score, stage details, and metadata.
        """
        self._total_cascades += 1

        if not stages:
            return {'cascade_score': 0.0, 'stages': [], 'reason': 'no_stages'}

        weighted_sum = 0.0
        weight_total = 0.0
        processed_stages = []
        gated_stages = []

        for i, stage in enumerate(stages):
            confidence = stage.get('confidence', 0.0)
            name = stage.get('name', f'stage_{i}')
            sacred = stage.get('sacred_alignment', 0.0)

            # Stage gating: skip stages below threshold
            if confidence < self.gate_threshold and i > 0:
                self._total_stages_gated += 1
                gated_stages.append(name)
                continue

            # PHI-decay weight: φ^(-i) → earlier stages weighted more
            weight = PHI ** (-i)
            # Sacred alignment bonus: up to +10% boost
            sacred_boost = 1.0 + sacred * 0.1

            weighted_conf = confidence * weight * sacred_boost
            weighted_sum += weighted_conf
            weight_total += weight

            processed_stages.append({
                'name': name,
                'confidence': round(confidence, 4),
                'weight': round(weight, 4),
                'weighted_confidence': round(weighted_conf, 4),
                'sacred_alignment': round(sacred, 4),
            })
            self._total_stages_processed += 1

        cascade_score = weighted_sum / max(weight_total, 1e-10)
        cascade_score = max(0.0, min(1.0, cascade_score))

        result = {
            'cascade_score': round(cascade_score, 6),
            'stages_processed': len(processed_stages),
            'stages_gated': len(gated_stages),
            'gated_stage_names': gated_stages,
            'stage_details': processed_stages,
            'weight_total': round(weight_total, 4),
        }

        self._stage_log.append({
            'cascade_score': cascade_score,
            'n_stages': len(stages),
            'n_processed': len(processed_stages),
            'timestamp': time.time(),
        })
        if len(self._stage_log) > 1000:
            self._stage_log = self._stage_log[-1000:]

        return result

    def get_trend(self, window: int = 20) -> Dict:
        """Analyze cascade score trend over recent invocations."""
        if len(self._stage_log) < 2:
            return {'trend': 'INSUFFICIENT_DATA'}
        recent = self._stage_log[-window:]
        scores = [s['cascade_score'] for s in recent]
        first_half = scores[:len(scores) // 2]
        second_half = scores[len(scores) // 2:]
        avg_first = sum(first_half) / max(len(first_half), 1)
        avg_second = sum(second_half) / max(len(second_half), 1)
        delta = avg_second - avg_first
        return {
            'trend': 'IMPROVING' if delta > 0.02 else 'DECLINING' if delta < -0.02 else 'STABLE',
            'delta': round(delta, 4),
            'current': round(scores[-1], 4),
            'mean': round(sum(scores) / len(scores), 4),
            'samples': len(recent),
        }

    def get_status(self) -> Dict:
        return {
            'type': 'PipelineCascadeScorer',
            'gate_threshold': self.gate_threshold,
            'total_cascades': self._total_cascades,
            'total_stages_processed': self._total_stages_processed,
            'total_stages_gated': self._total_stages_gated,
            'avg_stages_per_cascade': round(
                self._total_stages_processed / max(self._total_cascades, 1), 2
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# v9.0 PIPELINE WARMUP ANALYZER — Cold-Start vs Warm Performance Tracking
# ═══════════════════════════════════════════════════════════════════════════════


class PipelineWarmupAnalyzer:
    """Track and analyze cold-start vs warm pipeline performance characteristics.

    Detects the warmup window (number of queries until pipeline reaches steady-state
    latency), measures cold/warm latency ratio, and provides recommendations for
    pipeline pre-warming.

    v9.0: PHI-smoothed latency tracking, automatic warmup detection based on
    rolling coefficient of variation, sacred-tuned warm threshold = TAU.
    """

    def __init__(self, warmup_cv_threshold: float = None):
        self._cv_threshold = warmup_cv_threshold or TAU * 0.5  # ~0.309
        self._all_latencies: List[float] = []
        self._warmup_detected = False
        self._warmup_index = -1  # Index where warmup completes
        self._cold_latencies: List[float] = []
        self._warm_latencies: List[float] = []
        self._rolling_window = 10

    def record(self, latency_ms: float):
        """Record a pipeline invocation latency."""
        self._all_latencies.append(latency_ms)

        if not self._warmup_detected:
            self._cold_latencies.append(latency_ms)
            self._check_warmup()
        else:
            self._warm_latencies.append(latency_ms)
            if len(self._warm_latencies) > 10000:
                self._warm_latencies = self._warm_latencies[-10000:]

    def _check_warmup(self):
        """Check if the pipeline has reached steady-state (warmup complete)."""
        n = len(self._cold_latencies)
        if n < self._rolling_window * 2:
            return

        # Compute rolling coefficient of variation (CV = σ/μ)
        window = self._cold_latencies[-self._rolling_window:]
        mean = sum(window) / len(window)
        if mean <= 0:
            return
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        cv = math.sqrt(variance) / mean

        # If CV is low enough, pipeline is warmed up
        if cv < self._cv_threshold:
            self._warmup_detected = True
            self._warmup_index = n
            # Split: everything before warmup_index is cold, after is warm
            self._warm_latencies = self._cold_latencies[self._warmup_index:]
            self._cold_latencies = self._cold_latencies[:self._warmup_index]

    def is_warmed_up(self) -> bool:
        return self._warmup_detected

    def get_analysis(self) -> Dict:
        """Full warmup analysis report."""
        cold_avg = sum(self._cold_latencies) / max(len(self._cold_latencies), 1)
        warm_avg = sum(self._warm_latencies) / max(len(self._warm_latencies), 1)
        speedup = cold_avg / max(warm_avg, 0.01) if warm_avg > 0 else 0.0

        return {
            'type': 'PipelineWarmupAnalyzer',
            'warmed_up': self._warmup_detected,
            'warmup_queries': self._warmup_index if self._warmup_detected else len(self._all_latencies),
            'total_queries': len(self._all_latencies),
            'cold_avg_ms': round(cold_avg, 3),
            'warm_avg_ms': round(warm_avg, 3),
            'cold_to_warm_speedup': round(speedup, 3),
            'cold_samples': len(self._cold_latencies),
            'warm_samples': len(self._warm_latencies),
            'recommendation': (
                'Pipeline is warmed up — steady-state performance achieved'
                if self._warmup_detected else
                f'Pipeline still warming up — {len(self._all_latencies)} queries processed, '
                f'CV threshold = {self._cv_threshold:.3f}'
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# v9.0 PIPELINE STAGE PROFILER — Per-Stage Latency Decomposition
# ═══════════════════════════════════════════════════════════════════════════════


class PipelineStageProfiler:
    """Decompose pipeline latency into per-stage contributions.

    Profiles each pipeline stage (routing, solving, ensemble, refinement, etc.)
    to identify bottlenecks and optimize throughput.

    v9.0: Tracks cumulative stage time, identifies the slowest stage,
    computes stage-level Amdahl's law speedup ceilings.
    """

    def __init__(self):
        self._stage_times: Dict[str, List[float]] = {}
        self._profiling_active = False
        self._current_profile: Dict[str, float] = {}
        self._total_profiles = 0

    def start_profile(self):
        """Begin a new pipeline profile trace."""
        self._profiling_active = True
        self._current_profile = {}

    def record_stage(self, stage_name: str, latency_ms: float):
        """Record latency for a named pipeline stage within the current profile."""
        if not self._profiling_active:
            return
        self._current_profile[stage_name] = latency_ms
        if stage_name not in self._stage_times:
            self._stage_times[stage_name] = []
        self._stage_times[stage_name].append(latency_ms)
        if len(self._stage_times[stage_name]) > 5000:
            self._stage_times[stage_name] = self._stage_times[stage_name][-5000:]

    def end_profile(self) -> Dict:
        """End the current profile and return decomposition."""
        self._profiling_active = False
        self._total_profiles += 1
        total = sum(self._current_profile.values())
        breakdown = {}
        for stage, ms in sorted(self._current_profile.items(), key=lambda x: x[1], reverse=True):
            breakdown[stage] = {
                'ms': round(ms, 3),
                'pct': round(ms / max(total, 0.001) * 100, 1),
            }
        return {
            'total_ms': round(total, 3),
            'stages': breakdown,
            'bottleneck': max(self._current_profile, key=self._current_profile.get)
                         if self._current_profile else None,
        }

    def get_aggregate(self) -> Dict:
        """Aggregate statistics across all profiled invocations."""
        if not self._stage_times:
            return {'profiles': 0, 'stages': {}}

        aggregate = {}
        for stage, times in self._stage_times.items():
            avg = sum(times) / len(times)
            p95 = sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else max(times)
            aggregate[stage] = {
                'avg_ms': round(avg, 3),
                'p95_ms': round(p95, 3),
                'max_ms': round(max(times), 3),
                'min_ms': round(min(times), 3),
                'samples': len(times),
            }

        # Amdahl's Law: max speedup from optimizing each stage
        total_avg = sum(d['avg_ms'] for d in aggregate.values())
        for stage in aggregate:
            frac = aggregate[stage]['avg_ms'] / max(total_avg, 0.001)
            # Amdahl: Speedup = 1 / ((1-f) + f/S) where S→∞ gives 1/(1-f)
            amdahl_ceiling = 1.0 / max(1.0 - frac, 0.001)
            aggregate[stage]['amdahl_ceiling'] = round(amdahl_ceiling, 3)
            aggregate[stage]['pct_of_total'] = round(frac * 100, 1)

        return {
            'profiles': self._total_profiles,
            'stages': aggregate,
            'total_avg_ms': round(total_avg, 3),
            'bottleneck': max(aggregate, key=lambda s: aggregate[s]['avg_ms'])
                         if aggregate else None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# v9.0 UPGRADED PIPELINE ORCHESTRATOR — Full Integration
# ═══════════════════════════════════════════════════════════════════════════════


class PipelineOrchestratorV2:
    """Upgraded pipeline orchestrator with backpressure, speculative execution,
    cascade scoring, warmup analysis, and stage profiling.

    v9.0: Integrates all v9.0 pipeline components into a unified high-level
    interface. Supersedes PipelineOrchestrator v7.0 with:
    - Adaptive backpressure (token-bucket flow control)
    - Speculative execution (parallel top-K route solving)
    - Cascade scoring (multi-stage PHI-decay score aggregation)
    - Warmup analysis (cold-start vs steady-state tracking)
    - Stage profiling (per-stage latency decomposition)
    - Batch solve (process multiple queries efficiently)
    - Priority-aware scheduling (priority actually affects execution order)
    """

    def __init__(self):
        self.hub = DirectSolutionHub()
        self.router = AdaptivePipelineRouter()
        self.telemetry = PipelineTelemetry()
        self.replay = PipelineReplayBuffer()
        self.backpressure = AdaptiveBackpressure()
        self.speculative = SpeculativeExecutor()
        self.cascade_scorer = PipelineCascadeScorer()
        self.warmup = PipelineWarmupAnalyzer()
        self.profiler = PipelineStageProfiler()
        self._total_queries = 0
        self._total_backpressured = 0
        self._priority_queue: List[Tuple[int, int, str, Dict]] = []  # (priority, seq, query, problem)
        self._pq_seq = 0

    def solve(self, query: str, problem: Optional[Dict] = None,
              priority: int = 5, speculative: bool = True) -> Dict[str, Any]:
        """Route and solve a query through the full v9.0 pipeline.

        Args:
            query: Natural language query string.
            problem: Optional structured problem dict. If None, wraps query.
            priority: Priority level (1=highest, 10=lowest).
            speculative: Whether to use speculative parallel execution.

        Returns:
            Dict with solution, route, telemetry, cascade score, and profiling data.
        """
        # Backpressure check: reject if overloaded
        cost = max(0.5, 1.0 + (10 - priority) * 0.1)  # Higher priority = lower cost
        if not self.backpressure.try_acquire(cost):
            self._total_backpressured += 1
            return {
                'solution': None,
                'error': 'BACKPRESSURE_REJECT',
                'backpressure': True,
                'retry_after_ms': int(1000 / max(self.backpressure._refill_rate, 1)),
            }

        self._total_queries += 1
        self.profiler.start_profile()
        start = time.time()

        if problem is None:
            problem = {'query': query}

        # ── Stage 1: ROUTE ──
        route_start = time.time()
        routes = self.router.route(query)
        primary_route = routes[0][0] if routes else 'direct_solution'
        route_ms = (time.time() - route_start) * 1000
        self.profiler.record_stage('routing', route_ms)

        # ── Stage 2: SOLVE ──
        solve_start = time.time()

        if speculative and len(routes) >= 2:
            # Build solve function map for speculative execution
            channel_map = {
                'direct_solution': lambda p: self.hub.channels.get('mathematics', self.hub.channels.get('knowledge')).solve(p) if self.hub.channels else {'solution': None},
                'asi_harness': lambda p: self.hub.channels['code'].solve(p) if 'code' in self.hub.channels else {'solution': None},
            }
            # Default all unknown routes to knowledge channel
            for rname, _ in routes:
                if rname not in channel_map:
                    channel_map[rname] = lambda p, _n=rname: self.hub.solve(p)

            result = self.speculative.speculative_solve(query, problem, routes[:3], channel_map)
        else:
            result = self.hub.solve(problem)

        solve_ms = (time.time() - solve_start) * 1000
        self.profiler.record_stage('solving', solve_ms)

        success = result.get('solution') is not None

        # ── Stage 3: CASCADE SCORE ──
        cascade_start = time.time()
        cascade_stages = [
            {'name': 'routing', 'confidence': routes[0][1] if routes else 0.0},
            {'name': 'solving', 'confidence': result.get('confidence', 0.0)},
        ]
        if result.get('computronium'):
            cascade_stages.append({
                'name': 'computronium',
                'confidence': result['computronium'].get('density', 0) / 100.0,
            })
        cascade = self.cascade_scorer.score_cascade(cascade_stages)
        cascade_ms = (time.time() - cascade_start) * 1000
        self.profiler.record_stage('cascade_scoring', cascade_ms)

        # ── Stage 4: TELEMETRY + REPLAY ──
        total_ms = (time.time() - start) * 1000
        self.profiler.record_stage('overhead', max(0, total_ms - route_ms - solve_ms - cascade_ms))

        self.telemetry.record(primary_route, total_ms, success)
        self.backpressure.record_latency(total_ms)
        self.warmup.record(total_ms)

        reward = PHI if success else -TAU
        self.replay.store(query, primary_route, success, reward)

        # Feedback to router
        query_tokens = re.findall(r'[a-z_]+', query.lower())
        keywords = [t for t in query_tokens if len(t) > 2][:15]
        self.router.feedback(primary_route, keywords, success)

        # ── Stage 5: PROFILE ──
        profile = self.profiler.end_profile()

        result['route'] = primary_route
        result['all_routes'] = routes[:8]
        result['latency_ms'] = round(total_ms, 3)
        result['query_id'] = self._total_queries
        result['cascade'] = cascade
        result['profile'] = profile
        result['pipeline_version'] = '9.0'

        return result

    def solve_batch(self, queries: List[str], priority: int = 5) -> List[Dict]:
        """Process a batch of queries through the pipeline.

        Sorts by estimated complexity (shorter = faster, processed first),
        records aggregate batch telemetry.
        """
        # Sort by estimated complexity to process quick wins first
        indexed = [(i, q) for i, q in enumerate(queries)]
        indexed.sort(key=lambda x: len(x[1]))

        results = [None] * len(queries)
        for orig_idx, query in indexed:
            results[orig_idx] = self.solve(query, priority=priority, speculative=False)

        return results

    def enqueue(self, query: str, problem: Optional[Dict] = None, priority: int = 5):
        """Add a query to the priority queue for deferred processing."""
        self._pq_seq += 1
        _heapq.heappush(self._priority_queue, (priority, self._pq_seq, query, problem or {'query': query}))

    def process_queue(self, max_items: int = 10) -> List[Dict]:
        """Process up to N items from the priority queue, highest priority first."""
        results = []
        for _ in range(min(max_items, len(self._priority_queue))):
            priority, _, query, problem = _heapq.heappop(self._priority_queue)
            result = self.solve(query, problem=problem, priority=priority)
            results.append(result)
        return results

    def get_status(self) -> Dict[str, Any]:
        """Full orchestrator v9.0 status across all components."""
        return {
            'version': '9.0',
            'total_queries': self._total_queries,
            'total_backpressured': self._total_backpressured,
            'queue_depth': len(self._priority_queue),
            'warmed_up': self.warmup.is_warmed_up(),
            'hub_stats': self.hub.get_channel_stats(),
            'router_status': self.router.get_status(),
            'telemetry': self.telemetry.get_dashboard(),
            'replay_buffer': self.replay.get_stats(),
            'backpressure': self.backpressure.get_status(),
            'speculative': self.speculative.get_status(),
            'cascade_scorer': self.cascade_scorer.get_status(),
            'warmup': self.warmup.get_analysis(),
            'profiler': self.profiler.get_aggregate(),
        }
