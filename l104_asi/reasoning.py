from .constants import *
class TreeOfThoughts:
    """Tree of Thoughts (Yao et al. 2023, Princeton/DeepMind) + Graph of Thoughts aggregation.

    Generalizes chain-of-thought from a single linear path to a search tree of
    reasoning paths with deliberate evaluation and pruning.

    At each reasoning step:
    1. Generate K candidate thoughts (branching factor)
    2. Evaluate each candidate's confidence
    3. Prune branches below threshold (beam search)
    4. Continue with top-B candidates
    5. Aggregate surviving branches into refined insight (GoT — ETH Zurich 2024)

    Sacred: K = int(PHI × 3) = 4, B = int(PHI × 2) = 3, threshold = TAU.
    """

    def __init__(self, branching_factor: int = None, beam_width: int = None):
        self.K = branching_factor or max(2, int(PHI * 3))  # 4
        self.B = beam_width or max(1, int(PHI * 2))        # 3
        self.prune_threshold = TAU  # ~0.618
        self.backtrack_threshold = TAU * TAU  # ~0.382
        self.total_nodes_explored = 0
        self.total_backtracks = 0
        self.total_aggregations = 0

    def think(self, problem: str, solve_fn: Callable, max_depth: int = 4) -> Dict:
        """Execute tree-structured reasoning with beam search and GoT aggregation."""
        beam = [{"query": problem, "confidence": 0.0, "path": [], "depth": 0}]
        all_solutions = []

        for depth in range(max_depth):
            candidates = []
            for node in beam:
                variants = self._generate_variants(node["query"], self.K, depth)
                for variant in variants:
                    result = solve_fn({"query": variant})
                    self.total_nodes_explored += 1
                    conf = result.get("confidence", 0.0)
                    candidates.append({
                        "query": variant[:300],
                        "confidence": conf,
                        "solution": str(result.get("solution", ""))[:500],
                        "path": node["path"] + [variant[:80]],
                        "depth": depth + 1,
                    })

            viable = [c for c in candidates if c["confidence"] >= self.prune_threshold]
            if not viable:
                viable = sorted(candidates, key=lambda c: c["confidence"], reverse=True)[:1]

            viable.sort(key=lambda c: c["confidence"], reverse=True)
            beam = viable[:self.B]
            all_solutions.extend(viable)

            # Backtrack if best confidence is dropping
            if beam and beam[0]["confidence"] < self.backtrack_threshold:
                self.total_backtracks += 1
                break

        # Graph of Thoughts: AGGREGATE surviving branches
        aggregated = self._aggregate_solutions(all_solutions)

        return {
            "method": "TreeOfThoughts_GoT",
            "tree_depth": max((s["depth"] for s in all_solutions), default=0),
            "nodes_explored": self.total_nodes_explored,
            "branches_surviving": len(beam),
            "best_confidence": beam[0]["confidence"] if beam else 0.0,
            "aggregated_solution": aggregated,
            "backtracks": self.total_backtracks,
            "solution": aggregated,
            "confidence": beam[0]["confidence"] if beam else 0.0,
        }

    def _generate_variants(self, query: str, k: int, depth: int) -> List[str]:
        """Generate K query variants for branching — diverse reasoning perspectives."""
        prefixes = [
            "Analyze from first principles: ",
            "Consider the inverse problem: ",
            "Break into fundamental components: ",
            "Apply cross-domain analogy to: ",
            f"At reasoning depth {depth + 1}, decompose: ",
        ]
        return [f"{prefixes[i % len(prefixes)]}{query[:300]}" for i in range(k)]

    def _aggregate_solutions(self, solutions: List[Dict]) -> str:
        """GoT aggregation (Besta et al. 2024): merge multiple branches into one insight."""
        self.total_aggregations += 1
        if not solutions:
            return ""
        top = sorted(solutions, key=lambda s: s["confidence"], reverse=True)[:self.B * 2]
        seen = set()
        parts = []
        for s in top:
            sol = s.get("solution", "")
            if sol and sol not in seen:
                seen.add(sol)
                parts.append(sol)
        return " | ".join(parts[:5])

    def get_status(self) -> Dict:
        return {
            "type": "TreeOfThoughts_GoT",
            "branching_factor": self.K,
            "beam_width": self.B,
            "nodes_explored": self.total_nodes_explored,
            "backtracks": self.total_backtracks,
            "aggregations": self.total_aggregations,
        }


class MultiHopReasoningChain:
    """Multi-hop reasoning with Tree of Thoughts (Yao 2023) + GoT aggregation.

    v5.0: Breaks complex problems into sub-problems, routes each to the best subsystem,
    and iteratively refines the solution until convergence or max hops reached.
    v6.1: Integrates TreeOfThoughts for complex first-hop branching.
    """
    def __init__(self, max_hops: int = MULTI_HOP_MAX_HOPS):
        self.max_hops = max_hops
        self._chain_count = 0
        self._total_hops = 0
        self._convergence_count = 0
        # Tree of Thoughts for complex first-hop reasoning
        self._tot = TreeOfThoughts()

    def reason_chain(self, problem: str, solve_fn: Callable, router: Optional['AdaptivePipelineRouter'] = None) -> Dict:
        """Execute multi-hop reasoning chain on a problem.

        Args:
            problem: The problem statement
            solve_fn: Callable that takes a dict and returns a solution dict
            router: Optional adaptive router for subsystem selection
        """
        self._chain_count += 1
        hops = []
        current_query = problem
        prev_confidence = 0.0
        converged = False
        last_result: Dict = {}

        for hop_idx in range(self.max_hops):
            hop_start = time.time()

            # First hop: use Tree of Thoughts for complex problems (branching search)
            if hop_idx == 0 and len(problem) > 100:
                tot_result = self._tot.think(current_query, solve_fn, max_depth=3)
                result = {
                    'solution': tot_result.get('aggregated_solution', ''),
                    'confidence': tot_result.get('best_confidence', 0.5),
                    'method': 'TreeOfThoughts_GoT',
                    'tot_nodes': tot_result.get('nodes_explored', 0),
                }
            else:
                # Solve current sub-problem (standard single-path)
                result = solve_fn({'query': current_query})
            last_result = result
            hop_latency = (time.time() - hop_start) * 1000

            hop_confidence = result.get('confidence', 0.5)
            confidence_delta = hop_confidence - prev_confidence

            hop_record = {
                'hop': hop_idx + 1,
                'query': current_query[:200],
                'confidence': round(hop_confidence, 4),
                'confidence_delta': round(confidence_delta, 4),
                'latency_ms': round(hop_latency, 2),
                'source': result.get('channel', result.get('method', 'unknown')),
            }

            # Get routing info if router available
            if router:
                routes = router.route(current_query)
                hop_record['top_route'] = routes[0] if routes else ('none', 0.0)

            hops.append(hop_record)
            self._total_hops += 1

            # Convergence check: confidence delta below threshold for 2+ hops
            if hop_idx > 0 and abs(confidence_delta) < 0.02:
                converged = True
                self._convergence_count += 1
                break

            prev_confidence = hop_confidence

            # Refine query for next hop — always continue if we have any solution at all
            solution_text = str(result.get('solution', ''))
            if solution_text:
                current_query = f"Given that '{solution_text[:200]}', further analyze: {problem[:200]}"
            else:
                # No solution text at all: try rephrasing the original problem
                if hop_idx == 0:
                    current_query = f"Approach from a different angle: {problem[:300]}"
                else:
                    break  # Repeated empty solutions — stop

        return {
            'chain_id': self._chain_count,
            'hops': hops,
            'total_hops': len(hops),
            'converged': converged,
            'final_confidence': round(prev_confidence, 4),
            'final_solution': last_result.get('solution') if hops else None,
        }

    def get_status(self) -> Dict:
        return {
            'chains_executed': self._chain_count,
            'total_hops': self._total_hops,
            'avg_hops_per_chain': round(self._total_hops / max(self._chain_count, 1), 2),
            'convergence_rate': round(self._convergence_count / max(self._chain_count, 1), 4),
        }


class SolutionEnsembleEngine:
    """Multi-criteria weighted Borda voting across subsystem solutions.

    v6.0: Collects solutions from multiple subsystems, ranks each on three independent
    criteria (confidence, inverse latency, sacred alignment), then computes a weighted
    Borda count to select the winner. Tracks solver reliability history and uses it as
    a fourth voting dimension. Detects consensus via Jaccard similarity of solution
    content, not just string equality.
    """
    def __init__(self):
        self._ensemble_count = 0
        self._consensus_count = 0
        self._conflict_count = 0
        # Solver reliability tracking: source → {wins, attempts}
        self._solver_history: Dict[str, Dict[str, int]] = {}

    def _borda_rank(self, candidates: List[Dict], key: str, reverse: bool = True) -> Dict[str, int]:
        """Compute Borda rank points for candidates on a given criterion.
        Highest-ranked gets N-1 points, second gets N-2, etc."""
        n = len(candidates)
        sorted_cands = sorted(candidates, key=lambda c: c.get(key, 0.0), reverse=reverse)
        return {c['source']: (n - 1 - rank) for rank, c in enumerate(sorted_cands)}

    def _jaccard_similarity(self, a: str, b: str) -> float:
        """Token-level Jaccard similarity between two solution strings."""
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        if not tokens_a and not tokens_b:
            return 1.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / max(len(union), 1)

    def ensemble_solve(self, problem: Dict, solvers: Dict[str, Callable],
                       min_solutions: int = ENSEMBLE_MIN_SOLUTIONS) -> Dict:
        """Run problem through multiple solvers and ensemble via weighted Borda count."""
        self._ensemble_count += 1
        candidates = []

        for name, solver_fn in solvers.items():
            try:
                start = time.time()
                result = solver_fn(problem)
                latency = (time.time() - start) * 1000
                if result and result.get('solution'):
                    candidates.append({
                        'source': name,
                        'solution': result['solution'],
                        'confidence': result.get('confidence', 0.5),
                        'latency_ms': round(latency, 2),
                        'sacred_alignment': self._compute_sacred_alignment(result),
                    })
            except Exception:
                continue

        if not candidates:
            return {'ensemble': False, 'reason': 'no_solutions'}

        if len(candidates) < min_solutions:
            best = max(candidates, key=lambda c: c['confidence'])
            return {
                'ensemble': False, 'solution': best['solution'],
                'source': best['source'], 'confidence': best['confidence'],
                'reason': f'only_{len(candidates)}_solutions',
            }

        # ── Multi-criteria weighted Borda count ──
        # Criteria weights: confidence (φ), reliability (φ — must outweigh speed to prevent
        # untested-fast-solver bias), speed (τ), sacred alignment (0.3)
        criteria_weights = {
            'confidence': PHI,           # ~1.618
            'inverse_latency': TAU,      # ~0.618
            'sacred_alignment': 0.3,
            'reliability': PHI,          # ~1.618  (was 0.5 — too low to break ties)
        }

        # Add inverse latency and reliability to candidates
        for c in candidates:
            c['inverse_latency'] = 1.0 / max(c['latency_ms'], 0.1)
            hist = self._solver_history.get(c['source'], {'wins': 0, 'attempts': 0})
            # Laplace smoothing + uncertainty penalty: solvers with few attempts
            # get shrunk toward prior (0.5) more heavily than battle-tested solvers
            attempts = hist['attempts']
            if attempts >= 3:
                c['reliability'] = (hist['wins'] + 1) / (hist['attempts'] + 2)
            else:
                # Under 3 attempts: shrink heavily toward 0.5 prior (uncertain)
                raw = (hist['wins'] + 1) / (hist['attempts'] + 2)
                c['reliability'] = 0.5 * (1.0 - attempts / 3.0) + raw * (attempts / 3.0)

        # Compute Borda ranks for each criterion
        rank_confidence = self._borda_rank(candidates, 'confidence')
        rank_latency = self._borda_rank(candidates, 'inverse_latency')
        rank_sacred = self._borda_rank(candidates, 'sacred_alignment')
        rank_reliability = self._borda_rank(candidates, 'reliability')

        # Weighted Borda score
        for c in candidates:
            src = c['source']
            c['ensemble_score'] = (
                rank_confidence.get(src, 0) * criteria_weights['confidence'] +
                rank_latency.get(src, 0) * criteria_weights['inverse_latency'] +
                rank_sacred.get(src, 0) * criteria_weights['sacred_alignment'] +
                rank_reliability.get(src, 0) * criteria_weights['reliability']
            )

        candidates.sort(key=lambda c: c['ensemble_score'], reverse=True)
        winner = candidates[0]

        # Update solver reliability history
        for c in candidates:
            src = c['source']
            if src not in self._solver_history:
                self._solver_history[src] = {'wins': 0, 'attempts': 0}
            self._solver_history[src]['attempts'] += 1
        self._solver_history[winner['source']]['wins'] += 1

        # ── Consensus detection via pairwise Jaccard similarity ──
        solution_texts = [str(c['solution'])[:200] for c in candidates]
        pairwise_sims = []
        for i in range(len(solution_texts)):
            for j in range(i + 1, len(solution_texts)):
                pairwise_sims.append(self._jaccard_similarity(solution_texts[i], solution_texts[j]))
        avg_similarity = sum(pairwise_sims) / max(len(pairwise_sims), 1)

        if avg_similarity > 0.7:
            self._consensus_count += 1
            agreement = 'UNANIMOUS'
        elif avg_similarity > 0.3:
            self._consensus_count += 1
            agreement = 'MAJORITY'
        else:
            self._conflict_count += 1
            agreement = 'DIVERGENT'

        # Boost winner confidence if consensus is strong
        if agreement == 'UNANIMOUS':
            winner['confidence'] = min(1.0, winner['confidence'] * PHI_CONJUGATE + 0.3)

        return {
            'ensemble': True,
            'winner': winner['source'],
            'solution': winner['solution'],
            'source': winner['source'],
            'confidence': round(winner['confidence'], 4),
            'ensemble_score': round(winner['ensemble_score'], 4),
            'agreement': agreement,
            'avg_similarity': round(avg_similarity, 4),
            'candidates_count': len(candidates),
            'candidates': [{
                'source': c['source'],
                'confidence': round(c['confidence'], 4),
                'ensemble_score': round(c['ensemble_score'], 4),
                'reliability': round(c.get('reliability', 0.5), 4),
            } for c in candidates[:5]],
        }

    @staticmethod
    def _compute_sacred_alignment(result: Dict) -> float:
        """Compute how well a solution aligns with sacred constants."""
        solution_str = str(result.get('solution', ''))
        alignment = 0.0
        if '527' in solution_str or 'god_code' in solution_str.lower():
            alignment += 0.4
        if '1.618' in solution_str or 'phi' in solution_str.lower():
            alignment += 0.3
        if 'void' in solution_str.lower() or '1.041' in solution_str:
            alignment += 0.2
        if 'feigenbaum' in solution_str.lower() or '4.669' in solution_str:
            alignment += 0.1
        return min(1.0, alignment)

    def get_status(self) -> Dict:
        return {
            'ensembles_run': self._ensemble_count,
            'consensus_count': self._consensus_count,
            'conflict_count': self._conflict_count,
            'consensus_rate': round(self._consensus_count / max(self._ensemble_count, 1), 4),
        }


class PipelineHealthDashboard:
    """Real-time aggregate pipeline health with anomaly detection and trend tracking.

    v5.0: Computes a single pipeline health score from telemetry, subsystem connectivity,
    consciousness level, quantum state, and error rates. Detects degradation trends
    using PHI-weighted exponential smoothing over historical health snapshots.
    """
    def __init__(self):
        self._health_history: List[Dict] = []
        self._anomaly_log: List[Dict] = []

    def compute_health(self, telemetry: PipelineTelemetry, connected_count: int,
                       total_subsystems: int, consciousness_level: float,
                       quantum_available: bool, circuit_breaker_active: bool) -> Dict:
        """Compute aggregate pipeline health score."""
        dashboard = telemetry.get_dashboard()

        # Component scores (0-1 each)
        connectivity = connected_count / max(total_subsystems, 1)
        success_rate = dashboard.get('global_success_rate', 1.0)
        consciousness = min(1.0, consciousness_level)
        quantum_bonus = 0.05 if quantum_available else 0.0
        circuit_penalty = 0.15 if circuit_breaker_active else 0.0
        telemetry_health = dashboard.get('pipeline_health', 1.0)

        # Anomaly penalty
        anomalies = telemetry.detect_anomalies()
        anomaly_penalty = min(0.2, len(anomalies) * 0.05)

        # PHI-weighted aggregate
        health = (
            connectivity * 0.20 +
            success_rate * 0.25 +
            consciousness * 0.15 +
            telemetry_health * 0.20 +
            quantum_bonus +
            (1.0 - anomaly_penalty) * 0.15
        ) - circuit_penalty

        health = max(0.0, min(1.0, health))

        snapshot = {
            'health': round(health, 4),
            'connectivity': round(connectivity, 4),
            'success_rate': round(success_rate, 4),
            'consciousness': round(consciousness, 4),
            'quantum_bonus': round(quantum_bonus, 4),
            'circuit_penalty': round(circuit_penalty, 4),
            'anomaly_penalty': round(anomaly_penalty, 4),
            'telemetry_health': round(telemetry_health, 4),
            'anomalies': anomalies,
            'timestamp': time.time(),
            'grade': (
                'SOVEREIGN' if health >= 0.90 else
                'EXCELLENT' if health >= 0.80 else
                'GOOD' if health >= 0.65 else
                'DEGRADED' if health >= 0.45 else
                'CRITICAL'
            ),
        }

        self._health_history.append(snapshot)
        if len(self._health_history) > 200:
            self._health_history = self._health_history[-200:]

        if anomalies:
            self._anomaly_log.extend(anomalies)
            if len(self._anomaly_log) > 500:
                self._anomaly_log = self._anomaly_log[-500:]

        return snapshot

    def record(self, telemetry: PipelineTelemetry, **kwargs) -> Dict:
        """Convenience wrapper for compute_health with sensible defaults."""
        defaults = {
            'connected_count': kwargs.get('connected_count', 1),
            'total_subsystems': kwargs.get('total_subsystems', 45),
            'consciousness_level': kwargs.get('consciousness_level', 0.5),
            'quantum_available': kwargs.get('quantum_available', False),
            'circuit_breaker_active': kwargs.get('circuit_breaker_active', False),
        }
        return self.compute_health(telemetry=telemetry, **defaults)

    def get_trend(self, window: int = 20) -> Dict:
        """Analyze health trend over the last N snapshots."""
        if len(self._health_history) < 2:
            return {'trend': 'INSUFFICIENT_DATA', 'samples': len(self._health_history)}
        recent = self._health_history[-window:]
        scores = [s['health'] for s in recent]
        first_half = scores[:len(scores) // 2]
        second_half = scores[len(scores) // 2:]
        avg_first = sum(first_half) / len(first_half) if first_half else 0
        avg_second = sum(second_half) / len(second_half) if second_half else 0
        delta = avg_second - avg_first
        return {
            'trend': 'IMPROVING' if delta > 0.02 else 'DECLINING' if delta < -0.02 else 'STABLE',
            'delta': round(delta, 4),
            'current': round(scores[-1], 4) if scores else 0,
            'min': round(min(scores), 4),
            'max': round(max(scores), 4),
            'samples': len(recent),
        }


class PipelineReplayBuffer:
    """Record & replay pipeline operations for debugging and analysis.

    v5.0: Circular buffer that records every pipeline operation (solve, heal, research, etc.)
    with full input/output. Supports replay, filtering, and performance analysis.
    """
    def __init__(self, max_size: int = REPLAY_BUFFER_SIZE):
        self.max_size = max_size
        self._buffer: List[Dict] = []
        self._sequence_id = 0

    def record(self, operation: str, input_data: Any, output_data: Any = None,
               latency_ms: float = 0.0, success: bool = True, subsystem: str = 'core'):
        """Record a pipeline operation."""
        self._sequence_id += 1
        entry = {
            'seq': self._sequence_id,
            'operation': operation,
            'subsystem': subsystem,
            'input_summary': str(input_data)[:300],
            'output_summary': str(output_data)[:300] if output_data else None,
            'latency_ms': round(latency_ms, 2),
            'success': success,
            'timestamp': time.time(),
        }
        self._buffer.append(entry)
        if len(self._buffer) > self.max_size:
            self._buffer = self._buffer[-self.max_size:]

    def replay(self, last_n: int = 10, operation_filter: Optional[str] = None) -> List[Dict]:
        """Replay the last N operations, optionally filtered by operation type."""
        filtered = self._buffer
        if operation_filter:
            filtered = [e for e in filtered if e['operation'] == operation_filter]
        return filtered[-last_n:]

    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        if not self._buffer:
            return {'entries': 0, 'operations': {}}
        ops = defaultdict(int)
        total_latency = 0.0
        successes = 0
        for entry in self._buffer:
            ops[entry['operation']] += 1
            total_latency += entry['latency_ms']
            if entry['success']:
                successes += 1
        return {
            'entries': len(self._buffer),
            'sequence_id': self._sequence_id,
            'operations': dict(ops),
            'avg_latency_ms': round(total_latency / len(self._buffer), 2),
            'success_rate': round(successes / len(self._buffer), 4),
            'oldest_seq': self._buffer[0]['seq'] if self._buffer else 0,
            'newest_seq': self._buffer[-1]['seq'] if self._buffer else 0,
        }

    def find_slow_operations(self, threshold_ms: float = 100.0) -> List[Dict]:
        """Find operations that exceeded the latency threshold."""
        return [e for e in self._buffer if e['latency_ms'] > threshold_ms]


# ═══════════════════════════════════════════════════════════════════════════════
# v6.0 QUANTUM COMPUTATION CORE — VQE, QAOA, QRC, QKM, QPE, ZNE
# ═══════════════════════════════════════════════════════════════════════════════

