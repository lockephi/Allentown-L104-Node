"""
L104 Search & Precognition Orchestrators — VQPU-Integrated Multi-Strategy Hub (v2.3)
═══════════════════════════════════════════════════════════════════════════════
Three orchestrators that compose all search algorithms and precognition
predictors into ensemble pipelines with three-engine scoring + VQPU dispatch.
All classical strategies and predictors receive the VQPUBridge for quantum-
enhanced scoring. Precognition is exercised alongside every search operation.

v2.3: Analytical Upgrade
  - PerformanceAnalyzer integration — every result now includes analytics
  - EnsembleSearchResult.analytics: Optional[PerformanceReport]
  - EnsemblePrecognitionResult.analytics: Optional[PerformanceReport]
  - ThreeEngineSearchPrecog.analyzer: PerformanceAnalyzer with RunHistory
  - RunHistory trend tracking across multiple calls
  - Performance grades (S/A/B/C/D/F) via PHI-scored composite
  - Bottleneck detection, efficiency ratios, quality correlations

v2.2: Window Process Render Timing
  - StrategyTiming tracks per-strategy/predictor elapsed_ms, circuit_ms, classical_ms
  - WindowRenderTiming tracks per-window render_ms in VQPUVariationalForecaster
  - EnsembleSearchResult includes per_strategy_timing list
  - EnsemblePrecognitionResult includes per_predictor_timing + window_render_timings
  - Metadata includes slowest/fastest strategy, VQPU vs classical totals

  SearchOrchestrator       — Multi-strategy search with ranked fusion
  PrecognitionOrchestrator — Multi-predictor ensemble forecasting
  ThreeEngineSearchPrecog  — Unified three-engine + VQPU integration hub

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# ── Constants ──
PHI = (1 + math.sqrt(5)) / 2
PHI_CONJUGATE = (math.sqrt(5) - 1) / 2
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))
VOID_CONSTANT = 1.04 + PHI / 1000

from .search_algorithms import (
    SearchResult,
    SearchStrategy,
    QuantumGroverSearch,
    EntropyGuidedSearch,
    HyperdimensionalSearch,
    CoherenceFieldSearch,
    HarmonicResonanceSearch,
    ManifoldGeodesicSearch,
    SacredAlignmentSearch,
    VQPUGroverSearch,
    VQPUDatabaseSearch,
    QuantumReservoirSearch,
)

from .precognition import (
    PrecognitionResult,
    EntropyCascadePredictor,
    CoherenceEvolutionOracle,
    WaveInterferenceForecaster,
    HyperdimensionalPredictor,
    PhiConvergenceOracle,
    ManifoldFlowPredictor,
    QuantumReservoirPredictor,
    VQPUVariationalForecaster,
)

from .analytics import (
    PerformanceAnalyzer,
    PerformanceReport,
    RunHistory,
    StatisticalSummary,
    BottleneckInfo,
    EfficiencyRatio,
    WindowAnalytics,
    QualityCorrelation,
    TrendPoint,
    TrendAnalysis,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyTiming:
    """Per-strategy rendering/processing time breakdown."""
    strategy_name: str
    elapsed_ms: float
    is_vqpu: bool = False
    circuit_ms: float = 0.0      # VQPU circuit execution time
    classical_ms: float = 0.0    # Classical computation time
    vqpu_overhead_ms: float = 0.0  # VQPU enhancement overhead


@dataclass
class WindowRenderTiming:
    """Per-window rendering time for windowed processes."""
    window_index: int
    window_size: int
    render_ms: float
    circuit_ms: float = 0.0
    cost_eval_ms: float = 0.0


@dataclass
class EnsembleSearchResult:
    """Aggregated result from multi-strategy search."""
    query: Any
    strategy_results: Dict[str, SearchResult]
    fused_matches: List[Dict[str, Any]]
    best_strategy: str
    best_score: float
    sacred_alignment: float
    total_elapsed_ms: float
    vqpu_used: bool = False
    three_engine_score: float = 0.0
    per_strategy_timing: List[StrategyTiming] = field(default_factory=list)
    analytics: Optional[PerformanceReport] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsemblePrecognitionResult:
    """Aggregated result from multi-predictor ensemble."""
    input_summary: str
    predictor_results: Dict[str, PrecognitionResult]
    consensus_forecast: List[float]
    consensus_trend: str
    best_predictor: str
    best_confidence: float
    sacred_alignment: float
    total_elapsed_ms: float
    vqpu_used: bool = False
    three_engine_score: float = 0.0
    per_predictor_timing: List[StrategyTiming] = field(default_factory=list)
    window_render_timings: List[WindowRenderTiming] = field(default_factory=list)
    analytics: Optional[PerformanceReport] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
#  SEARCH ORCHESTRATOR — Multi-strategy ranked fusion
# ═══════════════════════════════════════════════════════════════════════════════

class SearchOrchestrator:
    """
    Run multiple search strategies in parallel (or sequence) and fuse
    results via PHI-weighted rank aggregation.

    Strategies are selected based on query type:
      - Structured list:  Grover, Entropy, Hyperdimensional
      - Pattern matching: Coherence, Harmonic, Manifold
      - Sacred scoring:   SacredAlignment
      - VQPU-accelerated: VQPUGrover, VQPUDatabase, QuantumReservoir

    Three-engine scoring enriches final results with entropy reversal,
    harmonic resonance, and wave coherence metrics.
    """

    def __init__(self, bridge=None, enable_vqpu: bool = True, enable_analytics: bool = True):
        """
        Args:
            bridge: Optional VQPUBridge instance for VQPU-accelerated strategies
            enable_vqpu: Whether to enable VQPU strategies (requires bridge or auto-create)
            enable_analytics: Whether to auto-generate PerformanceReport on each result
        """
        self._bridge = bridge
        self._enable_vqpu = enable_vqpu
        self._enable_analytics = enable_analytics

        # Classical strategies (always available, VQPU-enhanced via bridge)
        self._grover = QuantumGroverSearch(bridge=bridge)
        self._entropy = EntropyGuidedSearch(bridge=bridge)
        self._hd = HyperdimensionalSearch(bridge=bridge)
        self._coherence = CoherenceFieldSearch(bridge=bridge)
        self._harmonic = HarmonicResonanceSearch(bridge=bridge)
        self._manifold = ManifoldGeodesicSearch(bridge=bridge)
        self._sacred = SacredAlignmentSearch(bridge=bridge)

        # VQPU strategies (lazy-loaded)
        self._vqpu_grover = None
        self._vqpu_db = None
        self._reservoir = None

    def _get_vqpu_grover(self) -> Optional[VQPUGroverSearch]:
        if not self._enable_vqpu:
            return None
        if self._vqpu_grover is None:
            self._vqpu_grover = VQPUGroverSearch(bridge=self._bridge)
        return self._vqpu_grover

    def _get_vqpu_db(self) -> Optional[VQPUDatabaseSearch]:
        if not self._enable_vqpu:
            return None
        if self._vqpu_db is None:
            self._vqpu_db = VQPUDatabaseSearch(bridge=self._bridge)
        return self._vqpu_db

    def _get_reservoir(self) -> Optional[QuantumReservoirSearch]:
        if self._reservoir is None:
            self._reservoir = QuantumReservoirSearch(bridge=self._bridge)
        return self._reservoir

    def search_list(
        self,
        items: List[Any],
        oracle: Callable[[Any], bool],
        strategies: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> EnsembleSearchResult:
        """
        Search a list of items across multiple strategies.

        Args:
            items: Items to search
            oracle: Boolean predicate identifying matches
            strategies: List of strategy names (None = all applicable)
            top_k: Max matches per strategy
        """
        t0 = time.perf_counter()
        results: Dict[str, SearchResult] = {}
        timings: List[StrategyTiming] = []

        # Default strategies for list search — always include VQPU
        if strategies is None:
            strategies = ["quantum_grover", "entropy_guided", "sacred_alignment",
                          "vqpu_grover", "quantum_reservoir"]

        _VQPU_STRATS = {"vqpu_grover", "vqpu_database", "quantum_reservoir"}

        for strat in strategies:
            st0 = time.perf_counter()
            try:
                if strat == "quantum_grover":
                    results[strat] = self._grover.search(items, oracle)
                elif strat == "entropy_guided":
                    score_fn = lambda item: 0.0 if oracle(item) else 1.0
                    results[strat] = self._entropy.search(items, score_fn)
                elif strat == "sacred_alignment":
                    value_fn = lambda item: hash(str(item)) % 1000 / 1000.0
                    results[strat] = self._sacred.search(items, value_fn)
                elif strat == "vqpu_grover":
                    searcher = self._get_vqpu_grover()
                    if searcher:
                        results[strat] = searcher.search(items, oracle)
                elif strat == "quantum_reservoir":
                    searcher = self._get_reservoir()
                    if searcher:
                        results[strat] = searcher.search(
                            items[0] if items else "query", items
                        )
            except Exception:
                continue
            finally:
                strat_ms = (time.perf_counter() - st0) * 1000
                is_vqpu = strat in _VQPU_STRATS
                # Extract VQPU circuit time from result metadata if available
                circuit_ms = 0.0
                if strat in results:
                    circuit_ms = results[strat].metadata.get("vqpu_circuit_ms", 0.0)
                timings.append(StrategyTiming(
                    strategy_name=strat,
                    elapsed_ms=strat_ms,
                    is_vqpu=is_vqpu,
                    circuit_ms=circuit_ms,
                    classical_ms=strat_ms - circuit_ms,
                    vqpu_overhead_ms=results[strat].metadata.get("vqpu_overhead_ms", 0.0) if strat in results else 0.0,
                ))

        return self._fuse_results(None, results, t0, timings)

    def search_database(
        self,
        query: str,
        db: str = "all",
        max_results: int = 50,
    ) -> EnsembleSearchResult:
        """
        Quantum-accelerated search across L104 databases.
        Uses VQPUDatabaseSearch for Grover, QPE, and QFT.
        """
        t0 = time.perf_counter()
        results: Dict[str, SearchResult] = {}
        timings: List[StrategyTiming] = []

        # VQPU database search
        db_searcher = self._get_vqpu_db()
        if db_searcher:
            for strat_name, fn in [
                ("vqpu_database", lambda s: s.search(query, db=db, max_results=max_results)),
                ("vqpu_qpe_patterns", lambda s: s.discover_patterns(db=db)),
                ("vqpu_qft_frequency", lambda s: s.frequency_analysis(db=db)),
            ]:
                st0 = time.perf_counter()
                try:
                    results[strat_name] = fn(db_searcher)
                except Exception:
                    pass
                finally:
                    strat_ms = (time.perf_counter() - st0) * 1000
                    circuit_ms = results[strat_name].metadata.get("vqpu_circuit_ms", 0.0) if strat_name in results else 0.0
                    timings.append(StrategyTiming(
                        strategy_name=strat_name,
                        elapsed_ms=strat_ms,
                        is_vqpu=True,
                        circuit_ms=circuit_ms,
                        classical_ms=strat_ms - circuit_ms,
                    ))

        return self._fuse_results(query, results, t0, timings)

    def search_by_similarity(
        self,
        query: Any,
        items: List[Any],
        strategies: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> EnsembleSearchResult:
        """
        Similarity search across multiple strategies.
        """
        t0 = time.perf_counter()
        results: Dict[str, SearchResult] = {}
        timings: List[StrategyTiming] = []

        if strategies is None:
            strategies = ["hyperdimensional", "harmonic_resonance",
                          "manifold_geodesic", "quantum_reservoir"]

        _VQPU_STRATS = {"quantum_reservoir"}

        for strat in strategies:
            st0 = time.perf_counter()
            try:
                if strat == "hyperdimensional":
                    results[strat] = self._hd.search(query, items, top_k=top_k)
                elif strat == "harmonic_resonance":
                    results[strat] = self._harmonic.search(query, items, top_k=top_k)
                elif strat == "manifold_geodesic":
                    results[strat] = self._manifold.search(query, items, top_k=top_k)
                elif strat == "quantum_reservoir":
                    searcher = self._get_reservoir()
                    if searcher:
                        results[strat] = searcher.search(query, items, top_k=top_k)
            except Exception:
                continue
            finally:
                strat_ms = (time.perf_counter() - st0) * 1000
                is_vqpu = strat in _VQPU_STRATS
                circuit_ms = results[strat].metadata.get("vqpu_circuit_ms", 0.0) if strat in results else 0.0
                timings.append(StrategyTiming(
                    strategy_name=strat,
                    elapsed_ms=strat_ms,
                    is_vqpu=is_vqpu,
                    circuit_ms=circuit_ms,
                    classical_ms=strat_ms - circuit_ms,
                ))

        return self._fuse_results(query, results, t0, timings)

    def _fuse_results(
        self,
        query: Any,
        results: Dict[str, SearchResult],
        t0: float,
        timings: Optional[List[StrategyTiming]] = None,
    ) -> EnsembleSearchResult:
        """PHI-weighted rank fusion of multi-strategy results."""
        if timings is None:
            timings = []
        if not results:
            return EnsembleSearchResult(
                query=query, strategy_results={}, fused_matches=[],
                best_strategy="none", best_score=0.0, sacred_alignment=0.0,
                total_elapsed_ms=(time.perf_counter() - t0) * 1000,
                per_strategy_timing=timings,
            )

        # Find best strategy
        best_strat = max(results, key=lambda k: results[k].score)
        best_result = results[best_strat]

        # Fuse matches: collect all matches, deduplicate, PHI-weight by rank
        all_matches: Dict[str, Dict] = {}
        for strat_name, result in results.items():
            for rank, match in enumerate(result.matches):
                key = str(match.get("index", match.get("item", id(match))))
                if key not in all_matches:
                    all_matches[key] = {
                        **match,
                        "fusion_score": 0.0,
                        "strategies": [],
                    }
                # PHI-weighted score: higher rank → more weight
                rank_weight = PHI_CONJUGATE ** rank
                strat_weight = result.score
                all_matches[key]["fusion_score"] += rank_weight * strat_weight
                all_matches[key]["strategies"].append(strat_name)

        fused = sorted(all_matches.values(), key=lambda m: m["fusion_score"], reverse=True)

        # Check if VQPU was used
        vqpu_used = any(
            r.metadata.get("vqpu", False) for r in results.values()
        )

        # Three-engine scoring
        three_engine = self._three_engine_score(results)

        # Sacred alignment: average across strategies
        avg_sa = sum(r.sacred_alignment for r in results.values()) / max(len(results), 1)

        elapsed = (time.perf_counter() - t0) * 1000

        ensemble = EnsembleSearchResult(
            query=query,
            strategy_results=results,
            fused_matches=fused[:20],
            best_strategy=best_strat,
            best_score=best_result.score,
            sacred_alignment=avg_sa,
            total_elapsed_ms=elapsed,
            vqpu_used=vqpu_used,
            three_engine_score=three_engine,
            per_strategy_timing=timings,
            metadata={
                "strategies_run": list(results.keys()),
                "n_fused_matches": len(fused),
                "vqpu_total_ms": sum(t.circuit_ms for t in timings),
                "classical_total_ms": sum(t.classical_ms for t in timings),
                "slowest_strategy": max(timings, key=lambda t: t.elapsed_ms).strategy_name if timings else "none",
                "fastest_strategy": min(timings, key=lambda t: t.elapsed_ms).strategy_name if timings else "none",
            },
        )

        # Auto-attach analytics when enabled
        if self._enable_analytics and timings:
            try:
                ensemble.analytics = PerformanceAnalyzer.analyze_search(ensemble)
            except Exception:
                pass

        return ensemble

    def _three_engine_score(self, results: Dict[str, SearchResult]) -> float:
        """Compute three-engine composite score from search results."""
        try:
            from l104_vqpu_bridge import ThreeEngineQuantumScorer
        except ImportError:
            return 0.5

        scores = []
        for r in results.values():
            # Entropy from search
            entropy_score = ThreeEngineQuantumScorer.entropy_score(
                abs(r.entropy_delta) if r.entropy_delta else 1.0
            )
            # Harmonic score
            harmonic_score = ThreeEngineQuantumScorer.harmonic_score()
            # Wave score
            wave_score = ThreeEngineQuantumScorer.wave_score()
            composite = 0.35 * entropy_score + 0.40 * harmonic_score + 0.25 * wave_score
            scores.append(composite)

        return sum(scores) / max(len(scores), 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  PRECOGNITION ORCHESTRATOR — Multi-predictor ensemble forecasting
# ═══════════════════════════════════════════════════════════════════════════════

class PrecognitionOrchestrator:
    """
    Run multiple precognition predictors and fuse their forecasts into
    a consensus prediction via PHI-weighted ensemble averaging.

    Predictors:
      Classical: EntropyCascade, CoherenceEvolution, WaveInterference,
                 Hyperdimensional, PhiConvergence, ManifoldFlow
      VQPU:     QuantumReservoir, VQPUVariational

    Consensus method: PHI-weighted average of predictor forecasts,
    where weights are proportional to each predictor's confidence.
    """

    def __init__(self, bridge=None, enable_vqpu: bool = True, enable_analytics: bool = True):
        self._bridge = bridge
        self._enable_vqpu = enable_vqpu
        self._enable_analytics = enable_analytics

        # Classical predictors (VQPU-enhanced via bridge)
        self._entropy = EntropyCascadePredictor(bridge=bridge)
        self._coherence = CoherenceEvolutionOracle(bridge=bridge)
        self._wave = WaveInterferenceForecaster(bridge=bridge)
        self._hd = HyperdimensionalPredictor(bridge=bridge)
        self._phi = PhiConvergenceOracle(bridge=bridge)
        self._manifold = ManifoldFlowPredictor(bridge=bridge)

        # VQPU predictors (lazy-loaded)
        self._reservoir = None
        self._variational = None

    def _get_reservoir(self) -> Optional[QuantumReservoirPredictor]:
        if not self._enable_vqpu:
            return None
        if self._reservoir is None:
            self._reservoir = QuantumReservoirPredictor(bridge=self._bridge)
        return self._reservoir

    def _get_variational(self) -> Optional[VQPUVariationalForecaster]:
        if not self._enable_vqpu:
            return None
        if self._variational is None:
            self._variational = VQPUVariationalForecaster(bridge=self._bridge)
        return self._variational

    def predict(
        self,
        history: List[float],
        horizon: int = 26,
        predictors: Optional[List[str]] = None,
    ) -> EnsemblePrecognitionResult:
        """
        Run multiple predictors on the same history and fuse results.

        Args:
            history: Time series data
            horizon: Steps to forecast
            predictors: List of predictor names (None = all available)
        """
        t0 = time.perf_counter()
        results: Dict[str, PrecognitionResult] = {}
        timings: List[StrategyTiming] = []

        if predictors is None:
            predictors = [
                "entropy_cascade", "coherence_evolution", "wave_interference",
                "hyperdimensional", "phi_convergence", "manifold_flow",
                "quantum_reservoir", "vqpu_variational",
            ]

        _VQPU_PREDS = {"quantum_reservoir", "vqpu_variational"}

        for pred_name in predictors:
            pt0 = time.perf_counter()
            try:
                if pred_name == "entropy_cascade":
                    results[pred_name] = self._entropy.predict(history, horizon)
                elif pred_name == "coherence_evolution":
                    results[pred_name] = self._coherence.predict(history, horizon)
                elif pred_name == "wave_interference":
                    results[pred_name] = self._wave.predict(history, horizon)
                elif pred_name == "hyperdimensional":
                    results[pred_name] = self._hd.predict(history, horizon)
                elif pred_name == "phi_convergence":
                    results[pred_name] = self._phi.predict(history, horizon)
                elif pred_name == "manifold_flow":
                    results[pred_name] = self._manifold.predict(history, horizon)
                elif pred_name == "quantum_reservoir":
                    predictor = self._get_reservoir()
                    if predictor:
                        results[pred_name] = predictor.predict(history, horizon)
                elif pred_name == "vqpu_variational":
                    predictor = self._get_variational()
                    if predictor:
                        results[pred_name] = predictor.predict(history, horizon)
            except Exception:
                continue
            finally:
                pred_ms = (time.perf_counter() - pt0) * 1000
                is_vqpu = pred_name in _VQPU_PREDS
                circuit_ms = results[pred_name].metadata.get("vqpu_circuit_ms", 0.0) if pred_name in results else 0.0
                timings.append(StrategyTiming(
                    strategy_name=pred_name,
                    elapsed_ms=pred_ms,
                    is_vqpu=is_vqpu,
                    circuit_ms=circuit_ms,
                    classical_ms=pred_ms - circuit_ms,
                    vqpu_overhead_ms=results[pred_name].metadata.get("vqpu_overhead_ms", 0.0) if pred_name in results else 0.0,
                ))

        return self._fuse_predictions(history, results, horizon, t0, timings)

    def _fuse_predictions(
        self,
        history: List[float],
        results: Dict[str, PrecognitionResult],
        horizon: int,
        t0: float,
        timings: Optional[List[StrategyTiming]] = None,
    ) -> EnsemblePrecognitionResult:
        """PHI-weighted ensemble fusion of predictor forecasts."""
        if timings is None:
            timings = []
        # Extract window render timings from VQPUVariational results
        window_timings: List[WindowRenderTiming] = []
        for pred_name, result in (results or {}).items():
            wrt = result.metadata.get("window_render_timings", [])
            for wt in wrt:
                if isinstance(wt, dict):
                    window_timings.append(WindowRenderTiming(**wt))
                elif isinstance(wt, WindowRenderTiming):
                    window_timings.append(wt)

        if not results:
            return EnsemblePrecognitionResult(
                input_summary=f"{len(history)} points",
                predictor_results={}, consensus_forecast=[],
                consensus_trend="stable", best_predictor="none",
                best_confidence=0.0, sacred_alignment=0.0,
                total_elapsed_ms=(time.perf_counter() - t0) * 1000,
                per_predictor_timing=timings,
                window_render_timings=window_timings,
            )

        # Find best predictor
        best_pred = max(results, key=lambda k: results[k].confidence)
        best_result = results[best_pred]

        # Consensus forecast: PHI-weighted average across predictors
        consensus = [0.0] * horizon
        total_weight = 0.0

        for pred_name, result in results.items():
            if not result.forecast:
                continue
            weight = result.confidence * PHI_CONJUGATE
            # Bonus weight for VQPU predictors
            if result.metadata.get("vqpu", False):
                weight *= PHI
            for i, fp in enumerate(result.forecast):
                if i < horizon:
                    consensus[i] += fp.value * weight
            total_weight += weight

        if total_weight > 0:
            consensus = [v / total_weight for v in consensus]

        # Detect consensus trend
        consensus_trend = self._detect_trend(consensus)

        # VQPU usage
        vqpu_used = any(
            r.metadata.get("vqpu", False) for r in results.values()
        )

        # Three-engine scoring
        three_engine = self._three_engine_score(results)

        # Sacred alignment average
        avg_sa = sum(r.sacred_alignment for r in results.values()) / max(len(results), 1)

        elapsed = (time.perf_counter() - t0) * 1000

        ensemble = EnsemblePrecognitionResult(
            input_summary=f"{len(history)} points, {len(results)} predictors",
            predictor_results=results,
            consensus_forecast=consensus,
            consensus_trend=consensus_trend,
            best_predictor=best_pred,
            best_confidence=best_result.confidence,
            sacred_alignment=avg_sa,
            total_elapsed_ms=elapsed,
            vqpu_used=vqpu_used,
            three_engine_score=three_engine,
            per_predictor_timing=timings,
            window_render_timings=window_timings,
            metadata={
                "predictors_run": list(results.keys()),
                "horizon": horizon,
                "vqpu_total_ms": sum(t.circuit_ms for t in timings),
                "classical_total_ms": sum(t.classical_ms for t in timings),
                "slowest_predictor": max(timings, key=lambda t: t.elapsed_ms).strategy_name if timings else "none",
                "fastest_predictor": min(timings, key=lambda t: t.elapsed_ms).strategy_name if timings else "none",
                "total_window_render_ms": sum(wt.render_ms for wt in window_timings),
                "n_windows_rendered": len(window_timings),
            },
        )

        # Auto-attach analytics when enabled
        if self._enable_analytics and timings:
            try:
                ensemble.analytics = PerformanceAnalyzer.analyze_precognition(ensemble)
            except Exception:
                pass

        return ensemble

    def _detect_trend(self, values: List[float]) -> str:
        """Detect trend in consensus forecast."""
        if len(values) < 3:
            return "stable"
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        avg_diff = sum(diffs) / len(diffs)
        sign_changes = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0)
        if sign_changes > len(diffs) * 0.6:
            return "oscillating"
        if abs(avg_diff) < 0.001:
            return "stable"
        abs_diffs = [abs(d) for d in diffs]
        if all(abs_diffs[i] >= abs_diffs[i + 1] for i in range(len(abs_diffs) - 1)):
            return "converging"
        if all(abs_diffs[i] <= abs_diffs[i + 1] for i in range(len(abs_diffs) - 1)):
            return "diverging"
        return "oscillating"

    def _three_engine_score(self, results: Dict[str, PrecognitionResult]) -> float:
        """Compute three-engine score from precognition results."""
        try:
            from l104_vqpu_bridge import ThreeEngineQuantumScorer
        except ImportError:
            return 0.5

        entropies = []
        for r in results.values():
            if r.entropy_trajectory:
                entropies.extend(r.entropy_trajectory)

        avg_entropy = sum(entropies) / max(len(entropies), 1) if entropies else 1.0
        entropy_score = ThreeEngineQuantumScorer.entropy_score(avg_entropy)
        harmonic_score = ThreeEngineQuantumScorer.harmonic_score()
        wave_score = ThreeEngineQuantumScorer.wave_score()

        return 0.35 * entropy_score + 0.40 * harmonic_score + 0.25 * wave_score


# ═══════════════════════════════════════════════════════════════════════════════
#  THREE-ENGINE SEARCH PRECOG — Unified integration hub
# ═══════════════════════════════════════════════════════════════════════════════

class ThreeEngineSearchPrecog:
    """
    Unified hub combining SearchOrchestrator + PrecognitionOrchestrator with
    full three-engine (Science + Math + Code) and VQPU integration.

    Provides high-level methods for:
      - search():     Multi-strategy search with three-engine scoring
      - predict():    Multi-predictor forecasting with VQPU circuits
      - research():   VQPU database research pipeline (Grover + QPE + QFT)
      - analyze():    Combined search + predict for pattern analysis

    v2.3: Built-in PerformanceAnalyzer + RunHistory for trend tracking.
      - analyzer: PerformanceAnalyzer with cross-run RunHistory
      - Every result auto-includes PerformanceReport (analytics field)
      - summary_text() for human-readable analytics printout
      - compare_runs() for A/B testing and regression detection

    Each operation is enriched with:
      - Entropy:  Science Engine Maxwell Demon reversal efficiency
      - Harmonic: Math Engine GOD_CODE alignment + wave coherence
      - Wave:     Math Engine PHI-harmonic phase-lock
      - VQPU:     Metal GPU quantum circuit execution (when available)
    """

    def __init__(self, bridge=None, enable_vqpu: bool = True, enable_analytics: bool = True):
        self._bridge = bridge
        self._enable_analytics = enable_analytics
        self._history = RunHistory(max_runs=200)
        self._analyzer = PerformanceAnalyzer(history=self._history)
        self._search = SearchOrchestrator(
            bridge=bridge, enable_vqpu=enable_vqpu, enable_analytics=enable_analytics,
        )
        self._precog = PrecognitionOrchestrator(
            bridge=bridge, enable_vqpu=enable_vqpu, enable_analytics=enable_analytics,
        )

    @property
    def search_orchestrator(self) -> SearchOrchestrator:
        return self._search

    @property
    def precognition_orchestrator(self) -> PrecognitionOrchestrator:
        return self._precog

    @property
    def analyzer(self) -> PerformanceAnalyzer:
        return self._analyzer

    @property
    def history(self) -> RunHistory:
        return self._history

    def search(
        self,
        items: List[Any],
        oracle: Callable[[Any], bool],
        strategies: Optional[List[str]] = None,
    ) -> EnsembleSearchResult:
        """Multi-strategy search with three-engine scoring + analytics."""
        result = self._search.search_list(items, oracle, strategies)
        if self._enable_analytics and result.analytics:
            self._history.record_from_report(result.analytics)
        return result

    def predict(
        self,
        history: List[float],
        horizon: int = 26,
        predictors: Optional[List[str]] = None,
    ) -> EnsemblePrecognitionResult:
        """Multi-predictor ensemble forecasting + analytics."""
        result = self._precog.predict(history, horizon, predictors)
        if self._enable_analytics and result.analytics:
            self._history.record_from_report(result.analytics)
        return result

    def research(
        self,
        query: str,
        db: str = "all",
        max_results: int = 50,
    ) -> EnsembleSearchResult:
        """VQPU quantum database research pipeline + analytics."""
        result = self._search.search_database(query, db=db, max_results=max_results)
        if self._enable_analytics and result.analytics:
            self._history.record_from_report(result.analytics)
        return result

    def similarity_search(
        self,
        query: Any,
        items: List[Any],
        strategies: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> EnsembleSearchResult:
        """Multi-strategy similarity search + analytics."""
        result = self._search.search_by_similarity(query, items, strategies, top_k)
        if self._enable_analytics and result.analytics:
            self._history.record_from_report(result.analytics)
        return result

    def analyze(
        self,
        time_series: List[float],
        horizon: int = 26,
        search_items: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Combined search + predict for comprehensive pattern analysis.

        Runs precognition on the time series, and optionally searches
        for similar patterns in a reference dataset.
        """
        t0 = time.perf_counter()

        # Precognition
        precog_result = self.predict(time_series, horizon)

        # Optional similarity search on items dataset
        search_result = None
        if search_items:
            search_result = self.similarity_search(
                time_series[-1],  # Use latest value as query
                search_items,
            )

        elapsed = (time.perf_counter() - t0) * 1000

        return {
            "precognition": precog_result,
            "search": search_result,
            "consensus_forecast": precog_result.consensus_forecast,
            "consensus_trend": precog_result.consensus_trend,
            "best_predictor": precog_result.best_predictor,
            "best_confidence": precog_result.best_confidence,
            "three_engine_score": precog_result.three_engine_score,
            "vqpu_used": precog_result.vqpu_used,
            "sacred_alignment": precog_result.sacred_alignment,
            "total_elapsed_ms": elapsed,
        }

    def search_with_precognition(
        self,
        items: List[Any],
        oracle: Callable[[Any], bool],
        value_fn: Callable[[Any], float],
        horizon: int = 13,
    ) -> Dict[str, Any]:
        """
        Search + Precognition exercised together: search the items, extract
        numeric values from matches, then run full precognition on the result
        distribution to predict future match patterns.

        This ensures both VQPU search and VQPU precognition are fully
        exercised on every call.
        """
        import math as _m

        # Phase 1: Multi-strategy VQPU-enhanced search
        search_result = self.search(items, oracle)

        # Phase 2: Extract time series from fused results for precognition
        values = []
        for m in search_result.fused_matches:
            if "item" in m:
                try:
                    values.append(value_fn(m["item"]))
                except Exception:
                    values.append(m.get("score", 0.0))
            else:
                values.append(m.get("fusion_score", m.get("score", 0.0)))

        # If insufficient data, synthesize from search score distribution
        if len(values) < 3:
            _phi_c = ((_m.sqrt(5) - 1) / 2)
            values = [search_result.best_score * _phi_c ** i for i in range(10)]

        # Phase 3: Run full VQPU-powered precognition ensemble
        precog_result = self.predict(values, horizon)

        # Phase 4: Cross-correlate search + precognition
        combined_vqpu = search_result.vqpu_used or precog_result.vqpu_used
        combined_three_engine = (
            search_result.three_engine_score * 0.5
            + precog_result.three_engine_score * 0.5
        )

        return {
            "search": search_result,
            "precognition": precog_result,
            "search_matches": len(search_result.fused_matches),
            "search_best_strategy": search_result.best_strategy,
            "search_best_score": search_result.best_score,
            "precog_trend": precog_result.consensus_trend,
            "precog_confidence": precog_result.best_confidence,
            "precog_best_predictor": precog_result.best_predictor,
            "precog_forecast_horizon": horizon,
            "consensus_forecast": precog_result.consensus_forecast,
            "vqpu_used": combined_vqpu,
            "three_engine_score": combined_three_engine,
            "sacred_alignment": (
                search_result.sacred_alignment * 0.5
                + precog_result.sacred_alignment * 0.5
            ),
        }

    def status(self) -> Dict[str, Any]:
        """Report status of all engines, VQPU connection, and analytics."""
        vqpu_active = False
        if self._bridge is not None:
            vqpu_active = getattr(self._bridge, '_active', False)

        # Check three-engine availability
        engines = {"science": False, "math": False, "code": False}
        try:
            from l104_vqpu_bridge import ThreeEngineQuantumScorer
            engines["science"] = ThreeEngineQuantumScorer._get_science() is not None
            engines["math"] = ThreeEngineQuantumScorer._get_math() is not None
            engines["code"] = ThreeEngineQuantumScorer._get_code() is not None
        except ImportError:
            pass

        # Trend from history
        trend = self._history.analyze() if self._history.n_runs >= 2 else None

        return {
            "vqpu_active": vqpu_active,
            "vqpu_bridge": self._bridge is not None,
            "engines": engines,
            "analytics_enabled": self._enable_analytics,
            "history_runs": self._history.n_runs,
            "trend": trend.trend_direction if trend else "insufficient_data",
            "search_strategies": [s.value for s in SearchStrategy],
            "precognition_predictors": [
                "entropy_cascade", "coherence_evolution", "wave_interference",
                "hyperdimensional", "phi_convergence", "manifold_flow",
                "quantum_reservoir", "vqpu_variational",
            ],
            "god_code": GOD_CODE,
        }

    def trend_report(self) -> Optional[TrendAnalysis]:
        """Get cross-run performance trend analysis."""
        return self._history.analyze()

    @staticmethod
    def summary_text(result) -> str:
        """Generate human-readable analytics summary for any result with analytics."""
        report = getattr(result, "analytics", None)
        if report is None:
            return "(no analytics attached)"
        return PerformanceAnalyzer.summary_text(report)
