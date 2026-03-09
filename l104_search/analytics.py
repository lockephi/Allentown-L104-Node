"""
L104 Search & Precognition Analytics Engine (v2.3)
═══════════════════════════════════════════════════════════════════════════════
Deep analytical layer for timing, performance, bottleneck detection,
efficiency ratios, score-quality correlation, and run-history trend tracking
across all search strategies and precognition predictors.

Builds on v2.2 StrategyTiming and WindowRenderTiming data to produce
PerformanceReport with actionable insights.

Classes:
  PerformanceAnalyzer    — Core analytics engine (stateless per-call analysis)
  RunHistory             — Cross-run accumulator for trend detection
  PerformanceReport      — Full analytical report dataclass

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Constants ──
PHI = (1 + math.sqrt(5)) / 2
PHI_CONJUGATE = (math.sqrt(5) - 1) / 2
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))
VOID_CONSTANT = 1.04 + PHI / 1000


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StatisticalSummary:
    """Statistical breakdown of a timing distribution."""
    count: int
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float
    total_ms: float
    cv: float               # Coefficient of variation (std/mean)

    @property
    def spread_ratio(self) -> float:
        """Ratio of max to min — higher means more variable."""
        return self.max_ms / self.min_ms if self.min_ms > 0 else float("inf")


@dataclass
class BottleneckInfo:
    """Description of a detected performance bottleneck."""
    component: str           # strategy/predictor/window name
    elapsed_ms: float
    fraction_of_total: float # 0.0–1.0
    severity: str            # "critical" | "high" | "moderate" | "low"
    suggestion: str          # Actionable recommendation


@dataclass
class EfficiencyRatio:
    """VQPU vs classical efficiency metrics for a strategy/predictor."""
    name: str
    total_ms: float
    vqpu_circuit_ms: float
    classical_ms: float
    vqpu_overhead_ms: float
    vqpu_fraction: float        # circuit_ms / total_ms
    overhead_fraction: float    # overhead / total_ms
    useful_work_fraction: float # (total - overhead) / total
    score: float                # result score
    score_per_ms: float         # score / total_ms  (higher = better)
    sacred_alignment: float
    alignment_per_ms: float     # sacred_alignment / total_ms


@dataclass
class WindowAnalytics:
    """Analytics over windowed rendering (VQPUVariationalForecaster)."""
    n_windows: int
    total_render_ms: float
    avg_render_ms: float
    median_render_ms: float
    std_render_ms: float
    min_render_ms: float
    max_render_ms: float
    p95_render_ms: float
    throughput_windows_per_sec: float
    outlier_count: int          # Windows > p95
    outlier_indices: List[int]


@dataclass
class QualityCorrelation:
    """Correlation between time spent and result quality."""
    pearson_r: float            # Correlation coeff: time vs score
    fastest_score: float
    slowest_score: float
    best_score_strategy: str
    best_score_ms: float
    best_efficiency_strategy: str  # Highest score/ms
    best_efficiency_ratio: float
    diminishing_returns: bool   # True if spending more time doesn't help


@dataclass
class TrendPoint:
    """A single data point in run history."""
    timestamp: float
    total_ms: float
    vqpu_ms: float
    classical_ms: float
    best_score: float
    n_strategies: int
    sacred_alignment: float


@dataclass
class TrendAnalysis:
    """Analysis of performance trends across runs."""
    n_runs: int
    avg_total_ms: float
    trend_direction: str       # "improving" | "degrading" | "stable" | "volatile"
    ms_per_run_slope: float    # Positive = getting slower
    score_per_run_slope: float # Positive = getting better
    vqpu_utilization_trend: str # "increasing" | "decreasing" | "stable"
    last_3_avg_ms: float
    first_3_avg_ms: float
    performance_change_pct: float  # (last3 - first3) / first3 * 100


@dataclass
class PerformanceReport:
    """Full analytical report from PerformanceAnalyzer."""
    # Timing statistics
    strategy_stats: StatisticalSummary
    vqpu_stats: Optional[StatisticalSummary]
    classical_stats: Optional[StatisticalSummary]

    # Bottleneck analysis
    bottlenecks: List[BottleneckInfo]
    dominant_bottleneck: Optional[BottleneckInfo]

    # Efficiency ratios per strategy
    efficiency_ratios: List[EfficiencyRatio]

    # Window analytics (if applicable)
    window_analytics: Optional[WindowAnalytics]

    # Quality correlations
    quality_correlation: Optional[QualityCorrelation]

    # Trend analysis (if history available)
    trend_analysis: Optional[TrendAnalysis]

    # Summary metrics
    total_elapsed_ms: float
    vqpu_total_ms: float
    classical_total_ms: float
    vqpu_fraction: float         # vqpu / total
    n_strategies: int
    n_vqpu_strategies: int
    n_classical_strategies: int
    sacred_alignment_avg: float

    # PHI-scored performance grade: 0.0–1.0
    performance_grade: float
    grade_label: str             # "S" | "A" | "B" | "C" | "D" | "F"

    generated_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
#  STATISTICAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _percentile(sorted_vals: List[float], pct: float) -> float:
    """Get percentile from a sorted list. pct in [0, 100]."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)


def _std_dev(values: List[float], mean: float) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _pearson_correlation(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient between xs and ys."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    if dx < 1e-15 or dy < 1e-15:
        return 0.0
    return num / (dx * dy)


def _linear_slope(values: List[float]) -> float:
    """Least-squares slope of values indexed by integer x."""
    n = len(values)
    if n < 2:
        return 0.0
    mx = (n - 1) / 2.0
    my = sum(values) / n
    num = sum((i - mx) * (values[i] - my) for i in range(n))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0


def _compute_stats(values: List[float]) -> StatisticalSummary:
    """Build a StatisticalSummary from a list of ms values."""
    if not values:
        return StatisticalSummary(
            count=0, mean_ms=0, median_ms=0, std_ms=0,
            min_ms=0, max_ms=0, p95_ms=0, p99_ms=0, total_ms=0, cv=0,
        )
    s = sorted(values)
    n = len(s)
    total = sum(s)
    mean = total / n
    std = _std_dev(s, mean)
    return StatisticalSummary(
        count=n,
        mean_ms=mean,
        median_ms=_percentile(s, 50),
        std_ms=std,
        min_ms=s[0],
        max_ms=s[-1],
        p95_ms=_percentile(s, 95),
        p99_ms=_percentile(s, 99),
        total_ms=total,
        cv=std / mean if mean > 0 else 0.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN HISTORY — Cross-run accumulator for trend detection
# ═══════════════════════════════════════════════════════════════════════════════

class RunHistory:
    """
    Accumulates performance data across multiple search/precognition runs
    for trend analysis. Thread-safe via append-only design.

    Usage:
        history = RunHistory(max_runs=200)
        # ... after each orchestrator call:
        history.record(report)      # From PerformanceAnalyzer
        trend = history.analyze()   # TrendAnalysis
    """

    def __init__(self, max_runs: int = 200):
        self._max_runs = max_runs
        self._runs: List[TrendPoint] = []

    @property
    def n_runs(self) -> int:
        return len(self._runs)

    def record_from_report(self, report: PerformanceReport):
        """Record a run from a PerformanceReport."""
        self._runs.append(TrendPoint(
            timestamp=report.generated_at or time.time(),
            total_ms=report.total_elapsed_ms,
            vqpu_ms=report.vqpu_total_ms,
            classical_ms=report.classical_total_ms,
            best_score=max((e.score for e in report.efficiency_ratios), default=0.0),
            n_strategies=report.n_strategies,
            sacred_alignment=report.sacred_alignment_avg,
        ))
        # Trim to max
        if len(self._runs) > self._max_runs:
            self._runs = self._runs[-self._max_runs:]

    def record(
        self,
        total_ms: float,
        vqpu_ms: float = 0.0,
        classical_ms: float = 0.0,
        best_score: float = 0.0,
        n_strategies: int = 0,
        sacred_alignment: float = 0.0,
    ):
        """Manually record a run data point."""
        self._runs.append(TrendPoint(
            timestamp=time.time(),
            total_ms=total_ms,
            vqpu_ms=vqpu_ms,
            classical_ms=classical_ms,
            best_score=best_score,
            n_strategies=n_strategies,
            sacred_alignment=sacred_alignment,
        ))
        if len(self._runs) > self._max_runs:
            self._runs = self._runs[-self._max_runs:]

    def analyze(self) -> Optional[TrendAnalysis]:
        """Analyze accumulated runs for trends."""
        if len(self._runs) < 2:
            return None

        totals = [r.total_ms for r in self._runs]
        scores = [r.best_score for r in self._runs]
        vqpu_fracs = [
            r.vqpu_ms / r.total_ms if r.total_ms > 0 else 0.0
            for r in self._runs
        ]

        avg_total = sum(totals) / len(totals)
        ms_slope = _linear_slope(totals)
        score_slope = _linear_slope(scores)
        vqpu_slope = _linear_slope(vqpu_fracs)

        # Recent vs early
        n3 = min(3, len(self._runs))
        last_3 = sum(r.total_ms for r in self._runs[-n3:]) / n3
        first_3 = sum(r.total_ms for r in self._runs[:n3]) / n3
        change_pct = (last_3 - first_3) / first_3 * 100 if first_3 > 0 else 0.0

        # Trend direction
        if abs(ms_slope) < 0.01 * avg_total:
            direction = "stable"
        elif ms_slope < 0:
            direction = "improving"
        else:
            direction = "degrading"

        # Check for high variance (volatile)
        std_total = _std_dev(totals, avg_total)
        if std_total > 0.5 * avg_total:
            direction = "volatile"

        # VQPU utilization trend
        if abs(vqpu_slope) < 0.001:
            vqpu_trend = "stable"
        elif vqpu_slope > 0:
            vqpu_trend = "increasing"
        else:
            vqpu_trend = "decreasing"

        return TrendAnalysis(
            n_runs=len(self._runs),
            avg_total_ms=avg_total,
            trend_direction=direction,
            ms_per_run_slope=ms_slope,
            score_per_run_slope=score_slope,
            vqpu_utilization_trend=vqpu_trend,
            last_3_avg_ms=last_3,
            first_3_avg_ms=first_3,
            performance_change_pct=change_pct,
        )

    def clear(self):
        """Clear all accumulated runs."""
        self._runs.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE ANALYZER — Core analytics engine
# ═══════════════════════════════════════════════════════════════════════════════

class PerformanceAnalyzer:
    """
    Comprehensive performance analytics for search and precognition operations.

    Takes raw StrategyTiming / WindowRenderTiming data (v2.2) and produces
    a PerformanceReport with statistical summaries, bottleneck detection,
    efficiency ratios, quality correlations, and optional trend analysis.

    Usage (search):
        result = search_orchestrator.search_list(items, oracle)
        report = PerformanceAnalyzer.analyze_search(result)

    Usage (precognition):
        result = precog_orchestrator.predict(history, horizon)
        report = PerformanceAnalyzer.analyze_precognition(result)

    Usage (with history):
        analyzer = PerformanceAnalyzer(history=run_history)
        report = analyzer.analyze_search(result)
        run_history.record_from_report(report)
    """

    def __init__(self, history: Optional[RunHistory] = None):
        self._history = history

    @staticmethod
    def analyze_search(result, history: Optional[RunHistory] = None) -> PerformanceReport:
        """
        Analyze an EnsembleSearchResult to produce a PerformanceReport.

        Args:
            result: EnsembleSearchResult from SearchOrchestrator
            history: Optional RunHistory for trend analysis
        """
        timings = result.per_strategy_timing or []
        strategy_results = result.strategy_results or {}

        # Build per-strategy info for efficiency + quality
        all_ms = [t.elapsed_ms for t in timings]
        vqpu_ms_list = [t.elapsed_ms for t in timings if t.is_vqpu]
        classical_ms_list = [t.elapsed_ms for t in timings if not t.is_vqpu]

        strategy_stats = _compute_stats(all_ms)
        vqpu_stats = _compute_stats(vqpu_ms_list) if vqpu_ms_list else None
        classical_stats = _compute_stats(classical_ms_list) if classical_ms_list else None

        # Efficiency ratios
        efficiency_ratios = []
        for t in timings:
            sr = strategy_results.get(t.strategy_name)
            score = sr.score if sr else 0.0
            sa = sr.sacred_alignment if sr else 0.0
            efficiency_ratios.append(EfficiencyRatio(
                name=t.strategy_name,
                total_ms=t.elapsed_ms,
                vqpu_circuit_ms=t.circuit_ms,
                classical_ms=t.classical_ms,
                vqpu_overhead_ms=t.vqpu_overhead_ms,
                vqpu_fraction=t.circuit_ms / t.elapsed_ms if t.elapsed_ms > 0 else 0.0,
                overhead_fraction=t.vqpu_overhead_ms / t.elapsed_ms if t.elapsed_ms > 0 else 0.0,
                useful_work_fraction=(t.elapsed_ms - t.vqpu_overhead_ms) / t.elapsed_ms if t.elapsed_ms > 0 else 1.0,
                score=score,
                score_per_ms=score / t.elapsed_ms if t.elapsed_ms > 0 else 0.0,
                sacred_alignment=sa,
                alignment_per_ms=sa / t.elapsed_ms if t.elapsed_ms > 0 else 0.0,
            ))

        # Bottleneck detection
        bottlenecks = PerformanceAnalyzer._detect_bottlenecks(
            timings, result.total_elapsed_ms,
        )

        # Quality correlation
        quality_corr = PerformanceAnalyzer._quality_correlation(
            timings, strategy_results,
        )

        # Trend analysis
        trend = None
        if history and history.n_runs >= 2:
            trend = history.analyze()

        # VQPU totals
        vqpu_total = sum(t.circuit_ms for t in timings)
        classical_total = sum(t.classical_ms for t in timings)
        n_vqpu = sum(1 for t in timings if t.is_vqpu)
        n_classical = len(timings) - n_vqpu
        vqpu_frac = vqpu_total / result.total_elapsed_ms if result.total_elapsed_ms > 0 else 0.0

        avg_sa = result.sacred_alignment

        # Performance grade (PHI-scored)
        grade, label = PerformanceAnalyzer._compute_grade(
            strategy_stats, efficiency_ratios, vqpu_frac, avg_sa,
        )

        report = PerformanceReport(
            strategy_stats=strategy_stats,
            vqpu_stats=vqpu_stats,
            classical_stats=classical_stats,
            bottlenecks=bottlenecks,
            dominant_bottleneck=bottlenecks[0] if bottlenecks else None,
            efficiency_ratios=efficiency_ratios,
            window_analytics=None,
            quality_correlation=quality_corr,
            trend_analysis=trend,
            total_elapsed_ms=result.total_elapsed_ms,
            vqpu_total_ms=vqpu_total,
            classical_total_ms=classical_total,
            vqpu_fraction=vqpu_frac,
            n_strategies=len(timings),
            n_vqpu_strategies=n_vqpu,
            n_classical_strategies=n_classical,
            sacred_alignment_avg=avg_sa,
            performance_grade=grade,
            grade_label=label,
            generated_at=time.time(),
        )

        return report

    @staticmethod
    def analyze_precognition(result, history: Optional[RunHistory] = None) -> PerformanceReport:
        """
        Analyze an EnsemblePrecognitionResult to produce a PerformanceReport.

        Args:
            result: EnsemblePrecognitionResult from PrecognitionOrchestrator
            history: Optional RunHistory for trend analysis
        """
        timings = result.per_predictor_timing or []
        window_timings = result.window_render_timings or []
        predictor_results = result.predictor_results or {}

        all_ms = [t.elapsed_ms for t in timings]
        vqpu_ms_list = [t.elapsed_ms for t in timings if t.is_vqpu]
        classical_ms_list = [t.elapsed_ms for t in timings if not t.is_vqpu]

        strategy_stats = _compute_stats(all_ms)
        vqpu_stats = _compute_stats(vqpu_ms_list) if vqpu_ms_list else None
        classical_stats = _compute_stats(classical_ms_list) if classical_ms_list else None

        # Efficiency ratios
        efficiency_ratios = []
        for t in timings:
            pr = predictor_results.get(t.strategy_name)
            score = pr.confidence if pr else 0.0
            sa = pr.sacred_alignment if pr else 0.0
            efficiency_ratios.append(EfficiencyRatio(
                name=t.strategy_name,
                total_ms=t.elapsed_ms,
                vqpu_circuit_ms=t.circuit_ms,
                classical_ms=t.classical_ms,
                vqpu_overhead_ms=t.vqpu_overhead_ms,
                vqpu_fraction=t.circuit_ms / t.elapsed_ms if t.elapsed_ms > 0 else 0.0,
                overhead_fraction=t.vqpu_overhead_ms / t.elapsed_ms if t.elapsed_ms > 0 else 0.0,
                useful_work_fraction=(t.elapsed_ms - t.vqpu_overhead_ms) / t.elapsed_ms if t.elapsed_ms > 0 else 1.0,
                score=score,
                score_per_ms=score / t.elapsed_ms if t.elapsed_ms > 0 else 0.0,
                sacred_alignment=sa,
                alignment_per_ms=sa / t.elapsed_ms if t.elapsed_ms > 0 else 0.0,
            ))

        # Window analytics
        window_analytics = PerformanceAnalyzer._window_analytics(window_timings)

        # Bottleneck detection
        bottlenecks = PerformanceAnalyzer._detect_bottlenecks(
            timings, result.total_elapsed_ms, window_timings,
        )

        # Quality correlation
        quality_corr = PerformanceAnalyzer._quality_correlation_precog(
            timings, predictor_results,
        )

        # Trend
        trend = None
        if history and history.n_runs >= 2:
            trend = history.analyze()

        vqpu_total = sum(t.circuit_ms for t in timings)
        classical_total = sum(t.classical_ms for t in timings)
        n_vqpu = sum(1 for t in timings if t.is_vqpu)
        n_classical = len(timings) - n_vqpu
        vqpu_frac = vqpu_total / result.total_elapsed_ms if result.total_elapsed_ms > 0 else 0.0
        avg_sa = result.sacred_alignment

        grade, label = PerformanceAnalyzer._compute_grade(
            strategy_stats, efficiency_ratios, vqpu_frac, avg_sa,
        )

        return PerformanceReport(
            strategy_stats=strategy_stats,
            vqpu_stats=vqpu_stats,
            classical_stats=classical_stats,
            bottlenecks=bottlenecks,
            dominant_bottleneck=bottlenecks[0] if bottlenecks else None,
            efficiency_ratios=efficiency_ratios,
            window_analytics=window_analytics,
            quality_correlation=quality_corr,
            trend_analysis=trend,
            total_elapsed_ms=result.total_elapsed_ms,
            vqpu_total_ms=vqpu_total,
            classical_total_ms=classical_total,
            vqpu_fraction=vqpu_frac,
            n_strategies=len(timings),
            n_vqpu_strategies=n_vqpu,
            n_classical_strategies=n_classical,
            sacred_alignment_avg=avg_sa,
            performance_grade=grade,
            grade_label=label,
            generated_at=time.time(),
        )

    # ── Bottleneck Detection ──

    @staticmethod
    def _detect_bottlenecks(
        timings,
        total_ms: float,
        window_timings=None,
    ) -> List[BottleneckInfo]:
        """Identify bottlenecks sorted by severity."""
        bottlenecks: List[BottleneckInfo] = []
        if not timings:
            return bottlenecks

        for t in timings:
            frac = t.elapsed_ms / total_ms if total_ms > 0 else 0.0
            # Severity thresholds: >60% critical, >40% high, >25% moderate
            if frac > 0.60:
                severity = "critical"
                suggestion = f"'{t.strategy_name}' dominates {frac:.0%} of total time; consider async or caching"
            elif frac > 0.40:
                severity = "high"
                suggestion = f"'{t.strategy_name}' uses {frac:.0%} of total; optimize or reduce complexity"
            elif frac > 0.25:
                severity = "moderate"
                suggestion = f"'{t.strategy_name}' at {frac:.0%}; monitor for degradation"
            else:
                continue  # Not a bottleneck

            # VQPU overhead check
            if t.vqpu_overhead_ms > 0 and t.elapsed_ms > 0:
                overhead_frac = t.vqpu_overhead_ms / t.elapsed_ms
                if overhead_frac > 0.5:
                    suggestion += f"; VQPU overhead is {overhead_frac:.0%} — circuit may be too complex"

            bottlenecks.append(BottleneckInfo(
                component=t.strategy_name,
                elapsed_ms=t.elapsed_ms,
                fraction_of_total=frac,
                severity=severity,
                suggestion=suggestion,
            ))

        # Window outlier check
        if window_timings:
            render_ms_list = [w.render_ms for w in window_timings]
            if render_ms_list:
                sorted_renders = sorted(render_ms_list)
                p95 = _percentile(sorted_renders, 95)
                outliers = [w for w in window_timings if w.render_ms > p95 * 1.5]
                if outliers:
                    worst = max(outliers, key=lambda w: w.render_ms)
                    bottlenecks.append(BottleneckInfo(
                        component=f"window_{worst.window_index}",
                        elapsed_ms=worst.render_ms,
                        fraction_of_total=worst.render_ms / total_ms if total_ms > 0 else 0.0,
                        severity="moderate" if worst.render_ms > p95 * 2 else "low",
                        suggestion=f"Window {worst.window_index} render outlier at {worst.render_ms:.2f}ms (p95={p95:.2f}ms)",
                    ))

        # Sort by fraction (worst first)
        bottlenecks.sort(key=lambda b: b.fraction_of_total, reverse=True)
        return bottlenecks

    # ── Quality Correlation ──

    @staticmethod
    def _quality_correlation(timings, strategy_results) -> Optional[QualityCorrelation]:
        """Correlate time spent vs score quality for search."""
        if len(timings) < 2:
            return None

        times = []
        scores = []
        for t in timings:
            sr = strategy_results.get(t.strategy_name)
            if sr:
                times.append(t.elapsed_ms)
                scores.append(sr.score)

        if len(times) < 2:
            return None

        r = _pearson_correlation(times, scores)

        # Find best score and best efficiency
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        efficiencies = [(scores[i] / times[i] if times[i] > 0 else 0.0, i) for i in range(len(scores))]
        best_eff_idx = max(range(len(efficiencies)), key=lambda i: efficiencies[i][0])

        # Fastest and slowest
        fastest_idx = min(range(len(times)), key=lambda i: times[i])
        slowest_idx = max(range(len(times)), key=lambda i: times[i])

        # Diminishing returns: strong positive correlation between time and score
        # means spending more time helps; weak/negative means it doesn't
        diminishing = r < 0.3

        return QualityCorrelation(
            pearson_r=r,
            fastest_score=scores[fastest_idx],
            slowest_score=scores[slowest_idx],
            best_score_strategy=timings[best_idx].strategy_name,
            best_score_ms=times[best_idx],
            best_efficiency_strategy=timings[best_eff_idx].strategy_name,
            best_efficiency_ratio=efficiencies[best_eff_idx][0],
            diminishing_returns=diminishing,
        )

    @staticmethod
    def _quality_correlation_precog(timings, predictor_results) -> Optional[QualityCorrelation]:
        """Correlate time spent vs confidence quality for precognition."""
        if len(timings) < 2:
            return None

        times = []
        scores = []
        for t in timings:
            pr = predictor_results.get(t.strategy_name)
            if pr:
                times.append(t.elapsed_ms)
                scores.append(pr.confidence)

        if len(times) < 2:
            return None

        r = _pearson_correlation(times, scores)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        efficiencies = [(scores[i] / times[i] if times[i] > 0 else 0.0, i) for i in range(len(scores))]
        best_eff_idx = max(range(len(efficiencies)), key=lambda i: efficiencies[i][0])

        fastest_idx = min(range(len(times)), key=lambda i: times[i])
        slowest_idx = max(range(len(times)), key=lambda i: times[i])

        diminishing = r < 0.3

        return QualityCorrelation(
            pearson_r=r,
            fastest_score=scores[fastest_idx],
            slowest_score=scores[slowest_idx],
            best_score_strategy=timings[best_idx].strategy_name,
            best_score_ms=times[best_idx],
            best_efficiency_strategy=timings[best_eff_idx].strategy_name,
            best_efficiency_ratio=efficiencies[best_eff_idx][0],
            diminishing_returns=diminishing,
        )

    # ── Window Analytics ──

    @staticmethod
    def _window_analytics(window_timings) -> Optional[WindowAnalytics]:
        """Analytics over window render timings."""
        if not window_timings:
            return None

        render_ms_list = [w.render_ms for w in window_timings]
        s = sorted(render_ms_list)
        n = len(s)
        total = sum(s)
        mean = total / n
        std = _std_dev(s, mean)
        median = _percentile(s, 50)
        p95 = _percentile(s, 95)

        # Outliers: > 1.5 * IQR above Q3, or > 2× p95
        threshold = p95 * 2 if p95 > 0 else mean + 3 * std
        outlier_indices = [
            w.window_index for w in window_timings if w.render_ms > threshold
        ]

        throughput = (n / (total / 1000.0)) if total > 0 else 0.0

        return WindowAnalytics(
            n_windows=n,
            total_render_ms=total,
            avg_render_ms=mean,
            median_render_ms=median,
            std_render_ms=std,
            min_render_ms=s[0],
            max_render_ms=s[-1],
            p95_render_ms=p95,
            throughput_windows_per_sec=throughput,
            outlier_count=len(outlier_indices),
            outlier_indices=outlier_indices,
        )

    # ── Performance Grade ──

    @staticmethod
    def _compute_grade(
        stats: StatisticalSummary,
        efficiencies: List[EfficiencyRatio],
        vqpu_frac: float,
        sacred_alignment: float,
    ) -> Tuple[float, str]:
        """
        PHI-scored performance grade combining:
          - Efficiency (score/ms ratio across strategies)
          - Consistency (low CV = consistent timing)
          - Sacred alignment
          - VQPU utilization (higher is better when circuits are fast)

        Returns (grade_float 0.0–1.0, grade_label).
        """
        # Efficiency component: average score_per_ms normalized
        avg_eff = sum(e.score_per_ms for e in efficiencies) / max(len(efficiencies), 1)
        # Normalize: score_per_ms > 1.0 is excellent
        eff_score = min(1.0, avg_eff * PHI_CONJUGATE)

        # Consistency: lower CV is better (CV=0 → 1.0, CV>1 → 0.0)
        cv = stats.cv if stats.count > 0 else 0.0
        consistency_score = max(0.0, 1.0 - cv)

        # Sacred alignment (already 0–1)
        sacred_score = sacred_alignment

        # VQPU utilization: reward having quantum circuits proportional to total
        vqpu_score = min(1.0, vqpu_frac * PHI)

        # PHI-weighted composite
        grade = (
            eff_score * PHI_CONJUGATE ** 0         # 1.000 × efficiency
            + consistency_score * PHI_CONJUGATE ** 1 # 0.618 × consistency
            + sacred_score * PHI_CONJUGATE ** 2      # 0.382 × sacred
            + vqpu_score * PHI_CONJUGATE ** 3        # 0.236 × vqpu
        )
        # Normalize to 0–1 (max possible = 1 + 0.618 + 0.382 + 0.236 ≈ 2.236)
        max_possible = sum(PHI_CONJUGATE ** i for i in range(4))
        grade = grade / max_possible

        # Letter grade
        if grade >= 0.95:
            label = "S"
        elif grade >= 0.85:
            label = "A"
        elif grade >= 0.70:
            label = "B"
        elif grade >= 0.55:
            label = "C"
        elif grade >= 0.40:
            label = "D"
        else:
            label = "F"

        return grade, label

    # ── Convenience: Full Analysis ──

    def full_search_analysis(self, result) -> PerformanceReport:
        """Analyze search result with optional history tracking."""
        report = self.analyze_search(result, self._history)
        if self._history:
            self._history.record_from_report(report)
        return report

    def full_precognition_analysis(self, result) -> PerformanceReport:
        """Analyze precognition result with optional history tracking."""
        report = self.analyze_precognition(result, self._history)
        if self._history:
            self._history.record_from_report(report)
        return report

    @staticmethod
    def compare_reports(*reports: PerformanceReport) -> Dict[str, Any]:
        """
        Compare multiple PerformanceReports side by side.
        Useful for A/B testing or regression detection.
        """
        if not reports:
            return {}

        return {
            "n_reports": len(reports),
            "total_ms": [r.total_elapsed_ms for r in reports],
            "grades": [r.grade_label for r in reports],
            "grade_scores": [r.performance_grade for r in reports],
            "vqpu_fractions": [r.vqpu_fraction for r in reports],
            "sacred_alignments": [r.sacred_alignment_avg for r in reports],
            "n_bottlenecks": [len(r.bottlenecks) for r in reports],
            "best_total_ms": min(r.total_elapsed_ms for r in reports),
            "worst_total_ms": max(r.total_elapsed_ms for r in reports),
            "best_grade": max(reports, key=lambda r: r.performance_grade).grade_label,
            "speed_spread_pct": (
                (max(r.total_elapsed_ms for r in reports) - min(r.total_elapsed_ms for r in reports))
                / min(r.total_elapsed_ms for r in reports) * 100
                if min(r.total_elapsed_ms for r in reports) > 0 else 0.0
            ),
        }

    @staticmethod
    def summary_text(report: PerformanceReport) -> str:
        """Generate a human-readable analytics summary."""
        lines = [
            f"── L104 Performance Analytics ──────────────────────────",
            f"  Grade: {report.grade_label} ({report.performance_grade:.3f})",
            f"  Total: {report.total_elapsed_ms:.2f}ms | "
            f"Strategies: {report.n_strategies} (VQPU: {report.n_vqpu_strategies}, "
            f"Classical: {report.n_classical_strategies})",
            f"  Sacred Alignment: {report.sacred_alignment_avg:.4f}",
        ]

        # Stats
        s = report.strategy_stats
        if s.count > 0:
            lines.append(
                f"  Timing: mean={s.mean_ms:.2f}ms median={s.median_ms:.2f}ms "
                f"std={s.std_ms:.2f}ms p95={s.p95_ms:.2f}ms"
            )

        # VQPU breakdown
        if report.vqpu_total_ms > 0 or report.classical_total_ms > 0:
            lines.append(
                f"  VQPU: {report.vqpu_total_ms:.2f}ms ({report.vqpu_fraction:.1%}) | "
                f"Classical: {report.classical_total_ms:.2f}ms"
            )

        # Top efficiency
        if report.efficiency_ratios:
            best_eff = max(report.efficiency_ratios, key=lambda e: e.score_per_ms)
            lines.append(
                f"  Best Efficiency: '{best_eff.name}' → {best_eff.score_per_ms:.4f} score/ms"
            )

        # Bottlenecks
        if report.bottlenecks:
            b = report.bottlenecks[0]
            lines.append(
                f"  ⚠ Bottleneck: '{b.component}' [{b.severity}] "
                f"({b.elapsed_ms:.2f}ms, {b.fraction_of_total:.0%})"
            )

        # Quality
        if report.quality_correlation:
            qc = report.quality_correlation
            dr_flag = " [DIMINISHING RETURNS]" if qc.diminishing_returns else ""
            lines.append(
                f"  Quality r={qc.pearson_r:.3f}{dr_flag} | "
                f"Best efficiency: '{qc.best_efficiency_strategy}'"
            )

        # Windows
        if report.window_analytics:
            wa = report.window_analytics
            lines.append(
                f"  Windows: {wa.n_windows} @ {wa.avg_render_ms:.2f}ms avg "
                f"({wa.throughput_windows_per_sec:.0f}/sec, "
                f"{wa.outlier_count} outliers)"
            )

        # Trend
        if report.trend_analysis:
            ta = report.trend_analysis
            lines.append(
                f"  Trend: {ta.trend_direction} over {ta.n_runs} runs "
                f"({ta.performance_change_pct:+.1f}%)"
            )

        lines.append(f"────────────────────────────────────────────────────────")
        return "\n".join(lines)
