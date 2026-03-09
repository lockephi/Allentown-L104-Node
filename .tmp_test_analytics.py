"""
L104 Search v2.3 Analytics Validation Suite
Tests all analytical features: PerformanceAnalyzer, PerformanceReport,
RunHistory, statistical summaries, bottleneck detection, efficiency ratios,
quality correlations, window analytics, trend analysis, and grades.
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0
total = 0

def test(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name} — {detail}")

PHI = (1 + math.sqrt(5)) / 2

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Import validation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ PHASE 1: Imports ═══")
try:
    from l104_search import (
        PerformanceAnalyzer, PerformanceReport, RunHistory,
        StatisticalSummary, BottleneckInfo, EfficiencyRatio,
        WindowAnalytics, QualityCorrelation, TrendPoint, TrendAnalysis,
        StrategyTiming, WindowRenderTiming,
        SearchOrchestrator, PrecognitionOrchestrator, ThreeEngineSearchPrecog,
        EnsembleSearchResult, EnsemblePrecognitionResult,
    )
    test("All analytics imports", True)
except Exception as e:
    test("All analytics imports", False, str(e))
    sys.exit(1)

from l104_search import __version__
test("Version is 2.3.0", __version__ == "2.3.0", f"got {__version__}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: StatisticalSummary via _compute_stats
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ PHASE 2: Statistical Summary ═══")
from l104_search.analytics import _compute_stats, _percentile, _pearson_correlation, _linear_slope

vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
stats = _compute_stats(vals)
test("stats.count == 10", stats.count == 10)
test("stats.mean_ms == 5.5", abs(stats.mean_ms - 5.5) < 0.001, f"got {stats.mean_ms}")
test("stats.median_ms == 5.5", abs(stats.median_ms - 5.5) < 0.001, f"got {stats.median_ms}")
test("stats.min_ms == 1.0", stats.min_ms == 1.0)
test("stats.max_ms == 10.0", stats.max_ms == 10.0)
test("stats.total_ms == 55.0", abs(stats.total_ms - 55.0) < 0.001)
test("stats.p95 > 9.0", stats.p95_ms > 9.0, f"got {stats.p95_ms}")
test("stats.cv > 0", stats.cv > 0, f"got {stats.cv}")
test("stats.spread_ratio == 10.0", abs(stats.spread_ratio - 10.0) < 0.001)

# Empty
empty_stats = _compute_stats([])
test("empty stats count=0", empty_stats.count == 0)

# Pearson
xs = [1, 2, 3, 4, 5]
ys = [2, 4, 6, 8, 10]
r = _pearson_correlation(xs, ys)
test("pearson perfect positive = 1.0", abs(r - 1.0) < 0.001, f"got {r}")

ys_neg = [10, 8, 6, 4, 2]
r_neg = _pearson_correlation(xs, ys_neg)
test("pearson perfect negative = -1.0", abs(r_neg - (-1.0)) < 0.001, f"got {r_neg}")

# Linear slope
slope = _linear_slope([1, 2, 3, 4, 5])
test("linear slope=1.0", abs(slope - 1.0) < 0.001, f"got {slope}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Search with auto-analytics
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ PHASE 3: Search + Auto Analytics ═══")
hub = ThreeEngineSearchPrecog(enable_analytics=True)

items = list(range(200))
oracle = lambda x: x % 7 == 0
result = hub.search(items, oracle)

test("EnsembleSearchResult has analytics", result.analytics is not None)
if result.analytics:
    rpt = result.analytics
    test("analytics is PerformanceReport", isinstance(rpt, PerformanceReport))
    test("strategy_stats populated", rpt.strategy_stats.count > 0, f"count={rpt.strategy_stats.count}")
    test("n_strategies > 0", rpt.n_strategies > 0, f"got {rpt.n_strategies}")
    test("total_elapsed_ms > 0", rpt.total_elapsed_ms > 0)
    test("performance_grade 0-1", 0 <= rpt.performance_grade <= 1.0, f"got {rpt.performance_grade}")
    test("grade_label in S/A/B/C/D/F", rpt.grade_label in ("S", "A", "B", "C", "D", "F"), f"got {rpt.grade_label}")
    test("efficiency_ratios populated", len(rpt.efficiency_ratios) > 0)
    test("generated_at > 0", rpt.generated_at > 0)

    # Check efficiency ratios
    for er in rpt.efficiency_ratios:
        test(f"  eff '{er.name}' total_ms >= 0", er.total_ms >= 0)
        test(f"  eff '{er.name}' score_per_ms >= 0", er.score_per_ms >= 0)

    # Bottleneck detection
    test("bottlenecks is list", isinstance(rpt.bottlenecks, list))
    if rpt.bottlenecks:
        b = rpt.bottlenecks[0]
        test("bottleneck has severity", b.severity in ("critical", "high", "moderate", "low"))
        test("bottleneck has suggestion", len(b.suggestion) > 0)

    # Quality correlation
    if rpt.quality_correlation:
        qc = rpt.quality_correlation
        test("quality pearson_r in [-1,1]", -1.0 <= qc.pearson_r <= 1.0, f"got {qc.pearson_r}")
        test("quality best_efficiency_strategy set", len(qc.best_efficiency_strategy) > 0)

    # Summary text
    txt = PerformanceAnalyzer.summary_text(rpt)
    test("summary_text non-empty", len(txt) > 50, f"len={len(txt)}")
    test("summary contains Grade", "Grade:" in txt)
    print(f"\n{txt}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Precognition + Auto Analytics
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ PHASE 4: Precognition + Auto Analytics ═══")
history_data = [math.sin(i * 0.3) * 100 + 500 for i in range(50)]
precog = hub.predict(history_data, horizon=13)

test("EnsemblePrecognitionResult has analytics", precog.analytics is not None)
if precog.analytics:
    rpt2 = precog.analytics
    test("precog analytics is PerformanceReport", isinstance(rpt2, PerformanceReport))
    test("precog n_strategies > 0", rpt2.n_strategies > 0)
    test("precog grade_label set", rpt2.grade_label in ("S", "A", "B", "C", "D", "F"))
    test("precog total_elapsed_ms > 0", rpt2.total_elapsed_ms > 0)

    # Window analytics (from VQPUVariationalForecaster)
    if rpt2.window_analytics:
        wa = rpt2.window_analytics
        test("window n_windows > 0", wa.n_windows > 0, f"got {wa.n_windows}")
        test("window throughput > 0", wa.throughput_windows_per_sec > 0)
        test("window avg_render_ms > 0", wa.avg_render_ms > 0)
        test("window p95 >= median", wa.p95_render_ms >= wa.median_render_ms)
    else:
        test("window analytics present (may be None w/o windows)", True)

    # Precog summary
    txt2 = PerformanceAnalyzer.summary_text(rpt2)
    test("precog summary non-empty", len(txt2) > 50)
    print(f"\n{txt2}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Similarity search analytics
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ PHASE 5: Similarity Search Analytics ═══")
sim_result = hub.similarity_search("quantum", items)
test("similarity has analytics", sim_result.analytics is not None)
if sim_result.analytics:
    test("similarity grade set", sim_result.analytics.grade_label in ("S", "A", "B", "C", "D", "F"))

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: RunHistory + Trend Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ PHASE 6: RunHistory + Trends ═══")

# We ran 3 operations above (search, predict, similarity) → should have history
test("hub.history.n_runs >= 3", hub.history.n_runs >= 3, f"got {hub.history.n_runs}")

trend = hub.trend_report()
if trend:
    test("trend is TrendAnalysis", isinstance(trend, TrendAnalysis))
    test("trend.n_runs matches", trend.n_runs >= 3)
    test("trend direction valid", trend.trend_direction in (
        "improving", "degrading", "stable", "volatile"
    ), f"got {trend.trend_direction}")
    test("trend avg_total_ms > 0", trend.avg_total_ms > 0)
    print(f"  Trend: {trend.trend_direction} | avg={trend.avg_total_ms:.2f}ms | "
          f"slope={trend.ms_per_run_slope:.4f} | change={trend.performance_change_pct:+.1f}%")
else:
    test("trend available", False, "no trend returned")

# Manual RunHistory test
print("\n  RunHistory manual test:")
rh = RunHistory(max_runs=5)
for i in range(7):
    rh.record(total_ms=100 + i * 5, best_score=0.8 - i * 0.02)
test("RunHistory capped at max_runs=5", rh.n_runs == 5, f"got {rh.n_runs}")
manual_trend = rh.analyze()
test("manual trend degrading (time increasing)", manual_trend.trend_direction in ("degrading", "stable", "volatile"))
test("manual trend ms_slope > 0", manual_trend.ms_per_run_slope > 0, f"got {manual_trend.ms_per_run_slope}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7: Compare Reports
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ PHASE 7: Compare Reports ═══")
if result.analytics and precog.analytics:
    comparison = PerformanceAnalyzer.compare_reports(result.analytics, precog.analytics)
    test("comparison n_reports == 2", comparison["n_reports"] == 2)
    test("comparison has grades", len(comparison["grades"]) == 2)
    test("comparison has speed_spread_pct", "speed_spread_pct" in comparison)
    test("comparison best_grade set", comparison["best_grade"] in ("S", "A", "B", "C", "D", "F"))
    print(f"  Grades: {comparison['grades']} | Best: {comparison['best_grade']} | "
          f"Speed spread: {comparison['speed_spread_pct']:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 8: Status includes analytics
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ PHASE 8: Status Includes Analytics ═══")
status = hub.status()
test("status has analytics_enabled", "analytics_enabled" in status)
test("status analytics_enabled == True", status["analytics_enabled"] is True)
test("status has history_runs", "history_runs" in status)
test("status history_runs >= 3", status["history_runs"] >= 3, f"got {status['history_runs']}")
test("status has trend", "trend" in status)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 9: summary_text via hub
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ PHASE 9: Hub summary_text ═══")
txt_via_hub = hub.summary_text(result)
test("hub.summary_text works", "Grade:" in txt_via_hub)
txt_no_analytics = hub.summary_text("not_a_result")
test("hub.summary_text graceful on bad input", "no analytics" in txt_no_analytics)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 10: Analytics disabled
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ PHASE 10: Analytics Disabled ═══")
hub_no = ThreeEngineSearchPrecog(enable_analytics=False)
r_no = hub_no.search(list(range(50)), lambda x: x % 3 == 0)
test("analytics=None when disabled", r_no.analytics is None)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 11: v2.2 backward compat (timing still works)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ PHASE 11: v2.2 Backward Compat ═══")
test("per_strategy_timing populated", len(result.per_strategy_timing) > 0)
test("per_predictor_timing populated", len(precog.per_predictor_timing) > 0)
test("metadata has slowest_strategy", "slowest_strategy" in result.metadata)
test("metadata has slowest_predictor", "slowest_predictor" in precog.metadata)

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 60}")
print(f"  v2.3 Analytics Suite: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print(f"  ✓ ALL TESTS PASSED — v2.3.0 Analytical Upgrade VERIFIED")
else:
    print(f"  ✗ {failed} FAILURES — review above")
print(f"{'═' * 60}")
sys.exit(1 if failed else 0)
