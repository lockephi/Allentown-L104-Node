"""
L104 Search & Data Precognition Engine v3.0 — Cross-Engine Integration
═══════════════════════════════════════════════════════════════════════════════
Three-Engine + VQPU integrated search algorithms + predictive data precognition.
VQPU quantum scoring is applied to EVERY operation — all 10 search strategies
and all 8 predictors submit circuits to the Metal GPU vQPU when a bridge is
available, with graceful fallback to classical scoring.

v2.3 additions:
  - PerformanceAnalyzer    — Statistical summaries, bottleneck detection, efficiency ratios
  - PerformanceReport      — Full analytical report attached to every result (.analytics)
  - RunHistory             — Cross-run accumulator for trend detection
  - StatisticalSummary     — mean/median/std/p95/p99 timing distributions
  - BottleneckInfo         — Auto-detected performance bottlenecks with severity
  - EfficiencyRatio        — Per-strategy score/ms, VQPU utilization, overhead fractions
  - WindowAnalytics        — Windowed process throughput, outlier detection
  - QualityCorrelation     — Pearson correlation: time spent vs result quality
  - TrendAnalysis          — Performance trend direction across multiple runs
  - PHI-scored performance grades (S/A/B/C/D/F)
  - ThreeEngineSearchPrecog.analyzer / .history / .trend_report() / .summary_text()

v2.2 additions:
  - StrategyTiming      — Per-strategy/predictor rendering time breakdown
  - WindowRenderTiming  — Per-window render timing for windowed processes
  - All orchestrator results include per_strategy_timing / per_predictor_timing
  - VQPUVariationalForecaster tracks per-window render time (training + forecast)
  - VQPU circuit_ms vs classical_ms breakdown in all timing data
  - Metadata includes slowest/fastest strategy, total VQPU vs classical time

Engines:
  - Science Engine  → Entropy-guided search, coherence-based similarity, multidimensional projection
  - Math Engine     → Hyperdimensional VSA search, harmonic pattern matching, PHI-based pruning
  - Code Engine     → Semantic code search, structural pattern recognition, complexity analysis
  - VQPUBridge     → Metal GPU quantum circuit execution on ALL strategies + predictors

Search Algorithms (10 strategies — all VQPU-enhanced):
  Classical + VQPU scoring:
    - QuantumGroverSearch      — Grover amplitude amplification + VQPU quantum scoring
    - EntropyGuidedSearch      — Maxwell Demon entropy-reversal + VQPU scoring
    - HyperdimensionalSearch   — 10,000-dim VSA nearest-neighbor + VQPU scoring
    - CoherenceFieldSearch     — Coherence field pattern discovery + VQPU scoring
    - HarmonicResonanceSearch  — Frequency-domain resonance matching + VQPU scoring
    - ManifoldGeodesicSearch   — Geodesic shortest-path on manifolds + VQPU scoring
    - SacredAlignmentSearch    — GOD_CODE/PHI alignment + VQPU scoring
  Full VQPU circuit execution:
    - VQPUGroverSearch         — Real quantum Grover via Metal GPU vQPU
    - VQPUDatabaseSearch       — Quantum-accelerated L104 database research
    - QuantumReservoirSearch   — Quantum reservoir computing pattern matching

Data Precognition (8 predictors — all VQPU-enhanced):
  Classical + VQPU scoring:
    - EntropyCascadePredictor  — Entropy cascades + VQPU quantum scoring
    - CoherenceEvolutionOracle — Coherence evolution + VQPU scoring
    - WaveInterferenceForecaster — Wave superposition + VQPU scoring
    - HyperdimensionalPredictor — HD compute extrapolation + VQPU scoring
    - PhiConvergenceOracle     — PHI-convergence attractor + VQPU scoring
    - ManifoldFlowPredictor    — Geodesic flow + VQPU scoring
  Full VQPU circuit execution:
    - QuantumReservoirPredictor  — VQPU reservoir computing time-series forecasting
    - VQPUVariationalForecaster  — VQE-style variational circuit optimization

Orchestrators:
    - SearchOrchestrator       — Multi-strategy search (always includes VQPU)
    - PrecognitionOrchestrator — Multi-predictor ensemble (always includes VQPU)
    - ThreeEngineSearchPrecog  — Unified hub + search_with_precognition()

Analytics:
    - PerformanceAnalyzer      — Core analytics engine (stateless + history-aware)
    - PerformanceReport        — Full analytical report dataclass
    - RunHistory               — Cross-run accumulator for trend detection

Version: 3.0.0  |  INVARIANT: 527.5184818492612  |  PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

__version__ = "3.0.0"

from .search_algorithms import (
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
    SearchResult,
    SearchStrategy,
)

from .precognition import (
    EntropyCascadePredictor,
    CoherenceEvolutionOracle,
    WaveInterferenceForecaster,
    HyperdimensionalPredictor,
    PhiConvergenceOracle,
    ManifoldFlowPredictor,
    QuantumReservoirPredictor,
    VQPUVariationalForecaster,
    PrecognitionResult,
    ForecastPoint,
    AttractorState,
)

from .orchestrator import (
    SearchOrchestrator,
    PrecognitionOrchestrator,
    ThreeEngineSearchPrecog,
    EnsembleSearchResult,
    EnsemblePrecognitionResult,
    StrategyTiming,
    WindowRenderTiming,
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

__all__ = [
    # Search — Classical
    "QuantumGroverSearch",
    "EntropyGuidedSearch",
    "HyperdimensionalSearch",
    "CoherenceFieldSearch",
    "HarmonicResonanceSearch",
    "ManifoldGeodesicSearch",
    "SacredAlignmentSearch",
    # Search — VQPU-Accelerated
    "VQPUGroverSearch",
    "VQPUDatabaseSearch",
    "QuantumReservoirSearch",
    # Search data structures
    "SearchResult",
    "SearchStrategy",
    # Precognition — Classical
    "EntropyCascadePredictor",
    "CoherenceEvolutionOracle",
    "WaveInterferenceForecaster",
    "HyperdimensionalPredictor",
    "PhiConvergenceOracle",
    "ManifoldFlowPredictor",
    # Precognition — VQPU-Powered
    "QuantumReservoirPredictor",
    "VQPUVariationalForecaster",
    # Precognition data structures
    "PrecognitionResult",
    "ForecastPoint",
    "AttractorState",
    # Orchestrators
    "SearchOrchestrator",
    "PrecognitionOrchestrator",
    "ThreeEngineSearchPrecog",
    "EnsembleSearchResult",
    "EnsemblePrecognitionResult",
    # Timing data structures
    "StrategyTiming",
    "WindowRenderTiming",
    # Analytics
    "PerformanceAnalyzer",
    "PerformanceReport",
    "RunHistory",
    "StatisticalSummary",
    "BottleneckInfo",
    "EfficiencyRatio",
    "WindowAnalytics",
    "QualityCorrelation",
    "TrendPoint",
    "TrendAnalysis",
]
