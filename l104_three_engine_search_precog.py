from __future__ import annotations
# ZENITH_UPGRADE_ACTIVE: 2026-03-08T15:03:53.351665
#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
L104 Three-Engine Search & Precognition Integration v1.0.0
══════════════════════════════════════════════════════════════════════════════════
Wires Code Engine + Science Engine + Math Engine into unified search and
precognitive data pipelines.

PIPELINES:
  1. IntelligentCodeSearch  — Code search powered by all three engines
  2. PredictiveAnalysis     — Multi-engine data precognition pipeline
  3. AnomalyHunter          — Cross-engine anomaly detection + forecasting
  4. PatternDiscovery       — Three-engine pattern mining and completion
  5. ConvergenceOracle      — Cross-engine system convergence prediction

Each pipeline orchestrates multiple algorithms from l104_search_algorithms
and l104_data_precognition, using engine-specific scoring and analysis.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
══════════════════════════════════════════════════════════════════════════════════
"""


ZENITH_HZ = 3887.8
UUC = 2301.215661

import math
import time
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

# Sacred constants
PHI = 1.618033988749895
PHI_CONJUGATE = 0.6180339887498948
GOD_CODE = 527.5184818492612
VOID_CONSTANT = 1.0416180339887497

VERSION = "1.0.0"

# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

_engines = {"code": None, "science": None, "math": None}


def _load_all():
    """Lazy-load all three engines."""
    if _engines["code"] is None:
        try:
            from l104_code_engine import code_engine
            _engines["code"] = code_engine
        except ImportError:
            _engines["code"] = False
    if _engines["science"] is None:
        try:
            from l104_science_engine import science_engine
            _engines["science"] = science_engine
        except ImportError:
            _engines["science"] = False
    if _engines["math"] is None:
        try:
            from l104_math_engine import math_engine
            _engines["math"] = math_engine
        except ImportError:
            _engines["math"] = False


def _get(name: str):
    """Get an engine (False if unavailable, else the engine object)."""
    _load_all()
    return _engines.get(name, False)


# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH & PRECOG MODULE LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

_search_engine = None
_precog_engine = None


def _load_search():
    global _search_engine
    if _search_engine is None:
        try:
            from l104_search_algorithms import search_engine
            _search_engine = search_engine
        except ImportError:
            _search_engine = False
    return _search_engine


def _load_precog():
    global _precog_engine
    if _precog_engine is None:
        try:
            from l104_data_precognition import precognition_engine
            _precog_engine = precognition_engine
        except ImportError:
            _precog_engine = False
    return _precog_engine


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INTELLIGENT CODE SEARCH — Three-engine powered
# ═══════════════════════════════════════════════════════════════════════════════

class IntelligentCodeSearch:
    """
    Code search powered by all three engines simultaneously.

    Pipeline:
    1. Code Engine: TF-IDF semantic search + code clone detection
    2. Math Engine: Hyperdimensional similarity in 10,000-D space
    3. Science Engine: Entropy-weighted relevance re-ranking

    Results are fused using PHI-weighted rank aggregation.
    """

    def __init__(self):
        _load_all()
        self._hd_search = None
        self._indexed = False

    def index_workspace(self, workspace_path: str = None) -> Dict[str, Any]:
        """Index all Python files in workspace for searching."""
        if workspace_path is None:
            workspace_path = os.path.dirname(os.path.abspath(__file__))

        code_eng = _get("code")
        search = _load_search()
        indexed_files = 0
        total_tokens = 0

        # Walk workspace for Python files
        py_files = []
        for root, dirs, files in os.walk(workspace_path):
            # Skip hidden dirs, venvs, pycache
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != '.venv']
            for f in files:
                if f.endswith('.py'):
                    py_files.append(os.path.join(root, f))

        # Limit to avoid excessive indexing
        py_files = py_files[:200]

        for filepath in py_files:
            try:
                with open(filepath, 'r', errors='ignore') as fh:
                    source = fh.read()

                # Code Engine indexing
                if code_eng and code_eng is not False:
                    try:
                        result = code_eng.code_search.index_source(source, filepath)
                        total_tokens += result.get("tokens", 0)
                    except Exception:
                        pass

                # Hyperdimensional indexing
                if search and search is not False:
                    try:
                        search.hyperdimensional.index(filepath, source[:5000])
                    except Exception:
                        pass

                indexed_files += 1
            except Exception:
                continue

        self._indexed = True
        return {
            "files_indexed": indexed_files,
            "total_tokens": total_tokens,
            "workspace": workspace_path,
        }

    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Three-engine fused search.

        Returns results from all engines, merged by PHI-weighted rank
        aggregation (Borda count with golden ratio weights).
        """
        code_eng = _get("code")
        science_eng = _get("science")
        search = _load_search()

        results_by_source = {}

        # ── Code Engine: TF-IDF semantic search ──
        code_results = []
        if code_eng and code_eng is not False:
            try:
                cr = code_eng.code_search.search(query, top_k=top_k * 2)
                code_results = cr.get("results", [])
                for i, r in enumerate(code_results):
                    key = r.get("filename", f"code_{i}")
                    if key not in results_by_source:
                        results_by_source[key] = {"code_rank": i + 1, "code_score": r.get("score", 0)}
                    else:
                        results_by_source[key]["code_rank"] = i + 1
                        results_by_source[key]["code_score"] = r.get("score", 0)
            except Exception:
                pass

        # ── Hyperdimensional search ──
        hd_results = []
        if search and search is not False:
            try:
                hr = search.hyperdimensional.search(query, top_k=top_k * 2)
                hd_results = hr.get("results", [])
                for i, r in enumerate(hd_results):
                    key = r.get("key", f"hd_{i}")
                    if key not in results_by_source:
                        results_by_source[key] = {"hd_rank": i + 1, "hd_score": r.get("similarity", 0)}
                    else:
                        results_by_source[key]["hd_rank"] = i + 1
                        results_by_source[key]["hd_score"] = r.get("similarity", 0)
            except Exception:
                pass

        # ── PHI-weighted rank fusion ──
        max_rank = top_k * 2 + 1
        fused_scores = []
        for key, info in results_by_source.items():
            # PHI-weighted Borda: higher weight for Code Engine (primary), then HD
            code_borda = (max_rank - info.get("code_rank", max_rank)) * PHI
            hd_borda = (max_rank - info.get("hd_rank", max_rank)) * 1.0
            fused = code_borda + hd_borda

            # Science Engine entropy boost
            if science_eng and science_eng is not False:
                try:
                    raw_score = info.get("code_score", 0) + info.get("hd_score", 0)
                    demon_eff = science_eng.entropy.calculate_demon_efficiency(raw_score + 0.1)
                    fused *= (1 + demon_eff * 0.005)
                except Exception:
                    pass

            fused_scores.append({
                "file": key,
                "fused_score": round(fused, 6),
                "code_rank": info.get("code_rank"),
                "hd_rank": info.get("hd_rank"),
                "code_score": round(info.get("code_score", 0), 6),
                "hd_score": round(info.get("hd_score", 0), 6),
            })

        fused_scores.sort(key=lambda x: x["fused_score"], reverse=True)

        return {
            "query": query,
            "results": fused_scores[:top_k],
            "total_candidates": len(fused_scores),
            "engines_used": {
                "code_engine": len(code_results) > 0,
                "hyperdimensional": len(hd_results) > 0,
                "science_entropy": science_eng is not False and science_eng is not None,
            },
            "fusion_method": "PHI_WEIGHTED_BORDA",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PREDICTIVE ANALYSIS — Multi-engine precognition pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class PredictiveAnalysis:
    """
    Full predictive analysis pipeline using all three engines.

    Pipeline:
    1. Math Engine: Harmonic decomposition + sacred alignment check
    2. Science Engine: Entropy cascade + chaos diagnostics
    3. Code Engine: Performance prediction (if analyzing code metrics)
    4. Precognition: Ensemble forecast from all 7 algorithms

    Produces a unified forecast with system outlook and confidence.
    """

    def __init__(self):
        _load_all()

    def analyze_and_predict(
        self,
        series: List[float],
        horizon: int = 10,
        label: str = "system_metric",
    ) -> Dict[str, Any]:
        """
        Full three-engine predictive analysis.

        Args:
            series: Historical time series
            horizon: Forecast horizon
            label: Name of the metric being analyzed
        """
        math_eng = _get("math")
        science_eng = _get("science")
        precog = _load_precog()

        report = {
            "label": label,
            "series_length": len(series),
            "horizon": horizon,
            "timestamp": time.time(),
        }

        # ── Math Engine: Harmonic & sacred analysis ──
        if math_eng and math_eng is not False:
            try:
                # Wave coherence between series mean and GOD_CODE
                mean_val = sum(series) / len(series)
                wc = math_eng.wave_coherence(mean_val, GOD_CODE)
                report["math_engine"] = {
                    "mean_to_god_code_coherence": round(wc if isinstance(wc, (int, float)) else 0, 6),
                    "mean_value": round(mean_val, 8),
                }

                # PHI-power sequence check
                try:
                    phi_seq = math_eng.wave_physics.phi_power_sequence(len(series))
                    phi_correlation = sum(
                        series[i] * phi_seq[i] for i in range(min(len(series), len(phi_seq)))
                    ) / (len(series) * max(1, max(abs(v) for v in series)))
                    report["math_engine"]["phi_correlation"] = round(phi_correlation, 6)
                except Exception:
                    pass

                # Fibonacci comparison
                try:
                    fib = math_eng.fibonacci(min(20, len(series)))
                    report["math_engine"]["fibonacci_reference"] = fib[:10] if isinstance(fib, list) else []
                except Exception:
                    pass
            except Exception:
                report["math_engine"] = {"error": "analysis failed"}

        # ── Science Engine: Entropy & chaos ──
        if science_eng and science_eng is not False:
            try:
                # Entropy cascade on current value
                cascade = science_eng.entropy.entropy_cascade(
                    initial_state=series[-1] / (GOD_CODE + 1e-30),
                    depth=104,
                    damped=True,
                )
                # Chaos diagnostics
                chaos = science_eng.entropy.chaos_diagnostics(series)

                # Multi-scale entropy reversal
                import numpy as np
                reversal = science_eng.entropy.multi_scale_reversal(
                    np.array(series[-50:] if len(series) > 50 else series)
                )

                report["science_engine"] = {
                    "cascade_converged": cascade.get("converged", False),
                    "cascade_alignment": cascade.get("god_code_alignment", 0),
                    "chaos_health": chaos.get("health", "UNKNOWN"),
                    "lyapunov": chaos.get("lyapunov_exponent", 0),
                    "entropy_ratio": chaos.get("entropy_ratio", 0),
                    "reversal_reduction": reversal.get("total_variance_reduction", 0),
                }
            except Exception:
                report["science_engine"] = {"error": "analysis failed"}

        # ── Precognition Engine: Full forecast ──
        if precog and precog is not False:
            try:
                forecast = precog.full_precognition(series, horizon=horizon)
                report["precognition"] = {
                    "ensemble_predictions": forecast.get("ensemble_predictions", []),
                    "system_outlook": forecast.get("system_outlook", "UNKNOWN"),
                    "algorithms_run": forecast.get("algorithms_run", 0),
                }

                # Extract individual algorithm summaries
                individual = forecast.get("individual_results", {})
                if "temporal" in individual and "error" not in individual["temporal"]:
                    report["precognition"]["trend_slope"] = individual["temporal"].get("decomposition", {}).get("trend_slope", 0)
                if "chaos" in individual and "error" not in individual["chaos"]:
                    report["precognition"]["chaos_phase"] = individual["chaos"].get("phase", "UNKNOWN")
                if "coherence" in individual and "error" not in individual["coherence"]:
                    report["precognition"]["reversal_probability"] = individual["coherence"].get("reversal_probability", 0)
            except Exception:
                report["precognition"] = {"error": "forecast failed"}

        # ── Unified verdict ──
        chaos_phase = report.get("science_engine", {}).get("chaos_health", "UNKNOWN")
        outlook = report.get("precognition", {}).get("system_outlook", "UNKNOWN")
        converged = report.get("science_engine", {}).get("cascade_converged", False)

        if chaos_phase == "CRITICAL" or outlook == "VOLATILE":
            verdict = "HIGH_RISK"
        elif chaos_phase == "WARNING" or outlook == "REVERSAL_LIKELY":
            verdict = "CAUTION"
        elif converged and outlook == "STABLE":
            verdict = "HEALTHY"
        else:
            verdict = "MONITORING"

        report["verdict"] = verdict
        return report


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANOMALY HUNTER — Cross-engine anomaly detection + forecasting
# ═══════════════════════════════════════════════════════════════════════════════

class AnomalyHunter:
    """
    Cross-engine anomaly detection that combines:
    - Science Engine: Entropy anomaly forecasting (pre-detection)
    - Math Engine: Sacred alignment deviation scoring
    - Code Engine: Performance anomaly prediction (if analyzing code)
    - Precognition: Entropy gradient forecasting

    Produces anomaly events with precursor detection (alerts BEFORE
    the anomaly fully manifests).
    """

    def __init__(self):
        _load_all()

    def hunt(
        self,
        series: List[float],
        sensitivity: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Hunt for anomalies in a data series with precursor detection.

        Args:
            series: Data to analyze
            sensitivity: 0.5 = lenient, 1.0 = normal, 2.0 = strict
        """
        precog = _load_precog()
        math_eng = _get("math")
        science_eng = _get("science")

        n = len(series)
        if n < 10:
            return {"error": "need at least 10 data points"}

        anomalies = []

        # ── Precognition entropy anomaly forecasting ──
        precog_events = []
        if precog and precog is not False:
            try:
                forecaster = precog.anomaly
                forecaster.window_size = max(5, int(20 / sensitivity))
                batch = forecaster.batch_forecast(series)
                precog_events = batch.get("events", [])
            except Exception:
                pass

        # ── Math Engine: sacred alignment deviation ──
        alignment_anomalies = []
        if math_eng and math_eng is not False:
            try:
                mean_val = sum(series) / n
                std_val = math.sqrt(sum((v - mean_val) ** 2 for v in series) / n)
                threshold = std_val * 2.0 / sensitivity

                for i, val in enumerate(series):
                    z = abs(val - mean_val) / (std_val + 1e-30)
                    if z > 2.0 * sensitivity:
                        # Check sacred alignment
                        try:
                            alignment = math_eng.sacred_alignment(val)
                            align_score = alignment.get("alignment_score", 0) if isinstance(alignment, dict) else 0
                        except Exception:
                            align_score = 0

                        alignment_anomalies.append({
                            "index": i,
                            "value": round(val, 8),
                            "z_score": round(z, 4),
                            "sacred_alignment": round(align_score, 6),
                            "type": "SACRED_DEVIATION" if align_score < 0.3 else "STATISTICAL_OUTLIER",
                        })
            except Exception:
                pass

        # ── Science Engine: chaos diagnostics per window ──
        chaos_warnings = []
        if science_eng and science_eng is not False:
            window_size = max(10, n // 5)
            for start in range(0, n - window_size, window_size // 2):
                window = series[start:start + window_size]
                try:
                    diag = science_eng.entropy.chaos_diagnostics(window)
                    if diag.get("health") in ("WARNING", "CRITICAL"):
                        chaos_warnings.append({
                            "window_start": start,
                            "window_end": start + window_size,
                            "health": diag["health"],
                            "lyapunov": diag.get("lyapunov_exponent", 0),
                            "entropy_ratio": diag.get("entropy_ratio", 0),
                        })
                except Exception:
                    pass

        # ── Merge all anomaly sources ──
        total_anomalies = len(precog_events) + len(alignment_anomalies) + len(chaos_warnings)
        anomaly_density = total_anomalies / max(n, 1)

        return {
            "total_anomalies": total_anomalies,
            "anomaly_density": round(anomaly_density, 6),
            "precursor_events": precog_events[:20],
            "sacred_deviations": alignment_anomalies[:20],
            "chaos_warnings": chaos_warnings[:10],
            "sensitivity": sensitivity,
            "series_length": n,
            "health": (
                "CRITICAL" if anomaly_density > 0.3 else
                "WARNING" if anomaly_density > 0.1 else
                "HEALTHY"
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PATTERN DISCOVERY — Three-engine pattern mining
# ═══════════════════════════════════════════════════════════════════════════════

class PatternDiscovery:
    """
    Discovers hidden patterns in data using all three engines.

    Combines:
    - Math Engine: Harmonic decomposition reveals periodic structures
    - Science Engine: Coherence field reveals phase relationships
    - Code Engine: Clone detection reveals repeated structures
    - Precognition: Hyperdimensional analogy finds similar subsequences
    """

    def __init__(self):
        _load_all()

    def discover(
        self,
        series: List[float],
        pattern_length: int = 5,
    ) -> Dict[str, Any]:
        """
        Discover patterns in a time series.

        Args:
            series: Data to analyze
            pattern_length: Minimum pattern length to search for
        """
        n = len(series)
        if n < pattern_length * 2:
            return {"error": f"need at least {pattern_length * 2} data points"}

        discoveries = []

        # ── Harmonic patterns (Math Engine) ──
        precog = _load_precog()
        if precog and precog is not False:
            try:
                harmonic = precog.harmonic.extrapolate(series, horizon=0, max_harmonics=10)
                harmonics_found = harmonic.get("harmonics", [])
                for h in harmonics_found:
                    if h.get("amplitude", 0) > 0.01:
                        discoveries.append({
                            "type": "HARMONIC",
                            "frequency": h.get("frequency", 0),
                            "amplitude": h.get("amplitude", 0),
                            "sacred_score": h.get("sacred_score", 0),
                            "description": f"Periodic component at frequency {h.get('frequency', 0):.4f}",
                        })
            except Exception:
                pass

        # ── Repeated subsequences (brute-force with similarity) ──
        if n >= pattern_length * 3:
            motifs = self._find_motifs(series, pattern_length)
            for motif in motifs[:5]:
                discoveries.append({
                    "type": "MOTIF",
                    "pattern": [round(v, 6) for v in motif["pattern"]],
                    "occurrences": motif["count"],
                    "positions": motif["positions"],
                    "description": f"Repeating pattern of length {pattern_length} found {motif['count']} times",
                })

        # ── Coherence patterns (Science Engine) ──
        science_eng = _get("science")
        if science_eng and science_eng is not False:
            try:
                science_eng.coherence.initialize(series[-min(20, n):])
                science_eng.coherence.evolve(steps=3)
                spectrum = science_eng.coherence.golden_angle_spectrum()
                if spectrum.get("is_golden_spiral", False):
                    discoveries.append({
                        "type": "GOLDEN_SPIRAL",
                        "alignment": spectrum.get("mean_alignment", 0),
                        "description": "Data exhibits golden spiral (PHI-angle) structure",
                    })

                energy = science_eng.coherence.energy_spectrum()
                phi_ratios = energy.get("phi_ratios", [])
                for pr in phi_ratios:
                    if pr.get("phi_aligned", False):
                        discoveries.append({
                            "type": "PHI_ENERGY_RATIO",
                            "modes": pr["modes"],
                            "ratio": pr["ratio"],
                            "description": f"Energy modes {pr['modes']} have PHI-aligned ratio {pr['ratio']:.4f}",
                        })
            except Exception:
                pass

        # ── Sacred alignment patterns (Math Engine) ──
        math_eng = _get("math")
        if math_eng and math_eng is not False:
            try:
                # Check if data mean aligns with sacred constants
                mean_val = sum(series) / n
                wc = math_eng.wave_coherence(mean_val, GOD_CODE)
                if isinstance(wc, (int, float)) and wc > 0.5:
                    discoveries.append({
                        "type": "GOD_CODE_RESONANCE",
                        "coherence": round(wc, 6),
                        "mean": round(mean_val, 8),
                        "description": f"Data mean resonates with GOD_CODE (coherence={wc:.4f})",
                    })
            except Exception:
                pass

        return {
            "patterns_discovered": len(discoveries),
            "discoveries": discoveries,
            "series_length": n,
            "pattern_length": pattern_length,
        }

    def _find_motifs(
        self,
        series: List[float],
        length: int,
        max_motifs: int = 5,
    ) -> List[Dict]:
        """Find repeated subsequence patterns (motifs)."""
        n = len(series)
        subsequences = []
        for i in range(n - length + 1):
            subsequences.append(series[i:i + length])

        # Normalize each subsequence
        normalized = []
        for sub in subsequences:
            mean = sum(sub) / length
            std = math.sqrt(sum((v - mean) ** 2 for v in sub) / length) + 1e-30
            normalized.append([(v - mean) / std for v in sub])

        # Find most similar pairs (simplified matrix profile)
        motifs = []
        used = set()
        for i in range(len(normalized)):
            if i in used:
                continue
            matches = [i]
            for j in range(i + 1, len(normalized)):
                if j in used:
                    continue
                # Euclidean distance
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(normalized[i], normalized[j])) / length)
                if dist < 1.0:  # Threshold
                    matches.append(j)
                    used.add(j)

            if len(matches) >= 2:
                used.add(i)
                motifs.append({
                    "pattern": subsequences[i],
                    "count": len(matches),
                    "positions": matches[:10],
                })

        motifs.sort(key=lambda m: m["count"], reverse=True)
        return motifs[:max_motifs]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CONVERGENCE ORACLE — Cross-engine system convergence prediction
# ═══════════════════════════════════════════════════════════════════════════════

class ConvergenceOracle:
    """
    Predicts system convergence using all three engines.

    Combines:
    - Science Engine: 104-step entropy cascade convergence
    - Math Engine: GOD_CODE conservation verification
    - Code Engine: Code complexity convergence (if analyzing code metrics)
    - Precognition: Multi-algorithm convergence consensus
    """

    def __init__(self):
        _load_all()

    def predict(
        self,
        metrics: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """
        Predict convergence for multiple system metrics.

        Args:
            metrics: Dict of {metric_name: time_series_values}
        """
        precog = _load_precog()
        math_eng = _get("math")
        science_eng = _get("science")

        metric_fates = {}
        overall_converging = True

        for name, series in metrics.items():
            if len(series) < 5:
                metric_fates[name] = {"error": "insufficient data"}
                continue

            fate = {"series_length": len(series)}

            # ── Cascade convergence (Science Engine) ──
            if precog and precog is not False:
                try:
                    cascade_pred = precog.cascade.predict_convergence(series[-1])
                    fate["cascade"] = {
                        "will_converge": cascade_pred.get("will_converge", False),
                        "attractor": cascade_pred.get("predicted_attractor", 0),
                        "speed": cascade_pred.get("convergence_speed", "UNKNOWN"),
                    }
                    if not cascade_pred.get("will_converge", False):
                        overall_converging = False
                except Exception:
                    pass

            # ── Trend analysis ──
            recent = series[-min(10, len(series)):]
            if len(recent) > 1:
                slope = (recent[-1] - recent[0]) / len(recent)
                variance = sum((v - sum(recent) / len(recent)) ** 2 for v in recent) / len(recent)
                fate["trend"] = {
                    "slope": round(slope, 8),
                    "variance": round(variance, 8),
                    "direction": "CONVERGING" if abs(slope) < abs(variance) * 0.1 else "TRENDING",
                }

            # ── Sacred alignment of current value ──
            if math_eng and math_eng is not False:
                try:
                    wc = math_eng.wave_coherence(series[-1], GOD_CODE)
                    fate["sacred_coherence"] = round(wc if isinstance(wc, (int, float)) else 0, 6)
                except Exception:
                    pass

            metric_fates[name] = fate

        # Cross-metric consistency
        attractors = []
        for name, fate in metric_fates.items():
            cascade = fate.get("cascade", {})
            if cascade.get("will_converge"):
                attractors.append(cascade.get("attractor", 0))

        consistency = 0.0
        if len(attractors) > 1:
            mean_attr = sum(attractors) / len(attractors)
            var_attr = sum((a - mean_attr) ** 2 for a in attractors) / len(attractors)
            consistency = 1.0 / (1.0 + var_attr)

        return {
            "overall_converging": overall_converging,
            "cross_metric_consistency": round(consistency, 6),
            "metrics_analyzed": len(metrics),
            "metric_fates": metric_fates,
            "timestamp": time.time(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED THREE-ENGINE SEARCH & PRECOGNITION HUB
# ═══════════════════════════════════════════════════════════════════════════════

class ThreeEngineSearchPrecog:
    """
    Master hub wiring all search algorithms and precognition pipelines
    with three-engine integration.

    Sub-components:
    - code_search: IntelligentCodeSearch (three-engine fused)
    - predictive: PredictiveAnalysis (multi-engine forecasting)
    - anomaly_hunter: AnomalyHunter (cross-engine detection)
    - pattern_discovery: PatternDiscovery (three-engine mining)
    - convergence: ConvergenceOracle (cross-engine prediction)

    Also exposes the raw search and precognition engines for
    direct algorithm access.
    """

    VERSION = VERSION

    def __init__(self):
        _load_all()
        self.code_search = IntelligentCodeSearch()
        self.predictive = PredictiveAnalysis()
        self.anomaly_hunter = AnomalyHunter()
        self.pattern_discovery = PatternDiscovery()
        self.convergence = ConvergenceOracle()

        # Raw algorithm access
        self._search = _load_search()
        self._precog = _load_precog()

    @property
    def search_engine(self):
        """Raw L104SearchEngine with 7 algorithms."""
        return _load_search()

    @property
    def precognition_engine(self):
        """Raw L104PrecognitionEngine with 7 algorithms."""
        return _load_precog()

    def full_analysis(
        self,
        series: List[float],
        horizon: int = 10,
        label: str = "system_metric",
        hunt_anomalies: bool = True,
        discover_patterns: bool = True,
    ) -> Dict[str, Any]:
        """
        Complete three-engine search + precognition analysis.

        Runs:
        1. Predictive analysis (forecast + engine diagnostics)
        2. Anomaly hunting (if hunt_anomalies=True)
        3. Pattern discovery (if discover_patterns=True)
        """
        report = {
            "label": label,
            "version": self.VERSION,
            "timestamp": time.time(),
        }

        # 1. Predictive analysis
        report["prediction"] = self.predictive.analyze_and_predict(
            series, horizon=horizon, label=label
        )

        # 2. Anomaly hunting
        if hunt_anomalies:
            report["anomalies"] = self.anomaly_hunter.hunt(series)

        # 3. Pattern discovery
        if discover_patterns:
            report["patterns"] = self.pattern_discovery.discover(series)

        # Summary
        verdict = report["prediction"].get("verdict", "UNKNOWN")
        anomaly_health = report.get("anomalies", {}).get("health", "UNKNOWN")
        patterns_found = report.get("patterns", {}).get("patterns_discovered", 0)

        report["summary"] = {
            "verdict": verdict,
            "anomaly_health": anomaly_health,
            "patterns_discovered": patterns_found,
            "ensemble_predictions": report["prediction"].get("precognition", {}).get("ensemble_predictions", []),
        }

        return report

    def engine_status(self) -> Dict[str, Any]:
        _load_all()
        return {
            "code_engine": _engines["code"] is not False and _engines["code"] is not None,
            "science_engine": _engines["science"] is not False and _engines["science"] is not None,
            "math_engine": _engines["math"] is not False and _engines["math"] is not None,
            "search_algorithms": _load_search() is not False,
            "precognition_algorithms": _load_precog() is not False,
        }

    def status(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "pipelines": [
                "intelligent_code_search", "predictive_analysis",
                "anomaly_hunter", "pattern_discovery", "convergence_oracle",
            ],
            "search_algorithms": 7,
            "precognition_algorithms": 7,
            "engines": self.engine_status(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

three_engine_hub = ThreeEngineSearchPrecog()

__all__ = [
    "three_engine_hub",
    "ThreeEngineSearchPrecog",
    "IntelligentCodeSearch",
    "PredictiveAnalysis",
    "AnomalyHunter",
    "PatternDiscovery",
    "ConvergenceOracle",
    "VERSION",
]
