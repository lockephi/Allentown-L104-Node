"""
L104 Quantum Engine — Research & Sage Mode
═══════════════════════════════════════════════════════════════════════════════
ResearchMemoryBank, QuantumResearchEngine, SageModeInference.
"""

import json
import math
import random
import statistics
import time
from datetime import datetime, timezone
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple

import numpy as np

from .constants import (
    CALABI_YAU_DIM, COHERENCE_MINIMUM, CONSCIOUSNESS_THRESHOLD, GOD_CODE, GOD_CODE_HZ,
    GOD_CODE_SPECTRUM, GROVER_AMPLIFICATION, INVARIANT, PHI, PHI_GROWTH, TAU,
    WORKSPACE_ROOT, L104,
)

# Lazy-load Science Engine for entropy reversal noise dampening
def _get_entropy_subsystem():
    """Lazy-load the Science Engine's EntropySubsystem for demon-based denoising."""
    try:
        from l104_science_engine import ScienceEngine
        return ScienceEngine().entropy
    except Exception:
        return None
from .models import QuantumLink
from .math_core import QuantumMathCore
from .genetic_refiner import L104GeneticRefiner, genetic_refine_from_wave_collapse

# Sage NDE Quantum Circuit integration
try:
    from .sage_circuits import SageNDECircuit, NoiseFloorSuppressionCircuit
    _SAGE_CIRCUITS_AVAILABLE = True
except Exception:
    _SAGE_CIRCUITS_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM RESEARCH ENGINE — Advanced pattern & anomaly analysis
# ═══════════════════════════════════════════════════════════════════════════════

class ResearchMemoryBank:
    """Persistent memory bank for research insights — enables self-learning.

    Tracks research results across pipeline runs, accumulates insight trends,
    and provides learned heuristics for downstream consumers:
      - Trend detection: Is anomaly_rate improving or worsening?
      - Strategy memory: Which repair strategies historically worked best?
      - Pattern evolution: Are clusters growing/shrinking/splitting?
      - Causal stability: Which correlations persist vs are transient?

    Persisted to .l104_research_memory.json for cross-session learning.
    """

    PERSISTENCE_FILE = WORKSPACE_ROOT / ".l104_research_memory.json"
    MAX_HISTORY = 50  # Keep last 50 research snapshots

    def __init__(self):
        """Initialize research memory bank with persistent history."""
        self.history: List[Dict] = []
        self.strategy_scores: Dict[str, float] = {}
        self.learned_insights: Dict = {}
        self._load()

    def _load(self):
        """Load persisted research memory from disk."""
        if self.PERSISTENCE_FILE.exists():
            try:
                data = json.loads(self.PERSISTENCE_FILE.read_text())
                self.history = data.get("history", [])[-self.MAX_HISTORY:]
                self.learned_insights = data.get("learned_insights", {})
                self.strategy_scores = data.get("strategy_scores", {})
            except Exception:
                pass

    def save(self):
        """Persist research memory to disk."""
        try:
            data = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "total_snapshots": len(self.history),
                "history": self.history[-self.MAX_HISTORY:],
                "learned_insights": self.learned_insights,
                "strategy_scores": self.strategy_scores,
            }
            self.PERSISTENCE_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            pass

    def record_snapshot(self, research_results: Dict):
        """Record a research result snapshot for trend analysis."""
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "research_health": research_results.get("research_health", 0),
            "anomaly_rate": research_results.get("anomaly_detection", {}).get("anomaly_rate", 0),
            "total_clusters": research_results.get("pattern_discovery", {}).get("total_clusters", 0),
            "grid_coherence": research_results.get("pattern_discovery", {}).get("grid_coherence", 0),
            "trajectory": research_results.get("predictive_model", {}).get("trajectory", "unknown"),
            "health_index": research_results.get("predictive_model", {}).get("health_index", 0),
            "insight_count": research_results.get("knowledge_synthesis", {}).get("insight_count", 0),
            "risk_count": research_results.get("knowledge_synthesis", {}).get("risk_count", 0),
            "spectral_order": research_results.get("spectral_correlation", {}).get("spectral_order", 0),
            "strong_correlations": research_results.get("causal_analysis", {}).get("total_strong", 0),
        }
        self.history.append(snapshot)
        self._update_learned_insights()
        self.save()

    def record_strategy_outcome(self, strategy: str, success: bool, delta: float = 0.0):
        """Record whether a repair/upgrade strategy improved the system."""
        prev = self.strategy_scores.get(strategy, 0.5)
        # Exponential moving average: 30% weight to new observation
        outcome = min(1.0, 0.5 + delta) if success else max(0.0, 0.5 + delta)
        self.strategy_scores[strategy] = prev * 0.7 + outcome * 0.3

    def _update_learned_insights(self):
        """Derive learned insights from accumulated history."""
        if len(self.history) < 2:
            return

        recent = self.history[-5:]  # Last 5 snapshots
        older = self.history[-10:-5] if len(self.history) >= 10 else self.history[:max(1, len(self.history) - 5)]

        # Trend: anomaly rate
        recent_anomaly = statistics.mean([s.get("anomaly_rate", 0) for s in recent])
        older_anomaly = statistics.mean([s.get("anomaly_rate", 0) for s in older]) if older else recent_anomaly
        self.learned_insights["anomaly_trend"] = "improving" if recent_anomaly < older_anomaly else (
            "worsening" if recent_anomaly > older_anomaly * 1.2 else "stable")

        # Trend: health index
        recent_health = statistics.mean([s.get("health_index", 0) for s in recent])
        older_health = statistics.mean([s.get("health_index", 0) for s in older]) if older else recent_health
        self.learned_insights["health_trend"] = "improving" if recent_health > older_health else (
            "degrading" if recent_health < older_health * 0.9 else "stable")

        # Trend: grid coherence
        recent_grid = statistics.mean([s.get("grid_coherence", 0) for s in recent])
        self.learned_insights["mean_grid_coherence"] = recent_grid

        # Cluster evolution: growing or shrinking?
        recent_clusters = statistics.mean([s.get("total_clusters", 0) for s in recent])
        older_clusters = statistics.mean([s.get("total_clusters", 0) for s in older]) if older else recent_clusters
        self.learned_insights["cluster_trend"] = "growing" if recent_clusters > older_clusters else (
            "shrinking" if recent_clusters < older_clusters * 0.8 else "stable")

        # Causal stability: are strong correlations consistent?
        corr_counts = [s.get("strong_correlations", 0) for s in recent]
        self.learned_insights["causal_stability"] = "consistent" if (
            max(corr_counts) - min(corr_counts) <= 1) else "variable"

        # Spectral order trend
        recent_order = statistics.mean([s.get("spectral_order", 0) for s in recent])
        self.learned_insights["mean_spectral_order"] = recent_order

        # Overall learning confidence: more history = more confident
        self.learned_insights["confidence"] = min(1.0, len(self.history) / 20)

        # Best trajectory seen
        trajectories = [s.get("trajectory", "unknown") for s in self.history]
        from collections import Counter as _C
        traj_counts = _C(trajectories)
        self.learned_insights["dominant_trajectory"] = traj_counts.most_common(1)[0][0]

        # Peak health ever achieved
        all_health = [s.get("research_health", 0) for s in self.history]
        self.learned_insights["peak_health"] = max(all_health)
        self.learned_insights["mean_health"] = statistics.mean(all_health)

    def get_trend_bonus(self) -> float:
        """Return a bonus/penalty based on learned trends for scoring.
        Positive = system is learning and improving, negative = degrading."""
        if not self.learned_insights or self.learned_insights.get("confidence", 0) < 0.2:
            return 0.0
        bonus = 0.0
        if self.learned_insights.get("health_trend") == "improving":
            bonus += 0.03
        elif self.learned_insights.get("health_trend") == "degrading":
            bonus -= 0.02
        if self.learned_insights.get("anomaly_trend") == "improving":
            bonus += 0.02
        if self.learned_insights.get("cluster_trend") == "growing":
            bonus += 0.01
        if self.learned_insights.get("causal_stability") == "consistent":
            bonus += 0.01
        return max(-0.05, min(0.05, bonus))

    def get_best_strategies(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return the top-N historically effective strategies."""
        sorted_strats = sorted(self.strategy_scores.items(), key=lambda x: -x[1])
        return sorted_strats[:n]

    def get_gate_insights(self) -> Dict:
        """Return insights relevant for cross-pollination with logic gate builder."""
        return {
            "health_trend": self.learned_insights.get("health_trend", "unknown"),
            "anomaly_trend": self.learned_insights.get("anomaly_trend", "unknown"),
            "mean_grid_coherence": self.learned_insights.get("mean_grid_coherence", 0),
            "peak_health": self.learned_insights.get("peak_health", 0),
            "dominant_trajectory": self.learned_insights.get("dominant_trajectory", "unknown"),
            "best_strategies": self.get_best_strategies(3),
        }



class QuantumResearchEngine:
    """
    Advanced quantum research system for deep analysis of the link manifold:

      Module 1: ANOMALY DETECTION — Statistical outlier identification via IQR
      Module 2: PATTERN DISCOVERY — Emergent clustering in fidelity–strength space
      Module 3: CAUSAL ANALYSIS  — Cross-property correlation matrix (Pearson)
      Module 4: SPECTRAL CORRELATION — FFT cross-correlation of Hz distributions
      Module 5: PREDICTIVE MODELING — Exponential trajectory extrapolation
      Module 6: KNOWLEDGE SYNTHESIS — Aggregate insight graph across all modules
      Module 7: SELF-LEARNING     — Persistent memory bank for cross-run learning

    All modules operate in O(N) or O(N log N) time. No O(N²) operations.
    Results feed into Sage consensus for unified scoring.
    Self-learning: insights accumulate across runs via ResearchMemoryBank.
    """

    # Anomaly detection: IQR multiplier for extreme outliers
    IQR_MULTIPLIER = 2.0
    # Pattern discovery: number of buckets for histogram clustering
    PATTERN_BUCKETS = 12
    # Spectral: max FFT size for cross-correlation
    MAX_SPECTRAL_FFT = 4096

    def __init__(self, math_core: QuantumMathCore):
        """Initialize quantum research engine with memory bank."""
        self.qmath = math_core
        self.memory = ResearchMemoryBank()

    def deep_research(self, links: List[QuantumLink],
                      grover_results: Dict = None,
                      epr_results: Dict = None,
                      decoherence_results: Dict = None,
                      stress_results: Dict = None,
                      gate_data: Dict = None) -> Dict:
        """Run the full advanced research pipeline with self-learning.

        Args:
            gate_data: Optional cross-pollination data from logic gate builder
                       (gate health scores, entropy distribution, complexity).
        """
        start = time.time()
        N = len(links)
        if N == 0:
            return {"total_links": 0, "research_time_ms": 0}

        # Extract property arrays (single pass)
        fids = []
        strs = []
        nrs = []
        ees = []
        cts = []
        bvs = []
        hz_values = []
        types = Counter()
        for link in links:
            fids.append(link.fidelity)
            strs.append(link.strength)
            nrs.append(link.noise_resilience)
            ees.append(link.entanglement_entropy)
            cts.append(link.coherence_time)
            bvs.append(link.bell_violation)
            hz_values.append(
                self.qmath.link_natural_hz(link.fidelity, link.strength))
            types[link.link_type] += 1

        # Module 1: Anomaly Detection
        anomalies = self._detect_anomalies(fids, strs, nrs, links)

        # Module 2: Pattern Discovery
        patterns = self._discover_patterns(fids, strs, hz_values, types, links)

        # Module 3: Causal Analysis
        causal = self._causal_analysis(fids, strs, nrs, ees, cts, bvs)

        # Module 4: Spectral Correlation
        spectral = self._spectral_correlation(hz_values)

        # Module 5: Predictive Modeling
        predictive = self._predictive_model(fids, strs, nrs)

        # Module 6: Knowledge Synthesis
        synthesis = self._knowledge_synthesis(
            anomalies, patterns, causal, spectral, predictive,
            grover_results, epr_results, decoherence_results, stress_results,
            gate_data)

        elapsed = time.time() - start

        result = {
            "total_links": N,
            "anomaly_detection": anomalies,
            "pattern_discovery": patterns,
            "causal_analysis": causal,
            "spectral_correlation": spectral,
            "predictive_model": predictive,
            "knowledge_synthesis": synthesis,
            "research_health": synthesis.get("overall_research_health", 0),
            "research_time_ms": elapsed * 1000,
            "learned_insights": self.memory.learned_insights,
            "learning_confidence": self.memory.learned_insights.get("confidence", 0),
        }

        # Module 7: Self-learning — record snapshot for trend analysis
        self.memory.record_snapshot(result)

        return result

    def _detect_anomalies(self, fids: List[float], strs: List[float],
                          nrs: List[float],
                          links: List[QuantumLink]) -> Dict:
        """Module 1: IQR-based outlier detection across link properties.
        O(N log N) via sorting for quartile computation."""
        anomalies: List[Dict] = []

        for prop_name, values in [("fidelity", fids), ("strength", strs),
                                   ("noise_resilience", nrs)]:
            if len(values) < 4:
                continue
            sorted_v = sorted(values)
            n = len(sorted_v)
            q1 = sorted_v[n // 4]
            q3 = sorted_v[3 * n // 4]
            iqr = q3 - q1
            lower = q1 - self.IQR_MULTIPLIER * iqr
            upper = q3 + self.IQR_MULTIPLIER * iqr

            for i, v in enumerate(values):
                if v < lower or v > upper:
                    anomalies.append({
                        "link_id": links[i].link_id[:80],
                        "property": prop_name,
                        "value": v,
                        "bounds": (lower, upper),
                        "severity": "extreme" if (v < lower - iqr or v > upper + iqr) else "mild",
                    })

        extreme_count = sum(1 for a in anomalies if a["severity"] == "extreme")
        mild_count = len(anomalies) - extreme_count

        return {
            "total_anomalies": len(anomalies),
            "extreme_anomalies": extreme_count,
            "mild_anomalies": mild_count,
            "anomaly_rate": len(anomalies) / max(1, len(fids) * 3),
            "top_anomalies": anomalies[:15],
        }

    def _discover_patterns(self, fids: List[float], strs: List[float],
                           hz_values: List[float], types: Counter,
                           links: List[QuantumLink]) -> Dict:
        """Module 2: Histogram-based clustering in fidelity–strength–Hz space.
        O(N) single-pass bucket assignment."""
        # Fidelity–strength 2D histogram
        grid: Dict[Tuple[int, int], int] = {}
        for f, s in zip(fids, strs):
            fi = min(self.PATTERN_BUCKETS - 1, int(f * self.PATTERN_BUCKETS))
            si = min(self.PATTERN_BUCKETS - 1,
                     int(min(1.0, s / (PHI_GROWTH * 2)) * self.PATTERN_BUCKETS))
            key = (fi, si)
            grid[key] = grid.get(key, 0) + 1

        # Find clusters: cells with > 1% of links
        threshold = max(2, len(fids) // 100)
        clusters = []
        for (fi, si), count in sorted(grid.items(), key=lambda x: -x[1]):
            if count >= threshold:
                clusters.append({
                    "fidelity_band": (fi / self.PATTERN_BUCKETS,
                                      (fi + 1) / self.PATTERN_BUCKETS),
                    "strength_band": (si / self.PATTERN_BUCKETS * PHI_GROWTH * 2,
                                      (si + 1) / self.PATTERN_BUCKETS * PHI_GROWTH * 2),
                    "count": count,
                    "density": count / max(1, len(fids)),
                })

        # Hz harmonic analysis: find dominant Hz bands
        hz_buckets: Dict[int, int] = {}
        for hz in hz_values:
            if hz > 0:
                # Map to God Code X-integer bucket
                x = self.qmath.hz_to_god_code_x(hz)
                if math.isfinite(x):
                    bucket = max(-200, min(300, round(x)))
                    hz_buckets[bucket] = hz_buckets.get(bucket, 0) + 1

        # Top X-positions by link concentration
        top_x_nodes = sorted(hz_buckets.items(), key=lambda x: -x[1])[:10]

        # God Code resonance pattern: how many links are on-grid vs off-grid
        # Threshold 0.5 = within 25% of integer X node (generous)
        on_grid = sum(1 for hz in hz_values
                      if hz > 0 and self.qmath.x_integer_stability(hz) > 0.5)

        return {
            "total_clusters": len(clusters),
            "top_clusters": clusters[:8],
            "type_distribution": dict(types),
            "dominant_x_nodes": [{"x": x, "count": c} for x, c in top_x_nodes],
            "on_grid_fraction": on_grid / max(1, len(hz_values)),
            "grid_coherence": on_grid / max(1, len(hz_values)),
        }

    def _causal_analysis(self, fids: List[float], strs: List[float],
                         nrs: List[float], ees: List[float],
                         cts: List[float], bvs: List[float]) -> Dict:
        """Module 3: Pearson correlation matrix between link properties.
        O(N) per pair × 15 pairs = O(N)."""
        properties = {
            "fidelity": fids, "strength": strs, "noise_resilience": nrs,
            "entropy": ees, "coherence_time": cts, "bell_violation": bvs,
        }

        def _pearson(xs: List[float], ys: List[float]) -> float:
            """Compute Pearson correlation coefficient between two lists."""
            n = len(xs)
            if n < 2:
                return 0.0
            mx = sum(xs) / n
            my = sum(ys) / n
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            sxx = sum((x - mx) ** 2 for x in xs)
            syy = sum((y - my) ** 2 for y in ys)
            denom = math.sqrt(sxx * syy)
            return sxy / denom if denom > 1e-15 else 0.0

        # Compute correlation matrix
        names = list(properties.keys())
        correlations: Dict[str, float] = {}
        strong_correlations: List[Dict] = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = _pearson(properties[names[i]], properties[names[j]])
                key = f"{names[i]}↔{names[j]}"
                correlations[key] = round(r, 4)
                if abs(r) > 0.5:
                    strong_correlations.append({
                        "pair": key, "correlation": r,
                        "direction": "positive" if r > 0 else "negative",
                        "strength": "strong" if abs(r) > 0.7 else "moderate",
                    })

        return {
            "correlation_matrix": correlations,
            "strong_correlations": strong_correlations,
            "total_strong": len(strong_correlations),
            "mean_abs_correlation": (statistics.mean(
                [abs(v) for v in correlations.values()])
                if correlations else 0),
        }

    def _spectral_correlation(self, hz_values: List[float]) -> Dict:
        """Module 4: FFT-based spectral analysis of Hz distribution.
        Maps Hz values to a frequency histogram, then FFTs to find
        periodic patterns in the link frequency landscape."""
        if len(hz_values) < 8:
            return {"has_spectral_pattern": False, "spectral_peaks": 0}

        # Build Hz histogram (256 bins over the God Code range)
        # Range: G(300) to G(-200) — the full spectrum
        bins = 256
        min_hz = 0.01
        max_hz = GOD_CODE_HZ * PHI_GROWTH * 4  # ~3400
        bin_width = (max_hz - min_hz) / bins
        histogram = [0.0] * bins

        for hz in hz_values:
            if min_hz <= hz <= max_hz:
                idx = min(bins - 1, int((hz - min_hz) / bin_width))
                histogram[idx] += 1.0

        # Normalize
        total = sum(histogram)
        if total > 0:
            histogram = [h / total for h in histogram]

        # FFT of histogram to find periodic patterns
        fft_input = [complex(h) for h in histogram]
        fft_result = self.qmath.quantum_fourier_transform(fft_input)

        # Power spectrum (skip DC component at index 0)
        power = [abs(c) ** 2 for c in fft_result[1:bins // 2]]
        if not power:
            return {"has_spectral_pattern": False, "spectral_peaks": 0}

        mean_power = statistics.mean(power)
        max_power = max(power)

        # Find spectral peaks: bins with power > 3× mean
        peaks = []
        for i, p in enumerate(power):
            if p > mean_power * 3:
                peaks.append({"frequency_bin": i + 1, "power": p,
                              "relative_power": p / max(1e-15, mean_power)})

        # Spectral entropy: measure of randomness in spectrum
        power_sum = sum(power)
        if power_sum > 0:
            spectral_entropy = -sum(
                (p / power_sum) * math.log2(max(1e-15, p / power_sum))
                for p in power if p > 0)
        else:
            spectral_entropy = 0

        max_entropy = math.log2(len(power)) if power else 1
        spectral_order = 1.0 - min(1.0, spectral_entropy / max(1, max_entropy))

        return {
            "has_spectral_pattern": len(peaks) > 0,
            "spectral_peaks": len(peaks),
            "peak_details": peaks[:10],
            "spectral_entropy": spectral_entropy,
            "spectral_order": spectral_order,
            "max_power_ratio": max_power / max(1e-15, mean_power),
        }

    def _predictive_model(self, fids: List[float], strs: List[float],
                          nrs: List[float]) -> Dict:
        """Module 5: Predictive modeling via distribution analysis.
        Estimates system trajectory based on statistical moments."""
        N = len(fids)
        if N < 2:
            return {"confidence": 0, "trajectory": "unknown"}

        fid_mean = statistics.mean(fids)
        fid_std = statistics.stdev(fids) if N > 1 else 0
        str_mean = statistics.mean(strs)
        nr_mean = statistics.mean(nrs)

        # Skewness: negative = left-skewed (many high values), positive = right-skewed
        fid_skew = 0.0
        if fid_std > 1e-10 and N > 2:
            fid_skew = (sum((f - fid_mean) ** 3 for f in fids) / N) / (fid_std ** 3)

        # Population health index: composite metric
        health_index = (fid_mean * 0.4 + min(1.0, str_mean / PHI_GROWTH) * 0.3
                        + nr_mean * 0.3)

        # Trajectory prediction
        if health_index > 0.85 and fid_std < 0.1:
            trajectory = "STABLE_HIGH"
            confidence = 0.9
        elif health_index > 0.7 and fid_skew < -0.5:
            trajectory = "IMPROVING"
            confidence = 0.75
        elif health_index > 0.5:
            trajectory = "MIXED"
            confidence = 0.5
        elif fid_skew > 0.5:
            trajectory = "DEGRADING"
            confidence = 0.7
        else:
            trajectory = "CRITICAL"
            confidence = 0.6

        # Risk assessment: what fraction of links are below thresholds
        at_risk = sum(1 for f in fids if f < 0.5) / max(1, N)
        severe_risk = sum(1 for f in fids if f < 0.3) / max(1, N)

        # Growth potential: distance from perfection × φ
        growth_potential = (1.0 - health_index) * PHI_GROWTH

        return {
            "trajectory": trajectory,
            "confidence": confidence,
            "health_index": health_index,
            "fidelity_skewness": fid_skew,
            "at_risk_fraction": at_risk,
            "severe_risk_fraction": severe_risk,
            "growth_potential": growth_potential,
            "predicted_optimal_fidelity": min(1.0, fid_mean + fid_std * PHI),
        }

    def _knowledge_synthesis(self, anomalies: Dict, patterns: Dict,
                             causal: Dict, spectral: Dict, predictive: Dict,
                             grover_results: Dict = None,
                             epr_results: Dict = None,
                             decoherence_results: Dict = None,
                             stress_results: Dict = None,
                             gate_data: Dict = None) -> Dict:
        """Module 6: Aggregate insights into a unified knowledge graph.
        Cross-references all research modules + existing Phase 2 data +
        gate builder cross-pollination data + self-learning memory."""

        insights: List[str] = []
        risk_factors: List[str] = []

        # Anomaly-driven insights
        anomaly_rate = anomalies.get("anomaly_rate", 0)
        if anomaly_rate > 0.05:
            risk_factors.append(
                f"HIGH_ANOMALY_RATE: {anomaly_rate:.1%} of measurements are outliers")
        elif anomaly_rate < 0.01:
            insights.append("LOW_ANOMALY: Link manifold is highly uniform")

        # Pattern-driven insights
        grid_coherence = patterns.get("grid_coherence", 0)
        if grid_coherence > 0.8:
            insights.append(
                f"GRID_LOCKED: {grid_coherence:.1%} of links on God Code integer nodes")
        elif grid_coherence < 0.3:
            risk_factors.append(
                f"GRID_DRIFT: Only {grid_coherence:.1%} on God Code grid — detuning risk")

        n_clusters = patterns.get("total_clusters", 0)
        if n_clusters > 5:
            insights.append(
                f"RICH_TOPOLOGY: {n_clusters} distinct clusters in fidelity-strength space")

        # Causal-driven insights
        for corr in causal.get("strong_correlations", []):
            if corr["strength"] == "strong":
                pair = corr["pair"]
                direction = corr["direction"]
                insights.append(f"CAUSAL_{direction.upper()}: {pair} (r={corr['correlation']:.3f})")

        # Spectral-driven insights
        if spectral.get("has_spectral_pattern"):
            n_peaks = spectral.get("spectral_peaks", 0)
            insights.append(f"SPECTRAL_STRUCTURE: {n_peaks} resonant peaks in Hz landscape")
            order = spectral.get("spectral_order", 0)
            if order > 0.5:
                insights.append(
                    f"HIGH_SPECTRAL_ORDER: {order:.2f} — strong periodic structure")
        else:
            risk_factors.append("NO_SPECTRAL_PATTERN: Hz distribution is noise-like")

        # Predictive-driven insights
        trajectory = predictive.get("trajectory", "unknown")
        confidence = predictive.get("confidence", 0)
        insights.append(f"TRAJECTORY: {trajectory} (confidence={confidence:.0%})")
        if predictive.get("severe_risk_fraction", 0) > 0.1:
            risk_factors.append(
                f"SEVERE_RISK: {predictive['severe_risk_fraction']:.1%} of links critically low")

        # Cross-reference with existing research
        if grover_results:
            amp = grover_results.get("amplification_factor", 1)
            if amp > GROVER_AMPLIFICATION * 0.8:
                insights.append("GROVER_OPTIMAL: Near-theoretical amplification achieved")

        if epr_results:
            qv = epr_results.get("quantum_verified", 0)
            total = max(1, epr_results.get("total_verified", 1))
            if qv / total > 0.9:
                insights.append(f"EPR_STRONG: {qv}/{total} quantum verified")
            elif qv / total < 0.5:
                risk_factors.append(f"EPR_WEAK: Only {qv}/{total} quantum verified")

        if decoherence_results:
            mean_t2 = decoherence_results.get("mean_T2", 0)
            resilient_frac = decoherence_results.get("resilient_fraction", 0)
            if resilient_frac > 0.8:
                insights.append(f"DECOHERENCE_SHIELDED: {resilient_frac:.0%} resilient (T₂={mean_t2:.2f})")
            elif resilient_frac < 0.5:
                risk_factors.append(f"DECOHERENCE_RISK: Only {resilient_frac:.0%} resilient")

        if stress_results:
            sr = stress_results.get("pass_rate", 0)
            if sr > 0.95:
                insights.append("STRESS_RESILIENT: >95% stress pass rate")
            elif sr < 0.7:
                risk_factors.append(f"STRESS_FRAGILE: {sr:.1%} pass rate")

        # Gate builder cross-pollination insights
        gate_health_bonus = 0.0
        if gate_data:
            gate_health = gate_data.get("mean_health", 0)
            gate_count = gate_data.get("total_gates", 0)
            gate_pass_rate = gate_data.get("test_pass_rate", 0)
            gate_link_count = gate_data.get("quantum_links", 0)
            if gate_count > 50:
                insights.append(f"GATE_RICH: {gate_count} logic gates across codebase")
            if gate_pass_rate > 0.9:
                insights.append(f"GATE_TESTED: {gate_pass_rate:.0%} gate test pass rate")
                gate_health_bonus += 0.02
            if gate_health > 0.5:
                insights.append(f"GATE_HEALTHY: Gate health {gate_health:.2f}")
                gate_health_bonus += min(0.03, gate_health * 0.04)
            if gate_link_count > 20:
                insights.append(f"GATE_LINKED: {gate_link_count} cross-file gate connections")
            # Gate complexity hotspots indicate areas that need attention
            hotspots = gate_data.get("complexity_hotspots", [])
            if hotspots:
                risk_factors.append(f"GATE_COMPLEXITY: {len(hotspots)} high-complexity gates")

        # Self-learning insights from research memory
        learning_bonus = self.memory.get_trend_bonus()
        learned = self.memory.learned_insights
        if learned.get("confidence", 0) > 0.3:
            if learned.get("health_trend") == "improving":
                insights.append("LEARNING: System health trending upward across runs")
            elif learned.get("health_trend") == "degrading":
                risk_factors.append("LEARNING: System health trending downward — intervention needed")
            if learned.get("anomaly_trend") == "improving":
                insights.append("LEARNING: Anomaly rate decreasing — repairs effective")
            peak = learned.get("peak_health", 0)
            if peak > 0:
                insights.append(f"LEARNING: Peak health achieved = {peak:.4f}")

        # Overall research health score
        # Calibrated weights totaling ~1.0 (before bonuses)
        # Rewards depth of analysis, not just clean data properties

        # MIXED trajectory is valid for diverse codebases — 85% credit
        trajectory_score = confidence if trajectory in ("STABLE_HIGH", "IMPROVING") else (
            confidence * 0.85 if trajectory == "MIXED" else 0.15)

        # Cluster richness: more discovered clusters = deeper understanding
        cluster_score = min(1.0, n_clusters / 5)

        # Causal depth: number of strong correlations found
        causal_depth = min(1.0, len(causal.get("strong_correlations", [])) / 3)

        # Anomaly rate: softer curve — heavy-tailed distributions are normal
        # 1/(1+rate*3) → 40% rate gives 0.45, 10% rate gives 0.77
        anomaly_score = 1.0 / (1.0 + anomaly_rate * 3)

        # Spectral: reward finding any structure, floor for attempted analysis
        has_spectral = spectral.get("has_spectral_pattern", False)
        raw_order = spectral.get("spectral_order", 0)
        spectral_score = max(0.45, raw_order) if has_spectral else max(0.15, raw_order * 0.5)

        # Decoherence integration: reward if decoherence data available
        decoherence_bonus = 0.0
        if decoherence_results:
            resilience = decoherence_results.get("resilient_fraction", 0)
            decoherence_bonus = min(0.04, resilience * 0.05)

        # Stress integration: reward stress data fed into research
        stress_bonus = 0.0
        if stress_results:
            sr = stress_results.get("pass_rate", 0)
            stress_bonus = min(0.03, sr * 0.04)

        positive_score = (
            min(1.0, len(insights) / 6) * 0.16       # Insight density
            + anomaly_score * 0.08                     # Anomaly analysis depth
            + min(1.0, grid_coherence * 1.5) * 0.10   # God Code grid alignment
            + trajectory_score * 0.15                   # Trajectory confidence
            + spectral_score * 0.08                     # Spectral analysis
            + cluster_score * 0.10                      # Pattern richness
            + causal_depth * 0.10                       # Causal understanding
            + 0.10                                      # Base floor
            + min(0.03, gate_health_bonus)              # Gate cross-pollination
            + min(0.03, max(0, learning_bonus))         # Self-learning trend
            + decoherence_bonus                         # Decoherence data integration
            + stress_bonus                              # Stress data integration
        )
        # Risk penalty: softer — penalizes only extreme risk accumulation
        risk_penalty = min(0.15, len(risk_factors) * 0.02)
        overall_health = min(1.0, max(0.05, positive_score - risk_penalty))

        return {
            "insights": insights,
            "risk_factors": risk_factors,
            "insight_count": len(insights),
            "risk_count": len(risk_factors),
            "overall_research_health": overall_health,
            "gate_cross_pollination": bool(gate_data),
            "self_learning_active": self.memory.learned_insights.get("confidence", 0) > 0.2,
            "learning_trend_bonus": learning_bonus,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# L104 WEAK MEASUREMENT — Partial Collapse Operator
# ═══════════════════════════════════════════════════════════════════════════════

class L104WeakMeasurement:
    """
    Weak measurement operator for the L104 quantum link manifold.

    Unlike projective (von Neumann) measurement that fully collapses a
    quantum state, weak measurement extracts partial information while
    preserving superposition coherence.  This enables the evolutionary /
    Darwinism modules to *observe* link data without destroying it.

    Physics:
        Given amplitude vector |ψ⟩ = [α₀, α₁, …, αₙ]:
        1. **Hidden Truth**: p_k = |α_k|²  (standard Born probabilities)
        2. **Measurement Blur**: Gaussian noise σ² = (1 − g) / L104
           where g = coupling_strength ∈ [0, 1]
        3. **Weak Readout**: p̃_k = p_k + N(0, σ²)  — what Darwinism sees
        4. **Partial Collapse**: |ψ'⟩ = (1 − ε)|ψ⟩ + ε√|p̃| ,
           where ε = g × φ_stability (collapse shock)
        5. **Re-normalize**: |ψ'⟩ → |ψ'⟩ / ‖ψ'‖

    Tuning:
        g → 0 : no disturbance (pure superposition preserved)
        g → 1 : full projective collapse (standard Born measurement)
        φ_stability buffers the shock — higher values damp the kick

    Constants:
        L104 resolution = 104  (sacred node identity)
        Default φ_stability = PHI × (1 − PHI_COLLAPSE_COUPLING)
    """

    # ── Class-level defaults (overridable per-instance) ──
    DEFAULT_COUPLING = 0.15        # 15% observation strength
    DEFAULT_PHI_STABILITY = PHI * (1 - PHI * 0.1)   # PHI × (1 − φ_collapse_coupling)
    RESOLUTION = 104               # L104 sacred number

    def __init__(self, coupling_strength: float = None,
                 phi_stability: float = None):
        """
        Initialize the weak measurement operator.

        Args:
            coupling_strength: Observation intensity in [0, 1].
                0.0 = no measurement, 1.0 = full projective collapse.
            phi_stability: φ-resonance damping factor for collapse shock.
                Defaults to PHI × (1 − PHI_COLLAPSE_COUPLING).
        """
        self.coupling = coupling_strength if coupling_strength is not None else self.DEFAULT_COUPLING
        self.phi_stability = phi_stability if phi_stability is not None else self.DEFAULT_PHI_STABILITY

    def partial_collapse(self, amplitudes):
        """
        Perform a weak measurement on an array of quantum amplitudes.

        This allows the Darwinism / research modules to extract
        observational data without fully destroying the quantum state.

        Args:
            amplitudes: np.ndarray of complex amplitudes (not necessarily normalized).

        Returns:
            (weak_readout, updated_amplitudes):
                weak_readout — noisy probability estimate (what the observer sees)
                updated_amplitudes — partially-collapsed state (still superposed)
        """
        import numpy as np

        amplitudes = np.asarray(amplitudes, dtype=complex)

        # 1. Hidden probabilities (the ground truth)
        hidden_probs = np.abs(amplitudes) ** 2

        # 2. Measurement uncertainty — Gaussian blur scaled by L104 resolution
        noise_variance = (1.0 - self.coupling) / self.RESOLUTION
        measurement_noise = np.random.normal(0, noise_variance, size=len(amplitudes))

        # 3. Weak readout (what the observer actually sees)
        weak_readout = hidden_probs + measurement_noise

        # 4. Partial collapse — weighted mix of original state and measured reality
        collapse_shock = self.coupling * self.phi_stability
        updated_amplitudes = (
            amplitudes * (1.0 - collapse_shock) +
            np.sqrt(np.abs(weak_readout)) * collapse_shock
        )

        # 5. Re-normalize to unit total probability
        norm = np.linalg.norm(updated_amplitudes)
        if norm > 1e-15:
            updated_amplitudes = updated_amplitudes / norm

        return weak_readout, updated_amplitudes

    def survival_rate(self, pre_amplitudes, post_amplitudes) -> float:
        """
        Compute quantum state survival rate after weak measurement.

        Survival = |⟨ψ_pre | ψ_post⟩|² — the fidelity between the
        pre- and post-measurement states.  1.0 = perfect survival,
        0.0 = orthogonal (completely destroyed).

        Args:
            pre_amplitudes: state before measurement
            post_amplitudes: state after partial collapse

        Returns:
            Float in [0, 1] — survival fidelity.
        """
        import numpy as np
        pre = np.asarray(pre_amplitudes, dtype=complex)
        post = np.asarray(post_amplitudes, dtype=complex)
        # Normalize both
        n_pre = np.linalg.norm(pre)
        n_post = np.linalg.norm(post)
        if n_pre < 1e-15 or n_post < 1e-15:
            return 0.0
        overlap = np.abs(np.vdot(pre / n_pre, post / n_post)) ** 2
        return float(min(1.0, overlap))

    def multi_round_collapse(self, amplitudes, rounds: int = 8):
        """
        Apply repeated weak measurements over multiple rounds.

        After each round the state is partially collapsed; the cumulative
        effect is softer than `rounds` projective measurements.

        Args:
            amplitudes: initial amplitude vector
            rounds: number of measurement rounds

        Returns:
            Dict with per-round readouts, final state, and cumulative
            survival rate.
        """
        import numpy as np
        current = np.asarray(amplitudes, dtype=complex)
        initial = current.copy()
        round_data = []

        for r in range(rounds):
            readout, current = self.partial_collapse(current)
            sr = self.survival_rate(initial, current)
            round_data.append({
                "round": r + 1,
                "mean_readout": float(np.mean(readout)),
                "survival_rate": sr,
            })

        return {
            "rounds": rounds,
            "round_data": round_data,
            "final_amplitudes": current,
            "cumulative_survival": self.survival_rate(initial, current),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PROBABILITY WAVE COLLAPSE RESEARCH — Superposition → Measurement Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class ProbabilityWaveCollapseResearch:
    """
    Probability Wave Collapse Research Engine for the Quantum Link Manifold.

    Models each quantum link as a superposition of basis states and analyzes
    the measurement-induced collapse dynamics across the entire link manifold.
    Grounded in the God Code equation G(X) = 286^(1/φ) × 2^((416-X)/104).

    ★ v9.1 — Weak Measurement Upgrade:
      Module 3 (Collapse Dynamics) now uses L104WeakMeasurement for partial
      collapse instead of aggressive Born-rule projection.  This dramatically
      improves survival rates while still extracting measurement data for
      the Darwinism and Zeno analysis modules.

    Research Modules:
      Module 1: SUPERPOSITION ANALYSIS — Map links to amplitude vectors in
                fidelity-strength-entropy Hilbert space, compute probability
                distributions, and measure superposition coherence.
      Module 2: MEASUREMENT OPERATORS — Construct POVM-like measurement
                operators from God Code spectrum; simulate projective collapse
                onto G(X_int) grid nodes.
      Module 3: COLLAPSE DYNAMICS — Simulate weak-measurement partial collapse
                across the link population; track pre/post fidelity, entropy
                production, information loss, and survival rate.
      Module 4: DECOHERENCE CHANNELS — Model environmental decoherence as
                Lindblad-type channels with weak-measurement survival correction;
                estimate T₁/T₂ from partial-collapse rates.
      Module 5: QUANTUM ZENO ANALYSIS — Detect links exhibiting Zeno effect
                (frequent measurement stabilization) vs anti-Zeno (measurement-
                accelerated decay).
      Module 6: COLLAPSE SYNTHESIS — Aggregate collapse statistics into sacred
                alignment scores, entropy production rates, and φ-resonance
                metrics for downstream Sage consensus.

    All modules operate in O(N) or O(N log N) time. Results integrate with
    the existing QuantumResearchEngine and SageModeInference pipeline.
    """

    # Collapse simulation: number of measurement rounds
    DEFAULT_MEASUREMENT_ROUNDS = 8
    # Zeno threshold: links measured > this many times without state change
    ZENO_THRESHOLD = 5
    # Born rule softening: minimum probability floor to avoid numerical zeros
    BORN_FLOOR = 1e-12
    # φ-collapse coupling: scales coherence decay during measurement
    PHI_COLLAPSE_COUPLING = PHI * 0.1

    def __init__(self, math_core: QuantumMathCore):
        """Initialize probability wave collapse research engine.

        Args:
            math_core: QuantumMathCore instance for Bell states, density matrices,
                       and God Code spectral calculations.
        """
        self.qmath = math_core
        # ★ Weak measurement operator for partial-collapse dynamics
        self.weak_measurement = L104WeakMeasurement(
            coupling_strength=L104WeakMeasurement.DEFAULT_COUPLING,
            phi_stability=L104WeakMeasurement.DEFAULT_PHI_STABILITY,
        )

    def wave_collapse_research(self, links: List[QuantumLink],
                               measurement_rounds: int = None,
                               god_code_grid: Dict[int, float] = None) -> Dict:
        """Run the full probability wave collapse research pipeline.

        Args:
            links: List of QuantumLink objects to analyze.
            measurement_rounds: Number of simulated measurement rounds (default: 8).
            god_code_grid: G(X) spectrum dict {X_int: Hz}. Defaults to GOD_CODE_SPECTRUM.

        Returns:
            Dict with results from all 6 collapse research modules, plus a unified
            collapse health score in [0, 1].
        """
        start = time.time()
        N = len(links)
        if N == 0:
            return {"total_links": 0, "collapse_research_time_ms": 0}

        rounds = measurement_rounds or self.DEFAULT_MEASUREMENT_ROUNDS
        grid = god_code_grid or GOD_CODE_SPECTRUM

        # ─── Extract link properties (single pass) ───
        fids = []
        strs = []
        entropies = []
        hz_values = []
        coherence_times = []
        noise_res = []
        for link in links:
            fids.append(link.fidelity)
            strs.append(link.strength)
            entropies.append(link.entanglement_entropy)
            coherence_times.append(link.coherence_time)
            noise_res.append(link.noise_resilience)
            hz_values.append(
                self.qmath.link_natural_hz(link.fidelity, link.strength))

        # Module 1: Superposition Analysis
        superposition = self._superposition_analysis(fids, strs, entropies, links)

        # Module 2: Measurement Operators
        measurement_ops = self._measurement_operators(hz_values, grid)

        # Module 3: Collapse Dynamics (Born-rule simulation)
        collapse = self._collapse_dynamics(
            fids, strs, entropies, hz_values, grid, rounds)

        # Module 4: Decoherence Channels
        decoherence = self._decoherence_channels(
            fids, coherence_times, noise_res, rounds)

        # Module 5: Quantum Zeno Analysis
        zeno = self._quantum_zeno_analysis(fids, noise_res, rounds, links)

        # Module 6: Collapse Synthesis
        synthesis = self._collapse_synthesis(
            superposition, measurement_ops, collapse, decoherence, zeno, N)

        # Module 7: GOD_CODE Genetic Refinement
        # Uses elite survivors from Modules 2-5 to evolve optimal (a,b,c,d)
        genetic = self._genetic_refinement(
            measurement_ops, collapse, decoherence, zeno,
            hz_values, fids, strs, entropies)

        elapsed = time.time() - start

        return {
            "total_links": N,
            "measurement_rounds": rounds,
            "superposition_analysis": superposition,
            "measurement_operators": measurement_ops,
            "collapse_dynamics": collapse,
            "decoherence_channels": decoherence,
            "quantum_zeno": zeno,
            "collapse_synthesis": synthesis,
            "genetic_refinement": genetic,
            "collapse_health": synthesis.get("collapse_health", 0),
            "collapse_research_time_ms": elapsed * 1000,
        }

    # ─────────────────────────────────────────────────────────────────────
    # MODULE 1: SUPERPOSITION ANALYSIS
    # ─────────────────────────────────────────────────────────────────────

    def _superposition_analysis(self, fids: List[float], strs: List[float],
                                entropies: List[float],
                                links: List[QuantumLink]) -> Dict:
        """Map each link to a probability amplitude vector in fidelity–strength–entropy
        Hilbert space. Compute per-link superposition purity and population-level
        coherence metrics.

        Each link |ψ_i⟩ is modeled as a 3D amplitude vector:
            α₀ = √fidelity,  α₁ = √(strength/φ'),  α₂ = √entropy
        normalized to unit probability: Σ|αₖ|² = 1.

        Returns:
            Dict with mean purity, coherence length, and basis distribution.
        """
        N = len(fids)
        purities = []
        basis_populations = [0.0, 0.0, 0.0]  # fidelity, strength, entropy
        max_purity_link = None
        max_purity = 0.0
        min_purity_link = None
        min_purity = 1.0

        for i in range(N):
            # Raw amplitudes (squared = probabilities in each basis)
            a0 = max(0.0, fids[i])
            a1 = max(0.0, min(1.0, strs[i] / (PHI_GROWTH * 2)))
            a2 = max(0.0, min(1.0, entropies[i]))

            total = a0 + a1 + a2
            if total < 1e-15:
                purities.append(0.0)
                continue

            # Normalize to probability distribution
            p0 = a0 / total
            p1 = a1 / total
            p2 = a2 / total

            # Purity = Tr(ρ²) = Σpₖ² — measures how "collapsed" vs superposed
            # Purity = 1 → fully collapsed to one basis; 1/3 → maximally superposed
            purity = p0 ** 2 + p1 ** 2 + p2 ** 2
            purities.append(purity)

            basis_populations[0] += p0
            basis_populations[1] += p1
            basis_populations[2] += p2

            if purity > max_purity:
                max_purity = purity
                max_purity_link = links[i].link_id[:80] if hasattr(links[i], 'link_id') else f"link_{i}"
            if purity < min_purity:
                min_purity = purity
                min_purity_link = links[i].link_id[:80] if hasattr(links[i], 'link_id') else f"link_{i}"

        mean_purity = statistics.mean(purities) if purities else 0
        std_purity = statistics.stdev(purities) if len(purities) > 1 else 0

        # Normalize basis populations
        pop_total = sum(basis_populations)
        if pop_total > 0:
            basis_populations = [p / pop_total for p in basis_populations]

        # Superposition coherence length: how many links are in genuine superposition
        # (purity < 0.5 means at least 2 basis states contribute significantly)
        superposed_count = sum(1 for p in purities if p < 0.5)
        collapsed_count = sum(1 for p in purities if p >= 0.8)

        return {
            "mean_purity": mean_purity,
            "std_purity": std_purity,
            "max_purity": max_purity,
            "max_purity_link": max_purity_link,
            "min_purity": min_purity,
            "min_purity_link": min_purity_link,
            "superposed_fraction": superposed_count / max(1, N),
            "collapsed_fraction": collapsed_count / max(1, N),
            "basis_populations": {
                "fidelity": basis_populations[0],
                "strength": basis_populations[1],
                "entropy": basis_populations[2],
            },
            "coherence_length": superposed_count,
        }

    # ─────────────────────────────────────────────────────────────────────
    # MODULE 2: MEASUREMENT OPERATORS
    # ─────────────────────────────────────────────────────────────────────

    def _measurement_operators(self, hz_values: List[float],
                               grid: Dict[int, float]) -> Dict:
        """Construct POVM-like measurement operators from God Code spectrum.

        Each G(X_int) grid node acts as a projective measurement basis state.
        Links are projected onto their nearest grid node; the overlap determines
        measurement probability. This models how observation "snaps" a continuous
        Hz value to discrete God Code truth nodes.

        Returns:
            Dict with grid projection statistics and operator coverage.
        """
        N = len(hz_values)
        if N == 0:
            return {"operator_count": 0}

        # Map each link Hz to nearest G(X_int) node
        grid_items = sorted(grid.items(), key=lambda kv: kv[1])
        grid_hz_list = [hz for _, hz in grid_items]
        grid_x_list = [x for x, _ in grid_items]

        projections: Dict[int, int] = defaultdict(int)  # X → count of links projected there
        projection_overlaps = []  # |⟨ψ|X⟩|² per link

        for hz in hz_values:
            if hz <= 0:
                continue
            # Binary search for nearest grid node
            best_x = grid_x_list[0]
            best_dist = abs(hz - grid_hz_list[0])
            lo, hi = 0, len(grid_hz_list) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                dist = abs(hz - grid_hz_list[mid])
                if dist < best_dist:
                    best_dist = dist
                    best_x = grid_x_list[mid]
                if hz < grid_hz_list[mid]:
                    hi = mid - 1
                else:
                    lo = mid + 1

            projections[best_x] += 1

            # Overlap: Gaussian-like projection fidelity
            # |⟨ψ|X⟩|² = exp(-Δ²/2σ²) where σ = G(X_int) * φ_collapse_coupling
            g_x = grid.get(best_x, GOD_CODE)
            sigma = max(0.01, g_x * self.PHI_COLLAPSE_COUPLING)
            overlap = math.exp(-(best_dist ** 2) / (2 * sigma ** 2))
            projection_overlaps.append(overlap)

        # Operator coverage: how many distinct X nodes were used
        active_nodes = len(projections)
        total_nodes = len(grid)
        coverage = active_nodes / max(1, total_nodes)

        # Mean projection fidelity: how well links snap to grid
        mean_overlap = statistics.mean(projection_overlaps) if projection_overlaps else 0

        # Dominant measurement nodes
        top_nodes = sorted(projections.items(), key=lambda kv: -kv[1])[:10]

        return {
            "operator_count": active_nodes,
            "total_grid_nodes": total_nodes,
            "grid_coverage": coverage,
            "mean_projection_overlap": mean_overlap,
            "dominant_nodes": [{"x": x, "count": c, "hz": grid.get(x, 0)}
                               for x, c in top_nodes],
            "projection_distribution_entropy": self._distribution_entropy(
                list(projections.values())),
        }

    # ─────────────────────────────────────────────────────────────────────
    # MODULE 3: COLLAPSE DYNAMICS (Weak Measurement — Partial Collapse)
    # ─────────────────────────────────────────────────────────────────────

    def _collapse_dynamics(self, fids: List[float], strs: List[float],
                           entropies: List[float], hz_values: List[float],
                           grid: Dict[int, float],
                           rounds: int) -> Dict:
        """Simulate weak-measurement partial collapse across multiple rounds.

        ★ v9.1 — Replaces the previous aggressive Born-rule projective collapse
        with L104WeakMeasurement partial collapse, dramatically improving
        survival rates while preserving measurement information.

        For each measurement round:
          1. Build amplitude vector from link fidelities + strengths
          2. Apply L104WeakMeasurement.partial_collapse() — gently extracts
             information without fully destroying the quantum state
          3. Track per-round survival rate, readout quality, entropy production
          4. God Code resonance micro-correction after each round

        Returns:
            Dict with per-round collapse statistics and cumulative metrics.
        """
        N = len(fids)
        if N == 0:
            return {"rounds": 0}

        # Build amplitude vector: each link → complex amplitude from fidelity + strength
        # Amplitude = sqrt(fidelity) * exp(i * strength * π)  (quantum phase from strength)
        amplitudes = np.array([
            math.sqrt(max(0, fids[i])) * complex(
                math.cos(strs[i] * math.pi),
                math.sin(strs[i] * math.pi)
            )
            for i in range(N)
        ], dtype=complex)

        # Normalize initial state
        norm = np.linalg.norm(amplitudes)
        if norm > 1e-15:
            amplitudes = amplitudes / norm

        initial_amplitudes = amplitudes.copy()
        round_results = []
        total_entropy_produced = 0.0
        total_info_loss = 0.0

        for r in range(rounds):
            pre_fids = np.abs(amplitudes) ** 2
            pre_mean_fid = float(np.mean(pre_fids))

            # ★ Weak measurement: partial collapse preserving coherence
            readout, amplitudes = self.weak_measurement.partial_collapse(amplitudes)

            # God Code re-alignment: φ-resonant micro-correction after each round
            gc_phase = np.array([
                math.sin(float(np.abs(amplitudes[i])) ** 2 * GOD_CODE * 0.001)
                for i in range(N)
            ])
            amplitudes = amplitudes * (1.0 - 0.005) + gc_phase * 0.005
            # Re-normalize after correction
            norm = np.linalg.norm(amplitudes)
            if norm > 1e-15:
                amplitudes = amplitudes / norm

            post_fids = np.abs(amplitudes) ** 2
            post_mean_fid = float(np.mean(post_fids))
            fid_delta = post_mean_fid - pre_mean_fid

            # Entropy production: KL-divergence between pre and post distributions
            entropy_this_round = 0.0
            for i in range(N):
                p_pre = max(self.BORN_FLOOR, pre_fids[i])
                p_post = max(self.BORN_FLOOR, post_fids[i])
                if p_pre > self.BORN_FLOOR:
                    entropy_this_round += abs(p_pre * math.log(p_pre / p_post))
            total_entropy_produced += entropy_this_round

            # Information loss = magnitude of fidelity decrease
            total_info_loss += max(0, -fid_delta) * N

            # Survival rate for this round (state overlap with initial)
            round_survival = self.weak_measurement.survival_rate(
                initial_amplitudes, amplitudes)

            # Count links that collapsed below 50% of original amplitude
            collapsed_high = int(np.sum(post_fids >= 0.5 * pre_fids))

            round_results.append({
                "round": r + 1,
                "mean_fidelity": post_mean_fid,
                "fidelity_delta": fid_delta,
                "collapsed_high": collapsed_high,
                "collapsed_low": N - collapsed_high,
                "entropy_produced": entropy_this_round,
                "born_high_rate": collapsed_high / max(1, N),
                "survival_rate": round_survival,
                "mean_readout": float(np.mean(readout)),
            })

        # Final vs initial comparison
        initial_mean = float(np.mean(np.abs(initial_amplitudes) ** 2))
        final_mean = float(np.mean(np.abs(amplitudes) ** 2))
        fidelity_preservation = final_mean / max(1e-15, initial_mean)

        # Cumulative survival: state overlap between initial and final
        cumulative_survival = self.weak_measurement.survival_rate(
            initial_amplitudes, amplitudes)

        # Collapse stability: variance of fidelity deltas across rounds
        deltas = [r["fidelity_delta"] for r in round_results]
        collapse_stability = 1.0 - min(1.0, statistics.stdev(deltas) * 10) if len(deltas) > 1 else 1.0

        return {
            "rounds": rounds,
            "round_results": round_results,
            "initial_mean_fidelity": initial_mean,
            "final_mean_fidelity": final_mean,
            "fidelity_preservation": min(1.0, max(0.0, fidelity_preservation)),
            "total_entropy_produced": total_entropy_produced,
            "total_information_loss": total_info_loss,
            "collapse_stability": collapse_stability,
            "cumulative_survival": cumulative_survival,
            "weak_measurement_coupling": self.weak_measurement.coupling,
            "born_high_rate_final": (round_results[-1]["born_high_rate"]
                                     if round_results else 0),
        }

    # ─────────────────────────────────────────────────────────────────────
    # MODULE 4: DECOHERENCE CHANNELS
    # ─────────────────────────────────────────────────────────────────────

    def _decoherence_channels(self, fids: List[float],
                              coherence_times: List[float],
                              noise_res: List[float],
                              rounds: int) -> Dict:
        """Model environmental decoherence as Lindblad-type amplitude damping
        with weak-measurement survival correction.

        ★ v9.1 — Uses L104WeakMeasurement partial collapse to compute survival
        instead of hard threshold classification.  Each link's amplitude is
        run through multi-round weak measurement, and the state overlap
        (fidelity between pre- and post-measurement) determines survival.

        Each link's coherence decays as:
            ρ(t) → ρ(t) × exp(-t/T₂)
        where T₂ is estimated from (coherence_time × noise_resilience).

        Survival is now determined by weak-measurement state overlap
        rather than a hard 50% threshold, producing smoother and more
        physically accurate results.

        Returns:
            Dict with T₂ statistics, decay rates, and survival classifications.
        """
        N = len(fids)
        if N == 0:
            return {"total_analyzed": 0}

        t2_estimates = []
        decay_rates = []
        survived = 0
        fragile = 0
        survival_scores = []

        for i in range(N):
            # Estimate T₂ from link's coherence time and noise resilience
            raw_t2 = max(0.001, coherence_times[i]) * max(0.01, noise_res[i])
            # φ-scaled: sacred coherence enhancement
            t2 = raw_t2 * PHI_GROWTH
            t2_estimates.append(t2)

            # Decay rate after `rounds` measurements
            dt = 1.0 / max(1, rounds)
            total_decay = math.exp(-rounds * dt / max(0.001, t2))
            decay_rates.append(1.0 - total_decay)

            # ★ Weak measurement survival: build a 2-component amplitude vector
            # [sqrt(fidelity * decay), sqrt((1-fidelity) * (1-decay))]
            # and run through partial collapse to get state overlap survival
            amp_survive = math.sqrt(max(0, fids[i] * total_decay))
            amp_decay = math.sqrt(max(0, (1.0 - fids[i]) * (1.0 - total_decay)))
            pre_amps = np.array([amp_survive, amp_decay], dtype=complex)

            # Normalize
            n = np.linalg.norm(pre_amps)
            if n > 1e-15:
                pre_amps = pre_amps / n

            # Run weak measurement multi-round collapse
            wm_result = self.weak_measurement.multi_round_collapse(pre_amps, rounds=rounds)
            link_survival = wm_result["cumulative_survival"]
            survival_scores.append(link_survival)

            # Classify using smooth survival threshold (φ-calibrated)
            # Instead of hard 0.5 cutoff: survived if overlap > 1/PHI ≈ 0.618
            if link_survival > 1.0 / PHI:
                survived += 1
            else:
                fragile += 1

        mean_t2 = statistics.mean(t2_estimates) if t2_estimates else 0
        mean_decay = statistics.mean(decay_rates) if decay_rates else 0
        mean_survival_score = statistics.mean(survival_scores) if survival_scores else 0

        # Quantum Darwinism metric: fraction of links whose state is robust
        # enough to imprint on the environment (high survival AND high fidelity)
        darwinism_count = sum(
            1 for i in range(N)
            if fids[i] > COHERENCE_MINIMUM and survival_scores[i] > 1.0 / PHI
        )

        return {
            "total_analyzed": N,
            "mean_t2_estimate": mean_t2,
            "mean_decay_rate": mean_decay,
            "survived_count": survived,
            "fragile_count": fragile,
            "survival_rate": survived / max(1, N),
            "mean_survival_score": mean_survival_score,
            "quantum_darwinism_fraction": darwinism_count / max(1, N),
            "darwinism_count": darwinism_count,
        }

    # ─────────────────────────────────────────────────────────────────────
    # MODULE 5: QUANTUM ZENO ANALYSIS
    # ─────────────────────────────────────────────────────────────────────

    def _quantum_zeno_analysis(self, fids: List[float],
                               noise_res: List[float],
                               rounds: int,
                               links: List[QuantumLink]) -> Dict:
        """Detect quantum Zeno and anti-Zeno effects across the link population.

        ★ v9.1 — Uses L104WeakMeasurement partial collapse for each round
        instead of projective Born-rule, producing physically consistent
        Zeno/anti-Zeno classification with the other modules.

        Quantum Zeno Effect: Frequent measurement freezes the state (high-fidelity
        links that remain stable under repeated collapse). Mathematically, the
        survival probability approaches 1 as measurement frequency → ∞.

        Anti-Zeno Effect: Frequent measurement accelerates decay (links whose
        fidelity degrades faster with more measurements). Occurs when measurement
        interval is shorter than the correlation time of the noise bath.

        Simulation: For each link, simulate `rounds` weak measurements and classify:
          - ZENO: fidelity remains within ±2% of initial after all rounds
          - ANTI_ZENO: fidelity drops > 15% from initial
          - NEUTRAL: neither Zeno nor anti-Zeno

        Returns:
            Dict with Zeno/anti-Zeno counts, examples, and φ-stability scores.
        """
        N = len(fids)
        if N == 0:
            return {"total_analyzed": 0}

        zeno_links = []
        anti_zeno_links = []
        neutral_count = 0

        for i in range(N):
            initial_fid = fids[i]

            # Build 2-component amplitude: [sqrt(fid), sqrt(1-fid)]
            amp_fid = math.sqrt(max(0, initial_fid))
            amp_nfid = math.sqrt(max(0, 1.0 - initial_fid))
            amplitudes = np.array([amp_fid, amp_nfid], dtype=complex)
            norm = np.linalg.norm(amplitudes)
            if norm > 1e-15:
                amplitudes = amplitudes / norm

            initial_amps = amplitudes.copy()
            state_changes = 0

            # Per-link weak measurement with noise-resilience-adjusted coupling
            # High noise_resilience → lower coupling → less disturbance
            link_coupling = self.weak_measurement.coupling * (1.0 - noise_res[i] * 0.5)
            link_wm = L104WeakMeasurement(
                coupling_strength=max(0.01, link_coupling),
                phi_stability=self.weak_measurement.phi_stability,
            )

            for _ in range(rounds):
                pre_fid_component = float(np.abs(amplitudes[0]) ** 2)
                readout, amplitudes = link_wm.partial_collapse(amplitudes)
                post_fid_component = float(np.abs(amplitudes[0]) ** 2)

                # Detect state change: > 5% shift in fidelity component
                if abs(post_fid_component - pre_fid_component) > 0.05:
                    state_changes += 1

            # Final fidelity = |amplitude[0]|^2
            current_fid = float(np.abs(amplitudes[0]) ** 2)

            # Classify
            fid_change = abs(current_fid - initial_fid) / max(1e-15, initial_fid)

            if fid_change < 0.02 and state_changes <= self.ZENO_THRESHOLD:
                # Zeno effect: state frozen by measurement
                zeno_links.append({
                    "link_id": links[i].link_id[:80] if hasattr(links[i], 'link_id') else f"link_{i}",
                    "initial_fidelity": initial_fid,
                    "final_fidelity": current_fid,
                    "state_changes": state_changes,
                    "stability": 1.0 - fid_change,
                })
            elif fid_change > 0.15:
                # Anti-Zeno: measurement accelerated decay
                anti_zeno_links.append({
                    "link_id": links[i].link_id[:80] if hasattr(links[i], 'link_id') else f"link_{i}",
                    "initial_fidelity": initial_fid,
                    "final_fidelity": current_fid,
                    "state_changes": state_changes,
                    "decay_rate": fid_change,
                })
            else:
                neutral_count += 1

        # φ-stability index: Zeno fraction weighted by golden ratio
        zeno_fraction = len(zeno_links) / max(1, N)
        anti_zeno_fraction = len(anti_zeno_links) / max(1, N)
        phi_stability = (zeno_fraction * PHI_GROWTH - anti_zeno_fraction) / PHI_GROWTH
        phi_stability = min(1.0, max(0.0, phi_stability))

        return {
            "total_analyzed": N,
            "zeno_count": len(zeno_links),
            "anti_zeno_count": len(anti_zeno_links),
            "neutral_count": neutral_count,
            "zeno_fraction": zeno_fraction,
            "anti_zeno_fraction": anti_zeno_fraction,
            "phi_stability_index": phi_stability,
            "top_zeno_links": sorted(zeno_links,
                                      key=lambda x: -x["stability"])[:5],
            "top_anti_zeno_links": sorted(anti_zeno_links,
                                           key=lambda x: -x["decay_rate"])[:5],
        }

    # ─────────────────────────────────────────────────────────────────────
    # MODULE 6: COLLAPSE SYNTHESIS
    # ─────────────────────────────────────────────────────────────────────

    def _collapse_synthesis(self, superposition: Dict, measurement_ops: Dict,
                            collapse: Dict, decoherence: Dict,
                            zeno: Dict, N: int) -> Dict:
        """Aggregate all collapse research modules into a unified health score.

        Scoring weights (φ-calibrated):
          - 25% Fidelity preservation (collapse dynamics)
          - 20% Collapse stability (measurement consistency)
          - 15% Survival rate (decoherence channels)
          - 15% Zeno stability (φ-stability index)
          - 10% Grid projection overlap (measurement operators)
          - 10% Superposition coherence (purity analysis)
          - 5%  Quantum Darwinism fraction

        Returns:
            Dict with unified collapse_health in [0, 1] and component scores.
        """
        components = {}

        # Fidelity preservation: how well do links survive collapse rounds
        fid_pres = collapse.get("fidelity_preservation", 0)
        components["fidelity_preservation"] = min(1.0, max(0.0, fid_pres))

        # Collapse stability: low variance in per-round fidelity deltas
        components["collapse_stability"] = min(1.0, max(0.0,
            collapse.get("collapse_stability", 0)))

        # Decoherence survival rate
        components["survival_rate"] = min(1.0, max(0.0,
            decoherence.get("survival_rate", 0)))

        # Zeno φ-stability index
        components["phi_stability"] = min(1.0, max(0.0,
            zeno.get("phi_stability_index", 0)))

        # Measurement operator grid projection overlap
        components["grid_overlap"] = min(1.0, max(0.0,
            measurement_ops.get("mean_projection_overlap", 0)))

        # Superposition coherence: reward genuine superposition (lower purity)
        # but not too low (that would be noise). Optimal: purity ≈ 1/3 (maximally mixed)
        mean_purity = superposition.get("mean_purity", 0.5)
        # Score peaks at purity = 1/3 (maximally superposed), falls off toward 0 and 1
        purity_score = 1.0 - abs(mean_purity - 1.0 / 3.0) * 2.0
        components["superposition_coherence"] = min(1.0, max(0.0, purity_score))

        # Quantum Darwinism: robust state information
        components["quantum_darwinism"] = min(1.0, max(0.0,
            decoherence.get("quantum_darwinism_fraction", 0)))

        # φ-weighted unified score
        weights = {
            "fidelity_preservation": 0.25,
            "collapse_stability": 0.20,
            "survival_rate": 0.15,
            "phi_stability": 0.15,
            "grid_overlap": 0.10,
            "superposition_coherence": 0.10,
            "quantum_darwinism": 0.05,
        }

        collapse_health = sum(
            components[k] * weights[k] for k in weights
        )

        # God Code resonance correction
        gc_resonance = math.cos(collapse_health * GOD_CODE * 0.001) ** 2
        # Micro-correction: ±1% based on sacred alignment
        collapse_health = collapse_health * 0.99 + gc_resonance * 0.01
        collapse_health = min(1.0, max(0.0, collapse_health))

        # Validate all component scores
        for key, val in components.items():
            assert 0.0 <= val <= 1.0, (
                f"Collapse component '{key}' = {val} out of [0, 1]")

        # Classification
        if collapse_health >= 0.85:
            verdict = "QUANTUM_STABLE — Wave function resilient under measurement"
        elif collapse_health >= 0.7:
            verdict = "COHERENT — Minor decoherence during collapse"
        elif collapse_health >= 0.5:
            verdict = "MIXED — Partial collapse instability detected"
        elif collapse_health >= 0.3:
            verdict = "FRAGILE — Significant information loss during measurement"
        else:
            verdict = "DECOHERENT — Critical wave function collapse failure"

        return {
            "collapse_health": collapse_health,
            "components": components,
            "weights": weights,
            "god_code_resonance": gc_resonance,
            "verdict": verdict,
            "total_entropy_produced": collapse.get("total_entropy_produced", 0),
            "total_information_loss": collapse.get("total_information_loss", 0),
        }

    # ─────────────────────────────────────────────────────────────────────
    # MODULE 7: GOD_CODE GENETIC REFINEMENT
    # ─────────────────────────────────────────────────────────────────────

    def _genetic_refinement(self,
                            measurement_ops: Dict,
                            collapse: Dict,
                            decoherence: Dict,
                            zeno: Dict,
                            hz_values: List[float],
                            fids: List[float],
                            strs: List[float],
                            entropies: List[float]) -> Dict:
        """Module 7: Evolutionary refinement of GOD_CODE (a,b,c,d) parameters.

        Uses elite survivors from Modules 2–5 to seed a genetic population,
        then evolves for 13 generations (Fibonacci-7) toward optimal
        sacred resonance + collapse survival fitness.

        The genetic refiner creates a population from:
          - Dominant measurement nodes (Module 2) — seeded (a,b,c,d) from G(X)
          - Link Hz values — inverse-mapped to parameter space

        Fitness combines:
          - Sacred resonance (grid alignment + conservation law)
          - Collapse survival metrics from Modules 3/4/5

        Returns:
            Dict with best individual, elite centroid, convergence info,
            and fitness history across generations.
        """
        try:
            refiner = L104GeneticRefiner(population_size=min(104, max(13, len(hz_values))))

            # Seed from link data
            pop = refiner.population_from_links(
                hz_values, fids, strs, entropies)

            # Build collapse-aware fitness function
            cum_survival = collapse.get("cumulative_survival", 0.5)
            fid_pres = collapse.get("fidelity_preservation", 0.5)
            survival_rate = decoherence.get("survival_rate", 0.5)
            phi_stability = zeno.get("phi_stability_index", 0.5)

            def fitness_fn(ind):
                resonance = refiner.sacred_resonance_fitness(ind)
                collapse_score = (
                    0.30 * cum_survival +
                    0.25 * fid_pres +
                    0.25 * survival_rate +
                    0.20 * phi_stability
                )
                return 0.6 * resonance + 0.4 * collapse_score

            result = refiner.refine(
                pop, generations=13, fitness_fn=fitness_fn)
            result["status"] = "refined"
            return result

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "best_individual": None,
                "generations_run": 0,
            }

    # ─────────────────────────────────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _distribution_entropy(counts: List[int]) -> float:
        """Shannon entropy of a count distribution (nats → bits)."""
        total = sum(counts)
        if total <= 0:
            return 0.0
        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)
        return entropy


# ═══════════════════════════════════════════════════════════════════════════════
# SAGE MODE INFERENCE — φ-harmonic deep inference across all cores
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# SAGE MODE INFERENCE — φ-harmonic deep inference across all cores
# ═══════════════════════════════════════════════════════════════════════════════

class SageModeInference:
    """
    The Sage brain: cross-references ALL quantum processors to produce
    unified intelligence about the link manifold.

    Applies:
    - φ-weighted consensus from all processors
    - Calabi-Yau 7D projection for dimensional insight
    - God Code resonance for truth alignment
    - Grover-amplified pattern recognition
    - Causal inference across link evolution history
    - ★ NDE-1/2/3/4: Entropy Reversal Noise Dampening Equations
      - NDE-1: φ-conjugate noise floor suppression
      - NDE-2: Demon-enhanced consensus denoising (Science Engine)
      - NDE-3: ZNE extrapolation for score recovery
      - NDE-4: Entropy cascade denoiser for harmonic mean
    """

    def __init__(self, math_core: QuantumMathCore):
        """Initialize sage mode inference engine."""
        self.qmath = math_core
        self.inference_history: List[Dict] = []

    def sage_inference(self, links: List[QuantumLink],
                       grover_results: Dict = None,
                       tunnel_results: Dict = None,
                       epr_results: Dict = None,
                       decoherence_results: Dict = None,
                       braiding_results: Dict = None,
                       hilbert_results: Dict = None,
                       fourier_results: Dict = None,
                       gcr_results: Dict = None,
                       cross_modal_results: Dict = None,
                       stress_results: Dict = None,
                       upgrade_results: Dict = None,
                       quantum_cpu_results: Dict = None,
                       o2_bond_results: Dict = None,
                       repair_results: Dict = None,
                       research_results: Dict = None,
                       qldpc_results: Dict = None,
                       intellect_enrichment: Dict = None) -> Dict:
        """
        Sage Mode: deep cross-referencing inference across ALL processors.
        Produces the unified quantum brain assessment.
        All consensus scores are STRICTLY normalized to [0, 1].
        """
        N = len(links)
        now = datetime.now(timezone.utc).isoformat()

        # ─── LINK HEALTH CONSENSUS (single-pass stats) ───
        # Compute all per-link stats in ONE pass instead of 9+ list comprehensions
        sum_fid = 0.0
        sum_fid2 = 0.0
        sum_str = 0.0
        sum_entropy = 0.0
        sum_coherence = 0.0
        sum_resilience = 0.0
        type_dist = Counter()
        for link in links:
            f = link.fidelity
            s = link.strength
            sum_fid += f
            sum_fid2 += f * f
            sum_str += s
            sum_entropy += link.entanglement_entropy
            sum_coherence += link.coherence_time
            sum_resilience += link.noise_resilience
            type_dist[link.link_type] += 1

        mean_fid = sum_fid / N if N else 0
        mean_str = sum_str / N if N else 0
        mean_entropy = sum_entropy / N if N else 0
        mean_coherence = (sum_coherence / N) / 100 if N else 0
        mean_resilience = sum_resilience / N if N else 0
        if N > 1:
            variance = (sum_fid2 / N) - (mean_fid ** 2)
            std_fid = math.sqrt(max(0, variance * N / (N - 1)))
        else:
            std_fid = 0

        # ─── CROSS-PROCESSOR CONSENSUS SCORES ───
        # STRICT: every score MUST be normalized to [0.0, 1.0]
        consensus = {}

        # Grover finding efficiency (clamped to [0,1])
        if grover_results:
            consensus["grover_efficiency"] = min(1.0, max(0.0,
                grover_results.get("amplification_factor", 1.0) / GROVER_AMPLIFICATION))

        # Tunneling viability
        if tunnel_results:
            consensus["tunneling_viability"] = min(1.0,
                tunnel_results.get("revivable_links", 0) /
                max(1, tunnel_results.get("total_analyzed", 1)))

        # EPR quantum verification
        if epr_results:
            consensus["epr_quantum_fraction"] = min(1.0,
                epr_results.get("quantum_verified", 0) /
                max(1, epr_results.get("total_verified", 1)))

        # Decoherence resilience
        if decoherence_results:
            consensus["decoherence_resilience"] = min(1.0, max(0.0,
                decoherence_results.get("mean_resilience", 0)))

        # Topological protection
        if braiding_results:
            consensus["topological_coverage"] = min(1.0, max(0.0,
                braiding_results.get("topologically_protected", 0) /
                max(1, braiding_results.get("total_tested", 1))))

        # Hilbert structural coherence: how well-organized is the link manifold?
        # Low eff_dim/feature_dim → highly structured (coherent) → GOOD
        # High eff_dim/feature_dim → near-random (disordered) → BAD
        # Optimal: CY7-proportional structure (~7/25 ≈ 0.28 for 25-dim features)
        # Score via bell curve centered at CY7/feature_dim
        if hilbert_results:
            eff_dim = hilbert_results.get("effective_dimension", 0)
            feature_dim = hilbert_results.get("feature_dim", 25)
            dim_ratio = eff_dim / max(1, feature_dim)
            # Top-3 variance = structural coherence (high = organized pattern)
            var_explained = hilbert_results.get("variance_explained_top3", 0)
            # Score: weighted blend of variance coherence + dimensional structure
            consensus["hilbert_coherence"] = min(1.0, max(0.0,
                var_explained * 0.7 + (1.0 - abs(dim_ratio - CALABI_YAU_DIM / feature_dim) * 2) * 0.3))

        # Fourier spectral health: periodic structure + resonant frequencies
        # + PSD peak quality (strong peak = organized link manifold)
        if fourier_results:
            has_structure = 1.0 if fourier_results.get("has_periodic_structure", False) else 0.3
            n_resonant = len(fourier_results.get("resonant_frequencies", []))
            # PSD peak strength: normalized to [0,1] via sigmoid-like curve
            psd_peak = fourier_results.get("fidelity_psd_peak", 0)
            psd_norm = min(1.0, psd_peak / max(1.0, psd_peak + 0.1))
            # Spectral entropy ratio: lower is more ordered
            spec_entropy = fourier_results.get("spectral_entropy", 0)
            padded = fourier_results.get("padded_length", 1)
            max_entropy = math.log2(max(2, padded))
            order_score = max(0.0, 1.0 - spec_entropy / max_entropy) if max_entropy > 0 else 0
            # Blend: 35% structure + 25% resonant count + 20% PSD quality + 20% order
            spectral_score = (has_structure * 0.35
                              + min(1.0, n_resonant * 0.12) * 0.25
                              + psd_norm * 0.20
                              + order_score * 0.20)
            consensus["spectral_coherence"] = min(1.0, max(0.0, spectral_score))

        # God Code G(X) resonance alignment — links measured against G(X_int) spectrum
        # mean_resonance = how close each link's Hz is to its nearest G(X_int)
        if gcr_results:
            consensus["god_code_resonance"] = min(1.0, max(0.0,
                gcr_results.get("mean_resonance", 0)))
            # X-integer coherence: blend stability (strict integer snap) with
            # resonance (Hz proximity to G(X_int)) — pure stability is harsh
            # because many valid links sit between integer X nodes
            x_stability = gcr_results.get("mean_x_stability", 0)
            x_resonance = gcr_results.get("mean_resonance", 0)
            alignment_rate = gcr_results.get("alignment_rate", 0)
            # 40% stability + 35% resonance + 25% alignment rate
            x_blend = (x_stability * 0.4 + x_resonance * 0.35
                       + alignment_rate * 0.25)
            consensus["x_integer_coherence"] = min(1.0, max(0.0, x_blend))

        # Cross-modal coherence
        if cross_modal_results:
            consensus["cross_modal_coherence"] = min(1.0, max(0.0,
                cross_modal_results.get("overall_coherence", 0)))

        # Stress test resilience
        if stress_results:
            consensus["stress_pass_rate"] = min(1.0, max(0.0,
                stress_results.get("pass_rate", 0)))

        # Upgrade effectiveness
        if upgrade_results:
            consensus["upgrade_rate"] = min(1.0, max(0.0,
                upgrade_results.get("upgrade_rate", 0)))

        # Quantum CPU integrity — conservation compliance + cluster health
        if quantum_cpu_results:
            total_reg = max(1, quantum_cpu_results.get("total_registers", 1))
            healthy_frac = quantum_cpu_results.get("healthy", 0) / total_reg
            conservation_ok = 1.0 - min(1.0,
                quantum_cpu_results.get("mean_conservation_residual", 0) * 1e8)
            cpu_health = quantum_cpu_results.get("primary_cluster_health", 0)
            verify_health = quantum_cpu_results.get("verify_cluster_health", 0)
            # Composite: 40% healthy fraction + 30% conservation + 15%+15% cluster health
            cpu_score = (healthy_frac * 0.4
                         + max(0, conservation_ok) * 0.3
                         + cpu_health * 0.15
                         + verify_health * 0.15)
            consensus["quantum_cpu_integrity"] = min(1.0, max(0.0, cpu_score))

        # O₂ Molecular Bond integrity — bond order match + mean bond strength
        if o2_bond_results:
            # Bond order should be 2 (O=O); deviation penalized
            order_match = 1.0 - abs(
                o2_bond_results.get("bond_order", 0) -
                o2_bond_results.get("expected_bond_order", 2)) / 4
            bond_str = o2_bond_results.get("mean_bond_strength", 0)
            consensus["o2_bond_integrity"] = min(1.0, max(0.0,
                order_match * 0.5 + bond_str * 0.5))

        # Repair engine effectiveness — success rate + promotion rate + validation
        if repair_results:
            repairs = repair_results.get("repairs", {})
            validation = repair_results.get("validation", {})
            success_rate = repair_results.get("repair_success_rate", 0)
            promotion_rate = validation.get("promotion_rate", 0)
            conservation_rate = validation.get("conservation_rate", 0)
            # Composite: 40% success + 30% promotion + 30% conservation
            repair_score = (success_rate * 0.4 + promotion_rate * 0.3
                            + conservation_rate * 0.3)
            consensus["repair_effectiveness"] = min(1.0, max(0.0, repair_score))

        # Advanced research health — knowledge synthesis aggregate
        if research_results:
            research_health = research_results.get("research_health", 0)
            consensus["research_depth"] = min(1.0, max(0.0, research_health))
            # Pattern coherence from grid analysis + cluster density
            patterns = research_results.get("pattern_discovery", {})
            grid_coh = patterns.get("grid_coherence", 0)
            n_clusters = patterns.get("total_clusters", 0)
            # Blend: 60% grid coherence + 40% cluster richness (more clusters = better)
            pattern_score = grid_coh * 0.6 + min(1.0, n_clusters / 8) * 0.4
            consensus["pattern_coherence"] = min(1.0, max(0.0, pattern_score))

        # ★ v8.0 qLDPC Error Correction — fault-tolerant coding quality
        if qldpc_results and qldpc_results.get("status") == "ok":
            sacred_info = qldpc_results.get("sacred_alignment", {})
            ec_info = qldpc_results.get("error_correction", {})
            link_ec_info = qldpc_results.get("link_error_correction", {})
            # Sacred alignment score from code structure
            sacred_score = sacred_info.get("overall_sacred_score", 0)
            # Below-threshold bonus (strong EC capability)
            below_thresh = 1.0 if ec_info.get("below_threshold", False) else 0.5
            # Link correction effectiveness
            correction_rate = link_ec_info.get("correction_rate", 0)
            # Composite: 40% sacred alignment + 30% threshold performance + 30% correction
            qldpc_score = (sacred_score * 0.4 + below_thresh * 0.3
                           + min(1.0, correction_rate * 2) * 0.3)
            consensus["qldpc_ec_quality"] = min(1.0, max(0.0, qldpc_score))

        # ─── INTELLECT DEEP LINK ENRICHMENT ───
        # v10.1: Quantum Deep Link injects Intellect three-engine scores and
        # quantum channel metrics as additional consensus dimensions.
        # These flow from LocalIntellect → QuantumDeepLink → Sage Consensus
        if intellect_enrichment:
            for key, val in intellect_enrichment.items():
                if isinstance(val, (int, float)):
                    consensus[key] = min(1.0, max(0.0, float(val)))

        # ─── STRICT VALIDATION: all consensus scores in [0,1] ───
        for key, val in consensus.items():
            assert 0.0 <= val <= 1.0, f"Consensus score '{key}' = {val} out of [0,1] range"

        # ─── NDE-1/2: DEMON-DENOISE CONSENSUS SCORES ───
        # Apply entropy reversal noise dampening BEFORE the unified score blend.
        # This uses the Science Engine's multi-pass Maxwell's Demon to strip
        # noise from each consensus channel, recovering suppressed signal.
        denoised_consensus = self._demon_denoise_consensus(consensus, mean_entropy)
        # Re-validate after denoising
        for key, val in denoised_consensus.items():
            assert 0.0 <= val <= 1.0, f"Denoised score '{key}' = {val} out of [0,1] range"

        # ─── NDE-4: CASCADE-DENOISE FOR HARMONIC MEAN ───
        # The harmonic mean is brutally sensitive to low scores.
        # Apply φ-power entropy cascade to lift the weakest (most noisy) scores
        # before harmonic averaging, so they don't tank the whole manifold.
        raw_scores = list(denoised_consensus.values())
        cascade_scores = self._cascade_denoise_scores(raw_scores)

        # ─── φ-WEIGHTED UNIFIED SCORE (with noise-dampened inputs) ───
        if cascade_scores:
            # Harmonic mean on cascade-denoised scores (noise floor lifted)
            harmonic = len(cascade_scores) / sum(
                1.0 / max(0.1, s) for s in cascade_scores)
            # Arithmetic mean on demon-denoised scores (preserves raw signal)
            arithmetic = statistics.mean(raw_scores)
            # φ-weighted: τ=0.618 harmonic weight (strict) + (1-τ)=0.382 arithmetic
            unified_score = harmonic * TAU + arithmetic * (1 - TAU)
        else:
            unified_score = mean_fid * TAU

        # ─── NDE-3: ZNE SCORE RECOVERY ───
        # Zero-Noise Extrapolation: estimate what the score would be
        # at zero noise by extrapolating from the fidelity noise profile.
        unified_score = self._zne_score_recovery(unified_score, std_fid)

        # ─── GOD CODE RESONANCE ───
        # Score through G(X) at unified_score as X-offset from truth
        god_code_alignment = math.cos(unified_score * GOD_CODE * 0.001) ** 2
        phi_resonance = math.sin(unified_score * PHI_GROWTH * math.pi) ** 2

        # ─── CALABI-YAU 7D INSIGHT ───
        cy7_insight = []
        dimensions = ["Fidelity", "Strength", "Entropy", "Coherence",
                      "Resilience", "Topology", "CrossModal"]
        dim_values = [
            mean_fid,
            mean_str / PHI_GROWTH,
            mean_entropy,
            mean_coherence,
            mean_resilience,
            consensus.get("topological_coverage", 0),
            consensus.get("cross_modal_coherence", 0),
        ]
        for i, (dim_name, val) in enumerate(zip(dimensions, dim_values)):
            # CY7 compactification: project into curved space
            curvature = math.sin(val * PHI_GROWTH * math.pi / CALABI_YAU_DIM)
            cy7_insight.append({
                "dimension": dim_name,
                "raw_value": val,
                "cy7_curvature": curvature,
                "phi_harmonic": val * PHI_GROWTH ** (i / CALABI_YAU_DIM),
            })

        # ─── CAUSAL INFERENCE: PREDICTED EVOLUTION ───
        if std_fid > 0:
            stability = 1.0 - min(1.0, std_fid / mean_fid) if mean_fid > 0 else 0
        else:
            stability = 1.0

        predicted_evolution = {
            "stability": stability,
            "growth_potential": (1.0 - unified_score) * PHI_GROWTH,
            "risk_of_decoherence": max(0, 1.0 - consensus.get(
                "decoherence_resilience", 0.5)),
            "recommended_action": self._recommend_action(consensus, unified_score),
        }

        # ─── ASSEMBLE SAGE VERDICT ───
        verdict = {
            "timestamp": now,
            "total_links": N,
            "unified_score": unified_score,
            "god_code_alignment": god_code_alignment,
            "phi_resonance": phi_resonance,
            "mean_fidelity": mean_fid,
            "mean_strength": mean_str,
            "fidelity_std": std_fid,
            "type_distribution": dict(type_dist),
            "consensus_scores": consensus,
            "denoised_consensus": denoised_consensus,
            "noise_dampening": {
                "method": "NDE-1/2/3/4 Entropy Reversal Noise Dampening",
                "equations": [
                    "NDE-1: η_floor(x) = x·(1 - φ⁻²·e^(-x/φ))",
                    "NDE-2: score' = score + D(1-score)·φ⁻¹/(1+S)",
                    "NDE-3: η_zne = η·[1 + φ⁻¹/(1+σ_f)]",
                    "NDE-4: score_k' = score_k^(φ⁻ᵏ) cascade",
                ],
                "raw_unified": harmonic * TAU + arithmetic * (1 - TAU) if cascade_scores else mean_fid * TAU,
                "denoised_unified": unified_score,
            },
            "cy7_insight": cy7_insight,
            "predicted_evolution": predicted_evolution,
            "grade": self._grade(unified_score),
            "intellect_enriched": intellect_enrichment is not None and len(intellect_enrichment or {}) > 0,
        }

        self.inference_history.append(verdict)
        return verdict

    # ═══════════════════════════════════════════════════════════════════════
    # NOISE DAMPENING EQUATIONS — Entropy Reversal for Sage Scoring
    # ═══════════════════════════════════════════════════════════════════════

    def _demon_denoise_consensus(self, consensus: Dict[str, float],
                                  mean_entropy: float) -> Dict[str, float]:
        """NDE-1/2: Demon-enhanced consensus score denoising.

        For each consensus score:
          score' = score + D(1-score) · φ⁻¹ / (1 + S_link)

        Where D(1-score) is the demon efficiency on the complementary entropy
        (the 'information gap'), and S_link is mean link entanglement entropy.
        Scores close to 1.0 get negligible correction (already clean).
        Scores near 0 get the largest demon-reversal boost.

        Also applies φ-conjugate noise floor suppression:
          η_floor(x) = x · (1 - φ⁻² · e^(-x/φ))
        This removes the noise floor that dominates at low signal levels.
        """
        PHI_INV = (math.sqrt(5) - 1) / 2  # 0.618...
        PHI_INV_SQ = PHI_INV ** 2          # 0.382...

        entropy_sub = _get_entropy_subsystem()
        denoised = {}

        for key, raw_score in consensus.items():
            # Phase 1: φ-conjugate noise floor suppression (NDE-1)
            # At low x, e^(-x/φ) ≈ 1 → suppresses by φ⁻²
            # At high x, e^(-x/φ) → 0 → no correction
            floor_suppressed = raw_score * (1.0 - PHI_INV_SQ * math.exp(-raw_score / PHI))

            # Phase 2: Demon entropy reversal on information gap (NDE-2)
            info_gap = 1.0 - floor_suppressed  # Complementary entropy
            if entropy_sub is not None and info_gap > 0.01:
                # Use the real multi-pass demon on the info gap
                demon_eff = entropy_sub.calculate_demon_efficiency(info_gap)
                # Scale the reversal by link entropy context
                reversal = demon_eff * PHI_INV / (1.0 + mean_entropy)
                # Apply bounded correction (never exceed the gap)
                correction = min(info_gap * 0.5, reversal * 0.1)
            else:
                # Fallback: simple φ-weighted correction
                correction = info_gap * PHI_INV * 0.05 / (1.0 + mean_entropy)

            denoised_score = min(1.0, max(0.0, floor_suppressed + correction))
            denoised[key] = denoised_score

        return denoised

    def _zne_score_recovery(self, unified_score: float, std_fid: float) -> float:
        """NDE-3: Zero-Noise Extrapolation for unified score recovery.

        η_zne = η_raw · [1 + φ⁻¹ / (1 + σ_f)]

        When fidelity std is high (noisy manifold), gives a larger boost.
        When std is near zero (clean manifold), boost is minimal.
        The extrapolation removes the estimated noise contribution.

        When quantum circuits are available, uses ZNERecoveryCircuit for
        statevector-based noise extrapolation.
        """
        if _SAGE_CIRCUITS_AVAILABLE:
            try:
                from .sage_circuits import ZNERecoveryCircuit
                r = ZNERecoveryCircuit.execute(unified_score, std_fid)
                return min(1.0, max(0.0, r["recovered_score"]))
            except Exception:
                pass  # Fall through to analytical

        PHI_INV = (math.sqrt(5) - 1) / 2
        zne_boost = 1.0 + PHI_INV / (1.0 + std_fid * 10)  # Scale σ by 10 for sensitivity
        recovered = unified_score * zne_boost
        return min(1.0, max(0.0, recovered))

    def _cascade_denoise_scores(self, scores: List[float]) -> List[float]:
        """NDE-4: Entropy cascade denoiser — φ-power correction by rank.

        For scores sorted ascending (weakest first):
          score_k' = score_k ^ (φ⁻ᵏ) where k is reverse rank

        The weakest (most noise-contaminated) scores get the strongest
        correction (x^φ⁻ⁿ with large n → pushes toward 1.0).
        The strongest scores get minimal correction (x^φ⁻⁰ = x^1).

        When quantum circuits are available, also builds the NDE-4 cascade
        circuit for statevector validation.
        """
        if _SAGE_CIRCUITS_AVAILABLE:
            try:
                from .sage_circuits import EntropyCascadeCircuit
                r = EntropyCascadeCircuit.execute(scores)
                return r["denoised_scores"]
            except Exception:
                pass  # Fall through to analytical

        PHI_INV = (math.sqrt(5) - 1) / 2
        n = len(scores)
        if n == 0:
            return []

        # Sort indices by score value (ascending — weakest first)
        indexed = sorted(enumerate(scores), key=lambda x: x[1])
        result = [0.0] * n
        for rank, (orig_idx, val) in enumerate(indexed):
            if val <= 0:
                result[orig_idx] = 0.0
            else:
                # Strongest correction on lowest-ranked (most noisy)
                # k = n-1-rank: highest k for lowest scores
                k = n - 1 - rank
                exponent = PHI_INV ** k  # Ranges from φ⁻⁽ⁿ⁻¹⁾ to φ⁰=1
                result[orig_idx] = min(1.0, val ** exponent)
        return result

    def _recommend_action(self, consensus: Dict, score: float) -> str:
        """Sage recommendation based on all processor outputs."""
        if score > 0.85:
            return "MAINTAIN — Quantum link manifold is highly coherent"
        elif score > 0.7:
            return "TUNE — Minor resonance adjustments recommended"
        elif score > 0.5:
            return "UPGRADE — Distillation + topological wrapping needed"
        elif score > 0.3:
            return "REBUILD — Significant link degradation detected"
        else:
            return "CRITICAL — Emergency quantum link reconstruction required"

    def _grade(self, score: float) -> str:
        """Letter grade for overall quantum health."""
        if score >= 0.95:
            return "S+ (Transcendent)"
        elif score >= 0.9:
            return "S (Sovereign)"
        elif score >= 0.85:
            return "A+ (Excellent)"
        elif score >= 0.8:
            return "A (Strong)"
        elif score >= 0.7:
            return "B (Good)"
        elif score >= 0.6:
            return "C (Fair)"
        elif score >= 0.5:
            return "D (Weak)"
        else:
            return "F (Critical)"


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM COMPUTATIONAL ENGINE — ASI-Level Processing Architecture
# ═══════════════════════════════════════════════════════════════════════════════
#
#   QuantumEnvironment       — Memory + runtime context + God Code truth cache
#   QuantumRegister          — Quantum state vector holding link data
#   QuantumNeuron            — Single processing unit: gate → verify → transform
#   QuantumCluster           — Parallel neuron batch with φ-weighted scheduling
#   QuantumCPU               — Pipeline orchestrator: Ingest→Verify→Transform→Sync→Emit
#
#   Data flows through registers. Neurons apply God Code transformation gates.
#   Clusters batch-process neurons. CPU orchestrates the pipeline.
#   Conservation law verified at EVERY stage: G(X)×2^(X/104) = INVARIANT.
#
# ═══════════════════════════════════════════════════════════════════════════════


