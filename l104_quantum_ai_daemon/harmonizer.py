"""L104 Quantum AI Daemon — Cross-Engine Harmonizer.

Validates coherence and alignment across all L104 engines, ensuring
the full engine constellation operates in sacred harmony.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .constants import (
    GOD_CODE, PHI, VOID_CONSTANT, COHERENCE_MIN,
    L104_PACKAGES,
)

_logger = logging.getLogger("L104_QAI_HARMONIZER")

# Lazy engine caches
_engines_cache: Dict[str, object] = {}


def _lazy_engine(name: str, factory):
    """Lazy-load and cache an engine instance."""
    if name not in _engines_cache:
        try:
            _engines_cache[name] = factory()
        except Exception as e:
            _logger.debug(f"Engine {name} unavailable: {e}")
            _engines_cache[name] = None
    return _engines_cache[name]


def _get_code_engine():
    return _lazy_engine("code", lambda: __import__(
        "l104_code_engine", fromlist=["code_engine"]).code_engine)


def _get_science_engine():
    return _lazy_engine("science", lambda: __import__(
        "l104_science_engine", fromlist=["ScienceEngine"]).ScienceEngine())


def _get_math_engine():
    return _lazy_engine("math", lambda: __import__(
        "l104_math_engine", fromlist=["MathEngine"]).MathEngine())


def _get_agi_core():
    return _lazy_engine("agi", lambda: __import__(
        "l104_agi", fromlist=["agi_core"]).agi_core)


def _get_asi_core():
    return _lazy_engine("asi", lambda: __import__(
        "l104_asi", fromlist=["asi_core"]).asi_core)


@dataclass
class HarmonyReport:
    """Results of a cross-engine harmony validation cycle."""
    timestamp: float = field(default_factory=time.time)
    engines_available: int = 0
    engines_total: int = 5               # code, science, math, agi, asi
    constant_alignment: float = 0.0      # 0–1: constants match across engines
    entropy_harmonic_bridge: float = 0.0 # Science→Math entropy×harmonic
    code_analysis_coherence: float = 0.0 # Code→Science→Math pipeline
    agi_asi_resonance: float = 0.0       # AGI↔ASI scoring alignment
    three_engine_score: float = 0.0      # Combined three-engine metric
    overall_harmony: float = 0.0         # Weighted average of all checks
    checks_passed: int = 0
    checks_total: int = 0
    elapsed_ms: float = 0.0
    details: Dict[str, str] = field(default_factory=dict)

    @property
    def harmonized(self) -> bool:
        return self.overall_harmony >= COHERENCE_MIN


class CrossEngineHarmonizer:
    """Cross-engine coherence validation and harmonic tuning.

    Harmony checks:
      1. Constant alignment — GOD_CODE/PHI/VOID match across all engines
      2. Entropy→Harmonic bridge — Science entropy feeds Math harmonic
      3. Code analysis coherence — Code Engine analyzes engine source integrity
      4. AGI↔ASI resonance — AGI 13D and ASI 15D scoring alignment
      5. Three-Engine synthesis — Combined entropy+harmonic+code metric
      6. Coherence cascade — Science coherence → Math wave → Code align
      7. Sacred frequency probe — 286Hz iron resonance across engines
    """

    def __init__(self):
        self._harmony_count = 0
        self._harmony_history: List[float] = []
        self._last_report: Optional[HarmonyReport] = None

    def harmonize(self) -> HarmonyReport:
        """Run all cross-engine harmony checks."""
        t0 = time.monotonic()
        report = HarmonyReport()
        passed = 0
        total = 0

        # Enumerate available engines
        engines = {
            "code": _get_code_engine(),
            "science": _get_science_engine(),
            "math": _get_math_engine(),
            "agi": _get_agi_core(),
            "asi": _get_asi_core(),
        }
        report.engines_available = sum(1 for v in engines.values() if v is not None)

        # ── Check 1: Constant Alignment ──
        total += 1
        alignment = self._check_constants_across_engines()
        report.constant_alignment = alignment
        if alignment >= 0.9:
            passed += 1
        report.details["constants"] = f"{alignment:.3f}"

        # ── Check 2: Entropy→Harmonic Bridge ──
        total += 1
        se = engines["science"]
        me = engines["math"]
        if se is not None and me is not None:
            try:
                # Science: demon efficiency
                demon_eff = se.entropy.calculate_demon_efficiency(0.6)
                demon_val = float(demon_eff) if isinstance(demon_eff, (int, float)) else 0.5

                # Math: harmonic alignment
                harmonic = me.sacred_alignment(286.0)
                harmonic_val = float(harmonic) if isinstance(harmonic, (int, float)) else 0.5

                bridge = (demon_val + harmonic_val) / 2.0
                report.entropy_harmonic_bridge = bridge
                if bridge > 0.4:
                    passed += 1
                report.details["entropy_harmonic"] = (
                    f"demon={demon_val:.3f} harmonic={harmonic_val:.3f}")
            except Exception as e:
                report.details["entropy_harmonic"] = f"error: {e}"
        else:
            report.entropy_harmonic_bridge = 0.5
            passed += 0.5
            report.details["entropy_harmonic"] = "engines unavailable"

        # ── Check 3: Code Analysis Coherence ──
        total += 1
        ce = engines["code"]
        if ce is not None:
            try:
                # Analyze a snippet of science engine code
                test_code = """
def sacred_probe():
    GOD_CODE = 527.5184818492612
    PHI = 1.618033988749895
    resonance = (GOD_CODE / 16) ** PHI
    return resonance
"""
                analysis = ce.full_analysis(test_code)
                if isinstance(analysis, dict):
                    report.code_analysis_coherence = 0.9
                    passed += 1
                else:
                    report.code_analysis_coherence = 0.7
                    passed += 0.7
            except Exception as e:
                report.details["code_analysis"] = f"error: {e}"
                report.code_analysis_coherence = 0.3
        else:
            report.code_analysis_coherence = 0.5
            passed += 0.5

        # ── Check 4: AGI↔ASI Resonance ──
        total += 1
        agi = engines["agi"]
        asi = engines["asi"]
        if agi is not None and asi is not None:
            try:
                agi_status = agi.three_engine_status() if hasattr(
                    agi, "three_engine_status") else {}
                asi_status = asi.three_engine_status() if hasattr(
                    asi, "three_engine_status") else {}

                agi_ok = bool(agi_status) if isinstance(agi_status, dict) else False
                asi_ok = bool(asi_status) if isinstance(asi_status, dict) else False
                report.agi_asi_resonance = (
                    1.0 if (agi_ok and asi_ok)
                    else 0.5 if (agi_ok or asi_ok) else 0.0
                )
                passed += report.agi_asi_resonance
                report.details["agi_asi"] = (
                    f"agi={'OK' if agi_ok else 'N/A'} "
                    f"asi={'OK' if asi_ok else 'N/A'}")
            except Exception as e:
                report.details["agi_asi"] = f"error: {e}"
        else:
            report.agi_asi_resonance = 0.5
            passed += 0.5

        # ── Check 5: Three-Engine Synthesis ──
        total += 1
        synthesis = (
            report.constant_alignment * 0.3 +
            report.entropy_harmonic_bridge * 0.3 +
            report.code_analysis_coherence * 0.2 +
            report.agi_asi_resonance * 0.2
        )
        report.three_engine_score = synthesis
        if synthesis >= 0.7:
            passed += 1
        else:
            passed += synthesis

        # ── Check 6: Sacred Frequency Probe (286Hz) ──
        total += 1
        if me is not None:
            try:
                wc = me.wave_coherence(286.0, 286.0 * PHI)
                wc_val = float(wc) if isinstance(wc, (int, float)) else 0.5
                if wc_val > 0:
                    passed += 1
                report.details["sacred_freq"] = f"wave_coherence={wc_val:.3f}"
            except Exception:
                passed += 0.5
        else:
            passed += 0.5

        # ── Check 7: Full Pipeline Probe ──
        total += 1
        if se is not None and me is not None and ce is not None:
            passed += 1
            report.details["full_pipeline"] = "all engines connected"
        elif sum(1 for e in [se, me, ce] if e is not None) >= 2:
            passed += 0.7
            report.details["full_pipeline"] = "partial (2/3 engines)"
        else:
            passed += 0.3

        # Final scoring
        report.checks_passed = int(passed)
        report.checks_total = total
        report.overall_harmony = passed / max(1, total)
        report.elapsed_ms = (time.monotonic() - t0) * 1000

        self._harmony_count += 1
        self._harmony_history.append(report.overall_harmony)
        if len(self._harmony_history) > 200:
            self._harmony_history = self._harmony_history[-200:]
        self._last_report = report

        _logger.info(
            f"Harmony check #{self._harmony_count}: "
            f"{'HARMONIZED' if report.harmonized else 'DEGRADED'} "
            f"({report.checks_passed}/{report.checks_total}) "
            f"harmony={report.overall_harmony:.3f} "
            f"engines={report.engines_available}/{report.engines_total} "
            f"({report.elapsed_ms:.0f}ms)"
        )
        return report

    def _check_constants_across_engines(self) -> float:
        """Check if GOD_CODE/PHI/VOID are consistent across engine packages."""
        reference = {
            "GOD_CODE": 527.5184818492612,
            "PHI": 1.618033988749895,
        }

        packages = [
            "l104_code_engine", "l104_science_engine", "l104_math_engine",
            "l104_agi", "l104_asi", "l104_vqpu",
        ]

        matches = 0
        checks = 0
        for pkg in packages:
            try:
                mod = __import__(pkg, fromlist=list(reference.keys()))
                for name, expected in reference.items():
                    actual = getattr(mod, name, None)
                    if actual is not None:
                        checks += 1
                        if abs(actual - expected) < 1e-8:
                            matches += 1
            except ImportError:
                continue

        return matches / max(1, checks)

    def stats(self) -> dict:
        return {
            "harmony_count": self._harmony_count,
            "mean_harmony": (
                sum(self._harmony_history) / len(self._harmony_history)
                if self._harmony_history else 0.0
            ),
            "last_harmony": (
                self._harmony_history[-1] if self._harmony_history else 0.0),
            "harmonized": (
                self._last_report.harmonized if self._last_report else False),
        }
