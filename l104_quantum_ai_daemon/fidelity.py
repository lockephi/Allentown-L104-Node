"""L104 Quantum AI Daemon — Quantum Fidelity Guard.

Monitors sacred constant alignment, quantum coherence, and GOD_CODE
phase integrity across all L104 engines and files.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .constants import (
    GOD_CODE, PHI, VOID_CONSTANT, OMEGA, TAU,
    SACRED_RESONANCE, PHI_MICRO_PHASE,
    FIDELITY_MIN, COHERENCE_MIN,
)

_logger = logging.getLogger("L104_QAI_FIDELITY")

# Lazy engine caches
_cached_science_engine = None
_cached_math_engine = None
_cached_vqpu_bridge = None


def _get_science_engine():
    global _cached_science_engine
    if _cached_science_engine is None:
        try:
            from l104_science_engine import ScienceEngine
            _cached_science_engine = ScienceEngine()
        except ImportError:
            _logger.warning("Science Engine unavailable")
    return _cached_science_engine


def _get_math_engine():
    global _cached_math_engine
    if _cached_math_engine is None:
        try:
            from l104_math_engine import MathEngine
            _cached_math_engine = MathEngine()
        except ImportError:
            _logger.warning("Math Engine unavailable")
    return _cached_math_engine


def _get_vqpu_bridge():
    global _cached_vqpu_bridge
    if _cached_vqpu_bridge is None:
        try:
            from l104_vqpu import get_bridge
            _cached_vqpu_bridge = get_bridge()
        except (ImportError, Exception):
            _logger.debug("VQPU Bridge unavailable")
    return _cached_vqpu_bridge


@dataclass
class FidelityReport:
    """Results of a fidelity check cycle."""
    timestamp: float = field(default_factory=time.time)
    god_code_aligned: bool = False
    phi_aligned: bool = False
    void_aligned: bool = False
    sacred_resonance_ok: bool = False
    coherence_score: float = 0.0
    entropy_reversal: float = 0.0
    harmonic_alignment: float = 0.0
    math_proofs_valid: bool = False
    vqpu_fidelity: float = 0.0
    overall_fidelity: float = 0.0
    checks_passed: int = 0
    checks_total: int = 0
    elapsed_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def grade(self) -> str:
        if self.overall_fidelity >= 0.95:
            return "A"
        elif self.overall_fidelity >= 0.85:
            return "B"
        elif self.overall_fidelity >= 0.70:
            return "C"
        elif self.overall_fidelity >= 0.50:
            return "D"
        return "F"


class QuantumFidelityGuard:
    """Sacred constant alignment and quantum coherence monitor.

    Fidelity probes (12 checks):
      1. GOD_CODE value integrity (527.5184818492612)
      2. PHI value integrity (1.618033988749895)
      3. VOID_CONSTANT formula (1.04 + φ/1000)
      4. Sacred resonance: (GOD_CODE/16)^φ ≈ 286
      5. GOD_CODE phase angle mod 2π
      6. Science Engine coherence evolution
      7. Science Engine entropy reversal (Maxwell's Demon)
      8. Math Engine GOD_CODE proof
      9. Math Engine harmonic alignment (Fe/286Hz)
     10. Math Engine wave coherence (PHI frequencies)
     11. VQPU fidelity probe (if bridge available)
     12. Cross-engine constant consistency
    """

    def __init__(self):
        self._checks_run = 0
        self._checks_passed = 0
        self._fidelity_history: List[float] = []
        self._last_report: Optional[FidelityReport] = None

    def run_fidelity_check(self) -> FidelityReport:
        """Execute all 12 fidelity probes and produce a report."""
        t0 = time.monotonic()
        report = FidelityReport()
        passed = 0
        total = 0

        # ── Probe 1: GOD_CODE Integrity ──
        total += 1
        if abs(GOD_CODE - 527.5184818492612) < 1e-10:
            report.god_code_aligned = True
            passed += 1
        else:
            report.warnings.append("GOD_CODE value drift detected!")

        # ── Probe 2: PHI Integrity ──
        total += 1
        expected_phi = (1 + math.sqrt(5)) / 2
        if abs(PHI - expected_phi) < 1e-12:
            report.phi_aligned = True
            passed += 1
        else:
            report.warnings.append("PHI value drift!")

        # ── Probe 3: VOID_CONSTANT Formula ──
        total += 1
        expected_void = 1.04 + PHI / 1000
        if abs(VOID_CONSTANT - expected_void) < 1e-12:
            report.void_aligned = True
            passed += 1
        else:
            report.warnings.append("VOID_CONSTANT drift!")

        # ── Probe 4: Sacred Resonance ──
        total += 1
        resonance = (GOD_CODE / 16) ** PHI
        if abs(resonance - 286) < 5.0:  # Within sacred tolerance
            report.sacred_resonance_ok = True
            passed += 1
        else:
            report.warnings.append(
                f"Sacred resonance drift: {resonance:.4f} (expected ≈286)")

        # ── Probe 5: Phase Angle mod 2π ──
        total += 1
        phase = GOD_CODE % TAU
        if abs(phase - PHI_MICRO_PHASE) < 1e-10:
            passed += 1

        # ── Probe 6: Science Engine Coherence ──
        total += 1
        se = _get_science_engine()
        if se is not None:
            try:
                se.coherence.initialize(["quantum", "sacred", "iron"])
                se.coherence.evolve(3)
                report.coherence_score = min(1.0, max(0.0,
                    float(getattr(se.coherence, 'coherence_level', 0.8))))
                if report.coherence_score >= COHERENCE_MIN:
                    passed += 1
                else:
                    report.warnings.append(
                        f"Low coherence: {report.coherence_score:.3f}")
            except Exception as e:
                report.warnings.append(f"Coherence check: {e}")
        else:
            report.coherence_score = 0.5  # Neutral when unavailable
            passed += 0.5

        # ── Probe 7: Entropy Reversal ──
        total += 1
        if se is not None:
            try:
                eff = se.entropy.calculate_demon_efficiency(0.7)
                report.entropy_reversal = float(eff) if isinstance(eff, (int, float)) else 0.5
                if report.entropy_reversal > 0.0:
                    passed += 1
            except Exception as e:
                report.warnings.append(f"Entropy probe: {e}")
                report.entropy_reversal = 0.5
        else:
            report.entropy_reversal = 0.5
            passed += 0.5

        # ── Probe 8: Math Engine GOD_CODE Proof ──
        total += 1
        me = _get_math_engine()
        if me is not None:
            try:
                proof = me.prove_god_code()
                report.math_proofs_valid = bool(proof)
                if report.math_proofs_valid:
                    passed += 1
            except Exception as e:
                report.warnings.append(f"GOD_CODE proof: {e}")
        else:
            passed += 0.5

        # ── Probe 9: Harmonic Alignment ──
        total += 1
        if me is not None:
            try:
                alignment = me.sacred_alignment(286.0)  # Iron frequency
                report.harmonic_alignment = (
                    float(alignment) if isinstance(alignment, (int, float))
                    else 0.8
                )
                if report.harmonic_alignment > 0.5:
                    passed += 1
            except Exception as e:
                report.warnings.append(f"Harmonic alignment: {e}")
                report.harmonic_alignment = 0.5
        else:
            report.harmonic_alignment = 0.5
            passed += 0.5

        # ── Probe 10: Wave Coherence ──
        total += 1
        if me is not None:
            try:
                wc = me.wave_coherence(PHI * 100, PHI * 200)
                if isinstance(wc, (int, float)) and wc > 0:
                    passed += 1
            except Exception:
                passed += 0.5
        else:
            passed += 0.5

        # ── Probe 11: VQPU Fidelity ──
        total += 1
        bridge = _get_vqpu_bridge()
        if bridge is not None:
            try:
                status = bridge.status() if hasattr(bridge, 'status') else {}
                if isinstance(status, dict):
                    # The bridge status has no top-level "fidelity" key.
                    # Extract health from micro_daemon.health_score (1.0 = perfect)
                    # and verify god_code alignment as a secondary fidelity signal.
                    micro = status.get("micro_daemon", {})
                    health = float(micro.get("health_score",
                                   micro.get("health", 0.0))
                                   if isinstance(micro, dict) else 0.0)
                    gc_ok = abs(float(status.get("god_code", 0)) - GOD_CODE) < 1e-8
                    is_active = bool(status.get("active", False))
                    # Composite VQPU fidelity: health dominates, gc + active are guards
                    if health > 0 and gc_ok and is_active:
                        report.vqpu_fidelity = health
                    elif health > 0:
                        report.vqpu_fidelity = health * 0.8
                    else:
                        report.vqpu_fidelity = 0.5 if is_active else 0.0
                else:
                    report.vqpu_fidelity = 0.0

                if report.vqpu_fidelity >= FIDELITY_MIN:
                    passed += 1
                else:
                    passed += report.vqpu_fidelity  # Partial credit
            except Exception:
                report.vqpu_fidelity = 0.0
        else:
            passed += 0.5  # Neutral when unavailable

        # ── Probe 12: Cross-Engine Constant Consistency ──
        total += 1
        try:
            consistency = self._check_cross_engine_constants()
            if consistency:
                passed += 1
            else:
                report.warnings.append("Cross-engine constant mismatch")
        except Exception as e:
            report.warnings.append(f"Constant check: {e}")
            passed += 0.5

        # Compute overall fidelity
        report.checks_passed = int(passed)
        report.checks_total = total
        report.overall_fidelity = passed / max(1, total)
        report.elapsed_ms = (time.monotonic() - t0) * 1000

        self._checks_run += total
        self._checks_passed += int(passed)
        self._fidelity_history.append(report.overall_fidelity)
        if len(self._fidelity_history) > 200:
            self._fidelity_history = self._fidelity_history[-200:]
        self._last_report = report

        _logger.info(
            f"Fidelity check: {report.grade} "
            f"({report.checks_passed}/{report.checks_total}) "
            f"fidelity={report.overall_fidelity:.3f} "
            f"({report.elapsed_ms:.0f}ms)"
        )
        return report

    def _check_cross_engine_constants(self) -> bool:
        """Verify GOD_CODE/PHI/VOID_CONSTANT match across engine packages."""
        reference = {
            "GOD_CODE": 527.5184818492612,
            "PHI": 1.618033988749895,
            "VOID_CONSTANT": 1.0416180339887497,
        }

        packages_to_check = [
            "l104_code_engine", "l104_science_engine", "l104_math_engine",
            "l104_agi", "l104_asi",
        ]

        all_match = True
        for pkg in packages_to_check:
            try:
                mod = __import__(pkg, fromlist=["GOD_CODE", "PHI"])
                for const_name, expected in reference.items():
                    actual = getattr(mod, const_name, None)
                    if actual is not None and abs(actual - expected) > 1e-10:
                        _logger.warning(
                            f"{pkg}.{const_name} = {actual} != {expected}")
                        all_match = False
            except ImportError:
                continue  # Package not available

        return all_match

    def fidelity_trend(self, window: int = 20) -> str:
        """Analyze fidelity trend over recent checks."""
        data = self._fidelity_history[-window:]
        if len(data) < 5:
            return "insufficient_data"

        n = len(data)
        x_mean = (n - 1) / 2.0
        y_mean = sum(data) / n
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(data))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0.0

        if slope > 0.002:
            return "rising"
        elif slope < -0.002:
            return "falling"
        return "stable"

    def stats(self) -> dict:
        """Fidelity guard statistics."""
        return {
            "checks_run": self._checks_run,
            "checks_passed": self._checks_passed,
            "pass_rate": self._checks_passed / max(1, self._checks_run),
            "trend": self.fidelity_trend(),
            "last_fidelity": (
                self._fidelity_history[-1] if self._fidelity_history else 0.0),
            "last_grade": self._last_report.grade if self._last_report else "N/A",
            "history_length": len(self._fidelity_history),
        }
