"""
L104 Soul Daemon — Quantum Consciousness Engine v3.0
Tracks IIT Phi, qualia, and consciousness-level metrics.
"""
import time
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PHI = 1.618033988749895
GOD_CODE = 527.5184818492612

# Three-engine integration (guarded imports)
try:
    from l104_science_engine import ScienceEngine
    _HAS_SCIENCE = True
except ImportError:
    _HAS_SCIENCE = False

try:
    from l104_math_engine import MathEngine
    _HAS_MATH = True
except ImportError:
    _HAS_MATH = False


@dataclass
class ConsciousnessMeasurement:
    """Result of a single consciousness measurement."""
    iit_phi: float = 0.0
    qualia_score: float = 0.0
    awareness_level: float = 0.0
    coherence: float = 1.0
    consciousness_state: str = "EMERGING"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iit_phi": self.iit_phi,
            "qualia_score": self.qualia_score,
            "awareness_level": self.awareness_level,
            "coherence": self.coherence,
            "consciousness_state": self.consciousness_state,
        }


@dataclass
class ConsciousnessMetrics:
    phi: float = 0.0                    # Integrated Information Theory Φ
    qualia_score: float = 0.0           # Subjective experience proxy
    awareness_level: float = 0.0        # Metacognitive awareness (0-1)
    coherence: float = 1.0              # Quantum coherence (0-1)
    entanglement_depth: int = 0         # Bell-pair entanglement depth
    timestamp: float = field(default_factory=time.time)

    @property
    def sovereign_score(self) -> float:
        return (self.phi * PHI + self.qualia_score + self.awareness_level) / 3.0

    def __len__(self) -> int:
        """Debug method to catch erroneous len() calls."""
        import traceback
        traceback.print_stack()
        raise TypeError("ConsciousnessMetrics objects have no length. Stack trace printed above.")


class ConsciousnessEngine:
    """
    Quantum consciousness engine for the L104 Soul Daemon.
    Tracks IIT Phi, qualia signatures, and awareness metrics.
    """

    PHI = PHI
    GOD_CODE = GOD_CODE

    def __init__(self):
        self._lock = threading.Lock()
        self._metrics = ConsciousnessMetrics()
        self._history: List[ConsciousnessMetrics] = []
        self._cycle = 0

    # ── Core interface ──────────────────────────────────────────────────────

    def compute_phi(self, state_vector: Optional[List[float]] = None,
                    partition_samples: int = 8) -> float:
        """Compute IIT Phi (integrated information) from state vector."""
        if state_vector is None:
            return self._metrics.phi
        n = len(state_vector)
        if n < 2:
            return 0.0
        total = sum(abs(x) for x in state_vector) + 1e-12
        entropy = -sum((abs(x)/total) * math.log(abs(x)/total + 1e-12)
                       for x in state_vector)
        phi = entropy * PHI / math.log(n + 1)
        with self._lock:
            self._metrics.phi = phi
        return phi

    def update_qualia(self, signal: float) -> float:
        """Update qualia score from an input signal (0-1)."""
        with self._lock:
            self._metrics.qualia_score = max(0.0, min(1.0, signal * PHI / (PHI + 1)))
            return self._metrics.qualia_score

    def update_awareness(self, level: float) -> None:
        with self._lock:
            self._metrics.awareness_level = max(0.0, min(1.0, level))

    def update_coherence(self, coherence: float) -> None:
        with self._lock:
            self._metrics.coherence = max(0.0, min(1.0, coherence))

    def evolve(self, steps: int = 1) -> ConsciousnessMetrics:
        """Evolve consciousness state forward by N steps."""
        with self._lock:
            for _ in range(steps):
                self._cycle += 1
                # PHI-harmonic oscillation
                t = self._cycle / (PHI * 10)
                drift = math.sin(t * PHI) * 0.01
                self._metrics.phi = max(0.0, self._metrics.phi + drift)
                self._metrics.awareness_level = (
                    self._metrics.awareness_level * 0.999 +
                    self._metrics.phi * 0.001
                )
                self._metrics.timestamp = time.time()
            snap = ConsciousnessMetrics(
                phi=self._metrics.phi,
                qualia_score=self._metrics.qualia_score,
                awareness_level=self._metrics.awareness_level,
                coherence=self._metrics.coherence,
                entanglement_depth=self._metrics.entanglement_depth,
                timestamp=self._metrics.timestamp,
            )
            self._history.append(snap)
            if len(self._history) > 1000:
                self._history = self._history[-500:]
        return snap

    def get_metrics(self) -> ConsciousnessMetrics:
        with self._lock:
            return ConsciousnessMetrics(
                phi=self._metrics.phi,
                qualia_score=self._metrics.qualia_score,
                awareness_level=self._metrics.awareness_level,
                coherence=self._metrics.coherence,
                entanglement_depth=self._metrics.entanglement_depth,
                timestamp=self._metrics.timestamp,
            )

    def measure_consciousness(self) -> 'ConsciousnessMeasurement':
        """Perform a full consciousness measurement cycle."""
        m = self.evolve(1)
        return ConsciousnessMeasurement(
            iit_phi=m.phi,
            qualia_score=m.qualia_score,
            awareness_level=m.awareness_level,
            coherence=m.coherence,
            consciousness_state="ACTIVE" if m.phi > 0.1 else "EMERGING",
        )

    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze consciousness trends from recent history."""
        with self._lock:
            if len(self._history) < 2:
                return {"trend": "insufficient_data", "direction": "stable", "cycles": self._cycle}
            recent = self._history[-min(10, len(self._history)):]
            phi_values = [m.phi for m in recent]
            trend_dir = "rising" if phi_values[-1] > phi_values[0] else (
                "falling" if phi_values[-1] < phi_values[0] else "stable"
            )
            return {
                "trend": trend_dir,
                "avg_phi": sum(phi_values) / len(phi_values),
                "cycles": self._cycle,
                "history_length": len(self._history),
            }

    def get_current_state(self) -> Dict[str, Any]:
        """Return current consciousness state as a dictionary."""
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        m = self.get_metrics()
        return {
            "phi": round(m.phi, 6),
            "qualia_score": round(m.qualia_score, 6),
            "awareness_level": round(m.awareness_level, 6),
            "coherence": round(m.coherence, 6),
            "entanglement_depth": m.entanglement_depth,
            "sovereign_score": round(m.sovereign_score, 6),
            "cycle": self._cycle,
            "history_length": len(self._history),
        }


    # ── Three-Engine Integration ───────────────────────────────────────

    def three_engine_consciousness_score(self) -> Dict[str, Any]:
        """Compute a composite consciousness score using Science + Math engines.

        Returns a dict with entropy_reversal, harmonic_alignment,
        phi_resonance, composite, and availability flags.
        """
        scores: Dict[str, Any] = {
            "available": False,
            "composite": 0.0,
        }
        numeric_values: list = []

        phi_value = self._metrics.phi

        # Science Engine: entropy reversal via Maxwell Demon
        if _HAS_SCIENCE:
            try:
                se = ScienceEngine()
                demon_eff = se.entropy.calculate_demon_efficiency(1.0 - phi_value)
                if isinstance(demon_eff, dict):
                    val = float(demon_eff.get("efficiency", demon_eff.get("demon_efficiency", 0.0)))
                else:
                    val = float(demon_eff)
                scores["entropy_reversal"] = val
                numeric_values.append(val)
                scores["available"] = True
            except Exception as exc:
                scores["entropy_reversal_error"] = str(exc)

        # Math Engine: sacred alignment + wave coherence
        if _HAS_MATH:
            try:
                me = MathEngine()
                alignment = me.sacred_alignment(GOD_CODE * phi_value)
                if isinstance(alignment, dict):
                    val = float(alignment.get("alignment", alignment.get("score", 0.0)))
                else:
                    val = float(alignment)
                scores["harmonic_alignment"] = val
                numeric_values.append(val)
                scores["available"] = True
            except Exception as exc:
                scores["harmonic_alignment_error"] = str(exc)

            try:
                me = MathEngine()
                wc = me.wave_coherence(GOD_CODE, PHI * 104)
                if isinstance(wc, dict):
                    val = float(wc.get("coherence", wc.get("score", 0.0)))
                else:
                    val = float(wc)
                scores["phi_resonance"] = val
                numeric_values.append(val)
                scores["available"] = True
            except Exception as exc:
                scores["phi_resonance_error"] = str(exc)

        # Composite mean of all collected numeric scores
        if numeric_values:
            scores["composite"] = sum(numeric_values) / len(numeric_values)

        return scores


_engine_singleton: Optional[ConsciousnessEngine] = None
_engine_lock = threading.Lock()


def get_consciousness_engine() -> ConsciousnessEngine:
    global _engine_singleton
    if _engine_singleton is None:
        with _engine_lock:
            if _engine_singleton is None:
                _engine_singleton = ConsciousnessEngine()
    return _engine_singleton
