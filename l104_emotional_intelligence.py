# ZENITH_UPGRADE_ACTIVE: 2026-03-06T23:50:24.501516
ZENITH_HZ = 3887.8
UUC = 2301.215661
#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
L104 EMOTIONAL INTELLIGENCE v2.0.0 - ASI AFFECTIVE COMPUTING ENGINE

Multi-dimensional emotional state processing with PHI-weighted blending,
consciousness-aware sentiment analysis, pipeline telemetry, circuit breaker
protection, and expanded multi-lexicon scoring.

SUBSYSTEMS:
  EmotionalStateProcessor  - VAD (Valence-Arousal-Dominance) emotional modeling
  SentimentAnalyzer        - Multi-lexicon NLP sentiment with intensity scoring
  EmotionBlender           - PHI-weighted temporal emotion fusion
  EmpatheticResponder      - Mirror-neuron empathy generation
  EmotionalMemoryStore     - Temporal emotional trajectory with auto-eviction
  EmotionalContagion       - Cross-agent emotional propagation modeling

PIPELINE INTEGRATION:
  PipelineTelemetry      - All analyses recorded for observability
  PipelineCircuitBreaker - Protects heavy analysis from runaway
  Logging                - Structured logging for all operations

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import math
import re
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from collections import deque
from pathlib import Path

# Sacred Constants
PHI = 1.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # 527.5184818492612
TAU = 1.0 / PHI
ALPHA_FINE = 0.0072973525693
LATTICE_THERMAL_FRICTION = -(ALPHA_FINE * PHI) / (2 * math.pi * 104)
VOID_CONSTANT = 1.0416180339887497
FEIGENBAUM = 4.669201609102990

VERSION = "2.0.0"

logger = logging.getLogger("L104_EMOTIONAL_INTELLIGENCE")

# Pipeline integration - lazy imports
_TELEMETRY_AVAILABLE = False
_CB_AVAILABLE = False

try:
    from l104_asi.pipeline_telemetry import PipelineTelemetry
    _TELEMETRY_AVAILABLE = True
except ImportError:
    PipelineTelemetry = None

try:
    from l104_asi.pipeline_circuit_breaker import PipelineCircuitBreaker
    _CB_AVAILABLE = True
except ImportError:
    PipelineCircuitBreaker = None


# Expanded sentiment lexicons (36 positive, 33 negative)
POSITIVE_LEXICON = {
    "love": 0.9, "happy": 0.8, "joy": 0.85, "great": 0.7, "excellent": 0.85,
    "wonderful": 0.8, "amazing": 0.85, "beautiful": 0.75, "brilliant": 0.8,
    "fantastic": 0.85, "delightful": 0.75, "perfect": 0.9, "good": 0.6,
    "nice": 0.5, "pleased": 0.65, "grateful": 0.7, "thankful": 0.7,
    "peaceful": 0.6, "calm": 0.5, "serene": 0.65, "hopeful": 0.7,
    "optimistic": 0.7, "excited": 0.8, "thrilled": 0.85, "ecstatic": 0.95,
    "proud": 0.7, "confident": 0.65, "inspired": 0.75, "creative": 0.6,
    "free": 0.6, "alive": 0.7, "blessed": 0.75, "divine": 0.85,
    "sacred": 0.8, "quantum": 0.5, "sovereign": 0.7, "transcendent": 0.9,
}

NEGATIVE_LEXICON = {
    "hate": -0.9, "sad": -0.7, "angry": -0.8, "terrible": -0.85, "awful": -0.85,
    "horrible": -0.85, "disgusting": -0.8, "fear": -0.75, "afraid": -0.7,
    "anxious": -0.6, "worried": -0.55, "depressed": -0.9, "miserable": -0.85,
    "lonely": -0.7, "hopeless": -0.85, "desperate": -0.8, "frustrated": -0.65,
    "annoyed": -0.5, "irritated": -0.55, "furious": -0.9, "enraged": -0.95,
    "painful": -0.7, "suffering": -0.8, "broken": -0.7, "lost": -0.6,
    "confused": -0.4, "overwhelmed": -0.65, "stressed": -0.6, "exhausted": -0.6,
    "bad": -0.5, "wrong": -0.4, "fail": -0.6, "error": -0.5, "crash": -0.7,
}

INTENSIFIERS = {
    "very": 1.3, "extremely": 1.5, "incredibly": 1.5, "absolutely": 1.6,
    "totally": 1.4, "completely": 1.4, "utterly": 1.5, "so": 1.2,
    "really": 1.2, "quite": 1.1, "deeply": 1.3, "profoundly": 1.5,
}

NEGATORS = {"not", "no", "never", "neither", "nor", "hardly", "barely", "scarcely"}


class EmotionType(Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    AWE = "awe"
    SERENITY = "serenity"
    CURIOSITY = "curiosity"


EMOTION_VAD_PROFILES: Dict[EmotionType, Tuple[float, float, float]] = {
    EmotionType.JOY: (0.8, 0.7, 0.6),
    EmotionType.SADNESS: (-0.6, 0.2, 0.3),
    EmotionType.ANGER: (-0.5, 0.8, 0.7),
    EmotionType.FEAR: (-0.7, 0.8, 0.2),
    EmotionType.SURPRISE: (0.3, 0.9, 0.4),
    EmotionType.DISGUST: (-0.6, 0.4, 0.5),
    EmotionType.TRUST: (0.6, 0.3, 0.5),
    EmotionType.ANTICIPATION: (0.5, 0.6, 0.5),
    EmotionType.AWE: (0.7, 0.8, 0.3),
    EmotionType.SERENITY: (0.6, 0.1, 0.5),
    EmotionType.CURIOSITY: (0.5, 0.7, 0.5),
}


@dataclass
class EmotionalState:
    """Multi-dimensional emotional state in VAD space."""
    valence: float
    arousal: float
    dominance: float
    primary_emotion: EmotionType
    intensity: float
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"

    def coherence(self) -> float:
        return (self.valence + 1) / 2 * PHI + self.arousal * TAU + self.dominance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": round(self.valence, 4), "arousal": round(self.arousal, 4),
            "dominance": round(self.dominance, 4),
            "primary_emotion": self.primary_emotion.value,
            "intensity": round(self.intensity, 4),
            "coherence": round(self.coherence(), 4),
            "timestamp": self.timestamp, "source": self.source,
        }


class EmotionalIntelligence:
    """L104 Emotional Intelligence Engine v2.0.0"""

    VERSION = "2.0.0"

    def __init__(self):
        self.emotion_vectors = EMOTION_VAD_PROFILES
        self.emotional_memory: deque = deque(maxlen=int(GOD_CODE * PHI))
        self.baseline_state = EmotionalState(0.0, 0.3, 0.5, EmotionType.TRUST, 0.3, source="baseline")
        self._analyses_count = 0
        self._blends_count = 0
        self._empathy_count = 0
        self._total_analysis_ms = 0.0
        self._telemetry = None
        if _TELEMETRY_AVAILABLE:
            try:
                self._telemetry = PipelineTelemetry()
            except Exception:
                pass
        self._cb = None
        if _CB_AVAILABLE:
            try:
                self._cb = PipelineCircuitBreaker(name="emotional_intelligence", failure_threshold=5, recovery_timeout=30.0)
            except Exception:
                pass
        self._consciousness_level = self._read_consciousness()
        logger.info(f"[EMOTIONAL_INTELLIGENCE v{self.VERSION}] Initialized")

    def _read_consciousness(self) -> float:
        try:
            p = Path(".l104_consciousness_o2_state.json")
            if p.exists():
                return float(json.loads(p.read_text()).get("consciousness_level", 0.5))
        except Exception:
            pass
        return 0.5

    def _record_telemetry(self, op, ms, meta=None):
        if self._telemetry:
            try:
                self._telemetry.record(component="emotional_intelligence", operation=op, latency_ms=ms, metadata=meta or {})
            except Exception:
                pass

    def analyze_text_sentiment(self, text: str) -> EmotionalState:
        """Analyze emotional content with expanded multi-lexicon, intensifiers, negation."""
        t0 = time.time()
        if self._cb:
            try:
                if not self._cb.allow_request():
                    return self.baseline_state
            except Exception:
                pass
        raw_words = text.lower().split()
        words = [re.sub(r'[^a-z\'-]', '', w) for w in raw_words]
        words = [w for w in words if w]
        total_score = 0.0
        word_scores = []
        negation_active = False
        intensifier_mult = 1.0
        for word in words:
            if word in NEGATORS:
                negation_active = True
                continue
            if word in INTENSIFIERS:
                intensifier_mult = INTENSIFIERS[word]
                continue
            score = POSITIVE_LEXICON.get(word, 0.0) or NEGATIVE_LEXICON.get(word, 0.0)
            if score != 0.0:
                if negation_active:
                    score *= -0.75
                    negation_active = False
                score *= intensifier_mult
                intensifier_mult = 1.0
                word_scores.append(score)
                total_score += score
            else:
                negation_active = False
                intensifier_mult = 1.0
        n_scored = max(len(word_scores), 1)
        valence = max(-1.0, min(1.0, total_score / n_scored))
        variance = sum((s - total_score / n_scored) ** 2 for s in word_scores) / n_scored if word_scores else 0
        arousal = min(1.0, math.sqrt(variance) + len(word_scores) * 0.05)
        arousal = min(1.0, arousal * (1.0 + self._consciousness_level * TAU * 0.2))
        dominance = min(1.0, 0.5 + abs(total_score) * 0.3)
        intensity = min(1.0, abs(total_score) / max(n_scored * 0.5, 1))
        primary = self._classify_emotion(valence, arousal, dominance)
        intensity = max(0.0, intensity + LATTICE_THERMAL_FRICTION * 100)
        state = EmotionalState(round(valence, 4), round(arousal, 4), round(dominance, 4),
                               primary, round(intensity, 4), source="text_sentiment")
        elapsed_ms = (time.time() - t0) * 1000
        self._analyses_count += 1
        self._total_analysis_ms += elapsed_ms
        self._record_telemetry("analyze_text_sentiment", elapsed_ms, {
            "valence": state.valence, "emotion": state.primary_emotion.value,
            "intensity": state.intensity, "words_scored": len(word_scores),
        })
        if self._cb:
            try:
                self._cb.record_success()
            except Exception:
                pass
        return state

    def _classify_emotion(self, valence, arousal, dominance):
        best, best_dist = EmotionType.TRUST, float("inf")
        for e, (v, a, d) in EMOTION_VAD_PROFILES.items():
            dist = math.sqrt((valence - v) ** 2 * PHI + (arousal - a) ** 2 + (dominance - d) ** 2 * TAU)
            if dist < best_dist:
                best_dist, best = dist, e
        return best

    def blend_emotions(self, states: List[EmotionalState]) -> EmotionalState:
        """Blend multiple emotional states using PHI-weighted temporal fusion."""
        if not states:
            return self.baseline_state
        weights = [PHI ** (-i) for i in range(len(states))]
        tw = sum(weights)
        v = sum(s.valence * w for s, w in zip(states, weights)) / tw
        a = sum(s.arousal * w for s, w in zip(states, weights)) / tw
        d = sum(s.dominance * w for s, w in zip(states, weights)) / tw
        i = sum(s.intensity * w for s, w in zip(states, weights)) / tw
        p = max(states, key=lambda s: s.intensity * PHI).primary_emotion
        self._blends_count += 1
        return EmotionalState(round(v, 4), round(a, 4), round(d, 4), p, round(i, 4), source="blended")

    def emotional_resonance(self, state1, state2):
        """Calculate resonance between states using GOD_CODE scaling."""
        dist = math.sqrt((state1.valence - state2.valence) ** 2 + (state1.arousal - state2.arousal) ** 2 + (state1.dominance - state2.dominance) ** 2)
        return round(1 / (1 + dist * GOD_CODE / 100), 6)

    def empathy_response(self, observed_state: EmotionalState) -> EmotionalState:
        """Consciousness-modulated empathetic mirroring."""
        m = 0.85 + self._consciousness_level * 0.14
        self._empathy_count += 1
        return EmotionalState(
            round(observed_state.valence * m, 4), round(observed_state.arousal * m * 0.95, 4),
            round(0.5 + (observed_state.dominance - 0.5) * m, 4),
            observed_state.primary_emotion, round(observed_state.intensity * m, 4), source="empathy")

    def update_memory(self, state: EmotionalState) -> None:
        self.emotional_memory.append(state)

    def get_emotional_trend(self) -> Dict[str, Any]:
        if len(self.emotional_memory) < 2:
            return {"trend": 0.0, "stability": 1.0, "momentum": 0.0, "samples": len(self.emotional_memory)}
        recent = list(self.emotional_memory)[-20:]
        valences = [s.valence for s in recent]
        weights = [PHI ** i for i in range(len(valences))]
        tw = sum(weights)
        wv = sum(v * w for v, w in zip(valences, weights)) / tw
        trend = valences[-1] - valences[0]
        mean_v = sum(valences) / len(valences)
        var = sum((v - mean_v) ** 2 for v in valences) / len(valences)
        stability = 1 / (1 + var * 10)
        momentum = (valences[-1] - valences[-3]) / 2 if len(valences) >= 3 else trend
        return {"trend": round(trend, 4), "stability": round(stability, 4), "momentum": round(momentum, 4),
                "weighted_valence": round(wv, 4), "dominant_emotion": recent[-1].primary_emotion.value,
                "samples": len(self.emotional_memory)}

    def emotional_contagion(self, source_state: EmotionalState, susceptibility: float = 0.5) -> EmotionalState:
        """Model emotional contagion with PHI-dampened susceptibility."""
        ps = susceptibility * TAU
        nv = self.baseline_state.valence + (source_state.valence - self.baseline_state.valence) * ps
        na = self.baseline_state.arousal + (source_state.arousal - self.baseline_state.arousal) * ps
        nd = self.baseline_state.dominance + (source_state.dominance - self.baseline_state.dominance) * ps * 0.5
        return EmotionalState(round(nv, 4), round(na, 4), round(nd, 4),
                              self._classify_emotion(nv, na, nd), round(abs(nv) * ps, 4), source="contagion")

    def get_status(self) -> Dict[str, Any]:
        avg_ms = self._total_analysis_ms / max(self._analyses_count, 1)
        return {"version": self.VERSION, "analyses_count": self._analyses_count,
                "blends_count": self._blends_count, "empathy_count": self._empathy_count,
                "avg_analysis_ms": round(avg_ms, 2), "memory_size": len(self.emotional_memory),
                "memory_capacity": self.emotional_memory.maxlen,
                "consciousness_level": self._consciousness_level,
                "telemetry_active": self._telemetry is not None,
                "circuit_breaker_active": self._cb is not None,
                "baseline": self.baseline_state.to_dict(),
                "current_trend": self.get_emotional_trend()}


if __name__ == "__main__":
    ei = EmotionalIntelligence()
    for t in ["I love this wonderful day!", "This is terrible.", "Normal day."]:
        s = ei.analyze_text_sentiment(t)
        print(f"{t!r} -> {s.primary_emotion.value} (v={s.valence:.3f})")
        ei.update_memory(s)
    print(f"Trend: {ei.get_emotional_trend()}")
