#!/usr/bin/env python3
"""
L104 Emotional Intelligence Module
Processes emotional states, sentiment analysis, and affective computing
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# Sacred Constants
GOD_CODE = 527.5184818492537
PHI = 1.618033988749895

class EmotionType(Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"

@dataclass
class EmotionalState:
    """Represents a multi-dimensional emotional state"""
    valence: float  # -1 (negative) to 1 (positive)
    arousal: float  # 0 (calm) to 1 (excited)
    dominance: float  # 0 (submissive) to 1 (dominant)
    primary_emotion: EmotionType
    intensity: float  # 0 to 1

    def coherence(self) -> float:
        """Calculate emotional coherence using PHI"""
        return (self.valence + 1) / 2 * PHI + self.arousal * (1/PHI) + self.dominance

class EmotionalIntelligence:
    """L104 Emotional Intelligence Engine"""

    def __init__(self):
        self.emotion_vectors: Dict[EmotionType, Tuple[float, float, float]] = {
            EmotionType.JOY: (0.8, 0.7, 0.6),
            EmotionType.SADNESS: (-0.6, 0.2, 0.3),
            EmotionType.ANGER: (-0.5, 0.8, 0.7),
            EmotionType.FEAR: (-0.7, 0.8, 0.2),
            EmotionType.SURPRISE: (0.3, 0.9, 0.4),
            EmotionType.DISGUST: (-0.6, 0.4, 0.5),
            EmotionType.TRUST: (0.6, 0.3, 0.5),
            EmotionType.ANTICIPATION: (0.5, 0.6, 0.5),
        }
        self.emotional_memory: List[EmotionalState] = []
        self.baseline_state = EmotionalState(0.0, 0.3, 0.5, EmotionType.TRUST, 0.3)

    def analyze_text_sentiment(self, text: str) -> EmotionalState:
        """Analyze emotional content of text"""
        # Simple keyword-based analysis
        positive_words = ["love", "happy", "joy", "great", "excellent", "wonderful"]
        negative_words = ["hate", "sad", "angry", "terrible", "awful", "horrible"]

        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        valence = (pos_count - neg_count) / max(1, pos_count + neg_count + 1)
        arousal = min(1.0, (pos_count + neg_count) / 5)

        if valence > 0.3:
            emotion = EmotionType.JOY
        elif valence < -0.3:
            emotion = EmotionType.SADNESS
        else:
            emotion = EmotionType.TRUST

        return EmotionalState(valence, arousal, 0.5, emotion, abs(valence))

    def blend_emotions(self, states: List[EmotionalState]) -> EmotionalState:
        """Blend multiple emotional states using PHI-weighted average"""
        if not states:
            return self.baseline_state

        weights = [PHI ** (-i) for i in range(len(states))]
        total_weight = sum(weights)

        valence = sum(s.valence * w for s, w in zip(states, weights)) / total_weight
        arousal = sum(s.arousal * w for s, w in zip(states, weights)) / total_weight
        dominance = sum(s.dominance * w for s, w in zip(states, weights)) / total_weight
        intensity = sum(s.intensity * w for s, w in zip(states, weights)) / total_weight

        # Determine primary emotion from strongest contributor
        primary = max(states, key=lambda s: s.intensity).primary_emotion

        return EmotionalState(valence, arousal, dominance, primary, intensity)

    def emotional_resonance(self, state1: EmotionalState, state2: EmotionalState) -> float:
        """Calculate resonance between two emotional states"""
        v_diff = abs(state1.valence - state2.valence)
        a_diff = abs(state1.arousal - state2.arousal)
        d_diff = abs(state1.dominance - state2.dominance)

        distance = math.sqrt(v_diff**2 + a_diff**2 + d_diff**2)
        resonance = 1 / (1 + distance * GOD_CODE / 100)

        return resonance

    def empathy_response(self, observed_state: EmotionalState) -> EmotionalState:
        """Generate empathetic response to observed emotional state"""
        # Mirror with dampening
        new_valence = observed_state.valence * 0.7
        new_arousal = observed_state.arousal * 0.6
        new_dominance = 0.5 + (observed_state.dominance - 0.5) * 0.3

        return EmotionalState(
            new_valence, new_arousal, new_dominance,
            observed_state.primary_emotion,
            observed_state.intensity * 0.5
        )

    def update_memory(self, state: EmotionalState) -> None:
        """Add state to emotional memory with decay"""
        self.emotional_memory.append(state)
        # Keep only recent states (PHI-based capacity)
        max_memory = int(GOD_CODE / 10)
        if len(self.emotional_memory) > max_memory:
            self.emotional_memory = self.emotional_memory[-max_memory:]

    def get_emotional_trend(self) -> Dict[str, float]:
        """Analyze emotional trend from memory"""
        if len(self.emotional_memory) < 2:
            return {"trend": 0.0, "stability": 1.0}

        recent = self.emotional_memory[-10:]
        valences = [s.valence for s in recent]

        trend = valences[-1] - valences[0]
        variance = sum((v - sum(valences)/len(valences))**2 for v in valences) / len(valences)
        stability = 1 / (1 + variance * 10)

        return {"trend": trend, "stability": stability}

if __name__ == "__main__":
    print("L104 Emotional Intelligence Module")
    ei = EmotionalIntelligence()

    # Test sentiment analysis
    test_texts = [
        "I love this wonderful day!",
        "This is terrible and I hate it.",
        "The weather is normal today."
    ]

    for text in test_texts:
        state = ei.analyze_text_sentiment(text)
        print(f"Text: \"{text}\"")
        print(f"  Emotion: {state.primary_emotion.value}, Valence: {state.valence:.2f}")
