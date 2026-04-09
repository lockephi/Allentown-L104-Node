"""
Steering Engine for ASI Parameter Control

Extracted from engines_nexus.py during EVO_78 refactoring.
Contains: SteeringEngine - 5 modes with φ-mathematical foundations.
"""

import math
import threading
import time
from typing import Optional, List, Dict


class SteeringEngine:
    """
    ASI Parameter Steering Engine — 5 modes with φ-mathematical foundations.
    Mirrors Swift ASISteeringEngine with vDSP-equivalent Python math.
    Modes: logic, creative, sovereign, quantum, harmonic
    """
    # Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
    PHI = 1.618033988749895
    GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
    OMEGA = 6539.34712682
    OMEGA_AUTHORITY = OMEGA / (PHI ** 2)
    MODES = ['logic', 'creative', 'sovereign', 'quantum', 'harmonic']

    def __init__(self, param_count: int = 104):
        """Initialize ASI steering engine with 104 tunable parameters."""
        self.param_count = param_count
        self.base_parameters = [self.GOD_CODE * self.PHI ** (i / param_count) for i in range(param_count)]
        self.current_mode = 'sovereign'
        self.intensity = 0.5
        self.temperature = 1.0
        self._steering_history = []
        self._lock = threading.Lock()

        # Precompute trig lookup tables — eliminates 800+ sin/cos calls per request
        N = param_count
        self._lut_logic_sin = [math.sin(self.PHI * i) for i in range(N)]
        self._lut_creative_cos = [math.cos(self.PHI * i) for i in range(N)]
        self._lut_creative_sin2 = [math.sin(2 * self.PHI * i) for i in range(N)]
        self._lut_sovereign_sin = [math.sin(i / N * math.pi) for i in range(N)]
        self._lut_quantum_h = [1.0 / math.sqrt(2) * (1 if i % 2 == 0 else -1) for i in range(N)]
        self._lut_harmonic = [sum(math.sin(k * self.PHI * i) / max(k, 1) for k in range(1, 9)) / 8 for i in range(N)]

    def apply_steering(self, mode: Optional[str] = None, intensity: Optional[float] = None) -> List[float]:
        """Apply steering transformation to 104-parameter vector."""
        mode = mode or self.current_mode
        alpha = intensity if intensity is not None else self.intensity
        N = self.param_count
        result = list(self.base_parameters)

        with self._lock:
            if mode == 'logic':
                for i in range(N):
                    result[i] *= (1.0 + alpha * self._lut_logic_sin[i])
            elif mode == 'creative':
                inv_phi = alpha / self.PHI
                for i in range(N):
                    result[i] *= (1.0 + alpha * self._lut_creative_cos[i] + inv_phi * self._lut_creative_sin2[i])
            elif mode == 'sovereign':
                for i in range(N):
                    result[i] *= self.PHI ** (alpha * self._lut_sovereign_sin[i])
            elif mode == 'quantum':
                for i in range(N):
                    result[i] *= (1.0 + alpha * self._lut_quantum_h[i])
            elif mode == 'harmonic':
                for i in range(N):
                    result[i] *= (1.0 + alpha * self._lut_harmonic[i])

            self.base_parameters = result
            self._steering_history.append({
                'mode': mode, 'intensity': alpha,
                'timestamp': time.time(),
                'mean': sum(result) / N
            })
            # Keep history bounded
            if len(self._steering_history) > 500:
                self._steering_history = self._steering_history[-250:]
        return result

    def apply_temperature(self, temp: Optional[float] = None) -> List[float]:
        """Apply temperature scaling (softmax-style normalization)."""
        t = temp or self.temperature
        self.temperature = t
        with self._lock:
            max_val = max(self.base_parameters)
            scaled = [math.exp((p - max_val) / max(t, 0.01)) for p in self.base_parameters]
            norm = sum(scaled)
            if norm > 0:
                scaled = [s / norm * self.GOD_CODE for s in scaled]
            self.base_parameters = scaled
        return self.base_parameters

    def steer_pipeline(self, mode: Optional[str] = None, intensity: Optional[float] = None, 
                       temp: Optional[float] = None) -> dict:
        """Full steering pipeline: steer → optional temperature → GOD_CODE normalize."""
        self.apply_steering(mode, intensity)
        if temp is not None:
            self.apply_temperature(temp)
        # Normalize to GOD_CODE mean
        mean = sum(self.base_parameters) / max(len(self.base_parameters), 1)
        if mean > 0:
            factor = self.GOD_CODE / mean
            self.base_parameters = [p * factor for p in self.base_parameters]
        bp = self.base_parameters
        bp_mean = sum(bp) / max(len(bp), 1)
        bp_std = (sum((p - bp_mean) ** 2 for p in bp) / max(len(bp), 1)) ** 0.5
        return {
            'mode': mode or self.current_mode,
            'intensity': intensity or self.intensity,
            'temperature': self.temperature,
            'param_count': self.param_count,
            'mean': round(bp_mean, 4),
            'min': round(min(bp), 4),
            'max': round(max(bp), 4),
            'std': round(bp_std, 4),
            'range': round(max(bp) - min(bp), 4),
            'god_code_resonance': round(bp_mean / self.GOD_CODE, 6)
        }

    def get_modes(self) -> List[str]:
        """Get available steering modes."""
        return list(self.MODES)

    def set_mode(self, mode: str) -> bool:
        """Set current steering mode."""
        if mode in self.MODES:
            self.current_mode = mode
            return True
        return False

    def get_status(self) -> dict:
        """Get steering engine status."""
        bp = self.base_parameters
        return {
            'mode': self.current_mode,
            'intensity': self.intensity,
            'temperature': self.temperature,
            'param_count': self.param_count,
            'mean': round(sum(bp) / max(len(bp), 1), 4),
            'history_count': len(self._steering_history),
        }


# Singleton instance
_nexus_steering = None

def get_steering() -> SteeringEngine:
    """Get singleton SteeringEngine instance."""
    global _nexus_steering
    if _nexus_steering is None:
        _nexus_steering = SteeringEngine()
    return _nexus_steering


__all__ = ['SteeringEngine', 'get_steering']
