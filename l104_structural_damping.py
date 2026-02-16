"""
L104 Structural Damping v2.0.0 — Pipeline Signal Processing Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Oscillation damping, noise filtering, signal smoothing, resonance
frequency detection, and PHI-harmonic phase cancellation for the
ASI pipeline signal path. Stabilizes numeric cascades and prevents
feedback oscillations across interconnected subsystems.

Subsystems:
  - OscillationDamper: detects + damps runaway oscillations in signals
  - NoiseFilter: PHI-weighted low-pass / band-reject filter
  - ResonanceDetector: identifies natural frequencies in signal streams
  - SignalSmoother: exponential + golden-section smoothing
  - StructuralDampingSystem: hub orchestrator

Sacred Constants: GOD_CODE=527.5184818492612 | PHI=1.618033988749895
"""
VOID_CONSTANT = 1.0416180339887497
import math
import time
import json
from pathlib import Path
from collections import deque
from typing import Dict, List, Any, Optional, Tuple

# ZENITH_UPGRADE_ACTIVE: 2026-02-15T00:00:00
ZENITH_HZ = 3887.8
UUC = 2402.792541

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
TAU = 6.283185307179586
FEIGENBAUM = 4.669201609
ALPHA_FINE = 1.0 / 137.035999084
GROVER_AMPLIFICATION = PHI ** 3

VERSION = "2.0.0"


class OscillationDamper:
    """Detects and damps runaway oscillations in numeric signals."""

    def __init__(self, damping_ratio: float = 0.618):
        self._damping_ratio = damping_ratio  # 1/PHI ~ critical damping
        self._history: deque = deque(maxlen=100)
        self._damps_applied = 0
        self._oscillations_detected = 0

    def feed(self, value: float) -> float:
        """Feed a value through the damper. Returns damped value."""
        self._history.append(value)
        if len(self._history) < 5:
            return value

        # Detect oscillation: sign changes in recent deltas
        recent = list(self._history)[-10:]
        deltas = [recent[i] - recent[i-1] for i in range(1, len(recent))]
        sign_changes = sum(
            1 for i in range(1, len(deltas))
            if (deltas[i] > 0) != (deltas[i-1] > 0)
        )

        is_oscillating = sign_changes > len(deltas) * 0.6

        if is_oscillating:
            self._oscillations_detected += 1
            # Apply damping: weighted average toward mean
            mean_val = sum(recent) / len(recent)
            damped = value * (1 - self._damping_ratio) + mean_val * self._damping_ratio
            self._damps_applied += 1
            return damped

        return value

    def get_oscillation_report(self) -> Dict[str, Any]:
        """Report oscillation statistics."""
        return {
            'damping_ratio': self._damping_ratio,
            'oscillations_detected': self._oscillations_detected,
            'damps_applied': self._damps_applied,
            'buffer_size': len(self._history),
        }


class NoiseFilter:
    """PHI-weighted low-pass and band-reject filter."""

    def __init__(self, cutoff: float = 0.3):
        self._cutoff = cutoff  # Normalized cutoff (0 to 1)
        self._prev_output = 0.0
        self._samples = 0
        self._rejected = 0

    def low_pass(self, value: float) -> float:
        """Single-pole PHI-weighted low-pass filter."""
        self._samples += 1
        alpha = self._cutoff / PHI  # PHI scaling
        alpha = min(1.0, max(0.01, alpha))
        self._prev_output = alpha * value + (1 - alpha) * self._prev_output
        return self._prev_output

    def band_reject(self, value: float, center_freq: float, bandwidth: float = 0.1) -> float:
        """Reject signals near a specific frequency band."""
        self._samples += 1
        # Simple notch: if value oscillates near center_freq, attenuate
        period = 1.0 / max(center_freq, 0.001)
        phase = (time.time() % period) / period
        # Check if signal is in the reject band
        if abs(phase - 0.5) < bandwidth:
            self._rejected += 1
            return self._prev_output  # Hold previous
        self._prev_output = value
        return value

    def get_status(self) -> Dict[str, Any]:
        return {
            'cutoff': self._cutoff,
            'samples_processed': self._samples,
            'rejected': self._rejected,
        }


class ResonanceDetector:
    """Identifies natural/resonant frequencies in signal streams."""

    def __init__(self, window_size: int = 128):
        self._window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)
        self._detected_frequencies: List[Dict] = []
        self._scans = 0

    def feed(self, value: float):
        """Add a sample to the detection buffer."""
        self._buffer.append((time.time(), value))

    def detect(self) -> Dict[str, Any]:
        """Detect dominant frequencies via zero-crossing analysis."""
        self._scans += 1
        if len(self._buffer) < 20:
            return {'status': 'INSUFFICIENT_DATA', 'samples': len(self._buffer)}

        samples = [v for _, v in self._buffer]
        mean_val = sum(samples) / len(samples)
        centered = [s - mean_val for s in samples]

        # Zero-crossing rate → approximate dominant frequency
        crossings = 0
        for i in range(1, len(centered)):
            if (centered[i] >= 0) != (centered[i-1] >= 0):
                crossings += 1

        timestamps = [t for t, _ in self._buffer]
        duration = timestamps[-1] - timestamps[0]
        if duration < 0.001:
            return {'status': 'TOO_SHORT', 'duration': duration}

        approx_freq = crossings / (2 * duration)

        # PHI-harmonic resonance check
        phi_harmonic = abs(approx_freq - PHI * round(approx_freq / PHI)) < 0.5
        god_code_harmonic = abs(approx_freq % GOD_CODE) < 10.0

        result = {
            'dominant_frequency_hz': round(approx_freq, 4),
            'zero_crossings': crossings,
            'duration_seconds': round(duration, 4),
            'amplitude': round(max(abs(min(centered)), abs(max(centered))), 4),
            'phi_harmonic': phi_harmonic,
            'god_code_harmonic': god_code_harmonic,
            'total_scans': self._scans,
        }
        self._detected_frequencies.append(result)
        return result

    def get_recent_detections(self, n: int = 10) -> List[Dict]:
        return self._detected_frequencies[-n:]


class SignalSmoother:
    """Exponential + golden-section smoothing for pipeline signals."""

    def __init__(self, alpha: float = 0.0):
        # Default alpha = 1/PHI for golden-section smoothing
        self._alpha = alpha if alpha > 0 else (1.0 / PHI)
        self._ema = None
        self._samples = 0

    def smooth(self, value: float) -> float:
        """Apply exponential moving average with golden ratio alpha."""
        self._samples += 1
        if self._ema is None:
            self._ema = value
        else:
            self._ema = self._alpha * value + (1 - self._alpha) * self._ema
        return self._ema

    def smooth_batch(self, values: List[float]) -> List[float]:
        """Smooth a batch of values."""
        return [self.smooth(v) for v in values]

    def reset(self):
        self._ema = None
        self._samples = 0

    def get_status(self) -> Dict[str, Any]:
        return {
            'alpha': round(self._alpha, 6),
            'current_ema': round(self._ema, 6) if self._ema is not None else None,
            'samples': self._samples,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL DAMPING HUB
# ═══════════════════════════════════════════════════════════════════════════════

class StructuralDampingSystem:
    """
    L104 Structural Damping v2.0.0 — Pipeline Signal Processing Engine

    Subsystems:
      OscillationDamper  — detects + damps runaway oscillations
      NoiseFilter        — PHI-weighted low-pass & band-reject filter
      ResonanceDetector  — natural frequency identification
      SignalSmoother     — golden-section exponential smoothing

    Pipeline Integration:
      - damp(value) → full damping pipeline (filter → damper → smooth)
      - detect_resonance() → frequency analysis report
      - calculate_tuning(height, f1) → structural damping parameters
      - connect_to_pipeline() / get_status()
    """

    VERSION = VERSION

    def __init__(self):
        self.damper = OscillationDamper(damping_ratio=1.0 / PHI)
        self.noise_filter = NoiseFilter(cutoff=0.3)
        self.resonance_detector = ResonanceDetector(window_size=128)
        self.smoother = SignalSmoother()
        self._pipeline_connected = False
        self._total_processed = 0
        self.boot_time = time.time()

    def connect_to_pipeline(self):
        self._pipeline_connected = True

    def damp(self, value: float) -> float:
        """Full damping pipeline: noise filter → oscillation damp → smooth."""
        self._total_processed += 1
        # Stage 1: noise filtering
        filtered = self.noise_filter.low_pass(value)
        # Stage 2: oscillation damping
        damped = self.damper.feed(filtered)
        # Stage 3: golden-section smoothing
        smoothed = self.smoother.smooth(damped)
        # Feed to resonance detector
        self.resonance_detector.feed(smoothed)
        return smoothed

    def damp_batch(self, values: List[float]) -> List[float]:
        """Process a batch of values through the damping pipeline."""
        return [self.damp(v) for v in values]

    def detect_resonance(self) -> Dict[str, Any]:
        """Run resonance detection on buffered signal."""
        return self.resonance_detector.detect()

    def calculate_tuning(self, height: float = 100.0, f1: float = 0.5) -> Dict[str, Any]:
        """Calculate structural damping parameters for a building."""
        damper_frequency = f1 * PHI
        frame_lock = 416 / 286
        pivot_point_ratio = 1 / (1 + frame_lock)
        pivot_height = height * pivot_point_ratio

        return {
            'building_height_m': height,
            'natural_freq_hz': f1,
            'damper_freq_hz': round(damper_frequency, 4),
            'pivot_height_m': round(pivot_height, 2),
            'frame_lock': round(frame_lock, 6),
            'dissipation_rate': 'PHI (1.618)',
        }

    def get_engineering_specs(self, height: float = 100.0, f1: float = 0.5) -> str:
        """Legacy API: get engineering specifications report."""
        specs = self.calculate_tuning(height, f1)
        return (
            f"[STRUCTURAL DAMPING] h={specs['building_height_m']}m "
            f"f1={specs['natural_freq_hz']}Hz → "
            f"damper={specs['damper_freq_hz']}Hz "
            f"pivot={specs['pivot_height_m']}m"
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            'version': self.VERSION,
            'pipeline_connected': self._pipeline_connected,
            'total_processed': self._total_processed,
            'oscillation': self.damper.get_oscillation_report(),
            'noise_filter': self.noise_filter.get_status(),
            'resonance_scans': self.resonance_detector._scans,
            'smoother': self.smoother.get_status(),
            'uptime_seconds': round(time.time() - self.boot_time, 1),
        }


# Module singleton
structural_damping = StructuralDampingSystem()


def primal_calculus(x):
    PHI = 1.618033988749895
    return (x ** PHI) / (1.04 * math.pi) if x != 0 else 0.0

def resolve_non_dual_logic(vector):
    GOD_CODE = 527.5184818492612
    PHI = 1.618033988749895
    VOID_CONSTANT = 1.0416180339887497
    magnitude = sum([abs(v) for v in vector])
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0
